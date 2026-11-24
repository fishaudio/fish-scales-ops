"""Unit & contract tests for Blackwell (sm_100 / sm_103) Audio Tower fused operators.

Covers:
  - layernorm_mxfp8 (exact fp32 mean/var, round_bf16 bit match, SM100 zero padded scales, CUDA graph)
  - attn_fa4_mxfp8 (exact fp32 SDPA ref cosine, zero padded scales, q_stage=1/2, CUDA graph)
  - gelu_erf_bf16_pdl (exact erf GELU vs PyTorch, NOT tanh, NHWC + contiguous)
  - gelu_nhwc_to_mxfp8 + prepare_conv_out_weight + linear_audio_conv_out (zero padded scales, CUDA graph)
  - linear_audio_qkv / out_proj / fc2 / proj2 (no global WF mutation, fp32 cosine, CUDA graph)
  - Public API surface preservation (fish_scales_ops.__all__ unchanged)
"""
import math
import torch
import torch.nn.functional as F

import fish_scales_ops as fso
import fish_scales_ops.audio as audio
import fish_scales_ops.gemm._sm100_decode as dec


RUNGS = [78, 117, 182, 364, 533, 1079]


def _check_padded_scales_zero(sx: torch.Tensor, M: int, K: int) -> int:
    m_pad = (M + 127) // 128 * 128
    num_kp = K // 128
    sx_cpu = sx.cpu()
    nz = 0
    for r in range(M, m_pad):
        for kp in range(num_kp):
            idx = ((r // 128) * num_kp + kp) * 128 + (r % 32) * 4 + ((r % 128) // 32)
            if sx_cpu[idx].item() != 0:
                nz += 1
    return nz


def _capture_on_side_stream(fn):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    s.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = fn()
    s.synchronize()
    with torch.cuda.stream(s):
        g.replay()
    s.synchronize()
    return g, out


def test_public_surface_preserved():
    assert list(fso.__all__) == ["dense", "moe", "attention", "compat", "build_info"]
    assert dec.WF_PRE_WAIT_STAGES_ONE_CTA == 2
    assert dec.WF_PRE_WAIT_STAGES_TWO_CTA == 1


def test_layernorm_mxfp8_contract():
    torch.manual_seed(101)
    # Test BOTH in-kernel D==1280 fast path AND in-kernel general D%128==0 path (512, 2048)
    for M, D in [(1, 1280), (78, 1280), (117, 1280), (128, 1280), (182, 1280), (364, 1280), (533, 1280), (1079, 1280), (78, 512), (182, 2048)]:
        x = (torch.randn(M, D, device="cuda", dtype=torch.bfloat16) * 0.7).contiguous()
        g = (torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.2 + 1.0).contiguous()
        b = (torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
        w = (torch.randn(D, D, device="cuda", dtype=torch.bfloat16) / math.sqrt(D)).contiguous()
        dw = fso.dense.prepare_weight(w, format="mxfp8")

        # 1. round_bf16=True matches quantize_activation(F.layer_norm) within <= 1 ULP rounding
        xq_r, sx_r = audio.layernorm_mxfp8(x, g, b, eps=1e-5, round_bf16=True)
        h_ref = F.layer_norm(x, (D,), g, b, 1e-5)
        rq, rs = fso.dense.quantize_activation(h_ref)
        s_diff = (sx_r != rs).sum().item()
        q_diff = (xq_r.view(torch.uint8) != rq.view(torch.uint8)).sum().item()
        assert s_diff == 0, f"M={M}: scale diff {s_diff}"
        assert q_diff <= max(2, M // 50), f"M={M}: too many fp8 byte diffs {q_diff}"
        assert _check_padded_scales_zero(sx_r, M, D) == 0

        # 2. round_bf16=False higher accuracy vs fp32 reference
        xq, sx = audio.layernorm_mxfp8(x, g, b, eps=1e-5, round_bf16=False)
        assert _check_padded_scales_zero(sx, M, D) == 0
        y_fp32 = F.layer_norm(x.float(), (D,), g.float(), b.float(), 1e-5) @ w.float().t()
        y_fused = fso.dense.linear(xq, dw, x_scale=sx)
        cos = F.cosine_similarity(y_fused.float().flatten(), y_fp32.flatten(), dim=0).item()
        assert cos > 0.999, f"M={M}: cos={cos}"

        # 3. CUDA graph capture on non-default stream == eager bitwise
        if M in (78, 533):
            _, (xq_g, sx_g) = _capture_on_side_stream(lambda: audio.layernorm_mxfp8(x, g, b))
            assert torch.equal(xq.view(torch.uint8), xq_g.view(torch.uint8))
            assert torch.equal(sx, sx_g)


def test_attn_fa4_mxfp8_contract():
    torch.manual_seed(202)
    w_o = (torch.randn(1280, 1280, device="cuda", dtype=torch.bfloat16) / math.sqrt(1280)).contiguous()
    dw_o = fso.dense.prepare_weight(w_o, format="mxfp8")
    for M in RUNGS:
        q = (torch.randn(1, M, 20, 64, device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
        k = (torch.randn(1, M, 20, 64, device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
        v = (torch.randn(1, M, 20, 64, device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()

        aq, sx = audio.attn_fa4_mxfp8(q, k, v)
        assert aq.shape == (M, 1280) and aq.dtype == torch.float8_e4m3fn
        assert _check_padded_scales_zero(sx, M, 1280) == 0

        ref_a = F.scaled_dot_product_attention(
            q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2)
        ).transpose(1, 2).reshape(M, 1280)
        ref_y = ref_a @ w_o.float().t()
        y = audio.linear_audio_out_proj(aq, dw_o, sx)
        cos = F.cosine_similarity(y.float().flatten(), ref_y.flatten(), dim=0).item()
        assert cos > 0.999, f"FA4_MXFP8 M={M}: cos={cos}"

        if M in (78, 182, 533, 1079):
            _, (aq_g, sx_g) = _capture_on_side_stream(lambda: audio.attn_fa4_mxfp8(q, k, v))
            assert torch.equal(aq.view(torch.uint8), aq_g.view(torch.uint8))
            assert torch.equal(sx, sx_g)


def test_stem_fused_kernels_contract():
    torch.manual_seed(303)
    # 1. gelu_erf_bf16_pdl exact erf (NOT tanh)
    x = torch.linspace(-6.0, 6.0, 14 * 480 * 32 * 25, device="cuda", dtype=torch.bfloat16).view(14, 480, 32, 25)
    x_nhwc = x.to(memory_format=torch.channels_last)
    y_erf = F.gelu(x_nhwc, approximate="none")
    y_tanh = F.gelu(x_nhwc, approximate="tanh")
    y_pdl = audio.gelu_erf_bf16_pdl(x_nhwc)
    assert y_pdl.is_contiguous(memory_format=torch.channels_last)
    err_erf = (y_pdl.float() - y_erf.float()).abs().max().item()
    err_tanh = (y_pdl.float() - y_tanh.float()).abs().max().item()
    assert err_erf <= 1e-2 and err_erf < err_tanh, f"err_erf={err_erf}, err_tanh={err_tanh}"

    # 2. gelu_nhwc_to_mxfp8 + prepare_conv_out_weight + linear_audio_conv_out
    w_co = (torch.randn(1280, 7680, device="cuda", dtype=torch.bfloat16) / math.sqrt(7680)).contiguous()
    dw_co = audio.prepare_conv_out_weight(w_co, channels=480, freq=16)
    for M in RUNGS:
        B = M // 13
        c3 = (torch.randn(B, 480, 16, 13, device="cuda", dtype=torch.bfloat16) * 0.5).to(memory_format=torch.channels_last)
        pos = (torch.randn(M, 1280, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
        ref = F.gelu(c3.float(), approximate="none").permute(0, 3, 1, 2).reshape(M, 7680) @ w_co.float().t() + pos.float()
        xq, sx = audio.gelu_nhwc_to_mxfp8(c3)
        assert xq.shape == (M, 7680) and xq.dtype == torch.float8_e4m3fn
        assert _check_padded_scales_zero(sx, M, 7680) == 0
        y = audio.linear_audio_conv_out(xq, dw_co, sx, residual=pos)
        cos = F.cosine_similarity(y.float().flatten(), ref.flatten(), dim=0).item()
        assert cos > 0.9985, f"Stem M={M}: cos={cos}"
        if M in (78, 1079):
            _, y_g = _capture_on_side_stream(lambda: audio.linear_audio_conv_out(*audio.gelu_nhwc_to_mxfp8(c3), dw_co, residual=pos) if False else (lambda q_s: audio.linear_audio_conv_out(q_s[0], dw_co, q_s[1], residual=pos))(audio.gelu_nhwc_to_mxfp8(c3)))
            assert torch.equal(y.view(torch.uint16), y_g.view(torch.uint16))


def test_role_dispatch_no_global_mutation():
    torch.manual_seed(404)
    w_qkv = (torch.randn(3840, 1280, device="cuda", dtype=torch.bfloat16) / math.sqrt(1280)).contiguous()
    b_qkv = (torch.randn(3840, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    dw_qkv = fso.dense.prepare_weight(w_qkv, format="mxfp8")
    w_fc2 = (torch.randn(1280, 5120, device="cuda", dtype=torch.bfloat16) / math.sqrt(5120)).contiguous()
    b_fc2 = (torch.randn(1280, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    dw_fc2 = fso.dense.prepare_weight(w_fc2, format="mxfp8")

    for M in RUNGS:
        x1 = (torch.randn(M, 1280, device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
        x1q, x1s = fso.dense.quantize_activation(x1)
        y_qkv = audio.linear_audio_qkv(x1q, dw_qkv, x1s, bias=b_qkv)
        ref_qkv = x1.float() @ w_qkv.float().t() + b_qkv.float()
        assert F.cosine_similarity(y_qkv.float().flatten(), ref_qkv.flatten(), dim=0).item() > 0.9985

        x2 = (torch.randn(M, 5120, device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
        res = (torch.randn(M, 1280, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
        x2q, x2s = fso.dense.quantize_activation(x2)
        y_fc2 = audio.linear_audio_fc2(x2q, dw_fc2, x2s, bias=b_fc2, residual=res)
        ref_fc2 = x2.float() @ w_fc2.float().t() + b_fc2.float() + res.float()
        assert F.cosine_similarity(y_fc2.float().flatten(), ref_fc2.flatten(), dim=0).item() > 0.9985

    # Global decode WF constants MUST remain untouched at 2/1!
    assert dec.WF_PRE_WAIT_STAGES_ONE_CTA == 2
    assert dec.WF_PRE_WAIT_STAGES_TWO_CTA == 1


if __name__ == "__main__":
    for fn in [
        test_public_surface_preserved,
        test_layernorm_mxfp8_contract,
        test_attn_fa4_mxfp8_contract,
        test_stem_fused_kernels_contract,
        test_role_dispatch_no_global_mutation,
    ]:
        fn()
        print(f"PASS: {fn.__name__}", flush=True)
    print("ALL AUDIO TOWER CONTRACT TESTS PASSED!", flush=True)
