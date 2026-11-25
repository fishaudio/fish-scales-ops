#!/usr/bin/env python3
"""Unit tests for s3-drama (Family C) kernel-internal unified SM100 kernels."""
import importlib.util
import sys

import torch
import torch.nn.functional as F
import fish_scales_ops as fso
from fish_scales_ops import drama3_unified as d3u


def _sm100_skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA not available"
    major = torch.cuda.get_device_capability(0)[0]
    if major != 10:
        return f"drama3 unified kernels are sm_100/sm_103-only (found sm_{major}x)"
    return None


def _dense_dsl_skip_reason() -> str | None:
    reason = _sm100_skip_reason()
    if reason is not None:
        return reason
    if importlib.util.find_spec("cutlass") is None:
        return "optional nvidia-cutlass-dsl is not installed"
    return None


def _require_or_skip(reason: str | None) -> bool:
    if reason is None:
        return True
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(reason)
    print(f"SKIP: {reason}")
    return False


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af, bf = a.float().reshape(-1), b.float().reshape(-1)
    if af.norm().item() == 0.0 and bf.norm().item() == 0.0:
        return 1.0
    return float(torch.dot(af, bf) / (af.norm() * bf.norm() + 1e-12))


def test_unified_dense_all_m_and_epilogues():
    if not _require_or_skip(_dense_dsl_skip_reason()):
        return
    torch.manual_seed(123)
    shapes = [(1024, 2048), (2048, 512), (2048, 4096), (9216, 2048), (12288, 2048)]
    ms = [1, 2, 3, 7, 8, 9, 16, 31, 32, 33, 64, 65, 96, 127, 128, 129, 255, 256, 512, 1024, 2048]
    for N, K in shapes:
        w = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
        wf, sw = d3u.prepare_dense_weight_unified(w)
        # Immediate back-to-back pre-quantized linear after prepare_dense_weight_unified
        # verifies the non-PDL completion barrier prevents weights-first PDL races.
        x0 = (torch.randn(16, K, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
        xf0, sx0 = fso.compat.quantize_1x32_fp8(x0)
        wf_imm, sw_imm = d3u.prepare_dense_weight_unified(w)
        y_imm = d3u.linear_unified(xf0, wf_imm, sw_imm, x_scales=sx0)
        torch.cuda.synchronize()
        y_sync = d3u.linear_unified(xf0, wf_imm, sw_imm, x_scales=sx0)
        assert torch.equal(y_imm, y_sync), f"prepare->linear PDL race {(N, K)}"

        bias = (torch.randn(N, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
        zero_bias = torch.zeros(N, device="cuda", dtype=torch.bfloat16)
        for M in ms:
            x = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
            xf, sx = fso.compat.quantize_1x32_fp8(x)
            y1 = d3u.linear_unified(xf, wf, sw, x_scales=sx)
            y2 = d3u.linear_unified(xf, wf, sw, x_scales=sx)
            assert torch.equal(y1, y2), f"nondet dense {(N, K)} M={M}"
            ref = x.float() @ w.float().t()
            assert cos_sim(y1, ref) > 0.997, f"cos fail dense {(N, K)} M={M}"
            # Zero bias == no bias bitwise contract!
            yz = d3u.linear_unified(xf, wf, sw, x_scales=sx, bias=zero_bias)
            assert torch.equal(y1, yz), f"zero-bias contract fail {(N, K)} M={M}"
            if M in (1, 16, 65, 129, 512):
                res = (torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
                ye = d3u.linear_unified(xf, wf, sw, x_scales=sx, bias=bias, residual=res, act=1)
                ref_e = F.gelu(ref + bias.float(), approximate="none") + res.float()
                assert cos_sim(ye, ref_e) > 0.996, f"epi cos fail {(N, K)} M={M}"
    print("[PASS] test_unified_dense_all_m_and_epilogues")


def test_unified_dense_cudagraph_and_multistream():
    if not _require_or_skip(_dense_dsl_skip_reason()):
        return
    torch.manual_seed(7)
    w = (torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    wf, sw = d3u.prepare_dense_weight_unified(w)
    for M in (1, 16, 64, 128, 256, 1024):
        x = (torch.randn(M, 4096, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
        xf, sx = fso.compat.quantize_1x32_fp8(x)
        ye = d3u.linear_unified(xf, wf, sw, x_scales=sx)
        s = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()
        out = [None]
        with torch.cuda.stream(s):
            d3u.linear_unified(xf, wf, sw, x_scales=sx)
            with torch.cuda.graph(g, stream=s):
                out[0] = d3u.linear_unified(xf, wf, sw, x_scales=sx)
        torch.cuda.synchronize()
        for _ in range(5):
            g.replay()
            torch.cuda.synchronize()
            assert torch.equal(ye, out[0]), f"dense replay != eager M={M}"
    # Multi-stream concurrency
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    x1 = (torch.randn(16, 4096, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
    x2 = (torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
    r1 = d3u.linear_unified(x1, wf, sw)
    r2 = d3u.linear_unified(x2, wf, sw)
    torch.cuda.synchronize()
    with torch.cuda.stream(s1):
        o1 = d3u.linear_unified(x1, wf, sw)
    with torch.cuda.stream(s2):
        o2 = d3u.linear_unified(x2, wf, sw)
    torch.cuda.synchronize()
    assert torch.equal(o1, r1) and torch.equal(o2, r2)
    print("[PASS] test_unified_dense_cudagraph_and_multistream")


def test_unified_moe_all_regimes():
    if not _require_or_skip(_sm100_skip_reason()):
        return
    torch.manual_seed(99)
    E, TOPK, H, I = 256, 8, 2048, 512
    w13 = (torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    w2 = (torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    sw13 = (torch.randn(2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    sw2 = (torch.randn(H, I, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()

    exp_r = d3u.prepare_moe_unified(w13, w2)
    exp_s = d3u.prepare_moe_unified(w13, w2, shared_w13=sw13, shared_w2=sw2)
    fso_exp = fso.moe.prepare_experts(w13, w2, format="mxfp8")

    # Test uniform M=1..4096, padded, skew, all-padded edge case (0 live routed rows!)
    for M in (1, 8, 64, 128, 256, 1024, 2048, 4096):
        h = (torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
        ids = torch.rand(M, E, device="cuda").topk(TOPK, dim=-1).indices.to(torch.int32).contiguous()
        tw = torch.softmax(torch.randn(M, TOPK, device="cuda"), dim=-1).to(torch.float32).contiguous()
        yr = d3u.moe_layer_unified(h, exp_r, ids, tw)
        yf = fso.moe.layer(h, fso_exp, ids, tw)
        assert cos_sim(yr, yf) > 0.9999, f"routed mismatch vs fso M={M}"
        # Bitwise determinism under CUDA graph replay
        sg = torch.sigmoid(torch.randn(M, device="cuda", dtype=torch.float32))
        ys_eager = d3u.moe_layer_unified(h, exp_s, ids, tw, shared_gate=sg)
        s = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()
        ys_g = [None]
        with torch.cuda.stream(s):
            d3u.moe_layer_unified(h, exp_s, ids, tw, shared_gate=sg)
            with torch.cuda.graph(g, stream=s):
                ys_g[0] = d3u.moe_layer_unified(h, exp_s, ids, tw, shared_gate=sg)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(ys_eager, ys_g[0]), f"moe replay != eager M={M}"

    # All-padded edge case (0 live routed pairs -> routed output must be exact 0!)
    h16 = (torch.randn(16, H, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
    ids_dead = torch.full((16, TOPK), -1, device="cuda", dtype=torch.int32)
    tw_dead = torch.zeros((16, TOPK), device="cuda", dtype=torch.float32)
    y_dead = d3u.moe_layer_unified(h16, exp_r, ids_dead, tw_dead)
    assert y_dead.abs().max().item() == 0.0, "all-padded routed output not zero"

    # Multi-stream concurrency
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    ids16 = torch.rand(16, E, device="cuda").topk(TOPK, dim=-1).indices.to(torch.int32).contiguous()
    tw16 = torch.softmax(torch.randn(16, TOPK, device="cuda"), dim=-1).to(torch.float32).contiguous()
    ref16 = d3u.moe_layer_unified(h16, exp_s, ids16, tw16)
    torch.cuda.synchronize()
    with torch.cuda.stream(s1):
        a1 = d3u.moe_layer_unified(h16, exp_s, ids16, tw16)
    with torch.cuda.stream(s2):
        a2 = d3u.moe_layer_unified(h16, exp_s, ids16, tw16)
    torch.cuda.synchronize()
    assert torch.equal(a1, ref16) and torch.equal(a2, ref16)
    print("[PASS] test_unified_moe_all_regimes")


if __name__ == "__main__":
    reason = _sm100_skip_reason()
    if reason is not None:
        print(f"SKIP: {reason}")
        sys.exit(0)
    test_unified_dense_all_m_and_epilogues()
    test_unified_dense_cudagraph_and_multistream()
    test_unified_moe_all_regimes()
    print("ALL DRAMA3 UNIFIED TESTS PASSED!")
