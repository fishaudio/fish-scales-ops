import math
import torch
import torch.nn.functional as F
import fish_scales_ops as fso
from fish_scales_ops.audio.fa4_mxfp8 import attn_fa4_mxfp8
from probe_baseline import make_tower_weights, run_fso030_fused_qkv, cudnn_attn, capture, time_graphs
from bench_phase1 import run_phase1_tower


def verify_fa4_mxfp8():
    print("=== Verifying FA4 SM100 Fused Attention -> MXFP8 1x32 ===")
    for M in [78, 117, 182, 364, 533, 1079]:
        torch.manual_seed(2000 + M)
        qkv = torch.randn(M, 3840, device="cuda", dtype=torch.bfloat16) * 0.5
        q, k, v = qkv.view(M, 3, 20, 64).unbind(1)
        q4, k4, v4 = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        xq, sx = attn_fa4_mxfp8(q4, k4, v4)

        # Reference: fp32 SDPA -> quantize_activation
        qf = q4.transpose(1, 2).float()
        kf = k4.transpose(1, 2).float()
        vf = v4.transpose(1, 2).float()
        a_fp32 = F.scaled_dot_product_attention(qf, kf, vf).transpose(1, 2).reshape(M, 1280)
        a_bf16 = F.scaled_dot_product_attention(q4.transpose(1, 2), k4.transpose(1, 2), v4.transpose(1, 2)).transpose(1, 2).reshape(M, 1280)
        q_ref, s_ref = fso.dense.quantize_activation(a_bf16)

        w_o = (torch.randn(1280, 1280, device="cuda", dtype=torch.bfloat16) / math.sqrt(1280)).contiguous()
        dw_o = fso.dense.prepare_weight(w_o, format="mxfp8")
        y_fp32 = a_fp32 @ w_o.float().t()
        y_ref = fso.dense.linear(q_ref, dw_o, x_scale=s_ref)
        y_fa4 = fso.dense.linear(xq, dw_o, x_scale=sx)

        cos_ref = F.cosine_similarity(y_ref.float().flatten(), y_fp32.flatten(), dim=0).item()
        cos_fa4 = F.cosine_similarity(y_fa4.float().flatten(), y_fp32.flatten(), dim=0).item()
        rel_ref = (y_ref.float() - y_fp32).abs().max().item() / y_fp32.abs().max().item()
        rel_fa4 = (y_fa4.float() - y_fp32).abs().max().item() / y_fp32.abs().max().item()
        # Check padded rows in sx are strictly zero
        M_pad = (M + 127) // 128 * 128
        pad_nz = 0
        for r in range(M, M_pad):
            for kp in range(10):
                idx = ((r // 128) * 10 + kp) * 128 + (r % 32) * 4 + ((r % 128) // 32)
                if sx[idx].item() != 0:
                    pad_nz += 1
        print(f"  M={M:4d} | cos(ref/fa4_mx)={cos_ref:.6f}/{cos_fa4:.6f} | max_rel(ref/fa4_mx)={rel_ref:.5f}/{rel_fa4:.5f} | pad_nz={pad_nz}")
        assert pad_nz == 0, f"Non-zero padded scale words: {pad_nz}"
        assert cos_fa4 > 0.9985, f"Low cosine: {cos_fa4}"
    print("  ALL FA4_MXFP8 ACCURACY & PADDED-ZERO CONTRACT CHECKS PASSED!")


def run_phase2_tower(x_conv, W):
    x = fso.dense.linear(x_conv, W["conv"][1])
    M, D = x.shape
    for L in W["layers"]:
        hq, hs = torch.ops.fish_scales_ops.layernorm_mxfp8_sm100(x, L["ln1"][0], L["ln1"][1], 1e-5, False, 1)
        qkv = fso.dense.linear(hq, L["dqkv"], x_scale=hs, bias=L["qkv"][1])
        q, k, v = qkv.view(M, 3, 20, 64).unbind(1)
        aq, as_ = attn_fa4_mxfp8(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
        x = fso.dense.linear(aq, L["do"], x_scale=as_, bias=L["o"][1], residual=x)
        h2q, h2s = torch.ops.fish_scales_ops.layernorm_mxfp8_sm100(x, L["ln2"][0], L["ln2"][1], 1e-5, False, 1)
        uq, us = fso.dense.linear_quantized(h2q, L["dfc1"], x_scale=h2s, bias=L["fc1"][1], activation="gelu_erf")
        x = fso.dense.linear(uq, L["dfc2"], x_scale=us, bias=L["fc2"][1], residual=x)
    xq, xs = torch.ops.fish_scales_ops.layernorm_mxfp8_sm100(x, W["ln_post"][0], W["ln_post"][1], 1e-5, False, 1)
    xq2, xs2 = fso.dense.linear_quantized(xq, W["p1"][2], x_scale=xs, bias=W["p1"][1], activation="gelu_erf")
    return fso.dense.linear(xq2, W["p2"][2], x_scale=xs2, bias=W["p2"][1])


if __name__ == "__main__":
    verify_fa4_mxfp8()
    W = make_tower_weights()
    print("\n=== Phase 2: Full 32-Layer Audio Tower (LN_MXFP8 + FA4_MXFP8 + PDL) ===")
    print(f"{'M':>5} | {'0.3.0 fqkv':>10} | {'Phase 1':>9} | {'Phase 2':>9} | {'P2 vs P1':>9} | {'P2 vs 0.3.0':>11}")
    print("-" * 68)
    for M in [78, 117, 182, 364, 533, 1079]:
        x = torch.randn(M, 7680, device="cuda", dtype=torch.bfloat16) * 0.5
        g0, _ = capture(lambda: run_fso030_fused_qkv(x, W, cudnn_attn))
        g1, _ = capture(lambda: run_phase1_tower(x, W, cudnn_attn, 1))
        g2, _ = capture(lambda: run_phase2_tower(x, W))
        t = time_graphs({"base": g0, "p1": g1, "p2": g2})
        print(f"{M:>5} | {t['base']:>10.1f} | {t['p1']:>9.1f} | {t['p2']:>9.1f} | {t['p2']-t['p1']:>+8.1f} | {(t['p2']/t['base']-1)*100:>+10.1f}%")
