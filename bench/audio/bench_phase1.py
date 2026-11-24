import math
import sys
import time
import torch
import torch.nn.functional as F
import fish_scales_ops as fso
from probe_baseline import make_tower_weights, run_bf16, run_fso030_fused_qkv, cudnn_attn, capture, time_graphs


def verify_layernorm_mxfp8():
    print("=== Verifying layernorm_mxfp8_sm100 accuracy & contract ===")
    for M in [78, 117, 182, 364, 533, 1079]:
        torch.manual_seed(1000 + M)
        x = torch.randn(M, 1280, device="cuda", dtype=torch.bfloat16) * 0.7 + 0.15
        g = torch.ones(1280, device="cuda", dtype=torch.bfloat16) + 0.1 * torch.randn(1280, device="cuda", dtype=torch.bfloat16)
        b = 0.1 * torch.randn(1280, device="cuda", dtype=torch.bfloat16)
        # Reference: F.layer_norm -> quantize_activation
        ln_ref = F.layer_norm(x, (1280,), g, b, 1e-5)
        q_ref, s_ref = fso.dense.quantize_activation(ln_ref)
        # Also fp32 reference
        xf = x.float()
        mean = xf.mean(dim=-1, keepdim=True)
        var = ((xf - mean) ** 2).mean(dim=-1, keepdim=True)
        ln_fp32 = (xf - mean) * torch.rsqrt(var + 1e-5) * g.float() + b.float()
        w_test = (torch.randn(1280, 1280, device="cuda", dtype=torch.bfloat16) / math.sqrt(1280)).contiguous()
        dw = fso.dense.prepare_weight(w_test, format="mxfp8")
        y_fp32 = ln_fp32 @ w_test.float().t()
        y_ref = fso.dense.linear(q_ref, dw, x_scale=s_ref)

        for variant in [1, 2, 3]:
            q_bf, s_bf = torch.ops.fish_scales_ops.layernorm_mxfp8_sm100(x, g, b, 1e-5, True, variant)
            q_fp, s_fp = torch.ops.fish_scales_ops.layernorm_mxfp8_sm100(x, g, b, 1e-5, False, variant)
            # Check padded rows in s_bf and s_fp are zero
            M_pad = (M + 127) // 128 * 128
            assert s_bf.shape == s_ref.shape and s_fp.shape == s_ref.shape
            # Check exact match or near-exact with round_bf16=True (tiny 1-ulp rsqrtf diff possible vs torch LN)
            q_diff = (q_bf.view(torch.uint8) != q_ref.view(torch.uint8)).sum().item()
            s_diff = (s_bf != s_ref).sum().item()
            y_bf = fso.dense.linear(q_bf, dw, x_scale=s_bf)
            y_fp = fso.dense.linear(q_fp, dw, x_scale=s_fp)
            cos_ref = F.cosine_similarity(y_ref.float().flatten(), y_fp32.flatten(), dim=0).item()
            cos_bf = F.cosine_similarity(y_bf.float().flatten(), y_fp32.flatten(), dim=0).item()
            cos_fp = F.cosine_similarity(y_fp.float().flatten(), y_fp32.flatten(), dim=0).item()
            rel_ref = (y_ref.float() - y_fp32).abs().max().item() / y_fp32.abs().max().item()
            rel_fp = (y_fp.float() - y_fp32).abs().max().item() / y_fp32.abs().max().item()
            if variant == 1:
                print(f"  M={M:4d} v={variant} | q_diff={q_diff:5d}/{M*1280} s_diff={s_diff:3d} | "
                      f"cos(ref/bf/fp)={cos_ref:.6f}/{cos_bf:.6f}/{cos_fp:.6f} | max_rel(ref/fp)={rel_ref:.5f}/{rel_fp:.5f}")
                q1, s1 = q_fp, s_fp
            else:
                v_qdiff = (q_fp.view(torch.uint8) != q1.view(torch.uint8)).sum().item()
                v_sdiff = (s_fp != s1).sum().item()
                assert v_qdiff <= 16 and v_sdiff == 0, f"variant {variant} M={M} diff too big: q={v_qdiff} s={v_sdiff}"
                assert cos_fp >= cos_ref - 1e-5, f"variant {variant} cos drop: {cos_fp} vs {cos_ref}"
    print("  ALL VARIANTS PASS ACCURACY & CONTRACT CHECKS!")


def run_phase1_tower(x_conv, W, attn_fn, ln_variant=0):
    x = fso.dense.linear(x_conv, W["conv"][1])
    M, D = x.shape
    for L in W["layers"]:
        hq, hs = torch.ops.fish_scales_ops.layernorm_mxfp8_sm100(x, L["ln1"][0], L["ln1"][1], 1e-5, False, ln_variant)
        qkv = fso.dense.linear(hq, L["dqkv"], x_scale=hs, bias=L["qkv"][1])
        q, k, v = qkv.view(M, 3, 20, 64).unbind(1)
        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)
        a = attn_fn(q, k, v).transpose(1, 2).reshape(M, D)
        x = fso.dense.linear(a, L["do"], bias=L["o"][1], residual=x)
        h2q, h2s = torch.ops.fish_scales_ops.layernorm_mxfp8_sm100(x, L["ln2"][0], L["ln2"][1], 1e-5, False, ln_variant)
        uq, us = fso.dense.linear_quantized(h2q, L["dfc1"], x_scale=h2s, bias=L["fc1"][1], activation="gelu_erf")
        x = fso.dense.linear(uq, L["dfc2"], x_scale=us, bias=L["fc2"][1], residual=x)
    xq, xs = torch.ops.fish_scales_ops.layernorm_mxfp8_sm100(x, W["ln_post"][0], W["ln_post"][1], 1e-5, False, ln_variant)
    xq2, xs2 = fso.dense.linear_quantized(xq, W["p1"][2], x_scale=xs, bias=W["p1"][1], activation="gelu_erf")
    return fso.dense.linear(xq2, W["p2"][2], x_scale=xs2, bias=W["p2"][1])


if __name__ == "__main__":
    verify_layernorm_mxfp8()
    W = make_tower_weights()
    print("\n=== Phase 1: Full 32-Layer Audio Tower (cuDNN attn) — LN_MXFP8 Variant Sweep ===")
    print(f"{'M':>5} | {'0.3.0 fqkv':>10} | {'v1 cta160':>10} | {'v2 warp1':>10} | {'v3 warp4':>10} | {'best save':>10}")
    print("-" * 68)
    for M in [78, 117, 182, 364, 533, 1079]:
        x = torch.randn(M, 7680, device="cuda", dtype=torch.bfloat16) * 0.5
        g0, _ = capture(lambda: run_fso030_fused_qkv(x, W, cudnn_attn))
        g1, _ = capture(lambda: run_phase1_tower(x, W, cudnn_attn, 1))
        g2, _ = capture(lambda: run_phase1_tower(x, W, cudnn_attn, 2))
        g3, _ = capture(lambda: run_phase1_tower(x, W, cudnn_attn, 3))
        t = time_graphs({"base": g0, "v1": g1, "v2": g2, "v3": g3})
        best = min(t["v1"], t["v2"], t["v3"])
        save = t["base"] - best
        print(f"{M:>5} | {t['base']:>10.1f} | {t['v1']:>10.1f} | {t['v2']:>10.1f} | {t['v3']:>10.1f} | {-save:>+8.1f}us ({(best/t['base']-1)*100:+.1f}%)")
