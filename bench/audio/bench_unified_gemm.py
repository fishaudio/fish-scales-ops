"""2032 MHz ABBA benchmark of the Zero-Host-Routing Unified Audio Tower
(single unified layernorm_mxfp8.cu + single unified Cluster-DSMEM GEMM with 100%
in-kernel S in {1,2,4}, dynamic WF, and adaptive L2/GPC rasterization) vs BF16 & 0.3.0 fqkv!
"""
import math
import sys
sys.path.insert(0, "/home/aniya/fso-audio-extreme-opt/bench/audio")

import torch
import torch.nn.functional as F
import fish_scales_ops as fso
import fish_scales_ops.audio as audio
from fish_scales_ops.audio.unified_gemm_sm100 import linear_mxfp8_unified
from probe_baseline import make_tower_weights, run_bf16, run_fso030_fused_qkv, cudnn_attn, capture
from bench_p2_ablation import time_graphs_clocked


def run_unified_tower(x_conv, W, tile=(128, 64), early_pdl=True):
    xc_q, xc_s = fso.dense.quantize_activation(x_conv)
    x = linear_mxfp8_unified(xc_q, W["conv"][1].weight, xc_s, W["conv"][1].scale, early_pdl=early_pdl, mma_tiler_mn=tile)
    M, _ = x.shape
    for L in W["layers"]:
        hq, hs = audio.layernorm_mxfp8(x, L["ln1"][0], L["ln1"][1])
        qkv = linear_mxfp8_unified(hq, L["dqkv"].weight, hs, L["dqkv"].scale, bias=L["qkv"][1], early_pdl=early_pdl, mma_tiler_mn=tile)
        q, k, v = qkv.view(M, 3, 20, 64).unbind(1)
        aq, as_ = audio.attn_fa4_mxfp8(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
        x = linear_mxfp8_unified(aq, L["do"].weight, as_, L["do"].scale, bias=L["o"][1], residual=x, early_pdl=early_pdl, mma_tiler_mn=tile)
        h2q, h2s = audio.layernorm_mxfp8(x, L["ln2"][0], L["ln2"][1])
        uq, us = fso.dense.linear_quantized(h2q, L["dfc1"], x_scale=h2s, bias=L["fc1"][1], activation="gelu_erf")
        x = linear_mxfp8_unified(uq, L["dfc2"].weight, us, L["dfc2"].scale, bias=L["fc2"][1], residual=x, early_pdl=early_pdl, mma_tiler_mn=tile)
    xq, xs = audio.layernorm_mxfp8(x, W["ln_post"][0], W["ln_post"][1])
    xq2, xs2 = fso.dense.linear_quantized(xq, W["p1"][2], x_scale=xs, bias=W["p1"][1], activation="gelu_erf")
    return linear_mxfp8_unified(xq2, W["p2"][2].weight, xs2, W["p2"][2].scale, bias=W["p2"][1], early_pdl=False, mma_tiler_mn=tile)


def main():
    W = make_tower_weights(n_layers=32)
    print("=== 32L Core Audio Tower: Zero-Host-Routing Unified Cluster-DSMEM v2.2 (2032 MHz, us) ===", flush=True)
    print(f"{'M':>5} | {'BF16':>8} | {'0.3.0 fqkv':>10} | {'Uni 128x64':>10} | {'vs 0.3.0':>9} | {'vs BF16':>9} | {'Uni 128x128':>11} | {'cos(BF16)':>9} | clocks", flush=True)
    print("-" * 108, flush=True)
    for M in [78, 117, 182, 364, 533, 1079]:
        torch.manual_seed(6000 + M)
        xc = (torch.randn(M, 7680, device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
        y_bf = run_bf16(xc, W, cudnn_attn)
        y_64 = run_unified_tower(xc, W, (128, 64))
        cos64 = F.cosine_similarity(y_bf.float().flatten(), y_64.float().flatten(), dim=0).item()
        assert cos64 > 0.99, f"cos too low at M={M}: {cos64}"

        g_bf, _ = capture(lambda: run_bf16(xc, W, cudnn_attn))
        g_03, _ = capture(lambda: run_fso030_fused_qkv(xc, W, cudnn_attn))
        g_64, y_64_g = capture(lambda: run_unified_tower(xc, W, (128, 64)))
        g_128, _ = capture(lambda: run_unified_tower(xc, W, (128, 128)))
        assert torch.equal(y_64.view(torch.uint16), y_64_g.view(torch.uint16)), f"Eager!=Graph at M={M}"

        t, _, c1 = time_graphs_clocked({"bf": g_bf, "f03": g_03, "u64": g_64, "u128": g_128}, iters=40, rounds=4)
        print(
            f"{M:>5} | {t['bf']:>8.1f} | {t['f03']:>10.1f} | {t['u64']:>10.1f} | "
            f"{(t['u64']/t['f03']-1)*100:>+8.1f}% | {(t['u64']/t['bf']-1)*100:>+8.1f}% | "
            f"{t['u128']:>11.1f} | {cos64:>9.6f} | {c1}",
            flush=True,
        )
        del g_bf, g_03, g_64, g_128


if __name__ == "__main__":
    main()
