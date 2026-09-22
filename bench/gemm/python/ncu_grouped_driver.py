#!/usr/bin/env python3
"""Minimal eager-loop driver for ncu profiling of the grouped MXFP8 GEMM.

Runs ONLY linear_mxfp8_grouped_masked in a fixed-shape loop (no CUDA graph —
ncu replays individual kernel launches). Shapes via argv:

    ncu_grouped_driver.py <proj: gate_up|down> <M> [iters]

Geometry is the Qwen3-30B-A3B expert MLP (E=128, topk=8). Inputs are built
once with the production quantize path; the loop then launches the GEMM
`iters` times back-to-back.
"""
import sys

import torch

import fish_scales_ops as fso

E, TOPK, HIDDEN, INTER = 128, 8, 2048, 768
PROJ = {"gate_up": (2 * INTER, HIDDEN), "down": (HIDDEN, INTER)}


def main():
    proj = sys.argv[1]
    M = int(sys.argv[2])
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    N, K = PROJ[proj]
    torch.manual_seed(M * 1009 + N)

    m_cap = (M + 3) // 4 * 4
    expected_m = max(1, (M * TOPK + E - 1) // E)
    # Host-static upper bound on the number of experts that can hold a row
    # (a top-k router gives an expert at most one row per token). The sm_100
    # grouped dispatcher needs it to size the decode route's grid; without it
    # that route is never taken, so profiling here would miss it.
    max_active_groups = min(M * TOPK, E)

    g = torch.Generator(device="cpu").manual_seed(M * 7 + 3)
    topk_ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(M)])
    topk_ids = topk_ids.to("cuda", torch.int32)
    masked_m, row_map, slot_of_flat = fso.gemm.moe_build_routing(topk_ids, E, m_cap)

    a_fp8, sa = fso.gemm.quantize_1x32_grouped_gather_fp8(
        torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1,
        slot_of_flat, TOPK, E, m_cap)
    w = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16) / (K ** 0.5)
    w_fp8, sw = fso.gemm.quantize_moe_weights_1x32_fp8(w)

    torch.cuda.synchronize()
    for _ in range(iters):
        fso.gemm.linear_mxfp8_grouped_masked(a_fp8, w_fp8, sa, sw, masked_m,
                                             expected_m, max_active_groups)
    torch.cuda.synchronize()
    print(f"done {proj} M={M} iters={iters}")


if __name__ == "__main__":
    main()
