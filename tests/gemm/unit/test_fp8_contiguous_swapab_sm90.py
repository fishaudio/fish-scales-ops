#!/usr/bin/env python3
"""H4b: sm_90 contiguous swap-AB grouped MoE (block_n=16 activation tiling).
Validates the swap-AB layer against a per-expert BF16 reference + graph reroute.
Qwen3-30B-A3B: E=128, topk=8, HIDDEN=2048, INTER=768, block_n=16."""
import sys, math
import torch
import fish_scales_ops as fso

E, TOPK, HIDDEN, INTER = 128, 8, 2048, 768
BN = 16


def moe_ref(hidden, w13, w2, topk_ids, topk_w):
    M = hidden.shape[0]
    out = torch.zeros(M, HIDDEN, device="cuda", dtype=torch.float32)
    for t in range(M):
        for j in range(TOPK):
            e = int(topk_ids[t, j])
            gu = hidden[t].float() @ w13[e].T.float()
            gate, up = gu.chunk(2, dim=-1)
            h = torch.nn.functional.silu(gate) * up
            out[t] += float(topk_w[t, j]) * (h @ w2[e].T.float())
    return out.to(torch.bfloat16)


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def swap_layer(hidden, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w, expected_m):
    se, fts, npad = fso.gemm.moe_build_sorted(topk_ids, E, BN)
    p_max = se.numel()
    hq, sh = fso.gemm.quantize_1x128_sorted_gather_sm90(hidden, fts, p_max, TOPK)
    gu = fso.gemm.linear_fp8_grouped_contiguous_swapab(hq, w13_fp8, sh, sw13, se, BN, expected_m)
    dq, sd = fso.gemm.silu_chunk_mul_quantize_1x128_sorted_sm90(gu, fts)
    dn = fso.gemm.linear_fp8_grouped_contiguous_swapab(dq, w2_fp8, sd, sw2, se, BN, expected_m)
    return fso.gemm.moe_combine_sorted(dn, fts, topk_w)


def main():
    torch.manual_seed(0)
    w13 = torch.randn(E, 2 * INTER, HIDDEN, device="cuda", dtype=torch.bfloat16) / math.sqrt(HIDDEN)
    w2 = torch.randn(E, HIDDEN, INTER, device="cuda", dtype=torch.bfloat16) / math.sqrt(INTER)
    w13_fp8, sw13 = fso.gemm.quantize_moe_weights_1x128_fp8_sm90(w13)
    w2_fp8, sw2 = fso.gemm.quantize_moe_weights_1x128_fp8_sm90(w2)

    print("=== swap-AB layer cos vs per-expert BF16 ===")
    for M in (1, 8, 32, 64, 128):
        hidden = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1
        topk_ids = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(M)]).to(torch.int32)
        topk_w = torch.rand(M, TOPK, device="cuda", dtype=torch.float32)
        expected_m = max(1, (M * TOPK + E - 1) // E)
        y = swap_layer(hidden, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w, expected_m)
        ref = moe_ref(hidden, w13, w2, topk_ids, topk_w)
        c = cos(y, ref)
        print(f"M={M:4d} : cos={c:.4f}  {'OK' if c > 0.995 else 'FAIL'}")
        assert c > 0.995, f"cos {c} too low at M={M}"

    print("\n=== graph capture + reroute (M=8) ===")
    M = 8
    hidden = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1
    topk_ids = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(M)]).to(torch.int32)
    topk_w = torch.rand(M, TOPK, device="cuda", dtype=torch.float32)
    expected_m = max(1, (M * TOPK + E - 1) // E)
    fn = lambda: swap_layer(hidden, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w, expected_m)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = fn()
    ni = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(M)]).to(torch.int32)
    topk_ids.copy_(ni)
    topk_w.copy_(torch.rand(M, TOPK, device="cuda"))
    hidden.copy_(torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1)
    g.replay(); torch.cuda.synchronize()
    ref = moe_ref(hidden, w13, w2, topk_ids, topk_w)
    c = cos(y, ref)
    print(f"graph reroute : cos={c:.4f}  {'OK' if c > 0.995 else 'FAIL'}")
    assert c > 0.995
    print("\nH4b swap-AB layer: ALL PASS")


if __name__ == "__main__":
    sys.exit(main())
