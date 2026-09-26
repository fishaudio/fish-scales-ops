#!/usr/bin/env python3
"""The composed sm_90 MoE layer entry moe_layer_fp8_sm90 with its internal
dispatch (swap-AB with block_n = 16 / 32 / 64 chosen from the routed rows per
active expert, non-swap contiguous block_m=64 above 64 rows per expert).
Validates correctness at every cascade step and across the last threshold, and
CUDA-graph capture safety for a decode M and a prefill M."""
import sys, math
import torch
import fish_scales_ops as fso

E, TOPK, HIDDEN, INTER = 128, 8, 2048, 768


def moe_ref(hidden, w13, w2, topk_ids, topk_w):
    M = hidden.shape[0]
    out = torch.zeros(M, HIDDEN, device="cuda", dtype=torch.float32)
    for t in range(M):
        for j in range(TOPK):
            e = int(topk_ids[t, j])
            gu = hidden[t].float() @ w13[e].T.float()
            g, u = gu.chunk(2, dim=-1)
            out[t] += float(topk_w[t, j]) * ((torch.nn.functional.silu(g) * u) @ w2[e].T.float())
    return out.to(torch.bfloat16)


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def main():
    # This suite drives the composed sm_90 (H200) layer entry only; on any other
    # device the first op raises NotImplementedError, which used to exit 1 and
    # read as a failure. Skip explicitly instead, in the form
    # test_fp8_grouped_sm90.py uses (run b300_round3_20260922/M-A3).
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        where = "no CUDA device" if not torch.cuda.is_available() \
            else f"device is sm_{torch.cuda.get_device_capability()[0]}x"
        print(f"SKIP: test_moe_layer_dispatch_sm90 is sm_90 (H200) only ({where})")
        return 0

    torch.manual_seed(0)
    w13 = torch.randn(E, 2 * INTER, HIDDEN, device="cuda", dtype=torch.bfloat16) / math.sqrt(HIDDEN)
    w2 = torch.randn(E, HIDDEN, INTER, device="cuda", dtype=torch.bfloat16) / math.sqrt(INTER)
    w13f, sw13 = fso.gemm.quantize_moe_weights_1x128_fp8_sm90(w13)
    w2f, sw2 = fso.gemm.quantize_moe_weights_1x128_fp8_sm90(w2)
    thr = fso.gemm.moe_swap_ab_max_m(E, TOPK)  # inclusive upper M of the swap-AB path
    print(f"moe_swap_ab_max_m(E={E}, topk={TOPK}) = {thr}")
    # E=128/top-8: rows per expert = M/16 -> block_n 16 up to M=192, 32 up to 384, 64 up to 1024

    print("=== dispatch cos across the threshold ===")
    for M in (1, 8, 64, 192, 193, 384, 385, thr, thr + 1):
        hidden = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1
        topk_ids = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(M)]).to(torch.int32)
        topk_w = torch.rand(M, TOPK, device="cuda", dtype=torch.float32)
        y = fso.gemm.moe_layer_fp8_sm90(hidden, w13f, sw13, w2f, sw2, topk_ids, topk_w)
        bn = fso.gemm.moe_swap_ab_block_n(M, E, TOPK)
        path = f"swap-AB/{bn}" if bn is not None else "contig"
        c = cos(y, moe_ref(hidden, w13, w2, topk_ids, topk_w))
        print(f"M={M:4d} -> {path:10s} cos={c:.4f}  {'OK' if c > 0.995 else 'FAIL'}")
        assert c > 0.995, f"cos {c} at M={M}"

    print("\n=== CUDA-graph capture + reroute (decode M=8, prefill M=256) ===")
    for M in (8, 256):
        hidden = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1
        topk_ids = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(M)]).to(torch.int32)
        topk_w = torch.rand(M, TOPK, device="cuda", dtype=torch.float32)
        fn = lambda: fso.gemm.moe_layer_fp8_sm90(hidden, w13f, sw13, w2f, sw2, topk_ids, topk_w)
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
        topk_ids.copy_(torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(M)]).to(torch.int32))
        topk_w.copy_(torch.rand(M, TOPK, device="cuda"))
        hidden.copy_(torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1)
        g.replay(); torch.cuda.synchronize()
        c = cos(y, moe_ref(hidden, w13, w2, topk_ids, topk_w))
        print(f"M={M:4d} graph reroute cos={c:.4f}  {'OK' if c > 0.995 else 'FAIL'}")
        assert c > 0.995
    print("\nmoe_layer_fp8_sm90 dispatch: ALL PASS")


if __name__ == "__main__":
    sys.exit(main())
