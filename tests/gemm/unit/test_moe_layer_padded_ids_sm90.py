#!/usr/bin/env python3
"""Masked padded-row expert ids through moe_layer_fp8_sm90. sglang fills the expert ids of the rows past
num_token_non_padded of a CUDA-graph or piecewise-graph bucket with -1 or with num_experts (its moe_align
overflow slot). Both must take no expert compute, leave every real row bit-identical to the unmasked call
and zero the masked rows, eagerly and through a graph captured with one real-row count and replayed with
another. Added 2026-09-25 after the first serving run of the apex fork's --moe-runner-backend fso: the
sorted builder indexed cnt[E] / off[E], the gather wrote through a garbage sorted row and the first padded
prefill bucket died (surfacing as CUBLAS_STATUS_EXECUTION_FAILED at the next cuBLAS call)."""
import sys
import torch
import fish_scales_ops as fso

FAMILIES = {
    # (E, topk, hidden, inter)
    "C_35a3": (256, 8, 2048, 512),
    "B_30a3": (128, 8, 2048, 768),
}


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        where = "no CUDA device" if not torch.cuda.is_available() \
            else f"device is sm_{torch.cuda.get_device_capability()[0]}x"
        print(f"SKIP: test_moe_layer_padded_ids_sm90 is sm_90 (H200) only ({where})")
        return 0
    torch.manual_seed(0)
    dev = "cuda"
    for fam, (E, TOPK, HIDDEN, INTER) in FAMILIES.items():
        w13 = torch.randn(E, 2 * INTER, HIDDEN, device=dev, dtype=torch.bfloat16) * 0.02
        w2 = torch.randn(E, HIDDEN, INTER, device=dev, dtype=torch.bfloat16) * 0.02
        w13f, sw13 = fso.gemm.quantize_moe_weights_1x128_fp8_sm90(w13)
        w2f, sw2 = fso.gemm.quantize_moe_weights_1x128_fp8_sm90(w2)
        layer = lambda h, ids, w: fso.gemm.moe_layer_fp8_sm90(h, w13f, sw13, w2f, sw2, ids, w)
        for M in (8, 64, 512, 1024, 4096):  # swap-AB 16 / 32 / 64 tiles and the non-swap path
            g = torch.Generator(device=dev)
            g.manual_seed(1234 + M)
            hidden = torch.randn(M, HIDDEN, device=dev, dtype=torch.bfloat16, generator=g)
            ids = torch.rand(M, E, device=dev, generator=g).topk(TOPK, dim=1).indices.to(torch.int32)
            topk_w = torch.softmax(torch.rand(M, TOPK, device=dev, generator=g), dim=1).float()
            ref = layer(hidden, ids, topk_w)
            torch.cuda.synchronize()
            bn = fso.gemm.moe_swap_ab_block_n(M, E, TOPK)
            path = f"swap-AB/{bn}" if bn is not None else "contig"
            for fill in (E, -1):
                for n_real in (M // 2, 1, M - 1, 0):
                    masked = ids.clone()
                    masked[n_real:] = fill
                    out = layer(hidden, masked, topk_w)
                    torch.cuda.synchronize()
                    assert torch.equal(out[:n_real], ref[:n_real]), \
                        f"{fam} M={M} fill={fill} n_real={n_real}: real rows differ from the unmasked call"
                    assert bool((out[n_real:] == 0).all()), \
                        f"{fam} M={M} fill={fill} n_real={n_real}: masked rows are not zero"
            # Graph: capture with M//2 real rows, replay with M-1 real rows written into the same buffers.
            ids_buf = ids.clone()
            ids_buf[M // 2:] = E
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    layer(hidden, ids_buf, topk_w)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=s):
                out_g = layer(hidden, ids_buf, topk_w)
            torch.cuda.synchronize()
            ids_buf.copy_(ids)
            ids_buf[M - 1:] = -1
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out_g[:M - 1], ref[:M - 1]), f"{fam} M={M}: graph replay real rows differ"
            assert bool((out_g[M - 1:] == 0).all()), f"{fam} M={M}: graph replay masked row is not zero"
            print(f"{fam} M={M:5d} {path:10s}: fills (E, -1) x real rows (M/2, 1, M-1, 0) bit-identical, "
                  f"masked rows zero; graph replay with a different real-row count == eager  OK")
    print("\nmoe_layer_fp8_sm90 masked padded ids: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
