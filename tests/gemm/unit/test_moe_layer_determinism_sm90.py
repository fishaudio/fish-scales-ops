#!/usr/bin/env python3
"""Bitwise run-to-run stability of the composed sm_90 MoE layer (moe_layer_fp8_sm90) on both shape
families and on both GEMM paths. Every eager call must produce the identical tensor, and a CUDA-graph
replay must equal the eager result. Added 2026-09-25 after the two-resident-CTA swap-AB build produced
one corrupted 16 x 128 FC1 tile every few launches at M >= 512 when its stage count did not divide K
(dispatch.cuh keeps two-CTA builds on K-divisible stage counts); a test that runs each M once cannot
see that, and the dispatch test's cosine gate cannot either."""
import sys
import torch
import fish_scales_ops as fso

FAMILIES = {
    # (E, topk, hidden, inter)
    "C_35a3": (256, 8, 2048, 512),   # K = 2048 / 512
    "B_30a3": (128, 8, 2048, 768),   # K = 2048 / 768
}
RUNS = 20


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        where = "no CUDA device" if not torch.cuda.is_available() \
            else f"device is sm_{torch.cuda.get_device_capability()[0]}x"
        print(f"SKIP: test_moe_layer_determinism_sm90 is sm_90 (H200) only ({where})")
        return 0
    torch.manual_seed(0)
    dev = "cuda"
    for fam, (E, TOPK, HIDDEN, INTER) in FAMILIES.items():
        w13 = torch.randn(E, 2 * INTER, HIDDEN, device=dev, dtype=torch.bfloat16) * 0.02
        w2 = torch.randn(E, HIDDEN, INTER, device=dev, dtype=torch.bfloat16) * 0.02
        w13f, sw13 = fso.gemm.quantize_moe_weights_1x128_fp8_sm90(w13)
        w2f, sw2 = fso.gemm.quantize_moe_weights_1x128_fp8_sm90(w2)
        for M in (8, 32, 256, 512, 1024, 2048):
            g = torch.Generator(device=dev)
            g.manual_seed(1234 + M)
            hidden = torch.randn(M, HIDDEN, device=dev, dtype=torch.bfloat16, generator=g)
            topk_ids = torch.rand(M, E, device=dev, generator=g).topk(TOPK, dim=1).indices.to(torch.int32)
            topk_w = torch.softmax(torch.rand(M, TOPK, device=dev, generator=g), dim=1).float()
            fn = lambda: fso.gemm.moe_layer_fp8_sm90(hidden, w13f, sw13, w2f, sw2, topk_ids, topk_w)
            ref = fn()
            torch.cuda.synchronize()
            bad = 0
            for _ in range(RUNS):
                y = fn()
                torch.cuda.synchronize()
                if not torch.equal(y, ref):
                    bad += 1
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=s):
                out = fn()
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            graph_ok = torch.equal(out, ref)
            bn = fso.gemm.moe_swap_ab_block_n(M, E, TOPK)
            path = f"swap-AB/{bn}" if bn is not None else "contig"
            status = "OK" if (bad == 0 and graph_ok) else "FAIL"
            print(f"{fam} M={M:5d} {path:10s}: {RUNS} eager runs, {bad} differ; graph replay "
                  f"{'==' if graph_ok else '!='} eager  {status}")
            assert bad == 0, f"{fam} M={M}: {bad}/{RUNS} eager runs differ from the first"
            assert graph_ok, f"{fam} M={M}: graph replay differs from eager"
    print("\nmoe_layer_fp8_sm90 determinism: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
