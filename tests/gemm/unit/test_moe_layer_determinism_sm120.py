#!/usr/bin/env python3
"""Bitwise run-to-run stability of the composed sm_120 MXFP8 MoE layer
(moe_layer_mxfp8_sm120) on both shape families and in both expert-parallel forms
of family C. Two things are asserted per cell: every eager call produces the
identical tensor, and a CUDA-graph replay equals the eager result. The sm_120
twin of test_moe_layer_determinism_sm90.py; it is what would catch a
non-deterministic tile or epilogue in the grouped MXFP8 kernels, which a
cosine-gated test cannot see."""
import sys
import torch
import fish_scales_ops as fso

FAMILIES = {
    # label: (E_local, topk, hidden, inter)
    "C_35a3_tp1": (256, 8, 2048, 512),  # the drama artifact, one rank
    "C_35a3_ep2": (128, 8, 2048, 512),  # the same artifact at tp2/ep2: half the experts per rank
    "B_30a3": (128, 8, 2048, 768),
}
RUNS = 20


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        where = "no CUDA device" if not torch.cuda.is_available() \
            else f"device is sm_{torch.cuda.get_device_capability()[0]}x"
        print(f"SKIP: test_moe_layer_determinism_sm120 is sm_120/121 (RTX 5090) only ({where})")
        return 0
    torch.manual_seed(0)
    dev = "cuda"
    for fam, (E, TOPK, HIDDEN, INTER) in FAMILIES.items():
        w13 = torch.randn(E, 2 * INTER, HIDDEN, device=dev, dtype=torch.bfloat16) * 0.02
        w2 = torch.randn(E, HIDDEN, INTER, device=dev, dtype=torch.bfloat16) * 0.02
        w13f, sw13 = fso.compat.quantize_moe_weights_1x32_fp8(w13)
        w2f, sw2 = fso.compat.quantize_moe_weights_1x32_fp8(w2)
        del w13, w2
        for M in (1, 8, 32, 64, 256, 512, 1024):
            g = torch.Generator(device=dev)
            g.manual_seed(1234 + M)
            hidden = torch.randn(M, HIDDEN, device=dev, dtype=torch.bfloat16, generator=g)
            topk_ids = torch.rand(M, E, device=dev, generator=g).topk(TOPK, dim=1).indices.to(torch.int32)
            topk_w = torch.softmax(torch.rand(M, TOPK, device=dev, generator=g), dim=1).float()
            fn = lambda: fso.compat.moe_layer_mxfp8_sm120(
                hidden, w13f, sw13, w2f, sw2, topk_ids, topk_w)
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
            status = "OK" if (bad == 0 and graph_ok) else "FAIL"
            print(f"{fam:12s} M={M:5d}: {RUNS} eager runs, {bad} differ from the first; graph replay "
                  f"{'==' if graph_ok else '!='} eager  {status}")
            assert bad == 0, f"{fam} M={M}: {bad}/{RUNS} eager runs differ from the first"
            assert graph_ok, f"{fam} M={M}: graph replay differs from eager"
    print("\nmoe_layer_mxfp8_sm120 determinism: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
