#!/usr/bin/env python3
"""sm_90 fused-FC1 step 1: the two-warp-group non-swap grouped-contiguous FC1 must equal today's FC1 bit for bit.

linear_fp8_grouped_contiguous_2wg computes, in one CTA, gate block b and up block b of one 64-row activation tile:
math warp-group 0 multiplies the gate rows of the stacked [gate; up] weight and math warp-group 1 the up rows, and
each warp-group runs today's WGMMA and promotion sequence. Its bf16 output [P_max, 2I] must therefore equal
linear_fp8_grouped_contiguous on the same inputs.

The inputs are built exactly as moe_layer_fp8_sm90 builds them on its non-swap path: moe_build_sorted with block 64,
then quantize_1x128_sorted_gather_sm90. Routings are random from fixed seeds, and about one entry in ten carries a
masked expert id (-1 or E, the ids sglang writes into the padded rows of a graph bucket), which the layer skips. The
comparison covers every real row, i.e. every sorted row that flat_to_sorted assigns to a routed pair (the rows the
gather wrote). Intra-expert padding rows and the slack past the padded length are not compared: the gather leaves
their activations and scales uninitialised, so both GEMMs produce undefined values there.

A CUDA-graph capture of the whole FC1 input chain plus the new op, replayed after the routing, the hidden states and
the masked entries changed, must again equal today's FC1 run on the replayed buffers.

Families: B = Qwen3-30B-A3B routed experts (E=128, top-8, H=2048, I=768) and C = Qwen3.5-35B-A3B routed experts
(E=256, top-8, H=2048, I=512), at M in {1, 64, 256, 1024, 4096, 8192}: 12 cells.
"""
import math
import sys

import torch

import fish_scales_ops as fso
from fish_scales_ops.gemm.fp8 import linear_fp8_grouped_contiguous_2wg

FAMILIES = {
    # (E, topk, hidden, inter)
    "B": (128, 8, 2048, 768),
    "C": (256, 8, 2048, 512),
}
MS = (1, 64, 256, 1024, 4096, 8192)
BLOCK_M = 64


def make_topk_ids(M, E, topk, gen):
    """Random routing without replacement per token, with about 10 % of the entries masked to -1 or E (at least one
    of each: entry (0, 0) gets -1 and entry (M-1, topk-1) gets E)."""
    ids = torch.rand(M, E, device="cuda", generator=gen).topk(topk, dim=1).indices.to(torch.int32)
    mask = torch.rand(M, topk, device="cuda", generator=gen) < 0.1
    mask[0, 0] = True
    mask[M - 1, topk - 1] = True
    flat = torch.arange(M * topk, device="cuda").view(M, topk)
    fill = torch.where(flat % 2 == 0, -1, E).to(torch.int32)
    fill[M - 1, topk - 1] = E
    return torch.where(mask, fill, ids).contiguous()


def fc1_inputs(hidden, topk_ids, E, topk):
    """The non-swap FC1 inputs of moe_layer_fp8_sm90 (block 64)."""
    ops = torch.ops.fish_scales_ops
    se, fts, npad = ops.moe_build_sorted(topk_ids, E, BLOCK_M)
    p_max = se.numel()
    hq, sh = ops.quantize_1x128_sorted_gather_sm90(hidden, fts, p_max, topk)
    return se, fts, npad, hq, sh


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        where = "no CUDA device" if not torch.cuda.is_available() \
            else f"device is sm_{torch.cuda.get_device_capability()[0]}x"
        print(f"SKIP: test_fp8_contiguous_2wg_sm90 is sm_90 (H200) only ({where})")
        return 0

    ops = torch.ops.fish_scales_ops
    passed, total = 0, 0
    weights = {}
    print("=== two-warp-group FC1 vs today's FC1, real rows, torch.equal ===")
    for fam, (E, TOPK, H, I) in FAMILIES.items():
        torch.manual_seed(20260930 + E)
        w13 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) / math.sqrt(H)
        w13q, sw13 = fso.compat.quantize_moe_weights_1x128_fp8_sm90(w13)
        del w13
        weights[fam] = (w13q, sw13)
        for M in MS:
            gen = torch.Generator(device="cuda")
            gen.manual_seed(7919 * M + E)
            hidden = torch.randn(M, H, device="cuda", dtype=torch.bfloat16, generator=gen)
            topk_ids = make_topk_ids(M, E, TOPK, gen)
            expected_m = max(1, (M * TOPK + E - 1) // E)
            se, fts, npad, hq, sh = fc1_inputs(hidden, topk_ids, E, TOPK)
            ref = ops.linear_fp8_grouped_contiguous(hq, w13q, sh, sw13, se, BLOCK_M, expected_m)
            out = linear_fp8_grouped_contiguous_2wg(hq, w13q, sh, sw13, se, BLOCK_M, expected_m)
            torch.cuda.synchronize()
            real = fts[fts >= 0].long()
            n_masked = int((fts < 0).sum())
            # The compared rows must hold real data, so equality is not vacuous.
            sane = real.numel() > 0 and bool(torch.isfinite(ref[real].float()).all()) \
                and bool((ref[real] != 0).any())
            ok = sane and torch.equal(out[real], ref[real])
            total += 1
            passed += int(ok)
            print(f"{fam} M={M:5d}: P_max={se.numel():6d} padded={int(npad.item()):6d} real rows={real.numel():6d} "
                  f"masked entries={n_masked:4d}  {'OK' if ok else 'FAIL'}")

    print("\n=== CUDA-graph capture, replayed with a new routing ===")
    fam, M = "B", 1024
    E, TOPK, H, I = FAMILIES[fam]
    w13q, sw13 = weights[fam]
    expected_m = max(1, (M * TOPK + E - 1) // E)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(12345)
    hidden = torch.randn(M, H, device="cuda", dtype=torch.bfloat16, generator=gen)
    topk_ids = make_topk_ids(M, E, TOPK, gen)

    def chain():
        se, fts, npad, hq, sh = fc1_inputs(hidden, topk_ids, E, TOPK)
        out = linear_fp8_grouped_contiguous_2wg(hq, w13q, sh, sw13, se, BLOCK_M, expected_m)
        return se, fts, hq, sh, out

    for _ in range(3):
        chain()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            chain()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        se_g, fts_g, hq_g, sh_g, out_g = chain()
    torch.cuda.synchronize()
    hidden.copy_(torch.randn(M, H, device="cuda", dtype=torch.bfloat16, generator=gen))
    topk_ids.copy_(make_topk_ids(M, E, TOPK, gen))
    out_g.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    ref = ops.linear_fp8_grouped_contiguous(hq_g, w13q, sh_g, sw13, se_g, BLOCK_M, expected_m)
    torch.cuda.synchronize()
    real = fts_g[fts_g >= 0].long()
    ok = real.numel() > 0 and torch.equal(out_g[real], ref[real])
    total += 1
    passed += int(ok)
    print(f"{fam} M={M}: graph replay after reroute == today's FC1 on the replayed buffers "
          f"({real.numel()} real rows)  {'OK' if ok else 'FAIL'}")

    print(f"\n{passed}/{total} checks passed")
    if passed != total:
        print("two-warp-group FC1: FAIL")
        return 1
    print("two-warp-group FC1: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
