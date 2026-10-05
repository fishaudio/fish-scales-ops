#!/usr/bin/env python3
"""H4: validate moe_build_sorted (the contiguous/triton-style sorted-layout
builder) against a python reference. Checks the invariants the DeepGEMM
GroupedContiguous scheduler and the gather/silu/combine consumers rely on:
  - num_padded_dev == sum of per-expert runs padded to block_m
  - sorted_expert_ids block-START labels == owning expert, -1 past the padded
    length (the only entries the scheduler reads, at m_block*BLOCK_M)
  - flat_to_sorted is a bijection: distinct rows, each inside its expert's run,
    and the block owning each pair's row is labelled with that pair's expert
Intra-expert order is atomic-arrival (a permutation), so we test sets, not order.

Ids outside [0, E) (-1 or E, as a serving stack's padded rows and expert-parallel
remote entries) are masked: counted into no expert, flat_to_sorted -1. Both builder
kernels are covered: up to 2048 routed pairs the single-warp-group kernel, above
that the multi-warp kernel with per-warp privatised counters (H2 Phase 2a,
2026-10-05), including expert counts that shrink its warps to fit shared memory
(E = 1024). Every case also checks the sorted gather-quantize behind the builder
(quantize_1x128_sorted_gather_sm90): at every routed pair's row, the FP8 bytes of
that pair's token equal the flat quantizer's (quantize_1x128_fp8 with
use_ue8m0=False, the sm_90 scales) and its 1x128 scales equal amax * (1 / 448).
"""
import bisect
import sys
import torch
import fish_scales_ops as fso


def reference_check(topk_ids, E, block_m, sorted_eids, flat_to_sorted, p_actual):
    M, topk = topk_ids.shape
    R = M * topk
    flat = topk_ids.reshape(-1).tolist()
    cnt = [0] * E
    for e in flat:
        if 0 <= e < E:
            cnt[e] += 1
    padded = [((c + block_m - 1) // block_m) * block_m if c > 0 else 0 for c in cnt]
    off = [0] * (E + 1)
    for e in range(E):
        off[e + 1] = off[e] + padded[e]
    P = off[E]
    assert p_actual == P, f"num_padded {p_actual} != ref {P}"

    se = sorted_eids.tolist()
    fts = flat_to_sorted.tolist()
    p_max = len(se)

    # Block-start expert labels (the only sorted_expert_ids entries the
    # scheduler reads): real blocks -> owning expert (the last expert whose run
    # starts at or before the block), slack starts -> -1.
    for b in range(p_max // block_m):
        r0 = b * block_m
        if r0 < P:
            e = bisect.bisect_right(off, r0) - 1
            assert se[r0] == e, f"block {b} start eid {se[r0]} != owning {e}"
        else:
            assert se[r0] == -1, f"slack block {b} start eid {se[r0]} != -1"

    # flat_to_sorted: -1 at the masked pairs; on the routed pairs a bijection
    # onto distinct rows, each in its expert's run, and the block owning the row
    # is labelled with the pair's expert.
    routed = [i for i in range(R) if 0 <= flat[i] < E]
    for i in range(R):
        if not (0 <= flat[i] < E):
            assert fts[i] == -1, f"masked pair {i} (id {flat[i]}) row {fts[i]} != -1"
    assert len({fts[i] for i in routed}) == len(routed), "flat_to_sorted rows not distinct"
    for i in routed:
        r = fts[i]
        e = flat[i]
        assert off[e] <= r < off[e] + padded[e], f"pair {i} row {r} outside expert {e} run"
        bstart = (r // block_m) * block_m
        assert se[bstart] == e, f"pair {i} block start eid {se[bstart]} != {e}"
    return len(routed)


def gather_check(hidden, topk, flat_to_sorted, p_max):
    """quantize_1x128_sorted_gather_sm90 at every routed pair's sorted row: the
    FP8 bytes equal the flat quantizer's for the pair's token row (both use
    qs = 448 / amax), and the scales equal amax * (1 / 448) in fp32, amax the
    block's |max| clamped at 1e-10 (the flat quantizer stores 1 / qs instead,
    which can differ in the last bit)."""
    hq, sh = fso.compat.quantize_1x128_sorted_gather_sm90(hidden, flat_to_sorted, p_max, topk)
    xq, _ = fso.compat.quantize_1x128_fp8(hidden, use_ue8m0=False)
    M, K = hidden.shape
    amax = hidden.float().abs().view(M, K // 128, 128).amax(dim=2).clamp_min(1e-10)
    ref_s = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32, device=hidden.device)
    fts = flat_to_sorted.long()
    keep = fts >= 0
    rows = fts[keep]
    tokens = torch.arange(fts.numel(), device=fts.device)[keep] // topk
    ok_q = torch.equal(hq.view(torch.uint8)[rows], xq.view(torch.uint8)[tokens])
    ok_s = torch.equal(sh[:, rows].t().contiguous().view(torch.int32), ref_s[tokens].contiguous().view(torch.int32))
    assert ok_q, "sorted gather-quantize: FP8 bytes differ from the flat quantizer's"
    assert ok_s, "sorted gather-quantize: scales differ from amax * (1 / 448)"


def draw_ids(M, E, topk, masked):
    ids = torch.stack([torch.randperm(E, device="cuda")[:topk] for _ in range(M)]).to(torch.int32)
    if masked:
        # about one entry in ten masked, alternately -1 and E
        mask = torch.rand(M, topk, device="cuda") < 0.1
        fill = torch.where(torch.arange(M * topk, device="cuda").view(M, topk) % 2 == 0, -1, E).to(torch.int32)
        ids = torch.where(mask, fill, ids)
    return ids.contiguous()


def main():
    torch.manual_seed(0)
    E = 128
    H = 2048
    cases = [
        (1, 8, 64), (1, 8, 16),        # decode-1, block_m 64 and swap-AB 16
        (2, 8, 64), (8, 8, 64), (8, 8, 16),
        (32, 8, 64), (64, 8, 16), (128, 8, 64), (128, 8, 16),
        (256, 8, 64),                  # most experts active, 2048 pairs: the last single-warp-group size
        (512, 8, 64), (1024, 8, 16),   # prefill: the multi-warp builder
        (4096, 8, 64), (8192, 8, 16),
    ]
    # (M, topk, block_m, E, masked): masked ids on both builders, and expert counts
    # that shrink the multi-warp builder's warps to its shared-memory budget
    cases = [(M, topk, bm, E, False) for M, topk, bm in cases] + [
        (64, 8, 16, 256, True), (100, 8, 32, 256, True),
        (1024, 8, 64, 256, True), (8192, 8, 64, 256, True),
        (512, 8, 64, 1024, True), (4096, 8, 16, 1024, True),
        (1024, 8, 64, 128, "all"),     # every entry masked: nothing routed
    ]
    # moe_build_sorted runs on every architecture; the sorted gather-quantize it feeds is an sm_90-only op
    # (quantize_1x128_sorted_gather_sm90 refuses other devices), so its check runs on sm_90 only.
    gather_on_device = torch.cuda.get_device_capability(0)[0] == 9
    if not gather_on_device:
        print("  note: the sorted gather-quantize is sm_90 only; its check is skipped on this device")
    for M, topk, block_m, e_count, masked in cases:
        if masked == "all":
            topk_ids = torch.full((M, topk), -1, device="cuda", dtype=torch.int32)
        else:
            topk_ids = draw_ids(M, e_count, topk, masked)
        se, fts, npad = fso.compat.moe_build_sorted(topk_ids, e_count, block_m)
        p_actual = int(npad.item())
        routed = reference_check(topk_ids.cpu(), e_count, block_m, se.cpu(), fts.cpu(), p_actual)
        hidden = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
        if gather_on_device:
            gather_check(hidden, topk, fts, int(se.shape[0]))
        ids = topk_ids.reshape(-1)
        ids = ids[(ids >= 0) & (ids < e_count)].long()
        active = int((torch.bincount(ids, minlength=e_count) > 0).sum())
        print(f"M={M:4d} topk={topk} bm={block_m:3d} E={e_count:4d} masked={str(masked):5s}: "
              f"P_max={se.numel():6d} P_actual={p_actual:6d} routed={routed:6d} "
              f"active_experts={active:4d}  OK")

    # CUDA-graph capture safety: fixed P_max buffers, device length scalar;
    # replay with different routing must re-sort correctly. Once per builder
    # kernel (64 pairs, and 8192 pairs for the multi-warp builder).
    for M, topk, block_m in ((8, 8, 64), (1024, 8, 64)):
        topk_ids = torch.stack(
            [torch.randperm(E, device="cuda")[:topk] for _ in range(M)]).to(torch.int32)
        for _ in range(3):
            fso.compat.moe_build_sorted(topk_ids, E, block_m)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            se, fts, npad = fso.compat.moe_build_sorted(topk_ids, E, block_m)
        new = draw_ids(M, E, topk, True)
        topk_ids.copy_(new)
        g.replay(); torch.cuda.synchronize()
        reference_check(topk_ids.cpu(), E, block_m, se.cpu(), fts.cpu(), int(npad.item()))
        print(f"graph replay with rerouted topk (M={M}, masked ids) : OK")
    print("\nH4a moe_build_sorted: ALL PASS")


if __name__ == "__main__":
    sys.exit(main())
