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
"""
import sys
import torch
import fish_scales_ops as fso


def reference_check(topk_ids, E, block_m, sorted_eids, flat_to_sorted, p_actual):
    M, topk = topk_ids.shape
    R = M * topk
    flat = topk_ids.reshape(-1).tolist()
    cnt = [0] * E
    for e in flat:
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
    # scheduler reads): real blocks -> owning expert, slack starts -> -1.
    for b in range(p_max // block_m):
        r0 = b * block_m
        if r0 < P:
            e = max(e for e in range(E) if off[e] <= r0)
            assert se[r0] == e, f"block {b} start eid {se[r0]} != owning {e}"
        else:
            assert se[r0] == -1, f"slack block {b} start eid {se[r0]} != -1"

    # flat_to_sorted is a bijection: distinct rows, each in its expert's run,
    # and the block owning the row is labelled with the pair's expert.
    assert len(set(fts)) == R, "flat_to_sorted rows not distinct"
    for i in range(R):
        r = fts[i]
        e = flat[i]
        assert off[e] <= r < off[e] + padded[e], f"pair {i} row {r} outside expert {e} run"
        bstart = (r // block_m) * block_m
        assert se[bstart] == e, f"pair {i} block start eid {se[bstart]} != {e}"


def main():
    torch.manual_seed(0)
    E = 128
    cases = [
        (1, 8, 64), (1, 8, 16),        # decode-1, block_m 64 and swap-AB 16
        (2, 8, 64), (8, 8, 64), (8, 8, 16),
        (32, 8, 64), (64, 8, 16), (128, 8, 64), (128, 8, 16),
        (256, 8, 64),                  # prefill-ish (most experts active)
    ]
    for M, topk, block_m in cases:
        topk_ids = torch.stack(
            [torch.randperm(E, device="cuda")[:topk] for _ in range(M)]
        ).to(torch.int32)
        se, fts, npad = fso.gemm.moe_build_sorted(topk_ids, E, block_m)
        p_actual = int(npad.item())
        reference_check(topk_ids.cpu(), E, block_m, se.cpu(), fts.cpu(), p_actual)
        R = M * topk
        active = int((torch.bincount(topk_ids.reshape(-1), minlength=E) > 0).sum())
        print(f"M={M:4d} topk={topk} bm={block_m:3d} : "
              f"P_max={se.numel():5d} P_actual={p_actual:5d} "
              f"active_experts={active:3d}  OK")

    # CUDA-graph capture safety: fixed P_max buffers, device length scalar;
    # replay with different routing must re-sort correctly.
    M, topk, block_m = 8, 8, 64
    topk_ids = torch.stack(
        [torch.randperm(E, device="cuda")[:topk] for _ in range(M)]).to(torch.int32)
    for _ in range(3):
        fso.gemm.moe_build_sorted(topk_ids, E, block_m)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        se, fts, npad = fso.gemm.moe_build_sorted(topk_ids, E, block_m)
    new = torch.stack(
        [torch.randperm(E, device="cuda")[:topk] for _ in range(M)]).to(torch.int32)
    topk_ids.copy_(new)
    g.replay(); torch.cuda.synchronize()
    reference_check(topk_ids.cpu(), E, block_m, se.cpu(), fts.cpu(), int(npad.item()))
    print("graph replay with rerouted topk : OK")
    print("\nH4a moe_build_sorted: ALL PASS")


if __name__ == "__main__":
    sys.exit(main())
