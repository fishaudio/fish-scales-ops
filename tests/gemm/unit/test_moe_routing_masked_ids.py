#!/usr/bin/env python3
"""Expert ids outside [0, num_experts) through the masked-layout routing builders
and the gather-quantize behind them. Runs on every architecture that has the
masked grouped path: the builders and the combine are architecture-independent
glue, so the guard has to hold on sm_90, sm_100/103 and sm_120/121 alike.

Where such ids come from. A serving engine captures one CUDA graph per token
bucket and pads the unused rows of a bucket; sglang labels the padded rows with
the expert id num_experts (the overflow slot its moe_align kernel uses) or with
-1. Under expert parallelism the dispatcher additionally rewrites every expert
another rank owns to -1, which lands on individual top-k entries of real tokens.
A pair carrying such an id names no expert of this rank: it must be counted into
no group, occupy no row of the activation slab, and contribute nothing to its
token's output.

What is checked, against the routing contract:

  * masked_m[g] equals the number of pairs whose id is exactly g, counting only
    the ids inside range;
  * slot_of_flat is -1 at exactly the skipped pairs, and nowhere else;
  * the surviving slots lie inside their own expert's block, below that expert's
    count, and are a permutation (no two pairs share a row);
  * row_map at a surviving slot names that pair's source token;
  * the packed active-expert list, when asked for, is the ascending list of
    experts with a non-zero count followed by -1, i.e. a skipped id never
    appears in it;
  * the fused gather-quantize writes, at every surviving slot, exactly the FP8
    bytes the flat quantizer produces for that pair's source token row -- which
    is what says the skipped pairs neither wrote a row of their own nor
    displaced anyone else's.

Both builder kernels are covered where the dispatcher selects them: the
single-CTA kernel on every architecture, and the multi-CTA kernel (at least
4096 routed pairs) on sm_100/103, where it is selected. The last line names
which ran."""
import sys

import torch

import fish_scales_ops as fso

TOPK = 8
MULTI_MIN_PAIRS = 4096  # kRoutingMultiMinPairs in csrc/gemm/ops/moe_glue.cu


def draw(m, e_pool, topk, seed, device):
    """`topk` distinct ids per token, drawn from the first `e_pool` ids."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack([torch.randperm(e_pool, generator=g)[:topk] for _ in range(m)])
    return ids.to(device, torch.int32)


def check(ids, e, m_cap, topk, masked, row_map, slot, slot_to_expert):
    """Return a list of failure strings; empty means the result is valid."""
    bad = []
    flat = ids.flatten().to(torch.int64)
    keep = (flat >= 0) & (flat < e)
    counts = torch.bincount(flat[keep], minlength=e).to(torch.int32)
    if not torch.equal(masked, counts):
        d = masked.to(torch.int64) - counts.to(torch.int64)
        bad.append(f"masked_m mismatch, max |delta| {int(d.abs().max())}")
    s = slot.to(torch.int64)
    if not torch.equal(s < 0, ~keep):
        n = int(((s < 0) != (~keep)).sum())
        bad.append(f"slot_of_flat marks the wrong pairs as skipped ({n} entries)")
    sv = s[keep]
    if sv.numel():
        if not torch.equal(sv // m_cap, flat[keep]):
            bad.append("slot is not inside the pair's own expert block")
        if bool((sv % m_cap >= counts[flat[keep]].to(torch.int64)).any()):
            bad.append("slot rank is at or beyond the expert's row count")
        if int(torch.unique(sv).numel()) != int(sv.numel()):
            bad.append("slots are not a permutation (duplicate slot)")
        src = torch.arange(ids.shape[0], device=ids.device,
                           dtype=torch.int32).repeat_interleave(topk)[keep]
        if not torch.equal(row_map.to(torch.int64)[sv].to(torch.int32), src):
            bad.append("row_map at the slot does not name the source token")
    if slot_to_expert is not None:
        active = torch.nonzero(counts > 0).flatten().to(torch.int32)
        ref = torch.full((e,), -1, dtype=torch.int32, device=counts.device)
        ref[: active.numel()] = active
        if not torch.equal(slot_to_expert, ref):
            bad.append(f"slot_to_expert mismatch in {int((slot_to_expert != ref).sum())} of {e} entries")
    return bad


def gather_bytes_check(hidden, ids, e, m_cap, topk, slot):
    """The gather-quantize's FP8 bytes at every surviving slot must equal the
    flat quantizer's bytes for that pair's source row."""
    sm = torch.cuda.get_device_capability()[0]
    if sm == 9:
        hq, _ = fso.gemm.quantize_1x128_grouped_gather_sm90(hidden, slot, topk, e, m_cap)
        xq, _ = fso.gemm.quantize_1x128_fp8(hidden)
    else:
        hq, _ = fso.gemm.quantize_1x32_grouped_gather_fp8(hidden, slot, topk, e, m_cap)
        xq, _ = fso.gemm.quantize_1x32_fp8(hidden)
    s = slot.to(torch.int64)
    keep = s >= 0
    if not bool(keep.any()):
        return []
    src = torch.arange(ids.shape[0], device=ids.device,
                       dtype=torch.int64).repeat_interleave(topk)[keep]
    got = hq.view(-1, hidden.shape[1])[s[keep]].view(torch.uint8)
    want = xq[src].view(torch.uint8)
    if not torch.equal(got, want):
        n = int((got != want).any(dim=1).sum())
        return [f"gather-quantize wrote the wrong bytes at {n} of {int(keep.sum())} slots"]
    return []


def run_case(label, ids, e, hidden, with_slots, want_multi):
    m = int(ids.shape[0])
    m_cap = (m + 3) // 4 * 4
    out = fso.gemm.moe_build_routing(ids, e, m_cap, with_slots=with_slots)
    masked, row_map, slot = out[0], out[1], out[2]
    ste = out[3] if with_slots else None
    torch.cuda.synchronize()
    bad = check(ids, e, m_cap, TOPK, masked, row_map, slot, ste)
    bad += gather_bytes_check(hidden, ids, e, m_cap, TOPK, slot)
    n_skip = int(((ids < 0) | (ids >= e)).sum())
    kernel = "multi-CTA" if want_multi else "single-CTA"
    status = "OK" if not bad else "FAIL"
    print(f"  {label:34s} M={m:5d} E={e:4d} pairs={m * TOPK:6d} skipped={n_skip:6d} "
          f"{kernel:10s} {status}")
    for b in bad:
        print(f"      {b}")
    return bad


def main():
    if not torch.cuda.is_available():
        print("SKIP: test_moe_routing_masked_ids needs a CUDA device")
        return 0
    sm = torch.cuda.get_device_capability()[0]
    if sm not in (9, 10, 12):
        print(f"SKIP: test_moe_routing_masked_ids needs sm_90, sm_100/103 or sm_120/121 (device is sm_{sm}x)")
        return 0
    dev = "cuda"
    E = 128
    HIDDEN = 2048
    failures = []
    ran = []

    # Section 1: the single-CTA builder, which every architecture launches.
    # M * topk stays below the multi-CTA threshold.
    ran.append("single-CTA")
    print("single-CTA builder (every architecture):")
    for m in (1, 8, 64, 256):
        g = torch.Generator(device=dev).manual_seed(7 * m)
        hidden = torch.randn(m, HIDDEN, device=dev, dtype=torch.bfloat16, generator=g)
        base = draw(m, E, TOPK, 100 + m, dev)
        # (a) trailing padded rows, both fills the engine uses.
        for fill in (E, -1):
            n_real = max(0, m // 2)
            ids = base.clone()
            ids[n_real:] = fill
            failures += run_case(f"trailing fill={fill}", ids, E, hidden, True, False)
        # (b) an expert-parallel draw: ids from a 2x expert set, remote -> -1.
        wide = draw(m, 2 * E, TOPK, 500 + m, dev)
        ids = torch.where(wide < E, wide, torch.full_like(wide, -1))
        failures += run_case("EP draw (half the experts remote)", ids, E, hidden, True, False)
        # (c) every entry skipped: no expert runs at all.
        failures += run_case("all skipped", torch.full_like(base, -1), E, hidden, True, False)

    # Section 2: the multi-CTA builder, selected on sm_100/103 from 4096 pairs.
    m = 1024
    if sm == 10 and m * TOPK >= MULTI_MIN_PAIRS:
        ran.append("multi-CTA")
        print("multi-CTA builder (sm_100/103):")
        g = torch.Generator(device=dev).manual_seed(99)
        hidden = torch.randn(m, HIDDEN, device=dev, dtype=torch.bfloat16, generator=g)
        base = draw(m, E, TOPK, 999, dev)
        for fill in (E, -1):
            ids = base.clone()
            ids[m // 2:] = fill
            failures += run_case(f"trailing fill={fill}", ids, E, hidden, True, True)
        wide = draw(m, 2 * E, TOPK, 1999, dev)
        failures += run_case("EP draw (half the experts remote)",
                             torch.where(wide < E, wide, torch.full_like(wide, -1)),
                             E, hidden, True, True)
        failures += run_case("all skipped", torch.full_like(base, -1), E, hidden, True, True)
    else:
        print(f"multi-CTA builder: skipped (selected on sm_100/103 only; device is sm_{sm}x)")

    print(f"\nran: {', '.join(ran)}; " + ("ALL PASS" if not failures else f"{len(failures)} FAILURES"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
