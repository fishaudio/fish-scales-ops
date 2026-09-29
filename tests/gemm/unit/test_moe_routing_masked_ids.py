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
4096 routed pairs) on sm_100/103 and, since 2026-09-28, on sm_120/121, which is
what moe_glue_routing_multi_enabled() selects when FSO_MOE_ROUTING_MULTI is
unset (the knob forces either builder on any architecture, and this file follows
it). In the multi-CTA section every case is also compared with the single-CTA
builder on the same draw, run in a child process with FSO_MOE_ROUTING_MULTI=0
because the knob is read once per process, and a CUPTI record of each side shows
which kernel it launched: the multi-CTA kernel here, the single-CTA one in the
child. The last line names which sections ran."""
import os
import re
import subprocess
import sys
import tempfile

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


def multi_cta_selected(sm):
    """(selected, reason): the mirror of moe_glue_routing_multi_enabled(). A set
    FSO_MOE_ROUTING_MULTI forces the multi-CTA builder on or off on any
    architecture (read like the C++ side's atoi); unset, sm_100/103 and sm_120/121
    select it."""
    env = os.environ.get("FSO_MOE_ROUTING_MULTI", "")
    if env != "":
        lead = re.match(r"\s*[+-]?\d+", env)
        on = lead is not None and int(lead.group(0)) != 0
        return on, f"FSO_MOE_ROUTING_MULTI={env} forces it {'on' if on else 'off'}"
    if sm in (10, 12):
        return True, f"selected on sm_{sm}x"
    return False, f"not selected on sm_{sm}x; sm_100/103 and sm_120/121 select it"


def multi_cta_cases(m, e):
    """The multi-CTA section's draws, built on the host (see main() for why)."""
    base = draw(m, e, TOPK, 999, "cpu")
    cases = []
    for fill in (e, -1):
        ids = base.clone()
        ids[m // 2:] = fill
        cases.append((f"trailing fill={fill}", ids))
    wide = draw(m, 2 * e, TOPK, 1999, "cpu")
    cases.append(("EP draw (half the experts remote)",
                  torch.where(wide < e, wide, torch.full_like(wide, -1))))
    cases.append(("all skipped", torch.full_like(base, -1)))
    return cases


def routing_call(ids, e, m_cap, with_slots):
    """One moe_build_routing call under CUPTI: the host copies of (masked_m,
    row_map, slot_of_flat, slot_to_expert or None) and the names of the routing
    kernels it launched."""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = fso.gemm.moe_build_routing(ids, e, m_cap, with_slots=with_slots)
        torch.cuda.synchronize()
    names = sorted({ev.key for ev in prof.key_averages() if "moe_build_routing" in ev.key})
    ste = out[3].cpu() if with_slots else None
    return (out[0].cpu(), out[1].cpu(), out[2].cpu(), ste), names


def single_cta_reference(cases):
    """routing_call() of every (ids, e, m_cap, with_slots) case in a child process
    run with FSO_MOE_ROUTING_MULTI=0, i.e. through the single-CTA builder. Call it
    before this process opens its CUDA context: on a card in exclusive-process
    compute mode the child cannot open one while the parent holds one."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "cases.pt"), os.path.join(d, "reference.pt")
        torch.save([(ids.cpu(), e, m_cap, ws) for ids, e, m_cap, ws in cases], src)
        env = dict(os.environ, FSO_MOE_ROUTING_MULTI="0")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--single-cta-reference", src, dst],
                           env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("single-CTA reference process failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
        return torch.load(dst)


def write_single_cta_reference(src, dst):
    """The child side of single_cta_reference()."""
    torch.save([routing_call(ids.cuda(), e, m_cap, ws) for ids, e, m_cap, ws in torch.load(src)], dst)
    return 0


def same_routing(e, m_cap, a, b):
    """What two builders must agree on for one draw: masked_m, which pairs are
    skipped, the packed expert list, and, expert by expert, the multiset of source
    tokens its rows name. The row a pair gets inside its expert's block is a free
    permutation that the two kernels assign in different orders, so slot_of_flat
    is compared only through its -1 pattern."""
    (ma, ra, sa, ea), (mb, rb, sb, eb) = a, b
    if not torch.equal(ma, mb):
        return ["masked_m differs"]
    bad = []
    if not torch.equal(sa < 0, sb < 0):
        bad.append("the skipped pairs differ")
    if (ea is None) != (eb is None) or (ea is not None and not torch.equal(ea, eb)):
        bad.append("slot_to_expert differs")
    valid = torch.arange(m_cap).unsqueeze(0) < ma.to(torch.int64).unsqueeze(1)
    big = torch.iinfo(torch.int32).max
    ta = torch.where(valid, ra.view(e, m_cap), big).sort(dim=1).values
    tb = torch.where(valid, rb.view(e, m_cap), big).sort(dim=1).values
    if not torch.equal(ta, tb):
        bad.append("some expert's rows name a different set of source tokens")
    return bad


def compare_with_single_cta(cases, reference, e, m_cap):
    """Every multi-CTA case against the single-CTA builder's result on the same
    draw, with the kernel each side launched checked from its CUPTI record (a
    routing call falls back to the single-CTA kernel when no scratch buffer is
    available, so the name is what shows the production path ran)."""
    failures = []
    for (label, ids), (rout, rnames) in zip(cases, reference):
        out, names = routing_call(ids, e, m_cap, True)
        bad = []
        if not names or not all("moe_build_routing_multi" in n for n in names):
            bad.append(f"this process launched {names or 'no recorded routing kernel'}, not the multi-CTA kernel")
        if not rnames or any("moe_build_routing_multi" in n for n in rnames):
            bad.append(f"the FSO_MOE_ROUTING_MULTI=0 process launched {rnames or 'no recorded routing kernel'}")
        bad += same_routing(e, m_cap, out, rout)
        print(f"  vs single-CTA builder {label:34s} {'OK' if not bad else 'FAIL'}")
        for b in bad:
            print(f"      {b}")
        failures += [f"{label} vs single-CTA: {b}" for b in bad]
    return failures


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

    # The multi-CTA section's draws and their single-CTA reference come first. The
    # reference runs in a child process (FSO_MOE_ROUTING_MULTI is read once per
    # process), and on a card in exclusive-process compute mode that child can only
    # open a CUDA context while this process holds none, i.e. before anything below
    # touches the device.
    m_multi = 1024
    m_cap_multi = (m_multi + 3) // 4 * 4
    multi, why = multi_cta_selected(sm)
    run_multi = multi and m_multi * TOPK >= MULTI_MIN_PAIRS
    if run_multi:
        cases = multi_cta_cases(m_multi, E)
        reference = single_cta_reference([(ids, E, m_cap_multi, True) for _, ids in cases])

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

    # Section 2: the multi-CTA builder, where the dispatcher selects it from 4096
    # routed pairs: sm_100/103, and sm_120/121 since 2026-09-28.
    if run_multi:
        ran.append("multi-CTA")
        print(f"multi-CTA builder ({why}):")
        g = torch.Generator(device=dev).manual_seed(99)
        hidden = torch.randn(m_multi, HIDDEN, device=dev, dtype=torch.bfloat16, generator=g)
        cases = [(label, ids.to(dev)) for label, ids in cases]
        for label, ids in cases:
            failures += run_case(label, ids, E, hidden, True, True)
        failures += compare_with_single_cta(cases, reference, E, m_cap_multi)
    else:
        print(f"multi-CTA builder: skipped ({why})")

    print(f"\nran: {', '.join(ran)}; " + ("ALL PASS" if not failures else f"{len(failures)} FAILURES"))
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--single-cta-reference":
        sys.exit(write_single_cta_reference(sys.argv[2], sys.argv[3]))
    sys.exit(main())
