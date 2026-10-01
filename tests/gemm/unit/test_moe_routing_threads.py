#!/usr/bin/env python3
"""Two-thread / two-stream stress test for the `moe_build_routing` builders.

One entry point, two device kernels. `moe_build_routing_kernel` is a single
CTA and is what every architecture launches; on sm_100 / sm_103 and, since
2026-09-28, on sm_120 / sm_121, a routing call with at least
`kRoutingMultiMinPairs` = 4096 routed pairs is handed to
`moe_build_routing_multi_kernel` instead (the selection lives in
`moe_glue_routing_multi_enabled()`, `csrc/gemm/ops/moe_glue.cu`, and
FSO_MOE_ROUTING_MULTI forces either kernel on any architecture). The multi-CTA kernel needs a small global
scratch (per-expert counters plus an arrival counter) which must be zero when
a launch starts and which the launch restores to zero before it exits, so two
launches that can execute at the same time must never be handed the same
scratch buffer.

This file therefore has two sections.

  1. SINGLE-CTA, runs on EVERY architecture. The draws are sized so that
     `M * topk` stays below the multi-CTA threshold, which means every device
     - sm_90, sm_120 and sm_100 alike - takes `moe_build_routing_kernel`.
     It is driven under the same two-thread / two-stream stress as section 2,
     over three kinds of routing draw, and every call asks for the packed
     active-expert list. On sm_90 this is the ONLY coverage the routing
     builder gets, because the multi-CTA kernel is not selected there.
  2. MULTI-CTA, runs where that kernel is selected, i.e. sm_100 / sm_103 and
     sm_120 / sm_121 (or wherever FSO_MOE_ROUTING_MULTI forces it). Skipped
     elsewhere with the reason printed. Before its stress, every draw runs once
     un-threaded under CUPTI and is compared with the single-CTA builder on the
     same draw, run in a child process with FSO_MOE_ROUTING_MULTI=0 (the knob is
     read once per process); the CUPTI records show that this process launched
     the multi-CTA kernel and the child the single-CTA one.

Both sections issue their calls in bursts with no host synchronisation inside
a burst, and the two threads are released into each burst by a barrier,
because that is what makes the two streams actually run their routing kernels
at the same time - a version of this test that validated after every single
call kept the kernels apart and passed even on a build with the bug.

Every result is checked against the definition of the routing contract:

  * masked_m[g] equals the number of routed pairs whose expert is g
    (torch.bincount over topk_ids, which is what the single-CTA builder
    computes as well);
  * every pair's slot lies inside its own expert's block and below that
    expert's count;
  * the slots are a permutation - no two pairs share a slot;
  * row_map at a pair's slot names that pair's source token;
  * slot_to_expert, when asked for, lists exactly the experts with a non-zero
    count, ascending, followed by -1 in every remaining entry.

The three draw kinds exist because the fourth output is easy to get right for
the easy case and wrong for the others. A random draw with many active
experts is satisfied by almost any implementation; a hot-expert draw makes the
-1 tail most of the list; and a dead-prefix draw uses only high expert ids, so
a list built by writing expert g into slot g - which passes every random draw
- fails immediately.

A scratch buffer shared between the two streams shows up immediately as a
count that is too large (both threads' pairs land in one counter) or as a
duplicated slot.

The exit code reflects what actually ran. Section 1 is not optional, so this
file exits non-zero when it did not run or did not pass, and the last line
always names which sections ran and which were skipped. Exit 0 therefore
means the routing builder was exercised on this device, not merely that the
file reached its end.
"""
import os
import re
import subprocess
import sys
import tempfile
import threading

import torch

import fish_scales_ops as fso

E = 128
TOPK = 8
M = 1024         # 1024 * 8 = 8192 routed pairs, well past the 4096 threshold
SINGLE_M = 256   # 256 * 8 = 2048 routed pairs, below it on every arch
BURST = 25       # launches queued per thread before any host synchronisation
ROUNDS = 8       # 8 * 25 = 200 calls per thread
DRAWS = 8

SEC_SINGLE = "single-CTA"
SEC_MULTI = "multi-CTA"


def make_routing(m, e, topk, seed, device, pool=None):
    """`topk` distinct expert ids per token, drawn from `pool` (all E by default)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    cand = torch.arange(e) if pool is None else torch.as_tensor(pool, dtype=torch.int64)
    n = int(cand.numel())
    ids = torch.stack([cand[torch.randperm(n, generator=g)[:topk]] for _ in range(m)])
    return ids.to(device, torch.int32)


def slot_list_reference(counts, e):
    """The packed active-expert list the routing kernel must emit."""
    active = torch.nonzero(counts > 0).flatten().to(torch.int32)
    ref = torch.full((e,), -1, dtype=torch.int32, device=counts.device)
    ref[: active.numel()] = active
    return ref


def check(ids, masked, row_map, slot, e, m_cap, topk, slot_to_expert=None):
    """Return a list of failure strings; empty means the result is valid."""
    bad = []
    flat = ids.flatten().to(torch.int64)
    counts = torch.bincount(flat, minlength=e).to(torch.int32)
    if slot_to_expert is not None:
        ref = slot_list_reference(counts, e)
        if not torch.equal(slot_to_expert, ref):
            n = int((slot_to_expert != ref).sum())
            bad.append(f"slot_to_expert mismatch in {n} of {e} entries")
    if not torch.equal(masked, counts):
        d = (masked.to(torch.int64) - counts.to(torch.int64))
        bad.append(f"masked_m mismatch, max |delta| {int(d.abs().max())}")
    s = slot.to(torch.int64)
    grp = s // m_cap
    rank = s % m_cap
    if not torch.equal(grp, flat):
        bad.append("slot is not inside the pair's own expert block")
    if bool((rank >= counts[flat].to(torch.int64)).any()):
        bad.append("slot rank is at or beyond the expert's row count")
    if int(torch.unique(s).numel()) != int(s.numel()):
        bad.append("slots are not a permutation (duplicate slot)")
    src = torch.arange(ids.shape[0], device=ids.device,
                       dtype=torch.int32).repeat_interleave(topk)
    if not torch.equal(row_map.to(torch.int64)[s].to(torch.int32), src):
        bad.append("row_map at the slot does not name the source token")
    return bad


def single_cta_draws(base_seed, device):
    """Six draws for one thread: two random, two hot-expert, two dead-prefix.

    Each entry is (label, topk_ids, num_experts). The hot draws route every
    token inside a set of 8 or 9 experts, so the packed list is 8 or 9 ids
    followed by a -1 tail that is most of the list. The dead-prefix draws use
    only high expert ids, so the list must start at 100 and at 200.
    """
    m = SINGLE_M
    return [
        ("random E=128",
         make_routing(m, 128, TOPK, base_seed + 1, device), 128),
        ("random E=256",
         make_routing(m, 256, TOPK, base_seed + 2, device), 256),
        ("hot 8 of 128",
         make_routing(m, 128, TOPK, base_seed + 3, device,
                      pool=[3, 11, 12, 40, 41, 77, 90, 127]), 128),
        ("hot 9 of 256",
         make_routing(m, 256, TOPK, base_seed + 4, device,
                      pool=[5, 6, 7, 8, 100, 101, 200, 254, 255]), 256),
        ("dead prefix >=100 of 128",
         make_routing(m, 128, TOPK, base_seed + 5, device,
                      pool=list(range(100, 128))), 128),
        ("dead prefix >=200 of 256",
         make_routing(m, 256, TOPK, base_seed + 6, device,
                      pool=list(range(200, 256))), 256),
    ]


def anchor(draws, m_cap, topk):
    """One un-threaded call per draw, with and without the fourth output.

    Running each draw with `with_slots=False` as well is not redundant: the
    packed list is produced by the same final pass that publishes `masked_m`,
    so this is what shows that asking for it does not disturb the three
    outputs that were there before.
    """
    failures = []
    for label, ids, e in draws:
        ms, rm, sl, se = fso.compat.moe_build_routing(ids, e, m_cap, with_slots=True)
        bad = check(ids, ms, rm, sl, e, m_cap, topk, se)
        ms3, rm3, sl3 = fso.compat.moe_build_routing(ids, e, m_cap)
        bad += [f"three-output form: {b}"
                for b in check(ids, ms3, rm3, sl3, e, m_cap, topk)]
        n_active = int((ms > 0).sum())
        if bad:
            failures.append(f"{label}: " + "; ".join(bad))
            print(f"  FAIL anchor {label:26s} " + "; ".join(bad))
        else:
            print(f"  anchor {label:26s} E={e:>3} {n_active:>3} active experts  OK")
    return failures


def stress(draws, m_cap, topk, rounds, burst):
    """Two host threads, one CUDA stream each, `rounds` bursts of `burst` calls.

    `draws` maps thread id -> list of (label, topk_ids, num_experts). Returns
    the list of failure strings; empty means every call satisfied the contract.
    """
    dev = draws[0][0][1].device
    failures = []
    lock = threading.Lock()
    gate = threading.Barrier(2)
    # Release gate: thread 0 queues a long dummy kernel on its own stream and
    # records `go` behind it; both worker streams then wait on `go` before
    # their burst, so the two bursts are released at the same instant and their
    # routing kernels genuinely run at the same time. Without this the two
    # threads' launches interleave in Python but the kernels still land one
    # after the other, and the test passes even on a build with the bug.
    trig = torch.cuda.Stream(device=dev)
    filler = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    go = [torch.cuda.Event() for _ in range(rounds)]

    def worker(tid):
        mine = draws[tid]
        stream = torch.cuda.Stream(device=dev)
        try:
            with torch.cuda.stream(stream):
                # One eager call first: the scratch pool allocates on the first
                # call of each host thread and never again.
                _, ids0, e0 = mine[0]
                fso.compat.moe_build_routing(ids0, e0, m_cap, with_slots=True)
                stream.synchronize()
                for rnd in range(rounds):
                    if tid == 0:
                        with torch.cuda.stream(trig):
                            x = filler
                            for _ in range(6):
                                x = x @ filler * 1e-3
                            go[rnd].record(trig)
                    gate.wait()
                    stream.wait_event(go[rnd])
                    out = []
                    for b in range(burst):
                        label, ids, e = mine[(rnd * burst + b) % len(mine)]
                        # with_slots on every call: the fourth output is
                        # produced by the same final pass that publishes
                        # masked_m, so a scratch buffer shared between the two
                        # streams corrupts it too and the check catches it.
                        out.append((label, ids, e) + tuple(fso.compat.moe_build_routing(
                            ids, e, m_cap, with_slots=True)))
                    stream.synchronize()
                    for b, (label, ids, e, masked, row_map, slot, sl_exp) in enumerate(out):
                        bad = check(ids, masked, row_map, slot, e, m_cap, topk,
                                    sl_exp)
                        if bad:
                            with lock:
                                failures.append(
                                    f"thread {tid} round {rnd} call {b} "
                                    f"({label}): " + "; ".join(bad))
                            gate.abort()
                            return
                    gate.wait()
            stream.synchronize()
        except threading.BrokenBarrierError:
            return
        except Exception as exc:  # noqa: BLE001 - report, do not hide
            with lock:
                failures.append(f"thread {tid} raised {exc!r}")
            gate.abort()

    threads = [threading.Thread(target=worker, args=(t,)) for t in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    torch.cuda.synchronize()
    return failures


def multi_cta_selected(major):
    """(selected, reason): the mirror of moe_glue_routing_multi_enabled(). A set
    FSO_MOE_ROUTING_MULTI forces the multi-CTA builder on or off on any
    architecture (read like the C++ side's atoi); unset, sm_100/103 and sm_120/121
    select it."""
    env = os.environ.get("FSO_MOE_ROUTING_MULTI", "")
    if env != "":
        lead = re.match(r"\s*[+-]?\d+", env)
        on = lead is not None and int(lead.group(0)) != 0
        return on, f"FSO_MOE_ROUTING_MULTI={env} forces it {'on' if on else 'off'}"
    if major in (10, 12):
        return True, f"selected on sm_{major}x"
    return False, f"not selected on sm_{major}x; sm_100/103 and sm_120/121 select it"


def routing_call(ids, e, m_cap, with_slots):
    """One moe_build_routing call under CUPTI: the host copies of (masked_m,
    row_map, slot_of_flat, slot_to_expert or None) and the names of the routing
    kernels it launched."""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = fso.compat.moe_build_routing(ids, e, m_cap, with_slots=with_slots)
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
    """What two builders must agree on for one draw: masked_m, the packed expert
    list, and, expert by expert, the multiset of source tokens its rows name. The
    row a pair gets inside its expert's block is a free permutation that the two
    kernels assign in different orders, so slot_of_flat is not compared."""
    (ma, ra, _sa, ea), (mb, rb, _sb, eb) = a, b
    if not torch.equal(ma, mb):
        return ["masked_m differs"]
    bad = []
    if not torch.equal(ea, eb):
        bad.append("slot_to_expert differs")
    valid = torch.arange(m_cap).unsqueeze(0) < ma.to(torch.int64).unsqueeze(1)
    big = torch.iinfo(torch.int32).max
    ta = torch.where(valid, ra.view(e, m_cap), big).sort(dim=1).values
    tb = torch.where(valid, rb.view(e, m_cap), big).sort(dim=1).values
    if not torch.equal(ta, tb):
        bad.append("some expert's rows name a different set of source tokens")
    return bad


def multi_cta_draws(device):
    """The multi-CTA section's draws, thread id -> [(label, topk_ids, E)]."""
    return {t: [(f"random E={E} seed{i}",
                 make_routing(M, E, TOPK, 20260917 + 1000 * t + i, device), E)
                for i in range(DRAWS)] for t in (0, 1)}


def multi_cta_anchor(draws, reference, m_cap, topk):
    """Each multi-CTA draw once, un-threaded: the routing contract, the kernel the
    call launched (the multi-CTA kernel, not the single-CTA fallback a call takes
    when no scratch buffer is available), and agreement with the single-CTA
    builder's result on the same draw (FSO_MOE_ROUTING_MULTI=0, child process)."""
    failures = []
    for (label, ids, e), (rout, rnames) in zip(draws, reference):
        out, names = routing_call(ids, e, m_cap, True)
        masked, row_map, slot, ste = out
        bad = check(ids.cpu(), masked, row_map, slot, e, m_cap, topk, ste)
        if not names or not all("moe_build_routing_multi" in n for n in names):
            bad.append(f"this process launched {names or 'no recorded routing kernel'}, not the multi-CTA kernel")
        if not rnames or any("moe_build_routing_multi" in n for n in rnames):
            bad.append(f"the FSO_MOE_ROUTING_MULTI=0 process launched {rnames or 'no recorded routing kernel'}")
        bad += [f"vs single-CTA: {b}" for b in same_routing(e, m_cap, out, rout)]
        if bad:
            failures.append(f"anchor {label}: " + "; ".join(bad))
            print(f"  FAIL anchor {label:26s} " + "; ".join(bad))
        else:
            print(f"  anchor {label:26s} multi-CTA kernel, contract OK, same routing as the single-CTA builder")
    return failures


def section_single_cta(dev):
    """The arch-independent section. Returns the list of failure strings."""
    m_cap = (SINGLE_M + 3) // 4 * 4
    draws = {t: single_cta_draws(20260922 + 1000 * t, dev) for t in (0, 1)}
    failures = anchor(draws[0], m_cap, TOPK)
    failures += stress(draws, m_cap, TOPK, ROUNDS, BURST)
    if not failures:
        print(f"  2 threads x 2 streams x {ROUNDS * BURST} calls at M={SINGLE_M} "
              f"({SINGLE_M * TOPK} routed pairs, bursts of {BURST}, "
              f"6 draw kinds)  OK")
    return failures


def section_multi_cta(dev, reference):
    """The multi-CTA section (sm_100/103, sm_120/121). `reference` is the
    single-CTA builder's result on every draw, thread 0's draws first. Returns the
    list of failure strings."""
    m_cap = (M + 3) // 4 * 4
    draws = multi_cta_draws(dev)
    failures = multi_cta_anchor(draws[0] + draws[1], reference, m_cap, TOPK)
    failures += stress(draws, m_cap, TOPK, ROUNDS, BURST)
    if not failures:
        print(f"  2 threads x 2 streams x {ROUNDS * BURST} calls at M={M} "
              f"({M * TOPK} routed pairs, bursts of {BURST})  OK")
    return failures


def main():
    ran, skipped, failures = [], [], []
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device — the arch-independent section cannot run")
        print(f"ran: none, skipped: {SEC_SINGLE} (no CUDA device), "
              f"{SEC_MULTI} (no CUDA device)")
        return 1

    dev = torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(0)

    # The multi-CTA section's single-CTA reference comes first. It runs in a child
    # process (FSO_MOE_ROUTING_MULTI is read once per process), and on a card in
    # exclusive-process compute mode that child can only open a CUDA context while
    # this process holds none, i.e. before anything below touches the device.
    multi, why = multi_cta_selected(major)
    if multi:
        cpu_draws = multi_cta_draws("cpu")
        reference = single_cta_reference([(ids, e, (M + 3) // 4 * 4, True)
                                          for _, ids, e in cpu_draws[0] + cpu_draws[1]])

    print(f"== {SEC_SINGLE} routing builder (every arch; device is "
          f"sm_{major}{minor}) ==")
    bad = section_single_cta(dev)
    ran.append(SEC_SINGLE)
    if bad:
        failures += [f"{SEC_SINGLE}: {b}" for b in bad]

    print(f"== {SEC_MULTI} routing builder (sm_100 / sm_103, sm_120 / sm_121; {why}) ==")
    if multi:
        bad = section_multi_cta(dev, reference)
        ran.append(SEC_MULTI)
        if bad:
            failures += [f"{SEC_MULTI}: {b}" for b in bad]
    else:
        print(f"  SKIP: {why}")
        skipped.append(f"{SEC_MULTI} ({why})")

    for f in failures[:10]:
        print("FAIL " + f)
    summary = (f"ran: {', '.join(ran)}, "
               f"skipped: {', '.join(skipped) if skipped else 'none'}")
    if failures:
        print("FAILED")
        print(summary)
        return 1
    print("ALL OK")
    print(summary)
    # Exit 0 only when the arch-independent section ran and passed; an arch
    # that skips the multi-CTA section has still tested the kernel it uses.
    return 0 if SEC_SINGLE in ran else 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--single-cta-reference":
        sys.exit(write_single_cta_reference(sys.argv[2], sys.argv[3]))
    sys.exit(main())
