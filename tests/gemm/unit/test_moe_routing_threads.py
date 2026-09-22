#!/usr/bin/env python3
"""Two-thread / two-stream stress test for the sm_100 multi-CTA routing builder.

`moe_build_routing` picks a multi-CTA kernel on sm_100 from 4096 routed pairs
upwards. That kernel needs a small global scratch (per-expert counters plus an
arrival counter) which must be zero when a launch starts and which the launch
restores to zero before it exits. Two launches that can execute at the same
time must therefore never be handed the same scratch buffer.

This test drives exactly that situation: two host threads, each with its own
CUDA stream, each issuing `iters` routing calls with its own routing draws, so
the two kernels overlap on the device. The calls are issued in bursts with no
host synchronisation inside a burst, and the two threads are released into
each burst by a barrier, because that is what makes the two streams actually
run their routing kernels at the same time - a version of this test that
validated after every single call kept the kernels apart and passed even on a
build with the bug. Every result is checked against the definition of the
routing contract:

  * masked_m[g] equals the number of routed pairs whose expert is g
    (torch.bincount over topk_ids, which is what the single-CTA builder
    computes as well);
  * every pair's slot lies inside its own expert's block and below that
    expert's count;
  * the slots are a permutation - no two pairs share a slot;
  * row_map at a pair's slot names that pair's source token;
  * slot_to_expert, when asked for, lists exactly the experts with a non-zero
    count, ascending, followed by -1 in every remaining entry.

A scratch buffer shared between the two streams shows up immediately as a
count that is too large (both threads' pairs land in one counter) or as a
duplicated slot.

Skips on anything that is not sm_100 / sm_103, where the single-CTA builder is
used and no scratch exists.
"""
import sys
import threading

import torch

import fish_scales_ops as fso

E = 128
TOPK = 8
M = 1024         # 1024 * 8 = 8192 routed pairs, well past the 4096 threshold
BURST = 25       # launches queued per thread before any host synchronisation
ROUNDS = 8       # 8 * 25 = 200 calls per thread
DRAWS = 8


def make_routing(m, e, topk, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack([torch.randperm(e, generator=g)[:topk] for _ in range(m)])
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


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    major = torch.cuda.get_device_capability(0)[0]
    if major != 10:
        print(f"SKIP: multi-CTA routing builder is sm_100/sm_103 only "
              f"(device is sm_{major}x)")
        return 0

    dev = torch.device("cuda")
    m_cap = (M + 3) // 4 * 4
    draws = {t: [make_routing(M, E, TOPK, 20260917 + 1000 * t + i, dev)
                 for i in range(DRAWS)] for t in (0, 1)}

    # Sanity anchor: the same input below the multi-CTA threshold goes through
    # the single-CTA builder and must satisfy the same contract.
    small = make_routing(256, E, TOPK, 7, dev)
    ms, rm, sl, se = fso.gemm.moe_build_routing(
        small, E, (256 + 3) // 4 * 4, with_slots=True)
    bad = check(small, ms, rm, sl, E, (256 + 3) // 4 * 4, TOPK, se)
    if bad:
        print("FAIL single-CTA anchor: " + "; ".join(bad))
        return 1
    print(f"  single-CTA anchor (M=256, {256 * TOPK} pairs, with_slots)  OK")

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
    go = [torch.cuda.Event() for _ in range(ROUNDS)]

    def worker(tid):
        stream = torch.cuda.Stream(device=dev)
        try:
            with torch.cuda.stream(stream):
                # One eager call first: the scratch pool allocates on the first
                # call of each host thread and never again.
                fso.gemm.moe_build_routing(draws[tid][0], E, m_cap, with_slots=True)
                stream.synchronize()
                for rnd in range(ROUNDS):
                    if tid == 0:
                        with torch.cuda.stream(trig):
                            x = filler
                            for _ in range(6):
                                x = x @ filler * 1e-3
                            go[rnd].record(trig)
                    gate.wait()
                    stream.wait_event(go[rnd])
                    out = []
                    for b in range(BURST):
                        ids = draws[tid][(rnd * BURST + b) % DRAWS]
                        # with_slots on every call: the fourth output is
                        # produced by the same final pass that publishes
                        # masked_m, so a scratch buffer shared between the two
                        # streams corrupts it too and the check catches it.
                        out.append((ids,) + tuple(fso.gemm.moe_build_routing(
                            ids, E, m_cap, with_slots=True)))
                    stream.synchronize()
                    for b, (ids, masked, row_map, slot, sl_exp) in enumerate(out):
                        bad = check(ids, masked, row_map, slot, E, m_cap, TOPK,
                                    sl_exp)
                        if bad:
                            with lock:
                                failures.append(
                                    f"thread {tid} round {rnd} call {b}: "
                                    + "; ".join(bad))
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

    if failures:
        for f in failures[:10]:
            print("FAIL " + f)
        return 1
    print(f"  2 threads x 2 streams x {ROUNDS * BURST} calls at M={M} "
          f"({M * TOPK} routed pairs, bursts of {BURST})  OK")
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
