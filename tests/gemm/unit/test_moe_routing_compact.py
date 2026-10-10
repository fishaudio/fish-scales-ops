#!/usr/bin/env python3
"""The compact 128-row-block routing builder, `moe_build_routing_compact`, against
its contract (docs/api, the Python wrapper's docstring).

The compact layout gives expert g the ceil(cnt[g] / 128) consecutive 128-row
blocks that start at block blk0[g] (the exclusive prefix of the block counts);
its routed rows are blk0[g]*128 + r, r < cnt[g]. What is checked:

  1. Contract, against a torch reference, on the grid M x G x topk x routing
     (uniform; one hot expert taking every token; trailing rows padded with -1
     and with G; the expert-parallel form, ids drawn over 2G and the upper half
     mapped to -1): masked_m, blk0 (and blk0[G] <= num_blocks, i.e. the
     host-static bound the caller passes is enough), every live pair's row
     inside its expert's rows and distinct from every other pair's, skipped
     pairs -1, row_map and weight_of_slot at live rows, the packed active-expert
     list, the problem shapes and the per-call pointer arrays. Half the cases ask
     for every optional output and half for none, so both the written and the
     skipped form of every output run. Where G < topk the ids are drawn with
     replacement (a token may repeat an expert), which the builder must place
     like any other pair.
  2. Both launch shapes at every pair count (the one-CTA kernel, which the
     default takes up to 4096 pairs, and clusters of 2, 4 and 8 CTAs, which it
     takes above), forced through FSO_ROUTING_COMPACT_PATH in a child process
     (the knob is read once per process), with a profiler record that the
     forced kernel is the one that ran.
  3. Glue reuse (sm_100/103 and sm_120/121, the architectures with the MXFP8
     grouped glue): the gather-quantize (both its pair-space and, on sm_100/103,
     its token-scatter form), the grouped SwiGLU requantize and the combine,
     called on the one-group view of the compact buffer (num_groups = 1,
     m_cap = R_max = num_blocks * 128; CHARTER D1 as amended 2026-10-10) with the
     compact slot_of_flat, against the same ops on the 0.2.0 masked layout:
     identical bytes (FP8 rows and their scale words) at every live row, and a
     bit-identical combine. On sm_100/103 the [num_blocks, 128] view is also
     checked to give the same bytes as the one-group view (one scale atom per
     128 rows either way).
  4. Capture: 64 distinct CUDA-graph captures at M = 4096 in one thread; every
     replay after rewriting topk_ids in place matches the eager result of the
     new ids (the order-free outputs exactly, the rows by the contract), and
     nothing is printed to stderr.
  5. No persistent device memory: the caching allocator's view and the driver's
     free memory are unchanged by a burst of calls once their outputs are
     released.

Exit code 0 when everything passes. Runs on any CUDA device of sm_90 or newer
(the cluster shape needs sm_90); section 3 skips itself elsewhere."""
import os
import subprocess
import sys
import tempfile

import torch

import fish_scales_ops as fso
from fish_scales_ops._arch import sm_major
from fish_scales_ops.gemm.mxfp8 import moe_compact_num_blocks

ROWS = 128
M_GRID = (1, 2, 3, 7, 64, 255, 256, 511, 512, 1000, 4096, 8192)
G_GRID = (1, 8, 128, 256, 1024)
TOPK_GRID = (1, 2, 8)
ROUTINGS = ("uniform", "hot", "pad-1", "padG", "ep")
PS_PAIRS = [(1024, 2048), (2048, 512)]          # two GEMMs, as a layer's FC1 / FC2
PTR_STRIDES = [2048, 128 * 2048 // 32, 1024,      # FC1: A bytes / row, SFA bytes / block, D bytes / row
               512, 128 * 512 // 32, 2 * 2048]    # FC2
PATHS = {  # forced shape -> (largest pair count it takes, kernel-name fragments)
    "cta": (None, (", false>", ", 0>")),
    "cluster2": (None, (", true>", ", 1>")),
    "cluster4": (None, (", true>", ", 1>")),
    "cluster8": (None, (", true>", ", 1>")),
}


def draw(m, g, topk, routing, seed, device):
    """int32 [m, topk] ids for one routing form."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    pool = 2 * g if routing == "ep" else g
    if topk <= pool:
        ids = torch.argsort(torch.rand(m, pool, generator=gen), dim=1)[:, :topk]
    else:  # fewer experts than top-k: with replacement
        ids = torch.randint(0, pool, (m, topk), generator=gen)
    if routing == "hot":
        # expert 0 in every token, the other k - 1 from the rest
        if g > 1 and topk <= g:
            rest = 1 + torch.argsort(torch.rand(m, g - 1, generator=gen), dim=1)[:, : topk - 1]
            ids = torch.cat([torch.zeros(m, 1, dtype=torch.long), rest], dim=1)
        else:
            ids[:, 0] = 0
    elif routing in ("pad-1", "padG"):
        ids[(m + 1) // 2:] = -1 if routing == "pad-1" else g
    elif routing == "ep":
        ids = torch.where(ids < g, ids, torch.full_like(ids, -1))
    return ids.to(device=device, dtype=torch.int32).contiguous()


def check_contract(ids, g, nb_max, out, want_all, topk_w, bases):
    """The list of contract violations of one call's outputs (empty when it holds)."""
    masked_m, blk0, sof, slots, ps, ptrs, row_map, wslot, sched = out
    errs = []
    m, topk = ids.shape
    flat = ids.reshape(-1).long()
    valid = (flat >= 0) & (flat < g)
    cnt = torch.bincount(flat[valid], minlength=g)
    nb = (cnt + ROWS - 1) // ROWS
    blk0_ref = torch.zeros(g + 1, dtype=torch.long, device=ids.device)
    blk0_ref[1:] = torch.cumsum(nb, 0)
    if not torch.equal(masked_m.long(), cnt):
        errs.append("masked_m")
    if not torch.equal(blk0.long(), blk0_ref):
        errs.append("blk0")
    if int(blk0_ref[g]) > nb_max:
        errs.append(f"blocks in use {int(blk0_ref[g])} > num_blocks {nb_max}")
    sof = sof.long()
    if not torch.equal(sof[~valid], torch.full_like(sof[~valid], -1)):
        errs.append("skipped pairs not -1")
    rows = sof[valid]
    e = flat[valid]
    lo = blk0_ref[e] * ROWS
    if not bool(((rows >= lo) & (rows < lo + cnt[e])).all()):
        errs.append("row outside its expert's rows")
    if rows.numel() and torch.unique(rows).numel() != rows.numel():
        errs.append("two pairs share a row")
    pair_idx = torch.nonzero(valid).reshape(-1)
    if want_all:
        if row_map.numel() != nb_max * ROWS or not torch.equal(row_map.long()[rows], pair_idx // topk):
            errs.append("row_map")
        if wslot.numel() != nb_max * ROWS or not torch.equal(wslot[rows], topk_w.reshape(-1)[pair_idx]):
            errs.append("weight_of_slot")
        active = torch.nonzero(cnt > 0).reshape(-1)
        slot_ref = torch.full((g,), -1, dtype=torch.long, device=ids.device)
        slot_ref[: active.numel()] = active
        if not torch.equal(slots.long(), slot_ref):
            errs.append("slot_to_expert")
        # the slot route's work-stealing counter: two int32 words, zero (S10b-1)
        if sched.dtype != torch.int32 or sched.numel() != 2 or bool((sched != 0).any()):
            errs.append("slot_sched")
        ps_ref = torch.stack([torch.stack([cnt, torch.full_like(cnt, n), torch.full_like(cnt, k)], dim=1)
                              for n, k in PS_PAIRS])
        if tuple(ps.shape) != (len(PS_PAIRS), g, 3) or not torch.equal(ps.long(), ps_ref):
            errs.append("problem_shapes")
        b = blk0_ref[:g]
        ptr_ref = []
        for p in range(len(PS_PAIRS)):
            a_row, sfa_blk, d_row = PTR_STRIDES[3 * p: 3 * p + 3]
            ba, bs, bd = (bases[3 * p + j].data_ptr() for j in range(3))
            ptr_ref.append(torch.stack([ba + b * ROWS * a_row, bs + b * sfa_blk, bd + b * ROWS * d_row]))
        if tuple(ptrs.shape) != (len(PS_PAIRS), 3, g) or not torch.equal(ptrs, torch.stack(ptr_ref)):
            errs.append("ptr_arrays")
    else:
        if slots.numel() or ps.numel() or ptrs.numel() or row_map.numel() or wslot.numel() or sched.numel():
            errs.append("an output that was not asked for is not empty")
    return errs


def call(ids, g, want_all, topk_w, bases, num_blocks=None):
    nb_max = num_blocks if num_blocks is not None else moe_compact_num_blocks(ids.shape[0], ids.shape[1], g)
    if want_all:
        out = fso.compat.moe_build_routing_compact(
            ids, g, nb_max, with_slots=True, with_row_map=True, topk_w=topk_w, problem_shapes_for=PS_PAIRS,
            ptr_bases=bases, ptr_strides=PTR_STRIDES)
    else:
        out = fso.compat.moe_build_routing_compact(ids, g, nb_max)
    return nb_max, out


def make_bases(dev):
    return [torch.empty(16 + j, dtype=torch.uint8, device=dev) for j in range(3 * len(PS_PAIRS))]


def contract_grid(dev, cases=None, label="default shapes"):
    """Section 1 (and, with `cases`, the cases a forced shape runs in section 2)."""
    bases = make_bases(dev)
    n = 0
    bad = []
    if cases is None:
        cases = [(m, g, k, r) for m in M_GRID for g in G_GRID for k in TOPK_GRID for r in ROUTINGS]
    for idx, (m, g, k, r) in enumerate(cases):
        ids = draw(m, g, k, r, seed=m * 7919 + g * 31 + k * 3 + ROUTINGS.index(r), device=dev)
        topk_w = torch.rand(m, k, device=dev, dtype=torch.float32)
        want_all = idx % 2 == 0
        nb_max, out = call(ids, g, want_all, topk_w, bases)
        errs = check_contract(ids, g, nb_max, out, want_all, topk_w, bases)
        n += 1
        if errs:
            bad.append(f"M={m} G={g} topk={k} {r} all={want_all}: {', '.join(errs)}")
    torch.cuda.synchronize()
    print(f"  {label}: {n} cases, {len(bad)} failing" + "".join(f"\n    FAIL {b}" for b in bad[:20]))
    return bad


def forced_child(path):
    """Child process of section 2: the contract grid under one forced launch shape, and the kernel that ran."""
    dev = torch.device("cuda")
    limit, frag = PATHS[path]
    cases = [(m, g, k, r) for m in M_GRID for g in (1, 8, 256, 1024) for k in (1, 8) for r in ("uniform", "hot", "ep")
             if limit is None or m * k <= limit]
    bad = contract_grid(dev, cases, label=f"forced {path}")
    ids = draw(2048 if limit is None else limit // 8, 256, 8, "uniform", 1, dev)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        call(ids, 256, True, torch.rand(ids.shape, device=dev), make_bases(dev))
        torch.cuda.synchronize()
    names = sorted({ev.key for ev in prof.key_averages() if "moe_routing_compact" in ev.key})
    frags = frag if isinstance(frag, tuple) else (frag,)
    ran = bool(names) and all(any(f in nm for f in frags) for nm in names)
    print(f"  forced {path}: kernel {names} -> {'ok' if ran else 'WRONG KERNEL'}")
    if not ran:
        bad.append(f"forced {path} ran {names}")
    return 1 if bad else 0


def forced_shapes():
    """Section 2: every launch shape, each in a child process."""
    bad = []
    for path in PATHS:
        env = dict(os.environ, FSO_ROUTING_COMPACT_PATH=path)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--forced-child", path], env=env,
                           capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            bad.append(path)
            sys.stdout.write(r.stderr[-2000:])
    print(f"  forced shapes: {len(PATHS) - len(bad)}/{len(PATHS)} pass" + (f" (failing: {bad})" if bad else ""))
    return bad


def sm100_word(rows, num_kp):
    """Word index of (row, kp) in one group's Sm1xx atom scale slab (fish_scales_ops.gemm.mxfp8._sm100_atom_word_index)."""
    r = rows % 128
    base = (rows // 128) * (num_kp * 128) + (r % 32) * 4 + (r // 32)
    return base[:, None] + torch.arange(num_kp, device=rows.device)[None, :] * 128


def scale_words(sx, slots, m_cap, num_kp):
    """[len(slots), num_kp] scale words of the given rows of a grouped scale handle."""
    g = slots // m_cap
    r = slots % m_cap
    if sm_major() == 10:
        return sx.reshape(sx.shape[0], -1)[g[:, None], sm100_word(r, num_kp)]
    return sx[g[:, None], torch.arange(num_kp, device=slots.device)[None, :], r[:, None]]  # sm_120: [G, K/128, m_cap]


def glue_reuse(dev):
    """Section 3: the slot-addressed glue on the compact layout vs the masked layout."""
    if sm_major() not in (10, 12):
        print(f"  glue reuse: skipped (sm_{sm_major()}x has no MXFP8 grouped glue)")
        return []
    ops = torch.ops.fish_scales_ops
    hidden, inter = 512, 256
    bad = []
    n = 0
    for m, g, k, r in [(1, 256, 8, "uniform"), (7, 8, 2, "pad-1"), (64, 128, 8, "hot"), (255, 256, 8, "ep"),
                       (512, 128, 8, "uniform"), (1000, 256, 8, "padG"), (4096, 256, 8, "hot"),
                       (4096, 128, 8, "uniform")]:
        ids = draw(m, g, k, r, seed=17 * m + g, device=dev)
        x = torch.randn(m, hidden, device=dev, dtype=torch.bfloat16)
        w = torch.rand(m, k, device=dev, dtype=torch.float32)
        m_cap = (m + 3) // 4 * 4
        _mm, _rm, sof_m = fso.compat.moe_build_routing(ids, g, m_cap)
        nb_max = moe_compact_num_blocks(m, k, g)
        out = fso.compat.moe_build_routing_compact(ids, g, nb_max)
        sof_c = out[2]
        live = sof_m >= 0
        if not torch.equal(live, sof_c >= 0):
            bad.append(f"M={m} G={g} {r}: skipped pairs differ")
            continue
        sm_, sc_ = sof_m[live].long(), sof_c[live].long()
        r_max = nb_max * ROWS  # the one-group view of the compact buffer: num_groups = 1, m_cap = R_max
        # gather-quantize
        hq_m, sh_m = ops.quantize_1x32_grouped_gather(x, sof_m, k, g, m_cap)
        hq_c, sh_c = ops.quantize_1x32_grouped_gather(x, sof_c, k, 1, r_max)
        if not torch.equal(hq_m.reshape(-1, hidden).view(torch.uint8)[sm_],
                           hq_c.reshape(-1, hidden).view(torch.uint8)[sc_]):
            bad.append(f"M={m} G={g} {r}: gathered FP8 rows differ")
        if not torch.equal(scale_words(sh_m, sm_, m_cap, hidden // 128), scale_words(sh_c, sc_, r_max, hidden // 128)):
            bad.append(f"M={m} G={g} {r}: gathered scale words differ")
        if sm_major() == 10:
            # sm_100/103: the [num_blocks, 128] view lays out the same bytes (one scale atom per 128 rows either way)
            hq_b, sh_b = ops.quantize_1x32_grouped_gather(x, sof_c, k, nb_max, ROWS)
            if not (torch.equal(hq_b.reshape(-1, hidden).view(torch.uint8)[sc_],
                                hq_c.reshape(-1, hidden).view(torch.uint8)[sc_])
                    and torch.equal(sh_b.reshape(-1)[sm100_word(sc_, hidden // 128)],
                                    sh_c.reshape(-1)[sm100_word(sc_, hidden // 128)])):
                bad.append(f"M={m} G={g} {r}: sm_100 [num_blocks, 128] view differs from the one-group view")
        # grouped SwiGLU requantize (chunked [gate; up] rows)
        gu_m = torch.randn(g, m_cap, 2 * inter, device=dev, dtype=torch.bfloat16)
        gu_c = torch.randn(1, r_max, 2 * inter, device=dev, dtype=torch.bfloat16)
        gu_c.view(-1, 2 * inter)[sc_] = gu_m.view(-1, 2 * inter)[sm_]
        dq_m, sd_m = ops.silu_chunk_mul_quantize_1x32_grouped(gu_m, sof_m, True, False)
        dq_c, sd_c = ops.silu_chunk_mul_quantize_1x32_grouped(gu_c, sof_c, True, False)
        if not torch.equal(dq_m.reshape(-1, inter).view(torch.uint8)[sm_], dq_c.reshape(-1, inter).view(torch.uint8)[sc_]):
            bad.append(f"M={m} G={g} {r}: SwiGLU-requantized rows differ")
        if not torch.equal(scale_words(sd_m, sm_, m_cap, inter // 128), scale_words(sd_c, sc_, r_max, inter // 128)):
            bad.append(f"M={m} G={g} {r}: SwiGLU-requantized scale words differ")
        # combine
        dn_m = torch.randn(g, m_cap, hidden, device=dev, dtype=torch.bfloat16)
        dn_c = torch.randn(1, r_max, hidden, device=dev, dtype=torch.bfloat16)
        dn_c.view(-1, hidden)[sc_] = dn_m.view(-1, hidden)[sm_]
        o_m = ops.moe_combine(dn_m, sof_m, w, None, None, None)
        o_c = ops.moe_combine(dn_c, sof_c, w, None, None, None)
        if not torch.equal(o_m, o_c):
            bad.append(f"M={m} G={g} {r}: combine output differs")
        n += 1
    torch.cuda.synchronize()
    print(f"  glue reuse: {n} cases, {len(bad)} failing" + "".join(f"\n    FAIL {b}" for b in bad))
    return bad


def capture(dev):
    """Section 4: 64 distinct captures at M = 4096, replays on rewritten ids, stderr silent."""
    m, g, k = 4096, 256, 8
    bases = make_bases(dev)
    bad = []
    err_file = tempfile.TemporaryFile()
    sys.stderr.flush()
    saved = os.dup(2)
    os.dup2(err_file.fileno(), 2)
    try:
        ids = draw(m, g, k, "uniform", 5, dev)
        w = torch.rand(m, k, device=dev)
        call(ids, g, True, w, bases)  # eager first, as any capture-safe op is used
        torch.cuda.synchronize()
        graphs = []
        for c in range(64):
            want_all = c % 2 == 0
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                nb_max, out = call(ids, g, want_all, w, bases)
            graphs.append((gr, want_all, nb_max, out))
        for c, (gr, want_all, nb_max, out) in enumerate(graphs):
            new = draw(m, g, k, ("uniform", "hot", "ep", "pad-1")[c % 4], 1000 + c, dev)
            ids.copy_(new)
            gr.replay()
            torch.cuda.synchronize()
            _, ref = call(ids, g, want_all, w, bases)
            errs = check_contract(ids, g, nb_max, out, want_all, w, bases)
            # the order-free outputs equal the eager call's exactly
            for j, name in ((0, "masked_m"), (1, "blk0"), (3, "slot_to_expert"), (4, "problem_shapes"),
                            (5, "ptr_arrays")):
                if not torch.equal(out[j], ref[j]):
                    errs.append(f"{name} != eager")
            if not torch.equal(out[2] >= 0, ref[2] >= 0):
                errs.append("skipped pairs != eager")
            if errs:
                bad.append(f"capture {c}: {', '.join(errs)}")
        torch.cuda.synchronize()
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
    err_file.seek(0)
    text = err_file.read().decode(errors="replace")
    if text.strip():
        bad.append(f"stderr not empty: {text[:500]!r}")
    print(f"  capture: 64 graphs at M={m}, {len(bad)} failing" + "".join(f"\n    FAIL {b}" for b in bad[:10]))
    return bad


def no_persistent_memory(dev):
    """Section 5: a burst of calls leaves neither allocator nor driver memory behind once the outputs are gone."""
    bases = make_bases(dev)
    ids_all = [draw(m, 256, 8, "uniform", m, dev) for m in (1, 64, 1000, 8192)]
    w_all = [torch.rand(i.shape, device=dev) for i in ids_all]
    for ids, w in zip(ids_all, w_all):  # first calls: module load, launch setup
        call(ids, 256, True, w, bases)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    alloc0 = torch.cuda.memory_allocated()
    free0, _ = torch.cuda.mem_get_info()
    for _ in range(20):
        for ids, w in zip(ids_all, w_all):
            call(ids, 256, True, w, bases)
            call(ids, 256, False, w, bases)
    torch.cuda.synchronize()
    alloc1 = torch.cuda.memory_allocated()
    torch.cuda.empty_cache()
    free1, _ = torch.cuda.mem_get_info()
    bad = []
    if alloc1 != alloc0:
        bad.append(f"allocated {alloc0} -> {alloc1}")
    if free1 != free0:
        bad.append(f"driver free memory {free0} -> {free1}")
    print(f"  no persistent memory: allocated {alloc0} -> {alloc1} B, driver free {free0} -> {free1} B"
          + ("" if not bad else "  FAIL"))
    return bad


def refusals(dev):
    """The argument checks of the contract."""
    bad = []
    ids = draw(64, 256, 8, "uniform", 3, dev)
    need = moe_compact_num_blocks(64, 8, 256)
    for label, fn in (
        ("num_blocks below the bound", lambda: fso.compat.moe_build_routing_compact(ids, 256, need - 1)),
        ("num_groups > 1024", lambda: fso.compat.moe_build_routing_compact(ids, 1025)),
        ("ptr_bases without problem shapes", lambda: fso.compat.moe_build_routing_compact(
            ids, 256, ptr_bases=make_bases(dev)[:3], ptr_strides=[1, 1, 1])),
        ("ptr_strides without ptr_bases", lambda: fso.compat.moe_build_routing_compact(
            ids, 256, problem_shapes_for=PS_PAIRS[:1], ptr_strides=[1, 1, 1])),
        ("topk_w of the wrong size", lambda: fso.compat.moe_build_routing_compact(
            ids, 256, topk_w=torch.rand(3, device=dev))),
    ):
        try:
            fn()
            bad.append(f"not refused: {label}")
        except RuntimeError:
            pass
    print(f"  refusals: {5 - len(bad)}/5 refused" + "".join(f"\n    FAIL {b}" for b in bad))
    return bad


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        print("skip: needs a CUDA device of sm_90 or newer")
        return 0
    dev = torch.device("cuda")
    print(f"device sm_{sm_major()}x, torch {torch.__version__}")
    failures = []
    print("1. contract grid")
    failures += contract_grid(dev)
    print("2. forced launch shapes")
    failures += forced_shapes()
    print("3. glue reuse on the compact layout")
    failures += glue_reuse(dev)
    print("4. capture")
    failures += capture(dev)
    print("5. persistent memory")
    failures += no_persistent_memory(dev)
    print("6. refusals")
    failures += refusals(dev)
    print("ALL PASS" if not failures else f"{len(failures)} FAILURES")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--forced-child":
        sys.exit(forced_child(sys.argv[2]))
    sys.exit(main())
