"""sm_100/103: the grouped MXFP8 GEMMs on the compact activation layout.

The compact layout (fso 0.2.1, charter D1) replaces the masked slab's
``G x align(M, 4)`` rows by ONE group of ``R = 128 * B_alloc`` rows in which
expert g owns the 128-row blocks starting at ``blk0[g]``. Only the GEMMs see two
index spaces (weights by expert, activations by compact row), so what this file
checks is that the two compact ops

* ``linear_mxfp8_grouped_compact`` (plain FC1 / FC2) and
* ``linear_mxfp8_grouped_compact_swiglu`` (fused-SwiGLU FC1)

write, at every live row, exactly the bytes the masked ops
(``linear_mxfp8_grouped_masked`` / ``_masked_swiglu``) write for the same
routing, and that they write nothing outside the live rows (on both routes:
the slot route's stores are bounded by each expert's routed rows since
S10b-1). Both routes, every
pointer-array tile ``FSO_FORCE_TILE`` can name, Families B and C, the M grid of
the brief, uniform / one-hot / padded / expert-parallel routing.

Cases (each runs in its own process, so the process-static knobs
``FSO_GROUPED_SLOT`` and ``FSO_FORCE_TILE`` apply to one child only):

* ``rule``      the default route rule, every M, every routing, all three GEMMs,
                plus the prep-path variant (no routing-supplied arrays);
* ``ptrarray``  ``FSO_GROUPED_SLOT=0``: the pointer-array route at every M;
* ``slot``      ``FSO_GROUPED_SLOT=force``: the slot route wherever it is legal
                (m_cap <= 64);
* ``tile:<code>`` every unfused ``FSO_FORCE_TILE`` code and the three fused
                widths, with the slot route off;
* ``chain``     a Python chain (routing reference -> gather -> compact FC1 ->
                compact FC2 -> ``moe_combine`` on the one-group dn) equals
                ``fso.moe.layer`` bit for bit, eager and inside a captured graph
                (the routing rewritten in place between replays);
* ``ref020``    the same layer outputs from the 0.2.0 release wheel
                (``FSO_BASE020`` names its install directory); ``chain``
                compares against them when the file exists;
* ``arena``     40 layer-weight sets x 20 buckets captured into graphs: the
                weight arena binds 2 blocks per layer and nothing per capture;
* ``slack``     the end-of-buffer 2-SM tile with and without the one block of
                allocation slack, for ``compute-sanitizer --tool memcheck``;
* ``sched``     ``FSO_GROUPED_SLOT=force``: the slot route's work-stealing
                counter (``slot_sched``): the routing op returns it zero, every
                slot GEMM leaves it zero, and one counter serves consecutive
                slot GEMMs eagerly and inside one captured graph (every call
                writes its own poisoned output, so a tile a later call skipped
                would show), with the bytes of a call that passes none.

The routing is built in torch, exactly to the contract of
``moe_build_routing_compact`` (S3): masked_m, blk0 (exclusive prefix of
ceil(cnt / 128)), slot_of_flat, the packed active-expert list, the per-GEMM
problem shapes and the per-call pointer arrays.

Run: ``PYTHONPATH=python python tests/gemm/unit/test_mxfp8_compact_sm100.py
[--cases rule,ptrarray,...] [--quick]``. The parent never creates a CUDA
context; each child is the only process on the card while it runs, so an
Exclusive_Process card needs no mode switch. ``FSO_TEST_GPU_LOCK=<file>`` makes
the parent take that lock (``flock -w 1500``) around every child. On any
architecture other than sm_100/103 the children exit 0 without testing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

FAMILIES = {
    "B": dict(E=128, H=2048, I=768, topk=8),
    "C": dict(E=256, H=2048, I=512, topk=8),
}
M_GRID = [1, 2, 3, 4, 7, 8, 16, 32, 40, 64, 96, 128, 256, 512, 1024, 2048, 4096, 8192]
ROUTINGS = ["uniform", "hot", "padded", "ep"]

# Every unfused code the sm_100 grouped force table wires (grouped_dispatch.cuh).
UNFUSED_TILES = [
    "128,64,1", "128,128,1", "128,192,1", "128,256,1", "256,128,2", "256,256,2", "128,128,3", "128,256,3",
    "256,128,4", "256,256,4", "128,64,5", "128,128,5", "128,192,5", "128,256,5", "256,128,6", "256,256,6",
    "128,128,7", "128,256,7", "256,128,8", "256,256,8", "128,128,9", "128,192,9", "128,256,9", "128,192,11",
    "128,256,11",
]
FUSED_TILES = ["128,128,1", "128,192,1", "128,256,1"]
TILE_M = [1, 40, 256, 2048]


# ----------------------------------------------------------------------------
# child side
# ----------------------------------------------------------------------------

def _child_imports():
    global torch, fso, ops, mx
    import torch  # noqa: F811
    import fish_scales_ops as fso  # noqa: F811
    import fish_scales_ops.gemm.mxfp8 as mx  # noqa: F811
    ops = torch.ops.fish_scales_ops


def make_topk(M, E, topk, kind, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "ep":
        ids = torch.rand(M, 2 * E, generator=g, device="cuda").topk(topk, dim=1).indices
        ids = torch.where(ids >= E, torch.full_like(ids, -1), ids)
        return ids.int().contiguous()
    r = torch.rand(M, E, generator=g, device="cuda")
    if kind == "hot":
        r[:, 0] = 2.0  # expert 0 is in every token's top-k
    ids = r.topk(topk, dim=1).indices.int()
    if kind == "padded":
        rows = torch.arange(M, device="cuda")
        ids[rows % 3 == 2] = -1   # graph-bucket padding rows carry -1 ...
        ids[rows % 5 == 4] = E    # ... or the id E
        if M == 1:
            ids[0, -1] = -1
    return ids.contiguous()


def b_max_of(P, G):
    return max(1, (P + 127 * min(G, P)) // 128)


class Routing:
    """The compact routing of one topk_ids tensor, built in torch to S3's contract,
    and the masked routing with the SAME rank of every pair inside its expert."""

    def __init__(self, ids, G):
        M, topk = ids.shape
        flat = ids.reshape(-1).long()
        P = flat.numel()
        valid = (flat >= 0) & (flat < G)
        key = torch.where(valid, flat, torch.full_like(flat, G))
        order = torch.sort(key, stable=True).indices
        cnt_all = torch.bincount(key, minlength=G + 1)
        cnt = cnt_all[:G]
        start = torch.zeros(G + 2, dtype=torch.long, device=flat.device)
        start[1:] = torch.cumsum(cnt_all, 0)
        rank = torch.empty(P, dtype=torch.long, device=flat.device)
        rank[order] = torch.arange(P, device=flat.device) - start[key[order]]
        nb = (cnt + 127) // 128
        blk0 = torch.zeros(G + 1, dtype=torch.long, device=flat.device)
        blk0[1:] = torch.cumsum(nb, 0)
        e = flat.clamp(0, G - 1)
        self.M, self.topk, self.G, self.P = M, topk, G, P
        self.m_cap = (M + 3) // 4 * 4
        self.b_max = b_max_of(P, G)
        assert int(blk0[G]) <= self.b_max, "B_max bound violated"
        self.b_alloc = self.b_max + 1
        self.R = 128 * self.b_alloc
        self.valid = valid
        self.cnt = cnt.int().contiguous()
        self.blk0 = blk0.int().contiguous()
        self.blk0_l = blk0
        self.slot_c = torch.where(valid, blk0[e] * 128 + rank, torch.full_like(flat, -1)).int().contiguous()
        self.slot_m = torch.where(valid, e * self.m_cap + rank, torch.full_like(flat, -1)).int().contiguous()
        active = torch.nonzero(cnt > 0).flatten().int()
        slots = torch.full((G,), -1, dtype=torch.int32, device=flat.device)
        slots[: active.numel()] = active
        self.slots = slots
        self.expected_m = max(1, (M * topk + G - 1) // G)
        self.max_active = min(M * topk, G)

    def rewrite(self, other):
        """Copy another routing of the same (M, G) into this one's tensors in place."""
        for name in ("cnt", "blk0", "slot_c", "slot_m", "slots", "valid"):
            getattr(self, name).copy_(getattr(other, name))
        self.blk0_l.copy_(other.blk0_l)

    def problem_shapes(self, N, K):
        n = torch.full_like(self.cnt, N)
        k = torch.full_like(self.cnt, K)
        return torch.stack([self.cnt, n, k], dim=1).contiguous()

    def ptr_arrays(self, a, sa, d, K, d_row_bytes):
        b = self.blk0_l[: self.G]
        return torch.stack([
            a.data_ptr() + b * 128 * K,
            sa.data_ptr() + b * (K // 128) * 512,
            d.data_ptr() + b * 128 * d_row_bytes,
        ]).contiguous()

    def masked_shapes(self, N, K):
        n = torch.full_like(self.cnt, N)
        k = torch.full_like(self.cnt, K)
        return torch.stack([self.cnt.clamp(max=self.m_cap), n, k], dim=1).contiguous()


def atom_words(rows, num_kp):
    return mx._sm100_atom_word_index(rows.long(), num_kp)  # [len(rows), num_kp]


_W_CACHE = {}


def weights(fam):
    """bf16 weights quantized for the grouped GEMMs (FC1 interleaved when the fused
    FC1 exists for the shape, as prepare_experts does), cached per process."""
    if fam in _W_CACHE:
        return _W_CACHE[fam]
    f = FAMILIES[fam]
    E, H, I = f["E"], f["H"], f["I"]
    torch.manual_seed(1234 + E)
    w13 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) / (H ** 0.5)
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) / (I ** 0.5)
    inter = bool(mx.mxfp8_grouped_swiglu_available(2 * I, H))
    w13q, sw13 = mx.quantize_moe_weights_1x32_fp8(w13, w13_interleave=inter)
    w2q, sw2 = mx.quantize_moe_weights_1x32_fp8(w2)
    del w13, w2
    _W_CACHE[fam] = (w13q, sw13, w2q, sw2, inter)
    return _W_CACHE[fam]


def slot_taken(rt, N, K, fused, compact):
    """The route the op of the given layout lands on: the compact ops have their own rule
    (run fso_0.2.1_20261010/S10b-3a), the masked ops keep 0.2.0's."""
    return bool(ops.mxfp8_grouped_slot_possible(rt.m_cap, N, K, rt.G, rt.max_active, fused, compact))


def allowed_rows(rt, slot_route):
    """Rows of the compact buffer a GEMM may write: the live rows, on both routes (the
    slot route's stores are bounded by each expert's routed rows, run
    fso_0.2.1_20261010/S10b-1; before that it wrote an active expert's first m_cap)."""
    del slot_route  # the same rows on both routes
    allowed = torch.zeros(rt.R, dtype=torch.bool, device="cuda")
    allowed[rt.slot_c[rt.valid].long()] = True
    return allowed


def gather_both(x, rt, K):
    """The activation operand in both layouts, by the same gather kernel."""
    hq_m, sh_m = ops.quantize_1x32_grouped_gather(x, rt.slot_m, rt.topk, rt.G, rt.m_cap, True)
    hq_c, sh_c = ops.quantize_1x32_grouped_gather(x, rt.slot_c, rt.topk, 1, rt.R, True)
    return hq_m, sh_m, hq_c.view(rt.R, K), sh_c.view(-1)


def run_cell(fam, M, kind, gemm, results, *, prep=False, seed=0):
    """One (family, M, routing, GEMM) cell: compact vs masked at the live rows, and the
    bounded-store check. Appends one result dict."""
    f = FAMILIES[fam]
    E, H, I, topk = f["E"], f["H"], f["I"], f["topk"]
    w13q, sw13, w2q, sw2, inter = weights(fam)
    ids = make_topk(M, E, topk, kind, seed=M * 1009 + E * 17 + seed)
    rt = Routing(ids, E)
    torch.manual_seed(M * 7 + E)
    if gemm == "fc2":
        N, K, w, sw = H, I, w2q, sw2
    else:
        N, K, w, sw = 2 * I, H, w13q, sw13
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * (0.1 if gemm != "fc2" else 1.0)
    hq_m, sh_m, hq_c, sh_c = gather_both(x, rt, K)
    fused = gemm == "fc1_fused"
    if fused and not inter:
        return
    # Each layout's op is handed the routing tensors of the route ITS rule takes: the
    # compact rule may send a GEMM to the slot route where 0.2.0's masked rule keeps the
    # pointer-array cascade (both routes write the same bytes).
    slot_route = slot_taken(rt, N, K, fused, compact=True)
    slot_route_m = slot_taken(rt, N, K, fused, compact=False)
    if os.environ.get("FSO_FORCE_TILE") and not fused:
        slot_route = slot_route_m = False  # the unfused force table precedes the route rule
    ps_m = None if slot_route_m else rt.masked_shapes(N, K)
    slots_m = rt.slots if slot_route_m else None
    slots = rt.slots if slot_route else None
    rec = dict(fam=fam, M=M, routing=kind, gemm=gemm, route="slot" if slot_route else "ptr",
               route_masked="slot" if slot_route_m else "ptr",
               prep=prep, live=int(rt.valid.sum()), R=rt.R)
    if not fused:
        out_m = ops.linear_mxfp8_grouped_masked(hq_m, w, sh_m, sw, rt.cnt, rt.expected_m, rt.max_active,
                                                slots_m, ps_m)
        d = torch.empty(rt.R, N, device="cuda", dtype=torch.bfloat16)
        d.view(torch.int16).fill_(-1)
        ps = pa = None
        if not prep and not slot_route:
            ps = rt.problem_shapes(N, K)
            pa = rt.ptr_arrays(hq_c, sh_c, d, K, 2 * N)
        ops.linear_mxfp8_grouped_compact(hq_c, w, sh_c, sw, rt.cnt, rt.blk0, rt.m_cap, rt.expected_m,
                                         rt.max_active, slots, ps, pa, d)
        torch.cuda.synchronize()
        live_m = out_m.view(-1, N)[rt.slot_m[rt.valid].long()]
        live_c = d[rt.slot_c[rt.valid].long()]
        rec["equal"] = bool(torch.equal(live_m.view(torch.int16), live_c.view(torch.int16)))
        rec["finite"] = bool(torch.isfinite(live_c.float()).all())
        untouched = ~allowed_rows(rt, slot_route)
        rec["bounded"] = bool((d[untouched].view(torch.int16) == -1).all())
    else:
        dq_m, sd_m = ops.linear_mxfp8_grouped_masked_swiglu(hq_m, w, sh_m, sw, rt.cnt, rt.expected_m,
                                                            rt.max_active, slots_m, ps_m)
        dq = torch.empty(rt.R, I, device="cuda", dtype=torch.float8_e4m3fn)
        dq.view(torch.uint8).fill_(0xFF)
        sd = torch.full((rt.R * (I // 128),), -1, device="cuda", dtype=torch.int32)
        ps = pa = None
        if not prep and not slot_route:
            ps = rt.problem_shapes(N, K)
            pa = rt.ptr_arrays(hq_c, sh_c, dq, K, I)
        ops.linear_mxfp8_grouped_compact_swiglu(hq_c, w, sh_c, sw, rt.cnt, rt.blk0, rt.m_cap, rt.expected_m,
                                                rt.max_active, slots, ps, pa, dq, sd)
        torch.cuda.synchronize()
        num_kp = I // 128
        lv = rt.valid
        rows_m = rt.slot_m[lv].long()
        rows_c = rt.slot_c[lv].long()
        eq_h = torch.equal(dq_m.view(-1, I).view(torch.uint8)[rows_m], dq.view(torch.uint8)[rows_c])
        m_pad = (rt.m_cap + 127) // 128 * 128
        g_of = rows_m // rt.m_cap
        r_in = rows_m % rt.m_cap
        wm = g_of[:, None] * (m_pad * num_kp) + atom_words(r_in, num_kp)
        wc = atom_words(rows_c, num_kp)
        eq_s = torch.equal(sd_m.view(-1)[wm.flatten()], sd[wc.flatten()])
        rec["equal"] = bool(eq_h and eq_s)
        rec["equal_h"], rec["equal_sf"] = bool(eq_h), bool(eq_s)
        untouched_rows = torch.nonzero(~allowed_rows(rt, slot_route)).flatten()
        ok_h = bool((dq.view(torch.uint8)[untouched_rows] == 0xFF).all())
        ok_s = bool((sd[atom_words(untouched_rows, num_kp).flatten()] == -1).all()) if untouched_rows.numel() else True
        rec["bounded"] = ok_h and ok_s
        rec["finite"] = True
    rec["ok"] = rec["equal"] and rec["bounded"] and rec["finite"]
    results.append(rec)


def child_cells(case, fams, m_list, kinds, gemms, *, prep_ms=()):
    results = []
    for fam in fams:
        for M in m_list:
            for kind in kinds:
                for gemm in gemms:
                    run_cell(fam, M, kind, gemm, results)
                    if M in prep_ms:
                        run_cell(fam, M, kind, gemm, results, prep=True)
            torch.cuda.empty_cache()
        _W_CACHE.pop(fam, None)
        torch.cuda.empty_cache()
    return results


# ---- sched -----------------------------------------------------------------

def child_sched():
    """The compact slot route's work-stealing counter (run fso_0.2.1_20261010/S10b-1)."""
    import fish_scales_ops as fso
    results = []
    reps = 3
    for fam in ("B", "C"):
        f = FAMILIES[fam]
        E, H, I, topk = f["E"], f["H"], f["I"], f["topk"]
        w13q, sw13, w2q, sw2, inter = weights(fam)
        for M in (1, 8, 32, 64):
            for kind in ("uniform", "hot"):
                ids = make_topk(M, E, topk, kind, seed=M * 31 + E)
                rt = Routing(ids, E)
                out = fso.compat.moe_build_routing_compact(ids, E, rt.b_max, with_slots=True)
                sched = out[8]
                rec = dict(case="sched", fam=fam, M=M, routing=kind)
                rec["routing_zero"] = bool(sched.dtype == torch.int32 and sched.numel() == 2
                                           and not bool((sched != 0).any()))
                torch.manual_seed(M * 7 + E)
                x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 0.1
                _, _, hq, sh = gather_both(x, rt, H)
                live = rt.slot_c[rt.valid].long()

                def fc1(dq, sd, s=None):
                    ops.linear_mxfp8_grouped_compact_swiglu(hq, w13q, sh, sw13, rt.cnt, rt.blk0, rt.m_cap,
                                                            rt.expected_m, rt.max_active, rt.slots, None, None,
                                                            dq, sd, s)

                def fc2(a, sa, d, s=None):
                    ops.linear_mxfp8_grouped_compact(a, w2q, sa, sw2, rt.cnt, rt.blk0, rt.m_cap, rt.expected_m,
                                                     rt.max_active, rt.slots, None, None, d, s)

                def new_fc1_out():
                    dq = torch.empty(rt.R, I, device="cuda", dtype=torch.float8_e4m3fn)
                    dq.view(torch.uint8).fill_(0xFF)
                    sd = torch.full((rt.R * (I // 128),), -1, device="cuda", dtype=torch.int32)
                    return dq, sd

                def new_fc2_out():
                    d = torch.empty(rt.R, H, device="cuda", dtype=torch.bfloat16)
                    d.view(torch.int16).fill_(-1)
                    return d

                num_kp = I // 128
                wl = atom_words(live, num_kp).flatten()
                dq_ref, sd_ref = new_fc1_out()
                fc1(dq_ref, sd_ref)  # the op's own counter
                d_ref = new_fc2_out()
                fc2(dq_ref, sd_ref, d_ref)
                torch.cuda.synchronize()

                def same1(dq, sd):
                    return bool(torch.equal(dq.view(torch.uint8)[live], dq_ref.view(torch.uint8)[live])
                                and torch.equal(sd[wl], sd_ref[wl]))

                def same2(d):
                    return bool(torch.equal(d[live].view(torch.int16), d_ref[live].view(torch.int16)))

                ok_eager = True
                for _ in range(reps):
                    dq, sd = new_fc1_out()
                    fc1(dq, sd, sched)
                    d = new_fc2_out()
                    fc2(dq_ref, sd_ref, d, sched)
                    torch.cuda.synchronize()
                    ok_eager = ok_eager and same1(dq, sd) and same2(d) and not bool((sched != 0).any())
                bufs1 = [new_fc1_out() for _ in range(reps)]
                bufs2 = [new_fc2_out() for _ in range(reps)]
                g = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.graph(g, stream=s):
                    for i in range(reps):
                        fc1(bufs1[i][0], bufs1[i][1], sched)
                        fc2(dq_ref, sd_ref, bufs2[i], sched)
                torch.cuda.current_stream().wait_stream(s)
                ok_graph = True
                for _replay in range(2):
                    for dq, sd in bufs1:
                        dq.view(torch.uint8).fill_(0xFF)
                        sd.fill_(-1)
                    for d in bufs2:
                        d.view(torch.int16).fill_(-1)
                    g.replay()
                    torch.cuda.synchronize()
                    ok_graph = ok_graph and all(same1(dq, sd) for dq, sd in bufs1) and all(same2(d) for d in bufs2)
                    ok_graph = ok_graph and not bool((sched != 0).any())
                del g
                # Without a counter the op takes a zeroed one from the caching allocator; inside a capture that is a
                # captured fill, so every replay starts from zero again.
                b1, b2 = new_fc1_out(), new_fc2_out()
                g2 = torch.cuda.CUDAGraph()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.graph(g2, stream=s):
                    fc1(b1[0], b1[1])
                    fc2(dq_ref, sd_ref, b2)
                torch.cuda.current_stream().wait_stream(s)
                ok_own = True
                for _replay in range(2):
                    b1[0].view(torch.uint8).fill_(0xFF)
                    b1[1].fill_(-1)
                    b2.view(torch.int16).fill_(-1)
                    g2.replay()
                    torch.cuda.synchronize()
                    ok_own = ok_own and same1(b1[0], b1[1]) and same2(b2)
                del g2
                rec.update(eager=ok_eager, graph=ok_graph, graph_own_counter=ok_own, live=int(live.numel()))
                rec["ok"] = rec["routing_zero"] and ok_eager and ok_graph and ok_own
                results.append(rec)
        _W_CACHE.pop(fam, None)
        torch.cuda.empty_cache()
    return results


# ---- chain -----------------------------------------------------------------

CHAIN_M = [1, 7, 64, 513, 4096]


def chain_inputs(fam, M, seed):
    f = FAMILIES[fam]
    E, H = f["E"], f["H"]
    torch.manual_seed(seed)
    hidden = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    ids = make_topk(M, E, f["topk"], "uniform", seed=seed + 1)
    tw = torch.softmax(torch.randn(M, f["topk"], device="cuda"), dim=1).float().contiguous()
    return hidden, ids, tw


def chain_experts(fam):
    f = FAMILIES[fam]
    E, H, I = f["E"], f["H"], f["I"]
    torch.manual_seed(4321 + E)
    w13 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) / (H ** 0.5)
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) / (I ** 0.5)
    return fso.moe.prepare_experts(w13, w2, format="mxfp8")


class CompactChain:
    """gather -> compact FC1 (fused or plain + SwiGLU) -> compact FC2 -> moe_combine, with
    every buffer allocated up front (the layer's discipline: the routing op needs the
    base addresses)."""

    def __init__(self, ex, M, rt):
        self.ex = ex
        E, H, I = ex.num_experts, ex.hidden, ex.inter
        self.plan = mx._mxfp8_layer_plan(M, rt.topk, E, 2 * I, H, I, ex.w13_interleaved, False)
        self.rt = rt
        R = rt.R
        dev = "cuda"
        self.hq = torch.empty(R, H, device=dev, dtype=torch.float8_e4m3fn)
        self.sh = torch.empty(R * (H // 128), device=dev, dtype=torch.int32)
        self.dq = torch.empty(R, I, device=dev, dtype=torch.float8_e4m3fn)
        self.sd = torch.empty(R * (I // 128), device=dev, dtype=torch.int32)
        self.gu = None if self.plan.fused_fc1 else torch.empty(R, 2 * I, device=dev, dtype=torch.bfloat16)
        self.dn = torch.empty(R, H, device=dev, dtype=torch.bfloat16)
        self.refresh()

    def refresh(self):
        """(Re)build the per-call arrays of the routing in self.rt into static tensors."""
        rt, p, H, I = self.rt, self.plan, self.ex.hidden, self.ex.inter
        self.slots = rt.slots if p.want_slots else None
        if p.want_ps_fc1:
            d1, row1 = (self.dq, I) if p.fused_fc1 else (self.gu, 4 * I)
            new_ps1, new_pa1 = rt.problem_shapes(2 * I, H), rt.ptr_arrays(self.hq, self.sh, d1, H, row1)
            if getattr(self, "ps1", None) is None:
                self.ps1, self.pa1 = new_ps1, new_pa1
            else:
                self.ps1.copy_(new_ps1)
                self.pa1.copy_(new_pa1)
        else:
            self.ps1 = self.pa1 = None
        if p.want_ps_fc2:
            new_ps2, new_pa2 = rt.problem_shapes(H, I), rt.ptr_arrays(self.dq, self.sd, self.dn, I, 2 * H)
            if getattr(self, "ps2", None) is None:
                self.ps2, self.pa2 = new_ps2, new_pa2
            else:
                self.ps2.copy_(new_ps2)
                self.pa2.copy_(new_pa2)
        else:
            self.ps2 = self.pa2 = None

    def __call__(self, hidden, tw):
        rt, p, ex = self.rt, self.plan, self.ex
        H, I, R = ex.hidden, ex.inter, rt.R
        hq, sh = ops.quantize_1x32_grouped_gather(hidden, rt.slot_c, rt.topk, 1, R, True)
        self.hq.copy_(hq.view(R, H))
        self.sh.copy_(sh.view(-1))
        if p.fused_fc1:
            ops.linear_mxfp8_grouped_compact_swiglu(self.hq, ex.w13, self.sh, ex.sw13, rt.cnt, rt.blk0, p.m_cap,
                                                    p.expected_m, p.max_active_groups, self.slots, self.ps1,
                                                    self.pa1, self.dq, self.sd)
        else:
            ops.linear_mxfp8_grouped_compact(self.hq, ex.w13, self.sh, ex.sw13, rt.cnt, rt.blk0, p.m_cap,
                                             p.expected_m, p.max_active_groups, self.slots, self.ps1, self.pa1,
                                             self.gu)
            dq, sd = ops.silu_chunk_mul_quantize_1x32_grouped(self.gu.view(1, R, 2 * I), rt.slot_c, True,
                                                              ex.w13_interleaved)
            self.dq.copy_(dq.view(R, I))
            self.sd.copy_(sd.view(-1))
        ops.linear_mxfp8_grouped_compact(self.dq, ex.w2, self.sd, ex.sw2, rt.cnt, rt.blk0, p.m_cap,
                                         p.expected_m, p.max_active_groups, self.slots, self.ps2, self.pa2,
                                         self.dn)
        return ops.moe_combine(self.dn.view(1, R, H), rt.slot_c, tw)


def child_ref020(out_path):
    """Layer outputs of whatever fish_scales_ops is on the path (the 0.2.0 wheel)."""
    refs = {}
    for fam in ("B", "C"):
        ex = chain_experts(fam)
        for M in CHAIN_M:
            for seed in (11, 12):
                hidden, ids, tw = chain_inputs(fam, M, seed=M * 31 + seed)
                refs[f"{fam}/{M}/{seed}"] = fso.moe.layer(hidden, ex, ids, tw).cpu()
        del ex
        torch.cuda.empty_cache()
    torch.save(refs, out_path)
    return [dict(case="ref020", n=len(refs), ok=True, version=getattr(fso, "__version__", "?"))]


def child_chain(ref_path):
    results = []
    ref020 = torch.load(ref_path) if ref_path and os.path.exists(ref_path) else None
    for fam in ("B", "C"):
        ex = chain_experts(fam)
        for M in CHAIN_M:
            hidden, ids, tw = chain_inputs(fam, M, seed=M * 31 + 11)
            ref = fso.moe.layer(hidden, ex, ids, tw)
            rt = Routing(ids, ex.num_experts)
            chain = CompactChain(ex, M, rt)
            out = chain(hidden, tw)
            torch.cuda.synchronize()
            rec = dict(fam=fam, M=M, mode="eager", fused_fc1=chain.plan.fused_fc1,
                       slots=chain.plan.want_slots, eq_layer=bool(torch.equal(out, ref)))
            if ref020 is not None:
                rec["eq_020"] = bool(torch.equal(out.cpu(), ref020[f"{fam}/{M}/11"]))
            rec["ok"] = rec["eq_layer"] and rec.get("eq_020", True)
            results.append(rec)
            # Captured: the routing tensors and every buffer are static; the graph holds
            # the gather, both GEMMs and the combine. Then a second routing of the same
            # M is written into the same tensors and the graph replayed.
            s_hidden, s_tw = hidden.clone(), tw.clone()
            for _ in range(2):
                chain(s_hidden, s_tw)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                cap = chain(s_hidden, s_tw)
            cap.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            rec2 = dict(fam=fam, M=M, mode="captured", eq_layer=bool(torch.equal(cap, ref)))
            hidden2, ids2, tw2 = chain_inputs(fam, M, seed=M * 31 + 12)
            ref2 = fso.moe.layer(hidden2, ex, ids2, tw2)
            rt.rewrite(Routing(ids2, ex.num_experts))
            chain.refresh()
            s_hidden.copy_(hidden2)
            s_tw.copy_(tw2)
            g.replay()
            torch.cuda.synchronize()
            rec2["eq_layer_rerouted"] = bool(torch.equal(cap, ref2))
            if ref020 is not None:
                rec2["eq_020_rerouted"] = bool(torch.equal(cap.cpu(), ref020[f"{fam}/{M}/12"]))
            rec2["ok"] = rec2["eq_layer"] and rec2["eq_layer_rerouted"] and rec2.get("eq_020_rerouted", True)
            results.append(rec2)
            del g, chain
            torch.cuda.empty_cache()
        del ex
        torch.cuda.empty_cache()
    return results


# ---- arena -----------------------------------------------------------------

ARENA_LAYERS = 40
ARENA_BUCKETS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 3072, 4096, 8192]


def child_arena():
    """40 layers x 2 GEMMs over Family C's expert count at a small (N, K), 20 bucket
    captures. The weight arena prints one line per block bound (FSO_PRINT_TILE_INFO=1
    is set by the parent); the parent counts them."""
    G, H, I, topk = 256, 256, 256, 8
    torch.manual_seed(5)
    layers = []
    for _ in range(ARENA_LAYERS):
        w1 = (torch.randn(G, 2 * I, H, device="cuda", dtype=torch.bfloat16) / 16)
        w2 = (torch.randn(G, H, I, device="cuda", dtype=torch.bfloat16) / 16)
        layers.append(mx.quantize_moe_weights_1x32_fp8(w1) + mx.quantize_moe_weights_1x32_fp8(w2))
    graphs, checks = [], []
    for M in ARENA_BUCKETS:
        ids = make_topk(M, G, topk, "uniform", seed=M)
        rt = Routing(ids, G)
        x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)

        def body():
            hq, sh = ops.quantize_1x32_grouped_gather(x, rt.slot_c, topk, 1, rt.R, True)
            a, sa = hq.view(rt.R, H), sh.view(-1)
            d2 = None
            for (w1q, s1, w2q, s2) in layers:
                d1 = torch.empty(rt.R, 2 * I, device="cuda", dtype=torch.bfloat16)
                ops.linear_mxfp8_grouped_compact(a, w1q, sa, s1, rt.cnt, rt.blk0, rt.m_cap, rt.expected_m,
                                                 rt.max_active, None, rt.problem_shapes(2 * I, H),
                                                 rt.ptr_arrays(a, sa, d1, H, 4 * I), d1)
                h, shh = ops.silu_chunk_mul_quantize_1x32_grouped(d1.view(1, rt.R, 2 * I), rt.slot_c, True, False)
                h, shh = h.view(rt.R, I), shh.view(-1)
                d2 = torch.empty(rt.R, H, device="cuda", dtype=torch.bfloat16)
                ops.linear_mxfp8_grouped_compact(h, w2q, shh, s2, rt.cnt, rt.blk0, rt.m_cap, rt.expected_m,
                                                 rt.max_active, None, rt.problem_shapes(H, I),
                                                 rt.ptr_arrays(h, shh, d2, I, 2 * H), d2)
            return d2

        eager = body()  # eager warm-up of this bucket (as a serving boot does)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            cap = body()
        g.replay()
        torch.cuda.synchronize()
        live = rt.slot_c[rt.valid].long()
        checks.append(bool(torch.equal(cap[live], eager[live])))
        graphs.append(g)
        print(f"[arena] bucket M={M} captured", file=sys.stderr, flush=True)
    return [dict(case="arena", buckets=len(ARENA_BUCKETS), layers=ARENA_LAYERS, replay_equal=all(checks),
                 ok=all(checks))]


def child_arena_masked():
    """The 0.2.0 contrast to child_arena: the same 40 x 20 capture loop on the MASKED ops
    with routing-supplied problem shapes (their static path). Each capture's activations
    come from the graph's pool, so every (capture, GEMM) pins one more StaticArena block;
    run with a small FSO_GROUPED_ARG_POOL_MB the child aborts at the bucket where the
    arena fills. The parent records that bucket from the per-bucket stderr lines."""
    G, H, I, topk = 256, 256, 256, 8
    torch.manual_seed(5)
    layers = []
    for _ in range(ARENA_LAYERS):
        w1 = (torch.randn(G, 2 * I, H, device="cuda", dtype=torch.bfloat16) / 16)
        w2 = (torch.randn(G, H, I, device="cuda", dtype=torch.bfloat16) / 16)
        layers.append(mx.quantize_moe_weights_1x32_fp8(w1) + mx.quantize_moe_weights_1x32_fp8(w2))
    graphs = []
    for M in ARENA_BUCKETS:
        ids = make_topk(M, G, topk, "uniform", seed=M)
        rt = Routing(ids, G)
        x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)

        def body():
            a, sa = ops.quantize_1x32_grouped_gather(x, rt.slot_m, topk, G, rt.m_cap, True)
            d2 = None
            for (w1q, s1, w2q, s2) in layers:
                d1 = ops.linear_mxfp8_grouped_masked(a, w1q, sa, s1, rt.cnt, rt.expected_m, rt.max_active, None,
                                                     rt.masked_shapes(2 * I, H))
                h, shh = ops.silu_chunk_mul_quantize_1x32_grouped(d1, rt.slot_m, True, False)
                d2 = ops.linear_mxfp8_grouped_masked(h, w2q, shh, s2, rt.cnt, rt.expected_m, rt.max_active, None,
                                                     rt.masked_shapes(H, I))
            return d2

        body()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        g.replay()
        torch.cuda.synchronize()
        graphs.append(g)
        print(f"[arena] bucket M={M} captured", file=sys.stderr, flush=True)
    return [dict(case="arena_masked", buckets=len(ARENA_BUCKETS), ok=True)]


# ---- slack -----------------------------------------------------------------

def child_slack(slack_blocks, tile):
    """The worst case for the end of the buffer: every active expert holds cnt = 1 mod 128,
    so the blocks in use equal B_max and the LAST expert's 256-row tile reads one block
    past its region. With slack_blocks = 1 that block exists; with 0 it does not."""
    G, H, N = 8, 256, 256
    counts = [129, 1, 1, 1, 1, 1, 1, 1]
    flat = torch.cat([torch.full((c,), g, dtype=torch.int32) for g, c in enumerate(counts)]).cuda()
    ids = flat.view(-1, 1).contiguous()  # topk = 1
    rt = Routing(ids, G)
    assert int(rt.blk0[G]) == rt.b_max, (int(rt.blk0[G]), rt.b_max)
    b_alloc = rt.b_max + slack_blocks
    R = 128 * b_alloc
    torch.manual_seed(3)
    w = torch.randn(G, N, H, device="cuda", dtype=torch.bfloat16) / 16
    wq, sw = mx.quantize_moe_weights_1x32_fp8(w)
    x = torch.randn(ids.shape[0], H, device="cuda", dtype=torch.bfloat16)
    hq, sh = ops.quantize_1x32_grouped_gather(x, rt.slot_c, 1, 1, R, True)
    # Fresh, exactly sized allocations (run with PYTORCH_NO_CUDA_MEMORY_CACHING=1 so each
    # tensor is its own cudaMalloc and the sanitizer sees its end).
    a = hq.view(R, H).clone()
    sa = sh.view(-1).clone()
    d = torch.empty(R, N, device="cuda", dtype=torch.bfloat16)
    rt_ps = rt.problem_shapes(N, H)
    rt_pa = rt.ptr_arrays(a, sa, d, H, 2 * N)
    ops.linear_mxfp8_grouped_compact(a, wq, sa, sw, rt.cnt, rt.blk0, rt.m_cap, rt.expected_m, rt.max_active,
                                     None, rt_ps, rt_pa, d)
    torch.cuda.synchronize()
    m_ref = rt.m_cap
    hq_m, sh_m = ops.quantize_1x32_grouped_gather(x, rt.slot_m, 1, G, m_ref, True)
    out_m = ops.linear_mxfp8_grouped_masked(hq_m, wq, sh_m, sw, rt.cnt, rt.expected_m, rt.max_active, None,
                                            rt.masked_shapes(N, H))
    torch.cuda.synchronize()
    eq = torch.equal(out_m.view(-1, N)[rt.slot_m.long()], d[rt.slot_c.long()])
    return [dict(case="slack", slack_blocks=slack_blocks, tile=tile, b_max=rt.b_max, b_alloc=b_alloc,
                 equal=bool(eq), ok=bool(eq))]


class _GuardedBuffer:
    """A device buffer that ends exactly at the end of a mapped 2 MB VMM page whose
    successor page is reserved but NOT mapped, so a read one byte past its end faults
    (cudaErrorIllegalAddress) instead of returning whatever the allocator put there.
    compute-sanitizer's memcheck does not see the TMA (cp.async.bulk.tensor) reads of
    the GEMM, which is why the slack check uses this instead."""

    def __init__(self, nbytes):
        from cuda.bindings import driver as cu
        self.cu = cu
        dev = torch.cuda.current_device()
        prop = cu.CUmemAllocationProp()
        prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = dev
        err, gran = cu.cuMemGetAllocationGranularity(
            prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM)
        assert err == cu.CUresult.CUDA_SUCCESS, err
        size = (nbytes + gran - 1) // gran * gran
        err, va = cu.cuMemAddressReserve(size + gran, 0, 0, 0)
        assert err == cu.CUresult.CUDA_SUCCESS, err
        err, handle = cu.cuMemCreate(size, prop, 0)
        assert err == cu.CUresult.CUDA_SUCCESS, err
        (err,) = cu.cuMemMap(va, size, 0, handle, 0)
        assert err == cu.CUresult.CUDA_SUCCESS, err
        acc = cu.CUmemAccessDesc()
        acc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        acc.location.id = dev
        acc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        (err,) = cu.cuMemSetAccess(va, size, [acc], 1)
        assert err == cu.CUresult.CUDA_SUCCESS, err
        self.va, self.size, self.handle = va, size, handle
        self.ptr = int(va) + size - nbytes  # ends exactly at the last mapped byte
        self.nbytes = nbytes

    def tensor(self, dtype, numel):
        itemsize = torch.empty((), dtype=dtype).element_size()
        assert numel * itemsize == self.nbytes
        typestr = {torch.int32: "<i4", torch.uint8: "|u1"}[dtype]
        holder = type("H", (), {})()
        holder.__cuda_array_interface__ = dict(shape=(numel,), typestr=typestr, data=(self.ptr, False), version=3)
        return torch.as_tensor(holder, device="cuda")


def child_slack_guarded(slack_blocks, tile, which):
    """The end-of-buffer case of child_slack with `which` ('sa' = the activation scale
    atoms, 'a' = the activation data, 'd' = the output) placed against an unmapped page."""
    G, H, N = 8, 256, 256
    # The pointer-array case: expert 0 holds 129 rows, so the last expert's 256-row
    # tile starts in block B_max - 1. The slot case (FSO_GROUPED_SLOT=force, m_cap 8):
    # one row per expert, the last expert's token tile is its own first 64 rows.
    counts = [1] * 8 if tile == "slot" else [129, 1, 1, 1, 1, 1, 1, 1]
    flat = torch.cat([torch.full((c,), g, dtype=torch.int32) for g, c in enumerate(counts)]).cuda()
    ids = flat.view(-1, 1).contiguous()
    rt = Routing(ids, G)
    assert int(rt.blk0[G]) == rt.b_max
    b_alloc = rt.b_max + slack_blocks
    R = 128 * b_alloc
    torch.manual_seed(3)
    w = torch.randn(G, N, H, device="cuda", dtype=torch.bfloat16) / 16
    wq, sw = mx.quantize_moe_weights_1x32_fp8(w)
    x = torch.randn(ids.shape[0], H, device="cuda", dtype=torch.bfloat16)
    hq, sh = ops.quantize_1x32_grouped_gather(x, rt.slot_c, 1, 1, R, True)
    a = hq.view(R, H)
    sa = sh.view(-1)
    d = torch.empty(R, N, device="cuda", dtype=torch.bfloat16)
    if which == "sa":
        gb = _GuardedBuffer(sa.numel() * 4)
        g_sa = gb.tensor(torch.int32, sa.numel())
        g_sa.copy_(sa)
        sa = g_sa
    elif which == "a":
        gb = _GuardedBuffer(a.numel())
        g_a = gb.tensor(torch.uint8, a.numel())
        g_a.copy_(a.reshape(-1).view(torch.uint8))
        a = g_a.view(torch.float8_e4m3fn).view(R, H)
    elif which == "d":
        gb = _GuardedBuffer(d.numel() * 2)
        d = gb.tensor(torch.uint8, d.numel() * 2).view(torch.bfloat16).view(R, N)
    torch.cuda.synchronize()
    slots = rt.slots if tile == "slot" else None
    ops.linear_mxfp8_grouped_compact(a, wq, sa, sw, rt.cnt, rt.blk0, rt.m_cap, rt.expected_m, rt.max_active,
                                     slots, rt.problem_shapes(N, H), rt.ptr_arrays(a, sa, d, H, 2 * N), d)
    torch.cuda.synchronize()  # a read past the guarded buffer faults here
    hq_m, sh_m = ops.quantize_1x32_grouped_gather(x, rt.slot_m, 1, G, rt.m_cap, True)
    out_m = ops.linear_mxfp8_grouped_masked(hq_m, wq, sh_m, sw, rt.cnt, rt.expected_m, rt.max_active, slots,
                                            None if slots is not None else rt.masked_shapes(N, H))
    torch.cuda.synchronize()
    eq = torch.equal(out_m.view(-1, N)[rt.slot_m.long()], d[rt.slot_c.long()])
    return [dict(case="slackvmm", slack_blocks=slack_blocks, tile=tile, guarded=which, b_max=rt.b_max,
                 b_alloc=b_alloc, equal=bool(eq), ok=bool(eq))]


def child_main(args):
    _child_imports()
    if torch.cuda.get_device_capability()[0] != 10:
        print(json.dumps([dict(case=args.case, skipped="not sm_100/103", ok=True)]))
        return 0
    case = args.case
    quick = args.quick
    m_list = [1, 7, 40, 256] if quick else M_GRID
    if case == "rule":
        res = child_cells(case, ["B", "C"], m_list, ROUTINGS, ["fc1", "fc1_fused", "fc2"],
                          prep_ms=(1, 64, 2048))
    elif case == "ptrarray":
        res = child_cells(case, ["B", "C"], m_list, ["uniform", "hot"], ["fc1", "fc1_fused", "fc2"])
    elif case == "slot":
        res = child_cells(case, ["B", "C"], [m for m in m_list if (m + 3) // 4 * 4 <= 64], ROUTINGS,
                          ["fc1", "fc1_fused", "fc2"])
    elif case.startswith("tile:"):
        gemms = ["fc1_fused"] if case.startswith("tile:fused:") else ["fc1", "fc2"]
        res = child_cells(case, ["B", "C"], TILE_M, ["uniform", "hot"], gemms)
    elif case == "chain":
        res = child_chain(args.ref)
    elif case == "ref020":
        res = child_ref020(args.ref)
    elif case == "arena":
        res = child_arena()
    elif case == "arena_masked":
        res = child_arena_masked()
    elif case.startswith("slack:"):
        _, slack, tile = case.split(":")
        res = child_slack(int(slack), tile)
    elif case.startswith("slackvmm:"):
        _, slack, tile, which = case.split(":")
        res = child_slack_guarded(int(slack), tile, which)
    elif case == "sched":
        res = child_sched()
    else:
        raise SystemExit(f"unknown case {case}")
    print("@@RESULTS@@" + json.dumps(res))
    return 0


# ----------------------------------------------------------------------------
# parent side (no CUDA context)
# ----------------------------------------------------------------------------

def case_env(case):
    env = {}
    if case == "ptrarray":
        env["FSO_GROUPED_SLOT"] = "0"
    elif case in ("slot", "sched"):
        env["FSO_GROUPED_SLOT"] = "force"
    elif case.startswith("tile:"):
        env["FSO_GROUPED_SLOT"] = "0"
        env["FSO_FORCE_TILE"] = case.split(":")[-1]
    elif case == "arena":
        env["FSO_GROUPED_SLOT"] = "0"
        env["FSO_PRINT_TILE_INFO"] = "1"
    elif case.startswith("slack:"):
        env["FSO_GROUPED_SLOT"] = "0"
        env["FSO_FORCE_TILE"] = case.split(":")[2]
        env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    return env


def run_child(case, args):
    env = dict(os.environ)
    env.update(case_env(case))
    cmd = [sys.executable, os.path.abspath(__file__), "--child", case]
    if args.quick:
        cmd.append("--quick")
    if args.ref:
        cmd += ["--ref", args.ref]
    if case == "ref020":
        base = os.environ.get("FSO_BASE020")
        if not base:
            return dict(case=case, skipped="FSO_BASE020 not set", ok=True), ""
        env["PYTHONPATH"] = base
    lock = os.environ.get("FSO_TEST_GPU_LOCK")
    if lock:
        # Shared card: take the lock for this child only, and inside it refuse to start
        # (exit 75, retried after a pause) while any other process is on the card.
        idle = ("[ \"$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)\" = 0 ] "
                "|| exit 75; exec \"$@\"")
        cmd = ["flock", "-w", "1500", lock, "bash", "-c", idle, "child"] + cmd
    t0 = time.time()
    for _attempt in range(40):
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if p.returncode != 75:
            break
        time.sleep(30)
    dt = time.time() - t0
    res = None
    for line in p.stdout.splitlines():
        if line.startswith("@@RESULTS@@"):
            res = json.loads(line[len("@@RESULTS@@"):])
    return dict(case=case, rc=p.returncode, seconds=round(dt, 1), results=res), p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", default=None)
    ap.add_argument("--cases", default="rule,ptrarray,slot,sched,tiles,ref020,chain,arena")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ref", default=None, help="path of the 0.2.0 layer reference file (ref020 writes, chain reads)")
    ap.add_argument("--out", default=None, help="write the per-case JSON here")
    args = ap.parse_args()
    if args.child:
        args.case = args.child
        return child_main(args)
    cases = []
    for c in args.cases.split(","):
        if c == "tiles":
            cases += [f"tile:{t}" for t in UNFUSED_TILES] + [f"tile:fused:{t}" for t in FUSED_TILES]
        else:
            cases.append(c)
    failures = 0
    summary = []
    for case in cases:
        rec, err = run_child(case, args)
        res = rec.get("results")
        bad = [r for r in (res or []) if not r.get("ok")]
        ok = rec.get("skipped") is not None or (rec.get("rc") == 0 and res is not None and not bad)
        if case == "arena" and ok:
            new_blocks = [ln for ln in err.splitlines() if "[fso sm100 weight arena] new block" in ln]
            rebound = [ln for ln in err.splitlines() if "[fso sm100 weight arena] rebound" in ln]
            last = new_blocks[-1] if new_blocks else ""
            rec["arena_new_blocks"] = len(new_blocks)
            rec["arena_rebound"] = len(rebound)
            rec["arena_last_line"] = last
            ok = ok and len(new_blocks) == 2 * ARENA_LAYERS and not rebound
        if not ok:
            failures += 1
            rec["stderr_tail"] = err[-3000:]
        n = len(res) if res else 0
        print(f"[{'PASS' if ok else 'FAIL'}] {case}: {n} cells, rc={rec.get('rc')}, {rec.get('seconds')} s"
              + (f", {len(bad)} failing" if bad else "")
              + (f", arena new blocks {rec.get('arena_new_blocks')}" if case == "arena" else ""), flush=True)
        for r in bad[:20]:
            print("    ", r, flush=True)
        if not ok and not bad:
            print(err[-3000:], flush=True)
        summary.append(rec)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)
    print(f"== {len(cases) - failures}/{len(cases)} cases passed ==")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
