#!/usr/bin/env python3
"""The sm_120/121 grouped MXFP8 GEMMs on the compact activation layout (0.2.1, charter decisions D1 and D4).

The masked slab gives every expert ``m_cap = align(M, 4)`` rows; the compact layout packs the routed rows into ONE
group of ``R = 128 * (B_max + 1)`` rows made of 128-row blocks, expert ``g``'s rows starting at row
``blk0[g] * 128``. The compact ops must write, at every live row, exactly the bytes the masked ops write for the
same routing, and nothing anywhere else. What is checked, per family (B: E=128, H=2048, I=768; C: E=256, H=2048,
I=512), per M in {1, 2, 3, 4, 7, 8, 16, 32, 64, 96, 128, 256, 512, 1024, 2048, 4096, 8192} and per routing
(uniform; one hot expert taking every token; ids padded with -1 and with G; the expert-parallel form, ids drawn
over 2G with the upper half mapped to -1):

  1. the out-style glue ops write exactly the bytes their allocating twins return, and nothing else:
     ``quantize_1x32_grouped_gather_out`` against ``quantize_1x32_grouped_gather`` (same one-group arguments),
     ``silu_chunk_mul_quantize_1x32_grouped_out`` against ``silu_chunk_mul_quantize_1x32_grouped``;
  2. the plain grouped GEMM (FC1 and FC2), the fused-SwiGLU FC1 (FP8 bytes and scale words) and, through
     ``moe_combine``, the layer output: compact live rows ``torch.equal`` to the masked op's, pair by pair;
  3. bounded stores: every compact output buffer is pre-filled with a sentinel and every row (and scale word) no
     routed pair owns must still hold it afterwards; the compact activation's padding rows hold FP8 NaN, so a
     tile that reads past an expert's rows would poison a live row if GEMM rows were not independent;
  4. the fused-combine FC2: its adds are atomic (bf16 atomics in whatever order the CTAs finish), so it is not
     bit-reproducible. It is compared EXACTLY on a routing that gives every token one live entry (then each
     output element receives one add onto zero, which is the same rounding of the same fp32 product as the
     deterministic chain), and on full top-k routings against the masked fused combine and the deterministic
     chain within the bound of the bf16 accumulation (topk roundings of a running sum: |delta| <= topk * 2^-8 *
     the sum of |terms|, plus one final rounding);
  5. every grouped instance the cascade can pick, forced with FSO_FORCE_TILE (and FSO_FORCE_MIN_BLOCKS=2 for the
     2-CTA forms), in one subprocess each because the knob is read once per process; and the math-warp form of
     the fused combine (FSO_MOE_SCATTER_WARP=0);
  6. a full chain in Python (torch reference routing -> gather_out -> compact fused FC1 -> compact FC2 ->
     moe_combine on the one-group down output) equals ``fso.moe.layer`` bit for bit at M in {1, 7, 64, 513,
     4096} for both families, eager and replayed from a captured CUDA graph (the routing rewritten in place
     between replays). With ``--layer-ref PATH`` it is also compared with outputs that another build -- the
     0.2.0 release wheel -- dumped with ``--dump-layer-ref PATH``.

The routing is built in torch here (S3's ``moe_build_routing_compact`` contract: masked_m [G], blk0 [G+1],
slot_of_flat [M*topk] = blk0[e]*128 + rank or -1, row_map / weight_of_slot [B_max*128]); the reference stays in
the test when the op lands. It is capture-safe (no host sync), so the chain captures with it.
"""
import argparse
import os
import subprocess
import sys

import torch

import fish_scales_ops as fso

FAMILIES = {"B": (128, 8, 2048, 768), "C": (256, 8, 2048, 512)}   # E, topk, H, I
MS = [1, 2, 3, 4, 7, 8, 16, 32, 64, 96, 128, 256, 512, 1024, 2048, 4096, 8192]
ROUTINGS = ["uniform", "hot", "padded", "ep"]
CHAIN_MS = [1, 7, 64, 513, 4096]
# Every grouped instance the sm_120 cascade or FSO_FORCE_TILE can reach (mxfp8/dispatch.cuh,
# gemm_dispatch_sm120_mxfp8_grouped): (TileM, TileN, Stages[, MinBlocksPerSm]).
FORCED = [(16, 128, 4), (16, 64, 4), (32, 128, 4), (32, 64, 4), (64, 64, 4), (64, 128, 2), (32, 128, 2),
          (32, 64, 2), (64, 64, 2), (96, 64, 2), (160, 128, 2), (96, 128, 2), (128, 128, 2),
          (16, 64, 4, 2), (16, 64, 2, 2), (16, 128, 2, 2), (32, 64, 2, 2), (64, 64, 2, 2)]
FORCED_MS = [1, 7, 64, 200, 1000]          # 200 / 1000 hot: an expert of 2 / 8 blocks, so TileM 96 / 160 tiles
FORCED_ROUTINGS = ["uniform", "hot"]       # cross 128-row block boundaries
SENT_U8 = 0x7F                             # float8_e4m3fn NaN
SENT_I16 = 0x7FC1                          # a bf16 NaN
SENT_I32 = 0x5A5A5A5A
BAD_TOKEN = 1 << 30                        # row_map of rows no pair owns: an address no kernel may touch

ops = torch.ops.fish_scales_ops


def b_max(m, topk, g):
    rows = m * topk
    return max(1, (rows + 127 * min(g, rows)) // 128)


def hints(m, topk, g):
    """The masked layout's host-static hints, exactly as the layer plan derives them."""
    return (m + 3) // 4 * 4, max(1, (m * topk + g - 1) // g), min(m * topk, g)


def make_ids(kind, m, g, topk, gen, dev):
    if kind == "ep":
        ids = torch.rand(m, 2 * g, device=dev, generator=gen).topk(topk, dim=1).indices.to(torch.int32)
        return torch.where(ids >= g, torch.full_like(ids, -1), ids).contiguous()
    r = torch.rand(m, g, device=dev, generator=gen)
    if kind == "hot":
        r[:, g // 3] = 2.0                        # expert g // 3 takes every token
    ids = r.topk(topk, dim=1).indices.to(torch.int32)
    if kind == "padded":
        ids[::7, 0] = -1                          # an expert-parallel remote entry
        ids[::11, 1] = g                          # the padded-row sentinel
        if m >= 4:
            ids[3 * m // 4:] = g                  # a padded graph bucket's tail rows
    return ids.contiguous()


def routing(ids, g, m_cap, wts=None):
    """Torch reference of both layouts' routing, from one rank assignment (pairs of an expert in flat order).

    Returns a dict with masked_m [G], blk0 [G+1], slot_c [M*topk] (compact rows), slot_m (masked slots
    g*m_cap + rank), and, given weights, row_map_c / wslot_c [B_max*128] and row_map_m / wslot_m [G*m_cap].
    Capture-safe: no host synchronisation."""
    dev = ids.device
    m, topk = ids.shape
    n = m * topk
    flat = ids.reshape(-1).long()
    valid = (flat >= 0) & (flat < g)
    e = torch.where(valid, flat, torch.full_like(flat, g))
    cnt_ext = torch.zeros(g + 1, dtype=torch.long, device=dev).scatter_add_(0, e, torch.ones_like(e))
    order = torch.sort(e, stable=True).indices
    pos = torch.empty_like(order).scatter_(0, order, torch.arange(n, device=dev))
    start_ext = torch.cumsum(cnt_ext, 0) - cnt_ext
    rank = pos - start_ext[e]
    cnt = cnt_ext[:g]
    blk0 = torch.zeros(g + 1, dtype=torch.long, device=dev)
    blk0[1:] = torch.cumsum((cnt + 127) // 128, 0)
    ec = e.clamp(max=g - 1)
    neg = torch.full_like(flat, -1)
    out = {
        "masked_m": cnt.to(torch.int32),
        "blk0": blk0.to(torch.int32),
        "slot_c": torch.where(valid, blk0[ec] * 128 + rank, neg).to(torch.int32),
        "slot_m": torch.where(valid, ec * m_cap + rank, neg).to(torch.int32),
    }
    if wts is not None:
        bm = b_max(m, topk, g)
        tok = torch.arange(n, device=dev) // topk
        live = valid.nonzero().squeeze(1)
        for key, slot, size in (("c", out["slot_c"], bm * 128), ("m", out["slot_m"], g * m_cap)):
            rm = torch.full((size,), BAD_TOKEN, dtype=torch.int32, device=dev)
            ws = torch.full((size,), float("nan"), dtype=torch.float32, device=dev)
            s = slot.long()[live]
            rm[s] = tok[live].to(torch.int32)
            ws[s] = wts.reshape(-1)[live]
            out["row_map_" + key] = rm
            out["wslot_" + key] = ws
    return out


def filled(shape, dtype, dev):
    t = torch.empty(shape, dtype=dtype, device=dev)
    if dtype == torch.float8_e4m3fn:
        t.view(torch.uint8).fill_(SENT_U8)
    elif dtype == torch.bfloat16:
        t.view(torch.int16).fill_(SENT_I16)
    elif dtype == torch.int32:
        t.fill_(SENT_I32)
    else:
        raise ValueError(dtype)
    return t


def raw(t):
    return t.view({torch.float8_e4m3fn: torch.uint8, torch.bfloat16: torch.int16}.get(t.dtype, t.dtype))


def sentinel_of(t):
    return {torch.float8_e4m3fn: SENT_U8, torch.bfloat16: SENT_I16, torch.int32: SENT_I32}[t.dtype]


class Case:
    """One routing of one family at one M, and the comparisons on it."""

    def __init__(self, fam, w, m, kind, seed, ids=None):
        self.fam, self.m, self.kind = fam, m, kind
        e, topk, h, i = FAMILIES[fam]
        self.g, self.topk, self.h, self.i = e, topk, h, i
        dev = "cuda"
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
        self.x = torch.randn(m, h, device=dev, dtype=torch.bfloat16, generator=gen)
        self.ids = make_ids(kind, m, e, topk, gen, dev) if ids is None else ids
        self.wts = torch.softmax(torch.rand(m, topk, device=dev, generator=gen), dim=1).float().contiguous()
        self.m_cap, self.em, self.mag = hints(m, topk, e)
        self.bm = b_max(m, topk, e)
        self.R = 128 * (self.bm + 1)
        self.r = routing(self.ids, e, self.m_cap, self.wts)
        self.w = w
        self.fail = []
        live = (self.r["slot_c"] >= 0).nonzero().squeeze(1)
        self.live = live
        self.rows_c = self.r["slot_c"].long()[live]
        self.slots_m = self.r["slot_m"].long()[live]
        self.g_of = self.slots_m // self.m_cap
        self.rank_of = self.slots_m % self.m_cap
        mask = torch.zeros(self.R, dtype=torch.bool, device=dev)
        mask[self.rows_c] = True
        self.live_rows = mask
        self.n_live = int(live.numel())
        blk_used = int(self.r["blk0"][-1])
        if blk_used > self.bm:
            self.fail.append(f"routing used {blk_used} blocks > B_max {self.bm}")
        if self.rows_c.unique().numel() != self.n_live:
            self.fail.append("two live pairs share a compact row")

    def tag(self):
        return f"fam {self.fam} M={self.m} {self.kind}"

    def check(self, ok, what):
        if not ok:
            self.fail.append(f"{self.tag()}: {what}")
        return ok

    # -- comparisons -----------------------------------------------------------------------------------------
    def rows_vs_masked(self, compact, masked, what):
        """compact [R, N] vs masked [G, m_cap, N], pair by pair at the live rows."""
        n = compact.shape[-1]
        a = raw(compact).reshape(self.R, n)[self.rows_c]
        b = raw(masked).reshape(-1, n)[self.slots_m]
        return self.check(torch.equal(a, b), f"{what}: live rows differ from the masked op's")

    def words_vs_masked(self, compact, masked, kp, what):
        """one-group K-major words [1, kp, R] vs masked [G, kp, m_cap], pair by pair."""
        a = compact.reshape(kp, self.R).t()[self.rows_c]
        b = masked.reshape(self.g, kp, self.m_cap).permute(0, 2, 1)[self.g_of, self.rank_of]
        return self.check(torch.equal(a, b), f"{what}: live scale words differ from the masked op's")

    def untouched_rows(self, t, what):
        """Every row of a [R, ...] output that no pair owns still holds the sentinel."""
        rest = raw(t).reshape(self.R, -1)[~self.live_rows]
        return self.check(bool((rest == sentinel_of(t)).all()), f"{what}: a row no pair owns was written")

    def untouched_words(self, t, kp, what):
        rest = t.reshape(kp, self.R)[:, ~self.live_rows]
        return self.check(bool((rest == SENT_I32).all()), f"{what}: a scale word no pair owns was written")

    # -- the steps -------------------------------------------------------------------------------------------
    def run(self, do_glue=True, do_combine=True):
        w13q, sw13, w2q, sw2 = self.w
        r, g, h, i, R = self.r, self.g, self.h, self.i, self.R
        dev = "cuda"
        kph, kpi = h // 128, i // 128
        args_c = (r["masked_m"], r["blk0"], self.m_cap, self.em, self.mag)

        # 1. gather: the out-style op against the masked twin, and against its own allocating twin.
        hq_m, sh_m = ops.quantize_1x32_grouped_gather(self.x, r["slot_m"], self.topk, g, self.m_cap, True)
        hq_c = filled((R, h), torch.float8_e4m3fn, dev)
        sh_c = filled((1, kph, R), torch.int32, dev)
        ops.quantize_1x32_grouped_gather_out(self.x, r["slot_c"], self.topk, 1, R, True, hq_c, sh_c)
        self.rows_vs_masked(hq_c, hq_m, "gather_out")
        self.words_vs_masked(sh_c, sh_m, kph, "gather_out")
        self.untouched_rows(hq_c, "gather_out")
        self.untouched_words(sh_c, kph, "gather_out")
        if do_glue:
            tq, ts = ops.quantize_1x32_grouped_gather(self.x, r["slot_c"], self.topk, 1, R, True)
            self.check(torch.equal(raw(tq).reshape(R, h)[self.rows_c], raw(hq_c)[self.rows_c])
                       and torch.equal(ts.reshape(kph, R)[:, self.rows_c], sh_c.reshape(kph, R)[:, self.rows_c]),
                       "gather_out: bytes differ from quantize_1x32_grouped_gather with the same arguments")
            del tq, ts

        # 2. FC1, plain: masked slab vs compact rows.
        gu_m = ops.linear_mxfp8_grouped_masked(hq_m, w13q, sh_m, sw13, r["masked_m"], self.em, self.mag, None, None)
        gu_c = filled((R, 2 * i), torch.bfloat16, dev)
        ops.linear_mxfp8_grouped_compact(hq_c, w13q, sh_c, sw13, *args_c, None, None, None, gu_c)
        self.rows_vs_masked(gu_c, gu_m, "FC1 plain")
        self.untouched_rows(gu_c, "FC1 plain")
        del gu_m

        # 3. FC1 fused SwiGLU: FP8 bytes and scale words.
        dq_m, sd_m = ops.linear_mxfp8_grouped_masked_swiglu(
            hq_m, w13q, sh_m, sw13, r["masked_m"], self.em, self.mag, None, None)
        del hq_m, sh_m
        dq_c = filled((R, i), torch.float8_e4m3fn, dev)
        sd_c = filled((1, kpi, R), torch.int32, dev)
        ops.linear_mxfp8_grouped_compact_swiglu(hq_c, w13q, sh_c, sw13, *args_c, None, None, None, dq_c, sd_c)
        self.rows_vs_masked(dq_c, dq_m, "FC1 fused")
        self.words_vs_masked(sd_c, sd_m, kpi, "FC1 fused")
        self.untouched_rows(dq_c, "FC1 fused")
        self.untouched_words(sd_c, kpi, "FC1 fused")
        if do_glue:
            # The unfused compact pair (plain FC1 + the out-style SwiGLU requantize, pairwise for the interleaved
            # weights) writes the fused FC1's bytes; and the out-style op writes its allocating twin's bytes.
            dq2 = filled((R, i), torch.float8_e4m3fn, dev)
            sd2 = filled((1, kpi, R), torch.int32, dev)
            ops.silu_chunk_mul_quantize_1x32_grouped_out(gu_c.view(1, R, 2 * i), r["slot_c"], True, True, dq2, sd2)
            self.check(torch.equal(raw(dq2)[self.rows_c], raw(dq_c)[self.rows_c])
                       and torch.equal(sd2.reshape(kpi, R)[:, self.rows_c], sd_c.reshape(kpi, R)[:, self.rows_c]),
                       "compact plain FC1 + silu_out differs from the compact fused FC1")
            self.untouched_rows(dq2, "silu_out")
            self.untouched_words(sd2, kpi, "silu_out")
            tq, ts = ops.silu_chunk_mul_quantize_1x32_grouped(gu_c.view(1, R, 2 * i), r["slot_c"], True, True)
            self.check(torch.equal(raw(tq).reshape(R, i)[self.rows_c], raw(dq2)[self.rows_c])
                       and torch.equal(ts.reshape(kpi, R)[:, self.rows_c], sd2.reshape(kpi, R)[:, self.rows_c]),
                       "silu_out: bytes differ from silu_chunk_mul_quantize_1x32_grouped with the same arguments")
            del dq2, sd2, tq, ts
        del gu_c, hq_c, sh_c

        # 4. FC2, plain, and the deterministic combine.
        dn_m = ops.linear_mxfp8_grouped_masked(dq_m, w2q, sd_m, sw2, r["masked_m"], self.em, self.mag, None, None)
        dn_c = filled((R, h), torch.bfloat16, dev)
        ops.linear_mxfp8_grouped_compact(dq_c, w2q, sd_c, sw2, *args_c, None, None, None, dn_c)
        self.rows_vs_masked(dn_c, dn_m, "FC2 plain")
        self.untouched_rows(dn_c, "FC2 plain")
        out_m = ops.moe_combine(dn_m, r["slot_m"], self.wts, None, None, None)
        out_c = ops.moe_combine(dn_c.view(1, R, h), r["slot_c"], self.wts, None, None, None)
        self.check(torch.equal(out_c, out_m), "moe_combine over the compact rows differs from the masked chain")
        # sum_j w_j |dn_j| per output element (the weights are positive): the scale of the bf16 accumulation
        # error of the fused combine below.
        abs_sum = (ops.moe_combine(dn_c.abs().view(1, R, h), r["slot_c"], self.wts, None, None, None).float()
                   if do_combine else None)
        del dn_m, dn_c

        # 5. FC2 with the fused combine against the masked fused combine and the deterministic chain.
        if do_combine:
            fc_m = torch.zeros(self.m, h, device=dev, dtype=torch.bfloat16)
            ops.linear_mxfp8_grouped_masked_combine(dq_m, w2q, sd_m, sw2, r["masked_m"], r["row_map_m"],
                                                    r["wslot_m"], fc_m, self.em, self.mag)
            fc_c = torch.zeros(self.m, h, device=dev, dtype=torch.bfloat16)
            ops.linear_mxfp8_grouped_compact_combine(dq_c, w2q, sd_c, sw2, r["masked_m"], r["blk0"], r["row_map_c"],
                                                     r["wslot_c"], self.m_cap, self.em, self.mag, fc_c)
            self.combine_checks(fc_c, fc_m, out_m, abs_sum)
        del dq_m, sd_m, dq_c, sd_c

    def combine_checks(self, fc_c, fc_m, det, abs_sum):
        """The fused combine adds round(w_j * v_j) to a bf16 running sum with bf16 atomics in CTA-completion
        order; the deterministic chain sums w_j * v_j in fp32 and rounds once. With u = 2^-8 (bf16 unit
        roundoff) and S = sum_j |w_j v_j|: each fused result is within u * S * (1 + topk) of the exact sum (the
        product roundings plus topk running-sum roundings) and the deterministic one within u * S, so the
        two differ by at most (topk + 2) * u * S and two fused results by at most 2 * (topk + 1) * u * S.
        With one live entry per token there is a single add onto zero, the same rounding of the same fp32
        product as the deterministic chain: exact."""
        if self.kind == "single":
            self.check(torch.equal(fc_c, fc_m) and torch.equal(fc_c, det),
                       "fused combine with one live entry per token is not exact (vs the masked fused combine "
                       "and the deterministic chain)")
            return
        u = 2.0 ** -8
        slack = 1.05                                   # abs_sum is itself a bf16 result
        for other, name, k in ((det, "the deterministic chain", self.topk + 2),
                               (fc_m, "the masked fused combine", 2 * (self.topk + 1))):
            bound = k * u * abs_sum * slack
            dev_ = (fc_c.float() - other.float()).abs()
            self.check(bool((dev_ <= bound).all()),
                       f"fused combine differs from {name} beyond the bf16 accumulation bound "
                       f"(max excess {float((dev_ - bound).max()):.3e})")


def quantized_weights(fam, seed=11):
    e, topk, h, i = FAMILIES[fam]
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    w13 = (torch.randn(e, 2 * i, h, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(e, h, i, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    w13q, sw13 = fso.compat.quantize_moe_weights_1x32_fp8(w13, w13_interleave=True)
    w2q, sw2 = fso.compat.quantize_moe_weights_1x32_fp8(w2)
    return w13, w2, (w13q, sw13, w2q, sw2)


def single_entry_ids(m, g, topk, gen):
    ids = torch.rand(m, g, device="cuda", generator=gen).topk(topk, dim=1).indices.to(torch.int32)
    ids[:, 1:] = -1                       # one live entry per token
    return ids.contiguous()


def grid_checks(fams, ms, kinds, failures, light=False, label=""):
    n_cases = 0
    for fam in fams:
        _, _, wq = quantized_weights(fam)
        for m in ms:
            for kind in kinds:
                c = Case(fam, wq, m, kind, seed=m * 7919 + ROUTINGS.index(kind) * 131 + (fam == "C"))
                c.run(do_glue=not light)
                n_cases += 1
                status = "OK" if not c.fail else "FAIL"
                print(f"  {label}fam {fam} M={m:5d} {kind:8s} R={c.R:6d} live={c.n_live:6d} "
                      f"blocks={int(c.r['blk0'][-1]):4d}/{c.bm:4d}  {status}", flush=True)
                failures += c.fail
                del c
            # the single-live-entry routing: the fused combine is exact there
            if not light and m in (1, 64, 1024, 8192):
                gen = torch.Generator(device="cuda")
                gen.manual_seed(m + 5)
                e, topk, _, _ = FAMILIES[fam]
                c = Case(fam, wq, m, "single", seed=m * 31 + 7, ids=single_entry_ids(m, e, topk, gen))
                c.run(do_glue=False)
                print(f"  {label}fam {fam} M={m:5d} single   (one live entry per token; fused combine exact)  "
                      f"{'OK' if not c.fail else 'FAIL'}", flush=True)
                failures += c.fail
                del c
        del wq
        torch.cuda.empty_cache()
    return n_cases


def chain_compact(x, w13q, sw13, w2q, sw2, ids, wts, interleaved, bufs=None):
    """The compact layer chain: routing reference -> gather_out -> FC1 -> FC2 -> moe_combine. `bufs` carries the
    preallocated buffers (one bucket's), so a captured graph reuses them."""
    m, topk = ids.shape
    g, i = int(w13q.shape[0]), int(w2q.shape[2])
    h = int(x.shape[1])
    m_cap, em, mag = hints(m, topk, g)
    R = 128 * (b_max(m, topk, g) + 1)
    if bufs is None:
        bufs = dict(hq=torch.empty(R, h, dtype=torch.float8_e4m3fn, device="cuda"),
                    sh=torch.empty(1, h // 128, R, dtype=torch.int32, device="cuda"),
                    dq=torch.empty(R, i, dtype=torch.float8_e4m3fn, device="cuda"),
                    sd=torch.empty(1, i // 128, R, dtype=torch.int32, device="cuda"),
                    gu=torch.empty(R, 2 * i, dtype=torch.bfloat16, device="cuda"),
                    dn=torch.empty(R, h, dtype=torch.bfloat16, device="cuda"))
    r = routing(ids, g, m_cap)
    ops.quantize_1x32_grouped_gather_out(x, r["slot_c"], topk, 1, R, True, bufs["hq"], bufs["sh"])
    args_c = (r["masked_m"], r["blk0"], m_cap, em, mag, None, None, None)
    if interleaved:
        ops.linear_mxfp8_grouped_compact_swiglu(bufs["hq"], w13q, bufs["sh"], sw13, *args_c, bufs["dq"], bufs["sd"])
    else:
        ops.linear_mxfp8_grouped_compact(bufs["hq"], w13q, bufs["sh"], sw13, *args_c, bufs["gu"])
        ops.silu_chunk_mul_quantize_1x32_grouped_out(bufs["gu"], r["slot_c"], True, False, bufs["dq"], bufs["sd"])
    ops.linear_mxfp8_grouped_compact(bufs["dq"], w2q, bufs["sd"], sw2, *args_c, bufs["dn"])
    return ops.moe_combine(bufs["dn"].view(1, R, h), r["slot_c"], wts, None, None, None), bufs


def chain_inputs(fam, m):
    e, topk, h, i = FAMILIES[fam]
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1000 + m)
    hidden = torch.randn(m, h, device="cuda", dtype=torch.bfloat16, generator=gen)
    ids = torch.rand(m, e, device="cuda", generator=gen).topk(topk, dim=1).indices.to(torch.int32)
    ids[::7, 0] = -1
    ids[::11, 1] = e
    wts = torch.softmax(torch.rand(m, topk, device="cuda", generator=gen), dim=1).float()
    return hidden, ids.contiguous(), wts.contiguous()


def dump_layer_ref(path):
    """fso.moe.layer outputs of whatever build is imported, for the chain inputs (run it under the 0.2.0 wheel)."""
    res = {"fso_version": getattr(fso, "__version__", "?"), "fso_file": fso.__file__}
    for fam in FAMILIES:
        w13, w2, _ = quantized_weights(fam)
        experts = fso.moe.prepare_experts(w13, w2, format="mxfp8")
        del w13, w2
        for m in CHAIN_MS:
            hidden, ids, wts = chain_inputs(fam, m)
            res[f"{fam}/M{m}"] = fso.moe.layer(hidden, experts, ids, wts).cpu()
        del experts
        torch.cuda.empty_cache()
    torch.save(res, path)
    print(f"wrote {path} ({len(res) - 2} layer outputs from {res['fso_file']}, version {res['fso_version']})")
    return 0


def chain_checks(failures, layer_ref=None):
    ref_dump = torch.load(layer_ref) if layer_ref else None
    if ref_dump is not None:
        print(f"  reference dump: {layer_ref} (fish_scales_ops {ref_dump.get('fso_version')} from "
              f"{ref_dump.get('fso_file')})")
    for fam in FAMILIES:
        e, topk, h, i = FAMILIES[fam]
        w13, w2, _ = quantized_weights(fam)
        experts = fso.moe.prepare_experts(w13, w2, format="mxfp8")
        del w13, w2
        args_w = (experts.w13, experts.sw13, experts.w2, experts.sw2)
        for m in CHAIN_MS:
            hidden, ids, wts = chain_inputs(fam, m)
            ref = fso.moe.layer(hidden, experts, ids, wts)
            got, bufs = chain_compact(hidden, *args_w, ids, wts, experts.w13_interleaved)
            ok_eager = torch.equal(got, ref)
            ok_dump = True
            if ref_dump is not None:
                ok_dump = torch.equal(got.cpu(), ref_dump[f"{fam}/M{m}"])
            # Captured: the whole compact chain, routing included, in one graph; then the routing is rewritten
            # in place and the graph replayed against the layer on the new routing.
            static_ids = ids.clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    chain_compact(hidden, *args_w, static_ids, wts, experts.w13_interleaved, bufs)
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out_g, _ = chain_compact(hidden, *args_w, static_ids, wts, experts.w13_interleaved, bufs)
            graph.replay()
            ok_g1 = torch.equal(out_g, ref)
            gen = torch.Generator(device="cuda")
            gen.manual_seed(77 + m)
            new_ids = torch.rand(m, e, device="cuda", generator=gen).topk(topk, dim=1).indices.to(torch.int32)
            new_ids[::5, 2] = -1
            static_ids.copy_(new_ids)
            graph.replay()
            ref2 = fso.moe.layer(hidden, experts, new_ids, wts)
            ok_g2 = torch.equal(out_g, ref2)
            torch.cuda.synchronize()
            ok = ok_eager and ok_dump and ok_g1 and ok_g2
            print(f"  chain fam {fam} M={m:5d}: eager {'==' if ok_eager else '!='} fso.moe.layer"
                  + ("" if ref_dump is None else f", {'==' if ok_dump else '!='} 0.2.0 dump")
                  + f"; graph replay {'==' if ok_g1 else '!='}, after an in-place routing rewrite "
                  f"{'==' if ok_g2 else '!='}  {'OK' if ok else 'FAIL'}", flush=True)
            if not ok:
                failures.append(f"chain fam {fam} M={m}: eager={ok_eager} dump={ok_dump} graph={ok_g1} "
                                f"graph_rewritten={ok_g2}")
            del graph, out_g, bufs, got, ref, ref2
        del experts
        torch.cuda.empty_cache()


def refusal_checks(failures):
    """Argument checks that guard the layout contract."""
    dev = "cuda"
    _, _, wq = quantized_weights("C")
    w13q, sw13, _, _ = wq
    g, h = int(w13q.shape[0]), int(w13q.shape[2])
    masked_m = torch.zeros(g, dtype=torch.int32, device=dev)
    blk0 = torch.zeros(g + 1, dtype=torch.int32, device=dev)
    a = torch.empty(256, h, dtype=torch.float8_e4m3fn, device=dev)
    sa = torch.empty(1, h // 128, 256, dtype=torch.int32, device=dev)
    d = torch.empty(256, int(w13q.shape[1]), dtype=torch.bfloat16, device=dev)

    def refused(fn, what):
        try:
            fn()
        except (RuntimeError, ValueError, NotImplementedError):
            print(f"  OK   refused: {what}")
            return
        failures.append(f"not refused: {what}")
        print(f"  FAIL not refused: {what}")

    refused(lambda: ops.linear_mxfp8_grouped_compact(a, w13q, sa, sw13, masked_m, masked_m, 4, 1, 8,
                                                     None, None, None, d), "blk0 with G entries (masked_m by mistake)")
    refused(lambda: ops.linear_mxfp8_grouped_compact(a[:200], w13q, sa, sw13, masked_m, blk0, 4, 1, 8,
                                                     None, None, None, d[:200]), "R not a multiple of 128")
    refused(lambda: ops.linear_mxfp8_grouped_compact(a, w13q, sa, sw13, masked_m, blk0, 4, 1, 8,
                                                     masked_m, None, None, d), "an active-expert list on sm_120")
    refused(lambda: ops.linear_mxfp8_grouped_compact(a, w13q, sa, sw13, masked_m, blk0, 4, 1, 8,
                                                     None, None, torch.zeros(3, g, dtype=torch.int64, device=dev), d),
            "per-call pointer arrays on sm_120")
    refused(lambda: ops.linear_mxfp8_grouped_compact(a, w13q, sa[:, :, :128], sw13, masked_m, blk0, 4, 1, 8,
                                                     None, None, None, d), "activation scales of the wrong size")
    refused(lambda: ops.linear_mxfp8_grouped_compact(a, w13q, sa, sw13, masked_m, blk0, 6, 1, 8,
                                                     None, None, None, d), "m_cap not a multiple of 4")
    # A call with no routed row at all writes nothing.
    d.view(torch.int16).fill_(SENT_I16)
    ops.linear_mxfp8_grouped_compact(a, w13q, sa, sw13, masked_m, blk0, 4, 1, 8, None, None, None, d)
    torch.cuda.synchronize()
    ok = bool((d.view(torch.int16) == SENT_I16).all())
    print(f"  {'OK  ' if ok else 'FAIL'} a call with no routed row writes nothing")
    if not ok:
        failures.append("a call with no routed row wrote its output")


def forced_parent(failures):
    """One subprocess per forced instance (the knobs are read once per process)."""
    here = os.path.abspath(__file__)
    runs = [(f"FSO_FORCE_TILE={t[0]},{t[1]},{t[2]}" + (" FSO_FORCE_MIN_BLOCKS=2" if len(t) == 4 else ""),
             {"FSO_FORCE_TILE": f"{t[0]},{t[1]},{t[2]}", **({"FSO_FORCE_MIN_BLOCKS": "2"} if len(t) == 4 else {})})
            for t in FORCED]
    runs.append(("FSO_MOE_SCATTER_WARP=0 (math-warp fused combine, cascade picks)", {"FSO_MOE_SCATTER_WARP": "0"}))
    # The permuted group order is the one path on which the scheduler's slot is not the expert id, and the expert
    # id is what indexes blk0.
    runs.append(("FSO_MOE_GROUP_ORDER=auto FSO_MOE_GROUP_BLOCK=8 (permuted expert walk, cascade picks)",
                 {"FSO_MOE_GROUP_ORDER": "auto", "FSO_MOE_GROUP_BLOCK": "8"}))
    for label, extra in runs:
        env = dict(os.environ)
        for k in ("FSO_FORCE_TILE", "FSO_FORCE_MIN_BLOCKS", "FSO_FORCE_TILE_K", "FSO_MOE_SCATTER_WARP",
                  "FSO_MOE_GROUP_ORDER", "FSO_MOE_GROUP_BLOCK"):
            env.pop(k, None)
        env.update(extra)
        r = subprocess.run([sys.executable, here, "--forced-case"], env=env, capture_output=True, text=True)
        tail = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()]
        warned = [ln for ln in tail if "not wired" in ln or "ignoring" in ln]
        ok = r.returncode == 0 and not warned
        print(f"  {'OK  ' if ok else 'FAIL'} {label}: {tail[-1] if tail else '(no output)'}", flush=True)
        if not ok:
            failures.append(f"forced {label}: rc={r.returncode} " + " | ".join(tail[-6:]))


def probe_launch():
    """Launch the forced instance once in each layout. Returns {layout: None or the first line of its error}."""
    _, _, wq = quantized_weights("C")
    w13q, sw13, _, _ = wq
    c = Case("C", wq, 64, "uniform", seed=5)
    r = c.r
    hq_m, sh_m = ops.quantize_1x32_grouped_gather(c.x, r["slot_m"], c.topk, c.g, c.m_cap, True)
    hq_c = filled((c.R, c.h), torch.float8_e4m3fn, "cuda")
    sh_c = filled((1, c.h // 128, c.R), torch.int32, "cuda")
    ops.quantize_1x32_grouped_gather_out(c.x, r["slot_c"], c.topk, 1, c.R, True, hq_c, sh_c)
    gu_c = filled((c.R, 2 * c.i), torch.bfloat16, "cuda")
    calls = {
        "masked": lambda: ops.linear_mxfp8_grouped_masked(hq_m, w13q, sh_m, sw13, r["masked_m"], c.em, c.mag,
                                                          None, None),
        "compact": lambda: ops.linear_mxfp8_grouped_compact(hq_c, w13q, sh_c, sw13, r["masked_m"], r["blk0"],
                                                            c.m_cap, c.em, c.mag, None, None, None, gu_c),
    }
    res = {}
    for name, fn in calls.items():
        try:
            fn()
            torch.cuda.synchronize()
            res[name] = None
        except RuntimeError as ex:
            res[name] = str(ex).strip().splitlines()[-1]
    return res


def forced_child():
    failures = []
    # Two instances FSO_FORCE_TILE can name do not fit this device: (128,128,2) and (160,128,2) need more shared
    # memory with the grouped instances' separate epilogue buffer than an sm_120 CTA may have, so the masked op
    # refuses them (cudaFuncSetAttribute fails) and the cascade never picks them for these shapes. The compact
    # instantiation must then refuse in the same way, and there is nothing further to compare.
    probe = probe_launch()
    if probe["masked"] is not None or probe["compact"] is not None:
        both = all(v is not None and "cannot launch" in v for v in probe.values())
        print(f"{'N/A' if both else 'FAIL'}: masked: {probe['masked'] or 'launched'}; compact: "
              f"{probe['compact'] or 'launched'}" + (" -- the instance cannot launch on this device in either "
                                                     "layout" if both else ""))
        return 0 if both else 1
    n = grid_checks(list(FAMILIES), FORCED_MS, FORCED_ROUTINGS, failures, light=True, label="    ")
    if failures:
        for f in failures[:10]:
            print(f"FAIL {f}")
        return 1
    print(f"{n} cases (plain FC1/FC2, fused FC1, fused combine) equal to the masked ops")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forced-case", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--dump-layer-ref", metavar="PATH", help="write fso.moe.layer outputs of the imported build")
    ap.add_argument("--layer-ref", metavar="PATH", help="also compare the compact chain with such a dump")
    ap.add_argument("--quick", action="store_true", help="a reduced grid (development)")
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        where = "no CUDA device" if not torch.cuda.is_available() \
            else f"device is sm_{torch.cuda.get_device_capability()[0]}x"
        print(f"SKIP: test_mxfp8_compact_sm120 is sm_120/121 (RTX 5090) only ({where})")
        return 0
    if args.dump_layer_ref:
        return dump_layer_ref(args.dump_layer_ref)
    if args.forced_case:
        return forced_child()
    failures = []
    ms = [1, 7, 64, 513, 4096] if args.quick else MS
    print("== compact vs masked, pair by pair: gather_out, FC1 plain / fused, FC2, moe_combine, fused combine ==")
    n = grid_checks(list(FAMILIES), ms, ROUTINGS, failures)
    print(f"  {n} cases")
    print("== argument checks ==")
    refusal_checks(failures)
    print("== every grouped instance, forced (one subprocess each) ==")
    forced_parent(failures)
    print("== the compact chain against fso.moe.layer, eager and captured ==")
    chain_checks(failures, args.layer_ref)
    if failures:
        print("\nFAILURES:")
        for f in failures[:40]:
            print(f"  {f}")
        if len(failures) > 40:
            print(f"  ... {len(failures) - 40} more")
        return 1
    print("\nsm_120 compact-layout grouped GEMMs: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
