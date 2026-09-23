#!/usr/bin/env python3
"""MoE grouped-GEMM reference baselines — Qwen3-30B-A3B geometry.

Measures the *reference* kernels the fso MoE (grouped GEMM) project must meet
or beat, under the standard fso graph-replay protocol (see docs/perf/README.md):
per-cell subprocess, 15 eager warmups, side-stream warmup, one
torch.cuda.graph capture, then 3 reps x 50 replays -> median us.

Geometry (checkpoints/qwen3-30a3 config.json): E=128 experts, topk=8,
hidden=2048, moe_intermediate=768, 48 layers, gated silu, no shared expert.
  gate_up : [8M, 2048] x [E, 1536, 2048] -> [8M, 1536]
  down    : [8M,  768] x [E, 2048,  768] -> [8M, 2048]

Implementations benchmarked (all from the serving venv, no fso code):
  dg_fp8_cont    deep_gemm m_grouped_fp8_gemm_nt_contiguous  (1x128/128x128)
  dg_fp8_masked  deep_gemm fp8_m_grouped_gemm_nt_masked      (decode layout)
  dg_bf16_cont   deep_gemm m_grouped_bf16_gemm_nt_contiguous
  dg_bf16_masked deep_gemm m_grouped_bf16_gemm_nt_masked
  triton_bf16    sglang triton fused_experts (BF16)          [whole layer]
  triton_fp8b    sglang triton fused_experts w8a8 block=[128,128] [whole layer]
  dg_fp8_layer   production masked pipeline: moe_ep_deepgemm_preprocess ->
                 masked gemm -> silu_mul_quant -> masked gemm -> post_reorder
                 (mirrors sglang moe_runner/deep_gemm.py)     [whole layer]
  torch_smm_mxfp8_layer   torch 2.11 F.scaled_grouped_mm, MXFP8 1x32 blocked
                 scales, expert-sorted contiguous rows        [whole layer]
  torch_grouped_bf16_layer  torch 2.11 torch._grouped_mm, bf16, no quantize
                 (unquantized reference speed)                [whole layer]
  *_layer_shared  the two above plus Family C's dense shared expert
                 (F.scaled_mm / F.linear) and the residual add

deep_gemm's grouped kernels are the same kernels TensorRT-LLM drives via
DeepGemmFusedMoE / CutlassFp8BlockScaleGemmRunner::moeGemm on Hopper; the
triton w8a8-block path is the same family as TRT-LLM's
fused_moe_triton_fp8_block_scale.py (the sm_120 fallback), so it doubles as
the RTX 5090 reference where deep_gemm is unavailable.

Run inside the tiny_sglang serving venv (needs deep_gemm + sglang + triton):
  CUDA_VISIBLE_DEVICES=<idx> <venv>/bin/python bench_moe_qwen3_30a3.py \
      --run --out /data/bench-runs/<run>/moe_baseline.jsonl

Cell-level fields: us (graph-replay median), cos (vs unquantized bf16
reference), tflops (useful flops only: 2*8M*N*K, padding excluded), and
w_gbps (active-expert weight bytes / us — the decode-side roofline metric).
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys

# Routed-expert geometry per model (docs/perf/README.md §3). Shared experts
# (Qwen3.5) are dense GEMMs on every token and are benched by
# bench_qwen3_4b_mlp.py --family, not here.
MODELS = {
    # SHARED_INTER: intermediate size of the always-on shared expert (0 = none).
    # The `*_layer_shared` impls add it to the routed layer inside the same
    # CUDA graph (routed + shared dense MLP + residual add) — Family C's whole
    # MoE block (docs/perf/layer/README.md).
    "qwen3-30a3":   dict(E=128, TOPK=8, HIDDEN=2048, INTER=768, LAYERS=48, SHARED_INTER=0),    # Family B
    "qwen3.5-35a3": dict(E=256, TOPK=8, HIDDEN=2048, INTER=512, LAYERS=40, SHARED_INTER=512),  # Family C
}
MODEL = "qwen3-30a3"
E = TOPK = HIDDEN = INTER = LAYERS = None
SHARED_INTER = 0
PROJ = {}   # proj -> (N, K)


def set_model(name):
    global MODEL, E, TOPK, HIDDEN, INTER, LAYERS, PROJ, SHARED_INTER
    g = MODELS[name]
    MODEL = name
    E, TOPK, HIDDEN, INTER, LAYERS = g["E"], g["TOPK"], g["HIDDEN"], g["INTER"], g["LAYERS"]
    SHARED_INTER = g.get("SHARED_INTER", 0)
    PROJ = {"gate_up": (2 * INTER, HIDDEN), "down": (HIDDEN, INTER)}


set_model(MODEL)

M_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]  # docs/perf/README.md §4; off-grid points via --Ms
DECODE_M = [m for m in M_GRID if m <= 128]

KERNEL_IMPLS = ("dg_fp8_cont", "dg_bf16_cont", "dg_fp8_masked", "dg_bf16_masked",
                "fso_mxfp8_grouped")
LAYER_IMPLS = ("triton_bf16", "triton_fp8b", "dg_fp8_layer", "fso_mxfp8_layer",
               "fso_bsfp8_layer", "fso_mxfp8_layer_shared", "fso_bsfp8_layer_shared",
               "torch_smm_mxfp8_layer", "torch_grouped_bf16_layer",
               "torch_smm_mxfp8_layer_shared", "torch_grouped_bf16_layer_shared")
# torch-native layer baselines (torch >= 2.11): `torch.nn.functional.
# scaled_grouped_mm` (MXFP8 1x32, blocked scales) and `torch._grouped_mm`
# (bf16). Neither needs deep_gemm, sglang or fso. They are the same-device
# comparator for an fso grouped path where fso has none.
TORCH_IMPLS = ("torch_smm_mxfp8_layer", "torch_grouped_bf16_layer")
# Minimum compute-capability major version per torch-native impl.
# `torch._grouped_mm` (bf16) runs from sm_90 onwards, so Families B and C get
# a BF16 reference row on the H200 as well. The MXFP8 form of
# `scaled_grouped_mm` needs the Blackwell block-scaled grouped kernels, i.e.
# sm_100 or newer.
TORCH_IMPL_MIN_MAJOR = {"torch_smm_mxfp8_layer": 10,
                        "torch_grouped_bf16_layer": 9}


def torch_impls_for_device():
    """Which torch-native impls this device can run, without launching a kernel.

    Reads the compute capability of device 0 (a property query, not a kernel)
    and filters `TORCH_IMPLS` by `TORCH_IMPL_MIN_MAJOR`. Gating here keeps an
    unfiltered `--run` on sm_90 or sm_120 from spawning worker subprocesses
    that could only record `n/a`. `probe_torch_grouped` inside the worker
    stays as the second line of defence: a device that passes this gate but
    whose torch build lacks the operator still records the cell as
    `n/a: ...` instead of failing the run. If the capability cannot be read
    at all (no CUDA device visible) no torch-native cell is generated, since
    no cell could run anyway.
    """
    try:
        import torch
        major = torch.cuda.get_device_capability(0)[0]
    except Exception:
        return ()
    return tuple(i for i in TORCH_IMPLS if major >= TORCH_IMPL_MIN_MAJOR[i])

# fso masked grouped path: m_cap = pad4(M) slabs fit 32 GB through M=4096
# (peak ~7 GB at the layer cell); M=8192 still waits on a slab-free entry.
# 96/256 are off-grid sweep points (dense E33: off-grid mid-M is where
# cascade bugs hide).
FSO_M = sorted(set([m for m in M_GRID if m <= 4096] + [96, 256]))
# sm_90 expert-sorted contiguous layer (moe_layer_fp8_sm90): no slab, the full
# grid incl. M=8192 fits.
FSO_SM90_M = sorted(set(M_GRID + [96, 256]))


def make_cells():
    cells = []
    for proj in PROJ:
        for impl in ("dg_fp8_cont", "dg_bf16_cont"):
            cells += [dict(impl=impl, proj=proj, M=m) for m in M_GRID]
        for impl in ("dg_fp8_masked", "dg_bf16_masked"):
            cells += [dict(impl=impl, proj=proj, M=m) for m in DECODE_M]
        cells += [dict(impl="fso_mxfp8_grouped", proj=proj, M=m) for m in FSO_M]
    for impl in ("triton_bf16", "triton_fp8b"):
        cells += [dict(impl=impl, proj="layer", M=m) for m in M_GRID]
    cells += [dict(impl="dg_fp8_layer", proj="layer", M=m) for m in DECODE_M]
    cells += [dict(impl="fso_mxfp8_layer", proj="layer", M=m) for m in FSO_M]
    cells += [dict(impl="fso_bsfp8_layer", proj="layer", M=m) for m in FSO_SM90_M]
    torch_impls = torch_impls_for_device()
    for impl in torch_impls:
        cells += [dict(impl=impl, proj="layer", M=m) for m in M_GRID]
    if SHARED_INTER:
        cells += [dict(impl="fso_mxfp8_layer_shared", proj="layer", M=m) for m in FSO_M]
        cells += [dict(impl="fso_bsfp8_layer_shared", proj="layer", M=m) for m in FSO_SM90_M]
        for impl in torch_impls:
            cells += [dict(impl=impl + "_shared", proj="layer", M=m) for m in M_GRID]
    return cells


def build_fso_routing(topk_ids, topk_w, m_cap):
    """topk routing -> masked-layout index tensors (capture-safe torch ops).

    NOTE: torch.bincount is NOT capture-safe (it sizes its output from
    input.max(), a device->host sync); the zeros+scatter_add_ histogram is.
    """
    import torch
    M, topk = topk_ids.shape
    flat_e = topk_ids.flatten().long()
    order = torch.argsort(flat_e, stable=True)
    counts = torch.zeros(E, device="cuda", dtype=torch.int64)
    counts.scatter_add_(0, flat_e, torch.ones_like(flat_e))
    cum_excl = torch.cumsum(counts, 0) - counts
    ranks_sorted = torch.arange(M * topk, device="cuda") - cum_excl[flat_e[order]]
    slot_sorted = flat_e[order] * m_cap + ranks_sorted
    row_map = torch.full((E * m_cap,), -1, device="cuda", dtype=torch.int32)
    row_map[slot_sorted] = (order // topk).int()
    slot_of_flat = torch.empty(M * topk, device="cuda", dtype=torch.int64)
    slot_of_flat[order] = slot_sorted
    return counts.int(), row_map, slot_of_flat


# --------------------------------------------------------------------------
# torch-native grouped-MoE baselines (no fso, no deep_gemm, no sglang)
# --------------------------------------------------------------------------
#
# torch 2.11 exposes two grouped GEMMs that take a device-side `offs` tensor
# holding the exclusive-end row offset of every group, so the group sizes may
# be data dependent and the whole layer stays capturable:
#
#   torch._grouped_mm(mat_a[R,K], mat_b[G,K,N], offs=offs)            -> bf16
#   F.scaled_grouped_mm(mat_a, mat_b, scale_a, recipe, scale_b, ...)  -> MXFP8
#
# For the MXFP8 form the activation rows themselves need no padding, but the
# activation *scales* do: the kernel reads group g's scales from a cuBLAS
# 128x4 "blocked" tile array whose base element is
#     sum_{h<g} round_up(count_h, 128) * round_up(K/32, 4),
# i.e. every group's scale block is rounded up to 128 rows. Verified on the
# B300 against an FP32 per-expert reference for group sizes 128-aligned,
# 32-aligned and completely unaligned ([10, 30, 50, 70]) as well as for empty
# groups; a single un-grouped `to_blocked` over all rows is only correct when
# every group count happens to be a multiple of 128.
#
# The layer therefore lays the sorted rows out densely (no row padding, so the
# GEMM does exactly M*topk rows of useful work) and lays the scales out into a
# 128-row-per-group padded buffer whose capacity is the worst case
# `rows + 127 * n_active_experts`. `to_blocked` over that buffer then produces
# exactly the concatenation of the per-group blocked blocks.
SCALE_ROW_ALIGN = 128   # rows per group in the blocked activation-scale array


def scale_rows_cap(rows, n_groups):
    """Worst case of sum_g round_up(count_g, 128) for `rows` rows over `n_groups`."""
    n_active = min(n_groups, max(rows, 1))
    cap = rows + (SCALE_ROW_ALIGN - 1) * n_active
    return (cap + SCALE_ROW_ALIGN - 1) // SCALE_ROW_ALIGN * SCALE_ROW_ALIGN


def silu_chunk_mul(gu):
    """SwiGLU on a chunked [gate|up] tensor (compiled by the callers)."""
    import torch.nn.functional as F
    g_, u_ = gu.chunk(2, dim=-1)
    return F.silu(g_) * u_


def probe_torch_grouped(kind):
    """Tiny grouped-GEMM call so an unsupported arch/build is a recorded cell.

    Raises RuntimeError('n/a: ...') which the orchestrator stores in the row's
    `error` field, the same way any other failing cell is recorded.
    """
    import torch
    import torch.nn.functional as F
    try:
        G, K, N = 2, 128, 128
        w = torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16) / 16.0
        a = torch.randn(4, K, device="cuda", dtype=torch.bfloat16) * 0.1
        offs = torch.tensor([2, 4], device="cuda", dtype=torch.int32)
        if kind == "bf16":
            torch._grouped_mm(a, w.transpose(-2, -1), offs=offs,
                              out_dtype=torch.bfloat16)
        else:
            from torch.testing._internal.common_quantized import to_mxfp, to_blocked
            from torch._C import _ScalingType as ST, _SwizzleType as SW
            ws, wq = to_mxfp(w, 32, "mxfp8")
            wsb = torch.stack([to_blocked(ws[g]) for g in range(G)])
            asc, aq = to_mxfp(a.contiguous(), 32, "mxfp8")
            sp = torch.zeros(2 * SCALE_ROW_ALIGN, K // 32, device="cuda",
                             dtype=torch.uint8)
            sp[:2] = asc[:2].view(torch.uint8)
            sp[SCALE_ROW_ALIGN:SCALE_ROW_ALIGN + 2] = asc[2:].view(torch.uint8)
            sab = to_blocked(sp).view(2 * SCALE_ROW_ALIGN, K // 32).view(
                torch.float8_e8m0fnu)
            F.scaled_grouped_mm(aq, wq.transpose(-2, -1), sab, ST.BlockWise1x32,
                                wsb, ST.BlockWise1x32,
                                swizzle_a=SW.SWIZZLE_32_4_4,
                                swizzle_b=SW.SWIZZLE_32_4_4, offs=offs,
                                output_dtype=torch.bfloat16)
        torch.cuda.synchronize()
    except Exception as e:                       # noqa: BLE001 - reported as a cell
        raise RuntimeError(
            f"n/a: torch {kind} grouped mm unsupported on this device/build "
            f"({type(e).__name__}: {str(e)[:160]})")


def build_torch_moe_layer(kind, M, hidden, w13, w2, topk_ids, topk_w):
    """Whole routed MoE layer with torch ops only; returns (fn, info).

    kind == "mxfp8": F.scaled_grouped_mm with 1x32 block scales.
    kind == "bf16" : torch._grouped_mm, unquantized reference speed.

    Expert weights are quantized here, i.e. outside the timed graph, exactly
    like every other layer impl. Everything the returned closure does —
    routing from `topk_ids`, gather, activation quantize, both grouped GEMMs,
    SwiGLU, the weighted combine — happens inside the capture, so the timed
    boundary is the same as `fso_*_layer` and the triton / dg layer cells.
    """
    import torch
    import torch.nn.functional as F

    probe_torch_grouped(kind)
    rows = M * TOPK
    row_ix = torch.arange(rows, device="cuda")
    ones = torch.ones(rows, device="cuda", dtype=torch.int64)
    w_bf = topk_w.to(torch.bfloat16).unsqueeze(-1).contiguous()   # [M, topk, 1]
    silu = torch.compile(silu_chunk_mul, mode="default", dynamic=False)
    info = {}

    def combine_fn(dn, order):
        """expert-sorted rows -> per-token weighted sum, [M, HIDDEN] bf16.

        `order[i]` is the flat (token*topk + slot) index of sorted row i, so
        scattering the rows back and reducing over the topk axis is the same
        combine an `index_add_` would do. It is written this way because
        `index_add_` on the [M*topk, HIDDEN] source is a bf16 atomic scatter
        and measured 1348 us at M = 8192 against 217 us for this form
        (scratch/combine_probe2.py in this run directory); it is also more
        accurate, since the reduction accumulates in FP32 instead of
        round-tripping every partial sum through bf16.
        """
        t = torch.empty_like(dn)
        t.index_copy_(0, order, dn)
        return (t.view(M, TOPK, HIDDEN) * w_bf).sum(1, dtype=torch.float32).to(
            torch.bfloat16)

    combine = torch.compile(combine_fn, mode="default", dynamic=False)

    def route():
        """topk_ids -> (expert-sorted row order, per-expert counts, sorted expert id).

        `counts` is built with zeros+scatter_add_ rather than torch.bincount
        because bincount sizes its output from input.max(), a device->host
        sync that cannot be captured.
        """
        flat_e = topk_ids.flatten().long()
        order = torch.argsort(flat_e, stable=True)
        counts = torch.zeros(E, device="cuda", dtype=torch.int64)
        counts.scatter_add_(0, flat_e, ones)
        return order, counts, flat_e.index_select(0, order)

    if kind == "bf16":
        b13 = w13.transpose(-2, -1)      # [E, K, N], column major in last 2 dims
        b2 = w2.transpose(-2, -1)

        def layer_fn():
            order, counts, _ = route()
            offs = torch.cumsum(counts, 0).to(torch.int32)
            src = order // TOPK
            xg = hidden.index_select(0, src)
            gu = torch._grouped_mm(xg, b13, offs=offs, out_dtype=torch.bfloat16)
            dn = torch._grouped_mm(silu(gu), b2, offs=offs,
                                   out_dtype=torch.bfloat16)
            return combine(dn, order)

        return layer_fn, info

    from torch.testing._internal.common_quantized import to_mxfp, to_blocked
    from torch._C import _ScalingType as ST, _SwizzleType as SW

    sw13, q13 = to_mxfp(w13.contiguous(), 32, "mxfp8")
    sw2, q2 = to_mxfp(w2.contiguous(), 32, "mxfp8")
    sb13 = torch.stack([to_blocked(sw13[e]) for e in range(E)])
    sb2 = torch.stack([to_blocked(sw2[e]) for e in range(E)])
    b13, b2 = q13.transpose(-2, -1), q2.transpose(-2, -1)
    cap = scale_rows_cap(rows, E)
    k32_gu, k32_dn = HIDDEN // 32, INTER // 32
    info["scale_rows_cap"] = cap

    def quant_blocked(t, dest, k32):
        s, q = to_mxfp(t.contiguous(), 32, "mxfp8")
        s_pad = torch.zeros(cap, k32, device="cuda", dtype=torch.uint8)
        s_pad.index_copy_(0, dest, s.view(torch.uint8))
        return q, to_blocked(s_pad).view(cap, k32).view(torch.float8_e8m0fnu)

    qb = torch.compile(quant_blocked, mode="default", dynamic=False)

    def layer_fn():
        order, counts, e_sorted = route()
        csum = torch.cumsum(counts, 0)
        offs = csum.to(torch.int32)
        start = csum - counts
        padded = (counts + (SCALE_ROW_ALIGN - 1)) // SCALE_ROW_ALIGN * SCALE_ROW_ALIGN
        base = torch.cumsum(padded, 0) - padded
        dest = (base.index_select(0, e_sorted) + row_ix
                - start.index_select(0, e_sorted))
        src = order // TOPK
        xq, xs = qb(hidden.index_select(0, src), dest, k32_gu)
        gu = F.scaled_grouped_mm(xq, b13, xs, ST.BlockWise1x32,
                                 sb13, ST.BlockWise1x32,
                                 swizzle_a=SW.SWIZZLE_32_4_4,
                                 swizzle_b=SW.SWIZZLE_32_4_4,
                                 offs=offs, output_dtype=torch.bfloat16)
        hq, hs = qb(silu(gu), dest, k32_dn)
        dn = F.scaled_grouped_mm(hq, b2, hs, ST.BlockWise1x32,
                                 sb2, ST.BlockWise1x32,
                                 swizzle_a=SW.SWIZZLE_32_4_4,
                                 swizzle_b=SW.SWIZZLE_32_4_4,
                                 offs=offs, output_dtype=torch.bfloat16)
        return combine(dn, order)

    return layer_fn, info


def build_torch_shared_expert(kind, w13s, w2s):
    """Dense SwiGLU shared expert (Family C), torch ops only.

    MXFP8 uses `F.scaled_mm` with the same `to_mxfp` + `to_blocked` quantize
    the dense MLP bench drives as `smm_fast`; bf16 uses `F.linear`.
    """
    import torch
    import torch.nn.functional as F
    silu = torch.compile(silu_chunk_mul, mode="default", dynamic=False)

    if kind == "bf16":
        def shared_fn(x):
            return F.linear(silu(F.linear(x, w13s)), w2s)
        return shared_fn

    from torch.testing._internal.common_quantized import to_mxfp, to_blocked
    from torch._C import _ScalingType as ST, _SwizzleType as SW
    s13s, q13s = to_mxfp(w13s.contiguous(), 32, "mxfp8")
    s2s, q2s = to_mxfp(w2s.contiguous(), 32, "mxfp8")
    s13s_b, s2s_b = to_blocked(s13s), to_blocked(s2s)
    w13s_t, w2s_t = q13s.t(), q2s.t()

    def quant_blocked(t):
        s, q = to_mxfp(t.contiguous(), 32, "mxfp8")
        return q, to_blocked(s)

    qb = torch.compile(quant_blocked, mode="default", dynamic=False)

    def shared_fn(x):
        xq, sx = qb(x)
        gu = F.scaled_mm(xq, w13s_t, sx, ST.BlockWise1x32, s13s_b,
                         ST.BlockWise1x32, swizzle_a=SW.SWIZZLE_32_4_4,
                         swizzle_b=SW.SWIZZLE_32_4_4, output_dtype=torch.bfloat16)
        hq, sh = qb(silu(gu))
        return F.scaled_mm(hq, w2s_t, sh, ST.BlockWise1x32, s2s_b,
                           ST.BlockWise1x32, swizzle_a=SW.SWIZZLE_32_4_4,
                           swizzle_b=SW.SWIZZLE_32_4_4, output_dtype=torch.bfloat16)

    return shared_fn


# --------------------------------------------------------------------------
# Worker: one cell in one process
# --------------------------------------------------------------------------

def busy_warm(ms):
    import torch
    if ms <= 0:
        return
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    t0 = torch.cuda.Event(True)
    t1 = torch.cuda.Event(True)
    t0.record()
    while True:
        for _ in range(8):
            a = a @ a * 1e-3
        t1.record()
        torch.cuda.synchronize()
        if t0.elapsed_time(t1) >= ms:
            break


def graph_time_us(fn, warmup=15, stream_warm=3, reps=3, iters=50):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(stream_warm):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    vals = []
    for _ in range(reps):
        t0 = torch.cuda.Event(True)
        t1 = torch.cuda.Event(True)
        t0.record()
        for _ in range(iters):
            g.replay()
        t1.record()
        torch.cuda.synchronize()
        vals.append(t0.elapsed_time(t1) * 1000.0 / iters)
    return statistics.median(vals)


def make_routing(M, seed):
    """topk_ids [M, TOPK] int32 (no replacement), topk_weights [M, TOPK] fp32."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(M)])
    w = torch.softmax(torch.randn(M, TOPK, generator=g, dtype=torch.float32), dim=-1)
    return ids.to("cuda", torch.int32), w.cuda()


def quant_weights_fp8(w):
    """[E, N, K] bf16 -> fp8 + [E, N/128, K/128] fp32 block scales."""
    import torch
    import deep_gemm as dg
    Ecnt, N, K = w.shape
    w_fp8 = torch.empty_like(w, dtype=torch.float8_e4m3fn)
    w_sf = torch.empty(Ecnt, (N + 127) // 128, (K + 127) // 128,
                       device=w.device, dtype=torch.float32)
    for e in range(Ecnt):
        q, s = dg.per_block_cast_to_fp8(w[e], use_ue8m0=False)
        w_fp8[e].copy_(q)
        w_sf[e].copy_(s)
    return w_fp8, w_sf


def grouped_ref(x_rows, w, row_expert):
    """fp32 reference: out[i] = x_rows[i] @ w[row_expert[i]].T (bf16 operands)."""
    import torch
    out = torch.empty(x_rows.shape[0], w.shape[1], device=x_rows.device,
                      dtype=torch.float32)
    for e in row_expert.unique().tolist():
        sel = row_expert == e
        out[sel] = x_rows[sel].float() @ w[e].float().t()
    return out


def moe_layer_ref(hidden, w13, w2, topk_ids, topk_w):
    """fp32 whole-layer reference (chunked gate|up, silu, weighted combine)."""
    import torch
    M = hidden.shape[0]
    out = torch.zeros(M, HIDDEN, device=hidden.device, dtype=torch.float32)
    hf = hidden.float()
    for e in torch.unique(topk_ids).tolist():
        tok, slot = (topk_ids == e).nonzero(as_tuple=True)
        gu = hf[tok] @ w13[e].float().t()
        act = torch.nn.functional.silu(gu[:, :INTER]) * gu[:, INTER:]
        out.index_add_(0, tok,
                       (act @ w2[e].float().t()) * topk_w[tok, slot, None].float())
    return out


def cos_sim(a, b):
    import torch
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0).item()


def init_sglang_shim():
    """Minimal global ServerArgs so sglang kernel wrappers are importable."""
    import dataclasses
    from sglang.srt.server_args import (ServerArgs,
                                        set_global_server_args_for_scheduler)
    sa = ServerArgs.__new__(ServerArgs)  # skip __post_init__ side effects
    for f in dataclasses.fields(ServerArgs):
        if f.default is not dataclasses.MISSING:
            setattr(sa, f.name, f.default)
        elif f.default_factory is not dataclasses.MISSING:
            setattr(sa, f.name, f.default_factory())
        else:
            setattr(sa, f.name, None)
    set_global_server_args_for_scheduler(sa)


def run_worker(cell):
    import torch
    try:
        import deep_gemm as dg   # reference impls only; fso_* cells run without it
    except ImportError:
        dg = None

    impl, proj, M = cell["impl"], cell["proj"], cell["M"]
    torch.manual_seed(M * 1009 + len(proj) * 17 + 5)
    busy_warm(int(os.environ.get("FSO_BENCH_WARM_MS", "0")))

    topk_ids, topk_w = make_routing(M, seed=M * 7919 + 3)
    counts = torch.bincount(topk_ids.flatten().long(), minlength=E).int()
    active = int((counts > 0).sum())
    rows = M * TOPK

    result = dict(cell)
    result["active_experts"] = active

    if impl in KERNEL_IMPLS:
        N, K = PROJ[proj]
        w = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16) / math.sqrt(K)
        result["flops"] = 2.0 * rows * N * K
        wbytes = N * K * (1 if ("fp8" in impl or "mxfp8" in impl) else 2)
        result["w_bytes_active"] = active * wbytes

        if impl == "fso_mxfp8_grouped":
            import fish_scales_ops as fso
            m_cap = (M + 3) // 4 * 4
            expected_m = max(1, (rows + E - 1) // E)
            # Upper bound on how many experts can hold at least one row: a
            # top-k router gives an expert at most one row per token, so
            # min(M * TOPK, E) bounds it whatever the draw is. It is host
            # static per (M, TOPK, E), so a graph captured at this M stays
            # valid for every routing draw at that M.
            max_active_groups = min(M * TOPK, E)
            result["m_cap"] = m_cap
            result["max_active_groups"] = max_active_groups
            # layer-style prep: routing kernel + flat-pair gather-quant.
            # `with_slots=True` makes the routing kernel emit the packed
            # active-expert list as well, which is what lets the sm_100
            # slot-bound decode route skip building it again per GEMM. The list
            # is only worth its block-wide compaction where that route is
            # actually taken, so the library is asked first: outside the decode
            # band, and on every architecture that has no slot route, the cell
            # runs the routing kernel without it.
            want_slots = fso.gemm.mxfp8_grouped_slot_possible(
                m_cap, N, K, E, max_active_groups)
            result["with_slots"] = int(want_slots)
            # The pointer-array route's counterpart (run
            # b300_round3_20260922/M-A4): where the cascade serves the call,
            # the routing kernel also emits the per-group (rows, N, K) triples
            # and the GEMM launches without its argument-preparation kernel.
            want_ps = fso.gemm.mxfp8_grouped_problem_shapes_consumed(
                m_cap, N, K, E, max_active_groups)
            result["with_ps"] = int(want_ps)
            x_tok = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
            routing = fso.gemm.moe_build_routing(
                topk_ids, E, m_cap, with_slots=want_slots,
                problem_shapes_for=[(N, K)] if want_ps else None)
            masked_dev, row_map, slot_of_flat = routing[:3]
            slot_to_expert = routing[3] if want_slots else None
            problem_shapes = routing[-1][0] if want_ps else None
            a_fp8, sa = fso.gemm.quantize_1x32_grouped_gather_fp8(
                x_tok, slot_of_flat, TOPK, E, m_cap)
            w_fp8, sw = fso.gemm.quantize_moe_weights_1x32_fp8(w)
            fn = lambda: fso.gemm.linear_mxfp8_grouped_masked(
                a_fp8, w_fp8, sa, sw, masked_dev, expected_m, max_active_groups,
                slot_to_expert, problem_shapes)
            y = fn()
            torch.cuda.synchronize()
            cs, n_cs = 0.0, 0
            rm = row_map.cpu().view(E, m_cap)   # row_map[slot] = source token
            mm = masked_dev.cpu()
            for e in range(E):
                n_e = int(mm[e])
                if not n_e:
                    continue
                src = rm[e, :n_e].long()
                ref = x_tok[src].float() @ w[e].float().t()
                cs += cos_sim(y[e, :n_e], ref) * n_e
                n_cs += n_e
            result["cos"] = cs / max(n_cs, 1)
        elif impl.endswith("_cont"):
            # per-expert segments padded to the 128-row contiguous alignment;
            # padding rows are zero and point at the same expert (the layout
            # sglang's contiguous path uses), so they are computed and thrown
            # away -- the useful-flops tflops field exposes that waste.
            align = dg.get_mk_alignment_for_contiguous_layout()
            pad = torch.where(counts > 0, (counts + align - 1) // align * align,
                              torch.zeros_like(counts))
            Mc = int(pad.sum())
            x = torch.zeros(Mc, K, device="cuda", dtype=torch.bfloat16)
            m_indices = torch.empty(Mc, device="cuda", dtype=torch.int32)
            row_expert = torch.empty(Mc, device="cuda", dtype=torch.int64)
            off = 0
            for e in range(E):
                if int(pad[e]) == 0:
                    continue
                n_e = int(counts[e])
                x[off:off + n_e] = torch.randn(n_e, K, device="cuda",
                                               dtype=torch.bfloat16) * 0.1
                m_indices[off:off + int(pad[e])] = e
                row_expert[off:off + int(pad[e])] = e
                off += int(pad[e])
            d = torch.empty(Mc, N, device="cuda", dtype=torch.bfloat16)
            result["rows_padded"] = Mc
            if impl == "dg_fp8_cont":
                x_fp8, x_sf = dg.per_token_cast_to_fp8(x, use_ue8m0=False)
                w_fp8, w_sf = quant_weights_fp8(w)
                fn = lambda: dg.m_grouped_fp8_gemm_nt_contiguous(
                    (x_fp8, x_sf), (w_fp8, w_sf), d, m_indices)
            else:
                fn = lambda: dg.m_grouped_bf16_gemm_nt_contiguous(x, w, d, m_indices)
            fn()
            torch.cuda.synchronize()
            result["cos"] = cos_sim(d, grouped_ref(x, w, row_expert))
        else:
            # production masked layout: m_max padded like moe_ep_deepgemm_preprocess
            m_max = (M // 256 + 1) * 256
            expected_m = (rows - 1) // E + 1
            a = torch.zeros(E, m_max, K, device="cuda", dtype=torch.bfloat16)
            for e in range(E):
                n_e = int(counts[e])
                if n_e:
                    a[e, :n_e] = torch.randn(n_e, K, device="cuda",
                                             dtype=torch.bfloat16) * 0.1
            d = torch.empty(E, m_max, N, device="cuda", dtype=torch.bfloat16)
            result["m_max"] = m_max
            if impl == "dg_fp8_masked":
                a_fp8 = torch.empty_like(a, dtype=torch.float8_e4m3fn)
                a_sf = torch.empty(E, m_max, K // 128, device="cuda",
                                   dtype=torch.float32)
                for e in range(E):
                    q, s = dg.per_token_cast_to_fp8(a[e], use_ue8m0=False)
                    a_fp8[e].copy_(q)
                    a_sf[e].copy_(s)
                w_fp8, w_sf = quant_weights_fp8(w)
                fn = lambda: dg.fp8_m_grouped_gemm_nt_masked(
                    (a_fp8, a_sf), (w_fp8, w_sf), d, counts, expected_m)
            else:
                fn = lambda: dg.m_grouped_bf16_gemm_nt_masked(
                    a, w, d, counts, expected_m)
            fn()
            torch.cuda.synchronize()
            cs, n_cs = 0.0, 0
            for e in range(E):
                n_e = int(counts[e])
                if not n_e:
                    continue
                ref = a[e, :n_e].float() @ w[e].float().t()
                cs += cos_sim(d[e, :n_e], ref) * n_e
                n_cs += n_e
            result["cos"] = cs / max(n_cs, 1)

    else:  # whole-layer impls
        if impl.startswith("triton") or impl == "dg_fp8_layer":
            init_sglang_shim()   # sglang wrappers; the fso layers need neither sglang nor deep_gemm
        hidden = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1
        w13 = torch.randn(E, 2 * INTER, HIDDEN, device="cuda",
                          dtype=torch.bfloat16) / math.sqrt(HIDDEN)
        w2 = torch.randn(E, HIDDEN, INTER, device="cuda",
                         dtype=torch.bfloat16) / math.sqrt(INTER)
        result["flops"] = 2.0 * rows * (2 * INTER * HIDDEN + HIDDEN * INTER)
        wbytes_e = (2 * INTER * HIDDEN + HIDDEN * INTER)
        bf16_weights = impl in ("triton_bf16", "torch_grouped_bf16_layer",
                                "torch_grouped_bf16_layer_shared")
        result["w_bytes_active"] = active * wbytes_e * (2 if bf16_weights else 1)
        ref = moe_layer_ref(hidden, w13, w2, topk_ids, topk_w)

        # `*_layer_shared`: whole MoE block = routed layer + the always-on shared
        # expert (dense SwiGLU MLP on every token, same FP8 path as the family's
        # dense GEMMs) + residual add, all inside the one captured graph.
        shared = impl.endswith("_shared")
        base_impl = impl[: -len("_shared")] if shared else impl
        shared_fn = None
        if shared:
            import torch.nn.functional as F
            if base_impl.startswith("fso_"):
                import fish_scales_ops as fso
            assert SHARED_INTER, f"{MODEL} has no shared expert"
            Is = SHARED_INTER
            w13s = torch.randn(2 * Is, HIDDEN, device="cuda", dtype=torch.bfloat16) / math.sqrt(HIDDEN)
            w2s = torch.randn(HIDDEN, Is, device="cuda", dtype=torch.bfloat16) / math.sqrt(Is)
            result["flops"] += 2.0 * M * (2 * Is * HIDDEN + HIDDEN * Is)
            result["w_bytes_active"] += (2 * Is * HIDDEN + HIDDEN * Is) * (2 if bf16_weights else 1)
            result["shared_inter"] = Is
            gus = hidden.float() @ w13s.float().t()
            ref = ref + (F.silu(gus[:, :Is]) * gus[:, Is:]) @ w2s.float().t()
            if base_impl in TORCH_IMPLS:
                shared_fn = build_torch_shared_expert(
                    "bf16" if base_impl == "torch_grouped_bf16_layer" else "mxfp8",
                    w13s, w2s)
            elif base_impl == "fso_mxfp8_layer":
                w13s_q, s13s = fso.gemm.quantize_1x32_fp8(w13s)
                w2s_q, s2s = fso.gemm.quantize_1x32_fp8(w2s)

                def shared_fn(x):
                    xq, sx = fso.gemm.quantize_1x32_fp8(x)
                    gu = fso.gemm.linear_mxfp8(xq, w13s_q, sx, s13s)
                    hq, sh = fso.gemm.silu_chunk_mul_quantize_1x32_fp8(gu)
                    return fso.gemm.linear_mxfp8(hq, w2s_q, sh, s2s)
            else:  # sm_90 block-FP8: same closure as bench_qwen3_4b_mlp_forward's BSFP8 path
                w13s_q, s13s = fso.gemm.quantize_128x128_fp8(w13s)
                w2s_q, s2s = fso.gemm.quantize_128x128_fp8(w2s)

                @torch.compile(mode="default", dynamic=False)
                def _silu_chunk_mul(gu):
                    g_, u_ = gu.chunk(2, dim=-1)
                    return F.silu(g_) * u_

                def shared_fn(x):
                    xq, sx = fso.gemm.quantize_1x128_fp8(x, use_ue8m0=False)
                    gu = fso.gemm.linear_fp8(xq, w13s_q, sx, s13s)
                    h = _silu_chunk_mul(gu)
                    hq, sh = fso.gemm.quantize_1x128_fp8(h, use_ue8m0=False)
                    return fso.gemm.linear_fp8(hq, w2s_q, sh, s2s)

        if impl.startswith("triton"):
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
                fused_experts)
            from sglang.srt.layers.moe.topk import StandardTopKOutput
            import dataclasses as _dc
            cfg_kw = dict(num_experts=E, num_local_experts=E, top_k=TOPK,
                          hidden_size=HIDDEN,
                          intermediate_size_per_partition=INTER,
                          activation="silu", is_gated=True, inplace=False,
                          gate_up_interleaved=False)  # chunked [gate|up]
            fields = {f.name for f in _dc.fields(MoeRunnerConfig)}
            cfg = MoeRunnerConfig(**{k: v for k, v in cfg_kw.items()
                                     if k in fields})
            tk = StandardTopKOutput(topk_w, topk_ids,
                                    torch.empty(M, E, device="cuda"))
            if impl == "triton_bf16":
                fn = lambda: fused_experts(hidden, w13, w2, tk, cfg)
            else:
                w13_fp8, w13_sf = quant_weights_fp8(w13)
                w2_fp8, w2_sf = quant_weights_fp8(w2)
                fn = lambda: fused_experts(
                    hidden, w13_fp8, w2_fp8, tk, cfg, use_fp8_w8a8=True,
                    w1_scale=w13_sf, w2_scale=w2_sf, block_shape=[128, 128])
            out_holder = {}
            fn0 = fn
            def fn():
                out_holder["out"] = fn0()
            fn()
            torch.cuda.synchronize()
            result["cos"] = cos_sim(out_holder["out"], ref)
        elif base_impl in TORCH_IMPLS:
            # torch-only layer: routing (argsort of topk_ids + histogram) ->
            # gather -> [quantize to MXFP8 + blocked scales] -> grouped gate_up
            # -> SwiGLU -> [quantize] -> grouped down -> weighted combine, all
            # inside the captured graph. Same timed boundary as the fso and
            # triton / dg layer cells: only the expert weights are prepared
            # outside.
            layer_fn, info = build_torch_moe_layer(
                "bf16" if base_impl == "torch_grouped_bf16_layer" else "mxfp8",
                M, hidden, w13, w2, topk_ids, topk_w)
            result.update(info)
            if "scale_rows_cap" in info:
                # what the 128-row-per-group blocked scale layout actually
                # occupies at this routing, vs the worst-case allocation
                counts_padded = ((counts.long() + SCALE_ROW_ALIGN - 1)
                                 // SCALE_ROW_ALIGN * SCALE_ROW_ALIGN)
                result["scale_rows_used"] = int(counts_padded.sum())
            out_holder = {}
            if shared_fn is not None:
                routed_fn = layer_fn

                def layer_fn():
                    return routed_fn() + shared_fn(hidden)
            def fn():
                out_holder["out"] = layer_fn()
            fn()
            torch.cuda.synchronize()
            result["cos"] = cos_sim(out_holder["out"], ref)
        elif base_impl == "fso_mxfp8_layer":
            # fso M1 composed layer, 6 kernels total and zero torch-op glue:
            # moe_build_routing + gather-quant + grouped gate_up +
            # grouped silu-quant + grouped down + moe_combine. Routing is
            # derived from topk_ids ON DEVICE inside the timed graph, same
            # boundary as the triton / dg layer cells.
            import fish_scales_ops as fso
            m_cap = (M + 3) // 4 * 4
            expected_m = max(1, (rows + E - 1) // E)
            # See the kernel cell above: min(M * TOPK, E) is the host-static
            # upper bound on the number of experts that can hold a row, which
            # is what the sm_100 grouped dispatcher needs to size the decode
            # route's grid. `expected_m` cannot supply it (it is
            # ceil(rows / E) and equals 1 for every decode M at E = 128).
            max_active_groups = min(M * TOPK, E)
            result["m_cap"] = m_cap
            result["max_active_groups"] = max_active_groups
            # The fused FC1 (run b300_mxfp8_20260917/M-E2): one grouped GEMM
            # whose epilogue emits MXFP8(silu(gate) * up), so the separate
            # SwiGLU kernel and the bf16 [E, m_cap, 2*I] intermediate both
            # disappear. Two host-static decisions, both taken by the library
            # so the bench cannot invent a rule of its own:
            #   * `available` is the LOAD-time one. The fused op needs gate/up
            #     interleaved weight rows, and a model holds one layout, so it
            #     decides how w13 is quantised. It is false when
            #     FSO_FC1_FUSED=0, which is exactly the control arm: the layer
            #     is then byte-for-byte the one this round started from.
            #   * `fused_route` is the PER-CALL one. Both grouped routes carry
            #     a fused FC1 now (run b300_mxfp8_20260917/M-A2 added the
            #     swap-orientation one), so it answers yes on both sides of the
            #     route boundary and what it really selects is which fused
            #     kernel the call lands on. Where it answers no the layer keeps
            #     the unfused FC1 on interleaved weights, which is what the
            #     SwiGLU kernel's `pairwise` flag is for.
            n_w = 2 * INTER
            fc1_interleaved = fso.gemm.mxfp8_grouped_swiglu_available(n_w, HIDDEN)
            fc1_fused = fc1_interleaved and fso.gemm.mxfp8_grouped_swiglu_fused_route(
                m_cap, n_w, HIDDEN, E, max_active_groups)
            result["fc1_interleaved"] = int(fc1_interleaved)
            result["fc1_fused"] = int(fc1_fused)
            w13_fp8, sw13 = fso.gemm.quantize_moe_weights_1x32_fp8(
                w13, w13_interleave=fc1_interleaved)
            w2_fp8, sw2 = fso.gemm.quantize_moe_weights_1x32_fp8(w2)

            # The routing kernel emits the packed active-expert list in the
            # same launch (run b300_mxfp8_20260917/M-I1). Where the dispatcher
            # takes the slot-bound decode route it then launches only the GEMM.
            # Where it does not, nothing reads the list, and building it is a
            # block-wide compaction the layer pays for nothing — so the library
            # is asked, once per cell, whether EITHER of the layer's two grouped
            # GEMMs would take that route (FC1 is 2*INTER x HIDDEN, FC2 is
            # HIDDEN x INTER), and `with_slots` follows that answer. On sm_120,
            # and on sm_100 above the decode band, the answer is no and the
            # routing kernel does less work than it did.
            # Asked for the kernel each GEMM will actually run: the fused FC1
            # has a shorter row-capacity clause than the plain grouped GEMM
            # (run b300_round3_20260922/M-A3).
            want_slots = (
                fso.gemm.mxfp8_grouped_slot_possible(
                    m_cap, n_w, HIDDEN, E, max_active_groups, fused_swiglu=fc1_fused)
                or fso.gemm.mxfp8_grouped_slot_possible(
                    m_cap, HIDDEN, INTER, E, max_active_groups))
            result["with_slots"] = int(want_slots)
            # The pointer-array route's counterpart of the slot list (run
            # b300_round3_20260922/M-A4). Wherever the cascade serves a GEMM,
            # the routing kernel also emits that GEMM's per-group (rows, N, K)
            # triples in the pass that writes masked_m, and the GEMM then
            # launches without the argument-preparation kernel it used to run
            # ahead of itself: seven kernels per layer become five. Asked per
            # GEMM, so the decode band (slot route, no reader) and sm_120 keep
            # the routing kernel's old work exactly. Asked with the same
            # kernel flag as the slot query above, because the two answers
            # are complements only under the same flag: per GEMM exactly one
            # of the slot list and the problem shapes is then requested.
            want_ps = [
                fso.gemm.mxfp8_grouped_problem_shapes_consumed(
                    m_cap, n_w, HIDDEN, E, max_active_groups, fused_swiglu=fc1_fused),
                fso.gemm.mxfp8_grouped_problem_shapes_consumed(
                    m_cap, HIDDEN, INTER, E, max_active_groups)]
            ps_for = [nk for nk, want in zip([(n_w, HIDDEN), (HIDDEN, INTER)], want_ps) if want]
            result["with_ps"] = int(any(want_ps))

            def build_routing():
                r = fso.gemm.moe_build_routing(topk_ids, E, m_cap, with_slots=want_slots,
                                               problem_shapes_for=ps_for or None)
                ps = list(r[-1]) if ps_for else []
                ps1 = ps.pop(0) if want_ps[0] else None
                ps2 = ps.pop(0) if want_ps[1] else None
                return r[0], r[2], (r[3] if want_slots else None), ps1, ps2

            if fc1_fused:
                def layer_fn():
                    masked_dev, slot_of_flat, slot_to_expert, ps1, ps2 = build_routing()
                    hq, sh = fso.gemm.quantize_1x32_grouped_gather_fp8(
                        hidden, slot_of_flat, TOPK, E, m_cap)
                    dq, sd = fso.gemm.linear_mxfp8_grouped_masked_swiglu(
                        hq, w13_fp8, sh, sw13, masked_dev, expected_m,
                        max_active_groups, slot_to_expert, ps1)
                    dn = fso.gemm.linear_mxfp8_grouped_masked(
                        dq, w2_fp8, sd, sw2, masked_dev, expected_m,
                        max_active_groups, slot_to_expert, ps2)
                    return fso.gemm.moe_combine(dn, slot_of_flat, topk_w)
            else:
                def layer_fn():
                    masked_dev, slot_of_flat, slot_to_expert, ps1, ps2 = build_routing()
                    hq, sh = fso.gemm.quantize_1x32_grouped_gather_fp8(
                        hidden, slot_of_flat, TOPK, E, m_cap)
                    gu = fso.gemm.linear_mxfp8_grouped_masked(
                        hq, w13_fp8, sh, sw13, masked_dev, expected_m,
                        max_active_groups, slot_to_expert, ps1)
                    dq, sd = fso.gemm.silu_chunk_mul_quantize_1x32_grouped_fp8(
                        gu, slot_of_flat, pairwise=fc1_interleaved)
                    dn = fso.gemm.linear_mxfp8_grouped_masked(
                        dq, w2_fp8, sd, sw2, masked_dev, expected_m,
                        max_active_groups, slot_to_expert, ps2)
                    return fso.gemm.moe_combine(dn, slot_of_flat, topk_w)

            out_holder = {}
            if shared_fn is not None:
                routed_fn = layer_fn

                def layer_fn():
                    return routed_fn() + shared_fn(hidden)
            def fn():
                out_holder["out"] = layer_fn()
            fn()
            torch.cuda.synchronize()
            result["cos"] = cos_sim(out_holder["out"], ref)
        elif base_impl == "fso_bsfp8_layer":
            # fso sm_90 (H200) block-scale FP8 grouped layer via the single
            # moe_layer_fp8_sm90 dispatch entry: expert-sorted (contiguous)
            # 6-kernel layer, auto-routed by M (swap-AB block_n=16 for
            # M < MOE_SWAP_M_MAX, non-swap block_m=64 above). Same timed
            # boundary (routing derived from topk_ids on device in the graph).
            import fish_scales_ops as fso
            w13_fp8, sw13 = fso.gemm.quantize_moe_weights_1x128_fp8_sm90(w13)
            w2_fp8, sw2 = fso.gemm.quantize_moe_weights_1x128_fp8_sm90(w2)
            result["swap"] = int(M < fso.gemm.MOE_SWAP_M_MAX)

            def layer_fn():
                return fso.gemm.moe_layer_fp8_sm90(
                    hidden, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w)

            out_holder = {}
            if shared_fn is not None:
                routed_fn = layer_fn

                def layer_fn():
                    return routed_fn() + shared_fn(hidden)
            def fn():
                out_holder["out"] = layer_fn()
            fn()
            torch.cuda.synchronize()
            result["cos"] = cos_sim(out_holder["out"], ref)
        else:  # dg_fp8_layer: mirror sglang moe_runner/deep_gemm.py masked path
            from sglang.srt.layers import deep_gemm_wrapper
            from sglang.srt.layers.moe.ep_moe.kernels import (
                moe_ep_deepgemm_preprocess, post_reorder_triton_kernel)
            from sglang.srt.layers.moe.moe_runner.deep_gemm import (
                _varlen_deep_gemm_silu_mul_quant)
            w13_fp8, w13_sf = quant_weights_fp8(w13)
            w2_fp8, w2_sf = quant_weights_fp8(w2)
            out = torch.empty(M, HIDDEN, device="cuda", dtype=torch.bfloat16)

            def layer_fn():
                masked_m, expected_m, src2dst, h_q, h_sf = (
                    moe_ep_deepgemm_preprocess(
                        topk_ids, E, hidden, TOPK, [128, 128],
                        output_dtype=torch.float8_e4m3fn))
                n_grp, m_max, _ = h_q.shape
                if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
                    h_sf = deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(h_sf)
                gu = torch.empty(n_grp, m_max, 2 * INTER, device="cuda",
                                 dtype=torch.bfloat16)
                deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                    (h_q, h_sf), (w13_fp8, w13_sf), gu, masked_m, expected_m)
                di, di_sf = _varlen_deep_gemm_silu_mul_quant(
                    gu, masked_m, group_size=128, topk=TOPK)
                if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
                    di_sf = deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(di_sf)
                dn = torch.empty(n_grp, m_max, HIDDEN, device="cuda",
                                 dtype=torch.bfloat16)
                deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                    (di, di_sf), (w2_fp8, w2_sf), dn, masked_m, expected_m)
                post_reorder_triton_kernel[(M,)](
                    dn, out, src2dst, topk_ids, topk_w, TOPK, HIDDEN,
                    BLOCK_SIZE=512)
                return out

            fn = layer_fn
            fn()
            torch.cuda.synchronize()
            result["cos"] = cos_sim(out, ref)

    result["us"] = graph_time_us(fn)
    print("CELL " + json.dumps(result), flush=True)


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def derive(row):
    us = row["us"]
    row["tflops"] = row["flops"] / (us * 1e-6) / 1e12
    row["w_gbps"] = row["w_bytes_active"] / (us * 1e-6) / 1e9
    if row["proj"] == "layer":
        row["model_ms"] = us * LAYERS / 1e3
        row["moe_tok_s"] = 1e6 / (us * LAYERS)
    return row


def run_all(args):
    import torch
    cells = make_cells()
    if args.impls:
        keep = set(args.impls.split(","))
        cells = [c for c in cells if c["impl"] in keep]
    if args.Ms:
        keep_m = {int(m) for m in args.Ms.split(",")}
        cells = [c for c in cells if c["M"] in keep_m]
    if args.projs:
        keep_p = set(args.projs.split(","))
        cells = [c for c in cells if c["proj"] in keep_p]

    meta = dict(kind="meta", model=MODEL, E=E, topk=TOPK, hidden=HIDDEN,
                inter=INTER, layers=LAYERS,
                device=torch.cuda.get_device_name(0),
                torch=torch.__version__)
    rows = [meta]
    out_f = open(args.out, "w") if args.out else None
    if out_f:
        out_f.write(json.dumps(meta) + "\n")
        out_f.flush()

    for i, cell in enumerate(cells):
        cmd = [sys.executable, os.path.abspath(__file__), "--worker",
               "--model", MODEL, "--cell", json.dumps(cell)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("CELL ")), None)
        if line is None:
            err = (p.stderr or "").strip().splitlines()
            row = dict(cell, error=(err[-1] if err else f"rc={p.returncode}"))
            print(f"[{i+1}/{len(cells)}] {cell['impl']:14s} {cell['proj']:8s} "
                  f"M={cell['M']:<5d} FAILED: {row['error'][:120]}")
        else:
            row = derive(json.loads(line[5:]))
            if "Config file not found" in (p.stderr or ""):
                row["tuned_cfg"] = False
            extra = f" moe_tok/s={row['moe_tok_s']:8.1f}" if "moe_tok_s" in row else ""
            print(f"[{i+1}/{len(cells)}] {cell['impl']:14s} {cell['proj']:8s} "
                  f"M={cell['M']:<5d} {row['us']:9.2f} us  "
                  f"{row['tflops']:7.1f} TF  {row['w_gbps']:7.0f} GB/s  "
                  f"cos={row.get('cos', float('nan')):.5f}{extra}")
        rows.append(row)
        if out_f:
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
    if out_f:
        out_f.close()
        print(f"\nwrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--cell", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--impls", type=str, default=None,
                    help="comma list filter, e.g. dg_fp8_masked,triton_bf16")
    ap.add_argument("--Ms", type=str, default=None,
                    help="comma list filter, e.g. 1,8,128")
    ap.add_argument("--projs", type=str, default=None,
                    help="comma list filter, e.g. gate_up or down,layer")
    ap.add_argument("--model", choices=sorted(MODELS), default="qwen3-30a3",
                    help="routed-expert geometry (docs/perf/README.md §3): "
                         "qwen3-30a3 = Family B, qwen3.5-35a3 = Family C")
    args = ap.parse_args()
    set_model(args.model)
    if args.worker:
        run_worker(json.loads(args.cell))
    elif args.run:
        run_all(args)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
