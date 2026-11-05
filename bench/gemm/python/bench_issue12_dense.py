"""Issue #12 dense-projection bench: the audio-tower and LLM projections of the
realtime ASR stack in issue #12, timed the way a serving step pays for them.

What is measured. One cell is one (class, M): a projection class (N, K, bias or
not, how many times one tower call or one LLM step runs it) at one row count M.
Five arms run in every cell, each inside a CUDA graph:

* ``bf16``      -- ``torch.nn.functional.linear(x, w, bias)``: cuBLAS with the
  bias in its epilogue. On sm_120 the cuBLAS bf16 reduced-precision reduction is
  switched off (``allow_bf16_reduced_precision_reduction = False``), the setting
  the issue's RTX 5090 numbers use; elsewhere the torch default is kept.
* ``fso_chain`` -- today's serving chain: ``y = fso.dense.linear(x, dw)`` and,
  when the class has a bias, ``y = y + bias`` in torch. ``dw`` is
  ``fso.dense.prepare_weight(w, format=F)`` with F = "mxfp8" on sm_100/sm_103 and
  sm_120/sm_121 and "bsfp8" on sm_90.
* ``fso_gemm``  -- the GEMM alone on an activation quantized once outside the
  graph, no bias: ``fso.compat.linear_mxfp8(xq, dw.weight, sx, dw.scale)`` with
  ``xq, sx = fso.compat.quantize_1x32_fp8(x)`` on sm_100 and sm_120, and
  ``fso.compat.linear_fp8(xq, dw.weight, sx, dw.scale)`` with
  ``fso.compat.quantize_1x128_fp8(x)`` (sm_90 default, fp32 scales) on sm_90 --
  the pair ``bench_qwen3_4b_mlp.py`` drives on sm_90. Each cell records whether
  this output equals ``fso.dense.linear(x, dw)`` byte for byte
  (``gemm_eq_dense_linear``; on sm_90 also ``sm90_gemm_eq_chain``).
* ``fso_quant`` -- the activation quantize alone (the 1x32 quantize, or the sm_90
  1x128 quantize). On sm_90 the chain does not run this kernel: ``linear_qx``
  quantizes inside its own op, so quant + gemm is not the chain there.
* ``bias_add``  -- ``y + bias`` alone on a precomputed bf16 [M, N] y (bias classes
  only).

Cold-L2 protocol (copied from ``bench_qwen3_4b_mlp.py::weight_copies``): the graph
of an arm runs the op over R copies of the weight -- same values, distinct
memory, every quantized form built per copy, and one bias tensor per copy -- with
R x (bytes of one copy) >= 2 x L2 and R in [2, 256], and the arm reports replay
time / R. The bf16 arm counts the bf16 weight's bytes, the fso arms the fp8
weight plus its scale. ``fso_quant`` and ``bias_add`` have no weight: they run R
times with the fso arms' R, the quantize on the one warm activation, the bias add
on R copies of y and R bias tensors. A cell whose R x bytes falls short of
2 x L2 prints ``COLD-PROTOCOL:`` on stderr and fails. The activation stays warm
(one x for every copy).

Timing (protocol of ``time_fn_graph`` in ``bench_qwen3_4b_mlp.py``, split into
capture and timing so the arms can be interleaved): per arm 15 eager calls of the
whole R-copy closure, 3 calls on a side stream, capture on that side stream, one
untimed replay; then 3 rounds, each timing every arm once in the fixed order
bf16, fso_chain, fso_gemm, fso_quant, bias_add as 50 replays between two CUDA
events. ``us`` is the median of the three rounds' per-copy times and
``us_rounds`` keeps the three values. ``FSO_BENCH_WARM_MS`` runs the DVFS settle
spin of ``bench_qwen3_4b_mlp.py::_clock_warm_spin`` at the start of the cell and
again right before the timed rounds.

Per cell also: the kernels of one eager call of every arm (torch.profiler, CUDA
activity, run after all timing so that the profiler never overlaps a timed
region): ``kernels`` counts kernel launches, ``kernel_names`` lists them in launch
order, ``other_gpu_ops`` lists memsets / memcpys separately; on sm_100/sm_103 the
router's route for the GEMM from the predicates of ``_sm100_dispatch.route`` in
their order; accuracy of ``bf16``, ``fso_chain`` and ``fso_gemm`` against an fp32
reference ``x.float() @ w.float().T (+ bias.float())`` (TF32 off; ``fso_gemm``
against the reference without bias): ``cos`` is the cosine similarity and
``max_rel`` = max|y - ref| / max|ref| (normalised by the reference's largest
magnitude, because the elementwise ratio is unbounded near zero), ``max_abs`` =
max|y - ref|.

Inputs per cell: ``torch.manual_seed(M * 1009 + N * 17 + K)``,
``x = randn(M, K) * 0.5``, ``w = randn(N, K) / sqrt(K)``, ``bias = randn(N) * 0.1``,
all bf16 on the device; a cell fails if any 32-element block of x or w has
amax 0.

Process model. ``--run`` is the parent: it never imports torch or fish_scales_ops
and never creates a CUDA context (the B300 card runs in Exclusive_Process mode),
reads the device through a ``--probe`` subprocess, samples the SM clock with
``nvidia-smi --query-gpu=timestamp,clocks.sm -lms 200`` into ``<out>.clocks.csv``
(epoch-stamped), checks before every cell that the card has no compute process
and stops the pass if it has one, and runs each cell in its own ``--worker``
subprocess (a crashed cell becomes an error line). Worker stderr goes to
``<out>.worker.log``.

Classes (tag, N, K, bias, count): the tower classes at M = 78, 117, 182, 364,
533, 1079; the LLM qkv / o projections (count 48 per step), unsharded with
``--llm-tp 1`` (the default except on sm_120) and as the TP2 shards with
``--llm-tp 2`` (the default on sm_120), at the decode grid M = 1 .. 64 and the
extend grid M = 80 .. 16384 (capped at 2048 on sm_120). ``--classes`` selects
classes, ``--Ms`` replaces the M grid of every selected class; ``--tower-Ms`` and
``--llm-Ms`` replace the grid of the tower or the LLM classes only (and take
precedence over ``--Ms`` for their family).

Layer roles (``--roles``). A class above times one GEMM per shape, while a tower
call and an LLM step are made of layer roles whose epilogues differ. ``--run
--roles`` times one cell per (role, M) instead, written as lines with
``"group": "issue12_dense_roles"`` and a ``"role"`` key; a run without
``--roles`` writes exactly the class lines described above. Roles (N x K,
epilogue, count per tower call or per LLM step):

* tower, at the tower rungs: ``q`` (1280x1280, bias; 32), which also stands for
  ``k`` and ``v`` (the same shape, epilogue and input, so one cell serves the three
  and the composites count it 3 x 32 times); ``out_proj`` (1280x1280, bias +
  residual; 32); ``fc1`` (5120x1280, bias + erf-GELU; 32); ``fc2`` (1280x5120,
  bias + residual; 32); ``conv_out`` (1280x7680, no bias; 1); ``proj1``
  (1280x1280, bias + erf-GELU; 1); ``proj2`` (2048x1280, bias; 1).
* LLM, no bias, at the LLM grid: ``qkv`` and ``o`` (48 each; the shapes of
  ``llm_qkv`` / ``llm_o``), or ``qkv_tp2`` and ``o_tp2`` with ``--llm-tp 2``.

``--roles`` alone selects every tower role and the LLM roles of ``--llm-tp``;
``--roles q,fc1,o`` selects some (``k`` and ``v`` select ``q``). Inputs are those
of a class cell of the same (M, N, K), drawn in the same order, and a residual
role then draws ``residual = randn(M, N) * 0.5``. Arms of a role cell, in this
timing order, with the capture, cold-weight and interleaved-rounds protocol of
the class arms (``fso_quant`` again runs R times on the one warm activation):

* ``bf16``       -- ``F.linear(x, w, b)``, then ``F.gelu(y, approximate="none")``
  or ``y + residual`` in torch, as the role has them.
* ``fso_today``  -- ``fso.dense.linear(x, dw)``, then ``+ bias``, then the GELU
  or ``+ residual`` in torch: the serving chain the issue describes.
* ``fso_epi``    -- ``fso.dense.linear(x, dw, bias=b, residual=r, activation=a)``
  with the bf16 x (``None`` for what the role does not have).
* ``fso_epi_2b`` -- the same call with ``x = xq, x_scale = sx`` from
  ``fso.dense.quantize_activation(x)``, run once outside the graph (form 2b).
* ``fso_q_2b``   -- ``fc1`` and ``proj1`` only:
  ``fso.dense.linear_quantized(xq, dw, x_scale=sx, bias=b, activation="gelu_erf")``.
* ``fso_quant``  -- ``fso.dense.quantize_activation(x)`` alone at [M, K].

Before anything is captured, every arm is called once eagerly; an arm whose call
raises ``NotImplementedError`` (a route that does not apply that epilogue yet)
is recorded as ``{"na": "<first line of the message>"}`` and is not timed, and
the cell goes on. Any other exception fails the cell. Every arm that runs
records its kernel launches (torch.profiler, as above) and, except
``fso_q_2b`` and ``fso_quant``, its accuracy against the fp32 reference
``act(x.float() @ w.float().T + bias.float()) + residual.float()``. Two byte
checks are recorded when both arms ran: ``epi_2b_eq_epi`` (form 2b gives the
bytes of the bf16-x call) and ``q_2b_eq_unfused`` (``fso_q_2b`` gives the bytes
of ``quantize_activation`` applied to the ``fso_epi_2b`` result).

``--format`` renders, besides the per-role tables, the composites the issue is
answered with, computed from the raw cells (Σ multiplier x µs, and Σ multiplier
x kernels as launches). Tower, one call per rung: ``bf16``, ``fso_today`` and
``fso_epi`` sum their own arm over every role; ``fso_best_fsoquant`` is, per
layer, one ``fso_quant`` shared by q, k and v plus 3 x ``fso_epi_2b(q)``,
``fso_epi(out_proj)``, ``fso_quant(fc1) + fso_q_2b(fc1)``, ``fso_epi_2b(fc2)``
(its input is fc1's quantized output), times 32 layers, plus
``fso_epi(conv_out)``, ``fso_quant(proj1) + fso_q_2b(proj1)`` and
``fso_epi_2b(proj2)``; ``fso_best_producerquant`` is the same with every
``fso_quant`` term set to zero (the caller's LayerNorm writes the contract
layout itself). LLM, one step per M: ``bf16``, ``fso_today``,
``fso_best_fsoquant`` = 48 x (``fso_quant(qkv)`` + ``fso_epi_2b(qkv)`` +
``fso_epi(o)``) and ``fso_best_producerquant`` (without the quantize). A
composite with a term that is n/a, failed or not measured prints n/a and names
the roles and arms it is missing.

Usage::

    python bench_issue12_dense.py --run --out RUN/b300_pass1.jsonl --machine b300
    python bench_issue12_dense.py --run --roles --out RUN/b300_roles1.jsonl --machine b300
    python bench_issue12_dense.py --format --from B300=RUN/b300_pass1.jsonl
    python bench_issue12_dense.py --spread --from p1=RUN/b300_pass1.jsonl p2=RUN/b300_pass2.jsonl
    python bench_issue12_dense.py --checks --from B300=RUN/b300_pass1.jsonl
"""
import argparse
import datetime
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback

GROUP = "issue12_dense"

# (tag, N, K, has_bias, count per tower call / per LLM decode step)
TOWER_CLASSES = [
    ("tower_1280x1280", 1280, 1280, True, 129),   # q, k, v, out_proj in 32 layers + proj1
    ("tower_fc1",       5120, 1280, True, 32),
    ("tower_fc2",       1280, 5120, True, 32),
    ("tower_conv_out",  1280, 7680, False, 1),
    ("tower_proj2",     2048, 1280, True, 1),
]
LLM_TP1 = [
    ("llm_qkv", 5120, 2048, False, 48),
    ("llm_o",   2048, 4096, False, 48),
]
LLM_TP2 = [
    ("llm_qkv_tp2", 2560, 2048, False, 48),
    ("llm_o_tp2",   2048, 2048, False, 48),
]
ALL_CLASSES = {c[0]: c for c in TOWER_CLASSES + LLM_TP1 + LLM_TP2}
TOWER_MS = [78, 117, 182, 364, 533, 1079]
LLM_DECODE_MS = [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64]
LLM_EXTEND_MS = [80, 96, 112, 128, 144, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448,
                 512, 576, 640, 768, 1024, 1280, 1536, 2048, 3072, 4096, 8192, 16384]
SM120_LLM_M_CAP = 2048
ARM_ORDER = ["bf16", "fso_chain", "fso_gemm", "fso_quant", "bias_add"]

# Layer roles (--roles). (role, N, K, has_bias, has_residual, activation, count, aliases): count per tower call or
# per LLM step for each of the role and its aliases, which are the same experiment (same shape, epilogue and input)
# and are served by the role's one cell.
GROUP_ROLES = "issue12_dense_roles"
TOWER_ROLES = [
    ("q",        1280, 1280, True,  False, None,       32, ("k", "v")),
    ("out_proj", 1280, 1280, True,  True,  None,       32, ()),
    ("fc1",      5120, 1280, True,  False, "gelu_erf", 32, ()),
    ("fc2",      1280, 5120, True,  True,  None,       32, ()),
    ("conv_out", 1280, 7680, False, False, None,        1, ()),
    ("proj1",    1280, 1280, True,  False, "gelu_erf",  1, ()),
    ("proj2",    2048, 1280, True,  False, None,        1, ()),
]
LLM_ROLES_TP1 = [
    ("qkv", 5120, 2048, False, False, None, 48, ()),
    ("o",   2048, 4096, False, False, None, 48, ()),
]
LLM_ROLES_TP2 = [
    ("qkv_tp2", 2560, 2048, False, False, None, 48, ()),
    ("o_tp2",   2048, 2048, False, False, None, 48, ()),
]
ALL_ROLES = {r[0]: r for r in TOWER_ROLES + LLM_ROLES_TP1 + LLM_ROLES_TP2}
ROLE_ALIASES = {a: r[0] for r in ALL_ROLES.values() for a in r[7]}
LLM_ROLE_PAIRS = [("qkv", "o"), ("qkv_tp2", "o_tp2")]
ROLE_ARM_ORDER = ["bf16", "fso_today", "fso_epi", "fso_epi_2b", "fso_q_2b", "fso_quant"]
Q2B_ROLES = ("fc1", "proj1")      # the roles whose output is the next GEMM's activation (GELU -> fc2 / proj2)
ROLE_ARM_SHORT = {"bf16": "bf16", "fso_today": "today", "fso_epi": "epi", "fso_epi_2b": "epi_2b",
                  "fso_q_2b": "q_2b", "fso_quant": "quant"}
TOWER_COMPOSITES = ["bf16", "fso_today", "fso_epi", "fso_best_fsoquant", "fso_best_producerquant"]
LLM_COMPOSITES = ["bf16", "fso_today", "fso_best_fsoquant", "fso_best_producerquant"]
ROUNDS = 3
ITERS = 50
CELL_TIMEOUT_S = 900

# The issue's B300 numbers (fso 0.2.1, whose sm_100 machine code equals 0.2.2's),
# used only by --checks to say whether this bench agrees with them within 15 %.
ISSUE_REF = [
    ("tower_1280x1280", [78], "bf16", 3.4),
    ("tower_1280x1280", [78], "fso_chain", 8.5),
    ("llm_o", [80, 96, 112, 128], "fso_chain", 10.1),
]


# ---------------------------------------------------------------------------
# Copied from bench_qwen3_4b_mlp.py (no cross-script import).
# ---------------------------------------------------------------------------
def _clock_warm_spin():
    """Optional DVFS settle: busy tensor-core work before any timing.
    FSO_BENCH_WARM_MS=<ms> runs a dummy matmul spin for that long; default 0."""
    import torch
    ms = int(os.environ.get("FSO_BENCH_WARM_MS", "0"))
    if ms <= 0:
        return
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    t0 = time.monotonic()
    while (time.monotonic() - t0) * 1000.0 < ms:
        a = a @ a * 1e-3  # keep values bounded; result reused to defeat DCE
    torch.cuda.synchronize()


def weight_copies(active_bytes):
    """Cold-L2 protocol (2026-09-28): the captured graph runs the op over R copies
    of the weight (same values, distinct memory, every quantized form built per
    copy), R * bytes >= 2 x L2 with R in [2, 256], and reports replay time / R, so
    each copy's weight has been evicted before the graph returns to it -- as a
    serving step evicts every layer's weights by streaming the next layers'
    through. Replaying one warm copy served the M <= 128 rows from L2."""
    import torch
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    if os.environ.get("FSO_BENCH_WEIGHT_COPIES"):              # probe knob: an explicit copy count (1 = the old warm protocol)
        return max(1, int(os.environ["FSO_BENCH_WEIGHT_COPIES"]))
    mult = float(os.environ.get("FSO_BENCH_L2_MULT", "2"))   # probe knob: how many L2s of weights the rotation spans
    return max(2, min(256, int(math.ceil(mult * l2 / max(int(active_bytes), 1)))))


# ---------------------------------------------------------------------------
# Worker side (one cell per process)
# ---------------------------------------------------------------------------
def _capture(fn):
    """Eager warm-up, side-stream warm-up, capture on the side stream (the
    protocol of bench_qwen3_4b_mlp.py::time_fn_graph), then one untimed replay
    so that the graph's first upload never lands inside a timed block."""
    import torch
    for _ in range(15):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    g.replay()
    torch.cuda.synchronize()
    return g


def _time_graph(g, iters=ITERS):
    """Milliseconds per replay over `iters` back-to-back replays."""
    import torch
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def _kernels_of(fn_one):
    """GPU activities of one eager call (after one more warm call)."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    fn_one()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn_one()
        torch.cuda.synchronize()
    evs = [e for e in prof.events() if e.device_type.name == "CUDA"]
    evs.sort(key=lambda e: e.time_range.start)
    kern, other = [], []
    for e in evs:
        name = e.name
        if name.startswith("Memset") or name.startswith("Memcpy"):
            other.append(name[:200])
        else:
            kern.append(name[:200])
    return kern, other


def _sm100_route(m, n, k):
    """The tier `_sm100_dispatch.route` picks for this GEMM (identity epilogue),
    from its predicates in their order: the swap-AB row (``pick_config`` answers
    None outside its bands: the decode band M <= 64 and, from issue #12 P5, the
    narrow-N long-K band above it), cuBLAS scaled_mm, the cascade's wave rule,
    the CuTe-DSL tier, the C++ cascade."""
    from fish_scales_ops.gemm import _sm100_decode, _sm100_dsl, _sm100_smm
    cfg = _sm100_decode.pick_config(m, n, k)
    if cfg is not None:
        tile, cluster, _swap, split_k = cfg
        kind = f"splitk{split_k}" if split_k > 1 else "persistent"
        s = f"decode:{kind} tiler=({tile[0]},{tile[1]})"
        if tuple(cluster) != (1, 1):
            s += f" cluster=({cluster[0]},{cluster[1]})"
        return s
    if _sm100_smm.should_route(m, n, k):
        return "smm"
    if _sm100_smm.wave_tile_owns(m, n):
        return "cascade:wave"
    cfg = _sm100_dsl.pick_config(m, n, k)
    if cfg is not None:
        tile, cluster = cfg[0], cfg[1]
        stages = cfg[2] if len(cfg) > 2 else None
        swizzle = cfg[3] if len(cfg) > 3 else 1
        s = (f"dsl:tiler=({tile[0]},{tile[1]}) cluster=({cluster[0]},{cluster[1]}) "
             f"stages={stages} swizzle={swizzle}")
        if _sm100_dsl._init() is None:
            s += " [dsl unavailable -> cascade]"
        return s
    return "cascade"


def _device_meta():
    import torch
    import fish_scales_ops as fso
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    info = fso.build_info()
    return {
        "device": torch.cuda.get_device_name(0),
        "sm": cap[0] * 10 + cap[1],
        "l2_bytes": int(props.L2_cache_size),
        "sm_count": int(props.multi_processor_count),
        "torch": str(torch.__version__),
        "fso_version": getattr(fso, "__version__", None),
        "fso_commit": info.get("commit"),
        "fso_file": os.path.dirname(os.path.abspath(fso.__file__)),
    }


def bench_cell(cell):
    import torch
    import torch.nn.functional as F
    import fish_scales_ops as fso

    meta = _device_meta()
    sm = meta["sm"]
    major = sm // 10
    if major == 12:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    tag, M, N, K = cell["class"], int(cell["M"]), int(cell["N"]), int(cell["K"])
    has_bias, count = bool(cell["has_bias"]), int(cell["count"])
    out = {"group": GROUP, "machine": cell.get("machine"), **meta,
           "class": tag, "N": N, "K": K, "M": M, "has_bias": has_bias, "count": count,
           "bf16_reduced_precision_reduction":
               bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)}
    l2 = meta["l2_bytes"]
    fmt = "bsfp8" if major == 9 else "mxfp8"
    out["format"] = fmt

    _clock_warm_spin()
    torch.manual_seed(M * 1009 + N * 17 + K)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.5
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.5)
    bias = (torch.randn(N, dtype=torch.bfloat16, device="cuda") * 0.1) if has_bias else None
    torch.cuda.synchronize()

    def zero_blocks(t):
        return int((t.float().reshape(t.shape[0], -1, 32).abs().amax(-1) == 0).sum().item())
    zb = {"x": zero_blocks(x), "w": zero_blocks(w)}
    out["zero_blocks"] = zb
    if zb["x"] or zb["w"]:
        raise RuntimeError(f"input has 32-element blocks with amax 0: {zb}")

    # ---- weight copies ---------------------------------------------------
    cold_fail = []

    def check_cold(arm, R, nbytes):
        if R * nbytes < 2 * l2:
            msg = (f"COLD-PROTOCOL: {tag} M={M} arm {arm}: R={R} x {nbytes} B = {R * nbytes} B "
                   f"< 2 x L2 = {2 * l2} B")
            print(msg, file=sys.stderr, flush=True)
            cold_fail.append(msg)

    bf_bytes = N * K * 2
    R_bf = weight_copies(bf_bytes)
    check_cold("bf16", R_bf, bf_bytes)
    w_bf = [w] + [w.clone() for _ in range(R_bf - 1)]
    b_bf = ([bias] + [bias.clone() for _ in range(R_bf - 1)]) if has_bias else [None] * R_bf

    dw0 = fso.dense.prepare_weight(w, format=fmt)
    fso_bytes = (dw0.weight.numel() * dw0.weight.element_size()
                 + dw0.scale.numel() * dw0.scale.element_size())
    R_f = weight_copies(fso_bytes)
    for arm in ("fso_chain", "fso_gemm"):
        check_cold(arm, R_f, fso_bytes)
    if cold_fail:
        raise RuntimeError("; ".join(cold_fail))
    dws = [dw0]
    for _ in range(R_f - 1):
        wc = w.clone()
        dws.append(fso.dense.prepare_weight(wc, format=fmt))
        del wc
    b_f = ([bias] + [bias.clone() for _ in range(R_f - 1)]) if has_bias else [None] * R_f

    if major == 9:
        xq, sx = fso.compat.quantize_1x128_fp8(x)

        def quant():
            return fso.compat.quantize_1x128_fp8(x)

        def gemm(d):
            return fso.compat.linear_fp8(xq, d.weight, sx, d.scale)
    else:
        xq, sx = fso.compat.quantize_1x32_fp8(x)

        def quant():
            return fso.compat.quantize_1x32_fp8(x)

        def gemm(d):
            return fso.compat.linear_mxfp8(xq, d.weight, sx, d.scale)

    def chain(d, b):
        y = fso.dense.linear(x, d)
        if b is not None:
            y = y + b
        return y

    y0 = F.linear(x, w)
    ys = [y0.clone() for _ in range(R_f)] if has_bias else []
    torch.cuda.synchronize()

    # One-copy calls (accuracy, kernel lists) and R-copy closures (timing).
    one = {
        "bf16": lambda: F.linear(x, w, bias),
        "fso_chain": lambda: chain(dw0, bias),
        "fso_gemm": lambda: gemm(dw0),
        "fso_quant": quant,
    }

    def all_bf16():
        for r in range(R_bf):
            F.linear(x, w_bf[r], b_bf[r])

    def all_chain():
        for r in range(R_f):
            chain(dws[r], b_f[r])

    def all_gemm():
        for r in range(R_f):
            gemm(dws[r])

    def all_quant():
        for _ in range(R_f):
            quant()

    closures = {"bf16": (all_bf16, R_bf, bf_bytes), "fso_chain": (all_chain, R_f, fso_bytes),
                "fso_gemm": (all_gemm, R_f, fso_bytes), "fso_quant": (all_quant, R_f, 0)}
    if has_bias:
        def all_bias():
            for r in range(R_f):
                ys[r] + b_f[r]
        one["bias_add"] = lambda: y0 + bias
        closures["bias_add"] = (all_bias, R_f, 0)
    arms_present = [a for a in ARM_ORDER if a in closures]

    # ---- capture every arm, then interleaved timing rounds ---------------
    graphs = {a: _capture(closures[a][0]) for a in arms_present}
    _clock_warm_spin()
    rounds = {a: [] for a in arms_present}
    t_timing0 = time.time()
    for _ in range(ROUNDS):
        for a in arms_present:
            rounds[a].append(_time_graph(graphs[a]) * 1000.0 / closures[a][1])
    t_timing1 = time.time()
    del graphs
    torch.cuda.synchronize()
    out["timing_window"] = [t_timing0, t_timing1]

    arms = {}
    for a in arms_present:
        _, R, nbytes = closures[a]
        arms[a] = {"us": statistics.median(rounds[a]), "us_rounds": rounds[a],
                   "weight_copies": R, "copy_bytes": nbytes,
                   "copies_of": {"bf16": "bf16 weight + bias", "fso_chain": "prepared weight + bias",
                                 "fso_gemm": "prepared weight", "fso_quant": "none (warm x)",
                                 "bias_add": "y + bias"}[a]}

    # ---- accuracy against the fp32 reference ----------------------------
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    ref_nb = x.float() @ w.float().t()
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    ref = ref_nb + bias.float() if has_bias else ref_nb

    def acc(y, r):
        yf = y.float()
        err = float((yf - r).abs().max().item())
        den = float(r.abs().max().item())
        c = float(F.cosine_similarity(yf.flatten(), r.flatten(), dim=0).item())
        return {"cos": c, "max_rel": err / den if den > 0 else None, "max_abs": err}

    y_bf = one["bf16"]()
    y_chain = one["fso_chain"]()
    y_gemm = one["fso_gemm"]()
    y_dense = fso.dense.linear(x, dw0)
    torch.cuda.synchronize()
    arms["bf16"].update(acc(y_bf, ref))
    arms["fso_chain"].update(acc(y_chain, ref))
    arms["fso_gemm"].update(acc(y_gemm, ref_nb))
    eq = bool(torch.equal(y_gemm, y_dense))
    out["gemm_eq_dense_linear"] = eq
    out["sm90_gemm_eq_chain"] = eq if major == 9 else None
    del y_bf, y_chain, y_gemm, y_dense, ref, ref_nb

    # ---- route and kernels (after all timing) ---------------------------
    out["route"] = _sm100_route(M, N, K) if major == 10 else None
    for a in arms_present:
        kern, other = _kernels_of(one[a])
        arms[a]["kernels"] = len(kern)
        arms[a]["kernel_names"] = kern
        arms[a]["other_gpu_ops"] = other
    out["arms"] = arms
    out["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    return out


def _role_ident(cell):
    """The identifying fields of a role cell, as every role line (measured or failed) carries them."""
    return {k: cell.get(k) for k in ("role", "family", "aliases", "N", "K", "M", "has_bias", "has_residual",
                                     "activation", "count")}


def bench_role_cell(cell):
    """One (role, M) cell: the six role arms (see the module docstring), n/a for an arm the library refuses."""
    import torch
    import torch.nn.functional as F
    import fish_scales_ops as fso

    meta = _device_meta()
    sm = meta["sm"]
    major = sm // 10
    if major == 12:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    role, M, N, K = cell["role"], int(cell["M"]), int(cell["N"]), int(cell["K"])
    has_bias, has_res = bool(cell["has_bias"]), bool(cell["has_residual"])
    act = cell.get("activation")
    gelu = act == "gelu_erf"
    out = {"group": GROUP_ROLES, "machine": cell.get("machine"), **meta, **_role_ident(cell),
           "bf16_reduced_precision_reduction":
               bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)}
    l2 = meta["l2_bytes"]
    fmt = "bsfp8" if major == 9 else "mxfp8"
    out["format"] = fmt

    _clock_warm_spin()
    # The draws of a class cell of the same (M, N, K), in the same order; the residual after them.
    torch.manual_seed(M * 1009 + N * 17 + K)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.5
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.5)
    bias = (torch.randn(N, dtype=torch.bfloat16, device="cuda") * 0.1) if has_bias else None
    residual = (torch.randn(M, N, dtype=torch.bfloat16, device="cuda") * 0.5) if has_res else None
    torch.cuda.synchronize()

    def zero_blocks(t):
        return int((t.float().reshape(t.shape[0], -1, 32).abs().amax(-1) == 0).sum().item())
    zb = {"x": zero_blocks(x), "w": zero_blocks(w)}
    out["zero_blocks"] = zb
    if zb["x"] or zb["w"]:
        raise RuntimeError(f"input has 32-element blocks with amax 0: {zb}")

    # ---- weight copies (the protocol of bench_cell) ----------------------
    cold_fail = []

    def check_cold(arm, R, nbytes):
        if R * nbytes < 2 * l2:
            msg = (f"COLD-PROTOCOL: {role} M={M} arm {arm}: R={R} x {nbytes} B = {R * nbytes} B "
                   f"< 2 x L2 = {2 * l2} B")
            print(msg, file=sys.stderr, flush=True)
            cold_fail.append(msg)

    bf_bytes = N * K * 2
    R_bf = weight_copies(bf_bytes)
    check_cold("bf16", R_bf, bf_bytes)
    w_bf = [w] + [w.clone() for _ in range(R_bf - 1)]
    b_bf = ([bias] + [bias.clone() for _ in range(R_bf - 1)]) if has_bias else [None] * R_bf

    dw0 = fso.dense.prepare_weight(w, format=fmt)
    fso_bytes = (dw0.weight.numel() * dw0.weight.element_size()
                 + dw0.scale.numel() * dw0.scale.element_size())
    R_f = weight_copies(fso_bytes)
    weight_arms = ["fso_today", "fso_epi", "fso_epi_2b"] + (["fso_q_2b"] if role in Q2B_ROLES else [])
    for arm in weight_arms:
        check_cold(arm, R_f, fso_bytes)
    if cold_fail:
        raise RuntimeError("; ".join(cold_fail))
    dws = [dw0]
    for _ in range(R_f - 1):
        wc = w.clone()
        dws.append(fso.dense.prepare_weight(wc, format=fmt))
        del wc
    b_f = ([bias] + [bias.clone() for _ in range(R_f - 1)]) if has_bias else [None] * R_f

    # Form 2b: the activation quantized once, outside every graph.
    xq, sx = fso.dense.quantize_activation(x)
    torch.cuda.synchronize()

    def bf16_call(wr, br):
        y = F.linear(x, wr, br)
        if gelu:
            y = F.gelu(y, approximate="none")
        if has_res:
            y = y + residual
        return y

    def today_call(d, br):
        y = fso.dense.linear(x, d)
        if br is not None:
            y = y + br
        if gelu:
            y = F.gelu(y, approximate="none")
        if has_res:
            y = y + residual
        return y

    def epi_call(d, br):
        return fso.dense.linear(x, d, bias=br, residual=residual, activation=act)

    def epi_2b_call(d, br):
        return fso.dense.linear(xq, d, x_scale=sx, bias=br, residual=residual, activation=act)

    def q_2b_call(d, br):
        return fso.dense.linear_quantized(xq, d, x_scale=sx, bias=br, activation=act)

    def quant_call():
        return fso.dense.quantize_activation(x)

    one = {
        "bf16": lambda: bf16_call(w, bias),
        "fso_today": lambda: today_call(dw0, bias),
        "fso_epi": lambda: epi_call(dw0, bias),
        "fso_epi_2b": lambda: epi_2b_call(dw0, bias),
        "fso_quant": quant_call,
    }
    if role in Q2B_ROLES:
        one["fso_q_2b"] = lambda: q_2b_call(dw0, bias)

    # ---- probe: one eager call per arm; a refused epilogue is n/a, not a failure
    arms = {}
    for a in ROLE_ARM_ORDER:
        if a not in one:
            continue
        try:
            one[a]()
        except NotImplementedError as e:
            msg = str(e).strip().splitlines()
            arms[a] = {"na": msg[0] if msg else type(e).__name__}
    torch.cuda.synchronize()

    def rotate(call, R, ws, bs):
        def run():
            for r in range(R):
                call(ws[r], bs[r])
        return run

    def all_quant():
        for _ in range(R_f):
            quant_call()

    bias_txt = " + bias" if has_bias else ""
    closures = {
        "bf16": (rotate(bf16_call, R_bf, w_bf, b_bf), R_bf, bf_bytes, "bf16 weight" + bias_txt),
        "fso_today": (rotate(today_call, R_f, dws, b_f), R_f, fso_bytes, "prepared weight" + bias_txt),
        "fso_epi": (rotate(epi_call, R_f, dws, b_f), R_f, fso_bytes, "prepared weight" + bias_txt),
        "fso_epi_2b": (rotate(epi_2b_call, R_f, dws, b_f), R_f, fso_bytes, "prepared weight" + bias_txt),
        "fso_quant": (all_quant, R_f, 0, "none (warm x)"),
    }
    if role in Q2B_ROLES:
        closures["fso_q_2b"] = (rotate(q_2b_call, R_f, dws, b_f), R_f, fso_bytes, "prepared weight" + bias_txt)
    arms_present = [a for a in ROLE_ARM_ORDER if a in closures and a not in arms]

    # ---- capture every arm that runs, then interleaved timing rounds -----
    graphs = {a: _capture(closures[a][0]) for a in arms_present}
    _clock_warm_spin()
    rounds = {a: [] for a in arms_present}
    t_timing0 = time.time()
    for _ in range(ROUNDS):
        for a in arms_present:
            rounds[a].append(_time_graph(graphs[a]) * 1000.0 / closures[a][1])
    t_timing1 = time.time()
    del graphs
    torch.cuda.synchronize()
    out["timing_window"] = [t_timing0, t_timing1]
    for a in arms_present:
        _, R, nbytes, what = closures[a]
        arms[a] = {"us": statistics.median(rounds[a]), "us_rounds": rounds[a],
                   "weight_copies": R, "copy_bytes": nbytes, "copies_of": what}

    # ---- accuracy against the fp32 reference, byte checks ---------------
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = x.float() @ w.float().t()
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    if has_bias:
        ref = ref + bias.float()
    if gelu:
        ref = F.gelu(ref, approximate="none")
    if has_res:
        ref = ref + residual.float()

    def acc(y):
        yf = y.float()
        err = float((yf - ref).abs().max().item())
        den = float(ref.abs().max().item())
        c = float(F.cosine_similarity(yf.flatten(), ref.flatten(), dim=0).item())
        return {"cos": c, "max_rel": err / den if den > 0 else None, "max_abs": err}

    ys = {}
    for a in ("bf16", "fso_today", "fso_epi", "fso_epi_2b"):
        if a in arms_present:
            ys[a] = one[a]()
    torch.cuda.synchronize()
    for a, y in ys.items():
        arms[a].update(acc(y))

    def same_bytes(p, q):
        if p.dtype != q.dtype or tuple(p.shape) != tuple(q.shape):
            return False
        view = {torch.float8_e4m3fn: torch.uint8, torch.bfloat16: torch.int16, torch.float32: torch.int32}
        if p.dtype in view:
            p, q = p.view(view[p.dtype]), q.view(view[q.dtype])
        return bool(torch.equal(p, q))

    out["epi_2b_eq_epi"] = (same_bytes(ys["fso_epi_2b"], ys["fso_epi"])
                            if "fso_epi_2b" in ys and "fso_epi" in ys else None)
    q2b = None
    if "fso_q_2b" in arms_present and "fso_epi_2b" in ys:
        qn, sn = one["fso_q_2b"]()
        qr, sr = fso.dense.quantize_activation(ys["fso_epi_2b"])
        q2b = same_bytes(qn, qr) and same_bytes(sn, sr) and tuple(sn.stride()) == tuple(sr.stride())
    out["q_2b_eq_unfused"] = q2b
    del ys, ref

    # ---- route and kernels (after all timing) ---------------------------
    out["route"] = _sm100_route(M, N, K) if major == 10 else None
    if major == 10:
        try:
            from fish_scales_ops.gemm import _sm100_dispatch
            out["route_plan"] = _sm100_dispatch.plan(M, N, K)
        except Exception as e:  # noqa: BLE001 - the plan is a label, not a measurement
            out["route_plan"] = f"unavailable: {type(e).__name__}"
    for a in arms_present:
        kern, other = _kernels_of(one[a])
        arms[a]["kernels"] = len(kern)
        arms[a]["kernel_names"] = kern
        arms[a]["other_gpu_ops"] = other
    out["arms"] = {a: arms[a] for a in ROLE_ARM_ORDER if a in arms}
    out["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    return out


def worker_main():
    cell = json.loads(sys.stdin.read())
    is_role = "role" in cell
    try:
        res = bench_role_cell(cell) if is_role else bench_cell(cell)
    except Exception as e:  # noqa: BLE001 - a failed cell is an error line
        traceback.print_exc(file=sys.stderr)
        if is_role:
            res = {"group": GROUP_ROLES, "machine": cell.get("machine"), **_role_ident(cell),
                   "error": f"{type(e).__name__}: {str(e)[:600]}",
                   "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
        else:
            res = {"group": GROUP, "machine": cell.get("machine"), "class": cell.get("class"),
                   "N": cell.get("N"), "K": cell.get("K"), "M": cell.get("M"),
                   "has_bias": cell.get("has_bias"), "count": cell.get("count"),
                   "error": f"{type(e).__name__}: {str(e)[:600]}",
                   "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    sys.stdout.write(json.dumps(res) + "\n")
    sys.stdout.flush()


def probe_main():
    sys.stdout.write(json.dumps(_device_meta()) + "\n")


# ---------------------------------------------------------------------------
# Parent side (no torch, no CUDA context)
# ---------------------------------------------------------------------------
def _stem(path):
    return path[:-len(".jsonl")] if path.endswith(".jsonl") else path


def _smi_id(args):
    if args.smi_id:
        return args.smi_id
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd.startswith("GPU-") and "," not in cvd:
        return cvd
    sys.exit("--run needs CUDA_VISIBLE_DEVICES set to one GPU UUID, or --smi-id naming the card "
             "for nvidia-smi (the clock sampler and the idle check)")


def _compute_apps(smi_id):
    p = subprocess.run(["nvidia-smi", "-i", smi_id, "--query-compute-apps=pid,used_memory,process_name",
                        "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        return [f"nvidia-smi failed: {p.stderr.strip()[:200]}"]
    return [l.strip() for l in p.stdout.splitlines() if l.strip()]


def _foreign_apps(smi_id, own_pids):
    """Compute processes on the card other than this run's own (already exited)
    probe and workers. Asked twice, 3 s apart, when the first answer is not empty,
    so that a worker whose context is still being torn down is not taken for
    another user; anything still there is another user, and the caller stops."""
    apps = []
    for attempt in range(2):
        apps = [a for a in _compute_apps(smi_id) if a.split(",")[0].strip() not in own_pids]
        if not apps:
            return []
        if attempt == 0:
            time.sleep(3)
    return apps


def _run_child(argv, input_text, timeout, own_pids):
    """Run one child to completion; its pid joins `own_pids`. Returns (rc, stdout,
    stderr); raises subprocess.TimeoutExpired after killing a child that overran."""
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    own_pids.add(str(proc.pid))
    try:
        out, err = proc.communicate(input_text, timeout=timeout)
    except BaseException:
        proc.kill()
        proc.communicate()
        raise
    return proc.returncode, out, err


class ClockSampler:
    """nvidia-smi SM-clock sampler beside the whole run; each line is stamped with
    the epoch time it was read, so cells can be matched to their samples."""

    def __init__(self, smi_id, path):
        self.path = path
        self.proc = subprocess.Popen(
            ["nvidia-smi", "-i", smi_id, "--query-gpu=timestamp,clocks.sm",
             "--format=csv,noheader,nounits", "-lms", "200"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.f = open(path, "w")
        self.f.write("epoch,timestamp,clocks_sm_mhz\n")
        self.t = threading.Thread(target=self._pump, daemon=True)
        self.t.start()

    def _pump(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line:
                self.f.write(f"{time.time():.3f},{line}\n")
                self.f.flush()

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.t.join(timeout=10)
        self.f.close()


def _grid_override(args, family):
    """The M list replacing the grid of the tower or the LLM family, or None: --tower-Ms / --llm-Ms, else --Ms."""
    spec = getattr(args, "tower_Ms" if family == "tower" else "llm_Ms", None) or args.Ms
    return [int(m) for m in spec.split(",")] if spec else None


def build_role_cells(args, sm, tp):
    llm = LLM_ROLES_TP2 if tp == 2 else LLM_ROLES_TP1
    tower_names = [r[0] for r in TOWER_ROLES]
    if args.roles and args.roles != "all":
        names = []
        for n in (c.strip() for c in args.roles.split(",")):
            if not n:
                continue
            n = ROLE_ALIASES.get(n, n)
            if n not in ALL_ROLES:
                sys.exit(f"unknown role {n!r}; known: {sorted(ALL_ROLES)} (k and v select q)")
            if n not in names:
                names.append(n)
        roles = [ALL_ROLES[n] for n in names]
    else:
        roles = TOWER_ROLES + llm
    cells = []
    for role, n, k, has_bias, has_res, act, count, aliases in roles:
        family = "tower" if role in tower_names else "llm"
        ms = _grid_override(args, family)
        if ms is None:
            ms = TOWER_MS if family == "tower" else LLM_DECODE_MS + LLM_EXTEND_MS
            if family == "llm" and sm // 10 == 12:
                ms = [m for m in ms if m <= SM120_LLM_M_CAP]
        for m in ms:
            cells.append({"role": role, "family": family, "aliases": list(aliases), "N": n, "K": k, "M": m,
                          "has_bias": has_bias, "has_residual": has_res, "activation": act, "count": count})
    return cells


def build_cells(args, probe):
    sm = probe["sm"]
    tp = args.llm_tp if args.llm_tp else (2 if sm // 10 == 12 else 1)
    if getattr(args, "roles", None):
        return build_role_cells(args, sm, tp), tp
    llm = LLM_TP2 if tp == 2 else LLM_TP1
    if args.classes:
        names = [c.strip() for c in args.classes.split(",") if c.strip()]
        bad = [n for n in names if n not in ALL_CLASSES]
        if bad:
            sys.exit(f"unknown classes {bad}; known: {sorted(ALL_CLASSES)}")
        classes = [ALL_CLASSES[n] for n in names]
    else:
        classes = TOWER_CLASSES + llm
    cells = []
    for tag, n, k, has_bias, count in classes:
        override = _grid_override(args, "tower" if tag.startswith("tower_") else "llm")
        if override is not None:
            ms = override
        elif tag.startswith("tower_"):
            ms = TOWER_MS
        else:
            ms = LLM_DECODE_MS + LLM_EXTEND_MS
            if sm // 10 == 12:
                ms = [m for m in ms if m <= SM120_LLM_M_CAP]
        for m in ms:
            cells.append({"class": tag, "N": n, "K": k, "M": m, "has_bias": has_bias, "count": count})
    return cells, tp


def run_main(args):
    if not args.out:
        sys.exit("--out required with --run")
    smi_id = _smi_id(args)
    machine = args.machine or socket.gethostname()
    stem = _stem(args.out)
    logf = open(stem + ".worker.log", "a")

    def on_signal(signum, _frame):
        raise SystemExit(f"signal {signum}")
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_signal)

    own_pids = set()
    apps = _compute_apps(smi_id)
    if apps:
        sys.exit(f"card {smi_id} has compute processes, not launching (no queueing): {apps}")
    rc, pout, perr = _run_child([sys.executable, os.path.abspath(__file__), "--probe"], "", 600, own_pids)
    logf.write(f"=== probe rc={rc}\n{perr}\n")
    logf.flush()
    try:
        probe = json.loads(pout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        sys.exit(f"probe failed: rc={rc} stderr={perr[-800:]}")
    cells, tp = build_cells(args, probe)
    roles_mode = bool(getattr(args, "roles", None))
    header = {"_header": True, "group": GROUP_ROLES if roles_mode else GROUP, "machine": machine, **probe,
              "llm_tp": tp, "cells": len(cells), "argv": sys.argv,
              "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH",
                                                       "FSO_BENCH_WARM_MS", "CUDA_HOME")},
              "python": sys.executable, "smi_id": smi_id,
              "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    print(f"# {machine}: {probe['device']} (sm_{probe['sm']}), fso {probe['fso_version']} "
          f"({probe['fso_commit']}) from {probe['fso_file']}, llm_tp={tp}, cells={len(cells)}", flush=True)

    sampler = None if args.no_sampler else ClockSampler(smi_id, stem + ".clocks.csv")
    t0 = time.time()
    try:
        with open(args.out, "w") as f:
            f.write(json.dumps(header) + "\n")
            f.flush()
            for i, c in enumerate(cells):
                apps = _foreign_apps(smi_id, own_pids)
                if apps:
                    msg = f"card {smi_id} gained compute processes before cell {i}: {apps}; pass stopped"
                    f.write(json.dumps({"_abort": True, "reason": msg}) + "\n")
                    print("# " + msg, flush=True)
                    break
                c = dict(c, machine=machine)
                is_role = "role" in c
                name = c["role"] if is_role else c["class"]

                def failed(err):
                    if is_role:
                        return json.dumps({"group": GROUP_ROLES, "machine": machine, **_role_ident(c),
                                           "error": err})
                    return json.dumps({"group": GROUP, "machine": machine, **{k: c[k] for k in (
                        "class", "N", "K", "M", "has_bias", "count")}, "error": err})
                try:
                    rc, wout, werr = _run_child([sys.executable, os.path.abspath(__file__), "--worker"],
                                                json.dumps(c), CELL_TIMEOUT_S, own_pids)
                    logf.write(f"=== cell {i} {name} M={c['M']} rc={rc}\n{werr}\n")
                    logf.flush()
                    lines = [l for l in (wout or "").splitlines() if l.startswith("{")]
                    if lines:
                        line = lines[-1]
                    else:
                        line = failed(f"worker rc={rc}: {(werr or '')[-600:]}")
                except subprocess.TimeoutExpired:
                    line = failed(f"timeout after {CELL_TIMEOUT_S} s")
                f.write(line + "\n")
                f.flush()
                r = json.loads(line)
                el = time.time() - t0
                eta = el * (len(cells) - i - 1) / (i + 1)
                if "error" in r:
                    desc = "ERROR " + r["error"][:160]
                elif is_role:
                    a = r["arms"]
                    desc = "  ".join(f"{k}={_arm_txt(a[k])}" for k in ROLE_ARM_ORDER if k in a)
                    desc += ("  kern " + "/".join(f"{ROLE_ARM_SHORT[k]}={a[k]['kernels']}" for k in ROLE_ARM_ORDER
                                                  if k in a and "kernels" in a[k])
                             + f"  route={r.get('route')}")
                else:
                    a = r["arms"]
                    desc = "  ".join(f"{k}={a[k]['us']:.2f}" for k in ARM_ORDER if k in a)
                    desc += (f"  kern chain/bf16={a['fso_chain']['kernels']}/{a['bf16']['kernels']}"
                             f"  route={r.get('route')}")
                print(f"[{i + 1}/{len(cells)}] {name:>16} M={c['M']:5d}  {desc}  "
                      f"(t={el:.0f}s eta={eta:.0f}s)", flush=True)
    finally:
        if sampler is not None:
            sampler.stop()
        logf.close()
    print(f"# done in {time.time() - t0:.0f} s -> {args.out}", flush=True)


# ---------------------------------------------------------------------------
# Rendering (pure json; no torch)
# ---------------------------------------------------------------------------
def _load(path, group=GROUP):
    """(header, the lines of `group`, abort lines): GROUP is the class lines, GROUP_ROLES the role lines."""
    header, rows, aborts = None, [], []
    with open(path) as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            if r.get("_header"):
                header = r
            elif r.get("_abort"):
                aborts.append(r)
            elif r.get("group") == group:
                rows.append(r)
    return header or {}, rows, aborts


def _load_clocks(path):
    csv = _stem(path) + ".clocks.csv"
    if not os.path.exists(csv):
        return None
    out = []
    with open(csv) as f:
        next(f, None)
        for l in f:
            parts = [p.strip() for p in l.split(",")]
            try:
                out.append((float(parts[0]), float(parts[-1])))
            except (ValueError, IndexError):
                continue
    return out


def _cell_clock(clocks, r):
    if not clocks or "timing_window" not in r:
        return None
    t0, t1 = r["timing_window"]
    v = [c for t, c in clocks if t0 - 0.1 <= t <= t1 + 0.1]
    return statistics.median(v) if v else None


def _us(r, arm):
    a = r.get("arms", {}).get(arm)
    return a.get("us") if isinstance(a, dict) else None


def _f(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def _pct(a, b):
    """(a / b - 1) in percent."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or b <= 0:
        return "—"
    return f"{(a / b - 1) * 100:+.1f} %"


def _short(name):
    n = name[5:] if name.startswith("void ") else name
    for ch in "<(":
        i = n.find(ch)
        if i > 0:
            n = n[:i]
    return n[:80]


def _class_order(rows):
    seen = []
    for c in list(ALL_CLASSES):
        if any(r["class"] == c for r in rows):
            seen.append(c)
    for r in rows:
        if r["class"] not in seen:
            seen.append(r["class"])
    return seen


def format_one(label, path):
    header, rows, aborts = _load(path)
    clocks = _load_clocks(path)
    good = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]
    L = []
    dev = header.get("device") or (good[0]["device"] if good else "?")
    sm = header.get("sm") or (good[0]["sm"] if good else "?")
    L.append(f"## {label}: {dev} (sm_{sm})")
    L.append("")
    L.append(f"Source `{path}`; torch {header.get('torch')}, fso {header.get('fso_version')} "
             f"(build commit {header.get('fso_commit')}), imported from `{header.get('fso_file')}`; "
             f"{len(good)} cells measured, {len(errs)} failed"
             + (f", pass stopped early ({aborts[0]['reason']})" if aborts else "") + ".")
    if clocks:
        cv = [c for _, c in clocks]
        cells_clk = [x for x in (_cell_clock(clocks, r) for r in good) if x is not None]
        L.append(f"SM clock over the pass ({len(cv)} samples, 200 ms): median {statistics.median(cv):.0f} MHz, "
                 f"min {min(cv):.0f}, max {max(cv):.0f}; median over each cell's timed rounds: "
                 + (f"lowest cell {min(cells_clk):.0f} MHz, median cell {statistics.median(cells_clk):.0f} MHz."
                    if cells_clk else "no samples in the timed windows."))
    L.append("")
    if errs:
        L.append("Failed cells:")
        for r in errs:
            L.append(f"- `{r['class']}` M={r['M']}: {r['error'][:300]}")
        L.append("")

    by = {}
    for r in good:
        by.setdefault(r["class"], {})[r["M"]] = r
    L.append("Per cell (µs per call, CUDA-graph replay / weight copies, cold weights). Δ = fso_chain against "
             "bf16 (positive = fso slower). Kernels = launches of one eager call, fso_chain / bf16.")
    L.append("")
    for c in _class_order(good):
        cells = by[c]
        r0 = next(iter(cells.values()))
        L.append(f"### `{c}` (N={r0['N']}, K={r0['K']}, bias={'yes' if r0['has_bias'] else 'no'}, "
                 f"count {r0['count']})")
        L.append("")
        L.append("| M | bf16 µs | fso_chain µs | Δ chain vs bf16 | fso_gemm µs | fso_quant µs | bias_add µs "
                 "| kernels chain / bf16 | SM MHz | route |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---|")
        for m in sorted(cells):
            r = cells[m]
            a = r["arms"]
            clk = _cell_clock(clocks, r)
            L.append(f"| {m} | {_f(_us(r, 'bf16'))} | {_f(_us(r, 'fso_chain'))} | "
                     f"{_pct(_us(r, 'fso_chain'), _us(r, 'bf16'))} | {_f(_us(r, 'fso_gemm'))} | "
                     f"{_f(_us(r, 'fso_quant'))} | {_f(_us(r, 'bias_add'))} | "
                     f"{a['fso_chain'].get('kernels', '—')} / {a['bf16'].get('kernels', '—')} | "
                     f"{_f(clk, 0)} | {r.get('route') or '—'} |")
        L.append("")

    # Accuracy summary per class.
    L.append("Accuracy against the fp32 reference, worst cell of each class (cos = cosine similarity, "
             "max_rel = max|y - ref| / max|ref|).")
    L.append("")
    L.append("| class | bf16 min cos | bf16 max max_rel | fso_chain min cos | fso_chain max max_rel | "
             "fso_gemm min cos | gemm == dense.linear |")
    L.append("|:---|---:|---:|---:|---:|---:|:---:|")
    for c in _class_order(good):
        rs = list(by[c].values())

        def mn(arm, key, f=min):
            v = [r["arms"][arm].get(key) for r in rs if r["arms"].get(arm, {}).get(key) is not None]
            return f(v) if v else None
        eqs = [r.get("gemm_eq_dense_linear") for r in rs]
        eqs_s = "all" if all(e is True for e in eqs) else f"{sum(1 for e in eqs if e)}/{len(eqs)}"
        L.append(f"| `{c}` | {_f(mn('bf16', 'cos'), 6)} | {_f(mn('bf16', 'max_rel', max), 5)} | "
                 f"{_f(mn('fso_chain', 'cos'), 6)} | {_f(mn('fso_chain', 'max_rel', max), 5)} | "
                 f"{_f(mn('fso_gemm', 'cos'), 6)} | {eqs_s} |")
    L.append("")

    # Tower per call.
    tower = [t for t, *_ in TOWER_CLASSES]
    rungs = sorted({r["M"] for r in good if r["class"] in tower})
    if rungs:
        L.append("Audio tower, one call per rung: Σ over the five classes of count × µs (129 × 1280x1280, "
                 "32 × fc1, 32 × fc2, 1 × conv_out, 1 × proj2). `fso_gemm` alone is the floor of one launch "
                 "per linear with the activation already quantized. Launches = Σ count × kernels.")
        L.append("")
        L.append("| M | bf16 µs | fso_chain µs | Δ chain vs bf16 | fso_gemm µs | Δ gemm vs bf16 | "
                 "launches chain / bf16 | missing classes |")
        L.append("|---:|---:|---:|---:|---:|---:|:---:|:---|")
        for m in rungs:
            tot = {"bf16": 0.0, "fso_chain": 0.0, "fso_gemm": 0.0}
            launches = {"bf16": 0, "fso_chain": 0}
            missing = []
            for t, n, k, hb, cnt in TOWER_CLASSES:
                r = by.get(t, {}).get(m)
                if r is None:
                    missing.append(t)
                    continue
                for a in tot:
                    tot[a] += cnt * _us(r, a)
                for a in launches:
                    launches[a] += cnt * r["arms"][a].get("kernels", 0)
            L.append(f"| {m} | {_f(tot['bf16'], 1)} | {_f(tot['fso_chain'], 1)} | "
                     f"{_pct(tot['fso_chain'], tot['bf16'])} | {_f(tot['fso_gemm'], 1)} | "
                     f"{_pct(tot['fso_gemm'], tot['bf16'])} | {launches['fso_chain']} / {launches['bf16']} | "
                     f"{', '.join(missing) or '—'} |")
        L.append("")

    # LLM step.
    for qkv, o in (("llm_qkv", "llm_o"), ("llm_qkv_tp2", "llm_o_tp2")):
        if qkv not in by and o not in by:
            continue
        ms = sorted(set(by.get(qkv, {})) | set(by.get(o, {})))
        L.append(f"LLM, one step: 48 × (`{qkv}` + `{o}`) per batch size M (decode band M <= 64, extend above).")
        L.append("")
        L.append("| M | band | bf16 µs | fso_chain µs | Δ chain vs bf16 | fso_gemm µs | Δ gemm vs bf16 | "
                 "missing |")
        L.append("|---:|:---|---:|---:|---:|---:|---:|:---|")
        for m in ms:
            tot = {"bf16": 0.0, "fso_chain": 0.0, "fso_gemm": 0.0}
            missing = []
            for t in (qkv, o):
                r = by.get(t, {}).get(m)
                if r is None:
                    missing.append(t)
                    continue
                for a in tot:
                    tot[a] += 48 * _us(r, a)
            L.append(f"| {m} | {'decode' if m <= 64 else 'extend'} | {_f(tot['bf16'], 1)} | "
                     f"{_f(tot['fso_chain'], 1)} | {_pct(tot['fso_chain'], tot['bf16'])} | "
                     f"{_f(tot['fso_gemm'], 1)} | {_pct(tot['fso_gemm'], tot['bf16'])} | "
                     f"{', '.join(missing) or '—'} |")
        L.append("")
    return "\n".join(L) + "\n"


def spread_main(srcs):
    if len(srcs) != 2:
        sys.exit("--spread takes exactly two LABEL=path entries")
    (la, pa), (lb, pb) = srcs
    _, ra, _ = _load(pa)
    _, rb, _ = _load(pb)
    ka = {(r["class"], r["M"]): r for r in ra if "error" not in r}
    kb = {(r["class"], r["M"]): r for r in rb if "error" not in r}
    common = sorted(set(ka) & set(kb), key=lambda x: (list(ALL_CLASSES).index(x[0])
                                                       if x[0] in ALL_CLASSES else 99, x[1]))
    L = [f"Cross-pass spread, `{lb}` against `{la}`: |us_{lb} / us_{la} - 1| per cell, over the "
         f"{len(common)} cells both passes measured.", ""]
    L.append("| arm | scope | cells | median abs Δ | max abs Δ | cell of the max | signed median Δ |")
    L.append("|:---|:---|---:|---:|---:|:---|---:|")
    for arm in ARM_ORDER:
        for scope, pred in (("all", lambda c: True), ("tower", lambda c: c[0].startswith("tower_")),
                            ("llm", lambda c: c[0].startswith("llm_"))):
            d = []
            for c in common:
                if not pred(c):
                    continue
                ua, ub = _us(ka[c], arm), _us(kb[c], arm)
                if ua and ub:
                    d.append(((ub / ua - 1) * 100, c))
            if not d:
                continue
            ab = [abs(v) for v, _ in d]
            mx = max(d, key=lambda t: abs(t[0]))
            L.append(f"| {arm} | {scope} | {len(d)} | {statistics.median(ab):.2f} % | {max(ab):.2f} % | "
                     f"`{mx[1][0]}` M={mx[1][1]} ({mx[0]:+.2f} %) | {statistics.median([v for v, _ in d]):+.2f} % |")
    only_a = sorted(set(ka) - set(kb))
    only_b = sorted(set(kb) - set(ka))
    if only_a or only_b:
        L.append("")
        L.append(f"Cells in only one pass: {la}: {only_a or 'none'}; {lb}: {only_b or 'none'}.")
    L.append("")
    L.append("Within-cell spread of the three timed rounds, (max - min) / median of `us_rounds`, per pass and arm; "
             "cells above 5 % are counted.")
    L.append("")
    L.append("| pass | arm | cells | median range | max range | cell of the max | cells > 5 % |")
    L.append("|:---|:---|---:|---:|---:|:---|---:|")
    for lbl, kk in ((la, ka), (lb, kb)):
        for arm in ARM_ORDER:
            d = []
            for c, r in kk.items():
                v = r.get("arms", {}).get(arm)
                if not v or not v.get("us_rounds"):
                    continue
                rs = v["us_rounds"]
                med = statistics.median(rs)
                if med > 0:
                    d.append(((max(rs) - min(rs)) / med * 100, c))
            if not d:
                continue
            mx = max(d)
            L.append(f"| {lbl} | {arm} | {len(d)} | {statistics.median([v for v, _ in d]):.2f} % | {mx[0]:.2f} % | "
                     f"`{mx[1][0]}` M={mx[1][1]} | {sum(1 for v, _ in d if v > 5)} |")
    return "\n".join(L) + "\n"


def checks_one(label, path):
    header, rows, aborts = _load(path)
    good = [r for r in rows if "error" not in r]
    L = [f"### Checks, {label} (`{path}`)", ""]
    sm = header.get("sm") or (good[0]["sm"] if good else 0)
    l2 = header.get("l2_bytes") or (good[0].get("l2_bytes") if good else None)
    errs = [r for r in rows if "error" in r]
    L.append(f"- Cells: {len(good)} measured, {len(errs)} failed"
             + (f"; pass stopped: {aborts[0]['reason']}" if aborts else "") + ".")
    for r in errs:
        L.append(f"  - failed `{r['class']}` M={r['M']}: {r['error'][:300]}")
    # Cold protocol.
    short = []
    for r in good:
        for a, v in r["arms"].items():
            if v.get("copy_bytes") and l2 and v["weight_copies"] * v["copy_bytes"] < 2 * l2:
                short.append(f"{r['class']} M={r['M']} {a}")
    cold_err = [r for r in errs if "COLD-PROTOCOL" in r.get("error", "")]
    mins = []
    for r in good:
        for a in ("bf16", "fso_chain", "fso_gemm"):
            v = r["arms"].get(a)
            if v and v.get("copy_bytes") and l2:
                mins.append((v["weight_copies"] * v["copy_bytes"] / l2, r["class"], r["M"], a))
    mn = min(mins) if mins else None
    L.append(f"- Cold protocol: {len(short) + len(cold_err)} arm(s) below R x bytes >= 2 x L2"
             + (f" ({short + [r['class'] for r in cold_err]})" if short or cold_err else "")
             + (f"; the smallest rotation is {mn[0]:.2f} x L2 (`{mn[1]}` M={mn[2]} {mn[3]})." if mn else "."))
    # sm_90 equality.
    if sm // 10 == 9:
        eq = [r.get("sm90_gemm_eq_chain") for r in good]
        bad = [f"{r['class']} M={r['M']}" for r in good if r.get("sm90_gemm_eq_chain") is not True]
        L.append(f"- sm_90 `fso_gemm` (quantize_1x128_fp8 + linear_fp8) equals `fso.dense.linear` (linear_qx) "
                 f"byte for byte on {sum(1 for e in eq if e is True)} of {len(eq)} cells"
                 + (f"; not on {bad}." if bad else "."))
    else:
        bad = [f"{r['class']} M={r['M']}" for r in good if r.get("gemm_eq_dense_linear") is not True]
        L.append(f"- `fso_gemm` equals `fso.dense.linear` byte for byte on {len(good) - len(bad)} of {len(good)} cells"
                 + (f"; not on {bad}." if bad else "."))
    # Kernel counts per class and arm, with the distinct kernel-name sequences.
    L.append("- Kernel launches of one eager call, per class: count -> the M values with that count, then the "
             "distinct kernel sequences (short names; memsets / memcpys in brackets).")
    by = {}
    for r in good:
        by.setdefault(r["class"], []).append(r)
    for c in _class_order(good):
        rs = sorted(by[c], key=lambda r: r["M"])
        L.append(f"  - `{c}`:")
        for a in ("bf16", "fso_chain", "fso_gemm"):
            cnt, seqs = {}, {}
            for r in rs:
                v = r["arms"].get(a, {})
                cnt.setdefault(v.get("kernels"), []).append(r["M"])
                seq = " + ".join(_short(n) for n in v.get("kernel_names", []))
                if v.get("other_gpu_ops"):
                    seq += " [" + ", ".join(v["other_gpu_ops"]) + "]"
                seqs.setdefault(seq, []).append(r["M"])
            cs = "; ".join(f"{k} at M={_mlist(ms)}" for k, ms in sorted(cnt.items(), key=lambda t: str(t[0])))
            L.append(f"    - {a}: {cs}")
            for seq, ms in seqs.items():
                L.append(f"      - M={_mlist(ms)}: {seq}")
    # Sanity against the issue's B300 numbers.
    if sm // 10 == 10:
        L.append("- Sanity against the issue's B300 numbers (fso 0.2.1), agreement = within 15 %:")
        for cls, ms, arm, ref in ISSUE_REF:
            for m in ms:
                r = next((r for r in good if r["class"] == cls and r["M"] == m), None)
                if r is None:
                    L.append(f"  - `{cls}` M={m} {arm}: not measured.")
                    continue
                u = _us(r, arm)
                dev = (u / ref - 1) * 100
                L.append(f"  - `{cls}` M={m} {arm}: {u:.2f} µs against the issue's ~{ref} µs ({dev:+.1f} %), "
                         f"{'agrees' if abs(dev) <= 15 else 'DISAGREES'}.")
        r = next((r for r in good if r["class"] == "tower_conv_out" and r["M"] == 1079), None)
        if r is not None:
            c, b = _us(r, "fso_chain"), _us(r, "bf16")
            L.append(f"  - `tower_conv_out` M=1079: fso_chain {c:.2f} µs, bf16 {b:.2f} µs; the issue has the chain "
                     f"faster, here it is {'faster' if c < b else 'NOT faster'} ({_pct(c, b)}).")
    return "\n".join(L) + "\n"


def _mlist(ms):
    ms = sorted(ms)
    return ",".join(str(m) for m in ms) if len(ms) <= 8 else f"{ms[0]}..{ms[-1]} ({len(ms)} values)"


# ---------------------------------------------------------------------------
# Rendering of the layer-role lines (--roles) and their composites
# ---------------------------------------------------------------------------
def _arm_txt(v, nd=2):
    """An arm entry as table text: its µs, `n/a` for an arm the library refused, `err` for a failed one, `—` when
    the cell has no such arm."""
    if not isinstance(v, dict):
        return "—"
    if "na" in v:
        return "n/a"
    if "error" in v:
        return "err"
    u = v.get("us")
    return f"{u:.{nd}f}" if isinstance(u, (int, float)) else "—"


def _role_label(role):
    """The role and the roles its cell also stands for (q -> "q, k, v")."""
    al = ALL_ROLES[role][7] if role in ALL_ROLES else ()
    return ", ".join([role] + list(al))


def _role_order(rows):
    seen = [r for r in ALL_ROLES if any(x.get("role") == r for x in rows)]
    for x in rows:
        if x.get("role") not in seen:
            seen.append(x.get("role"))
    return seen


def _role_index(rows):
    """role -> M -> line, over every role line (failed ones too, so a composite can say a cell failed)."""
    by = {}
    for r in rows:
        by.setdefault(r.get("role"), {})[r.get("M")] = r
    return by


def _role_count(role):
    return ALL_ROLES[role][6]


def tower_terms(composite):
    """The terms of one tower-call composite, as (multiplier, role, arm, label): the composite is
    Σ multiplier x µs(role, arm). Multipliers come from the role table (q counts once per alias too)."""
    if composite in ("bf16", "fso_today", "fso_epi"):
        return [(cnt * (1 + len(al)), role, composite, _role_label(role))
                for role, _n, _k, _b, _r, _a, cnt, al in TOWER_ROLES]
    if composite not in ("fso_best_fsoquant", "fso_best_producerquant"):
        raise ValueError(f"unknown tower composite {composite!r}")
    with_quant = composite == "fso_best_fsoquant"
    n_qkv = 1 + len(ALL_ROLES["q"][7])
    plan = [
        (_role_count("q"), "q", "fso_quant", "q, k, v (one shared quantize per layer)", True),
        (_role_count("q") * n_qkv, "q", "fso_epi_2b", _role_label("q"), False),
        (_role_count("out_proj"), "out_proj", "fso_epi", "out_proj", False),
        (_role_count("fc1"), "fc1", "fso_quant", "fc1", True),
        (_role_count("fc1"), "fc1", "fso_q_2b", "fc1", False),
        (_role_count("fc2"), "fc2", "fso_epi_2b", "fc2", False),
        (_role_count("conv_out"), "conv_out", "fso_epi", "conv_out", False),
        (_role_count("proj1"), "proj1", "fso_quant", "proj1", True),
        (_role_count("proj1"), "proj1", "fso_q_2b", "proj1", False),
        (_role_count("proj2"), "proj2", "fso_epi_2b", "proj2", False),
    ]
    return [(m, role, arm, lbl) for m, role, arm, lbl, is_quant in plan if with_quant or not is_quant]


def llm_terms(composite, qkv, o):
    """The terms of one LLM-step composite for the pair (qkv, o), as in tower_terms."""
    cq, co = _role_count(qkv), _role_count(o)
    if composite in ("bf16", "fso_today"):
        return [(cq, qkv, composite, qkv), (co, o, composite, o)]
    if composite == "fso_best_fsoquant":
        return [(cq, qkv, "fso_quant", qkv), (cq, qkv, "fso_epi_2b", qkv), (co, o, "fso_epi", o)]
    if composite == "fso_best_producerquant":
        return [(cq, qkv, "fso_epi_2b", qkv), (co, o, "fso_epi", o)]
    raise ValueError(f"unknown LLM composite {composite!r}")


def _term(by, role, m, arm):
    """(status, µs, kernels) of one composite term; status is "ok", "n/a" (the library refused the call),
    "error" (the arm failed), "cell failed" or "not measured"."""
    r = by.get(role, {}).get(m)
    if r is None:
        return "not measured", None, None
    if "error" in r:
        return "cell failed", None, None
    v = r.get("arms", {}).get(arm)
    if not isinstance(v, dict):
        return "not measured", None, None
    if "na" in v:
        return "n/a", None, None
    if "error" in v:
        return "error", None, None
    if not isinstance(v.get("us"), (int, float)):
        return "not measured", None, None
    return "ok", v["us"], v.get("kernels")


def composite(by, m, terms):
    """{"us": Σ multiplier x µs or None, "launches": Σ multiplier x kernels or None, "missing": [(status, arm,
    label), ...]}. Any term that is not "ok" makes the composite n/a (us None) and is listed in "missing"."""
    tot, launches, missing = 0.0, 0, []
    for mult, role, arm, label in terms:
        st, us, kern = _term(by, role, m, arm)
        if st != "ok":
            missing.append((st, arm, label))
            continue
        tot += mult * us
        launches = None if (launches is None or not isinstance(kern, int)) else launches + mult * kern
    if missing:
        return {"us": None, "launches": None, "missing": missing}
    return {"us": tot, "launches": launches, "missing": []}


def _missing_txt(missing):
    """The missing terms of a composite, grouped by status and arm: "n/a fso_epi_2b: q, k, v, fc2; ..."."""
    groups = {}
    for st, arm, label in missing:
        groups.setdefault((st, arm), []).append(label)
    return "; ".join(f"{st} {arm}: {', '.join(lbls)}" for (st, arm), lbls in groups.items())


def _epi_txt(r):
    parts = ["bias" if r.get("has_bias") else "no bias"]
    if r.get("has_residual"):
        parts.append("residual")
    if r.get("activation"):
        parts.append(r["activation"])
    return " + ".join(parts)


def _eq_txt(v):
    return "yes" if v is True else ("NO" if v is False else "—")


def format_roles(label, path):
    header, rows, aborts = _load(path, GROUP_ROLES)
    clocks = _load_clocks(path)
    good = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]
    L = []
    dev = header.get("device") or (good[0]["device"] if good else "?")
    sm = header.get("sm") or (good[0]["sm"] if good else "?")
    L.append(f"## {label}: {dev} (sm_{sm}), layer roles")
    L.append("")
    L.append(f"Source `{path}`; torch {header.get('torch')}, fso {header.get('fso_version')} "
             f"(build commit {header.get('fso_commit')}), imported from `{header.get('fso_file')}`; "
             f"{len(good)} role cells measured, {len(errs)} failed"
             + (f", pass stopped early ({aborts[0]['reason']})" if aborts else "") + ".")
    if clocks:
        cv = [c for _, c in clocks]
        cells_clk = [x for x in (_cell_clock(clocks, r) for r in good) if x is not None]
        L.append(f"SM clock over the pass ({len(cv)} samples, 200 ms): median {statistics.median(cv):.0f} MHz, "
                 f"min {min(cv):.0f}, max {max(cv):.0f}; median over each cell's timed rounds: "
                 + (f"lowest cell {min(cells_clk):.0f} MHz, median cell {statistics.median(cells_clk):.0f} MHz."
                    if cells_clk else "no samples in the timed windows."))
    L.append("")
    if errs:
        L.append("Failed cells:")
        for r in errs:
            L.append(f"- `{r.get('role')}` M={r.get('M')}: {r['error'][:300]}")
        L.append("")

    by_all = _role_index(rows)
    by = _role_index(good)
    L.append("Per role (µs per call, CUDA-graph replay / weight copies, cold weights). `n/a` = the library refused "
             "the call with NotImplementedError (the reason is listed under the table), `—` = not an arm of this "
             "role. Kernels = launches of one eager call per arm that ran, in the column order. "
             "epi_2b == epi: form 2b gave the bytes of the bf16-x call.")
    L.append("")
    for role in _role_order(good):
        cells = by[role]
        r0 = next(iter(cells.values()))
        al = r0.get("aliases") or []
        also = f" (stands for {', '.join([role] + list(al))})" if al else ""
        L.append(f"### `{role}`{also}: N={r0['N']}, K={r0['K']}, {_epi_txt(r0)}, count {r0['count']}"
                 + (" each" if al else ""))
        L.append("")
        L.append("| M | bf16 µs | fso_today µs | fso_epi µs | fso_epi_2b µs | fso_q_2b µs | fso_quant µs | "
                 "kernels (ran) | epi_2b == epi | SM MHz | route |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|:---|:---:|---:|:---|")
        reasons = {}
        for m in sorted(cells):
            r = cells[m]
            a = r["arms"]
            kern = " / ".join(f"{ROLE_ARM_SHORT[k]} {a[k]['kernels']}" for k in ROLE_ARM_ORDER
                              if k in a and isinstance(a[k], dict) and "kernels" in a[k])
            for k in ROLE_ARM_ORDER:
                if isinstance(a.get(k), dict) and "na" in a[k]:
                    reasons.setdefault((k, a[k]["na"]), []).append(m)
            clk = _cell_clock(clocks, r)
            route = r.get("route") or "—"
            if r.get("route_plan"):
                route += f" [{r['route_plan']}]"
            L.append(f"| {m} | " + " | ".join(_arm_txt(a.get(k)) for k in ROLE_ARM_ORDER)
                     + f" | {kern or '—'} | {_eq_txt(r.get('epi_2b_eq_epi'))} | {_f(clk, 0)} | {route} |")
        L.append("")
        if reasons:
            L.append("n/a reasons:")
            for (k, msg), ms in reasons.items():
                L.append(f"- `{k}` at M={_mlist(ms)}: {msg}")
            L.append("")

    # Accuracy per role, worst cell.
    L.append("Accuracy against the fp32 reference act(x @ w^T + bias) + residual, worst cell of each role "
             "(cos = cosine similarity, max_rel = max|y - ref| / max|ref|), over the cells where the arm ran; "
             "q_2b == quantize(epi_2b) counts the cells where fso_q_2b gave the bytes of quantize_activation "
             "applied to the fso_epi_2b result.")
    L.append("")
    L.append("| role | bf16 cos / max_rel | fso_today cos / max_rel | fso_epi cos / max_rel | "
             "fso_epi_2b cos / max_rel | epi_2b == epi | q_2b == quantize(epi_2b) |")
    L.append("|:---|---:|---:|---:|---:|:---:|:---:|")
    for role in _role_order(good):
        rs = list(by[role].values())

        def worst(arm):
            cs = [r["arms"][arm]["cos"] for r in rs if isinstance(r["arms"].get(arm), dict)
                  and r["arms"][arm].get("cos") is not None]
            mr = [r["arms"][arm]["max_rel"] for r in rs if isinstance(r["arms"].get(arm), dict)
                  and r["arms"][arm].get("max_rel") is not None]
            if not cs:
                return "—"
            return f"{min(cs):.6f} / {_f(max(mr) if mr else None, 5)}"

        def count(key):
            v = [r.get(key) for r in rs if r.get(key) is not None]
            if not v:
                return "—"
            return f"{sum(1 for e in v if e is True)}/{len(v)}"
        L.append(f"| `{role}` | {worst('bf16')} | {worst('fso_today')} | {worst('fso_epi')} | "
                 f"{worst('fso_epi_2b')} | {count('epi_2b_eq_epi')} | {count('q_2b_eq_unfused')} |")
    L.append("")

    # Tower per call.
    tower = [r[0] for r in TOWER_ROLES]
    rungs = sorted({r["M"] for r in rows if r.get("role") in tower})
    if rungs:
        L.append("Audio tower, one call per rung, from the role cells: each composite is Σ multiplier × µs of its "
                 "terms, and launches Σ multiplier × kernels. `bf16`, `fso_today` and `fso_epi` sum that arm over "
                 "every role (q counts 3 × 32, for q, k and v; out_proj, fc1, fc2 32; conv_out, proj1, proj2 1). "
                 "`fso_best_fsoquant` = 32 × [fso_quant(q) shared by q, k, v + 3 × fso_epi_2b(q) + fso_epi(out_proj) "
                 "+ fso_quant(fc1) + fso_q_2b(fc1) + fso_epi_2b(fc2)] + fso_epi(conv_out) + fso_quant(proj1) + "
                 "fso_q_2b(proj1) + fso_epi_2b(proj2); `fso_best_producerquant` is the same with every fso_quant "
                 "term zero (the caller's LayerNorm writes the contract layout). A composite with a term that is "
                 "n/a, failed or not measured is n/a, and the last column names those terms.")
        L.append("")
        L.append("| M | composite | µs | Δ vs bf16 | launches | n/a terms |")
        L.append("|---:|:---|---:|---:|---:|:---|")
        for m in rungs:
            base = composite(by_all, m, tower_terms("bf16"))
            for name in TOWER_COMPOSITES:
                c = composite(by_all, m, tower_terms(name))
                us = _f(c["us"], 1) if c["us"] is not None else "n/a"
                d = _pct(c["us"], base["us"]) if name != "bf16" else "—"
                L.append(f"| {m} | `{name}` | {us} | {d} | "
                         f"{c['launches'] if c['launches'] is not None else '—'} | "
                         f"{_missing_txt(c['missing']) or '—'} |")
        L.append("")

    # LLM step.
    for qkv, o in LLM_ROLE_PAIRS:
        if qkv not in by_all and o not in by_all:
            continue
        ms = sorted(set(by_all.get(qkv, {})) | set(by_all.get(o, {})))
        L.append(f"LLM, one step per batch size M (decode band M <= 64, extend above), from the role cells: "
                 f"`bf16` and `fso_today` = {_role_count(qkv)} × {qkv} + {_role_count(o)} × {o} of that arm; "
                 f"`fso_best_fsoquant` = {_role_count(qkv)} × (fso_quant({qkv}) + fso_epi_2b({qkv})) + "
                 f"{_role_count(o)} × fso_epi({o}); `fso_best_producerquant` drops the fso_quant term (the caller's "
                 f"RMSNorm writes the contract layout). Launches in the column order bf16 / today / fsoquant / "
                 f"producerquant.")
        L.append("")
        L.append("| M | band | bf16 µs | fso_today µs | Δ today | fso_best_fsoquant µs | Δ fsoquant | "
                 "fso_best_producerquant µs | Δ producerquant | launches | n/a terms |")
        L.append("|---:|:---|---:|---:|---:|---:|---:|---:|---:|:---|:---|")
        for m in ms:
            c = {name: composite(by_all, m, llm_terms(name, qkv, o)) for name in LLM_COMPOSITES}
            b = c["bf16"]["us"]

            def u(name):
                return _f(c[name]["us"], 1) if c[name]["us"] is not None else "n/a"
            launches = " / ".join(str(c[n]["launches"]) if c[n]["launches"] is not None else "—"
                                  for n in LLM_COMPOSITES)
            miss = "; ".join(f"{n}: {_missing_txt(c[n]['missing'])}" for n in LLM_COMPOSITES if c[n]["missing"])
            L.append(f"| {m} | {'decode' if m <= 64 else 'extend'} | {u('bf16')} | {u('fso_today')} | "
                     f"{_pct(c['fso_today']['us'], b)} | {u('fso_best_fsoquant')} | "
                     f"{_pct(c['fso_best_fsoquant']['us'], b)} | {u('fso_best_producerquant')} | "
                     f"{_pct(c['fso_best_producerquant']['us'], b)} | {launches} | {miss or '—'} |")
        L.append("")
    return "\n".join(L) + "\n"


def spread_roles(srcs):
    """--spread for the role lines: cross-pass and within-cell spread per arm, as spread_main."""
    (la, pa), (lb, pb) = srcs
    _, ra, _ = _load(pa, GROUP_ROLES)
    _, rb, _ = _load(pb, GROUP_ROLES)
    ka = {(r["role"], r["M"]): r for r in ra if "error" not in r}
    kb = {(r["role"], r["M"]): r for r in rb if "error" not in r}
    order = list(ALL_ROLES)
    common = sorted(set(ka) & set(kb), key=lambda x: (order.index(x[0]) if x[0] in order else 99, x[1]))
    tower = {r[0] for r in TOWER_ROLES}
    L = [f"Cross-pass spread of the role cells, `{lb}` against `{la}`: |us_{lb} / us_{la} - 1| per cell and arm, "
         f"over the {len(common)} cells both passes measured (arms that ran in both).", ""]
    L.append("| arm | scope | cells | median abs Δ | max abs Δ | cell of the max | signed median Δ |")
    L.append("|:---|:---|---:|---:|---:|:---|---:|")
    for arm in ROLE_ARM_ORDER:
        for scope, pred in (("all", lambda c: True), ("tower", lambda c: c[0] in tower),
                            ("llm", lambda c: c[0] not in tower)):
            d = []
            for c in common:
                if not pred(c):
                    continue
                ua, ub = _us(ka[c], arm), _us(kb[c], arm)
                if ua and ub:
                    d.append(((ub / ua - 1) * 100, c))
            if not d:
                continue
            ab = [abs(v) for v, _ in d]
            mx = max(d, key=lambda t: abs(t[0]))
            L.append(f"| {arm} | {scope} | {len(d)} | {statistics.median(ab):.2f} % | {max(ab):.2f} % | "
                     f"`{mx[1][0]}` M={mx[1][1]} ({mx[0]:+.2f} %) | {statistics.median([v for v, _ in d]):+.2f} % |")
    only_a = sorted(set(ka) - set(kb))
    only_b = sorted(set(kb) - set(ka))
    if only_a or only_b:
        L.append("")
        L.append(f"Role cells in only one pass: {la}: {only_a or 'none'}; {lb}: {only_b or 'none'}.")
    L.append("")
    L.append("Within-cell spread of the three timed rounds, (max - min) / median of `us_rounds`, per pass and arm; "
             "cells above 5 % are counted.")
    L.append("")
    L.append("| pass | arm | cells | median range | max range | cell of the max | cells > 5 % |")
    L.append("|:---|:---|---:|---:|---:|:---|---:|")
    for lbl, kk in ((la, ka), (lb, kb)):
        for arm in ROLE_ARM_ORDER:
            d = []
            for c, r in kk.items():
                v = r.get("arms", {}).get(arm)
                if not isinstance(v, dict) or not v.get("us_rounds"):
                    continue
                rs = v["us_rounds"]
                med = statistics.median(rs)
                if med > 0:
                    d.append(((max(rs) - min(rs)) / med * 100, c))
            if not d:
                continue
            mx = max(d)
            L.append(f"| {lbl} | {arm} | {len(d)} | {statistics.median([v for v, _ in d]):.2f} % | {mx[0]:.2f} % | "
                     f"`{mx[1][0]}` M={mx[1][1]} | {sum(1 for v, _ in d if v > 5)} |")
    return "\n".join(L) + "\n"


def checks_roles(label, path):
    header, rows, aborts = _load(path, GROUP_ROLES)
    good = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]
    l2 = header.get("l2_bytes") or (good[0].get("l2_bytes") if good else None)
    L = [f"### Checks, {label}, layer roles (`{path}`)", ""]
    L.append(f"- Role cells: {len(good)} measured, {len(errs)} failed"
             + (f"; pass stopped: {aborts[0]['reason']}" if aborts else "") + ".")
    for r in errs:
        L.append(f"  - failed `{r.get('role')}` M={r.get('M')}: {r['error'][:300]}")
    short, mins = [], []
    for r in good:
        for a, v in r["arms"].items():
            if isinstance(v, dict) and v.get("copy_bytes") and l2:
                ratio = v["weight_copies"] * v["copy_bytes"] / l2
                mins.append((ratio, r["role"], r["M"], a))
                if ratio < 2:
                    short.append(f"{r['role']} M={r['M']} {a}")
    mn = min(mins) if mins else None
    L.append(f"- Cold protocol: {len(short)} arm(s) below R x bytes >= 2 x L2" + (f" ({short})" if short else "")
             + (f"; the smallest rotation is {mn[0]:.2f} x L2 (`{mn[1]}` M={mn[2]} {mn[3]})." if mn else "."))
    # n/a arms.
    na = {}
    for r in good:
        for a in ROLE_ARM_ORDER:
            v = r["arms"].get(a)
            if isinstance(v, dict) and "na" in v:
                na.setdefault(a, {}).setdefault(r["role"], []).append(r["M"])
    if na:
        L.append("- Arms refused by the library (n/a, NotImplementedError on the eager probe), role -> M:")
        for a in ROLE_ARM_ORDER:
            if a in na:
                L.append(f"  - {a}: " + "; ".join(f"`{role}` M={_mlist(ms)}" for role, ms in na[a].items()))
    else:
        L.append("- No arm was refused by the library.")
    ran = {a: sum(1 for r in good if isinstance(r["arms"].get(a), dict) and "us" in r["arms"][a])
           for a in ROLE_ARM_ORDER}
    L.append("- Arms timed: " + ", ".join(f"{a} {n}" for a, n in ran.items()) + f" (of {len(good)} cells).")
    for key, what in (("epi_2b_eq_epi", "fso_epi_2b gives the bytes of fso_epi"),
                      ("q_2b_eq_unfused", "fso_q_2b gives the bytes of quantize_activation(fso_epi_2b)")):
        v = [(r["role"], r["M"], r.get(key)) for r in good if r.get(key) is not None]
        bad = [f"{role} M={m}" for role, m, e in v if e is not True]
        L.append(f"- {what}: on {len(v) - len(bad)} of the {len(v)} cells where both arms ran"
                 + (f"; not on {bad}." if bad else "."))
    L.append("- Kernel launches of one eager call, per role and arm that ran: count -> the M values with that "
             "count, then the distinct kernel sequences (short names; memsets / memcpys in brackets).")
    by = {}
    for r in good:
        by.setdefault(r["role"], []).append(r)
    for role in _role_order(good):
        rs = sorted(by[role], key=lambda r: r["M"])
        L.append(f"  - `{role}`:")
        for a in ROLE_ARM_ORDER:
            cnt, seqs = {}, {}
            for r in rs:
                v = r["arms"].get(a)
                if not isinstance(v, dict) or "kernels" not in v:
                    continue
                cnt.setdefault(v.get("kernels"), []).append(r["M"])
                seq = " + ".join(_short(n) for n in v.get("kernel_names", []))
                if v.get("other_gpu_ops"):
                    seq += " [" + ", ".join(v["other_gpu_ops"]) + "]"
                seqs.setdefault(seq, []).append(r["M"])
            if not cnt:
                continue
            cs = "; ".join(f"{k} at M={_mlist(ms)}" for k, ms in sorted(cnt.items(), key=lambda t: str(t[0])))
            L.append(f"    - {a}: {cs}")
            for seq, ms in seqs.items():
                L.append(f"      - M={_mlist(ms)}: {seq}")
    return "\n".join(L) + "\n"


def _has_group(path, group):
    return bool(_load(path, group)[1])


def _render_file(fn_classes, fn_roles, label, path):
    """The class rendering of a file (unchanged) and, when the file has role lines, the role rendering; a file with
    role lines only gets the role rendering alone."""
    has_roles = _has_group(path, GROUP_ROLES)
    out = ""
    if not has_roles or _has_group(path, GROUP):
        out += fn_classes(label, path)
    if has_roles:
        out += fn_roles(label, path)
    return out


def _pairs(srcs):
    out = []
    for spec in srcs or []:
        if "=" not in spec:
            sys.exit(f"--from entry must be LABEL=path, got {spec!r}")
        out.append(tuple(spec.split("=", 1)))
    if not out:
        sys.exit("--from LABEL=path required")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--run", action="store_true", help="run the grid (parent; no CUDA context)")
    p.add_argument("--out", help="output jsonl (with --run)")
    p.add_argument("--machine", help="machine label recorded in every line (default: hostname)")
    p.add_argument("--classes", help="comma list of classes; default: tower + the LLM set of --llm-tp")
    p.add_argument("--Ms", help="comma list of M values replacing every selected class's M grid")
    p.add_argument("--tower-Ms", dest="tower_Ms",
                   help="comma list of M values replacing the grid of the tower classes / roles (over --Ms)")
    p.add_argument("--llm-Ms", dest="llm_Ms",
                   help="comma list of M values replacing the grid of the LLM classes / roles (over --Ms)")
    p.add_argument("--roles", nargs="?", const="all", default=None,
                   help="with --run: time the layer roles instead of the classes; alone every tower role and the "
                        "LLM roles of --llm-tp, or a comma list (q,out_proj,fc1,fc2,conv_out,proj1,proj2,qkv,o,"
                        "qkv_tp2,o_tp2; k and v select q)")
    p.add_argument("--llm-tp", type=int, choices=(1, 2), default=None,
                   help="LLM projections unsharded (1) or as TP2 shards (2); default 2 on sm_120, else 1")
    p.add_argument("--smi-id", help="card id for nvidia-smi (default: CUDA_VISIBLE_DEVICES when it is a UUID)")
    p.add_argument("--no-sampler", action="store_true", help="do not run the nvidia-smi clock sampler")
    p.add_argument("--format", action="store_true", help="markdown tables from jsonl (--from LABEL=path ...)")
    p.add_argument("--spread", action="store_true", help="cross-pass spread of two jsonl (--from A=p1 B=p2)")
    p.add_argument("--checks", action="store_true", help="acceptance checks per jsonl (--from LABEL=path ...)")
    p.add_argument("--from", dest="srcs", nargs="+", help="LABEL=jsonl pairs")
    p.add_argument("--md-out", default="-", help="markdown output file (default stdout)")
    args = p.parse_args()

    if args.worker:
        worker_main()
        return
    if args.probe:
        probe_main()
        return
    if args.run:
        run_main(args)
        return
    if args.format or args.spread or args.checks:
        pairs = _pairs(args.srcs)
        if args.format:
            md = "".join(_render_file(format_one, format_roles, lbl, path) for lbl, path in pairs)
        elif args.spread:
            if len(pairs) != 2:
                sys.exit("--spread takes exactly two LABEL=path entries")
            has_roles = all(_has_group(path, GROUP_ROLES) for _, path in pairs)
            has_classes = any(_has_group(path, GROUP) for _, path in pairs)
            md = ""
            if has_classes or not has_roles:
                md += spread_main(pairs)
            if has_roles:
                md += spread_roles(pairs)
        else:
            md = "".join(_render_file(checks_one, checks_roles, lbl, path) for lbl, path in pairs)
        if args.md_out == "-":
            sys.stdout.write(md)
        else:
            with open(args.md_out, "w") as f:
                f.write(md)
        return
    p.print_help()


if __name__ == "__main__":
    main()
