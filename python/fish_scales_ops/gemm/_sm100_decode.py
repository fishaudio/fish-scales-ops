"""Dense MXFP8 decode kernels for sm_100/sm_103 (M ≤ 64).

This module is the tier-2 *decode* row of the sm_100/sm_103 MXFP8 router. It
sits in front of the three tiers described in :mod:`._sm100_dispatch` and takes
only the band M ≤ 64, where every one of the three shipped tiers loses:

* cuBLAS ``scaled_mm`` (tier 1) picks a 128- or 256-wide N tile and cannot be
  asked for a narrower one, so on a decode shape it leaves most of the machine
  idle and still pays a full-width epilogue.
* the persistent CuTe-DSL kernel of :mod:`._sm100_dsl` (tier 2) declines the
  whole band — its ``pick_config`` returns None below M = 256.
* the C++ cascade (tier 3) answers a narrow-N decode cell with a *two-kernel*
  split-K: one GemmUniversal launch writes FP32 partials to a workspace and a
  second launch reduces them. The second launch cannot start before the first
  drains, and at M = 1 it occupies 3 of 148 SMs, so the op serialises two grid
  launches and pays an extra L2 round trip for the partials.

The kernels used here come from the vendored NVIDIA/FlashInfer package in
``_vendor/flashinfer_sm100_blockscaled`` (BSD-3-Clause; see its README and
LICENSE). Two classes are used, and which one runs is decided by the shape:

``Sm100BlockScaledSplitKGemmKernel``
    A physical ``(1, 1, split_k)`` cluster computes ONE output tile. Each CTA
    accumulates a disjoint K slice in FP32 in its own TMEM; in the epilogue the
    output subtiles are owned round-robin by the CTAs and the three non-owners
    push their FP32 fragments into the owner's shared memory over
    ``mapa.shared::cluster`` + ``st.async.shared::cluster.mbarrier::complete_tx``.
    Nothing but the final BF16 output is ever written to global memory, so the
    reduction costs neither a workspace nor a second launch. This is what the
    narrow-N shapes (``wo``, ``down``) want.
``Sm100BlockScaledPersistentDenseGemmKernel``
    The plain persistent block-scaled kernel at the same narrow token tile. On
    a wide-N shape the tile grid already fills the machine, so splitting K only
    multiplies the work (measured: +100 % at split_k = 2 and +265 % at 4 on
    ``gate_up``); the single-slice persistent kernel wins instead.

    Above M = 32 the split-K class refuses the cell (its ``supports_m``), and
    this kernel's own ``can_implement`` refuses a token tile narrower than the
    token count, so the band 32 < M ≤ 64 runs on this kernel with a 64-wide
    token tile and the 2-CTA 256-row weight tile in a cluster of two: one
    tcgen05 MMA spans both CTAs of the cluster and the token operand is
    multicast to them, so each weight row is streamed once and the token
    stream is halved against the 128-row tile at cluster (1, 1).

Both are driven in SWAP-AB orientation: the kernel's A operand is the weight
``[N, K]`` and its B operand is the activation ``[M, K]``, so the narrow token
dimension becomes the kernel's N and the 8/16/32/64-wide MMA tile is the token
tile. fso's ``quantize_1x32_fp8`` already emits its opaque int32 UE8M0 buffer
in the CUTLASS ``Sm1xxBlockScaledConfig<32>`` atom layout, which is
byte-identical to the 128×4 swizzled layout these kernels read, so the scale
buffers are handed over as raw pointers — no repack, no copy, no extra memory.

Contracts and gates:

* **DSL floor.** The vendored sources use CuTe-DSL surfaces that do not exist
  in 4.4.2 (``cute.nvgpu.OperandMajorMode``, ``Numeric.bitcast``, the newer
  ``make_blockscaled_trivial_tiled_mma`` arity, ``cute.struct`` field handles).
  4.5.0 is the floor and 4.8.0 is the recommended pin (its JIT is 2–3× faster
  than 4.4.2's). Below the floor this module is INERT: ``_init`` returns None
  before importing the vendored package, ``pick_config`` returns None, and
  ``route`` behaves exactly as it did before this module existed. Set
  ``FSO_LOG=1`` to get the one-line reason, printed once per process on
  stderr (stdout belongs to the benches' per-cell JSON protocol).
* **Capture.** ``cute.compile`` runs on the first call for a given tactic and
  is cached per (kernel class, tiler, cluster, split_k, pdl), so the eager
  warm-up every caller already performs before ``torch.cuda.graph`` capture
  covers it; nothing compiles, allocates or synchronises inside capture. The
  launch binds ``data_ptr()`` values, so for the life of a captured graph the
  activation, the weight, the OUTPUT **and both scale buffers** must keep their
  addresses — the scale buffers are the part that is new relative to the rest
  of fso, because they are passed as bare pointers rather than as tensors.
* **Env kills.** ``FSO_DISABLE_DSL=1`` (shared with tier 2) and
  ``FSO_DISABLE_DECODE_DSL=1`` (this module only) both make it inert.
  ``FSO_PRINT_TILE_INFO=1`` prints the picked tactic once per distinct
  (M, N, K).

Provenance of the routing rule: the tactic sweep in the B300 run
``b300_mxfp8_20260917/M-D1`` (400 timed rows over four shapes × six M × every
tactic the split-K kernel accepts), the calling-convention A/B in ``M-D3``, the
DSL version table in ``M-D2``, and for the 32 < M ≤ 64 band the persistent
kernel sweep in ``b300_round3_20260922/M-D5`` (every tiler, cluster,
orientation and prefetch setting the kernel accepts, four shapes × four M,
two passes).
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch

# (mma_tiler_mn, cluster_shape_mn, swap_ab, split_k)
_ConfigT = Tuple[Tuple[int, int], Tuple[int, int], bool, int]

#: First nvidia-cutlass-dsl release that runs the vendored kernels (M-D2).
MIN_DSL_VERSION = (4, 5, 0)
#: Width of this router row. Up to the split-K kernel's own MAX_M (32) the
#: narrow-N shapes split K in-cluster; from there to 64 the persistent kernel
#: runs a 64-wide token tile (see ``_compute_config``).
MAX_DECODE_M = 64
#: Token tile of the persistent kernel above the split-K class's MAX_M, and
#: the widest token count the row takes at one tile.
_WIDE_TOKEN_TILE = 64
#: Shallowest K the wide-token form takes, in elements: eight mainloop K-tiles
#: of 128. On a K = 512 shape (Family C `shared_down`) the one-slice mainloop
#: is four tiles, shorter than the pipeline that fills it, and the form gained
#: nothing on the per-op slope against the cuBLAS tier while its single-op
#: replay flipped one ~2 µs tick at M = 48; at K = 2048 it wins clearly. The
#: floor sits between the two measured points (b300_round3_20260922/M-D5,
#: tables/SWEEP_BC.txt).
MIN_WIDE_TOKEN_K = 1024
#: Programmatic dependent launch. The kernels own both sides of the
#: griddepcontrol pair (wait at entry, launch-dependents at the end of the
#: mainloop), which is what makes it capture-safe here; this is unrelated to
#: the closed "PDL on a CUTLASS GEMM with no GDC compiled in" item.
_ENABLE_PDL = True
#: Shortest K slice worth giving a CTA, in elements: four mainloop K-tiles of
#: 128. Splitting K costs a wider cluster launch and an in-cluster reduction,
#: and below four tiles the mainloop is shorter than the pipeline that fills it,
#: so the split does not pay for itself. Measured on the Family C shared-expert
#: `down` projection (N = 2048, K = 512), where two slices cost +63 % and four
#: cost +150 % against one slice at M = 32 (b300_dense_decode_20260922,
#: tables/SHORTK.txt).
MIN_SLICE_K = 512
#: Backstop on the number of operand layouts kept hoisted (see
#: ``_operand_tensor``); one device buffer is retained per occupied slot.
_ARG_CACHE_LIMIT = 64

_STATE: Optional[dict] = None
_CFG_CACHE: dict = {}
_ARG_CACHE: dict = {}
_ALPHA: dict = {}
_LOGGED: set = set()
_MISSING = object()


def _log_once(msg: str) -> None:
    """Print ``msg`` at most once per process, and only under ``FSO_LOG=1``.

    The notice goes to stderr, never stdout: the benches run one subprocess
    per cell and parse that subprocess's whole stdout as JSON
    (``bench_qwen3_4b_mlp_forward.py``), so a diagnostic line on stdout turns
    every decode cell into a parse error wherever the notice fires, i.e. in any
    venv below the DSL floor (run b300_round3_20260922/M-A3).
    """
    if not os.getenv("FSO_LOG"):
        return
    if msg in _LOGGED:
        return
    _LOGGED.add(msg)
    print("fso: " + msg, file=sys.stderr, flush=True)


def _parse_version(raw: str) -> Tuple[int, int, int]:
    """``"4.8.0.dev0"`` → ``(4, 8, 0)``. Unparseable → ``(0, 0, 0)``."""
    parts = []
    for chunk in str(raw).split(".")[:3]:
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


def _init() -> Optional[dict]:
    """Lazy one-time init. None when the decode row is unavailable or gated off.

    Order matters: the DSL version is checked BEFORE the vendored package is
    imported, because on 4.4.2 that import itself raises (the kernel sources
    spell ``cute.nvgpu.OperandMajorMode`` in a type annotation, which Python
    evaluates while the class body executes).
    """
    global _STATE
    if _STATE is not None:
        return _STATE if _STATE.get("ok") else None
    _STATE = {"ok": False}
    if os.getenv("FSO_DISABLE_DSL") or os.getenv("FSO_DISABLE_DECODE_DSL"):
        return None
    try:
        import cutlass

        ver = _parse_version(getattr(cutlass, "__version__", ""))
        if ver < MIN_DSL_VERSION:
            _log_once(
                "sm_100 MXFP8 decode rows need nvidia-cutlass-dsl >= %d.%d.%d "
                "(found %s); leaving the M <= %d band to the existing tiers"
                % (MIN_DSL_VERSION + (getattr(cutlass, "__version__", "?"),
                                      MAX_DECODE_M)))
            return None

        import cuda.bindings.driver as cuda_drv
        import cutlass.cute as cute
        import cutlass.utils as dsl_utils
        from cutlass.cute.runtime import from_dlpack, make_ptr

        from ._vendor.flashinfer_sm100_blockscaled.dense_blockscaled_gemm_sm100 \
            import Sm100BlockScaledPersistentDenseGemmKernel as PK
        from ._vendor.flashinfer_sm100_blockscaled.dense_blockscaled_gemm_sm100_splitk \
            import Sm100BlockScaledSplitKGemmKernel as SK

        _STATE.update(ok=True, cuda=cuda_drv, cutlass=cutlass, cute=cute,
                      utils=dsl_utils, from_dlpack=from_dlpack,
                      make_ptr=make_ptr, PK=PK, SK=SK, compiled={},
                      dsl_version=getattr(cutlass, "__version__", "?"))
        return _STATE
    except Exception as exc:                                 # pragma: no cover
        _log_once("sm_100 MXFP8 decode rows unavailable (%s: %s); routing is "
                  "unchanged" % (type(exc).__name__, str(exc)[:160]))
        return None


def enabled() -> bool:
    """True when the decode row can take cells. Cheap after the first call."""
    return _init() is not None


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def pick_config(m: int, n: int, k: int) -> Optional[_ConfigT]:
    """Tactic for one decode cell, or None to leave the cell to the tiers.

    The rule is expressed in ``m``, ``tiles_n`` and ``k`` and then checked
    against the kernels' own validity predicates; there is no per-shape table.

    * ``m > 64`` → None. Above the band the C++ cascade's two-kernel split-K
      keeps the narrow-N long-K cells (its route extends to M ≤ 128), its
      wave-tile rule keeps the cells it was measured to win from M = 64 up, and
      tier 1 keeps the rest. Nothing measured above M = 64 asked for a change.
    * ``32 < m <= 64`` runs on the persistent kernel with a 64-wide token tile
      and the 2-CTA 256-row weight tile in a cluster of two, swap-AB, one K
      slice (``_wide_token_config``); the split-K class refuses the band and the
      narrower token tiles refuse it too. Two shape classes are declined and
      left to the tiers below, which were measured ahead or level on them:
      the long-K narrow-N class (``k >= 8192`` with ``tiles_n <= 32``), whose
      two-kernel split-K in the C++ cascade keeps the whole machine busy while
      one weight tile per cluster cannot, and the short-K class
      (``k < MIN_WIDE_TOKEN_K``), where the form's mainloop is shorter than
      its pipeline.
    * How deep to split K is one question asked of every shape, and it has
      three parts, all of them about how much of the machine the shape already
      uses and how long each slice's mainloop would be:
        - the depth WANTED is four slices on a long-K shape (k >= 8192) or at
          M <= 8, and two otherwise; on a wide-N shape (``tiles_n > 32``) it is
          two at M <= 8 and none above, because there the N tiles alone already
          put ``tiles_n`` CTAs on the machine and only the narrow 8-wide token
          tile leaves a CTA little enough work for halving its mainloop to beat
          paying for the reduction;
        - a depth is refused if it would leave a slice shallower than
          ``MIN_SLICE_K``, or if ``tiles_n * split_k`` would exceed the SM
          count, which is the grid spilling past one wave;
        - and it is refused if the kernel's own ``is_valid_tactic`` says no,
          i.e. ``k % (tile_k * split_k) != 0`` with ``tile_k = 128``.
      A split that survives all three runs on the split-K kernel.
    * Everything that does not split — wide N above M = 8, a shape as shallow
      as K = 512, a K that divides neither tiling — runs on the persistent
      kernel with ONE K slice, after its own ``can_implement`` is asked with
      the operands in the swapped orientation the launch uses.

    The MMA tile is the vendored kernel's own ``mma_tiler_mn_for_m``: N tile 8
    for M ≤ 8, 16 for M ≤ 16, 32 for M ≤ 32, always with an M tile of 128 (the
    weight side) and cluster (1, 1).
    """
    if m <= 0 or m > MAX_DECODE_M:
        return None
    key = (m, n, k)
    hit = _CFG_CACHE.get(key, _MISSING)
    if hit is not _MISSING:
        return hit
    cfg = _compute_config(m, n, k)
    _CFG_CACHE[key] = cfg
    if cfg is not None and os.getenv("FSO_PRINT_TILE_INFO"):
        # stderr, like the C++ tile-info prints of the same knob and like
        # `_log_once` above: stdout is the benches' per-cell JSON channel.
        tile, cluster, swap_ab, split_k = cfg
        print("[fso sm100 decode] M=%d N=%d K=%d -> %s tiler=%s cluster=%s "
              "swap_ab=%s split_k=%d" % (
                  m, n, k, "splitk" if split_k > 1 else "persistent",
                  tile, cluster, swap_ab, split_k), file=sys.stderr, flush=True)
    return cfg


def _compute_config(m: int, n: int, k: int) -> Optional[_ConfigT]:
    st = _init()
    if st is None:
        return None
    if n % 128 != 0 or k % 128 != 0:
        # The scale buffers are only byte-compatible with the kernels' 128×4
        # swizzled view when K is a multiple of 128, and the 16-byte store
        # alignment of the swapped output needs N % 8 == 0; 128 covers both.
        return None
    from . import _sm100_smm

    SK, PK, cutlass = st["SK"], st["PK"], st["cutlass"]
    tiles_n = (n + 127) // 128
    sms = _sm100_smm._sm_count()
    if not SK.supports_m(m):
        return _wide_token_config(m, n, k, tiles_n, PK, cutlass)
    tile = SK.mma_tiler_mn_for_m(m)
    deepest = 4 if (k >= 8192 or m <= 8) else 2
    if tiles_n > 32:
        # Wide N: the N tiles alone already put `tiles_n` CTAs on the machine,
        # so a split has to earn its keep. It does exactly while the token tile
        # is the narrow 8-wide one, where a CTA's share of the output is small
        # enough that halving its mainloop beats paying for the reduction.
        deepest = 2 if m <= 8 else 1
    for split_k in (4, 2):
        if split_k > deepest or k // split_k < MIN_SLICE_K:
            continue
        if tiles_n * split_k > sms:
            # The split would push the grid past one wave, which costs a whole
            # extra pass over the machine for the tail.
            continue
        if SK.is_valid_tactic(m, k, cutlass.Float8E4M3FN, split_k):
            return (tile, (1, 1), True, split_k)
    # Nothing splits: fall through to one slice, below.
    # One K slice, on the persistent kernel. This is the wide-N answer and also
    # what a narrow-N shape too shallow to split ends up with. Under swap_ab the
    # kernel's M is the weight's N and the kernel's N is the token count, so that
    # is the order can_implement is asked in; the output is (N, M) with a unit
    # stride on its first mode, i.e. "m"-major.
    ok = PK.can_implement(
        cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32, cutlass.BFloat16,
        tile, (1, 1), n, m, k, 1, "k", "k", "m")
    if not ok:
        return None
    return (tile, (1, 1), True, 1)


def _wide_token_config(m: int, n: int, k: int, tiles_n: int,
                       PK, cutlass) -> Optional[_ConfigT]:
    """The 32 < M ≤ 64 form: persistent kernel, 64-wide token tile, 2-CTA
    256-row weight tile in a cluster of two, swap-AB, one K slice.

    Why this form and not another. The split-K class stops at M = 32
    (``supports_m``), and the persistent kernel's ``can_implement`` refuses any
    token tile narrower than the token count, so 64 is the narrowest tile that
    holds the band and the only question is how the weight rows are cut. The
    sweep in ``b300_round3_20260922/M-D5`` timed every cut the kernel accepts
    — 128- and 256-row weight tiles, clusters of one, two and four along the
    weight rows, both orientations, prefetch on and off — on four shapes at
    M = 40, 48, 56 and 64, and the 256-row tile in a cluster of two was the
    fastest or tied on every cell of every shape it takes: with one tcgen05
    MMA spanning the two CTAs of a cluster, the token operand is multicast to
    both, so each CTA streams its 128 weight rows once while the token stream
    is halved against the 128-row tile at cluster (1, 1), and unlike a cluster
    of four it never leaves a cluster slot idle on a 20-tile grid. Prefetch
    lost on every cell measured. FlashInfer's own autotuner picks this tactic
    for the narrow-N shape at M = 64 and M = 128.

    Two shape classes are declined, and the tier that keeps them was measured
    ahead or level in the same sweep:

    * long-K narrow-N (``k >= 8192`` with ``tiles_n <= 32``), which the C++
      cascade's two-kernel split-K route owns to M = 128. With twenty weight
      tiles and one K slice this kernel keeps twenty CTAs busy on 148 SMs for
      the whole of a 9728-deep mainloop, while the cascade's split spreads the
      same K over the machine and pays only a small reduce launch.
    * short K (``k < MIN_WIDE_TOKEN_K``), where the one-slice mainloop is
      shorter than the pipeline that fills it and the form is level with the
      cuBLAS tier at best (see the constant).

    The very wide shapes are NOT declined: on `gate_up` (152 weight tiles,
    more than the 74 clusters of two the persistent scheduler keeps resident)
    the form still beat cuBLAS below M = 64 and the cascade's 192-wide wave
    tile at M = 64, second round of clusters and all.
    """
    if k >= 8192 and tiles_n <= 32:
        return None
    if k < MIN_WIDE_TOKEN_K:
        return None
    tile, cluster = (256, _WIDE_TOKEN_TILE), (2, 1)
    ok = PK.can_implement(
        cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32, cutlass.BFloat16,
        tile, cluster, n, m, k, 1, "k", "k", "m")
    if not ok:
        return None
    return (tile, cluster, True, 1)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
def _get_compiled(st: dict, cfg: _ConfigT, sf_m: int, sf_n: int, sf_k: int):
    """One ``cute.compile`` per tactic; the problem shape stays symbolic.

    The compiled entry takes the three operands as CuTe tensors with dynamic
    extents and the three scale-factor extents as ``Int64`` runtime arguments,
    so a single artifact serves every (N, K, M) that picks the same tactic and
    the extents passed here — those of whichever cell compiled first — do not
    enter the cache key. Compiling costs 0.2–0.4 s and happens on the first
    eager call, never inside a capture.
    """
    tile, cluster, swap_ab, split_k = cfg
    key = (tile, cluster, swap_ab, split_k, _ENABLE_PDL)
    fn = st["compiled"].get(key)
    if fn is not None:
        return fn
    cutlass, cute, utils = st["cutlass"], st["cute"], st["utils"]
    make_ptr, cuda_drv = st["make_ptr"], st["cuda"]

    if split_k > 1:
        gemm = st["SK"](32, tile, split_k, _ENABLE_PDL)
    else:
        gemm = st["PK"](32, tile, cluster, False, _ENABLE_PDL)
    mac = utils.HardwareInfo().get_max_active_clusters(
        cluster[0] * cluster[1] * split_k)

    ms = _fake_memspace(st)
    sym_m, sym_n, sym_k = cute.sym_int(), cute.sym_int(), cute.sym_int()
    a_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (sym_m, sym_k), stride_order=(1, 0),
        assumed_align=16, memspace=ms)
    b_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (sym_n, sym_k), stride_order=(1, 0),
        assumed_align=16, memspace=ms)
    c_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_n, sym_m) if swap_ab else (sym_m, sym_n),
        stride_order=(0, 1) if swap_ab else (1, 0),
        assumed_align=16, memspace=ms)
    a_sf_fake = make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16)
    b_sf_fake = make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16)
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4, memspace=ms)
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    fn = cute.compile(gemm.wrapper, a_fake, b_fake, c_fake, sf_m, sf_n, sf_k,
                      1, a_sf_fake, b_sf_fake, alpha_fake, mac, stream,
                      swap_ab, options="--opt-level 2")
    st["compiled"][key] = fn
    return fn


def _fake_memspace(st: dict):
    """The address space ``from_dlpack`` gives a CUDA tensor.

    The fake descriptors compiled against must agree with the objects passed at
    launch, and the agreement is probed on a real tensor rather than assumed.
    On this stack the answer is ``gmem``, which is also the default.
    """
    ms = st.get("memspace")
    if ms is None:
        probe = torch.zeros(4, 8, dtype=torch.bfloat16, device="cuda")
        ct = st["from_dlpack"](probe, assumed_align=16)
        ct.element_type = st["cutlass"].BFloat16
        ms = ct.memspace
        st["memspace"] = ms
    return ms


def _make_cute_tensor(st: dict, t: torch.Tensor, dtype, leading_dim: int):
    """Wrap a torch tensor as a shape-dynamic CuTe tensor. About 4.5 µs."""
    ct = st["from_dlpack"](t, assumed_align=16)
    ct.element_type = dtype
    ct = ct.mark_layout_dynamic(leading_dim=leading_dim)
    return ct


def _operand_tensor(st: dict, t: torch.Tensor, dtype, leading_dim: int):
    """Same, hoisted across calls for the two OPERANDS, one slot per layout.

    ``from_dlpack`` keeps a reference to the torch tensor it wraps — it has to,
    because the CuTe tensor is a bare pointer and something must keep the
    storage alive — so a cache of these objects is also a cache of device
    buffers, and the policy has to be written with that in mind:

    * The cache holds **one slot per (dtype, shape, stride, leading dim)**, and
      a call whose address does not match the slot rebuilds and *replaces* it,
      which drops the previous object and releases whatever it was keeping
      alive. At most one buffer per layout is retained, no matter how the
      caller's addresses move.
    * The **output is deliberately not cached at all**. It is allocated inside
      the call, so a cached entry would keep the previous output alive, the
      allocator would be forced to hand out a different block on the next call,
      and the entry would miss every time — a cache that pins memory and never
      hits. Rebuilding its descriptor costs 4.5 µs of host time and nothing
      under CUDA-graph capture, where the whole closure runs once.
    * The weight and the activation of a decode call are caller-owned and keep
      their addresses in any loop worth capturing, so in the steady state this
      is two dict hits and no allocation at all.

    The slot count is bounded as a backstop against a caller that walks through
    unboundedly many distinct layouts; on overflow the cache is emptied rather
    than grown.
    """
    key = (dtype, tuple(t.shape), tuple(t.stride()), leading_dim)
    ptr = t.data_ptr()
    hit = _ARG_CACHE.get(key)
    if hit is not None and hit[0] == ptr:
        return hit[1]
    if hit is None and len(_ARG_CACHE) >= _ARG_CACHE_LIMIT:
        _ARG_CACHE.clear()
    ct = _make_cute_tensor(st, t, dtype, leading_dim)
    _ARG_CACHE[key] = (ptr, ct)
    return ct


def _alpha(st: dict, device: torch.device):
    """The ``(1,)`` fp32 epilogue scale, as a CuTe tensor with a STATIC layout.

    ``mark_layout_dynamic`` must NOT be applied here: the compiled entry takes
    alpha with a static extent, and making it dynamic no longer matches.
    """
    hit = _ALPHA.get(device)
    if hit is not None:
        return hit
    t = torch.ones(1, dtype=torch.float32, device=device)
    ct = st["from_dlpack"](t, assumed_align=4)
    ct.element_type = st["cutlass"].Float32
    _ALPHA[device] = (t, ct)
    return _ALPHA[device]


def linear_mxfp8_decode(x_fp8: torch.Tensor, w_fp8: torch.Tensor,
                        sx: torch.Tensor, sw: torch.Tensor,
                        config: _ConfigT) -> Optional[torch.Tensor]:
    """Run one decode-band MXFP8 GEMM. None → the caller falls through.

    The operands are handed over swapped: the weight ``[N, K]`` is the kernel's
    A and the activation ``[M, K]`` its B, with the two scale buffers following
    their operands. The BF16 output is allocated ``[M, N]`` row-major as usual
    and passed as the column-major view of itself, which is the layout the
    kernel writes in the swapped orientation; the buffer the caller receives is
    an ordinary row-major ``[M, N]``.
    """
    st = _init()
    if st is None:
        return None
    if x_fp8.dim() != 2 or w_fp8.dim() != 2:
        return None
    if x_fp8.stride(1) != 1 or w_fp8.stride(1) != 1:
        return None
    m, k = x_fp8.shape
    n = w_fp8.shape[0]
    tile, cluster, swap_ab, split_k = config

    out = torch.empty(m, n, dtype=torch.bfloat16, device=x_fp8.device)
    kernel_a, kernel_b = (w_fp8, x_fp8) if swap_ab else (x_fp8, w_fp8)
    a_sf, b_sf = (sw, sx) if swap_ab else (sx, sw)
    kernel_m, kernel_n = (n, m) if swap_ab else (m, n)
    launch_out = out.as_strided(out.shape, (1, out.shape[0])) if swap_ab else out
    c_lead = 0 if swap_ab else 1

    sf_m = (kernel_m + 127) // 128
    sf_n = (kernel_n + 127) // 128
    sf_k = (k // 32 + 3) // 4

    fn = _get_compiled(st, config, sf_m, sf_n, sf_k)
    cutlass, make_ptr = st["cutlass"], st["make_ptr"]
    gmem = st["cute"].AddressSpace.gmem
    ct_a = _operand_tensor(st, kernel_a, cutlass.Float8E4M3FN, 1)
    ct_b = _operand_tensor(st, kernel_b, cutlass.Float8E4M3FN, 1)
    ct_c = _make_cute_tensor(st, launch_out, cutlass.BFloat16, c_lead)
    p_sfa = make_ptr(cutlass.Float8E8M0FNU, a_sf.data_ptr(), gmem, assumed_align=16)
    p_sfb = make_ptr(cutlass.Float8E8M0FNU, b_sf.data_ptr(), gmem, assumed_align=16)
    _, ct_alpha = _alpha(st, x_fp8.device)
    # Queried per call on purpose: under torch.cuda.graph capture the current
    # stream is the capture side stream, and a hoisted stream object would
    # record the launch on the wrong one.
    stream = st["cuda"].CUstream(torch.cuda.current_stream().cuda_stream)
    fn(ct_a, ct_b, ct_c, sf_m, sf_n, sf_k, p_sfa, p_sfb, ct_alpha, stream)
    return out
