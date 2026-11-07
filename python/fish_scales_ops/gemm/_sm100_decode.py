"""Dense MXFP8 swap-AB kernels for sm_100/sm_103: the decode band (M ≤ 64) and
the narrow-N long-K and short-K bands above it.

This module is the swap-AB row of the sm_100/sm_103 MXFP8 router (named the
*decode* row after its first band). It sits in front of the three tiers
described in :mod:`._sm100_dispatch` and takes the bands that none of the
three shipped tiers fits:

* **The decode band, M ≤ 64.** cuBLAS ``scaled_mm`` (tier 1) picks a 128- or
  256-wide N tile and cannot be asked for a narrower one, so on a decode shape
  it leaves most of the machine idle and still pays a full-width epilogue; the
  persistent CuTe-DSL kernel of :mod:`._sm100_dsl` (tier 2) declines the whole
  band — its ``pick_config`` returns None below M = 256; and the C++ cascade
  (tier 3) answers a narrow-N decode cell with a *two-kernel* split-K: one
  GemmUniversal launch writes FP32 partials to a workspace and a second launch
  reduces them. The second launch cannot start before the first drains, and at
  M = 1 it occupies 3 of 148 SMs, so the op serialises two grid launches and
  pays an extra L2 round trip for the partials.
* **The narrow-N long-K band above it (issue #12, P5),** N ≤ 4096 (32 weight
  tiles) with K ≥ 4096 at M > 64, up to where tier 1's own grid fills the
  machine (``_narrow_long_k_config``). There every other kernel runs 10-64
  CTAs, each over the whole K; the split-K kernel with a 64-wide token tile
  puts two to four times the CTAs on the same output tiles, and the persistent
  2-CTA form takes the cells whose split would not fit one wave. A call
  carrying a field tier 1 does not apply (residual, erf-GELU) on a narrow-N
  long-K cell above that band is taken by the persistent form too, ahead of the
  C++ cascade (``pick_fallthrough_config``). The same split-K form also serves
  the narrow-N long-K class inside the decode band, from M = 33 (FB2).
* **The narrow-N short-K band above the decode band (FB2),** N ≤ 4096 with
  1024 ≤ K ≤ 2560 at M > 64 while the split-K form with 64-wide token tiles in
  two slices has at most 80 CTAs (``_narrow_short_k_config``): the tower's
  1280 x 1280 projections to M = 256, 2048-2560-wide shapes to M = 128,
  Family C shared.gate_up to M = 320. Tier 1 and the cascade put 20-40 CTAs on
  these problems.
* **Two kinds of call on short-K cells the row does not own by shape (FB2),**
  both on the 1-SM persistent form at the 128 x 64 tile while its grid fits one
  wave: the identity call of a wide-N cell above M = 64
  (``pick_identity_config``), and the erf-GELU call of a cell where the
  cascade's short-K tile rule has to leave its 64-wide tile
  (``pick_activation_config``). The router asks for them by the call's fields;
  the bias and residual calls of those cells keep their tiers.

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
    narrow-N shapes (``wo``, ``down``) want. Above M = 64 the class runs a
    64-wide token tile, several of them per problem on grid Y; a narrower token
    tile serves only a problem it covers in one tile (``SK.supports_tile``: the
    scale-factor slice of a second narrow tile does not exist, and such a launch
    computes wrong output).
``Sm100BlockScaledPersistentDenseGemmKernel``
    The plain persistent block-scaled kernel at the same narrow token tile. On
    a wide-N shape the tile grid already fills the machine, so splitting K only
    multiplies the work (measured on ``gate_up`` in the tactic sweep cited at
    the end of this docstring); the single-slice persistent kernel is used
    instead.

    Above M = 32 the split-K class refuses the cell (its ``supports_m``), and
    this kernel's own ``can_implement`` refuses a token tile narrower than the
    token count, so the band 32 < M ≤ 64 runs on this kernel with a 64-wide
    token tile and the 2-CTA 256-row weight tile in a cluster of two: one
    tcgen05 MMA spans both CTAs of the cluster and the token operand is
    multicast to them, so each weight row is streamed once and the token
    stream is halved against the 128-row tile at cluster (1, 1).

Both are driven in SWAP-AB orientation: the kernel's A operand is the weight
``[N, K]`` and its B operand is the activation ``[M, K]``, so the narrow token
dimension becomes the kernel's N and the 8/16/32/64/128-wide MMA tile is the
token tile. fso's ``quantize_1x32_fp8`` already emits its opaque int32 UE8M0 buffer
in the CUTLASS ``Sm1xxBlockScaledConfig<32>`` atom layout, which is
byte-identical to the 128×4 swizzled layout these kernels read, so the scale
buffers are handed over as raw pointers — no repack, no copy, no extra memory.

Contracts and gates:

* **DSL floor.** The vendored sources use CuTe-DSL surfaces that do not exist
  in 4.4.2 (``cute.nvgpu.OperandMajorMode``, ``Numeric.bitcast``, the newer
  ``make_blockscaled_trivial_tiled_mma`` arity, ``cute.struct`` field handles).
  4.5.0 is the floor and 4.8.0 is the recommended pin (the JIT compile time
  differs between releases; the version table is run ``M-D2`` below). Below the
  floor this module is INERT: ``_init`` returns None
  before importing the vendored package, ``pick_config`` returns None, and
  ``route`` behaves exactly as it did before this module existed. The row is
  silent only when it is switched off or ``nvidia-cutlass-dsl`` is not
  installed at all (``FSO_LOG=1`` then prints that reason once on stderr, since
  stdout belongs to the benches' per-cell JSON protocol); a DSL below the floor
  or a failed import of the vendored kernels warns once per process with
  one ``fso:`` line on stderr, ``FSO_LOG`` or not.
* **Capture.** ``cute.compile`` runs on the first call for a given tactic and
  is cached per (kernel class, tiler, cluster, split_k, pdl, epilogue), so the
  eager warm-up every caller already performs before ``torch.cuda.graph``
  capture covers it, as it covers the tactic cache and the one occupancy query
  per cluster size that the band above M = 64 asks (``_max_active_clusters``);
  nothing compiles, allocates or synchronises inside capture. The launch binds
  ``data_ptr()`` values, so for the life of a captured graph the activation,
  the weight, the OUTPUT, **both scale buffers** and the bias / residual of a
  fused call must keep their addresses — the scale
  buffers and the epilogue tensors are the part that is new relative to the
  rest of fso, because they are passed as bare pointers rather than as tensors.
* **Epilogue (issue #12, S1b1).** The dense epilogue of interface v1 (bias,
  erf-GELU, residual; ``_epilogue.SERVED["sm100.decode"]``) is applied inside
  both vendored kernels, in fp32 after the reduction and before the one bf16
  rounding, by the epilogue variant of the kernel (``_get_compiled``); the
  identity variant is today's kernel, compiled through the unchanged entry.
* **Env kills.** ``FSO_DISABLE_DSL=1`` (shared with tier 2) and
  ``FSO_DISABLE_DECODE_DSL=1`` (this module only) both make it inert.
  ``FSO_PRINT_TILE_INFO=1`` prints the picked tactic once per distinct
  (M, N, K). All of these switches follow the extension's rule
  (:mod:`fish_scales_ops._env`): unset, empty or a value starting with ``0``
  is off.

Provenance of the routing rule: the tactic sweep in the B300 run
``b300_mxfp8_20260917/M-D1`` (400 timed rows over four shapes × six M × every
tactic the split-K kernel accepts), the calling-convention A/B in ``M-D3``, the
DSL version table in ``M-D2``, and for the 32 < M ≤ 64 band the persistent
kernel sweep in ``b300_round3_20260922/M-D5`` (every tiler, cluster,
orientation and prefetch setting the kernel accepts, four shapes × four M,
two passes); for the narrow-N long-K band above M = 64 the design note and
gate-1 sweep of issue #12 (``fso_issue12_20261028/runs/P5`` and ``P5g1``:
every split-K and persistent tactic on o, fc2 and conv_out at M = 78-1079,
two to four passes per arm, cold weights); for the FB2 rules (the narrow-N
long-K class at 32 < M ≤ 64, the narrow-N short-K band, the identity and
erf-GELU calls above) the B300 re-measurement of gate 0's candidates
(``fso_issue12_20261028/runs/G0D`` and ``runs/FB2``: the forced swap-AB
configurations against the route of 0.2.3 on 140 cells with their M and N
neighbours, both entry forms, two passes, cold weights), cited in each rule.
"""
from __future__ import annotations

import sys
from typing import Optional, Tuple

import torch

from .._env import env_flag

# (mma_tiler_mn, cluster_shape_mn, swap_ab, split_k)
_ConfigT = Tuple[Tuple[int, int], Tuple[int, int], bool, int]

#: First nvidia-cutlass-dsl release that runs the vendored kernels (M-D2).
MIN_DSL_VERSION = (4, 5, 0)
#: Width of this router row's decode band. Up to the split-K kernel's own
#: MAX_M (32) the narrow-N shapes split K in-cluster; from there to 64 the
#: persistent kernel runs a 64-wide token tile (see ``_compute_config``). Above
#: it the row takes only the narrow-N long-K band (``_narrow_long_k_config``).
MAX_DECODE_M = 64
#: Token tile of the persistent kernel above the split-K class's MAX_M, and
#: the widest token count the row takes at one tile.
_WIDE_TOKEN_TILE = 64
#: Shallowest K the wide-token form takes, in elements: eight mainloop K-tiles
#: of 128. On a K = 512 shape (Family C `shared_down`) the one-slice mainloop
#: is four tiles, shorter than the pipeline that fills it, so the form has
#: nothing to gain there, while at K = 2048 the sweep put the cells on it. The
#: floor sits between the two measured points (b300_round3_20260922/M-D5,
#: tables/SWEEP_BC.txt).
MIN_WIDE_TOKEN_K = 1024
#: Programmatic dependent launch. The kernels own both sides of the
#: griddepcontrol pair (the wait before any load of the activation, its scale,
#: the bias or the residual, and launch-dependents at the kernel's end), which
#: is what makes it capture-safe here; this is unrelated to the closed "PDL on
#: a CUTLASS GEMM with no GDC compiled in" item. Under PDL the weight and its
#: scale are read before the wait (``_WEIGHTS_FIRST``).
#: ``FSO_DISABLE_PDL=1`` (read once per process, like the C++ launches) drops
#: the attribute from the launch; the kernels' griddepcontrol instructions are
#: then no-ops and the bytes are unchanged (test_dense_v1, case q).
_ENABLE_PDL = True
_PDL: Optional[bool] = None
#: Weights first (issue #12, FB1). Under PDL both kernels are built with ``fso_weights_first``: the TMA warp issues
#: the weight and weight-scale copies of its first tile's first stages (``_wf_pre_wait_stages``, never every stage)
#: before its griddepcontrol.wait and the activation and activation-scale copies after it, so the weight stream of a
#: call overlaps the kernel in front of it (the activation quantize of a bf16-x call, or the previous GEMM's tail in
#: a pre-quantized call). Every other warp waits where it did, so the bias, the residual and the output are touched
#: only after a wait. Without PDL the kernels are today's, byte for byte. The order of the loads changes nothing in
#: the arithmetic, so the output bytes are the same either way (test_dense_v1, cases q and x).
_WEIGHTS_FIRST = True
#: The most stages the weights-first TMA warp fills before its wait, a constant of each kernel variant: the one-CTA
#: tiles (the split-K kernel at every token tile and the persistent kernel at cluster (1, 1)) fill up to two, the
#: two-CTA tile (the persistent kernel's 256-row weight tile in a cluster of two) fills one. Measured on the B300
#: with cold weights (run fso_issue12_20261028/FB1, holds C and E1; run FB1b, holds S and S2; two passes each). With
#: every stage but one filled before the wait, the activation copies issued after it queued behind up to ten weight
#: copies in the TMA unit and the pre-quantized LLM decode call got 0.1-0.2 us slower than without weights first.
#: Four stages gave the largest gain on the pre-quantized call (fc2 1280 x 5120 at M = 78: 8.5 %), but on the
#: two-CTA tile the pre-quantized call of llm_o (2048 x 4096) was 0.06-0.07 us slower (+1.1 % at M = 64 and 416), and
#: in the issue's serving chain (fso.dense.linear(x, w) followed by torch's bias and residual adds, so that the
#: activation quantize launches behind a kernel that never releases early and the window in front of the GEMM is the
#: quantize alone) the bf16-x call of fc2 was 1.2-1.5 % slower than 0.2.3 at M = 78-364 with four stages and within
#: 0.3 % with one or two. Two stages keep 2.4-5.4 % of the pre-quantized call's gain on the decode-row cells
#: (fc2 78/182 -3.0/-2.9 %, conv_out 117 -3.2 %, q 78 -5.4 %; one stage: -1.5/-1.8, -2.4, -3.9 %).
WF_PRE_WAIT_STAGES_ONE_CTA = 2
WF_PRE_WAIT_STAGES_TWO_CTA = 1
#: The fewest mainloop K blocks (128 wide) a CTA's K slice must hold for the weights-first prologue to be built:
#: ``k / 128 / split_k`` at or above this count builds the weights-first variant, below it today's kernel. A slice
#: of four blocks is barely longer than the pipeline that fills it; there the two early weight stages gained nothing
#: and cost 1.0-1.8 % on the pre-quantized call and 0.4-1.2 % on the bf16-x call (the B300, cold weights: shared.down
#: 2048 x 512, one slice, M 1-16; shared.gate_up 1024 x 2048, four slices, M 1-8; PERF2's re-measure of the trunk and
#: run fso_issue12_20261028/FB1c, hold K). From five blocks the prologue pays: the tower's 1280 x 1280 projections
#: (two slices of five) gain 5.4-5.8 % on the pre-quantized call and 1.6-3.2 % on the bf16-x call, fc2 (four slices
#: of ten) 3.1 %, conv_out (four of fifteen) 3.2 %. Slices of six to eight blocks on the 8-token tile (2048 x 1536
#: and 5120 x 2048 at M <= 8, 2048 x 4096 at M <= 8) are within -0.4..+1.4 % either way.
WF_MIN_K_BLOCKS = 5


def _pdl_enabled() -> bool:
    global _PDL
    if _PDL is None:
        _PDL = _ENABLE_PDL and not env_flag("FSO_DISABLE_PDL")
    return _PDL


def _weights_first(k_blocks: Optional[int] = None) -> bool:
    """True when the kernels are built in the weights-first form: under PDL, which is the only setting in which a
    load can be issued before the kernel in front has finished (see ``_WEIGHTS_FIRST``), and, when the CTA's K slice
    is given as its count of 128-wide K blocks, only for a slice of at least ``WF_MIN_K_BLOCKS`` of them."""
    if not (_WEIGHTS_FIRST and _pdl_enabled()):
        return False
    return k_blocks is None or k_blocks >= WF_MIN_K_BLOCKS


def _wf_pre_wait_stages(cluster) -> int:
    """The weights-first stages of a kernel variant before its wait: ``WF_PRE_WAIT_STAGES_TWO_CTA`` for the
    persistent kernel's two-CTA tile (a cluster of two), ``WF_PRE_WAIT_STAGES_ONE_CTA`` for every one-CTA tile (the
    split-K kernel's tactics carry cluster (1, 1); its K slices are a separate cluster dimension)."""
    return WF_PRE_WAIT_STAGES_TWO_CTA if cluster[0] * cluster[1] > 1 else WF_PRE_WAIT_STAGES_ONE_CTA


#: Shortest K slice worth giving a CTA, in elements: four mainloop K-tiles of
#: 128. Splitting K costs a wider cluster launch and an in-cluster reduction,
#: and below four tiles the mainloop is shorter than the pipeline that fills it,
#: so the split does not pay for itself. Measured on the Family C shared-expert
#: `down` projection (N = 2048, K = 512), where one slice beat two and four at
#: M = 32 (b300_dense_decode_20260922, tables/SHORTK.txt).
MIN_SLICE_K = 512
#: Backstop on the number of operand layouts kept hoisted (see
#: ``_operand_tensor``); one device buffer is retained per occupied slot.
_ARG_CACHE_LIMIT = 64

# The narrow-N long-K band above the decode band (issue #12, P5; ``_narrow_long_k_config``).
#: Shallowest K of the band: the long-K narrow-N class, where every other tier runs 10-64 CTAs each over the
#: whole K (gate 1 measured the band on K = 4096, 5120 and 7680).
NARROW_LONG_K_MIN_K = 4096
#: Widest N of the band, in 128-row weight tiles (the narrow-N class of the other tiers' rules).
NARROW_LONG_K_MAX_TILES_N = 32
#: The band's top: a cell whose cuBLAS grid of 128 x 128 tiles, ``tiles_n * ceil(M / 128)`` CTAs, exceeds this
#: is tier 1's. Gate 1 measured tier 1 winning against every swap-AB form once its grid reached 90-96 CTAs (o
#: [2048 x 4096] at M = 768, fc2 / conv_out [1280 x 5120 / 7680] at M = 1079) and losing at 80 and below (o at
#: M = 256-640, fc2 / conv_out at M = 182-533). On the shapes measured the one-wave bound of the forms binds
#: first; this bound keeps a narrower N, whose forms would fit one wave far higher up, off tier 1's cells.
NARROW_LONG_K_MAX_CUBLAS_CTAS = 80
#: Token tile of the band's split-K form: the narrowest that may repeat along M (``SK.supports_tile``).
NARROW_LONG_K_TOKEN_TILE = 64
#: Shallowest K slice of a four-way split in the band: ten mainloop K-tiles. Gate 1 measured the four-way split
#: against the two-way one at the same grid: slower on o (K = 4096, slices of 1024: 5.66 against 5.39 us at
#: M = 80), faster on fc2 (K = 5120, 1280: 5.54 against 5.71) and conv_out (K = 7680, 1920: 6.49 against 7.37).
NARROW_LONG_K_MIN_SPLIT4_SLICE_K = 1280
# The narrow-N short-K band above the decode band (FB2; ``_narrow_short_k_config``).
#: Most CTAs the band's split-K form may launch: ``2 * tiles_n * ceil(M / 64)``. Every call measured on a grid of at
#: most 80 CTAs was faster on the form than on its previous tier, on both entry forms; from 96 CTAs some fused calls
#: were slower (see the function).
NARROW_SHORT_K_MAX_CTAS = 80

# The 1-SM persistent form above the decode band on short-K cells (FB2; ``pick_identity_config``,
# ``pick_activation_config``): the persistent kernel at the 128 x 64 MMA tile (128 weight rows x 64 tokens),
# cluster (1, 1), one K slice, swap-AB, one tile per CTA while its grid fits one wave.
#: The form, as a tactic.
SHORT_K_TOKEN_TILE_FORM = ((128, 64), (1, 1), True, 1)
#: Deepest K of the short-K cells the form takes above the decode band: the bound of the C++ cascade's short-K tile
#: rule (``pick_short_k_tile`` in dispatch.cuh, ``_sm100_smm.SHORT_K_MAX_K``). Measured at K = 1280, 2048 and 2560.
SHORT_K_TOKEN_TILE_MAX_K = 2560
#: Widest M of the activation calls the form takes: the top of the cascade short-K rule's M range (kShortKMaxM).
SHORT_K_TOKEN_TILE_MAX_M = 2048

_STATE: Optional[dict] = None
_CFG_CACHE: dict = {}
_FT_CACHE: dict = {}
_ID_CACHE: dict = {}
_ACT_CACHE: dict = {}
_CLUSTERS: dict = {}
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
    if not env_flag("FSO_LOG"):
        return
    if msg in _LOGGED:
        return
    _LOGGED.add(msg)
    print("fso: " + msg, file=sys.stderr, flush=True)


def _warn_unavailable(reason: str) -> None:
    """Warn, once per process and whatever ``FSO_LOG`` says, that the row
    should be there and is not: one line on stderr, in the ``_log_once`` format
    (stdout belongs to the benches' JSON protocol). The row's kernels are
    vendored in the package, so unlike the mid-band tier there is no kernel
    path to point elsewhere."""
    msg = ("sm_100/103 CuTe-DSL decode row (M <= %d) unavailable: %s; linear_mxfp8 / "
           "linear_fp8 leave that band to the tiers behind it. Install nvidia-cutlass-dsl "
           ">= 4.8.0 (the recommended pin), or set FSO_DISABLE_DECODE_DSL=1 to skip the "
           "row without this warning" % (MAX_DECODE_M, " ".join(str(reason).split())))
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
    if env_flag("FSO_DISABLE_DSL") or env_flag("FSO_DISABLE_DECODE_DSL"):
        return None
    try:
        import cutlass
    except ModuleNotFoundError as exc:
        if exc.name == "cutlass":
            # The optional `sm100` extra is not installed: documented, silent.
            _log_once("sm_100 MXFP8 decode rows unavailable: nvidia-cutlass-dsl is not "
                      "installed; leaving the M <= %d band to the existing tiers" % MAX_DECODE_M)
            return None
        _warn_unavailable("importing cutlass failed: %s: %s" % (type(exc).__name__, str(exc)[:160]))
        return None
    except Exception as exc:
        _warn_unavailable("importing cutlass failed: %s: %s" % (type(exc).__name__, str(exc)[:160]))
        return None
    try:
        ver = _parse_version(getattr(cutlass, "__version__", ""))
        if ver < MIN_DSL_VERSION:
            _warn_unavailable(
                "it needs nvidia-cutlass-dsl >= %d.%d.%d, found %s"
                % (MIN_DSL_VERSION + (getattr(cutlass, "__version__", "?"),)))
            return None

        import cuda.bindings.driver as cuda_drv
        import cutlass.cute as cute
        import cutlass.utils as dsl_utils
        from cutlass.cute.runtime import from_dlpack, make_ptr

        from ._vendor.flashinfer_sm100_blockscaled.dense_blockscaled_gemm_sm100 \
            import Sm100BlockScaledPersistentDenseGemmKernel as PK
        from ._vendor.flashinfer_sm100_blockscaled.dense_blockscaled_gemm_sm100_splitk \
            import Sm100BlockScaledSplitKGemmKernel as SK
        from ._vendor.flashinfer_sm100_blockscaled.dense_blockscaled_gemm_sm100_common \
            import fso_epilogue_supported

        _STATE.update(ok=True, cuda=cuda_drv, cutlass=cutlass, cute=cute,
                      utils=dsl_utils, from_dlpack=from_dlpack,
                      make_ptr=make_ptr, PK=PK, SK=SK, compiled={},
                      dsl_version=getattr(cutlass, "__version__", "?"),
                      epilogue=fso_epilogue_supported())
        if not _STATE["epilogue"]:
            _log_once("sm_100 MXFP8 decode row: this nvidia-cutlass-dsl (%s) lacks the explicit-rounding math "
                      "surface the fused epilogue needs; calls with bias / residual / activation leave the "
                      "M <= %d band to the tiers behind it" % (_STATE["dsl_version"], MAX_DECODE_M))
        return _STATE
    except Exception as exc:                                 # pragma: no cover
        _warn_unavailable("importing the vendored kernels failed: %s: %s"
                          % (type(exc).__name__, str(exc)[:160]))
        return None


def enabled() -> bool:
    """True when the decode row can take cells. Cheap after the first call."""
    return _init() is not None


def epilogue_available() -> bool:
    """True when the row can build its epilogue variants (``_epilogue.SERVED["sm100.decode"]``) on the installed
    DSL; False makes the router hand fused calls of the band to the next tier. Cheap after the first call."""
    st = _init()
    return st is not None and bool(st.get("epilogue"))


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def pick_config(m: int, n: int, k: int) -> Optional[_ConfigT]:
    """Tactic for one decode cell, or None to leave the cell to the tiers.

    The rule is expressed in ``m``, ``tiles_n`` and ``k`` and then checked
    against the kernels' own validity predicates; there is no per-shape table.

    * ``m > 64`` → the narrow-N long-K band (``_narrow_long_k_config``): only
      ``tiles_n <= 32`` with ``k >= 4096``, up to where tier 1's own grid fills
      the machine, outside the C++ cascade's two measured rules (its two-kernel
      split-K at ``k >= 8192, m <= 104`` and its wave tile); and the narrow-N
      short-K band (``_narrow_short_k_config``, FB2): ``tiles_n <= 32`` with
      ``1024 <= k <= 2560``, the split-K kernel with 64-wide token tiles in two
      slices while that grid has at most ``NARROW_SHORT_K_MAX_CTAS`` CTAs; None
      everywhere else, so the tiers below see every other cell exactly as
      before. (The router also asks this row for the identity calls and the
      erf-GELU calls of some short-K cells it does not own by shape:
      ``pick_identity_config``, ``pick_activation_config``.)
    * ``32 < m <= 64`` on the narrow-N long-K class (``tiles_n <= 32``,
      ``4096 <= k``, outside the cascade's two-kernel split-K at ``k >= 8192``)
      runs on the split-K kernel with the 64-wide token tile, the form of the
      band above (``_split_t64``: four slices where each keeps 1280 of K, else
      two; FB2). Every other cell of the band runs on the persistent kernel with
      a 64-wide token tile
      and the 2-CTA 256-row weight tile in a cluster of two, swap-AB, one K
      slice (``_wide_token_config``); the split-K class's narrow tiles refuse
      the band. Three shape classes are declined and
      left to the tiers below: the long-K narrow-N class (``k >= 8192`` with
      ``tiles_n <= 32``), whose two-kernel split-K in the C++ cascade keeps the
      whole machine busy while one weight tile per cluster cannot; the short-K
      class (``k < MIN_WIDE_TOKEN_K``), where the form's mainloop is shorter
      than its pipeline (both as the M-D5 sweep kept them); and a shape whose
      ``ceil(N / 256)`` clusters of two exceed the resident clusters where the
      cascade's wave rule owns the cell (``gate_up`` at M = 64).
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
    if m <= 0:
        return None
    key = (m, n, k)
    hit = _CFG_CACHE.get(key, _MISSING)
    if hit is not _MISSING:
        return hit
    if m <= MAX_DECODE_M:
        cfg = _compute_config(m, n, k)
    else:
        cfg = _narrow_long_k_config(m, n, k)
        if cfg is None:
            cfg = _narrow_short_k_config(m, n, k)
    _CFG_CACHE[key] = cfg
    _print_tactic("", m, n, k, cfg)
    return cfg


def pick_fallthrough_config(m: int, n: int, k: int) -> Optional[_ConfigT]:
    """Tactic for a call that tier 1 (cuBLAS ``scaled_mm``) owns by shape but cannot serve, or None.

    Tier 1 applies the bias alone; a residual or erf-GELU call on one of its cells falls through to the next tier
    that serves every field. On the cells of the narrow-N long-K class that ``pick_config`` leaves to tier 1
    (above the band's top, or where its forms do not fit one wave) that tier was the C++ cascade, which runs a 256-row tile per 128 x 128 x K chain there (fc2 [1280 x 5120] at M = 1079 with bias and
    residual: 18.0 us against tier 1's identity 8.4 us, issue #12 run S1b1-r3). This row's persistent form, the
    2-CTA 256-row weight tile in a cluster of two (``_persistent_wide``; at that cell the 128-wide token tile, 8.98 us
    identity, gate 1), takes such a call while its clusters fit one wave of resident clusters; beyond that the
    cascade keeps it.
    The router asks this only for a call carrying a field tier 1 does not serve, so the identity and bias-only
    calls of these cells stay on tier 1.
    """
    if m <= MAX_DECODE_M:
        return None
    key = (m, n, k)
    hit = _FT_CACHE.get(key, _MISSING)
    if hit is not _MISSING:
        return hit
    cfg = None
    st = _init()
    if st is not None and n % 128 == 0 and k % 128 == 0 and _narrow_long_k_class(m, n, k):
        cfg = _persistent_wide(m, n, k, (64, 128), st)
    _FT_CACHE[key] = cfg
    _print_tactic(" fall-through", m, n, k, cfg)
    return cfg


def pick_identity_config(m: int, n: int, k: int) -> Optional[_ConfigT]:
    """Tactic for an identity call (no epilogue field) above the decode band on a wide-N short-K cell, or None (FB2).

    The cells: ``m > 64``, wide N (``tiles_n > 32``), ``MIN_WIDE_TOKEN_K <= k <= SHORT_K_TOKEN_TILE_MAX_K``, while the
    1-SM persistent form's grid, ``ceil(M / 64) * tiles_n`` CTAs (128 weight rows by 64 tokens each), fits one wave of
    the SMs. On the B300's 148 SMs that is M <= 192 for 38-49 weight tiles (llm_qkv / Family B wqkv [5120 x 2048],
    fc1 [5120 x 1280], Family A wqkv [6144 x 2560]) and M <= 128 for 50-74 tiles (Family C wqkv_gated
    [9216 x 2048]); wider shapes (gate [9728 x 2560], gdn.in_proj [12288 x 2048], gate_up) never fit above M = 64.
    The form is ``SHORT_K_TOKEN_TILE_FORM`` (the persistent kernel at the 128 x 64 MMA tile, cluster (1, 1), one K
    slice), after the kernel's own ``can_implement`` in the swapped orientation.

    Why. Without the rule these calls run on the C++ cascade's wave rule at 64 < M <= 128 (the 1-SM 128 x 64 tile:
    128 tokens by 64 weight rows, ``ceil(N / 64)`` CTAs, the same count as this form's) and on tier 1 at 128 < M
    (cuBLAS's 128 x 128 kernel, ``tiles_n * ceil(M / 128)`` CTAs, two thirds of this form's at M = 129-192).
    Measured on the B300 with cold weights (run ``fso_issue12_20261028/runs/FB2``, steps 1 and 1b, two passes, both
    entry forms; 2b = pre-quantized x, bf16 x = ``fso.dense.linear(x)``, the activation quantize plus the GEMM):
    [5120 x 2048] at M = 65-128 5.29-5.33 -> 4.99-5.03 us (2b) and 6.59-6.84 -> 6.45-6.62 us (bf16 x), at
    M = 144-192 5.61-5.65 -> 5.09-5.11 and 7.03-7.07 -> 6.73-6.85; [5120 x 1280] at M = 78-192 4.23-4.60 -> 3.94-4.02
    and 5.68-5.89 -> 5.37-5.55; [6144 x 2560] at M = 65-192 6.01-6.32 -> 5.71-5.83 and 7.35-7.88 -> 7.32-7.74 (at
    M = 80 the bf16-x call 7.38 -> 7.44, the one cell measured slower, by 0.8 %); [9216 x 2048] at M = 65-128
    6.49-6.59 -> 6.22-6.33 and 7.95-8.02 -> 7.72-7.76. Where the grid does not fit, the form lost: [5120 x 1280] at
    M = 208 / 256 (160 CTAs) 4.51 -> 5.24 us, [12288 x 2048] at M = 96 / 128 and [9728 x 2560] at M = 96 by 16-19 %.
    The 2-CTA 256-row form with the same token tile was faster on the bf16-x call of [5120 x 2048] at M = 65-128
    (6.26-6.42 us) but slower on [9216 x 2048] and on [6144 x 2560] at M = 160 / 192, so the rule keeps one form.

    Why the identity call only. A bias call measured slower on this form, on both entry forms: [5120 x 2048] at
    M = 80 / 128 5.31 / 5.39 -> 5.72 us (cascade) and at M = 160 7.15 -> 7.51 us in the bf16-x form (tier 1), [6144 x
    2560] at M = 96 6.00 -> 6.48 us, [5120 x 1280] at M = 129-192 5.85-5.90 -> 6.30-6.32 us in the bf16-x form: the
    vendored kernels add the bias in their epilogue element by element, while the cascade and cuBLAS fuse it into a
    vectorised store. So the router asks this only for a call with no epilogue field (``_sm100_dispatch``); a bias,
    residual or activation call of these cells keeps the tier it had (erf-GELU calls: ``pick_activation_config``).
    """
    if m <= MAX_DECODE_M:
        return None
    key = (m, n, k)
    hit = _ID_CACHE.get(key, _MISSING)
    if hit is not _MISSING:
        return hit
    cfg = None
    if (n + 127) // 128 > NARROW_LONG_K_MAX_TILES_N and MIN_WIDE_TOKEN_K <= k <= SHORT_K_TOKEN_TILE_MAX_K:
        cfg = _short_k_token_tile(m, n, k)
    _ID_CACHE[key] = cfg
    _print_tactic(" identity", m, n, k, cfg)
    return cfg


def pick_activation_config(m: int, n: int, k: int) -> Optional[_ConfigT]:
    """Tactic for a call carrying an activation (erf-GELU) above the decode band on a short-K cell where the C++
    cascade's short-K tile rule has to leave its 64-wide tile, or None (FB2).

    The cells: ``64 < m <= SHORT_K_TOKEN_TILE_MAX_M`` and ``MIN_WIDE_TOKEN_K <= k <= SHORT_K_TOKEN_TILE_MAX_K``,
    where the cascade's 1-SM 128 x 64 grid, ``ceil(M / 128) * ceil(N / 64)`` CTAs, overflows one wave of the SMs
    while the 1-SM persistent form of this row, ``ceil(M / 64) * ceil(N / 128)`` CTAs, fits one: a token count in
    the first half of a 128-row tile, where the swap-AB form's 64-row token granularity saves a whole wave. On the
    B300's 148 SMs that is M = 129-192 for N = 4736-6272 (fc1 [5120 x 1280], the tower's bias + erf-GELU role),
    M = 385-448 for N = 2560 and M = 513-576 for N = 2048, and never for N = 1280 or 3840. The form is
    ``SHORT_K_TOKEN_TILE_FORM``, after the kernel's own ``can_implement``.

    Why. The short-K rule (``pick_short_k_tile`` in dispatch.cuh) gives a fused call the 1-SM 128 x 64 tile while
    that grid fits one wave, and otherwise a 128-wide tile (the 2-SM 256 x 128 form at fc1 M = 129-256, 80 CTAs),
    whose erf-GELU epilogue covers twice the outputs per CTA, fully exposed behind a ten-block mainloop. Measured on
    the B300 with cold weights (run ``fso_issue12_20261028/runs/FB2``, steps 1 and 1c, two passes, both entry forms):
    fc1 at M = 129-192 6.58-6.60 -> 5.99-6.01 us (2b) and 8.14-8.19 -> 7.59-7.62 us (bf16 x). Outside the cells the
    form lost: at M = 117, where the cascade's 128 x 64 grid fits (80 CTAs), 5.00 -> 5.98 us, and at M = 208-288,
    where this form's grid spills (160 CTAs), 6.60-7.00 -> 7.94-8.11 us. The vendored kernels' erf-GELU epilogue
    costs about 2 us over their identity epilogue at these shapes, so the rule takes only the cells where the
    cascade loses a wave. The router asks this only for a call with an activation (``_sm100_dispatch``); the
    quantized output of ``linear_quantized`` stays on the cascade, which alone writes it.
    """
    if m <= MAX_DECODE_M or m > SHORT_K_TOKEN_TILE_MAX_M:
        return None
    key = (m, n, k)
    hit = _ACT_CACHE.get(key, _MISSING)
    if hit is not _MISSING:
        return hit
    cfg = None
    if MIN_WIDE_TOKEN_K <= k <= SHORT_K_TOKEN_TILE_MAX_K:
        from . import _sm100_smm
        if ((m + 127) // 128) * ((n + 63) // 64) > _sm100_smm._sm_count():
            cfg = _short_k_token_tile(m, n, k)
    _ACT_CACHE[key] = cfg
    _print_tactic(" activation", m, n, k, cfg)
    return cfg


def _short_k_token_tile(m: int, n: int, k: int) -> Optional[_ConfigT]:
    """``SHORT_K_TOKEN_TILE_FORM`` for ``(m, n, k)`` when its grid, ``ceil(M / 64) * ceil(N / 128)`` CTAs (one tile
    each), fits one wave of the SMs and the kernel's ``can_implement`` accepts the swapped problem; else None."""
    st = _init()
    if st is None or n % 128 != 0 or k % 128 != 0:
        return None
    from . import _sm100_smm
    tile, cluster = SHORT_K_TOKEN_TILE_FORM[0], SHORT_K_TOKEN_TILE_FORM[1]
    if ((m + tile[1] - 1) // tile[1]) * ((n + tile[0] - 1) // tile[0]) > _sm100_smm._sm_count():
        return None
    PK, cutlass = st["PK"], st["cutlass"]
    if not PK.can_implement(cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32, cutlass.BFloat16,
                            tile, cluster, n, m, k, 1, "k", "k", "m"):
        return None
    return SHORT_K_TOKEN_TILE_FORM


def _print_tactic(what: str, m: int, n: int, k: int, cfg: Optional[_ConfigT]) -> None:
    if cfg is not None and env_flag("FSO_PRINT_TILE_INFO"):
        # stderr, like the C++ tile-info prints of the same knob and like
        # `_log_once` above: stdout is the benches' per-cell JSON channel.
        tile, cluster, swap_ab, split_k = cfg
        print("[fso sm100 decode%s] M=%d N=%d K=%d -> %s tiler=%s cluster=%s "
              "swap_ab=%s split_k=%d" % (
                  what, m, n, k, "splitk" if split_k > 1 else "persistent",
                  tile, cluster, swap_ab, split_k), file=sys.stderr, flush=True)


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
        if tiles_n <= NARROW_LONG_K_MAX_TILES_N and k >= NARROW_LONG_K_MIN_K and not _cascade_split_owns(m, k):
            # The narrow-N long-K class at 32 < M <= 64 (FB2): the band-above form, see _split_t64.
            cfg = _split_t64(m, n, k, st)
            if cfg is not None:
                return cfg
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
    for the narrow-N shape at M = 64 and M = 128. The sweep predates the split-K
    kernel's 64-wide token tile (issue #12, P5); on the narrow-N long-K class
    (``tiles_n <= 32``, ``4096 <= k < 8192``) that form is faster and is asked
    first (``_split_t64``, FB2), so this form takes those cells only where no
    split depth is valid for their K.

    Three shape classes are declined; the same sweep kept the first two on the
    tier that serves them:

    * long-K narrow-N (``k >= 8192`` with ``tiles_n <= 32``), which the C++
      cascade's two-kernel split-K route owns to M = 104. With twenty weight
      tiles and one K slice this kernel keeps twenty CTAs busy on 148 SMs for
      the whole of a 9728-deep mainloop, while the cascade's split spreads the
      same K over the machine and pays only a small reduce launch.
    * short K (``k < MIN_WIDE_TOKEN_K``), where the one-slice mainloop is
      shorter than the pipeline that fills it and the form has nothing to
      gain (see the constant).

    * very wide N where the cascade's wave rule owns the cell. The form puts
      ceil(N / 256) clusters of two on the machine and the persistent
      scheduler keeps at most half the SM count of them resident (74 on a
      148-SM B300), so a wider grid runs its excess clusters as a second round
      in which the rest of the machine is idle: `gate_up` (N = 19456) launches
      76. Where the wave rule of the C++ cascade owns the cell (M >= 64, see
      ``_sm100_smm.wave_tile_owns``) its tile covers the whole problem in one
      wave (128 x 192 on 102 SMs for `gate_up`), and measured inside the
      Qwen3-4B MLP block with cold weights it is 3.4 % faster at M = 64
      (27.56 against 28.54 us, run fso_0.2.1_20261010/P1 ab3_probe). Below
      M = 64 the wave rule owns nothing and this form stays: there the same
      measurement put both the cascade (+9 to +12 %) and tier 1 (+0.2 to
      +2.1 %) behind it, as the M-D5 sweep did with warm weights.
    """
    if k >= 8192 and tiles_n <= 32:
        return None
    if k < MIN_WIDE_TOKEN_K:
        return None
    from . import _sm100_smm
    if 2 * ((n + 255) // 256) > _sm100_smm._sm_count() and _sm100_smm.wave_tile_owns(m, n):
        return None
    tile, cluster = (256, _WIDE_TOKEN_TILE), (2, 1)
    ok = PK.can_implement(
        cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32, cutlass.BFloat16,
        tile, cluster, n, m, k, 1, "k", "k", "m")
    if not ok:
        return None
    return (tile, cluster, True, 1)


def _max_active_clusters(st: dict, size: int) -> int:
    """Clusters of ``size`` CTAs the device keeps resident at one CTA per SM (the vendored kernels' occupancy:
    their shared-memory stages fill an SM), from the DSL's occupancy query; cached per process. 148 / 74 / 33 for
    sizes 1 / 2 / 4 on a B300. Asked on the first eager call of a cell, like the compile."""
    hit = _CLUSTERS.get(size)
    if hit is None:
        # The query goes through the driver API on the current context; a host-only caller
        # (``_sm100_dispatch.plan`` before any device work) may not have one yet, and cudaSetDevice makes the
        # device's primary context current without allocating or synchronising anything.
        torch.cuda.set_device(torch.cuda.current_device())
        hit = int(st["utils"].HardwareInfo().get_max_active_clusters(size))
        _CLUSTERS[size] = hit
    return hit


def _cascade_split_owns(m: int, k: int) -> bool:
    """True where the C++ cascade's two-kernel split-K owns a narrow-N cell: ``k >= 8192`` at ``m <= 104`` (the long-K
    clause of its parallel split-K in ``dispatch.cuh``; its ``M <= 32, K >= 4096`` clause never sees a cell, since
    this row takes M <= 32 first)."""
    return k >= 8192 and m <= 104


def _narrow_long_k_class(m: int, n: int, k: int) -> bool:
    """The cells above the decode band that the swap-AB kernels may take: narrow N (``tiles_n <= 32``) with long
    K (``k >= 4096``), minus the two rules of the C++ cascade that were measured to win there: its two-kernel
    split-K (``_cascade_split_owns``) and its wave tile (``_sm100_smm.wave_tile_owns``: the 2-SM 256 x 64 tile for
    N = 2560 up to M = 256)."""
    from . import _sm100_smm
    if m <= MAX_DECODE_M or k < NARROW_LONG_K_MIN_K:
        return False
    if (n + 127) // 128 > NARROW_LONG_K_MAX_TILES_N:
        return False
    if _cascade_split_owns(m, k):
        return False
    return not _sm100_smm.wave_tile_owns(m, n)


def _narrow_long_k_config(m: int, n: int, k: int) -> Optional[_ConfigT]:
    """The M > 64 band of the row (issue #12, P5): narrow-N long-K cells, swap-AB, one launch, no workspace.

    Why. On N = 1280 / 2048 at M = 78-640 every other kernel puts 10-64 CTAs on the 148 SMs, each over the whole
    K: the cascade's 256-row tile (fc2 / conv_out at M <= 128, o at M = 80-128) runs a 128 x 128 x K tensor chain
    on a tile that is 50-70 % padding, tier 1's 128 x 128 tiles stream one K-long weight slice per CTA (design
    note ``fso_issue12_20261028/runs/P5/design.md``). Splitting K inside a cluster puts 2-4 times the CTAs on the
    same tiles.

    The form, in order:

    * the split-K kernel with a 64-wide token tile (several token tiles per problem) while its clusters, one per
      (weight tile, token tile), fit one wave of resident clusters: four slices when each slice keeps at least
      ``NARROW_LONG_K_MIN_SPLIT4_SLICE_K`` of K and the clusters of four fit, else two. The kernel's grid is not
      persistent, so a second wave would run the whole K chain again for the tail.
    * otherwise the persistent kernel's 2-CTA 256-row weight tile in a cluster of two with the 64-wide token tile
      (the decode row's form at 32 < M <= 64) while its ``ceil(N / 256) * ceil(M / 64)`` clusters fit one wave
      (``_persistent_wide``);
    * None (tier 1 or the cascade) where neither fits, and above the band's top
      (``NARROW_LONG_K_MAX_CUBLAS_CTAS``). The same form with a 128-wide token tile, which would fit one wave
      further up, is not used here: on its 40-cluster grid it beat tier 1 on o [2048 x 4096] at M = 640 (7.91
      against 8.50 us, gate 1) and lost on the Family A projections with the same grid and work, wo [2560 x 4096]
      and down [2560 x 9728] at M = 512 (8.02 against 7.55 and 14.70 against 14.26 us, run P5int famA, two passes):
      the difference is tier 1's own time at those cells, not anything the shape predicts. It serves the
      fall-through instead (``pick_fallthrough_config``), where the alternative is the cascade.

    Gate 1 (``fso_issue12_20261028/runs/P5g1``, two to four passes per arm, cold weights, graph replay) measured
    every form on o [2048 x 4096], fc2 [1280 x 5120] and conv_out [1280 x 7680] at M = 78-1079: the split-K form at
    o M = 80-256 (5.4-5.5 us, against 8.4-8.5 on the cascade and 7.2 on tier 1), fc2 / conv_out M = 78-364 (5.5-5.9
    / 6.5-7.4 us, against 8.2-9.8 / 11.0-13.5); the persistent form at o M = 384-512 (6.4-6.5 us against 7.5)
    and fc2 / conv_out M = 533 (7.2 / 9.4 against 8.5 / 11.4); on the Family A wo / down it took M = 257-448 at
    0.78-0.93 of tier 1's time (run P5int famA). A token tile narrower than 64 with M above it is never used: the
    kernel computes wrong output there (``SK.supports_tile``).
    """
    st = _init()
    if st is None or n % 128 != 0 or k % 128 != 0:
        return None
    if not _narrow_long_k_class(m, n, k):
        return None
    tiles_n = (n + 127) // 128
    if tiles_n * ((m + 127) // 128) > NARROW_LONG_K_MAX_CUBLAS_CTAS:
        return None
    cfg = _split_t64(m, n, k, st)
    if cfg is not None:
        return cfg
    return _persistent_wide(m, n, k, (NARROW_LONG_K_TOKEN_TILE,), st)


def _split_t64(m: int, n: int, k: int, st: dict) -> Optional[_ConfigT]:
    """The split-K kernel with the 64-wide token tile (``NARROW_LONG_K_TOKEN_TILE``), one cluster per (weight tile,
    token tile): four K slices when each slice keeps at least ``NARROW_LONG_K_MIN_SPLIT4_SLICE_K`` of K and the
    clusters of four fit one wave of resident clusters, else two (slices of at least ``MIN_SLICE_K``) while the
    clusters of two fit; the kernel's grid is not persistent, so a second wave would run the whole K chain again for
    the tail. Checked with the kernel's own ``is_valid_tactic``; None when no depth qualifies.

    Two callers. The narrow-N long-K band above the decode band (``_narrow_long_k_config``), and since FB2 the same
    class at 32 < M <= 64 (``_compute_config``): narrow N (``tiles_n <= 32``) with ``4096 <= k`` outside the cascade's
    two-kernel split-K (``k >= 8192``), where the band below used the persistent 2-CTA 256-row form with one K slice,
    whose ``ceil(N / 256)`` clusters put 16-20 CTAs on the machine (o [2048 x 4096], wo [2560 x 4096]). There one
    token tile covers the problem, so the split puts ``tiles_n * split`` CTAs on it instead. Measured on the B300 with
    cold weights against that form (run ``fso_issue12_20261028/runs/FB2``, step 1, two passes, both entry forms;
    2b = pre-quantized x, bf16 x = the activation quantize plus the GEMM): o [2048 x 4096] at M = 33-64 two slices,
    6.08-6.24 -> 5.36-5.40 us (2b) and 7.41-7.58 -> 6.77-6.88 us (bf16 x); wo [2560 x 4096] at M = 40-64 two slices,
    6.32-6.35 -> 5.57-5.58 and 7.57-7.69 -> 6.99-7.08; fc2 [1280 x 5120, bias + residual] at M = 48 / 64 four slices,
    8.05 / 8.16 -> 5.79 / 5.84 and 9.60 / 9.77 -> 7.28 / 7.52; conv_out [1280 x 7680] at M = 64 four slices, 9.27 ->
    6.29. The four-way split on o and wo (slices of 1024, below the four-way floor) measured 5.33-5.38 / 6.87-7.01
    and 5.76-5.77 / 7.17-7.56: on o within 0.03 us of two slices in the 2b form and slower in the bf16-x form, on wo
    slower in both, as the band's floor predicts."""
    SK, cutlass = st["SK"], st["cutlass"]
    tiles_n = (n + 127) // 128
    tile_n = NARROW_LONG_K_TOKEN_TILE
    clusters = tiles_n * ((m + tile_n - 1) // tile_n)
    for split_k in (4, 2):
        if k // split_k < (NARROW_LONG_K_MIN_SPLIT4_SLICE_K if split_k == 4 else MIN_SLICE_K):
            continue
        if clusters > _max_active_clusters(st, split_k):
            continue
        if SK.is_valid_tactic(m, k, cutlass.Float8E4M3FN, split_k, tile_n):
            return ((128, tile_n), (1, 1), True, split_k)
    return None


def _narrow_short_k_config(m: int, n: int, k: int) -> Optional[_ConfigT]:
    """The narrow-N short-K band above the decode band (FB2): narrow N (``tiles_n <= 32``) with
    ``MIN_WIDE_TOKEN_K <= k <= SHORT_K_TOKEN_TILE_MAX_K`` at ``m > 64``, the split-K kernel with 64-wide token tiles
    (several per problem) in two K slices, while its grid, ``2 * tiles_n * ceil(M / 64)`` CTAs, has at most
    ``NARROW_SHORT_K_MAX_CTAS``; every call of such a cell (identity, bias, residual, erf-GELU), after the kernel's own
    ``is_valid_tactic`` (``k % 256 == 0``). On the B300 that is M <= 256 for N = 1280 (the tower's q / k / v, out_proj
    and proj1), M <= 128 for N = 2048-2560 (proj2; the tp2 shards [2048 x 2048] / [2560 x 2048]) and M <= 320 for
    N = 1024 (Family C shared.gate_up [1024 x 2048]); N = 3072-4096 never qualifies above M = 64.

    Why. Without the band these cells run on tier 1 (identity calls, cuBLAS's 128 x 128 kernel over the whole K),
    on the C++ cascade's short-K tile (the bias calls tier 1 declines, ``_sm100_smm.short_k_owns``, and every
    residual or activation call: the 1-SM 128 x 64 tile, ``ceil(M / 128) * ceil(N / 64)`` CTAs, 20-40 on N = 1280),
    or on the cascade's wave tile (the tp2 qkv shard). Both leave most of the machine idle on these narrow
    problems; the split puts two CTAs on every (weight tile, token tile) of 128 x 64, each over half of K, and sums
    the halves inside the cluster without a second launch. Measured on the B300 with cold weights (run
    ``fso_issue12_20261028/runs/FB2``, steps 1, 1b and 1c, two passes, both entry forms; 2b = pre-quantized x,
    bf16 x = the activation quantize plus the GEMM), us, previous tier -> this form, 2b and bf16 x:

    * [1280 x 1280] at M = 78-256 (40-80 CTAs): identity 3.88-4.14 -> 3.38-3.59 and 5.36-5.51 -> 4.86-4.96; bias
      3.85-3.88 -> 3.79-3.83 and 5.41-5.44 -> 5.30-5.32; bias + residual 4.14-4.20 -> 4.10-4.14 and 5.77-5.88 ->
      5.59-5.62; bias + erf-GELU 4.62-4.73 -> 4.46-4.51 and 6.24-6.34 -> 6.01-6.04.
    * [2048 x 1280] at M = 78 / 117 (64 CTAs): identity 4.25 -> 3.49 and 5.69 -> 4.96; bias 3.92-3.94 -> 3.91-3.92
      and 5.47-5.48 -> 5.38-5.39; bias + erf-GELU 4.73 -> 4.59 and 6.34 -> 6.11; bias + residual 4.23 -> 4.22 and
      5.85 -> 5.69.
    * [1024 x 2048] at M = 96-320 (32-80 CTAs): identity 4.73-5.29 -> 3.84-3.97 and 6.28-6.72 -> 5.33-5.67; bias at
      M = 256 4.64 -> 4.21 and 6.19 -> 5.79. [2048 x 2048] at M = 96 / 128 (64 CTAs): identity 5.37-5.40 -> 4.02 and
      6.80-6.87 -> 5.49-5.51. [2560 x 2048] at M = 96 / 128 (80 CTAs): identity 4.71-4.72 -> 4.16-4.17 and
      6.24-6.27 -> 5.63-5.66; bias at M = 128 4.83 -> 4.48 and 6.35 -> 6.04.

    The CTA bound. Identity calls kept gaining on larger grids (to 128 CTAs: [1024 x 2048] at M = 512 4.76 -> 4.01
    and 6.50 -> 6.45), but fused calls did not: at 96 CTAs the bias call of [2048 x 1280] at M = 182 was 3.90 ->
    3.94 us (2b), at 120 CTAs the bias + residual and bias + erf-GELU calls of [1280 x 1280] at M = 364 were
    5.86 -> 6.05 and 6.35 -> 6.48 us (bf16 x) and the bias call of [3840 x 1280] at M = 78 4.10 -> 4.38 us (2b), at
    128 CTAs the bias call of [2048 x 1280] at M = 256 3.91 -> 3.99 and 5.48 -> 5.79 us. ``NARROW_SHORT_K_MAX_CTAS`` is
    the largest grid on which every measured call gained, so one rule serves every field of a cell and its calls
    keep one route. K below 1024 is not taken: a slice would be shorter than ``MIN_SLICE_K``, where one slice beat
    two (Family C shared.down [2048 x 512])."""
    st = _init()
    if st is None or n % 128 != 0 or k % 128 != 0:
        return None
    tiles_n = (n + 127) // 128
    if tiles_n > NARROW_LONG_K_MAX_TILES_N or not (MIN_WIDE_TOKEN_K <= k <= SHORT_K_TOKEN_TILE_MAX_K):
        return None
    tile_n = NARROW_LONG_K_TOKEN_TILE
    if 2 * tiles_n * ((m + tile_n - 1) // tile_n) > NARROW_SHORT_K_MAX_CTAS:
        return None
    SK, cutlass = st["SK"], st["cutlass"]
    if not SK.is_valid_tactic(m, k, cutlass.Float8E4M3FN, 2, tile_n):
        return None
    return ((128, tile_n), (1, 1), True, 2)


def _persistent_wide(m: int, n: int, k: int, token_tiles, st: dict) -> Optional[_ConfigT]:
    """The persistent kernel's 2-CTA 256-row weight tile in a cluster of two at the first token tile of
    ``token_tiles`` whose ``ceil(N / 256) * ceil(M / tile)`` clusters fit one wave of resident clusters of two
    (the persistent scheduler would run the excess as a second round on an otherwise idle machine), after the
    kernel's own ``can_implement`` in the swapped orientation; None when none fits."""
    PK, cutlass = st["PK"], st["cutlass"]
    resident = _max_active_clusters(st, 2)
    for tile_n in token_tiles:
        if ((n + 255) // 256) * ((m + tile_n - 1) // tile_n) > resident:
            continue
        tile, cluster = (256, tile_n), (2, 1)
        if PK.can_implement(cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32, cutlass.BFloat16,
                            tile, cluster, n, m, k, 1, "k", "k", "m"):
            return (tile, cluster, True, 1)
    return None


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
_EpiFlagsT = Optional[Tuple[bool, int, bool]]


def _epi_flags(epilogue) -> _EpiFlagsT:
    """The compile-time epilogue of a call: None for the identity, else
    ``(has_bias, act, has_residual)`` — the arguments the vendored kernels take
    as ``fso_epilogue``. Part of the compile cache key."""
    if epilogue is None:
        return None
    flags = (epilogue.bias is not None, int(epilogue.act_code), epilogue.residual is not None)
    return flags if any(flags) else None


def _get_compiled(st: dict, cfg: _ConfigT, sf_m: int, sf_n: int, sf_k: int,
                  epi: _EpiFlagsT = None, k: Optional[int] = None):
    """One ``cute.compile`` per (tactic, epilogue); the problem shape stays symbolic.

    The compiled entry takes the three operands as CuTe tensors with dynamic
    extents and the three scale-factor extents as ``Int64`` runtime arguments,
    so a single artifact serves every (N, K, M) that picks the same tactic and
    the extents passed here — those of whichever cell compiled first — do not
    enter the cache key. Compiling happens on the first eager call, never
    inside a capture.

    ``epi`` (``_epi_flags``) selects the epilogue variant of the kernel. The
    identity (``None``) constructs the kernel and calls ``cute.compile`` exactly
    as before the epilogue existed, through the vendored ``wrapper`` entry, so
    it generates the same code; a non-identity epilogue constructs the kernel
    with ``fso_epilogue`` and compiles its ``wrapper_epilogue`` entry, which
    takes the bias and the residual as raw bf16 pointers (``None`` for a field
    the variant does not apply) after alpha. The bias pointer is the ``[N]``
    tensor's; the residual pointer is read with the output's layout (under
    swap-AB the kernel's (N, M) view with unit stride on its first mode, which
    is a row-major ``[M, N]``). Both are assumed 2-byte aligned only, since
    the kernels read them element by element.
    """
    tile, cluster, swap_ab, split_k = cfg
    pdl = _pdl_enabled()
    # The weights-first form is a compile-time variant; ``k`` is the call's K, so one tactic has at most two
    # compiled entries, one per side of WF_MIN_K_BLOCKS (the problem shape itself stays symbolic).
    wf = _weights_first(None if k is None else k // 128 // split_k)
    cap = _wf_pre_wait_stages(cluster) if wf else None
    key = (tile, cluster, swap_ab, split_k, pdl, epi, wf, cap)
    fn = st["compiled"].get(key)
    if fn is not None:
        return fn
    cutlass, cute, utils = st["cutlass"], st["cute"], st["utils"]
    make_ptr, cuda_drv = st["make_ptr"], st["cuda"]

    kw = {} if epi is None else {"fso_epilogue": epi}
    if wf:
        kw["fso_weights_first"] = True
    if cap is not None:
        kw["fso_wf_cap"] = int(cap)
    if split_k > 1:
        gemm = st["SK"](32, tile, split_k, pdl, **kw)
    else:
        gemm = st["PK"](32, tile, cluster, False, pdl, **kw)
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

    if epi is None:
        fn = cute.compile(gemm.wrapper, a_fake, b_fake, c_fake, sf_m, sf_n, sf_k,
                          1, a_sf_fake, b_sf_fake, alpha_fake, mac, stream,
                          swap_ab, options="--opt-level 2")
    else:
        has_bias, _act, has_res = epi
        bias_fake = make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, 2) if has_bias else None
        res_fake = make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, 2) if has_res else None
        fn = cute.compile(gemm.wrapper_epilogue, a_fake, b_fake, c_fake, sf_m, sf_n, sf_k,
                          1, a_sf_fake, b_sf_fake, alpha_fake, bias_fake, res_fake, mac,
                          stream, swap_ab, options="--opt-level 2")
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
    """Wrap a torch tensor as a shape-dynamic CuTe tensor (host-side work on
    every call, which is why the operands are hoisted below)."""
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
      hits. Its descriptor is rebuilt on every eager call, which costs host
      time and nothing under CUDA-graph capture, where the whole closure runs
      once.
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
                        config: _ConfigT, epilogue=None) -> Optional[torch.Tensor]:
    """Run one decode-band MXFP8 GEMM. None → the caller falls through.

    ``epilogue`` (``_epilogue.Epilogue`` or None): bias, residual and erf-GELU
    are applied inside the kernel (``SERVED["sm100.decode"]``), in fp32 after
    the fp32 reduction (in the split-K class after the in-cluster sum, by the
    owner of each subtile, never per slice) and before the one bf16 rounding;
    the bias and residual are read by the epilogue warps after the kernel's
    griddepcontrol.wait with coherent loads. A field this row does not apply
    (the quantized output) is refused with NotImplementedError after the
    availability checks (so an unavailable row still falls through) and before
    anything is allocated or launched. The epilogue selects a compile variant
    (``_get_compiled``); the identity variant is today's kernel.

    The operands are handed over swapped: the weight ``[N, K]`` is the kernel's
    A and the activation ``[M, K]`` its B, with the two scale buffers following
    their operands. The BF16 output is allocated ``[M, N]`` row-major as usual
    and passed as the column-major view of itself, which is the layout the
    kernel writes in the swapped orientation; the buffer the caller receives is
    an ordinary row-major ``[M, N]``. The residual, of the output's shape and
    contiguous, is read through the same view.
    """
    st = _init()
    if st is None:
        return None
    if x_fp8.dim() != 2 or w_fp8.dim() != 2:
        return None
    if x_fp8.stride(1) != 1 or w_fp8.stride(1) != 1:
        return None
    if epilogue is not None:
        from . import _epilogue
        _epilogue.require("sm100.decode", epilogue, "linear_mxfp8")
    epi = _epi_flags(epilogue)
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

    fn = _get_compiled(st, config, sf_m, sf_n, sf_k, epi, k)
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
    if epi is None:
        fn(ct_a, ct_b, ct_c, sf_m, sf_n, sf_k, p_sfa, p_sfb, ct_alpha, stream)
        return out
    # The epilogue variant: the bias and the residual follow alpha as raw bf16
    # pointers; a field the variant does not apply is passed as None, the value
    # its slot was compiled with. The kernel binds the addresses, so a captured
    # graph reads the bias / residual at the addresses they had at capture,
    # like the operands and the scales.
    p_bias = make_ptr(cutlass.BFloat16, epilogue.bias.data_ptr(), gmem, assumed_align=2) if epi[0] else None
    p_res = make_ptr(cutlass.BFloat16, epilogue.residual.data_ptr(), gmem, assumed_align=2) if epi[2] else None
    fn(ct_a, ct_b, ct_c, sf_m, sf_n, sf_k, p_sfa, p_sfb, ct_alpha, p_bias, p_res, stream)
    return out
