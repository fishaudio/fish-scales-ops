"""CuTe DSL backend for the sm_100/sm_103 MXFP8 1×32 GEMM mid-band.

Wraps CUTLASS's ``Sm100BlockScaledPersistentDenseGemmKernel`` (the CuTe DSL
persistent blockscaled kernel with *overlapping accumulators*: TMEM
double-buffering lets the epilogue drain one accumulator while the MMA fills
the next).

The C++ path keeps decode (M ≤ 128, including its split-K), the narrow-N
short-K cells its TileK = 256 instantiations serve (wo / down at small M), and
the rows ``pick_config`` below does not name. The bring-up measurements of
2026-07-06 are in profile/ncu_sm100_2026-07-06/REPORT.md, outside the
repository; the published fso numbers are in ``docs/perf/gemm/sm100.md``.

Contracts:

* Inputs are exactly the C++ path's: fp8 e4m3 [M,K]/[N,K] K-contiguous +
  opaque int32 UE8M0 scales in the Sm1xx atom layout (bit-compatible —
  validated cos 0.9993 against the BF16 reference, identical to C++).
* Compile is per (tiler, cluster) — problem shape is a RUNTIME argument, so
  a handful of cached compiles serve every shape. First use of a config
  JIT-compiles it; the eager-warmup-before-capture contract from
  test_cuda_graph.py covers it, and kernel launches are capture-safe
  (validated: capture + NaN-clobber + replay → bit-identical cos).
* ``FSO_DISABLE_DSL=1`` kills the route (falls back to the C++ cascade); like
  every boolean ``FSO_*`` switch, a value starting with ``0`` means off
  (:mod:`fish_scales_ops._env`). The kernel source is the CUTLASS example under
  ``3rdparty/cutlass`` in the source tree, or ``FSO_DSL_KERNEL_PATH``.
* ``_init`` is silent only when the tier is switched off or the optional
  ``nvidia-cutlass-dsl`` package (the ``sm100`` extra) is not installed. Any
  other reason the tier cannot load — the kernel file is missing, as in an
  installation without the source tree, loading it raises, or part of the DSL
  stack fails to import — warns once per process with one ``fso:`` line on
  stderr, and the router falls back to the remaining tiers.
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch

from .._env import env_flag

_STATE: Optional[dict] = None


def _default_kernel_path() -> str:
    """The CUTLASS example the tier compiles, at its place in the source tree."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    # Base persistent blockscaled kernel — pointer-based compile
    # via scaled_mm(), shape-generic. The prefetch variant is not
    # used: under the production torch.cudagraph protocol on the
    # b300 (2026-07-07 v3.3 bench) it did not hold up the gain its
    # `cute.testing.benchmark` probes suggested and regressed on huge
    # cubic shapes. Kept for a future retry, disabled.
    return os.path.join(
        repo, "3rdparty", "cutlass", "examples", "python", "CuTeDSL",
        "blackwell", "dense_blockscaled_gemm_persistent.py")


def _warn_unavailable(reason: str) -> None:
    """Warn that the tier should be there and is not: one line on stderr, in
    the ``fso:`` format of the decode row's notices (stdout belongs to the
    benches' JSON protocol). ``_init`` runs once per process, so this prints at
    most once."""
    print("fso: sm_100/103 CuTe-DSL mid-band tier unavailable: %s; linear_mxfp8 / "
          "linear_fp8 fall back to the remaining tiers. Point FSO_DSL_KERNEL_PATH at "
          "CUTLASS's examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py, "
          "or set FSO_DISABLE_DSL=1 to skip the tier without this warning"
          % " ".join(str(reason).split()), file=sys.stderr, flush=True)


def _init() -> Optional[dict]:
    """Lazy one-time init. Returns None when the DSL stack is unavailable.

    Silent when ``FSO_DISABLE_DSL`` is on or ``nvidia-cutlass-dsl`` is not
    installed; every other failure warns once (``_warn_unavailable``).
    """
    global _STATE
    if _STATE is not None:
        return _STATE if _STATE.get("ok") else None
    _STATE = {"ok": False}
    if env_flag("FSO_DISABLE_DSL"):
        return None
    try:
        import cutlass
    except ModuleNotFoundError as exc:
        if exc.name == "cutlass":
            return None             # the optional `sm100` extra is not installed
        _warn_unavailable("importing cutlass failed: %s: %s" % (type(exc).__name__, exc))
        return None
    except Exception as exc:
        _warn_unavailable("importing cutlass failed: %s: %s" % (type(exc).__name__, exc))
        return None
    try:
        import importlib.util

        import cuda.bindings.driver as cuda_drv
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import cutlass.utils as dsl_utils
        from cutlass.cute.runtime import from_dlpack, make_ptr
    except Exception as exc:
        _warn_unavailable("importing the CuTe-DSL stack failed: %s: %s" % (type(exc).__name__, exc))
        return None

    kernel_path = os.getenv("FSO_DSL_KERNEL_PATH") or _default_kernel_path()
    if not os.path.exists(kernel_path):
        _warn_unavailable("kernel source %s does not exist" % kernel_path)
        return None
    try:
        spec = importlib.util.spec_from_file_location("_fso_sm100_dsl_kernel", kernel_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as exc:
        _warn_unavailable("loading kernel source %s raised %s: %s"
                          % (kernel_path, type(exc).__name__, exc))
        return None

    _STATE.update(ok=True, cuda=cuda_drv, cutlass=cutlass, cute=cute,
                  cutlass_torch=cutlass_torch, utils=dsl_utils,
                  make_ptr=make_ptr, from_dlpack=from_dlpack,
                  mod=mod, compiled={}, kernel_path=kernel_path)
    return _STATE


def _from_torch(st: dict, t: torch.Tensor, dtype, leading_dim: int):
    """Wrap a real torch tensor as a shape-dynamic CuTe tensor.

    ``leading_dim`` = the index of the dim whose stride is 1. Uses
    from_dlpack directly (no copy, no alloc) — unlike ``cute_tensor_like``
    which builds a fresh device buffer.
    """
    from_dlpack = st["from_dlpack"]
    ct = from_dlpack(t, assumed_align=16)
    ct.element_type = dtype
    ct = ct.mark_layout_dynamic(leading_dim=leading_dim)
    return ct


def _kernel_class(mod, cute_mod, utils_mod, force_stages: Optional[int],
                  swizzle_size: int):
    """Return the DSL kernel class, optionally with (a) an override of
    ``num_ab_stage`` and (b) a persistent-tile-scheduler ``swizzle_size``
    injection (L2 raster reorder — the same lever the C++ path uses via
    ``max_swizzle_size``; the DSL example defaults to swizzle=1).

    Stage probe (2026-07-07, sm100-v2-work/stage_probe.log): default
    heuristic near-optimal, only (256,128)c(2,1) cells win under S=8.

    Swizzle probe (2026-07-07, sm100-v2-work/swizzle_probe.log): swizzle = 8
    pays off on cubic shapes from 8192 up; small tile counts regress under a
    swizzle above 1.
    """
    base = mod.Sm100BlockScaledPersistentDenseGemmKernel
    if force_stages is None and swizzle_size == 1:
        return base

    class Custom(base):
        pass

    if force_stages is not None:
        _stages = force_stages

        @staticmethod
        def _compute_stages(*args, **kwargs):
            acc, _ab, c = base._compute_stages(*args, **kwargs)
            return acc, _stages, c
        Custom._compute_stages = _compute_stages

    if swizzle_size > 1:
        _sw = swizzle_size

        @staticmethod
        def _compute_grid(c, cta_tile_shape_mnk, cluster_shape_mn, max_active_clusters):
            c_shape = cute_mod.slice_(cta_tile_shape_mnk, (None, None, 0))
            gc = cute_mod.zipped_divide(c, tiler=c_shape)
            num_ctas_mnl = gc[(0, (None, None, None))].shape
            cluster_shape_mnl = (*cluster_shape_mn, 1)
            tsp = utils_mod.PersistentTileSchedulerParams(
                num_ctas_mnl, cluster_shape_mnl,
                swizzle_size=_sw, raster_along_m=True)
            grid = utils_mod.StaticPersistentTileScheduler.get_grid_shape(
                tsp, max_active_clusters)
            return tsp, grid
        Custom._compute_grid = _compute_grid

    Custom.__name__ = f"{base.__name__}_S{force_stages}_SW{swizzle_size}"
    return Custom


def _get_compiled(st: dict, tiler: Tuple[int, int], cluster: Tuple[int, int],
                  force_stages: Optional[int] = None, swizzle_size: int = 1):
    """Base variant: pointer-based compile via scaled_mm() — shape-generic,
    one compile per (tiler, cluster, force_stages, swizzle_size)."""
    key = (tiler, cluster, force_stages, swizzle_size)
    fn = st["compiled"].get(key)
    if fn is not None:
        return fn
    cutlass, mod, utils, cuda_drv, cute = (
        st["cutlass"], st["mod"], st["utils"], st["cuda"], st["cute"])
    KernelCls = _kernel_class(mod, cute, utils, force_stages, swizzle_size)
    gemm = KernelCls(32, tiler, cluster)
    mac = utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1])
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = mod.scaled_mm(
        gemm, cutlass.Float8E4M3FN, cutlass.BFloat16, cutlass.Float8E8M0FNU,
        "k", "k", "n", mac, stream, options="--opt-level 2")
    st["compiled"][key] = fn
    return fn


_ConfigT = Tuple[Tuple[int, int], Tuple[int, int], Optional[int], int]


def pick_config(m: int, n: int, k: int) -> Optional[_ConfigT]:
    """DSL routing + config table.

    Distilled from the 14-config × 32-shape best-shot sweep on b300 GPU1,
    2026-07-07 (``sm100-v2-work/dsl_full_sweep.log``). Rows chosen by
    minimum us with the current v3.2 C++ cascade times as the "keep C++"
    threshold. See ``sm100-v2-work/dispatch_derivation.md`` for the
    per-cell rationale. Returns None → fall through to the C++ cascade.
    """
    tiles_n = (n + 127) // 128

    # ---- Cubic square shapes (M == N == K) — dispatch by size ------------
    if m == n == k:
        # ONLY route cubic sizes we've actually measured. Non-standard
        # sizes (M=2560, 3072, 1536, ...) fall through to C++ — the sweep
        # data only covered {4096, 6144, 8192, 12288, 16384}, and
        # extrapolating regressed cubic-2560 badly in the v3.6 bench.
        if m == 4096:
            return (256, 128), (2, 1), 8, 1
        if m == 6144:
            return (256, 256), (2, 1), None, 1
        if m == 8192:
            return (256, 256), (2, 1), None, 8   # swizzle=8, per the swizzle probe
        if m in (12288, 16384):
            # Huge cubic: swizzle=8, as for 8192. The bench showed a smaller
            # gain than the probe on these cells, which the bench worker's
            # known scatter on huge-M cells of the b300 can hide; the row
            # stays ahead of the C++ path there.
            return (256, 256), (2, 1), None, 8
        return None

    # ---- Narrow-N band (tiles_n ≤ 32): wo / down / other narrow shapes --
    if tiles_n <= 32:
        # The DSL tier declines this whole class.
        #
        # A narrow-N shape gives the persistent scheduler at most 32 N-tiles
        # (20 for the Family A `down` shape), and every configuration the DSL
        # kernel can be built with turns that into a grid that either gives up
        # SMs or quantises badly. With a 4-CTA cluster the driver reports 33
        # max active clusters on a 148-SM part, so the grid is 132 CTAs and a
        # tenth of the machine is idle for the whole kernel. With a 2-CTA
        # cluster the grid fills the machine but the tile count divides into
        # 74-cluster waves, and on a 10- or 20-tile-wide problem the remainder
        # is a large fraction of the last wave. Neither lever removes the
        # other's cost.
        #
        # Measured over six M values on three narrow-N shapes (see the sweep
        # in the B300 run b300_mlp_tune_20260915): the sweep put no narrow-N
        # cell above M = 1024 on a DSL configuration — the low-wave and
        # short-K cells went to cuBLAS and the long-K deep-grid cells to the
        # C++ cascade, and `_sm100_smm.should_route` now encodes that split.
        # Three rows were removed here:
        #   * `k >= 4096 and m >= 2048` -> (256,128) c(2,2): the sweep put
        #     those cells on cuBLAS up to M = 3072 and on the C++ cascade above.
        #   * `k >= 4096 and m == 4096 and n <= 2560` -> same configuration,
        #     same verdict (it only restated the row above).
        #   * `k >= 8192 and m == 1024` -> (256,256) c(4,1): dead code, since
        #     `should_route` accepts narrow-N at 128 < m <= 1024 and takes the
        #     cell for cuBLAS first. Timed directly for this decision, it gave
        #     nothing over the routes that serve that cell, so nothing was lost
        #     by its being unreachable and nothing would be gained by reviving it.
        return None

    # ---- Wide-N band (tiles_n > 32): wqkv / gate / gate_up ---------------
    # NOTE: since the 2026-09-15 routing round `_sm100_smm.should_route`
    # accepts every wide-N shape at every M, so these rows are now only
    # reached when tier 1 is unavailable — K not a multiple of 128, or
    # `torch.nn.functional.scaled_mm` missing, or FSO_DISABLE_SMM=1. They are
    # kept as that fall-back, not as the production pick. The sweep behind the
    # change put every wide-N cell from M = 1536 to 8192 on cuBLAS rather than
    # on the best DSL configuration; if a future device or CUTLASS release
    # changes that, re-measure here rather than restoring the old M <= 1024 cap
    # in should_route.
    # Widened: gate_up M=512 goes DSL (256,128)c(2,1) per the sweep;
    # gate_up M=256 stays on the C++ path.
    if n >= 16384 and m < 512:
        return None
    if m < 256:
        return None                          # decode band belongs to C++

    if m >= 4096:
        return (256, 256), (2, 1), None, 1
    if m >= 2048:
        if n >= 16384:
            # gate_up M=2048 (256,128)c21: S=8, per the stage probe
            return (256, 128), (2, 1), 8, 1
        return (256, 256), (2, 2), None, 1
    if m >= 1024:
        if n >= 16384:
            return (256, 256), (4, 1), None, 1
        return (256, 128), (2, 1), None, 1
    # m in [256, 1024).
    # Wave-quant patch (decode_sweep.log 2026-07-07): wqkv-class shapes
    # (tiles_n ∈ [40, 64]) at M ∈ (256, 384] under-fill the machine with
    # (256,128) 2SM — 3-4 tiles × ~48 CTAs uses only ~1/3 of 148 SMs.
    # (128,128) 1SM: 3×tiles_n CTAs = 1 perfect wave. Beyond M=384 the
    # K=256 mainloop advantage of (256,128) wins back.
    if 256 < m <= 384 and 40 <= tiles_n <= 64:
        return (128, 128), (1, 1), None, 1
    return (256, 128), (2, 1), None, 1


def linear_mxfp8_dsl(x_fp8: torch.Tensor, w_fp8: torch.Tensor,
                     sx: torch.Tensor, sw: torch.Tensor,
                     config: _ConfigT) -> Optional[torch.Tensor]:
    """Run one MXFP8 GEMM through the DSL kernel. Returns None when the DSL
    stack is unavailable (caller falls back to the C++ op)."""
    st = _init()
    if st is None:
        return None
    m, k = x_fp8.shape
    n = w_fp8.shape[0]
    tile, cluster = config[0], config[1]
    stages = config[2] if len(config) > 2 else None
    swizzle = config[3] if len(config) > 3 else 1
    fn = _get_compiled(st, tile, cluster, stages, swizzle)
    cute, cuda_drv = st["cute"], st["cuda"]
    cutlass, make_ptr = st["cutlass"], st["make_ptr"]

    out = torch.empty(m, n, dtype=torch.bfloat16, device=x_fp8.device)
    gmem = cute.AddressSpace.gmem
    a_ptr = make_ptr(cutlass.Float8E4M3FN, x_fp8.data_ptr(), gmem, assumed_align=16)
    b_ptr = make_ptr(cutlass.Float8E4M3FN, w_fp8.data_ptr(), gmem, assumed_align=16)
    sfa_ptr = make_ptr(cutlass.Float8E8M0FNU, sx.data_ptr(), gmem, assumed_align=32)
    sfb_ptr = make_ptr(cutlass.Float8E8M0FNU, sw.data_ptr(), gmem, assumed_align=32)
    c_ptr = make_ptr(cutlass.BFloat16, out.data_ptr(), gmem, assumed_align=16)
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, n, k, 1), stream)
    return out
