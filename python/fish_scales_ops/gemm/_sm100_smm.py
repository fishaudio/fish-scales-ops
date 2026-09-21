"""cuBLAS ``scaled_mm`` dispatch for the sm_100/sm_103 MXFP8 small-M band.

The fused ``quantize_1x32_fp8`` kernel's raw int32-packed SF buffer is
byte-identical to the ``Sm1xxBlockScaledConfig<32>`` atom layout that
``torch.nn.functional.scaled_mm`` expects under ``SWIZZLE_32_4_4``. So we
can feed cuBLAS *the exact same* quantized payload — no re-permute, no
extra kernel launches — and route calls to whichever GEMM is faster per
shape.

Fair MLP bench (sm100-v2-work/fair_gemm_bench.py, 2026-07-07, b300 GPU1):

  M      fso        smm       fair/fso
    1    30.78      25.40     0.83× ← cuBLAS wins decode
   16    26.70      24.69     0.93×
   32    28.71      24.65     0.86×
  128    36.28      32.82     0.90×
  512    57.39      51.59     0.90×
 1024    85.37      82.54     0.97× ~tied
 2048   147.94     155.55     1.05× ← FSO DSL wins prefill
 4096   256.43     303.79     1.19×

Root cause: cuBLAS's ``nvjet_128x128_128x6_4x1_v_bz`` uses a 4-CTA
multicast cluster + 6 smem stages for small M; the CuTe DSL
``Sm100BlockScaledPersistentDenseGemmKernel`` defaults to c(1,1) at those
shapes.  Rather than write our own small-M cluster kernel, route the
band that cuBLAS wins to cuBLAS.

``FSO_DISABLE_SMM=1`` kills the route.
"""
from __future__ import annotations

import os
from typing import Optional

import torch


_SM_COUNT: Optional[int] = None


def _sm_count() -> int:
    """Multiprocessor count of the current device, read once.

    Used by the large-M narrow-N rule below to reason about how many
    cluster-waves a kernel will run. Cached so the routing decision stays a
    few integer operations on the hot path, and first touched during the
    eager warm-up every caller already performs before CUDA-graph capture.
    """
    global _SM_COUNT
    if _SM_COUNT is None:
        _SM_COUNT = torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count
    return _SM_COUNT


def _cascade_cluster_waves(m: int, n: int) -> float:
    """Cluster-waves the C++ cascade would run on a large-M narrow-N shape.

    Above M = 2048 with N < 4096 the C++ cascade
    (``gemm_dispatch_sm100_mxfp8``) selects a (256,128) 2-SM tile in a
    cluster(2,1), i.e. a 128x128 CTA tile with two CTAs per cluster. The
    persistent scheduler can keep ``SM_count / 2`` such clusters resident, so
    the number of waves that kernel runs is

        ceil(ceil(M/128) / 2) * ceil(N/128) / (SM_count / 2).

    The quantity matters because the last wave is the only partially filled
    one: at a low wave count the tail is a large fraction of the whole kernel,
    and the fixed per-CTA cost (prologue, TMA descriptor setup, epilogue
    store) is amortised over fewer K iterations.
    """
    ctas_m = (m + 127) // 128
    ctas_n = (n + 127) // 128
    clusters = ((ctas_m + 1) // 2) * ctas_n
    return clusters / max(_sm_count() // 2, 1)


def wave_tile_owns(m: int, n: int) -> bool:
    """True when the sm_100 C++ cascade's wave rule owns this (M, N) cell.

    Mirrors ``pick_wave_tile`` in
    ``csrc/gemm/include/blockscale_gemm/arch/sm100/mxfp8/dispatch.cuh``, whose
    comment carries the mechanism and the measured bounds. One CTA is resident
    per SM on this part (the block-scaled mainloop stages 200-230 KB of shared
    memory), so the CTA count is the number of SMs that do any work; the rule
    takes the tile that puts the most CTAs on the machine without spilling into
    a second wave, and then accepts the pick only if all four measured bounds
    hold: 64 <= M <= 1024, the pick is one of the N-tile widths added on
    2026-09-17 (64 or 192), the grid busies at least half the SMs, a 64-wide
    pick is not used above M = 128, and the 2-SM 192-wide form is never used.

    Both tier 1 (cuBLAS ``scaled_mm``) and tier 2 (the CuTe-DSL persistent
    kernel) decline a cell the rule owns: neither can be asked for a narrower N
    tile, so both leave SMs idle exactly where the cascade does not.
    """
    if not (64 <= m <= 1024):
        return False
    sms = _sm_count()
    best_ctas, best_n, best_rows = 0, 0, 0
    for rows, per in ((128, 1), (256, 2)):
        tiles_m = -(-m // rows)
        for tn in (256, 192, 128, 64):
            ctas = tiles_m * (-(-n // tn)) * per
            if ctas <= sms and ctas > best_ctas:
                best_ctas, best_n, best_rows = ctas, tn, rows
    if best_n not in (64, 192):
        return False
    if 2 * best_ctas < sms:
        return False
    if best_n == 64 and m > 128:
        return False
    if best_rows == 256 and best_n == 192:
        return False
    return True


_AVAILABLE: Optional[bool] = None
_ST_BLOCKWISE1X32 = None
_SW_32_4_4 = None
_SCALED_MM = None


def _init() -> bool:
    global _AVAILABLE, _ST_BLOCKWISE1X32, _SW_32_4_4, _SCALED_MM
    if _AVAILABLE is not None:
        return _AVAILABLE
    if os.getenv("FSO_DISABLE_SMM"):
        _AVAILABLE = False
        return False
    try:
        from torch._C import _ScalingType, _SwizzleType

        _ST_BLOCKWISE1X32 = _ScalingType.BlockWise1x32
        _SW_32_4_4 = _SwizzleType.SWIZZLE_32_4_4
        _SCALED_MM = torch.nn.functional.scaled_mm
        _AVAILABLE = True
    except (ImportError, AttributeError):
        _AVAILABLE = False
    return _AVAILABLE


def _sf_view(sx: torch.Tensor, mn: int, k: int) -> torch.Tensor:
    """View our int32-packed SF as ``float8_e8m0fnu`` with the byte count
    scaled_mm expects (``Sm1xxBlockScaledConfig<32>`` atom layout is
    byte-identical to ``SWIZZLE_32_4_4``)."""
    n_row_blocks = (mn + 127) // 128
    n_col_blocks = (((k + 31) // 32) + 3) // 4
    numel = n_row_blocks * n_col_blocks * 512
    return sx.view(torch.uint8).flatten()[:numel].view(torch.float8_e8m0fnu)


def should_route(m: int, n: int, k: int) -> bool:
    """Decide whether this GEMM should run on cuBLAS ``scaled_mm`` (tier 1).

    Small-M bounds are unchanged from the per-GEMM bench of 2026-07-07:
    * Wide-N (tiles_n ≥ 40) M ≤ 1024: cuBLAS wins.
    * Narrow-N (tiles_n ≤ 32, i.e. down/wo class) M ≤ 128 with K ≥ 4096:
      FSO's split-K double-kernel scheme wins → skip.
    * K % 128 alignment required by scaled_mm.

    The large-M rules (M > 1024) come from the per-projection sweep in
    the B300 run ``b300_mlp_tune_20260915`` (sweep1.jsonl,
    sweep2a/2b/2c.jsonl), which timed cuBLAS, every reachable CuTe-DSL
    configuration and the C++ cascade on six M values per shape, three
    wide-N shapes and three narrow-N shapes. Before that sweep this function
    simply returned ``m <= 1024``, so no cell above M = 1024 could reach
    cuBLAS at all and every one of them was taken by ``_sm100_dsl`` rows that
    had been extrapolated from M ≤ 4096 measurements.

    **Wide-N (tiles_n > 32) at any M → cuBLAS.** The DSL tier's kernel is
    persistent: it launches ``min(total_clusters, max_active_clusters) *
    cluster_size`` CTAs and loops over tiles. On this device the driver
    reports 148 / 74 / 33 max active clusters for cluster sizes 1 / 2 / 4, so
    a 4-CTA cluster configuration can occupy only 132 of the 148 SMs — it
    gives up a tenth of the machine before it starts — and a 2-CTA cluster
    configuration fills the machine but then quantises the tile count into
    74-cluster waves. cuBLAS's nvjet kernel is not persistent: its grid grows
    with the tile count, so on a wide-N shape it runs many waves and its tail
    is a small fraction of the whole. Measured on gate_up (tiles_n = 152),
    wqkv (48) and gdn.in_proj (96), cuBLAS is the fastest engine at every M
    from 1536 to 8192.

    **Narrow-N (tiles_n ≤ 32) above M = 1024** splits on K. The C++ cascade's
    kernel for this band amortises its fixed per-CTA cost over K/128 mainloop
    iterations, so it only reaches its asymptotic rate on long-K shapes; and
    it needs enough cluster-waves that its one partially filled tail wave is
    a small fraction of the kernel. Both conditions hold for the Family A
    ``down`` shape (K = 9728) from M = 4096 upward, where the cascade is the
    fastest engine; neither the K = 4096 narrow-N shapes (``wo``,
    ``gdn.out_proj``) nor the low-wave cells of ``down`` reach that point, and
    there cuBLAS wins. So: keep cuBLAS unless the shape is long-K **and** the
    cascade would run at least four cluster-waves.

    Square shapes above the decode band are excluded and left to
    ``_sm100_dsl.pick_config``, whose cubic table is separately measured.
    """
    if not _init():
        return False
    if k % 128 != 0:
        return False
    # Cubic shapes keep their own measured table in `_sm100_dsl.pick_config`;
    # tier 1 declines so the wide-N rule below cannot swallow them. Bounded to
    # m > 1024 so nothing in the decode band changes.
    if m > 1024 and m == n == k:
        return False
    # Decode and mid band: decline where the C++ cascade can fill more of the
    # machine than cuBLAS's kernel will. cuBLAS's nvjet picks a 128- or
    # 256-wide N tile and there is no way to ask it for a narrower one, so on a
    # shape whose tile grid is smaller than the SM count it leaves SMs idle for
    # the whole kernel (ncu, 2026-09-17: wqkv M = 64 runs 48 CTAs on 148 SMs,
    # gate_up M = 64 runs 76). The cascade's wave rule narrows its N tile until
    # the grid reaches the SM count from below; ``wave_tile_owns`` is true
    # exactly on the cells where that narrowing produces one of the new tile
    # widths and was measured to win.
    if wave_tile_owns(m, n):
        return False
    tiles_n = (n + 127) // 128
    if tiles_n <= 32:
        # Narrow-N decode band stays on FSO — the split-K path owns it.
        if m <= 128 and k >= 4096:
            return False
        if m > 1024:
            # Long-K narrow-N with a deep enough cascade grid: the C++
            # cascade wins, so decline and let it fall through (pick_config
            # also returns None for this class).
            return not (k >= 8192 and _cascade_cluster_waves(m, n) >= 4.0)
        return True
    # Wide-N: cuBLAS at every M.
    return True


def linear_mxfp8_smm(x_fp8: torch.Tensor, w_fp8: torch.Tensor,
                    sx: torch.Tensor, sw: torch.Tensor) -> Optional[torch.Tensor]:
    """Run one MXFP8 GEMM through ``torch.nn.functional.scaled_mm``.
    Returns None if the API is unavailable (caller falls through)."""
    if not _init():
        return None
    m, k = x_fp8.shape
    n = w_fp8.shape[0]
    sx_view = _sf_view(sx, m, k)
    sw_view = _sf_view(sw, n, k)
    return _SCALED_MM(
        x_fp8, w_fp8.t(),
        sx_view, _ST_BLOCKWISE1X32,
        sw_view, _ST_BLOCKWISE1X32,
        swizzle_a=_SW_32_4_4, swizzle_b=_SW_32_4_4,
        output_dtype=torch.bfloat16,
    )
