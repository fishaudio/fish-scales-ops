"""Top-level three-tier dispatcher for the sm_100/sm_103 MXFP8 GEMM.

Every ``linear_mxfp8`` call on Blackwell datacenter (SM100/SM103) enters
here. The tiers, in order:

0. **Decode kernels** (see :mod:`._sm100_decode`) — the M ≤ 32 band, and only
   that band. Two vendored NVIDIA/FlashInfer CuTe-DSL kernels in swap-AB
   orientation with an 8/16/32-wide token tile: an in-cluster split-K kernel
   for narrow-N shapes and the plain persistent kernel for wide-N shapes.
   This row exists only when the installed ``nvidia-cutlass-dsl`` is at least
   4.5.0; below that it is inert and the three tiers below behave exactly as
   they did before it was added.

1. **cuBLAS ``scaled_mm``** (see :mod:`._sm100_smm`) — the small-M /
   decode wide-N band. Fair MLP bench 2026-07-07: cuBLAS's
   ``nvjet_128x128_128x6_4x1_v_bz`` (4-CTA multicast cluster + 6-stage
   pipeline) beats our CuTe DSL kernel by 15-28% at M ≤ 512 on wide-N
   shapes. Our fused-quantize output is byte-identical to what
   ``scaled_mm`` reads under ``SWIZZLE_32_4_4`` (both use CUTLASS
   ``Sm1xxBlockScaledConfig<32>``), so cuBLAS gets the same payload —
   no extra kernel launches, no re-permute.

2. **CuTe DSL persistent block-scaled kernel** (see :mod:`._sm100_dsl`)
   — the prefill / mid-band and peak. Wins over cuBLAS on M ≥ 2048
   through explicit L2 raster swizzle + cluster tuning that the
   CollectiveBuilder doesn't pick by default.

3. **C++ cascade** (``dispatch_sm100_mxfp8`` in ``dispatch.cuh``) —
   fall-through catch-all. Owns the ``down``/``wo`` narrow-N band above the
   decode band, which uses a two-kernel parallel split-K scheme that neither
   cuBLAS nor the DSL kernel can match, plus a NoSmem-epilogue
   overlapping-accumulator path for square cubic shapes.

Env kills: ``FSO_DISABLE_SMM=1`` skips tier 1, ``FSO_DISABLE_DSL=1``
skips tier 2 and the decode row, ``FSO_DISABLE_DECODE_DSL=1`` skips the decode
row alone, all together = pure C++.
"""
from __future__ import annotations

from typing import Optional

import torch


def route(x_fp8: torch.Tensor, w_fp8: torch.Tensor,
          sx: torch.Tensor, sw: torch.Tensor) -> Optional[torch.Tensor]:
    """Return the MXFP8 GEMM result via the best-per-shape tier, or None
    if all tiers decline (caller falls to the C++ op)."""
    if x_fp8.dim() != 2:
        return None
    m, k = x_fp8.shape
    n = w_fp8.shape[0]

    # Decode row: M ≤ 32 only. It is asked first because every tier below
    # loses that band — tier 1 and tier 2 cannot narrow their N tile below 128
    # and tier 3 answers a narrow-N decode cell with two serialised launches.
    # ``pick_config`` returns None for every cell it does not own, including
    # all of M > 32, so the tiers below are reached exactly as before.
    if m <= 32:
        from . import _sm100_decode
        cfg = _sm100_decode.pick_config(m, n, k)
        if cfg is not None:
            y = _sm100_decode.linear_mxfp8_decode(x_fp8, w_fp8, sx, sw, cfg)
            if y is not None:
                return y

    # Tier 1: cuBLAS scaled_mm for the small-M wide-N decode band.
    from . import _sm100_smm
    if _sm100_smm.should_route(m, n, k):
        y = _sm100_smm.linear_mxfp8_smm(x_fp8, w_fp8, sx, sw)
        if y is not None:
            return y

    # The C++ cascade's wave rule owns the cells where narrowing its N tile
    # fills the machine (see ``_sm100_smm.wave_tile_owns``). Tier 2's kernel is
    # persistent and built on a fixed 128- or 256-wide N tile, so on those cells
    # it leaves the same SMs idle that tier 1 does; without this guard its
    # wide-N rows would swallow the cell before the cascade ever sees it.
    if _sm100_smm.wave_tile_owns(m, n):
        return None

    # Tier 2: CuTe DSL kernel for mid-band + peak.
    from . import _sm100_dsl
    cfg = _sm100_dsl.pick_config(m, n, k)
    if cfg is not None:
        y = _sm100_dsl.linear_mxfp8_dsl(x_fp8, w_fp8, sx, sw, cfg)
        if y is not None:
            return y

    return None
