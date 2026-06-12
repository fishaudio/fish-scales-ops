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
    """cuBLAS beats DSL/C++ on the wide-N decode/small-M band.

    Bounds derived from per-GEMM bench (v3.9 diff, 2026-07-07):
    * Wide-N (tiles_n ≥ 40) M ≤ 1024: cuBLAS wins by 15-28%.
    * Narrow-N (tiles_n ≤ 32, i.e. down/wo class) M ≤ 128 with K ≥ 4096:
      FSO's split-K double-kernel scheme wins by 10-33% (down M=16
      10.35 vs scaled_mm 12.34, down M=64 12.34 vs 16.42) → skip.
    * K % 128 alignment required by scaled_mm.
    """
    if not _init():
        return False
    if k % 128 != 0:
        return False
    tiles_n = (n + 127) // 128
    # Narrow-N shapes stay on FSO — the split-K decode path owns this band.
    if tiles_n <= 32 and m <= 128 and k >= 4096:
        return False
    # Wide-N decode/small-M band.
    return m <= 1024


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
