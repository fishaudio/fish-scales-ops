"""Public functional ops for the fish_scales_ops GEMM domain.

  * FP8 1×128 / 128×128 path — both archs (sm_90 deep_gemm JIT,
    sm_120 CUTLASS Sm120BlockScaledKernel).
  * MXFP8 1×32 path (sm_120 only) — CUTLASS BlockScaled mxf8f6f4.
  * BF16 convenience wrapper (single ATen op, internal quantize).

Canonical usage:

    import fish_scales_ops as fso

    # FP8 1×128
    xq, sx = fso.gemm.quantize_1x128_fp8(x, use_ue8m0=True)
    wq, sw = fso.gemm.quantize_128x128_fp8(w)
    y = fso.gemm.linear_fp8(xq, wq, sx, sw)

    # MXFP8 1×32 (sm_120 only)
    xq, sx = fso.gemm.quantize_1x32_fp8(x)
    wq, sw = fso.gemm.quantize_1x32_fp8(w)
    y = fso.gemm.linear_mxfp8(xq, wq, sx, sw)
"""
from __future__ import annotations

from .bf16 import linear_bf16
from .fp8 import (
    linear_fp8,
    linear_qx,
    quantize_128x128_fp8,
    quantize_1x128_fp8,
    quantize_1x128_fp8_packed,
    repack_fp8_act_scales,
    repack_fp8_wgt_scales,
)
from .mxfp8 import linear_mxfp8, quantize_1x32_fp8, silu_chunk_mul_quantize_1x32_fp8

__all__ = [
    # FP8 1×128 / 128×128
    "quantize_1x128_fp8",
    "quantize_1x128_fp8_packed",
    "quantize_128x128_fp8",
    "linear_fp8",
    "linear_qx",
    "repack_fp8_act_scales",
    "repack_fp8_wgt_scales",
    # MXFP8 1×32 (sm_120)
    "quantize_1x32_fp8",
    "silu_chunk_mul_quantize_1x32_fp8",
    "linear_mxfp8",
    # BF16 wrapper
    "linear_bf16",
]
