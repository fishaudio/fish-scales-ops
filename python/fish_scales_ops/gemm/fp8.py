"""1×128 / 128×128 FP8 ops — quantize, GEMM, scale repack, and the
internal-quantize ``linear_qx`` variant.

Auto-dispatches sm_90 deep_gemm vs sm_120 CUTLASS BlockScaledKernel.
"""
from __future__ import annotations

import torch


def linear_fp8(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    sx: torch.Tensor,
    sw: torch.Tensor,
) -> torch.Tensor:
    """Block-scaled FP8 (E4M3) GEMM. Auto-dispatches sm_90 / sm_120.

    On sm_120 (Blackwell sm_120a) you must produce scales
    via ``quantize_*_fp8(..., use_ue8m0=True)`` — the wrapper repacks
    them to the int32-packed UE8M0 layout the CUTLASS kernel expects.
    On sm_90 (Hopper / H200) ``use_ue8m0=False`` (the default) is fine.

    Args:
        x_fp8: float8_e4m3fn [M, K], contiguous.
        w_fp8: float8_e4m3fn [N, K], contiguous.
        sx:    float32 dequant scales for x (see quantize helper output).
        sw:    float32 dequant scales for w.

    Returns:
        bfloat16 [M, N].
    """
    return torch.ops.fish_scales_ops.linear_fp8(
        x_fp8.contiguous(),
        w_fp8.contiguous(),
        sx.contiguous(),
        sw.contiguous(),
    )


def quantize_1x128_fp8(
    x: torch.Tensor,
    use_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [..., K] → FP8 (E4M3) [..., K] + FP32 per-row 1×128 dequant scales.

    Args:
        x: bf16 tensor with K divisible by 128.
        use_ue8m0: round scales to powers of 2 (UE8M0-representable).
            Set to ``True`` on sm_120 — the FP8 GEMM wrapper repacks
            these into the int32-packed UE8M0 format CUTLASS expects.
            ``False`` on sm_90 (deep_gemm path) is fine.

    Returns:
        (x_fp8, sx) where sx is float32 [pad(M,4), K/128] (TMA-aligned;
        the trailing pad rows hold zeros).
    """
    return torch.ops.fish_scales_ops.quantize_1x128(x.contiguous(), use_ue8m0)


def quantize_1x128_fp8_packed(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [..., K] → FP8 (E4M3) [..., K] + int32-packed UE8M0 1×128 scales.

    Fused single-kernel replacement for
    ``repack_fp8_act_scales(quantize_1x128_fp8(x, use_ue8m0=True)[1])`` on
    sm_120 — eliminates the FP32 scale round-trip through global memory and
    the separate repack-kernel launch.

    Returns:
        (x_fp8, sx_packed) — packed int32 K-major ``[pad(M,4), K/512]``
        (4 UE8M0 bytes per int32). Drop this into :func:`linear_fp8`
        as the activation scale on sm_120.

    Constraints: ``K % 512 == 0`` (4 K-blocks per packed int32).

    Use this on sm_120 instead of ``quantize_1x128_fp8 + repack_fp8_act_scales``.
    On sm_90 stick with ``quantize_1x128_fp8`` (the deep_gemm path consumes
    FP32 scales directly).
    """
    return torch.ops.fish_scales_ops.quantize_1x128_packed(x.contiguous(), True)


def quantize_128x128_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [N, K] weight → FP8 (E4M3) [N, K] + FP32 per-128×128-block dequant scales.

    Returns:
        (w_fp8, sw) where sw is float32 [ceil(N,128), ceil(K,128)].
    """
    return torch.ops.fish_scales_ops.quantize_128x128(w.contiguous())


def repack_fp8_act_scales(sx_f32: torch.Tensor) -> torch.Tensor:
    """FP32 1×128 act scales [pad(M,4), K/128] → int32 K-major [pad(M,4), K/512].

    **sm_120 only.** Pre-pack activation scales once when reusing across
    calls and pass the int32 result to ``linear_fp8``; the wrapper takes
    the fast path and skips the ~9 μs per-call repack. On sm_90 the
    deep_gemm kernel consumes the FP32 scales directly — do NOT call
    this on sm_90 and pass the int32 output to ``linear_fp8``; the
    runner expects FP32 there.
    """
    return torch.ops.fish_scales_ops.repack_fp8_act_scales(sx_f32.contiguous())


def repack_fp8_wgt_scales(sw_f32: torch.Tensor) -> torch.Tensor:
    """FP32 128×128 wgt scales [N/128, K/128] → int32 K-major [pad(N,4), K/512].

    **sm_120 only.** Pre-pack weight scales once per cached weight
    tensor; the 128×128 block scales get row-expanded across 128 N rows
    in the output. Pass the int32 result to ``linear_fp8`` to skip the
    per-call repack on sm_120. Do NOT call this on sm_90 — the deep_gemm
    kernel reads FP32 weight scales directly.
    """
    return torch.ops.fish_scales_ops.repack_fp8_wgt_scales(sw_f32.contiguous())


def linear_qx(x_bf16: torch.Tensor, w_fp8: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    """y = x @ w.T with x quantized inside the op and w pre-quantized.

    Equivalent to ``linear_fp8(*quantize_1x128_fp8(x), w_fp8, sw)`` but
    fuses the activation quantize launch with the GEMM. The cached-weight
    inference path.
    """
    return torch.ops.fish_scales_ops.linear_qx(
        x_bf16.contiguous(), w_fp8.contiguous(), sw.contiguous()
    )
