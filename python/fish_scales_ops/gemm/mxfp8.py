"""1×32 MXFP8 (true OCP MXFP8) ops — quantize + GEMM. sm_120 only.

CUTLASS ``Sm120MxFP8BlockScaledKernel`` (kSFVecSize=32) with the
(M, tiles_n) cascade and Stream-K shared with the 1×128 FP8 path.

On sm_90 (Hopper) both ops raise ``NotImplementedError``. Use
``linear_fp8`` (deep_gemm WGMMA BlockScaled) on Hopper.
"""
from __future__ import annotations

import torch

from .._arch import sm_major


def _require_sm120() -> None:
    if sm_major() < 12:
        raise NotImplementedError(
            "linear_mxfp8 / quantize_1x32_fp8 are sm_120-only. On sm_90 "
            "(Hopper) use linear_fp8 + quantize_1x128_fp8 / quantize_128x128_fp8."
        )


def linear_mxfp8(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    sx: torch.Tensor,
    sw: torch.Tensor,
) -> torch.Tensor:
    """Block-scaled MXFP8 (1×32 UE8M0) GEMM. sm_120 only.

    True OCP MXFP8: one E8M0 byte per 32 K-elements on both A and B.
    Finer than the 1×128 / 128×128 scheme — meaningfully tighter
    quantization at the cost of 4× more scale bytes.

    Args:
        x_fp8: float8_e4m3fn [M, K], K%128==0, N%128==0.
        w_fp8: float8_e4m3fn [N, K].
        sx, sw: int32 K-major UE8M0 scales `[pad(M,4), K/128]` from
            :func:`quantize_1x32_fp8` (auto-packed).

    Returns:
        bfloat16 [M, N].

    Raises:
        NotImplementedError: on sm_90 (this path is sm_120-only).
    """
    _require_sm120()
    return torch.ops.fish_scales_ops.linear_mxfp8_raw(
        x_fp8.contiguous(),
        w_fp8.contiguous(),
        sx.contiguous() if sx.is_contiguous() else sx,
        sw.contiguous() if sw.is_contiguous() else sw,
    )


def quantize_1x32_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [..., K] → FP8 (E4M3) [..., K] + UE8M0 1×32 scales. sm_120 only.

    True OCP MXFP8: one E8M0 byte per 32 elements along K.

    Returns:
        (x_fp8, sx) — x_fp8 has the input's shape; sx is int32 K-major
        ``[pad(M,4), K/128]`` (4 UE8M0 bytes packed per int32 word;
        ready for the CUTLASS BlockScaled kernel).

    Constraints: K must be a multiple of 128. The flattened M must be a
    multiple of 4 (TMA alignment, padded internally).

    Raises:
        NotImplementedError: on sm_90 (this path is sm_120-only).
    """
    _require_sm120()
    # Fused quantize + packed-scale write — single CUDA kernel, no FP32 scale
    # round-trip. Bit-exact with the legacy `quantize_1x32` + `repack_mxfp8_scales`
    # two-step path.
    return torch.ops.fish_scales_ops.quantize_1x32_packed(x.contiguous(), True)


def silu_chunk_mul_quantize_1x32_fp8(gu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SwiGLU prologue + MXFP8 quantize. sm_120 only.

    Takes ``gu`` (bf16 ``[..., 2*INTER]``) where the first INTER cols along
    the last dim are ``gate`` and the second INTER cols are ``up``. Computes
    ``h = silu(gate) * up`` and quantizes ``h`` to FP8 + packed UE8M0 scale
    **without materialising ``h`` in global memory** — saves a M·INTER·2
    byte intermediate read+write vs the unfused chain
    (silu·chunk + ``quantize_1x32_fp8``).

    Returns:
        (x_fp8, sx_packed) — same packed layout as :func:`quantize_1x32_fp8`,
        ready for :func:`linear_mxfp8` as the activation operand.

    Constraints: ``INTER % 128 == 0``.

    Raises:
        NotImplementedError: on sm_90.
    """
    _require_sm120()
    return torch.ops.fish_scales_ops.silu_chunk_mul_quantize_1x32(gu.contiguous(), True)
