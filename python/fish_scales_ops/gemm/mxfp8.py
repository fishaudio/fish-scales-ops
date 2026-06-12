"""1×32 MXFP8 (true OCP MXFP8) ops — quantize + GEMM. sm_100/sm_103/sm_120.

Arch routing (inside ``linear_mxfp8_raw`` / ``quantize_1x32_packed``):

* sm_120/121 (consumer Blackwell): CUTLASS ``Sm120MxFP8BlockScaledKernel``
  (kSFVecSize=32) with the (M, tiles_n) cascade and Stream-K shared with
  the 1×128 FP8 path. Scales: int32 K-major ``[pad(M,4), K/128]``.
* sm_100/103 (B200 / B300 datacenter Blackwell): CUTLASS tcgen05
  BlockScaled collective (``arch::Sm100`` builder, compiled as sm_100f
  family target). Scales: opaque 1-D int32 in the CUTLASS
  ``Sm1xxBlockScaledConfig`` atom layout, ``[ceil(M/128)*128 * K/128]``.

The scale tensor is an *opaque handle*: always produce it with
:func:`quantize_1x32_fp8` / :func:`silu_chunk_mul_quantize_1x32_fp8` on the
same device the GEMM will run on — layouts are NOT interchangeable between
sm_120 and sm_100/103.

On sm_90 (Hopper) both ops raise ``NotImplementedError``. Use
``linear_fp8`` (deep_gemm WGMMA BlockScaled) on Hopper.
"""
from __future__ import annotations

import torch

from .._arch import sm_major


def _require_mxfp8_arch() -> None:
    if sm_major() not in (10, 12):
        raise NotImplementedError(
            "linear_mxfp8 / quantize_1x32_fp8 need sm_100/sm_103 (B200/B300) "
            "or sm_120/121 (RTX 5090 / RTX PRO 6000). On sm_90 (Hopper) use "
            "linear_fp8 + quantize_1x128_fp8 / quantize_128x128_fp8."
        )


# Backwards-compat alias (callers/tests may import the old name).
_require_sm120 = _require_mxfp8_arch


def linear_mxfp8(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    sx: torch.Tensor,
    sw: torch.Tensor,
) -> torch.Tensor:
    """Block-scaled MXFP8 (1×32 UE8M0) GEMM. sm_100/sm_103/sm_120.

    True OCP MXFP8: one E8M0 byte per 32 K-elements on both A and B.
    Finer than the 1×128 / 128×128 scheme — meaningfully tighter
    quantization at the cost of 4× more scale bytes.

    Args:
        x_fp8: float8_e4m3fn [M, K], K%128==0, N%128==0.
        w_fp8: float8_e4m3fn [N, K].
        sx, sw: opaque int32 UE8M0 scales from :func:`quantize_1x32_fp8`
            run on the SAME device arch (sm_120: K-major
            ``[pad(M,4), K/128]``; sm_100/103: 1-D Sm1xx atom layout).

    Returns:
        bfloat16 [M, N].

    Raises:
        NotImplementedError: on sm_90 (use ``linear_fp8`` there).
    """
    _require_mxfp8_arch()
    # sm_100/sm_103 (Blackwell datacenter) has a three-tier best-per-shape
    # dispatch — cuBLAS scaled_mm on decode / DSL on prefill+peak / C++
    # cascade for the narrow-N split-K decode + square cubic paths. Full
    # rationale in `_sm100_dispatch.py`.
    if sm_major() == 10:
        from . import _sm100_dispatch
        y = _sm100_dispatch.route(x_fp8, w_fp8, sx, sw)
        if y is not None:
            return y
    # sx / sw come out of quantize_1x32_fp8 in K-major byte order (PyTorch
    # strides (1, M_pad), which it labels non-contiguous). The CUTLASS kernel
    # reads `data_ptr()` raw bytes in K-major layout, so we must NOT call
    # `.contiguous()` on a K-major view — it would physically repack to
    # row-major and the kernel would read garbage (cos drops to ~0.5–0.87
    # across the Qwen3 grid; confirmed 2026-06-12 on RTX 5090). The ternary
    # preserves the K-major byte order from the quantizer while still
    # ensuring contiguity if the caller hands us a standard row-major scale.
    return torch.ops.fish_scales_ops.linear_mxfp8_raw(
        x_fp8.contiguous(),
        w_fp8.contiguous(),
        sx.contiguous() if sx.is_contiguous() else sx,
        sw.contiguous() if sw.is_contiguous() else sw,
    )


def quantize_1x32_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [..., K] → FP8 (E4M3) [..., K] + UE8M0 1×32 scales.

    True OCP MXFP8: one E8M0 byte per 32 elements along K. Supported on
    sm_100/sm_103 and sm_120/121.

    Returns:
        (x_fp8, sx) — x_fp8 has the input's shape; sx is an opaque int32
        scale tensor in the arch-native layout (sm_120: K-major
        ``[pad(M,4), K/128]``, 4 UE8M0 bytes per int32 word; sm_100/103:
        1-D ``[ceil(M/128)*128 * K/128]`` Sm1xxBlockScaledConfig atom
        layout). Ready for :func:`linear_mxfp8` on the same device arch.

    Constraints: K must be a multiple of 128.

    Raises:
        NotImplementedError: on sm_90 (this path needs MXFP8 hardware).
    """
    _require_mxfp8_arch()
    # Fused quantize + packed-scale write — single CUDA kernel, no FP32 scale
    # round-trip. Bit-exact with the legacy `quantize_1x32` + `repack_mxfp8_scales`
    # two-step path.
    return torch.ops.fish_scales_ops.quantize_1x32_packed(x.contiguous(), True)


def silu_chunk_mul_quantize_1x32_fp8(gu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SwiGLU prologue + MXFP8 quantize. sm_100/sm_103/sm_120.

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
    _require_mxfp8_arch()
    return torch.ops.fish_scales_ops.silu_chunk_mul_quantize_1x32(gu.contiguous(), True)
