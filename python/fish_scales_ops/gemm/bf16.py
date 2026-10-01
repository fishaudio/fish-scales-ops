"""BF16-in-BF16-out convenience wrapper around the FP8 1×128 path.

Internal-quantize variant — use only when both operands change every
step. For inference where the weight is fixed, pre-quantize with
:func:`fish_scales_ops.compat.quantize_128x128_fp8` and call ``linear_fp8``
directly.
"""
from __future__ import annotations

import torch


def linear_bf16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y = x @ w.T via the FP8 1×128 path with internal quantize on both.

    bf16 in and out, block-FP8 precision inside. sm_90 and sm_120 / sm_121
    run the block-FP8 runner path. On sm_100 / sm_103 the op is composed from
    the public ops (the UE8M0 quantizers, an unchecked weight-scale repack and
    the block-FP8 GEMM); nothing in it synchronizes the device, so it can be
    captured into a CUDA graph after one eager call, as on the other
    architectures.

    Args:
        x: bf16 [M, K], contiguous.
        w: bf16 [N, K], contiguous.

    Returns:
        bf16 [M, N].
    """
    return torch.ops.fish_scales_ops.linear_bf16(x.contiguous(), w.contiguous())
