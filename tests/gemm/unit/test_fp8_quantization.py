"""Numerical regression tests for the public FP8 quantizers."""

from __future__ import annotations

import torch

import fish_scales_ops as fso


def test_quantize_128x128_uses_amax_from_every_row() -> None:
    assert torch.cuda.is_available(), "CUDA required"

    weight = torch.ones((128, 128), dtype=torch.bfloat16, device="cuda")
    weight[-1, -1] = 16

    weight_fp8, scales = fso.gemm.quantize_128x128_fp8(weight)

    expected_scale = torch.full_like(scales, 16.0 / 448.0)
    torch.testing.assert_close(scales, expected_scale, rtol=1e-6, atol=0)

    dequantized_outlier = weight_fp8[-1, -1].float() * scales[0, 0]
    torch.testing.assert_close(
        dequantized_outlier,
        weight[-1, -1].float(),
        rtol=1e-6,
        atol=0,
    )
