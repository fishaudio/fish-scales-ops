"""Numerical regression tests for the public FP8 quantizers.

The 128x128 weight quantizer must take the block's amax from every row (a
2026-09-01 fix had it skip the last row). The check is made in both scale
formats the quantizer produces, because the default depends on the arch:
FP32 scales (the sm_90 default) must equal amax / 448 exactly, and UE8M0
scales (the default on sm_100 / sm_120, `use_ue8m0=None` -> `sm_major() >= 10`)
must be the power of two at or above that value. Before 2026-09-29 the test
hard-coded the FP32 expectation and failed on every Blackwell device.
"""

from __future__ import annotations

import math

import torch

import fish_scales_ops as fso


def _weight() -> torch.Tensor:
    weight = torch.ones((128, 128), dtype=torch.bfloat16, device="cuda")
    weight[-1, -1] = 16
    return weight


def test_quantize_128x128_uses_amax_from_every_row() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    weight = _weight()

    weight_fp8, scales = fso.gemm.quantize_128x128_fp8(weight, use_ue8m0=False)
    expected_scale = torch.full_like(scales, 16.0 / 448.0)
    torch.testing.assert_close(scales, expected_scale, rtol=1e-6, atol=0)

    dequantized_outlier = weight_fp8[-1, -1].float() * scales[0, 0]
    torch.testing.assert_close(dequantized_outlier, weight[-1, -1].float(), rtol=1e-6, atol=0)


def test_quantize_128x128_ue8m0_rounds_amax_scale_up() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    weight = _weight()

    weight_fp8, scales = fso.gemm.quantize_128x128_fp8(weight, use_ue8m0=True)
    expected = 2.0 ** math.ceil(math.log2(16.0 / 448.0))  # 0.0625: the power of two at or above amax / 448
    torch.testing.assert_close(scales.float(), torch.full_like(scales.float(), expected), rtol=0, atol=0)

    dequantized_outlier = weight_fp8[-1, -1].float() * scales.float()[0, 0]
    torch.testing.assert_close(dequantized_outlier, weight[-1, -1].float(), rtol=1e-6, atol=0)


def main() -> int:
    test_quantize_128x128_uses_amax_from_every_row()
    test_quantize_128x128_ue8m0_rounds_amax_scale_up()
    print("fp8 quantization: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
