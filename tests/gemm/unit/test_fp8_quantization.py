"""Numerical regression tests for the public FP8 quantizers.

The 128x128 weight quantizer must take the block's amax from every row (a
2026-09-01 fix had it skip the last row). The check is made in both scale
formats the quantizer produces, because the default depends on the arch:
FP32 scales (the sm_90 default) must equal amax / 448 exactly, and UE8M0
scales (the default on sm_100 / sm_120, `use_ue8m0=None` -> `sm_major() >= 10`)
must be the power of two at or above that value. Before 2026-09-29 the test
hard-coded the FP32 expectation and failed on every Blackwell device.

The 1x128 activation quantizer with FP32 scales (`use_ue8m0=False`, the sm_90
GEMM's input) must reproduce its arithmetic contract bit for bit: per row and
128-element group, amax = max |x| floored at 1e-10 (at 1e-10f when K is a
multiple of 512, at the BF16 value nearest to 1e-10 otherwise, the floors of the
two kernels it replaced in 2026-10), qs = 448 / amax, scale = 1 / qs, and
x * qs rounded to the nearest E4M3 value; scales are K-major,
[kb * align(M, 4) + m], with the padding rows' scales zeroed when K is a
multiple of 512.
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

    weight_fp8, scales = fso.compat.quantize_128x128_fp8(weight, use_ue8m0=False)
    expected_scale = torch.full_like(scales, 16.0 / 448.0)
    torch.testing.assert_close(scales, expected_scale, rtol=1e-6, atol=0)

    dequantized_outlier = weight_fp8[-1, -1].float() * scales[0, 0]
    torch.testing.assert_close(dequantized_outlier, weight[-1, -1].float(), rtol=1e-6, atol=0)


def test_quantize_128x128_ue8m0_rounds_amax_scale_up() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    weight = _weight()

    weight_fp8, scales = fso.compat.quantize_128x128_fp8(weight, use_ue8m0=True)
    expected = 2.0 ** math.ceil(math.log2(16.0 / 448.0))  # 0.0625: the power of two at or above amax / 448
    torch.testing.assert_close(scales.float(), torch.full_like(scales.float(), expected), rtol=0, atol=0)

    dequantized_outlier = weight_fp8[-1, -1].float() * scales.float()[0, 0]
    torch.testing.assert_close(dequantized_outlier, weight[-1, -1].float(), rtol=1e-6, atol=0)


def _reference_1x128(x: torch.Tensor, floor: float):
    """The FP32-scale 1x128 quantize in torch: float32 IEEE arithmetic on the same values as the kernel."""
    m, k = x.shape
    xf = x.float().view(m, k // 128, 128)
    amax = xf.abs().amax(dim=2).clamp_min(floor)
    # Tensor divisions: `scalar / tensor` is reciprocal(tensor) * scalar in torch, which rounds twice.
    qs = torch.full_like(amax, 448.0) / amax
    scales = torch.ones_like(qs) / qs
    q = (xf * qs.unsqueeze(2)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(m, k)
    return q, scales  # scales [M, K / 128]


def test_quantize_1x128_fp32_scales_match_reference() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    bf16_floor = float(torch.tensor(1e-10, dtype=torch.bfloat16).float())
    for k in (128, 384, 512, 640, 2560, 4224, 9728):
        floor = 1e-10 if k % 512 == 0 else bf16_floor
        for m in (1, 3, 8, 33, 130):
            g = torch.Generator(device="cuda").manual_seed(m * 131 + k)
            x = torch.randn(m, k, device="cuda", generator=g) * 0.1
            groups = x.view(m, k // 128, 128)
            groups[:, ::3, :] = 0.0                       # all-zero groups take the floor
            groups[:, 1::4, :] *= 1e-12                   # groups whose amax is below the floor
            x = x.to(torch.bfloat16)
            q, s = fso.compat.quantize_1x128_fp8(x, use_ue8m0=False)
            q_ref, s_ref = _reference_1x128(x, floor)
            m_pad = (m + 3) // 4 * 4
            s_kmajor = s.reshape(-1)[: (k // 128) * m_pad].view(k // 128, m_pad)
            assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)), (m, k)
            assert torch.equal(s_kmajor[:, :m], s_ref.t()), (m, k)
            if k % 512 == 0 and m_pad > m:
                assert torch.count_nonzero(s_kmajor[:, m:]) == 0, (m, k)


def main() -> int:
    test_quantize_128x128_uses_amax_from_every_row()
    test_quantize_128x128_ue8m0_rounds_amax_scale_up()
    test_quantize_1x128_fp32_scales_match_reference()
    print("fp8 quantization: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
