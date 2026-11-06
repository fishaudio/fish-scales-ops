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

The same contract is checked at token counts where the launcher caps the grid at
one resident wave, so that the threads of the lanes kernel walk several groups
(its grid-stride loop), and on the kernel's two other callers: the contract
quantizer of `fso.dense.quantize_activation` (sm_90: the BF16 floor for every
K, padding rows' scales zeroed) and the quantize that `fso.dense.linear` runs
inside the op on a bf16 x, which must give the bytes of the pre-quantized call
(2026-11-05, when the kernel's loads moved ahead of its index division).
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


def _activation(m: int, k: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(m * 131 + k)
    x = torch.randn(m, k, device="cuda", generator=g) * 0.1
    groups = x.view(m, k // 128, 128)
    groups[:, ::3, :] = 0.0
    groups[:, 1::4, :] *= 1e-12
    return x.to(torch.bfloat16)


def test_quantize_1x128_fp32_scales_match_reference_capped_grid() -> None:
    """Token counts at which the lanes kernel's grid is capped at one resident wave (at most six 256-thread CTAs per
    SM, 32 groups each), so that its threads walk several groups; K / 128 = 1, 5, 20 and 76 exercise the row and
    K-block split of the scale index."""
    assert torch.cuda.is_available(), "CUDA required"
    bf16_floor = float(torch.tensor(1e-10, dtype=torch.bfloat16).float())
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for k in (128, 640, 2560, 9728):
        m = (6 * sms * 32 * 128) // k * 3 // 2 + 3  # 1.5 waves of groups, M not a multiple of 4
        x = _activation(m, k)
        floor = 1e-10 if k % 512 == 0 else bf16_floor
        q, s = fso.compat.quantize_1x128_fp8(x, use_ue8m0=False)
        q_ref, s_ref = _reference_1x128(x, floor)
        m_pad = (m + 3) // 4 * 4
        s_kmajor = s.reshape(-1)[: (k // 128) * m_pad].view(k // 128, m_pad)
        assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)), (m, k)
        assert torch.equal(s_kmajor[:, :m], s_ref.t()), (m, k)
        if k % 512 == 0:
            assert torch.count_nonzero(s_kmajor[:, m:]) == 0, (m, k)


def test_quantize_activation_matches_reference() -> None:
    """sm_90: fso.dense.quantize_activation floors amax at the BF16 value of 1e-10 for every K and zeroes the padding
    rows' scales; x_scale is the M-fastest view (x_scale[m, kb]). Small M and capped-grid M."""
    assert torch.cuda.is_available(), "CUDA required"
    if torch.cuda.get_device_capability()[0] != 9:
        print("  quantize_activation reference check: sm_90 only, skipped")
        return
    bf16_floor = float(torch.tensor(1e-10, dtype=torch.bfloat16).float())
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for k in (128, 384, 640, 2560, 4224, 9728):
        for m in (1, 3, 8, 33, 130, (6 * sms * 32 * 128) // k * 3 // 2 + 1):
            x = _activation(m, k)
            q, s = fso.dense.quantize_activation(x)
            q_ref, s_ref = _reference_1x128(x, bf16_floor)
            assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)), (m, k)
            assert torch.equal(s[:m], s_ref), (m, k)
            assert torch.count_nonzero(s[m:]) == 0, (m, k)


def test_linear_internal_quantize_matches_prequantized_capped_grid() -> None:
    """sm_90: fso.dense.linear on a bf16 x quantizes it inside the op (the lanes kernel's linear_qx flavour); at token
    counts that cap its grid at one resident wave the result must still be the bytes of the pre-quantized call."""
    assert torch.cuda.is_available(), "CUDA required"
    if torch.cuda.get_device_capability()[0] != 9:
        print("  linear internal-quantize check: sm_90 only, skipped")
        return
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for n, k in ((2560, 9728), (1024, 2560)):
        m = (6 * sms * 32 * 128) // k * 3 // 2 + 2
        g = torch.Generator(device="cuda").manual_seed(n * 17 + k)
        w = (torch.randn(n, k, device="cuda", generator=g) / math.sqrt(k)).to(torch.bfloat16)
        wq, sw = fso.compat.quantize_128x128_fp8(w, use_ue8m0=False)
        dw = fso.dense.prepare_weight(wq, format="bsfp8", scale=sw)
        x = _activation(m, k)
        y = fso.dense.linear(x, dw)
        xq, sx = fso.dense.quantize_activation(x)
        y2 = fso.dense.linear(xq, dw, x_scale=sx)
        assert torch.equal(y.view(torch.int16), y2.view(torch.int16)), (m, n, k)


def main() -> int:
    test_quantize_128x128_uses_amax_from_every_row()
    test_quantize_128x128_ue8m0_rounds_amax_scale_up()
    test_quantize_1x128_fp32_scales_match_reference()
    test_quantize_1x128_fp32_scales_match_reference_capped_grid()
    test_quantize_activation_matches_reference()
    test_linear_internal_quantize_matches_prequantized_capped_grid()
    print("fp8 quantization: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
