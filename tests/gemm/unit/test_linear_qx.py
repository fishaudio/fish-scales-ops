#!/usr/bin/env python3
"""linear_qx (bf16 activation quantized inside the op, pre-quantized block-FP8
weight) against the two-step form, quantize_1x128_fp8 + linear_fp8, which must
give the bit-identical result on sm_90 and sm_120/121. On sm_100/103 the op is
not available and must say so."""
import sys

import torch
import fish_scales_ops as fso

SHAPES = ((1, 2048, 2048), (16, 4096, 2048), (128, 2048, 4096), (1024, 6144, 2048))


def main():
    if not torch.cuda.is_available():
        print("SKIP: test_linear_qx needs a CUDA device")
        return 0
    major = torch.cuda.get_device_capability()[0]
    torch.manual_seed(0)
    x = torch.randn(4, 256, device="cuda").bfloat16()
    if major == 10:
        wq, sw = fso.compat.quantize_128x128_fp8(torch.randn(256, 256, device="cuda").bfloat16())
        try:
            fso.compat.linear_qx(x, wq, sw)
        except RuntimeError as e:
            assert "sm_100" in str(e), e
            print("linear_qx on sm_10x: refused with a pointer to quantize_1x128_fp8_packed + linear_fp8  OK")
            return 0
        raise AssertionError("linear_qx should refuse sm_100/sm_103")
    if major not in (9, 12):
        print(f"SKIP: test_linear_qx covers sm_90, sm_100/103 and sm_120/121 (device is sm_{major}x)")
        return 0
    for M, N, K in SHAPES:
        x = (torch.randn(M, K, device="cuda") * 0.1).bfloat16()
        w = (torch.randn(N, K, device="cuda") / K ** 0.5).bfloat16()
        wq, sw = fso.compat.quantize_128x128_fp8(w)
        y_qx = fso.compat.linear_qx(x, wq, sw)
        xq, sx = fso.compat.quantize_1x128_fp8(x)
        y_two = fso.compat.linear_fp8(xq, wq, sx, sw)
        assert torch.isfinite(y_qx).all(), f"M={M} N={N} K={K}: non-finite output"
        assert torch.equal(y_qx, y_two), f"M={M} N={N} K={K}: linear_qx differs from quantize + linear_fp8"
        print(f"  M={M:5d} N={N} K={K}: bit-identical to quantize_1x128_fp8 + linear_fp8  OK")
    print("linear_qx: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
