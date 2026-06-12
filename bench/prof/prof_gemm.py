"""Minimal eager single-op driver for ncu profiling of FSO GEMM kernels.
Usage: prof_gemm.py {mxfp8|bsfp8} M N K [iters]
Pre-quantizes outside the loop; the timed region is just the linear op so ncu
captures the GEMM kernel (+ any streamk reduce) without graph/quantize noise.
"""
import sys, torch
import fish_scales_ops as fso

op = sys.argv[1]; M, N, K = (int(x) for x in sys.argv[2:5])
iters = int(sys.argv[5]) if len(sys.argv) > 5 else 30
sm = torch.cuda.get_device_capability(0)[0]
x = (torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1)
w = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.5))

if op == "mxfp8":
    xq, sxq = fso.gemm.quantize_1x32_fp8(x)
    wq, swq = fso.gemm.quantize_1x32_fp8(w)
    call = lambda: fso.gemm.linear_mxfp8(xq, wq, sxq, swq)
else:
    xq, sxq = fso.gemm.quantize_1x128_fp8(x, use_ue8m0=(sm >= 12))
    wq, swq = fso.gemm.quantize_128x128_fp8(w)
    if sm >= 12:
        sxqp = fso.gemm.repack_fp8_act_scales(sxq)
        swqp = fso.gemm.repack_fp8_wgt_scales(swq)
    else:
        sxqp, swqp = sxq, swq
    call = lambda: fso.gemm.linear_fp8(xq, wq, sxqp, swqp)

for _ in range(10):
    call()
torch.cuda.synchronize()
for _ in range(iters):
    call()
torch.cuda.synchronize()
print(f"done {op} M{M} N{N} K{K}")
