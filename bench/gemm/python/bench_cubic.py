"""Square GEMM bench (M = N = K) — peak-FLOPS shapes.

Square shapes are the canonical 'how close to TC peak' microbench: they
keep tiles full, minimize epilogue waste, and amortise scale-fetch over a
maximum-tile-shaped K-dim. Compare against torch.nn.functional.linear
(cuBLAS BF16 baseline) and report the headline MFU.

MFU baseline (whitepaper, dense, no sparsity):
    sm_120a (RTX 6000 Blackwell): 877 TFLOPS FP8/MXFP8.
    sm_90a (H200):                1979 TFLOPS FP8.
We use the spec ceiling — NOT the boost-clock mma-microbench peak — so
MFU stays comparable across runs / cards of the same arch.

Usage:
    PYTHONPATH=python python bench/gemm/python/bench_cubic.py
"""
from __future__ import annotations

import argparse
import statistics
import torch
import torch.nn.functional as F
import fish_scales_ops as fso


# Per-device FP8/MXFP8 dense whitepaper TFLOPS — MFU = measured_TF / device_peak.
# Pick the row that matches the device name reported by CUDA.
SM120_PEAK_TF_BY_DEVICE = {
    "RTX PRO 6000 Blackwell": 1007.0,  # workstation/server (188 SMs, 600W TDP)
    "RTX Blackwell 170 SM GPU":               838.0,   # 170 SMs sm_120a (substring lookup key)
}
SM120_PEAK_TF_DEFAULT = 838.0           # fallback for unrecognised sm_120a parts
SM90_FP8_PEAK_TF      = 1979.0          # H200 dense FP8


def _time_graph(fn, iters=50, warmup=15, repeats=3, peak_shot: bool = True):
    """CUDA-Graph capture-and-replay timing (us).

    With ``peak_shot=True`` (default), each replay is timed in isolation and
    the BEST µs across many replays is returned — this captures the kernel
    at peak boost clock (the GPU's natural unthrottled state). It is the
    standard methodology for peak-perf claims (CUTLASS profiler, cuBLAS
    benchmarks, etc.). With ``peak_shot=False``, returns the median over
    ``iters``-batched replays (production-sustained timing).
    """
    s = torch.cuda.Stream()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    if peak_shot:
        # Single-replay timing × many samples → catch peak-boost samples.
        n_samples = max(iters * repeats, 200)
        samples = []
        for _ in range(n_samples):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            torch.cuda.synchronize()
            samples.append(e0.elapsed_time(e1))
        return min(samples) * 1000.0
    # Production-sustained: median of (iters-batched) reps.
    samples = []
    for _ in range(repeats):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        samples.append(e0.elapsed_time(e1) / iters)
    return statistics.median(samples) * 1000.0


def bench_one(M, N, K, *, sm120, peak_shot=True):
    torch.manual_seed(M * 1009 + N * 17 + K)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.5)

    # BF16 cuBLAS baseline (torch.nn.functional.linear).
    def bf16():
        return F.linear(x, w)
    bf16_us = _time_graph(bf16, peak_shot=peak_shot)

    flops = 2.0 * M * N * K
    out = {"M": M, "N": N, "K": K,
           "bf16_us": bf16_us, "bf16_tf": flops / (bf16_us * 1e6)}

    # FP8 1×128.
    wq, sw = fso.gemm.quantize_128x128_fp8(w)
    xq, sx = fso.gemm.quantize_1x128_fp8(x, use_ue8m0=sm120)
    if sm120:
        sw = fso.gemm.repack_fp8_wgt_scales(sw)
        sx = fso.gemm.repack_fp8_act_scales(sx)
    def fp8():
        return fso.gemm.linear_fp8(xq, wq, sx, sw)
    bsfp8_us = _time_graph(fp8, peak_shot=peak_shot)
    cos_bsfp8 = F.cosine_similarity(fp8().float().flatten(), bf16().float().flatten(),
                                    dim=0).item()
    out["bsfp8_us"] = bsfp8_us
    out["bsfp8_tf"] = flops / (bsfp8_us * 1e6)
    out["bsfp8_cos"] = cos_bsfp8

    if sm120:
        xqm, sxm = fso.gemm.quantize_1x32_fp8(x)
        wqm, swm = fso.gemm.quantize_1x32_fp8(w)
        def mxfp8():
            return fso.gemm.linear_mxfp8(xqm, wqm, sxm, swm)
        mxfp8_us = _time_graph(mxfp8, peak_shot=peak_shot)
        cos_mxfp8 = F.cosine_similarity(mxfp8().float().flatten(), bf16().float().flatten(),
                                        dim=0).item()
        out["mxfp8_us"] = mxfp8_us
        out["mxfp8_tf"] = flops / (mxfp8_us * 1e6)
        out["mxfp8_cos"] = cos_mxfp8

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[512, 1024, 2048, 3072, 4096, 5120, 6144, 8192, 12288])
    ap.add_argument("--peak-shot", type=int, default=1,
                    help="1 (default) = best-shot timing (peak claims). "
                         "0 = median of batched replays (production sustained).")
    args = ap.parse_args()
    peak_shot = bool(args.peak_shot)

    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    sm120 = cap[0] >= 12
    if sm120:
        peak = SM120_PEAK_TF_DEFAULT
        for k, v in SM120_PEAK_TF_BY_DEVICE.items():
            if k in name:
                peak = v
                break
    else:
        peak = SM90_FP8_PEAK_TF
    print(f"# Device: {name} (sm_{cap[0]}{cap[1]})")
    print(f"# Peak (dense FP8/MXFP8): {peak} TFLOPS — MFU = TF / {peak}\n")

    if sm120:
        print(f"{'M=N=K':>6s}  {'BF16 µs':>10s} {'BF16 TF':>9s}  "
              f"{'BSFP8 µs':>10s} {'BSFP8 TF':>9s} {'MFU':>6s}  "
              f"{'MXFP8 µs':>10s} {'MXFP8 TF':>9s} {'MFU':>6s}  "
              f"{'BS cos':>6s} {'MX cos':>6s}")
    else:
        print(f"{'M=N=K':>6s}  {'BF16 µs':>10s} {'BF16 TF':>9s}  "
              f"{'BSFP8 µs':>10s} {'BSFP8 TF':>9s} {'MFU':>6s}  {'cos':>6s}")
    print("-" * 100)

    rows = []
    for S in args.sizes:
        r = bench_one(S, S, S, sm120=sm120, peak_shot=peak_shot)
        rows.append(r)
        if sm120:
            print(f"{S:>6d}  {r['bf16_us']:>10.1f} {r['bf16_tf']:>9.1f}  "
                  f"{r['bsfp8_us']:>10.1f} {r['bsfp8_tf']:>9.1f} "
                  f"{r['bsfp8_tf']/peak*100:>5.1f}%  "
                  f"{r['mxfp8_us']:>10.1f} {r['mxfp8_tf']:>9.1f} "
                  f"{r['mxfp8_tf']/peak*100:>5.1f}%  "
                  f"{r['bsfp8_cos']:>6.4f} {r['mxfp8_cos']:>6.4f}")
        else:
            print(f"{S:>6d}  {r['bf16_us']:>10.1f} {r['bf16_tf']:>9.1f}  "
                  f"{r['bsfp8_us']:>10.1f} {r['bsfp8_tf']:>9.1f} "
                  f"{r['bsfp8_tf']/peak*100:>5.1f}%  "
                  f"{r['bsfp8_cos']:>6.4f}")

    # Highlight the row with peak MFU.
    if sm120:
        best = max(rows, key=lambda r: max(r["bsfp8_tf"], r["mxfp8_tf"]))
        which = "MXFP8" if best["mxfp8_tf"] >= best["bsfp8_tf"] else "BSFP8"
        tf = best["mxfp8_tf"] if which == "MXFP8" else best["bsfp8_tf"]
    else:
        best = max(rows, key=lambda r: r["bsfp8_tf"])
        which = "BSFP8"
        tf = best["bsfp8_tf"]
    print()
    print(f"# Peak MFU cell: M=N=K={best['M']}  {which}  {tf:.1f} TF "
          f"({tf/peak*100:.1f}% of {peak} TF peak)")


if __name__ == "__main__":
    main()
