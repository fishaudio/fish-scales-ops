"""Sweep forced MXFP8 tile configs at one cubic shape, vs the cascade default.

FSO_FORCE_TILE is cached per-process, so each cell runs in a subprocess.
Reports best-shot CUDA-Graph timing (min over many isolated replays).

Usage:
    PYTHONPATH=python python bench/gemm/python/tile_sweep_cubic.py --m 4096
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

# (TileM, TileN, NumStages) — must match instantiations in dispatch.cuh.
CONFIGS = [
    (16, 128, 4),
    (16,  64, 4),
    (32, 128, 4),
    (32,  64, 4),
    (64, 128, 2),
    (64,  64, 4),
    (96, 128, 2),
    (128, 128, 2),
    (128,  64, 2),
    (160, 128, 2),
]

# Device-specific FP8 dense whitepaper peaks (must match bench_cubic.py).
PEAK_TF_BY_DEVICE = {
    "RTX PRO 6000 Blackwell": 1007.0,
    "RTX Blackwell 170 SM GPU":               838.0,
}
PEAK_TF_DEFAULT = 838.0


INNER = """
import torch, fish_scales_ops as fso
M = {M}
torch.manual_seed(0)
x = torch.randn(M, M, dtype=torch.bfloat16, device="cuda") * 0.1
w = torch.randn(M, M, dtype=torch.bfloat16, device="cuda") / (M ** 0.5)
xq, sx = fso.gemm.quantize_1x32_fp8(x)
wq, sw = fso.gemm.quantize_1x32_fp8(w)
fn = lambda: fso.gemm.linear_mxfp8(xq, wq, sx, sw)
for _ in range(15): fn()
torch.cuda.synchronize()
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(15): fn()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=s): _ = fn()
samples = []
for _ in range(200):
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
    samples.append(e0.elapsed_time(e1))
print("RESULT us=" + format(min(samples) * 1000.0, ".3f"))
"""


def pick_peak() -> float:
    import torch
    name = torch.cuda.get_device_name()
    for key, peak in PEAK_TF_BY_DEVICE.items():
        if key in name:
            return peak
    return PEAK_TF_DEFAULT


def run(M: int, env_extra: dict, peak: float):
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo}/python:{env.get('PYTHONPATH','')}"
    env.update(env_extra)
    res = subprocess.run([sys.executable, "-c", INNER.format(M=M)],
                         capture_output=True, env=env, text=True, timeout=600)
    if res.returncode != 0:
        err = res.stderr.strip().splitlines()
        return None, None, None, err[-1][:60] if err else f"rc={res.returncode}"
    line = [l for l in res.stdout.splitlines() if l.startswith("RESULT")]
    if not line:
        return None, None, None, "no RESULT"
    us = float(line[0].split("us=")[1])
    tf = 2.0 * M ** 3 / us / 1e6
    return us, tf, tf / peak * 100, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4096, help="cubic shape M=N=K")
    args = ap.parse_args()

    import torch
    peak = pick_peak()
    print(f"# Device: {torch.cuda.get_device_name()}  M=N=K={args.m}")
    print(f"# Peak: {peak} TF\n")
    print(f"{'Config':>26s}  {'µs':>8s}   {'TF':>6s}   {'MFU':>6s}")
    print("-" * 56)

    cascade_us, cascade_tf, cascade_mfu, err = run(args.m, {"FSO_FORCE_TILE": ""}, peak)
    if cascade_us is None:
        print(f"  cascade FAIL: {err}")
        return
    print(f"  {'cascade (no force)':26s}  {cascade_us:>8.2f}   {cascade_tf:>6.1f}   {cascade_mfu:>5.1f}%")

    results = []
    for TM, TN, ST in CONFIGS:
        us, tf, mfu, err = run(args.m, {"FSO_FORCE_TILE": f"{TM},{TN},{ST}"}, peak)
        label = f"forced ({TM},{TN},{ST})"
        if us is None:
            print(f"  {label:26s}  FAIL  {err}")
            continue
        mark = "  ←" if us < cascade_us else ""
        print(f"  {label:26s}  {us:>8.2f}   {tf:>6.1f}   {mfu:>5.1f}%{mark}")
        results.append((us, tf, mfu, TM, TN, ST))

    if results:
        best = min(results, key=lambda r: r[0])
        print()
        if best[0] < cascade_us:
            print(f"# Better than cascade: ({best[3]},{best[4]},{best[5]}) "
                  f"→ {best[1]:.1f} TF, {best[2]:.1f}% MFU "
                  f"(vs cascade {cascade_us:.1f} µs)")
        else:
            print(f"# Cascade is best at {cascade_us:.1f} µs")


if __name__ == "__main__":
    main()
