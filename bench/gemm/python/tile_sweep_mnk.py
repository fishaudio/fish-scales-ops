"""Sweep forced MXFP8 tile configs at an arbitrary (M,N,K) shape vs cascade.

FSO_FORCE_TILE is cached per-process, so each cell runs in a subprocess.
Reports best-shot CUDA-Graph timing (min over many isolated replays).

Usage:
    PYTHONPATH=python python bench/gemm/python/tile_sweep_mnk.py --m 512 --n 19456 --k 2560
"""
from __future__ import annotations
import argparse, os, subprocess, sys

# (TileM, TileN, NumStages) — must match instantiations in dispatch.cuh.
CONFIGS = [
    (32, 128, 4), (32, 64, 4),
    (64, 128, 2), (64, 64, 4),
    (96, 128, 2),
    (128, 128, 2), (128, 64, 2),
    (160, 128, 2),
]

INNER = """
import torch, fish_scales_ops as fso
M,N,K = {M},{N},{K}
torch.manual_seed(M*1009+N*17+K)
x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.5)
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


def run(M, N, K, env_extra):
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo}/python:{env.get('PYTHONPATH','')}"
    env.update(env_extra)
    res = subprocess.run([sys.executable, "-c", INNER.format(M=M, N=N, K=K)],
                         capture_output=True, env=env, text=True, timeout=600)
    if res.returncode != 0:
        err = res.stderr.strip().splitlines()
        return None, None, err[-1][:70] if err else f"rc={res.returncode}"
    line = [l for l in res.stdout.splitlines() if l.startswith("RESULT")]
    if not line:
        return None, None, "no RESULT"
    us = float(line[0].split("us=")[1])
    tf = 2.0 * M * N * K / us / 1e6
    return us, tf, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    args = ap.parse_args()
    import torch
    print(f"# {torch.cuda.get_device_name()}  M={args.m} N={args.n} K={args.k}")
    print(f"{'Config':>22s}  {'us':>8s}  {'TF':>7s}")
    print("-" * 42)
    cus, ctf, err = run(args.m, args.n, args.k, {"FSO_FORCE_TILE": ""})
    if cus is None:
        print(f"  cascade FAIL: {err}"); return
    print(f"  {'cascade':20s}  {cus:>8.2f}  {ctf:>7.1f}")
    results = []
    for TM, TN, ST in CONFIGS:
        us, tf, err = run(args.m, args.n, args.k, {"FSO_FORCE_TILE": f"{TM},{TN},{ST}"})
        if us is None:
            print(f"  forced ({TM},{TN},{ST}) FAIL {err}"); continue
        mark = "  <-- beats cascade" if us < cus * 0.999 else ""
        print(f"  ({TM},{TN},{ST}){'':10s}"[:22] + f"  {us:>8.2f}  {tf:>7.1f}{mark}")
        results.append((us, TM, TN, ST))
    best = min(results, key=lambda r: r[0])
    print()
    if best[0] < cus * 0.999:
        print(f"# BEST forced ({best[1]},{best[2]},{best[3]}) {best[0]:.2f}us vs cascade {cus:.2f}us "
              f"({100*(cus-best[0])/cus:+.1f}%)")
    else:
        print(f"# cascade already best ({cus:.2f}us)")


if __name__ == "__main__":
    main()
