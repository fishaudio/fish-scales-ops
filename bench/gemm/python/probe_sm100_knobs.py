"""Probe SM100 scheduler knobs: explicit split-K (StreamK variant) and
raster swizzle (default scheduler). Best-shot cudagraph protocol, one
subprocess per (shape, knob) cell — env vars are read once per process.

Usage:
    PYTHONPATH=python python bench/gemm/python/probe_sm100_knobs.py
"""
from __future__ import annotations
import os, subprocess, sys

INNER = """
import torch, fish_scales_ops as fso
M,N,K = {M},{N},{K}
torch.manual_seed(M*1009+N*17+K)
x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.5)
xq, sx = fso.gemm.quantize_1x32_fp8(x)
wq, sw = fso.gemm.quantize_1x32_fp8(w)
import torch.nn.functional as F
y_ref = F.linear(x, w)
fn = lambda: fso.gemm.linear_mxfp8(xq, wq, sx, sw)
y = fn(); torch.cuda.synchronize()
cos = F.cosine_similarity(y.double().flatten(), y_ref.double().flatten(), dim=0).item()
for _ in range(15): fn()
torch.cuda.synchronize()
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(15): fn()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=s): _ = fn()
samples = []
for _ in range(120):
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
    samples.append(e0.elapsed_time(e1))
print("RESULT us=" + format(min(samples) * 1000.0, ".3f") + " cos=" + format(cos, ".5f"))
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
        return None, None, (err[-1][:70] if err else f"rc={res.returncode}")
    line = [l for l in res.stdout.splitlines() if l.startswith("RESULT")]
    if not line:
        return None, None, "no RESULT"
    us = float(line[0].split("us=")[1].split()[0])
    cos = float(line[0].split("cos=")[1])
    return us, cos, ""


def main():
    print("== A. explicit split-K on down-class decode (forced 128,128 StreamK K128/K256) ==")
    for M in (1, 16, 128, 512):
        for st in (7, 8):
            for ks in (0, 2, 4, 8, 19):
                env = {"FSO_FORCE_TILE": f"128,128,{st}"}
                if ks: env["FSO_FORCE_KSPLIT"] = str(ks)
                us, cos, err = run(M, 2560, 9728, env)
                tag = f"st={st} ks={ks or 'heur'}"
                if us is None:
                    print(f"  down M={M:>4} {tag:>14}: FAIL {err}")
                else:
                    print(f"  down M={M:>4} {tag:>14}: {us:>7.2f} us  cos={cos:.4f}")

    print("\n== B. raster swizzle on peak cubic (forced 256,256,2 = c(2,1) K128) ==")
    for S in (4096, 8192):
        for sw in (0, 1, 2, 4, 8):
            env = {"FSO_FORCE_TILE": "256,256,2"}
            if sw: env["FSO_FORCE_SWIZZLE"] = str(sw)
            us, cos, err = run(S, S, S, env)
            if us is None:
                print(f"  cubic {S} swizzle={sw}: FAIL {err}")
            else:
                tf = 2.0 * S * S * S / us / 1e6
                print(f"  cubic {S} swizzle={sw}: {us:>8.2f} us  {tf:>7.1f} TF  cos={cos:.4f}")

    print("\n== C. swizzle on mid-band workhorse (256,128,4) ==")
    for (M, N, K) in ((1024, 6144, 2560), (2048, 9728, 2560)):
        for sw in (0, 2, 4, 8):
            env = {"FSO_FORCE_TILE": "256,128,4"}
            if sw: env["FSO_FORCE_SWIZZLE"] = str(sw)
            us, cos, err = run(M, N, K, env)
            if us is None:
                print(f"  ({M},{N},{K}) swizzle={sw}: FAIL {err}")
            else:
                print(f"  ({M},{N},{K}) swizzle={sw}: {us:>7.2f} us  cos={cos:.4f}")


if __name__ == "__main__":
    main()
