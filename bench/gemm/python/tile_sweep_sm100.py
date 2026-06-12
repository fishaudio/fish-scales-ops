"""SM100/SM103 MXFP8 tile sweep: forced tiles vs cascade over the Qwen3 grid.

Same subprocess-per-cell + CUDA-Graph best-shot protocol as tile_sweep_mnk.py,
but with the sm_100 tile table (TileM/TileN ∈ {128,256}; third field selects
1SM/2SM — see arch/sm100/mxfp8/dispatch.cuh).

Usage:
    PYTHONPATH=python python bench/gemm/python/tile_sweep_sm100.py [--out sweep.jsonl]
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys

CONFIGS = [
    # (TM, TN, ST) — ST encodes variant, see dispatch.cuh:
    # 1=1SM K128, 2=2SM c21 K128, 3=1SM K256, 4=2SM c21 K256,
    # 5=2SM c22 K128, 6=2SM c22 K256, 7=1SM K128 StreamK, 8=1SM K256 StreamK
    (128, 128, 1), (128, 256, 1),
    (128, 128, 3), (128, 256, 3),
    (128, 128, 7), (128, 128, 8),
    (256, 128, 2), (256, 256, 2),
    (256, 128, 4), (256, 256, 4),
    (256, 256, 5), (256, 256, 6),
]

SHAPES = [
    ("wqkv",    6144, 2560),
    ("wo",      2560, 4096),
    ("gate_up", 19456, 2560),
    ("gate",    9728, 2560),
    ("down",    2560, 9728),
]
MS = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096]
CUBICS = [4096, 8192]

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
for _ in range(120):
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
        return None, (err[-1][:70] if err else f"rc={res.returncode}")
    line = [l for l in res.stdout.splitlines() if l.startswith("RESULT")]
    if not line:
        return None, "no RESULT"
    return float(line[0].split("us=")[1]), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = open(args.out, "w") if args.out else None

    cells = [(tag, M, N, K) for tag, N, K in SHAPES for M in MS]
    cells += [("cubic", s, s, s) for s in CUBICS]

    for tag, M, N, K in cells:
        row = {"tag": tag, "M": M, "N": N, "K": K}
        cus, err = run(M, N, K, {"FSO_FORCE_TILE": ""})
        row["cascade"] = cus if cus else f"ERR:{err}"
        best = (cus or 1e18, "cascade")
        for TM, TN, ST in CONFIGS:
            us, err = run(M, N, K, {"FSO_FORCE_TILE": f"{TM},{TN},{ST}"})
            key = f"{TM}x{TN}s{ST}"
            row[key] = us if us else f"ERR:{err}"
            if us and us < best[0]:
                best = (us, key)
        row["best"] = best[1]
        row["best_us"] = best[0]
        if cus and best[0] < cus * 0.99:
            row["cascade_loss_pct"] = round(100 * (cus - best[0]) / cus, 1)
        print(json.dumps(row), flush=True)
        if out:
            out.write(json.dumps(row) + "\n"); out.flush()
    if out:
        out.close()


if __name__ == "__main__":
    main()
