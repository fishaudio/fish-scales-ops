"""Sweep forced DeepGEMM (bm,bn,ns) configs vs get_best_gemm_config picker (sm_90).
FSO_FORCE_DG is read per-process, so each cell runs in a subprocess.
Usage: python dg_config_sweep.py --m 8192 --n 8192 --k 8192
"""
import argparse, os, subprocess, sys

CONFIGS = [None, "128,128,4", "128,128,5", "128,128,6", "128,112,5", "128,112,6",
           "128,96,5", "128,96,6", "128,64,6", "128,64,8", "64,128,5", "64,128,6"]

INNER = """
import torch, fish_scales_ops as fso
M,N,K={M},{N},{K}
torch.manual_seed(M*1009+N*17+K)
x=torch.randn(M,K,dtype=torch.bfloat16,device='cuda')*0.1
w=torch.randn(N,K,dtype=torch.bfloat16,device='cuda')/(K**0.5)
xq,sx=fso.gemm.quantize_1x128_fp8(x,use_ue8m0=False); wq,sw=fso.gemm.quantize_128x128_fp8(w)
ybf=(x.float()@w.float().t())
fn=lambda: fso.gemm.linear_fp8(xq,wq,sx,sw)
y=fn()
cos=torch.nn.functional.cosine_similarity(y.float().flatten(),ybf.flatten(),dim=0).item()
for _ in range(15): fn()
torch.cuda.synchronize()
s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(15): fn()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g,stream=s): _=fn()
sm=[]
for _ in range(200):
    e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True)
    e0.record();g.replay();e1.record();torch.cuda.synchronize();sm.append(e0.elapsed_time(e1))
print('RESULT us=%.3f cos=%.4f'%(min(sm)*1000.0,cos))
"""

def run(M, N, K, cfg):
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    env = os.environ.copy(); env['PYTHONPATH'] = f"{repo}/python:" + env.get('PYTHONPATH', '')
    if cfg: env['FSO_FORCE_DG'] = cfg
    else: env.pop('FSO_FORCE_DG', None)
    r = subprocess.run([sys.executable, '-c', INNER.format(M=M, N=N, K=K)],
                       capture_output=True, env=env, text=True, timeout=600)
    if r.returncode != 0:
        e = r.stderr.strip().splitlines(); return None, None, (e[-1][:60] if e else 'rc')
    l = [x for x in r.stdout.splitlines() if x.startswith('RESULT')]
    if not l: return None, None, 'noRESULT'
    us = float(l[0].split('us=')[1].split()[0]); cos = float(l[0].split('cos=')[1])
    return us, cos, ''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, required=True)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--k', type=int, required=True)
    a = ap.parse_args()
    print(f"# M={a.m} N={a.n} K={a.k}"); base = None
    for cfg in CONFIGS:
        us, cos, err = run(a.m, a.n, a.k, cfg); lab = cfg or 'PICKER'
        if us is None: print(f"  {lab:14s} FAIL {err}"); continue
        if cfg is None: base = us
        tf = 2 * a.m * a.n * a.k / us / 1e6; mark = ''
        if base and cfg and us < base * 0.997: mark = f'  <-- beats picker {100*(base-us)/base:+.1f}%'
        print(f"  {lab:14s} {us:8.2f}us {tf:7.1f}TF cos={cos:.4f}{mark}")

if __name__ == '__main__':
    main()
