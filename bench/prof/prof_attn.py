"""Minimal eager single-op driver for ncu profiling of FSO attention prefill.
Usage: prof_attn.py B S Hq Hkv D causal[0/1] [iters]
Pre-quantizes outside the loop; timed region is just mxfp8_fwd.
"""
import sys, math, torch
import fish_scales_ops as fso
from fish_scales_ops.attention.backends import sm120_mxfp8 as fwd_bk

B, S, Hq, Hkv, D = (int(x) for x in sys.argv[1:6])
causal = bool(int(sys.argv[6])) if len(sys.argv) > 6 else True
iters = int(sys.argv[7]) if len(sys.argv) > 7 else 30
softmax_scale = 1.0 / math.sqrt(D)
q = torch.randn(B, S, Hq, D, dtype=torch.bfloat16, device="cuda") * 0.5
k = torch.randn(B, S, Hkv, D, dtype=torch.bfloat16, device="cuda") * 0.5
v = torch.randn(B, S, Hkv, D, dtype=torch.bfloat16, device="cuda") * 0.5
q_fp8, q_sc = fwd_bk.pre_quantize_q(q)
k_fp8, k_sc = fwd_bk.pre_quantize_k(k)
v_fp8, v_sc = fwd_bk.pre_quantize_v(v)
call = lambda: fwd_bk.mxfp8_fwd(q_fp8, q_sc, k_fp8, k_sc, v_fp8, v_sc,
                                softmax_scale=softmax_scale, causal=causal)
for _ in range(10):
    call()
torch.cuda.synchronize()
for _ in range(iters):
    call()
torch.cuda.synchronize()
print(f"done attn B{B} S{S} Hq{Hq} Hkv{Hkv} D{D} causal{int(causal)}")
