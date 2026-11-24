"""Exhaustive 32-layer real-pipeline role-tactic sweep across all 6 M rungs (2032 MHz).

Checks disp.plan() for every (role, M) and sweeps CuTe-DSL persistent & split-K tactics
(with wf_cap=3/5) for QKV [3840x1280], out_proj [1280x1280], and fc2 [1280x5120]
inside the real 32-layer cold-weight CUDA graph!
"""
import sys
sys.path.insert(0, "/home/aniya/fso-audio-extreme-opt/bench/audio")

import torch
import fish_scales_ops as fso
import fish_scales_ops.gemm._sm100_decode as dec
import fish_scales_ops.gemm._sm100_dispatch as disp
import fish_scales_ops.gemm._epilogue as ep
from fish_scales_ops.audio.fa4_mxfp8 import attn_fa4_mxfp8
from probe_baseline import make_tower_weights, capture
from bench_p2_ablation import time_graphs_clocked


def lin_audio(xq, dw, xs, bias=None, residual=None, cfg_override=None, early_pdl=True):
    m, k = int(xq.shape[0]), int(xq.shape[1])
    n = int(dw.weight.shape[0])
    epi = ep.Epilogue(bias=bias, residual=residual) if (bias is not None or residual is not None) else None
    cfg = cfg_override
    if cfg is None:
        for key, cand_cfg in disp._candidates(m, n, k, epi):
            if key == disp.DECODE:
                cfg = cand_cfg
            break
    if cfg is not None and dec.enabled() and (epi is None or dec.epilogue_available()):
        cluster = cfg[1]
        wf_cap = 5 if cluster[0] * cluster[1] > 1 else 3
        y = dec.linear_mxfp8_decode(xq, dw.weight, xs, dw.scale, cfg, epi, wf_cap=wf_cap, early_pdl=early_pdl)
        if y is not None:
            return y
    return fso.dense.linear(xq, dw, x_scale=xs, bias=bias, residual=residual)


def run_tower(x_conv, W, qkv_cfg=None, o_cfg=None, fc2_cfg=None, early_pdl=True):
    xcq, xcs = fso.dense.quantize_activation(x_conv)
    x = lin_audio(xcq, W["conv"][1], xcs, early_pdl=early_pdl)
    M, _ = x.shape
    for L in W["layers"]:
        hq, hs = torch.ops.fish_scales_ops.layernorm_mxfp8_sm100(x, L["ln1"][0], L["ln1"][1], 1e-5, False, 1)
        qkv = lin_audio(hq, L["dqkv"], hs, bias=L["qkv"][1], cfg_override=qkv_cfg, early_pdl=early_pdl)
        q, k, v = qkv.view(M, 3, 20, 64).unbind(1)
        aq, as_ = attn_fa4_mxfp8(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
        x = lin_audio(aq, L["do"], as_, bias=L["o"][1], residual=x, cfg_override=o_cfg, early_pdl=early_pdl)
        h2q, h2s = torch.ops.fish_scales_ops.layernorm_mxfp8_sm100(x, L["ln2"][0], L["ln2"][1], 1e-5, False, 1)
        uq, us = fso.dense.linear_quantized(h2q, L["dfc1"], x_scale=h2s, bias=L["fc1"][1], activation="gelu_erf")
        x = lin_audio(uq, L["dfc2"], us, bias=L["fc2"][1], residual=x, cfg_override=fc2_cfg, early_pdl=early_pdl)
    xq, xs = torch.ops.fish_scales_ops.layernorm_mxfp8_sm100(x, W["ln_post"][0], W["ln_post"][1], 1e-5, False, 1)
    xq2, xs2 = fso.dense.linear_quantized(xq, W["p1"][2], x_scale=xs, bias=W["p1"][1], activation="gelu_erf")
    return lin_audio(xq2, W["p2"][2], xs2, bias=W["p2"][1], early_pdl=False)


# Candidate tactics
T_128x64   = ((128, 64),  (1, 1), True, 1)
T_256x64   = ((256, 64),  (2, 1), True, 1)
T_128x128  = ((128, 128), (1, 1), True, 1)
T_256x128  = ((256, 128), (2, 1), True, 1)
T_SK2_64   = ((128, 64),  (1, 1), True, 2)
T_SK2_128  = ((128, 128), (1, 1), True, 2)


def main():
    W = make_tower_weights(n_layers=32)
    b_dummy = torch.empty(1, device="cuda", dtype=torch.bfloat16)
    e_b = ep.Epilogue(bias=b_dummy)
    e_br = ep.Epilogue(bias=b_dummy, residual=b_dummy)

    print("=== Default 0.3.0 dispatch plan ===", flush=True)
    for M in [78, 117, 182, 364, 533, 1079]:
        p_qkv = list(disp._candidates(M, 3840, 1280, e_b))
        p_o   = list(disp._candidates(M, 1280, 1280, e_br))
        p_fc2 = list(disp._candidates(M, 1280, 5120, e_br))
        print(f"  M={M:4d}: QKV={p_qkv[:1]} | O={p_o[:1]} | FC2={p_fc2[:1]}", flush=True)

    # Current baseline per-M configs
    base_cfg = {
        78:   (T_128x64, None, None),
        117:  (T_128x64, None, None),
        182:  (T_256x64, None, None),
        364:  (None,     None, None),
        533:  (None,     T_256x64, None),
        1079: (None,     None, None),
    }

    candidates_by_m = {
        78: {
            "qkv_128x128": (T_128x128, None, None),
            "qkv_sk2_64":  (T_SK2_64, None, None),
            "o_128x64":    (T_128x64, T_128x64, None),
            "o_sk2_64":    (T_128x64, T_SK2_64, None),
            "fc2_256x64":  (T_128x64, None, T_256x64),
        },
        117: {
            "qkv_128x128": (T_128x128, None, None),
            "qkv_sk2_64":  (T_SK2_64, None, None),
            "o_128x64":    (T_128x64, T_128x64, None),
            "o_128x128":   (T_128x64, T_128x128, None),
            "fc2_256x64":  (T_128x64, None, T_256x64),
        },
        182: {
            "qkv_128x128": (T_128x128, None, None),
            "qkv_256x128": (T_256x128, None, None),
            "o_256x64":    (T_256x64, T_256x64, None),
            "o_128x128":   (T_256x64, T_128x128, None),
            "fc2_256x64":  (T_256x64, None, T_256x64),
            "fc2_256x128": (T_256x64, None, T_256x128),
        },
        364: {
            "qkv_128x64":  (T_128x64, None, None),
            "qkv_128x128": (T_128x128, None, None),
            "qkv_256x128": (T_256x128, None, None),
            "o_128x128":   (None, T_128x128, None),
            "o_256x128":   (None, T_256x128, None),
            "fc2_256x64":  (None, None, T_256x64),
            "fc2_256x128": (None, None, T_256x128),
        },
        533: {
            "qkv_256x64":  (T_256x64, T_256x64, None),
            "qkv_256x128": (T_256x128, T_256x64, None),
            "qkv_128x128": (T_128x128, T_256x64, None),
            "qkv_128x64":  (T_128x64, T_256x64, None),
            "o_128x128":   (T_256x64, T_128x128, None),
            "fc2_256x128": (T_256x64, T_256x64, T_256x128),
        },
        1079: {
            "qkv_256x64":  (T_256x64, None, None),
            "qkv_256x128": (T_256x128, None, None),
            "qkv_128x128": (T_128x128, None, None),
            "o_256x64":    (None, T_256x64, None),
            "o_128x128":   (None, T_128x128, None),
            "qkv256x128_o256x64": (T_256x128, T_256x64, None),
        },
    }

    print("\n=== Full 32L Role-Tactic Sweep (2032 MHz, us) ===", flush=True)
    for M in [78, 117, 182, 364, 533, 1079]:
        torch.manual_seed(900 + M)
        x = (torch.randn(M, 7680, device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
        bq, bo, bf = base_cfg[M]
        arms = {}
        g_base, _ = capture(lambda: run_tower(x, W, bq, bo, bf))
        arms["base"] = g_base
        for name, (qc, oc, fc) in candidates_by_m[M].items():
            try:
                g, _ = capture(lambda qc=qc, oc=oc, fc=fc: run_tower(x, W, qc, oc, fc))
                arms[name] = g
            except Exception as e:
                print(f"  M={M} {name} SKIP: {e}", flush=True)
        t, _, c1 = time_graphs_clocked(arms, iters=30, rounds=3)
        best_k = min(t, key=t.get)
        parts = [f"{k}={v:.1f}" + ("*" if k == best_k else "") for k, v in t.items()]
        print(f"M={M:4d} ({c1}): " + " | ".join(parts), flush=True)
        del arms


if __name__ == "__main__":
    main()
