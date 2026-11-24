"""Clock-monitored ABBA ablation of Phase 2 / 2B FA4_MXFP8 configs + DVFS check."""
import math
import subprocess
import time
import torch
import torch.nn.functional as F
import cutlass.cute as cute
import cuda.bindings.driver as cuda_drv
from cutlass import Float32

from flash_attn.cute import config as fa_cfg
import fish_scales_ops as fso
from fish_scales_ops.audio.fa4_mxfp8 import (
    FlashAttentionForwardSm100MxFP8,
    _patch_launch_pdl,
    _to_cute,
)
from probe_baseline import make_tower_weights, run_fso030_fused_qkv, cudnn_attn, capture


def get_sm_clock():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    return out


def warm_gpu(secs=0.5):
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        c = a @ b
    torch.cuda.synchronize()
    del a, b, c


def time_graphs_clocked(graphs, iters=40, rounds=4):
    warm_gpu(0.4)
    clk0 = get_sm_clock()
    res = {k: [] for k in graphs}
    for r in range(rounds):
        keys = list(graphs.keys()) if r % 2 == 0 else list(reversed(graphs.keys()))
        for k in keys:
            g = graphs[k]
            # 5 unmeasured warm replays to keep SM clock hot on this graph
            for _ in range(5):
                g.replay()
            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            st.record()
            for _ in range(iters):
                g.replay()
            en.record()
            torch.cuda.synchronize()
            res[k].append(st.elapsed_time(en) * 1000.0 / iters)
    clk1 = get_sm_clock()
    return {k: sorted(v)[len(v) // 2] for k, v in res.items()}, clk0, clk1


class FA4MxConfigurable(FlashAttentionForwardSm100MxFP8):
    def __init__(self, *args, early_pdl: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_pdl = early_pdl

    @cute.jit
    def load(self, *args, **kwargs):
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        res = super(FlashAttentionForwardSm100MxFP8, self).load(*args, **kwargs)
        if cutlass.const_expr(self.enable_pdl and self.early_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        return res

    @cute.jit
    def epilogue_s2g(self, *args, **kwargs):
        res = super(FlashAttentionForwardSm100MxFP8, self).epilogue_s2g(*args, **kwargs)
        if cutlass.const_expr(self.enable_pdl and not self.early_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        return res


import cutlass

_CACHE = {}


def make_fa4_fn(q_stage, kv_stage, use_pp, is_pers, reg_sm, sm_cross, early_pdl, tile_n=128):
    def _attn(q, k, v):
        M = q.shape[1]
        H = q.shape[2]
        D = H * 64
        M_pad = (M + 127) // 128 * 128
        num_kp = D // 128
        out_q = torch.empty((1, M, H, 64), dtype=torch.float8_e4m3fn, device=q.device)
        out_scale = torch.empty((M_pad * num_kp,), dtype=torch.int32, device=q.device)
        sx_u16 = out_scale.view(torch.uint16).view(1, 1, M_pad * num_kp * 2)
        scale = Float32(1.0 / math.sqrt(64.0))
        ckey = (q_stage, kv_stage, use_pp, is_pers, reg_sm, sm_cross, early_pdl, tile_n, H, q.stride(1), q.stride(2))
        fn = _CACHE.get(ckey)
        q_ct = _to_cute(q, 3, 16)
        k_ct = _to_cute(k, 3, 16)
        v_ct = _to_cute(v, 3, 16)
        o_ct = _to_cute(out_q, 3, 16)
        sf_ct = _to_cute(sx_u16, 2, 16)
        stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
        if fn is None:
            cfg = fa_cfg.FwdConfig(
                device_capacity=10,
                tile_m=128,
                tile_n=tile_n,
                mma_pv_is_rs=False,
                intra_wg_overlap=False,
                q_stage=q_stage,
                pack_gqa=False,
                use_clc_scheduler=False,
                is_static_persistent=is_pers,
                use_tma_o=True,
                use_s_ping_pong=use_pp,
                num_splits=1,
                use_2cta_instrs=False,
                registers=fa_cfg.Sm100Registers(softmax=reg_sm, correction=64, other=48),
                ex2_emu_freq=0,
                ex2_emu_start_frg=0,
                overlap_sO_sQ=False,
                kv_stage=kv_stage,
                sm_stats_cross_stage=sm_cross,
            )
            fa_fwd = FA4MxConfigurable(
                64, 64, config=cfg, qhead_per_kvhead=1, is_causal=False, is_local=False,
                nheads_total=H, enable_pdl=True, early_pdl=early_pdl,
            )
            _patch_launch_pdl(fa_fwd)
            fn = cute.compile(fa_fwd, q_ct, k_ct, v_ct, o_ct, sf_ct, scale, stream=stream)
            _CACHE[ckey] = fn
        fn(q_ct, k_ct, v_ct, o_ct, sf_ct, scale, stream=stream)
        return out_q.view(M, D), out_scale
    return _attn


def run_tower(x_conv, W, fa4_fn):
    x = fso.dense.linear(x_conv, W["conv"][1])
    M, D = x.shape
    for L in W["layers"]:
        hq, hs = torch.ops.fish_scales_ops.layernorm_mxfp8_sm100(x, L["ln1"][0], L["ln1"][1], 1e-5, False, 1)
        qkv = fso.dense.linear(hq, L["dqkv"], x_scale=hs, bias=L["qkv"][1])
        q, k, v = qkv.view(M, 3, 20, 64).unbind(1)
        aq, as_ = fa4_fn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
        x = fso.dense.linear(aq, L["do"], x_scale=as_, bias=L["o"][1], residual=x)
        h2q, h2s = torch.ops.fish_scales_ops.layernorm_mxfp8_sm100(x, L["ln2"][0], L["ln2"][1], 1e-5, False, 1)
        uq, us = fso.dense.linear_quantized(h2q, L["dfc1"], x_scale=h2s, bias=L["fc1"][1], activation="gelu_erf")
        x = fso.dense.linear(uq, L["dfc2"], x_scale=us, bias=L["fc2"][1], residual=x)
    xq, xs = torch.ops.fish_scales_ops.layernorm_mxfp8_sm100(x, W["ln_post"][0], W["ln_post"][1], 1e-5, False, 1)
    xq2, xs2 = fso.dense.linear_quantized(xq, W["p1"][2], x_scale=xs, bias=W["p1"][1], activation="gelu_erf")
    return fso.dense.linear(xq2, W["p2"][2], x_scale=xs2, bias=W["p2"][1])


if __name__ == "__main__":
    print(f"Start GPU state: {get_sm_clock()}")
    W = make_tower_weights()
    print(f"{'M':>5} | {'0.3.0':>7} | {'p2_old':>7} | {'p2b_cur':>7} | {'p2b_late':>8} | {'p2b_np':>7} | {'p2b_q1':>7} | {'p2b_q1np':>8} | clocks (start -> end)")
    print("-" * 110)
    for M in [78, 117, 182, 364, 533, 1079]:
        x = torch.randn(M, 7680, device="cuda", dtype=torch.bfloat16) * 0.5
        qs_def = 1 if M <= 128 else 2
        fn_old = make_fa4_fn(qs_def, 3 if qs_def == 1 else 4, False, M > 512, 192, True, False)
        fn_cur = make_fa4_fn(qs_def, 12 if qs_def == 1 else 10, qs_def == 1, True, 200, False, True)
        fn_late = make_fa4_fn(qs_def, 12 if qs_def == 1 else 10, qs_def == 1, True, 200, False, False)
        fn_np = make_fa4_fn(qs_def, 12 if qs_def == 1 else 10, qs_def == 1, False, 200, False, True)
        fn_q1 = make_fa4_fn(1, 12, True, True, 200, False, True)
        fn_q1np = make_fa4_fn(1, 12, True, False, 200, False, True)

        # Quick accuracy check on fn_q1 at M=182, 364, 533
        if M in (78, 182, 533):
            y_cur = run_tower(x, W, fn_cur)
            y_q1 = run_tower(x, W, fn_q1)
            cos = F.cosine_similarity(y_cur.float().flatten(), y_q1.float().flatten(), dim=0).item()
            assert cos > 0.9999, f"fn_q1 mismatch at M={M}: cos={cos}"

        graphs = {
            "base": capture(lambda: run_fso030_fused_qkv(x, W, cudnn_attn))[0],
            "old": capture(lambda: run_tower(x, W, fn_old))[0],
            "cur": capture(lambda: run_tower(x, W, fn_cur))[0],
            "late": capture(lambda: run_tower(x, W, fn_late))[0],
            "np": capture(lambda: run_tower(x, W, fn_np))[0],
            "q1": capture(lambda: run_tower(x, W, fn_q1))[0],
            "q1np": capture(lambda: run_tower(x, W, fn_q1np))[0],
        }
        t, c0, c1 = time_graphs_clocked(graphs)
        print(f"{M:>5} | {t['base']:>7.1f} | {t['old']:>7.1f} | {t['cur']:>7.1f} | {t['late']:>8.1f} | {t['np']:>7.1f} | {t['q1']:>7.1f} | {t['q1np']:>8.1f} | {c0} -> {c1}", flush=True)
