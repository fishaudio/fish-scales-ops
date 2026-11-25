#!/usr/bin/env python3
"""End-to-end B300 benchmark & verification suite for s3-drama (Family C: Qwen3.5-35B-A3B).

Compares upstream fso v0.3.0 (multi-tier host-side size routing) against our
kernel-internal unified architecture (ZERO host-side `if M > X` dispatch):
  - Track C: ONE unified persistent 128x128x128 MXFP8 Dense GEMM kernel
  - Track A: ONE unified persistent FC1+FC2 MoE kernel (M=1..8192, padded, skew)
  - Track B: Shared Expert fused directly into the unified MoE flow (4 kernels total)

All timings use cold weights (> 260 MiB ring > 2x B300 126.5 MiB L2), CUDA Graph
capture & replay, interleaved A/B rounds, plus bitwise `replay == eager` and
FP32 cosine similarity checks.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import fish_scales_ops as fso
from fish_scales_ops import drama3_unified as d3u

# Family C: Qwen3.5-35B-A3B (s3-drama) geometry
E_ROUTED = 256
TOPK = 8
HIDDEN = 2048
MOE_INTER = 512
SHARED_INTER = 512
NUM_LAYERS = 40
NUM_FULL_ATTN = 10
NUM_GDN = 30
SKEW_S = 0.386

DENSE_SHAPES = [
    ("gdn_in_proj",    12288, 2048),  # 30 layers
    ("wqkv_gated",      9216, 2048),  # 10 layers
    ("wo_gdn_out",      2048, 4096),  # 40 layers (10 wo + 30 gdn.out_proj)
    ("shared_gate_up",  1024, 2048),  # 40 layers (standalone ref)
    ("shared_down",     2048,  512),  # 40 layers (standalone ref)
]

L2_FLUSH_BYTES = 280 * 1024 * 1024  # > 2x B300 126.5 MiB L2


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    na = af.norm()
    nb = bf.norm()
    if na.item() == 0.0 and nb.item() == 0.0:
        return 1.0
    return float(torch.dot(af, bf) / (na * nb + 1e-12))


def make_routing(M: int, routing: str = "uniform", live: int | None = None, seed: int = 0):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed if seed else M * 7919 + 3)
    if routing == "uniform":
        scores = torch.rand(M, E_ROUTED, generator=g)
        ids = scores.topk(TOPK, dim=-1).indices.to(torch.int32)
        w = torch.softmax(torch.randn(M, TOPK, generator=g), dim=-1).to(torch.float32)
    elif routing == "skew":
        perm = torch.randperm(E_ROUTED, generator=g)
        ranks = torch.arange(1, E_ROUTED + 1, dtype=torch.float64)
        probs = ranks.pow(-SKEW_S)
        probs = probs / probs.sum()
        inv_probs = torch.empty_like(probs)
        inv_probs[perm] = probs
        ids = torch.multinomial(inv_probs.expand(M, -1), TOPK, replacement=False, generator=g).to(torch.int32)
        w = torch.softmax(torch.randn(M, TOPK, generator=g), dim=-1).to(torch.float32)
    elif routing == "padded":
        assert live is not None and 1 <= live <= M
        ids = torch.full((M, TOPK), -1, dtype=torch.int32)
        w = torch.zeros((M, TOPK), dtype=torch.float32)
        scores = torch.rand(live, E_ROUTED, generator=g)
        ids[:live] = scores.topk(TOPK, dim=-1).indices.to(torch.int32)
        w[:live] = torch.softmax(torch.randn(live, TOPK, generator=g), dim=-1).to(torch.float32)
    else:
        raise ValueError(routing)
    return ids.cuda().contiguous(), w.cuda().contiguous()


def ref_moe_fp32(
    hidden: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    shared_w13: torch.Tensor | None = None,
    shared_w2: torch.Tensor | None = None,
    shared_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    M, H = hidden.shape
    hf = hidden.float()
    out = torch.zeros(M, H, device=hidden.device, dtype=torch.float32)
    ids_cpu = topk_ids.cpu()
    w_cpu = topk_w.cpu()
    # Group by expert for fast exact FP32 reference
    for e in range(w13.shape[0]):
        mask = (topk_ids == e)
        if not mask.any():
            continue
        tok_idx, k_idx = mask.nonzero(as_tuple=True)
        x_e = hf[tok_idx]
        gu = x_e @ w13[e].float().t()
        act = F.silu(gu[:, :MOE_INTER]) * gu[:, MOE_INTER:]
        y_e = act @ w2[e].float().t()
        wt = topk_w[tok_idx, k_idx].unsqueeze(-1)
        out.index_add_(0, tok_idx, y_e * wt)
    if shared_w13 is not None and shared_w2 is not None:
        gu = hf @ shared_w13.float().t()
        act = F.silu(gu[:, :SHARED_INTER]) * gu[:, SHARED_INTER:]
        ys = act @ shared_w2.float().t()
        if shared_gate is not None:
            ys = ys * shared_gate.float().unsqueeze(-1)
        out = out + ys
    return out


def time_graph_ab(fn_a, fn_b, reps_in_graph: int, rounds: int = 7, iters: int = 25):
    """Capture two CUDA graphs and run interleaved A/B rounds; returns (median_a_us, median_b_us)."""
    # Warmup eagerly (critical for workspace pool & CUDA allocator)
    for _ in range(3):
        out_a_eager = fn_a()
        out_b_eager = fn_b()
    torch.cuda.synchronize()

    s = torch.cuda.Stream()
    g_a = torch.cuda.CUDAGraph()
    g_b = torch.cuda.CUDAGraph()
    out_a_g = [None]
    out_b_g = [None]
    with torch.cuda.stream(s):
        for _ in range(2):
            fn_a()
        with torch.cuda.graph(g_a, stream=s):
            out_a_g[0] = fn_a()
        for _ in range(2):
            fn_b()
        with torch.cuda.graph(g_b, stream=s):
            out_b_g[0] = fn_b()
    torch.cuda.synchronize()

    # Bitwise determinism check: replay == eager
    g_a.replay()
    g_b.replay()
    torch.cuda.synchronize()
    eq_a = int(torch.equal(out_a_eager, out_a_g[0]))
    eq_b = int(torch.equal(out_b_eager, out_b_g[0]))

    def measure_one(g):
        # Warmup replay
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(iters):
            g.replay()
        ev1.record()
        torch.cuda.synchronize()
        return (ev0.elapsed_time(ev1) * 1000.0) / (iters * reps_in_graph)

    ta, tb = [], []
    for r in range(rounds):
        if r % 2 == 0:
            ta.append(measure_one(g_a))
            tb.append(measure_one(g_b))
        else:
            tb.append(measure_one(g_b))
            ta.append(measure_one(g_a))
    return statistics.median(ta), statistics.median(tb), eq_a, eq_b, out_a_eager, out_b_eager


def run_smoke():
    print("=== SMOKE & CORRECTNESS CHECK (TRACKS A, B, C) ===")
    torch.manual_seed(42)

    # 1. Track C: Unified Dense GEMM across all edge M + epilogues + replay == eager
    for name, N, K in DENSE_SHAPES:
        w_bf16 = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
        w_fp8, sw = d3u.prepare_dense_weight_unified(w_bf16)
        bias = (torch.randn(N, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
        for M in [1, 2, 7, 8, 16, 31, 32, 64, 65, 128, 129, 256, 512, 1024]:
            x = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
            res = (torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
            x_fp8, sx = fso.compat.quantize_1x32_fp8(x)
            # Identity
            y = d3u.linear_unified(x_fp8, w_fp8, sw, x_scales=sx)
            ref = x.float() @ w_bf16.float().t()
            cs = cos_sim(y, ref)
            assert cs > 0.997, f"dense id fail {name} M={M} cos={cs}"
            # Determinism (2 runs)
            y2 = d3u.linear_unified(x_fp8, w_fp8, sw, x_scales=sx)
            assert torch.equal(y, y2), f"dense nondeterministic {name} M={M}"
            # Epilogue (bias + gelu + residual) on subset
            if M in (1, 16, 65, 256):
                ye = d3u.linear_unified(x_fp8, w_fp8, sw, x_scales=sx, bias=bias, residual=res, act=1)
                ref_e = F.gelu(ref + bias.float(), approximate="none") + res.float()
                cse = cos_sim(ye, ref_e)
                assert cse > 0.996, f"dense epi fail {name} M={M} cos={cse}"
    print("[PASS] Track C Unified Dense GEMM: all shapes, edge M=1..1024, determinism, bias+GELU+residual")

    # 2. Track A & B: Unified MoE (routed-only AND routed+shared with gate) across edge M, padded, skew
    w13 = (torch.randn(E_ROUTED, 2 * MOE_INTER, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    w2 = (torch.randn(E_ROUTED, HIDDEN, MOE_INTER, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    sw13_bf = (torch.randn(2 * SHARED_INTER, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    sw2_bf = (torch.randn(HIDDEN, SHARED_INTER, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()

    exp_routed = d3u.prepare_moe_unified(w13, w2)
    exp_shared = d3u.prepare_moe_unified(w13, w2, shared_w13=sw13_bf, shared_w2=sw2_bf)

    cases = [
        (1, "uniform", None),
        (8, "uniform", None),
        (64, "uniform", None),
        (128, "uniform", None),
        (256, "uniform", None),
        (1024, "uniform", None),
        (4096, "uniform", None),
        (256, "skew", None),
        (1024, "skew", None),
        (256, "padded", 1),
        (256, "padded", 8),
        (512, "padded", 1),
    ]
    for M, routing, live in cases:
        hidden = (torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
        ids, tw = make_routing(M, routing=routing, live=live)
        sg = torch.sigmoid(torch.randn(M, device="cuda", dtype=torch.float32))

        # Routed only
        yr = d3u.moe_layer_unified(hidden, exp_routed, ids, tw)
        yr2 = d3u.moe_layer_unified(hidden, exp_routed, ids, tw)
        assert torch.equal(yr, yr2), f"moe routed nondet M={M} {routing}"
        ref_r = ref_moe_fp32(hidden, w13, w2, ids, tw)
        csr = cos_sim(yr, ref_r)
        assert csr > 0.995, f"moe routed cos fail M={M} {routing} live={live}: {csr}"

        # Routed + Fused Shared with sigmoid gate
        ys = d3u.moe_layer_unified(hidden, exp_shared, ids, tw, shared_gate=sg)
        ys2 = d3u.moe_layer_unified(hidden, exp_shared, ids, tw, shared_gate=sg)
        assert torch.equal(ys, ys2), f"moe shared nondet M={M} {routing}"
        ref_s = ref_moe_fp32(hidden, w13, w2, ids, tw, sw13_bf, sw2_bf, sg)
        css = cos_sim(ys, ref_s)
        assert css > 0.995, f"moe shared cos fail M={M} {routing} live={live}: {css}"
        print(f"  [OK] MoE M={M:4d} {routing:7s} live={str(live):4s} | routed cos={csr:.5f} | +shared cos={css:.5f}")

    # 3. Multi-stream residency deadlock check (2 concurrent streams running Unified MoE + Unified Dense)
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    h64 = (torch.randn(64, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
    id64, w64 = make_routing(64, "uniform")
    with torch.cuda.stream(s1):
        o1 = d3u.moe_layer_unified(h64, exp_shared, id64, w64)
    with torch.cuda.stream(s2):
        o2 = d3u.moe_layer_unified(h64, exp_shared, id64, w64)
    torch.cuda.synchronize()
    assert torch.equal(o1, o2), "multi-stream mismatch"
    print("[PASS] Multi-stream residency & determinism check!")


def bench_dense(ms, out_f=None):
    print("\n=== BENCH SUITE 1: s3-drama DENSE PROJECTIONS (Cold L2, CUDA Graph, BF16-in & Pre-Quant) ===")
    print(f"{'shape':16s} {'M':>5s} | {'fso030(us)':>10s} {'unif(us)':>10s} {'ratio':>7s} | {'unif_TFLOPS':>11s} {'cos':>8s} {'eq':>4s}")
    rows = []
    for name, N, K in DENSE_SHAPES:
        w_bytes = N * K + (N * K) // 32
        copies = max(2, min(64, math.ceil(L2_FLUSH_BYTES / w_bytes)))
        ws_bf16 = [(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous() for _ in range(copies)]
        ws_fso = [fso.dense.prepare_weight(w, format="mxfp8") for w in ws_bf16]
        ws_unif = [d3u.prepare_dense_weight_unified(w) for w in ws_bf16]
        for M in ms:
            x = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
            x_fp8, sx = fso.compat.quantize_1x32_fp8(x)
            x_q_fso, x_s_fso = fso.dense.quantize_activation(x)

            def fn_fso():
                y = None
                for w in ws_fso:
                    y = fso.dense.linear(x_q_fso, w, x_scale=x_s_fso)
                return y

            def fn_unif():
                y = None
                for wf, sw in ws_unif:
                    y = d3u.linear_unified(x_fp8, wf, sw, x_scales=sx)
                return y

            t_fso, t_unif, eq_f, eq_u, y_f, y_u = time_graph_ab(fn_fso, fn_unif, reps_in_graph=copies)
            ref = x.float() @ ws_bf16[-1].float().t()
            cs = cos_sim(y_u, ref)
            tflops = (2.0 * M * N * K) / (t_unif * 1e6)
            ratio = t_unif / t_fso
            print(f"{name:16s} {M:5d} | {t_fso:10.2f} {t_unif:10.2f} {ratio:6.3f}x | {tflops:11.1f} {cs:8.5f} {eq_f}/{eq_u}")
            rec = dict(suite="dense", shape=name, N=N, K=K, M=M, fso030_us=t_fso, unif_us=t_unif,
                       ratio=ratio, unif_tflops=tflops, cos=cs, eq_fso=eq_f, eq_unif=eq_u)
            rows.append(rec)
            if out_f:
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
    return rows


def bench_moe(ms, out_f=None):
    print("\n=== BENCH SUITE 2: s3-drama MoE ROUTED & ROUTED+SHARED BLOCK (Cold Weights, CUDA Graph) ===")
    # 1 layer of 256 experts = 768 MB >> 2x L2 (260 MB)! Even 1 copy is 6x L2!
    # Use 2 copies (~1.5 GB) so every replay sweeps completely cold weights even at M=1!
    copies = 2
    w13s = [(torch.randn(E_ROUTED, 2 * MOE_INTER, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous() for _ in range(copies)]
    w2s = [(torch.randn(E_ROUTED, HIDDEN, MOE_INTER, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous() for _ in range(copies)]
    sw13s = [(torch.randn(2 * SHARED_INTER, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous() for _ in range(copies)]
    sw2s = [(torch.randn(HIDDEN, SHARED_INTER, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous() for _ in range(copies)]

    fso_exps = [fso.moe.prepare_experts(w13s[i], w2s[i], format="mxfp8") for i in range(copies)]
    fso_sh_w13 = [fso.dense.prepare_weight(sw13s[i], format="mxfp8") for i in range(copies)]
    fso_sh_w2 = [fso.dense.prepare_weight(sw2s[i], format="mxfp8") for i in range(copies)]
    unif_routed_exps = [d3u.prepare_moe_unified(w13s[i], w2s[i]) for i in range(copies)]
    unif_block_exps = [d3u.prepare_moe_unified(w13s[i], w2s[i], shared_w13=sw13s[i], shared_w2=sw2s[i]) for i in range(copies)]

    print(f"{'case':18s} | {'fso_r(us)':>9s} {'uni_r(us)':>9s} {'r_rat':>6s} | {'fso_blk':>9s} {'uni_blk':>9s} {'b_rat':>6s} | {'cos_b':>7s} {'eq':>4s}")
    cases = []
    for M in ms:
        cases.append(("uniform", M, None))
    for M in [64, 256, 1024, 2048, 4096]:
        if M <= max(ms):
            cases.append(("skew", M, None))
    for B, L in [(128, 1), (256, 1), (256, 8), (256, 64), (512, 1), (512, 128)]:
        cases.append(("padded", B, L))

    rows = []
    for routing, M, live in cases:
        hidden = (torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
        ids, tw = make_routing(M, routing=routing, live=live)

        # Routed-only comparison
        def fn_fso_r():
            y = None
            for i in range(copies):
                y = fso.moe.layer(hidden, fso_exps[i], ids, tw)
            return y

        def fn_uni_r():
            y = None
            for i in range(copies):
                y = d3u.moe_layer_unified(hidden, unif_routed_exps[i], ids, tw)
            return y

        t_fr, t_ur, eq_fr, eq_ur, _, _ = time_graph_ab(fn_fso_r, fn_uni_r, reps_in_graph=copies, iters=15)

        # Full MoE Block (Routed + Shared) comparison
        def fn_fso_blk():
            y = None
            for i in range(copies):
                gu = fso.dense.linear(hidden, fso_sh_w13[i])
                g, u = gu.chunk(2, dim=-1)
                ys = fso.dense.linear(F.silu(g) * u, fso_sh_w2[i])
                y = fso.moe.layer(hidden, fso_exps[i], ids, tw, bias=ys)
            return y

        def fn_uni_blk():
            y = None
            for i in range(copies):
                y = d3u.moe_layer_unified(hidden, unif_block_exps[i], ids, tw)
            return y

        t_fb, t_ub, eq_fb, eq_ub, y_fb, y_ub = time_graph_ab(fn_fso_blk, fn_uni_blk, reps_in_graph=copies, iters=15)
        cs_b = cos_sim(y_ub, y_fb)
        label = f"{routing}:{M}" if live is None else f"pad:{M}/{live}"
        print(f"{label:18s} | {t_fr:9.2f} {t_ur:9.2f} {t_ur/t_fr:5.3f}x | {t_fb:9.2f} {t_ub:9.2f} {t_ub/t_fb:5.3f}x | {cs_b:7.5f} {eq_fb}/{eq_ub}")
        rec = dict(suite="moe", routing=routing, M=M, live=live,
                   fso_routed_us=t_fr, unif_routed_us=t_ur, routed_ratio=t_ur/t_fr,
                   fso_block_us=t_fb, unif_block_us=t_ub, block_ratio=t_ub/t_fb,
                   cos_vs_fso=cs_b, eq_fso=eq_fb, eq_unif=eq_ub)
        rows.append(rec)
        if out_f:
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
    return rows


def bench_e2e(out_f=None):
    """Full s3-drama 40-layer linear+MoE model step (cold weights, single CUDA Graph)."""
    print("\n=== BENCH SUITE 3: FULL s3-drama 40-LAYER LINEAR + MoE BLOCK STEP (ms/step) ===")
    # Represent exact 40-layer mix (3 GDN + 1 Full-Attn repeated 10x) using 4 cold weight sets
    # (4 MoE layers = 3.1 GB >> 2x L2, dense layers cycled across 16 copies > 280 MB!)
    n_moe_slots = 4
    n_dense_slots = 16
    w_gdn_in = [(torch.randn(12288, 2048, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous() for _ in range(n_dense_slots)]
    w_wqkv = [(torch.randn(9216, 2048, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous() for _ in range(n_dense_slots)]
    w_out = [(torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous() for _ in range(n_dense_slots)]
    w13s = [(torch.randn(E_ROUTED, 2 * MOE_INTER, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous() for _ in range(n_moe_slots)]
    w2s = [(torch.randn(E_ROUTED, HIDDEN, MOE_INTER, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous() for _ in range(n_moe_slots)]
    sw13s = [(torch.randn(2 * SHARED_INTER, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous() for _ in range(n_moe_slots)]
    sw2s = [(torch.randn(HIDDEN, SHARED_INTER, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous() for _ in range(n_moe_slots)]

    # Prepare fso v0.3.0 weights
    fso_gdn_in = [fso.dense.prepare_weight(w, format="mxfp8") for w in w_gdn_in]
    fso_wqkv = [fso.dense.prepare_weight(w, format="mxfp8") for w in w_wqkv]
    fso_out = [fso.dense.prepare_weight(w, format="mxfp8") for w in w_out]
    fso_moe = [fso.moe.prepare_experts(w13s[i], w2s[i], format="mxfp8") for i in range(n_moe_slots)]
    fso_sh1 = [fso.dense.prepare_weight(sw13s[i], format="mxfp8") for i in range(n_moe_slots)]
    fso_sh2 = [fso.dense.prepare_weight(sw2s[i], format="mxfp8") for i in range(n_moe_slots)]

    # Prepare drama3 unified weights
    u_gdn_in = [d3u.prepare_dense_weight_unified(w) for w in w_gdn_in]
    u_wqkv = [d3u.prepare_dense_weight_unified(w) for w in w_wqkv]
    u_out = [d3u.prepare_dense_weight_unified(w) for w in w_out]
    u_moe = [d3u.prepare_moe_unified(w13s[i], w2s[i], shared_w13=sw13s[i], shared_w2=sw2s[i]) for i in range(n_moe_slots)]

    e2e_cases = [
        ("uniform", 1, None),
        ("uniform", 8, None),
        ("uniform", 32, None),
        ("uniform", 64, None),
        ("uniform", 256, None),
        ("padded", 256, 1),
        ("padded", 256, 8),
        ("padded", 256, 64),
        ("skew", 256, None),
        ("skew", 1024, None),
        ("uniform", 1024, None),
        ("uniform", 2048, None),
        ("uniform", 4096, None),
    ]
    print(f"{'scenario':16s} | {'fso030(ms)':>10s} {'unif(ms)':>10s} {'speedup':>8s} {'delta%':>8s} | {'eq':>4s}")
    for routing, M, live in e2e_cases:
        h = (torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
        a_out_in = (torch.randn(M, 4096, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
        ids, tw = make_routing(M, routing=routing, live=live)

        def fn_fso_40():
            x = h
            for L in range(NUM_LAYERS):
                di = L % n_dense_slots
                mi = L % n_moe_slots
                if L % 4 == 3:
                    _ = fso.dense.linear(x, fso_wqkv[di])
                else:
                    _ = fso.dense.linear(x, fso_gdn_in[di])
                _ = fso.dense.linear(a_out_in, fso_out[di])
                gu = fso.dense.linear(x, fso_sh1[mi])
                g, u = gu.chunk(2, dim=-1)
                ys = fso.dense.linear(F.silu(g) * u, fso_sh2[mi])
                x = fso.moe.layer(x, fso_moe[mi], ids, tw, bias=ys)
            return x

        def fn_unif_40():
            x = h
            for L in range(NUM_LAYERS):
                di = L % n_dense_slots
                mi = L % n_moe_slots
                if L % 4 == 3:
                    _ = d3u.linear_unified(x, *u_wqkv[di])
                else:
                    _ = d3u.linear_unified(x, *u_gdn_in[di])
                _ = d3u.linear_unified(a_out_in, *u_out[di])
                x = d3u.moe_layer_unified(x, u_moe[mi], ids, tw)
            return x

        t_f_us, t_u_us, eq_f, eq_u, _, _ = time_graph_ab(fn_fso_40, fn_unif_40, reps_in_graph=1, rounds=5, iters=6)
        t_f_ms, t_u_ms = t_f_us / 1000.0, t_u_us / 1000.0
        sp = t_f_ms / t_u_ms
        delta = (t_u_ms - t_f_ms) / t_f_ms * 100.0
        lbl = f"{routing}:{M}" if live is None else f"pad:{M}/{live}"
        print(f"{lbl:16s} | {t_f_ms:10.3f} {t_u_ms:10.3f} {sp:7.3f}x {delta:+7.1f}% | {eq_f}/{eq_u}")
        rec = dict(suite="e2e40", routing=routing, M=M, live=live,
                   fso030_ms=t_f_ms, unif_ms=t_u_ms, speedup=sp, delta_pct=delta, eq_fso=eq_f, eq_unif=eq_u)
        if out_f:
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--suite", choices=["dense", "moe", "e2e", "all"], default="all")
    ap.add_argument("--Ms", default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return

    ms = [int(x) for x in args.Ms.split(",") if x.strip()]
    out_f = open(args.out, "w") if args.out else None
    try:
        if args.suite in ("dense", "all"):
            bench_dense(ms, out_f)
        if args.suite in ("moe", "all"):
            bench_moe(ms, out_f)
        if args.suite in ("e2e", "all"):
            bench_e2e(out_f)
    finally:
        if out_f:
            out_f.close()


if __name__ == "__main__":
    main()
