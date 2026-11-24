"""Final verification & 2032 MHz ABBA benchmark for:
  1. Core 32-Layer Audio Tower (conv_out + 32L Encoder + Tail): BF16 vs 0.3.0 fqkv vs Extreme Final
  2. Conv2d Stem only (3x Conv2d 1->480->480->480 + conv_out + pos_emb): BF16 NCHW vs 0.3.0 NHWC vs Fused Stem
  3. Full End-to-End Audio Tower (3x Conv2d Stem + 32L Encoder + Tail): BF16 NCHW vs 0.3.0 fqkv NHWC vs Extreme Final

Uses clean fish_scales_ops.audio.* APIs with ZERO global dec.WF_PRE_WAIT_STAGES_* mutation!
"""
import math
import os
import subprocess
import time
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import fish_scales_ops as fso
import fish_scales_ops.audio as audio
import fish_scales_ops.gemm._sm100_decode as dec
def make_tower_weights(n_layers=32, d=1280, d_ff=5120, d_out=2048, k_conv=7680, seed=42):
    torch.manual_seed(seed)
    dev, dt = "cuda", torch.bfloat16

    def mk_lin(n, k, bias=True):
        w = (torch.randn(n, k, device=dev, dtype=dt) / math.sqrt(k)).contiguous()
        b = (torch.randn(n, device=dev, dtype=dt) * 0.1).contiguous() if bias else None
        return w, b

    def mk_ln(dim):
        g = (torch.ones(dim, device=dev, dtype=dt) + 0.05 * torch.randn(dim, device=dev, dtype=dt)).contiguous()
        b = (0.05 * torch.randn(dim, device=dev, dtype=dt)).contiguous()
        return g, b

    conv_w, _ = mk_lin(d, k_conv, bias=False)
    layers = []
    for _ in range(n_layers):
        ln1 = mk_ln(d)
        wq, bq = mk_lin(d, d, True)
        wk, bk = mk_lin(d, d, True)
        wv, bv = mk_lin(d, d, True)
        wqkv = torch.cat([wq, wk, wv], dim=0).contiguous()
        bqkv = torch.cat([bq, bk, bv], dim=0).contiguous()
        wo, bo = mk_lin(d, d, True)
        ln2 = mk_ln(d)
        w1, b1 = mk_lin(d_ff, d, True)
        w2, b2 = mk_lin(d, d_ff, True)
        layers.append({
            "ln1": ln1, "q": (wq, bq), "k": (wk, bk), "v": (wv, bv),
            "qkv": (wqkv, bqkv), "o": (wo, bo), "ln2": ln2,
            "fc1": (w1, b1), "fc2": (w2, b2),
            "dqkv": fso.dense.prepare_weight(wqkv, format="mxfp8"),
            "do": fso.dense.prepare_weight(wo, format="mxfp8"),
            "dfc1": fso.dense.prepare_weight(w1, format="mxfp8"),
            "dfc2": fso.dense.prepare_weight(w2, format="mxfp8"),
        })
    ln_post = mk_ln(d)
    wp1, bp1 = mk_lin(d, d, True)
    wp2, bp2 = mk_lin(d_out, d, True)
    return {
        "conv": (conv_w, fso.dense.prepare_weight(conv_w, format="mxfp8")),
        "layers": layers,
        "ln_post": ln_post,
        "p1": (wp1, bp1, fso.dense.prepare_weight(wp1, format="mxfp8")),
        "p2": (wp2, bp2, fso.dense.prepare_weight(wp2, format="mxfp8")),
    }


def run_bf16(x_conv, W, attn_fn):
    x = F.linear(x_conv, W["conv"][0])
    M, D = x.shape
    for L in W["layers"]:
        h = F.layer_norm(x, (D,), L["ln1"][0], L["ln1"][1], 1e-5)
        q = F.linear(h, L["q"][0], L["q"][1]).view(1, M, 20, 64).transpose(1, 2)
        k = F.linear(h, L["k"][0], L["k"][1]).view(1, M, 20, 64).transpose(1, 2)
        v = F.linear(h, L["v"][0], L["v"][1]).view(1, M, 20, 64).transpose(1, 2)
        a = attn_fn(q, k, v).transpose(1, 2).reshape(M, D)
        x = F.linear(a, L["o"][0], L["o"][1]) + x
        h2 = F.layer_norm(x, (D,), L["ln2"][0], L["ln2"][1], 1e-5)
        u = F.gelu(F.linear(h2, L["fc1"][0], L["fc1"][1]), approximate="none")
        x = F.linear(u, L["fc2"][0], L["fc2"][1]) + x
    x = F.layer_norm(x, (D,), W["ln_post"][0], W["ln_post"][1], 1e-5)
    x = F.gelu(F.linear(x, W["p1"][0], W["p1"][1]), approximate="none")
    return F.linear(x, W["p2"][0], W["p2"][1])


def run_fso030_fused_qkv(x_conv, W, attn_fn):
    x = fso.dense.linear(x_conv, W["conv"][1])
    M, D = x.shape
    for L in W["layers"]:
        h = F.layer_norm(x, (D,), L["ln1"][0], L["ln1"][1], 1e-5)
        qkv = fso.dense.linear(h, L["dqkv"], bias=L["qkv"][1])
        q, k, v = qkv.view(M, 3, 20, 64).unbind(1)
        a = attn_fn(q.unsqueeze(0).transpose(1, 2), k.unsqueeze(0).transpose(1, 2), v.unsqueeze(0).transpose(1, 2))
        a = a.transpose(1, 2).reshape(M, D)
        x = fso.dense.linear(a, L["do"], bias=L["o"][1], residual=x)
        h2 = F.layer_norm(x, (D,), L["ln2"][0], L["ln2"][1], 1e-5)
        uq, us = fso.dense.linear_quantized(h2, L["dfc1"], bias=L["fc1"][1], activation="gelu_erf")
        x = fso.dense.linear(uq, L["dfc2"], x_scale=us, bias=L["fc2"][1], residual=x)
    x = F.layer_norm(x, (D,), W["ln_post"][0], W["ln_post"][1], 1e-5)
    xq, xs = fso.dense.linear_quantized(x, W["p1"][2], bias=W["p1"][1], activation="gelu_erf")
    return fso.dense.linear(xq, W["p2"][2], x_scale=xs, bias=W["p2"][1])


def cudnn_attn(q, k, v):
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v)


def capture(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    s.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        y = fn()
    s.synchronize()
    g.replay()
    torch.cuda.synchronize()
    return g, y


def time_graphs_clocked(graphs, iters=40, rounds=4):
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.4:
        c = a @ b
    torch.cuda.synchronize()
    del a, b, c
    def _clk():
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    clk0 = _clk()
    res = {k: [] for k in graphs}
    for r in range(rounds):
        keys = list(graphs.keys()) if r % 2 == 0 else list(reversed(graphs.keys()))
        for k in keys:
            g = graphs[k]
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
    return {k: sorted(v)[len(v) // 2] for k, v in res.items()}, clk0, _clk()


def make_stem_weights(seed=777):
    torch.manual_seed(seed)
    dev = "cuda"
    dt = torch.bfloat16
    c1_w = (torch.randn(480, 1, 3, 3, device=dev, dtype=dt) / math.sqrt(9)).contiguous()
    c1_b = (torch.randn(480, device=dev, dtype=dt) * 0.05).contiguous()
    c2_w = (torch.randn(480, 480, 3, 3, device=dev, dtype=dt) / math.sqrt(480 * 9)).contiguous()
    c2_b = (torch.randn(480, device=dev, dtype=dt) * 0.05).contiguous()
    c3_w = (torch.randn(480, 480, 3, 3, device=dev, dtype=dt) / math.sqrt(480 * 9)).contiguous()
    c3_b = (torch.randn(480, device=dev, dtype=dt) * 0.05).contiguous()
    return {
        "nchw": ((c1_w, c1_b), (c2_w, c2_b), (c3_w, c3_b)),
        "nhwc": (
            (c1_w.to(memory_format=torch.channels_last), c1_b),
            (c2_w.to(memory_format=torch.channels_last), c2_b),
            (c3_w.to(memory_format=torch.channels_last), c3_b),
        ),
    }


def run_core_extreme(x_conv, W):
    xc_q, xc_s = fso.dense.quantize_activation(x_conv)
    x = audio.linear_audio_conv_out(xc_q, W["conv"][1], xc_s)
    M, _ = x.shape
    for L in W["layers"]:
        hq, hs = audio.layernorm_mxfp8(x, L["ln1"][0], L["ln1"][1])
        qkv = audio.linear_audio_qkv(hq, L["dqkv"], hs, bias=L["qkv"][1])
        q, k, v = qkv.view(M, 3, 20, 64).unbind(1)
        aq, as_ = audio.attn_fa4_mxfp8(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
        x = audio.linear_audio_out_proj(aq, L["do"], as_, bias=L["o"][1], residual=x)
        h2q, h2s = audio.layernorm_mxfp8(x, L["ln2"][0], L["ln2"][1])
        uq, us = fso.dense.linear_quantized(h2q, L["dfc1"], x_scale=h2s, bias=L["fc1"][1], activation="gelu_erf")
        x = audio.linear_audio_fc2(uq, L["dfc2"], us, bias=L["fc2"][1], residual=x)
    xq, xs = audio.layernorm_mxfp8(x, W["ln_post"][0], W["ln_post"][1])
    xq2, xs2 = fso.dense.linear_quantized(xq, W["p1"][2], x_scale=xs, bias=W["p1"][1], activation="gelu_erf")
    return audio.linear_audio_proj2(xq2, W["p2"][2], xs2, bias=W["p2"][1])


def run_stem_bf16_nchw(mel, pos, S, W):
    (w1, b1), (w2, b2), (w3, b3) = S["nchw"]
    h = F.gelu(F.conv2d(mel, w1, b1, stride=2, padding=1), approximate="none")
    h = F.gelu(F.conv2d(h, w2, b2, stride=2, padding=1), approximate="none")
    h = F.gelu(F.conv2d(h, w3, b3, stride=2, padding=1), approximate="none")
    B, C, Fr, T = h.shape
    xc = h.permute(0, 3, 1, 2).reshape(B * T, C * Fr).contiguous()
    return F.linear(xc, W["conv"][0]) + pos


def run_stem_030_nhwc(mel_nhwc, pos, S, W):
    (w1, b1), (w2, b2), (w3, b3) = S["nhwc"]
    h = F.gelu(F.conv2d(mel_nhwc, w1, b1, stride=2, padding=1), approximate="none")
    h = F.gelu(F.conv2d(h, w2, b2, stride=2, padding=1), approximate="none")
    h = F.gelu(F.conv2d(h, w3, b3, stride=2, padding=1), approximate="none")
    B, C, Fr, T = h.shape
    xc = h.permute(0, 3, 1, 2).reshape(B * T, C * Fr).contiguous()
    return fso.dense.linear(xc, W["conv"][1], residual=pos)


def run_stem_extreme(mel_nhwc, pos, S, dw_co_perm):
    (w1, b1), (w2, b2), (w3, b3) = S["nhwc"]
    h = audio.gelu_erf_bf16_pdl(F.conv2d(mel_nhwc, w1, b1, stride=2, padding=1))
    h = audio.gelu_erf_bf16_pdl(F.conv2d(h, w2, b2, stride=2, padding=1))
    h = F.conv2d(h, w3, b3, stride=2, padding=1)
    xc_q, xc_s = audio.gelu_nhwc_to_mxfp8(h)
    return audio.linear_audio_conv_out(xc_q, dw_co_perm, xc_s, residual=pos)


def run_full_bf16(mel, pos, S, W):
    x = run_stem_bf16_nchw(mel, pos, S, W)
    M, D = x.shape
    for L in W["layers"]:
        h = F.layer_norm(x, (D,), L["ln1"][0], L["ln1"][1], 1e-5)
        qkv = F.linear(h, L["qkv"][0], L["qkv"][1])
        q, k, v = qkv.view(M, 3, 20, 64).unbind(1)
        a = cudnn_attn(q.unsqueeze(0).transpose(1, 2), k.unsqueeze(0).transpose(1, 2), v.unsqueeze(0).transpose(1, 2)).transpose(1, 2).reshape(M, D)
        x = F.linear(a, L["o"][0], L["o"][1]) + x
        h2 = F.layer_norm(x, (D,), L["ln2"][0], L["ln2"][1], 1e-5)
        u = F.gelu(F.linear(h2, L["fc1"][0], L["fc1"][1]), approximate="none")
        x = F.linear(u, L["fc2"][0], L["fc2"][1]) + x
    x = F.layer_norm(x, (D,), W["ln_post"][0], W["ln_post"][1], 1e-5)
    x = F.gelu(F.linear(x, W["p1"][0], W["p1"][1]), approximate="none")
    return F.linear(x, W["p2"][0], W["p2"][1])


def run_full_030_fqkv(mel_nhwc, pos, S, W):
    x = run_stem_030_nhwc(mel_nhwc, pos, S, W)
    M, D = x.shape
    for L in W["layers"]:
        h = F.layer_norm(x, (D,), L["ln1"][0], L["ln1"][1], 1e-5)
        qkv = fso.dense.linear(h, L["dqkv"], bias=L["qkv"][1])
        q, k, v = qkv.view(M, 3, 20, 64).unbind(1)
        a = cudnn_attn(q.unsqueeze(0).transpose(1, 2), k.unsqueeze(0).transpose(1, 2), v.unsqueeze(0).transpose(1, 2)).transpose(1, 2).reshape(M, D)
        x = fso.dense.linear(a, L["do"], bias=L["o"][1], residual=x)
        h2 = F.layer_norm(x, (D,), L["ln2"][0], L["ln2"][1], 1e-5)
        uq, us = fso.dense.linear_quantized(h2, L["dfc1"], bias=L["fc1"][1], activation="gelu_erf")
        x = fso.dense.linear(uq, L["dfc2"], x_scale=us, bias=L["fc2"][1], residual=x)
    x = F.layer_norm(x, (D,), W["ln_post"][0], W["ln_post"][1], 1e-5)
    xq, xs = fso.dense.linear_quantized(x, W["p1"][2], bias=W["p1"][1], activation="gelu_erf")
    return fso.dense.linear(xq, W["p2"][2], x_scale=xs, bias=W["p2"][1])


def run_full_extreme(mel_nhwc, pos, S, W, dw_co_perm):
    x = run_stem_extreme(mel_nhwc, pos, S, dw_co_perm)
    M, _ = x.shape
    for L in W["layers"]:
        hq, hs = audio.layernorm_mxfp8(x, L["ln1"][0], L["ln1"][1])
        qkv = audio.linear_audio_qkv(hq, L["dqkv"], hs, bias=L["qkv"][1])
        q, k, v = qkv.view(M, 3, 20, 64).unbind(1)
        aq, as_ = audio.attn_fa4_mxfp8(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
        x = audio.linear_audio_out_proj(aq, L["do"], as_, bias=L["o"][1], residual=x)
        h2q, h2s = audio.layernorm_mxfp8(x, L["ln2"][0], L["ln2"][1])
        uq, us = fso.dense.linear_quantized(h2q, L["dfc1"], x_scale=h2s, bias=L["fc1"][1], activation="gelu_erf")
        x = audio.linear_audio_fc2(uq, L["dfc2"], us, bias=L["fc2"][1], residual=x)
    xq, xs = audio.layernorm_mxfp8(x, W["ln_post"][0], W["ln_post"][1])
    xq2, xs2 = fso.dense.linear_quantized(xq, W["p1"][2], x_scale=xs, bias=W["p1"][1], activation="gelu_erf")
    return audio.linear_audio_proj2(xq2, W["p2"][2], xs2, bias=W["p2"][1])


def main():
    torch.backends.cudnn.benchmark = True
    assert dec.WF_PRE_WAIT_STAGES_ONE_CTA == 2 and dec.WF_PRE_WAIT_STAGES_TWO_CTA == 1
    S = make_stem_weights()
    W = make_tower_weights(n_layers=32)
    dw_co_perm = audio.prepare_conv_out_weight(W["conv"][0], channels=480, freq=16)

    # 1. Core 32L Audio Tower (from x_conv [M, 7680])
    print("=== [A] Core 32L Audio Tower (conv_out + 32L Encoder + Tail, 2032 MHz, us) ===", flush=True)
    print(f"{'M':>5} | {'BF16':>8} | {'0.3.0 fqkv':>10} | {'Extreme':>9} | {'vs 0.3.0':>10} | {'vs BF16':>10} | {'cos(BF16)':>9} | clocks", flush=True)
    print("-" * 92, flush=True)
    for M in [78, 117, 182, 364, 533, 1079]:
        torch.manual_seed(1000 + M)
        xc = (torch.randn(M, 7680, device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
        y_bf = run_bf16(xc, W, cudnn_attn)
        y_ex = run_core_extreme(xc, W)
        cos = F.cosine_similarity(y_bf.float().flatten(), y_ex.float().flatten(), dim=0).item()
        assert cos > 0.99, f"Core cos too low at M={M}: {cos}"
        g_bf, _ = capture(lambda: run_bf16(xc, W, cudnn_attn))
        g_03, _ = capture(lambda: run_fso030_fused_qkv(xc, W, cudnn_attn))
        g_ex, y_ex_g = capture(lambda: run_core_extreme(xc, W))
        assert torch.equal(y_ex.view(torch.uint16), y_ex_g.view(torch.uint16)), f"Core Eager!=Graph at M={M}"
        t, _, c1 = time_graphs_clocked({"bf": g_bf, "f03": g_03, "ex": g_ex}, iters=40, rounds=4)
        print(
            f"{M:>5} | {t['bf']:>8.1f} | {t['f03']:>10.1f} | {t['ex']:>9.1f} | "
            f"{(t['ex']/t['f03']-1)*100:>+9.1f}% | {(t['ex']/t['bf']-1)*100:>+9.1f}% | {cos:>9.6f} | {c1}",
            flush=True,
        )
        del g_bf, g_03, g_ex

    # 2. Stem-only & 3. Full End-to-End Audio Tower (3x Conv2d Stem + 32L Encoder + Tail)
    print("\n=== [B] 3x Conv2d Stem Only (Conv1+2+3 + conv_out + pos_emb, 2032 MHz, us) ===", flush=True)
    print(f"{'M':>5} | {'B':>3} | {'BF16 NCHW':>10} | {'0.3.0 NHWC':>10} | {'Fused Stem':>10} | {'vs 0.3.0':>9} | {'vs BF16':>9} | clocks", flush=True)
    print("-" * 86, flush=True)
    stem_rows = []
    for M in [78, 117, 182, 364, 533, 1079]:
        B = M // 13
        torch.manual_seed(2000 + M)
        mel = (torch.randn(B, 1, 128, 100, device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
        mel_nhwc = mel.to(memory_format=torch.channels_last)
        pos = (torch.randn(M, 1280, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
        # Warmup cuDNN algo selection
        for _ in range(2):
            run_stem_bf16_nchw(mel, pos, S, W)
            run_stem_030_nhwc(mel_nhwc, pos, S, W)
            run_stem_extreme(mel_nhwc, pos, S, dw_co_perm)
        g_sb, _ = capture(lambda: run_stem_bf16_nchw(mel, pos, S, W))
        g_s0, _ = capture(lambda: run_stem_030_nhwc(mel_nhwc, pos, S, W))
        g_se, _ = capture(lambda: run_stem_extreme(mel_nhwc, pos, S, dw_co_perm))
        ts, _, c1 = time_graphs_clocked({"bf": g_sb, "f03": g_s0, "ex": g_se}, iters=40, rounds=4)
        print(
            f"{M:>5} | {B:>3} | {ts['bf']:>10.1f} | {ts['f03']:>10.1f} | {ts['ex']:>10.1f} | "
            f"{(ts['ex']/ts['f03']-1)*100:>+8.1f}% | {(ts['ex']/ts['bf']-1)*100:>+8.1f}% | {c1}",
            flush=True,
        )
        del g_sb, g_s0, g_se

    print("\n=== [C] Full End-to-End Audio Tower (3x Conv2d Stem + 32L Encoder + Tail, 2032 MHz, us) ===", flush=True)
    print(f"{'M':>5} | {'BF16 NCHW':>10} | {'0.3.0 NHWC':>10} | {'Extreme':>9} | {'vs 0.3.0':>9} | {'vs BF16':>9} | {'cos(BF16)':>9} | clocks", flush=True)
    print("-" * 92, flush=True)
    for M in [78, 117, 182, 364, 533, 1079]:
        B = M // 13
        torch.manual_seed(3000 + M)
        mel = (torch.randn(B, 1, 128, 100, device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
        mel_nhwc = mel.to(memory_format=torch.channels_last)
        pos = (torch.randn(M, 1280, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()

        y_bf = run_full_bf16(mel, pos, S, W)
        y_ex = run_full_extreme(mel_nhwc, pos, S, W, dw_co_perm)
        cos = F.cosine_similarity(y_bf.float().flatten(), y_ex.float().flatten(), dim=0).item()
        assert cos > 0.99, f"Full E2E cos too low at M={M}: {cos}"

        g_bf, _ = capture(lambda: run_full_bf16(mel, pos, S, W))
        g_03, _ = capture(lambda: run_full_030_fqkv(mel_nhwc, pos, S, W))
        g_ex, y_ex_g = capture(lambda: run_full_extreme(mel_nhwc, pos, S, W, dw_co_perm))
        assert torch.equal(y_ex.view(torch.uint16), y_ex_g.view(torch.uint16)), f"Full Eager != Graph at M={M}"

        t, _, c1 = time_graphs_clocked({"bf": g_bf, "f03": g_03, "ex": g_ex}, iters=40, rounds=4)
        print(
            f"{M:>5} | {t['bf']:>10.1f} | {t['f03']:>10.1f} | {t['ex']:>9.1f} | "
            f"{(t['ex']/t['f03']-1)*100:>+8.1f}% | {(t['ex']/t['bf']-1)*100:>+8.1f}% | {cos:>9.6f} | {c1}",
            flush=True,
        )
        del g_bf, g_03, g_ex

    assert dec.WF_PRE_WAIT_STAGES_ONE_CTA == 2 and dec.WF_PRE_WAIT_STAGES_TWO_CTA == 1
    print("\nGlobal dec.WF_PRE_WAIT_STAGES verified UNTOUCHED (2 / 1).", flush=True)


if __name__ == "__main__":
    main()
