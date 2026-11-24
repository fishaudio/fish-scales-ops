import math
import os
import sys
import time
import torch
import torch.nn.functional as F
import fish_scales_ops as fso


def make_tower_weights(n_layers=32, d=1280, d_ff=5120, d_out=2048, k_conv=7680, seed=42):
    torch.manual_seed(seed)
    dev = "cuda"
    dt = torch.bfloat16

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
            "dq": fso.dense.prepare_weight(wq, format="mxfp8"),
            "dk": fso.dense.prepare_weight(wk, format="mxfp8"),
            "dv": fso.dense.prepare_weight(wv, format="mxfp8"),
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


def run_fso030_unfused(x_conv, W, attn_fn):
    x = fso.dense.linear(x_conv, W["conv"][1])
    M, D = x.shape
    for L in W["layers"]:
        h = F.layer_norm(x, (D,), L["ln1"][0], L["ln1"][1], 1e-5)
        hq, hs = fso.dense.quantize_activation(h)
        q = fso.dense.linear(hq, L["dq"], x_scale=hs, bias=L["q"][1]).view(1, M, 20, 64).transpose(1, 2)
        k = fso.dense.linear(hq, L["dk"], x_scale=hs, bias=L["k"][1]).view(1, M, 20, 64).transpose(1, 2)
        v = fso.dense.linear(hq, L["dv"], x_scale=hs, bias=L["v"][1]).view(1, M, 20, 64).transpose(1, 2)
        a = attn_fn(q, k, v).transpose(1, 2).reshape(M, D)
        x = fso.dense.linear(a, L["do"], bias=L["o"][1], residual=x)
        h2 = F.layer_norm(x, (D,), L["ln2"][0], L["ln2"][1], 1e-5)
        uq, us = fso.dense.linear_quantized(h2, L["dfc1"], bias=L["fc1"][1], activation="gelu_erf")
        x = fso.dense.linear(uq, L["dfc2"], x_scale=us, bias=L["fc2"][1], residual=x)
    x = F.layer_norm(x, (D,), W["ln_post"][0], W["ln_post"][1], 1e-5)
    xq, xs = fso.dense.linear_quantized(x, W["p1"][2], bias=W["p1"][1], activation="gelu_erf")
    return fso.dense.linear(xq, W["p2"][2], x_scale=xs, bias=W["p2"][1])


def run_fso030_fused_qkv(x_conv, W, attn_fn):
    x = fso.dense.linear(x_conv, W["conv"][1])
    M, D = x.shape
    for L in W["layers"]:
        h = F.layer_norm(x, (D,), L["ln1"][0], L["ln1"][1], 1e-5)
        qkv = fso.dense.linear(h, L["dqkv"], bias=L["qkv"][1])
        q, k, v = qkv.view(M, 3, 20, 64).unbind(1)
        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)
        a = attn_fn(q, k, v).transpose(1, 2).reshape(M, D)
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


def time_graphs(graphs, iters=30, rounds=3):
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.25:
        a = a @ a * 1e-3
    torch.cuda.synchronize()
    res = {k: [] for k in graphs}
    for r in range(rounds):
        keys = list(graphs.keys()) if r % 2 == 0 else list(reversed(graphs.keys()))
        for k in keys:
            g = graphs[k]
            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            st.record()
            for _ in range(iters):
                g.replay()
            en.record()
            torch.cuda.synchronize()
            res[k].append(st.elapsed_time(en) * 1000.0 / iters)
    return {k: sorted(v)[len(v) // 2] for k, v in res.items()}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    W = make_tower_weights()
    if mode == "nsys":
        M = int(sys.argv[2])
        arm = sys.argv[3]
        x = torch.randn(M, 7680, device="cuda", dtype=torch.bfloat16) * 0.5
        fn = {"bf16": run_bf16, "unfused": run_fso030_unfused, "fused_qkv": run_fso030_fused_qkv}[arm]
        g, _ = capture(lambda: fn(x, W, cudnn_attn))
        torch.cuda.nvtx.range_push(f"replay_{arm}_M{M}")
        g.replay()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    elif mode == "attn_bench":
        from flash_attn.cute import flash_attn_func as fa4_func
        print(f"{'M':>5} | {'cuDNN us':>9} | {'Flash2 us':>9} | {'FA4 us':>9} | {'FA4/cuDNN':>9}")
        print("-" * 55)
        for M in [78, 117, 182, 364, 533, 1079]:
            qkv = torch.randn(M, 3840, device="cuda", dtype=torch.bfloat16) * 0.5
            q, k, v = qkv.view(M, 3, 20, 64).unbind(1)
            q_bhmd = q.unsqueeze(0).transpose(1, 2)
            k_bhmd = k.unsqueeze(0).transpose(1, 2)
            v_bhmd = v.unsqueeze(0).transpose(1, 2)
            q_bmhd = q.unsqueeze(0)
            k_bmhd = k.unsqueeze(0)
            v_bmhd = v.unsqueeze(0)
            # Repeat 32 times per graph to mirror 32 layers cleanly
            def r_cudnn():
                o = None
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
                    for _ in range(32):
                        o = F.scaled_dot_product_attention(q_bhmd, k_bhmd, v_bhmd)
                return o
            def r_flash():
                o = None
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                    for _ in range(32):
                        o = F.scaled_dot_product_attention(q_bhmd, k_bhmd, v_bhmd)
                return o
            def r_fa4():
                o = None
                for _ in range(32):
                    o, _ = fa4_func(q_bmhd, k_bmhd, v_bmhd)
                return o
            gc, _ = capture(r_cudnn)
            gf, _ = capture(r_flash)
            g4, _ = capture(r_fa4)
            t = time_graphs({"cudnn": gc, "flash": gf, "fa4": g4}, iters=50)
            # Report per-call us (divide by 32)
            tc, tf, t4 = t["cudnn"] / 32.0, t["flash"] / 32.0, t["fa4"] / 32.0
            print(f"{M:>5} | {tc:>9.2f} | {tf:>9.2f} | {t4:>9.2f} | {t4/tc:>8.2f}x")
    else:
        Ms = [78, 117, 182, 364, 533, 1079]
        print(f"{'M':>5} | {'bf16 us':>9} | {'0.3.0 unfused':>13} | {'0.3.0 fqkv':>11} | {'fqkv vs bf16':>12}")
        print("-" * 62)
        for M in Ms:
            x = torch.randn(M, 7680, device="cuda", dtype=torch.bfloat16) * 0.5
            g1, _ = capture(lambda: run_bf16(x, W, cudnn_attn))
            g2, _ = capture(lambda: run_fso030_unfused(x, W, cudnn_attn))
            g3, _ = capture(lambda: run_fso030_fused_qkv(x, W, cudnn_attn))
            t = time_graphs({"bf16": g1, "unfused": g2, "fqkv": g3})
            pct = (t["fqkv"] / t["bf16"] - 1.0) * 100.0
            print(f"{M:>5} | {t['bf16']:>9.1f} | {t['unfused']:>13.1f} | {t['fqkv']:>11.1f} | {pct:>+11.1f}%")
