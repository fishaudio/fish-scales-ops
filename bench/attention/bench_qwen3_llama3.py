"""MXFP8 attention bench against torch SDPA on production GQA shapes.

Covers prefill (mxfp8_attn_fwd) and decode (mxfp8_decode_paged) on the
Llama-3 / Qwen2 / Qwen3 GQA configurations. CUDA-Graph capture+replay
timing — same protocol as bench/gemm/python/bench_qwen3_4b_mlp.py.

Reproduce:
    python bench/attention/bench_qwen3_llama3.py --out runs/attn-<tag>.jsonl

Default models (H_q / H_kv / D):
    Llama3-8B    32 /  8 / 128   gqa=4
    Llama3-70B   64 /  8 / 128   gqa=8
    Qwen3-4B     32 /  8 / 128   gqa=4
    Qwen3-32B    64 /  8 / 128   gqa=8
    Qwen2-72B    64 /  8 / 128   gqa=8
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from typing import Optional

import torch
import torch.nn.functional as F

import fish_scales_ops as fso  # noqa: F401  loads _C, registers torch.ops
from fish_scales_ops.attention.backends import sm120_mxfp8 as fwd_bk
from fish_scales_ops.attention.backends import sm120_mxfp8_decode as dec_bk


MODELS = [
    # (name, H_q, H_kv, D)
    ("Llama3-8B",  32, 8, 128),
    ("Llama3-70B", 64, 8, 128),
    ("Qwen3-4B",   32, 8, 128),
    ("Qwen3-32B",  64, 8, 128),
    ("Qwen2-72B",  64, 8, 128),
    # D=256 — dedicated dispatch (Bc=32, 1 CTA/SM for prefill; kStages=1 decode).
    ("D=256-GQA4", 16, 4, 256),
]


def _inject_outliers(t: torch.Tensor, *, channel_dim: int = -1,
                     channel_frac: float = 0.05, channel_scale: float = 20.0,
                     element_frac: float = 0.003, element_scale: float = 8.0,
                     seed: int = 0) -> torch.Tensor:
    """Stress MXFP8 1×32 quantization by injecting realistic outliers.

    Pattern (tuned to push MXFP8 cos meaningfully below the pure-Gaussian
    floor; mirrors the SmoothQuant retro observation that LLM activation
    outliers concentrate in a few consistent D-channels at 10-100× the
    bulk amax):
      * channel outliers: ``channel_frac`` of indices along ``channel_dim``
        get their slice multiplied by ``channel_scale``.
      * sparse element outliers: ~``element_frac`` of all entries scaled
        by ``element_scale``.
    Deterministic via the supplied seed.
    """
    g = torch.Generator(device=t.device).manual_seed(seed)
    if channel_dim < 0:
        channel_dim += t.dim()
    K = t.size(channel_dim)
    n_channels = max(1, int(K * channel_frac))
    chan_ids = torch.randperm(K, generator=g, device=t.device)[:n_channels]
    idx = [slice(None)] * t.dim()
    idx[channel_dim] = chan_ids
    t = t.clone()
    t[tuple(idx)] = (t[tuple(idx)].float() * channel_scale).to(t.dtype)
    mask = torch.rand(t.shape, generator=g, device=t.device) < element_frac
    t = torch.where(mask, (t.float() * element_scale).to(t.dtype), t)
    return t


def time_fn_graph(fn, iters: int = 50, warmup: int = 15, repeats: int = 3) -> float:
    """CUDA-Graph capture-and-replay timing. Returns median µs / iter.

    Matches the GEMM bench protocol: eager warmup populates static caches
    (TMA descriptors, autotuner state, pool allocations), then side-stream
    warmup, then capture once + replay `iters` × `repeats` reps; report the
    median across repeats.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()

    samples = []
    for _ in range(repeats):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        samples.append(e0.elapsed_time(e1) / iters)
    return statistics.median(samples) * 1000.0   # ms → µs


def bench_prefill(B: int, S: int, H_q: int, H_kv: int, D: int, causal: bool,
                  inject_outliers: bool = False):
    """Returns (mxfp8_us, mxfp8_tf, sdpa_us, sdpa_tf, cos)."""
    torch.manual_seed(B * 31 + S * 13 + H_q + D)
    q = torch.randn(B, S, H_q,  D, dtype=torch.bfloat16, device="cuda") * 0.5
    k = torch.randn(B, S, H_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
    v = torch.randn(B, S, H_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
    if inject_outliers:
        q = _inject_outliers(q, channel_dim=-1, seed=S * 31 + H_q)
        k = _inject_outliers(k, channel_dim=-1, seed=S * 31 + H_kv + 11)
        v = _inject_outliers(v, channel_dim=-1, seed=S * 31 + H_kv + 23)
    softmax_scale = 1.0 / math.sqrt(D)

    q_fp8, q_sc = fwd_bk.pre_quantize_q(q)
    k_fp8, k_sc = fwd_bk.pre_quantize_k(k)
    v_fp8, v_sc = fwd_bk.pre_quantize_v(v)

    def mxfp8_call():
        return fwd_bk.mxfp8_fwd(q_fp8, q_sc, k_fp8, k_sc, v_fp8, v_sc,
                                softmax_scale=softmax_scale, causal=causal)

    rep = H_q // H_kv
    kE = k.repeat_interleave(rep, dim=2)
    vE = v.repeat_interleave(rep, dim=2)
    qH = q.transpose(1, 2)
    kH = kE.transpose(1, 2)
    vH = vE.transpose(1, 2)

    def sdpa_call():
        return F.scaled_dot_product_attention(
            qH, kH, vH, is_causal=causal, scale=softmax_scale)

    # Reference cos (eager outputs).
    mxfp8_out = mxfp8_call()
    sdpa_out  = sdpa_call().transpose(1, 2).contiguous()
    cos = F.cosine_similarity(mxfp8_out.float().flatten(),
                              sdpa_out.float().flatten(), dim=0).item()

    mxfp8_us = time_fn_graph(mxfp8_call)
    sdpa_us  = time_fn_graph(sdpa_call)

    # FLOPs: 4 · B · S² · H_q · D (QK^T → softmax · V). Halve for causal.
    flops = 4.0 * B * S * S * H_q * D
    if causal:
        flops *= 0.5
    return mxfp8_us, flops / (mxfp8_us * 1e6), sdpa_us, flops / (sdpa_us * 1e6), cos


def bench_decode(B: int, KV: int, H_q: int, H_kv: int, D: int,
                 inject_outliers: bool = False):
    """Single-token paged-KV decode. Returns (mxfp8_us, mxfp8_tf, sdpa_us, sdpa_tf, cos)."""
    page_size = 64 if D in (64, 128) else 128
    assert KV % page_size == 0, f"KV={KV} must be a multiple of page_size={page_size}"
    num_pages = KV // page_size

    torch.manual_seed(B * 31 + KV * 13 + H_q + D)
    q       = torch.randn(B, H_q,  D, dtype=torch.bfloat16, device="cuda") * 0.5
    k_full  = torch.randn(B, KV, H_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
    v_full  = torch.randn(B, KV, H_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
    if inject_outliers:
        q      = _inject_outliers(q,      channel_dim=-1, seed=KV * 31 + H_q)
        k_full = _inject_outliers(k_full, channel_dim=-1, seed=KV * 31 + H_kv + 11)
        v_full = _inject_outliers(v_full, channel_dim=-1, seed=KV * 31 + H_kv + 23)
    softmax_scale = 1.0 / math.sqrt(D)

    q_fp8, q_sc = dec_bk.quantize_q_grouped(q, H_kv)
    # Channel-K + Channel-V: K/V scales are GLOBAL per (h_kv, D[/32]),
    # not per-page. quantize_kv_to_paged returns them in the new layout.
    K_cache, k_chan_scale, V_cache, v_chan_scale = dec_bk.quantize_kv_to_paged(
        k_full, v_full, page_size)
    K_pool = K_cache.reshape(B * num_pages, page_size, H_kv, D).contiguous()
    V_pool = V_cache.reshape(B * num_pages, D, H_kv, page_size).contiguous()
    block_table = torch.arange(B * num_pages, dtype=torch.int32,
                               device="cuda").reshape(B, num_pages)
    seq_lens = torch.full((B,), KV, dtype=torch.int32, device="cuda")

    plan = dec_bk.plan_decode_paged(B=B, H_q=H_q, H_kv=H_kv, D=D,
                                    max_blocks=num_pages)
    out_buf = torch.empty(B, H_q, D, dtype=torch.bfloat16, device="cuda")

    def mxfp8_call():
        return plan.run(q_fp8, q_sc, K_pool, k_chan_scale, V_pool, v_chan_scale,
                         block_table, seq_lens,
                         softmax_scale=softmax_scale, out=out_buf)

    rep = H_q // H_kv
    kE = k_full.repeat_interleave(rep, dim=2)
    vE = v_full.repeat_interleave(rep, dim=2)
    q_4d = q.unsqueeze(1)                                # [B, 1, H_q, D]
    qH = q_4d.transpose(1, 2)
    kH = kE.transpose(1, 2)
    vH = vE.transpose(1, 2)

    def sdpa_call():
        return F.scaled_dot_product_attention(qH, kH, vH, scale=softmax_scale)

    mxfp8_out = mxfp8_call()
    sdpa_out  = sdpa_call().transpose(1, 2).squeeze(1).contiguous()
    cos = F.cosine_similarity(mxfp8_out.float().flatten(),
                              sdpa_out.float().flatten(), dim=0).item()

    mxfp8_us = time_fn_graph(mxfp8_call)
    sdpa_us  = time_fn_graph(sdpa_call)

    # FLOPs: 4 · B · KV · H_q · D (QK + softmax·V over the full KV).
    flops = 4.0 * B * KV * H_q * D
    return mxfp8_us, flops / (mxfp8_us * 1e6), sdpa_us, flops / (sdpa_us * 1e6), cos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="-", help="JSONL output path or - for stdout-only")
    p.add_argument("--mode", choices=("prefill", "decode", "both"), default="both")
    p.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    p.add_argument("--seq-prefill", type=int, nargs="+",
                    default=[1024, 2048, 4096, 8192])
    p.add_argument("--seq-decode", type=int, nargs="+",
                    default=[1024, 4096, 16384])
    p.add_argument("--non-causal", action="store_true",
                    help="prefill: bench non-causal instead of causal")
    p.add_argument("--outliers", action="store_true",
                    help="inject channel + sparse outliers to stress MXFP8 cos")
    p.add_argument("--models", nargs="+", default=None,
                    help="filter MODELS list by name substring")
    args = p.parse_args()

    cap = torch.cuda.get_device_capability(0)
    dev = torch.cuda.get_device_name(0)
    sm = cap[0] * 10 + cap[1]
    print(f"Device: {dev}  sm_{sm}\n")

    models = MODELS
    if args.models:
        models = [m for m in MODELS if any(s in m[0] for s in args.models)]

    f = open(args.out, "w") if args.out != "-" else None
    if f:
        f.write(json.dumps({"_device": dev, "_sm": sm,
                            "_outliers": args.outliers}) + "\n")
        f.flush()
    if args.outliers:
        print("(outlier injection: 5% channels × 20, 0.3% elements × 8)\n")

    causal = not args.non_causal

    if args.mode in ("prefill", "both"):
        tag = "causal" if causal else "non-causal"
        print(f"== Prefill ({tag}) ==")
        print(f"{'model':<12s} {'B':>3s} {'S':>5s} {'H_q':>4s} {'gqa':>4s}  "
              f"{'mxfp8 µs':>10s} {'mxfp8 TF':>9s}  "
              f"{'sdpa µs':>9s} {'sdpa TF':>8s}  "
              f"{'speedup':>7s}  {'cos':>6s}")
        for name, h_q, h_kv, D in models:
            for B in args.batch:
                for S in args.seq_prefill:
                    try:
                        mu, mt, su, st, c = bench_prefill(
                            B, S, h_q, h_kv, D, causal,
                            inject_outliers=args.outliers)
                    except Exception as e:
                        print(f"{name:<12s} {B:>3d} {S:>5d} {h_q:>4d} {h_q//h_kv:>4d}  "
                              f"FAIL  {type(e).__name__}: {str(e)[:60]}")
                        continue
                    print(f"{name:<12s} {B:>3d} {S:>5d} {h_q:>4d} {h_q//h_kv:>4d}  "
                          f"{mu:>10.1f} {mt:>9.0f}  {su:>9.1f} {st:>8.0f}  "
                          f"{su/mu:>6.2f}x  {c:>6.4f}")
                    if f:
                        f.write(json.dumps({
                            "mode": "prefill", "model": name, "B": B, "S": S,
                            "H_q": h_q, "H_kv": h_kv, "D": D,
                            "causal": causal,
                            "mxfp8_us": mu, "mxfp8_tf": mt,
                            "sdpa_us": su, "sdpa_tf": st,
                            "speedup": su / mu, "cos": c,
                        }) + "\n")
                        f.flush()
        print()

    if args.mode in ("decode", "both"):
        print("== Decode (paged, S_q=1) ==")
        print(f"{'model':<12s} {'B':>3s} {'KV':>6s} {'H_q':>4s} {'gqa':>4s}  "
              f"{'mxfp8 µs':>10s} {'mxfp8 TF':>9s}  "
              f"{'sdpa µs':>9s} {'sdpa TF':>8s}  "
              f"{'speedup':>7s}  {'cos':>6s}")
        for name, h_q, h_kv, D in models:
            for B in args.batch:
                for KV in args.seq_decode:
                    try:
                        mu, mt, su, st, c = bench_decode(
                            B, KV, h_q, h_kv, D,
                            inject_outliers=args.outliers)
                    except Exception as e:
                        print(f"{name:<12s} {B:>3d} {KV:>6d} {h_q:>4d} {h_q//h_kv:>4d}  "
                              f"FAIL  {type(e).__name__}: {str(e)[:60]}")
                        continue
                    print(f"{name:<12s} {B:>3d} {KV:>6d} {h_q:>4d} {h_q//h_kv:>4d}  "
                          f"{mu:>10.1f} {mt:>9.0f}  {su:>9.1f} {st:>8.0f}  "
                          f"{su/mu:>6.2f}x  {c:>6.4f}")
                    if f:
                        f.write(json.dumps({
                            "mode": "decode", "model": name, "B": B, "KV": KV,
                            "H_q": h_q, "H_kv": h_kv, "D": D,
                            "mxfp8_us": mu, "mxfp8_tf": mt,
                            "sdpa_us": su, "sdpa_tf": st,
                            "speedup": su / mu, "cos": c,
                        }) + "\n")
                        f.flush()
        print()

    if f:
        f.close()


if __name__ == "__main__":
    main()
