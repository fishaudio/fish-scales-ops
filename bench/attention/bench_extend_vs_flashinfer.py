"""Extend (paged prefill) microbench: fish-scales-ops MXFP8 vs flashinfer BF16.

Compares the SAME extend workload on:
  - fish_scales_ops.attention.backends.sm120_mxfp8_paged_prefill (this lib, MXFP8 + paged)
  - flashinfer.BatchPrefillWithPagedKVCacheWrapper (BF16 baseline, paged)

Reports per-shape latency (CUDA Event timed, median of N replays) and
effective TFLOPS (assuming 4 * total_q * sum_kv_history * H_q * D flops
per causal extend, ignoring the lower-triangular factor of 2 since the
flashinfer kernel exploits causal early-exit while ours does too).

Workload: sglang-style extend, mixed ragged S_q per batch entry.
GPU: Blackwell sm_120a (170 SMs). page_size = 32 (MXFP8 sf_vec_size lower bound).
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

import flashinfer
from fish_scales_ops.attention.backends import sm120_mxfp8_decode as dec_bk
from fish_scales_ops.attention.backends import sm120_mxfp8_paged_prefill as pp_bk


# ----------------------------- shape definitions ---------------------------- #

@dataclass
class ExtendShape:
    name: str          # short label
    H_q: int
    H_kv: int
    D: int
    q_lens: List[int]   # per-batch S_q (ragged)
    kv_histories: List[int]   # per-batch prefix length (kv_seq_len = history + S_q)


def _seq(name, H_q, H_kv, D, q_lens, kv_hist) -> ExtendShape:
    return ExtendShape(name, H_q, H_kv, D, q_lens, kv_hist)


# Representative sglang-style extend cells: one batch entry per request.
# - "decode-only" (S_q=1): the most common per-token extend.
# - "small extend": chunked prefill ~64 tokens with growing prefix.
# - "large extend": full prefill of a fresh sequence (no prefix).
SHAPES: List[ExtendShape] = [
    _seq("Llama3-8B  decode B=8 kv=1024",  32, 8, 128, [1]*8,  [1024]*8),
    _seq("Llama3-8B  decode B=8 kv=8192",  32, 8, 128, [1]*8,  [8192]*8),
    _seq("Llama3-8B  extend64 B=4 kv=1024",32, 8, 128, [64]*4, [1024]*4),
    _seq("Llama3-8B  extend256 B=4 kv=4096",32,8, 128, [256]*4,[4096]*4),
    _seq("Llama3-8B  prefill B=2 S=2048",  32, 8, 128, [2048]*2,[0]*2),
    _seq("Llama3-8B  prefill B=1 S=8192",  32, 8, 128, [8192], [0]),
    _seq("Qwen3-32B  decode B=8 kv=4096",  32, 8, 128, [1]*8,  [4096]*8),
    _seq("Qwen3-32B  extend64 B=8 kv=4096",32, 8, 128, [64]*8, [4096]*8),
    _seq("Qwen2-72B  decode B=8 kv=4096",  64, 8, 128, [1]*8,  [4096]*8),
    _seq("Qwen2-72B  extend64 B=8 kv=4096",64, 8, 128, [64]*8, [4096]*8),
    _seq("Qwen3-32B  mixed B=4",           32, 8, 128, [1, 17, 64, 256], [1024, 2048, 4096, 4096]),
    # D=256 paths — dedicated dispatch (Bc=32, 1 CTA/SM for prefill / paged-prefill;
    # kStages=1 for decode). H_q/H_kv mimic a hypothetical large-head GQA model.
    _seq("D=256      decode B=8 kv=4096",   16, 4, 256, [1]*8,   [4096]*8),
    _seq("D=256      extend64 B=4 kv=4096", 16, 4, 256, [64]*4,  [4096]*4),
    _seq("D=256      prefill B=1 S=4096",   16, 4, 256, [4096],  [0]),
    _seq("D=256      prefill B=1 S=8192",   16, 4, 256, [8192],  [0]),
]


# ----------------------------- helpers ------------------------------------- #

def _build_qo_indptr(q_lens, device):
    qo = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device=device)
    qo[1:] = torch.tensor(q_lens, dtype=torch.int32, device=device).cumsum(0)
    return qo


def _build_paged_kv_bf16(k_per_b: List[torch.Tensor], v_per_b: List[torch.Tensor],
                          page_size: int, device):
    """flashinfer-style BF16 paged KV cache.

    K, V cache shape: [num_pages, page_size, num_kv_heads, head_dim]  bf16
    """
    H_kv = k_per_b[0].size(1)
    D    = k_per_b[0].size(2)
    pages_per_b = []
    last_lens = []
    K_pages, V_pages = [], []
    for kb, vb in zip(k_per_b, v_per_b):
        S = kb.size(0)
        n_pages = max((S + page_size - 1) // page_size, 1)
        S_pad   = n_pages * page_size
        if S_pad > S:
            kpad = torch.zeros(S_pad - S, H_kv, D,
                                dtype=kb.dtype, device=device)
            vpad = torch.zeros(S_pad - S, H_kv, D,
                                dtype=vb.dtype, device=device)
            k_full = torch.cat([kb, kpad], dim=0)
            v_full = torch.cat([vb, vpad], dim=0)
        else:
            k_full = kb
            v_full = vb
        last_lens.append(S - (n_pages - 1) * page_size if S > 0 else 0)
        K_pages.append(k_full.reshape(n_pages, page_size, H_kv, D))
        V_pages.append(v_full.reshape(n_pages, page_size, H_kv, D))
        pages_per_b.append(n_pages)
    K = torch.cat(K_pages, dim=0).contiguous()
    V = torch.cat(V_pages, dim=0).contiguous()
    paged_kv_indices = torch.arange(K.size(0), dtype=torch.int32, device=device)
    paged_kv_indptr  = torch.zeros(len(pages_per_b) + 1, dtype=torch.int32, device=device)
    paged_kv_indptr[1:] = torch.tensor(pages_per_b, dtype=torch.int32, device=device).cumsum(0)
    last_page_len = torch.tensor(last_lens, dtype=torch.int32, device=device)
    return K, V, paged_kv_indices, paged_kv_indptr, last_page_len


def _build_paged_kv_mxfp8(k_per_b: List[torch.Tensor], v_per_b: List[torch.Tensor],
                           page_size: int, device):
    """Channel-V MXFP8 paged KV cache (K block-scaled per page, V channel-scaled global)."""
    H_kv = k_per_b[0].size(1)
    D    = k_per_b[0].size(2)
    pages_per_b = []
    last_lens   = []
    padded_K, padded_V = [], []
    for kb, vb in zip(k_per_b, v_per_b):
        S = kb.size(0)
        n_pages = max((S + page_size - 1) // page_size, 1)
        S_pad   = n_pages * page_size
        if S_pad > S:
            kpad = torch.zeros(S_pad - S, H_kv, D, dtype=kb.dtype, device=device)
            vpad = torch.zeros(S_pad - S, H_kv, D, dtype=vb.dtype, device=device)
            k_full = torch.cat([kb, kpad], dim=0).reshape(1, S_pad, H_kv, D)
            v_full = torch.cat([vb, vpad], dim=0).reshape(1, S_pad, H_kv, D)
        else:
            k_full = kb.reshape(1, S_pad, H_kv, D)
            v_full = vb.reshape(1, S_pad, H_kv, D)
        padded_K.append(k_full)
        padded_V.append(v_full)
        pages_per_b.append(n_pages)
        last_lens.append(S - (n_pages - 1) * page_size if S > 0 else 0)

    # Global K + V channel scales across all batches.
    all_K_flat = torch.cat([kf.reshape(-1, H_kv, D) for kf in padded_K], dim=0)
    all_V_flat = torch.cat([vf.reshape(-1, H_kv, D) for vf in padded_V], dim=0)
    k_chan_scale = dec_bk.compute_k_chan_scale(all_K_flat)
    v_chan_scale = dec_bk.compute_v_chan_scale(all_V_flat)

    K_caches, V_caches = [], []
    for kf, vf in zip(padded_K, padded_V):
        Kq, _, Vq, _ = dec_bk.quantize_kv_to_paged(
            kf, vf, page_size,
            k_chan_scale=k_chan_scale, v_chan_scale=v_chan_scale)
        K_caches.append(Kq.squeeze(0))
        V_caches.append(Vq.squeeze(0))
    k_pool    = torch.cat(K_caches, dim=0).contiguous()
    v_pool    = torch.cat(V_caches, dim=0).contiguous()
    paged_kv_indices = torch.arange(k_pool.size(0), dtype=torch.int32, device=device)
    paged_kv_indptr  = torch.zeros(len(pages_per_b) + 1, dtype=torch.int32, device=device)
    paged_kv_indptr[1:] = torch.tensor(pages_per_b, dtype=torch.int32, device=device).cumsum(0)
    last_page_len = torch.tensor(last_lens, dtype=torch.int32, device=device)
    return (k_pool, k_chan_scale, v_pool, v_chan_scale,
            paged_kv_indices, paged_kv_indptr, last_page_len)


def _causal_flops(q_lens, kv_histories, H_q, D) -> float:
    """4 * (Σ_b Σ_q (kv_offset_b + q + 1) * H_q * D) flops (causal extend)."""
    flops = 0.0
    for q_len, kv_hist in zip(q_lens, kv_histories):
        for q in range(q_len):
            kv_cols = kv_hist + q + 1
            flops += 4.0 * kv_cols * H_q * D       # QK + PV
    return flops


# ------------------------------- timing ------------------------------------ #

def _cuda_event_time_ms(fn, n_warmup=3, n_iter=15) -> float:
    """Return median over n_iter of single-call latency in ms."""
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start_evts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    stop_evts  = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        start_evts[i].record()
        fn()
        stop_evts[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_evts, stop_evts)]
    times.sort()
    return times[len(times) // 2]


# ------------------------------ bench cell --------------------------------- #

def bench_one(shape: ExtendShape, page_size: int, *, device: torch.device, seed: int):
    torch.manual_seed(seed)
    B = len(shape.q_lens)
    H_q, H_kv, D = shape.H_q, shape.H_kv, shape.D
    q_lens = shape.q_lens
    kv_hist = shape.kv_histories
    total_q = sum(q_lens)
    total_kv = [q + h for q, h in zip(q_lens, kv_hist)]
    # Decode path is triggered when every batch entry has S_q == 1 (sglang
    # decode phase). Otherwise we use the extend/prefill path. This mirrors
    # what FlashInferAttnBackend does in sglang.
    is_decode_only = all(q == 1 for q in q_lens)

    # Build inputs (BF16 ground truth).
    q_list = [torch.randn(q_lens[b], H_q, D, dtype=torch.bfloat16, device=device) * 0.5
              for b in range(B)]
    k_list = [torch.randn(total_kv[b], H_kv, D, dtype=torch.bfloat16, device=device) * 0.5
              for b in range(B)]
    v_list = [torch.randn(total_kv[b], H_kv, D, dtype=torch.bfloat16, device=device) * 0.5
              for b in range(B)]
    q_full = torch.cat(q_list, dim=0).contiguous()    # [total_q, H_q, D]
    qo_indptr = _build_qo_indptr(q_lens, device)
    scale = 1.0 / math.sqrt(D)

    flops = _causal_flops(q_lens, kv_hist, H_q, D)

    # --- flashinfer BF16 ----------------------------------------------------
    K_bf16, V_bf16, indices_fi, indptr_fi, last_page_len_fi = _build_paged_kv_bf16(
        k_list, v_list, page_size, device)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    paged_kv = (K_bf16, V_bf16)

    if is_decode_only:
        # flashinfer's BatchDecode path.
        wrapper_fi = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
        wrapper_fi.plan(
            indptr_fi, indices_fi, last_page_len_fi,
            num_qo_heads=H_q, num_kv_heads=H_kv,
            head_dim=D, page_size=page_size,
            sm_scale=scale,
            q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
        # BatchDecode expects Q in [B, H_q, D].
        q_decode = torch.stack([q_list[b][0] for b in range(B)], dim=0).contiguous()
        def _flashinfer_run():
            return wrapper_fi.run(q_decode, paged_kv)
    else:
        wrapper_fi = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
        wrapper_fi.plan(
            qo_indptr, indptr_fi, indices_fi, last_page_len_fi,
            num_qo_heads=H_q, num_kv_heads=H_kv,
            head_dim_qk=D, page_size=page_size,
            causal=True, sm_scale=scale,
            q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
        def _flashinfer_run():
            return wrapper_fi.run(q_full, paged_kv)
    out_fi = _flashinfer_run()
    t_fi = _cuda_event_time_ms(_flashinfer_run)

    # --- fish-scales-ops MXFP8 ---------------------------------------------
    (k_pool, k_chan_scale, v_pool, v_chan_scale,
     indices_fso, indptr_fso, last_page_len_fso) = _build_paged_kv_mxfp8(
        k_list, v_list, page_size, device)
    if is_decode_only:
        # Decode-paged is single-token-per-batch. block_table is [B, max_blocks].
        max_pages_per_b = max(
            int((indptr_fso[b + 1] - indptr_fso[b]).item()) for b in range(B))
        block_table = torch.zeros(B, max_pages_per_b, dtype=torch.int32, device=device)
        for b in range(B):
            n = int((indptr_fso[b + 1] - indptr_fso[b]).item())
            block_table[b, :n] = indices_fso[int(indptr_fso[b].item()):
                                              int(indptr_fso[b + 1].item())]
        # seq_lens: full KV length per batch (including history + the 1 new token).
        seq_lens = torch.tensor(total_kv, dtype=torch.int32, device=device)
        q_decode_bf16 = torch.stack([q_list[b][0] for b in range(B)], dim=0).contiguous()
        q_fp8_dec, q_sc_dec = dec_bk.quantize_q_grouped(q_decode_bf16, H_kv)
        plan = dec_bk.plan_decode_paged(
            B=B, H_q=H_q, H_kv=H_kv, D=D,
            max_blocks=max_pages_per_b, device=device)
        def _fso_run():
            return plan.run(q_fp8_dec, q_sc_dec, k_pool, k_chan_scale, v_pool, v_chan_scale,
                            block_table, seq_lens, softmax_scale=scale)
    else:
        q_fp8, q_sc = pp_bk.quantize_q_ragged(q_full, H_q)
        plan = pp_bk.plan_paged_prefill(
            qo_indptr_cpu=qo_indptr.detach().cpu(), num_q_heads=H_q, device=device)
        def _fso_run():
            return plan.run(q_fp8, q_sc, k_pool, k_chan_scale, v_pool, v_chan_scale,
                            qo_indptr, indices_fso, indptr_fso, last_page_len_fso,
                            softmax_scale=scale, causal=True)
    out_fso = _fso_run()
    t_fso = _cuda_event_time_ms(_fso_run)

    # Cos similarity vs flashinfer (sanity).
    cos = F.cosine_similarity(out_fso.float().flatten(),
                              out_fi.float().flatten(), dim=0).item()

    tflops_fi  = flops / (t_fi  * 1e-3) / 1e12
    tflops_fso = flops / (t_fso * 1e-3) / 1e12
    return {
        "shape": shape.name,
        "mode":  "decode" if is_decode_only else "extend",
        "B": B, "total_q": total_q,
        "t_fi_us":  t_fi  * 1e3,
        "t_fso_us": t_fso * 1e3,
        "tflops_fi":  tflops_fi,
        "tflops_fso": tflops_fso,
        "speedup":    t_fi / t_fso,
        "cos":        cos,
    }


# ------------------------------- main -------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-size", type=int, default=32,
                    help="MXFP8 minimum is 32 (sf_vec_size).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f"# Device: {name} (sm_{cap[0]}{cap[1]})")
    print(f"# page_size={args.page_size}")
    print()
    header = (f"{'Shape':40s} {'mode':>6s} {'B':>3s} {'tQ':>6s}"
              f" {'FI µs':>8s} {'FSO µs':>8s}"
              f" {'FI TF':>7s} {'FSO TF':>7s}"
              f" {'speedup':>7s} {'cos':>6s}")
    print(header)
    print("-" * len(header))
    device = torch.device("cuda")
    for shape in SHAPES:
        try:
            r = bench_one(shape, args.page_size, device=device, seed=args.seed)
            print(f"{r['shape']:40s} {r['mode']:>6s} {r['B']:>3d} {r['total_q']:>6d}"
                  f" {r['t_fi_us']:>8.1f} {r['t_fso_us']:>8.1f}"
                  f" {r['tflops_fi']:>7.1f} {r['tflops_fso']:>7.1f}"
                  f" {r['speedup']:>6.2f}x {r['cos']:>6.4f}")
        except Exception as e:
            print(f"{shape.name:40s}  FAIL: {str(e)[:80]}")


if __name__ == "__main__":
    main()
