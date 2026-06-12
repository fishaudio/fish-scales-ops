"""SM120 MXFP8 paged-KV prefill (extend mode) correctness test.

Two layers:

1. **Dispatch + shape validation** (runs everywhere) — verifies the wrapper
   shape checks fire and the kernel is reached on sm < 12 (returning
   cudaErrorNotSupported via the Python op binding).

2. **Numerical correctness vs SDPA** (skipped on sm < 12) — re-quantises
   BF16 reference KV, runs the kernel, compares against a torch SDPA
   reference with causal + kv_offset_history.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import fish_scales_ops as fso
from fish_scales_ops.attention.backends import sm120_mxfp8_decode as dec_bk
from fish_scales_ops.attention.backends import sm120_mxfp8_paged_prefill as pp_bk


def _is_sm120() -> bool:
    if not torch.cuda.is_available():
        return False
    maj, _ = torch.cuda.get_device_capability()
    return maj >= 12


def _build_paged_kv(k_bf16_list, v_bf16_list, page_size: int, *, device):
    """Pack a list of per-batch [S, H_kv, D] KV tensors into the paged pool.

    V uses a single GLOBAL channel scale across all batches (max over all
    batches' V values per (h_kv, d)). This matches the production model:
    V_chan_scale is calibrated once and frozen for the entire KV cache.

    Returns (k_pool, k_scales, v_pool, v_chan_scale, paged_kv_indices,
             paged_kv_indptr, paged_kv_last_page_len). Pages are laid out
             contiguously by batch.
    """
    H_kv = k_bf16_list[0].size(1)
    D    = k_bf16_list[0].size(2)
    assert page_size % 32 == 0

    # First pass: pad each batch to page_size multiple, record pages_per_b.
    last_lens = []
    padded_K = []
    padded_V = []
    pages_per_b = []
    for k_bf16, v_bf16 in zip(k_bf16_list, v_bf16_list):
        S = k_bf16.size(0)
        n_pages = (S + page_size - 1) // page_size
        S_pad = n_pages * page_size
        if S_pad > S:
            k_pad = torch.zeros(S_pad - S, H_kv, D,
                                dtype=k_bf16.dtype, device=device)
            v_pad = torch.zeros(S_pad - S, H_kv, D,
                                dtype=v_bf16.dtype, device=device)
            k_full = torch.cat([k_bf16, k_pad], dim=0)
            v_full = torch.cat([v_bf16, v_pad], dim=0)
        else:
            k_full = k_bf16
            v_full = v_bf16
        last_lens.append(S - (n_pages - 1) * page_size if S > 0 else 0)
        padded_K.append(k_full.reshape(1, S_pad, H_kv, D))
        padded_V.append(v_full.reshape(1, S_pad, H_kv, D))
        pages_per_b.append(n_pages)

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

    k_pool       = torch.cat(K_caches, dim=0).contiguous()       # [P, page_size, H_kv, D]
    v_pool       = torch.cat(V_caches, dim=0).contiguous()       # [P, D, H_kv, page_size]
    total_P = k_pool.size(0)

    paged_kv_indices = torch.arange(total_P, dtype=torch.int32, device=device)
    paged_kv_indptr  = torch.zeros(len(pages_per_b) + 1, dtype=torch.int32, device=device)
    paged_kv_indptr[1:] = torch.tensor(pages_per_b, dtype=torch.int32, device=device).cumsum(0)
    last_page_len = torch.tensor(last_lens, dtype=torch.int32, device=device)
    return k_pool, k_chan_scale, v_pool, v_chan_scale, paged_kv_indices, paged_kv_indptr, last_page_len


def _build_qo_indptr(q_lens, device):
    qo = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device=device)
    qo[1:] = torch.tensor(q_lens, dtype=torch.int32, device=device).cumsum(0)
    return qo


def _ref_sdpa_extend(q_bf16_list, k_full_list, v_full_list, softmax_scale, causal):
    """Reference: per-batch SDPA with kv_offset_history = kv_len - q_len.

    q_bf16_list[b] : [S_q, H_q, D]
    k_full_list[b] : [S_kv, H_kv, D]  (full KV including history)
    Returns a flat tensor [total_q, H_q, D] matching the ragged Q layout.
    """
    outs = []
    for q, k, v in zip(q_bf16_list, k_full_list, v_full_list):
        S_q, H_q, D = q.shape
        S_kv, H_kv, _ = k.shape
        if H_kv != H_q:
            rep = H_q // H_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # SDPA with the appropriate causal-with-history mask.
        kv_offset = S_kv - S_q
        # Build mask [S_q, S_kv]: q row i attends to kv cols [0, kv_offset+i] when causal.
        if causal:
            mask = torch.zeros(S_q, S_kv, dtype=torch.bool, device=q.device)
            for i in range(S_q):
                mask[i, :kv_offset + i + 1] = True
            attn_mask = torch.where(mask, 0.0, float("-inf")).to(q.dtype)
        else:
            attn_mask = None
        q4 = q.transpose(0, 1).unsqueeze(0)                                # [1, H_q, S_q, D]
        k4 = k.transpose(0, 1).unsqueeze(0)
        v4 = v.transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(
            q4, k4, v4, attn_mask=attn_mask, scale=softmax_scale,
            is_causal=False)
        outs.append(o.squeeze(0).transpose(0, 1).contiguous())             # [S_q, H_q, D]
    return torch.cat(outs, dim=0).contiguous()


def test_shape_validation_rejects_small_page_size():
    """page_size=1 (sglang default) must raise NotImplementedError —
    only multiples of 32 (MXFP8 sf_vec_size) are supported."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    B, S, H_q, H_kv, D, page_size = 1, 16, 4, 1, 64, 1
    total_q = S
    num_pages = max(S // page_size, 1)
    q_fp8 = torch.zeros(total_q, H_q, D, dtype=torch.float8_e4m3fn, device="cuda")
    q_scales = torch.zeros((total_q + 15) // 16, H_q, D // 32,
                           dtype=torch.uint8, device="cuda")
    k_pool = torch.zeros(num_pages, page_size, H_kv, D,
                         dtype=torch.float8_e4m3fn, device="cuda")
    k_chan_scale = torch.zeros(H_kv, D // 32,
                               dtype=torch.uint8, device="cuda")
    v_pool = torch.zeros(num_pages, D, H_kv, page_size,
                         dtype=torch.float8_e4m3fn, device="cuda")
    v_chan_scale = torch.ones(H_kv, D, dtype=torch.float32, device="cuda")
    qo_indptr = torch.tensor([0, total_q], dtype=torch.int32, device="cuda")
    paged_kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    paged_kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device="cuda")
    last_page_len = torch.tensor([page_size], dtype=torch.int32, device="cuda")

    with pytest.raises(NotImplementedError, match="page_size"):
        pp_bk.mxfp8_paged_prefill_fwd(
            q_fp8, q_scales, k_pool, k_chan_scale, v_pool, v_chan_scale,
            qo_indptr, paged_kv_indices, paged_kv_indptr, last_page_len,
            softmax_scale=0.125, causal=True)


@pytest.mark.parametrize("D", [32, 64, 128, 256])
@pytest.mark.parametrize("page_size", [32, 64])
def test_dispatch_or_runs(D, page_size):
    """On sm < 12 the op must raise RuntimeError mentioning the op name.
    On sm == 12 the kernel runs and produces correctly-shaped output."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    torch.manual_seed(D * 11 + page_size)
    B, S_q, H_q, H_kv = 2, 64, 8, 2
    S_kv_per_b = [S_q + 64, S_q + 128]

    q_list = [torch.randn(S_q, H_q, D, dtype=torch.bfloat16, device="cuda") * 0.5
              for _ in range(B)]
    k_list = [torch.randn(S_kv_per_b[b], H_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
              for b in range(B)]
    v_list = [torch.randn(S_kv_per_b[b], H_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
              for b in range(B)]

    q_full = torch.cat(q_list, dim=0).contiguous()
    qo_indptr = _build_qo_indptr([S_q] * B, q_full.device)
    q_fp8, q_sc = pp_bk.quantize_q_ragged(q_full, H_q)

    (k_pool, k_chan_scale, v_pool, v_chan_scale,
     paged_kv_indices, paged_kv_indptr, last_page_len) = _build_paged_kv(
        k_list, v_list, page_size, device=q_full.device)

    if not _is_sm120():
        with pytest.raises(RuntimeError, match="mxfp8_attn_fwd_paged"):
            pp_bk.mxfp8_paged_prefill_fwd(
                q_fp8, q_sc, k_pool, k_chan_scale, v_pool, v_chan_scale,
                qo_indptr, paged_kv_indices, paged_kv_indptr, last_page_len,
                softmax_scale=1.0 / math.sqrt(D), causal=True)
        return

    out = pp_bk.mxfp8_paged_prefill_fwd(
        q_fp8, q_sc, k_pool, k_chan_scale, v_pool, v_chan_scale,
        qo_indptr, paged_kv_indices, paged_kv_indptr, last_page_len,
        softmax_scale=1.0 / math.sqrt(D), causal=True)
    assert out.shape == q_full.shape and out.dtype == torch.bfloat16


@pytest.mark.skipif(not _is_sm120(),
                    reason="MXFP8 paged-prefill kernel requires Blackwell sm_120a")
@pytest.mark.parametrize("D", [32, 64, 128, 256])
@pytest.mark.parametrize("page_size", [32, 64])
@pytest.mark.parametrize("causal", [True, False])
def test_correctness_vs_sdpa(D, page_size, causal):
    """Cosine alignment vs torch SDPA on causal+history extend shape."""
    torch.manual_seed(D * 17 + page_size * 7 + (1 if causal else 0))
    H_q, H_kv = 8, 2
    q_lens = [64, 128, 64]
    kv_lens = [192, 256, 64]    # last batch is decode-equivalent (S_q == S_kv → no history)
    assert all(kv >= q for q, kv in zip(q_lens, kv_lens))

    q_list = [torch.randn(q_lens[b], H_q, D, dtype=torch.bfloat16, device="cuda") * 0.5
              for b in range(len(q_lens))]
    k_list = [torch.randn(kv_lens[b], H_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
              for b in range(len(kv_lens))]
    v_list = [torch.randn(kv_lens[b], H_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
              for b in range(len(kv_lens))]

    q_full = torch.cat(q_list, dim=0).contiguous()
    qo_indptr = _build_qo_indptr(q_lens, q_full.device)
    q_fp8, q_sc = pp_bk.quantize_q_ragged(q_full, H_q)

    (k_pool, k_chan_scale, v_pool, v_chan_scale,
     paged_kv_indices, paged_kv_indptr, last_page_len) = _build_paged_kv(
        k_list, v_list, page_size, device=q_full.device)

    scale = 1.0 / math.sqrt(D)
    out = pp_bk.mxfp8_paged_prefill_fwd(
        q_fp8, q_sc, k_pool, k_chan_scale, v_pool, v_chan_scale,
        qo_indptr, paged_kv_indices, paged_kv_indptr, last_page_len,
        softmax_scale=scale, causal=causal)
    ref = _ref_sdpa_extend(q_list, k_list, v_list, scale, causal)

    cos = F.cosine_similarity(out.float().flatten(),
                              ref.float().flatten(), dim=0).item()
    assert cos > 0.985, \
        f"D={D} ps={page_size} causal={causal}: cos={cos:.6f}"


@pytest.mark.skipif(not _is_sm120(),
                    reason="MXFP8 paged-prefill kernel requires Blackwell sm_120a")
def test_ragged_tail_predication():
    """S_q values that are not multiples of kBr=64 must still be correct."""
    D, H_q, H_kv, page_size = 128, 8, 2, 32
    q_lens = [1, 7, 17, 64, 256]    # mixed ragged including kBr-aligned (64) and unaligned (17)
    kv_lens = [q + 128 for q in q_lens]

    torch.manual_seed(0xBEEF)
    q_list = [torch.randn(q_lens[b], H_q, D, dtype=torch.bfloat16, device="cuda") * 0.5
              for b in range(len(q_lens))]
    k_list = [torch.randn(kv_lens[b], H_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
              for b in range(len(kv_lens))]
    v_list = [torch.randn(kv_lens[b], H_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
              for b in range(len(kv_lens))]

    q_full = torch.cat(q_list, dim=0).contiguous()
    qo_indptr = _build_qo_indptr(q_lens, q_full.device)
    q_fp8, q_sc = pp_bk.quantize_q_ragged(q_full, H_q)

    (k_pool, k_chan_scale, v_pool, v_chan_scale,
     paged_kv_indices, paged_kv_indptr, last_page_len) = _build_paged_kv(
        k_list, v_list, page_size, device=q_full.device)

    scale = 1.0 / math.sqrt(D)
    out = pp_bk.mxfp8_paged_prefill_fwd(
        q_fp8, q_sc, k_pool, k_chan_scale, v_pool, v_chan_scale,
        qo_indptr, paged_kv_indices, paged_kv_indptr, last_page_len,
        softmax_scale=scale, causal=True)
    ref = _ref_sdpa_extend(q_list, k_list, v_list, scale, causal=True)
    cos = F.cosine_similarity(out.float().flatten(),
                              ref.float().flatten(), dim=0).item()
    assert cos > 0.985, f"ragged tail cos={cos:.6f}"


@pytest.mark.skipif(not _is_sm120(),
                    reason="MXFP8 paged-prefill kernel requires Blackwell sm_120a")
def test_plan_run_matches_single_call():
    """PrefillPagedPlan.run() must produce the same output as the single-call API."""
    D, H_q, H_kv, page_size = 64, 16, 4, 32
    q_lens = [32, 64]
    kv_lens = [128, 256]

    torch.manual_seed(42)
    q_list = [torch.randn(q_lens[b], H_q, D, dtype=torch.bfloat16, device="cuda") * 0.5
              for b in range(len(q_lens))]
    k_list = [torch.randn(kv_lens[b], H_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
              for b in range(len(kv_lens))]
    v_list = [torch.randn(kv_lens[b], H_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
              for b in range(len(kv_lens))]

    q_full = torch.cat(q_list, dim=0).contiguous()
    qo_indptr = _build_qo_indptr(q_lens, q_full.device)
    q_fp8, q_sc = pp_bk.quantize_q_ragged(q_full, H_q)
    (k_pool, k_chan_scale, v_pool, v_chan_scale,
     paged_kv_indices, paged_kv_indptr, last_page_len) = _build_paged_kv(
        k_list, v_list, page_size, device=q_full.device)

    scale = 1.0 / math.sqrt(D)
    out_call = pp_bk.mxfp8_paged_prefill_fwd(
        q_fp8, q_sc, k_pool, k_chan_scale, v_pool, v_chan_scale,
        qo_indptr, paged_kv_indices, paged_kv_indptr, last_page_len,
        softmax_scale=scale, causal=True)
    plan = pp_bk.plan_paged_prefill(
        qo_indptr_cpu=qo_indptr.detach().cpu(), num_q_heads=H_q,
        device=q_full.device)
    out_plan = plan.run(
        q_fp8, q_sc, k_pool, k_chan_scale, v_pool, v_chan_scale,
        qo_indptr, paged_kv_indices, paged_kv_indptr, last_page_len,
        softmax_scale=scale, causal=True)
    assert torch.equal(out_call, out_plan), "plan vs call diverged"
