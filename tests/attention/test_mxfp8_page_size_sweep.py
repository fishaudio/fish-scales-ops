"""Paged-decode page_size sweep — verifies Step 1 (page_size runtime-ization).

Before this change, the SM120 MXFP8 paged-decode kernel hardcoded
``page_size`` per head_dim (D=32 → 128, D=64/D=128 → 64). For sglang
integration the page_size is now a per-call runtime parameter; the
caller may pick any positive multiple of 32 (= MXFP8 sf_vec_size).

The test re-quantises the same BF16 KV tensors into multiple page_size
layouts and confirms the decode output is cosine-aligned across them
(it is not bit-exact because mma accumulation order differs slightly).

Falls back to a sm < 12 dispatch-only check (``cudaErrorNotSupported``)
when run on Hopper / Ada / older arches.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import fish_scales_ops as fso
from fish_scales_ops.attention.backends import sm120_mxfp8_decode as dec_bk


def _is_sm120() -> bool:
    if not torch.cuda.is_available():
        return False
    maj, _ = torch.cuda.get_device_capability()
    return maj >= 12


def _build_paged_kv(k_bf16, v_bf16, page_size):
    """Quantise + flatten a [B, S, H_kv, D] KV pair into the global pool layout.

    Returns (K_pool, K_pool_scales, V_pool, V_chan_scale, block_table) where
    V_chan_scale is the new per-(H_kv, D) fp32 channel scale (replaces
    the legacy per-page MXFP8 V scale tensor).
    """
    B, S, H_kv, D = k_bf16.shape
    assert S % page_size == 0, f"S={S} must divide page_size={page_size}"
    num_pages_per_seq = S // page_size

    K_cache, k_chan_scale, V_cache, v_chan_scale = dec_bk.quantize_kv_to_paged(
        k_bf16, v_bf16, page_size)
    # K_cache       : [B, num_pages, page_size, H_kv, D]
    # k_chan_scale  : [H_kv, D/32]  uint8 (UE8M0) — GLOBAL
    # V_cache       : [B, num_pages, D, H_kv, page_size]
    # v_chan_scale  : [H_kv, D]  fp32 — GLOBAL
    K_pool = K_cache.reshape(B * num_pages_per_seq, page_size, H_kv, D).contiguous()
    V_pool = V_cache.reshape(B * num_pages_per_seq, D, H_kv, page_size).contiguous()

    block_table = torch.arange(B * num_pages_per_seq,
                                dtype=torch.int32,
                                device=k_bf16.device
                                ).reshape(B, num_pages_per_seq)
    return K_pool, k_chan_scale, V_pool, v_chan_scale, block_table


def _ref_sdpa_decode(q_bf16, k_bf16, v_bf16, softmax_scale):
    """Reference SDPA: q [B, H_q, D] vs k,v [B, S, H_kv, D]. Returns [B, H_q, D]."""
    B, H_q, D = q_bf16.shape
    _, S, H_kv, _ = k_bf16.shape
    if H_kv != H_q:
        rep = H_q // H_kv
        k_bf16 = k_bf16.repeat_interleave(rep, dim=2)
        v_bf16 = v_bf16.repeat_interleave(rep, dim=2)
    q4 = q_bf16.unsqueeze(1)                                         # [B, 1, H_q, D]
    out = F.scaled_dot_product_attention(
        q4.transpose(1, 2), k_bf16.transpose(1, 2), v_bf16.transpose(1, 2),
        is_causal=False, scale=softmax_scale)
    return out.transpose(1, 2).squeeze(1).contiguous()               # [B, H_q, D]


def _quant_q(q_bf16, n_kv_heads, block=32):
    return dec_bk.quantize_q_grouped(q_bf16, n_kv_heads, block=block)


@pytest.mark.parametrize("D", [32, 64, 128, 256])
@pytest.mark.parametrize("page_size", [32, 64, 128])
def test_decode_page_size_dispatch_or_runs(D, page_size):
    """When sm < 12, expect cudaErrorNotSupported via dispatch raise.
    When sm >= 12, run the kernel and compare to SDPA reference."""
    torch.manual_seed(D * 13 + page_size)

    B, H_q, H_kv, S = 2, 8, 2, 256          # gqa=4, small smoke shape
    if S % page_size:
        pytest.skip(f"S={S} not divisible by page_size={page_size}")

    q_bf16 = torch.randn(B, H_q, D, dtype=torch.bfloat16, device="cuda") * 0.5
    k_bf16 = torch.randn(B, S, H_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
    v_bf16 = torch.randn(B, S, H_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5

    K_pool, k_chan_scale, V_pool, v_chan_scale, block_table = _build_paged_kv(
        k_bf16, v_bf16, page_size)
    seq_lens = torch.full((B,), S, dtype=torch.int32, device="cuda")

    q_fp8, q_sc = _quant_q(q_bf16, H_kv)
    softmax_scale = 1.0 / math.sqrt(D)

    if not _is_sm120():
        with pytest.raises(RuntimeError, match="mxfp8_decode_paged"):
            _ = dec_bk.mxfp8_decode_paged_fwd(
                q_fp8, q_sc, K_pool, k_chan_scale, V_pool, v_chan_scale,
                block_table, seq_lens, softmax_scale=softmax_scale)
        return

    out = dec_bk.mxfp8_decode_paged_fwd(
        q_fp8, q_sc, K_pool, k_chan_scale, V_pool, v_chan_scale,
        block_table, seq_lens, softmax_scale=softmax_scale)
    ref = _ref_sdpa_decode(q_bf16, k_bf16, v_bf16, softmax_scale)
    cos = F.cosine_similarity(out.float().flatten(),
                              ref.float().flatten(), dim=0).item()
    assert cos > 0.985, \
        f"D={D} page_size={page_size}: cos={cos:.6f} below 0.985"


@pytest.mark.skipif(not _is_sm120(),
                    reason="MXFP8 decode kernel requires Blackwell sm_120a")
@pytest.mark.parametrize("D", [32, 64, 128, 256])
def test_decode_page_size_cross_equivalence(D):
    """Same KV semantics under different supported page_sizes must produce
    cosine-aligned outputs (mma accumulation order varies slightly between
    layouts)."""
    torch.manual_seed(D * 7)

    B, H_q, H_kv, S = 2, 8, 2, 512
    q_bf16 = torch.randn(B, H_q, D, dtype=torch.bfloat16, device="cuda") * 0.5
    k_bf16 = torch.randn(B, S, H_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
    v_bf16 = torch.randn(B, S, H_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
    seq_lens = torch.full((B,), S, dtype=torch.int32, device="cuda")
    q_fp8, q_sc = _quant_q(q_bf16, H_kv)
    softmax_scale = 1.0 / math.sqrt(D)

    outputs = {}
    for ps in (32, 64, 128):
        K_pool, k_chan_scale, V_pool, v_chan_scale, block_table = _build_paged_kv(
            k_bf16, v_bf16, ps)
        outputs[ps] = dec_bk.mxfp8_decode_paged_fwd(
            q_fp8, q_sc, K_pool, k_chan_scale, V_pool, v_chan_scale,
            block_table, seq_lens, softmax_scale=softmax_scale)

    for ps in (32, 64):
        cos = F.cosine_similarity(outputs[ps].float().flatten(),
                                  outputs[128].float().flatten(), dim=0).item()
        assert cos > 0.999, f"D={D}: ps={ps} vs 128 cos={cos:.6f}"


def test_invalid_page_size_rejected():
    """page_size=1 (sglang default) must be rejected with a clear message
    pointing at the MXFP8 sf_vec_size=32 lower bound."""
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    B, H_q, H_kv, S, D, page_size = 1, 4, 1, 16, 64, 1
    num_pages = max(S // page_size, 1)
    q_fp8 = torch.zeros(B, H_q, D, dtype=torch.float8_e4m3fn, device="cuda")
    q_sc = torch.zeros(B, H_kv, D // 32, dtype=torch.uint8, device="cuda")
    K_pool = torch.zeros(num_pages, page_size, H_kv, D,
                         dtype=torch.float8_e4m3fn, device="cuda")
    k_chan_scale = torch.zeros(H_kv, D // 32,
                               dtype=torch.uint8, device="cuda")
    V_pool = torch.zeros(num_pages, D, H_kv, page_size,
                         dtype=torch.float8_e4m3fn, device="cuda")
    v_chan_scale = torch.ones(H_kv, D, dtype=torch.float32, device="cuda")
    block_table = torch.zeros(B, num_pages, dtype=torch.int32, device="cuda")
    seq_lens = torch.full((B,), S, dtype=torch.int32, device="cuda")

    with pytest.raises(NotImplementedError, match="page_size"):
        dec_bk.mxfp8_decode_paged_fwd(
            q_fp8, q_sc, K_pool, k_chan_scale, V_pool, v_chan_scale,
            block_table, seq_lens, softmax_scale=0.125)
