"""SM120 MXFP8 paged-KV prefill (extend mode) — production Python entrypoint.

Mirrors flashinfer's ``BatchPrefillWithPagedKVCache`` API so the sglang
``FlashInferAttnBackend`` plumbing can be swapped to this backend with
minimal rewiring.

Calling convention (ragged Q + paged KV):

    q_fp8                  : [total_q_tokens, H_q, D]                  fp8_e4m3
    q_scales               : [total_q_tokens / 16, H_q, D/32]          uint8 (UE8M0)
    k_pool_fp8             : [num_pages, page_size, H_kv, D]           fp8_e4m3
    k_chan_scale          : [num_pages, H_kv, D/32]                   uint8 (UE8M0)
                             (max-pooled across the page's S-rows)
    v_pool_fp8             : [num_pages, D, H_kv, page_size]           fp8_e4m3
                             (pre-transposed: D outermost within a page)
    v_chan_scale           : [H_kv, D]                                 fp32
                             (per-channel V scale, applied at epilogue;
                              not per-page → V tolerates any page_size)
    qo_indptr              : [B+1]   int32   cumulative Q-token counts
    paged_kv_indices       : [P_used] int32  flat list of page ids
    paged_kv_indptr        : [B+1]   int32   per-req slice into paged_kv_indices
    paged_kv_last_page_len : [B]     int32   valid tokens in the last page

Output: bf16 ``[total_q_tokens, H_q, D]``.

Constraints (Blackwell sm_120a):
    - sm == 12.0 (sm_120a) — the host dispatcher refuses on other arches.
    - head_dim D ∈ {32, 64, 128, 256}  (D=256 uses a dedicated path).
    - page_size ∈ {32, 64, 128, 256} (must be a multiple of 32 = MXFP8 sf_vec_size).
    - q_seq_len[b] need not be a multiple of 16 IF q-row OOB rows are
      masked at the Q-fragment load (the kernel handles ragged tail).
      The Q-scales array does need exactly ceil(total_q / 16) rows however —
      the wrapper pads short tails to the next 16 boundary.
    - H_q % H_kv == 0.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


FP8_E4M3_MAX = 448.0
_SUPPORTED_D = (32, 64, 128, 256)
_MIN_PAGE_SIZE = 32
_KBR = 64                    # kernel M-tile rows; q_tile boundary granularity


def _check_inputs(q_fp8, q_scales, k_pool, k_chan_scale, v_pool, v_chan_scale,
                  qo_indptr, paged_kv_indices, paged_kv_indptr,
                  paged_kv_last_page_len):
    assert q_fp8.dtype == torch.float8_e4m3fn, "q_fp8 must be float8_e4m3fn"
    assert k_pool.dtype == torch.float8_e4m3fn, "k_pool must be float8_e4m3fn"
    assert v_pool.dtype == torch.float8_e4m3fn, "v_pool must be float8_e4m3fn"
    assert q_scales.dtype == torch.uint8
    assert k_chan_scale.dtype == torch.uint8
    assert v_chan_scale.dtype == torch.float32, \
        "v_chan_scale must be fp32 [H_kv, D]"
    for t in (qo_indptr, paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len):
        assert t.dtype == torch.int32, f"index tensor must be int32, got {t.dtype}"

    assert q_fp8.dim() == 3, "q_fp8 must be [total_q, H_q, D]"
    total_q, H_q, D = q_fp8.shape
    if D not in _SUPPORTED_D:
        raise NotImplementedError(f"head_dim must be in {_SUPPORTED_D}; got D={D}")

    assert k_pool.dim() == 4
    num_pages, page_size, H_kv, D_k = k_pool.shape
    if page_size < _MIN_PAGE_SIZE or page_size % _MIN_PAGE_SIZE != 0:
        raise NotImplementedError(
            f"page_size={page_size} must be a positive multiple of "
            f"{_MIN_PAGE_SIZE} (MXFP8 sf_vec_size). For sglang, launch "
            f"with `--page-size {_MIN_PAGE_SIZE}` or larger.")
    assert D_k == D
    assert v_pool.shape == (num_pages, D, H_kv, page_size), (
        f"v_pool must be [num_pages, D, H_kv, page_size], got {tuple(v_pool.shape)}")
    assert H_q % H_kv == 0

    # Q-scales: one byte per (16-Q-row tile, H_q, D-block-of-32).
    expected_q_scales = ((total_q + 15) // 16, H_q, D // 32)
    assert q_scales.shape == expected_q_scales, (
        f"q_scales shape {tuple(q_scales.shape)} != expected {expected_q_scales}")

    # K channel scale: UE8M0 byte per (H_kv, D-block) — GLOBAL across pages.
    assert k_chan_scale.shape == (H_kv, D // 32), (
        f"k_chan_scale shape mismatch: {tuple(k_chan_scale.shape)}")
    # V channel scale: fp32 [H_kv, D]
    assert v_chan_scale.shape == (H_kv, D), \
        f"v_chan_scale must be [{H_kv}, {D}] fp32; got {tuple(v_chan_scale.shape)}"

    B = paged_kv_last_page_len.numel()
    assert qo_indptr.shape == (B + 1,), \
        f"qo_indptr shape {tuple(qo_indptr.shape)} != ({B + 1},)"
    assert paged_kv_indptr.shape == (B + 1,), \
        f"paged_kv_indptr shape {tuple(paged_kv_indptr.shape)} != ({B + 1},)"


def _build_work_units(qo_indptr_cpu: torch.Tensor, num_q_heads: int,
                       device: torch.device) -> tuple[torch.Tensor, int]:
    """Pack the (b, h, q_tile_in_b) work units for the kernel.

    qo_indptr_cpu : [B+1] int32 (CPU tensor for the iteration)
    Returns       : (work_units: [N*3] int32 on device, N: int)
    """
    B = qo_indptr_cpu.numel() - 1
    q_lens = (qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]).tolist()
    rows = []
    for b in range(B):
        n_tiles_b = (q_lens[b] + _KBR - 1) // _KBR
        for h in range(num_q_heads):
            for qt in range(n_tiles_b):
                rows.append((b, h, qt))
    N = len(rows)
    if N == 0:
        return torch.empty(0, dtype=torch.int32, device=device), 0
    flat = torch.tensor(rows, dtype=torch.int32, device=device).reshape(-1)
    return flat.contiguous(), N


def _round_up(x: int, mult: int) -> int:
    return ((x + mult - 1) // mult) * mult


def mxfp8_paged_prefill_fwd(
    q_fp8:                   torch.Tensor,
    q_scales:                torch.Tensor,
    k_pool_fp8:              torch.Tensor,
    k_chan_scale:           torch.Tensor,
    v_pool_fp8:              torch.Tensor,
    v_chan_scale:           torch.Tensor,
    qo_indptr:               torch.Tensor,
    paged_kv_indices:        torch.Tensor,
    paged_kv_indptr:         torch.Tensor,
    paged_kv_last_page_len:  torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SM120 MXFP8 paged-KV prefill / extend forward (single-call)."""
    _check_inputs(q_fp8, q_scales, k_pool_fp8, k_chan_scale,
                  v_pool_fp8, v_chan_scale, qo_indptr,
                  paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len)

    total_q, H_q, D = q_fp8.shape
    H_kv = k_pool_fp8.size(2)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    if out is None:
        out = torch.empty(total_q, H_q, D,
                          dtype=torch.bfloat16, device=q_fp8.device)

    # Build work units. We move qo_indptr to CPU once for the loop; the
    # Plan API caches this work_units array across replays.
    work_units, total_work = _build_work_units(
        qo_indptr.detach().cpu(), int(H_q), q_fp8.device)

    if total_work == 0:
        return out

    out_real = torch.ops.fish_scales_ops.mxfp8_attn_fwd_paged(
        q_fp8.contiguous(), q_scales.contiguous(),
        k_pool_fp8.contiguous(), k_chan_scale.contiguous(),
        v_pool_fp8.contiguous(), v_chan_scale.contiguous(),
        qo_indptr.contiguous(), paged_kv_indices.contiguous(),
        paged_kv_indptr.contiguous(), paged_kv_last_page_len.contiguous(),
        work_units,
        int(total_work), bool(causal), float(softmax_scale))
    out.copy_(out_real)
    return out


# -----------------------------------------------------------------------------
# Plan / run API (FlashInfer-style separation)
# -----------------------------------------------------------------------------

class PrefillPagedPlan:
    """Reusable plan for a fixed extend schedule.

    Pre-computes the (b, h_q, q_tile) work-unit array once so subsequent
    ``run()`` calls skip the host-side packing. Best for sglang's batched
    extend pass where the schedule's shape is stable across forward steps.
    """

    def __init__(self, *,
                 qo_indptr_cpu: torch.Tensor,
                 num_q_heads: int,
                 device: str | torch.device = "cuda"):
        if qo_indptr_cpu.device.type != "cpu":
            raise ValueError("qo_indptr_cpu must be a CPU int32 tensor for planning")
        if qo_indptr_cpu.dtype != torch.int32:
            raise ValueError("qo_indptr_cpu must be int32")
        self.num_q_heads = int(num_q_heads)
        self.device = torch.device(device)
        self.work_units, self.total_work = _build_work_units(
            qo_indptr_cpu, self.num_q_heads, self.device)

    def run(self,
            q_fp8:                   torch.Tensor,
            q_scales:                torch.Tensor,
            k_pool_fp8:              torch.Tensor,
            k_chan_scale:           torch.Tensor,
            v_pool_fp8:              torch.Tensor,
            v_chan_scale:           torch.Tensor,
            qo_indptr:               torch.Tensor,
            paged_kv_indices:        torch.Tensor,
            paged_kv_indptr:         torch.Tensor,
            paged_kv_last_page_len:  torch.Tensor,
            *,
            softmax_scale: Optional[float] = None,
            causal: bool = True,
            out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Hot-path extend call. Uses the pre-packed work_units; no host iteration."""
        _check_inputs(q_fp8, q_scales, k_pool_fp8, k_chan_scale,
                      v_pool_fp8, v_chan_scale, qo_indptr,
                      paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len)
        # Plan.run is the cudagraph hot path: refuse non-contiguous inputs
        # loudly rather than silently allocating via .contiguous() during
        # capture (which would bake stale src pointers into the graph).
        for _name, _t in (("q_fp8", q_fp8), ("q_scales", q_scales),
                          ("k_pool_fp8", k_pool_fp8), ("k_chan_scale", k_chan_scale),
                          ("v_pool_fp8", v_pool_fp8), ("v_chan_scale", v_chan_scale),
                          ("qo_indptr", qo_indptr), ("paged_kv_indices", paged_kv_indices),
                          ("paged_kv_indptr", paged_kv_indptr),
                          ("paged_kv_last_page_len", paged_kv_last_page_len)):
            assert _t.is_contiguous(), (
                f"PrefillPagedPlan.run: {_name} must be contiguous; "
                f"call .contiguous() before run() or use mxfp8_paged_prefill_fwd "
                f"(one-shot API auto-normalises)")
        total_q, H_q, D = q_fp8.shape
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(D)
        if out is None:
            out = torch.empty(total_q, H_q, D,
                              dtype=torch.bfloat16, device=q_fp8.device)
        if self.total_work == 0:
            return out
        out_real = torch.ops.fish_scales_ops.mxfp8_attn_fwd_paged(
            q_fp8, q_scales, k_pool_fp8, k_chan_scale, v_pool_fp8, v_chan_scale,
            qo_indptr, paged_kv_indices, paged_kv_indptr,
            paged_kv_last_page_len, self.work_units,
            int(self.total_work), bool(causal), float(softmax_scale))
        out.copy_(out_real)
        return out


def plan_paged_prefill(*,
                       qo_indptr_cpu: torch.Tensor,
                       num_q_heads: int,
                       device: str | torch.device = "cuda") -> PrefillPagedPlan:
    """One-time setup for a fixed-shape paged-KV extend workload."""
    return PrefillPagedPlan(qo_indptr_cpu=qo_indptr_cpu,
                            num_q_heads=num_q_heads, device=device)


# -----------------------------------------------------------------------------
# Calibration / test helpers — NOT part of the runtime path.
# -----------------------------------------------------------------------------

def _ue8m0_pack(log2_scale: torch.Tensor) -> torch.Tensor:
    e = log2_scale.clamp(-127, 127).round().to(torch.int32)
    return (e + 127).to(torch.uint8)


def quantize_q_ragged(q_bf16: torch.Tensor, n_q_heads: int, block: int = 32):
    """Quantize a ragged Q tensor to FP8 with prefill-style per-(16-row, H_q, D-block) scales.

    Input  : q_bf16  [total_q, H_q, D]    bf16
    Returns: (q_fp8, q_scales) where
             q_fp8    : [total_q, H_q, D]                       float8_e4m3fn
             q_scales : [ceil(total_q/16), H_q, D/32]           uint8 (UE8M0)
    """
    total_q, H_q, D = q_bf16.shape
    assert H_q == n_q_heads, f"H_q={H_q} != n_q_heads={n_q_heads}"
    n_kblk = D // block
    n_q_tile = (total_q + 15) // 16

    # Pad total_q up to a multiple of 16 (zero-pad rows participate in the
    # amax but never get read by the kernel because lo/hi_valid guards them).
    pad = n_q_tile * 16 - total_q
    if pad > 0:
        q_pad = torch.zeros(pad, H_q, D, dtype=q_bf16.dtype, device=q_bf16.device)
        q_full = torch.cat([q_bf16, q_pad], dim=0)
    else:
        q_full = q_bf16

    q5 = q_full.float().reshape(n_q_tile, 16, H_q, n_kblk, block)
    amax = q5.abs().amax(dim=(1, 4)).clamp_min(1e-30)             # [n_q_tile, H_q, n_kblk]
    log2_s = torch.ceil(torch.log2(amax / FP8_E4M3_MAX))
    sb = _ue8m0_pack(log2_s)
    sf = torch.pow(torch.tensor(2.0, device=q_bf16.device),
                    log2_s.to(torch.float32))
    sc = sf.unsqueeze(1).unsqueeze(-1)                            # [n_q_tile, 1, H_q, n_kblk, 1]
    qf = (q5 / sc).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX
            ).to(torch.float8_e4m3fn).reshape(n_q_tile * 16, H_q, D)
    if pad > 0:
        qf = qf[:total_q].contiguous()
    return qf, sb
