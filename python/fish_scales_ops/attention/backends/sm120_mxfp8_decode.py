"""SM120 MXFP8 paged-KV decode forward — production Python entrypoint.

Two APIs:

(A) ``mxfp8_decode_paged_fwd(q, qs, k_cache, k_chan_scale, v_cache, v_scales,
    block_table, seq_lens, ...)`` — single-call convenience. Allocates
    scratch + zeroes a sync counter each invocation.

(B) ``DecodePagedPlan`` / ``plan_decode_paged(...)`` — FlashInfer-style
    plan/run separation. The plan allocates scratch buffers once and
    tracks the expected-counter value across calls, so the hot-path
    ``run`` never re-zeroes the counter.

Both routes call into ``torch.ops.fish_scales_ops.mxfp8_decode_paged``
(built into ``fish_scales_ops._C``). The kernel is sm_120-only at
runtime; the C++ dispatcher returns ``cudaErrorNotSupported`` on
other arches and the binding raises ``RuntimeError``.

Single-token (S_q = 1) decode against a paged FP8 KV cache.

CALLER CONTRACT — Q/K/V are PRE-QUANTIZED, scales are GROUP-SHARED:

    Q_fp8       : [B, H_q, D]                           torch.float8_e4m3fn
    Q_scales    : [B, H_kv, D // 32]                    torch.uint8 (UE8M0,
                  group-shared across Q-heads within a kv-head group).
    K_cache_fp8 : [num_pages, page_size, H_kv, D]       torch.float8_e4m3fn
    K_chan_scale: [H_kv, D // 32]                       torch.uint8 (UE8M0)
                  (per-channel K scale, GLOBAL across all K-tokens; fed
                   into the QK mma's hardware block_scale operand.)
    V_cache_fp8 : [num_pages, D, H_kv, page_size]       torch.float8_e4m3fn
                  (pre-transposed, D outermost within a page).
    V_chan_scale: [H_kv, D]                             torch.float32
                  (per-channel V scale; applied as a post-mma FFMA in
                   the kernel epilogue. NOT per-page — removes V's
                   contribution to the page_size lower bound.)
    block_table : [B, max_blocks]                       torch.int32
    seq_lens    : [B]                                   torch.int32

Constraints (Blackwell sm_120a):
    - sm == 12.0 (sm_120a)
    - head_dim D ∈ {32, 64, 128, 256}  (D=256 runs at kStages=1 — smem-tight).
    - page_size ∈ {32, 64, 128, 256} — must be a positive multiple of 32.
      This comes from K's block-scaled QK mma (sf_vec_size=32 along D),
      NOT V. Channel-V is page-size-agnostic.
    - H_q % H_kv == 0; gqa_group = H_q / H_kv ≤ 64.

Output:
    O : [B, H_q, D]                                     bfloat16
"""
from __future__ import annotations

import math
from typing import Optional

import torch


FP8_E4M3_MAX = 448.0

# Supported head dims. Kernel internal Bc is fixed at 32 (= MXFP8 sf_vec_size)
# across all D; page_size is configurable per call as long as it is a positive
# multiple of 32. Recommended values are {32, 64, 128, 256}; 32 matches
# sglang's `--page-size 32` and is the MXFP8-natural minimum.
_SUPPORTED_D = (32, 64, 128, 256)
_KERNEL_BC = 32                       # MXFP8 sf_vec_size, matches kernel kBc
_MIN_PAGE_SIZE = 32                   # caller page_size must be a multiple of kBc


def _auto_kv_split(B: int, H_kv: int) -> int:
    """Pick kv_split_k targeting ~2× SM saturation on a 170-SM Blackwell GPU.

    Each (b, h_kv) decode work-unit launches one CTA; kv_split_k > 1 spreads
    a single seq across multiple SMs (FlashDecoding-style).
    """
    work_units = max(B * H_kv, 1)
    sm_count = 170
    target = max(2 * sm_count // work_units, 1)
    # Round to nearest power-of-two for clean K-tile partitioning.
    s = 1
    while s * 2 <= target:
        s *= 2
    return s


def _check_inputs(q_fp8, q_scales, k_cache, k_chan_scale, v_cache, v_chan_scale,
                  block_table, seq_lens):
    assert q_fp8.dtype == torch.float8_e4m3fn, "q_fp8 must be float8_e4m3fn"
    assert k_cache.dtype == torch.float8_e4m3fn, "k_cache must be float8_e4m3fn"
    assert v_cache.dtype == torch.float8_e4m3fn, "v_cache must be float8_e4m3fn"
    assert q_scales.dtype == torch.uint8 and k_chan_scale.dtype == torch.uint8
    assert v_chan_scale.dtype == torch.float32, \
        "v_chan_scale must be fp32 [H_kv, D]"
    assert block_table.dtype == torch.int32, "block_table must be int32"
    assert seq_lens.dtype == torch.int32, "seq_lens must be int32"

    B, H_q, D = q_fp8.shape
    if D not in _SUPPORTED_D:
        raise NotImplementedError(
            f"head_dim must be in {_SUPPORTED_D}; got D={D}")

    assert k_cache.dim() == 4, "K cache must be 4D [num_pages, page_size, H_kv, D]"
    num_pages, page_size, H_kv, D_k = k_cache.shape
    if page_size < _MIN_PAGE_SIZE or page_size % _MIN_PAGE_SIZE != 0:
        raise NotImplementedError(
            f"page_size={page_size} must be a positive multiple of "
            f"{_MIN_PAGE_SIZE} (MXFP8 sf_vec_size). For sglang, launch "
            f"with `--page-size {_MIN_PAGE_SIZE}` or larger.")
    assert D_k == D, "K cache head_dim must match Q"
    assert v_cache.shape == (num_pages, D, H_kv, page_size), (
        f"V cache must be [num_pages, D, H_kv, page_size], got {tuple(v_cache.shape)}")
    assert H_q % H_kv == 0, f"H_q={H_q} must be a multiple of H_kv={H_kv}"
    gqa_group = H_q // H_kv
    if gqa_group > 64:
        raise NotImplementedError(
            f"gqa_group={gqa_group} > 64 exceeds kernel M-row capacity (kBr=64)")

    assert q_scales.shape == (B, H_kv, D // 32)
    assert k_chan_scale.shape == (H_kv, D // 32)
    assert v_chan_scale.shape == (H_kv, D), \
        f"v_chan_scale must be [{H_kv}, {D}]; got {tuple(v_chan_scale.shape)}"
    assert block_table.dim() == 2 and block_table.size(0) == B
    assert seq_lens.shape == (B,)


def mxfp8_decode_paged_fwd(
    q_fp8:        torch.Tensor,
    q_scales:     torch.Tensor,
    k_cache_fp8:  torch.Tensor,
    k_chan_scale:     torch.Tensor,
    v_cache_fp8:  torch.Tensor,
    v_chan_scale: torch.Tensor,
    block_table:  torch.Tensor,
    seq_lens:     torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    out:          Optional[torch.Tensor] = None,
    kv_split_k:   Optional[int] = None,
) -> torch.Tensor:
    """Single-call paged-KV MXFP8 decode forward (sm_120 only).

    Allocates partial scratch + sync counter every call. Use
    :class:`DecodePagedPlan` for steady-state decode loops to amortize
    those allocations away.
    """
    _check_inputs(q_fp8, q_scales, k_cache_fp8, k_chan_scale, v_cache_fp8, v_chan_scale,
                  block_table, seq_lens)
    B, H_q, D = q_fp8.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    if out is None:
        out = torch.empty(B, H_q, D, dtype=torch.bfloat16, device=q_fp8.device)
    if kv_split_k is None:
        kv_split_k = _auto_kv_split(B, k_cache_fp8.size(2))
    assert kv_split_k >= 1

    dev = q_fp8.device
    if kv_split_k > 1:
        m_partial = torch.empty(B, H_q, kv_split_k, dtype=torch.float32, device=dev)
        l_partial = torch.empty(B, H_q, kv_split_k, dtype=torch.float32, device=dev)
        o_partial = torch.empty(B, H_q, kv_split_k, D, dtype=torch.float32, device=dev)
        sync_counter = torch.zeros(B, H_q, dtype=torch.int32, device=dev)
        target_counter = kv_split_k
    else:
        m_partial = torch.empty(0, dtype=torch.float32, device=dev)
        l_partial = torch.empty(0, dtype=torch.float32, device=dev)
        o_partial = torch.empty(0, dtype=torch.float32, device=dev)
        sync_counter = torch.empty(0, dtype=torch.int32, device=dev)
        target_counter = 1

    out_real = torch.ops.fish_scales_ops.mxfp8_decode_paged(
        q_fp8.contiguous(), q_scales.contiguous(),
        k_cache_fp8.contiguous(), k_chan_scale.contiguous(),
        v_cache_fp8.contiguous(), v_chan_scale.contiguous(),
        block_table.contiguous(), seq_lens.contiguous(),
        m_partial, l_partial, o_partial, sync_counter,
        int(kv_split_k), int(target_counter), float(softmax_scale))
    out.copy_(out_real)
    return out


# -----------------------------------------------------------------------------
# Plan / run API (FlashInfer-style separation)
# -----------------------------------------------------------------------------

class DecodePagedPlan:
    """Reusable plan for a fixed-shape decode workload.

    Allocates the partial-output scratch + sync counter once and tracks
    the running atomic-counter target so subsequent ``run()`` calls don't
    pay per-call ``cudaMemsetAsync`` (and no scratch allocation either).
    The counter accumulates eager-mode, and is re-zeroed inside CUDA-graph
    captures where eager state baking would break replays.
    """

    def __init__(self, *, B: int, H_q: int, H_kv: int, D: int, max_blocks: int,
                 kv_split_k: Optional[int] = None,
                 device: str | torch.device = "cuda"):
        if D not in _SUPPORTED_D:
            raise NotImplementedError(
                f"head_dim must be in {_SUPPORTED_D}; got D={D}")
        assert H_q % H_kv == 0
        gqa_group = H_q // H_kv
        if gqa_group > 64:
            raise NotImplementedError(
                f"gqa_group={gqa_group} > 64 exceeds kernel M-row capacity (kBr=64)")
        if kv_split_k is None:
            kv_split_k = _auto_kv_split(B, H_kv)

        self.B = B
        self.H_q = H_q
        self.H_kv = H_kv
        self.D = D
        self.max_blocks = max_blocks
        self.kv_split_k = kv_split_k

        if kv_split_k > 1:
            self._m_partial = torch.empty(B, H_q, kv_split_k, dtype=torch.float32, device=device)
            self._l_partial = torch.empty(B, H_q, kv_split_k, dtype=torch.float32, device=device)
            self._o_partial = torch.empty(B, H_q, kv_split_k, D, dtype=torch.float32, device=device)
            # Persistent counter — never re-zeroed in eager mode. Kernel reads
            # target_counter to identify the "last" CTA per (b, h_q).
            self._sync_counter = torch.zeros(B, H_q, dtype=torch.int32, device=device)
            self._expected_counter = 0
        else:
            self._m_partial = torch.empty(0, dtype=torch.float32, device=device)
            self._l_partial = torch.empty(0, dtype=torch.float32, device=device)
            self._o_partial = torch.empty(0, dtype=torch.float32, device=device)
            self._sync_counter = torch.empty(0, dtype=torch.int32, device=device)
            self._expected_counter = 0

    def run(self,
            q_fp8:        torch.Tensor,
            q_scales:     torch.Tensor,
            k_cache_fp8:  torch.Tensor,
            k_chan_scale:     torch.Tensor,
            v_cache_fp8:  torch.Tensor,
            v_chan_scale: torch.Tensor,
            block_table:  torch.Tensor,
            seq_lens:     torch.Tensor,
            *,
            softmax_scale: Optional[float] = None,
            out:          Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        """Hot-path decode call. No allocation or memset in eager mode."""
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(self.D)
        if out is None:
            out = torch.empty(self.B, self.H_q, self.D,
                              dtype=torch.bfloat16, device=q_fp8.device)

        # Plan.run is the cudagraph hot path: refuse non-contiguous inputs
        # loudly rather than silently allocating via .contiguous() during
        # capture (which would bake stale src pointers into the graph).
        for _name, _t in (("q_fp8", q_fp8), ("q_scales", q_scales),
                          ("k_cache_fp8", k_cache_fp8), ("k_chan_scale", k_chan_scale),
                          ("v_cache_fp8", v_cache_fp8), ("v_chan_scale", v_chan_scale),
                          ("block_table", block_table), ("seq_lens", seq_lens)):
            assert _t.is_contiguous(), (
                f"DecodePagedPlan.run: {_name} must be contiguous; "
                f"call .contiguous() before run() or use mxfp8_decode_paged_fwd "
                f"(one-shot API auto-normalises)")

        # Counter target. Eager mode accumulates so we can skip the memset;
        # CUDA-graph capture bakes a fixed target so the counter has to be
        # re-zeroed each replay (captured as a fast graph node, ~0.1 µs).
        if self.kv_split_k > 1:
            if torch.cuda.is_current_stream_capturing():
                self._sync_counter.zero_()
                target_counter = self.kv_split_k
                self._expected_counter = 0
            else:
                self._expected_counter += self.kv_split_k
                target_counter = self._expected_counter
        else:
            target_counter = 1

        out_real = torch.ops.fish_scales_ops.mxfp8_decode_paged(
            q_fp8, q_scales, k_cache_fp8, k_chan_scale, v_cache_fp8, v_chan_scale,
            block_table, seq_lens,
            self._m_partial, self._l_partial, self._o_partial, self._sync_counter,
            int(self.kv_split_k), int(target_counter), float(softmax_scale))
        out.copy_(out_real)
        return out


def plan_decode_paged(*, B: int, H_q: int, H_kv: int, D: int, max_blocks: int,
                      kv_split_k: Optional[int] = None,
                      device: str | torch.device = "cuda") -> DecodePagedPlan:
    """One-time setup for a fixed-shape paged-KV decode workload."""
    return DecodePagedPlan(B=B, H_q=H_q, H_kv=H_kv, D=D,
                           max_blocks=max_blocks, kv_split_k=kv_split_k,
                           device=device)


# -----------------------------------------------------------------------------
# Calibration / test helpers — NOT part of the runtime path.
# Engine-side quantize code should mirror these layouts.
# -----------------------------------------------------------------------------

def _ue8m0_pack(log2_scale: torch.Tensor) -> torch.Tensor:
    e = log2_scale.clamp(-127, 127).round().to(torch.int32)
    return (e + 127).to(torch.uint8)


def quantize_q_grouped(q_bf16: torch.Tensor, n_kv_heads: int, block: int = 32):
    """Quantize Q with GROUP-SHARED scales across each kv-head's Q-group.

    Input:  q_bf16  [B, H_q, D]                bf16
    Output: q_fp8   [B, H_q, D]                float8_e4m3fn
            q_sc    [B, H_kv, D/32]            uint8  (one byte per (b, h_kv, kb))
    """
    B, H_q, D = q_bf16.shape
    assert H_q % n_kv_heads == 0
    gqa = H_q // n_kv_heads
    n_kblk = D // block
    q3 = q_bf16.float().reshape(B, n_kv_heads, gqa, n_kblk, block)
    amax = q3.abs().amax(dim=(2, 4)).clamp_min(1e-30)              # [B, Hkv, n_kblk]
    log2_s = torch.ceil(torch.log2(amax / FP8_E4M3_MAX))
    sb = _ue8m0_pack(log2_s)
    sf = torch.pow(torch.tensor(2.0, device=q_bf16.device),
                   log2_s.to(torch.float32))
    sc = sf.unsqueeze(2).unsqueeze(-1)
    qf = (q3 / sc).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX
            ).to(torch.float8_e4m3fn).reshape(B, H_q, D)
    return qf, sb


def compute_v_chan_scale(v_bf16: torch.Tensor) -> torch.Tensor:
    """Compute the per-(H_kv, D) channel scale for V.

    Input  v_bf16 : [..., H_kv, D] bf16/fp32 — any leading shape is folded
                    into the max reduction.
    Output        : [H_kv, D] fp32 (positive amax / FP8_E4M3_MAX).

    Production calibration would pass a representative batch through this
    helper once at model load; the resulting scale is then frozen for the
    entire serving session (so the paged KV cache can grow without
    re-quantising older tokens).
    """
    H_kv = v_bf16.size(-2)
    D    = v_bf16.size(-1)
    V_f32 = v_bf16.float().reshape(-1, H_kv, D)
    amax  = V_f32.abs().amax(dim=0).clamp_min(1e-30)              # [H_kv, D]
    return (amax / FP8_E4M3_MAX).to(torch.float32).contiguous()


def compute_k_chan_scale(k_bf16: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Compute the per-(H_kv, D-block-of-32) channel scale for K (UE8M0).

    K's mma is still hardware block-scaled, so the scale stays UE8M0
    (1 byte per (h_kv, D/32)) — but the per-page dimension is collapsed
    into a single global scale per (h_kv, D-block). max is computed across
    all K-tokens in the input.

    Input  k_bf16 : [..., H_kv, D]
    Output        : [H_kv, D/32] uint8 (UE8M0 = e+127 where e = ceil(log2(amax / 448))).
    """
    H_kv = k_bf16.size(-2)
    D    = k_bf16.size(-1)
    n_kblk = D // block
    K_f32 = k_bf16.float().reshape(-1, H_kv, n_kblk, block)
    amax  = K_f32.abs().amax(dim=(0, 3)).clamp_min(1e-30)         # [H_kv, n_kblk]
    log2_s = torch.ceil(torch.log2(amax / FP8_E4M3_MAX))
    return _ue8m0_pack(log2_s).contiguous()


def quantize_kv_to_paged(k_bf16: torch.Tensor,
                         v_bf16: torch.Tensor,
                         page_size: int,
                         block: int = 32,
                         k_chan_scale: Optional[torch.Tensor] = None,
                         v_chan_scale: Optional[torch.Tensor] = None):
    """Quantize K and V into the paged layout the decode kernel expects.

    Both K and V now use *channel-scale* tensors of shape
      K_chan_scale : [H_kv, D/32] uint8 (UE8M0), applied via hardware
                     block_scale in the QK mma.
      V_chan_scale : [H_kv, D] fp32, applied as a post-mma FFMA in
                     the kernel epilogue.

    Neither scale depends on page granularity. The paged-KV cache shape
    is unchanged (K_cache_fp8, V_cache_fp8).

    If ``k_chan_scale`` / ``v_chan_scale`` are provided, they are used
    as-is (e.g. shared across multiple per-batch quantise calls in a
    ragged-extend workload). If None, the function computes one from
    this call's ``k_bf16`` / ``v_bf16`` alone.

    Returns (K_cache_fp8, K_chan_scale, V_cache_fp8, V_chan_scale).
    """
    B, S, H_kv, D = k_bf16.shape
    assert v_bf16.shape == (B, S, H_kv, D)
    assert S % page_size == 0, f"S={S} must be a multiple of page_size={page_size}"
    n_pages = S // page_size

    if k_chan_scale is None:
        k_chan_scale = compute_k_chan_scale(k_bf16, block=block)
    if v_chan_scale is None:
        v_chan_scale = compute_v_chan_scale(v_bf16)

    # Recover the fp32 K scale per (h_kv, D-block) from the UE8M0 byte and
    # divide K element-wise. K_chan_scale[h, kb] = e+127 → 2^(e) per
    # 32-element D-block, broadcast across S.
    log2_k = k_chan_scale.to(torch.int32) - 127                       # [H_kv, D/32]
    sfk = torch.pow(torch.tensor(2.0, device=k_bf16.device),
                    log2_k.to(torch.float32))                          # fp32
    scale_k = sfk.unsqueeze(0).unsqueeze(0).unsqueeze(-1)             # [1, 1, H_kv, D/32, 1]
    Kq = (k_bf16.float().reshape(B, S, H_kv, D // block, block) / scale_k
            ).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    K_cache = Kq.reshape(B, n_pages, page_size, H_kv, D)

    V_f32 = v_bf16.float()
    Vq = (V_f32 / v_chan_scale.unsqueeze(0).unsqueeze(0)).clamp(
        -FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    V_cache = (Vq.reshape(B, n_pages, page_size, H_kv, D)
                 .permute(0, 1, 4, 3, 2)
                 .contiguous())
    return K_cache, k_chan_scale, V_cache, v_chan_scale
