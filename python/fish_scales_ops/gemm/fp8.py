"""1×128 / 128×128 FP8 ops — quantize, GEMM, scale repack, and the
internal-quantize ``linear_qx`` variant.

Auto-dispatches sm_90 deep_gemm vs sm_120 CUTLASS BlockScaledKernel.
"""
from __future__ import annotations

import torch


def linear_fp8(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    sx: torch.Tensor,
    sw: torch.Tensor,
) -> torch.Tensor:
    """Block-scaled FP8 (E4M3) GEMM. Auto-dispatches sm_90 / sm_120.

    On sm_120 (Blackwell sm_120a) you must produce scales
    via ``quantize_*_fp8(..., use_ue8m0=True)`` — the wrapper repacks
    them to the int32-packed UE8M0 layout the CUTLASS kernel expects.
    On sm_90 (Hopper / H200) ``use_ue8m0=False`` (the default) is fine.

    Args:
        x_fp8: float8_e4m3fn [M, K], contiguous.
        w_fp8: float8_e4m3fn [N, K], contiguous.
        sx:    float32 dequant scales for x (see quantize helper output).
        sw:    float32 dequant scales for w.

    Returns:
        bfloat16 [M, N].
    """
    return torch.ops.fish_scales_ops.linear_fp8(
        x_fp8.contiguous(),
        w_fp8.contiguous(),
        sx.contiguous(),
        sw.contiguous(),
    )


def quantize_1x128_fp8(
    x: torch.Tensor,
    use_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [..., K] → FP8 (E4M3) [..., K] + FP32 per-row 1×128 dequant scales.

    Args:
        x: bf16 tensor with K divisible by 128.
        use_ue8m0: round scales to powers of 2 (UE8M0-representable).
            Set to ``True`` on sm_120 — the FP8 GEMM wrapper repacks
            these into the int32-packed UE8M0 format CUTLASS expects.
            ``False`` on sm_90 (deep_gemm path) is fine.

    Returns:
        (x_fp8, sx) where sx is float32 [pad(M,4), K/128] (TMA-aligned;
        the trailing pad rows hold zeros).
    """
    return torch.ops.fish_scales_ops.quantize_1x128(x.contiguous(), use_ue8m0)


def quantize_1x128_fp8_packed(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [..., K] → FP8 (E4M3) [..., K] + int32-packed UE8M0 1×128 scales.

    Fused single-kernel replacement for
    ``repack_fp8_act_scales(quantize_1x128_fp8(x, use_ue8m0=True)[1])`` on
    sm_120 — eliminates the FP32 scale round-trip through global memory and
    the separate repack-kernel launch.

    Returns:
        (x_fp8, sx_packed) — packed int32 K-major ``[pad(M,4), K/512]``
        (4 UE8M0 bytes per int32). Drop this into :func:`linear_fp8`
        as the activation scale on sm_120.

    Constraints: ``K % 512 == 0`` (4 K-blocks per packed int32).

    Use this on sm_120 instead of ``quantize_1x128_fp8 + repack_fp8_act_scales``.
    On sm_90 stick with ``quantize_1x128_fp8`` (the deep_gemm path consumes
    FP32 scales directly).
    """
    return torch.ops.fish_scales_ops.quantize_1x128_packed(x.contiguous(), True)


def quantize_128x128_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [N, K] weight → FP8 (E4M3) [N, K] + FP32 per-128×128-block dequant scales.

    Returns:
        (w_fp8, sw) where sw is float32 [ceil(N,128), ceil(K,128)].
    """
    return torch.ops.fish_scales_ops.quantize_128x128(w.contiguous())


def repack_fp8_act_scales(sx_f32: torch.Tensor) -> torch.Tensor:
    """FP32 1×128 act scales [pad(M,4), K/128] → int32 K-major [pad(M,4), K/512].

    **sm_120 only.** Pre-pack activation scales once when reusing across
    calls and pass the int32 result to ``linear_fp8``; the wrapper takes
    the fast path and skips the ~9 μs per-call repack. On sm_90 the
    deep_gemm kernel consumes the FP32 scales directly — do NOT call
    this on sm_90 and pass the int32 output to ``linear_fp8``; the
    runner expects FP32 there.
    """
    return torch.ops.fish_scales_ops.repack_fp8_act_scales(sx_f32.contiguous())


def repack_fp8_wgt_scales(sw_f32: torch.Tensor) -> torch.Tensor:
    """FP32 128×128 wgt scales [N/128, K/128] → int32 K-major [pad(N,4), K/512].

    **sm_120 only.** Pre-pack weight scales once per cached weight
    tensor; the 128×128 block scales get row-expanded across 128 N rows
    in the output. Pass the int32 result to ``linear_fp8`` to skip the
    per-call repack on sm_120. Do NOT call this on sm_90 — the deep_gemm
    kernel reads FP32 weight scales directly.
    """
    return torch.ops.fish_scales_ops.repack_fp8_wgt_scales(sw_f32.contiguous())


def linear_qx(x_bf16: torch.Tensor, w_fp8: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    """y = x @ w.T with x quantized inside the op and w pre-quantized.

    Equivalent to ``linear_fp8(*quantize_1x128_fp8(x), w_fp8, sw)`` but
    fuses the activation quantize launch with the GEMM. The cached-weight
    inference path.
    """
    return torch.ops.fish_scales_ops.linear_qx(
        x_bf16.contiguous(), w_fp8.contiguous(), sw.contiguous()
    )


# ---------------------------------------------------------------------------
# Grouped (MoE, masked) block-scale FP8 on sm_90 (H200) — H1.
#
# Revives the in-tree DeepGEMM GroupedMasked WGMMA kernel. Scales are FP32
# (sm_90 convention), passed straight to the same kernel dense linear_fp8
# uses. Per-group SFA layout = the dense quantize_1x128 output stacked:
# [G, K/128, pad(m_cap,4)] K-major (the kernel's TMA descriptor reads it as
# ColMajor [pad(m_cap,4), (K/128)*G]); SFB = per-expert 128x128 scales
# [G, N/128, K/128]. H2 replaces the caller-side scale prep with an fso
# layout-native fused gather-quant.


def linear_fp8_grouped_masked(a_fp8, w_fp8, sa, sw, masked_m, expected_m):
    """sm_90 grouped block-scale FP8 masked GEMM (DeepGEMM GroupedMasked).

    a_fp8 [G, m_cap, K] fp8, w_fp8 [G, N, K] fp8, sa/sw FP32 scales in the
    DeepGEMM masked layout, masked_m [G] int32 on device, expected_m host int.
    Returns bf16 [G, m_cap, N].
    """
    from .._arch import sm_major
    if sm_major() != 9:
        raise NotImplementedError(
            "linear_fp8_grouped_masked is the sm_90 (H200) grouped path; "
            "sm_120 uses the MXFP8 grouped path"
        )
    return torch.ops.fish_scales_ops.linear_fp8_grouped_masked(
        a_fp8, w_fp8, sa, sw, masked_m, expected_m
    )


def quantize_1x128_grouped_gather_sm90(x, slot_of_flat, topk, num_groups, m_cap):
    """H2 sm_90 layout-native fused gather + block-scale FP8 quantize.

    Emits the grouped SFA layout the GroupedMasked kernel reads directly
    ([G, K/128, m_cap] FP32 K-major) — no deep_gemm per_token_cast +
    tma_align two-step. Returns (a_fp8 [G, m_cap, K], sa [G, K/128, m_cap]).
    """
    from .._arch import sm_major
    if sm_major() != 9:
        raise NotImplementedError("sm_90 (H200) grouped path only")
    return torch.ops.fish_scales_ops.quantize_1x128_grouped_gather_sm90(
        x.contiguous(), slot_of_flat, topk, num_groups, m_cap)


def silu_chunk_mul_quantize_1x128_grouped_sm90(gu, slot_of_flat):
    """H2 sm_90 grouped SwiGLU + block-scale FP8 quantize (layout-native).

    gu is the grouped gate_up output [G, m_cap, 2*INTER]. Returns
    (h_fp8 [G, m_cap, INTER], sh [G, INTER/128, m_cap] FP32 K-major).
    """
    from .._arch import sm_major
    if sm_major() != 9:
        raise NotImplementedError("sm_90 (H200) grouped path only")
    return torch.ops.fish_scales_ops.silu_chunk_mul_quantize_1x128_grouped_sm90(
        gu, slot_of_flat)


# ---- H4 contiguous (triton-style sorted layout) sm_90 grouped path ----------

def linear_fp8_grouped_contiguous(a_fp8, w_fp8, sa, sw, sorted_expert_ids,
                                  block_m, expected_m):
    """sm_90 grouped block-scale FP8 contiguous GEMM (DeepGEMM
    GroupedContiguous). Consumes the expert-sorted layout from
    moe_build_sorted; work tracks the active padded blocks, not the expert
    count E.

    a_fp8 [P_max, K] fp8 (expert-sorted), w_fp8 [G, N, K] fp8, sa = SFA
    ColMajor [K/128, align(P_max,4)], sw = SFB [G, N/128, K/128],
    sorted_expert_ids [P_max] int32 (grouped_layout, -1 past the real length),
    block_m (= the moe_build_sorted padding granularity, 64 for H4a),
    expected_m host int. Returns bf16 [P_max, N] in sorted order.
    """
    from .._arch import sm_major
    if sm_major() != 9:
        raise NotImplementedError("sm_90 (H200) grouped path only")
    return torch.ops.fish_scales_ops.linear_fp8_grouped_contiguous(
        a_fp8, w_fp8, sa, sw, sorted_expert_ids, block_m, expected_m)


# M-dispatch threshold for the composed sm_90 MoE layer. Below it the layer is
# memory-bound (decode / small batch) and the swap-AB block_n=16 path wins; at
# or above it the layer turns compute-bound (prefill) and the non-swap
# block_m=64 contiguous path wins — swap-AB's tiny activation tile regresses
# hard there (measured H200: crossover ~M=224; M=4096 swap is 1.9x slower).
MOE_SWAP_M_MAX = 256


def moe_layer_fp8_sm90(hidden, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w):
    """Complete sm_90 (H200) grouped-MoE layer, best GEMM path auto-selected by
    M behind a single stable interface. Composes the 6-kernel expert-sorted
    layer (build-sorted + gather-quant + gate_up + silu-quant + down + combine)
    and routes decode / small batch (M < MOE_SWAP_M_MAX) to the swap-AB
    block_n=16 GEMMs and prefill (M >= MOE_SWAP_M_MAX) to the non-swap
    block_m=64 contiguous GEMMs — the measured best at every M.

    The choice is a host-side branch on M (tensor shapes are static per shape),
    so each M captures into its own CUDA graph safely.

    Args:
        hidden:  bf16 [M, HIDDEN].
        w13_fp8: float8_e4m3fn [E, 2*INTER, HIDDEN]; sw13 fp32 [E, 2*INTER/128, HIDDEN/128].
        w2_fp8:  float8_e4m3fn [E, HIDDEN, INTER];   sw2  fp32 [E, HIDDEN/128, INTER/128].
        topk_ids: int32 [M, topk]; topk_w: fp32 [M, topk].
    Returns:
        bf16 [M, HIDDEN].
    """
    from .._arch import sm_major
    if sm_major() != 9:
        raise NotImplementedError("moe_layer_fp8_sm90 is the sm_90 (H200) grouped path")
    ops = torch.ops.fish_scales_ops
    M = int(hidden.shape[0])
    E = int(w13_fp8.shape[0])
    topk = int(topk_ids.shape[1])
    expected_m = max(1, (M * topk + E - 1) // E)
    use_swap = M < MOE_SWAP_M_MAX
    block = 16 if use_swap else 64

    se, fts, _ = ops.moe_build_sorted(topk_ids, E, block)
    p_max = int(se.shape[0])
    hq, sh = ops.quantize_1x128_sorted_gather_sm90(hidden.contiguous(), fts, p_max, topk)
    if use_swap:
        gu = ops.linear_fp8_grouped_contiguous_swapab(hq, w13_fp8, sh, sw13, se, block, expected_m)
    else:
        gu = ops.linear_fp8_grouped_contiguous(hq, w13_fp8, sh, sw13, se, block, expected_m)
    dq, sd = ops.silu_chunk_mul_quantize_1x128_sorted_sm90(gu, fts)
    if use_swap:
        dn = ops.linear_fp8_grouped_contiguous_swapab(dq, w2_fp8, sd, sw2, se, block, expected_m)
    else:
        dn = ops.linear_fp8_grouped_contiguous(dq, w2_fp8, sd, sw2, se, block, expected_m)
    return ops.moe_combine_sorted(dn, fts, topk_w)


def linear_fp8_grouped_contiguous_swapab(a_fp8, w_fp8, sa, sw, sorted_expert_ids,
                                         block_n, expected_m):
    """sm_90 grouped contiguous GEMM, swap-AB (block_n=16 activation tiling for
    M>=8). Same layout as linear_fp8_grouped_contiguous but the activation is
    the swap-AB B matrix and the weight the A matrix; cuts per-expert padding
    4x at M>=8. sorted_expert_ids must be built with padding = block_n.
    """
    from .._arch import sm_major
    if sm_major() != 9:
        raise NotImplementedError("sm_90 (H200) grouped path only")
    return torch.ops.fish_scales_ops.linear_fp8_grouped_contiguous_swapab(
        a_fp8, w_fp8, sa, sw, sorted_expert_ids, block_n, expected_m)


def quantize_1x128_sorted_gather_sm90(x, flat_to_sorted, p_max, topk):
    """H4 sm_90 fused gather + block-scale FP8 quantize into the contiguous
    sorted layout. x [M, K] bf16, flat_to_sorted [R] (pair -> sorted row),
    p_max (output rows). Iterates the R real pairs (padding rows left
    uninitialised — row-independent GEMM + combine drop them). Returns
    (x_q [P_max, K] fp8, sa [K/128, align(P_max,4)] FP32 K-major).
    """
    from .._arch import sm_major
    if sm_major() != 9:
        raise NotImplementedError("sm_90 (H200) grouped path only")
    return torch.ops.fish_scales_ops.quantize_1x128_sorted_gather_sm90(
        x.contiguous(), flat_to_sorted, p_max, topk)


def silu_chunk_mul_quantize_1x128_sorted_sm90(gu, flat_to_sorted):
    """H4 sm_90 SwiGLU + block-scale FP8 requantize into the contiguous sorted
    layout. gu [P_max, 2*INTER] bf16 (sorted gate_up output), flat_to_sorted
    [R]. Returns (h_fp8 [P_max, INTER], sh [INTER/128, align(P_max,4)]).
    """
    from .._arch import sm_major
    if sm_major() != 9:
        raise NotImplementedError("sm_90 (H200) grouped path only")
    return torch.ops.fish_scales_ops.silu_chunk_mul_quantize_1x128_sorted_sm90(
        gu, flat_to_sorted)


def quantize_moe_weights_1x128_fp8_sm90(w):
    """Offline per-expert 128x128 weight quantize for the sm_90 grouped GEMM.

    w bf16 [G, N, K]. Loops the dense 128x128 quantizer and stacks; the
    grouped kernel reads SFB as [G, N/128, K/128] FP32 (per_block_cast
    convention). Returns (w_fp8 [G, N, K], sw [G, N/128, K/128]).
    """
    from .._arch import sm_major
    if sm_major() != 9:
        raise NotImplementedError("sm_90 (H200) grouped path only")
    if w.dim() != 3:
        raise ValueError("w must be [G, N, K]")
    G, N, K = w.shape
    if N % 128 != 0 or K % 128 != 0:
        raise ValueError("N and K must be multiples of 128")
    w_fp8 = torch.empty(G, N, K, device=w.device, dtype=torch.float8_e4m3fn)
    sw = torch.empty(G, N // 128, K // 128, device=w.device, dtype=torch.float32)
    for g in range(G):
        q, s = torch.ops.fish_scales_ops.quantize_128x128(w[g].contiguous())
        w_fp8[g].copy_(q)
        sw[g].copy_(s)
    return w_fp8, sw
