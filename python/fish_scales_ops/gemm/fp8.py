"""1×128 / 128×128 FP8 ops — quantize, GEMM, scale repack, and the
internal-quantize ``linear_qx`` variant.

Auto-dispatches sm_90 deep_gemm, sm_120 CUTLASS BlockScaledKernel and, since
2026-09-05, sm_100 / sm_103 through the MXFP8 tcgen05 tiers with replicated
scales.
"""
from __future__ import annotations

import torch


def linear_fp8(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    sx: torch.Tensor,
    sw: torch.Tensor,
) -> torch.Tensor:
    """Block-scaled FP8 (E4M3) GEMM. Auto-dispatches sm_90 / sm_100 / sm_120.

    On sm_120 (Blackwell sm_120a) and sm_100 / sm_103 (B200 / B300) the
    scales must be UE8M0 (powers of two), which is what both quantizers produce
    there by default. Pass them as FP32 (packed per call) or pre-packed int32
    from ``repack_fp8_{act,wgt}_scales`` / ``quantize_1x128_fp8_packed``, both
    in the same form: on sm_120 / sm_121 one FP32 and one int32 scale raise
    ``RuntimeError``, while on sm_100 / sm_103 the FP32 one is packed here.
    On sm_100 / sm_103 the GEMM runs on the MXFP8 tcgen05 tiers with each 1×128
    scale byte replicated into its four 32-wide slots. The per-call packing of
    FP32 scales does not check them and does not synchronize the device, on
    sm_100 / sm_103 as on sm_120 / sm_121, so the call can be captured into a
    CUDA graph with either scale form; a scale that is not a power of two is
    truncated to the power of two below it without an error, which is why the
    quantizers default to UE8M0 there. Pre-packing the weight scales once with
    ``repack_fp8_wgt_scales`` (which does check) saves the per-call packing.
    On sm_90 (Hopper / H200) the scales are FP32 as the quantizers produce them
    there; the int32 pre-packed forms are the Blackwell layout and are refused.

    Args:
        x_fp8: float8_e4m3fn [M, K], contiguous.
        w_fp8: float8_e4m3fn [N, K], contiguous.
        sx:    float32 dequant scales for x (see quantize helper output), or
               the pre-packed int32 form.
        sw:    float32 dequant scales for w, or the pre-packed int32 form.

    Returns:
        bfloat16 [M, N].
    """
    from .._arch import sm_major
    major = sm_major()
    if major == 9 and (sx.dtype != torch.float32 or sw.dtype != torch.float32):
        raise ValueError(
            "linear_fp8 on sm_90 takes FP32 scales from quantize_1x128_fp8 / quantize_128x128_fp8; "
            f"got sx {sx.dtype} and sw {sw.dtype}. The int32 pre-packed scales are the Blackwell "
            "(sm_100/103, sm_120/121) layout.")
    if major == 10:
        # sm_100 / sm_103: expand 1x128 scales to the atom layout, then take the
        # MXFP8 router (the CuTe-DSL decode row, cuBLAS scaled_mm, the CuTe DSL
        # tier, the C++ cascade). The per-call repack skips the power-of-two
        # check (check=False): the check copies a flag to the host, which would
        # synchronize every call and make the call uncapturable. Same kernels,
        # same bytes as the checked public wrappers.
        ops = torch.ops.fish_scales_ops
        if sx.dtype == torch.float32:
            sx = ops.repack_fp8_act_scales(sx.contiguous(), False)
        if sw.dtype == torch.float32:
            sw = ops.repack_fp8_wgt_scales(sw.contiguous(), False)
        from .mxfp8 import linear_mxfp8
        return linear_mxfp8(x_fp8.contiguous(), w_fp8.contiguous(), sx, sw)
    return torch.ops.fish_scales_ops.linear_fp8(
        x_fp8.contiguous(),
        w_fp8.contiguous(),
        sx.contiguous(),
        sw.contiguous(),
    )


def quantize_1x128_fp8(
    x: torch.Tensor,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [..., K] → FP8 (E4M3) [..., K] + FP32 per-row 1×128 dequant scales.

    Args:
        x: bf16 tensor with K divisible by 128.
        use_ue8m0: round each scale up to a power of two (UE8M0-exact FP32).
            ``None`` (default) resolves to ``True`` on sm_100 / sm_103 and
            sm_120 / sm_121 and to ``False`` on sm_90, the same rule as
            :func:`quantize_128x128_fp8`. The Blackwell GEMMs read each scale
            as its exponent byte, so a scale that is not a power of two would
            be truncated to the power of two below it and the block
            dequantized 0.5-1.0x too small, with no error; pass ``False``
            there only when the scales are not going to :func:`linear_fp8`.
            The sm_90 deep_gemm path takes FP32 scales as they are.

    Returns:
        (x_fp8, sx) where sx is float32 [pad(M,4), K/128] (TMA-aligned;
        the trailing pad rows hold zeros).
    """
    if use_ue8m0 is None:
        from .._arch import sm_major
        use_ue8m0 = sm_major() >= 10
    return torch.ops.fish_scales_ops.quantize_1x128(x.contiguous(), use_ue8m0)


def _require_blackwell_scale_layout(name: str) -> None:
    """The int32 pre-packed scale layout exists only for the Blackwell GEMMs;
    on sm_90 it would be read back as FP32 by the deep_gemm path."""
    from .._arch import sm_major
    major = sm_major()
    if major < 10:
        raise NotImplementedError(
            f"{name} produces the int32 UE8M0 scale layout of the Blackwell GEMMs "
            f"(sm_100/103, sm_120/121); this device is sm_{major}x. On sm_90 pass the FP32 "
            "scales of quantize_1x128_fp8 / quantize_128x128_fp8 to linear_fp8 directly.")


def quantize_1x128_fp8_packed(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [..., K] → FP8 (E4M3) [..., K] + int32-packed UE8M0 1×128 scales.

    Fused single-kernel replacement for
    ``repack_fp8_act_scales(quantize_1x128_fp8(x, use_ue8m0=True)[1])`` on
    sm_120 / sm_121 and sm_100 / sm_103 — no FP32 scale round-trip through
    global memory, no separate repack launch, and no device synchronization.

    Returns:
        (x_fp8, sx_packed) — the arch-native packed int32 scales: on sm_120 /
        sm_121 K-major ``[pad(M,4), ceil(K/512)]`` (4 UE8M0 bytes per int32),
        on sm_100 / sm_103 the 1-D Sm1xx atom layout. Drop this into
        :func:`linear_fp8` as the activation scale, with a pre-packed weight
        scale.

    Constraints: ``K % 128 == 0``. The fused kernel covers ``K % 512 == 0``;
    other K take the two-step path (FP32 quantize + an unchecked repack) inside
    the op, which stays capturable; on sm_120 / sm_121 the last packed word's
    unused bytes are zero (never read by the GEMM).

    Use this on Blackwell instead of ``quantize_1x128_fp8 + repack_fp8_act_scales``.
    On sm_90 it raises; stick with ``quantize_1x128_fp8`` (the deep_gemm path
    consumes FP32 scales directly).
    """
    _require_blackwell_scale_layout("quantize_1x128_fp8_packed")
    return torch.ops.fish_scales_ops.quantize_1x128_packed(x.contiguous(), True)


def quantize_128x128_fp8(
    w: torch.Tensor,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [N, K] weight → FP8 (E4M3) [N, K] + FP32 per-128×128-block dequant scales.

    Args:
        w: bf16 [N, K].
        use_ue8m0: round each block scale up to a power of two (UE8M0-exact
            FP32). ``None`` (default) resolves to ``True`` on sm_100 / sm_103
            and sm_120, ``False`` on sm_90. The Blackwell GEMMs consume scales as UE8M0
            exponent bytes, so a plain ``amax/448`` scale would be truncated
            to the power of two below it and the block dequantised 0.5–1.0×
            too small (silent accuracy loss, fixed 2026-09-05). The sm_90
            deep_gemm path takes FP32 scales as they are.

    Returns:
        (w_fp8, sw) where sw is float32 [ceil(N,128), ceil(K,128)].
    """
    if use_ue8m0 is None:
        from .._arch import sm_major
        use_ue8m0 = sm_major() >= 10
    return torch.ops.fish_scales_ops.quantize_128x128(w.contiguous(), use_ue8m0)


def repack_fp8_act_scales(sx_f32: torch.Tensor) -> torch.Tensor:
    """FP32 1×128 act scales [pad(M,4), K/128] → arch-native packed int32 scales
    (sm_120 / sm_121: K-major [pad(M,4), ceil(K/512)]; sm_100 / sm_103: the 1-D
    Sm1xx atom layout).

    **sm_120 and sm_100 / sm_103.** Pre-pack activation scales once when reusing across
    calls and pass the int32 result to ``linear_fp8``; the wrapper then skips
    its per-call repack. The op checks that every scale is a power of two,
    which synchronizes the device: call it at load time, outside any CUDA-graph
    capture. (The per-call packing inside ``linear_fp8`` reaches the same
    kernels through the op's ``check=False`` form, which neither checks nor
    synchronizes.) On sm_90 it raises: the deep_gemm kernel consumes the FP32
    scales directly.
    """
    _require_blackwell_scale_layout("repack_fp8_act_scales")
    return torch.ops.fish_scales_ops.repack_fp8_act_scales(sx_f32.contiguous())


def repack_fp8_wgt_scales(sw_f32: torch.Tensor) -> torch.Tensor:
    """FP32 128×128 wgt scales [N/128, K/128] → arch-native packed int32 scales
    (sm_120 / sm_121: K-major [pad(N,4), ceil(K/512)]; sm_100 / sm_103: the 1-D
    Sm1xx atom layout).

    **sm_120 and sm_100 / sm_103.** Pre-pack weight scales once per cached
    weight tensor; the 128×128 block scales get row-expanded across 128 N rows
    in the output. Pass the int32 result to ``linear_fp8`` to skip the
    per-call repack. Like :func:`repack_fp8_act_scales` it checks the scales and
    synchronizes the device, so it belongs at load time. On sm_90 this raises:
    the deep_gemm kernel reads FP32 weight scales directly.
    """
    _require_blackwell_scale_layout("repack_fp8_wgt_scales")
    return torch.ops.fish_scales_ops.repack_fp8_wgt_scales(sw_f32.contiguous())


def linear_qx(x_bf16: torch.Tensor, w_fp8: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    """y = x @ w.T with x quantized inside the op and w pre-quantized.

    Equivalent to ``linear_fp8(*quantize_1x128_fp8(x), w_fp8, sw)`` but
    fuses the activation quantize launch with the GEMM. The cached-weight
    inference path. ``sw`` is the FP32 output of :func:`quantize_128x128_fp8`
    on the same device.

    sm_90 and sm_120 / sm_121, where the result is bit-identical to the
    two-step form (``tests/gemm/unit/test_linear_qx.py``). On sm_100 / sm_103
    the op raises ``RuntimeError``; use :func:`quantize_1x128_fp8_packed` and
    :func:`linear_fp8` with pre-packed weight scales there.
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


def linear_fp8_grouped_contiguous_2wg(a_fp8, w_fp8, sa, sw, sorted_expert_ids,
                                      block_m, expected_m):
    """sm_90 grouped contiguous FC1, non-swap, with two math warp-groups split
    along N (the structural step toward the fused SwiGLU + 1x128 requantize FC1).
    Internal: not exported from ``fso.compat``; kept for tests and A/B
    measurements.

    Same arguments and result as :func:`linear_fp8_grouped_contiguous` for the
    stacked ``[gate; up]`` FC1 weight ``w_fp8 [G, 2I, K]`` (``2I % 256 == 0``):
    one CTA computes gate block b (warp-group 0) and up block b (warp-group 1)
    of one 64-row activation tile, and both bf16 halves land in their native
    columns of ``[P_max, 2I]``, bit-identical to today's FC1 on the rows the
    GEMM computes. ``block_m`` must be 64. :func:`moe_layer_fp8_sm90` uses its
    fused form, :func:`linear_fp8_grouped_contiguous_swiglu`.
    """
    from .._arch import sm_major
    if sm_major() != 9:
        raise NotImplementedError("sm_90 (H200) grouped path only")
    return torch.ops.fish_scales_ops.linear_fp8_grouped_contiguous_2wg(
        a_fp8, w_fp8, sa, sw, sorted_expert_ids, block_m, expected_m)


def linear_fp8_grouped_contiguous_swiglu(a_fp8, w_fp8, sa, sw, sorted_expert_ids,
                                         block_m, expected_m):
    """sm_90 grouped contiguous FC1, non-swap, with the SwiGLU + 1x128 FP8
    requantize fused into its epilogue. Internal: the non-swap FC1 of
    :func:`moe_layer_fp8_sm90`, not exported from ``fso.compat``.

    Same arguments as :func:`linear_fp8_grouped_contiguous` for the stacked
    ``[gate; up]`` FC1 weight ``w_fp8 [G, 2I, K]`` (``2I % 256 == 0``,
    ``K % 128 == 0``, ``block_m == 64``). Instead of the bf16 ``[P_max, 2I]``
    FC1 output it returns what :func:`silu_chunk_mul_quantize_1x128_sorted_sm90`
    makes of that output, in the same layouts: ``(dq [P_max, I] fp8,
    sd [I/128, align4(P_max)] fp32)``, bit-identical on every routed row (the
    rows ``flat_to_sorted`` assigns). Padding rows inside a visited 64-row block
    hold the SwiGLU of the gather's uninitialised padding; FC2 is row-independent
    and the combine reads only routed rows.
    """
    from .._arch import sm_major
    if sm_major() != 9:
        raise NotImplementedError("sm_90 (H200) grouped path only")
    return torch.ops.fish_scales_ops.linear_fp8_grouped_contiguous_swiglu(
        a_fp8, w_fp8, sa, sw, sorted_expert_ids, block_m, expected_m)


# Routing of the composed sm_90 MoE layer, set by the 2026-10-01 retune with
# every FC1 route fused (measurements: /data/bench-runs/sm90_retune_20261001).
# The swap-AB GEMMs tile the activation by block_n rows and stream an expert's
# weights once per activation tile, so the route is chosen from the routed rows
# per active expert, rows = M * topk / E, not from M alone. Up to 12 rows per
# expert the 16-row swap-AB tile is fastest, and up to 24 rows the 32-row tile.
# Above 24 rows the non-swap path (64-row blocks) is fastest. Its FC2 runs the
# swap-AB GEMM with a 64-row tile, which reads the same 64-row padded layout,
# gives a bit-identical result and is faster than the non-swap FC2 at every
# measured shape (see `_sm90_fc2_swap`). The 64-row swap-AB tile therefore no
# longer has a band of its own. Both measured expert shapes (I/128 = 4 and 6)
# put both crossovers at the same rows per expert, so the rule needs no other
# quantity. The one exception, the 64-row swap-AB tile being up to 1.8 % faster
# at 96 to 128 rows when I/128 = 4, is left out rather than keyed on I.
# FSO_SWAP_BN=16|32|64|0 forces one tile (0 = non-swap) and FSO_FC2_SWAP=0|1
# forces the FC2 GEMM of the non-swap path, for A/B runs.
MOE_SWAP_BLOCK_N_CASCADE = ((12, 16), (24, 32))


def moe_swap_ab_block_n(M, num_experts, topk):
    """Activation tile of the swap-AB path for this routing: 16 up to 12 routed
    rows per active expert and 32 up to 24 (``MOE_SWAP_BLOCK_N_CASCADE``), or
    ``None`` above that, where the layer takes the non-swap block_m=64 path.
    ``FSO_SWAP_BN=16|32|64|0`` forces a tile (``0`` = the non-swap path)."""
    import os
    forced = os.environ.get("FSO_SWAP_BN")
    if forced is not None:
        v = int(forced)
        return None if v == 0 else v
    rows = int(M) * int(topk) / float(num_experts)
    for max_rows, block_n in MOE_SWAP_BLOCK_N_CASCADE:
        if rows <= max_rows:
            return block_n
    return None


def moe_swap_ab_max_m(num_experts, topk):
    """Largest M the composed sm_90 layer still routes to the swap-AB GEMMs
    (inclusive): M * topk <= 24 * E, the last step of the cascade."""
    return (MOE_SWAP_BLOCK_N_CASCADE[-1][0] * int(num_experts)) // int(topk)


# FSO_FC1_FUSED, the switch the sm_100/sm_103 and sm_120/sm_121 fused FC1 already
# reads (csrc/gemm/ops/mxfp8.cu: fso_fc1_fused_enabled): `0` never uses the fused
# FC1, unset applies the rule of `_sm90_layer_plan`, and any other value applies it
# as well (on sm_90 the rule takes the fused FC1 on every route where it is legal,
# swap-AB and non-swap alike).
# Read once per process, so a CUDA graph and its replays cannot disagree.
_sm90_fc1_fused_env: bool | None = None


def _sm90_fc1_fused_allowed():
    global _sm90_fc1_fused_env
    if _sm90_fc1_fused_env is None:
        import os
        v = os.environ.get("FSO_FC1_FUSED")
        _sm90_fc1_fused_env = not (v is not None and v[:1] == "0")
    return _sm90_fc1_fused_env


def _sm90_layer_plan(M, num_experts, topk, hidden, inter):
    """The host-static decisions of one expert-sorted sm_90 layer call:
    ``(expected_m, use_swap, block, p_max, fused_fc1)``. `moe_layer_fp8_sm90` runs
    this plan and `moe_layer_transient_bytes_sm90` sizes it; ``p_max`` repeats
    ``moe_build_sorted``'s worst-case padded length (every active expert wastes up
    to ``block - 1`` rows, rounded up to a block multiple, moe_glue.cu).

    ``use_swap`` and ``block``: `moe_swap_ab_block_n` sends up to 12 routed rows
    per active expert to the swap-AB GEMMs with a 16-row activation tile and up to
    24 rows to the 32-row tile; above that the layer takes the non-swap path, which
    pads the sorted layout to 64-row blocks (``block == 64``) and runs its FC2 as
    `_sm90_fc2_swap` decides.

    ``fused_fc1``: every route runs the FC1 with the SwiGLU + 1x128 requantize
    fused into its epilogue instead of the FC1 + SwiGLU kernel pair, the swap-AB
    route through `linear_fp8_grouped_contiguous_swapab_swiglu` (its block_n
    tile) and the non-swap route through `linear_fp8_grouped_contiguous_swiglu`.
    It needs whole 128-column SwiGLU blocks (``2 * inter % 256 == 0``) and
    ``hidden % 128 == 0``, and ``FSO_FC1_FUSED=0`` restores the pair on every
    route."""
    rows = int(M) * int(topk)
    expected_m = max(1, (rows + num_experts - 1) // num_experts)
    block_n = moe_swap_ab_block_n(M, num_experts, topk)
    use_swap = block_n is not None
    block = block_n if use_swap else 64
    active_max = min(rows, int(num_experts))
    p_max = (rows + active_max * (block - 1) + block - 1) // block * block
    fused_fc1 = ((2 * int(inter)) % 256 == 0 and int(hidden) % 128 == 0
                 and _sm90_fc1_fused_allowed())
    return expected_m, use_swap, block, p_max, fused_fc1


def _sm90_fc2_swap(use_swap):
    """Whether the down projection (FC2) of an sm_90 layer call runs the swap-AB
    GEMM. A swap-AB route pads the sorted layout to its own block_n tile, so its
    FC2 is always swap-AB. The non-swap route pads to 64 rows per block, which is
    also the layout of the 64-wide swap-AB tile, so its FC2 may take either GEMM
    (the two give bit-identical results). The rule takes the swap-AB GEMM with
    block_n 64 there, because it was faster than the non-swap FC2 at every
    measured shape of both expert shapes in the 2026-10-01 retune
    (/data/bench-runs/sm90_retune_20261001). ``FSO_FC2_SWAP=1`` forces the swap-AB
    FC2 on the non-swap route and ``FSO_FC2_SWAP=0`` the non-swap one, for A/B
    runs; unset (or empty) applies the rule. Read on every call, like
    ``FSO_SWAP_BN``: set it before the first call and keep it between a capture
    and its replays."""
    if use_swap:
        return True
    import os
    forced = os.environ.get("FSO_FC2_SWAP")
    if forced:
        return forced[:1] != "0"
    return True


def moe_layer_transient_bytes_sm90(tokens, num_experts, topk, hidden, inter):
    """Device memory one :func:`moe_layer_fp8_sm90` call allocates for a bucket of
    ``tokens`` rows, in bytes: the expert-sorted routing tensors, the gathered FP8
    activation and its scales, the bf16 gate/up output (only with ``FSO_FC1_FUSED=0``:
    the fused FC1, which every route takes otherwise, writes the SwiGLU output
    directly), the FP8 SwiGLU output and
    its scales, the bf16 down output and the ``[tokens, hidden]`` result, each
    charged as the caching allocator can charge it (512-byte blocks, and up to 1 MiB
    more for a tensor above 1 MiB that a cached block serves unsplit), so the figure
    is an upper bound on what the call occupies. The sorted layout sizes
    everything by the routed rows (``tokens * topk`` plus at most ``block - 1``
    padding rows per active expert), so unlike the masked slab this grows with
    ``topk`` rather than with the expert count. Inputs, weights and the library's
    persistent pools are not counted. The layer runs every bucket as one call; this
    is what a caller reserves for its largest bucket.
    """
    for name, v in (("tokens", tokens), ("num_experts", num_experts), ("topk", topk),
                    ("hidden", hidden), ("inter", inter)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise ValueError(f"moe_layer_transient_bytes_sm90: {name} must be a non-negative int, got {v!r}")
    if tokens == 0:
        return 0
    if num_experts < 1 or topk < 1 or hidden % 128 or inter % 128:
        raise ValueError(
            "moe_layer_transient_bytes_sm90: needs num_experts >= 1, topk >= 1 and hidden and inter "
            f"multiples of 128, got num_experts={num_experts}, topk={topk}, hidden={hidden}, inter={inter}")

    def a(nbytes):
        # As the caching allocator charges a tensor, at most: 512-byte blocks, and up to
        # 1 MiB more above 1 MiB, where a cached block serves the request unsplit.
        n = int(nbytes)
        if n <= 0:
            return 0
        r = (n + 511) // 512 * 512
        return r + (1 << 20) if r > (1 << 20) else r

    _, _, _, p_max, fused_fc1 = _sm90_layer_plan(tokens, num_experts, topk, hidden, inter)
    ld = (p_max + 3) // 4 * 4
    total = a(4 * p_max) + a(4 * tokens * topk) + a(4)          # sorted ids, pair -> row, padded length
    total += a(p_max * hidden) + a(4 * (hidden // 128) * ld)     # gathered FP8 activation + scales
    if not fused_fc1:
        total += a(2 * p_max * 2 * inter)                        # bf16 gate/up output
    total += a(p_max * inter) + a(4 * (inter // 128) * ld)       # FP8 SwiGLU output + scales
    total += a(2 * p_max * hidden)                               # bf16 down output
    total += a(2 * tokens * hidden)                              # the combined result
    return total


def moe_layer_fp8_sm90(hidden, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w):
    """Complete sm_90 (H200) grouped-MoE layer, best GEMM path auto-selected by
    M behind a single stable interface. Composes the 6-kernel expert-sorted
    layer (build-sorted + gather-quant + gate_up + silu-quant + down + combine)
    and routes up to 24 routed rows per active expert (moe_swap_ab_block_n
    picks block_n = 16 up to 12 rows and 32 up to 24) to the swap-AB GEMMs and
    larger M to the non-swap path: the block_m=64 contiguous gate_up GEMM, then
    the down GEMM as the swap-AB GEMM with block_n 64 on the same 64-row layout
    (bit-identical to the non-swap down GEMM, which ``FSO_FC2_SWAP=0`` restores,
    see `_sm90_fc2_swap`). On both paths the gate_up GEMM carries the SwiGLU +
    1x128 requantize in its epilogue (five kernels, bit-identical to the six;
    ``FSO_FC1_FUSED=0`` restores the pair, see `_sm90_layer_plan`).

    The choice is a host-side branch on M (tensor shapes are static per shape),
    so each M captures into its own CUDA graph safely.

    Args:
        hidden:  bf16 [M, HIDDEN].
        w13_fp8: float8_e4m3fn [E, 2*INTER, HIDDEN]; sw13 fp32 [E, 2*INTER/128, HIDDEN/128].
        w2_fp8:  float8_e4m3fn [E, HIDDEN, INTER];   sw2  fp32 [E, HIDDEN/128, INTER/128].
        topk_ids: int32 [M, topk]; topk_w: fp32 [M, topk]. Entries outside
            [0, E) are skipped (sglang masks the rows past num_token_non_padded
            of a graph bucket to -1 or to num_experts): they take no expert
            compute and contribute nothing, and a token whose entries are all
            masked gets a zero output row.
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
    expected_m, use_swap, block, _, fused_fc1 = _sm90_layer_plan(
        M, E, topk, int(w13_fp8.shape[2]), int(w13_fp8.shape[1]) // 2)

    se, fts, _ = ops.moe_build_sorted(topk_ids, E, block)
    p_max = int(se.shape[0])
    hq, sh = ops.quantize_1x128_sorted_gather_sm90(hidden.contiguous(), fts, p_max, topk)
    if fused_fc1:
        if use_swap:
            dq, sd = ops.linear_fp8_grouped_contiguous_swapab_swiglu(hq, w13_fp8, sh, sw13, se, block, expected_m)
        else:
            dq, sd = ops.linear_fp8_grouped_contiguous_swiglu(hq, w13_fp8, sh, sw13, se, block, expected_m)
    else:
        if use_swap:
            gu = ops.linear_fp8_grouped_contiguous_swapab(hq, w13_fp8, sh, sw13, se, block, expected_m)
        else:
            gu = ops.linear_fp8_grouped_contiguous(hq, w13_fp8, sh, sw13, se, block, expected_m)
        dq, sd = ops.silu_chunk_mul_quantize_1x128_sorted_sm90(gu, fts)
    if _sm90_fc2_swap(use_swap):
        dn = ops.linear_fp8_grouped_contiguous_swapab(dq, w2_fp8, sd, sw2, se, block, expected_m)
    else:
        dn = ops.linear_fp8_grouped_contiguous(dq, w2_fp8, sd, sw2, se, block, expected_m)
    return ops.moe_combine_sorted(dn, fts, topk_w)


def linear_fp8_grouped_contiguous_swapab(a_fp8, w_fp8, sa, sw, sorted_expert_ids,
                                         block_n, expected_m):
    """sm_90 grouped contiguous GEMM, swap-AB (block_n = 16 / 32 / 64 activation
    tiling). Same layout as linear_fp8_grouped_contiguous but the activation is
    the swap-AB B matrix and the weight the A matrix; an expert's weights are
    streamed once per activation tile. sorted_expert_ids must be built with
    padding = block_n.
    """
    from .._arch import sm_major
    if sm_major() != 9:
        raise NotImplementedError("sm_90 (H200) grouped path only")
    return torch.ops.fish_scales_ops.linear_fp8_grouped_contiguous_swapab(
        a_fp8, w_fp8, sa, sw, sorted_expert_ids, block_n, expected_m)


def linear_fp8_grouped_contiguous_swapab_pair(a_fp8, w_fp8, sa, sw, sorted_expert_ids,
                                              block_n, expected_m):
    """sm_90 grouped contiguous FC1, swap-AB, in which one CTA owns gate block b
    and up block b (256 weight rows) of one activation tile: the mainloop of the
    fused swap-AB FC1 with a bf16 epilogue. Internal: not exported from
    ``fso.compat``; kept for tests and A/B measurements.

    Same arguments and result as :func:`linear_fp8_grouped_contiguous_swapab` for
    the stacked ``[gate; up]`` FC1 weight ``w_fp8 [G, 2I, K]`` (``2I % 256 == 0``):
    gu ``[P_max, 2I]`` bf16, bit-identical to the swap-AB FC1 on the rows the GEMM
    computes. ``block_n`` is 16, 32 or 64 and must equal the ``moe_build_sorted``
    padding.
    """
    from .._arch import sm_major
    if sm_major() != 9:
        raise NotImplementedError("sm_90 (H200) grouped path only")
    return torch.ops.fish_scales_ops.linear_fp8_grouped_contiguous_swapab_pair(
        a_fp8, w_fp8, sa, sw, sorted_expert_ids, block_n, expected_m)


def linear_fp8_grouped_contiguous_swapab_swiglu(a_fp8, w_fp8, sa, sw, sorted_expert_ids,
                                                block_n, expected_m):
    """sm_90 grouped contiguous FC1, swap-AB, with the SwiGLU + 1x128 FP8
    requantize fused into its epilogue. Internal: the swap-AB FC1 of
    :func:`moe_layer_fp8_sm90`, not exported from ``fso.compat``.

    Same arguments as :func:`linear_fp8_grouped_contiguous_swapab` for the
    stacked ``[gate; up]`` FC1 weight ``w_fp8 [G, 2I, K]`` (``2I % 256 == 0``,
    ``K % 128 == 0``, ``block_n`` 16, 32 or 64 and equal to the
    ``moe_build_sorted`` padding). Instead of the bf16 ``[P_max, 2I]`` FC1 output
    it returns what :func:`silu_chunk_mul_quantize_1x128_sorted_sm90` makes of
    that output, in the same layouts: ``(dq [P_max, I] fp8,
    sd [I/128, align4(P_max)] fp32)``, bit-identical on every routed row (the rows
    ``flat_to_sorted`` assigns). Padding rows inside a visited ``block_n``-row tile
    hold the SwiGLU of the gather's uninitialised padding; FC2 is row-independent
    and the combine reads only routed rows.
    """
    from .._arch import sm_major
    if sm_major() != 9:
        raise NotImplementedError("sm_90 (H200) grouped path only")
    return torch.ops.fish_scales_ops.linear_fp8_grouped_contiguous_swapab_swiglu(
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
        q, s = torch.ops.fish_scales_ops.quantize_128x128(w[g].contiguous(), False)  # sm_90: FP32 scales
        w_fp8[g].copy_(q)
        sw[g].copy_(s)
    return w_fp8, sw
