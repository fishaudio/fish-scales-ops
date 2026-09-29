"""1×32 MXFP8 (true OCP MXFP8) ops — quantize + GEMM. sm_100/sm_103/sm_120.

Arch routing (inside ``linear_mxfp8_raw`` / ``quantize_1x32_packed``):

* sm_120/121 (consumer Blackwell): CUTLASS ``Sm120MxFP8BlockScaledKernel``
  (kSFVecSize=32) with the (M, tiles_n) cascade and Stream-K shared with
  the 1×128 FP8 path. Scales: int32 K-major ``[pad(M,4), K/128]``.
* sm_100/103 (B200 / B300 datacenter Blackwell): CUTLASS tcgen05
  BlockScaled collective (``arch::Sm100`` builder, compiled as sm_100f
  family target). Scales: opaque 1-D int32 in the CUTLASS
  ``Sm1xxBlockScaledConfig`` atom layout, ``[ceil(M/128)*128 * K/128]``.

The scale tensor is an *opaque handle*: always produce it with
:func:`quantize_1x32_fp8` / :func:`silu_chunk_mul_quantize_1x32_fp8` on the
same device the GEMM will run on — layouts are NOT interchangeable between
sm_120 and sm_100/103.

On sm_90 (Hopper) both ops raise ``NotImplementedError``. Use
``linear_fp8`` (deep_gemm WGMMA BlockScaled) on Hopper.
"""
from __future__ import annotations

import os

import torch

from .._arch import sm_major


def _require_mxfp8_arch() -> None:
    if sm_major() not in (10, 12):
        raise NotImplementedError(
            "linear_mxfp8 / quantize_1x32_fp8 need sm_100/sm_103 (B200/B300) "
            "or sm_120/121 (RTX 5090 / RTX PRO 6000). On sm_90 (Hopper) use "
            "linear_fp8 + quantize_1x128_fp8 / quantize_128x128_fp8."
        )


# Backwards-compat alias (callers/tests may import the old name).
_require_sm120 = _require_mxfp8_arch


def linear_mxfp8(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    sx: torch.Tensor,
    sw: torch.Tensor,
) -> torch.Tensor:
    """Block-scaled MXFP8 (1×32 UE8M0) GEMM. sm_100/sm_103/sm_120.

    True OCP MXFP8: one E8M0 byte per 32 K-elements on both A and B.
    Finer than the 1×128 / 128×128 scheme — meaningfully tighter
    quantization at the cost of 4× more scale bytes.

    Args:
        x_fp8: float8_e4m3fn [M, K], K%128==0, N%128==0.
        w_fp8: float8_e4m3fn [N, K].
        sx, sw: opaque int32 UE8M0 scales from :func:`quantize_1x32_fp8`
            run on the SAME device arch (sm_120: K-major
            ``[pad(M,4), K/128]``; sm_100/103: 1-D Sm1xx atom layout).

    Returns:
        bfloat16 [M, N].

    Raises:
        NotImplementedError: on sm_90 (use ``linear_fp8`` there).
    """
    _require_mxfp8_arch()
    # sm_100/sm_103 (Blackwell datacenter) has a best-per-shape dispatch: an
    # M <= 64 decode row of vendored CuTe-DSL kernels in front of three tiers
    # — cuBLAS scaled_mm on the rest of the small-M band / DSL on
    # prefill+peak / C++ cascade for the narrow-N split-K and square cubic
    # paths. Full rationale in `_sm100_dispatch.py`.
    if sm_major() == 10:
        from . import _sm100_dispatch
        y = _sm100_dispatch.route(x_fp8, w_fp8, sx, sw)
        if y is not None:
            return y
    # sx / sw come out of quantize_1x32_fp8 in K-major byte order (PyTorch
    # strides (1, M_pad), which it labels non-contiguous). The CUTLASS kernel
    # reads `data_ptr()` raw bytes in K-major layout, so we must NOT call
    # `.contiguous()` on a K-major view — it would physically repack to
    # row-major and the kernel would read garbage (cos drops to ~0.5–0.87
    # across the Qwen3 grid; confirmed 2026-06-12 on RTX 5090). The ternary
    # preserves the K-major byte order from the quantizer while still
    # ensuring contiguity if the caller hands us a standard row-major scale.
    return torch.ops.fish_scales_ops.linear_mxfp8_raw(
        x_fp8.contiguous(),
        w_fp8.contiguous(),
        sx.contiguous() if sx.is_contiguous() else sx,
        sw.contiguous() if sw.is_contiguous() else sw,
    )


def quantize_1x32_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [..., K] → FP8 (E4M3) [..., K] + UE8M0 1×32 scales.

    True OCP MXFP8: one E8M0 byte per 32 elements along K. Supported on
    sm_100/sm_103 and sm_120/121.

    Returns:
        (x_fp8, sx) — x_fp8 has the input's shape; sx is an opaque int32
        scale tensor in the arch-native layout (sm_120: K-major
        ``[pad(M,4), K/128]``, 4 UE8M0 bytes per int32 word; sm_100/103:
        1-D ``[ceil(M/128)*128 * K/128]`` Sm1xxBlockScaledConfig atom
        layout). Ready for :func:`linear_mxfp8` on the same device arch.

    Constraints: K must be a multiple of 128.

    Raises:
        NotImplementedError: on sm_90 (this path needs MXFP8 hardware).
    """
    _require_mxfp8_arch()
    # Fused quantize + packed-scale write — single CUDA kernel, no FP32 scale
    # round-trip. Bit-exact with the legacy `quantize_1x32` + `repack_mxfp8_scales`
    # two-step path.
    return torch.ops.fish_scales_ops.quantize_1x32_packed(x.contiguous(), True)


def silu_chunk_mul_quantize_1x32_fp8(gu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SwiGLU prologue + MXFP8 quantize. sm_100/sm_103/sm_120.

    Takes ``gu`` (bf16 ``[..., 2*INTER]``) where the first INTER cols along
    the last dim are ``gate`` and the second INTER cols are ``up``. Computes
    ``h = silu(gate) * up`` and quantizes ``h`` to FP8 + packed UE8M0 scale
    **without materialising ``h`` in global memory** — saves a M·INTER·2
    byte intermediate read+write vs the unfused chain
    (silu·chunk + ``quantize_1x32_fp8``).

    Returns:
        (x_fp8, sx_packed) — same packed layout as :func:`quantize_1x32_fp8`,
        ready for :func:`linear_mxfp8` as the activation operand.

    Constraints: ``INTER % 128 == 0``.

    Raises:
        NotImplementedError: on sm_90.
    """
    _require_mxfp8_arch()
    return torch.ops.fish_scales_ops.silu_chunk_mul_quantize_1x32(gu.contiguous(), True)


# ---------------------------------------------------------------------------
# Grouped (MoE) masked-layout surface — sm_120/121 (M1) and sm_100/103 (M3).
#
# Masked semantics (DeepGEMM-style): G expert groups, each with a fixed row
# capacity ``m_cap`` and a per-group valid-row count ``masked_m[g]`` that
# lives ON DEVICE and is read only by the kernels — never on the host — so
# every op below is CUDA-Graph capture-safe with dynamic routing: replays
# honour whatever counts the masked_m buffer holds at replay time.
# Rows at or beyond masked_m[g] hold undefined bytes in every tensor.
#
# The activation/weight scale tensors are opaque handles whose per-group byte
# layout differs by architecture (sm_120 packs int32 words K-major per group;
# sm_100/103 writes one CUTLASS Sm1xx atom slab per group). Produce them with
# the grouped quantize ops on the same device the GEMM will run on.


def _require_grouped_arch() -> None:
    if sm_major() not in (10, 12):
        raise NotImplementedError(
            "the grouped MXFP8 (MoE) path needs sm_100/sm_103 (B200/B300) or "
            "sm_120/121 (RTX 5090 / RTX PRO 6000). On sm_90 (Hopper) use the "
            "block-FP8 grouped surface (linear_fp8_grouped_masked / "
            "moe_layer_fp8_sm90) — Hopper has no MXFP8 hardware."
        )


# Backwards-compat alias (the M1-era name).
_require_sm120_grouped = _require_grouped_arch


def linear_mxfp8_grouped_masked(
    a_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    sa: torch.Tensor,
    sw: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    max_active_groups: int = 0,
    slot_to_expert: torch.Tensor | None = None,
    problem_shapes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Grouped block-scaled MXFP8 GEMM over per-expert weights (masked).

    Args:
        a_fp8: float8_e4m3fn ``[G, m_cap, K]`` — per-group activation slab
            from :func:`quantize_1x32_grouped_gather_fp8` (or
            :func:`silu_chunk_mul_quantize_1x32_grouped_fp8`).
        w_fp8: float8_e4m3fn ``[G, N, K]`` — per-expert weights (see
            :func:`quantize_moe_weights_1x32_fp8`).
        sa: opaque int32 per-group UE8M0 scales from the grouped quantize
            ops on this arch (sm_120: ``[G, K/128, m_cap]`` K-major;
            sm_100/103: ``[G, pad(m_cap,128) * K/128]`` Sm1xx atom slabs).
        sw: opaque int32 per-group UE8M0 weight scales (sm_120:
            ``[G, K/128, N]``; sm_100/103: ``[G, N * K/128]``).
        masked_m: int32 ``[G]`` on device — valid rows per group. Caller
            contract: ``masked_m[g] <= m_cap`` for every g (the kernel does
            not check; an oversized count silently drops that group's
            overflow rows at the TMA bounds).
        expected_m: host-side static hint (``ceil(total_rows / G)``) used
            only for tile selection; per-call value must be a plain int so
            graph capture stays shape-static.
        max_active_groups: optional host-side static upper bound on how many
            groups can hold at least one row, i.e. ``min(M * topk, G)``. It is
            not recoverable from ``expected_m`` (which is ``ceil(rows / G)``
            and equals 1 across the whole decode band), and on sm_100/103 it
            is what lets the dispatcher size the slot-bound decode route's
            slot list and grid. ``0`` means "not supplied" and keeps that
            route off. sm_120/121 accept and ignore it.
        slot_to_expert: optional int32 ``[G]`` on device — the packed list of
            experts that hold at least one routed row (ascending ids in the low
            entries, ``-1`` after them), i.e. the fourth output of
            :func:`moe_build_routing` called with ``with_slots=True``. Only the
            sm_100/103 slot-bound decode route reads it, and passing it saves
            that route the one-block launch it otherwise makes to build the
            same list. Omitting it changes nothing but that launch.
        problem_shapes: optional int32 ``[G, 3]`` on device — the per-group
            ``(rows, N, K)`` triples of THIS GEMM, i.e. the entry of
            :func:`moe_build_routing`'s ``problem_shapes_for`` output that was
            requested for ``(N, K)``. Only the sm_100/103 pointer-array route
            reads it; with it that route launches the GEMM alone, without the
            per-call argument-preparation kernel, because the triple was the
            only routing-dependent argument that kernel still computed (the
            tensor-set arrays are bound once per tensor set). Omitting it keeps
            the preparation kernel. The slot route and sm_120/121 validate the
            shape and ignore it. Ask :func:`mxfp8_grouped_problem_shapes_consumed`
            whether a call would read it.

    Returns:
        bfloat16 ``[G, m_cap, N]``; rows ``>= masked_m[g]`` are undefined.

    Constraints: ``K % 128 == 0``, ``N % 128 == 0``, ``m_cap % 4 == 0``,
    ``0 <= max_active_groups <= G``, ``slot_to_expert`` (when given) int32
    ``[G]`` and ``problem_shapes`` (when given) int32 ``[G, 3]``, both
    contiguous and on ``masked_m``'s device.
    """
    _require_grouped_arch()
    return torch.ops.fish_scales_ops.linear_mxfp8_grouped_masked(
        a_fp8, w_fp8, sa, sw, masked_m, expected_m, max_active_groups, slot_to_expert, problem_shapes
    )


def linear_mxfp8_grouped_masked_swiglu(
    a_fp8: torch.Tensor,
    w13_fp8: torch.Tensor,
    sa: torch.Tensor,
    sw13: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    max_active_groups: int = 0,
    slot_to_expert: torch.Tensor | None = None,
    problem_shapes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped MoE FC1 with SwiGLU and the MXFP8 requantize fused into it.

    Replaces the pair :func:`linear_mxfp8_grouped_masked` (on the gate_up
    weights) + :func:`silu_chunk_mul_quantize_1x32_grouped_fp8`: the GEMM's
    epilogue computes ``silu(gate) * up`` in FP32 and writes the FP8 bytes and
    the 1×32 UE8M0 scales directly, so the bf16 ``[G, m_cap, 2*I]``
    intermediate is never written and the second kernel disappears.

    **The weights must be gate/up interleaved and this op cannot tell if they
    are not.** Row ``2j`` of ``w13_fp8`` must be ``gate_j`` and row ``2j+1``
    must be ``up_j``, not fso's usual ``[gate; up]`` stacking; produce them
    with ``quantize_moe_weights_1x32_fp8(w13, w13_interleave=True)`` or
    :func:`interleave_w13_fp8`. Given the stacked order the op returns a
    silently wrong result and raises nothing, because the two orders are the
    same bytes in a different sequence.

    Args:
        a_fp8: float8_e4m3fn ``[G, m_cap, K]`` activation slab.
        w13_fp8: float8_e4m3fn ``[G, 2*I, K]``, rows INTERLEAVED, ``I`` a
            multiple of 128.
        sa, sw13: the opaque int32 Sm1xx atom slabs of the two operands.
        masked_m: int32 ``[G]`` on device — valid rows per group.
        expected_m: host-side tile-selection hint (``ceil(total_rows / G)``).
        max_active_groups: the host-static bound ``min(M * topk, G)`` on how
            many experts can hold a row, exactly as for
            :func:`linear_mxfp8_grouped_masked`. It selects the route: on the
            decode band the call lands on the swap-orientation slot kernel,
            whose fused variant needs this bound to size its grid.
        slot_to_expert: the packed active-expert list,
            ``moe_build_routing(..., with_slots=True)``'s fourth output. Passing
            it lets a decode-band call launch nothing but the GEMM; leaving it
            ``None`` costs a one-block builder launch.
        problem_shapes: the per-group ``(rows, N, K)`` triples for this GEMM's
            ``N = 2 * I`` weight rows, from :func:`moe_build_routing`'s
            ``problem_shapes_for=[(2 * I, K), ...]`` output, exactly as for
            :func:`linear_mxfp8_grouped_masked`. Passing it lets a
            pointer-array-route call launch nothing but the GEMM; leaving it
            ``None`` costs the per-call argument-preparation launch.

    Returns:
        ``(h_fp8 [G, m_cap, I], sh)`` — exactly the pair
        :func:`silu_chunk_mul_quantize_1x32_grouped_fp8` returns, ready to be
        handed to :func:`linear_mxfp8_grouped_masked` as the FC2 activation
        operand. Rows ``>= masked_m[g]`` are undefined.

    Raises:
        NotImplementedError: on sm_90, which has neither MXFP8 hardware nor a
            fused FC1. On sm_100/sm_103 the epilogues are clones of CUTLASS's
            NoSmem epilogues; on sm_120/121 the grouped kernel carries its own
            (arch/sm120/mxfp8/fused_swiglu_epi.cuh). The two architectures'
            scale slabs differ, as everywhere else on this surface: sm_100/103
            returns the Sm1xx atom slab, sm_120/121 the K-major words.
    """
    if sm_major() not in (10, 12):
        raise NotImplementedError(
            f"linear_mxfp8_grouped_masked_swiglu needs sm_100/sm_103 (B200 / B300) or sm_120/121; this device "
            f"is sm_{sm_major()}0, which has no MXFP8 hardware. On sm_90 use the block-FP8 grouped surface."
        )
    return torch.ops.fish_scales_ops.linear_mxfp8_grouped_masked_swiglu(
        a_fp8, w13_fp8, sa, sw13, masked_m, expected_m, max_active_groups, slot_to_expert, problem_shapes
    )


def mxfp8_grouped_swiglu_fused_route(
    m_cap: int,
    n_w: int,
    k: int,
    num_groups: int,
    max_active_groups: int = 0,
) -> bool:
    """Should this FC1 shape use the fused op, or the old two-kernel pair?

    The answer is host-static and has to be taken before the layer is composed,
    because the two forms need different weights (interleaved vs ``[gate; up]``)
    and different follow-on kernels. Both sm_100 grouped routes now carry a
    fused FC1 — the pointer-array cascade and the swap-orientation slot kernel
    — so with the knob unset this answers ``True`` on both sides of the route
    boundary, and what the same ``slot_route`` verdict really decides is which
    of the two fused kernels the call lands on. It answers ``False`` only where
    no fused instantiation covers the shape, and on every non-sm_100 device.

    ``FSO_FC1_FUSED`` overrides it: ``0`` always ``False``, unset applies the
    rule, ``1`` always ``True`` where the op is legal. The variable is read
    once per process inside the extension.

    Args:
        m_cap: the per-group row capacity the layer will allocate.
        n_w: the FC1 weight-row count, ``2 * I``.
        k: the hidden size.
        num_groups: G.
        max_active_groups: ``min(M * topk, G)``.
    """
    return bool(
        torch.ops.fish_scales_ops.mxfp8_grouped_swiglu_fused_route(
            m_cap, n_w, k, num_groups, max_active_groups
        )
    )


def mxfp8_grouped_swiglu_available(n_w: int, k: int) -> bool:
    """Can this FC1 shape use the fused op at all, on any M?

    The load-time half of the routing decision. A model holds ONE weight
    layout, and the fused op needs the interleaved one while the unfused
    fallback then needs ``pairwise=True`` on the SwiGLU kernel to match, so
    the layout has to be chosen before the weights are quantized — whereas
    :func:`mxfp8_grouped_swiglu_fused_route` is asked per call. Both read the
    same ``FSO_FC1_FUSED`` setting out of the same process-wide static, so
    "the feature is off" and "this call does not take it" cannot disagree.

    Returns ``False`` on every architecture but sm_100/sm_103, and on any
    shape whose output width ``n_w / 2`` is not a multiple of 128.
    """
    return bool(torch.ops.fish_scales_ops.mxfp8_grouped_swiglu_available(n_w, k))


def mxfp8_grouped_slot_possible(
    m_cap: int,
    n_w: int,
    k: int,
    num_groups: int,
    max_active_groups: int = 0,
    fused_swiglu: bool = False,
) -> bool:
    """Would a grouped GEMM of this shape take the slot-bound decode route?

    The slot route is the only kernel that reads the packed active-expert list
    :func:`moe_build_routing` emits under ``with_slots=True``, so this is the
    question a caller has to answer before it decides whether to ask for that
    list. The list is not free — the routing kernel pays a block-wide
    compaction of the per-expert histogram to build it — and everywhere the
    slot route is not taken the result is a tensor no kernel ever looks at.
    A layer should therefore pass ``with_slots=`` the disjunction of this query
    over its two GEMMs (FC1 ``n_w = 2 * INTER``, ``k = HIDDEN``; FC2
    ``n_w = HIDDEN``, ``k = INTER``), which is what the benches and the layer
    test do.

    The verdict is the dispatcher's own ``slot_route``, so ``FSO_GROUPED_SLOT``
    steers this answer exactly as it steers the route, and the two cannot drift
    apart. Passing the list where this answers ``False`` is never wrong, only
    wasteful; withholding it where it answers ``True`` is also correct and
    costs that route one extra one-block launch per call.

    Returns ``False`` on every architecture but sm_100/sm_103, which have no
    slot route at all.

    Args:
        m_cap: the per-group row capacity the layer will allocate.
        n_w: the GEMM's weight-row count N.
        k: the GEMM's reduction extent K.
        num_groups: G.
        max_active_groups: ``min(M * topk, G)``; ``0`` (not supplied) keeps the
            slot route off and so answers ``False``.
        fused_swiglu: ``True`` when the call will be the fused-SwiGLU FC1
            (:func:`linear_mxfp8_grouped_masked_swiglu`), ``False`` for the
            plain grouped GEMM. The two land on different slot kernels with
            different row-capacity clauses — one 32-column epilogue chunk for
            the fused kernel, the full 64-wide token tile for the plain one
            (run b300_round3_20260922/M-A3) — so a layer asks for its FC1 with
            the flag set and for its FC2 with it clear.
    """
    return bool(
        torch.ops.fish_scales_ops.mxfp8_grouped_slot_possible(
            m_cap, n_w, k, num_groups, max_active_groups, fused_swiglu
        )
    )


def mxfp8_grouped_problem_shapes_consumed(
    m_cap: int,
    n_w: int,
    k: int,
    num_groups: int,
    max_active_groups: int = 0,
    fused_swiglu: bool = False,
) -> bool:
    """Would a grouped GEMM of this shape read a ``problem_shapes`` tensor?

    The pointer-array route's counterpart of :func:`mxfp8_grouped_slot_possible`.
    On sm_100/103 the per-group ``(rows, N, K)`` triples
    :func:`moe_build_routing` emits under ``problem_shapes_for`` are read by
    the pointer-array cascade and by nothing else: the slot route sizes its
    grid from ``masked_m`` and the slot list. Asking the routing kernel for the
    triples where no kernel reads them costs it a few stores per group for
    nothing, so a layer asks this per GEMM (FC1 ``n_w = 2 * INTER``,
    ``k = HIDDEN``; FC2 ``n_w = HIDDEN``, ``k = INTER``) and requests the
    shapes for the GEMMs that consume them, which is what the benches and the
    layer test do. The verdict is the dispatcher's own ``slot_route``, so
    ``FSO_GROUPED_SLOT`` steers it exactly as it steers the route.

    ``fused_swiglu`` names the kernel the call will land on, exactly as for
    :func:`mxfp8_grouped_slot_possible`: the fused-SwiGLU FC1
    (:func:`linear_mxfp8_grouped_masked_swiglu`) leaves the slot route one
    step earlier than the plain grouped GEMM (one 32-column epilogue chunk
    against the full 64-wide token tile), so the two queries are complements
    of each other only when both are asked with the same flag. A layer asks
    both for its FC1 with the flag set to whether it will call the fused op,
    and for its FC2 with it clear; then, per GEMM, exactly one of the slot
    list and the problem shapes is requested and the GEMM launches nothing
    but itself.

    Returns ``False`` on every architecture but sm_100/sm_103; sm_120/121
    accepts the tensor only so a caller can pass the same arguments everywhere.
    """
    return bool(
        torch.ops.fish_scales_ops.mxfp8_grouped_problem_shapes_consumed(
            m_cap, n_w, k, num_groups, max_active_groups, fused_swiglu
        )
    )


def _w13_interleave_perm(two_inter: int, device: torch.device) -> torch.Tensor:
    """Row permutation that turns ``[gate; up]`` into ``[g0, u0, g1, u1, …]``.

    ``perm[n_out]`` is the source row of output row ``n_out``: output row ``2j``
    takes source row ``j`` (gate) and output row ``2j+1`` takes source row
    ``I + j`` (up).
    """
    inter = two_inter // 2
    perm = torch.empty(two_inter, dtype=torch.long, device=device)
    idx = torch.arange(inter, dtype=torch.long, device=device)
    perm[0::2] = idx
    perm[1::2] = inter + idx
    return perm


def _sm100_atom_word_index(rows: torch.Tensor, num_kp: int) -> torch.Tensor:
    """Word index of every ``(row, kp)`` in one group's Sm1xx atom scale slab.

    The slab describes a ``(pad(rows,128), K)`` tensor as one 512-byte
    scale-factor block per ``128 × 128`` tile, with a row's four consecutive
    K-block bytes in one int32 word — the layout ``sf_word_index_grouped_atom``
    in ``csrc/gemm/ops/quant_kernels.cu`` writes and the GEMM's scale-factor
    TMA descriptor reads. Returns ``[len(rows), num_kp]``.
    """
    r = rows % 128
    base = (rows // 128) * (num_kp * 128) + (r % 32) * 4 + (r // 32)
    kp = torch.arange(num_kp, dtype=torch.long, device=rows.device)
    return base[:, None] + kp[None, :] * 128


def interleave_w13_fp8(
    w13_fp8: torch.Tensor,
    sw13: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-order already-quantized FC1 weights into the fused op's row order.

    For callers whose checkpoint is already MXFP8 in the ``[gate; up]``
    stacking: this returns the same bytes in the ``[g0, u0, g1, u1, …]`` order
    :func:`linear_mxfp8_grouped_masked_swiglu` requires. It is a pure
    permutation and not a requantization — the 1×32 quantizer is per row along
    K, so a row's FP8 bytes and its scale word depend only on that row, and
    moving rows around cannot change either. That is why quantizing the
    interleaved bf16 weights and interleaving the quantized weights give
    bit-identical results (``tests/gemm/unit/test_mxfp8_grouped.py`` checks it
    on both families).

    Args:
        w13_fp8: float8_e4m3fn ``[G, 2*I, K]`` in ``[gate; up]`` order.
        sw13: the matching int32 scale slabs ``[G, 2*I * K/128]``.

    Returns:
        ``(w13_fp8_interleaved, sw13_interleaved)``, freshly allocated.

    Raises:
        NotImplementedError: on any architecture but sm_100/sm_103 and
            sm_120/121, the ones whose fused FC1 consumes the layout.
    """
    if sm_major() not in (10, 12):
        raise NotImplementedError(
            f"interleave_w13_fp8 targets the fused FC1 weight layout of sm_100/sm_103 and sm_120/121; "
            f"this device is sm_{sm_major()}0, which has no fused FC1 and no use for the interleaved order."
        )
    if w13_fp8.dim() != 3:
        raise ValueError("w13_fp8 must be [G, 2*I, K]")
    g, two_inter, k = w13_fp8.shape
    if two_inter % 2 != 0 or (two_inter // 2) % 128 != 0:
        raise ValueError("w13_fp8.size(1) must be 2*I with I a multiple of 128")
    if k % 128 != 0:
        raise ValueError("K must be a multiple of 128")
    num_kp = k // 128
    if sw13.numel() != g * two_inter * num_kp:
        raise ValueError("sw13 must hold G * 2*I * K/128 int32 words")

    perm = _w13_interleave_perm(two_inter, w13_fp8.device)
    w_out = w13_fp8.index_select(1, perm).contiguous()

    if sm_major() == 12:
        # sm_120/121: the scale handle is [G, K/128, N] K-major words, one
        # word per (K-block, row), so a row permutation is a permutation of
        # the last dim and no word is split.
        s_out = (
            sw13.reshape(g, num_kp, two_inter)
            .index_select(2, perm.to(sw13.device))
            .contiguous()
        )
        return w_out, s_out.reshape(sw13.shape)

    dst = _sm100_atom_word_index(
        torch.arange(two_inter, dtype=torch.long, device=sw13.device), num_kp
    ).reshape(-1)
    src = _sm100_atom_word_index(perm.to(sw13.device), num_kp).reshape(-1)
    s_flat = sw13.reshape(g, -1)
    s_out = torch.empty_like(s_flat)
    s_out[:, dst] = s_flat[:, src]
    return w_out, s_out.reshape(sw13.shape)


def quantize_1x32_grouped_gather_fp8(
    x: torch.Tensor,
    slot_of_flat: torch.Tensor,
    topk: int,
    num_groups: int,
    m_cap: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused token-gather + MXFP8 quantize into the masked grouped layout.

    Indexed over the flat routed-pair space (host-static ``M * topk``): pair
    ``i`` reads source row ``i // topk`` of ``x`` (bf16 ``[M, K]``) and
    writes destination slot ``slot_of_flat[i]`` (= ``g * m_cap + m_in``,
    from :func:`moe_build_routing`). A pair whose entry is ``-1`` was skipped
    (padded graph row, or an expert another rank owns) and writes nothing. One
    launch, no dead warps — the original padded ``G * m_cap`` iteration burned
    ~27 µs of early-outs at M=512.

    Returns:
        (a_fp8 ``[G, m_cap, K]``, sa int32 opaque per-group scales — sm_120
        ``[G, K/128, m_cap]``, sm_100/103 ``[G, pad(m_cap,128) * K/128]``)
        ready for :func:`linear_mxfp8_grouped_masked`. Rows no pair maps to
        are undefined (the GEMM's masked contract ignores them).
    """
    _require_grouped_arch()
    return torch.ops.fish_scales_ops.quantize_1x32_grouped_gather(
        x.contiguous(), slot_of_flat, topk, num_groups, m_cap, True
    )


def silu_chunk_mul_quantize_1x32_grouped_fp8(
    gu: torch.Tensor,
    slot_of_flat: torch.Tensor,
    pairwise: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped SwiGLU prologue + MXFP8 quantize over the flat pair space.

    ``gu`` is the grouped gate_up output (bf16 ``[G, m_cap, 2*INTER]``). Each
    routed pair processes its own row (``slot_of_flat[i]``) in place; rows no
    pair maps to stay undefined, and a pair whose entry is ``-1`` was skipped
    and has no row to process.

    Args:
        gu: the grouped gate_up output.
        slot_of_flat: int32 ``[M * topk]`` from :func:`moe_build_routing`.
        pairwise: where gate and up sit inside a row. ``False`` (default) is
            the chunked order ``[gate_0..gate_{I-1}, up_0..up_{I-1}]``.
            ``True`` is the interleaved order ``[gate_0, up_0, gate_1, up_1,
            …]``, which is what the GEMM produces when its FC1 weights carry
            the fused op's interleaved row layout — a layer that holds
            interleaved weights but falls back to the unfused FC1 on the
            decode band needs this form. Passing the wrong value multiplies
            the wrong pairs together and raises nothing. sm_100/sm_103 only.

    Returns:
        (h_fp8 ``[G, m_cap, INTER]``, sh int32 opaque per-group scales — the
        same arch-native layout :func:`quantize_1x32_grouped_gather_fp8`
        produces, with INTER in place of K).

    Raises:
        NotImplementedError: with ``pairwise=True`` on anything but
            sm_100/sm_103. The interleaved row order exists only to feed the
            sm_100 fused FC1, so no other architecture ever produces a ``gu``
            in that order.
    """
    _require_grouped_arch()
    if pairwise and sm_major() not in (10, 12):
        raise NotImplementedError(
            f"silu_chunk_mul_quantize_1x32_grouped_fp8(pairwise=True) targets the interleaved gate_up "
            f"layout of the architectures that have a fused FC1 (sm_100/sm_103 and sm_120/121); this "
            f"device is sm_{sm_major()}0, whose grouped FC1 always produces the chunked [gate; up] "
            f"order. Call it with pairwise=False there."
        )
    return torch.ops.fish_scales_ops.silu_chunk_mul_quantize_1x32_grouped(
        gu, slot_of_flat, True, pairwise
    )


# The threshold at which the down-projection slab stops being incidental and starts
# dominating the layer's transient footprint. Expressed in bytes (it happens to be
# the card's L2 size) so it follows HIDDEN and topk rather than being an M number.
_FUSED_COMBINE_SLAB_BYTES = 96 * 1024 * 1024


def moe_layer_fused_combine_engages_sm120(m: int, topk: int, hidden: int) -> bool:
    """Would the fused-combine FC2 be taken for a bucket of ``m`` tokens?

    The route replaces a bf16 ``[G, m_cap, HIDDEN]`` slab and a combine launch with
    atomic adds straight into the output, and since 2026-09-28 it wins on both axes
    (run sm120_scatterwarp_20260928): the layer's transient footprint falls by about
    half — Qwen3.5-35B-A3B at M = 8192 goes from 14.0 GiB to 6.0 GiB, Qwen3-30B-A3B
    from 7.5 to 3.5 — and the layer is **2-7 % faster** than the slab pair at the
    token counts where this engages. For comparison, chunking the token dimension to
    bound the same peak costs +80 to +209 %.

    It used to cost 0.2-7.0 % of time, because the store it replaces is asynchronous
    (a TMA-store warp drains shared memory while the mainloop runs on) and the
    scatter was issued from the math warps, which is not. Moving it to the store warp
    and the spare fourth TMA warp restored that overlap; ``FSO_MOE_SCATTER_WARP=0``
    goes back to the math-warp form for an A/B.

    So this engages where the slab dominates the footprint, which is also where a
    32 GB card would otherwise refuse the bucket. Below that it does not engage at
    all, which is why a caller may pass ``fused_combine=True`` for every bucket and
    still get bit-identical results on the small ones. It stays opt-in even though it
    is now faster: the accumulation order is whatever order the CTAs finish in, so
    the result is not bit-reproducible run to run.
    """
    return m * topk * hidden * 2 >= _FUSED_COMBINE_SLAB_BYTES


def linear_mxfp8_grouped_masked_combine(
    a_fp8: torch.Tensor,
    w2_fp8: torch.Tensor,
    sa: torch.Tensor,
    sw2: torch.Tensor,
    masked_m: torch.Tensor,
    row_map: torch.Tensor,
    weight_of_slot: torch.Tensor,
    out: torch.Tensor,
    expected_m: int,
    max_active_groups: int = 0,
) -> torch.Tensor:
    """The grouped FC2 with the weighted combine folded into its epilogue
    (sm_120/121): each row is scaled by its combine weight and added into its
    token's row of ``out``, so the ``[G, m_cap, HIDDEN]`` slab and the combine
    launch both disappear.

    ``out`` is **accumulated into** — fill it first with whatever the layer adds to
    (zero, or a shared expert's gated output). ``row_map`` and ``weight_of_slot``
    come from :func:`moe_build_routing` (the latter by passing ``topk_w``).

    The adds are atomic, so the accumulation order follows however the CTAs
    interleave and the last bits of each element are not reproducible run to run
    (measured spread: about two bf16 ULP). That is why this is a separate op and why
    the layer takes it only when asked. It trades time for footprint — see
    :func:`moe_layer_fused_combine_engages_sm120` for the numbers and the rule.
    """
    _require_grouped_arch()
    if sm_major() != 12:
        raise NotImplementedError(
            f"linear_mxfp8_grouped_masked_combine is the sm_120/121 fused-combine FC2; this device is "
            f"sm_{sm_major()}0."
        )
    return torch.ops.fish_scales_ops.linear_mxfp8_grouped_masked_combine(
        a_fp8, w2_fp8, sa, sw2, masked_m, row_map, weight_of_slot, out, int(expected_m), int(max_active_groups)
    )



def moe_build_routing(
    topk_ids: torch.Tensor,
    num_groups: int,
    m_cap: int,
    *,
    with_slots: bool = False,
    problem_shapes_for: list[tuple[int, int]] | None = None,
    topk_w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """topk routing -> masked-layout index tensors, one kernel launch.

    Args:
        topk_ids: int32 ``[M, topk]``, no-replacement expert ids. An id outside
            ``[0, num_groups)`` is skipped: it is counted into no group, it
            occupies no slot, and its ``slot_of_flat`` entry is ``-1``, which
            every consumer of that array (the gather-quantize, the SwiGLU
            requantize and the combine) reads as "no routed row". That is what a
            serving engine needs for the padding rows of a CUDA-graph bucket,
            which carry the id ``num_experts`` or ``-1``, and for the entries an
            expert-parallel dispatcher rewrites to ``-1`` because another rank
            owns the expert.
        num_groups: G (<= 1024).
        m_cap: per-group row capacity (multiple of 4, >= M).
        with_slots: also return the packed active-expert list. It costs a
            block-wide scan over the per-expert histogram this kernel already
            holds, and it saves the sm_100/103 slot-bound grouped GEMM the
            one-block launch it otherwise makes per call to build the same
            list. Every architecture produces it; only that route reads it.
        topk_w: pass the combine weights and the kernel also publishes
            ``weight_of_slot`` fp32 ``[G * m_cap]``, each valid slot's weight,
            which is what the sm_120 fused FC2 epilogue needs to add its rows
            straight into the layer output (it knows which row it holds, not which
            routed pair produced it). It follows the other outputs in the returned
            tuple. Mutually exclusive with ``problem_shapes_for``: no route
            consumes both.
        problem_shapes_for: ``[(N, K), ...]``, at most four pairs — one per
            grouped GEMM the caller will run on this routing (a MoE layer's
            FC1 is ``(2 * INTER, HIDDEN)`` and its FC2 ``(HIDDEN, INTER)``).
            For each pair the kernel also writes the per-group CUTLASS problem
            shape, the int32 triple ``(rows, N, K)`` with ``rows`` clamped to
            ``m_cap``, in the same pass that writes ``masked_m``. That triple
            is the only routing-dependent argument the sm_100/103
            pointer-array grouped GEMM has, so handing the triples to the GEMM
            lets it launch without its per-call argument-preparation kernel.
            Every architecture produces them; only that route reads them, so
            ask :func:`mxfp8_grouped_problem_shapes_consumed` per GEMM first.

    Returns:
        (masked_m ``[G]`` int32, row_map ``[G * m_cap]`` int32 — slot ->
        source token, slots at or beyond masked_m[g] uninitialised by
        design, slot_of_flat ``[M * topk]`` int32, ``-1`` where the pair was
        skipped). Slot order within a
        group is atomic-arrival order; every consumer goes through these
        maps consistently. With ``with_slots=True`` a fourth tensor follows:
        slot_to_expert ``[G]`` int32, the ids of the experts holding at least
        one routed row in ascending order, then ``-1`` in every remaining
        entry. Pass it to :func:`linear_mxfp8_grouped_masked`. With
        ``problem_shapes_for`` a further element follows (after the slot list
        when both are requested): a list with one int32 ``[G, 3]`` tensor per
        requested pair, in the order given; pass entry ``i`` as
        ``problem_shapes=`` to the GEMM whose ``(N, K)`` is pair ``i``.
    """
    nk: list[int] = []
    if problem_shapes_for:
        for pair in problem_shapes_for:
            n, k = pair
            nk.extend((int(n), int(k)))
    # arch-agnostic glue (plain int32/bf16 + PDL, works on sm_90 and sm_120)
    masked_m, row_map, slot_of_flat, slot_to_expert, problem_shapes, weight_of_slot = (
        torch.ops.fish_scales_ops.moe_build_routing(topk_ids, num_groups, m_cap, with_slots, nk, topk_w)
    )
    out: tuple[torch.Tensor, ...] = (masked_m, row_map, slot_of_flat)
    if with_slots:
        out = out + (slot_to_expert,)
    if nk:
        # Views of one [P, G, 3] allocation: each is a contiguous [G, 3]
        # tensor, which is what the GEMM ops check for.
        out = out + ([problem_shapes[i] for i in range(problem_shapes.shape[0])],)
    if topk_w is not None:
        out = out + (weight_of_slot,)
    return out


def moe_topk_from_logits(
    logits: torch.Tensor,
    topk: int,
    *,
    renormalize: bool = True,
    with_shared_gate: bool = False,
    num_token_non_padded: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Router logits -> ``(topk_ids, topk_w, shared_gate)``, one kernel launch.

    Takes everything a serving MoE block does between the router GEMM and the
    expert layer: the softmax over experts, the top-k selection, the
    renormalisation of the selected probabilities, the padded-row sentinel of a
    CUDA-graph bucket, the expert-parallel global-to-local id remap, and
    optionally the shared expert's sigmoid gate. sglang spends four or five
    launches on that sequence; at decode each one costs about as much as a layer
    kernel.

    Args:
        logits: ``[M, num_experts]`` bf16, fp16 or fp32, with a contiguous
            expert dimension (a row-major slice is fine). With
            ``with_shared_gate`` the width is ``num_experts + 1`` and column
            ``num_experts`` is the shared expert's gate logit.
        topk: experts per token, at most 8.
        renormalize: divide the selected probabilities by their own sum, which
            is sglang's ``renormalize=True``. With ``False`` the weights are the
            plain softmax over all experts.
        num_token_non_padded: int32 ``[1]`` on device. The tokens at or past it
            are the padding rows of a graph bucket: their ids become the
            sentinel ``num_experts`` (mapped through ``expert_map`` when given)
            and their weights zero, which the grouped layer skips at no expert
            cost.
        expert_map: int32 ``[num_experts + 1]`` on device, the table an
            expert-parallel rank builds once — its own experts mapped into
            ``[0, num_local_experts)``, every remote expert and the sentinel to
            a value outside that range. Applying it here saves the gather launch
            the dispatcher would otherwise run.

    Returns:
        (topk_ids int32 ``[M, topk]``, topk_w fp32 ``[M, topk]``, shared_gate
        fp32 ``[M]`` — empty unless ``with_shared_gate``). Ids come out in
        descending weight order, ties broken toward the lower expert id.
    """
    return torch.ops.fish_scales_ops.moe_topk_from_logits(
        logits, int(topk), bool(renormalize), bool(with_shared_gate),
        num_token_non_padded, expert_map,
    )


def moe_router_topk(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int,
    *,
    renormalize: bool = True,
    with_shared_gate: bool = False,
    num_token_non_padded: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The whole router in two launches: the logit GEMM, then
    :func:`moe_topk_from_logits`.

    The GEMM stays with cuBLAS on purpose. The gate matrix is
    ``num_experts * hidden * 2`` bytes — 1 MB at 256 experts and hidden 2048 —
    and a hand-written per-token form has one CTA read all of it, which is far
    from the DRAM roofline the library GEMM reaches at every token count worth
    having; the launch this would save is not worth the memory it wastes.

    Args:
        hidden: bf16 ``[M, HIDDEN]``.
        router_weight: bf16 ``[num_experts, HIDDEN]``, or
            ``[num_experts + 1, HIDDEN]`` with ``with_shared_gate``, where the
            last row is the shared expert's gate. Concatenating that row once at
            load time is what makes the gate free here.
        The remaining arguments are :func:`moe_topk_from_logits`'s.
    """
    logits = torch.nn.functional.linear(hidden, router_weight)
    return moe_topk_from_logits(
        logits, topk, renormalize=renormalize, with_shared_gate=with_shared_gate,
        num_token_non_padded=num_token_non_padded, expert_map=expert_map,
    )


def moe_build_sorted(
    topk_ids: torch.Tensor,
    num_groups: int,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """topk routing -> contiguous (triton-style) sorted-layout tensors, one
    kernel launch. The M*topk routed pairs are sorted by expert into one
    compact list, each active expert's run padded up to ``block_m`` so the
    DeepGEMM GroupedContiguous scheduler enumerates only active padded blocks.

    Args:
        topk_ids: int32 ``[M, topk]``, no-replacement expert ids.
        num_groups: E (<= 1024).
        block_m: per-expert run padding granularity (= GEMM BLOCK_M).

    Returns:
        (sorted_expert_ids ``[P_max]`` int32 = the scheduler's grouped_layout
        (owning expert at each block-start row, -1 past the actual padded
        length), flat_to_sorted ``[M*topk]`` int32 (routed pair -> sorted row;
        drives the gather-quant, silu and combine), num_padded_dev ``[1]``
        int32 (actual padded length, the GEMM's device length gate)). ``P_max``
        is rounded up to a block_m multiple and fixed for CUDA-graph capture.
    """
    return torch.ops.fish_scales_ops.moe_build_sorted(topk_ids, num_groups, block_m)


def moe_combine(
    dn: torch.Tensor,
    slot_of_flat: torch.Tensor,
    topk_w: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    bias_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted combine of routed expert outputs, one kernel launch.

    ``out[t] = sum_j topk_w[t, j] * dn.view(-1, H)[slot_of_flat[t*topk+j]]``,
    with the entries whose slot is ``-1`` (skipped pairs — see
    :func:`moe_build_routing`) left out of the sum, so a token whose every entry
    is skipped gets an all-zero row.

    Args:
        bias: bf16 ``[M, H]`` added on top of the weighted sum — a model's
            shared-expert output. Folding it in here replaces an elementwise
            pass that reads and writes the whole block twice.
        bias_scale: fp32 ``[M]`` per-token factor applied to ``bias``, which is
            where a sigmoid shared-expert gate goes
            (:func:`moe_topk_from_logits` produces it).
        out: bf16 ``[M, H]`` destination. A data-parallel or reduce-scatter path
            already owns the buffer the result must land in; writing straight
            into it saves the copy.
    """
    # arch-agnostic glue (works on sm_90 and sm_120)
    return torch.ops.fish_scales_ops.moe_combine(dn, slot_of_flat, topk_w, bias, bias_scale, out)


def moe_combine_sorted(
    dn: torch.Tensor,
    flat_to_sorted: torch.Tensor,
    topk_w: torch.Tensor,
) -> torch.Tensor:
    """Weighted combine for the contiguous (triton-style) sorted layout.

    ``out[t] = sum_j topk_w[t, j] * dn[flat_to_sorted[t*topk+j]]`` where dn is
    the GroupedContiguous GEMM output ``[P_max, H]`` in expert-sorted rows and
    ``flat_to_sorted`` (from :func:`moe_build_sorted`) maps each routed pair to
    its sorted row.
    """
    return torch.ops.fish_scales_ops.moe_combine_sorted(dn, flat_to_sorted, topk_w)


def quantize_moe_weights_1x32_fp8(
    w: torch.Tensor,
    w13_interleave: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Offline per-expert weight quantize for the grouped GEMM.

    ``w`` is bf16 ``[G, N, K]``. Loops experts through the flat 1×32
    quantizer and stacks results — offline cost, not a serving-path op.

    Args:
        w: bf16 ``[G, N, K]`` per-expert weights.
        w13_interleave: for the FC1 (gate_up) weights only. When ``True`` the
            rows are re-ordered from ``[gate; up]`` to ``[g0, u0, g1, u1, …]``
            before quantizing, which is the layout
            :func:`linear_mxfp8_grouped_masked_swiglu` requires and cannot
            detect the absence of. Because the quantizer is per row along K,
            re-ordering before quantizing and re-ordering the quantized bytes
            afterwards (:func:`interleave_w13_fp8`) give bit-identical
            results. Leave it ``False`` for the FC2 (down) weights and for
            every caller of the unfused grouped GEMM. sm_100/sm_103 only.

    Returns:
        (w_fp8 ``[G, N, K]``, sw int32). ``sw`` is ``[G, K/128, N]``
        K-major words on sm_120/121 and ``[G, N * K/128]`` Sm1xx atom slabs
        on sm_100/103; in both cases it is exactly the per-expert output of
        ``quantize_1x32_fp8`` restacked, so no layout knowledge lives here
        beyond the reshape.

    Constraints: ``N % 128 == 0`` (so the per-expert pad(N,4) == N on sm_120
    and pad(N,128) == N on sm_100/103), ``K % 128 == 0``. With
    ``w13_interleave`` also ``(N/2) % 128 == 0``.
    """
    _require_grouped_arch()
    if w.dim() != 3:
        raise ValueError("w must be [G, N, K]")
    G, N, K = w.shape
    if N % 128 != 0 or K % 128 != 0:
        raise ValueError("N and K must be multiples of 128")
    if w13_interleave:
        if sm_major() not in (10, 12):
            raise NotImplementedError(
                f"quantize_moe_weights_1x32_fp8(w13_interleave=True) targets the fused FC1 of sm_100/sm_103 and "
                f"sm_120/121; this device is sm_{sm_major()}0, which has no fused FC1."
            )
        if (N // 2) % 128 != 0:
            raise ValueError("w13_interleave needs N = 2*I with I a multiple of 128")
        # A pure row permutation applied before the quantizer, which is per row
        # along K, so every row's FP8 bytes and its scale word are what they
        # would have been in the stacked order.
        w = w.index_select(1, _w13_interleave_perm(N, w.device))
    kp = K // 128
    w_fp8 = torch.empty(G, N, K, device=w.device, dtype=torch.float8_e4m3fn)
    if sm_major() == 10:
        # sm_100/103: the flat quantizer already emits the Sm1xx atom slab for
        # an (pad(N,128), K) tensor as a 1-D int32 buffer; N % 128 == 0 makes
        # that exactly N * K/128 words, so the per-expert slab is copied
        # verbatim.
        sw = torch.empty(G, N * kp, device=w.device, dtype=torch.int32)
        for g in range(G):
            q, s = torch.ops.fish_scales_ops.quantize_1x32_packed(w[g].contiguous(), True)
            w_fp8[g].copy_(q)
            sw[g].copy_(s)
        return w_fp8, sw
    sw = torch.empty(G, kp, N, device=w.device, dtype=torch.int32)
    for g in range(G):
        q, s = torch.ops.fish_scales_ops.quantize_1x32_packed(w[g].contiguous(), True)
        w_fp8[g].copy_(q)
        # s is [pad(N,4), K/128] with strides (1, N) — K-major. Its raw byte
        # order equals the [K/128, N] slab the grouped kernel expects.
        sw[g].copy_(s.t().view(kp, N))
    return w_fp8, sw


# Optional cap on the transient slabs of one composed sm_120 MoE layer call, in
# MiB, from FSO_MOE_SLAB_BUDGET_MB. Unset means no cap: one call per layer,
# whatever the token count. Setting it trades speed for peak memory, and
# `moe_layer_mxfp8_sm120` documents that trade.
_moe_slab_budget_bytes_cache: int = -1


def _moe_slab_budget_bytes() -> int:
    """The per-call transient-slab cap in bytes, read from the environment once;
    zero when no cap is set."""
    global _moe_slab_budget_bytes_cache
    if _moe_slab_budget_bytes_cache < 0:
        mb = 0
        raw = os.environ.get("FSO_MOE_SLAB_BUDGET_MB", "").strip()
        if raw:
            try:
                parsed = int(raw)
            except ValueError:
                parsed = 0
            if parsed > 0:
                mb = parsed
        _moe_slab_budget_bytes_cache = mb * 1024 * 1024
    return _moe_slab_budget_bytes_cache


def moe_layer_slab_bytes_per_token_sm120(
    num_experts: int, hidden: int, inter: int
) -> int:
    """Bytes of transient slab the composed sm_120 MoE layer holds per row of
    per-expert capacity, i.e. per unit of ``m_cap``.

    The masked-slab layout gives every expert its own ``m_cap``-row window in
    each intermediate tensor, so the five tensors a layer call allocates all
    scale with ``num_experts * m_cap`` and not with the number of routed rows.
    This function returns their summed per-``m_cap`` cost, which is what a caller
    sizing a graph bucket needs and what
    :func:`moe_layer_chunk_tokens_sm120` divides an optional slab cap by:

    * the FP8 activation slab ``[E, m_cap, HIDDEN]`` and its scale words,
    * the bf16 gate/up intermediate ``[E, m_cap, 2 * INTER]``,
    * the FP8 SwiGLU output ``[E, m_cap, INTER]`` and its scale words,
    * the bf16 down-projection output ``[E, m_cap, HIDDEN]``.

    Args:
        num_experts: the expert count the layer sees (under expert parallelism
            the rank-local count, not the model's).
        hidden: model hidden size.
        inter: per-expert intermediate size.
    """
    per_row = (
        hidden + 4 * (hidden // 128)     # FP8 activation row + one int32 scale word per 128 columns
        + 2 * (2 * inter)                # bf16 gate/up row
        + inter + 4 * (inter // 128)     # FP8 SwiGLU row + its scale words
        + 2 * hidden                     # bf16 down-projection row
    )
    return num_experts * per_row


def moe_layer_chunk_tokens_sm120(m: int, num_experts: int, hidden: int, inter: int) -> int:
    """Tokens per layer call that :func:`moe_layer_mxfp8_sm120` will use for a
    bucket of ``m`` tokens: ``m`` itself unless ``FSO_MOE_SLAB_BUDGET_MB`` caps
    the transient slabs, in which case the fewest chunks that fit the cap, made
    as equal as a multiple of four allows.

    Splitting the token dimension evenly over the resulting chunk count, rather
    than filling each chunk to the cap, keeps the last chunk from being a stub:
    8192 tokens under a cap of 1908 run as five calls of 1640 rather than four of
    1908 and one of 560, and a 560-row call reaches a smaller share of the GEMM's
    peak than a 1640-row one. A cap too small for even four tokens degrades to
    four-token chunks instead of raising.

    Chunking is a memory decision with a real cost, which is why it is off unless
    asked for: each call walks every expert's weights once, so n chunks read the
    expert weights n times. Measured on the RTX 5090 (Qwen3.5-35B-A3B routed
    layer, E = 256), two chunks at 1024 tokens cost +80 % and five chunks at 4096
    tokens +209 % against the single call, and the same layer's values are
    bit-identical either way.
    """
    budget = _moe_slab_budget_bytes()
    if budget <= 0:
        return m
    per_token_slot = moe_layer_slab_bytes_per_token_sm120(num_experts, hidden, inter)
    cap = int(budget // max(1, per_token_slot)) // 4 * 4
    if cap < 4:
        cap = 4
    if m <= cap:
        return m
    n_chunks = (m + cap - 1) // cap
    return min(cap, ((m + n_chunks - 1) // n_chunks + 3) // 4 * 4)


# The M1-era private name, kept because the bench and the tests referred to it.
_moe_chunk_tokens_sm120 = moe_layer_chunk_tokens_sm120


def _moe_layer_mxfp8_one(
    hidden: torch.Tensor,
    w13_fp8: torch.Tensor,
    sw13: torch.Tensor,
    w2_fp8: torch.Tensor,
    sw2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    num_experts: int,
    topk: int,
    bias: torch.Tensor | None = None,
    bias_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    join=None,
    w13_interleaved: bool = False,
    fused_combine: bool = False,
) -> torch.Tensor:
    """One masked-slab layer call over all rows of `hidden`, on sm_120/121 or on
    sm_100/103. Six kernels, five with the fused FC1, and on sm_120/121 four with the
    fused combine as well; no host-visible dependence on the routing result.

    The two architecture families run the same chain and differ only in what the
    routing kernel is asked to emit and what the two grouped GEMMs are handed.
    sm_100/103 has two grouped routes: a slot-bound decode route that reads the packed
    active-expert list and a pointer-array cascade that reads the per-group problem
    shapes. The dispatcher's own route queries say which route each GEMM will take, so
    the routing kernel is asked for exactly the tensors those routes read and each GEMM
    launches without a helper kernel; this is the composition that
    `bench/gemm/python/bench_moe_qwen3_30a3.py` (`fso_mxfp8_layer`) and
    `tests/gemm/unit/test_mxfp8_grouped.py` drive. sm_120/121 has a single grouped route
    that reads neither tensor, so there the routing kernel is asked for neither and the
    code below is the one this function ran before it served sm_100/103. The fused
    combine exists only on sm_120/121, so `fused_combine` is a permission that the
    sm_120 engagement rule acts on and that sm_100/103 never takes.

    `join` is an internal hook the block entry uses: it is called after the down
    projection and before the combine, which is the latest point at which work
    issued on another stream (the shared expert) has to be waited for, and
    therefore the point that leaves that work the whole layer to overlap with.
    """
    ops = torch.ops.fish_scales_ops
    m = int(hidden.shape[0])
    # A group receives at most one row per token (top-k draws without
    # replacement), so a capacity of M rows per expert can never overflow. The
    # routing op enforces exactly this bound.
    m_cap = (m + 3) // 4 * 4
    rows = m * topk
    # Both hints are host-static functions of (M, topk, num_experts): the tile
    # cascade reads `expected_m`, and `max_active_groups` bounds the number of
    # experts that can hold a row. A captured graph therefore bakes one value of
    # each per bucket.
    expected_m = max(1, (rows + num_experts - 1) // num_experts)
    max_active_groups = min(rows, num_experts)
    n_w = int(w13_fp8.shape[1])
    hidden_size = int(hidden.shape[1])
    # The fused FC1 needs the interleaved weight rows, which is what
    # `w13_interleaved` asserts the caller quantized; whether this call takes it is
    # the library's host-static route query. It is asked before the routing is built
    # because on sm_100/103 the fused and the plain FC1 leave the slot route at
    # different row capacities, so the routing tensors the FC1 reads depend on it.
    fused_fc1 = w13_interleaved and ops.mxfp8_grouped_swiglu_fused_route(
        m_cap, n_w, hidden_size, num_experts, max_active_groups)
    slot_to_expert = None
    ps_fc1 = None
    ps_fc2 = None
    scatter = False
    if sm_major() == 10:
        # sm_100/103: ask, per GEMM, whether it takes the slot route (which reads the
        # packed active-expert list) or the pointer-array cascade (which reads the
        # per-group (rows, N, K) triples), with the FC1 asked under the kernel it will
        # actually run. The two answers are complements per GEMM, so exactly one of
        # the two tensors is requested for each GEMM and nothing is built that no
        # kernel reads.
        inter = int(w2_fp8.shape[2])
        want_slots = (
            ops.mxfp8_grouped_slot_possible(
                m_cap, n_w, hidden_size, num_experts, max_active_groups, fused_fc1)
            or ops.mxfp8_grouped_slot_possible(
                m_cap, hidden_size, inter, num_experts, max_active_groups, False))
        want_ps_fc1 = ops.mxfp8_grouped_problem_shapes_consumed(
            m_cap, n_w, hidden_size, num_experts, max_active_groups, fused_fc1)
        want_ps_fc2 = ops.mxfp8_grouped_problem_shapes_consumed(
            m_cap, hidden_size, inter, num_experts, max_active_groups, False)
        nk = ([n_w, hidden_size] if want_ps_fc1 else []) + ([hidden_size, inter] if want_ps_fc2 else [])
        masked_m, row_map, slot_of_flat, slots, shapes, _w = ops.moe_build_routing(
            topk_ids, num_experts, m_cap, want_slots, nk)
        if want_slots:
            slot_to_expert = slots
        # `shapes` is one [P, G, 3] allocation, one [G, 3] entry per requested pair
        # in the order requested.
        if want_ps_fc1:
            ps_fc1 = shapes[0]
        if want_ps_fc2:
            ps_fc2 = shapes[1 if want_ps_fc1 else 0]
    else:
        # sm_120 has neither the slot-bound decode route nor the pointer-array
        # cascade of sm_100/103, so no kernel here reads the packed active-expert
        # list or the per-group problem shapes and the routing kernel is not asked
        # for either (mxfp8_grouped_slot_possible and
        # mxfp8_grouped_problem_shapes_consumed both answer False on this arch).
        # The fused combine needs two further routing outputs (slot -> token and
        # slot -> weight) and is taken only where the slab it removes would have
        # spilled the L2; both are host-static decisions.
        scatter = fused_combine and moe_layer_fused_combine_engages_sm120(m, topk, hidden_size)
        if scatter:
            masked_m, row_map, slot_of_flat, _slots, _shapes, weight_of_slot = ops.moe_build_routing(
                topk_ids, num_experts, m_cap, False, [], topk_w)
        else:
            masked_m, row_map, slot_of_flat, _slots, _shapes, _w = ops.moe_build_routing(
                topk_ids, num_experts, m_cap)
    hq, sh = ops.quantize_1x32_grouped_gather(hidden, slot_of_flat, topk, num_experts, m_cap)
    if fused_fc1:
        # The FC1 with its SwiGLU + MXFP8 requantize epilogue: no bf16 gate/up
        # slab and one kernel fewer.
        dq, sd = ops.linear_mxfp8_grouped_masked_swiglu(
            hq, w13_fp8, sh, sw13, masked_m, expected_m, max_active_groups, slot_to_expert, ps_fc1)
    else:
        gu = ops.linear_mxfp8_grouped_masked(
            hq, w13_fp8, sh, sw13, masked_m, expected_m, max_active_groups, slot_to_expert, ps_fc1)
        # Interleaved weights without the fused FC1 put gate_j and up_j in
        # adjacent columns, which is what the SwiGLU kernel's pairwise form reads.
        dq, sd = ops.silu_chunk_mul_quantize_1x32_grouped(gu, slot_of_flat, True, w13_interleaved)
    if scatter:
        # The FC2 adds its rows into the output itself, so the output must hold what
        # the layer adds to before that GEMM runs: a shared expert's gated row, or
        # zero.
        dst = out if out is not None else torch.empty(
            m, int(hidden.shape[1]), device=hidden.device, dtype=torch.bfloat16)
        if join is not None:
            join()
        if bias is None:
            dst.zero_()
        elif bias_scale is None:
            dst.copy_(bias)
        else:
            torch.mul(bias, bias_scale.unsqueeze(1), out=dst)
        return ops.linear_mxfp8_grouped_masked_combine(
            dq, w2_fp8, sd, sw2, masked_m, row_map, weight_of_slot, dst, expected_m, max_active_groups)
    dn = ops.linear_mxfp8_grouped_masked(
        dq, w2_fp8, sd, sw2, masked_m, expected_m, max_active_groups, slot_to_expert, ps_fc2)
    if join is not None:
        join()
    return ops.moe_combine(dn, slot_of_flat, topk_w, bias, bias_scale, out)


# The pre-2026-09-29 name, from when the function served sm_120/121 only.
_moe_layer_mxfp8_sm120_one = _moe_layer_mxfp8_one


def _moe_layer_mxfp8(
    x: torch.Tensor,
    w13_fp8: torch.Tensor,
    sw13: torch.Tensor,
    w2_fp8: torch.Tensor,
    sw2: torch.Tensor,
    ids: torch.Tensor,
    wts: torch.Tensor,
    *,
    chunk_tokens: int | None = None,
    bias: torch.Tensor | None = None,
    bias_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    on_output=None,
    join=None,
    w13_interleaved: bool = False,
    fused_combine: bool = False,
) -> torch.Tensor:
    """The masked-slab MXFP8 layer over all rows of ``x``, split into token chunks
    when asked: the body behind :func:`moe_layer_mxfp8_sm120` and behind the
    unified ``fish_scales_ops::moe_layer`` op (``fso.moe.layer``) on sm_100/103 and
    sm_120/121.

    The arguments are already conditioned: ``x`` bf16 ``[M, HIDDEN]`` contiguous,
    ``ids`` int32 and ``wts`` fp32 ``[M, topk]`` contiguous, the weights and their
    scale handles as the grouped quantizer produced them on this architecture. The
    chunk count is ``chunk_tokens`` when given and otherwise
    :func:`moe_layer_chunk_tokens_sm120`, which is one call unless
    ``FSO_MOE_SLAB_BUDGET_MB`` caps the transient slabs; that rule is the sm_120
    slab estimate and is applied unchanged on sm_100/103, where the per-group scale
    slabs pad the row capacity to 128 and the estimate is therefore slightly low for
    small chunks.
    """
    num_experts = int(w13_fp8.shape[0])
    hidden_size = int(x.shape[1])
    inter = int(w2_fp8.shape[2])
    m = int(x.shape[0])
    topk = int(ids.shape[1])
    if m == 0:
        return out if out is not None else torch.empty_like(x)  # an empty bucket: nothing to route

    chunk = int(chunk_tokens) if chunk_tokens else moe_layer_chunk_tokens_sm120(
        m, num_experts, hidden_size, inter)
    if chunk < 1:
        raise ValueError("chunk_tokens must be positive")
    if chunk >= m:
        result = _moe_layer_mxfp8_one(
            x, w13_fp8, sw13, w2_fp8, sw2, ids, wts, num_experts, topk, bias, bias_scale, out, join,
            w13_interleaved, fused_combine)
        if on_output is not None:
            on_output(result, 0, m)
        return result
    # Chunked: each call writes its own rows of the destination, so a caller that
    # supplied `out` pays no copy at all and one that did not pays the same single
    # copy per chunk it would have paid for the concatenation.
    dst = out if out is not None else torch.empty_like(x)
    for start in range(0, m, chunk):
        end = min(start + chunk, m)
        _moe_layer_mxfp8_one(
            x[start:end], w13_fp8, sw13, w2_fp8, sw2, ids[start:end], wts[start:end],
            num_experts, topk,
            None if bias is None else bias[start:end],
            None if bias_scale is None else bias_scale[start:end],
            dst[start:end],
            join if start == 0 else None,  # one join, before the first combine
            w13_interleaved, fused_combine)
        if on_output is not None:
            # These rows are final now: a collective may start on them while the
            # next chunk computes.
            on_output(dst[start:end], start, end)
    return dst


def moe_layer_mxfp8_sm120(
    hidden: torch.Tensor,
    w13_fp8: torch.Tensor,
    sw13: torch.Tensor,
    w2_fp8: torch.Tensor,
    sw2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    *,
    chunk_tokens: int | None = None,
    bias: torch.Tensor | None = None,
    bias_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    on_output=None,
    join=None,
    w13_interleaved: bool = False,
    fused_combine: bool = False,
) -> torch.Tensor:
    """Complete sm_120/121 (RTX 5090 / RTX PRO 6000) grouped-MoE layer behind a
    single stable interface — the MXFP8 masked-slab twin of
    :func:`~fish_scales_ops.gemm.moe_layer_fp8_sm90`, and the sm_120/121
    implementation behind ``fso.moe.layer``, the MoE surface for every
    architecture, which runs the same chain on sm_100/sm_103 as well.

    One call runs the six kernels of the layer: the routing builder, the fused
    token-gather + MXFP8 quantize, the grouped gate/up GEMM, the fused SwiGLU +
    MXFP8 requantize, the grouped down GEMM, and the weighted combine. The
    routing is derived on the device from ``topk_ids``; the host never reads a
    routing result, nothing allocates on a graph replay, and every shape is a
    function of the argument shapes alone, so a caller can capture the call per
    graph bucket.

    Expert-parallel-local ids. ``num_experts = w13_fp8.shape[0]`` is the expert
    count this call sees, which under expert parallelism is the rank-local count
    and not the model's. Every routed entry whose id falls outside
    ``[0, num_experts)`` is skipped at no expert cost: that covers the padding
    rows of a CUDA-graph or piecewise-graph bucket, which sglang labels with the
    id ``num_experts`` (its moe_align overflow slot) or ``-1``, and the entries
    an expert-parallel dispatcher rewrites to ``-1`` because another rank owns
    the expert. A skipped entry contributes nothing to its token's output row,
    and a token whose every entry is skipped gets an all-zero row, which is the
    identity for the cross-rank sum the caller performs afterwards.

    Capacity and chunking. The masked-slab layout gives each expert its own
    ``m_cap``-row window, and with top-k drawn without replacement one expert
    can receive as many rows as there are tokens, so the layer sizes
    ``m_cap = align(tokens, 4)`` — the smallest capacity that cannot overflow
    for any routing. Top-k without replacement is a precondition rather than a
    convenience: a token that named one expert twice would give that expert two
    rows, an expert could then hold more rows than there are tokens, and the
    routing builder would place the overflow in the next expert's window (past
    the slab, for the last expert). Every router produces distinct ids per token,
    and ``num_experts`` duplicated ids are not the same thing as the skipped ids
    above, which take no row at all. The transient slabs therefore cost
    ``num_experts * m_cap`` rows in five tensors at once
    (:func:`moe_layer_slab_bytes_per_token_sm120` gives the per-row total: about
    1.1 MB per token at 128 local experts, hidden 2048 and inter 512). A
    prefill bucket of 8192 tokens therefore asks for roughly 9 GB of them, which
    does not fit next to the weights on a 32 GB card.

    The default is nevertheless one call per layer, at any token count, because
    the alternative is expensive rather than merely different: a call walks every
    expert's weights once, so splitting a bucket into n chunks reads the expert
    weights n times. Measured on the RTX 5090, two chunks at 1024 tokens cost
    +80 % and five chunks at 4096 tokens +209 % on the Qwen3.5-35B-A3B routed
    layer. A caller that has to bound the peak instead of the latency asks for
    chunking explicitly, with ``chunk_tokens`` or by capping the slabs with
    ``FSO_MOE_SLAB_BUDGET_MB`` (:func:`moe_layer_chunk_tokens_sm120` answers what
    a given bucket would do); chunking is exact rather than an approximation,
    since a token's output depends only on its own row and on the expert weights,
    so the rows of a chunk are bit-identical to what one call produces. Both
    settings are host-side decisions taken from shapes, so a captured bucket's
    chunk count is fixed at capture. Above the point where the slabs stop fitting,
    the real answer is a contiguous (expert-sorted) sm_120 entry that sizes its
    activation by the routed rows instead of by capacity; that entry does not
    exist yet, and this one raises the allocator's out-of-memory error rather
    than quietly running several times slower.

    Weights. ``w13_fp8`` / ``w2_fp8`` are the per-expert 1×32 MXFP8 weights and
    ``sw13`` / ``sw2`` their opaque scale handles, as
    :func:`quantize_moe_weights_1x32_fp8` produces them **on this
    architecture** (the scale layout is arch-specific and not portable). The
    row order of ``w13`` is either the checkpoint's own ``[gate; up]``
    (``w13_interleaved=False``: the unfused FC1 followed by the SwiGLU kernel)
    or the gate/up-interleaved order that
    ``quantize_moe_weights_1x32_fp8(w13, w13_interleave=True)`` and
    :func:`interleave_w13_fp8` produce (``w13_interleaved=True``: the FC1
    carries the SwiGLU in its own epilogue, see the argument below). A serving
    artifact that ships block-scaled 128×128 FP8 experts is converted at load
    time by ``fso.moe.prepare_experts(..., format="bsfp8")``, which dequantizes
    each block to bf16 and requantizes it with that op on this device; that is a
    second quantization, and quantizing the bf16 master is the single-rounding
    alternative.

    Args:
        hidden: bf16 ``[M, HIDDEN]``.
        w13_fp8: float8_e4m3fn ``[num_experts, 2 * INTER, HIDDEN]``; ``sw13``
            its int32 scale handle.
        w2_fp8: float8_e4m3fn ``[num_experts, HIDDEN, INTER]``; ``sw2`` its
            int32 scale handle.
        topk_ids: int32 or int64 ``[M, topk]``; ids outside
            ``[0, num_experts)`` are skipped (see above).
        topk_w: float32 ``[M, topk]`` combine weights.
        chunk_tokens: tokens per layer call, overriding the budget rule.
        bias: bf16 ``[M, HIDDEN]`` added to the combine — a shared expert's
            output, folded in so the block does not pay another pass over the
            result (see :func:`moe_combine`).
        bias_scale: fp32 ``[M]`` per-token factor on ``bias``, e.g. a sigmoid
            shared-expert gate.
        fused_combine: allow the FC2 to add its rows into the output itself instead
            of storing the ``[G, m_cap, HIDDEN]`` slab a separate combine kernel
            reads back. This trades time for footprint: 58-61 % less transient
            memory for 0.2-7.0 % more time
            (:func:`moe_layer_fused_combine_engages_sm120` has the numbers), which
            is what makes a large prefill bucket fit at all where chunking — the
            other way to bound that peak — would cost +80 % or more. Off by default
            for two reasons: below the gate it buys nothing, and the adds being
            atomic makes the result stable in aggregate but not bit-reproducible run
            to run, which the rest of this surface guarantees.
        w13_interleaved: the FC1 weights were quantized with
            ``w13_interleave=True``, i.e. their rows alternate gate and up. That
            layout lets the FC1 carry the SwiGLU and the MXFP8 requantize in its
            own epilogue, which drops the bf16 ``[G, m_cap, 2*INTER]``
            intermediate and one kernel; where the fused kernel is unavailable
            the layer falls back to the unfused FC1 and the pairwise SwiGLU
            kernel, which reads the same layout. It is a load-time decision,
            because a model holds one weight layout.
        out: bf16 ``[M, HIDDEN]`` destination, for a caller that already owns the
            buffer the result has to land in — a collective's symmetric-memory
            buffer, for instance, so the all-reduce or reduce-scatter reads the
            result in place.
        on_output: ``callable(view, start, end)`` invoked on the current stream
            immediately after the output rows ``[start, end)`` are final. This is
            the communication hook: the layer's output is the rank's partial sum,
            and a caller that chunks a bucket gets each slice as it completes, so
            it can start its collective on its own stream (forking and joining
            with events, as this entry does for the block's shared expert) while
            the next slice still computes. Without chunking it fires once for the
            whole output. The callback must only issue CUDA work — it runs inside
            a graph capture like everything else here — and must not read the
            result on the host.

    Returns:
        bf16 ``[M, HIDDEN]`` — ``out`` itself when it was given.
    """
    _require_grouped_arch()
    if sm_major() != 12:
        raise NotImplementedError(
            f"moe_layer_mxfp8_sm120 is the sm_120/121 grouped MXFP8 MoE layer; this device is "
            f"sm_{sm_major()}0. fso.moe.layer is the MoE entry for every architecture: on "
            f"sm_100/sm_103 it runs this chain with the routing tensors that architecture's "
            f"grouped routes read (the packed active-expert list, the per-group problem shapes), "
            f"and on sm_90 (Hopper) it runs moe_layer_fp8_sm90."
        )
    if hidden.dim() != 2 or hidden.dtype != torch.bfloat16:
        raise ValueError("hidden must be bf16 [M, HIDDEN]")
    if w13_fp8.dim() != 3 or w2_fp8.dim() != 3:
        raise ValueError("w13_fp8 must be [E, 2*INTER, HIDDEN] and w2_fp8 [E, HIDDEN, INTER]")
    if topk_ids.dim() != 2 or topk_w.dim() != 2:
        raise ValueError("topk_ids and topk_w must be [M, topk]")
    num_experts = int(w13_fp8.shape[0])
    two_inter = int(w13_fp8.shape[1])
    hidden_size = int(hidden.shape[1])
    inter = int(w2_fp8.shape[2])
    if w2_fp8.shape[0] != num_experts:
        raise ValueError("w13_fp8 and w2_fp8 must hold the same expert count")
    if two_inter != 2 * inter:
        raise ValueError("w13_fp8 must have 2*INTER rows per expert, INTER from w2_fp8")
    if w13_fp8.shape[2] != hidden_size or w2_fp8.shape[1] != hidden_size:
        raise ValueError("w13_fp8 and w2_fp8 must both be contracted over HIDDEN")
    m = int(hidden.shape[0])
    topk = int(topk_ids.shape[1])
    if int(topk_w.shape[0]) != m or int(topk_ids.shape[0]) != m:
        raise ValueError("topk_ids and topk_w must have one row per token of hidden")
    if int(topk_w.shape[1]) != topk:
        raise ValueError("topk_ids and topk_w must have the same topk")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must be int32 or int64")

    # Argument conditioning, all of it host-side: int64 ids (torch.topk's own
    # dtype) are narrowed, weights are taken to fp32, and non-contiguous inputs
    # are made contiguous. Each is a no-op when the caller already passes the
    # expected form, which is what a serving stack does after the first call.
    ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
    ids = ids.contiguous()
    wts = topk_w if topk_w.dtype == torch.float32 else topk_w.to(torch.float32)
    wts = wts.contiguous()
    x = hidden if hidden.is_contiguous() else hidden.contiguous()
    if bias is not None and (bias.dim() != 2 or bias.shape[0] != m or bias.shape[1] != hidden_size):
        raise ValueError("bias must be [M, HIDDEN]")
    if bias_scale is not None and bias_scale.numel() != m:
        raise ValueError("bias_scale must have one entry per token")
    if out is not None and (out.dim() != 2 or out.shape[0] != m or out.shape[1] != hidden_size
                            or out.dtype != torch.bfloat16):
        raise ValueError("out must be bf16 [M, HIDDEN]")
    return _moe_layer_mxfp8(
        x, w13_fp8, sw13, w2_fp8, sw2, ids, wts, chunk_tokens=chunk_tokens, bias=bias,
        bias_scale=bias_scale, out=out, on_output=on_output, join=join,
        w13_interleaved=w13_interleaved, fused_combine=fused_combine)


# One side stream per process for the block's shared expert. It is created on
# first use and never again, like the grouped argument pool and the routing
# scratch, because creating a stream or an event inside a CUDA-graph capture
# would put host-side setup in the middle of the capture.
_moe_block_side_stream = None


def _moe_block_side_stream_get():
    global _moe_block_side_stream
    if _moe_block_side_stream is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "moe_block_mxfp8_sm120 wants to run the shared expert on a side stream, and the "
                "first call of the process creates that stream, which cannot happen inside a CUDA "
                "graph capture. Call the block once eagerly before capturing, or pass "
                "overlap_shared=False (FSO_MOE_BLOCK_OVERLAP=0 disables it process-wide)."
            )
        _moe_block_side_stream = torch.cuda.Stream()
    return _moe_block_side_stream


_moe_block_overlap_default: bool | None = None


def _moe_block_overlap_enabled() -> bool:
    """Whether the block overlaps its shared expert by default, read once."""
    global _moe_block_overlap_default
    if _moe_block_overlap_default is None:
        raw = os.environ.get("FSO_MOE_BLOCK_OVERLAP", "").strip()
        _moe_block_overlap_default = raw != "0"
    return _moe_block_overlap_default


def _shared_expert_mxfp8_sm120(hidden, w13_fp8, sw13, w2_fp8, sw2):
    """The always-on dense expert of a MoE block, MXFP8, four kernels: quantize,
    gate/up, fused SwiGLU + requantize, down."""
    ops = torch.ops.fish_scales_ops
    xq, sx = ops.quantize_1x32_packed(hidden, True)
    gu = ops.linear_mxfp8_raw(xq, w13_fp8, sx, sw13)
    hq, sh = ops.silu_chunk_mul_quantize_1x32(gu, True)
    return ops.linear_mxfp8_raw(hq, w2_fp8, sh, sw2)


def moe_block_mxfp8_sm120(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    w13_fp8: torch.Tensor,
    sw13: torch.Tensor,
    w2_fp8: torch.Tensor,
    sw2: torch.Tensor,
    *,
    topk: int,
    renormalize: bool = True,
    num_token_non_padded: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    shared_w13_fp8: torch.Tensor | None = None,
    shared_sw13: torch.Tensor | None = None,
    shared_w2_fp8: torch.Tensor | None = None,
    shared_sw2: torch.Tensor | None = None,
    shared_gate_in_router: bool = False,
    w13_interleaved: bool = False,
    fused_combine: bool = False,
    shared_out: torch.Tensor | None = None,
    shared_gate: torch.Tensor | None = None,
    overlap_shared: bool | None = None,
    out: torch.Tensor | None = None,
    on_output=None,
    chunk_tokens: int | None = None,
) -> torch.Tensor:
    """A whole sm_120/121 MoE block in one capture-safe call: the router, the
    routed experts, the shared expert, and the gated add that joins them.

    This is the shape a serving stack's block has — sglang's
    ``Qwen2MoeSparseMoeBlock`` / ``Qwen3MoeSparseMoeBlock`` — minus the
    collective at the end, which stays with the caller because a library has no
    business owning the process group. What it replaces, per layer:

    * the router GEMM, the softmax top-k, the padded-row masking and (under
      expert parallelism) the id remap: 4-5 launches become 2, of which one is
      the GEMM (see :func:`moe_router_topk`);
    * the routed expert layer: as :func:`moe_layer_mxfp8_sm120`, 6 kernels;
    * the shared expert: 4 kernels, its sigmoid gate computed in the router for
      free when its weight row is concatenated onto the router's;
    * the gated add of the two outputs: folded into the combine, so the block
      does not read and write the whole ``[M, HIDDEN]`` result twice more.

    Parallelism. The block is written for every form the 5090 family serves.
    *Tensor parallel* shards the experts along the intermediate dimension, so
    ``INTER`` here is the per-rank size and has to stay a multiple of 128 (the
    scale layout's block); with ``INTER = 512`` that admits tp 1, 2 and 4, and
    tp 8 is refused with a message rather than silently mis-scaled. The block's
    output is the rank's partial sum, which the caller all-reduces exactly as it
    does today. *Expert parallel* gives the rank a subset of the experts: pass
    ``expert_map`` (int32 ``[num_experts + 1]``, the dispatcher's global-to-local
    table, remote experts and the padded-row sentinel mapped outside
    ``[0, num_local_experts)``) and the block routes over all experts, computes
    the local ones, and leaves every other entry at zero cost. *Data parallel*
    attention hands the block a padded token buffer whose valid length is a
    device value: pass ``num_token_non_padded`` and the padded rows cost nothing
    and come out zero, and pass ``out`` to write straight into the buffer the
    scatter or reduce-scatter already owns.

    Args:
        hidden: bf16 ``[M, HIDDEN]``.
        router_weight: bf16 ``[num_experts, HIDDEN]``, or
            ``[num_experts + 1, HIDDEN]`` with ``shared_gate_in_router``, whose
            last row is the shared expert's gate.
        w13_fp8, sw13, w2_fp8, sw2: the rank-local routed experts, exactly as
            :func:`moe_layer_mxfp8_sm120` takes them.
        topk, renormalize: the router's selection, matching sglang's ``TopK``.
        num_token_non_padded: int32 ``[1]`` on device (see above).
        expert_map: int32 ``[num_experts + 1]`` on device (see above). Without
            it the router's ids are global, so the local expert count must equal
            ``num_experts``.
        shared_w13_fp8, shared_sw13, shared_w2_fp8, shared_sw2: the shared
            expert's MXFP8 weights (``[2*INTER_s, HIDDEN]`` and
            ``[HIDDEN, INTER_s]``). Omit them and the block is routed-only.
        shared_gate_in_router: the router weight carries the shared expert's
            gate row, so its sigmoid comes out of the router kernel.
        w13_interleaved: the routed experts' FC1 weights carry the interleaved
            gate/up row order, which enables the fused FC1 epilogue — see
            :func:`moe_layer_mxfp8_sm120`.
        fused_combine: allow the fused-combine FC2 — see
            :func:`moe_layer_mxfp8_sm120`.
        shared_out: a shared-expert output the caller computed itself, bf16
            ``[M, HIDDEN]``. A caller that already overlaps its shared expert
            with the routed path on a second stream keeps that overlap and still
            gets the fused add; it is an error to pass both this and the shared
            weights.
        shared_gate: fp32 ``[M]`` gate for ``shared_out``, when the caller
            computed that too.
        overlap_shared: run the shared expert on a side stream so it overlaps the
            routed path, joining just before the combine that adds it. On by
            default when the block owns the shared expert;
            ``FSO_MOE_BLOCK_OVERLAP=0`` turns it off process-wide and
            ``overlap_shared=False`` per call. The output is bit-identical either
            way — the join is a stream dependency, not a change of arithmetic —
            and the side stream is created on the first call, so that call has to
            be eager, like the first call of any other pool in this library.
        out: bf16 ``[M, HIDDEN]`` destination, which may be a collective's own
            buffer.
        on_output: the communication hook, ``callable(view, start, end)``, called
            on the current stream as each slice of the output becomes final — see
            :func:`moe_layer_mxfp8_sm120`. The block's output is the rank's
            partial sum, so this is where an all-reduce (tensor or expert
            parallel) or a reduce-scatter (data parallel) starts.
        chunk_tokens: as :func:`moe_layer_mxfp8_sm120`.

    Returns:
        bf16 ``[M, HIDDEN]`` — ``out`` itself when it was given.
    """
    _require_grouped_arch()
    if sm_major() != 12:
        raise NotImplementedError(
            f"moe_block_mxfp8_sm120 is the sm_120/121 MoE block; this device is sm_{sm_major()}0. "
            f"See moe_layer_mxfp8_sm120 for the architecture note."
        )
    if hidden.dim() != 2 or hidden.dtype != torch.bfloat16:
        raise ValueError("hidden must be bf16 [M, HIDDEN]")
    if router_weight.dim() != 2 or router_weight.dtype != torch.bfloat16:
        raise ValueError("router_weight must be bf16 [num_experts(+1), HIDDEN]")
    if router_weight.shape[1] != hidden.shape[1]:
        raise ValueError("router_weight must contract over HIDDEN")
    has_shared_weights = shared_w13_fp8 is not None
    if has_shared_weights and shared_out is not None:
        raise ValueError(
            "pass either the shared expert's weights (the block runs it) or shared_out "
            "(the caller ran it), not both"
        )
    if has_shared_weights and (shared_sw13 is None or shared_w2_fp8 is None or shared_sw2 is None):
        raise ValueError("the shared expert needs all four of its weight and scale tensors")
    num_experts = int(router_weight.shape[0]) - (1 if shared_gate_in_router else 0)
    num_local = int(w13_fp8.shape[0])
    if expert_map is None and num_local != num_experts:
        raise ValueError(
            f"the router selects over {num_experts} experts but this rank holds {num_local}; "
            f"an expert-parallel rank must pass expert_map (int32 [num_experts + 1]) so the ids "
            f"are remapped to its own range"
        )
    inter = int(w2_fp8.shape[2])
    if inter % 128 != 0:
        raise ValueError(
            f"INTER per rank is {inter}, which is not a multiple of 128: the 1x32 scale layout "
            f"blocks the K dimension by 128, so a tensor-parallel split has to keep INTER/tp a "
            f"multiple of 128 (at INTER=512 that is tp <= 4)"
        )
    m = int(hidden.shape[0])
    x = hidden if hidden.is_contiguous() else hidden.contiguous()

    ids, weights, gate = moe_router_topk(
        x, router_weight, topk, renormalize=renormalize,
        with_shared_gate=shared_gate_in_router,
        num_token_non_padded=num_token_non_padded, expert_map=expert_map,
    )
    if shared_gate_in_router:
        shared_gate = gate
    join = None
    if has_shared_weights and m > 0:
        overlap = _moe_block_overlap_enabled() if overlap_shared is None else bool(overlap_shared)
        if overlap:
            # The shared expert depends only on the block's input, so it is issued
            # on a side stream and waited for at the latest point that needs it:
            # after the routed layer's down projection, just before the combine
            # that adds it. The two branches then run concurrently, in eager mode
            # and inside a captured graph alike -- `wait_stream` records an event
            # on each side, which is what capture follows to build the two
            # branches. sglang's block does the same thing from the model side
            # with an alt stream; a caller that already does can pass shared_out
            # and keep its own arrangement.
            current = torch.cuda.current_stream()
            side = _moe_block_side_stream_get()
            side.wait_stream(current)
            with torch.cuda.stream(side):
                shared_out = _shared_expert_mxfp8_sm120(
                    x, shared_w13_fp8, shared_sw13, shared_w2_fp8, shared_sw2)
            # The combine reads it on the current stream, so the allocator must
            # know not to reuse the block before that read (a no-op under
            # capture, where the graph's pool owns it).
            shared_out.record_stream(current)
            join = lambda: current.wait_stream(side)  # noqa: E731
        else:
            shared_out = _shared_expert_mxfp8_sm120(
                x, shared_w13_fp8, shared_sw13, shared_w2_fp8, shared_sw2)
    if shared_out is None and shared_gate is not None:
        raise ValueError("shared_gate needs a shared expert output to scale")

    return moe_layer_mxfp8_sm120(
        x, w13_fp8, sw13, w2_fp8, sw2, ids, weights,
        chunk_tokens=chunk_tokens, bias=shared_out, bias_scale=shared_gate, out=out,
        on_output=on_output, join=join, w13_interleaved=w13_interleaved, fused_combine=fused_combine,
    )
