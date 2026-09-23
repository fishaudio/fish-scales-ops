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
        NotImplementedError: on sm_120/121 and sm_90. The two fused epilogues
            are clones of CUTLASS's sm_100 NoSmem epilogues, which exist only
            on sm_100/sm_103; the other architectures drive different grouped
            kernels (and sm_90 has no MXFP8 hardware at all).
    """
    if sm_major() != 10:
        raise NotImplementedError(
            f"linear_mxfp8_grouped_masked_swiglu needs sm_100/sm_103 (B200 / B300); this device is sm_{sm_major()}0. "
            "The fused FC1 epilogues are clones of CUTLASS's sm_100 NoSmem epilogues and exist only there. "
            "On sm_120/121 use linear_mxfp8_grouped_masked + silu_chunk_mul_quantize_1x32_grouped_fp8; "
            "on sm_90 use the block-FP8 grouped surface."
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
        NotImplementedError: on any architecture but sm_100/sm_103, which is
            the only one with a fused FC1 to consume the layout.
    """
    if sm_major() != 10:
        raise NotImplementedError(
            f"interleave_w13_fp8 targets the sm_100/sm_103 fused FC1 weight layout; this device is "
            f"sm_{sm_major()}0, which has no fused FC1 and no use for the interleaved order."
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
    from :func:`moe_build_routing`). One launch, no dead warps — the
    original padded ``G * m_cap`` iteration burned ~27 µs of early-outs at
    M=512.

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
    pair maps to stay undefined.

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
    if pairwise and sm_major() != 10:
        raise NotImplementedError(
            f"silu_chunk_mul_quantize_1x32_grouped_fp8(pairwise=True) targets the sm_100/sm_103 "
            f"interleaved gate_up layout; this device is sm_{sm_major()}0, whose grouped FC1 always "
            f"produces the chunked [gate; up] order. Call it with pairwise=False there."
        )
    return torch.ops.fish_scales_ops.silu_chunk_mul_quantize_1x32_grouped(
        gu, slot_of_flat, True, pairwise
    )


def moe_build_routing(
    topk_ids: torch.Tensor,
    num_groups: int,
    m_cap: int,
    *,
    with_slots: bool = False,
    problem_shapes_for: list[tuple[int, int]] | None = None,
) -> tuple[torch.Tensor, ...]:
    """topk routing -> masked-layout index tensors, one kernel launch.

    Args:
        topk_ids: int32 ``[M, topk]``, no-replacement expert ids.
        num_groups: G (<= 1024).
        m_cap: per-group row capacity (multiple of 4, >= M).
        with_slots: also return the packed active-expert list. It costs a
            block-wide scan over the per-expert histogram this kernel already
            holds, and it saves the sm_100/103 slot-bound grouped GEMM the
            one-block launch it otherwise makes per call to build the same
            list. Every architecture produces it; only that route reads it.
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
        design, slot_of_flat ``[M * topk]`` int32). Slot order within a
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
    masked_m, row_map, slot_of_flat, slot_to_expert, problem_shapes = (
        torch.ops.fish_scales_ops.moe_build_routing(topk_ids, num_groups, m_cap, with_slots, nk)
    )
    out: tuple[torch.Tensor, ...] = (masked_m, row_map, slot_of_flat)
    if with_slots:
        out = out + (slot_to_expert,)
    if nk:
        # Views of one [P, G, 3] allocation: each is a contiguous [G, 3]
        # tensor, which is what the GEMM ops check for.
        out = out + ([problem_shapes[i] for i in range(problem_shapes.shape[0])],)
    return out


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
) -> torch.Tensor:
    """Weighted combine of routed expert outputs, one kernel launch.

    ``out[t] = sum_j topk_w[t, j] * dn.view(-1, H)[slot_of_flat[t*topk+j]]``.
    """
    # arch-agnostic glue (works on sm_90 and sm_120)
    return torch.ops.fish_scales_ops.moe_combine(dn, slot_of_flat, topk_w)


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
        if sm_major() != 10:
            raise NotImplementedError(
                f"quantize_moe_weights_1x32_fp8(w13_interleave=True) targets the sm_100/sm_103 fused FC1; "
                f"this device is sm_{sm_major()}0, which has no fused FC1."
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
