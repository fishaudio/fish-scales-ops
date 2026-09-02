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
    # sm_100/sm_103 (Blackwell datacenter) has a three-tier best-per-shape
    # dispatch — cuBLAS scaled_mm on decode / DSL on prefill+peak / C++
    # cascade for the narrow-N split-K decode + square cubic paths. Full
    # rationale in `_sm100_dispatch.py`.
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
# Grouped (MoE) masked-layout surface — sm_120 only (M1; sm_100/103 in M3).
#
# Masked semantics (DeepGEMM-style): G expert groups, each with a fixed row
# capacity ``m_cap`` and a per-group valid-row count ``masked_m[g]`` that
# lives ON DEVICE and is read only by the kernels — never on the host — so
# every op below is CUDA-Graph capture-safe with dynamic routing: replays
# honour whatever counts the masked_m buffer holds at replay time.
# Rows at or beyond masked_m[g] hold undefined bytes in every tensor.


def _require_sm120_grouped() -> None:
    if sm_major() != 12:
        raise NotImplementedError(
            "the grouped MXFP8 (MoE) path is sm_120/121-only for now; "
            "sm_100/103 lands with milestone M3, sm_90 with M4."
        )


def linear_mxfp8_grouped_masked(
    a_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    sa: torch.Tensor,
    sw: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
) -> torch.Tensor:
    """Grouped block-scaled MXFP8 GEMM over per-expert weights (masked).

    Args:
        a_fp8: float8_e4m3fn ``[G, m_cap, K]`` — per-group activation slab
            from :func:`quantize_1x32_grouped_gather_fp8` (or
            :func:`silu_chunk_mul_quantize_1x32_grouped_fp8`).
        w_fp8: float8_e4m3fn ``[G, N, K]`` — per-expert weights (see
            :func:`quantize_moe_weights_1x32_fp8`).
        sa: int32 ``[G, K/128, m_cap]`` per-group K-major packed UE8M0.
        sw: int32 ``[G, K/128, N]`` per-group K-major packed UE8M0.
        masked_m: int32 ``[G]`` on device — valid rows per group. Caller
            contract: ``masked_m[g] <= m_cap`` for every g (the kernel does
            not check; an oversized count silently drops that group's
            overflow rows at the TMA bounds).
        expected_m: host-side static hint (``ceil(total_rows / G)``) used
            only for tile selection; per-call value must be a plain int so
            graph capture stays shape-static.

    Returns:
        bfloat16 ``[G, m_cap, N]``; rows ``>= masked_m[g]`` are undefined.

    Constraints: ``K % 128 == 0``, ``N % 128 == 0``, ``m_cap % 4 == 0``.
    """
    _require_sm120_grouped()
    return torch.ops.fish_scales_ops.linear_mxfp8_grouped_masked(
        a_fp8, w_fp8, sa, sw, masked_m, expected_m
    )


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
        (a_fp8 ``[G, m_cap, K]``, sa int32 ``[G, K/128, m_cap]``) ready for
        :func:`linear_mxfp8_grouped_masked`. Rows no pair maps to are
        undefined (the GEMM's masked contract ignores them).
    """
    _require_sm120_grouped()
    return torch.ops.fish_scales_ops.quantize_1x32_grouped_gather(
        x.contiguous(), slot_of_flat, topk, num_groups, m_cap, True
    )


def silu_chunk_mul_quantize_1x32_grouped_fp8(
    gu: torch.Tensor,
    slot_of_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped SwiGLU prologue + MXFP8 quantize over the flat pair space.

    ``gu`` is the grouped gate_up output (bf16 ``[G, m_cap, 2*INTER]``,
    gate || up chunked). Each routed pair processes its own row
    (``slot_of_flat[i]``) in place; rows no pair maps to stay undefined.

    Returns:
        (h_fp8 ``[G, m_cap, INTER]``, sh int32 ``[G, INTER/128, m_cap]``).
    """
    _require_sm120_grouped()
    return torch.ops.fish_scales_ops.silu_chunk_mul_quantize_1x32_grouped(
        gu, slot_of_flat, True
    )


def moe_build_routing(
    topk_ids: torch.Tensor,
    num_groups: int,
    m_cap: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """topk routing -> masked-layout index tensors, one kernel launch.

    Args:
        topk_ids: int32 ``[M, topk]``, no-replacement expert ids.
        num_groups: G (<= 1024).
        m_cap: per-group row capacity (multiple of 4, >= M).

    Returns:
        (masked_m ``[G]`` int32, row_map ``[G * m_cap]`` int32 — slot ->
        source token, slots at or beyond masked_m[g] uninitialised by
        design, slot_of_flat ``[M * topk]`` int32). Slot order within a
        group is atomic-arrival order; every consumer goes through these
        maps consistently.
    """
    _require_sm120_grouped()
    return torch.ops.fish_scales_ops.moe_build_routing(topk_ids, num_groups, m_cap)


def moe_combine(
    dn: torch.Tensor,
    slot_of_flat: torch.Tensor,
    topk_w: torch.Tensor,
) -> torch.Tensor:
    """Weighted combine of routed expert outputs, one kernel launch.

    ``out[t] = sum_j topk_w[t, j] * dn.view(-1, H)[slot_of_flat[t*topk+j]]``.
    """
    _require_sm120_grouped()
    return torch.ops.fish_scales_ops.moe_combine(dn, slot_of_flat, topk_w)


def quantize_moe_weights_1x32_fp8(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Offline per-expert weight quantize for the grouped GEMM (sm_120).

    ``w`` is bf16 ``[G, N, K]``. Loops experts through the flat 1×32
    quantizer and stacks results — offline cost, not a serving-path op.

    Returns:
        (w_fp8 ``[G, N, K]``, sw int32 ``[G, K/128, N]``).

    Constraints: ``N % 128 == 0`` (so the per-expert pad(N,4) == N),
    ``K % 128 == 0``.
    """
    _require_sm120_grouped()
    if w.dim() != 3:
        raise ValueError("w must be [G, N, K]")
    G, N, K = w.shape
    if N % 128 != 0 or K % 128 != 0:
        raise ValueError("N and K must be multiples of 128")
    w_fp8 = torch.empty(G, N, K, device=w.device, dtype=torch.float8_e4m3fn)
    sw = torch.empty(G, K // 128, N, device=w.device, dtype=torch.int32)
    for g in range(G):
        q, s = torch.ops.fish_scales_ops.quantize_1x32_packed(w[g].contiguous(), True)
        w_fp8[g].copy_(q)
        # s is [pad(N,4), K/128] with strides (1, N) — K-major. Its raw byte
        # order equals the [K/128, N] slab the grouped kernel expects.
        sw[g].copy_(s.t().view(K // 128, N))
    return w_fp8, sw
