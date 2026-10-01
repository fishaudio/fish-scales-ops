"""``fish_scales_ops.compat`` — the explicit dense ops and the MoE per-step pieces.

This namespace keeps the names that existing callers use. New code uses the two
stable entries instead: :mod:`fish_scales_ops.dense` (``prepare_weight`` once at
load time, ``linear`` per call) and :mod:`fish_scales_ops.moe`
(``prepare_experts`` and ``layer``). Each of those runs the architecture dispatch
inside one torch custom op, so its caller never names an architecture or a scale
layout.

What this namespace promises:

* Every name below keeps resolving to the same object.
* The eleven explicit dense ops keep their signatures and their behaviour. They
  are format-specific: the caller picks block-FP8 or MXFP8, and the scale
  tensors they take and return are architecture-native handles whose layout the
  caller has to respect (``docs/api/compat.md``, *Scale layouts*).
* The thirty-four MoE pieces are the building blocks of the ``fso.moe`` chains.
  Their arguments follow the kernels: arguments, layouts and architecture
  coverage change when a kernel or a route changes. They are public so that a
  benchmark, a test or a caller composing its own chain can reach them.

Before fish-scales-ops 0.2.0 these names were exported from
:mod:`fish_scales_ops.gemm`. Every ``fish_scales_ops.gemm.<name>`` still resolves
to the same object as ``fish_scales_ops.compat.<name>`` and raises one
``DeprecationWarning`` per name; fish-scales-ops 0.3.0 removes
``fish_scales_ops.gemm``. The implementation modules stay where they are
(``fish_scales_ops/gemm/fp8.py``, ``mxfp8.py``, ``bf16.py``); this module only
re-exports from them.

Groups:

* block-FP8, 1x128 activation scales and 128x128 weight scales, on every
  architecture: ``quantize_1x128_fp8``, ``quantize_1x128_fp8_packed``,
  ``quantize_128x128_fp8``, ``repack_fp8_act_scales``,
  ``repack_fp8_wgt_scales``, ``linear_fp8`` and ``linear_qx`` (bf16 activation,
  quantized inside the op; sm_90 and sm_120/121);
* MXFP8 1x32 on sm_100/103 and sm_120/121: ``quantize_1x32_fp8``,
  ``silu_chunk_mul_quantize_1x32_fp8`` and ``linear_mxfp8``;
* bf16 in and out, block-FP8 inside: ``linear_bf16``;
* the per-architecture MoE layers and the sm_120/121 block:
  ``moe_layer_fp8_sm90``, ``moe_layer_mxfp8_sm120``, ``moe_block_mxfp8_sm120``;
* MoE memory planning: ``moe_layer_transient_bytes_sm90``,
  ``moe_layer_transient_bytes_mxfp8``, ``moe_layer_fused_combine_engages_sm120``;
* the MoE router: ``moe_router_topk``, ``moe_topk_from_logits``;
* the MoE routing builders and combines: ``moe_build_routing``,
  ``moe_build_sorted``, ``moe_combine``, ``moe_combine_sorted``;
* the sm_90 block-FP8 grouped chain: ``linear_fp8_grouped_masked``,
  ``linear_fp8_grouped_contiguous``, ``linear_fp8_grouped_contiguous_swapab``,
  ``quantize_1x128_grouped_gather_sm90``, ``quantize_1x128_sorted_gather_sm90``,
  ``silu_chunk_mul_quantize_1x128_grouped_sm90``,
  ``silu_chunk_mul_quantize_1x128_sorted_sm90``,
  ``quantize_moe_weights_1x128_fp8_sm90`` and the swap-AB tile rule
  ``MOE_SWAP_BLOCK_N_CASCADE``, ``moe_swap_ab_block_n``, ``moe_swap_ab_max_m``;
* the MXFP8 grouped chain of sm_100/103 and sm_120/121:
  ``linear_mxfp8_grouped_masked``, ``linear_mxfp8_grouped_masked_swiglu``,
  ``linear_mxfp8_grouped_masked_combine``, ``quantize_1x32_grouped_gather_fp8``,
  ``silu_chunk_mul_quantize_1x32_grouped_fp8``, ``quantize_moe_weights_1x32_fp8``,
  ``interleave_w13_fp8`` and the route queries
  ``mxfp8_grouped_swiglu_fused_route``, ``mxfp8_grouped_swiglu_available``,
  ``mxfp8_grouped_slot_possible``, ``mxfp8_grouped_problem_shapes_consumed``.
"""
from ..gemm.bf16 import linear_bf16
from ..gemm.fp8 import (
    MOE_SWAP_BLOCK_N_CASCADE,
    linear_fp8,
    linear_fp8_grouped_contiguous,
    linear_fp8_grouped_contiguous_swapab,
    linear_fp8_grouped_masked,
    linear_qx,
    moe_layer_fp8_sm90,
    moe_layer_transient_bytes_sm90,
    moe_swap_ab_block_n,
    moe_swap_ab_max_m,
    quantize_128x128_fp8,
    quantize_1x128_fp8,
    quantize_1x128_fp8_packed,
    quantize_1x128_grouped_gather_sm90,
    quantize_1x128_sorted_gather_sm90,
    quantize_moe_weights_1x128_fp8_sm90,
    repack_fp8_act_scales,
    repack_fp8_wgt_scales,
    silu_chunk_mul_quantize_1x128_grouped_sm90,
    silu_chunk_mul_quantize_1x128_sorted_sm90,
)
from ..gemm.mxfp8 import (
    interleave_w13_fp8,
    linear_mxfp8,
    linear_mxfp8_grouped_masked,
    linear_mxfp8_grouped_masked_combine,
    linear_mxfp8_grouped_masked_swiglu,
    moe_block_mxfp8_sm120,
    moe_build_routing,
    moe_build_sorted,
    moe_combine,
    moe_combine_sorted,
    moe_layer_fused_combine_engages_sm120,
    moe_layer_mxfp8_sm120,
    moe_layer_transient_bytes_mxfp8,
    moe_router_topk,
    moe_topk_from_logits,
    mxfp8_grouped_problem_shapes_consumed,
    mxfp8_grouped_slot_possible,
    mxfp8_grouped_swiglu_available,
    mxfp8_grouped_swiglu_fused_route,
    quantize_1x32_fp8,
    quantize_1x32_grouped_gather_fp8,
    quantize_moe_weights_1x32_fp8,
    silu_chunk_mul_quantize_1x32_fp8,
    silu_chunk_mul_quantize_1x32_grouped_fp8,
)

__all__ = [
    # dense: block-FP8 1x128 / 128x128, every architecture
    "quantize_1x128_fp8",
    "quantize_1x128_fp8_packed",
    "quantize_128x128_fp8",
    "repack_fp8_act_scales",
    "repack_fp8_wgt_scales",
    "linear_fp8",
    "linear_qx",
    # dense: MXFP8 1x32, sm_100/103 and sm_120/121
    "quantize_1x32_fp8",
    "silu_chunk_mul_quantize_1x32_fp8",
    "linear_mxfp8",
    # dense: bf16 in and out
    "linear_bf16",
    # MoE: per-architecture layers and the sm_120/121 block
    "moe_layer_fp8_sm90",
    "moe_layer_mxfp8_sm120",
    "moe_block_mxfp8_sm120",
    # MoE: memory planning
    "moe_layer_transient_bytes_sm90",
    "moe_layer_transient_bytes_mxfp8",
    "moe_layer_fused_combine_engages_sm120",
    # MoE: router
    "moe_router_topk",
    "moe_topk_from_logits",
    # MoE: routing builders and combines
    "moe_build_routing",
    "moe_build_sorted",
    "moe_combine",
    "moe_combine_sorted",
    # MoE: sm_90 block-FP8 grouped chain
    "linear_fp8_grouped_masked",
    "linear_fp8_grouped_contiguous",
    "linear_fp8_grouped_contiguous_swapab",
    "quantize_1x128_grouped_gather_sm90",
    "quantize_1x128_sorted_gather_sm90",
    "silu_chunk_mul_quantize_1x128_grouped_sm90",
    "silu_chunk_mul_quantize_1x128_sorted_sm90",
    "quantize_moe_weights_1x128_fp8_sm90",
    "MOE_SWAP_BLOCK_N_CASCADE",
    "moe_swap_ab_block_n",
    "moe_swap_ab_max_m",
    # MoE: MXFP8 grouped chain (sm_100/103, sm_120/121)
    "linear_mxfp8_grouped_masked",
    "linear_mxfp8_grouped_masked_swiglu",
    "linear_mxfp8_grouped_masked_combine",
    "quantize_1x32_grouped_gather_fp8",
    "silu_chunk_mul_quantize_1x32_grouped_fp8",
    "quantize_moe_weights_1x32_fp8",
    "interleave_w13_fp8",
    "mxfp8_grouped_swiglu_fused_route",
    "mxfp8_grouped_swiglu_available",
    "mxfp8_grouped_slot_possible",
    "mxfp8_grouped_problem_shapes_consumed",
]
