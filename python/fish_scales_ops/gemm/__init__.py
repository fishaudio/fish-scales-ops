"""``fish_scales_ops.gemm`` — deprecated: the old path of the explicit GEMM ops.

Every name this namespace used to export resolves to the same object as
``fish_scales_ops.compat.<name>``, and the first lookup of each name in a
process raises one ``DeprecationWarning``. fish-scales-ops 0.3.0 removes
``fish_scales_ops.gemm``.

* The eleven dense ops (``quantize_1x128_fp8``, ``quantize_1x128_fp8_packed``,
  ``quantize_128x128_fp8``, ``repack_fp8_act_scales``, ``repack_fp8_wgt_scales``,
  ``linear_fp8``, ``linear_qx``, ``quantize_1x32_fp8``,
  ``silu_chunk_mul_quantize_1x32_fp8``, ``linear_mxfp8``, ``linear_bf16``) are
  in :mod:`fish_scales_ops.compat`. New code uses :mod:`fish_scales_ops.dense`,
  whose ``prepare_weight`` and ``linear`` run the architecture dispatch inside
  one torch custom op.
* The thirty-four MoE per-step pieces are in :mod:`fish_scales_ops.compat` as
  well. New code uses :mod:`fish_scales_ops.moe`.

This package also holds the implementation modules (``fp8.py``, ``mxfp8.py``,
``bf16.py`` and the sm_100/sm_103 router ``_sm100_*.py``). They stay here, and
importing them by their module path raises no warning.
"""
import warnings as _warnings

from . import bf16, fp8, mxfp8  # noqa: F401  (the implementation modules, not the deprecated names)

# The dense ops this namespace exported, in the order of its former __all__.
_DENSE = (
    "quantize_1x128_fp8",
    "quantize_1x128_fp8_packed",
    "quantize_128x128_fp8",
    "repack_fp8_act_scales",
    "repack_fp8_wgt_scales",
    "linear_fp8",
    "linear_qx",
    "quantize_1x32_fp8",
    "silu_chunk_mul_quantize_1x32_fp8",
    "linear_mxfp8",
    "linear_bf16",
)

# The MoE per-step pieces this namespace exported before they moved.
_MOE = (
    "moe_layer_fp8_sm90", "moe_layer_mxfp8_sm120", "moe_block_mxfp8_sm120",
    "moe_layer_transient_bytes_sm90", "moe_layer_transient_bytes_mxfp8",
    "moe_layer_fused_combine_engages_sm120",
    "moe_router_topk", "moe_topk_from_logits",
    "moe_build_routing", "moe_build_sorted", "moe_combine", "moe_combine_sorted",
    "linear_fp8_grouped_masked", "linear_fp8_grouped_contiguous",
    "linear_fp8_grouped_contiguous_swapab", "quantize_1x128_grouped_gather_sm90",
    "quantize_1x128_sorted_gather_sm90", "silu_chunk_mul_quantize_1x128_grouped_sm90",
    "silu_chunk_mul_quantize_1x128_sorted_sm90", "quantize_moe_weights_1x128_fp8_sm90",
    "MOE_SWAP_BLOCK_N_CASCADE", "moe_swap_ab_block_n", "moe_swap_ab_max_m",
    "linear_mxfp8_grouped_masked", "linear_mxfp8_grouped_masked_swiglu",
    "linear_mxfp8_grouped_masked_combine", "quantize_1x32_grouped_gather_fp8",
    "silu_chunk_mul_quantize_1x32_grouped_fp8", "quantize_moe_weights_1x32_fp8",
    "interleave_w13_fp8", "mxfp8_grouped_swiglu_fused_route", "mxfp8_grouped_swiglu_available",
    "mxfp8_grouped_slot_possible", "mxfp8_grouped_problem_shapes_consumed",
)

_DEPRECATED = frozenset(_DENSE + _MOE)

# A star import keeps importing the eleven dense names this namespace exported,
# each with its warning.
__all__ = list(_DENSE)


def __getattr__(name: str):
    if name in _DEPRECATED:
        if name in _DENSE:
            instead = (f"use fish_scales_ops.compat.{name}, the same object; new code uses "
                       f"fish_scales_ops.dense (prepare_weight once at load time, linear per call), which "
                       f"runs the architecture dispatch inside one torch op")
        else:
            instead = (f"use fish_scales_ops.compat.{name}, the same object; the stable MoE interface is "
                       f"fish_scales_ops.moe")
        _warnings.warn(
            f"fish_scales_ops.gemm.{name} is deprecated and fish-scales-ops 0.3.0 removes "
            f"fish_scales_ops.gemm: {instead}.",
            DeprecationWarning, stacklevel=2)
        from .. import compat
        obj = getattr(compat, name)
        globals()[name] = obj  # warn once per name per process, including for `from ... import`
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _DEPRECATED)
