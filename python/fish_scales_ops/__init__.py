"""fish-scales-ops — block-scaled FP8 / MXFP8 dense GEMM, MoE layer and Flash
Attention for sm_90, sm_100/sm_103 and sm_120/sm_121.

Three stable namespaces, no top-level re-exports. Always go through the
namespace so domain ops never collide:

    import fish_scales_ops as fso

    # Dense linear: one surface for every architecture; the arch dispatch is
    # inside the torch op torch.ops.fish_scales_ops.dense_linear.
    weight = fso.dense.prepare_weight(w, format="bsfp8", scale=w_scale)   # load time
    y = fso.dense.linear(x, weight)                                        # per call
    fso.dense.supported("mxfp8")   # capability query for this device

    # MoE layer: one surface for every architecture; the arch dispatch is
    # inside the torch op torch.ops.fish_scales_ops.moe_layer.
    experts = fso.moe.prepare_experts(w13, w2, format="bsfp8", sw13=sw13, sw2=sw2)
    out = fso.moe.layer(hidden, experts, topk_ids, topk_w)
    fso.moe.supported("mxfp8")     # capability query for this device

    # Attention (forward-only): torch SDPA on every arch, and the sm_120/121
    # MXFP8 kernels over pre-quantized inputs
    o = fso.attention.flash_attn_fwd(q, k, v, causal=True)
    o = fso.attention.mxfp8_fwd(q_fp8, q_sc, k_fp8, k_sc, v_fp8, v_sc, causal=True)

``fso.compat`` keeps the explicit, format-specific dense ops (``linear_fp8``,
``linear_qx``, ``linear_mxfp8``, their quantizers and scale repacks,
``linear_bf16``) and the MoE per-step pieces under the names existing callers
use. ``fso.gemm`` is the deprecated old path of the same names, removed in
fish-scales-ops 0.3.0. See ``docs/api/`` for the per-domain contracts and
``docs/perf/`` for frozen reference numbers.
"""
from __future__ import annotations

import torch  # noqa: F401  CUDA torch required at import

# Single C++ extension. Registers torch.ops.fish_scales_ops.* (the GEMM, MoE
# and attention ops); fso.dense and fso.moe register one Python custom op each.
from . import _C  # noqa: F401
from . import attention, compat, dense, gemm, moe

__all__ = ["dense", "moe", "attention", "compat"]

__version__ = "0.2.0"  # keep in step with python/pyproject.toml
