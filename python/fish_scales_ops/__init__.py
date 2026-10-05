"""fish-scales-ops — block-scaled FP8 / MXFP8 dense GEMM, MoE layer and Flash
Attention for sm_90, sm_100/sm_103 and sm_120/sm_121.

Three stable namespaces and one top-level function, ``build_info()``; no
top-level re-exports. Always go through the namespace so domain ops never
collide:

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

``fso.build_info()`` says how this copy was built: the contents of the
``BUILD_INFO.json`` a wheel carries, or ``source_build: True`` for an in-place
build of the source tree.
"""
from __future__ import annotations

import copy

import torch  # noqa: F401  CUDA torch required at import

# A wheel names the torch its extension was compiled against; another torch
# fails here with ImportError instead of crashing inside the extension.
from . import _build_info

_BUILD_INFO = _build_info.read()
_build_info.check_torch(_BUILD_INFO)

# Single C++ extension. Registers torch.ops.fish_scales_ops.* (the GEMM, MoE
# and attention ops); fso.dense and fso.moe register one Python custom op each.
from . import _C  # noqa: F401,E402
from . import attention, compat, dense, gemm, moe  # noqa: E402

__all__ = ["dense", "moe", "attention", "compat", "build_info"]

__version__ = "0.2.0"  # keep in step with python/pyproject.toml


def build_info() -> dict:
    """How this copy of fish_scales_ops was built, as a new dict on every call.

    A wheel built by ``scripts/build_wheel.sh`` returns its ``BUILD_INFO.json``:
    ``version``, ``commit``, ``dirty``, ``built_at`` (UTC), ``image`` (the build
    image and the digest of its base), ``cuda_toolkit``, ``nvcc``, ``gcc``,
    ``python``, ``torch`` (the version the extension was compiled against),
    ``cutlass_commit``, ``nvrtc`` (the bundled NVRTC's version and the sha256
    of its wheel), ``arch_list`` and ``glibc_required`` (the newest
    ``GLIBC_x.y`` symbol version the extension needs). An in-place build of the
    source tree has no such file and returns ``{"source_build": True,
    "version": <package version>, "torch": <the running torch>}``.
    """
    if _BUILD_INFO is None:
        return {"source_build": True, "version": __version__, "torch": str(torch.__version__)}
    return copy.deepcopy(_BUILD_INFO)
