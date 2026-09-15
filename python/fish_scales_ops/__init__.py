"""fish-scales-ops — FP8 block-scaled GEMM + Flash Attention for sm_90 / sm_120.

Two subpackages, no top-level re-exports. Always go through the
namespace so domain ops never collide:

    import fish_scales_ops as fso

    # GEMM (FP8 1×128 act + 128×128 wgt)
    wq, sw = fso.gemm.quantize_128x128_fp8(w_bf16)
    xq, sx = fso.gemm.quantize_1x128_fp8(x_bf16, use_ue8m0=True)
    y = fso.gemm.linear_fp8(xq, wq, sx, sw)

    # Attention (forward-only)
    o = fso.attention.flash_attn_fwd(q, k, v, causal=True)

See ``docs/api/`` for the per-domain contracts and ``docs/perf/`` for
frozen reference numbers.
"""
from __future__ import annotations

import torch  # noqa: F401  CUDA torch required at import

# Single C++ extension. Registers torch.ops.fish_scales_ops.* (GEMM ops)
# and exposes attention pybind11 helpers (probe_device_caps, etc.).
from . import _C  # noqa: F401
from . import attention, gemm

__all__ = ["gemm", "attention"]
