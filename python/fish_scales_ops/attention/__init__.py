"""Forward-only attention.

* :func:`flash_attn_fwd` — convenience attention through torch SDPA on every
  architecture (FP8 inputs are dequantized to bf16 first).
* The sm_120/sm_121 MXFP8 kernels, over pre-quantized Q/K/V with UE8M0 scales;
  on any other architecture they raise ``NotImplementedError``:

  - contiguous prefill: :func:`mxfp8_fwd`, with the calibration-time quantizers
    :func:`pre_quantize_q`, :func:`pre_quantize_k`, :func:`pre_quantize_v`;
  - paged prefill and extend: :func:`mxfp8_paged_prefill_fwd`, or
    :func:`plan_paged_prefill` / :class:`PrefillPagedPlan` for a fixed set of
    per-request Q lengths, with :func:`quantize_q_ragged`;
  - paged decode: :func:`mxfp8_decode_paged_fwd`, or :func:`plan_decode_paged` /
    :class:`DecodePagedPlan` for a decode loop or a CUDA graph, with
    :func:`quantize_q_grouped`;
  - KV calibration and a reference paged layout: :func:`compute_k_chan_scale`,
    :func:`compute_v_chan_scale`, :func:`quantize_kv_to_paged`.

``docs/api/attention.md`` holds the tensor layouts and the plan contracts.
"""
from .backends.sm120_mxfp8 import (  # noqa: F401
    mxfp8_fwd,
    pre_quantize_k,
    pre_quantize_q,
    pre_quantize_v,
)
from .backends.sm120_mxfp8_decode import (  # noqa: F401
    DecodePagedPlan,
    compute_k_chan_scale,
    compute_v_chan_scale,
    mxfp8_decode_paged_fwd,
    plan_decode_paged,
    quantize_kv_to_paged,
    quantize_q_grouped,
)
from .backends.sm120_mxfp8_paged_prefill import (  # noqa: F401
    PrefillPagedPlan,
    mxfp8_paged_prefill_fwd,
    plan_paged_prefill,
    quantize_q_ragged,
)
from .flash_attn_func import FlashAttnDispatch, flash_attn_fwd  # noqa: F401

__all__ = [
    # torch SDPA, every architecture
    "flash_attn_fwd",
    "FlashAttnDispatch",
    # sm_120/sm_121 MXFP8 contiguous prefill
    "mxfp8_fwd",
    "pre_quantize_q",
    "pre_quantize_k",
    "pre_quantize_v",
    # sm_120/sm_121 MXFP8 paged prefill / extend
    "mxfp8_paged_prefill_fwd",
    "plan_paged_prefill",
    "PrefillPagedPlan",
    "quantize_q_ragged",
    # sm_120/sm_121 MXFP8 paged decode
    "mxfp8_decode_paged_fwd",
    "plan_decode_paged",
    "DecodePagedPlan",
    "quantize_q_grouped",
    # KV calibration and the reference paged layout
    "compute_k_chan_scale",
    "compute_v_chan_scale",
    "quantize_kv_to_paged",
]
