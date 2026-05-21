"""Forward-only flash attention.

Public surface:
  * :func:`flash_attn_fwd` — convenience dispatch that routes every dtype
    through ``torch.nn.functional.scaled_dot_product_attention`` (FP8
    inputs are dequantised to BF16 first).
  * :class:`FlashAttnDispatch` — kernel-selector dataclass.
  * ``backends.sm120_mxfp8.mxfp8_fwd`` — production SM120 MXFP8 forward
    over pre-quantized Q/K/V + UE8M0 scales (wraps
    ``torch.ops.fish_scales_ops.mxfp8_attn_fwd``).
  * ``backends.sm120_mxfp8_decode.{mxfp8_decode_paged_fwd, plan_decode_paged}`` —
    single-token paged-KV decode (wraps
    ``torch.ops.fish_scales_ops.mxfp8_decode_paged``).
"""
from .flash_attn_func import FlashAttnDispatch, flash_attn_fwd  # noqa: F401

__all__ = ["flash_attn_fwd", "FlashAttnDispatch"]
