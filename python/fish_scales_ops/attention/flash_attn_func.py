"""Forward-only flash attention dispatch.

Every dtype routes through ``torch.nn.functional.scaled_dot_product_attention``
(FP8 inputs are dequantised to BF16 first). For the SM120 MXFP8 fast
path, call ``backends.sm120_mxfp8.mxfp8_fwd(...)`` or
``backends.sm120_mxfp8_decode.{mxfp8_decode_paged_fwd, plan_decode_paged}``
directly — both wrap ``torch.ops.fish_scales_ops.mxfp8_*`` and expect
pre-quantized Q/K/V plus UE8M0 scales (see the backend module docstrings
for layout contracts).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class FlashAttnDispatch:
    kernel_id: str
    sm_version: int
    backend: str   # "cuda_ext" | "torch_sdpa"


_DEVICE_CAPS_CACHE: dict = {}


def _device_caps(device_index: int) -> dict:
    cached = _DEVICE_CAPS_CACHE.get(device_index)
    if cached is not None:
        return cached
    major, minor = torch.cuda.get_device_capability(device_index)
    sm = major * 10 + minor
    caps = {
        "sm_major": major, "sm_minor": minor, "sm_version": sm,
        "has_tma": sm >= 90, "has_fp8": sm >= 89,
        "has_mxfp8": sm >= 120, "has_cudnn_mxfp8_sdpa": False,
    }
    _DEVICE_CAPS_CACHE[device_index] = caps
    return caps


def _is_fp8(t: torch.Tensor) -> bool:
    return t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


# Pure-Python kernel selection — does not call into the C++ extension to
# avoid an unnecessary cudaGetDeviceProperties on the hot path.
def _select_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   causal: bool, local_window: bool) -> str:
    sm = _device_caps(q.device.index or 0)["sm_version"]

    # FP8 + sm_120 routes to the in-tree MXFP8 kernel via torch.ops. The caller
    # is expected to have pre-quantized Q/K/V plus their UE8M0 scales — this
    # path only sees FP8 element tensors at the public API surface when the
    # caller already controls quantization. The pre-quantized path is invoked
    # directly through `torch.ops.fish_scales_ops.mxfp8_attn_fwd` (see
    # `backends/sm120_mxfp8.py:mxfp8_fwd`), so `flash_attn_fwd` itself only
    # accepts BF16/FP16/torch-quantized FP8 tensors and falls back to SDPA on
    # dequantised inputs.

    # Decode short-circuit: torch SDPA has a dedicated decode path that beats
    # every prefill-tuned kernel we wrap.
    if q.size(1) == 1:
        return "torch_fallback_fwd"

    # Everything else through torch SDPA. FP8-pre-quantized callers should
    # bypass this function and call the backend op directly.
    return "torch_fallback_fwd"


def _torch_sdpa_fwd(q, k, v, softmax_scale, causal, window_left, window_right):
    # SDPA needs a dtype it understands. Dequant FP8 → BF16 first.
    if _is_fp8(q):
        q = q.to(torch.bfloat16)
    if _is_fp8(k):
        k = k.to(torch.bfloat16)
    if _is_fp8(v):
        v = v.to(torch.bfloat16)

    # GQA expansion: SDPA doesn't natively broadcast K/V heads.
    h_q, h_kv = q.size(2), k.size(2)
    if h_kv != h_q:
        rep = h_q // h_kv
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
    # q/k/v come in [B,S,H,D]; SDPA wants [B,H,S,D].
    qH = q.transpose(1, 2)
    kH = k.transpose(1, 2)
    vH = v.transpose(1, 2)
    if window_left >= 0 or window_right >= 0:
        sq, sk = q.size(1), k.size(1)
        device = q.device
        ii = torch.arange(sq, device=device).view(-1, 1)
        jj = torch.arange(sk, device=device).view(1, -1)
        mask = torch.zeros(sq, sk, device=device, dtype=torch.bool)
        if window_left >= 0:
            mask |= (jj < ii - window_left)
        if window_right >= 0:
            mask |= (jj > ii + window_right)
        if causal:
            mask |= (jj > ii)
        attn_mask = torch.zeros(sq, sk, device=device, dtype=q.dtype)
        attn_mask.masked_fill_(mask, float("-inf"))
        out = torch.nn.functional.scaled_dot_product_attention(
            qH, kH, vH, attn_mask=attn_mask, scale=softmax_scale)
    else:
        out = torch.nn.functional.scaled_dot_product_attention(
            qH, kH, vH, is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2).contiguous()


def flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_left: int = -1,
    window_right: int = -1,
    return_dispatch: bool = False,
    force_kernel: Optional[str] = None,
) -> torch.Tensor | Tuple[torch.Tensor, FlashAttnDispatch]:
    """Forward-only flash attention — convenience dispatch through torch SDPA.

    q/k/v: [B, S, H, D] CUDA tensors. Every dtype routes through
    ``torch.nn.functional.scaled_dot_product_attention``; FP8 inputs are
    dequantised to BF16 first. For the SM120 MXFP8 fast path call the
    pre-quantized backend ops directly:

      * Prefill: ``fso.attention.backends.sm120_mxfp8.mxfp8_fwd(...)``
      * Decode:  ``fso.attention.backends.sm120_mxfp8_decode.mxfp8_decode_paged_fwd(...)``
                 or the ``plan_decode_paged`` / ``DecodePagedPlan.run`` pair.

    Both backends wrap ``torch.ops.fish_scales_ops.mxfp8_*`` and expect
    UE8M0-scaled FP8 inputs that this function does not produce.

    ``force_kernel="torch_fallback_fwd"`` is accepted as a no-op (the
    only kernel id this function honours).
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "inputs must be CUDA"
    assert q.dim() == k.dim() == v.dim() == 4, "q/k/v must be 4D [B,S,H,D]"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.size(-1))

    local_window = (window_left >= 0) or (window_right >= 0)
    chosen = force_kernel or _select_kernel(q, k, v, causal, local_window)

    sm_version = _device_caps(q.device.index or 0)["sm_version"]

    # Future expansion: add a BF16→FP8 in-tree quantize + mxfp8_attn_fwd
    # path here once the scale-derivation kernels land. For now everything
    # routes through torch SDPA.
    chosen = "torch_fallback_fwd"

    out = _torch_sdpa_fwd(q, k, v, softmax_scale, causal,
                           window_left, window_right)
    backend = "torch_sdpa"

    if return_dispatch:
        return out, FlashAttnDispatch(kernel_id=chosen,
                                       sm_version=sm_version,
                                       backend=backend)
    return out
