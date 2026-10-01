"""Forward-only attention through torch SDPA, on every architecture.

:func:`flash_attn_fwd` is the convenience entry: every dtype on every device
goes through ``torch.nn.functional.scaled_dot_product_attention`` (FP8 inputs
are dequantized to bf16 first, GQA heads are expanded). The sm_120/sm_121 MXFP8
kernels take pre-quantized inputs and are separate entries of
:mod:`fish_scales_ops.attention`: ``mxfp8_fwd`` (contiguous prefill),
``mxfp8_paged_prefill_fwd`` / ``plan_paged_prefill`` (paged prefill and extend)
and ``mxfp8_decode_paged_fwd`` / ``plan_decode_paged`` (paged decode).
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
    caps = {"sm_version": major * 10 + minor}
    _DEVICE_CAPS_CACHE[device_index] = caps
    return caps


def _is_fp8(t: torch.Tensor) -> bool:
    return t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


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
) -> torch.Tensor | Tuple[torch.Tensor, FlashAttnDispatch]:
    """Forward-only attention through torch SDPA, on every architecture.

    q/k/v: [B, S, H, D] CUDA tensors, bf16, fp16 or FP8 (dequantized to bf16
    first); GQA is expanded with ``repeat_interleave``. ``window_left`` /
    ``window_right`` >= 0 add a sliding-window mask. With ``return_dispatch`` the
    call also returns ``FlashAttnDispatch(kernel_id="torch_fallback_fwd",
    sm_version, backend="torch_sdpa")``.

    For the sm_120/sm_121 MXFP8 kernels over pre-quantized inputs use
    :func:`fish_scales_ops.attention.mxfp8_fwd`,
    :func:`fish_scales_ops.attention.mxfp8_paged_prefill_fwd` or
    :func:`fish_scales_ops.attention.mxfp8_decode_paged_fwd`.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "inputs must be CUDA"
    assert q.dim() == k.dim() == v.dim() == 4, "q/k/v must be 4D [B,S,H,D]"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.size(-1))
    out = _torch_sdpa_fwd(q, k, v, softmax_scale, causal, window_left, window_right)
    if return_dispatch:
        sm_version = _device_caps(q.device.index or 0)["sm_version"]
        return out, FlashAttnDispatch(kernel_id="torch_fallback_fwd",
                                      sm_version=sm_version, backend="torch_sdpa")
    return out
