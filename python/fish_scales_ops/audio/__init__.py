"""Blackwell (sm_100 / sm_103) Audio Tower fused operators with in-kernel adaptive routing.

Provides end-to-end fused operators for realtime ASR Audio Tower (32-layer Transformer Encoder,
d_model=1280, n_heads=20, head_dim=64, d_ff=5120, exact erf-GELU, 3x Conv2d Stem) while keeping
routing decisions INSIDE the GPU kernels (zero host-side `if M <= ...` kernel tables on the
default unified path):

  - layernorm_mxfp8:         Single unified PDL CUDA kernel (`ln_mxfp8_unified_kernel`):
                             in-kernel D==1280 register-resident fast path + general D%128==0 path,
                             SM100 atom layout, zero padded scales. Zero host D/M/variant routing.
  - attn_fa4_mxfp8:          FlashAttention-4 SM100 fused Attention -> MXFP8 1x32 + Early PDL.
  - gelu_nhwc_to_mxfp8:      Single PDL CUDA kernel: Conv3 NHWC exact erf-GELU + (F, C) permute -> MXFP8.
  - gelu_erf_bf16_pdl:       Single PDL CUDA kernel: packed f32x2 exact erf-GELU on BF16.
  - prepare_conv_out_weight: Static load-time (C, F) -> (F, C) column reorder for NHWC stem fusion.
  - linear_mxfp8_unified_sm100 / linear_audio_dense / linear_audio_conv_out / linear_audio_qkv /
    linear_audio_out_proj / linear_audio_fc2 / linear_audio_proj2:
                             Shared CuTe-DSL Cluster-DSMEM SM100 MXFP8 Dense GEMM (`cluster=(1,1,4)`,
                             zero global workspace, zero atomics, zero host shape routing!):
                             dynamically selects `S in {4, 2, 1}` Cluster-DSMEM Split-K vs Data-Parallel,
                             weights-first prefetch depth, and adaptive rasterization INSIDE the GPU kernel.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .. import dense as _dense

_AUDIO_DEFAULT_EARLY_PDL = True
_UNIFIED_DENSE_FN = None
_FA4_MXFP8_FN = None


def _get_unified_dense_fn():
    global _UNIFIED_DENSE_FN
    if _UNIFIED_DENSE_FN is None:
        try:
            from ..gemm._unified_dense_sm100 import linear_mxfp8_unified_sm100 as _fn
        except ImportError as exc:
            raise ImportError(
                "fish_scales_ops.audio unified SM100 dense GEMM requires the optional sm100 dependency "
                "(nvidia-cutlass-dsl)"
            ) from exc
        _UNIFIED_DENSE_FN = _fn
    return _UNIFIED_DENSE_FN


def _get_fa4_mxfp8_fn():
    global _FA4_MXFP8_FN
    if _FA4_MXFP8_FN is None:
        try:
            from .fa4_mxfp8 import attn_fa4_mxfp8 as _fn
        except ImportError as exc:
            raise ImportError(
                "fish_scales_ops.audio.attn_fa4_mxfp8 requires optional sm100 FA4 dependencies "
                "(nvidia-cutlass-dsl, flash_attn.cute, quack)"
            ) from exc
        _FA4_MXFP8_FN = _fn
    return _FA4_MXFP8_FN


def linear_mxfp8_unified_sm100(*args, **kwargs):
    """Lazy wrapper around fish_scales_ops.gemm._unified_dense_sm100.linear_mxfp8_unified_sm100."""
    return _get_unified_dense_fn()(*args, **kwargs)


def attn_fa4_mxfp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    out_q: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lazy wrapper around SM100 FlashAttention-4 fused Attention -> MXFP8 1x32."""
    return _get_fa4_mxfp8_fn()(
        q, k, v, softmax_scale=softmax_scale, out_q=out_q, out_scale=out_scale
    )


def layernorm_mxfp8(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
    round_bf16: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single unified PDL LayerNorm -> MXFP8 1x32 returning (xq [M, D] fp8, sx int32)."""
    return torch.ops.fish_scales_ops.layernorm_mxfp8_sm100(x, gamma, beta, float(eps), bool(round_bf16))


def gelu_nhwc_to_mxfp8(x_nhwc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused PDL exact erf-GELU + NHWC [B, C, F, T] -> [B*T, F*C] MXFP8 1x32."""
    return torch.ops.fish_scales_ops.gelu_nhwc_to_mxfp8_sm100(x_nhwc)


def gelu_erf_bf16_pdl(x: torch.Tensor) -> torch.Tensor:
    """Fast packed-f32x2 exact erf-GELU on BF16 with PDL."""
    return torch.ops.fish_scales_ops.gelu_erf_bf16_pdl_sm100(x)


def prepare_conv_out_weight(
    w_conv_out: torch.Tensor, channels: int = 480, freq: int = 16
) -> _dense.DenseWeight:
    """Reorder conv_out input columns (C, F) -> (F, C) once at load time for zero-copy NHWC stem fusion."""
    if w_conv_out.ndim != 2:
        raise ValueError(f"Expected 2-D conv_out weight [N, C*F], got {w_conv_out.ndim}-D")
    if channels <= 0 or freq <= 0:
        raise ValueError(f"channels and freq must be positive, got channels={channels}, freq={freq}")
    n = int(w_conv_out.shape[0])
    if int(w_conv_out.shape[1]) != channels * freq:
        raise ValueError(
            f"Expected conv_out input dim {channels * freq} ({channels}*{freq}), got {w_conv_out.shape[1]}"
        )
    w_perm = w_conv_out.view(n, channels, freq).permute(0, 2, 1).reshape(n, channels * freq).contiguous()
    return _dense.prepare_weight(w_perm, format="mxfp8")


def linear_audio_dense(
    xq: torch.Tensor,
    dw: _dense.DenseWeight,
    xs: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    early_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Run the single unified in-kernel adaptive Cluster-DSMEM SM100 MXFP8 GEMM (zero host shape routing)."""
    epdl = _AUDIO_DEFAULT_EARLY_PDL if early_pdl is None else bool(early_pdl)
    return linear_mxfp8_unified_sm100(
        xq, dw.weight, xs, dw.scale, bias=bias, residual=residual, early_pdl=epdl
    )


def linear_audio_conv_out(
    xq: torch.Tensor,
    dw_conv_out: _dense.DenseWeight,
    xs: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    early_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Unified in-kernel MXFP8 conv_out [1280x7680] (+ optional positional embedding residual)."""
    return linear_audio_dense(xq, dw_conv_out, xs, residual=residual, early_pdl=early_pdl)


def linear_audio_qkv(
    hq: torch.Tensor,
    dw_qkv: _dense.DenseWeight,
    hs: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    early_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Unified in-kernel MXFP8 QKV [3840x1280] between early-PDL LN1 and PDL-waiting FA4."""
    return linear_audio_dense(hq, dw_qkv, hs, bias=bias, early_pdl=early_pdl)


def linear_audio_out_proj(
    aq: torch.Tensor,
    dw_o: _dense.DenseWeight,
    as_: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    early_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Unified in-kernel MXFP8 out_proj [1280x1280] behind early-PDL FA4_MXFP8."""
    return linear_audio_dense(aq, dw_o, as_, bias=bias, residual=residual, early_pdl=early_pdl)


def linear_audio_fc2(
    uq: torch.Tensor,
    dw_fc2: _dense.DenseWeight,
    us: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    early_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Unified in-kernel MXFP8 fc2 [1280x5120] (in-kernel S=4/2/1 Cluster-DSMEM Split-K)."""
    return linear_audio_dense(uq, dw_fc2, us, bias=bias, residual=residual, early_pdl=early_pdl)


def linear_audio_proj2(
    xq: torch.Tensor,
    dw_proj2: _dense.DenseWeight,
    xs: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unified in-kernel MXFP8 tail proj2 [2048x1280]."""
    return linear_audio_dense(xq, dw_proj2, xs, bias=bias, early_pdl=False)


__all__ = [
    "layernorm_mxfp8",
    "attn_fa4_mxfp8",
    "gelu_nhwc_to_mxfp8",
    "gelu_erf_bf16_pdl",
    "prepare_conv_out_weight",
    "linear_mxfp8_unified_sm100",
    "linear_audio_dense",
    "linear_audio_conv_out",
    "linear_audio_qkv",
    "linear_audio_out_proj",
    "linear_audio_fc2",
    "linear_audio_proj2",
]
