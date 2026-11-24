"""Blackwell (sm_100 / sm_103) Audio Tower fused operators and pipeline-optimal dispatch.

Provides end-to-end fused operators for realtime ASR Audio Tower (32-layer Transformer Encoder,
d_model=1280, n_heads=20, head_dim=64, d_ff=5120, exact erf-GELU, 3x Conv2d Stem):

  - layernorm_mxfp8:         Fused PDL LayerNorm -> MXFP8 1x32 (SM100 atom layout, zero padded scales)
  - attn_fa4_mxfp8:          FlashAttention-4 SM100 fused Attention -> MXFP8 1x32 + Early PDL + single-wave q_stage=1
  - gelu_nhwc_to_mxfp8:      Fused PDL Conv3 NHWC exact erf-GELU + (F, C) permute + MXFP8 1x32 quantize
  - gelu_erf_bf16_pdl:       Packed f32x2 exact erf-GELU on BF16 with PDL
  - prepare_conv_out_weight: Static load-time (C, F) -> (F, C) column reorder for NHWC stem fusion
  - linear_audio_dense / linear_audio_conv_out / linear_audio_qkv / linear_audio_out_proj /
    linear_audio_fc2 / linear_audio_proj2:
                             Pipeline-aware MXFP8 GEMM dispatch for pre-quantized Audio Tower roles
                             using per-call wf_cap=3/5 and TMA-warp early PDL (never mutating global
                             fso.dense defaults).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .. import dense as _dense
from ..gemm import _epilogue as _ep
from ..gemm import _sm100_decode as _dec
from ..gemm import _sm100_dispatch as _disp
from .fa4_mxfp8 import attn_fa4_mxfp8

# Audio Tower all-prequantized GEMMs follow long (3-15 us) early-PDL producers
# (LN_MXFP8, FA4_MXFP8, fc1_q), so deeper weights-first prefetch (3 stages for
# 1-CTA split-K, 5 stages for 2-CTA persistent) beats the 2/1 global default
# (which protects LLM bf16-x quantize->GEMM). Passed per-call via wf_cap.
_AUDIO_WF_CAP_ONE_CTA = 3
_AUDIO_WF_CAP_TWO_CTA = 5
_AUDIO_DEFAULT_EARLY_PDL = True


def layernorm_mxfp8(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
    round_bf16: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused PDL LayerNorm -> MXFP8 1x32 returning (xq [M, D] fp8, sx [align(M,128)*(D/128)] int32)."""
    return torch.ops.fish_scales_ops.layernorm_mxfp8_sm100(x, gamma, beta, float(eps), bool(round_bf16), 1)


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
    cfg_override=None,
    early_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Dispatch a pre-quantized Audio Tower MXFP8 GEMM with per-call wf_cap=3/5 and early PDL."""
    m, k = int(xq.shape[0]), int(xq.shape[1])
    n = int(dw.weight.shape[0])
    epi = _ep.Epilogue(bias=bias, residual=residual) if (bias is not None or residual is not None) else None
    cfg = cfg_override
    if cfg is None:
        for key, cand_cfg in _disp._candidates(m, n, k, epi):
            if key == _disp.DECODE:
                cfg = cand_cfg
            break
    if cfg is not None and _dec.enabled() and (epi is None or _dec.epilogue_available()):
        cluster = cfg[1]
        wf_cap = _AUDIO_WF_CAP_TWO_CTA if cluster[0] * cluster[1] > 1 else _AUDIO_WF_CAP_ONE_CTA
        epdl = _AUDIO_DEFAULT_EARLY_PDL if early_pdl is None else bool(early_pdl)
        y = _dec.linear_mxfp8_decode(
            xq, dw.weight, xs, dw.scale, cfg, epi, wf_cap=wf_cap, early_pdl=epdl
        )
        if y is not None:
            return y
    return _dense.linear(xq, dw, x_scale=xs, bias=bias, residual=residual)


def linear_audio_conv_out(
    xq: torch.Tensor,
    dw_conv_out: _dense.DenseWeight,
    xs: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    early_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Pipeline-optimal MXFP8 conv_out [1280x7680] (+ optional positional embedding residual)."""
    return linear_audio_dense(xq, dw_conv_out, xs, residual=residual, early_pdl=early_pdl)


def linear_audio_qkv(
    hq: torch.Tensor,
    dw_qkv: _dense.DenseWeight,
    hs: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    early_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Pipeline-optimal MXFP8 QKV [3840x1280] dispatch between early-PDL LN1 and PDL-waiting FA4.

    Across all 6 ASR serving rungs (M=78..1079), routing QKV to weights-first + early-PDL CuTe-DSL
    persistent tactics keeps the 32-layer PDL chain unbroken (avoiding non-PDL cublasLt at M>=364
    and non-weights-first C++ cascade at M<=182), saving -8.3 to -45.5 us across 32 layers:
      - M <= 128:      128x64 (1-CTA, wf_cap=3)
      - 128 < M <= 256: 256x64 (2-CTA, wf_cap=5)
      - 256 < M <= 384: 256x128 (2-CTA, wf_cap=5; 90 CTAs = 1 wave)
      - 384 < M <= 576: 256x64 (2-CTA, wf_cap=5; avoids 150-CTA 2-CTA spill on 148 SMs)
      - M > 576:       256x128 (2-CTA, wf_cap=5)
    """
    m = int(hq.shape[0])
    if m <= 128:
        cfg = ((128, 64), (1, 1), True, 1)
    elif m <= 256:
        cfg = ((256, 64), (2, 1), True, 1)
    elif m <= 384:
        cfg = ((256, 128), (2, 1), True, 1)
    elif m <= 576:
        cfg = ((256, 64), (2, 1), True, 1)
    else:
        cfg = ((256, 128), (2, 1), True, 1)
    return linear_audio_dense(hq, dw_qkv, hs, bias=bias, cfg_override=cfg, early_pdl=early_pdl)


def linear_audio_out_proj(
    aq: torch.Tensor,
    dw_o: _dense.DenseWeight,
    as_: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    early_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Pipeline-optimal MXFP8 out_proj [1280x1280] dispatch behind early-PDL FA4_MXFP8."""
    m = int(aq.shape[0])
    cfg = ((256, 64), (2, 1), True, 1) if 384 < m <= 640 else None
    return linear_audio_dense(aq, dw_o, as_, bias=bias, residual=residual, cfg_override=cfg, early_pdl=early_pdl)


def linear_audio_fc2(
    uq: torch.Tensor,
    dw_fc2: _dense.DenseWeight,
    us: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    early_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Pipeline-optimal MXFP8 fc2 [1280x5120] dispatch with per-call wf_cap=3/5."""
    return linear_audio_dense(uq, dw_fc2, us, bias=bias, residual=residual, early_pdl=early_pdl)


def linear_audio_proj2(
    xq: torch.Tensor,
    dw_proj2: _dense.DenseWeight,
    xs: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pipeline-optimal MXFP8 tail proj2 [2048x1280] dispatch with per-call wf_cap=3/5."""
    return linear_audio_dense(xq, dw_proj2, xs, bias=bias, early_pdl=False)


__all__ = [
    "layernorm_mxfp8",
    "attn_fa4_mxfp8",
    "gelu_nhwc_to_mxfp8",
    "gelu_erf_bf16_pdl",
    "prepare_conv_out_weight",
    "linear_audio_dense",
    "linear_audio_conv_out",
    "linear_audio_qkv",
    "linear_audio_out_proj",
    "linear_audio_fc2",
    "linear_audio_proj2",
]
