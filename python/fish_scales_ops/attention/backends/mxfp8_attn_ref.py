"""Phase 7 reference: MXFP8 QK + mixed-PV attention forward.

Software implementation of the data path described in plan §9.1:
    Q: MXFP8
    K: MXFP8
    V: BF16 or FP8 (here BF16)
    P: BF16 (softmax stays full-precision)
    O: BF16

Used as the correctness baseline for the future hardware kernel and to
quantify the §9.3 acceptance metrics (output drift, causal correctness,
long-context drift, decode behaviour).
"""
from __future__ import annotations

import torch

from mxfp8_ref import quantize_mxfp8, dequantize_mxfp8, DEFAULT_BLOCK


def mxfp8_qk_mixed_pv_fwd(
    q_bf16: torch.Tensor,
    k_bf16: torch.Tensor,
    v_bf16: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool = False,
    block: int = DEFAULT_BLOCK,
) -> torch.Tensor:
    """Full attention forward with MXFP8 Q/K, BF16 V/P/O.

    q_bf16, k_bf16, v_bf16: [B, S, H, D] BF16 (or any float).
    """
    B, Sq, H, D = q_bf16.shape
    Sk = k_bf16.size(1)
    assert k_bf16.shape[-1] == D and v_bf16.shape[-1] == D

    # Quantize Q/K to MXFP8 along the head_dim (inner reduction axis of QK).
    # Reshape to [..., D] so quantize_mxfp8 sees D as inner.
    q_flat = q_bf16.permute(0, 2, 1, 3).contiguous().reshape(-1, D)  # [B*H*Sq, D]
    k_flat = k_bf16.permute(0, 2, 1, 3).contiguous().reshape(-1, D)  # [B*H*Sk, D]
    q_mx = quantize_mxfp8(q_flat, block=block)
    k_mx = quantize_mxfp8(k_flat, block=block)

    # Dequantize to BF16 for the matmul (the hardware kernel would do this on
    # the fly per inner block — software path is correctness-only).
    q_deq = dequantize_mxfp8(q_mx).reshape(B, H, Sq, D).to(torch.float32)
    k_deq = dequantize_mxfp8(k_mx).reshape(B, H, Sk, D).to(torch.float32)

    # S = Q * K^T  -> [B, H, Sq, Sk]
    s = torch.matmul(q_deq, k_deq.transpose(-1, -2)) * softmax_scale
    if causal:
        ii = torch.arange(Sq, device=s.device).view(-1, 1)
        jj = torch.arange(Sk, device=s.device).view(1, -1)
        s = s.masked_fill(jj > ii, float("-inf"))

    # P (BF16-precision softmax) and O = P * V (V stays BF16).
    p = torch.softmax(s, dim=-1).to(torch.bfloat16)
    v = v_bf16.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    o = torch.matmul(p.float(), v.float())

    # Back to [B, Sq, H, D] BF16.
    return o.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
