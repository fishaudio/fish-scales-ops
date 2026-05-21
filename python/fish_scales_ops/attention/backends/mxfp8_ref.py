"""MXFP8 software reference (Phase 6 microkernel).

OCP Microscaling FP8 spec, recap:
- Elements: FP8 E4M3 (4-bit exponent, 3-bit mantissa, finite/no NaN-inf split)
- Block size: typically 32 elements along the inner reduction dim
- Block scale: UE8M0 — unsigned 8-bit-exponent-only float (no mantissa, no sign)
  representable values are 2^k for k in [-127, 127], encoded by the raw byte.

Reference uses these blocks along K (the reduction axis). For QK in attention,
that means K=head_dim is split into chunks of 32; each Q row owns one scale
per chunk, and each K row likewise.

Hardware MXFP8 MMA exists on SM100 (tcgen05) and SM120 (warp.MmaSM120BlockScaledOp)
but the DSL example is SM100-only, so this module gives a correctness baseline
the future hardware kernel can be validated against.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

FP8_E4M3_MAX = 448.0   # representable max
SCALE_LOG_MIN = -127
SCALE_LOG_MAX =  127
DEFAULT_BLOCK = 32     # OCP Microscaling block size


@dataclass
class MxFp8Tensor:
    """E4M3 elements + UE8M0 block scales over the inner axis."""
    data:   torch.Tensor   # ..., D, dtype=torch.float8_e4m3fn
    scales: torch.Tensor   # ..., D // block, dtype=torch.uint8 (UE8M0 raw)
    block:  int


def _ue8m0_pack(log2_scale: torch.Tensor) -> torch.Tensor:
    """Encode log2(scale) (clamped to [-127, 127]) as UE8M0 raw byte.

    UE8M0 stores the unsigned 8-bit exponent with bias 127, so an input of 0
    encodes to byte 127 (= 2^0), input of -1 to byte 126 (= 2^-1), etc.
    """
    e = log2_scale.clamp(SCALE_LOG_MIN, SCALE_LOG_MAX).round().to(torch.int32)
    return (e + 127).to(torch.uint8)


def _ue8m0_unpack(raw: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 raw byte into a fp32 scale = 2^(byte - 127)."""
    e = raw.to(torch.int32) - 127
    return torch.pow(torch.tensor(2.0, device=raw.device, dtype=torch.float32),
                      e.to(torch.float32))


def quantize_mxfp8(x: torch.Tensor, block: int = DEFAULT_BLOCK) -> MxFp8Tensor:
    """Quantize a tensor to MXFP8.

    Inner dim of x must be a multiple of `block`. Per-block amax is encoded as
    a UE8M0 scale = 2^ceil(log2(amax / FP8_MAX)). Elements are divided by the
    scale and cast to E4M3.
    """
    assert x.shape[-1] % block == 0, f"inner dim {x.shape[-1]} not multiple of {block}"
    orig_shape = x.shape
    x = x.reshape(-1, x.shape[-1] // block, block)
    amax = x.abs().amax(dim=-1).clamp_min(1e-30)                # [..., n_blocks]
    log2_scale = torch.ceil(torch.log2(amax / FP8_E4M3_MAX))
    scale_raw = _ue8m0_pack(log2_scale)                          # uint8
    scale_f32 = _ue8m0_unpack(scale_raw)                         # fp32
    q = x.float() / scale_f32.unsqueeze(-1)
    q = q.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return MxFp8Tensor(
        data=q.reshape(orig_shape),
        scales=scale_raw.reshape(*orig_shape[:-1], orig_shape[-1] // block),
        block=block,
    )


def dequantize_mxfp8(t: MxFp8Tensor) -> torch.Tensor:
    """Decode MXFP8 back to FP32. Inverse of quantize_mxfp8."""
    block = t.block
    inner = t.data.shape[-1]
    assert inner % block == 0
    data = t.data.reshape(*t.data.shape[:-1], inner // block, block).float()
    scale = _ue8m0_unpack(t.scales).unsqueeze(-1)
    return (data * scale).reshape(*t.data.shape)


def qk_mxfp8(q_mx: MxFp8Tensor, k_mx: MxFp8Tensor) -> torch.Tensor:
    """Compute Q @ K^T in FP32 with MXFP8 inputs (software reference).

    q_mx.data: [..., Br, D] — MXFP8 along D
    k_mx.data: [..., Bc, D] — MXFP8 along D
    Returns: [..., Br, Bc] FP32.

    Uses dequantize-then-matmul. A real hardware kernel would interleave
    the per-block dequantize with the inner-product accumulation.
    """
    assert q_mx.block == k_mx.block, "Q and K must share the block size"
    assert q_mx.data.shape[-1] == k_mx.data.shape[-1], "D mismatch"
    q_f = dequantize_mxfp8(q_mx)
    k_f = dequantize_mxfp8(k_mx)
    return torch.matmul(q_f, k_f.transpose(-1, -2))
