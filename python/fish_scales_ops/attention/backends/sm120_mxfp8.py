"""SM120 MXFP8 FlashAttention — production Python entrypoint.

The single SM120 MXFP8 kernel is `kernels/sm120/mxfp8_attn_fwd.cu`,
a CuTe-style port of the v10 grid-stride algorithm with templated
`(kHeadDim, kBr, kBc, kStages, kCtasPerSm, kIsCausal)` and runtime
`(head_dim, causal)` dispatch. Native GQA via stride-0 K/V broadcast.

The kernel is compiled into ``fish_scales_ops._C`` and exposed via
``torch.ops.fish_scales_ops.mxfp8_attn_fwd``. There is no JIT path.

CALLER CONTRACT — Q/K/V are PRE-QUANTIZED.

The kernel does no quantization. Callers (e.g. an inference engine that
keeps FP8 KV cache, or a model with FP8 weight projections) are responsible
for producing the FP8 element tensors and the UE8M0 block-scale tensors
in the layouts described below.

Layouts (block size = 32 along the K axis of each MMA, Bc per head_dim
chosen by the kernel: D=32 → Bc=128, D=64 → Bc=64, D=128 → Bc=64):

    Q_fp8     : [B, S_q, H_q, D]              torch.float8_e4m3fn
                BSHD-contiguous (D innermost).
    Q_scales  : [B, S_q // 16, H_q, D // 32]  torch.uint8        (UE8M0)
                One scale per 16-row Q tile, broadcast across all 16 rows
                and across the 32-element K-block of D.

    K_fp8     : [B, S_k, H_kv, D]             torch.float8_e4m3fn
                BSHD-contiguous (D innermost).
    K_scales  : [B, S_k // Bc, H_kv, D // 32] torch.uint8
                One scale per Bc-row K tile, per K-block of D.

    V_fp8     : [B, D, H_kv, S_k]             torch.float8_e4m3fn
                "DHS"-contiguous (S innermost). This pre-transposed layout
                is required so the FP8 PV mma B-fragment can do 16-byte
                K-contiguous loads. Callers that hold V in BSHD must
                transpose once at FP8-quantisation time.
    V_scales  : [B, S_k // Bc, H_kv, Bc // 32] torch.uint8
                One scale per Bc-row V tile, per K-block (32 elements
                along the K=sequence axis).

Constraints (Blackwell sm_120a, CUDA 12.8+):
    - sm == 12.0 (sm_120a) — host dispatcher returns cudaErrorNotSupported below.
    - head_dim D ∈ {32, 64, 128, 256}  (D=256 uses a dedicated Bc=32 path)
    - S_q % 64 == 0
    - S_k % Bc == 0  (Bc per the table above)
    - H_q % H_kv == 0  (GQA via stride-0 broadcast; H_q == H_kv = MHA)

Output:
    O      : [B, S_q, H_q, D]                torch.bfloat16  (BSHD-contiguous)

Optional helpers `pre_quantize_q`, `pre_quantize_k`, `pre_quantize_v` are
provided ONLY as test-time / calibration-time utilities. They are not part
of the kernel's runtime path.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


FP8_E4M3_MAX = 448.0


# Per-D Bc table — must match the kernel's launch-side dispatch.
# (D=32 → Bc=128 amortises per-tile launch cost when the per-CTA work is small;
#  D=64/128 → Bc=64 keeps 2-CTA/SM occupancy; D=256 → Bc=32 because smem +
#  register pressure forces 1 CTA/SM and a smaller KV tile.)
_BC_BY_D = {32: 128, 64: 64, 128: 64, 256: 32}


def mxfp8_fwd(
    q_fp8:  torch.Tensor,                                            # [B,Sq,H_q,D]
    q_scales: torch.Tensor,                                          # [B,Sq/16,H_q,D/32]
    k_fp8:  torch.Tensor,                                            # [B,Sk,H_kv,D]
    k_scales: torch.Tensor,                                          # [B,Sk/Bc,H_kv,D/32]
    v_fp8:  torch.Tensor,                                            # [B,D,H_kv,Sk]
    v_scales: torch.Tensor,                                          # [B,Sk/Bc,H_kv,Bc/32]
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward pass over PRE-QUANTISED MXFP8 Q, K, V.

    Thin wrapper over ``torch.ops.fish_scales_ops.mxfp8_attn_fwd``. The
    kernel is built into ``fish_scales_ops._C``; no JIT path.

    GQA: K and V may have fewer heads than Q. The kernel uses stride-0
    broadcast (each Q-head's group of size H_q/H_kv shares one KV head),
    so K/V do not need to be expanded by `repeat_interleave` outside;
    H_q and H_kv only need to satisfy H_q % H_kv == 0.

    The optional ``out`` argument is accepted for API symmetry but the
    underlying torch.ops binding always allocates its own output tensor.
    Pass ``out`` only if you intend to copy into it after the call.

    Returns a BF16 tensor of shape [B, S_q, H_q, D].
    """
    assert q_fp8.dtype == torch.float8_e4m3fn, "q_fp8 must be float8_e4m3fn"
    assert k_fp8.dtype == torch.float8_e4m3fn, "k_fp8 must be float8_e4m3fn"
    assert v_fp8.dtype == torch.float8_e4m3fn, "v_fp8 must be float8_e4m3fn"
    assert q_scales.dtype == torch.uint8 and k_scales.dtype == torch.uint8
    assert v_scales.dtype == torch.uint8

    B, Sq, H_q, D = q_fp8.shape
    Sk = k_fp8.size(1)
    H_kv = k_fp8.size(2)
    assert k_fp8.shape == (B, Sk, H_kv, D), "K must be [B,Sk,H_kv,D]"
    assert v_fp8.shape == (B, D, H_kv, Sk), "V must be [B,D,H_kv,Sk] (pre-transposed)"
    assert H_q % H_kv == 0, f"H_q ({H_q}) must be a multiple of H_kv ({H_kv}) for GQA"
    if D not in _BC_BY_D:
        raise RuntimeError(
            f"SM120 MXFP8 kernel supports head_dim ∈ {sorted(_BC_BY_D)} (got D={D})")
    bc = _BC_BY_D[D]
    assert Sq % 64 == 0 and Sk % bc == 0, \
        f"D={D} requires Sq%64==0 and Sk%{bc}==0 (got Sq={Sq}, Sk={Sk})"

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    o = torch.ops.fish_scales_ops.mxfp8_attn_fwd(
        q_fp8.contiguous(), q_scales.contiguous(),
        k_fp8.contiguous(), k_scales.contiguous(),
        v_fp8.contiguous(), v_scales.contiguous(),
        float(softmax_scale), bool(causal))

    if out is not None:
        out.copy_(o)
        return out
    return o


# -----------------------------------------------------------------------------
# Test-time / calibration-time helpers — NOT part of the runtime kernel path.
# Production callers should produce the FP8 + scale tensors however their
# data pipeline already does (e.g. fused with the linear projections).
# -----------------------------------------------------------------------------

def _ue8m0_pack(log2_scale: torch.Tensor) -> torch.Tensor:
    e = log2_scale.clamp(-127, 127).round().to(torch.int32)
    return (e + 127).to(torch.uint8)


def pre_quantize_q(q_bf16: torch.Tensor, q_tile_rows: int = 16, block: int = 32):
    """[B, Sq, H, D] BF16 -> (Q_fp8 BSHD, Q_scales [B,Sq/16,H,D/32]).

    One UE8M0 scale per (q_tile, h, k_block); broadcast across all 16 rows
    and across the 32-element D-block.
    """
    B, S, H, D = q_bf16.shape
    n_tiles, n_kblk = S // q_tile_rows, D // block
    fp8 = torch.empty_like(q_bf16, dtype=torch.float8_e4m3fn)
    sc  = torch.empty(B, n_tiles, H, n_kblk, dtype=torch.uint8, device=q_bf16.device)
    for b in range(B):
        for h in range(H):
            for qt in range(n_tiles):
                tile = q_bf16[b, qt*q_tile_rows:(qt+1)*q_tile_rows, h, :].float()
                t3 = tile.reshape(q_tile_rows, n_kblk, block)
                amax = t3.abs().amax(dim=(0, 2)).clamp_min(1e-30)         # [n_kblk]
                log2_s = torch.ceil(torch.log2(amax / FP8_E4M3_MAX))
                sb = _ue8m0_pack(log2_s)
                sf = torch.pow(torch.tensor(2.0, device=q_bf16.device),
                                log2_s.to(torch.float32))
                q  = (t3 / sf.view(1, -1, 1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX
                        ).to(torch.float8_e4m3fn).reshape(q_tile_rows, D)
                fp8[b, qt*q_tile_rows:(qt+1)*q_tile_rows, h, :] = q
                sc[b, qt, h, :] = sb
    return fp8, sc


def pre_quantize_k(k_bf16: torch.Tensor, kv_tile_rows: Optional[int] = None,
                   block: int = 32):
    """[B, Sk, H, D] BF16 -> (K_fp8 BSHD, K_scales [B,Sk/Bc,H,D/32]).

    ``kv_tile_rows`` defaults to ``_BC_BY_D[D]`` (the per-head-dim Bc the
    kernel dispatches to). Pass an explicit value only if you intentionally
    want a different scale granularity (uncommon).
    """
    if kv_tile_rows is None:
        D = k_bf16.size(-1)
        if D not in _BC_BY_D:
            raise ValueError(f"no default kv_tile_rows for D={D}; pass explicitly")
        kv_tile_rows = _BC_BY_D[D]
    return pre_quantize_q(k_bf16, q_tile_rows=kv_tile_rows, block=block)


def pre_quantize_v(v_bf16: torch.Tensor, kv_tile_rows: Optional[int] = None,
                   block: int = 32):
    """[B, Sk, H, D] BF16 -> (V_fp8 [B,D,H,Sk] DHS, V_scales [B,Sk/Bc,H,Bc/32]).

    ``kv_tile_rows`` defaults to ``_BC_BY_D[D]`` (per-head-dim Bc).

    Note the layout transpose: V is stored DHS-contiguous (S innermost) so
    the FP8 PV mma B fragment reads K-contiguous bytes.
    """
    B, S, H, D = v_bf16.shape
    if kv_tile_rows is None:
        if D not in _BC_BY_D:
            raise ValueError(f"no default kv_tile_rows for D={D}; pass explicitly")
        kv_tile_rows = _BC_BY_D[D]
    n_tiles, n_kblk = S // kv_tile_rows, kv_tile_rows // block
    V_bdhs = v_bf16.permute(0, 3, 2, 1).contiguous()                    # [B,D,H,S]
    V_t6   = V_bdhs.view(B, D, H, n_tiles, n_kblk, block)
    amax = V_t6.float().abs().amax(dim=(1, 5)).clamp_min(1e-30)          # [B,H,n_tiles,n_kblk]
    log2_s = torch.ceil(torch.log2(amax / FP8_E4M3_MAX))
    sb = _ue8m0_pack(log2_s)                                             # [B,H,n_tiles,n_kblk]
    sf = torch.pow(torch.tensor(2.0, device=v_bf16.device),
                    log2_s.to(torch.float32))
    sc = sf.unsqueeze(1).unsqueeze(-1)                                   # [B,1,H,n_t,n_k,1]
    V_q = (V_t6.float() / sc).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX
            ).to(torch.float8_e4m3fn)
    V_t = V_q.reshape(B, D, H, S).contiguous()
    Vs  = sb.permute(0, 2, 1, 3).contiguous()                            # [B,n_t,H,n_k]
    return V_t, Vs

