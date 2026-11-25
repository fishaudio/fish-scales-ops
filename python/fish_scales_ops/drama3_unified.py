"""High-level unified kernel-internal API for s3-drama (Family C: Qwen3.5-35B-A3B) on B300.

ZERO host-side `if M > X` or `if expected_m > X` kernel routing:
  - Dense projections: ONE shared CuTe-DSL Cluster-DSMEM SM100 MXFP8 kernel
    (`fish_scales_ops.gemm._unified_dense_sm100.linear_mxfp8_unified_sm100`, shared with
    Audio Tower): hardware Cluster-DSMEM S in {4,2,1} + sA SMEM overlay, in-kernel 2D L2
    swizzle, dynamic weights-first + TMA Early PDL, zero global workspace, zero atomics.
  - MoE block (routed-only OR routed + fused Shared Expert): ONE unified 8-CTA cluster
    routing kernel, ONE unified scatter-quantize kernel (warp-uniform padded skip +
    shared quantize), ONE unified 128x128x128 persistent FC1+FC2 kernel
    (`linear_mxfp8_unified_moe`), and ONE unified streaming combine kernel.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import torch

from .dense import quantize_activation as _quantize_activation_dense
from .gemm.mxfp8 import (
    quantize_1x32_fp8,
    quantize_moe_weights_1x32_fp8,
)


@dataclasses.dataclass
class UnifiedMoeExperts:
    w13_fp8: torch.Tensor
    sw13: torch.Tensor
    w2_fp8: torch.Tensor
    sw2: torch.Tensor
    num_routed: int
    has_shared: bool
    inter: int
    hidden: int


def prepare_moe_unified(
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    shared_w13: Optional[torch.Tensor] = None,
    shared_w2: Optional[torch.Tensor] = None,
) -> UnifiedMoeExperts:
    """Offline preparation of routed experts (+ optional fused Shared Expert as expert E)."""
    if (shared_w13 is None) != (shared_w2 is None):
        raise ValueError("prepare_moe_unified: shared_w13 and shared_w2 must be provided together or both omitted")
    num_routed = int(w13.shape[0])
    has_shared = shared_w13 is not None
    if has_shared:
        w13_all = torch.cat([w13, shared_w13.unsqueeze(0)], dim=0).contiguous()
        w2_all = torch.cat([w2, shared_w2.unsqueeze(0)], dim=0).contiguous()
    else:
        w13_all = w13.contiguous()
        w2_all = w2.contiguous()
    w13_fp8, sw13 = quantize_moe_weights_1x32_fp8(w13_all, w13_interleave=True)
    w2_fp8, sw2 = quantize_moe_weights_1x32_fp8(w2_all, w13_interleave=False)
    inter = int(w2_all.shape[2])
    hidden = int(w2_all.shape[1])
    return UnifiedMoeExperts(
        w13_fp8=w13_fp8,
        sw13=sw13,
        w2_fp8=w2_fp8,
        sw2=sw2,
        num_routed=num_routed,
        has_shared=has_shared,
        inter=inter,
        hidden=hidden,
    )


def prepare_dense_weight_unified(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Offline MXFP8 1x32 weight quantization in SM1xx atom layout.

    Ends with a 1-word plain (non-PDL) fill on sm_100/sm_103 (matching
    `fish_scales_ops.dense._complete_handle`) so immediately following the call
    with a weights-first PDL `linear_unified` on the same stream cannot race.
    """
    w_fp8, sw = quantize_1x32_fp8(w_bf16.contiguous())
    if w_fp8.is_cuda and torch.cuda.get_device_capability(w_fp8.device)[0] == 10:
        torch.zeros(1, dtype=torch.int32, device=w_fp8.device)
    return w_fp8, sw


def linear_unified(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    sw: torch.Tensor,
    *,
    x_scales: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    act: int = 0,
) -> torch.Tensor:
    """Unified SM100 dense MXFP8 linear — ONE shared CuTe-DSL Cluster-DSMEM kernel across ALL M."""
    from .gemm._unified_dense_sm100 import linear_mxfp8_unified_sm100

    if x.dtype == torch.bfloat16:
        # Use the fast SM100 PDL activation quantizer (matches fso.dense.linear's exact quantizer!).
        x_fp8, sx = _quantize_activation_dense(x)
    else:
        if x_scales is None:
            raise ValueError("linear_unified: x_scales is required when x is pre-quantized FP8")
        x_fp8 = x
        sx = x_scales
    return linear_mxfp8_unified_sm100(
        x_fp8, w_fp8, sx, sw, bias=bias, residual=residual, act=int(act)
    )


def moe_layer_unified(
    hidden: torch.Tensor,
    experts: UnifiedMoeExperts,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    *,
    shared_gate: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    bias_scale: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unified SM100 MoE layer (routed-only OR routed + fused Shared Expert).

    Executes the EXACT SAME 4 kernels for ALL M = 1..8192+, padded CUDA Graph
    buckets, and skewed routing — ZERO host-side `if M > X` / `if expected_m > X` dispatch:
      1. moe_build_routing_compact_unified (8-CTA cluster, Shared Expert blocks first)
      2. quantize_1x32_grouped_gather_unified_out (unified scatter quantize, padded skip)
      3. linear_mxfp8_unified_moe (ONE persistent 128x128x128 FC1+FC2 kernel)
      4. moe_combine_unified (high-bandwidth streaming combine + fused shared gate)
    """
    ops = torch.ops.fish_scales_ops
    m = int(hidden.shape[0])
    topk = int(topk_ids.shape[1])
    num_routed = experts.num_routed
    has_shared = experts.has_shared
    g = num_routed + (1 if has_shared else 0)
    hsz = experts.hidden
    inter = experts.inter
    dev = hidden.device
    fp8 = torch.float8_e4m3fn

    pairs_eff = m * topk + (m if has_shared else 0)
    num_blocks = max(1, (pairs_eff + 127 * min(g, pairs_eff)) // 128)
    r = 128 * (num_blocks + 1)

    hq = torch.empty(1, r, hsz, device=dev, dtype=fp8)
    sh = torch.empty(1, r * (hsz // 128), device=dev, dtype=torch.int32)
    dq = torch.empty(1, r, inter, device=dev, dtype=fp8)
    sd = torch.empty(1, r * (inter // 128), device=dev, dtype=torch.int32)
    dn = torch.empty(1, r, hsz, device=dev, dtype=torch.bfloat16)

    masked_m, blk0, slot_of_flat, slot_sched, block_list = (
        ops.moe_build_routing_compact_unified(topk_ids, num_routed, num_blocks, has_shared)
    )

    shared_id = num_routed if has_shared else -1
    ops.quantize_1x32_grouped_gather_unified_out(
        hidden, slot_of_flat, topk, blk0, shared_id, True, hq, sh
    )

    ops.linear_mxfp8_unified_moe(
        hq,
        experts.w13_fp8,
        sh,
        experts.sw13,
        experts.w2_fp8,
        experts.sw2,
        masked_m,
        blk0,
        slot_sched,
        dq,
        sd,
        dn,
        block_list,
    )

    return ops.moe_combine_unified(
        dn, slot_of_flat, topk_w, blk0, shared_id, shared_gate, bias, bias_scale, out
    )
