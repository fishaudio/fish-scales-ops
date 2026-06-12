"""Minimal smoke test for the attention dispatch.

The only attention kernels compiled into `fish_scales_ops._C` are the
SM120 MXFP8 prefill (`torch.ops.fish_scales_ops.mxfp8_attn_fwd`) and
paged decode (`mxfp8_decode_paged`). The convenience
`fso.attention.flash_attn_fwd(...)` always routes through torch SDPA;
the SM120 fast path is reached via the backend modules.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import fish_scales_ops as fso


def _ref_sdpa(q, k, v, scale, causal):
    h_q, h_kv = q.size(2), k.size(2)
    if h_kv != h_q:
        rep = h_q // h_kv
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=causal, scale=scale)
    return out.transpose(1, 2).contiguous()


def _check(B, S, H_q, H_kv, D, causal):
    torch.manual_seed(B * 31 + S * 13 + H_q + D)
    q = torch.randn(B, S, H_q, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, S, H_kv, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, S, H_kv, D, dtype=torch.bfloat16, device="cuda")
    scale = D ** -0.5
    out = fso.attention.flash_attn_fwd(
        q, k, v, softmax_scale=scale, causal=causal,
        force_kernel="torch_fallback_fwd")
    ref = _ref_sdpa(q, k, v, scale, causal)
    cos = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
    print(f"  BHSD={B}x{H_q}/{H_kv}x{S}x{D}  causal={causal}  cos={cos:.6f}")
    assert cos > 0.99, f"cos={cos:.6f} below 0.99 floor"


def main():
    cap = torch.cuda.get_device_capability(0)
    print(f"Device: {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})\n")

    print("== torch_fallback_fwd (SDPA) ==")
    _check(2, 256, 8, 8, 128, causal=False)
    _check(2, 256, 8, 8, 128, causal=True)
    # GQA
    _check(1, 1024, 32, 8, 128, causal=True)

    # Verify the SM120 MXFP8 attention op is registered.
    assert hasattr(torch.ops.fish_scales_ops, "mxfp8_attn_fwd"), \
        "torch.ops.fish_scales_ops.mxfp8_attn_fwd not registered"
    print("\nflash_attn_fwd dispatch + torch.ops.fish_scales_ops.mxfp8_attn_fwd OK.")


if __name__ == "__main__":
    main()
