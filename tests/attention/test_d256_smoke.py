"""D=256 contiguous prefill correctness vs SDPA.

Covers the dedicated D=256 dispatch path in mxfp8_attn_fwd.cu (Bc=32, 1 CTA/SM).
The D=256 paged-prefill / decode paths are exercised by
``test_mxfp8_paged_prefill.py`` and ``test_mxfp8_page_size_sweep.py`` via
the parametrized D=[32, 64, 128, 256] sweeps.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F


def _is_sm120() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 12


@pytest.mark.skipif(not _is_sm120(), reason="D=256 path requires Blackwell sm_120a")
@pytest.mark.parametrize("causal", [False, True])
def test_contiguous_prefill_d256(causal):
    from fish_scales_ops.attention.backends import sm120_mxfp8 as bk

    torch.manual_seed(0xD256 ^ int(causal))
    B, S, H_q, H_kv, D = 1, 256, 8, 2, 256
    q = torch.randn(B, S, H_q, D, dtype=torch.bfloat16, device="cuda") * 0.5
    k = torch.randn(B, S, H_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
    v = torch.randn(B, S, H_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5

    q_fp8, q_sc = bk.pre_quantize_q(q)
    k_fp8, k_sc = bk.pre_quantize_k(k)
    v_fp8, v_sc = bk.pre_quantize_v(v)

    out = bk.mxfp8_fwd(q_fp8, q_sc, k_fp8, k_sc, v_fp8, v_sc, causal=causal)
    assert out.shape == (B, S, H_q, D) and out.dtype == torch.bfloat16

    k_g = k.repeat_interleave(H_q // H_kv, dim=2)
    v_g = v.repeat_interleave(H_q // H_kv, dim=2)
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2), k_g.transpose(1, 2), v_g.transpose(1, 2),
        is_causal=causal, scale=1.0 / math.sqrt(D)).transpose(1, 2)
    cos = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
    assert cos > 0.98, f"D=256 causal={causal} cos={cos:.6f}"
