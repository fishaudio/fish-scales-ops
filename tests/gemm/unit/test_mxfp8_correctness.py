"""MXFP8 (1×32 UE8M0) correctness test — quantize round-trip + GEMM vs BF16.

Runs on sm_120/121 (Sm120BlockScaledKernel) and sm_100/sm_103 (tcgen05
BlockScaled). Two layers of checking:

1. **Quantize round-trip**: decode the packed int32 UE8M0 scale buffer with
   a pure-Python reference of the arch-native layout (sm_120 K-major vs
   sm_100 Sm1xxBlockScaledConfig atom), dequantize x_fp8 and compare against
   the BF16 input. This catches scale-layout addressing bugs directly —
   a wrong SF index shows up here even when the GEMM cos looks plausible.
2. **GEMM**: linear_mxfp8 vs F.linear BF16 reference, cos ≥ 0.999 per cell
   (repo-typical MXFP8 cos is ≥ 0.9993 on the Qwen3 grid).

Also covers the fused silu_chunk_mul_quantize_1x32_fp8 prologue.
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

import fish_scales_ops as fso

COS_GEMM = 0.999
COS_ROUNDTRIP = 0.999


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.double().flatten(), b.double().flatten(), dim=0).item()


def _decode_scales(packed: torch.Tensor, M: int, K: int, sm_major: int) -> torch.Tensor:
    """Decode the opaque packed UE8M0 buffer into FP32 scales [M, K/32].

    Pure-Python mirror of the C++ `sf_word_index<SF_LAYOUT>` mapping:
      sm_120: word (m, kp) at int32 index kp * pad(M,4) + m
      sm_100: word (m, kp) at ((m//128) * (K//128) + kp) * 128
                              + (m % 32) * 4 + (m % 128) // 32
    Each int32 packs 4 UE8M0 bytes = the 4 consecutive k-blocks kp*4..kp*4+3.
    """
    num_kp = K // 128
    if sm_major == 10:
        words = packed.view(-1).cpu()  # opaque 1-D buffer
        m_idx = torch.arange(M)
        idx = torch.empty(M, num_kp, dtype=torch.long)
        for kp in range(num_kp):
            idx[:, kp] = ((m_idx // 128) * num_kp + kp) * 128 \
                + (m_idx % 32) * 4 + (m_idx % 128) // 32
        sel = words[idx.view(-1)].view(M, num_kp)
    else:
        # K-major [pad(M,4), K/128] with strides (1, M_pad). packed.t() is
        # [K/128, M_pad] row-major, so .contiguous().view(-1) reproduces the
        # RAW word order (index = kp * M_pad + m) without repacking bytes.
        M_pad = (M + 3) // 4 * 4
        words = packed.t().contiguous().view(-1).cpu()
        m_idx = torch.arange(M)
        idx = torch.empty(M, num_kp, dtype=torch.long)
        for kp in range(num_kp):
            idx[:, kp] = kp * M_pad + m_idx
        sel = words[idx.view(-1)].view(M, num_kp)
    # Unpack 4 UE8M0 bytes per word (LSB-first: byte b = k-block kp*4+b) →
    # scale 2^(byte-127).
    sel = sel.to(torch.int64) & 0xFFFFFFFF
    out = torch.empty(M, num_kp * 4, dtype=torch.float64)
    for kp in range(num_kp):
        w = sel[:, kp]
        for b in range(4):
            byte = (w >> (8 * b)) & 0xFF
            out[:, kp * 4 + b] = torch.pow(2.0, byte.double() - 127.0)
    return out  # [M, K/32]


def test_quantize_roundtrip(M: int, K: int, sm_major: int) -> None:
    torch.manual_seed(M * 1009 + K)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    xq, sx = fso.gemm.quantize_1x32_fp8(x)
    torch.cuda.synchronize()

    scales = _decode_scales(sx, M, K, sm_major)  # [M, K/32]
    deq = xq.float().cpu().double() * scales.repeat_interleave(32, dim=1)
    c = _cos(deq.float(), x.float().cpu())
    assert c >= COS_ROUNDTRIP, \
        f"quantize round-trip M={M} K={K}: cos={c:.6f} < {COS_ROUNDTRIP} (SF layout addressing bug?)"
    print(f"  roundtrip  M={M:>5} K={K:>5}  cos={c:.6f}  OK")


def test_gemm(M: int, N: int, K: int) -> None:
    torch.manual_seed(M * 1009 + N * 17 + K)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    w = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.5))
    xq, sx = fso.gemm.quantize_1x32_fp8(x)
    wq, sw = fso.gemm.quantize_1x32_fp8(w)
    y = fso.gemm.linear_mxfp8(xq, wq, sx, sw)
    y_ref = F.linear(x, w)
    c = _cos(y, y_ref)
    assert c >= COS_GEMM, f"linear_mxfp8 M={M} N={N} K={K}: cos={c:.6f} < {COS_GEMM}"
    assert torch.isfinite(y).all(), f"linear_mxfp8 M={M} N={N} K={K}: NaN/Inf in output"
    print(f"  gemm       M={M:>5} N={N:>5} K={K:>5}  cos={c:.6f}  OK")


def test_silu_fused(M: int, inter: int, K: int) -> None:
    """gate_up output [M, 2*inter] → fused silu·mul quantize → down GEMM."""
    torch.manual_seed(M * 31 + inter)
    gu = torch.randn(M, 2 * inter, dtype=torch.bfloat16, device="cuda") * 0.5
    w = (torch.randn(K, inter, dtype=torch.bfloat16, device="cuda") / (inter ** 0.5))
    hq, sh = fso.gemm.silu_chunk_mul_quantize_1x32_fp8(gu)
    wq, sw = fso.gemm.quantize_1x32_fp8(w)
    y = fso.gemm.linear_mxfp8(hq, wq, sh, sw)
    gate, up = gu.split(inter, dim=-1)
    h_ref = F.silu(gate.float()) * up.float()
    y_ref = F.linear(h_ref.bfloat16(), w)
    c = _cos(y, y_ref)
    assert c >= COS_GEMM, f"silu+mxfp8 M={M} inter={inter}: cos={c:.6f} < {COS_GEMM}"
    print(f"  silu+gemm  M={M:>5} I={inter:>5} N={K:>5}  cos={c:.6f}  OK")


def main() -> int:
    assert torch.cuda.is_available(), "CUDA required"
    sm = torch.cuda.get_device_capability(0)
    print(f"Device: {torch.cuda.get_device_name(0)} sm_{sm[0]}{sm[1]}\n")
    if sm[0] not in (10, 12):
        print("MXFP8 needs sm_100/103/120/121 — skipping.")
        return 0

    # M values chosen to cross padding boundaries on BOTH layouts:
    # 1/2 (decode), 4 (sm_120 pad unit), 96/100 (< 128, exercises sm_100
    # 128-row padding), 128/129 (block boundary), 512.
    print("== quantize round-trip (SF layout decode) ==")
    for M in (1, 2, 4, 96, 100, 128, 129, 512):
        for K in (2560, 9728):
            test_quantize_roundtrip(M, K, sm[0])

    print("\n== linear_mxfp8 vs BF16 (Qwen3-4B grid + boundary M) ==")
    qwen = [(6144, 2560), (2560, 4096), (19456, 2560), (9728, 2560), (2560, 9728)]
    for M in (1, 16, 100, 128, 129, 512, 1024):
        for N, K in qwen:
            test_gemm(M, N, K)

    print("\n== fused silu_chunk_mul + linear_mxfp8 ==")
    for M in (1, 100, 512):
        test_silu_fused(M, 9728, 2560)

    print("\nAll MXFP8 correctness tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
