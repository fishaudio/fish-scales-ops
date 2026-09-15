"""sm_120 and sm_100/sm_103 block-FP8 (1x128 / 128x128): K % 512 != 0 shapes and exactness vs the dequantised inputs.

Until 2026-09-05 the sm_120 path required K % 512 == 0: the int32-packed UE8M0
scale words hold 4 K-blocks each and the kernel's K loop was padded to the
4-stage ring. The kernel now stops at the real k-tile count and the repack
zero-pads the tail word, so K only needs to be a multiple of 128 (K=768 = 6
blocks -> 2 words, the second half-filled). This test drives every host path
that produces packed scales and checks them against each other and against a
BF16 reference.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import fish_scales_ops as fso

COS_MIN = 0.95   # weak BF16 sanity; the strong check is cos >= 0.9995 vs the dequantized reference


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


def run(M: int, N: int, K: int) -> None:
    torch.manual_seed(M * 1009 + N * 17 + K)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.5)
    ref = F.linear(x, w)

    # path 1: FP32 UE8M0 scales, packed inside linear_fp8 (per-call repack)
    xq, sx = fso.gemm.quantize_1x128_fp8(x, use_ue8m0=True)
    wq, sw = fso.gemm.quantize_128x128_fp8(w)
    assert sx.shape == (((M + 3) // 4) * 4, K // 128), sx.shape
    y1 = fso.gemm.linear_fp8(xq, wq, sx, sw)

    # path 2: explicit pre-pack ops
    sx_p = fso.gemm.repack_fp8_act_scales(sx)
    sw_p = fso.gemm.repack_fp8_wgt_scales(sw)
    sm100 = torch.cuda.get_device_capability(0)[0] == 10
    words = (K // 128 + 3) // 4
    if sm100:
        # Sm1xx atom layout: one replicated word per (row, K-block), rows padded to 128.
        assert sx_p.numel() == ((M + 127) // 128) * 128 * (K // 128) and sw_p.numel() == N * (K // 128), (sx_p.shape, sw_p.shape)
        w0 = int(sx_p.reshape(-1)[0].item()) & 0xFFFFFFFF
        assert w0 == (w0 & 0xFF) * 0x01010101, "atom word must hold the same UE8M0 byte in all 4 slots"
    else:
        assert sx_p.shape == (sx.shape[0], words) and sw_p.shape[1] == words, (sx_p.shape, sw_p.shape)
    if (K // 128) % 4 and not sm100:
        # tail bytes of the last word must be zero. The packed tensor's metadata
        # is row-major [M_pad, words] but the physical layout is K-major (word
        # (m, kp) at kp * M_pad + m), so take the last word through the flat view.
        m_pad = sx_p.shape[0]
        tail = sx_p.reshape(-1)[(words - 1) * m_pad:words * m_pad].to(torch.int64) & 0xFFFFFFFF
        used = (K // 128) % 4
        assert int((tail >> (8 * used)).max()) == 0, "tail bytes of the last packed word are not zero"
    y2 = fso.gemm.linear_fp8(xq, wq, sx_p, sw_p)

    # path 3: fused quantize + pack (falls back to two-step when K % 512 != 0)
    xq3, sx3 = fso.gemm.quantize_1x128_fp8_packed(x)
    assert torch.equal(xq3, xq), "quantize_1x128_fp8_packed fp8 payload differs from quantize_1x128_fp8"
    assert torch.equal(sx3, sx_p), "quantize_1x128_fp8_packed scales differ from repack_fp8_act_scales"
    y3 = fso.gemm.linear_fp8(xq3, wq, sx3, sw_p)

    assert torch.isfinite(y1).all() and torch.isfinite(y2).all() and torch.isfinite(y3).all(), "NaN/Inf in output"
    assert torch.equal(y1, y2) and torch.equal(y2, y3), "the three scale paths must produce identical outputs"

    # dequantized reference: the same FP8 payload and UE8M0 scales evaluated in FP32 by torch.
    # sx is stored K-major (word (m, kb) at kb * M_pad + m) behind row-major [M_pad, K/128] metadata;
    # sw is row-major [N/128, K/128] (one scale per 128x128 block).
    kb = K // 128
    m_pad = sx.shape[0]
    sx_l = sx.reshape(-1)[: kb * m_pad].view(kb, m_pad).t()[:M]              # [M, K/128] logical
    x_deq = xq.float() * sx_l.repeat_interleave(128, dim=1)
    w_deq = wq.float() * sw.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:N, :K]
    ref_deq = x_deq @ w_deq.t()
    c_deq = cos_sim(y1, ref_deq)
    assert c_deq >= 0.9995, f"M={M} N={N} K={K}: cos vs dequantized reference {c_deq:.6f} — kernel/scale-path error"
    c = cos_sim(y1, ref)
    assert c >= COS_MIN, f"M={M} N={N} K={K}: cos={c:.6f} < {COS_MIN}"
    print(f"  fp8 1x128  M={M:5d} N={N:5d} K={K:5d}  words/row={K // 128 if sm100 else words}  cos={c:.6f}  cos_vs_dequant={c_deq:.6f}  OK")


def test_fp32_weight_scales_rejected() -> None:
    """Non-power-of-two (plain amax/448) weight scales cannot be represented as UE8M0 exponent bytes;
    the explicit pre-pack op must refuse them instead of truncating silently (the 2026-09-05 bug)."""
    w = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
    _, sw_fp32 = fso.gemm.quantize_128x128_fp8(w, use_ue8m0=False)
    try:
        fso.gemm.repack_fp8_wgt_scales(sw_fp32)
    except RuntimeError as e:
        assert "UE8M0" in str(e), str(e)
        print("  repack_fp8_wgt_scales rejects FP32 (non-UE8M0) weight scales  OK")
        return
    raise AssertionError("repack_fp8_wgt_scales accepted non-power-of-two FP32 scales")


def main() -> None:
    cap = torch.cuda.get_device_capability(0)
    print(f"Device: {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
    if cap[0] not in (10, 12):
        print("sm_100 / sm_120 test; skipping on this device")
        return
    test_fp32_weight_scales_rejected()
    for M, N, K in (
        (8, 256, 384),        # 3 blocks: one word, 3 of 4 bytes used
        (64, 2048, 768),      # Family B moe_inter as a dense K: 6 blocks -> 2 words
        (1024, 2048, 768),
        (4096, 2048, 768),
        (128, 4096, 1280),    # 10 blocks -> 3 words
        (256, 2048, 512),     # control: whole span
        (512, 2048, 2048),    # control: whole spans
    ):
        run(M, N, K)
    print("All sm_120 K%128 block-FP8 tests passed.")


if __name__ == "__main__":
    main()
