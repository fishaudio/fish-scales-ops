#!/usr/bin/env python3
"""The sm_120 fused-SwiGLU FC1 (linear_mxfp8_grouped_masked_swiglu) against the
two kernels it replaces.

The MoE layer's FC1 writes a bf16 [G, m_cap, 2*INTER] gate/up slab and a separate
kernel reads it back to compute silu(gate) * up and requantize to MXFP8. The fused
epilogue does that on the tile while it is still in shared memory, so the slab
never exists. The arithmetic is meant to be the separate kernel's instruction for
instruction -- the same bf16x2 silu through tanh.approx, the same amax over the
same 32 columns, the same UE8M0 derivation and the same paired satfinite converts
-- so this test demands bit equality of both outputs (the FP8 rows and the packed
scale words) at every row the masked layout declares valid, not a tolerance.

Padding rows are excluded on purpose: neither path writes them (the separate
kernel iterates the routed-pair space, the fused one predicates on masked_m), so
their contents are undefined by contract.
"""
import sys

import torch

import fish_scales_ops as fso

CASES = [
    # (G, M, topk, HIDDEN, INTER)
    (8, 4, 2, 2048, 512),
    (8, 64, 2, 2048, 512),
    (16, 128, 4, 2048, 512),
    (8, 64, 2, 2048, 768),
    (32, 256, 8, 2048, 512),
]


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        where = "no CUDA device" if not torch.cuda.is_available() \
            else f"device is sm_{torch.cuda.get_device_capability()[0]}x"
        print(f"SKIP: test_mxfp8_fused_fc1_sm120 is sm_120/121 (RTX 5090) only ({where})")
        return 0
    dev = "cuda"
    failures = []
    for G, M, topk, H, I in CASES:
        g = torch.Generator(device=dev).manual_seed(G * 1000 + M)
        hidden = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
        ids = torch.rand(M, G, device=dev, generator=g).topk(topk, dim=1).indices.to(torch.int32)
        w13 = (torch.randn(G, 2 * I, H, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        # Interleaved rows: the fused epilogue pairs gate_j with up_j inside one
        # thread's accumulator, which only holds when the weight rows alternate.
        w13i, sw13i = fso.gemm.quantize_moe_weights_1x32_fp8(w13, w13_interleave=True)
        avail = fso.gemm.mxfp8_grouped_swiglu_available(2 * I, H)
        if not avail:
            failures.append(f"G={G} I={I}: mxfp8_grouped_swiglu_available said no on sm_120")
            continue
        m_cap = (M + 3) // 4 * 4
        rows = M * topk
        em = max(1, (rows + G - 1) // G)
        masked, _rm, slot, *_ = fso.gemm.moe_build_routing(ids, G, m_cap)
        a_fp8, sa = fso.gemm.quantize_1x32_grouped_gather_fp8(hidden, slot, topk, G, m_cap)
        # The pair it replaces, on the same interleaved weights.
        gu = fso.gemm.linear_mxfp8_grouped_masked(a_fp8, w13i, sa, sw13i, masked, em)
        dq_ref, sd_ref = fso.gemm.silu_chunk_mul_quantize_1x32_grouped_fp8(gu, slot, pairwise=True)
        # The fused FC1.
        dq, sd = fso.gemm.linear_mxfp8_grouped_masked_swiglu(a_fp8, w13i, sa, sw13i, masked, em)
        torch.cuda.synchronize()
        if dq.shape != dq_ref.shape or sd.numel() != sd_ref.numel():
            failures.append(f"G={G} M={M} I={I}: shapes differ, {tuple(dq.shape)} vs {tuple(dq_ref.shape)}")
            continue
        counts = masked.tolist()
        kp = I // 128
        bad_rows = 0
        bad_words = 0
        a = dq.view(torch.uint8)
        b = dq_ref.view(torch.uint8)
        sa_w = sd.view(G, kp, m_cap)
        sb_w = sd_ref.view(G, kp, m_cap)
        for gi, n in enumerate(counts):
            if n == 0:
                continue
            if not torch.equal(a[gi, :n], b[gi, :n]):
                bad_rows += int((a[gi, :n] != b[gi, :n]).any(dim=-1).sum())
            if not torch.equal(sa_w[gi, :, :n], sb_w[gi, :, :n]):
                bad_words += int((sa_w[gi, :, :n] != sb_w[gi, :, :n]).sum())
        if bad_rows or bad_words:
            failures.append(f"G={G} M={M} I={I}: {bad_rows} rows and {bad_words} scale words differ")
        print(f"  G={G:3d} M={M:4d} topk={topk} I={I:4d} m_cap={m_cap:4d} em={em:3d}: "
              f"{sum(1 for c in counts if c)} active groups, {sum(counts)} rows "
              f"bit-identical to FC1 + SwiGLU-quant  {'OK' if not (bad_rows or bad_words) else 'FAIL'}")

    # The whole layer on interleaved weights (fused route) against the same layer
    # on stacked weights (unfused): the same values by construction, so the
    # outputs must agree to the bf16 rounding of one requantize.
    G, M, topk, H, I = 32, 256, 8, 2048, 512
    g = torch.Generator(device=dev).manual_seed(4242)
    hidden = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
    ids = torch.rand(M, G, device=dev, generator=g).topk(topk, dim=1).indices.to(torch.int32)
    wts = torch.softmax(torch.rand(M, topk, device=dev, generator=g), dim=1).float()
    w13 = (torch.randn(G, 2 * I, H, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(G, H, I, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    w2f, sw2 = fso.gemm.quantize_moe_weights_1x32_fp8(w2)
    w13p, sw13p = fso.gemm.quantize_moe_weights_1x32_fp8(w13)
    w13i, sw13i = fso.gemm.quantize_moe_weights_1x32_fp8(w13, w13_interleave=True)
    plain = fso.gemm.moe_layer_mxfp8_sm120(hidden, w13p, sw13p, w2f, sw2, ids, wts)
    fused = fso.gemm.moe_layer_mxfp8_sm120(
        hidden, w13i, sw13i, w2f, sw2, ids, wts, w13_interleaved=True)
    torch.cuda.synchronize()
    pf = plain.float().flatten()
    ff = fused.float().flatten()
    cos = float((pf @ ff) / (pf.norm() * ff.norm() + 1e-30))
    same = torch.equal(plain, fused)
    if cos < 0.99999:
        failures.append(f"layer: fused route cos={cos:.6f} against the stacked-weight layer")
    print(f"  layer G={G} M={M}: fused route vs stacked route cos={cos:.6f}"
          f"{' (bit-identical)' if same else ''}  OK")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nsm_120 fused-SwiGLU FC1: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
