#!/usr/bin/env python3
"""The sm_120 fused-combine FC2 (linear_mxfp8_grouped_masked_combine) against the
two kernels it replaces.

The FC2 normally stores a bf16 [G, m_cap, HIDDEN] slab of per-pair rows and a
separate kernel sums each token's topk of them, weighted, into the layer output.
This op adds each row into the token's row directly with eight-byte atomic
reductions, so the slab and the combine launch both disappear. Because the adds are
atomic, the accumulation order is whatever order the CTAs finish in: the result is
NOT bit-reproducible, so every check here is a tolerance against the deterministic
path, and one case measures the run-to-run spread instead of asserting equality.

What is checked:
  1. against moe_combine on the same GEMM output: cosine and a bound on the
     per-element deviation, at several shapes;
  2. the accumulate contract: a pre-filled output (a shared expert's rows) comes
     out added to, not overwritten;
  3. skipped pairs and padded rows contribute nothing, so a token with no routed
     entry keeps exactly what the caller put in the buffer;
  4. run-to-run spread, reported;
  5. the layer entry's gate: `fused_combine=True` engages only where the slab would
     spill the L2, and the layer agrees with the deterministic layer to tolerance
     there and bit-exactly below the gate (where it must not engage at all).
"""
import sys

import torch

import fish_scales_ops as fso

CASES = [
    # (G, M, topk, HIDDEN, INTER)
    (8, 64, 2, 2048, 512),
    (32, 512, 8, 2048, 512),
    (128, 4096, 8, 2048, 512),
]


def cos(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        where = "no CUDA device" if not torch.cuda.is_available() \
            else f"device is sm_{torch.cuda.get_device_capability()[0]}x"
        print(f"SKIP: test_mxfp8_fused_combine_sm120 is sm_120/121 (RTX 5090) only ({where})")
        return 0
    dev = "cuda"
    failures = []
    ops = torch.ops.fish_scales_ops
    for G, M, topk, H, I in CASES:
        g = torch.Generator(device=dev).manual_seed(G * 977 + M)
        hidden = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
        ids = torch.rand(M, G, device=dev, generator=g).topk(topk, dim=1).indices.to(torch.int32)
        wts = torch.softmax(torch.rand(M, topk, device=dev, generator=g), dim=1).float()
        w2 = (torch.randn(G, H, I, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        w2f, sw2 = fso.gemm.quantize_moe_weights_1x32_fp8(w2)
        del w2
        m_cap = (M + 3) // 4 * 4
        em = max(1, (M * topk + G - 1) // G)
        masked, row_map, slot, _s, _p, wslot = ops.moe_build_routing(ids, G, m_cap, False, [], wts)
        # A stand-in FC2 activation operand in the masked layout.
        gu = (torch.randn(G, m_cap, 2 * I, device=dev, generator=g) * 0.2).to(torch.bfloat16)
        dq, sd = fso.gemm.silu_chunk_mul_quantize_1x32_grouped_fp8(gu, slot)
        del gu
        dn = fso.gemm.linear_mxfp8_grouped_masked(dq, w2f, sd, sw2, masked, em)
        ref = fso.gemm.moe_combine(dn, slot, wts)
        del dn
        out = torch.zeros(M, H, device=dev, dtype=torch.bfloat16)
        got = fso.gemm.linear_mxfp8_grouped_masked_combine(
            dq, w2f, sd, sw2, masked, row_map, wslot, out, em)
        torch.cuda.synchronize()
        c = cos(got, ref)
        scale = float(ref.float().abs().max())
        dev_max = float((got.float() - ref.float()).abs().max()) / max(scale, 1e-9)
        if c < 0.9999 or dev_max > 5e-2:
            failures.append(f"G={G} M={M}: fused combine cos={c:.6f} maxdev={dev_max:.2e} vs moe_combine")
        # 2: accumulate into a pre-filled buffer.
        pre = (torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.1)
        out2 = pre.clone()
        fso.gemm.linear_mxfp8_grouped_masked_combine(
            dq, w2f, sd, sw2, masked, row_map, wslot, out2, em)
        torch.cuda.synchronize()
        c2 = cos(out2, (ref.float() + pre.float()).to(torch.bfloat16))
        if c2 < 0.9999:
            failures.append(f"G={G} M={M}: accumulate contract broken, cos={c2:.6f}")
        # 4: run-to-run spread (atomics: expected non-zero, reported not asserted).
        out3 = torch.zeros_like(out)
        fso.gemm.linear_mxfp8_grouped_masked_combine(
            dq, w2f, sd, sw2, masked, row_map, wslot, out3, em)
        torch.cuda.synchronize()
        spread = float((out3.float() - got.float()).abs().max()) / max(scale, 1e-9)
        print(f"  G={G:4d} M={M:5d} topk={topk} I={I}: vs moe_combine cos={c:.6f} maxdev={dev_max:.2e}; "
              f"accumulate cos={c2:.6f}; run-to-run spread {spread:.2e}  "
              f"{'OK' if c >= 0.9999 and dev_max <= 5e-2 and c2 >= 0.9999 else 'FAIL'}")
        del dq, sd, out, out2, out3, ref, got
        torch.cuda.empty_cache()

    # 3: skipped pairs and padded rows.
    G, M, topk, H, I = 32, 512, 8, 2048, 512
    g = torch.Generator(device=dev).manual_seed(31)
    hidden = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
    ids = torch.rand(M, G, device=dev, generator=g).topk(topk, dim=1).indices.to(torch.int32)
    ids[M // 2:] = G          # the padded-row sentinel
    wts = torch.softmax(torch.rand(M, topk, device=dev, generator=g), dim=1).float()
    w13 = (torch.randn(G, 2 * I, H, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(G, H, I, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    w13f, sw13 = fso.gemm.quantize_moe_weights_1x32_fp8(w13)
    w2f, sw2 = fso.gemm.quantize_moe_weights_1x32_fp8(w2)
    del w13, w2
    m_cap = (M + 3) // 4 * 4
    em = max(1, (M * topk + G - 1) // G)
    masked, row_map, slot, _s, _p, wslot = ops.moe_build_routing(ids, G, m_cap, False, [], wts)
    hq, sh = fso.gemm.quantize_1x32_grouped_gather_fp8(hidden, slot, topk, G, m_cap)
    gu = fso.gemm.linear_mxfp8_grouped_masked(hq, w13f, sh, sw13, masked, em)
    dq, sd = fso.gemm.silu_chunk_mul_quantize_1x32_grouped_fp8(gu, slot)
    sentinel = torch.full((M, H), 0.25, device=dev, dtype=torch.bfloat16)
    out = sentinel.clone()
    fso.gemm.linear_mxfp8_grouped_masked_combine(dq, w2f, sd, sw2, masked, row_map, wslot, out, em)
    torch.cuda.synchronize()
    if not torch.equal(out[M // 2:], sentinel[M // 2:]):
        failures.append("rows whose every routed entry was skipped were written")
    print(f"  skipped rows: {M - M // 2} of {M} tokens had every entry skipped and kept the caller's buffer "
          f"untouched  {'OK' if torch.equal(out[M // 2:], sentinel[M // 2:]) else 'FAIL'}")

    # 5: the layer gate.
    for M_gate, expect in ((512, False), (4096, True)):
        engages = fso.gemm.moe_layer_fused_combine_engages_sm120(M_gate, 8, 2048)
        if engages != expect:
            failures.append(f"gate at M={M_gate}: engages={engages}, expected {expect}")
    g = torch.Generator(device=dev).manual_seed(77)
    for M_l in (512, 4096):
        hid = torch.randn(M_l, H, device=dev, dtype=torch.bfloat16, generator=g)
        idl = torch.rand(M_l, G, device=dev, generator=g).topk(topk, dim=1).indices.to(torch.int32)
        wl = torch.softmax(torch.rand(M_l, topk, device=dev, generator=g), dim=1).float()
        det = fso.gemm.moe_layer_mxfp8_sm120(hid, w13f, sw13, w2f, sw2, idl, wl)
        fus = fso.gemm.moe_layer_mxfp8_sm120(hid, w13f, sw13, w2f, sw2, idl, wl, fused_combine=True)
        torch.cuda.synchronize()
        engaged = fso.gemm.moe_layer_fused_combine_engages_sm120(M_l, topk, H)
        c = cos(det, fus)
        same = torch.equal(det, fus)
        ok = (same if not engaged else (c > 0.9999 and not same))
        if not ok:
            failures.append(f"layer M={M_l}: engaged={engaged} identical={same} cos={c:.6f}")
        print(f"  layer M={M_l:5d}: gate {'on' if engaged else 'off'}, "
              f"{'bit-identical to the deterministic layer' if same else f'cos={c:.6f} against it'}  "
              f"{'OK' if ok else 'FAIL'}")
        del hid, idl, wl, det, fus
        torch.cuda.empty_cache()

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nsm_120 fused-combine FC2: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
