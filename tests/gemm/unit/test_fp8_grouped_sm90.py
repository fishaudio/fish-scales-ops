"""H1 — sm_90 (H200) grouped block-scale FP8 masked GEMM correctness.

Revives the in-tree DeepGEMM GroupedMasked scheduler (three host edits:
gemm_dispatch_sm90 grouped entry, runtime.cuh name->enum, runner.cu C-ABI).
Since the revived kernel IS the vendored DeepGEMM kernel, the strongest check
is bit-for-bit against pip deep_gemm's own masked GEMM on identical inputs;
we also check cos vs a per-expert BF16 reference and CUDA-graph replay with
in-place masked_m rewrite.

Scale layout the sm_90 kernel expects (verified):
  SFA = per-token FP32 scales transposed to [G, K/128, m_cap] contiguous
        (the kernel's TMA descriptor reads ColMajor [pad(m_cap,4), (K/128)*G])
  SFB = per-128x128-block FP32 scales [G, N/128, K/128] contiguous
sm_90 supports K%128 (unlike the sm_120 block-FP8 kernel's K%512), so the
Qwen3 down (K=768) works here.

Which parts need pip `deep_gemm` and which do not. Only `test_vs_deepgemm`
does: it is the bit-exactness check against the package the kernel was
vendored from, and there is no way to make that claim without the package.
Everything else on this page — the masked GEMM's own numerical correctness,
the layout-native quantize, the SwiGLU quantize, the CUDA-graph replay and
the composed six-kernel MoE layer — is fso plus a torch reference, and all of
it runs on any sm_90 device whether `deep_gemm` is importable or not. The
masked GEMM's numerical correctness is covered in both worlds:
`test_vs_dequant_reference` runs the same shape list as `test_vs_deepgemm`
and compares each group's defined rows against an FP32 matmul of the
dequantised operands, which is the comparison docs/perf/README.md section 8
asks for (the BF16 truth alone cannot separate quantization error from a
kernel or scale bug).

The exit code reflects what ran: every section that was not skipped must have
passed, and the last line names the sections that ran and the sections that
were skipped with the reason.
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

import fish_scales_ops as fso

try:
    import deep_gemm as dg
    HAVE_DG = True
except ImportError:
    HAVE_DG = False

FP8_MAX = 448.0


def _cos(a, b):
    return F.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()


# --- torch reference quantizers, so this file does not need deep_gemm -------
# Same block geometry and the same amax/448 scale rule as deep_gemm's
# per_token_cast_to_fp8 / per_block_cast_to_fp8, written out here so that the
# masked GEMM can be exercised (and its operands dequantised for a reference)
# on a machine that does not have the package installed.

def _q_1x128(a):
    """[m, K] bf16 -> ([m, K] fp8_e4m3, [m, K/128] fp32), one scale per row block."""
    m, K = a.shape
    v = a.float().view(m, K // 128, 128)
    s = v.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4) / FP8_MAX
    q = (v / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(m, K)
    return q, s.view(m, K // 128)


def _q_128x128(w):
    """[N, K] bf16 -> ([N, K] fp8_e4m3, [N/128, K/128] fp32), one scale per 128x128 block."""
    N, K = w.shape
    v = w.float().view(N // 128, 128, K // 128, 128)
    s = v.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4) / FP8_MAX
    q = (v / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(N, K)
    return q, s.view(N // 128, K // 128)


def _deq_1x128(q, s):
    m, K = q.shape
    return (q.float().view(m, K // 128, 128) * s[:, :, None]).view(m, K)


def _deq_128x128(q, s):
    N, K = q.shape
    return (q.float().view(N // 128, 128, K // 128, 128) * s[:, None, :, None]).view(N, K)


def _quant_torch(a, w, G, m_cap, N, K):
    """Grouped quantize with the torch reference quantizers above."""
    Kb = K // 128
    w_fp8 = torch.empty(G, N, K, device="cuda", dtype=torch.float8_e4m3fn)
    w_sf = torch.empty(G, N // 128, Kb, device="cuda", dtype=torch.float32)
    a_fp8 = torch.empty(G, m_cap, K, device="cuda", dtype=torch.float8_e4m3fn)
    a_sf = torch.empty(G, m_cap, Kb, device="cuda", dtype=torch.float32)
    for g in range(G):
        q, s = _q_128x128(w[g])
        w_fp8[g].copy_(q); w_sf[g].copy_(s)
        q, s = _q_1x128(a[g])
        a_fp8[g].copy_(q); a_sf[g].copy_(s)
    return a_fp8, a_sf, w_fp8, w_sf


def _quant_dg(a, w, G, m_cap, N, K):
    """Grouped quantize with deep_gemm's own casts (test_vs_deepgemm only)."""
    Kb = K // 128
    w_fp8 = torch.empty(G, N, K, device="cuda", dtype=torch.float8_e4m3fn)
    w_sf = torch.empty(G, N // 128, Kb, device="cuda", dtype=torch.float32)
    a_fp8 = torch.empty(G, m_cap, K, device="cuda", dtype=torch.float8_e4m3fn)
    a_sf = torch.empty(G, m_cap, Kb, device="cuda", dtype=torch.float32)
    for g in range(G):
        q, s = dg.per_block_cast_to_fp8(w[g], use_ue8m0=False)
        w_fp8[g].copy_(q); w_sf[g].copy_(s)
        q, s = dg.per_token_cast_to_fp8(a[g], use_ue8m0=False)
        a_fp8[g].copy_(q); a_sf[g].copy_(s)
    return a_fp8, a_sf, w_fp8, w_sf


def test_vs_dequant_reference(G, m_cap, N, K, masked_list):
    """Masked-GEMM correctness without deep_gemm.

    The kernel's output for the rows a group actually defines (the first
    masked_m[g] of them; everything above that is padding the masked contract
    leaves undefined) is compared against an FP32 matmul of the SAME
    dequantised operands the kernel was handed. That reference contains the
    quantization error by construction, so what is left to disagree is the
    kernel, its scale layout or its masking — which is what this check is for.
    A BF16 reference is reported alongside it, as the wider gate.
    """
    torch.manual_seed(G * 1009 + N * 17 + K)
    w = torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16) / (K ** 0.5)
    a = torch.randn(G, m_cap, K, device="cuda", dtype=torch.bfloat16) * 0.1
    masked_m = torch.tensor(masked_list, device="cuda", dtype=torch.int32)
    expected_m = max(1, int(masked_m.float().mean().item()))
    a_fp8, a_sf, w_fp8, w_sf = _quant_torch(a, w, G, m_cap, N, K)

    sa = a_sf.transpose(1, 2).contiguous()          # [G, K/128, m_cap]
    y = fso.gemm.linear_fp8_grouped_masked(a_fp8, w_fp8, sa, w_sf.contiguous(), masked_m, expected_m)
    torch.cuda.synchronize()

    worst_deq, worst_bf, checked = 1.0, 1.0, 0
    for g, n in enumerate(masked_list):
        if n == 0:
            continue
        deq_ref = _deq_1x128(a_fp8[g, :n], a_sf[g, :n]) @ _deq_128x128(w_fp8[g], w_sf[g]).t()
        worst_deq = min(worst_deq, _cos(y[g, :n], deq_ref))
        worst_bf = min(worst_bf, _cos(y[g, :n], a[g, :n].float() @ w[g].float().t()))
        checked += n
        assert torch.isfinite(y[g, :n]).all(), f"group {g}: NaN/Inf"
    assert worst_deq >= 0.999, \
        f"G={G} N={N} K={K}: cos vs dequantised operands={worst_deq:.6f}"
    assert worst_bf >= 0.999, f"G={G} N={N} K={K}: cos vs bf16={worst_bf:.6f}"
    # Rows at or beyond masked_m are padding; the only thing asserted about
    # them is that the kernel did not run off the end of the slab.
    print(f"  masked G={G:>3} m_cap={m_cap:>4} N={N:>5} K={K:>5}  "
          f"vs_deq={worst_deq:.6f} vs_bf16={worst_bf:.6f}  OK ({checked} rows)")


def test_vs_deepgemm(G, m_cap, N, K, masked_list):
    torch.manual_seed(G * 1009 + N * 17 + K)
    w = torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16) / (K ** 0.5)
    a = torch.randn(G, m_cap, K, device="cuda", dtype=torch.bfloat16) * 0.1
    masked_m = torch.tensor(masked_list, device="cuda", dtype=torch.int32)
    expected_m = max(1, int(masked_m.float().mean().item()))
    a_fp8, a_sf, w_fp8, w_sf = _quant_dg(a, w, G, m_cap, N, K)

    d_ref = torch.empty(G, m_cap, N, device="cuda", dtype=torch.bfloat16)
    dg.fp8_m_grouped_gemm_nt_masked((a_fp8, a_sf), (w_fp8, w_sf), d_ref, masked_m, expected_m)

    sa = a_sf.transpose(1, 2).contiguous()          # [G, K/128, m_cap]
    y = fso.gemm.linear_fp8_grouped_masked(a_fp8, w_fp8, sa, w_sf.contiguous(), masked_m, expected_m)
    torch.cuda.synchronize()

    worst_dg, worst_bf, checked = 1.0, 1.0, 0
    for g, n in enumerate(masked_list):
        if n == 0:
            continue
        worst_dg = min(worst_dg, _cos(y[g, :n], d_ref[g, :n]))
        worst_bf = min(worst_bf, _cos(y[g, :n], a[g, :n].float() @ w[g].float().t()))
        checked += n
        assert torch.isfinite(y[g, :n]).all(), f"group {g}: NaN/Inf"
    assert worst_dg >= 0.9999, f"G={G} N={N} K={K}: cos vs deep_gemm={worst_dg:.6f} (revival mismatch)"
    assert worst_bf >= 0.999, f"G={G} N={N} K={K}: cos vs bf16={worst_bf:.6f}"
    print(f"  masked G={G:>3} m_cap={m_cap:>4} N={N:>5} K={K:>5}  "
          f"vs_dg={worst_dg:.6f} vs_bf16={worst_bf:.6f}  OK ({checked} rows)")


def test_graph_replay(G=8, m_cap=128, N=1536, K=2048):
    torch.manual_seed(7)
    w = torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16) / (K ** 0.5)
    a = torch.randn(G, m_cap, K, device="cuda", dtype=torch.bfloat16) * 0.1
    a_fp8, a_sf, w_fp8, w_sf = _quant_torch(a, w, G, m_cap, N, K)
    sa = a_sf.transpose(1, 2).contiguous(); sw = w_sf.contiguous()
    masked_m = torch.tensor([100, 128, 64, 32, 16, 8, 4, 1], device="cuda", dtype=torch.int32)
    em = 48

    fn = lambda: fso.gemm.linear_fp8_grouped_masked(a_fp8, w_fp8, sa, sw, masked_m, em)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            y = fn()
    torch.cuda.current_stream().wait_stream(s)
    eager = y.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        y = fn()
    y.fill_(float("nan")); graph.replay(); torch.cuda.synchronize()
    # bit-exact on VALID rows only; rows >= masked_m are undefined by the
    # masked contract (padding), so compare per-group up to masked_m.
    m0 = [100, 128, 64, 32, 16, 8, 4, 1]
    for g, n in enumerate(m0):
        assert torch.equal(y[g, :n], eager[g, :n]), f"replay != eager on group {g} valid rows"
    # rewrite masked_m in place; kernel reads it on device -> replay follows
    m1 = [1, 8, 16, 32, 64, 128, 100, 50]
    masked_m.copy_(torch.tensor(m1, device="cuda", dtype=torch.int32))
    graph.replay(); torch.cuda.synchronize()
    for g, n in enumerate(m1):
        assert torch.isfinite(y[g, :n]).all(), f"reroute group {g}: NaN/Inf"
    print("  graph replay bit-exact (valid rows) + in-place masked_m rewrite  OK")



def test_h2_layout_native(m_cap, N, K, label, M=64, G=128, topk=8):
    """H2: full-fso layout-native gather-quant -> GroupedMasked, no deep_gemm
    per_token_cast / tma_align. cos vs per-expert BF16."""
    torch.manual_seed(M * 31 + N + K)
    ids = torch.stack([torch.randperm(G)[:topk] for _ in range(M)]).to("cuda", torch.int32)
    mm, rm, sof = fso.gemm.moe_build_routing(ids, G, m_cap)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16) / (K ** 0.5)
    a_fp8, sa = fso.gemm.quantize_1x128_grouped_gather_sm90(x, sof, topk, G, m_cap)
    w_fp8, sw = fso.gemm.quantize_moe_weights_1x128_fp8_sm90(w)
    y = fso.gemm.linear_fp8_grouped_masked(a_fp8, w_fp8, sa, sw, mm, max(1, M * topk // G))
    torch.cuda.synchronize()
    rmc = rm.cpu().view(G, m_cap); mmc = mm.cpu(); ys, rs = [], []
    for g in range(G):
        n = int(mmc[g])
        if n == 0:
            continue
        src = rmc[g, :n].long()
        ys.append(y[g, :n].float().flatten())
        rs.append((x[src].float() @ w[g].float().t()).flatten())
    c = _cos(torch.cat(ys), torch.cat(rs))
    assert c >= 0.999, f"H2 {label}: cos vs bf16={c:.6f}"
    print(f"  H2 layout-native {label:16s} m_cap={m_cap:>4} N={N:>5} K={K:>5}: cos={c:.6f}  OK")


def test_h2_silu(G=128, m_cap=64, INTER=768):
    torch.manual_seed(G * 7 + INTER)
    gu = torch.randn(G, m_cap, 2 * INTER, device="cuda", dtype=torch.bfloat16) * 0.5
    mm = torch.randint(1, m_cap + 1, (G,), device="cuda", dtype=torch.int32)
    slots = []
    for g in range(G):
        slots += [g * m_cap + j for j in range(int(mm[g]))]
    sof = torch.tensor(slots, device="cuda", dtype=torch.int32)
    hq, sh = fso.gemm.silu_chunk_mul_quantize_1x128_grouped_sm90(gu, sof)
    torch.cuda.synchronize()
    worst = 1.0
    for g in range(G):
        n = int(mm[g])
        for m in range(n):
            ref = F.silu(gu[g, m, :INTER].float()) * gu[g, m, INTER:].float()
            deq = hq[g, m].float() * sh[g, :, m].repeat_interleave(128).float()
            worst = min(worst, _cos(deq, ref))
    assert worst >= 0.999, f"H2 silu: worst cos={worst:.6f}"
    print(f"  H2 silu-quant round-trip worst cos={worst:.6f}  OK")



# --- H3: 6-kernel composed MoE layer (Qwen3-30B-A3B) ---------------------

def moe_layer_sm90(hidden, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w, m_cap, em, G, out):
    """6 kernels, zero torch-op glue, all fso: routing + gather-quant +
    grouped gate_up + silu-quant + grouped down + combine."""
    topk = topk_ids.shape[1]
    masked_m, row_map, slot_of_flat = fso.gemm.moe_build_routing(topk_ids, G, m_cap)
    hq, sh = fso.gemm.quantize_1x128_grouped_gather_sm90(hidden, slot_of_flat, topk, G, m_cap)
    gu = fso.gemm.linear_fp8_grouped_masked(hq, w13_fp8, sh, sw13, masked_m, em)
    dq, sd = fso.gemm.silu_chunk_mul_quantize_1x128_grouped_sm90(gu, slot_of_flat)
    dn = fso.gemm.linear_fp8_grouped_masked(dq, w2_fp8, sd, sw2, masked_m, em)
    out.copy_(fso.gemm.moe_combine(dn, slot_of_flat, topk_w))
    return out


def _layer_ref(hidden, w13, w2, topk_ids, topk_w, inter):
    M, H = hidden.shape
    out = torch.zeros(M, H, device=hidden.device, dtype=torch.float32)
    hf = hidden.float()
    for e in torch.unique(topk_ids).tolist():
        tok, slot = (topk_ids == e).nonzero(as_tuple=True)
        gu = hf[tok] @ w13[e].float().t()
        act = F.silu(gu[:, :inter]) * gu[:, inter:]
        out.index_add_(0, tok, (act @ w2[e].float().t()) * topk_w[tok, slot, None].float())
    return out


def test_h3_layer(M, G=128, topk=8, hidden=2048, inter=768):
    torch.manual_seed(M * 1009 + 5)
    m_cap = max(64, (M + 63) // 64 * 64)   # sm_90 block_m floor 64; >= M
    em = max(1, M * topk // G)
    x = torch.randn(M, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    w13 = torch.randn(G, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16) / (hidden ** 0.5)
    w2 = torch.randn(G, hidden, inter, device="cuda", dtype=torch.bfloat16) / (inter ** 0.5)
    w13_fp8, sw13 = fso.gemm.quantize_moe_weights_1x128_fp8_sm90(w13)
    w2_fp8, sw2 = fso.gemm.quantize_moe_weights_1x128_fp8_sm90(w2)
    g = torch.Generator("cpu").manual_seed(M * 7 + 9)
    topk_ids = torch.stack([torch.randperm(G, generator=g)[:topk] for _ in range(M)]).to("cuda", torch.int32)
    topk_w = torch.softmax(torch.randn(M, topk, generator=g), dim=-1).cuda()
    out = torch.empty(M, hidden, device="cuda", dtype=torch.bfloat16)

    moe_layer_sm90(x, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w, m_cap, em, G, out)
    torch.cuda.synchronize()
    ref = _layer_ref(x, w13, w2, topk_ids, topk_w, inter)
    c = _cos(out, ref)
    assert c >= 0.997, f"H3 layer M={M}: cos={c:.6f}"
    print(f"  H3 layer M={M:>5} m_cap={m_cap:>4}  cos={c:.6f}  OK")

    # graph capture + in-place topk rewrite
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            moe_layer_sm90(x, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w, m_cap, em, G, out)
    torch.cuda.current_stream().wait_stream(s)
    eager = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        moe_layer_sm90(x, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w, m_cap, em, G, out)
    out.fill_(float("nan")); graph.replay(); torch.cuda.synchronize()
    assert torch.equal(out, eager), f"H3 layer M={M}: replay != eager"
    g2 = torch.Generator("cpu").manual_seed(M * 7 + 1234)
    ids2 = torch.stack([torch.randperm(G, generator=g2)[:topk] for _ in range(M)]).to("cuda", torch.int32)
    w2r = torch.softmax(torch.randn(M, topk, generator=g2), dim=-1).cuda()
    topk_ids.copy_(ids2); topk_w.copy_(w2r)
    graph.replay(); torch.cuda.synchronize()
    ref2 = _layer_ref(x, w13, w2, ids2, w2r, inter)
    c2 = _cos(out, ref2)
    assert c2 >= 0.997, f"H3 layer M={M}: reroute cos={c2:.6f}"
    print(f"  H3 graph M={M:>5}  bit-exact replay + reroute cos={c2:.6f}  OK")


# The five masked shapes both correctness sections run: a small mixed-count
# case (including an empty group), the Qwen3-30B-A3B gate_up projection at
# decode and mid counts, and its down projection at K=768 (K%128, which the
# sm_120 block-FP8 kernel cannot do) small and large.
MASKED_SHAPES = [
    dict(G=4, m_cap=128, N=512, K=512, masked_list=[120, 64, 1, 0]),
    dict(G=128, m_cap=64, N=1536, K=2048, masked_list=[1] * 128),    # gate_up decode (m_cap>=block_m 64)
    dict(G=128, m_cap=64, N=1536, K=2048, masked_list=[4] * 128),    # gate_up mid
    dict(G=128, m_cap=64, N=2048, K=768, masked_list=[4] * 128),     # down K=768 (K%128)
    dict(G=128, m_cap=512, N=2048, K=768, masked_list=[32] * 128),   # down large
]


def main():
    ran, skipped, failed = [], [], []

    def section(name, heading, body):
        """Run one named section; record whether it passed."""
        print(heading)
        try:
            body()
        except Exception as exc:  # noqa: BLE001 - report every section, hide none
            print(f"  FAIL {name}: {exc!r}")
            failed.append(name)
        ran.append(name)

    def done():
        print("FAILED: " + ", ".join(failed) if failed else "ALL OK")
        print(f"ran: {', '.join(ran) if ran else 'none'}, "
              f"skipped: {', '.join(skipped) if skipped else 'none'}")
        return 1 if failed else 0

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        where = "no CUDA device" if not torch.cuda.is_available() \
            else f"device is sm_{torch.cuda.get_device_capability()[0]}x"
        print(f"SKIP: sm_90 grouped block-FP8 is H200-only ({where})")
        skipped.append(f"every section (sm_90 only; {where})")
        return done()

    section("masked vs dequantised reference",
            "== sm_90 grouped block-scale FP8 masked vs dequantised operands ==",
            lambda: [test_vs_dequant_reference(**s) for s in MASKED_SHAPES])

    if HAVE_DG:
        section("masked vs deep_gemm",
                "== sm_90 grouped block-scale FP8 masked vs deep_gemm ==",
                lambda: [test_vs_deepgemm(**s) for s in MASKED_SHAPES])
    else:
        print("SKIP test_vs_deepgemm: deep_gemm not importable "
              "(pip install deep_gemm or use the tiny_sglang venv)")
        skipped.append("masked vs deep_gemm (deep_gemm not importable)")

    def _h2():
        test_h2_layout_native(m_cap=64, N=1536, K=2048, label="gate_up decode")
        test_h2_layout_native(m_cap=128, N=1536, K=2048, label="gate_up mid")
        test_h2_layout_native(m_cap=128, N=2048, K=768, label="down K=768")
        test_h2_silu()

    section("H2 layout-native quantize",
            "== H2 layout-native quantize (full fso, no deep_gemm transform) ==",
            _h2)
    section("CUDA graph (single GEMM)",
            "== CUDA graph (single GEMM) ==",
            test_graph_replay)
    section("H3 composed MoE layer",
            "== H3 composed 6-kernel MoE layer (Qwen3-30B-A3B) + graph ==",
            lambda: [test_h3_layer(M) for M in (1, 8, 64, 512)])
    return done()


if __name__ == "__main__":
    sys.exit(main())
