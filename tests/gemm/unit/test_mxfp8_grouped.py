"""Grouped (MoE, masked layout) MXFP8 correctness — sm_120 (M1) and sm_100/103 (M3).

Layered like test_mxfp8_correctness.py:

1. **Gather-quant round-trip**: decode the per-group packed UE8M0 buffer with
   a pure-Python mirror of whichever layout the device's architecture uses,
   dequantize and compare against the gathered BF16 source rows. Catches
   grouped scale-factor addressing bugs.
2. **Grouped GEMM**: linear_mxfp8_grouped_masked against two references over
   the valid rows of every group — the per-expert FP32 reference computed
   from the original BF16 inputs (which mixes quantization error into the
   comparison) and, tighter, the reference computed from the *dequantized*
   FP8 operands. The second one has no quantization error left in it, so a
   scale-layout bug cannot hide behind the first one's slack.
3. **Grouped silu quant**: fused SwiGLU + quantize on the grouped layout.
4. **Full MoE layer** (Qwen3-30B-A3B geometry, E=128 topk=8 hidden=2048
   inter=768): routing -> gather-quant -> grouped gate_up -> silu-quant ->
   grouped down -> weighted combine, vs a BF16 expert-loop reference.
5. **CUDA graph**: capture the full layer, replay (bit-compare vs eager),
   then REWRITE the routing buffers in place and replay again — the second
   replay must match the new routing's BF16 reference. This is the grouped
   path's core graph property: masked_m / row_map are read on device at
   replay time, so one capture serves dynamic routing.

Scale layouts decoded here (both are opaque to callers; see docs/api/gemm.md):
  sm_120/121  per group, int32 words K-major: word (m, kp) at kp * rows + m.
  sm_100/103  per group, one CUTLASS Sm1xxBlockScaledConfig<32> atom slab of
              pad(rows,128) * K/128 words: word (m, kp) at
              ((m // 128) * K/128 + kp) * 128 + (m % 32) * 4 + (m % 128) // 32.
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

import fish_scales_ops as fso

COS_GEMM = 0.999
COS_ROUNDTRIP = 0.999
COS_DEQUANT = 0.9995  # GEMM vs the dequantized-operand reference: no quantization
                      # error left, so this is a pure layout / accumulation check
COS_LAYER = 0.997  # two chained 1x32 quantized GEMMs + quantized intermediate;
                   # same accuracy class as the triton fp8-block baseline (~0.9980)


def _sm_major() -> int:
    return torch.cuda.get_device_capability()[0]


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()


def _decode_scales_grouped(packed: torch.Tensor, rows: int, K: int) -> torch.Tensor:
    """Per-group opaque UE8M0 slab -> FP32 dequant scales [G, rows, K/32].

    `rows` is the leading extent the slab was written for: m_cap for an
    activation slab, N for a weight slab.
    """
    num_kp = K // 128
    G = packed.shape[0]
    if _sm_major() == 10:
        words = packed.reshape(G, -1).cpu().to(torch.int64) & 0xFFFFFFFF
        m_pad = (rows + 127) // 128 * 128
        assert words.shape[1] == m_pad * num_kp, (words.shape, m_pad, num_kp)
        m = torch.arange(rows)
        r = m % 128
        base = (m // 128) * (num_kp * 128) + (r % 32) * 4 + (r // 32)   # [rows]
        idx = base[:, None] + torch.arange(num_kp)[None, :] * 128       # [rows, kp]
        sel = words[:, idx.reshape(-1)].reshape(G, rows, num_kp)
    else:
        assert packed.shape[1:] == (num_kp, rows), packed.shape
        sel = (packed.cpu().to(torch.int64) & 0xFFFFFFFF).permute(0, 2, 1)  # [G, rows, kp]
    out = torch.empty(G, rows, num_kp * 4, dtype=torch.float64)
    for b in range(4):
        byte = (sel >> (8 * b)) & 0xFF                  # [G, rows, kp]
        out[:, :, b::4] = torch.pow(2.0, byte.double() - 127.0)
    return out  # [G, rows, K/32]


def build_routing(M: int, G: int, topk: int, m_cap: int, seed: int):
    """Random no-replacement routing -> masked-layout index tensors.

    Returns (topk_ids [M,topk] i32, topk_w [M,topk] f32, masked_m [G] i32,
    row_map [G*m_cap] i32, slot_of_flat [M*topk] i64). All device tensors,
    built with capture-safe ops only (the same recipe the bench uses).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    topk_ids = torch.stack([torch.randperm(G, generator=g)[:topk] for _ in range(M)])
    topk_ids = topk_ids.to("cuda", torch.int32)
    topk_w = torch.softmax(
        torch.randn(M, topk, generator=g, dtype=torch.float32), dim=-1).cuda()

    flat_e = topk_ids.flatten().long()                      # [M*topk]
    order = torch.argsort(flat_e, stable=True)
    # capture-safe histogram (torch.bincount syncs on input.max() and breaks
    # CUDA graph capture; keep this recipe in sync with the bench harness)
    counts = torch.zeros(G, device="cuda", dtype=torch.int64)
    counts.scatter_add_(0, flat_e, torch.ones_like(flat_e))  # <= M <= m_cap
    cum_excl = torch.cumsum(counts, 0) - counts
    ranks_sorted = torch.arange(M * topk, device="cuda") - cum_excl[flat_e[order]]
    slot_sorted = flat_e[order] * m_cap + ranks_sorted      # [M*topk]
    row_map = torch.full((G * m_cap,), -1, device="cuda", dtype=torch.int32)
    row_map[slot_sorted] = (order // topk).int()            # source token row
    slot_of_flat = torch.empty(M * topk, device="cuda", dtype=torch.int64)
    slot_of_flat[order] = slot_sorted
    return topk_ids, topk_w, counts.int(), row_map, slot_of_flat


def test_gather_quant_roundtrip(M: int, G: int, topk: int, K: int) -> None:
    torch.manual_seed(M * 1009 + K)
    m_cap = (M + 3) // 4 * 4
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    _, _, masked_m, row_map, slot_of_flat = build_routing(M, G, topk, m_cap, seed=M * 7 + 1)

    a_fp8, sa = fso.gemm.quantize_1x32_grouped_gather_fp8(x, slot_of_flat.int(), topk, G, m_cap)
    torch.cuda.synchronize()

    scales = _decode_scales_grouped(sa, m_cap, K)           # [G, m_cap, K/32]
    deq = a_fp8.float().cpu().double() * scales.repeat_interleave(32, dim=2)
    mm = masked_m.cpu()
    rm = row_map.cpu().view(G, m_cap)
    xs = x.float().cpu().double()
    worst = 1.0
    for gi in range(G):
        n = int(mm[gi])
        if n == 0:
            continue
        src = rm[gi, :n].long()
        c = _cos(deq[gi, :n].float(), xs[src].float())
        worst = min(worst, c)
    assert worst >= COS_ROUNDTRIP, \
        f"gather-quant round-trip M={M} K={K}: worst per-group cos={worst:.6f}"
    print(f"  gather-quant M={M:>5} G={G:>4} K={K:>5}  worst cos={worst:.6f}  OK")


def test_grouped_gemm(M: int, G: int, topk: int, N: int, K: int) -> None:
    torch.manual_seed(M * 1009 + N * 17 + K)
    m_cap = (M + 3) // 4 * 4
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    w = torch.randn(G, N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.5)
    _, _, masked_m, row_map, slot_of_flat = build_routing(M, G, topk, m_cap, seed=M * 7 + 3)
    expected_m = max(1, (M * topk + G - 1) // G)

    a_fp8, sa = fso.gemm.quantize_1x32_grouped_gather_fp8(x, slot_of_flat.int(), topk, G, m_cap)
    w_fp8, sw = fso.gemm.quantize_moe_weights_1x32_fp8(w)
    y = fso.gemm.linear_mxfp8_grouped_masked(a_fp8, w_fp8, sa, sw, masked_m, expected_m)
    torch.cuda.synchronize()

    # Dequantized operands: exactly what the GEMM should be multiplying. Any
    # mistake in the per-group scale-factor addressing shows up here even
    # though the quantization error itself has been divided out.
    sa_f = _decode_scales_grouped(sa, m_cap, K).repeat_interleave(32, dim=2)   # [G, m_cap, K]
    sw_f = _decode_scales_grouped(sw, N, K).repeat_interleave(32, dim=2)       # [G, N, K]
    a_deq = a_fp8.float().cpu().double() * sa_f
    w_deq = w_fp8.float().cpu().double() * sw_f

    mm = masked_m.cpu()
    rm = row_map.cpu().view(G, m_cap)
    worst, checked = 1.0, 0
    ys, refs, deq_refs = [], [], []
    for gi in range(G):
        n = int(mm[gi])
        if n == 0:
            continue
        src = rm[gi, :n].long()
        ref = x[src].float() @ w[gi].float().t()
        c = _cos(y[gi, :n], ref)
        worst = min(worst, c)
        checked += n
        ys.append(y[gi, :n].float().flatten())
        refs.append(ref.flatten())
        deq_refs.append((a_deq[gi, :n].float() @ w_deq[gi].float().t()).flatten())
        assert torch.isfinite(y[gi, :n]).all(), f"group {gi}: NaN/Inf in valid rows"
    assert checked == M * topk
    # Global cos carries the accuracy bar (matches the dense test's whole-
    # tensor statistic); the per-group floor is a layout tripwire — an SF
    # addressing bug sends a group to ~0.5-0.9, while small-sample UE8M0
    # noise on a 2-row group can legitimately dip a hair under 0.999.
    ys_cat = torch.cat(ys)
    g_cos = _cos(ys_cat, torch.cat(refs))
    d_cos = _cos(ys_cat.cpu(), torch.cat(deq_refs))
    assert g_cos >= COS_GEMM, \
        f"grouped gemm M={M} N={N} K={K}: global cos={g_cos:.6f}"
    assert worst >= 0.997, \
        f"grouped gemm M={M} N={N} K={K}: worst per-group cos={worst:.6f} (layout bug?)"
    assert d_cos >= COS_DEQUANT, \
        f"grouped gemm M={M} N={N} K={K}: cos vs dequantized operands={d_cos:.6f} " \
        f"< {COS_DEQUANT} (scale-layout bug?)"
    print(f"  grouped-gemm M={M:>5} G={G:>4} N={N:>5} K={K:>5}  "
          f"cos={g_cos:.6f} worst-grp={worst:.6f} cos_vs_dequant={d_cos:.6f}  OK")


def test_silu_grouped(G: int, m_cap: int, inter: int) -> None:
    torch.manual_seed(G * 31 + inter)
    gu = torch.randn(G, m_cap, 2 * inter, dtype=torch.bfloat16, device="cuda") * 0.5
    masked_m = torch.randint(0, m_cap + 1, (G,), device="cuda", dtype=torch.int32)
    # flat-pair slot list covering exactly the valid rows
    slots = []
    for gi in range(G):
        n = int(masked_m[gi])
        slots += [gi * m_cap + j for j in range(n)]
    slot_of_flat = torch.tensor(slots, device="cuda", dtype=torch.int32)
    hq, sh = fso.gemm.silu_chunk_mul_quantize_1x32_grouped_fp8(gu, slot_of_flat)
    torch.cuda.synchronize()

    scales = _decode_scales_grouped(sh, m_cap, inter)
    deq = hq.float().cpu().double() * scales.repeat_interleave(32, dim=2)
    ref = (F.silu(gu[..., :inter].float()) * gu[..., inter:].float()).cpu().double()
    mm = masked_m.cpu()
    worst = 1.0
    for gi in range(G):
        n = int(mm[gi])
        if n == 0:
            continue
        worst = min(worst, _cos(deq[gi, :n].float(), ref[gi, :n].float()))
    assert worst >= COS_ROUNDTRIP, f"grouped silu quant: worst cos={worst:.6f}"
    print(f"  silu-grouped G={G:>4} m_cap={m_cap:>4} I={inter:>5}  worst cos={worst:.6f}  OK")


# --- full MoE layer (Qwen3-30B-A3B geometry) --------------------------------

def moe_layer_fso(hidden, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w,
                  m_cap, expected_m, out, G):
    """The M1 composed layer — 6 kernels, zero torch-op glue. Routing is
    derived from topk_ids on device each call, so a captured graph follows
    whatever ids/weights the buffers hold at replay time."""
    masked_m, row_map, slot_of_flat = fso.gemm.moe_build_routing(topk_ids, G, m_cap)
    topk = topk_ids.shape[1]
    hq, sh = fso.gemm.quantize_1x32_grouped_gather_fp8(hidden, slot_of_flat, topk, G, m_cap)
    gu = fso.gemm.linear_mxfp8_grouped_masked(hq, w13_fp8, sh, sw13, masked_m, expected_m)
    dq, sd = fso.gemm.silu_chunk_mul_quantize_1x32_grouped_fp8(gu, slot_of_flat)
    dn = fso.gemm.linear_mxfp8_grouped_masked(dq, w2_fp8, sd, sw2, masked_m, expected_m)
    out.copy_(fso.gemm.moe_combine(dn, slot_of_flat, topk_w))
    return out


def moe_layer_ref(hidden, w13, w2, topk_ids, topk_w, inter):
    M, hidden_dim = hidden.shape
    out = torch.zeros(M, hidden_dim, device=hidden.device, dtype=torch.float32)
    hf = hidden.float()
    for e in torch.unique(topk_ids).tolist():
        tok, slot = (topk_ids == e).nonzero(as_tuple=True)
        gucols = hf[tok] @ w13[e].float().t()
        act = F.silu(gucols[:, :inter]) * gucols[:, inter:]
        out.index_add_(0, tok, (act @ w2[e].float().t())
                       * topk_w[tok, slot, None].float())
    return out


def test_moe_layer(M: int, G: int = 128, topk: int = 8,
                   hidden: int = 2048, inter: int = 768) -> None:
    torch.manual_seed(M * 1009 + 5)
    m_cap = (M + 3) // 4 * 4
    expected_m = max(1, (M * topk + G - 1) // G)

    x = torch.randn(M, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    w13 = torch.randn(G, 2 * inter, hidden, dtype=torch.bfloat16, device="cuda") / (hidden ** 0.5)
    w2 = torch.randn(G, hidden, inter, dtype=torch.bfloat16, device="cuda") / (inter ** 0.5)
    w13_fp8, sw13 = fso.gemm.quantize_moe_weights_1x32_fp8(w13)
    w2_fp8, sw2 = fso.gemm.quantize_moe_weights_1x32_fp8(w2)

    topk_ids, topk_w, _, _, _ = build_routing(M, G, topk, m_cap, seed=M * 7 + 9)
    out = torch.empty(M, hidden, device="cuda", dtype=torch.bfloat16)

    moe_layer_fso(x, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w,
                  m_cap, expected_m, out, G)
    torch.cuda.synchronize()
    ref = moe_layer_ref(x, w13, w2, topk_ids, topk_w, inter)
    c = _cos(out, ref)
    assert c >= COS_LAYER, f"moe layer M={M}: cos={c:.6f} < {COS_LAYER}"
    print(f"  moe-layer  M={M:>5}  cos={c:.6f}  OK")

    # ---- CUDA graph: capture, replay, then rewrite routing in place -------
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            moe_layer_fso(x, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w,
                          m_cap, expected_m, out, G)
    torch.cuda.current_stream().wait_stream(s)
    eager_out = out.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        moe_layer_fso(x, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w,
                      m_cap, expected_m, out, G)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager_out), f"moe layer M={M}: replay != eager"
    print(f"  moe-graph  M={M:>5}  replay bit-exact  OK")

    # Rewrite topk_ids / topk_w in place with a fresh assignment; the same
    # graph must now compute the new routing's answer (the whole routing
    # derivation runs on device inside the captured graph).
    topk_ids2, topk_w2, _, _, _ = build_routing(M, G, topk, m_cap, seed=M * 7 + 1234)
    topk_ids.copy_(topk_ids2)
    topk_w.copy_(topk_w2)
    graph.replay()
    torch.cuda.synchronize()
    ref2 = moe_layer_ref(x, w13, w2, topk_ids2, topk_w2, inter)
    c2 = _cos(out, ref2)
    assert c2 >= COS_LAYER, \
        f"moe layer M={M}: replay after routing rewrite cos={c2:.6f} < {COS_LAYER}"
    print(f"  moe-reroute M={M:>4}  cos={c2:.6f}  OK (dynamic routing under replay)")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable; skip")
        return 0
    major, _ = torch.cuda.get_device_capability()
    if major not in (10, 12):
        print(f"sm_{major}x device: grouped MXFP8 needs sm_100/103 or sm_120/121; skip")
        return 0
    print(f"device: sm_{major}x  ({torch.cuda.get_device_name()})")

    print("== grouped gather-quant round-trip ==")
    test_gather_quant_roundtrip(M=8, G=16, topk=4, K=256)
    test_gather_quant_roundtrip(M=64, G=128, topk=8, K=2048)
    test_gather_quant_roundtrip(M=96, G=8, topk=4, K=768)      # m_cap=96 < 128
    test_gather_quant_roundtrip(M=256, G=256, topk=8, K=512)   # m_cap=256 > 128

    print("== grouped GEMM ==")
    # Coverage sweep: G in {8, 128, 256}, m_cap in {4, 96, 256, 1024},
    # K in {512, 768, 2048}, N in {1024, 1536, 2048, 4096}. The m_cap values
    # straddle the 128-row scale-factor block the sm_100 atom layout is built
    # from (4 and 96 are below it, 256 and 1024 are multiples of it).
    test_grouped_gemm(M=8, G=16, topk=4, N=256, K=256)          # tiny smoke
    test_grouped_gemm(M=4, G=8, topk=4, N=1024, K=512)          # m_cap=4, small G
    test_grouped_gemm(M=4, G=128, topk=8, N=1536, K=2048)       # gate_up decode
    test_grouped_gemm(M=96, G=128, topk=8, N=2048, K=768)       # down, m_cap=96
    test_grouped_gemm(M=96, G=8, topk=4, N=4096, K=2048)        # wide N
    test_grouped_gemm(M=256, G=256, topk=8, N=1024, K=2048)     # Family C gate_up
    test_grouped_gemm(M=256, G=256, topk=8, N=2048, K=512)      # Family C down
    test_grouped_gemm(M=1024, G=128, topk=8, N=1536, K=2048)    # gate_up prefill
    test_grouped_gemm(M=1024, G=128, topk=8, N=2048, K=768)     # down prefill

    print("== grouped silu quant ==")
    test_silu_grouped(G=16, m_cap=32, inter=256)
    test_silu_grouped(G=128, m_cap=64, inter=768)
    test_silu_grouped(G=8, m_cap=256, inter=512)

    print("== full MoE layer (Qwen3-30B-A3B geometry) + CUDA graph ==")
    for M in (1, 8, 64, 512):
        test_moe_layer(M)

    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
