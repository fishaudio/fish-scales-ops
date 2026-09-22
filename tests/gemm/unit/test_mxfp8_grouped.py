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

import os
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


def _grouped_sf_words(packed: torch.Tensor, rows: int, K: int) -> torch.Tensor:
    """Per-group opaque UE8M0 slab -> the raw int32 words as [G, rows, K/128].

    The single place either scale layout is addressed from the test side.
    `rows` is the leading extent the slab was written for: m_cap for an
    activation slab, N for a weight slab. The result stays on the device the
    slab lives on and is widened to int64 only so the sign bit of the int32
    word does not leak into the comparison.
    """
    num_kp = K // 128
    G = packed.shape[0]
    dev = packed.device
    if _sm_major() == 10:
        words = packed.reshape(G, -1).to(torch.int64) & 0xFFFFFFFF
        m_pad = (rows + 127) // 128 * 128
        assert words.shape[1] == m_pad * num_kp, (words.shape, m_pad, num_kp)
        m = torch.arange(rows, device=dev)
        r = m % 128
        base = (m // 128) * (num_kp * 128) + (r % 32) * 4 + (r // 32)   # [rows]
        idx = base[:, None] + torch.arange(num_kp, device=dev)[None, :] * 128   # [rows, kp]
        return words[:, idx.reshape(-1)].reshape(G, rows, num_kp)
    assert packed.shape[1:] == (num_kp, rows), packed.shape
    return (packed.to(torch.int64) & 0xFFFFFFFF).permute(0, 2, 1)       # [G, rows, kp]


def _decode_scales_grouped(packed: torch.Tensor, rows: int, K: int) -> torch.Tensor:
    """Per-group opaque UE8M0 slab -> FP32 dequant scales [G, rows, K/32].

    `rows` is the leading extent the slab was written for: m_cap for an
    activation slab, N for a weight slab.
    """
    num_kp = K // 128
    G = packed.shape[0]
    sel = _grouped_sf_words(packed, rows, K).cpu()          # [G, rows, kp]
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
    masked_m, row_map, slot_of_flat = routing_index_tensors(topk_ids, G, m_cap)
    return topk_ids, topk_w, masked_m, row_map, slot_of_flat


def routing_index_tensors(topk_ids: torch.Tensor, G: int, m_cap: int):
    """topk_ids [M, topk] -> (masked_m [G] i32, row_map [G*m_cap] i32,
    slot_of_flat [M*topk] i64), the flat-pair layout moe_build_routing
    produces. Split out of build_routing so a hand-built draw (the hot-expert
    case further down) goes through exactly the same recipe."""
    M, topk = topk_ids.shape
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
    return counts.int(), row_map, slot_of_flat


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


# --- gather-quantize: pair-space and token-scatter must agree bit for bit ---
#
# The grouped activation quantize has two implementations behind one op. The
# pair-space kernel gives a warp to every routed (token, expert) pair, so a
# token routed to topk experts is loaded, amax-reduced and converted topk
# times. The token-scatter kernel does that work once per token and stores the
# finished FP8 word and scale word to each of the topk destinations. Both are
# selected by `FSO_GATHER_QUANT_ONCE` (0 / 1) or, unset, by the launcher's
# rule, and the sm_100/103 scale layout is the only layout that offers the
# second one.
#
# The two forms perform exactly the same arithmetic on exactly the same inputs,
# so the contract is stricter than a cosine: every byte of `a_fp8` and every
# packed scale word on a DEFINED row must be identical. A defined row is a row
# below masked_m[g]; rows above it are never written by either kernel, so they
# hold whatever the allocator returned and must be excluded from the
# comparison or the test would compare uninitialised memory.
#
# The knob is read once per process into a function-local static (it has to be:
# a captured graph must launch the same kernel on every replay), so the two
# forms cannot be exercised in one process and each arm runs in its own child.
#
# The cells below cover the M values where the launcher's rule falls on either
# side of its crossover (1, 3, 8, 64 below it; 1000, 4096 above it), both group
# counts of the published MoE families, a hot-expert draw in which one expert
# holds every token and another holds none — which is the draw a uniform random
# top-k never produces and the one that stresses the destination loop — and two
# cells with a topk other than 8 so the runtime-topk instantiation of the
# scatter kernel is exercised alongside the compile-time-topk one.
QUANT_ONCE_CELLS = [
    # (M, G, topk, K, draw)
    (1, 128, 8, 2048, "random"),
    (3, 128, 8, 2048, "random"),
    (8, 128, 8, 2048, "random"),
    (64, 128, 8, 2048, "random"),
    (1000, 128, 8, 2048, "random"),
    (4096, 128, 8, 2048, "random"),
    (64, 128, 8, 2048, "hot"),
    (1000, 128, 8, 2048, "hot"),
    (8, 256, 8, 2048, "random"),
    (1000, 256, 8, 2048, "random"),
    (64, 32, 5, 768, "random"),
    (1000, 64, 4, 512, "random"),
]


def quant_once_digest(M: int, G: int, topk: int, K: int, draw: str):
    """(sha256 over the defined bytes, number of defined rows) for one cell.

    Everything the cell depends on is seeded, so two processes that run this
    function with the same arguments feed the op identical activations and an
    identical routing and may only differ in which kernel the launcher picked.
    """
    import hashlib
    m_cap = (M + 3) // 4 * 4
    torch.manual_seed(M * 1009 + G * 31 + topk * 7 + K)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    if draw == "hot":
        topk_ids = hot_expert_ids(M, G, topk, hot_expert=0, cold_expert=G - 1,
                                  seed=M * 7 + 11)
        masked_m, _, slot_of_flat = routing_index_tensors(topk_ids, G, m_cap)
    else:
        _, _, masked_m, _, slot_of_flat = build_routing(M, G, topk, m_cap, seed=M * 7 + 1)

    a_fp8, sa = fso.gemm.quantize_1x32_grouped_gather_fp8(x, slot_of_flat.int(), topk, G, m_cap)
    torch.cuda.synchronize()

    keep = (torch.arange(m_cap, device="cuda").view(1, -1)
            < masked_m.to(torch.int64).view(-1, 1))             # [G, m_cap]
    rows = a_fp8.view(G, m_cap, K)[keep].view(torch.uint8)      # [defined, K]
    words = _grouped_sf_words(sa, m_cap, K)[keep]               # [defined, K/128]
    h = hashlib.sha256()
    h.update(rows.cpu().numpy().tobytes())
    h.update(words.cpu().numpy().tobytes())
    return h.hexdigest(), int(keep.sum().item())


def quant_once_child() -> None:
    for M, G, topk, K, draw in QUANT_ONCE_CELLS:
        digest, n = quant_once_digest(M, G, topk, K, draw)
        print(f"DIGEST {M} {G} {topk} {K} {draw} {n} {digest}", flush=True)


def run_quant_once_cases() -> None:
    """Parent side: one child per setting of the per-process knob, then the
    per-cell digests of every other setting must equal the pair-space kernel's.
    """
    import subprocess
    here = os.path.abspath(__file__)
    knobs = ("FSO_GATHER_QUANT_ONCE", "FSO_GATHER_QUANT_TOPK_STATIC")

    def child(extra):
        env = dict(os.environ)
        for k in knobs:
            env.pop(k, None)
        env.update(extra)
        r = subprocess.run([sys.executable, here, "--quant-once-case"], env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, \
            f"gather-quantize child {extra} failed (exit {r.returncode}):\n{r.stderr[-2000:]}"
        out = {}
        for line in r.stdout.splitlines():
            if line.startswith("DIGEST "):
                f = line.split()
                out[(int(f[1]), int(f[2]), int(f[3]), int(f[4]), f[5])] = (int(f[6]), f[7])
        assert len(out) == len(QUANT_ONCE_CELLS), \
            f"gather-quantize child {extra} produced {len(out)} digests:\n{r.stdout[-2000:]}"
        return out

    base = child({"FSO_GATHER_QUANT_ONCE": "0"})
    arms = (
        ("scatter, compile-time topk", {"FSO_GATHER_QUANT_ONCE": "1"}),
        ("scatter, runtime topk", {"FSO_GATHER_QUANT_ONCE": "1",
                                   "FSO_GATHER_QUANT_TOPK_STATIC": "0"}),
        ("launcher rule", {}),
    )
    for name, extra in arms:
        got = child(extra)
        for key in sorted(base):
            n0, d0 = base[key]
            n1, d1 = got[key]
            assert n1 == n0, \
                f"gather-quantize [{name}] cell {key}: {n1} defined rows, pair-space kernel had {n0}"
            assert d1 == d0, \
                f"gather-quantize [{name}] cell (M,G,topk,K,draw)={key}: output differs from " \
                f"the pair-space kernel on the defined rows ({d1[:16]} vs {d0[:16]})"
        print(f"  quant-once {name:<28} {len(base)} cells, "
              f"{sum(v[0] for v in base.values())} defined rows bit-identical  OK")


def test_grouped_gemm(M: int, G: int, topk: int, N: int, K: int) -> None:
    torch.manual_seed(M * 1009 + N * 17 + K)
    m_cap = (M + 3) // 4 * 4
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    w = torch.randn(G, N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.5)
    _, _, masked_m, row_map, slot_of_flat = build_routing(M, G, topk, m_cap, seed=M * 7 + 3)
    expected_m = max(1, (M * topk + G - 1) // G)
    # The host-static bound a top-k router guarantees. Supplying it is what lets
    # the sm_100 dispatcher consider the slot-bound decode route, so the small-M
    # cells of this sweep check that route's numbers under the shipped rule and
    # the larger ones check the pointer-array route.
    max_active_groups = min(M * topk, G)

    a_fp8, sa = fso.gemm.quantize_1x32_grouped_gather_fp8(x, slot_of_flat.int(), topk, G, m_cap)
    w_fp8, sw = fso.gemm.quantize_moe_weights_1x32_fp8(w)
    y = fso.gemm.linear_mxfp8_grouped_masked(a_fp8, w_fp8, sa, sw, masked_m, expected_m,
                                             max_active_groups)
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
                  m_cap, expected_m, out, G, max_active_groups=0):
    """The M1 composed layer — 6 kernels, zero torch-op glue. Routing is
    derived from topk_ids on device each call, so a captured graph follows
    whatever ids/weights the buffers hold at replay time.

    `max_active_groups` is the host-static upper bound on how many experts can
    hold at least one row, i.e. min(M * topk, G): a top-k router gives an
    expert at most one row per token. It is constant for a given (M, topk, G),
    so a graph captured at this M stays valid for every routing draw at that M,
    including the hot-expert draw the slot cases below rewrite in."""
    masked_m, row_map, slot_of_flat = fso.gemm.moe_build_routing(topk_ids, G, m_cap)
    topk = topk_ids.shape[1]
    hq, sh = fso.gemm.quantize_1x32_grouped_gather_fp8(hidden, slot_of_flat, topk, G, m_cap)
    gu = fso.gemm.linear_mxfp8_grouped_masked(hq, w13_fp8, sh, sw13, masked_m, expected_m,
                                              max_active_groups)
    dq, sd = fso.gemm.silu_chunk_mul_quantize_1x32_grouped_fp8(gu, slot_of_flat)
    dn = fso.gemm.linear_mxfp8_grouped_masked(dq, w2_fp8, sd, sw2, masked_m, expected_m,
                                              max_active_groups)
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
    mag = min(M * topk, G)

    x = torch.randn(M, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    w13 = torch.randn(G, 2 * inter, hidden, dtype=torch.bfloat16, device="cuda") / (hidden ** 0.5)
    w2 = torch.randn(G, hidden, inter, dtype=torch.bfloat16, device="cuda") / (inter ** 0.5)
    w13_fp8, sw13 = fso.gemm.quantize_moe_weights_1x32_fp8(w13)
    w2_fp8, sw2 = fso.gemm.quantize_moe_weights_1x32_fp8(w2)

    topk_ids, topk_w, _, _, _ = build_routing(M, G, topk, m_cap, seed=M * 7 + 9)
    out = torch.empty(M, hidden, device="cuda", dtype=torch.bfloat16)

    moe_layer_fso(x, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w,
                  m_cap, expected_m, out, G, mag)
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
                          m_cap, expected_m, out, G, mag)
    torch.cuda.current_stream().wait_stream(s)
    eager_out = out.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        moe_layer_fso(x, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w,
                      m_cap, expected_m, out, G, mag)
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


# --- sm_100/103 fused-SwiGLU FC1 --------------------------------------------
#
# The fused FC1 is one grouped GEMM whose epilogue emits MXFP8(silu(gate) * up)
# instead of the bf16 gate_up tensor, so the separate SwiGLU-and-requantise
# kernel disappears from the layer. Four properties need their own cases and
# none of them is reachable from the tests above.
#
#  * The op REQUIRES gate/up interleaved weight rows and cannot detect the
#    wrong layout: fed the usual [gate; up] stacking it multiplies the wrong
#    pairs together and returns a finite, wrong answer with no error. Both the
#    interleave itself and a negative control for the wrong layout are checked.
#  * Its numerics differ from the two-kernel path's on purpose, because the
#    bf16 intermediate is gone, so a bit-comparison against the old path would
#    fail for the right reason. The discriminating comparison is against an
#    FP32 reference built from the dequantised operands, where the fused output
#    must be at least as close as the two-kernel output is.
#  * A layer that carries interleaved weights but falls back to the unfused FC1
#    on the decode band feeds the old SwiGLU kernel an interleaved gate_up
#    tensor, which is what `pairwise=True` is for. That form must be
#    bit-identical to the old kernel fed the de-interleaved tensor, and the old
#    kernel fed the interleaved one must be visibly wrong.
#  * The layer built on the fused FC1 must still capture into one CUDA graph
#    and follow an in-place routing rewrite.

FUSED_FAMILIES = {
    # name: (G, topk, hidden, inter). Family B is the Qwen3-30B-A3B routed
    # layer, Family C the Qwen3.5-35B-A3B one.
    "B": (128, 8, 2048, 768),
    "C": (256, 8, 2048, 512),
}
# The fused output tracks an FP32 reference at least this well. It is the same
# bar the dequantised-operand check above uses, minus the one e4m3 rounding the
# output now carries (the reference is not quantised, the output is).
COS_FUSED = 0.999


def fused_topk_ids(draw: str, M: int, G: int, topk: int, seed: int) -> torch.Tensor:
    """Three routing draws whose coverage differs in ways the fused epilogue
    can tell apart.

    `random` spreads M*topk rows over every expert, so most groups hold a
    handful of rows and the epilogue only ever sees the first rows of a tile.
    `hot` puts expert 0 in every token's set, so one group holds M rows — the
    only draw that fills a group past the 128-row scale-factor block, which is
    where the scale slab's row addressing changes. `dead` makes half the
    experts unreachable, so half the groups have a zero problem shape and
    contribute no tiles at all.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    if draw == "random":
        ids = torch.stack([torch.randperm(G, generator=g)[:topk] for _ in range(M)])
    elif draw == "hot":
        rest = torch.stack(
            [torch.randperm(G - 1, generator=g)[: topk - 1] + 1 for _ in range(M)])
        ids = torch.cat([torch.zeros(M, 1, dtype=rest.dtype), rest], dim=1)
    elif draw == "dead":
        half = G // 2
        ids = torch.stack([torch.randperm(half, generator=g)[:topk] for _ in range(M)])
    else:
        raise ValueError(f"unknown draw {draw!r}")
    return ids.to("cuda", torch.int32)


def _deq_group(fp8_rows: torch.Tensor, scale_bytes: torch.Tensor) -> torch.Tensor:
    """One group's FP8 rows times its decoded 1x32 scales, in FP32 on device.

    `scale_bytes` is the [rows, K/32] slice `_decode_scales_grouped` produced,
    so the only thing happening here is the 32-fold broadcast along K.
    """
    return fp8_rows.float() * scale_bytes.to(fp8_rows.device, torch.float32).repeat_interleave(32, dim=1)


def fused_reference(a_fp8, sa, w13_fp8, sw13, live, m_cap, hidden, inter):
    """FP32 h = silu(gate) * up per live group, from the DEQUANTISED operands.

    Building the reference from the dequantised operands rather than from the
    original bf16 tensors takes the input quantisation error out of the
    comparison, so what is left measures the epilogue and the output
    quantisation alone. Computed one group at a time because the dequantised
    weight tensor for all experts at once would be tens of gigabytes.
    """
    sa_b = _decode_scales_grouped(sa, m_cap, hidden)             # [G, m_cap, hidden/32]
    sw_b = _decode_scales_grouped(sw13, 2 * inter, hidden)       # [G, 2I, hidden/32]
    out = {}
    for gi, n in live.items():
        w_deq = _deq_group(w13_fp8[gi], sw_b[gi])                # [2I, hidden]
        a_deq = _deq_group(a_fp8[gi, :n], sa_b[gi, :n])          # [n, hidden]
        acc = a_deq @ w_deq.t()                                  # [n, 2I]
        out[gi] = F.silu(acc[:, :inter]) * acc[:, inter:]
        del w_deq, a_deq, acc
    return out


def _live_groups(masked_m: torch.Tensor) -> dict:
    mm = masked_m.cpu()
    return {gi: int(mm[gi]) for gi in range(mm.numel()) if int(mm[gi]) > 0}


def _cos_against(h_fp8, sh, live, m_cap, inter, ref) -> float:
    """Cosine of a dequantised [G, m_cap, inter] MXFP8 activation slab against
    the per-group FP32 reference, over the defined rows only."""
    sh_b = _decode_scales_grouped(sh, m_cap, inter)
    got, want = [], []
    for gi, n in live.items():
        got.append(_deq_group(h_fp8[gi, :n], sh_b[gi, :n]).flatten())
        want.append(ref[gi].flatten())
    return _cos(torch.cat(got), torch.cat(want))


def _fused_max_ulp_gap(h_a, sa_, h_b, sb_, live, m_cap, inter) -> float:
    """Largest disagreement between two MXFP8 activation slabs, measured in
    steps of the e4m3 grid at each element's own magnitude.

    Both slabs carry their own 1x32 UE8M0 scales, so an element's grid spacing
    is `scale * 2^(e-3)` for a normal e4m3 number of exponent e and
    `scale * 2^-9` at the bottom of the subnormal range. Taking the coarser of
    the two arms' spacings makes the measure symmetric.
    """
    sa_b = _decode_scales_grouped(sa_, m_cap, inter)
    sb_b = _decode_scales_grouped(sb_, m_cap, inter)
    worst = 0.0
    for gi, n in live.items():
        a = _deq_group(h_a[gi, :n], sa_b[gi, :n])
        b = _deq_group(h_b[gi, :n], sb_b[gi, :n])
        sa_e = sa_b[gi, :n].to(a.device, torch.float32).repeat_interleave(32, dim=1)
        sb_e = sb_b[gi, :n].to(a.device, torch.float32).repeat_interleave(32, dim=1)
        mag = torch.maximum(a.abs(), b.abs())
        step = torch.maximum(mag * 0.125, torch.maximum(sa_e, sb_e) * (2.0 ** -9))
        worst = max(worst, float(((a - b).abs() / step).max()))
    return worst


def fused_cell(fam: str, M: int, draw: str, chain: bool = False) -> None:
    """One (family, M, draw) cell: the fused FC1 against the two-kernel pair,
    both scored on the same FP32 reference, plus the two negative controls and
    optionally the FC1 -> FC2 chain."""
    G, topk, hidden, inter = FUSED_FAMILIES[fam]
    m_cap = (M + 3) // 4 * 4
    expected_m = max(1, (M * topk + G - 1) // G)
    mag = min(M * topk, G)
    torch.manual_seed(hash((fam, M, draw)) % (2 ** 31))

    x = torch.randn(M, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    w13 = torch.randn(G, 2 * inter, hidden, dtype=torch.bfloat16, device="cuda") / (hidden ** 0.5)
    topk_ids = fused_topk_ids(draw, M, G, topk, seed=M * 31 + len(draw) + len(fam))
    masked_m, _, slot_of_flat = routing_index_tensors(topk_ids, G, m_cap)
    slot_i32 = slot_of_flat.int()
    live = _live_groups(masked_m)

    a_fp8, sa = fso.gemm.quantize_1x32_grouped_gather_fp8(x, slot_i32, topk, G, m_cap)
    w13_fp8, sw13 = fso.gemm.quantize_moe_weights_1x32_fp8(w13)
    w13i_fp8, sw13i = fso.gemm.quantize_moe_weights_1x32_fp8(w13, w13_interleave=True)

    # Control: the pair this replaces.
    gu = fso.gemm.linear_mxfp8_grouped_masked(a_fp8, w13_fp8, sa, sw13, masked_m,
                                              expected_m, mag)
    h_two, sh_two = fso.gemm.silu_chunk_mul_quantize_1x32_grouped_fp8(gu, slot_i32)
    # The fused FC1.
    h_fus, sh_fus = fso.gemm.linear_mxfp8_grouped_masked_swiglu(
        a_fp8, w13i_fp8, sa, sw13i, masked_m, expected_m, mag)
    torch.cuda.synchronize()

    for gi, n in live.items():
        assert torch.isfinite(h_fus[gi, :n].float()).all(), \
            f"fused FC1 {fam} M={M} {draw}: non-finite output in group {gi}"

    ref = fused_reference(a_fp8, sa, w13_fp8, sw13, live, m_cap, hidden, inter)
    c_fus = _cos_against(h_fus, sh_fus, live, m_cap, inter, ref)
    c_two = _cos_against(h_two, sh_two, live, m_cap, inter, ref)
    assert c_fus >= COS_FUSED, \
        f"fused FC1 {fam} M={M} {draw}: cos vs FP32 reference {c_fus:.7f} < {COS_FUSED}"
    # The fused path removes one bf16 rounding, so it must not be WORSE than
    # the path it replaces on the same draw. The slack is one float32 ulp of
    # the cosine itself, not a tolerance on the numerics.
    assert c_fus >= c_two - 1e-6, \
        f"fused FC1 {fam} M={M} {draw}: cos {c_fus:.7f} below the two-kernel path's {c_two:.7f}"

    # Negative control 1: the same op fed the STACKED weights. The layouts are
    # the same bytes in a different row order, so nothing raises; what must
    # happen is that the answer fails the accuracy gate.
    h_bad, sh_bad = fso.gemm.linear_mxfp8_grouped_masked_swiglu(
        a_fp8, w13_fp8, sa, sw13, masked_m, expected_m, mag)
    torch.cuda.synchronize()
    c_bad = _cos_against(h_bad, sh_bad, live, m_cap, inter, ref)
    assert c_bad < 0.9, \
        f"fused FC1 {fam} M={M} {draw}: NON-interleaved weights scored {c_bad:.7f}, which the " \
        f"accuracy gate would accept — the negative control no longer discriminates"

    # The two arms are not bit-identical on purpose (the fused one never
    # rounds the gate_up product to bf16), so the discriminating check is that
    # they land on neighbouring points of the SAME e4m3 grid. One step of that
    # grid at a value's own magnitude is 2^-3 of it for a normal e4m3 number,
    # and the grid does not get finer than 2^-9 of the block's dequant scale,
    # which is the floor below. A store that reduced the amax over the wrong
    # 32 values would move whole blocks by a power of two and fail this.
    d_ulp = _fused_max_ulp_gap(h_fus, sh_fus, h_two, sh_two, live, m_cap, inter)
    assert d_ulp <= 1.0 + 1e-6, \
        f"fused FC1 {fam} M={M} {draw}: the fused and two-kernel outputs differ by " \
        f"{d_ulp:.3f} e4m3 steps, more than the one step the removed bf16 staging explains"

    print(f"  fused-fc1  {fam} M={M:>5} {draw:<6} live={len(live):>3}  "
          f"cos={c_fus:.7f} (two-kernel {c_two:.7f}, non-interleaved control {c_bad:.4f}, "
          f"gap {d_ulp:.3f} e4m3 steps)  OK")

    if chain:
        # FC1 -> FC2: hand each FC1 output to the unchanged grouped GEMM and
        # score both chains on the same FP32 reference chain.
        w2 = torch.randn(G, hidden, inter, dtype=torch.bfloat16, device="cuda") / (inter ** 0.5)
        w2_fp8, sw2 = fso.gemm.quantize_moe_weights_1x32_fp8(w2)
        dn_fus = fso.gemm.linear_mxfp8_grouped_masked(h_fus, w2_fp8, sh_fus, sw2, masked_m,
                                                      expected_m, mag)
        dn_two = fso.gemm.linear_mxfp8_grouped_masked(h_two, w2_fp8, sh_two, sw2, masked_m,
                                                      expected_m, mag)
        torch.cuda.synchronize()
        sw2_b = _decode_scales_grouped(sw2, hidden, inter)
        got_f, got_t, want = [], [], []
        for gi, n in live.items():
            w2_deq = _deq_group(w2_fp8[gi], sw2_b[gi])           # [hidden, inter]
            want.append((ref[gi] @ w2_deq.t()).flatten())
            got_f.append(dn_fus[gi, :n].float().flatten())
            got_t.append(dn_two[gi, :n].float().flatten())
            del w2_deq
        want_c = torch.cat(want)
        cc_f = _cos(torch.cat(got_f), want_c)
        cc_t = _cos(torch.cat(got_t), want_c)
        assert cc_f >= COS_FUSED, f"fused chain {fam} M={M} {draw}: cos={cc_f:.7f}"
        assert cc_f >= cc_t - 1e-6, \
            f"fused chain {fam} M={M} {draw}: cos {cc_f:.7f} below the two-kernel chain's {cc_t:.7f}"
        print(f"  fused-chain {fam} M={M:>4} {draw:<6} FC1->FC2 cos={cc_f:.7f} "
              f"(two-kernel chain {cc_t:.7f})  OK")


def test_fused_interleave_bit_exact(fam: str) -> None:
    """Quantising interleaved weights and interleaving quantised weights give
    the same bytes.

    The 1x32 weight quantiser works per row along K, so a row's FP8 bytes and
    its UE8M0 scale word depend on that row alone; the Sm1xx atom slab keeps a
    row's four K-block bytes in one int32 word, so permuting rows permutes
    whole words and never splits one. Both facts together mean the interleave
    is a relabelling with no numerical content, and this asserts it on the
    bytes rather than arguing it.
    """
    G, _topk, hidden, inter = FUSED_FAMILIES[fam]
    torch.manual_seed(len(fam) * 977 + inter)
    # A handful of experts is enough: the property is per row.
    Gs = 4
    w13 = torch.randn(Gs, 2 * inter, hidden, dtype=torch.bfloat16, device="cuda") / (hidden ** 0.5)
    w_stacked, s_stacked = fso.gemm.quantize_moe_weights_1x32_fp8(w13)
    w_direct, s_direct = fso.gemm.quantize_moe_weights_1x32_fp8(w13, w13_interleave=True)
    w_perm, s_perm = fso.gemm.interleave_w13_fp8(w_stacked, s_stacked)
    torch.cuda.synchronize()
    assert torch.equal(w_direct.view(torch.uint8), w_perm.view(torch.uint8)), \
        f"interleave {fam}: fp8 bytes differ between quantise-then-permute and permute-then-quantise"
    assert torch.equal(s_direct, s_perm), \
        f"interleave {fam}: scale words differ between quantise-then-permute and permute-then-quantise"
    # And the permutation really is the [g0, u0, g1, u1, ...] one.
    assert torch.equal(w_perm[:, 0::2].view(torch.uint8), w_stacked[:, :inter].view(torch.uint8))
    assert torch.equal(w_perm[:, 1::2].view(torch.uint8), w_stacked[:, inter:].view(torch.uint8))
    print(f"  interleave {fam}  {Gs} experts x {2 * inter} rows: fp8 bytes and scale words "
          f"bit-identical both ways  OK")


def test_fused_pairwise_kernel(fam: str, M: int = 64) -> None:
    """The pairwise SwiGLU kernel against the old one fed a de-interleaved
    tensor, and the old one fed the interleaved tensor as a negative control.

    This is the kernel a layer still needs on the decode band: there the
    dispatcher takes the swap-orientation slot route, which cannot fuse, so the
    unfused FC1 runs — but on interleaved weights, so its gate_up output has
    gate and up in alternating columns.
    """
    G, topk, hidden, inter = FUSED_FAMILIES[fam]
    m_cap = (M + 3) // 4 * 4
    torch.manual_seed(M * 13 + inter)
    topk_ids = fused_topk_ids("random", M, G, topk, seed=M * 3 + 77)
    masked_m, _, slot_of_flat = routing_index_tensors(topk_ids, G, m_cap)
    slot_i32 = slot_of_flat.int()

    gu_inter = torch.randn(G, m_cap, 2 * inter, dtype=torch.bfloat16, device="cuda") * 0.5
    gu_stack = torch.empty_like(gu_inter)
    gu_stack[..., :inter] = gu_inter[..., 0::2]
    gu_stack[..., inter:] = gu_inter[..., 1::2]

    h_pair, s_pair = fso.gemm.silu_chunk_mul_quantize_1x32_grouped_fp8(
        gu_inter, slot_i32, pairwise=True)
    h_old, s_old = fso.gemm.silu_chunk_mul_quantize_1x32_grouped_fp8(gu_stack, slot_i32)
    # Negative control: the old kernel on the interleaved tensor pairs
    # gate_{2i} with gate_{2i+I}, which is a different function entirely.
    h_wrong, s_wrong = fso.gemm.silu_chunk_mul_quantize_1x32_grouped_fp8(gu_inter, slot_i32)
    torch.cuda.synchronize()

    live = _live_groups(masked_m)
    same_rows = 0
    for gi, n in live.items():
        assert torch.equal(h_pair[gi, :n].view(torch.uint8), h_old[gi, :n].view(torch.uint8)), \
            f"pairwise SwiGLU {fam}: fp8 bytes differ from the old kernel in group {gi}"
        same_rows += n
    sw_pair = _grouped_sf_words(s_pair, m_cap, inter)
    sw_old = _grouped_sf_words(s_old, m_cap, inter)
    for gi, n in live.items():
        assert torch.equal(sw_pair[gi, :n], sw_old[gi, :n]), \
            f"pairwise SwiGLU {fam}: scale words differ from the old kernel in group {gi}"
    wrong_rows = sum(
        1 for gi, n in live.items()
        if not torch.equal(h_wrong[gi, :n].view(torch.uint8), h_old[gi, :n].view(torch.uint8)))
    assert wrong_rows == len(live), \
        f"pairwise SwiGLU {fam}: the old kernel fed the INTERLEAVED tensor agreed with the " \
        f"de-interleaved answer in {len(live) - wrong_rows} of {len(live)} groups — the negative " \
        f"control no longer discriminates"
    print(f"  pairwise-silu {fam} M={M:>4}  {same_rows} defined rows bit-identical to the old "
          f"kernel; old kernel on interleaved input differs in all {len(live)} live groups  OK")


def moe_layer_fso_fused(hidden, w13i_fp8, sw13i, w2_fp8, sw2, topk_ids, topk_w,
                        m_cap, expected_m, out, G, max_active_groups=0):
    """The layer with the fused FC1: five kernels instead of six. The SwiGLU
    kernel is gone because FC1's epilogue already produced FC2's operand.

    The packed active-expert list is asked for only where the dispatcher would
    take the slot-bound route, since that route is its only reader; everywhere
    else the routing kernel skips the compaction that builds it. The layer's
    two GEMMs are FC1 (N = 2 * inter, K = hidden) and FC2 (N = hidden,
    K = inter) and either one can be on the slot route, so the flag is the
    disjunction over both."""
    inter2 = w13i_fp8.shape[1]           # 2 * inter
    hidden_dim = w2_fp8.shape[1]
    inter = w2_fp8.shape[2]
    want_slots = (
        fso.gemm.mxfp8_grouped_slot_possible(m_cap, inter2, hidden_dim, G, max_active_groups)
        or fso.gemm.mxfp8_grouped_slot_possible(m_cap, hidden_dim, inter, G, max_active_groups))
    routing = fso.gemm.moe_build_routing(topk_ids, G, m_cap, with_slots=want_slots)
    masked_m, _row_map, slot_of_flat = routing[:3]
    slot_to_expert = routing[3] if want_slots else None
    topk = topk_ids.shape[1]
    hq, sh = fso.gemm.quantize_1x32_grouped_gather_fp8(hidden, slot_of_flat, topk, G, m_cap)
    dq, sd = fso.gemm.linear_mxfp8_grouped_masked_swiglu(
        hq, w13i_fp8, sh, sw13i, masked_m, expected_m, max_active_groups, slot_to_expert)
    dn = fso.gemm.linear_mxfp8_grouped_masked(dq, w2_fp8, sd, sw2, masked_m, expected_m,
                                              max_active_groups, slot_to_expert)
    out.copy_(fso.gemm.moe_combine(dn, slot_of_flat, topk_w))
    return out


def test_fused_layer_graph(M: int, fam: str = "B") -> None:
    """The fused layer end to end, then captured and replayed, then replayed
    again after the routing is rewritten in place."""
    G, topk, hidden, inter = FUSED_FAMILIES[fam]
    torch.manual_seed(M * 1009 + 77)
    m_cap = (M + 3) // 4 * 4
    expected_m = max(1, (M * topk + G - 1) // G)
    mag = min(M * topk, G)

    x = torch.randn(M, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    w13 = torch.randn(G, 2 * inter, hidden, dtype=torch.bfloat16, device="cuda") / (hidden ** 0.5)
    w2 = torch.randn(G, hidden, inter, dtype=torch.bfloat16, device="cuda") / (inter ** 0.5)
    w13i_fp8, sw13i = fso.gemm.quantize_moe_weights_1x32_fp8(w13, w13_interleave=True)
    w2_fp8, sw2 = fso.gemm.quantize_moe_weights_1x32_fp8(w2)

    topk_ids, topk_w, _, _, _ = build_routing(M, G, topk, m_cap, seed=M * 7 + 31)
    out = torch.empty(M, hidden, device="cuda", dtype=torch.bfloat16)

    moe_layer_fso_fused(x, w13i_fp8, sw13i, w2_fp8, sw2, topk_ids, topk_w,
                        m_cap, expected_m, out, G, mag)
    torch.cuda.synchronize()
    ref = moe_layer_ref(x, w13, w2, topk_ids, topk_w, inter)
    c = _cos(out, ref)
    assert c >= COS_LAYER, f"fused moe layer M={M}: cos={c:.6f} < {COS_LAYER}"
    print(f"  fused-layer  M={M:>5}  cos={c:.6f}  OK")

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            moe_layer_fso_fused(x, w13i_fp8, sw13i, w2_fp8, sw2, topk_ids, topk_w,
                                m_cap, expected_m, out, G, mag)
    torch.cuda.current_stream().wait_stream(s)
    eager_out = out.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        moe_layer_fso_fused(x, w13i_fp8, sw13i, w2_fp8, sw2, topk_ids, topk_w,
                            m_cap, expected_m, out, G, mag)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager_out), f"fused moe layer M={M}: replay != eager"
    print(f"  fused-graph  M={M:>5}  replay bit-exact  OK")

    topk_ids2, topk_w2, _, _, _ = build_routing(M, G, topk, m_cap, seed=M * 7 + 4242)
    topk_ids.copy_(topk_ids2)
    topk_w.copy_(topk_w2)
    graph.replay()
    torch.cuda.synchronize()
    ref2 = moe_layer_ref(x, w13, w2, topk_ids2, topk_w2, inter)
    c2 = _cos(out, ref2)
    assert c2 >= COS_LAYER, \
        f"fused moe layer M={M}: replay after routing rewrite cos={c2:.6f} < {COS_LAYER}"
    print(f"  fused-reroute M={M:>4}  cos={c2:.6f}  OK (dynamic routing under replay)")


def test_fused_route_helper() -> None:
    """The host-side router. Both grouped routes carry a fused FC1 now — the
    pointer-array cascade and the swap-orientation slot kernel — so with the
    knob unset the answer is yes on both sides of the route boundary, and
    FSO_FC1_FUSED=0 must still turn the whole feature off."""
    G, topk, hidden, inter = FUSED_FAMILIES["B"]
    n_w = 2 * inter
    knob = os.environ.get("FSO_FC1_FUSED")
    # m_cap = 4 with a real slot bound is inside the slot route's band; m_cap =
    # 1024 is far outside it (the slot kernel's token tile is 64 wide).
    decode = fso.gemm.mxfp8_grouped_swiglu_fused_route(4, n_w, hidden, G, min(1 * topk, G))
    prefill = fso.gemm.mxfp8_grouped_swiglu_fused_route(1024, n_w, hidden, G, G)
    no_bound = fso.gemm.mxfp8_grouped_swiglu_fused_route(4, n_w, hidden, G, 0)
    if knob == "0":
        assert not decode and not prefill and not no_bound, \
            f"FSO_FC1_FUSED=0 still routed to the fused FC1: {decode} {prefill} {no_bound}"
    elif knob == "1":
        assert decode and prefill and no_bound, \
            f"FSO_FC1_FUSED=1 did not force the fused FC1: {decode} {prefill} {no_bound}"
    else:
        assert decode, "the router refused the fused FC1 at m_cap=4 with a slot bound, where " \
                       "the dispatcher takes the slot route and that route has its own fused " \
                       "epilogue (run b300_mxfp8_20260917/M-A2)"
        assert prefill, "the router refused the fused FC1 at m_cap=1024, where the dispatcher " \
                        "takes the pointer-array cascade"
        assert no_bound, "the router refused the fused FC1 with no slot bound, where the slot " \
                         "route is unreachable and the cascade always runs"
    print(f"  fused-route  FSO_FC1_FUSED={knob or '<unset>'}  m_cap=4/bound -> {decode}, "
          f"m_cap=1024 -> {prefill}, m_cap=4/no bound -> {no_bound}  OK")


def test_fused_slot_persistent_skip(fam: str, M: int) -> None:
    """The fused FC1 on the slot route, at a grid with more tiles than the
    machine has SMs.

    Why this needs its own case. CUTLASS's static persistent scheduler
    truncates the grid to the SM count, so past that point one CTA walks
    several tiles and the kernel re-tests liveness at every one of them. The
    fused epilogue adds a shared buffer and two named-barrier arrivals per
    tile, which is exactly the kind of state a skipped tile could leave
    inconsistent: if a dead tile took the barrier and a live one did not, or the
    other way round, the epilogue warps of one CTA would deadlock or read the
    previous tile's maxima. A hot draw is used so that one expert holds every
    row, which maximises the number of live tiles per CTA.
    """
    G, topk, hidden, inter = FUSED_FAMILIES[fam]
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    m_cap = (M + 3) // 4 * 4
    mag = min(M * topk, G)
    tiles = mag * ((2 * inter + 127) // 128)
    assert tiles > sms, \
        f"fused slot persistent case {fam} M={M}: {tiles} tiles does not exceed {sms} SMs, so " \
        f"the scheduler would not truncate the grid and the case tests nothing"
    fused_cell(fam, M, "hot")
    print(f"  fused-slot-persistent {fam} M={M:>3}  {tiles} tiles over {sms} SMs  OK")


def run_fused_cases() -> None:
    """Everything the fused FC1 needs, in this process except the knob cases."""
    import subprocess
    here = os.path.abspath(__file__)

    for fam in ("B", "C"):
        test_fused_interleave_bit_exact(fam)
    # M <= 32 is the decode band, where the dispatcher takes the
    # swap-orientation slot kernel, so these cells exercise the slot route's own
    # fused epilogue; M = 64 and 1024 stay on the pointer-array one.
    for fam in ("B", "C"):
        for M in (1, 2, 4, 8, 16, 32, 64, 1024):
            for draw in ("random", "hot", "dead"):
                fused_cell(fam, M, draw, chain=(draw == "random"))
    for fam in ("B", "C"):
        test_fused_slot_persistent_skip(fam, 16)
        test_fused_slot_persistent_skip(fam, 32)
    for fam in ("B", "C"):
        test_fused_pairwise_kernel(fam)
    for M in (1, 8, 64, 1024):
        test_fused_layer_graph(M)

    # FSO_FC1_FUSED is a per-process static, so each setting is its own
    # subprocess.
    for knob in (None, "0", "1"):
        env = dict(os.environ)
        env.pop("FSO_FC1_FUSED", None)
        if knob is not None:
            env["FSO_FC1_FUSED"] = knob
        r = subprocess.run([sys.executable, here, "--fused-case", "route"], env=env,
                           capture_output=True, text=True)
        print(r.stdout, end="")
        assert r.returncode == 0, \
            f"fused route case (FSO_FC1_FUSED={knob}) failed (exit {r.returncode}):\n{r.stderr}"


# --- sm_100/103 slot-bound decode route -------------------------------------
#
# The slot route is the second entry of the sm_100 grouped cascade: a
# swap-orientation kernel (expert weights on M, routed tokens on a 64-wide N)
# that is only correct while the whole row capacity fits one token tile. Three
# properties need their own cases, and none of them is reachable from the
# random-draw tests above.
#
#  * A random top-k draw over 128 or 256 experts puts about M*topk/G rows on an
#    expert — four at M = 64 on Family B — so with the tokens on the N axis the
#    whole-CTA early exit kills every token tile after the first. A sweep that
#    passes therefore says nothing at all about the second and later token
#    tiles, which is exactly where the fault lives. The HOT-EXPERT case forces
#    one expert to hold m_cap rows and another to hold none.
#  * FSO_GROUPED_SLOT is read once per process into a function-local static (so
#    it cannot change between capture and replay), so every case that needs a
#    different setting runs in its own subprocess.
#  * The guard is the only thing standing between a caller and a silently wrong
#    answer: past one token tile the kernel returns 2^-127 times the right
#    result, with no NaN and no error. The guard cases assert the refusal.

SLOT_SHAPES = {
    "B_gate_up": dict(N=1536, K=2048, G=128, topk=8),
    "B_down": dict(N=2048, K=768, G=128, topk=8),
    "C_gate_up": dict(N=1024, K=2048, G=256, topk=8),
    "C_down": dict(N=2048, K=512, G=256, topk=8),
}

_SLOT_W_CACHE: dict = {}


def slot_weights(G: int, N: int, K: int):
    """(w bf16, w_fp8, sw, w_deq) for one shape, built once per process.

    `w_deq` is the dequantised weight the reference multiplies, kept in float32
    rather than float64: the UE8M0 scales are exact powers of two and the fp8
    values are exact, so the product is exact in float32, and a sweep over six
    M values would otherwise rebuild a multi-gigabyte float64 slab per cell."""
    key = (G, N, K)
    if key not in _SLOT_W_CACHE:
        torch.manual_seed(N * 17 + K)
        w = torch.randn(G, N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.5)
        w_fp8, sw = fso.gemm.quantize_moe_weights_1x32_fp8(w)
        sw_f = _decode_scales_grouped(sw, N, K).repeat_interleave(32, dim=2)
        w_deq = (w_fp8.float().cpu().double() * sw_f).float()
        del sw_f
        _SLOT_W_CACHE.clear()          # one shape resident at a time
        _SLOT_W_CACHE[key] = (w, w_fp8, sw, w_deq)
    return _SLOT_W_CACHE[key]


def slot_cell(shape: str, M: int, seed_off: int = 0, hot: bool = False,
              draw: str = None, use_list: bool = False):
    """Build one (shape, M) cell and run it through the grouped op with the
    slot bound supplied. Returns (y, cos_vs_dequantised_reference, masked_m).

    `draw` names the top-k draw: "random" (a uniform permutation per token),
    "hot" (expert 0 in every token's set, expert G-1 in none) or "tail" (only
    the last `SLOT_TAIL_HOT` experts are reachable, so almost every slot the
    host-static bound provides is dead). `hot=True` is the old spelling of
    draw="hot" and is kept so the existing cases read unchanged."""
    s = SLOT_SHAPES[shape]
    G, K, N, topk = s["G"], s["K"], s["N"], s["topk"]
    m_cap = (M + 3) // 4 * 4
    draw = draw or ("hot" if hot else "random")
    torch.manual_seed(M * 1009 + N * 17 + K + seed_off)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    if draw == "hot":
        topk_ids = hot_expert_ids(M, G, topk, hot_expert=0, cold_expert=G - 1,
                                  seed=M * 7 + 11 + seed_off)
    elif draw == "tail":
        topk_ids = tail_expert_ids(M, G, topk, SLOT_TAIL_HOT, seed=M * 7 + 23 + seed_off)
    else:
        g = torch.Generator(device="cpu").manual_seed(M * 7 + 3 + seed_off)
        topk_ids = torch.stack([torch.randperm(G, generator=g)[:topk] for _ in range(M)])
        topk_ids = topk_ids.to("cuda", torch.int32)
    masked_m, row_map, slot_of_flat = routing_index_tensors(topk_ids, G, m_cap)

    _, w_fp8, sw, w_deq = slot_weights(G, N, K)
    a_fp8, sa = fso.gemm.quantize_1x32_grouped_gather_fp8(x, slot_of_flat.int(), topk, G, m_cap)
    expected_m = max(1, (M * topk + G - 1) // G)
    max_active_groups = min(M * topk, G)
    # `use_list` hands the route the packed active-expert list the routing op
    # emits, instead of letting it build the same list with its own one-block
    # kernel. The two must produce bit-identical output; that is what
    # slot_case_list_bit_identity checks.
    slot_to_expert = None
    if use_list:
        _mm2, _rm2, _sof2, slot_to_expert = fso.gemm.moe_build_routing(
            topk_ids, G, m_cap, with_slots=True)
    y = fso.gemm.linear_mxfp8_grouped_masked(
        a_fp8, w_fp8, sa, sw, masked_m, expected_m, max_active_groups, slot_to_expert)
    torch.cuda.synchronize()

    sa_f = _decode_scales_grouped(sa, m_cap, K).repeat_interleave(32, dim=2)
    a_deq = (a_fp8.float().cpu().double() * sa_f).float()
    mm = masked_m.cpu()
    ys, deq_refs = [], []
    for gi in range(G):
        n = int(mm[gi])
        if n == 0:
            continue
        assert torch.isfinite(y[gi, :n]).all(), f"{shape} M={M} group {gi}: NaN/Inf in valid rows"
        ys.append(y[gi, :n].float().cpu().flatten())
        deq_refs.append((a_deq[gi, :n] @ w_deq[gi].t()).flatten())
    d_cos = _cos(torch.cat(ys), torch.cat(deq_refs))
    return y, d_cos, masked_m


def hot_expert_ids(M: int, G: int, topk: int, hot_expert: int, cold_expert: int, seed: int):
    """A top-k draw in which `hot_expert` appears for every token (so it ends up
    holding exactly M rows, i.e. m_cap rows whenever M is a multiple of 4) and
    `cold_expert` appears for none (so it holds zero)."""
    assert hot_expert != cold_expert and topk <= G - 1
    g = torch.Generator(device="cpu").manual_seed(seed)
    pool = [e for e in range(G) if e not in (hot_expert, cold_expert)]
    pool_t = torch.tensor(pool)
    rows = []
    for _ in range(M):
        pick = pool_t[torch.randperm(len(pool), generator=g)[:topk - 1]]
        rows.append(torch.cat([torch.tensor([hot_expert]), pick]))
    return torch.stack(rows).to("cuda", torch.int32)


SLOT_TAIL_HOT = 28


def tail_expert_ids(M: int, G: int, topk: int, n_hot: int, seed: int):
    """A top-k draw in which only the LAST `n_hot` experts are reachable, so
    experts 0 .. G-n_hot-1 all end up empty.

    This is the draw that leaves the largest number of dead slots inside the
    host-static bound the caller supplies. The bound is min(M * topk, G) and
    knows nothing about the draw, so at M = 64 with top-8 and G = 128 the
    kernel is given 128 slots of which only `n_hot` hold an expert."""
    assert n_hot >= topk and n_hot < G
    g = torch.Generator(device="cpu").manual_seed(seed)
    pool = torch.arange(G - n_hot, G)
    rows = [pool[torch.randperm(n_hot, generator=g)[:topk]] for _ in range(M)]
    return torch.stack(rows).to("cuda", torch.int32)


def slot_tile_count(shape: str, M: int) -> tuple:
    """(tiles the slot route's grid is built from, this device's SM count).

    The route runs a 128 x 64 tile in swap orientation, so the tile space is
    ceil(N_w / 128) M tiles by ceil(m_cap / 64) token tiles by S slots, with
    S = min(M * topk, G) the host-static bound the caller passes. CUTLASS's
    static persistent scheduler clamps the launched grid to the SM count, so
    `tiles > sms` is exactly the condition under which a CTA loops over several
    tiles -- the regime the per-tile skip exists for."""
    s = SLOT_SHAPES[shape]
    m_cap = (M + 3) // 4 * 4
    S = min(M * s["topk"], s["G"])
    tiles = S * ((s["N"] + 127) // 128) * ((m_cap + 63) // 64)
    return tiles, torch.cuda.get_device_properties(0).multi_processor_count


def slot_valid_rows(y: torch.Tensor, masked_m: torch.Tensor) -> torch.Tensor:
    """The rows the masked contract defines. Rows at or beyond masked_m[g], and
    every row of an inactive expert, are never written by either kernel, so a
    comparison that includes them compares whatever torch.empty returned."""
    mm = masked_m.to(torch.int64).view(-1, 1)
    keep = torch.arange(y.shape[1], device=y.device).view(1, -1) < mm
    return y[keep]


def slot_case_forced_sweep() -> None:
    for shape in SLOT_SHAPES:
        for M in (1, 2, 4, 8, 16, 64):
            _, d_cos, _ = slot_cell(shape, M)
            assert d_cos >= COS_DEQUANT, \
                f"slot route {shape} M={M}: cos vs dequantized operands={d_cos:.6f} < {COS_DEQUANT}"
            print(f"  slot-forced {shape:<10} M={M:>3}  cos_vs_dequant={d_cos:.6f}  OK")


def slot_case_hot_expert() -> None:
    # m_cap = 64 is the token tile's width, i.e. the largest legal capacity, and
    # it is the only place a full tile of live token rows is ever executed.
    # m_cap = 128 needs a second token tile and must therefore be refused.
    #
    # Two faults live past the first token tile and the guard covers both. The
    # scale-factor one — the SFB tile index counting TileN-wide tiles while the
    # scale tensor is tiled in 128-column blocks, so the second tile reads an
    # out-of-bounds block and a zero UE8M0 byte scales the row by 2^-127 — is
    # reconciled by hand inside CUTLASS for exactly the tile widths it supports,
    # 64 among them, so the 64-wide tile this phase ships is in fact correct
    # past one tile and this case will not see that fault at TileN = 64. It
    # would at TileN 8, 16 or 32. The other fault does apply here: the
    # persistent grid is truncated to the SM count and the whole-CTA early exit
    # tests only the CTA's first tile, so a CTA whose first tile is dead drops
    # every live tile after it, and whether that happens depends on the draw.
    # The guard is therefore the thing this case defends; a cosine that is still
    # right at m_cap = 128 does not make the configuration safe.
    for shape in SLOT_SHAPES:
        for M in (8, 16, 32, 64, 128):
            try:
                _, d_cos, masked_m = slot_cell(shape, M, hot=True)
            except RuntimeError as e:
                assert M > 64 and "exceeds TileN" in str(e), \
                    f"slot route {shape} M={M} hot expert: unexpected refusal: {e}"
                print(f"  slot-hot    {shape:<10} M={M:>3}  guard refused m_cap={M}  OK")
                continue
            assert M <= 64, \
                f"slot route {shape} M={M} hot expert: m_cap={M} needs a second token tile, which the " \
                f"guard is supposed to forbid, and it did not fire (cos vs dequantized " \
                f"operands={d_cos:.6f})"
            assert int(masked_m[0]) == M, f"hot expert holds {int(masked_m[0])} rows, expected {M}"
            assert int(masked_m[-1]) == 0, "cold expert is not empty"
            assert d_cos >= COS_DEQUANT, \
                f"slot route {shape} M={M} hot expert: cos vs dequantized operands={d_cos:.6f} < {COS_DEQUANT}"
            print(f"  slot-hot    {shape:<10} M={M:>3}  hot={int(masked_m[0])} cold=0  "
                  f"cos_vs_dequant={d_cos:.6f}  OK")


def slot_case_determinism() -> None:
    for shape in SLOT_SHAPES:
        for M in (1, 8, 64):
            y1, _, mm1 = slot_cell(shape, M)
            y1 = slot_valid_rows(y1, mm1).clone()
            y2, _, mm2 = slot_cell(shape, M)
            assert torch.equal(mm1, mm2)
            assert torch.equal(y1, slot_valid_rows(y2, mm2)), \
                f"slot route {shape} M={M}: two runs differ over the masked-contract rows"
            print(f"  slot-det    {shape:<10} M={M:>3}  {y1.numel()} rows bit-identical  OK")


def slot_case_guard_mcap() -> None:
    # m_cap = 128 exceeds the 64-wide token tile. Forced, the op must refuse.
    try:
        slot_cell("B_gate_up", 128)
    except RuntimeError as e:
        assert "exceeds TileN" in str(e), f"wrong refusal message: {e}"
        print(f"  slot-guard  m_cap=128 forced -> refused  OK")
        return
    raise AssertionError("FSO_GROUPED_SLOT=force with m_cap=128 was not refused")


def slot_case_guard_bound() -> None:
    # Legality includes the host-static slot bound: without it the grid cannot
    # be sized, so a forced call that does not supply it must be refused too.
    s = SLOT_SHAPES["B_gate_up"]
    G, K, N, topk, M = s["G"], s["K"], s["N"], s["topk"], 4
    torch.manual_seed(1)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    _, _, masked_m, _, slot_of_flat = build_routing(M, G, topk, M, seed=5)
    _, w_fp8, sw, _ = slot_weights(G, N, K)
    a_fp8, sa = fso.gemm.quantize_1x32_grouped_gather_fp8(x, slot_of_flat.int(), topk, G, M)
    try:
        fso.gemm.linear_mxfp8_grouped_masked(a_fp8, w_fp8, sa, sw, masked_m, 1)
    except RuntimeError as e:
        assert "max_active_groups" in str(e), f"wrong refusal message: {e}"
        print("  slot-guard  max_active_groups=0 forced -> refused  OK")
        return
    raise AssertionError("FSO_GROUPED_SLOT=force with max_active_groups=0 was not refused")


def slot_case_route_pick(mag: int) -> None:
    """Capture the grouped GEMM as the FIRST call of a fresh process and let the
    route name itself.

    The two routes produce bit-identical output — both accumulate the same
    K-tiles in the same order into a float32 accumulator and round once — so the
    numbers cannot say which one ran. What can say it is the pool each route
    fills on its first call: the pointer-array route's argument-array pool and
    the slot route's slot-list pool both refuse to allocate inside a capture,
    and each names itself when it does. Skipping the eager warm-up therefore
    turns the route choice into an observable message, with no debug API and no
    profiler.
    """
    s = SLOT_SHAPES["B_gate_up"]
    G, K, N, topk, M, m_cap = s["G"], s["K"], s["N"], s["topk"], 4, 4
    torch.manual_seed(M * 1009 + N * 17 + K)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    _, _, masked_m, _, slot_of_flat = build_routing(M, G, topk, m_cap, seed=M * 7 + 3)
    _, w_fp8, sw, _ = slot_weights(G, N, K)
    a_fp8, sa = fso.gemm.quantize_1x32_grouped_gather_fp8(x, slot_of_flat.int(), topk, G, m_cap)
    torch.cuda.synchronize()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=st):
        fso.gemm.linear_mxfp8_grouped_masked(a_fp8, w_fp8, sa, sw, masked_m, 1, mag)
    raise AssertionError("a pool allocated during capture instead of refusing")


def slot_case_route_off(ref_path: str, write_ref: bool) -> None:
    """The default rule must not change any existing caller's answer: with
    max_active_groups left at 0 the result has to be bit-identical to the same
    call under FSO_GROUPED_SLOT=0. (Which route ran is decided by
    slot_case_route_pick above; this is the numerical half.)"""
    s = SLOT_SHAPES["B_gate_up"]
    G, K, N, topk, M, m_cap = s["G"], s["K"], s["N"], s["topk"], 4, 4
    torch.manual_seed(M * 1009 + N * 17 + K)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    _, _, masked_m, _, slot_of_flat = build_routing(M, G, topk, m_cap, seed=M * 7 + 3)
    _, w_fp8, sw, _ = slot_weights(G, N, K)
    a_fp8, sa = fso.gemm.quantize_1x32_grouped_gather_fp8(x, slot_of_flat.int(), topk, G, m_cap)

    y_nobound = slot_valid_rows(
        fso.gemm.linear_mxfp8_grouped_masked(a_fp8, w_fp8, sa, sw, masked_m, 1, 0), masked_m)
    torch.cuda.synchronize()
    if write_ref:
        torch.save(y_nobound.cpu(), ref_path)
        print("  slot-off    FSO_GROUPED_SLOT=0 reference written  OK")
        return
    ref = torch.load(ref_path).cuda()
    assert torch.equal(y_nobound, ref), \
        "max_active_groups=0 under the default rule did not reproduce the FSO_GROUPED_SLOT=0 result"
    print("  slot-off    max_active_groups=0 reproduces the FSO_GROUPED_SLOT=0 result bit-exactly  OK")


def slot_case_layer_graph_hot() -> None:
    """The whole six-kernel layer captured once at a decode M, then replayed
    after the routing buffers have been rewritten in place with a hot-expert
    draw.

    Why this case exists, and why the layer rather than the GEMM alone. The
    slot bound `max_active_groups` the layer now passes is min(M * topk, G),
    which is a function of M only. The routing draw is not part of it, so a
    graph captured on one draw has to stay correct for every other draw at the
    same M — including the worst one, where a single expert receives a row from
    every token and another receives none. That is the draw that puts m_cap
    live rows in one group's token tile and leaves other groups' tiles empty,
    which is precisely the configuration the slot route's packed slot list and
    whole-CTA early exit have to get right, and a random top-k draw over 128
    experts never produces it. A failure here would surface as the replay
    disagreeing with an eager call on the same rewritten routing, or as a
    layer cosine that falls away from the FP32 expert-loop reference.

    The layer is used rather than the GEMM because the bound is wired in at the
    layer call sites, and because the rewrite has to travel through
    moe_build_routing, the gather-quant, both grouped GEMMs and the combine
    inside the captured graph, which is what a deployed decode loop does."""
    G, topk, hidden_dim, inter = 128, 8, 2048, 768
    for M in (1, 4):
        m_cap = (M + 3) // 4 * 4
        expected_m = max(1, (M * topk + G - 1) // G)
        mag = min(M * topk, G)
        torch.manual_seed(M * 1009 + 77)
        x = torch.randn(M, hidden_dim, dtype=torch.bfloat16, device="cuda") * 0.1
        w13 = torch.randn(G, 2 * inter, hidden_dim, dtype=torch.bfloat16,
                          device="cuda") / (hidden_dim ** 0.5)
        w2 = torch.randn(G, hidden_dim, inter, dtype=torch.bfloat16,
                         device="cuda") / (inter ** 0.5)
        w13_fp8, sw13 = fso.gemm.quantize_moe_weights_1x32_fp8(w13)
        w2_fp8, sw2 = fso.gemm.quantize_moe_weights_1x32_fp8(w2)
        topk_ids, topk_w, _, _, _ = build_routing(M, G, topk, m_cap, seed=M * 7 + 9)
        out = torch.empty(M, hidden_dim, device="cuda", dtype=torch.bfloat16)

        # Eager first, on the default stream, then on the capture stream: the
        # slot route's slot-list pool and the CUTLASS workspace both allocate
        # on their first call and both refuse to do so inside a capture.
        for _ in range(3):
            moe_layer_fso(x, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w,
                          m_cap, expected_m, out, G, mag)
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                moe_layer_fso(x, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w,
                              m_cap, expected_m, out, G, mag)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            moe_layer_fso(x, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w,
                          m_cap, expected_m, out, G, mag)

        ids_hot = hot_expert_ids(M, G, topk, hot_expert=0, cold_expert=G - 1,
                                 seed=M * 7 + 4242)
        w_hot = torch.softmax(torch.randn(M, topk, device="cuda", dtype=torch.float32), dim=-1)
        topk_ids.copy_(ids_hot)
        topk_w.copy_(w_hot)
        mm_hot, _, _ = fso.gemm.moe_build_routing(topk_ids, G, m_cap)
        torch.cuda.synchronize()
        assert int(mm_hot[0]) == M, \
            f"layer hot draw M={M}: expert 0 holds {int(mm_hot[0])} rows, expected {M}"
        assert int(mm_hot[-1]) == 0, f"layer hot draw M={M}: cold expert is not empty"

        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        replayed = out.clone()

        moe_layer_fso(x, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w,
                      m_cap, expected_m, out, G, mag)
        torch.cuda.synchronize()
        assert torch.equal(replayed, out), \
            f"layer hot draw M={M}: replay after the in-place routing rewrite " \
            f"differs from an eager call on the same routing"
        c = _cos(replayed, moe_layer_ref(x, w13, w2, topk_ids, topk_w, inter))
        assert c >= COS_LAYER, f"layer hot draw M={M}: cos={c:.6f} < {COS_LAYER}"
        print(f"  slot-layer  M={M:>3}  hot={int(mm_hot[0])} cold=0  replay==eager "
              f"bit-exact  cos={c:.6f}  OK")


def slot_case_big_grid() -> None:
    """Correctness where the launched grid is SMALLER than the tile count, so
    every CTA loops over several tiles.

    Why this case exists. CUTLASS's static persistent scheduler asks for its
    grid with truncation to the SM count, so a problem with more tiles than the
    machine has SMs is run by CTAs that each walk a strided run of the tile
    space. Whether a tile is live depends on the routing draw, which a capture
    cannot know, so liveness has to be decided per tile rather than once per
    CTA. Both draws below put the grid firmly in that regime and differ in how
    many of the slots are dead: the hot draw leaves almost none (S experts hold
    rows), the tail draw leaves almost all of them (only the last 28 experts are
    reachable, while the bound still provides S slots).

    Determinism is checked over five runs rather than two, because a dropped or
    doubled tile does not have to show up on every launch -- which CTA walks
    which tile depends on nothing that changes between runs, but a pipeline
    state advanced for a skipped tile would surface as a race."""
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for shape in SLOT_SHAPES:
        for draw in ("hot", "tail"):
            for M in (8, 64):
                tiles, _ = slot_tile_count(shape, M)
                assert tiles > sms, \
                    f"{shape} M={M}: {tiles} tiles is not more than {sms} SMs, so this case " \
                    f"would not exercise a looping CTA at all"
                ys = []
                for run in range(5):
                    y, d_cos, masked_m = slot_cell(shape, M, draw=draw)
                    if run == 0:
                        first_cos, first_mm = d_cos, masked_m.clone()
                        active = int((masked_m > 0).sum())
                        assert torch.isfinite(slot_valid_rows(y, masked_m)).all(), \
                            f"{shape} M={M} draw={draw}: NaN or Inf in the masked-contract rows"
                        assert d_cos >= COS_DEQUANT, \
                            f"slot route {shape} M={M} draw={draw}: cos vs dequantized " \
                            f"operands={d_cos:.6f} < {COS_DEQUANT}"
                        if draw == "tail":
                            assert active == SLOT_TAIL_HOT, \
                                f"{shape} M={M} tail draw: {active} active experts, expected {SLOT_TAIL_HOT}"
                            assert int(masked_m[0]) == 0, "tail draw: expert 0 is not empty"
                        else:
                            assert int(masked_m[0]) == M, \
                                f"{shape} M={M} hot draw: expert 0 holds {int(masked_m[0])} rows, expected {M}"
                    else:
                        assert torch.equal(masked_m, first_mm)
                    ys.append(slot_valid_rows(y, masked_m).clone())
                for run in range(1, 5):
                    assert torch.equal(ys[0], ys[run]), \
                        f"slot route {shape} M={M} draw={draw}: run {run} differs from run 0 " \
                        f"over the masked-contract rows"
                S = min(M * SLOT_SHAPES[shape]["topk"], SLOT_SHAPES[shape]["G"])
                print(f"  slot-grid   {shape:<10} M={M:>3} draw={draw:<4} tiles={tiles:>5} "
                      f"grid={sms} slots={S} active={active}  5 runs bit-identical  "
                      f"cos_vs_dequant={first_cos:.6f}  OK")


def slot_case_big_grid_graph() -> None:
    """One capture, two very different draws, in the large-grid regime.

    The grid the graph records is built from the host-static bound
    min(M * topk, G) and from nothing the draw decides, so a capture taken on
    one draw must stay correct when the routing is rewritten in place to
    another. The pair used here is the worst available: a hot draw in which
    almost every slot is live, rewritten to a tail draw in which only 28 of the
    128 experts are reachable and the rest of the slots go dead. The replay
    must agree bit for bit with a fresh eager call on the rewritten routing."""
    shape, M = "B_gate_up", 64
    s = SLOT_SHAPES[shape]
    G, K, N, topk = s["G"], s["K"], s["N"], s["topk"]
    m_cap = (M + 3) // 4 * 4
    tiles, sms = slot_tile_count(shape, M)
    assert tiles > sms, f"{tiles} tiles is not more than {sms} SMs"
    expected_m = max(1, (M * topk + G - 1) // G)
    mag = min(M * topk, G)
    _, w_fp8, sw, _ = slot_weights(G, N, K)

    torch.manual_seed(M * 1009 + 4242)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    ids = hot_expert_ids(M, G, topk, hot_expert=0, cold_expert=G - 1, seed=99)
    masked_m, _, slot_of_flat = routing_index_tensors(ids, G, m_cap)
    a_fp8, sa = fso.gemm.quantize_1x32_grouped_gather_fp8(x, slot_of_flat.int(), topk, G, m_cap)

    def run():
        return fso.gemm.linear_mxfp8_grouped_masked(
            a_fp8, w_fp8, sa, sw, masked_m, expected_m, mag)

    out = run()                                   # eager: pools and workspace
    torch.cuda.synchronize()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            out = run()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=st):
        captured = run()

    ids_tail = tail_expert_ids(M, G, topk, SLOT_TAIL_HOT, seed=7)
    mm_tail, _, slot_tail = routing_index_tensors(ids_tail, G, m_cap)
    a_tail, sa_tail = fso.gemm.quantize_1x32_grouped_gather_fp8(
        x, slot_tail.int(), topk, G, m_cap)
    # In place, so the graph's recorded pointers stay valid.
    masked_m.copy_(mm_tail)
    a_fp8.copy_(a_tail)
    sa.copy_(sa_tail)
    torch.cuda.synchronize()
    assert int((masked_m > 0).sum()) == SLOT_TAIL_HOT

    captured.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    replayed = slot_valid_rows(captured, masked_m).clone()

    eager = run()
    torch.cuda.synchronize()
    assert torch.equal(replayed, slot_valid_rows(eager, masked_m)), \
        "large-grid capture: the replay after the in-place routing rewrite differs " \
        "from an eager call on the same routing"
    assert torch.isfinite(replayed).all(), "large-grid capture: NaN or Inf after the replay"
    print(f"  slot-grid   {shape:<10} M={M:>3} capture  tiles={tiles} grid={sms}  "
          f"hot draw -> tail draw ({SLOT_TAIL_HOT} active)  replay==eager bit-exact  OK")



def slot_list_reference(topk_ids: torch.Tensor, G: int) -> torch.Tensor:
    """The packed active-expert list `moe_build_routing(with_slots=True)` owes.

    Definition, in torch: an expert is active when at least one routed pair
    names it; the active ids go into the low entries in ascending order and
    every remaining entry is -1."""
    counts = torch.bincount(topk_ids.flatten().long(), minlength=G)
    active = torch.nonzero(counts > 0).flatten().to(torch.int32)
    ref = torch.full((G,), -1, dtype=torch.int32, device="cuda")
    ref[: active.numel()] = active
    return ref


def slot_case_list_contract() -> None:
    """The fourth output of the routing op against a torch reference.

    Four draws per shape family: a uniform permutation, the `hot` draw (one
    expert in every token's set and one in none), the `tail` draw (only the
    last SLOT_TAIL_HOT experts reachable, so most experts are empty) and a
    single-expert draw (every pair names expert 0, so exactly one entry is
    active). M is chosen to exercise BOTH routing kernels: M = 4 and M = 64 are
    below the 4096-routed-pair threshold and take the single-CTA builder, while
    M = 1024 at top-8 is 8192 pairs and takes the multi-CTA one on sm_100."""
    for G, topk in ((128, 8), (256, 8)):
        for M in (4, 64, 1024):
            m_cap = (M + 3) // 4 * 4
            pairs = M * topk
            kernel = "multi-CTA" if (pairs >= 4096 and _sm_major() == 10) else "single-CTA"
            for draw in ("random", "hot", "tail", "single"):
                if draw == "hot":
                    ids = hot_expert_ids(M, G, topk, 0, G - 1, seed=M * 7 + 11)
                elif draw == "tail":
                    ids = tail_expert_ids(M, G, topk, SLOT_TAIL_HOT, seed=M * 7 + 23)
                elif draw == "single":
                    ids = torch.zeros(M, topk, dtype=torch.int32, device="cuda")
                else:
                    g = torch.Generator(device="cpu").manual_seed(M * 7 + 3 + G)
                    ids = torch.stack([torch.randperm(G, generator=g)[:topk]
                                       for _ in range(M)]).to("cuda", torch.int32)
                mm, rm, sof, se = fso.gemm.moe_build_routing(ids, G, m_cap, with_slots=True)
                ref = slot_list_reference(ids, G)
                assert se.dtype == torch.int32 and se.shape == (G,), \
                    f"slot_to_expert must be int32 [G]; got {se.dtype} {tuple(se.shape)}"
                assert torch.equal(se, ref), (
                    f"slot_to_expert mismatch G={G} M={M} draw={draw}: "
                    f"{int((se != ref).sum())} of {G} entries differ")
                # masked_m must still be what it always was.
                counts = torch.bincount(ids.flatten().long(), minlength=G).to(torch.int32)
                assert torch.equal(mm, counts), f"masked_m changed G={G} M={M} draw={draw}"
                n_active = int((counts > 0).sum())
                print(f"  slot-list   G={G:>3} M={M:>4} {kernel:<10} draw={draw:<6} "
                      f"active={n_active:>3}  OK")
            # with_slots=False must still return exactly three tensors.
            out = fso.gemm.moe_build_routing(ids, G, m_cap)
            assert len(out) == 3, f"moe_build_routing default arity changed: {len(out)}"


def slot_case_list_bit_identity() -> None:
    """Passing the routing-emitted list must not change a single output byte.

    The list the route builds for itself and the list the routing op emits are
    the same function of the same `masked_m`, so the GEMM sees identical
    arguments and CUTLASS is deterministic for a fixed launch. Anything other
    than bit identity means the two lists disagree. Run on all four published
    MoE GEMM classes and on the draw that leaves the most dead slots."""
    for shape in SLOT_SHAPES:
        for M in (1, 4, 16, 64):
            for draw in ("random", "tail"):
                y_own, cos_own, mm = slot_cell(shape, M, draw=draw, use_list=False)
                y_fed, cos_fed, mm2 = slot_cell(shape, M, draw=draw, use_list=True)
                assert torch.equal(mm, mm2), f"{shape} M={M} {draw}: routing draw drifted"
                a = slot_valid_rows(y_own, mm)
                b = slot_valid_rows(y_fed, mm2)
                assert torch.equal(a, b), (
                    f"{shape} M={M} draw={draw}: the routing-emitted slot list changed the "
                    f"output in {int((a != b).sum())} of {a.numel()} valid elements")
                assert abs(cos_own - cos_fed) < 1e-9, \
                    f"{shape} M={M} draw={draw}: cos moved {cos_own} -> {cos_fed}"
                print(f"  slot-list-eq {shape:<10} M={M:>3} draw={draw:<6} "
                      f"bit-identical, cos={cos_fed:.6f}  OK")


def slot_case_list_validation() -> None:
    """A malformed list must be refused on the host, not read by the kernel."""
    shape = "B_gate_up"
    s = SLOT_SHAPES[shape]
    G, K, N, topk = s["G"], s["K"], s["N"], s["topk"]
    M, m_cap = 4, 4
    g = torch.Generator(device="cpu").manual_seed(5)
    ids = torch.stack([torch.randperm(G, generator=g)[:topk] for _ in range(M)]
                      ).to("cuda", torch.int32)
    mm, rm, sof, se = fso.gemm.moe_build_routing(ids, G, m_cap, with_slots=True)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    _, w_fp8, sw, _ = slot_weights(G, N, K)
    a_fp8, sa = fso.gemm.quantize_1x32_grouped_gather_fp8(x, sof.int(), topk, G, m_cap)
    em, mag = 1, min(M * topk, G)

    bad = {
        "wrong length": se[: G // 2].contiguous(),
        "wrong dtype": se.to(torch.int64),
        "non-contiguous": se.repeat(2)[::2],
        "on the host": se.cpu(),
    }
    for why, t in bad.items():
        try:
            fso.gemm.linear_mxfp8_grouped_masked(a_fp8, w_fp8, sa, sw, mm, em, mag, t)
        except (RuntimeError, ValueError) as e:
            assert "slot_to_expert" in str(e), f"{why}: unhelpful message {e}"
            print(f"  slot-list-bad {why:<16} refused  OK")
        else:
            raise AssertionError(f"a {why} slot_to_expert was accepted")


SLOT_CASES = {
    "forced_sweep": (slot_case_forced_sweep, "force"),
    "hot_expert": (slot_case_hot_expert, "force"),
    "determinism": (slot_case_determinism, "force"),
    "guard_mcap": (slot_case_guard_mcap, "force"),
    "guard_bound": (slot_case_guard_bound, "force"),
    # Once with the route forced, once under the shipped rule: at m_cap = 4 the
    # rule takes the slot route for both of the layer's GEMMs, so the second
    # run is the one that covers what a caller actually gets.
    "layer_graph_hot": (slot_case_layer_graph_hot, "force"),
    "layer_graph_hot_rule": (slot_case_layer_graph_hot, None),
    # Grids larger than the SM count, where a CTA loops over several tiles and
    # the liveness of each one has to be decided separately.
    "big_grid": (slot_case_big_grid, "force"),
    "big_grid_graph": (slot_case_big_grid_graph, "force"),
    # The packed active-expert list the routing op emits (run
    # b300_mxfp8_20260917/M-I1): its contract against a torch reference, bit
    # identity of the GEMM with and without it, and the host-side validation.
    # `list_contract` is pure routing and does not care about the route knob.
    "list_contract": (slot_case_list_contract, None),
    "list_bit_identity": (slot_case_list_bit_identity, "force"),
    "list_bit_identity_rule": (slot_case_list_bit_identity, None),
    "list_validation": (slot_case_list_validation, "force"),
}


def run_slot_cases() -> None:
    """Parent side: one subprocess per case, because FSO_GROUPED_SLOT is a
    per-process static."""
    import subprocess
    import tempfile
    here = os.path.abspath(__file__)

    def child(args, knob):
        env = dict(os.environ)
        env.pop("FSO_GROUPED_SLOT", None)
        if knob is not None:
            env["FSO_GROUPED_SLOT"] = knob
        return subprocess.run([sys.executable, here, *args], env=env,
                              capture_output=True, text=True)

    for name, (_, mode) in SLOT_CASES.items():
        r = child(["--slot-case", name], mode)
        print(r.stdout, end="")
        assert r.returncode == 0, f"slot case {name} failed (exit {r.returncode}):\n{r.stderr}"

    # Which route the dispatcher picked, read off the pool that refuses to
    # allocate inside a capture. Expected: no slot bound -> pointer-array route,
    # slot bound at m_cap = 4 -> slot route, FSO_GROUPED_SLOT=0 -> pointer-array
    # route whatever the bound says.
    for knob, mag, want in ((None, "0", "argument-array pool"),
                            (None, "32", "slot-list pool"),
                            ("0", "32", "argument-array pool")):
        r = child(["--slot-case", "route_pick", "--mag", mag], knob)
        assert r.returncode != 0, \
            f"route_pick (knob={knob}, mag={mag}) allocated during capture instead of refusing"
        assert want in r.stderr, \
            f"route_pick (knob={knob}, mag={mag}) expected the {want} to refuse, got:\n{r.stderr[-600:]}"
        print(f"  slot-route  FSO_GROUPED_SLOT={knob or '<unset>'} max_active_groups={mag} "
              f"-> {want}  OK")

    with tempfile.TemporaryDirectory() as td:
        ref = os.path.join(td, "route_off_ref.pt")
        for knob, write in (("0", "1"), (None, "0")):
            r = child(["--slot-case", "route_off", "--ref", ref, "--write-ref", write], knob)
            print(r.stdout, end="")
            assert r.returncode == 0, \
                f"slot case route_off (knob={knob}) failed (exit {r.returncode}):\n{r.stderr}"


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--quant-once-case":
        quant_once_child()
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--fused-case":
        print(f"device: {torch.cuda.get_device_name()}  fused case={sys.argv[2]} "
              f"FSO_FC1_FUSED={os.environ.get('FSO_FC1_FUSED', '<unset>')}")
        test_fused_route_helper()
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--slot-case":
        case = sys.argv[2]
        print(f"device: {torch.cuda.get_device_name()}  case={case} "
              f"FSO_GROUPED_SLOT={os.environ.get('FSO_GROUPED_SLOT', '<unset>')}")
        if case == "route_off":
            ref = sys.argv[sys.argv.index("--ref") + 1]
            write = sys.argv[sys.argv.index("--write-ref") + 1] == "1"
            slot_case_route_off(ref, write)
        elif case == "route_pick":
            slot_case_route_pick(int(sys.argv[sys.argv.index("--mag") + 1]))
        else:
            SLOT_CASES[case][0]()
        return 0

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

    if major == 10:
        print("== gather-quant pair-space vs token-scatter, bit identity "
              "(subprocess per knob setting) ==")
        run_quant_once_cases()

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

    if major == 10:
        print("== sm_100/103 fused-SwiGLU FC1 ==")
        run_fused_cases()

    if major == 10:
        print("== sm_100/103 slot-bound decode route (subprocess per knob setting) ==")
        run_slot_cases()

    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
