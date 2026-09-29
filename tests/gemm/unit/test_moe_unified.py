#!/usr/bin/env python3
"""The unified MoE surface fso.moe (prepare_experts / layer / supported / describe)
against the per-architecture entries it dispatches to.

fso.moe.layer is one torch custom op whose body picks the architecture's chain, so
the property worth asserting is that going through it changes nothing: on every
architecture it has to produce exactly the tensor the per-architecture entry
produces for the same weights and inputs. The cases, each run where the hardware
exists and skipped with a message elsewhere:

  a.  sm_120/121: layer() on prepare_experts(format="mxfp8") weights is torch.equal
      to moe_layer_mxfp8_sm120(..., w13_interleaved=True), for M in
      {1, 4, 8, 64, 512, 2048}, with and without bias / bias_scale, with ids that
      contain -1 and E (skipped), for int32 and int64 ids; the prepared weights are
      byte-identical to quantize_moe_weights_1x32_fp8's; a CUDA graph captured once
      replays equal to eager, also after the routing is rewritten in place.
  a2. sm_120/121: the fused combine is opt-in. At M = 4096 the engagement rule would
      take it, so with the knob unset layer() must still be torch.equal to the
      deterministic entry; with FSO_MOE_FUSED_COMBINE=1 it is compared to the
      deterministic entry and to moe_layer_mxfp8_sm120(fused_combine=True) to a
      tolerance and its run-to-run spread is reported. The knob is read once per
      process into a module-level cache, and the RTX 5090 cards run in
      exclusive-process compute mode, where a child process cannot open a context
      beside this one, so the on case resets that cache in-process rather than
      starting a subprocess.
  a100. sm_100/103 (written for the B300 pod, not run on sm_120 or sm_90): the same
      grid as (a), with the reference being the chain
      bench/gemm/python/bench_moe_qwen3_30a3.py composes (fso_mxfp8_layer): the
      slot list and the per-GEMM problem shapes requested exactly where the route
      queries say a GEMM reads them.
  b.  sm_100/103 and sm_120/121: prepare_experts(format="bsfp8") from block-FP8
      experts (128x128 blocks, scale = amax / 448, value = fp8 * scale) runs through
      a second quantization to MXFP8 1x32. layer() is compared to an fp32 torch
      reference on the DEQUANTIZED block-FP8 weights (cosine and max-abs per M), next
      to the one-pass calibration (bf16 masters quantized once to MXFP8, against the
      fp32 reference on the masters, the class COS_LAYER = 0.997 in
      test_mxfp8_grouped.py was set for) and the end-to-end error from the bf16
      masters through both quantizations.
  c.  sm_90: layer() on prepare_experts(format="bsfp8") (the tensors kept as they
      are) is torch.equal to moe_layer_fp8_sm90 for M in {1, 8, 64, 512, 2048} with
      clean and skipped ids and both id dtypes; the bias fold is torch.equal to
      ref + bias, and to ref + bias_scale[:, None] * bias evaluated exactly and rounded
      once to fp32 and once to bf16 (the fused pass is an FMA; the literal torch
      expression, which rounds the product first, is reported next to it); a graph
      replays equal to eager; format="mxfp8" raises NotImplementedError naming sm_90.
  d.  every architecture: the supported() matrix, format errors (bf16 and unknown
      formats), shape / dtype / device errors in prepare_experts and argument errors
      in layer(), each message naming the architecture.
  e.  every architecture: a function that calls layer() compiles with
      torch.compile(fullgraph=True) — the custom op is opaque to dynamo, so the
      compiled graph holds it as one node, typed by its fake implementation — and the
      compiled call is torch.equal to the eager one.

Families: C = Qwen3.5-35B-A3B routed experts (E 256, top-8, H 2048, I 512) and
B = Qwen3-30B-A3B (E 128, top-8, H 2048, I 768).
"""
import dataclasses
import os
import sys

import torch
import torch.nn.functional as F

import fish_scales_ops as fso

FAMILIES = {
    # label: (E, topk, hidden, inter)
    "C_35a3": (256, 8, 2048, 512),
    "B_30a3": (128, 8, 2048, 768),
}
M_MXFP8 = (1, 4, 8, 64, 512, 2048)
M_SM90 = (1, 8, 64, 512, 2048)

# Gates for case (b), set from the calibration run on the RTX 5090 on 2026-09-29
# (run fso_moe_unified_20260929/dev1): over both families and all six M, the double
# quantization (block-FP8 -> bf16 -> MXFP8 1x32) against the fp32 reference on the
# dequantized block-FP8 weights measured cosine 0.99771-0.99791, the same class as one
# MXFP8 pass against its bf16 masters (0.99773-0.99790), so it takes COS_LAYER's gate
# and margin; the largest max |y - ref| / max |ref| was 0.076, gated at 0.12 because a
# maximum over a few thousand outputs is a noisy statistic at small M.
COS_LAYER = 0.997            # one MXFP8 pass vs the bf16 masters (test_mxfp8_grouped.py)
COS_DOUBLE_QUANT = 0.997     # bsfp8 -> MXFP8 vs the dequantized block-FP8 weights
REL_MAXABS_DOUBLE_QUANT = 0.12  # max |y - ref| / max |ref|, same comparison
# Case (a2): the fused combine against the deterministic combine at the same weights.
# The two differ in accumulation order and in where the bf16 roundings happen (the
# fused FC2 adds each routed row into the bf16 output, pre-filled with the bias term),
# so this is the gate test_mxfp8_fused_combine_sm120.py applies to the same comparison.
COS_FUSED_COMBINE = 0.9999

failures: list[str] = []


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()


def _maxabs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _check(ok: bool, msg: str) -> bool:
    if not ok:
        failures.append(msg)
    return ok


def _arch_label() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return f"sm_{major}{minor}"


def make_weights(E, H, I, seed):
    """bf16 masters with the magnitudes test_mxfp8_grouped.py calibrates COS_LAYER on."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    w13 = (torch.randn(E, 2 * I, H, device="cuda", generator=g) / H ** 0.5).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device="cuda", generator=g) / I ** 0.5).to(torch.bfloat16)
    return w13, w2


def make_inputs(M, E, TOPK, H, seed):
    """hidden, int64 ids (torch.topk's own dtype, distinct per token), fp32 weights,
    a skipped-id variant, a bias and a per-token bias scale."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden = (torch.randn(M, H, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    ids = torch.rand(M, E, device="cuda", generator=g).topk(TOPK, dim=1).indices
    wts = torch.softmax(torch.rand(M, TOPK, device="cuda", generator=g), dim=1).float()
    bias = (torch.randn(M, H, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    bscale = torch.sigmoid(torch.randn(M, device="cuda", generator=g)).float()
    # Skipped entries: -1 (another expert-parallel rank's expert, or sglang's padded
    # row) and E (the padded-row sentinel). Every fourth token is skipped entirely,
    # which must come out as its bias term or as zero.
    skip = ids.clone()
    if M == 1:
        skip[0, 0] = -1
        skip[0, TOPK - 1] = E
    else:
        t = torch.arange(M, device="cuda")
        skip[t % 4 == 0] = E
        r1 = (t % 4 == 1).nonzero().flatten()
        skip[r1, (r1 % TOPK)] = -1
        r2 = (t % 4 == 2).nonzero().flatten()
        skip[r2, TOPK - 1] = E
        skip[r2, 0] = -1
    return hidden, ids, wts, skip, bias, bscale


def block_quantize_fp8(w: torch.Tensor):
    """bf16 [E, N, K] -> (float8_e4m3fn [E, N, K], fp32 [E, N/128, K/128]): one scale
    per 128x128 block, scale = amax / 448, stored value * scale = weight (the
    bsgemm-moe / DeepSeek block-FP8 convention)."""
    E, N, K = w.shape
    q = torch.empty(E, N, K, dtype=torch.float8_e4m3fn, device=w.device)
    s = torch.empty(E, N // 128, K // 128, dtype=torch.float32, device=w.device)
    for e0 in range(0, E, 16):
        e1 = min(E, e0 + 16)
        wf = w[e0:e1].float().view(e1 - e0, N // 128, 128, K // 128, 128)
        amax = wf.abs().amax(dim=(2, 4))
        sc = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
        q[e0:e1] = (wf / sc[:, :, None, :, None]).view(e1 - e0, N, K).to(torch.float8_e4m3fn)
        s[e0:e1] = sc
    return q, s


def block_dequantize(q: torch.Tensor, s: torch.Tensor, e: int) -> torch.Tensor:
    N, K = int(q.shape[1]), int(q.shape[2])
    return (q[e].float().view(N // 128, 128, K // 128, 128)
            * s[e].view(N // 128, 1, K // 128, 1)).view(N, K)


def ref_layer(hidden, ids, wts, w13_of, w2_of, E, bias=None, bias_scale=None):
    """fp32 torch reference: sum over each token's in-range ids of
    w * (silu(x Wg^T) * (x Wu^T)) Wd^T, with w13_of(e) = [gate; up] fp32 and
    w2_of(e) = down fp32; skipped ids contribute nothing."""
    M, H = hidden.shape
    out = torch.zeros(M, H, dtype=torch.float32, device=hidden.device)
    hf = hidden.float()
    ids = ids.long()
    valid = (ids >= 0) & (ids < E)
    for e in torch.unique(ids[valid]).tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        w13e = w13_of(e)
        w2e = w2_of(e)
        inter = w2e.shape[1]
        gu = hf[tok] @ w13e.t()
        act = F.silu(gu[:, :inter]) * gu[:, inter:]
        out.index_add_(0, tok, (act @ w2e.t()) * wts[tok, slot, None].float())
    if bias is not None:
        out += bias.float() * (1.0 if bias_scale is None else bias_scale.float()[:, None])
    return out


BIAS_MODES = ("none", "bias", "bias+scale")


def bias_kwargs(mode, bias, bscale):
    if mode == "none":
        return {}
    if mode == "bias":
        return {"bias": bias}
    return {"bias": bias, "bias_scale": bscale}


def skipped_rows_ok(out, skip, E, mode, bias, bscale) -> bool:
    """A token whose every id is skipped comes out as its bias term, or zero."""
    dead = ((skip < 0) | (skip >= E)).all(dim=1)
    if not bool(dead.any()):
        return True
    if mode == "none":
        want = torch.zeros_like(out[dead])
    elif mode == "bias":
        want = bias[dead]
    else:
        want = (bias[dead].float() * bscale[dead, None]).to(torch.bfloat16)
    return torch.equal(out[dead], want)


def bench_chain_sm100(hidden, ex, ids32, wts, bias=None, bias_scale=None):
    """The sm_100/103 layer as bench/gemm/python/bench_moe_qwen3_30a3.py composes it
    (fso_mxfp8_layer), through the public per-step ops, plus the combine's bias."""
    E, H, I = ex.num_experts, ex.hidden, ex.inter
    M, TOPK = int(hidden.shape[0]), int(ids32.shape[1])
    m_cap = (M + 3) // 4 * 4
    expected_m = max(1, (M * TOPK + E - 1) // E)
    mag = min(M * TOPK, E)
    n_w = 2 * I
    fc1_fused = ex.w13_interleaved and fso.gemm.mxfp8_grouped_swiglu_fused_route(m_cap, n_w, H, E, mag)
    want_slots = (fso.gemm.mxfp8_grouped_slot_possible(m_cap, n_w, H, E, mag, fused_swiglu=fc1_fused)
                  or fso.gemm.mxfp8_grouped_slot_possible(m_cap, H, I, E, mag))
    want_ps = [fso.gemm.mxfp8_grouped_problem_shapes_consumed(m_cap, n_w, H, E, mag, fused_swiglu=fc1_fused),
               fso.gemm.mxfp8_grouped_problem_shapes_consumed(m_cap, H, I, E, mag)]
    ps_for = [nk for nk, want in zip([(n_w, H), (H, I)], want_ps) if want]
    r = fso.gemm.moe_build_routing(ids32, E, m_cap, with_slots=want_slots, problem_shapes_for=ps_for or None)
    ps = list(r[-1]) if ps_for else []
    ps1 = ps.pop(0) if want_ps[0] else None
    ps2 = ps.pop(0) if want_ps[1] else None
    masked, slot_of_flat = r[0], r[2]
    slot_to_expert = r[3] if want_slots else None
    hq, sh = fso.gemm.quantize_1x32_grouped_gather_fp8(hidden, slot_of_flat, TOPK, E, m_cap)
    if fc1_fused:
        dq, sd = fso.gemm.linear_mxfp8_grouped_masked_swiglu(
            hq, ex.w13, sh, ex.sw13, masked, expected_m, mag, slot_to_expert, ps1)
    else:
        gu = fso.gemm.linear_mxfp8_grouped_masked(
            hq, ex.w13, sh, ex.sw13, masked, expected_m, mag, slot_to_expert, ps1)
        dq, sd = fso.gemm.silu_chunk_mul_quantize_1x32_grouped_fp8(gu, slot_of_flat, pairwise=ex.w13_interleaved)
    dn = fso.gemm.linear_mxfp8_grouped_masked(dq, ex.w2, sd, ex.sw2, masked, expected_m, mag, slot_to_expert, ps2)
    return fso.gemm.moe_combine(dn, slot_of_flat, wts, bias=bias, bias_scale=bias_scale)


def graph_case(label, ex, M, E, TOPK, H, seed):
    """Capture layer() once (int64 ids, bias and bias_scale, so the id narrowing and
    the fold are inside the graph), replay against eager, rewrite the routing in
    place and replay again."""
    hidden, ids, wts, skip, bias, bscale = make_inputs(M, E, TOPK, H, seed)
    ids_buf = ids.clone()
    w_buf = wts.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fso.moe.layer(hidden, ex, ids_buf, w_buf, bias=bias, bias_scale=bscale)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        out_g = fso.moe.layer(hidden, ex, ids_buf, w_buf, bias=bias, bias_scale=bscale)
    torch.cuda.synchronize()
    out_g.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    eager = fso.moe.layer(hidden, ex, ids_buf, w_buf, bias=bias, bias_scale=bscale)
    torch.cuda.synchronize()
    ok1 = torch.equal(out_g, eager)
    # New routing, with skipped entries, written into the captured buffers.
    _, ids2, wts2, skip2, _, _ = make_inputs(M, E, TOPK, H, seed + 7)
    ids_buf.copy_(skip2)
    w_buf.copy_(wts2)
    graph.replay()
    torch.cuda.synchronize()
    eager2 = fso.moe.layer(hidden, ex, ids_buf, w_buf, bias=bias, bias_scale=bscale)
    torch.cuda.synchronize()
    ok2 = torch.equal(out_g, eager2)
    _check(ok1 and ok2, f"{label} M={M}: graph replay {'==' if ok1 else '!='} eager, "
                        f"after reroute {'==' if ok2 else '!='} eager")
    print(f"  graph   {label:8s} M={M:5d}: replay {'==' if ok1 else '!='} eager; routing rewritten in "
          f"place (with skipped ids), replay {'==' if ok2 else '!='} eager  {'OK' if ok1 and ok2 else 'FAIL'}")
    del graph


# --- (a) / (a100) ------------------------------------------------------------------

def case_a(major: int) -> None:
    for fam, (E, TOPK, H, I) in FAMILIES.items():
        w13, w2 = make_weights(E, H, I, seed=11 + E + I)
        ex = fso.moe.prepare_experts(w13, w2, format="mxfp8")
        interleave = fso.gemm.mxfp8_grouped_swiglu_available(2 * I, H)
        _check(ex.kind == "mxfp8" and ex.arch // 10 == major and ex.num_experts == E and ex.hidden == H
               and ex.inter == I and ex.w13_interleaved == interleave,
               f"{fam}: prepare_experts(format='mxfp8') returned {ex!r}")
        w13q, s13 = fso.gemm.quantize_moe_weights_1x32_fp8(w13, w13_interleave=interleave)
        w2q, s2 = fso.gemm.quantize_moe_weights_1x32_fp8(w2)
        same_w = (torch.equal(ex.w13.view(torch.uint8), w13q.view(torch.uint8)) and torch.equal(ex.sw13, s13)
                  and torch.equal(ex.w2.view(torch.uint8), w2q.view(torch.uint8)) and torch.equal(ex.sw2, s2))
        _check(same_w, f"{fam}: prepared MXFP8 weights differ from quantize_moe_weights_1x32_fp8's")
        print(f"  prepare {fam:8s}: kind={ex.kind} arch=sm_{ex.arch} w13_interleaved={ex.w13_interleaved}; "
              f"weights and scale words byte-identical to quantize_moe_weights_1x32_fp8  "
              f"{'OK' if same_w else 'FAIL'}")
        del w13, w2
        for M in M_MXFP8:
            hidden, ids, wts, skip, bias, bscale = make_inputs(M, E, TOPK, H, seed=1234 + M)
            n_eq = n_all = 0
            dead_ok = True
            for ids_name, ids64 in (("clean", ids), ("skipped", skip)):
                ids32 = ids64.to(torch.int32)
                for mode in BIAS_MODES:
                    kw = bias_kwargs(mode, bias, bscale)
                    if major == 12:
                        ref = fso.gemm.moe_layer_mxfp8_sm120(
                            hidden, w13q, s13, w2q, s2, ids32, wts, w13_interleaved=interleave, **kw)
                    else:
                        ref = bench_chain_sm100(hidden, ex, ids32, wts, **kw)
                    for id_dtype, idt in (("int32", ids32), ("int64", ids64)):
                        out = fso.moe.layer(hidden, ex, idt, wts, **kw)
                        eq = torch.equal(out, ref)
                        n_all += 1
                        n_eq += int(eq)
                        _check(eq, f"{fam} M={M} ids={ids_name}/{id_dtype} {mode}: layer != reference "
                                   f"(max |diff| {_maxabs(out, ref):.3e})")
                        if ids_name == "skipped" and id_dtype == "int64":
                            dead_ok &= skipped_rows_ok(out, skip, E, mode, bias, bscale)
            _check(dead_ok, f"{fam} M={M}: a fully skipped token is not its bias term / zero")
            n_dead = int(((skip < 0) | (skip >= E)).all(dim=1).sum())
            what = "moe_layer_mxfp8_sm120" if major == 12 else "the bench chain"
            print(f"  equal   {fam:8s} M={M:5d}: {n_eq}/{n_all} torch.equal to {what} (ids clean/skipped x "
                  f"int32/int64 x bias none/bias/bias+scale); {n_dead} fully skipped tokens = bias term / zero "
                  f"{'yes' if dead_ok else 'NO'}  {'OK' if n_eq == n_all and dead_ok else 'FAIL'}")
        for M in (8, 512):
            graph_case(fam, ex, M, E, TOPK, H, seed=777 + M)
        del ex, w13q, s13, w2q, s2
        torch.cuda.empty_cache()


# --- (a2) ---------------------------------------------------------------------------

def a2_inputs():
    E, TOPK, H, I = FAMILIES["C_35a3"]
    M = 4096
    w13, w2 = make_weights(E, H, I, seed=4242)
    hidden, ids, wts, skip, bias, bscale = make_inputs(M, E, TOPK, H, seed=4243)
    return E, TOPK, H, I, M, w13, w2, hidden, ids, wts, bias, bscale


def case_a2() -> None:
    E, TOPK, H, I, M, w13, w2, hidden, ids, wts, bias, bscale = a2_inputs()
    engages = fso.gemm.moe_layer_fused_combine_engages_sm120(M, TOPK, H)
    _check(engages, f"a2: the fused-combine rule does not engage at M={M}; pick a larger M")
    ex = fso.moe.prepare_experts(w13, w2, format="mxfp8")
    interleave = ex.w13_interleaved
    del w13, w2
    ids32 = ids.to(torch.int32)
    # Default (knob unset): the fused combine is opt-in, so the bucket the rule would
    # take must still be the deterministic entry bit for bit.
    _check(not fso.moe._fused_combine_allowed(), "a2: FSO_MOE_FUSED_COMBINE is set in this environment; "
           "the default case needs it unset")
    for mode in ("none", "bias+scale"):
        kw = bias_kwargs(mode, bias, bscale)
        det = fso.gemm.moe_layer_mxfp8_sm120(hidden, ex.w13, ex.sw13, ex.w2, ex.sw2, ids32, wts,
                                             w13_interleaved=interleave, **kw)
        u = fso.moe.layer(hidden, ex, ids, wts, **kw)
        torch.cuda.synchronize()
        eq = torch.equal(u, det)
        _check(eq, f"a2 M={M} {mode}: with FSO_MOE_FUSED_COMBINE unset the layer is not the deterministic "
               f"entry (max|d|={_maxabs(u, det):.3e})")
        print(f"  fused-combine C_35a3 M={M} {mode:10s} knob unset: rule would engage={engages}; layer "
              f"torch.equal to the deterministic entry  OK")
    # Opt-in (FSO_MOE_FUSED_COMBINE=1): the rule engages, the adds are atomic, so the
    # layer is compared to a tolerance and its run-to-run spread is reported. The knob
    # is read once per process into fso.moe._fused_combine_env; the check sets the
    # variable, clears that cache so the next layer() call parses it again, and
    # restores both afterwards.
    saved_env = os.environ.get("FSO_MOE_FUSED_COMBINE")
    saved_cache = fso.moe._fused_combine_env
    try:
        os.environ["FSO_MOE_FUSED_COMBINE"] = "1"
        fso.moe._fused_combine_env = None
        _check(fso.moe._fused_combine_allowed(), "a2: FSO_MOE_FUSED_COMBINE=1 did not enable the fused combine")
        for mode in ("none", "bias+scale"):
            kw = bias_kwargs(mode, bias, bscale)
            det = fso.gemm.moe_layer_mxfp8_sm120(hidden, ex.w13, ex.sw13, ex.w2, ex.sw2, ids32, wts,
                                                 w13_interleaved=interleave, **kw)
            fc = fso.gemm.moe_layer_mxfp8_sm120(hidden, ex.w13, ex.sw13, ex.w2, ex.sw2, ids32, wts,
                                                w13_interleaved=interleave, fused_combine=True, **kw)
            u1 = fso.moe.layer(hidden, ex, ids, wts, **kw)
            u2 = fso.moe.layer(hidden, ex, ids, wts, **kw)
            torch.cuda.synchronize()
            c_det, m_det = _cos(u1, det), _maxabs(u1, det)
            c_fc, m_fc = _cos(u1, fc), _maxabs(u1, fc)
            spread = _maxabs(u1, u2)
            ok = c_det >= COS_FUSED_COMBINE and c_fc >= COS_FUSED_COMBINE
            _check(ok, f"a2 M={M} {mode} knob=1: cos vs deterministic {c_det:.7f}, vs fused entry {c_fc:.7f}")
            print(f"  fused-combine C_35a3 M={M} {mode:10s} knob=1: layer vs deterministic entry "
                  f"cos={c_det:.7f} max|d|={m_det:.3e} (bit-equal {torch.equal(u1, det)}); vs "
                  f"moe_layer_mxfp8_sm120(fused_combine=True) cos={c_fc:.7f} max|d|={m_fc:.3e}; run-to-run "
                  f"max|d|={spread:.3e}  {'OK' if ok else 'FAIL'}")
    finally:
        if saved_env is None:
            os.environ.pop("FSO_MOE_FUSED_COMBINE", None)
        else:
            os.environ["FSO_MOE_FUSED_COMBINE"] = saved_env
        fso.moe._fused_combine_env = saved_cache


# --- (b) ----------------------------------------------------------------------------

def case_b(major: int) -> None:
    print(f"  {'family':8s} {'M':>5s} | {'bsfp8->MXFP8 vs deq. block-FP8 (rel = max|d| / max|ref|)':>45s} | "
          f"{'one MXFP8 pass vs bf16':>24s} | {'bsfp8->MXFP8 vs bf16':>24s} | ref max|y|")
    for fam, (E, TOPK, H, I) in FAMILIES.items():
        w13, w2 = make_weights(E, H, I, seed=31 + E + I)
        q13, s13 = block_quantize_fp8(w13)
        q2, s2 = block_quantize_fp8(w2)
        ex_b = fso.moe.prepare_experts(q13, q2, format="bsfp8", sw13=s13, sw2=s2)
        ex_m = fso.moe.prepare_experts(w13, w2, format="mxfp8")
        _check(ex_b.kind == "mxfp8" and ex_b.arch // 10 == major and ex_b.w13_interleaved == ex_m.w13_interleaved,
               f"{fam}: prepare_experts(format='bsfp8') on sm_{major}x returned {ex_b!r}")
        for M in M_MXFP8:
            hidden, ids, wts, _skip, _b, _s = make_inputs(M, E, TOPK, H, seed=99 + M)
            y_b = fso.moe.layer(hidden, ex_b, ids, wts)
            y_m = fso.moe.layer(hidden, ex_m, ids, wts)
            ref_deq = ref_layer(hidden, ids, wts, lambda e: block_dequantize(q13, s13, e),
                                lambda e: block_dequantize(q2, s2, e), E)
            ref_bf = ref_layer(hidden, ids, wts, lambda e: w13[e].float(), lambda e: w2[e].float(), E)
            c_dq, a_dq = _cos(y_b, ref_deq), _maxabs(y_b, ref_deq)
            c_1, a_1 = _cos(y_m, ref_bf), _maxabs(y_m, ref_bf)
            c_e2e, a_e2e = _cos(y_b, ref_bf), _maxabs(y_b, ref_bf)
            ref_max = ref_deq.abs().max().item()
            rel = a_dq / max(ref_max, 1e-30)
            ok = (c_dq >= COS_DOUBLE_QUANT and rel <= REL_MAXABS_DOUBLE_QUANT and c_1 >= COS_LAYER)
            _check(ok, f"{fam} M={M}: double-quant cos={c_dq:.6f} rel max-abs={rel:.4f}, one-pass cos={c_1:.6f}")
            print(f"  {fam:8s} {M:5d} | cos={c_dq:.6f} max|d|={a_dq:.3e} rel={rel:.4f} | cos={c_1:.6f} "
                  f"max|d|={a_1:.3e} | cos={c_e2e:.6f} max|d|={a_e2e:.3e} | {ref_max:.3e}  {'OK' if ok else 'FAIL'}")
        del ex_b, ex_m, q13, s13, q2, s2, w13, w2
        torch.cuda.empty_cache()


# --- (c) ----------------------------------------------------------------------------

def case_c() -> None:
    for fam, (E, TOPK, H, I) in FAMILIES.items():
        w13, w2 = make_weights(E, H, I, seed=51 + E + I)
        q13, s13 = block_quantize_fp8(w13)
        q2, s2 = block_quantize_fp8(w2)
        ex = fso.moe.prepare_experts(q13, q2, format="bsfp8", sw13=s13, sw2=s2)
        kept = (ex.kind == "bsfp8" and not ex.w13_interleaved and ex.w13.data_ptr() == q13.data_ptr()
                and ex.sw13.data_ptr() == s13.data_ptr() and ex.w2.data_ptr() == q2.data_ptr()
                and ex.sw2.data_ptr() == s2.data_ptr())
        _check(kept, f"{fam}: sm_90 prepare_experts(format='bsfp8') did not keep the tensors: {ex!r}")
        print(f"  prepare {fam:8s}: kind={ex.kind} arch=sm_{ex.arch}; the block-FP8 tensors are kept as they "
              f"are (same storage)  {'OK' if kept else 'FAIL'}")
        for M in M_SM90:
            hidden, ids, wts, skip, bias, bscale = make_inputs(M, E, TOPK, H, seed=2345 + M)
            n_eq = n_all = 0
            fold_ok = True
            dead_ok = True
            two_round_diff = 0
            two_round_max = 0.0
            for ids_name, ids64 in (("clean", ids), ("skipped", skip)):
                ids32 = ids64.to(torch.int32)
                ref = fso.gemm.moe_layer_fp8_sm90(hidden, q13, s13, q2, s2, ids32, wts)
                # The fold is one fused elementwise pass, so ref + bias_scale * bias is
                # rounded once in fp32 (as an FMA) and then to bf16. The reference
                # therefore evaluates ref + bias_scale[:, None] * bias exactly (in fp64,
                # where the product of an fp32 and a bf16 value is exact) and rounds it
                # the same two times. The literal torch expression rounds the product to
                # fp32 before the add; the elements where that extra rounding moves the
                # bf16 result are counted and reported, not gated.
                single = (ref.double() + bscale.double()[:, None] * bias.double()).float().to(torch.bfloat16)
                literal = (ref + bscale[:, None] * bias).to(torch.bfloat16)
                refs = {"none": ref, "bias": ref + bias, "bias+scale": single}
                for mode in BIAS_MODES:
                    kw = bias_kwargs(mode, bias, bscale)
                    for id_dtype, idt in (("int32", ids32), ("int64", ids64)):
                        out = fso.moe.layer(hidden, ex, idt, wts, **kw)
                        eq = torch.equal(out, refs[mode])
                        n_all += 1
                        n_eq += int(eq)
                        if mode != "none":
                            fold_ok &= eq
                        _check(eq, f"{fam} M={M} ids={ids_name}/{id_dtype} {mode}: layer != "
                                   f"{'moe_layer_fp8_sm90' if mode == 'none' else 'the bias fold reference'} "
                                   f"(max |diff| {_maxabs(out, refs[mode]):.3e})")
                        if ids_name == "skipped" and id_dtype == "int64":
                            dead_ok &= skipped_rows_ok(out, skip, E, mode, bias, bscale)
                        if mode == "bias+scale" and id_dtype == "int32":
                            two_round_diff += int((out != literal).sum())
                            two_round_max = max(two_round_max, _maxabs(out, literal))
            _check(dead_ok, f"{fam} M={M}: a fully skipped token is not its bias term / zero")
            # Context, not a gate: the block-FP8 layer against the fp32 reference on
            # the dequantized weights it serves.
            y = fso.moe.layer(hidden, ex, ids, wts)
            ref32 = ref_layer(hidden, ids, wts, lambda e: block_dequantize(q13, s13, e),
                              lambda e: block_dequantize(q2, s2, e), E)
            bn = fso.gemm.moe_swap_ab_block_n(M, E, TOPK)
            path = f"swap-AB/{bn}" if bn is not None else "contig"
            n_dead = int(((skip < 0) | (skip >= E)).all(dim=1).sum())
            print(f"  equal   {fam:8s} M={M:5d} {path:10s}: {n_eq}/{n_all} torch.equal (moe_layer_fp8_sm90; bias "
                  f"fold == ref + bias and == ref + bias_scale[:, None] * bias rounded once: "
                  f"{'yes' if fold_ok else 'NO'}; the literal two-rounding expression differs in "
                  f"{two_round_diff} of {2 * M * H} elements, max|d|={two_round_max:.3e}); {n_dead} fully skipped "
                  f"tokens = bias term / zero {'yes' if dead_ok else 'NO'}; vs fp32 on the dequantized weights "
                  f"cos={_cos(y, ref32):.6f} max|d|={_maxabs(y, ref32):.3e}  "
                  f"{'OK' if n_eq == n_all and dead_ok else 'FAIL'}")
        for M in (8, 2048):
            graph_case(fam, ex, M, E, TOPK, H, seed=888 + M)
        # format="mxfp8" is refused on sm_90, naming it.
        try:
            fso.moe.prepare_experts(w13, w2, format="mxfp8")
            _check(False, f"{fam}: prepare_experts(format='mxfp8') did not raise on sm_90")
            print(f"  refuse  {fam:8s}: format='mxfp8' did not raise  FAIL")
        except NotImplementedError as exc:
            ok = "sm_90" in str(exc)
            _check(ok, f"{fam}: the sm_90 mxfp8 refusal does not name sm_90: {exc}")
            print(f"  refuse  {fam:8s}: format='mxfp8' -> NotImplementedError naming sm_90 "
                  f"{'yes' if ok else 'NO'}  {'OK' if ok else 'FAIL'}")
        del ex, q13, s13, q2, s2, w13, w2
        torch.cuda.empty_cache()


# --- (d) ----------------------------------------------------------------------------

def _expect_raise(label, exc_types, fn):
    arch = _arch_label()
    try:
        fn()
    except exc_types as exc:
        named = arch in str(exc)
        _check(named, f"{label}: {type(exc).__name__} does not name {arch}: {exc}")
        return f"{type(exc).__name__}{'' if named else ' (arch NOT named)'}"
    except Exception as exc:  # noqa: BLE001
        _check(False, f"{label}: raised {type(exc).__name__} instead of {exc_types}: {exc}")
        return f"wrong exception {type(exc).__name__}"
    _check(False, f"{label}: did not raise")
    return "did not raise"


def case_d(major: int) -> None:
    arch = _arch_label()
    expect = {"bsfp8": {9, 10, 12}, "mxfp8": {10, 12}}
    archs = {"sm_90": 9, "sm_90a": 9, "sm_100": 10, "sm_100a": 10, "sm_103": 10, "sm_120": 12, "sm_121": 12,
             "sm_80": 8, "sm_89": 8, 9: 9, 10: 10, 12: 12, 90: 9, 103: 10, 120: 12, (12, 0): 12, "12.0": 12}
    bad = 0
    for a, maj in archs.items():
        for f in ("bsfp8", "mxfp8", "BSFP8", "bf16", "int4", "nonsense", "", None, 3):
            want = isinstance(f, str) and maj in expect.get(f.strip().lower(), set())
            got = fso.moe.supported(f, a)
            if got != want:
                bad += 1
                _check(False, f"supported({f!r}, {a!r}) = {got}, want {want}")
    here = {f: fso.moe.supported(f) for f in ("bsfp8", "mxfp8", "bf16")}
    here_ok = here == {"bsfp8": major in (9, 10, 12), "mxfp8": major in (10, 12), "bf16": False}
    _check(here_ok, f"supported() on this device ({arch}) = {here}")
    try:
        fso.moe.supported("bsfp8", "gpu")
        arch_err = "no error"
        _check(False, "supported('bsfp8', 'gpu') did not raise")
    except ValueError:
        arch_err = "ValueError"
    print(f"  supported: {len(archs)} arch spellings x 9 formats, {bad} mismatches; this device {arch}: {here}; "
          f"arch='gpu' -> {arch_err}  {'OK' if bad == 0 and here_ok else 'FAIL'}")
    text = fso.moe.describe()
    desc_ok = arch in text and "this device" in text and "bsfp8" in text and "mxfp8" in text
    _check(desc_ok, f"describe() does not name this device ({arch}):\n{text}")
    print(f"  describe: {len(text.splitlines())} lines, names {arch} and marks its rows  {'OK' if desc_ok else 'FAIL'}")

    # Tiny layer: E 4, H 256, I 128.
    E, H, I = 4, 256, 128
    w13, w2 = make_weights(E, H, I, seed=5)
    q13, s13 = block_quantize_fp8(w13)
    q2, s2 = block_quantize_fp8(w2)
    P = fso.moe.prepare_experts
    results = {
        "format='bf16'": _expect_raise("format bf16", NotImplementedError, lambda: P(w13, w2, format="bf16")),
        "format='int4'": _expect_raise("format int4", NotImplementedError, lambda: P(w13, w2, format="int4")),
        "format='nonsense'": _expect_raise("format nonsense", ValueError, lambda: P(w13, w2, format="nonsense")),
        "format=None": _expect_raise("format None", ValueError, lambda: P(w13, w2, format=None)),
        "bsfp8 with bf16 weights": _expect_raise(
            "bsfp8 bf16 weights", ValueError, lambda: P(w13, w2, format="bsfp8", sw13=s13, sw2=s2)),
        "bsfp8 without scales": _expect_raise("bsfp8 no scales", ValueError, lambda: P(q13, q2, format="bsfp8")),
        "bsfp8 sw13 wrong shape": _expect_raise(
            "bsfp8 sw13 shape", ValueError, lambda: P(q13, q2, format="bsfp8", sw13=s13[:, :1], sw2=s2)),
        "bsfp8 sw2 fp16": _expect_raise(
            "bsfp8 sw2 dtype", ValueError, lambda: P(q13, q2, format="bsfp8", sw13=s13, sw2=s2.half())),
        "w13 rows != 2*I": _expect_raise(
            "w13 rows", ValueError, lambda: P(q13[:, :128], q2, format="bsfp8", sw13=s13[:, :1], sw2=s2)),
        "expert counts differ": _expect_raise(
            "expert counts", ValueError, lambda: P(q13[:3], q2, format="bsfp8", sw13=s13[:3], sw2=s2)),
        "H % 128 != 0": _expect_raise(
            "H not multiple of 128", ValueError,
            lambda: P(q13[:, :, :192], q2[:, :192], format="bsfp8", sw13=s13, sw2=s2)),
        "I % 128 != 0": _expect_raise(
            "I not multiple of 128", ValueError,
            lambda: P(q13[:, :128], q2[:, :, :64], format="bsfp8", sw13=s13[:, :1], sw2=s2)),
        "2-D w13": _expect_raise("2-D w13", ValueError, lambda: P(q13[0], q2, format="bsfp8", sw13=s13, sw2=s2)),
        "weights on the CPU": _expect_raise(
            "cpu weights", ValueError,
            lambda: P(q13.cpu(), q2.cpu(), format="bsfp8", sw13=s13.cpu(), sw2=s2.cpu())),
    }
    if major in (10, 12):
        results["mxfp8 with scales"] = _expect_raise(
            "mxfp8 with scales", ValueError, lambda: P(w13, w2, format="mxfp8", sw13=s13, sw2=s2))
        results["mxfp8 with fp8 weights"] = _expect_raise(
            "mxfp8 fp8 weights", ValueError, lambda: P(q13, q2, format="mxfp8"))
    else:
        results["format='mxfp8' on sm_90"] = _expect_raise(
            "mxfp8 on sm_90", NotImplementedError, lambda: P(w13, w2, format="mxfp8"))

    ex = P(q13, q2, format="bsfp8", sw13=s13, sw2=s2)
    M, TOPK = 5, 2
    hidden, ids, wts, _skip, bias, bscale = make_inputs(M, E, TOPK, H, seed=6)
    L = fso.moe.layer
    other = 120 if major != 12 else 90
    results.update({
        "layer: hidden size != H": _expect_raise(
            "layer hidden size", ValueError, lambda: L(hidden[:, :128].contiguous(), ex, ids, wts)),
        "layer: fp16 hidden": _expect_raise("layer fp16 hidden", ValueError, lambda: L(hidden.half(), ex, ids, wts)),
        "layer: topk_w shape": _expect_raise("layer topk_w shape", ValueError, lambda: L(hidden, ex, ids, wts[:, :1])),
        "layer: float ids": _expect_raise("layer float ids", ValueError, lambda: L(hidden, ex, ids.float(), wts)),
        "layer: bias_scale without bias": _expect_raise(
            "layer bias_scale alone", ValueError, lambda: L(hidden, ex, ids, wts, bias_scale=bscale)),
        "layer: bias shape": _expect_raise(
            "layer bias shape", ValueError, lambda: L(hidden, ex, ids, wts, bias=bias[:2])),
        "layer: not a MoeExperts": _expect_raise("layer not handle", TypeError, lambda: L(hidden, object(), ids, wts)),
        "layer: handle from another arch": _expect_raise(
            "layer other arch", ValueError, lambda: L(hidden, dataclasses.replace(ex, arch=other), ids, wts)),
        "raw op: unknown kind": _expect_raise(
            "raw op unknown kind", ValueError,
            lambda: torch.ops.fish_scales_ops.moe_layer(hidden, ex.w13, ex.sw13, ex.w2, ex.sw2, ids.int(), wts,
                                                        None, None, "int8", False)),
        "raw op: kind of another arch": _expect_raise(
            "raw op other kind", NotImplementedError,
            lambda: torch.ops.fish_scales_ops.moe_layer(hidden, ex.w13, ex.sw13, ex.w2, ex.sw2, ids.int(), wts,
                                                        None, None, "bsfp8" if major != 9 else "mxfp8", False)),
    })
    for k, v in results.items():
        print(f"  refuse  {k:34s} -> {v}")
    # An empty bucket is not an error: it returns an empty [0, H] tensor.
    empty = L(hidden[:0], ex, ids[:0], wts[:0], bias=bias[:0], bias_scale=bscale[:0])
    torch.cuda.synchronize()
    empty_ok = tuple(empty.shape) == (0, H) and empty.dtype == torch.bfloat16
    _check(empty_ok, f"layer on an empty bucket returned {empty.dtype} {tuple(empty.shape)}")
    print(f"  empty   M=0 bucket -> bf16 {tuple(empty.shape)}  {'OK' if empty_ok else 'FAIL'}")


# --- (e) ----------------------------------------------------------------------------

def case_e(major: int) -> None:
    fam = "C_35a3"
    E, TOPK, H, I = FAMILIES[fam]
    w13, w2 = make_weights(E, H, I, seed=61)
    q13, s13 = block_quantize_fp8(w13)
    q2, s2 = block_quantize_fp8(w2)
    ex = fso.moe.prepare_experts(q13, q2, format="bsfp8", sw13=s13, sw2=s2)
    del w13, w2

    def block(h, ids, w, b, s):
        return fso.moe.layer(h, ex, ids, w, bias=b, bias_scale=s) * 2.0

    compiled = torch.compile(block, fullgraph=True, dynamic=False)
    for M in (8, 512):
        hidden, ids, wts, skip, bias, bscale = make_inputs(M, E, TOPK, H, seed=3456 + M)
        want = block(hidden, skip, wts, bias, bscale)
        got = compiled(hidden, skip, wts, bias, bscale)
        torch.cuda.synchronize()
        ok = torch.equal(got, want)
        _check(ok, f"e {fam} M={M}: torch.compile(fullgraph=True) != eager (max |diff| {_maxabs(got, want):.3e})")
        print(f"  compile {fam:8s} M={M:5d} kind={ex.kind}: torch.compile(fullgraph=True) of a function calling "
              f"layer() (int64 skipped ids, bias + bias_scale) {'==' if ok else '!='} eager  {'OK' if ok else 'FAIL'}")
    del ex, q13, s13, q2, s2
    torch.cuda.empty_cache()


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: test_moe_unified needs a CUDA device")
        return 0
    torch.backends.cuda.matmul.allow_tf32 = False
    major, minor = torch.cuda.get_device_capability(0)
    print(f"device: {torch.cuda.get_device_name(0)} sm_{major}{minor}; torch {torch.__version__}")
    covered, skipped = [], []
    if major == 12:
        print("== (a) layer on format='mxfp8' vs moe_layer_mxfp8_sm120 ==")
        case_a(major)
        covered.append("a")
        print("== (a2) fused-combine engagement rule ==")
        case_a2()
        covered.append("a2")
    else:
        skipped += ["a (sm_120/121)", "a2 (sm_120/121)"]
    if major == 10:
        print("== (a100) layer on format='mxfp8' vs the bench's sm_100 chain ==")
        case_a(major)
        covered.append("a100")
    else:
        skipped.append("a100 (sm_100/103)")
    if major in (10, 12):
        print("== (b) format='bsfp8' -> MXFP8 double quantization vs fp32 references ==")
        case_b(major)
        covered.append("b")
    else:
        skipped.append("b (sm_100/103, sm_120/121)")
    if major == 9:
        print("== (c) layer on format='bsfp8' vs moe_layer_fp8_sm90, bias fold ==")
        case_c()
        covered.append("c")
    else:
        skipped.append("c (sm_90)")
    print("== (d) capability matrix and refusals ==")
    case_d(major)
    covered.append("d")
    print("== (e) torch.compile ==")
    case_e(major)
    covered.append("e")
    print(f"\ncovered: {', '.join(covered)}; skipped on sm_{major}{minor}: {', '.join(skipped) or 'none'}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("fso.moe unified surface: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
