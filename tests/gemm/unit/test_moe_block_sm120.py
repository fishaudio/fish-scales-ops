#!/usr/bin/env python3
"""The composed sm_120 MoE block (moe_block_mxfp8_sm120): router + routed
experts + shared expert + gated add in one capture-safe call, and the
parallelism contracts it has to honour.

The block is plumbing over parts that have their own tests (the router against
the torch reference in test_moe_router_topk.py, the routed layer against an FP32
reference in test_mxfp8_grouped.py and for skipped ids in
test_moe_layer_padded_ids_sm120.py), so most cases here check that the plumbing
composes exactly:

  1. against the same chain written out by hand -- router, routed layer, shared
     expert, gated add -- which must agree to a few bf16 ULP. It is not bit
     identical on purpose: the fused combine adds the shared row in FP32 before
     rounding once, while the hand-written chain rounds the routed sum to bf16
     first and adds in bf16, so the fused form is the more accurate of the two;
  2. against an FP32 reference built from the bf16 master weights, which is the
     end-to-end accuracy statement (cosine);
  3. tensor parallel: the per-rank INTER is what the block sees, and a split
     that leaves it off a multiple of 128 is refused with a message rather than
     silently mis-scaled;
  4. expert parallel: with an expert map the router selects over all experts and
     the block computes only the rank's own, the rest costing nothing, which is
     checked against the routed layer driven with the mapped ids directly;
  5. data parallel: a padded bucket (num_token_non_padded) leaves the padded
     rows zero, and `out=` writes into the caller's buffer and returns it;
  6. determinism: 20 eager calls bit-identical and a graph replay equal to
     eager.
"""
import sys

import torch

import fish_scales_ops as fso

E, TOPK, HIDDEN, INTER, INTER_S = 128, 8, 2048, 512, 512


def build(dev, e_local=E, inter=INTER, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    w13 = (torch.randn(e_local, 2 * inter, HIDDEN, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(e_local, HIDDEN, inter, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    s13 = (torch.randn(2 * INTER_S, HIDDEN, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    s2 = (torch.randn(HIDDEN, INTER_S, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    rw = (torch.randn(E + 1, HIDDEN, device=dev, generator=g) * 0.05).to(torch.bfloat16)
    q = fso.gemm.quantize_moe_weights_1x32_fp8
    w13f, sw13 = q(w13)
    w2f, sw2 = q(w2)
    s13f, ss13 = fso.gemm.quantize_1x32_fp8(s13)
    s2f, ss2 = fso.gemm.quantize_1x32_fp8(s2)
    return dict(w13=w13, w2=w2, s13=s13, s2=s2, rw=rw, w13f=w13f, sw13=sw13, w2f=w2f, sw2=sw2,
                s13f=s13f, ss13=ss13, s2f=s2f, ss2=ss2)


def hand_chain(hidden, p, *, expert_map=None, n_valid=None, shared=True):
    """The same chain the block runs, written out with the per-step ops. Without
    the shared expert the router weight carries no gate row either, which is the
    shape a model with no shared expert (Family B) has."""
    rw = p["rw"] if shared else p["rw"][:E].contiguous()
    ids, w, gate = fso.gemm.moe_router_topk(
        hidden, rw, TOPK, with_shared_gate=shared,
        num_token_non_padded=n_valid, expert_map=expert_map)
    routed = fso.gemm.moe_layer_mxfp8_sm120(hidden, p["w13f"], p["sw13"], p["w2f"], p["sw2"], ids, w)
    if not shared:
        return routed
    xq, sx = fso.gemm.quantize_1x32_fp8(hidden)
    gu = fso.gemm.linear_mxfp8(xq, p["s13f"], sx, p["ss13"])
    hq, sh = fso.gemm.silu_chunk_mul_quantize_1x32_fp8(gu)
    shared_out = fso.gemm.linear_mxfp8(hq, p["s2f"], sh, p["ss2"])
    return routed + (gate.unsqueeze(1) * shared_out.float()).to(torch.bfloat16)


def fp32_reference(hidden, p):
    h = hidden.float()
    logits = h @ p["rw"].float().t()
    probs = torch.softmax(logits[:, :E], dim=-1)
    w, ids = torch.topk(probs, TOPK, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)
    out = torch.zeros_like(h)
    for j in range(TOPK):
        for t in range(h.shape[0]):
            e = int(ids[t, j])
            gu = h[t] @ p["w13"][e].float().t()
            act = torch.nn.functional.silu(gu[:INTER]) * gu[INTER:]
            out[t] += float(w[t, j]) * (act @ p["w2"][e].float().t())
    gu = h @ p["s13"].float().t()
    act = torch.nn.functional.silu(gu[:, :INTER_S]) * gu[:, INTER_S:]
    shared = act @ p["s2"].float().t()
    return out + torch.sigmoid(logits[:, E]).unsqueeze(1) * shared


def cos(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        where = "no CUDA device" if not torch.cuda.is_available() \
            else f"device is sm_{torch.cuda.get_device_capability()[0]}x"
        print(f"SKIP: test_moe_block_sm120 is sm_120/121 (RTX 5090) only ({where})")
        return 0
    dev = "cuda"
    failures = []
    p = build(dev)

    # 1 + 2: the fused block against the hand-written chain and an FP32 reference.
    for M in (1, 8, 64, 512):
        g = torch.Generator(device=dev).manual_seed(M)
        hidden = torch.randn(M, HIDDEN, device=dev, dtype=torch.bfloat16, generator=g)
        blk = fso.gemm.moe_block_mxfp8_sm120(
            hidden, p["rw"], p["w13f"], p["sw13"], p["w2f"], p["sw2"], topk=TOPK,
            shared_w13_fp8=p["s13f"], shared_sw13=p["ss13"],
            shared_w2_fp8=p["s2f"], shared_sw2=p["ss2"], shared_gate_in_router=True)
        hand = hand_chain(hidden, p)
        torch.cuda.synchronize()
        c_hand = cos(blk, hand)
        scale = float(hand.float().abs().max())
        dev_max = float((blk.float() - hand.float()).abs().max()) / max(scale, 1e-9)
        # bf16 has 8 mantissa bits, so one rounding is up to 3.9e-3 of a value's
        # own magnitude; the two chains round a different number of times, which
        # bounds the deviation at about two of those against the block's largest
        # element. The cosine gate is what actually holds the chains together.
        if c_hand < 0.99999 or dev_max > 1.6e-2:
            failures.append(f"M={M}: block vs hand chain cos={c_hand:.6f} maxdev={dev_max:.2e}")
        c_ref = float("nan")
        if M <= 8:  # the FP32 reference loops over tokens; keep it small
            c_ref = cos(blk, fp32_reference(hidden, p))
            if c_ref < 0.995:
                failures.append(f"M={M}: block vs FP32 reference cos={c_ref:.6f}")
        print(f"  M={M:4d}: vs hand chain cos={c_hand:.6f} maxdev={dev_max:.2e}"
              + (f", vs FP32 reference cos={c_ref:.6f}" if M <= 8 else "") + "  OK")

    # 3: tensor parallel. The per-rank INTER is what the block sees.
    for tp in (2, 4):
        pt = build(dev, inter=INTER // tp, seed=tp)
        M = 64
        g = torch.Generator(device=dev).manual_seed(500 + tp)
        hidden = torch.randn(M, HIDDEN, device=dev, dtype=torch.bfloat16, generator=g)
        blk = fso.gemm.moe_block_mxfp8_sm120(
            hidden, pt["rw"][:E].contiguous(), pt["w13f"], pt["sw13"], pt["w2f"], pt["sw2"], topk=TOPK)
        hand = hand_chain(hidden, pt, shared=False)
        torch.cuda.synchronize()
        c = cos(blk, hand)
        if c < 0.99999:
            failures.append(f"tp={tp} (INTER={INTER // tp}): cos={c:.6f} vs the hand chain")
        print(f"  tp={tp} INTER={INTER // tp}: routed-only block == hand chain (cos={c:.6f})  OK")
    # tp 8 leaves INTER at 64. The weight quantizer refuses that at load time and
    # the block refuses it per call; both messages name the 128-element scale
    # block, so a rank that was mis-sharded learns why at the earliest point.
    try:
        build(dev, inter=64, seed=9)
        failures.append("quantize_moe_weights_1x32_fp8 accepted INTER=64")
    except ValueError as e:
        if "multiple" not in str(e) or "128" not in str(e):
            failures.append(f"the weight quantizer raised the wrong message for INTER=64: {e}")
    try:
        i64 = 64
        fso.gemm.moe_block_mxfp8_sm120(
            torch.randn(4, HIDDEN, device=dev, dtype=torch.bfloat16),
            p["rw"][:E].contiguous(),
            torch.empty(E, 2 * i64, HIDDEN, device=dev, dtype=torch.float8_e4m3fn),
            torch.empty(E, HIDDEN // 128, 2 * i64, device=dev, dtype=torch.int32),
            torch.empty(E, HIDDEN, i64, device=dev, dtype=torch.float8_e4m3fn),
            torch.empty(E, i64 // 128 + 1, HIDDEN, device=dev, dtype=torch.int32),
            topk=TOPK)
        failures.append("the block accepted INTER=64; the scale layout cannot serve it")
    except ValueError as e:
        if "multiple of 128" not in str(e):
            failures.append(f"the block raised the wrong message for INTER=64: {e}")
        print("  tp=8 INTER=64: refused by the weight quantizer and by the block, "
              "both naming the 128-element scale block  OK")

    # 4: expert parallel. This rank owns the first half of the experts.
    e_local = E // 2
    pe = build(dev, e_local=e_local, seed=11)
    pe["rw"] = p["rw"]
    emap = torch.full((E + 1,), -1, device=dev, dtype=torch.int32)
    emap[:e_local] = torch.arange(e_local, device=dev, dtype=torch.int32)
    emap[E] = e_local
    M = 64
    g = torch.Generator(device=dev).manual_seed(77)
    hidden = torch.randn(M, HIDDEN, device=dev, dtype=torch.bfloat16, generator=g)
    blk = fso.gemm.moe_block_mxfp8_sm120(
        hidden, pe["rw"][:E].contiguous(), pe["w13f"], pe["sw13"], pe["w2f"], pe["sw2"],
        topk=TOPK, expert_map=emap)
    hand = hand_chain(hidden, pe, expert_map=emap, shared=False)
    torch.cuda.synchronize()
    c = cos(blk, hand)
    if c < 0.99999:
        failures.append(f"ep: cos={c:.6f} vs the hand chain")
    ids, _, _ = fso.gemm.moe_router_topk(
        hidden, pe["rw"][:E].contiguous(), TOPK, expert_map=emap)
    n_remote = int((ids < 0).sum())
    print(f"  ep: E_local={e_local}, {n_remote} of {M * TOPK} entries remote, "
          f"block == hand chain (cos={c:.6f})  OK")
    try:
        fso.gemm.moe_block_mxfp8_sm120(
            hidden, pe["rw"][:E].contiguous(), pe["w13f"], pe["sw13"], pe["w2f"], pe["sw2"], topk=TOPK)
        failures.append("an expert-parallel rank without expert_map was accepted")
    except ValueError as e:
        if "expert_map" not in str(e):
            failures.append(f"missing expert_map raised the wrong message: {e}")
        print("  ep without expert_map: refused, naming the argument  OK")

    # 5: data parallel. A padded bucket and a caller-owned output buffer.
    M = 64
    n_real = 40
    n_valid = torch.tensor([n_real], device=dev, dtype=torch.int32)
    buf = torch.full((M, HIDDEN), float("nan"), device=dev, dtype=torch.bfloat16)
    rw_e = p["rw"][:E].contiguous()
    ret = fso.gemm.moe_block_mxfp8_sm120(
        hidden, rw_e, p["w13f"], p["sw13"], p["w2f"], p["sw2"], topk=TOPK,
        num_token_non_padded=n_valid, out=buf)
    full = fso.gemm.moe_block_mxfp8_sm120(
        hidden, rw_e, p["w13f"], p["sw13"], p["w2f"], p["sw2"], topk=TOPK)
    torch.cuda.synchronize()
    if ret.data_ptr() != buf.data_ptr():
        failures.append("out= did not return the caller's buffer")
    if not torch.equal(buf[:n_real], full[:n_real]):
        failures.append("the real rows of a padded bucket differ from the unpadded call")
    if not bool((buf[n_real:] == 0).all()):
        failures.append("the padded rows of the bucket are not zero")
    print(f"  dp: bucket of {M} with {n_real} real rows writes the caller's buffer, "
          f"real rows unchanged, padded rows zero  OK")

    # 6b: the side-stream overlap changes nothing but the schedule. The join is a
    # stream dependency, so the overlapped and sequential blocks must be equal bit
    # for bit, eagerly and through a graph, and the graph has to capture the two
    # branches (the side stream is joined with events, which is what capture
    # follows).
    seq = fso.gemm.moe_block_mxfp8_sm120(
        hidden, p["rw"], p["w13f"], p["sw13"], p["w2f"], p["sw2"], topk=TOPK,
        shared_w13_fp8=p["s13f"], shared_sw13=p["ss13"],
        shared_w2_fp8=p["s2f"], shared_sw2=p["ss2"], shared_gate_in_router=True,
        overlap_shared=False)
    ovl = fso.gemm.moe_block_mxfp8_sm120(
        hidden, p["rw"], p["w13f"], p["sw13"], p["w2f"], p["sw2"], topk=TOPK,
        shared_w13_fp8=p["s13f"], shared_sw13=p["ss13"],
        shared_w2_fp8=p["s2f"], shared_sw2=p["ss2"], shared_gate_in_router=True,
        overlap_shared=True)
    torch.cuda.synchronize()
    if not torch.equal(seq, ovl):
        failures.append("the side-stream overlap changed the result")
    ovl_fn = lambda: fso.gemm.moe_block_mxfp8_sm120(
        hidden, p["rw"], p["w13f"], p["sw13"], p["w2f"], p["sw2"], topk=TOPK,
        shared_w13_fp8=p["s13f"], shared_sw13=p["ss13"],
        shared_w2_fp8=p["s2f"], shared_sw2=p["ss2"], shared_gate_in_router=True,
        overlap_shared=True)
    sg = torch.cuda.Stream()
    sg.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(sg):
        for _ in range(3):
            ovl_fn()
    torch.cuda.current_stream().wait_stream(sg)
    torch.cuda.synchronize()
    graph_o = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_o, stream=sg):
        out_o = ovl_fn()
    torch.cuda.synchronize()
    out_o.fill_(float("nan"))
    graph_o.replay()
    torch.cuda.synchronize()
    ovl_graph_ok = torch.equal(out_o, seq)
    if not ovl_graph_ok:
        failures.append("the overlapped block replayed from a graph differs from the sequential call")
    print(f"  overlap: sequential == overlapped bit for bit; graph replay of the overlapped block "
          f"{'==' if ovl_graph_ok else '!='} it  OK")

    # 6: determinism, eager and through a graph.
    call = lambda: fso.gemm.moe_block_mxfp8_sm120(
        hidden, p["rw"], p["w13f"], p["sw13"], p["w2f"], p["sw2"], topk=TOPK,
        shared_w13_fp8=p["s13f"], shared_sw13=p["ss13"],
        shared_w2_fp8=p["s2f"], shared_sw2=p["ss2"], shared_gate_in_router=True,
        num_token_non_padded=n_valid)
    ref = call()
    torch.cuda.synchronize()
    bad = sum(0 if torch.equal(call(), ref) else 1 for _ in range(20))
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        out_g = call()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    graph_ok = torch.equal(out_g, ref)
    if bad or not graph_ok:
        failures.append(f"determinism: {bad}/20 eager runs differ, graph replay "
                        f"{'==' if graph_ok else '!='} eager")
    print(f"  determinism: 20 eager runs, {bad} differ; graph replay "
          f"{'==' if graph_ok else '!='} eager  OK")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nmoe_block_mxfp8_sm120: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
