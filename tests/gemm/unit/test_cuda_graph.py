"""CUDA Graph capture + replay compatibility test.

Validates that every public op in `blockscale_gemm` survives capture into a
CUDA Graph and produces the same output on replay as in eager mode.

Things that could trip CUDA Graph capture and are explicitly exercised:

  - `cudaFuncSetAttribute` is FORBIDDEN inside an active stream capture.
    The E8 static-bool guard avoids the call after the first launch — so
    every op is invoked at least once in eager mode before capture.
  - The Stream-K wrapper has its own pool of side streams + scratch buffer
    that gets lazily allocated. `streamk.cuh` claims it warms up correctly
    if the largest shape runs before capture starts. This test exercises
    Stream-K shapes (narrow-N + K≥9728 + small M).
  - The E9 Params cache stores TMA descriptors keyed by pointers + shape.
    Lookup is host-side so it does not affect capture itself; the captured
    kernel uses whatever Params the cache returned at capture time, and
    replay re-issues those same Params against the same buffers — which
    is exactly the contract CUDA Graphs already require from the caller.

Tolerance: outputs after replay are compared to outputs captured at the
end of the eager warmup pass. They share inputs and the same kernel, so
agreement is expected to be bit-exact (no rtol slack).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

import fish_scales_ops as fso


def _check_graph_op(name, eager_call, graph_call):
    """Run `eager_call()` once to get a reference, then capture `graph_call()`
    on a side stream and replay; confirm the captured tensor matches the
    eager output bit-for-bit on replay (same kernel, same inputs, same
    memory → no tolerance slack).

    `graph_call()` must be a no-arg lambda that returns the captured output
    tensor (e.g. `lambda: fso.gemm.linear_bf16(x, w)`). The tensor is allocated
    in the graph's private mempool; its data_ptr stays valid across
    replays and gets overwritten by each replay.
    """
    # Step 1 — eager reference (cold path; also seeds the E9 Params cache
    # and trips the E8 static cudaFuncSetAttribute guard).
    ref = eager_call().clone()
    assert torch.isfinite(ref).all(), f"{name}: eager reference itself has NaN/Inf — input is hitting a quant edge case, fix the test inputs"

    # Step 2 — warm up on the capture stream so the streamk pool, autotuner,
    # allocator pools, etc. settle before the actual capture window.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _ = graph_call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Step 3 — capture (PyTorch's mempool tracks all in-graph allocations so
    # the captured output tensor reuses the same address on every replay).
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        captured = graph_call()

    # Step 4 — clobber the captured tensor with NaN so we can prove replay
    # actually wrote into it. Then replay.
    captured.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()

    # Step 5 — replay must produce the same kernel output as the eager pass.
    assert torch.isfinite(captured).all(), f"{name}: replay produced NaN/Inf"
    if not torch.equal(captured, ref):
        diff = (captured.float() - ref.float()).abs().max().item()
        raise AssertionError(f"{name}: capture/replay output diverged (max abs diff = {diff})")


def _shape_str(M, N, K):
    return f"M={M:>5} N={N:>5} K={K:>5}"


def _make_inputs(M, N, K):
    """Build (x, w) with bench-matching seed (so we don't hit the all-zero-block
    quant edge case in linear_bf16) and full-magnitude inputs (not * 0.1)."""
    torch.manual_seed(M * 1009 + N * 17 + K)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").contiguous()
    w = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K**0.5)).contiguous()
    return x, w


def test_linear_bf16(M, N, K):
    x, w = _make_inputs(M, N, K)
    _check_graph_op("linear_bf16",
        eager_call=lambda: fso.gemm.linear_bf16(x, w),
        graph_call=lambda: fso.gemm.linear_bf16(x, w))
    print(f"  linear_bf16       {_shape_str(M, N, K)}  OK")


def test_linear_fp8(M, N, K):
    """BS-FP8 (1×128). Pre-quantize OUTSIDE the capture (mirrors production:
    weight quant happens at module init, the captured op is just the GEMM)."""
    sm = torch.cuda.get_device_capability(0)[0]
    x_bf, w_bf = _make_inputs(M, N, K)
    xq, sxq = fso.gemm.quantize_1x128_fp8(x_bf, use_ue8m0=(sm >= 12))
    wq, swq = fso.gemm.quantize_128x128_fp8(w_bf)
    if sm >= 12:
        sxqp = fso.gemm.repack_fp8_act_scales(sxq)
        swqp = fso.gemm.repack_fp8_wgt_scales(swq)
    else:
        sxqp, swqp = sxq, swq

    _check_graph_op("linear_fp8",
        eager_call=lambda: fso.gemm.linear_fp8(xq, wq, sxqp, swqp),
        graph_call=lambda: fso.gemm.linear_fp8(xq, wq, sxqp, swqp))
    print(f"  linear_fp8        {_shape_str(M, N, K)}  OK")


def test_linear_mxfp8(M, N, K):
    """MXFP8 (1×32). sm_120: Sm120BlockScaledKernel + Stream-K; sm_100/103:
    CUTLASS tcgen05 BlockScaled via GemmUniversalAdapter — the eager warmup
    must cover the adapter cache's initialize() (cudaFuncSetAttribute + TMA
    descriptor encode + possible workspace cudaMalloc), all of which are
    forbidden mid-capture. Pre-quantize outside the capture (production
    pattern: weight quant at init, activations via the fused quantize which
    is itself a plain kernel launch and captures fine — covered by the
    quantize-in-graph variant below)."""
    x_bf, w_bf = _make_inputs(M, N, K)
    xq, sxq = fso.gemm.quantize_1x32_fp8(x_bf)
    wq, swq = fso.gemm.quantize_1x32_fp8(w_bf)

    _check_graph_op("linear_mxfp8",
        eager_call=lambda: fso.gemm.linear_mxfp8(xq, wq, sxq, swq),
        graph_call=lambda: fso.gemm.linear_mxfp8(xq, wq, sxq, swq))
    print(f"  linear_mxfp8      {_shape_str(M, N, K)}  OK")

    # Quantize + GEMM inside one capture (decode-loop pattern: activation
    # quantize is part of the captured forward).
    def quant_gemm():
        xq2, sxq2 = fso.gemm.quantize_1x32_fp8(x_bf)
        return fso.gemm.linear_mxfp8(xq2, wq, sxq2, swq)

    _check_graph_op("quantize_1x32+linear_mxfp8",
        eager_call=quant_gemm,
        graph_call=quant_gemm)
    print(f"  quant+mxfp8       {_shape_str(M, N, K)}  OK")


# --- sm_100/103 slot-bound grouped decode route -----------------------------
#
# The slot route replaces the pointer-array prep kernel with its own one-block
# slot-list kernel and keeps the list in a thread-local pool, so it has the same
# two capture obligations as the route it sits beside: the pool must be filled
# by an eager call before capture, and everything the routing decides must be
# rebuilt on device at replay time rather than baked into the captured Params.
# Both are checked below. FSO_GROUPED_SLOT is a per-process static, so these run
# as subprocesses of main().

SLOT_G, SLOT_TOPK, SLOT_N, SLOT_K, SLOT_M = 128, 8, 1536, 2048, 4


def _slot_layer_inputs():
    torch.manual_seed(SLOT_M * 1009 + SLOT_N * 17 + SLOT_K)
    m_cap = (SLOT_M + 3) // 4 * 4
    x = torch.randn(SLOT_M, SLOT_K, dtype=torch.bfloat16, device="cuda") * 0.1
    w = torch.randn(SLOT_G, SLOT_N, SLOT_K, dtype=torch.bfloat16, device="cuda") / (SLOT_K ** 0.5)
    w_fp8, sw = fso.gemm.quantize_moe_weights_1x32_fp8(w)
    g = torch.Generator(device="cpu").manual_seed(7)
    topk_ids = torch.stack(
        [torch.randperm(SLOT_G, generator=g)[:SLOT_TOPK] for _ in range(SLOT_M)]
    ).to("cuda", torch.int32)
    return x, w_fp8, sw, topk_ids, m_cap


def _slot_call(x, w_fp8, sw, topk_ids, m_cap, with_slots=False):
    """Routing -> gather-quant -> slot-route GEMM, all on device. The routing is
    derived from topk_ids inside the call, so a captured graph follows whatever
    ids the buffer holds at replay time.

    With `with_slots` the routing op also emits the packed active-expert list
    and the GEMM is handed it, which is the form the MoE layer uses: the list
    is then rebuilt on the device on every replay, exactly like `masked_m`, and
    the route launches no prep kernel of its own -- so the slot-list pool is
    never touched and a capture needs no eager warm-up for it."""
    if with_slots:
        masked_m, _, slot_of_flat, slot_to_expert = fso.gemm.moe_build_routing(
            topk_ids, SLOT_G, m_cap, with_slots=True)
    else:
        masked_m, _, slot_of_flat = fso.gemm.moe_build_routing(topk_ids, SLOT_G, m_cap)
        slot_to_expert = None
    a_fp8, sa = fso.gemm.quantize_1x32_grouped_gather_fp8(
        x, slot_of_flat, SLOT_TOPK, SLOT_G, m_cap)
    return fso.gemm.linear_mxfp8_grouped_masked(
        a_fp8, w_fp8, sa, sw, masked_m, 1, min(SLOT_M * SLOT_TOPK, SLOT_G),
        slot_to_expert), masked_m


def test_slot_grouped_graph(with_slots=False):
    """Capture at a fixed M, rewrite topk_ids in place, replay, and compare
    against a fresh eager call on the new routing. Only the rows the masked
    contract defines are compared: rows at or beyond masked_m[g] are never
    written, so they hold whatever torch.empty returned.

    `with_slots` runs the same thing through the form the MoE layer uses, where
    the routing op emits the packed active-expert list inside the graph and the
    GEMM consumes it instead of building its own. What that has to prove is the
    same contract plus one more thing: the list is written by a kernel several
    launches back in the same graph, and the GEMM reads it in its prologue, so a
    replay that got the ordering wrong would drop an expert's rows and the
    comparison against the eager call would fail."""
    tag = "list" if with_slots else "own "
    x, w_fp8, sw, topk_ids, m_cap = _slot_layer_inputs()

    for _ in range(3):
        _slot_call(x, w_fp8, sw, topk_ids, m_cap, with_slots)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _slot_call(x, w_fp8, sw, topk_ids, m_cap, with_slots)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        captured, captured_mm = _slot_call(x, w_fp8, sw, topk_ids, m_cap, with_slots)

    captured.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    ref, ref_mm = _slot_call(x, w_fp8, sw, topk_ids, m_cap, with_slots)
    torch.cuda.synchronize()
    keep = torch.arange(m_cap, device="cuda").view(1, -1) < ref_mm.long().view(-1, 1)
    assert torch.equal(captured_mm, ref_mm), "slot grouped: replay rebuilt a different routing"
    assert torch.equal(captured[keep], ref[keep]), "slot grouped: replay != eager"
    print(f"  mxfp8 slot grouped   M={SLOT_M} slots={tag} replay bit-exact  OK")

    # New routing, same buffers: the slot list, masked_m and the activation
    # slab are all rebuilt on device inside the graph, and the grid does not
    # move because min(M*topk, G) is host-static at this M.
    g2 = torch.Generator(device="cpu").manual_seed(4242)
    topk_ids.copy_(torch.stack(
        [torch.randperm(SLOT_G, generator=g2)[:SLOT_TOPK] for _ in range(SLOT_M)]
    ).to("cuda", torch.int32))
    g.replay()
    torch.cuda.synchronize()
    ref2, ref2_mm = _slot_call(x, w_fp8, sw, topk_ids, m_cap, with_slots)
    torch.cuda.synchronize()
    keep2 = torch.arange(m_cap, device="cuda").view(1, -1) < ref2_mm.long().view(-1, 1)
    assert torch.equal(captured_mm, ref2_mm), "slot grouped: reroute replay kept the old routing"
    assert torch.equal(captured[keep2], ref2[keep2]), "slot grouped: reroute replay != eager"
    print(f"  mxfp8 slot reroute   M={SLOT_M} slots={tag} replay bit-exact after in-place "
          f"rewrite  OK")


def test_slot_grouped_graph_list_no_warmup():
    """The layer's form needs no eager warm-up of the SLOT-LIST pool.

    When the caller supplies the list the route never calls the pool, so the
    only host-side state a first call still has to populate is the
    cudaFuncSetAttribute guard and the CUTLASS workspace, both of which the
    eager warm-up below covers. Capturing straight after one eager call must
    therefore succeed -- the same sequence without the list aborts, which
    test_slot_pool_refuses_capture covers."""
    x, w_fp8, sw, topk_ids, m_cap = _slot_layer_inputs()
    _slot_call(x, w_fp8, sw, topk_ids, m_cap, with_slots=True)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _slot_call(x, w_fp8, sw, topk_ids, m_cap, with_slots=True)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        captured, _ = _slot_call(x, w_fp8, sw, topk_ids, m_cap, with_slots=True)
    captured.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(captured[:, :1]).any(), "replay did not write"
    print("  mxfp8 slot list      capture after one eager call, no slot-list pool  OK")


def test_slot_pool_refuses_capture():
    """No eager call at all, straight into a capture: the slot-list pool must
    refuse to allocate and say so, rather than calling cudaMalloc inside the
    capture. Runs as its own process because the refusal aborts."""
    x, w_fp8, sw, topk_ids, m_cap = _slot_layer_inputs()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        _slot_call(x, w_fp8, sw, topk_ids, m_cap)
    raise AssertionError("the slot-list pool allocated during capture instead of refusing")


def _run_slot_subprocess(case):
    import subprocess
    env = dict(os.environ, FSO_GROUPED_SLOT="force")
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--slot-case", case],
                       env=env, capture_output=True, text=True)
    return r


def run_slot_graph_cases():
    for case in ("graph", "graph_list", "list_no_warmup"):
        r = _run_slot_subprocess(case)
        print(r.stdout, end="")
        assert r.returncode == 0, f"slot graph case {case} failed (exit {r.returncode}):\n{r.stderr}"

    r = _run_slot_subprocess("no_warmup")
    assert r.returncode != 0, "capture without an eager warmup was allowed to allocate"
    assert "slot-list pool is empty during stream capture" in r.stderr, \
        f"unexpected failure instead of the pool refusal:\n{r.stderr}"
    print("  mxfp8 slot pool      capture without eager warmup refused with a message  OK")



# --- sm_100/103 fused-SwiGLU FC1 -------------------------------------------
#
# The fused FC1 is a second CUTLASS instantiation with its OWN host-side state:
# its own CUTLASS workspace buffer, its own Params cache and its own
# cudaFuncSetAttribute guard. The argument-array pool and the prep kernel are
# shared with the unfused launcher and are covered above, but none of the three
# new ones is, and every one of them would break a capture if it were touched
# for the first time inside it. Two cases:
#
#  * warmed up eagerly first, the op must capture and replay bit-exactly, and
#    must follow an in-place routing rewrite like every other grouped op;
#  * warmed up on the UNFUSED op only — so the shared argument-array pool is
#    already allocated and cannot be the one that refuses — capturing the fused
#    op must abort with the fused launcher's own message rather than call
#    cudaMalloc inside the capture.

FUSED_G, FUSED_TOPK, FUSED_INTER, FUSED_K, FUSED_M = 128, 8, 768, 2048, 64


def _fused_layer_inputs():
    torch.manual_seed(FUSED_M * 1009 + FUSED_INTER * 17 + FUSED_K)
    m_cap = (FUSED_M + 3) // 4 * 4
    x = torch.randn(FUSED_M, FUSED_K, dtype=torch.bfloat16, device="cuda") * 0.1
    w13 = torch.randn(FUSED_G, 2 * FUSED_INTER, FUSED_K, dtype=torch.bfloat16,
                      device="cuda") / (FUSED_K ** 0.5)
    w13i_fp8, sw13i = fso.gemm.quantize_moe_weights_1x32_fp8(w13, w13_interleave=True)
    return x, w13i_fp8, sw13i, m_cap


def _fused_routing(m_cap, seed):
    """Routing derived ONCE, outside anything that will be captured.

    `moe_build_routing` places a token in a slot of its expert group in ATOMIC
    ARRIVAL ORDER, so two runs of it on the same input can put the same tokens
    in different slots of the same group. The layer does not care — its combine
    sums over a token's slots and is invariant under that permutation, which is
    why the layer-level graph tests compare the combined output — but a
    PER-SLOT tensor such as the FC1 output is only reproducible while the
    routing itself is held fixed. So the routing is built here, outside the
    capture, and the captured region starts at the gather-quantise. The dynamic
    part of the contract is then exercised by rewriting these buffers in place
    and replaying, which is exactly what a caller that owns its own routing
    does.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    topk_ids = torch.stack(
        [torch.randperm(FUSED_G, generator=g)[:FUSED_TOPK] for _ in range(FUSED_M)]
    ).to("cuda", torch.int32)
    masked_m, _row_map, slot_of_flat = fso.gemm.moe_build_routing(topk_ids, FUSED_G, m_cap)
    torch.cuda.synchronize()
    return masked_m.clone(), slot_of_flat.clone()


def _fused_call(x, w13i_fp8, sw13i, masked_m, slot_of_flat, m_cap):
    """Gather-quantise + fused FC1 against a routing the caller supplies. Both
    buffers are read on the device, so a replay follows whatever they hold."""
    a_fp8, sa = fso.gemm.quantize_1x32_grouped_gather_fp8(
        x, slot_of_flat, FUSED_TOPK, FUSED_G, m_cap)
    expected_m = max(1, (FUSED_M * FUSED_TOPK + FUSED_G - 1) // FUSED_G)
    return fso.gemm.linear_mxfp8_grouped_masked_swiglu(
        a_fp8, w13i_fp8, sa, sw13i, masked_m, expected_m, min(FUSED_M * FUSED_TOPK, FUSED_G))


def _fused_defined_sf(sh, masked_m, m_cap):
    """The scale words of the DEFINED rows only, as one flat tensor.

    The slab is written for `pad(m_cap, 128)` rows because that is the extent
    the GEMM's scale-factor descriptor is built over, but the fused store only
    writes rows below `masked_m[g]` — everything past that is undefined by the
    masked contract and holds whatever the allocator handed out. Comparing the
    whole slab would therefore compare uninitialised memory, which is how the
    first version of this case failed. The word index is the same
    `Sm1xxBlockScaledConfig` atom formula `sf_word_index_grouped_atom` in
    csrc/gemm/ops/quant_kernels.cu writes and the GEMM reads.
    """
    num_kp = FUSED_INTER // 128
    m_pad = (m_cap + 127) // 128 * 128
    rows = torch.arange(m_pad, device=sh.device)
    r = rows % 128
    base = (rows // 128) * (num_kp * 128) + (r % 32) * 4 + (r // 32)
    idx = base[:, None] + torch.arange(num_kp, device=sh.device)[None, :] * 128  # [m_pad, kp]
    words = sh.reshape(sh.shape[0], -1)[:, idx.reshape(-1)].reshape(sh.shape[0], m_pad, num_kp)
    live = torch.arange(m_pad, device=sh.device).view(1, -1) < masked_m.long().view(-1, 1)
    return words[live]


def test_fused_grouped_graph():
    x, w13i_fp8, sw13i, m_cap = _fused_layer_inputs()
    masked_m, slot_of_flat = _fused_routing(m_cap, seed=11)

    for _ in range(3):
        _fused_call(x, w13i_fp8, sw13i, masked_m, slot_of_flat, m_cap)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _fused_call(x, w13i_fp8, sw13i, masked_m, slot_of_flat, m_cap)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        cap_h, cap_sh = _fused_call(x, w13i_fp8, sw13i, masked_m, slot_of_flat, m_cap)

    # Clobber both outputs, so a replay that writes nothing cannot pass.
    cap_h.view(torch.uint8).fill_(0xFF)
    cap_sh.fill_(-1)
    g.replay()
    torch.cuda.synchronize()
    ref_h, ref_sh = _fused_call(x, w13i_fp8, sw13i, masked_m, slot_of_flat, m_cap)
    torch.cuda.synchronize()
    keep = torch.arange(m_cap, device="cuda").view(1, -1) < masked_m.long().view(-1, 1)
    assert torch.equal(cap_h.view(torch.uint8)[keep], ref_h.view(torch.uint8)[keep]), \
        "fused FC1: replayed fp8 bytes != eager"
    assert torch.equal(_fused_defined_sf(cap_sh, masked_m, m_cap),
                       _fused_defined_sf(ref_sh, masked_m, m_cap)), \
        "fused FC1: replayed scale slab != eager on the defined rows"
    print(f"  mxfp8 fused FC1      M={FUSED_M} replay bit-exact (fp8 bytes and scale slab)  OK")

    # A different routing, written into the same buffers: the per-group problem
    # shapes and the activation slab are both rebuilt on device inside the
    # graph, and the grid does not move because min(M*topk, G) is host-static.
    masked_m2, slot2 = _fused_routing(m_cap, seed=2024)
    masked_m.copy_(masked_m2)
    slot_of_flat.copy_(slot2)
    g.replay()
    torch.cuda.synchronize()
    ref2_h, ref2_sh = _fused_call(x, w13i_fp8, sw13i, masked_m, slot_of_flat, m_cap)
    torch.cuda.synchronize()
    keep2 = torch.arange(m_cap, device="cuda").view(1, -1) < masked_m.long().view(-1, 1)
    assert torch.equal(cap_h.view(torch.uint8)[keep2], ref2_h.view(torch.uint8)[keep2]), \
        "fused FC1: reroute replay != eager"
    assert torch.equal(_fused_defined_sf(cap_sh, masked_m, m_cap),
                       _fused_defined_sf(ref2_sh, masked_m, m_cap)), \
        "fused FC1: reroute replay scale slab != eager on the defined rows"
    print(f"  mxfp8 fused reroute  M={FUSED_M} replay bit-exact after in-place rewrite  OK")


def test_fused_pool_refuses_capture():
    """Warm up the UNFUSED grouped GEMM only, then capture the fused one. The
    shared argument-array pool is already allocated, so whatever refuses has to
    be the fused launcher's own workspace pool."""
    x, w13i_fp8, sw13i, m_cap = _fused_layer_inputs()
    masked_m, slot_of_flat = _fused_routing(m_cap, seed=11)
    a_fp8, sa = fso.gemm.quantize_1x32_grouped_gather_fp8(
        x, slot_of_flat, FUSED_TOPK, FUSED_G, m_cap)
    fso.gemm.linear_mxfp8_grouped_masked(a_fp8, w13i_fp8, sa, sw13i, masked_m, 4, 0)
    torch.cuda.synchronize()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        _fused_call(x, w13i_fp8, sw13i, masked_m, slot_of_flat, m_cap)
    raise AssertionError("the fused FC1 workspace pool allocated during capture instead of refusing")


def run_fused_graph_cases():
    import subprocess
    here = os.path.abspath(__file__)

    def child(case):
        env = dict(os.environ)
        env.pop("FSO_GROUPED_SLOT", None)
        env.pop("FSO_FC1_FUSED", None)
        return subprocess.run([sys.executable, here, "--fused-case", case], env=env,
                              capture_output=True, text=True)

    r = child("graph")
    print(r.stdout, end="")
    assert r.returncode == 0, f"fused graph case failed (exit {r.returncode}):\n{r.stderr}"

    r = child("no_warmup")
    assert r.returncode != 0, "capture without an eager fused warmup was allowed to allocate"
    assert "fused-SwiGLU grouped MXFP8" in r.stderr, \
        f"unexpected failure instead of the fused workspace refusal:\n{r.stderr[-800:]}"
    print("  mxfp8 fused pool     capture without eager warmup refused with a message  OK")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--fused-case":
        if sys.argv[2] == "graph":
            test_fused_grouped_graph()
        else:
            test_fused_pool_refuses_capture()
        return
    if len(sys.argv) > 2 and sys.argv[1] == "--slot-case":
        if sys.argv[2] == "graph":
            test_slot_grouped_graph()
        elif sys.argv[2] == "graph_list":
            test_slot_grouped_graph(with_slots=True)
        elif sys.argv[2] == "list_no_warmup":
            test_slot_grouped_graph_list_no_warmup()
        else:
            test_slot_pool_refuses_capture()
        return

    assert torch.cuda.is_available(), "CUDA required"
    dev = torch.cuda.get_device_name(0)
    sm = torch.cuda.get_device_capability(0)
    print(f"Device: {dev} sm_{sm[0]}{sm[1]}\n")

    # Cover shape classes that exercise different cascade paths:
    #   - large-M wide-N        → vanilla single-launch
    #   - large-M narrow-N+K9728→ Stream-K M-saturation gate (E6 skip)
    #   - mid-M  narrow-N+K9728 → Stream-K still engaged
    #   - small-M narrow-N+K9728→ E7 override single-launch (32, 64, 4)
    #   - very-small-M (M=2)    → Stream-K (E7 gate is M≥4)
    shapes = [
        (1024, 19456, 2560),  # gate_up
        (1024,  9728, 2560),  # gate
        (4096,  2560, 9728),  # down M=4096 (cascade + streamk-skip)
        (1024,  2560, 9728),  # down M=1024 (streamk active)
        (  32,  2560, 9728),  # down M=32   (E7 override)
        (   2,  2560, 9728),  # down M=2    (streamk path)
    ]

    # BSFP8 (1×128) paths exist on sm_90 (DeepGEMM JIT) and sm_120
    # (Sm120BlockScaledKernel) only; sm_100/103 serves MXFP8 exclusively.
    if sm[0] in (9, 12):
        print("== linear_bf16 ==")
        for M, N, K in shapes: test_linear_bf16(M, N, K)
        print()

        print("== linear_fp8 (1×128) ==")
        for M, N, K in shapes: test_linear_fp8(M, N, K)
    else:
        print("== linear_bf16 / linear_fp8: skipped (BSFP8 1×128 is sm_90/sm_120-only) ==")

    if sm[0] in (10, 12):
        print("\n== linear_mxfp8 (1×32) ==")
        for M, N, K in shapes: test_linear_mxfp8(M, N, K)

    if sm[0] == 10:
        print("\n== sm_100/103 slot-bound grouped route (subprocess per case) ==")
        run_slot_graph_cases()

        print("\n== sm_100/103 fused-SwiGLU FC1 (subprocess per case) ==")
        run_fused_graph_cases()

    print("\nAll CUDA Graph capture+replay tests passed.")


if __name__ == "__main__":
    main()
