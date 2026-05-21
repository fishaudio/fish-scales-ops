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


def main():
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

    print("== linear_bf16 ==")
    for M, N, K in shapes: test_linear_bf16(M, N, K)
    print()

    print("== linear_fp8 (1×128) ==")
    for M, N, K in shapes: test_linear_fp8(M, N, K)

    print("\nAll CUDA Graph capture+replay tests passed.")


if __name__ == "__main__":
    main()
