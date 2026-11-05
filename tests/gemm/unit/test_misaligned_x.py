#!/usr/bin/env python3
"""A bf16 activation whose data pointer is not 16-byte aligned (issue #12, subtask AL).

The activation quantize kernels read their bf16 input in vectors straight from its data pointer: 16 bytes in the
kernels that are selected only for 16-byte aligned pointers (the sm_90 lanes kernel, the sm_100/sm_103 wide 1x32
kernels), and 8 or 4 bytes in the kernels they fall back to (the K % 512 1x128 kernel, the narrow 1x32 kernels,
scale_1x128_kernel). A contiguous view at an element offset that is not a multiple of the vector's element count
made the fallback fault on the device ("misaligned address"), a sticky error after which every CUDA call of the
process fails. Now:

  1.  the raw ops (quantize_1x128, quantize_1x128_contract, quantize_1x128_packed, quantize_1x32_packed,
      silu_chunk_mul_quantize_1x32, linear_qx, linear_bf16, on sm_100/sm_103 and sm_120/sm_121 the raw form of
      linear_quantized, quantize_1x32_packed followed by linear_mxfp8_quantized_raw, and on sm_120/sm_121
      linear_mxfp8_bf16x, the bf16-x GEMM whose small-M form G2 is taken for a 16-byte aligned x only, so a
      misaligned one runs quantize_1x32_packed and its narrow kernel inside the op) refuse, with ValueError naming
      the requirement and before anything is launched (torch.profiler records no CUDA kernel, memcpy or memset),
      every bf16 input whose rows the kernel the call selects cannot read: for each op, architecture and K the test
      states the width independently of the C++ (need() below) and checks every byte offset 2, 4, .., 14 at M in
      {1, 16, 80, 1079} and K in {4096, 1280}. Every offset the op accepts gives exactly the bytes of the same
      values in a tensor of their own (the fallback kernels are bitwise identical to the 16-byte ones). quantize_1x32
      reads one element per load and accepts every offset with the same bytes. The row pitch is checked too:
      quantize_1x128 on [2, 1001] (rows 2002 bytes apart) is refused, [2, 1002] runs.
  2.  fso.dense (linear, linear_quantized, quantize_activation) never refuses a misaligned bf16 activation: it
      copies one that is not 16-byte aligned into a tensor of its own first, so every offset gives the aligned
      call's bytes, eagerly, under torch.compile(fullgraph=True) and inside a CUDA-graph capture (the copy is
      captured with the call: a replay after the input's values change gives the bytes of the new values). The
      pre-quantized entry is not copied: a misaligned fp8 x or x_scale raises ValueError naming the 16-byte rule.
  3.  after every refusal the CUDA context is intact: a synchronize succeeds and the same op on an aligned copy
      gives its reference bytes again.
  4.  a non-contiguous x is copied by fso.dense before any kernel reads it: a column slice whose first element sits
      at a misaligned address runs through linear and quantize_activation with the aligned bytes.
  5.  the raw GEMMs' pre-quantized activation and its scale (linear_fp8; linear_mxfp8_raw and
      linear_mxfp8_quantized_raw on sm_100/sm_103 and sm_120/sm_121), which TMA reads: an address that is not a
      multiple of 16 bytes raises ValueError before anything is launched (CuTe's host assertion aborted the process
      before), and the context stays usable.
  6.  (cleanup CL1) the MoE glue raw ops (quantize_1x32_grouped_gather and its _out form,
      silu_chunk_mul_quantize_1x32_grouped and its _out form in both row orders, the sm_90
      quantize_1x128_grouped_gather_sm90 / quantize_1x128_sorted_gather_sm90 /
      silu_chunk_mul_quantize_1x128_grouped_sm90 / silu_chunk_mul_quantize_1x128_sorted_sm90, and the combines'
      dn and bias) refuse, with ValueError and no device event, every bf16 input their kernel cannot read: every even
      offset 2..14 that is not a multiple of the kernel's width (8 bytes; 16 for the pairwise SwiGLU form and the
      combines); offset 8 of an 8-byte kernel runs with the aligned bytes, as before. M in {1, 16, 256}, K in
      {2048, 1280}. On the trunk these offsets faulted on the device.
  7.  (CL1) fso.moe.layer copies a hidden (and on the MXFP8 layer a bias) that is not contiguous or not 16-byte
      aligned before any kernel reads it: every even offset 2..14 gives the aligned call's bytes at M in
      {1, 16, 80, 256}, also through the registered op called directly, under torch.compile(fullgraph=True) and in a
      CUDA-graph replay after the input's values change.
  8.  (CL1) fso.compat.linear_mxfp8 refuses an x_fp8 or sx that is not 16-byte aligned with ValueError naming the
      operand, before routing, on a cell of every sm_100 tier _sm100_dispatch.plan names (decode row, cuBLAS
      scaled_mm, CuTe-DSL where its kernel source is found, C++ cascade) and on sm_120/121. What it accepts runs on
      every tier: a row-strided x_fp8 at an aligned address gives the contiguous call's bytes (it is copied before
      routing), and an sx at a 16-byte offset runs (the CuTe-DSL tier, which takes 32-byte aligned scales only,
      hands it to the C++ cascade).
  9.  (CL2) the weight and the int32 weight scale of the raw dense GEMMs (linear_fp8; linear_qx on sm_90 and
      sm_120/121; linear_mxfp8_raw and linear_mxfp8_quantized_raw on sm_100/103 and sm_120/121), which TMA reads:
      an address that is not a multiple of 16 bytes raises ValueError naming the operand with no device event (on
      sm_100/103 and sm_120/121 CuTe's host assertion aborted the process). An fp32 weight scale at any 4-byte offset
      and linear_bf16's bf16 weight at any even offset run with the aligned bytes.
  10. (CL2) fso.compat.linear_mxfp8 refuses a w_fp8 or sw that is not 16-byte aligned before routing, on a cell of
      every sm_100 tier and on sm_120/121; an sw at a 16-byte offset runs (the CuTe-DSL tier hands it to the cascade).
  11. (CL2) the grouped GEMMs and out-style glue ops underneath fso.moe.layer, fed by the layer's own routing: every
      TMA-read operand off a 16-byte boundary and every caller-owned output at an offset its kernel's stores cannot
      take raise ValueError naming it; what the kernels can read or write gives the layer's aligned bytes.
  12. (CL2) the caller-owned outputs of the out-style glue ops and moe_combine(out=...): refused where their
      kernel's stores cannot take the offset (4-byte fp8 words; the sorted gather's 8-byte words from 32 tokens on;
      16-byte vectors in moe_combine), run with the aligned bytes elsewhere.
  13. (CL2) fso.dense and fso.moe.layer refuse a handle whose weights (or int32 scales) are not 16-byte aligned,
      never copying a weight per call; the sm_90 fp32 block scales take any 4-byte offset; prepare_weight and
      prepare_experts return 16-byte aligned tensors for checkpoint views at other offsets.

usage: python tests/gemm/unit/test_misaligned_x.py
"""
import sys

import torch

import fish_scales_ops as fso

D = fso.dense
T = torch.ops.fish_scales_ops
MS = (1, 16, 80, 1079)
KS = (4096, 1280)
OFFSETS = (2, 4, 6, 8, 10, 12, 14)
N = 256
failures: list[str] = []
counts = {"refused": 0, "accepted": 0}


def _check(ok: bool, msg: str) -> bool:
    if not ok:
        failures.append(msg)
    return ok


def need(op: str, major: int, K: int) -> int:
    """The width (bytes) of the loads with which the kernel the op runs for an x that is NOT 16-byte aligned reads it;
    2 = one element per load (never refused). From the kernels in csrc/gemm/ops/quant_kernels.cu and
    common/scale_kernels.cuh, written here independently of the op bodies' checks."""
    if op.startswith("dense"):
        return 2                                    # fso.dense copies a misaligned bf16 x first: never refused
    if op in ("q1x128_f32", "q1x128_ue8m0", "q1x128_packed"):
        return 8 if K % 512 == 0 else 4             # fp8bs_quantize_1x128_packed_kernel (LDG.64) / scale_1x128_kernel
    if op == "q1x128_contract":
        return 4                                    # scale_1x128_kernel
    if op == "q1x32":
        return 2                                    # fp8bs_quantize_1x32_kernel, one element per lane
    if op in ("q1x32_packed", "silu1x32", "raw_lq_chain", "bf16x"):
        return 8                                    # the narrow 1x32 kernels (LDG.64)
    if op == "linear_qx":
        return 4                                    # fp8_gemm_run's scale_1x128_kernel
    if op == "linear_bf16":
        if major == 10:
            return 8 if K % 512 == 0 else 4         # quantize_1x128_packed's kernels
        return 4                                    # fp8_gemm_run's scale_1x128_kernel
    raise KeyError(op)


def ops_for(major: int) -> list[str]:
    ops = ["q1x128_f32", "q1x128_ue8m0", "q1x128_packed", "q1x32", "q1x32_packed", "silu1x32", "linear_bf16",
           "dense_linear", "dense_lq", "dense_qa"]
    if major == 9:
        ops.insert(2, "q1x128_contract")
    if major in (9, 12):
        ops.append("linear_qx")
    if major in (10, 12):
        ops.append("raw_lq_chain")
    if major == 12:
        ops.append("bf16x")
    return ops


class Weights:
    """The N x K weight of the linear ops, in each op's form, built once per K."""

    def __init__(self, major: int, K: int):
        torch.manual_seed(K)
        w = (torch.randn(N, K, device="cuda") / K ** 0.5).bfloat16()
        self.w = w
        self.wq, self.sw = T.quantize_128x128(w, major == 12)
        self.dense = D.prepare_weight(w, format="bsfp8" if major == 9 else "mxfp8")


def x_numel(op: str, M: int, K: int) -> int:
    return M * 2 * K if op == "silu1x32" else M * K


def run(op: str, x: torch.Tensor, M: int, K: int, W: Weights):
    """The op on the flat bf16 buffer x (M * K elements, 2 * M * K for the SwiGLU form)."""
    if op == "silu1x32":
        return T.silu_chunk_mul_quantize_1x32(x.view(M, 2 * K), True)
    x2 = x.view(M, K)
    if op == "q1x128_f32":
        return T.quantize_1x128(x2, False)
    if op == "q1x128_ue8m0":
        return T.quantize_1x128(x2, True)
    if op == "q1x128_contract":
        return T.quantize_1x128_contract(x2)
    if op == "q1x128_packed":
        return T.quantize_1x128_packed(x2, True)
    if op == "q1x32":
        return T.quantize_1x32(x2, True)
    if op == "q1x32_packed":
        return T.quantize_1x32_packed(x2, True)
    if op == "linear_qx":
        return T.linear_qx(x2, W.wq, W.sw)
    if op == "linear_bf16":
        return T.linear_bf16(x2, W.w)
    if op == "bf16x":   # sm_120/sm_121: the bf16-x GEMM (G2 on an aligned x where the small-M rule R2 picks it)
        return T.linear_mxfp8_bf16x(x2, W.dense.weight, W.dense.scale)
    if op == "raw_lq_chain":   # the raw form of linear_quantized on a bf16 x: quantize, then the quantized GEMM
        xq, sx = T.quantize_1x32_packed(x2, True)
        return T.linear_mxfp8_quantized_raw(xq, W.dense.weight, sx, W.dense.scale)
    if op == "dense_linear":
        return D.linear(x2, W.dense)
    if op == "dense_lq":
        return D.linear_quantized(x2, W.dense)
    if op == "dense_qa":
        return D.quantize_activation(x2)
    raise KeyError(op)


def live(op: str, major: int, M: int, K: int, outs) -> list[torch.Tensor]:
    """The deterministic part of the op's outputs: everything, except the scales of the padded rows where the op
    leaves them unwritten (quantize_1x128 and quantize_1x128_packed's two-step path when K % 512 != 0)."""
    outs = list(outs) if isinstance(outs, (tuple, list)) else [outs]
    if K % 512 == 0 or op not in ("q1x128_f32", "q1x128_ue8m0", "q1x128_packed"):
        return outs
    q, s = outs
    kb, mp = K // 128, (M + 3) // 4 * 4
    if op != "q1x128_packed":
        return [q, s.reshape(-1)[: kb * mp].view(kb, mp)[:, :M]]  # fp32 scale (kb, m) at kb * pad(M, 4) + m
    if major == 10:   # the Sm1xx atom layout, one word per (row, K-block)
        m = torch.arange(M, device=s.device).view(-1, 1)
        kp = torch.arange(kb, device=s.device).view(1, -1)
        idx = ((m // 128) * kb + kp) * 128 + (m % 32) * 4 + (m % 128) // 32
        return [q, s.reshape(-1)[idx]]
    kw = (kb + 3) // 4
    return [q, s.reshape(-1)[: kw * mp].view(kw, mp)[:, :M]]  # int32 word (kw, m) at kw * pad(M, 4) + m


_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32}


def same_bytes(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    for u, v in zip(a, b):
        if u.dtype != v.dtype or tuple(u.shape) != tuple(v.shape):
            return False
        if not torch.equal(u.contiguous().view(_INT[u.element_size()]), v.contiguous().view(_INT[v.element_size()])):
            return False
    return True


def device_events(fn):
    """(exception or None, number of CUDA-side events: kernels, memcpys, memsets) of one call of fn."""
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    exc = None
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - the caller judges the type
            exc = e
        torch.cuda.synchronize()
    return exc, sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)


def case_ops(major: int) -> None:
    for K in KS:
        W = Weights(major, K)
        for op in ops_for(major):
            vb = need(op, major, K)
            row = []
            for M in MS:
                n = x_numel(op, M, K)
                torch.manual_seed(1009 * M + 17 * K + 3)
                vals = (torch.randn(n, device="cuda") * 0.1).bfloat16()
                ref = None
                ref_exc = None
                try:
                    ref = live(op, major, M, K, run(op, vals.clone(), M, K, W))
                except NotImplementedError as e:   # dense_lq: no route writes it in this build
                    ref_exc = e
                torch.cuda.synchronize()
                buf = torch.empty(n + 8, dtype=torch.bfloat16, device="cuda")
                verdicts = []
                for off in OFFSETS:
                    x = buf[off // 2: off // 2 + n]
                    x.copy_(vals)
                    if x.data_ptr() % vb != 0:
                        exc, ev = device_events(lambda: run(op, x, M, K, W))
                        ok = isinstance(exc, ValueError) and "aligned" in str(exc) and ev == 0
                        _check(ok, f"{op} K={K} M={M} offset {off}: expected ValueError and no device event, got "
                                   f"{type(exc).__name__ if exc else 'no exception'} ({str(exc)[:160]}), {ev} events")
                        counts["refused"] += 1
                        verdicts.append("R" if ok else "!")
                    else:
                        try:
                            got = live(op, major, M, K, run(op, x, M, K, W))
                            torch.cuda.synchronize()
                            ok = ref is not None and same_bytes(got, ref)
                            _check(ok, f"{op} K={K} M={M} offset {off}: accepted, but the bytes differ from the "
                                       f"aligned copy's")
                        except NotImplementedError as e:
                            ok = ref_exc is not None and type(e) is type(ref_exc)
                            _check(ok, f"{op} K={K} M={M} offset {off}: {e}")
                        counts["accepted"] += 1
                        verdicts.append("a" if ok else "!")
                # The context survived: the aligned call again gives its reference bytes.
                torch.cuda.synchronize()
                if ref is not None:
                    again = live(op, major, M, K, run(op, vals.clone(), M, K, W))
                    torch.cuda.synchronize()
                    _check(same_bytes(again, ref), f"{op} K={K} M={M}: the aligned call after the refusals differs")
                row.append(f"M={M}:{''.join(verdicts)}")
            print(f"  {op:16s} K={K:5d} need {vb:2d} B  offsets {OFFSETS}: {'  '.join(row)}  "
                  f"(R refused, a accepted with the aligned bytes)")
        del W
        torch.cuda.empty_cache()


def case_pitch(major: int) -> None:
    """The row pitch: quantize_1x128 (K need not be a multiple of 128) on [2, 1001] has rows 2002 bytes apart, which
    scale_1x128_kernel's 4-byte loads cannot read: refused; [2, 1002] runs and equals its aligned copy's live rows."""
    x = (torch.randn(2, 1001, device="cuda") * 0.1).bfloat16()
    exc, ev = device_events(lambda: T.quantize_1x128(x, False))
    ok = isinstance(exc, ValueError) and "pitch" in str(exc) and ev == 0
    _check(ok, f"pitch: [2, 1001] expected ValueError naming the row pitch, got {type(exc).__name__ if exc else None}"
               f" ({exc}), {ev} events")
    print(f"  pitch [2, 1001] (2002-byte rows): {type(exc).__name__ if exc else 'ran'}, {ev} device events  "
          f"{'OK' if ok else 'FAIL'}")
    x = (torch.randn(2, 1002, device="cuda") * 0.1).bfloat16()
    q, s = T.quantize_1x128(x, False)
    torch.cuda.synchronize()
    q2, s2 = T.quantize_1x128(x.clone(), False)
    ok = torch.equal(q.view(torch.uint8), q2.view(torch.uint8)) and torch.equal(
        s.reshape(-1)[: 8 * 4].view(8, 4)[:, :2], s2.reshape(-1)[: 8 * 4].view(8, 4)[:, :2])
    _check(ok, "pitch: [2, 1002] does not run deterministically")
    print(f"  pitch [2, 1002] (2004-byte rows): runs  {'OK' if ok else 'FAIL'}")


def case_noncontig(major: int) -> None:
    """fso.dense copies a non-contiguous bf16 x into a buffer of its own before any kernel reads it, so a column slice
    whose first element sits at a misaligned address runs, with the bytes of the same values in a tensor of their
    own."""
    M, K = 16, 4096
    W = Weights(major, K)
    torch.manual_seed(7)
    vals = (torch.randn(M, K, device="cuda") * 0.1).bfloat16()
    buf = torch.empty(M * (K + 8) + 8, dtype=torch.bfloat16, device="cuda")
    x = buf[3: 3 + M * (K + 8)].view(M, K + 8)[:, :K]
    x.copy_(vals)
    ok = x.data_ptr() % 16 != 0 and not x.is_contiguous()
    y = D.linear(x, W.dense)
    xq, sx = D.quantize_activation(x)
    xq0, sx0 = D.quantize_activation(vals.clone())
    ok = ok and same_bytes([y], [D.linear(vals.clone(), W.dense)]) and same_bytes([xq, sx], [xq0, sx0])
    torch.cuda.synchronize()
    _check(ok, "noncontig: a non-contiguous x at a misaligned address did not run with the aligned bytes")
    print(f"  noncontig: a column slice starting {x.data_ptr() % 16} bytes past a 16-byte boundary runs through "
          f"fso.dense.linear and quantize_activation with the aligned bytes  {'OK' if ok else 'FAIL'}")


def case_fp8_operands(major: int) -> None:
    """The pre-quantized activation of the raw GEMMs (linear_fp8 everywhere; linear_mxfp8_raw and
    linear_mxfp8_quantized_raw on sm_100/sm_103 and sm_120/sm_121) and its scale: TMA reads them, so an address that
    is not a multiple of 16 bytes is refused with ValueError before anything is launched (before the check, CuTe's
    host assertion aborted the process on sm_100 / sm_120), and the context stays usable. An aligned copy of the same
    bytes runs."""
    M, K = 16, 4096
    W = Weights(major, K)
    torch.manual_seed(11)
    x0 = (torch.randn(M, K, device="cuda") * 0.1).bfloat16()
    calls = []
    if major == 9:
        xq0, sx0 = T.quantize_1x128_contract(x0)
        calls.append(("linear_fp8", lambda xq, sx: T.linear_fp8(xq, W.wq, sx, W.sw)))
    else:
        xq0, sx0 = T.quantize_1x32_packed(x0, True)
        calls.append(("linear_mxfp8_raw", lambda xq, sx: T.linear_mxfp8_raw(xq, W.dense.weight, sx, W.dense.scale)))
        calls.append(("linear_mxfp8_quantized_raw",
                      lambda xq, sx: T.linear_mxfp8_quantized_raw(xq, W.dense.weight, sx, W.dense.scale)))
        xb0, sb0 = T.quantize_1x128_packed(x0, True)
        wq128, sw128 = T.quantize_128x128(W.w, True)
        sw128 = T.repack_fp8_wgt_scales(sw128, False)
        calls.append(("linear_fp8", lambda xq, sx: T.linear_fp8(xq, wq128, sx, sw128)))
    for name, fn in calls:
        xa, sa = (xb0, sb0) if (name == "linear_fp8" and major != 9) else (xq0, sx0)
        ref = fn(xa, sa)
        torch.cuda.synchronize()
        verdicts = []
        for off in (1, 2, 4, 8, 12):
            raw = torch.empty(xa.numel() + 32, dtype=torch.uint8, device="cuda")
            xq = raw[off: off + xa.numel()].view(torch.float8_e4m3fn).view(xa.shape)
            xq.copy_(xa)
            exc, ev = device_events(lambda: fn(xq, sa))
            ok = isinstance(exc, ValueError) and "TMA" in str(exc) and ev == 0
            _check(ok, f"fp8 {name} x at byte offset {off}: expected ValueError and no device event, got "
                       f"{type(exc).__name__ if exc else 'no exception'} ({str(exc)[:160]}), {ev} events")
            verdicts.append("R" if ok else "!")
        for off in (4, 8, 12):   # the scale words, 4 bytes each
            n = sa.numel()
            raw = torch.empty(n + 8, dtype=sa.dtype, device="cuda")
            sx = raw[off // sa.element_size(): off // sa.element_size() + n].as_strided(sa.shape, sa.stride())
            sx.copy_(sa)
            exc, ev = device_events(lambda: fn(xa, sx))
            ok = isinstance(exc, ValueError) and "TMA" in str(exc) and ev == 0
            _check(ok, f"fp8 {name} scale at byte offset {off}: expected ValueError and no device event, got "
                       f"{type(exc).__name__ if exc else 'no exception'} ({str(exc)[:160]}), {ev} events")
            verdicts.append("R" if ok else "!")
        again = fn(xa.clone(), sa.clone() if sa.is_contiguous() else sa)
        torch.cuda.synchronize()
        outs = list(ref) if isinstance(ref, (tuple, list)) else [ref]
        outs2 = list(again) if isinstance(again, (tuple, list)) else [again]
        ok = same_bytes(outs2, outs)
        _check(ok, f"fp8 {name}: the aligned call after the refusals differs")
        print(f"  fp8 x / scale  {name:28s} x offsets (1, 2, 4, 8, 12) and scale offsets (4, 8, 12): "
              f"{''.join(verdicts)}; aligned call after them gives its bytes  {'OK' if ok else 'FAIL'}")


def case_dense_compile_capture(major: int) -> None:
    """fso.dense on a misaligned bf16 x under torch.compile(fullgraph=True) (the op bodies run on real tensors, the
    fakes on fake ones) and inside a CUDA-graph capture (the copy of the misaligned x is captured with the call)."""
    M, K = 16, 4096
    W = Weights(major, K)
    torch.manual_seed(13)
    vals = (torch.randn(M, K, device="cuda") * 0.1).bfloat16()
    buf = torch.empty(M * K + 8, dtype=torch.bfloat16, device="cuda")
    x = buf[3: 3 + M * K].view(M, K)
    x.copy_(vals)
    ok_ptr = x.data_ptr() % 16 != 0
    torch._dynamo.reset()
    f = torch.compile(lambda t: D.linear(t, W.dense), fullgraph=True)
    y = f(x)
    torch.cuda.synchronize()
    ok = ok_ptr and same_bytes([y], [D.linear(vals.clone(), W.dense)])
    _check(ok, "compile: fso.dense.linear on a misaligned x under torch.compile differs from the aligned call")
    print(f"  compile  torch.compile(fullgraph=True) of fso.dense.linear on x {x.data_ptr() % 16} bytes past a 16-byte "
          f"boundary == the aligned call  {'OK' if ok else 'FAIL'}")
    torch._dynamo.reset()

    def call():
        q, sq = D.quantize_activation(x)
        return D.linear(x, W.dense), q, sq

    call()  # one eager call on the default stream
    torch.cuda.synchronize()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=st):
        out = call()
    torch.manual_seed(14)
    vals2 = (torch.randn(M, K, device="cuda") * 0.1).bfloat16()
    x.copy_(vals2)                      # new values in the same misaligned storage
    out[0].fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    ref = [D.linear(vals2.clone(), W.dense), *D.quantize_activation(vals2.clone())]
    ok = same_bytes(list(out), ref)
    _check(ok, "capture: the replay of fso.dense calls on a misaligned x differs from the eager aligned calls")
    print(f"  capture  linear + quantize_activation on the misaligned x captured on a side stream; replay after "
          f"rewriting x == the aligned calls on the new values  {'OK' if ok else 'FAIL'}")


def case_dense_prequant(major: int) -> None:
    """The pre-quantized entry of fso.dense is not copied: an fp8 x or an x_scale whose data pointer is not 16-byte
    aligned raises ValueError naming the 16-byte rule, with no device event."""
    M, K = 16, 4096
    W = Weights(major, K)
    torch.manual_seed(15)
    xq0, sx0 = D.quantize_activation((torch.randn(M, K, device="cuda") * 0.1).bfloat16())
    verdicts = []
    for off in (1, 4, 8):
        raw = torch.empty(xq0.numel() + 32, dtype=torch.uint8, device="cuda")
        xq = raw[off: off + xq0.numel()].view(torch.float8_e4m3fn).view(xq0.shape)
        xq.copy_(xq0)
        exc, ev = device_events(lambda: D.linear(xq, W.dense, x_scale=sx0))
        ok = isinstance(exc, ValueError) and "16-byte" in str(exc) and ev == 0
        _check(ok, f"prequant: fp8 x at byte offset {off}: {type(exc).__name__ if exc else 'no exception'} "
                   f"({str(exc)[:160]}), {ev} events")
        verdicts.append("R" if ok else "!")
    for off in (4, 8):
        es = sx0.element_size()
        raw = torch.empty(sx0.numel() + 8, dtype=sx0.dtype, device="cuda")
        sx = raw[off // es: off // es + sx0.numel()].as_strided(sx0.shape, sx0.stride())
        sx.copy_(sx0)
        exc, ev = device_events(lambda: D.linear(xq0, W.dense, x_scale=sx))
        ok = isinstance(exc, ValueError) and "16-byte" in str(exc) and ev == 0
        _check(ok, f"prequant: x_scale at byte offset {off}: {type(exc).__name__ if exc else 'no exception'} "
                   f"({str(exc)[:160]}), {ev} events")
        verdicts.append("R" if ok else "!")
    print(f"  prequant fso.dense.linear(xq, w, x_scale=sx): fp8 x at offsets (1, 4, 8) and x_scale at (4, 8): "
          f"{''.join(verdicts)} (R = ValueError naming 16 bytes, no device event)")


def case_control(major: int) -> None:
    """Positive control of the event counter: a served quantize shows its kernel under the profiler."""
    x = (torch.randn(16, 4096, device="cuda") * 0.1).bfloat16()
    exc, ev = device_events(lambda: D.quantize_activation(x))
    ok = exc is None and ev >= 1
    _check(ok, f"control: a served quantize_activation raised {exc} or showed {ev} device events")
    print(f"  control: a served quantize_activation shows {ev} device event(s)  {'OK' if ok else 'FAIL'}")


# ----------------------------------------------------------------------------------------------------------------------
# Issue #12 cleanup CL1: the MoE glue raw ops, fso.moe.layer and fso.compat.linear_mxfp8.

GLUE_E, GLUE_TOPK, GLUE_I = 8, 2, 512
GLUE_MS = (1, 16, 256)
GLUE_KS = (2048, 1280)


def glue_need(op: str) -> int:
    """The width (bytes) of the loads with which the kernel of a MoE glue op reads its bf16 input, whatever the
    pointer. From the kernels in csrc/gemm/ops/quant_kernels.cu and moe_glue.cu, written here independently of the
    op bodies' checks: four bf16 per lane in the gather and [gate | up] SwiGLU quantizers (LDG.64), eight interleaved
    values per lane in the pairwise SwiGLU form and eight bf16 per thread in the combines (LDG.128)."""
    if op in ("silu1x32_pw", "silu1x32_out_pw", "combine_dn", "combine_bias", "combine_sorted_dn"):
        return 16
    return 8


def glue_ops(major: int) -> list[str]:
    ops = ["q1x32_gather", "q1x32_gather_out", "silu1x32", "silu1x32_out", "combine_dn", "combine_bias"]
    if major in (10, 12):
        ops += ["silu1x32_pw", "silu1x32_out_pw"]   # the interleaved FC1 order exists on sm_100/103 and sm_120/121
    if major == 9:
        ops += ["q1x128_gather_sm90", "q1x128_sorted_sm90", "silu1x128_sm90", "silu1x128_sorted_sm90",
                "combine_sorted_dn"]
    return ops


class GlueInputs:
    """One routed batch: M tokens, top-2 of 8 experts (distinct per token), the masked routing of moe_build_routing
    (row capacity align4(M * topk)) and the sorted routing of moe_build_sorted (64-row blocks), with the bf16
    operands at their aligned addresses."""

    def __init__(self, M: int, K: int):
        g = torch.Generator(device="cuda").manual_seed(4049 * M + K)
        self.M, self.K = M, K
        self.ids = torch.rand(M, GLUE_E, device="cuda", generator=g).topk(GLUE_TOPK, dim=1).indices.int().contiguous()
        self.wts = torch.softmax(torch.rand(M, GLUE_TOPK, device="cuda", generator=g), dim=1).float().contiguous()
        self.m_cap = max(4, (M * GLUE_TOPK + 3) // 4 * 4)
        _mm, _rm, self.slot_of_flat, _s, _ps, _w = T.moe_build_routing(self.ids, GLUE_E, self.m_cap, False, [], None)
        self.slots = self.slot_of_flat[self.slot_of_flat >= 0].long()
        self.se, self.fts, _np = T.moe_build_sorted(self.ids, GLUE_E, 64)
        self.p_max = int(self.se.shape[0])
        self.rows = self.fts[self.fts >= 0].long()
        self.x = (torch.randn(M, K, device="cuda", generator=g) * 0.1).bfloat16()
        self.gu = torch.randn(GLUE_E, self.m_cap, 2 * GLUE_I, device="cuda", generator=g).bfloat16()
        self.gu_sorted = torch.randn(self.p_max, 2 * GLUE_I, device="cuda", generator=g).bfloat16()
        self.dn = torch.randn(GLUE_E, self.m_cap, K, device="cuda", generator=g).bfloat16()
        self.dn_sorted = torch.randn(self.p_max, K, device="cuda", generator=g).bfloat16()
        self.bias = (torch.randn(M, K, device="cuda", generator=g) * 0.1).bfloat16()
        # The caller-owned outputs of the out-style ops (the arch's scale words: one Sm1xx atom slab per group of
        # pad(m_cap, 128) rows on sm_100/103, K-major words elsewhere), allocated here so that a refused call
        # launches nothing at all.
        major = torch.cuda.get_device_capability()[0]
        rows_sf = (self.m_cap + 127) // 128 * 128 if major == 10 else self.m_cap
        self.xq_out = torch.zeros(GLUE_E * self.m_cap * K, dtype=torch.float8_e4m3fn, device="cuda")
        self.sc_out = torch.zeros(GLUE_E * rows_sf * (K // 128), dtype=torch.int32, device="cuda")
        self.dq_out = torch.zeros(GLUE_E * self.m_cap * GLUE_I, dtype=torch.float8_e4m3fn, device="cuda")
        self.sd_out = torch.zeros(GLUE_E * rows_sf * (GLUE_I // 128), dtype=torch.int32, device="cuda")

    def operand(self, op: str) -> torch.Tensor:
        """The bf16 operand the op's alignment rule is about."""
        if op.startswith("silu1x128_sorted"):
            return self.gu_sorted
        if op.startswith("silu"):
            return self.gu
        if op == "combine_dn":
            return self.dn
        if op == "combine_sorted_dn":
            return self.dn_sorted
        if op == "combine_bias":
            return self.bias
        return self.x


def _grouped_live(major: int, xq, sc, slots, m_cap: int, kp: int, fp32: bool) -> list:
    """The routed rows of a masked-layout quantize output and their scales (the rows no pair maps to are undefined):
    fp32 [G, kp, m_cap] (sm_90 1x128), the Sm1xx atom slab per group (sm_100/103) or K-major words per group."""
    g, m_in = slots // m_cap, slots % m_cap
    kps = torch.arange(kp, device=slots.device)
    if not fp32 and major == 10:
        m_pad = (m_cap + 127) // 128 * 128
        r = m_in % 128
        idx = g[:, None] * (kp * m_pad) + ((m_in // 128)[:, None] * kp + kps[None, :]) * 128 \
            + (r % 32)[:, None] * 4 + (r // 32)[:, None]
    else:
        idx = (g[:, None] * kp + kps[None, :]) * m_cap + m_in[:, None]
    return [xq.reshape(-1, xq.shape[-1])[slots], sc.reshape(-1)[idx]]


def run_glue(op: str, major: int, B: GlueInputs, t: torch.Tensor, prep: bool = True) -> list:
    """The glue op with its bf16 operand replaced by `t`; the deterministic part of the result. `prep` False: the
    out-style ops' buffers are not cleared first, so a call the op refuses launches nothing."""
    K, I = B.K, GLUE_I
    if op == "q1x32_gather":
        xq, sc = T.quantize_1x32_grouped_gather(t, B.slot_of_flat, GLUE_TOPK, GLUE_E, B.m_cap, True)
        return _grouped_live(major, xq, sc, B.slots, B.m_cap, K // 128, False)
    if op == "q1x32_gather_out":
        if prep:
            B.xq_out.zero_()
            B.sc_out.zero_()
        T.quantize_1x32_grouped_gather_out(t, B.slot_of_flat, GLUE_TOPK, GLUE_E, B.m_cap, True, B.xq_out, B.sc_out)
        return [B.xq_out.clone(), B.sc_out.clone()]
    if op in ("silu1x32", "silu1x32_pw"):
        xq, sc = T.silu_chunk_mul_quantize_1x32_grouped(t, B.slot_of_flat, True, op.endswith("_pw"))
        return _grouped_live(major, xq, sc, B.slots, B.m_cap, I // 128, False)
    if op in ("silu1x32_out", "silu1x32_out_pw"):
        if prep:
            B.dq_out.zero_()
            B.sd_out.zero_()
        T.silu_chunk_mul_quantize_1x32_grouped_out(t, B.slot_of_flat, True, op.endswith("_pw"), B.dq_out, B.sd_out)
        return [B.dq_out.clone(), B.sd_out.clone()]
    if op == "q1x128_gather_sm90":
        xq, sc = T.quantize_1x128_grouped_gather_sm90(t, B.slot_of_flat, GLUE_TOPK, GLUE_E, B.m_cap)
        return _grouped_live(major, xq, sc, B.slots, B.m_cap, K // 128, True)
    if op == "q1x128_sorted_sm90":
        xq, sc = T.quantize_1x128_sorted_gather_sm90(t, B.fts, B.p_max, GLUE_TOPK, None, None)
        kps = torch.arange(K // 128, device="cuda")
        return [xq[B.rows], sc.reshape(-1)[kps[None, :] * sc.shape[-1] + B.rows[:, None]]]
    if op == "silu1x128_sm90":
        xq, sc = T.silu_chunk_mul_quantize_1x128_grouped_sm90(t, B.slot_of_flat)
        return _grouped_live(major, xq, sc, B.slots, B.m_cap, I // 128, True)
    if op == "silu1x128_sorted_sm90":
        xq, sc = T.silu_chunk_mul_quantize_1x128_sorted_sm90(t, B.fts)
        kps = torch.arange(I // 128, device="cuda")
        return [xq[B.rows], sc.reshape(-1)[kps[None, :] * sc.shape[-1] + B.rows[:, None]]]
    if op == "combine_dn":
        return [T.moe_combine(t, B.slot_of_flat, B.wts, B.bias, None, None)]
    if op == "combine_bias":
        return [T.moe_combine(B.dn, B.slot_of_flat, B.wts, t, None, None)]
    if op == "combine_sorted_dn":
        return [T.moe_combine_sorted(t, B.fts, B.wts)]
    raise KeyError(op)


def _at_offset(t: torch.Tensor, off: int) -> torch.Tensor:
    """t's values, shape and strides in fresh storage whose data pointer is `off` bytes past a 16-byte boundary."""
    es = t.element_size()
    span = sum((s - 1) * st for s, st in zip(t.shape, t.stride())) + 1
    raw = torch.empty(span + off // es + 64, dtype=t.dtype, device=t.device)
    v = raw[off // es:].as_strided(t.shape, t.stride())
    v.copy_(t)
    return v


def case_moe_glue(major: int) -> None:
    """Item 1 of CL1: the MoE gather / SwiGLU quantizer raw ops and the two combines refuse, with ValueError naming
    the requirement and before anything is launched, every bf16 input their kernel cannot read: at every even byte
    offset 2..14 that is not a multiple of the kernel's width (glue_need); the offset 8 that an 8-byte kernel reads
    runs and gives the aligned copy's bytes, as on the trunk. After each op's offsets the context is intact and the
    aligned call gives its reference bytes again."""
    for K in GLUE_KS:
        for M in GLUE_MS:
            B = GlueInputs(M, K)
            for op in glue_ops(major):
                vb = glue_need(op)
                src = B.operand(op)
                ref = run_glue(op, major, B, src)
                torch.cuda.synchronize()
                verdicts = []
                for off in OFFSETS:
                    t = _at_offset(src, off)
                    if off % vb != 0:
                        exc, ev = device_events(lambda: run_glue(op, major, B, t, prep=False))
                        ok = isinstance(exc, ValueError) and "aligned" in str(exc) and ev == 0
                        _check(ok, f"glue {op} K={K} M={M} offset {off}: expected ValueError and no device event, got "
                                   f"{type(exc).__name__ if exc else 'no exception'} ({str(exc)[:160]}), {ev} events")
                        counts["refused"] += 1
                        verdicts.append("R" if ok else "!")
                    else:
                        got = run_glue(op, major, B, t)
                        torch.cuda.synchronize()
                        ok = same_bytes(got, ref)
                        _check(ok, f"glue {op} K={K} M={M} offset {off}: accepted, but the bytes differ from the "
                                   f"aligned call's")
                        counts["accepted"] += 1
                        verdicts.append("a" if ok else "!")
                again = run_glue(op, major, B, src.clone())
                torch.cuda.synchronize()
                _check(same_bytes(again, ref), f"glue {op} K={K} M={M}: the aligned call after the refusals differs")
                print(f"  glue {op:22s} K={K:5d} M={M:4d} need {vb:2d} B  offsets {OFFSETS}: {''.join(verdicts)}")
    torch.cuda.empty_cache()


class LayerSetup:
    """One MoE layer's experts for fso.moe.layer: E 8, top-2, H 2048, I 512; kind "bsfp8" on sm_90 (block-FP8
    experts, kept as they are), "mxfp8" on sm_100/103 and sm_120/121 (bf16 experts quantized at load time)."""

    def __init__(self, major: int):
        E, H, I = GLUE_E, 2048, GLUE_I
        g = torch.Generator(device="cuda").manual_seed(1234)
        w13 = (torch.randn(E, 2 * I, H, device="cuda", generator=g) / H ** 0.5).bfloat16()
        w2 = (torch.randn(E, H, I, device="cuda", generator=g) / I ** 0.5).bfloat16()
        if major == 9:
            def bq(w):
                e, n, k = w.shape
                wf = w.float().view(e, n // 128, 128, k // 128, 128)
                amax = wf.abs().amax(dim=(2, 4))
                sc = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
                return (wf / sc[:, :, None, :, None]).view(e, n, k).to(torch.float8_e4m3fn), sc.contiguous()
            q13, s13 = bq(w13)
            q2, s2 = bq(w2)
            self.ex = fso.moe.prepare_experts(q13, q2, format="bsfp8", sw13=s13, sw2=s2)
        else:
            self.ex = fso.moe.prepare_experts(w13, w2, format="mxfp8")
        self.H = H

    def inputs(self, M: int, seed: int):
        g = torch.Generator(device="cuda").manual_seed(seed)
        hidden = (torch.randn(M, self.H, device="cuda", generator=g) * 0.1).bfloat16()
        ids = torch.rand(M, GLUE_E, device="cuda", generator=g).topk(GLUE_TOPK, dim=1).indices.int().contiguous()
        wts = torch.softmax(torch.rand(M, GLUE_TOPK, device="cuda", generator=g), dim=1).float().contiguous()
        bias = (torch.randn(M, self.H, device="cuda", generator=g) * 0.1).bfloat16()
        return hidden, ids, wts, bias

    def raw_op(self, hidden, ids, wts, bias=None):
        """torch.ops.fish_scales_ops.moe_layer called directly, past fso.moe.layer's own conditioning."""
        ex = self.ex
        return torch.ops.fish_scales_ops.moe_layer(hidden, ex.w13, ex.sw13, ex.w2, ex.sw2, ids, wts, bias, None,
                                                   ex.kind, ex.w13_interleaved)


def case_moe_layer(major: int) -> None:
    """Item 2 of CL1: fso.moe.layer never hands a misaligned bf16 operand to a kernel. A hidden (both kinds) or a
    bias (the MXFP8 layer, whose combine reads it in 16-byte vectors) that is not 16-byte aligned or not contiguous is
    copied into a tensor of its own in the op body, so every even offset 2..14 gives the aligned call's bytes, at
    M in {1, 16, 80, 256}; the same through the registered op called directly (a misaligned hidden, a
    non-contiguous hidden at a misaligned address), under torch.compile(fullgraph=True), and in a CUDA-graph replay
    after the input's values change in the same misaligned storage."""
    L = LayerSetup(major)
    ex = L.ex
    for M in (1, 16, 80, 256):
        hidden, ids, wts, bias = L.inputs(M, 77 + M)
        ref = [fso.moe.layer(hidden, ex, ids, wts)]
        ref_b = [fso.moe.layer(hidden, ex, ids, wts, bias=bias)]
        torch.cuda.synchronize()
        verdicts = []
        for off in OFFSETS:
            ok = same_bytes([fso.moe.layer(_at_offset(hidden, off), ex, ids, wts)], ref)
            ok = ok and same_bytes([fso.moe.layer(hidden, ex, ids, wts, bias=_at_offset(bias, off))], ref_b)
            ok = ok and same_bytes([L.raw_op(_at_offset(hidden, off), ids, wts)], ref)
            torch.cuda.synchronize()
            _check(ok, f"moe.layer kind {ex.kind} M={M} offset {off}: a misaligned hidden or bias does not give the "
                       f"aligned call's bytes")
            verdicts.append("a" if ok else "!")
        print(f"  moe.layer kind {ex.kind} M={M:4d}: hidden, bias and the registered op's hidden at offsets {OFFSETS}: "
              f"{''.join(verdicts)} (a = the aligned call's bytes)")
    # A non-contiguous hidden handed to the registered op directly: a column slice whose first element is 6 bytes past
    # a 16-byte boundary.
    M = 16
    hidden, ids, wts, bias = L.inputs(M, 99)
    raw = torch.empty(M * (L.H + 8) + 8, dtype=torch.bfloat16, device="cuda")
    nc = raw[3: 3 + M * (L.H + 8)].view(M, L.H + 8)[:, :L.H]
    nc.copy_(hidden)
    ok = not nc.is_contiguous() and nc.data_ptr() % 16 != 0
    ok = ok and same_bytes([L.raw_op(nc, ids, wts)], [fso.moe.layer(hidden.clone(), ex, ids, wts)])
    torch.cuda.synchronize()
    _check(ok, f"moe.layer kind {ex.kind}: the registered op on a non-contiguous misaligned hidden differs")
    print(f"  moe.layer kind {ex.kind}: the registered op on a non-contiguous hidden {nc.data_ptr() % 16} bytes past a "
          f"16-byte boundary == the aligned call  {'OK' if ok else 'FAIL'}")
    # torch.compile(fullgraph=True): the op is opaque, its fake runs on fake tensors and its body on the real ones.
    buf = torch.empty(M * L.H + 8, dtype=torch.bfloat16, device="cuda")
    x = buf[3: 3 + M * L.H].view(M, L.H)
    x.copy_(hidden)
    torch._dynamo.reset()
    f = torch.compile(lambda t: fso.moe.layer(t, ex, ids, wts), fullgraph=True)
    y = f(x)
    torch.cuda.synchronize()
    ok = x.data_ptr() % 16 != 0 and same_bytes([y], [fso.moe.layer(hidden.clone(), ex, ids, wts)])
    _check(ok, f"moe.layer kind {ex.kind}: torch.compile on a misaligned hidden differs from the aligned call")
    print(f"  moe.layer kind {ex.kind}: torch.compile(fullgraph=True) on a hidden {x.data_ptr() % 16} bytes past a "
          f"16-byte boundary == the aligned call  {'OK' if ok else 'FAIL'}")
    torch._dynamo.reset()
    # CUDA-graph capture: the copy of the misaligned hidden is captured with the call.
    fso.moe.layer(x, ex, ids, wts)          # one eager call of the shape
    torch.cuda.synchronize()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=st):
        out = fso.moe.layer(x, ex, ids, wts)
    hidden2, _ids, _wts, _b = L.inputs(M, 100)
    x.copy_(hidden2)                         # new values in the same misaligned storage
    out.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    ok = same_bytes([out], [fso.moe.layer(hidden2.clone(), ex, ids, wts)])
    _check(ok, f"moe.layer kind {ex.kind}: the replay on a misaligned hidden differs from the eager aligned call")
    print(f"  moe.layer kind {ex.kind}: captured on a misaligned hidden; replay after rewriting it == the aligned call "
          f"on the new values  {'OK' if ok else 'FAIL'}")
    del g, L
    torch.cuda.empty_cache()


# One dense shape per sm_100 tier, as _sm100_dispatch.plan names them on the B300 (issue #12 run CL1,
# probe/b300_trunk_compat.jsonl); the DSL tier exists only where its kernel source is found (FSO_DSL_KERNEL_PATH, a
# wheel's copy or the source tree's CUTLASS examples), and plan() says so.
COMPAT_TIER_SHAPES = {"sm100.decode": (16, 2048, 4096), "sm100.smm": (256, 6144, 2048),
                      "sm100.dsl": (4096, 4096, 4096), "sm100.cascade": (128, 2560, 9728)}


def case_compat_prequant(major: int) -> None:
    """Item 3 of CL1: fso.compat.linear_mxfp8 (the compat wrapper) applies the flat 16-byte rule of fso.dense's
    pre-quantized entry to its x_fp8 and sx itself, before routing: a pointer that is not a multiple of 16 bytes
    raises ValueError naming the operand and the offset, with no device event, on a cell of every sm_100 tier (decode
    row, cuBLAS scaled_mm, CuTe-DSL, C++ cascade) and on sm_120/121 (the raw-op path); a row-strided x_fp8 whose first
    element is misaligned is refused the same way; the aligned call after the refusals gives its bytes."""
    if major == 9:
        print("  compat linear_mxfp8: sm_90 has no MXFP8 (NotImplementedError), not run")
        return
    shapes = list(COMPAT_TIER_SHAPES.items()) if major == 10 else [("sm120.mxfp8", (16, 2048, 4096)),
                                                                     ("sm120.mxfp8", (256, 6144, 2048))]
    for want, (m, n, k) in shapes:
        tier = want
        if major == 10:
            from fish_scales_ops.gemm import _sm100_dispatch
            tier = _sm100_dispatch.plan(m, n, k)
        g = torch.Generator(device="cuda").manual_seed(m * 1009 + n * 17 + k)
        w = (torch.randn(n, k, device="cuda", generator=g) / k ** 0.5).bfloat16()
        dw = D.prepare_weight(w, format="mxfp8")
        xq0, sx0 = T.quantize_1x32_packed((torch.randn(m, k, device="cuda", generator=g) * 0.1).bfloat16(), True)
        L = fso.compat.linear_mxfp8
        ref = L(xq0, dw.weight, sx0, dw.scale)
        torch.cuda.synchronize()
        verdicts = []
        cases = [("x_fp8", off) for off in (1, 2, 4, 8)] + [("sx", off) for off in (4, 8, 12)] + [("x_fp8 rows", 8)]
        for name, off in cases:
            if name == "x_fp8":
                args = (_at_offset(xq0, off), dw.weight, sx0, dw.scale)
            elif name == "sx":
                args = (xq0, dw.weight, _at_offset(sx0, off), dw.scale)
            else:   # rows k + 16 bytes apart, unit last stride, the first element `off` bytes past a 16-byte boundary
                rawx = torch.empty(m * (k + 16) + 64, dtype=torch.uint8, device="cuda")
                v = rawx[off: off + m * (k + 16)].view(torch.float8_e4m3fn).view(m, k + 16)[:, :k]
                v.copy_(xq0)
                args = (v, dw.weight, sx0, dw.scale)
            exc, ev = device_events(lambda: L(*args))
            oname = name.split()[0]
            ok = isinstance(exc, ValueError) and "16-byte" in str(exc) and oname in str(exc) and ev == 0
            _check(ok, f"compat {want} {m}x{n}x{k} ({tier}) {name} at byte offset {off}: expected ValueError naming "
                       f"{oname} and no device event, got {type(exc).__name__ if exc else 'no exception'} "
                       f"({str(exc)[:160]}), {ev} events")
            counts["refused"] += 1
            verdicts.append("R" if ok else "!")
        # Accepted: a row-strided x_fp8 at a 16-byte aligned address is copied before routing and gives the
        # contiguous call's bytes (on the trunk the CuTe-DSL tiers read it in place and gave other bytes); an sx at a
        # 16-byte offset runs on every tier, the DSL tier handing it to the C++ cascade (it takes 32-byte aligned
        # scales only), whose bytes the raw op gives.
        rawx = torch.empty(m * (k + 16) + 64, dtype=torch.uint8, device="cuda")
        v = rawx[: m * (k + 16)].view(torch.float8_e4m3fn).view(m, k + 16)[:, :k]
        v.copy_(xq0)
        got = L(v, dw.weight, sx0, dw.scale)
        torch.cuda.synchronize()
        ok_rows = not v.is_contiguous() and v.data_ptr() % 16 == 0 and same_bytes([got], [ref])
        _check(ok_rows, f"compat {want} {m}x{n}x{k} ({tier}): a row-strided aligned x_fp8 differs from the contiguous "
                        f"call")
        sx16 = _at_offset(sx0, 16)
        got = L(xq0, dw.weight, sx16, dw.scale)
        torch.cuda.synchronize()
        want16 = ref
        if tier == "sm100.dsl" and sx16.data_ptr() % 32 != 0:
            want16 = T.linear_mxfp8_raw(xq0, dw.weight, sx0, dw.scale)
            torch.cuda.synchronize()
        ok16 = same_bytes([got], [want16])
        _check(ok16, f"compat {want} {m}x{n}x{k} ({tier}): an sx at a 16-byte offset did not give the bytes of "
                     f"{'the C++ cascade' if want16 is not ref else 'the aligned call'}")
        verdicts.append(("a" if ok_rows else "!") + ("a" if ok16 else "!"))
        counts["accepted"] += 2
        again = L(xq0.clone(), dw.weight, sx0.clone() if sx0.is_contiguous() else sx0, dw.scale)
        torch.cuda.synchronize()
        ok = same_bytes([again], [ref])
        _check(ok, f"compat {want} {m}x{n}x{k}: the aligned call after the refusals differs")
        note = ""
        if major == 10 and tier != want:
            note = (f" (the router's plan for this shape is {tier}, not {want}: in this process that tier is not "
                    f"available, e.g. the CuTe-DSL kernel source is missing; set FSO_DSL_KERNEL_PATH to cover it)")
        print(f"  compat linear_mxfp8 {m}x{n}x{k} on {tier:13s} x_fp8 offsets (1, 2, 4, 8), sx offsets (4, 8, 12), "
              f"row-strided x_fp8 at 8: {''.join(verdicts[:-1])} (R refused); row-strided x_fp8 at 0, sx at 16: "
              f"{verdicts[-1]} (a run with the expected bytes); aligned call after them gives its bytes  "
              f"{'OK' if ok else 'FAIL'}{note}")
        del dw, w
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------------------------
# CL2: the weights, the weight scales and the caller-owned outputs (cases 9-13).
# ----------------------------------------------------------------------------------------------------------------------

def _refused(fn, names: tuple, what: str) -> str:
    """fn raises ValueError naming one of `names`, and launches nothing (no device event)."""
    exc, ev = device_events(fn)
    ok = isinstance(exc, ValueError) and any(n in str(exc) for n in names) and ev == 0
    _check(ok, f"{what}: expected ValueError naming {'/'.join(names)} and no device event, got "
               f"{type(exc).__name__ if exc else 'no exception'} ({str(exc)[:160]}), {ev} events")
    counts["refused"] += 1
    return "R" if ok else "!"


def _same(fn, ref: list, what: str) -> str:
    """fn runs and gives the bytes of `ref` (the aligned call's)."""
    got = fn()
    torch.cuda.synchronize()
    got = list(got) if isinstance(got, (tuple, list)) else [got]
    ok = same_bytes(got, ref)
    _check(ok, f"{what}: accepted, but the bytes differ from the aligned call's")
    counts["accepted"] += 1
    return "a" if ok else "!"


def _ref(fn) -> list:
    r = fn()
    torch.cuda.synchronize()
    return list(r) if isinstance(r, (tuple, list)) else [r]


def case_dense_weights(major: int) -> None:
    """9. (CL2) The weight and the int32 weight scale of the raw dense GEMMs, which TMA reads (linear_fp8; linear_qx
    on sm_90 and sm_120/121; linear_mxfp8_raw and linear_mxfp8_quantized_raw on sm_100/103 and sm_120/121;
    linear_mxfp8_bf16x on sm_120/121, whose small-M form reads them in place too): an
    address that is not a multiple of 16 bytes raises ValueError naming the operand, before anything is launched,
    and the aligned call after it gives its bytes. What runs keeps running with the aligned call's bytes: an fp32
    weight scale at any 4-byte offset (sm_90 reads it with plain loads; sm_100/103 and sm_120/121 repack it into a
    tensor of their own), a weight at a 16-byte offset, and linear_bf16's bf16 weight at any even offset (its weight
    quantize reads one element per load)."""
    M, K, Nw = 16, 4096, 2048
    g = torch.Generator(device="cuda").manual_seed(4111)
    w = (torch.randn(Nw, K, device="cuda", generator=g) / K ** 0.5).bfloat16()
    x = (torch.randn(M, K, device="cuda", generator=g) * 0.1).bfloat16()
    calls = []   # (name, fn(wq, sw), wq0, sw0, int32 scale offsets refused, fp32 scale offsets accepted)
    if major == 9:
        wq0, sw0 = T.quantize_128x128(w, False)
        xq0, sx0 = T.quantize_1x128_contract(x)
        calls.append(("linear_fp8", lambda wq, sw: [T.linear_fp8(xq0, wq, sx0, sw)], wq0, sw0, (), (4, 8, 12)))
        calls.append(("linear_qx", lambda wq, sw: [T.linear_qx(x, wq, sw)], wq0, sw0, (), (4, 8, 12)))
    else:
        wq128, sw32 = T.quantize_128x128(w, True)
        swp = T.repack_fp8_wgt_scales(sw32, False)
        xb, sb = T.quantize_1x128_packed(x, True)
        calls.append(("linear_fp8", lambda wq, sw: [T.linear_fp8(xb, wq, sb, sw)], wq128, swp, (4, 8, 12), ()))
        xf, sf = T.quantize_1x128(x, True)
        calls.append(("linear_fp8 fp32-scales", lambda wq, sw: [T.linear_fp8(xf, wq, sf, sw)], wq128, sw32, (),
                      (4, 8, 12)))
        dw = D.prepare_weight(w, format="mxfp8")
        xm, sm = T.quantize_1x32_packed(x, True)
        calls.append(("linear_mxfp8_raw", lambda wq, sw: [T.linear_mxfp8_raw(xm, wq, sm, sw)], dw.weight, dw.scale,
                      (4, 8, 12), ()))
        calls.append(("linear_mxfp8_quantized_raw", lambda wq, sw: list(T.linear_mxfp8_quantized_raw(xm, wq, sm, sw)),
                      dw.weight, dw.scale, (4, 8, 12), ()))
        if major == 12:
            calls.append(("linear_qx", lambda wq, sw: [T.linear_qx(x, wq, sw)], wq128, sw32, (), (4, 8, 12)))
            # the bf16-x GEMM: its small-M form G2 and its linear_mxfp8_raw form read the weight in place alike
            calls.append(("linear_mxfp8_bf16x", lambda wq, sw: [T.linear_mxfp8_bf16x(x, wq, sw)], dw.weight,
                          dw.scale, (4, 8, 12), ()))
    for name, fn, wq0, sw0, s_refused, s_accepted in calls:
        ref = _ref(lambda: fn(wq0, sw0))
        v = ""
        for off in (1, 2, 4, 8, 12):
            t = _at_offset(wq0, off)   # built before the profiled call: its copy kernel is not the op's
            v += _refused(lambda: fn(t, sw0), ("w_fp8",), f"{name} weight at byte offset {off}")
        for off in s_refused:
            t = _at_offset(sw0, off)
            v += _refused(lambda: fn(wq0, t), ("sw",), f"{name} int32 scale at offset {off}")
        for off in s_accepted:
            t = _at_offset(sw0, off)
            v += _same(lambda: fn(wq0, t), ref, f"{name} fp32 scale at offset {off}")
        t = _at_offset(wq0, 16)
        v += _same(lambda: fn(t, sw0), ref, f"{name} weight at a 16-byte offset")
        v += _same(lambda: fn(wq0.clone(), sw0.clone()), ref, f"{name} aligned call after the refusals")
        print(f"  dense weight  {name:28s} w at (1, 2, 4, 8, 12), int32 sw at {s_refused}, fp32 sw at {s_accepted}, "
              f"w at 16, aligned again: {v}")
    ref = _ref(lambda: T.linear_bf16(x, w))
    v = "".join(_same(lambda: T.linear_bf16(x, _at_offset(w, off)), ref, f"linear_bf16 weight at offset {off}")
                for off in (2, 4, 8, 12))
    print(f"  dense weight  {'linear_bf16':28s} bf16 w at (2, 4, 8, 12): {v} (a = the aligned call's bytes)")
    torch.cuda.empty_cache()


def case_compat_weights(major: int) -> None:
    """10. (CL2) fso.compat.linear_mxfp8 applies the flat 16-byte rule to its w_fp8 and sw too, on the tensors as they
    are passed and before routing, so that the outcome does not depend on the tier a shape lands on: ValueError
    naming the operand, no device event, on a cell of every sm_100 tier and on sm_120/121. A weight scale at a
    16-byte offset runs on every tier: the CuTe-DSL tier, which takes 32-byte aligned scales only, hands it to the
    C++ cascade, whose bytes the raw op gives."""
    if major == 9:
        print("  compat linear_mxfp8 weights: sm_90 has no MXFP8 (NotImplementedError), not run")
        return
    shapes = list(COMPAT_TIER_SHAPES.items()) if major == 10 else [("sm120.mxfp8", (16, 2048, 4096)),
                                                                     ("sm120.mxfp8", (256, 6144, 2048))]
    L = fso.compat.linear_mxfp8
    for want, (m, n, k) in shapes:
        tier = want
        if major == 10:
            from fish_scales_ops.gemm import _sm100_dispatch
            tier = _sm100_dispatch.plan(m, n, k)
        g = torch.Generator(device="cuda").manual_seed(m * 1009 + n * 17 + k)
        w = (torch.randn(n, k, device="cuda", generator=g) / k ** 0.5).bfloat16()
        dw = D.prepare_weight(w, format="mxfp8")
        xq0, sx0 = T.quantize_1x32_packed((torch.randn(m, k, device="cuda", generator=g) * 0.1).bfloat16(), True)
        ref = _ref(lambda: L(xq0, dw.weight, sx0, dw.scale))
        v = ""
        for off in (1, 2, 4, 8, 12):
            t = _at_offset(dw.weight, off)
            v += _refused(lambda: L(xq0, t, sx0, dw.scale), ("w_fp8",), f"compat {m}x{n}x{k} ({tier}) w_fp8 at {off}")
        for off in (4, 8, 12):
            t = _at_offset(dw.scale, off)
            v += _refused(lambda: L(xq0, dw.weight, sx0, t), ("sw",), f"compat {m}x{n}x{k} ({tier}) sw at {off}")
        sw16 = _at_offset(dw.scale, 16)
        want16 = ref
        if tier == "sm100.dsl" and sw16.data_ptr() % 32 != 0:
            want16 = _ref(lambda: T.linear_mxfp8_raw(xq0, dw.weight, sx0, dw.scale))
        v += _same(lambda: L(xq0, dw.weight, sx0, sw16), want16, f"compat {m}x{n}x{k} ({tier}) sw at a 16-byte offset")
        v += _same(lambda: L(xq0, dw.weight.clone(), sx0, dw.scale.clone()), ref,
                   f"compat {m}x{n}x{k} ({tier}) aligned call after the refusals")
        print(f"  compat linear_mxfp8 {m}x{n}x{k} on {tier:13s} w_fp8 at (1, 2, 4, 8, 12), sw at (4, 8, 12), "
              f"sw at 16, aligned again: {v}")
    torch.cuda.empty_cache()


def _with_op_arg_at_offset(opname: str, arg: str, off: int, fn, nth: int = 0):
    """Run fn() (a layer call) with the `arg` argument of the nth call of torch.ops.fish_scales_ops.<opname> replaced
    by a copy at byte offset `off` (an output is copied back after the call); the layer's own routing feeds the op.
    Returns (exception or None, number of device events of that one op call, whether the op was called with it, the
    layer's result). The per-call pointer arrays of the sm_100 pointer-array route were built from the layer's own
    buffers, so when an activation-side tensor is replaced they are dropped (both or neither), and the op builds its
    arguments from the passed pointers."""
    ns = torch.ops.fish_scales_ops
    orig = getattr(ns, opname)
    sch = orig.default._schema.arguments
    names = [a.name for a in sch]
    writes = {a.name for a in sch if a.alias_info is not None and a.alias_info.is_write} | {"out"}
    weights = ("b", "sb", "w13", "sw13", "w2", "sw2", "w_fp8", "w13_fp8", "w2_fp8", "sw", "sw_int32", "sw13_int32",
               "sw2_int32")
    st = {"n": 0, "hit": False, "exc": None, "ev": -1}

    def wrapper(*args, **kw):
        k = st["n"]
        st["n"] += 1
        if k != nth:
            return orig(*args, **kw)
        args = list(args)
        i = names.index(arg)
        t = args[i] if i < len(args) else kw.get(arg)
        if t is None:
            return orig(*args, **kw)
        tm = _at_offset(t, off)
        if i < len(args):
            args[i] = tm
        else:
            kw[arg] = tm
        if arg not in weights:
            for nm in ("problem_shapes", "ptr_arrays"):
                if nm in names:
                    j = names.index(nm)
                    if j < len(args):
                        args[j] = None
                    elif nm in kw:
                        kw[nm] = None
        st["hit"] = True
        box = {}

        def one():
            box["r"] = orig(*args, **kw)
        exc, ev = device_events(one)
        st["exc"], st["ev"] = exc, ev
        if exc is not None:
            raise exc
        if arg in writes:
            t.copy_(tm)
        res = box["r"]

        def back(r):
            return t if isinstance(r, torch.Tensor) and r.numel() and r.data_ptr() == tm.data_ptr() else r
        return type(res)(back(r) for r in res) if isinstance(res, (tuple, list)) else back(res)
    setattr(ns, opname, wrapper)
    try:
        try:
            y = fn()
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001 - the caller judges the type
            y = None
            if st["exc"] is None:
                st["exc"] = e
    finally:
        delattr(ns, opname)
    return st["exc"], st["ev"], st["hit"], y


# What the grouped GEMMs underneath fso.moe.layer refuse in this build's default configuration, per architecture,
# token count and op: the operand, the offsets refused (ValueError) and the offsets that run with the aligned call's
# bytes. From the kernels (the TMA loads of every operand; the stores of each route) and the CL2 probe
# (runs/CL2/probe), written here independently of the op bodies' checks.
LAYER_OPS = {
    9: [(16, "linear_fp8_grouped_contiguous_swapab_swiglu", (("a_fp8", (1, 8), ()), ("w_fp8", (1, 8), ()),
                                                             ("sa", (4, 8), ()), ("sw", (), (4, 8)))),
        (256, "linear_fp8_grouped_contiguous_swiglu", (("a_fp8", (1, 8), ()), ("w_fp8", (1, 8), ()),
                                                       ("sa", (4, 8), ()), ("sw", (), (4, 8)))),
        (16, "linear_fp8_grouped_contiguous_swapab", (("w_fp8", (1, 8), ()), ("out", (2, 8), ()))),
        (256, "linear_fp8_grouped_contiguous_swapab", (("w_fp8", (1, 8), ()), ("out", (2, 8), ()))),
        (16, "quantize_1x128_sorted_gather_sm90", (("x_q", (1, 2), (4, 8)), ("sa", (), (4, 8)))),
        (256, "quantize_1x128_sorted_gather_sm90", (("x_q", (1, 2, 4), (8,)), ("sa", (), (4, 8))))],
    # sm_100/103, the compact layout: at M = 16 and 256 both GEMMs take the compact slot route and run as the fused
    # FC1 + FC2 kernel, whose FC2 half reads dq and sd back with TMA and which writes dn one bf16 at a time; at
    # M = 2048 they take the pointer-array route, which writes d and dq in 16-byte vectors and sd one byte at a time.
    10: [(m, "linear_mxfp8_grouped_compact_fc1_fc2", (("a", (1, 8), ()), ("sa", (4, 8), ()), ("w13", (1, 8), ()),
                                                     ("sw13", (4, 8), ()), ("w2", (1, 8), ()), ("sw2", (4, 8), ()),
                                                     ("dq", (1, 8), ()), ("sd", (4, 8), ()), ("dn", (), (2, 8))))
         for m in (16, 256)]
        + [(2048, "linear_mxfp8_grouped_compact_swiglu", (("a", (1, 8), ()), ("b", (1, 8), ()), ("sa", (4, 8), ()),
                                                           ("sb", (4, 8), ()), ("dq", (1, 8), ()), ("sd", (), (4, 8)))),
           (2048, "linear_mxfp8_grouped_compact", (("a", (1, 8), ()), ("b", (1, 8), ()), ("sa", (4, 8), ()),
                                                    ("sb", (4, 8), ()), ("d", (2, 8), ())))]
        + [(m, "quantize_1x32_grouped_gather_out", (("x_q", (1, 2), (4, 8)), ("packed", (), (4, 8))))
           for m in (16, 256)],
    # sm_120/121, the compact layout: the fused-SwiGLU FC1 and the plain FC2, each with its TMA store descriptor built
    # on its output (dq, d); sd is written one byte at a time.
    12: [(m, "linear_mxfp8_grouped_compact_swiglu", (("a", (1, 8), ()), ("b", (1, 8), ()), ("sa", (4, 8), ()),
                                                     ("sb", (4, 8), ()), ("dq", (1, 8), ()), ("sd", (), (4, 8))))
         for m in (16, 256)]
        + [(m, "linear_mxfp8_grouped_compact", (("a", (1, 8), ()), ("b", (1, 8), ()), ("sa", (4, 8), ()),
                                                 ("sb", (4, 8), ()), ("d", (2, 8), ()))) for m in (16, 256)]
        + [(m, "quantize_1x32_grouped_gather_out", (("x_q", (1, 2), (4, 8)), ("packed", (), (4, 8))))
           for m in (16, 256)],
}


def case_layer_ops(major: int) -> None:
    """11. (CL2) The raw grouped GEMMs and out-style glue ops underneath fso.moe.layer, fed by the layer's own routing:
    every operand their kernels read with TMA (the activation, its scale, the weights and their int32 scales) at an
    offset that is not a multiple of 16 bytes, and every caller-owned output at an offset its kernel's stores cannot
    take, raises ValueError naming it, with no device event in that op; what the kernels can read or write runs and
    the layer gives the aligned call's bytes (on sm_90 the fp32 weight scales at any 4-byte offset, the sorted
    gather's x_q at a multiple of its store width)."""
    ops = LAYER_OPS.get(major, [])
    L = LayerSetup(major)
    for M, opname, args in ops:
        hidden, ids, wts, _b = L.inputs(M, 501 + M)
        ref = _ref(lambda: fso.moe.layer(hidden, L.ex, ids, wts))
        v = ""
        for arg, refused, accepted in args:
            for off in refused + accepted:
                exc, ev, hit, y = _with_op_arg_at_offset(opname, arg, off,
                                                         lambda: fso.moe.layer(hidden, L.ex, ids, wts))
                if not hit:
                    _check(False, f"layer M={M}: {opname} was not called with {arg}")
                    v += "!"
                    continue
                if off in refused:
                    ok = isinstance(exc, ValueError) and arg in str(exc) and ev == 0
                    _check(ok, f"layer M={M} {opname} {arg} at offset {off}: expected ValueError naming it and no "
                               f"device event, got {type(exc).__name__ if exc else 'no exception'} "
                               f"({str(exc)[:160]}), {ev} events")
                    counts["refused"] += 1
                    v += "R" if ok else "!"
                else:
                    ok = exc is None and same_bytes([y], ref)
                    _check(ok, f"layer M={M} {opname} {arg} at offset {off}: expected the aligned call's bytes, got "
                               f"{type(exc).__name__ if exc else 'other bytes'} ({str(exc)[:160] if exc else ''})")
                    counts["accepted"] += 1
                    v += "a" if ok else "!"
        again = _ref(lambda: fso.moe.layer(hidden, L.ex, ids, wts))
        okc = same_bytes(again, ref)
        _check(okc, f"layer M={M} {opname}: the aligned call after the refusals differs")
        print(f"  layer M={M:4d} {opname:44s} {' '.join(f'{a}{r}{c}' for a, r, c in args)}: {v}; aligned again "
              f"{'OK' if okc else 'FAIL'}")
    torch.cuda.empty_cache()


def case_grouped_direct(major: int) -> None:
    """11b. (CL2) The grouped GEMMs called directly, on a masked or sorted routing of 16 tokens (top-2 of 8 experts,
    H 2048, I 512): every operand their kernels read with TMA, at an offset that is not a multiple of 16 bytes, raises
    ValueError naming it with no device event, and the aligned call after the refusals gives its bytes. sm_90: the
    masked, contiguous and swap-AB GEMMs (a_fp8, sa, w_fp8 refused; the fp32 sw at 4 and 8 runs with the aligned
    bytes; a caller-owned out= that is not 16-byte aligned refused, the GEMM stores it with TMA). sm_100/103: the masked
    GEMM on the pointer-array route and on the slot route; sm_120/121: the masked GEMM (all four operands refused)."""
    M, H, I = 16, 2048, GLUE_I
    g = torch.Generator(device="cuda").manual_seed(911 + M)
    ids = torch.rand(M, GLUE_E, device="cuda", generator=g).topk(GLUE_TOPK, dim=1).indices.int().contiguous()
    x = (torch.randn(M, H, device="cuda", generator=g) * 0.1).bfloat16()
    w13 = (torch.randn(GLUE_E, 2 * I, H, device="cuda", generator=g) / H ** 0.5).bfloat16()
    pairs = M * GLUE_TOPK
    exp_m = (pairs + GLUE_E - 1) // GLUE_E
    calls = []   # (name, fn(a, sa, w, sw, out), (a, sa, w, sw), {arg: (refused, accepted)}, out0)
    if major == 9:
        wf = w13.float().view(GLUE_E, 2 * I // 128, 128, H // 128, 128)
        amax = wf.abs().amax(dim=(2, 4))
        sc = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
        wq0 = (wf / sc[:, :, None, :, None]).view(GLUE_E, 2 * I, H).to(torch.float8_e4m3fn)
        sw0 = sc.contiguous()
        m_cap = 64
        masked_m, _rm, slot_of_flat, _s, _ps, _w = T.moe_build_routing(ids, GLUE_E, m_cap, False, [], None)
        slots = slot_of_flat[slot_of_flat >= 0].long()
        a0, sa0 = T.quantize_1x128_grouped_gather_sm90(x, slot_of_flat, GLUE_TOPK, GLUE_E, m_cap)
        calls.append(("linear_fp8_grouped_masked",
                      lambda a, sa, w, sw, out: [T.linear_fp8_grouped_masked(a, w, sa, sw, masked_m, exp_m)
                                                 .reshape(-1, 2 * I)[slots]],
                      (a0, sa0, wq0, sw0), None))
        for name, block in (("linear_fp8_grouped_contiguous", 64), ("linear_fp8_grouped_contiguous_swapab", 16)):
            se, fts, _np = T.moe_build_sorted(ids, GLUE_E, block)
            p_max = int(se.shape[0])
            rows = fts[fts >= 0].long()
            a1, sa1 = T.quantize_1x128_sorted_gather_sm90(x, fts, p_max, GLUE_TOPK, None, None)
            op = getattr(T, name)
            calls.append((name, lambda a, sa, w, sw, out, op=op, se=se, block=block, rows=rows:
                          [op(a, w, sa, sw, se, block, exp_m, out)[rows]],
                          (a1, sa1, wq0, sw0), torch.zeros(p_max, 2 * I, dtype=torch.bfloat16, device="cuda")))
        refused = {"a": (1, 8), "sa": (4, 8), "w": (1, 8)}
        accepted = {"sw": (4, 8)}
    else:
        from fish_scales_ops.gemm.mxfp8 import quantize_moe_weights_1x32_fp8
        wq0, sw0 = quantize_moe_weights_1x32_fp8(w13)
        m_cap = max(4, (M + 3) // 4 * 4)
        for route in (("ptr", "slot") if major == 10 else ("one",)):
            slot_route = route == "slot"
            masked_m, _rm, slot_of_flat, s2e, _ps, _w = T.moe_build_routing(ids, GLUE_E, m_cap, slot_route, [], None)
            slots = slot_of_flat[slot_of_flat >= 0].long()
            a0, sa0 = T.quantize_1x32_grouped_gather(x, slot_of_flat, GLUE_TOPK, GLUE_E, m_cap, True)
            mag = min(pairs, GLUE_E) if slot_route else 0
            calls.append((f"linear_mxfp8_grouped_masked ({route} route)" if major == 10 else
                          "linear_mxfp8_grouped_masked",
                          lambda a, sa, w, sw, out, mm=masked_m, mag=mag, s2e=s2e if slot_route else None, sl=slots:
                          [T.linear_mxfp8_grouped_masked(a, w, sa, sw, mm, exp_m, mag, s2e, None)
                           .reshape(-1, 2 * I)[sl]],
                          (a0, sa0, wq0, sw0), None))
        refused = {"a": (1, 8), "sa": (4, 8), "w": (1, 8), "sw": (4, 8)}
        accepted = {}
    names = {"a": ("a_fp8",), "sa": ("sa",), "w": ("w_fp8",), "sw": ("sw",)}
    for name, fn, ops0, out0 in calls:
        a0, sa0, w0, s0 = ops0
        ref = _ref(lambda: fn(a0, sa0, w0, s0, out0))
        v = ""
        for i, arg in enumerate(("a", "sa", "w", "sw")):
            for off in refused.get(arg, ()) + accepted.get(arg, ()):
                args = list(ops0)
                args[i] = _at_offset(ops0[i], off)
                if off in refused.get(arg, ()):
                    v += _refused(lambda: fn(*args, out0), names[arg], f"{name} {arg} at offset {off}")
                else:
                    v += _same(lambda: fn(*args, out0), ref, f"{name} fp32 {arg} at offset {off}")
        if out0 is not None:
            for off in (2, 8):
                o = _at_offset(out0, off)
                v += _refused(lambda: fn(a0, sa0, w0, s0, o), ("out",), f"{name} out at offset {off}")
        v += _same(lambda: fn(a0.clone(), sa0.clone(), w0.clone(), s0.clone(), out0), ref,
                   f"{name} aligned call after the refusals")
        outs = ", out at (2, 8) refused" if out0 is not None else ""
        print(f"  grouped {name:46s} refused {refused}, run {accepted}{outs}, aligned again: {v}")
    torch.cuda.empty_cache()


def case_out_buffers(major: int) -> None:
    """12. (CL2) The caller-owned outputs of the out-style glue ops and moe_combine(out=...): every offset their
    kernel's stores cannot take raises ValueError naming the buffer, with no device event; every other offset runs
    with the aligned call's bytes. The fp8 codes of quantize_1x32_grouped_gather_out and
    silu_chunk_mul_quantize_1x32_grouped_out are stored in 4-byte words, those of quantize_1x128_sorted_gather_sm90
    in 4-byte words below 32 tokens and 8-byte words from 32 tokens on, moe_combine's out in 16-byte vectors; the
    int32 / fp32 scale outputs are written one element at a time and take any (element-aligned) offset."""
    for M in (16, 64, 256):
        B = GlueInputs(M, 2048)
        cells = [("q1x32_gather_out", "x_q", (1, 2), (4, 8, 12)), ("q1x32_gather_out", "packed", (), (4, 8)),
                 ("silu1x32_out", "dq", (1, 2), (4, 8, 12)), ("silu1x32_out", "sd", (), (4, 8)),
                 ("sorted", "x_q", (1, 2) if M < 32 else (1, 2, 4, 12), (4, 8) if M < 32 else (8,)),
                 ("sorted", "sa", (), (4, 8)),
                 ("combine", "out", (2, 4, 6, 8, 10, 12, 14), (16,))]
        if major in (10, 12):
            cells.insert(4, ("silu1x32_out_pw", "dq", (1, 2), (4, 8, 12)))
        for op, arg, refused, accepted in cells:
            def run(buf, op=op, arg=arg):
                """The op with the output `arg` given as `buf` (zeroed by the caller); its deterministic result."""
                if op == "q1x32_gather_out":
                    xq = buf if arg == "x_q" else B.xq_out
                    pk = buf if arg == "packed" else B.sc_out
                    T.quantize_1x32_grouped_gather_out(B.x, B.slot_of_flat, GLUE_TOPK, GLUE_E, B.m_cap, True, xq, pk)
                    return [xq.clone(), pk.clone()]
                if op.startswith("silu1x32_out"):
                    dq = buf if arg == "dq" else B.dq_out
                    sd = buf if arg == "sd" else B.sd_out
                    T.silu_chunk_mul_quantize_1x32_grouped_out(B.gu, B.slot_of_flat, True, op.endswith("_pw"), dq, sd)
                    return [dq.clone(), sd.clone()]
                if op == "sorted":
                    xq, sa = T.quantize_1x128_sorted_gather_sm90(B.x, B.fts, B.p_max, GLUE_TOPK,
                                                                  buf if arg == "x_q" else None,
                                                                  buf if arg == "sa" else None)
                    kps = torch.arange(B.K // 128, device="cuda")
                    return [xq[B.rows], sa.reshape(-1)[kps[None, :] * sa.shape[-1] + B.rows[:, None]]]
                return [T.moe_combine(B.dn, B.slot_of_flat, B.wts, B.bias, None, buf)]
            if op == "q1x32_gather_out":
                base = B.xq_out if arg == "x_q" else B.sc_out
            elif op.startswith("silu1x32_out"):
                base = B.dq_out if arg == "dq" else B.sd_out
            elif op == "sorted":
                base = (torch.zeros(B.p_max, B.K, dtype=torch.float8_e4m3fn, device="cuda") if arg == "x_q"
                        else torch.zeros(B.K // 128, (B.p_max + 3) // 4 * 4, dtype=torch.float32, device="cuda"))
            else:
                base = torch.zeros(M, B.K, dtype=torch.bfloat16, device="cuda")
            B.xq_out.zero_()
            B.sc_out.zero_()
            B.dq_out.zero_()
            B.sd_out.zero_()
            base.zero_()
            ref = _ref(lambda: run(_at_offset(base, 0)))
            v = ""
            for off in refused + accepted:
                B.xq_out.zero_()
                B.sc_out.zero_()
                B.dq_out.zero_()
                B.sd_out.zero_()
                t = _at_offset(base, off)   # a zeroed buffer at the offset, built before the profiled call
                if off in refused:
                    v += _refused(lambda: run(t), (arg,), f"{op} M={M} {arg} at offset {off}")
                else:
                    v += _same(lambda: run(t), ref, f"{op} M={M} {arg} at offset {off}")
            print(f"  out buffer M={M:4d} {op:18s} {arg:6s} refused {refused}, run {accepted}: {v}")
    torch.cuda.empty_cache()


def case_entries(major: int) -> None:
    """13. (CL2) The public entries never copy a weight per call: fso.dense.linear / linear_quantized with a handle
    whose weight (or, on kind "mxfp8", int32 scale) is not 16-byte aligned, and fso.moe.layer with experts whose
    weights (or, on kind "mxfp8", int32 scale handles) are not, raise ValueError naming the tensor before anything is
    launched. The fp32 block scales of kind "bsfp8" (sm_90) take any 4-byte offset and give the aligned call's
    bytes. prepare_weight and prepare_experts return tensors that start on a 16-byte boundary even when the caller's
    checkpoint tensors are views at other offsets, and the layer / linear on them give the aligned prepare's bytes."""
    M, K, Nw = 16, 4096, 2048
    g = torch.Generator(device="cuda").manual_seed(707)
    w = (torch.randn(Nw, K, device="cuda", generator=g) / K ** 0.5).bfloat16()
    x = (torch.randn(M, K, device="cuda", generator=g) * 0.1).bfloat16()
    dw = D.prepare_weight(w, format="bsfp8" if major == 9 else "mxfp8")
    import dataclasses
    ref = _ref(lambda: D.linear(x, dw))
    v = ""
    for off in (1, 8):
        h = dataclasses.replace(dw, weight=_at_offset(dw.weight, off))
        v += _refused(lambda: D.linear(x, h), ("weight",), f"fso.dense.linear weight at offset {off}")
    for off in (4, 8):
        h = dataclasses.replace(dw, scale=_at_offset(dw.scale, off))
        if major == 9:
            v += _same(lambda: D.linear(x, h), ref, f"fso.dense.linear fp32 scale at offset {off}")
        else:
            v += _refused(lambda: D.linear(x, h), ("scale",), f"fso.dense.linear scale at offset {off}")
    if major != 9:
        for off in (1, 8):
            h = dataclasses.replace(dw, weight=_at_offset(dw.weight, off))
            v += _refused(lambda: D.linear_quantized(x, h), ("weight",), f"fso.dense.linear_quantized weight at {off}")
    lq = "" if major == 9 else ", linear_quantized weight at (1, 8)"
    print(f"  entry fso.dense kind {dw.kind}: weight at (1, 8), scale at (4, 8){lq}: {v}")
    L = LayerSetup(major)
    ex = L.ex
    hidden, ids, wts, _b = L.inputs(16, 808)
    ref = _ref(lambda: fso.moe.layer(hidden, ex, ids, wts))
    v = ""
    for name, offs in (("w13", (1, 8)), ("w2", (1, 8)), ("sw13", (4, 8)), ("sw2", (4, 8))):
        for off in offs:
            e = dataclasses.replace(ex, **{name: _at_offset(getattr(ex, name), off)})
            if major == 9 and name.startswith("sw"):
                v += _same(lambda: fso.moe.layer(hidden, e, ids, wts), ref, f"fso.moe.layer fp32 {name} at {off}")
            else:
                v += _refused(lambda: fso.moe.layer(hidden, e, ids, wts), (name,), f"fso.moe.layer {name} at {off}")
    print(f"  entry fso.moe.layer kind {ex.kind}: w13, w2 at (1, 8), sw13, sw2 at (4, 8): {v}")
    # prepare_* on checkpoint tensors that are views at a misaligned offset: aligned handles, the aligned bytes.
    v = ""
    if major == 9:
        wq, s = T.quantize_128x128(w, False)
        h0 = D.prepare_weight(wq, format="bsfp8", scale=s)
        ref = _ref(lambda: D.linear(x, h0))
        for off in (1, 8):
            h = D.prepare_weight(_at_offset(wq, off), format="bsfp8", scale=_at_offset(s, 4))
            ok = h.weight.data_ptr() % 16 == 0 and h.scale.data_ptr() % 16 == 0
            _check(ok, f"prepare_weight (bsfp8) on views at offsets {off} / 4: the handle is not 16-byte aligned")
            v += (_same(lambda: D.linear(x, h), ref, f"prepare_weight (bsfp8) view at {off}") if ok else "!")
        q13, s13 = ex.w13, ex.sw13
        e0 = fso.moe.prepare_experts(q13, ex.w2, format="bsfp8", sw13=s13, sw2=ex.sw2)
        ref = _ref(lambda: fso.moe.layer(hidden, e0, ids, wts))
        e1 = fso.moe.prepare_experts(_at_offset(q13, 8), _at_offset(ex.w2, 1), format="bsfp8",
                                     sw13=_at_offset(s13, 4), sw2=_at_offset(ex.sw2, 4))
        ok = all(t.data_ptr() % 16 == 0 for t in (e1.w13, e1.sw13, e1.w2, e1.sw2))
        _check(ok, "prepare_experts (bsfp8) on views at offsets 8 / 1 / 4 / 4: the handle is not 16-byte aligned")
        v += (_same(lambda: fso.moe.layer(hidden, e1, ids, wts), ref, "prepare_experts (bsfp8) on views") if ok
              else "!")
    else:
        wf = w.float().view(Nw, K // 32, 32)
        e = torch.ceil(torch.log2(wf.abs().amax(dim=-1).clamp_min(1e-30) / 448.0)).clamp(-127, 127)
        wq = (wf / torch.exp2(e)[..., None]).view(Nw, K).to(torch.float8_e4m3fn)
        sb = (e + 127).to(torch.uint8).contiguous()
        h0 = D.prepare_weight(wq, format="mxfp8", scale=sb)
        ref = _ref(lambda: D.linear(x, h0))
        for off in (1, 8):
            h = D.prepare_weight(_at_offset(wq, off), format="mxfp8", scale=_at_offset(sb, 1))
            ok = h.weight.data_ptr() % 16 == 0 and h.scale.data_ptr() % 16 == 0
            _check(ok, f"prepare_weight (mxfp8 checkpoint) on views at offsets {off} / 1: the handle is not aligned")
            v += (_same(lambda: D.linear(x, h), ref, f"prepare_weight (mxfp8 checkpoint) view at {off}") if ok else "!")
    print(f"  entry prepare_* on misaligned checkpoint views: the handles are aligned and give the aligned bytes: {v}")
    torch.cuda.empty_cache()


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: test_misaligned_x needs a CUDA device")
        return 0
    major, minor = torch.cuda.get_device_capability()
    if major not in (9, 10, 12):
        print(f"SKIP: the dense quantizers exist on sm_90, sm_100/sm_103 and sm_120/sm_121 (device sm_{major}{minor})")
        return 0
    print(f"test_misaligned_x on sm_{major}{minor} ({torch.cuda.get_device_name()})")
    case_control(major)
    case_ops(major)
    case_pitch(major)
    case_noncontig(major)
    case_fp8_operands(major)
    case_dense_compile_capture(major)
    case_dense_prequant(major)
    case_moe_glue(major)
    case_moe_layer(major)
    case_compat_prequant(major)
    case_dense_weights(major)
    case_compat_weights(major)
    case_layer_ops(major)
    case_grouped_direct(major)
    case_out_buffers(major)
    case_entries(major)
    print(f"  {counts['refused']} refused calls, {counts['accepted']} accepted misaligned calls")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures[:40]:
            print("  " + f)
        return 1
    print("test_misaligned_x: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
