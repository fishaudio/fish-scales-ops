#!/usr/bin/env python3
"""The dense surface fso.dense (prepare_weight / linear / supported / describe)
against the explicit composition from fso.compat.

fso.dense.linear is one torch custom op, torch.ops.fish_scales_ops.dense_linear,
whose body picks the architecture's route, so the property worth asserting is
that going through it changes nothing: on every architecture it has to produce
exactly the tensor the explicit fso.compat composition produces for the same
weight and input. The cases, each run where the hardware exists and skipped with
a message elsewhere:

  a.  sm_90, format "bsfp8": a block-FP8 checkpoint (quantize_128x128_fp8 with the
      sm_90 default, fp32 scales amax / 448) is kept as it is, and linear() is
      torch.equal to quantize_1x128_fp8 + linear_fp8 and to linear_qx; a bf16
      weight prepared with "bsfp8" holds the bytes quantize_128x128_fp8 produces;
      a checkpoint quantized in torch (scale = amax / 448) is served unchanged.
      a2. an N that is not a multiple of 128 (N = 320), which sm_90 accepts.
  b.  sm_100/103 and sm_120/121, format "bsfp8": a block-FP8 checkpoint (scale =
      amax / 448, value = fp8 * scale) is dequantized and requantized to MXFP8;
      the prepared weight and scale are torch.equal to doing the same steps by
      hand (fp8 * block scale in fp32, rounded once to bf16, quantize_1x32_fp8),
      and linear() is torch.equal to quantize_1x32_fp8 + linear_mxfp8 on them.
      b2. the conversion forced into ragged chunks of 384 rows gives the same
      bytes and, on sm_120/121, the same K-major strides; on sm_100/103 this is
      what checks that the atom layout keeps each 128-row block contiguous.
  c.  sm_100/103 and sm_120/121, format "mxfp8": from bf16 (the bytes of
      quantize_1x32_fp8), and pre-quantized with the scale in both stride forms:
      the K-major tensor quantize_1x32_fp8 returns and the .contiguous() copy a
      safetensors round trip produces (plus a literal safetensors round trip when
      the package is installed). All torch.equal to quantize_1x32_fp8 +
      linear_mxfp8, and on sm_120/121 also to the raw-op sequence quantize_1x32 +
      repack_mxfp8_scales + linear_mxfp8_raw on the restored scale. As a control,
      the row-major scale handed to linear_mxfp8 directly is reported, which is
      the silent error the restore prevents.
  Shapes for a, b and c: the Qwen3-4B projections (wqkv 6144x2560, wo 2560x4096,
  gate_up 19456x2560, down 2560x9728) at M in {1000, 64, 7, 1}; the narrow-N
  shapes and the largest M run first so that the first sm_120 Stream-K call
  sizes the Stream-K scratch for the largest one.
  d.  N-d input ([B, S, K], [K], a non-contiguous view, empty M) gives the
      reshaped result of the 2-D call.
  e.  torch.compile(fullgraph=True) of a function that calls linear(): a
      recording backend sees exactly one dense_linear node and no other
      fish_scales_ops node, and compiles twice for three M (static, then dynamic,
      then reused); under inductor the compiled call is torch.equal to eager for
      M = 64, 1000 and 7.
  f.  CUDA-graph capture after warm-up, replay into a NaN-filled output,
      torch.equal to eager; an eager call under torch.cuda.set_sync_debug_mode
      ("error") shows that nothing inside synchronizes the device.
  g.  the refusals: sm_90 "mxfp8", malformed scales, K % 128 != 0, N % 128 != 0
      on the MXFP8 architectures, unknown and unserved formats, a weight without
      its scale and a bf16 weight with one, linear() and raw-op argument errors;
      every message names the architecture.
  h.  supported() over every architecture form and describe() (also without a
      CUDA device).
"""
import dataclasses
import os
import sys
import tempfile
import warnings

import torch

import fish_scales_ops as fso

C = fso.compat
D = fso.dense

SHAPES = (("down", 2560, 9728), ("wo", 2560, 4096), ("wqkv", 6144, 2560), ("gate_up", 19456, 2560))
MS = (1000, 64, 7, 1)

failures: list[str] = []


def _check(ok: bool, msg: str) -> bool:
    if not ok:
        failures.append(msg)
    return ok


def _ok(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def _arch() -> tuple[int, int]:
    return torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)


def _arch_label() -> str:
    major, minor = _arch()
    return f"sm_{major}{minor}" if major else "no CUDA device"


def _maxrel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def make_x(M: int, K: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return (torch.randn(M, K, device="cuda") * 0.1).bfloat16()


def make_w(N: int, K: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return (torch.randn(N, K, device="cuda") / K ** 0.5).bfloat16()


def block_quantize_fp8(w: torch.Tensor):
    """bf16 [N, K] -> (float8_e4m3fn [N, K], fp32 [N/128, K/128]): one scale per
    128x128 block, scale = amax / 448, stored value * scale = weight (the DeepSeek
    block-FP8 checkpoint convention)."""
    N, K = w.shape
    wf = w.float().view(N // 128, 128, K // 128, 128)
    amax = wf.abs().amax(dim=(1, 3))
    sc = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    q = (wf / sc[:, None, :, None]).view(N, K).to(torch.float8_e4m3fn)
    return q, sc


def block_dequantize_bf16(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """The by-hand form of the load-time rule: fp8 * block scale in fp32, rounded once to bf16."""
    N, K = q.shape
    return (q.float().view(N // 128, 128, K // 128, 128) * s.view(N // 128, 1, K // 128, 1)).view(N, K).bfloat16()


def same_scale(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal values and equal strides, so the bytes the GEMM reads are equal too."""
    return a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and a.stride() == b.stride() and torch.equal(a, b)


def _expect_raise(label, exc_types, fn, name_arch=True):
    arch = _arch_label()
    try:
        fn()
    except exc_types as exc:
        named = (arch in str(exc)) or not name_arch
        _check(named, f"{label}: {type(exc).__name__} does not name {arch}: {exc}")
        return f"{type(exc).__name__}{'' if named else ' (arch NOT named)'}"
    except Exception as exc:  # noqa: BLE001
        _check(False, f"{label}: raised {type(exc).__name__} instead of {exc_types}: {exc}")
        return f"wrong exception {type(exc).__name__}: {exc}"
    _check(False, f"{label}: did not raise")
    return "did not raise"


# --- (a) sm_90 --------------------------------------------------------------------

def case_a() -> None:
    for name, N, K in SHAPES:
        w = make_w(N, K, seed=N * 17 + K)
        wq, sw = C.quantize_128x128_fp8(w)
        dw = D.prepare_weight(wq, format="bsfp8", scale=sw)
        kept = dw.kind == "bsfp8" and dw.weight.data_ptr() == wq.data_ptr() and dw.scale.data_ptr() == sw.data_ptr()
        _check(kept, f"a {name}: the block-FP8 checkpoint was not kept as it is ({dw!r})")
        dwb = D.prepare_weight(w, format="bsfp8")
        bf_ok = dwb.kind == "bsfp8" and torch.equal(dwb.weight, wq) and same_scale(dwb.scale, sw)
        _check(bf_ok, f"a {name}: a bf16 weight prepared with 'bsfp8' differs from quantize_128x128_fp8")
        tq, ts = block_quantize_fp8(w)
        dwt = D.prepare_weight(tq, format="bsfp8", scale=ts)
        for M in MS:
            x = make_x(M, K, seed=M * 1009 + N * 17 + K)
            y = D.linear(x, dw)
            xq, sx = C.quantize_1x128_fp8(x)
            ref = C.linear_fp8(xq, wq, sx, sw)
            ref_qx = C.linear_qx(x, wq, sw)
            yb = D.linear(x, dwb)
            yt = D.linear(x, dwt)
            ref_t = C.linear_qx(x, tq, ts)
            ok = (tuple(y.shape) == (M, N) and y.dtype == torch.bfloat16 and bool(torch.isfinite(y).all())
                  and torch.equal(y, ref) and torch.equal(y, ref_qx) and torch.equal(yb, ref)
                  and torch.equal(yt, ref_t))
            _check(ok, f"a {name} M={M}: linear() != quantize_1x128_fp8 + linear_fp8 / linear_qx")
            print(f"  a  {name:8s} N={N:5d} K={K:5d} M={M:5d}: bsfp8 checkpoint kept, == quantize_1x128_fp8 + "
                  f"linear_fp8 == linear_qx; bf16 weight == quantize_128x128_fp8; torch-quantized checkpoint "
                  f"== linear_qx  {_ok(ok and kept and bf_ok)}")
        del w, wq, sw, dw, dwb, tq, ts, dwt
    torch.cuda.empty_cache()


def case_a2() -> None:
    N, K = 320, 2560
    w = make_w(N, K, seed=4242)
    wq, sw = C.quantize_128x128_fp8(w)
    dw = D.prepare_weight(wq, format="bsfp8", scale=sw)
    for M in (64, 7, 1):
        x = make_x(M, K, seed=M + 77)
        y = D.linear(x, dw)
        xq, sx = C.quantize_1x128_fp8(x)
        ref = C.linear_fp8(xq, wq, sx, sw)
        ok = tuple(y.shape) == (M, N) and torch.equal(y, ref)
        _check(ok, f"a2 N={N} M={M}: linear() != quantize_1x128_fp8 + linear_fp8 (sm_90 with N % 128 != 0)")
        print(f"  a2 N={N} (not a multiple of 128) K={K} M={M:3d}: == quantize_1x128_fp8 + linear_fp8  {_ok(ok)}")


# --- (b) Blackwell, bsfp8 ---------------------------------------------------------

def case_b(major: int) -> None:
    for name, N, K in SHAPES:
        w = make_w(N, K, seed=N * 17 + K + 1)
        q, s = block_quantize_fp8(w)
        dw = D.prepare_weight(q, format="bsfp8", scale=s)
        wq2, sw2 = C.quantize_1x32_fp8(block_dequantize_bf16(q, s))
        prep_ok = (dw.kind == "mxfp8" and dw.format == "bsfp8" and torch.equal(dw.weight, wq2)
                   and same_scale(dw.scale, sw2))
        _check(prep_ok, f"b {name}: the requantized checkpoint differs from the by-hand dequantize + "
                        f"quantize_1x32_fp8")
        # The scale may arrive in any layout with the right values; only the values count.
        dwc = D.prepare_weight(q, format="bsfp8", scale=s.t().contiguous().t())
        col_ok = torch.equal(dwc.weight, wq2) and same_scale(dwc.scale, sw2)
        _check(col_ok, f"b {name}: a column-major copy of the block scales changed the result")
        dwb = D.prepare_weight(w, format="bsfp8")
        wq1, sw1 = C.quantize_1x32_fp8(w)
        bf_ok = dwb.kind == "mxfp8" and torch.equal(dwb.weight, wq1) and same_scale(dwb.scale, sw1)
        _check(bf_ok, f"b {name}: a bf16 weight prepared with 'bsfp8' differs from quantize_1x32_fp8")
        for M in MS:
            x = make_x(M, K, seed=M * 1009 + N * 17 + K)
            y = D.linear(x, dw)
            xq, sx = C.quantize_1x32_fp8(x)
            ref = C.linear_mxfp8(xq, wq2, sx, sw2)
            ref1 = C.linear_mxfp8(xq, wq1, sx, sw1)
            yb = D.linear(x, dwb)
            ok = (tuple(y.shape) == (M, N) and y.dtype == torch.bfloat16 and bool(torch.isfinite(y).all())
                  and torch.equal(y, ref) and torch.equal(yb, ref1))
            _check(ok, f"b {name} M={M}: linear() != quantize_1x32_fp8 + linear_mxfp8 on the requantized weight")
            print(f"  b  {name:8s} N={N:5d} K={K:5d} M={M:5d}: bsfp8 checkpoint requantized == by hand, "
                  f"linear() == quantize_1x32_fp8 + linear_mxfp8; bf16 'bsfp8' == quantize_1x32_fp8 path  "
                  f"{_ok(ok and prep_ok and col_ok and bf_ok)}")
        del w, q, s, dw, dwc, dwb, wq2, sw2, wq1, sw1
    torch.cuda.empty_cache()


def case_b2(major: int) -> None:
    saved = D._CONVERT_BUDGET_BYTES
    try:
        for name, N, K in SHAPES:
            w = make_w(N, K, seed=N * 17 + K + 2)
            q, s = block_quantize_fp8(w)
            one = D.prepare_weight(q, format="bsfp8", scale=s)
            rows = 384  # three 128-row blocks; N is not a multiple of it for every shape here
            D._CONVERT_BUDGET_BYTES = rows * K * 4
            chunked = D.prepare_weight(q, format="bsfp8", scale=s)
            D._CONVERT_BUDGET_BYTES = saved
            n_chunks = -(-N // rows)
            ok = torch.equal(chunked.weight, one.weight) and same_scale(chunked.scale, one.scale)
            if major == 12:
                ok = ok and chunked.scale.stride() == (1, N)
            _check(ok and n_chunks >= 3, f"b2 {name}: the chunked conversion ({n_chunks} chunks) differs from one call")
            print(f"  b2 {name:8s} N={N:5d} K={K:5d}: {n_chunks} chunks of {rows} rows (last "
                  f"{N - (n_chunks - 1) * rows}) == one conversion, weight bytes and scale words "
                  f"{tuple(chunked.scale.shape)} strides {tuple(chunked.scale.stride())}  {_ok(ok)}")
            del w, q, s, one, chunked
    finally:
        D._CONVERT_BUDGET_BYTES = saved
    torch.cuda.empty_cache()


# --- (c) Blackwell, mxfp8 ---------------------------------------------------------

def _safetensors_round_trip(t: torch.Tensor):
    try:
        from safetensors.torch import load_file, save_file
    except ImportError:
        return None
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "scale.safetensors")
        save_file({"s": t.contiguous().cpu()}, path)
        return load_file(path)["s"].to(t.device)


def case_c(major: int) -> None:
    for name, N, K in SHAPES:
        w = make_w(N, K, seed=N * 17 + K + 3)
        wq, sw = C.quantize_1x32_fp8(w)
        dwb = D.prepare_weight(w, format="mxfp8")
        bf_ok = dwb.kind == "mxfp8" and dwb.format == "mxfp8" and torch.equal(dwb.weight, wq) and same_scale(dwb.scale, sw)
        _check(bf_ok, f"c {name}: a bf16 weight prepared with 'mxfp8' differs from quantize_1x32_fp8")
        dwk = D.prepare_weight(wq, format="mxfp8", scale=sw)
        k_ok = torch.equal(dwk.weight, wq) and same_scale(dwk.scale, sw)
        sw_rt = sw.contiguous()
        dwr = D.prepare_weight(wq, format="mxfp8", scale=sw_rt)
        r_ok = torch.equal(dwr.weight, wq) and same_scale(dwr.scale, sw)
        _check(k_ok and r_ok, f"c {name}: a pre-quantized scale was not kept (K-major) or restored (.contiguous())")
        st = _safetensors_round_trip(sw)
        dws = D.prepare_weight(wq, format="mxfp8", scale=st) if st is not None else None
        s_ok = dws is None or same_scale(dws.scale, sw)
        _check(s_ok, f"c {name}: the scale after a safetensors round trip was not restored")
        rt_form = (f"strides {tuple(sw_rt.stride())}" if sw_rt.stride() != sw.stride()
                   else "the same strides (this architecture's scale is 1-D)")
        for M in MS:
            x = make_x(M, K, seed=M * 1009 + N * 17 + K)
            xq, sx = C.quantize_1x32_fp8(x)
            ref = C.linear_mxfp8(xq, wq, sx, sw)
            ys = [D.linear(x, h) for h in (dwb, dwk, dwr) + ((dws,) if dws is not None else ())]
            ok = (tuple(ref.shape) == (M, N) and bool(torch.isfinite(ref).all())
                  and all(torch.equal(y, ref) for y in ys))
            _check(ok, f"c {name} M={M}: linear() != quantize_1x32_fp8 + linear_mxfp8")
            if major == 12:
                # A caller of the raw ops on sm_120/121: the two-step activation quantize, the
                # K-major weight scale restored from its .contiguous() copy, linear_mxfp8_raw.
                ops = torch.ops.fish_scales_ops
                xr, sxr = ops.quantize_1x32(x, True)
                raw = ops.linear_mxfp8_raw(xr, wq, ops.repack_mxfp8_scales(sxr), sw_rt.t().contiguous().t())
                raw_ok = torch.equal(raw, ys[2])
                ok = ok and raw_ok
                _check(raw_ok, f"c {name} M={M}: linear() != quantize_1x32 + repack_mxfp8_scales + "
                               f"linear_mxfp8_raw on sm_120")
            control = ""
            if major == 12 and sw_rt.stride() != sw.stride() and M == MS[0]:
                bad = C.linear_mxfp8(xq, wq, sx, sw_rt)
                control = (f"; control: the .contiguous() scale passed to linear_mxfp8 as it is gives max rel. "
                           f"error {_maxrel(bad, ref):.2f}")
            raw_note = "; == quantize_1x32 + repack_mxfp8_scales + linear_mxfp8_raw" if major == 12 else ""
            print(f"  c  {name:8s} N={N:5d} K={K:5d} M={M:5d}: from bf16, K-major scale, .contiguous() scale "
                  f"({rt_form}){', safetensors round trip' if dws is not None else ''} all == quantize_1x32_fp8 + "
                  f"linear_mxfp8{raw_note}  {_ok(ok and bf_ok and k_ok and r_ok and s_ok)}{control}")
        del w, wq, sw, sw_rt, dwb, dwk, dwr, dws
    torch.cuda.empty_cache()


# --- (d) N-d input ----------------------------------------------------------------

def _prepared(major: int, N: int, K: int, seed: int):
    w = make_w(N, K, seed=seed)
    return D.prepare_weight(w, format="bsfp8" if major == 9 else "mxfp8")


def case_d(major: int) -> None:
    N, K = 2560, 4096
    dw = _prepared(major, N, K, seed=11)
    x3 = make_x(2 * 5, K, seed=12).view(2, 5, K)
    y3 = D.linear(x3, dw)
    flat = D.linear(x3.reshape(10, K), dw)
    ok3 = tuple(y3.shape) == (2, 5, N) and torch.equal(y3, flat.view(2, 5, N))
    _check(ok3, "d: [B, S, K] input is not the reshaped 2-D result")
    x1 = make_x(1, K, seed=13)[0]
    y1 = D.linear(x1, dw)
    ok1 = tuple(y1.shape) == (N,) and torch.equal(y1, D.linear(x1.view(1, K), dw)[0])
    _check(ok1, "d: a 1-D [K] input is not the [N] result of the 2-D call")
    big = make_x(64, 2 * K, seed=14)
    xv = big[:, K:]  # a non-contiguous view
    okv = (not xv.is_contiguous()) and torch.equal(D.linear(xv, dw), D.linear(xv.contiguous(), dw))
    _check(okv, "d: a non-contiguous input changes the result")
    e2 = D.linear(make_x(1, K, seed=15)[:0], dw)
    e3 = D.linear(x3[:, :0], dw)
    oke = (tuple(e2.shape) == (0, N) and tuple(e3.shape) == (2, 0, N) and e2.dtype == torch.bfloat16
           and e3.dtype == torch.bfloat16)
    _check(oke, f"d: an empty input gave {tuple(e2.shape)} / {tuple(e3.shape)}")
    print(f"  d  [2, 5, K] -> [2, 5, N] == 2-D call {_ok(ok3)}; [K] -> [N] {_ok(ok1)}; non-contiguous view "
          f"{_ok(okv)}; empty [0, K] -> [0, N] and [2, 0, K] -> [2, 0, N] {_ok(oke)}")


# --- (e) torch.compile ------------------------------------------------------------

def case_e(major: int) -> None:
    N, K = 2560, 4096
    dw = _prepared(major, N, K, seed=21)

    def f(x):
        return D.linear(x, dw)

    xs = {M: make_x(M, K, seed=22 + M) for M in (64, 1000, 7)}
    want = {M: f(x) for M, x in xs.items()}

    graphs = []

    def record(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    rec = torch.compile(f, backend=record, fullgraph=True)
    rec_ok = all(torch.equal(rec(x), want[M]) for M, x in xs.items())
    targets = [str(n.target) for g in graphs for n in g.graph.nodes if n.op == "call_function"]
    ours = [t for t in targets if "fish_scales_ops" in t]
    one_node = all(sum("fish_scales_ops" in str(n.target) and "dense_linear" in str(n.target)
                       for n in g.graph.nodes if n.op == "call_function") == 1 for g in graphs)
    node_ok = one_node and all("dense_linear" in t for t in ours)
    count_ok = len(graphs) == 2
    _check(rec_ok and node_ok and count_ok,
           f"e: recording backend: {len(graphs)} graphs for three M, fish_scales_ops nodes {ours}, equal {rec_ok}")
    print(f"  e  recording backend, fullgraph=True: {len(graphs)} graphs for M = 64, 1000, 7 (static, dynamic, "
          f"reused), one dense_linear node per graph and no other fish_scales_ops node, == eager  "
          f"{_ok(rec_ok and node_ok and count_ok)}")

    torch._dynamo.reset()
    ind = torch.compile(f, fullgraph=True)
    for M, x in xs.items():
        got = ind(x)
        torch.cuda.synchronize()
        ok = torch.equal(got, want[M])
        _check(ok, f"e: inductor M={M}: compiled != eager")
        print(f"  e  inductor, fullgraph=True, M={M:5d}: compiled == eager  {_ok(ok)}")
    torch._dynamo.reset()


# --- (f) CUDA graph -----------------------------------------------------------------

def _graph_case(label, fn) -> None:
    ref = fn().clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = fn()
    out.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    ok = bool(torch.isfinite(out).all()) and torch.equal(out, ref)
    _check(ok, f"f {label}: replay != eager")
    print(f"  f  {label}: capture after warm-up, replay into a NaN-filled output == eager  {_ok(ok)}")


def case_f(major: int) -> None:
    for name, N, K in (("down", 2560, 9728), ("wqkv", 6144, 2560)):
        dw = _prepared(major, N, K, seed=31 + N)
        for M in MS:
            x = make_x(M, K, seed=32 + M)
            _graph_case(f"{name:5s} N={N:5d} K={K:5d} M={M:5d} kind={dw.kind}", lambda: D.linear(x, dw))
        x3 = make_x(12, K, seed=33).view(3, 4, K)
        _graph_case(f"{name:5s} N={N:5d} K={K:5d} [3, 4, K] kind={dw.kind}", lambda: D.linear(x3, dw))
        x = make_x(64, K, seed=34)
        D.linear(x, dw)
        torch.cuda.synchronize()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # torch labels the mode a prototype
                torch.cuda.set_sync_debug_mode("error")
            D.linear(x, dw)
            sync_ok, why = True, ""
        except RuntimeError as e:
            sync_ok, why = False, str(e).splitlines()[0]
        finally:
            torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()
        _check(sync_ok, f"f {name}: an eager call synchronized the device: {why}")
        print(f"  f  {name:5s} M=64: no device synchronization under set_sync_debug_mode('error')  {_ok(sync_ok)}")


# --- (g) refusals -----------------------------------------------------------------

def case_g(major: int) -> None:
    P = D.prepare_weight
    N, K = 256, 512
    w = make_w(N, K, seed=41)
    q, s = block_quantize_fp8(w)
    results = {
        "unknown format": _expect_raise("unknown format", ValueError, lambda: P(w, format="fp7")),
        "format None": _expect_raise("format None", ValueError, lambda: P(w, format=None)),
        "format 'bf16'": _expect_raise("format bf16", NotImplementedError, lambda: P(w, format="bf16")),
        "format 'nvfp4'": _expect_raise("format nvfp4", NotImplementedError, lambda: P(w, format="nvfp4")),
    }
    if major not in (9, 10, 12):
        results["'bsfp8' on this architecture"] = _expect_raise(
            "bsfp8 here", NotImplementedError, lambda: P(w, format="bsfp8"))
        for k, v in results.items():
            print(f"  g  refuse {k:40s} -> {v}")
        return
    fmt = "bsfp8" if major == 9 else "mxfp8"
    results.update({
        "K % 128 != 0": _expect_raise("K % 128", ValueError, lambda: P(make_w(256, 200, seed=42), format="bsfp8")),
        "3-D weight": _expect_raise("3-D weight", ValueError, lambda: P(w.view(2, 128, K), format="bsfp8")),
        "weight on the CPU": _expect_raise("cpu weight", ValueError, lambda: P(w.cpu(), format="bsfp8")),
        "fp16 weight": _expect_raise("fp16 weight", ValueError, lambda: P(w.half(), format="bsfp8")),
        "fp8 weight without its scale": _expect_raise("fp8 no scale", ValueError, lambda: P(q, format="bsfp8")),
        "bf16 weight with a scale": _expect_raise("bf16 with scale", ValueError, lambda: P(w, format="bsfp8", scale=s)),
        "bsfp8 scale of the wrong shape": _expect_raise(
            "bsfp8 scale shape", ValueError, lambda: P(q, format="bsfp8", scale=s[:, :1].contiguous())),
        "bsfp8 scale of the wrong dtype": _expect_raise(
            "bsfp8 scale dtype", ValueError, lambda: P(q, format="bsfp8", scale=s.to(torch.int32))),
    })
    if major == 9:
        results["format 'mxfp8' on sm_90"] = _expect_raise("mxfp8 on sm_90", NotImplementedError,
                                                           lambda: P(w, format="mxfp8"))
    else:
        wq, sw = C.quantize_1x32_fp8(w)
        kb = K // 128
        results["N % 128 != 0"] = _expect_raise("N % 128", ValueError, lambda: P(make_w(200, 256, seed=43),
                                                                                 format="mxfp8"))
        results["mxfp8 scale of the wrong dtype"] = _expect_raise(
            "mxfp8 scale dtype", ValueError, lambda: P(wq, format="mxfp8", scale=sw.float()))
        results["mxfp8 scale of the wrong shape"] = _expect_raise(
            "mxfp8 scale shape", ValueError,
            lambda: P(wq, format="mxfp8", scale=torch.zeros(N, K // 32, dtype=torch.int32, device="cuda")))
        if major == 12:
            strided = torch.zeros(N, 2 * kb, dtype=torch.int32, device="cuda")[:, ::2]  # strides (2*kb, 2)
            results["mxfp8 scale with foreign strides"] = _expect_raise(
                "mxfp8 scale strides", ValueError, lambda: P(wq, format="mxfp8", scale=strided))
            results["mxfp8 scale transposed"] = _expect_raise(
                "mxfp8 scale transposed", ValueError, lambda: P(wq, format="mxfp8", scale=sw.t()))
        else:
            results["mxfp8 scale, a 2-D view"] = _expect_raise(
                "mxfp8 scale 2-D", ValueError, lambda: P(wq, format="mxfp8", scale=sw.view(N, kb)))
    dw = P(w, format=fmt)
    x = make_x(5, K, seed=44)
    L = D.linear
    other = 120 if major != 12 else 90
    op = torch.ops.fish_scales_ops.dense_linear
    results.update({
        "linear: not a DenseWeight": _expect_raise("linear handle", TypeError, lambda: L(x, object())),
        "linear: a handle from another arch": _expect_raise(
            "linear other arch", ValueError, lambda: L(x, dataclasses.replace(dw, arch=other))),
        "linear: fp16 x": _expect_raise("linear fp16", ValueError, lambda: L(x.half(), dw)),
        "linear: x of the wrong K": _expect_raise("linear K", ValueError, lambda: L(x[:, :256].contiguous(), dw)),
        "raw op: unknown kind": _expect_raise("raw op kind", ValueError, lambda: op(x, dw.weight, dw.scale, "int8")),
        "raw op: kind of another arch": _expect_raise(
            "raw op other kind", NotImplementedError,
            lambda: op(x, dw.weight, dw.scale, "bsfp8" if major != 9 else "mxfp8")),
        "raw op: 3-D x": _expect_raise("raw op 3-D", ValueError, lambda: op(x.view(1, 5, K), dw.weight, dw.scale, dw.kind)),
    })
    if major == 9:
        results["raw op: int32 scale on sm_90"] = _expect_raise(
            "raw op int32 scale", ValueError, lambda: op(x, dw.weight, dw.scale.to(torch.int32), "bsfp8"))
    elif major == 12:
        results["raw op: the row-major scale on sm_120"] = _expect_raise(
            "raw op row-major scale", ValueError, lambda: op(x, dw.weight, dw.scale.contiguous(), "mxfp8"))
    for k, v in results.items():
        print(f"  g  refuse {k:40s} -> {v}")


# --- (h) supported / describe -----------------------------------------------------

def case_h(major: int) -> None:
    expect = {9: {"bsfp8"}, 10: {"bsfp8", "mxfp8"}, 12: {"bsfp8", "mxfp8"}}
    forms = {9: ("sm_90", "sm_90a", 90, 9, (9, 0), "9.0"), 10: ("sm_100", "sm_100a", "sm_103", 103, (10, 3), "10.3"),
             12: ("sm_120", "sm_120a", 121, 12, (12, 1), "12.0")}
    wrong = []
    for fam, names in forms.items():
        for a in names:
            for fmt in ("bsfp8", "mxfp8", "bf16", "int8", "nvfp4"):
                if D.supported(fmt, arch=a) != (fmt in expect[fam]):
                    wrong.append((fmt, a))
    for a in ("sm_80", 89, (8, 6)):
        if D.supported("bsfp8", arch=a) or D.supported("mxfp8", arch=a):
            wrong.append(("any", a))
    if not D.supported(" BSFP8 ", arch="sm_90") or D.supported(None) or D.supported(3, arch=90):
        wrong.append("normalisation")
    here = {f for f in D.FORMATS if D.supported(f)}
    if here != expect.get(major, set()):
        wrong.append(("this device", sorted(here)))
    try:
        D.supported("bsfp8", arch="h200")
        wrong.append("arch 'h200' did not raise")
    except ValueError:
        pass
    _check(not wrong, f"h: supported() matrix wrong at {wrong}")
    print(f"  h  supported(): the matrix over {sum(len(v) for v in forms.values()) + 3} arch spellings, the "
          f"normalisation and the refusal of a non-arch string  {_ok(not wrong)}")
    text = D.describe()
    marks = [ln for ln in text.splitlines() if ln.endswith("<- this device")]
    want_marks = 2 if major in (9, 10, 12) else 0
    ok = (_arch_label() in text and "format 'bsfp8'" in text and "format 'mxfp8'" in text
          and len(marks) == want_marks)
    _check(ok, f"h: describe() does not name this device or mark its {want_marks} rows:\n{text}")
    print(f"  h  describe(): names {_arch_label()}, both formats, {len(marks)} rows marked as this device  {_ok(ok)}")


def main() -> int:
    major, minor = _arch()
    print(f"device: {torch.cuda.get_device_name(0) + ' ' if major else ''}{_arch_label()}; torch {torch.__version__}")
    covered, skipped = [], []
    print("== (h) supported() and describe() ==")
    case_h(major)
    covered.append("h")
    if not major:
        print("SKIP: cases a-g need a CUDA device")
        skipped.append("a-g (no CUDA device)")
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        if major == 9:
            print("== (a) sm_90 'bsfp8' vs quantize_1x128_fp8 + linear_fp8 and linear_qx ==")
            case_a()
            print("== (a2) sm_90 with N % 128 != 0 ==")
            case_a2()
            covered += ["a", "a2"]
        else:
            skipped += ["a, a2 (sm_90)"]
        if major in (10, 12):
            print("== (b) Blackwell 'bsfp8': dequantize + requantize vs the same steps by hand ==")
            case_b(major)
            print("== (b2) the chunked conversion ==")
            case_b2(major)
            print("== (c) Blackwell 'mxfp8': from bf16 and pre-quantized in both scale stride forms ==")
            case_c(major)
            covered += ["b", "b2", "c"]
        else:
            skipped += ["b, b2, c (sm_100/103, sm_120/121)"]
        if major in (9, 10, 12):
            print("== (d) N-d input ==")
            case_d(major)
            print("== (e) torch.compile ==")
            case_e(major)
            print("== (f) CUDA graph ==")
            case_f(major)
            covered += ["d", "e", "f"]
        else:
            skipped += ["d, e, f (no format is served here)"]
        print("== (g) refusals ==")
        case_g(major)
        covered.append("g")
    print(f"\ncovered: {', '.join(covered)}; skipped on {_arch_label()}: {', '.join(skipped) or 'none'}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("fso.dense: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
