#!/usr/bin/env python3
"""MXFP8 weights loaded as a checkpoint stores them (issue #9).

An MXFP8 checkpoint holds float8_e4m3fn weights and one UE8M0 scale byte per 32
elements along K as uint8 ``[.., N, K/32]``, row-major, byte b meaning the scale
2^(b - 127): the same bytes on every architecture. ``fso.moe.prepare_experts`` and
``fso.dense.prepare_weight`` with ``format="mxfp8"`` take them as they are and only move
bytes: the scale bytes are packed into this device's int32 layout and the FC1 rows are
interleaved. Every layout below is computed in this file from the layout formulas
(``csrc/gemm/ops/quant_kernels.cu``, ``sf_word_index``); the library's packer is never
called here, so a round trip compares the library against an independent inverse.

The cases, on sm_100/103 and sm_120/121 unless noted:

  a.  Round trip. bf16 experts -> prepare_experts(bf16) -> the fp8 bytes and uint8 scales
      recovered from that handle by the inverse written here -> prepare_experts(fp8, uint8).
      Every tensor of the two handles is equal (scales also in shape and strides), and
      layer() outputs are torch.equal at M in {1, 7, 64, 513}. Families C (E 256, H 2048,
      I 512, top-8), B (E 128, H 2048, I 768, top-8) and a small one (E 8, H 256, I 256,
      top-4).
  a2. The same with the FC1 interleave off: the query prepare_experts asks is replaced for
      the case (the in-process stand-in for FSO_FC1_FUSED=0, whose C++ switch is read once
      per process), so both handles keep the [gate; up] order and the layer takes the
      unfused FC1. The handle keeps the caller's weight tensors there.
  b.  Codes no RCEIL rule produces: random finite e4m3 codes (never 0x7F / 0xFF) and random
      UE8M0 bytes in [118, 136] drawn from a separate generator, so the scales have nothing
      to do with each block's amax. The handle's weight bytes equal the input bytes after the
      interleave and its scale words equal the packing computed here, which is the proof
      that nothing was requantized; layer() against an fp32 reference on the dequantized
      weights (same routing) has cosine >= 0.995 as a sanity check of the byte meaning.
  c.  Dense. prepare_weight(fp8, uint8) gives the same handle (weight bytes, scale values
      and strides) as prepare_weight(fp8, int32) in both int32 stride forms and as
      prepare_weight(bf16), for fso-quantized bytes; also from a column-major uint8 copy, a
      uint8 slice of a wider buffer and a contiguous uint8 view that starts 3 bytes into its
      buffer. linear() outputs are torch.equal. Shapes: the Qwen3-4B
      projections at M in {1000, 64, 7, 1}, and K = 128 (one scale word per row) for the
      handle alone. The MXFP8 dense GEMMs require N % 128 == 0, so no other N is accepted
      (case d checks the refusal).
  c2. Dense with random codes and bytes as in (b): weight kept, scale equal to the packing
      computed here, linear() against the fp32 reference cosine >= 0.995.
  d.  Every rejected form raises ValueError naming the architecture, and, where the scale
      is at fault, the expected dtype (uint8) and shape.
  e.  sm_90: format="mxfp8" with float8_e4m3fn weights and uint8 scales raises
      NotImplementedError naming sm_90 on both entries (sm_90 has no MXFP8 hardware).
"""
import sys

import torch
import torch.nn.functional as F

import fish_scales_ops as fso

C = fso.compat
D = fso.dense

FAMILIES = {
    # label: (E, topk, hidden, inter)
    "C_35a3": (256, 8, 2048, 512),
    "B_30a3": (128, 8, 2048, 768),
    "small": (8, 4, 256, 256),
}
M_LAYER = (1, 7, 64, 513)
DENSE_SHAPES = (("down", 2560, 9728), ("wo", 2560, 4096), ("wqkv", 6144, 2560), ("gate_up", 19456, 2560))
M_DENSE = (1000, 64, 7, 1)
COS_RANDOM_CODES = 0.995
FP8 = torch.float8_e4m3fn

failures: list[str] = []


def _check(ok: bool, msg: str) -> bool:
    if not ok:
        failures.append(msg)
    return ok


def _ok(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def _arch_label() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return f"sm_{major}{minor}"


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()


def same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal dtype, shape and bytes element by element (fp8 compared as uint8)."""
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    if a.dtype == FP8:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    return torch.equal(a, b)


def same_scale(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal values and equal strides, so the bytes a GEMM reads by raw pointer are equal too."""
    return a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and a.stride() == b.stride() and torch.equal(a, b)


# --- the layouts, written from the formulas -------------------------------------------

def words_from_bytes(u8: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K/32] -> int32 [..., K/128]: the four bytes of one 128-wide K block in one
    word, the lowest K block in the lowest byte (b0 | b1 << 8 | b2 << 16 | b3 << 24)."""
    b = u8.to(torch.int64).reshape(*u8.shape[:-1], -1, 4)
    w = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)
    w = torch.where(w >= 2 ** 31, w - 2 ** 32, w)
    return w.to(torch.int32)


def bytes_from_words(w: torch.Tensor) -> torch.Tensor:
    """int32 [..., K/128] -> uint8 [..., K/32], the inverse of words_from_bytes."""
    w64 = w.to(torch.int64) & 0xFFFFFFFF
    b = torch.stack([(w64 >> (8 * j)) & 0xFF for j in range(4)], dim=-1)
    return b.reshape(*w.shape[:-1], -1).to(torch.uint8)


def atom_index(n_rows: int, kb: int, device) -> torch.Tensor:
    """[n_rows, kb] word positions of the sm_100/103 Sm1xx atom layout: one 128-word block per
    (128-row block, K block), blocks K-minor, row r of a block at (r % 32) * 4 + r / 32."""
    m = torch.arange(n_rows, dtype=torch.long, device=device)[:, None]
    kp = torch.arange(kb, dtype=torch.long, device=device)[None, :]
    r = m % 128
    return ((m // 128) * kb + kp) * 128 + (r % 32) * 4 + r // 32


def grouped_words_from_handle(sw: torch.Tensor, n: int, kb: int, major: int) -> torch.Tensor:
    """A grouped scale handle -> its logical words [G, N, kb]."""
    if major == 12:  # [G, kb, N], K-major words
        return sw.transpose(1, 2)
    g = int(sw.shape[0])
    return sw.reshape(g, -1)[:, atom_index(n, kb, sw.device).reshape(-1)].reshape(g, n, kb)


def grouped_handle_from_words(words: torch.Tensor, major: int) -> torch.Tensor:
    """Logical words [G, N, kb] -> the grouped handle (the reference packing of case b)."""
    g, n, kb = (int(v) for v in words.shape)
    if major == 12:
        return words.transpose(1, 2).contiguous()
    n_pad = (n + 127) // 128 * 128
    out = torch.zeros(g, n_pad * kb, dtype=torch.int32, device=words.device)
    out[:, atom_index(n, kb, words.device).reshape(-1)] = words.reshape(g, -1)
    return out


def dense_words_from_handle(s: torch.Tensor, n: int, kb: int, major: int) -> torch.Tensor:
    if major == 12:  # [N, kb], strides (1, N)
        return s
    return s[atom_index(n, kb, s.device).reshape(-1)].reshape(n, kb)


def dense_handle_from_words(words: torch.Tensor, major: int) -> torch.Tensor:
    n, kb = (int(v) for v in words.shape)
    if major == 12:
        return words.t().contiguous().t()  # the K-major strides (1, N)
    out = torch.zeros(((n + 127) // 128 * 128) * kb, dtype=torch.int32, device=words.device)
    out[atom_index(n, kb, words.device).reshape(-1)] = words.reshape(-1)
    return out


def interleave_rows(t: torch.Tensor) -> torch.Tensor:
    """[G, 2I, ...] in [gate; up] order -> rows g0, u0, g1, u1, ..."""
    i = int(t.shape[1]) // 2
    out = torch.empty_like(t)
    out[:, 0::2] = t[:, :i]
    out[:, 1::2] = t[:, i:]
    return out


def deinterleave_rows(t: torch.Tensor) -> torch.Tensor:
    return torch.cat([t[:, 0::2], t[:, 1::2]], dim=1)


def dequantize(codes_u8: torch.Tensor, sf_u8: torch.Tensor) -> torch.Tensor:
    """fp32 values of MXFP8 rows: e4m3 value times 2^(b - 127) of its 32-element block."""
    n, k = int(codes_u8.shape[-2]), int(codes_u8.shape[-1])
    v = codes_u8.view(FP8).float().reshape(*codes_u8.shape[:-1], k // 32, 32)
    s = torch.exp2(sf_u8.float() - 127.0)
    return (v * s[..., None]).reshape(*codes_u8.shape[:-2], n, k)


# --- inputs ------------------------------------------------------------------------

def make_weights(E, H, I, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w13 = (torch.randn(E, 2 * I, H, device="cuda", generator=g) / H ** 0.5).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device="cuda", generator=g) / I ** 0.5).to(torch.bfloat16)
    return w13, w2


def make_inputs(M, E, TOPK, H, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden = (torch.randn(M, H, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    ids = torch.rand(M, E, device="cuda", generator=g).topk(TOPK, dim=1).indices
    wts = torch.softmax(torch.rand(M, TOPK, device="cuda", generator=g), dim=1).float()
    return hidden, ids, wts


def random_codes(shape, seed) -> torch.Tensor:
    """uint8 e4m3 codes, uniform over every finite code (0x7F and 0xFF, the NaNs, moved
    to 0x7E and 0xFE)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    c = torch.randint(0, 256, shape, device="cuda", generator=g, dtype=torch.int32).to(torch.uint8)
    nan = (c == 0x7F) | (c == 0xFF)
    return torch.where(nan, c - 1, c)


def random_scale_bytes(shape, seed) -> torch.Tensor:
    """UE8M0 bytes in [118, 136] from their own generator: scales 2^-9 .. 2^9, unrelated to
    any block's amax."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(118, 137, shape, device="cuda", generator=g, dtype=torch.int32).to(torch.uint8)


def ref_layer(hidden, ids, wts, deq13, deq2, E, I):
    """fp32 MoE layer on dequantized weights; deq13(e) is [2I, H] in [gate; up] order."""
    x = hidden.float()
    out = torch.zeros_like(x)
    for e in range(E):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        w13 = deq13(e)
        w2 = deq2(e)
        xe = x[tok]
        h = F.silu(xe @ w13[:I].t()) * (xe @ w13[I:].t())
        out.index_add_(0, tok, (h @ w2.t()) * wts[tok, slot][:, None])
    return out


# --- (a) / (a2) round trip ------------------------------------------------------------

def recover_checkpoint(ex, major):
    """The checkpoint form of an MXFP8 handle: fp8 weights in [gate; up] order and uint8
    scales [E, N, K/32], as new tensors."""
    E, H, I = ex.num_experts, ex.hidden, ex.inter
    q13 = ex.w13.view(torch.uint8)
    words13 = grouped_words_from_handle(ex.sw13, 2 * I, H // 128, major)
    if ex.w13_interleaved:
        q13 = deinterleave_rows(q13)
        words13 = deinterleave_rows(words13)
    sf13 = bytes_from_words(words13).contiguous()
    sf2 = bytes_from_words(grouped_words_from_handle(ex.sw2, H, I // 128, major)).contiguous()
    return (q13.contiguous().clone().view(FP8), ex.w2.view(torch.uint8).clone().view(FP8), sf13, sf2)


def handles_equal(a, b) -> list[str]:
    bad = []
    for f in ("kind", "arch", "num_experts", "hidden", "inter", "w13_interleaved"):
        if getattr(a, f) != getattr(b, f):
            bad.append(f"{f} {getattr(a, f)!r} != {getattr(b, f)!r}")
    for f in ("w13", "w2"):
        if not same_bytes(getattr(a, f), getattr(b, f)):
            bad.append(f)
    for f in ("sw13", "sw2"):
        if not same_scale(getattr(a, f), getattr(b, f)):
            bad.append(f"{f} ({getattr(a, f).dtype} {tuple(getattr(a, f).shape)} {getattr(a, f).stride()} vs "
                       f"{getattr(b, f).dtype} {tuple(getattr(b, f).shape)} {getattr(b, f).stride()})")
    return bad


def round_trip(major: int, families, label: str) -> None:
    for fam in families:
        E, TOPK, H, I = FAMILIES[fam]
        w13, w2 = make_weights(E, H, I, seed=101 + E + I)
        ex1 = fso.moe.prepare_experts(w13, w2, format="mxfp8")
        q13, q2, sf13, sf2 = recover_checkpoint(ex1, major)
        ex2 = fso.moe.prepare_experts(q13, q2, format="mxfp8", sw13=sf13, sw2=sf2)
        bad = handles_equal(ex1, ex2)
        _check(not bad, f"{label} {fam}: the handle from fp8 + uint8 differs from the bf16 one in {bad}")
        kept_w2 = ex2.w2.data_ptr() == q2.data_ptr()
        kept_w13 = ex2.w13.data_ptr() == q13.data_ptr()
        _check(kept_w2 and kept_w13 == (not ex2.w13_interleaved),
               f"{label} {fam}: tensor sharing with the caller is not as documented (w2 kept {kept_w2}, w13 kept "
               f"{kept_w13}, interleaved {ex2.w13_interleaved})")
        n_eq = 0
        for M in M_LAYER:
            hidden, ids, wts = make_inputs(M, E, TOPK, H, seed=7 * M + E)
            y1 = fso.moe.layer(hidden, ex1, ids, wts)
            y2 = fso.moe.layer(hidden, ex2, ids, wts)
            eq = torch.equal(y1, y2) and bool(torch.isfinite(y1).all())
            n_eq += int(eq)
            _check(eq, f"{label} {fam} M={M}: layer() on the fp8 + uint8 handle != on the bf16 handle")
        ok = not bad and n_eq == len(M_LAYER)
        print(f"  {label:2s} {fam:7s} E={E:3d} H={H} I={I}: interleaved={ex2.w13_interleaved}; handle from fp8 + "
              f"uint8 == handle from bf16 ({'all tensors' if not bad else 'DIFFERS: ' + ', '.join(bad)}); "
              f"layer() torch.equal at M {list(M_LAYER)}: {n_eq}/{len(M_LAYER)}; caller's w2 kept "
              f"{kept_w2}, w13 kept {kept_w13}  {_ok(ok)}")
        del w13, w2, ex1, ex2, q13, q2, sf13, sf2
        torch.cuda.empty_cache()


def case_a(major: int) -> None:
    round_trip(major, FAMILIES, "a")


def case_a2(major: int) -> None:
    moe_mod = fso.moe
    saved = moe_mod.mxfp8_grouped_swiglu_available
    moe_mod.mxfp8_grouped_swiglu_available = lambda n_w, k: False
    try:
        round_trip(major, ("small", "C_35a3"), "a2")
    finally:
        moe_mod.mxfp8_grouped_swiglu_available = saved


# --- (b) codes no RCEIL rule produces -------------------------------------------------

def case_b(major: int) -> None:
    for fam, (E, TOPK, H, I) in FAMILIES.items():
        c13 = random_codes((E, 2 * I, H), seed=11 + E)
        c2 = random_codes((E, H, I), seed=13 + E)
        s13 = random_scale_bytes((E, 2 * I, H // 32), seed=17 + E)
        s2 = random_scale_bytes((E, H, I // 32), seed=19 + E)
        ex = fso.moe.prepare_experts(c13.view(FP8), c2.view(FP8), format="mxfp8", sw13=s13, sw2=s2)
        want13 = interleave_rows(c13) if ex.w13_interleaved else c13
        words13 = words_from_bytes(s13)
        ref_sw13 = grouped_handle_from_words(interleave_rows(words13) if ex.w13_interleaved else words13, major)
        ref_sw2 = grouped_handle_from_words(words_from_bytes(s2), major)
        bytes_ok = torch.equal(ex.w13.view(torch.uint8), want13) and torch.equal(ex.w2.view(torch.uint8), c2)
        scale_ok = same_scale(ex.sw13, ref_sw13) and same_scale(ex.sw2, ref_sw2)
        _check(bytes_ok, f"b {fam}: the handle's weight bytes are not the input bytes (after the interleave)")
        _check(scale_ok, f"b {fam}: the handle's scale words are not the packing of the input bytes")
        cos_all = []
        for M in M_LAYER:
            hidden, ids, wts = make_inputs(M, E, TOPK, H, seed=31 * M + E)
            y = fso.moe.layer(hidden, ex, ids, wts)
            ref = ref_layer(hidden, ids, wts, lambda e: dequantize(c13[e], s13[e]),
                            lambda e: dequantize(c2[e], s2[e]), E, I)
            c = _cos(y, ref) if bool(torch.isfinite(y).all()) else float("nan")
            cos_all.append(c)
            _check(c >= COS_RANDOM_CODES, f"b {fam} M={M}: cosine {c:.6f} < {COS_RANDOM_CODES} against the fp32 "
                                          f"reference on the dequantized weights")
        ok = bytes_ok and scale_ok and all(c >= COS_RANDOM_CODES for c in cos_all)
        print(f"  b  {fam:7s}: random e4m3 codes + UE8M0 bytes in [118, 136]: weight bytes == input "
              f"{'yes' if bytes_ok else 'NO'}, scale words == packing computed here {'yes' if scale_ok else 'NO'}; "
              f"layer() vs fp32 on the dequantized weights cos at M {list(M_LAYER)} = "
              f"{', '.join(f'{c:.6f}' for c in cos_all)}  {_ok(ok)}")
        del c13, c2, s13, s2, ex
        torch.cuda.empty_cache()


# --- (c) / (c2) dense -----------------------------------------------------------------

def make_w(N, K, seed):
    torch.manual_seed(seed)
    return (torch.randn(N, K, device="cuda") / K ** 0.5).bfloat16()


def make_x(M, K, seed):
    torch.manual_seed(seed)
    return (torch.randn(M, K, device="cuda") * 0.1).bfloat16()


def case_c(major: int) -> None:
    for name, N, K in DENSE_SHAPES + (("k128", 256, 128),):
        kb = K // 128
        w = make_w(N, K, seed=N * 17 + K + 5)
        wq, sw = C.quantize_1x32_fp8(w)
        u8 = bytes_from_words(dense_words_from_handle(sw, N, kb, major)).contiguous()
        dw_u8 = D.prepare_weight(wq, format="mxfp8", scale=u8)
        others = {
            "int32 as quantized": D.prepare_weight(wq, format="mxfp8", scale=sw),
            "int32 .contiguous()": D.prepare_weight(wq, format="mxfp8", scale=sw.contiguous()),
            "bf16": D.prepare_weight(w, format="mxfp8"),
            "uint8 column-major": D.prepare_weight(wq, format="mxfp8", scale=u8.t().contiguous().t()),
        }
        wide = torch.zeros(N, K // 32 + 8, dtype=torch.uint8, device="cuda")
        wide[:, 3:3 + K // 32] = u8
        others["uint8 slice"] = D.prepare_weight(wq, format="mxfp8", scale=wide[:, 3:3 + K // 32])
        # Contiguous, but 3 bytes into its buffer: not reinterpretable as int32 in place.
        flat = torch.zeros(N * (K // 32) + 3, dtype=torch.uint8, device="cuda")
        flat[3:] = u8.reshape(-1)
        others["uint8 at byte offset 3"] = D.prepare_weight(wq, format="mxfp8", scale=flat[3:].view(N, K // 32))
        bad =[k for k, dw in others.items()
               if not (dw.kind == dw_u8.kind and same_bytes(dw.weight, dw_u8.weight) and same_scale(dw.scale, dw_u8.scale))]
        kept = dw_u8.weight.data_ptr() == wq.data_ptr()
        _check(not bad, f"c {name}: prepare_weight(fp8, uint8) differs from {bad}")
        _check(kept, f"c {name}: prepare_weight(fp8, uint8) did not keep the caller's weight tensor")
        n_eq = 0
        ms = M_DENSE if name != "k128" else ()
        for M in ms:
            x = make_x(M, K, seed=M * 1009 + N * 17 + K)
            ys = [D.linear(x, dw) for dw in [dw_u8] + list(others.values())]
            eq = all(torch.equal(y, ys[0]) for y in ys[1:]) and bool(torch.isfinite(ys[0]).all())
            n_eq += int(eq)
            _check(eq, f"c {name} M={M}: linear() outputs differ between the scale forms")
        ok = not bad and kept and n_eq == len(ms)
        lin = f"linear() torch.equal at M {list(ms)}: {n_eq}/{len(ms)}" if ms else "handle only"
        print(f"  c  {name:8s} N={N:5d} K={K:5d}: uint8 scale == {', '.join(others)} (weight bytes, scale values and "
              f"strides {tuple(dw_u8.scale.stride())}); weight kept {kept}; {lin}  {_ok(ok)}")
        del w, wq, sw, u8, dw_u8, others, wide, flat
    torch.cuda.empty_cache()


def case_c2(major: int) -> None:
    for name, N, K in (("wo", 2560, 4096), ("gate_up", 19456, 2560)):
        codes = random_codes((N, K), seed=N + K)
        sf = random_scale_bytes((N, K // 32), seed=N + K + 1)
        dw = D.prepare_weight(codes.view(FP8), format="mxfp8", scale=sf)
        ref_s = dense_handle_from_words(words_from_bytes(sf), major)
        bytes_ok = torch.equal(dw.weight.view(torch.uint8), codes)
        scale_ok = same_scale(dw.scale, ref_s)
        _check(bytes_ok and scale_ok, f"c2 {name}: weight bytes kept {bytes_ok}, scale == packing here {scale_ok}")
        wf = dequantize(codes, sf)
        cos_all = []
        for M in M_DENSE:
            x = make_x(M, K, seed=M + N)
            y = D.linear(x, dw)
            c = _cos(y, x.float() @ wf.t()) if bool(torch.isfinite(y).all()) else float("nan")
            cos_all.append(c)
            _check(c >= COS_RANDOM_CODES, f"c2 {name} M={M}: cosine {c:.6f} < {COS_RANDOM_CODES}")
        ok = bytes_ok and scale_ok and all(c >= COS_RANDOM_CODES for c in cos_all)
        print(f"  c2 {name:8s} N={N:5d} K={K:5d}: random codes + bytes: weight bytes kept {bytes_ok}, scale == "
              f"packing here {scale_ok}; linear() vs fp32 cos at M {list(M_DENSE)} = "
              f"{', '.join(f'{c:.6f}' for c in cos_all)}  {_ok(ok)}")
        del codes, sf, dw, wf
    torch.cuda.empty_cache()


# --- (d) refusals ---------------------------------------------------------------------

def _expect_value_error(label: str, fn, needles=()) -> str:
    arch = _arch_label()
    try:
        fn()
    except ValueError as exc:
        msg = str(exc)
        missing = [s for s in (arch,) + tuple(needles) if s not in msg]
        _check(not missing, f"{label}: the message does not name {missing}: {msg}")
        return "ValueError" + (f" (does NOT name {missing})" if missing else "")
    except Exception as exc:  # noqa: BLE001
        _check(False, f"{label}: raised {type(exc).__name__} instead of ValueError: {exc}")
        return f"wrong exception {type(exc).__name__}"
    _check(False, f"{label}: did not raise")
    return "did not raise"


def case_d(major: int) -> None:
    E, H, I = 4, 256, 128
    w13, w2 = make_weights(E, H, I, seed=3)
    ex = fso.moe.prepare_experts(w13, w2, format="mxfp8")
    q13, q2, s13, s2 = recover_checkpoint(ex, major)
    want13, want2 = str([E, 2 * I, H // 32]), str([E, H, I // 32])
    P = fso.moe.prepare_experts
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    cases = {
        "experts: bf16 with uint8 scales": (lambda: P(w13, w2, format="mxfp8", sw13=s13, sw2=s2),
                                            ("uint8", want13, want2)),
        "experts: fp8 without scales": (lambda: P(q13, q2, format="mxfp8"), ("uint8", want13)),
        "experts: fp8 with sw13 only": (lambda: P(q13, q2, format="mxfp8", sw13=s13), ("uint8", want2)),
        "experts: fp8 w13 with bf16 w2": (lambda: P(q13, w2, format="mxfp8", sw13=s13, sw2=s2),
                                          ("uint8", want13, want2)),
        "experts: fp16 weights": (lambda: P(w13.half(), w2.half(), format="mxfp8"), ("uint8", want13, want2)),
        "experts: int32 handle as scales": (lambda: P(q13, q2, format="mxfp8", sw13=ex.sw13, sw2=ex.sw2),
                                            ("uint8", want13)),
        "experts: int32 [E, N, K/32] scales": (lambda: P(q13, q2, format="mxfp8", sw13=s13.int(), sw2=s2.int()),
                                               ("uint8", want13)),
        "experts: fp32 scales": (lambda: P(q13, q2, format="mxfp8", sw13=s13.float(), sw2=s2.float()),
                                 ("uint8", want13)),
        "experts: sw13 [E, 2I, H/128]": (lambda: P(q13, q2, format="mxfp8", sw13=s13[:, :, ::4].contiguous(),
                                                   sw2=s2), ("uint8", want13)),
        "experts: sw2 [E, I, H/32] (transposed)": (
            lambda: P(q13, q2, format="mxfp8", sw13=s13,
                      sw2=torch.zeros(E, I, H // 32, dtype=torch.uint8, device="cuda")), ("uint8", want2)),
        "experts: scales on the CPU": (lambda: P(q13, q2, format="mxfp8", sw13=s13.cpu(), sw2=s2.cpu()), ()),
    }
    if e8m0 is not None:
        cases["experts: float8_e8m0fnu scales"] = (
            lambda: P(q13, q2, format="mxfp8", sw13=s13.view(e8m0), sw2=s2.view(e8m0)),
            ("uint8", want13, "view(torch.uint8)"))
    N, K = 256, 512
    w = make_w(N, K, seed=4)
    wq, sw = C.quantize_1x32_fp8(w)
    u8 = bytes_from_words(dense_words_from_handle(sw, N, K // 128, major)).contiguous()
    wantd = str([N, K // 32])
    PW = D.prepare_weight
    cases.update({
        "dense: uint8 [N, K/128]": (lambda: PW(wq, format="mxfp8", scale=u8[:, ::4].contiguous()), ("uint8", wantd)),
        "dense: uint8 [K/32, N]": (lambda: PW(wq, format="mxfp8", scale=u8.t().contiguous()), ("uint8", wantd)),
        "dense: bf16 weight with a uint8 scale": (lambda: PW(w, format="mxfp8", scale=u8), ("uint8", wantd)),
        "dense: fp8 weight without its scale": (lambda: PW(wq, format="mxfp8"), ("uint8", wantd)),
        "dense: fp32 scale": (lambda: PW(wq, format="mxfp8", scale=u8.float()), ("uint8", wantd)),
        "dense: uint8 scale on the CPU": (lambda: PW(wq, format="mxfp8", scale=u8.cpu()), ()),
        "dense: N % 128 != 0 with a uint8 scale": (
            lambda: PW(random_codes((200, 256), seed=5).view(FP8), format="mxfp8",
                       scale=random_scale_bytes((200, 8), seed=6)), ()),
    })
    if e8m0 is not None:
        cases["dense: float8_e8m0fnu scale"] = (lambda: PW(wq, format="mxfp8", scale=u8.view(e8m0)),
                                                ("uint8", wantd, "view(torch.uint8)"))
    for label, (fn, needles) in cases.items():
        got = _expect_value_error(label, fn, needles)
        print(f"  d  refuse {label:42s} -> {got}")


# --- (e) sm_90 ------------------------------------------------------------------------

def case_e() -> None:
    E, H, I = 4, 256, 128
    q13 = random_codes((E, 2 * I, H), seed=1).view(FP8)
    q2 = random_codes((E, H, I), seed=2).view(FP8)
    s13 = random_scale_bytes((E, 2 * I, H // 32), seed=3)
    s2 = random_scale_bytes((E, H, I // 32), seed=4)
    wq = random_codes((256, 512), seed=5).view(FP8)
    sq = random_scale_bytes((256, 16), seed=6)
    for label, fn in (("prepare_experts", lambda: fso.moe.prepare_experts(q13, q2, format="mxfp8", sw13=s13, sw2=s2)),
                      ("prepare_weight", lambda: D.prepare_weight(wq, format="mxfp8", scale=sq))):
        try:
            fn()
            _check(False, f"e {label}: format='mxfp8' with fp8 + uint8 did not raise on sm_90")
            got = "did not raise"
        except NotImplementedError as exc:
            named = "sm_90" in str(exc)
            _check(named, f"e {label}: the refusal does not name sm_90: {exc}")
            got = "NotImplementedError" + ("" if named else " (sm_90 NOT named)")
        except Exception as exc:  # noqa: BLE001
            _check(False, f"e {label}: raised {type(exc).__name__} instead of NotImplementedError: {exc}")
            got = f"wrong exception {type(exc).__name__}"
        print(f"  e  {label}(format='mxfp8', fp8 + uint8) on sm_90 -> {got}")


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: test_prequant_mxfp8 needs a CUDA device")
        return 0
    torch.backends.cuda.matmul.allow_tf32 = False
    major, minor = torch.cuda.get_device_capability(0)
    print(f"device: {torch.cuda.get_device_name(0)} sm_{major}{minor}; torch {torch.__version__}")
    covered, skipped = [], []
    if major in (10, 12):
        print("== (a) round trip: bf16 -> handle -> fp8 + uint8 -> handle ==")
        case_a(major)
        print("== (a2) round trip with the FC1 interleave off ==")
        case_a2(major)
        print("== (b) codes and scales no RCEIL rule produces ==")
        case_b(major)
        print("== (c) dense: uint8 scale vs int32 forms and bf16 ==")
        case_c(major)
        print("== (c2) dense: random codes and scale bytes ==")
        case_c2(major)
        print("== (d) refusals ==")
        case_d(major)
        covered += ["a", "a2", "b", "c", "c2", "d"]
        skipped.append("e (sm_90)")
    elif major == 9:
        print("== (e) sm_90 refuses format='mxfp8' ==")
        case_e()
        covered.append("e")
        skipped.append("a, a2, b, c, c2, d (sm_100/103, sm_120/121: no MXFP8 hardware here)")
    else:
        print(f"SKIP: sm_{major}{minor} serves no MXFP8 format")
        return 0
    print(f"\ncovered: {', '.join(covered)}; skipped on sm_{major}{minor}: {', '.join(skipped)}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("pre-quantized MXFP8 entries: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
