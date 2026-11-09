"""The sm_120/sm_121 dense MXFP8 cascade's ring-mainloop instances and the routes of issue #12 F5B
(arch/sm120/mxfp8/dispatch.cuh, fp8/gemm_1d1d.cuh).

1. **The routes.** On the RTX 5090's 170 SMs the route query ``torch.ops.fish_scales_ops.mxfp8_dense_route_sm120``
   returns the table below: the stage rule (a single-wave launch of (64, 128) or (96, 128) takes 3 stages with 3 or
   more M-tiles, or with 2 M-tiles on no more than 3/4 of the SMs), the small-tile stage rule (a single launch of
   (32, 64, 4) takes 6 stages when K / 128 is not a multiple of 4, or when M <= 64, K >= 2048 and the grid has at least
   40 tiles), the tie-break of the wave-fill rule ((64, 128) over (128, 64) at equal waves x (TileM + TileN)),
   and the decode rules that replaced the Stream-K splits at M <= 64 (narrow N with K >= 8192 at M <= 16: (16, 64, 6);
   narrow N at 32 < M <= 64 with K >= 9728: (32, 64, 6); 32 <= tiles_n < 96 at M <= 32: (16, 128, 4)), all single
   launches; narrow N at 16 < M <= 32 keeps its Stream-K split. A change of a rule must update the table on purpose.
2. **The bytes do not depend on the stage count.** Each ring instance (3 or 6 stages, which the original loop cannot
   run) gives the bytes of the original-loop instance of the same tile (2 or 4 stages), forced through
   ``FSO_FORCE_TILE`` (one subprocess each, the knob is read once per process), on the 2b call, the bf16-x call and
   ``linear_quantized``, with a bias + erf-GELU epilogue, at K values whose last 4-k-tile scale span is partial (the ring
   takes the real k-tile count) and at an M whose tiles outnumber the SMs (CTAs run several tiles, the ring position
   carries from tile to tile).
3. **The decode single launch.** On the Qwen3-4B ``down`` shape at M = 1 and 16 the route is the single launch of
   (16, 64, 6); its output equals the forced (16, 64, 4) single launch byte for byte (a single launch sums K in fp32 in
   one pass, whatever the tile), and it is at least as close to the fp32 reference as the forced Stream-K route it
   replaced (FSO_FORCE_TILE=16,128,4, FSO_FORCE_KSPLIT=4, bf16 partial sums).

Runs on sm_120/sm_121 only; skips elsewhere. Run: ``PYTHONPATH=python python tests/gemm/unit/test_sm120_ring_stages.py``.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

KNOBS = ("FSO_FORCE_TILE", "FSO_FORCE_KSPLIT", "FSO_FORCE_SMALLM", "FSO_FORCE_MIN_BLOCKS", "FSO_FORCE_SCHED_GROUP",
         "FSO_FORCE_TILE_K", "FSO_DISABLE_OVERRIDES", "FSO_DISABLE_STREAMK", "FSO_SMALLM_ROUTE")
SMS_TABLE = 170
R_EXPECTED = (  # (M, N, K, TileM, TileN, stages, k_split, what) at 170 SMs
    (128, 6144, 2560, 64, 128, 3, 1, "stage rule, 2 M-tiles"), (256, 6144, 2560, 96, 128, 3, 1, "stage rule"),
    (512, 2560, 4096, 64, 128, 3, 1, "tie-break, then the stage rule"), (64, 19456, 2560, 64, 128, 2, 1, "1 M-tile"),
    (182, 5120, 1280, 64, 128, 3, 1, "stage rule"), (1079, 1280, 1280, 64, 128, 3, 1, "stage rule"),
    (533, 2048, 1280, 64, 128, 3, 1, "tie-break"), (1024, 1024, 2048, 64, 128, 3, 1, "tie-break"),
    (128, 9216, 2048, 64, 128, 2, 1, "2 M-tiles on more than 3/4 of the SMs"),
    (192, 12288, 2048, 64, 128, 2, 1, "multi-wave (F5C's tile rule: 96 x 128 -> 64 x 128, no third stage)"),
    (533, 5120, 1280, 160, 128, 2, 1, "multi-wave (F5C's tile rule: 96 x 128 -> 160 x 128)"),
    (117, 1280, 1280, 32, 64, 6, 1, "small-tile rule, 10 k-tiles"), (48, 2560, 2048, 32, 64, 6, 1, "small-tile rule"),
    (64, 1024, 2048, 32, 64, 4, 1, "small-tile rule, 32 tiles"), (96, 2048, 2048, 32, 64, 4, 1, "3 M-tiles"),
    (64, 2048, 512, 32, 64, 4, 1, "4 k-tiles"),
    (1, 2560, 9728, 16, 64, 6, 1, "decode, was (16,128,4) Stream-K 4"),
    (16, 2560, 9728, 16, 64, 6, 1, "decode, was (16,128,4) Stream-K 4"),
    (32, 2560, 9728, 32, 64, 4, 4, "decode, narrow N at 16 < M <= 32 keeps Stream-K"),
    (64, 2560, 9728, 32, 64, 6, 1, "decode, was (64,64,4) Stream-K 4"),
    (32, 6144, 2560, 16, 128, 4, 1, "decode, was (32,128,4) Stream-K 2"),
    (32, 9728, 2560, 16, 128, 4, 1, "decode, was (32,128,4) Stream-K 2"),
    (128, 2560, 9728, 64, 128, 2, 4, "Stream-K kept above M = 64"),
    (512, 2560, 9728, 128, 128, 2, 2, "Stream-K kept"),
)
# (ring instance, original-loop instance of the same tile, M, N, K): K = 1280 and 768 leave a partial last scale span,
# M = 2048 on 64-128-row tiles gives more tiles than SMs
PAIRS = (
    ("64,128,3", "64,128,2", 300, 2560, 1280), ("64,128,3", "64,128,2", 2048, 4096, 768),
    ("96,128,3", "96,128,2", 400, 4096, 1280), ("96,128,3", "96,128,2", 1500, 5120, 2048),
    ("16,64,6", "16,64,4", 16, 2560, 9728), ("32,64,6", "32,64,4", 64, 2560, 1280),
)
failures: list[str] = []


def check(ok: bool, msg: str) -> bool:
    if not ok:
        failures.append(msg)
    return ok


def route_of(M, N, K):
    import torch
    r = [int(v) for v in torch.ops.fish_scales_ops.mxfp8_dense_route_sm120(M, N, K)]
    return (r[0], r[1], r[2]), r[5]


def case_routes(sms: int) -> None:
    if sms != SMS_TABLE:
        print(f"  1  the route table is for {SMS_TABLE} SMs; this device has {sms}: table not checked")
        return
    bad = []
    for M, N, K, tm, tn, st, ks, what in R_EXPECTED:
        t, k = route_of(M, N, K)
        if (t, k) != ((tm, tn, st), ks):
            bad.append((what, M, N, K, (t, k), ((tm, tn, st), ks)))
    check(not bad, f"routes: {len(bad)} of {len(R_EXPECTED)} table cells differ, e.g. {bad[:4]}")
    print(f"  1  the route table at {SMS_TABLE} SMs: {len(R_EXPECTED) - len(bad)} of {len(R_EXPECTED)} cells match  "
          f"{'OK' if not bad else 'FAILED'}")


def worker(spec: str) -> int:
    import torch
    import torch.nn.functional as F
    import fish_scales_ops as fso
    c = json.loads(spec)
    M, N, K = c["M"], c["N"], c["K"]
    torch.manual_seed(M * 1009 + N * 17 + K)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.5
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.5)
    b = torch.randn(N, dtype=torch.bfloat16, device="cuda") * 0.1
    dw = fso.dense.prepare_weight(w, format="mxfp8")
    xq, sx = fso.dense.quantize_activation(x)
    md5 = lambda t: hashlib.md5(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()  # noqa: E731
    y2 = fso.dense.linear(xq, dw, x_scale=sx)
    ye = fso.dense.linear(xq, dw, x_scale=sx, bias=b, activation="gelu_erf")
    yx = fso.dense.linear(x, dw, bias=b, activation="gelu_erf")
    out = {"md5_2b": md5(y2), "md5_epi": md5(ye), "md5_x_epi": md5(yx)}
    if N % 128 == 0:
        q, s = fso.dense.linear_quantized(xq, dw, x_scale=sx, bias=b, activation="gelu_erf")
        out["md5_q"] = md5(q) + md5(s)
    torch.cuda.synchronize()
    # the fp32 reference of the 2b call: dequantized operands, fp32 product (TF32 off)
    def deq(qt, sc, rows):
        s8 = sc[:rows].contiguous().view(torch.uint8)
        return qt.float() * torch.exp2(s8.float() - 127.0).repeat_interleave(32, dim=1)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = deq(xq, sx, M) @ deq(dw.weight, dw.scale, N).t()
    torch.backends.cuda.matmul.allow_tf32 = prev
    d = (y2.float() - ref).abs()
    out["max_abs"] = float(d.max())
    out["cos"] = float(F.cosine_similarity(y2.float().flatten(), ref.flatten(), dim=0))
    t, ks = route_of(M, N, K)
    out["route"] = list(t)
    out["k_split"] = ks
    print("RS_RESULT " + json.dumps(out))
    return 0


def run_worker(spec: dict, env_add: dict) -> dict | None:
    env = {k: v for k, v in os.environ.items() if k not in KNOBS}
    env.update(env_add)
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", json.dumps(spec)], env=env,
                       capture_output=True, text=True, timeout=600)
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("RS_RESULT ")]
    if p.returncode != 0 or not line:
        check(False, f"worker {spec} {env_add} failed rc {p.returncode}: {p.stderr[-800:]}")
        return None
    return json.loads(line[0][len("RS_RESULT "):])


def case_bytes() -> None:
    for ring, orig, M, N, K in PAIRS:
        spec = {"M": M, "N": N, "K": K}
        a = run_worker(spec, {"FSO_FORCE_TILE": ring})
        b = run_worker(spec, {"FSO_FORCE_TILE": orig})
        if a is None or b is None:
            continue
        check(tuple(a["route"]) == tuple(int(v) for v in ring.split(",")),
              f"bytes: M={M} N={N} K={K}: FSO_FORCE_TILE={ring} ran {a['route']}")
        same = all(a[k] == b[k] for k in ("md5_2b", "md5_epi", "md5_x_epi", "md5_q") if k in a)
        check(same, f"bytes: M={M} N={N} K={K}: ring {ring} differs from {orig}")
        check(a["md5_epi"] == a["md5_x_epi"], f"bytes: M={M} N={N} K={K}: ring {ring}: bf16-x differs from the 2b call")
        print(f"  2  ring ({ring}) against ({orig}) at M={M} N={N} K={K}: 2b, epilogue, bf16-x"
              f"{', linear_quantized' if 'md5_q' in a else ''} identical  {'OK' if same else 'FAILED'}")


def case_decode() -> None:
    for M in (1, 16):
        spec = {"M": M, "N": 2560, "K": 9728}
        nat = run_worker(spec, {})
        single = run_worker(spec, {"FSO_FORCE_TILE": "16,64,4"})
        sk = run_worker(spec, {"FSO_FORCE_TILE": "16,128,4", "FSO_FORCE_KSPLIT": "4"})
        if nat is None or single is None or sk is None:
            continue
        check(tuple(nat["route"]) == (16, 64, 6) and nat["k_split"] == 1,
              f"decode: M={M}: the route is {nat['route']} k_split {nat['k_split']}, not the (16, 64, 6) single launch")
        check(nat["md5_2b"] == single["md5_2b"], f"decode: M={M}: (16, 64, 6) differs from the (16, 64, 4) single launch")
        check(nat["max_abs"] <= sk["max_abs"] and nat["cos"] >= sk["cos"],
              f"decode: M={M}: the single launch is farther from fp32 than Stream-K: max_abs {nat['max_abs']} vs "
              f"{sk['max_abs']}, cos {nat['cos']} vs {sk['cos']}")
        print(f"  3  decode M={M}: route {tuple(nat['route'])} single launch = (16,64,4) bytes; vs fp32 max_abs "
              f"{nat['max_abs']:.3g} (Stream-K {sk['max_abs']:.3g}), cos {nat['cos']:.7f} (Stream-K {sk['cos']:.7f})  "
              f"{'OK' if not failures else 'FAILED'}")


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        return worker(sys.argv[2])
    import torch
    import fish_scales_ops  # registers the ops
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        print("SKIP: the ring-mainloop instances are the sm_120/sm_121 MXFP8 cascade's")
        return 0
    if any(os.environ.get(k) for k in KNOBS):
        print(f"SKIP: a cascade knob is set ({[k for k in KNOBS if os.environ.get(k)]}); run with the defaults")
        return 0
    sms = int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
    print(f"== sm_120 MXFP8 cascade, ring-mainloop instances and the F5B routes ({torch.cuda.get_device_name()}, "
          f"{sms} SMs, fish_scales_ops {fish_scales_ops.__version__}) ==")
    case_routes(sms)
    case_bytes()
    case_decode()
    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for f in failures:
            print("  - " + f)
        return 1
    print("\nsm120 ring stages: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
