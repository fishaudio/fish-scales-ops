"""The wave-fill tile rule of the sm_120/sm_121 dense MXFP8 cascade (arch/sm120/mxfp8/dispatch.cuh,
sm120_mxfp8_dense_route): above M = 128, a pick that is a single launch whose grid has fewer CTAs than the device has
SMs is replaced by the existing cascade tile with the smallest waves x (TileM + TileN), waves = ceil(CTAs / SMs); a
32-row tile is a candidate only when the pick's last M-tile is partial.

1. **The routes.** On the RTX 5090's 170 SMs the route query ``torch.ops.fish_scales_ops.mxfp8_dense_route_sm120``
   returns the table below for the issue #12 tower and LLM TP2 shapes and the dense families' shapes (the rule's own
   arithmetic applied to the cascade's picks, issue #12 F5A; a change of the rule or of the cascade must update the
   table on purpose). On any device: every route above M = 64 is one of the cascade's tiles, and an under-filled
   single-launch route above M = 128 has a waves x (TileM + TileN) no larger than any tile with more than 32 rows.
2. **The bytes do not depend on the tile.** On cells the rule moves, the natural route and other forced single-launch
   tiles (``FSO_FORCE_TILE``, one subprocess each, since the knob is read once per process) give the same output on
   the 2b call and on the bf16-x call, with the issue #12 tower epilogues (bias, bias + residual, bias + erf-GELU), and
   the bf16-x call equals the 2b call byte for byte.
3. **The switch.** ``FSO_DISABLE_OVERRIDES=1`` gives back the route without the rule (a moved cell then runs its pick).

Runs on sm_120/sm_121 only (the route query and the cascade need the device); skips elsewhere.
Run: ``PYTHONPATH=python python tests/gemm/unit/test_sm120_wave_fill.py``.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

TILES = ((32, 64, 4), (32, 128, 4), (64, 64, 4), (64, 128, 2), (96, 128, 2), (128, 64, 2), (128, 128, 2), (160, 128, 2))
KNOBS = ("FSO_FORCE_TILE", "FSO_FORCE_KSPLIT", "FSO_FORCE_SMALLM", "FSO_FORCE_MIN_BLOCKS", "FSO_FORCE_SCHED_GROUP",
         "FSO_FORCE_TILE_K", "FSO_DISABLE_OVERRIDES", "FSO_DISABLE_STREAMK", "FSO_SMALLM_ROUTE")
SMS_TABLE = 170
W_EXPECTED = (  # (M, N, K, TileM, TileN, stages, k_split, tag) at 170 SMs
    (182, 1280, 1280, 32, 64, 4, 1, "q"), (364, 1280, 1280, 64, 64, 4, 1, "q"),
    (533, 1280, 1280, 32, 128, 4, 1, "q"), (1079, 1280, 1280, 64, 128, 2, 1, "q"),
    (182, 5120, 1280, 64, 128, 2, 1, "fc1"), (364, 5120, 1280, 96, 128, 2, 1, "fc1"),
    (533, 5120, 1280, 96, 128, 2, 1, "fc1"), (1079, 5120, 1280, 128, 128, 2, 1, "fc1"),
    
    
    (182, 1280, 5120, 32, 64, 4, 1, "fc2"), (364, 1280, 5120, 64, 64, 4, 1, "fc2"),
    (533, 1280, 5120, 32, 128, 4, 1, "fc2"), (1079, 1280, 5120, 64, 128, 2, 1, "fc2"),
    
    
    (182, 1280, 7680, 32, 64, 4, 1, "conv_out"), (364, 1280, 7680, 64, 64, 4, 1, "conv_out"),
    (533, 1280, 7680, 32, 128, 4, 1, "conv_out"), (1079, 1280, 7680, 64, 128, 2, 1, "conv_out"),
    (182, 2048, 1280, 64, 64, 4, 1, "proj2"), (364, 2048, 1280, 64, 128, 2, 1, "proj2"),
    (533, 2048, 1280, 128, 64, 2, 1, "proj2"), (1079, 2048, 1280, 128, 128, 2, 1, "proj2"),
    (160, 2560, 2048, 64, 64, 4, 1, "qkv_tp2"), (288, 2560, 2048, 64, 128, 2, 1, "qkv_tp2"),
    (576, 2560, 2048, 96, 128, 2, 1, "qkv_tp2"), (768, 2560, 2048, 96, 128, 2, 1, "qkv_tp2"),
    (1024, 2560, 2048, 128, 128, 2, 1, "qkv_tp2"), (2048, 2560, 2048, 128, 128, 2, 1, "qkv_tp2"),
    (144, 2048, 2048, 32, 64, 4, 1, "o_tp2"), (160, 2048, 2048, 32, 64, 4, 1, "o_tp2"),
    (288, 2048, 2048, 64, 64, 4, 1, "o_tp2"), (320, 2048, 2048, 64, 64, 4, 1, "o_tp2"),
    (512, 2048, 2048, 128, 64, 2, 1, "o_tp2"), (768, 2048, 2048, 96, 128, 2, 1, "o_tp2"),
    (1536, 2048, 2048, 128, 128, 2, 1, "o_tp2"), (160, 6144, 2560, 64, 128, 2, 1, "A_wqkv"),
    (192, 6144, 2560, 64, 128, 2, 1, "A_wqkv"), (256, 6144, 2560, 96, 128, 2, 1, "A_wqkv"),
    (512, 6144, 2560, 96, 128, 2, 1, "A_wqkv"), (1024, 6144, 2560, 160, 128, 2, 1, "A_wqkv"),
    (160, 2560, 4096, 64, 64, 4, 1, "A_wo"), (256, 2560, 4096, 64, 64, 4, 1, "A_wo"),
    (512, 2560, 4096, 128, 64, 2, 1, "A_wo"), (1024, 2560, 4096, 128, 128, 2, 1, "A_wo"),
    (160, 19456, 2560, 160, 128, 2, 1, "A_gate_up"), (256, 19456, 2560, 128, 128, 2, 1, "A_gate_up"),
    (1024, 19456, 2560, 128, 128, 2, 1, "A_gate_up"), (160, 2560, 9728, 128, 128, 2, 4, "A_down"),
    (256, 2560, 9728, 128, 128, 2, 4, "A_down"), (512, 2560, 9728, 128, 128, 2, 2, "A_down"),
    (160, 5120, 2048, 64, 128, 2, 1, "B_wqkv"), (192, 5120, 2048, 64, 128, 2, 1, "B_wqkv"),
    (256, 5120, 2048, 64, 128, 2, 1, "B_wqkv"), (512, 5120, 2048, 96, 128, 2, 1, "B_wqkv"),
    (160, 2048, 4096, 32, 64, 4, 1, "BC_wo"), (256, 2048, 4096, 64, 64, 4, 1, "BC_wo"),
    (512, 2048, 4096, 128, 64, 2, 1, "BC_wo"), (160, 9216, 2048, 96, 128, 2, 1, "C_wqkv_gated"),
    (192, 9216, 2048, 96, 128, 2, 1, "C_wqkv_gated"), (256, 9216, 2048, 128, 128, 2, 1, "C_wqkv_gated"),
    (160, 12288, 2048, 160, 128, 2, 1, "C_gdn_in"), (192, 12288, 2048, 96, 128, 2, 1, "C_gdn_in"),
    (256, 12288, 2048, 128, 128, 2, 1, "C_gdn_in"), (1024, 12288, 2048, 128, 128, 2, 1, "C_gdn_in"),
    (160, 1024, 2048, 32, 64, 4, 1, "C_shared_gate_up"), (192, 1024, 2048, 64, 64, 4, 1, "C_shared_gate_up"),
    (256, 1024, 2048, 64, 64, 4, 1, "C_shared_gate_up"), (512, 1024, 2048, 64, 64, 4, 1, "C_shared_gate_up"),
    (1024, 1024, 2048, 128, 64, 2, 1, "C_shared_gate_up"), (160, 2048, 512, 32, 64, 4, 1, "C_shared_down"),
    (256, 2048, 512, 64, 64, 4, 1, "C_shared_down"), (512, 2048, 512, 128, 64, 2, 1, "C_shared_down"),
)
# (M, N, K, bias, residual, activation, rule): cells the rules move on the RTX 5090; the bytes checks need only that the
# forced tiles run, so they run on any sm_120 device
A_CELLS = ((364, 1280, 1280, True, False, None, "a"), (1079, 1280, 5120, True, True, None, "a"),
           (182, 1280, 1280, True, False, "gelu_erf", "a"), (160, 2048, 4096, False, False, None, "a"))
failures: list[str] = []


def check(ok: bool, msg: str) -> bool:
    if not ok:
        failures.append(msg)
    return ok


def ctas(t, M, N):
    return -(-M // t[0]) * -(-N // t[1])


def cost(t, M, N, sms):
    return -(-ctas(t, M, N) // sms) * (t[0] + t[1])


def route_of(M, N, K):
    import torch
    r = [int(v) for v in torch.ops.fish_scales_ops.mxfp8_dense_route_sm120(M, N, K)]
    return (r[0], r[1], r[2]), r[5], bool(r[6])


def case_routes(sms: int) -> None:
    n_inv = 0
    for N in (1024, 1280, 2048, 2560, 4096, 5120, 6144, 9216, 12288, 19456):
        for K in (512, 1280, 2048, 4096, 5120, 7680, 9728):
            for M in (65, 96, 128, 129, 144, 160, 182, 192, 256, 288, 364, 512, 533, 768, 1079, 1536, 2048, 4096):
                t, ks, smallm = route_of(M, N, K)
                n_inv += 1
                check(t in TILES, f"routes: M={M} N={N} K={K} route {t} is not one of the cascade's tiles")
                if M > 128 and ks <= 1 and not smallm and t in TILES and ctas(t, M, N) < sms:
                    best = min(cost(u, M, N, sms) for u in TILES if u[0] != 32)
                    check(cost(t, M, N, sms) <= best,
                          f"routes: M={M} N={N} K={K} under-filled route {t} ({ctas(t, M, N)} CTAs < {sms} SMs) has "
                          f"waves x (TM + TN) {cost(t, M, N, sms)} > {best} of a tile with more than 32 rows")
    print(f"  1  {n_inv} cells: every route is a cascade tile, and the under-filled single launches above M = 128 have "
          f"the smallest waves x (TM + TN) among the tiles with more than 32 rows  {'OK' if not failures else 'FAILED'}")
    if sms != SMS_TABLE:
        print(f"  1  the route table is for {SMS_TABLE} SMs; this device has {sms}: table not checked")
        return
    bad = []
    for M, N, K, tm, tn, st, ks, tag in W_EXPECTED:
        t, k, _ = route_of(M, N, K)
        if (t, k) != ((tm, tn, st), ks):
            bad.append((tag, M, N, K, (t, k), ((tm, tn, st), ks)))
    check(not bad, f"routes: {len(bad)} of {len(W_EXPECTED)} table cells differ, e.g. {bad[:4]}")
    print(f"  1  the route table at {SMS_TABLE} SMs: {len(W_EXPECTED) - len(bad)} of {len(W_EXPECTED)} cells match  "
          f"{'OK' if not bad else 'FAILED'}")


def worker(spec: str) -> int:
    import torch
    import fish_scales_ops as fso
    c = json.loads(spec)
    M, N, K = c["M"], c["N"], c["K"]
    torch.manual_seed(M * 1009 + N * 17 + K)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.5
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.5)
    b = torch.randn(N, dtype=torch.bfloat16, device="cuda") * 0.1 if c["bias"] else None
    res = torch.randn(M, N, dtype=torch.bfloat16, device="cuda") * 0.5 if c["residual"] else None
    dw = fso.dense.prepare_weight(w, format="mxfp8")
    xq, sx = fso.dense.quantize_activation(x)
    run2 = lambda: fso.dense.linear(xq, dw, x_scale=sx, bias=b, residual=res, activation=c["act"])  # noqa: E731
    y2 = run2()
    y2b = run2()
    yx = fso.dense.linear(x, dw, bias=b, residual=res, activation=c["act"])
    torch.cuda.synchronize()
    ref = torch.nn.functional.linear(x.float(), w.float(), None if b is None else b.float())
    if c["act"] == "gelu_erf":
        ref = torch.nn.functional.gelu(ref, approximate="none")
    if res is not None:
        ref = ref + res.float()
    cos = float(torch.nn.functional.cosine_similarity(y2.float().flatten(), ref.flatten(), dim=0))
    md5 = lambda t: hashlib.md5(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()  # noqa: E731
    t, ks, _ = route_of(M, N, K)
    out = c.get("save")
    if out:
        torch.save(y2.cpu(), out)
    print("WF_RESULT " + json.dumps({"md5_2b": md5(y2), "md5_2b_again": md5(y2b), "md5_x": md5(yx), "route": list(t),
                                     "k_split": ks, "cos": cos}))
    return 0


def run_worker(spec: dict, env_add: dict) -> dict | None:
    env = {k: v for k, v in os.environ.items() if k not in KNOBS}
    env.update(env_add)
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", json.dumps(spec)], env=env,
                       capture_output=True, text=True, timeout=600)
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("WF_RESULT ")]
    if p.returncode != 0 or not line:
        check(False, f"worker {spec} {env_add} failed rc {p.returncode}: {p.stderr[-800:]}")
        return None
    return json.loads(line[0][len("WF_RESULT "):])


def case_bytes() -> None:
    for M, N, K, hb, hr, act, rule in A_CELLS:
        spec = {"M": M, "N": N, "K": K, "bias": hb, "residual": hr, "act": act}
        nat = run_worker(spec, {})
        if nat is None:
            continue
        forced = {}
        for f in ("128,128,2", "64,64,4", "32,128,4"):
            o = run_worker(spec, {"FSO_FORCE_TILE": f})
            if o is not None:
                forced[f] = o
                check(o["md5_2b"] == nat["md5_2b"] and o["md5_x"] == nat["md5_x"],
                      f"bytes (a): M={M} N={N} K={K}: forced {f} differs from the natural route {nat['route']}")
        check(nat["md5_2b"] == nat["md5_x"] and nat["md5_2b"] == nat["md5_2b_again"],
              f"bytes (a): M={M} N={N} K={K}: the bf16-x call differs from the 2b call, or two calls differ")
        print(f"  2  (a) M={M} N={N} K={K} bias={hb} residual={hr} act={act}: route {tuple(nat['route'])}; forced "
              f"{sorted(forced)} give its bytes on both calls; bf16-x == 2b  {'OK' if not failures else 'FAILED'}")


def case_switches() -> None:
    import torch
    M, N, K = 1079, 1280, 1280
    o = run_worker({"M": M, "N": N, "K": K, "bias": False, "residual": False, "act": None}, {"FSO_DISABLE_OVERRIDES": "1"})
    nat, _, _ = route_of(M, N, K)
    if o is not None:
        check(tuple(o["route"]) != nat, f"switch: FSO_DISABLE_OVERRIDES=1 kept the rule's tile {nat} at M={M} N={N} K={K}")
        print(f"  3  FSO_DISABLE_OVERRIDES=1 at M={M} N={N} K={K}: route {tuple(o['route'])} instead of the rule's {nat}  "
              f"{'OK' if not failures else 'FAILED'}")
    del torch


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        return worker(sys.argv[2])
    import torch
    import fish_scales_ops  # registers the ops
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        print("SKIP: the wave-fill tile rule is the sm_120/sm_121 MXFP8 cascade's")
        return 0
    if any(os.environ.get(k) for k in KNOBS):
        print(f"SKIP: a cascade knob is set ({[k for k in KNOBS if os.environ.get(k)]}); run with the defaults")
        return 0
    sms = int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
    print(f"== sm_120 MXFP8 cascade, the wave-fill tile rule ({torch.cuda.get_device_name()}, {sms} SMs, "
          f"fish_scales_ops {fish_scales_ops.__version__}) ==")
    case_routes(sms)
    case_bytes()
    case_switches()
    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for f in failures:
            print("  - " + f)
        return 1
    print("\nsm120 wave-fill tile rule: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
