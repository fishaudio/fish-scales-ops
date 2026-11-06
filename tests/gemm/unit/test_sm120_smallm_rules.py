"""The route rules and the configuration of the sm_120/sm_121 small-M dense MXFP8 route, on the host.

The rules R1 (the 2b call) and R2 (the bf16-x call) and the kernel configuration of ``sm120.smallm``
(csrc/gemm/include/blockscale_gemm/arch/sm120/mxfp8/smallm_route.h) are host arithmetic in (M, N, K). The extension
exposes them as ``torch.ops.fish_scales_ops.sm120_smallm_rules(M, N, K, sms)`` -> ``[takes_2b, takes_bf16x, r1, r2,
nc, rt, wk, cj, smem]`` for a device of ``sms`` multiprocessors (R1's one-round plan guard depends on it) without
touching a device, so this file runs on any machine that can import the extension (``CUDA_VISIBLE_DEVICES=""`` is
fine); it uses the RTX 5090's 170 SMs, and a smaller device's 84 for the dependence on ``sms``.

1. **The formulas.** The op agrees with this file's transcription of the two rules and of the configuration policy on
   a dense (M, N, K) grid (every M 1..64, N in 1024..19456, K in 512..9728), at 170 and at 84 SMs.
2. **Invariants.** R2 implies R1; nothing above M = 64, below M = 1, at K % 128 != 0 or N % 32 != 0 is served (every
   field 0); every served shape has a configuration that fits the opt-in shared memory (99 KB minus the margin), so
   a rule that says "route" is never undone by the plan; nc covers M; the output tiles divide N; the reduction order
   (wk) is a function of the shape alone.
3. **The band the rules serve on the issue-12 decode shapes.** For the twelve Q120a shapes the largest M each rule
   routes at 170 SMs is the fitted table (the rules were refitted on library timings of this kernel on an RTX 5090 in
   issue #12 Q120d; a change of a budget, a guard or the formula must update this table on purpose).
4. **The Python helper.** ``fish_scales_ops.dense._sm120_route`` names ``"sm120.smallm"`` exactly where the op says
   the call takes the route, ``"sm120.mxfp8"`` elsewhere, and both keys are served routes of ``_epilogue.SERVED``.

Run: ``PYTHONPATH=python python tests/gemm/unit/test_sm120_smallm_rules.py``.
"""
from __future__ import annotations

import math
import os
import sys

import torch

from fish_scales_ops import dense  # importing the package loads the extension and registers the ops
from fish_scales_ops.gemm import _epilogue

OP = torch.ops.fish_scales_ops.sm120_smallm_rules

K_MAX_M = 64
SMEM_MAX = 101376
SMEM_MARGIN = 1024
RED_STRIDE = 20
SMS = 170                                   # the RTX 5090's SM count, the device the rules were fitted on
BUDGET_R1, N0_R1 = 73728.0, 16384.0
R1_RESIDENT_ONLY, R1_NC_MAX = True, 5       # R1's plan guards (smallm_route.h kR1ResidentOnly, kR1NcMax)
R1_ONE_ROUND = True                         # R1's plan guard kR1OneRound: N / (16 rt) tile groups <= sms
BUDGET_R2, N0_R2 = 65536.0, 24576.0
R2_TILE_GROUPS0 = 96.0                      # R2's budget grows in proportion to the tile groups above this many (F5A)
R2_KEXP_QUARTERS, R2_K_MAX = 1, 1 << 30     # q = 1/4; no K cap

failures: list[str] = []


def check(ok: bool, msg: str) -> bool:
    if not ok:
        failures.append(msg)
    return ok


# ---------------------------------------------------------------------------------------------- the transcription
def shape_served(M: int, N: int, K: int) -> bool:
    return 1 <= M <= K_MAX_M and K > 0 and K % 128 == 0 and N > 0 and N % 32 == 0


def rule_r1(M: int, N: int, K: int, sms: int = SMS) -> bool:
    if not 1 <= M <= K_MAX_M:
        return False
    if float((M + 7) // 8 * 8) * K * (1.0 + N / N0_R1) > BUDGET_R1:
        return False
    if R1_RESIDENT_ONLY or R1_NC_MAX < 8 or R1_ONE_ROUND:
        nc, rt, _wk, cj, _smem = config(M, N, K)
        if R1_RESIDENT_ONLY and cj != 0:
            return False
        if nc > R1_NC_MAX:
            return False
        if R1_ONE_ROUND and N // (16 * rt) > sms:
            return False
    return True


def r2_kq(K: int) -> float:
    r = K / 2048.0
    return 1.0 if R2_KEXP_QUARTERS == 0 else (math.sqrt(math.sqrt(r)) if R2_KEXP_QUARTERS == 1 else math.sqrt(r))


def r2_cost(m: int, N: int, K: int) -> float:
    return float(m) * K * (1.0 + N / N0_R2) * r2_kq(K)


def r2_budget(M: int, N: int, K: int) -> float:
    """BUDGET_R2 * max(1, T / R2_TILE_GROUPS0), T = N / (16 rt) the tile groups of the shape's configuration."""
    _nc, rt, _wk, _cj, _smem = config(M, N, K)
    t = float(N // (16 * rt))
    return BUDGET_R2 * (t / R2_TILE_GROUPS0) if t > R2_TILE_GROUPS0 else BUDGET_R2


def rule_r2(M: int, N: int, K: int, sms: int = SMS) -> bool:
    if not 1 <= M <= K_MAX_M or K > R2_K_MAX:
        return False
    b = r2_budget(M, N, K)
    if r2_cost(M, N, K) > b:
        return False
    if M > 8 and r2_cost((M + 7) // 8 * 8, N, K) > b:
        return False
    return rule_r1(M, N, K, sms)


def smem_total(nc: int, rt: int, wk: int, K: int, cj: int) -> int:
    nb = K // 32
    cb = nb if cj == 0 else min(nb, cj * wk)
    rs = cb * 32 + 32
    xq = 8 * nc * rs
    xb = (8 * nc * cb + 15) // 16 * 16
    red = wk * rt * 8 * nc * RED_STRIDE * 4
    return xq + xb + red if cj == 0 else max(xq + xb, red)


def fits(nc: int, rt: int, wk: int, K: int, cj: int = 0) -> bool:
    return smem_total(nc, rt, wk, K, cj) <= SMEM_MAX - SMEM_MARGIN


def config(M: int, N: int, K: int) -> tuple[int, int, int, int, int]:
    """(nc, rt, wk, cj, smem) as smallm_route.h config()."""
    nc0 = (M + 7) // 8
    nb = K // 32
    short_k = nb // 8 <= 4
    nc = 2 if (nc0 == 1 and N >= 4096) else nc0
    order = [(8, 2), (4, 2), (8, 1), (4, 1)] if short_k else [(8, 1), (4, 1)]
    for wk, rt in order:
        if fits(nc, rt, wk, K, 0):
            return nc, rt, wk, 0, smem_total(nc, rt, wk, K, 0)
    wk, rt = 8, (2 if short_k else 1)
    if nc > nc0 and fits(nc0, rt, wk, K, 0):
        return nc0, rt, wk, 0, smem_total(nc0, rt, wk, K, 0)
    nbw = (nb + wk - 1) // wk
    cj = max(c for c in range(1, nbw + 1) if fits(nc, rt, wk, K, c))
    return nc, rt, wk, cj, smem_total(nc, rt, wk, K, cj)


def expected(M: int, N: int, K: int, sms: int = SMS) -> list[int]:
    if not shape_served(M, N, K):
        return [0] * 9
    nc, rt, wk, cj, smem = config(M, N, K)
    r1, r2 = rule_r1(M, N, K, sms), rule_r2(M, N, K, sms)
    return [int(r1), int(r2), int(r1), int(r2), nc, rt, wk, cj, smem]


# ------------------------------------------------------------------------------------------------------ the cases
Q120A_SHAPES = {  # tag: (N, K), the twelve Q120a decode shapes (issue-12 LLM roles at TP1 / TP2 and Family A / C)
    "llm_qkv_tp2": (2560, 2048), "llm_o_tp2": (2048, 2048), "llm_qkv": (5120, 2048), "llm_o": (2048, 4096),
    "A_wqkv": (6144, 2560), "A_wo": (2560, 4096), "A_gate_up": (19456, 2560), "A_down": (2560, 9728),
    "C_wqkv_gated": (9216, 2048), "C_gdn_in_proj": (12288, 2048), "C_shared_gate_up": (1024, 2048),
    "C_shared_down": (2048, 512),
}
# The fitted band at the RTX 5090's 170 SMs: (largest M R1 routes, largest M R2 routes) over every M = 1..64, with no
# hole below the top (the refit on library timings of issue #12 Q120d, runs/Q120d/refit, with R2's tile-group term of
# issue #12 F5A, runs/F5A/refit; the table is the refit tool's own rule applied to every M, not the extension's).
# Shapes whose tile groups need more than one round of the persistent grid on 170 SMs are never routed (0, 0).
Q120A_BAND = {
    "C_shared_down": (40, 40), "C_shared_gate_up": (32, 24), "llm_o_tp2": (32, 32), "llm_qkv_tp2": (24, 24),
    "llm_o": (16, 16), "A_wo": (8, 8), "A_down": (0, 0), "A_gate_up": (0, 0), "A_wqkv": (0, 0),
    "C_gdn_in_proj": (0, 0), "C_wqkv_gated": (0, 0), "llm_qkv": (0, 0),
}


def case_formulas() -> None:
    ns = (1024, 1280, 2048, 2560, 4096, 5120, 6144, 9216, 12288, 19456)
    ks = (512, 1024, 1280, 2048, 2560, 3072, 4096, 5120, 7680, 9728)
    n_cells = 0
    bad = []
    for sms in (SMS, 84):
        for K in ks:
            for N in ns:
                for M in range(0, 70):
                    got = [int(v) for v in OP(M, N, K, sms)]
                    want = expected(M, N, K, sms)
                    n_cells += 1
                    if got != want:
                        bad.append((M, N, K, sms, got, want))
    check(not bad, f"formulas: {len(bad)} of {n_cells} cells differ from the transcription, e.g. {bad[:5]}")
    print(f"  1  the op == the transcription of R1 / R2 / config() on {n_cells} (M, N, K, sms) cells at {SMS} and 84 "
          f"SMs  {'OK' if not bad else 'FAILED'}")


def case_invariants() -> None:
    ns = (1024, 2048, 2560, 5120, 9216, 19456)
    ks = (512, 2048, 2560, 4096, 9728)
    n_served = 0
    for K in ks:
        for N in ns:
            for M in range(1, 65):
                v = [int(x) for x in OP(M, N, K, SMS)]
                takes_2b, takes_x, r1, r2, nc, rt, wk, cj, smem = v
                n_served += 1
                check(not r2 or r1, f"invariant: R2 without R1 at M={M} N={N} K={K}")
                check(takes_2b == r1 and takes_x == r2,
                      f"invariant: the plan does not follow the rules at M={M} N={N} K={K}: {v} (a rule said route "
                      f"but no configuration fits?)")
                check(0 < smem <= SMEM_MAX - SMEM_MARGIN, f"invariant: smem {smem} out of range at M={M} N={N} K={K}")
                check(8 * nc >= M and 1 <= nc <= 8, f"invariant: nc {nc} does not cover M={M} (N={N} K={K})")
                check(rt in (1, 2) and wk in (4, 8) and N % (16 * rt) == 0,
                      f"invariant: rt {rt} / wk {wk} at M={M} N={N} K={K}")
                check(cj == 0 or (1 <= cj <= (K // 32 + wk - 1) // wk),
                      f"invariant: cj {cj} out of range at M={M} N={N} K={K}")
                # the reduction order is the shape's alone: the two forms (same shape) share wk by construction;
                # here: wk is identical for the 2b and the bf16-x call because the op has no form argument.
    for M, N, K in ((0, 2048, 2048), (65, 2048, 2048), (100, 2048, 2048), (8, 2048, 2000), (8, 2048, 0),
                    (8, 1000, 2048), (8, 0, 2048)):
        v = [int(x) for x in OP(M, N, K, SMS)]
        check(v == [0] * 9, f"invariant: an unserved shape M={M} N={N} K={K} is not all-zero: {v}")
    print(f"  2  R2 => R1, plan == rules, smem fits, nc covers M, tiles divide N on {n_served} served cells; 7 "
          f"unserved shapes all-zero  {'OK' if not failures else 'FAILED'}")


def case_band() -> None:
    lines = []
    for tag, (N, K) in Q120A_SHAPES.items():
        top1 = max((M for M in range(1, 65) if int(OP(M, N, K, SMS)[0])), default=0)
        top2 = max((M for M in range(1, 65) if int(OP(M, N, K, SMS)[1])), default=0)
        want1, want2 = Q120A_BAND[tag]
        ok = (top1, top2) == (want1, want2)
        check(ok, f"band: {tag} N={N} K={K}: R1 up to M={top1}, R2 up to M={top2}; fitted {want1} / {want2}")
        # the band is a prefix: no hole below the top
        holes1 = [M for M in range(1, top1 + 1) if not int(OP(M, N, K, SMS)[0])]
        holes2 = [M for M in range(1, top2 + 1) if not int(OP(M, N, K, SMS)[1])]
        check(not holes1 and not holes2, f"band: {tag}: holes below the top: R1 {holes1}, R2 {holes2}")
        lines.append(f"{tag:16s} N={N:5d} K={K:4d}: R1 M<={top1:2d}, R2 M<={top2:2d} {'OK' if ok else 'FAILED'}")
    print("  3  the fitted band on the twelve Q120a shapes:")
    for ln in lines:
        print("      " + ln)


def case_python_helper() -> None:
    n = 0
    for tag, (N, K) in Q120A_SHAPES.items():
        for M in (1, 4, 8, 16, 24, 32, 40, 48, 64):
            v = OP(M, N, K, SMS)
            for bf16x in (False, True):
                key = dense._sm120_route(M, N, K, bf16x, sms=SMS)
                want = "sm120.smallm" if int(v[1 if bf16x else 0]) else "sm120.mxfp8"
                check(key == want, f"helper: _sm120_route({M}, {N}, {K}, bf16x={bf16x}) = {key}, op says {want}")
                check(key in _epilogue.SERVED and key in _epilogue.ROUTE_LABELS, f"helper: {key} not a labelled route")
                n += 1
    check(_epilogue.SERVED["sm120.smallm"] == frozenset({"bias", "residual", "activation"}),
          f"helper: sm120.smallm serves {sorted(_epilogue.SERVED['sm120.smallm'])}")
    print(f"  4  dense._sm120_route follows the op on {n} calls; sm120.smallm serves bias / residual / activation  "
          f"{'OK' if not failures else 'FAILED'}")


def main() -> int:
    mode = os.environ.get("FSO_SMALLM_ROUTE")
    knobs = [k for k in ("FSO_FORCE_TILE", "FSO_FORCE_KSPLIT", "FSO_FORCE_SMALLM", "FSO_FORCE_MIN_BLOCKS",
                         "FSO_FORCE_SCHED_GROUP") if os.environ.get(k)]
    if (mode not in (None, "", "rules")) or knobs:
        print(f"SKIP: FSO_SMALLM_ROUTE={mode!r} / force knobs {knobs} change what the op reports; run with the defaults")
        return 0
    print(f"torch {torch.__version__}; CUDA visible: {torch.cuda.is_available()} (not needed)")
    print("== sm120.smallm: the rules R1 / R2 and the configuration, host arithmetic ==")
    case_formulas()
    case_invariants()
    case_band()
    case_python_helper()
    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for f in failures:
            print("  - " + f)
        return 1
    print("\nsm120.smallm rules: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
