#!/usr/bin/env python3
"""sm_90 fused swap-AB FC1 (fused-FC1 P2a): the swap-AB FC1 with the SwiGLU + 1x128 FP8 requantize in its epilogue
must equal the unfused chain bit for bit, and so must its bf16 mainloop vehicle.

The unfused swap-AB chain is the FC1 (linear_fp8_grouped_contiguous_swapab, bf16 gu [P_max, 2I]) followed by
silu_chunk_mul_quantize_1x128_sorted_sm90 (dq [P_max, I] fp8, sd [I/128, align4(P_max)] fp32). The fused op
(fish_scales_ops.gemm.fp8.linear_fp8_grouped_contiguous_swapab_swiglu) produces dq and sd directly, and
moe_layer_fp8_sm90 uses it on the swap-AB route unless FSO_FC1_FUSED=0. Its mainloop vehicle,
linear_fp8_grouped_contiguous_swapab_pair, runs the same mainloop (one CTA owns gate block b and up block b) with a
bf16 epilogue into gu. For Family B (Qwen3-30B-A3B routed experts: E=128, top-8, H=2048, I=768) and Family C
(Qwen3.5-35B-A3B: E=256, top-8, H=2048, I=512), with about one routed entry in ten masked to -1 or E, this test checks:

  1. Op level, for block_n 16, 32 and 64, M in {1, 8, 64, 256, 1024, 2048}, and the CTA mode the dispatch picks as
     well as one and two CTAs per SM forced through FSO_SWAPAB_CTAS_PER_SM: the fused op's dq bytes and sd floats
     (compared as int32 bit patterns) equal the unfused pair's on every routed row, i.e. every sorted row
     flat_to_sorted assigns, and the pair op's gu equals the swap-AB FC1's gu on every routed row. The unfused SwiGLU
     kernel writes only routed rows; the fused kernel also writes the padding rows of every tile it visits, whose
     inputs the gather leaves uninitialised, so padding rows are not compared (FC2 is row-independent and the combine
     reads routed rows only).
  2. Layer level, under FSO_SWAP_BN=16, 32 and 64 and under the default dispatch, M in {1, 8, 64, 256, 1024, 2048}.
     In this process (FSO_FC1_FUSED unset) the layer output must equal the explicit unfused chain built from the
     public ops. A child process with FSO_FC1_FUSED=0 (the knob is read once per process) checks that the unfused
     layer equals the same chain and reports a sha256 of every cell's output; the fused layer's hashes must match.
  3. Determinism. For block_n 16, 32 and 64 (forced), twenty eager calls of the fused layer are bitwise identical and
     a CUDA-graph replay equals the eager result; one graph is also replayed after its routing, hidden states and
     masked entries change.

Script style: run as `python tests/gemm/unit/test_fp8_fused_fc1_swapab_sm90.py` on an H200. `--quick` restricts every
part to M in {1, 8} and block_n 16 (the decode cells).
"""
import hashlib
import json
import math
import os
import subprocess
import sys

import torch

import fish_scales_ops as fso
from fish_scales_ops.gemm.fp8 import (
    _sm90_layer_plan,
    linear_fp8_grouped_contiguous_swapab_pair,
    linear_fp8_grouped_contiguous_swapab_swiglu,
)

FAMILIES = {
    # (E, topk, hidden, inter)
    "B": (128, 8, 2048, 768),
    "C": (256, 8, 2048, 512),
}
QUICK = "--quick" in sys.argv
MS = (1, 8) if QUICK else (1, 8, 64, 256, 1024, 2048)
BNS = (16,) if QUICK else (16, 32, 64)
CTA_MODES = (None, "1", "2")  # None: the dispatch's own plan
LAYER_MODES = ("bn16",) if QUICK else ("bn16", "bn32", "bn64", "default")
DET_RUNS = 20


def set_env(name, value):
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def set_mode(mode):
    set_env("FSO_SWAP_BN", None if mode == "default" else mode[2:])


def make_topk_ids(M, E, topk, gen):
    """Random routing without replacement per token, with about 10 % of the entries masked to -1 or E (at least one
    of each: entry (0, 0) gets -1 and entry (M-1, topk-1) gets E)."""
    ids = torch.rand(M, E, device="cuda", generator=gen).topk(topk, dim=1).indices.to(torch.int32)
    mask = torch.rand(M, topk, device="cuda", generator=gen) < 0.1
    mask[0, 0] = True
    mask[M - 1, topk - 1] = True
    flat = torch.arange(M * topk, device="cuda").view(M, topk)
    fill = torch.where(flat % 2 == 0, -1, E).to(torch.int32)
    fill[M - 1, topk - 1] = E
    return torch.where(mask, fill, ids).contiguous()


def family_weights(fam):
    E, _, H, I = FAMILIES[fam]
    gen = torch.Generator(device="cuda")
    gen.manual_seed(20261001 + E)
    w13 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16, generator=gen) / math.sqrt(H)
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16, generator=gen) / math.sqrt(I)
    w13q, sw13 = fso.compat.quantize_moe_weights_1x128_fp8_sm90(w13)
    w2q, sw2 = fso.compat.quantize_moe_weights_1x128_fp8_sm90(w2)
    return w13q, sw13, w2q, sw2


def cell_inputs(fam, M, salt=0):
    E, TOPK, H, _ = FAMILIES[fam]
    gen = torch.Generator(device="cuda")
    gen.manual_seed(7919 * M + E + 104729 * salt + 17)
    hidden = torch.randn(M, H, device="cuda", dtype=torch.bfloat16, generator=gen)
    topk_ids = make_topk_ids(M, E, TOPK, gen)
    topk_w = torch.softmax(torch.rand(M, TOPK, device="cuda", generator=gen), dim=1).float().contiguous()
    return hidden, topk_ids, topk_w


def unfused_chain(hidden, w13q, sw13, w2q, sw2, topk_ids, topk_w):
    """The unfused layer, spelled out with the public ops and the layer's own host plan."""
    M = int(hidden.shape[0])
    E, H, I = int(w13q.shape[0]), int(w13q.shape[2]), int(w13q.shape[1]) // 2
    topk = int(topk_ids.shape[1])
    expected_m, use_swap, block, _, _ = _sm90_layer_plan(M, E, topk, H, I)
    g = fso.compat
    se, fts, _ = g.moe_build_sorted(topk_ids, E, block)
    hq, sh = g.quantize_1x128_sorted_gather_sm90(hidden, fts, int(se.shape[0]), topk)
    gemm = g.linear_fp8_grouped_contiguous_swapab if use_swap else g.linear_fp8_grouped_contiguous
    gu = gemm(hq, w13q, sh, sw13, se, block, expected_m)
    dq, sd = g.silu_chunk_mul_quantize_1x128_sorted_sm90(gu, fts)
    dn = gemm(dq, w2q, sd, sw2, se, block, expected_m)
    return g.moe_combine_sorted(dn, fts, topk_w)


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()


def op_level():
    """Part 1: the fused op against the swap-AB FC1 + SwiGLU kernel pair, and the pair op against the swap-AB FC1."""
    ops = torch.ops.fish_scales_ops
    passed = total = 0
    print("=== 1. op level: fused swap-AB FC1 (dq, sd) vs FC1 + silu_chunk_mul_quantize_1x128_sorted_sm90, and the "
          "pair op's gu vs the FC1's, routed rows ===")
    for fam, (E, TOPK, H, I) in FAMILIES.items():
        w13q, sw13, _, _ = family_weights(fam)
        for M in MS:
            hidden, topk_ids, _ = cell_inputs(fam, M)
            expected_m = max(1, (M * TOPK + E - 1) // E)
            for bn in BNS:
                set_env("FSO_SWAPAB_CTAS_PER_SM", None)
                se, fts, npad = ops.moe_build_sorted(topk_ids, E, bn)
                p_max = int(se.shape[0])
                hq, sh = ops.quantize_1x128_sorted_gather_sm90(hidden, fts, p_max, TOPK)
                gu = ops.linear_fp8_grouped_contiguous_swapab(hq, w13q, sh, sw13, se, bn, expected_m)
                dq_ref, sd_ref = ops.silu_chunk_mul_quantize_1x128_sorted_sm90(gu, fts)
                torch.cuda.synchronize()
                real = fts[fts >= 0].long()
                ref_b, ref_s = dq_ref[real].view(torch.uint8), sd_ref[:, real].view(torch.int32)
                # The compared rows must hold real data, so equality is not vacuous.
                sane = (real.numel() > 0 and bool(torch.isfinite(sd_ref[:, real]).all())
                        and bool((sd_ref[:, real] > 0).all()) and bool((ref_b != 0).any()))
                for mode in CTA_MODES:
                    set_env("FSO_SWAPAB_CTAS_PER_SM", mode)
                    dq, sd = linear_fp8_grouped_contiguous_swapab_swiglu(hq, w13q, sh, sw13, se, bn, expected_m)
                    gu2 = linear_fp8_grouped_contiguous_swapab_pair(hq, w13q, sh, sw13, se, bn, expected_m)
                    torch.cuda.synchronize()
                    set_env("FSO_SWAPAB_CTAS_PER_SM", None)
                    shapes = (tuple(dq.shape) == tuple(dq_ref.shape) == (p_max, I)
                              and tuple(sd.shape) == tuple(sd_ref.shape) == (I // 128, (p_max + 3) // 4 * 4))
                    ok_fused = shapes and torch.equal(dq[real].view(torch.uint8), ref_b) \
                        and torch.equal(sd[:, real].view(torch.int32), ref_s)
                    ok_pair = torch.equal(gu2[real].view(torch.int16), gu[real].view(torch.int16))
                    ok = sane and ok_fused and ok_pair
                    if shapes and not ok_fused:
                        bad_rows = int((dq[real].view(torch.uint8) != ref_b).any(dim=1).sum())
                        bad_sd = int((sd[:, real].view(torch.int32) != ref_s).sum())
                        print(f"    mismatch: {bad_rows} dq rows, {bad_sd} sd entries differ")
                    total += 1
                    passed += int(ok)
                    print(f"{fam} M={M:5d} bn={bn:2d} ctas={mode or 'plan'}: P_max={p_max:6d} "
                          f"padded={int(npad.item()):6d} routed rows={real.numel():6d}  fused "
                          f"{'==' if ok_fused else '!='} ref, pair {'==' if ok_pair else '!='} FC1  "
                          f"{'OK' if ok else 'FAIL'}")
                del gu
        del w13q, sw13
        torch.cuda.empty_cache()
    set_env("FSO_SWAPAB_CTAS_PER_SM", None)
    return passed, total


def layer_cells(expect_fused):
    """Part 2 body, run in both processes: every (mode, family, M) cell's layer output against the explicit unfused
    chain. Returns ({cell: sha256}, passed, total)."""
    hashes = {}
    passed = total = 0
    for fam, (E, TOPK, H, I) in FAMILIES.items():
        w13q, sw13, w2q, sw2 = family_weights(fam)
        for mode in LAYER_MODES:
            set_mode(mode)
            for M in MS:
                hidden, topk_ids, topk_w = cell_inputs(fam, M)
                _, use_swap, block, _, fused = _sm90_layer_plan(M, E, TOPK, H, I)
                y = fso.compat.moe_layer_fp8_sm90(hidden, w13q, sw13, w2q, sw2, topk_ids, topk_w)
                ref = unfused_chain(hidden, w13q, sw13, w2q, sw2, topk_ids, topk_w)
                torch.cuda.synchronize()
                route_ok = fused == expect_fused
                sane = bool(torch.isfinite(y.float()).all()) and bool((y != 0).any())
                ok = route_ok and sane and torch.equal(y.view(torch.int16), ref.view(torch.int16))
                hashes[f"{mode}/{fam}/{M}"] = digest(y)
                total += 1
                passed += int(ok)
                path = (f"swap-AB/{block}" if use_swap else "non-swap") + (" fused" if fused else " unfused")
                print(f"{mode:7s} {fam} M={M:5d} {path:20s}: layer {'==' if ok else '!='} explicit unfused chain  "
                      f"{'OK' if ok else 'FAIL'}")
        set_mode("default")
        del w13q, sw13, w2q, sw2
        torch.cuda.empty_cache()
    return hashes, passed, total


def determinism():
    """Part 3: twenty eager calls and graph replays of the fused swap-AB layer."""
    passed = total = 0
    print("\n=== 3. determinism of the fused swap-AB layer ===")
    for fam, (E, TOPK, H, I) in FAMILIES.items():
        w13q, sw13, w2q, sw2 = family_weights(fam)
        for bn in BNS:
            set_mode(f"bn{bn}")
            for M in ((1, 8) if QUICK else (8, 256)):
                hidden, topk_ids, topk_w = cell_inputs(fam, M, salt=1)
                _, use_swap, _, _, fused = _sm90_layer_plan(M, E, TOPK, H, I)
                fn = lambda: fso.compat.moe_layer_fp8_sm90(hidden, w13q, sw13, w2q, sw2, topk_ids, topk_w)
                ref = fn()
                torch.cuda.synchronize()
                bad = 0
                for _ in range(DET_RUNS):
                    y = fn()
                    torch.cuda.synchronize()
                    bad += int(not torch.equal(y.view(torch.int16), ref.view(torch.int16)))
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        fn()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=s):
                    out = fn()
                torch.cuda.synchronize()
                out.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                graph_ok = torch.equal(out.view(torch.int16), ref.view(torch.int16))
                ok = use_swap and fused and bad == 0 and graph_ok
                total += 1
                passed += int(ok)
                print(f"{fam} M={M:5d} bn={bn:2d} fused={fused}: {DET_RUNS} eager runs, {bad} differ; graph replay "
                      f"{'==' if graph_ok else '!='} eager  {'OK' if ok else 'FAIL'}")
                # One graph replayed after the routing, the hidden states and the masked entries changed.
                if bn == BNS[0] and M == 8:
                    h2, ids2, w2_ = cell_inputs(fam, M, salt=2)
                    hidden.copy_(h2)
                    topk_ids.copy_(ids2)
                    topk_w.copy_(w2_)
                    out.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    ref2 = unfused_chain(hidden, w13q, sw13, w2q, sw2, topk_ids, topk_w)
                    ok2 = torch.equal(out.view(torch.int16), ref2.view(torch.int16))
                    total += 1
                    passed += int(ok2)
                    print(f"{fam} M={M:5d} bn={bn:2d}: graph replay after rerouting == explicit unfused chain on the "
                          f"new inputs  {'OK' if ok2 else 'FAIL'}")
                del graph, out
        del w13q, sw13, w2q, sw2
        torch.cuda.empty_cache()
    set_mode("default")
    return passed, total


def unfused_worker():
    """Child process body (FSO_FC1_FUSED=0): the unfused layer against the explicit chain, and every cell's hash."""
    assert os.environ.get("FSO_FC1_FUSED") == "0"
    hashes, passed, total = layer_cells(expect_fused=False)
    print("HASHES " + json.dumps(dict(hashes=hashes, passed=passed, total=total)), flush=True)
    return 0 if passed == total else 1


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        where = "no CUDA device" if not torch.cuda.is_available() \
            else f"device is sm_{torch.cuda.get_device_capability()[0]}x"
        print(f"SKIP: test_fp8_fused_fc1_swapab_sm90 is sm_90 (H200) only ({where})")
        return 0
    if "--unfused-worker" in sys.argv:
        return unfused_worker()
    if os.environ.get("FSO_FC1_FUSED", "")[:1] == "0":
        print("SKIP: run with FSO_FC1_FUSED unset (the test starts its own FSO_FC1_FUSED=0 child)")
        return 0
    saved = {k: os.environ.get(k) for k in ("FSO_SWAP_BN", "FSO_SWAPAB_CTAS_PER_SM")}

    passed, total = op_level()

    print("\n=== 2. layer level, FSO_FC1_FUSED unset: fused layer == explicit unfused chain ===")
    fused_hashes, p, t = layer_cells(expect_fused=True)
    passed += p
    total += t

    print("\n=== 2b. child process, FSO_FC1_FUSED=0: unfused layer == explicit unfused chain ===")
    env = dict(os.environ)
    env["FSO_FC1_FUSED"] = "0"
    env.pop("FSO_SWAP_BN", None)
    env.pop("FSO_SWAPAB_CTAS_PER_SM", None)
    child = subprocess.run([sys.executable, os.path.abspath(__file__), "--unfused-worker"]
                           + (["--quick"] if QUICK else []), env=env, capture_output=True, text=True)
    sys.stdout.write("".join("  | " + l + "\n" for l in child.stdout.splitlines() if not l.startswith("HASHES ")))
    line = next((l for l in child.stdout.splitlines() if l.startswith("HASHES ")), None)
    total += 1
    if child.returncode != 0 or line is None:
        print(f"unfused child failed (exit {child.returncode}):\n{child.stderr[-3000:]}")
    else:
        passed += 1
        res = json.loads(line[len("HASHES "):])
        same = 0
        for key, h in fused_hashes.items():
            eq = res["hashes"].get(key) == h
            same += int(eq)
            if not eq:
                print(f"  {key}: fused layer hash differs from the FSO_FC1_FUSED=0 layer")
        total += 1
        passed += int(same == len(fused_hashes))
        print(f"fused layer (this process) vs FSO_FC1_FUSED=0 layer (child): {same}/{len(fused_hashes)} cells "
              f"bit-identical (sha256)  {'OK' if same == len(fused_hashes) else 'FAIL'}")

    p, t = determinism()
    passed += p
    total += t

    for k, v in saved.items():
        set_env(k, v)
    print(f"\n{passed}/{total} checks passed")
    if passed != total:
        print("sm_90 fused swap-AB FC1: FAIL")
        return 1
    print("sm_90 fused swap-AB FC1: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
