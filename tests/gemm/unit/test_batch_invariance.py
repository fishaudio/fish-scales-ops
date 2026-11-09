#!/usr/bin/env python3
"""FSO_BATCH_INVARIANT=1: a row's dense output does not depend on M (docs/api/dense.md, "Determinism and batch
invariance").

sm_90 (the guarantee of this release). Each mode runs in its own child process, because the extension reads the
switch once per process:

  1. switch on: for o 2560 x 4096, down 2560 x 9216, llm_o 2048 x 4096 and conv_out 1280 x 7680, the first M rows of
     one fixed activation give, at every M in {1, 8, 32, 56, 64, 72, 96, 117, 128, 512}, rows that are torch.equal to
     the same rows at M = 512, on the bf16-x call (fso.dense.linear(x, w)) and on the pre-quantized call
     (quantize_activation + linear with x_scale); and no call launches an in-cluster split-K kernel
     (fp8_gemm_kernel_swapAB_splitk of the decode band, fp8_gemm_kernel_swapAB_splitk2 above it).
  2. negative control, switch unset: the split-K routes take their bands (the one-warp-group route M <= 56 at
     K = 4096, M <= 64 at K = 9216; the two-warp-group route, issue #12 FH7, the cells of 56 < M <= 128 its rule takes:
     _splitk2 mirrors it on the H200's 132 SMs), and somewhere in the bands a row differs from the same row at
     M = 512, by a few units in the last place at most (the routes add their per-slice partial sums in rank order, not
     in K order). The cells outside the bands stay identical.
  3. the switch with FSO_SM90_DENSE_SPLITK=force: the switch wins (no split-K launch) and one line on stderr says so.

sm_100/103 and sm_120/121 (no guarantee in this release): with the switch set, the first dense call warns once
(one RuntimeWarning for two calls) and the outputs equal those of a process without the switch.

usage: PYTHONPATH=python python tests/gemm/unit/test_batch_invariance.py
"""
import json
import os
import subprocess
import sys
import tempfile

SHAPES = [("o", 2560, 4096), ("down", 2560, 9216), ("llm_o", 2048, 4096), ("conv_out", 1280, 7680)]
MS = [1, 8, 32, 56, 64, 72, 96, 117, 128, 512]
M_MAX = max(MS)
# Clusters of c CTAs the H200 runs at once at one CTA per SM (FH5, FH6, FH7 of the issue-12 round).
_CLUSTERS_132 = {2: 66, 3: 39, 4: 30, 5: 22, 6: 17, 7: 15, 8: 15}


def _splitk2(m, n, k, sms):
    """Whether the two-warp-group split-K route takes an M x N x K call on `sms` SMs (mirror of
    deep_gemm::jit::get_dense_splitk2_config on 132 SMs); None where the mirror does not know the device."""
    if not (56 < m <= 128 and k >= 4096):
        return False
    if sms != 132:
        return None
    best = None
    for wh in (False, True):
        tiles = -(-n // (128 if wh else 64))
        for s in range(8, 1, -1):
            if _CLUSTERS_132[s] < tiles or s * tiles > sms or 2 * s > k // 128:
                continue
            t = max(16, -(-m // 8) * 8) if wh else max(32, -(-m // 16) * 16)
            if best is None or s * tiles > best[3]:
                best = (wh, s, t, s * tiles)
            break
    if best is None or 2 * best[3] < sms:
        return False
    wh, s, t, _ = best
    return (k >= 8192 and m <= 104) if wh else (s >= 5 or (s >= 3 and m <= 104))


def worker(out_path, ms):
    import torch
    from torch.profiler import ProfilerActivity, profile
    import fish_scales_ops as fso

    res = {"kernels": {}}
    tensors = {}
    for name, n, k in SHAPES:
        g = torch.Generator(device="cuda")
        g.manual_seed(n * 17 + k)
        x = torch.randn(M_MAX, k, device="cuda", dtype=torch.bfloat16, generator=g) * 0.1
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) / k ** 0.5
        dw = fso.dense.prepare_weight(w, format="bsfp8")
        for m in ms:
            xm = x[:m].contiguous()
            y = fso.dense.linear(xm, dw)
            xq, sx = fso.dense.quantize_activation(xm)
            yq = fso.dense.linear(xq, dw, x_scale=sx)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fso.dense.linear(xm, dw)
                torch.cuda.synchronize()
            res["kernels"][f"{name},{m}"] = [e.name for e in prof.events()
                                            if e.device_type == torch.autograd.DeviceType.CUDA
                                            and "gemm" in e.name.lower()]
            tensors[f"{name},{m},bf16x"] = y.cpu()
            tensors[f"{name},{m},prequant"] = yq.cpu()
    torch.save(tensors, out_path)
    print("RESULT " + json.dumps(res), flush=True)


def warn_worker(out_path):
    """Non-sm_90: two dense calls; report the RuntimeWarnings naming the switch and save the outputs."""
    import warnings

    import torch
    import fish_scales_ops as fso

    name, n, k = SHAPES[0]
    g = torch.Generator(device="cuda")
    g.manual_seed(n * 17 + k)
    x = torch.randn(64, k, device="cuda", dtype=torch.bfloat16, generator=g) * 0.1
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) / k ** 0.5
    dw = fso.dense.prepare_weight(w, format="bsfp8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y1 = fso.dense.linear(x[:1].contiguous(), dw)
        y2 = fso.dense.linear(x, dw)
        torch.cuda.synchronize()
    hits = [str(c.message) for c in caught if "FSO_BATCH_INVARIANT" in str(c.message)]
    torch.save({"y1": y1.cpu(), "y64": y2.cpu()}, out_path)
    print("RESULT " + json.dumps({"warnings": hits}), flush=True)


def run(env_extra, out_path, mode="worker", ms=MS):
    env = {k_: v for k_, v in os.environ.items() if not k_.startswith("FSO_")}
    env.update(env_extra)
    args = [sys.executable, os.path.abspath(__file__), "--" + mode, out_path, ",".join(str(m) for m in ms)]
    p = subprocess.run(args, env=env, capture_output=True, text=True, timeout=3600)
    res = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[7:])
    if p.returncode != 0 or res is None:
        raise SystemExit(f"child ({env_extra}) failed, exit {p.returncode}:\n{p.stderr[-3000:]}")
    return res, p.stderr


def ulps(a, b):
    """Units in the last place between the bf16 elements of a and b that have the same sign (int16 views); elements of
    opposite signs (a sum that nearly cancels) are left out."""
    import torch
    same = (a.float() >= 0) == (b.float() >= 0)
    ai = a[same].view(torch.int16).to(torch.int32)
    bi = b[same].view(torch.int16).to(torch.int32)
    return (ai - bi).abs()


def main_sm90(torch, tmp):
    failures = []
    on, _ = run({"FSO_BATCH_INVARIANT": "1"}, os.path.join(tmp, "on.pt"))
    off, _ = run({}, os.path.join(tmp, "off.pt"))
    t_on = torch.load(os.path.join(tmp, "on.pt"))
    t_off = torch.load(os.path.join(tmp, "off.pt"))
    # 1. the switch on: every row identical at every M, and no split-K launch
    n_rows = 0
    for name, _, _ in SHAPES:
        for call in ("bf16x", "prequant"):
            ref = t_on[f"{name},{M_MAX},{call}"]
            for m in MS:
                y = t_on[f"{name},{m},{call}"]
                n_rows += m
                if not torch.equal(y, ref[:m]):
                    failures.append(f"switch on: {name} {call} M = {m}: rows differ from the same rows at M = {M_MAX}")
        for m in MS:
            if any("splitk" in kn for kn in on["kernels"][f"{name},{m}"]):
                failures.append(f"switch on: {name} M = {m} launched the split-K kernel: "
                                f"{on['kernels'][f'{name},{m}']}")
    print(f"[1] FSO_BATCH_INVARIANT=1: {n_rows} rows over {len(SHAPES)} shapes, {len(MS)} M and 2 calls are identical "
          f"to the same rows at M = {M_MAX}; no split-K launch")
    # 2. negative control: split-K in the bands, some row differs there by a few ulps at most, identical outside them
    total_diff, band_cells, worst, worst_abs, band2_cells = 0, 0, 0, 0.0, 0
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for name, n, k in SHAPES:
        band = 64 if k >= 8192 else 56
        for m in MS:
            split = any("splitk" in kn for kn in off["kernels"][f"{name},{m}"])
            split2 = any("splitk2" in kn for kn in off["kernels"][f"{name},{m}"])
            want2 = _splitk2(m, n, k, sms)
            if want2 is None:
                want2 = split2  # an unknown device: the launch decides which band the cell is in
            in_band = m <= band or want2
            band2_cells += want2
            if split != in_band or split2 != want2:
                failures.append(f"switch unset: {name} M = {m}: split-K launched = {split} (two-warp-group {split2}), "
                                f"expected {in_band} ({want2})")
            for call in ("bf16x", "prequant"):
                y, ref = t_off[f"{name},{m},{call}"], t_off[f"{name},{M_MAX},{call}"][:m]
                d = (y != ref)
                nd = int(d.sum())
                if not in_band and nd:
                    failures.append(f"switch unset: {name} {call} M = {m} (outside the bands): {nd} elements differ")
                if in_band:
                    band_cells += 1
                    total_diff += nd
                    if nd:
                        u = ulps(y[d], ref[d])
                        worst = max(worst, int(u.max()) if u.numel() else 0)
                        rel = float((y[d].float() - ref[d].float()).abs().max() / ref.float().abs().max())
                        worst_abs = max(worst_abs, rel)
                # the switch-on bytes are the bytes this process gives outside the bands
                if not in_band and not torch.equal(y, t_on[f"{name},{m},{call}"]):
                    failures.append(f"{name} {call} M = {m}: the switch changed bytes outside the bands")
    if total_diff == 0:
        failures.append("switch unset: no element in the split-K bands differs from M = 512 (the control shows nothing)")
    if sms == 132 and band2_cells == 0:
        failures.append("switch unset: no cell of the two-warp-group split-K route (the control does not reach it)")
    # one bf16 rounding of a partial sum: far below the output's scale (FH5: at most one unit in the last place of the
    # largest outputs; a few more units on values that nearly cancel)
    if worst_abs > 2.0 ** -6:
        failures.append(f"switch unset: an element in the band differs by {worst_abs:.2e} of the output's "
                        f"largest value")
    print(f"[2] switch unset: split-K takes the bands ({band2_cells} cells of the two-warp-group route); "
          f"{total_diff} elements over {band_cells} band cells differ from "
          f"the same rows at M = {M_MAX} (at most {worst} units in the last place, {worst_abs:.1e} of the largest "
          f"output); outside the band identical")
    # 3. the switch wins over FSO_SM90_DENSE_SPLITK=force
    forced, err = run({"FSO_BATCH_INVARIANT": "1", "FSO_SM90_DENSE_SPLITK": "force"},
                      os.path.join(tmp, "forced.pt"), ms=[8])
    t_forced = torch.load(os.path.join(tmp, "forced.pt"))
    if err.count("FSO_BATCH_INVARIANT=1 turns the sm_90 split-K route off") != 1:
        failures.append("FSO_BATCH_INVARIANT=1 with FSO_SM90_DENSE_SPLITK=force: no single line on stderr")
    for name, _, _ in SHAPES:
        if any("splitk" in kn for kn in forced["kernels"][f"{name},8"]):
            failures.append(f"FSO_SM90_DENSE_SPLITK=force overrode the switch on {name} M = 8")
        for call in ("bf16x", "prequant"):
            if not torch.equal(t_forced[f"{name},8,{call}"], t_on[f"{name},8,{call}"]):
                failures.append(f"FSO_SM90_DENSE_SPLITK=force changed {name} {call} M = 8 under the switch")
    print("[3] FSO_SM90_DENSE_SPLITK=force under the switch: one line on stderr, no split-K launch, the switch's bytes")
    return failures


def main_other(torch, tmp):
    failures = []
    on, _ = run({"FSO_BATCH_INVARIANT": "1"}, os.path.join(tmp, "w_on.pt"), mode="warn-worker")
    off, _ = run({}, os.path.join(tmp, "w_off.pt"), mode="warn-worker")
    if len(on["warnings"]) != 1:
        failures.append(f"switch set: {len(on['warnings'])} warnings naming FSO_BATCH_INVARIANT for two calls, want 1")
    if off["warnings"]:
        failures.append(f"switch unset: unexpected warnings {off['warnings']}")
    a, b = torch.load(os.path.join(tmp, "w_on.pt")), torch.load(os.path.join(tmp, "w_off.pt"))
    for key in ("y1", "y64"):
        if not torch.equal(a[key], b[key]):
            failures.append(f"switch set: the output {key} differs from a process without it (it must have no effect)")
    print("[1] this architecture: the switch warns once per process and leaves the outputs unchanged")
    return failures


def main():
    import torch
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    major = torch.cuda.get_device_capability(0)[0]
    with tempfile.TemporaryDirectory() as tmp:
        failures = main_sm90(torch, tmp) if major == 9 else main_other(torch, tmp)
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\ntest_batch_invariance: ALL PASS")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2], [int(m) for m in sys.argv[3].split(",")])
        sys.exit(0)
    if len(sys.argv) > 2 and sys.argv[1] == "--warn-worker":
        warn_worker(sys.argv[2])
        sys.exit(0)
    sys.exit(main())
