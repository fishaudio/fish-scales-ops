#!/usr/bin/env python3
"""The sm_90 dense block-FP8 GEMM's second kernel: DeepGEMM v2's sm90_fp8_gemm_1d2d (vendored with the JIT headers,
arch/sm90/fp8/jit/deep_gemm/impls/), which gemm_dispatch_sm90 runs at DeepGEMM's own configuration for the tile
classes deep_gemm::jit::use_dg2_dense accepts (256 x 128 with multicast on A and 128 x 160 / 192 from M = 512 on, every
DeepGEMM pick but its 80-wide tiles at 128 < M < 256, 64-row tiles at M = 64 / 128 with K >= 8192), at the tiles of
deep_gemm::jit::use_dense_v2_override (FH1 of the issue-12 round: one wave at M <= 256 where the route needs two or
more, the 128 x 160 / 192 pair over N in a quarter fewer waves for wide outputs at 256 < M <= 672, the 64 x 128 pair
over M for narrow outputs at 256 < M <= 384), and fp8_gemm_kernel otherwise. FSO_SM90_DENSE_DG2 (read once per process)
selects the route: 1 (default) the rule, 0 fp8_gemm_kernel only, all every DeepGEMM configuration.

Each mode runs in its own child process over the same cells, and the test checks:

  1. bitwise equality: linear_fp8 and linear_qx give torch.equal outputs in all three modes on every cell (both kernels
     accumulate each 128-wide K block in FP32 in one CTA and promote it with the same scale product in the same order);
  2. the route: with the default, every cell of the rule's classes launches sm90_fp8_gemm_1d2d_impl with the expected
     tile (and, where the expectation names one, the cluster: c2x1 pairs two M blocks, c1x2 two N blocks), the cells of
     FH1's two swap-AB bands and of its 64-row weight tiles at 8 <= M <= 32 launch fp8_gemm_kernel_swapAB with the
     expected tile, the cells of the in-cluster split-K route (FH5) launch fp8_gemm_kernel_swapAB_splitk with the
     expected slice count, and every control cell (swap-AB small M, M = 256 gate_up, one-wave 256 x 128 picks, the
     neighbours of FH1's rules) launches fso's kernels; FSO_SM90_DENSE_DG2=0 never launches the v2 kernel (the split-K
     route's cells keep that route in every mode);
  3. graph capture: in the default mode a CUDA graph of the quantize + GEMM chain on seven v2 cells, with the output
     clobbered between capture and replay, replays to the eager result.
  4. a value other than 0, 1 or all prints one warning and keeps the default.

The cells include off-grid M (partial 256-row and 64-row blocks, M below the tile) and an N that is not a multiple of
the tile, and the shapes of rule 5 (fso's tile with at least 48 rows of its last M block empty, below M = 1024: the
v2 kernel at DeepGEMM's single-CTA pick, or at fso's tile when DeepGEMM picks a two-CTA cluster, as long as the v2
configuration gives at least three quarters of the SMs a tile) with its controls.
sm_90 only (prints SKIP elsewhere). usage: python tests/gemm/unit/test_dense_dg2_sm90.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile

# (M, N, K, expected kernel with the default rule: "dg2 <block_m>x<block_n>" or "fso")
CELLS = [
    (1024, 19456, 2560, "dg2 256x128"),   # 256 x 128, multicast on A
    (1000, 19456, 2560, "dg2 256x128"),   # partial last 256-row block
    (8192, 2560, 9728, "dg2 256x128"),
    (1024, 2560, 9728, "dg2 128x160"),    # 128 x 160, one CTA, N compiled in
    (2048, 2560, 9728, "dg2 128x160"),    # 128 x 160, multicast on B
    (512, 6144, 2560, "dg2 128x192"),     # 128 x 192
    (2048, 10008, 2048, "dg2 256x128"),   # N not a multiple of 2 x 128: the multicast pairs end in an odd tail
    (512, 10008, 2048, "dg2 128x160"),    # N not a multiple of 160, multicast on B
    (64, 2560, 9728, "splitk 2"),         # FH5: the in-cluster split-K route takes M <= 64 at K >= 8192 for N <= 4224
    #                                       (it was the v2 decode class, 64 x 32; the 64-row v2 tile stays at M = 128)
    (128, 2560, 9728, "dg2 64x48"),
    (100, 2560, 9728, "fso"),             # decode M with a partly filled last 64-row block: fso's kernel
    #                                       (fp8_gemm_kernel_swapAB with a 40-token tile since H4 of issue 12)
    (96, 2560, 9728, "fso"),              # (fp8_gemm_kernel_swapAB with a 32-token tile since H3 of issue 12)
    (160, 19456, 2560, "dg2 128x160"),    # 128 < M < 256: DeepGEMM's pick (multicast on B)
    (160, 2560, 9728, "dg2 64x64"),
    (192, 6144, 2560, "dg2 64x144"),
    (224, 9216, 2048, "dg2 128x144"),
    (224, 2560, 4096, "fso"),             # 128 < M < 256 but DeepGEMM picks 64 x 80: fso's kernel
    (224, 19456, 2560, "dg2 128x160"),
    (16, 2560, 9728, "splitk 3"),         # FH5: split-K, three slices on SMs shared two to one (token tile 16)
    (256, 19456, 2560, "fso"),            # 128 x 160 at M = 256 (DeepGEMM's pick, two waves): fso's kernel
    (2048, 2048, 4096, "fso"),            # DeepGEMM picks one wave of 256 x 128 without multicast
    (64, 2560, 4096, "fso"),              # 64-row tile but K < 8192
    # rule 5: fso's tile leaves at least 48 rows of its last M block empty (below M = 1024)
    (80, 5120, 2048, "dg2 64x80"),        # DeepGEMM's single-CTA pick (fso: 128 x 40, 48 rows empty)
    (136, 2048, 2048, "dg2 64x48"),       # DeepGEMM's single-CTA pick (fso: 64 x 32, 56 rows empty)
    (136, 2048, 4096, "fso"),             # the same at K = 4096: fp8_gemm_kernel_swapAB (40 tokens) since H4
    # rule-5 shapes that the swap-AB route takes since H3 of issue 12 (64 < M <= 96 with a 64-row weight tile:
    # fp8_gemm_kernel_swapAB, which this test counts as "fso"): the v2 kernel's 64 x 32 tile at M = 78 / 80
    (80, 2048, 4096, "fso"),
    (78, 2048, 1280, "fso"),
    (78, 1280, 5120, "fso"),              # (was fp8_gemm_kernel: 80 tiles of the v2 kernel, under three quarters)
    (288, 5120, 2048, "dg2 128x128"),     # DeepGEMM picks a two-CTA cluster: fso's one-wave tile on the v2 kernel
    (416, 5120, 2048, "dg2 128x160"),     # one wave instead of fso's two
    (533, 1280, 5120, "dg2 64x96"),
    (533, 5120, 1280, "fso"),             # DeepGEMM's pick clustered and fso's tile two waves: fso's kernel
    (576, 5120, 2048, "fso"),
    (112, 5120, 2048, "fso"),             # 16 rows empty: below the threshold
    (1079, 1280, 5120, "fso"),            # from M = 1024 the v2 classes alone decide
    # FH1 of the issue-12 round (deep_gemm::jit::use_dense_v2_override and the two swap-AB bands of
    # use_dense_swap_ab_above_96; /data/bench-runs/fso_issue12_20261028/runs/FH1). One wave of the v2 kernel at
    # M <= 256 where the route needs two or more: the fewest rows, then the narrowest tile, single-CTA, paired over
    # M (c2x1) for 128-row tiles at 128 < M <= 192; up to M = 128 only widths with one main loop (not 96 / 160 / 192).
    (8, 18432, 2560, "dg2 16x144"),       # was fp8_gemm_kernel_swapAB 128 x 16 (two waves of 144 tiles)
    (64, 18432, 2560, "dg2 64x144"),      # was fp8_gemm_kernel 64 x 72
    (128, 18432, 2560, "dg2 128x144"),    # was fp8_gemm_kernel 128 x 72
    (8, 19456, 2560, "fso"),              # one wave needs a 160- or 192-wide tile: kept (swap-AB)
    (128, 19456, 2560, "fso"),            # the same: kept (fp8_gemm_kernel 128 x 80)
    (160, 9728, 2560, "dg2 128x160 c2x1"),  # was DeepGEMM's 64 x 112 pair
    (192, 12288, 2048, "dg2 128x192 c2x1"),
    (256, 10240, 2560, "dg2 128x160"),    # was fp8_gemm_kernel 128 x 80 (three waves)
    # swap-AB, three token tiles: narrow outputs at 144 < M <= 216 (64-row weight tiles, K >= 8192, or against
    # fp8_gemm_kernel's tile), wide outputs at 256 < M <= 288 (128-row weight tiles, at most 96 tokens)
    (152, 2560, 9216, "swapAB 64x56"),    # was the v2 kernel's 64 x 64
    (216, 2560, 4096, "swapAB 64x72"),    # was fp8_gemm_kernel 128 x 40
    (200, 2560, 4096, "dg2 64x80"),       # K < 8192 against the v2 main path: kept
    (272, 10240, 2560, "swapAB 128x96"),  # was fp8_gemm_kernel 128 x 128 (three waves, 96 of 384 rows empty)
    (296, 10240, 2560, "fso"),            # above the band: kept
    # wide outputs at 256 < M <= 672: the v2 kernel's 128 x 160 / 192 pair over N (c1x2) in fewer waves
    (288, 18432, 2560, "dg2 128x160 c1x2"),
    (400, 18432, 2560, "dg2 128x192 c1x2"),
    (512, 19456, 2560, "fso"),            # one wave fewer out of five (DeepGEMM picks 256 x 112): kept
    (680, 18432, 2560, "fso"),            # above M = 672: kept
    (472, 5120, 2048, "fso"),             # the pair would need one wave only: kept
    # narrow outputs at 256 < M <= 384, K >= 4096: the v2 kernel's 64 x 128 pair over M (c2x1)
    (288, 2560, 4096, "dg2 64x128 c2x1"),  # was the v2 kernel's 64 x 112 (five M blocks, weights read five times)
    (344, 2560, 9216, "dg2 64x128 c2x1"),  # was fp8_gemm_kernel 128 x 64
    (328, 2560, 4096, "dg2 64x128"),      # the route is DeepGEMM's own 64 x 128 already: kept
    (288, 2560, 2048, "dg2 64x112"),      # K < 4096: kept
    # swap-AB at 8 <= M <= 32: 64-row weight tiles with one token tile covering M where they fit one wave (N <= 8448)
    (16, 5120, 2048, "swapAB 64x16"),     # was 128 x 16
    (32, 6144, 2560, "swapAB 64x32"),     # was 128 x 16 in two token tiles
    (4, 6144, 2560, "swapAB 128x16"),     # M < 8: kept
    (16, 9216, 2048, "swapAB 128x16"),    # 144 tiles of 64 rows would need two waves: kept
]
GRAPH_CELLS = [(1024, 19456, 2560), (128, 2560, 9728), (80, 5120, 2048), (288, 5120, 2048), (160, 9728, 2560),
               (288, 18432, 2560), (288, 2560, 4096)]


def kernel_of(names):
    """"dg2 <block_m>x<block_n>" (with " c2x1" for a two-CTA cluster that shares the weight, multicast on B, and
    " c1x2" for one that shares the activation, multicast on A), "splitk <slices>" for the in-cluster split-K kernel
    fp8_gemm_kernel_swapAB_splitk, "swapAB <weight rows>x<token tile>" for fp8_gemm_kernel_swapAB, "fso" for
    fp8_gemm_kernel. The v2 kernel's template arguments run <major, 0, N, 0, 1,
    BLOCK_M, BLOCK_N, 128, 128, 128, swizzle, stages, 128, threads, cluster size, multicast on A, ...>."""
    for n in names:
        m = re.search(r"sm90_fp8_gemm_1d2d_impl<[^,]*, 0u?, \d+u?, 0u?, 1u?, (\d+)u?, (\d+)u?, 128u?, 128u?, 128u?, "
                      r"\d+u?, \d+u?, 128u?, \d+u?, (\d+)u?, (true|false)", n)
        if m:
            cluster = "" if m.group(3) == "1" else (" c1x2" if m.group(4) == "true" else " c2x1")
            return f"dg2 {m.group(1)}x{m.group(2)}{cluster}"
    for n in names:
        m = re.search(r"fp8_gemm_kernel_swapAB_splitk<\d+u?, \d+u?, (\d+)u?,", n)
        if m:
            return f"splitk {m.group(1)}"
    for n in names:
        m = re.search(r"fp8_gemm_kernel_swapAB<\d+u?, \d+u?, (\d+)u?, (\d+)u?,", n)
        if m:
            return f"swapAB {m.group(1)}x{m.group(2)}"
    for n in names:
        if "fp8_gemm_kernel" in n:
            return "fso"
    return "none: " + " | ".join(names)


def matches(got: str, want: str) -> bool:
    """An expectation without a cluster suffix accepts every cluster form of its tile, and "fso" accepts both of
    fso's kernels (fp8_gemm_kernel and fp8_gemm_kernel_swapAB); a suffix or a swap-AB tile must match exactly."""
    if got == want:
        return True
    if want == "fso":
        return got.startswith("swapAB")
    return want.startswith("dg2") and want.count(" ") == 1 and got.rsplit(" ", 1)[0] == want


def worker(out_path):
    import torch
    from torch.profiler import ProfilerActivity, profile
    import fish_scales_ops as fso

    res = {"kernels": {}, "graph": {}}
    tensors = {}
    for M, N, K, _ in CELLS:
        g = torch.Generator(device="cuda")
        g.manual_seed(M * 1009 + N * 17 + K)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g) * 0.1
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g) / K ** 0.5
        xq, sx = fso.compat.quantize_1x128_fp8(x, use_ue8m0=False)
        wq, sw = fso.compat.quantize_128x128_fp8(w)
        y = fso.compat.linear_fp8(xq, wq, sx, sw)
        yq = torch.ops.fish_scales_ops.linear_qx(x, wq, sw)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fso.compat.linear_fp8(xq, wq, sx, sw)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        key = f"{M},{N},{K}"
        res["kernels"][key] = kernel_of(names)
        tensors[key + "/y"] = y.cpu()
        tensors[key + "/yq"] = yq.cpu()
        if (M, N, K) in GRAPH_CELLS:
            def call():
                q, s = fso.compat.quantize_1x128_fp8(x, use_ue8m0=False)
                return fso.compat.linear_fp8(q, wq, s, sw)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    call()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                yg = call()
            yg.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            res["graph"][key] = bool(torch.equal(yg, y))
    torch.save(tensors, out_path)
    print("RESULT " + json.dumps(res), flush=True)


def run(mode, out_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("FSO_")}
    if mode is not None:
        env["FSO_SM90_DENSE_DG2"] = mode
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", out_path], env=env,
                       capture_output=True, text=True, timeout=1800)
    res = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[7:])
    if p.returncode != 0 or res is None:
        raise SystemExit(f"worker (FSO_SM90_DENSE_DG2={mode}) failed, exit {p.returncode}:\n{p.stderr[-3000:]}")
    return res, p.stderr


def main():
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 9:
        print("SKIP: sm_90 (H200) only")
        return 0
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        outs, results = {}, {}
        for mode in (None, "0", "all"):
            path = os.path.join(tmp, f"out_{mode}.pt")
            results[mode], _ = run(mode, path)
            outs[mode] = torch.load(path)
        # 1. bitwise equality across the three modes
        for key in outs[None]:
            for mode in ("0", "all"):
                if not torch.equal(outs[None][key], outs[mode][key]):
                    failures.append(f"{key}: default and FSO_SM90_DENSE_DG2={mode} differ")
        print(f"[1] {len(outs[None])} outputs bitwise identical in the default, 0 and all modes"
              if not failures else f"[1] {len(failures)} mismatches")
        # 2. the route
        n_dg2 = 0
        for M, N, K, want in CELLS:
            key = f"{M},{N},{K}"
            got = results[None]["kernels"][key]
            if want is not None and not matches(got, want):
                failures.append(f"{key}: default mode launched {got}, expected {want}")
            n_dg2 += got.startswith("dg2")
            # The split-K route's calls are on neither of the paths the knob selects between: they keep the route.
            want0 = want if want is not None and want.startswith("splitk") else "fso"
            if not matches(results["0"]["kernels"][key], want0):
                failures.append(f"{key}: FSO_SM90_DENSE_DG2=0 launched {results['0']['kernels'][key]}, "
                                f"expected {want0}")
        print(f"[2] default route: {n_dg2} of {len(CELLS)} cells on sm90_fp8_gemm_1d2d_impl, as expected; "
              "FSO_SM90_DENSE_DG2=0 runs fp8_gemm_kernel on every cell outside the split-K route")
        # 3. graph capture and replay
        for key, ok in results[None]["graph"].items():
            if not ok:
                failures.append(f"{key}: the graph replay differs from the eager result")
        print(f"[3] graph replay equals eager on {len(results[None]['graph'])} v2 cells")
        # 4. an invalid value warns once and keeps the default
        res_bad, err = run("bogus", os.path.join(tmp, "out_bad.pt"))
        if err.count("FSO_SM90_DENSE_DG2=bogus is not 0, 1 or all") != 1:
            failures.append("FSO_SM90_DENSE_DG2=bogus: no single warning on stderr")
        if res_bad["kernels"] != results[None]["kernels"]:
            failures.append("FSO_SM90_DENSE_DG2=bogus: the route differs from the default")
        print("[4] FSO_SM90_DENSE_DG2=bogus warns once and keeps the default route")
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\ntest_dense_dg2_sm90: ALL PASS")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
        sys.exit(0)
    sys.exit(main())
