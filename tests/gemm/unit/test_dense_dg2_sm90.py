#!/usr/bin/env python3
"""The sm_90 dense block-FP8 GEMM's second kernel: DeepGEMM v2's sm90_fp8_gemm_1d2d (vendored with the JIT headers,
arch/sm90/fp8/jit/deep_gemm/impls/), which gemm_dispatch_sm90 runs at DeepGEMM's own configuration for the tile
classes deep_gemm::jit::use_dg2_dense accepts (256 x 128 with multicast on A and 128 x 160 / 192 from M = 512 on, every
DeepGEMM pick but its 80-wide tiles at 128 < M < 256, 64-row tiles at M = 64 / 128 with K >= 8192), and
fp8_gemm_kernel otherwise. FSO_SM90_DENSE_DG2 (read once per process)
selects the route: 1 (default) the rule, 0 fp8_gemm_kernel only, all every DeepGEMM configuration.

Each mode runs in its own child process over the same cells, and the test checks:

  1. bitwise equality: linear_fp8 and linear_qx give torch.equal outputs in all three modes on every cell (both kernels
     accumulate each 128-wide K block in FP32 in one CTA and promote it with the same scale product in the same order);
  2. the route: with the default, every cell of the rule's classes launches sm90_fp8_gemm_1d2d_impl with the expected
     tile, and every control cell (swap-AB small M, M = 128 / 256 gate_up, 256 x 112 and one-wave 256 x 128 picks)
     launches fp8_gemm_kernel; FSO_SM90_DENSE_DG2=0 never launches the v2 kernel;
  3. graph capture: in the default mode a CUDA graph of the quantize + GEMM chain on two v2 cells, with the output
     clobbered between capture and replay, replays to the eager result.
  4. a value other than 0, 1 or all prints one warning and keeps the default.

The cells include off-grid M (partial 256-row and 64-row blocks, M below the tile) and an N that is not a multiple of
the tile. sm_90 only (prints SKIP elsewhere). usage: python tests/gemm/unit/test_dense_dg2_sm90.py
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
    (64, 2560, 9728, "dg2 64x32"),        # decode class: 64-row tile, K >= 8192
    (128, 2560, 9728, "dg2 64x48"),
    (100, 2560, 9728, "fso"),             # decode M with a partly filled last 64-row block: fso's kernel
    (96, 2560, 9728, "fso"),
    (160, 19456, 2560, "dg2 128x160"),    # 128 < M < 256: DeepGEMM's pick (multicast on B)
    (160, 2560, 9728, "dg2 64x64"),
    (192, 6144, 2560, "dg2 64x144"),
    (224, 9216, 2048, "dg2 128x144"),
    (224, 2560, 4096, "fso"),             # 128 < M < 256 but DeepGEMM picks 64 x 80: fso's kernel
    (224, 19456, 2560, "dg2 128x160"),
    (16, 2560, 9728, "fso"),              # swap-AB small M
    (128, 19456, 2560, "fso"),            # 128 x 160 at M = 128 / 256: fso's kernel
    (256, 19456, 2560, "fso"),
    (512, 19456, 2560, "fso"),            # DeepGEMM picks 256 x 112
    (2048, 2048, 4096, "fso"),            # DeepGEMM picks one wave of 256 x 128 without multicast
    (64, 2560, 4096, "fso"),              # 64-row tile but K < 8192
]
GRAPH_CELLS = [(1024, 19456, 2560), (128, 2560, 9728)]


def kernel_of(names):
    for n in names:
        m = re.search(r"sm90_fp8_gemm_1d2d_impl<[^,]*, 0u?, \d+u?, 0u?, 1u?, (\d+)u?, (\d+)u?,", n)
        if m:
            return f"dg2 {m.group(1)}x{m.group(2)}"
    for n in names:
        if "fp8_gemm_kernel" in n:
            return "fso"
    return "none: " + " | ".join(names)


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
            if want is not None and got != want:
                failures.append(f"{key}: default mode launched {got}, expected {want}")
            n_dg2 += got.startswith("dg2")
            if results["0"]["kernels"][key] != "fso":
                failures.append(f"{key}: FSO_SM90_DENSE_DG2=0 launched {results['0']['kernels'][key]}")
        print(f"[2] default route: {n_dg2} of {len(CELLS)} cells on sm90_fp8_gemm_1d2d_impl, as expected; "
              "FSO_SM90_DENSE_DG2=0 runs fp8_gemm_kernel on every cell")
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
