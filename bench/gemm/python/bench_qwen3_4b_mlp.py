"""Frozen perf bench for the blockscale_gemm public PyTorch interface.
Pre-quantize / pre-pack scales OUTSIDE the timing loop.

Shapes: M ∈ {1, 2, 4, 8, 16, 32, 64, 128, 512, 1024, 2048, 4096}
        × Qwen3-4B attention + MLP projections (hidden=2560, 32 Q-heads,
          8 KV-heads, head_dim=128, intermediate=9728):
          - wqkv  (fused Q+K+V): N=6144,  K=2560
          - wo    (attn out):    N=2560,  K=4096
          - gate_up (fused):     N=19456, K=2560
          - gate / up:           N=9728,  K=2560
          - down:                N=2560,  K=9728

Precisions: BF16 + BlockScale FP8 (1×128). Both archs supported.

Output: JSON-per-line. The reformatter in this file (--format mode) builds
the markdown table for docs/perf.md.

Subprocess-per-cell so a crash in one cell can't poison the rest.
"""
import argparse, json, os, statistics, subprocess, sys, time
import torch, torch.nn.functional as F


def cos(a, b):
    return float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item())


def time_fn(fn, iters=50, warmup=15, repeats=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters): fn()
        e1.record(); torch.cuda.synchronize()
        samples.append(e0.elapsed_time(e1) / iters)
    return statistics.median(samples)


def time_fn_graph(fn, iters=50, warmup=15, repeats=3):
    """Capture-and-replay timing — what production decode actually sees once
    the forward pass is wrapped in `torch.cuda.graph`. Eliminates per-call
    host overhead (PyTorch op dispatch, cudaLaunchKernelEx, allocator hits)
    and isolates the steady-state GPU kernel time.

    `fn` must be safe to capture (no host syncs, no cudaMalloc inside the
    captured region). All the public blockscale_gemm linear ops satisfy
    this after the standard eager-warmup pass — see tests/unit/test_cuda_graph.py.
    """
    # Eager warmup populates static smem-cap guard, Params cache, and the
    # Stream-K pool's lazy cudaMalloc/cudaStreamCreate. All of these would
    # be illegal mid-capture.
    for _ in range(warmup): fn()
    torch.cuda.synchronize()

    # Capture-stream warmup (PyTorch convention).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()

    samples = []
    for _ in range(repeats):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters): g.replay()
        e1.record(); torch.cuda.synchronize()
        samples.append(e0.elapsed_time(e1) / iters)
    return statistics.median(samples)


SHAPES = [
    # Attention projections (Qwen3-4B: 32 Q-heads + 8 KV-heads, head_dim=128,
    # hidden=2560 → WQKV out = (32 + 2*8)*128 = 6144; WO in = 32*128 = 4096).
    ("wqkv",    6144,  2560),
    ("wo",      2560,  4096),
    # MLP projections (intermediate_size=9728, hidden=2560).
    ("gate_up", 19456, 2560),
    ("gate",    9728,  2560),
    ("down",    2560,  9728),
]
M_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 512, 1024, 2048, 4096]

# Cubic (M=N=K) sizes covering small-and-host-bound (1024) through peak-MFU
# (16384). Single cell per S; not crossed with M_GRID.
CUBIC_GRID = [1024, 1536, 2048, 2560, 3072, 4096, 6144, 8192, 12288, 16384]


def _inject_outliers(t: torch.Tensor, *, channel_dim: int,
                     channel_frac: float = 0.01, channel_scale: float = 30.0,
                     element_frac: float = 0.003, element_scale: float = 8.0,
                     seed: int = 0) -> torch.Tensor:
    """Stress FP8 block-scale quantization by injecting realistic outliers.

    Real LLM activations / weights have heavy-tailed distributions with a
    small number of consistently large channels (e.g. SmoothQuant retro).
    Pure Gaussian inputs give an unfairly clean cos floor (≈0.9993); with
    outliers, the block-scale quantization is exercised meaningfully.

    Pattern (defaults tuned to push FP8 1×128 cos meaningfully below the
    pure-Gaussian floor of ≈0.999; mimics SmoothQuant-class outlier
    intensity, ~1% of channels at 30× plus 0.3% sparse element outliers
    at 8×):
      * channel outliers: ``channel_frac`` of indices along ``channel_dim``
        get their slice multiplied by ``channel_scale``.
      * sparse element outliers: ~``element_frac`` of all entries scaled
        by ``element_scale``.
    Deterministic via the supplied seed.
    """
    g = torch.Generator(device=t.device).manual_seed(seed)
    K = t.size(channel_dim)
    n_channels = max(1, int(K * channel_frac))
    chan_ids = torch.randperm(K, generator=g, device=t.device)[:n_channels]
    idx = [slice(None)] * t.dim()
    idx[channel_dim] = chan_ids
    t = t.clone()
    t[tuple(idx)] = (t[tuple(idx)].float() * channel_scale).to(t.dtype)
    mask = torch.rand(t.shape, generator=g, device=t.device) < element_frac
    t = torch.where(mask, (t.float() * element_scale).to(t.dtype), t)
    return t


def bench_one(M, N, K, sm_major, inject_outliers: bool = False):
    import fish_scales_ops as fso
    out = {"M": M, "N": N, "K": K}
    torch.manual_seed(M * 1009 + N * 17 + K)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K**0.5)
    if inject_outliers:
        # Stress test: channel + sparse element outliers (SmoothQuant-class).
        x = _inject_outliers(x, channel_dim=1, seed=M * 31 + K)
        w = _inject_outliers(w, channel_dim=1, seed=N * 31 + K + 7)
    torch.cuda.synchronize()

    # All timings use CUDA-Graph capture-and-replay only. Eager timing was
    # dropped (it includes PyTorch op dispatch + cudaLaunchKernelEx overhead
    # that vanishes under graph capture; treating eager as the kernel
    # baseline misleads kernel-side tuning toward host-bound shapes that
    # aren't actually host-bound in production).
    try:
        y_bf = F.linear(x, w)
        out["bf16"] = {"us": time_fn_graph(lambda: F.linear(x, w)) * 1000.0, "cos": 1.0}
    except Exception as e:
        out["bf16"] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
        return out

    # BlockScale FP8 (1x128)
    try:
        xq, sxq = fso.gemm.quantize_1x128_fp8(x, use_ue8m0=(sm_major >= 12))
        wq, swq = fso.gemm.quantize_128x128_fp8(w)
        if sm_major >= 12:
            sxqp = fso.gemm.repack_fp8_act_scales(sxq)
            swqp = fso.gemm.repack_fp8_wgt_scales(swq)
        else:
            sxqp, swqp = sxq, swq
        torch.cuda.synchronize()
        y = fso.gemm.linear_fp8(xq, wq, sxqp, swqp)
        torch.cuda.synchronize()
        out["bsfp8"] = {
            "us":  time_fn_graph(lambda: fso.gemm.linear_fp8(xq, wq, sxqp, swqp)) * 1000.0,
            "cos": cos(y, y_bf),
        }
    except Exception as e:
        out["bsfp8"] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}

    # MXFP8 1×32 (sm_120 only; H200 row stays empty — calling on sm_90 raises)
    if sm_major >= 12:
        try:
            xqm, sxqm = fso.gemm.quantize_1x32_fp8(x)
            wqm, swqm = fso.gemm.quantize_1x32_fp8(w)
            torch.cuda.synchronize()
            y = fso.gemm.linear_mxfp8(xqm, wqm, sxqm, swqm)
            torch.cuda.synchronize()
            out["mxfp8"] = {
                "us":  time_fn_graph(lambda: fso.gemm.linear_mxfp8(xqm, wqm, sxqm, swqm)) * 1000.0,
                "cos": cos(y, y_bf),
            }
        except Exception as e:
            out["mxfp8"] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}

    # cuBLAS MXFP8 via torch.nn.functional.scaled_mm (sm_120+ only; sm_90 raises
    # CUBLAS_STATUS_NOT_SUPPORTED). Quantize via PyTorch's reference to_mxfp +
    # to_blocked (cuBLAS d-block-scaling layout); same numerical FP8 input as
    # the FSO MXFP8 path so the comparison is purely kernel speed.
    if sm_major >= 12:
        try:
            from torch.testing._internal.common_quantized import to_mxfp, to_blocked
            from torch._C import _ScalingType as ST, _SwizzleType as SW
            sA, A = to_mxfp(x.contiguous(), 32, "mxfp8")
            sB, B = to_mxfp(w.contiguous(), 32, "mxfp8")
            sA_b = to_blocked(sA)
            sB_b = to_blocked(sB)
            B_t = B.t()
            def _smm():
                return F.scaled_mm(A, B_t, sA_b, ST.BlockWise1x32, sB_b, ST.BlockWise1x32,
                                   swizzle_a=SW.SWIZZLE_32_4_4, swizzle_b=SW.SWIZZLE_32_4_4,
                                   output_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            y = _smm()
            torch.cuda.synchronize()
            out["smm_mxfp8"] = {
                "us":  time_fn_graph(_smm) * 1000.0,
                "cos": cos(y, y_bf),
            }
        except Exception as e:
            out["smm_mxfp8"] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}

    return out


def run_grid(out_path, inject_outliers: bool = False):
    sm_major = torch.cuda.get_device_capability(0)[0]
    sm_minor = torch.cuda.get_device_capability(0)[1]
    name = torch.cuda.get_device_name(0)
    cells = [(M, N, K, tag) for tag, N, K in SHAPES for M in M_GRID]
    cells += [(S, S, S, "cubic") for S in CUBIC_GRID]
    print(f"# Device: {name} (sm_{sm_major}{sm_minor})  cells: {len(cells)}"
          f"  outliers={inject_outliers}", flush=True)

    t0 = time.time()
    env = os.environ.copy()
    worker_argv = [sys.executable, __file__, "--worker"]
    if inject_outliers:
        worker_argv.append("--outliers")
    with open(out_path, "w") as f:
        f.write(json.dumps({"_device": name, "_sm": sm_major * 10 + sm_minor,
                            "_outliers": inject_outliers}) + "\n")
        for i, (M, N, K, tag) in enumerate(cells):
            try:
                proc = subprocess.run(
                    worker_argv,
                    input=f"{M} {N} {K} {tag}\n",
                    capture_output=True, text=True, timeout=300, env=env,
                )
                line = (proc.stdout or "").strip().splitlines()[-1] if proc.stdout else ""
                if not line.startswith("{"):
                    line = json.dumps({"M": M, "N": N, "K": K, "tag": tag,
                                       "error": (proc.stderr or "")[:200]})
            except Exception as e:
                line = json.dumps({"M": M, "N": N, "K": K, "tag": tag,
                                   "error": str(e)[:200]})
            f.write(line + "\n"); f.flush()
            try:
                r = json.loads(line)
                paths = "  ".join(
                    f"{k}={v.get('us', 'X'):.1f}"
                    if isinstance(v, dict) and "us" in v else f"{k}=ERR"
                    for k, v in r.items()
                    if k in ("bf16", "bsfp8", "mxfp8", "smm_mxfp8") and isinstance(v, dict))
            except Exception:
                paths = "PARSE_ERR"
            elapsed = time.time() - t0
            eta = elapsed * (len(cells) - i - 1) / max(i + 1, 1)
            print(f"[{i+1}/{len(cells)}] {tag:>7} M={M:5d} N={N:5d} K={K:5d}  {paths}  (t={elapsed:.0f}s eta={eta:.0f}s)",
                  flush=True)


def format_table(jsonl_paths_with_label):
    """Render docs/perf.md from one-or-more jsonl files (one per device).

    Each row reports both µs (lower is better) AND TFLOPS (higher is
    better). TFLOPS = 2*M*N*K / (µs * 1e6); only the GEMM is timed so this
    is steady-state op throughput, not end-to-end goodput.
    """
    def tflops(M, N, K, us):
        if us is None or us <= 0: return None
        return 2.0 * M * N * K / (us * 1e6)

    lines = []
    lines.append("# fish-scales-ops — frozen reference perf (GEMM)")
    lines.append("")
    lines.append("All `µs` numbers below are **CUDA-Graph capture-and-replay** timings — "
                 "each `linear_*` / `scaled_mm` call is captured into a `torch.cuda.CUDAGraph` "
                 "and replayed in a tight loop, then the median over 50 iters × 3 reps is "
                 "reported. Single-call eager timing was removed: it conflates kernel time "
                 "with PyTorch op dispatch + `cudaLaunchKernelEx` overhead, both of which a "
                 "graph-captured production decode loop pays exactly **once** (at capture), "
                 "so an eager baseline misleads kernel-side tuning toward host-bound shapes "
                 "that are not host-bound in production. `TF` (TFLOPS = `2·M·N·K / (µs · 1e6)`) "
                 "is computed from these graph numbers. BF16 baseline is "
                 "`torch.nn.functional.linear`. Pre-quantize + pre-pack happen outside the "
                 "timing loop. `cos` is cosine similarity vs BF16.")
    lines.append("")
    lines.append("Columns:")
    lines.append("- `BSFP8` — fish-scales-ops `linear_fp8` (1×128 act / 128×128 wgt, "
                 "UE8M0 scales on sm_120).")
    lines.append("- `MXFP8` — fish-scales-ops `linear_mxfp8` (1×32, UE8M0). sm_120 only.")
    lines.append("- `sMM`   — `torch.nn.functional.scaled_mm` with the same 1×32 UE8M0 "
                 "block-scaled FP8 inputs (cuBLAS / cuBLASLt path); sm_120+ only. Apples-"
                 "to-apples kernel comparison vs `MXFP8`.")
    lines.append("")
    lines.append("Shapes are Qwen3-4B (hidden=2560, 32 Q-heads + 8 KV-heads, head_dim=128, intermediate=9728):")
    lines.append("- `wqkv`     N=6144  K=2560 (fused Q+K+V projection)")
    lines.append("- `wo`       N=2560  K=4096 (attention output projection)")
    lines.append("- `gate_up`  N=19456 K=2560 (fused gate+up)")
    lines.append("- `gate`     N=9728  K=2560 (single gate or up)")
    lines.append("- `down`     N=2560  K=9728")
    lines.append("Plus a `cubic` (M=N=K) sweep ∈ {1024, 1536, 2048, 2560, 3072, 4096, "
                 "6144, 8192, 12288, 16384} for the peak-MFU regime.")
    lines.append("")
    lines.append("Reproduce: `PYTHONPATH=python python bench/gemm/python/bench_qwen3_4b_mlp.py --run --out <jsonl>`.")
    lines.append("")

    def fmt_us(v):  return f"{v:.2f}" if v is not None else "—"
    def fmt_tf(v):  return f"{v:.0f}" if v is not None else "—"
    def fmt_cos(v): return f"{v:.4f}" if v is not None else "—"

    for label, path in jsonl_paths_with_label:
        with open(path) as f:
            recs = [json.loads(l) for l in f if l.strip()]
        if not recs:
            continue
        meta = recs[0] if "_device" in recs[0] else {"_device": "?", "_sm": "?"}
        rows = [r for r in recs if "_device" not in r]

        lines.append(f"## {label} — {meta['_device']} (sm_{meta['_sm']})")
        lines.append("")
        by_tag = {}
        for r in rows:
            tag = r.get("tag")
            if tag is None:
                for t, n, k in SHAPES:
                    if r.get("N") == n and r.get("K") == k:
                        tag = t; break
            by_tag.setdefault(tag, []).append(r)

        def _render_row(c, label_val, label_w=5):
            M = c["M"]; N = c["N"]; K = c["K"]
            def pick(field):
                return field.get("graph_us") or field.get("us")
            bf = pick(c.get("bf16", {}))
            bs = pick(c.get("bsfp8", {}))
            mx = pick(c.get("mxfp8", {}))
            sm = pick(c.get("smm_mxfp8", {}))
            bs_c = c.get("bsfp8", {}).get("cos")
            mx_c = c.get("mxfp8", {}).get("cos")
            sm_c = c.get("smm_mxfp8", {}).get("cos")
            bs_t = tflops(M, N, K, bs); mx_t = tflops(M, N, K, mx)
            sm_t = tflops(M, N, K, sm)
            lbl = f"{label_val:>{label_w}}"
            cells_str = [f"| {lbl}",
                         f"{fmt_us(bf):>7}",
                         f"{fmt_us(bs):>8}", f"{fmt_tf(bs_t):>8}"]
            if has_mxfp8:
                cells_str += [f"{fmt_us(mx):>8}", f"{fmt_tf(mx_t):>8}"]
            if has_smm:
                cells_str += [f"{fmt_us(sm):>8}", f"{fmt_tf(sm_t):>8}"]
            cells_str += [f"{fmt_cos(bs_c):>9}"]
            if has_mxfp8:
                cells_str += [f"{fmt_cos(mx_c):>9}"]
            if has_smm:
                cells_str += [f"{fmt_cos(sm_c):>9}"]
            return " | ".join(cells_str) + " |"

        def _render_header(label_name, label_w=5):
            cols = [f"{label_name:>{label_w}}", "BF16 µs", "BSFP8 µs", "BSFP8 TF"]
            if has_mxfp8:
                cols += ["MXFP8 µs", "MXFP8 TF"]
            if has_smm:
                cols += ["sMM µs", "sMM TF"]
            cols += ["BSFP8 cos"]
            if has_mxfp8:
                cols += ["MXFP8 cos"]
            if has_smm:
                cols += ["sMM cos"]
            head = "| " + " | ".join(f"{c:>{max(label_w, len(c))}}" for c in cols) + " |"
            sep  = "| " + " | ".join(f"{'---:':>{max(label_w, len(c))}}" for c in cols) + " |"
            return head, sep

        # Determine whether sMM / MXFP8 columns should appear at all (this
        # device's bench produced them on at least one cell).
        has_mxfp8 = any("mxfp8" in c and "us" in c.get("mxfp8", {}) for c in rows)
        has_smm   = any("smm_mxfp8" in c and "us" in c.get("smm_mxfp8", {}) for c in rows)

        for tag, n, k in SHAPES:
            cells = by_tag.get(tag, [])
            cells.sort(key=lambda r: r["M"])
            if not cells: continue
            lines.append(f"### `{tag}` (N={n}, K={k})")
            lines.append("")
            head, sep = _render_header("M")
            lines.append(head); lines.append(sep)
            for c in cells:
                lines.append(_render_row(c, c["M"]))
            lines.append("")

        # Cubic section: M=N=K per row, single section spanning the cubic grid.
        cubic_cells = by_tag.get("cubic", [])
        cubic_cells.sort(key=lambda r: r["M"])
        if cubic_cells:
            lines.append("### cubic (M=N=K) — peak-MFU sweep")
            lines.append("")
            head, sep = _render_header("M=N=K", label_w=6)
            lines.append(head); lines.append(sep)
            for c in cubic_cells:
                lines.append(_render_row(c, c["M"], label_w=6))
            lines.append("")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--run", action="store_true", help="run grid")
    p.add_argument("--out", help="output jsonl (with --run)")
    p.add_argument("--outliers", action="store_true",
                   help="inject channel + sparse outliers to stress FP8 cos")
    p.add_argument("--format", action="store_true", help="format jsonl(s) into perf.md")
    p.add_argument("--from", dest="srcs", nargs="+",
                   help="LABEL=jsonl pairs, e.g. Blackwell=sm120.jsonl H200=h200.jsonl")
    p.add_argument("--md-out", default="-", help="markdown output (with --format)")
    args = p.parse_args()

    if args.worker:
        line = sys.stdin.read().strip().split()
        M, N, K = int(line[0]), int(line[1]), int(line[2])
        tag = line[3] if len(line) > 3 else None
        sm_major = torch.cuda.get_device_capability(0)[0]
        res = bench_one(M, N, K, sm_major, inject_outliers=args.outliers)
        if tag is not None: res["tag"] = tag
        sys.stdout.write(json.dumps(res) + "\n")
        return

    if args.run:
        if not args.out:
            sys.exit("--out required with --run")
        run_grid(args.out, inject_outliers=args.outliers)
        return

    if args.format:
        if not args.srcs:
            sys.exit("--from required with --format")
        pairs = []
        for spec in args.srcs:
            if "=" not in spec:
                sys.exit(f"--from entry must be LABEL=path: got {spec!r}")
            label, path = spec.split("=", 1)
            pairs.append((label, path))
        md = format_table(pairs)
        if args.md_out == "-":
            sys.stdout.write(md)
        else:
            with open(args.md_out, "w") as f: f.write(md)
        return

    p.print_help()


if __name__ == "__main__":
    main()
