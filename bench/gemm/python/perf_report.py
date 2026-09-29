#!/usr/bin/env python3
"""perf_report.py — the one path from a perf run directory to baselines, tables and reports.

A perf run leaves raw jsonl files in a run directory outside the git tree
(5090: /mnt/share/stone-bench-runs/<run>/, H200: /data/bench-runs/<run>/). This
tool turns them into the canonical baseline files, installs them, renders the
docs, and prints the comparison reports — so a rerun after any change is
`merge` → `diff` → (if accepted) `install`, and never a hand-written script.

  python bench/gemm/python/perf_report.py merge   --run <dir> [--out <dir>]      raw files -> canonical baseline files (n/a cells listed)
  python bench/gemm/python/perf_report.py diff    --a <dir|baselines> --b <dir>   cell-by-cell delta of every shared cell (A/B of an optimisation)
  python bench/gemm/python/perf_report.py report  [--run <dir>]                  layer-level and kernel-level comparison tables, ×fso and band geomeans
  python bench/gemm/python/perf_report.py install --run <dir>                    merge + diff against the installed baselines + copy + render docs

Raw-file manifest (device suffix `--device`, families 30a3 / 35a3):
  perf_moe_qwen3_<fam>_<dev>.jsonl            fso rows (fso_mxfp8_grouped + fso_mxfp8_layer; sm_90: fso_bsfp8_layer)  -> same name
  perf_moe_qwen3_35a3_shared_<dev>.jsonl      fso routed + shared block                              -> same name
  ref_new_35a3_shared_<dev>.jsonl             the block's comparators (sm_100: torch-native)          -> ref_moe_qwen3_35a3_shared_<dev>.jsonl
  ref_new_*, ref_fix_*, ref_trt_*, ref_sgl020_*, ref_torchma_*, ref_torchma_smm_*   layer comparators -> ref_moe_qwen3_<fam>_<dev>.jsonl
  ref_kern_c1_vllm_*, ref_kern_c1_sgl_*, ref_kern_c1_fi_*         kernel-level       -> ref_kern_moe_qwen3_<fam>_<dev>.jsonl
  gemm_sm<sm>_qwen3_4b.jsonl, gemm_sm<sm>_qwen3_4b_mlp_fwd.jsonl, gemm_sm<sm>_qwen3_{30a3,35a3}_dense.jsonl   dense tables -> same name
Bands (stone, 2026-09-28): decode = M 1..128, prefill = M > 128.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BASELINES = os.path.join(ROOT, "tests", "baselines")
FAMILIES = {"30a3": "Family B — Qwen3-30B-A3B routed MoE layer (E=128, top-8, moe_inter 768)",
            "35a3": "Family C — Qwen3.5-35B-A3B routed MoE layer (E=256, top-8, moe_inter 512)"}
BANDS = [("decode M 1–128", lambda m: m <= 128), ("prefill M > 128", lambda m: m > 128), ("all", lambda m: True)]
DASH = "—"

# impl -> short column title (layer-level comparators, in report order)
LAYER_IMPLS = [("vllm_fp8b", "vLLM triton FP8-block"), ("triton_fp8b", "sglang triton FP8-block"),
               ("trtllm_cutlass_fp8", "TRT-LLM CUTLASS FP8 per-tensor"),
               ("vllm_bf16", "vLLM triton BF16"), ("triton_bf16", "sglang triton BF16"),
               ("trtllm_cutlass_bf16", "TRT-LLM CUTLASS BF16"),
               ("torch_grouped_bf16_layer_maxautotune", "torch _grouped_mm BF16 + compile max-autotune"),
               ("torch_grouped_bf16_layer", "torch _grouped_mm BF16"),
               ("torch_grouped_bf16_layer_compiled", "torch _grouped_mm BF16 + compile"),
               ("torch_smm_mxfp8_layer_maxautotune", "torch scaled_grouped_mm MXFP8 + compile max-autotune"),
               ("torch_smm_mxfp8_layer", "torch scaled_grouped_mm MXFP8"),
               ("dg_fp8_layer", "deep_gemm masked pipeline FP8")]
# the fso row of each device's tables: MXFP8 on sm_120 / sm_100, block-FP8 (BSFP8) on sm_90
FSO_LAYER_IMPLS = ["fso_mxfp8_layer", "fso_bsfp8_layer"]
FSO_KERN_IMPLS = ["fso_mxfp8_grouped", "fso_bsfp8_grouped"]
KERN_IMPLS = [("vllm_triton_grouped_fp8b", "vLLM triton FP8-block"), ("sgl_triton_grouped_fp8b", "sglang triton FP8-block"),
              ("fi_cudnn_grouped_mxfp8", "FlashInfer cuDNN MXFP8"), ("vllm_triton_grouped_bf16", "vLLM triton BF16"),
              ("sgl_triton_grouped_bf16", "sglang triton BF16"), ("fi_cudnn_grouped_bf16", "FlashInfer cuDNN BF16")]


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def merge_files(out_path, sources):
    """One meta row (the first source's, with every source's torch recorded), then every
    cell that has a µs; cells without one are returned as n/a with their error."""
    meta, cells, na = None, [], []
    for src in sources:
        for r in load(src):
            if r.get("kind") == "meta":
                if meta is None:
                    meta = dict(r)
                    meta["sources"] = []
                meta["sources"].append({"file": os.path.basename(src), "torch": r.get("torch")})
                continue
            (cells if r.get("us") is not None else na).append(r)
    if meta is None:
        meta = {"kind": "meta", "sources": [os.path.basename(s) for s in sources]}
    with open(out_path, "w") as f:
        f.write(json.dumps(meta) + "\n")
        for r in cells:
            f.write(json.dumps(r) + "\n")
    return len(cells), na


def manifest(run, dev, sm):
    """baseline file name -> list of existing raw files in `run` that feed it."""
    m = {}
    for fam in FAMILIES:
        m[f"perf_moe_qwen3_{fam}_{dev}.jsonl"] = [f"perf_moe_qwen3_{fam}_{dev}.jsonl"]
        m[f"ref_moe_qwen3_{fam}_{dev}.jsonl"] = [f"ref_new_{fam}_{dev}.jsonl", f"ref_fix_{fam}_*_{dev}.jsonl", f"ref_trt_{fam}_{dev}.jsonl",
                                                 f"ref_sgl020_{fam}_{dev}.jsonl", f"ref_torchma_{fam}_{dev}.jsonl",
                                                 f"ref_torchma_smm_{fam}_{dev}.jsonl"]
        m[f"ref_kern_moe_qwen3_{fam}_{dev}.jsonl"] = [f"ref_kern_c1_vllm_{fam}_{dev}.jsonl", f"ref_kern_c1_sgl_{fam}_{dev}.jsonl",
                                                      f"ref_kern_c1_fi_{fam}_{dev}.jsonl"]
    m[f"perf_moe_qwen3_35a3_shared_{dev}.jsonl"] = [f"perf_moe_qwen3_35a3_shared_{dev}.jsonl"]
    m[f"ref_moe_qwen3_35a3_shared_{dev}.jsonl"] = [f"ref_new_35a3_shared_{dev}.jsonl"]   # the block's comparators (sm_100: torch-native)
    for n in (f"gemm_sm{sm}_qwen3_4b.jsonl", f"gemm_sm{sm}_qwen3_4b_mlp_fwd.jsonl",
              f"gemm_sm{sm}_qwen3_30a3_dense.jsonl", f"gemm_sm{sm}_qwen3_35a3_dense.jsonl"):
        m[n] = [n]
    out = {}
    for name, pats in m.items():
        files = []
        for p in pats:
            files += sorted(glob.glob(os.path.join(run, p)))
        if files:
            out[name] = files
    return out


L2_BYTES = {"5090": 96 * 1024 * 1024, "h200": 60 * 1024 * 1024, "b300": 132644864}   # per --device suffix


def cold_protocol_check(path, dev, mult=2.0):
    """Rows whose weight rotation does not reach `mult` x the device L2 (the cold
    protocol's target, ../../../docs/perf/README.md section 2): a row measured with
    fewer copies than that was served from L2 for part of its replay and is not a
    cold number. Returns (n_timed_rows_with_copies, [short rows]). The 2026-09-28
    copy cap of 16 left the 1-2 MB shared-expert projections warm on every device
    and was found this way on the B300 on 2026-09-29."""
    l2 = L2_BYTES.get(dev)
    if l2 is None:
        return 0, []
    n, short = 0, []
    for r in load(path):
        if r.get("kind") == "meta":
            continue
        if "impl" in r:                                   # MoE row: one dtype, active bytes recorded
            R, b = r.get("weight_copies"), r.get("w_bytes_active")
            if R and b:
                n += 1
                if R * b < mult * l2:
                    short.append((r.get("impl"), r.get("proj"), r.get("M"), R, b))
        elif "N" in r and "K" in r:                       # dense row: one entry per dtype
            for dt in ("bf16", "bsfp8", "mxfp8", "smm_mxfp8"):
                c = r.get(dt)
                if isinstance(c, dict) and c.get("weight_copies") and c.get("us"):
                    n += 1
                    b = r["N"] * r["K"] * (2 if dt == "bf16" else 1)
                    if c["weight_copies"] * b < mult * l2:
                        short.append((r.get("tag"), r.get("M"), dt, c["weight_copies"], b))
    return n, short


def cmd_merge(args):
    out_dir = args.out or args.run
    os.makedirs(out_dir, exist_ok=True)
    for name, files in manifest(args.run, args.device, args.sm).items():
        if name.startswith("gemm_"):
            if os.path.abspath(files[0]) != os.path.abspath(os.path.join(out_dir, name)):
                shutil.copy(files[0], os.path.join(out_dir, name))
            print(f"{name}: copied")
            nc, short = cold_protocol_check(os.path.join(out_dir, name), args.device)
            if short:
                print(f"   COLD-PROTOCOL: {len(short)} of {nc} rows rotate fewer than 2 x L2 of weights (not cold): "
                      + ", ".join(str(x) for x in short[:6]) + (" ..." if len(short) > 6 else ""))
            continue
        n, na = merge_files(os.path.join(out_dir, name), files)
        print(f"{name}: {n} cells from {len(files)} file(s)")
        nc, short = cold_protocol_check(os.path.join(out_dir, name), args.device)
        if short:
            print(f"   COLD-PROTOCOL: {len(short)} of {nc} rows rotate fewer than 2 x L2 of weights (not cold): "
                  + ", ".join(str(x) for x in short[:6]) + (" ..." if len(short) > 6 else ""))
        seen = set()
        for r in na:
            key = (r.get("impl"), r.get("proj"))
            if key in seen:
                continue
            seen.add(key)
            same = [x for x in na if (x.get("impl"), x.get("proj")) == key]
            print(f"   n/a {r.get('impl')} {r.get('proj') or ''} ({len(same)} cells): {(r.get('error') or '')[:110]}")


# ----------------------------------------------------------------------------- cells of any baseline file
def cells_of(path):
    """(key, µs) for every timed cell: MoE rows keyed (impl, proj, M); dense rows
    keyed (tag, M, dtype); MLP-block rows keyed ('mlp', M, dtype)."""
    out = {}
    for r in load(path):
        if r.get("kind") == "meta":
            continue
        if "impl" in r and r.get("us") is not None:
            out[(r["impl"], r.get("proj", "layer"), r["M"])] = r["us"]
        elif "tag" in r and "M" in r:
            for dt, v in r.items():
                if isinstance(v, dict) and v.get("us") is not None:
                    out[(r["tag"], r["M"], dt)] = v["us"]
        elif "M" in r and "hidden" in r:
            for dt, v in r.items():
                if isinstance(v, dict) and v.get("graph_us") is not None:
                    out[("mlp", r["M"], dt)] = v["graph_us"]
    return out


def merged_view(run, dev, sm):
    """A run directory as the canonical files see it: every jsonl copied into a scratch
    directory, then the manifest's merges re-run there. A run directory may carry a
    stale merged file (`ref_kern_moe_*` left by an earlier install of the same
    directory) next to fresher raw files; reading it directly would pair one run's
    fso rows with another run's comparators, so report and diff always go through
    this view. Directories without raw sources (tests/baselines) come back as-is."""
    import tempfile
    m = manifest(run, dev, sm)
    if all(name == os.path.basename(f) for name, files in m.items() for f in files):
        return run                       # nothing to merge: canonical files only
    view = tempfile.mkdtemp(prefix="perf_report_view_")
    for f in glob.glob(os.path.join(run, "*.jsonl")):
        shutil.copy(f, view)
    for name, files in m.items():
        if not name.startswith("gemm_"):
            merge_files(os.path.join(view, name), files)
    return view


def pass_dirs(spec, dev, sm):
    """`--a` / `--b` value -> list of directories, one per pass: 'baselines', or a
    comma-separated list of run directories (an interleaved A/B leaves one
    directory per pass per arm; their cells are averaged, and the spread across
    passes is printed next to every moved cell as the noise tick)."""
    if spec in ("baselines", "installed"):
        return [BASELINES]
    return [merged_view(d, dev, sm) for d in spec.split(",")]


def cells_avg(dirs, name):
    """cell -> mean µs over the passes that have it in every pass, plus the spread
    (max − min) / mean in percent."""
    per = [cells_of(os.path.join(d, name)) for d in dirs if os.path.exists(os.path.join(d, name))]
    if not per:
        return {}, {}
    keys = set(per[0]).intersection(*per[1:]) if len(per) > 1 else set(per[0])
    mean = {k: statistics.fmean(c[k] for c in per) for k in keys}
    spread = {k: (max(c[k] for c in per) - min(c[k] for c in per)) / mean[k] * 100 for k in keys}
    return mean, spread


def cmd_diff(args):
    a_dirs, b_dirs = pass_dirs(args.a, args.device, args.sm), pass_dirs(args.b, args.device, args.sm)
    names = sorted(set().union(*(os.listdir(d) for d in a_dirs)) & set().union(*(os.listdir(d) for d in b_dirs)))
    names = [n for n in names if n.endswith(".jsonl") and (n.startswith("perf_") or n.startswith("gemm_") or n.startswith("ref_"))]
    if not names:
        print("no shared baseline-named files")
    if len(a_dirs) > 1 or len(b_dirs) > 1:
        print(f"passes: A={len(a_dirs)} B={len(b_dirs)} (cell µs averaged over passes; 'spread' = (max-min)/mean over the passes of that arm)")
    for n in names:
        (a, sa), (b, sb) = cells_avg(a_dirs, n), cells_avg(b_dirs, n)
        shared = [k for k in b if k in a]
        if not shared:
            continue
        d = {k: (b[k] / a[k] - 1) * 100 for k in shared}
        xs = list(d.values())
        beyond = sorted(((k, v) for k, v in d.items() if abs(v) > args.threshold), key=lambda kv: -abs(kv[1]))
        print(f"\n{n}: n={len(shared)} median {statistics.median(xs):+.2f}%  min {min(xs):+.1f}%  max {max(xs):+.1f}%  beyond ±{args.threshold}%: {len(beyond)}")
        for name, pred in BANDS[:2]:
            ys = [v for k, v in d.items() if pred(k[2] if isinstance(k[2], int) else k[1])]
            if ys:
                print(f"   {name}: median {statistics.median(ys):+.2f}% (n={len(ys)})")
        for k, v in beyond[: args.show]:
            tick = f"  spread A {sa[k]:.1f}% B {sb[k]:.1f}%" if max(sa[k], sb[k]) > 0 else ""
            print(f"   {k}: {a[k]:.1f} -> {b[k]:.1f}  {v:+.1f}%{tick}")


# ----------------------------------------------------------------------------- reports
def geo(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def layer_report(perf, ref, title):
    rows_perf = load(perf)
    fso_impl = next((i for i in FSO_LAYER_IMPLS if any(r.get("impl") == i for r in rows_perf)), FSO_LAYER_IMPLS[0])
    fso = {r["M"]: r["us"] for r in rows_perf if r.get("impl") == fso_impl and r.get("us")}
    rows = {}
    for r in load(ref):
        if r.get("us") is not None and "impl" in r:
            rows[(r["impl"], r["M"])] = r
    cols = [(i, t) for i, t in LAYER_IMPLS if any(k[0] == i for k in rows)]
    eager = {i for i, _ in cols if any(rows[k].get("timing") == "eager" for k in rows if k[0] == i)}
    out = [f"\n### {title}\n", "|    M | fso µs |" + "".join(f" {t}{' (eager)' if i in eager else ''} µs | ×fso |" for i, t in cols),
           "|-----:|-------:|" + "".join("------:|-----:|" for _ in cols)]
    for M in sorted(fso):
        if M == 96:
            continue
        line = f"| {M:4d} | {fso[M]:6.1f} |"
        for i, _ in cols:
            r = rows.get((i, M))
            line += f" {r['us']:6.1f} | {r['us'] / fso[M]:4.2f} |" if r else "      — |    — |"
        out.append(line)
    out += ["", "| geomean ×fso |" + "".join(f" {t} |" for _, t in cols), "|---|" + "---:|" * len(cols)]
    for name, pred in BANDS:
        xs = {i: [rows[(i, M)]["us"] / fso[M] for M in fso if M != 96 and pred(M) and (i, M) in rows] for i, _ in cols}
        out.append(f"| {name} |" + "".join(f" {geo(xs[i]):.2f}× (n={len(xs[i])}) |" if xs[i] else " — |" for i, _ in cols))
    return "\n".join(out)


def kernel_report(perf, ref, title):
    rows_perf = load(perf)
    fso_impl = next((i for i in FSO_KERN_IMPLS if any(r.get("impl") == i for r in rows_perf)), FSO_KERN_IMPLS[0])
    fso = {(r["proj"], r["M"]): r for r in rows_perf if r.get("impl") == fso_impl and r.get("us")}
    rows = {}
    for r in load(ref):
        if r.get("us") is not None and "impl" in r:
            rows[(r["impl"], r["proj"], r["M"])] = r
    cols = [(i, t) for i, t in KERN_IMPLS if any(k[0] == i for k in rows)]
    out = []
    for proj in ("gate_up", "down"):
        Ms = sorted(M for (p, M) in fso if p == proj and M != 96)
        out += [f"\n### {title} — `moe.{proj}`\n", "|    M | fso MXFP8 µs | cos |" + "".join(f" {t} µs | ×fso | cos |" for _, t in cols),
                "|-----:|-------------:|----:|" + "".join("-----:|-----:|----:|" for _ in cols)]
        for M in Ms:
            f = fso[(proj, M)]
            line = f"| {M:4d} | {f['us']:12.1f} | {f['cos']:.4f} |"
            for i, _ in cols:
                r = rows.get((i, proj, M))
                line += f" {r['us']:8.1f} | {r['us'] / f['us']:4.2f} | {r['cos']:.4f} |" if r else "        — |    — |      — |"
            out.append(line)
        out += ["", "| geomean ×fso |" + "".join(f" {t} |" for _, t in cols), "|---|" + "---:|" * len(cols)]
        for name, pred in BANDS:
            xs = {i: [rows[(i, proj, M)]["us"] / fso[(proj, M)]["us"] for M in Ms if pred(M) and (i, proj, M) in rows] for i, _ in cols}
            out.append(f"| {name} |" + "".join(f" {geo(xs[i]):.2f}× (n={len(xs[i])}) |" if xs[i] else " — |" for i, _ in cols))
    return "\n".join(out)


def cmd_report(args):
    d = merged_view(args.run, args.device, args.sm) if args.run else BASELINES
    dev = args.device
    for fam, title in FAMILIES.items():
        perf, ref, kern = (os.path.join(d, f"perf_moe_qwen3_{fam}_{dev}.jsonl"), os.path.join(d, f"ref_moe_qwen3_{fam}_{dev}.jsonl"),
                           os.path.join(d, f"ref_kern_moe_qwen3_{fam}_{dev}.jsonl"))
        if os.path.exists(perf) and os.path.exists(ref):
            print(layer_report(perf, ref, title))
        if os.path.exists(perf) and os.path.exists(kern):
            print(kernel_report(perf, kern, title.split(" routed")[0]))


def cmd_install(args):
    cmd_merge(args)
    print("\n##### diff: installed baselines -> this run")
    args.a, args.b = "baselines", args.out or args.run
    cmd_diff(args)
    for name in manifest(args.run, args.device, args.sm):
        src = os.path.join(args.out or args.run, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(BASELINES, name))
    subprocess.run([sys.executable, os.path.join(HERE, "render_perf_docs.py")], check=True)
    subprocess.run([sys.executable, os.path.join(HERE, "render_perf_docs.py"), "--check"], check=True)
    print("installed and rendered")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["merge", "diff", "report", "install"])
    ap.add_argument("--run", help="run directory with the raw jsonl files")
    ap.add_argument("--out", help="merge: where to write the canonical files (default: the run dir)")
    ap.add_argument("--a", help="diff: run dir, or 'baselines' for tests/baselines")
    ap.add_argument("--b", help="diff: run dir to compare against --a")
    ap.add_argument("--device", default="5090", help="device suffix of the baseline files (5090 | h200 | b300)")
    ap.add_argument("--sm", default="120", help="SM tag of the dense files (120 | 90 | 100)")
    ap.add_argument("--threshold", type=float, default=1.0, help="diff: list cells moving more than this percent")
    ap.add_argument("--show", type=int, default=40, help="diff: how many moved cells to list")
    args = ap.parse_args()
    if args.cmd in ("merge", "install") and not args.run:
        ap.error("--run is required")
    if args.cmd == "diff" and not (args.a and args.b):
        ap.error("--a and --b are required")
    {"merge": cmd_merge, "diff": cmd_diff, "report": cmd_report, "install": cmd_install}[args.cmd](args)


if __name__ == "__main__":
    main()
