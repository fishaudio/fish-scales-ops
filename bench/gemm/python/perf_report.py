#!/usr/bin/env python3
"""perf_report.py — the one path from a perf run directory to baselines, tables and reports.

A perf run leaves raw jsonl files in a run directory outside the git tree
(5090: /mnt/share/stone-bench-runs/<run>/, H200: /data/bench-runs/<run>/). This
tool turns them into the canonical baseline files, installs them, renders the
docs, and prints the comparison reports — so a rerun after any change is
`merge` → `diff` → (if accepted) `install`, and never a hand-written script.

  python bench/gemm/python/perf_report.py merge   --run <dir> [--out <dir>]      raw files -> canonical baseline files (n/a cells listed)
  python bench/gemm/python/perf_report.py diff    --a <dir|baselines> --b <dir>   cell-by-cell delta of every shared cell (A/B of an optimisation)
  python bench/gemm/python/perf_report.py report  [--run <dir>]                  MLP-block, layer-level and kernel-level comparison tables, ×fso and band geomeans
  python bench/gemm/python/perf_report.py install --run <dir> [--accept-drift]   merge + diff against the installed baselines + copy + provenance + render docs

install takes only a run directory written by bench/run_perf.py: it reads <dir>/manifest.json, takes the device and
the SM tag from it, and refuses a --smoke run always, a run that did not complete, a file whose producing step exited
non-zero or that no step of the manifest wrote, and a run that deviated from its environment lock unless
--accept-drift is given. It merges into
<dir>/merged (the raw files are left as they are), copies the canonical files into tests/baselines/, and records for
each file it installs where it came from in tests/baselines/provenance/<device>.json: the run, its time window, the
machine, card, driver, clock policy and observed clocks, the versions of the environments of the steps that fed it,
the fish-scales-ops version, commit and extension sha256, and the sm_90 JIT compiler. The entries of the files it did
not install are kept, so a partial install (a run of some table groups) leaves the others' provenance alone.
render_perf_docs.py renders those files into the "Environments of record" block of docs/perf/README.md.

Raw-file manifest (device suffix `--device`, families 30a3 / 35a3):
  perf_moe_qwen3_<fam>_<dev>.jsonl            fso rows (fso_mxfp8_grouped + fso_mxfp8_layer; sm_90: fso_bsfp8_layer)  -> same name
  perf_moe_qwen3_35a3_shared_<dev>.jsonl      fso routed + shared block                              -> same name
  ref_new_35a3_shared_<dev>.jsonl             the block's comparators (sm_100: torch-native)          -> ref_moe_qwen3_35a3_shared_<dev>.jsonl
  ref_new_*, ref_fix_*, ref_trt_*, ref_sgl020_*, ref_torchma_*, ref_torchma_smm_*   layer comparators -> ref_moe_qwen3_<fam>_<dev>.jsonl
  ref_kern_c1_vllm_*, ref_kern_c1_sgl_*, ref_kern_c1_fi_*         kernel-level       -> ref_kern_moe_qwen3_<fam>_<dev>.jsonl
  gemm_sm<sm>_qwen3_4b.jsonl, gemm_sm<sm>_qwen3_4b_mlp_fwd.jsonl, gemm_sm<sm>_qwen3_{30a3,35a3}_dense.jsonl   dense tables -> same name
  ref_mlp_sgl_<dev>.jsonl, ref_mlp_vllm_<dev>.jsonl, ref_mlp_cublas_<dev>.jsonl   Family A MLP-block comparators, one
                                              row per M with one entry per dtype, merged by M -> ref_mlp_qwen3_4b_<dev>.jsonl
Bands (stone, 2026-09-28): decode = M 1..128, prefill = M > 128.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
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
PROVENANCE = os.path.join(BASELINES, "provenance")
FAMILIES = {"30a3": "Family B — Qwen3-30B-A3B routed MoE layer (E=128, top-8, moe_inter 768)",
            "35a3": "Family C — Qwen3.5-35B-A3B routed MoE layer (E=256, top-8, moe_inter 512)"}
BANDS = [("decode M 1–128", lambda m: m <= 128), ("prefill M > 128", lambda m: m > 128), ("all", lambda m: True)]
DASH = "—"

# impl -> short column title (layer-level comparators, in report order)
LAYER_IMPLS = [("vllm_fp8b", "vLLM triton FP8-block"), ("triton_fp8b", "sglang triton FP8-block"),
               ("trtllm_gen_fp8b", "TRT-LLM trtllm-gen FP8-block"),
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
# Family A MLP-block comparators (dtype keys of ref_mlp_qwen3_4b_<dev>.jsonl), set against fso's BSFP8 block: the same
# 1x128 activation / 128x128 weight block-FP8 recipe
MLP_IMPLS = [("sgl_fp8b", "sglang block-FP8 linear"), ("vllm_fp8b", "vLLM block-FP8 linear"),
             ("cublas_fp8b", "cuBLAS scaled_mm block-FP8")]


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


def merge_mlp_files(out_path, sources):
    """The Family A MLP-block comparator files, merged by M: one meta row (the first source's `_device` row, with
    every source and its dtypes recorded), then one row per M carrying every source's dtype entries. Entries
    without a graph_us (n/a or failed cells) are kept in the row and returned as (dtype, M, error)."""
    meta, rows, na = None, {}, []
    for src in sources:
        for r in load(src):
            if "M" not in r:
                if meta is None:
                    meta = {k: v for k, v in r.items() if k != "dtypes"}
                    meta["sources"] = []
                meta["sources"].append({"file": os.path.basename(src), "dtypes": r.get("dtypes"), "torch": r.get("torch")})
                continue
            row = rows.setdefault(r["M"], {k: v for k, v in r.items() if not isinstance(v, dict)})
            for dt, v in r.items():
                if isinstance(v, dict):
                    row[dt] = v
                    if v.get("graph_us") is None:
                        na.append((dt, r["M"], v.get("error")))
    if meta is None:
        meta = {"sources": [{"file": os.path.basename(s)} for s in sources]}
    with open(out_path, "w") as f:
        f.write(json.dumps(meta) + "\n")
        for M in sorted(rows):
            f.write(json.dumps(rows[M]) + "\n")
    cells = sum(1 for r in rows.values() for v in r.values() if isinstance(v, dict) and v.get("graph_us") is not None)
    return cells, na


def merge_baseline(name, out_path, sources):
    """Merge the raw files of one baseline file, by the file's kind (MLP-block comparators by M, MoE rows by cell)."""
    return (merge_mlp_files if name.startswith("ref_mlp_") else merge_files)(out_path, sources)


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
    # the Family A MLP-block comparators (bench_qwen3_4b_mlp_forward.py --dtypes), one raw file per library
    m[f"ref_mlp_qwen3_4b_{dev}.jsonl"] = [f"ref_mlp_sgl_{dev}.jsonl", f"ref_mlp_vllm_{dev}.jsonl", f"ref_mlp_cublas_{dev}.jsonl"]
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
        elif "hidden" in r and "intermediate" in r:       # MLP-block row: both projections' weights per dtype
            for dt, c in r.items():
                if isinstance(c, dict) and c.get("weight_copies") and c.get("graph_us"):
                    n += 1
                    b = 3 * r["hidden"] * r["intermediate"] * (2 if dt == "bf16" else 1)
                    if c["weight_copies"] * b < mult * l2:
                        short.append(("mlp", r.get("M"), dt, c["weight_copies"], b))
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
        n, na = merge_baseline(name, os.path.join(out_dir, name), files)
        print(f"{name}: {n} cells from {len(files)} file(s)")
        nc, short = cold_protocol_check(os.path.join(out_dir, name), args.device)
        if short:
            print(f"   COLD-PROTOCOL: {len(short)} of {nc} rows rotate fewer than 2 x L2 of weights (not cold): "
                  + ", ".join(str(x) for x in short[:6]) + (" ..." if len(short) > 6 else ""))
        if name.startswith("ref_mlp_"):
            for dt in sorted({d for d, _, _ in na}):
                same = [x for x in na if x[0] == dt]
                print(f"   n/a {dt} ({len(same)} cells, M {','.join(str(M) for _, M, _ in same)}): {(same[0][2] or '')[:110]}")
            continue
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
            merge_baseline(name, os.path.join(view, name), files)
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


def mlp_report(perf, ref, title):
    """Family A MLP block: each comparator against fso's BSFP8 block (the same block-FP8 recipe), µs, ×fso and cos,
    with the backend each comparator recorded, and the band geomeans."""
    fso = {r["M"]: r["bsfp8"] for r in load(perf) if "M" in r and isinstance(r.get("bsfp8"), dict)
           and r["bsfp8"].get("graph_us") is not None}
    rows = {r["M"]: r for r in load(ref) if "M" in r}
    cols = [(i, t) for i, t in MLP_IMPLS if any((r.get(i) or {}).get("graph_us") is not None for r in rows.values())]
    backend = {i: "/".join(sorted({str(r[i].get("backend")) for r in rows.values()
                                   if (r.get(i) or {}).get("graph_us") is not None})) for i, _ in cols}
    out = [f"\n### {title}\n", "|    M | fso BSFP8 µs |    cos |" + "".join(f" {t} ({backend[i]}) µs | ×fso |    cos |" for i, t in cols),
           "|-----:|------------:|-------:|" + "".join("------:|-----:|-------:|" for _ in cols)]
    for M in sorted(fso):
        line = f"| {M:4d} | {fso[M]['graph_us']:11.2f} | {fso[M]['cos']:.4f} |"
        for i, _ in cols:
            c = (rows.get(M) or {}).get(i) or {}
            line += (f" {c['graph_us']:7.2f} | {c['graph_us'] / fso[M]['graph_us']:4.2f} | {c['cos']:.4f} |"
                     if c.get("graph_us") is not None else "       — |    — |      — |")
        out.append(line)
    out += ["", "| geomean ×fso |" + "".join(f" {t} |" for _, t in cols), "|---|" + "---:|" * len(cols)]
    for name, pred in BANDS:
        xs = {i: [rows[M][i]["graph_us"] / fso[M]["graph_us"] for M in fso if pred(M) and M in rows
                  and (rows[M].get(i) or {}).get("graph_us") is not None] for i, _ in cols}
        out.append(f"| {name} |" + "".join(f" {geo(xs[i]):.2f}× (n={len(xs[i])}) |" if xs[i] else " — |" for i, _ in cols))
    return "\n".join(out)


def cmd_report(args):
    d = merged_view(args.run, args.device, args.sm) if args.run else BASELINES
    dev = args.device
    perf, ref = os.path.join(d, f"gemm_sm{args.sm}_qwen3_4b_mlp_fwd.jsonl"), os.path.join(d, f"ref_mlp_qwen3_4b_{dev}.jsonl")
    if os.path.exists(perf) and os.path.exists(ref):
        print(mlp_report(perf, ref, "Family A — Qwen3-4B dense MLP block, against fso BSFP8 (×fso = comparator µs / fso µs)"))
    for fam, title in FAMILIES.items():
        perf, ref, kern = (os.path.join(d, f"perf_moe_qwen3_{fam}_{dev}.jsonl"), os.path.join(d, f"ref_moe_qwen3_{fam}_{dev}.jsonl"),
                           os.path.join(d, f"ref_kern_moe_qwen3_{fam}_{dev}.jsonl"))
        if os.path.exists(perf) and os.path.exists(ref):
            print(layer_report(perf, ref, title))
        if os.path.exists(perf) and os.path.exists(kern):
            print(kernel_report(perf, kern, title.split(" routed")[0]))


# ----------------------------------------------------------------------------- install: manifest and provenance
def load_run_manifest(run):
    """(manifest dict, sha256 of the file) of a run directory written by bench/run_perf.py, or (None, None)."""
    path = os.path.join(run, "manifest.json")
    if not os.path.isfile(path):
        return None, None
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def feeding_steps(man, raw_files):
    """The manifest's steps whose output is one of a baseline file's raw files."""
    srcs = {os.path.basename(f) for f in raw_files}
    return [s for s in man.get("steps", []) if s.get("out") in srcs]


def install_refusals(man, args, names, device_conflict):
    """Every reason not to install this run, as sentences; an empty list means install may proceed."""
    if man is None:
        return [f"{args.run} has no manifest.json: install takes only run directories written by bench/run_perf.py, "
                "because the baselines' provenance is taken from the manifest"]
    why = []
    if man.get("smoke"):
        why.append(f"it is a --smoke run (M in {{{man.get('smoke_ms')}}} only); a smoke run is never installed")
    if man.get("status") != "complete":
        why.append(f"the run did not complete (status: {man.get('status')!r})")
    if man.get("drift") and not args.accept_drift:
        why.append(f"the run deviated from its environment lock in {len(man['drift'])} item(s) ("
                   + "; ".join(d.get("text", str(d)) for d in man["drift"])
                   + "); --accept-drift installs it anyway and stores the drift in the provenance")
    why += device_conflict
    written = {s.get("out") for s in man.get("steps", [])}
    for name, files in sorted(names.items()):
        bad = [s["name"] for s in feeding_steps(man, files) if s.get("exit_code") != 0]
        if bad:
            why.append(f"{name}: the step(s) that produced it exited non-zero ({', '.join(bad)})")
        foreign = [os.path.basename(f) for f in files if os.path.basename(f) not in written]
        if foreign:
            why.append(f"{name}: no step of the manifest wrote {', '.join(foreign)}, so its provenance is unknown")
    if not names:
        why.append("the run directory holds no raw file of the manifest above")
    return why


def provenance_entry(man, man_sha, files, run):
    """Where one installed baseline file came from, from the run's manifest."""
    steps = feeding_steps(man, files)
    pre_envs = (man.get("preflight") or {}).get("environments") or {}
    lock = man.get("lock") or {}
    fso = man.get("fso") or {}
    cp = man.get("clock_policy") or {}
    policy = cp.get("policy") or (lock.get("gpu") or {}).get("clock_policy") or {}
    envs = {}
    for e in sorted({s["env"] for s in steps}):
        p = pre_envs.get(e) or {}
        envs[e] = {"python": (lock.get("environments") or {}).get(e, {}).get("python"),
                   "python_version": p.get("python_version"), "packages": p.get("packages")}
    tree = man.get("bench_tree") or {}
    return {
        "run": os.path.basename(os.path.normpath(os.path.abspath(run))),
        "run_dir": os.path.abspath(run),
        "manifest_sha256": man_sha,
        "window_utc": ([min(s["start_utc"] for s in steps), max(s["end_utc"] for s in steps)] if steps
                       else [man.get("start_utc"), man.get("end_utc")]),
        "machine": man.get("machine"),
        "host": man.get("host"),
        "card": man.get("card"),
        "driver": (man.get("card") or {}).get("driver"),
        "clock_policy": {"mode": policy.get("mode"), "mhz": policy.get("mhz"), "applied": cp.get("applied"),
                         "max_sm_mhz": cp.get("max_sm_mhz")},
        "observed_clocks": {s["name"]: s.get("clocks") for s in steps},
        "clock_flags": {s["name"]: s["clock_flags"] for s in steps if s.get("clock_flags")},
        "steps": [{k: s.get(k) for k in ("name", "env", "command", "start_utc", "end_utc", "exit_code", "rows",
                                         "error_rows")} for s in steps],
        "environments": envs,
        "bench_env": lock.get("bench_env"),
        "fso": {k: fso.get(k) for k in ("version", "commit", "source_build", "extension_sha256", "build_info")},
        "jit_compiler_sm90": fso.get("jit_compiler_sm90"),
        "bench_tree": {k: tree.get(k) for k in ("source", "commit", "bench_dirty", "fingerprint")},
        "lock_sha256": man.get("lock_sha256"),
        "drift_accepted": man.get("drift") or [],
        "installed_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def write_provenance(dev, entries):
    """Merge `entries` (baseline file name -> provenance) into tests/baselines/provenance/<dev>.json, keeping the
    entries of every other file."""
    os.makedirs(PROVENANCE, exist_ok=True)
    path = os.path.join(PROVENANCE, f"{dev}.json")
    prov = {"device": dev, "files": {}}
    if os.path.isfile(path):
        with open(path) as f:
            prov = json.load(f)
    prov["note"] = ("Written by bench/gemm/python/perf_report.py install from each run's manifest.json and rendered by "
                    "render_perf_docs.py into docs/perf/README.md (Environments of record). Do not edit by hand.")
    prov.setdefault("files", {}).update(entries)
    with open(path, "w") as f:
        json.dump(prov, f, indent=1, sort_keys=True)
        f.write("\n")
    return path


def cmd_install(args):
    man, man_sha = load_run_manifest(args.run)
    conflict = []
    if man is not None:
        mdev, msm = man.get("machine"), str((man.get("lock") or {}).get("sm"))
        if args.device and args.device != mdev:
            conflict.append(f"--device {args.device} contradicts the manifest's machine {mdev}")
        if args.sm and str(args.sm) != msm:
            conflict.append(f"--sm {args.sm} contradicts the manifest's SM tag {msm}")
        args.device, args.sm = mdev, msm
    args.device, args.sm = args.device or "5090", args.sm or "120"
    names = manifest(args.run, args.device, args.sm)
    why = install_refusals(man, args, names, conflict)
    if why:
        print(f"install refused: {args.run}")
        for w in why:
            print(f" - {w}")
        sys.exit(1)
    args.out = args.out or os.path.join(args.run, "merged")
    cmd_merge(args)
    print("\n##### diff: installed baselines -> this run")
    args.a, args.b = "baselines", args.out
    cmd_diff(args)
    entries = {}
    for name, files in names.items():
        src = os.path.join(args.out, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(BASELINES, name))
            entries[name] = provenance_entry(man, man_sha, files, args.run)
    path = write_provenance(args.device, entries)
    print(f"\nprovenance of {len(entries)} file(s) recorded in {os.path.relpath(path, ROOT)}"
          + (f" (drift accepted: {len(man['drift'])} item(s))" if man.get("drift") else ""))
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
    ap.add_argument("--device", default=None,
                    help="device suffix of the baseline files (5090 | h200 | b300); default 5090, and install takes it from the manifest")
    ap.add_argument("--sm", default=None,
                    help="SM tag of the dense files (120 | 90 | 100); default 120, and install takes it from the manifest")
    ap.add_argument("--threshold", type=float, default=1.0, help="diff: list cells moving more than this percent")
    ap.add_argument("--show", type=int, default=40, help="diff: how many moved cells to list")
    ap.add_argument("--accept-drift", action="store_true",
                    help="install: install a run that deviated from its environment lock (the drift goes into the provenance)")
    args = ap.parse_args()
    if args.cmd in ("merge", "install") and not args.run:
        ap.error("--run is required")
    if args.cmd == "diff" and not (args.a and args.b):
        ap.error("--a and --b are required")
    if args.cmd != "install":
        args.device, args.sm = args.device or "5090", args.sm or "120"
    {"merge": cmd_merge, "diff": cmd_diff, "report": cmd_report, "install": cmd_install}[args.cmd](args)


if __name__ == "__main__":
    main()
