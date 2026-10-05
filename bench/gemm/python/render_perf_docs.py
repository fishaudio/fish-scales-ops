#!/usr/bin/env python3
"""Regenerate the fish-scales-ops performance tables from tests/baselines/*.jsonl.

Targets (only the first table after each known heading is rewritten; prose is left alone):
  docs/perf/gemm/sm90.md    Family A / B / C dense sections
  docs/perf/gemm/sm120.md   Family A / B / C dense sections, Family B / C grouped kernel tables
  docs/perf/gemm/sm100.md   same structure as sm120.md, from the B300 baselines
  docs/perf/layer/sm90.md   Family A MLP block, Family B MoE layer, Family C MoE block (+ comparators; Family A's
                            serving-library columns come from ref_mlp_qwen3_4b_<dev>.jsonl and appear once it has rows)
  docs/perf/layer/sm120.md  same for the RTX 5090, plus the grouped GEMM kernel-level comparison tables
  docs/perf/layer/sm100.md  same for the B300 (torch scaled_grouped_mm / _grouped_mm comparators)
  README.md                 the two hot-shape tables (sm_90, sm_120) — no comparisons, by policy
  docs/perf/README.md       the "Environments of record" block between its BEGIN GENERATED / END GENERATED markers:
                            one table per machine, one row per installed baseline file, from
                            tests/baselines/provenance/<device>.json (written by perf_report.py install)

usage: python bench/gemm/python/render_perf_docs.py [--check]
  --check   regenerate into memory and exit 1 if any target would change (CI-style drift check)

Numbers are formatted exactly as the docs print them (dense µs 2 decimals, layer µs 1 decimal, TFLOPS
integer for dense / 1 decimal for layers, GB/s integer, cos 4 decimals, model ms 2 decimals).
Layers per model for `model ms`: A 36, B 48, C 40 (docs/perf/layer/README.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BASE = os.path.join(ROOT, "tests", "baselines")
DASH = "—"

# Per SM version: the suffix the MoE baseline files carry, and a substring every baseline of that
# device records in its meta row. sm_100 and sm_103 share one target (the B300 is sm_103, built
# for the 10.0f family target), so 100 is used as the key for both.
SFX = {90: "h200", 100: "b300", 120: "5090"}
DEVICE = {90: "H200", 100: "B300", 120: "RTX 5090"}


def load(name):
    with open(os.path.join(BASE, name)) as f:
        return [json.loads(l) for l in f if l.strip()]


def check_device(sm, *names):
    """Fail loudly if a baseline file is wired to the wrong SM's document."""
    for n in names:
        path = os.path.join(BASE, n)
        if not os.path.exists(path):
            continue
        meta = load(n)[0]
        got = meta.get("_device") or meta.get("device") or ""
        if got and DEVICE[sm] not in got:
            raise SystemExit(f"{n}: device {got!r} is not the sm_{sm} device ({DEVICE[sm]})")


def dense_rows(name):
    out = {}
    for r in load(name):
        if "tag" in r and "M" in r:
            out.setdefault(r["tag"], {})[r["M"]] = r
    return out


def moe_rows(*names, proj="layer"):
    """(impl, M) -> row for one projection kind ("layer" for whole-layer impls), merged over several jsonl files."""
    out = {}
    for n in names:
        if not os.path.exists(os.path.join(BASE, n)):
            continue
        for r in load(n):
            if "impl" in r and "us" in r and r.get("proj", "layer") == proj:
                out[(r["impl"], r["M"])] = r
    return out


def tflops(M, N, K, us):
    return 2.0 * M * N * K / us / 1e6


def f1(x):
    return DASH if x is None else f"{x:.1f}"


def f2(x):
    return DASH if x is None else f"{x:.2f}"


def f0(x):
    return DASH if x is None else f"{x:.0f}"


def f4(x):
    return DASH if x is None else f"{x:.4f}"


# ----------------------------------------------------------------------------- dense tables
def dense_table_3dtype(rows, N, K, op=None):
    """BF16 / BSFP8 / MXFP8 columns — sm_120 and sm_100/103."""
    hdr = ["| M | BF16 µs | BSFP8 µs | BSFP8 TFLOPS | BSFP8 cos | MXFP8 µs | MXFP8 TFLOPS | MXFP8 cos |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    if op is not None:
        hdr = ["| op | M | BF16 µs | BSFP8 µs | BSFP8 TFLOPS | BSFP8 cos | MXFP8 µs | MXFP8 TFLOPS | MXFP8 cos |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    body = []
    for M in sorted(rows):
        r = rows[M]
        b, s, x = r["bf16"], r["bsfp8"], r["mxfp8"]
        line = (f"| {M} | {f2(b['us'])} | {f2(s['us'])} | {f0(tflops(M, N, K, s['us']))} | {f4(s['cos'])} | "
                f"{f2(x['us'])} | {f0(tflops(M, N, K, x['us']))} | {f4(x['cos'])} |")
        body.append(line if op is None else f"| `{op}` " + line)
    return hdr, body


def dense_table_2dtype(rows, N, K, op=None):
    """BF16 / BSFP8 columns — sm_90 (no MXFP8 path on that arch)."""
    hdr = ["| M | BF16 µs | BSFP8 µs | BSFP8 TFLOPS | cos |", "|---:|---:|---:|---:|---:|"]
    if op is not None:
        hdr = ["| op | M | BF16 µs | BSFP8 µs | BSFP8 TFLOPS | cos |", "|---|---:|---:|---:|---:|---:|"]
    body = []
    for M in sorted(rows):
        r = rows[M]
        b, s = r["bf16"], r["bsfp8"]
        line = f"| {M} | {f2(b['us'])} | {f2(s['us'])} | {f0(tflops(M, N, K, s['us']))} | {f4(s['cos'])} |"
        body.append(line if op is None else f"| `{op}` " + line)
    return hdr, body


def dims(rows):
    r = next(iter(rows.values()))
    return r["N"], r["K"]


# ----------------------------------------------------------------------------- grouped kernel table (sm_120)
def grouped_kernel_table(name):
    """gemm/sm120.md: the per-projection grouped GEMM cells, fso only (rule 2: no comparison columns in gemm/)."""
    hdr = ["| op | M | m_cap | MXFP8 µs | TFLOPS | weight GB/s | cos |", "|---|---:|---:|---:|---:|---:|---:|"]
    body = []
    for proj in ("gate_up", "down"):
        rows = moe_rows(name, proj=proj)
        for (impl, M), r in sorted(rows.items(), key=lambda kv: kv[0][1]):
            if impl == "fso_mxfp8_grouped":
                body.append(f"| `moe.{proj}` | {M} | {r['m_cap']} | {f1(r['us'])} | {f1(r['tflops'])} | {f0(r['w_gbps'])} | {f4(r['cos'])} |")
    return hdr, body


# ----------------------------------------------------------------------------- grouped kernel comparison (layer/sm120.md)
# The Triton block-scaled grouped GEMM that sglang's and vLLM's fused_experts run per
# projection, on moe_align-sorted rows with no routing weight, so the cell is one grouped
# GEMM over exactly the rows the fso cell computes. Read from `ref_kern_<same suffix>`
# next to the fso perf file; rendered into layer/sm120.md, the one place comparisons are
# allowed (docs/README.md rule 2), under its "Grouped GEMM kernel-level comparison" section.
KERN_CMP = [("fi_cudnn_grouped_mxfp8", "FlashInfer cuDNN MXFP8 1×32 µs"),
            ("sgl_triton_grouped_fp8b", "sglang triton FP8 w8a8-block µs"),
            ("vllm_triton_grouped_fp8b", "vLLM triton FP8 w8a8-block µs"),
            ("fi_cudnn_grouped_bf16", "FlashInfer cuDNN BF16 µs"),
            ("sgl_triton_grouped_bf16", "sglang triton BF16 µs"),
            ("vllm_triton_grouped_bf16", "vLLM triton BF16 µs")]


def grouped_kernel_cmp_table(name):
    kern = name.replace("perf_moe_", "ref_kern_moe_", 1)
    cmp = {}
    for r in load(kern):
        if r.get("kind") != "meta" and r.get("us") is not None:
            cmp[(r["impl"], r["proj"], r["M"])] = r
    cols = [c for c in KERN_CMP if any(k[0] == c[0] for k in cmp)]
    hdr = ["| op | M | m_cap | fso MXFP8 µs | TFLOPS | cos |" + "".join(f" {t} |" for _, t in cols),
           "|---|---:|---:|---:|---:|---:|" + "---:|" * len(cols)]
    body = []
    for proj in ("gate_up", "down"):
        rows = moe_rows(name, proj=proj)
        for (impl, M), r in sorted(rows.items(), key=lambda kv: kv[0][1]):
            if impl == "fso_mxfp8_grouped":
                extra = "".join(f" {f1(cmp[(c, proj, M)]['us']) if (c, proj, M) in cmp else DASH} |" for c, _ in cols)
                body.append(f"| `moe.{proj}` | {M} | {r['m_cap']} | {f1(r['us'])} | {f1(r['tflops'])} | {f4(r['cos'])} |" + extra)
    return hdr, body


# ----------------------------------------------------------------------------- layer tables
# Family A MLP-block comparators (docs/README.md rule 2: layer/ only): the serving libraries' block-FP8 dense linear
# and cuBLAS's, read from ref_mlp_qwen3_4b_<dev>.jsonl (perf_report.py merges one raw file per library into it) and
# joined to the fso rows by M. A comparator's columns appear once the file has a timed cell of it, so the table
# renders unchanged until then; its title names the GEMM backend its cells recorded; ×fso BSFP8 is its µs over fso's
# BSFP8 µs (the same 1x128 / 128x128 block-FP8 recipe), so above 1 is fso ahead.
MLP_CMP = [("sgl_fp8b", "sglang block-FP8 linear"), ("vllm_fp8b", "vLLM block-FP8 linear"),
           ("cublas_fp8b", "cuBLAS scaled_mm block-FP8")]


def mlp_cmp_columns(ref_name):
    """(M -> comparator row, [(dtype, column title)]) of the MLP-block comparator file, or ({}, []) without one."""
    if not ref_name or not os.path.exists(os.path.join(BASE, ref_name)):
        return {}, []
    rows = {r["M"]: r for r in load(ref_name) if "M" in r}
    cols = []
    for dt, title in MLP_CMP:
        timed = [r[dt] for r in rows.values() if isinstance(r.get(dt), dict) and r[dt].get("graph_us") is not None]
        if timed:
            cols.append((dt, f"{title} ({'/'.join(sorted({str(c.get('backend')) for c in timed}))}) µs"))
    return rows, cols


def mlp_table(name, sm, ref_name=None):
    rows = {r["M"]: r for r in load(name) if "M" in r}
    cmp_rows, cmp_cols = mlp_cmp_columns(ref_name)
    if sm != 90:
        hdr = ["| M | BF16 (torch) µs | BSFP8 µs | BSFP8 TFLOPS | BSFP8 cos | model ms | MXFP8 µs | MXFP8 TFLOPS | MXFP8 cos | model ms | "
               "cuBLAS scaled_mm MXFP8 + reference quantize µs | cuBLAS scaled_mm MXFP8 + torch.compile quantize µs |",
               "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    else:
        hdr = ["| M | BF16 (torch) µs | BSFP8 µs | BSFP8 TFLOPS | BSFP8 cos | model ms |", "|---:|---:|---:|---:|---:|---:|"]
    if cmp_cols:
        hdr = [hdr[0] + "".join(f" {t} | ×fso BSFP8 |" for _, t in cmp_cols), hdr[1] + "---:|---:|" * len(cmp_cols)]
    body = []
    for M in sorted(rows):
        r = rows[M]
        H, I = r["hidden"], r["intermediate"]
        fl = 2.0 * M * (2 * I * H + H * I)
        b, s = r["bf16"], r["bsfp8"]
        line = f"| {M} | {f2(b['graph_us'])} | {f2(s['graph_us'])} | {f0(fl / s['graph_us'] / 1e6)} | {f4(s['cos'])} | {f2(s['graph_us'] * 36 / 1000)} |"
        if sm != 90:
            x, c, cf = r["mxfp8"], r["smm"], r["smm_fast"]
            line += (f" {f2(x['graph_us'])} | {f0(fl / x['graph_us'] / 1e6)} | {f4(x['cos'])} | {f2(x['graph_us'] * 36 / 1000)} | "
                     f"{f2(c['graph_us'])} | {f2(cf['graph_us'])} |")
        for dt, _ in cmp_cols:
            us, fso_us = ((cmp_rows.get(M) or {}).get(dt) or {}).get("graph_us"), s.get("graph_us")
            line += f" {f2(us)} | {f2(us / fso_us if us is not None and fso_us else None)} |"
        body.append(line)
    return hdr, body


def moe_layer_table(fso, cmp, fso_impl, sm):
    """Family B whole-layer table. cmp: list of (impl, column title)."""
    Ms = sorted({M for (impl, M) in list(fso) + list(cmp_rows_keys(cmp))})
    pathcol = "path" if sm == 90 else "m_cap"
    hdr = [f"| M | active experts | {pathcol} | fso block µs | TFLOPS | weight GB/s | cos | model ms | " + " | ".join(t for _, t in cmp) + " |",
           ("|---:|---:|:---|---:|---:|---:|---:|---:|" if sm == 90 else "|---:|---:|---:|---:|---:|---:|---:|---:|") + "---:|" * len(cmp)]
    body = []
    for M in Ms:
        r = fso.get((fso_impl, M))
        anyrow = r or next((cmp_rows_get(c, M) for c, _ in cmp if cmp_rows_get(c, M)), None)
        ae = anyrow["active_experts"] if anyrow and "active_experts" in anyrow else DASH
        if r is None:
            cells = [DASH] * 6
        else:
            path = ("swap-AB contiguous" if r.get("swap") else "contiguous") if sm == 90 else str(r["m_cap"])
            cells = [path, f1(r["us"]), f1(r["tflops"]), f0(r["w_gbps"]), f4(r["cos"]), f2(r["model_ms"])]
        comps = [f1(cmp_rows_get(c, M)["us"]) if cmp_rows_get(c, M) else DASH for c, _ in cmp]
        body.append(f"| {M} | {ae} | " + " | ".join(cells + comps) + " |")
    return hdr, body


_CMP_ROWS = {}


def set_cmp_rows(rows):
    _CMP_ROWS.clear()
    _CMP_ROWS.update(rows)


def cmp_rows_get(impl, M):
    return _CMP_ROWS.get((impl, M))


def cmp_rows_keys(cmp):
    return [k for k in _CMP_ROWS if k[0] in {c for c, _ in cmp}]


def moe_block_table(fso, routed_impl, shared_impl, cmp, cmp_suffix=" (routed)"):
    """Family C: routed / routed + shared / comparators.

    `cmp_suffix` is appended to every comparator column title; pass "" when the caller's titles
    already say which block each comparator ran (the B300 has both routed and routed + shared).
    """
    Ms = sorted({M for (impl, M) in list(fso) + list(cmp_rows_keys(cmp))})
    hdr = ["| M | active experts | fso routed µs | fso routed + shared µs | shared expert µs | TFLOPS (block) | weight GB/s (block) | cos (block) | model ms (block) | "
           + " | ".join(f"{t}{cmp_suffix}" for _, t in cmp) + " |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|---:|" + "---:|" * len(cmp)]
    body = []
    for M in Ms:
        r, s = fso.get((routed_impl, M)), fso.get((shared_impl, M))
        anyrow = r or s or next((cmp_rows_get(c, M) for c, _ in cmp if cmp_rows_get(c, M)), None)
        ae = anyrow["active_experts"] if anyrow and "active_experts" in anyrow else DASH
        if r is None or s is None:
            cells = [DASH] * 7
        else:
            cells = [f1(r["us"]), f1(s["us"]), f1(s["us"] - r["us"]), f1(s["tflops"]), f0(s["w_gbps"]), f4(s["cos"]), f2(s["model_ms"])]
        comps = [f1(cmp_rows_get(c, M)["us"]) if cmp_rows_get(c, M) else DASH for c, _ in cmp]
        body.append(f"| {M} | {ae} | " + " | ".join(cells + comps) + " |")
    return hdr, body


# ----------------------------------------------------------------------------- document surgery
def align(table):
    """Pad every cell to its column width so the markdown source reads as a table.

    Renderers elsewhere leave the source ragged, which is valid markdown but makes
    a 10-column perf table unreadable in a diff or an editor. The separator row's
    markers decide the direction: `---:` is a right-aligned column, so its cells
    are padded on the left, everything else on the right. Rows with a different
    cell count than the header (there are none today) are left untouched rather
    than silently reflowed.
    """
    split = [[c.strip() for c in row.strip().strip("|").split("|")] for row in table]
    ncol = len(split[0])
    if any(len(r) != ncol for r in split):
        return table
    sep = split[1] if len(split) > 1 else []
    right = [i < len(sep) and sep[i].endswith(":") and not sep[i].startswith(":") for i in range(ncol)]
    # The separator cells themselves are dashes; they stretch to the column width.
    width = [max(len(r[i]) for j, r in enumerate(split) if j != 1) for i in range(ncol)]
    out = []
    for j, row in enumerate(split):
        cells = []
        for i, c in enumerate(row):
            w = width[i]
            if j == 1:
                cells.append(("-" * (w + 1) + ":") if right[i] else ("-" * (w + 2)))
            else:
                cells.append(" " + (c.rjust(w) if right[i] else c.ljust(w)) + " ")
        out.append("|" + "|".join(cells) + "|")
    return out


def replace_table_after(lines, pred, table, start=0):
    h = next(i for i in range(start, len(lines)) if pred(lines[i]))
    t0 = next(i for i in range(h + 1, len(lines)) if lines[i].startswith("|"))
    t1 = t0
    while t1 < len(lines) and lines[t1].startswith("|"):
        t1 += 1
    table = align(table)
    lines[t0:t1] = table
    return t0 + len(table)


def starts(prefix):
    return lambda s, p=prefix: s.startswith(p)


def render_gemm_3dtype(lines, sm):
    """gemm/sm120.md and gemm/sm100.md — identical section structure, different baselines."""
    sfx = SFX[sm]
    check_device(sm, f"gemm_sm{sm}_qwen3_4b.jsonl", f"gemm_sm{sm}_qwen3_30a3_dense.jsonl",
                 f"gemm_sm{sm}_qwen3_35a3_dense.jsonl", f"perf_moe_qwen3_30a3_{sfx}.jsonl",
                 f"perf_moe_qwen3_35a3_{sfx}.jsonl")
    A = dense_rows(f"gemm_sm{sm}_qwen3_4b.jsonl")
    B = dense_rows(f"gemm_sm{sm}_qwen3_30a3_dense.jsonl")
    C = dense_rows(f"gemm_sm{sm}_qwen3_35a3_dense.jsonl")
    pos = 0
    for tag in ("wqkv", "wo", "gate_up", "down"):
        N, K = dims(A[tag])
        h, b = dense_table_3dtype({m: r for m, r in A[tag].items() if m <= 8192}, N, K)
        pos = replace_table_after(lines, starts(f"### `{tag}` (N="), h + b, pos)
    body = []
    for tag in ("wqkv", "wo"):
        N, K = dims(B[tag])
        body += dense_table_3dtype(B[tag], N, K, op=tag)[1]
    h, _ = dense_table_3dtype({}, 0, 0, op="x")
    pos = replace_table_after(lines, starts("### Dense projections `wqkv` (N=5120, K=2048), `wo` (N=2048, K=4096)"), h + body, pos)
    hB, bB = grouped_kernel_table(f"perf_moe_qwen3_30a3_{sfx}.jsonl")
    pos = replace_table_after(lines, starts("### Grouped GEMM kernels `moe.gate_up` (N=1536, K=2048), `moe.down` (N=2048, K=768)"), hB + bB, pos)
    for tag, head in (("wqkv_gated", "#### `wqkv_gated`"), ("wo", "#### `wo`"), ("gdn_in_proj", "#### `gdn.in_proj`"), ("gdn_out_proj", "#### `gdn.out_proj`")):
        N, K = dims(C[tag])
        h, b = dense_table_3dtype(C[tag], N, K)
        pos = replace_table_after(lines, starts(head), h + b, pos)
    hC, bC = grouped_kernel_table(f"perf_moe_qwen3_35a3_{sfx}.jsonl")
    pos = replace_table_after(lines, starts("### Grouped GEMM kernels `moe.gate_up` (N=1024, K=2048), `moe.down` (N=2048, K=512)"), hC + bC, pos)
    for tag, head in (("shared_gate_up", "#### `shared.gate_up`"), ("shared_down", "#### `shared.down`")):
        N, K = dims(C[tag])
        h, b = dense_table_3dtype(C[tag], N, K)
        pos = replace_table_after(lines, starts(head), h + b, pos)


def render_gemm_sm90(lines):
    check_device(90, "gemm_sm90_qwen3_4b.jsonl", "gemm_sm90_qwen3_30a3_dense.jsonl", "gemm_sm90_qwen3_35a3_dense.jsonl")
    A = dense_rows("gemm_sm90_qwen3_4b.jsonl")
    B = dense_rows("gemm_sm90_qwen3_30a3_dense.jsonl")
    C = dense_rows("gemm_sm90_qwen3_35a3_dense.jsonl")
    pos = 0
    for tag in ("wqkv", "wo", "gate_up", "down"):
        N, K = dims(A[tag])
        h, b = dense_table_2dtype({m: r for m, r in A[tag].items() if m <= 8192}, N, K)
        pos = replace_table_after(lines, starts(f"### `{tag}` (N="), h + b, pos)
    body = []
    for tag in ("wqkv", "wo"):
        N, K = dims(B[tag])
        body += dense_table_2dtype(B[tag], N, K, op=tag)[1]
    h, _ = dense_table_2dtype({}, 0, 0, op="x")
    pos = replace_table_after(lines, starts("### Dense projections `wqkv` (N=5120, K=2048), `wo` (N=2048, K=4096)"), h + body, pos)
    for tag, head in (("wqkv_gated", "#### `wqkv_gated`"), ("wo", "#### `wo`"), ("gdn_in_proj", "#### `gdn.in_proj`"), ("gdn_out_proj", "#### `gdn.out_proj`"),
                      ("shared_gate_up", "#### `shared.gate_up`"), ("shared_down", "#### `shared.down`")):
        N, K = dims(C[tag])
        h, b = dense_table_2dtype(C[tag], N, K)
        pos = replace_table_after(lines, starts(head), h + b, pos)


# Comparator columns per SM, allowed only in docs/perf/layer/ (docs/README.md rule 2). The sm_90 and
# sm_120 hosts have sglang (and deep_gemm on sm_90); the B300 pod has neither, so its comparators are
# the two grouped entry points torch 2.11 itself provides.
def layer_comparators(sm, shared=False):
    if sm == 100:
        sfxs = "_shared" if shared else ""
        where = " (routed + shared)" if shared else " (routed)"
        return [(f"torch_smm_mxfp8_layer{sfxs}", f"torch scaled_grouped_mm MXFP8 µs{where}"),
                (f"torch_grouped_bf16_layer{sfxs}", f"torch _grouped_mm BF16 µs{where}")]
    cmp = [("triton_bf16", "sglang triton BF16 µs"), ("triton_fp8b", "sglang triton FP8 w8a8-block µs")]
    if sm == 90:
        cmp.append(("dg_fp8_layer", "deep_gemm masked pipeline FP8 µs"))
    if sm == 120:
        # 2026-09-28, torch 2.13 base environment: the same-device MoE implementations a
        # torch or vLLM user gets on this card. vLLM's triton fused_experts (no tuned json
        # for the RTX 5090 at these shapes, so its default tile config; stated in the
        # provenance) and torch's own _grouped_mm, which has no fused sm_120 kernel and
        # runs a host loop over experts -- it cannot be captured, so its cells are eager
        # timings, and the whole-layer torch.compile form is shown beside it as the dense
        # tables show the compiled quantize. Under torch.compile(mode="max-autotune-no-cudagraphs")
        # Inductor replaces the host loop with its Triton grouped-GEMM template, so that
        # form captures and its cells are graph-replay timings like every other column.
        # TensorRT-LLM's CUTLASS fused MoE, JIT-built for sm_120 by the FlashInfer
        # wheel (the `tensorrt_llm` wheel pins torch <= 2.10): BF16 on its Ampere
        # kernels, per-tensor FP8 on its SM89 fallback kernels; tactics autotuned.
        cmp = [("vllm_bf16", "vLLM triton BF16 µs"), ("vllm_fp8b", "vLLM triton FP8 w8a8-block µs"),
               ("trtllm_cutlass_bf16", "TRT-LLM CUTLASS fused MoE BF16 µs"),
               ("trtllm_cutlass_fp8", "TRT-LLM CUTLASS fused MoE FP8 per-tensor µs")] + cmp + [
            ("torch_grouped_bf16_layer", "torch _grouped_mm BF16 µs (eager)"),
            ("torch_grouped_bf16_layer_compiled", "torch _grouped_mm BF16 + torch.compile µs (eager)"),
            ("torch_grouped_bf16_layer_maxautotune", "torch _grouped_mm BF16 + torch.compile max-autotune µs")]
    return cmp


def render_layer(lines, sm):
    sfx = SFX[sm]
    dt = "bsfp8" if sm == 90 else "mxfp8"
    check_device(sm, f"gemm_sm{sm}_qwen3_4b_mlp_fwd.jsonl", f"ref_mlp_qwen3_4b_{sfx}.jsonl", f"perf_moe_qwen3_30a3_{sfx}.jsonl",
                 f"ref_moe_qwen3_30a3_{sfx}.jsonl", f"perf_moe_qwen3_35a3_{sfx}.jsonl",
                 f"perf_moe_qwen3_35a3_shared_{sfx}.jsonl", f"ref_moe_qwen3_35a3_{sfx}.jsonl",
                 f"ref_moe_qwen3_35a3_shared_{sfx}.jsonl", f"ref_kern_moe_qwen3_30a3_{sfx}.jsonl",
                 f"ref_kern_moe_qwen3_35a3_{sfx}.jsonl")
    h, b = mlp_table(f"gemm_sm{sm}_qwen3_4b_mlp_fwd.jsonl", sm, f"ref_mlp_qwen3_4b_{sfx}.jsonl")
    pos = replace_table_after(lines, starts("## Family A — Qwen3-4B dense MLP block"), h + b)
    fsoB = moe_rows(f"perf_moe_qwen3_30a3_{sfx}.jsonl")
    set_cmp_rows({**moe_rows(f"perf_moe_qwen3_30a3_{sfx}.jsonl"), **moe_rows(f"ref_moe_qwen3_30a3_{sfx}.jsonl")})
    cmpB = layer_comparators(sm)
    h, b = moe_layer_table(fsoB, cmpB, f"fso_{dt}_layer", sm)
    pos = replace_table_after(lines, starts("## Family B — Qwen3-30B-A3B routed MoE layer"), h + b, pos)
    fsoC = moe_rows(f"perf_moe_qwen3_35a3_{sfx}.jsonl", f"perf_moe_qwen3_35a3_shared_{sfx}.jsonl")
    # The B300 comparator run measured the shared-expert block as well; sm_90 / sm_120 have no such
    # file, and moe_rows() skips the ones that do not exist.
    set_cmp_rows(moe_rows(f"ref_moe_qwen3_35a3_{sfx}.jsonl", f"ref_moe_qwen3_35a3_shared_{sfx}.jsonl"))
    if sm == 100:
        cmpC = layer_comparators(sm) + layer_comparators(sm, shared=True)
        h, b = moe_block_table(fsoC, f"fso_{dt}_layer", f"fso_{dt}_layer_shared", cmpC, cmp_suffix="")
    else:
        h, b = moe_block_table(fsoC, f"fso_{dt}_layer", f"fso_{dt}_layer_shared", layer_comparators(sm))
    pos = replace_table_after(lines, starts("## Family C — Qwen3.5-35B-A3B MoE block"), h + b, pos)
    for fam, letter in (("30a3", "B"), ("35a3", "C")):
        name = f"perf_moe_qwen3_{fam}_{sfx}.jsonl"
        if os.path.exists(os.path.join(BASE, name.replace("perf_moe_", "ref_kern_moe_", 1))):
            h, b = grouped_kernel_cmp_table(name)
            pos = replace_table_after(lines, starts(f"### Family {letter} — `moe.gate_up` / `moe.down` against the Triton grouped GEMM"), h + b, pos)


# ----------------------------------------------------------------------------- README hot rows (no comparisons)
def readme_rows(sm):
    sfx = SFX[sm]
    A = dense_rows(f"gemm_sm{sm}_qwen3_4b.jsonl")["gate_up"]
    N, K = dims(A)
    rows = []
    if sm == 120:
        for M in (1, 4096):
            x = A[M]["mxfp8"]
            rows.append(f"| A Qwen3-4B | `gate_up` | {N}×{K} | M={M} | MXFP8 1×32 | {f2(x['us'])} | {f0(tflops(M, N, K, x['us']))} |")
        s = A[4096]["bsfp8"]
        rows.append(f"| A Qwen3-4B | `gate_up` | {N}×{K} | M=4096 | block-FP8 1×128 | {f2(s['us'])} | {f0(tflops(4096, N, K, s['us']))} |")
        dtB = "MXFP8 1×32"
    else:
        for M in (1, 4096):
            s = A[M]["bsfp8"]
            rows.append(f"| A Qwen3-4B | `gate_up` | {N}×{K} | M={M} | block-FP8 1×128 | {f2(s['us'])} | {f0(tflops(M, N, K, s['us']))} |")
        dtB = "block-FP8 1×128"
    dt = "mxfp8" if sm == 120 else "bsfp8"
    B = moe_rows(f"perf_moe_qwen3_30a3_{sfx}.jsonl")
    for M in (1, 2048):
        r = B[(f"fso_{dt}_layer", M)]
        rows.append(f"| B Qwen3-30B-A3B | MoE layer (routed, E=128, top-8) | 1536×2048 + 2048×768 per expert | M={M} | {dtB} | {f1(r['us'])} | {f1(r['tflops'])} |")
    C = moe_rows(f"perf_moe_qwen3_35a3_shared_{sfx}.jsonl")
    for M in (1, 2048):
        r = C[(f"fso_{dt}_layer_shared", M)]
        rows.append(f"| C Qwen3.5-35B-A3B | MoE block (routed E=256 top-8 + shared expert) | 1024×2048 + 2048×512 per expert, shared 1024×2048 + 2048×512 | M={M} | {dtB} | {f1(r['us'])} | {f1(r['tflops'])} |")
    return rows


def render_readme(lines):
    pos = replace_table_after(lines, starts("### sm_90 — NVIDIA H200"), ["| family | op | shape | M / S | dtype | µs | TFLOPS |", "|---|---|---|---|---|---:|---:|"] + readme_rows(90))
    replace_table_after(lines, starts("### sm_120 — NVIDIA RTX 5090"), ["| family | op | shape | M / S | dtype | µs | TFLOPS |", "|---|---|---|---|---|---:|---:|"] + readme_rows(120), pos)


# ----------------------------------------------------------------------------- docs/perf/README.md: environments of record
PROV = os.path.join(BASE, "provenance")
ENV_BEGIN = "<!-- BEGIN GENERATED: environments of record (bench/gemm/python/render_perf_docs.py) -->"
ENV_END = "<!-- END GENERATED: environments of record -->"
# (device suffix of the baseline files, SM tag of the dense files, heading)
ENV_MACHINES = [("h200", 90, "H200 (sm_90)"), ("5090", 120, "RTX 5090 (sm_120)"),
                ("b300", 100, "B300 (sm_103; the dense files are tagged sm100)")]
NO_MANIFEST = "installed before the harness (no manifest)"


def device_baselines(sfx, sm):
    return sorted(n for n in os.listdir(BASE)
                  if n.endswith(".jsonl") and (n.endswith(f"_{sfx}.jsonl") or n.startswith(f"gemm_sm{sm}_")))


def _window(w):
    """['2026-10-01T10:29:03.123Z', '2026-10-01T10:41:57.456Z'] -> '2026-10-01 10:29–10:41'."""
    if not w or not w[0] or not w[1]:
        return DASH
    (d0, t0), (d1, t1) = (x.replace("Z", "").split("T") for x in w)
    return f"{d0} {t0[:5]}–{t1[:5]}" if d0 == d1 else f"{d0} {t0[:5]} – {d1} {t1[:5]}"


def _clocks(p):
    cp = p.get("clock_policy") or {}
    head = f"locked at {cp.get('mhz')} MHz" if cp.get("mode") == "locked" else (cp.get("mode") or "?") + " clock"
    obs = [c for c in (p.get("observed_clocks") or {}).values() if c and c.get("sm_mhz_median_busy") is not None]
    if not obs:
        return head
    med = sorted(c["sm_mhz_median_busy"] for c in obs)
    lo = min(c["sm_mhz_min_busy"] for c in obs)
    cap = max(c.get("power_capped_share_of_busy") or 0 for c in obs)
    med_s = f"{med[0]:g}" if med[0] == med[-1] else f"{med[0]:g}–{med[-1]:g}"
    flags = sum(len(v) for v in (p.get("clock_flags") or {}).values())
    return (f"{head}; busy median {med_s} MHz, min {lo:g} MHz, power-capped in up to {cap * 100:.0f} % of busy samples"
            + (f"; {flags} clock flag(s) in the manifest" if flags else ""))


def _environments(p):
    out = []
    for name, e in sorted((p.get("environments") or {}).items()):
        pk = ", ".join(f"{k} {v}" for k, v in sorted((e.get("packages") or {}).items()))
        out.append(f"{name}: Python {e.get('python_version')}, {pk}")
    return "; ".join(out) or DASH


def _fso(p):
    f = p.get("fso") or {}
    sha = (f.get("extension_sha256") or "")[:12]
    origin = f"commit {f['commit'][:12]}" if f.get("commit") else ("source build, no commit recorded" if f.get("source_build") else "commit unknown")
    if isinstance(f.get("build_info"), dict) and f["build_info"].get("dirty"):
        origin += " (built from a modified tree)"
    return f"{f.get('version') or '?'}, {origin}, extension sha256 {sha or '?'}…"


def env_block_lines():
    out = ["### Environments of record", "",
           "Generated by `bench/gemm/python/render_perf_docs.py` from `tests/baselines/provenance/<device>.json`, which "
           "`perf_report.py install` writes from the `manifest.json` of the run it installs; do not edit it by hand. "
           "One row per installed baseline file: the run that produced it, its time window, the card, driver and "
           "clocks, the package versions of the environments its steps ran in, and the fish-scales-ops build.", ""]
    for sfx, sm, title in ENV_MACHINES:
        prov = {}
        path = os.path.join(PROV, f"{sfx}.json")
        if os.path.isfile(path):
            with open(path) as f:
                prov = json.load(f).get("files", {})
        cols = ["baseline file", "run", "window (UTC)", "card", "driver", "clocks", "environments", "fish-scales-ops"]
        if sm == 90:
            cols.append("sm_90 JIT compiler")
        table = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for name in device_baselines(sfx, sm):
            p = prov.get(name)
            if p is None:
                row = [f"`{name}`", NO_MANIFEST] + [DASH] * (len(cols) - 2)
            else:
                c = p.get("card") or {}
                run = f"`{p.get('run')}`"
                if p.get("drift_accepted"):
                    run += " (drift accepted: " + "; ".join(d.get("text", str(d)) for d in p["drift_accepted"]) + ")"
                row = [f"`{name}`", run, _window(p.get("window_utc")),
                       f"{c.get('name')}, PCI {c.get('pci')}, {c.get('uuid')}", p.get("driver") or DASH,
                       _clocks(p), _environments(p), _fso(p)]
                if sm == 90:
                    row.append(p.get("jit_compiler_sm90") or DASH)
            table.append("| " + " | ".join(str(x).replace("|", "/") for x in row) + " |")
        out += [f"#### {title}", ""] + align(table) + [""]
    return out


def render_env_block(lines):
    b = next((i for i, l in enumerate(lines) if l.strip() == ENV_BEGIN), None)
    e = next((i for i in range(b + 1, len(lines)) if lines[i].strip() == ENV_END), None) if b is not None else None
    if b is None or e is None:
        raise SystemExit(f"docs/perf/README.md: the markers {ENV_BEGIN!r} and {ENV_END!r} are missing")
    lines[b + 1:e] = env_block_lines()


TARGETS = {
    "docs/perf/gemm/sm120.md": lambda L: render_gemm_3dtype(L, 120),
    "docs/perf/gemm/sm90.md": render_gemm_sm90,
    "docs/perf/gemm/sm100.md": lambda L: render_gemm_3dtype(L, 100),
    "docs/perf/layer/sm120.md": lambda L: render_layer(L, 120),
    "docs/perf/layer/sm90.md": lambda L: render_layer(L, 90),
    "docs/perf/layer/sm100.md": lambda L: render_layer(L, 100),
    "README.md": render_readme,
    "docs/perf/README.md": render_env_block,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if any target would change")
    ap.add_argument("--only", nargs="*", default=None, help="subset of targets")
    args = ap.parse_args()
    changed = []
    for rel, fn in TARGETS.items():
        if args.only and rel not in args.only:
            continue
        path = os.path.join(ROOT, rel)
        old = open(path).read()
        lines = old.split("\n")
        fn(lines)
        new = "\n".join(lines)
        if new != old:
            changed.append(rel)
            if not args.check:
                open(path, "w").write(new)
    if args.check:
        print("would change:" if changed else "up to date:", ", ".join(changed) or "all targets")
        sys.exit(1 if changed else 0)
    print("rewrote:", ", ".join(changed) if changed else "nothing (all tables already match the baselines)")


if __name__ == "__main__":
    main()
