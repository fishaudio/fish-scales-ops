#!/usr/bin/env python3
"""Regenerate the fish-scales-ops performance tables from tests/baselines/*.jsonl.

Targets (only the first table after each known heading is rewritten; prose is left alone):
  docs/perf/gemm/sm90.md    Family A / B / C dense sections
  docs/perf/gemm/sm120.md   Family A / B / C dense sections, Family B / C grouped kernel tables
  docs/perf/gemm/sm100.md   same structure as sm120.md, from the B300 baselines
  docs/perf/layer/sm90.md   Family A MLP block, Family B MoE layer, Family C MoE block (+ comparators)
  docs/perf/layer/sm120.md  same for the RTX 5090
  docs/perf/layer/sm100.md  same for the B300 (torch scaled_grouped_mm / _grouped_mm comparators)
  README.md                 the two hot-shape tables (sm_90, sm_120) — no comparisons, by policy

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
    hdr = ["| op | M | m_cap | MXFP8 µs | TFLOPS | weight GB/s | cos |", "|---|---:|---:|---:|---:|---:|---:|"]
    body = []
    for proj in ("gate_up", "down"):
        rows = moe_rows(name, proj=proj)
        for (impl, M), r in sorted(rows.items(), key=lambda kv: kv[0][1]):
            if impl == "fso_mxfp8_grouped":
                body.append(f"| `moe.{proj}` | {M} | {r['m_cap']} | {f1(r['us'])} | {f1(r['tflops'])} | {f0(r['w_gbps'])} | {f4(r['cos'])} |")
    return hdr, body


# ----------------------------------------------------------------------------- layer tables
def mlp_table(name, sm):
    rows = {r["M"]: r for r in load(name) if "M" in r}
    if sm != 90:
        hdr = ["| M | BF16 (torch) µs | BSFP8 µs | BSFP8 TFLOPS | BSFP8 cos | model ms | MXFP8 µs | MXFP8 TFLOPS | MXFP8 cos | model ms | "
               "cuBLAS scaled_mm MXFP8 + reference quantize µs | cuBLAS scaled_mm MXFP8 + torch.compile quantize µs |",
               "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    else:
        hdr = ["| M | BF16 (torch) µs | BSFP8 µs | BSFP8 TFLOPS | BSFP8 cos | model ms |", "|---:|---:|---:|---:|---:|---:|"]
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
def replace_table_after(lines, pred, table, start=0):
    h = next(i for i in range(start, len(lines)) if pred(lines[i]))
    t0 = next(i for i in range(h + 1, len(lines)) if lines[i].startswith("|"))
    t1 = t0
    while t1 < len(lines) and lines[t1].startswith("|"):
        t1 += 1
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
    return cmp


def render_layer(lines, sm):
    sfx = SFX[sm]
    dt = "bsfp8" if sm == 90 else "mxfp8"
    check_device(sm, f"gemm_sm{sm}_qwen3_4b_mlp_fwd.jsonl", f"perf_moe_qwen3_30a3_{sfx}.jsonl",
                 f"ref_moe_qwen3_30a3_{sfx}.jsonl", f"perf_moe_qwen3_35a3_{sfx}.jsonl",
                 f"perf_moe_qwen3_35a3_shared_{sfx}.jsonl", f"ref_moe_qwen3_35a3_{sfx}.jsonl",
                 f"ref_moe_qwen3_35a3_shared_{sfx}.jsonl")
    h, b = mlp_table(f"gemm_sm{sm}_qwen3_4b_mlp_fwd.jsonl", sm)
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
    replace_table_after(lines, starts("## Family C — Qwen3.5-35B-A3B MoE block"), h + b, pos)


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


TARGETS = {
    "docs/perf/gemm/sm120.md": lambda L: render_gemm_3dtype(L, 120),
    "docs/perf/gemm/sm90.md": render_gemm_sm90,
    "docs/perf/gemm/sm100.md": lambda L: render_gemm_3dtype(L, 100),
    "docs/perf/layer/sm120.md": lambda L: render_layer(L, 120),
    "docs/perf/layer/sm90.md": lambda L: render_layer(L, 90),
    "docs/perf/layer/sm100.md": lambda L: render_layer(L, 100),
    "README.md": render_readme,
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
