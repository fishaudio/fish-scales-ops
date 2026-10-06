#!/usr/bin/env python3
"""The generated summaries of the performance docs, without a GPU and without writing a file.

  - Rendering is idempotent: every target of bench/gemm/python/render_perf_docs.py, rendered from its file and
    then rendered again, comes out unchanged the second time (in memory; `render_perf_docs.py --check` compares the
    rendering with the files on disk).
  - A layer page whose summary blocks are missing gets them inserted where the replace path writes them: the
    page rendered with its blocks deleted equals the page rendered with them in place.
  - Every summary row equals what `perf_report.py report` prints for the same data: the band geomeans of the
    routed-layer comparators of Families B and C and of Family A's comparators against fso's BSFP8 block. The rows
    perf_report.py has no report for (Family A against the MXFP8 block, the comparators of the Family C routed +
    shared block) and the "faster than fso" lists are recomputed here from the raw cells of the baseline files.
    perf_report.py drops M = 96 from its geomeans, while a summary takes every M where fso and the comparator both
    have a value; the two agree because no comparator file has an M = 96 row. A comparator row at M = 96 would make
    this test fail, on purpose: decide then whether the summary should drop that row too.

Runs with any Python 3.8+ and no third-party package, directly or under pytest:
  python3 tests/bench/test_render_perf_summary.py
"""
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "bench", "gemm", "python"))
import perf_report  # noqa: E402
import render_perf_docs as rpd  # noqa: E402

DEVICES = {90: "h200", 100: "b300", 120: "5090"}
LAYER_PAGES = {90: "docs/perf/layer/sm90.md", 100: "docs/perf/layer/sm100.md", 120: "docs/perf/layer/sm120.md"}


def _read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


def _render(rel, text):
    lines = text.split("\n")
    rpd.TARGETS[rel](lines)
    return "\n".join(lines)


def test_render_is_idempotent():
    for rel in rpd.TARGETS:
        once = _render(rel, _read(rel))
        assert _render(rel, once) == once, f"{rel}: a second rendering changed the document"


def _drop_summary_blocks(text):
    """The page without its summary blocks: each block and the one blank line before it."""
    lines, out, skip = text.split("\n"), [], False
    for line in lines:
        if line.startswith("<!-- BEGIN GENERATED: summary Family "):
            assert out and out[-1] == "", "a summary block must follow one blank line"
            out.pop()
            skip = True
        if not skip:
            out.append(line)
        if skip and line.startswith("<!-- END GENERATED: summary Family "):
            skip = False
    return "\n".join(out)


def test_missing_summary_blocks_are_inserted_where_they_belong():
    for sm, rel in LAYER_PAGES.items():
        rendered = _render(rel, _read(rel))
        stripped = _drop_summary_blocks(rendered)
        assert "GENERATED: summary" not in stripped
        assert _render(rel, stripped) == rendered, f"{rel}: the inserted summary blocks differ from the replaced ones"


def _report_geomeans(text):
    """{(column title, band name): '1.07×'} from the geomean tables of one perf_report.py report section."""
    out, cols = {}, None
    for line in text.split("\n"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "geomean ×fso":
            cols = cells[1:]
        elif cols and cells[0] in [b for b, _ in perf_report.BANDS]:
            for title, v in zip(cols, cells[1:]):
                out[(title, cells[0])] = v.split(" (")[0]
    return out


def _sections(sm):
    return {s["family"]: s for s in rpd.layer_sections(sm) if s["summary"] is not None}


def test_moe_summaries_match_perf_report():
    names = dict(perf_report.LAYER_IMPLS)
    for sm, dev in DEVICES.items():
        secs = _sections(sm)
        for fam, letter in (("30a3", "B"), ("35a3", "C")):
            perf = os.path.join(perf_report.BASELINES, f"perf_moe_qwen3_{fam}_{dev}.jsonl")
            ref = os.path.join(perf_report.BASELINES, f"ref_moe_qwen3_{fam}_{dev}.jsonl")
            report = _report_geomeans(perf_report.layer_report(perf, ref, "x"))
            routed = [r for r in secs[letter]["summary"] if r["against"] in ("fso block", "fso routed")]
            assert routed, f"sm_{sm} Family {letter}: no routed-layer summary row"
            for r in routed:
                for (band, _), g in zip(perf_report.BANDS, r["bands"]):
                    want = report.get((names[r["key"]], band), "—")
                    assert rpd.fx(g) == want, f"sm_{sm} Family {letter} {r['key']} {band}: {rpd.fx(g)} != report {want}"


def test_mlp_summaries_against_bsfp8_match_perf_report():
    names = dict(perf_report.MLP_IMPLS)
    for sm, dev in DEVICES.items():
        perf = os.path.join(perf_report.BASELINES, f"gemm_sm{sm}_qwen3_4b_mlp_fwd.jsonl")
        ref = os.path.join(perf_report.BASELINES, f"ref_mlp_qwen3_4b_{dev}.jsonl")
        report = _report_geomeans(perf_report.mlp_report(perf, ref, "x"))
        rows = [r for r in _sections(sm)["A"]["summary"] if r["against"] == "fso BSFP8 block" and r["key"] in names]
        assert len(rows) == len({t for (t, _) in report}), f"sm_{sm}: summary rows {[r['key'] for r in rows]} vs report"
        for r in rows:
            for (band, _), g in zip(perf_report.BANDS, r["bands"]):
                assert rpd.fx(g) == report[(names[r["key"]], band)], f"sm_{sm} Family A {r['key']} {band}"


def _independent(fso, cmp):
    """Band geomeans and the 'faster' M list of one comparator, computed directly from {M: µs} cells."""
    both = sorted(M for M in fso if M in cmp)
    bands = []
    for _, pred in perf_report.BANDS[:2]:
        xs = [cmp[M] / fso[M] for M in both if pred(M)]
        bands.append(math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None)
    return bands, [M for M in both if cmp[M] < 0.99 * fso[M]]


def test_every_summary_row_matches_the_raw_cells():
    for sm, dev in DEVICES.items():
        secs = _sections(sm)
        base = perf_report.BASELINES
        mlp = perf_report.cells_of(os.path.join(base, f"gemm_sm{sm}_qwen3_4b_mlp_fwd.jsonl"))
        ref_path = os.path.join(base, f"ref_mlp_qwen3_4b_{dev}.jsonl")
        mlp_ref = perf_report.cells_of(ref_path) if os.path.exists(ref_path) else {}
        moe = {}
        for name in (f"perf_moe_qwen3_30a3_{dev}.jsonl", f"ref_moe_qwen3_30a3_{dev}.jsonl", f"perf_moe_qwen3_35a3_{dev}.jsonl",
                     f"perf_moe_qwen3_35a3_shared_{dev}.jsonl", f"ref_moe_qwen3_35a3_{dev}.jsonl",
                     f"ref_moe_qwen3_35a3_shared_{dev}.jsonl"):
            if os.path.exists(os.path.join(base, name)):
                fam = "B" if "30a3" in name else "C"
                for (impl, proj, M), us in perf_report.cells_of(os.path.join(base, name)).items():
                    if proj == "layer":
                        moe.setdefault((fam, impl), {})[M] = us
        dt = "bsfp8" if sm == 90 else "mxfp8"
        fso_of = {("A", "fso BSFP8 block"): {M: us for (_, M, d), us in mlp.items() if d == "bsfp8"},
                  ("A", "fso MXFP8 block"): {M: us for (_, M, d), us in mlp.items() if d == "mxfp8"},
                  ("B", "fso block"): moe.get(("B", f"fso_{dt}_layer"), {}),
                  ("C", "fso routed"): moe.get(("C", f"fso_{dt}_layer"), {}),
                  ("C", "fso routed + shared"): moe.get(("C", f"fso_{dt}_layer_shared"), {})}
        for letter, sec in secs.items():
            for r in sec["summary"]:
                if letter == "A":
                    src = mlp if r["key"] in ("bf16", "smm", "smm_fast") else mlp_ref
                    cmp = {M: us for (_, M, d), us in src.items() if d == r["key"]}
                else:
                    cmp = moe[(letter, r["key"])]
                bands, faster = _independent(fso_of[(letter, r["against"])], cmp)
                where = f"sm_{sm} Family {letter} {r['key']} against {r['against']}"
                assert [rpd.fx(g) for g in r["bands"]] == [rpd.fx(g) for g in bands], where
                assert [M for M, _ in r["faster"]] == faster, where


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
