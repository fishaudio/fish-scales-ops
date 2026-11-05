#!/usr/bin/env python3
"""The layer-role composites of bench/gemm/python/bench_issue12_dense.py, without a GPU.

A synthetic role jsonl (written to a temporary directory) gives every arm of every role a known µs and kernel count,
and the test checks that:

  - the tower composites equal their definitions, summed here by hand: bf16 / fso_today / fso_epi over every role
    with q counted 3 x 32 (it stands for k and v); fso_best_fsoquant = 32 x [quant(q) + 3 epi_2b(q) + epi(out_proj)
    + quant(fc1) + q_2b(fc1) + epi_2b(fc2)] + epi(conv_out) + quant(proj1) + q_2b(proj1) + epi_2b(proj2), and
    fso_best_producerquant the same without the three quantize terms; launches the same sums over kernels;
  - the LLM composites equal 48 x (qkv + o) per arm and 48 x (quant(qkv) + epi_2b(qkv) + epi(o)) (without the
    quantize for producerquant);
  - one n/a arm makes exactly the composites that use it n/a, and the rendered row names the missing roles
    ("q, k, v") and the arm; a missing cell is named as not measured;
  - a file with role lines only renders no class table, and --format of a class-only file is the class rendering
    alone (the unchanged S0a path).

Runs with any Python 3.8+ and no third-party package, directly or under pytest:
  python3 tests/bench/test_issue12_roles_render.py
"""
import json
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "bench", "gemm", "python"))
import bench_issue12_dense as B  # noqa: E402

M = 78
LLM_M = 16


def _us(role, arm):
    """A distinct, exactly representable µs per (role, arm)."""
    return float(1 + list(B.ALL_ROLES).index(role)) + B.ROLE_ARM_ORDER.index(arm) / 8.0


def _kern(role, arm):
    return 1 + (list(B.ALL_ROLES).index(role) + B.ROLE_ARM_ORDER.index(arm)) % 3


def _line(role, m, na=()):
    name, n, k, has_bias, has_res, act, count, aliases = B.ALL_ROLES[role]
    arms = {}
    for a in B.ROLE_ARM_ORDER:
        if a == "fso_q_2b" and role not in B.Q2B_ROLES:
            continue
        if a in na:
            arms[a] = {"na": f"{a} refused on {role}"}
        else:
            arms[a] = {"us": _us(role, a), "us_rounds": [_us(role, a)] * 3, "weight_copies": 4,
                       "copy_bytes": 1 << 26, "kernels": _kern(role, a), "kernel_names": ["k"] * _kern(role, a),
                       "other_gpu_ops": []}
    return {"group": B.GROUP_ROLES, "machine": "test", "device": "synthetic", "sm": 90, "l2_bytes": 1 << 26,
            "role": role, "family": "tower" if role in {r[0] for r in B.TOWER_ROLES} else "llm",
            "aliases": list(aliases), "N": n, "K": k, "M": m, "has_bias": has_bias, "has_residual": has_res,
            "activation": act, "count": count, "arms": arms, "epi_2b_eq_epi": True, "q_2b_eq_unfused": None}


def _write(path, lines, header=True):
    with open(path, "w") as f:
        if header:
            f.write(json.dumps({"_header": True, "group": B.GROUP_ROLES, "device": "synthetic", "sm": 90,
                                "l2_bytes": 1 << 26}) + "\n")
        for l in lines:
            f.write(json.dumps(l) + "\n")


def _all_lines(na=None, skip=()):
    na = na or {}
    out = [_line(r[0], M, na.get(r[0], ())) for r in B.TOWER_ROLES if r[0] not in skip]
    out += [_line(r, LLM_M, na.get(r, ())) for r in ("qkv", "o") if r not in skip]
    return out


def _expected_tower():
    u, kn = _us, _kern
    per_arm = {}
    for arm in ("bf16", "fso_today", "fso_epi"):
        per_arm[arm] = (96 * u("q", arm) + 32 * (u("out_proj", arm) + u("fc1", arm) + u("fc2", arm))
                        + u("conv_out", arm) + u("proj1", arm) + u("proj2", arm),
                        96 * kn("q", arm) + 32 * (kn("out_proj", arm) + kn("fc1", arm) + kn("fc2", arm))
                        + kn("conv_out", arm) + kn("proj1", arm) + kn("proj2", arm))
    pq_us = (32 * (3 * u("q", "fso_epi_2b") + u("out_proj", "fso_epi") + u("fc1", "fso_q_2b")
                   + u("fc2", "fso_epi_2b"))
             + u("conv_out", "fso_epi") + u("proj1", "fso_q_2b") + u("proj2", "fso_epi_2b"))
    pq_k = (32 * (3 * kn("q", "fso_epi_2b") + kn("out_proj", "fso_epi") + kn("fc1", "fso_q_2b")
                  + kn("fc2", "fso_epi_2b"))
            + kn("conv_out", "fso_epi") + kn("proj1", "fso_q_2b") + kn("proj2", "fso_epi_2b"))
    q_us = 32 * (u("q", "fso_quant") + u("fc1", "fso_quant")) + u("proj1", "fso_quant")
    q_k = 32 * (kn("q", "fso_quant") + kn("fc1", "fso_quant")) + kn("proj1", "fso_quant")
    per_arm["fso_best_producerquant"] = (pq_us, pq_k)
    per_arm["fso_best_fsoquant"] = (pq_us + q_us, pq_k + q_k)
    return per_arm


def test_tower_composites_match_their_definitions():
    by = B._role_index(_all_lines())
    exp = _expected_tower()
    for name in B.TOWER_COMPOSITES:
        c = B.composite(by, M, B.tower_terms(name))
        assert c["missing"] == [], (name, c["missing"])
        assert abs(c["us"] - exp[name][0]) < 1e-9, (name, c["us"], exp[name][0])
        assert c["launches"] == exp[name][1], (name, c["launches"], exp[name][1])


def test_llm_composites_match_their_definitions():
    by = B._role_index(_all_lines())
    u, kn = _us, _kern
    exp = {
        "bf16": (48 * (u("qkv", "bf16") + u("o", "bf16")), 48 * (kn("qkv", "bf16") + kn("o", "bf16"))),
        "fso_today": (48 * (u("qkv", "fso_today") + u("o", "fso_today")),
                      48 * (kn("qkv", "fso_today") + kn("o", "fso_today"))),
        "fso_best_fsoquant": (48 * (u("qkv", "fso_quant") + u("qkv", "fso_epi_2b") + u("o", "fso_epi")),
                              48 * (kn("qkv", "fso_quant") + kn("qkv", "fso_epi_2b") + kn("o", "fso_epi"))),
        "fso_best_producerquant": (48 * (u("qkv", "fso_epi_2b") + u("o", "fso_epi")),
                                   48 * (kn("qkv", "fso_epi_2b") + kn("o", "fso_epi"))),
    }
    for name in B.LLM_COMPOSITES:
        c = B.composite(by, LLM_M, B.llm_terms(name, "qkv", "o"))
        assert c["missing"] == [] and abs(c["us"] - exp[name][0]) < 1e-9 and c["launches"] == exp[name][1], \
            (name, c, exp[name])


def test_na_term_names_the_missing_roles():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "roles.jsonl")
        _write(path, _all_lines(na={"q": ("fso_epi_2b", "fso_epi")}, skip=("proj2",)))
        by = B._role_index(B._load(path, B.GROUP_ROLES)[1])
        # bf16 / fso_today use neither refused arm, but proj2 has no cell: not measured.
        c = B.composite(by, M, B.tower_terms("bf16"))
        assert c["us"] is None and c["missing"] == [("not measured", "bf16", "proj2")], c
        c = B.composite(by, M, B.tower_terms("fso_best_producerquant"))
        assert c["us"] is None
        assert ("n/a", "fso_epi_2b", "q, k, v") in c["missing"], c["missing"]
        assert ("not measured", "fso_epi_2b", "proj2") in c["missing"], c["missing"]
        md = B.format_roles("T", path)
        row = [l for l in md.splitlines() if l.startswith(f"| {M} | `fso_best_producerquant` |")]
        assert len(row) == 1 and "| n/a |" in row[0], row
        assert "n/a fso_epi_2b: q, k, v" in row[0] and "not measured fso_epi_2b: proj2" in row[0], row[0]
        row = [l for l in md.splitlines() if l.startswith(f"| {M} | `fso_epi` |")]
        assert "n/a fso_epi: q, k, v" in row[0], row[0]
        # The per-role table prints n/a and lists the refusal message.
        assert "- `fso_epi_2b` at M=78: fso_epi_2b refused on q" in md
        # LLM composites are untouched by the tower's n/a terms.
        llm = [l for l in md.splitlines() if l.startswith(f"| {LLM_M} | decode |")]
        assert len(llm) == 1 and "n/a" not in llm[0], llm


def test_role_only_file_renders_no_class_tables():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "roles.jsonl")
        _write(path, _all_lines())
        md = B._render_file(B.format_one, B.format_roles, "T", path)
        assert "layer roles" in md and "Per cell (µs per call" not in md, md[:400]
        ck = B._render_file(B.checks_one, B.checks_roles, "T", path)
        assert "Checks, T, layer roles" in ck and "No arm was refused" in ck, ck[:400]
        sp = B.spread_roles([("a", path), ("b", path)])
        assert "| fso_epi | all |" in sp and "0.00 %" in sp, sp[:400]


def test_class_only_file_renders_the_class_tables_alone():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "classes.jsonl")
        line = {"group": B.GROUP, "class": "tower_conv_out", "N": 1280, "K": 7680, "M": 78, "has_bias": False,
                "count": 1, "device": "synthetic", "sm": 90,
                "arms": {a: {"us": 1.0, "us_rounds": [1.0] * 3, "weight_copies": 4, "copy_bytes": 1 << 26,
                             "kernels": 1, "kernel_names": ["k"], "other_gpu_ops": []}
                         for a in ("bf16", "fso_chain", "fso_gemm", "fso_quant")}}
        _write(path, [line], header=False)
        assert B._render_file(B.format_one, B.format_roles, "T", path) == B.format_one("T", path)
        assert B._render_file(B.checks_one, B.checks_roles, "T", path) == B.checks_one("T", path)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
