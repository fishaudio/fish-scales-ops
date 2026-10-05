#!/usr/bin/env python3
"""The performance harness without a GPU: the environment locks, the suite of every machine, the plan
bench/run_perf.py resolves from them, and the raw-file manifest of bench/gemm/python/perf_report.py.

  - Each lock (bench/env/<machine>.lock.json) parses and has every required key, and its L2 size is the one
    perf_report.py's cold-protocol check uses for that device.
  - Each suite step names an existing bench script, options and implementation names that script knows, and a
    table group; every script accepts the --Ms option a smoke run adds.
  - Each step's output file is one of the raw files perf_report.manifest() merges into a baseline file.
  - The Family A MLP-block comparator files (group mlp_ref) merge by M into ref_mlp_qwen3_4b_<dev>.jsonl, and
    render_perf_docs.py joins them to the Family A table only once that file has timed cells.
  - The plan resolves for every machine (no preflight), a smoke plan adds --Ms 1,64 to every step, and the step
    environment drops every FSO_* knob but the lock's bench environment.
  - The preflight's drift detection lists a package whose version differs from the lock.
  - run_perf.py --dry-run runs for the local machine's lock (when this host is one of the locked machines) and
    prints the complete plan and a preflight verdict; it creates nothing.

Runs with any Python 3.8+ and no third-party package, directly or under pytest:
  python3 tests/bench/test_run_perf_plan.py
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "bench"))
sys.path.insert(0, os.path.join(ROOT, "bench", "gemm", "python"))
import perf_report  # noqa: E402
import perf_suite  # noqa: E402
import render_perf_docs  # noqa: E402
import run_perf  # noqa: E402

RUN_PERF = os.path.join(ROOT, "bench", "run_perf.py")


def _locks():
    return {m: run_perf.load_lock(run_perf.lock_path_for(m))[0] for m in run_perf.MACHINES}


def test_locks_parse_and_have_required_keys():
    for m, lock in _locks().items():
        problems = run_perf.validate_lock(lock)
        assert not problems, f"{m}: {problems}"
        assert lock["machine"] == m, f"{m}: the lock names machine {lock['machine']!r}"
        assert set(perf_suite.SUITES) == set(run_perf.MACHINES)
    assert sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "bench", "env", "*.lock.json"))) == \
        sorted(f"{m}.lock.json" for m in run_perf.MACHINES), "a lock file without a machine, or a machine without one"


def test_lock_l2_matches_perf_report():
    for m, lock in _locks().items():
        assert lock["gpu"]["l2_bytes"] == perf_report.L2_BYTES[m], \
            f"{m}: lock L2 {lock['gpu']['l2_bytes']} != perf_report.L2_BYTES {perf_report.L2_BYTES[m]}"


def test_suite_steps_name_known_scripts_and_impls():
    locks = _locks()
    for m in run_perf.MACHINES:
        problems = run_perf.validate_suite(m, locks[m])
        assert not problems, f"{m}: " + "; ".join(problems)
        for s in perf_suite.SUITES[m]:
            assert os.path.isfile(os.path.join(ROOT, s["script"])), s
    # the checker itself rejects what it should
    know = run_perf.script_knowledge(os.path.join(ROOT, perf_suite.MOE_C))
    assert "fso_bsfp8_layer" in know["impls"] and "--Ms" in know["options"], "the 35a3 front-end resolves to the 30a3 bench"
    bad = dict(perf_suite.SUITES["h200"][0], name="bad", args=["--run", "--impls", "no_such_impl"], out="x.jsonl")
    perf_suite.SUITES["_t"] = [bad]
    try:
        assert any("no_such_impl" in p or "--impls" in p for p in run_perf.validate_suite("_t")), "an unknown impl passed"
    finally:
        del perf_suite.SUITES["_t"]
    # the MLP-block bench's --dtypes: its DTYPES tuple (a sum of two tuples) is read from the source and checked
    know = run_perf.script_knowledge(os.path.join(ROOT, perf_suite.MLP))
    assert {"bf16", "bsfp8", "sgl_fp8b", "vllm_fp8b", "cublas_fp8b"} <= know["dtypes"] and "--dtypes" in know["options"], know
    bad = perf_suite.step("bad", "main", perf_suite.MLP, ["--run", "--dtypes", "sgl_fp8b,no_such_dtype"], "x.jsonl", "mlp_ref")
    perf_suite.SUITES["_t"] = [bad]
    try:
        assert any("no_such_dtype" in p for p in run_perf.validate_suite("_t")), "an unknown dtype passed"
    finally:
        del perf_suite.SUITES["_t"]


def test_suite_outputs_are_covered_by_the_manifest():
    for m, lock in _locks().items():
        tmp = tempfile.mkdtemp(prefix="perf_plan_manifest_")
        try:
            outs = [s["out"] for s in perf_suite.SUITES[m]]
            for o in outs:
                open(os.path.join(tmp, o), "w").close()
            covered = {os.path.basename(f) for files in perf_report.manifest(tmp, m, str(lock["sm"])).values() for f in files}
            missing = [o for o in outs if o not in covered]
            assert not missing, f"{m}: outputs no baseline file takes: {missing}"
        finally:
            shutil.rmtree(tmp)


def test_plan_resolution_for_every_machine():
    for m, lock in _locks().items():
        run_dir = f"/nonexistent/run_{m}"
        full = run_perf.resolve_plan(lock, run_dir, list(run_perf.GROUPS), smoke=False)
        assert len(full) == len(perf_suite.SUITES[m])
        for s in full:
            env = lock["environments"][s["env"]]
            assert s["argv"][0] == env["python"]
            assert os.path.isfile(s["argv"][1])
            assert s["argv"][-2:] == ["--out", os.path.join(run_dir, s["out"])]
            assert "--Ms" not in s["argv"]
        smoke = run_perf.resolve_plan(lock, run_dir, list(run_perf.GROUPS), smoke=True)
        assert all(s["argv"][-2:] == run_perf.SMOKE_ARGS for s in smoke)
        dense = run_perf.resolve_plan(lock, run_dir, ["dense"], smoke=False)
        assert dense and all(s["group"] == "dense" for s in dense) and len(dense) == 4
        groups = {s["group"] for s in full}
        assert groups <= set(run_perf.GROUPS) and {"dense", "moe", "moe_ref", "shared", "mlp_ref"} <= groups, (m, groups)
        mlp = run_perf.resolve_plan(lock, run_dir, ["mlp_ref"], smoke=False)
        assert mlp and all(s["group"] == "mlp_ref" and s["script"] == perf_suite.MLP and "--dtypes" in s["argv"]
                           for s in mlp), (m, mlp)
        # the step environment: FSO_* knobs gone except the lock's bench environment, overlays and fso path set
        base = {"PATH": "/usr/bin", "FSO_SWAP_BN": "1", "FSO_BENCH_WARM_MS": "7", "TRTLLM_DG_CACHE_DIR": "/x",
                "PYTHONPATH": "/elsewhere", "LD_LIBRARY_PATH": "/old", "HOME": "/home/u"}
        for name, e in lock["environments"].items():
            env, setv, removed = run_perf.step_environment(lock, name, "/fso", "GPU-x", base=base)
            assert "FSO_SWAP_BN" not in env and "TRTLLM_DG_CACHE_DIR" not in env
            assert env.get("FSO_BENCH_WARM_MS") == lock["bench_env"].get("FSO_BENCH_WARM_MS")
            assert env["PYTHONPATH"].split(os.pathsep) == ["/fso"] + e["pythonpath_overlays"]
            assert env["PATH"].split(os.pathsep)[0] == os.path.dirname(e["python"])
            assert env["CUDA_VISIBLE_DEVICES"] == "GPU-x" and env["PYTHONDONTWRITEBYTECODE"] == "1"
            assert env.get("LD_LIBRARY_PATH") == e["env"].get("LD_LIBRARY_PATH")
            assert env["HOME"] == "/home/u"
    print("  plans: " + ", ".join(f"{m} {len(perf_suite.SUITES[m])} steps" for m in run_perf.MACHINES))


def test_mlp_ref_merge_and_render():
    """Two comparator raw files (one with an n/a cell) merge by M into ref_mlp_qwen3_4b_h200.jsonl; the Family A
    table of the installed H200 baseline renders unchanged without that file and gains a µs and a ×fso BSFP8 column
    per comparator with it."""
    tmp = tempfile.mkdtemp(prefix="perf_plan_mlp_ref_")
    saved = render_perf_docs.BASE
    try:
        meta = {"_device": "NVIDIA H200", "_sm": 90, "torch": "2.13.0+cu130"}
        row = {"hidden": 2560, "intermediate": 9728}
        raw = {"ref_mlp_sgl_h200.jsonl": [dict(meta, dtypes=["sgl_fp8b"])] + [
                   dict(row, M=M, sgl_fp8b={"backend": "DeepGEMM", "cos": 0.998, "weight_copies": 2, "graph_us": 50.0 + M})
                   for M in (1, 64)],
               "ref_mlp_cublas_h200.jsonl": [dict(meta, dtypes=["cublas_fp8b"]),
                   dict(row, M=1, cublas_fp8b={"backend": "cuBLASLt", "cos": 0.998, "weight_copies": 2, "graph_us": 60.0}),
                   dict(row, M=64, cublas_fp8b={"error": "n/a: test"})]}
        for name, rows in raw.items():
            with open(os.path.join(tmp, name), "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
        files = perf_report.manifest(tmp, "h200", "90")["ref_mlp_qwen3_4b_h200.jsonl"]
        assert [os.path.basename(f) for f in files] == ["ref_mlp_sgl_h200.jsonl", "ref_mlp_cublas_h200.jsonl"], files
        out = os.path.join(tmp, "ref_mlp_qwen3_4b_h200.jsonl")
        n, na = perf_report.merge_baseline(os.path.basename(out), out, files)
        assert n == 3 and na == [("cublas_fp8b", 64, "n/a: test")], (n, na)
        merged = perf_report.load(out)
        assert merged[0]["_device"] == "NVIDIA H200" and len(merged[0]["sources"]) == 2
        assert [r["M"] for r in merged[1:]] == [1, 64] and {"sgl_fp8b", "cublas_fp8b"} <= set(merged[1])
        assert perf_report.cells_of(out)[("mlp", 64, "sgl_fp8b")] == 114.0
        base = os.path.join(tmp, "base")
        os.makedirs(base)
        fso = "gemm_sm90_qwen3_4b_mlp_fwd.jsonl"
        shutil.copy(os.path.join(saved, fso), base)
        render_perf_docs.BASE = base
        h0, b0 = render_perf_docs.mlp_table(fso, 90, os.path.basename(out))
        assert (h0, b0) == render_perf_docs.mlp_table(fso, 90), "the table changed without a comparator file"
        shutil.copy(out, base)
        h1, b1 = render_perf_docs.mlp_table(fso, 90, os.path.basename(out))
        assert h1[0].startswith(h0[0]) and h1[0][len(h0[0]):] == (
            " sglang block-FP8 linear (DeepGEMM) µs | ×fso BSFP8 | cuBLAS scaled_mm block-FP8 (cuBLASLt) µs | ×fso BSFP8 |"), h1[0]
        bs = {r["M"]: r["bsfp8"]["graph_us"] for r in render_perf_docs.load(fso) if "M" in r}
        rows0 = {int(l.split("|")[1]): l for l in b0}
        rows1 = {int(l.split("|")[1]): l for l in b1}
        assert rows1[1] == rows0[1] + f" 51.00 | {51.0 / bs[1]:.2f} | 60.00 | {60.0 / bs[1]:.2f} |", rows1[1]
        assert rows1[64] == rows0[64] + f" 114.00 | {114.0 / bs[64]:.2f} | — | — |", rows1[64]
        assert all(rows1[M] == rows0[M] + " — | — | — | — |" for M in rows0 if M not in (1, 64)), \
            "an M without comparator rows must read — in every comparator column"
    finally:
        render_perf_docs.BASE = saved
        shutil.rmtree(tmp)


def test_drift_is_listed():
    """The preflight's comparison, on a doctored lock and a canned probe: a torch version that differs is drift."""
    lock = json.loads(json.dumps(_locks()["h200"]))
    lock["environments"]["main"]["packages"]["torch"] = "2.12.0+cu130"
    lock["environments"]["main"]["python"] = sys.executable          # an interpreter that exists on any host
    probe = {"executable": "/x/python", "prefix": "/x", "python_version": lock["environments"]["main"]["python_version"],
             "packages": dict(_locks()["h200"]["environments"]["main"]["packages"]),
             "fso": {"file": "/fso/fish_scales_ops/__init__.py", "version": "0.2.0", "build_info": None,
                     "extension_sha256": "0" * 64, "jit_compiler_sm90": "NVRTC 13.0 (/fso/fish_scales_ops/_nvrtc/libnvrtc.so.13)"}}
    saved = (run_perf.run_probe, run_perf.query_cards, run_perf.compute_apps)
    card = {"index": "6", "pci": "00000000:CB:00.0", "uuid": lock["gpu"]["select"]["uuids"][0], "name": lock["device_name"],
            "driver": lock["driver"], "compute_mode": "Default", "clocks_max_sm_mhz": 1980, "power_limit_w": 700,
            "memory_total_mib": 143771}
    try:
        run_perf.run_probe = lambda *a, **k: (probe, None)
        run_perf.query_cards = lambda: [card]
        run_perf.compute_apps = lambda uuid: []
        plan = run_perf.resolve_plan(lock, "/nonexistent", ["shared"], smoke=True)
        pre = run_perf.preflight(lock, plan, None, None, 60)
    finally:
        run_perf.run_probe, run_perf.query_cards, run_perf.compute_apps = saved
    torch_drift = [d for d in pre["drift"] if d.get("name") == "torch"]
    assert torch_drift and torch_drift[0]["lock"] == "2.12.0+cu130" and torch_drift[0]["found"] == "2.13.0+cu130", pre["drift"]
    assert run_perf.verdict(pre, allow_drift=False)[0] is False
    assert run_perf.verdict(pre, allow_drift=True)[0] is (not pre["blockers"])


def local_machine():
    """The lock whose host pattern matches this host, or None."""
    import fnmatch
    import socket
    host = socket.gethostname()
    for m, lock in _locks().items():
        if fnmatch.fnmatchcase(host, lock["host_pattern"]):
            return m
    return None


def test_dry_run_on_the_local_machine():
    m = local_machine()
    if m is None:
        print("  skip: this host is none of the locked machines; the plan resolution above covers every lock")
        return
    out = os.path.join(tempfile.mkdtemp(prefix="perf_plan_dry_"), "run")
    cmd = [sys.executable, RUN_PERF, "--machine", m, "--out", out, "--dry-run"]
    fso = os.environ.get("PERF_PLAN_FSO_PATH")   # optional: a fish_scales_ops directory for the preflight to import
    if fso:
        cmd += ["--fso-path", fso]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=1800)
    assert p.returncode in (0, run_perf.EXIT_REFUSED), f"dry run exited {p.returncode}:\n{p.stdout[-3000:]}"
    for s in perf_suite.SUITES[m]:
        assert f"{s['name']} [{s['group']}]" in p.stdout and os.path.join(out, s["out"]) in p.stdout, s["name"]
    assert "preflight verdict:" in p.stdout and "dry run: nothing was started" in p.stdout
    assert not os.path.exists(out), "a dry run created the run directory"
    verdict = next(l for l in p.stdout.splitlines() if l.startswith("preflight verdict:"))
    print(f"  dry run on {m}: exit {p.returncode}; {verdict}")


def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {name}: {e}")
    print("run_perf plan: " + ("ALL PASS" if not failures else f"{failures} FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
