#!/usr/bin/env python3
"""Run the whole test suite against one installed wheel, on the machine this script runs on.

This is the test half of the release harness: the same wheel is installed on every machine and the same tests run
against it, so a result never depends on a per-machine source build.

    python scripts/ci/run_suite.py --wheel W.whl --testkit KIT.tar.gz --work DIR --machine h200

``--machine`` reads the machine's environment lock from the test kit (``bench/env/<machine>.lock.json``, the file the
performance harness uses as well): the interpreter, the card and its lock file, the compute-mode handling, the
``PYTHONPATH`` overlays, the environment variables and the reference dump for the bitwise comparison. Without it, the
same things are given by hand:

    python scripts/ci/run_suite.py --wheel W.whl --testkit KIT.tar.gz --python /venv/bin/python --work DIR \\
        --gpu-uuid GPU-... [--gpu-lock FILE] [--compute-mode default] [--pythonpath-extra DIR ...] \\
        [--layer-ref REF.pt]

What it does, in order:

1. Unpacks the test kit (the ``git archive`` of tests/, bench/, scripts/, docs/ that ``scripts/build_wheel.sh`` writes
   next to the wheel) into ``DIR/kit``.
2. Checks the wheel against ``W.whl.sha256`` when that file exists.
3. Installs the wheel with ``pip install --no-deps --target DIR/site`` using the machine's interpreter. The
   interpreter's own site-packages are not modified, so a shared serving venv stays as it is. The kit and the wheel
   must come from the same commit.
4. Selects one GPU (``--gpu-uuid``, or ``--gpu-index`` for a machine whose card changes), takes ``--gpu-lock`` with
   flock, refuses a card that already runs a compute process, and with ``--compute-mode default`` switches the card to
   the Default compute mode for the run and restores the mode it found.
5. Runs every ``tests/**/test_*.py`` of the kit in its own process, then ``pytest tests/attention`` and
   ``render_perf_docs.py --check``, with ``PYTHONPATH`` = the installed wheel (plus ``--pythonpath-extra``) and every
   ``FSO_*`` variable unset. Tests for another architecture skip themselves.
6. With ``--layer-ref``, runs ``tests/tools/dump_layer_outputs.py`` and compares its tensors bitwise with the
   reference dump (a dump of the same script from a reference build).
7. Prints one line per test and writes ``DIR/result.json``. The exit code is 0 only if everything passed.

``--site DIR`` skips the installation and tests an existing package directory instead (an in-place build's ``python/``), with
``--kit-dir`` naming the tree that holds ``tests/``. Standard library only.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path


# Extra runs of one test under an environment switch: (step name, test path in the kit, environment).
VARIANTS = (
    ("test_moe_transient_bytes__FC1_FUSED_0", "tests/gemm/unit/test_moe_transient_bytes.py", {"FSO_FC1_FUSED": "0"}),
)


def say(msg: str) -> None:
    print(msg, flush=True)


def run(cmd, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def smi(*args: str) -> str:
    r = run(["nvidia-smi", *args])
    if r.returncode != 0:
        raise SystemExit(f"nvidia-smi {' '.join(args)} failed: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()


def install_wheel(python: str, wheel: Path, site: Path) -> None:
    sha_file = Path(str(wheel) + ".sha256")
    if sha_file.exists():
        want = sha_file.read_text().split()[0]
        got = sha256_of(wheel)
        if got != want:
            raise SystemExit(f"{wheel.name}: sha256 {got} does not match {sha_file.name} ({want})")
        say(f"[suite] wheel sha256 {got} matches {sha_file.name}")
    if site.exists():
        shutil.rmtree(site)
    r = run([python, "-m", "pip", "install", "--no-deps", "--no-compile", "--disable-pip-version-check",
             "--target", str(site), str(wheel)])
    if r.returncode != 0:
        raise SystemExit(f"pip install --target failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    say(f"[suite] installed {wheel.name} into {site}")


def unpack_kit(kit: Path, dest: Path) -> tuple[Path, str | None]:
    """Unpack the test kit; return the directory that holds tests/ and the commit `git archive` recorded."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(kit) as t:
        commit = (t.pax_headers or {}).get("comment")
        try:
            t.extractall(dest, filter="data")
        except TypeError:  # a Python without the extraction filters
            t.extractall(dest)
    roots = [dest] + [p for p in dest.iterdir() if p.is_dir()]
    for root in roots:
        if (root / "tests").is_dir():
            return root, commit
    raise SystemExit(f"{kit.name} holds no tests/ directory")


def select_gpu(args) -> dict:
    fields = "index,uuid,pci.bus_id,name,compute_mode,driver_version"
    rows = [[c.strip() for c in line.split(",")] for line in smi(f"--query-gpu={fields}", "--format=csv,noheader").splitlines()]
    gpus = [dict(zip(fields.split(","), row)) for row in rows]
    if args.gpu_uuid:
        hit = [g for g in gpus if g["uuid"] == args.gpu_uuid]
    else:
        hit = [g for g in gpus if g["index"] == str(args.gpu_index)]
    if len(hit) != 1:
        raise SystemExit(f"GPU selector matched {len(hit)} cards; this machine has: {[(g['index'], g['uuid']) for g in gpus]}")
    if args.gpu_name and hit[0]["name"] != args.gpu_name:
        raise SystemExit(f"the selected card is a {hit[0]['name']!r}, the lock expects {args.gpu_name!r}")
    return hit[0]


def apply_machine_lock(args, kit: Path) -> dict:
    """Fill the arguments the command line left open from bench/env/<machine>.lock.json of the kit."""
    path = kit / "bench" / "env" / f"{args.machine}.lock.json"
    if not path.exists():
        raise SystemExit(f"--machine {args.machine}: {path} does not exist")
    lock = json.loads(path.read_text())
    main, gpu, suite = lock["environments"]["main"], lock["gpu"], lock.get("suite", {})
    if args.python is None:
        args.python = main["python"]
    if args.gpu_uuid is None and args.gpu_index is None:
        sel = gpu["select"]
        if sel["by"] == "uuid":
            args.gpu_uuid = sel["uuids"][0]
        else:
            args.gpu_index, args.gpu_name = sel["index"], lock["device_name"]
    if args.gpu_lock is None:
        args.gpu_lock = gpu["lock_file"]
    if args.compute_mode is None and (gpu.get("compute_mode") or {}).get("set") == "DEFAULT":
        args.compute_mode = "default"
    args.pythonpath_extra = [*main.get("pythonpath_overlays", []), *suite.get("pythonpath_overlays", []),
                             *args.pythonpath_extra]
    if args.layer_ref is None and not args.no_layer_ref:
        args.layer_ref = suite.get("layer_ref")
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "env": dict(main.get("env", {}))}


def gpu_busy(uuid: str) -> list[str]:
    out = smi("-i", uuid, "--query-compute-apps=pid,process_name", "--format=csv,noheader")
    return [line for line in out.splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wheel")
    ap.add_argument("--testkit")
    ap.add_argument("--site", help="test this existing package directory instead of installing a wheel")
    ap.add_argument("--kit-dir", help="with --site: the tree that holds tests/ (default: this script's repository)")
    ap.add_argument("--machine", help="read the interpreter, the card and the overlays from bench/env/<machine>.lock.json")
    ap.add_argument("--python", help="the interpreter under test (default: the lock's, else the one running this script)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--gpu-uuid")
    ap.add_argument("--gpu-index", type=int)
    ap.add_argument("--gpu-lock", help="lock file taken with flock for the whole run")
    ap.add_argument("--compute-mode", choices=("keep", "default"))
    ap.add_argument("--pythonpath-extra", action="append", default=[])
    ap.add_argument("--layer-ref", help="reference dump for tests/tools/dump_layer_outputs.py (bitwise comparison)")
    ap.add_argument("--no-layer-ref", action="store_true", help="skip the bitwise comparison the lock names")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per test file")
    args = ap.parse_args()
    args.gpu_name = None
    if args.gpu_uuid is not None and args.gpu_index is not None:
        ap.error("give at most one of --gpu-uuid / --gpu-index")
    if not args.machine and args.gpu_uuid is None and args.gpu_index is None:
        ap.error("give --machine, or one of --gpu-uuid / --gpu-index")
    if bool(args.site) == bool(args.wheel):
        ap.error("give exactly one of --wheel (with --testkit) / --site")
    if args.wheel and not args.testkit:
        ap.error("--wheel needs --testkit")

    work = Path(args.work).resolve()
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    result: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": os.uname().nodename,
                    "tests": []}

    if args.wheel:
        kit, kit_commit = unpack_kit(Path(args.testkit).resolve(), work / "kit")
    else:
        kit = Path(args.kit_dir).resolve() if args.kit_dir else Path(__file__).resolve().parents[2]
        kit_commit = None
    lock_env: dict = {}
    if args.machine:
        lock_info = apply_machine_lock(args, kit)
        lock_env = lock_info.pop("env")
        result["lock"] = lock_info
    args.python = args.python or sys.executable
    args.compute_mode = args.compute_mode or "keep"
    result["python"] = args.python

    if args.wheel:
        wheel = Path(args.wheel).resolve()
        site = work / "site"
        install_wheel(args.python, wheel, site)
        result.update(wheel=wheel.name, wheel_sha256=sha256_of(wheel), testkit=Path(args.testkit).name,
                      testkit_commit=kit_commit)
    else:
        site = Path(args.site).resolve()
        result.update(site=str(site), kit=str(kit))

    gpu = select_gpu(args)
    env = {k: v for k, v in os.environ.items() if not k.startswith("FSO_")}
    env.update(lock_env)
    env.update(PYTHONPATH=os.pathsep.join([str(site), *args.pythonpath_extra]), CUDA_VISIBLE_DEVICES=gpu["uuid"],
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    env["PATH"] = os.pathsep.join([str(Path(args.python).parent), env.get("PATH", "")])
    result["gpu"] = gpu

    lock_fd = None
    if args.gpu_lock:
        lock_fd = open(args.gpu_lock, "a")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"{args.gpu_lock} is held by another run; not queueing behind it")
    busy = gpu_busy(gpu["uuid"])
    if busy:
        raise SystemExit(f"card {gpu['pci.bus_id']} already runs a compute process ({busy}); not starting")

    mode_found = gpu["compute_mode"]
    switched = False
    failures: list[str] = []

    def record(name: str, rc: int, log: Path, seconds: float) -> None:
        lines = [l for l in log.read_text(errors="replace").splitlines() if l.strip()] if log.exists() else []
        last = lines[-1][:160] if lines else ""
        result["tests"].append({"name": name, "exit": rc, "seconds": round(seconds, 1), "last_line": last})
        say(f"{name} exit {rc} ({seconds:.0f}s) : {last}")
        if rc != 0:
            failures.append(name)

    def step(name: str, cmd: list[str], cwd: Path, timeout: int | None = None) -> int:
        log = logs / f"{name}.log"
        t0 = time.time()
        with open(log, "w") as f:
            try:
                rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                    timeout=timeout or args.timeout).returncode
            except subprocess.TimeoutExpired:
                f.write(f"\nTIMEOUT after {timeout or args.timeout} s\n")
                rc = 124
        record(name, rc, log, time.time() - t0)
        return rc

    try:
        if args.compute_mode == "default" and mode_found != "Default":
            r = run(["sudo", "-n", "nvidia-smi", "-i", gpu["uuid"], "-c", "DEFAULT"])
            if r.returncode != 0:
                raise SystemExit(f"cannot switch the card to the Default compute mode: {r.stdout}{r.stderr}")
            switched = True
            say(f"[suite] compute mode: {mode_found} -> Default for the run")

        py = args.python
        ident = ("import json, torch, fish_scales_ops as f\n"
                 "info = f.build_info() if hasattr(f, 'build_info') else {'source_build': True}\n"
                 "major = torch.cuda.get_device_capability(0)[0]\n"
                 "jit = torch.ops.fish_scales_ops.jit_compiler_sm90() if major == 9 else None\n"
                 "print('IDENT ' + json.dumps({'version': f.__version__, 'file': f.__file__, 'torch': torch.__version__,\n"
                 "      'device': torch.cuda.get_device_name(0), 'sm': list(torch.cuda.get_device_capability(0)),\n"
                 "      'surface': {n: len(getattr(f, n).__all__) for n in ('dense', 'moe', 'compat', 'attention')},\n"
                 "      'build_info': info, 'jit_compiler_sm90': jit}))\n")
        rc = step("import", [py, "-W", "error::DeprecationWarning", "-c", ident], kit)
        if rc == 0:
            for line in (logs / "import.log").read_text().splitlines():
                if line.startswith("IDENT "):
                    result["fso"] = json.loads(line[6:])
            fso = result.get("fso", {})
            if not str(fso.get("file", "")).startswith(str(site)):
                failures.append("import-location")
                say(f"[suite] FAIL: fish_scales_ops was imported from {fso.get('file')}, not from {site}")
            build_commit = (fso.get("build_info") or {}).get("commit")
            if kit_commit and build_commit and not kit_commit.startswith(build_commit[:12]) \
                    and not build_commit.startswith(kit_commit[:12]):
                failures.append("commit-mismatch")
                say(f"[suite] FAIL: the wheel was built from {build_commit}, the test kit from {kit_commit}")

        tests = sorted(p for p in (kit / "tests").rglob("test_*.py") if "__pycache__" not in p.parts)
        for t in tests:
            step(t.stem, [py, str(t.relative_to(kit))], kit)
        # The same test under a production switch: the unfused FC1 changes what a layer call allocates.
        for name, rel, extra in VARIANTS:
            if (kit / rel).exists():
                env_saved = dict(env)
                env.update(extra)
                step(name, [py, rel], kit)
                env.clear(); env.update(env_saved)
        if (kit / "tests" / "attention").is_dir():
            step("pytest_attention", [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/attention/"], kit)
        render = kit / "bench" / "gemm" / "python" / "render_perf_docs.py"
        if render.exists():
            step("render_perf_docs_check", [py, str(render.relative_to(kit)), "--check"], kit)

        if args.layer_ref:
            dump = kit / "tests" / "tools" / "dump_layer_outputs.py"
            out = work / "layer_outputs.pt"
            rc = step("dump_layer_outputs", [py, str(dump.relative_to(kit)), str(out)], kit)
            if rc == 0:
                cmp_src = ("import sys, torch\n"
                           "a = torch.load(sys.argv[1]); b = torch.load(sys.argv[2])\n"
                           "bad = [k for k in a if k not in b or not torch.equal(a[k], b[k])]\n"
                           "ok = not bad and a.keys() == b.keys()\n"
                           "print(f'layer outputs: {len(a)} compared, {len(bad)} differ' + (f': {bad[:6]}' if bad else '')\n"
                           "      + '  BITWISE ' + ('PASS' if ok else 'FAIL'))\n"
                           "sys.exit(0 if ok else 1)\n")
                step("layer_outputs_bitwise", [py, "-c", cmp_src, args.layer_ref, str(out)], kit)
    finally:
        if switched:
            restore = "EXCLUSIVE_PROCESS" if mode_found == "Exclusive_Process" else "DEFAULT"
            run(["sudo", "-n", "nvidia-smi", "-i", gpu["uuid"], "-c", restore])
            now = smi("-i", gpu["uuid"], "--query-gpu=compute_mode", "--format=csv,noheader")
            say(f"[suite] compute mode restored to what it was ({mode_found}): {now}")
        left = gpu_busy(gpu["uuid"])
        say(f"[suite] compute processes left on the card: {len(left)}")
        result.update(finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), failures=failures,
                      passed=not failures)
        (work / "result.json").write_text(json.dumps(result, indent=1) + "\n")
        if lock_fd is not None:
            lock_fd.close()

    n = len(result["tests"])
    say(f"[suite] {n - len([t for t in result['tests'] if t['exit'] != 0])}/{n} steps passed"
        + (f"; FAILED: {', '.join(failures)}" if failures else "; ALL PASS"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
