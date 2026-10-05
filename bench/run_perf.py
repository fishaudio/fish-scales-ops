#!/usr/bin/env python3
"""run_perf.py -- run the performance tables of record on this machine, under the machine's environment lock.

  python3 bench/run_perf.py --machine h200 --out /data/bench-runs/<run> [--tables dense,moe,... | all]
                            [--fso-path DIR] [--smoke] [--dry-run] [--allow-drift]

There is one command per machine (h200, 5090, b300). It reads the machine's lock, bench/env/<machine>.lock.json
(the interpreter and the exact package versions of every environment, the driver, the card, the clock and
compute-mode policy, the GPU lock file and the bench environment), and the machine's steps in bench/perf_suite.py.

1. Preflight. Every interpreter the selected steps use is queried in a subprocess that sees no GPU: its Python and
   package versions and, with --fso-path first on PYTHONPATH, the fish_scales_ops it imports (version,
   build_info(), the sha256 of the extension, and on sm_90 the compiler that jit_compiler_sm90() names).
   nvidia-smi gives the driver and the card. A difference from the lock is drift, and the run refuses it (exit 3)
   unless --allow-drift is given, in which case the drift list goes into the manifest and perf_report.py install
   later refuses the run unless it is given --accept-drift. A blocker refuses the run in every case: the wrong
   host, a missing interpreter or overlay, fish_scales_ops that does not import or does not come from --fso-path,
   a card that is missing or has the wrong name, or a card with a compute process on it.
2. --dry-run prints the plan (steps, commands, environments) and the preflight result, then stops. It does not
   touch any GPU's compute state (nvidia-smi queries only) and creates nothing. Its exit code is the verdict the
   run would reach: 0 when the run would proceed, 3 when it would refuse.
3. The run takes the GPU lock file with flock (a bounded wait, --lock-wait) and then refuses a card that has a
   compute process rather than queueing behind it. It applies the lock's compute mode and clock policy and
   verifies them, and starts a 200 ms nvidia-smi sampler. Each step runs as a subprocess with a cleaned
   environment: every FSO_* variable unset except the lock's bench environment, PYTHONPATH set to --fso-path and
   the lock's overlays only, the venv's bin first on PATH, CUDA_VISIBLE_DEVICES set to the card's UUID,
   PYTHONDONTWRITEBYTECODE=1, and PYTHONUNBUFFERED=1 so that a line's timestamp is the time it was printed.
   Before each step the card must be idle; each step's output goes to logs/<step>.log with a UTC timestamp on
   every line. The clock lock, the compute mode and the GPU lock are released in a
   finally block, also on SIGTERM.
4. After the steps it runs perf_report.py merge into <run_dir>/merged and perf_report.py diff against the
   installed baselines, and saves both outputs (merge.txt, diff_vs_baselines.txt). It never installs anything.

--smoke runs every step on the cells with M in {1, 64} only and marks the manifest smoke: true, so that
perf_report.py install refuses the run. <run_dir>/manifest.json records the machine and host, the time window,
the lock (an embedded copy and its sha256), the versions the preflight found and the drift list, the
fish_scales_ops identity, the card, the clock policy and the observed clocks of every step, every step's command,
window, row count and failures, and the commit of the tree that holds the benches.

Exit codes: 0 done (or a dry run that would proceed); 2 usage error; 3 refused by the preflight; 4 the GPU lock
file stayed busy; 5 the card has a compute process; 6 the clock or compute-mode policy could not be applied or
verified; 7 the run was aborted (a step saw the device busy, or a signal arrived).

Standard library only, Python 3.8 or newer: it runs on the target machine with any system python3.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import errno
import fcntl
import fnmatch
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENV_DIR = os.path.join(HERE, "env")
PERF_REPORT = os.path.join(HERE, "gemm", "python", "perf_report.py")
KIT_COMMIT_FILE = os.path.join(ENV_DIR, "KIT_COMMIT")
sys.dont_write_bytecode = True          # the harness writes nothing into the tree or kit it runs from
sys.path.insert(0, HERE)
import perf_suite  # noqa: E402  (bench/perf_suite.py, beside this file)

MACHINES = ("h200", "5090", "b300")
GROUPS = perf_suite.GROUPS
SMOKE_MS = "1,64"
SMOKE_ARGS = ["--Ms", SMOKE_MS]
SAMPLE_MS = 200

EXIT_USAGE, EXIT_REFUSED, EXIT_LOCK, EXIT_BUSY, EXIT_POLICY, EXIT_ABORTED = 2, 3, 4, 5, 6, 7

# Bits of nvidia-smi's clocks_event_reasons.active (nvml.h, nvmlClocksEventReason*).
REASON_IDLE = 0x1
REASON_SW_POWER_CAP = 0x4
REASON_HW_SLOWDOWN = 0x8
REASON_SW_THERMAL = 0x20
REASON_HW_THERMAL = 0x40
REASON_HW_POWER_BRAKE = 0x80

# A worker that found the card taken by someone else; the run stops rather than measure on a shared card.
DEVICE_BUSY_TEXT = re.compile(r"busy or unavailable|DevicesUnavailable|all CUDA-capable devices are busy")
# Lines of a bench log that report a failed cell: the MoE bench prints FAILED, the dense bench prints <dtype>=ERR.
FAILURE_TEXT = re.compile(r"FAILED|Traceback|=ERR\b")

# The step environment inherits the caller's, minus every variable that steers the benches from outside the lock:
# all FSO_* knobs (the lock's bench environment is set again afterwards), the TensorRT-LLM aliases of the sm_90 JIT
# knobs, the device order, and the Python path variables, which the lock sets. The variables in LOCK_CONTROLLED are
# removed unless the lock's environment sets them, so that a run never depends on what the caller's shell exported.
UNSET_PREFIXES = ("FSO_", "TRTLLM_DG_")
UNSET_EXACT = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "LD_PRELOAD")
LOCK_CONTROLLED = ("CUDA_HOME", "LD_LIBRARY_PATH", "FLASHINFER_CUDA_ARCH_LIST", "TORCH_CUDA_ARCH_LIST")

# nvidia-smi compute mode, as queried -> as set with -c
MODE_ARG = {"Default": "DEFAULT", "Exclusive_Process": "EXCLUSIVE_PROCESS", "Prohibited": "PROHIBITED"}

# The --projs values the MoE bench knows (its PROJ table plus the whole-layer cells).
KNOWN_PROJS = ("gate_up", "down", "layer")


class Refused(Exception):
    """The run cannot start or continue; carries the exit code."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class Interrupted(Exception):
    pass


# ----------------------------------------------------------------------------- small helpers
def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def iso(t=None):
    t = t or utc_now()
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + "%03dZ" % (t.microsecond // 1000)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
        f.write("\n")
    os.replace(tmp, path)


def rel(path):
    """A path under the tree root, as the tree sees it; any other path unchanged."""
    p = os.path.abspath(path)
    return os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else p


class Log:
    """The harness's own lines: stdout, and <run_dir>/run_perf.log, which receives the lines printed before the
    run directory existed (the plan and the preflight) when it is opened."""

    def __init__(self):
        self.f = None
        self.early = []

    def open(self, path):
        self.f = open(path, "a")
        self.f.writelines(self.early)
        self.f.flush()
        self.early = []

    def __call__(self, msg=""):
        print(msg, flush=True)
        lines = [f"{iso()} {line}\n" for line in (str(msg).splitlines() or [""])]
        if self.f:
            self.f.writelines(lines)
            self.f.flush()
        else:
            self.early.extend(lines)


log = Log()


# ----------------------------------------------------------------------------- the lock
LOCK_KEYS = {"schema": int, "machine": str, "host_pattern": str, "sm": int, "arch": str, "device_name": str,
             "driver": str, "gpu": dict, "jit_compiler_sm90_prefix": (str, type(None)), "bench_env": dict,
             "environments": dict}
GPU_KEYS = {"select": dict, "lock_file": str, "l2_bytes": int, "clock_policy": dict, "compute_mode": (dict, type(None))}
ENV_KEYS = {"python": str, "python_version": str, "packages": dict, "env": dict, "path_prepend": list,
            "pythonpath_overlays": list, "fso": bool}


def lock_path_for(machine):
    return os.path.join(ENV_DIR, f"{machine}.lock.json")


def load_lock(path):
    """(lock dict, sha256 of the file's bytes)."""
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def validate_lock(lock):
    """The problems of a lock's structure, as sentences; an empty list means the lock is usable."""
    problems = []

    def need(d, keys, where):
        ok = True
        for k, t in keys.items():
            if k not in d:
                problems.append(f"{where}: missing key {k!r}")
                ok = False
            elif not isinstance(d[k], t) or (t is int and isinstance(d[k], bool)):
                problems.append(f"{where}: {k!r} has the wrong type ({type(d[k]).__name__})")
                ok = False
        return ok

    if not need(lock, LOCK_KEYS, "lock"):
        return problems
    if lock["machine"] not in MACHINES:
        problems.append(f"lock: unknown machine {lock['machine']!r} (known: {', '.join(MACHINES)})")
    gpu = lock["gpu"]
    if need(gpu, GPU_KEYS, "gpu"):
        sel = gpu["select"]
        if sel.get("by") == "uuid":
            u = sel.get("uuids")
            if not (isinstance(u, list) and u and all(isinstance(x, str) and x.startswith("GPU-") for x in u)):
                problems.append("gpu.select: 'uuids' must be a non-empty list of GPU-... UUIDs")
        elif sel.get("by") == "index":
            if not isinstance(sel.get("index"), int) or isinstance(sel.get("index"), bool):
                problems.append("gpu.select: 'index' must be an integer")
        else:
            problems.append("gpu.select: 'by' must be 'uuid' or 'index'")
        cp = gpu["clock_policy"]
        if cp.get("mode") == "locked":
            if not isinstance(cp.get("mhz"), int):
                problems.append("gpu.clock_policy: a locked policy needs an integer 'mhz'")
        elif cp.get("mode") != "natural":
            problems.append("gpu.clock_policy: 'mode' must be 'natural' or 'locked'")
        if not isinstance(cp.get("tolerance_mhz"), int):
            problems.append("gpu.clock_policy: 'tolerance_mhz' must be an integer")
        cm = gpu["compute_mode"]
        if cm is not None and (cm.get("set") not in MODE_ARG.values() or not isinstance(cm.get("restore"), bool)):
            problems.append(f"gpu.compute_mode: 'set' must be one of {sorted(MODE_ARG.values())} and 'restore' a boolean")
    if lock["sm"] == 90 and not lock["jit_compiler_sm90_prefix"]:
        problems.append("lock: an sm_90 lock must name the expected jit_compiler_sm90() prefix")
    for k, v in lock["bench_env"].items():
        if not (isinstance(k, str) and isinstance(v, str)):
            problems.append(f"bench_env: {k!r} must map to a string")
    if "main" not in lock["environments"]:
        problems.append("environments: there must be a 'main' environment")
    for name, e in lock["environments"].items():
        where = f"environments.{name}"
        if not isinstance(e, dict) or not need(e, ENV_KEYS, where):
            continue
        if not os.path.isabs(e["python"]):
            problems.append(f"{where}: 'python' must be an absolute path")
        if not e["packages"] or not all(isinstance(v, str) for v in e["packages"].values()):
            problems.append(f"{where}: 'packages' must map every distribution to an exact version string")
        if not all(isinstance(v, str) for v in e["env"].values()):
            problems.append(f"{where}: 'env' values must be strings")
        if any(k.startswith(UNSET_PREFIXES) or k in UNSET_EXACT for k in e["env"]):
            problems.append(f"{where}: 'env' may not set FSO_*, TRTLLM_DG_* or the variables the harness owns; "
                            "bench knobs go into bench_env, overlays into pythonpath_overlays")
        for p in e["path_prepend"] + e["pythonpath_overlays"]:
            if not (isinstance(p, str) and os.path.isabs(p)):
                problems.append(f"{where}: path entries must be absolute paths ({p!r})")
    return problems


# ----------------------------------------------------------------------------- the suite and what the benches accept
def script_knowledge(path):
    """What a bench script accepts, read from its source and never imported (the benches import torch): its
    options, its implementation names (KERNEL_IMPLS and LAYER_IMPLS), its dense families and its models. A
    front-end that runs another bench through runpy (bench_moe_qwen3_35a3.py) knows what the bench it runs knows."""
    with open(path) as f:
        src = f.read()
    m = re.search(r'run_path\(.*?"(bench_[A-Za-z0-9_]+\.py)"', src, re.S)
    if m:
        return script_knowledge(os.path.join(os.path.dirname(path), m.group(1)))
    know = {"options": set(re.findall(r'add_argument\(\s*"(--[A-Za-z0-9_-]+)"', src)), "impls": set(),
            "families": set(), "models": set()}
    for node in ast.parse(src).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name in ("KERNEL_IMPLS", "LAYER_IMPLS"):
            know["impls"].update(ast.literal_eval(node.value))
        elif name in ("FAMILIES", "MODELS") and isinstance(node.value, ast.Dict):
            keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            know["families" if name == "FAMILIES" else "models"].update(keys)
    return know


def validate_suite(machine, lock=None):
    """The problems of a machine's suite, as sentences: unknown scripts, options, implementations, families or
    projections, duplicate names or outputs, groups outside GROUPS, environments the lock lacks, and scripts
    without the --Ms option a smoke run needs."""
    problems = []
    steps = perf_suite.SUITES.get(machine)
    if not steps:
        return [f"no suite for machine {machine!r}"]
    seen_names, seen_outs, cache = set(), set(), {}
    for s in steps:
        where = f"{machine}/{s.get('name')}"
        for k in ("name", "env", "script", "args", "out", "group"):
            if k not in s:
                problems.append(f"{where}: missing {k!r}")
        if s["name"] in seen_names:
            problems.append(f"{where}: duplicate step name")
        if s["out"] in seen_outs:
            problems.append(f"{where}: duplicate output file {s['out']}")
        seen_names.add(s["name"])
        seen_outs.add(s["out"])
        if s["group"] not in GROUPS:
            problems.append(f"{where}: group {s['group']!r} is not one of {', '.join(GROUPS)}")
        if lock is not None and s["env"] not in lock["environments"]:
            problems.append(f"{where}: environment {s['env']!r} is not in the lock")
        if not s["out"].endswith(".jsonl") or os.sep in s["out"]:
            problems.append(f"{where}: output {s['out']!r} must be a bare .jsonl file name")
        script = os.path.join(ROOT, s["script"])
        if not os.path.isfile(script):
            problems.append(f"{where}: bench script {s['script']} does not exist")
            continue
        know = cache.setdefault(script, script_knowledge(script))
        for opt in ("--run", "--out", "--Ms"):
            if opt not in know["options"]:
                problems.append(f"{where}: {s['script']} has no {opt} option")
        args = list(s["args"])
        for i, a in enumerate(args):
            if not a.startswith("--"):
                continue
            if a not in know["options"]:
                problems.append(f"{where}: {s['script']} has no {a} option")
            if a in ("--out", "--Ms"):
                problems.append(f"{where}: {a} is added by run_perf.py and may not appear in the suite")
            val = args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else None
            if a == "--impls":
                for impl in (val or "").split(","):
                    if impl not in know["impls"]:
                        problems.append(f"{where}: {s['script']} does not know the implementation {impl!r}")
            elif a == "--family" and val not in know["families"]:
                problems.append(f"{where}: {s['script']} does not know the family {val!r}")
            elif a == "--model" and val not in know["models"]:
                problems.append(f"{where}: {s['script']} does not know the model {val!r}")
            elif a == "--projs":
                for p in (val or "").split(","):
                    if p not in KNOWN_PROJS:
                        problems.append(f"{where}: unknown projection {p!r}")
    return problems


def resolve_plan(lock, run_dir, tables, smoke):
    """The machine's steps in the selected table groups, each with its full command line."""
    plan = []
    for s in perf_suite.SUITES[lock["machine"]]:
        if s["group"] not in tables:
            continue
        e = lock["environments"][s["env"]]
        argv = [e["python"], os.path.join(ROOT, s["script"])] + list(s["args"]) + ["--out", os.path.join(run_dir, s["out"])]
        if smoke:
            argv += SMOKE_ARGS
        plan.append(dict(s, argv=argv))
    return plan


def step_environment(lock, env_name, fso_path, cvd, base=None):
    """(environment for a subprocess, the variables set, the inherited variables removed). See UNSET_* above."""
    base = dict(os.environ if base is None else base)
    e = lock["environments"][env_name]
    removed = sorted(k for k in base if k.startswith(UNSET_PREFIXES) or k in UNSET_EXACT
                     or (k in LOCK_CONTROLLED and k not in e["env"]))
    env = {k: v for k, v in base.items() if k not in removed}
    setv = {}
    setv.update(lock["bench_env"])
    setv.update(e["env"])
    setv["PATH"] = os.pathsep.join([os.path.dirname(e["python"])] + list(e["path_prepend"])
                                   + ([base["PATH"]] if base.get("PATH") else []))
    pythonpath = ([fso_path] if fso_path else []) + list(e["pythonpath_overlays"])
    if pythonpath:
        setv["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if cvd is not None:
        setv["CUDA_VISIBLE_DEVICES"] = cvd
    setv["PYTHONDONTWRITEBYTECODE"] = "1"
    setv["PYTHONUNBUFFERED"] = "1"
    env.update(setv)
    return env, setv, removed


# ----------------------------------------------------------------------------- nvidia-smi
CARD_FIELDS = ("index", "pci.bus_id", "uuid", "name", "driver_version", "compute_mode", "clocks.max.sm",
               "power.limit", "memory.total")


def nvsmi(args, timeout=60):
    p = subprocess.run(["nvidia-smi"] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"nvidia-smi {' '.join(args)} exited {p.returncode}: {(p.stderr or p.stdout).strip()[:300]}")
    return p.stdout


def query_cards():
    out = nvsmi(["--query-gpu=" + ",".join(CARD_FIELDS), "--format=csv,noheader,nounits"])
    cards = []
    for line in out.splitlines():
        if line.strip():
            vals = [v.strip() for v in line.split(",")]
            c = dict(zip(CARD_FIELDS, vals))
            cards.append({"index": c["index"], "pci": c["pci.bus_id"], "uuid": c["uuid"], "name": c["name"],
                          "driver": c["driver_version"], "compute_mode": c["compute_mode"],
                          "clocks_max_sm_mhz": _num(c["clocks.max.sm"]), "power_limit_w": _num(c["power.limit"]),
                          "memory_total_mib": _num(c["memory.total"])})
    return cards


def compute_apps(card_id):
    out = nvsmi(["-i", card_id, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    return [l.strip() for l in out.splitlines() if l.strip() and not l.lower().startswith("no running")]


def card_field(card_id, field):
    return nvsmi(["-i", card_id, f"--query-gpu={field}", "--format=csv,noheader,nounits"]).strip()


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() else f


def resolve_card(lock, cards, want_uuid=None):
    """(card, None) or (None, why not)."""
    sel = lock["gpu"]["select"]
    if sel["by"] == "uuid":
        present = [c for c in cards if c["uuid"] in sel["uuids"]]
        if want_uuid:
            if want_uuid not in sel["uuids"]:
                return None, f"--gpu {want_uuid} is not one of the lock's cards {sel['uuids']}"
            present = [c for c in present if c["uuid"] == want_uuid]
        if not present:
            return None, f"none of the lock's cards ({', '.join(sel['uuids'])}) is on this machine"
        if len(present) > 1:
            return None, "several of the lock's cards are present; choose one with --gpu"
        return present[0], None
    if want_uuid:
        return None, "--gpu applies to locks that select cards by UUID"
    card = next((c for c in cards if c["index"] == str(sel["index"])), None)
    if card is None:
        return None, f"there is no card at nvidia-smi index {sel['index']}"
    return card, None


# ----------------------------------------------------------------------------- preflight
# Runs in each environment's interpreter with no GPU visible. argv: the distribution names (JSON), then "1"/"0" for
# "import fish_scales_ops" and "1"/"0" for "ask it for the sm_90 JIT compiler". Prints one line, PROBE <json>.
PROBE_SRC = r'''
import hashlib, json, sys
out = {"executable": sys.executable, "prefix": sys.prefix, "python_version": "%d.%d.%d" % tuple(sys.version_info[:3])}
names, want_fso, want_jit = json.loads(sys.argv[1]), sys.argv[2] == "1", sys.argv[3] == "1"
from importlib import metadata
pk = {}
for n in names:
    try:
        pk[n] = metadata.version(n)
    except Exception:
        pk[n] = None
out["packages"] = pk
if want_fso:
    f = {}
    try:
        import fish_scales_ops as fso
        import torch
        f["file"] = fso.__file__
        f["version"] = getattr(fso, "__version__", None)
        f["torch"] = torch.__version__
        bi = getattr(fso, "build_info", None)
        if callable(bi):
            try:
                f["build_info"] = bi()
            except Exception as e:
                f["build_info"] = None
                f["build_info_error"] = "%s: %s" % (type(e).__name__, e)
        else:
            f["build_info"] = None
        ext = getattr(getattr(fso, "_C", None), "__file__", None)
        f["extension"] = ext
        if ext:
            h = hashlib.sha256()
            with open(ext, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            f["extension_sha256"] = h.hexdigest()
        if want_jit:
            try:
                f["jit_compiler_sm90"] = torch.ops.fish_scales_ops.jit_compiler_sm90()
            except Exception as e:
                f["jit_compiler_sm90"] = None
                f["jit_compiler_sm90_error"] = "%s: %s" % (type(e).__name__, " ".join(str(e).split())[:300])
    except BaseException as e:
        f["import_error"] = "%s: %s" % (type(e).__name__, " ".join(str(e).split())[:600])
    out["fso"] = f
print("PROBE " + json.dumps(out, default=str))
'''


def run_probe(lock, env_name, fso_path, timeout):
    e = lock["environments"][env_name]
    env, _, _ = step_environment(lock, env_name, fso_path, cvd="")    # no GPU visible to the probe
    want_jit = bool(e["fso"] and lock["jit_compiler_sm90_prefix"])
    cmd = [e["python"], "-c", PROBE_SRC, json.dumps(sorted(e["packages"])), "1" if e["fso"] else "0",
           "1" if want_jit else "0"]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=env,
                           timeout=timeout, cwd="/")
    except subprocess.TimeoutExpired:
        return None, f"the probe did not finish within {timeout} s"
    line = next((l for l in p.stdout.splitlines() if l.startswith("PROBE ")), None)
    if line is None:
        tail = " ".join((p.stderr or "").strip().splitlines()[-3:])
        return None, f"the probe exited {p.returncode} without a result: {tail[:400]}"
    return json.loads(line[6:]), None


def fso_identity(f):
    """The fish_scales_ops identity of one probe, as the manifest records it."""
    bi = f.get("build_info")
    source_build = not isinstance(bi, dict) or bool(bi.get("source_build"))
    return {"version": f.get("version"), "build_info": bi, "source_build": source_build,
            "commit": bi.get("commit") if isinstance(bi, dict) else None, "file": f.get("file"),
            "extension": f.get("extension"), "extension_sha256": f.get("extension_sha256"),
            "jit_compiler_sm90": f.get("jit_compiler_sm90"), "torch": f.get("torch")}


def preflight(lock, plan, fso_path, want_uuid, probe_timeout):
    """Everything the run checks before it touches the card. Returns a dict with the versions found per
    environment, the card, the fish_scales_ops identity, and the two lists of findings: drift (versions,
    driver, compiler; --allow-drift may accept it) and blockers (never accepted)."""
    res = {"host": socket.gethostname(), "environments": {}, "card": None, "fso": None, "drift": [], "blockers": []}
    drift, block = res["drift"], res["blockers"]
    if not fnmatch.fnmatchcase(res["host"], lock["host_pattern"]):
        block.append({"what": "host", "lock": lock["host_pattern"], "found": res["host"],
                      "text": f"this host is {res['host']}, not the lock's machine ({lock['host_pattern']})"})
    if fso_path and not os.path.isfile(os.path.join(fso_path, "fish_scales_ops", "__init__.py")):
        block.append({"what": "fso_path", "found": fso_path,
                      "text": f"--fso-path {fso_path} holds no fish_scales_ops package"})
    env_names = []
    for s in plan:
        if s["env"] not in env_names:
            env_names.append(s["env"])
    identities = {}
    for name in env_names:
        e = lock["environments"][name]
        if not os.path.isfile(e["python"]):
            block.append({"what": "interpreter", "env": name, "found": None,
                          "text": f"{name}: the interpreter {e['python']} does not exist"})
            continue
        for ov in e["pythonpath_overlays"]:
            if not os.path.isdir(ov):
                block.append({"what": "overlay", "env": name, "found": ov,
                              "text": f"{name}: the PYTHONPATH overlay {ov} does not exist"})
        probe, err = run_probe(lock, name, fso_path, probe_timeout)
        if probe is None:
            block.append({"what": "probe", "env": name, "text": f"{name}: {err}"})
            continue
        res["environments"][name] = {"python": e["python"], "executable": probe["executable"],
                                     "prefix": probe["prefix"], "python_version": probe["python_version"],
                                     "packages": probe["packages"]}
        if probe["python_version"] != e["python_version"]:
            drift.append({"what": "python_version", "env": name, "lock": e["python_version"],
                          "found": probe["python_version"],
                          "text": f"{name}: Python {probe['python_version']} (lock {e['python_version']})"})
        for pkg, want in e["packages"].items():
            got = probe["packages"].get(pkg)
            if got != want:
                drift.append({"what": "package", "env": name, "name": pkg, "lock": want, "found": got,
                              "text": f"{name}: {pkg} {got or 'not installed'} (lock {want})"})
        if not e["fso"]:
            continue
        f = probe.get("fso") or {}
        res["environments"][name]["fso"] = f
        if f.get("import_error"):
            block.append({"what": "fso_import", "env": name, "found": f["import_error"],
                          "text": f"{name}: fish_scales_ops does not import: {f['import_error']}"})
            continue
        if fso_path and not os.path.realpath(f.get("file") or "").startswith(os.path.realpath(fso_path) + os.sep):
            block.append({"what": "fso_origin", "env": name, "found": f.get("file"),
                          "text": f"{name}: fish_scales_ops came from {f.get('file')}, not from --fso-path {fso_path}"})
        prefix = lock["jit_compiler_sm90_prefix"]
        if prefix:
            got = f.get("jit_compiler_sm90")
            if not (got or "").startswith(prefix):
                found = got if got else "unavailable (" + str(f.get("jit_compiler_sm90_error")) + ")"
                drift.append({"what": "jit_compiler_sm90", "env": name, "lock": prefix + "...", "found": found,
                              "text": f"{name}: jit_compiler_sm90() is {found!r} (lock: starts with {prefix!r})"})
        identities[name] = fso_identity(f)
    if identities:
        first = "main" if "main" in identities else next(iter(identities))
        res["fso"] = identities[first]
        shas = {n: i["extension_sha256"] for n, i in identities.items()}
        if len(set(shas.values())) > 1:
            drift.append({"what": "fso_extension", "lock": "one extension in every environment", "found": shas,
                          "text": "the environments import different fish_scales_ops extensions: "
                                  + ", ".join(f"{n} {s[:12] if s else None}" for n, s in shas.items())})
    try:
        cards = query_cards()
    except Exception as e:  # noqa: BLE001 - reported as a blocker
        block.append({"what": "nvidia-smi", "text": f"nvidia-smi failed: {e}"})
        return res
    card, err = resolve_card(lock, cards, want_uuid)
    if card is None:
        block.append({"what": "card", "text": err})
        return res
    res["card"] = card
    if card["name"] != lock["device_name"]:
        block.append({"what": "card_name", "lock": lock["device_name"], "found": card["name"],
                      "text": f"the selected card is a {card['name']}, not a {lock['device_name']}"})
    if card["driver"] != lock["driver"]:
        drift.append({"what": "driver", "lock": lock["driver"], "found": card["driver"],
                      "text": f"driver {card['driver']} (lock {lock['driver']})"})
    try:
        apps = compute_apps(card["uuid"])
    except Exception as e:  # noqa: BLE001
        apps = [f"(query failed: {e})"]
    card["compute_apps"] = apps
    if apps:
        block.append({"what": "card_busy", "found": apps,
                      "text": f"the card has {len(apps)} compute process(es): {'; '.join(apps)}"})
    return res


def verdict(pre, allow_drift):
    """(proceed?, sentence)."""
    if pre["blockers"]:
        return False, f"REFUSED: {len(pre['blockers'])} blocker(s)" + (f" and {len(pre['drift'])} drift item(s)" if pre["drift"] else "")
    if pre["drift"] and not allow_drift:
        return False, f"REFUSED: {len(pre['drift'])} drift item(s) against the lock (--allow-drift records them and proceeds)"
    if pre["drift"]:
        return True, f"PROCEED WITH DRIFT: {len(pre['drift'])} drift item(s) accepted by --allow-drift and recorded in the manifest"
    return True, "PASS: the environment matches the lock"


# ----------------------------------------------------------------------------- the tree that holds the benches
def bench_tree_identity():
    """Where the benches come from: the git commit of the work tree (with a dirty flag for bench/ and
    tests/baselines/), or, in a test kit made with git archive, the commit git wrote into bench/env/KIT_COMMIT
    (export-subst); plus a sha256 over the harness, the suite, the locks and the bench scripts, which identifies
    the benches whatever the source."""
    ident = {"root": ROOT, "source": "unknown", "commit": None, "bench_dirty": None}
    try:
        top = subprocess.run(["git", "-C", ROOT, "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, universal_newlines=True, timeout=30)
        if top.returncode == 0 and os.path.realpath(top.stdout.strip()) == os.path.realpath(ROOT):
            head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], stdout=subprocess.PIPE,
                                  universal_newlines=True, timeout=30).stdout.strip()
            st = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "bench", "tests/baselines"],
                                stdout=subprocess.PIPE, universal_newlines=True, timeout=60).stdout.splitlines()
            ident.update(source="git", commit=head, bench_dirty=bool(st), dirty_paths=[l[3:] for l in st][:40])
    except (OSError, subprocess.SubprocessError):
        pass
    if ident["source"] == "unknown" and os.path.isfile(KIT_COMMIT_FILE):
        kit = open(KIT_COMMIT_FILE).read().strip()
        if re.fullmatch(r"[0-9a-f]{40}", kit):
            ident.update(source="kit", commit=kit)
    h = hashlib.sha256()
    files = [os.path.join(HERE, "run_perf.py"), os.path.join(HERE, "perf_suite.py")]
    files += sorted(os.path.join(ENV_DIR, f) for f in os.listdir(ENV_DIR) if f.endswith(".json"))
    pydir = os.path.join(HERE, "gemm", "python")
    files += sorted(os.path.join(pydir, f) for f in os.listdir(pydir) if f.endswith(".py"))
    for p in files:
        h.update(rel(p).encode())
        with open(p, "rb") as f:
            h.update(f.read())
    ident["fingerprint"] = h.hexdigest()
    ident["fingerprint_files"] = len(files)
    return ident


# ----------------------------------------------------------------------------- printing the plan
def print_plan(lock, lock_file, lock_sha, plan, envs, run_dir, tables, smoke, fso_path, card, tree):
    gpu = lock["gpu"]
    cp, cm = gpu["clock_policy"], gpu["compute_mode"]
    log(f"machine {lock['machine']} ({lock['arch']}, tables tagged sm{lock['sm']}); lock {rel(lock_file)} sha256 {lock_sha[:16]}")
    origin = {"git": f"git commit {tree['commit']}" + (", bench/ or tests/baselines/ modified" if tree.get("bench_dirty") else ""),
              "kit": f"test kit of commit {tree['commit']}"}.get(
        tree["source"], "neither a git work tree nor a kit with a commit; identified by the bench fingerprint only")
    log(f"tree {ROOT} ({origin}); bench fingerprint {tree['fingerprint'][:16]}")
    smoke_text = "yes (M in {" + SMOKE_MS + "} only, never installable)" if smoke else "no"
    log(f"run directory {run_dir}; tables {', '.join(tables)}; smoke {smoke_text}")
    log(f"fish_scales_ops path {fso_path or '(none: the environments must provide fish_scales_ops)'}")
    if card:
        log(f"card {card['uuid']}, PCI {card['pci']}, index {card['index']}, {card['name']}, driver {card['driver']}, "
            f"compute mode {card['compute_mode']}, max SM clock {card['clocks_max_sm_mhz']} MHz")
    else:
        log("card: not resolved (see the preflight)")
    if cp["mode"] == "locked":
        log(f"clock policy: locked at {cp['mhz']} MHz for the run (nvidia-smi -lgc, {'sudo -n' if cp.get('sudo') else 'no sudo'}), "
            f"verified at +-{cp['tolerance_mhz']} MHz, released (-rgc) at the end")
    else:
        log(f"clock policy: natural, nothing applied; a step whose busy samples free of power, thermal and slowdown "
            f"limits never reach the maximum SM clock (within {cp['tolerance_mhz']} MHz) is flagged")
    log("compute mode: " + (f"set to {cm['set']} for the run" + (", the mode found restored after it" if cm.get("restore") else "")
                            if cm else "left as found"))
    log(f"GPU lock file {gpu['lock_file']} (flock); clock sampler every {SAMPLE_MS} ms -> clocks.csv")
    log(f"bench environment: {' '.join(f'{k}={v}' for k, v in lock['bench_env'].items()) or '(none)'}")
    for name, (setv, removed) in envs.items():
        log(f"environment {name}: {lock['environments'][name]['python']}")
        log("    sets " + " ".join(f"{k}={shlex.quote(v)}" for k, v in setv.items()))
        log("    unsets " + (", ".join(removed) if removed else "(nothing inherited needed removing)")
            + f"; FSO_*, TRTLLM_DG_* and {', '.join(UNSET_EXACT)} are never inherited")
    log(f"plan: {len(plan)} step(s), each after an idle check of the card, output in {run_dir}")
    for i, s in enumerate(plan, 1):
        log(f"  {i:2d}. {s['name']} [{s['group']}] env {s['env']} -> {s['out']}")
        log(f"      {shlex.join(s['argv'])}")
    log("after the steps:")
    for cmd in post_commands(lock, run_dir):
        log(f"      {shlex.join(cmd)}")


def print_preflight(pre, allow_drift):
    for name, e in pre["environments"].items():
        pk = ", ".join(f"{k} {v}" for k, v in sorted(e["packages"].items()))
        log(f"preflight {name}: Python {e['python_version']} ({e['prefix']}); {pk}")
        f = e.get("fso")
        if f and not f.get("import_error"):
            bi = f.get("build_info")
            log(f"    fish_scales_ops {f.get('version')} from {f.get('file')}; "
                + ("build_info() " + json.dumps(bi, default=str) if bi else "no build_info() (a source build)")
                + f"; extension sha256 {(f.get('extension_sha256') or '?')[:16]}"
                + (f"; jit_compiler_sm90() {f['jit_compiler_sm90']!r}" if f.get("jit_compiler_sm90") else ""))
    c = pre.get("card")
    if c:
        log(f"preflight card: {c['uuid']} {c['pci']} {c['name']} driver {c['driver']}, "
            f"compute processes: {len(c.get('compute_apps') or [])}")
    for d in pre["drift"]:
        log(f"DRIFT   {d['text']}")
    for b in pre["blockers"]:
        log(f"BLOCKER {b['text']}")
    ok, text = verdict(pre, allow_drift)
    log(f"preflight verdict: {text}")
    return ok


def post_commands(lock, run_dir):
    dev, sm = lock["machine"], str(lock["sm"])
    merged = os.path.join(run_dir, "merged")
    return [[sys.executable, PERF_REPORT, "merge", "--run", run_dir, "--out", merged, "--device", dev, "--sm", sm],
            [sys.executable, PERF_REPORT, "diff", "--a", "baselines", "--b", merged, "--device", dev, "--sm", sm]]


# ----------------------------------------------------------------------------- the run
class Sampler:
    """nvidia-smi every SAMPLE_MS on the card, each line stamped with the epoch time it arrived (nvidia-smi flushes
    every line, and its own timestamp is local time) and kept in memory for the per-step statistics."""
    FIELDS = "timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active"

    def __init__(self, card_id, path):
        self.samples = []
        self.lock = threading.Lock()
        self.f = open(path, "w")
        self.f.write("epoch," + self.FIELDS + "\n")
        self.proc = subprocess.Popen(["nvidia-smi", "-i", card_id, "--query-gpu=" + self.FIELDS,
                                      "--format=csv,noheader,nounits", "-lms", str(SAMPLE_MS)],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True,
                                     bufsize=1)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            t = time.time()
            line = line.strip()
            if not line:
                continue
            self.f.write(f"{t:.3f},{line}\n")
            self.f.flush()
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 7:
                continue
            reasons = int(parts[6], 16) if parts[6].startswith("0x") else None
            with self.lock:   # (epoch, SM MHz, W, deg C, utilization %, event reasons)
                self.samples.append((t, _num(parts[1]), _num(parts[3]), _num(parts[4]), _num(parts[5]), reasons))

    def window(self, t0, t1):
        with self.lock:
            return [s for s in self.samples if t0 <= s[0] <= t1]

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.thread.join(timeout=10)
        self.f.close()


def is_busy(s):
    """A sample in which kernels ran: GPU utilization above zero. The idle event reason is the fallback only,
    because a card under a clock lock never reports it (the RTX 5090 logs of 2026-10-01 read 0x0 at idle)."""
    if s[4] is not None:
        return s[4] > 0
    return s[5] is not None and not s[5] & REASON_IDLE


def clock_stats(samples, policy, max_sm):
    """Per-step clock statistics. The median, minimum and maximum SM clock and the power-capped share are taken
    over the busy samples (is_busy); samples between cells, while a worker starts, are left out."""
    busy = [s for s in samples if is_busy(s)]
    sms = [s[1] for s in busy if s[1] is not None]

    def count(bit):
        return sum(1 for s in busy if s[5] is not None and s[5] & bit)
    st = {"samples": len(samples), "busy_samples": len(busy),
          "sm_mhz_median_busy": statistics.median(sms) if sms else None,
          "sm_mhz_min_busy": min(sms) if sms else None, "sm_mhz_max_busy": max(sms) if sms else None,
          "power_capped_busy_samples": count(REASON_SW_POWER_CAP),
          "power_capped_share_of_busy": round(count(REASON_SW_POWER_CAP) / len(busy), 4) if busy else None,
          "hw_slowdown_busy_samples": count(REASON_HW_SLOWDOWN | REASON_HW_POWER_BRAKE),
          "thermal_busy_samples": count(REASON_SW_THERMAL | REASON_HW_THERMAL),
          "power_w_max": max((s[2] for s in samples if s[2] is not None), default=None),
          "temperature_c_max": max((s[3] for s in samples if s[3] is not None), default=None)}
    flags = []
    tol = policy["tolerance_mhz"]
    # A natural-clock card runs at its maximum clock unless a limiter holds it lower, so the busy samples that no
    # limiter flagged must reach that maximum; if they do not, something (a clock lock) holds the card lower.
    limiters = REASON_SW_POWER_CAP | REASON_HW_SLOWDOWN | REASON_SW_THERMAL | REASON_HW_THERMAL | REASON_HW_POWER_BRAKE
    free = [s[1] for s in busy if s[1] is not None and not (s[5] or 0) & limiters]
    if free and policy["mode"] == "natural" and max_sm and max(free) < max_sm - tol:
        flags.append(f"no busy sample free of power, thermal and slowdown limits reached the card's maximum SM clock "
                     f"{max_sm} MHz (highest {max(free)} MHz): a clock lock may be in force")
    if sms and policy["mode"] == "locked" and abs(st["sm_mhz_median_busy"] - policy["mhz"]) > tol:
        flags.append(f"the busy median {st['sm_mhz_median_busy']} MHz is not within {tol} MHz of the lock {policy['mhz']} MHz")
    if not busy:
        flags.append("no busy sample in the step's window")
    return st, flags


def count_rows(path):
    """(rows, rows with a timing, rows with an error) of a bench output; the meta row is not counted."""
    rows = timed = errors = 0
    if not os.path.isfile(path):
        return None, None, None
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("kind") == "meta" or ("_device" in r and "M" not in r):
                continue
            rows += 1
            nested = [v for v in r.values() if isinstance(v, dict)]
            if r.get("us") is not None or any(v.get("us") is not None or v.get("graph_us") is not None for v in nested):
                timed += 1
            if "error" in r or any("error" in v for v in nested):
                errors += 1
    return rows, timed, errors


def run_step(s, env, run_dir, timeout):
    """One bench invocation in its own session, its output timestamped line by line into logs/<name>.log."""
    log_path = os.path.join(run_dir, "logs", s["name"] + ".log")
    rec = {"start_utc": iso(), "log": os.path.relpath(log_path, run_dir)}
    t0 = time.time()
    proc = subprocess.Popen(s["argv"], cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1, errors="replace",
                            start_new_session=True)
    timed_out = []

    def kill(sig=signal.SIGTERM):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass

    def on_timeout():
        timed_out.append(True)
        kill()
        time.sleep(30)
        if proc.poll() is None:      # never signal the group id once the step is gone: the id may be reused
            kill(signal.SIGKILL)
    timer = threading.Timer(timeout, on_timeout)
    timer.daemon = True
    timer.start()
    busy = failures = 0
    try:
        with open(log_path, "w") as lf:
            for line in proc.stdout:
                lf.write(f"{iso()} {line}")
                lf.flush()
                if FAILURE_TEXT.search(line):
                    failures += 1
                if DEVICE_BUSY_TEXT.search(line):
                    busy += 1
        rc = proc.wait()
    except BaseException:
        kill()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            kill(signal.SIGKILL)
        raise
    finally:
        timer.cancel()
    t1 = time.time()
    rows, timed, errors = count_rows(os.path.join(run_dir, s["out"]))
    rec.update(end_utc=iso(), seconds=round(t1 - t0, 1), exit_code=rc, timed_out=bool(timed_out), rows=rows,
               timed_rows=timed, error_rows=errors, failure_lines=failures, device_busy_lines=busy)
    return rec, t0, t1


def acquire_flock(path, wait_s):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
    deadline = time.time() + wait_s
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError as e:
            if e.errno not in (errno.EAGAIN, errno.EACCES):
                os.close(fd)
                raise
        if time.time() >= deadline:
            os.close(fd)
            return None
        time.sleep(2)


def sudo_prefix(policy):
    return ["sudo", "-n"] if policy and policy.get("sudo") else []


def run_cmd(cmd, timeout=120):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
    return p.returncode, p.stdout.strip()


def apply_compute_mode(lock, card, rec):
    cm = lock["gpu"]["compute_mode"]
    found = card_field(card["uuid"], "compute_mode")
    rec.update(policy=cm, found=found)
    if not cm:
        rec["applied"] = "left as found"
        return None
    if MODE_ARG.get(found) == cm["set"]:
        rec["applied"] = f"already {found}"
        return None
    cmd = sudo_prefix(cm) + ["nvidia-smi", "-i", card["uuid"], "-c", cm["set"]]
    rc, out = run_cmd(cmd)
    now = card_field(card["uuid"], "compute_mode")
    rec.update(applied=shlex.join(cmd), exit_code=rc, now=now)
    if rc != 0 or MODE_ARG.get(now) != cm["set"]:
        raise Refused(EXIT_POLICY, f"could not set compute mode {cm['set']} (exit {rc}: {out[:200]}); the card reads {now}")
    if cm.get("restore") and found in MODE_ARG:
        return sudo_prefix(cm) + ["nvidia-smi", "-i", card["uuid"], "-c", MODE_ARG[found]]
    return None


def apply_clock_policy(lock, card, rec):
    cp = lock["gpu"]["clock_policy"]
    rec.update(policy=cp, max_sm_mhz=card["clocks_max_sm_mhz"])
    if cp["mode"] == "natural":
        rec["applied"] = "nothing: natural clock"
        return None
    cmd = sudo_prefix(cp) + ["nvidia-smi", "-i", card["pci"], "-lgc", str(cp["mhz"])]
    rc, out = run_cmd(cmd)
    release = sudo_prefix(cp) + ["nvidia-smi", "-i", card["pci"], "-rgc"]
    rec.update(applied=shlex.join(cmd), exit_code=rc, output=out[:300])
    if rc != 0:
        raise Refused(EXIT_POLICY, f"could not lock the clock (exit {rc}): {out[:200]}")
    time.sleep(2)
    clk = _num(card_field(card["uuid"], "clocks.sm"))
    rec["verify_sm_mhz_after_lock"] = clk
    if clk is None or abs(clk - cp["mhz"]) > cp["tolerance_mhz"]:
        # the release still runs: the caller's finally gets the command through the exception
        raise Refused(EXIT_POLICY, f"the clock lock did not take: the card reads {clk} MHz, the lock is {cp['mhz']} MHz")
    return release


def _on_signal(signum, frame):
    raise Interrupted(f"signal {signum}")


def run(args, lock, lock_file, lock_sha, plan, envs, pre, tree, run_dir):
    os.makedirs(os.path.join(run_dir, "logs"))
    log.open(os.path.join(run_dir, "run_perf.log"))
    card = pre["card"]
    man = {"schema": 1, "tool": "bench/run_perf.py", "machine": lock["machine"], "host": pre["host"],
           "run_dir": run_dir, "start_utc": iso(), "end_utc": None, "status": "running",
           "smoke": bool(args.smoke), "smoke_ms": SMOKE_MS if args.smoke else None, "allow_drift": bool(args.allow_drift),
           "tables": list(args.tables), "argv": sys.argv, "harness_python": sys.executable,
           "lock_file": lock_file, "lock_sha256": lock_sha, "lock": lock,
           "suite": {"file": rel(perf_suite.__file__), "sha256": sha256_file(perf_suite.__file__)},
           "bench_tree": tree, "fso_path": args.fso_path,
           "preflight": pre, "drift": pre["drift"], "fso": pre["fso"],
           "card": {k: card[k] for k in ("uuid", "pci", "index", "name", "driver", "compute_mode", "clocks_max_sm_mhz",
                                         "power_limit_w", "memory_total_mib")},
           "clock_policy": {}, "compute_mode": {}, "gpu_lock": {"file": lock["gpu"]["lock_file"]},
           "sampler": {"file": "clocks.csv", "interval_ms": SAMPLE_MS, "fields": "epoch," + Sampler.FIELDS,
                       "busy": "a sample with utilization.gpu above 0; the per-step clock statistics use busy samples only"},
           "environments": {n: {"python": lock["environments"][n]["python"], "set": sv, "removed": rm}
                            for n, (sv, rm) in envs.items()},
           "steps": [], "post": {}}
    mpath = os.path.join(run_dir, "manifest.json")
    write_json_atomic(mpath, man)
    if args.smoke:
        with open(os.path.join(run_dir, "SMOKE_RUN_NOT_INSTALLABLE"), "w") as f:
            f.write("A --smoke run of bench/run_perf.py: M in {" + SMOKE_MS + "} only. perf_report.py install refuses it.\n")
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _on_signal)
    code, release_clock, restore_mode, sampler, fd = 0, None, None, None, None
    try:
        t_lock = time.time()
        log(f"waiting up to {args.lock_wait} s for the GPU lock {lock['gpu']['lock_file']}")
        fd = acquire_flock(lock["gpu"]["lock_file"], args.lock_wait)
        man["gpu_lock"]["waited_s"] = round(time.time() - t_lock, 1)
        if fd is None:
            raise Refused(EXIT_LOCK, f"the GPU lock {lock['gpu']['lock_file']} stayed busy for {args.lock_wait} s")
        log(f"GPU lock held after {man['gpu_lock']['waited_s']} s")
        apps = compute_apps(card["uuid"])
        if apps:
            raise Refused(EXIT_BUSY, f"the card has compute process(es), not queueing behind them: {'; '.join(apps)}")
        restore_mode = apply_compute_mode(lock, card, man["compute_mode"])
        log(f"compute mode: {man['compute_mode'].get('applied')}")
        try:
            release_clock = apply_clock_policy(lock, card, man["clock_policy"])
        except Refused:
            if lock["gpu"]["clock_policy"]["mode"] == "locked":
                cp = lock["gpu"]["clock_policy"]
                release_clock = sudo_prefix(cp) + ["nvidia-smi", "-i", card["pci"], "-rgc"]
            raise
        log(f"clock policy: {man['clock_policy'].get('applied')}"
            + (f"; the card reads {man['clock_policy'].get('verify_sm_mhz_after_lock')} MHz" if release_clock else ""))
        sampler = Sampler(card["uuid"], os.path.join(run_dir, "clocks.csv"))
        time.sleep(1)
        write_json_atomic(mpath, man)
        for i, s in enumerate(plan, 1):
            apps = compute_apps(card["uuid"])
            if apps:
                raise Refused(EXIT_BUSY, f"before step {s['name']}: the card has compute process(es): {'; '.join(apps)}")
            env, _, _ = step_environment(lock, s["env"], args.fso_path, card["uuid"])
            log(f"[{i}/{len(plan)}] start {s['name']} ({s['env']}): {shlex.join(s['argv'])}")
            rec, t0, t1 = run_step(s, env, run_dir, args.step_timeout)
            stats, flags = clock_stats(sampler.window(t0, t1), lock["gpu"]["clock_policy"], card["clocks_max_sm_mhz"])
            entry = {k: s[k] for k in ("name", "group", "env", "script", "args", "out")}
            entry.update(command=s["argv"], **rec)
            entry.update(clocks=stats, clock_flags=flags)
            man["steps"].append(entry)
            write_json_atomic(mpath, man)
            log(f"[{i}/{len(plan)}] end {s['name']}: exit {rec['exit_code']}{' (timed out)' if rec['timed_out'] else ''}, "
                f"{rec['rows']} rows ({rec['timed_rows']} timed, {rec['error_rows']} with an error), "
                f"{rec['failure_lines']} failure lines, {rec['seconds']} s; SM median {stats['sm_mhz_median_busy']} "
                f"min {stats['sm_mhz_min_busy']} MHz, power-capped {stats['power_capped_share_of_busy']} of busy samples"
                + "".join(f"; FLAG {x}" for x in flags))
            if rec["device_busy_lines"]:
                raise Refused(EXIT_ABORTED, f"step {s['name']} reported the device busy or unavailable; the run stops")
        man["status"] = "complete"
    except Refused as e:
        code = e.code
        man["status"] = f"aborted: {e}"
        log(f"ABORT: {e}")
    except (Interrupted, KeyboardInterrupt) as e:
        code = EXIT_ABORTED
        why = str(e) or "SIGINT"
        man["status"] = f"aborted: interrupted ({why})"
        log(f"ABORT: interrupted ({why})")
    finally:
        if sampler:
            sampler.stop()
        if release_clock:
            rc, out = run_cmd(release_clock)
            try:
                now = card_field(card["uuid"], "clocks.sm")
            except Exception:  # noqa: BLE001
                now = "?"
            man["clock_policy"]["released"] = {"command": shlex.join(release_clock), "exit_code": rc, "sm_mhz_after": now}
            log(f"clock lock released ({shlex.join(release_clock)}: exit {rc}); the card reads {now} MHz")
        if restore_mode:
            try:
                apps = compute_apps(card["uuid"])
            except Exception as e:  # noqa: BLE001
                apps = [f"(query failed: {e})"]
            if apps:
                man["compute_mode"]["restored"] = f"not restored: the card has compute process(es) {apps}"
            else:
                rc, out = run_cmd(restore_mode)
                man["compute_mode"]["restored"] = {"command": shlex.join(restore_mode), "exit_code": rc,
                                                   "now": card_field(card["uuid"], "compute_mode")}
            log(f"compute mode restore: {man['compute_mode']['restored']}")
        if fd is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            log("GPU lock released")
        man["end_utc"] = iso()
        write_json_atomic(mpath, man)
    if man["steps"]:
        for name, cmd in zip(("merge", "diff"), post_commands(lock, run_dir)):
            out_name = "merge.txt" if name == "merge" else "diff_vs_baselines.txt"
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            with open(os.path.join(run_dir, out_name), "w") as f:
                f.write(p.stdout)
            man["post"][name] = {"command": cmd, "exit_code": p.returncode, "output": out_name}
            log(f"{name}: exit {p.returncode} -> {out_name}")
        write_json_atomic(mpath, man)
    failed = [s["name"] for s in man["steps"] if s["exit_code"] != 0]
    log(f"run {man['status']}: {len(man['steps'])} of {len(plan)} step(s) ran"
        + (f", {len(failed)} exited non-zero ({', '.join(failed)})" if failed else "")
        + f"; manifest {mpath}" + (" (smoke: never installable)" if args.smoke else ""))
    return code


# ----------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", choices=MACHINES, help="the machine whose lock and suite to use")
    ap.add_argument("--lock", help="a lock file other than bench/env/<machine>.lock.json (its machine must match)")
    ap.add_argument("--out", required=True, help="the run directory; it must not exist or be empty")
    ap.add_argument("--tables", default="all", help=f"comma list of table groups ({', '.join(GROUPS)}) or 'all'")
    ap.add_argument("--fso-path", help="directory put first on PYTHONPATH; it holds the fish_scales_ops package to measure")
    ap.add_argument("--smoke", action="store_true", help=f"every step on M in {{{SMOKE_MS}}} only; never installable")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the preflight result, then stop")
    ap.add_argument("--allow-drift", action="store_true", help="proceed despite drift and record it in the manifest")
    ap.add_argument("--gpu", help="the UUID to use when the lock lists several cards")
    ap.add_argument("--lock-wait", type=int, default=600, help="seconds to wait for the GPU lock file (default 600)")
    ap.add_argument("--step-timeout", type=int, default=10800, help="seconds before a step is killed (default 10800)")
    ap.add_argument("--probe-timeout", type=int, default=600, help="seconds per preflight probe (default 600)")
    args = ap.parse_args(argv)
    if not args.machine and not args.lock:
        ap.error("--machine or --lock is required")
    tables = list(GROUPS) if args.tables == "all" else [t.strip() for t in args.tables.split(",") if t.strip()]
    bad = [t for t in tables if t not in GROUPS]
    if bad or not tables:
        ap.error(f"unknown table group(s) {bad}; known: {', '.join(GROUPS)} or all")
    args.tables = tables
    args.out = os.path.abspath(os.path.expanduser(args.out))
    if args.fso_path:
        args.fso_path = os.path.abspath(os.path.expanduser(args.fso_path))
    return args


def main(argv=None):
    args = parse_args(argv)
    lock_file = os.path.abspath(args.lock) if args.lock else lock_path_for(args.machine)
    lock, lock_sha = load_lock(lock_file)
    problems = validate_lock(lock)
    if args.machine and lock.get("machine") != args.machine:
        problems.append(f"the lock is for machine {lock.get('machine')!r}, not {args.machine!r}")
    problems += validate_suite(lock.get("machine"), lock) if not problems else []
    if problems:
        for p in problems:
            print(f"lock or suite problem: {p}", file=sys.stderr)
        return EXIT_USAGE
    if not args.dry_run and os.path.exists(args.out) and (not os.path.isdir(args.out) or os.listdir(args.out)):
        print(f"run directory {args.out} exists and is not empty; a run needs its own directory", file=sys.stderr)
        return EXIT_USAGE
    plan = resolve_plan(lock, args.out, args.tables, args.smoke)
    if not plan:
        print(f"no step of machine {lock['machine']} is in the tables {args.tables}", file=sys.stderr)
        return EXIT_USAGE
    tree = bench_tree_identity()
    log(f"run_perf.py {'dry run' if args.dry_run else 'run'} on {socket.gethostname()} at {iso()}")
    pre = preflight(lock, plan, args.fso_path, args.gpu, args.probe_timeout)
    cvd = pre["card"]["uuid"] if pre["card"] else "<card UUID>"
    envs = {}
    for s in plan:
        if s["env"] not in envs:
            envs[s["env"]] = step_environment(lock, s["env"], args.fso_path, cvd)[1:]
    print_plan(lock, lock_file, lock_sha, plan, envs, args.out, args.tables, args.smoke, args.fso_path, pre["card"], tree)
    ok = print_preflight(pre, args.allow_drift)
    if args.dry_run:
        log("preflight (json): " + json.dumps(pre, default=str, sort_keys=True))
        log("dry run: nothing was started" + ("" if ok else "; the run would refuse"))
        return 0 if ok else EXIT_REFUSED
    if not ok:
        log("the run refuses to start; nothing was locked or measured")
        return EXIT_REFUSED
    if args.allow_drift and pre["drift"]:
        log(f"--allow-drift: proceeding with {len(pre['drift'])} drift item(s); they go into the manifest")
    return run(args, lock, lock_file, lock_sha, plan, envs, pre, tree, args.out)


if __name__ == "__main__":
    sys.exit(main())
