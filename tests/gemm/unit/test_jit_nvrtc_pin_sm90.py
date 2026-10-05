#!/usr/bin/env python3
"""The sm_90 JIT compiles with the NVRTC 13.0.88 bundled with fish_scales_ops, never with the NVRTC torch loaded.

The extension does not link libnvrtc. On its first sm_90 compile the deep_gemm JIT (arch/sm90/fp8/jit/deep_gemm/
jit_utils.cuh and compiler.cuh) loads <package>/_nvrtc/libnvrtc.so.13, which scripts/vendor_nvrtc.py unpacks from the
pinned nvidia-cuda-nvrtc wheel, with dlopen(RTLD_NOW | RTLD_LOCAL), or the library FSO_JIT_NVRTC_LIB names, and calls
NVRTC only through the functions it resolved there. Nothing else is tried, so a missing library raises instead of
falling back to the libnvrtc.so.13 already in the process. This script checks:

  every architecture, on the built extension:
    1. `readelf -d` lists no libnvrtc among the NEEDED libraries;
    2. `nm -D --undefined-only` lists no nvrtc symbol. A direct call left behind would bind to torch's NVRTC at run
       time without any error;
  sm_90 only, each case in its own child process (the library is loaded once per process):
    3. default: torch's own libnvrtc.so.13 is loaded first with RTLD_GLOBAL. A dense FP8 GEMM and a small MoE layer
       then run. jit_compiler_sm90() must report NVRTC 13.0 and the bundled path, /proc/self/maps must hold the
       bundled libnvrtc.so.13 and libnvrtc-builtins.so.13.0, every cubin the run dumps must carry the 13.0 toolkit
       note, and both describe() texts must end with the compiler line. When torch's NVRTC has another version than
       the bundled one, it must also report its own version before and after the run;
    4. override: with FSO_JIT_NVRTC_LIB set to torch's library the same calls give bit-identical outputs and the
       bundled library is not mapped. Both cases run with FSO_JIT_DUMP_CUBIN=1 and a cache directory of their own.
       When torch's NVRTC has another version than the bundled one (torch 2.13.0+cu130 ships the same 13.0.88, so
       this part is skipped there with a note), exactly one stderr line carries the version notice and the cubin of
       every kernel differs between the cases in its cache key directory and in its bytes;
    5. missing library: FSO_JIT_NVRTC_LIB=/nonexistent/libnvrtc.so.13 makes jit_compiler_sm90() and the GEMM raise a
       RuntimeError that names the path and the remedy (no fallback), and describe() still returns, with the error in
       its compiler line.

usage: python tests/gemm/unit/test_jit_nvrtc_pin_sm90.py [--workdir DIR] [--keep]
"""
import argparse
import ctypes
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

NAME_DIR = re.compile(r"^([0-9a-f]{16})_(gemm_.+)$")
KERNEL = "nvrtc_kernel.cubin"
BUNDLED = (13, 0)
BUNDLED_TEXT = f"{BUNDLED[0]}.{BUNDLED[1]}"
NOTICE = f"the sm_90 kernels are validated and measured with NVRTC {BUNDLED_TEXT}"
DEBUG_LINE = "sm_90 JIT compiler: NVRTC"
REMEDY = ("scripts/vendor_nvrtc.py", "FSO_JIT_NVRTC_LIB")
MISSING = "/nonexistent/libnvrtc.so.13"
TOOLKIT_NOTE = re.compile(rb"Cuda compilation tools, release (\d+)\.(\d+), V[\d.]+")


class Failure(AssertionError):
    pass


def check(cond, msg):
    if not cond:
        raise Failure(msg)


def nvrtc_version(lib):
    major, minor = ctypes.c_int(), ctypes.c_int()
    rc = lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    if rc != 0:
        raise RuntimeError(f"nvrtcVersion returned {rc}")
    return major.value, minor.value


def mapped(fragment):
    """Real paths of the files mapped into this process whose path contains `fragment`."""
    out = set()
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.split(maxsplit=5)
            if len(parts) == 6 and fragment in parts[5]:
                out.add(os.path.realpath(parts[5].strip()))
    return sorted(out)


# ----------------------------------------------------------------------------------------------------- worker
def workload():
    """A dense FP8 GEMM at a swap-AB and a non-swap M, and a small MoE layer at a swap-AB and a non-swap M."""
    import torch
    import fish_scales_ops as fso

    outs = {}
    g = torch.Generator(device="cuda")
    g.manual_seed(5)
    w = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16, generator=g) / 45.0
    weight = fso.dense.prepare_weight(w, format="bsfp8")
    for M in (16, 128):
        g.manual_seed(100 + M)
        x = torch.randn(M, 2048, device="cuda", dtype=torch.bfloat16, generator=g)
        outs[f"dense_M{M}"] = fso.dense.linear(x, weight)

    E, TOPK, H, I = 32, 8, 2048, 512
    g.manual_seed(7)
    w13 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w13q, s13 = fso.compat.quantize_moe_weights_1x128_fp8_sm90(w13)
    w2q, s2 = fso.compat.quantize_moe_weights_1x128_fp8_sm90(w2)
    experts = fso.moe.prepare_experts(w13q, w2q, format="bsfp8", sw13=s13, sw2=s2)
    for M in (32, 512):  # 8 rows per expert: swap-AB; 128 rows per expert: the non-swap path
        g.manual_seed(1000 + M)
        hidden = torch.randn(M, H, device="cuda", dtype=torch.bfloat16, generator=g)
        ids = torch.rand(M, E, device="cuda", generator=g).topk(TOPK, dim=1).indices.to(torch.int32)
        tw = torch.softmax(torch.rand(M, TOPK, device="cuda", generator=g), dim=1).float()
        outs[f"moe_M{M}"] = fso.moe.layer(hidden, experts, ids, tw)
    torch.cuda.synchronize()
    return outs


def worker(case, out_path, torch_nvrtc):
    # torch's own NVRTC goes in first and globally, as in a serving process where torch has already loaded it.
    torch_lib, before = None, None
    if torch_nvrtc:
        torch_lib = ctypes.CDLL(torch_nvrtc, mode=os.RTLD_NOW | ctypes.RTLD_GLOBAL)
        before = nvrtc_version(torch_lib)
    import torch
    import fish_scales_ops as fso

    props = torch.cuda.get_device_properties(0)
    print(f"worker device: {props.name} sm_{props.major}{props.minor} pci_bus_id=0x{props.pci_bus_id:02X}", flush=True)
    res = dict(case=case, torch_nvrtc_before=before)
    if case == "missing":
        errors = {}
        try:
            res["compiler"] = torch.ops.fish_scales_ops.jit_compiler_sm90()
        except RuntimeError as e:
            errors["jit_compiler_sm90"] = str(e)
        try:
            workload()
        except RuntimeError as e:
            errors["gemm"] = str(e)
        res["errors"] = errors
    else:
        outs = workload()
        torch.save({k: v.cpu() for k, v in outs.items()}, out_path)
        res["compiler"] = torch.ops.fish_scales_ops.jit_compiler_sm90()
    res["describe"] = {"dense": fso.dense.describe().splitlines()[-1], "moe": fso.moe.describe().splitlines()[-1]}
    res["mapped_nvrtc"] = mapped("libnvrtc.so")
    res["mapped_builtins"] = mapped("libnvrtc-builtins")
    res["torch_nvrtc_after"] = nvrtc_version(torch_lib) if torch_lib is not None else None
    res["package_dir"] = os.path.dirname(os.path.abspath(fso.__file__))
    print("RESULT " + json.dumps(res), flush=True)


# ----------------------------------------------------------------------------------------------------- driver
def package_root():
    spec = importlib.util.find_spec("fish_scales_ops")
    if spec is None or spec.origin is None:
        raise SystemExit("fish_scales_ops is not importable (set PYTHONPATH=python from the repository root)")
    return os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))


def static_checks(so_path):
    """Checks 1 and 2: no link-time libnvrtc and no unresolved nvrtc symbol in the extension."""
    tools = [t for t in ("readelf", "nm") if shutil.which(t) is None]
    if tools:
        print(f"[static] SKIP: {', '.join(tools)} not on PATH (binutils), so the extension cannot be inspected")
        return
    dyn = subprocess.run(["readelf", "-d", so_path], capture_output=True, text=True, check=True).stdout
    needed = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", dyn)
    check(needed, f"readelf -d {so_path} lists no NEEDED entry at all: {dyn[:500]}")
    check(not [n for n in needed if "nvrtc" in n.lower()], f"the extension links libnvrtc: NEEDED {needed}")
    und = subprocess.run(["nm", "-D", "--undefined-only", so_path], capture_output=True, text=True, check=True).stdout
    nvrtc_syms = [ln.split()[-1] for ln in und.splitlines() if "nvrtc" in ln.lower()]
    check(not nvrtc_syms, f"the extension has unresolved NVRTC symbols, which would bind to torch's NVRTC: {nvrtc_syms}")
    print(f"[static] {os.path.basename(so_path)}: NEEDED {needed} holds no libnvrtc; no undefined nvrtc symbol")


def torch_nvrtc_path():
    """torch's own libnvrtc: the one `import torch` mapped (the cu130 wheel loads it eagerly), else the wheel layout
    next to torch. None when neither exists."""
    import torch  # noqa: F401
    for p in mapped("libnvrtc.so"):
        if "builtins" not in p and "_nvrtc" not in p:
            return p
    spec = importlib.util.find_spec("torch")
    site = os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))
    for sub in ("nvidia/cu13/lib/libnvrtc.so.13", "nvidia/cuda_nvrtc/lib/libnvrtc.so.13"):
        p = os.path.join(site, sub)
        if os.path.isfile(p):
            return os.path.realpath(p)
    return None


class Runner:
    def __init__(self, workdir, torch_nvrtc):
        self.workdir = workdir
        self.pythonpath = package_root()
        self.torch_nvrtc = torch_nvrtc

    def env(self, **extra):
        env = {k: v for k, v in os.environ.items() if not (k.startswith("FSO_") or k.startswith("TRTLLM_DG_"))}
        env["PYTHONPATH"] = self.pythonpath + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.update(extra)
        return env

    def run(self, case, env):
        out = os.path.join(self.workdir, f"{case}_out.pt")
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", case, out, "--torch-nvrtc", self.torch_nvrtc or ""]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        with open(os.path.join(self.workdir, f"{case}.log"), "w") as f:
            f.write(p.stdout + "\n--- stderr ---\n" + p.stderr)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
        check(p.returncode == 0 and line is not None,
              f"{case}: worker failed (rc {p.returncode}); last stderr:\n{p.stderr[-3000:]}")
        res = json.loads(line[len("RESULT "):])
        res["out"], res["stderr"] = out, p.stderr
        return res


def cubins(cache):
    """{kernel name: (key, cubin bytes)} of a dump cache."""
    root = os.path.join(cache, "cache")
    out = {}
    for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        m = NAME_DIR.match(d)
        check(m is not None, f"{root}: unexpected entry {d!r}")
        with open(os.path.join(root, d, KERNEL), "rb") as f:
            out[m.group(2)] = (m.group(1), f.read())
    return out


def same_outputs(a, b):
    import torch
    ra, rb = torch.load(a), torch.load(b)
    check(sorted(ra) == sorted(rb), f"output sets differ: {sorted(ra)} vs {sorted(rb)}")
    for k in ra:
        check(torch.equal(ra[k], rb[k]), f"output {k} is not bit-identical between the two compilers")
    return sorted(ra)


def run_all(workdir):
    import torch
    torch_nvrtc = torch_nvrtc_path()
    torch_ver = None
    if torch_nvrtc:
        torch_ver = nvrtc_version(ctypes.CDLL(torch_nvrtc, mode=os.RTLD_NOW | ctypes.RTLD_GLOBAL))
    differs = torch_ver is not None and torch_ver != BUNDLED
    print(f"torch {torch.__version__}: NVRTC {torch_ver} at {torch_nvrtc}")
    if not differs:
        print(f"  note: the environment's torch NVRTC has the bundled version {BUNDLED_TEXT} (or was not found), so "
              "the checks that need a second compiler version are skipped")
    R = Runner(workdir, torch_nvrtc)
    bundled = os.path.join(R.pythonpath, "fish_scales_ops", "_nvrtc")
    bundled_lib = os.path.realpath(os.path.join(bundled, "libnvrtc.so.13"))
    bundled_builtins = os.path.realpath(os.path.join(bundled, f"libnvrtc-builtins.so.{BUNDLED_TEXT}"))
    check(os.path.isfile(bundled_lib) and os.path.isfile(bundled_builtins),
          f"no bundled NVRTC in {bundled}: run `python scripts/vendor_nvrtc.py` (scripts/build.sh does it)")

    # 3. default
    c1 = os.path.join(workdir, "cache_default")
    r1 = R.run("default", R.env(FSO_JIT_DUMP_CUBIN="1", FSO_JIT_CACHE_DIR=c1, FSO_JIT_DEBUG="1"))
    if differs:
        check(tuple(r1["torch_nvrtc_before"]) == torch_ver,
              f"default: torch's NVRTC reported {r1['torch_nvrtc_before']} before the run, want {torch_ver}")
        check(tuple(r1["torch_nvrtc_after"]) == torch_ver, f"default: torch's NVRTC now reports "
                                                           f"{r1['torch_nvrtc_after']}")
    want_prefix = f"NVRTC {BUNDLED_TEXT} ("
    check(r1["compiler"].startswith(want_prefix) and r1["compiler"].endswith(")")
          and os.path.realpath(r1["compiler"][len(want_prefix):-1]) == bundled_lib,
          f"default: jit_compiler_sm90() = {r1['compiler']!r}, want NVRTC {BUNDLED_TEXT} ({bundled_lib})")
    check(bundled_lib in r1["mapped_nvrtc"], f"default: {bundled_lib} not mapped: {r1['mapped_nvrtc']}")
    check(bundled_builtins in r1["mapped_builtins"],
          f"default: {bundled_builtins} not mapped: {r1['mapped_builtins']}")
    check(NOTICE not in r1["stderr"], f"default: the version notice printed for the bundled {BUNDLED_TEXT}")
    check(r1["stderr"].count(DEBUG_LINE) == 1, f"default: the FSO_JIT_DEBUG compiler line printed "
                                               f"{r1['stderr'].count(DEBUG_LINE)} times, want once")
    for which, line in r1["describe"].items():
        check(line == f"sm_90 JIT compiler: {r1['compiler']}", f"default: {which}.describe() ends with {line!r}")
    k1 = cubins(c1)
    check(len(k1) >= 4, f"default: only {len(k1)} cubins dumped: {sorted(k1)}")
    for name, (_key, blob) in k1.items():
        notes = {m.groups() for m in TOOLKIT_NOTE.finditer(blob)}
        want_note = (str(BUNDLED[0]).encode(), str(BUNDLED[1]).encode())
        check(notes == {want_note}, f"default: {name} carries the toolkit notes {notes}, want release {BUNDLED_TEXT}")
    print(f"[3] default: {r1['compiler']}; mapped {os.path.basename(bundled_lib)} and "
          f"{os.path.basename(bundled_builtins)} from {bundled}; {len(k1)} cubins, each noted release {BUNDLED_TEXT}; "
          + (f"torch's NVRTC reports {'.'.join(map(str, torch_ver))} before and after" if differs else
             "torch's NVRTC version check skipped"))

    # 4. override with torch's library
    if torch_nvrtc:
        c2 = os.path.join(workdir, "cache_override")
        r2 = R.run("override", R.env(FSO_JIT_NVRTC_LIB=torch_nvrtc, FSO_JIT_DUMP_CUBIN="1", FSO_JIT_CACHE_DIR=c2,
                                     FSO_JIT_DEBUG="1"))
        want = f"NVRTC {torch_ver[0]}.{torch_ver[1]} ({torch_nvrtc})"
        check(os.path.realpath(r2["compiler"][r2["compiler"].index("(") + 1:-1]) == os.path.realpath(torch_nvrtc)
              and r2["compiler"].startswith(f"NVRTC {torch_ver[0]}.{torch_ver[1]} ("),
              f"override: jit_compiler_sm90() = {r2['compiler']!r}, want {want!r}")
        check(bundled_lib not in r2["mapped_nvrtc"], "override: the bundled library was mapped too")
        names = same_outputs(r1["out"], r2["out"])
        k2 = cubins(c2)
        check(sorted(k2) == sorted(k1), f"override: kernels {sorted(k2)} differ from the default case's {sorted(k1)}")
        if differs:
            n = r2["stderr"].count(NOTICE)
            check(n == 1, f"override: the version notice printed {n} times, want once")
            for name in k1:
                check(k1[name][0] != k2[name][0], f"override: {name} has the same cache key under both compilers")
                check(k1[name][1] != k2[name][1], f"override: {name} has the same cubin bytes under both compilers")
                notes = {m.groups() for m in TOOLKIT_NOTE.finditer(k2[name][1])}
                check(notes == {(str(torch_ver[0]).encode(), str(torch_ver[1]).encode())},
                      f"override: {name} carries the toolkit notes {notes}")
            print(f"[4] override: {r2['compiler']}; outputs {', '.join(names)} bit-identical to [3]; one version "
                  f"notice; the {len(k1)} cubins differ from [3] in key and bytes and are noted release "
                  f"{torch_ver[0]}.{torch_ver[1]}")
        else:
            print(f"[4] override: {r2['compiler']}; outputs bit-identical to [3] (torch's NVRTC has the bundled "
                  f"version {BUNDLED_TEXT}, so the notice and the cubin differences are not checked)")
    else:
        print("[4] SKIP: torch's NVRTC was not found, so there is no second library to override with")

    # 5. a missing library raises, with the remedy, and nothing falls back
    r3 = R.run("missing", R.env(FSO_JIT_NVRTC_LIB=MISSING))
    for what in ("jit_compiler_sm90", "gemm"):
        msg = r3["errors"].get(what)
        check(msg is not None, f"missing: {what} did not raise")
        for part in (MISSING,) + REMEDY:
            check(part in msg, f"missing: the {what} error does not name {part!r}: {msg}")
    check("compiler" not in r3, f"missing: jit_compiler_sm90() returned {r3.get('compiler')!r}")
    check(bundled_lib not in r3["mapped_nvrtc"], "missing: the bundled library was loaded instead")
    for which, line in r3["describe"].items():
        check(line.startswith("sm_90 JIT compiler: unavailable: ") and MISSING in line,
              f"missing: {which}.describe() ends with {line!r}")
    print("[5] missing library: jit_compiler_sm90() and the GEMM raise RuntimeError naming the path, "
          "scripts/vendor_nvrtc.py and FSO_JIT_NVRTC_LIB; nothing else loaded; describe() returns with the error")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2, metavar=("CASE", "OUT"))
    ap.add_argument("--torch-nvrtc", default="")
    ap.add_argument("--workdir", help="parent directory for this run's temporary directories")
    ap.add_argument("--keep", action="store_true", help="keep the temporary directories (logs, caches)")
    a = ap.parse_args()
    if a.worker:
        worker(a.worker[0], a.worker[1], a.torch_nvrtc or None)
        return 0

    import fish_scales_ops._C as ext
    try:
        static_checks(os.path.abspath(ext.__file__))
    except Failure as e:
        print(f"\nFAIL: {e}")
        return 1

    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 9:
        where = "no CUDA device" if not torch.cuda.is_available() \
            else f"device is sm_{torch.cuda.get_device_capability(0)[0]}x"
        print(f"SKIP: the run-time checks are sm_90 (H200) only ({where})")
        print("\ntest_jit_nvrtc_pin_sm90: static checks PASS")
        return 0
    if a.workdir:
        os.makedirs(a.workdir, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="test_jit_nvrtc_pin_sm90_", dir=a.workdir)
    print(f"workdir: {workdir}")
    try:
        run_all(workdir)
    except Failure as e:
        print(f"\nFAIL: {e}\n(logs in {workdir})")
        return 1
    print("\ntest_jit_nvrtc_pin_sm90: ALL PASS")
    if not a.keep:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
