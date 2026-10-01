#!/usr/bin/env python3
"""The sm_90 deep_gemm JIT cache: fish_scales_ops's own location and knobs, and a disk cache keyed by content.

Every sm_90 GEMM goes through the deep_gemm JIT (arch/sm90/fp8/jit/deep_gemm/compiler.cuh). By default each kernel
is compiled once per process with NVRTC and kept in memory; FSO_JIT_DUMP_CUBIN=1 or FSO_JIT_USE_NVCC=1 add a disk
cache under FSO_JIT_CACHE_DIR (default ${XDG_CACHE_HOME:-$HOME/.cache}/fish_scales_ops/jit), where each cubin lives
in cache/<16-hex content key>_<kernel name>/ and the key covers the generated source, the complete flag list, the
compiler and every JIT header file. Until 2026-09-30 the cache shared TensorRT-LLM's DeepGEMM location and layout
(~/.tensorrt_llm/cache/<kernel name>/), keyed by the kernel name alone, and was read on every in-memory miss even in
the default mode, so cubins built from other source were loaded for matching names.

This script drives worker subprocesses (each one a fresh process: the in-memory cache and the once-per-process
knobs start empty) that run the same workload: dense linear_fp8 at a normal shape (M = 128) and a swap-AB small-M
shape (M = 16), and fso.moe.layer at a swap-AB cell that takes the two-resident-CTA kernel (name suffix _c2) and a
non-swap cell whose FC1 is the fused SwiGLU kernel. It checks:

  1. dump mode (FSO_JIT_DUMP_CUBIN=1) writes one cubin directory per kernel, named <16 hex>_<kernel name>;
  2. a second process with the same environment loads every kernel from those directories (FSO_JIT_DEBUG log) and
     compiles none, the _c2 cubin included, with bit-identical outputs;
  3. a copy of the JIT headers passed through FSO_JIT_INCLUDE_DIRS, then the same copy with one comment line
     appended to fp8_gemm_impl.cuh, then an extra FSO_JIT_EXTRA_FLAGS macro: each change compiles every kernel
     afresh under new keys instead of loading the previous cubins, with bit-identical outputs;
  4. garbage nvrtc_kernel.cubin files planted at the legacy layout (<cache>/<kernel name>/, no key) and at the old
     default location (~/.tensorrt_llm/cache/<kernel name>/ under a temporary HOME) are never read: no error, no
     load, bit-identical outputs, the files untouched; this also runs through the deprecated TensorRT-LLM aliases
     (TRTLLM_DG_CACHE_DIR, TRTLLM_DG_JIT_DUMP_CUBIN, TRTLLM_DG_JIT_DEBUG), each of which must print its notice once;
  5. the default mode writes nothing: an FSO_JIT_CACHE_DIR stays empty, and without one the default directory is
     not created;
  6. with FSO_JIT_CACHE_DIR unset the dump lands in ${XDG_CACHE_HOME}/fish_scales_ops/jit/cache (and in
     $HOME/.cache/fish_scales_ops/jit/cache without XDG_CACHE_HOME, checked in 4).

sm_90 only (prints SKIP elsewhere). Temporary directories go under --workdir (default: a fresh tempfile.mkdtemp),
removed at the end unless --keep. usage: python tests/gemm/unit/test_jit_cache_sm90.py [--workdir DIR] [--keep]
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

NAME_DIR = re.compile(r"^([0-9a-f]{16})_(gemm_.+)$")
LOG_COMPILE = re.compile(r"Compiling JIT runtime (\S+) with options")
LOG_LOADED = re.compile(r"Loaded JIT runtime (\S+) from the disk cache: (.+?)\s*$")
LOG_WROTE = re.compile(r"Wrote JIT runtime (\S+) to the disk cache: (.+?)\s*$")
LOG_INCLUDE = re.compile(r"^\[blockscale_gemm\]\[INFO\] -I(.+?)\s*$")
KERNEL = "nvrtc_kernel.cubin"
GARBAGE = b"not a cubin: planted by test_jit_cache_sm90 and must never be loaded\n" * 64

# The kernels the workload must reach (patterns over kernel names).
REQUIRED = {
    "dense non-swap": re.compile(r"^gemm_\d.*_Normal$"),
    "dense swap-AB": re.compile(r"^gemm_swapAB_.*_Normal$"),
    "grouped swap-AB, two CTAs per SM": re.compile(r"^gemm_swapAB_.*_GroupedContiguous_c2$"),
    "fused SwiGLU FC1": re.compile(r"^gemm_2wg_swiglu_.*_GroupedContiguous$"),
}


# ----------------------------------------------------------------------------------------------------- worker
def worker(out_path):
    import torch
    import fish_scales_ops as fso

    props = torch.cuda.get_device_properties(0)
    print(f"worker device: {props.name} sm_{props.major}{props.minor} pci_bus_id=0x{props.pci_bus_id:02X}",
          flush=True)
    assert props.major == 9, "sm_90 only"
    outs = {}
    for M in (128, 16):  # M = 128: non-swap; M = 16: swap-AB (M < 32)
        g = torch.Generator(device="cuda")
        g.manual_seed(100 + M)
        x = torch.randn(M, 2048, device="cuda", dtype=torch.bfloat16, generator=g)
        w = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16, generator=g) / 45.0
        xq, sx = fso.compat.quantize_1x128_fp8(x)
        wq, sw = fso.compat.quantize_128x128_fp8(w)
        outs[f"dense_M{M}"] = fso.compat.linear_fp8(xq, wq, sx, sw)

    E, TOPK, H, I = 32, 8, 2048, 512
    g = torch.Generator(device="cuda")
    g.manual_seed(7)
    w13 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w13q, s13 = fso.compat.quantize_moe_weights_1x128_fp8_sm90(w13)
    w2q, s2 = fso.compat.quantize_moe_weights_1x128_fp8_sm90(w2)
    experts = fso.moe.prepare_experts(w13q, w2q, format="bsfp8", sw13=s13, sw2=s2)
    # M = 32: 8 routed rows per expert -> swap-AB with a 16-row activation tile, and enough tiles for two CTAs per
    # SM; M = 512: 128 rows per expert -> the non-swap path, whose FC1 is the fused SwiGLU kernel.
    for M in (32, 512):
        g.manual_seed(1000 + M)
        hidden = torch.randn(M, H, device="cuda", dtype=torch.bfloat16, generator=g)
        ids = torch.rand(M, E, device="cuda", generator=g).topk(TOPK, dim=1).indices.to(torch.int32)
        tw = torch.softmax(torch.rand(M, TOPK, device="cuda", generator=g), dim=1).float()
        outs[f"moe_M{M}"] = fso.moe.layer(hidden, experts, ids, tw)
    torch.cuda.synchronize()
    torch.save({k: v.cpu() for k, v in outs.items()}, out_path)
    print("WORKER-OK", flush=True)


# ----------------------------------------------------------------------------------------------------- driver
class Failure(AssertionError):
    pass


def check(cond, msg):
    if not cond:
        raise Failure(msg)


def package_root():
    spec = importlib.util.find_spec("fish_scales_ops")
    if spec is None or spec.origin is None:
        raise SystemExit("fish_scales_ops is not importable (set PYTHONPATH=python from the repository root)")
    return os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))


class Runner:
    def __init__(self, workdir, expect_pci_bus_id):
        self.workdir = workdir
        self.pythonpath = package_root()
        self.expect_pci_bus_id = expect_pci_bus_id
        self.summary = {}

    def base_env(self, home):
        env = {k: v for k, v in os.environ.items()
               if not (k.startswith("FSO_") or k.startswith("TRTLLM_DG_") or k == "XDG_CACHE_HOME")}
        env["HOME"] = home
        env["PYTHONPATH"] = self.pythonpath + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return env

    def run(self, phase, env):
        out = os.path.join(self.workdir, f"{phase}_out.pt")
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", out], env=env,
                           capture_output=True, text=True, timeout=1800)
        with open(os.path.join(self.workdir, f"{phase}.log"), "w") as f:
            f.write(p.stdout + "\n--- stderr ---\n" + p.stderr)
        check(p.returncode == 0 and "WORKER-OK" in p.stdout,
              f"{phase}: worker failed (rc {p.returncode}); last stderr:\n{p.stderr[-3000:]}")
        if self.expect_pci_bus_id is not None:
            check(f"pci_bus_id=0x{self.expect_pci_bus_id:02X}" in p.stdout, f"{phase}: ran on another device")
        err = p.stderr.splitlines()
        res = dict(
            out=out,
            stderr=p.stderr,
            compiled=[m.group(1) for line in err for m in [LOG_COMPILE.search(line)] if m],
            loaded=[(m.group(1), m.group(2)) for line in err for m in [LOG_LOADED.search(line)] if m],
            wrote=[(m.group(1), m.group(2)) for line in err for m in [LOG_WROTE.search(line)] if m],
            includes=[],
        )
        for line in err:  # the -I flags of the first compile, in order
            m = LOG_INCLUDE.match(line)
            if m:
                if m.group(1) in res["includes"]:
                    break
                res["includes"].append(m.group(1))
        self.summary[phase] = dict(compiled=len(res["compiled"]), loaded=len(res["loaded"]),
                                   wrote=len(res["wrote"]))
        return res

    @staticmethod
    def same_outputs(phase, a, b):
        import torch
        ra, rb = torch.load(a), torch.load(b)
        check(sorted(ra) == sorted(rb), f"{phase}: output sets differ")
        for k in ra:
            check(torch.equal(ra[k], rb[k]), f"{phase}: output {k} is not bit-identical to the reference")


def cubin_dirs(cache):
    """{kernel name: [key, ...]} of the content-keyed directories under <root>/cache, checking the layout."""
    out = {}
    if not os.path.isdir(cache):
        return out
    for d in sorted(os.listdir(cache)):
        m = NAME_DIR.match(d)
        check(m is not None, f"{cache}: unexpected entry {d!r} (want <16 hex>_<kernel name>)")
        files = os.listdir(os.path.join(cache, d))
        check(files == [KERNEL], f"{cache}/{d}: holds {files}, want [{KERNEL!r}]")
        check(os.path.getsize(os.path.join(cache, d, KERNEL)) > 1000, f"{cache}/{d}: cubin is empty")
        out.setdefault(m.group(2), []).append(m.group(1))
    return out


def plant_garbage(root, names):
    paths = []
    for n in names:
        d = os.path.join(root, n)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, KERNEL)
        with open(p, "wb") as f:
            f.write(GARBAGE)
        paths.append(p)
    return paths


def untouched(paths):
    for p in paths:
        with open(p, "rb") as f:
            if f.read() != GARBAGE:
                return False
    return True


def run_all(workdir, expect_pci_bus_id):
    R = Runner(workdir, expect_pci_bus_id)
    W = lambda *p: os.path.join(workdir, *p)  # noqa: E731
    for d in ("home", "c1"):
        os.makedirs(W(d), exist_ok=True)

    # 1. dump mode: one <key>_<name> directory per kernel
    env1 = R.base_env(W("home"))
    env1.update(FSO_JIT_CACHE_DIR=W("c1"), FSO_JIT_DUMP_CUBIN="1", FSO_JIT_DEBUG="1")
    r1 = R.run("p1_dump", env1)
    names = sorted(set(r1["compiled"]))
    check(len(names) == len(r1["compiled"]), f"p1: a kernel compiled twice: {r1['compiled']}")
    check(not r1["loaded"], f"p1: loaded from a fresh cache: {r1['loaded']}")
    dirs1 = cubin_dirs(W("c1", "cache"))
    check(sorted(dirs1) == names, f"p1: cubin directories {sorted(dirs1)} != compiled kernels {names}")
    check(all(len(v) == 1 for v in dirs1.values()), f"p1: several keys for one kernel: {dirs1}")
    check(sorted(n for n, _ in r1["wrote"]) == names, "p1: not every compiled kernel was written")
    for what, pat in REQUIRED.items():
        check(any(pat.match(n) for n in names), f"p1: the workload did not reach a {what} kernel: {names}")
    staging = os.listdir(W("c1", "tmp")) if os.path.isdir(W("c1", "tmp")) else []
    check(not staging, f"p1: leftovers in the staging directory: {staging}")
    print(f"[1] dump mode: {len(names)} kernels compiled, one <16 hex>_<name> directory each:")
    for n in names:
        print(f"      {dirs1[n][0]}_{n}")

    # 2. the same environment again: everything loads from disk, nothing compiles, outputs bit-identical
    r2 = R.run("p2_reload", env1)
    check(not r2["compiled"], f"p2: compiled {r2['compiled']} although p1 wrote them")
    check(sorted(n for n, _ in r2["loaded"]) == names, f"p2: loaded {sorted(n for n, _ in r2['loaded'])}")
    for n, path in r2["loaded"]:
        check(os.path.basename(path) == f"{dirs1[n][0]}_{n}", f"p2: {n} loaded from {path}")
    check(cubin_dirs(W("c1", "cache")) == dirs1, "p2: the cache changed")
    R.same_outputs("p2", r1["out"], r2["out"])
    c2 = [n for n in names if n.endswith("_c2")]
    print(f"[2] reload: {len(r2['loaded'])} kernels loaded from disk ({', '.join(c2)} included), "
          "0 compiled, outputs bit-identical")

    # 3. a header edit and a flag change each get new keys
    includes = r1["includes"]
    jit_dirs = [d for d in includes if os.path.isfile(os.path.join(d, "deep_gemm", "fp8_gemm_impl.cuh"))]
    check(jit_dirs, f"p1: no include dir holds deep_gemm/fp8_gemm_impl.cuh: {includes}")
    copy_root = W("jit_copy")
    shutil.copytree(os.path.join(jit_dirs[0], "deep_gemm"), os.path.join(copy_root, "deep_gemm"))
    env3 = dict(env1)
    env3["FSO_JIT_INCLUDE_DIRS"] = ":".join(copy_root if d == jit_dirs[0] else d for d in includes)
    r3a = R.run("p3a_header_copy", env3)  # new -I path: new keys (the flags changed)
    check(sorted(r3a["compiled"]) == names and not r3a["loaded"], "p3a: the header copy reused cubins")
    R.same_outputs("p3a", r1["out"], r3a["out"])
    dirs3a = cubin_dirs(W("c1", "cache"))
    with open(os.path.join(copy_root, "deep_gemm", "fp8_gemm_impl.cuh"), "a") as f:
        f.write("\n// test_jit_cache_sm90: one appended comment line\n")
    r3b = R.run("p3b_header_edit", env3)  # same flags, one header byte sequence changed
    check(not r3b["loaded"], f"p3b: reused {r3b['loaded']} after a header edit")
    check(sorted(r3b["compiled"]) == names, f"p3b: compiled {sorted(r3b['compiled'])}")
    R.same_outputs("p3b", r1["out"], r3b["out"])
    dirs3b = cubin_dirs(W("c1", "cache"))
    for n in names:
        new = set(dirs3b[n]) - set(dirs3a[n])
        check(len(dirs3a[n]) == 2 and len(new) == 1, f"p3: keys of {n}: after the copy {dirs3a[n]}, "
                                                     f"after the edit {dirs3b[n]}")
    env3c = dict(env1)
    env3c["FSO_JIT_EXTRA_FLAGS"] = "-DFSO_TEST_JIT_CACHE_UNUSED_MACRO=1"
    r3c = R.run("p3c_extra_flag", env3c)  # the baked headers, one extra flag
    check(not r3c["loaded"] and sorted(r3c["compiled"]) == names, "p3c: an extra flag reused cubins")
    R.same_outputs("p3c", r1["out"], r3c["out"])
    dirs3c = cubin_dirs(W("c1", "cache"))
    check(all(len(dirs3c[n]) == 4 for n in names), f"p3c: {dirs3c}")
    print("[3] header copy, header edit (one comment line) and one extra flag: each compiled all "
          f"{len(names)} kernels under new keys (4 directories per kernel now), outputs bit-identical")

    # 4. garbage at the legacy layouts is never read; the TensorRT-LLM aliases still work, with a notice each
    h4 = W("home4")
    old_default = plant_garbage(os.path.join(h4, ".tensorrt_llm", "cache"), names)
    new_root = os.path.join(h4, ".cache", "fish_scales_ops", "jit")
    legacy_in_new_root = plant_garbage(os.path.join(new_root, "cache"), names)
    env4 = R.base_env(h4)
    env4.update(FSO_JIT_DUMP_CUBIN="1", FSO_JIT_DEBUG="1")  # FSO_JIT_CACHE_DIR and XDG_CACHE_HOME unset
    r4a = R.run("p4a_garbage_default_root", env4)
    check(not r4a["loaded"] and sorted(r4a["compiled"]) == names, "p4a: loaded from a legacy directory")
    check(untouched(old_default + legacy_in_new_root), "p4a: a planted file changed")
    R.same_outputs("p4a", r1["out"], r4a["out"])
    keyed = {n: v for n, v in cubin_dirs_mixed(os.path.join(new_root, "cache")).items()}
    check(sorted(keyed) == names and all(len(v) == 1 for v in keyed.values()),
          f"p4a: content-keyed directories under $HOME/.cache/fish_scales_ops/jit/cache: {keyed}")
    c4b = W("c4b")
    legacy_alias = plant_garbage(os.path.join(c4b, "cache"), names)
    env4b = R.base_env(h4)
    env4b.update(TRTLLM_DG_CACHE_DIR=c4b, TRTLLM_DG_JIT_DUMP_CUBIN="1", TRTLLM_DG_JIT_DEBUG="1")
    r4b = R.run("p4b_garbage_legacy_aliases", env4b)
    for alias, name in (("TRTLLM_DG_CACHE_DIR", "FSO_JIT_CACHE_DIR"), ("TRTLLM_DG_JIT_DUMP_CUBIN", "FSO_JIT_DUMP_CUBIN"),
                        ("TRTLLM_DG_JIT_DEBUG", "FSO_JIT_DEBUG")):
        notice = f"{alias} is deprecated and will be removed in fish-scales-ops 0.3.0; set {name} instead"
        check(r4b["stderr"].count(notice) == 1, f"p4b: the {alias} notice printed "
                                                f"{r4b['stderr'].count(notice)} times")
    check(not r4b["loaded"] and sorted(r4b["compiled"]) == names, "p4b: loaded from a legacy directory")
    check(untouched(old_default + legacy_alias), "p4b: a planted file changed")
    R.same_outputs("p4b", r1["out"], r4b["out"])
    keyed = cubin_dirs_mixed(os.path.join(c4b, "cache"))
    check(sorted(keyed) == names and all(len(v) == 1 for v in keyed.values()), f"p4b: {keyed}")
    print(f"[4] {3 * len(names)} planted garbage cubins (legacy layout in the new root and in the "
          "TRTLLM_DG_CACHE_DIR root, old ~/.tensorrt_llm default) never read; new keyed directories written; "
          "each TensorRT-LLM alias noticed once; outputs bit-identical")

    # 5. the default mode writes nothing
    c5, h5, x5 = W("c5"), W("home5"), W("xdg5")
    for d in (c5, h5, x5):
        os.makedirs(d, exist_ok=True)
    old_default5 = plant_garbage(os.path.join(h5, ".tensorrt_llm", "cache"), names)
    env5a = R.base_env(h5)
    env5a.update(FSO_JIT_CACHE_DIR=c5, FSO_JIT_DEBUG="1")
    r5a = R.run("p5a_default_mode_cache_dir", env5a)
    check(sorted(r5a["compiled"]) == names and not r5a["loaded"] and not r5a["wrote"], "p5a: disk activity")
    check(os.listdir(c5) == [], f"p5a: FSO_JIT_CACHE_DIR is not empty: {os.listdir(c5)}")
    R.same_outputs("p5a", r1["out"], r5a["out"])
    env5b = R.base_env(h5)
    env5b.update(XDG_CACHE_HOME=x5, FSO_JIT_DEBUG="1")
    r5b = R.run("p5b_default_mode_default_dir", env5b)
    check(sorted(r5b["compiled"]) == names and not r5b["loaded"] and not r5b["wrote"], "p5b: disk activity")
    for d in (os.path.join(x5, "fish_scales_ops"), os.path.join(h5, ".cache", "fish_scales_ops")):
        check(not os.path.exists(d), f"p5b: the default mode created {d}")
    check(untouched(old_default5), "p5: a planted file changed")
    R.same_outputs("p5b", r1["out"], r5b["out"])
    print("[5] default mode: every kernel compiled in memory; FSO_JIT_CACHE_DIR stayed empty, "
          "no fish_scales_ops directory under XDG_CACHE_HOME or $HOME/.cache, the planted old-default cubins unread")

    # 6. the XDG default root in dump mode
    x6 = W("xdg6")
    os.makedirs(x6, exist_ok=True)
    env6 = R.base_env(W("home"))
    env6.update(XDG_CACHE_HOME=x6, FSO_JIT_DUMP_CUBIN="1", FSO_JIT_DEBUG="1")
    r6 = R.run("p6_xdg_default_root", env6)
    keyed = cubin_dirs(os.path.join(x6, "fish_scales_ops", "jit", "cache"))
    check(sorted(keyed) == names, f"p6: {keyed}")
    check({n: v[0] for n, v in keyed.items()} == {n: v[0] for n, v in dirs1.items()},
          "p6: the same build under another cache root got other keys")
    R.same_outputs("p6", r1["out"], r6["out"])
    print("[6] FSO_JIT_CACHE_DIR unset: the dump landed in ${XDG_CACHE_HOME}/fish_scales_ops/jit/cache under the "
          "same keys as in [1] (the key does not depend on the cache location)")

    with open(W("summary.json"), "w") as f:
        json.dump(dict(kernels={n: dirs1[n][0] for n in names}, phases=R.summary), f, indent=1)


def cubin_dirs_mixed(cache):
    """Like cubin_dirs, but skips the planted legacy directories (no key) living next to the keyed ones."""
    out = {}
    for d in sorted(os.listdir(cache)):
        m = NAME_DIR.match(d)
        if m is None:
            check(os.listdir(os.path.join(cache, d)) == [KERNEL], f"{cache}/{d}: unexpected contents")
            continue
        check(os.listdir(os.path.join(cache, d)) == [KERNEL], f"{cache}/{d}: unexpected contents")
        out.setdefault(m.group(2), []).append(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", metavar="OUT")
    ap.add_argument("--workdir", help="parent directory for this run's temporary directories")
    ap.add_argument("--keep", action="store_true", help="keep the temporary directories (logs, caches)")
    ap.add_argument("--expect-pci-bus-id", type=lambda s: int(s, 0), default=None,
                    help="fail unless the worker runs on this PCI bus id (e.g. 0xCB)")
    a = ap.parse_args()
    if a.worker:
        worker(a.worker)
        return 0

    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 9:
        where = "no CUDA device" if not torch.cuda.is_available() \
            else f"device is sm_{torch.cuda.get_device_capability(0)[0]}x"
        print(f"SKIP: test_jit_cache_sm90 is sm_90 (H200) only ({where})")
        return 0
    if a.workdir:
        os.makedirs(a.workdir, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="test_jit_cache_sm90_", dir=a.workdir)
    print(f"workdir: {workdir}")
    try:
        run_all(workdir, a.expect_pci_bus_id)
    except Failure as e:
        print(f"\nFAIL: {e}\n(logs in {workdir})")
        return 1
    print("\ntest_jit_cache_sm90: ALL PASS")
    if not a.keep:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
