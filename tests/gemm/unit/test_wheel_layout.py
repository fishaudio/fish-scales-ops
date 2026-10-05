#!/usr/bin/env python3
"""The layout of an installed fish-scales-ops wheel: the files it carries so that it runs without the source tree.

A wheel built by scripts/build_wheel.sh carries next to the extension: BUILD_INFO.json, the bundled NVRTC in _nvrtc/,
every header of the sm_90 JIT in _jit_include/ and the sm_100/sm_103 CuTe-DSL kernel in _dsl/. For the imported
package this script checks:

  an installed wheel (the package holds a BUILD_INFO.json), every architecture:
    1. the packaged files exist next to the extension; fish_scales_ops.build_info() returns every key of
       BUILD_INFO.json, with the package's version and the running torch; dense.describe() and moe.describe() name the
       build on their second line;
    2. the import guard: a child process imports a copy of the package whose BUILD_INFO.json names another torch and
       must get an ImportError that names both versions, before the extension is loaded;
  an installed wheel, sm_90:
    3. a child process with FSO_JIT_INCLUDE_DIRS unset and FSO_JIT_DEBUG=1 compiles an FP8 GEMM. Its JIT debug log
       must name the packaged tree as the include source, and every -I directory of the compile must lie inside the
       package; the output must be finite and the compiler the bundled NVRTC;
  an installed wheel, sm_100 / sm_103:
    4. _sm100_dsl resolves its kernel to the packaged copy, and a child process with FSO_DSL_KERNEL_PATH unset loads
       the CuTe-DSL tier from it without a notice (skipped with a note when nvidia-cutlass-dsl is not installed);
  an in-place build (no BUILD_INFO.json):
    build_info() reports source_build with the package version and the running torch, and the describe() lines say
    so; the wheel checks are skipped with a note.

usage: python tests/gemm/unit/test_wheel_layout.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile

import torch
import fish_scales_ops as fso

KEYS = ("version", "commit", "dirty", "built_at", "image", "cuda_toolkit", "nvcc", "gcc", "python", "torch",
        "cutlass_commit", "nvrtc", "arch_list", "glibc_required")
PACKAGED = ("BUILD_INFO.json", "_nvrtc/libnvrtc.so.13", "_nvrtc/libnvrtc-builtins.so.13.2", "_nvrtc/VERSION",
            "_jit_include/THIRD_PARTY_NOTICES", "_jit_include/deep_gemm/fp8_gemm_impl.cuh",
            "_jit_include/cuda_fp8.h", "_jit_include/cuda_bf16.h", "_jit_include/cuda_fp16.h",
            "_jit_include/cutlass/cutlass.h", "_dsl/dense_blockscaled_gemm_persistent.py")
INCLUDE_ORIGIN = "the include tree packaged with fish_scales_ops"
LOG_INCLUDE_DIRS = re.compile(r"sm_90 JIT include directories \(([^)]*)\): (.*?)\s*$", re.M)
LOG_INCLUDE = re.compile(r"^\[blockscale_gemm\]\[INFO\] -I(.+?)\s*$", re.M)
DSL_NOTICE = "fso: sm_100/103 CuTe-DSL mid-band tier unavailable"
PKG = os.path.dirname(os.path.abspath(fso.__file__))


class Failure(AssertionError):
    pass


def check(cond, msg):
    if not cond:
        raise Failure(msg)


def clean_env(**extra):
    """The caller's environment without any FSO_* or TRTLLM_DG_* variable, with the directory that holds this package
    first on PYTHONPATH (so a child imports the same copy), plus ``extra``."""
    env = {k: v for k, v in os.environ.items() if not (k.startswith("FSO_") or k.startswith("TRTLLM_DG_"))}
    env["PYTHONPATH"] = os.path.dirname(PKG) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(extra)
    return env


def child(code, env, timeout=1800):
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def inside(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root + os.sep)


def describe_lines():
    return {"dense": fso.dense.describe().splitlines()[1], "moe": fso.moe.describe().splitlines()[1]}


# ------------------------------------------------------------------------------------------------- in-place build
def source_build():
    info = fso.build_info()
    want = {"source_build": True, "version": fso.__version__, "torch": str(torch.__version__)}
    check(info == want, f"build_info() of an in-place build is {info}, want {want}")
    line = f"build: fish-scales-ops {fso.__version__}, an in-place build of the source tree, running torch " \
           f"{torch.__version__}"
    for which, got in describe_lines().items():
        check(got == line, f"{which}.describe() line 2 is {got!r}, want {line!r}")
    print(f"[source] build_info() = {info}; both describe() texts say so on their second line")
    print("SKIP: no BUILD_INFO.json next to the package (an in-place build of the source tree), so the wheel checks "
          "(packaged files, import guard, packaged JIT include tree, packaged CuTe-DSL kernel) do not apply; they run "
          "on a wheel built by scripts/build_wheel.sh")


# ------------------------------------------------------------------------------------------------------ the wheel
def files_and_info():
    missing = [p for p in PACKAGED if not os.path.isfile(os.path.join(PKG, p))]
    check(not missing, f"the package {PKG} lacks {missing}")
    ext_dir = os.path.dirname(os.path.abspath(fso._C.__file__))
    check(os.path.realpath(ext_dir) == os.path.realpath(PKG), f"the extension lives in {ext_dir}, not in {PKG}")
    info = fso.build_info()
    with open(os.path.join(PKG, "BUILD_INFO.json")) as f:
        on_disk = json.load(f)
    check(info == on_disk, f"build_info() {info} differs from BUILD_INFO.json {on_disk}")
    info["nvrtc"]["probe"] = info.pop("commit")
    check(fso.build_info() == on_disk, "changing the dict build_info() returned changed the next call's result")
    info = fso.build_info()
    absent = [k for k in KEYS if info.get(k) in (None, "")]
    check(not absent, f"build_info() lacks {absent}")
    check("source_build" not in info, "build_info() of a wheel says source_build")
    check(info["version"] == fso.__version__, f"BUILD_INFO.json version {info['version']} != {fso.__version__}")
    check(info["torch"] == str(torch.__version__), f"BUILD_INFO.json torch {info['torch']} != {torch.__version__}")
    for key in ("version", "sha256"):
        check(info["nvrtc"].get(key), f"build_info()['nvrtc'] has no {key}: {info['nvrtc']}")
    with open(os.path.join(PKG, "_nvrtc", "VERSION")) as f:
        check(f"sha256 {info['nvrtc']['sha256']}" in f.read(), "BUILD_INFO.json names another NVRTC wheel than "
                                                              "_nvrtc/VERSION")
    commit = str(info["commit"])[:12] + (" (dirty)" if info["dirty"] else "")
    line = f"build: fish-scales-ops {info['version']}, commit {commit}, built against torch {info['torch']}"
    for which, got in describe_lines().items():
        check(got == line, f"{which}.describe() line 2 is {got!r}, want {line!r}")
    print(f"[1] {len(PACKAGED)} packaged files present next to the extension in {PKG}; build_info() names all "
          f"{len(KEYS)} keys: version {info['version']}, commit {info['commit']}{' (dirty)' if info['dirty'] else ''}, "
          f"torch {info['torch']}, arch {info['arch_list']}, {info['glibc_required']}; describe() line 2: {line}")
    return info


def import_guard(info):
    fake = "0.0.0+fso.wheel.layout.test"
    with tempfile.TemporaryDirectory(prefix="test_wheel_layout_guard_") as tmp:
        copy = os.path.join(tmp, "fish_scales_ops")
        os.mkdir(copy)
        for name in os.listdir(PKG):
            if name not in ("BUILD_INFO.json", "__pycache__"):
                os.symlink(os.path.join(PKG, name), os.path.join(copy, name))
        with open(os.path.join(copy, "BUILD_INFO.json"), "w") as f:
            json.dump(dict(info, torch=fake), f)
        code = ("import sys\n"
                "try:\n"
                "    import fish_scales_ops\n"
                "except ImportError as e:\n"
                "    print('IMPORT-ERROR ' + ' '.join(str(e).split()))\n"
                "    print('EXTENSION-LOADED', 'fish_scales_ops._C' in sys.modules)\n"
                "    print('FROM', getattr(sys.modules.get('fish_scales_ops._build_info'), '__file__', None))\n"
                "    sys.exit(0)\n"
                "print('NO-ERROR')\n"
                "sys.exit(3)\n")
        rc, out, err = child(code, clean_env(PYTHONPATH=tmp, CUDA_VISIBLE_DEVICES=""), timeout=600)
    msg = next((ln[len("IMPORT-ERROR "):] for ln in out.splitlines() if ln.startswith("IMPORT-ERROR ")), None)
    check(rc == 0 and msg is not None, f"importing a package whose BUILD_INFO.json names torch {fake} did not raise "
                                       f"ImportError (exit {rc}):\n{out}\n{err[-2000:]}")
    check(fake in msg and str(torch.__version__) in msg, f"the ImportError does not name both versions: {msg}")
    check("EXTENSION-LOADED False" in out, f"the extension was loaded before the guard raised:\n{out}")
    check(any(ln.startswith("FROM " + tmp) for ln in out.splitlines()), f"the child imported another copy:\n{out}")
    print(f"[2] import guard: BUILD_INFO.json naming torch {fake} -> ImportError before the extension loads: {msg}")


def packaged_jit_includes():
    code = ("import torch, fish_scales_ops as fso\n"
            "g = torch.Generator(device='cuda').manual_seed(3)\n"
            "w = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16, generator=g) / 32\n"
            "x = torch.randn(128, 1024, device='cuda', dtype=torch.bfloat16, generator=g)\n"
            "y = fso.dense.linear(x, fso.dense.prepare_weight(w, format='bsfp8'))\n"
            "torch.cuda.synchronize()\n"
            "print('FINITE', bool(torch.isfinite(y.float()).all()))\n"
            "print('COMPILER', torch.ops.fish_scales_ops.jit_compiler_sm90())\n")
    rc, out, err = child(code, clean_env(FSO_JIT_DEBUG="1"))
    check(rc == 0, f"the sm_90 GEMM child failed (exit {rc}):\n{err[-3000:]}")
    found = LOG_INCLUDE_DIRS.findall(err)
    check(len(found) == 1, f"the JIT debug log names the include directories {len(found)} times, want once:\n"
                           f"{err[-3000:]}")
    origin, listed = found[0]
    want = os.path.join(PKG, "_jit_include")
    check(origin == INCLUDE_ORIGIN and listed.split(":") == [want],
          f"the JIT took its headers from {origin}: {listed}; want {INCLUDE_ORIGIN}: {want}")
    flags = LOG_INCLUDE.findall(err)
    check(flags, f"no -I flag in the JIT debug log:\n{err[-3000:]}")
    outside = sorted({d for d in flags if not inside(d, PKG)})
    check(not outside, f"the JIT compiled with include directories outside the package {PKG}: {outside}")
    check("FINITE True" in out, f"the GEMM output is not finite:\n{out}")
    compiler = next((ln[len("COMPILER "):] for ln in out.splitlines() if ln.startswith("COMPILER ")), "")
    bundled = os.path.join(PKG, "_nvrtc", "libnvrtc.so.13")
    check(compiler.startswith("NVRTC 13.2 (") and os.path.realpath(compiler[len("NVRTC 13.2 ("):-1])
          == os.path.realpath(bundled), f"jit_compiler_sm90() = {compiler!r}, want NVRTC 13.2 ({bundled})")
    print(f"[3] sm_90 JIT: FSO_JIT_INCLUDE_DIRS unset -> {origin}: {listed}; {len(flags)} -I flags, all inside the "
          f"package; output finite; {compiler}")


def packaged_dsl_kernel():
    from fish_scales_ops.gemm import _sm100_dsl

    want = os.path.join(PKG, "_dsl", "dense_blockscaled_gemm_persistent.py")
    path = _sm100_dsl._default_kernel_path()
    check(os.path.realpath(path) == os.path.realpath(want), f"_sm100_dsl resolves its kernel to {path}, want {want}")
    try:
        import cutlass  # noqa: F401
    except ImportError:
        print(f"[4] sm_100: _sm100_dsl resolves its kernel to the packaged {want}; the tier itself is not loaded "
              "(nvidia-cutlass-dsl, the sm100 extra, is not installed)")
        return
    code = ("from fish_scales_ops.gemm import _sm100_dsl\n"
            "st = _sm100_dsl._init()\n"
            "print('LIVE', st is not None)\n"
            "print('KERNEL', (st or {}).get('kernel_path'))\n")
    rc, out, err = child(code, clean_env())
    check(rc == 0, f"the CuTe-DSL child failed (exit {rc}):\n{err[-3000:]}")
    check("LIVE True" in out and DSL_NOTICE not in err, f"the CuTe-DSL tier did not load:\n{out}\n{err[-3000:]}")
    kernel = next((ln[len("KERNEL "):] for ln in out.splitlines() if ln.startswith("KERNEL ")), "")
    check(os.path.realpath(kernel) == os.path.realpath(want), f"the tier loaded {kernel}, want {want}")
    print(f"[4] sm_100: the CuTe-DSL tier is live, loaded from the packaged {kernel}")


def main():
    print(f"package: {PKG}; torch {torch.__version__}")
    try:
        if not os.path.isfile(os.path.join(PKG, "BUILD_INFO.json")):
            source_build()
            print("\ntest_wheel_layout: source build PASS (wheel checks skipped)")
            return 0
        info = files_and_info()
        import_guard(info)
        major = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
        device = f"sm_{major}x" if major else "no CUDA device"
        if major == 9:
            packaged_jit_includes()
        else:
            print(f"[3] SKIP: the packaged JIT include check runs on sm_90 ({device} here)")
        if major == 10:
            packaged_dsl_kernel()
        else:
            print(f"[4] SKIP: the packaged CuTe-DSL kernel check runs on sm_100 / sm_103 ({device} here)")
    except Failure as e:
        print(f"\nFAIL: {e}")
        return 1
    print("\ntest_wheel_layout: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
