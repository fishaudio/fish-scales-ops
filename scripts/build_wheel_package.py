#!/usr/bin/env python3
"""The package-data and checking steps of a wheel build. scripts/build_wheel.sh runs them inside its container
(scripts/build_wheel_inner.sh); they also run on a host for a hand-staged package or to re-check a wheel.

  stage     Writes into python/fish_scales_ops/, before the extension is built:
              _jit_include/    every header the sm_90 JIT compiles with, as one include root: the deep_gemm JIT
                               headers, CUTLASS include/, and the CUDA toolkit headers with the CCCL headers merged
                               into the root and the CUDA library headers left out; plus THIRD_PARTY_NOTICES
              _dsl/            the CUTLASS CuTe-DSL kernel of the sm_100/sm_103 mid-band tier, and its notice
              BUILD_INFO.json  the build information; glibc_required stays null until finalize
            The bundled NVRTC (_nvrtc/, scripts/vendor_nvrtc.py) must already be in place: BUILD_INFO.json records it.
  finalize  WHEEL: reads the newest GLIBC_x.y symbol version that the wheel's extension needs and writes it as
            glibc_required into the wheel's BUILD_INFO.json (python -m wheel unpack and pack, which regenerates
            RECORD) and into the source package's copy.
  check     WHEEL: the post-build checks, one line each; the exit status is 1 when any fails:
              the wheel holds the extension, _nvrtc/libnvrtc.so.13, _nvrtc/libnvrtc-builtins.so.13.0, _jit_include/,
              _dsl/dense_blockscaled_gemm_persistent.py and BUILD_INFO.json with every key;
              the extension needs at most GLIBC_2.35 and GLIBCXX_3.4.30, and glibc_required names its newest GLIBC;
              it has no libnvrtc among its NEEDED libraries and no undefined nvrtc* symbol;
              it carries a cubin for every architecture of the arch list (sm_90a, the sm_100 family target sm_100f
              and sm_120a for 9.0a;10.0f;12.0a).

Standard library only, except that stage imports torch to record its version. check needs readelf and nm (binutils)
and cuobjdump (CUDA_HOME/bin, or PATH).

usage: build_wheel_package.py stage --commit SHA --dirty 0|1 --image NAME --image-id ID --base-image REF
                                    --cutlass-commit SHA [--cutlass DIR] [--cuda-home DIR] [--arch-list LIST]
                                    [--package DIR]
       build_wheel_package.py finalize WHEEL
       build_wheel_package.py check WHEEL [--arch-list LIST] [--max-glibc 2.35] [--max-glibcxx 3.4.30]
"""
from __future__ import annotations

import argparse
import datetime
import fnmatch
import filecmp
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "python" / "fish_scales_ops"
DEEP_GEMM = REPO / "csrc" / "gemm" / "include" / "blockscale_gemm" / "arch" / "sm90" / "fp8" / "jit" / "deep_gemm"
DSL_KERNEL = Path("examples") / "python" / "CuTeDSL" / "blackwell" / "dense_blockscaled_gemm_persistent.py"
DEFAULT_ARCH_LIST = "9.0a;10.0f;12.0a"

# The keys of BUILD_INFO.json, in the order they are written.
BUILD_INFO_KEYS = ("version", "commit", "dirty", "built_at", "image", "cuda_toolkit", "nvcc", "gcc", "python", "torch",
                   "cutlass_commit", "nvrtc", "arch_list", "glibc_required")

# Top-level entries of the CUDA toolkit's include/ directory that the packaged JIT tree leaves out: the headers of the
# CUDA libraries and tools, which the JIT kernels never include. Everything else is copied whole.
CUDA_LIBRARY_HEADERS = (
    "cublas*", "nvblas.h",                          # cuBLAS
    "cufft*", "cudalibxt.h",                        # cuFFT
    "curand*",                                      # cuRAND
    "cusolver*",                                    # cuSOLVER
    "cusparse*",                                    # cuSPARSE
    "npp*",                                         # NPP
    "nvjpeg*",                                      # nvJPEG
    "nvml.h",                                       # NVML
    "cupti*", "nvperf*", "generated_*_meta.h",      # CUPTI and its callback metadata
    "cufile*", "cuobj*",                            # GPUDirect Storage
    "nvtx3",                                        # NVTX
    "nvrtc.h", "nvJitLink.h", "nvPTXCompiler.h", "nvFatbin.h",  # the host-side compiler library APIs
    "nvsandboxutils*",                              # sandbox utilities
    "nv_decode.h",                                  # the host-side C++ name demangler
    "Openacc", "Openmp",                            # OpenACC / OpenMP offload
)

# Files a complete packaged JIT tree must hold (relative to _jit_include/): the kernel's own headers, the CUDA headers
# the kernels include today, and one file of every other source.
JIT_TREE_SENTINELS = ("THIRD_PARTY_NOTICES", "deep_gemm/fp8_gemm_impl.cuh", "deep_gemm/nvrtc_cutlass.cuh",
                      "deep_gemm/nvrtc_std.cuh", "cuda_fp8.h", "cuda_bf16.h", "cuda_fp16.h", "crt/host_defines.h",
                      "nv/target", "cuda/std/cstdint", "cutlass/cutlass.h", "cute/tensor.hpp")


class Failure(RuntimeError):
    pass


def say(msg: str) -> None:
    print(f"[build_wheel_package] {msg}", flush=True)


def run(cmd: list[str], **kw) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


# ------------------------------------------------------------------------------------------------------------- stage
def merge_tree(src: Path, dst: Path, origin: dict, skip_top=()) -> int:
    """Copy every file under ``src`` to the same relative path under ``dst``, following symbolic links (a wheel holds
    no links). Top-level entries matching a ``skip_top`` pattern are left out. A path already staged from an earlier
    source is kept when its bytes are identical and is an error otherwise, so merging several include roots into one
    can never shadow a header silently."""
    copied = 0
    for root, dirs, files in os.walk(src, followlinks=True):
        rel_root = Path(root).relative_to(src)
        if rel_root == Path("."):
            dirs[:] = [d for d in dirs if not any(fnmatch.fnmatchcase(d, p) for p in skip_top)]
            files = [f for f in files if not any(fnmatch.fnmatchcase(f, p) for p in skip_top)]
        dirs.sort()
        for name in sorted(files):
            rel = rel_root / name
            s, d = Path(root) / name, dst / rel
            if d.exists():
                if filecmp.cmp(s, d, shallow=False):
                    continue
                raise Failure(f"{rel} is in both {origin[rel]} and {src} with different contents")
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(s, d)
            os.chmod(d, 0o644)
            origin[rel] = src
            copied += 1
    return copied


def first_line(cmd: list[str], pattern: str | None = None) -> str:
    out = run(cmd).splitlines()
    for line in out:
        if line.strip() and (pattern is None or re.search(pattern, line)):
            return line.strip()
    raise Failure(f"no {'line matching ' + pattern if pattern else 'output'} from {' '.join(cmd)}")


def cuda_toolkit_version(cuda_home: Path) -> str:
    """The toolkit version: CUDA_VERSION, which the nvidia/cuda images set; else version.json of the toolkit."""
    if os.environ.get("CUDA_VERSION"):
        return os.environ["CUDA_VERSION"]
    try:
        return json.loads((cuda_home / "version.json").read_text())["cuda"]["version"]
    except (OSError, ValueError, KeyError) as e:
        raise Failure(f"cannot tell the CUDA toolkit version of {cuda_home}: CUDA_VERSION is unset and "
                      f"version.json is unreadable ({e})") from e


def package_version() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', (REPO / "python" / "pyproject.toml").read_text(), re.M)
    if not m:
        raise Failure("python/pyproject.toml has no version line")
    return m.group(1)


def bundled_nvrtc(pkg: Path) -> dict:
    """The NVRTC that scripts/vendor_nvrtc.py unpacked into _nvrtc/, from its VERSION file."""
    try:
        fields = dict(line.split(" ", 1) for line in (pkg / "_nvrtc" / "VERSION").read_text().splitlines() if line)
    except (OSError, ValueError) as e:
        raise Failure(f"no usable {pkg / '_nvrtc' / 'VERSION'} ({e}); run scripts/vendor_nvrtc.py first") from e
    return {"version": fields["nvidia-cuda-nvrtc"], "wheel": fields["wheel"], "sha256": fields["sha256"]}


def notices(cutlass_commit: str, toolkit: str, base_image: str, commit: str, cutlass_license: str) -> str:
    return f"""Third-party files in fish_scales_ops/_jit_include
==================================================

The sm_90 GEMM kernels of fish-scales-ops are compiled at run time by NVRTC. This directory holds every header they
compile with, as one include root. scripts/build_wheel_package.py assembled it when the wheel was built, from the
sources below; when two sources hold the same path the files are identical.

deep_gemm/
    Source: fish-scales-ops commit {commit},
    csrc/gemm/include/blockscale_gemm/arch/sm90/fp8/jit/deep_gemm/.
    These JIT headers come from NVIDIA TensorRT-LLM, which adapted them from DeepSeek's DeepGEMM (MIT License);
    each file keeps its NVIDIA copyright header and its license, the Apache License 2.0.

cute/, cutlass/
    Source: NVIDIA CUTLASS commit {cutlass_commit} (v4.4.2), include/.
    License: BSD-3-Clause. Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. The full text is the
    CUTLASS LICENSE.txt, reproduced at the end of this file.

cub/, cuda/, nv/, thrust/
    Source: CCCL (the CUDA C++ Core Compute Libraries) as shipped in the NVIDIA CUDA Toolkit {toolkit}, the
    include/cccl/ directory of the build image, merged into this root.
    License: Apache License 2.0 with LLVM Exceptions (https://llvm.org/LICENSE.txt), the license of
    https://github.com/NVIDIA/cccl. Copyright NVIDIA CORPORATION & AFFILIATES and the LLVM project contributors.

Every other file (cuda_fp8.h, cuda_bf16.h, cuda_fp16.h, crt/, cooperative_groups/, ...)
    Source: the NVIDIA CUDA Toolkit {toolkit} headers, the include/ directory of the build image
    ({base_image}), without the headers of the CUDA libraries and tools (cuBLAS, cuFFT, cuRAND, cuSOLVER,
    cuSPARSE, NPP, nvJPEG, NVML, CUPTI, cuFile, NVTX, the compiler library APIs).
    License: the NVIDIA CUDA Toolkit End User License Agreement, https://docs.nvidia.com/cuda/eula/index.html.
    Copyright NVIDIA CORPORATION & AFFILIATES.

CUTLASS LICENSE.txt
-------------------
{cutlass_license}"""


def stage(args) -> int:
    cutlass = Path(args.cutlass or os.environ.get("CUTLASS_DIR") or REPO / "3rdparty" / "cutlass").resolve()
    cuda_home = Path(args.cuda_home or os.environ.get("CUDA_HOME") or "/usr/local/cuda").resolve()
    arch_list = args.arch_list or os.environ.get("TORCH_CUDA_ARCH_LIST") or DEFAULT_ARCH_LIST
    for path in (DEEP_GEMM / "fp8_gemm_impl.cuh", cutlass / "include" / "cutlass" / "cutlass.h",
                 cutlass / "LICENSE.txt", cutlass / DSL_KERNEL, cuda_home / "include" / "cuda_fp8.h",
                 cuda_home / "include" / "cccl", cuda_home / "bin" / "nvcc"):
        if not path.exists():
            raise Failure(f"{path} does not exist")
    pkg = Path(args.package).resolve() if args.package else PKG
    toolkit = cuda_toolkit_version(cuda_home)
    nvrtc = bundled_nvrtc(pkg)

    # _jit_include/: one include root. The JIT passes it as its only -I directory, so the CCCL headers, which the
    # toolkit keeps in include/cccl/ behind a second -I, are merged into the root; merge_tree refuses a path that two
    # sources hold with different contents (the toolkit's include/nv/ duplicates include/cccl/nv/, byte for byte).
    jit = pkg / "_jit_include"
    stage_dir = pkg / "_jit_include.staging"
    shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir()
    origin: dict = {}
    counts = {
        "deep_gemm": merge_tree(DEEP_GEMM, stage_dir / "deep_gemm", origin),
        "cutlass": merge_tree(cutlass / "include", stage_dir, origin),
        "cuda": merge_tree(cuda_home / "include", stage_dir, origin, skip_top=CUDA_LIBRARY_HEADERS + ("cccl",)),
        "cccl": merge_tree(cuda_home / "include" / "cccl", stage_dir, origin),
    }
    (stage_dir / "THIRD_PARTY_NOTICES").write_text(
        notices(args.cutlass_commit, toolkit, args.base_image, args.commit, (cutlass / "LICENSE.txt").read_text()))
    shutil.rmtree(jit, ignore_errors=True)
    os.rename(stage_dir, jit)
    size = sum(f.stat().st_size for f in jit.rglob("*") if f.is_file())
    say(f"staged {jit}: {sum(counts.values())} files ({size / 2**20:.1f} MiB): "
        + ", ".join(f"{k} {v}" for k, v in counts.items()))

    # _dsl/: the CuTe-DSL kernel, unchanged; its header carries its license.
    dsl = pkg / "_dsl"
    shutil.rmtree(dsl, ignore_errors=True)
    dsl.mkdir()
    shutil.copyfile(cutlass / DSL_KERNEL, dsl / DSL_KERNEL.name)
    (dsl / "THIRD_PARTY_NOTICES").write_text(
        f"{DSL_KERNEL.name}\n"
        f"    Source: NVIDIA CUTLASS commit {args.cutlass_commit} (v4.4.2), {DSL_KERNEL.as_posix()}, copied unchanged.\n"
        f"    License: BSD-3-Clause, stated in the file's header. Copyright (c) 2025 - 2026 NVIDIA CORPORATION &\n"
        f"    AFFILIATES.\n")
    say(f"staged {dsl / DSL_KERNEL.name}")

    import torch

    info = {
        "version": package_version(),
        "commit": args.commit,
        "dirty": bool(int(args.dirty)),
        "built_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "image": {"name": args.image, "id": args.image_id, "base": args.base_image},
        "cuda_toolkit": toolkit,
        "nvcc": first_line([str(cuda_home / "bin" / "nvcc"), "--version"], r"^Cuda compilation tools"),
        "gcc": first_line(["gcc", "--version"]),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "torch": str(torch.__version__),
        "cutlass_commit": args.cutlass_commit,
        "nvrtc": nvrtc,
        "arch_list": arch_list,
        "glibc_required": None,
    }
    assert tuple(info) == BUILD_INFO_KEYS
    (pkg / "BUILD_INFO.json").write_text(json.dumps(info, indent=2) + "\n")
    say(f"wrote {pkg / 'BUILD_INFO.json'}:\n{json.dumps(info, indent=2)}")
    return 0


# ---------------------------------------------------------------------------------------------- the extension's ELF
def symbol_versions(so: Path) -> dict:
    """{"GLIBC": "GLIBC_2.34", "GLIBCXX": ..., "CXXABI": ...}: the newest version of each that ``so`` needs."""
    newest: dict = {}
    for kind, ver in re.findall(r"Name:\s+(GLIBC|GLIBCXX|CXXABI)_(\d+(?:\.\d+)*)\b", run(["readelf", "-V", "-W", str(so)])):
        key = tuple(int(x) for x in ver.split("."))
        if kind not in newest or key > newest[kind][0]:
            newest[kind] = (key, f"{kind}_{ver}")
    return {k: v[1] for k, v in newest.items()}


def version_tuple(text: str) -> tuple:
    return tuple(int(x) for x in text.split("_")[-1].split("."))


def unpacked_extension(root: Path) -> Path:
    sos = sorted((root / "fish_scales_ops").glob("_C.*.so"))
    if len(sos) != 1:
        raise Failure(f"expected one fish_scales_ops/_C.*.so in the wheel, found {[s.name for s in sos]}")
    return sos[0]


# ---------------------------------------------------------------------------------------------------------- finalize
def finalize(args) -> int:
    wheel = Path(args.wheel).resolve()
    with tempfile.TemporaryDirectory(prefix="fso_wheel_finalize_") as tmp:
        tmp = Path(tmp)
        run([sys.executable, "-m", "wheel", "unpack", "--dest", str(tmp / "unpacked"), str(wheel)])
        (root,) = list((tmp / "unpacked").iterdir())
        versions = symbol_versions(unpacked_extension(root))
        if "GLIBC" not in versions:
            raise Failure("the extension names no GLIBC symbol version")
        path = root / "fish_scales_ops" / "BUILD_INFO.json"
        info = json.loads(path.read_text())
        info["glibc_required"] = versions["GLIBC"]
        text = json.dumps(info, indent=2) + "\n"
        path.write_text(text)
        (tmp / "packed").mkdir()
        run([sys.executable, "-m", "wheel", "pack", "--dest-dir", str(tmp / "packed"), str(root)])
        (packed,) = list((tmp / "packed").iterdir())
        if packed.name != wheel.name:
            raise Failure(f"wheel pack produced {packed.name}, not {wheel.name}")
        shutil.move(str(packed), str(wheel))
    (PKG / "BUILD_INFO.json").write_text(text)
    say(f"glibc_required = {versions['GLIBC']} written into {wheel.name} and {PKG / 'BUILD_INFO.json'} "
        f"(GLIBCXX {versions.get('GLIBCXX')}, CXXABI {versions.get('CXXABI')})")
    return 0


# ------------------------------------------------------------------------------------------------------------- check
def cubin_label(spec: str) -> str:
    """The `arch =` label cuobjdump prints for one TORCH_CUDA_ARCH_LIST entry: 9.0a -> sm_90a, 10.0f -> sm_100f."""
    m = re.fullmatch(r"\s*(\d+)\.(\d+)([a-zA-Z]?)\s*(\+PTX)?\s*", spec)
    if not m:
        raise Failure(f"unparseable arch list entry {spec!r}")
    return f"sm_{m.group(1)}{m.group(2)}{m.group(3)}"


def cuobjdump() -> str:
    for cand in (os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "cuobjdump"), shutil.which("cuobjdump")):
        if cand and os.path.isfile(cand):
            return cand
    raise Failure("cuobjdump not found (CUDA_HOME/bin or PATH)")


def check(args) -> int:
    wheel = Path(args.wheel).resolve()
    arch_list = args.arch_list or DEFAULT_ARCH_LIST
    results = []

    def record(ok: bool, what: str) -> None:
        results.append((ok, what))
        print(f"  {'OK  ' if ok else 'FAIL'} {what}", flush=True)

    print(f"check {wheel.name} ({wheel.stat().st_size / 2**20:.1f} MiB)")
    with zipfile.ZipFile(wheel) as z:
        names = set(z.namelist())
        sizes = {i.filename: i.file_size for i in z.infolist()}
        sos = sorted(n for n in names if re.fullmatch(r"fish_scales_ops/_C\.[^/]*\.so", n))
        record(len(sos) == 1, f"one extension: {sos}")
        for member in ("_nvrtc/libnvrtc.so.13", "_nvrtc/libnvrtc-builtins.so.13.0", "_nvrtc/VERSION",
                       "_dsl/dense_blockscaled_gemm_persistent.py", "_dsl/THIRD_PARTY_NOTICES", "BUILD_INFO.json"):
            record(f"fish_scales_ops/{member}" in names,
                   f"fish_scales_ops/{member} ({sizes.get('fish_scales_ops/' + member, 0)} bytes)")
        jit = sorted(n for n in names if n.startswith("fish_scales_ops/_jit_include/"))
        missing = [s for s in JIT_TREE_SENTINELS if f"fish_scales_ops/_jit_include/{s}" not in names]
        library = sorted({n.split("/")[2] for n in jit
                          if any(fnmatch.fnmatchcase(n.split("/")[2], p) for p in CUDA_LIBRARY_HEADERS)})
        record(not missing and not library,
               f"fish_scales_ops/_jit_include/: {len(jit)} files, {sum(sizes[n] for n in jit) / 2**20:.1f} MiB"
               + (f"; missing {missing}" if missing else "") + (f"; library headers {library}" if library else ""))
        info = json.loads(z.read("fish_scales_ops/BUILD_INFO.json")) if "fish_scales_ops/BUILD_INFO.json" in names \
            else {}
        absent = [k for k in BUILD_INFO_KEYS if info.get(k) in (None, "")]
        record(not absent, "BUILD_INFO.json names every key" + (f"; empty or missing: {absent}" if absent else ""))
        record(info.get("arch_list") == arch_list, f"BUILD_INFO.json arch_list {info.get('arch_list')!r} == "
                                                   f"{arch_list!r}")
        if len(sos) != 1:
            return finish(results)
        with tempfile.TemporaryDirectory(prefix="fso_wheel_check_") as tmp:
            so = Path(z.extract(sos[0], tmp))
            dyn = run(["readelf", "-d", "-W", str(so)])
            needed = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", dyn)
            record(bool(needed) and not [n for n in needed if "nvrtc" in n.lower()],
                   f"NEEDED {needed} holds no libnvrtc")
            undefined = [ln.split()[-1] for ln in run(["nm", "-D", "--undefined-only", str(so)]).splitlines()
                         if "nvrtc" in ln.lower()]
            record(not undefined, "no undefined nvrtc* symbol" + (f": {undefined}" if undefined else ""))
            versions = symbol_versions(so)
            glibc, glibcxx = versions.get("GLIBC"), versions.get("GLIBCXX")
            record(glibc is not None and version_tuple(glibc) <= version_tuple(args.max_glibc),
                   f"newest GLIBC needed: {glibc} (at most GLIBC_{args.max_glibc})")
            record(glibcxx is None or version_tuple(glibcxx) <= version_tuple(args.max_glibcxx),
                   f"newest GLIBCXX needed: {glibcxx} (at most GLIBCXX_{args.max_glibcxx}); "
                   f"newest CXXABI: {versions.get('CXXABI')}")
            record(info.get("glibc_required") == glibc,
                   f"BUILD_INFO.json glibc_required {info.get('glibc_required')!r} == {glibc!r}")
            tool = cuobjdump()
            archs = sorted(set(re.findall(r"^arch = (\S+)", run([tool, "-res-usage", str(so)]), re.M)))
            want = [cubin_label(s) for s in arch_list.replace(",", ";").split(";") if s.strip()]
            record(all(w in archs for w in want), f"cubin architectures {archs} include {want}")
            ptx = [ln for ln in run([tool, "--list-ptx", str(so)]).splitlines() if ln.strip()]
            print(f"  note PTX entries: {len(ptx)}", flush=True)
    return finish(results)


def finish(results) -> int:
    bad = [w for ok, w in results if not ok]
    print(f"wheel check: {'PASS' if not bad else f'{len(bad)} FAILED'} ({len(results)} checks)", flush=True)
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage", help="write _jit_include/, _dsl/ and BUILD_INFO.json into python/fish_scales_ops/")
    s.add_argument("--commit", required=True, help="the commit the source comes from")
    s.add_argument("--dirty", required=True, choices=("0", "1"), help="1: the source has uncommitted changes")
    s.add_argument("--image", required=True, help="the build image's name")
    s.add_argument("--image-id", required=True, help="the build image's ID (sha256 of its configuration)")
    s.add_argument("--base-image", required=True, help="the base image, pinned by digest")
    s.add_argument("--cutlass-commit", required=True, help="the CUTLASS commit (checked by scripts/build_wheel.sh)")
    s.add_argument("--cutlass", help="CUTLASS checkout (default: CUTLASS_DIR, else 3rdparty/cutlass)")
    s.add_argument("--cuda-home", help="CUDA toolkit (default: CUDA_HOME, else /usr/local/cuda)")
    s.add_argument("--arch-list", help=f"the TORCH_CUDA_ARCH_LIST built (default: that variable, else "
                                       f"{DEFAULT_ARCH_LIST})")
    s.add_argument("--package", help="the package directory to stage into (default: python/fish_scales_ops of this "
                                     "repository); it must hold _nvrtc/ already")
    f = sub.add_parser("finalize", help="write glibc_required into the wheel's BUILD_INFO.json")
    f.add_argument("wheel")
    c = sub.add_parser("check", help="the post-build checks of a wheel")
    c.add_argument("wheel")
    c.add_argument("--arch-list", help=f"the architectures the extension must carry (default {DEFAULT_ARCH_LIST})")
    c.add_argument("--max-glibc", default="2.35")
    c.add_argument("--max-glibcxx", default="3.4.30")
    args = ap.parse_args(argv)
    try:
        return {"stage": stage, "finalize": finalize, "check": check}[args.cmd](args)
    except (Failure, subprocess.CalledProcessError) as e:
        detail = f"\n{e.stderr}" if isinstance(e, subprocess.CalledProcessError) and e.stderr else ""
        print(f"[build_wheel_package] ERROR: {e}{detail}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
