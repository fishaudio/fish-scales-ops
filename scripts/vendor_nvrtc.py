#!/usr/bin/env python3
"""Bundle the NVRTC that the sm_90 JIT compiles with: nvidia-cuda-nvrtc 13.2.78, unpacked next to the extension.

fish_scales_ops compiles its sm_90 GEMM kernels at run time with NVRTC. The extension does not link libnvrtc: on its
first sm_90 compile it loads ``python/fish_scales_ops/_nvrtc/libnvrtc.so.13`` privately (dlopen with RTLD_LOCAL), so
the kernels always come from the same compiler, whatever NVRTC torch or the CUDA toolkit provide. This script puts
that library there. It uses the standard library only and needs no GPU.

1. The wheel: ``--wheel PATH``, else the file the ``FSO_NVRTC_WHEEL`` environment variable names (offline builds),
   else ``python -m pip download nvidia-cuda-nvrtc==13.2.78 --no-deps --only-binary=:all:`` into a temporary
   directory, with the interpreter that runs this script.
2. Its sha256 must equal the hash pinned below for its file name (the hashes PyPI publishes for 13.2.78); any other
   file is refused.
3. ``libnvrtc.so.13``, ``libnvrtc-builtins.so.13.2`` and the wheel's license file are unpacked into a staging
   directory beside ``_nvrtc/``. The staged library is loaded with ctypes (RTLD_LOCAL), must report version 13.2, and
   must compile a trivial kernel for sm_90a with the builtins library staged next to it. A missing or foreign builtins
   library therefore fails here, not in a serving process.
4. A ``VERSION`` file (package version, wheel file name, sha256) is written, and the staging directory replaces
   ``_nvrtc/``.

When ``_nvrtc/VERSION`` already names the wheel this run would use and both libraries exist, the script does nothing.
Every failure prints a message and exits non-zero, leaving an existing ``_nvrtc/`` as it was. ``scripts/build.sh``
runs this script before it compiles an sm_90 build; ``FSO_SKIP_NVRTC_VENDOR=1`` skips that step, and a process then
needs ``FSO_JIT_NVRTC_LIB`` pointing at a CUDA 13.2 ``libnvrtc.so.13``.

usage: python scripts/vendor_nvrtc.py [--wheel PATH]
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import platform
import posixpath
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path

PACKAGE = "nvidia-cuda-nvrtc"
VERSION = "13.2.78"
# file name -> (sha256, machine, download URL), from https://pypi.org/pypi/nvidia-cuda-nvrtc/13.2.78/json. Every Linux
# wheel of the release is listed.
WHEELS = {
    "nvidia_cuda_nvrtc-13.2.78-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl": (
        "a9049031da08cbedd0c20e3470e5a978dc330af0e0326b3b05774718c665dc3e",
        "x86_64",
        "https://files.pythonhosted.org/packages/5f/96/237b40b171e06eb65905375c4ad5c96f78c2f861ac6e8ae7f650d95e1dfd/"
        "nvidia_cuda_nvrtc-13.2.78-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
    ),
    "nvidia_cuda_nvrtc-13.2.78-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl": (
        "a50367a7e2a0bd00fb27e5648179149cc7a60e7c7811740a5ff559f06234526d",
        "aarch64",
        "https://files.pythonhosted.org/packages/af/be/8476aa006686fb264d61de43e0408a8dbd001003a702574759b25e645587/"
        "nvidia_cuda_nvrtc-13.2.78-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl",
    ),
}
LIBRARY = "libnvrtc.so.13"
BUILTINS = "libnvrtc-builtins.so.13.2"
LICENSE = "License.txt"
WANT_VERSION = (13, 2)

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "python" / "fish_scales_ops" / "_nvrtc"


class VendorError(RuntimeError):
    pass


def say(msg: str) -> None:
    print(f"[vendor_nvrtc] {msg}", flush=True)


def version_text(wheel_name: str, sha256: str) -> str:
    return f"{PACKAGE} {VERSION}\nwheel {wheel_name}\nsha256 {sha256}\n"


def machine_wheel(machine: str) -> str:
    names = [n for n, (_sha, m, _url) in WHEELS.items() if m == machine]
    if sys.platform != "linux" or len(names) != 1:
        raise VendorError(f"no pinned {PACKAGE} {VERSION} wheel for {sys.platform} {machine}; the bundled NVRTC "
                          f"exists for Linux x86_64 and aarch64 only")
    return names[0]


def is_current(wheel_name: str) -> bool:
    """True when _nvrtc/ already holds what this run would unpack: VERSION names the wheel and its pinned hash, and
    both libraries exist."""
    if wheel_name not in WHEELS:
        return False
    try:
        text = (DEST / "VERSION").read_text()
    except OSError:
        return False
    return text == version_text(wheel_name, WHEELS[wheel_name][0]) and \
        (DEST / LIBRARY).is_file() and (DEST / BUILTINS).is_file()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(tmp: Path, wheel_name: str) -> Path:
    url = WHEELS[wheel_name][2]
    cmd = [sys.executable, "-m", "pip", "download", f"{PACKAGE}=={VERSION}", "--no-deps", "--only-binary=:all:",
           "--progress-bar", "off", "--dest", str(tmp)]
    say("downloading: " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise VendorError(f"pip download failed (exit {r.returncode}). For an offline build, fetch {url} and pass "
                          f"it with --wheel PATH or FSO_NVRTC_WHEEL=PATH")
    wheels = sorted(tmp.glob("*.whl"))
    if len(wheels) != 1:
        raise VendorError(f"pip download left {[w.name for w in wheels]} in {tmp}, not one wheel")
    return wheels[0]


def unpack(wheel: Path, stage: Path) -> None:
    """Copy the two libraries and the license file out of the wheel. The wheel's other files (the header, its
    metadata, and the "alt" library variants that some releases carry) are not needed."""
    wanted = {LIBRARY: None, BUILTINS: None, LICENSE: None}
    try:
        with zipfile.ZipFile(wheel) as z:
            for info in z.infolist():
                base = posixpath.basename(info.filename)
                if base in (LIBRARY, BUILTINS) and "/lib/" in "/" + info.filename:
                    key = base
                elif base == LICENSE and ".dist-info/" in info.filename:
                    key = LICENSE
                else:
                    continue
                if wanted[key] is not None:
                    raise VendorError(f"{wheel.name} holds two {key} files: {wanted[key].filename}, {info.filename}")
                wanted[key] = info
            missing = [k for k, v in wanted.items() if v is None]
            if missing:
                raise VendorError(f"{wheel.name} has no {', '.join(missing)}")
            for key, info in wanted.items():
                with z.open(info) as src, open(stage / key, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                os.chmod(stage / key, 0o644)
    except zipfile.BadZipFile as e:
        raise VendorError(f"{wheel} is not a readable wheel: {e}") from e


def mapped_files(fragment: str) -> list[str] | None:
    """The files of this process's memory map whose path contains ``fragment``; None when /proc is unavailable."""
    try:
        with open("/proc/self/maps") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    return sorted({ln.split(maxsplit=5)[5] for ln in lines if len(ln.split(maxsplit=5)) == 6 and fragment in ln})


def self_check(stage: Path) -> str:
    """Load the staged NVRTC privately, check its version and compile a trivial kernel for sm_90a; returns a summary."""
    lib_path = stage / LIBRARY
    try:
        lib = ctypes.CDLL(str(lib_path), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    except OSError as e:
        raise VendorError(f"cannot load {lib_path}: {e}") from e
    lib.nvrtcGetErrorString.restype = ctypes.c_char_p
    lib.nvrtcGetErrorString.argtypes = [ctypes.c_int]

    def err(rc: int) -> str:
        return (lib.nvrtcGetErrorString(rc) or b"?").decode(errors="replace")

    major, minor = ctypes.c_int(), ctypes.c_int()
    rc = lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    if rc != 0:
        raise VendorError(f"nvrtcVersion failed in {lib_path}: {err(rc)}")
    if (major.value, minor.value) != WANT_VERSION:
        raise VendorError(f"{lib_path} reports NVRTC {major.value}.{minor.value}, not "
                          f"{WANT_VERSION[0]}.{WANT_VERSION[1]}")

    src = b'extern "C" __global__ void fso_vendor_nvrtc_check(float* x) { x[threadIdx.x] *= 2.0f; }\n'
    prog = ctypes.c_void_p()
    rc = lib.nvrtcCreateProgram(ctypes.byref(prog), src, b"fso_vendor_nvrtc_check.cu", 0, None, None)
    if rc != 0:
        raise VendorError(f"nvrtcCreateProgram failed: {err(rc)}")
    try:
        opts = (ctypes.c_char_p * 1)(b"--gpu-architecture=sm_90a")
        rc = lib.nvrtcCompileProgram(prog, 1, opts)
        if rc != 0:
            size = ctypes.c_size_t()
            lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
            log = ctypes.create_string_buffer(max(size.value, 1))
            lib.nvrtcGetProgramLog(prog, log)
            raise VendorError(f"NVRTC from {lib_path} cannot compile a trivial sm_90a kernel ({err(rc)}); a missing "
                              f"or unusable {BUILTINS} is the usual cause. Log: {log.value.decode(errors='replace')}")
        size = ctypes.c_size_t()
        rc = lib.nvrtcGetCUBINSize(prog, ctypes.byref(size))
        if rc != 0 or size.value == 0:
            raise VendorError(f"NVRTC from {lib_path} produced no cubin ({err(rc)})")
    finally:
        lib.nvrtcDestroyProgram(ctypes.byref(prog))

    # NVRTC finds its builtins library through the normal library search, in which a system copy (an ld.so.cache
    # entry of a CUDA 13.2 toolkit, say) can stand in for a missing staged one; insist on the staged file.
    builtins = mapped_files("libnvrtc-builtins")
    if builtins is not None:
        want = os.path.realpath(stage / BUILTINS)
        if [os.path.realpath(p) for p in builtins] != [want]:
            raise VendorError(f"NVRTC from {lib_path} mapped the builtins library {builtins}, not {want}")
    return f"NVRTC {major.value}.{minor.value} compiled a test kernel for sm_90a ({size.value} bytes of cubin)"


def replace_dest(stage: Path) -> None:
    old = None
    if DEST.exists() or DEST.is_symlink():
        old = DEST.with_name(f"{DEST.name}.old-{uuid.uuid4().hex[:8]}")
        os.rename(DEST, old)
    os.rename(stage, DEST)
    if old is not None:
        if old.is_dir() and not old.is_symlink():
            shutil.rmtree(old, ignore_errors=True)
        else:
            old.unlink()


def run(wheel_arg: str | None) -> None:
    machine = platform.machine()
    wanted_name = Path(wheel_arg).name if wheel_arg else machine_wheel(machine)
    if is_current(wanted_name):
        say(f"{DEST} already holds {PACKAGE} {VERSION} ({wanted_name}); nothing to do")
        return
    if wheel_arg is None:
        say(f"bundling {PACKAGE} {VERSION} for {machine} into {DEST} (about 120 MB unpacked)")
    with tempfile.TemporaryDirectory(prefix="fso_nvrtc_wheel_") as tmp:
        if wheel_arg:
            wheel = Path(wheel_arg)
            if not wheel.is_file():
                raise VendorError(f"the wheel {wheel} does not exist")
            say(f"using the wheel {wheel}")
        else:
            wheel = download(Path(tmp), wanted_name)
        if wheel.name not in WHEELS:
            raise VendorError(f"{wheel.name} is not a pinned {PACKAGE} {VERSION} wheel; the pinned files are "
                              + ", ".join(WHEELS))
        pinned, wheel_machine, url = WHEELS[wheel.name]
        if wheel_machine != machine:
            raise VendorError(f"{wheel.name} is the {wheel_machine} wheel, and this machine is {machine}")
        digest = sha256_of(wheel)
        if digest != pinned:
            raise VendorError(f"{wheel} has sha256 {digest}, but {wheel.name} is pinned to {pinned}; refusing it "
                              f"(the release file is {url})")
        say(f"sha256 {digest} matches the pin")

        DEST.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f"{DEST.name}.staging-", dir=DEST.parent))
        try:
            unpack(wheel, stage)
            summary = self_check(stage)
            (stage / "VERSION").write_text(version_text(wheel.name, digest))
            os.chmod(stage, 0o755)
            replace_dest(stage)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    say(f"bundled {PACKAGE} {VERSION} into {DEST}: {LIBRARY}, {BUILTINS}, {LICENSE}, VERSION; {summary}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wheel", metavar="PATH",
                    help="the nvidia-cuda-nvrtc 13.2.78 wheel to unpack, for offline builds (default: the "
                         "FSO_NVRTC_WHEEL environment variable, else pip download)")
    args = ap.parse_args(argv)
    wheel_arg = args.wheel or os.environ.get("FSO_NVRTC_WHEEL") or None
    try:
        run(wheel_arg)
    except VendorError as e:
        print(f"[vendor_nvrtc] ERROR: {e}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
