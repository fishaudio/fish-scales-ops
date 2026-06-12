"""Build the unified fish_scales_ops PyTorch extension.

One CUDAExtension (`fish_scales_ops._C`) for both domains:

  csrc/gemm/      — FP8 1×128 / 128×128 (sm_90 deep_gemm JIT + sm_120
                    CUTLASS Sm120BlockScaledKernel) and MXFP8 1×32
                    (sm_120 only). Registers torch.ops under
                    `fish_scales_ops` via TORCH_LIBRARY_FRAGMENT.
  csrc/attention/ — SM120 MXFP8 attention forward + paged decode
                    (`mxfp8_attn_fwd`, `mxfp8_decode_paged`).

Default archs: 9.0a + 12.0a (Hopper + Blackwell consumer). Override via:
    export TORCH_CUDA_ARCH_LIST="12.0a"

CUTLASS root: defaults to `../3rdparty/cutlass` (the repo submodule).
Override with `CUTLASS_DIR` env var.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension


HERE = Path(__file__).resolve().parent           # python/
REPO = HERE.parent                                # fish-scales-ops/
CSRC = REPO / "csrc"
THIRDPARTY = REPO / "3rdparty"


def _rel(p: Path | str) -> str:
    """Path relative to setup.py (= python/). setuptools rejects absolute
    paths in `sources` during editable installs (PEP 660 build_editable),
    so always pass sources as ../csrc/... relative strings."""
    p = Path(p)
    try:
        return str(p.relative_to(HERE))
    except ValueError:
        # Outside of HERE — express as relative via os.path.relpath.
        return os.path.relpath(p, start=HERE)


def _resolve_cutlass() -> Path:
    for cand in (os.environ.get("CUTLASS_DIR"), os.environ.get("BSGEMM_CUTLASS_DIR")):
        if cand:
            p = Path(cand)
            if (p / "include" / "cutlass").is_dir():
                return p
            raise RuntimeError(f"CUTLASS_DIR={cand} but no include/cutlass under it")
    sub = THIRDPARTY / "cutlass"
    if (sub / "include" / "cutlass").is_dir():
        return sub.resolve()
    raise RuntimeError(
        f"CUTLASS not found at {sub}. Initialise the submodule "
        "(git submodule update --init --recursive) or set CUTLASS_DIR."
    )


CUTLASS = _resolve_cutlass()

# Default arch list — Hopper + Blackwell consumer. setup.py honours the
# user's TORCH_CUDA_ARCH_LIST if set; otherwise it preseeds the env.
if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a;12.0a"


_TORCH_ARCH_RE = re.compile(r"^\s*(\d+)\.(\d+)([a-zA-Z]?)\s*(\+PTX)?\s*$")


def _explicit_gencode_flags() -> list[str]:
    """Emit -gencode flags ourselves so PyTorch's _get_cuda_arch_flags
    doesn't skip its own emission when any user cflag contains "arch"
    (our JIT include path ...arch/sm90/fp8/jit... trips that check)."""
    raw = os.environ.get("TORCH_CUDA_ARCH_LIST", "").replace(",", ";").replace(" ", ";")
    flags: list[str] = []
    for spec in raw.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        m = _TORCH_ARCH_RE.match(spec)
        if not m:
            raise ValueError(f"unparseable TORCH_CUDA_ARCH_LIST entry: {spec!r}")
        major, minor, suffix, ptx = m.group(1), m.group(2), m.group(3) or "", m.group(4) or ""
        cap = f"{major}{minor}{suffix}"
        flags.append(f"-gencode=arch=compute_{cap},code=sm_{cap}")
        if ptx:
            flags.append(f"-gencode=arch=compute_{cap},code=compute_{cap}")
    return flags


def _gemm_sources() -> list[str]:
    base = CSRC / "gemm"
    return [
        _rel(base / "bindings.cpp"),
        _rel(base / "ops" / "fp8.cu"),
        _rel(base / "ops" / "mxfp8.cu"),
        _rel(base / "ops" / "mxfp8_kernel.cu"),
        _rel(base / "ops" / "mxfp8_sm100_kernel.cu"),
        _rel(base / "ops" / "quant_kernels.cu"),
        _rel(base / "src" / "runner.cu"),
    ]


def _attention_sources() -> list[str]:
    base = CSRC / "attention"
    cpp = [_rel(base / "csrc" / "flash_attn_ext.cpp")]
    # SM120 MXFP8 attention forward kernel — the only attention .cu's compiled
    # into _C.so. Per-D translation units (d32, d64, d128) keep each ptxas
    # pass independent; the top-level dispatch (mxfp8_attn_fwd.cu) refuses
    # to launch on sm < 12.0 at runtime. The kernel bodies are guarded with
    # #if __CUDA_ARCH__ >= 1200 so the sm_90a SASS pass emits an empty stub.
    sm120 = base / "kernels" / "sm120"
    # Production kernels only. Historical experiments + profiling probes
    # live in kernels/sm120/experiments/ (see that dir's README for the
    # full index). To rebuild one, add its .cu here and re-add the
    # matching extern decl + dispatcher entry in mxfp8_attn_fwd.cu /
    # flash_attn_ext.cpp.
    cu_files = [
        sm120 / "mxfp8_attn_fwd.cu",
        sm120 / "mxfp8_attn_fwd_d32.cu",
        sm120 / "mxfp8_attn_fwd_d64.cu",
        sm120 / "mxfp8_attn_fwd_d128.cu",
        sm120 / "mxfp8_attn_fwd_d256.cu",
        sm120 / "mxfp8_decode_paged.cu",
        sm120 / "mxfp8_decode_paged_d32.cu",
        sm120 / "mxfp8_decode_paged_d64.cu",
        sm120 / "mxfp8_decode_paged_d128.cu",
        sm120 / "mxfp8_decode_paged_d256.cu",
        sm120 / "mxfp8_attn_fwd_paged.cu",
        sm120 / "mxfp8_attn_fwd_paged_d32.cu",
        sm120 / "mxfp8_attn_fwd_paged_d64.cu",
        sm120 / "mxfp8_attn_fwd_paged_d128.cu",
        sm120 / "mxfp8_attn_fwd_paged_d256.cu",
    ]
    cu = [_rel(p) for p in cu_files]
    return cpp + cu


JIT_INCLUDE_DIRS = [
    str(CUTLASS / "include"),
    str(CUTLASS / "tools/util/include"),
    str(CSRC / "gemm/include/blockscale_gemm/arch/sm90/fp8/jit"),
]
if CUDA_HOME:
    JIT_INCLUDE_DIRS.append(str(Path(CUDA_HOME) / "include"))
    cccl = Path(CUDA_HOME) / "include" / "cccl"
    if cccl.is_dir():
        JIT_INCLUDE_DIRS.append(str(cccl))


include_dirs = [
    str(CSRC / "gemm/include"),
    str(CSRC / "common/compat/include"),
    str(CUTLASS / "include"),
    str(CUTLASS / "tools/util/include"),
]

nvcc_flags = (
    _explicit_gencode_flags()
    + [
        "-O3", "-std=c++17",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "-Xcompiler=-Wno-psabi",
        "--diag-suppress=20012,20013,20014,177",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        f'-DFSO_JIT_INCLUDE_DIRS_DEFAULT="{":".join(JIT_INCLUDE_DIRS)}"',
        "-DENABLE_BF16", "-DENABLE_FP8",
        "-DCOMPILE_HOPPER_TMA_GEMMS", "-DCOMPILE_HOPPER_TMA_GROUPED_GEMMS",
    ]
)

cxx_flags = ["-O3", "-std=c++17", "-Wno-psabi", "-Wno-deprecated-declarations"]


ext = CUDAExtension(
    name="fish_scales_ops._C",
    sources=_gemm_sources() + _attention_sources(),
    include_dirs=include_dirs,
    extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
    libraries=["cuda", "nvrtc"],
)


setup(
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
