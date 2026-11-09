"""Build the unified fish_scales_ops PyTorch extension.

One CUDAExtension (`fish_scales_ops._C`) for both domains:

  csrc/gemm/      — block-FP8 1×128 / 128×128 (sm_90 deep_gemm JIT, sm_120
                    CUTLASS Sm120BlockScaledKernel, sm_100/sm_103 on the
                    MXFP8 tcgen05 kernels) and MXFP8 1×32 (sm_100/sm_103 and
                    sm_120/sm_121), dense and grouped MoE. Registers torch.ops
                    under `fish_scales_ops` via TORCH_LIBRARY_FRAGMENT.
  csrc/attention/ — SM120 MXFP8 attention: contiguous prefill, paged decode
                    and paged prefill (`mxfp8_attn_fwd`, `mxfp8_decode_paged`,
                    `mxfp8_attn_fwd_paged`).

Default archs: 9.0a + 12.0a (Hopper + Blackwell consumer). B200 / B300
(sm_100 / sm_103) need the 10.0f family target in the list, which the default
omits; without it the extension carries no kernels for those devices.
Override via:
    export TORCH_CUDA_ARCH_LIST="9.0a;10.0f;12.0a"

CUTLASS root: defaults to `../3rdparty/cutlass` (the repo submodule).
Override with `CUTLASS_DIR` env var.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
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


# Two sm_100 kernels are forks of a CUTLASS header, produced by in-tree
# generators at build time: the slot-bound grouped kernel and the fused-SwiGLU
# FC1 epilogue. Nothing under 3rdparty/ is modified and neither generated header
# is ever written into the source tree: both go into the build directory, and
# only the translation units that include them get that directory on their
# include path.
SLOT_TU = CSRC / "gemm" / "ops" / "mxfp8_sm100_slot_kernel.cu"
# The compact slot GEMMs with the row-bounded token load (issue #13 lever 3), in their own unit so that SLOT_TU keeps
# its device code (run fso_issue12_20261028/R1); it needs both slot headers too.
SLOT_ROWS_TU = CSRC / "gemm" / "ops" / "mxfp8_sm100_slot_rows_kernel.cu"
GROUPED_TU = CSRC / "gemm" / "ops" / "mxfp8_sm100_grouped_kernel.cu"
# The persistent FC1 + FC2 slot kernel of the MoE layer (issue #13 lever 5), in its own unit for the same reason; it
# needs both slot headers and the fused class's own generated header.
SLOT_MOE_TU = CSRC / "gemm" / "ops" / "mxfp8_sm100_slot_moe_kernel.cu"
# The same kernel's token-tile mode (issue #13 lever 5 step 3), in a unit of its own so that the slot mode's unit keeps
# its device code (run fso_issue12_20261028/R1); the same three generated headers.
SLOT_MOE_LIST_TU = CSRC / "gemm" / "ops" / "mxfp8_sm100_slot_moe_list_kernel.cu"

# The sm_100 dense-epilogue units (bias / erf-GELU / residual and the quantized output of linear_quantized, one
# configuration group each, blockscale_gemm/arch/sm100/mxfp8/dense_common.cuh). All of them include
# dense_epilogue_gemm.cuh, which includes the multi-warpgroup clone of CUTLASS's sm_100 TMA epilogue (issue #12 FB3;
# only the quantized-output units instantiate it); the identity kernels' unit (mxfp8_sm100_kernel.cu) does not include
# it, so its device code is untouched.
DENSE_EPI_TUS = [CSRC / "gemm" / "ops" / f"mxfp8_sm100_epi{q}_kernel_{g}.cu"
                 for q in ("", "_quant") for g in ("1sm", "2sm", "nosmem")]

# (generator script, header path inside the generated-include root, the
# translation unit that includes it[, extra generator arguments]). The TU column
# is what scopes the include path; a second entry for the same TU would simply
# add a second header.
GENERATED_HEADERS = [
    (CSRC / "gemm" / "tools" / "make_sm100_slot_kernel.py",
     Path("blockscale_gemm/arch/sm100/mxfp8/sm100_slot_gemm_kernel.hpp"),
     SLOT_TU),
    (CSRC / "gemm" / "tools" / "make_sm100_fused_swiglu_epilogue.py",
     Path("blockscale_gemm/arch/sm100/mxfp8/sm100_fused_swiglu_epilogue.hpp"),
     GROUPED_TU),
    # The slot route's own fused-SwiGLU epilogue. It is a second header for the
    # SAME translation unit as the slot kernel, which is what the (generator,
    # header, TU) shape of this list is for.
    (CSRC / "gemm" / "tools" / "make_sm100_fused_swiglu_slot_epilogue.py",
     Path("blockscale_gemm/arch/sm100/mxfp8/sm100_fused_swiglu_slot_epilogue.hpp"),
     SLOT_TU),
    (CSRC / "gemm" / "tools" / "make_sm100_slot_kernel.py",
     Path("blockscale_gemm/arch/sm100/mxfp8/sm100_slot_gemm_kernel.hpp"),
     SLOT_ROWS_TU),
    (CSRC / "gemm" / "tools" / "make_sm100_fused_swiglu_slot_epilogue.py",
     Path("blockscale_gemm/arch/sm100/mxfp8/sm100_fused_swiglu_slot_epilogue.hpp"),
     SLOT_ROWS_TU),
    (CSRC / "gemm" / "tools" / "make_sm100_slot_kernel.py",
     Path("blockscale_gemm/arch/sm100/mxfp8/sm100_slot_gemm_kernel.hpp"),
     SLOT_MOE_TU),
    (CSRC / "gemm" / "tools" / "make_sm100_fused_swiglu_slot_epilogue.py",
     Path("blockscale_gemm/arch/sm100/mxfp8/sm100_fused_swiglu_slot_epilogue.hpp"),
     SLOT_MOE_TU),
    (CSRC / "gemm" / "tools" / "make_sm100_slot_kernel.py",
     Path("blockscale_gemm/arch/sm100/mxfp8/sm100_slot_moe_kernel.hpp"),
     SLOT_MOE_TU, ("--moe",)),
    (CSRC / "gemm" / "tools" / "make_sm100_slot_kernel.py",
     Path("blockscale_gemm/arch/sm100/mxfp8/sm100_slot_gemm_kernel.hpp"),
     SLOT_MOE_LIST_TU),
    (CSRC / "gemm" / "tools" / "make_sm100_fused_swiglu_slot_epilogue.py",
     Path("blockscale_gemm/arch/sm100/mxfp8/sm100_fused_swiglu_slot_epilogue.hpp"),
     SLOT_MOE_LIST_TU),
    (CSRC / "gemm" / "tools" / "make_sm100_slot_kernel.py",
     Path("blockscale_gemm/arch/sm100/mxfp8/sm100_slot_moe_kernel.hpp"),
     SLOT_MOE_LIST_TU, ("--moe",)),
] + [
    # The dense epilogue's multi-warpgroup TMA epilogue (issue #12 FB3), one row per dense-epilogue unit.
    (CSRC / "gemm" / "tools" / "make_sm100_dense_epi_multiwg.py",
     Path("blockscale_gemm/arch/sm100/mxfp8/sm100_dense_epi_multiwg.hpp"),
     tu)
    for tu in DENSE_EPI_TUS
]


def _generate_headers(gen_root: Path) -> None:
    """Run every generator into `gen_root`. Each one's textual edits assert
    their exact match count, so a CUTLASS bump that moved any of them fails the
    build here, naming the edit, rather than producing a silently different
    kernel."""
    env = dict(os.environ, CUTLASS_DIR=str(CUTLASS))
    done = set()
    for generator, header, _tu, *extra in GENERATED_HEADERS:
        if header in done:  # a header two translation units include is generated once
            continue
        done.add(header)
        dst = gen_root / header
        dst.parent.mkdir(parents=True, exist_ok=True)
        flags = list(extra[0]) if extra else []
        subprocess.run([sys.executable, str(generator), *flags, str(dst)], check=True, env=env)


class FsoBuildExtension(BuildExtension):
    """BuildExtension that generates the sm_100 headers listed in
    GENERATED_HEADERS and scopes their include path to the translation units
    that need them.

    setuptools compiles every source of one extension with one flag set, so the
    scoping is done by splitting the compile call in two: the translation units
    that consume a generated header are compiled on their own with the
    generated-header directory appended to the include path, everything else
    with the include path the extension declares. That keeps the command line of
    every sm_90 / sm_120 object unchanged. `self.compiler.compile` is wrapped
    here rather than in `build_extensions` because the base class installs its
    own ninja wrapper before it reaches this point.
    """

    def build_extension(self, ext) -> None:
        gen_root = Path(self.build_temp).resolve() / "fso_generated"
        _generate_headers(gen_root)
        gen_srcs = {os.path.normpath(_rel(row[2])) for row in GENERATED_HEADERS}
        original_compile = self.compiler.compile

        def split_compile(sources, **kwargs):
            gen = [s for s in sources if os.path.normpath(s) in gen_srcs]
            rest = [s for s in sources if os.path.normpath(s) not in gen_srcs]
            objects = list(original_compile(rest, **kwargs)) if rest else []
            if gen:
                gen_kwargs = dict(kwargs)
                gen_kwargs["include_dirs"] = list(kwargs.get("include_dirs") or []) + [str(gen_root)]
                objects += list(original_compile(gen, **gen_kwargs))
            return objects

        self.compiler.compile = split_compile
        try:
            super().build_extension(ext)
        finally:
            self.compiler.compile = original_compile


def _gemm_sources() -> list[str]:
    base = CSRC / "gemm"
    return [
        _rel(base / "bindings.cpp"),
        _rel(base / "ops" / "fp8.cu"),
        _rel(base / "ops" / "mxfp8.cu"),
        _rel(base / "ops" / "mxfp8_kernel.cu"),
        _rel(base / "ops" / "mxfp8_dense_epi_kernel.cu"),
        # The quantized-output (linear_quantized) instances of the sm_120 cascade and their Stream-K reduce.
        _rel(base / "ops" / "mxfp8_dense_quant_kernel.cu"),
        # The sm_120 small-M route of the decode band: its host entry and its kernels, one unit per (form, epilogue).
        _rel(base / "ops" / "mxfp8_smallm_kernel.cu"),
        _rel(base / "ops" / "mxfp8_smallm_q_kernel.cu"),
        _rel(base / "ops" / "mxfp8_smallm_q_epi_kernel.cu"),
        _rel(base / "ops" / "mxfp8_smallm_bf16_kernel.cu"),
        _rel(base / "ops" / "mxfp8_smallm_bf16_epi_kernel.cu"),
        _rel(base / "ops" / "mxfp8_compact.cu"),
        _rel(base / "ops" / "mxfp8_compact_kernel.cu"),
        _rel(base / "ops" / "mxfp8_sm100_kernel.cu"),
        # The dense-epilogue (bias / erf-GELU / residual) kernels of the sm_100 cascade, one configuration group
        # per unit so they compile in parallel (blockscale_gemm/arch/sm100/mxfp8/dense_common.cuh).
        _rel(base / "ops" / "mxfp8_sm100_epi_kernel_1sm.cu"),
        _rel(base / "ops" / "mxfp8_sm100_epi_kernel_2sm.cu"),
        _rel(base / "ops" / "mxfp8_sm100_epi_kernel_nosmem.cu"),
        # The split-K reduce that applies the dense epilogue, in its own unit so the identity kernels' unit
        # (mxfp8_sm100_kernel.cu) keeps exactly the device code it had before the epilogue (issue #12, R1).
        _rel(base / "ops" / "mxfp8_sm100_splitk_epi_kernel.cu"),
        # The quantized-output (linear_quantized) kernels of the same configurations, grouped the same way, plus the
        # quantized reduce of the two-kernel split-K (in the nosmem unit).
        _rel(base / "ops" / "mxfp8_sm100_epi_quant_kernel_1sm.cu"),
        _rel(base / "ops" / "mxfp8_sm100_epi_quant_kernel_2sm.cu"),
        _rel(base / "ops" / "mxfp8_sm100_epi_quant_kernel_nosmem.cu"),
        _rel(GROUPED_TU),
        _rel(SLOT_TU),
        _rel(SLOT_ROWS_TU),
        _rel(SLOT_MOE_TU),
        _rel(SLOT_MOE_LIST_TU),
        _rel(base / "ops" / "quant_kernels.cu"),
        _rel(base / "ops" / "moe_glue.cu"),
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
# The include root of blockscale_gemm/common/epilogue_device.cuh, the dense epilogue the sm_90 JIT kernels share with
# the AOT routes (the only JIT header outside deep_gemm/). Last, so that it shadows nothing.
JIT_INCLUDE_DIRS.append(str(CSRC / "gemm/include"))


# fso/activation_contract.h (the published activation contract; it ships in the package): the quantized-output
# epilogues (sm_120: blockscale_gemm/arch/sm120/mxfp8/dense_quant_epi.cuh; sm_100:
# blockscale_gemm/arch/sm100/mxfp8/dense_epilogue_gemm.cuh and the split-K reduce of
# ops/mxfp8_sm100_epi_quant_kernel_nosmem.cu) quantize with its functions. Last, so that it shadows nothing.
ACTIVATION_CONTRACT_INCLUDE = Path(__file__).resolve().parent / "fish_scales_ops" / "include"

include_dirs = [
    str(CSRC / "gemm/include"),
    # fso/activation_contract.h (the activation quantizer rule; the sm_120 small-M route quantizes with it)
    str(HERE / "fish_scales_ops" / "include"),
    str(CSRC / "common/compat/include"),
    str(CUTLASS / "include"),
    str(CUTLASS / "tools/util/include"),
    str(ACTIVATION_CONTRACT_INCLUDE),  # fso/activation_contract.h
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


# No libnvrtc here: the sm_90 JIT loads the NVRTC that scripts/vendor_nvrtc.py bundles in fish_scales_ops/_nvrtc/
# privately at run time (csrc/gemm/include/blockscale_gemm/arch/sm90/fp8/jit/deep_gemm/jit_utils.cuh). A link-time
# libnvrtc would bind every NVRTC call to whichever libnvrtc.so.13 the process loaded first, which is torch's.
ext = CUDAExtension(
    name="fish_scales_ops._C",
    sources=_gemm_sources() + _attention_sources(),
    include_dirs=include_dirs,
    extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
    libraries=["cuda"],
)


setup(
    ext_modules=[ext],
    cmdclass={"build_ext": FsoBuildExtension},
)
