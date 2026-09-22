# Vendored SM100 block-scaled CuTe-DSL GEMM kernels

These three Python files are a trimmed copy of NVIDIA's SM100 block-scaled
CuTe-DSL GEMM kernels as shipped inside FlashInfer. They are the kernels the
sm_100/sm_103 MXFP8 **decode** row of fish-scales-ops runs; see
`python/fish_scales_ops/gemm/_sm100_decode.py` for how they are configured,
routed and launched.

## Source

| item | value |
|---|---|
| package | `flashinfer-python` |
| version | 0.6.18.post1 (wheel `flashinfer_python-0.6.18.post1-py3-none-any.whl`) |
| paths inside the wheel | `flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100.py`, `flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100_splitk.py`, `flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100_common.py` |
| copied on | 2026-09-22 |
| md5 of the wheel files | `99892be684e2b5af81662ed0768ce9bb`, `d2f29587f1247136bab1acd37bca040a`, `dba4261b2cac84e24bcf79bc46dcb0ba` (in the order above) |

`MANIFEST.md5` in this directory records the md5 of every file as vendored and
of the wheel file it came from, so the edits below can be audited by diffing a
fresh extraction of the same wheel against this directory.

## Licence

Each of the three files carries, unmodified, the NVIDIA copyright notice, the
`SPDX-License-Identifier: BSD-3-Clause` tag, the three BSD conditions and the
disclaimer, exactly as they appear in the source wheel. `LICENSE` in this
directory carries the same text once more so the notice is visible without
opening a source file. The *package* `flashinfer-python` declares Apache-2.0
for the project as a whole, but these particular files are separately licensed
BSD-3-Clause with NVIDIA as the copyright holder, and BSD-3-Clause is what
applies to them here.

fish-scales-ops is Apache-2.0 (`python/pyproject.toml`). BSD-3-Clause imposes
no obligation that Apache-2.0 conflicts with: on redistribution in source form
the notice, the three conditions and the disclaimer must travel with the files,
which they do; in binary form the same three items must appear in the
accompanying documentation, which is what `LICENSE` and this README are for.
The third condition forbids using NVIDIA's name or its contributors' names to
endorse or promote derived products, so nothing in fish-scales-ops does.

The split-K file additionally states, in its own header, that it "keeps
FlashInfer's block-scaled MXFP8 mainloop and adapts the cluster-local reduction
protocol from NVIDIA DKG's FP8 split-K tutorial". That statement is preserved.

## What is here, and why these three files

`dense_blockscaled_gemm_sm100_splitk.py`
: `Sm100BlockScaledSplitKGemmKernel` — a physical `(1, 1, split_k)` cluster
  computes one output tile, each CTA accumulates a disjoint K slice in FP32 in
  its own TMEM, and the peers push their partials into the owner CTA's shared
  memory over `mapa.shared::cluster` +
  `st.async.shared::cluster.mbarrier::complete_tx.bytes.v4.b32`. No workspace,
  no second reduction kernel. Used for narrow-N decode shapes.

`dense_blockscaled_gemm_sm100.py`
: `Sm100BlockScaledPersistentDenseGemmKernel` — the plain persistent
  block-scaled kernel. Used for wide-N decode shapes, where splitting K only
  multiplies work on a grid that already fills the machine.

`dense_blockscaled_gemm_sm100_common.py`
: `_Sm100BlockScaledGemmCommon` — the base class both of them derive from. It
  owns the TMA descriptor construction, the shared-memory stage computation and
  the `wrapper` entry point that takes dynamic problem shapes.

Between them the three files import nothing but `cutlass`, `cutlass.cute`,
`cutlass.pipeline`, `cutlass.utils`, `cuda.bindings.driver` and each other, so
they form a complete unit with no dependency on FlashInfer itself.

The wheel also ships `dense_blockscaled_gemm_sm103.py`. It is **not** vendored,
and it is not the right file for a B300: FlashInfer's own dispatcher
(`flashinfer/gemm/gemm_base.py`) instantiates the Sm103 class only at
`sm_version == 107` and leaves it disabled on sm_100/sm_103, where it uses
exactly the two files vendored here.

## What was trimmed, exactly

The files as shipped contain no autotuner, no runner and no torch-op
registration — those live in other FlashInfer modules (`gemm_base.py`,
`gemm_mm_fp4_cute_dsl.py`) which are not vendored and not imported. What
remained to trim is the FP4/NVFP4 dtype plumbing, which fish-scales-ops has no
use for because its only block-scaled datacenter format is MXFP8 1×32
(`Float8E4M3FN` operands, `Float8E8M0FNU` scales, `sf_vec_size = 32`):

1. `dense_blockscaled_gemm_sm100_common.py`, in `_Sm100BlockScaledGemmCommon.wrapper`:
   the branch that recognised `Uint8` operands as two FP4 values packed per
   byte and recast them to `Float4E2M1FN` with a doubled K extent. The MXFP8
   branch and the mixed-dtype `TypeError` are kept, so an operand pair that is
   not a matching pair of FP8 tensors still fails loudly instead of silently
   taking a path that is no longer there. The corresponding two lines of the
   argument docstring that described the FP4 encoding were dropped with it.
2. `dense_blockscaled_gemm_sm100.py`, in
   `Sm100BlockScaledPersistentDenseGemmKernel.is_valid_dtypes_and_scale_factor_vec_size`:
   `cutlass.Float4E2M1FN` removed from the accepted `ab_dtype` set.
3. `dense_blockscaled_gemm_sm100.py`, in
   `Sm100BlockScaledPersistentDenseGemmKernel.is_valid_layouts`: the
   `ab_dtype is cutlass.Float4E2M1FN` clause, which after (2) can never be
   true, removed. The method is kept because `can_implement` calls it.

Every trim site carries an `fso-trim:` comment naming what was removed, so the
edits are greppable. Nothing else was changed: no class, method or attribute
was renamed, no behaviour on the MXFP8 path was touched, and the file headers
are byte-identical to the wheel's.

One sentence that was deliberately NOT edited: the docstring of
`_Sm100BlockScaledGemmCommon.wrapper` still opens with \"Uses TVM-FFI for
efficient tensor passing\". That describes how FlashInfer calls the entry
point, not a requirement of it. fish-scales-ops calls the same entry through
the plain CuTe convention instead (operands wrapped with `from_dlpack`, a real
`CUstream` argument, `--opt-level 2` with no `--enable-tvm-ffi`), which is why
`apache-tvm-ffi` is not a dependency here; the kernel body cannot tell the
difference, and the two conventions were measured to launch the same device
kernels with the same per-op time. The sentence is left alone so the vendored
text stays as close to the wheel as possible.

Note that after trim (2) the `sf_vec_size = 16` and `sf_dtype = Float8E4M3FN`
cases in the same method are already unreachable — they are only legal
together with FP4 operands — so they were left in place rather than unwound.

## Keeping this in step with upstream

There is no build step and no patch queue: the files are the unit of vendoring.
To refresh them, extract the same three paths from a newer `flashinfer-python`
wheel, re-apply the three trims above (they are small and each is one
contiguous hunk), update the version, date and md5 table here and in
`MANIFEST.md5`, and re-run
`tests/gemm/unit/test_mxfp8_decode_sm100.py` plus the decode band of
`bench/gemm/python/bench_qwen3_4b_mlp.py` on an sm_100/sm_103 device.
