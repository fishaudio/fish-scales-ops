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
  no second reduction kernel. Used for narrow-N decode shapes, and (P5, below)
  for the narrow-N long-K cells above the decode band.

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

## What was added: the fso dense epilogue (issue #12, S1b1, 2026-10-29)

fish-scales-ops' interface v1 gives every dense GEMM an optional epilogue,
`y = RN_bf16(act(acc + b) + r)` with one rounding (`b` the bias with +0.0
replaced by -0.0, `act` the identity or erf-GELU in fp32, `r` the residual),
and the decode row has to apply it inside these kernels, because as shipped
they cast to bf16 first (`epilogue_op((alpha * acc).to(c_dtype))`) and a
second kernel on the bf16 result would round twice. The additions keep the
shipped behaviour byte for byte: a kernel built without the new constructor
argument generates exactly the MLIR, PTX and cubin it generated before (checked
on every tactic of the decode row, run `fso_issue12_20261028/runs/S1b1`,
`identity_compare.json`), and the compiled entry of such a kernel is the
unchanged `wrapper`. Every edit site carries an `fso-epilogue` comment, so the
additions are greppable like the trims.

1. `dense_blockscaled_gemm_sm100_common.py`: module-level device helpers
   `_fso_add_rn`, `_fso_mul_rn` (fp32 add / multiply through
   `cute.math.add/mul(..., rounding=NEAREST_EVEN)`, which lower to the NVVM
   `add.rn.f32` / `mul.rn.f32` intrinsics that ptxas never contracts into an
   fma), `_fso_select`, `_fso_bias_term` (the -0.0 rule), `_fso_clamp_index`,
   `_fso_fma_rn`, `_fso_erff` (the CUDA 13.0 toolkit's `erff` written out
   operation for operation, with the coefficients nvcc 13.0 emits — the DSL's
   own `cute.math.erf` is another libdevice revision and differs in the last
   fp32 bit on a few per cent of inputs, which would break the byte equality
   with the C++ routes), `_fso_gelu_erf` (`0.5f * v * (1.f + erff(v *
   0.70710678f))` in the operation order of fso's `epilogue_device.cuh`),
   `fso_epilogue_supported()`; on
   `_Sm100BlockScaledGemmCommon` the attributes `fso_epi*`, `_fso_set_epilogue`
   and the methods `_fso_epi_partition_mma` (the coordinate tensor of C,
   partitioned like `tCgC`), `_fso_epi_setup`, `_fso_epi_tile`,
   `_fso_epi_apply` (per subtile: bias / residual read at the clamped
   coordinate of each accumulator element with plain coherent loads, then the
   fp32 formula, then the one bf16 conversion); the abstract `__call__` gains
   `fso_bias` / `fso_res`; and a second compiled entry `wrapper_epilogue`,
   which is `wrapper` with the bias and residual pointers after alpha (the
   bias seen as an `(m, n, l)` tensor with stride 0 along n, the residual with
   C's layout).
2. `dense_blockscaled_gemm_sm100.py` and `dense_blockscaled_gemm_sm100_splitk.py`:
   the constructor argument `fso_epilogue=None` (`(has_bias, act, has_residual)`
   selects the variant); `__call__` and `kernel` take `fso_bias` / `fso_res`
   (None in the identity); in the kernel prologue the coordinate partition
   (behind `const_expr(self.fso_epi)`), in the epilogue warps the per-CTA
   setup and the per-tile slice, and at the one place each kernel converts the
   accumulator (the persistent kernel's convert step; the split-K kernel's
   owner branch, after the in-cluster fp32 sum, so a term is never added per
   slice) the fused formula in place of the alpha-multiply-and-cast. The bias
   and residual are read by the epilogue warps only, after the kernel's
   `griddepcontrol_wait()` (which precedes the role split), so a producer that
   writes them under programmatic dependent launch is read correctly.

## What was changed: the split-K kernel's wide token tiles (issue #12, P5, 2026-11-02)

The swap-AB row of fish-scales-ops also takes the narrow-N long-K cells above
M = 64 (N ≤ 4096 with K ≥ 4096: on those every other sm_100 kernel runs 10-64
CTAs, each over the whole K), and it runs the split-K kernel there with 64-wide
token tiles. That needed four edits, all in
`dense_blockscaled_gemm_sm100_splitk.py`, each site marked `fso-p5`:

1. Token tiles (128, 64) and (128, 128) in `SUPPORTED_MMA_TILER_MN`, which may
   repeat along the token dimension (several token tiles per problem on grid Y;
   the scheduler, the grid and the scale-factor path already index them).
   `supports_tile(m, tile_n)` and an optional `tile_n` argument of
   `is_valid_tactic` say which tile serves which M: a narrow tile (8/16/32)
   only a problem it covers in one tile, because the scale-factor slice of a
   second narrow tile does not exist and such a launch computes wrong output
   (measured in the scratch fork, run `fso_issue12_20261028/runs/P5g1`).
   `MAX_M`, `supports_m` and `mma_tiler_mn_for_m` keep their meaning (the
   narrow tiles).
2. A compact reduction mailbox: a CTA only receives the subtiles it owns (one
   in `split_k_slices`, round-robin), so the mailbox holds that many slots,
   indexed by the owned subtile's ordinal, instead of the whole tile. On the
   one-subtile tiles (token tile ≤ 32) the size, and so the stage count, is the
   shipped one; on a 128-token four-way split the shipped reserve left no
   shared memory for a single mainloop stage.
3. A chunk-interleaved mailbox layout: 16-byte chunk c of epilogue thread t at
   `c * (threads * 4) + t * 4` floats, so a warp's chunk c is 512 contiguous
   bytes. The shipped layout (thread t's values contiguous) put a warp's 32
   lanes 128 bytes apart, i.e. in one bank group, and every remote `st.async`
   and every owner `LDS.128` was a 16-32-way bank conflict (ncu, run P5g1:
   2048 wavefronts per `LDS.128` against 256 ideal).
4. A push-first epilogue walk: every CTA first pushes all the subtiles it does
   not own, then waits for, sums, finishes and stores the ones it owns, so that
   no owner's wait sits behind its own previous add and store. The staging
   buffer of an owned subtile follows its ordinal among the CTA's owned
   subtiles, so consecutive stores of one CTA alternate buffers for any split.
   The walk does not implement the overlapping accumulator's early release, and
   `_setup_attributes` refuses a configuration that would need it (none of the
   supported token tiles does).

The sum is unchanged — the owner's own slice, then the peers in rank order, in
fp32, before the one conversion — so a tactic's output bytes are those of the
shipped kernel at the same token tile and split (checked on 67 cells against
the scratch fork's shipped walk and layout, and on the decode band against the
previous fish-scales-ops build, run `fso_issue12_20261028/runs/P5int`).

## What was added: weights first (issue #12, FB1, 2026-11-05)

Under programmatic dependent launch the GEMM of a dense call is created while
the kernel in front of it still runs (the activation quantize of a bf16-x
call, which releases it early), but as shipped both kernels execute
`griddepcontrol_wait()` in every warp before the role split, so the TMA warp
issues its first weight copy only after that kernel has finished. The weights
are not written by the kernel in front, so both kernels gained a variant that
fetches them first. Every edit site carries an `fso-pdl` comment:

1. `dense_blockscaled_gemm_sm100_common.py`: the attributes `fso_wf` and
   `fso_wf_cap`, their setter `_fso_set_weights_first`, and `_fso_wf_stages()`
   (the number of stages filled before the wait: at most the cap, which fso's
   decode row sets to 2 on a one-CTA tile and to 1 on the two-CTA tile, and never
   all of them, so that no try-acquire ever sits on a stage the consumer cannot
   free before the activation arrives).
2. `dense_blockscaled_gemm_sm100.py` and `dense_blockscaled_gemm_sm100_splitk.py`:
   the constructor arguments `fso_weights_first=False` and `fso_wf_cap=None`; with
   the first, every warp but
   the TMA warp waits where it waited before (so the bias, the residual and
   the output are touched only after a wait, as before), and the TMA warp
   handles the CTA's first tile in a block peeled out of its tile loop:
   acquire up to `_fso_wf_stages()` stages and issue only the weight (A, swap-AB)
   and weight-scale (SFA) copies; `griddepcontrol_wait()`; the activation (B)
   and activation-scale (SFB) copies of the same stages, against the same
   barriers; then the tile's remaining k-blocks as the loop issues them. Each
   acquire arms the stage's full barrier with the whole stage's bytes, so the
   MMA warp reads a stage only once its activation has landed too. The tile
   loop itself is unchanged and serves the CTA's later tiles.

A kernel built without the argument (`fso_weights_first=False`, the default)
generates exactly the code it generated before; `_sm100_decode` builds the
weights-first variant whenever PDL is on and today's kernel when
`FSO_DISABLE_PDL=1`. The loads are only reordered, so the output bytes are the
same either way (test_dense_v1, cases q and x).

`TRIM.diff` is the diff of the three files against the wheel's (the trims
above and these additions together); the first column of `MANIFEST.md5` is
the md5 of the files as they are here.

## Keeping this in step with upstream

There is no build step and no patch queue: the files are the unit of vendoring.
To refresh them, extract the same three paths from a newer `flashinfer-python`
wheel, re-apply the three trims, the fso epilogue additions, the split-K
changes and the weights-first variant above (`TRIM.diff` holds all of them; the
trims are one contiguous hunk each, the additions are the `fso-epilogue`,
`fso-p5` and `fso-pdl` sites), update the version, date and md5 table
here and in `MANIFEST.md5`, regenerate `TRIM.diff`, and re-run
`tests/gemm/unit/test_mxfp8_decode_sm100.py`, case q of
`tests/gemm/unit/test_dense_v1.py` and the decode band of
`bench/gemm/python/bench_qwen3_4b_mlp.py` on an sm_100/sm_103 device.
