# Changelog

Performance figures are in `docs/perf/`, not here.

## 0.2.0 — 2026-10-05

First stable release. The supported interface is `fso.dense`, `fso.moe` and `fso.attention`. The same
calls work on sm_90 (H200), sm_100/sm_103 (B200/B300) and sm_120/sm_121 (RTX 5090), and the architecture
and shape dispatch happen inside the library. `fso.compat` keeps every explicit building block under its
existing name.

### Interface changes (action needed)

- **New dense entry: `fso.dense`.**
  - `prepare_weight(w, *, format, scale=None)` is called once at load time; `format` names what the
    caller holds: `"bsfp8"` or `"mxfp8"`.
  - `linear(x, weight)` takes bf16 and returns bf16, and quantizes the activation inside. It is one torch
    custom op, `fish_scales_ops::dense_linear`, which ships its own fake implementation for torch.compile
    and dispatches by architecture inside.
  - On Blackwell a block-FP8 checkpoint is dequantized and requantized to MXFP8, as in `fso.moe`. An
    MXFP8 checkpoint's scales are accepted in either memory layout: the K-major one
    `quantize_1x32_fp8` returns, and the row-major one a safetensors round trip produces.
  - `supported` and `describe` complete it.
- **New compatibility namespace: `fso.compat`.**
  - It holds the 11 explicit dense ops (formerly `fso.gemm`) and the 34 MoE per-step pieces under their
    existing names.
  - `fso.gemm.<name>` still resolves to the same objects and raises a `DeprecationWarning` once per name.
    0.3.0 removes `fso.gemm`.
  - `torch.ops.fish_scales_ops.*` keeps every existing op and schema.
- **One call per bucket.** The MoE layer always runs a bucket as one call; chunked execution and its
  `on_output` callback are gone. Use `fso.moe.transient_bytes` to size the reservation.
- **`force_kernel` removed.** `fso.attention.flash_attn_fwd` no longer takes it.
- **sm_120 attention entries exported.** The native sm_120/sm_121 MXFP8 attention entries (prefill,
  paged prefill, paged decode, and their planning and quantize helpers) are exported from
  `fso.attention`. On other architectures they raise `NotImplementedError`.
- **Quantizer default.** `quantize_1x128_fp8` and `quantize_128x128_fp8` default `use_ue8m0` to the form
  the device's GEMM needs: power-of-two scales on sm_100 and newer, plain FP32 on sm_90.
- **sm_90 refuses the int32 scale layout.** On sm_90 the int32 pre-packed scale layout (the Blackwell
  one) is refused instead of misread: `linear_fp8` raises `ValueError`, and the packed quantizer and the
  repack ops raise `NotImplementedError`.
- **`check` argument on the repack ops.** `repack_fp8_act_scales` and `repack_fp8_wgt_scales` gain a
  trailing `check=True` argument in their torch-op schemas. The Python wrappers keep checking.
- **JIT variables renamed.** The sm_90 JIT environment variables are now `FSO_JIT_*`; the `TRTLLM_DG_*`
  names still work until 0.3.0.
- **Boolean variables parsed one way.** Every boolean `FSO_*` variable read in Python is off when unset
  or empty and when the value starts with `0`, as the C++ side already did. `FSO_DISABLE_DSL=0` therefore
  no longer disables the tier.

### New

- **Fused SwiGLU on sm_90.** On sm_90 the MoE FC1 GEMM computes SwiGLU and the 1x128 FP8 requantize in
  its epilogue on every route, so the layer is five kernels at every M. `FSO_FC1_FUSED=0` restores the
  separate kernel for A/B runs.
- **sm_90 MoE routing retuned for the fused kernels.** Swap-AB with a 16-row tile up to 12 routed rows per
  expert, a 32-row tile up to 24, and above that the non-swap FC1 followed by the swap-AB FC2 on the same
  64-row layout (bit-identical to the non-swap FC2). `FSO_FC2_SWAP=0|1` forces the FC2 for A/B runs.
  `MOE_SWAP_BLOCK_N_CASCADE` and `moe_swap_ab_max_m` in `fso.compat` change accordingly.
- **Faster sm_90 dense GEMMs.** The sm_90 dense FP8 GEMM stores its output through shared memory, one
  TMA store per 64-column slab with the 128-byte swizzle, which is the output path of DeepGEMM's
  `sm90_fp8_gemm_1d2d`. Its tile picker widens a 128-row tile to 128 columns where that adds no wave, and
  at M = 65–128 it uses a 64-row tile twice as wide when the picked 128-row tile is 32 columns or
  narrower. On the swap-AB path, which serves the smallest batches, the weight tile drops to 64 rows when
  128-row tiles would give work to at most a quarter of the SMs. The 1x128 activation quantize gives each
  128-element group to eight lanes. The quantize and
  the GEMM are chained with programmatic dependent launch (PDL) on every shape, and `FSO_DISABLE_PDL=1`
  turns it off for A/B runs. Outputs are bitwise identical to the previous kernels.
  `docs/perf/gemm/sm90.md` and `docs/perf/layer/sm90.md` give the tables.
- **Faster sm_90 MoE prefill glue.** Above 2048 routed pairs the routing sort gives each warp private
  counters. From 32 tokens on, the sorted gather-quantize reads and quantizes each token row once instead
  of once per routed pair. The layer outputs are bitwise identical; the order of the routed pairs inside
  an expert stays unspecified, as before.
- **Split fused FC1 for the smallest sm_90 decode.** When the swap-AB fused FC1 launches few enough blocks
  (four times the block count fits on the SMs), two CTAs of a cluster share each 128-column SwiGLU block.
  The result is bitwise identical, and `FSO_SWAPAB_SPLIT=0|1` forces the choice for A/B runs.
- **Bundled sm_90 compiler.** The sm_90 kernels are compiled by a bundled NVRTC 13.2.78 that the library
  loads privately. Before, the compiler was whichever NVRTC torch had loaded: 13.0 in the torch 2.13 cu130
  wheel, which generates slower code for the dense FP8 GEMMs (`docs/perf/README.md`).
  - `scripts/build.sh` fetches the pinned wheel for an sm_90 build.
  - `FSO_JIT_NVRTC_LIB` selects another library.
  - `torch.ops.fish_scales_ops.jit_compiler_sm90()` and `describe()` report the compiler in use.
  - The extension no longer links `libnvrtc`.
- **`fish_scales_ops.__version__`.**
- **One wheel for every serving machine.** `scripts/build_wheel.sh` builds a wheel for sm_90,
  sm_100/sm_103 and sm_120 in a fixed container (`docker/build-wheel.Dockerfile`), and the wheel runs
  without the source tree.
  - It carries the sm_90 JIT headers in `_jit_include/` and the CuTe-DSL kernel in `_dsl/`; the extension
    looks for both next to itself before it looks at the source tree.
  - `fish_scales_ops.build_info()` returns its `BUILD_INFO.json` (commit, toolchain, the torch it was built
    against, NVRTC, architectures), and both `describe()` texts name the build.
  - Importing it under another torch than the one it was built against raises `ImportError`.
- **Release harness.** One wheel is built, then tested and measured as that artifact on every machine
  (`docs/harness.md`): `scripts/ci/run_suite.py` runs the whole test suite against an installed wheel,
  `bench/run_perf.py` measures the tables of record under a per-machine environment lock
  (`bench/env/<machine>.lock.json`) and refuses drift, and `perf_report.py install` records each table's
  provenance, rendered into `docs/perf/README.md`. `.github/workflows/wheel.yml` builds the wheel and
  runs the checks that need no GPU.

### Fixed

- **sm_100/sm_103 per-call device sync.** `linear_fp8` given FP32 scales and `linear_bf16` used to
  synchronize the device on every call and could not be captured into a CUDA graph. Both now pack their
  scales without a check, so they no longer synchronize and can be captured.
- **sm_100/sm_103 split-K workspace growth.** When the workspace grew, its old buffer was freed while
  graphs captured earlier still used it. Old buffers are now kept. A capture that needs a larger
  workspace raises `RuntimeError` instead of aborting.
- **sm_120/sm_121 scale scratch growth.** The same fix applies to the scratch of `linear_bf16` and
  `linear_qx`.
- **sm_90 silent register fallback.** When the two-CTA swap-AB build does not compile to its expected
  register count, the fallback to one CTA is now reported on stderr.
- **sm_90 missing JIT headers.** When the JIT include directories are missing, an error message names
  them.
- **sm_90 JIT disk cache.** The key now covers the kernel source, the flags, the compiler and the JIT
  headers, so a changed header can no longer load a stale cubin.
- **sm_120 paged decode plan.** An eager call after a CUDA-graph capture resets the split-K counter.
- **sm_100/sm_103 CuTe-DSL tiers.** They no longer turn off silently. Every failure other than the
  optional `nvidia-cutlass-dsl` package being absent prints one line.
- **`mxfp8_attn_ref` import.** `fish_scales_ops.attention.backends.mxfp8_attn_ref` imports again.

### Known issues

- **sm_90 MoE layer against sglang's Triton FP8 MoE layer at some decode batch sizes.** Run with apex's
  tuned Triton configs, sglang 0.5.20's Triton FP8 MoE layer is 1.4–4.1 % faster than fso on Family C from
  M = 16 to M = 128, and 2.3 % faster on Family B at M = 4. fso is level with it or ahead at every other
  batch size of both families. `docs/perf/layer/sm90.md` gives the comparison.
- **sm_90 dense block-FP8 MLP block at some batch sizes.** sglang 0.5.20's block-FP8 linear layer, which
  runs DeepGEMM's `sm90_fp8_gemm_1d2d` kernel (sgl-deep-gemm 0.2.0) with the programmatic dependent launch
  sglang serves it with, is faster than fso on the Family A MLP block at M = 64 to 256, by 1.6–5.6 %, and
  at M = 1024, 2048 and 8192, by 6.0–10.5 %. fso is ahead at M = 1 to 32, at M = 512 and at M = 4096.
  `docs/perf/layer/sm90.md` gives the comparison.
- **sm_100/sm_103 MoE at middle batch sizes.** TensorRT-LLM's trtllm-gen block-FP8 routed MoE (through
  FlashInfer 0.6.18) is up to 6 % faster than fso on Family B from M = 32 to M = 128, and up to 8 % faster on
  Family C from M = 16 to M = 512. On Family C, sglang 0.5.20's Triton FP8 layer with apex's tuned configs is
  also 1.5–1.8 % faster at M = 64 and 128. fso is level with trtllm-gen at Family B's M = 16 and ahead of it
  at M ≤ 8, from M = 256 up on Family B and from M = 1024 up on Family C. `docs/perf/layer/sm100.md` gives
  the comparison.
- **sm_120 Family C MoE layer at M = 1024.** On the RTX 5090, sglang 0.5.20's and vLLM 0.29.0's Triton FP8
  MoE layers are 3.6–4.3 % faster than fso's Family C routed layer at M = 1024, and TensorRT-LLM's CUTLASS
  FP8 per-tensor MoE is 1.1 % faster there; vLLM's layer is also 1.6 % faster at M = 64. fso is level or
  ahead at the other Family C batch sizes and at every Family B batch size. `docs/perf/layer/sm120.md`
  gives the comparison.
- **Blackwell dense MLP block at some batch sizes.** On the B300, sglang's block-FP8 linear (DeepGEMM) is
  faster than fso's MXFP8 block from M = 32 to M = 2048, by up to 12.5 % at M = 128; fso is ahead at M ≤ 16
  and from M = 4096 up. On the RTX 5090, vLLM's (DeepGEMM, run as eager ops) is 4 % faster at M = 512 and
  level with fso at M = 1024; fso is ahead at the other batch sizes. `docs/perf/layer/sm100.md` and
  `docs/perf/layer/sm120.md` give the comparisons.
- **sm_100/sm_103 build toolchain.** The wheel compiles the sm_100f kernels with the CUDA 13.2.1 toolkit.
  The same kernel sources built with CUDA 13.0 are up to 3 % faster on most grouped MoE GEMM cells at
  prefill on the B300; the MoE layer moves by under 2 %. How the toolchain is pinned is under review.
  `docs/perf/gemm/sm100.md` gives the measurement.

### Install

Serving machines install the wheel that `scripts/build_wheel.sh` builds (`pip install --no-deps`), into an
environment with the torch it was built against (`fish_scales_ops.build_info()["torch"]`).

An in-place build reads files from the source tree that built it, so build in place (`EDITABLE=1` or the
default in-place build) and keep the tree where it is:
- the sm_90 JIT headers; `FSO_JIT_INCLUDE_DIRS` overrides their location;
- the bundled NVRTC 13.2.78 in `python/fish_scales_ops/_nvrtc/`, which `scripts/build.sh` downloads for an
  sm_90 build (`FSO_NVRTC_WHEEL` for an offline build);
- the sm_100/sm_103 CuTe-DSL kernel file; `FSO_DSL_KERNEL_PATH` overrides its location.

The runtime needs torch 2.11 or newer with a CUDA 13 runtime.
