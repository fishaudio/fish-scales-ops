# Changelog

Performance figures are in `docs/perf/`, not here.

## 0.2.1 — 2026-10-10

### Changed (no action needed; reservations get smaller)

- **The Blackwell MoE layer's intermediates are sized by the routed rows (#7, #10).** On sm_100/sm_103 and
  sm_120/sm_121 `fso.moe.layer` no longer gives every local expert `align(tokens, 4)` rows of every intermediate (the
  masked slab). Each expert now holds its routed rows in whole 128-row blocks of one compact buffer, so a call needs
  memory for `tokens * topk` rows plus at most 127 rows per expert that holds one. `fso.moe.transient_bytes` returns
  the new, much smaller figure (Family C at 8192 tokens: about 0.65 GiB per call on a B300 instead of 13.2 GiB, and
  about 0.54 GiB per RTX 5090 rank with 128 local experts instead of 6.6 GiB). The layout is the same at every
  number of tokens, the layer's outputs are bit-identical to 0.2.0's, and `FSO_MOE_COMPACT=0` restores the masked
  slab as an A/B arm.
- **No routing scratch, no argument-arena abort (#7).** The compact layout's routing op keeps no device state between
  calls, so the 0.2.0 routing builder's limit of 16 graph captures per host thread (after which it fell back to the
  single-CTA builder) no longer applies to `fso.moe.layer`. On sm_100/sm_103 the grouped GEMMs take their per-call
  activation pointers from that routing op and keep only weight-side arguments in their arena, two blocks per layer
  whatever the number of captures; the arena grows on demand and no longer aborts a capture.
- **PDL at every size.** Every MoE glue launch (routing, gather-quantize, SwiGLU requantize, combine, the router) now
  follows `FSO_DISABLE_PDL` at every grid size; 0.2.0 launched grids above 4096 CTAs without PDL whatever the
  setting.

### New

- **MXFP8 weights load as stored (#9).** `fso.moe.prepare_experts(w13, w2, format="mxfp8", sw13=..., sw2=...)`
  and `fso.dense.prepare_weight(w, format="mxfp8", scale=...)` take an MXFP8 checkpoint as it is written:
  float8_e4m3fn weights with uint8 UE8M0 scales, one byte per 32 elements along K (`sw13 [E, 2I, H/32]`,
  `sw2 [E, H, I/32]`, `scale [N, K/32]`), row-major, byte `b` meaning 2^(b − 127). The bytes are the same on
  sm_100/sm_103 and sm_120/sm_121. The library only packs the scale bytes into the device's layout and
  re-orders the FC1 rows for the fused FC1; it requantizes nothing, so calibrated checkpoints are served as
  written and loading needs no bf16 copy of the layer. The bf16 forms and the int32 dense scale stay as
  they were; sm_90 still refuses `format="mxfp8"`.
- **The compact layout's building blocks in `fso.compat`.** For callers that compose the MoE steps themselves:
  the routing op `moe_build_routing_compact`, the compact grouped GEMMs (`linear_mxfp8_grouped_compact`,
  `linear_mxfp8_grouped_compact_swiglu`, `linear_mxfp8_grouped_compact_combine`) and the out-style glue
  (`quantize_1x32_grouped_gather_out_fp8`, `silu_chunk_mul_quantize_1x32_grouped_out_fp8`). On sm_100/sm_103 the
  compact GEMMs have their own route rule, so the route queries (`mxfp8_grouped_slot_possible`,
  `mxfp8_grouped_problem_shapes_consumed`, `mxfp8_grouped_swiglu_fused_route`) take a new keyword `compact=False`;
  a caller composing the compact ops asks them with `compact=True`, and without it they answer for the masked ops
  as in 0.2.0. The contracts are in `docs/api/compat.md`.

### Faster

- **B300 (sm_100/sm_103) MoE layer and block.** The compact layer's grouped GEMMs take the slot-bound route up to
  512 tokens per call, which hands its tiles out dynamically over the experts that hold rows and writes only the
  routed rows. Both families are faster from 16 to 512 tokens per call, under uniform, skewed and padded routing
  alike, and the layer no longer runs into the card's power cap there. fso is now level with or faster than
  TensorRT-LLM's trtllm-gen block-FP8 MoE at every batch size of the tables.
- **H200 (sm_90) MoE layer and block.** The grouped GEMMs and the expert-sorted gather-quantize now launch with PDL,
  so each starts while its predecessor finishes; the layer is faster at decode and in prefill.

### Known issues

The gaps below are measured against the serving libraries on the same card; the summary under each table of
`docs/perf/layer/` gives the batch sizes and the size of every gap.

- **H200, MoE layer.** sglang 0.5.20's Triton FP8 MoE layer, run with apex's tuned configs, is faster than fso on
  Family C at M = 16, 32 and 64.
- **H200, dense MLP block.** sglang 0.5.20's block-FP8 linear layer (DeepGEMM `sm90_fp8_gemm_1d2d`, with the
  programmatic dependent launch sglang serves it with) is faster than fso's block on the Family A MLP block from
  M = 64 to M = 256 and from M = 1024 up.
- **RTX 5090, Family C MoE layer.** vLLM 0.29.0's Triton FP8 MoE layer is faster than fso at M = 64 and 1024, and
  sglang 0.5.20's at M = 512 and 1024.
- **RTX 5090, Family C MoE block at M = 1.** With the shared expert computed after the routed layer and added to its
  output, the block at M = 1 is 1.9 % slower than in 0.2.0: the shared expert's first GEMM, whose machine code is
  unchanged, runs slower right after the compact routed chain, and the routing kernel's start is no longer hidden
  behind the preceding add. Passing the shared expert's output to `fso.moe.layer` as `bias` instead of adding it
  afterwards brings the block within 1 % of 0.2.0 in the same form, and is faster than the add form in either
  release.
- **Blackwell, dense MLP block.** On the B300, sglang's block-FP8 linear (DeepGEMM) is faster than fso's MXFP8 block
  from M = 32 to M = 256 and at M = 1024 and 2048; on the RTX 5090, vLLM's block-FP8 linear (DeepGEMM, run as eager
  ops) is faster at M = 512 and 1024.

## 0.2.0 — 2026-10-06

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

- **Faster on the H200 (sm_90).** The dense FP8 linear, the MoE layer and the MoE block with its shared
  expert are faster at decode and prefill: the MoE FC1 GEMM now computes SwiGLU and its FP8 requantize in
  one kernel, the dense GEMM and its activation quantize run as one launch chain, and the MoE routing glue
  scales better with the batch size. Outputs are unchanged bit for bit. `docs/perf/layer/sm90.md` and
  `docs/perf/gemm/sm90.md` give the tables. `FSO_FC1_FUSED=0`, `FSO_FC2_SWAP=0|1`, `FSO_SWAPAB_SPLIT=0|1`
  and `FSO_DISABLE_PDL=1` switch the new paths off or force a choice, for A/B runs only.
- **Compiled with the CUDA that torch uses.** The wheel is compiled with CUDA 13.0.3, and the sm_90 kernels
  are compiled at their first call by a bundled NVRTC 13.0.88: the same CUDA 13.0 that torch
  2.13.0+cu130 is built with, and the same NVRTC version it ships. The extension no longer links
  `libnvrtc`. `fish_scales_ops.build_info()` and `torch.ops.fish_scales_ops.jit_compiler_sm90()` report
  the toolchain, and `FSO_JIT_NVRTC_LIB` selects another NVRTC library.
- **One wheel for every serving machine.** One wheel serves sm_90, sm_100/sm_103 and sm_120 and runs
  without the source tree. `fish_scales_ops.build_info()` returns how it was built, both `describe()`
  texts name the build, and importing it under another torch than the one it was built against raises
  `ImportError`.
- **Release harness and CI.** GitHub Actions builds the wheel in a fixed container
  (`docker/build-wheel.Dockerfile`, `scripts/build_wheel.sh`) with a compiler cache, and attaches it to
  the release of each tag. The same wheel is then tested and measured on every machine
  (`scripts/ci/run_suite.py`, `bench/run_perf.py` under a per-machine environment lock), and
  `perf_report.py install` records where each table came from (`docs/harness.md`).
- **`fish_scales_ops.__version__`.**

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

The gaps below are measured against the serving libraries on the same card; the summary under each table of
`docs/perf/layer/` gives the batch sizes and the size of every gap.

- **H200, MoE layer.** sglang 0.5.20's Triton FP8 MoE layer, run with apex's tuned configs, is faster than fso at
  some decode batch sizes: on Family C from M = 16 to M = 128, on Family B at M = 4.
- **H200, dense MLP block.** sglang 0.5.20's block-FP8 linear layer (DeepGEMM `sm90_fp8_gemm_1d2d`, with the
  programmatic dependent launch sglang serves it with) is faster than fso's block on the Family A MLP block from
  M = 64 to M = 256 and from M = 1024 up.
- **B300, MoE layer at middle batch sizes.** TensorRT-LLM's trtllm-gen block-FP8 routed MoE (through FlashInfer
  0.6.18) is faster than fso on Family B at M = 32 and 64 and on Family C from M = 32 to M = 256; sglang's Triton FP8
  layer is also faster on Family C at M = 128.
- **RTX 5090, Family C MoE layer.** vLLM 0.29.0's and sglang 0.5.20's Triton FP8 MoE layers and TensorRT-LLM's CUTLASS
  FP8 MoE are faster than fso at M = 1024; vLLM's is also faster at M = 64.
- **Blackwell, dense MLP block.** On the B300, sglang's block-FP8 linear (DeepGEMM) is faster than fso's MXFP8 block
  from M = 32 to M = 256 and at M = 1024 and 2048; on the RTX 5090, vLLM's block-FP8 linear (DeepGEMM, run as eager
  ops) is faster at M = 512 and 1024.

### Install

Install the wheel attached to the GitHub release with `pip install --no-deps`, into an environment with
torch 2.13.0+cu130 (`fish_scales_ops.build_info()["torch"]`). An in-place build of the source tree
(`scripts/build.sh`) keeps reading files from that tree, so keep it where it is: the sm_90 JIT headers
(`FSO_JIT_INCLUDE_DIRS` overrides their location), the bundled NVRTC 13.0.88 that `scripts/build.sh`
downloads for an sm_90 build (`FSO_NVRTC_WHEEL` for an offline build), and the sm_100/sm_103 CuTe-DSL
kernel file (`FSO_DSL_KERNEL_PATH`). The runtime needs torch 2.11 or newer with a CUDA 13 runtime.
