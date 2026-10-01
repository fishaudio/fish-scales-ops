# Changelog

Performance figures are in `docs/perf/`, not here.

## 0.2.0 — 2026-10-01

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

- **sm_90 routed + shared-expert block at M = 1.** On sm_90 the Family C block of routed experts plus the
  shared expert is slightly slower at M = 1 than it was before the fused FC1, and faster from M = 2 up.
  The cause is the fused FC1's first memory access in that context. `docs/perf/layer/sm90.md` gives the
  measurement.
- **sm_90 Family C prefill.** On Family C prefill, sglang 0.5.20's Triton FP8 MoE layer is ahead of fso
  from M = 1024 to M = 4096. `docs/perf/layer/sm90.md` gives the comparison.

- **sm_90 kernels compiled by an older NVRTC.** The sm_90 kernels are compiled at run time by the NVRTC
  that torch loads. The torch 2.13 cu130 wheel bundles NVRTC 13.0, which generates slower code than 13.2
  for the dense FP8 GEMMs; the MoE layer is barely affected. Setting `FSO_JIT_USE_NVCC=1` with a CUDA 13.2
  nvcc (`FSO_JIT_NVCC_COMPILER`) gives 13.2's code. `docs/perf/README.md` gives the measurement.

### Install

The extension reads files from the source tree that built it, so build in place (`EDITABLE=1` or the
default in-place build) and keep the tree where it is:
- the sm_90 JIT headers; `FSO_JIT_INCLUDE_DIRS` overrides their location;
- the sm_100/sm_103 CuTe-DSL kernel file; `FSO_DSL_KERNEL_PATH` overrides its location.

The runtime needs torch 2.11 or newer with a CUDA 13 runtime.
