# Changelog

Performance figures are in `docs/perf/`, not here.

## 0.2.2 — 2026-10-15

A patch release: one fix, no API change, no new kernel.

### Fixed

- **sm_120/sm_121: wrong (sometimes non-finite) rows from the MoE layer and the dense GEMMs under same-context
  concurrency (#11).** Affected were sm_120 callers whose fish-scales-ops GEMMs ran while kernels of another stream
  of the same CUDA context shared the SMs, for example auxiliary kernels on a high-priority side stream next to a
  CUDA-graph forward; a second process never triggered it, and neither did a GEMM running alone. The sm_120
  block-scaled GEMM kernel released each TMA-filled shared-memory stage right after issuing its `ldmatrix` /
  `ld.shared` reads of it, with no `fence.proxy.async` between the reads and the release, so the producer's next TMA
  write could overwrite the stage before a delayed read had completed. When CTAs of another kernel on the same SM
  delayed those reads, some 8x8 fragments of the weight operand were taken from the next k-tile, which made 8- to
  96-column windows of one expert's rows wrong in FC2 (non-finite when the wrong bytes met a large scale) and, through
  a wrong SwiGLU block in FC1, a small error over whole rows. The kernel now executes `fence.proxy.async.shared::cta`
  before every stage release. The dense FP8 and MXFP8 GEMMs on sm_120 share the kernel and are fixed by the same
  change. Outputs are unchanged bit for bit. The issue's script, unchanged, went from about 1 wrong replay in 3,500 to
  0 in 332,148 with its high-priority interferer and 0 in 130,640 without.

### Not affected

sm_90 (H200) and sm_100/sm_103 (B300) are not affected, and their kernels are unchanged. The race needs a
generic-proxy read of a TMA-filled operand stage that the stage's release does not wait for; on sm_120 that read is
the `ldmatrix` of the A/B tiles, because `mma.sync` takes its operands from registers. On sm_90 the WGMMA reads the
A/B tiles from shared memory itself, through descriptors, in the async proxy that the TMA writes in, and the stage
is released only after `warpgroup_wait<0>()`, that is, after the WGMMA has finished reading it; on sm_100/sm_103 the
tcgen05 UMMA reads shared memory and the stage is released by its commit. So the race fixed here cannot occur on
their operand paths. The only generic-proxy reads of a TMA-filled stage on sm_90 are the per-stage scale factors, a
few floats read with `ld.shared` before the WGMMA batch is issued, whose release is not formally ordered after the
load (no proxy fence; as in upstream DeepGEMM) but follows a whole WGMMA batch; a stress test of the 0.2.1 kernels
on an H200 under the issue's conditions found no wrong output in 4,205,661 compared replays, and a per-stage proxy
fence there cost the H200 dense tables 2–3 % at the median and up to 9 % on the narrow-N decode GEMMs, so the sm_90
kernels are left as they are (README TODO).

### Known issues

- **RTX 5090: the fence costs a fixed amount per pipeline stage, so the shortest GEMMs pay the most.** Against
  0.2.1, the narrow-N decode projections (wo, gdn.out_proj, shared_gate_up, shared_down) are up to 5.6 % slower,
  every other dense and MoE cell stays within 2.3 %, and the table medians move by 0.03 to 0.62 %. The per-cell
  numbers are in `docs/perf/`; winning the cost back elsewhere is in the README TODO.

The H200 and B300 tables are 0.2.1's: the H200 runs the same sm_90 machine code, and the B300's sm_100/sm_103
kernels are byte-identical to 0.2.1's.

### Install

As 0.2.1: install the wheel attached to the GitHub release with `pip install --no-deps` into an environment with
torch 2.13.0+cu130. After upgrading, a caller that serialized its auxiliary streams with the forward because of #11
can remove that serialization.

## 0.2.1 — 2026-10-13

### Changed (no action needed; reservations get smaller)

- **The Blackwell MoE layer's intermediates are sized by the routed rows (#7, #10).** On sm_100/sm_103 and
  sm_120/sm_121 `fso.moe.layer` no longer gives every local expert `align(tokens, 4)` rows of every intermediate (the
  masked slab). Each expert now holds its routed rows in whole 128-row blocks of one compact buffer, so a call needs
  memory for `tokens * topk` rows plus at most 127 rows per expert that holds one. `fso.moe.transient_bytes` returns
  the new, much smaller figure (Family C at 8192 tokens: about 0.46 GiB per call on a B300 instead of 13.2 GiB, and
  about 0.39 GiB per RTX 5090 rank with 128 local experts instead of 6.6 GiB, with the shared allocation below).
  The layout is the same at every number of tokens, the layer's outputs are bit-identical to 0.2.0's, and
  `FSO_MOE_COMPACT=0` restores the masked slab as an A/B arm.
- **One allocation for the gathered activation and the down-projection output, on every architecture.** The two
  buffers are never alive at the same time within a call, so the layer now places them in one region, and
  `fso.moe.transient_bytes` charges it once: about 30 % less per call (H200 Family C at 8192 tokens: 0.55 → 0.39
  GiB). A chain of layer calls captured into one graph pool no longer keeps an extra copy of the largest
  intermediate through allocator splitting. Outputs are unchanged.
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
  `mxfp8_grouped_problem_shapes_consumed`, `mxfp8_grouped_swiglu_fused_route`) take two new keywords:
  `compact=False`, which a caller composing the compact ops sets to `True` (without it they answer for the masked
  ops as in 0.2.0), and `expected_m=0`, the expected rows per expert `ceil(M * topk / G)` that the compact rule
  bounds (passed as the compact ops receive it, the queries return the route those ops take). On sm_90 the
  expert-sorted gather-quantize and the grouped FC2 GEMMs (`quantize_1x128_sorted_gather_sm90`,
  `linear_fp8_grouped_contiguous[_swapab]`) take optional output tensors, so a caller can place them in memory of its
  own. The contracts are in `docs/api/compat.md`.

### Faster

- **B300 (sm_100/sm_103) MoE layer and block.** The compact layer's grouped GEMMs take the slot-bound route while
  an expert holds at most 64 routed rows on average (`ceil(tokens * topk / experts) <= 64`: up to 2048 tokens per
  call on Qwen3.5-35B-A3B, 1024 on Qwen3-30B-A3B), which hands its tiles out dynamically over the experts that hold
  rows and writes only the routed rows. Both families are faster from 16 tokens per call up to that bound, under
  uniform, skewed and padded routing alike, and the layer no longer runs into the card's power cap there. fso is now level with or faster than
  TensorRT-LLM's trtllm-gen block-FP8 MoE at every batch size of the tables.
- **B300 (sm_100/sm_103) dense linear.** The activation quantizer reads and writes 16 bytes per thread, and the
  quantizer and the dense GEMMs launch with PDL; small-batch tile and split-K choices were corrected, and the
  2560-wide projections (`wo`, `down`) run the 2-SM 256×64 tile up to 256 tokens instead of a 40-CTA cuBLAS grid.
  The Family A MLP block is faster at every batch size, and level with or faster than sglang's block-FP8 (DeepGEMM)
  linear at every batch size of the tables except M = 96 (see Known issues).
- **RTX 5090 (sm_120/sm_121).** The dense down projection picks its Stream-K split by the rounds each slice runs
  (the MLP block is faster from 256 to 768 tokens), the MoE gather-quantize quantizes each token once and writes its
  top-k copies, and the Qwen3.5-35B-A3B experts take a better grouped tile around 1024 tokens; the MoE layer is
  2–5 % faster from 1024 tokens up.
- **H200 (sm_90) dense linear.** The block-FP8 GEMM also runs DeepGEMM v2's dense kernel (`sm90_fp8_gemm_1d2d`,
  vendored and compiled by the bundled NVRTC) at DeepGEMM's own configuration, for the tiles fso's kernel cannot run
  (256×128 with multicast, 128×160 and 128×192 from 512 tokens), for batches of 129–255 tokens, where fso's 128-row
  tiles left a second, mostly empty block, and for the long-K projections at 64 and 128 tokens. The choice depends
  on the shape only, and the outputs are bitwise identical either way. The Family A MLP block is about a third faster
  at 160 and 192 tokens, about 4 % faster at 64 and 128 tokens and 3–11 % faster from 1024 tokens up;
  `FSO_SM90_DENSE_DG2=0` keeps every dense GEMM on fso's own kernel.
- **H200 (sm_90) MoE layer and block.** The grouped GEMMs and the expert-sorted gather-quantize now launch with PDL,
  so each starts while its predecessor finishes. The swap-AB grouped GEMMs of the decode routes (the 16- and 32-row
  activation tiles) load the expert weights evict-first in L2, so the weights they stream once no longer push the
  layer's own intermediates out of L2 between the kernel that writes them and the one that reads them. The layer is
  2–9 % faster at decode (the most at the smallest batches), and level or faster in prefill under uniform and
  skewed routing.

### Known issues

The gaps below are measured against the serving libraries on the same card; the summary under each table of
`docs/perf/layer/` gives the batch sizes and the size of every gap.

- **H200, dense MLP block.** sglang 0.5.20's block-FP8 linear layer (DeepGEMM `sm90_fp8_gemm_1d2d`, with the
  programmatic dependent launch sglang serves it with) is faster than fso's block on the Family A MLP block at
  M = 96, 160 and 192 by 6–12 % and at M = 64, 128, 256, 1024 and 8192 by 1–3 %. At 160 and 192 both libraries run
  the same DeepGEMM kernels. fso's activation quantizer lets the next GEMM start early, so the GEMM's thread blocks
  land on other SMs, and the 64-row down-projection tiles then read through other L2 partitions, which costs them
  about 3 µs at M = 160.
- **H200, MoE layer under one-EP-rank routing in prefill, against 0.2.0.** With E / 2 local experts the layer is
  2.4 % slower than 0.2.0 on Family B at M = 2048 and 1.9 % slower on Family C at M = 8192; at the other prefill
  sizes, and under uniform and skewed routing, it is level with 0.2.0 or faster. About 1.3 % of the Family C cell is
  the cost of the shared allocation that takes 30 % off the call's transient memory.
- **RTX 5090, Family C MoE layer at M = 64.** vLLM 0.29.0's Triton FP8 MoE layer is 1.8 % faster than fso at M = 64,
  under uniform and under skewed routing; under skewed routing sglang 0.5.20's is also 1.6 % faster at M = 1024.
- **RTX 5090, Family C MoE at M = 1 and 2, against 0.2.0.** The routed layer at M = 2 is 1.2 % slower than in
  0.2.0; at every other M it is within 1 % of 0.2.0 or faster. With the shared expert computed after the routed layer
  and added to its output, the block at M = 1 is 2.0 % slower than in 0.2.0: the shared expert's first GEMM, whose
  machine code is unchanged, runs slower right after the compact routed chain, and the routing kernel's start is no
  longer hidden behind the preceding add. Passing the shared expert's output to `fso.moe.layer` as `bias` instead of
  adding it afterwards brings the block within 1 % of 0.2.0 in the same form, and is faster than the add form in
  either release.
- **Blackwell, dense MLP block at one batch size each.** On the B300, sglang's block-FP8 linear (DeepGEMM) is 1.6 %
  faster than fso's MXFP8 block at M = 96, where fso's 2560-wide down projection runs on 80 thread blocks and
  DeepGEMM's on 120; on the RTX 5090, vLLM's block-FP8 linear (DeepGEMM, run as eager ops) is 1.2 % faster at
  M = 1024.
- **Blackwell, the direct block-FP8 GEMM.** On sm_100/sm_103 and sm_120/sm_121, `fso.dense` serves a block-FP8
  checkpoint through the MXFP8 path, the MXFP8 rows of the tables. The direct block-FP8 GEMM that `fso.compat`
  keeps for callers who pass 1×128 / 128×128 operands themselves (`linear_fp8`, the BSFP8 rows) is slower than the
  comparators' block-FP8 linear: on the B300 sglang's is faster at every M from 16 to 8192, by 16–21 % from M = 32
  to M = 192; on the RTX 5090 vLLM's is faster from M = 512 up, by 5–14 %.

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
