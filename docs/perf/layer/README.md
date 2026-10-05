# Layer-level performance — definitions

The `gemm/` files publish one GEMM at a time. This directory publishes what a
serving loop actually pays per transformer layer for the **MLP / MoE block**
of each shape family, measured as one CUDA-graph replay containing every
kernel of the block (activation quantize, GEMMs, activation function, routing
and combine for MoE, the shared-expert path and the residual add for Family
C). Attention, norms and the router gate GEMM are not part of the block.

| file | device | content |
|---|---|---|
| `sm90.md` | NVIDIA H200 (sm_90) | Family A / B / C block tables, BSFP8 |
| `sm120.md` | NVIDIA RTX 5090 (sm_120) | Family A / B / C block tables, BSFP8 + MXFP8 |
| `sm100.md` | NVIDIA B300 (sm_103) | Family A / B / C block tables, BSFP8 + MXFP8; unlocked clocks (see `../gemm/sm100.md`) |

Protocol, clock policy, M grids and the shape families are defined in
[`../README.md`](../README.md); this file only defines what each family's
"block" contains and how its numbers are derived.

## What each family's block contains

| family | block | kernels in the timed graph |
|---|---|---|
| A — Qwen3-4B | dense SwiGLU MLP: `gate_up` (N=19456, K=2560) → silu·mul → `down` (N=2560, K=9728) | act quantize → gate_up GEMM → silu·mul (+ quantize) → down GEMM. sm_90 BSFP8: `torch.compile`d silu·mul + separate quantize; sm_120 and sm_100/103 MXFP8: fused `silu_chunk_mul_quantize_1x32_fp8`, while their BSFP8 blocks pay the separate 1×128 quantize. Bench: `bench_qwen3_4b_mlp_forward.py`. |
| B — Qwen3-30B-A3B | routed MoE layer, 128 experts, top-8, moe_inter 768, no shared expert | routing → gather-quant → grouped gate_up with the SwiGLU and requantize fused into its epilogue → grouped down → combine: five kernels on every arch, routing derived on device from `topk_ids` (before that fusion a separate silu-quant kernel between the two GEMMs made six). sm_90: expert-sorted contiguous BSFP8 (`moe_layer_fp8_sm90`), whose routing kernel builds the sorted layout; sm_120: masked-slab MXFP8; sm_100/103: the same masked slab, with the routing kernel also supplying the grouped GEMMs' arguments (eight kernels, with one argument-preparation kernel ahead of each grouped GEMM, when the path landed on 2026-09-15). Bench: `bench_moe_qwen3_30a3.py --impls fso_*_layer`. |
| C — Qwen3.5-35B-A3B | routed MoE layer, 256 experts, top-8, moe_inter 512, **plus the shared expert** (dense SwiGLU MLP, intermediate 512, every token) and the residual add of the two outputs | the routed kernels above + shared-expert act quantize → gate_up (N=1024, K=2048) → silu·mul (+ quantize) → down (N=2048, K=512) + one bf16 add. Bench: `bench_moe_qwen3_35a3.py --impls fso_*_layer_shared`; the routed-only number (`fso_*_layer`) is published alongside so the shared expert's share is visible. |

BF16 columns: Family A has a torch BF16 reference (two `F.linear` + eager
SwiGLU). Families B and C have no fso BF16 column — fso has no BF16 grouped
path — and their `cos` is measured against an FP32 per-expert reference. Where
a file carries a BF16 number for Families B or C it is a comparator, not an fso
row: on the B300 that is torch's own `_grouped_mm` in BF16, declared with the
other comparators in `sm100.md`.

## Reported quantities

| quantity | definition |
|---|---|
| `block µs` | graph-replay median of the whole block at `M` tokens |
| `model ms` | `block µs × layers / 1000` — the block's share of one forward step over the whole model (A: 36 layers, B: 48, C: 40). It is what the block costs per token step at that batch; attention and everything else come on top. |
| `TFLOPS` | `FLOPs / µs × 1e-6` with FLOPs = GEMM FLOPs of the block: A `2·M·(19456·2560 + 2560·9728)`; B `2·M·8·(1536·2048 + 2048·768)`; C routed `2·M·8·(1024·2048 + 2048·512)` plus shared `2·M·(1024·2048 + 2048·512)` when the shared expert is included. Quantize, silu, routing and combine add time but no FLOPs — on purpose: the block number is what serving pays. |
| `weight GB/s` (B, C) | active-expert FP8 weight bytes (plus shared-expert weights for C) per block / time; above the device's DRAM bandwidth means the row was measured warm (`../README.md` §8) |
| `cos` | cosine similarity vs the reference output of the whole block |

## Reading the tables

- Every row carries `weight_copies` and was measured with cold weights — the
  graph rotates ≥ 2 × L2 of weight copies, as a serving step evicts every
  layer's weights (`../README.md` §1). Rows of the single-copy protocol used
  before 2026-09-28 replayed one warm copy, and their small-M values were
  L2-resident optimism, `model ms` included; none of them is published any
  more. Each file's Provenance section lists its baseline files, and the
  generated block [Environments of record](../README.md#environments-of-record)
  names the run behind each.
- Family C rows come in two flavours: routed-only and routed + shared. The
  difference is the price of the shared expert at that M (a K=512 `down` and a
  narrow `gate_up`, both poorly amortised — see the dense rows in `gemm/`).
- The masked-slab MoE layout's transient tensors scale with `G × m_cap`, and the
  layer runs every bucket as one call, so the bench asks
  `moe_layer_transient_bytes_mxfp8` whether a grid point's tensors fit beside the
  weights instead of capping M: on the RTX 5090 the sm_120 tables run the full
  grid to M = 8192 (since 2026-09-28, once the fused FC1 dropped the gate/up
  slab), the sm_100/103 tables run to M = 8192 since the 0.2.0 release, whose
  bench extends the fit test to that arch (before it they stopped at M = 4096),
  and sm_90's contiguous layout has always run to M = 8192. Comparator columns are published at M = 8192 where they
  were measured, next to an empty fso cell if that device has none.
- Comparator columns differ per device and are not comparable across files. For
  the MoE layers, the sm_90 file carries sglang triton `fused_experts` and the
  deep_gemm masked pipeline; the sm_120 file carries the MoE implementations a
  torch or serving-stack user gets on that card — vLLM triton `fused_experts`,
  sglang triton `fused_experts`, TensorRT-LLM's CUTLASS fused MoE (JIT-built for
  sm_120 through the FlashInfer wheel), torch's own `_grouped_mm` (eager, plain
  and `torch.compile`d; it cannot be graph-captured on sm_120) — plus a
  kernel-level section that puts fso's grouped GEMM next to the Triton grouped
  GEMM those stacks run per projection; the B300 file carries TensorRT-LLM's
  trtllm-gen fused MoE, sglang triton `fused_experts` and torch's
  `scaled_grouped_mm` and `_grouped_mm`. For the Family A MLP block every file
  carries sglang's block-FP8 linear, the sm_90 file also cuBLAS's block-FP8
  `scaled_mm` (torch accepts that recipe on sm_90 only) and the sm_120 file also
  vLLM's block-FP8 linear. The library versions each comparator ran under are in
  the generated block
  [Environments of record](../README.md#environments-of-record), and each file
  states the comparator caveats (the backend each column ran, tuned or default
  tile configs, eager timings) next to its columns.
- Every B300 row is an unlocked-clock number; read it with the caveats in
  `../README.md` §8 before comparing it with anything.
