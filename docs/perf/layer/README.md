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
| `sm100.md` | NVIDIA B300 (sm_103) | n/a — pending decision (see `../gemm/sm100.md`) |

Protocol, clock locks, M grids and the shape families are defined in
[`../README.md`](../README.md); this file only defines what each family's
"block" contains and how its numbers are derived.

## What each family's block contains

| family | block | kernels in the timed graph |
|---|---|---|
| A — Qwen3-4B | dense SwiGLU MLP: `gate_up` (N=19456, K=2560) → silu·mul → `down` (N=2560, K=9728) | act quantize → gate_up GEMM → silu·mul (+ quantize) → down GEMM. sm_90 BSFP8: `torch.compile`d silu·mul + separate quantize; sm_120 MXFP8: fused `silu_chunk_mul_quantize_1x32_fp8`. Bench: `bench_qwen3_4b_mlp_forward.py`. |
| B — Qwen3-30B-A3B | routed MoE layer, 128 experts, top-8, moe_inter 768, no shared expert | routing → gather-quant → grouped gate_up → silu-quant → grouped down → combine (six kernels, routing derived on device from `topk_ids`). sm_90: expert-sorted contiguous BSFP8 (`moe_layer_fp8_sm90`); sm_120: masked-slab MXFP8. Bench: `bench_moe_qwen3_30a3.py --impls fso_*_layer`. |
| C — Qwen3.5-35B-A3B | routed MoE layer, 256 experts, top-8, moe_inter 512, **plus the shared expert** (dense SwiGLU MLP, intermediate 512, every token) and the residual add of the two outputs | the six routed kernels above + shared-expert act quantize → gate_up (N=1024, K=2048) → silu·mul (+ quantize) → down (N=2048, K=512) + one bf16 add. Bench: `bench_moe_qwen3_35a3.py --impls fso_*_layer_shared`; the routed-only number (`fso_*_layer`) is published alongside so the shared expert's share is visible. |

BF16 columns: Family A has a torch BF16 reference (two `F.linear` + eager
SwiGLU). Families B and C have none — fso has no BF16 grouped path and
third-party MoE implementations are excluded by policy (`../../README.md`
rule 2); their `cos` is measured against an FP32 per-expert reference.

## Reported quantities

| quantity | definition |
|---|---|
| `block µs` | graph-replay median of the whole block at `M` tokens |
| `model ms` | `block µs × layers / 1000` — the block's share of one forward step over the whole model (A: 36 layers, B: 48, C: 40). It is what the block costs per token step at that batch; attention and everything else come on top. |
| `TFLOPS` | `FLOPs / µs × 1e-6` with FLOPs = GEMM FLOPs of the block: A `2·M·(19456·2560 + 2560·9728)`; B `2·M·8·(1536·2048 + 2048·768)`; C routed `2·M·8·(1024·2048 + 2048·512)` plus shared `2·M·(1024·2048 + 2048·512)` when the shared expert is included. Quantize, silu, routing and combine add time but no FLOPs — on purpose: the block number is what serving pays. |
| `weight GB/s` (B, C) | active-expert FP8 weight bytes (plus shared-expert weights for C) per block / time; above the device's DRAM bandwidth means L2-fed (`../README.md` §8) |
| `cos` | cosine similarity vs the reference output of the whole block |

## Reading the tables

- Small-M rows are L2-resident numbers (the bench does not flush L2); see the
  caveat in each `gemm/` file. `model ms` inherits that optimism.
- Family C rows come in two flavours: routed-only and routed + shared. The
  difference is the price of the shared expert at that M (a K=512 `down` and a
  narrow `gate_up`, both poorly amortised — see the dense rows in `gemm/`).
- The masked-slab MoE layout on sm_120 stops at M = 4096; sm_90's contiguous
  layout runs the full grid to M = 8192.
