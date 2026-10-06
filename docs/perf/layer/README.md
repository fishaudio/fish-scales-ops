# Layer-level performance — definitions

The layer pages publish what a serving loop pays per transformer layer for the
MLP or MoE block of each shape family, timed as one CUDA-graph replay that
contains every kernel of the block. Attention, norms and the router's gate GEMM
are not part of the block. Method, shape families and the cross-machine
summary: [`../README.md`](../README.md).

| page | device |
|---|---|
| [`sm90.md`](sm90.md) | NVIDIA H200 (sm_90) |
| [`sm100.md`](sm100.md) | NVIDIA B300 (sm_103) |
| [`sm120.md`](sm120.md) | NVIDIA RTX 5090 (sm_120) |

## What each block contains

| family | block | operations in the timed graph | fso precision |
|---|---|---|---|
| A — Qwen3-4B | dense SwiGLU MLP: `gate_up` (N=19456, K=2560), `down` (N=2560, K=9728) | activation quantize → `gate_up` GEMM → SiLU·mul and quantize → `down` GEMM | BSFP8 on every card; on sm_100 / sm_103 and sm_120 also MXFP8, the path `fso.dense` serves a block-FP8 checkpoint through there |
| B — Qwen3-30B-A3B | routed MoE layer: 128 experts, top-8, per expert `gate_up` (N=1536, K=2048) and `down` (N=2048, K=768); no shared expert | routing → activation gather and quantize → grouped `gate_up` → SwiGLU and quantize → grouped `down` → weighted combine | BSFP8 on sm_90; MXFP8 on sm_100 / sm_103 and sm_120 |
| C — Qwen3.5-35B-A3B | routed MoE layer (256 experts, top-8, per expert `gate_up` (N=1024, K=2048) and `down` (N=2048, K=512)) plus the shared expert, a dense SwiGLU MLP on every token (`gate_up` N=1024, K=2048; `down` N=2048, K=512), and the residual add of the two | Family B's operations, the shared expert's quantize → GEMM → SiLU·mul and quantize → GEMM, and one add | as Family B |

Family C is published twice: the routed layer alone (`fso routed`) and the
whole block (`fso routed + shared`); `shared expert µs` is their difference.
Comparators that implement only the routed layer are set against `fso routed`.

## Columns

| column | meaning |
|---|---|
| `µs` | median CUDA-graph replay time of the whole block at M tokens, with cold weights (`../README.md`) |
| `×fso` | a comparator's µs divided by fso's µs at the same M: above 1 means fso is faster; Family A gives it against the BSFP8 block and, on sm_100 / sm_103 and sm_120, also against the MXFP8 block |
| `model ms` | block µs × layers / 1000: the block's share of one forward step over the whole model (A: 36 layers, B: 48, C: 40) |
| `TFLOPS` | GEMM FLOPs of the block / µs × 1e-6, with FLOPs A `2·M·(19456·2560 + 2560·9728)`; B `2·M·8·(1536·2048 + 2048·768)`; C `2·M·8·(1024·2048 + 2048·512)`, plus `2·M·(1024·2048 + 2048·512)` when the shared expert is included. Quantize, activation, routing and combine add time but no FLOPs |
| `weight GB/s` | FP8 weight bytes of the active experts (plus the shared expert's, for the Family C block) / µs |
| `active experts` | experts that receive at least one token at that M |
| `path` (sm_90), `m_cap` (sm_100 / sm_103, sm_120) | the expert layout the layer used at that M; `m_cap` is the number of rows reserved per expert |
| `cos` | cosine similarity of the block's output against the bench's unquantized reference |
| `BF16 (torch)` | Family A's BF16 reference: two `F.linear` and an eager SwiGLU. fso has no BF16 MoE path, so Families B and C have no such column |
| summary | under each table, generated with it: per comparator column, the geometric mean of ×fso over the decode rows (M 1–128) and the prefill rows (M > 128) where both have a value, and every M at which the comparator is more than 1 % faster than fso |

Comparator columns differ per machine, so a column of one page is not
comparable with a column of another page.
