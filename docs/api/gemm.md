# `fish_scales_ops.gemm` — public API

Frozen op contracts for the GEMM domain. **No performance numbers in this
file**; reference numbers live in [`../perf/`](../perf/README.md) and the
acceptance rule in [`../perf/README.md`](../perf/README.md) §7. Code is the source of
truth for signatures: the Python wrappers are in
`python/fish_scales_ops/gemm/{fp8,mxfp8,bf16}.py` and, for the MoE layer surface
`fso.moe`, in `python/fish_scales_ops/moe.py`; the `torch.ops` schemas are in
`csrc/gemm/bindings.cpp`, except `fish_scales_ops::moe_layer`, which `moe.py`
registers as a `torch.library.custom_op`.

## Overview

fso implements two block-scaled FP8 GEMM formats and routes each call by the
device's compute capability at runtime (cached once per process, so the
routing itself is CUDA-graph safe):

| format | scale granularity | what it is for |
|---|---|---|
| **block-FP8** ("BSFP8", 1×128 activation / 128×128 weight) | one scale per 128 K-elements of an activation row, one per 128×128 weight block | the DeepSeek / Qwen FP8 checkpoint format; the only block-scaled path on Hopper |
| **MXFP8** (OCP, 1×32) | one UE8M0 byte per 32 K-elements on both operands | tighter quantization; hardware-native on Blackwell |

| arch | block-FP8 1×128 | MXFP8 1×32 | grouped MoE |
|---|---|---|---|
| sm_90 (H200) | deep_gemm WGMMA kernels, NVRTC-JIT-compiled in-process at first call; FP32 scales | not available (`NotImplementedError`) | block-FP8, expert-sorted contiguous layout (`moe_layer_fp8_sm90`) and a masked-layout variant |
| sm_120 / sm_121 (RTX 5090, RTX PRO 6000) | CUTLASS `Sm120BlockScaledKernel`; UE8M0 scales packed into int32 words | CUTLASS `Sm120BlockScaledKernel` | MXFP8, masked slab layout (`moe_layer_mxfp8_sm120`) |
| sm_100 / sm_103 (B200 / B300) | since 2026-09-05: the MXFP8 tcgen05 path with each 1×128 UE8M0 scale byte replicated into its four 32-wide slots (same kernels, same bytes as MXFP8) | CUTLASS tcgen05 BlockScaled behind a router: an M ≤ 64 decode row (vendored NVIDIA CuTe-DSL split-K / persistent kernels, swap-AB, 8/16/32-wide token tile to M = 32 and a 64-wide token tile on the 2-CTA 256-row weight tile above it, long-K narrow-N cells above M = 32 excepted) in front of three tiers (cuBLAS `scaled_mm`, CuTe DSL persistent kernel, C++ cascade) | MXFP8, masked slab layout — same Python surface as sm_120, CUTLASS pointer-array (grouped) tcgen05 kernels |

A caller reaches the grouped MoE layer through one surface on every
architecture, `fso.moe` (*MoE layer — `fso.moe`* below); the per-arch entries in
the last column of the table are its implementation.

Scale tensors are **arch-native and opaque**. Quantize on the device arch the
GEMM runs on; a scale tensor produced on one arch generation is not valid on
another (the layouts are listed under *Scale layouts* below).

## Quick start

```python
import torch, fish_scales_ops as fso

# ---- block-FP8 1x128 / 128x128 (sm_90, sm_120, sm_100/103) -------------------
# Weights, once at load time. On sm_120 and sm_100/103 the scales must be
# UE8M0 (powers of two); the weight quantizer defaults to that there and to
# FP32 scales on sm_90.
wq, sw = fso.gemm.quantize_128x128_fp8(w_bf16)          # fp8 [N, K], fp32 [N/128, K/128]
sm = torch.cuda.get_device_capability(0)[0]
if sm >= 10:
    sw = fso.gemm.repack_fp8_wgt_scales(sw)             # pre-pack once; int32, arch-native layout

# Activations, per call.
if sm >= 10:
    xq, sx = fso.gemm.quantize_1x128_fp8_packed(x_bf16) # fused quantize + pack, UE8M0
else:
    xq, sx = fso.gemm.quantize_1x128_fp8(x_bf16)        # fp32 scales for deep_gemm
y = fso.gemm.linear_fp8(xq, wq, sx, sw)                 # bf16 [M, N]

# ---- MXFP8 1x32 (sm_100/103, sm_120) -----------------------------------------
if sm in (10, 12):
    wqm, swm = fso.gemm.quantize_1x32_fp8(w_bf16)       # fp8 [N, K] + opaque int32 UE8M0 scales
    xqm, sxm = fso.gemm.quantize_1x32_fp8(x_bf16)
    ym = fso.gemm.linear_mxfp8(xqm, wqm, sxm, swm)      # bf16 [M, N]

# ---- MoE layer, every arch -------------------------------------------------
# Once per layer at load time, on the serving device: the rank's local experts as
# the checkpoint stores them, w13 [E, 2I, H] with the gate rows first and
# w2 [E, H, I]. "bsfp8" = fp8 weights with fp32 128x128 block scales, "mxfp8" =
# bf16 weights (quantized here). The arch dispatch is inside the op.
experts = fso.moe.prepare_experts(w13, w2, format="bsfp8", sw13=sw13, sw2=sw2)
# Per call. topk_ids (int32 or int64) may carry ids outside [0, E) for padded
# graph rows and for experts another expert-parallel rank owns; they are skipped.
out = fso.moe.layer(hidden, experts, topk_ids, topk_w)
```

`fso.moe.layer` runs the per-arch entries described under *Grouped MoE* —
`moe_layer_fp8_sm90` on sm_90, and the masked-slab MXFP8 chain behind
`moe_layer_mxfp8_sm120` on sm_120/121 and sm_100/103 — and those entries and the
per-step ops they compose stay public. `bench/gemm/python/bench_moe_qwen3_30a3.py`
(`fso_mxfp8_layer`) drives the steps directly; it is where the sm_100/103
composition that `fso.moe.layer` runs, with its fused FC1 and its slot-bound
decode route, was developed.

## Ops

### Dense

| function | in | out | notes |
|---|---|---|---|
| `quantize_1x128_fp8(x, use_ue8m0=False)` | bf16 `[..., K]`, `K % 128 == 0` | (fp8 e4m3 `[..., K]`, fp32 `[pad(M,4), K/128]` K-major) | `use_ue8m0=True` rounds each scale up to a power of two; required before feeding an sm_120 or sm_100/103 GEMM |
| `quantize_1x128_fp8_packed(x)` | bf16 `[..., K]`, `K % 128 == 0` | (fp8 `[..., K]`, int32 packed scales, arch-native) | UE8M0 always; one kernel on `K % 512 == 0`, two kernels otherwise; sm_120 and sm_100/103 |
| `quantize_128x128_fp8(w, use_ue8m0=None)` | bf16 `[N, K]` | (fp8 `[N, K]`, fp32 `[ceil(N/128), ceil(K/128)]`) | `None` → UE8M0 on sm_100/103 and sm_120, FP32 on sm_90 |
| `repack_fp8_act_scales(sx_f32)` | fp32 `[pad(M,4), K/128]` UE8M0-exact | int32 packed, arch-native | sm_120 and sm_100/103; rejects non-power-of-two scales (one device sync — a weight-load-time op) |
| `repack_fp8_wgt_scales(sw_f32)` | fp32 `[N/128, K/128]` UE8M0-exact | int32 packed, arch-native (one scale per N row) | same |
| `linear_fp8(x_fp8, w_fp8, sx, sw)` | fp8 `[M, K]`, fp8 `[N, K]`, scales fp32 (packed per call) or int32 (pre-packed) | bf16 `[M, N]` | `K % 128 == 0`; `N % 128 == 0` on sm_120 and sm_100/103; sm_90 takes FP32 scales only |
| `linear_bf16(x, w)` | bf16 `[M, K]`, bf16 `[N, K]` | bf16 `[M, N]` | block-FP8 path with internal quantization of both operands; convenience only, use when both operands change every call |
| `linear_qx(x_bf16, w_fp8, sw)` | bf16 `[M, K]`, fp8 `[N, K]`, fp32 `[N/128, K/128]` | bf16 `[M, N]` | block-FP8 path, fused activation quantize; sm_90 and sm_120; raises on sm_100/103 |
| `quantize_1x32_fp8(x)` | bf16 `[..., K]`, `K % 128 == 0` | (fp8 `[..., K]`, opaque int32 scales, arch-native) | MXFP8, UE8M0 always; sm_100/103 and sm_120 |
| `silu_chunk_mul_quantize_1x32_fp8(gu)` | bf16 `[..., 2*INTER]` (`gate ‖ up`), `INTER % 128 == 0` | (fp8 `[..., INTER]`, opaque int32 scales) | `silu(gate) * up` quantized without materialising the BF16 intermediate; sm_100/103 and sm_120 |
| `linear_mxfp8(x_fp8, w_fp8, sx, sw)` | fp8 `[M, K]`, fp8 `[N, K]`, opaque int32 scales from `quantize_1x32_fp8` on the same arch | bf16 `[M, N]` | `K % 128 == 0`, `N % 128 == 0`; sm_100/103 routes through the decode row and the three tiers, sm_120 through the CUTLASS cascade |

The same names are exported from `fso.gemm` and from the submodules
`fso.gemm.fp8`, `fso.gemm.mxfp8`, `fso.gemm.bf16`.

#### Scale layouts

The FP32 scale tensors carry PyTorch metadata that does not describe their
physical byte order; the packed int32 tensors are opaque handles. Never
`.contiguous()`, transpose or slice them, and never move them between arch
generations.

| tensor | arch | logical shape | physical layout |
|---|---|---|---|
| 1×128 activation scales, fp32 (`quantize_1x128_fp8`) | all | `[pad(M,4), K/128]` | K-major: `data[kb * pad(M,4) + m]` (deep_gemm convention) |
| 128×128 weight scales, fp32 (`quantize_128x128_fp8`) | all | `[ceil(N/128), ceil(K/128)]` | row-major |
| packed 1×128 activation scales, int32 | sm_120 | `[pad(M,4), ceil(K/512)]` | K-major words, 4 UE8M0 bytes per word (one per 128-block); the last word is zero-padded when `K/128 % 4 != 0` |
| packed 128×128 weight scales, int32 | sm_120 | `[pad(N,4), ceil(K/512)]` | one row per N, block scale replicated over the 128 rows of its block |
| packed 1×128 / 128×128 scales, int32 | sm_100/103 | 1-D `[pad(M,128) * K/128]` / `[N * K/128]` | CUTLASS `Sm1xxBlockScaledConfig<32>` atom layout; each word holds the block's UE8M0 byte in all four 32-wide slots |
| MXFP8 1×32 scales, int32 (`quantize_1x32_fp8`) | sm_120 | `[pad(M,4), K/128]` | K-major words, 4 UE8M0 bytes per word (one per 32-block) |
| MXFP8 1×32 scales, int32 | sm_100/103 | 1-D `[pad(M,128) * K/128]` | the same atom layout, one distinct byte per 32-block |
| grouped MXFP8 activation scales, int32 | sm_120 | `[G, K/128, m_cap]` | per group, K-major words: word `(m, kp)` of group `g` at `g * (K/128) * m_cap + kp * m_cap + m` |
| grouped MXFP8 weight scales, int32 | sm_120 | `[G, K/128, N]` | the same, with `N` in place of `m_cap` |
| grouped MXFP8 activation scales, int32 | sm_100/103 | `[G, pad(m_cap,128) * K/128]` | per group, one `Sm1xxBlockScaledConfig<32>` atom slab for a `(pad(m_cap,128), K)` tensor: word `(m, kp)` at `((m / 128) * (K/128) + kp) * 128 + (m % 32) * 4 + (m % 128) / 32` inside the slab |
| grouped MXFP8 weight scales, int32 | sm_100/103 | `[G, N * K/128]` | the same atom slab for an `(N, K)` tensor (`N % 128 == 0`, so no row padding) |

On sm_100/103 the block-FP8 packed layout is therefore a special case of the
MXFP8 layout, which is why the block-FP8 GEMM there is the MXFP8 GEMM.

### MoE layer — `fso.moe`

`fso.moe` is the MoE layer surface for every architecture. A caller prepares
each layer's local experts once, on the device that will serve them, and then
makes one call per batch; the architecture dispatch happens inside a torch
custom op, so the caller never names an architecture and nothing falls back.
The per-arch entries under *Grouped MoE* below — `moe_layer_fp8_sm90` and the
masked-slab MXFP8 chain behind `moe_layer_mxfp8_sm120` — are the implementation
behind it; they stay public for benches and experiments.

```python
experts = fso.moe.prepare_experts(w13, w2, format="bsfp8", sw13=sw13, sw2=sw2)       # load time
out     = fso.moe.layer(hidden, experts, topk_ids, topk_w, bias=None, bias_scale=None)  # per call
ok      = fso.moe.supported("mxfp8")   # this device, or arch="sm_90", "sm_103", 120, (12, 0)
text    = fso.moe.describe()           # the architecture matrix, for logs and error messages
```

**Preparing the experts.** `prepare_experts(w13, w2, *, format, sw13=None, sw2=None)`
takes this rank's local experts in the layout the checkpoint dialect stores:
`w13 [E_local, 2 * I_local, H]` with the gate rows first (`[gate; up]`) and
`w2 [E_local, H, I_local]`, on the CUDA device that will serve them. `format`
names what the caller holds, and the table says what each architecture does
with it.

| `format` | what the caller holds | sm_90 (H200, MIG) | sm_100 / sm_103 (B300, MIG) | sm_120 / sm_121 (RTX 5090) |
|---|---|---|---|---|
| `"bsfp8"` | float8_e4m3fn weights with fp32 128×128 block scales `sw13 [E, 2I/128, H/128]` and `sw2 [E, H/128, I/128]`, weight = fp8 value × block scale (the `bsgemm-moe` dialect) | kept as they are (validated, made contiguous if they are not); kind `"bsfp8"` | each expert dequantized block by block to bf16 and requantized to MXFP8 1×32 with the FC1 rows interleaved; kind `"mxfp8"` | same as sm_100 |
| `"mxfp8"` | bf16 weights (the `mxfp8` dialect keeps its experts bf16 in the checkpoint) | `NotImplementedError`: sm_90 has no MXFP8 hardware | quantized to MXFP8 1×32 with the FC1 rows interleaved; kind `"mxfp8"` | same as sm_100 |
| anything else (bf16 experts, int8, fp4, …) | — | `NotImplementedError` for a known dialect, `ValueError` for an unknown string; no fallback | same | same |

It returns an `fso.moe.MoeExperts`, a frozen handle with the fields `kind`,
`arch` (the compute capability it was prepared on, as `major * 10 + minor`),
`num_experts`, `hidden`, `inter`, `w13`, `sw13`, `w2`, `sw2` and
`w13_interleaved`. The tensors inside are in the layout the architecture's
kernels read, and the MXFP8 scale handles are arch-native (*Scale layouts*
above), so a handle is valid only on the architecture family it was prepared
on, and `layer` refuses a handle from another family. The constraints are
`H % 128 == 0`, `I_local % 128 == 0` (under tensor parallelism, a tp that keeps
the intermediate size per rank a multiple of 128), at most 1024 local experts,
and a router that draws its top-k without replacement. Every rejection names
the architecture, the format and what to use instead.

Two properties of the MXFP8 preparation matter to a caller. First, `"bsfp8"` on
sm_100/103 and sm_120/121 is a double quantization: every weight is already an
fp8 value times its block scale, and requantizing it to 1×32 rounds it a second
time. That is the price of serving a block-FP8 checkpoint there, because the
grouped MXFP8 kernels are the only grouped path those architectures have.
`tests/gemm/unit/test_moe_unified.py` measures the error against an fp32
reference on the dequantized block-FP8 weights and gates it in the same
accuracy class as one MXFP8 pass against a bf16 master; a checkpoint that still
has its bf16 master can take the single-rounding route with `"mxfp8"`. Second,
the FC1 rows are interleaved (gate_j and up_j adjacent) whenever
`mxfp8_grouped_swiglu_available(2 * I_local, H)` says the fused FC1 can serve
the shape, which it does unless `FSO_FC1_FUSED=0`; the handle's
`w13_interleaved` records the choice and the layer follows it. The conversion
runs a few experts at a time under a fixed scratch budget, so preparing a layer
never holds a second full copy of its weights. On sm_90 the handle keeps the
caller's own block-FP8 tensors, so freeing the source parameters afterwards
releases no memory there; on the MXFP8 architectures it does.

**The layer.** `layer(hidden, experts, topk_ids, topk_w, *, bias=None, bias_scale=None)`
computes, for every token `t`,
`out[t] = Σ_j topk_w[t, j] · expert_{topk_ids[t, j]}(hidden[t]) + bias_scale[t] · bias[t]`,
where `expert_e(x) = (silu(x W_gate^T) ⊙ x W_up^T) W_down^T` over the rank's
local experts. `hidden` is bf16 `[M, H]`. `topk_ids` may be int32 or int64 (the
sglang top-k produces int64) and is narrowed to int32 inside the call, and
`topk_w` is taken to fp32. `bias` is bf16 `[M, H]` and `bias_scale` one factor
per token, which is where a shared expert's output and its sigmoid gate go;
`bias_scale` requires `bias`. The result is a new bf16 `[M, H]` tensor.

*Skipped ids.* Every id outside `[0, E_local)` is skipped at no expert cost on
every architecture. That covers the padded rows of a graph bucket, which sglang
labels `num_experts` or −1, the entries an expert-parallel dispatcher maps to −1
because another rank owns the expert, and the padded-row sentinel that
dispatcher maps to `E_local`. A token whose ids are all skipped comes out as its
bias term, or as zero without a bias. The result is this rank's partial sum; the
caller reduces it across tensor- and expert-parallel ranks exactly as it does
for any other MoE runner.

*The bias fold.* On sm_100/103 and sm_120/121 the bias is added inside the
combine kernel's fp32 accumulation, so the sum is rounded to bf16 once. On sm_90
the combine kernel has no bias input yet, so the bias is folded with one fused
elementwise pass over the combined result: `out + bias`, or `out + bias_scale · bias`
evaluated as one fp32 multiply-add and then rounded to bf16. Because that pass
does not round the product on its own, its result can differ in the last bf16
bit from the literal torch expression `out + bias_scale[:, None] * bias`, which
rounds the product to fp32 before the add; the test therefore checks the fold
exactly against the single-rounding form. A bias input on the sm_90 combine
kernel is a later kernel step.

**The op and its dispatch.** `layer` validates the arguments against the handle
and calls `torch.ops.fish_scales_ops.moe_layer(hidden, w13, sw13, w2, sw2,
topk_ids, topk_w, bias, bias_scale, kind, w13_interleaved)`, a
`torch.library.custom_op` whose body dispatches on the handle's kind and on the
device's architecture:

| kind | architecture | what runs |
|---|---|---|
| `"bsfp8"` | sm_90 | `moe_layer_fp8_sm90` (the expert-sorted contiguous layout, swap-AB or non-swap picked from `M`), then the bias fold above |
| `"mxfp8"` | sm_120 / sm_121 | the masked-slab chain of `moe_layer_mxfp8_sm120`, unchanged, with `w13_interleaved` from the handle and the fused combine allowed (see below); the chunking rule `moe_layer_chunk_tokens_sm120` applies as it does there |
| `"mxfp8"` | sm_100 / sm_103 | the same chain, with the routing kernel asked for exactly the tensors this architecture's grouped routes read: per GEMM, `mxfp8_grouped_slot_possible` and `mxfp8_grouped_problem_shapes_consumed` (the FC1 asked under the kernel it will run) choose between the packed active-expert list for the slot-bound decode route and the per-group problem shapes for the pointer-array cascade, which is the composition `bench/gemm/python/bench_moe_qwen3_30a3.py` (`fso_mxfp8_layer`) drives. The fused FC1 is taken where `mxfp8_grouped_swiglu_fused_route` says so, there is no fused combine, and the chunking rule is the sm_120 one |
| anything else | any | `NotImplementedError` naming the architecture and the format to use instead, or `ValueError` for an unknown kind |

*The fused combine.* On sm_120/121 the fused-combine FC2 adds each routed row
into its token's output row with atomic reductions instead of storing the
down-projection slab for a separate combine kernel. Where
`moe_layer_fused_combine_engages_sm120` takes it (the buckets whose slab would
dominate the layer's transient memory) that is both faster and about half the
peak, but the order of the adds follows the order in which the CTAs finish, so
those buckets are stable in aggregate and not bit-reproducible run to run.
`fso.moe.layer` therefore keeps it opt-in, the same default as
`moe_layer_mxfp8_sm120`: `FSO_MOE_FUSED_COMBINE=1` allows it and the engagement
rule then decides per bucket; unset, every bucket keeps the slab and the combine
kernel and is reproducible, and a caller that has to bound the peak instead
uses `FSO_MOE_SLAB_BUDGET_MB`.

*Capture and compile.* The op can be captured into a CUDA graph after one eager
call of the same shape on the capturing thread, like every op in this library:
that call creates the per-thread pools and, on sm_90, JIT-compiles the grouped
kernels. Every host-side decision — the chunk count, the static GEMM hints, the
route queries, the swap-AB branch on sm_90 — is a function of the argument
shapes, so one capture per token bucket replays correctly for any routing of
that bucket, skipped ids included. Under `torch.compile` the op is opaque: its
fake implementation returns an empty `[M, H]` tensor, dynamo keeps the call as
one graph node without a graph break, and the compiled call runs the same body.

**Capability queries.** `supported(format, arch=None)` answers from the
architecture matrix: `"bsfp8"` on sm_90, sm_100/103 and sm_120/121, `"mxfp8"`
on sm_100/103 and sm_120/121, and nothing else anywhere. `arch` defaults to this
device. The query never raises for a format it does not know (it answers
`False`), so a model-level resolver can ask it before it chooses this layer; an
`arch` that names no compute capability raises `ValueError`. `describe()`
returns the same matrix as text with this device's rows marked.

`tests/gemm/unit/test_moe_unified.py` runs on every architecture and skips what
the device cannot run. It asserts that `fso.moe.layer` is `torch.equal` to the
per-arch entry for the same weights and inputs — `moe_layer_mxfp8_sm120` on
sm_120/121, `moe_layer_fp8_sm90` on sm_90, and on sm_100/103 the bench's
composition — across decode and prefill token counts, both id dtypes, skipped
ids, the bias forms, a captured graph and a `torch.compile` call. It also
measures the `"bsfp8"` double quantization, checks the fused-combine rule and its
switch, and checks the refusals and the capability matrix.

### Grouped MoE (added 2026-09)

Two layouts, one per arch. Both keep every per-expert row count **on the
device**: the host never reads routing results, so a layer captured once into a
CUDA graph replays correctly for any routing (§ CUDA-graph compatibility).

**sm_120 and sm_100/103 — masked slab layout (MXFP8).** Every expert `g` owns
a slab of `m_cap` rows; `masked_m[g]` (device int32) says how many are valid;
rows at or past it hold undefined bytes in every tensor. The layer is six
kernels, and the Python surface is identical on both arch families — only the
opaque scale byte layout differs (see the two `grouped MXFP8 … scales` rows in
*Scale layouts* above; the shapes in the table below are the sm_120 ones):

| step | function | in → out |
|---|---|---|
| whole block (sm_120/121) | `moe_block_mxfp8_sm120(hidden, router_weight, w13_fp8, sw13, w2_fp8, sw2, topk=…, renormalize=True, num_token_non_padded=None, expert_map=None, shared_*=None, shared_gate_in_router=False, shared_out=None, shared_gate=None, out=None, chunk_tokens=None)` | bf16 `[M, HIDDEN]` + the router weight + the routed experts (+ the shared expert) → bf16 `[M, HIDDEN]`; the router, the routed experts, the shared expert and the gated add that joins them, in one capture-safe call. See *the composed sm_120 block* below |
| router | `moe_router_topk(hidden, router_weight, topk, renormalize=True, with_shared_gate=False, num_token_non_padded=None, expert_map=None)` | bf16 `[M, HIDDEN]` × bf16 `[num_experts(+1), HIDDEN]` → (int32 `[M, topk]`, fp32 `[M, topk]`, fp32 `[M]`); the logit GEMM (cuBLAS) followed by the kernel below |
| router, logits given | `moe_topk_from_logits(logits, topk, renormalize=True, with_shared_gate=False, num_token_non_padded=None, expert_map=None)` | `[M, num_experts(+1)]` bf16/fp16/fp32 → the same three tensors, one launch: softmax, top-k, renormalise, the padded-row sentinel, the expert-parallel id remap and the shared expert's sigmoid gate |
| unified entry (sm_120/121) | `moe_layer_mxfp8_sm120(hidden, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w, chunk_tokens=None)` | bf16 `[M, HIDDEN]`, per-expert fp8 weights with their opaque int32 scales, int32 or int64 `[M, topk]`, fp32 `[M, topk]` → bf16 `[M, HIDDEN]`; runs the six kernels below with every host-side decision taken from the argument shapes, so one call captures per graph bucket. See *the composed sm_120 layer* below for the expert-parallel id contract, the `m_cap` rule and the optional token chunking |
| routing | `moe_build_routing(topk_ids, num_groups, m_cap, with_slots=False, problem_shapes_for=None)` | int32 `[M, topk]` → (`masked_m [G]`, `row_map [G*m_cap]`, `slot_of_flat [M*topk]`), plus `slot_to_expert [G]` when `with_slots=True`, plus a list of int32 `[G, 3]` per-group `(rows, N, K)` tensors, one per `(N, K)` pair in `problem_shapes_for`; `G <= 1024`, `m_cap % 4 == 0`, `m_cap >= M`. An entry of `topk_ids` outside [0, `num_groups`) is skipped: no group counts it, it takes no slab row, and its `slot_of_flat` entry is −1, which the gather-quantize, the SwiGLU requantize and the combine all read as "no routed row". Ask for `with_slots` only when `mxfp8_grouped_slot_possible(…)` says a GEMM of this layer's shape would take the sm_100/103 slot route, which is the list's only reader, and for `problem_shapes_for` only the pairs for which `mxfp8_grouped_problem_shapes_consumed(…)` answers `True` (see *the slot-bound decode route* and *routing-supplied problem shapes* below) |
| gather + quantize | `quantize_1x32_grouped_gather_fp8(x, slot_of_flat, topk, num_groups, m_cap)` | bf16 `[M, K]` → (fp8 `[G, m_cap, K]`, int32 `[G, K/128, m_cap]`) |
| gate_up | `linear_mxfp8_grouped_masked(a_fp8, w13_fp8, sa, sw13, masked_m, expected_m, max_active_groups=0, slot_to_expert=None, problem_shapes=None)` | → bf16 `[G, m_cap, 2*INTER]` |
| SwiGLU + quantize | `silu_chunk_mul_quantize_1x32_grouped_fp8(gu, slot_of_flat)` | → (fp8 `[G, m_cap, INTER]`, int32 `[G, INTER/128, m_cap]`) |
| down | `linear_mxfp8_grouped_masked(h_fp8, w2_fp8, sh, sw2, masked_m, expected_m, max_active_groups=0, slot_to_expert=None, problem_shapes=None)` | → bf16 `[G, m_cap, HIDDEN]` |
| combine | `moe_combine(dn, slot_of_flat, topk_w, bias=None, bias_scale=None, out=None)` | → bf16 `[M, HIDDEN]`, `out[t] = Σ_j topk_w[t,j] · dn[slot_of_flat[t*topk+j]] + bias_scale[t] · bias[t]`. `bias` is a shared-expert output and `bias_scale` its per-token gate, folded in so the block does not pay another read-modify-write of the whole result; `out` is a caller-owned destination (a data-parallel or reduce-scatter buffer) The kernel reads `slot_of_flat` and `topk_w` before its PDL wait (they are routing outputs, two or more launches upstream in every chain of this library), so a direct caller must not produce either of them in the kernel launched immediately before the combine on the same stream; only `dn` may come from the immediately preceding kernel. On sm_120 the launch uses 128-thread blocks while the grid is at most 256 such blocks (M ≤ 128 at HIDDEN = 2048), so that one combine CTA fits beside two resident FC2 CTAs and PDL can start it early (run `sm120_r2_p1_20260929`) |
| weights (offline) | `quantize_moe_weights_1x32_fp8(w)` | bf16 `[G, N, K]` → (fp8 `[G, N, K]`, int32 `[G, K/128, N]`); `N % 128 == 0`, `K % 128 == 0` |

`expected_m` is a **host-side static hint** (`ceil(M * topk / G)`) used only for
tile selection; pass a plain `int` so the captured graph stays shape-static.
Caller contract: `masked_m[g] <= m_cap` for every group (the kernel does not
check; overflow rows are dropped at the TMA bounds). The two grouped GEMM
constraints are `K % 128 == 0`, `N % 128 == 0`.

`linear_mxfp8_grouped_masked` takes one further optional host-side static hint,
`max_active_groups` (default `0`, meaning "not supplied"): an upper bound on how
many groups can hold at least one row, i.e. `min(M * topk, G)`. It carries
information `expected_m` cannot — `expected_m` is `ceil(rows / G)` and stays at
`1` across the whole decode band while the number of groups that can hold rows
runs from `topk` to `G` — and on sm_100/103 it is what lets the dispatcher size
the slot-bound decode route described below. Like `expected_m` it must be a
plain `int` and must not vary between a capture and its replays. It is
constrained to `0 <= max_active_groups <= G`. sm_120/121 accept and ignore it.

**The composed sm_120 layer.** `moe_layer_mxfp8_sm120` is the sm_120/121 entry
behind `fso.moe.layer`: one call, the six kernels above, and no host-visible
dependence on the routing. It is the sm_120/121 twin of `moe_layer_fp8_sm90` and
refuses on any other architecture, naming it. sm_100/103 serves its decode band
from a slot-bound route and the rest from a pointer-array cascade, and both read
routing tensors (the packed active-expert list, the per-group problem shapes)
that this entry does not ask the routing kernel for; `fso.moe.layer` runs the
same chain there with exactly those tensors requested.

*Expert-parallel-local ids.* The expert count the call sees is
`num_experts = w13_fp8.shape[0]`, which under expert parallelism is the
rank-local count and not the model's. Every routed entry whose id falls outside
`[0, num_experts)` is skipped at no expert cost. Two kinds of entry need that.
A serving engine captures one graph per token bucket and pads the rows a bucket
does not use; sglang labels the padded rows with the id `num_experts` (the
overflow slot of its `moe_align` kernel) or with −1. An expert-parallel
dispatcher additionally rewrites every expert another rank owns to −1, which
lands on individual top-k entries of real tokens rather than on whole trailing
rows. A skipped entry contributes nothing to its token's output row, and a token
whose every entry is skipped gets an all-zero row, which is the identity for the
cross-rank sum the caller performs afterwards.

*Static hints.* `expected_m` is `ceil(M · topk / num_experts)` and
`max_active_groups` is `min(M · topk, num_experts)`; both are functions of the
argument shapes alone, so each captured bucket bakes one value of each and no
replay can invalidate them.

*Capacity, and the memory it costs.* The masked slab gives every expert its own
`m_cap`-row window, and with top-k drawn without replacement a single expert can
receive one row per token, so the layer sizes `m_cap = align(tokens, 4)` — the
smallest capacity that cannot overflow for any routing, which is also what
`moe_build_routing` enforces. Top-k without replacement is a precondition of that
bound: a token naming one expert twice gives that expert two rows, so an expert
could hold more rows than there are tokens and the builder would place the
overflow in the next expert's window, or past the slab for the last expert.
(Duplicated ids are not the same case as the skipped ids above, which take no row
at all.) The consequence is that the transient slabs cost `num_experts · m_cap`
rows in five tensors at once, about 1.1 MB per token at 128 local experts with
hidden 2048 and inter 512
(`moe_layer_slab_bytes_per_token_sm120(num_experts, hidden, inter)` returns the
exact per-row figure). A prefill bucket of 8192 tokens therefore asks for roughly
9 GB of scratch, which does not fit beside the weights on a 32 GB card.

*Chunking is available and off by default.* The entry can split the token
dimension and run the layer once per chunk, which bounds the peak at the chunk's
slabs, and the result is bit-identical to the single call — a token's output row
depends only on its own activation row and on the expert weights, which
`tests/gemm/unit/test_moe_layer_determinism_sm120.py` asserts across chunk counts.
It is off unless asked for because the cost is not small: each call walks every
expert's weights once, so n chunks read the expert weights n times, and on the
RTX 5090 two chunks at 1024 tokens measured +80 % and five chunks at 4096 tokens
+209 % on the Qwen3.5-35B-A3B routed layer. A caller that must bound the peak
asks for it with `chunk_tokens=` or by capping the slabs with
`FSO_MOE_SLAB_BUDGET_MB`, and `moe_layer_chunk_tokens_sm120(M, num_experts,
hidden, inter)` answers what a given bucket would then do; both are host-side
decisions taken from shapes, so a captured bucket's chunk count is fixed at
capture. Without a cap a bucket whose slabs do not fit raises the allocator's
out-of-memory error instead of quietly running several times slower. The real
answer above that point is a contiguous (expert-sorted) sm_120 entry that sizes
its activation by the routed rows rather than by capacity, as the sm_90 layer
does; it does not exist yet, which is also why
[`docs/perf/layer/sm120.md`](../perf/layer/sm120.md) has no M = 8192 fso cell.

*Weights.* `w13_fp8` / `w2_fp8` and their scale handles are what
`quantize_moe_weights_1x32_fp8` produces **on this architecture** — scale
layouts are arch-native and opaque, so a set quantized on sm_100/103 is not
valid here. The `w13` row order is either the checkpoint's own `[gate; up]`
(`w13_interleaved=False`) or the gate/up-interleaved order that
`quantize_moe_weights_1x32_fp8(w13, w13_interleave=True)` and
`interleave_w13_fp8` produce (`w13_interleaved=True`, the fused FC1 described
next; the helper permutes the K-major scale words along with the rows). A
serving artifact that ships block-scaled 128×128 FP8 experts is converted at
load time by `fso.moe.prepare_experts(..., format="bsfp8")`, which dequantizes
each block to bf16 and requantizes it to 1×32 MXFP8 on this device; that is a
second quantization, and quantizing the bf16 master (`format="mxfp8"`) is the
single-rounding alternative.

**The fused FC1 on sm_120/121.** With FC1 weights quantized in the interleaved
gate/up row order (`quantize_moe_weights_1x32_fp8(w13, w13_interleave=True)`), the
FC1 carries the SwiGLU and the MXFP8 requantize in its own epilogue:
`linear_mxfp8_grouped_masked_swiglu` returns the FP8 `[G, m_cap, INTER]` slab and
its K-major scale words directly, so the bf16 `[G, m_cap, 2*INTER]` intermediate
and the kernel that used to read it back both disappear, and the layer runs five
kernels instead of six. Pass `w13_interleaved=True` to `moe_layer_mxfp8_sm120` or
`moe_block_mxfp8_sm120` and the layer takes that route where it is available and
falls back to the unfused FC1 with the pairwise SwiGLU kernel — which reads the
same interleaved layout — where it is not. It is a load-time decision, because a
model holds one weight layout: ask `mxfp8_grouped_swiglu_available(2*INTER,
HIDDEN)` before quantizing, and `FSO_FC1_FUSED=0` turns the whole feature off (the
query then answers false, so a caller driven by it keeps the stacked layout).

The epilogue reads the tile back from the shared-memory staging buffer the
epilogue already fills rather than pairing gate with up inside the accumulator:
the requantize needs an amax over 32 INTER columns, which is 64 N columns spread
over several warps' fragments, and shared memory makes that local — each lane owns
two INTER columns, so a scale block is sixteen lanes of one row and the reduction
never leaves the warp. The arithmetic is the separate kernel's instruction for
instruction (the same bf16 silu through `tanh.approx`, the same amax, the same
UE8M0 derivation, the same paired satfinite converts), and
`tests/gemm/unit/test_mxfp8_fused_fc1_sm120.py` asserts bit equality of both the
FP8 rows and the packed scale words at every row the masked layout declares valid.
One detail that is a correctness requirement rather than an optimisation: a scale
word packs four 32-column blocks that four different N tiles own, so the epilogue
stores its UE8M0 byte as a single byte — a read-modify-write of the word would
race.

**The fused combine on sm_120/121 — faster and about half the peak.**
`fused_combine=True` lets the FC2 add each row into its token's output row with
eight-byte atomic reductions (`linear_mxfp8_grouped_masked_combine`), instead of
storing the bf16 `[G, m_cap, HIDDEN]` slab that `moe_combine` then reads back. It
removes that slab entirely, so the layer's transient footprint falls by about half —
Qwen3.5-35B-A3B at M = 8192 goes from 14.0 GiB to 6.0 GiB, Qwen3-30B-A3B from 7.5 to
3.5 — which is the difference between fitting and not fitting beside the weights on a
32 GB card. Since 2026-09-28 it is also **2–7 % faster** than the slab pair at the
token counts where it engages (M ≥ 4096 on both families), because the scatter runs
on the store warp and the spare fourth TMA warp and therefore overlaps the next
tile's mainloop the way the TMA store it replaces does. Written in the math warps it
had cost 0.2–7.0 % instead; `FSO_MOE_SCATTER_WARP=0` restores that form for an A/B.
Compare the other way to bound the same peak, chunking the token dimension, which
costs +80 % to +209 %.

Two consequences for a caller. The route engages only where the slab dominates the
footprint (`moe_layer_fused_combine_engages_sm120(M, topk, hidden)`), so
`fused_combine=True` can be set once for every bucket and the small ones stay bit
identical. And where it does engage the adds are atomic, so the result is stable in
aggregate — cosine 0.99999 against the deterministic path, run-to-run spread about
two bf16 ULP — but not bit-reproducible, which is why it is off by default and why
`tests/gemm/unit/test_mxfp8_fused_combine_sm120.py` checks it to a tolerance and
reports the spread rather than asserting equality. The op accumulates into the
output, so the layer pre-fills it with the shared expert's gated rows or with zero;
that is also what gives a token whose every routed entry was skipped the right
value.

**The composed sm_120 block.** `moe_block_mxfp8_sm120` is the next surface out:
the router, the routed experts, the shared expert and the gated add that joins
them, in one capture-safe call. It is the shape a serving stack's MoE block has
— sglang's `Qwen2MoeSparseMoeBlock` / `Qwen3MoeSparseMoeBlock` — minus the
collective at the end, which stays with the caller because a library has no
business owning the process group. Per layer it replaces:

- the router GEMM, the softmax top-k with renormalisation, the padded-row
  masking and, under expert parallelism, the id remap: five launches become two,
  of which one is the GEMM (`moe_router_topk`; `moe_topk_from_logits` is the
  fused half for a caller that computes its own logits);
- the routed expert layer, as `moe_layer_mxfp8_sm120`;
- the shared expert, four kernels, whose sigmoid gate the router computes for
  free when its weight row is concatenated onto the router weight
  (`shared_gate_in_router=True`, which makes the router weight
  `[num_experts + 1, HIDDEN]`);
- the gated add of the two outputs, folded into the combine, so the block does
  not read and write the whole `[M, HIDDEN]` result twice more.

When the block owns the shared expert it runs it on a side stream, so it overlaps
the routed path instead of queueing behind it: the shared expert depends only on
the block's input, and the routed path needs its result only at the combine, so
the join sits after the down projection and the two branches run concurrently —
eagerly and inside a captured graph alike, since `wait_stream` records the events
capture follows to build the two branches. sglang does the same thing from the
model side with an alt stream. The output is bit-identical either way, because a
stream dependency is not a change of arithmetic. The side stream is created on the
first call, which therefore has to be eager, exactly like the first call of the
multi-CTA routing scratch or the grouped argument pool; a first call inside a
capture aborts with a message naming the knob. `overlap_shared=False` per call, or
`FSO_MOE_BLOCK_OVERLAP=0` process-wide, keeps the sequential order.

A caller that already overlaps its shared expert with the routed path on its own
second stream keeps that arrangement and passes the result as `shared_out` (with
`shared_gate` if it gated it); the fused add still applies. Passing both the
shared weights and `shared_out` is an error, since only one of them can be the
shared expert.

*Tensor parallel.* The experts are sharded along the intermediate dimension, so
the `INTER` this entry sees is the per-rank size, and it has to stay a multiple
of 128 — the 1×32 scale layout blocks the K dimension of the FC2 by 128. With
`INTER = 512` that admits tp 1, 2 and 4; tp 8 leaves 64 and is refused, by the
weight quantizer at load time and by the block per call, both naming the reason.
The block's output is the rank's partial sum, which the caller all-reduces
exactly as it does with any other runner.

*Expert parallel.* The rank holds a subset of the experts, so pass `expert_map`,
the int32 `[num_experts + 1]` table the dispatcher already builds: this rank's
experts mapped into `[0, num_local_experts)`, every remote expert and the
padded-row sentinel mapped outside it. The router then selects over the whole
expert set (its weight is replicated, so it must), the map is applied inside the
router kernel rather than in a separate gather, and every entry this rank does
not own costs nothing and contributes nothing — the skip described above. Without
`expert_map` the router's ids are global, so the local expert count must equal
`num_experts`, which the entry checks and says.

*Data parallel.* Attention under data parallelism hands the block a padded token
buffer whose valid length is a device value, which is exactly what
`num_token_non_padded` takes: the padded rows get the sentinel id inside the
router kernel, cost no expert compute, and come out zero. `out` writes the result
straight into the buffer the scatter or reduce-scatter already owns, which also
holds for a chunked call, where each chunk writes its own rows.

*Determinism.* Run-to-run bit-identical output, graph replay equal to eager, and
chunked equal to unchunked are asserted for both shape families and both
expert-parallel forms by `tests/gemm/unit/test_moe_layer_determinism_sm120.py`;
`tests/gemm/unit/test_moe_layer_padded_ids_sm120.py` covers the skipped ids
end to end and `tests/gemm/unit/test_moe_routing_masked_ids.py` covers the
routing builders and the gather on every architecture. The block's own plumbing,
the three parallel forms and its determinism are in
`tests/gemm/unit/test_moe_block_sm120.py`, and the router against the torch
reference (both logit dtypes, renormalisation on and off, padded rows, the expert
map, the shared gate, graph replay) in
`tests/gemm/unit/test_moe_router_topk.py`, which runs on every architecture.

On sm_100/103 the GEMM is a CUTLASS pointer-array (grouped) tcgen05
block-scaled kernel. Its per-group problem shapes, base pointers, strides and
scale-factor layouts live in device memory. Without `problem_shapes` they are
rebuilt from `masked_m` by a small kernel that runs immediately before every
GEMM launch, which is what keeps a captured graph correct across re-routing.
That preparation kernel is split around a grid-dependency barrier — the arrays
that depend only on the tensor set are written before it, the two that depend
on the routing after — and both it and the GEMM are launched as programmatic
dependent launches, so the routing-independent half runs while the kernel
ahead of it is still on the machine. With `problem_shapes` the preparation
kernel is not launched at all; see *routing-supplied problem shapes* below.
`FSO_DISABLE_PDL=1` turns every launch of the chain back into a plain
serialised one. Two further consequences follow from the hardware's 128-row /
128-column block-scaled tile granularity: a group with fewer than 128 valid
rows still runs one 128-row tile (the padding rows are zero-filled on load and
dropped on store), and a group with zero valid rows contributes no tiles at
all.

**`moe_build_routing` on sm_100/103 and sm_120/121 — the multi-CTA builder.**
On these arches `moe_build_routing` selects a multi-CTA kernel once the call
has at least 4096 routed pairs (`M · topk`); below that, and on every other
arch, it keeps the single-CTA kernel (sm_120 joined on 2026-09-28: the
single-CTA builder had cost the Family B layer at M = 4096 about 4 % on that
card; `FSO_MOE_ROUTING_MULTI=0` restores the single-CTA kernel everywhere, and
the routing tests compare the two builders on the same draw). Three parts of its contract are worth stating because a
caller can observe them.

- **Slot order is a free permutation, and this kernel uses a different one.**
  The output the caller is promised is: `masked_m[g]` is the number of routed
  pairs assigned to expert `g`; every pair's slot lies inside its own expert's
  block and below that expert's count; the slots are a permutation, no two
  pairs sharing one; and `row_map` at a pair's slot names that pair's source
  token. Which slot inside a group a given pair gets is *not* promised — the
  single-CTA kernel already documents its order as one permutation of the
  sorted recipe, and the multi-CTA kernel produces another (rank within a CTA's
  reserved block rather than global arrival order). Every consumer addresses
  rows through `row_map` / `slot_of_flat`, the GEMM treats a group's rows
  independently, and `moe_combine` sums a token's `topk` slots, so the layer
  output does not depend on the choice. Code that hard-codes a slot index, or
  compares two runs' `row_map` element by element, does.
- **One eager call per host thread before capture.** The multi-CTA kernel needs
  a small global scratch (per-expert counters plus an arrival counter) that
  must be zero when a launch starts; the launch restores it to zero before it
  exits. That scratch is allocated and zeroed on the thread's first eager call
  and never again, so a *first* call made inside a stream capture aborts with a
  message — the same warm-up contract the grouped GEMM's argument pool has.
- **The scratch is per host thread and per stream or capture.** Two launches
  may share a buffer only if they are ordered with respect to each other. The
  pool is `thread_local`, and within a thread it keys one buffer per capture
  sequence id while a stream is capturing and per stream handle otherwise. The
  capture key matters because `torch.cuda.graph` reuses one capture stream by
  default, so two graphs captured on that stream would otherwise share a buffer
  and could then be replayed concurrently. The case the keying does not cover
  is one captured `cudaGraph_t` instantiated into two `cudaGraphExec_t` and
  replayed at the same time; PyTorch instantiates once per
  `torch.cuda.CUDAGraph`, so it does not arise through the supported API. If a
  thread exceeds the pool's 16 slots the call warns once and falls back to the
  single-CTA builder, which needs no scratch. `tests/gemm/unit/test_moe_routing_threads.py`
  is the stress test for this: two host threads on two streams, released into
  each burst by a barrier so the kernels really overlap, checking every result
  against the contract above.

**sm_100/103 — the slot-bound decode route.** The sm_100/103 dispatcher has a
second grouped kernel for the decode band. It swaps the operand roles: the
expert weight rows go on the GEMM's M axis, where the block-scaled tile's
128-row granularity is exact, and the routed token rows go on a 64-wide N axis,
where a decode-sized expert wastes far less of the tile. The output is written
through a column-major `D` view that addresses the caller's ordinary row-major
`[G, m_cap, N]` buffer in place, so nothing downstream changes, and both scale
slabs are consumed exactly as the grouped quantizers emit them — the swap only
changes which slab is handed to which operand. The grid is sized from `max_active_groups` rather than from `G`, and instead of
the pointer-array prep kernel the route needs a slot list: the ids of the
experts holding at least one row, packed ascending into the low entries with
`-1` after them. `linear_mxfp8_grouped_masked` takes that list as a fourth
optional argument, `slot_to_expert` (int32 `[G]` on device, contiguous, on
`masked_m`'s device); pass `moe_build_routing(..., with_slots=True)`'s fourth
output and the route launches nothing but the GEMM. Omit it and the route
builds the same list itself with a one-block kernel before the GEMM, which is
what it did before, so existing callers keep working unchanged.
`linear_mxfp8_grouped_masked_swiglu` takes the same argument and uses it the
same way, because its decode-band form is this route's kernel with the fused
epilogue.

Ask for the list only where it will be read. `with_slots=True` makes the
routing kernel compact the per-expert histogram it already holds into that
list, and this route is the only consumer, so
`mxfp8_grouped_slot_possible(m_cap, n_w, k, num_groups, max_active_groups, fused_swiglu=False)`
answers — from the dispatcher's own route rule, under the same
`FSO_GROUPED_SLOT` setting — whether a GEMM of that shape would take it; a
layer passes `with_slots=` the disjunction of the query over its two GEMMs and
so pays for the list on the decode band only. Passing the list where the answer
is `False` is wasteful but never wrong, and withholding it where the answer is
`True` costs the route one extra one-block launch per call. Every architecture
but sm_100/103 answers `False`, because no other architecture has the route.

**sm_100/103 — routing-supplied problem shapes (the pointer-array route
without its preparation kernel).** Of the eleven per-group arrays the
pointer-array kernel reads, only the problem shapes depend on the routing: the
`(rows, N, K)` triple of every group. `moe_build_routing` already holds every
group's row count when it publishes `masked_m`, so
`moe_build_routing(..., problem_shapes_for=[(N, K), ...])` writes those
triples — one int32 `[G, 3]` tensor per named GEMM, `rows` clamped to `m_cap`
— in the same pass, and a grouped GEMM handed its tensor as `problem_shapes=`
launches nothing but itself. The ten remaining arrays (base addresses,
strides, scale-factor layouts) are a function of the tensor set alone and are
built once per distinct tensor set into a key-addressed block of a per-thread
arena, then reused by every call that presents the same addresses and shapes;
the key is compared on the host, and nothing reads device memory to decide.
Both ops take the argument; a layer asks
`mxfp8_grouped_problem_shapes_consumed(m_cap, n_w, k, num_groups,
max_active_groups, fused_swiglu=False)` per GEMM — the complement of
`mxfp8_grouped_slot_possible`, from the same route rule — and requests the
shapes for the GEMMs that consume them, which is what the benches and the
layer test do. The two queries are complements only when both are asked with
the same `fused_swiglu` flag, because the fused-SwiGLU FC1 leaves the slot
route one step earlier than the plain grouped GEMM (see *the slot-bound
decode route* above): a layer asks both for its FC1 with the flag set to
whether it will call the fused op, and for its FC2 with it clear, and then
per GEMM exactly one of the slot list and the problem shapes is requested. The slot route derives
its grid from `masked_m` and the slot list and ignores the tensor; sm_120/121
validates the shape and ignores it, so a caller can pass the same arguments on
every architecture. Omitting the argument keeps the preparation kernel and
changes nothing for existing callers.

Two consequences of the arena a caller can observe. First, the capture
contract gains one item, stated again under *CUDA-graph compatibility*: a
`torch.cuda.graph` capture allocates its tensors from the graph's private pool,
so the block a captured GEMM needs is bound during the capture, written outside
it (on the arena's own stream, with the host waiting, under the relaxed
capture mode) so that the graph carries no writer of its arrays, and then
pinned for the life of the process, because the graph's kernel node keeps
reading it and the library cannot learn when a graph is destroyed. Memory
therefore grows with the number of distinct (capture, GEMM) pairs a host
thread creates — about 13 KB per pair at 128 experts, 106 KB at the 1024
cap — inside one allocation of `FSO_GROUPED_ARG_POOL_MB` megabytes (default
128; a serving boot pins one block per captured graph bucket and MoE layer,
up to about 1,840 pairs, which the earlier 16 MB default could not hold) made
on the thread's first eager grouped call. A capture that finds the
arena full aborts with a message naming the variable, because falling back to
the preparation kernel inside the graph would silently give back the launch
this route removes; an eager call that finds it full takes that fallback,
which is always correct. Second, the shapes are read by the GEMM's tile
scheduler several launches after the routing kernel wrote them, under
programmatic dependent launch: that is safe because the CUTLASS grouped kernel
executes its grid-dependency wait before it constructs the scheduler (the
first read), and every kernel between the two executes its own wait before it
can retire, so the wait on the immediate predecessor implies the routing
kernel's completion. The content of the tensor cannot be checked on the device
— a triple built for another GEMM would walk tiles that do not exist —
so `FSO_CHECK_PROBLEM_SHAPES=1` makes the op copy it to the host and verify
`N`, `K` and `rows <= m_cap` on every call, for debugging only.

CUTLASS's static persistent scheduler clamps the launched grid to the SM count,
so whenever the tile space is larger than the machine each CTA loops over
several tiles. The kernel therefore decides whether a tile is live — its slot
holds an expert, and its token tile starts before that expert's routed row
count — separately for every tile, ahead of that tile's first TMA, rather than
once for the CTA's first tile. There is no whole-CTA early exit and no knob
that restores one: an exit taken before the persistent loops begin drops every
live tile assigned to a CTA whose first tile happens to be dead, and it was
also measured as never faster.

Two things a caller must know. First, the route is only taken when
`max_active_groups` is supplied; with the default `0` the call always goes to
the pointer-array kernel, so existing callers are unaffected. Second, the route
is correct only while the whole row capacity fits one token tile
(`m_cap <= 64`); the dispatcher enforces that as a hard check and the op raises
rather than running the kernel outside it. `FSO_GROUPED_SLOT` switches the
route: `0` never takes it, unset or `1` applies the dispatcher's rule, `force`
takes it whenever it is legal and raises when it is not, and `force@<N>` forces
it only for the GEMM whose `N` it names and sends every other shape to the
pointer-array kernel (the per-GEMM A/B form: a MoE layer's two grouped GEMMs
have different `N`, so this routes exactly one of them). The `force` forms are
A/B knobs, not production settings. The variable is read once per process, so it
must be set before the first grouped call and cannot change between a capture
and its replays.

The rule itself has two clauses and is about the size of the **live** tile
space and about which slot kernel the call lands on, not about how many experts
are idle. The live tiles the slot grid is built from,
`max_active_groups * ceil(N / 128)`, must be at most fourteen waves of the
device's SMs; and the row capacity must fit the kernel's token tile — the whole
64-wide tile for the plain slot kernel (`linear_mxfp8_grouped_masked`), but only
one 32-column epilogue chunk for the fused-SwiGLU slot kernel
(`linear_mxfp8_grouped_masked_swiglu`), so `m_cap <= 64` and `m_cap <= 32`
respectively.

Why that quantity. What the route buys is mainloop depth: the swap puts the
expert weight rows on the 128-row M axis and the routed tokens on a 64-wide N
axis, the activation stage in shared memory shrinks with it, and the freed
memory becomes pipeline stages — eight, against six and four for the
pointer-array kernel's tiles — which is what turns the expert weight stream from
latency-bound into bandwidth-bound. That advantage is a rate, so it is earned
again on every wave of live tiles, and it is present even when every expert
holds a row. It is bounded, though, by what the pointer-array kernel's wider
token tile saves per issue, which also accumulates with the wave count, so past
a certain number of live waves the wider tile wins. The second clause is about
the epilogue: the swap orientation puts one token's output element in each
lane, so the fused slot kernel's fp8 bytes and scale bytes leave transposed, one
byte per lane per token, in 32-column chunks that each cost a warp reduction, a
shared-memory exchange and a barrier, while the pointer-array kernel's fused
epilogue writes a thread's whole row with vector stores. That cost grows with
`m_cap` and overtakes the mainloop-depth advantage once a second chunk is
needed, which is why the fused slot kernel stops at one chunk while the plain
kernel, whose TMA-store epilogue has no such term, runs to the full tile.

With top-`k` routing `max_active_groups` is `min(M * topk, G)`, so for
Qwen3-30B-A3B (`G = 128`, top-8, `N = 1536` and `2048`) the fused FC1 takes the
route up to `M = 32` and `down` up to `M = 64`, and for Qwen3.5-35B-A3B
(`G = 256`, top-8, `N = 1024` and `2048`) the fused FC1 takes it up to `M = 32`
while `down` stops at `M = 16`. Ask `mxfp8_grouped_slot_possible` with
`fused_swiglu=True` for the FC1 that will run the fused op, as the benches do.

**sm_100/103 — the fused FC1.** On the pointer-array route the first grouped
GEMM can also do the SwiGLU and the MXFP8 requantize in its own epilogue, which
removes both the bf16 `[G, m_cap, 2*INTER]` intermediate and the separate
`silu_chunk_mul_quantize_1x32_grouped_fp8` launch:

| step | function | in → out |
|---|---|---|
| gate_up + SwiGLU + quantize | `linear_mxfp8_grouped_masked_swiglu(a_fp8, w13_fp8, sa, sw13, masked_m, expected_m, max_active_groups=0)` | → (fp8 `[G, m_cap, INTER]`, int32 per-group Sm1xx atom slabs) — the same pair `silu_chunk_mul_quantize_1x32_grouped_fp8` returns, so `down` consumes it unchanged |
| weights (offline) | `quantize_moe_weights_1x32_fp8(w13, w13_interleave=True)` | as above, with the FC1 rows re-ordered |
| weights (already quantized) | `interleave_w13_fp8(w13_fp8, sw13)` | → the same bytes in the interleaved row order |
| load-time router | `mxfp8_grouped_swiglu_available(n_w, k)` | → bool: can this shape use the fused FC1 at all? |
| per-call router | `mxfp8_grouped_swiglu_fused_route(m_cap, n_w, k, num_groups, max_active_groups)` | → bool: does this call take it? |

**`linear_mxfp8_grouped_masked_swiglu` requires the FC1 weight rows to be
gate/up interleaved — row `2j` is `gate_j`, row `2j+1` is `up_j`, not the usual
`[gate; up]` stacking — and it cannot detect the wrong layout: given the
stacked order it returns a finite, silently wrong answer and raises nothing,
because the two orders are the same bytes in a different sequence.** Produce
the interleaved form with `quantize_moe_weights_1x32_fp8(w13,
w13_interleave=True)`, or, for an already-quantized checkpoint, with
`interleave_w13_fp8`. Both are pure row permutations and give bit-identical
bytes, because the 1×32 weight quantizer works per row along K. The op and
both helpers exist on sm_100/103 and sm_120/121 (each permuting its own
arch-native scale layout) and raise `NotImplementedError` naming the
architecture elsewhere.

Why the interleave is needed. The fusion works because gate and up have to
meet in the same place in the accumulator, and the interleave is what puts them
there. On the pointer-array route the epilogue's TMEM-to-register copy hands one
thread one output row and 64 consecutive N columns of it, so with interleaved
weight rows those 64 columns are 32 gate/up pairs — exactly one 1×32 output
scale block — and the pairing, the SwiGLU and the 32-element amax all happen
inside one thread's registers. On the swap-orientation decode route the weight
rows sit on the accumulator's M axis instead, so with the same interleave
`gate_j` is row `2j` and `up_j` is row `2j+1`, i.e. two adjacent LANES of one
warp: the pairing is one `shfl_xor`, the amax over a block is one warp-wide
reduction plus one step across the warp pair that holds the block's two halves,
and the fp8 bytes go out transposed. Either way, with the stacked order `gate_j`
and `up_j` are `INTER` rows or columns apart and never meet at all.

Two decisions, both host-static, both taken by the library so a caller never
restates the rule. `mxfp8_grouped_swiglu_available(n_w, k)` is the load-time
one: a model holds one weight layout, so it decides how `w13` is quantized.
`mxfp8_grouped_swiglu_fused_route(m_cap, n_w, k, num_groups,
max_active_groups)` is the per-call one. Both grouped routes carry a fused FC1,
so with the knob unset it answers yes on both sides of the route boundary; what
it really decides, through the same `slot_route` the unfused FC1 asks, is WHICH
fused kernel the call lands on. A caller that ends up on the unfused FC1 anyway
(because it set `FSO_FC1_FUSED=0` for only part of its model, or because a
future shape has no fused instantiation) must pass `pairwise=True` to
`silu_chunk_mul_quantize_1x32_grouped_fp8` if its weights are interleaved; that
kernel then reads `gate_i` at column `2i` and `up_i` at column `2i+1` instead of
from the two halves of the row. The flag is sm_100/103 only — `pairwise=True`
raises `NotImplementedError` naming the architecture anywhere else, because no
other architecture's grouped FC1 ever produces a row in that order — and, like
the weight layout, its *value* cannot be inferred: on sm_100/103 the wrong one
multiplies the wrong pairs together and raises nothing.

`FSO_FC1_FUSED` switches the whole feature: `0` never uses the fused FC1 (and
`mxfp8_grouped_swiglu_available` then answers false, so a caller driven by it
keeps the stacked weight layout as well), unset applies the rule above, and `1`
uses it wherever it is legal. Like the other knobs it is read once per process
and must not change between a capture and its replays.

On the swap-orientation decode route the fused FC1 is the same 128×64×128
kernel the unfused decode route runs, with the direct-store epilogue in place of
the TMA-store one; its mainloop stage count is unchanged at eight and it uses
less shared memory than the unfused kernel, because the 1 KiB the cross-warp
amax step needs is far smaller than the staged bf16 D tiles it replaces.
On the pointer-array route the fused FC1 runs on the N tile that cascade would
pick for the unfused GEMM, with the direct-store epilogue in every case, and
with one documented exception: the cascade's rule that prefers a 192-wide N tile when
192 divides `N` and that tile's last wave of CTAs is fuller is capped at
`expected_m <= 16` for the unfused GEMM, and that cap does not apply to the
fused FC1. The cap exists because a wide BF16 direct store stops being cheap
once the rows pile up; the fused store writes about a quarter of those bytes,
so the wave arithmetic is allowed to decide at every `expected_m`.

**sm_90 — expert-sorted contiguous layout (block-FP8).** The `M * topk` routed
pairs are sorted by expert into one compact list, each expert's run padded to
the GEMM block size, so the scheduler enumerates active padded blocks rather
than all `E` experts. `P_max` (the padded row count) is rounded up and fixed per
`M`, so each `M` captures into its own graph.

| step | function | in → out |
|---|---|---|
| unified entry | `moe_layer_fp8_sm90(hidden, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w)` | bf16 `[M, HIDDEN]`, per-expert fp8 weights with fp32 `[E, N/128, K/128]` scales, int32 `[M, topk]`, fp32 `[M, topk]` → bf16 `[M, HIDDEN]`; picks the swap-AB GEMMs with an activation tile of 16 / 32 / 64 rows while the routed rows per active expert (`M * topk / E`) are at most 12 / 24 / 64 (`moe_swap_ab_block_n(M, E, topk)`; M <= 2048 on E=256/top-8) and the block_m = 64 GEMMs otherwise, a host-side branch on the static `M`; `topk_ids` entries outside [0, E) (sglang's masked rows past `num_token_non_padded`: −1 or `num_experts`) take no expert compute and a token whose entries are all masked gets a zero row |
| routing | `moe_build_sorted(topk_ids, num_groups, block_m)` | → (`sorted_expert_ids [P_max]` (−1 past the real length), `flat_to_sorted [M*topk]` (−1 for entries whose expert id is outside [0, num_groups); the gather / SwiGLU-quant / combine kernels skip them), `num_padded [1]`) |
| gather + quantize | `quantize_1x128_sorted_gather_sm90(x, flat_to_sorted, p_max, topk)` | → (fp8 `[P_max, K]`, fp32 `[K/128, align(P_max,4)]` K-major) |
| GEMMs | `linear_fp8_grouped_contiguous(a, w, sa, sw, sorted_expert_ids, block_m, expected_m)` and `linear_fp8_grouped_contiguous_swapab(..., block_n, expected_m)` | → bf16 `[P_max, N]` in sorted order; `sorted_expert_ids` must have been built with the same block size |
| SwiGLU + quantize | `silu_chunk_mul_quantize_1x128_sorted_sm90(gu, flat_to_sorted)` | → (fp8 `[P_max, INTER]`, fp32 `[INTER/128, align(P_max,4)]`) |
| combine | `moe_combine_sorted(dn, flat_to_sorted, topk_w)` | → bf16 `[M, HIDDEN]`; `flat_to_sorted` and `topk_w` are read before the kernel's PDL wait, so neither may be produced by the kernel launched immediately before the combine on the same stream (the sorted-layout builder is always further upstream) |
| weights (offline) | `quantize_moe_weights_1x128_fp8_sm90(w)` | bf16 `[G, N, K]` → (fp8 `[G, N, K]`, fp32 `[G, N/128, K/128]`) |

The sm_90 masked-layout variant (`linear_fp8_grouped_masked`,
`quantize_1x128_grouped_gather_sm90`,
`silu_chunk_mul_quantize_1x128_grouped_sm90`, driven by `moe_build_routing` /
`moe_combine`) is still public; the unified layer entry uses the contiguous
layout. `moe_build_routing`, `moe_build_sorted`, `moe_combine` and
`moe_combine_sorted` are arch-agnostic glue kernels.

## Constraints

- `K % 128 == 0` for every op on every arch. The former `K % 512` requirement
  of the sm_120 block-FP8 path was lifted 2026-09-05 (the packed scale word is
  zero-padded and the padded k-tiles are zero-filled by TMA).
- `N % 128 == 0` for `linear_fp8` and `linear_mxfp8` on sm_120 and sm_100/103
  and for every grouped GEMM. sm_90 dense `linear_fp8` accepts any `N`.
- **UE8M0 scales on Blackwell.** On sm_120 and sm_100/103 both the activation
  and the weight scales must be exact powers of two; the packed formats hold
  only the exponent byte. `quantize_1x128_fp8(..., use_ue8m0=True)`,
  `quantize_1x128_fp8_packed` and the weight quantizer's default produce them.
  The pre-pack ops refuse any other FP32 scale. (Until 2026-09-05 the weight
  quantizer produced plain `amax/448` scales, which the repack silently
  truncated; see `../perf/README.md` §8.)
- **FP32 scales on sm_90.** deep_gemm reads FP32 scales directly; do not call
  `repack_fp8_*_scales` there.
- Scale tensors are arch-native (table above) and are not portable across
  arch generations or between the block-FP8 and MXFP8 formats.
- MXFP8 raises `NotImplementedError` on sm_90. Grouped MXFP8 raises on
  anything but sm_120/121 and sm_100/103; the sm_90 grouped ops raise on
  anything but sm_90.
- The masked layout needs `m_cap % 4 == 0`, `m_cap >= M`, `G <= 1024`, and
  `masked_m[g] <= m_cap`; the sorted layout needs the same block size for
  `moe_build_sorted` and the GEMMs that consume its output.
- `linear_qx` and the `linear_bf16` runner path exist on sm_90 and sm_120;
  `linear_bf16` on sm_100/103 is composed from the public ops, `linear_qx`
  raises there.
- On sm_100/103 the cuBLAS `scaled_mm` tier needs a torch build with MXFP8
  `scaled_mm` (2.12); on older torch the router uses the DSL and C++ tiers
  only. The DSL rows JIT-compile each configuration at first use, which on a
  B300 with nvidia-cutlass-dsl 4.8.0 costs 0.2-0.6 s per configuration per
  process.
- The sm_100/103 CuTe-DSL rows need `nvidia-cutlass-dsl` (the `sm100` extra).
  The M ≤ 64 decode row needs at least **4.5.0** and **4.8.0 is the
  recommended pin**: 4.4.2 cannot run its kernels at all, 4.5.0 through 4.5.2
  make the JIT about four times slower than 4.4.2 on these configurations, and
  from 4.6.1 on it is two to three times faster than 4.4.2. Below the floor,
  or with the package missing entirely, the decode row disables itself and the
  router behaves exactly as it did before the row existed; `FSO_LOG=1` prints
  the one-line reason once per process.
- The decode row costs more HOST time per call than the tiers it displaces,
  and that is a property of the calling convention rather than of the kernels.
  Launching it through the plain (non-TVM-FFI) CuTe-DSL convention this
  library uses takes roughly 26 to 29 µs of host time per eager call, against
  roughly 9 µs for the C++ cascade op and 17 µs for the cuBLAS `scaled_mm`
  tier, because each call rebuilds the output descriptor, queries the current
  stream and marshals ten arguments through the DSL's Python-level host entry.
  A CUDA-graph-captured caller pays none of it: the marshalling happens once,
  at capture, and each replay is one `cudaGraphLaunch`. Decode loops should
  capture; a caller that must run the op eagerly one token at a time should
  set `FSO_DISABLE_DECODE_DSL=1` and keep the previous tiers.

## Picking the right op

| situation | op |
|---|---|
| Hopper (sm_90) dense inference, cached FP8 weight | `linear_fp8` with `quantize_1x128_fp8(x)` per call, FP32 scales |
| Blackwell consumer (sm_120) dense inference, block-FP8 checkpoint | `linear_fp8` with pre-packed weight scales and `quantize_1x128_fp8_packed(x)` per call |
| Blackwell consumer (sm_120), free choice of format | `linear_mxfp8` with `quantize_1x32_fp8` (tighter quantization; the perf tables show where it is also faster) |
| Blackwell datacenter (sm_100/103) | `linear_mxfp8` with `quantize_1x32_fp8`; block-FP8 checkpoints run through `linear_fp8` on the same kernels |
| SwiGLU MLP on Blackwell | `silu_chunk_mul_quantize_1x32_fp8` between the two GEMMs (no BF16 intermediate) |
| MoE layer, any architecture | `fso.moe.prepare_experts` once per layer at load time, then `fso.moe.layer` per call |
| one architecture's MoE chain directly (benches, experiments) | `moe_layer_fp8_sm90` on sm_90, `moe_layer_mxfp8_sm120` on sm_120/121, or the masked-layout ops composed as `bench/gemm/python/bench_moe_qwen3_30a3.py` does on sm_100/103 |
| both operands change every call | `linear_bf16` |
| BF16 activation, cached FP8 weight, one op (sm_90 / sm_120) | `linear_qx` |

## Lower-level interface (torch.ops bindings)

The Python wrappers are thin shims over `torch.ops.fish_scales_ops.*`. They
add the arch checks, the arch-dependent defaults, and on sm_100/103 the
Python-side tier router; call the raw ops only when you replicate those.

```
linear_bf16(Tensor x, Tensor w) -> Tensor
linear_fp8(Tensor x_fp8, Tensor w_fp8, Tensor sx, Tensor sw) -> Tensor
linear_qx(Tensor x_bf16, Tensor w_fp8, Tensor sw) -> Tensor
quantize_1x128(Tensor x, bool use_ue8m0=False) -> (Tensor, Tensor)
quantize_1x128_packed(Tensor x, bool use_ue8m0=True) -> (Tensor, Tensor)
quantize_128x128(Tensor w, bool use_ue8m0=False) -> (Tensor, Tensor)
repack_fp8_act_scales(Tensor sx_f32) -> Tensor
repack_fp8_wgt_scales(Tensor sw_f32) -> Tensor

quantize_1x32(Tensor x, bool use_ue8m0=True) -> (Tensor, Tensor)          # legacy two-step: fp32 scales
quantize_1x32_packed(Tensor x, bool use_ue8m0=True) -> (Tensor, Tensor)   # fused, arch-native packed scales
silu_chunk_mul_quantize_1x32(Tensor gu, bool use_ue8m0=True) -> (Tensor, Tensor)
repack_mxfp8_scales(Tensor scales_f32) -> Tensor                          # legacy two-step
linear_mxfp8_raw(Tensor x_fp8, Tensor w_fp8, Tensor sx_int32, Tensor sw_int32) -> Tensor

linear_mxfp8_grouped_masked(Tensor a_fp8, Tensor w_fp8, Tensor sa_int32, Tensor sw_int32, Tensor masked_m, int expected_m, int max_active_groups=0, Tensor? slot_to_expert=None, Tensor? problem_shapes=None) -> Tensor
quantize_1x32_grouped_gather(Tensor x, Tensor slot_of_flat, int topk, int num_groups, int m_cap, bool use_ue8m0=True) -> (Tensor, Tensor)
silu_chunk_mul_quantize_1x32_grouped(Tensor gu, Tensor slot_of_flat, bool use_ue8m0=True, bool pairwise=False) -> (Tensor, Tensor)
linear_mxfp8_grouped_masked_swiglu(Tensor a_fp8, Tensor w13_fp8, Tensor sa_int32, Tensor sw13_int32, Tensor masked_m, int expected_m, int max_active_groups=0, Tensor? slot_to_expert=None, Tensor? problem_shapes=None) -> (Tensor, Tensor)
mxfp8_grouped_swiglu_available(int n_w, int k) -> bool
mxfp8_grouped_swiglu_fused_route(int m_cap, int n_w, int k, int num_groups, int max_active_groups) -> bool
mxfp8_grouped_slot_possible(int m_cap, int n_w, int k, int num_groups, int max_active_groups, bool fused_swiglu=False) -> bool
mxfp8_grouped_problem_shapes_consumed(int m_cap, int n_w, int k, int num_groups, int max_active_groups, bool fused_swiglu=False) -> bool
moe_build_routing(Tensor topk_ids, int num_groups, int m_cap, bool with_slots=False, int[] problem_shapes_nk=[]) -> (Tensor, Tensor, Tensor, Tensor, Tensor)   # the Python wrapper takes problem_shapes_for=[(N, K), ...] and returns the [P, G, 3] tensor as a list of [G, 3] views
moe_combine(Tensor dn, Tensor slot_of_flat, Tensor topk_w) -> Tensor
moe_build_sorted(Tensor topk_ids, int num_groups, int block_m) -> (Tensor, Tensor, Tensor)
moe_combine_sorted(Tensor dn, Tensor flat_to_sorted, Tensor topk_w) -> Tensor

linear_fp8_grouped_masked(Tensor a_fp8, Tensor w_fp8, Tensor sa, Tensor sw, Tensor masked_m, int expected_m) -> Tensor
quantize_1x128_grouped_gather_sm90(Tensor x, Tensor slot_of_flat, int topk, int num_groups, int m_cap) -> (Tensor, Tensor)
silu_chunk_mul_quantize_1x128_grouped_sm90(Tensor gu, Tensor slot_of_flat) -> (Tensor, Tensor)
linear_fp8_grouped_contiguous(Tensor a_fp8, Tensor w_fp8, Tensor sa, Tensor sw, Tensor sorted_expert_ids, int block_m, int expected_m) -> Tensor
linear_fp8_grouped_contiguous_swapab(Tensor a_fp8, Tensor w_fp8, Tensor sa, Tensor sw, Tensor sorted_expert_ids, int block_n, int expected_m) -> Tensor
quantize_1x128_sorted_gather_sm90(Tensor x, Tensor flat_to_sorted, int p_max, int topk) -> (Tensor, Tensor)
silu_chunk_mul_quantize_1x128_sorted_sm90(Tensor gu, Tensor flat_to_sorted) -> (Tensor, Tensor)
```

`linear_mxfp8_raw` and `linear_fp8` read the packed scale buffers by raw
pointer in the arch-native layout; on sm_100/103 the Python `linear_mxfp8`
wrapper tries the `scaled_mm` and DSL tiers first and falls through to the raw
op.

## CUDA-graph compatibility

Every op is capture-safe **after one eager warmup of the same call**. The
warmup populates the process-static state that must not be created inside a
capture: the `cudaFuncSetAttribute` guard of each kernel instantiation, the
Stream-K side-stream pool and its scratch buffer (sm_120), the int32 scale
scratch pool of the fused runner paths, the grouped GEMM's argument pool and
its static-array arena, the multi-CTA routing builder's scratch pool and the
slot-bound route's slot-list pool (sm_100/103; all are per host thread, so
each thread that captures needs its own eager call), the deep_gemm NVRTC
compilation of each
kernel configuration (sm_90; in-memory only, no disk cache), and the DSL JIT
per configuration (sm_100/103), for the mid-band tier and for the M ≤ 64
decode row alike. `tests/gemm/unit/test_cuda_graph.py` is the
reference discipline: eager reference → three warm calls on the capture
stream → capture → replay, compared bit-exactly.

The sm_100/103 decode row adds one requirement to that contract, and it is the
only place in this library where it applies to a scale buffer. Its launch path
binds raw `data_ptr()` values rather than passing tensors, and the two MXFP8
scale buffers are passed as bare pointers because their six-dimensional
block-scaled layout cannot be expressed as a torch tensor. A captured graph
therefore keeps reading the addresses that were live at capture time, for the
activation, the weight, the output **and both scale buffers**. Rewriting any
of those buffers in place and replaying is correct and bit-exact; handing the
op a different tensor and replaying silently reuses the old address, exactly as
it would for any captured kernel, with nothing to raise about it.

What may change between replays: the routing. Both MoE layouts keep the
per-expert counts and index maps on the device and derive them inside the
graph (`moe_build_routing` / `moe_build_sorted` are captured), so a layer
captured for one `M` replays with any routing of that `M`. What must not
change: tensor shapes, `m_cap`, `expected_m`, `max_active_groups`, and the
sm_90 `M`-dispatch branch (all host-static per `M`; capture one graph per `M`).

The sm_100/103 grouped GEMMs handed `problem_shapes` add one more item. The
static argument block such a GEMM reads is bound during the capture and written
outside it — on the arena's own stream with the host waiting, under the relaxed
capture mode — so the captured graph contains no writer of those arrays, and
the block is then pinned for the life of the process. Every distinct
(capture, GEMM) pair a host thread creates therefore keeps its block (about
13 KB at 128 experts, 106 KB at 1024) inside the `FSO_GROUPED_ARG_POOL_MB`
arena (default 128 MB); a capture that finds the arena full aborts with a
message rather than putting a launch back into the graph. Rewriting the
routing buffers in place and replaying is correct; handing a graph's GEMM a
different tensor set means a different graph.

Two ops synchronize the device and belong at weight-load time, outside any
capture: `repack_fp8_act_scales` and `repack_fp8_wgt_scales` (the UE8M0
check). The per-call paths (`linear_fp8` with FP32 scales,
`quantize_1x128_fp8_packed`) do not synchronize.

## Env-var overrides (debugging / A-B only)

All are read once at first use and cached for the process; the cached value
is what a captured graph keeps. None of them is part of the production
contract — the cascades already encode the measured picks.

| variable | scope | effect |
|---|---|---|
| `FSO_FORCE_TILE="TM,TN,ST"` | sm_120 dense and grouped cascades; sm_100/103 dense and grouped cascades | force one tile instantiation instead of the cascade pick. The wire format is shared, but `ST` means different things per arch: on sm_120 it is the stage count, on sm_100/103 it names the (SM count, TileK, cluster, epilogue) variant — see the two tables below. An `(TM, TN, ST)` triple that is not instantiated prints a warning and falls through to the cascade |
| `FSO_FORCE_TILE_K=<K>` | sm_120, sm_100/103 | apply `FSO_FORCE_TILE` only to GEMMs whose `K` equals the value — lets a layer-level sweep force one projection while the others keep their picks |
| `FSO_FORCE_KSPLIT=<n>` | sm_120, sm_100 | force the Stream-K / split-K factor |
| `FSO_FORCE_MIN_BLOCKS=2` | sm_120 | run the 2-CTA/SM instantiation of the tiles that have one |
| `FSO_FORCE_SMALLM=1` | sm_120 MXFP8 | experimental small-M kernel variant |
| `FSO_FORCE_SCHED_GROUP=<n>` | sm_120 MXFP8 | persistent-scheduler swizzle group size |
| `FSO_DISABLE_OVERRIDES=1` | sm_120 | skip the shape-specific single-launch overrides, cascade table only |
| `FSO_DISABLE_STREAMK=1` | sm_120 dense | single launch everywhere |
| `FSO_STREAMK_POOL_MB=<n>` | sm_120 dense | Stream-K partial-sum scratch capacity (default 64 MB; allocated once, before capture) |
| `FSO_DISABLE_PDL=1` | sm_120 MoE chain; sm_100/103 MoE chain, grouped argument-preparation kernel and grouped GEMM | drop the programmatic-dependent-launch attributes (kernel-side waits become no-ops). On sm_100/103 this also covers the split prep kernel and the CUTLASS grouped GEMM, which are launched with PDL by default and whose grid-dependency barriers are compiled in for the sm_100/103 device passes |
| `FSO_MOE_BLOCK_OVERLAP=0` | sm_120/121 composed MoE block | run the shared expert on the current stream instead of the side stream, i.e. sequentially with the routed path. The result is unchanged; this is the A/B knob for the overlap |
| `FSO_MOE_SLAB_BUDGET_MB=<n>` | sm_120/121 composed MoE layer, and `fso.moe.layer` on sm_120/121 and sm_100/103 | cap on the transient slabs of one layer call, in MiB. Unset (the default) means one call per layer at any token count. Set, the entry splits the token dimension into as few chunks as fit the cap, which bounds the peak and costs one extra pass over the expert weights per chunk (+80 % at two chunks, +209 % at five on the 5090 Family C layer). The chunk count is a host-side decision fixed at capture |
| `FSO_MOE_FUSED_COMBINE=1` | `fso.moe.layer` on sm_120/121 | allow the fused-combine FC2 wherever `moe_layer_fused_combine_engages_sm120` says the down-projection slab dominates the layer's transient memory: faster and about half the peak there, but the atomic adds make those buckets not bit-reproducible run to run. Unset (the default) keeps the slab and the combine kernel at every token count, so every bucket is reproducible |
| `FSO_MOE_SCATTER_WARP=0` | sm_120/121 fused-combine FC2 | scatter the weighted rows from the math warps instead of from the store warp plus the spare fourth TMA warp. The default (the store warps) overlaps the atomics with the next tile's mainloop the way the TMA store does and is 6-8 % faster on the routed layer at M >= 4096; this is the A/B knob for it, and the result is unchanged either way |
| `-DFSO_PROLOGUE_TRACE` (build-time; runtime `FSO_PROLOGUE_TRACE_PTR`) | sm_120/121 grouped MXFP8 GEMM | compile-time-optional prologue trace, off by default and zero code when off (the production kernels are SASS-identical): with the macro defined, the kernel writes `%globaltimer` / `%clock64` stamps at every prologue boundary (descriptor prefetch, barrier init, staging, prefix scan, PDL wait, first tile, per-warp exit) for CTA 0 and the last CTA into a device buffer whose address the dispatcher reads from `FSO_PROLOGUE_TRACE_PTR`. Built into a separate tree copy for attribution runs (2026-09-28 prologue round), never into a production build; the result is unchanged either way |
| `FSO_MOE_DECODE_GATE=0` | sm_120/121 grouped MoE GEMM, decode band | restore the cascade's `m_cap`-only decode gate. The default (unset) is the tile-count rule of 2026-09-28: at row caps of 4–16 the grouped GEMM takes the solo (16,128,4) instance at one CTA per SM only when its worst-case tile count, `max_active_groups × ceil(m_cap/16) × ceil(N/128)`, lies between 0.6 × and 1 × the SM count — one full wave — and the 2-CTA/SM (16,64,4) instance otherwise; very short K (≤ 4 k-tiles) and a call without the `max_active_groups` hint keep the `m_cap` gate. The rule is decided on the routed layer (`docs/perf/README.md` §8): the isolated kernel cell ranks the two instances differently, because the layer's FC2 is launched under PDL while the FC1 still runs and packs two CTAs per SM behind the smaller 2-CTA FC1. The result is unchanged either way |
| `FSO_MOE_GROUP_ORDER=auto|<n>` | sm_120/121 grouped MoE GEMM | order the persistent scheduler visits the experts in. The default (unset, or 1) is expert-id order. `auto` walks them with a stride of about a quarter of the expert count, so consecutive tiles come from experts spread across the set instead of from one contiguous run; `<n>` sets the stride explicitly and is rejected (with a warning, falling back to expert-id order) unless it is coprime with the number of steps. `FSO_MOE_GROUP_BLOCK=<w>` (default 1, must divide the expert count) makes each step carry `w` consecutive experts, which keeps runs of that many adjacent in memory; `w = 8` measured best. The result is unchanged either way. This is worth taking only when a deployment's rows-per-expert are correlated with the expert id, which is when a block of lightly loaded experts would otherwise become a stretch of the launch with too little arithmetic to cover the weight tiles it must load: it is then worth about 2 % of the layer, and costs about 1-2 % when no such correlation exists |
| `FSO_SWAP_BN=16|32|64|0` | sm_90 composed MoE layer | force the swap-AB activation tile (0 = the non-swap block_m = 64 path) instead of the rows-per-expert cascade |
| `FSO_SWAP_STAGES=<n>` | sm_90 swap-AB grouped GEMM | pipeline depth of the single-CTA swap-AB kernel (clamped to the smem budget); two-CTA builds pick their own depth (the deepest count that fits twice and divides K, see `dispatch.cuh`) and ignore it |
| `FSO_SWAPAB_CTAS_PER_SM=1|2` | sm_90 swap-AB grouped GEMM | resident CTAs per SM (default 2: the math warp-groups run on 96 registers and the grid is doubled; 1 restores the single persistent CTA with 232 registers) |
| `FSO_JIT_EXTRA_FLAGS="-DFOO=1 ..."` | sm_90 | extra NVRTC flags appended to every deep_gemm JIT compile (developer A/B of kernel-side `#if` switches; the cubin cache key ignores them, so pair with a fresh `TRTLLM_DG_CACHE_DIR`) |
| `FSO_JIT_INCLUDE_DIRS=a:b:c` | sm_90 | NVRTC include directories for the deep_gemm JIT (default baked at build time) |
| `TRTLLM_DG_JIT_DEBUG=1`, `TRTLLM_DG_JIT_DUMP_CUBIN=1`, `TRTLLM_DG_JIT_USE_NVCC=1`, `TRTLLM_DG_NVCC_COMPILER=<path>`, `TRTLLM_DG_CACHE_DIR=<dir>` | sm_90 | deep_gemm JIT diagnostics: verbose compile, dump cubins, compile with nvcc instead of NVRTC, compiler path, dump directory |
| `FSO_DISABLE_SMM=1`, `FSO_DISABLE_DSL=1` | sm_100/103 MXFP8 router | skip the cuBLAS `scaled_mm` tier / both CuTe-DSL rows, the M ≤ 64 decode row and the mid-band tier (both set = pure C++ cascade) |
| `FSO_DISABLE_DECODE_DSL=1` | sm_100/103 MXFP8 router | skip the M ≤ 64 decode row only, leaving the mid-band DSL tier alive; this is the A/B knob for the decode row |
| `FSO_LOG=1` | sm_100/103 MXFP8 decode row | print, once per process, why the decode row is inactive (DSL too old, package missing, import failed). Silent when the row is working |
| `FSO_DSL_KERNEL_PATH=<file>` | sm_100/103 | alternative DSL kernel source for the mid-band tier (the decode row's kernels are vendored in-tree and not overridable) |
| `FSO_FORCE_SWIZZLE=<n>`, `FSO_FORCE_RASTER={1,2}`, `FSO_SK_NDET=1`, `FSO_SK_DECOMP={1,2,3}` | sm_100/103 C++ cascade | scheduler raster swizzle size, raster direction (along M / along N), nondeterministic Stream-K reduction, decomposition mode — probe knobs, never wired into the cascade |
| `FSO_PRINT_TILE_INFO=1` | sm_100/103 dense and grouped cascades | make every kernel instantiation print, once, the mainloop stage count `StageCountAutoCarveout` derived for it and its shared-memory footprint. That is the quantity that says whether a narrower `TileN` bought pipeline depth or only extra CTAs, and it is the only way to see a stage collapse (a tile whose epilogue eats the carve-out and leaves one mainloop stage) without guessing |
| `FSO_GROUPED_SLOT={0,1,force,force@<N>}` | sm_100/103 grouped MoE decode route | `0` never takes the slot-bound swap-orientation route, unset or `1` applies the dispatcher's rule, `force` takes it wherever it is legal and raises where it is not, `force@<N>` forces it for the GEMM whose `N` it names only (see the grouped section above; the `force` forms are A/B knobs) |
| `FSO_GATHER_QUANT_ONCE={0,1}` | sm_100/103 grouped MoE gather-quantize | `0` always quantizes per routed (token, expert) pair, unset applies the launcher's rule (the token-space form once its grid covers one full wave of SMs), `1` always quantizes each token once and scatters the bytes to its top-k destinations |
| `FSO_FC1_FUSED={0,1}` | sm_100/103 and sm_120/121 grouped MoE FC1, and the FC1 row order `fso.moe.prepare_experts` chooses | `0` never uses the fused FC1 (and `mxfp8_grouped_swiglu_available` then answers false, so a caller keeps the `[gate; up]` weight layout too), unset applies the router, `1` uses it wherever it is legal |
| `FSO_GROUPED_ARG_POOL_MB=<n>` | sm_100/103 grouped GEMMs with `problem_shapes` | capacity of the per-thread static-array arena (default 128 MB, enough for a serving boot's captures over every MoE layer; allocated once, on the first eager grouped call). Every distinct (capture, GEMM) pair pins one block of it; a capture that finds it full aborts with a message, an eager call falls back to the preparation kernel |
| `FSO_CHECK_PROBLEM_SHAPES=1` | sm_100/103 grouped GEMMs with `problem_shapes` | copy the caller's `[G, 3]` tensor to the host on every call and verify that its `N` and `K` are this GEMM's and that every row count lies in `[0, m_cap]`; read on every call (not cached), debugging only |
| `FSO_BENCH_WARM_MS=<ms>` | benches only | spin the GPU before each cell's timing (needed on unlocked devices, see `../perf/README.md` §5) |

### `FSO_FORCE_TILE` on the sm_100/103 dense cascade

`TM` is the tile's M extent, `TN` its N extent, `ST` the variant. The
block-scaled scale-factor copy atom fills 128 TMEM lanes, so `TM ∈ {128, 256}`
(128 for a 1-SM tile, 256 for a 2-SM cluster); CUTLASS rounds the scale block
up on the N axis, so `TN ∈ {64, 128, 192, 256}` is legal, but only the rows
below are compiled in.

| `ST` | variant | instantiated (`TM`, `TN`) |
|---|---|---|
| 1 | 1-SM cluster (1,1), TileK 128 | (128, 128), (128, 256) |
| 2 | 2-SM cluster (2,1), TileK 128 | (256, 128), (256, 256) |
| 3 | 1-SM cluster (1,1), TileK 256 | (128, 128), (128, 256) |
| 4 | 2-SM cluster (2,1), TileK 256 | (256, 128), (256, 256) |
| 5 / 6 | 2-SM cluster (2,2), TileK 128 / 256 | (256, 256) |
| 7 / 8 | 1-SM cluster (1,1) Stream-K, TileK 128 / 256 | (128, 128) |
| 9 | 1-SM parallel split-K, two kernels (splits from `FSO_FORCE_KSPLIT`, else auto) | (128, 128) |
| 10 | 1-SM cluster (2,2), TileK 128 | (128, 128), (128, 256) |
| 11 | 2-SM cluster (2,2), TileK 256 | (256, 128) |
| 20 / 21 | 1-SM cluster (1,1), TileK 128, direct-store / TMA epilogue | (128, 64), (128, 128), (128, 192) |
| 22 / 23 | 1-SM cluster (1,1), TileK 256, direct-store / TMA epilogue | (128, 64) |
| 24 / 25 | 2-SM cluster (2,1), TileK 128, direct-store / TMA epilogue | (256, 64), (256, 192) |

Codes 1–11 bake one epilogue choice into each row, which is what the shipped
cascade tiles need. Codes 20–25 were added on 2026-09-17 for sweeping a *new*
tile, where the epilogue is itself one of the things under test, so they name
the epilogue explicitly and carry the 64- and 192-wide N tiles. Those two
widths are what the dense wave-tile rule picks from; the rule itself is
`pick_wave_tile` in
`csrc/gemm/include/blockscale_gemm/arch/sm100/mxfp8/dispatch.cuh`, mirrored in
Python by `wave_tile_owns` in `python/fish_scales_ops/gemm/_sm100_smm.py` so
that the `scaled_mm` and DSL tiers decline the cells the cascade owns.

### `FSO_FORCE_TILE` on the sm_100/103 grouped cascade

Same wire format, a different `ST` meaning. `TN` carries the N-tile width.

| `ST` | variant | instantiated (`TM`, `TN`) |
|---|---|---|
| 1 | 1-SM cluster (1,1), TileK 128, TMA epilogue | (128, 64), (128, 128), (128, 192), (128, 256) |
| 2 | 2-SM cluster (2,1), TileK 128, TMA epilogue | (256, 128), (256, 256) |
| 3 / 4 | 1-SM (1,1) / 2-SM (2,1), TileK 256, TMA epilogue | (128, 128), (128, 256) / (256, 128), (256, 256) |
| 5 | 1-SM cluster (1,1), TileK 128, direct-store (NoSmem) epilogue | (128, 64), (128, 128), (128, 192), (128, 256) |
| 6 | 2-SM cluster (2,1), TileK 128, direct-store epilogue | (256, 128), (256, 256) |
| 7 / 8 | 1-SM (1,1) / 2-SM (2,1), TileK 256, direct-store epilogue | (128, 128), (128, 256) / (256, 128), (256, 256) |

The 64- and 192-wide widths were added on 2026-09-17 and exist only on the two
variants the cascade itself uses (`ST=1` and `ST=5`). The 2-SM and TileK 256
variants were deliberately not extended to them: a narrow-N version of a
variant that lost everywhere in the tile sweep would only enlarge the binary.
`TileN = 64` on this path is a measured negative and is kept only so a future
sweep does not have to rebuild it.

Build-time variables (`TORCH_CUDA_ARCH_LIST`, `CUDA_HOME`, `CUTLASS_DIR` /
`BSGEMM_CUTLASS_DIR`) are documented in the README's Install section.
The swap-AB tile / non-swap choice is the Python cascade `MOE_SWAP_BLOCK_N_CASCADE` on routed rows per active expert (`moe_swap_ab_block_n(M, E, topk)`); `FSO_SWAP_BN=16|32|64|0` forces one tile (0 = non-swap) for A/B runs.
