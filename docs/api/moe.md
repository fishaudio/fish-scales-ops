# `fish_scales_ops.moe` — public API

`fso.moe` is the stable MoE interface: `prepare_experts`, `layer`,
`transient_bytes`, `supported` and `describe`, the `MoeExperts` handle and
`FORMATS`. One torch custom op runs the architecture's chain, so the same calls
work on sm_90, sm_100/103 and sm_120/121. Changes to these names and arguments go
through a deprecation release.

The pieces the chains are built from — the per-architecture layer entries, the
sm_120/121 block, the grouped GEMMs, the routing builders and combines, the
router, the memory planners and the route queries — are in `fso.compat` and are
documented in [`compat.md`](compat.md), *MoE per-step pieces*. Their names keep
working there, but they are building blocks whose arguments follow the kernels
and change when a kernel or a route changes. Before fish-scales-ops 0.2.0 they
were exported from `fso.gemm`; `fso.gemm.<name>` still resolves to the same
object as `fso.compat.<name>` and raises a `DeprecationWarning`, and
fish-scales-ops 0.3.0 removes `fso.gemm`.

**No performance numbers in this file**; reference numbers live in
[`../perf/`](../perf/README.md). Code is the source of truth for signatures:
`python/fish_scales_ops/moe/__init__.py`, with the implementations in
`python/fish_scales_ops/gemm/{fp8,mxfp8}.py`; the `torch.ops` schemas are in
`csrc/gemm/bindings.cpp`, except `fish_scales_ops::moe_layer`, which
`moe/__init__.py` registers as a `torch.library.custom_op`. Constraints, the
`torch.ops` schemas, the CUDA-graph rules and the environment knobs are shared
with the dense ops and live in [`compat.md`](compat.md).

## Quick start

```python
import torch, fish_scales_ops as fso

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

`fso.moe.layer` runs the architecture's chain: `moe_layer_fp8_sm90` on sm_90, and
the masked-slab MXFP8 chain behind `moe_layer_mxfp8_sm120` on sm_120/121 and
sm_100/103. Both are in `fso.compat` with the per-step ops they compose
([`compat.md`](compat.md), *MoE per-step pieces*).
`bench/gemm/python/bench_moe_qwen3_30a3.py` (`fso_mxfp8_layer`) drives the steps
directly; it is where the sm_100/103 composition that `fso.moe.layer` runs, with
its fused FC1 and its slot-bound decode route, was developed.

## The stable interface — `fso.moe`

`fso.moe` is the MoE layer surface for every architecture. A caller prepares
each layer's local experts once, on the device that will serve them, and then
makes one call per batch; the architecture dispatch happens inside a torch
custom op, so the caller never names an architecture and nothing falls back.
The per-arch entries — `moe_layer_fp8_sm90` and the masked-slab MXFP8 chain
behind `moe_layer_mxfp8_sm120` — are the implementation behind it and live in
`fso.compat` ([`compat.md`](compat.md), *MoE per-step pieces*), where their names
keep working and their arguments follow the kernels.

```python
experts = fso.moe.prepare_experts(w13, w2, format="bsfp8", sw13=sw13, sw2=sw2)       # load time
out     = fso.moe.layer(hidden, experts, topk_ids, topk_w, bias=None, bias_scale=None)  # per call
ok      = fso.moe.supported("mxfp8")   # this device, or arch="sm_90", "sm_103", 120, (12, 0)
text    = fso.moe.describe()           # the architecture matrix, for logs and error messages
reserve = fso.moe.transient_bytes(experts, max_tokens, topk)  # bytes one call allocates; reserve for the largest bucket
```

**Preparing the experts.** `prepare_experts(w13, w2, *, format, sw13=None, sw2=None)`
takes this rank's local experts in the layout the checkpoint dialect stores:
`w13 [E_local, 2 * I_local, H]` with the gate rows first (`[gate; up]`) and
`w2 [E_local, H, I_local]`, on the CUDA device that will serve them. `format`
names what the caller holds, and the table says what each architecture does
with it.

| `format` | what the caller holds | sm_90 (H200) | sm_100 / sm_103 (B200, B300) | sm_120 / sm_121 (RTX 5090) |
|---|---|---|---|---|
| `"bsfp8"` | float8_e4m3fn weights with fp32 128×128 block scales `sw13 [E, 2I/128, H/128]` and `sw2 [E, H/128, I/128]`, weight = fp8 value × block scale (the `bsgemm-moe` dialect) | kept as they are (validated, made contiguous if they are not); kind `"bsfp8"` | each expert dequantized block by block to bf16 and requantized to MXFP8 1×32 with the FC1 rows interleaved; kind `"mxfp8"` | same as sm_100 |
| `"mxfp8"` | bf16 weights (the `mxfp8` dialect keeps its experts bf16 in the checkpoint) | `NotImplementedError`: sm_90 has no MXFP8 hardware | quantized to MXFP8 1×32 with the FC1 rows interleaved; kind `"mxfp8"` | same as sm_100 |
| anything else (bf16 experts, int8, fp4, …) | — | `NotImplementedError` for a known dialect, `ValueError` for an unknown string; no fallback | same | same |

It returns an `fso.moe.MoeExperts`, a frozen handle with the fields `kind`,
`arch` (the compute capability it was prepared on, as `major * 10 + minor`),
`num_experts`, `hidden`, `inter`, `w13`, `sw13`, `w2`, `sw2` and
`w13_interleaved`. The tensors inside are in the layout the architecture's
kernels read, and the MXFP8 scale handles are arch-native (*Scale layouts* in
[`compat.md`](compat.md)), so a handle is valid only on the architecture family
it was prepared on, and `layer` refuses a handle from another family. The constraints are
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

**Memory.** `layer` runs every bucket as one call on every architecture. The
transient memory of a call is the caller's to reserve, and
`fso.moe.transient_bytes(experts, tokens, topk)` returns it exactly: the routing
tensors, the gathered and quantized activation, the intermediates of both grouped
GEMMs and the `[tokens, H]` result, for the route the layer takes at that token
count on this device (the layer and the query read one host-side plan). Each
tensor is charged as the caching allocator can charge it — 512-byte blocks, and up
to 1 MiB more for a tensor above 1 MiB that a cached block serves without splitting
— so the figure is an upper bound, tight to about 1 MiB per large tensor. It does not count the inputs,
the weights, a `bias` or the library's persistent pools. The intermediates are
freed when the call returns, so one reservation for the largest bucket a forward
can carry covers every MoE layer of a model. The two layouts scale differently:
sm_90's expert-sorted layout grows with `tokens · topk` plus per-expert padding,
while the masked slab of sm_100/103 and sm_120/121 grows with `E_local · tokens`,
which is the larger of the two whenever `E_local` exceeds `topk`. A bucket whose
tensors do not fit raises the allocator's out-of-memory error.

**The op and its dispatch.** `layer` validates the arguments against the handle
and calls `torch.ops.fish_scales_ops.moe_layer(hidden, w13, sw13, w2, sw2,
topk_ids, topk_w, bias, bias_scale, kind, w13_interleaved)`, a
`torch.library.custom_op` whose body dispatches on the handle's kind and on the
device's architecture:

| kind | architecture | what runs |
|---|---|---|
| `"bsfp8"` | sm_90 | `moe_layer_fp8_sm90` (the expert-sorted contiguous layout, swap-AB or non-swap picked from `M`), then the bias fold above |
| `"mxfp8"` | sm_120 / sm_121 | the masked-slab chain of `moe_layer_mxfp8_sm120`, unchanged, with `w13_interleaved` from the handle and the fused combine allowed (see below); one call per bucket |
| `"mxfp8"` | sm_100 / sm_103 | the same chain, with the routing kernel asked for exactly the tensors this architecture's grouped routes read: per GEMM, `mxfp8_grouped_slot_possible` and `mxfp8_grouped_problem_shapes_consumed` (the FC1 asked under the kernel it will run) choose between the packed active-expert list for the slot-bound decode route and the per-group problem shapes for the pointer-array cascade, which is the composition `bench/gemm/python/bench_moe_qwen3_30a3.py` (`fso_mxfp8_layer`) drives. The fused FC1 is taken where `mxfp8_grouped_swiglu_fused_route` says so and there is no fused combine; one call per bucket |
| anything else | any | `NotImplementedError` naming the architecture and the format to use instead, or `ValueError` for an unknown kind |

*The fused combine.* On sm_120/121 the fused-combine FC2 adds each routed row
into its token's output row with atomic reductions instead of storing the
down-projection slab for a separate combine kernel. Where
`moe_layer_fused_combine_engages_sm120` takes it (the buckets whose slab would
dominate the layer's transient memory) the slab is never allocated, but the
order of the adds follows the order in which the CTAs finish, so those buckets
are stable in aggregate and not bit-reproducible run to run.
`fso.moe.layer` therefore keeps it opt-in, the same default as
`moe_layer_mxfp8_sm120`: `FSO_MOE_FUSED_COMBINE=1` allows it and the engagement
rule then decides per bucket; unset, every bucket keeps the slab and the combine
kernel and is reproducible. `fso.moe.transient_bytes` follows the setting, so a
caller that allows the fused combine reserves less for the buckets that take it.

*Capture and compile.* The op can be captured into a CUDA graph after one eager
call of the same shape on the capturing thread, like every op in this library:
that call creates the per-thread pools and, on sm_90, JIT-compiles the grouped
kernels. Every host-side decision — the static GEMM hints, the
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
returns the same matrix as text with this device's rows marked. On sm_90 its
last line names the compiler of the JIT that builds the sm_90 kernels
(`torch.ops.fish_scales_ops.jit_compiler_sm90()`: the bundled NVRTC 13.0 and its
path), or the error that keeps that compiler from loading; `describe()` itself
never raises.

`tests/gemm/unit/test_moe_unified.py` runs on every architecture and skips what
the device cannot run. It asserts that `fso.moe.layer` is `torch.equal` to the
per-arch entry for the same weights and inputs — `moe_layer_mxfp8_sm120` on
sm_120/121, `moe_layer_fp8_sm90` on sm_90, and on sm_100/103 the bench's
composition — across decode and prefill token counts, both id dtypes, skipped
ids, the bias forms, a captured graph and a `torch.compile` call. It also
measures the `"bsfp8"` double quantization, checks the fused-combine rule and its
switch, and checks the refusals and the capability matrix.

## Per-step pieces

The per-architecture entries behind `fso.moe.layer` (`moe_layer_fp8_sm90`,
`moe_layer_mxfp8_sm120` and the whole-block `moe_block_mxfp8_sm120`), the grouped
GEMMs of both layouts, the routing builders and combines, the router, the memory
planners and the route queries are `fso.compat.<name>`. Their contracts — the
masked slab and expert-sorted layouts, the fused FC1 and the fused combine, the
slot-bound decode route and the routing-supplied problem shapes of sm_100/103,
the multi-CTA routing builder — are in [`compat.md`](compat.md), *MoE per-step
pieces*.
