# `fish_scales_ops.compat` — the explicit ops and the MoE per-step pieces

`fso.compat` keeps, under the names existing callers use, the eleven explicit
dense ops and the thirty-four MoE per-step pieces. New code uses the two stable
entries instead. Each runs the architecture dispatch inside one torch custom op
and never hands the caller a scale layout: `fso.dense` for a dense linear
([`dense.md`](dense.md)) and `fso.moe` for an MoE layer ([`moe.md`](moe.md)).

What this namespace promises:

* Every name keeps resolving to the same object.
* The dense ops keep their signatures and their behaviour. They are
  format-specific: the caller picks block-FP8 or MXFP8, and the scale tensors
  they take and return are architecture-native (*Scale layouts* below).
* The MoE pieces are building blocks whose arguments follow the kernels: their
  arguments, layouts and architecture coverage change when a kernel or a route
  changes (*MoE per-step pieces* below).

Before fish-scales-ops 0.2.0 these names were exported from `fso.gemm`.
`fso.gemm.<name>` still resolves to the same object as `fso.compat.<name>` and
raises one `DeprecationWarning` per name per process, naming the new home;
fish-scales-ops 0.3.0 removes `fso.gemm`. The implementation modules stay where
they were (`python/fish_scales_ops/gemm/{fp8,mxfp8,bf16}.py`), and importing them
by their module path raises no warning.

This file also holds the reference sections that every GEMM and MoE op shares,
`fso.dense` and `fso.moe` included: the constraints, the generated `torch.ops`
schemas, CUDA-graph compatibility and the environment variables. **No
performance numbers in this file**; reference numbers live in
[`../perf/`](../perf/README.md) and the acceptance rule in
[`../perf/README.md`](../perf/README.md) §7. Code is the source of truth for
signatures: the Python wrappers are in
`python/fish_scales_ops/gemm/{fp8,mxfp8,bf16}.py`, re-exported by
`python/fish_scales_ops/compat/__init__.py`, and the `torch.ops` schemas are in
`csrc/gemm/bindings.cpp`.

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
| sm_90 (H200) | deep_gemm WGMMA kernels, JIT-compiled in-process at first call by the NVRTC 13.2.78 bundled with the package (`FSO_JIT_NVRTC_LIB`); FP32 scales | not available (`NotImplementedError`) | block-FP8, expert-sorted contiguous layout (`moe_layer_fp8_sm90`) and a masked-layout variant |
| sm_120 / sm_121 (RTX 5090, RTX PRO 6000) | CUTLASS `Sm120BlockScaledKernel`; UE8M0 scales packed into int32 words | CUTLASS `Sm120BlockScaledKernel` | MXFP8, masked slab layout (`moe_layer_mxfp8_sm120`) |
| sm_100 / sm_103 (B200 / B300) | since 2026-09-05: the MXFP8 tcgen05 path with each 1×128 UE8M0 scale byte replicated into its four 32-wide slots (same kernels, same bytes as MXFP8) | CUTLASS tcgen05 BlockScaled behind a router: an M ≤ 64 decode row (vendored NVIDIA CuTe-DSL split-K / persistent kernels, swap-AB, 8/16/32-wide token tile to M = 32 and a 64-wide token tile on the 2-CTA 256-row weight tile above it, long-K narrow-N cells above M = 32 excepted) in front of three tiers (cuBLAS `scaled_mm`, CuTe DSL persistent kernel, C++ cascade) | MXFP8, masked slab layout — same Python surface as sm_120, CUTLASS pointer-array (grouped) tcgen05 kernels |

A caller reaches a dense linear through one surface on every architecture,
`fso.dense` ([`dense.md`](dense.md)), and the grouped MoE layer through another,
`fso.moe` ([`moe.md`](moe.md)). The explicit ops of the first two columns and the
per-architecture MoE entries of the last are their implementation, and this file
documents them.

Scale tensors are **arch-native and opaque**. Quantize on the device arch the
GEMM runs on; a scale tensor produced on one arch generation is not valid on
another (the layouts are listed under *Scale layouts* below).

## Quick start

For a dense linear on any architecture, `fso.dense.prepare_weight` once per
weight and `fso.dense.linear` per call is the stable route
([`dense.md`](dense.md)). The explicit ops below are for a caller that picks the
format and handles the scale forms itself.

```python
import torch, fish_scales_ops as fso

# ---- block-FP8 1x128 / 128x128 (sm_90, sm_120, sm_100/103) -------------------
# Weights, once at load time. On sm_120 and sm_100/103 the scales must be
# UE8M0 (powers of two); both quantizers default to that there and to FP32
# scales on sm_90.
wq, sw = fso.compat.quantize_128x128_fp8(w_bf16)          # fp8 [N, K], fp32 [N/128, K/128]
sm = torch.cuda.get_device_capability(0)[0]
if sm >= 10:
    sw = fso.compat.repack_fp8_wgt_scales(sw)             # pre-pack once; int32, arch-native layout

# Activations, per call.
if sm >= 10:
    xq, sx = fso.compat.quantize_1x128_fp8_packed(x_bf16) # fused quantize + pack, UE8M0
else:
    xq, sx = fso.compat.quantize_1x128_fp8(x_bf16)        # fp32 scales for deep_gemm
y = fso.compat.linear_fp8(xq, wq, sx, sw)                 # bf16 [M, N]
# Pass both scales pre-packed (int32) or both FP32. FP32 scales are packed inside
# every call, without a device sync, so either form can be captured into a CUDA
# graph; pre-packing skips that per-call work.

# ---- MXFP8 1x32 (sm_100/103, sm_120) -----------------------------------------
if sm in (10, 12):
    wqm, swm = fso.compat.quantize_1x32_fp8(w_bf16)       # fp8 [N, K] + opaque int32 UE8M0 scales
    xqm, sxm = fso.compat.quantize_1x32_fp8(x_bf16)
    ym = fso.compat.linear_mxfp8(xqm, wqm, sxm, swm)      # bf16 [M, N]
```

The stable dense entry has its own page, [`dense.md`](dense.md), and so does
the MoE layer, [`moe.md`](moe.md).

## Ops

### Dense

| function | in | out | notes |
|---|---|---|---|
| `quantize_1x128_fp8(x, use_ue8m0=None)` | bf16 `[..., K]`, `K % 128 == 0` | (fp8 e4m3 `[..., K]`, fp32 `[pad(M,4), K/128]` K-major) | `None` → UE8M0 on sm_100/103 and sm_120/121, FP32 on sm_90, the rule of `quantize_128x128_fp8`. The Blackwell GEMMs need UE8M0 scales (powers of two). With `False` there, `linear_fp8` truncates every FP32 scale that is not a power of two to the power of two below it without an error, on sm_120/121 and sm_100/103 alike, while the pre-pack ops raise on such a scale |
| `quantize_1x128_fp8_packed(x)` | bf16 `[..., K]`, `K % 128 == 0` | (fp8 `[..., K]`, int32 packed scales, arch-native) | UE8M0 always; one kernel on `K % 512 == 0`, two kernels otherwise; sm_120 and sm_100/103 |
| `quantize_128x128_fp8(w, use_ue8m0=None)` | bf16 `[N, K]` | (fp8 `[N, K]`, fp32 `[ceil(N/128), ceil(K/128)]`) | `None` → UE8M0 on sm_100/103 and sm_120, FP32 on sm_90 |
| `repack_fp8_act_scales(sx_f32)` | fp32 `[pad(M,4), K/128]` UE8M0-exact | int32 packed, arch-native | sm_120 and sm_100/103; rejects non-power-of-two scales (one device sync — a weight-load-time op). The torch op's `check=False` form skips the check and the sync; the per-call packing paths use it |
| `repack_fp8_wgt_scales(sw_f32)` | fp32 `[N/128, K/128]` UE8M0-exact | int32 packed, arch-native (one scale per N row) | same |
| `linear_fp8(x_fp8, w_fp8, sx, sw)` | fp8 `[M, K]`, fp8 `[N, K]`, scales fp32 (packed per call) or int32 (pre-packed) | bf16 `[M, N]` | `K % 128 == 0`; `N % 128 == 0` on sm_120 and sm_100/103; sm_90 takes FP32 scales only. On sm_120/121 both scales must be FP32 or both int32, and a mix raises `RuntimeError`; on sm_100/103 the wrapper accepts a mix and packs the FP32 one. FP32 scales are packed inside every call without a check and without a device sync, so the call can be captured with either scale form; pre-packed int32 scales skip that packing (quick start) |
| `linear_bf16(x, w)` | bf16 `[M, K]`, bf16 `[N, K]` | bf16 `[M, N]` | bf16 in and out, block-FP8 inside: both operands are quantized to block-FP8 on every call, so the precision is block-FP8's, not BF16's. Convenience only, for when both operands change every call. On sm_100/103 it is composed from the public ops (the UE8M0 quantizers, an unchecked weight-scale repack and the block-FP8 GEMM); it does not synchronize the device and can be captured after one eager call |
| `linear_qx(x_bf16, w_fp8, sw)` | bf16 `[M, K]`, fp8 `[N, K]`, fp32 `[N/128, K/128]` | bf16 `[M, N]` | block-FP8 path with the activation quantized inside the op; `sw` is the FP32 output of `quantize_128x128_fp8` on the same device. sm_90 and sm_120/121, where the result is bit-identical to `quantize_1x128_fp8` + `linear_fp8` (`tests/gemm/unit/test_linear_qx.py`); refused on sm_100/103 with a `RuntimeError` that points at `quantize_1x128_fp8_packed` + `linear_fp8` |
| `quantize_1x32_fp8(x)` | bf16 `[..., K]`, `K % 128 == 0` | (fp8 `[..., K]`, opaque int32 scales, arch-native) | MXFP8, UE8M0 always; sm_100/103 and sm_120 |
| `silu_chunk_mul_quantize_1x32_fp8(gu)` | bf16 `[..., 2*INTER]` (`gate ‖ up`), `INTER % 128 == 0` | (fp8 `[..., INTER]`, opaque int32 scales) | `silu(gate) * up` quantized without materialising the BF16 intermediate; sm_100/103 and sm_120 |
| `linear_mxfp8(x_fp8, w_fp8, sx, sw)` | fp8 `[M, K]`, fp8 `[N, K]`, opaque int32 scales from `quantize_1x32_fp8` on the same arch | bf16 `[M, N]` | `K % 128 == 0`, `N % 128 == 0`; sm_100/103 routes through the decode row and the three tiers, sm_120 through the CUTLASS cascade |

The same objects live in the implementation modules
`fish_scales_ops.gemm.fp8`, `fish_scales_ops.gemm.mxfp8` and
`fish_scales_ops.gemm.bf16`, and, until 0.3.0, under the deprecated
`fso.gemm.<name>` path.

#### Scale layouts

The FP32 scale tensors carry PyTorch metadata that does not describe their
physical byte order; the packed int32 tensors are opaque handles. Never
`.contiguous()`, transpose or slice them, and never move them between arch
generations. A checkpoint that stored an sm_120/121 MXFP8 weight scale with
`.contiguous()` (a safetensors round trip does) holds the right values in the
wrong byte order; `fso.dense.prepare_weight` accepts that form and restores the
K-major layout, while `linear_mxfp8` reads it as it is and returns a wrong result
without an error ([`dense.md`](dense.md), *prepare_weight*).

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

## MoE per-step pieces

Every name in this section is `fso.compat.<name>`. These are the pieces the
`fso.moe` chains are built from ([`moe.md`](moe.md)); their names keep working,
and their arguments follow the kernels (see the top of this file).

Two layouts, one per arch. Both keep every per-expert row count **on the
device**: the host never reads routing results, so a layer captured once into a
CUDA graph replays correctly for any routing (*CUDA-graph compatibility* below).

**sm_120 and sm_100/103 — masked slab layout (MXFP8).** Every expert `g` owns
a slab of `m_cap` rows; `masked_m[g]` (device int32) says how many are valid;
rows at or past it hold undefined bytes in every tensor. The layer is six
kernels, and the Python surface is identical on both arch families — only the
opaque scale byte layout differs (see the two `grouped MXFP8 … scales` rows in
*Scale layouts* above; the shapes in the table below are the sm_120 ones):

| step | function | in → out |
|---|---|---|
| whole block (sm_120/121) | `moe_block_mxfp8_sm120(hidden, router_weight, w13_fp8, sw13, w2_fp8, sw2, *, topk, renormalize=True, num_token_non_padded=None, expert_map=None, shared_w13_fp8=None, shared_sw13=None, shared_w2_fp8=None, shared_sw2=None, shared_gate_in_router=False, w13_interleaved=False, fused_combine=False, shared_out=None, shared_gate=None, overlap_shared=None, out=None)` | bf16 `[M, HIDDEN]` + the router weight + the routed experts (+ the shared expert) → bf16 `[M, HIDDEN]`; the router, the routed experts, the shared expert and the gated add that joins them, in one capture-safe call. `topk` is the router's, so at most 8. See *the composed sm_120 block* below |
| router | `moe_router_topk(hidden, router_weight, topk, *, renormalize=True, with_shared_gate=False, num_token_non_padded=None, expert_map=None)` | bf16 `[M, HIDDEN]` × bf16 `[num_experts(+1), HIDDEN]` → (int32 `[M, topk]`, fp32 `[M, topk]`, fp32 `[M]`, empty unless `with_shared_gate`); the logit GEMM (cuBLAS) followed by the kernel below, so the same limits apply. Both router entries run on every architecture, sm_90 included |
| router, logits given | `moe_topk_from_logits(logits, topk, *, renormalize=True, with_shared_gate=False, num_token_non_padded=None, expert_map=None)` | `[M, num_experts(+1)]` bf16/fp16/fp32 → the same three tensors, one launch: softmax, top-k, renormalise, the padded-row sentinel, the expert-parallel id remap and the shared expert's sigmoid gate. `topk` must lie in `[1, min(8, num_experts)]` and `num_experts` in `[1, 1024]`, or the op raises; a model with a wider top-k cannot use this router. Every argument after `topk` is keyword-only |
| unified entry (sm_120/121) | `moe_layer_mxfp8_sm120(hidden, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w, *, bias=None, bias_scale=None, out=None, join=None, w13_interleaved=False, fused_combine=False)` | bf16 `[M, HIDDEN]`, per-expert fp8 weights with their opaque int32 scales, int32 or int64 `[M, topk]`, fp32 `[M, topk]` → bf16 `[M, HIDDEN]`; runs the six kernels below with every host-side decision taken from the argument shapes, so one call captures per graph bucket. `bias` / `bias_scale` are `moe_combine`'s; `out` is a caller-owned bf16 `[M, HIDDEN]` destination, returned when given; `join` is an internal hook of the block entry. See *the composed sm_120 layer* below for the expert-parallel id contract, the `m_cap` rule and the memory a caller reserves |
| memory to reserve (sm_100/103, sm_120/121) | `moe_layer_transient_bytes_mxfp8(tokens, num_experts, topk, hidden, inter, *, w13_interleaved, fused_combine=False)` (`w13_interleaved` is required) | the exact device bytes one masked-slab layer call allocates for a bucket of `tokens` rows on this device; `fso.moe.transient_bytes` calls it for an `"mxfp8"` handle |
| routing | `moe_build_routing(topk_ids, num_groups, m_cap, *, with_slots=False, problem_shapes_for=None, topk_w=None)` | int32 `[M, topk]` → (`masked_m [G]`, `row_map [G*m_cap]`, `slot_of_flat [M*topk]`), plus `slot_to_expert [G]` when `with_slots=True`, plus a list of int32 `[G, 3]` per-group `(rows, N, K)` tensors, one per `(N, K)` pair in `problem_shapes_for` (at most four), plus `weight_of_slot` fp32 `[G*m_cap]` (each valid slot's combine weight, what the sm_120 fused-combine FC2 reads) when `topk_w` is given, in that order; `topk_w` and `problem_shapes_for` are mutually exclusive; `G <= 1024`, `m_cap % 4 == 0`, `m_cap >= M`. An entry of `topk_ids` outside [0, `num_groups`) is skipped: no group counts it, it takes no slab row, and its `slot_of_flat` entry is −1, which the gather-quantize, the SwiGLU requantize and the combine all read as "no routed row". Ask for `with_slots` only when `mxfp8_grouped_slot_possible(…)` says a GEMM of this layer's shape would take the sm_100/103 slot route, which is the list's only reader, and for `problem_shapes_for` only the pairs for which `mxfp8_grouped_problem_shapes_consumed(…)` answers `True` (see *the slot-bound decode route* and *routing-supplied problem shapes* below) |
| gather + quantize | `quantize_1x32_grouped_gather_fp8(x, slot_of_flat, topk, num_groups, m_cap)` | bf16 `[M, K]` → (fp8 `[G, m_cap, K]`, int32 `[G, K/128, m_cap]`) |
| gate_up | `linear_mxfp8_grouped_masked(a_fp8, w_fp8, sa, sw, masked_m, expected_m, max_active_groups=0, slot_to_expert=None, problem_shapes=None)` with the gathered activation as `a_fp8, sa` and the FC1 weights `w13` as `w_fp8, sw` | → bf16 `[G, m_cap, 2*INTER]` |
| SwiGLU + quantize | `silu_chunk_mul_quantize_1x32_grouped_fp8(gu, slot_of_flat, pairwise=False)` | → (fp8 `[G, m_cap, INTER]`, int32 `[G, INTER/128, m_cap]`); `pairwise=True` reads gate/up-interleaved rows (*the fused FC1* below) |
| down | `linear_mxfp8_grouped_masked(a_fp8, w_fp8, sa, sw, masked_m, expected_m, max_active_groups=0, slot_to_expert=None, problem_shapes=None)` with the SwiGLU output as `a_fp8, sa` and the FC2 weights `w2` as `w_fp8, sw` | → bf16 `[G, m_cap, HIDDEN]` |
| combine | `moe_combine(dn, slot_of_flat, topk_w, *, bias=None, bias_scale=None, out=None)` | → bf16 `[M, HIDDEN]`, `out[t] = Σ_j topk_w[t,j] · dn[slot_of_flat[t*topk+j]] + bias_scale[t] · bias[t]`. `bias` is a shared-expert output and `bias_scale` its per-token gate, folded in so the block does not pay another read-modify-write of the whole result; `out` is a caller-owned destination (a data-parallel or reduce-scatter buffer). The kernel reads `slot_of_flat` and `topk_w` before its PDL wait (they are routing outputs, two or more launches upstream in every chain of this library), so a direct caller must not produce either of them in the kernel launched immediately before the combine on the same stream; only `dn` may come from the immediately preceding kernel. On sm_120 the launch uses 128-thread blocks while the grid is at most 256 such blocks (M ≤ 128 at HIDDEN = 2048), so that one combine CTA fits beside two resident FC2 CTAs and PDL can start it early |
| down + combine, fused (sm_120/121) | `linear_mxfp8_grouped_masked_combine(a_fp8, w2_fp8, sa, sw2, masked_m, row_map, weight_of_slot, out, expected_m, max_active_groups=0)` | → `out`, bf16 `[M, HIDDEN]`. Replaces the down GEMM and the combine: each row is scaled by its slot's weight and **added into** its token's row of `out`, which the caller pre-fills (zero, or the shared expert's gated rows). `row_map` and `weight_of_slot` come from `moe_build_routing(..., topk_w=topk_w)`. The adds are atomic, so the result is not bit-reproducible run to run (*the fused combine* below); refused on every other architecture |
| weights (offline) | `quantize_moe_weights_1x32_fp8(w, w13_interleave=False)` | bf16 `[G, N, K]` → (fp8 `[G, N, K]`, int32 `[G, K/128, N]`); `N % 128 == 0`, `K % 128 == 0`; `w13_interleave=True` is for FC1 weights only (*the fused FC1* below) |

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
constrained to `0 <= max_active_groups <= G`. On sm_120/121 the grouped
cascade reads it for its decode-band tile rule (`FSO_MOE_DECODE_GATE` under
*Environment variables* below); `0` keeps that cascade's `m_cap`-only gate.

**The composed sm_120 layer.** `moe_layer_mxfp8_sm120` is the sm_120/121 entry
behind `fso.moe.layer`: one call, the six kernels above, and no host-visible
dependence on the routing. It is the sm_120/121 twin of `moe_layer_fp8_sm90` and
refuses on any other architecture, naming it. sm_100/103 serves its decode band
from a slot-bound route and the rest from a pointer-array cascade, and both read
routing tensors (the packed active-expert list, the per-group problem shapes)
that this entry does not ask the routing kernel for; `fso.moe.layer` runs the
same chain there with exactly those tensors requested. Besides the routed
experts the entry takes `bias` and `bias_scale`, folded into the combine as in
`moe_combine`, and `out`, a caller-owned bf16 `[M, HIDDEN]` destination such as
a collective's buffer; the result is the rank's partial sum and is final when
the call returns on the current stream.

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
at all.) The consequence is that the transient tensors scale with
`num_experts · m_cap`; `moe_layer_transient_bytes_mxfp8(tokens, num_experts, topk,
hidden, inter, w13_interleaved=…)` returns the exact bytes of one call.

*One call per bucket.* The memory a bucket needs is the caller's to reserve, from
`moe_layer_transient_bytes_mxfp8` (or `fso.moe.transient_bytes` for a prepared
handle) evaluated at the largest bucket a forward can carry, and a bucket whose
tensors do not fit raises the allocator's out-of-memory error. The structural
answer to the size of that reservation is a contiguous (expert-sorted) sm_120
entry that sizes its activation by the routed rows rather than by capacity, as
the sm_90 layer does; it would cut the reservation by roughly `num_experts /
topk`, and it does not exist yet.

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

**The fused combine on sm_120/121.**
`fused_combine=True` lets the FC2 add each row into its token's output row with
eight-byte atomic reductions (`linear_mxfp8_grouped_masked_combine`), instead of
storing the bf16 `[G, m_cap, HIDDEN]` slab that `moe_combine` then reads back. It
removes that slab and the combine launch, so a bucket that takes it needs less
transient memory, and `moe_layer_transient_bytes_mxfp8(..., fused_combine=True)`
reports the smaller figure. The scatter runs on the store warp and the spare fourth
TMA warp, so the atomic adds overlap the next tile's mainloop the way the TMA store
they replace does; `FSO_MOE_SCATTER_WARP=0` moves them to the math warps for an
A/B. The measured memory and time of both routes are in
[`../perf/layer/sm120.md`](../perf/layer/sm120.md).

Two consequences for a caller. The route engages only where the slab dominates the
footprint (`moe_layer_fused_combine_engages_sm120(m, topk, hidden)`), so
`fused_combine=True` can be set once for every bucket and the small ones stay bit
identical. And where it does engage the adds are atomic, so the result is stable in
aggregate but not bit-reproducible, which is why it is off by default and why
`tests/gemm/unit/test_mxfp8_fused_combine_sm120.py` checks it against the
deterministic path to a tolerance and reports the run-to-run spread rather than
asserting equality. The op accumulates into the
output, so the layer pre-fills it with the shared expert's gated rows or with zero;
that is also what gives a token whose every routed entry was skipped the right
value.

**The composed sm_120 block.** `moe_block_mxfp8_sm120` is the next surface out. Its weights are the raw tensors, not a `prepare_experts` handle, so the
row order travels separately: pass `w13_interleaved=experts.w13_interleaved` when the weights come
from `fso.moe.prepare_experts` (on sm_120/121 it interleaves gate and up rows whenever the fused FC1
can serve the shape), or the FC1 reads the interleaved rows as `[gate; up]` and the output is wrong
without an error. Beyond that, the block is
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
capture raises a `RuntimeError` naming the knob. `overlap_shared=False` per call, or
`FSO_MOE_BLOCK_OVERLAP=0` process-wide (read once), keeps the sequential order.

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
straight into the buffer the scatter or reduce-scatter already owns.

*Determinism.* Run-to-run bit-identical output and graph replay equal to eager are
asserted for both shape families and both
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
arch, it keeps the single-CTA kernel. `FSO_MOE_ROUTING_MULTI=0` restores the
single-CTA kernel everywhere, a nonzero integer selects the multi-CTA kernel on
every architecture (sm_90 included) from the same threshold, and the routing
tests compare the two builders on the same draw. Three parts of its contract are
worth stating because a caller can observe them.

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
`mxfp8_grouped_slot_possible(m_cap, n_w, k, num_groups, max_active_groups=0, fused_swiglu=False)`
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
max_active_groups=0, fused_swiglu=False)` per GEMM — the complement of
`mxfp8_grouped_slot_possible`, from the same route rule — and requests the
shapes for the GEMMs that consume them, which is what the benches and the
layer test do. The two queries are complements only when both are asked with
the same `fused_swiglu` flag, because the fused-SwiGLU FC1 leaves the slot
route one step earlier than the plain grouped GEMM (see *the slot-bound
decode route* above): a layer asks both for its FC1 with the flag set to
whether it will call the fused op, and for its FC2 with it clear, and then
per GEMM exactly one of the slot list and the problem shapes is requested. The slot route derives
its grid from `masked_m` and the slot list and ignores the tensor. On sm_120/121,
which has neither route, `linear_mxfp8_grouped_masked` validates `problem_shapes`
and `slot_to_expert` and ignores them, while `linear_mxfp8_grouped_masked_swiglu`
refuses a non-empty one of either with a `RuntimeError`; a caller composing the
fused FC1 there passes neither (the route queries answer `False` on sm_120/121,
so a caller driven by them never asks for them). Omitting the argument keeps the
preparation kernel and changes nothing for existing callers.

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
live tile assigned to a CTA whose first tile happens to be dead.

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
| gate_up + SwiGLU + quantize | `linear_mxfp8_grouped_masked_swiglu(a_fp8, w13_fp8, sa, sw13, masked_m, expected_m, max_active_groups=0, slot_to_expert=None, problem_shapes=None)` | → (fp8 `[G, m_cap, INTER]`, int32 per-group Sm1xx atom slabs) — the same pair `silu_chunk_mul_quantize_1x32_grouped_fp8` returns, so `down` consumes it unchanged |
| weights (offline) | `quantize_moe_weights_1x32_fp8(w13, w13_interleave=True)` | as above, with the FC1 rows re-ordered |
| weights (already quantized) | `interleave_w13_fp8(w13_fp8, sw13)` | → the same bytes in the interleaved row order |
| load-time router | `mxfp8_grouped_swiglu_available(n_w, k)` | → bool: can this shape use the fused FC1 at all? |
| per-call router | `mxfp8_grouped_swiglu_fused_route(m_cap, n_w, k, num_groups, max_active_groups=0)` | → bool: does this call take it? |

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
max_active_groups=0)` is the per-call one. On sm_100/103 both grouped routes
carry a fused FC1, so with the knob unset it answers yes on both sides of the
route boundary; what it really decides, through the same `slot_route` the
unfused FC1 asks, is WHICH fused kernel the call lands on. On sm_120/121, which
has one grouped route, the per-call answer is the load-time one. A caller that
ends up on the unfused FC1 anyway (because it set `FSO_FC1_FUSED=0` for only
part of its model, or because a shape has no fused instantiation) must pass
`pairwise=True` to `silu_chunk_mul_quantize_1x32_grouped_fp8` if its weights are
interleaved; that kernel then reads `gate_i` at column `2i` and `up_i` at column
`2i+1` instead of from the two halves of the row. The flag is accepted on
sm_100/103 and sm_120/121, the architectures with a fused FC1 (the sm_120/121
layer passes it whenever it holds interleaved weights and does not take the
fused FC1), and raises `NotImplementedError` naming the architecture anywhere
else. Like the weight layout, its *value* cannot be inferred: the wrong one
multiplies the wrong pairs together and raises nothing.

`FSO_FC1_FUSED` switches the whole feature: `0` never uses the fused FC1 (and
`mxfp8_grouped_swiglu_available` then answers false, so a caller driven by it
keeps the stacked weight layout as well), unset applies the rule above, and `1`
uses it wherever it is legal. It is read once per process, so it cannot change
between a capture and its replays.

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
| memory to reserve | `moe_layer_transient_bytes_sm90(tokens, num_experts, topk, hidden, inter)` | the exact device bytes one `moe_layer_fp8_sm90` call allocates for a bucket of `tokens` rows (the expert-sorted layout, sized by the routed rows); `fso.moe.transient_bytes` calls it for a `"bsfp8"` handle |
| unified entry | `moe_layer_fp8_sm90(hidden, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w)` | bf16 `[M, HIDDEN]`, per-expert fp8 weights with fp32 `[E, N/128, K/128]` scales, int32 `[M, topk]`, fp32 `[M, topk]` → bf16 `[M, HIDDEN]`; picks the swap-AB GEMMs with an activation tile of 16 / 32 rows while the routed rows per active expert (`M * topk / E`) are at most 12 / 24 (`moe_swap_ab_block_n(M, E, topk)`; M <= 768 on E=256/top-8), and otherwise the block_m = 64 gate_up GEMM followed by the swap-AB down GEMM with a 64-row activation tile on the same 64-row layout (bit-identical to the block_m = 64 down GEMM; `FSO_FC2_SWAP=0` selects that one), a host-side branch on the static `M`; on every path the gate_up GEMM computes the SwiGLU + 1x128 requantize in its epilogue, bit-identical to the GEMM + `silu_chunk_mul_quantize_1x128_sorted_sm90` pair, unless `FSO_FC1_FUSED=0`; `topk_ids` entries outside [0, E) (sglang's masked rows past `num_token_non_padded`: −1 or `num_experts`) take no expert compute and a token whose entries are all masked gets a zero row |
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
  only the exponent byte. Both quantizers' defaults and `quantize_1x128_fp8_packed`
  produce them. The pre-pack ops refuse any other FP32 scale. The per-call
  packing of FP32 scales (inside `linear_fp8`, and inside `linear_bf16` on
  sm_100/103) cannot check without a device sync, so on sm_120/121 and
  sm_100/103 alike it truncates a non-power-of-two scale to the power of two
  below it without an error. (Until
  2026-09-05 the weight quantizer, and until 2026-09-30 the activation
  quantizer, defaulted to plain `amax/448` scales there.) On sm_90 `linear_fp8`
  takes FP32 scales only and refuses the int32 pre-packed forms, and the
  pre-pack ops raise.
- **FP32 scales on sm_90.** deep_gemm reads FP32 scales directly; do not call
  `repack_fp8_*_scales` there.
- **One scale form per call.** `linear_fp8` takes both scales as FP32 or both
  as pre-packed int32. On sm_120/121 a mix raises `RuntimeError` ("sx and sw
  must both be FP32 … or both be int32"); on sm_100/103 the Python wrapper
  accepts a mix and packs the FP32 operand on every call, unchecked and without
  a device sync; sm_90 takes FP32 only.
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
  only. The DSL rows JIT-compile each configuration on its first call in a
  process, which the eager warm-up before a capture covers. The mid-band DSL
  tier loads its kernel source from `FSO_DSL_KERNEL_PATH` when that is set,
  else from the copy a wheel carries in `fish_scales_ops/_dsl/`, else from
  `3rdparty/cutlass/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py`
  in the source tree. When that file does not exist, as in an in-place
  installation whose source tree moved, or fails to load, the
  router uses the remaining tiers and the first call that reaches the tier
  prints one `fso:` line on stderr naming the tier, the reason and
  `FSO_DSL_KERNEL_PATH`. The tier stays silent only when it is switched off
  (`FSO_DISABLE_DSL`) or the optional `nvidia-cutlass-dsl` package is not
  installed.
- The sm_100/103 CuTe-DSL rows need `nvidia-cutlass-dsl` (the `sm100` extra).
  The M ≤ 64 decode row needs at least **4.5.0** and **4.8.0 is the
  recommended pin**: 4.4.2 cannot run its kernels at all, and the JIT compile
  time of these configurations differs between releases. Below the floor,
  or with the package missing entirely, the decode row disables itself and the
  router behaves exactly as it did before the row existed. A DSL below the
  floor, or vendored kernels that fail to import, print one `fso:` line on
  stderr once per process; a missing package stays silent unless `FSO_LOG=1`.
- The decode row is launched through the plain (non-TVM-FFI) CuTe-DSL calling
  convention: every eager call rebuilds the output descriptor, queries the
  current stream and marshals ten arguments through the DSL's Python-level host
  entry. A CUDA-graph-captured caller does that once, at capture, and each
  replay is one `cudaGraphLaunch`. Decode loops should capture; a caller that
  must run the op eagerly one token at a time can set
  `FSO_DISABLE_DECODE_DSL=1`, which hands the band back to the tiers behind the
  row.

## Picking the right op

| situation | op |
|---|---|
| dense linear, any architecture | `fso.dense.prepare_weight` once per weight at load time, then `fso.dense.linear` per call ([`dense.md`](dense.md)) |
| MoE layer, any architecture | `fso.moe.prepare_experts` once per layer at load time, then `fso.moe.layer` per call ([`moe.md`](moe.md)) |
| Hopper (sm_90) dense inference, cached FP8 weight | `linear_fp8` with `quantize_1x128_fp8(x)` per call, FP32 scales |
| Blackwell consumer (sm_120) dense inference, block-FP8 checkpoint | `linear_fp8` with pre-packed weight scales and `quantize_1x128_fp8_packed(x)` per call |
| Blackwell consumer (sm_120), free choice of format | `linear_mxfp8` with `quantize_1x32_fp8` (tighter quantization; [`../perf/gemm/sm120.md`](../perf/gemm/sm120.md) lists both formats) |
| Blackwell datacenter (sm_100/103) | `linear_mxfp8` with `quantize_1x32_fp8`; block-FP8 checkpoints run through `linear_fp8` on the same kernels |
| SwiGLU MLP on Blackwell | `silu_chunk_mul_quantize_1x32_fp8` between the two GEMMs (no BF16 intermediate) |
| one architecture's MoE chain directly (benches, experiments) | the `fso.compat` MoE pieces, whose arguments follow the kernels: `moe_layer_fp8_sm90` on sm_90, `moe_layer_mxfp8_sm120` on sm_120/121, or the masked-layout ops composed as `bench/gemm/python/bench_moe_qwen3_30a3.py` does on sm_100/103 (*MoE per-step pieces* above) |
| both operands change every call | `linear_bf16` |
| BF16 activation, cached FP8 weight, one op (sm_90 / sm_120) | `linear_qx` |

## Lower-level interface (torch.ops bindings)

The Python wrappers are thin shims over `torch.ops.fish_scales_ops.*`. They
add the arch checks, the arch-dependent defaults, and on sm_100/103 the
Python-side tier router; call the raw ops only when you replicate those. The
schemas below are generated from the registrations in the source and cover
every registered GEMM and MoE op, including internal ones that the Python
namespaces do not export.

<!-- BEGIN GENERATED torch.ops schemas (compat): do not edit by hand; regenerated from the m.def strings and the Python custom_op registrations -->

Dense block-FP8 (1x128 activation, 128x128 weight) and the bf16 convenience ops, registered in `csrc/gemm/bindings.cpp`:

```
linear_bf16(Tensor x, Tensor w) -> Tensor
linear_fp8(Tensor x_fp8, Tensor w_fp8, Tensor sx, Tensor sw) -> Tensor
linear_qx(Tensor x_bf16, Tensor w_fp8, Tensor sw) -> Tensor
quantize_1x128(Tensor x, bool use_ue8m0=False) -> (Tensor, Tensor)
quantize_1x128_packed(Tensor x, bool use_ue8m0=True) -> (Tensor, Tensor)
quantize_128x128(Tensor w, bool use_ue8m0=False) -> (Tensor, Tensor)
repack_fp8_act_scales(Tensor sx_f32, bool check=True) -> Tensor
repack_fp8_wgt_scales(Tensor sw_f32, bool check=True) -> Tensor
```

Dense MXFP8 (1x32), registered in `csrc/gemm/bindings.cpp`:

```
quantize_1x32(Tensor x, bool use_ue8m0=True) -> (Tensor, Tensor)
quantize_1x32_packed(Tensor x, bool use_ue8m0=True) -> (Tensor, Tensor)
silu_chunk_mul_quantize_1x32(Tensor gu, bool use_ue8m0=True) -> (Tensor, Tensor)
repack_mxfp8_scales(Tensor scales_f32) -> Tensor
linear_mxfp8_raw(Tensor x_fp8, Tensor w_fp8, Tensor sx_int32, Tensor sw_int32) -> Tensor
```

MoE: the MXFP8 grouped GEMMs of sm_100/103 and sm_120/121, their route queries, and the router, routing builders and combines of both layouts, registered in `csrc/gemm/bindings.cpp`:

```
linear_mxfp8_grouped_masked(Tensor a_fp8, Tensor w_fp8, Tensor sa_int32, Tensor sw_int32, Tensor masked_m, int expected_m, int max_active_groups=0, Tensor? slot_to_expert=None, Tensor? problem_shapes=None) -> Tensor
quantize_1x32_grouped_gather(Tensor x, Tensor slot_of_flat, int topk, int num_groups, int m_cap, bool use_ue8m0=True) -> (Tensor, Tensor)
silu_chunk_mul_quantize_1x32_grouped(Tensor gu, Tensor slot_of_flat, bool use_ue8m0=True, bool pairwise=False) -> (Tensor, Tensor)
linear_mxfp8_grouped_masked_swiglu(Tensor a_fp8, Tensor w13_fp8, Tensor sa_int32, Tensor sw13_int32, Tensor masked_m, int expected_m, int max_active_groups=0, Tensor? slot_to_expert=None, Tensor? problem_shapes=None) -> (Tensor, Tensor)
mxfp8_grouped_swiglu_fused_route(int m_cap, int n_w, int k, int num_groups, int max_active_groups) -> bool
mxfp8_grouped_swiglu_available(int n_w, int k) -> bool
mxfp8_grouped_slot_possible(int m_cap, int n_w, int k, int num_groups, int max_active_groups, bool fused_swiglu=False) -> bool
mxfp8_grouped_problem_shapes_consumed(int m_cap, int n_w, int k, int num_groups, int max_active_groups, bool fused_swiglu=False) -> bool
moe_build_routing(Tensor topk_ids, int num_groups, int m_cap, bool with_slots=False, int[] problem_shapes_nk=[], Tensor? topk_w=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
linear_mxfp8_grouped_masked_combine(Tensor a_fp8, Tensor w2_fp8, Tensor sa_int32, Tensor sw2_int32, Tensor masked_m, Tensor row_map, Tensor weight_of_slot, Tensor(a!) out, int expected_m, int max_active_groups=0) -> Tensor(a!)
moe_combine(Tensor dn, Tensor slot_of_flat, Tensor topk_w, Tensor? bias=None, Tensor? bias_scale=None, Tensor? out=None) -> Tensor
moe_topk_from_logits(Tensor logits, int topk, bool renormalize=True, bool with_shared_gate=False, Tensor? num_token_non_padded=None, Tensor? expert_map=None) -> (Tensor, Tensor, Tensor)
moe_build_sorted(Tensor topk_ids, int num_groups, int block_m) -> (Tensor, Tensor, Tensor)
moe_combine_sorted(Tensor dn, Tensor flat_to_sorted, Tensor topk_w) -> Tensor
```

MoE: the sm_90 block-FP8 grouped GEMMs and quantizers, and the compiler query of the sm_90 JIT, registered in `csrc/gemm/bindings.cpp`:

```
linear_fp8_grouped_masked(Tensor a_fp8, Tensor w_fp8, Tensor sa, Tensor sw, Tensor masked_m, int expected_m) -> Tensor
quantize_1x128_grouped_gather_sm90(Tensor x, Tensor slot_of_flat, int topk, int num_groups, int m_cap) -> (Tensor, Tensor)
silu_chunk_mul_quantize_1x128_grouped_sm90(Tensor gu, Tensor slot_of_flat) -> (Tensor, Tensor)
linear_fp8_grouped_contiguous(Tensor a_fp8, Tensor w_fp8, Tensor sa, Tensor sw, Tensor sorted_expert_ids, int block_m, int expected_m) -> Tensor
linear_fp8_grouped_contiguous_swapab(Tensor a_fp8, Tensor w_fp8, Tensor sa, Tensor sw, Tensor sorted_expert_ids, int block_n, int expected_m) -> Tensor
linear_fp8_grouped_contiguous_swapab_pair(Tensor a_fp8, Tensor w_fp8, Tensor sa, Tensor sw, Tensor sorted_expert_ids, int block_n, int expected_m) -> Tensor
linear_fp8_grouped_contiguous_swapab_swiglu(Tensor a_fp8, Tensor w_fp8, Tensor sa, Tensor sw, Tensor sorted_expert_ids, int block_n, int expected_m) -> (Tensor, Tensor)
linear_fp8_grouped_contiguous_2wg(Tensor a_fp8, Tensor w_fp8, Tensor sa, Tensor sw, Tensor sorted_expert_ids, int block_m, int expected_m) -> Tensor
linear_fp8_grouped_contiguous_swiglu(Tensor a_fp8, Tensor w_fp8, Tensor sa, Tensor sw, Tensor sorted_expert_ids, int block_m, int expected_m) -> (Tensor, Tensor)
quantize_1x128_sorted_gather_sm90(Tensor x, Tensor flat_to_sorted, int p_max, int topk) -> (Tensor, Tensor)
silu_chunk_mul_quantize_1x128_sorted_sm90(Tensor gu, Tensor flat_to_sorted) -> (Tensor, Tensor)
jit_compiler_sm90() -> str
```

Registered from Python (torch.library.custom_op), in `python/fish_scales_ops/dense/__init__.py` and `python/fish_scales_ops/moe/__init__.py`:

```
dense_linear(Tensor x, Tensor weight, Tensor scale, str kind) -> Tensor
moe_layer(Tensor hidden, Tensor w13, Tensor sw13, Tensor w2, Tensor sw2, Tensor topk_ids, Tensor topk_w, Tensor? bias, Tensor? bias_scale, str kind, bool w13_interleaved) -> Tensor
```

<!-- END GENERATED torch.ops schemas (compat) -->

Notes on the raw ops:

- `linear_mxfp8_raw` and `linear_fp8` read the packed scale buffers by raw
  pointer in the arch-native layout. On sm_100/103 the Python `linear_mxfp8`
  wrapper tries the decode row, the `scaled_mm` tier and the DSL tier first and
  falls through to the raw op, and the Python `linear_fp8` packs FP32 scales and
  calls `linear_mxfp8`, so it never reaches the raw `linear_fp8` there.
- `quantize_1x32` (FP32 scales) followed by `repack_mxfp8_scales` is the legacy
  two-step form of `quantize_1x32_packed`, the fused op `quantize_1x32_fp8`
  calls.
- `moe_build_routing` always returns six tensors: `masked_m`, `row_map`,
  `slot_of_flat`, `slot_to_expert` (empty unless `with_slots`), the per-group
  problem shapes as one `[P, G, 3]` tensor for the `P` pairs flattened into
  `problem_shapes_nk` (`[N0, K0, N1, K1, …]`, at most four pairs), and
  `weight_of_slot` (empty unless `topk_w` is given). `topk_w` and
  `problem_shapes_nk` are mutually exclusive. The Python wrapper takes
  `problem_shapes_for=[(N, K), …]`, returns only the requested outputs, and
  hands the problem shapes back as a list of `[G, 3]` views.
- `linear_mxfp8_grouped_masked_combine` accumulates into `out` and returns it;
  the schema marks `out` as mutated.
- `dense_linear` and `moe_layer` are registered from Python as
  `torch.library.custom_op`s, each with its own fake implementation, so
  `torch.compile` keeps either as one opaque node. `fso.dense.linear` is the
  caller of `dense_linear` ([`dense.md`](dense.md), *linear*) and `fso.moe.layer`
  the caller of `moe_layer` ([`moe.md`](moe.md), *the op and its dispatch*). No
  other op in the namespace has a fake implementation registered by this
  library.
- The sm_90 fused FC1s and their test vehicles. `linear_fp8_grouped_contiguous_swiglu`
  (non-swap) and `linear_fp8_grouped_contiguous_swapab_swiglu` (swap-AB, `block_n`
  16 / 32 / 64) are the FC1s `moe_layer_fp8_sm90` runs: the gate_up GEMM with the
  SwiGLU + 1x128 requantize in its epilogue, returning the `(dq, sd)` pair
  `silu_chunk_mul_quantize_1x128_sorted_sm90` makes of the unfused GEMM's output,
  bit-identical on every routed row. `linear_fp8_grouped_contiguous_2wg` and
  `linear_fp8_grouped_contiguous_swapab_pair` are test and A/B vehicles only: the
  same two mainloops (one CTA owns gate block b and up block b) with a bf16
  epilogue into the unfused `[P_max, 2I]` layout, bit-identical to
  `linear_fp8_grouped_contiguous` and `linear_fp8_grouped_contiguous_swapab` on the
  rows the GEMM computes. None of the four is exported from Python.

## CUDA-graph compatibility

Every op is capture-safe **after one eager warmup of the same call**, except
the synchronizing calls listed at the end of this section. The warmup
populates the process-static state that must not be created inside a capture:
the `cudaFuncSetAttribute` guard of each kernel instantiation, the Stream-K
side-stream pool and its scratch buffer (sm_120/121; the scratch is sized by
the first Stream-K call, see `FSO_STREAMK_POOL_MB`), the int32 scale scratch
pool of the fused runner paths, the dense split-K workspace, the grouped
GEMM's argument pool, its static-array arena and the slot-bound route's
slot-list pool (sm_100/103), the multi-CTA routing builder's scratch pool
(sm_100/103 and sm_120/121) — these five are per host thread, so each thread
that captures needs its own eager call —, the deep_gemm NVRTC compilation of each kernel configuration (sm_90;
kept in memory, and written to the disk cache only under `FSO_JIT_DUMP_CUBIN=1`
or `FSO_JIT_USE_NVCC=1`), and the DSL JIT per configuration (sm_100/103), for
the mid-band tier and for the M ≤ 64 decode row alike. `tests/gemm/unit/test_cuda_graph.py` is the
reference discipline: eager reference → three warm calls on the capture
stream → capture → replay, compared bit-exactly.

On sm_120/121 the Stream-K scratch is the one buffer whose size the warmup
fixes for good: it is allocated once, by the first Stream-K call, at the larger
of `FSO_STREAMK_POOL_MB` and that call's need, and any later call, eager or
captured, that needs more aborts the process, so there the warmup has to cover
the largest Stream-K shape the process will run. On sm_100/103 the dense C++
cascade keeps the partial sums of its two-kernel split-K in a per-thread
workspace that grows on eager calls and never frees a buffer: when a call needs
more, the current buffer stays allocated for the life of the process, because a
graph captured earlier still reads and writes it, and a buffer at least twice
as large takes its place. A graph captured before a later eager call grew the
workspace therefore keeps replaying correctly. A capture that would need more
than the capturing thread's eager calls allocated raises `RuntimeError` naming
the remedy (one eager call of that shape on that thread before the capture) and
leaves the process usable. The int32 scale scratch that `linear_bf16` and
`linear_qx` pack into on sm_120/121 follows the same rule: it grows on eager
calls, keeps every buffer it has handed out, and a capture that would need it
to grow raises `RuntimeError` with the same remedy.

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
arena (default 128 MB); a capture that finds the arena full aborts the
process with a message rather than putting a launch back into the graph, and
so does a thread's first such GEMM when it is made inside a capture. Rewriting the
routing buffers in place and replaying is correct; handing a graph's GEMM a
different tensor set means a different graph.

**Calls that synchronize the device, and so cannot be captured.**
`repack_fp8_act_scales` and `repack_fp8_wgt_scales` check that every scale is a
power of two, which copies a flag to the host: one device synchronization per
call. They belong at weight-load time, outside any capture. The per-call
packing paths do not run that check: `linear_fp8` given FP32 scales (inside the
op on sm_120/121, in the Python wrapper on sm_100/103 through the torch ops'
`check=False` form) and `linear_bf16` on sm_100/103 pack with the same kernels
without checking, so they neither synchronize nor refuse a capture, on every
Blackwell architecture. `quantize_1x128_fp8_packed` does not synchronize on any
architecture.

## Environment variables

Every variable below is read by the library itself, from `python/` or from the
extension, except `FSO_BENCH_WARM_MS`, which only the benches read. The **read**
column says when a value takes effect. *Once* means the value is read on first
use and cached for the life of the process, so changing the environment
afterwards has no effect; *every call* means it is read again on each call. A
captured CUDA graph replays whatever its captured calls decided in either case,
so set every variable before the first call of the process and keep it
unchanged between a capture and its replays.

**Boolean switches.** Every boolean switch the Python package reads
(`FSO_DISABLE_SMM`, `FSO_DISABLE_DSL`, `FSO_DISABLE_DECODE_DSL`, `FSO_LOG`,
`FSO_PRINT_TILE_INFO`, `FSO_MOE_FUSED_COMBINE`, `FSO_MOE_BLOCK_OVERLAP`) follows
the rule the extension applies to `FSO_PRINT_TILE_INFO`: unset or empty means
the switch's default, a value whose first character is `0` means off, and any
other value means on. So `FSO_DISABLE_DSL=0` leaves the tier on. Switches that
only the extension reads keep the forms their rows give (`FSO_FC1_FUSED`,
`FSO_DISABLE_PDL`, `FSO_DISABLE_STREAMK`, `FSO_DISABLE_OVERRIDES`,
`FSO_CHECK_PROBLEM_SHAPES`). Valued variables (sizes, paths, tile triples) are
parsed as their rows say.

### Production switches and capacities

These change what a serving process computes, how much memory it holds, or
whether a long-running process survives a larger shape, and a deployment may
set them on purpose.

| variable | scope | default | read | effect |
|---|---|---|---|---|
| `FSO_MOE_FUSED_COMBINE=1` | `fso.moe.layer` and `fso.moe.transient_bytes` on sm_120/121 | off | once (Python, at the first call of either) | on (the boolean rule above, e.g. `1`) allows the fused-combine FC2 for the buckets where `moe_layer_fused_combine_engages_sm120` takes it: those buckets neither allocate the bf16 `[G, m_cap, HIDDEN]` down-projection slab nor launch the combine kernel, and `transient_bytes` reports the smaller reservation. The FC2's atomic adds make those buckets not bit-reproducible run to run. Off (unset, empty or `0`) keeps the slab and the combine kernel at every token count, so every bucket is reproducible. This is the only way to enable the fused combine through `fso.moe.layer`; the `fso.compat` entries take `fused_combine=` per call instead. Other architectures ignore it ([`moe.md`](moe.md), *the fused combine*) |
| `FSO_FC1_FUSED={0,1}` | sm_100/103 and sm_120/121 grouped MoE FC1, and the FC1 row order `fso.moe.prepare_experts` chooses; on sm_90 the FC1 of `moe_layer_fp8_sm90` on every route | unset: the rule | once (inside the extension for sm_100/103 and sm_120/121, in Python for sm_90) | `0` never uses the fused FC1 (and `mxfp8_grouped_swiglu_available` then answers false, so a caller keeps the `[gate; up]` weight layout too), unset applies the router, `1` uses it wherever it is legal. On sm_90 the rule is the legality: every route, the swap-AB GEMMs at each activation tile and the non-swap path alike, runs the gate_up GEMM with the SwiGLU + 1x128 requantize in its epilogue whenever `2 * INTER % 256 == 0` and `HIDDEN % 128 == 0` (the weights stay `[gate; up]`), and `0` restores the GEMM + SwiGLU kernel pair on every route. A value whose first character is `0` means `0`; any other value means `1`. Set it before `fso.moe.prepare_experts` runs, because it decides the weight layout the handle holds; the handle records that layout in `w13_interleaved` and `fso.moe.layer` follows the record, so a handle is always served in the layout it was prepared with |
| `FSO_MOE_BLOCK_OVERLAP=0` | `moe_block_mxfp8_sm120` (sm_120/121) | on (also when empty) | once (Python, at the first block call) | `0` (any value starting with `0`) runs the block's shared expert on the current stream, sequentially with the routed path, instead of on a side stream; the per-call `overlap_shared=` argument overrides it either way. The output is bit-identical in both modes. With the overlap on, the first block call of the process creates the side stream and has to be eager: a first call inside a capture raises `RuntimeError` naming this variable |
| `FSO_DISABLE_DECODE_DSL=1` | sm_100/103 `linear_mxfp8`, and `linear_fp8` and `fso.dense.linear`, which run on it there | the decode row serves M ≤ 64 wherever the vendored kernels import, `nvidia-cutlass-dsl` ≥ 4.5.0 is installed and `FSO_DISABLE_DSL` is unset | once (at the first call that reaches the row) | skip the M ≤ 64 decode row and leave those cells to the tiers behind it. The row's eager launch path rebuilds descriptors and marshals its arguments in Python on every call (*Constraints*), which a captured caller does once, at capture; a caller that runs the op eagerly one token at a time can set this to keep the previous tiers. Boolean rule above: `1` disables the row, `0` leaves it on |
| `FSO_STREAMK_POOL_MB=<n>` | sm_120/121 dense `linear_fp8`, `linear_mxfp8` and `fso.dense.linear` wherever the cascade splits K (Stream-K) | 64 MB, or the first Stream-K call's need when that is larger | once, at the first Stream-K launch of the process | size of the process-wide BF16 partial-sum scratch, allocated once and never grown, because a re-allocation would invalidate captured graphs. A later Stream-K call that needs more prints `StreamKPool capacity exhausted … Bump FSO_STREAMK_POOL_MB` and **aborts the process** (`std::abort`). Size it for the largest Stream-K shape the process will run, or make that shape's call the first one. A value that does not parse to a positive integer keeps the default |
| `FSO_GROUPED_ARG_POOL_MB=<n>` | sm_100/103 grouped GEMMs called with `problem_shapes` (`fso.moe.layer` passes them) | 128 MB per host thread (at least 1) | once per process; each host thread allocates its own arena on its first eager grouped call | capacity of the per-thread static-array arena. Every distinct (capture, GEMM) pair pins one block of it for the life of the process (*routing-supplied problem shapes* under *MoE per-step pieces* above). A capture that finds the arena full **aborts the process** with a message naming the variable, and so does a thread's first such call when it is made inside a capture; an eager call that finds the arena full falls back to the per-launch preparation kernel and warns once |
| `FSO_JIT_INCLUDE_DIRS=a:b:c` | sm_90 (the deep_gemm NVRTC JIT) | a wheel: `_jit_include/` next to the extension, one include root with the deep_gemm, CUTLASS and CUDA 13.2.1 headers. An in-place build: the directories baked in at build time, which are the CUTLASS headers, the repository's deep_gemm headers and the build host's CUDA headers | once, at the first JIT compile | colon-separated include directories for the JIT, used in place of the default. Needed when an in-place build's source tree has moved. With `FSO_JIT_DEBUG=1` the choice is logged as `sm_90 JIT include directories (<origin>): <list>` |
| `FSO_JIT_NVRTC_LIB=<path>` | sm_90 (the deep_gemm NVRTC JIT) | the NVRTC 13.2.78 bundled with the package: `_nvrtc/libnvrtc.so.13` next to the extension, which `scripts/build.sh` unpacks for an sm_90 build (`scripts/vendor_nvrtc.py`) | once, at the first JIT compile or the first `jit_compiler_sm90` call | the `libnvrtc.so` file the JIT compiles with, in place of the bundled one; a relative path is taken from the working directory. The extension does not link NVRTC: it loads this one library privately (`RTLD_LOCAL`), so the `libnvrtc.so.13` that torch loads, whose version depends on the torch build, compiles no sm_90 kernel unless this variable names that very file. Nothing else is tried either: when the library cannot be loaded, the first sm_90 GEMM raises `RuntimeError` naming the path, the loader's message and the remedy (run `python scripts/vendor_nvrtc.py`, or set this variable to a CUDA 13.2 `libnvrtc.so.13`). A build made with `FSO_SKIP_NVRTC_VENDOR=1` has no bundled copy and needs this variable. A library whose version is not 13.2 is used as given and prints one line on stderr, because the sm_90 kernels are validated and measured with NVRTC 13.2 and other NVRTC versions emit different code. `torch.ops.fish_scales_ops.jit_compiler_sm90()` returns the compiler in use (`NVRTC 13.2 (<path>)`, or `nvcc <path>` under `FSO_JIT_USE_NVCC`), and on sm_90 the last line of `fso.dense.describe()` and `fso.moe.describe()` repeats it |

### Debugging and A/B knobs

These exist to measure, sweep or diagnose. None of them is needed for a correct
or complete deployment, and the cascades already encode the measured picks.

| variable | scope | default | read | effect |
|---|---|---|---|---|
| `FSO_FORCE_TILE="TM,TN,ST"` | sm_120/121 block-FP8 and MXFP8 dense cascades and the grouped MXFP8 cascade; sm_100/103 dense and grouped cascades | unset: the cascade picks | once | force one tile instantiation instead of the cascade pick. The wire format is shared, but `ST` means different things per arch: on sm_120 it is the stage count, on sm_100/103 it names the (SM count, TileK, cluster, epilogue) variant — see the two tables below. An `(TM, TN, ST)` triple that is not instantiated falls through to the cascade: the grouped dispatchers print a warning, the dense ones do not |
| `FSO_FORCE_TILE_K=<K>` | wherever `FSO_FORCE_TILE` applies | unset: every `K` | once | apply `FSO_FORCE_TILE` only to GEMMs whose `K` equals the value — lets a layer-level sweep force one projection while the others keep their picks |
| `FSO_FORCE_KSPLIT=<n>` | with `FSO_FORCE_TILE`: the sm_120/121 dense Stream-K split, and the sm_100/103 dense Stream-K and parallel split-K variants (`ST` 7, 8 and 9) | unset: no forced split | once | force the Stream-K / split-K factor; values ≤ 1 force nothing |
| `FSO_FORCE_MIN_BLOCKS=2` | sm_120/121 MXFP8 dense and grouped, with `FSO_FORCE_TILE` | unset | once | run the 2-CTA/SM instantiation of the forced tile where one exists |
| `FSO_FORCE_SMALLM=1` | sm_120/121 MXFP8 dense, with `FSO_FORCE_TILE=16,…` | off | once | experimental small-M kernel variant |
| `FSO_FORCE_SCHED_GROUP=<n>` | sm_120/121 MXFP8 dense, with `FSO_FORCE_TILE=16,…` | unset | once | persistent-scheduler swizzle group size |
| `FSO_DISABLE_OVERRIDES=1` | sm_120/121 block-FP8 and MXFP8 dense cascades | off | once | skip the shape-specific single-launch overrides, cascade table only |
| `FSO_DISABLE_STREAMK=1` | sm_120/121 dense, both formats | off | once | single launch everywhere; the Stream-K pool is then never allocated |
| `FSO_DISABLE_PDL=1` | the MoE glue on every architecture (`moe_build_routing` in both builders, `moe_build_sorted`, `moe_combine`, `moe_combine_sorted`) and the router `moe_topk_from_logits`; the MXFP8 grouped gather-quantize and SwiGLU requantize (sm_100/103, sm_120/121); every sm_120/121 MXFP8 GEMM launch, dense and grouped; on sm_100/103 the dense parallel split-K reduce, the grouped argument-preparation kernel, the pointer-array grouped GEMM and the slot-route kernels | off: PDL on | once (each translation unit caches its own copy of the same variable) | drop the programmatic-dependent-launch attribute from those launches; the kernel-side waits become no-ops and the result is unchanged. Because the glue kernels are shared, an sm_90 A/B run that sets it changes the sm_90 layer's routing builder and combine too. The sm_120/121 block-FP8 GEMM always launches with PDL, and the sm_90 deep_gemm kernels never do; neither reads the variable. The glue and quantize kernels whose grid exceeds 4096 CTAs launch without PDL whatever the setting. A value whose first character is `1` disables |
| `FSO_MOE_ROUTING_MULTI={0,1}` | `moe_build_routing` on every architecture | the multi-CTA builder on sm_100/103 and sm_120/121 once `M · topk ≥ 4096`, the single-CTA builder otherwise | once | `0`, or any value that does not parse to a nonzero integer, restores the single-CTA builder on every architecture; a nonzero integer selects the multi-CTA builder on every architecture, sm_90 included, from the same 4096-pair threshold. The two builders order the slots inside a group differently and give the same layer output (*the multi-CTA builder* under *MoE per-step pieces* above); the multi-CTA builder needs one eager call per host thread before a capture |
| `FSO_MOE_SCATTER_WARP=0` | sm_120/121 fused-combine FC2 | the store warps | every call | `0` scatters the weighted rows from the math warps instead of from the store warp plus the spare fourth TMA warp. The default overlaps the atomic adds with the next tile's mainloop the way the TMA store it replaces does; the result is unchanged either way, and the measurements are in [`../perf/layer/sm120.md`](../perf/layer/sm120.md) |
| `-DFSO_PROLOGUE_TRACE` (build-time; runtime `FSO_PROLOGUE_TRACE_PTR`) | sm_120/121 grouped MXFP8 GEMM | not compiled | once (only in a build with the macro) | compile-time-optional prologue trace, off by default and zero code when off (the production kernels are SASS-identical): with the macro defined, the kernel writes `%globaltimer` / `%clock64` stamps at every prologue boundary (descriptor prefetch, barrier init, staging, prefix scan, PDL wait, first tile, per-warp exit) for CTA 0 and the last CTA into a device buffer whose address the dispatcher reads from `FSO_PROLOGUE_TRACE_PTR`. Built into a separate tree copy for attribution runs, never into a production build; the result is unchanged either way |
| `FSO_MOE_DECODE_GATE=0` | sm_120/121 grouped MoE GEMM, decode band | the tile-count rule | once | restore the cascade's `m_cap`-only decode gate. The default rule: at row caps of 4–16 the grouped GEMM takes the solo (16,128,4) instance at one CTA per SM only when its worst-case tile count, `max_active_groups × ceil(m_cap/16) × ceil(N/128)`, lies between 0.6 × and 1 × the SM count — one full wave — and the 2-CTA/SM (16,64,4) instance otherwise; very short K (≤ 4 k-tiles) and a call without the `max_active_groups` hint keep the `m_cap` gate. The rule is decided on the routed layer rather than on the isolated kernel, because the layer's FC2 is launched under PDL while the FC1 still runs and packs two CTAs per SM behind the smaller 2-CTA FC1 (`../perf/README.md` §8). The result is unchanged either way |
| `FSO_MOE_GROUP_ORDER=auto\|<n>`, `FSO_MOE_GROUP_BLOCK=<w>` | sm_120/121 grouped MoE GEMM | expert-id order, blocks of 1 | once | the order the persistent scheduler visits the experts in. Unset, `0` or `1` is expert-id order. `auto` walks them with a stride of about a quarter of the expert count, so consecutive tiles come from experts spread across the set instead of from one contiguous run; `<n>` sets the stride explicitly and is rejected (with a warning, falling back to expert-id order) unless it is coprime with the number of steps. `FSO_MOE_GROUP_BLOCK=<w>` (must divide the expert count) makes each step carry `w` consecutive experts, which keeps runs of that many adjacent in memory. Fewer than eight experts or eight blocks keep expert-id order. The result is unchanged either way. It can help only when a deployment's rows-per-expert are correlated with the expert id, which is when a block of lightly loaded experts would otherwise become a stretch of the launch with too little arithmetic to cover the weight tiles it must load; without such a correlation it can cost time |
| `FSO_SWAP_BN=16\|32\|64\|0` | sm_90 composed MoE layer | the rows-per-expert cascade | every call (Python, in `moe_swap_ab_block_n`, which every `moe_layer_fp8_sm90` call and every sm_90 `transient_bytes` call runs) | force the swap-AB activation tile (0 = the non-swap block_m = 64 path) instead of the rows-per-expert cascade |
| `FSO_FC2_SWAP=0\|1` | sm_90 composed MoE layer, non-swap path | unset: the rule, which runs the swap-AB GEMM with block_n 64 | every call (Python, in `_sm90_fc2_swap`, which every `moe_layer_fp8_sm90` call runs) | the down-projection (FC2) GEMM of the non-swap path: `1` the swap-AB GEMM with block_n 64, `0` the non-swap block_m = 64 GEMM. Both read the same 64-row padded layout and give bit-identical results. A value whose first character is `0` means `0`; any other non-empty value means `1`. The swap-AB routes always run the swap-AB FC2 of their own tile. Set it before the first call and keep it between a capture and its replays |
| `FSO_SWAP_STAGES=<n>` | sm_90 swap-AB grouped GEMM | at least 6 stages (the picker's count when it is deeper), lowered until the shared memory fits | every call | pipeline depth, honoured for 1 to 12 and ignored otherwise. The value is applied before the CTA count is chosen. In the two-CTA build it is a cap: the dispatcher lowers it until two CTAs fit and the count divides `K / 128`, and falls back to one CTA when no count of 2 or more does. In the single-CTA build it is applied again and clamped to the shared-memory budget |
| `FSO_SWAPAB_CTAS_PER_SM=1\|2` | sm_90 swap-AB grouped GEMM | 2 when the sorted layout can feed two CTAs per SM (`(p_max / block_n) × (N / block_m) > 2 × SMs`), 1 otherwise | every call | resident CTAs per SM, honoured for `1` and `2` and ignored otherwise. Two CTAs run the math warp-groups on 96 registers and double the grid; one is the single persistent CTA with 232 registers. A default or requested 2 still falls back to 1 when no stage count of 2 or more that divides `K / 128` lets two CTAs fit, and when the compiled two-CTA kernel does not use exactly 80 registers |
| `FSO_JIT_EXTRA_FLAGS="-DFOO=1 ..."` | sm_90 | none | at each JIT compile of a kernel not yet built in the process | extra NVRTC flags appended to every deep_gemm JIT compile (developer A/B of kernel-side `#if` switches). They are part of the disk cache's content key (see `FSO_JIT_CACHE_DIR`), so a cubin built under other flags is never loaded in their place |
| `FSO_JIT_DEBUG=1`, `FSO_JIT_DUMP_CUBIN=1`, `FSO_JIT_USE_NVCC=1`, `FSO_JIT_NVCC_COMPILER=<path>` | sm_90 | off; nvcc from `$CUDA_HOME/bin`, else from `PATH` | once (the three switches when the library loads, the compiler path at the first nvcc compile) | deep_gemm JIT diagnostics: verbose compile and cache log on stderr, which names the NVRTC library and its version once; also write every compiled cubin to the disk cache and load from it; compile with nvcc instead of the bundled NVRTC (no NVRTC is loaded then, and `FSO_JIT_NVRTC_LIB` is not read; its cubins always go through the disk cache); the nvcc to run (default `$CUDA_HOME/bin/nvcc`, else `nvcc` on `PATH`). The TensorRT-LLM names `TRTLLM_DG_JIT_DEBUG`, `TRTLLM_DG_JIT_DUMP_CUBIN`, `TRTLLM_DG_JIT_USE_NVCC` and `TRTLLM_DG_NVCC_COMPILER` are deprecated aliases, honoured until fish-scales-ops 0.3.0 removes them: each is read only when its `FSO_JIT_*` name is unset, and prints a one-time notice on stderr when it supplies the value. The three switches accept `1` or `true` |
| `FSO_JIT_CACHE_DIR=<dir>` | sm_90 | `${XDG_CACHE_HOME:-$HOME/.cache}/fish_scales_ops/jit` | once, at the first disk-cache access | root of the deep_gemm JIT disk cache. By default every kernel is compiled once per process with NVRTC and kept in memory, and the JIT neither reads nor writes the disk; the disk cache is used only with `FSO_JIT_DUMP_CUBIN=1` or `FSO_JIT_USE_NVCC=1`. Its root defaults to `${XDG_CACHE_HOME:-$HOME/.cache}/fish_scales_ops/jit` (`%LOCALAPPDATA%\fish_scales_ops\jit` on Windows, `<temp dir>/fish_scales_ops/jit` without a home directory). The cache is keyed by content: a cubin lives in `<root>/cache/<key>_<kernel name>/`, where `<key>` is 16 hex digits of a hash over the generated kernel source, the complete flag list (`FSO_JIT_EXTRA_FLAGS` and the include directories among them), the compiler (the NVRTC version, or the nvcc path) and every file in the `deep_gemm/` header directory the build compiles against. A header edit, a flag change or a compiler change therefore builds a new cubin instead of loading an old one, and cubins without a key (TensorRT-LLM's DeepGEMM cache, dumps of older fish_scales_ops builds) are never read. `TRTLLM_DG_CACHE_DIR` is the deprecated alias, same rule as above |
| `CUDA_HOME` | sm_90 JIT under `FSO_JIT_USE_NVCC=1` | unset | once, at the first nvcc compile | locates `nvcc` when `FSO_JIT_NVCC_COMPILER` is unset; it is also a build variable (README, *Install*) |
| `FSO_DISABLE_SMM=1`, `FSO_DISABLE_DSL=1` | sm_100/103 MXFP8 router | off | once (at the first call that reaches the tier) | skip the cuBLAS `scaled_mm` tier / both CuTe-DSL rows, the M ≤ 64 decode row and the mid-band tier (both set = pure C++ cascade). Boolean rule above: `1` disables, `0` leaves the tier on. A tier switched off this way prints nothing |
| `FSO_LOG=1` | sm_100/103 MXFP8 decode row | off | at each message; each message prints at most once per process | print, once per process on stderr, that the decode row is inactive because `nvidia-cutlass-dsl` is not installed, the one reason that is otherwise silent. A DSL below the 4.5.0 floor or vendored kernels that fail to import print their `fso:` line without this switch. Silent when the row is working. Boolean rule above |
| `FSO_DSL_KERNEL_PATH=<file>` | sm_100/103 | a wheel: its copy of the CUTLASS CuTe-DSL example in `fish_scales_ops/_dsl/`. An in-place build: the example under `3rdparty/cutlass` in the source tree | once (at the DSL tier's first use) | alternative DSL kernel source for the mid-band tier (the decode row's kernels are vendored in-tree and not overridable). When the file (given here or at the default place) does not exist, or loading it raises, the tier is skipped and the first call that reaches it prints one `fso:` line on stderr naming the tier, the reason and this variable; `FSO_DISABLE_DSL=1` skips the tier without the line |
| `FSO_FORCE_SWIZZLE=<n>`, `FSO_FORCE_RASTER={1,2}`, `FSO_SK_NDET=1`, `FSO_SK_DECOMP={1,2,3}` | sm_100/103 C++ dense cascade | unset | once | scheduler raster swizzle size, raster direction (along M / along N), nondeterministic Stream-K reduction, decomposition mode — probe knobs, never wired into the cascade |
| `FSO_PRINT_TILE_INFO=1` | sm_100/103 dense and grouped cascades, the slot route and the decode row | off | once per dispatcher; the decode row prints once per distinct `(M, N, K)` | make every kernel instantiation print, once, the mainloop stage count `StageCountAutoCarveout` derived for it and its shared-memory footprint. That is the quantity that says whether a narrower `TileN` bought pipeline depth or only extra CTAs, and it is the only way to see a stage collapse (a tile whose epilogue eats the carve-out and leaves one mainloop stage) without guessing. The decode row prints the tactic it picked instead, and the sm_100/103 dense cascade prints a line each time its per-thread split-K workspace grows. Boolean rule above, in the C++ dispatchers and the decode row alike |
| `FSO_GROUPED_SLOT={0,1,force,force@<N>}` | sm_100/103 grouped MoE decode route | the dispatcher's rule | once | `0` never takes the slot-bound swap-orientation route, unset or `1` applies the dispatcher's rule, `force` takes it wherever it is legal and raises where it is not, `force@<N>` forces it for the GEMM whose `N` it names only (see *the slot-bound decode route* under *MoE per-step pieces* above; the `force` forms are A/B knobs) |
| `FSO_GATHER_QUANT_ONCE={0,1}` | sm_100/103 grouped MoE gather-quantize | the launcher's rule | once | `0` always quantizes per routed (token, expert) pair, unset applies the launcher's rule (the token-space form once its grid covers one full wave of SMs), `1` always quantizes each token once and scatters the bytes to its top-k destinations |
| `FSO_GATHER_QUANT_TOPK_STATIC=0` | sm_100/103 grouped gather-quantize, token-scatter form | on | once | `0` forbids the compile-time top-k = 8 instantiation of the token-scatter kernel, so the runtime-top-k loop runs instead. A measurement knob that only changes which instantiation runs |
| `FSO_CHECK_PROBLEM_SHAPES=1` | sm_100/103 grouped GEMMs with `problem_shapes` | off | every call | copy the caller's `[G, 3]` tensor to the host on every call and verify that its `N` and `K` are this GEMM's and that every row count lies in `[0, m_cap]`; read on every call (not cached), debugging only |
| `FSO_BENCH_WARM_MS=<ms>` | benches only | 0 | once per bench process | spin the GPU before each cell's timing (needed on unlocked devices, see `../perf/README.md` §5) |

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
cascade tiles need. Codes 20–25 exist for sweeping a *new*
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

The 64- and 192-wide widths exist only on the two variants the cascade itself
uses (`ST=1` and `ST=5`); the 2-SM and TileK 256 variants, which the cascade
does not pick, were not extended to them, to keep the binary small. The cascade
does not pick `TileN = 64` on this path either; it is kept so that a future
sweep does not have to rebuild it.

Build-time variables (`TORCH_CUDA_ARCH_LIST`, `CUDA_HOME`, `CUTLASS_DIR` /
`BSGEMM_CUTLASS_DIR`) are documented in the README's Install section.
The swap-AB tile / non-swap choice is the Python cascade `MOE_SWAP_BLOCK_N_CASCADE` on routed rows per active expert (`moe_swap_ab_block_n(M, E, topk)`); `FSO_SWAP_BN=16|32|64|0` forces one tile (0 = non-swap) for A/B runs.
