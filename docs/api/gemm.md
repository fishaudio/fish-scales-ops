# `fish_scales_ops.gemm` — public API

Frozen op contracts for the GEMM domain. **No performance numbers in this
file**; reference numbers live in [`../perf/`](../perf/README.md) and the
acceptance rule in [`../perf/README.md`](../perf/README.md) §7. Code is the source of
truth for signatures: the Python wrappers are in
`python/fish_scales_ops/gemm/{fp8,mxfp8,bf16}.py`, the `torch.ops` schemas in
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
| sm_90 (H200) | deep_gemm WGMMA kernels, NVRTC-JIT-compiled in-process at first call; FP32 scales | not available (`NotImplementedError`) | block-FP8, expert-sorted contiguous layout (`moe_layer_fp8_sm90`) and a masked-layout variant |
| sm_120 / sm_121 (RTX 5090, RTX PRO 6000) | CUTLASS `Sm120BlockScaledKernel`; UE8M0 scales packed into int32 words | CUTLASS `Sm120BlockScaledKernel` | MXFP8, masked slab layout |
| sm_100 / sm_103 (B200 / B300) | since 2026-09-05: the MXFP8 tcgen05 path with each 1×128 UE8M0 scale byte replicated into its four 32-wide slots (same kernels, same bytes as MXFP8) | CUTLASS tcgen05 BlockScaled behind a three-tier router (cuBLAS `scaled_mm`, CuTe DSL persistent kernel, C++ cascade) | not implemented (milestone M3) |

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

# ---- MoE layer, sm_90 --------------------------------------------------------
# w13/w2 quantized offline per expert with quantize_moe_weights_1x128_fp8_sm90.
out = fso.gemm.moe_layer_fp8_sm90(hidden, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w)
```

The sm_120 MoE layer is composed from the masked-layout ops listed under
*Grouped MoE*; `bench/gemm/python/bench_moe_qwen3_30a3.py` (`fso_mxfp8_layer`)
is the reference composition.

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
| `linear_mxfp8(x_fp8, w_fp8, sx, sw)` | fp8 `[M, K]`, fp8 `[N, K]`, opaque int32 scales from `quantize_1x32_fp8` on the same arch | bf16 `[M, N]` | `K % 128 == 0`, `N % 128 == 0`; sm_100/103 routes through the three tiers, sm_120 through the CUTLASS cascade |

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

On sm_100/103 the block-FP8 packed layout is therefore a special case of the
MXFP8 layout, which is why the block-FP8 GEMM there is the MXFP8 GEMM.

### Grouped MoE (added 2026-09)

Two layouts, one per arch. Both keep every per-expert row count **on the
device**: the host never reads routing results, so a layer captured once into a
CUDA graph replays correctly for any routing (§ CUDA-graph compatibility).

**sm_120 — masked slab layout (MXFP8).** Every expert `g` owns a slab of
`m_cap` rows; `masked_m[g]` (device int32) says how many are valid; rows at or
past it hold undefined bytes in every tensor. The layer is six kernels:

| step | function | in → out |
|---|---|---|
| routing | `moe_build_routing(topk_ids, num_groups, m_cap)` | int32 `[M, topk]` → (`masked_m [G]`, `row_map [G*m_cap]`, `slot_of_flat [M*topk]`); `G <= 1024`, `m_cap % 4 == 0`, `m_cap >= M` |
| gather + quantize | `quantize_1x32_grouped_gather_fp8(x, slot_of_flat, topk, num_groups, m_cap)` | bf16 `[M, K]` → (fp8 `[G, m_cap, K]`, int32 `[G, K/128, m_cap]`) |
| gate_up | `linear_mxfp8_grouped_masked(a_fp8, w13_fp8, sa, sw13, masked_m, expected_m)` | → bf16 `[G, m_cap, 2*INTER]` |
| SwiGLU + quantize | `silu_chunk_mul_quantize_1x32_grouped_fp8(gu, slot_of_flat)` | → (fp8 `[G, m_cap, INTER]`, int32 `[G, INTER/128, m_cap]`) |
| down | `linear_mxfp8_grouped_masked(h_fp8, w2_fp8, sh, sw2, masked_m, expected_m)` | → bf16 `[G, m_cap, HIDDEN]` |
| combine | `moe_combine(dn, slot_of_flat, topk_w)` | → bf16 `[M, HIDDEN]`, `out[t] = Σ_j topk_w[t,j] · dn[slot_of_flat[t*topk+j]]` |
| weights (offline) | `quantize_moe_weights_1x32_fp8(w)` | bf16 `[G, N, K]` → (fp8 `[G, N, K]`, int32 `[G, K/128, N]`); `N % 128 == 0`, `K % 128 == 0` |

`expected_m` is a **host-side static hint** (`ceil(M * topk / G)`) used only for
tile selection; pass a plain `int` so the captured graph stays shape-static.
Caller contract: `masked_m[g] <= m_cap` for every group (the kernel does not
check; overflow rows are dropped at the TMA bounds). The two grouped GEMM
constraints are `K % 128 == 0`, `N % 128 == 0`.

**sm_90 — expert-sorted contiguous layout (block-FP8).** The `M * topk` routed
pairs are sorted by expert into one compact list, each expert's run padded to
the GEMM block size, so the scheduler enumerates active padded blocks rather
than all `E` experts. `P_max` (the padded row count) is rounded up and fixed per
`M`, so each `M` captures into its own graph.

| step | function | in → out |
|---|---|---|
| unified entry | `moe_layer_fp8_sm90(hidden, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w)` | bf16 `[M, HIDDEN]`, per-expert fp8 weights with fp32 `[E, N/128, K/128]` scales, int32 `[M, topk]`, fp32 `[M, topk]` → bf16 `[M, HIDDEN]`; picks the swap-AB GEMMs for `M < MOE_SWAP_M_MAX` (= 256, a Python constant) and the block_m = 64 GEMMs otherwise, a host-side branch on the static `M` |
| routing | `moe_build_sorted(topk_ids, num_groups, block_m)` | → (`sorted_expert_ids [P_max]` (−1 past the real length), `flat_to_sorted [M*topk]`, `num_padded [1]`) |
| gather + quantize | `quantize_1x128_sorted_gather_sm90(x, flat_to_sorted, p_max, topk)` | → (fp8 `[P_max, K]`, fp32 `[K/128, align(P_max,4)]` K-major) |
| GEMMs | `linear_fp8_grouped_contiguous(a, w, sa, sw, sorted_expert_ids, block_m, expected_m)` and `linear_fp8_grouped_contiguous_swapab(..., block_n, expected_m)` | → bf16 `[P_max, N]` in sorted order; `sorted_expert_ids` must have been built with the same block size |
| SwiGLU + quantize | `silu_chunk_mul_quantize_1x128_sorted_sm90(gu, flat_to_sorted)` | → (fp8 `[P_max, INTER]`, fp32 `[INTER/128, align(P_max,4)]`) |
| combine | `moe_combine_sorted(dn, flat_to_sorted, topk_w)` | → bf16 `[M, HIDDEN]` |
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
  anything but sm_120/121; the sm_90 grouped ops raise on anything but sm_90.
- The masked layout needs `m_cap % 4 == 0`, `m_cap >= M`, `G <= 1024`, and
  `masked_m[g] <= m_cap`; the sorted layout needs the same block size for
  `moe_build_sorted` and the GEMMs that consume its output.
- `linear_qx` and the `linear_bf16` runner path exist on sm_90 and sm_120;
  `linear_bf16` on sm_100/103 is composed from the public ops, `linear_qx`
  raises there.
- On sm_100/103 the cuBLAS `scaled_mm` tier needs a torch build with MXFP8
  `scaled_mm` (2.12); on older torch the router uses the DSL and C++ tiers
  only. The DSL tier JIT-compiles each (tile, cluster) configuration at first
  use (about 10 s per configuration per process).

## Picking the right op

| situation | op |
|---|---|
| Hopper (sm_90) dense inference, cached FP8 weight | `linear_fp8` with `quantize_1x128_fp8(x)` per call, FP32 scales |
| Blackwell consumer (sm_120) dense inference, block-FP8 checkpoint | `linear_fp8` with pre-packed weight scales and `quantize_1x128_fp8_packed(x)` per call |
| Blackwell consumer (sm_120), free choice of format | `linear_mxfp8` with `quantize_1x32_fp8` (tighter quantization; the perf tables show where it is also faster) |
| Blackwell datacenter (sm_100/103) | `linear_mxfp8` with `quantize_1x32_fp8`; block-FP8 checkpoints run through `linear_fp8` on the same kernels |
| SwiGLU MLP on Blackwell | `silu_chunk_mul_quantize_1x32_fp8` between the two GEMMs (no BF16 intermediate) |
| MoE layer on sm_90 | `moe_layer_fp8_sm90` |
| MoE layer on sm_120 | the six masked-layout ops above |
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

linear_mxfp8_grouped_masked(Tensor a_fp8, Tensor w_fp8, Tensor sa_int32, Tensor sw_int32, Tensor masked_m, int expected_m) -> Tensor
quantize_1x32_grouped_gather(Tensor x, Tensor slot_of_flat, int topk, int num_groups, int m_cap, bool use_ue8m0=True) -> (Tensor, Tensor)
silu_chunk_mul_quantize_1x32_grouped(Tensor gu, Tensor slot_of_flat, bool use_ue8m0=True) -> (Tensor, Tensor)
moe_build_routing(Tensor topk_ids, int num_groups, int m_cap) -> (Tensor, Tensor, Tensor)
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
scratch pool of the fused runner paths, the deep_gemm NVRTC compilation of each
kernel configuration (sm_90; in-memory only, no disk cache), and the DSL JIT
per configuration (sm_100/103). `tests/gemm/unit/test_cuda_graph.py` is the
reference discipline: eager reference → three warm calls on the capture
stream → capture → replay, compared bit-exactly.

What may change between replays: the routing. Both MoE layouts keep the
per-expert counts and index maps on the device and derive them inside the
graph (`moe_build_routing` / `moe_build_sorted` are captured), so a layer
captured for one `M` replays with any routing of that `M`. What must not
change: tensor shapes, `m_cap`, `expected_m`, and the sm_90 `M`-dispatch branch
(all host-static per `M`; capture one graph per `M`).

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
| `FSO_FORCE_TILE="TM,TN,ST"` | sm_120 dense and grouped cascades; sm_100 Stream-K (`ks` only) | force one tile instantiation instead of the cascade pick |
| `FSO_FORCE_TILE_K=<K>` | sm_120 | apply `FSO_FORCE_TILE` only to GEMMs whose `K` equals the value — lets a layer-level sweep force one projection while the others keep their picks |
| `FSO_FORCE_KSPLIT=<n>` | sm_120, sm_100 | force the Stream-K / split-K factor |
| `FSO_FORCE_MIN_BLOCKS=2` | sm_120 | run the 2-CTA/SM instantiation of the tiles that have one |
| `FSO_FORCE_SMALLM=1` | sm_120 MXFP8 | experimental small-M kernel variant |
| `FSO_FORCE_SCHED_GROUP=<n>` | sm_120 MXFP8 | persistent-scheduler swizzle group size |
| `FSO_DISABLE_OVERRIDES=1` | sm_120 | skip the shape-specific single-launch overrides, cascade table only |
| `FSO_DISABLE_STREAMK=1` | sm_120 dense | single launch everywhere |
| `FSO_STREAMK_POOL_MB=<n>` | sm_120 dense | Stream-K partial-sum scratch capacity (default 64 MB; allocated once, before capture) |
| `FSO_DISABLE_PDL=1` | sm_120 MoE chain | drop the programmatic-dependent-launch attributes (kernel-side waits become no-ops) |
| `FSO_SWAP_STAGES=<n>` | sm_90 swap-AB grouped GEMM | pipeline depth of the swap-AB kernel (clamped to the smem budget) |
| `FSO_JIT_INCLUDE_DIRS=a:b:c` | sm_90 | NVRTC include directories for the deep_gemm JIT (default baked at build time) |
| `TRTLLM_DG_JIT_DEBUG=1`, `TRTLLM_DG_JIT_DUMP_CUBIN=1`, `TRTLLM_DG_JIT_USE_NVCC=1`, `TRTLLM_DG_NVCC_COMPILER=<path>`, `TRTLLM_DG_CACHE_DIR=<dir>` | sm_90 | deep_gemm JIT diagnostics: verbose compile, dump cubins, compile with nvcc instead of NVRTC, compiler path, dump directory |
| `FSO_DISABLE_SMM=1`, `FSO_DISABLE_DSL=1` | sm_100/103 MXFP8 router | skip the cuBLAS `scaled_mm` tier / the CuTe DSL tier (both set = pure C++ cascade) |
| `FSO_DSL_KERNEL_PATH=<file>` | sm_100/103 | alternative DSL kernel source |
| `FSO_FORCE_SWIZZLE=<n>`, `FSO_FORCE_RASTER={1,2}`, `FSO_SK_NDET=1`, `FSO_SK_DECOMP={1,2,3}` | sm_100/103 C++ cascade | scheduler raster swizzle size, raster direction (along M / along N), nondeterministic Stream-K reduction, decomposition mode — probe knobs, never wired into the cascade |
| `FSO_BENCH_WARM_MS=<ms>` | benches only | spin the GPU before each cell's timing (needed on unlocked devices, see `../perf/README.md` §5) |

Build-time variables (`TORCH_CUDA_ARCH_LIST`, `CUDA_HOME`, `CUTLASS_DIR` /
`BSGEMM_CUTLASS_DIR`) are documented in the README's Install section.
`MOE_SWAP_M_MAX` is a Python constant (256), not an environment variable.
