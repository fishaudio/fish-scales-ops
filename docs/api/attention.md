# `fish_scales_ops.attention` — public API

Frozen op contracts for the attention domain. **No performance numbers in this
file**; reference numbers live in [`../perf/attention/`](../perf/README.md).
Code is the source of truth: `python/fish_scales_ops/attention/flash_attn_func.py`
and `python/fish_scales_ops/attention/backends/sm120_mxfp8*.py` for the
wrappers, `csrc/attention/csrc/flash_attn_ext.cpp` for the `torch.ops`
schemas.

## Overview

Forward-only attention. fso has native kernels for one arch only:

| arch | native kernel | what `flash_attn_fwd` does |
|---|---|---|
| sm_120 / sm_121 (RTX 5090, RTX PRO 6000) | MXFP8 FlashAttention: contiguous-KV prefill, paged-KV single-token decode, paged-KV ragged prefill (extend); head_dim ∈ {32, 64, 128, 256}, native GQA | routes to torch SDPA (BF16); the MXFP8 kernels are reached through the backend ops below |
| sm_90 (H200), sm_100 / sm_103 (B200 / B300) | none | torch SDPA |

The MXFP8 kernels do **no quantization**: the caller hands over FP8 (E4M3)
Q / K / V plus UE8M0 scale tensors in the layouts below, the way a serving
engine with an FP8 KV cache already holds them. The `pre_quantize_*` /
`quantize_*` helpers in the backend modules are calibration- and test-time
tools written in plain PyTorch, not runtime ops.

## Quick start — convenience path (`flash_attn_fwd`)

```python
import torch, fish_scales_ops as fso

B, S, H_q, H_kv, D = 2, 4096, 32, 8, 128
q = torch.randn(B, S, H_q,  D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, S, H_kv, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, S, H_kv, D, dtype=torch.bfloat16, device="cuda")

o = fso.attention.flash_attn_fwd(q, k, v, causal=True)       # bf16 [B, S, H_q, D]
o, disp = fso.attention.flash_attn_fwd(q, k, v, causal=True, return_dispatch=True)
disp.kernel_id, disp.backend, disp.sm_version                # "torch_fallback_fwd", "torch_sdpa", 120
```

## `flash_attn_fwd` signature and dispatch order

```python
fso.attention.flash_attn_fwd(
    q, k, v,                      # [B, S, H, D] CUDA tensors; bf16 / fp16 / fp8
    *,
    softmax_scale=None,           # default 1 / sqrt(D)
    causal=False,
    window_left=-1,               # local-attention window; -1 disables
    window_right=-1,
    return_dispatch=False,        # also return FlashAttnDispatch(kernel_id, sm_version, backend)
    force_kernel=None,            # only "torch_fallback_fwd" is honoured
) -> torch.Tensor                 # bf16 [B, S, H_q, D]
```

Today every dtype on every arch takes `torch.nn.functional.scaled_dot_product_attention`
(FP8 inputs are dequantised to BF16 first; GQA is expanded with
`repeat_interleave`; a local window builds an explicit additive mask). The
kernel selector exists but is bypassed because `flash_attn_fwd` has no
in-graph quantizer for Q / K / V yet; a BF16 → MXFP8 in-tree path is the
planned extension. Use the backend ops for the native kernels.

## SM120 MXFP8 prefill — `mxfp8_fwd`

```python
from fish_scales_ops.attention.backends.sm120_mxfp8 import (
    mxfp8_fwd, pre_quantize_q, pre_quantize_k, pre_quantize_v)

q_fp8, q_sc = pre_quantize_q(q_bf16)          # test / calibration helpers
k_fp8, k_sc = pre_quantize_k(k_bf16)
v_fp8, v_sc = pre_quantize_v(v_bf16)          # V comes out pre-transposed [B, D, H_kv, S_k]
o = mxfp8_fwd(q_fp8, q_sc, k_fp8, k_sc, v_fp8, v_sc, softmax_scale=1 / D**0.5, causal=True)
```

Layouts (`Bc` is the K/V tile the kernel picks per head_dim: D=32 → 128,
D=64 → 64, D=128 → 64, D=256 → 32; all scale blocks are 32 elements along
the reduction axis):

| tensor | shape | dtype | notes |
|---|---|---|---|
| `q_fp8` | `[B, S_q, H_q, D]` | float8_e4m3fn | BSHD-contiguous |
| `q_scales` | `[B, S_q/16, H_q, D/32]` | uint8 (UE8M0) | one scale per 16-row Q tile per 32-wide D block |
| `k_fp8` | `[B, S_k, H_kv, D]` | float8_e4m3fn | BSHD-contiguous |
| `k_scales` | `[B, S_k/Bc, H_kv, D/32]` | uint8 | one per Bc-row K tile per D block |
| `v_fp8` | `[B, D, H_kv, S_k]` | float8_e4m3fn | **pre-transposed**, S innermost (the PV MMA reads 16-byte K-contiguous fragments); transpose once when the cache is quantized |
| `v_scales` | `[B, S_k/Bc, H_kv, Bc/32]` | uint8 | one per Bc-row V tile per 32-token block |
| output | `[B, S_q, H_q, D]` | bfloat16 | |

Constraints: `H_q % H_kv == 0` (GQA by stride-0 broadcast, no expansion);
`D ∈ {32, 64, 128, 256}`; `S_q % 64 == 0`; `S_k % Bc == 0`. The `out=`
argument is accepted for symmetry; the op allocates its own output and the
wrapper copies into `out`. On any other arch the host dispatcher returns
"operation not supported" and the binding raises `RuntimeError`.

## SM120 MXFP8 paged decode — `mxfp8_decode_paged_fwd` (single-call and plan/run)

Single-token (`S_q = 1`) decode against a paged FP8 KV cache. Q and K are
scaled by the hardware block-scale operand of the QK MMA; V uses a
**per-channel FP32 scale applied in the epilogue**, so the V scale does not
depend on the page size.

```python
from fish_scales_ops.attention.backends.sm120_mxfp8_decode import (
    mxfp8_decode_paged_fwd, plan_decode_paged,
    quantize_q_grouped, quantize_kv_to_paged, compute_k_chan_scale, compute_v_chan_scale)

k_sc = compute_k_chan_scale(k_calib)          # once per model: [H_kv, D/32] uint8
v_sc = compute_v_chan_scale(v_calib)          # once per model: [H_kv, D]    fp32
K_cache, _, V_cache, _ = quantize_kv_to_paged(k_full, v_full, page_size=32,
                                              k_chan_scale=k_sc, v_chan_scale=v_sc)
q_fp8, q_sc = quantize_q_grouped(q_bf16, H_kv)

# one-shot: allocates the split-K scratch and zeroes the sync counter every call
o = mxfp8_decode_paged_fwd(q_fp8, q_sc, K_cache, k_sc, V_cache, v_sc,
                           block_table, seq_lens, softmax_scale=1 / D**0.5)   # bf16 [B, H_q, D]

# plan/run: scratch allocated once, counter tracked across calls (decode loops, CUDA graphs)
plan = plan_decode_paged(B=B, H_q=H_q, H_kv=H_kv, D=D, max_blocks=max_blocks)   # kv_split_k=None → auto
for step in range(steps):
    o = plan.run(q_fp8, q_sc, K_cache, k_sc, V_cache, v_sc, block_table, seq_lens, out=o)
```

| tensor | shape | dtype | notes |
|---|---|---|---|
| `q_fp8` | `[B, H_q, D]` | float8_e4m3fn | |
| `q_scales` | `[B, H_kv, D/32]` | uint8 (UE8M0) | one scale shared by the Q heads of a KV-head group |
| `k_cache_fp8` | `[num_pages, page_size, H_kv, D]` | float8_e4m3fn | |
| `k_chan_scale` | `[H_kv, D/32]` | uint8 (UE8M0) | one per KV head per D block, **global across pages and tokens**; frozen at model load |
| `v_cache_fp8` | `[num_pages, D, H_kv, page_size]` | float8_e4m3fn | pre-transposed within a page |
| `v_chan_scale` | `[H_kv, D]` | float32 | per-channel, applied after the MMA; frozen at model load |
| `block_table` | `[B, max_blocks]` | int32 | |
| `seq_lens` | `[B]` | int32 | |
| output | `[B, H_q, D]` | bfloat16 | |

Constraints: `D ∈ {32, 64, 128, 256}`; `page_size` a positive multiple of 32
(the MXFP8 scale vector along D; sglang `--page-size 32` is the natural
minimum); `H_q % H_kv == 0` with `H_q / H_kv <= 64`. `kv_split_k` (FlashDecoding
split of one sequence over several CTAs) defaults to a power of two aiming at
about two waves over 170 SMs; the plan owns the `m/l/o_partial` scratch and
the `sync_counter` it needs. `DecodePagedPlan.run` refuses non-contiguous
inputs instead of copying them (a copy inside a capture would bake a stale
pointer into the graph) and, when called during stream capture, re-zeroes the
counter inside the graph so replays are self-contained.

## SM120 MXFP8 paged prefill / extend — `mxfp8_paged_prefill_fwd` (single-call and plan/run)

Ragged Q against a paged KV cache, the FlashInfer `BatchPrefillWithPagedKVCache`
calling convention (so an sglang FlashInfer backend can be rewired to it).

```python
from fish_scales_ops.attention.backends.sm120_mxfp8_paged_prefill import (
    mxfp8_paged_prefill_fwd, plan_paged_prefill, quantize_q_ragged)

q_fp8, q_sc = quantize_q_ragged(q_bf16, H_q)                 # [total_q, H_q, D], [ceil(total_q/16), H_q, D/32]
o = mxfp8_paged_prefill_fwd(q_fp8, q_sc, K_pool, k_sc, V_pool, v_sc,
                            qo_indptr, paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len,
                            causal=True)                       # bf16 [total_q, H_q, D]

plan = plan_paged_prefill(qo_indptr_cpu=qo_indptr.cpu(), num_q_heads=H_q)   # packs the (b, h, q_tile) work list once
o = plan.run(q_fp8, q_sc, K_pool, k_sc, V_pool, v_sc,
             qo_indptr, paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len, causal=True)
```

| tensor | shape | dtype | notes |
|---|---|---|---|
| `q_fp8` | `[total_q, H_q, D]` | float8_e4m3fn | ragged over the batch |
| `q_scales` | `[ceil(total_q/16), H_q, D/32]` | uint8 | the helper zero-pads the tail rows to the 16-row tile |
| `k_pool_fp8` | `[num_pages, page_size, H_kv, D]` | float8_e4m3fn | |
| `k_chan_scale` | `[H_kv, D/32]` | uint8 | global, as in decode |
| `v_pool_fp8` | `[num_pages, D, H_kv, page_size]` | float8_e4m3fn | pre-transposed within a page |
| `v_chan_scale` | `[H_kv, D]` | float32 | |
| `qo_indptr` | `[B+1]` | int32 | cumulative Q tokens |
| `paged_kv_indices` | `[P_used]` | int32 | flat page ids |
| `paged_kv_indptr` | `[B+1]` | int32 | per-request slice into `paged_kv_indices` |
| `paged_kv_last_page_len` | `[B]` | int32 | valid tokens in each request's last page |
| output | `[total_q, H_q, D]` | bfloat16 | |

Constraints: `D ∈ {32, 64, 128, 256}`; `page_size` a positive multiple of 32;
`H_q % H_kv == 0`; per-request Q lengths need not be multiples of 16 (the
kernel masks the ragged tail). The single-call form walks `qo_indptr` on the
host to build the work list each call; the plan does that once for a fixed
batch shape and is the CUDA-graph form (`run` also refuses non-contiguous
inputs).

## torch.ops bindings (lower-level)

```
mxfp8_attn_fwd(Tensor q_fp8, Tensor q_scales, Tensor k_fp8, Tensor k_scales, Tensor v_fp8, Tensor v_scales,
               float softmax_scale, bool causal) -> Tensor                                  # bf16 [B, S_q, H_q, D]

mxfp8_decode_paged(Tensor q_fp8, Tensor q_scales, Tensor k_pool, Tensor k_chan_scale, Tensor v_pool, Tensor v_chan_scale,
                   Tensor block_table, Tensor seq_lens,
                   Tensor m_partial, Tensor l_partial, Tensor o_partial, Tensor sync_counter,
                   int num_splits, int target_counter, float softmax_scale) -> Tensor      # bf16 [B, H_q, D]

mxfp8_attn_fwd_paged(Tensor q_fp8, Tensor q_scales, Tensor k_pool, Tensor k_chan_scale, Tensor v_pool, Tensor v_chan_scale,
                     Tensor qo_indptr, Tensor paged_kv_indices, Tensor paged_kv_indptr, Tensor paged_kv_last_page_len,
                     Tensor work_units, int total_work, bool causal, float softmax_scale) -> Tensor   # bf16 [total_q, H_q, D]
```

The wrappers own the scratch buffers, the split-K sync-counter bookkeeping,
the work-unit packing, contiguity checks and the arch check; call the raw ops
only when integrating into a graph compiler that reproduces those. The
combined extension also exposes `fish_scales_ops._C.probe_device_caps()` for
the selector's device capability probe.

## Reference and test helpers

`backends.mxfp8_ref` (`quantize_mxfp8`, `dequantize_mxfp8`, `qk_mxfp8`) and
`backends.mxfp8_attn_ref` (`mxfp8_qk_mixed_pv_fwd`) are plain-PyTorch
reference implementations of the block-scaled attention arithmetic used by the
tests to gate the kernels; they are slow by design and are not part of the
runtime API.
