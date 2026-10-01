# `fish_scales_ops.attention` — public API

Frozen op contracts for the attention domain. **No performance numbers in this
file**; reference numbers live in [`../perf/attention/`](../perf/README.md).
Code is the source of truth: `python/fish_scales_ops/attention/__init__.py` for
the public names, `attention/flash_attn_func.py` and
`attention/backends/sm120_mxfp8*.py` for the wrappers,
`csrc/attention/csrc/flash_attn_ext.cpp` for the `torch.ops` schemas.

## Overview

Forward-only attention. fso has native kernels for one arch only:

| arch | native kernel | what `flash_attn_fwd` does |
|---|---|---|
| sm_120 / sm_121 (RTX 5090, RTX PRO 6000) | MXFP8 FlashAttention: contiguous-KV prefill, paged-KV single-token decode, paged-KV ragged prefill (extend); head_dim ∈ {32, 64, 128, 256}, native GQA | torch SDPA; the MXFP8 kernels are the separate entries below |
| sm_90 (H200), sm_100 / sm_103 (B200 / B300) | none | torch SDPA |

The MXFP8 kernels do **no quantization**: the caller hands over FP8 (E4M3)
Q / K / V plus UE8M0 scale tensors in the layouts below, the way a serving
engine with an FP8 KV cache already holds them. The `pre_quantize_*`,
`quantize_*` and `compute_*_chan_scale` helpers exported next to them are
calibration- and test-time tools written in plain PyTorch, not runtime ops.

Every MXFP8 entry (`mxfp8_fwd`, `mxfp8_paged_prefill_fwd`, `PrefillPagedPlan` /
`plan_paged_prefill`, `mxfp8_decode_paged_fwd`, `DecodePagedPlan` /
`plan_decode_paged`) checks the architecture first and raises
`NotImplementedError` on anything but sm_120/sm_121, naming `flash_attn_fwd` as
the alternative. Until 2026-09-30 these entries were reachable only through
`fish_scales_ops.attention.backends.*`; those modules still work and export the
same objects.

## Quick start — convenience path (`flash_attn_fwd`)

```python
import torch, fish_scales_ops as fso

B, S, H_q, H_kv, D = 2, 4096, 32, 8, 128
q = torch.randn(B, S, H_q,  D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, S, H_kv, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, S, H_kv, D, dtype=torch.bfloat16, device="cuda")

o = fso.attention.flash_attn_fwd(q, k, v, causal=True)       # [B, S, H_q, D], bf16 like the inputs
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
) -> torch.Tensor | tuple[torch.Tensor, FlashAttnDispatch]   # [B, S_q, H_q, D]
```

Every dtype on every arch takes `torch.nn.functional.scaled_dot_product_attention`
(FP8 inputs are dequantised to BF16 first; GQA is expanded with
`repeat_interleave`; a local window builds an explicit additive mask). The output
has the dtype SDPA returns for the inputs it receives: bf16 for bf16 and FP8
inputs, fp16 for fp16 inputs. With `return_dispatch=True` the call returns the
pair `(out, FlashAttnDispatch)`. The `force_kernel` argument was removed on
2026-09-30: it never selected anything. Use the MXFP8 entries below for the
native kernels.

`causal=True` follows SDPA's alignment: when `S_q != S_k` the causal mask is
aligned to the top-left corner (query `i` sees keys `0 … i`), and the windowed
mask uses the same diagonal. A call whose queries are the last `S_q` positions
of a longer key sequence therefore does not get the mask it needs from
`causal=True`: a single-token decode call (`S_q = 1`) should pass
`causal=False`, and an extend call (`1 < S_q < S_k`) needs another attention
path, such as the paged MXFP8 prefill below on sm_120/121.

## SM120 MXFP8 prefill — `mxfp8_fwd`

```python
import fish_scales_ops as fso

q_fp8, q_sc = fso.attention.pre_quantize_q(q_bf16)   # test / calibration helpers
k_fp8, k_sc = fso.attention.pre_quantize_k(k_bf16)
v_fp8, v_sc = fso.attention.pre_quantize_v(v_bf16)   # V comes out pre-transposed [B, D, H_kv, S_k]
o = fso.attention.mxfp8_fwd(q_fp8, q_sc, k_fp8, k_sc, v_fp8, v_sc,
                            softmax_scale=1 / D**0.5, causal=True)
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
wrapper copies into `out`. On any other arch the entry raises
`NotImplementedError` before launching anything.

## SM120 MXFP8 paged decode — `mxfp8_decode_paged_fwd` (single-call and plan/run)

Single-token (`S_q = 1`) decode against a paged FP8 KV cache. Q and K are
scaled by the hardware block-scale operand of the QK MMA; V uses a
**per-channel FP32 scale applied in the epilogue**, so the V scale does not
depend on the page size.

```python
import torch, fish_scales_ops as fso

k_sc = fso.attention.compute_k_chan_scale(k_calib)   # once per model: [H_kv, D/32] uint8
v_sc = fso.attention.compute_v_chan_scale(v_calib)   # once per model: [H_kv, D]    fp32

# A reference paged cache for B sequences of S tokens (S % page == 0). The helper
# returns per-sequence pages, K [B, S/page, page, H_kv, D] and V [B, S/page, D, H_kv, page];
# the kernel reads a page pool through block_table, so flatten the pages into the pool.
page, n = 32, S // 32
K, _, V, _ = fso.attention.quantize_kv_to_paged(k_full, v_full, page_size=page,
                                                k_chan_scale=k_sc, v_chan_scale=v_sc)
K_cache = K.reshape(B * n, page, H_kv, D)            # [num_pages, page, H_kv, D]
V_cache = V.reshape(B * n, D, H_kv, page)            # [num_pages, D, H_kv, page]
block_table = torch.arange(B * n, dtype=torch.int32, device="cuda").reshape(B, n)
seq_lens = torch.full((B,), S, dtype=torch.int32, device="cuda")
q_fp8, q_sc = fso.attention.quantize_q_grouped(q_bf16, H_kv)   # q_bf16 [B, H_q, D]

# one-shot: allocates the split-K scratch and zeroes the sync counter every call
o = fso.attention.mxfp8_decode_paged_fwd(q_fp8, q_sc, K_cache, k_sc, V_cache, v_sc,
                                         block_table, seq_lens, softmax_scale=1 / D**0.5)  # bf16 [B, H_q, D]

# plan/run: scratch allocated once, counter tracked across calls (decode loops, CUDA graphs)
plan = fso.attention.plan_decode_paged(B=B, H_q=H_q, H_kv=H_kv, D=D, max_blocks=n)  # kv_split_k=None → auto
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
about two waves over this device's SMs, read once per device; the plan owns the
`m/l/o_partial` scratch and the `sync_counter` it needs. `DecodePagedPlan.run`
refuses non-contiguous inputs instead of copying them (a copy inside a capture
would bake a stale pointer into the graph). When called during stream capture it
re-zeroes the counter inside the graph so replays are self-contained, and every
eager `run` on a plan that has been captured re-zeroes it too, because the
replays move the device counter without the host seeing it. One plan must not
run on two streams at once: its scratch is shared.

## SM120 MXFP8 paged prefill / extend — `mxfp8_paged_prefill_fwd` (single-call and plan/run)

Ragged Q against a paged KV cache, the FlashInfer `BatchPrefillWithPagedKVCache`
calling convention (so an sglang FlashInfer backend can be rewired to it).

```python
import fish_scales_ops as fso

q_fp8, q_sc = fso.attention.quantize_q_ragged(q_bf16, H_q)   # [total_q, H_q, D], [ceil(total_q/16), H_q, D/32]
o = fso.attention.mxfp8_paged_prefill_fwd(q_fp8, q_sc, K_pool, k_sc, V_pool, v_sc,
                                          qo_indptr, paged_kv_indices, paged_kv_indptr,
                                          paged_kv_last_page_len, causal=True)   # bf16 [total_q, H_q, D]

plan = fso.attention.plan_paged_prefill(qo_indptr_cpu=qo_indptr.cpu(), num_q_heads=H_q)  # packs the (b, h, q_tile) work list once
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
host to build the work list each call; the plan does that once and is the
CUDA-graph form (`run` also refuses non-contiguous inputs). A plan is valid for
the `qo_indptr` it was built from — every request's Q length, not only the batch
size. `run` cannot check that without a device-to-host copy, so a plan reused
with different Q lengths computes the wrong rows without an error; build a new
plan whenever any request's Q length changes.

## torch.ops bindings (lower-level)

The schemas below are generated from the registrations in the source.

<!-- BEGIN GENERATED torch.ops schemas (attention): do not edit by hand; regenerated from the m.def strings and the Python custom_op registrations -->

Registered in `csrc/attention/csrc/flash_attn_ext.cpp`:

```
mxfp8_attn_fwd(Tensor q_fp8, Tensor q_scales, Tensor k_fp8, Tensor k_scales, Tensor v_fp8, Tensor v_scales, float softmax_scale, bool causal) -> Tensor
mxfp8_decode_paged(Tensor q_fp8, Tensor q_scales, Tensor k_pool, Tensor k_chan_scale, Tensor v_pool, Tensor v_chan_scale, Tensor block_table, Tensor seq_lens, Tensor m_partial, Tensor l_partial, Tensor o_partial, Tensor sync_counter, int num_splits, int target_counter, float softmax_scale) -> Tensor
mxfp8_attn_fwd_paged(Tensor q_fp8, Tensor q_scales, Tensor k_pool, Tensor k_chan_scale, Tensor v_pool, Tensor v_chan_scale, Tensor qo_indptr, Tensor paged_kv_indices, Tensor paged_kv_indptr, Tensor paged_kv_last_page_len, Tensor work_units, int total_work, bool causal, float softmax_scale) -> Tensor
```

<!-- END GENERATED torch.ops schemas (attention) -->

All three return a new bf16 tensor: `[B, S_q, H_q, D]` from `mxfp8_attn_fwd`,
`[B, H_q, D]` from `mxfp8_decode_paged` and `[total_q, H_q, D]` from
`mxfp8_attn_fwd_paged`. `m_partial` and `l_partial` (fp32 `[B, H_q, num_splits]`),
`o_partial` (fp32 `[B, H_q, num_splits, D]`) and `sync_counter` (int32
`[B, H_q]`) are the decode op's split-K scratch, and `target_counter` is the
counter value at which a split's CTA finalizes; `work_units` is the packed int32
list of `(b, h_q, q_tile)` triples the paged prefill walks. On any architecture
other than sm_120/121 the raw ops raise `RuntimeError` from the launcher.
The wrappers own the scratch buffers, the split-K sync-counter bookkeeping,
the work-unit packing, contiguity checks and the arch check; call the raw ops
only when integrating into a graph compiler that reproduces those.

## Reference and test helpers

`backends.mxfp8_ref` (`quantize_mxfp8`, `dequantize_mxfp8`, `qk_mxfp8`) and
`backends.mxfp8_attn_ref` (`mxfp8_qk_mixed_pv_fwd`) are plain-PyTorch
reference implementations of the block-scaled attention arithmetic. They are
slow by design and are not part of the runtime API. Both import as
`fish_scales_ops.attention.backends.<name>`; `mxfp8_attn_ref` reaches
`mxfp8_ref` through a package-relative import. No test in `tests/` exercises
their arithmetic today; the import check in `tests/attention/test_smoke.py`
loads both modules.
