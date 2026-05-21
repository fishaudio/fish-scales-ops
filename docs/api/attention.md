# `fish_scales_ops.attention` — public API

Forward-only flash attention. Two production paths plus a torch-SDPA
convenience dispatch:

* **MXFP8 prefill** — `fso.attention.backends.sm120_mxfp8.mxfp8_fwd(...)`
  wraps `torch.ops.fish_scales_ops.mxfp8_attn_fwd`. SM120 only.
  Pre-quantized Q/K/V + UE8M0 scales.
* **MXFP8 paged decode** —
  `fso.attention.backends.sm120_mxfp8_decode.mxfp8_decode_paged_fwd(...)`
  and the lower-overhead plan/run pair `plan_decode_paged` +
  `DecodePagedPlan.run`. Wraps `torch.ops.fish_scales_ops.mxfp8_decode_paged`.
  SM120 only.
* **`flash_attn_fwd(q, k, v, ...)`** — convenience entrypoint that
  always routes through torch SDPA today (FP8 inputs are dequantised
  to BF16 first). Use this when you don't want to manage MXFP8
  quantization yourself; use the pre-quantized backend ops above for
  the SM120 fast path.

## Quick start — convenience path

```python
import torch, fish_scales_ops as fso

B, S, H_q, H_kv, D = 2, 4096, 32, 8, 128
q = torch.randn(B, S, H_q,  D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, S, H_kv, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, S, H_kv, D, dtype=torch.bfloat16, device="cuda")

o = fso.attention.flash_attn_fwd(q, k, v, causal=True)   # bf16 [B, S, H_q, D]
```

GQA is handled natively for the MXFP8 ops (stride-0 K/V broadcast in
the kernel); the SDPA path does manual `repeat_interleave`.

## `flash_attn_fwd` signature

```python
fso.attention.flash_attn_fwd(
    q, k, v,                     # [B, S, H, D] CUDA tensors, bf16/fp16/fp8
    *,
    softmax_scale=None,          # default: 1 / sqrt(D)
    causal=False,
    window_left=-1,              # local-attention window; -1 disables
    window_right=-1,
    return_dispatch=False,       # also return FlashAttnDispatch metadata
    force_kernel=None,           # bypass selector; only "torch_fallback_fwd" is honoured
) -> torch.Tensor                # bf16 [B, S, H_q, D]
```

All dtypes currently route through `torch.nn.functional.scaled_dot_product_attention`
(FP8 → dequant to BF16 first). For the SM120 MXFP8 native kernel, call
the backend ops directly — they have a strict pre-quantized interface
described below.

## SM120 MXFP8 prefill — `mxfp8_fwd`

```python
from fish_scales_ops.attention.backends.sm120_mxfp8 import (
    mxfp8_fwd, pre_quantize_q, pre_quantize_k, pre_quantize_v,
)

# Pre-quantize (calibration / test helpers; production engines should
# mirror these layouts internally).
q_fp8, q_sc = pre_quantize_q(q_bf16)        # [B, S, H_q, D]    + uint8 scales
k_fp8, k_sc = pre_quantize_k(k_bf16)        # [B, S, H_kv, D]   + uint8 scales
v_fp8, v_sc = pre_quantize_v(v_bf16)        # [B, D, H_kv, S]   + uint8 scales (V pre-transposed)

o = mxfp8_fwd(q_fp8, q_sc, k_fp8, k_sc, v_fp8, v_sc,
              softmax_scale=1.0 / D**0.5, causal=True)   # bf16 [B, S, H_q, D]
```

Constraints: `H_q % H_kv == 0`, `D ∈ {32, 64, 128}`, `S_q % 64 == 0`,
`S_k % bc == 0` (bc=128 for D=32, bc=64 for D∈{64,128}).
Calling on sm < 120 raises `RuntimeError("mxfp8_attn_fwd launch failed: operation not supported")`.

## SM120 MXFP8 paged decode — `mxfp8_decode_paged_fwd`

Single-token (S_q=1) decode against a paged FP8 KV cache.

```python
from fish_scales_ops.attention.backends.sm120_mxfp8_decode import (
    mxfp8_decode_paged_fwd, plan_decode_paged,
    quantize_q_grouped, quantize_kv_to_paged,
)

# One-time pre-quant of Q + paged KV cache.
q_fp8, q_sc          = quantize_q_grouped(q_bf16, H_kv)
K_cache, k_sc, V_cache, v_sc = quantize_kv_to_paged(k_full, v_full, page_size=64)

# Single-call API (allocates partial scratch every call).
o = mxfp8_decode_paged_fwd(
    q_fp8, q_sc, K_cache, k_sc, V_cache, v_sc,
    block_table, seq_lens,
    softmax_scale=1.0 / D**0.5, kv_split_k=None)         # bf16 [B, H_q, D]

# Plan/run API (no per-step allocation; recommended for decode loops).
plan = plan_decode_paged(B=B, H_q=H_q, H_kv=H_kv, D=D, max_blocks=max_blocks)
for step in range(num_steps):
    o = plan.run(q_fp8, q_sc, K_cache, k_sc, V_cache, v_sc,
                 block_table, seq_lens, softmax_scale=scale, out=output_tensor)
```

Layout contracts: see module docstring of
`fso.attention.backends.sm120_mxfp8_decode`. SM120-only.

## torch.ops bindings (lower-level)

```
torch.ops.fish_scales_ops.mxfp8_attn_fwd(
    q_fp8, q_scales, k_fp8, k_scales, v_fp8, v_scales,
    softmax_scale, causal) -> Tensor                       # bf16 [B, S, H_q, D]

torch.ops.fish_scales_ops.mxfp8_decode_paged(
    q_fp8, q_scales, k_pool, k_pool_scales, v_pool, v_pool_scales,
    block_table, seq_lens,
    m_partial, l_partial, o_partial, sync_counter,
    num_splits, target_counter, softmax_scale) -> Tensor   # bf16 [B, H_q, D]
```

The Python wrappers above take care of scratch allocation, sync-counter
bookkeeping, contiguous-tensor normalisation, and arch checks. Reach for
the `torch.ops` form only if you're integrating into a graph compiler.
