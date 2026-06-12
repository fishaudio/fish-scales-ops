# `fish_scales_ops.gemm` — public API

Block-scaled FP8 GEMM, two flavors:

* **FP8 1×128 / 128×128** — 1×128 row-scale activations + 128×128
  block-scale weights. Both archs: sm_90 deep_gemm JIT (Hopper), sm_120
  CUTLASS `Sm120BlockScaledKernel` (Blackwell consumer).
* **MXFP8 1×32** (OCP MXFP8) — 1 UE8M0 byte per 32 K-elements on both
  operands. **sm_100/sm_103 (B200/B300, tcgen05 BlockScaled — build with
  `ARCH="10.0f"`) and sm_120/121**. Finer block-scale granularity →
  tighter quantization (cos ≈ 0.9993 vs 0.987–0.996 for 1×128) and
  ~20–30% faster than 1×128 on large M (sm_120 numbers).

  The MXFP8 scale tensor is an **opaque handle in the arch-native
  layout**: sm_120 emits int32 K-major `[pad(M,4), K/128]`; sm_100/103
  emits a 1-D `Sm1xxBlockScaledConfig` atom-interleaved buffer. Always
  quantize on the same device arch the GEMM runs on — the layouts are
  not interchangeable.

## Quick start

```python
import torch, fish_scales_ops as fso

sm120 = torch.cuda.get_device_capability(0)[0] >= 12

# --- FP8 1×128 (both archs) ---
wq, sw = fso.gemm.quantize_128x128_fp8(w_bf16)         # bf16 [N,K] → fp8 [N,K] + fp32 [⌈N/128⌉,⌈K/128⌉]
if sm120:
    sw = fso.gemm.repack_fp8_wgt_scales(sw)            # → int32 K-major [pad(N,4), K/512]

xq, sx = fso.gemm.quantize_1x128_fp8(x_bf16, use_ue8m0=sm120)
if sm120:
    sx = fso.gemm.repack_fp8_act_scales(sx)
y = fso.gemm.linear_fp8(xq, wq, sx, sw)                # bf16 [M, N]

# --- MXFP8 1×32 (sm_100/103 + sm_120) ---
mxfp8_ok = torch.cuda.get_device_capability(0)[0] in (10, 12)
if mxfp8_ok:
    wqm, swm = fso.gemm.quantize_1x32_fp8(w_bf16)      # fp8 [N,K] + int32-packed UE8M0
    xqm, sxm = fso.gemm.quantize_1x32_fp8(x_bf16)
    ym = fso.gemm.linear_mxfp8(xqm, wqm, sxm, swm)     # bf16 [M, N]
```

## API

| Function | In | Out | Notes |
|---|---|---|---|
| `quantize_1x128_fp8(x, use_ue8m0=False)` | bf16 `[..., K]`, K%128==0 | (fp8_e4m3 `[..., K]`, fp32 `[pad(M,4), K/128]`) | set `use_ue8m0=True` on sm_120 |
| `quantize_128x128_fp8(w)` | bf16 `[N, K]` | (fp8_e4m3 `[N, K]`, fp32 `[⌈N/128⌉, ⌈K/128⌉]`) | both archs |
| `linear_fp8(x_fp8, w_fp8, sx, sw)` | fp8 + fp32 *or* int32-packed scales | bf16 `[M, N]` | dispatches sm_90 deep_gemm JIT or sm_120 CUTLASS |
| `quantize_1x32_fp8(x)` | bf16 `[..., K]`, K%128==0 | (fp8_e4m3 `[..., K]`, opaque int32 scales, arch-native layout) | **sm_100/103 + sm_120**; UE8M0 baked in |
| `linear_mxfp8(x_fp8, w_fp8, sx_int32, sw_int32)` | fp8 + opaque int32 UE8M0 scales | bf16 `[M, N]` | **sm_100/103 + sm_120**; both operands MXFP8 1×32 |
| `linear_bf16(x, w)` | bf16 + bf16 | bf16 `[M, N]` | FP8 1×128 path, internal quant; use when both operands change every step |
| `linear_qx(x_bf16, w_fp8, sw)` | bf16 x + cached fp8 w | bf16 `[M, N]` | FP8 1×128 path, fused activation quant + GEMM |
| `repack_fp8_act_scales(sx_fp32)` | fp32 act scales | int32 K-major | **sm_120 only** — skip per-call repack |
| `repack_fp8_wgt_scales(sw_fp32)` | fp32 wgt scales | int32 K-major | **sm_120 only** — skip per-call repack |

The same names are also accessible under the `fso.gemm.fp8` / `fso.gemm.mxfp8` / `fso.gemm.bf16` submodules.

## Constraints

- `K % 128 == 0` everywhere.
- `N % 128 == 0` for `linear_fp8` and `linear_mxfp8` on sm_100/103/120.
- **FP8 1×128 on sm_120:** `use_ue8m0=True` is mandatory; without it the
  CUTLASS kernel returns NaN.
- **MXFP8 1×32:** sm_100/sm_103/sm_120/sm_121 only. Calling on sm_90
  raises `NotImplementedError`. Scale tensors are arch-native and opaque
  (see above) — do not move them between arch generations.
- **FP8 1×128 on sm_100/103:** not implemented (MXFP8 is the only
  block-scaled path on datacenter Blackwell).
- **sm_90 repack helpers:** do not call `repack_fp8_*_scales` on sm_90 —
  the deep_gemm kernel reads FP32 scales directly; passing the int32
  output to `linear_fp8` produces garbage.

## Picking the right op

| Situation | Op |
|---|---|
| sm_90 (H200) production inference (cached fp8 weight) | `linear_fp8` + pre-quantized weight; activation via `quantize_1x128_fp8` per call |
| sm_120a (Blackwell) production inference (cached fp8 weight) | `linear_fp8` + pre-packed scales (`repack_fp8_wgt_scales`), `quantize_1x128_fp8(use_ue8m0=True)` + `repack_fp8_act_scales` per call |
| sm_120 production with finer quantization | `linear_mxfp8` + `quantize_1x32_fp8` — better cos AND faster than 1×128 at M ≥ 512 |
| sm_100/sm_103 (B200/B300) production inference | `linear_mxfp8` + `quantize_1x32_fp8` — the only block-scaled path on datacenter Blackwell |
| Both operands change every step (decode-1, RoPE, etc.) | `linear_bf16` |
| Activation streaming + cached fp8 weight + one ATen op | `linear_qx` (FP8 1×128 only) |

## Lower-level interface

The functional wrappers are thin shims over `torch.ops.fish_scales_ops.*`:

```python
torch.ops.fish_scales_ops.{linear_bf16, linear_fp8, linear_qx,
                           quantize_1x128, quantize_128x128,
                           repack_fp8_act_scales, repack_fp8_wgt_scales,
                           quantize_1x32, repack_mxfp8_scales, linear_mxfp8_raw}
```

`linear_mxfp8_raw` is the raw entry point: it expects pre-packed int32
UE8M0 scales (output of `quantize_1x32` + `repack_mxfp8_scales`, both
auto-applied inside the `fso.gemm.linear_mxfp8` Python wrapper).

## CUDA-graph compatibility

All ops are graph-capture safe **after a single eager warmup** that
populates static state (cudaFuncSetAttribute guard, Stream-K side-stream
pool, Sm120BfPackPool int32 scratch, PyTorch op dispatch caches). See
`tests/gemm/unit/test_cuda_graph.py` for the exact warmup discipline.

## Env-var overrides (debugging / A-B)

| Var | Effect |
|---|---|
| `FSO_FORCE_TILE="TM,TN,ST"` | Force a specific tile, bypass cascade (FP8 + MXFP8) |
| `FSO_FORCE_KSPLIT="N"` | Force Stream-K split factor |
| `FSO_FORCE_MIN_BLOCKS=N` | Override `__launch_bounds__(maxThreadsPerBlock, minBlocks)` |
| `FSO_FORCE_SMALLM=1` | Force the MXFP8 small-M (TileM=16) builder variant |
| `FSO_FORCE_SCHED_GROUP=N` | Set CUTLASS scheduler 1-D blocks-per-group |
| `FSO_DISABLE_OVERRIDES=1` | Disable shape-specific single-launch overrides |
| `FSO_DISABLE_STREAMK=1` | Disable Stream-K K-split entirely |
| `FSO_JIT_INCLUDE_DIRS=…` | Override NVRTC include path for sm_90 deep_gemm |
| `FSO_STREAMK_POOL_MB=N` | Stream-K side-stream scratch pool capacity |

All read once at first call and cached for graph-capture safety.
