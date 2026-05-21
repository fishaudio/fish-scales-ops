# `arch/sm90/fp8/jit/` — third-party JIT subtree

This directory vendors DeepSeek's `deep_gemm` headers (MIT-licensed)
and the NVRTC pipeline that compiles them on first call. **It is the
only AOT-incompatible code path in the library**: on sm_90, the first
`linear_fp8` call NVRTC-compiles a kernel specialised for the runtime
`(N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, NUM_TMA_MULTICAST)`
tuple and caches the resulting cubin under `~/.tensorrt_llm/cache/`.

## Compile-time cost

~300–800 ms per unique tuple at first call; cached after. The dispatcher
heuristic in `arch/sm90/fp8/dispatch.cuh::gemm_dispatch_sm90` picks the
tuple, so distinct shapes generally produce distinct cubins.

## Why JIT, and what it would take to AOT this

The kernel template `fp8_gemm_kernel<SHAPE_N, SHAPE_K, BLOCK_M, ...>`
takes `SHAPE_N` and `SHAPE_K` as **compile-time constants** (used for
SMEM layout sizing, loop unroll bounds, and `if constexpr` fast-paths
on `SHAPE_K % kFullKOfAllStages == 0`). To AOT this, every supported
production `(N, K)` pair would need an explicit instantiation in the
.so — feasible, but a separate refactor sub-plan. See REORG_PLAN.md
phase 5 (D2.c) and the discussion in
`docs/skills/blockscale-gemm-tuning/references/v15a-transfer-plans.md`.

## Lint contract

AOT code (anything outside this directory) must **not** include
`<deep_gemm/...>` or any `arch/sm90/fp8/jit/...` header except for
`arch/sm90/fp8/dispatch.cuh`, which is the dedicated sm_90 entry
point. (The companion `moe_dispatch.cuh` was removed 2026-05-21 with
the rest of the MoE path.) Verify with:

```bash
grep -rn 'deep_gemm/\|arch/sm90/fp8/jit/' include/ src/ pytorch/csrc/ \
  | grep -v 'arch/sm90/fp8/jit/' \
  | grep -v 'arch/sm90/fp8/dispatch.cuh'
```

The result should be empty.

## File layout

| File | Role |
|---|---|
| `deep_gemm/fp8_gemm.cuh` | TMA descriptor builders + run-host wrappers |
| `deep_gemm/fp8_gemm_impl.cuh` | the kernel `fp8_gemm_kernel<...>` |
| `deep_gemm/scheduler.cuh` | persistent + grouped + strided-batched schedulers |
| `deep_gemm/compiler.cuh` | NVRTC driver — generates kernel source, compiles, caches |
| `deep_gemm/runtime.cuh` | runtime cubin loader and launch wrapper |
| `deep_gemm/jit_utils.cuh` | `get_best_gemm_config` + `get_smem_size` heuristic |
| `deep_gemm/{mma,tma}_utils.cuh` | WGMMA + TMA helpers |
| `deep_gemm/utils.cuh` | misc primitives (`ceil_div`, `lane_id`, etc.) |
| `deep_gemm/nvrtc_std.cuh` | minimal `<type_traits>` / `<utility>` shim for NVRTC |
| `deep_gemm/nvrtc_cutlass.cuh` | minimal CUTLASS shim for NVRTC |

`nvrtc_*.cuh` files are **only** parsed by NVRTC at JIT time, never by
the host AOT compile.
