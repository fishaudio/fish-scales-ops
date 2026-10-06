# `arch/sm90/fp8/jit/` — third-party JIT subtree

This directory vendors DeepSeek's `deep_gemm` headers (MIT-licensed)
and the NVRTC pipeline that compiles them on first call. **It is the
only AOT-incompatible code path in the library**: on sm_90, the first
`linear_fp8` call NVRTC-compiles a kernel specialised for the runtime
`(N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, NUM_TMA_MULTICAST)`
tuple and keeps the resulting cubin in memory for the rest of the process.

## Compile-time cost

~300–800 ms per unique tuple at first call; cached in memory after that, so
every later launch of the same kernel costs one name construction and one
map lookup (`Compiler::build`). The dispatcher heuristic in
`arch/sm90/fp8/dispatch.cuh::gemm_dispatch_sm90` picks the tuple, so
distinct shapes generally produce distinct cubins.

## Disk cache (opt-in)

By default nothing is read from or written to disk. With
`FSO_JIT_DUMP_CUBIN=1` (or `FSO_JIT_USE_NVCC=1`, whose cubins always go
through it) each cubin is written to, and on a later process loaded from,
`<root>/cache/<key>_<kernel name>/`, with `<root>` = `FSO_JIT_CACHE_DIR` or
`${XDG_CACHE_HOME:-$HOME/.cache}/fish_scales_ops/jit`. `<key>` is 16 hex
digits of an FNV-1a 64 over the generated kernel source, the complete flag
list (`FSO_JIT_EXTRA_FLAGS` and the `-I` directories included), the compiler
(NVRTC version, or the nvcc path) and every file of the `deep_gemm/`
directory the build compiles against, so editing a header, changing a flag
or switching compilers never loads a stale cubin. The key is computed only on
an in-memory miss, never on the per-launch path. The knobs and the
`TRTLLM_DG_*` aliases they replace are listed in `docs/api/compat.md`
(env-var overrides).

## Why JIT, and what it would take to AOT this

The kernel template `fp8_gemm_kernel<SHAPE_N, SHAPE_K, BLOCK_M, ...>`
takes `SHAPE_N` and `SHAPE_K` as **compile-time constants** (used for
SMEM layout sizing, loop unroll bounds, and `if constexpr` fast-paths
on `SHAPE_K % kFullKOfAllStages == 0`). To AOT this, every supported
production `(N, K)` pair would need an explicit instantiation in the
.so — feasible, but a separate refactor. It was scoped as a follow-up when
the library was extracted from blockscale_gemm and has not been started; the
JIT remains the only sm_90 path, and the NVRTC build that compiles it is
therefore part of the measured kernel: another NVRTC version emits different code.

## Which NVRTC

The extension does not link libnvrtc. On the first NVRTC compile (or the
first disk-cache key, or the first `jit_compiler_sm90` call) `jit_utils.cuh`
loads one library with `dlopen(RTLD_NOW | RTLD_LOCAL)` and resolves the nine
NVRTC functions the JIT calls into a table; every call in `compiler.cuh` goes
through that table, so none can bind to the `libnvrtc.so.13` torch loaded.
The library is `FSO_JIT_NVRTC_LIB` when set, else `_nvrtc/libnvrtc.so.13`
next to the extension (found with `dladdr`), which `scripts/vendor_nvrtc.py`
unpacks from the pinned `nvidia-cuda-nvrtc==13.0.88` wheel (the NVRTC torch 2.13.0+cu130
depends on). Nothing else is
tried: a failed load raises `RuntimeError` with the remedy. A version other
than 13.0 is used with one stderr line. `FSO_JIT_USE_NVCC` never loads NVRTC.

## Which headers

`getJitIncludeDirs()` in `compiler.cuh` resolves the include directories once
per process and takes the first of these that names any:

1. `FSO_JIT_INCLUDE_DIRS`, a colon-separated list;
2. `_jit_include/` next to the extension (found with `dladdr`, like the
   bundled NVRTC), when it exists. A wheel built by `scripts/build_wheel.sh`
   carries one include root there: this `deep_gemm/` directory, the CUTLASS
   `include/` directory and the CUDA 13.0.3 headers of the build container
   (CCCL merged into the root, the CUDA library headers left out);
3. `FSO_JIT_INCLUDE_DIRS_DEFAULT`, the source-tree directories
   `python/setup.py` bakes into an in-place build.

The kernels currently open seven headers of `deep_gemm/` (the generated
source's `nvrtc_std.cuh`, `nvrtc_cutlass.cuh` and `fp8_gemm_impl.cuh`, and the
four that one includes) and, from CUDA, `cuda_fp8.h`, `cuda_bf16.h` and
`cuda_fp16.h` with their `.hpp` parts. The packaged tree still ships whole
directories, so that a kernel that starts including another CUDA or CUTLASS
header keeps compiling. With `FSO_JIT_DEBUG=1` the choice is logged as
`sm_90 JIT include directories (<origin>): <list>`.

## Lint contract

AOT code (anything outside this directory) must **not** include
`<deep_gemm/...>` or any `arch/sm90/fp8/jit/...` header except for
`arch/sm90/fp8/dispatch.cuh`, which is the dedicated sm_90 entry
point. (A companion `moe_dispatch.cuh` existed until 2026-05-21; the sm_90
grouped MoE path revived in 2026-09 — the grouped and strided-batched
schedulers under `deep_gemm/` — is reached through the same `dispatch.cuh`
entry, so the contract is unchanged.) Verify with
`bash tests/gemm/unit/lint_jit_isolation.sh` from the repository root, or:

```bash
grep -rn '#include.*\(deep_gemm/\|arch/sm90/fp8/jit/\)' csrc/ \
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
| `deep_gemm/compiler.cuh` | NVRTC driver — generates kernel source, compiles through the NVRTC table (loaded once per process), keys the opt-in disk cache by content |
| `deep_gemm/runtime.cuh` | runtime cubin loader, the in-memory runtime cache and the `FSO_JIT_*` knobs |
| `deep_gemm/jit_utils.cuh` | `get_best_gemm_config` + `get_smem_size` heuristic (every caller), `get_dense_gemm_config` (the dense GEMM's two tile rules on top of it); the NVRTC function table and its loader |
| `deep_gemm/{mma,tma}_utils.cuh` | WGMMA + TMA helpers |
| `deep_gemm/utils.cuh` | misc primitives (`ceil_div`, `lane_id`, etc.) |
| `deep_gemm/nvrtc_std.cuh` | minimal `<type_traits>` / `<utility>` shim for NVRTC |
| `deep_gemm/nvrtc_cutlass.cuh` | minimal CUTLASS shim for NVRTC |

`nvrtc_*.cuh` files are **only** parsed by NVRTC at JIT time, never by
the host AOT compile.
