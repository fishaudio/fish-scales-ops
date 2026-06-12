/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SM100/SM103 MXFP8 (1×32, SFVecSize=32) dispatch and kernel launcher.
//
// Mirrors the sm_120 dispatcher's host-overhead discipline:
//   * E9-equivalent: a thread-local cache of *initialized*
//     GemmUniversalAdapter instances keyed by (ptrs, shape). On a hit the
//     call is just adapter.run(stream) — no cuTensorMapEncodeTiled, no
//     cudaFuncSetAttribute (the adapter does both only in initialize()).
//   * E8-equivalent: cudaFuncSetAttribute happens inside
//     Gemm::initialize(), i.e. only on cache miss. CUDA-graph capture is
//     therefore safe as long as callers warm up eagerly first — the same
//     contract test_cuda_graph.py enforces for sm_120.
//   * Workspace: the dense BlockScaled kernel with the default tile
//     scheduler needs no global workspace on the shapes we serve; if
//     CUTLASS ever asks for one we grow a thread-local pool OUTSIDE
//     capture (same contract as Sm120BfPackPool) and abort with a clear
//     message if a capture-mode call needs more.
//
// Cascade status: bring-up heuristic only (C1). The (M, tiles_n) rules
// below are placeholders pending the b300 tile sweep (FSO_FORCE_TILE grid
// + cudagraph-µs bench per docs/perf.md methodology). Retune before
// trusting any cell.
//
// Env overrides: FSO_FORCE_TILE=TM,TN,ST reuses the sm_120 wire format so
// the shared sweep harness drives this path unchanged. For sm_100 the
// third field selects the (SM-count, TileK, cluster, scheduler) variant
// instead of the stage count (stages are StageCountAutoCarveout-derived):
//   ST=1 → 1SM cluster(1,1) K128          ST=2 → 2SM cluster(2,1) K128
//   ST=3 → 1SM cluster(1,1) K256          ST=4 → 2SM cluster(2,1) K256
//   ST=5 → 2SM cluster(2,2) K128          ST=6 → 2SM cluster(2,2) K256
//   ST=7 → 1SM cluster(1,1) K128 StreamK  ST=8 → 1SM cluster(1,1) K256 StreamK
//   ST=9 → 1SM parallel split-K (two-kernel; splits from FSO_FORCE_KSPLIT,
//          else auto pick_splits) — (128,128) only
//   ST=10 → 1SM cluster(2,2) K128 (TMA multicast ×2 on both A and B — the
//           nvjet mid-band shape; ncu F3)      ST=11 → 2SM cluster(2,2) K256
// Unknown combos fall through to the cascade (same behavior as sm_120).

#pragma once

#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#include "blockscale_gemm/arch/sm100/mxfp8/gemm_types.cuh"
#include "blockscale_gemm/arch/sm120/common/env_overrides.cuh" // FSO_FORCE_TILE parsing (arch-agnostic)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace sm100_blockscaled_gemm
{

namespace detail
{

// FSO_FORCE_SWIZZLE: raster swizzle group size for the default (CLC)
// scheduler — sweep knob for L2 locality on large shapes. 0 = CUTLASS
// heuristic default. Read once (function-static), same contract as the
// other FSO_* overrides.
inline int read_force_swizzle() noexcept
{
    static int s_cache = -2;
    if (s_cache == -2)
    {
        char const* env = std::getenv("FSO_FORCE_SWIZZLE");
        s_cache = (env && *env) ? std::atoi(env) : 0;
    }
    return s_cache;
}

// Thread-local CUTLASS workspace pool. Expected to stay empty (dense
// BlockScaled + default scheduler needs no workspace); exists so a future
// scheduler change degrades to a clear runtime contract instead of UB.
struct Sm100WorkspacePool
{
    void* ptr = nullptr;
    std::size_t bytes = 0;

    static Sm100WorkspacePool& instance()
    {
        static thread_local Sm100WorkspacePool p;
        return p;
    }

    void* ensure(std::size_t needed)
    {
        if (needed == 0)
            return nullptr;
        if (bytes >= needed)
            return ptr;
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(nullptr, &cap);
        if (cap == cudaStreamCaptureStatusActive)
        {
            std::fprintf(stderr,
                "[blockscale_gemm] Sm100WorkspacePool: needs %zu bytes during stream capture but pool has %zu. "
                "Warm up linear_mxfp8 on the largest shape before capture.\n",
                needed, bytes);
            std::abort();
        }
        if (ptr)
            cudaFree(ptr);
        cudaMalloc(&ptr, needed);
        bytes = needed;
        return ptr;
    }
};

} // namespace detail

// Launch one (TileM, TileN, ClusterM, ClusterN) instantiation. Returns
// cudaSuccess on success; kernel runtime errors surface on the next sync
// (post-launch cudaGetLastError check intentionally omitted — see the E8
// note in the sm_120 launcher).
//
// Capture-safety split (mirror of the sm_120 E8/E9 discipline):
//   * cudaFuncSetAttribute → per-instantiation static guard, fires on the
//     FIRST call only. Callers must warm up eagerly before capture.
//   * Params (TMA descriptors) → thread-local cache. A cache MISS during
//     capture is safe: to_underlying_arguments is host-only work, and
//     GemmUniversalAdapter's *static* run(params, stream) issues nothing
//     but the (cluster-)launch itself.
template <int TileM, int TileN, int ClusterM, int ClusterN, int TileK = 128, bool UseStreamK = false,
    bool NoSmemEpi = false>
cudaError_t launch_sm100_mxfp8_gemm(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int M, int N, int K, cudaStream_t stream, int swizzle = 0)
{
    using Config = Sm100MxFP8GemmConfig<TileM, TileN, ClusterM, ClusterN, TileK, UseStreamK, NoSmemEpi>;
    using Gemm = typename Config::Gemm;
    using GemmKernel = typename Config::GemmKernel;
    using Args = typename Gemm::Arguments;
    using Params = typename Gemm::Params;

    auto ptr_A = reinterpret_cast<cutlass::float_e4m3_t const*>(mat_a);
    auto ptr_B = reinterpret_cast<cutlass::float_e4m3_t const*>(mat_b);
    auto ptr_SFA = reinterpret_cast<cutlass::float_ue8m0_t const*>(scales_a);
    auto ptr_SFB = reinterpret_cast<cutlass::float_ue8m0_t const*>(scales_b);
    auto ptr_D = reinterpret_cast<cutlass::bfloat16_t*>(mat_d);

    auto problem = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M, N, 1});
    auto layout_SFA = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem);
    auto layout_SFB = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem);

    Args args{cutlass::gemm::GemmUniversalMode::kGemm, problem,
        {ptr_A, stride_A, ptr_B, stride_B, ptr_SFA, layout_SFA, ptr_SFB, layout_SFB},
        {{1.0f, 0.0f}, nullptr, stride_D, ptr_D, stride_D}};

    if constexpr (UseStreamK)
    {
        // splits > 1 bypasses the stream-K heuristic in favour of an
        // explicit split-K decomposition (the heuristic mode measured
        // WORSE than no split at all on down-class decode shapes —
        // 30.9 µs vs 17.5 µs at M=1, 2026-07-05 sweep). FSO_FORCE_KSPLIT
        // drives the sweep; the cascade sets nothing yet.
        int const ks = tensorrt_llm::kernels::blockscale_gemm::read_force_tile().ks;
        if (ks > 1)
            args.scheduler.splits = ks;
        // Probe knobs (2026-07-06, both read once into static caches):
        //   FSO_SK_NDET=1   → ReductionMode::Nondeterministic (atomic
        //                     accumulation; the 2026-07-05 losses were all
        //                     under the default Deterministic turnstile
        //                     lock, which serializes the fixup).
        //   FSO_SK_DECOMP   → 1=DataParallel 2=SplitK 3=StreamK (default
        //                     heuristic).
        static int const s_ndet = []() {
            char const* e = std::getenv("FSO_SK_NDET");
            return (e && *e) ? std::atoi(e) : 0;
        }();
        static int const s_decomp = []() {
            char const* e = std::getenv("FSO_SK_DECOMP");
            return (e && *e) ? std::atoi(e) : 0;
        }();
        using cutlass::gemm::kernel::detail::ReductionMode;
        using cutlass::gemm::kernel::detail::DecompositionMode;
        if (s_ndet)
            args.scheduler.reduction_mode = ReductionMode::Nondeterministic;
        if (s_decomp == 1)
            args.scheduler.decomposition_mode = DecompositionMode::DataParallel;
        else if (s_decomp == 2)
            args.scheduler.decomposition_mode = DecompositionMode::SplitK;
        else if (s_decomp == 3)
            args.scheduler.decomposition_mode = DecompositionMode::StreamK;
    }
    else
    {
        // Cascade-provided value; FSO_FORCE_SWIZZLE overrides for sweeps.
        int const sw = detail::read_force_swizzle();
        int const eff = sw > 0 ? sw : swizzle;
        if (eff > 0)
            args.scheduler.max_swizzle_size = eff;
        // FSO_FORCE_RASTER: 1 = AlongM, 2 = AlongN (default Heuristic).
        // Probe knob (2026-07-06): nvjet picks h_bz (horizontal raster) for
        // the narrow-N down-class at mid/large M where we lose ~17%.
        // PROBED NEUTRAL (raster_probe.txt): ±2% noise on every mid/large-M
        // cell; gate_up AlongN −5%. The CLC Heuristic already picks right —
        // do not wire into the cascade.
        static int const s_raster = []() {
            char const* e = std::getenv("FSO_FORCE_RASTER");
            return (e && *e) ? std::atoi(e) : 0;
        }();
        using cutlass::gemm::kernel::detail::RasterOrderOptions;
        if (s_raster == 1)
            args.scheduler.raster_order = RasterOrderOptions::AlongM;
        else if (s_raster == 2)
            args.scheduler.raster_order = RasterOrderOptions::AlongN;
    }

    // E9-style cache: key on everything the initialized Params bake in.
    struct CacheKey
    {
        void const* p[5];
        int dims[4];  // M, N, K, swizzle
        bool operator==(CacheKey const& o) const { return std::memcmp(this, &o, sizeof(o)) == 0; }
    };
    struct CacheHash
    {
        std::size_t operator()(CacheKey const& k) const noexcept
        {
            std::size_t h = 1469598103934665603ULL;
            auto const* p = reinterpret_cast<unsigned char const*>(&k);
            for (std::size_t i = 0; i < sizeof(k); ++i) { h ^= p[i]; h *= 1099511628211ULL; }
            return h;
        }
    };
    static thread_local std::unordered_map<CacheKey, Params, CacheHash> s_params_cache;
    CacheKey key;
    std::memset(&key, 0, sizeof(key));
    key.p[0] = ptr_A; key.p[1] = ptr_B; key.p[2] = ptr_SFA; key.p[3] = ptr_SFB; key.p[4] = ptr_D;
    key.dims[0] = M; key.dims[1] = N; key.dims[2] = K; key.dims[3] = swizzle;

    Params kernel_params;
    auto it = s_params_cache.find(key);
    if (it != s_params_cache.end())
    {
        kernel_params = it->second;
        if constexpr (UseStreamK)
        {
            // Stream-K reduction barriers must be clean before every launch;
            // re-enqueue the (stream-ordered, capture-legal) workspace init
            // even on a Params-cache hit. A captured init simply re-runs on
            // every replay — exactly the semantics the barriers need.
            std::size_t const ws_bytes = Gemm::get_workspace_size(args);
            void* ws = detail::Sm100WorkspacePool::instance().ensure(ws_bytes);
            if (ws_bytes > 0
                && GemmKernel::initialize_workspace(args, ws, stream) != cutlass::Status::kSuccess)
                return cudaErrorUnknown;
        }
    }
    else
    {
        if (Gemm::can_implement(args) != cutlass::Status::kSuccess)
            return cudaErrorInvalidValue;
        std::size_t const ws_bytes = Gemm::get_workspace_size(args);
        void* ws = detail::Sm100WorkspacePool::instance().ensure(ws_bytes);
        if (ws_bytes > 0)
        {
            // Stream-ordered workspace init (capture-legal; a captured init
            // simply re-runs on every replay, which is the semantics the
            // scheduler workspace wants anyway).
            if (GemmKernel::initialize_workspace(args, ws, stream) != cutlass::Status::kSuccess)
                return cudaErrorUnknown;
        }
        kernel_params = GemmKernel::to_underlying_arguments(args, ws); // host-only
        if (s_params_cache.size() < 64u)
            s_params_cache.emplace(key, kernel_params);
    }

    // E8-equivalent: set max dynamic smem once per instantiation, on the
    // first (eager) call. GemmUniversalAdapter::initialize would do this on
    // every cache miss — including mid-capture — so we do it ourselves.
    static bool s_smem_configured = false;
    if (!s_smem_configured)
    {
        if (GemmKernel::SharedStorageSize >= (48 << 10))
        {
            cudaError_t result = cudaFuncSetAttribute(cutlass::device_kernel<GemmKernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, GemmKernel::SharedStorageSize);
            if (result != cudaSuccess)
                return result;
        }
        s_smem_configured = true;
    }

    return Gemm::run(kernel_params, stream) == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorLaunchFailure;
}

// ---------------------------------------------------------------------------
// Parallel split-K (two-kernel) decode path.
//
// ncu 2026-07-06 (profile/ncu_sm100_2026-07-06/REPORT.md, F1): at down-class
// decode shapes (N=2560 → 20 tile-CTAs on 148 SMs) cuBLAS runs the SAME
// 128×128 tile as a data-parallel split-K×3 (grid 60) plus a trivial
// splitKreduce kernel, and wins ~0.75×. CUTLASS's StreamK scheduler is the
// wrong kind of split-K here (in-kernel semaphore fixup: heuristic 30.9 µs,
// splits=2 18.7 µs vs 16.7 plain). This path reproduces the cuBLAS scheme:
//   1. ONE batched GEMM launch (L = splits) writing FP32 partials to the
//      workspace pool — batch b covers K-slice [b*K/S, (b+1)*K/S). A/B are
//      addressed in-place via batch strides (K/S along K); the SF tensors
//      keep the full-K quantize layout via stride surgery (see below).
//   2. A float4 grid-stride reduce kernel: partials [S, M, N] → BF16 D.
// Both launches are capture-safe (workspace from the pool, eager-warmup
// contract unchanged).
//
// SF layout stride surgery: tile_atom_to_shape_SFA((M,N,K/S,S)) has the
// right SHAPE for a K-slice batch, but its strides describe S independent
// contiguous tensors. In the full-K quantize layout (K-minor blocks per
// 128-row block), the correct strides are:
//   * M-block-row stride = 512 * (K/128)      = default * S
//   * batch (L) stride   = 512 * (K/(128*S))  = the DEFAULT M-block-row stride
// so the fix is: stride(Mrest) *= S; stride(L) = old stride(Mrest).

static __global__ void sm100_mxfp8_splitk_reduce_kernel(
    float4 const* __restrict__ partials, ushort4* __restrict__ out, int splits, long mn4)
{
    long const stride = static_cast<long>(gridDim.x) * blockDim.x;
    for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; i < mn4; i += stride)
    {
        float4 acc = partials[i];
        for (int s = 1; s < splits; ++s)
        {
            float4 const p = partials[s * mn4 + i];
            acc.x += p.x; acc.y += p.y; acc.z += p.z; acc.w += p.w;
        }
        ushort4 o;
        o.x = __bfloat16_as_ushort(__float2bfloat16(acc.x));
        o.y = __bfloat16_as_ushort(__float2bfloat16(acc.y));
        o.z = __bfloat16_as_ushort(__float2bfloat16(acc.z));
        o.w = __bfloat16_as_ushort(__float2bfloat16(acc.w));
        out[i] = o;
    }
}

namespace detail
{

// Largest split count in [2, min(8, sm_count / tiles)] that divides the SF
// K-block count evenly (SF blocks are 128-K-wide and cannot straddle a
// slice). Returns 1 when no useful split exists — including when the tile
// grid already fills the machine, which self-limits the route to decode.
inline int pick_splits(int tiles, int kblks) noexcept
{
    static int s_sms = 0;
    if (s_sms == 0)
    {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&s_sms, cudaDevAttrMultiProcessorCount, dev);
        if (s_sms <= 0)
            s_sms = 148;
    }
    int const budget = s_sms / (tiles > 0 ? tiles : 1);
    for (int s = budget < 8 ? budget : 8; s >= 2; --s)
        if (kblks % s == 0)
            return s;
    return 1;
}

} // namespace detail

// Batched-slice GEMM into FP32 partials + reduce. Caller guarantees
// K % 128 == 0 and (K/128) % splits == 0 (pick_splits does).
template <int TileM, int TileN>
cudaError_t launch_sm100_mxfp8_gemm_splitk(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int M, int N, int K, int splits, cudaStream_t stream)
{
    // NoSmemEpi: decode band, partial stores are a sliver (see the M ≤ 32
    // NOSMEM rationale in the cascade). ElementD float → FP32 partials.
    using Config = Sm100MxFP8GemmConfig<TileM, TileN, 1, 1, 128, false, true, float>;
    using Gemm = typename Config::Gemm;
    using GemmKernel = typename Config::GemmKernel;
    using Args = typename Gemm::Arguments;
    using Params = typename Gemm::Params;

    auto ptr_A = reinterpret_cast<cutlass::float_e4m3_t const*>(mat_a);
    auto ptr_B = reinterpret_cast<cutlass::float_e4m3_t const*>(mat_b);
    auto ptr_SFA = reinterpret_cast<cutlass::float_ue8m0_t const*>(scales_a);
    auto ptr_SFB = reinterpret_cast<cutlass::float_ue8m0_t const*>(scales_b);

    int const kslice = K / splits;
    long const mn = static_cast<long>(M) * N;

    // Workspace: FP32 partials [splits, M, N], plus whatever CUTLASS asks
    // for (expected 0 with the default CLC scheduler) appended behind.
    std::size_t const partial_bytes = static_cast<std::size_t>(splits) * mn * sizeof(float);

    auto problem = cute::make_shape(M, N, kslice, splits);
    // In-place K-slices: leading strides stay full-K, batch stride = kslice.
    auto stride_A = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, kslice, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, kslice, 1});
    cute::get<0>(stride_A) = K;
    cute::get<2>(stride_A) = kslice;
    cute::get<0>(stride_B) = K;
    cute::get<2>(stride_B) = kslice;
    // Partials are genuinely batched [S, M, N] — packed stride.
    auto stride_D = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M, N, splits});

    // SF layouts: shape from the sliced problem, strides fixed up per the
    // header comment (Mrest *= S; L = old Mrest). In-place mutation of a
    // by-value copy keeps the exact LayoutSFA/LayoutSFB types.
    // LayoutSF{A,B} structure: shape (((32,4),Mrest), ((32,4),Krest), (1,L))
    // stride (((16,4),sMrest), ((0,1),512), (0,sL)) — the scalars to patch
    // are get<0,1> (M-block-row) and get<2,1> (batch).
    auto fix_sf = [splits](auto l) {
        auto const row = cute::get<0, 1>(l.stride());
        cute::get<0, 1>(l.stride()) = row * splits;
        cute::get<2, 1>(l.stride()) = row;
        return l;
    };
    typename Config::LayoutSFA layout_SFA
        = fix_sf(Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem));
    typename Config::LayoutSFB layout_SFB
        = fix_sf(Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem));

    // Params cache (E9 discipline; ws ptr participates via ptr_D).
    struct CacheKey
    {
        void const* p[5];
        int dims[4];  // M, N, K, splits
        bool operator==(CacheKey const& o) const { return std::memcmp(this, &o, sizeof(o)) == 0; }
    };
    struct CacheHash
    {
        std::size_t operator()(CacheKey const& k) const noexcept
        {
            std::size_t h = 1469598103934665603ULL;
            auto const* p = reinterpret_cast<unsigned char const*>(&k);
            for (std::size_t i = 0; i < sizeof(k); ++i) { h ^= p[i]; h *= 1099511628211ULL; }
            return h;
        }
    };
    static thread_local std::unordered_map<CacheKey, Params, CacheHash> s_params_cache;

    // The pool ptr is stable unless it grows; growth only happens outside
    // capture (ensure() aborts otherwise), and the ptr is part of the key.
    Args args{cutlass::gemm::GemmUniversalMode::kGemm, problem,
        {ptr_A, stride_A, ptr_B, stride_B, ptr_SFA, layout_SFA, ptr_SFB, layout_SFB},
        {{1.0f, 0.0f}, nullptr, stride_D, nullptr /* ptr_D patched below */, stride_D}};

    std::size_t const ws_bytes = Gemm::get_workspace_size(args);
    void* base = detail::Sm100WorkspacePool::instance().ensure(partial_bytes + ws_bytes);
    if (base == nullptr)
        return cudaErrorMemoryAllocation;
    float* partials = static_cast<float*>(base);
    void* cutlass_ws = static_cast<char*>(base) + partial_bytes;
    args.epilogue.ptr_D = partials;

    CacheKey key;
    std::memset(&key, 0, sizeof(key));
    key.p[0] = ptr_A; key.p[1] = ptr_B; key.p[2] = ptr_SFA; key.p[3] = ptr_SFB; key.p[4] = partials;
    key.dims[0] = M; key.dims[1] = N; key.dims[2] = K; key.dims[3] = splits;

    Params kernel_params;
    auto it = s_params_cache.find(key);
    if (it != s_params_cache.end())
        kernel_params = it->second;
    else
    {
        // NOTE: Gemm::can_implement is intentionally SKIPPED. The mainloop's
        // check literally recomputes tile_atom_to_shape_SFA(problem) and
        // compares strides, so the split-K stride surgery always "fails" it
        // even though to_underlying_arguments consumes args.layout_SFA/SFB
        // verbatim (verified: make_tensor(args.ptr_SFA, args.layout_SFA) →
        // TMA descriptor). The constraints it would enforce hold by
        // construction here: K % 128 == 0, kslice % 128 == 0 (16B TMA
        // alignment on every batch base), N % 4 == 0 for the FP32 partials.
        if (ws_bytes > 0
            && GemmKernel::initialize_workspace(args, cutlass_ws, stream) != cutlass::Status::kSuccess)
            return cudaErrorUnknown;
        kernel_params = GemmKernel::to_underlying_arguments(args, cutlass_ws); // host-only
        if (s_params_cache.size() < 64u)
            s_params_cache.emplace(key, kernel_params);
    }

    static bool s_smem_configured = false;
    if (!s_smem_configured)
    {
        if (GemmKernel::SharedStorageSize >= (48 << 10))
        {
            cudaError_t result = cudaFuncSetAttribute(cutlass::device_kernel<GemmKernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, GemmKernel::SharedStorageSize);
            if (result != cudaSuccess)
                return result;
        }
        s_smem_configured = true;
    }

    if (Gemm::run(kernel_params, stream) != cutlass::Status::kSuccess)
        return cudaErrorLaunchFailure;

    // Reduce: N is a multiple of 128 on this surface, so M*N % 4 == 0.
    long const mn4 = mn / 4;
    int const threads = 256;
    int const blocks = static_cast<int>((mn4 + threads - 1) / threads) < 1024
        ? static_cast<int>((mn4 + threads - 1) / threads)
        : 1024;
    sm100_mxfp8_splitk_reduce_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float4 const*>(partials), reinterpret_cast<ushort4*>(mat_d), splits, mn4);
    return cudaGetLastError();
}

// Top-level sm_100/sm_103 MXFP8 dispatcher.
//
// Bring-up cascade (C1, placeholder — retune on b300 hardware):
//   * decode / small M          → (64, 128) 1SM
//   * mid M                     → (128, 128) 1SM;  wide N → (128, 256) 1SM
//   * large M (> 256)           → (256, 128) 2SM;  wide N → (256, 256) 2SM
// No Stream-K: the CLC-based persistent scheduler already load-balances
// tail waves; revisit only if the b300 sweep shows narrow-N long-K decode
// shapes losing to a K-split (compare against sm_120's K≥9728 gate).
inline cudaError_t gemm_dispatch_sm100_mxfp8(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int M, int N, int K, cudaStream_t stream)
{
    // Epilogue binding (ncu A/B 2026-07-06, see gemm_types.cuh): NOSMEM
    // (direct-store) on the latency-bound decode tiles and the L2-headroom
    // peak tile; TMA epilogue everywhere L2 is near-bound (mid-band
    // (256,128)K256) or unmeasured (StreamK, forced-only variants, and the
    // (256,256)K128 *mid-band* narrow-N route — same tile as peak but bound
    // TMA until probed, hence the separate NOSMEM/TMA call sites below).
#define DISPATCH_SM100_MX(TM_, TN_, CM_, CN_, TK_, SK_) \
    launch_sm100_mxfp8_gemm<TM_, TN_, CM_, CN_, TK_, SK_>(mat_a, mat_b, mat_d, scales_a, scales_b, M, N, K, stream)
#define DISPATCH_SM100_MX_SW(TM_, TN_, CM_, CN_, TK_, SK_, SW_) \
    launch_sm100_mxfp8_gemm<TM_, TN_, CM_, CN_, TK_, SK_>(mat_a, mat_b, mat_d, scales_a, scales_b, M, N, K, stream, SW_)
#define DISPATCH_SM100_MX_NOSMEM(TM_, TN_, CM_, CN_, TK_, SK_) \
    launch_sm100_mxfp8_gemm<TM_, TN_, CM_, CN_, TK_, SK_, true>(mat_a, mat_b, mat_d, scales_a, scales_b, M, N, K, stream)
#define DISPATCH_SM100_MX_NOSMEM_SW(TM_, TN_, CM_, CN_, TK_, SK_, SW_) \
    launch_sm100_mxfp8_gemm<TM_, TN_, CM_, CN_, TK_, SK_, true>(mat_a, mat_b, mat_d, scales_a, scales_b, M, N, K, stream, SW_)

    // FSO_FORCE_TILE=TM,TN,ST — ST picks the variant (see header comment).
    // Valid tiles: TileM/TileN ∈ {128, 256} only (SF Blk_MN=128 granularity).
    auto forced = tensorrt_llm::kernels::blockscale_gemm::read_force_tile();
    if (forced.active() && forced.tm > 0)
    {
        int const tm = forced.tm, tn = forced.tn, st = forced.st;
        // 1SM rows (TileM=128) — K128 rows carry the cascade's NOSMEM binding
        // so FSO_FORCE_TILE sweeps exercise the production instantiation.
        if (tm == 128 && tn == 128 && st == 1) return DISPATCH_SM100_MX_NOSMEM(128, 128, 1, 1, 128, false);
        if (tm == 128 && tn == 256 && st == 1) return DISPATCH_SM100_MX_NOSMEM(128, 256, 1, 1, 128, false);
        if (tm == 128 && tn == 128 && st == 3) return DISPATCH_SM100_MX(128, 128, 1, 1, 256, false);
        if (tm == 128 && tn == 256 && st == 3) return DISPATCH_SM100_MX(128, 256, 1, 1, 256, false);
        if (tm == 128 && tn == 128 && st == 7) return DISPATCH_SM100_MX(128, 128, 1, 1, 128, true);
        if (tm == 128 && tn == 128 && st == 8) return DISPATCH_SM100_MX(128, 128, 1, 1, 256, true);
        if (tm == 128 && tn == 128 && st == 9 && (K % 128) == 0)
        {
            int const tiles = ((M + 127) / 128) * ((N + 127) / 128);
            int const ks = forced.ks > 1 ? forced.ks : detail::pick_splits(tiles, K / 128);
            if (ks > 1 && (K / 128) % ks == 0)
                return launch_sm100_mxfp8_gemm_splitk<128, 128>(
                    mat_a, mat_b, mat_d, scales_a, scales_b, M, N, K, ks, stream);
            // No valid split → plain decode tile.
            return DISPATCH_SM100_MX_NOSMEM(128, 128, 1, 1, 128, false);
        }
        // 1SM cluster(2,2) probes (ncu F3: nvjet's gate M=2048 config is
        // 128×256/CTA with a 4-CTA cluster → half our L2 traffic).
        // NEGATIVE RESULT (2026-07-06 c22_probe): all three lose across the
        // whole mid-band — (128,256)c22 +15..28% (gate M2048 52.6 vs 41.0),
        // (128,128)c22 +9..20%, (256,128)c22-K256 +1..8%. The L2-traffic
        // saving does not survive the CLC scheduler's coarser 4-CTA
        // allocation granularity + cluster barriers. nvjet's mid-band edge
        // is scheduler/pipeline-level, not reachable by cluster shape alone.
        // Kept as force codes for future sweeps; do NOT wire into the
        // cascade.
        if (tm == 128 && tn == 128 && st == 10) return DISPATCH_SM100_MX(128, 128, 2, 2, 128, false);
        if (tm == 128 && tn == 256 && st == 10) return DISPATCH_SM100_MX(128, 256, 2, 2, 128, false);
        if (tm == 256 && tn == 128 && st == 11) return DISPATCH_SM100_MX(256, 128, 2, 2, 256, false);
        // 2SM rows (TileM=256)
        if (tm == 256 && tn == 128 && st == 2) return DISPATCH_SM100_MX(256, 128, 2, 1, 128, false);
        if (tm == 256 && tn == 256 && st == 2) return DISPATCH_SM100_MX_NOSMEM(256, 256, 2, 1, 128, false);
        if (tm == 256 && tn == 128 && st == 4) return DISPATCH_SM100_MX(256, 128, 2, 1, 256, false);
        if (tm == 256 && tn == 256 && st == 4) return DISPATCH_SM100_MX(256, 256, 2, 1, 256, false);
        if (tm == 256 && tn == 256 && st == 5) return DISPATCH_SM100_MX(256, 256, 2, 2, 128, false);
        if (tm == 256 && tn == 256 && st == 6) return DISPATCH_SM100_MX(256, 256, 2, 2, 256, false);
        // Unknown forced combo → fall through to the cascade (sm_120 contract).
    }

    // Cascade v2 — rules distilled from the 2026-07-05 b300 round-2 tile
    // sweep (runs/b300_tile_sweep_v2.jsonl: 12 variants × Qwen3 grid,
    // best-shot cudagraph µs). Key findings baked in:
    //   * (256,128) 2SM c(2,1) K256 is the mid-band workhorse
    //     (128 < M ≤ 2048: best-or-tie on wqkv/wo/gate/down).
    //   * StreamK in heuristic mode LOSES to plain tiles everywhere
    //     (down M=1: 30.9 vs 17.5 µs) — removed from the default route
    //     until explicit-splits (FSO_FORCE_KSPLIT) proves out.
    //   * c(2,2) and K256 do NOT help the peak band (cubic-8192:
    //     c22 410 µs / K256 435 µs vs c21-K128 370 µs).
    //   * down-class decode floor is (256,128)K256 ≈ 16.7 µs vs cuBLAS
    //     ~10.4 µs — the remaining gap needs split-K or a custom path,
    //     tracked separately.
    int const tiles_n = (N + 127) / 128;

    // Parallel split-K for down-class decode (ncu F1): narrow-N long-K
    // shapes whose tile grid can't fill the machine. pick_splits returns 1
    // (→ fall through) once the grid fills, so the route self-limits to
    // decode. Band from the 2026-07-06 sk_probe (best-shot µs):
    //   * M ≤ 32, K ≥ 4096: always wins — down 17.2→11.7 (BEATS cuBLAS
    //     12.3), wo 10.6→9.9. wqkv/gate K=2560 excluded (forced-SK loses,
    //     11.4 vs 10.1: split ×2 + reduce isn't amortized at short K).
    //   * 32 < M ≤ 128: only long-K — down K=9728 wins (16.4→12.9 M=64,
    //     15.7 M=128) but wo K=4096 loses (10.3→13.9) → K ≥ 8192.
    //   * Deeper splits lose (down M=1: S=19 14.9 vs S=4 12.9) — the ≤ 8
    //     cap plus largest-divisor pick in pick_splits is the sweet spot.
    if (tiles_n <= 32 && (K % 128) == 0
        && ((M <= 32 && K >= 4096) || (M <= 128 && K >= 8192)))
    {
        int const tiles = ((M + 127) / 128) * tiles_n;
        int const ks = detail::pick_splits(tiles, K / 128);
        if (ks > 1)
            return launch_sm100_mxfp8_gemm_splitk<128, 128>(
                mat_a, mat_b, mat_d, scales_a, scales_b, M, N, K, ks, stream);
    }

    if (M <= 8)
    {
        // Extreme decode: 1SM (128,128); very wide N amortizes the per-CTA
        // prologue with a 256-wide tile (gate_up 15.7 vs 19.9, gate 12.5
        // vs 13.5 µs). NOSMEM: latency-bound, L2 SOL ≈ 12% — direct store
        // wins (wqkv M1 10.75→10.50, down M1 17.60→17.15).
        if (N >= 9728)
            return DISPATCH_SM100_MX_NOSMEM(128, 256, 1, 1, 128, false);
        return DISPATCH_SM100_MX_NOSMEM(128, 128, 1, 1, 128, false);
    }
    if (M <= 128)
    {
        // NOSMEM only while the M-tile is mostly empty (M ≤ 32: stores are
        // a sliver, latency wins — gate M1-32 −9..−16%). At fuller M-tiles
        // with this band's short-K shapes the epilogue dominates and the
        // TMA bulk store wins big (v23 bench: gate M=128 8.2→10.3 µs,
        // gate_up M=128 12.3→14.4 µs under NOSMEM → reverted to TMA).
        bool const nosmem = (M <= 32);
        // gate_up-class stays 1SM 256-wide through M=128 (13.3 vs 15.6+).
        if (N >= 16384)
            return nosmem ? DISPATCH_SM100_MX_NOSMEM(128, 256, 1, 1, 128, false)
                          : DISPATCH_SM100_MX(128, 256, 1, 1, 128, false);
        // gate M ≤ 16 also prefers the 256-wide 1SM tile (12.4 vs 12.7).
        if (M <= 16 && N >= 9728)
            return DISPATCH_SM100_MX_NOSMEM(128, 256, 1, 1, 128, false);
        // One 128-row M-tile: 2SM (256,128)K256 wins when the wave still
        // fits (wqkv tn=48: 9.3, down tn=20: 16.7); wide N overflows with
        // 2× CTAs and loses (gate tn=76: 12.1 vs 9.4 1SM) → stay 1SM.
        // M ≤ 32 is still latency-dominated and prefers 1SM (wqkv M=32:
        // 8.3 vs 10.3 µs under the M>16 2SM rule — bench 2026-07-05).
        if (M > 32 && tiles_n <= 60)
            return DISPATCH_SM100_MX(256, 128, 2, 1, 256, false);
        return nosmem ? DISPATCH_SM100_MX_NOSMEM(128, 128, 1, 1, 128, false)
                      : DISPATCH_SM100_MX(128, 128, 1, 1, 128, false);
    }
    if (M <= 2048)
    {
        // Narrow N at M≈1024 halves the wave count with a 256-wide tile
        // (down/wo M=1024) — long-K only (cubic-1024/1536 regressed) and
        // not at M=2048 (wo regressed 18%).
        if (tiles_n <= 20 && M >= 1024 && M < 2048 && K >= 4096)
            return DISPATCH_SM100_MX(256, 256, 2, 1, 128, false);
        // Wide-N many-wave cells also benefit from raster swizzle
        // (gate_up M=2048 K256: sw8 75.3 vs sw0 79.3 µs p50, warm probe
        // 2026-07-05); narrow-N mid cells are flat → gate on tiles_n.
        int const mid_swizzle = (M >= 1024 && tiles_n >= 32) ? 8 : 0;
        return DISPATCH_SM100_MX_SW(256, 128, 2, 1, 256, false, mid_swizzle);
    }
    // Peak band: 256-wide once N supplies enough tiles; K128 (K256 loses
    // ~6% at cubic-4096+, deeper pipeline wins over fewer iterations).
    // Raster swizzle 8 recovers L2 locality on many-wave problems
    // (cubic-8192: 370 → 349 µs +6%, cubic-12288/16384 +11/16%) — but
    // ONLY when N supplies at least a full swizzle group of tiles
    // (down M=4096, tiles_n=20: swizzle=8 cost 24%).
    // NOSMEM on the peak tile only when the mainloop is long enough to
    // amortize the slower scattered-STG store (K ≥ 4096: cubic-8192
    // 352→327 µs = 3360 TF ≈ cuBLAS parity, cubic-4096 −3.8%, L2 SOL 49%
    // has headroom). Short-K epilogue-heavy peak cells keep the TMA bulk
    // store (v23 bench: gate M=4096 K=2560 74.3→85.0 µs under NOSMEM).
    // Narrow-N (256,128) stays TMA (unmeasured).
    int const peak_swizzle = (tiles_n >= 32) ? 8 : 0;
    if (N >= 4096)
    {
        if (K >= 4096)
            return DISPATCH_SM100_MX_NOSMEM_SW(256, 256, 2, 1, 128, false, peak_swizzle);
        return DISPATCH_SM100_MX_SW(256, 256, 2, 1, 128, false, peak_swizzle);
    }
    return DISPATCH_SM100_MX_SW(256, 128, 2, 1, 128, false, peak_swizzle);

#undef DISPATCH_SM100_MX
#undef DISPATCH_SM100_MX_SW
}

} // namespace sm100_blockscaled_gemm

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
