/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SM120 MXFP8 (1×32, kSFVecSize=32) dispatch and kernel launcher.
//
// Mirrors `sm120_dispatch.cuh` for the VS=32 path. Reuses the same
// `SM120BlockScaledKernel<KT>` template (in `fp8/gemm_1d1d.cuh`) — only
// the builder differs. The (M, tiles_n) cascade and Stream-K logic
// match the 1×128 path bit-for-bit (C2 ports the 5 production tile
// templates and the K≥9728 Stream-K guard from `fea4158`). C6 will
// retune cascade thresholds based on a VS=32-specific tile sweep.

#pragma once

#include "blockscale_gemm/common/kernel_utils.cuh"
#include "blockscale_gemm/arch/sm120/common/env_overrides.cuh"
#include "blockscale_gemm/arch/sm120/fp8/gemm_1d1d.cuh"  // SM120BlockScaledKernel<KT>
#include "blockscale_gemm/arch/sm120/mxfp8/utils.cuh"  // SM120MxFP8BlockScaledBuilder
#include "blockscale_gemm/arch/sm120/mxfp8/smallm_kernel.cuh"  // SM120MxFP8SmallMKernel<KT> (E28)
#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

TRTLLM_NAMESPACE_BEGIN
namespace kernels::blockscale_gemm
{

template <int TileM, int TileN, int NumStages, int MinBlocksPerSm = 1, int SchedGroup = 16,
    bool SeparateSmemD = false, bool GroupedLayoutSmem = false>
void launch_sm120_mxfp8_gemm_kernel(__nv_fp8_e4m3* mat_a, int64_t ld_a, int64_t stride_a, __nv_fp8_e4m3* mat_b,
    int64_t ld_b, int64_t stride_b, __nv_bfloat16* mat_d, int64_t ld_d, int64_t stride_d, int32_t* scales_a,
    int64_t /*stride_scales_a*/, int32_t* scales_b, int64_t /*stride_scales_b*/, uint32_t num_problems,
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, cudaStream_t stream, int num_device_sms = kNumDeviceSMs,
    int32_t* grouped_layout = nullptr)
{
    if (num_device_sms < 0)
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

    using ElementInput = cute::float_e4m3_t;
    using ElementOutput = cute::bfloat16_t;
    using ElementBlockScale = int32_t;
    using KT = sm120_blockscaled_gemm::SM120MxFP8BlockScaledBuilder<TileM, TileN, NumStages, MinBlocksPerSm, SchedGroup, SeparateSmemD, GroupedLayoutSmem>;
    using GemmKernel = sm120_blockscaled_gemm::SM120BlockScaledKernel<KT>;
    using Params = typename GemmKernel::Params;
    using Arguments = typename GemmKernel::Arguments;
    using ProblemShape = typename GemmKernel::ProblemShape;
    ProblemShape problem_shape = make_shape((int) shape_m, (int) shape_n, (int) shape_k, (int) num_problems);

    auto ptr_A = reinterpret_cast<ElementInput*>(mat_a);
    auto ptr_B = reinterpret_cast<ElementInput*>(mat_b);
    auto ptr_SFA = reinterpret_cast<ElementBlockScale*>(scales_a);
    auto ptr_SFB = reinterpret_cast<ElementBlockScale*>(scales_b);
    auto ptr_D = reinterpret_cast<ElementOutput*>(mat_d);

    typename KT::StrideA dA = make_stride(ld_a, Int<1>{}, stride_a);
    typename KT::StrideB dB = make_stride(ld_b, Int<1>{}, stride_b);
    typename KT::StrideSFA dSFA = KT::deduce_sfa_layout(problem_shape).stride();
    typename KT::StrideSFB dSFB = KT::deduce_sfb_layout(problem_shape).stride();
    typename KT::StrideD dD = make_stride(ld_d, Int<1>{}, stride_d);

    Arguments args{ptr_A, dA, ptr_B, dB, ptr_SFA, dSFA, ptr_SFB, dSFB, ptr_D, dD, grouped_layout};

    // E9 (2026-05-10): cache Params (which holds 5 CUtensorMap descriptors) to
    // skip the 5x cuTensorMapEncodeTiled calls inside `to_underlying_arguments`.
    // Each call costs ~3 µs host-side → ~15 µs total, which is THE small-M
    // launch bottleneck (M=1 down: BF16 enqueue 8 µs, MXFP8 enqueue 22 µs after
    // E8; the 14 µs gap matches the 5x TMA encode cost).
    //
    // The cache hits when consecutive calls share (ptrs, shape, strides). For
    // PyTorch inference: weight ptr_B / ptr_SFB stay stable, and torch's CUDA
    // caching allocator usually returns the same memory for the recycled
    // activation buffers (ptr_A, ptr_SFA, ptr_D), so hit rate is high in the
    // tight bench loop and the production forward path.
    //
    // The cache is per-instantiation and thread-local; size capped at 64
    // entries (~45 KB) so a workload with many unique shapes doesn't blow up.
    struct CacheKey {
        void* p[5]; int dims[4]; int64_t ld[3]; int64_t st[3];
        bool operator==(CacheKey const& o) const { return std::memcmp(this, &o, sizeof(o)) == 0; }
    };
    struct CacheHash {
        std::size_t operator()(CacheKey const& k) const noexcept {
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
    key.dims[0] = (int) shape_m; key.dims[1] = (int) shape_n; key.dims[2] = (int) shape_k; key.dims[3] = (int) num_problems;
    key.ld[0] = ld_a; key.ld[1] = ld_b; key.ld[2] = ld_d;
    key.st[0] = stride_a; key.st[1] = stride_b; key.st[2] = stride_d;

    Params kernel_params;
    auto cache_it = s_params_cache.find(key);
    if (cache_it != s_params_cache.end())
    {
        kernel_params = cache_it->second;
    }
    else
    {
        kernel_params = GemmKernel::to_underlying_arguments(problem_shape, args);
        if (s_params_cache.size() < 64u)
            s_params_cache.emplace(key, kernel_params);
    }
    // Grouped path: the masked_m buffer address is not part of the cache key
    // (it may be reallocated between calls while shapes stay identical), and
    // Params only carries the raw pointer — patch it after every cache
    // lookup so a cached Params never replays a stale grouped_layout.
    kernel_params.grouped_layout = grouped_layout;
    auto kernel_ptr = &cutlass::device_kernel<GemmKernel>;

    // E8 (2026-05-10): cudaFuncSetAttribute is idempotent per (function, device,
    // attribute) — only the first call per process needs the syscall path.
    // Saves ~0.8 µs/launch in the small-M regime where host overhead dominates.
    // The static guard works because each template instantiation has its own
    // function symbol and its own kSmemSize.
    static bool s_smem_configured = false;
    if (!s_smem_configured)
    {
        cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, GemmKernel::kSmemSize);
        auto result = cudaGetLastError();
        TLLM_CHECK_WITH_INFO(result == cudaSuccess, "sm120 mxfp8 gemm kernel cannot launch: %s", cudaGetErrorString(result));
        s_smem_configured = true;
    }

    cudaLaunchConfig_t launch_config;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = fso_pdl_enabled() ? 1 : 0;
    // MinBlocksPerSm > 1 instances co-schedule that many CTAs per SM (the
    // persistent scheduler strides by gridDim.x, so a larger grid just
    // partitions the same tile queue). ncu on the grouped decode cells
    // showed every pipe under 55% SOL at 1 CTA/SM — latency-bound with the
    // co-residency the 44 KB (16,64,4) instance already fits going unused.
    launch_config.gridDim = dim3(num_device_sms * KT::MinBlocksPerSm, 1, 1);
    launch_config.blockDim = GemmKernel::get_block_shape();
    launch_config.dynamicSmemBytes = GemmKernel::kSmemSize;
    launch_config.stream = stream;
    launch_config.attrs = attrs;
    launch_config.numAttrs = 1;
    cudaLaunchKernelEx(&launch_config, kernel_ptr, kernel_params);
    // The post-launch cudaGetLastError check was removed (was ~0.5 µs/launch).
    // Kernel runtime errors surface on next cudaStreamSynchronize.
}

// E28 (2026-05-11): Forced-only experimental SmallM launcher. Uses
// `SM120MxFP8SmallMKernel<KT>` which replaces A's TMA load with per-row
// cp.async.bulk. TileM is fixed at 16 (sm_120 MXFP8 mma atom floor).
// Enabled only via `BSGEMM_FORCE_SMALLM=1` combined with
// `BSGEMM_FORCE_TILE=16,TN,ST`. Not on the default route.
template <int TileN, int NumStages>
void launch_sm120_mxfp8_smallm_gemm_kernel(__nv_fp8_e4m3* mat_a, int64_t ld_a, int64_t stride_a,
    __nv_fp8_e4m3* mat_b, int64_t ld_b, int64_t stride_b, __nv_bfloat16* mat_d, int64_t ld_d, int64_t stride_d,
    int32_t* scales_a, int64_t /*stride_scales_a*/, int32_t* scales_b, int64_t /*stride_scales_b*/,
    uint32_t num_problems, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, cudaStream_t stream,
    int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

    using ElementInput = cute::float_e4m3_t;
    using ElementOutput = cute::bfloat16_t;
    using ElementBlockScale = int32_t;
    using KT = sm120_blockscaled_gemm::SM120MxFP8BlockScaledBuilder<16, TileN, NumStages>;
    using GemmKernel = sm120_blockscaled_gemm::SM120MxFP8SmallMKernel<KT>;
    using Params = typename GemmKernel::Params;
    using Arguments = typename GemmKernel::Arguments;
    using ProblemShape = typename GemmKernel::ProblemShape;
    ProblemShape problem_shape = make_shape((int) shape_m, (int) shape_n, (int) shape_k, (int) num_problems);

    auto ptr_A = reinterpret_cast<ElementInput*>(mat_a);
    auto ptr_B = reinterpret_cast<ElementInput*>(mat_b);
    auto ptr_SFA = reinterpret_cast<ElementBlockScale*>(scales_a);
    auto ptr_SFB = reinterpret_cast<ElementBlockScale*>(scales_b);
    auto ptr_D = reinterpret_cast<ElementOutput*>(mat_d);

    typename KT::StrideA dA = make_stride(ld_a, Int<1>{}, stride_a);
    typename KT::StrideB dB = make_stride(ld_b, Int<1>{}, stride_b);
    typename KT::StrideSFA dSFA = KT::deduce_sfa_layout(problem_shape).stride();
    typename KT::StrideSFB dSFB = KT::deduce_sfb_layout(problem_shape).stride();
    typename KT::StrideD dD = make_stride(ld_d, Int<1>{}, stride_d);

    Arguments args{ptr_A, dA, ptr_B, dB, ptr_SFA, dSFA, ptr_SFB, dSFB, ptr_D, dD};

    // Cache Params (3× TMA descriptors instead of 5× for the regular kernel —
    // SmallM skips TMA_A and TMA_SFA-no-actually-keeps-SFA. We drop TMA_A only.)
    struct CacheKey {
        void* p[5]; int dims[4]; int64_t ld[3]; int64_t st[3];
        bool operator==(CacheKey const& o) const { return std::memcmp(this, &o, sizeof(o)) == 0; }
    };
    struct CacheHash {
        std::size_t operator()(CacheKey const& k) const noexcept {
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
    key.dims[0] = (int) shape_m; key.dims[1] = (int) shape_n; key.dims[2] = (int) shape_k; key.dims[3] = (int) num_problems;
    key.ld[0] = ld_a; key.ld[1] = ld_b; key.ld[2] = ld_d;
    key.st[0] = stride_a; key.st[1] = stride_b; key.st[2] = stride_d;

    Params kernel_params;
    auto cache_it = s_params_cache.find(key);
    if (cache_it != s_params_cache.end())
    {
        kernel_params = cache_it->second;
    }
    else
    {
        kernel_params = GemmKernel::to_underlying_arguments(problem_shape, args);
        if (s_params_cache.size() < 64u)
            s_params_cache.emplace(key, kernel_params);
    }
    auto kernel_ptr = &cutlass::device_kernel<GemmKernel>;

    static bool s_smem_configured = false;
    if (!s_smem_configured)
    {
        cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, GemmKernel::kSmemSize);
        auto result = cudaGetLastError();
        TLLM_CHECK_WITH_INFO(result == cudaSuccess,
            "sm120 mxfp8 smallm gemm kernel cannot launch: %s", cudaGetErrorString(result));
        s_smem_configured = true;
    }

    cudaLaunchConfig_t launch_config;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    launch_config.gridDim = dim3(num_device_sms, 1, 1);
    launch_config.blockDim = GemmKernel::get_block_shape();
    launch_config.dynamicSmemBytes = GemmKernel::kSmemSize;
    launch_config.stream = stream;
    launch_config.attrs = attrs;
    launch_config.numAttrs = 1;
    cudaLaunchKernelEx(&launch_config, kernel_ptr, kernel_params);
}

} // namespace kernels::blockscale_gemm
TRTLLM_NAMESPACE_END

// Stream-K wrapper depends on launch_sm120_mxfp8_gemm_kernel above.
#include "blockscale_gemm/arch/sm120/mxfp8/streamk.cuh"

TRTLLM_NAMESPACE_BEGIN
namespace kernels::blockscale_gemm
{

// Top-level dispatcher. Mirror of `gemm_dispatch_sm120` but for the
// VS=32 (MXFP8 1×32) path. Uses the same (M, tiles_n) cascade and the
// same `sm120_streamk_choose_k_split` heuristic — including the K≥9728
// Stream-K gate from commit fea4158. Env-var overrides
// (BSGEMM_FORCE_TILE / BSGEMM_FORCE_KSPLIT / BSGEMM_DISABLE_STREAMK /
// BSGEMM_DISABLE_OVERRIDES) follow the exact same wire format so a
// shared sweep harness can drive both paths.
//
// Tile differences vs the 1×128 path: (64, 128, 4) is replaced by
// (64, 128, 2) everywhere because at VS=32 + kTileSF=4 the per-CTA
// SMEM (A=32KB + B=64KB + SF=3KB) saturates the 99KB Blackwell consumer
// limit and `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` fails
// with `invalid argument`. Stages=2 brings AB down to 32+32=64KB, total
// well within budget. C6 will retune once the cuobjdump baseline is
// captured. The other 4 tiles ((32,128,4), (96,128,2), (128,128,2),
// (160,128,2)) fit unchanged — verified at TileM=160 N=128 ST=2:
// 40+32+2.5+2 = 76 KB.
inline void gemm_dispatch_sm120_mxfp8(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
    cudaStream_t stream, int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

    int64_t ld_a = shape_k;
    int64_t ld_b = shape_k;
    int64_t ld_d = shape_n;
    constexpr int64_t stride = 0;
    constexpr uint32_t num_problems = 1;

    // E15 (2026-05-10): considered routing M ≤ 4 to a hand-rolled scalar
    // GEMV (`common/gemv.cuh`) to skip TMA descriptor setup. NEGATIVE
    // RESULT: even with 31/32 M-row waste, the tensor-core matmul kernel
    // dominates a scalar GEMV — 170-SM Blackwell has ~970 TF (TC) vs ~21 TF
    // (scalar ALU). The GEMV kernel kept around for reference but NOT
    // dispatched; turn on via `BSGEMM_USE_GEMV=1` env if needed for
    // A/B testing. See `common/gemv.cuh` for the design notes.

    // BSGEMM_FORCE_TILE / BSGEMM_FORCE_KSPLIT — see env_overrides.cuh.
    // Tile combinations advertised here are MXFP8-specific (NS=2 at
    // TileM=64; the FP8 path uses NS=4 there but kSFVecSize=32 + Stages=4
    // would exceed the 99 KB SMEM budget).
    auto forced = read_force_tile();
    if (force_tile_applies(forced, shape_k))
    {
        int const k_split = forced.stream_k() ? forced.ks : 1;
        bool handled = false;
        // E28 (2026-05-11): forced SmallM variant — manual cp.async.bulk
        // A loads. **NEGATIVE RESULT — keep behind force flag.** See
        // `smallm_kernel.cuh` and `docs/skills/blockscale-gemm-tuning/
        // references/sm120-mxfp8-smallm-manual-a-negative.md`. Manual A
        // ran 2-13× slower than TMA because the SW128 swizzle forces
        // 8 × M_actual 16-byte cp.async.bulks per stage. Default
        // dispatch is unchanged. Forced only via BSGEMM_FORCE_SMALLM=1
        // + BSGEMM_FORCE_TILE=16,128,4 for A/B reproduction.
        if (forced.smallm() && k_split <= 1 && forced.tm == 16)
        {
            if (forced.tn == 128 && forced.st == 4)
            {
                launch_sm120_mxfp8_smallm_gemm_kernel<128, 4>(mat_a, ld_a, stride, mat_b, ld_b, stride, mat_d,
                    ld_d, stride, scales_a, stride, scales_b, stride, num_problems, shape_m, shape_n, shape_k,
                    stream, num_device_sms);
                return;
            }
        }
        // E29 T4 (2026-05-12): forced scheduler-group override for TileM=16.
        // BSGEMM_FORCE_SCHED_GROUP={8,32} selects an alt-SchedGroup launcher
        // for the TileM=16 small-M paths. Only 16x128x4 and 16x64x4 are
        // wired — these are the only TileM=16 tile shapes that the default
        // routes E24/E27 use. Forced-only experiment; default unchanged.
        if (forced.sched_group_forced() && k_split <= 1 && forced.tm == 16
            && (forced.sg == 8 || forced.sg == 32))
        {
            if (forced.tn == 128 && forced.st == 4)
            {
                if (forced.sg == 8)
                    launch_sm120_mxfp8_gemm_kernel<16, 128, 4, 1, 8>(mat_a, ld_a, stride, mat_b, ld_b,
                        stride, mat_d, ld_d, stride, scales_a, stride, scales_b, stride, num_problems,
                        shape_m, shape_n, shape_k, stream, num_device_sms);
                else
                    launch_sm120_mxfp8_gemm_kernel<16, 128, 4, 1, 32>(mat_a, ld_a, stride, mat_b, ld_b,
                        stride, mat_d, ld_d, stride, scales_a, stride, scales_b, stride, num_problems,
                        shape_m, shape_n, shape_k, stream, num_device_sms);
                return;
            }
            if (forced.tn == 64 && forced.st == 4)
            {
                if (forced.sg == 8)
                    launch_sm120_mxfp8_gemm_kernel<16, 64, 4, 1, 8>(mat_a, ld_a, stride, mat_b, ld_b,
                        stride, mat_d, ld_d, stride, scales_a, stride, scales_b, stride, num_problems,
                        shape_m, shape_n, shape_k, stream, num_device_sms);
                else
                    launch_sm120_mxfp8_gemm_kernel<16, 64, 4, 1, 32>(mat_a, ld_a, stride, mat_b, ld_b,
                        stride, mat_d, ld_d, stride, scales_a, stride, scales_b, stride, num_problems,
                        shape_m, shape_n, shape_k, stream, num_device_sms);
                return;
            }
        }
        if (k_split <= 1)
        {
            #define DISPATCH_FORCED_MX(TM_, TN_, ST_) \
                launch_sm120_mxfp8_gemm_kernel<TM_, TN_, ST_>(mat_a, ld_a, stride, mat_b, ld_b, stride, mat_d, \
                    ld_d, stride, scales_a, stride, scales_b, stride, num_problems, shape_m, shape_n, shape_k, \
                    stream, num_device_sms)
            // 16x128x4 with optional MinBlocksPerSm=2 hint (T4 experiment).
            // BSGEMM_FORCE_MIN_BLOCKS=2 selects the (16, 128, 4, 2) builder
            // instead of the default (16, 128, 4, 1). SMEM = 74 KB per CTA
            // exceeds half the per-SM SMEM budget, so 2 CTAs/SM is
            // unlikely to materialise — the hint mostly throttles
            // register usage. Treat as forced-only diagnostic.
            #define DISPATCH_FORCED_MX_MB(TM_, TN_, ST_, MB_) \
                launch_sm120_mxfp8_gemm_kernel<TM_, TN_, ST_, MB_>(mat_a, ld_a, stride, mat_b, ld_b, stride, mat_d, \
                    ld_d, stride, scales_a, stride, scales_b, stride, num_problems, shape_m, shape_n, shape_k, \
                    stream, num_device_sms)
            // 16x128x2 evaluated (2026-05-11): negative result — stage-2
            // loses 1-19% vs stage-4 on every targeted cell (e.g., wqkv M=4
            // st2=12.34 vs st4=10.39). Deeper K-pipeline at stage-4 keeps
            // tensor cores fed; stage-2 stalls. Removed from forced list to
            // avoid drift; if reintroduced, A/B against `runs/tilem16_ab.jsonl`.
            // 16x64x4 doesn't compile: SM75_U32x4_LDSM_N for the B-fragment
            // needs more vals per thread than 8 N-warps × TileN=64 supply
            // (TiledNumVal % AtomNumVal != 0). Keep TileM=16 to TileN=128.
            if      (forced.tm == 16 && forced.tn == 128 && forced.st == 4 && forced.min_blocks_2())
            { DISPATCH_FORCED_MX_MB(16, 128, 4, 2); handled = true; }
            else if (forced.tm ==  16 && forced.tn == 128 && forced.st == 4) { DISPATCH_FORCED_MX( 16, 128, 4); handled = true; }
            else if (forced.tm ==  16 && forced.tn ==  64 && forced.st == 4) { DISPATCH_FORCED_MX( 16,  64, 4); handled = true; }
            else if (forced.tm ==  32 && forced.tn == 128 && forced.st == 4) { DISPATCH_FORCED_MX( 32, 128, 4); handled = true; }
            else if (forced.tm ==  32 && forced.tn ==  64 && forced.st == 4) { DISPATCH_FORCED_MX( 32,  64, 4); handled = true; }
            else if (forced.tm ==  64 && forced.tn == 128 && forced.st == 2) { DISPATCH_FORCED_MX( 64, 128, 2); handled = true; }
            else if (forced.tm ==  64 && forced.tn ==  64 && forced.st == 4) { DISPATCH_FORCED_MX( 64,  64, 4); handled = true; }
            else if (forced.tm ==  96 && forced.tn == 128 && forced.st == 2) { DISPATCH_FORCED_MX( 96, 128, 2); handled = true; }
            else if (forced.tm == 128 && forced.tn == 128 && forced.st == 2) { DISPATCH_FORCED_MX(128, 128, 2); handled = true; }
            else if (forced.tm == 128 && forced.tn ==  64 && forced.st == 2) { DISPATCH_FORCED_MX(128,  64, 2); handled = true; }
            else if (forced.tm == 160 && forced.tn == 128 && forced.st == 2) { DISPATCH_FORCED_MX(160, 128, 2); handled = true; }
            #undef DISPATCH_FORCED_MX
            #undef DISPATCH_FORCED_MX_MB
        }
        else
        {
            #define DISPATCH_FORCED_SK_MX(TM_, TN_, ST_) \
                launch_sm120_mxfp8_streamk_gemm<TM_, TN_, ST_>(mat_a, static_cast<int>(ld_a), mat_b, \
                    static_cast<int>(ld_b), mat_d, static_cast<int>(ld_d), static_cast<int>(shape_m), \
                    static_cast<int>(shape_n), static_cast<int>(shape_k), scales_a, scales_b, k_split, \
                    num_device_sms, stream)
            // 16x128x4 Stream-K is a T2 experiment (forced-only). The
            // tilem16_band production gate runs single-launch only; if a
            // future tune shows Stream-K wins on narrow-N small-M K≥4096,
            // promote this to the cascade via the same gate.
            if      (forced.tm ==  16 && forced.tn == 128 && forced.st == 4) { DISPATCH_FORCED_SK_MX( 16, 128, 4); handled = true; }
            else if (forced.tm ==  32 && forced.tn == 128 && forced.st == 4) { DISPATCH_FORCED_SK_MX( 32, 128, 4); handled = true; }
            else if (forced.tm ==  32 && forced.tn ==  64 && forced.st == 4) { DISPATCH_FORCED_SK_MX( 32,  64, 4); handled = true; }
            else if (forced.tm ==  64 && forced.tn == 128 && forced.st == 2) { DISPATCH_FORCED_SK_MX( 64, 128, 2); handled = true; }
            else if (forced.tm ==  64 && forced.tn ==  64 && forced.st == 4) { DISPATCH_FORCED_SK_MX( 64,  64, 4); handled = true; }
            else if (forced.tm == 128 && forced.tn == 128 && forced.st == 2) { DISPATCH_FORCED_SK_MX(128, 128, 2); handled = true; }
            else if (forced.tm == 128 && forced.tn ==  64 && forced.st == 2) { DISPATCH_FORCED_SK_MX(128,  64, 2); handled = true; }
            #undef DISPATCH_FORCED_SK_MX
        }
        if (handled) return;
    }

    int const tiles_n = static_cast<int>((shape_n + 127) / 128);
    int const tiles_n64 = static_cast<int>((shape_n + 63) / 64);
    int const k_split = sm120_streamk_choose_k_split(static_cast<int>(shape_m), static_cast<int>(shape_n),
        static_cast<int>(shape_k), num_device_sms);

#define DISPATCH_TILE_MX(TM_, TN_, ST_)                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if (k_split <= 1)                                                                                              \
        {                                                                                                              \
            launch_sm120_mxfp8_gemm_kernel<TM_, TN_, ST_>(mat_a, ld_a, stride, mat_b, ld_b, stride, mat_d, ld_d,        \
                stride, scales_a, stride, scales_b, stride, num_problems, shape_m, shape_n, shape_k, stream,           \
                num_device_sms);                                                                                       \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            launch_sm120_mxfp8_streamk_gemm<TM_, TN_, ST_>(mat_a, static_cast<int>(ld_a), mat_b,                        \
                static_cast<int>(ld_b), mat_d, static_cast<int>(ld_d), static_cast<int>(shape_m),                      \
                static_cast<int>(shape_n), static_cast<int>(shape_k), scales_a, scales_b, k_split, num_device_sms,     \
                stream);                                                                                               \
        }                                                                                                              \
    } while (0)

    if (!read_disable_overrides())
    {
        // Override 1: ported verbatim from the 1×128 cascade. Stream-K
        // never amortizes for K ≤ 2048, force single launch with the
        // smallest tile.
        // E28 (2026-05-28): cap the tiles_n<32 branch at M≤128 — without
        // that cap the override fires on cubic-1024/1536/2048 (tn=8/12/16,
        // K≤2048) and falls through to the (32,128,4) bottom-of-route,
        // which is calibrated for small-M / short-K. At M ≥ 1024 the
        // route gives 165/236/297 TF vs main-cascade (128,128,2)
        // 193/477/494 TF (sweep: tile_sweep_losers 2026-05-28). Mid-/
        // small-M paths (M≤32 universally, plus small_m_narrow_n M≤128 /
        // mid_m_narrow_n_64 M≤256 below) have their own M gates and are
        // unaffected.
        bool const short_k = shape_k <= 2048
                          && (shape_m <= 32 || (tiles_n < 32 && shape_m <= 128));

        // Override 2: MX-specific (added 2026-05-08, C8 small-M tune;
        // E20 refinement, 2026-05-11).
        // The shared heuristic returns k_split=2 at (tiles_n in [32, 96),
        // M <= 32) because the 1×128 path's SF reduce is light enough
        // to amortize. The 1×32 path has 4× more SF data, so the same
        // k_split=2 *can* pay a heavier reduce cost than the parallelism
        // win — but only sometimes. Re-validated against cudagraph µs:
        //
        //   gate (N=9728 tn=76):
        //     M=1..8  override 14.43 vs cascade 16.45 (+14% — keep)
        //     M=16    noise (both ~12.4)
        //     M=32    override 13.27 vs cascade 12.53 (-5.6% — drop)
        //   wqkv (N=6144 tn=48):
        //     M=1     override 12.37 vs cascade 13.42 (+8% — keep)
        //     M=4     override 14.41 vs cascade 12.37 (-14% — drop)
        //     M=8..32 marginal (both ~12.4) — but drop is cheaper
        //
        // So the original "M<=32, tn∈[32,96)" gate over-fires. The
        // refined condition fires only where override actually wins:
        //   (tn >= 64) AND (M <= 16)   — gate-style mid-wide N
        //   OR
        //   (M <= 2)                    — universal small-M (cascade
        //                                  with Stream-K reduce loses
        //                                  to a single TMA launch when
        //                                  the GEMM is M-light)
        // Source: `runs/rerun-tuning-2026-05-11/no_overrides.jsonl`.
        bool const small_m_mid_n =
            ((shape_m <= 16 && tiles_n >= 64 && tiles_n < 96)
             || (shape_m <= 2 && tiles_n >= 32 && tiles_n < 96))
            && shape_k <= 4096;

        // Override 3 (E7+E14, 2026-05-10): narrow-N small/mid-M. Two bands:
        //
        // small_m_narrow_n (M ∈ [1, 128], tn < 32, K < 9728): single-launch
        // (32, 64, 4). (32, 64, 4) gives 4 M-tiles × 2N-tiles-per-N=128 = 2×
        // more CTAs than (32, 128, 4) → 94% SM coverage at M=128, tn=20 vs
        // 47% for TileN=128. Kernel-only TF (cu130, 170-SM Blackwell):
        //   M=  4 N=2560 K=4096: (32,64,4)= 15 vs (32,128,4)=  5  (+200%)
        //   M= 32 N=2560 K=4096: (32,64,4)= 68 vs (32,128,4)= 48  (+42%)
        //   M=128 N=2560 K=4096: (32,64,4)=226 vs (32,128,4)=155  (+45%)
        //
        // mid_m_narrow_n_64 (M ∈ (128, 256], tn < 32, K < 9728): single-launch
        // (64, 64, 4). At M = 160..256 the workload per CTA grows enough
        // that TileM=32 is wasteful (too many CTAs of tiny work); TileM=64
        // + TileN=64 sweet spot. (Validated K=4096 only.)
        //   M=160 N=2560 K=4096: (64,64,4)=234 vs cascade (64,128,2)=~195 (+20%)
        //   M=192 N=2560 K=4096: (64,64,4)=284 vs cascade (64,128,2)=~231 (+23%)
        //   M=256 N=2560 K=4096: (64,64,4)=354 vs cascade (64,128,2)=~250 (+42%)
        //
        // E18 (2026-05-11): gate BOTH bands on K<9728. The earlier (E16)
        // gating bumped the small-M floor to 4 at K≥9728 but kept M=4..128
        // routed to single (32, 64, 4). End-to-end cudagraph µs (down
        // shape, N=2560 K=9728, baseline vs `BSGEMM_DISABLE_OVERRIDES=1`):
        //   M=  4: override single (32,64,4) 24.69 µs vs cascade+sk4 14.41 (-42%)
        //   M=  8: override 24.82 vs cascade 14.42 (-42%)
        //   M= 16: override 22.62 vs cascade 14.41 (-36%)
        //   M= 32: override 22.63 vs cascade 14.41 (-36%)
        //   M= 64: override 24.70 vs cascade 17.16 (-30%)
        //   M=128: override 27.04 vs cascade 20.53 (-24%)
        // The E16 kernel-only TF comparison (+54% at M=128 K=9728 vs
        // (64,128,2)+sk4) compared the WRONG cascade pick. The actual
        // cascade picks (32,128,4) + Stream-K k=4, not (64,128,2),
        // and that path wins decisively under cudagraph end-to-end.
        // Source: `runs/rerun-tuning-2026-05-11/no_overrides.jsonl`.
        bool const small_m_narrow_n = shape_k < 9728
                                   && shape_m >= 1 && shape_m <= 128
                                   && tiles_n >= 3 && tiles_n < 32;
        bool const mid_m_narrow_n_64 = shape_k < 9728
                                    && shape_m > 128 && shape_m <= 256
                                    && tiles_n >= 3 && tiles_n < 32;

        // E24 (2026-05-11): extreme small-M, wide-N TileM=16 route.
        // The hw MMA atom is 16x8x32 in M/N/K, so atom_M=16 is the floor.
        // A new SM120MxFP8 builder variant uses (1 warp M × 8 warps N) +
        // PermMmaTileM=Int<16> instead of the standard (2 warps M × 4 warps N)
        // + PermMmaTileM=Int<32>, halving the per-CTA M-tile waste at M ≤ 16
        // and doubling the in-flight CTAs across N.
        //
        // Cudagraph µs A/B (170-SM Blackwell, K=2560, all M ∈ {1, 2, 4, 8, 16}):
        //   wqkv    tn=48:  default=12.38 → 16x128x4=10.34   -16.5%
        //   gate    tn=76:  default=14.41 → 16x128x4=12.36   -14.3%
        //   gate_up tn=152: default=18.18 → 16x128x4=12.36   -32.0%
        // Beyond M=16 the gain reverses (TileM=32 fully utilises its
        // 32-row tile; e.g. gate_up M=32 16x128x4 = 22.63 vs default 14.42,
        // +57% LOSS) — that's the hard upper bound.
        // Narrow-N (tn < 32) regresses universally because TileN=64
        // streamk options exist; gate stays at tiles_n >= 48 so we don't
        // touch them.
        // Source: `runs/tilem16_ab.jsonl`.
        bool const tilem16_band = shape_k <= 4096
                               && shape_m <= 16
                               && tiles_n >= 48;

        if (tilem16_band)
        {
            launch_sm120_mxfp8_gemm_kernel<16, 128, 4>(mat_a, ld_a, stride, mat_b, ld_b, stride, mat_d, ld_d,
                stride, scales_a, stride, scales_b, stride, num_problems, shape_m, shape_n, shape_k, stream,
                num_device_sms);
            return;
        }

        // E26 (2026-05-11): narrow-N + long-K + small-M → TileM=16 + Stream-K k=4.
        // Companion to tilem16_band for the `down`-style shape (Qwen3-4B
        // N=2560, K=9728, tn=20). Single-launch 16x128x4 catastrophically
        // loses on narrow-N because only 20 N-tiles × 1 M-tile = 20 CTAs
        // run on 170 SMs (12% wave utilisation). Stream-K k=4 quadruples
        // the in-flight CTAs to ~80 (~47% utilisation) while keeping
        // TileM=16's halved M-row waste.
        //
        // Cudagraph µs A/B (170-SM Blackwell, GPU 1 / GPU 6, `down` N=2560 K=9728):
        //   M= 1: default 14.45 → 16x128+ks4 13.15 / 12.44   -9% / -14%
        //   M= 2: default 15.42 → 14.24 / 13.54              -8% / -12%
        //   M= 4: default 15.65 → 14.26 / 14.33              -9% / -9%
        //   M= 8: default 14.44 → 13.26 / 12.38              -8% / -14%
        //   M=16: default 14.20 → 13.31 / 12.40              -6% / -10%
        // 16x128 single-launch (ks=1): +108..+127% (12% wave); ks=2:
        // +30..+50%. ks=4 is the sweet spot. Wins reproduce on both GPUs.
        // Source: `runs/tilem16_down_streamk.jsonl`.
        //
        // Gate on shape_k >= 8192 to keep the route inside the validated
        // (K=9728 only so far) regime; K < 8192 narrow-N small-M is owned
        // by E27 immediately below (TileM=16, TileN=64 single launch).
        bool const tilem16_narrow_streamk = shape_k >= 8192
                                         && shape_m <= 16
                                         && tiles_n >= 3 && tiles_n < 32;
        if (tilem16_narrow_streamk)
        {
            launch_sm120_mxfp8_streamk_gemm<16, 128, 4>(mat_a, static_cast<int>(ld_a), mat_b,
                static_cast<int>(ld_b), mat_d, static_cast<int>(ld_d), static_cast<int>(shape_m),
                static_cast<int>(shape_n), static_cast<int>(shape_k), scales_a, scales_b,
                /*k_split=*/4, num_device_sms, stream);
            return;
        }

        // E27 (2026-05-11): narrow-N + short-K small-M → TileM=16, TileN=64.
        // Companion to E26 for shapes that have narrow N (tn∈[3,32)) but K
        // short enough that Stream-K doesn't pay off (K < 8192). The
        // (16, 64, 4) builder uses LDSM_x2 for the B-fragment so the
        // 8-N-warp × TileN=64 partition satisfies CuTe's vals-per-thread
        // divisibility check (LDSM_x4 fails, see SmemCopyAtomBForTileMN).
        //
        // Cudagraph µs A/B (170-SM Blackwell GPU 1, `wo` N=2560 K=4096 tn=20):
        //   M= 1: default (32,64,4)=12.41 → (16,64,4)=10.34   -16.7%
        //   M= 2: default            12.39 → 10.34            -16.6%
        //   M= 4: default            12.40 → 10.35            -16.5%
        //   M= 8: default            12.41 → 10.36            -16.6%
        //   M=16: default            12.39 → 10.32            -16.7%
        // Guard shapes confirm scope: `down` (tn=20 K=9728) loses
        // +40..+62% (E26 owns that band — excluded via shape_k<8192);
        // `gate` (tn=76 K=2560) is wide-N — excluded via tiles_n<32.
        // Source: `runs/tilem16_wo_tn64.jsonl`.
        bool const tilem16_narrow_single = shape_k < 8192
                                        && shape_m <= 16
                                        && tiles_n >= 3 && tiles_n < 32;
        if (tilem16_narrow_single)
        {
            launch_sm120_mxfp8_gemm_kernel<16, 64, 4>(mat_a, ld_a, stride, mat_b, ld_b, stride, mat_d, ld_d,
                stride, scales_a, stride, scales_b, stride, num_problems, shape_m, shape_n, shape_k, stream,
                num_device_sms);
            return;
        }

        if (short_k || small_m_mid_n || small_m_narrow_n || mid_m_narrow_n_64)
        {
            // C9-followup: TileN=64 builder bug fixed (utils.cuh:60). Narrow-N
            // tile choice mirrors the cascade table:
            //   tn=1, M ≥ 4096        → (64, 64, 4)     (sweep 253 TF vs (32,64,4) 172)
            //   tn=1, M < 4096        → (32, 64, 4)     (+22~46% over TN=128)
            //   tn=2, M ≤ 1024        → (32, 64, 4)     (+54% over (32,128,4))
            //   tn=2, M ≥ 2048        → (32, 128, 4)    (TileN=64 regresses -21%)
            //   tn ≥ 3, M ∈ (128,256] → (64, 64, 4)     (E14: +20~42% over (64,128,2))
            //   tn ≥ 3, M ≤ 128       → (32, 64, 4)     (E14: +42~54% over TN=128)
            //   tn ≥ 3 (short_k only) → (32, 128, 4)    (small_m_mid_n path)
            if (tiles_n == 1 && shape_m >= 4096)
                launch_sm120_mxfp8_gemm_kernel<64, 64, 4>(mat_a, ld_a, stride, mat_b, ld_b, stride, mat_d, ld_d,
                    stride, scales_a, stride, scales_b, stride, num_problems, shape_m, shape_n, shape_k, stream,
                    num_device_sms);
            else if (tiles_n == 1 || (tiles_n == 2 && shape_m <= 1024))
                launch_sm120_mxfp8_gemm_kernel<32, 64, 4>(mat_a, ld_a, stride, mat_b, ld_b, stride, mat_d, ld_d,
                    stride, scales_a, stride, scales_b, stride, num_problems, shape_m, shape_n, shape_k, stream,
                    num_device_sms);
            else if (mid_m_narrow_n_64)
                // E14: M ∈ (128, 256] narrow-N → TileM=64, TileN=64.
                launch_sm120_mxfp8_gemm_kernel<64, 64, 4>(mat_a, ld_a, stride, mat_b, ld_b, stride, mat_d, ld_d,
                    stride, scales_a, stride, scales_b, stride, num_problems, shape_m, shape_n, shape_k, stream,
                    num_device_sms);
            else if (small_m_narrow_n)
                // E7+E14: M ≤ 128 narrow-N (any K) → TileM=32, TileN=64.
                launch_sm120_mxfp8_gemm_kernel<32, 64, 4>(mat_a, ld_a, stride, mat_b, ld_b, stride, mat_d, ld_d,
                    stride, scales_a, stride, scales_b, stride, num_problems, shape_m, shape_n, shape_k, stream,
                    num_device_sms);
            else
                launch_sm120_mxfp8_gemm_kernel<32, 128, 4>(mat_a, ld_a, stride, mat_b, ld_b, stride, mat_d, ld_d,
                    stride, scales_a, stride, scales_b, stride, num_problems, shape_m, shape_n, shape_k, stream,
                    num_device_sms);
            return;
        }
    }

    // NOTE (2026-05-10, C9 + C9-followup): TileN=64 paths were broken before
    // C9 (cudaErrorMisalignedAddress — wrong PermMmaTileN size-128 spilled
    // R2S writes past SmemLayoutD). Fixed in utils.cuh by shrinking
    // PermMmaTileN<64> to a size-64 layout. Sweep (C9b,
    // `tests/baselines/mxfp8_tile_sweep_2026-05-10_C9b.txt`) shows TileN=64
    // wins decisively for tiles_n ≤ 2 (N ≤ 256) — every M-bucket gets
    // +22%~+54%. TileN=64 also wins by single digits at tiles_n == 4 (N=512)
    // for M ≤ 2048; at M ≥ 4096 N=512 TileN=128 reclaims the lead.
    // tiles_n ≥ 8 is TileN=128 territory across all M.

    // tiles_n <= 2 (N ≤ 256): TileN=64 wins on tn=1 universally and on tn=2
    // for M ≤ 1024; at tn=2 M ≥ 2048 TileN=128 reclaims the lead (interface
    // rebench `narrow_n_bench`):
    //   M=2048 N=256 K=4096:  TileN=64 22.8 vs TileN=128 18.1 µs  → TN128 -21%
    //   M=2048 N=256 K=2048:  TileN=64 12.6 vs TileN=128 10.0 µs  → TN128 -21%
    if (tiles_n == 1)
    {
        // N ≤ 128 — TileN=64 always wins; switch to (64,64,4) at large M.
        if (shape_m >= 4096)
            DISPATCH_TILE_MX(64, 64, 4);
        else
            DISPATCH_TILE_MX(32, 64, 4);
    }
    else if (tiles_n == 2 && shape_m <= 1024)
    {
        // N=256 small M: (32,64,4) wins +27..54% over (32,128,4).
        DISPATCH_TILE_MX(32, 64, 4);
    }
    else if (tiles_n == 2)
    {
        // N=256 M ≥ 2048: TileN=128 wins; pick small TileM since N is narrow.
        DISPATCH_TILE_MX(32, 128, 4);
    }
    else if (shape_m <= 32)
    {
        // E22 (2026-05-11): at narrow-N (tn ∈ [3, 32)) K ≥ 9728 M ∈ [16, 32],
        // (32, 64, 4) + Stream-K k=4 beats (32, 128, 4) + k=4 by 6-13%.
        // This path is only reachable for K ≥ 9728 since K < 9728 narrow-N
        // is caught by small_m_narrow_n first. End-to-end cudagraph µs
        // (down N=2560 K=9728):
        //   M=16: (32,128,4) k=4 = 14.34 → (32,64,4) k=4 = 13.48  -6%
        //   M=32: (32,128,4) k=4 = 14.34 → (32,64,4) k=4 = 12.46  -13%
        // M ≤ 8 the trend reverses ((32,128,4) k=4 wins by 1-3 µs) —
        // each (32,64,4) k=4 side stream gets less work per slice and
        // the launch+reduce overhead dominates. Crossover at M ≈ 16.
        if (tiles_n >= 3 && tiles_n < 32 && shape_m >= 16)
            DISPATCH_TILE_MX(32, 64, 4);
        else
            DISPATCH_TILE_MX(32, 128, 4);
    }
    else if (shape_m <= 64)
    {
        if (tiles_n >= 96)
            DISPATCH_TILE_MX(64, 128, 2);
        else if (tiles_n >= 32)
            // E21 (2026-05-11): at M ∈ (32, 64] mid-N (tn∈[32,96)),
            // (64, 64, 4) wins over (32, 128, 4). At M=64 both tiles do
            // the same total work (output 32×128 = 64×64 = 4096 elems
            // per CTA, same K-loop depth) but (64, 64, 4) hits 1 wave
            // exactly on 170 SMs at gate (tn=76 → 1×152=152 CTAs ≈ 0.9
            // wave) instead of 0.9-wave-times-2 with (32, 128, 4)'s
            // 2×76=152 CTAs (same CTA count but split across M-direction
            // doesn't benefit at low M). Verified cudagraph µs:
            //   gate M=64 K=2560 tn=76: (32,128,4)=14.43 vs (64,64,4)=12.43 (-14%)
            //   wqkv M=64 K=2560 tn=48: (32,128,4)=12.65 vs (64,64,4)=12.36 (-2.3%)
            // Reproduced across 3 GPU 1 runs.
            DISPATCH_TILE_MX(64, 64, 4);
        else
            // E23 (2026-05-11): at M ∈ (32, 64] narrow-N (tn < 32),
            // K ≥ 9728 (this path is K<9728-unreachable due to
            // small_m_narrow_n override above), (64, 64, 4) + Stream-K
            // k=4 also wins decisively over (32, 128, 4) + k=4:
            //   down M=64 K=9728: (32,128,4) k=4 = 16.95 → (64,64,4) k=4 = 15.46 (-9%)
            //   M=48 same trend (~-7%); M=96 different cascade branch.
            // Same kernel-throughput story as E21 mid-N: at M=64 the
            // (64, 64, 4) tile fits 1 wave per Stream-K slice, while
            // (32, 128, 4) wastes 31/32 of the M dim per CTA at slice
            // boundaries.
            DISPATCH_TILE_MX(64, 64, 4);
    }
    else if (shape_m <= 96)
    {
        if (tiles_n >= 96)
            DISPATCH_TILE_MX(96, 128, 2);
        // E33 (2026-06): M∈(64,96] mid-N (tn∈[64,96)) — off-grid band the bench
        // (M=64,128) never exercised. The (32,128,4) fallthrough gives 3 M-tiles
        // of thin work; (64,128,2) is +30% (gate M=80/96 N=9728: 28.8→20.1 µs).
        // tn<64 (e.g. wqkv tn=48) keeps (32,128,4) — there (64,128,2) regresses.
        else if (tiles_n >= 64)
            DISPATCH_TILE_MX(64, 128, 2);
        else if (k_split >= 4)
            DISPATCH_TILE_MX(64, 128, 2);
        else
            DISPATCH_TILE_MX(32, 128, 4);
    }
    else if (shape_m <= 128)
    {
        if (tiles_n >= 96)
            DISPATCH_TILE_MX(128, 128, 2);
        else if (tiles_n >= 48)
            DISPATCH_TILE_MX(64, 128, 2);
        else if (k_split >= 4)
            DISPATCH_TILE_MX(64, 128, 2);
        else
            DISPATCH_TILE_MX(32, 128, 4);
    }
    else
    {
        // M > 128. C8 sweep (sm120_mxfp8_tile_sweep_bench, 170-SM Blackwell):
        //
        //   M=1024 N=2560  K=4096  (tn=20): (128)=653  (160)=592  → (128)
        //   M=1024 N=4096  K=4096  (tn=32): (64)=452 (96)=483 (128)=563 (160)=497 → (128)
        //   M=1024 N=6144  K=2560  (tn=48): (64)=461 (96)=511 (128)=556 (160)=640 → (160)
        //   M=1024 N=8192  K=8192  (tn=64): (64)=513 (128)=585 (160)=650        → (160)
        //   M=1024 N=19456 K=2560  (tn=152): (128)=644 (160)=613               → (128) by 5%
        //   M=2048 N=4096  K=4096  (tn=32): (128)=592 (160)=639                → (160)
        //   M=4096 N=4096  K=4096  (tn=32): (128)=641 (160)=720                → (160)
        //   M=4096 N=8192  K=8192  (tn=64): (128)=593 (160)=661                → (160)
        //   M=4096 N=2048  K=2048  (tn=16): (128)=557 (160)=595                → (160)
        //   M=2048 N=1024  K=2048  (tn=8):  (128)=518 (160)=445                → (128)
        //   M=512  N=4096  K=4096  (tn=32):           (128)=555                → (128)
        //
        // Pattern: M ≥ 1024 && tn ∈ [48, 96) → (160); M ≥ 1024 && tn ≤ 32 →
        // (128); M ≥ 2048 → (160) regardless. (224,128,2) consistently lost
        // to (160,128,2) by 5–15% (smem 95KB caps occupancy at 1/SM and
        // bigger M-tile doesn't amortize), so it never appears in the
        // cascade.
        if (tiles_n >= 96)
        {
            // E29 (2026-05-28): TileM=160 only when M ≥ 2048 — at smaller M
            // the second/last M-tile is under-utilised (M=256 → 60% util on
            // 2nd tile; M=1024 → 40% on 7th). (128,128,2) gives clean
            // tiles_m at these M with better wave fill. Sweep:
            //   M=256  N=19456 K=2560 (tn=152): (160)=523 → (128)=597 +14%
            //   M=1024 N=9728  K=2560 (tn=76):  (160)=529 → (128)=620 +17%
            if (shape_m >= 2048)
                DISPATCH_TILE_MX(160, 128, 2);
            else
            {
                // E33 (2026-06): M∈(128,2048) wide-N — the bench grid jumps
                // 128→512, so off-grid M whose TileM doesn't divide M cleanly
                // wasted up to half the last M-tile on the fixed (128,128,2)
                // pick. Choose TileM ∈ {96,128,160} minimising M-tile waste
                // (tie → larger TileM). Clean multiples of 128 (256/384/512/
                // 1024) are unchanged → (128); the win is on the in-between M:
                // M=160 +35%, M=192/176 +16%, M=288/320 +18% (gate_up N=19456).
                int const w96 = ((int(shape_m) + 95) / 96) * 96 - int(shape_m);
                int const w128 = ((int(shape_m) + 127) / 128) * 128 - int(shape_m);
                int const w160 = ((int(shape_m) + 159) / 160) * 160 - int(shape_m);
                if (w160 <= w128 && w160 <= w96)
                    DISPATCH_TILE_MX(160, 128, 2);
                else if (w128 <= w96)
                    DISPATCH_TILE_MX(128, 128, 2);
                else
                    DISPATCH_TILE_MX(96, 128, 2);
            }
        }
        else if (shape_m == 2048 && tiles_n <= 20)
            // E6 (2026-05-10): M=2048 narrow-N (tn ∈ {16, 20}): (128) wins
            // +9~17% over (160). tn=24 falls through to (160) (boundary
            // non-monotonic — at tn=24 (160) reclaims +18~23%).
            //   M=2048 N=2048 K=2048..8192 (tn=16): (128)=521..594 (160)=455..505 +14~17%
            //   M=2048 N=2560 K=2048..9728 (tn=20): (128)=617..672 (160)=556..619 +9~11%
            //   M=2048 N=3072 K=4096..9728 (tn=24): (128)=567..581 (160)=685..699 -17~19% [keep (160)]
            DISPATCH_TILE_MX(128, 128, 2);
        else if (shape_m >= 4096 && tiles_n == 20)
            // E6: down M=4096 N=2560 (tn=20): (128) wins +12~17% — narrowly
            // tn=20 only. tn=16 keeps (160) at M=4096 (different from M=2048).
            //   M=4096 N=2560 K=4096: (128)=672 (160)=602 +11.6%
            //   M=4096 N=2560 K=9728: (128)=685 (160)=587 +16.7%
            DISPATCH_TILE_MX(128, 128, 2);
        else if (shape_m >= 2048)
            DISPATCH_TILE_MX(160, 128, 2);
        else if (shape_m >= 1024 && tiles_n >= 48)
        {
            // M=1024 with tn ∈ [48, 96): wide-grid favors (160) at the
            // narrow end of that range, (128) at the wide end. Same
            // M=1024 last-M-tile waste either way (TM=160 → 40%
            // util, TM=128 → 100% util) but at tn ∈ [48, 64) the
            // per-CTA arithmetic intensity gain from a wider TileM
            // outweighs the M-waste; at tn ≥ 64 the higher tiles_n
            // already saturates and (128)'s clean tiles_m=8 wins.
            // E29 (2026-05-28) sweep:
            //   qwen-wqkv M=1024 N=6144  K=2560 (tn=48): (160)=640 (128)=521 → (160)
            //   qwen-gate M=1024 N=9728  K=2560 (tn=76): (160)=529 (128)=620 → (128) +17%
            if (tiles_n < 64)
                DISPATCH_TILE_MX(160, 128, 2);
            else
                DISPATCH_TILE_MX(128, 128, 2);
        }
        else if (shape_m >= 512 && tiles_n <= 32)
        {
            // E30 (2026-05-28): cubic-1024 wave-fill rescue. (128,128,2) at
            // M=N=K=1024 gives tiles_m × tiles_n = 8 × 8 = 64 CTAs on
            // 170 SMs = 0.38 wave (huge underfill). (128,64,2) doubles
            // tiles_n to 16 → 128 CTAs → 0.75 wave; same TileM=128
            // keeps M-bandwidth per CTA. Sweep: 193→236 TF (+22%, beats
            // cuBLAS 194 TF). Gate on tiles_m_128 * tiles_n < sm_count/2
            // so it only fires when wave fill is genuinely poor — at
            // M=2048+ or wider tiles_n the (128,128,2) tile already fills.
            int const tiles_m_128 = (static_cast<int>(shape_m) + 127) / 128;
            if (tiles_m_128 * tiles_n * 2 < num_device_sms)
                DISPATCH_TILE_MX(128, 64, 2);
            else
                DISPATCH_TILE_MX(128, 128, 2);
        }
        else if (shape_m >= 1024)
            // M ≥ 1024 with tn ∈ (32, 48): boundary band — (128) over (64).
            DISPATCH_TILE_MX(128, 128, 2);
        else
        {
            // M ∈ (128, 512). Per-shape sweep (E31, 2026-05-28):
            //   M=256 N=2560  K=9728 (tn=20, narrow-N + long-K):
            //     (64)=369 (96)=317 (128)=n/a   → keep (64,128,2)
            //   M=256 N=6144  K=2560 (tn=48):
            //     (64)=288 (96)=415 (128)=376   → (96,128,2) +44%
            //   M=256 N=9728  K=2560 (tn=76):
            //     (64)=422 (128)=562            → (128,128,2) +33%
            // The boundaries: tn < 32 is narrow-N (wave already poor at
            // M=256; the smaller (64,128,2) tile doubles tiles_m and
            // wins); tn ∈ [32, 48] wants (96) for cleanest tiles_m at
            // M=256; tn > 48 wants (128) to saturate.
            if (tiles_n < 32)
                DISPATCH_TILE_MX(64, 128, 2);   // narrow-N fallback (E31 guard)
            else if (tiles_n <= 48)
                DISPATCH_TILE_MX(96, 128, 2);
            else
                DISPATCH_TILE_MX(128, 128, 2);
        }
    }
#undef DISPATCH_TILE_MX
}

// ---------------------------------------------------------------------------
// Grouped (MoE) MXFP8 dispatch — masked layout, per-expert weights.
//
// Revives the dormant grouped path of SM120BlockScaledKernel: the persistent
// SM120BlockScaledScheduler already walks (group, m_block, n_block) work items
// when `grouped_layout != nullptr` (per-group valid-row counts, DeepGEMM
// masked semantics), and every TMA descriptor carries the group as its L
// (batch) dimension. This function only supplies the grouped strides:
//
//   A   [G, m_cap, K]  fp8   — per-group activation slab, rows beyond
//                              grouped_layout[g] are never scheduled
//   B   [G, N, K]      fp8   — per-expert weights
//   SFA [G, K/128, pad(m_cap,4)]  int32 packed UE8M0, K-major per group
//   SFB [G, K/128, pad(N,4)]      int32 packed UE8M0, K-major per group
//   D   [G, m_cap, N]  bf16  — rows beyond grouped_layout[g] are garbage
//
// `expected_m` is a host-side static hint (ceil(total_rows / G)) used only
// for tile selection — never read from device memory, so the launch is
// CUDA-Graph capture-safe and replays follow whatever masked counts the
// grouped_layout buffer holds at replay time.
//
// v0 cascade: trimmed from the dense (M, tiles_n) cascade with expected_m
// standing in for M. No Stream-K (per-group tile parallelism G×tiles_n is
// already ≥ grid size for the MoE shapes this serves); no smallm variant.
// Retune against a FORCE_TILE sweep on 5090 before freezing (M1 gate).
inline void gemm_dispatch_sm120_mxfp8_grouped(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int32_t* grouped_layout, uint32_t num_groups, uint32_t m_cap,
    uint32_t shape_n, uint32_t shape_k, uint32_t expected_m, cudaStream_t stream,
    int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

    int64_t const ld_a = shape_k;
    int64_t const ld_b = shape_k;
    int64_t const ld_d = shape_n;
    int64_t const stride_a = static_cast<int64_t>(m_cap) * shape_k;
    int64_t const stride_b = static_cast<int64_t>(shape_n) * shape_k;
    int64_t const stride_d = static_cast<int64_t>(m_cap) * shape_n;

#define DISPATCH_GROUPED_TILE_MX(TM, TN, ST)                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        launch_sm120_mxfp8_gemm_kernel<TM, TN, ST, 1, 16, true, true>(mat_a, ld_a, stride_a, mat_b, ld_b, stride_b,           \
            mat_d, ld_d, stride_d, scales_a, 0, scales_b, 0, num_groups, m_cap, shape_n, shape_k, stream,              \
            num_device_sms, grouped_layout);                                                                                           \
        return;                                                                                                        \
    } while (0)

    // FSO_FORCE_TILE=TM,TN,ST — A/B knob for the grouped tile sweep.
    // Stream-K / smallm / sched-group force flags are ignored on this path.
    auto forced = read_force_tile();
    if (force_tile_applies(forced, shape_k))
    {
        // FSO_FORCE_MIN_BLOCKS=2: co-schedule 2 CTAs/SM (grid doubles via
        // the launcher). ncu verdict on the decode cells: every pipe < 55%
        // SOL at 1 CTA/SM (latency-bound, achieved occupancy ~23%); only
        // the ~44 KB (16,64,x) instances fit 2 CTAs in the 99 KB budget.
        if (forced.min_blocks_2())
        {
            if (forced.tm == 16 && forced.tn == 64 && forced.st == 4)
            {
                launch_sm120_mxfp8_gemm_kernel<16, 64, 4, 2, 16, true, true>(mat_a, ld_a, stride_a, mat_b, ld_b, stride_b,
                    mat_d, ld_d, stride_d, scales_a, 0, scales_b, 0, num_groups, m_cap, shape_n, shape_k,
                    stream, num_device_sms, grouped_layout);
                return;
            }
            if (forced.tm == 16 && forced.tn == 64 && forced.st == 2)
            {
                launch_sm120_mxfp8_gemm_kernel<16, 64, 2, 2, 16, true, true>(mat_a, ld_a, stride_a, mat_b, ld_b, stride_b,
                    mat_d, ld_d, stride_d, scales_a, 0, scales_b, 0, num_groups, m_cap, shape_n, shape_k,
                    stream, num_device_sms, grouped_layout);
                return;
            }
            if (forced.tm == 16 && forced.tn == 128 && forced.st == 2)
            {
                launch_sm120_mxfp8_gemm_kernel<16, 128, 2, 2, 16, true, true>(mat_a, ld_a, stride_a, mat_b, ld_b, stride_b,
                    mat_d, ld_d, stride_d, scales_a, 0, scales_b, 0, num_groups, m_cap, shape_n, shape_k,
                    stream, num_device_sms, grouped_layout);
                return;
            }
            if (forced.tm == 32 && forced.tn == 64 && forced.st == 2)
            {
                launch_sm120_mxfp8_gemm_kernel<32, 64, 2, 2, 16, true, true>(mat_a, ld_a, stride_a, mat_b, ld_b, stride_b,
                    mat_d, ld_d, stride_d, scales_a, 0, scales_b, 0, num_groups, m_cap, shape_n, shape_k,
                    stream, num_device_sms, grouped_layout);
                return;
            }
            if (forced.tm == 64 && forced.tn == 64 && forced.st == 2)
            {
                launch_sm120_mxfp8_gemm_kernel<64, 64, 2, 2, 16, true, true>(mat_a, ld_a, stride_a, mat_b, ld_b, stride_b,
                    mat_d, ld_d, stride_d, scales_a, 0, scales_b, 0, num_groups, m_cap, shape_n, shape_k,
                    stream, num_device_sms, grouped_layout);
                return;
            }
            std::fprintf(stderr,
                "[fso] FSO_FORCE_MIN_BLOCKS=2 grouped: wired for (16,64,x)/(16,128,2)/(32,64,2)/(64,64,2); ignoring\n");
        }
        if (forced.tm == 16 && forced.tn == 128 && forced.st == 4) DISPATCH_GROUPED_TILE_MX(16, 128, 4);
        if (forced.tm == 16 && forced.tn == 64 && forced.st == 4)  DISPATCH_GROUPED_TILE_MX(16, 64, 4);
        if (forced.tm == 32 && forced.tn == 128 && forced.st == 4) DISPATCH_GROUPED_TILE_MX(32, 128, 4);
        if (forced.tm == 32 && forced.tn == 64 && forced.st == 4)  DISPATCH_GROUPED_TILE_MX(32, 64, 4);
        if (forced.tm == 64 && forced.tn == 64 && forced.st == 4)  DISPATCH_GROUPED_TILE_MX(64, 64, 4);
        if (forced.tm == 64 && forced.tn == 128 && forced.st == 2) DISPATCH_GROUPED_TILE_MX(64, 128, 2);
        if (forced.tm == 32 && forced.tn == 128 && forced.st == 2) DISPATCH_GROUPED_TILE_MX(32, 128, 2);
        if (forced.tm == 32 && forced.tn == 64 && forced.st == 2)  DISPATCH_GROUPED_TILE_MX(32, 64, 2);
        if (forced.tm == 64 && forced.tn == 64 && forced.st == 2)  DISPATCH_GROUPED_TILE_MX(64, 64, 2);
        if (forced.tm == 96 && forced.tn == 64 && forced.st == 2)  DISPATCH_GROUPED_TILE_MX(96, 64, 2);
        if (forced.tm == 160 && forced.tn == 128 && forced.st == 2) DISPATCH_GROUPED_TILE_MX(160, 128, 2);
        if (forced.tm == 96 && forced.tn == 128 && forced.st == 2) DISPATCH_GROUPED_TILE_MX(96, 128, 2);
        if (forced.tm == 128 && forced.tn == 128 && forced.st == 2) DISPATCH_GROUPED_TILE_MX(128, 128, 2);
        std::fprintf(stderr,
            "[fso] FSO_FORCE_TILE=%d,%d,%d not wired on the grouped path; falling through to cascade\n",
            forced.tm, forced.tn, forced.st);
    }

    uint32_t const tiles_n = (shape_n + 127) / 128;
    uint32_t const em = expected_m == 0 ? 1 : expected_m;

#define DISPATCH_GROUPED_TILE_MX_MB2(TM, TN, ST)                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        launch_sm120_mxfp8_gemm_kernel<TM, TN, ST, 2, 16, true, true>(mat_a, ld_a, stride_a, mat_b, ld_b, stride_b,           \
            mat_d, ld_d, stride_d, scales_a, 0, scales_b, 0, num_groups, m_cap, shape_n, shape_k, stream,              \
            num_device_sms, grouped_layout);                                                                                           \
        return;                                                                                                        \
    } while (0)

    // v1 cascade (2026-09-01, FSO_FORCE_TILE sweep on 5090 C1, Qwen3-30B-A3B
    // narrow-N shapes): the grouped kernel is latency/occupancy-bound at
    // 1 CTA/SM (ncu: no pipe > 55% SOL at decode, 76% DRAM at streaming);
    // the 2-CTAs/SM (16,64,4) instance wins or ties every narrow-N cell —
    // -10.8..-12.5% at em=1 decode, -0.2..-2.7% in the streaming band —
    // except short-K em>=16 where (32,128,4) takes -4.4% (down M=512).
    // Known concession: gate_up-like M=4 prefers (16,128,4) by ~4%; kept on
    // the mb2 route for cascade simplicity.
    if (tiles_n < 32)
    {
        // v4 (2026-09-02, dual-GPU 18-config x 14-M sweep, swM_*): the
        // grouped narrow-N cascade has TWO axes, mirroring the dense
        // E30/E31/E33 wave rules:
        //   * expected_m (rows per group): once em grows past ~32 the small
        //     decode tiles collapse — TileM must track em or large-M runs
        //     2-3x slow (gate_up M=4096: (96,128,2) 453.8 us vs the decode
        //     route's 1363.8; down M=2048: 202.6 vs 331.2).
        //   * SF-span partiality (K % 512): partial-span shapes take the
        //     pad-free Stages=2 instances throughout.
        bool const partial_span = ((shape_k + 127) / 128) % 4 != 0;
        // v5 (2026-09-04, Family C / Qwen3.5-35B-A3B, 5090 C1,
        // `sweep_sm120_20260904/`): very short K — one SF span, <= 4 k-tiles
        // (moe.down N=2048 K=512) — at em 16..24 takes the 2-CTA/SM (16,64,4)
        // instance instead of the (32,128,4) the em>=16 short-K rule below
        // picks. Decided on the ROUTED LAYER cell (down forced via
        // FSO_FORCE_TILE + FSO_FORCE_TILE_K=512, gate_up on its cascade pick,
        // 2 passes, µs graph median):
        //   em=16 (M=512):  layer 585.3 vs (32,128,4) 607.7   -3.7%   block+shared 618.0 vs 627.0  -1.4%
        //   em=20 (M=640):  layer 606.4 vs (32,128,4) 615.4   -1.5%
        //   em=24 (M=768):  layer 618.3 vs (32,128,4) 618.0    tie
        //   em=28 (M=896):  layer 632.0 vs (32,128,4) 632.1    tie   -> old pick kept
        //   em=32 (M=1024): layer 657.1 vs (32,128,4) 653.0   +0.6%  -> old pick kept
        // The isolated kernel cell (fso_mxfp8_grouped) ranks tiles
        // differently — it had (16,64,4) at -6.5% for em=32 and (16,128,4) at
        // -7.4% for em=64, yet in the layer those picks cost +2.8% and +0.5%
        // ((16,64,4) at em=64 is +16% in the layer) — so this rule and any
        // future grouped tile change is accepted on the layer cell only (see
        // docs/perf/README.md section 8). em=64: (32,64,4) is 0.4% better than the
        // (32,128,4) pick in the layer, inside the noise floor, unchanged.
        // gate_up (K=2048) and the Family B down (K=768, 6 k-tiles) are outside
        // the gate.
        bool const very_short_k = ((shape_k + 127) / 128) <= 4;
        if (very_short_k && em >= 16 && em <= 24)
            DISPATCH_GROUPED_TILE_MX_MB2(16, 64, 4);
        if (em > 128)
        {
            // em=256 sweep point: (96,128,2) beats (64,128,2) by 13%/10.7%
            // (gate_up/down). Larger em unswept (slab sizes) — stays here.
            DISPATCH_GROUPED_TILE_MX(96, 128, 2);
        }
        if (em > 64)
            DISPATCH_GROUPED_TILE_MX(64, 128, 2);
        if (em > 32)
        {
            if (partial_span)
                DISPATCH_GROUPED_TILE_MX(32, 128, 2);
            DISPATCH_GROUPED_TILE_MX(32, 128, 4);
        }
        // em <= 32 decode/mid band: v3 partial-span route, then v2 rules.
        if (partial_span)
        {
            launch_sm120_mxfp8_gemm_kernel<16, 128, 2, 2, 16, true, true>(mat_a, ld_a, stride_a, mat_b, ld_b, stride_b,
                mat_d, ld_d, stride_d, scales_a, 0, scales_b, 0, num_groups, m_cap, shape_n, shape_k,
                stream, num_device_sms, grouped_layout);
            return;
        }
        if (em >= 16 && shape_k <= 1024)
            DISPATCH_GROUPED_TILE_MX(32, 128, 4);
        // v2 (sweep3, post scheduler fix): at decode row-caps the wider
        // (16,128,4) single-CTA instance wins big (down M=1 -18.8%, M=4/8
        // -10/-6.8%, gate_up M=4 -10.9%) because its tile count
        // (G_active x tiles_n(128)) stays at or under the SM count, so every
        // CTA runs solo — the K/N partition sweeps measured a ~2x per-k-iter
        // penalty whenever two active CTAs share an SM. v4 refinement
        // (swM sweep): only M=1 (m_cap=4) prefers the solo (16,128,4); M=2
        // (m_cap=8) is +10.9% there and wants the 2-CTA (16,64,4) instead,
        // so gate the solo route at m_cap<=4.
        if (m_cap <= 4 && em <= 1)
            DISPATCH_GROUPED_TILE_MX(16, 128, 4);
        DISPATCH_GROUPED_TILE_MX_MB2(16, 64, 4);
    }
    if (em <= 16)
    {
        if (tiles_n >= 48 && shape_k <= 4096)
            DISPATCH_GROUPED_TILE_MX(16, 128, 4);
        DISPATCH_GROUPED_TILE_MX(32, 128, 4);
    }
    if (em <= 32)
    {
        DISPATCH_GROUPED_TILE_MX(32, 128, 4);
    }
    if (em <= 64)
    {
        if (tiles_n >= 96)
            DISPATCH_GROUPED_TILE_MX(64, 128, 2);
        DISPATCH_GROUPED_TILE_MX(64, 64, 4);
    }
    if (em <= 96)
    {
        if (tiles_n >= 32)
            DISPATCH_GROUPED_TILE_MX(96, 128, 2);
        DISPATCH_GROUPED_TILE_MX(64, 128, 2);
    }
    if (tiles_n < 32)
        DISPATCH_GROUPED_TILE_MX(64, 128, 2);
    if (tiles_n <= 48)
        DISPATCH_GROUPED_TILE_MX(96, 128, 2);
    DISPATCH_GROUPED_TILE_MX(128, 128, 2);
#undef DISPATCH_GROUPED_TILE_MX_MB2
#undef DISPATCH_GROUPED_TILE_MX
}

} // namespace kernels::blockscale_gemm
TRTLLM_NAMESPACE_END
