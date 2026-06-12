/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SM120 (Blackwell sm_120a) dispatch for fp8 blockscale GEMM.
// Uses CUTLASS Sm120 BlockScaledKernel with mxf8f6f4 MMA (UE8M0 scales).
// Large- and mid-N regime: single persistent launch with tile chosen for
// SM occupancy. Narrow-N regime (tiles_n < 32): Stream-K (concurrent K-split
// across side streams) to fight SM under-utilization.

#pragma once

#include "blockscale_gemm/common/kernel_utils.cuh"
#include "blockscale_gemm/arch/sm120/common/env_overrides.cuh"
#include "blockscale_gemm/arch/sm120/fp8/gemm_1d1d.cuh"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

TRTLLM_NAMESPACE_BEGIN
namespace kernels::blockscale_gemm
{

template <int TileM, int TileN, int NumStages, int MinBlocksPerSm = 1>
void launch_sm120_gemm_kernel(__nv_fp8_e4m3* mat_a, int64_t ld_a, int64_t stride_a, __nv_fp8_e4m3* mat_b, int64_t ld_b,
    int64_t stride_b, __nv_bfloat16* mat_d, int64_t ld_d, int64_t stride_d, float* scales_a, int64_t /*stride_scales_a*/,
    float* scales_b, int64_t /*stride_scales_b*/, uint32_t num_problems, uint32_t shape_m, uint32_t shape_n,
    uint32_t shape_k, cudaStream_t stream, int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

    using ElementInput = cute::float_e4m3_t;
    using ElementOutput = cute::bfloat16_t;
    using ElementBlockScale = int32_t;
    using KT = sm120_blockscaled_gemm::SM120BlockScaledBuilder<TileM, TileN, NumStages, MinBlocksPerSm>;
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

    Arguments args{ptr_A, dA, ptr_B, dB, ptr_SFA, dSFA, ptr_SFB, dSFB, ptr_D, dD};
    Params kernel_params = GemmKernel::to_underlying_arguments(problem_shape, args);
    auto kernel_ptr = &cutlass::device_kernel<GemmKernel>;

    cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, GemmKernel::kSmemSize);
    auto result = cudaGetLastError();
    TLLM_CHECK_WITH_INFO(result == cudaSuccess, "sm120 gemm kernel cannot launch: %s", cudaGetErrorString(result));

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

    result = cudaGetLastError();
    TLLM_CHECK_WITH_INFO(result == cudaSuccess, "sm120 gemm kernel runtime error: %s", cudaGetErrorString(result));
}

} // namespace kernels::blockscale_gemm
TRTLLM_NAMESPACE_END

// Stream-K wrapper depends on launch_sm120_gemm_kernel above; include after.
#include "blockscale_gemm/arch/sm120/common/streamk.cuh"

TRTLLM_NAMESPACE_BEGIN
namespace kernels::blockscale_gemm
{

inline void gemm_dispatch_sm120(void* mat_a, void* mat_b, void* mat_d, float* scales_a, float* scales_b,
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, cudaStream_t stream, int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

    auto* a = reinterpret_cast<__nv_fp8_e4m3*>(mat_a);
    auto* b = reinterpret_cast<__nv_fp8_e4m3*>(mat_b);
    auto* d = reinterpret_cast<__nv_bfloat16*>(mat_d);
    int64_t ld_a = shape_k;
    int64_t ld_b = shape_k;
    int64_t ld_d = shape_n;
    constexpr int64_t stride = 0;
    constexpr uint32_t num_problems = 1;

    // E15: same negative result as the MXFP8 path — see mxfp8/dispatch.cuh
    // note. GEMV kernel kept in `common/gemv.cuh` for reference but not
    // wired into the BSFP8 dispatcher either.

    // Diagnostic env-var override. See env_overrides.cuh for the wire format.
    // FSO_FORCE_TILE / FSO_FORCE_KSPLIT are shared with the MXFP8 path
    // so a single sweep harness drives both. Tile combinations advertised
    // here are FP8-specific (NS=4 at TileM ≤ 64, NS=2 at TileM ≥ 96).
    //
    // P3 footnote: a (32,128,2) variant exists in the builder template but
    // is NOT advertised here. The occupancy probe at
    // docs/skills/blockscale-gemm-tuning/references/baselines/p3_occupancy_probe.cu
    // showed (32,128,2) already runs at 2 CTAs/SM by virtue of its 43 KB
    // smem footprint; bench-wise it loses 5-7 % to (32,128,4) at small M
    // because the deeper NS=4 pipeline beats the occupancy lift the smem
    // budget already grants. See cuobjdump-baseline.md "P3 negative result".
    auto forced = read_force_tile();
    if (forced.active())
    {
        int const k_split = forced.stream_k() ? forced.ks : 1;
        bool handled = false;
        if (k_split <= 1)
        {
            #define DISPATCH_FORCED(TM_, TN_, ST_) \
                launch_sm120_gemm_kernel<TM_, TN_, ST_>(a, ld_a, stride, b, ld_b, stride, d, ld_d, stride, \
                    scales_a, stride, scales_b, stride, num_problems, shape_m, shape_n, shape_k, stream, num_device_sms)
            if      (forced.tm ==  32 && forced.tn == 128 && forced.st == 4) { DISPATCH_FORCED( 32, 128, 4); handled = true; }
            else if (forced.tm ==  32 && forced.tn ==  64 && forced.st == 4) { DISPATCH_FORCED( 32,  64, 4); handled = true; }
            else if (forced.tm ==  64 && forced.tn == 128 && forced.st == 4) { DISPATCH_FORCED( 64, 128, 4); handled = true; }
            else if (forced.tm ==  64 && forced.tn ==  64 && forced.st == 4) { DISPATCH_FORCED( 64,  64, 4); handled = true; }
            else if (forced.tm ==  96 && forced.tn == 128 && forced.st == 2) { DISPATCH_FORCED( 96, 128, 2); handled = true; }
            else if (forced.tm == 128 && forced.tn == 128 && forced.st == 2) { DISPATCH_FORCED(128, 128, 2); handled = true; }
            else if (forced.tm == 160 && forced.tn == 128 && forced.st == 2) { DISPATCH_FORCED(160, 128, 2); handled = true; }
            #undef DISPATCH_FORCED
        }
        else
        {
            #define DISPATCH_FORCED_SK(TM_, TN_, ST_) \
                launch_sm120_streamk_gemm<TM_, TN_, ST_>(a, static_cast<int>(ld_a), b, static_cast<int>(ld_b), d, \
                    static_cast<int>(ld_d), static_cast<int>(shape_m), static_cast<int>(shape_n), \
                    static_cast<int>(shape_k), scales_a, scales_b, k_split, num_device_sms, stream)
            if      (forced.tm ==  32 && forced.st == 4) { DISPATCH_FORCED_SK( 32, 128, 4); handled = true; }
            else if (forced.tm ==  64 && forced.st == 4) { DISPATCH_FORCED_SK( 64, 128, 4); handled = true; }
            else if (forced.tm == 128 && forced.st == 2) { DISPATCH_FORCED_SK(128, 128, 2); handled = true; }
            #undef DISPATCH_FORCED_SK
        }
        if (handled) return;
        // unknown combo: fall through to regular dispatch
    }

    // 2-D dispatch table tuned on a 170-SM Blackwell GPU (sm_120a, 99 KB SMEM/block).
    //   tiles_n >= 96       -> single launch full SM, big TileM
    //   tiles_n in [32, 96) -> single launch mid TileM
    //   tiles_n < 32        -> Stream-K (k_split chosen by heuristic),
    //                          k_split=1 falls through to single launch
    int const tiles_n = static_cast<int>((shape_n + 127) / 128);
    int const k_split = sm120_streamk_choose_k_split(static_cast<int>(shape_m), static_cast<int>(shape_n),
        static_cast<int>(shape_k), num_device_sms);

#define DISPATCH_TILE(TM_, TN_, ST_)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        if (k_split <= 1)                                                                                              \
        {                                                                                                              \
            launch_sm120_gemm_kernel<TM_, TN_, ST_>(a, ld_a, stride, b, ld_b, stride, d, ld_d, stride, scales_a,       \
                stride, scales_b, stride, num_problems, shape_m, shape_n, shape_k, stream, num_device_sms);            \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            launch_sm120_streamk_gemm<TM_, TN_, ST_>(a, static_cast<int>(ld_a), b, static_cast<int>(ld_b), d,           \
                static_cast<int>(ld_d), static_cast<int>(shape_m), static_cast<int>(shape_n),                          \
                static_cast<int>(shape_k), scales_a, scales_b, k_split, num_device_sms, stream);                       \
        }                                                                                                              \
    } while (0)

    // K-aware override (2026-05-05 P2 retune + E19 narrow-N port).
    //
    // History: the original 2026-05 sweep found that K <= 4096 + small-M was
    // best served by single-launch (32,128,4). However that sweep was run
    // against a CUTLASS path that returned NaN, so the "wins" were partly
    // against broken kernels. Re-validating against the fixed CUTLASS plus
    // the P2-tightened k_split heuristic, the K boundary is much lower:
    //
    //   K <= 2048  + (M<=32 || tiles_n<32) : Stream-K reduce overhead dominates
    //                                         (only ~8-16 K-tiles per side
    //                                          stream); force single-launch.
    //   K  > 2048                          : let the main cascade pick.
    //                                         Stream-K k_split=2 wins by
    //                                         35-45 % on QKV-style shapes.
    //
    // E19 (2026-05-11): port the MXFP8 narrow-N (32, 64, 4) override.
    // Pre-port BSFP8 baseline at wo (N=2560 K=4096 tn=20) was stuck at
    // (32, 128, 4) single-launch (K<9728 narrow-N → no Stream-K), losing
    // 25-29% to MXFP8 which had the override. After porting C9 PermMmaTileN
    // fix to fp8/utils.cuh, the FP8 builder accepts TileN=64 too. End-to-end
    // cudagraph µs at wo M=1..128 closes the gap with MXFP8.
    if (!read_disable_overrides())
    {
        if (shape_k <= 2048 && (shape_m <= 32 || tiles_n < 32))
        {
            launch_sm120_gemm_kernel<32, 128, 4>(a, ld_a, stride, b, ld_b, stride, d, ld_d, stride, scales_a,
                stride, scales_b, stride, num_problems, shape_m, shape_n, shape_k, stream, num_device_sms);
            return;
        }
        // Narrow-N small/mid-M: gated on K<9728 (same as MXFP8 E18 — for
        // K≥9728 the cascade's Stream-K path beats single-launch (32,64,4)
        // by 23-42% under cudagraph).
        bool const small_m_narrow_n = shape_k < 9728
                                   && shape_m >= 1 && shape_m <= 128
                                   && tiles_n >= 3 && tiles_n < 32;
        bool const mid_m_narrow_n_64 = shape_k < 9728
                                    && shape_m > 128 && shape_m <= 256
                                    && tiles_n >= 3 && tiles_n < 32;
        if (small_m_narrow_n || mid_m_narrow_n_64)
        {
            if (tiles_n == 1 && shape_m >= 4096)
                launch_sm120_gemm_kernel<64, 64, 4>(a, ld_a, stride, b, ld_b, stride, d, ld_d, stride, scales_a,
                    stride, scales_b, stride, num_problems, shape_m, shape_n, shape_k, stream, num_device_sms);
            else if (mid_m_narrow_n_64)
                launch_sm120_gemm_kernel<64, 64, 4>(a, ld_a, stride, b, ld_b, stride, d, ld_d, stride, scales_a,
                    stride, scales_b, stride, num_problems, shape_m, shape_n, shape_k, stream, num_device_sms);
            else
                launch_sm120_gemm_kernel<32, 64, 4>(a, ld_a, stride, b, ld_b, stride, d, ld_d, stride, scales_a,
                    stride, scales_b, stride, num_problems, shape_m, shape_n, shape_k, stream, num_device_sms);
            return;
        }
    }

    if (shape_m <= 32)
    {
        DISPATCH_TILE(32, 128, 4);
    }
    else if (shape_m <= 64)
    {
        if (tiles_n >= 96)
            DISPATCH_TILE(64, 128, 4);
        else
            DISPATCH_TILE(32, 128, 4);
    }
    else if (shape_m <= 96)
    {
        if (tiles_n >= 96)
            DISPATCH_TILE(96, 128, 2);
        else if (k_split >= 4)
            DISPATCH_TILE(64, 128, 4);
        else
            DISPATCH_TILE(32, 128, 4);
    }
    else if (shape_m <= 128)
    {
        if (tiles_n >= 96)
            DISPATCH_TILE(128, 128, 2);
        else if (tiles_n >= 48)
            DISPATCH_TILE(64, 128, 4);
        else if (k_split >= 4)
            // Stream-K mode: each side stream gets sm_count/k_split SMs.
            // (64,128,4) at M=128 leaves 2*tiles_n<2*48=96 tiles per stream
            // for ~42 SMs/stream — good fit. (32,128,4) would oversubscribe
            // per-stream (80 tiles vs 42 SMs) and lose because the deeper
            // pipeline can't compensate the per-stream contention. Bench:
            // QKV M=128 N=2560 K=9728 with Stream-K k=4 hits 363 T on
            // (64,128,4) but only 259 T on (32,128,4). Keep (64,128,4) here.
            DISPATCH_TILE(64, 128, 4);
        else
            // Single launch + tiles_n < 48: (64,128,4) under-saturates
            // (M-tiles=2 → total<96 vs 170 SMs). (32,128,4) quadruples
            // M-tile count and recovers ~75 % SM occupancy at tiles_n=32.
            // Validated on M=128 N=4096 across K ∈ {1024..12288}:
            // (32,128,4) beats (64,128,4) by 12-15 %. Source data:
            // docs/skills/blockscale-gemm-tuning/references/baselines/p2_tile_sweep.txt.
            DISPATCH_TILE(32, 128, 4);
    }
    else
    {
        if (tiles_n >= 96)
            DISPATCH_TILE(160, 128, 2);
        else
            DISPATCH_TILE(64, 128, 4);
    }
#undef DISPATCH_TILE
}

} // namespace kernels::blockscale_gemm
TRTLLM_NAMESPACE_END
