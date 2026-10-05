/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "mma_utils.cuh"
#include "scheduler.cuh"

// Resident CTAs per SM for the swap-AB kernel. The host passes 2 for grouped-contiguous (MoE) builds when two
// CTAs' shared memory fits and sizes the grid to match (dispatch.cuh); dense builds keep 1. With BLOCK_N = 16
// the math warp-groups hold only eight accumulators, so they run on 96 registers instead of 232 and two CTAs
// share the register file: one CTA's TMA stream then covers the other's per-block epilogue and WGMMA-wait
// chain, which a single persistent CTA exposes on every block of a short-K projection. The JIT compiler adds
// the define and a "_c2" suffix to the cubin cache name.
#ifndef FSO_SWAPAB_CTAS_PER_SM
#define FSO_SWAPAB_CTAS_PER_SM 1
#endif
#include "tma_utils.cuh"
#include "utils.cuh"

namespace deep_gemm
{

// Programmatic dependent launch (PDL) for the dense GEMM: `fso_pdl_wait` blocks until the preceding kernel in the
// stream has completed and its memory is visible, and is a no-op when the launch carries no programmatic dependency
// (a plain launch, or FSO_DISABLE_PDL=1 on the host side). The instruction is written out because NVRTC does not
// declare cudaGridDependencySynchronize.
//
// The dense GEMM does not trigger its own dependents early (griddepcontrol.launch_dependents): they launch when it
// completes, as in DeepGEMM v2's sm90_fp8_gemm_1d2d. On the H200 an early trigger by thread 0 after the wait made the
// GEMM itself up to 2 % slower (0.5-1 % on most decode cells of wo, wqkv and gate_up) when its successor is not a
// programmatic dependent (measured with FSO_DISABLE_PDL=1, 2026-10-05), and gained nothing measurable in the MLP
// block, whose GEMMs are followed by silu*mul; it only helped back-to-back GEMMs.
__device__ __forceinline__ void fso_pdl_wait()
{
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

enum class Layout
{
    RowMajor,
    ColMajor
};

template <uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup>
__device__ __host__ constexpr int get_num_threads_per_sm(int block_m)
{
    DG_STATIC_ASSERT(kNumMathThreadsPerGroup == 128, "Only support 128 threads per math group");
    return (block_m == 64 ? 1 : 2) * kNumMathThreadsPerGroup + kNumTMAThreads;
}

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t NUM_WARPS_PER_BLOCK>
static __device__ __forceinline__ void write_result_to_gmem(__nv_bfloat16* gmem_d_this_block,
    __nv_bfloat16 const* smem_d, uint32_t const m_offset, uint32_t const m_boundary, uint32_t const n_offset,
    uint32_t const shape_n, uint32_t const ld_output)
{
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;
    constexpr int int4_per_tile_line = BLOCK_N * sizeof(__nv_bfloat16) / sizeof(int4);
    int int4_per_global_line = shape_n * sizeof(__nv_bfloat16) / sizeof(int4);
    constexpr auto num_lines = BLOCK_M;
    constexpr auto num_warps = NUM_WARPS_PER_BLOCK;
    int4 const* smem_d_int4 = reinterpret_cast<int4 const*>(smem_d);
    bool is_last_n_block = n_offset + BLOCK_N > shape_n;
    int int4_per_line = is_last_n_block ? int4_per_global_line % int4_per_tile_line : int4_per_tile_line;

    for (int line_idx = warp_idx; line_idx < num_lines; line_idx += num_warps)
    {
        if (m_offset + line_idx >= m_boundary)
        {
            break;
        }
        for (int elem_idx = lane_idx; elem_idx < int4_per_line; elem_idx += 32)
        {
            uint64_t idx = (uint64_t) line_idx * ld_output + n_offset;
            int4* g_data_addr = reinterpret_cast<int4*>(&gmem_d_this_block[idx]) + elem_idx;
            int4 const* s_data_addr = &smem_d_int4[line_idx * (int4_per_tile_line) + elem_idx];
            *g_data_addr = *s_data_addr;
        }
        __syncwarp();
    }
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kNumGroups,
    uint32_t kNumStages, uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup, uint32_t kNumTMAMulticast,
    typename SchedulerType, typename InputType>
__global__ void __launch_bounds__(get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M), 1)
    fp8_gemm_kernel(__nv_bfloat16* gmem_d, float* scales_b, InputType problem_input,
        __grid_constant__ const CUtensorMap tensor_map_a, __grid_constant__ const CUtensorMap tensor_map_b,
        __grid_constant__ const CUtensorMap tensor_map_scales_a, __grid_constant__ const CUtensorMap tensor_map_d)
{
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ == 900))
    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(ceil_div(BLOCK_N, BLOCK_K) == 1, "Too much B scales in a single block");

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Shared memory
    static constexpr int kMustUseUniformedScaleB = (BLOCK_K % BLOCK_N == 0);
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_SCALES_A_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    static constexpr uint32_t SHAPE_K_SCALES = ceil_div(SHAPE_K, BLOCK_K);
    static constexpr uint32_t SMEM_SCALES_B_SIZE
        = ceil_div<uint32_t>(SHAPE_K_SCALES * (kMustUseUniformedScaleB ? 1 : 2) * sizeof(float), sizeof(Barrier))
        * sizeof(Barrier);

    // Configs
    constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K;
    constexpr uint32_t kNumThreads = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
    constexpr uint32_t kNumMathThreads = kNumThreads - kNumTMAThreads;
    constexpr uint32_t kNumIterations = ceil_div(SHAPE_K, kFullKOfAllStages);
    uint32_t const warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t const lane_idx = get_lane_id();

    // Prefetch TMA descriptors at very beginning
    if (threadIdx.x == kNumMathThreads)
    {
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_scales_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_d));
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    __nv_fp8_e4m3* smem_a[kNumStages];
    __nv_fp8_e4m3* smem_b[kNumStages];
    float* smem_scales_a[kNumStages];
    float* smem_scales_b;

    // TMA Barrier for both divisible and non-divisible cases
    Barrier* full_barriers[kNumStages];
    Barrier* empty_barriers[kNumStages];

// Fill shared memory pointers
#pragma unroll
    for (int i = 0; i < kNumStages; ++i)
    {
        smem_a[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
        smem_b[i] = reinterpret_cast<__nv_fp8_e4m3*>(
            smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
        smem_scales_a[i] = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE
            + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) + i * SMEM_SCALES_A_SIZE_PER_STAGE);
    }
    smem_scales_b = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE
        + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE));

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_scales_b) + SMEM_SCALES_B_SIZE);
#pragma unroll
    for (int i = 0; i < kNumStages; ++i)
    {
        full_barriers[i] = barrier_start_ptr + i;
        empty_barriers[i] = barrier_start_ptr + kNumStages + i;
    }

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast");
    if (threadIdx.x == kNumMathThreads)
    {
#pragma unroll
        for (int i = 0; i < kNumStages; ++i)
        {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_view_async_shared();
        (kNumTMAMulticast > 1) ? cutlass::arch::fence_barrier_init() : void();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // PDL, dense GEMM only. Everything above reads only kernel parameters and shared memory, so it may overlap the
    // preceding kernel (the activation quantize, which lets this GEMM launch as soon as its CTAs have started); every
    // global access below (A and its scales, the B scales, D) comes after the wait. The grouped (MoE) instantiations
    // compile without it.
    if constexpr (SchedulerType::gemm_type == GemmType::Normal)
        fso_pdl_wait();

    // For pipeline unrolling
    struct DivisibleK
    {
    };

    struct NotDivisibleK
    {
    };

    auto launch_k_iterations = [](auto const& func)
    {
        if constexpr (SHAPE_K % kFullKOfAllStages == 0)
        {
            for (int k_iter = 0; k_iter < kNumIterations; ++k_iter)
                func(k_iter, DivisibleK{});
        }
        else
        {
            for (int k_iter = 0; k_iter < kNumIterations - 1; ++k_iter)
                func(k_iter, DivisibleK{});
            func(kNumIterations - 1, NotDivisibleK{});
        }
    };

    // Register reconfigurations
    constexpr int kNumTMARegisters = 40;
    constexpr int kNumMathRegisters = 232;

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = SchedulerType(problem_input);

    if (threadIdx.x >= kNumMathThreads)
    {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        if (threadIdx.x == kNumMathThreads)
        {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx))
            {
                launch_k_iterations(
                    [&](int k_iter, auto type)
                    {
                        constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                        constexpr int kNumInnerStages
                            = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                        DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

#pragma unroll
                        for (uint32_t s = 0; s < kNumInnerStages; ++s)
                        {
                            // Wait consumer release
                            empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);

                            // Issue TMA A with broadcasting
                            auto& full_barrier = *full_barriers[s];
                            int k_idx = k_iter * kFullKOfAllStages + s * BLOCK_K;
                            tma_copy<kNumTMAMulticast>(&tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                smem_a[s], k_idx, scheduler.get_global_m_idx(m_block_idx));

                            if constexpr (SchedulerType::gemm_type == GemmType::GroupedWithOffset)
                            {
                                tma_copy<kNumTMAMulticast>(&tensor_map_scales_a,
                                    reinterpret_cast<uint64_t*>(&full_barrier), smem_scales_a[s],
                                    scheduler.get_global_scales_a_idx(m_block_idx), k_idx / BLOCK_K);
                            }
                            else
                            {
                                tma_copy<kNumTMAMulticast>(&tensor_map_scales_a,
                                    reinterpret_cast<uint64_t*>(&full_barrier), smem_scales_a[s], m_block_idx * BLOCK_M,
                                    scheduler.get_global_scales_a_idx(k_idx / BLOCK_K));
                            }

                            // Issue TMA B without broadcasting
                            tma_copy(&tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier), smem_b[s], k_idx,
                                scheduler.get_global_n_idx(SHAPE_N, BLOCK_N, n_block_idx, m_block_idx));
                            full_barrier.arrive_and_expect_tx(
                                SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE);
                        }

// Wait unaligned cases
#pragma unroll
                        for (uint32_t s = kNumInnerStages; s < kNumStages; ++s)
                        {
                            empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);
                            full_barriers[s]->arrive();
                        }
                    });
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1)
            {
#pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++s)
                    empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + 1) & 1);
            }
        }
    }
    else
    {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        auto const math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / kNumMathThreadsPerGroup, 0);
        auto const r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            // Decide the number of scales B to load
            DG_STATIC_ASSERT(SHAPE_N % 8 == 0, "Invalid shape N");
            uint32_t num_former_iters = BLOCK_N / 8, num_full_iters = num_former_iters;
            if constexpr (not kMustUseUniformedScaleB)
            {
                num_former_iters = min(BLOCK_N, BLOCK_K - n_block_idx * BLOCK_N % BLOCK_K) / 8;
                num_full_iters = min(SHAPE_N - n_block_idx * BLOCK_N, BLOCK_N) / 8;
            }
            uint32_t num_scales_b = SHAPE_K_SCALES * (num_former_iters >= num_full_iters ? 1 : 2);

            // Load B scales with math warp-groups
            // NOTES: except the first warp, we want to overlap loading B scales with TMA stores between tasks
            if (threadIdx.x >= 32)
            {
                auto num_previous_lines
                    = scheduler.get_global_scales_b_idx(ceil_div(SHAPE_N, BLOCK_K), 0, 0, m_block_idx);
                ;
                auto local_scales_b
                    = scales_b + (num_previous_lines + ((n_block_idx * BLOCK_N) / BLOCK_K)) * SHAPE_K_SCALES;
#pragma unroll
                for (uint32_t i = threadIdx.x - 32; i < num_scales_b; i += kNumMathThreads - 32)
                    st_shared(smem_scales_b + i, __ldg(local_scales_b + i));
            }
            cutlass::arch::NamedBarrier(kNumMathThreads).sync();

            // Accumulation for WGMMA or CUDA promotion
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum] = {0};

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](int s)
            {
                if constexpr (kNumTMAMulticast == 1)
                {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                }
                else
                {
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(lane_idx) : void();
                }
            };

            // Launch MMAs
            launch_k_iterations(
                [&](int k_iter, auto type)
                {
                    constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                    constexpr int kNumInnerStages
                        = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                    DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

#pragma unroll
                    for (int s = 0; s < kNumInnerStages; ++s)
                    {
                        // Read B scales
                        float scale_b_0 = ld_shared(smem_scales_b + k_iter * kNumStages + s), scale_b_1 = 1.0f;
                        // NOTES: even some blocks do not need to read the second row, but we still load one to align
                        // with other blocks
                        if constexpr (not kMustUseUniformedScaleB)
                            scale_b_1 = ld_shared(smem_scales_b + k_iter * kNumStages + s + SHAPE_K_SCALES);

                        // Wait TMA arrivals
                        full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);

                        // Read A scales
                        // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled
                        // block polluting the results
                        auto scale_a_0 = ld_shared(smem_scales_a[s] + r_0),
                             scale_a_1 = ld_shared(smem_scales_a[s] + r_1);

// Commit WGMMA instructions
#pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum; ++i)
                            warpgroup_fence_operand(accum[i]);
                        warpgroup_arrive();
#pragma unroll
                        for (int k = 0; k < BLOCK_K / WGMMA::K; ++k)
                        {
                            auto desc_a
                                = make_smem_desc(smem_a[s] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                            auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        warpgroup_commit_batch();
#pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum; ++i)
                            warpgroup_fence_operand(accum[i]);
                        warpgroup_wait<0>();

                        // Notify barrier arrival
                        empty_barrier_arrive(s);

                        // Promote with scales
                        float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
                        float scale_0_1, scale_1_1;
                        if constexpr (not kMustUseUniformedScaleB)
                            scale_0_1 = scale_a_0 * scale_b_1, scale_1_1 = scale_a_1 * scale_b_1;
#pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum / 4; ++i)
                        {
                            bool predicate = kMustUseUniformedScaleB or i < num_former_iters;
                            final_accum[i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
                            final_accum[i * 4 + 1] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
                            final_accum[i * 4 + 2] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
                            final_accum[i * 4 + 3] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
                        }
                    }

// Wait unaligned cases
#pragma unroll
                    for (uint32_t s = kNumInnerStages; s < kNumStages; ++s)
                    {
                        full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);
                        empty_barrier_arrive(s);
                    }
                });

            // Dense GEMM with BLOCK_N a multiple of 64: the D store path of DeepGEMM v2's sm90_fp8_gemm_1d2d. STSM
            // writes D into shared memory as BLOCK_N / 64 slabs of BLOCK_M x 64 bf16 (128 bytes per row) with the
            // 128-byte TMA swizzle, so the stores of a warp hit distinct banks (the unswizzled tile below, whose rows
            // are BLOCK_N * 2 bytes apart, puts the eight rows of an 8x8 matrix in the same banks). Threads
            // 0 .. BLOCK_N / 64 - 1 each store one slab with its own TMA store, and the wait for the previous tile's
            // stores moves to just before this tile overwrites the staging buffer, so the store overlaps the next
            // tile's main loop. The arithmetic and the bytes written are unchanged: only the shared-memory staging
            // layout and the store issue differ, and the output is bitwise identical to the path below.
            constexpr bool kSwizzledD = (SchedulerType::gemm_type == GemmType::Normal) && (BLOCK_N % 64 == 0);
            if constexpr (kSwizzledD)
            {
                DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
                // smem_d is single-buffered: the previous tile's stores must have read it before it is rewritten.
                if (threadIdx.x < BLOCK_N / 64)
                    cute::tma_store_wait<0>();
                cutlass::arch::NamedBarrier(kNumMathThreads).sync();
#pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++i)
                {
                    // Accumulators 4i .. 4i + 3 are columns 8i .. 8i + 7 of the warp's rows (lane / 4) and
                    // (lane / 4 + 8). They form two 8x8 matrices that one STSM x2 writes; lanes 0 .. 15 give the
                    // row addresses (row = lane within the warp's 16 rows). Column chunk i % 8 of slab i / 8 is
                    // 16 bytes wide and sits at chunk (i % 8) ^ (row % 8) of its 128-byte row (the TMA swizzle).
                    uint32_t const atom = i / 8, in_atom = i % 8;
                    uint32_t const row = lane_idx;
                    uint32_t const col = in_atom ^ (row % 8);
                    auto smem_ptr = reinterpret_cast<uint8_t*>(smem_d) + warp_idx * (16 * 128) + atom * BLOCK_M * 128
                        + row * 128 + col * 16;
                    SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                        __float22bfloat162_rn({final_accum[i * 4 + 0], final_accum[i * 4 + 1]}),
                        __float22bfloat162_rn({final_accum[i * 4 + 2], final_accum[i * 4 + 3]}), smem_ptr);
                }
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier(kNumMathThreads).sync();
                if (threadIdx.x < BLOCK_N / 64)
                {
                    cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_d + threadIdx.x * BLOCK_M * 64,
                        n_block_idx * BLOCK_N + threadIdx.x * 64, scheduler.get_global_m_idx(m_block_idx));
                    cute::tma_store_arrive();
                }
                __syncwarp();
                continue;
            }

            // Write back to shared memory using STSM
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
#pragma unroll
            for (auto i = 0; i < WGMMA::kNumAccum / 8; ++i)
            {
                SM90_U32x4_STSM_N<nv_bfloat162>::copy(
                    __float22bfloat162_rn({final_accum[i * 8 + 0], final_accum[i * 8 + 1]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 2], final_accum[i * 8 + 3]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 4], final_accum[i * 8 + 5]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 6], final_accum[i * 8 + 7]}),
                    smem_d + (warp_idx * 16 + lane_idx % 16) * BLOCK_N + i * 16 + 8 * (lane_idx / 16));
            }
            if constexpr (WGMMA::kNumAccum % 8 != 0)
            {
                SM90_U32x2_STSM_N<nv_bfloat162>::copy(__float22bfloat162_rn({final_accum[WGMMA::kNumAccum / 8 * 8 + 0],
                                                          final_accum[WGMMA::kNumAccum / 8 * 8 + 1]}),
                    __float22bfloat162_rn(
                        {final_accum[WGMMA::kNumAccum / 8 * 8 + 2], final_accum[WGMMA::kNumAccum / 8 * 8 + 3]}),
                    smem_d + (warp_idx * 16 + lane_idx % 16) * BLOCK_N + WGMMA::kNumAccum / 8 * 16);
            }

            if constexpr (SchedulerType::gemm_type == GemmType::GroupedWithOffset)
            {
                auto m_global_idx = scheduler.get_global_m_idx(m_block_idx);
                bool cross_boundary = (m_global_idx + BLOCK_M) > scheduler.m_boundary;
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier(kNumMathThreads).sync();
                if (!cross_boundary)
                {
                    // Use TMA store to write back to global memory
                    if (threadIdx.x == 0)
                    {
                        cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_d, n_block_idx * BLOCK_N, m_global_idx);
                        cute::tma_store_arrive();
                        cute::tma_store_wait<0>();
                    }
                }
                else
                {
                    __nv_bfloat16* gmem_d_this_block = gmem_d + m_global_idx * SHAPE_N;
                    constexpr int NUM_WARPS
                        = (get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M) - 128) / 32;
                    write_result_to_gmem<BLOCK_M, BLOCK_N, NUM_WARPS>(gmem_d_this_block, smem_d, m_global_idx,
                        scheduler.m_boundary, n_block_idx * BLOCK_N, SHAPE_N, SHAPE_N);
                }
            }
            else if constexpr (SchedulerType::gemm_type == GemmType::StridedBatched)
            {
                cutlass::arch::NamedBarrier(kNumMathThreads).sync();
                __nv_bfloat16* gmem_d_this_block;
                auto m_global_idx = scheduler.get_global_m_idx(m_block_idx);
                gmem_d_this_block = gmem_d + scheduler.curr_group_idx * problem_input.stride_d
                    + (m_block_idx * BLOCK_M) * problem_input.ld_d;
                constexpr int NUM_WARPS
                    = (get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M) - 128) / 32;
                write_result_to_gmem<BLOCK_M, BLOCK_N, NUM_WARPS>(gmem_d_this_block, smem_d, m_global_idx,
                    scheduler.m_boundary, n_block_idx * BLOCK_N, SHAPE_N, problem_input.ld_d);
            }
            else
            {
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier(kNumMathThreads).sync();
                // Use TMA store to write back to global memory
                if (threadIdx.x == 0)
                {
                    cute::SM90_TMA_STORE_2D::copy(
                        &tensor_map_d, smem_d, n_block_idx * BLOCK_N, scheduler.get_global_m_idx(m_block_idx));
                    cute::tma_store_arrive();
                    cute::tma_store_wait<0>();
                }
            }

            __syncwarp();
        }

        // Swizzled-D path: the last tile's TMA stores must finish reading shared memory before the CTA exits.
        if constexpr ((SchedulerType::gemm_type == GemmType::Normal) && (BLOCK_N % 64 == 0))
        {
            if (threadIdx.x < BLOCK_N / 64)
                cute::tma_store_wait<0>();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

template <uint32_t SHAPE_M, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kNumGroups,
    uint32_t kNumStages, uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup, uint32_t kNumTMAMulticast,
    typename SchedulerType, typename InputType>
__global__ void __launch_bounds__(
    get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M), FSO_SWAPAB_CTAS_PER_SM)
    fp8_gemm_kernel_swapAB(__nv_bfloat16* gmem_d, float* scales_a, InputType problem_input,
        const __grid_constant__ CUtensorMap tensor_map_a,        // weight (previously act)
        const __grid_constant__ CUtensorMap tensor_map_b,        // act (previously weight)
        const __grid_constant__ CUtensorMap tensor_map_scales_b, // act scales (previously tensor_map_scales_a)
        const __grid_constant__ CUtensorMap tensor_map_d)
{
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(ceil_div(BLOCK_M, BLOCK_K) == 1, "Too much A scales in a single block");

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Shared memory
    DG_STATIC_ASSERT(BLOCK_K % BLOCK_M == 0, "BLOCK_M should be 64 or 128 and BLOCK_K should be 128");
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_N * BLOCK_M * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_SCALES_B_SIZE_PER_STAGE = BLOCK_N * sizeof(float); // B matrix (act) scales
    static constexpr uint32_t SMEM_SCALES_B_SIZE_PER_STAGE_PADDED
        = ceil_div<uint32_t>(BLOCK_N * sizeof(float), 128) * 128; // B matrix (act) scales, 128B aligned
    static constexpr uint32_t SHAPE_K_SCALES = ceil_div(SHAPE_K, BLOCK_K);
    static constexpr uint32_t SMEM_SCALES_A_SIZE = ceil_div<uint32_t>(SHAPE_K_SCALES * sizeof(float), sizeof(Barrier))
        * sizeof(Barrier); // renamed to A (weight)

    // Configs
    constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K;
    constexpr uint32_t kNumThreads = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
    constexpr uint32_t kNumMathThreads = kNumThreads - kNumTMAThreads;
    constexpr uint32_t kNumIterations = ceil_div(SHAPE_K, kFullKOfAllStages);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_id();

    // Prefetch TMA descriptors at very beginning
    if (threadIdx.x == kNumMathThreads)
    {
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_scales_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_d));
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");
    // The alignment attribute is a promise the launch has to keep: with more than one resident CTA per SM
    // the allocation base is only 1024-aligned if the host rounded the dynamic size (dispatch.cuh). Trap
    // rather than compute on a mis-phased swizzle.
    if (threadIdx.x == 0)
        DG_DEVICE_ASSERT((static_cast<uint32_t>(__cvta_generic_to_shared(smem_buffer)) & 1023u) == 0u
            and "swap-AB smem base must be 1024-byte aligned");

    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    __nv_fp8_e4m3* smem_a[kNumStages]; // weight
    __nv_fp8_e4m3* smem_b[kNumStages]; // act
    float* smem_scales_b[kNumStages];  // act scales
    float* smem_scales_a;              // weight scales

    // TMA Barrier for both divisible and non-divisible cases
    Barrier* full_barriers[kNumStages];
    Barrier* empty_barriers[kNumStages];

// Fill shared memory pointers
#pragma unroll
    for (int i = 0; i < kNumStages; ++i)
    {
        smem_a[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
        smem_b[i] = reinterpret_cast<__nv_fp8_e4m3*>(
            smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
        smem_scales_b[i] = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE
            + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) + i * SMEM_SCALES_B_SIZE_PER_STAGE_PADDED);
    }
    smem_scales_a = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE
        + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_B_SIZE_PER_STAGE_PADDED));

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_scales_a) + SMEM_SCALES_A_SIZE);
#pragma unroll
    for (int i = 0; i < kNumStages; ++i)
    {
        full_barriers[i] = barrier_start_ptr + i;
        empty_barriers[i] = barrier_start_ptr + kNumStages + i;
    }

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast");
    if (threadIdx.x == kNumMathThreads)
    {
#pragma unroll
        for (int i = 0; i < kNumStages; ++i)
        {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_view_async_shared();
        (kNumTMAMulticast > 1) ? cutlass::arch::fence_barrier_init() : void();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // PDL, dense swap-AB GEMM only: the same wait as fp8_gemm_kernel (see there). The grouped (MoE) instantiations
    // compile without it.
    if constexpr (SchedulerType::gemm_type == GemmType::Normal)
        fso_pdl_wait();

    // For pipeline unrolling
    struct DivisibleK
    {
    };

    struct NotDivisibleK
    {
    };

    auto launch_k_iterations = [](auto const& func)
    {
        if constexpr (SHAPE_K % kFullKOfAllStages == 0)
        {
            for (int k_iter = 0; k_iter < kNumIterations; ++k_iter)
                func(k_iter, DivisibleK{});
        }
        else
        {
            for (int k_iter = 0; k_iter < kNumIterations - 1; ++k_iter)
                func(k_iter, DivisibleK{});
            func(kNumIterations - 1, NotDivisibleK{});
        }
    };

    // Register reconfigurations. setmaxnreg only redistributes the registers the CTA was launched with,
    // so the TMA warp-group's release must cover the math warp-groups' increase:
    //   (compiled - kNumTMARegisters) * 128 >= (kNumMathRegisters - compiled) * 256.
    // One CTA per SM compiles to 168 registers (40 / 232 balance exactly, the DeepGEMM split). Two CTAs
    // per SM cap the kernel at 80 registers (65536 / 768, rounded down to a multiple of 8); 40 / 96 then
    // balances with 1024 registers to spare, and the swap-AB math warps (BLOCK_N = 16 -> 8 accumulators)
    // compile without spills at 80. A larger math budget deadlocks the increase (measured with 104).
    // The host refuses the two-CTA build if ptxas did not reach 80 registers (dispatch.cuh).
    constexpr int kNumTMARegisters = 40;
    constexpr int kNumMathRegisters = (FSO_SWAPAB_CTAS_PER_SM >= 2) ? 96 : 232;

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = SchedulerType(problem_input);

    if (threadIdx.x >= kNumMathThreads)
    {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        if (threadIdx.x == kNumMathThreads)
        {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx))
            {
                launch_k_iterations(
                    [&](int k_iter, auto type)
                    {
                        constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                        constexpr int kNumInnerStages
                            = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                        DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

#pragma unroll
                        for (uint32_t s = 0; s < kNumInnerStages; ++s)
                        {
                            // Wait consumer release
                            empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);

                            // Issue TMA A (weight) now without broadcasting
                            auto& full_barrier = *full_barriers[s];
                            int k_idx = k_iter * kFullKOfAllStages + s * BLOCK_K;
                            tma_copy(&tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier), smem_a[s], k_idx,
                                scheduler.get_global_m_idx(SHAPE_M, BLOCK_M, m_block_idx, n_block_idx));

                            // Issue TMA B (act) with broadcasting
                            tma_copy<kNumTMAMulticast>(&tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier),
                                smem_b[s], k_idx, scheduler.get_global_n_idx(n_block_idx));

                            // Issue TMA scales_b (act scales) for B matrix
                            if constexpr (SchedulerType::gemm_type == GemmType::GroupedWithOffset)
                            {
                                tma_copy<kNumTMAMulticast>(&tensor_map_scales_b,
                                    reinterpret_cast<uint64_t*>(&full_barrier), smem_scales_b[s],
                                    scheduler.get_global_scales_b_idx(n_block_idx), k_idx / BLOCK_K);
                            }
                            else
                            {
                                tma_copy<kNumTMAMulticast>(&tensor_map_scales_b,
                                    reinterpret_cast<uint64_t*>(&full_barrier), smem_scales_b[s], n_block_idx * BLOCK_N,
                                    scheduler.get_global_scales_b_idx(k_idx / BLOCK_K));
                            }

                            full_barrier.arrive_and_expect_tx(
                                SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_B_SIZE_PER_STAGE);
                        }

// Wait unaligned cases
#pragma unroll
                        for (uint32_t s = kNumInnerStages; s < kNumStages; ++s)
                        {
                            empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);
                            full_barriers[s]->arrive();
                        }
                    });
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1)
            {
#pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++s)
                    empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + 1) & 1);
            }
        }
    }
    else
    {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        auto const math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / kNumMathThreadsPerGroup, 0);

        // Each thread loads consecutive 2 scales
        const uint32_t scale_offset = (lane_idx % 4) * 2;

        // Grouped-contiguous (MoE) builds read the per-block weight scales straight from global memory per
        // stage (SHAPE_K_SCALES floats, L1-resident after the first touch) instead of staging them in
        // shared memory behind a CTA-wide NamedBarrier at every block start. With BLOCK_N = 16 a block of
        // the K = 512 down projection is only four pipeline stages of work, so that barrier was a visible
        // part of the block loop (ncu: barrier 5.2 vs long_scoreboard 3.5 stalled warps per issue on that
        // kernel; -5 % on it, bit-identical output). The dense and offset paths keep the smem staging.
        constexpr bool kFsoRegScales = (SchedulerType::gemm_type == GemmType::GroupedContiguous);

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            // Load weight scales (scales_a) - these are associated with tensor_map_a (weight)
            // Decide the number of scales A to load
            DG_STATIC_ASSERT(SHAPE_M % 8 == 0, "Invalid shape M");
            uint32_t num_scales_a = SHAPE_K_SCALES;

            auto const num_previous_lines
                = scheduler.get_global_scales_a_idx(ceil_div(SHAPE_M, BLOCK_K), 0, 0, n_block_idx);
            float const* const local_scales_a
                = scales_a + (num_previous_lines + ((m_block_idx * BLOCK_M) / BLOCK_K)) * SHAPE_K_SCALES;
            if constexpr (!kFsoRegScales)
            {
                // Load A scales with math warp-groups (weight scales)
                if (threadIdx.x >= 32)
                {
#pragma unroll
                    for (uint32_t i = threadIdx.x - 32; i < num_scales_a; i += kNumMathThreads - 32)
                        st_shared(smem_scales_a + i, __ldg(local_scales_a + i));
                }
                cutlass::arch::NamedBarrier(kNumMathThreads).sync();
            }

            // Accumulation for WGMMA or CUDA promotion
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum] = {0};

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](int s)
            {
                if constexpr (kNumTMAMulticast == 1)
                {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                }
                else
                {
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(lane_idx) : void();
                }
            };

            // Launch MMAs
            launch_k_iterations(
                [&](int k_iter, auto type)
                {
                    constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                    constexpr int kNumInnerStages
                        = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                    DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

#pragma unroll
                    for (int s = 0; s < kNumInnerStages; ++s)
                    {
                        // Read weight scales (A scales)
                        float scale_a_0;
                        if constexpr (kFsoRegScales)
                            scale_a_0 = __ldg(local_scales_a + k_iter * kNumStages + s);
                        else
                            scale_a_0 = ld_shared(smem_scales_a + k_iter * kNumStages + s);

                        // Wait TMA arrivals
                        full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);

                        // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled
                        // block polluting the results
                        // Each thread reads consecutive two b scales, each thread needs to read WGMMA::N / 4 * 2 b
                        // scales
                        float scale_0_0[WGMMA::kNumAccum / 4], scale_0_1[WGMMA::kNumAccum / 4];
#pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum / 4; ++i)
                        {
                            float2 scale_b
                                = ld_shared(reinterpret_cast<const float2*>(smem_scales_b[s] + i * 8 + scale_offset));
                            scale_0_0[i] = scale_a_0 * scale_b.x;
                            scale_0_1[i] = scale_a_0 * scale_b.y;
                        }

// Commit WGMMA instructions
#pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum; ++i)
                            warpgroup_fence_operand(accum[i]);
                        warpgroup_arrive();
#pragma unroll
                        for (int k = 0; k < BLOCK_K / WGMMA::K; ++k)
                        {
                            auto desc_a
                                = make_smem_desc(smem_a[s] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                            auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        warpgroup_commit_batch();
#pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum; ++i)
                            warpgroup_fence_operand(accum[i]);
                        warpgroup_wait<0>();

                        // Notify barrier arrival
                        empty_barrier_arrive(s);

// Promote with scales
#pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum / 4; ++i)
                        {
                            final_accum[i * 4 + 0] += scale_0_0[i] * accum[i * 4 + 0];
                            final_accum[i * 4 + 1] += scale_0_1[i] * accum[i * 4 + 1];
                            final_accum[i * 4 + 2] += scale_0_0[i] * accum[i * 4 + 2];
                            final_accum[i * 4 + 3] += scale_0_1[i] * accum[i * 4 + 3];
                        }
                    }

// Wait unaligned cases
#pragma unroll
                    for (uint32_t s = kNumInnerStages; s < kNumStages; ++s)
                    {
                        full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);
                        empty_barrier_arrive(s);
                    }
                });

            if constexpr (kFsoRegScales)
            {
                // smem_d is single-buffered: the previous block's TMA store must have finished reading it
                // before any math warp overwrites it below. The original kernel got this for free from the
                // block-start NamedBarrier (thread 0 waited on the store before it), which the register-read
                // scales removed; placing the wait here instead gives the store the whole mainloop to drain.
                // Without this, two resident CTAs per SM produced one corrupted 16 x 128 D tile every few
                // launches at M >= 512 (found 2026-09-25, det_localize).
                if (threadIdx.x == 0)
                    cute::tma_store_wait<0>();
                cutlass::arch::NamedBarrier(kNumMathThreads).sync();
            }

            // Write back to shared memory using STSM
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            int tid = 0;
            if (lane_idx < 8)
            {
                tid = lane_idx * BLOCK_M;
            }
            else if (lane_idx < 16)
            {
                tid = (lane_idx - 8) * BLOCK_M + 8;
            }
            else if (lane_idx < 24)
            {
                tid = (lane_idx - 8) * BLOCK_M;
            }
            else
            {
                tid = (lane_idx - 16) * BLOCK_M + 8;
            }
#pragma unroll
            for (auto i = 0; i < WGMMA::kNumAccum / 8; ++i)
            {
                SM90_U32x4_STSM_T<nv_bfloat162>::copy(
                    __float22bfloat162_rn({final_accum[i * 8 + 0], final_accum[i * 8 + 1]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 2], final_accum[i * 8 + 3]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 4], final_accum[i * 8 + 5]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 6], final_accum[i * 8 + 7]}),
                    smem_d + warp_idx * 16 + i * 16 * BLOCK_M + tid);
            }
            if constexpr (WGMMA::kNumAccum % 8 != 0)
            {
                SM90_U32x2_STSM_T<nv_bfloat162>::copy(__float22bfloat162_rn({final_accum[WGMMA::kNumAccum / 8 * 8 + 0],
                                                          final_accum[WGMMA::kNumAccum / 8 * 8 + 1]}),
                    __float22bfloat162_rn(
                        {final_accum[WGMMA::kNumAccum / 8 * 8 + 2], final_accum[WGMMA::kNumAccum / 8 * 8 + 3]}),
                    smem_d + warp_idx * 16 + WGMMA::kNumAccum / 8 * 16 * BLOCK_M + tid);
            }

            if constexpr (SchedulerType::gemm_type == GemmType::GroupedWithOffset)
            {
                auto n_global_idx = scheduler.get_global_n_idx(n_block_idx);
                bool cross_boundary = (n_global_idx + BLOCK_N) > scheduler.n_boundary;
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier(kNumMathThreads).sync();
                if (!cross_boundary)
                {
                    // Use TMA store to write back to global memory
                    if (threadIdx.x == 0)
                    {
                        cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_d, m_block_idx * BLOCK_M, n_global_idx);
                        cute::tma_store_arrive();
                        cute::tma_store_wait<0>();
                    }
                }
                else
                {
                    __nv_bfloat16* gmem_d_this_block = gmem_d + n_global_idx * SHAPE_M;
                    constexpr int NUM_WARPS
                        = (get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M) - 128) / 32;
                    write_result_to_gmem<BLOCK_N, BLOCK_M, NUM_WARPS>(gmem_d_this_block, smem_d, n_global_idx,
                        scheduler.n_boundary, m_block_idx * BLOCK_M, SHAPE_M, SHAPE_M);
                }
            }
            else if constexpr (SchedulerType::gemm_type == GemmType::StridedBatched)
            {
                cutlass::arch::NamedBarrier(kNumMathThreads).sync();
                __nv_bfloat16* gmem_d_this_block;
                auto n_global_idx = scheduler.get_global_n_idx(n_block_idx);
                gmem_d_this_block = gmem_d + scheduler.curr_group_idx * problem_input.stride_d
                    + (n_block_idx * BLOCK_N) * problem_input.ld_d;
                constexpr int NUM_WARPS
                    = (get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M) - 128) / 32;
                write_result_to_gmem<BLOCK_N, BLOCK_M, NUM_WARPS>(gmem_d_this_block, smem_d, n_global_idx,
                    scheduler.n_boundary, m_block_idx * BLOCK_M, SHAPE_M, problem_input.ld_d);
            }
            else
            {
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier(kNumMathThreads).sync();
                // Use TMA store to write back to global memory
                if (threadIdx.x == 0)
                {
                    cute::SM90_TMA_STORE_2D::copy(
                        &tensor_map_d, smem_d, m_block_idx * BLOCK_M, scheduler.get_global_n_idx(n_block_idx));
                    cute::tma_store_arrive();
                    if constexpr (!kFsoRegScales)
                        cute::tma_store_wait<0>();
                    // reg-scales builds wait right before the next block's STSM (see above) and at exit
                }
            }

            __syncwarp();
        }
        if constexpr (kFsoRegScales)
        {
            // Drain the last block's TMA store before the CTA releases its shared memory.
            if (threadIdx.x == 0)
                cute::tma_store_wait<0>();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

// Non-swap GroupedContiguous FC1 with two math warp-groups split along N (sm_90 fused-FC1, step 1, 2026-09-30).
//
// The FC1 weight of an expert is the stacked [gate; up] matrix [SHAPE_N = 2I, K]. One CTA computes, for one
// BLOCK_M = 64 activation tile and one 128-column block b of the SwiGLU output, both halves that the fused SwiGLU
// epilogue will pair: math warp-group 0 multiplies weight rows b*128 .. b*128+127 (gate block b) and math
// warp-group 1 multiplies rows I + b*128 .. I + b*128+127 (up block b). The producer fetches the two 128-row weight
// boxes with two TMA loads into adjacent shared memory, so the weights stay stacked (no interleave). Both warp-groups
// read the same A stage and the same A scales. Each warp-group runs the instruction stream of fp8_gemm_kernel above
// unchanged (m64n128k32 WGMMA, the same promotion into final_accum, one 128x128 weight-scale row), so every output
// element is accumulated exactly as fp8_gemm_kernel accumulates it; this step still stores bf16 gate and up to their
// native columns b*128 and I + b*128 of D [P_max, 2I].
//
// Row-index trap: fp8_gemm_kernel derives a thread's rows (A-scale reads, STSM address) from the CTA-wide warp index,
// which is right only when a second math warp-group owns rows 64..127. Here both warp-groups own rows 0..63, so the
// rows come from the warp's index inside its warp-group (warp_idx % 4); the halves differ only in the B descriptor
// (+128 weight rows), the weight-scale row (+I/128) and the output column (+I).
//
// Registers: 384 threads and one CTA per SM compile to 168 registers; setmaxnreg then moves the TMA warp-group to 40
// and the math warp-groups to 232, which balances exactly ((168 - 40) * 128 == (232 - 168) * 256). A cubin compiled
// to fewer than 168 registers would deadlock the increase, so the host refuses any other count (dispatch.cuh).
template <uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup>
__device__ __host__ constexpr int get_num_threads_per_sm_2wg()
{
    DG_STATIC_ASSERT(kNumMathThreadsPerGroup == 128, "Only support 128 threads per math group");
    return 2 * kNumMathThreadsPerGroup + kNumTMAThreads;
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kNumGroups,
    uint32_t kNumStages, uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup, uint32_t kNumTMAMulticast,
    typename SchedulerType, typename InputType>
__global__ void __launch_bounds__(get_num_threads_per_sm_2wg<kNumTMAThreads, kNumMathThreadsPerGroup>(), 1)
    fp8_gemm_kernel_2wg(__nv_bfloat16* gmem_d, float* scales_b, InputType problem_input,
        __grid_constant__ const CUtensorMap tensor_map_a, __grid_constant__ const CUtensorMap tensor_map_b,
        __grid_constant__ const CUtensorMap tensor_map_scales_a, __grid_constant__ const CUtensorMap tensor_map_d)
{
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ == 900))
    // Scaling and shape checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(BLOCK_N == BLOCK_K, "Each warp-group half must be exactly one 128-row weight-scale block");
    DG_STATIC_ASSERT(BLOCK_M == 64, "Both warp-groups own the same 64 rows (one WGMMA row block)");
    DG_STATIC_ASSERT(SHAPE_N % (2 * BLOCK_N) == 0, "SHAPE_N must be 2 * I with I a multiple of BLOCK_N");
    DG_STATIC_ASSERT(kNumTMAMulticast == 1, "No TMA multicast: the per-block weights differ");
    DG_STATIC_ASSERT(SchedulerType::gemm_type == GemmType::GroupedContiguous, "Grouped-contiguous FC1 only");

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(WGMMA::M == BLOCK_M, "One WGMMA row block per warp-group");

    // Shared memory: [D gate | D up] [A x stages] [B gate | B up x stages] [A scales x stages] [B scales gate | up]
    static constexpr uint32_t kNumMathWarpGroups = 2;
    static constexpr uint32_t SHAPE_N_HALF = SHAPE_N / 2; // I: the first up row of an expert's [gate; up] weight
    static constexpr uint32_t SMEM_D_SIZE_PER_WG = BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_D_SIZE = kNumMathWarpGroups * SMEM_D_SIZE_PER_WG;
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_WG = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = kNumMathWarpGroups * SMEM_B_SIZE_PER_WG;
    static constexpr uint32_t SMEM_SCALES_A_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    static constexpr uint32_t SHAPE_K_SCALES = ceil_div(SHAPE_K, BLOCK_K);
    static constexpr uint32_t SMEM_SCALES_B_SIZE
        = ceil_div<uint32_t>(kNumMathWarpGroups * SHAPE_K_SCALES * sizeof(float), sizeof(Barrier)) * sizeof(Barrier);

    // Configs
    constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K;
    constexpr uint32_t kNumThreads = get_num_threads_per_sm_2wg<kNumTMAThreads, kNumMathThreadsPerGroup>();
    constexpr uint32_t kNumMathThreads = kNumThreads - kNumTMAThreads;
    constexpr uint32_t kNumIterations = ceil_div(SHAPE_K, kFullKOfAllStages);
    uint32_t const warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t const lane_idx = get_lane_id();

    // Prefetch TMA descriptors at very beginning
    if (threadIdx.x == kNumMathThreads)
    {
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_scales_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_d));
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(SMEM_B_SIZE_PER_WG % 1024 == 0, "The up half of a B stage must start on a swizzle atom");

    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    __nv_fp8_e4m3* smem_a[kNumStages];
    __nv_fp8_e4m3* smem_b[kNumStages];
    float* smem_scales_a[kNumStages];
    float* smem_scales_b;

    // TMA Barrier for both divisible and non-divisible cases
    Barrier* full_barriers[kNumStages];
    Barrier* empty_barriers[kNumStages];

// Fill shared memory pointers
#pragma unroll
    for (int i = 0; i < kNumStages; ++i)
    {
        smem_a[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
        smem_b[i] = reinterpret_cast<__nv_fp8_e4m3*>(
            smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
        smem_scales_a[i] = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE
            + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) + i * SMEM_SCALES_A_SIZE_PER_STAGE);
    }
    smem_scales_b = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE
        + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE));

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_scales_b) + SMEM_SCALES_B_SIZE);
#pragma unroll
    for (int i = 0; i < kNumStages; ++i)
    {
        full_barriers[i] = barrier_start_ptr + i;
        empty_barriers[i] = barrier_start_ptr + kNumStages + i;
    }

    // Initialize barriers. Every stage is consumed by both math warp-groups: one arrival per math warp (8).
    if (threadIdx.x == kNumMathThreads)
    {
#pragma unroll
        for (int i = 0; i < kNumStages; ++i)
        {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_view_async_shared();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    __syncthreads();

    // For pipeline unrolling
    struct DivisibleK
    {
    };

    struct NotDivisibleK
    {
    };

    auto launch_k_iterations = [](auto const& func)
    {
        if constexpr (SHAPE_K % kFullKOfAllStages == 0)
        {
            for (int k_iter = 0; k_iter < kNumIterations; ++k_iter)
                func(k_iter, DivisibleK{});
        }
        else
        {
            for (int k_iter = 0; k_iter < kNumIterations - 1; ++k_iter)
                func(k_iter, DivisibleK{});
            func(kNumIterations - 1, NotDivisibleK{});
        }
    };

    // Register reconfigurations (see the note above the kernel: 168 compiled, 40 / 232 after setmaxnreg)
    constexpr int kNumTMARegisters = 40;
    constexpr int kNumMathRegisters = 232;

    // Block scheduler. SchedulerType enumerates SHAPE_N / (2 * BLOCK_N) N-blocks, one per SwiGLU column block b.
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = SchedulerType(problem_input);

    if (threadIdx.x >= kNumMathThreads)
    {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        if (threadIdx.x == kNumMathThreads)
        {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx))
            {
                launch_k_iterations(
                    [&](int k_iter, auto type)
                    {
                        constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                        constexpr int kNumInnerStages
                            = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                        DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

#pragma unroll
                        for (uint32_t s = 0; s < kNumInnerStages; ++s)
                        {
                            // Wait consumer release
                            empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);

                            // Issue TMA A and the A scales
                            auto& full_barrier = *full_barriers[s];
                            int k_idx = k_iter * kFullKOfAllStages + s * BLOCK_K;
                            tma_copy<kNumTMAMulticast>(&tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                smem_a[s], k_idx, scheduler.get_global_m_idx(m_block_idx));
                            tma_copy<kNumTMAMulticast>(&tensor_map_scales_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                smem_scales_a[s], m_block_idx * BLOCK_M,
                                scheduler.get_global_scales_a_idx(k_idx / BLOCK_K));

                            // Issue TMA B: the gate box (rows expert * 2I + b * 128) and the up box (I rows further)
                            // into adjacent shared memory
                            uint32_t const n_gate_idx
                                = scheduler.get_global_n_idx(SHAPE_N, BLOCK_N, n_block_idx, m_block_idx);
                            tma_copy(&tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier), smem_b[s], k_idx,
                                n_gate_idx);
                            tma_copy(&tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier),
                                smem_b[s] + SMEM_B_SIZE_PER_WG, k_idx, n_gate_idx + SHAPE_N_HALF);
                            full_barrier.arrive_and_expect_tx(
                                SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE);
                        }

// Wait unaligned cases
#pragma unroll
                        for (uint32_t s = kNumInnerStages; s < kNumStages; ++s)
                        {
                            empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);
                            full_barriers[s]->arrive();
                        }
                    });
            }
        }
    }
    else
    {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        auto const math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / kNumMathThreadsPerGroup, 0);
        // Both warp-groups own rows 0..63: index rows by the warp inside the warp-group (the row-index trap above)
        uint32_t const wg_warp_idx = warp_idx % (kNumMathThreadsPerGroup / 32);
        uint32_t const wg_thread_idx = threadIdx.x % kNumMathThreadsPerGroup;
        auto const r_0 = wg_warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8;

        // This warp-group's half: its D staging tile, its weight-scale row and its B box
        auto smem_d_wg = smem_d + math_wg_idx * (SMEM_D_SIZE_PER_WG / sizeof(__nv_bfloat16));
        auto smem_scales_b_wg = smem_scales_b + math_wg_idx * SHAPE_K_SCALES;
        uint32_t const smem_b_wg_offset = math_wg_idx * SMEM_B_SIZE_PER_WG;
        uint32_t const n_out_offset = math_wg_idx * SHAPE_N_HALF;

        // Each warp-group only touches its own scale row and D tile, so it synchronizes on its own named barrier
        // (user ids 1 and 2, i.e. hardware barriers 7 and 8); the two halves are coupled only through the stage
        // barriers. NOTES: the static sync, not a NamedBarrier object: the constructor's CUTLASS_ASSERT on a runtime
        // id compiles to an __assertfail call, and any call makes ptxas serialize every WGMMA (C7510).
        uint32_t const wg_barrier_id = 1 + math_wg_idx;
        auto const wg_barrier_sync = [&]() { cutlass::arch::NamedBarrier::sync(kNumMathThreadsPerGroup, wg_barrier_id); };

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            // Load this warp-group's weight-scale row (gate row b, or up row I/128 + b) with the warp-group's math
            // threads. NOTES: except the warp-group's first warp, whose thread 0 issued the previous TMA store
            if (wg_thread_idx >= 32)
            {
                auto num_previous_lines
                    = scheduler.get_global_scales_b_idx(ceil_div(SHAPE_N, BLOCK_K), 0, 0, m_block_idx);
                auto local_scales_b = scales_b
                    + (num_previous_lines + math_wg_idx * (SHAPE_N_HALF / BLOCK_K) + ((n_block_idx * BLOCK_N) / BLOCK_K))
                        * SHAPE_K_SCALES;
#pragma unroll
                for (uint32_t i = wg_thread_idx - 32; i < SHAPE_K_SCALES; i += kNumMathThreadsPerGroup - 32)
                    st_shared(smem_scales_b_wg + i, __ldg(local_scales_b + i));
            }
            wg_barrier_sync();

            // Accumulation for WGMMA or CUDA promotion
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum] = {0};

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](int s) { lane_idx == 0 ? empty_barriers[s]->arrive() : void(); };

            // Launch MMAs
            launch_k_iterations(
                [&](int k_iter, auto type)
                {
                    constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                    constexpr int kNumInnerStages
                        = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                    DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

#pragma unroll
                    for (int s = 0; s < kNumInnerStages; ++s)
                    {
                        // Read B scales
                        float scale_b_0 = ld_shared(smem_scales_b_wg + k_iter * kNumStages + s);

                        // Wait TMA arrivals
                        full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);

                        // Read A scales
                        // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled
                        // block polluting the results
                        auto scale_a_0 = ld_shared(smem_scales_a[s] + r_0),
                             scale_a_1 = ld_shared(smem_scales_a[s] + r_1);

// Commit WGMMA instructions: A at offset 0 for both warp-groups, B at this warp-group's 128-row half
#pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum; ++i)
                            warpgroup_fence_operand(accum[i]);
                        warpgroup_arrive();
#pragma unroll
                        for (int k = 0; k < BLOCK_K / WGMMA::K; ++k)
                        {
                            auto desc_a = make_smem_desc(smem_a[s] + k * WGMMA::K, 1);
                            auto desc_b = make_smem_desc(smem_b[s] + smem_b_wg_offset + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        warpgroup_commit_batch();
#pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum; ++i)
                            warpgroup_fence_operand(accum[i]);
                        warpgroup_wait<0>();

                        // Notify barrier arrival
                        empty_barrier_arrive(s);

                        // Promote with scales (BLOCK_N == BLOCK_K: one uniform weight scale per stage)
                        float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
#pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum / 4; ++i)
                        {
                            final_accum[i * 4 + 0] += scale_0_0 * accum[i * 4 + 0];
                            final_accum[i * 4 + 1] += scale_0_0 * accum[i * 4 + 1];
                            final_accum[i * 4 + 2] += scale_1_0 * accum[i * 4 + 2];
                            final_accum[i * 4 + 3] += scale_1_0 * accum[i * 4 + 3];
                        }
                    }

// Wait unaligned cases
#pragma unroll
                    for (uint32_t s = kNumInnerStages; s < kNumStages; ++s)
                    {
                        full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);
                        empty_barrier_arrive(s);
                    }
                });

            // Write back to this warp-group's shared-memory tile using STSM (rows indexed inside the warp-group)
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 8 == 0, "Invalid STSM x4 vectorization");
#pragma unroll
            for (auto i = 0; i < WGMMA::kNumAccum / 8; ++i)
            {
                SM90_U32x4_STSM_N<nv_bfloat162>::copy(
                    __float22bfloat162_rn({final_accum[i * 8 + 0], final_accum[i * 8 + 1]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 2], final_accum[i * 8 + 3]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 4], final_accum[i * 8 + 5]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 6], final_accum[i * 8 + 7]}),
                    smem_d_wg + (wg_warp_idx * 16 + lane_idx % 16) * BLOCK_N + i * 16 + 8 * (lane_idx / 16));
            }

            // TMA-store the warp-group's 64 x 128 tile to its native columns: gate at b * 128, up at I + b * 128
            cute::tma_store_fence();
            wg_barrier_sync();
            if (wg_thread_idx == 0)
            {
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_d_wg, n_out_offset + n_block_idx * BLOCK_N,
                    scheduler.get_global_m_idx(m_block_idx));
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }

            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

// Non-swap GroupedContiguous FC1 with the SwiGLU + 1x128 FP8 requantize fused into the epilogue (sm_90 fused-FC1
// step 2, 2026-09-30). Design: /data/bench-runs/sm90_swiglu_fusion_20260930/DESIGN.md, section 3.3 (option B).
//
// What it replaces. The non-swap MoE layer ran FC1 -> bf16 gu [P_max, 2I] -> silu_chunk_mul_quantize_1x128_sorted
// -> (dq [P_max, I] fp8, sd [I/128, align4(P_max)] fp32). This kernel writes dq and sd directly, so gu never exists
// and the SwiGLU launch disappears.
//
// Mainloop. Exactly fp8_gemm_kernel_2wg's (step 1): one CTA owns one 64-row activation tile and one 128-column SwiGLU
// block b; math warp-group 0 accumulates gate block b and warp-group 1 up block b with today's m64n128k32 WGMMA and
// promotion, so every fp32 accumulator equals the unfused FC1's bit for bit.
//
// Epilogue (per tile, both warp-groups, three 256-thread barriers):
//   1. Each warp-group rounds its accumulators to bf16 with __float22bfloat162_rn, exactly as the unfused FC1 does
//      before its store; the SwiGLU is computed on those bf16 values, as the unfused kernel reads them from gu.
//   2. The silu work is split by columns: warp-group 0 finishes columns 0..63, warp-group 1 columns 64..127. Warp-group
//      0 hands its gate columns 64..127 to warp-group 1 and warp-group 1 its up columns 0..63 to warp-group 0 through
//      shared memory, in fragment order (both warp-groups share the m64n128 accumulator layout, so the partner thread
//      with the same index holds the same rows and columns). Barrier A.
//   3. silu2_mul (copied from quant_kernels.cu, tanh.approx.bf16x2 included) on the bf16x2 pairs; the row amax over
//      this warp-group's 64 columns is local to a lane quad (two shuffles); the two half-row amaxes meet in shared
//      memory. Barrier B. amax = max(both halves), clamped at 1e-10f.
//   4. qs = 448.f / amax and dequant = amax * (1.f / 448.f) as the unfused kernel writes them; paired satfinite E4M3
//      converts (lower column in .x) into a 128B-swizzled 64 x 128 fp8 staging tile; dequant to
//      sd[b * sd_ld + row]. Barrier C, then one TMA store of the tile into dq [P_max, I].
// The staging tile is single-buffered: the issuing thread waits for the previous tile's store to finish reading it
// right before barrier B of the next tile (the whole mainloop in between), and once more before the CTA exits.
//
// Rows. The kernel writes every row of every 64-row block the scheduler visits, including an expert's intra-block
// padding rows (whose inputs the gather left uninitialised); the unfused kernel writes only routed rows. FC2 is
// row-independent and the combine reads only routed rows, so the layer output is unaffected.
//
// Registers: as fp8_gemm_kernel_2wg, 384 threads at one CTA per SM must compile to exactly 168 registers for the
// 40 / 232 setmaxnreg split; the host refuses any other count (dispatch.cuh).
namespace sm90_swiglu_smem
{
// Shared-memory layout of fp8_gemm_kernel_2wg_swiglu, in bytes from the dynamic shared-memory base. The kernel (NVRTC)
// and the host's stage pick (dispatch.cuh, nvcc) both read these, so the two cannot drift apart.
//   [fp8 dq staging 64 x 128, 128B-swizzled] [bf16 exchange: WG0 -> WG1, WG1 -> WG0]
//   [A x stages] [B gate | up x stages] [A scales x stages] [half-row amax, 2 x 64] [B scales gate | up] [barriers]
constexpr uint32_t kBlockM = 64, kBlockN = 128, kBlockK = 128;
constexpr uint32_t kDqStagingOffset = 0; // TMA store with the 128B swizzle: must be 1024-byte aligned
constexpr uint32_t kDqStagingBytes = kBlockM * kBlockN;
constexpr uint32_t kXchgOffset = kDqStagingOffset + kDqStagingBytes;
constexpr uint32_t kXchgBytesPerWG = kBlockM * (kBlockN / 2) * 2; // 64 rows x 64 bf16 columns
constexpr uint32_t kStagesOffset = kXchgOffset + 2 * kXchgBytesPerWG;
constexpr uint32_t kABytesPerStage = kBlockM * kBlockK;
constexpr uint32_t kBBytesPerWGPerStage = kBlockN * kBlockK;
constexpr uint32_t kBBytesPerStage = 2 * kBBytesPerWGPerStage;
constexpr uint32_t kScalesABytesPerStage = kBlockM * 4;
constexpr uint32_t kAmaxBytes = 2 * kBlockM * 4;
constexpr uint32_t kBarrierBytes = 8; // cutlass::arch::ClusterTransactionBarrier

__device__ __host__ constexpr uint32_t a_offset(uint32_t stage)
{
    return kStagesOffset + stage * kABytesPerStage;
}

__device__ __host__ constexpr uint32_t b_offset(uint32_t num_stages, uint32_t stage)
{
    return kStagesOffset + num_stages * kABytesPerStage + stage * kBBytesPerStage;
}

__device__ __host__ constexpr uint32_t scales_a_offset(uint32_t num_stages, uint32_t stage)
{
    return kStagesOffset + num_stages * (kABytesPerStage + kBBytesPerStage) + stage * kScalesABytesPerStage;
}

__device__ __host__ constexpr uint32_t amax_offset(uint32_t num_stages)
{
    return kStagesOffset + num_stages * (kABytesPerStage + kBBytesPerStage + kScalesABytesPerStage);
}

__device__ __host__ constexpr uint32_t scales_b_offset(uint32_t num_stages)
{
    return amax_offset(num_stages) + kAmaxBytes;
}

__device__ __host__ constexpr uint32_t scales_b_bytes(uint32_t shape_k)
{
    return (2 * ((shape_k + kBlockK - 1) / kBlockK) * 4 + kBarrierBytes - 1) / kBarrierBytes * kBarrierBytes;
}

__device__ __host__ constexpr uint32_t barriers_offset(uint32_t num_stages, uint32_t shape_k)
{
    return scales_b_offset(num_stages) + scales_b_bytes(shape_k);
}

__device__ __host__ constexpr uint32_t total_bytes(uint32_t num_stages, uint32_t shape_k)
{
    return barriers_offset(num_stages, shape_k) + 2 * num_stages * kBarrierBytes;
}
} // namespace sm90_swiglu_smem

namespace sm90_swiglu
{
// tanh2_approx and silu2_mul are copied verbatim from csrc/gemm/ops/quant_kernels.cu, whose
// silu_chunk_mul_quantize_1x128_fp32_sorted_kernel this epilogue must match bit for bit. Copy the instruction, not
// the math: two fp32 tanh.approx instead of one tanh.approx.bf16x2 differ by about one bf16 ULP and flip FP8 bytes
// and scales (measured on sm_120, harness section 15).
__device__ __forceinline__ __nv_bfloat162 tanh2_approx(__nv_bfloat162 x)
{
    uint32_t r;
    asm("tanh.approx.bf16x2 %0, %1;" : "=r"(r) : "r"(*reinterpret_cast<uint32_t const*>(&x)));
    return *reinterpret_cast<__nv_bfloat162 const*>(&r);
}

// h01 = silu(g01) * u01 for a bf16x2 pair: xh = g/2; h = (xh + xh*tanh(xh))*u.
__device__ __forceinline__ __nv_bfloat162 silu2_mul(__nv_bfloat162 g, __nv_bfloat162 u)
{
    __nv_bfloat162 const half2v = __float2bfloat162_rn(0.5f);
    __nv_bfloat162 const xh = __hmul2(g, half2v);
    __nv_bfloat162 const t = tanh2_approx(xh);
    return __hmul2(__hfma2(xh, t, xh), u);
}

// Shared-memory accesses of the epilogue, on 32-bit shared addresses. They are volatile asm like the barriers around
// them, so the compiler keeps their order relative to the bar.sync instructions.
__device__ __forceinline__ uint32_t smem_addr(void const* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void sts_v4(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(a), "r"(b), "r"(c), "r"(d));
}

__device__ __forceinline__ uint4 lds_v4(uint32_t addr)
{
    uint4 v;
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
    return v;
}

__device__ __forceinline__ void sts_f32(uint32_t addr, float v)
{
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(addr), "f"(v));
}

__device__ __forceinline__ float lds_f32(uint32_t addr)
{
    float v;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(addr));
    return v;
}

__device__ __forceinline__ void sts_u16(uint32_t addr, uint16_t v)
{
    asm volatile("st.shared.b16 [%0], %1;" ::"r"(addr), "h"(v));
}

__device__ __forceinline__ uint32_t bf162_bits(__nv_bfloat162 v)
{
    return *reinterpret_cast<uint32_t const*>(&v);
}

__device__ __forceinline__ __nv_bfloat162 bits_bf162(uint32_t v)
{
    return *reinterpret_cast<__nv_bfloat162 const*>(&v);
}

// Byte offset of (row, column) in the 64 x 128 fp8 staging tile under the TMA 128-byte swizzle: the 16-byte chunk
// index (column bits 4..6) is XORed with the row's position in its 8-row, 1024-byte atom. A warp's 16-bit stores of
// one (chunk, row half) then land in 16 distinct 4-byte words instead of eight rows on one bank.
__device__ __forceinline__ uint32_t dq_staging_offset(uint32_t row, uint32_t col)
{
    return row * 128u + ((((col >> 4) ^ (row & 7u)) << 4) | (col & 15u));
}
} // namespace sm90_swiglu

template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kNumGroups,
    uint32_t kNumStages, uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup, uint32_t kNumTMAMulticast,
    typename SchedulerType, typename InputType>
__global__ void __launch_bounds__(get_num_threads_per_sm_2wg<kNumTMAThreads, kNumMathThreadsPerGroup>(), 1)
    fp8_gemm_kernel_2wg_swiglu(float* gmem_sd, float* scales_b, InputType problem_input,
        __grid_constant__ const CUtensorMap tensor_map_a, __grid_constant__ const CUtensorMap tensor_map_b,
        __grid_constant__ const CUtensorMap tensor_map_scales_a, __grid_constant__ const CUtensorMap tensor_map_dq,
        uint32_t sd_ld)
{
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ == 900))
    namespace L = sm90_swiglu_smem;
    namespace F = sm90_swiglu;

    // Scaling and shape checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(BLOCK_N == BLOCK_K, "Each warp-group half must be exactly one 128-row weight-scale block");
    DG_STATIC_ASSERT(BLOCK_M == 64, "Both warp-groups own the same 64 rows (one WGMMA row block)");
    DG_STATIC_ASSERT(BLOCK_M == L::kBlockM and BLOCK_N == L::kBlockN and BLOCK_K == L::kBlockK,
        "The shared-memory layout is written for 64 x 128 x 128 tiles");
    DG_STATIC_ASSERT(SHAPE_N % (2 * BLOCK_N) == 0, "SHAPE_N must be 2 * I with I a multiple of BLOCK_N");
    DG_STATIC_ASSERT(kNumTMAMulticast == 1, "No TMA multicast: the per-block weights differ");
    DG_STATIC_ASSERT(SchedulerType::gemm_type == GemmType::GroupedContiguous, "Grouped-contiguous FC1 only");

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(WGMMA::M == BLOCK_M, "One WGMMA row block per warp-group");
    DG_STATIC_ASSERT(sizeof(Barrier) == L::kBarrierBytes, "Barrier size");

    // Shapes
    static constexpr uint32_t SHAPE_N_HALF = SHAPE_N / 2; // I: the first up row of an expert's [gate; up] weight
    static constexpr uint32_t SHAPE_K_SCALES = ceil_div(SHAPE_K, BLOCK_K);
    static constexpr uint32_t kTxBytesPerStage = L::kABytesPerStage + L::kBBytesPerStage + L::kScalesABytesPerStage;

    // Configs
    constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K;
    constexpr uint32_t kNumThreads = get_num_threads_per_sm_2wg<kNumTMAThreads, kNumMathThreadsPerGroup>();
    constexpr uint32_t kNumMathThreads = kNumThreads - kNumTMAThreads;
    constexpr uint32_t kNumIterations = ceil_div(SHAPE_K, kFullKOfAllStages);
    uint32_t const warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t const lane_idx = get_lane_id();

    // Prefetch TMA descriptors at very beginning
    if (threadIdx.x == kNumMathThreads)
    {
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_scales_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_dq));
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B (the A / B stages and the dq staging tile)
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(L::kDqStagingOffset % 1024 == 0, "The dq staging tile must start on a swizzle atom");
    DG_STATIC_ASSERT(L::kStagesOffset % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(L::kABytesPerStage % 1024 == 0 and L::kBBytesPerWGPerStage % 1024 == 0,
        "Every A stage and both B halves must start on a swizzle atom");

    // Data on shared memory
    uint8_t* const smem_dq = smem_buffer + L::kDqStagingOffset;
    uint8_t* const smem_xchg = smem_buffer + L::kXchgOffset;
    __nv_fp8_e4m3* smem_a[kNumStages];
    __nv_fp8_e4m3* smem_b[kNumStages];
    float* smem_scales_a[kNumStages];
    float* const smem_amax = reinterpret_cast<float*>(smem_buffer + L::amax_offset(kNumStages));
    float* const smem_scales_b = reinterpret_cast<float*>(smem_buffer + L::scales_b_offset(kNumStages));

    // TMA Barrier for both divisible and non-divisible cases
    Barrier* full_barriers[kNumStages];
    Barrier* empty_barriers[kNumStages];

// Fill shared memory pointers
#pragma unroll
    for (int i = 0; i < kNumStages; ++i)
    {
        smem_a[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + L::a_offset(i));
        smem_b[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + L::b_offset(kNumStages, i));
        smem_scales_a[i] = reinterpret_cast<float*>(smem_buffer + L::scales_a_offset(kNumStages, i));
    }

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + L::barriers_offset(kNumStages, SHAPE_K));
#pragma unroll
    for (int i = 0; i < kNumStages; ++i)
    {
        full_barriers[i] = barrier_start_ptr + i;
        empty_barriers[i] = barrier_start_ptr + kNumStages + i;
    }

    // Initialize barriers. Every stage is consumed by both math warp-groups: one arrival per math warp (8).
    if (threadIdx.x == kNumMathThreads)
    {
#pragma unroll
        for (int i = 0; i < kNumStages; ++i)
        {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_view_async_shared();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    __syncthreads();

    // For pipeline unrolling
    struct DivisibleK
    {
    };

    struct NotDivisibleK
    {
    };

    auto launch_k_iterations = [](auto const& func)
    {
        if constexpr (SHAPE_K % kFullKOfAllStages == 0)
        {
            for (int k_iter = 0; k_iter < kNumIterations; ++k_iter)
                func(k_iter, DivisibleK{});
        }
        else
        {
            for (int k_iter = 0; k_iter < kNumIterations - 1; ++k_iter)
                func(k_iter, DivisibleK{});
            func(kNumIterations - 1, NotDivisibleK{});
        }
    };

    // Register reconfigurations (see the note above fp8_gemm_kernel_2wg: 168 compiled, 40 / 232 after setmaxnreg)
    constexpr int kNumTMARegisters = 40;
    constexpr int kNumMathRegisters = 232;

    // Block scheduler. SchedulerType enumerates SHAPE_N / (2 * BLOCK_N) N-blocks, one per SwiGLU column block b.
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = SchedulerType(problem_input);

    if (threadIdx.x >= kNumMathThreads)
    {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        if (threadIdx.x == kNumMathThreads)
        {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx))
            {
                launch_k_iterations(
                    [&](int k_iter, auto type)
                    {
                        constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                        constexpr int kNumInnerStages
                            = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                        DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

#pragma unroll
                        for (uint32_t s = 0; s < kNumInnerStages; ++s)
                        {
                            // Wait consumer release
                            empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);

                            // Issue TMA A and the A scales
                            auto& full_barrier = *full_barriers[s];
                            int k_idx = k_iter * kFullKOfAllStages + s * BLOCK_K;
                            tma_copy<kNumTMAMulticast>(&tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                smem_a[s], k_idx, scheduler.get_global_m_idx(m_block_idx));
                            tma_copy<kNumTMAMulticast>(&tensor_map_scales_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                smem_scales_a[s], m_block_idx * BLOCK_M,
                                scheduler.get_global_scales_a_idx(k_idx / BLOCK_K));

                            // Issue TMA B: the gate box (rows expert * 2I + b * 128) and the up box (I rows further)
                            // into adjacent shared memory
                            uint32_t const n_gate_idx
                                = scheduler.get_global_n_idx(SHAPE_N, BLOCK_N, n_block_idx, m_block_idx);
                            tma_copy(&tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier), smem_b[s], k_idx,
                                n_gate_idx);
                            tma_copy(&tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier),
                                smem_b[s] + L::kBBytesPerWGPerStage, k_idx, n_gate_idx + SHAPE_N_HALF);
                            full_barrier.arrive_and_expect_tx(kTxBytesPerStage);
                        }

// Wait unaligned cases
#pragma unroll
                        for (uint32_t s = kNumInnerStages; s < kNumStages; ++s)
                        {
                            empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);
                            full_barriers[s]->arrive();
                        }
                    });
            }
        }
    }
    else
    {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        auto const math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / kNumMathThreadsPerGroup, 0);
        // Both warp-groups own rows 0..63: index rows by the warp inside the warp-group (fp8_gemm_kernel_2wg's
        // row-index trap)
        uint32_t const wg_warp_idx = warp_idx % (kNumMathThreadsPerGroup / 32);
        uint32_t const wg_thread_idx = threadIdx.x % kNumMathThreadsPerGroup;
        auto const r_0 = wg_warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8;
        uint32_t const shape_m = problem_input.shape_m;

        // This warp-group's weight-scale row and B box
        auto smem_scales_b_wg = smem_scales_b + math_wg_idx * SHAPE_K_SCALES;
        uint32_t const smem_b_wg_offset = math_wg_idx * L::kBBytesPerWGPerStage;

        // The per-warp-group barrier guards the warp-group's own scale row (user ids 1 and 2); the three epilogue
        // barriers span both math warp-groups (user id 3). NOTES: the static sync, not a NamedBarrier object: the
        // constructor's CUTLASS_ASSERT on a runtime id compiles to an __assertfail call, and any call makes ptxas
        // serialize every WGMMA (C7510).
        uint32_t const wg_barrier_id = 1 + math_wg_idx;
        auto const wg_barrier_sync = [&]() { cutlass::arch::NamedBarrier::sync(kNumMathThreadsPerGroup, wg_barrier_id); };
        auto const math_barrier_sync = []() { cutlass::arch::NamedBarrier::sync(kNumMathThreads, 3); };

        // Shared addresses of the epilogue buffers
        uint32_t const dq_staging_addr = F::smem_addr(smem_dq);
        uint32_t const xchg_addr = F::smem_addr(smem_xchg);
        uint32_t const amax_addr = F::smem_addr(smem_amax);

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            // Load this warp-group's weight-scale row (gate row b, or up row I/128 + b) with the warp-group's math
            // threads. NOTES: except the warp-group's first warp (the dq store issuer lives in warp-group 0's)
            if (wg_thread_idx >= 32)
            {
                auto num_previous_lines
                    = scheduler.get_global_scales_b_idx(ceil_div(SHAPE_N, BLOCK_K), 0, 0, m_block_idx);
                auto local_scales_b = scales_b
                    + (num_previous_lines + math_wg_idx * (SHAPE_N_HALF / BLOCK_K) + ((n_block_idx * BLOCK_N) / BLOCK_K))
                        * SHAPE_K_SCALES;
#pragma unroll
                for (uint32_t i = wg_thread_idx - 32; i < SHAPE_K_SCALES; i += kNumMathThreadsPerGroup - 32)
                    st_shared(smem_scales_b_wg + i, __ldg(local_scales_b + i));
            }
            wg_barrier_sync();

            // Accumulation for WGMMA or CUDA promotion
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum] = {0};

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](int s) { lane_idx == 0 ? empty_barriers[s]->arrive() : void(); };

            // Launch MMAs
            launch_k_iterations(
                [&](int k_iter, auto type)
                {
                    constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                    constexpr int kNumInnerStages
                        = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                    DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

#pragma unroll
                    for (int s = 0; s < kNumInnerStages; ++s)
                    {
                        // Read B scales
                        float scale_b_0 = ld_shared(smem_scales_b_wg + k_iter * kNumStages + s);

                        // Wait TMA arrivals
                        full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);

                        // Read A scales
                        // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled
                        // block polluting the results
                        auto scale_a_0 = ld_shared(smem_scales_a[s] + r_0),
                             scale_a_1 = ld_shared(smem_scales_a[s] + r_1);

// Commit WGMMA instructions: A at offset 0 for both warp-groups, B at this warp-group's 128-row half
#pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum; ++i)
                            warpgroup_fence_operand(accum[i]);
                        warpgroup_arrive();
#pragma unroll
                        for (int k = 0; k < BLOCK_K / WGMMA::K; ++k)
                        {
                            auto desc_a = make_smem_desc(smem_a[s] + k * WGMMA::K, 1);
                            auto desc_b = make_smem_desc(smem_b[s] + smem_b_wg_offset + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        warpgroup_commit_batch();
#pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum; ++i)
                            warpgroup_fence_operand(accum[i]);
                        warpgroup_wait<0>();

                        // Notify barrier arrival
                        empty_barrier_arrive(s);

                        // Promote with scales (BLOCK_N == BLOCK_K: one uniform weight scale per stage)
                        float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
#pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum / 4; ++i)
                        {
                            final_accum[i * 4 + 0] += scale_0_0 * accum[i * 4 + 0];
                            final_accum[i * 4 + 1] += scale_0_0 * accum[i * 4 + 1];
                            final_accum[i * 4 + 2] += scale_1_0 * accum[i * 4 + 2];
                            final_accum[i * 4 + 3] += scale_1_0 * accum[i * 4 + 3];
                        }
                    }

// Wait unaligned cases
#pragma unroll
                    for (uint32_t s = kNumInnerStages; s < kNumStages; ++s)
                    {
                        full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);
                        empty_barrier_arrive(s);
                    }
                });

            // Fused SwiGLU + 1x128 requantize epilogue. The warp-group index is a compile-time constant inside, so
            // every accumulator and fragment index stays static (a runtime index would put the arrays in local memory).
            uint32_t const m_global = scheduler.get_global_m_idx(m_block_idx);
            auto const epilogue = [&](auto wg_tag)
            {
                constexpr uint32_t kWG = decltype(wg_tag)::value; // 0: holds gate block b, 1: holds up block b
                constexpr uint32_t kChunks = WGMMA::kNumAccum / 4; // 16 n8 column chunks per thread
                constexpr uint32_t kHalf = kChunks / 2;            // 8 chunks = 64 columns
                constexpr uint32_t kOwnFirst = kWG * kHalf;        // WG0 finishes columns 0..63, WG1 64..127
                constexpr uint32_t kSendFirst = (1 - kWG) * kHalf;
                constexpr uint32_t kWords = kHalf * 2;             // bf16x2 words per thread and half: 8 chunks x 2 rows
                DG_STATIC_ASSERT(kChunks == 16 and kWords % 4 == 0, "m64n128 accumulator layout");

                // 1. Round to bf16 as the unfused FC1 does before its store. Word (j, s) holds row r_s, columns
                //    8j + 2 (lane % 4) + {0, 1}, the lower column in .x.
                uint32_t own[kWords], send[kWords];
#pragma unroll
                for (uint32_t jl = 0; jl < kHalf; ++jl)
                {
#pragma unroll
                    for (uint32_t s = 0; s < 2; ++s)
                    {
                        own[jl * 2 + s] = F::bf162_bits(__float22bfloat162_rn(
                            {final_accum[4 * (kOwnFirst + jl) + 2 * s], final_accum[4 * (kOwnFirst + jl) + 2 * s + 1]}));
                        send[jl * 2 + s] = F::bf162_bits(__float22bfloat162_rn({final_accum[4 * (kSendFirst + jl) + 2 * s],
                            final_accum[4 * (kSendFirst + jl) + 2 * s + 1]}));
                    }
                }

                // 2. Exchange the half the partner finishes, in fragment order: word w of thread t at
                //    ((w / 4) * 128 + t) * 16 bytes + (w % 4) * 4, so a warp's 16-byte stores cover 512 contiguous
                //    bytes, and the partner thread with the same t reads exactly its own fragment positions.
                uint32_t const xchg_mine = xchg_addr + kWG * L::kXchgBytesPerWG;
                uint32_t const xchg_partner = xchg_addr + (1 - kWG) * L::kXchgBytesPerWG;
#pragma unroll
                for (uint32_t g = 0; g < kWords / 4; ++g)
                    F::sts_v4(xchg_mine + (g * kNumMathThreadsPerGroup + wg_thread_idx) * 16, send[4 * g + 0],
                        send[4 * g + 1], send[4 * g + 2], send[4 * g + 3]);
                math_barrier_sync(); // A: both halves exchanged
                uint32_t partner[kWords];
#pragma unroll
                for (uint32_t g = 0; g < kWords / 4; ++g)
                {
                    uint4 const v = F::lds_v4(xchg_partner + (g * kNumMathThreadsPerGroup + wg_thread_idx) * 16);
                    partner[4 * g + 0] = v.x, partner[4 * g + 1] = v.y, partner[4 * g + 2] = v.z,
                                    partner[4 * g + 3] = v.w;
                }

                // 3. silu(gate) * up on this warp-group's 64 columns (WG0: own gate, partner up; WG1: partner gate,
                //    own up), and each row's |max| over them: 16 values per thread and row, then the lane quad.
                uint32_t h[kWords];
                float ax[2];
#pragma unroll
                for (uint32_t jl = 0; jl < kHalf; ++jl)
                {
#pragma unroll
                    for (uint32_t s = 0; s < 2; ++s)
                    {
                        uint32_t const w = jl * 2 + s;
                        __nv_bfloat162 hh;
                        if constexpr (kWG == 0)
                            hh = F::silu2_mul(F::bits_bf162(own[w]), F::bits_bf162(partner[w]));
                        else
                            hh = F::silu2_mul(F::bits_bf162(partner[w]), F::bits_bf162(own[w]));
                        h[w] = F::bf162_bits(hh);
                        float2 const f = __bfloat1622float2(hh);
                        float const m = fmaxf(fabsf(f.x), fabsf(f.y));
                        if (jl == 0)
                            ax[s] = m;
                        else
                            ax[s] = fmaxf(ax[s], m);
                    }
                }
#pragma unroll
                for (uint32_t s = 0; s < 2; ++s)
                {
                    ax[s] = fmaxf(ax[s], __shfl_xor_sync(0xffffffffu, ax[s], 1));
                    ax[s] = fmaxf(ax[s], __shfl_xor_sync(0xffffffffu, ax[s], 2));
                }

                // The two half-row amaxes meet in shared memory: [warp-group][row]
                if (lane_idx % 4 == 0)
                {
                    F::sts_f32(amax_addr + (kWG * BLOCK_M + r_0) * 4, ax[0]);
                    F::sts_f32(amax_addr + (kWG * BLOCK_M + r_1) * 4, ax[1]);
                }
                // The previous tile's dq store must have finished reading the staging tile before step 4 rewrites it
                if constexpr (kWG == 0)
                {
                    if (wg_thread_idx == 0)
                        cute::tma_store_wait<0>();
                }
                math_barrier_sync(); // B: half-row amaxes published, staging tile free

                // 4. The unfused kernel's requantize: amax over the 128 columns, clamped at 1e-10f;
                //    qs = 448 / amax and dequant = amax * (1 / 448), exactly as written there.
                float qs[2], dequant[2];
#pragma unroll
                for (uint32_t s = 0; s < 2; ++s)
                {
                    float amax
                        = fmaxf(ax[s], F::lds_f32(amax_addr + ((1 - kWG) * BLOCK_M + (s == 0 ? r_0 : r_1)) * 4));
                    amax = fmaxf(amax, 1e-10f);
                    qs[s] = 448.f / amax;
                    dequant[s] = amax * (1.f / 448.f);
                }
#pragma unroll
                for (uint32_t jl = 0; jl < kHalf; ++jl)
                {
#pragma unroll
                    for (uint32_t s = 0; s < 2; ++s)
                    {
                        float2 const f = __bfloat1622float2(F::bits_bf162(h[jl * 2 + s]));
                        __nv_fp8x2_storage_t const q
                            = __nv_cvt_float2_to_fp8x2(make_float2(f.x * qs[s], f.y * qs[s]), __NV_SATFINITE, __NV_E4M3);
                        uint32_t const row = (s == 0) ? r_0 : r_1;
                        uint32_t const col = 8 * (kOwnFirst + jl) + 2 * (lane_idx % 4);
                        F::sts_u16(dq_staging_addr + F::dq_staging_offset(row, col), static_cast<uint16_t>(q));
                    }
                }
                // One dequant scale per row: both warp-groups hold every row's full amax, so warp-group 0 writes the
                // rows r_0 and warp-group 1 the rows r_1 (sd [I/128, sd_ld], column-major per 128-column block).
                if (lane_idx % 4 == 0)
                {
                    uint32_t const row = (kWG == 0) ? r_0 : r_1;
                    if (m_global + row < shape_m)
                        gmem_sd[static_cast<uint64_t>(n_block_idx) * sd_ld + m_global + row] = dequant[kWG];
                }

                // Publish the staging tile to the TMA unit, then one thread stores it to dq [P_max, I]
                cute::tma_store_fence();
                math_barrier_sync(); // C: staging tile complete
                if constexpr (kWG == 0)
                {
                    if (wg_thread_idx == 0)
                    {
                        cute::SM90_TMA_STORE_2D::copy(&tensor_map_dq, smem_dq, n_block_idx * BLOCK_N, m_global);
                        cute::tma_store_arrive();
                    }
                }
            };
            if (math_wg_idx == 0)
                epilogue(std::integral_constant<uint32_t, 0>{});
            else
                epilogue(std::integral_constant<uint32_t, 1>{});

            __syncwarp();
        }

        // Drain the last tile's dq store before the CTA releases its shared memory
        if (threadIdx.x == 0)
            cute::tma_store_wait<0>();
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

// Swap-AB GroupedContiguous FC1 in which one CTA owns a whole 128-column SwiGLU block (sm_90 fused-FC1, subtask P2a,
// design S2 of /data/bench-runs/sm90_swiglu_fusion_20260930/DESIGN.md section 4.3; 2026-10-01).
//
// Why. The 1x128 requantize after SwiGLU takes one amax per (token, 128 SwiGLU columns), and SwiGLU column j needs
// weight row j (gate) and weight row I + j (up). For the amax to stay inside one CTA, the CTA must own all 128 columns
// of a block b, which is 256 weight rows. fp8_gemm_kernel_swapAB above owns 128 weight rows (one weight-scale block),
// so it cannot carry the fused epilogue.
//
// Mainloop (shared by both kernels below). The weight is the swap-AB A operand, stacked [gate; up] per expert as today
// ([G * 2I, K], no interleave). One CTA computes, for one BLOCK_N-row activation tile and one SwiGLU block b, gate
// block b and up block b. The producer fetches the gate box (weight rows expert * 2I + b * 128, 128 rows) and the up
// box (I rows further) with two TMA loads into adjacent shared memory, so one stage holds 256 weight rows. Math
// warp-group w owns weight rows 64w .. 64w+63 of both halves: per k32 step it issues one m64nBNk32 WGMMA on the gate
// half and one on the up half (two accumulator sets), so the gate and up values of one (j, token) sit at the same
// fragment position of the same thread. Each accumulator set runs today's instruction stream (the same WGMMA shape and
// k order, the same per-stage promotion `final += (scale_w * scale_act) * accum` with the gate or up weight scale), so
// every fp32 accumulator equals fp8_gemm_kernel_swapAB's for the same weight rows bit for bit.
//
// Epilogues.
//   fp8_gemm_kernel_swapAB_pair (kFused = false): rounds both accumulator sets to bf16 exactly as today and TMA-stores
//   the gate tile to columns b * 128 and the up tile to columns I + b * 128 of today's gu [P_max, 2I], so its output is
//   bit-identical to fp8_gemm_kernel_swapAB's on every row the scheduler computes. It is the mainloop's bit-exact test
//   vehicle (the swap-AB twin of fp8_gemm_kernel_2wg).
//   fp8_gemm_kernel_swapAB_swiglu (kFused = true): the SwiGLU + 1x128 FP8 requantize of
//   silu_chunk_mul_quantize_1x128_sorted_sm90, bit for bit, writing dq [P_max, I] fp8 and sd [I/128, sd_ld] fp32:
//     1. round gate and up to bf16 with __float22bfloat162_rn, as the unfused FC1 does before its store;
//     2. silu2_mul on bf16x2 pairs (copied from quant_kernels.cu, tanh.approx.bf16x2 included). A pair here holds two
//        tokens of one column j instead of two columns of one token; every bf16x2 operation works lane by lane, so the
//        values are the same;
//     3. per-token |max| over the thread's two rows, shuffles over the eight lanes that share lane % 4 (16 rows), then
//        the eight warps' partials meet in shared memory ([8][BLOCK_N] fp32); barrier A; amax over the 128 columns,
//        clamped at 1e-10f; qs = 448.f / amax and dequant = amax * (1.f / 448.f) exactly as the unfused kernel writes
//        them (max is exact, so the reduction order does not matter);
//     4. paired satfinite E4M3 converts (element by element, as the unfused kernel's paired converts), then one byte
//        exchange with the lane four apart (row j + 1) so that every lane stores one 16-bit (token, j, j + 1) pair into
//        a 128B-swizzled [BLOCK_N][128] fp8 staging tile; dequant goes to sd[b * sd_ld + row] from four lanes;
//     5. barrier B, then one TMA store of the staging tile into dq. The staging tile is single-buffered: the issuing
//        thread waits for the previous store right before barrier A of the next tile, and once more before exiting.
//   Rows: the fused kernel writes every row of every BLOCK_N-row tile the scheduler visits, including an expert's
//   padding rows (whose inputs the gather left uninitialised); the unfused SwiGLU kernel writes only routed rows. FC2 is
//   row-independent and the combine reads only routed rows, so the layer output is unaffected.
//
// Scheduler. GroupedContiguousSchedulerSwapAB with kNumMBlocks = I / 128 (passed explicitly by the JIT source), so it
// enumerates one weight block per SwiGLU block b; the -1 length gate on grouped_layout is unchanged.
//
// Registers. 384 threads as today's swap-AB kernel (one TMA and two math warp-groups). One CTA per SM compiles to 168
// registers (setmaxnreg 40 / 232); two CTAs per SM (FSO_SWAPAB_CTAS_PER_SM=2) cap the kernel at 80 registers and
// rebalance to 40 / 96, which deadlocks unless ptxas compiled exactly 80, so the host refuses any other count (and any
// build that spills).
namespace sm90_swapab_pair_smem
{
// Shared-memory layout of fp8_gemm_kernel_swapAB_pair (fused = false) and fp8_gemm_kernel_swapAB_swiglu (fused = true),
// in bytes from the dynamic shared-memory base. The kernel (NVRTC) and the host's stage pick (dispatch.cuh, nvcc) both
// read these functions, so the two cannot drift apart.
//   [epilogue region, padded to 1 KB] [A: gate box | up box  x stages] [B (activation) x stages]
//   [activation scales x stages, 128 B each] [full barriers x stages] [empty barriers x stages]
// Epilogue region: pair kernel = bf16 gate tile [BLOCK_N][128] + bf16 up tile [BLOCK_N][128] (the TMA store sources);
// fused kernel = fp8 dq staging tile [BLOCK_N][128] (128B-swizzled, at offset 0) + per-warp per-token amax partials
// [8][BLOCK_N] fp32.
constexpr uint32_t kBlockM = 128; // weight rows per TMA box (one weight-scale block, one gate or up half)
constexpr uint32_t kBlockK = 128;
constexpr uint32_t kNumMathWarps = 8;
constexpr uint32_t kHalfBytesPerStage = kBlockM * kBlockK; // one 128-row weight box: 16 KB
constexpr uint32_t kABytesPerStage = 2 * kHalfBytesPerStage; // gate box + up box: 32 KB
constexpr uint32_t kBarrierBytes = 8; // cutlass::arch::ClusterTransactionBarrier

__device__ __host__ constexpr uint32_t align_up(uint32_t x, uint32_t a)
{
    return (x + a - 1) / a * a;
}

__device__ __host__ constexpr uint32_t d_tile_bytes(uint32_t block_n) // one bf16 [block_n][128] tile
{
    return block_n * kBlockM * 2;
}

__device__ __host__ constexpr uint32_t dq_tile_bytes(uint32_t block_n) // the fp8 [block_n][128] staging tile
{
    return block_n * kBlockM;
}

__device__ __host__ constexpr uint32_t amax_offset(uint32_t block_n) // fused only: [8 warps][block_n] fp32 partials
{
    return align_up(dq_tile_bytes(block_n), 128);
}

__device__ __host__ constexpr uint32_t epilogue_bytes(uint32_t block_n, bool fused)
{
    return fused ? align_up(amax_offset(block_n) + kNumMathWarps * block_n * 4, 1024)
                 : align_up(2 * d_tile_bytes(block_n), 1024);
}

__device__ __host__ constexpr uint32_t b_bytes_per_stage(uint32_t block_n)
{
    return block_n * kBlockK;
}

__device__ __host__ constexpr uint32_t scales_b_bytes_per_stage(uint32_t block_n) // TMA destination, 128 B aligned
{
    return align_up(block_n * 4, 128);
}

__device__ __host__ constexpr uint32_t a_offset(uint32_t block_n, bool fused, uint32_t stage)
{
    return epilogue_bytes(block_n, fused) + stage * kABytesPerStage;
}

__device__ __host__ constexpr uint32_t b_offset(uint32_t block_n, bool fused, uint32_t num_stages, uint32_t stage)
{
    return epilogue_bytes(block_n, fused) + num_stages * kABytesPerStage + stage * b_bytes_per_stage(block_n);
}

__device__ __host__ constexpr uint32_t scales_b_offset(uint32_t block_n, bool fused, uint32_t num_stages, uint32_t stage)
{
    return epilogue_bytes(block_n, fused) + num_stages * (kABytesPerStage + b_bytes_per_stage(block_n))
        + stage * scales_b_bytes_per_stage(block_n);
}

__device__ __host__ constexpr uint32_t barriers_offset(uint32_t block_n, bool fused, uint32_t num_stages)
{
    return scales_b_offset(block_n, fused, num_stages, num_stages);
}

__device__ __host__ constexpr uint32_t total_bytes(uint32_t block_n, bool fused, uint32_t num_stages)
{
    return barriers_offset(block_n, fused, num_stages) + 2 * num_stages * kBarrierBytes;
}
} // namespace sm90_swapab_pair_smem

namespace sm90_swapab_swiglu
{
// 64-bit shared accesses of the fused epilogue on 32-bit shared addresses; volatile asm like the barriers around them,
// so the compiler keeps their order relative to the bar.sync instructions (sm90_swiglu holds the other accessors).
__device__ __forceinline__ void sts_v2_f32(uint32_t addr, float a, float b)
{
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};" ::"r"(addr), "f"(a), "f"(b));
}

__device__ __forceinline__ float2 lds_v2_f32(uint32_t addr)
{
    float2 v;
    asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(addr));
    return v;
}
} // namespace sm90_swapab_swiglu

// The body both S2 kernels run. kFused selects the epilogue; everything before it is one code path.
template <uint32_t SHAPE_M, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kNumGroups,
    uint32_t kNumStages, uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup, bool kFused, typename SchedulerType,
    typename InputType>
__device__ __forceinline__ void swapab_gate_up_body(void* gmem_out, uint32_t sd_ld, float* scales_a,
    InputType& problem_input, CUtensorMap const* tensor_map_a, CUtensorMap const* tensor_map_b,
    CUtensorMap const* tensor_map_scales_b, CUtensorMap const* tensor_map_out)
{
    namespace L = sm90_swapab_pair_smem;

    // Scaling and shape checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(BLOCK_M == L::kBlockM and BLOCK_K == L::kBlockK, "The layout is written for 128-row boxes");
    DG_STATIC_ASSERT(SHAPE_M % (2 * BLOCK_M) == 0, "SHAPE_M must be 2 * I with I a multiple of 128");
    DG_STATIC_ASSERT(BLOCK_N == 16 or BLOCK_N == 32 or BLOCK_N == 64, "Activation tile of 16, 32 or 64 rows");
    DG_STATIC_ASSERT(SchedulerType::gemm_type == GemmType::GroupedContiguous, "Grouped-contiguous FC1 only");

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(2 * WGMMA::M == BLOCK_M, "Two math warp-groups of 64 weight rows cover one 128-row half");
    DG_STATIC_ASSERT(WGMMA::kNumAccum % 8 == 0, "Invalid STSM x4 vectorization");
    DG_STATIC_ASSERT(sizeof(Barrier) == L::kBarrierBytes, "Barrier size");

    // Shapes
    constexpr uint32_t SHAPE_M_HALF = SHAPE_M / 2; // I: the first up row of an expert's [gate; up] weight
    constexpr uint32_t SHAPE_K_SCALES = ceil_div(SHAPE_K, BLOCK_K);
    constexpr uint32_t kTxBytesPerStage = L::kABytesPerStage + L::b_bytes_per_stage(BLOCK_N) + BLOCK_N * sizeof(float);
    constexpr uint32_t SMEM_D_TILE_ELEMS = BLOCK_N * BLOCK_M; // pair kernel: one bf16 [BLOCK_N][128] output tile

    // Configs
    constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K;
    constexpr uint32_t kNumThreads = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
    constexpr uint32_t kNumMathThreads = kNumThreads - kNumTMAThreads;
    DG_STATIC_ASSERT(kNumMathThreads == L::kNumMathWarps * 32, "Two math warp-groups");
    constexpr uint32_t kNumIterations = ceil_div(SHAPE_K, kFullKOfAllStages);
    uint32_t const warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t const lane_idx = get_lane_id();

    // Prefetch TMA descriptors at very beginning
    if (threadIdx.x == kNumMathThreads)
    {
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(tensor_map_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(tensor_map_scales_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(tensor_map_out));
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B. With two resident CTAs per SM the allocation base is 1024-aligned only if
    // the host rounded the dynamic size (dispatch.cuh); trap rather than compute on a mis-phased swizzle. NOTES: a bare
    // trap, not DG_DEVICE_ASSERT, whose printf is a call (CALL.ABS).
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(L::epilogue_bytes(BLOCK_N, kFused) % 1024 == 0, "The stages must start on a swizzle atom");
    DG_STATIC_ASSERT(L::kHalfBytesPerStage % 1024 == 0 and L::b_bytes_per_stage(BLOCK_N) % 1024 == 0,
        "Every A half and every B stage must start on a swizzle atom");
    if (threadIdx.x == 0 and (static_cast<uint32_t>(__cvta_generic_to_shared(smem_buffer)) & 1023u) != 0u)
        asm volatile("trap;");

    // Data on shared memory
    __nv_fp8_e4m3* smem_a[kNumStages]; // weight: gate box, then up box
    __nv_fp8_e4m3* smem_b[kNumStages]; // activation
    float* smem_scales_b[kNumStages];  // activation scales

    // TMA Barrier for both divisible and non-divisible cases
    Barrier* full_barriers[kNumStages];
    Barrier* empty_barriers[kNumStages];

// Fill shared memory pointers
#pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i)
    {
        smem_a[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + L::a_offset(BLOCK_N, kFused, i));
        smem_b[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + L::b_offset(BLOCK_N, kFused, kNumStages, i));
        smem_scales_b[i] = reinterpret_cast<float*>(smem_buffer + L::scales_b_offset(BLOCK_N, kFused, kNumStages, i));
    }

    // Fill barriers
    auto barrier_start_ptr
        = reinterpret_cast<Barrier*>(smem_buffer + L::barriers_offset(BLOCK_N, kFused, kNumStages));
#pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i)
    {
        full_barriers[i] = barrier_start_ptr + i;
        empty_barriers[i] = barrier_start_ptr + kNumStages + i;
    }

    // Initialize barriers. Every stage is consumed by both math warp-groups: one arrival per math warp (8).
    if (threadIdx.x == kNumMathThreads)
    {
#pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i)
        {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_view_async_shared();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    __syncthreads();

    // For pipeline unrolling
    struct DivisibleK
    {
    };

    struct NotDivisibleK
    {
    };

    auto launch_k_iterations = [](auto const& func)
    {
        if constexpr (SHAPE_K % kFullKOfAllStages == 0)
        {
            for (int k_iter = 0; k_iter < kNumIterations; ++k_iter)
                func(k_iter, DivisibleK{});
        }
        else
        {
            for (int k_iter = 0; k_iter < kNumIterations - 1; ++k_iter)
                func(k_iter, DivisibleK{});
            func(kNumIterations - 1, NotDivisibleK{});
        }
    };

    // Register reconfigurations: as fp8_gemm_kernel_swapAB (40 / 232 at one CTA per SM, 40 / 96 at two, the latter
    // only around a compiled count of exactly 80, which the host checks).
    constexpr int kNumTMARegisters = 40;
    constexpr int kNumMathRegisters = (FSO_SWAPAB_CTAS_PER_SM >= 2) ? 96 : 232;

    // Block scheduler. m_block_idx is the SwiGLU block b (SchedulerType enumerates I / 128 of them).
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = SchedulerType(problem_input);

    if (threadIdx.x >= kNumMathThreads)
    {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        if (threadIdx.x == kNumMathThreads)
        {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx))
            {
                launch_k_iterations(
                    [&](int k_iter, auto type)
                    {
                        constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                        constexpr int kNumInnerStages
                            = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                        DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

#pragma unroll
                        for (uint32_t s = 0; s < kNumInnerStages; ++s)
                        {
                            // Wait consumer release
                            empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);

                            // Issue TMA A: the gate box (weight rows expert * 2I + b * 128) and the up box (I rows
                            // further) into adjacent shared memory
                            auto& full_barrier = *full_barriers[s];
                            int k_idx = k_iter * kFullKOfAllStages + s * BLOCK_K;
                            uint32_t const gate_row
                                = scheduler.get_global_m_idx(SHAPE_M, BLOCK_M, m_block_idx, n_block_idx);
                            tma_copy(tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier), smem_a[s], k_idx,
                                gate_row);
                            tma_copy(tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                smem_a[s] + L::kHalfBytesPerStage, k_idx, gate_row + SHAPE_M_HALF);

                            // Issue TMA B (activation) and its scales
                            tma_copy(tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier), smem_b[s], k_idx,
                                scheduler.get_global_n_idx(n_block_idx));
                            tma_copy(tensor_map_scales_b, reinterpret_cast<uint64_t*>(&full_barrier),
                                smem_scales_b[s], n_block_idx * BLOCK_N,
                                scheduler.get_global_scales_b_idx(k_idx / BLOCK_K));

                            full_barrier.arrive_and_expect_tx(kTxBytesPerStage);
                        }

// Wait unaligned cases
#pragma unroll
                        for (uint32_t s = kNumInnerStages; s < kNumStages; ++s)
                        {
                            empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);
                            full_barriers[s]->arrive();
                        }
                    });
            }
        }
    }
    else
    {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        auto const math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / kNumMathThreadsPerGroup, 0);

        // Each thread loads consecutive 2 activation scales
        uint32_t const scale_offset = (lane_idx % 4) * 2;

        // This warp-group's 64 weight rows inside each 128-row half of a stage (gate rows 64w.., up rows 64w..)
        uint32_t const smem_a_wg_offset = math_wg_idx * WGMMA::M * BLOCK_K;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            // Weight scales of gate block b (row b of the expert's 2I/128 scale rows) and up block b (row I/128 + b),
            // read straight from global memory per stage like fp8_gemm_kernel_swapAB's grouped-contiguous path
            auto const num_previous_lines
                = scheduler.get_global_scales_a_idx(ceil_div(SHAPE_M, BLOCK_K), 0, 0, n_block_idx);
            float const* const local_scales_gate
                = scales_a + (num_previous_lines + ((m_block_idx * BLOCK_M) / BLOCK_K)) * SHAPE_K_SCALES;
            float const* const local_scales_up = local_scales_gate + (SHAPE_M_HALF / BLOCK_K) * SHAPE_K_SCALES;

            // Accumulation for WGMMA or CUDA promotion: one set for the gate half, one for the up half
            float accum_g[WGMMA::kNumAccum], final_g[WGMMA::kNumAccum] = {0};
            float accum_u[WGMMA::kNumAccum], final_u[WGMMA::kNumAccum] = {0};

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](int s) { lane_idx == 0 ? empty_barriers[s]->arrive() : void(); };

            // Launch MMAs
            launch_k_iterations(
                [&](int k_iter, auto type)
                {
                    constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                    constexpr int kNumInnerStages
                        = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                    DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

#pragma unroll
                    for (int s = 0; s < kNumInnerStages; ++s)
                    {
                        // Read the gate and up weight scales of this k block
                        float const scale_a_g = __ldg(local_scales_gate + k_iter * kNumStages + s);
                        float const scale_a_u = __ldg(local_scales_up + k_iter * kNumStages + s);

                        // Wait TMA arrivals
                        full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);

                        // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled
                        // block polluting the results. Each thread reads consecutive two activation scales per n8
                        // chunk and forms the gate and up products exactly as fp8_gemm_kernel_swapAB forms its one.
                        float scale_g_0[WGMMA::kNumAccum / 4], scale_g_1[WGMMA::kNumAccum / 4];
                        float scale_u_0[WGMMA::kNumAccum / 4], scale_u_1[WGMMA::kNumAccum / 4];
#pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum / 4; ++i)
                        {
                            float2 scale_b
                                = ld_shared(reinterpret_cast<const float2*>(smem_scales_b[s] + i * 8 + scale_offset));
                            scale_g_0[i] = scale_a_g * scale_b.x;
                            scale_g_1[i] = scale_a_g * scale_b.y;
                            scale_u_0[i] = scale_a_u * scale_b.x;
                            scale_u_1[i] = scale_a_u * scale_b.y;
                        }

// Commit WGMMA instructions: four on the gate half, then four on the up half, one batch
#pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum; ++i)
                        {
                            warpgroup_fence_operand(accum_g[i]);
                            warpgroup_fence_operand(accum_u[i]);
                        }
                        warpgroup_arrive();
#pragma unroll
                        for (int k = 0; k < BLOCK_K / WGMMA::K; ++k)
                        {
                            auto desc_a = make_smem_desc(smem_a[s] + smem_a_wg_offset + k * WGMMA::K, 1);
                            auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum_g, k);
                        }
#pragma unroll
                        for (int k = 0; k < BLOCK_K / WGMMA::K; ++k)
                        {
                            auto desc_a = make_smem_desc(
                                smem_a[s] + L::kHalfBytesPerStage + smem_a_wg_offset + k * WGMMA::K, 1);
                            auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum_u, k);
                        }
                        warpgroup_commit_batch();
#pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum; ++i)
                        {
                            warpgroup_fence_operand(accum_g[i]);
                            warpgroup_fence_operand(accum_u[i]);
                        }
                        warpgroup_wait<0>();

                        // Notify barrier arrival
                        empty_barrier_arrive(s);

// Promote with scales (the expression of fp8_gemm_kernel_swapAB, once per half)
#pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum / 4; ++i)
                        {
                            final_g[i * 4 + 0] += scale_g_0[i] * accum_g[i * 4 + 0];
                            final_g[i * 4 + 1] += scale_g_1[i] * accum_g[i * 4 + 1];
                            final_g[i * 4 + 2] += scale_g_0[i] * accum_g[i * 4 + 2];
                            final_g[i * 4 + 3] += scale_g_1[i] * accum_g[i * 4 + 3];
                            final_u[i * 4 + 0] += scale_u_0[i] * accum_u[i * 4 + 0];
                            final_u[i * 4 + 1] += scale_u_1[i] * accum_u[i * 4 + 1];
                            final_u[i * 4 + 2] += scale_u_0[i] * accum_u[i * 4 + 2];
                            final_u[i * 4 + 3] += scale_u_1[i] * accum_u[i * 4 + 3];
                        }
                    }

// Wait unaligned cases
#pragma unroll
                    for (uint32_t s = kNumInnerStages; s < kNumStages; ++s)
                    {
                        full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);
                        empty_barrier_arrive(s);
                    }
                });

            if constexpr (!kFused)
            {
                // bf16 epilogue. The output tiles are single-buffered: the previous tile's two TMA stores must have
                // finished reading them before any math warp overwrites them (fp8_gemm_kernel_swapAB's reg-scales
                // placement, which gives the store the whole mainloop to drain).
                auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer); // gate tile, then up tile
                if (threadIdx.x == 0)
                    cute::tma_store_wait<0>();
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                // Write back to shared memory using STSM, with fp8_gemm_kernel_swapAB's addressing: warp_idx * 16 is
                // the warp's first weight row inside the 128-row half (warp-group 1's warps 4..7 hold rows 64..127),
                // which holds for the gate tile and the up tile alike
                int tid = 0;
                if (lane_idx < 8)
                {
                    tid = lane_idx * BLOCK_M;
                }
                else if (lane_idx < 16)
                {
                    tid = (lane_idx - 8) * BLOCK_M + 8;
                }
                else if (lane_idx < 24)
                {
                    tid = (lane_idx - 8) * BLOCK_M;
                }
                else
                {
                    tid = (lane_idx - 16) * BLOCK_M + 8;
                }
#pragma unroll
                for (auto i = 0; i < WGMMA::kNumAccum / 8; ++i)
                {
                    SM90_U32x4_STSM_T<nv_bfloat162>::copy(
                        __float22bfloat162_rn({final_g[i * 8 + 0], final_g[i * 8 + 1]}),
                        __float22bfloat162_rn({final_g[i * 8 + 2], final_g[i * 8 + 3]}),
                        __float22bfloat162_rn({final_g[i * 8 + 4], final_g[i * 8 + 5]}),
                        __float22bfloat162_rn({final_g[i * 8 + 6], final_g[i * 8 + 7]}),
                        smem_d + warp_idx * 16 + i * 16 * BLOCK_M + tid);
                    SM90_U32x4_STSM_T<nv_bfloat162>::copy(
                        __float22bfloat162_rn({final_u[i * 8 + 0], final_u[i * 8 + 1]}),
                        __float22bfloat162_rn({final_u[i * 8 + 2], final_u[i * 8 + 3]}),
                        __float22bfloat162_rn({final_u[i * 8 + 4], final_u[i * 8 + 5]}),
                        __float22bfloat162_rn({final_u[i * 8 + 6], final_u[i * 8 + 7]}),
                        smem_d + SMEM_D_TILE_ELEMS + warp_idx * 16 + i * 16 * BLOCK_M + tid);
                }

                // TMA-store the gate tile to columns b * 128 and the up tile to columns I + b * 128 of gu [P_max, 2I]
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);
                if (threadIdx.x == 0)
                {
                    uint32_t const n_global_idx = scheduler.get_global_n_idx(n_block_idx);
                    cute::SM90_TMA_STORE_2D::copy(tensor_map_out, smem_d, m_block_idx * BLOCK_M, n_global_idx);
                    cute::SM90_TMA_STORE_2D::copy(tensor_map_out, smem_d + SMEM_D_TILE_ELEMS,
                        SHAPE_M_HALF + m_block_idx * BLOCK_M, n_global_idx);
                    cute::tma_store_arrive();
                }
            }
            else
            {
                // Fused SwiGLU + 1x128 FP8 requantize epilogue (see the comment above sm90_swapab_pair_smem).
                namespace F = sm90_swiglu;
                namespace G = sm90_swapab_swiglu;
                constexpr uint32_t kChunks = WGMMA::kNumAccum / 4; // n8 token chunks per thread: BLOCK_N / 8
                uint8_t* const smem_dq = smem_buffer;              // the 128B-swizzled staging tile, offset 0
                uint32_t const dq_addr = F::smem_addr(smem_dq);
                uint32_t const amax_addr = F::smem_addr(smem_buffer + L::amax_offset(BLOCK_N));
                // The thread's SwiGLU columns j_0 and j_0 + 8 inside block b (rows of the two 128-row halves) and its
                // tokens 8 i + tok_lo + {0, 1} of chunk i (the m64nNk32 accumulator layout)
                uint32_t const j_0 = math_wg_idx * WGMMA::M + (warp_idx % 4) * 16 + lane_idx / 4;
                uint32_t const tok_lo = 2 * (lane_idx % 4);
                bool const odd_row = ((lane_idx / 4) & 1u) != 0u; // j_0 odd: the partner four lanes away has j_0 - 1

                // 1. Round to bf16 as the unfused FC1 does before its store; 2. SwiGLU on (token, token + 1) pairs of
                //    one column; 3. each token's |max| over the thread's two columns
                uint32_t h[kChunks][2];
                float ax[kChunks][2];
#pragma unroll
                for (uint32_t i = 0; i < kChunks; ++i)
                {
#pragma unroll
                    for (uint32_t s = 0; s < 2; ++s)
                    {
                        __nv_bfloat162 const hh = F::silu2_mul(
                            __float22bfloat162_rn({final_g[4 * i + 2 * s], final_g[4 * i + 2 * s + 1]}),
                            __float22bfloat162_rn({final_u[4 * i + 2 * s], final_u[4 * i + 2 * s + 1]}));
                        h[i][s] = F::bf162_bits(hh);
                        float2 const f = __bfloat1622float2(hh);
                        if (s == 0)
                        {
                            ax[i][0] = fabsf(f.x);
                            ax[i][1] = fabsf(f.y);
                        }
                        else
                        {
                            ax[i][0] = fmaxf(ax[i][0], fabsf(f.x));
                            ax[i][1] = fmaxf(ax[i][1], fabsf(f.y));
                        }
                    }
                }
                // ... then over the eight lanes with the same lane % 4 (the warp's 16 columns)
#pragma unroll
                for (uint32_t i = 0; i < kChunks; ++i)
                {
#pragma unroll
                    for (uint32_t t = 0; t < 2; ++t)
                    {
                        ax[i][t] = fmaxf(ax[i][t], __shfl_xor_sync(0xffffffffu, ax[i][t], 4));
                        ax[i][t] = fmaxf(ax[i][t], __shfl_xor_sync(0xffffffffu, ax[i][t], 8));
                        ax[i][t] = fmaxf(ax[i][t], __shfl_xor_sync(0xffffffffu, ax[i][t], 16));
                    }
                }
                // ... and the eight warps' partials meet in shared memory: [warp][token]
                if (lane_idx < 4)
                {
#pragma unroll
                    for (uint32_t i = 0; i < kChunks; ++i)
                        G::sts_v2_f32(amax_addr + (warp_idx * BLOCK_N + 8 * i + tok_lo) * 4, ax[i][0], ax[i][1]);
                }
                // The previous tile's dq store must have finished reading the staging tile before step 4 rewrites it
                if (threadIdx.x == 0)
                    cute::tma_store_wait<0>();
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0); // A: partials published, staging tile free

                // amax over the 128 columns, clamped at 1e-10f; qs = 448 / amax and dequant = amax * (1 / 448),
                // exactly as silu_chunk_mul_quantize_1x128_fp32_sorted_kernel writes them
                float qs[kChunks][2], dequant[kChunks][2];
#pragma unroll
                for (uint32_t i = 0; i < kChunks; ++i)
                {
                    float2 m = G::lds_v2_f32(amax_addr + (8 * i + tok_lo) * 4);
#pragma unroll
                    for (uint32_t w = 1; w < L::kNumMathWarps; ++w)
                    {
                        float2 const v = G::lds_v2_f32(amax_addr + (w * BLOCK_N + 8 * i + tok_lo) * 4);
                        m.x = fmaxf(m.x, v.x);
                        m.y = fmaxf(m.y, v.y);
                    }
                    float const amax_0 = fmaxf(m.x, 1e-10f), amax_1 = fmaxf(m.y, 1e-10f);
                    qs[i][0] = 448.f / amax_0;
                    qs[i][1] = 448.f / amax_1;
                    dequant[i][0] = amax_0 * (1.f / 448.f);
                    dequant[i][1] = amax_1 * (1.f / 448.f);
                }

                // 4. Requantize: one paired satfinite E4M3 convert per (chunk, column) gives (token, token + 1) of
                //    column j; one byte exchange with the lane four apart (column j + 1 or j - 1) turns it into a
                //    (token, j, j + 1) pair: the even-column lane keeps token 8i + tok_lo, the odd-column lane token
                //    8i + tok_lo + 1, and each stores 16 bits into the staging tile at its even column.
#pragma unroll
                for (uint32_t i = 0; i < kChunks; ++i)
                {
#pragma unroll
                    for (uint32_t s = 0; s < 2; ++s)
                    {
                        float2 const f = __bfloat1622float2(F::bits_bf162(h[i][s]));
                        uint32_t const q = static_cast<uint32_t>(__nv_cvt_float2_to_fp8x2(
                            make_float2(f.x * qs[i][0], f.y * qs[i][1]), __NV_SATFINITE, __NV_E4M3));
                        uint32_t const send = odd_row ? (q & 0xffu) : (q >> 8);
                        uint32_t const recv = __shfl_xor_sync(0xffffffffu, send, 4);
                        uint32_t const pair16 = odd_row ? (recv | (q & 0xff00u)) : ((q & 0xffu) | (recv << 8));
                        uint32_t const token = 8 * i + tok_lo + (odd_row ? 1u : 0u);
                        uint32_t const col = j_0 + 8 * s - (odd_row ? 1u : 0u);
                        F::sts_u16(dq_addr + F::dq_staging_offset(token, col), static_cast<uint16_t>(pair16));
                    }
                }

                // One dequant scale per token, sd [I/128, sd_ld] (column-major per 128-column block): every thread holds
                // the full amax of its tokens, so the first warp's four lane groups cover the BLOCK_N tokens
                uint32_t const n_global_idx = scheduler.get_global_n_idx(n_block_idx);
                if (warp_idx == 0 and lane_idx < 4)
                {
                    float* const sd_row = static_cast<float*>(gmem_out) + static_cast<uint64_t>(m_block_idx) * sd_ld
                        + n_global_idx + tok_lo;
#pragma unroll
                    for (uint32_t i = 0; i < kChunks; ++i)
                        *reinterpret_cast<float2*>(sd_row + 8 * i) = make_float2(dequant[i][0], dequant[i][1]);
                }

                // 5. Publish the staging tile to the TMA unit, then one thread stores it to dq [P_max, I]
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0); // B: staging tile complete
                if (threadIdx.x == 0)
                {
                    cute::SM90_TMA_STORE_2D::copy(tensor_map_out, smem_dq, m_block_idx * BLOCK_M, n_global_idx);
                    cute::tma_store_arrive();
                }
            }

            __syncwarp();
        }

        // Drain the last tile's TMA stores before the CTA releases its shared memory
        if (threadIdx.x == 0)
            cute::tma_store_wait<0>();
    }
}

template <uint32_t SHAPE_M, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kNumGroups,
    uint32_t kNumStages, uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup, uint32_t kNumTMAMulticast,
    typename SchedulerType, typename InputType>
__global__ void __launch_bounds__(
    get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M), FSO_SWAPAB_CTAS_PER_SM)
    fp8_gemm_kernel_swapAB_pair(__nv_bfloat16* gmem_d, float* scales_a, InputType problem_input,
        const __grid_constant__ CUtensorMap tensor_map_a,        // weight [G * 2I, K], one 128-row box
        const __grid_constant__ CUtensorMap tensor_map_b,        // activation [P_max, K], one BLOCK_N-row box
        const __grid_constant__ CUtensorMap tensor_map_scales_b, // activation scales
        const __grid_constant__ CUtensorMap tensor_map_d)        // gu [P_max, 2I] bf16, one BLOCK_N x 128 box
{
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ == 900))
    DG_STATIC_ASSERT(kNumTMAMulticast == 1, "No TMA multicast: the per-block weights differ");
    swapab_gate_up_body<SHAPE_M, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, kNumGroups, kNumStages, kNumTMAThreads,
        kNumMathThreadsPerGroup, false, SchedulerType>(gmem_d, 0u, scales_a, problem_input, &tensor_map_a,
        &tensor_map_b, &tensor_map_scales_b, &tensor_map_d);
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

template <uint32_t SHAPE_M, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kNumGroups,
    uint32_t kNumStages, uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup, uint32_t kNumTMAMulticast,
    typename SchedulerType, typename InputType>
__global__ void __launch_bounds__(
    get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M), FSO_SWAPAB_CTAS_PER_SM)
    fp8_gemm_kernel_swapAB_swiglu(float* gmem_sd, float* scales_a, InputType problem_input,
        const __grid_constant__ CUtensorMap tensor_map_a,        // weight [G * 2I, K], one 128-row box
        const __grid_constant__ CUtensorMap tensor_map_b,        // activation [P_max, K], one BLOCK_N-row box
        const __grid_constant__ CUtensorMap tensor_map_scales_b, // activation scales
        const __grid_constant__ CUtensorMap tensor_map_dq,       // dq [P_max, I] fp8, one BLOCK_N x 128 box, 128B swizzle
        uint32_t sd_ld)                                          // sd [I/128, sd_ld] fp32, sd_ld = align4(P_max)
{
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ == 900))
    DG_STATIC_ASSERT(kNumTMAMulticast == 1, "No TMA multicast: the per-block weights differ");
    swapab_gate_up_body<SHAPE_M, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, kNumGroups, kNumStages, kNumTMAThreads,
        kNumMathThreadsPerGroup, true, SchedulerType>(gmem_sd, sd_ld, scales_a, problem_input, &tensor_map_a,
        &tensor_map_b, &tensor_map_scales_b, &tensor_map_dq);
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// Fused swap-AB FC1 split across a two-CTA cluster (fp8_gemm_kernel_swapAB_swiglu_split; H2 Phase 2a, 2026-10-05).
//
// fp8_gemm_kernel_swapAB_swiglu gives one CTA a whole SwiGLU block: for one BLOCK_N-row activation tile, gate block b
// and up block b of an expert (256 weight rows). At the smallest token counts that is too few CTAs: at M = 1 the eight
// active experts make 8 * (I / 128) blocks, 48 CTAs on 132 SMs at I = 768 and 32 at I = 512, and each CTA's two math
// warp-groups work through 16 stages of eight WGMMAs each, one stage after the other. This kernel gives each SwiGLU
// block to the two CTAs of a cluster, without splitting K: CTA r of the cluster owns gate rows 64r .. 64r+63 and up
// rows 64r .. 64r+63 of block b, which is the work of math warp-group r of fp8_gemm_kernel_swapAB_swiglu, and it gives
// the gate rows and the up rows to two math warp-groups of their own (four WGMMAs per stage each). Every accumulator
// is computed as in that kernel (the same m64nBNk32 WGMMA on the same rows in the same k order, the same per-stage
// promotion), so the fp32 sums are that kernel's bit for bit.
//
// Epilogue. The up warp-group rounds its sums to bf16 as the unsplit kernel does and hands them to the gate warp-group
// through shared memory (the two warp-groups hold the same (row, token) fragment layout, so the hand-over is one 32-bit
// word per thread and value pair); the gate warp-group forms the SwiGLU of this CTA's 64 output columns. The 1x128
// requantize needs each token's |max| over all 128 columns of the block: each CTA reduces its 64 to one value per
// token, sends it to the peer CTA's shared memory (st.shared::cluster) and signals the peer's exchange barrier
// (mbarrier.arrive.release.cluster); the peer waits (acquire.cluster) and takes the max of the two values, so both
// CTAs hold the block's exact amax (max is exact, so the order does not matter), clamp it at 1e-10f and quantize their
// 64 columns with qs = 448/amax as the unsplit kernel does. Each CTA stores its [BLOCK_N][64] fp8 tile with one TMA
// store; CTA 0 writes the dequant scales. Output bytes and scales are those of fp8_gemm_kernel_swapAB_swiglu.
//
// Threads: two math warp-groups (gate, up) and one TMA warp-group, 384 in all, one CTA per SM, no register
// reconfiguration. Scheduling: the cluster, not the CTA, is the persistent unit; both CTAs of a cluster walk the same
// sequence of (activation tile, SwiGLU block) pairs (GroupedContiguousSchedulerSwapAB with the cluster index), so they
// always work on the same block. The exchange buffers are double-buffered by tile parity: a CTA can only write the
// peer's buffer for tile t + 2 after the peer has sent its value for tile t + 1, which it does after reading the buffer
// for tile t.
namespace sm90_swapab_split_smem
{
// Shared-memory layout, in bytes from the dynamic shared-memory base, read by the kernel (NVRTC) and the host's stage
// pick (dispatch.cuh) alike.
//   [epilogue, padded to 1 KB: fp8 dq staging tile [BLOCK_N][64] (no swizzle), per-warp amax partials [4][BLOCK_N],
//    the peer's amax values [2][BLOCK_N], the two exchange barriers, the up warp-group's bf16 pairs [BLOCK_N/4][128]]
//   [A: gate box 64 rows | up box 64 rows  x stages] [B (activation) x stages] [activation scales x stages, 128 B]
//   [full barriers x stages] [empty barriers x stages]
constexpr uint32_t kRowsPerCta = 64; // weight rows of each half per CTA (one WGMMA M)
constexpr uint32_t kOutCols = 64;    // SwiGLU output columns per CTA
constexpr uint32_t kBlockK = 128;
constexpr uint32_t kNumEpilogueWarps = 4; // the gate warp-group
constexpr uint32_t kNumMathWarps = 8;     // gate and up warp-groups
constexpr uint32_t kHalfBytesPerStage = kRowsPerCta * kBlockK; // one 64-row weight box: 8 KB
constexpr uint32_t kABytesPerStage = 2 * kHalfBytesPerStage;   // gate box + up box: 16 KB
constexpr uint32_t kBarrierBytes = 8;

__device__ __host__ constexpr uint32_t align_up(uint32_t x, uint32_t a)
{
    return (x + a - 1) / a * a;
}

__device__ __host__ constexpr uint32_t dq_tile_bytes(uint32_t block_n)
{
    return block_n * kOutCols;
}

__device__ __host__ constexpr uint32_t amax_offset(uint32_t block_n) // [4 warps][block_n] fp32
{
    return align_up(dq_tile_bytes(block_n), 128);
}

__device__ __host__ constexpr uint32_t peer_offset(uint32_t block_n) // [2 parities][block_n] fp32
{
    return amax_offset(block_n) + kNumEpilogueWarps * block_n * 4;
}

__device__ __host__ constexpr uint32_t xchg_barrier_offset(uint32_t block_n) // [2] mbarriers
{
    return align_up(peer_offset(block_n) + 2 * block_n * 4, 8);
}

__device__ __host__ constexpr uint32_t up_offset(uint32_t block_n) // [2 * block_n / 8][128 threads] bf16x2 words
{
    return align_up(xchg_barrier_offset(block_n) + 2 * kBarrierBytes, 16);
}

__device__ __host__ constexpr uint32_t epilogue_bytes(uint32_t block_n)
{
    return align_up(up_offset(block_n) + (2 * block_n / 8) * 128 * 4, 1024);
}

__device__ __host__ constexpr uint32_t b_bytes_per_stage(uint32_t block_n)
{
    return block_n * kBlockK;
}

__device__ __host__ constexpr uint32_t scales_b_bytes_per_stage(uint32_t block_n) // TMA destination, 128 B aligned
{
    return align_up(block_n * 4, 128);
}

__device__ __host__ constexpr uint32_t a_offset(uint32_t block_n, uint32_t stage)
{
    return epilogue_bytes(block_n) + stage * kABytesPerStage;
}

__device__ __host__ constexpr uint32_t b_offset(uint32_t block_n, uint32_t num_stages, uint32_t stage)
{
    return epilogue_bytes(block_n) + num_stages * kABytesPerStage + stage * b_bytes_per_stage(block_n);
}

__device__ __host__ constexpr uint32_t scales_b_offset(uint32_t block_n, uint32_t num_stages, uint32_t stage)
{
    return epilogue_bytes(block_n) + num_stages * (kABytesPerStage + b_bytes_per_stage(block_n))
        + stage * scales_b_bytes_per_stage(block_n);
}

__device__ __host__ constexpr uint32_t barriers_offset(uint32_t block_n, uint32_t num_stages)
{
    return scales_b_offset(block_n, num_stages, num_stages);
}

__device__ __host__ constexpr uint32_t total_bytes(uint32_t block_n, uint32_t num_stages)
{
    return barriers_offset(block_n, num_stages) + 2 * num_stages * kBarrierBytes;
}
} // namespace sm90_swapab_split_smem

namespace sm90_swapab_split
{
// Cluster-scope shared-memory operations of the amax exchange.
__device__ __forceinline__ uint32_t peer_addr(uint32_t local_smem_addr, uint32_t peer_rank)
{
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(local_smem_addr), "r"(peer_rank));
    return r;
}

__device__ __forceinline__ void st_cluster_v2_f32(uint32_t cluster_addr, float a, float b)
{
    asm volatile("st.shared::cluster.v2.f32 [%0], {%1, %2};" ::"r"(cluster_addr), "f"(a), "f"(b) : "memory");
}

__device__ __forceinline__ void arrive_cluster(uint32_t cluster_barrier_addr)
{
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(cluster_barrier_addr) : "memory");
}

__device__ __forceinline__ void wait_cluster(uint32_t barrier_addr, uint32_t parity)
{
    uint32_t done = 0;
    while (!done)
    {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
            "selp.u32 %0, 1, 0, p;\n"
            "}\n"
            : "=r"(done)
            : "r"(barrier_addr), "r"(parity)
            : "memory");
    }
}

__device__ __forceinline__ uint32_t cta_rank_in_cluster()
{
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void sts_u32(uint32_t addr, uint32_t v)
{
    asm volatile("st.shared.u32 [%0], %1;" ::"r"(addr), "r"(v));
}

__device__ __forceinline__ uint32_t lds_u32(uint32_t addr)
{
    uint32_t v;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr));
    return v;
}
} // namespace sm90_swapab_split

template <uint32_t SHAPE_M, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kNumGroups,
    uint32_t kNumStages, uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup, uint32_t kNumTMAMulticast,
    typename SchedulerType, typename InputType>
__global__ void __launch_bounds__(384, 1)
    fp8_gemm_kernel_swapAB_swiglu_split(float* gmem_sd, float* scales_a, InputType problem_input,
        const __grid_constant__ CUtensorMap tensor_map_a,        // weight [G * 2I, K], one 64-row box
        const __grid_constant__ CUtensorMap tensor_map_b,        // activation [P_max, K], one BLOCK_N-row box
        const __grid_constant__ CUtensorMap tensor_map_scales_b, // activation scales
        const __grid_constant__ CUtensorMap tensor_map_dq,       // dq [P_max, I] fp8, one BLOCK_N x 64 box, no swizzle
        uint32_t sd_ld)                                          // sd [I/128, sd_ld] fp32, sd_ld = align4(P_max)
{
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ == 900))
    namespace L = sm90_swapab_split_smem;
    namespace F = sm90_swiglu;
    namespace G = sm90_swapab_swiglu;
    namespace X = sm90_swapab_split;

    DG_STATIC_ASSERT(kNumTMAMulticast == 1, "No TMA multicast: the per-block weights differ");
    DG_STATIC_ASSERT(BLOCK_K == 128 and BLOCK_K == L::kBlockK, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(BLOCK_M == 128, "BLOCK_M is the SwiGLU block (128 output columns, two CTAs of 64)");
    DG_STATIC_ASSERT(SHAPE_M % (2 * BLOCK_M) == 0, "SHAPE_M must be 2 * I with I a multiple of 128");
    DG_STATIC_ASSERT(BLOCK_N == 16 or BLOCK_N == 32 or BLOCK_N == 64, "Activation tile of 16, 32 or 64 rows");
    DG_STATIC_ASSERT(SchedulerType::gemm_type == GemmType::GroupedContiguous, "Grouped-contiguous FC1 only");

    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(WGMMA::M == L::kRowsPerCta, "One math warp-group of 64 weight rows per half");
    DG_STATIC_ASSERT(sizeof(Barrier) == L::kBarrierBytes, "Barrier size");

    constexpr uint32_t SHAPE_M_HALF = SHAPE_M / 2; // I
    constexpr uint32_t SHAPE_K_SCALES = ceil_div(SHAPE_K, BLOCK_K);
    constexpr uint32_t kTxBytesPerStage = L::kABytesPerStage + L::b_bytes_per_stage(BLOCK_N) + BLOCK_N * sizeof(float);
    constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K;
    constexpr uint32_t kNumMathThreads = L::kNumMathWarps * 32; // 256: the gate and the up warp-group
    constexpr uint32_t kNumEpilogueThreads = L::kNumEpilogueWarps * 32;
    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreadsPerGroup == 128, "128-thread warp-groups");
    DG_STATIC_ASSERT(SHAPE_K % kFullKOfAllStages == 0, "The stage count divides K / 128 (sm90_swapab_split_stages)");
    constexpr uint32_t kNumIterations = SHAPE_K / kFullKOfAllStages;
    // Named barriers (user ids; NamedBarrier adds the reserved count): the gate warp-group's own, the up values being
    // published, and their having been read
    constexpr uint32_t kBarGate = 0, kBarUpReady = 1, kBarUpFree = 2;
    uint32_t const warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t const lane_idx = get_lane_id();
    uint32_t const cta_rank = X::cta_rank_in_cluster();
    uint32_t const cluster_idx = blockIdx.x / 2;
    uint32_t const num_clusters = gridDim.x / 2;

    if (threadIdx.x == kNumMathThreads)
    {
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_scales_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_dq));
    }
    __syncwarp();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(L::epilogue_bytes(BLOCK_N) % 1024 == 0, "The stages must start on a swizzle atom");
    DG_STATIC_ASSERT(L::kHalfBytesPerStage % 1024 == 0 and L::b_bytes_per_stage(BLOCK_N) % 1024 == 0,
        "Every A half and every B stage must start on a swizzle atom");
    if (threadIdx.x == 0 and (static_cast<uint32_t>(__cvta_generic_to_shared(smem_buffer)) & 1023u) != 0u)
        asm volatile("trap;");

    __nv_fp8_e4m3* smem_a[kNumStages];
    __nv_fp8_e4m3* smem_b[kNumStages];
    float* smem_scales_b[kNumStages];
    Barrier* full_barriers[kNumStages];
    Barrier* empty_barriers[kNumStages];
#pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i)
    {
        smem_a[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + L::a_offset(BLOCK_N, i));
        smem_b[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + L::b_offset(BLOCK_N, kNumStages, i));
        smem_scales_b[i] = reinterpret_cast<float*>(smem_buffer + L::scales_b_offset(BLOCK_N, kNumStages, i));
    }
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + L::barriers_offset(BLOCK_N, kNumStages));
#pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i)
    {
        full_barriers[i] = barrier_start_ptr + i;
        empty_barriers[i] = barrier_start_ptr + kNumStages + i;
    }
    auto xchg_barriers = reinterpret_cast<Barrier*>(smem_buffer + L::xchg_barrier_offset(BLOCK_N));

    // Barriers: one arrival per math warp (both warp-groups) frees a stage; the four peer lanes that send the amax
    // values complete an exchange barrier. Only the initialisation has to be published to the peer before its first
    // remote arrival (fence_barrier_init), so the cluster arrive can be relaxed, as in CUTLASS's cluster kernels.
    if (threadIdx.x == kNumMathThreads)
    {
#pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i)
        {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(L::kNumMathWarps);
        }
        xchg_barriers[0].init(4);
        xchg_barriers[1].init(4);
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::fence_barrier_init();
    }
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();

    // m_block_idx is the SwiGLU block b (I / 128 of them); the cluster is the scheduling unit.
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = SchedulerType(problem_input);

    if (threadIdx.x >= kNumMathThreads)
    {
        if (threadIdx.x == kNumMathThreads)
        {
            while (scheduler.get_next_block_unit(m_block_idx, n_block_idx, cluster_idx, num_clusters))
            {
                // This CTA's 64 gate rows (expert * 2I + b * 128 + 64 * rank) and the up rows I further
                uint32_t const gate_row = scheduler.get_global_m_idx(SHAPE_M, BLOCK_M, m_block_idx, n_block_idx)
                    + cta_rank * L::kRowsPerCta;
                uint32_t const n_idx = scheduler.get_global_n_idx(n_block_idx);
                for (uint32_t k_iter = 0; k_iter < kNumIterations; ++k_iter)
                {
#pragma unroll
                    for (uint32_t s = 0; s < kNumStages; ++s)
                    {
                        empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);
                        auto& full_barrier = *full_barriers[s];
                        int const k_idx = static_cast<int>(k_iter * kFullKOfAllStages + s * BLOCK_K);
                        tma_copy(&tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier), smem_a[s], k_idx, gate_row);
                        tma_copy(&tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier),
                            smem_a[s] + L::kHalfBytesPerStage, k_idx, gate_row + SHAPE_M_HALF);
                        tma_copy(&tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier), smem_b[s], k_idx, n_idx);
                        tma_copy(&tensor_map_scales_b, reinterpret_cast<uint64_t*>(&full_barrier), smem_scales_b[s],
                            n_block_idx * BLOCK_N, scheduler.get_global_scales_b_idx(k_idx / BLOCK_K));
                        full_barrier.arrive_and_expect_tx(kTxBytesPerStage);
                    }
                }
            }
        }
    }
    else
    {
        // Warp-group 0 computes the gate rows and runs the epilogue, warp-group 1 the up rows
        uint32_t const math_wg = warp_idx / 4;
        uint32_t const tid_wg = threadIdx.x % 128;
        uint32_t const scale_offset = (lane_idx % 4) * 2;
        constexpr uint32_t kChunks = WGMMA::kNumAccum / 4; // n8 token chunks per thread: BLOCK_N / 8
        uint8_t* const smem_dq = smem_buffer;
        uint32_t const dq_addr = F::smem_addr(smem_dq);
        uint32_t const amax_addr = F::smem_addr(smem_buffer + L::amax_offset(BLOCK_N));
        uint32_t const up_addr = F::smem_addr(smem_buffer + L::up_offset(BLOCK_N)) + tid_wg * 4;
        uint32_t const peer_buf_local = F::smem_addr(smem_buffer + L::peer_offset(BLOCK_N));
        uint32_t const xchg_local = F::smem_addr(xchg_barriers);
        uint32_t const peer_rank = cta_rank ^ 1u;
        uint32_t const peer_buf_remote = X::peer_addr(peer_buf_local, peer_rank);
        uint32_t const xchg_remote = X::peer_addr(xchg_local, peer_rank);
        // The thread's output columns j_0 and j_0 + 8 of this CTA's 64 and its tokens 8 i + tok_lo + {0, 1}
        uint32_t const j_0 = (warp_idx % 4) * 16 + lane_idx / 4;
        uint32_t const tok_lo = 2 * (lane_idx % 4);
        bool const odd_row = ((lane_idx / 4) & 1u) != 0u;
        uint32_t const a_half = math_wg * L::kHalfBytesPerStage;
        uint32_t tile = 0;

        while (scheduler.get_next_block_unit(m_block_idx, n_block_idx, cluster_idx, num_clusters))
        {
            // Weight scales of gate block b (row b of the expert's 2I/128 scale rows) or up block b (row I/128 + b)
            auto const num_previous_lines
                = scheduler.get_global_scales_a_idx(ceil_div(SHAPE_M, BLOCK_K), 0, 0, n_block_idx);
            float const* const local_scales = scales_a
                + (num_previous_lines + ((m_block_idx * BLOCK_M) / BLOCK_K) + math_wg * (SHAPE_M_HALF / BLOCK_K))
                    * SHAPE_K_SCALES;

            // The unsplit kernel's mainloop for one half: per stage four WGMMAs, then the promotion with
            // scale_a * scale_b (fp8_gemm_kernel_swapAB's expression)
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum] = {0};
            for (uint32_t k_iter = 0; k_iter < kNumIterations; ++k_iter)
            {
#pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++s)
                {
                    float const scale_a = __ldg(local_scales + k_iter * kNumStages + s);
                    full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);

                    float scale_0[WGMMA::kNumAccum / 4], scale_1[WGMMA::kNumAccum / 4];
#pragma unroll
                    for (int i = 0; i < WGMMA::kNumAccum / 4; ++i)
                    {
                        float2 scale_b
                            = ld_shared(reinterpret_cast<const float2*>(smem_scales_b[s] + i * 8 + scale_offset));
                        scale_0[i] = scale_a * scale_b.x;
                        scale_1[i] = scale_a * scale_b.y;
                    }

#pragma unroll
                    for (int i = 0; i < WGMMA::kNumAccum; ++i)
                        warpgroup_fence_operand(accum[i]);
                    warpgroup_arrive();
#pragma unroll
                    for (int k = 0; k < BLOCK_K / WGMMA::K; ++k)
                    {
                        auto desc_a = make_smem_desc(smem_a[s] + a_half + k * WGMMA::K, 1);
                        auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    warpgroup_commit_batch();
#pragma unroll
                    for (int i = 0; i < WGMMA::kNumAccum; ++i)
                        warpgroup_fence_operand(accum[i]);
                    warpgroup_wait<0>();
                    if (lane_idx == 0)
                        empty_barriers[s]->arrive();

#pragma unroll
                    for (int i = 0; i < WGMMA::kNumAccum / 4; ++i)
                    {
                        final_accum[i * 4 + 0] += scale_0[i] * accum[i * 4 + 0];
                        final_accum[i * 4 + 1] += scale_1[i] * accum[i * 4 + 1];
                        final_accum[i * 4 + 2] += scale_0[i] * accum[i * 4 + 2];
                        final_accum[i * 4 + 3] += scale_1[i] * accum[i * 4 + 3];
                    }
                }
            }

            if (math_wg == 1)
            {
                // The up sums rounded to bf16 pairs, as the unsplit kernel rounds them, to the gate warp-group. The
                // previous tile's words must have been read first.
                if (tile > 0)
                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, kBarUpFree);
#pragma unroll
                for (uint32_t i = 0; i < kChunks; ++i)
                {
#pragma unroll
                    for (uint32_t s = 0; s < 2; ++s)
                        X::sts_u32(up_addr + (2 * i + s) * 128 * 4,
                            F::bf162_bits(
                                __float22bfloat162_rn({final_accum[4 * i + 2 * s], final_accum[4 * i + 2 * s + 1]})));
                }
                cutlass::arch::NamedBarrier::arrive(kNumMathThreads, kBarUpReady);
                ++tile;
                continue;
            }

            // Gate warp-group: the unsplit kernel's epilogue steps 1-3 on this CTA's 64 columns
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, kBarUpReady);
            uint32_t h[kChunks][2];
            float ax[kChunks][2];
#pragma unroll
            for (uint32_t i = 0; i < kChunks; ++i)
            {
#pragma unroll
                for (uint32_t s = 0; s < 2; ++s)
                {
                    __nv_bfloat162 const hh = F::silu2_mul(
                        __float22bfloat162_rn({final_accum[4 * i + 2 * s], final_accum[4 * i + 2 * s + 1]}),
                        F::bits_bf162(X::lds_u32(up_addr + (2 * i + s) * 128 * 4)));
                    h[i][s] = F::bf162_bits(hh);
                    float2 const f = __bfloat1622float2(hh);
                    if (s == 0)
                    {
                        ax[i][0] = fabsf(f.x);
                        ax[i][1] = fabsf(f.y);
                    }
                    else
                    {
                        ax[i][0] = fmaxf(ax[i][0], fabsf(f.x));
                        ax[i][1] = fmaxf(ax[i][1], fabsf(f.y));
                    }
                }
            }
            cutlass::arch::NamedBarrier::arrive(kNumMathThreads, kBarUpFree);
#pragma unroll
            for (uint32_t i = 0; i < kChunks; ++i)
            {
#pragma unroll
                for (uint32_t t = 0; t < 2; ++t)
                {
                    ax[i][t] = fmaxf(ax[i][t], __shfl_xor_sync(0xffffffffu, ax[i][t], 4));
                    ax[i][t] = fmaxf(ax[i][t], __shfl_xor_sync(0xffffffffu, ax[i][t], 8));
                    ax[i][t] = fmaxf(ax[i][t], __shfl_xor_sync(0xffffffffu, ax[i][t], 16));
                }
            }
            if (lane_idx < 4)
            {
#pragma unroll
                for (uint32_t i = 0; i < kChunks; ++i)
                    G::sts_v2_f32(amax_addr + (warp_idx * BLOCK_N + 8 * i + tok_lo) * 4, ax[i][0], ax[i][1]);
            }
            // The previous tile's dq store must have finished reading the staging tile before it is rewritten
            if (threadIdx.x == 0)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, kBarGate); // A: partials published, staging free

            // This CTA's |max| over its 64 columns per token, then the exchange: warp 0's four lanes (one per lane % 4
            // class, together all BLOCK_N tokens) send theirs to the peer and arrive on the peer's barrier.
            float local[kChunks][2];
#pragma unroll
            for (uint32_t i = 0; i < kChunks; ++i)
            {
                float2 m = G::lds_v2_f32(amax_addr + (8 * i + tok_lo) * 4);
#pragma unroll
                for (uint32_t w = 1; w < L::kNumEpilogueWarps; ++w)
                {
                    float2 const v = G::lds_v2_f32(amax_addr + (w * BLOCK_N + 8 * i + tok_lo) * 4);
                    m.x = fmaxf(m.x, v.x);
                    m.y = fmaxf(m.y, v.y);
                }
                local[i][0] = m.x;
                local[i][1] = m.y;
            }
            uint32_t const parity_buf = tile & 1u;
            if (warp_idx == 0 and lane_idx < 4)
            {
#pragma unroll
                for (uint32_t i = 0; i < kChunks; ++i)
                    X::st_cluster_v2_f32(peer_buf_remote + (parity_buf * BLOCK_N + 8 * i + tok_lo) * 4, local[i][0],
                        local[i][1]);
                X::arrive_cluster(xchg_remote + parity_buf * L::kBarrierBytes);
            }
            X::wait_cluster(xchg_local + parity_buf * L::kBarrierBytes, (tile >> 1) & 1u);

            // amax over the 128 columns, clamped at 1e-10f; qs = 448 / amax and dequant = amax * (1 / 448), exactly as
            // fp8_gemm_kernel_swapAB_swiglu and silu_chunk_mul_quantize_1x128_fp32_sorted_kernel write them
            float qs[kChunks][2], dequant[kChunks][2];
#pragma unroll
            for (uint32_t i = 0; i < kChunks; ++i)
            {
                float2 const pv = G::lds_v2_f32(peer_buf_local + (parity_buf * BLOCK_N + 8 * i + tok_lo) * 4);
                float const amax_0 = fmaxf(fmaxf(local[i][0], pv.x), 1e-10f);
                float const amax_1 = fmaxf(fmaxf(local[i][1], pv.y), 1e-10f);
                qs[i][0] = 448.f / amax_0;
                qs[i][1] = 448.f / amax_1;
                dequant[i][0] = amax_0 * (1.f / 448.f);
                dequant[i][1] = amax_1 * (1.f / 448.f);
            }

            // Requantize this CTA's 64 columns into the [BLOCK_N][64] staging tile (row-major, no swizzle), with the
            // unsplit kernel's paired converts and lane-four byte exchange
#pragma unroll
            for (uint32_t i = 0; i < kChunks; ++i)
            {
#pragma unroll
                for (uint32_t s = 0; s < 2; ++s)
                {
                    float2 const f = __bfloat1622float2(F::bits_bf162(h[i][s]));
                    uint32_t const q = static_cast<uint32_t>(__nv_cvt_float2_to_fp8x2(
                        make_float2(f.x * qs[i][0], f.y * qs[i][1]), __NV_SATFINITE, __NV_E4M3));
                    uint32_t const send = odd_row ? (q & 0xffu) : (q >> 8);
                    uint32_t const recv = __shfl_xor_sync(0xffffffffu, send, 4);
                    uint32_t const pair16 = odd_row ? (recv | (q & 0xff00u)) : ((q & 0xffu) | (recv << 8));
                    uint32_t const token = 8 * i + tok_lo + (odd_row ? 1u : 0u);
                    uint32_t const col = j_0 + 8 * s - (odd_row ? 1u : 0u);
                    F::sts_u16(dq_addr + token * L::kOutCols + col, static_cast<uint16_t>(pair16));
                }
            }

            // One dequant scale per token, written by CTA 0 (both CTAs hold the same values)
            uint32_t const n_global_idx = scheduler.get_global_n_idx(n_block_idx);
            if (cta_rank == 0 and warp_idx == 0 and lane_idx < 4)
            {
                float* const sd_row = gmem_sd + static_cast<uint64_t>(m_block_idx) * sd_ld + n_global_idx + tok_lo;
#pragma unroll
                for (uint32_t i = 0; i < kChunks; ++i)
                    *reinterpret_cast<float2*>(sd_row + 8 * i) = make_float2(dequant[i][0], dequant[i][1]);
            }

            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, kBarGate); // B: staging tile complete
            if (threadIdx.x == 0)
            {
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_dq, smem_dq, m_block_idx * BLOCK_M + cta_rank * L::kOutCols,
                    n_global_idx);
                cute::tma_store_arrive();
            }
            __syncwarp();
            ++tile;
        }

        // Balance the last up-free arrival of the gate warp-group; drain the last tile's dq store
        if (math_wg == 1 and tile > 0)
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, kBarUpFree);
        if (threadIdx.x == 0)
            cute::tma_store_wait<0>();
    }

    // Neither CTA leaves while its peer might still address its shared memory. The peer's only accesses to this
    // CTA's shared memory are its amax stores and exchange-barrier arrivals, and this CTA has already waited (with
    // acquire) for the arrivals of every tile it ran, so no ordering is left for the sync to provide: the arrive is
    // relaxed, which spares each thread a fence on its outstanding global stores.
    __syncwarp();
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}
} // namespace deep_gemm
