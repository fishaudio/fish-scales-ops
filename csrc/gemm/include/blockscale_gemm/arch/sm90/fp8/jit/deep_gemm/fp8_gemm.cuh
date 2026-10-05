/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/MIT
 *
 *
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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include "compiler.cuh"
#include "fp8_gemm_impl.cuh"
#include "mma_utils.cuh"
#include "scheduler.cuh"
#include "tma_utils.cuh"
#include "utils.cuh"

namespace deep_gemm
{
template <typename T>
static CUtensorMap make_2d_tma_a_desc(T* global_address, uint32_t shape_m, uint32_t shape_k, uint32_t block_m,
    uint32_t block_k, uint32_t num_groups, GemmType gemm_type, uint64_t global_stride_in_bytes = 0)
{
    return make_2d_tma_desc(global_address, Layout::RowMajor,
        shape_m * (gemm_type == GemmType::GroupedMasked ? num_groups : 1), shape_k, block_m, block_k,
        global_stride_in_bytes);
}

template <typename T>
CUtensorMap make_2d_tma_b_desc(T* global_address, uint32_t shape_n, uint32_t shape_k, uint32_t block_n,
    uint32_t block_k, uint32_t num_groups, GemmType gemm_type, uint64_t global_stride_in_bytes = 0)
{
    return make_2d_tma_desc(global_address, Layout::ColMajor, shape_k,
        shape_n * (gemm_type != GemmType::Normal ? num_groups : 1), block_k, block_n, global_stride_in_bytes);
}

template <typename T>
CUtensorMap make_2d_tma_d_desc(T* global_address, uint32_t shape_m, uint32_t shape_n, uint32_t block_m,
    uint32_t block_n, uint32_t num_groups, GemmType gemm_type, uint64_t global_stride_in_bytes = 0)
{
    return make_2d_tma_desc(global_address, Layout::RowMajor,
        shape_m * (gemm_type == GemmType::GroupedMasked ? num_groups : 1), shape_n, min(block_m, shape_m),
        min(block_n, shape_n), global_stride_in_bytes, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
}

template <typename T>
CUtensorMap make_2d_tma_scales_a_desc(T* global_address, uint32_t shape_m, uint32_t shape_k, uint32_t block_m,
    uint32_t block_k, uint32_t num_groups, GemmType gemm_type, uint64_t global_stride_in_bytes = 0)
{
    // Make TMA aligned to 16 bytes
    constexpr uint32_t kAlignment = 16 / sizeof(T);
    shape_m = ceil_div(shape_m, kAlignment) * kAlignment;

    return make_2d_tma_desc(global_address, Layout::ColMajor, shape_m,
        ceil_div(shape_k, block_k)
            * ((gemm_type == GemmType::GroupedMasked || gemm_type == GemmType::StridedBatched) ? num_groups : 1),
        block_m, 1, global_stride_in_bytes, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
}

template <typename T>
CUtensorMap make_tma_scales_a_offset_desc(T* global_address, int64_t max_m_padded_total, uint32_t shape_k,
    uint32_t block_m, uint32_t block_k, uint64_t global_stride_in_bytes = 0)
{
    return make_2d_tma_desc(global_address, Layout::ColMajor, max_m_padded_total, ceil_div(shape_k, block_k), block_m,
        1, global_stride_in_bytes, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
}

template <typename T>
CUtensorMap make_2d_tma_a_desc_swapAB(T* global_address, uint32_t shape_m, uint32_t shape_k, uint32_t block_m,
    uint32_t block_k, uint32_t num_groups, GemmType gemm_type, uint64_t global_stride_in_bytes = 0)
{
    return make_2d_tma_desc(global_address, Layout::RowMajor,
        shape_m * (gemm_type != GemmType::Normal ? num_groups : 1), shape_k, block_m, block_k, global_stride_in_bytes);
}

template <typename T>
CUtensorMap make_2d_tma_b_desc_swapAB(T* global_address, uint32_t shape_n, uint32_t shape_k, uint32_t block_n,
    uint32_t block_k, uint32_t num_groups, GemmType gemm_type, uint64_t global_stride_in_bytes = 0)
{
    return make_2d_tma_desc(global_address, Layout::ColMajor, shape_k,
        shape_n * (gemm_type == GemmType::GroupedMasked ? num_groups : 1), block_k, block_n, global_stride_in_bytes);
}

template <typename T>
CUtensorMap make_2d_tma_d_desc_swapAB(T* global_address, uint32_t shape_m, uint32_t shape_n, uint32_t block_m,
    uint32_t block_n, uint32_t num_groups, GemmType gemm_type, uint64_t global_stride_in_bytes = 0)
{
    return make_2d_tma_desc(global_address, Layout::RowMajor,
        shape_n * (gemm_type == GemmType::GroupedMasked ? num_groups : 1), shape_m, min(block_n, shape_n),
        min(block_m, shape_m), global_stride_in_bytes, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
}

template <typename T>
CUtensorMap make_2d_tma_scales_b_desc_swapAB(T* global_address, uint32_t shape_n, uint32_t shape_k, uint32_t block_n,
    uint32_t block_k, uint32_t num_groups, GemmType gemm_type, uint64_t global_stride_in_bytes = 0)
{
    // Make TMA aligned to 16 bytes
    constexpr uint32_t kAlignment = 16 / sizeof(T);
    shape_n = ceil_div(shape_n, kAlignment) * kAlignment;

    return make_2d_tma_desc(global_address, Layout::RowMajor,
        ceil_div(shape_k, block_k)
            * ((gemm_type == GemmType::GroupedMasked || gemm_type == GemmType::StridedBatched) ? num_groups : 1),
        shape_n, 1, block_n, global_stride_in_bytes, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
}

template <typename T>
CUtensorMap make_tma_scales_b_offset_desc_swapAB(T* global_address, int64_t max_n_padded_total, uint32_t shape_k,
    uint32_t block_n, uint32_t block_k, uint64_t global_stride_in_bytes = 0)
{
    return make_2d_tma_desc(global_address, Layout::RowMajor, ceil_div(shape_k, block_k), max_n_padded_total, 1,
        block_n, global_stride_in_bytes, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
}

template <typename T>
CUtensorMap make_2d_tma_desc(T* global_address, Layout layout, uint32_t gmem_rows, uint32_t gmem_cols,
    uint32_t smem_rows, uint32_t smem_cols, uint64_t global_stride_in_bytes,
    CUtensorMapSwizzle swizzle_type = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B)
{
    if (layout == Layout::RowMajor)
    {
        uint64_t gmem_dim[2] = {gmem_cols, gmem_rows};
        uint32_t smem_dim[2] = {smem_cols, smem_rows};
        global_stride_in_bytes = global_stride_in_bytes ? global_stride_in_bytes : gmem_cols * sizeof(T);
        return make_2d_tma_copy_desc(global_address, gmem_dim, global_stride_in_bytes, smem_dim, swizzle_type);
    }
    else
    {
        uint64_t gmem_dim[2] = {gmem_rows, gmem_cols};
        uint32_t smem_dim[2] = {smem_rows, smem_cols};
        global_stride_in_bytes = global_stride_in_bytes ? global_stride_in_bytes : gmem_rows * sizeof(T);
        return make_2d_tma_copy_desc(global_address, gmem_dim, gmem_rows * sizeof(T), smem_dim, swizzle_type);
    }
}

// `pdl` launches with programmatic stream serialization (the dense GEMM's kernel waits on its predecessor with
// griddepcontrol.wait; the grouped instantiations carry no wait, so their callers keep the default false).
template <typename LayoutIndexType>
void runGemm(cudaKernel_t kernel, void* mat_a, int ld_a, void* mat_b, int ld_b, void* mat_d, int ld_d, float* scales_a,
    float* scales_b, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, uint32_t block_m, uint32_t block_n,
    uint32_t block_k, uint32_t num_groups, uint32_t num_tma_multicast, GemmType gemm_type,
    LayoutIndexType* grouped_layout, cudaStream_t stream, int num_sms, uint32_t smem_size, bool pdl = false)
{
    auto tma_a_desc = make_2d_tma_a_desc(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_a), shape_m, shape_k, block_m, block_k, num_groups, gemm_type, ld_a);
    auto tma_b_desc = make_2d_tma_b_desc(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_b), shape_n, shape_k, block_n, block_k, num_groups, gemm_type, ld_b);
    auto tma_scales_a_desc
        = make_2d_tma_scales_a_desc(scales_a, shape_m, shape_k, block_m, block_k, num_groups, gemm_type);
    // The dense kernel with block_n a multiple of 64 stages D as 64-column slabs with the 128-byte swizzle
    // (fp8_gemm_kernel, kSwizzledD) and stores each slab with one TMA store, so its D box is min(block_m, M) rows by
    // 64 columns with that swizzle. This condition must match kSwizzledD exactly.
    bool const swizzled_d = gemm_type == GemmType::Normal && block_n % 64 == 0;
    auto tma_d_desc = swizzled_d
        ? make_2d_tma_desc(reinterpret_cast<__nv_bfloat16*>(mat_d), Layout::RowMajor, shape_m, shape_n,
            min(block_m, shape_m), 64u, static_cast<uint64_t>(ld_d) * 2, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B)
        : make_2d_tma_d_desc(reinterpret_cast<__nv_bfloat16*>(mat_d), shape_m, shape_n, block_m, block_n, num_groups,
            gemm_type, ld_d * 2);

    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumMathThreadsPerGroup = 128;
    DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

    // Cluster launch
    cudaLaunchConfig_t config;
    config.gridDim = num_sms;
    config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(static_cast<int32_t>(block_m));
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    // Clusters for TMA multicast
    // NOTES: `>= 4` cluster size will cause performance degradation
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {num_tma_multicast, 1, 1};
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attrs;
    config.numAttrs = pdl ? 2 : 1;

    NormalSchedulerInput input;
    input.shape_m = shape_m;
    input.grouped_layout = grouped_layout;

    // Launch
    auto status = cudaLaunchKernelEx(&config, kernel, reinterpret_cast<__nv_bfloat16*>(mat_d), scales_b, input,
        tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc);
    DG_HOST_ASSERT(status == cudaSuccess);
}

// `pdl`: as runGemm (only the dense swap-AB kernel waits on its predecessor).
template <typename LayoutIndexType>
void runGemmSwapAB(cudaKernel_t kernel, void* mat_a, int ld_a, void* mat_b, int ld_b, void* mat_d, int ld_d,
    float* scales_a, float* scales_b, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, uint32_t block_m,
    uint32_t block_n, uint32_t block_k, uint32_t num_groups, uint32_t num_tma_multicast, GemmType gemm_type,
    LayoutIndexType* grouped_layout, cudaStream_t stream, int num_sms, uint32_t smem_size, bool pdl = false)
{
    auto tma_a_desc = make_2d_tma_a_desc_swapAB(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_a), shape_m, shape_k, block_m, block_k, num_groups, gemm_type, ld_a);
    auto tma_b_desc = make_2d_tma_b_desc_swapAB(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_b), shape_n, shape_k, block_n, block_k, num_groups, gemm_type, ld_b);
    auto tma_scales_b_desc
        = make_2d_tma_scales_b_desc_swapAB(scales_b, shape_n, shape_k, block_n, block_k, num_groups, gemm_type);
    auto tma_d_desc = make_2d_tma_d_desc_swapAB(
        reinterpret_cast<__nv_bfloat16*>(mat_d), shape_m, shape_n, block_m, block_n, num_groups, gemm_type, ld_d * 2);

    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumMathThreadsPerGroup = 128;
    DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

    // Cluster launch
    cudaLaunchConfig_t config;
    config.gridDim = num_sms;
    config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(static_cast<int32_t>(block_m));
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    // Clusters for TMA multicast
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {num_tma_multicast, 1, 1};
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attrs;
    config.numAttrs = pdl ? 2 : 1;

    NormalSchedulerInputSwapAB input;
    input.shape_n = shape_n;
    input.grouped_layout = grouped_layout;

    auto status = cudaLaunchKernelEx(&config, kernel, reinterpret_cast<__nv_bfloat16*>(mat_d), scales_a, input,
        tma_a_desc, tma_b_desc, tma_scales_b_desc, tma_d_desc);
    DG_HOST_ASSERT(status == cudaSuccess);
}

template <typename LayoutIndexType>
void runGemm(cudaKernel_t kernel, void* mat_a, int ld_a, void* mat_b, int ld_b, void* mat_d, int ld_d, float* scales_a,
    float* scales_b, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, uint32_t block_m, uint32_t block_n,
    uint32_t block_k, uint32_t num_groups, uint32_t num_tma_multicast, GemmType gemm_type,
    LayoutIndexType* problem_m_offsets, cudaStream_t stream, int num_sms, uint32_t smem_size,
    uint32_t max_shape_m_padded)
{
    auto tma_a_desc = make_2d_tma_a_desc(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_a), shape_m, shape_k, block_m, block_k, num_groups, gemm_type);
    auto tma_b_desc = make_2d_tma_b_desc(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_b), shape_n, shape_k, block_n, block_k, num_groups, gemm_type);
    auto tma_scales_a_desc = make_tma_scales_a_offset_desc(scales_a, max_shape_m_padded, shape_k, block_m, block_k);
    auto tma_d_desc = make_2d_tma_d_desc(
        reinterpret_cast<__nv_bfloat16*>(mat_d), shape_m, shape_n, block_m, block_n, num_groups, gemm_type);
    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumMathThreadsPerGroup = 128;
    DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

    // Cluster launch
    cudaLaunchConfig_t config;
    config.gridDim = num_sms;
    config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(static_cast<int32_t>(block_m));
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    // Clusters for TMA multicast
    // NOTES: `>= 4` cluster size will cause performance degradation
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {num_tma_multicast, 1, 1};
    config.attrs = &attr;
    config.numAttrs = 1;

    GroupedWithOffsetSchedulerInput input;
    input.problem_m_offsets = problem_m_offsets;

    // Launch
    auto status = cudaLaunchKernelEx(&config, kernel, reinterpret_cast<__nv_bfloat16*>(mat_d), scales_b, input,
        tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc);
    DG_HOST_ASSERT(status == cudaSuccess);
}

template <typename LayoutIndexType>
void runGemmSwapAB(cudaKernel_t kernel, void* mat_a /* weight*/, int ld_a, void* mat_b /* act*/, int ld_b, void* mat_d,
    int ld_d, float* scales_a /* weight scales*/, float* scales_b /* act scales*/, uint32_t shape_m, uint32_t shape_n,
    uint32_t shape_k, uint32_t block_m, uint32_t block_n, uint32_t block_k, uint32_t num_groups,
    uint32_t num_tma_multicast, GemmType gemm_type, LayoutIndexType* problem_n_offsets, cudaStream_t stream,
    int num_sms, uint32_t smem_size, uint32_t max_shape_n_padded)
{
    // Create tensor mappings using swapAB version functions, note the parameter order
    auto tma_a_desc = make_2d_tma_a_desc_swapAB(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_a), shape_m, shape_k, block_m, block_k, num_groups, gemm_type);
    auto tma_b_desc = make_2d_tma_b_desc_swapAB(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_b), shape_n, shape_k, block_n, block_k, num_groups, gemm_type);
    auto tma_scales_b_desc
        = make_tma_scales_b_offset_desc_swapAB(scales_b, max_shape_n_padded, shape_k, block_n, block_k);
    auto tma_d_desc = make_2d_tma_d_desc_swapAB(
        reinterpret_cast<__nv_bfloat16*>(mat_d), shape_m, shape_n, block_m, block_n, num_groups, gemm_type);
    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumMathThreadsPerGroup = 128;
    DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

    // Cluster launch
    cudaLaunchConfig_t config;
    config.gridDim = num_sms;
    config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(static_cast<int32_t>(block_m));
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    // Clusters for TMA multicast
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {num_tma_multicast, 1, 1};
    config.attrs = &attr;
    config.numAttrs = 1;

    // Update input structure to use N dimension offsets
    GroupedWithOffsetSchedulerInputSwapAB input;
    input.problem_n_offsets = problem_n_offsets; // Now offsets are for N dimension

    auto status = cudaLaunchKernelEx(&config, kernel, reinterpret_cast<__nv_bfloat16*>(mat_d), scales_a, input,
        tma_a_desc, tma_b_desc, tma_scales_b_desc, tma_d_desc);
    DG_HOST_ASSERT(status == cudaSuccess);
}

void runGemm(cudaKernel_t kernel, void* mat_a, uint64_t ld_a, uint64_t stride_a, void* mat_b, uint64_t ld_b,
    uint64_t stride_b, void* mat_d, uint64_t ld_d, uint64_t stride_d, float* scales_a, float* scales_b,
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, uint32_t block_m, uint32_t block_n, uint32_t block_k,
    uint32_t num_groups, uint32_t num_tma_multicast, GemmType gemm_type, cudaStream_t stream, int num_sms,
    uint32_t smem_size)
{
    auto tma_a_desc = make_2d_tma_a_desc(reinterpret_cast<__nv_fp8_e4m3*>(mat_a), shape_m * num_groups, shape_k,
        block_m, block_k, num_groups, gemm_type, ld_a);
    auto tma_b_desc = make_2d_tma_b_desc(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_b), shape_n, shape_k, block_n, block_k, num_groups, gemm_type, ld_b);
    auto tma_scales_a_desc
        = make_2d_tma_scales_a_desc(scales_a, shape_m, shape_k, block_m, block_k, num_groups, gemm_type);
    auto tma_d_desc = make_2d_tma_d_desc(
        reinterpret_cast<__nv_bfloat16*>(mat_d), shape_m, shape_n, block_m, block_n, num_groups, gemm_type, ld_d * 2);
    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumMathThreadsPerGroup = 128;
    DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

    // Cluster launch
    cudaLaunchConfig_t config;
    config.gridDim = num_sms;
    config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(static_cast<int32_t>(block_m));
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    // Clusters for TMA multicast
    // NOTES: `>= 4` cluster size will cause performance degradation
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {num_tma_multicast, 1, 1};
    config.attrs = &attr;
    config.numAttrs = 1;

    StridedBatchedSchedulerInput input{shape_m, ld_a, stride_a, ld_b, stride_b, ld_d, stride_d};
    // Launch
    auto status = cudaLaunchKernelEx(&config, kernel, reinterpret_cast<__nv_bfloat16*>(mat_d), scales_b, input,
        tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc);
    DG_HOST_ASSERT(status == cudaSuccess);
}

// Launch of fp8_gemm_kernel_2wg (non-swap GroupedContiguous FC1, two math warp-groups split along N). The tensor maps
// are the ones runGemm builds for today's GroupedContiguous FC1 with block_n = 128: the B box is one 128-row half
// (the kernel issues it twice per stage, gate rows and up rows) and the D box is one warp-group's 64 x 128 bf16 tile.
// Only the thread count differs: 2 math warp-groups + 1 TMA warp-group = 384 at block_m = 64.
template <typename LayoutIndexType>
void runGemm2wg(cudaKernel_t kernel, void* mat_a, int ld_a, void* mat_b, int ld_b, void* mat_d, int ld_d,
    float* scales_a, float* scales_b, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, uint32_t block_m,
    uint32_t block_n, uint32_t block_k, uint32_t num_groups, LayoutIndexType* grouped_layout, cudaStream_t stream,
    int num_sms, uint32_t smem_size)
{
    constexpr auto gemm_type = GemmType::GroupedContiguous;
    auto tma_a_desc = make_2d_tma_a_desc(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_a), shape_m, shape_k, block_m, block_k, num_groups, gemm_type, ld_a);
    auto tma_b_desc = make_2d_tma_b_desc(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_b), shape_n, shape_k, block_n, block_k, num_groups, gemm_type, ld_b);
    auto tma_scales_a_desc
        = make_2d_tma_scales_a_desc(scales_a, shape_m, shape_k, block_m, block_k, num_groups, gemm_type);
    auto tma_d_desc = make_2d_tma_d_desc(
        reinterpret_cast<__nv_bfloat16*>(mat_d), shape_m, shape_n, block_m, block_n, num_groups, gemm_type, ld_d * 2);

    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumMathThreadsPerGroup = 128;
    DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

    cudaLaunchConfig_t config;
    config.gridDim = num_sms;
    config.blockDim = get_num_threads_per_sm_2wg<kNumTMAThreads, kNumMathThreadsPerGroup>();
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    // No multicast (per-block weights differ); keep the cluster attribute at 1 like the other grouped launches.
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {1, 1, 1};
    config.attrs = &attr;
    config.numAttrs = 1;

    GroupedContiguousSchedulerInput input;
    input.shape_m = shape_m;
    input.grouped_layout = grouped_layout;

    auto status = cudaLaunchKernelEx(&config, kernel, reinterpret_cast<__nv_bfloat16*>(mat_d), scales_b, input,
        tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc);
    DG_HOST_ASSERT(status == cudaSuccess);
}

// Launch of fp8_gemm_kernel_2wg_swiglu (the same FC1 with the SwiGLU + 1x128 FP8 requantize fused into the epilogue,
// fused-FC1 step 2). A, B and the A scales use runGemm2wg's tensor maps. The output is the SwiGLU kernel's pair:
// dq [shape_m, shape_n / 2] fp8, stored through its own tensor map with one 64 x 128 byte box per tile and the 128-byte
// swizzle (the kernel stages the tile in that layout, which keeps its 16-bit shared-memory stores conflict-free), and
// sd [shape_n / 256, sd_ld] fp32, which the kernel writes directly (one scale per row and 128-column block).
template <typename LayoutIndexType>
void runGemm2wgSwiglu(cudaKernel_t kernel, void* mat_a, int ld_a, void* mat_b, int ld_b, void* mat_dq, float* mat_sd,
    uint32_t sd_ld, float* scales_a, float* scales_b, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
    uint32_t block_m, uint32_t block_n, uint32_t block_k, uint32_t num_groups, LayoutIndexType* grouped_layout,
    cudaStream_t stream, int num_sms, uint32_t smem_size)
{
    constexpr auto gemm_type = GemmType::GroupedContiguous;
    auto tma_a_desc = make_2d_tma_a_desc(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_a), shape_m, shape_k, block_m, block_k, num_groups, gemm_type, ld_a);
    auto tma_b_desc = make_2d_tma_b_desc(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_b), shape_n, shape_k, block_n, block_k, num_groups, gemm_type, ld_b);
    auto tma_scales_a_desc
        = make_2d_tma_scales_a_desc(scales_a, shape_m, shape_k, block_m, block_k, num_groups, gemm_type);
    uint32_t const shape_i = shape_n / 2;
    auto tma_dq_desc = make_2d_tma_desc(reinterpret_cast<__nv_fp8_e4m3*>(mat_dq), Layout::RowMajor, shape_m, shape_i,
        min(block_m, shape_m), block_n, static_cast<uint64_t>(shape_i), CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);

    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumMathThreadsPerGroup = 128;
    DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

    cudaLaunchConfig_t config;
    config.gridDim = num_sms;
    config.blockDim = get_num_threads_per_sm_2wg<kNumTMAThreads, kNumMathThreadsPerGroup>();
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    // No multicast (per-block weights differ); keep the cluster attribute at 1 like the other grouped launches.
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {1, 1, 1};
    config.attrs = &attr;
    config.numAttrs = 1;

    GroupedContiguousSchedulerInput input;
    input.shape_m = shape_m;
    input.grouped_layout = grouped_layout;

    auto status = cudaLaunchKernelEx(&config, kernel, mat_sd, scales_b, input, tma_a_desc, tma_b_desc,
        tma_scales_a_desc, tma_dq_desc, sd_ld);
    DG_HOST_ASSERT(status == cudaSuccess);
}

// Launch of fp8_gemm_kernel_swapAB_swiglu (the swap-AB FC1 in which one CTA owns gate block b and up block b, with the
// SwiGLU + 1x128 FP8 requantize fused into the epilogue; fused-FC1 P2a). The weight, activation and activation-scale
// tensor maps are runGemmSwapAB's (one 128-row weight box, issued twice per stage by the kernel). The output is the
// SwiGLU kernel's pair: dq [shape_n, shape_m / 2] fp8 (shape_n = P_max rows, shape_m = 2I), stored through its own
// tensor map with one block_n x 128 byte box per tile and the 128-byte swizzle (the kernel stages the tile in that
// layout), and sd [shape_m / 256, sd_ld] fp32, which the kernel writes directly. 384 threads, as runGemmSwapAB.
template <typename LayoutIndexType>
void runGemmSwapABSwiglu(cudaKernel_t kernel, void* mat_a, int ld_a, void* mat_b, int ld_b, void* mat_dq, float* mat_sd,
    uint32_t sd_ld, float* scales_a, float* scales_b, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
    uint32_t block_m, uint32_t block_n, uint32_t block_k, uint32_t num_groups, LayoutIndexType* grouped_layout,
    cudaStream_t stream, int num_sms, uint32_t smem_size)
{
    constexpr auto gemm_type = GemmType::GroupedContiguous;
    auto tma_a_desc = make_2d_tma_a_desc_swapAB(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_a), shape_m, shape_k, block_m, block_k, num_groups, gemm_type, ld_a);
    auto tma_b_desc = make_2d_tma_b_desc_swapAB(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_b), shape_n, shape_k, block_n, block_k, num_groups, gemm_type, ld_b);
    auto tma_scales_b_desc
        = make_2d_tma_scales_b_desc_swapAB(scales_b, shape_n, shape_k, block_n, block_k, num_groups, gemm_type);
    uint32_t const shape_i = shape_m / 2;
    auto tma_dq_desc = make_2d_tma_desc(reinterpret_cast<__nv_fp8_e4m3*>(mat_dq), Layout::RowMajor, shape_n, shape_i,
        min(block_n, shape_n), block_m, static_cast<uint64_t>(shape_i), CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);

    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumMathThreadsPerGroup = 128;
    DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

    cudaLaunchConfig_t config;
    config.gridDim = num_sms;
    config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(static_cast<int32_t>(block_m));
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    // No multicast (per-block weights differ); keep the cluster attribute at 1 like the other grouped launches.
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {1, 1, 1};
    config.attrs = &attr;
    config.numAttrs = 1;

    NormalSchedulerInputSwapAB input;
    input.shape_n = shape_n;
    input.grouped_layout = grouped_layout;

    auto status = cudaLaunchKernelEx(&config, kernel, mat_sd, scales_a, input, tma_a_desc, tma_b_desc,
        tma_scales_b_desc, tma_dq_desc, sd_ld);
    DG_HOST_ASSERT(status == cudaSuccess);
}

// Launch of fp8_gemm_kernel_swapAB_swiglu_split (fp8_gemm_impl.cuh): the fused swap-AB FC1 with each SwiGLU block
// shared by the two CTAs of a cluster. Same inputs and outputs as runGemmSwapABSwiglu; the weight box is 64 rows (one
// CTA's half of each 128-row block), the dq box is BLOCK_N x 64 without swizzle (one CTA's columns), and the launch has
// 384 threads per CTA (gate, up and TMA warp-groups) in clusters of two. num_sms must be even.
template <typename LayoutIndexType>
void runGemmSwapABSwigluSplit(cudaKernel_t kernel, void* mat_a, int ld_a, void* mat_b, int ld_b, void* mat_dq,
    float* mat_sd, uint32_t sd_ld, float* scales_a, float* scales_b, uint32_t shape_m, uint32_t shape_n,
    uint32_t shape_k, uint32_t block_n, uint32_t block_k, uint32_t num_groups, LayoutIndexType* grouped_layout,
    cudaStream_t stream, int num_sms, uint32_t smem_size)
{
    constexpr auto gemm_type = GemmType::GroupedContiguous;
    constexpr uint32_t kRowsPerCta = 64;
    DG_HOST_ASSERT(num_sms % 2 == 0);
    auto tma_a_desc = make_2d_tma_a_desc_swapAB(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_a), shape_m, shape_k, kRowsPerCta, block_k, num_groups, gemm_type, ld_a);
    auto tma_b_desc = make_2d_tma_b_desc_swapAB(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_b), shape_n, shape_k, block_n, block_k, num_groups, gemm_type, ld_b);
    auto tma_scales_b_desc
        = make_2d_tma_scales_b_desc_swapAB(scales_b, shape_n, shape_k, block_n, block_k, num_groups, gemm_type);
    uint32_t const shape_i = shape_m / 2;
    auto tma_dq_desc = make_2d_tma_desc(reinterpret_cast<__nv_fp8_e4m3*>(mat_dq), Layout::RowMajor, shape_n, shape_i,
        min(block_n, shape_n), kRowsPerCta, static_cast<uint64_t>(shape_i),
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);

    DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

    cudaLaunchConfig_t config;
    config.gridDim = num_sms;
    config.blockDim = 384;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {2, 1, 1};
    config.attrs = &attr;
    config.numAttrs = 1;

    NormalSchedulerInputSwapAB input;
    input.shape_n = shape_n;
    input.grouped_layout = grouped_layout;

    auto status = cudaLaunchKernelEx(&config, kernel, mat_sd, scales_a, input, tma_a_desc, tma_b_desc,
        tma_scales_b_desc, tma_dq_desc, sd_ld);
    DG_HOST_ASSERT(status == cudaSuccess);
}

}; // namespace deep_gemm

#pragma clang diagnostic pop
