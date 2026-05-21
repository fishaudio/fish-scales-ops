/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SM90 (Hopper / H100 / H200) dispatch for fp8 blockscale GEMM. Uses
// DeepSeek's deep_gemm JIT pipeline (NVRTC-compiled WGMMA kernels). The
// JIT compiler resolves <deep_gemm/...> via include paths configured at
// build time (FSO_JIT_INCLUDE_DIRS_DEFAULT) or env var
// FSO_JIT_INCLUDE_DIRS.

#pragma once

#include "blockscale_gemm/common/kernel_utils.cuh"
#include "blockscale_gemm/arch/sm90/fp8/jit/deep_gemm/fp8_gemm.cuh"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

TRTLLM_NAMESPACE_BEGIN
namespace kernels::blockscale_gemm
{

inline void gemm_dispatch_sm90(void* mat_a, int ld_a, void* mat_b, int ld_b, void* mat_d, int ld_d, float* scales_a,
    float* scales_b, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, cudaStream_t stream,
    int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

    constexpr uint32_t block_k = 128;
    constexpr uint32_t num_problems = 1;

    if (shape_m >= 32u)
    {
        auto [bm, bn, ns, ntm, smem]
            = deep_gemm::jit::get_best_gemm_config(shape_m, shape_n, shape_k, num_problems, num_device_sms);
        auto runtime = deep_gemm::jit::getGlobalCompiler().build(
            shape_n, shape_k, bm, bn, block_k, num_problems, ns, ntm, deep_gemm::GemmType::Normal);
        auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
        deep_gemm::runGemm(kernel, mat_a, ld_a, mat_b, ld_b, mat_d, ld_d, scales_a, scales_b, shape_m, shape_n, shape_k,
            bm, bn, block_k, num_problems, ntm, deep_gemm::GemmType::Normal, static_cast<int*>(nullptr), stream,
            num_device_sms, static_cast<uint32_t>(smem));
    }
    else
    {
        // Tiny M: swap A/B inside the kernel so the persistent scheduler still
        // has enough tiles along the now-M axis.
        auto [bm, bn, ns, ntm, smem]
            = deep_gemm::jit::get_best_gemm_config(shape_n, shape_m, shape_k, num_problems, num_device_sms, false, true);
        auto runtime = deep_gemm::jit::getGlobalCompiler().build(
            shape_n, shape_k, bm, bn, block_k, num_problems, ns, ntm, deep_gemm::GemmType::Normal, true);
        auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
        deep_gemm::runGemmSwapAB(kernel, mat_b, ld_b, mat_a, ld_a, mat_d, ld_d, scales_b, scales_a, shape_n, shape_m,
            shape_k, bm, bn, block_k, num_problems, ntm, deep_gemm::GemmType::Normal, static_cast<int*>(nullptr),
            stream, num_device_sms, static_cast<uint32_t>(smem));
    }
}

} // namespace kernels::blockscale_gemm
TRTLLM_NAMESPACE_END
