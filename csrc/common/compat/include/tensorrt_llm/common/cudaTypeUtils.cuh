/*
 * Standalone shim for tensorrt_llm/common/cudaTypeUtils.cuh
 *
 * Only the symbols actually referenced by blockscale_gemm
 * (cuda_max<T>(T, T)) are provided.
 */
#pragma once

#include "tensorrt_llm/common/config.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN
namespace common
{

template <typename T>
__device__ __host__ inline T cuda_max(T a, T b)
{
    return (a > b) ? a : b;
}

template <typename T>
__device__ __host__ inline T cuda_min(T a, T b)
{
    return (a < b) ? a : b;
}

} // namespace common
TRTLLM_NAMESPACE_END
