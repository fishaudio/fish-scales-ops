/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

// Arch-agnostic helpers used by the blockscale_gemm kernels.
// Originally lived in blockscale_gemm_kernel.cuh top-of-file.

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cassert>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN
namespace kernel_utils
{

inline void find_divisor(uint32_t& mul, uint32_t& shr, int x)
{
    auto find_log_2 = [](int x, bool round_up = false)
    {
        auto clz = [](int x)
        {
            for (int i = 31; i >= 0; --i)
                if ((1 << i) & x)
                    return 31 - i;
            return 32;
        };
        int a = 31 - clz(x);
        if (round_up)
            a += (x & (x - 1)) ? 1 : 0;
        return a;
    };

    assert(x != 0);
    if (x == 1)
    {
        mul = 0;
        shr = 0;
    }
    else
    {
        uint32_t p = 31 + find_log_2(x, true);
        uint32_t m = static_cast<uint32_t>(((1ull << p) + static_cast<uint32_t>(x) - 1) / static_cast<uint32_t>(x));
        mul = m;
        shr = p - 32;
    }
}

__device__ __forceinline__ void fast_divmod(uint32_t& div, uint32_t& mod, int x, int y, uint32_t mul, uint32_t shr)
{
    if (y == 1)
    {
        div = x;
        mod = 0;
    }
    else
    {
        div = __umulhi(static_cast<uint32_t>(x), mul) >> shr;
        mod = x - div * y;
    }
}

template <typename T>
__inline__ __device__ T warpReduceSum(T val)
{
    constexpr uint32_t FINAL_MASK = 0xffffffff;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

template <>
__inline__ __device__ __nv_bfloat16 warpReduceSum(__nv_bfloat16 val)
{
    constexpr uint32_t FINAL_MASK = 0xffffffff;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = __hmax(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

__inline__ __device__ uint32_t elect_one_sync([[maybe_unused]] int lane_id)
{
    uint32_t pred = 0;
#if __CUDA_ARCH__ >= 900
    uint32_t laneid = 0;
    asm volatile(
        "\n\
    {\n\
        .reg .b32 %rx;\n\
        .reg .pred %px;\n\
        elect.sync %rx|%px, %2;\n\
        @%px mov.s32 %1, 1;\n\
        mov.s32 %0, %rx;\n\
    }\n\
  "
        : "+r"(laneid), "+r"(pred)
        : "r"(0xFFFFFFFF));
#else
    return lane_id == 0;
#endif
    return pred;
}

} // namespace kernel_utils

namespace kernels::blockscale_gemm
{

template <typename T>
__device__ __host__ constexpr T div_up(T a, int b)
{
    return (a + b - 1) / b;
}

template <typename T>
__forceinline__ __device__ T find_max_elem_in_warp(T value)
{
    for (int offset = 16; offset > 0; offset /= 2)
        value = T(std::max(float(value), __shfl_down_sync(0xFFFFFFFF, float(value), offset)));
    value = T(__shfl_sync(0xffffffff, float(value), 0));
    return value;
}

// Number of SMs of the active device, lazily filled.
inline int kNumDeviceSMs = -1;

} // namespace kernels::blockscale_gemm
TRTLLM_NAMESPACE_END
