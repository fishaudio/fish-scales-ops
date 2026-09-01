/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Quantization helper kernels: BF16 -> FP8 with per-token (1x128) or per-block
// (128x128) dequant scales. Used by the BF16-input runner instantiations and
// also exposed as standalone functions through the runner API.
//
// Originally lived in blockscale_gemm_kernel.cuh.

#pragma once

#include "blockscale_gemm/common/kernel_utils.cuh"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN
namespace kernels::blockscale_gemm
{

template <typename InputType, typename OutputType, typename ScaleType = float, bool USE_UE8M0 = false>
__global__ void scale_1x128_kernel(
    OutputType* output, ScaleType* scales, InputType const* const input, int dim_x, int dim_y)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890))
    size_t scales_along_dim_x = div_up(dim_x, 128);
    size_t scales_along_dim_y = div_up(dim_y, 1);
    size_t stride_scale_dim_y = div_up(dim_y, 4) * 4;
    using Input2Type = typename std::conditional<std::is_same<InputType, half>::value, half2, __nv_bfloat162>::type;
    for (size_t warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
         warp_idx < scales_along_dim_x * scales_along_dim_y; warp_idx += gridDim.x * blockDim.x / 32)
    {
        int scales_idx_y = warp_idx / scales_along_dim_x;
        int scales_idx_x = warp_idx % scales_along_dim_x;

        InputType const* input_line = input + (size_t) scales_idx_y * dim_x + scales_idx_x * 128;
        InputType input_amax = InputType(0);
        int lane_id = threadIdx.x % 32 * 2;

        Input2Type input_frag2[2] = {Input2Type(0, 0), Input2Type(0, 0)};
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x)
                break;
            input_frag2[i] = *((Input2Type*) (input_line) + lane_id / 2);
            input_line += 64;
        }
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x)
                break;
            input_amax = InputType(__hmax(input_amax, __hmax(__habs(input_frag2[i].x), __habs(input_frag2[i].y))));
        }

        InputType amax = find_max_elem_in_warp(input_amax);
        amax = tensorrt_llm::common::cuda_max(amax, InputType(1e-10f));
        ScaleType quant_scale = 448.f / ScaleType(amax);
        ScaleType dequant_scale;

        if constexpr (USE_UE8M0)
        {
            ScaleType dequant_scale_raw = 1.f / quant_scale;
            __nv_fp8_e8m0 ue8m0_scale;
            ue8m0_scale.__x = __nv_cvt_float_to_e8m0(float(dequant_scale_raw), __NV_SATFINITE, cudaRoundPosInf);
            dequant_scale = ScaleType(static_cast<float>(ue8m0_scale));
            quant_scale = dequant_scale != ScaleType(0.f) ? 1.f / dequant_scale : 1.f;
        }
        else
        {
            dequant_scale = 1.f / quant_scale;
        }

        if (lane_id == 0)
            scales[(size_t) scales_idx_x * stride_scale_dim_y + scales_idx_y] = dequant_scale;

        OutputType* output_line = output + (size_t) scales_idx_y * dim_x + scales_idx_x * 128;
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x)
                break;
            ScaleType value_1 = ScaleType(input_frag2[i].x) * quant_scale;
            ScaleType value_2 = ScaleType(input_frag2[i].y) * quant_scale;
            output_line[lane_id] = OutputType(value_1);
            output_line[lane_id + 1] = OutputType(value_2);
            output_line += 64;
        }
    }
#endif
}

// Grouped MoE variant removed 2026-05-21.

// input: [dim_y, dim_h, dim_x]
// output: [dim_h, dim_y, dim_x], cs[dim_h, dim_x/128, padding(dim_y)]
template <typename InputType, typename OutputType, typename ScaleType = float>
__global__ void scale_1x128_reshape_kernel(
    OutputType* output, ScaleType* scales, InputType const* const input, int dim_x, int dim_h, int dim_y, int stride_x)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890))
    size_t scales_along_dim_x = div_up(dim_x, 128);
    size_t scales_along_dim_y = div_up(dim_y, 1);
    size_t scales_along_dim_h = div_up(dim_h, 1);
    size_t stride_scale_dim_y = div_up(dim_y, 4) * 4;
    using Input2Type = typename std::conditional<std::is_same<InputType, half>::value, half2, __nv_bfloat162>::type;
    for (size_t warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
         warp_idx < scales_along_dim_x * scales_along_dim_y * scales_along_dim_h;
         warp_idx += gridDim.x * blockDim.x / 32)
    {
        int scales_idx_y = warp_idx / (scales_along_dim_x * scales_along_dim_h);
        int scales_idx_h = (warp_idx % (scales_along_dim_x * scales_along_dim_h)) / scales_along_dim_x;
        int scales_idx_x = warp_idx % scales_along_dim_x;

        InputType const* input_line
            = input + (size_t) scales_idx_y * stride_x * dim_h + (size_t) scales_idx_h * stride_x + scales_idx_x * 128;
        InputType input_amax = InputType(0);
        int lane_id = threadIdx.x % 32 * 2;

        Input2Type input_frag2[2] = {Input2Type(0, 0), Input2Type(0, 0)};
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x)
                break;
            input_frag2[i] = *((Input2Type*) (input_line) + lane_id / 2);
            input_line += 64;
        }
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x)
                break;
            input_amax = InputType(__hmax(input_amax, __hmax(__habs(input_frag2[i].x), __habs(input_frag2[i].y))));
        }

        InputType amax = find_max_elem_in_warp(input_amax);
        amax = tensorrt_llm::common::cuda_max(amax, InputType(1e-10f));
        ScaleType scale = 448.f / ScaleType(amax);

        if (lane_id == 0)
        {
            scales[(size_t) scales_idx_h * scales_along_dim_x * stride_scale_dim_y
                + (size_t) scales_idx_x * stride_scale_dim_y + scales_idx_y]
                = ScaleType(1.f / scale);
        }

        OutputType* output_line
            = output + (size_t) scales_idx_h * dim_y * dim_x + (size_t) scales_idx_y * dim_x + scales_idx_x * 128;
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x)
                break;
            ScaleType value_1 = ScaleType(input_frag2[i].x) * scale;
            ScaleType value_2 = ScaleType(input_frag2[i].y) * scale;
            output_line[lane_id] = OutputType(value_1);
            output_line[lane_id + 1] = OutputType(value_2);
            output_line += 64;
        }
    }
#endif
}

template <typename InputType, typename OutputType, typename ScaleType = float, bool USE_UE8M0 = false>
__global__ void scale_128x128_kernel(
    OutputType* output, ScaleType* scales, InputType const* const input, int dim_x, int dim_y)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    int scales_along_dim_x = div_up(dim_x, 128);
    int scales_along_dim_y = div_up(dim_y, 128);

    for (int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
         warp_idx < scales_along_dim_x * scales_along_dim_y; warp_idx += gridDim.x * blockDim.x / 32)
    {
        int scales_idx_y = warp_idx / scales_along_dim_x;
        int scales_idx_x = warp_idx % scales_along_dim_x;

        InputType const* input_line = input + scales_idx_y * 128 * dim_x + scales_idx_x * 128;
        InputType input_amax = InputType(0);
        int lane_id = threadIdx.x % 32;

        for (int i = 0; i < 128; i++)
        {
            if (scales_idx_y * 128 + i >= dim_y)
                break;
            InputType const* input_d = input_line;
            for (int j = 0; j < 4; j++)
            {
                if (scales_idx_x * 128 + j * 32 + lane_id >= dim_x)
                    break;
                input_amax = InputType(std::max(float(input_amax), std::fabs(float(input_d[lane_id]))));
                input_d += 32;
            }
            input_line += dim_x;
        }

        InputType amax = find_max_elem_in_warp(input_amax);
        amax = tensorrt_llm::common::cuda_max(amax, InputType(1e-10f));
        ScaleType quant_scale = 448.f / ScaleType(amax);
        ScaleType dequant_scale;
        if constexpr (USE_UE8M0)
        {
            ScaleType dequant_scale_raw = 1.f / quant_scale;
            __nv_fp8_e8m0 ue8m0_scale;
            ue8m0_scale.__x = __nv_cvt_float_to_e8m0(float(dequant_scale_raw), __NV_SATFINITE, cudaRoundPosInf);
            dequant_scale = ScaleType(static_cast<float>(ue8m0_scale));
            quant_scale = dequant_scale != ScaleType(0.f) ? 1.f / dequant_scale : 1.f;
        }
        else
        {
            dequant_scale = 1.f / quant_scale;
        }

        if (lane_id == 0)
            scales[scales_idx_y * scales_along_dim_x + scales_idx_x] = dequant_scale;

        input_line = input + scales_idx_y * 128 * dim_x + scales_idx_x * 128;
        OutputType* output_line = output + scales_idx_y * 128 * dim_x + scales_idx_x * 128;

        for (int i = 0; i < 128; i++)
        {
            if (scales_idx_y * 128 + i >= dim_y)
                break;
            InputType const* input_d = input_line;
            OutputType* output_d = output_line;
            for (int j = 0; j < 4; j++)
            {
                if (scales_idx_x * 128 + j * 32 + lane_id >= dim_x)
                    break;
                output_d[lane_id] = OutputType(ScaleType(input_d[lane_id]) * quant_scale);
                input_d += 32;
                output_d += 32;
            }
            input_line += dim_x;
            output_line += dim_x;
        }
    }
#endif
}

template <typename OutputType>
__global__ void fill_kernel(OutputType* output, size_t num_elems, float value)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elems; idx += gridDim.x * blockDim.x)
        output[idx] = OutputType(value);
}

template <typename InputType, typename OutputType>
__global__ void convert_kernel(OutputType* output, InputType const* const input, size_t num_elems)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elems; idx += gridDim.x * blockDim.x)
    {
        float value = float(input[idx]);
        if (std::isnan(value))
            output[idx] = OutputType(448);
        else
            output[idx] = OutputType(value);
    }
}

// Host launchers used by the runner.
inline void fp8_1x128_cs(__nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x, int shape_y,
    cudaStream_t stream, bool use_ue8m0 = false)
{
    if (kNumDeviceSMs < 0)
        kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    if (use_ue8m0)
    {
        scale_1x128_kernel<__nv_bfloat16, __nv_fp8_e4m3, float, true>
            <<<kNumDeviceSMs * 8, 256, 0, stream>>>(mat_quant, scales, mat, shape_x, shape_y);
    }
    else
    {
        scale_1x128_kernel<__nv_bfloat16, __nv_fp8_e4m3, float, false>
            <<<kNumDeviceSMs * 8, 256, 0, stream>>>(mat_quant, scales, mat, shape_x, shape_y);
    }
}

inline void fp8_1x128_cs_reshape(__nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x,
    int shape_h, int shape_y, int stride_x, cudaStream_t stream)
{
    if (kNumDeviceSMs < 0)
        kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    scale_1x128_reshape_kernel<<<kNumDeviceSMs * 8, 256, 0, stream>>>(
        mat_quant, scales, mat, shape_x, shape_h, shape_y, stride_x);
}

inline void fp8_128x128_cs(
    __nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x, int shape_y, cudaStream_t stream)
{
    if (kNumDeviceSMs < 0)
        kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    convert_kernel<<<kNumDeviceSMs, 256, 0, stream>>>(mat_quant, mat, shape_x * shape_y);
    fill_kernel<<<kNumDeviceSMs, 256, 0, stream>>>(scales, div_up(shape_x, 128) * div_up(shape_y, 128), 1);
}

// Proper per-block-scaled BF16 -> FP8 quantize (the one the runner uses
// internally for weights). Computes amax per 128x128 block, scales to E4M3,
// writes dequant scales to `scales`.
inline void fp8_128x128_quant(
    __nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x, int shape_y, cudaStream_t stream)
{
    if (kNumDeviceSMs < 0)
        kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    scale_128x128_kernel<__nv_bfloat16, __nv_fp8_e4m3, float>
        <<<kNumDeviceSMs, 256, 0, stream>>>(mat_quant, scales, mat, shape_x, shape_y);
}

} // namespace kernels::blockscale_gemm
TRTLLM_NAMESPACE_END
