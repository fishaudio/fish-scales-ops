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

// 1x128 activation quantize (BF16 -> FP8 E4M3 with one FP32 dequant scale per row and 128-element group), with each
// group spread over 8 lanes that load 16 elements each (two 16-byte loads), so a warp quantizes four groups at once
// and every lane does one load, a three-step shuffle reduction and one 16-byte store per group. This is the
// organisation of sglang's per_token_group_quant kernel. scale_1x128_kernel above (one warp per group, four
// elements per lane, five shuffles) and the K % 512 kernel of the quantize op (one warp per four consecutive groups,
// walked one after the other) carry a serial chain per thread that bounds them by latency at small M.
//
// The arithmetic is the one both of those kernels perform, so the outputs are bitwise identical to theirs: amax is
// the exact maximum of |x| over the group (NaN elements drop out of the maximum in all three), floored at 1e-10 (the
// quantize op's K % 512 kernel floors at 1e-10f; scale_1x128_kernel floors at the BF16 value nearest to it,
// 1.00044e-10, which kBf16AmaxFloor selects), quant scale qs = 448.f / amax, dequant scale 1.f / qs, and each element
// x * qs converted with round-to-nearest and saturation to +-448. Scales are K-major: scales[kb * align(M, 4) + m].
// kZeroPadRows also writes 0 into the scales of the padding rows M .. align(M, 4) - 1, as the quantize op's K % 512
// kernel does; scale_1x128_kernel leaves them unwritten.
//
// Requirements (checked by fp8_1x128_lanes_supported): K a multiple of 128, input and output 16-byte aligned.
// The grid is at most one wave (the CTAs that fit on the device at once) and loops over the groups, so every CTA gets
// the same share at large M without a partial second wave, and a programmatic dependent launch (pdl) never floods
// the SMs with waiting CTAs. With pdl the kernel waits for its predecessor before its first global access and then
// lets its own dependent (the GEMM) launch, so the GEMM's launch and prologue overlap the quantize.
template <bool kBf16AmaxFloor, bool kZeroPadRows>
__global__ void __launch_bounds__(256) scale_1x128_lanes_kernel(__nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scales, __nv_bfloat16 const* __restrict__ input, int dim_x, int dim_y, bool pdl)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890))
#if (__CUDA_ARCH__ >= 900)
    if (pdl)
    {
        cudaGridDependencySynchronize();
        if (threadIdx.x == 0)
            cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
    int64_t const k_groups = dim_x / 128;
    int64_t const m_pad = (static_cast<int64_t>(dim_y) + 3) / 4 * 4;
    int64_t const num_groups = (kZeroPadRows ? m_pad : static_cast<int64_t>(dim_y)) * k_groups;
    unsigned const lane = threadIdx.x & 31u;
    unsigned const sub = lane & 7u;                // lane within the group's eight lanes
    unsigned const sub_mask = 0xFFu << (lane & 24u); // the eight lanes of this group
    float const amax_floor = kBf16AmaxFloor ? __bfloat162float(__float2bfloat16(1e-10f)) : 1e-10f;

    int64_t const stride = (static_cast<int64_t>(gridDim.x) * blockDim.x) >> 3;
    for (int64_t g = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 3; g < num_groups; g += stride)
    {
        int64_t const m = g / k_groups;
        int64_t const kb = g - m * k_groups;
        if (kZeroPadRows && m >= dim_y)
        {
            if (sub == 0)
                scales[kb * m_pad + m] = 0.f;
            continue;
        }

        int64_t const offset = m * dim_x + kb * 128 + sub * 16;
        uint4 const v[2] = {*reinterpret_cast<uint4 const*>(input + offset),
            *reinterpret_cast<uint4 const*>(input + offset + 8)};
        float x[16];
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            __nv_bfloat162 const* h = reinterpret_cast<__nv_bfloat162 const*>(&v[i]);
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float2 const f = __bfloat1622float2(h[j]);
                x[i * 8 + j * 2 + 0] = f.x;
                x[i * 8 + j * 2 + 1] = f.y;
            }
        }

        float amax = fabsf(x[0]);
#pragma unroll
        for (int i = 1; i < 16; ++i)
            amax = fmaxf(amax, fabsf(x[i]));
        amax = fmaxf(amax, __shfl_xor_sync(sub_mask, amax, 4));
        amax = fmaxf(amax, __shfl_xor_sync(sub_mask, amax, 2));
        amax = fmaxf(amax, __shfl_xor_sync(sub_mask, amax, 1));
        amax = fmaxf(amax, amax_floor);

        float const quant_scale = 448.f / amax;
        uint32_t words[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            __nv_fp8x2_storage_t const lo = __nv_cvt_float2_to_fp8x2(
                make_float2(x[i * 4 + 0] * quant_scale, x[i * 4 + 1] * quant_scale), __NV_SATFINITE, __NV_E4M3);
            __nv_fp8x2_storage_t const hi = __nv_cvt_float2_to_fp8x2(
                make_float2(x[i * 4 + 2] * quant_scale, x[i * 4 + 3] * quant_scale), __NV_SATFINITE, __NV_E4M3);
            words[i] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
        }
        *reinterpret_cast<uint4*>(output + offset) = make_uint4(words[0], words[1], words[2], words[3]);
        if (sub == 0)
            scales[kb * m_pad + m] = 1.f / quant_scale;
    }
#endif
}

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

// Whether scale_1x128_lanes_kernel can quantize this tensor (K a multiple of 128, 16-byte aligned rows and bases).
inline bool fp8_1x128_lanes_supported(__nv_fp8_e4m3 const* mat_quant, __nv_bfloat16 const* mat, int shape_x)
{
    return shape_x % 128 == 0 && reinterpret_cast<uintptr_t>(mat_quant) % 16 == 0
        && reinterpret_cast<uintptr_t>(mat) % 16 == 0;
}

// Resident CTAs per SM of one scale_1x128_lanes_kernel instantiation (256 threads), queried once per process.
template <bool kBf16AmaxFloor, bool kZeroPadRows>
inline int fp8_1x128_lanes_ctas_per_sm()
{
    static int const v = []
    {
        int n = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &n, scale_1x128_lanes_kernel<kBf16AmaxFloor, kZeroPadRows>, 256, 0)
                != cudaSuccess
            || n < 1)
        {
            (void) cudaGetLastError();
            n = 1;
        }
        return n;
    }();
    return v;
}

// Launches scale_1x128_lanes_kernel (FP32 dequant scales, no UE8M0 rounding). bf16_amax_floor selects the 1e-10
// floor of scale_1x128_kernel (true) or of the quantize op's K % 512 kernel (false); zero_pad_rows writes 0 into the
// padding rows' scales. pdl launches with programmatic stream serialization.
inline void fp8_1x128_lanes(__nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x,
    int shape_y, cudaStream_t stream, bool bf16_amax_floor, bool zero_pad_rows, bool pdl)
{
    if (kNumDeviceSMs < 0)
        kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    int64_t const m_pad = (static_cast<int64_t>(shape_y) + 3) / 4 * 4;
    int64_t const groups = (zero_pad_rows ? m_pad : static_cast<int64_t>(shape_y)) * (shape_x / 128);
    constexpr int kThreads = 256;
    constexpr int kGroupsPerCta = kThreads / 8;
    int const per_sm = bf16_amax_floor
        ? (zero_pad_rows ? fp8_1x128_lanes_ctas_per_sm<true, true>() : fp8_1x128_lanes_ctas_per_sm<true, false>())
        : (zero_pad_rows ? fp8_1x128_lanes_ctas_per_sm<false, true>() : fp8_1x128_lanes_ctas_per_sm<false, false>());
    int64_t const max_ctas = static_cast<int64_t>(kNumDeviceSMs) * per_sm;
    int64_t ctas = (groups + kGroupsPerCta - 1) / kGroupsPerCta;
    ctas = ctas < 1 ? 1 : (ctas > max_ctas ? max_ctas : ctas);

    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.gridDim = dim3(static_cast<unsigned>(ctas));
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = pdl ? 1 : 0;
    auto kernel = bf16_amax_floor
        ? (zero_pad_rows ? scale_1x128_lanes_kernel<true, true> : scale_1x128_lanes_kernel<true, false>)
        : (zero_pad_rows ? scale_1x128_lanes_kernel<false, true> : scale_1x128_lanes_kernel<false, false>);
    check_cuda_error(cudaLaunchKernelEx(&cfg, kernel, mat_quant, scales, mat, shape_x, shape_y, pdl));
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
// `use_ue8m0` rounds each block scale up to a power of two (UE8M0-exact FP32),
// which the sm_120 path requires: its scale repack keeps only the exponent byte,
// so a plain amax/448 scale would be silently truncated to the power of two
// below it and every block dequantised 0.5-1.0x too small (2026-09-05 fix).
inline void fp8_128x128_quant(__nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x,
    int shape_y, cudaStream_t stream, bool use_ue8m0 = false)
{
    if (kNumDeviceSMs < 0)
        kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    if (use_ue8m0)
        scale_128x128_kernel<__nv_bfloat16, __nv_fp8_e4m3, float, /*USE_UE8M0=*/true>
            <<<kNumDeviceSMs, 256, 0, stream>>>(mat_quant, scales, mat, shape_x, shape_y);
    else
        scale_128x128_kernel<__nv_bfloat16, __nv_fp8_e4m3, float, /*USE_UE8M0=*/false>
            <<<kNumDeviceSMs, 256, 0, stream>>>(mat_quant, scales, mat, shape_x, shape_y);
}

} // namespace kernels::blockscale_gemm
TRTLLM_NAMESPACE_END
