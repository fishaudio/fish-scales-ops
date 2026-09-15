/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SM120 BlockScaled FP8 scale-repack helpers.
//
// The CUTLASS Sm120BlockScaledKernel reads scale-factor tensors as
// **int32-packed UE8M0** — 4 UE8M0 bytes per int32 word, one row per M
// (or N), K-major layout. The runner's internal FP32 scale buffer
// (output of `scale_1x128_kernel` / `scale_128x128_kernel`) needs to be
// repacked into that layout before the SM120 GEMM can consume it,
// otherwise the kernel reads garbage and returns NaN.
//
// Two modes:
//   1. SFA (activations): input already per-row (1 scale per M row,
//      K-major shape [pad(M,4), K/128]). Just K-pack 4 bytes per int32.
//   2. SFB (weights): input is per-128-N-block (1 scale per
//      (N/128, K/128) tile). CUTLASS expects per-N-row. Repack with
//      an EXPANSION: each output row n reads from input row (n/128)
//      and packs 4 K-blocks of that scale.
//
// Both kernels assume the input FP32 has been UE8M0-quantized (its
// mantissa bits are zero so extracting the exponent byte is lossless).
// For non-UE8M0 input the repack effectively rounds DOWN to the nearest
// power of 2 (since mantissa is dropped) — usable but lossy.
//
// Mirrors the PyTorch wrapper's `repack_ue8m0_scales_*` helpers in
// `pytorch/csrc/ops/quant_kernels.cu`; kept in lockstep so the
// runner-internal path matches the pre-quantized `linear_fp8` path
// numerically.

#pragma once

#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN
namespace kernels::blockscale_gemm
{

namespace detail
{

// Per-row pack: src shape [outer_pad, K/128] K-major FP32.
// Out shape [outer_pad, ceil(K/512)] K-major int32 (4 K-consecutive bytes
// packed). When K/128 is not a multiple of 4 the last word's missing bytes
// are zero (UE8M0 0 = 2^-127); the kernel never consumes them because its K
// loop stops at the real k-tile count (partial final round, 2026-09-05).
__global__ inline void sm120_repack_ue8m0_per_row_kernel(
    int32_t* __restrict__ dst, float const* __restrict__ src, int outer_pad, int K_blocks_in, int K_blocks_out)
{
    int const o = blockIdx.x * blockDim.x + threadIdx.x;
    int const kp = blockIdx.y * blockDim.y + threadIdx.y;
    if (o >= outer_pad || kp >= K_blocks_out) return;
    int const stride_outer = outer_pad;
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int const kb = kp * 4 + i;
        if (kb < K_blocks_in)
        {
            float const s = src[kb * stride_outer + o];
            uint32_t const fbits = __float_as_uint(s);
            uint32_t const e8m0 = (fbits >> 23) & 0xFFu;
            packed |= (e8m0 << (i * 8));
        }
    }
    dst[kp * stride_outer + o] = static_cast<int32_t>(packed);
}

// SFB expand-and-pack: src shape [N/128, K/128] FP32 row-major.
// Out shape [N_pad, ceil(K/512)] int32 K-major (tail bytes zero, see above).
__global__ inline void sm120_repack_ue8m0_expand_n_kernel(
    int32_t* __restrict__ dst, float const* __restrict__ src, int N_pad, int N_blocks_in, int K_blocks_in_per_row)
{
    int const n = blockIdx.x * blockDim.x + threadIdx.x;
    int const kp = blockIdx.y * blockDim.y + threadIdx.y;
    int const K_blocks_out = (K_blocks_in_per_row + 3) / 4;
    if (n >= N_pad || kp >= K_blocks_out) return;
    int const nb = n / 128;
    uint32_t packed = 0;
    if (nb < N_blocks_in)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int const kb = kp * 4 + i;
            if (kb >= K_blocks_in_per_row)
                break;
            float const s = src[nb * K_blocks_in_per_row + kb];
            uint32_t const fbits = __float_as_uint(s);
            uint32_t const e8m0 = (fbits >> 23) & 0xFFu;
            packed |= (e8m0 << (i * 8));
        }
    }
    int const stride_n = N_pad;
    dst[kp * stride_n + n] = static_cast<int32_t>(packed);
}

} // namespace detail

// Host launchers. `stream` is the CUDA stream to issue work on.

inline void sm120_repack_sfa(
    int32_t* dst, float const* src, int M_pad, int K_blocks_in, cudaStream_t stream)
{
    int const K_blocks_out = (K_blocks_in + 3) / 4;
    dim3 const block(32, 4, 1);
    dim3 const grid((M_pad + block.x - 1) / block.x, (K_blocks_out + block.y - 1) / block.y, 1);
    detail::sm120_repack_ue8m0_per_row_kernel<<<grid, block, 0, stream>>>(dst, src, M_pad, K_blocks_in, K_blocks_out);
}

inline void sm120_repack_sfb(
    int32_t* dst, float const* src, int N_pad, int N_blocks_in, int K_blocks_in_per_row, cudaStream_t stream)
{
    int const K_blocks_out = (K_blocks_in_per_row + 3) / 4;
    dim3 const block(32, 4, 1);
    dim3 const grid((N_pad + block.x - 1) / block.x, (K_blocks_out + block.y - 1) / block.y, 1);
    detail::sm120_repack_ue8m0_expand_n_kernel<<<grid, block, 0, stream>>>(
        dst, src, N_pad, N_blocks_in, K_blocks_in_per_row);
}

} // namespace kernels::blockscale_gemm
TRTLLM_NAMESPACE_END
