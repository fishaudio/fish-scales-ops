/*
 * Wrappers around the standalone library's host-side quantize launchers, with
 * plain function signatures (no ATen). Lives in its own TU so we can include
 * the heavy CUTLASS / cute headers without the `cute::Layout` symbol leaking
 * into the ATen-aware ops.cu (which would conflict with `at::Layout`).
 */

#include "blockscale_gemm/common/scale_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace blockscale_gemm
{
namespace detail
{

void fp8bs_quantize_1x128(__nv_fp8_e4m3* x_q, float* scales, __nv_bfloat16 const* x, int M, int K, cudaStream_t stream,
    bool use_ue8m0)
{
    tensorrt_llm::kernels::blockscale_gemm::fp8_1x128_cs(x_q, scales, x, K, M, stream, use_ue8m0);
}

void fp8bs_quantize_128x128(
    __nv_fp8_e4m3* w_q, float* scales, __nv_bfloat16 const* w, int N, int K, cudaStream_t stream)
{
    // The runner-internal weight quant path. NOT fp8_128x128_cs, which is a
    // cast-only placeholder that fills scales with 1.0.
    tensorrt_llm::kernels::blockscale_gemm::fp8_128x128_quant(w_q, scales, w, K, N, stream);
}

// MXFP8 1×32 quantize — restored 2026-05-21 for the fish-scales-ops MXFP8
// GEMM path. The repack helpers below are VS-agnostic and shared with the
// 1×128 FP8 path.

namespace
{

constexpr int kMxFp8VecSize = 32;  // OCP MXFP8 hardware-native granularity

template <bool USE_UE8M0>
__global__ void fp8bs_quantize_1x32_kernel(
    __nv_fp8_e4m3* __restrict__ out, float* __restrict__ scales,
    __nv_bfloat16 const* __restrict__ input, int M, int K)
{
    int const warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int const lane_id = threadIdx.x & 31;
    int const k_blocks = K / kMxFp8VecSize;
    int const total_warps = M * k_blocks;
    if (warp_id >= total_warps) return;

    int const m  = warp_id / k_blocks;
    int const kb = warp_id % k_blocks;
    int const k  = kb * kMxFp8VecSize + lane_id;

    __nv_bfloat16 const x = input[m * K + k];
    float const ax = fabsf(__bfloat162float(x));

    float amax = ax;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float const other = __shfl_xor_sync(0xFFFFFFFFu, amax, off);
        amax = fmaxf(amax, other);
    }
    amax = fmaxf(amax, 1e-10f);

    float quant_scale = 448.f / amax;
    float dequant_scale;
    if constexpr (USE_UE8M0) {
        float const dequant_raw = 1.f / quant_scale;
        __nv_fp8_e8m0 ue8m0_scale;
        ue8m0_scale.__x = __nv_cvt_float_to_e8m0(dequant_raw, __NV_SATFINITE, cudaRoundPosInf);
        dequant_scale = static_cast<float>(ue8m0_scale);
        quant_scale = dequant_scale != 0.f ? 1.f / dequant_scale : 1.f;
    } else {
        dequant_scale = 1.f / quant_scale;
    }

    int const m_pad = ((M + 3) / 4) * 4;
    if (lane_id == 0) {
        scales[kb * m_pad + m] = dequant_scale;
    }

    float const v = __bfloat162float(x) * quant_scale;
    float const v_sat = fmaxf(-448.f, fminf(448.f, v));
    out[m * K + k] = __nv_fp8_e4m3(v_sat);
}

} // anonymous namespace

void fp8bs_quantize_1x32(__nv_fp8_e4m3* x_q, float* scales, __nv_bfloat16 const* x, int M, int K, cudaStream_t stream,
    bool use_ue8m0)
{
    int const k_blocks = K / kMxFp8VecSize;
    int const total_warps = M * k_blocks;
    constexpr int kThreadsPerBlock = 256;
    constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;
    int const grid = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;

    int const m_pad = ((M + 3) / 4) * 4;
    if (m_pad != M) {
        cudaMemsetAsync(scales, 0, sizeof(float) * m_pad * k_blocks, stream);
    }

    if (use_ue8m0)
        fp8bs_quantize_1x32_kernel<true><<<grid, kThreadsPerBlock, 0, stream>>>(x_q, scales, x, M, K);
    else
        fp8bs_quantize_1x32_kernel<false><<<grid, kThreadsPerBlock, 0, stream>>>(x_q, scales, x, M, K);
}

// Repack UE8M0-rounded FP32 dequant scales into the int32-packed layout
// CUTLASS Sm120BlockScaledKernel expects: 4 K-consecutive UE8M0 bytes per
// int32 word, **per-row** along the M (or N) dim (so the SF tensor has
// shape [scale_outer = align(M_or_N, 4), K/512] int32 in K-major).
//
// Two modes:
//   1. SFA (activations): input is already per-row (1 scale per M row,
//      K-major shape [pad(M,4), K/128] from quantize_1x128). Just K-pack.
//   2. SFB (weights): our `quantize_128x128` produces per-128-block
//      scales (1 per (N/128, K/128) tuple). CUTLASS expects per-N-row.
//      We repack with an EXPANSION: each output row n reads from
//      input row (n / 128) and packs 4 K-blocks of that scale.

// Mode 1: per-row pack. src shape [outer_pad, K/128] K-major FP32.
__global__ void repack_ue8m0_per_row_kernel(int32_t* __restrict__ dst,
    float const* __restrict__ src, int outer_pad, int K_blocks_out)
{
    int const o = blockIdx.x * blockDim.x + threadIdx.x;
    int const kp = blockIdx.y * blockDim.y + threadIdx.y;
    if (o >= outer_pad || kp >= K_blocks_out) return;
    int const stride_outer = outer_pad;  // K-major: data[kb * stride + outer]
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int const kb = kp * 4 + i;
        float const s = src[kb * stride_outer + o];
        uint32_t const fbits = __float_as_uint(s);
        uint32_t const e8m0 = (fbits >> 23) & 0xFFu;
        packed |= (e8m0 << (i * 8));
    }
    dst[kp * stride_outer + o] = static_cast<int32_t>(packed);
}

// Mode 2: SFB expand-and-pack. src shape [N/128, K/128] FP32 row-major.
// Out shape [N_pad, K/512] int32 K-major. Out row n reads from
// src[n/128, kp*4..kp*4+3] (the same 128-N-block scale repeats across
// 128 consecutive N rows in the output).
__global__ void repack_ue8m0_expand_n_kernel(int32_t* __restrict__ dst,
    float const* __restrict__ src, int N_pad, int N_blocks_in, int K_blocks_in_per_row)
{
    int const n = blockIdx.x * blockDim.x + threadIdx.x;
    int const kp = blockIdx.y * blockDim.y + threadIdx.y;
    int const K_blocks_out = K_blocks_in_per_row / 4;
    if (n >= N_pad || kp >= K_blocks_out) return;
    int const nb = n / 128;
    uint32_t packed = 0;
    if (nb < N_blocks_in)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int const kb = kp * 4 + i;
            // src is row-major: data[nb * K_blocks_in_per_row + kb]
            float const s = src[nb * K_blocks_in_per_row + kb];
            uint32_t const fbits = __float_as_uint(s);
            uint32_t const e8m0 = (fbits >> 23) & 0xFFu;
            packed |= (e8m0 << (i * 8));
        }
    }
    int const stride_n = N_pad;  // K-major: dst[kp * stride_n + n]
    dst[kp * stride_n + n] = static_cast<int32_t>(packed);
}

void repack_ue8m0_scales_for_sm120(int32_t* dst, float const* src, int M_pad, int K_blocks_in,
    cudaStream_t stream)
{
    int const K_blocks_out = K_blocks_in / 4;
    dim3 const block(32, 4, 1);
    dim3 const grid((M_pad + block.x - 1) / block.x,
                    (K_blocks_out + block.y - 1) / block.y, 1);
    repack_ue8m0_per_row_kernel<<<grid, block, 0, stream>>>(dst, src, M_pad, K_blocks_out);
}

void repack_ue8m0_scales_sfb_for_sm120(int32_t* dst, float const* src, int N_pad,
    int N_blocks_in, int K_blocks_in_per_row, cudaStream_t stream)
{
    int const K_blocks_out = K_blocks_in_per_row / 4;
    dim3 const block(32, 4, 1);
    dim3 const grid((N_pad + block.x - 1) / block.x,
                    (K_blocks_out + block.y - 1) / block.y, 1);
    repack_ue8m0_expand_n_kernel<<<grid, block, 0, stream>>>(dst, src, N_pad, N_blocks_in,
        K_blocks_in_per_row);
}

} // namespace detail
} // namespace blockscale_gemm
