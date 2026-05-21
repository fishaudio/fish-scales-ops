/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SM120 FP8 / MXFP8 GEMV for very-small M (M ∈ {1, 2, 3, 4}).
//
// **NEGATIVE-RESULT NOTE (E15, 2026-05-10)**: this kernel is *correct*
// (cos ≥ 0.99928 across M ∈ {1..4} × 3 production shapes vs the matmul
// path) but **slower** than the CUTLASS matmul kernel it was meant to
// replace. The intuition behind writing a GEMV was right — at very
// small M the matmul kernel wastes 31/32 of its M dim and we are
// memory-bandwidth bound on B. The intuition that was wrong is the
// throughput-per-arithmetic-unit gap on Blackwell consumer:
//
//   - sm_120 mxf8f6f4 tensor-core peak ≈ 974 TF
//   - sm_120 FP32 scalar ALU peak       ≈ 21 TF
//   - Ratio ≈ 46×
//
// So even with M-row waste of 32× in the matmul, tensor cores at
// 974/32 ≈ 30 TF still beat scalar ALU at 21 TF. End-to-end on
// `down` M=1 K=9728 the GEMV measured 50 µs vs the matmul path's
// 22 µs after E8/E9/E14 — a hard regression.
//
// The kernel is kept for reference and to document the dead end; the
// dispatcher does NOT route to it. Two paths forward if anyone wants
// to revisit:
//   1. Use tensor cores from the GEMV: pad M to 16, mask the unused
//      rows in the epilogue. That's essentially what the matmul kernel
//      already does, just with more boilerplate.
//   2. Capture decode forward passes into a CUDA Graph (eliminates
//      per-launch host overhead, which is the real M=1 latency floor
//      after E8/E9 already trimmed ~4 µs of host work).
//
// This kernel is a straight GEMV:
//   - 1 warp per output (N-row).
//   - All warps in the CTA cooperatively load A (M·K FP8 bytes) into smem
//     once and broadcast — A reads are amortised across all 8 warps.
//   - Each warp streams its B row from gmem, dequantises with the int32-
//     packed UE8M0 scale factors, and writes M BF16 outputs.
//   - No TMA descriptors. No CUTLASS template. Plain `cudaLaunchKernel`.
//
// Templated on `USE_MXFP8`:
//   - false (FP8 1×128): one UE8M0 byte per 128 K-elements.
//   - true  (MXFP8 1×32): one UE8M0 byte per 32 K-elements.
//
// Both layouts pack 4 UE8M0 bytes per int32 word; SFA stride along M is
// `pad(M, 4)`, SFB stride along N is `pad(N, 4)`. SFB is already expanded
// to per-N-row in the matmul path (by `sm120_repack_sfb`) and the GEMV
// reads it in the same shape, so the two paths can share the int32-packed
// buffers without any conversion.

#pragma once

#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN
namespace kernels::blockscale_gemm
{

namespace detail
{

// Decode a UE8M0 exponent byte into FP32 — UE8M0 has 8 exponent bits + 0
// mantissa bits, i.e. the byte IS the FP32 exponent field. Reconstruct by
// shifting into position. Byte 0xFF (e = 255) yields +inf which is fine
// — the upstream repack guarantees finite scales.
__device__ __forceinline__ float ue8m0_byte_to_fp32(uint8_t b)
{
    return __uint_as_float(static_cast<uint32_t>(b) << 23);
}

constexpr int kSmallMGemvMaxM = 4;
constexpr int kSmallMGemvWarpsPerBlock = 8;
constexpr int kSmallMGemvBlockThreads = kSmallMGemvWarpsPerBlock * 32;

template <bool USE_MXFP8>
__global__ void sm120_fp8_gemv_kernel(__nv_bfloat16* __restrict__ out,
    __nv_fp8_e4m3 const* __restrict__ mat_a, __nv_fp8_e4m3 const* __restrict__ mat_b,
    int32_t const* __restrict__ packed_sfa, int32_t const* __restrict__ packed_sfb,
    int M, int N, int K)
{
    constexpr int kVecSize = USE_MXFP8 ? 32 : 128;   // K-elements per UE8M0 byte
    constexpr int kElemsPerLane = kVecSize / 32;     // FP8: 4, MXFP8: 1
    constexpr int kWarpsPerBlock = kSmallMGemvWarpsPerBlock;

    int const tid = threadIdx.x;
    int const warp_id = tid / 32;
    int const lane = tid & 31;
    int const n = blockIdx.x * kWarpsPerBlock + warp_id;

    extern __shared__ __nv_fp8_e4m3 smem_A[];
    int const total_A = M * K;
    {
        // Vectorised 8-byte cooperative load (K is always divisible by 128 so
        // total_A divides 8 cleanly).
        auto* smem_A_u8 = reinterpret_cast<unsigned long long*>(smem_A);
        auto const* gmem_A_u8 = reinterpret_cast<unsigned long long const*>(mat_a);
        int const total_8B = total_A / 8;
        for (int i = tid; i < total_8B; i += blockDim.x)
            smem_A_u8[i] = gmem_A_u8[i];
    }
    __syncthreads();

    if (n >= N)
        return;

    int const M_pad = (M + 3) & ~3;
    int const N_pad = (N + 3) & ~3;
    int const num_sub_blocks = K / kVecSize;

    // Per-lane partial accumulator per M row. The KEY trick: the per-sub-
    // block scale `scale_a[m] * scale_b` is a scalar (same for all 32 lanes
    // in a warp), so we can pre-multiply each lane's a·b contribution by
    // the scale BEFORE summing — moving the only warp-wide reduce to the
    // very end of the K loop instead of doing one per sub-block × per M.
    // That collapses 304 × 4 = 1216 warp-reduces (MXFP8 K=9728) to just 4.
    float acc[kSmallMGemvMaxM];
#pragma unroll
    for (int m = 0; m < kSmallMGemvMaxM; ++m)
        acc[m] = 0.f;

    for (int sb = 0; sb < num_sub_blocks; ++sb)
    {
        int const k_base = sb * kVecSize + lane * kElemsPerLane;
        // Load B sub-block (each lane reads kElemsPerLane FP8 elements).
        __nv_fp8_e4m3 b_pack[kElemsPerLane];
#pragma unroll
        for (int e = 0; e < kElemsPerLane; ++e)
            b_pack[e] = mat_b[static_cast<int64_t>(n) * K + k_base + e];
        float b_f[kElemsPerLane];
#pragma unroll
        for (int e = 0; e < kElemsPerLane; ++e)
            b_f[e] = float(b_pack[e]);

        // SFB at (n, sb) — packed_sfb is K-major shape [N_pad, K/(kVecSize*4)].
        int const kp = sb / 4;
        int const byte_idx = sb & 3;
        uint32_t const sfb_word = static_cast<uint32_t>(packed_sfb[kp * N_pad + n]);
        uint8_t const sfb_byte = (sfb_word >> (byte_idx * 8)) & 0xFFu;
        float const scale_b = ue8m0_byte_to_fp32(sfb_byte);

#pragma unroll
        for (int m = 0; m < kSmallMGemvMaxM; ++m)
        {
            if (m >= M)
                break;
            // SFA at (m, sb) — same scalar for every lane in the warp.
            uint32_t const sfa_word = static_cast<uint32_t>(packed_sfa[kp * M_pad + m]);
            uint8_t const sfa_byte = (sfa_word >> (byte_idx * 8)) & 0xFFu;
            float const scale = ue8m0_byte_to_fp32(sfa_byte) * scale_b;
            __nv_fp8_e4m3 a_pack[kElemsPerLane];
#pragma unroll
            for (int e = 0; e < kElemsPerLane; ++e)
                a_pack[e] = smem_A[m * K + k_base + e];
#pragma unroll
            for (int e = 0; e < kElemsPerLane; ++e)
                acc[m] += float(a_pack[e]) * b_f[e] * scale;
        }
    }

    // Single warp reduce per M row at the very end.
#pragma unroll
    for (int m = 0; m < kSmallMGemvMaxM; ++m)
    {
        if (m >= M)
            break;
        float total = acc[m];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            total += __shfl_xor_sync(0xffffffffu, total, off);
        if (lane == 0)
            out[m * N + n] = __float2bfloat16(total);
    }
}

} // namespace detail

// Host launcher. Returns true if the kernel was launched (caller should
// NOT also dispatch the matmul path). Returns false if the shape isn't
// supported (caller falls back).
template <bool USE_MXFP8>
inline bool launch_sm120_fp8_gemv(__nv_bfloat16* out, __nv_fp8_e4m3 const* mat_a, __nv_fp8_e4m3 const* mat_b,
    int32_t const* packed_sfa, int32_t const* packed_sfb, int M, int N, int K, cudaStream_t stream)
{
    if (M <= 0 || M > detail::kSmallMGemvMaxM)
        return false;
    // Both layouts require K to be a multiple of (kVecSize * 4) so the int32
    // SF stride fits cleanly; downstream callers already enforce K%128==0.
    constexpr int kVecSize = USE_MXFP8 ? 32 : 128;
    if (K % (kVecSize * 4) != 0)
        return false;

    int const BLOCK_N = detail::kSmallMGemvWarpsPerBlock;
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, 1, 1);
    dim3 block(detail::kSmallMGemvBlockThreads, 1, 1);
    size_t const smem_bytes = static_cast<size_t>(M) * K * sizeof(__nv_fp8_e4m3);

    auto kernel_ptr = detail::sm120_fp8_gemv_kernel<USE_MXFP8>;
    // Max dynamic SMEM cap — one-shot per (function, device). Same E8
    // static-guard discipline as the matmul launcher so capture mode is
    // safe after eager warmup.
    static bool s_configured = false;
    if (!s_configured)
    {
        cudaFuncSetAttribute(reinterpret_cast<void const*>(kernel_ptr),
            cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        cudaGetLastError();
        s_configured = true;
    }

    void* args[] = {&out, const_cast<__nv_fp8_e4m3**>(&mat_a), const_cast<__nv_fp8_e4m3**>(&mat_b),
        const_cast<int32_t**>(&packed_sfa), const_cast<int32_t**>(&packed_sfb), &M, &N, &K};
    cudaLaunchKernel(reinterpret_cast<void const*>(kernel_ptr), grid, block, args, smem_bytes, stream);
    return true;
}

} // namespace kernels::blockscale_gemm
TRTLLM_NAMESPACE_END
