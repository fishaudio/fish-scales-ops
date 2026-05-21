/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Top-level architecture router for the regular FP8 blockscale GEMM. Each
// call inspects the active device's SM version and forwards to the matching
// arch-specific dispatch. Supported archs: SM90 (Hopper, deep_gemm JIT) and
// SM120/121 (Blackwell consumer, mxf8 block-scale). Ada (SM89) is no longer
// supported.

#pragma once

#include "blockscale_gemm/common/kernel_utils.cuh"
#include "blockscale_gemm/common/scale_kernels.cuh"
#include "blockscale_gemm/arch/sm120/common/scale_repack.cuh"
#include "blockscale_gemm/arch/sm120/fp8/dispatch.cuh"
#include "blockscale_gemm/arch/sm90/fp8/dispatch.cuh"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

TRTLLM_NAMESPACE_BEGIN
namespace kernels::blockscale_gemm
{

// Pure FP8/FP8 -> BF16 path (caller-provided FP8 inputs and dequant scales).
// On sm_120 the kernel reinterprets scales_a/b as int32-packed UE8M0; callers
// must pass the int32-packed buffers (typed as float* for layering reasons).
// The bf16-input overload below handles the repack so the Runner-internal
// path needs no further changes.
inline void fp8_gemm_run(__nv_fp8_e4m3* mat_a, int ld_a, __nv_fp8_e4m3* mat_b, int ld_b, __nv_bfloat16* mat_d, int ld_d,
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, float* scales_a, float* scales_b, cudaStream_t stream)
{
    if (shape_m == 0)
        return;
    int const arch = tensorrt_llm::common::getSMVersion();
    if (arch == 120 || arch == 121)
        gemm_dispatch_sm120(mat_a, mat_b, mat_d, scales_a, scales_b, shape_m, shape_n, shape_k, stream);
    else
        gemm_dispatch_sm90(mat_a, ld_a, mat_b, ld_b, mat_d, ld_d, scales_a, scales_b, shape_m, shape_n, shape_k, stream);
}

namespace detail
{

// Thread-local pool for sm_120's int32-packed SFA/SFB scratch buffers used by
// the bf16-input fused path. Lazy-allocates on first use, grows on demand.
//
// CUDA-graph compatibility: cudaMalloc/cudaFree are forbidden during stream
// capture, so callers must warm up the bf16 path on the largest shape they
// plan to use BEFORE entering capture. If a later capture-mode call needs
// more, we abort with a clear message (same contract as `StreamKPool`).
struct Sm120BfPackPool
{
    int32_t* sfa = nullptr;
    std::size_t sfa_bytes = 0;
    int32_t* sfb = nullptr;
    std::size_t sfb_bytes = 0;

    static Sm120BfPackPool& instance()
    {
        static thread_local Sm120BfPackPool p;
        return p;
    }

    int32_t* ensure_sfa(std::size_t needed)
    {
        if (sfa_bytes >= needed)
            return sfa;
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(nullptr, &cap);
        if (cap == cudaStreamCaptureStatusActive)
        {
            std::fprintf(stderr,
                "[blockscale_gemm] Sm120BfPackPool: sfa needs %zu bytes during stream capture but pool has %zu. "
                "Warm up linear_bf16 on the largest shape before capture.\n",
                needed, sfa_bytes);
            std::abort();
        }
        if (sfa)
            cudaFree(sfa);
        cudaMalloc(&sfa, needed);
        sfa_bytes = needed;
        return sfa;
    }

    int32_t* ensure_sfb(std::size_t needed)
    {
        if (sfb_bytes >= needed)
            return sfb;
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(nullptr, &cap);
        if (cap == cudaStreamCaptureStatusActive)
        {
            std::fprintf(stderr,
                "[blockscale_gemm] Sm120BfPackPool: sfb needs %zu bytes during stream capture but pool has %zu. "
                "Warm up linear_bf16 on the largest shape before capture.\n",
                needed, sfb_bytes);
            std::abort();
        }
        if (sfb)
            cudaFree(sfb);
        cudaMalloc(&sfb, needed);
        sfb_bytes = needed;
        return sfb;
    }
};

} // namespace detail

// BF16/BF16 -> BF16 path with internal quantization fused before the GEMM.
//
// On sm_120 the SM120BlockScaledKernel reads scale tensors as int32-packed
// UE8M0 (4 K-consecutive UE8M0 bytes per int32 row entry; SFB also expanded
// from per-128-N-block to per-N-row). The runner-internal `scale_*_kernel`
// helpers produce FP32 scales in the legacy sm_90 layout, so a repack is
// required before launching the sm_120 GEMM. The bf16 input quant for SFA
// is routed through `scale_1x128_kernel<USE_UE8M0=true>` so the FP32
// scales are exact powers of 2 (their bit-pattern's exponent byte IS the
// final UE8M0 value); SFB uses raw `scale_128x128_kernel` (arbitrary FP32)
// and the repack rounds DOWN to nearest power of 2 — same trade-off as the
// pre-quantized `linear_fp8` path.
inline void fp8_gemm_run(__nv_bfloat16 const* mat_a, __nv_fp8_e4m3* fp8_mat_a, int ld_a, float* scales_a,
    __nv_bfloat16 const* mat_b, __nv_fp8_e4m3* fp8_mat_b, int ld_b, float* scales_b, __nv_bfloat16* mat_d, int ld_d,
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, cudaStream_t stream, bool internal_quantize_a = true,
    bool internal_quantize_b = true)
{
    if (shape_m == 0)
        return;
    if (kNumDeviceSMs < 0)
        kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

    int const arch = tensorrt_llm::common::getSMVersion();
    bool const is_sm120 = (arch == 120 || arch == 121);

    if (internal_quantize_a)
    {
        if (is_sm120)
        {
            // UE8M0 quant so the FP32 scales' exponent byte IS the final value.
            scale_1x128_kernel<__nv_bfloat16, __nv_fp8_e4m3, float, /*USE_UE8M0=*/true>
                <<<kNumDeviceSMs * 8, 256, 0, stream>>>(fp8_mat_a, scales_a, mat_a, shape_k, shape_m);
        }
        else
        {
            scale_1x128_kernel<__nv_bfloat16, __nv_fp8_e4m3, float, /*USE_UE8M0=*/false>
                <<<kNumDeviceSMs * 8, 256, 0, stream>>>(fp8_mat_a, scales_a, mat_a, shape_k, shape_m);
        }
    }
    if (internal_quantize_b)
    {
        if (is_sm120)
        {
            scale_128x128_kernel<__nv_bfloat16, __nv_fp8_e4m3, float, /*USE_UE8M0=*/true>
                <<<kNumDeviceSMs, 256, 0, stream>>>(fp8_mat_b, scales_b, mat_b, shape_k, shape_n);
        }
        else
        {
            scale_128x128_kernel<__nv_bfloat16, __nv_fp8_e4m3, float, /*USE_UE8M0=*/false>
                <<<kNumDeviceSMs, 256, 0, stream>>>(fp8_mat_b, scales_b, mat_b, shape_k, shape_n);
        }
    }

    if (is_sm120)
    {
        // Repack FP32 → int32-packed UE8M0 for both SFA and SFB.
        int const M_pad = static_cast<int>(div_up(shape_m, 4) * 4);
        int const N_pad = static_cast<int>(div_up(shape_n, 4) * 4);
        int const K_blocks = static_cast<int>(div_up(shape_k, 128));
        int const N_blocks = static_cast<int>(div_up(shape_n, 128));
        std::size_t const sfa_bytes = static_cast<std::size_t>(M_pad) * (K_blocks / 4) * sizeof(int32_t);
        std::size_t const sfb_bytes = static_cast<std::size_t>(N_pad) * (K_blocks / 4) * sizeof(int32_t);
        int32_t* packed_sfa = detail::Sm120BfPackPool::instance().ensure_sfa(sfa_bytes);
        int32_t* packed_sfb = detail::Sm120BfPackPool::instance().ensure_sfb(sfb_bytes);
        sm120_repack_sfa(packed_sfa, scales_a, M_pad, K_blocks, stream);
        sm120_repack_sfb(packed_sfb, scales_b, N_pad, N_blocks, K_blocks, stream);
        // gemm_dispatch_sm120 takes scales typed as float* but reinterprets to
        // int32* internally; pass the packed buffers through that channel.
        gemm_dispatch_sm120(fp8_mat_a, fp8_mat_b, mat_d, reinterpret_cast<float*>(packed_sfa),
            reinterpret_cast<float*>(packed_sfb), shape_m, shape_n, shape_k, stream);
    }
    else
    {
        gemm_dispatch_sm90(fp8_mat_a, ld_a, fp8_mat_b, ld_b, mat_d, ld_d, scales_a, scales_b, shape_m, shape_n,
            shape_k, stream);
    }
}

} // namespace kernels::blockscale_gemm
TRTLLM_NAMESPACE_END
