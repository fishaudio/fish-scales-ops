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
#include <vector>

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
// the bf16-input path (`linear_bf16`, `linear_qx`). Lazy-allocates on first
// use and grows on demand.
//
// CUDA-graph compatibility: a graph captured earlier keeps the buffer address
// it was captured with, so a buffer is never freed once handed out — growth
// allocates a larger one and retires the old one for the life of the process
// (so a graph captured before the growth still replays into live memory).
// Growing during a capture is impossible (cudaMalloc is not capturable), so
// it raises a RuntimeError that asks for one eager call of the larger shape
// on the capturing thread before capture. The capture check queries the
// stream the call runs on: torch.cuda.graph captures on a side stream, which
// a query of the legacy stream does not see.
struct Sm120BfPackPool
{
    int32_t* sfa = nullptr;
    std::size_t sfa_bytes = 0;
    int32_t* sfb = nullptr;
    std::size_t sfb_bytes = 0;
    std::vector<void*> retired; // never freed: graphs captured earlier may hold them

    static Sm120BfPackPool& instance()
    {
        static thread_local Sm120BfPackPool p;
        return p;
    }

    int32_t* ensure_sfa(std::size_t needed, cudaStream_t stream)
    {
        return grow(sfa, sfa_bytes, needed, stream, "sfa");
    }

    int32_t* ensure_sfb(std::size_t needed, cudaStream_t stream)
    {
        return grow(sfb, sfb_bytes, needed, stream, "sfb");
    }

private:
    int32_t* grow(int32_t*& buf, std::size_t& bytes, std::size_t needed, cudaStream_t stream, char const* what)
    {
        if (bytes >= needed)
            return buf;
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess)
        {
            (void) cudaGetLastError();
            cap = cudaStreamCaptureStatusActive;
        }
        if (cap != cudaStreamCaptureStatusNone)
            TLLM_THROW(
                "linear_bf16 / linear_qx on sm_120: the packed %s scale scratch needs %zu bytes but holds %zu, and it cannot "
                "grow while the stream is capturing. Run the same call once eagerly on the capturing thread "
                "before capturing it.",
                what, needed, bytes);
        constexpr std::size_t kGranule = std::size_t(1) << 20;
        std::size_t target = needed > 2 * bytes ? needed : 2 * bytes;
        target = (target + kGranule - 1) / kGranule * kGranule;
        void* fresh = nullptr;
        if (cudaMalloc(&fresh, target) != cudaSuccess)
        {
            (void) cudaGetLastError();
            TLLM_THROW("linear_bf16 / linear_qx on sm_120: cudaMalloc of %zu bytes for the packed %s scale scratch failed",
                target, what);
        }
        if (buf)
            retired.push_back(buf);
        buf = static_cast<int32_t*>(fresh);
        bytes = target;
        return buf;
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
        else if (fp8_1x128_lanes_supported(fp8_mat_a, mat_a, static_cast<int>(shape_k)))
        {
            // sm_90: the 8-lanes-per-group quantize, bitwise identical to scale_1x128_kernel (same BF16 amax floor,
            // padding-row scales unwritten), launched with PDL like the GEMM that follows unless FSO_DISABLE_PDL=1.
            fp8_1x128_lanes(fp8_mat_a, scales_a, mat_a, static_cast<int>(shape_k), static_cast<int>(shape_m), stream,
                /*bf16_amax_floor=*/true, /*zero_pad_rows=*/false, fso_pdl_enabled());
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
        int const K_words = static_cast<int>(div_up(K_blocks, 4)); // ceil(K/512) packed words per row
        std::size_t const sfa_bytes = static_cast<std::size_t>(M_pad) * K_words * sizeof(int32_t);
        std::size_t const sfb_bytes = static_cast<std::size_t>(N_pad) * K_words * sizeof(int32_t);
        int32_t* packed_sfa = detail::Sm120BfPackPool::instance().ensure_sfa(sfa_bytes, stream);
        int32_t* packed_sfb = detail::Sm120BfPackPool::instance().ensure_sfb(sfb_bytes, stream);
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
