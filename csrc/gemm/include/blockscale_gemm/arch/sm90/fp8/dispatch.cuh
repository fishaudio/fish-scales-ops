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
#include <cstdlib>

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

    // Small-M routing. The main path's block_m floor is 64 (Hopper WGMMA
    // 2-warpgroup layout), so at M=32 it wastes half the M-tile and the grid
    // under-fills the SMs — making M=32 a ~15% local pessimum vs both M=16
    // (swap-A/B) and M=64 (full block_m=64). Route M<=32 through swap-A/B,
    // which turns the small dim into the scheduler's N and recovers fill.
    // Exclude ultra-wide N (>= 16384, e.g. Qwen3-4B gate_up N=19456): there
    // the main path already has enough N-tiles and swap is ~2% slower.
    // FSO (2026-06): wqkv/gate/down M=32 -15%, gate_up unchanged. M>=40
    // keeps the main path (block_m=64 utilisation recovers, swap regresses).
    bool const use_swap_ab = shape_m < 32u || (shape_m == 32u && shape_n < 16384u);

    if (!use_swap_ab)
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

// Grouped (MoE, masked) block-scale FP8 on sm_90 — revives the in-tree
// DeepGEMM GroupedMasked scheduler (jit/deep_gemm/scheduler.cuh). The kernel
// template, TMA descriptors, grouped runGemm wrapper, and the FP32 K-major
// scale format all already exist; this entry is the only host-side gap (plus
// the GroupedMasked case added to runtime.cuh's cubin name->enum map).
//
// Masked/DeepGEMM layout, identical to the sm_120 masked path:
//   A   [G, m_cap, K]  fp8   — per-group activation slab (shape_m = m_cap)
//   B   [G, N, K]      fp8   — per-expert weights
//   D   [G, m_cap, N]  bf16
//   SFA [G, ceil(m_cap,16), K/128]  FP32 K-major (TMA descriptor)
//   SFB [G, N/128, K/128]           FP32
//   masked_m [G] int32 on device — per-group valid row count = grouped_layout;
//     the scheduler reads it via __ldg(grouped_layout + curr_group_idx).
// expected_m (host-static ceil(total_rows/G)) drives only the config pick;
// masked_m is read on device, so the launch is CUDA-Graph capture-safe.
inline void gemm_dispatch_sm90_grouped_masked(void* mat_a, void* mat_b, void* mat_d, float* scales_a, float* scales_b,
    int* masked_m, uint32_t num_groups, uint32_t m_cap, uint32_t shape_n, uint32_t shape_k, uint32_t expected_m,
    cudaStream_t stream, int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

    constexpr uint32_t block_k = 128;

    // Config picked on expected_m (per-group expected rows), grouped over
    // num_groups. num_groups > 1 forces TMA multicast = 1 inside the picker.
    auto [bm, bn, ns, ntm, smem]
        = deep_gemm::jit::get_best_gemm_config(expected_m, shape_n, shape_k, num_groups, num_device_sms);
    auto runtime = deep_gemm::jit::getGlobalCompiler().build(
        shape_n, shape_k, bm, bn, block_k, num_groups, ns, ntm, deep_gemm::GemmType::GroupedMasked);
    auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
    deep_gemm::runGemm(kernel, mat_a, static_cast<int>(shape_k), mat_b, static_cast<int>(shape_k), mat_d,
        static_cast<int>(shape_n), scales_a, scales_b, m_cap, shape_n, shape_k, bm, bn, block_k, num_groups, ntm,
        deep_gemm::GemmType::GroupedMasked, masked_m, stream, num_device_sms, static_cast<uint32_t>(smem));
}

// Grouped (MoE, contiguous/sorted) block-scale FP8 on sm_90 — H4. Consumes the
// triton-style expert-sorted layout from moe_build_sorted and drives the
// in-tree DeepGEMM GroupedContiguous scheduler, whose per-block work tracks the
// active padded blocks (via the -1 length gate added to the scheduler) instead
// of the masked path's O(kNumGroups) scan.
//
// Contiguous layout:
//   A   [P_max, K]   fp8   — expert-sorted activation (shape_m = P_max, a fixed
//                            host upper bound so the launch is capture-safe)
//   B   [G, N, K]    fp8   — per-expert weights
//   D   [P_max, N]   bf16
//   SFA [align(P_max,4), K/128]  FP32 ColMajor (dense K-major TMA layout)
//   SFB [G, N/128, K/128]        FP32
//   sorted_expert_ids [P_max] int32 = grouped_layout — expert id per sorted row
//     (>= 0 real/pad, -1 past the actual padded length; the scheduler skips -1).
//
// block_m is passed explicitly (not chosen by the picker) because it must equal
// the granularity moe_build_sorted padded each expert's run to — otherwise a
// tiling block would straddle two experts. The picker is invoked only to choose
// block_n / num_stages for that block_m (via its shape_m<=64 -> block_m=64
// branch); H4a fixes block_m = 64. num_groups is passed so multicast is
// disabled (per-block weights differ).
inline void gemm_dispatch_sm90_grouped_contiguous(void* mat_a, void* mat_b, void* mat_d, float* scales_a,
    float* scales_b, int* sorted_expert_ids, uint32_t num_groups, uint32_t p_max, uint32_t shape_n, uint32_t shape_k,
    uint32_t block_m, uint32_t expected_m, cudaStream_t stream, int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

    constexpr uint32_t block_k = 128;
    (void) expected_m;
    (void) block_m; // H4a fixes block_m = 64 (the picker's shape_m<=64 branch);
                    // the caller pads moe_build_sorted to the same 64.

    // Config pick for the fixed block_m. Calling the picker with shape_m = 64
    // and is_grouped_contiguous = false selects its block_m = 64 branch and a
    // matching block_n / num_stages / smem; num_groups disables multicast. The
    // gemm_type handed to build/runGemm below is GroupedContiguous regardless.
    // (Forcing block_n = 64 for "more tiles" was tried and lost: M=1 regressed
    // and larger M hit a launch failure — block_n = 128 is the right pick.)
    auto [bm, bn, ns, ntm, smem]
        = deep_gemm::jit::get_best_gemm_config(64u, shape_n, shape_k, num_groups, num_device_sms, false, false);
    auto runtime = deep_gemm::jit::getGlobalCompiler().build(
        shape_n, shape_k, bm, bn, block_k, num_groups, ns, ntm, deep_gemm::GemmType::GroupedContiguous);
    auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
    deep_gemm::runGemm(kernel, mat_a, static_cast<int>(shape_k), mat_b, static_cast<int>(shape_k), mat_d,
        static_cast<int>(shape_n), scales_a, scales_b, p_max, shape_n, shape_k, bm, bn, block_k, num_groups, ntm,
        deep_gemm::GemmType::GroupedContiguous, sorted_expert_ids, stream, num_device_sms,
        static_cast<uint32_t>(smem));
}

// Contiguous grouped block-scale FP8 on sm_90, swap-AB — H4b. For M >= 8 the
// non-swap path pads each active expert's few rows to block_m = 64 (heavy
// padding that tips the GEMM compute-bound). Swap-AB tiles the activation rows
// by BLOCK_N = 16 instead (the weight becomes the A matrix, tiled by block_m
// along N), matching triton's own M >= 8 strategy — 4x less activation padding.
// Buffer layouts are identical to the non-swap path (the swap-AB descriptors
// read the same bytes): A_sorted [P_max, K], SFA K-major, weights [G,N,K],
// SFB [G,N/128,K/128], D [P_max, N]. The caller pads moe_build_sorted to
// BLOCK_N = 16 (= the activation tiling), and grouped_layout provides the
// per-block expert + the -1 length gate.
inline void gemm_dispatch_sm90_grouped_contiguous_swapab(void* mat_a_wgt, void* mat_b_act, void* mat_d, float* sfb_wgt,
    float* sfa_act, int* sorted_expert_ids, uint32_t num_groups, uint32_t p_max, uint32_t shape_n, uint32_t shape_k,
    uint32_t block_n, uint32_t expected_m, cudaStream_t stream, int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

    constexpr uint32_t block_k = 128;
    (void) expected_m;
    (void) block_n; // fixed to 16 below (= moe_build_sorted padding)

    // Picker chooses block_m (weight-N tiling) + num_stages for the swap-AB
    // shape; we then force block_n = 16 (activation tiling / padding) and
    // recompute the swap-AB smem. shape_m for the picker is the weight N (large
    // -> block_m = 128); shape_n is the activation extent.
    auto [bm, bn, ns, ntm, smem] = deep_gemm::jit::get_best_gemm_config(
        shape_n, p_max, shape_k, num_groups, num_device_sms, false, true);
    bn = 16u;
    // Deepen the pipeline. The picker sizes num_stages for a large-tile GEMM,
    // but the swap GEMM's activation tile is only BLOCK_N=16 wide, so it is
    // latency- rather than smem-bound: a deeper prefetch (6 vs the picker's ~5)
    // measurably raises achieved bandwidth (M=32 157->154us, M=64 172->169us,
    // closing the gap to triton) with no downside at small M. FSO_SWAP_STAGES
    // overrides for tuning; the value is clamped so the smem still fits.
    ns = ns < 6u ? 6u : ns;
    if (char const* e = std::getenv("FSO_SWAP_STAGES"))
    {
        int const s = atoi(e);
        if (s >= 1 && s <= 12)
            ns = static_cast<uint32_t>(s);
    }
    constexpr uint32_t kSm90MaxSmem = 227u * 1024u;
    smem = static_cast<uint32_t>(deep_gemm::jit::get_smem_size(
        static_cast<int>(ns), static_cast<int>(shape_k), static_cast<int>(bm), static_cast<int>(bn),
        static_cast<int>(block_k), true));
    while (ns > 1u && smem > kSm90MaxSmem)
    {
        --ns;
        smem = static_cast<uint32_t>(deep_gemm::jit::get_smem_size(
            static_cast<int>(ns), static_cast<int>(shape_k), static_cast<int>(bm), static_cast<int>(bn),
            static_cast<int>(block_k), true));
    }
    auto runtime = deep_gemm::jit::getGlobalCompiler().build(
        shape_n, shape_k, bm, bn, block_k, num_groups, ns, ntm, deep_gemm::GemmType::GroupedContiguous, true);
    auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
    deep_gemm::runGemmSwapAB(kernel, mat_a_wgt, static_cast<int>(shape_k), mat_b_act, static_cast<int>(shape_k), mat_d,
        static_cast<int>(shape_n), sfb_wgt, sfa_act, shape_n, p_max, shape_k, bm, bn, block_k, num_groups, ntm,
        deep_gemm::GemmType::GroupedContiguous, sorted_expert_ids, stream, num_device_sms,
        static_cast<uint32_t>(smem));
}

} // namespace kernels::blockscale_gemm
TRTLLM_NAMESPACE_END
