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
#include "blockscale_gemm/arch/sm120/common/env_overrides.cuh" // fso_pdl_enabled (arch-agnostic)
#include "blockscale_gemm/arch/sm90/fp8/jit/deep_gemm/fp8_gemm.cuh"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

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

    // Programmatic dependent launch for the dense chain (activation quantize -> GEMM): the GEMM launches while the
    // quantize runs and waits on it with griddepcontrol.wait after its prologue (fp8_gemm_impl.cuh). On for every
    // shape, off for every shape with FSO_DISABLE_PDL=1.
    bool const pdl = fso_pdl_enabled();

    // Tile rules of the dense GEMM on top of the shared picker: deep_gemm::jit::get_dense_gemm_config.
    if (!use_swap_ab)
    {
        auto [bm, bn, ns, ntm, smem]
            = deep_gemm::jit::get_dense_gemm_config(shape_m, shape_n, shape_k, num_device_sms, false);
        auto runtime = deep_gemm::jit::getGlobalCompiler().build(
            shape_n, shape_k, bm, bn, block_k, num_problems, ns, ntm, deep_gemm::GemmType::Normal);
        auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
        deep_gemm::runGemm(kernel, mat_a, ld_a, mat_b, ld_b, mat_d, ld_d, scales_a, scales_b, shape_m, shape_n, shape_k,
            bm, bn, block_k, num_problems, ntm, deep_gemm::GemmType::Normal, static_cast<int*>(nullptr), stream,
            num_device_sms, static_cast<uint32_t>(smem), pdl);
    }
    else
    {
        // Tiny M: swap A/B inside the kernel so the persistent scheduler still
        // has enough tiles along the now-M axis.
        auto [bm, bn, ns, ntm, smem]
            = deep_gemm::jit::get_dense_gemm_config(shape_n, shape_m, shape_k, num_device_sms, true);
        auto runtime = deep_gemm::jit::getGlobalCompiler().build(
            shape_n, shape_k, bm, bn, block_k, num_problems, ns, ntm, deep_gemm::GemmType::Normal, true);
        auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
        deep_gemm::runGemmSwapAB(kernel, mat_b, ld_b, mat_a, ld_a, mat_d, ld_d, scales_b, scales_a, shape_n, shape_m,
            shape_k, bm, bn, block_k, num_problems, ntm, deep_gemm::GemmType::Normal, static_cast<int*>(nullptr),
            stream, num_device_sms, static_cast<uint32_t>(smem), pdl);
    }
}

// Grouped (MoE, masked) block-scale FP8 on sm_90 — revives the in-tree
// DeepGEMM GroupedMasked scheduler (jit/deep_gemm/scheduler.cuh). The kernel
// template, TMA descriptors, grouped runGemm wrapper, and the FP32 K-major
// scale format all already exist; this entry is the only host-side gap.
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

// Contiguous grouped FC1 on sm_90, non-swap, with two math warp-groups split along N (fused-FC1 step 1,
// 2026-09-30; fp8_gemm_kernel_2wg). Same buffers and contract as gemm_dispatch_sm90_grouped_contiguous, for the FC1
// only: mat_b is the stacked [gate; up] weight [G, shape_n = 2I, K], and one CTA computes gate block b (warp-group 0)
// and up block b (warp-group 1) of one 64-row activation tile, storing both bf16 halves to their native columns of
// D [p_max, 2I]. block_m must be 64 (the moe_build_sorted padding); block_n is fixed at 128 per warp-group.
//
// Stages: the largest count whose shared memory fits. A stage carries A (8 KB), both 128-row weight halves (32 KB)
// and the A scales, and the bf16 staging of the two 64 x 128 output tiles takes 32 KB, so 4 stages fit (197,824 B;
// 5 would need 239,056 B). Four stages of this kernel hold the same MMA work as today's eight, since every stage
// feeds both warp-groups.
//
// Registers: the kernel is built for 384 threads at one CTA per SM, so ptxas must compile it to exactly 168 registers
// for setmaxnreg's 40 / 232 split to balance; a lower count would deadlock the math warp-groups' increase. The host
// therefore refuses any cubin that is not at 168 registers or that uses local memory (checked once per cubin).
// The two-warp-group FC1 kernels (fp8_gemm_kernel_2wg and fp8_gemm_kernel_2wg_swiglu) are launched with 384 threads at
// one CTA per SM, so ptxas must compile them to exactly 168 registers for setmaxnreg's 40 / 232 split to balance
// ((168 - 40) * 128 == (232 - 168) * 256); a lower count would deadlock the math warp-groups' increase, and nothing
// inside the kernel could recover. Checked once per cubin, together with the absence of local memory (spills).
inline void sm90_check_2wg_kernel_resources(cudaKernel_t kernel, char const* what)
{
    static std::vector<void*> s_checked;
    if (std::find(s_checked.begin(), s_checked.end(), reinterpret_cast<void*>(kernel)) != s_checked.end())
        return;
    int num_regs = 0, local_bytes = -1;
    auto const ck = reinterpret_cast<CUkernel>(kernel);
    if (cuKernelGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, ck, 0) != CUDA_SUCCESS || num_regs != 168)
        TLLM_THROW("%s: the cubin has %d registers per thread, not the 168 its 40 / 232 setmaxnreg split needs; "
                   "refusing to launch (it would deadlock)",
            what, num_regs);
    if (cuKernelGetAttribute(&local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, ck, 0) != CUDA_SUCCESS
        || local_bytes != 0)
        TLLM_THROW("%s: the cubin uses %d bytes of local memory per thread (spills)", what, local_bytes);
    s_checked.push_back(reinterpret_cast<void*>(kernel));
}

inline uint32_t sm90_grouped_contiguous_2wg_smem_size(uint32_t num_stages, uint32_t shape_k)
{
    constexpr uint32_t bm = 64u, bn = 128u, bk = 128u;
    uint32_t const smem_d = 2u * bm * bn * 2u;                     // two bf16 64 x 128 staging tiles
    uint32_t const per_stage = bm * bk + 2u * bn * bk + bm * 4u;   // A + (gate, up) B halves + A scales
    uint32_t const smem_scales_b = (2u * ((shape_k + bk - 1u) / bk) * 4u + 7u) / 8u * 8u; // gate and up scale rows
    uint32_t const smem_barrier = num_stages * 8u * 2u;
    return smem_d + num_stages * per_stage + smem_scales_b + smem_barrier;
}

inline void gemm_dispatch_sm90_grouped_contiguous_2wg(void* mat_a, void* mat_b, void* mat_d, float* scales_a,
    float* scales_b, int* sorted_expert_ids, uint32_t num_groups, uint32_t p_max, uint32_t shape_n, uint32_t shape_k,
    uint32_t block_m, uint32_t expected_m, cudaStream_t stream, int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

    constexpr uint32_t block_k = 128;
    constexpr uint32_t bm = 64u, bn = 128u;
    (void) expected_m;
    if (block_m != bm)
        TLLM_THROW("two-warp-group grouped FC1: block_m must be 64 (= the moe_build_sorted padding), got %u", block_m);
    if (shape_n % (2u * bn) != 0u || shape_k % block_k != 0u)
        TLLM_THROW("two-warp-group grouped FC1: N must be a multiple of 256 (2I, I % 128 == 0) and K of 128, "
                   "got N=%u K=%u",
            shape_n, shape_k);

    constexpr uint32_t kSm90MaxSmem = 232448u; // the picker's sm90_capacity (227 KB of dynamic shared memory)
    uint32_t ns = 8u;
    while (ns > 1u && sm90_grouped_contiguous_2wg_smem_size(ns, shape_k) > kSm90MaxSmem)
        --ns;
    uint32_t const smem = sm90_grouped_contiguous_2wg_smem_size(ns, shape_k);

    auto runtime = deep_gemm::jit::getGlobalCompiler().build(shape_n, shape_k, bm, bn, block_k, num_groups, ns, 1u,
        deep_gemm::GemmType::GroupedContiguous, false, 1u, true);
    auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());

    // Once per cubin: exactly 168 registers (see above) and no local memory.
    sm90_check_2wg_kernel_resources(kernel, "two-warp-group grouped FC1");

    deep_gemm::runGemm2wg(kernel, mat_a, static_cast<int>(shape_k), mat_b, static_cast<int>(shape_k), mat_d,
        static_cast<int>(shape_n), scales_a, scales_b, p_max, shape_n, shape_k, bm, bn, block_k, num_groups,
        sorted_expert_ids, stream, num_device_sms, smem);
}

// Contiguous grouped FC1 on sm_90, non-swap, with the SwiGLU + 1x128 FP8 requantize fused into the epilogue (fused-FC1
// step 2, 2026-09-30; fp8_gemm_kernel_2wg_swiglu). Same inputs as gemm_dispatch_sm90_grouped_contiguous_2wg (mat_b is
// the stacked [gate; up] weight [G, shape_n = 2I, K]); the outputs are what silu_chunk_mul_quantize_1x128_sorted_sm90
// makes of that FC1's bf16 result: mat_dq [p_max, I] fp8 and mat_sd [I / 128, sd_ld] fp32 (sd_ld = align4(p_max)).
//
// Stages: the largest count whose shared memory fits (the layout is sm90_swiglu_smem in fp8_gemm_impl.cuh, which the
// kernel reads too). A stage is A (8 KB) + both 128-row weight halves (32 KB) + the A scales; the epilogue holds an
// 8 KB fp8 staging tile, 16 KB of bf16 exchange and 512 B of half-row amaxes. At K = 2048 that is 5 stages
// (231,376 B of 232,448).
//
// Registers: the same 168 / 40 / 232 contract and guard as the step-1 kernel.
inline uint32_t sm90_grouped_contiguous_swiglu_smem_size(uint32_t num_stages, uint32_t shape_k)
{
    return deep_gemm::sm90_swiglu_smem::total_bytes(num_stages, shape_k);
}

inline uint32_t sm90_grouped_contiguous_swiglu_num_stages(uint32_t shape_k)
{
    constexpr uint32_t kSm90MaxSmem = 232448u; // the picker's sm90_capacity (227 KB of dynamic shared memory)
    uint32_t ns = 8u;
    while (ns > 1u && sm90_grouped_contiguous_swiglu_smem_size(ns, shape_k) > kSm90MaxSmem)
        --ns;
    return ns;
}

inline void gemm_dispatch_sm90_grouped_contiguous_swiglu(void* mat_a, void* mat_b, void* mat_dq, float* mat_sd,
    float* scales_a, float* scales_b, int* sorted_expert_ids, uint32_t num_groups, uint32_t p_max, uint32_t shape_n,
    uint32_t shape_k, uint32_t block_m, uint32_t expected_m, cudaStream_t stream, int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

    constexpr uint32_t block_k = 128;
    constexpr uint32_t bm = 64u, bn = 128u;
    (void) expected_m;
    if (block_m != bm)
        TLLM_THROW("fused SwiGLU grouped FC1: block_m must be 64 (= the moe_build_sorted padding), got %u", block_m);
    if (shape_n % (2u * bn) != 0u || shape_k % block_k != 0u)
        TLLM_THROW("fused SwiGLU grouped FC1: N must be a multiple of 256 (2I, I % 128 == 0) and K of 128, "
                   "got N=%u K=%u",
            shape_n, shape_k);

    uint32_t const ns = sm90_grouped_contiguous_swiglu_num_stages(shape_k);
    uint32_t const smem = sm90_grouped_contiguous_swiglu_smem_size(ns, shape_k);
    uint32_t const sd_ld = (p_max + 3u) / 4u * 4u;

    auto runtime = deep_gemm::jit::getGlobalCompiler().build(shape_n, shape_k, bm, bn, block_k, num_groups, ns, 1u,
        deep_gemm::GemmType::GroupedContiguous, false, 1u, true, true);
    auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
    sm90_check_2wg_kernel_resources(kernel, "fused SwiGLU grouped FC1");

    deep_gemm::runGemm2wgSwiglu(kernel, mat_a, static_cast<int>(shape_k), mat_b, static_cast<int>(shape_k), mat_dq,
        mat_sd, sd_ld, scales_a, scales_b, p_max, shape_n, shape_k, bm, bn, block_k, num_groups, sorted_expert_ids,
        stream, num_device_sms, smem);
}

// Two-CTA swap-AB builds (fp8_gemm_kernel_swapAB and the gate/up pair kernel below) rebalance their registers 40 / 96
// around a compiled count of exactly 80 (fp8_gemm_impl.cuh). ptxas decides that count, so another NVRTC build or
// another kernel body can land elsewhere, and below 80 the math warp-groups' setmaxnreg increase deadlocks; such a
// cubin must run at one CTA per SM instead. With check_local, a build that spills to local memory is refused the same
// way (the pair kernels hold two accumulator sets, so a large tile can spill at the 96-register two-CTA budget).
// Returns whether the kernel may run at two CTAs per SM. When it may not, it prints one stderr line per kernel name per
// process, naming the kernel, the count ptxas produced (and the local bytes) and the one-CTA fallback, so a deployment
// whose NVRTC moves the count does not lose two-CTA residency without a signal.
inline bool sm90_swapab_two_cta_regs_ok(cudaKernel_t kernel, char const* what, bool check_local = false)
{
    int num_regs = 0, local_bytes = 0;
    auto const ck = reinterpret_cast<CUkernel>(kernel);
    bool const regs_ok
        = cuKernelGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, ck, 0) == CUDA_SUCCESS && num_regs == 80;
    bool const local_ok = !check_local
        || (cuKernelGetAttribute(&local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, ck, 0) == CUDA_SUCCESS
            && local_bytes == 0);
    if (regs_ok && local_ok)
        return true;
    char const* name = nullptr;
    if (cuKernelGetName(&name, ck) != CUDA_SUCCESS || name == nullptr)
        name = "(unnamed)";
    static std::vector<std::string> s_reported;
    if (std::find(s_reported.begin(), s_reported.end(), std::string(name)) == s_reported.end())
    {
        s_reported.emplace_back(name);
        if (check_local)
            std::fprintf(stderr,
                "[fish_scales_ops] sm_90 %s: the two-CTA build of %s compiled to %d registers per thread and %d bytes "
                "of local memory, not the 80 registers without spills its 40 / 96 setmaxnreg split needs; falling back "
                "to one CTA per SM\n",
                what, name, num_regs, local_bytes);
        else
            std::fprintf(stderr,
                "[fish_scales_ops] sm_90 %s: the two-CTA build of %s compiled to %d registers per thread, not the 80 "
                "its 40 / 96 setmaxnreg split needs; falling back to one CTA per SM\n",
                what, name, num_regs);
    }
    return false;
}

// FSO_SWAP_STAGES (1..12) and FSO_SWAPAB_CTAS_PER_SM (1 or 2), the swap-AB tuning knobs, read on every call like the
// rest of the swap-AB dispatch; 0 when unset or out of range.
inline uint32_t sm90_env_swap_stages()
{
    char const* e = std::getenv("FSO_SWAP_STAGES");
    int const s = e ? atoi(e) : 0;
    return (s >= 1 && s <= 12) ? static_cast<uint32_t>(s) : 0u;
}

inline uint32_t sm90_env_swapab_ctas()
{
    char const* e = std::getenv("FSO_SWAPAB_CTAS_PER_SM");
    int const v = e ? atoi(e) : 0;
    return (v == 1 || v == 2) ? static_cast<uint32_t>(v) : 0u;
}

// Contiguous grouped block-scale FP8 on sm_90, swap-AB — H4b. For M >= 8 the
// non-swap path pads each active expert's few rows to block_m = 64 (heavy
// padding that tips the GEMM compute-bound). Swap-AB tiles the activation rows
// by BLOCK_N = block_n instead (the weight becomes the A matrix, tiled by
// block_m along N), matching triton's own M >= 8 strategy. block_n is 16, 32
// or 64: the caller picks it from the routed rows per active expert (the
// swap-AB kernel re-streams an expert's weights once per activation tile, so
// the tile grows with the rows an expert carries; see moe_layer_fp8_sm90) and
// pads moe_build_sorted to the same value. Buffer layouts are identical to the
// non-swap path (the swap-AB descriptors read the same bytes): A_sorted
// [P_max, K], SFA K-major, weights [G,N,K], SFB [G,N/128,K/128], D [P_max, N];
// grouped_layout provides the per-block expert + the -1 length gate.
inline void gemm_dispatch_sm90_grouped_contiguous_swapab(void* mat_a_wgt, void* mat_b_act, void* mat_d, float* sfb_wgt,
    float* sfa_act, int* sorted_expert_ids, uint32_t num_groups, uint32_t p_max, uint32_t shape_n, uint32_t shape_k,
    uint32_t block_n, uint32_t expected_m, cudaStream_t stream, int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

    constexpr uint32_t block_k = 128;
    (void) expected_m;
    if (block_n != 16u && block_n != 32u && block_n != 64u)
        TLLM_THROW("swap-AB grouped GEMM: block_n must be 16, 32 or 64 (= the moe_build_sorted padding), got %u", block_n);

    // Picker chooses block_m (weight-N tiling) + num_stages for the swap-AB
    // shape; we then force block_n to the caller's activation tiling / padding
    // and recompute the swap-AB smem. shape_m for the picker is the weight N
    // (large -> block_m = 128); shape_n is the activation extent.
    auto [bm, bn, ns, ntm, smem] = deep_gemm::jit::get_best_gemm_config(
        shape_n, p_max, shape_k, num_groups, num_device_sms, false, true);
    bn = block_n;
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
    // Two resident CTAs per SM: each CTA's math warp-groups only carry a BLOCK_N = 16 accumulator, so they
    // fit in 96 registers and two CTAs share the register file; the second CTA's TMA stream covers the
    // first's per-block epilogue and WGMMA-wait chain, which a single persistent CTA exposes on every block
    // of the short-K down projection. The stage count is reduced until two CTAs' shared memory (plus the
    // 1 KB the driver reserves per CTA) fits. FSO_SWAPAB_CTAS_PER_SM=1|2 overrides.
    // Only when the sorted layout can feed both CTAs: p_max / 16 activation tiles times the weight tiles is
    // the upper bound on real blocks, and below two per SM the doubled grid just adds idle CTAs (Family B
    // M = 1 / M = 2 read +1.5-1.8 % with the doubled grid, neutral below the bound elsewhere).
    uint32_t const max_blocks = (p_max / bn) * (shape_n / bm);
    uint32_t ctas = max_blocks > 2u * static_cast<uint32_t>(num_device_sms) ? 2u : 1u;
    if (char const* e = std::getenv("FSO_SWAPAB_CTAS_PER_SM"))
    {
        int const v = atoi(e);
        if (v == 1 || v == 2)
            ctas = static_cast<uint32_t>(v);
    }
    constexpr uint32_t kReservedSmemPerCta = 1024u;
    // The kernel's TMA tiles use the 128-byte swizzle, whose pattern repeats every 1024 bytes, so every
    // tile must sit at a 1024-byte-aligned shared-memory address. `extern __shared__ __align__(1024)` only
    // fixes the offsets inside a CTA's allocation; the allocation itself starts where the previous resident
    // CTA's ends. With one CTA per SM that base is 0. With two, the second CTA's base is the first CTA's
    // dynamic size plus the driver's 1 KB reservation, so the dynamic size must itself be a multiple of
    // 1024 bytes or the second CTA's tiles are written with one swizzle phase and read with another
    // (measured: 4-9 of 20 layer outputs differed at M = 1024 before this rounding, none with one CTA).
    auto fits = [&](uint32_t stages) -> uint32_t
    {
        uint32_t const bytes = static_cast<uint32_t>(deep_gemm::jit::get_smem_size(static_cast<int>(stages),
            static_cast<int>(shape_k), static_cast<int>(bm), static_cast<int>(bn), static_cast<int>(block_k), true));
        return (bytes + 1023u) / 1024u * 1024u;
    };
    smem = fits(ns);
    while (ns > 1u && ctas * (smem + kReservedSmemPerCta) > kSm90MaxSmem)
    {
        --ns;
        smem = fits(ns);
    }
    if (ctas == 2u)
    {
        // With two resident CTAs the kernel's NotDivisibleK tail (a K extent that is not a multiple of
        // num_stages x 128, handled by plain barrier arrivals on the unused stages) is not deterministic:
        // 40-run bitwise tests at M = 1024 read 8-12 distinct layer outputs with 3 or 5 stages (tail path
        // taken) and 40 identical ones with 4 stages (K = 512 and K = 2048 both divide), while one CTA per
        // SM is bit-stable with any stage count (2026-09-25, det_test / det_localize: exactly one activation
        // x weight tile of the FC1 GEMM differs per bad launch). The two-CTA mode therefore only uses stage
        // counts that divide K, and falls back to the single-CTA build when none of 2..ns fits.
        uint32_t const k_stages = shape_k / block_k;
        while (ns > 1u && k_stages % ns != 0u)
            --ns;
        if (ns < 2u)
            ctas = 1u;
        else
            smem = fits(ns);
    }
    if (ctas == 1u)
    {
        ns = ns < 6u ? 6u : ns;
        if (char const* e = std::getenv("FSO_SWAP_STAGES"))
        {
            int const s = atoi(e);
            if (s >= 1 && s <= 12)
                ns = static_cast<uint32_t>(s);
        }
        smem = fits(ns);
        while (ns > 1u && smem + kReservedSmemPerCta > kSm90MaxSmem)
        {
            --ns;
            smem = fits(ns);
        }
    }
    auto runtime = deep_gemm::jit::getGlobalCompiler().build(shape_n, shape_k, bm, bn, block_k, num_groups, ns, ntm,
        deep_gemm::GemmType::GroupedContiguous, true, ctas);
    auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
    if (ctas == 2u)
    {
        // The two-CTA kernel rebalances registers 40 / 96 around a compiled count of exactly 80 (see
        // fp8_gemm_impl.cuh); a cubin ptxas compiled below 80 would deadlock the math warp-groups' increase. The
        // fallback says so on stderr once per kernel and takes the one-CTA stage pick above, FSO_SWAP_STAGES included.
        if (!sm90_swapab_two_cta_regs_ok(kernel, "swap-AB grouped GEMM"))
        {
            ctas = 1u;
            ns = ns < 6u ? 6u : ns;
            if (uint32_t const s = sm90_env_swap_stages())
                ns = s;
            smem = fits(ns);
            while (ns > 1u && smem + kReservedSmemPerCta > kSm90MaxSmem)
            {
                --ns;
                smem = fits(ns);
            }
            runtime = deep_gemm::jit::getGlobalCompiler().build(shape_n, shape_k, bm, bn, block_k, num_groups, ns,
                ntm, deep_gemm::GemmType::GroupedContiguous, true, 1u);
            kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
        }
    }
    deep_gemm::runGemmSwapAB(kernel, mat_a_wgt, static_cast<int>(shape_k), mat_b_act, static_cast<int>(shape_k), mat_d,
        static_cast<int>(shape_n), sfb_wgt, sfa_act, shape_n, p_max, shape_k, bm, bn, block_k, num_groups, ntm,
        deep_gemm::GemmType::GroupedContiguous, sorted_expert_ids, stream, static_cast<int>(num_device_sms * ctas),
        static_cast<uint32_t>(smem));
}

// Contiguous grouped FC1 on sm_90, swap-AB, with one CTA per (activation tile, 128-column SwiGLU block) (fused-FC1
// P2a, design S2, 2026-10-01; fp8_gemm_kernel_swapAB_pair). Same buffers and contract as
// gemm_dispatch_sm90_grouped_contiguous_swapab for the FC1: mat_a_wgt is the stacked [gate; up] weight
// [G, shape_n = 2I, K] (I a multiple of 128), and one CTA computes gate block b and up block b (256 weight rows) of one
// block_n-row activation tile. D is today's gu [p_max, 2I] bf16, bit-identical to the swap-AB FC1's on the rows the
// scheduler computes.
//
// Launch plan (sm90_swapab_pair_plan), the swap-AB FC1's rules applied to this tile:
//   * CTAs per SM: one, for every shape. Measured on the fused kernel (P2a REPORT.md, the D runs): one CTA per SM at
//     the stage rule below beat two CTAs at 2 stages each at every bn16 cell of both families (by 0.1-19 %), and at
//     the bn32 cells it won by up to 0.6 % or lost by at most 0.2 %; at block_n = 64 a two-CTA build spills (two
//     accumulator sets and their promotion copies need 2 * block_n registers per math thread, 128 > the 96 a
//     two-CTA math warp-group has).
//     FSO_SWAPAB_CTAS_PER_SM=2 forces two CTAs (an A/B knob; a build that misses 80 registers or spills still falls
//     back to one CTA).
//   * Stages, one CTA per SM: the largest stage count that fits the shared-memory budget AND divides K / 128 (4 at
//     K = 2048: 8 does not fit, 5 and 6 do not divide 16). Measured at M = 1, where every CTA runs a single tile: the
//     fused FC1 took 9.17 / 8.85 / 8.93 us (Family C) and 10.76 / 10.25 / 10.08 us (Family B) in the layer at
//     6 / 5 / 4 stages, and only below 6 does the layer meet the 1 % bar at C M=1. The pair kernel (same mainloop, bf16
//     epilogue) and today's swap-AB kernel do not slow down at 6 stages, so the cost sits in the fused kernel at 6:
//     a stage count that does not divide K / 128 makes the compiler emit a second, partial copy of the unrolled
//     pipeline body for the last pass (the NotDivisibleK path), and the fused 6-stage kernel is 42 KiB of code against
//     30 KiB at 4; with one pass through the code per CTA at tiny M, code size is the likely cost (UNVERIFIED, P2a
//     GATE1.md). A divisor of K / 128 never has that second copy.
//   * Stages, two CTAs per SM: FSO_SWAP_STAGES or 6, reduced until both CTAs' shared memory fits, then until it
//     divides K / 128 (the swap-AB two-CTA determinism rule); one CTA when none of 2.. fits.
//   * Shared memory: the sm90_swapab_pair_smem layout, rounded to 1 KB so a second CTA's base stays 1024-byte aligned,
//     plus the 1 KB the driver reserves per CTA. FSO_SWAP_STAGES overrides the stage count of either mode (reduced
//     until it fits).
//   * Registers: a two-CTA build must have exactly 80 and no local memory (sm90_swapab_two_cta_regs_ok; otherwise one
//     CTA, with a stderr line); a one-CTA build exactly 168, the 40 / 232 setmaxnreg split of 384 threads, or it is
//     refused.
struct Sm90SwapAbPairPlan
{
    uint32_t ctas;   // resident CTAs per SM (the grid is num_sms * ctas)
    uint32_t stages; // pipeline stages per CTA
    uint32_t smem;   // dynamic shared memory per CTA in bytes, a multiple of 1 KB
};

inline uint32_t sm90_swapab_pair_smem_size(uint32_t block_n, bool fused, uint32_t num_stages)
{
    return (deep_gemm::sm90_swapab_pair_smem::total_bytes(block_n, fused, num_stages) + 1023u) / 1024u * 1024u;
}

inline Sm90SwapAbPairPlan sm90_swapab_pair_plan(uint32_t p_max, uint32_t shape_n, uint32_t shape_k, uint32_t block_n,
    bool fused, uint32_t forced_ctas, int num_device_sms)
{
    constexpr uint32_t kSm90MaxSmem = 227u * 1024u; // as gemm_dispatch_sm90_grouped_contiguous_swapab
    constexpr uint32_t kReservedSmemPerCta = 1024u;
    (void) p_max;
    (void) shape_n;
    (void) num_device_sms;
    uint32_t ctas = forced_ctas ? forced_ctas : 1u;
    uint32_t const first = sm90_env_swap_stages() ? sm90_env_swap_stages() : 6u;
    uint32_t ns = first;
    if (ctas == 2u)
    {
        while (ns > 1u && 2u * (sm90_swapab_pair_smem_size(block_n, fused, ns) + kReservedSmemPerCta) > kSm90MaxSmem)
            --ns;
        uint32_t const k_stages = shape_k / 128u;
        while (ns > 1u && k_stages % ns != 0u)
            --ns;
        if (ns < 2u)
            ctas = 1u;
    }
    if (ctas == 1u)
    {
        auto const fits_one_cta = [&](uint32_t stages)
        { return sm90_swapab_pair_smem_size(block_n, fused, stages) + kReservedSmemPerCta <= kSm90MaxSmem; };
        if (sm90_env_swap_stages())
        {
            ns = sm90_env_swap_stages();
            while (ns > 1u && !fits_one_cta(ns))
                --ns;
        }
        else
        {
            uint32_t const k_stages = shape_k / 128u;
            ns = 1u;
            for (uint32_t stages = k_stages; stages > 1u; --stages)
            {
                if (k_stages % stages == 0u && fits_one_cta(stages))
                {
                    ns = stages;
                    break;
                }
            }
        }
    }
    return {ctas, ns, sm90_swapab_pair_smem_size(block_n, fused, ns)};
}

// One-CTA builds of the pair kernels: 384 threads at one CTA per SM must compile to exactly 168 registers for the
// 40 / 232 setmaxnreg split to balance ((168 - 40) * 128 == (232 - 168) * 256), as the two-warp-group FC1 kernels;
// refuse anything else (it could deadlock). Checked once per cubin.
inline void sm90_check_swapab_pair_one_cta(cudaKernel_t kernel, char const* what)
{
    static std::vector<void*> s_checked;
    if (std::find(s_checked.begin(), s_checked.end(), reinterpret_cast<void*>(kernel)) != s_checked.end())
        return;
    int num_regs = 0;
    if (cuKernelGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, reinterpret_cast<CUkernel>(kernel), 0)
            != CUDA_SUCCESS
        || num_regs != 168)
        TLLM_THROW("%s: the one-CTA cubin has %d registers per thread, not the 168 its 40 / 232 setmaxnreg split "
                   "needs; refusing to launch (it would deadlock)",
            what, num_regs);
    s_checked.push_back(reinterpret_cast<void*>(kernel));
}

inline void gemm_dispatch_sm90_grouped_contiguous_swapab_pair(void* mat_a_wgt, void* mat_b_act, void* mat_d,
    float* sfb_wgt, float* sfa_act, int* sorted_expert_ids, uint32_t num_groups, uint32_t p_max, uint32_t shape_n,
    uint32_t shape_k, uint32_t block_n, uint32_t expected_m, cudaStream_t stream, int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

    constexpr uint32_t block_k = 128;
    constexpr uint32_t bm = 128u;
    (void) expected_m;
    if (block_n != 16u && block_n != 32u && block_n != 64u)
        TLLM_THROW("swap-AB gate/up pair FC1: block_n must be 16, 32 or 64 (= the moe_build_sorted padding), got %u",
            block_n);
    if (shape_n % (2u * bm) != 0u || shape_k % block_k != 0u)
        TLLM_THROW("swap-AB gate/up pair FC1: N must be a multiple of 256 (2I, I % 128 == 0) and K of 128, "
                   "got N=%u K=%u",
            shape_n, shape_k);

    auto build = [&](Sm90SwapAbPairPlan const& plan)
    {
        auto runtime = deep_gemm::jit::getGlobalCompiler().build(shape_n, shape_k, bm, block_n, block_k, num_groups,
            plan.stages, 1u, deep_gemm::GemmType::GroupedContiguous, true, plan.ctas, false, false, true);
        return reinterpret_cast<cudaKernel_t>(runtime->getKernel());
    };
    Sm90SwapAbPairPlan plan
        = sm90_swapab_pair_plan(p_max, shape_n, shape_k, block_n, false, sm90_env_swapab_ctas(), num_device_sms);
    cudaKernel_t kernel = build(plan);
    if (plan.ctas == 2u && !sm90_swapab_two_cta_regs_ok(kernel, "swap-AB gate/up pair FC1", true))
    {
        plan = sm90_swapab_pair_plan(p_max, shape_n, shape_k, block_n, false, 1u, num_device_sms);
        kernel = build(plan);
    }
    if (plan.ctas == 1u)
        sm90_check_swapab_pair_one_cta(kernel, "swap-AB gate/up pair FC1");

    deep_gemm::runGemmSwapAB(kernel, mat_a_wgt, static_cast<int>(shape_k), mat_b_act, static_cast<int>(shape_k), mat_d,
        static_cast<int>(shape_n), sfb_wgt, sfa_act, shape_n, p_max, shape_k, bm, block_n, block_k, num_groups, 1u,
        deep_gemm::GemmType::GroupedContiguous, sorted_expert_ids, stream,
        static_cast<int>(num_device_sms * plan.ctas), plan.smem);
}

// FSO_SWAPAB_SPLIT=0|1 forces the cluster-split fused swap-AB FC1 off or on, for A/B runs; unset (or empty) applies
// the rule of sm90_swapab_split_route. Read on every call, like FSO_SWAPAB_CTAS_PER_SM: set it before the first call
// and keep it between a capture and its replays.
inline int sm90_env_swapab_split()
{
    char const* e = std::getenv("FSO_SWAPAB_SPLIT");
    if (e == nullptr || e[0] == '\0')
        return -1;
    return e[0] == '0' ? 0 : 1;
}

// Whether the fused swap-AB FC1 runs as fp8_gemm_kernel_swapAB_swiglu_split (fp8_gemm_impl.cuh), each SwiGLU block
// shared by the two CTAs of a cluster, instead of one CTA per block (H2 Phase 2a, 2026-10-05). The unsplit kernel gives
// a whole block (256 gate/up weight rows) to one CTA; when there are few blocks, most SMs stay idle and each busy SM
// works through all K / 128 stages of its block, eight WGMMAs per warp-group each. The split kernel halves every CTA's
// rows but pays for it with a longer epilogue (the up values handed between warp-groups, the amax exchanged between the
// two CTAs through distributed shared memory), about 1 us of serial latency. Measured on H200
// (/data/bench-runs/sm90_dg2_moe_20261005/phase2a; cold-L2 weights, K = 2048, block_n = 16): at 32 blocks (I = 512,
// M = 1) the split kernel ran 9.12 us against 10.28 us; at 48 blocks (I = 768, M = 1) 11.06 against 10.90, and at
// 64 blocks (I = 512, M = 2) 13.57 against 13.45. It only wins while the unsplit kernel occupies at most a quarter of
// the SMs, hence the rule: 4 * (p_max / block_n activation tiles, an upper bound on the routed tiles) * (I / 128
// blocks per tile) <= SMs. The output is bit-identical either way; only the launch shape changes. Not with two CTAs
// per SM forced (FSO_SWAPAB_CTAS_PER_SM=2), and only on an even SM count.
inline bool sm90_swapab_split_route(uint32_t p_max, uint32_t shape_n, uint32_t block_n, int num_device_sms)
{
    int const forced = sm90_env_swapab_split();
    if (forced >= 0)
        return forced == 1 && num_device_sms % 2 == 0;
    if (sm90_env_swapab_ctas() == 2u || num_device_sms % 2 != 0)
        return false;
    uint64_t const blocks = static_cast<uint64_t>(p_max / block_n) * (shape_n / 256u);
    return 4u * blocks <= static_cast<uint64_t>(num_device_sms);
}

// Stages of the split kernel: the largest count that divides K / 128 (no NotDivisibleK tail copy of the pipeline, as
// the unsplit kernel's rule) and whose shared memory fits one CTA per SM. 8 at K = 2048 and block_n = 16 (152,704 B).
inline uint32_t sm90_swapab_split_stages(uint32_t shape_k, uint32_t block_n)
{
    constexpr uint32_t kSm90MaxSmem = 227u * 1024u;
    constexpr uint32_t kReservedSmemPerCta = 1024u;
    uint32_t const k_stages = shape_k / 128u;
    for (uint32_t stages = k_stages; stages > 1u; --stages)
    {
        if (k_stages % stages == 0u
            && deep_gemm::sm90_swapab_split_smem::total_bytes(block_n, stages) + kReservedSmemPerCta <= kSm90MaxSmem)
            return stages;
    }
    return 1u;
}

// The split kernel has no register rebalancing (384 threads, one CTA per SM, no setmaxnreg), so any count ptxas
// chooses is safe; a build that spills to local memory is refused (the caller then takes the unsplit kernel). Checked
// once per cubin, like sm90_check_swapab_pair_one_cta.
inline bool sm90_swapab_split_build_ok(cudaKernel_t kernel)
{
    static std::vector<std::pair<void*, bool>> s_checked;
    for (auto const& entry : s_checked)
    {
        if (entry.first == reinterpret_cast<void*>(kernel))
            return entry.second;
    }
    int local_bytes = -1;
    bool const ok = cuKernelGetAttribute(&local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                        reinterpret_cast<CUkernel>(kernel), 0)
            == CUDA_SUCCESS
        && local_bytes == 0;
    if (!ok)
        std::fprintf(stderr,
            "fish_scales_ops: the cluster-split fused swap-AB FC1 cubin spills (%d bytes of local memory per thread); "
            "the one-CTA-per-block kernel runs instead\n",
            local_bytes);
    s_checked.emplace_back(reinterpret_cast<void*>(kernel), ok);
    return ok;
}

// Contiguous grouped FC1 on sm_90, swap-AB, with the SwiGLU + 1x128 FP8 requantize fused into the epilogue (fused-FC1
// P2a, design S2; fp8_gemm_kernel_swapAB_swiglu). Same inputs as gemm_dispatch_sm90_grouped_contiguous_swapab_pair
// (mat_a_wgt is the stacked [gate; up] weight [G, shape_n = 2I, K]); the outputs are what
// silu_chunk_mul_quantize_1x128_sorted_sm90 makes of that FC1's bf16 result: mat_dq [p_max, I] fp8 and
// mat_sd [I / 128, sd_ld] fp32 (sd_ld = align4(p_max)). The launch plan is sm90_swapab_pair_plan on the fused shared-
// memory layout (an fp8 staging tile and the amax partials instead of the two bf16 tiles), with the same register
// guards; at block_n = 16 and K = 2048 one CTA runs 4 stages (143,360 B), and two CTAs 2 stages each. Where the
// SwiGLU blocks are few enough (sm90_swapab_split_route), the cluster-split kernel runs instead.
inline void gemm_dispatch_sm90_grouped_contiguous_swapab_swiglu(void* mat_a_wgt, void* mat_b_act, void* mat_dq,
    float* mat_sd, float* sfb_wgt, float* sfa_act, int* sorted_expert_ids, uint32_t num_groups, uint32_t p_max,
    uint32_t shape_n, uint32_t shape_k, uint32_t block_n, uint32_t expected_m, cudaStream_t stream,
    int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

    constexpr uint32_t block_k = 128;
    constexpr uint32_t bm = 128u;
    (void) expected_m;
    if (block_n != 16u && block_n != 32u && block_n != 64u)
        TLLM_THROW("fused SwiGLU swap-AB FC1: block_n must be 16, 32 or 64 (= the moe_build_sorted padding), got %u",
            block_n);
    if (shape_n % (2u * bm) != 0u || shape_k % block_k != 0u)
        TLLM_THROW("fused SwiGLU swap-AB FC1: N must be a multiple of 256 (2I, I % 128 == 0) and K of 128, "
                   "got N=%u K=%u",
            shape_n, shape_k);

    uint32_t const sd_ld_split = (p_max + 3u) / 4u * 4u;
    if (sm90_swapab_split_route(p_max, shape_n, block_n, num_device_sms))
    {
        uint32_t const stages = sm90_swapab_split_stages(shape_k, block_n);
        auto runtime = deep_gemm::jit::getGlobalCompiler().build(shape_n, shape_k, bm, block_n, block_k, num_groups,
            stages, 1u, deep_gemm::GemmType::GroupedContiguous, true, 1u, false, true, true, true);
        auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
        if (sm90_swapab_split_build_ok(kernel))
        {
            deep_gemm::runGemmSwapABSwigluSplit(kernel, mat_a_wgt, static_cast<int>(shape_k), mat_b_act,
                static_cast<int>(shape_k), mat_dq, mat_sd, sd_ld_split, sfb_wgt, sfa_act, shape_n, p_max, shape_k,
                block_n, block_k, num_groups, sorted_expert_ids, stream, num_device_sms,
                deep_gemm::sm90_swapab_split_smem::total_bytes(block_n, stages));
            return;
        }
    }

    auto build = [&](Sm90SwapAbPairPlan const& plan)
    {
        auto runtime = deep_gemm::jit::getGlobalCompiler().build(shape_n, shape_k, bm, block_n, block_k, num_groups,
            plan.stages, 1u, deep_gemm::GemmType::GroupedContiguous, true, plan.ctas, false, true, true);
        return reinterpret_cast<cudaKernel_t>(runtime->getKernel());
    };
    Sm90SwapAbPairPlan plan
        = sm90_swapab_pair_plan(p_max, shape_n, shape_k, block_n, true, sm90_env_swapab_ctas(), num_device_sms);
    cudaKernel_t kernel = build(plan);
    if (plan.ctas == 2u && !sm90_swapab_two_cta_regs_ok(kernel, "fused SwiGLU swap-AB FC1", true))
    {
        plan = sm90_swapab_pair_plan(p_max, shape_n, shape_k, block_n, true, 1u, num_device_sms);
        kernel = build(plan);
    }
    if (plan.ctas == 1u)
        sm90_check_swapab_pair_one_cta(kernel, "fused SwiGLU swap-AB FC1");

    uint32_t const sd_ld = (p_max + 3u) / 4u * 4u;
    deep_gemm::runGemmSwapABSwiglu(kernel, mat_a_wgt, static_cast<int>(shape_k), mat_b_act, static_cast<int>(shape_k),
        mat_dq, mat_sd, sd_ld, sfb_wgt, sfa_act, shape_n, p_max, shape_k, bm, block_n, block_k, num_groups,
        sorted_expert_ids, stream, static_cast<int>(num_device_sms * plan.ctas), plan.smem);
}

} // namespace kernels::blockscale_gemm
TRTLLM_NAMESPACE_END
