/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SM120 MXFP8 (1×32, kSFVecSize=32) dispatch and kernel launcher.
//
// Mirrors `sm120_dispatch.cuh` for the VS=32 path. Reuses the same
// `SM120BlockScaledKernel<KT>` template (in `fp8/gemm_1d1d.cuh`) — only
// the builder differs. The (M, tiles_n) cascade and Stream-K logic
// match the 1×128 path bit-for-bit (C2 ports the 5 production tile
// templates and the K≥9728 Stream-K guard from `fea4158`). C6 will
// retune cascade thresholds based on a VS=32-specific tile sweep.

#pragma once

#include "blockscale_gemm/common/epilogue.cuh"
#include "blockscale_gemm/common/kernel_utils.cuh"
#include "blockscale_gemm/arch/sm120/common/env_overrides.cuh"
#include "blockscale_gemm/arch/sm120/fp8/gemm_1d1d.cuh"  // SM120BlockScaledKernel<KT>
#include "blockscale_gemm/arch/sm120/mxfp8/utils.cuh"  // SM120MxFP8BlockScaledBuilder
#include "blockscale_gemm/arch/sm120/mxfp8/smallm_kernel.cuh"  // SM120MxFP8SmallMKernel<KT> (E28)
#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

TRTLLM_NAMESPACE_BEGIN
namespace kernels::blockscale_gemm
{

// CompactActs (with `blk0` and `act_rows`) launches the compact-activation-layout
// instantiation of a grouped kernel: A, SFA and D are one group of `act_rows` rows in
// which expert g's rows start at row blk0[g] * 128, while B / SFB / grouped_layout keep
// the group index (see SM120BlockScaledKernel). shape_m stays the masked layout's
// row capacity m_cap, which only the grid bound below reads (the cascade above the
// launcher reads it as its tile hint), so a compact launch has the grid of the masked
// launch it replaces.
//
// DenseEpi (with `epilogue`) launches the dense-epilogue instantiation of a dense tile: the kernel over
// SM120DenseEpiBuilder<builder>, which applies epilogue->bias / residual / act to the fp32 accumulator before the
// one rounding (fp8/gemm_1d1d.cuh, apply_dense_epilogue). Its fields are per-call pointers, patched into Params
// after the cache lookup like grouped_layout. Every other launch passes DenseEpi = false and is the launch it was.
//
// DenseQuant (with DenseEpi and `epilogue`) launches the quantized-output instantiation instead: the kernel over
// SM120DenseQuantBuilder<builder> (fso.dense.linear_quantized), which writes epilogue->out_q and its scale words
// (epilogue->out_scale, the sm_120 contract layout) and never the bf16 D; `mat_d` is then out_q itself, only so that
// the D descriptor Params carries is built over a valid, aligned address (the fused SwiGLU's precedent: never
// issued, never read) and the Params cache key differs from a bf16 call's.
template <int TileM, int TileN, int NumStages, int MinBlocksPerSm = 1, int SchedGroup = 16,
    bool SeparateSmemD = false, bool GroupedLayoutSmem = false, bool FusedSwiGLU = false,
    bool ScatterCombine = false, bool ScatterInStoreWarp = false, bool CompactActs = false, bool DenseEpi = false,
    bool DenseQuant = false>
void launch_sm120_mxfp8_gemm_kernel(__nv_fp8_e4m3* mat_a, int64_t ld_a, int64_t stride_a, __nv_fp8_e4m3* mat_b,
    int64_t ld_b, int64_t stride_b, __nv_bfloat16* mat_d, int64_t ld_d, int64_t stride_d, int32_t* scales_a,
    int64_t /*stride_scales_a*/, int32_t* scales_b, int64_t /*stride_scales_b*/, uint32_t num_problems,
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, cudaStream_t stream, int num_device_sms = kNumDeviceSMs,
    int32_t* grouped_layout = nullptr, __nv_fp8_e4m3* out_fp8 = nullptr, int32_t* out_sf = nullptr,
    __nv_bfloat16* out_tokens = nullptr, int32_t const* row_map = nullptr, float const* weight_of_slot = nullptr,
    uint32_t max_active_groups = 0, int32_t const* blk0 = nullptr, uint32_t act_rows = 0,
    fso::DenseEpilogue const* epilogue = nullptr)
{
    if (num_device_sms < 0)
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

    using ElementInput = cute::float_e4m3_t;
    using ElementOutput = cute::bfloat16_t;
    using ElementBlockScale = int32_t;
    using KTBase = sm120_blockscaled_gemm::SM120MxFP8BlockScaledBuilder<TileM, TileN, NumStages, MinBlocksPerSm, SchedGroup, SeparateSmemD, GroupedLayoutSmem>;
    static_assert(!DenseQuant || DenseEpi, "the quantized output is a form of the dense epilogue");
    using KTEpi = cute::conditional_t<DenseEpi, sm120_blockscaled_gemm::SM120DenseEpiBuilder<KTBase>, KTBase>;
    using KT = cute::conditional_t<DenseQuant, sm120_blockscaled_gemm::SM120DenseQuantBuilder<KTBase>, KTEpi>;
    using GemmKernel = sm120_blockscaled_gemm::SM120BlockScaledKernel<KT, FusedSwiGLU, ScatterCombine,
        ScatterInStoreWarp, CompactActs>;
    using Params = typename GemmKernel::Params;
    using Arguments = typename GemmKernel::Arguments;
    using ProblemShape = typename GemmKernel::ProblemShape;
    ProblemShape problem_shape = make_shape((int) shape_m, (int) shape_n, (int) shape_k, (int) num_problems);

    auto ptr_A = reinterpret_cast<ElementInput*>(mat_a);
    auto ptr_B = reinterpret_cast<ElementInput*>(mat_b);
    auto ptr_SFA = reinterpret_cast<ElementBlockScale*>(scales_a);
    auto ptr_SFB = reinterpret_cast<ElementBlockScale*>(scales_b);
    auto ptr_D = reinterpret_cast<ElementOutput*>(mat_d);

    typename KT::StrideA dA = make_stride(ld_a, Int<1>{}, stride_a);
    typename KT::StrideB dB = make_stride(ld_b, Int<1>{}, stride_b);
    // SFA follows the activation tensor: the problem shape, or with the compact layout
    // one group of act_rows rows.
    typename KT::StrideSFA dSFA
        = KT::deduce_sfa_layout(GemmKernel::act_problem_shape(problem_shape, static_cast<int>(act_rows))).stride();
    typename KT::StrideSFB dSFB = KT::deduce_sfb_layout(problem_shape).stride();
    typename KT::StrideD dD = make_stride(ld_d, Int<1>{}, stride_d);

    // The fused SwiGLU epilogue stores MXFP8 rows and their scale bytes itself,
    // so the bf16 D tensor is never written; its TMA descriptor is still built
    // (Params carries it unconditionally) over the FP8 slab's base, which is
    // valid, aligned and never read.
    // Grouped launches may walk the experts in a permuted order (FSO_MOE_GROUP_ORDER);
    // dense ones always walk 0..L-1. Resolved here, once, because this is the only
    // place that sees both the grouped_layout pointer and the expert count.
    GroupOrder const order
        = (grouped_layout != nullptr) ? sm120_group_perm_for(static_cast<int>(num_problems)) : GroupOrder{};
    Arguments args{ptr_A, dA, ptr_B, dB, ptr_SFA, dSFA, ptr_SFB, dSFB, ptr_D, dD, grouped_layout, out_fp8, out_sf,
        static_cast<int>(shape_m), static_cast<int>(shape_n) / 2, out_tokens, row_map, weight_of_slot, order.stride,
        order.block};
    if constexpr (CompactActs)
    {
        // The fused epilogues address the one-group view: a slab of act_rows rows, group 0.
        args.m_cap = static_cast<int>(act_rows);
        args.blk0 = blk0;
        args.act_rows = static_cast<int>(act_rows);
    }

    // E9 (2026-05-10): cache Params (which holds 5 CUtensorMap descriptors) to
    // skip the 5x cuTensorMapEncodeTiled calls inside `to_underlying_arguments`.
    // Each call costs ~3 µs host-side → ~15 µs total, which is THE small-M
    // launch bottleneck (M=1 down: BF16 enqueue 8 µs, MXFP8 enqueue 22 µs after
    // E8; the 14 µs gap matches the 5x TMA encode cost).
    //
    // The cache hits when consecutive calls share (ptrs, shape, strides). For
    // PyTorch inference: weight ptr_B / ptr_SFB stay stable, and torch's CUDA
    // caching allocator usually returns the same memory for the recycled
    // activation buffers (ptr_A, ptr_SFA, ptr_D), so hit rate is high in the
    // tight bench loop and the production forward path.
    //
    // The cache is per-instantiation and thread-local; size capped at 64
    // entries (~45 KB) so a workload with many unique shapes doesn't blow up.
    // dims[4] is the compact layout's act_rows (0 otherwise): it sizes the A / SFA / D
    // descriptors, so two compact calls that differ only in it must not share Params.
    struct CacheKey {
        void* p[5]; int dims[5]; int64_t ld[3]; int64_t st[3];
        bool operator==(CacheKey const& o) const { return std::memcmp(this, &o, sizeof(o)) == 0; }
    };
    struct CacheHash {
        std::size_t operator()(CacheKey const& k) const noexcept {
            std::size_t h = 1469598103934665603ULL;
            auto const* p = reinterpret_cast<unsigned char const*>(&k);
            for (std::size_t i = 0; i < sizeof(k); ++i) { h ^= p[i]; h *= 1099511628211ULL; }
            return h;
        }
    };
    static thread_local std::unordered_map<CacheKey, Params, CacheHash> s_params_cache;
    CacheKey key;
    std::memset(&key, 0, sizeof(key));
    key.p[0] = ptr_A; key.p[1] = ptr_B; key.p[2] = ptr_SFA; key.p[3] = ptr_SFB; key.p[4] = ptr_D;
    key.dims[0] = (int) shape_m; key.dims[1] = (int) shape_n; key.dims[2] = (int) shape_k; key.dims[3] = (int) num_problems;
    key.dims[4] = CompactActs ? (int) act_rows : 0;
    key.ld[0] = ld_a; key.ld[1] = ld_b; key.ld[2] = ld_d;
    key.st[0] = stride_a; key.st[1] = stride_b; key.st[2] = stride_d;

    Params kernel_params;
    auto cache_it = s_params_cache.find(key);
    if (cache_it != s_params_cache.end())
    {
        kernel_params = cache_it->second;
    }
    else
    {
        kernel_params = GemmKernel::to_underlying_arguments(problem_shape, args);
        if (s_params_cache.size() < 64u)
            s_params_cache.emplace(key, kernel_params);
    }
    // Grouped path: the masked_m buffer address is not part of the cache key
    // (it may be reallocated between calls while shapes stay identical), and
    // Params only carries the raw pointer — patch it after every cache
    // lookup so a cached Params never replays a stale grouped_layout.
    kernel_params.grouped_layout = grouped_layout;
    if constexpr (CompactActs)
    {
        // Same reason: blk0 is a per-call routing output.
        kernel_params.blk0 = blk0;
    }
    // Not part of the cache key (it is a process-wide setting), but a cached
    // Params must not carry another launch's order.
    kernel_params.group_perm = order.stride;
    kernel_params.group_block = order.block;
    if constexpr (FusedSwiGLU)
    {
        // Same reason as grouped_layout: these buffers are per-call.
        kernel_params.out_fp8 = out_fp8;
        kernel_params.out_sf = out_sf;
    }
    if constexpr (ScatterCombine)
    {
        kernel_params.out_tokens = out_tokens;
        kernel_params.row_map = row_map;
        kernel_params.weight_of_slot = weight_of_slot;
    }
    if constexpr (DenseEpi)
    {
        // Same reason: the epilogue's operands are per-call.
        TLLM_CHECK_WITH_INFO(epilogue != nullptr, "sm120 mxfp8 dense-epilogue launch without its epilogue");
        kernel_params.epi_bias = epilogue->bias;
        kernel_params.epi_residual = epilogue->residual;
        kernel_params.epi_ld_residual = epilogue->ld_residual;
        kernel_params.epi_act = epilogue->act;
    }
    if constexpr (DenseQuant)
    {
        kernel_params.epi_out_q = epilogue->out_q;
        kernel_params.epi_out_scale = reinterpret_cast<int32_t*>(epilogue->out_scale);
    }
#if defined(FSO_PROLOGUE_TRACE)
    // Scratch builds only: see the FSO_PROLOGUE_TRACE block in fp8/gemm_1d1d.cuh.
    kernel_params.trace = sm120_blockscaled_gemm::prologue_trace::buffer_from_env();
#endif
    auto kernel_ptr = &cutlass::device_kernel<GemmKernel>;

    // E8 (2026-05-10): cudaFuncSetAttribute is idempotent per (function, device,
    // attribute) — only the first call per process needs the syscall path.
    // Saves ~0.8 µs/launch in the small-M regime where host overhead dominates.
    // The static guard works because each template instantiation has its own
    // function symbol and its own kSmemSize.
    static bool s_smem_configured = false;
    if (!s_smem_configured)
    {
        cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, GemmKernel::kSmemSize);
        auto result = cudaGetLastError();
        TLLM_CHECK_WITH_INFO(result == cudaSuccess, "sm120 mxfp8 gemm kernel cannot launch: %s", cudaGetErrorString(result));
        s_smem_configured = true;
    }

    cudaLaunchConfig_t launch_config;
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = fso_pdl_enabled() ? 1 : 0;
    unsigned num_attrs = 1;
    // MinBlocksPerSm > 1 instances co-schedule that many CTAs per SM (the
    // persistent scheduler strides by gridDim.x, so a larger grid just
    // partitions the same tile queue). ncu on the grouped decode cells
    // showed every pipe under 55% SOL at 1 CTA/SM — latency-bound with the
    // co-residency the 44 KB (16,64,4) instance already fits going unused.
    int grid = num_device_sms * KT::MinBlocksPerSm;
    if (grouped_layout != nullptr && max_active_groups > 0)
    {
        // Grouped decode launches get no more CTAs than they can have tiles
        // (max_active_groups x ceil(m_cap / TileM) x ceil(N / TileN), the host-static
        // bound the op passes; the scheduler strides by gridDim.x, so any grid size
        // is correct). A grid of every SM slot puts CTAs on the SMs the previous
        // kernel of the chain still holds -- PDL launches this grid while it runs --
        // and those CTAs cannot start until it leaves (run sm120_prologue_20260928).
        uint64_t const tiles = uint64_t(max_active_groups) * ((shape_m + TileM - 1) / TileM)
            * ((shape_n + TileN - 1) / TileN);
        if (tiles < uint64_t(grid))
            grid = tiles > 0 ? int(tiles) : 1;
    }
    else if (grouped_layout == nullptr)
    {
        // Dense launches (issue #12 F5B): where the CTAs of a launch run, by the number of tiles against the CTA slots
        // (SMs x MinBlocksPerSm):
        //   tiles <= 0.8 x slots: one CTA per tile, as a 1 x 1 x 1 cluster launch;
        //   slots <= tiles < 4 x slots: the persistent grid of every slot, as a 1 x 1 x 1 cluster launch;
        //   otherwise: the persistent grid of every slot without a cluster attribute (fp8-round's launch).
        // None of the three changes which CTA computes which tile or how (the scheduler gives CTA b the tiles b,
        // b + grid, ...; with grid >= tiles that is tile b alone either way), so the kernel and the bytes are the
        // same. Why: the CTA dispatcher places consecutive CTAs of a launch without a cluster attribute on the SMs of
        // one GPC before it moves to the next, so the tile CTAs of an under-filled launch run packed on a few GPCs (up
        // to 14 per GPC for 80 tiles on the RTX 5090), where they share each GPC's path to L2, and a multi-wave launch
        // runs the tiles of its partial last wave on the first GPCs alone. Any cluster launch, even of size 1,
        // spreads consecutive CTAs across GPCs (the mechanism issue #12 FH6 found on the H200), and a grid of the tile
        // count leaves no idle CTA on the GPCs it fills first. The CuTe DSL kernel of issue #12 R040 (the same tile,
        // stages and data) launches this way, and the per-tile gap to it (16-38 %) was this placement. Between the two
        // bounds nearly every GPC already holds tile CTAs, and the cluster launch only cost time: on the DRAM-bound
        // gate_up 19456 x 2560 decode cells (152 tiles) 1.0-1.3 % with PDL off, with either grid, and on the tower proj2
        // 2048 x 1280 at M = 1079 (144 tiles, (128, 128, 2), bias epilogue) 2.5 % on the pre-quantized call (18.89 ->
        // 19.36 us; the bf16-x call 22.48 -> 21.81 us). From 4 waves up, the cluster launch made the bf16-x call's step
        // from the activation quantize into the GEMM slower in proportion to the size of the call, which ate the
        // pre-quantized call's gain: the bf16-x call minus the pre-quantized call took 6.0 -> 10.6 us on gate_up at M =
        // 1024 (7 waves, +1.7 % on that call), 32.6 -> 84.7 us at M = 8192 (46 waves); below 4 waves it took the same or
        // less (5120 x 2048 at M = 512: 3.4 -> 2.1 us). Measured on the RTX
        // 5090 (2407 MHz, cold weights, graph replay, the launches interleaved in one process, PDL on / off): wo 2560 x
        // 4096 at M = 64 on (32, 64, 4), 80 tiles, 11.75 -> 10.16 us; at M = 128 on (64, 64, 4) 14.72 -> 12.23 us;
        // 1280 x 7680 at M = 364, 120 tiles, 28.98 -> 27.73 / 28.85 -> 23.71 us; 2048 x 512 at M = 2048, 1.5 waves,
        // 15.92 -> 15.51 / 16.25 -> 15.07 us; gate_up at M = 2, 152 tiles, 35.91 -> 35.96 / 36.45 -> 36.92 us as a
        // cluster launch. In the calls of fso.dense.linear the 43 cells of 1 to 4 waves of the final A/B took 0.965-1.004
        // of fp8-round's time on the pre-quantized call and 0.937-1.000 on the bf16-x call. The cubin's TMA copies are shared::cta, so a cluster launch adds no driver syscall buffer (no
        // __cuda_syscall in the cubin; the free device memory at the first cluster launch does not move).
        uint64_t const tiles
            = uint64_t(num_problems) * ((shape_m + TileM - 1) / TileM) * ((shape_n + TileN - 1) / TileN);
        bool const one_cta_per_tile = 5 * tiles <= 4 * uint64_t(grid);
        if (one_cta_per_tile || (tiles >= uint64_t(grid) && tiles < 4 * uint64_t(grid)))
        {
            if (one_cta_per_tile)
                grid = tiles > 0 ? int(tiles) : 1;
            attrs[1].id = cudaLaunchAttributeClusterDimension;
            attrs[1].val.clusterDim.x = 1;
            attrs[1].val.clusterDim.y = 1;
            attrs[1].val.clusterDim.z = 1;
            num_attrs = 2;
        }
    }
    launch_config.gridDim = dim3(grid, 1, 1);
    launch_config.blockDim = GemmKernel::get_block_shape();
    launch_config.dynamicSmemBytes = GemmKernel::kSmemSize;
    launch_config.stream = stream;
    launch_config.attrs = attrs;
    launch_config.numAttrs = num_attrs;
    cudaLaunchKernelEx(&launch_config, kernel_ptr, kernel_params);
    // The post-launch cudaGetLastError check was removed (was ~0.5 µs/launch).
    // Kernel runtime errors surface on next cudaStreamSynchronize.
}

// E28 (2026-05-11): Forced-only experimental SmallM launcher. Uses
// `SM120MxFP8SmallMKernel<KT>` which replaces A's TMA load with per-row
// cp.async.bulk. TileM is fixed at 16 (sm_120 MXFP8 mma atom floor).
// Enabled only via `BSGEMM_FORCE_SMALLM=1` combined with
// `BSGEMM_FORCE_TILE=16,TN,ST`. Not on the default route.
template <int TileN, int NumStages>
void launch_sm120_mxfp8_smallm_gemm_kernel(__nv_fp8_e4m3* mat_a, int64_t ld_a, int64_t stride_a,
    __nv_fp8_e4m3* mat_b, int64_t ld_b, int64_t stride_b, __nv_bfloat16* mat_d, int64_t ld_d, int64_t stride_d,
    int32_t* scales_a, int64_t /*stride_scales_a*/, int32_t* scales_b, int64_t /*stride_scales_b*/,
    uint32_t num_problems, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, cudaStream_t stream,
    int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

    using ElementInput = cute::float_e4m3_t;
    using ElementOutput = cute::bfloat16_t;
    using ElementBlockScale = int32_t;
    using KT = sm120_blockscaled_gemm::SM120MxFP8BlockScaledBuilder<16, TileN, NumStages>;
    using GemmKernel = sm120_blockscaled_gemm::SM120MxFP8SmallMKernel<KT>;
    using Params = typename GemmKernel::Params;
    using Arguments = typename GemmKernel::Arguments;
    using ProblemShape = typename GemmKernel::ProblemShape;
    ProblemShape problem_shape = make_shape((int) shape_m, (int) shape_n, (int) shape_k, (int) num_problems);

    auto ptr_A = reinterpret_cast<ElementInput*>(mat_a);
    auto ptr_B = reinterpret_cast<ElementInput*>(mat_b);
    auto ptr_SFA = reinterpret_cast<ElementBlockScale*>(scales_a);
    auto ptr_SFB = reinterpret_cast<ElementBlockScale*>(scales_b);
    auto ptr_D = reinterpret_cast<ElementOutput*>(mat_d);

    typename KT::StrideA dA = make_stride(ld_a, Int<1>{}, stride_a);
    typename KT::StrideB dB = make_stride(ld_b, Int<1>{}, stride_b);
    typename KT::StrideSFA dSFA = KT::deduce_sfa_layout(problem_shape).stride();
    typename KT::StrideSFB dSFB = KT::deduce_sfb_layout(problem_shape).stride();
    typename KT::StrideD dD = make_stride(ld_d, Int<1>{}, stride_d);

    Arguments args{ptr_A, dA, ptr_B, dB, ptr_SFA, dSFA, ptr_SFB, dSFB, ptr_D, dD};

    // Cache Params (3× TMA descriptors instead of 5× for the regular kernel —
    // SmallM skips TMA_A and TMA_SFA-no-actually-keeps-SFA. We drop TMA_A only.)
    struct CacheKey {
        void* p[5]; int dims[4]; int64_t ld[3]; int64_t st[3];
        bool operator==(CacheKey const& o) const { return std::memcmp(this, &o, sizeof(o)) == 0; }
    };
    struct CacheHash {
        std::size_t operator()(CacheKey const& k) const noexcept {
            std::size_t h = 1469598103934665603ULL;
            auto const* p = reinterpret_cast<unsigned char const*>(&k);
            for (std::size_t i = 0; i < sizeof(k); ++i) { h ^= p[i]; h *= 1099511628211ULL; }
            return h;
        }
    };
    static thread_local std::unordered_map<CacheKey, Params, CacheHash> s_params_cache;
    CacheKey key;
    std::memset(&key, 0, sizeof(key));
    key.p[0] = ptr_A; key.p[1] = ptr_B; key.p[2] = ptr_SFA; key.p[3] = ptr_SFB; key.p[4] = ptr_D;
    key.dims[0] = (int) shape_m; key.dims[1] = (int) shape_n; key.dims[2] = (int) shape_k; key.dims[3] = (int) num_problems;
    key.ld[0] = ld_a; key.ld[1] = ld_b; key.ld[2] = ld_d;
    key.st[0] = stride_a; key.st[1] = stride_b; key.st[2] = stride_d;

    Params kernel_params;
    auto cache_it = s_params_cache.find(key);
    if (cache_it != s_params_cache.end())
    {
        kernel_params = cache_it->second;
    }
    else
    {
        kernel_params = GemmKernel::to_underlying_arguments(problem_shape, args);
        if (s_params_cache.size() < 64u)
            s_params_cache.emplace(key, kernel_params);
    }
    auto kernel_ptr = &cutlass::device_kernel<GemmKernel>;

    static bool s_smem_configured = false;
    if (!s_smem_configured)
    {
        cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, GemmKernel::kSmemSize);
        auto result = cudaGetLastError();
        TLLM_CHECK_WITH_INFO(result == cudaSuccess,
            "sm120 mxfp8 smallm gemm kernel cannot launch: %s", cudaGetErrorString(result));
        s_smem_configured = true;
    }

    cudaLaunchConfig_t launch_config;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    launch_config.gridDim = dim3(num_device_sms, 1, 1);
    launch_config.blockDim = GemmKernel::get_block_shape();
    launch_config.dynamicSmemBytes = GemmKernel::kSmemSize;
    launch_config.stream = stream;
    launch_config.attrs = attrs;
    launch_config.numAttrs = 1;
    cudaLaunchKernelEx(&launch_config, kernel_ptr, kernel_params);
}

// The single launch of the dense-epilogue instance (tile_m, tile_n, stages) of a dense MXFP8 tile: the kernel over
// SM120DenseEpiBuilder<SM120MxFP8BlockScaledBuilder<tile_m, tile_n, stages>> with `epilogue`'s fields. False when the
// tile has no such instance. Defined in ops/mxfp8_dense_epi_kernel.cu, the translation unit that holds the
// instantiations (one per single-launch instance of the route), so that they compile in parallel with the identity
// instances of ops/mxfp8_kernel.cu.
bool launch_sm120_mxfp8_dense_epi_single(int tile_m, int tile_n, int stages, __nv_fp8_e4m3* mat_a,
    __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d, int32_t* scales_a, int32_t* scales_b, uint32_t shape_m,
    uint32_t shape_n, uint32_t shape_k, cudaStream_t stream, int num_device_sms, fso::DenseEpilogue const& epilogue);

// The quantized-output twins (fso.dense.linear_quantized), defined in ops/mxfp8_dense_quant_kernel.cu, the translation
// unit that holds their instantiations and the Stream-K reduce kernel, so that no other unit carries their code.
// launch_sm120_mxfp8_dense_quant_single: the single launch of the quantized-output instance (tile_m, tile_n, stages)
// (the kernel over SM120DenseQuantBuilder<SM120MxFP8BlockScaledBuilder<...>>), writing epilogue.out_q and
// epilogue.out_scale; false when the tile has no such instance.
// launch_sm120_mxfp8_streamk_reduce_quant: the Stream-K route's reduce of the K_split bf16 partials into
// epilogue.out_q / epilogue.out_scale (mxfp8/dense_quant_epi.cuh, sm120_streamk_reduce_quant_kernel).
bool launch_sm120_mxfp8_dense_quant_single(int tile_m, int tile_n, int stages, __nv_fp8_e4m3* mat_a,
    __nv_fp8_e4m3* mat_b, int32_t* scales_a, int32_t* scales_b, uint32_t shape_m, uint32_t shape_n,
    uint32_t shape_k, cudaStream_t stream, int num_device_sms, fso::DenseEpilogue const& epilogue);
void launch_sm120_mxfp8_streamk_reduce_quant(__nv_bfloat16 const* partials, int M, int N, int K_split,
    std::size_t partial_stride, fso::DenseEpilogue const& epilogue, cudaStream_t stream);

// The quantized output's own preconditions (host only), checked before anything is launched: its scale buffer in the
// sm_120 contract layout, no residual (linear_quantized takes none), N a multiple of 128 (it is the next GEMM's K).
inline void sm120_mxfp8_dense_require_quant_out(fso::DenseEpilogue const& epilogue, uint32_t shape_n)
{
    if (epilogue.out_q == nullptr)
        return;
    if (epilogue.residual != nullptr)
        throw fso::DenseEpilogueNotImplemented("sm_120/sm_121 dense MXFP8 GEMM (gemm_dispatch_sm120_mxfp8): the "
            "quantized output (linear_quantized) takes no residual");
    TLLM_CHECK_WITH_INFO(epilogue.out_scale != nullptr && epilogue.out_scale_layout == fso::kDenseOutScaleMxSm120,
        "sm120 mxfp8 quantized output: the scale buffer must be given in the sm_120 contract layout "
        "(kDenseOutScaleMxSm120), got %p / layout %d", epilogue.out_scale, epilogue.out_scale_layout);
    TLLM_CHECK_WITH_INFO(shape_n % 128 == 0, "sm120 mxfp8 quantized output: N = %u is not a multiple of 128", shape_n);
}

} // namespace kernels::blockscale_gemm
TRTLLM_NAMESPACE_END

// FSO_SM120_MXFP8_LAUNCHERS_ONLY (defined by ops/mxfp8_dense_epi_kernel.cu only): stop here, after the launchers.
// gemm_dispatch_sm120_mxfp8 below is a non-template inline function, so every translation unit that sees its body
// implicitly instantiates every kernel it names, launched or not; the epilogue TU must not, or it would carry a
// third copy of each identity kernel (ops/mxfp8_kernel.cu holds the one that runs, ops/mxfp8_compact_kernel.cu a
// second one since before issue #12).
#if !defined(FSO_SM120_MXFP8_LAUNCHERS_ONLY)

// Stream-K wrapper depends on launch_sm120_mxfp8_gemm_kernel above.
#include "blockscale_gemm/arch/sm120/mxfp8/streamk.cuh"

TRTLLM_NAMESPACE_BEGIN
namespace kernels::blockscale_gemm
{

// The dense cascade. Mirror of `gemm_dispatch_sm120` but for the
// VS=32 (MXFP8 1×32) path. Uses the same (M, tiles_n) cascade and the
// same `sm120_streamk_choose_k_split` heuristic — including the K≥9728
// Stream-K gate from commit fea4158. Env-var overrides
// (BSGEMM_FORCE_TILE / BSGEMM_FORCE_KSPLIT / BSGEMM_DISABLE_STREAMK /
// BSGEMM_DISABLE_OVERRIDES) follow the exact same wire format so a
// shared sweep harness can drive both paths.
//
// Tile differences vs the 1×128 path: (64, 128, 4) is replaced by
// (64, 128, 2) everywhere because at VS=32 + kTileSF=4 the per-CTA
// SMEM (A=32KB + B=64KB + SF=3KB) saturates the 99KB Blackwell consumer
// limit and `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` fails
// with `invalid argument`. Stages=2 brings AB down to 32+32=64KB, total
// well within budget. C6 will retune once the cuobjdump baseline is
// captured. The other 4 tiles ((32,128,4), (96,128,2), (128,128,2),
// (160,128,2)) fit unchanged — verified at TileM=160 N=128 ST=2:
// 40+32+2.5+2 = 76 KB.
//
// The cascade is sm120_mxfp8_dense_route_cascade below, and sm120_mxfp8_dense_route after it applies the wave-fill rule
// to the cascade's pick: the route returns the instance and the K split a call runs, and gemm_dispatch_sm120_mxfp8
// (further down) launches it. The cascade's rules are the ones this dispatcher has always applied, in
// the same order, with each launch statement replaced by the route it launched; the launcher's own clamps of the
// Stream-K split (sm120_streamk_effective_split) are applied here, so that the route states what runs. One source of
// truth serves the launch, the Stream-K scratch size the op body allocates (Sm120DenseRoute::streamk_scratch_bytes)
// and the host-side route query (sm120_mxfp8_dense_route_query in ops/mxfp8_kernel.cu).
inline Sm120DenseRoute sm120_mxfp8_dense_route_cascade(
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_device_sms)
{
    // The route of instance (tm, tn, st) with the given MinBlocksPerSm / SchedGroup, K split and SmallM flag; the
    // split is clamped as launch_sm120_mxfp8_streamk_gemm clamps it (one slice per 128-element scale word at most,
    // at most StreamKPool::kMaxKSplit slices; 1 means a single launch of the same tile).
    auto mx_route = [&](int tm, int tn, int st, int mb = 1, int sg = 16, int ks = 1, bool smallm = false)
    {
        Sm120DenseRoute r;
        r.tile_m = tm;
        r.tile_n = tn;
        r.stages = st;
        r.min_blocks = mb;
        r.sched_group = sg;
        r.k_split = sm120_streamk_effective_split(ks, static_cast<int>(shape_k), /*sf_block=*/128);
        r.smallm = smallm;
        return r;
    };

    // E15 (2026-05-10): considered routing M ≤ 4 to a hand-rolled scalar
    // GEMV (`common/gemv.cuh`) to skip TMA descriptor setup. NEGATIVE
    // RESULT: even with 31/32 M-row waste, the tensor-core matmul kernel
    // dominates a scalar GEMV — 170-SM Blackwell has ~970 TF (TC) vs ~21 TF
    // (scalar ALU). The GEMV kernel kept around for reference but NOT
    // dispatched; turn on via `BSGEMM_USE_GEMV=1` env if needed for
    // A/B testing. See `common/gemv.cuh` for the design notes.

    // BSGEMM_FORCE_TILE / BSGEMM_FORCE_KSPLIT — see env_overrides.cuh.
    // Tile combinations advertised here are MXFP8-specific (NS=2 at
    // TileM=64; the FP8 path uses NS=4 there but kSFVecSize=32 + Stages=4
    // would exceed the 99 KB SMEM budget).
    auto forced = read_force_tile();
    if (force_tile_applies(forced, shape_k))
    {
        int const k_split = forced.stream_k() ? forced.ks : 1;
        // E28 (2026-05-11): forced SmallM variant — manual cp.async.bulk
        // A loads. **NEGATIVE RESULT — keep behind force flag.** See
        // `smallm_kernel.cuh` and `docs/skills/blockscale-gemm-tuning/
        // references/sm120-mxfp8-smallm-manual-a-negative.md`. Manual A
        // ran 2-13× slower than TMA because the SW128 swizzle forces
        // 8 × M_actual 16-byte cp.async.bulks per stage. Default
        // dispatch is unchanged. Forced only via BSGEMM_FORCE_SMALLM=1
        // + BSGEMM_FORCE_TILE=16,128,4 for A/B reproduction.
        if (forced.smallm() && k_split <= 1 && forced.tm == 16)
        {
            if (forced.tn == 128 && forced.st == 4)
                return mx_route(16, 128, 4, 1, 16, 1, /*smallm=*/true);
        }
        // E29 T4 (2026-05-12): forced scheduler-group override for TileM=16.
        // BSGEMM_FORCE_SCHED_GROUP={8,32} selects an alt-SchedGroup launcher
        // for the TileM=16 small-M paths. Only 16x128x4 and 16x64x4 are
        // wired — these are the only TileM=16 tile shapes that the default
        // routes E24/E27 use. Forced-only experiment; default unchanged.
        if (forced.sched_group_forced() && k_split <= 1 && forced.tm == 16
            && (forced.sg == 8 || forced.sg == 32))
        {
            if (forced.tn == 128 && forced.st == 4)
                return mx_route(16, 128, 4, 1, forced.sg);
            if (forced.tn == 64 && forced.st == 4)
                return mx_route(16, 64, 4, 1, forced.sg);
        }
        if (k_split <= 1)
        {
            // 16x128x4 with optional MinBlocksPerSm=2 hint (T4 experiment).
            // BSGEMM_FORCE_MIN_BLOCKS=2 selects the (16, 128, 4, 2) builder
            // instead of the default (16, 128, 4, 1). SMEM = 74 KB per CTA
            // exceeds half the per-SM SMEM budget, so 2 CTAs/SM is
            // unlikely to materialise — the hint mostly throttles
            // register usage. Treat as forced-only diagnostic.
            // 16x128x2 evaluated (2026-05-11): negative result — stage-2
            // loses 1-19% vs stage-4 on every targeted cell (e.g., wqkv M=4
            // st2=12.34 vs st4=10.39). Deeper K-pipeline at stage-4 keeps
            // tensor cores fed; stage-2 stalls. Removed from forced list to
            // avoid drift; if reintroduced, A/B against `runs/tilem16_ab.jsonl`.
            // 16x64x4 doesn't compile: SM75_U32x4_LDSM_N for the B-fragment
            // needs more vals per thread than 8 N-warps × TileN=64 supply
            // (TiledNumVal % AtomNumVal != 0). Keep TileM=16 to TileN=128.
            if      (forced.tm == 16 && forced.tn == 128 && forced.st == 4 && forced.min_blocks_2())
                return mx_route(16, 128, 4, 2);
            else if (forced.tm ==  16 && forced.tn == 128 && forced.st == 4) return mx_route( 16, 128, 4);
            else if (forced.tm ==  16 && forced.tn ==  64 && forced.st == 4) return mx_route( 16,  64, 4);
            else if (forced.tm ==  32 && forced.tn == 128 && forced.st == 4) return mx_route( 32, 128, 4);
            else if (forced.tm ==  32 && forced.tn ==  64 && forced.st == 4) return mx_route( 32,  64, 4);
            else if (forced.tm ==  64 && forced.tn == 128 && forced.st == 2) return mx_route( 64, 128, 2);
            else if (forced.tm ==  64 && forced.tn ==  64 && forced.st == 4) return mx_route( 64,  64, 4);
            else if (forced.tm ==  96 && forced.tn == 128 && forced.st == 2) return mx_route( 96, 128, 2);
            else if (forced.tm == 128 && forced.tn == 128 && forced.st == 2) return mx_route(128, 128, 2);
            else if (forced.tm == 128 && forced.tn ==  64 && forced.st == 2) return mx_route(128,  64, 2);
            else if (forced.tm == 160 && forced.tn == 128 && forced.st == 2) return mx_route(160, 128, 2);
            // The 3-stage instances of the stage rule (issue #12 F5B).
            else if (forced.tm ==  64 && forced.tn == 128 && forced.st == 3) return mx_route( 64, 128, 3);
            else if (forced.tm ==  96 && forced.tn == 128 && forced.st == 3) return mx_route( 96, 128, 3);
            // The 6-stage decode instances (issue #12 F5B).
            else if (forced.tm ==  16 && forced.tn ==  64 && forced.st == 6) return mx_route( 16,  64, 6);
            else if (forced.tm ==  32 && forced.tn ==  64 && forced.st == 6) return mx_route( 32,  64, 6);
        }
        else
        {
            // 16x128x4 Stream-K is a T2 experiment (forced-only). The
            // tilem16_band production gate runs single-launch only; if a
            // future tune shows Stream-K wins on narrow-N small-M K≥4096,
            // promote this to the cascade via the same gate.
            if      (forced.tm ==  16 && forced.tn == 128 && forced.st == 4) return mx_route( 16, 128, 4, 1, 16, k_split);
            else if (forced.tm ==  32 && forced.tn == 128 && forced.st == 4) return mx_route( 32, 128, 4, 1, 16, k_split);
            else if (forced.tm ==  32 && forced.tn ==  64 && forced.st == 4) return mx_route( 32,  64, 4, 1, 16, k_split);
            else if (forced.tm ==  64 && forced.tn == 128 && forced.st == 2) return mx_route( 64, 128, 2, 1, 16, k_split);
            else if (forced.tm ==  64 && forced.tn ==  64 && forced.st == 4) return mx_route( 64,  64, 4, 1, 16, k_split);
            else if (forced.tm == 128 && forced.tn == 128 && forced.st == 2) return mx_route(128, 128, 2, 1, 16, k_split);
            else if (forced.tm == 128 && forced.tn ==  64 && forced.st == 2) return mx_route(128,  64, 2, 1, 16, k_split);
        }
        // unknown combo: fall through to the cascade
    }

        int const tiles_n = static_cast<int>((shape_n + 127) / 128);
    int const tiles_n64 = static_cast<int>((shape_n + 63) / 64);
    int const k_split = sm120_streamk_choose_k_split(static_cast<int>(shape_m), static_cast<int>(shape_n),
        static_cast<int>(shape_k), num_device_sms);

// The instance (TM_, TN_, ST_), single launch when k_split <= 1, else Stream-K with k_split slices.
#define DISPATCH_TILE_MX(TM_, TN_, ST_) return mx_route(TM_, TN_, ST_, 1, 16, k_split)

    if (!read_disable_overrides())
    {
        // Override 1: ported verbatim from the 1×128 cascade. Stream-K
        // never amortizes for K ≤ 2048, force single launch with the
        // smallest tile.
        // E28 (2026-05-28): cap the tiles_n<32 branch at M≤128 — without
        // that cap the override fires on cubic-1024/1536/2048 (tn=8/12/16,
        // K≤2048) and falls through to the (32,128,4) bottom-of-route,
        // which is calibrated for small-M / short-K. At M ≥ 1024 the
        // route gives 165/236/297 TF vs main-cascade (128,128,2)
        // 193/477/494 TF (sweep: tile_sweep_losers 2026-05-28). Mid-/
        // small-M paths (M≤32 universally, plus small_m_narrow_n M≤128 /
        // mid_m_narrow_n_64 M≤256 below) have their own M gates and are
        // unaffected.
        bool const short_k = shape_k <= 2048
                          && (shape_m <= 32 || (tiles_n < 32 && shape_m <= 128));

        // Override 2: MX-specific (added 2026-05-08, C8 small-M tune;
        // E20 refinement, 2026-05-11).
        // The shared heuristic returns k_split=2 at (tiles_n in [32, 96),
        // M <= 32) because the 1×128 path's SF reduce is light enough
        // to amortize. The 1×32 path has 4× more SF data, so the same
        // k_split=2 *can* pay a heavier reduce cost than the parallelism
        // win — but only sometimes. Re-validated against cudagraph µs:
        //
        //   gate (N=9728 tn=76):
        //     M=1..8  override 14.43 vs cascade 16.45 (+14% — keep)
        //     M=16    noise (both ~12.4)
        //     M=32    override 13.27 vs cascade 12.53 (-5.6% — drop)
        //   wqkv (N=6144 tn=48):
        //     M=1     override 12.37 vs cascade 13.42 (+8% — keep)
        //     M=4     override 14.41 vs cascade 12.37 (-14% — drop)
        //     M=8..32 marginal (both ~12.4) — but drop is cheaper
        //
        // So the original "M<=32, tn∈[32,96)" gate over-fires. The
        // refined condition fires only where override actually wins:
        //   (tn >= 64) AND (M <= 16)   — gate-style mid-wide N
        //   OR
        //   (M <= 2)                    — universal small-M (cascade
        //                                  with Stream-K reduce loses
        //                                  to a single TMA launch when
        //                                  the GEMM is M-light)
        // Source: `runs/rerun-tuning-2026-05-11/no_overrides.jsonl`.
        bool const small_m_mid_n =
            ((shape_m <= 16 && tiles_n >= 64 && tiles_n < 96)
             || (shape_m <= 2 && tiles_n >= 32 && tiles_n < 96))
            && shape_k <= 4096;

        // Override 3 (E7+E14, 2026-05-10): narrow-N small/mid-M. Two bands:
        //
        // small_m_narrow_n (M ∈ [1, 128], tn < 32, K < 9728): single-launch
        // (32, 64, 4). (32, 64, 4) gives 4 M-tiles × 2N-tiles-per-N=128 = 2×
        // more CTAs than (32, 128, 4) → 94% SM coverage at M=128, tn=20 vs
        // 47% for TileN=128. Kernel-only TF (cu130, 170-SM Blackwell):
        //   M=  4 N=2560 K=4096: (32,64,4)= 15 vs (32,128,4)=  5  (+200%)
        //   M= 32 N=2560 K=4096: (32,64,4)= 68 vs (32,128,4)= 48  (+42%)
        //   M=128 N=2560 K=4096: (32,64,4)=226 vs (32,128,4)=155  (+45%)
        //
        // mid_m_narrow_n_64 (M ∈ (128, 256], tn < 32, K < 9728): single-launch
        // (64, 64, 4). At M = 160..256 the workload per CTA grows enough
        // that TileM=32 is wasteful (too many CTAs of tiny work); TileM=64
        // + TileN=64 sweet spot. (Validated K=4096 only.)
        //   M=160 N=2560 K=4096: (64,64,4)=234 vs cascade (64,128,2)=~195 (+20%)
        //   M=192 N=2560 K=4096: (64,64,4)=284 vs cascade (64,128,2)=~231 (+23%)
        //   M=256 N=2560 K=4096: (64,64,4)=354 vs cascade (64,128,2)=~250 (+42%)
        //
        // E18 (2026-05-11): gate BOTH bands on K<9728. The earlier (E16)
        // gating bumped the small-M floor to 4 at K≥9728 but kept M=4..128
        // routed to single (32, 64, 4). End-to-end cudagraph µs (down
        // shape, N=2560 K=9728, baseline vs `BSGEMM_DISABLE_OVERRIDES=1`):
        //   M=  4: override single (32,64,4) 24.69 µs vs cascade+sk4 14.41 (-42%)
        //   M=  8: override 24.82 vs cascade 14.42 (-42%)
        //   M= 16: override 22.62 vs cascade 14.41 (-36%)
        //   M= 32: override 22.63 vs cascade 14.41 (-36%)
        //   M= 64: override 24.70 vs cascade 17.16 (-30%)
        //   M=128: override 27.04 vs cascade 20.53 (-24%)
        // The E16 kernel-only TF comparison (+54% at M=128 K=9728 vs
        // (64,128,2)+sk4) compared the WRONG cascade pick. The actual
        // cascade picks (32,128,4) + Stream-K k=4, not (64,128,2),
        // and that path wins decisively under cudagraph end-to-end.
        // Source: `runs/rerun-tuning-2026-05-11/no_overrides.jsonl`.
        bool const small_m_narrow_n = shape_k < 9728
                                   && shape_m >= 1 && shape_m <= 128
                                   && tiles_n >= 3 && tiles_n < 32;
        bool const mid_m_narrow_n_64 = shape_k < 9728
                                    && shape_m > 128 && shape_m <= 256
                                    && tiles_n >= 3 && tiles_n < 32;

        // E24 (2026-05-11): extreme small-M, wide-N TileM=16 route.
        // The hw MMA atom is 16x8x32 in M/N/K, so atom_M=16 is the floor.
        // A new SM120MxFP8 builder variant uses (1 warp M × 8 warps N) +
        // PermMmaTileM=Int<16> instead of the standard (2 warps M × 4 warps N)
        // + PermMmaTileM=Int<32>, halving the per-CTA M-tile waste at M ≤ 16
        // and doubling the in-flight CTAs across N.
        //
        // Cudagraph µs A/B (170-SM Blackwell, K=2560, all M ∈ {1, 2, 4, 8, 16}):
        //   wqkv    tn=48:  default=12.38 → 16x128x4=10.34   -16.5%
        //   gate    tn=76:  default=14.41 → 16x128x4=12.36   -14.3%
        //   gate_up tn=152: default=18.18 → 16x128x4=12.36   -32.0%
        // Beyond M=16 the gain reverses (TileM=32 fully utilises its
        // 32-row tile; e.g. gate_up M=32 16x128x4 = 22.63 vs default 14.42,
        // +57% LOSS) — that's the hard upper bound.
        // Narrow-N (tn < 32) regresses universally because TileN=64
        // streamk options exist; gate stays at tiles_n >= 48 so we don't
        // touch them.
        // Source: `runs/tilem16_ab.jsonl`.
        bool const tilem16_band = shape_k <= 4096
                               && shape_m <= 16
                               && tiles_n >= 48;

        if (tilem16_band)
        {
            return mx_route(16, 128, 4);
        }

        // E26 (2026-05-11): narrow-N + long-K + small-M → TileM=16 + Stream-K k=4.
        // Companion to tilem16_band for the `down`-style shape (Qwen3-4B
        // N=2560, K=9728, tn=20). Single-launch 16x128x4 catastrophically
        // loses on narrow-N because only 20 N-tiles × 1 M-tile = 20 CTAs
        // run on 170 SMs (12% wave utilisation). Stream-K k=4 quadruples
        // the in-flight CTAs to ~80 (~47% utilisation) while keeping
        // TileM=16's halved M-row waste.
        //
        // Cudagraph µs A/B (170-SM Blackwell, GPU 1 / GPU 6, `down` N=2560 K=9728):
        //   M= 1: default 14.45 → 16x128+ks4 13.15 / 12.44   -9% / -14%
        //   M= 2: default 15.42 → 14.24 / 13.54              -8% / -12%
        //   M= 4: default 15.65 → 14.26 / 14.33              -9% / -9%
        //   M= 8: default 14.44 → 13.26 / 12.38              -8% / -14%
        //   M=16: default 14.20 → 13.31 / 12.40              -6% / -10%
        // 16x128 single-launch (ks=1): +108..+127% (12% wave); ks=2:
        // +30..+50%. ks=4 is the sweet spot. Wins reproduce on both GPUs.
        // Source: `runs/tilem16_down_streamk.jsonl`.
        //
        // Gate on shape_k >= 8192 to keep the route inside the validated
        // (K=9728 only so far) regime; K < 8192 narrow-N small-M is owned
        // by E27 immediately below (TileM=16, TileN=64 single launch).
        //
        // Issue #12 F5B (2026-11-06): this band runs one launch of the 6-stage (16, 64) instance (ring mainloop), 40 CTAs
        // for N = 2560, instead of the Stream-K split. With one CTA per tile, spread as a cluster launch (see
        // launch_sm120_mxfp8_gemm_kernel), the single launch streams the weights faster than the four slices, their
        // bf16 partials and the reduce; it also runs on the caller's stream alone (the Stream-K slices run on the
        // StreamKPool's side streams, outside a green-context partition), and it sums K in fp32 in one pass instead of
        // through bf16 partial sums, so the bytes move toward the fp32 result. Measured on the RTX 5090 (2407 MHz, cold
        // weights, graph replay, the routes interleaved in one process, both with the new launch), down 2560 x 9728:
        // M = 1 / 4 / 16 Stream-K 21.44 / 21.07 / 21.15 us against 20.17 / 20.16 / 20.17 us (PDL off: 21.71 / 21.46 /
        // 21.48 against 20.89 / 20.72 / 20.74). 6 stages beat 9 (20.64-20.69 us) and 4 (21.72-21.75 us).
        bool const tilem16_narrow_streamk = shape_k >= 8192
                                         && shape_m <= 16
                                         && tiles_n >= 3 && tiles_n < 32;
        if (tilem16_narrow_streamk)
        {
            return mx_route(16, 64, 6);
        }

        // E27 (2026-05-11): narrow-N + short-K small-M → TileM=16, TileN=64.
        // Companion to E26 for shapes that have narrow N (tn∈[3,32)) but K
        // short enough that Stream-K doesn't pay off (K < 8192). The
        // (16, 64, 4) builder uses LDSM_x2 for the B-fragment so the
        // 8-N-warp × TileN=64 partition satisfies CuTe's vals-per-thread
        // divisibility check (LDSM_x4 fails, see SmemCopyAtomBForTileMN).
        //
        // Cudagraph µs A/B (170-SM Blackwell GPU 1, `wo` N=2560 K=4096 tn=20):
        //   M= 1: default (32,64,4)=12.41 → (16,64,4)=10.34   -16.7%
        //   M= 2: default            12.39 → 10.34            -16.6%
        //   M= 4: default            12.40 → 10.35            -16.5%
        //   M= 8: default            12.41 → 10.36            -16.6%
        //   M=16: default            12.39 → 10.32            -16.7%
        // Guard shapes confirm scope: `down` (tn=20 K=9728) loses
        // +40..+62% (E26 owns that band — excluded via shape_k<8192);
        // `gate` (tn=76 K=2560) is wide-N — excluded via tiles_n<32.
        // Source: `runs/tilem16_wo_tn64.jsonl`.
        bool const tilem16_narrow_single = shape_k < 8192
                                        && shape_m <= 16
                                        && tiles_n >= 3 && tiles_n < 32;
        if (tilem16_narrow_single)
        {
            return mx_route(16, 64, 4);
        }

        if (short_k || small_m_mid_n || small_m_narrow_n || mid_m_narrow_n_64)
        {
            // C9-followup: TileN=64 builder bug fixed (utils.cuh:60). Narrow-N
            // tile choice mirrors the cascade table:
            //   tn=1, M ≥ 4096        → (64, 64, 4)     (sweep 253 TF vs (32,64,4) 172)
            //   tn=1, M < 4096        → (32, 64, 4)     (+22~46% over TN=128)
            //   tn=2, M ≤ 1024        → (32, 64, 4)     (+54% over (32,128,4))
            //   tn=2, M ≥ 2048        → (32, 128, 4)    (TileN=64 regresses -21%)
            //   tn ≥ 3, M ∈ (128,256] → (64, 64, 4)     (E14: +20~42% over (64,128,2))
            //   tn ≥ 3, M ≤ 128       → (32, 64, 4)     (E14: +42~54% over TN=128)
            //   tn ≥ 3 (short_k only) → (32, 128, 4)    (small_m_mid_n path)
            if (tiles_n == 1 && shape_m >= 4096)
                return mx_route(64, 64, 4);
            else if (tiles_n == 1 || (tiles_n == 2 && shape_m <= 1024))
                return mx_route(32, 64, 4);
            else if (mid_m_narrow_n_64)
                // E14: M ∈ (128, 256] narrow-N → TileM=64, TileN=64.
                return mx_route(64, 64, 4);
            else if (small_m_narrow_n)
                // E7+E14: M ≤ 128 narrow-N (any K) → TileM=32, TileN=64.
                return mx_route(32, 64, 4);
            else
                return mx_route(32, 128, 4);
        }
    }

    // NOTE (2026-05-10, C9 + C9-followup): TileN=64 paths were broken before
    // C9 (cudaErrorMisalignedAddress — wrong PermMmaTileN size-128 spilled
    // R2S writes past SmemLayoutD). Fixed in utils.cuh by shrinking
    // PermMmaTileN<64> to a size-64 layout. Sweep (C9b,
    // `tests/baselines/mxfp8_tile_sweep_2026-05-10_C9b.txt`) shows TileN=64
    // wins decisively for tiles_n ≤ 2 (N ≤ 256) — every M-bucket gets
    // +22%~+54%. TileN=64 also wins by single digits at tiles_n == 4 (N=512)
    // for M ≤ 2048; at M ≥ 4096 N=512 TileN=128 reclaims the lead.
    // tiles_n ≥ 8 is TileN=128 territory across all M.

    // tiles_n <= 2 (N ≤ 256): TileN=64 wins on tn=1 universally and on tn=2
    // for M ≤ 1024; at tn=2 M ≥ 2048 TileN=128 reclaims the lead (interface
    // rebench `narrow_n_bench`):
    //   M=2048 N=256 K=4096:  TileN=64 22.8 vs TileN=128 18.1 µs  → TN128 -21%
    //   M=2048 N=256 K=2048:  TileN=64 12.6 vs TileN=128 10.0 µs  → TN128 -21%
    if (tiles_n == 1)
    {
        // N ≤ 128 — TileN=64 always wins; switch to (64,64,4) at large M.
        if (shape_m >= 4096)
            DISPATCH_TILE_MX(64, 64, 4);
        else
            DISPATCH_TILE_MX(32, 64, 4);
    }
    else if (tiles_n == 2 && shape_m <= 1024)
    {
        // N=256 small M: (32,64,4) wins +27..54% over (32,128,4).
        DISPATCH_TILE_MX(32, 64, 4);
    }
    else if (tiles_n == 2)
    {
        // N=256 M ≥ 2048: TileN=128 wins; pick small TileM since N is narrow.
        DISPATCH_TILE_MX(32, 128, 4);
    }
    else if (shape_m <= 32)
    {
        // E22 (2026-05-11): at narrow-N (tn ∈ [3, 32)) K ≥ 9728 M ∈ [16, 32],
        // (32, 64, 4) + Stream-K k=4 beats (32, 128, 4) + k=4 by 6-13%.
        // This path is only reachable for K ≥ 9728 since K < 9728 narrow-N
        // is caught by small_m_narrow_n first. End-to-end cudagraph µs
        // (down N=2560 K=9728):
        //   M=16: (32,128,4) k=4 = 14.34 → (32,64,4) k=4 = 13.48  -6%
        //   M=32: (32,128,4) k=4 = 14.34 → (32,64,4) k=4 = 12.46  -13%
        // M ≤ 8 the trend reverses ((32,128,4) k=4 wins by 1-3 µs) —
        // each (32,64,4) k=4 side stream gets less work per slice and
        // the launch+reduce overhead dominates. Crossover at M ≈ 16.
        //
        // Issue #12 F5B (2026-11-06): the mid-N Stream-K splits of this branch (32 <= tiles_n < 96, where the K-split
        // chooser returns 2 or 4 at M <= 32) are replaced by single launches of (16, 128, 4) (see E26's note above for
        // why a single launch now wins and what it changes in the bytes). Measured (as for E26): wqkv 6144 x 2560 at
        // M = 32, (32, 128, 4) Stream-K 2 16.13 us against (16, 128, 4) 12.80 us (PDL off 16.09 / 13.26); gate 9728 x
        // 2560 at M = 32, 22.61 against 18.99 us (PDL off 22.69 / 19.20); in the calls of fso.dense.linear the
        // pre-quantized call 16.22 -> 12.91 us and the bf16-x call 17.72 -> 15.18 us at wqkv M = 32. Narrow N (reached
        // only with K >= 9728) keeps the (32, 64, 4) Stream-K split: the single launch of (32, 64, 6) was faster on the
        // pre-quantized call (down 2560 x 9728 at M = 32: 21.19 -> 20.73 us) but slower on the bf16-x call, where the
        // activation quantize runs right before it (22.52 -> 22.97 us). A wide N (tiles_n >= 96) never split and keeps
        // (32, 128, 4).
        if (tiles_n >= 3 && tiles_n < 32 && shape_m >= 16)
            DISPATCH_TILE_MX(32, 64, 4);
        else if (tiles_n < 96)
            return mx_route(16, 128, 4);
        else
            DISPATCH_TILE_MX(32, 128, 4);
    }
    else if (shape_m <= 64)
    {
        if (tiles_n >= 96)
            DISPATCH_TILE_MX(64, 128, 2);
        else if (tiles_n >= 32)
            // E21 (2026-05-11): at M ∈ (32, 64] mid-N (tn∈[32,96)),
            // (64, 64, 4) wins over (32, 128, 4). At M=64 both tiles do
            // the same total work (output 32×128 = 64×64 = 4096 elems
            // per CTA, same K-loop depth) but (64, 64, 4) hits 1 wave
            // exactly on 170 SMs at gate (tn=76 → 1×152=152 CTAs ≈ 0.9
            // wave) instead of 0.9-wave-times-2 with (32, 128, 4)'s
            // 2×76=152 CTAs (same CTA count but split across M-direction
            // doesn't benefit at low M). Verified cudagraph µs:
            //   gate M=64 K=2560 tn=76: (32,128,4)=14.43 vs (64,64,4)=12.43 (-14%)
            //   wqkv M=64 K=2560 tn=48: (32,128,4)=12.65 vs (64,64,4)=12.36 (-2.3%)
            // Reproduced across 3 GPU 1 runs.
            DISPATCH_TILE_MX(64, 64, 4);
        else
            // E23 (2026-05-11): at M ∈ (32, 64] narrow-N (tn < 32),
            // K ≥ 9728 (this path is K<9728-unreachable due to
            // small_m_narrow_n override above), (64, 64, 4) + Stream-K
            // k=4 also wins decisively over (32, 128, 4) + k=4:
            //   down M=64 K=9728: (32,128,4) k=4 = 16.95 → (64,64,4) k=4 = 15.46 (-9%)
            //   M=48 same trend (~-7%); M=96 different cascade branch.
            // Same kernel-throughput story as E21 mid-N: at M=64 the
            // (64, 64, 4) tile fits 1 wave per Stream-K slice, while
            // (32, 128, 4) wastes 31/32 of the M dim per CTA at slice
            // boundaries.
            // Issue #12 F5B (2026-11-06): a single launch of the 6-stage (32, 64) instance instead (see E26's note
            // above): down 2560 x 9728 at M = 64, (64, 64, 4) Stream-K 22.08 us against 20.88 us (PDL off 22.50 /
            // 21.41).
            return mx_route(32, 64, 6);
    }
    else if (shape_m <= 96)
    {
        if (tiles_n >= 96)
            DISPATCH_TILE_MX(96, 128, 2);
        // E33 (2026-06): M∈(64,96] mid-N (tn∈[64,96)) — off-grid band the bench
        // (M=64,128) never exercised. The (32,128,4) fallthrough gives 3 M-tiles
        // of thin work; (64,128,2) is +30% (gate M=80/96 N=9728: 28.8→20.1 µs).
        // tn<64 (e.g. wqkv tn=48) keeps (32,128,4) — there (64,128,2) regresses.
        else if (tiles_n >= 64)
            DISPATCH_TILE_MX(64, 128, 2);
        else if (k_split >= 4)
            DISPATCH_TILE_MX(64, 128, 2);
        else
            DISPATCH_TILE_MX(32, 128, 4);
    }
    else if (shape_m <= 128)
    {
        if (tiles_n >= 96)
            DISPATCH_TILE_MX(128, 128, 2);
        else if (tiles_n >= 48)
            DISPATCH_TILE_MX(64, 128, 2);
        else if (k_split >= 4)
            DISPATCH_TILE_MX(64, 128, 2);
        else
            DISPATCH_TILE_MX(32, 128, 4);
    }
    else
    {
        // M > 128. C8 sweep (sm120_mxfp8_tile_sweep_bench, 170-SM Blackwell):
        //
        //   M=1024 N=2560  K=4096  (tn=20): (128)=653  (160)=592  → (128)
        //   M=1024 N=4096  K=4096  (tn=32): (64)=452 (96)=483 (128)=563 (160)=497 → (128)
        //   M=1024 N=6144  K=2560  (tn=48): (64)=461 (96)=511 (128)=556 (160)=640 → (160)
        //   M=1024 N=8192  K=8192  (tn=64): (64)=513 (128)=585 (160)=650        → (160)
        //   M=1024 N=19456 K=2560  (tn=152): (128)=644 (160)=613               → (128) by 5%
        //   M=2048 N=4096  K=4096  (tn=32): (128)=592 (160)=639                → (160)
        //   M=4096 N=4096  K=4096  (tn=32): (128)=641 (160)=720                → (160)
        //   M=4096 N=8192  K=8192  (tn=64): (128)=593 (160)=661                → (160)
        //   M=4096 N=2048  K=2048  (tn=16): (128)=557 (160)=595                → (160)
        //   M=2048 N=1024  K=2048  (tn=8):  (128)=518 (160)=445                → (128)
        //   M=512  N=4096  K=4096  (tn=32):           (128)=555                → (128)
        //
        // Pattern: M ≥ 1024 && tn ∈ [48, 96) → (160); M ≥ 1024 && tn ≤ 32 →
        // (128); M ≥ 2048 → (160) regardless. (224,128,2) consistently lost
        // to (160,128,2) by 5–15% (smem 95KB caps occupancy at 1/SM and
        // bigger M-tile doesn't amortize), so it never appears in the
        // cascade.
        //
        // P3 (2026-10-11, run fso_0.2.1_20261010/P3, RTX 5090): the Stream-K band above M = 128. k_split > 1
        // here means the chooser found a narrow-N, long-K shape (tn < 32, K >= 9728) whose (128,128) tiles
        // under-fill the machine: the Family A `down` (N=2560, K=9728) at M in (128, 1024). The chooser's
        // split (K / 2048, capped at 4) and the tiles the rules below would pick -- (64,128,2) at M < 512,
        // E30's (128,64,2) at M >= 512 -- ran the K slices over more rounds than the machine needs, and those
        // smaller tiles run at about two thirds of the (128,128,2) tile's per-SM rate on this shape. This
        // branch keeps the (128,128,2) tile and picks the split k in [1, min(4, K / 2048)] that minimises
        // the critical path in full-K tile times, ceil(tiles / floor(SMs / k)) / k (the launcher gives each
        // slice floor(SMs / k) CTAs), the smaller k on a tie because every slice adds a partial and the
        // reduce. Per GEMM, cold weights, graph median (µs, cascade before -> this rule):
        //   M=256 (64,128,2)k4 36.5 -> k4 29.0   M=384 (64,128,2)k4 52.3 -> k2 46.8
        //   M=512 (128,64,2)k4 68.3 -> k2 50.9   M=640 (128,128,2)k4 84.6 -> k3 70.1
        //   M=768 (128,128,2)k4 77.9 -> k4 77.7
        // and the MLP block (bench_qwen3_4b_mlp_forward.py, 3 alternating passes) M=256 97.2 -> 88.1, M=512
        // 186.7 -> 166.8, every other M within 0.1 %. M >= 1024 never reaches this branch (the chooser's 90 %
        // saturation gate returns 1), and no other dense shape of the tables does (K >= 9728 with tn < 32).
        if (k_split > 1)
        {
            int const sms = std::max(1, num_device_sms);
            int const tiles = ((static_cast<int>(shape_m) + 127) / 128) * tiles_n;
            int const max_k = std::min(4, std::max(1, static_cast<int>(shape_k) / 2048));
            int best_k = 1;
            int best_rounds = (tiles + sms - 1) / sms;
            for (int k = 2; k <= max_k; ++k)
            {
                int const slice_ctas = std::max(1, sms / k);
                int const rounds = (tiles + slice_ctas - 1) / slice_ctas;
                if (rounds * best_k < best_rounds * k) // rounds / k < best_rounds / best_k
                {
                    best_k = k;
                    best_rounds = rounds;
                }
            }
            if (best_k <= 1)
                return mx_route(128, 128, 2);
            else
                return mx_route(128, 128, 2, 1, 16, best_k);
        }
        if (tiles_n >= 96)
        {
            // E29 (2026-05-28): TileM=160 only when M ≥ 2048 — at smaller M
            // the second/last M-tile is under-utilised (M=256 → 60% util on
            // 2nd tile; M=1024 → 40% on 7th). (128,128,2) gives clean
            // tiles_m at these M with better wave fill. Sweep:
            //   M=256  N=19456 K=2560 (tn=152): (160)=523 → (128)=597 +14%
            //   M=1024 N=9728  K=2560 (tn=76):  (160)=529 → (128)=620 +17%
            if (shape_m >= 2048)
                DISPATCH_TILE_MX(160, 128, 2);
            else
            {
                // E33 (2026-06): M∈(128,2048) wide-N — the bench grid jumps
                // 128→512, so off-grid M whose TileM doesn't divide M cleanly
                // wasted up to half the last M-tile on the fixed (128,128,2)
                // pick. Choose TileM ∈ {96,128,160} minimising M-tile waste
                // (tie → larger TileM). Clean multiples of 128 (256/384/512/
                // 1024) are unchanged → (128); the win is on the in-between M:
                // M=160 +35%, M=192/176 +16%, M=288/320 +18% (gate_up N=19456).
                int const w96 = ((int(shape_m) + 95) / 96) * 96 - int(shape_m);
                int const w128 = ((int(shape_m) + 127) / 128) * 128 - int(shape_m);
                int const w160 = ((int(shape_m) + 159) / 160) * 160 - int(shape_m);
                if (w160 <= w128 && w160 <= w96)
                    DISPATCH_TILE_MX(160, 128, 2);
                else if (w128 <= w96)
                    DISPATCH_TILE_MX(128, 128, 2);
                else
                    DISPATCH_TILE_MX(96, 128, 2);
            }
        }
        else if (shape_m == 2048 && tiles_n <= 20)
            // E6 (2026-05-10): M=2048 narrow-N (tn ∈ {16, 20}): (128) wins
            // +9~17% over (160). tn=24 falls through to (160) (boundary
            // non-monotonic — at tn=24 (160) reclaims +18~23%).
            //   M=2048 N=2048 K=2048..8192 (tn=16): (128)=521..594 (160)=455..505 +14~17%
            //   M=2048 N=2560 K=2048..9728 (tn=20): (128)=617..672 (160)=556..619 +9~11%
            //   M=2048 N=3072 K=4096..9728 (tn=24): (128)=567..581 (160)=685..699 -17~19% [keep (160)]
            DISPATCH_TILE_MX(128, 128, 2);
        else if (shape_m >= 4096 && tiles_n == 20)
            // E6: down M=4096 N=2560 (tn=20): (128) wins +12~17% — narrowly
            // tn=20 only. tn=16 keeps (160) at M=4096 (different from M=2048).
            //   M=4096 N=2560 K=4096: (128)=672 (160)=602 +11.6%
            //   M=4096 N=2560 K=9728: (128)=685 (160)=587 +16.7%
            DISPATCH_TILE_MX(128, 128, 2);
        else if (shape_m >= 2048)
            DISPATCH_TILE_MX(160, 128, 2);
        else if (shape_m >= 1024 && tiles_n >= 48)
        {
            // M=1024 with tn ∈ [48, 96): wide-grid favors (160) at the
            // narrow end of that range, (128) at the wide end. Same
            // M=1024 last-M-tile waste either way (TM=160 → 40%
            // util, TM=128 → 100% util) but at tn ∈ [48, 64) the
            // per-CTA arithmetic intensity gain from a wider TileM
            // outweighs the M-waste; at tn ≥ 64 the higher tiles_n
            // already saturates and (128)'s clean tiles_m=8 wins.
            // E29 (2026-05-28) sweep:
            //   qwen-wqkv M=1024 N=6144  K=2560 (tn=48): (160)=640 (128)=521 → (160)
            //   qwen-gate M=1024 N=9728  K=2560 (tn=76): (160)=529 (128)=620 → (128) +17%
            if (tiles_n < 64)
                DISPATCH_TILE_MX(160, 128, 2);
            else
                DISPATCH_TILE_MX(128, 128, 2);
        }
        else if (shape_m >= 512 && tiles_n <= 32)
        {
            // E30 (2026-05-28): cubic-1024 wave-fill rescue. (128,128,2) at
            // M=N=K=1024 gives tiles_m × tiles_n = 8 × 8 = 64 CTAs on
            // 170 SMs = 0.38 wave (huge underfill). (128,64,2) doubles
            // tiles_n to 16 → 128 CTAs → 0.75 wave; same TileM=128
            // keeps M-bandwidth per CTA. Sweep: 193→236 TF (+22%, beats
            // cuBLAS 194 TF). Gate on tiles_m_128 * tiles_n < sm_count/2
            // so it only fires when wave fill is genuinely poor — at
            // M=2048+ or wider tiles_n the (128,128,2) tile already fills.
            int const tiles_m_128 = (static_cast<int>(shape_m) + 127) / 128;
            if (tiles_m_128 * tiles_n * 2 < num_device_sms)
                DISPATCH_TILE_MX(128, 64, 2);
            else
                DISPATCH_TILE_MX(128, 128, 2);
        }
        else if (shape_m >= 1024)
            // M ≥ 1024 with tn ∈ (32, 48): boundary band — (128) over (64).
            DISPATCH_TILE_MX(128, 128, 2);
        else
        {
            // M ∈ (128, 512). Per-shape sweep (E31, 2026-05-28):
            //   M=256 N=2560  K=9728 (tn=20, narrow-N + long-K):
            //     (64)=369 (96)=317 (128)=n/a   → keep (64,128,2)
            //   M=256 N=6144  K=2560 (tn=48):
            //     (64)=288 (96)=415 (128)=376   → (96,128,2) +44%
            //   M=256 N=9728  K=2560 (tn=76):
            //     (64)=422 (128)=562            → (128,128,2) +33%
            // The boundaries: tn < 32 is narrow-N (wave already poor at
            // M=256; the smaller (64,128,2) tile doubles tiles_m and
            // wins); tn ∈ [32, 48] wants (96) for cleanest tiles_m at
            // M=256; tn > 48 wants (128) to saturate.
            if (tiles_n < 32)
                DISPATCH_TILE_MX(64, 128, 2);   // narrow-N fallback (E31 guard)
            else if (tiles_n <= 48)
                DISPATCH_TILE_MX(96, 128, 2);
            else
                DISPATCH_TILE_MX(128, 128, 2);
        }
    }
#undef DISPATCH_TILE_MX
    // Every branch above returns (the compiler proves the end unreachable).
}

// The wave-fill rule over the cascade's pick (issue #12 F5A, 2026-11-05): above M = 128, when the pick above is a single
// launch whose grid has fewer CTAs than the device has SMs, the route takes the existing cascade tile with the smallest
// waves x (TileM + TileN), waves = ceil(CTAs / SMs), if that is strictly smaller than the pick's own (ties: see the
// tie-break of issue #12 F5B in the function below). A tile of 32 rows is a candidate only when the
// pick's last M-tile is partial (M is not a multiple of its TileM). FSO_DISABLE_OVERRIDES=1 and the forced diagnostic
// configurations (FSO_FORCE_TILE / FSO_FORCE_KSPLIT, ...) bypass it, and a Stream-K pick is left alone: the rule only
// ever exchanges one single-launch tile for another.
//
// Why: on the short-K and narrow-N shapes of this band one CTA's main loop takes a time close to proportional to the
// operand bytes it streams per K step, (TileM + TileN) x 128, so the grid's time is about its waves times that. The
// cascade's rules above were fitted on the dense families' shapes, and on 1280-wide to 2560-wide outputs at M from 129
// to about 1100 their picks ran 60-100 CTAs on the RTX 5090's 170 SMs. Measured there (2407 MHz, cold weights,
// CUDA-graph replay, each tile forced through fso.dense.linear, the issue #12 tower roles with their epilogues and the
// dense families' shapes, two passes), the rule's tile was faster than the pick on both the 2b and the bf16-x call on
// each of the 42 cells it moves, by 2-38 %: 1280 x 1280 at M = 1079 from 18.5 to 12.1 us, 1280 x 5120 at M = 364 from
// 28.7 to 21.7 us, 1280 x 7680 at M = 364 from 41.0 to 28.9 us. A first form without the 32-row condition also moved
// 1024 x 2048 at M = 192 and 256 from 64 to 32 rows, where the pick's M-tiles were all full and both grids fit one wave,
// and measured 0.995 and 1.011 of the pick on the 2b call: splitting the rows finer then shortens no CTA's main loop,
// and the 32-row tiles won only where the pick pads its last M-tile. A pick whose grid already fills the machine is
// left alone, so the multi-wave tiles of the dense families' large-M cells keep their routes.
//
// Bytes: every tile has identity, epilogue and quantized-output instantiations, and the K accumulation order of a
// single launch does not depend on the tile, so the route gives the bytes of the pick, on both entries and for
// linear_quantized.
inline Sm120DenseRoute sm120_mxfp8_dense_route(
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_device_sms)
{
    Sm120DenseRoute r = sm120_mxfp8_dense_route_cascade(shape_m, shape_n, shape_k, num_device_sms);
    if (r.forced_only_instance() || read_disable_overrides() || force_tile_applies(read_force_tile(), shape_k))
        return r;
    int const sms = std::max(1, num_device_sms);
    auto ctas = [&](int tm, int tn)
    { return static_cast<int>((shape_m + tm - 1) / tm) * static_cast<int>((shape_n + tn - 1) / tn); };
    // 1. The wave-fill rule (F5A), with the tie-break of issue #12 F5B: among tiles of equal waves x (TileM + TileN),
    // the one with the smaller 2 x TileM + TileN wins, the pick included (before F5B a tie kept the pick). 2 x TileM +
    // TileN is in proportion to the shared-memory ldmatrix bytes a k-tile costs in the 2 x 4 math-warp layout of these
    // tiles (the A rows are read by the 4 N-warps, the B rows by the 2 M-warps), so of two tiles that stream the same
    // operand bytes per CTA in the same number of waves it takes the one that reads less shared memory: (64, 128)
    // over (128, 64). Measured on the RTX 5090 (2407 MHz, cold weights, graph replay, the same process): wo 2560 x
    // 4096 at M = 512, 160 CTAs either way, (64, 128, 2) 26.29 against the pick (128, 64, 2) 27.50 us. It moves none of
    // the 42 cells F5A's rule moves; it moves 9 cells of (128, 64, 2) picks at M = 512-1079 to (64, 128).
    if (shape_m > 128 && r.k_split <= 1 && ctas(r.tile_m, r.tile_n) < sms)
    {
        bool const pick_partial = shape_m % static_cast<uint32_t>(r.tile_m) != 0;
        auto cost = [&](int tm, int tn) { return (ctas(tm, tn) + sms - 1) / sms * (tm + tn); };
        auto smem_reads = [](int tm, int tn) { return 2 * tm + tn; };
        constexpr int kTiles[8][3] = {
            {32, 64, 4}, {32, 128, 4}, {64, 64, 4}, {64, 128, 2}, {96, 128, 2}, {128, 64, 2}, {128, 128, 2}, {160, 128, 2}};
        int best = cost(r.tile_m, r.tile_n);
        int best_reads = smem_reads(r.tile_m, r.tile_n);
        for (auto const& t : kTiles)
        {
            if (t[0] == 32 && !pick_partial)
                continue;
            int const c = cost(t[0], t[1]);
            int const rd = smem_reads(t[0], t[1]);
            if (c < best || (c == best && rd < best_reads))
            {
                best = c;
                best_reads = rd;
                r.tile_m = t[0];
                r.tile_n = t[1];
                r.stages = t[2];
            }
        }
    }
    // 2. The stage rule (issue #12 F5B): a single-wave launch (no more tiles than SMs) of (64, 128) or (96, 128) runs
    // the 3-stage instance (the ring mainloop of fp8/gemm_1d1d.cuh) instead of the 2-stage one when its tile has 3 or
    // more M-tiles, or 2 M-tiles and no more tiles than 3/4 of the SMs. Why: with several M-tiles every weight panel is
    // read by several CTAs, mostly from L2, so each CTA's main loop is bound by the latency of its own loads, and the
    // third stage hides part of it. With 2 M-tiles on most of the SMs the grid already streams the weights at close
    // to the DRAM rate, and in a multi-wave launch a CTA's next tile overlaps the previous one's tail; in both cases
    // the third stage gained nothing and cost 1-2 %. Measured on the RTX 5090 (2407 MHz, cold weights, graph replay,
    // 2 and 3 stages interleaved in one process, both on the cluster launch above, PDL on): wqkv 6144 x 2560 at M =
    // 128 on (64, 128), 96 tiles, 16.01 -> 15.14 us; at M = 256 on (96, 128) 20.66 -> 19.70 us; 2048 x 2048 at M = 416
    // 13.97 -> 13.14 us; 1280 x 7680 at M = 1079, 170 tiles, 47.43 -> 46.18 us. The cells the conditions exclude:
    // 9216 x 2048 at M = 128 (2 M-tiles, 144 tiles) 15.34 -> 15.68 us, 12288 x 2048 at M = 192 (192 tiles) 33.44 ->
    // 33.97 us, 5120 x 1280 at M = 533 (240 tiles) 22.55 -> 22.81 us. The MMAs and their order do not depend on the
    // stage count, so the bytes are those of the 2-stage instance. (128, 64) has no 3-stage instance: after the
    // tie-break above no single-wave route keeps a (128, 64) tile with more than one M-tile on the measured shapes.
    if (r.k_split <= 1 && r.stages == 2 && (r.tile_m == 64 || r.tile_m == 96) && r.tile_n == 128)
    {
        int const m_tiles = static_cast<int>((shape_m + r.tile_m - 1) / r.tile_m);
        int const tiles = ctas(r.tile_m, r.tile_n);
        if (tiles <= sms && (m_tiles >= 3 || (m_tiles == 2 && 4 * tiles <= 3 * sms)))
            r.stages = 3;
    }
    // 3. The small-tile stage rule (issue #12 F5B): a single launch of (32, 64, 4) runs the 6-stage instance (ring
    // mainloop) when the k-tile count K / 128 is not a multiple of 4, or when M <= 64, K >= 2048 and the grid has at
    // least 40 tiles. Why: the 4-stage instance runs the original loop, which pads the k-tiles to a multiple of its 4
    // stages, so at K = 1280 it streams 12 k-tiles where the ring streams the 10 real ones (the padded k-tiles hold
    // zeros, so the sums are the same); and at M <= 64 the grid's at most 2 M-tiles of 32 rows leave each CTA's main
    // loop bound by its own load latency, which 2 more stages hide once the loop is long enough to use them. On fewer
    // than 16 k-tiles, on more M-tiles (more CTAs share each weight panel) and on the smallest grids (1024 x 2048 at
    // M = 64, 32 tiles) 6 stages were 1-4 % slower. Measured on the RTX 5090 (2407 MHz, cold weights, graph replay, 4
    // and 6 stages interleaved in one process, both on the cluster launch above, PDL on / off): 1280 x 1280 at M = 117
    // 4.90 -> 4.54 / 5.43 -> 5.03 us; 2048 x 1280 at M = 117 6.03 -> 5.66 / 6.13 -> 5.88 us; 2048 x 4096 at M = 64
    // 9.51 -> 9.32 / 10.07 -> 9.64 us; 2560 x 2048 at M = 48 6.52 -> 6.33 / 7.03 -> 6.85 us. The MMAs and their order do
    // not depend on the stage count, so the bytes are those of the 4-stage instance.
    if (r.k_split <= 1 && r.tile_m == 32 && r.tile_n == 64 && r.stages == 4)
    {
        uint32_t const k_tiles = (shape_k + 127) / 128;
        if (k_tiles % 4 != 0 || (shape_m <= 64 && k_tiles >= 16 && ctas(32, 64) >= 40))
            r.stages = 6;
    }
    return r;
}

// A non-identity epilogue on a route whose instance exists only behind a diagnostic knob: refused, before anything
// is launched or allocated. Those instances (the FSO_FORCE_SMALLM kernel, the FSO_FORCE_MIN_BLOCKS=2 and
// FSO_FORCE_SCHED_GROUP={8,32} variants of the TileM = 16 tiles) have no dense-epilogue instantiation.
inline void sm120_mxfp8_dense_require_epilogue(Sm120DenseRoute const& route, fso::DenseEpilogue const& epilogue)
{
    if (epilogue.is_identity() || !route.forced_only_instance())
        return;
    char const* what = route.smallm ? "the SmallM kernel SM120MxFP8SmallMKernel (FSO_FORCE_SMALLM=1)"
        : route.min_blocks != 1     ? "the MinBlocksPerSm=2 variant (FSO_FORCE_MIN_BLOCKS=2)"
                                    : "the scheduler-group variant (FSO_FORCE_SCHED_GROUP)";
    throw fso::DenseEpilogueNotImplemented(std::string("sm_120/sm_121 dense MXFP8 GEMM (gemm_dispatch_sm120_mxfp8): the "
        "forced diagnostic instance ") + what + " of tile (" + std::to_string(route.tile_m) + ", "
        + std::to_string(route.tile_n) + ", " + std::to_string(route.stages) + ") has no dense-epilogue "
        "instantiation, so the dense epilogue (" + fso::dense_epilogue_field_names(epilogue.fields())
        + ") is refused; unset the knob, or call without the epilogue");
}

// The dense MXFP8 GEMM on sm_120/sm_121: the route of sm120_mxfp8_dense_route, launched.
//
// `epilogue` (common/epilogue.cuh, interface v1): bias, residual and the erf GELU are applied on every route of the
// cascade -- by the dense-epilogue instance of the tile on a single launch, by the Stream-K reduce after the
// reduction on a split one -- with one rounding. A forced diagnostic instance refuses a non-identity epilogue
// (sm120_mxfp8_dense_require_epilogue), as does any field outside kDenseEpiServedSm120Mxfp8Cascade.
// With epilogue.out_q (fso.dense.linear_quantized; bias and act only) every route writes the quantized form instead
// of the bf16 D: the quantized-output instance of the tile on a single launch, the quantized reduce on Stream-K; the
// caller passes mat_d = out_q (see launch_sm120_mxfp8_gemm_kernel's DenseQuant).
//
// `streamk_scratch` / `streamk_scratch_bytes`: the caller's bf16 partial scratch for a Stream-K route
// (route.streamk_scratch_bytes(M, N) bytes; the op body allocates it from the torch caching allocator on `stream`).
// A single-launch route reads neither; a Stream-K route without enough scratch is an error.
inline void gemm_dispatch_sm120_mxfp8(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
    cudaStream_t stream, int num_device_sms = kNumDeviceSMs, fso::DenseEpilogue const& epilogue = fso::DenseEpilogue{},
    void* streamk_scratch = nullptr, std::size_t streamk_scratch_bytes = 0)
{
    fso::require_dense_epilogue(epilogue, fso::kDenseEpiServedSm120Mxfp8Cascade,
        "sm_120/sm_121 dense MXFP8 GEMM (gemm_dispatch_sm120_mxfp8, the CUTLASS cascade)");
    if (num_device_sms < 0)
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    Sm120DenseRoute const route = sm120_mxfp8_dense_route(shape_m, shape_n, shape_k, num_device_sms);
    sm120_mxfp8_dense_require_epilogue(route, epilogue);
    sm120_mxfp8_dense_require_quant_out(epilogue, shape_n);

    int64_t ld_a = shape_k;
    int64_t ld_b = shape_k;
    int64_t ld_d = shape_n;
    constexpr int64_t stride = 0;
    constexpr uint32_t num_problems = 1;
    auto is = [&](int tm, int tn, int st) { return route.tile_m == tm && route.tile_n == tn && route.stages == st; };

    if (route.smallm)
    {
        launch_sm120_mxfp8_smallm_gemm_kernel<128, 4>(mat_a, ld_a, stride, mat_b, ld_b, stride, mat_d,
            ld_d, stride, scales_a, stride, scales_b, stride, num_problems, shape_m, shape_n, shape_k,
            stream, num_device_sms);
        return;
    }
    if (route.k_split > 1)
    {
#define SK_MX(TM_, TN_, ST_)                                                                                           \
    if (is(TM_, TN_, ST_))                                                                                             \
    {                                                                                                                  \
        launch_sm120_mxfp8_streamk_gemm<TM_, TN_, ST_>(mat_a, static_cast<int>(ld_a), mat_b, static_cast<int>(ld_b),    \
            mat_d, static_cast<int>(ld_d), static_cast<int>(shape_m), static_cast<int>(shape_n),                        \
            static_cast<int>(shape_k), scales_a, scales_b, route.k_split, num_device_sms, stream, streamk_scratch,     \
            streamk_scratch_bytes, epilogue);                                                                          \
        return;                                                                                                        \
    }
        SK_MX(16, 128, 4) SK_MX(32, 128, 4) SK_MX(32, 64, 4) SK_MX(64, 128, 2) SK_MX(64, 64, 4) SK_MX(128, 128, 2)
        SK_MX(128, 64, 2)
#undef SK_MX
        TLLM_THROW("gemm_dispatch_sm120_mxfp8: no Stream-K instance (%d, %d, %d)", route.tile_m, route.tile_n,
            route.stages);
    }
    if (epilogue.out_q != nullptr)
    {
        bool const launched = launch_sm120_mxfp8_dense_quant_single(route.tile_m, route.tile_n, route.stages, mat_a,
            mat_b, scales_a, scales_b, shape_m, shape_n, shape_k, stream, num_device_sms, epilogue);
        TLLM_CHECK_WITH_INFO(launched, "gemm_dispatch_sm120_mxfp8: no quantized-output instance (%d, %d, %d)",
            route.tile_m, route.tile_n, route.stages);
        return;
    }
    if (!epilogue.is_identity())
    {
        // Issue #12 F5B: a dense epilogue that adds a residual runs the 2-stage instance of a 3-stage route (the stage
        // rule of sm120_mxfp8_dense_route; the route query reports the identity call's route) when the main loop has
        // fewer than 16 k-tiles. Measured on the RTX 5090 (2407 MHz, cold weights, graph replay, each build in its own
        // fresh process): tower out_proj 1280 x 1280 at M = 1079 with bias + residual, 10 k-tiles, took 13.52 us with 3
        // stages and 13.13 us with 2 (fp8-round 13.14), while the same shape with the bias alone took 11.78 us with 3
        // stages against 12.12 (fp8-round), and fc2 1280 x 5120 with bias + residual (40 k-tiles) gained 0.9 % with 3.
        // The forced diagnostic configurations keep the stage count they name. The bytes are the same either way.
        int stages = route.stages;
        if (epilogue.residual != nullptr && stages == 3 && (shape_k + 127) / 128 < 16
            && !force_tile_applies(read_force_tile(), shape_k))
            stages = 2;
        bool const launched = launch_sm120_mxfp8_dense_epi_single(route.tile_m, route.tile_n, stages, mat_a,
            mat_b, mat_d, scales_a, scales_b, shape_m, shape_n, shape_k, stream, num_device_sms, epilogue);
        TLLM_CHECK_WITH_INFO(launched, "gemm_dispatch_sm120_mxfp8: no dense-epilogue instance (%d, %d, %d)",
            route.tile_m, route.tile_n, stages);
        return;
    }
#define ONE_MX(TM_, TN_, ST_, MB_, SG_)                                                                                \
    if (is(TM_, TN_, ST_) && route.min_blocks == MB_ && route.sched_group == SG_)                                      \
    {                                                                                                                  \
        launch_sm120_mxfp8_gemm_kernel<TM_, TN_, ST_, MB_, SG_>(mat_a, ld_a, stride, mat_b, ld_b, stride, mat_d, ld_d,  \
            stride, scales_a, stride, scales_b, stride, num_problems, shape_m, shape_n, shape_k, stream,               \
            num_device_sms);                                                                                           \
        return;                                                                                                        \
    }
    ONE_MX(16, 128, 4, 1, 16) ONE_MX(16, 128, 4, 2, 16) ONE_MX(16, 128, 4, 1, 8) ONE_MX(16, 128, 4, 1, 32)
    ONE_MX(16, 64, 4, 1, 16) ONE_MX(16, 64, 4, 1, 8) ONE_MX(16, 64, 4, 1, 32)
    ONE_MX(32, 128, 4, 1, 16) ONE_MX(32, 64, 4, 1, 16) ONE_MX(64, 128, 2, 1, 16) ONE_MX(64, 64, 4, 1, 16)
    ONE_MX(96, 128, 2, 1, 16) ONE_MX(128, 128, 2, 1, 16) ONE_MX(128, 64, 2, 1, 16) ONE_MX(160, 128, 2, 1, 16)
    // The 3-stage instances of the stage rule and the 6-stage decode instances (issue #12 F5B; ring mainloop).
    ONE_MX(64, 128, 3, 1, 16) ONE_MX(96, 128, 3, 1, 16)
    ONE_MX(16, 64, 6, 1, 16) ONE_MX(32, 64, 6, 1, 16)
#undef ONE_MX
    TLLM_THROW("gemm_dispatch_sm120_mxfp8: no instance (%d, %d, %d, min_blocks %d, sched_group %d)", route.tile_m,
        route.tile_n, route.stages, route.min_blocks, route.sched_group);
}
// ---------------------------------------------------------------------------
// Grouped (MoE) MXFP8 dispatch — masked layout, per-expert weights.
//
// Revives the dormant grouped path of SM120BlockScaledKernel: the persistent
// SM120BlockScaledScheduler already walks (group, m_block, n_block) work items
// when `grouped_layout != nullptr` (per-group valid-row counts, DeepGEMM
// masked semantics), and every TMA descriptor carries the group as its L
// (batch) dimension. This function only supplies the grouped strides:
//
//   A   [G, m_cap, K]  fp8   — per-group activation slab, rows beyond
//                              grouped_layout[g] are never scheduled
//   B   [G, N, K]      fp8   — per-expert weights
//   SFA [G, K/128, pad(m_cap,4)]  int32 packed UE8M0, K-major per group
//   SFB [G, K/128, pad(N,4)]      int32 packed UE8M0, K-major per group
//   D   [G, m_cap, N]  bf16  — rows beyond grouped_layout[g] are garbage
//
// `expected_m` is a host-side static hint (ceil(total_rows / G)) used only
// for tile selection — never read from device memory, so the launch is
// CUDA-Graph capture-safe and replays follow whatever masked counts the
// grouped_layout buffer holds at replay time.
//
// v0 cascade: trimmed from the dense (M, tiles_n) cascade with expected_m
// standing in for M. No Stream-K (per-group tile parallelism G×tiles_n is
// already ≥ grid size for the MoE shapes this serves); no smallm variant.
// Retune against a FORCE_TILE sweep on 5090 before freezing (M1 gate).
// With FusedSwiGLU the same cascade launches the fused-epilogue instantiations:
// the kernel writes MXFP8 [G, m_cap, N/2] rows and their K-major scale words
// instead of a bf16 [G, m_cap, N] slab, so the layer's separate SwiGLU kernel and
// that slab both disappear. The tile rule is deliberately the same one -- the FC1
// shape has not changed, only its store -- and `mat_d` is then the FP8 slab's
// base (see the launcher).
//
// `max_active_groups` is the second host-side static hint: an upper bound on
// how many groups can hold a row, min(M * topk, G), which the op already
// receives from its caller. It bounds the decode route's tile count, which
// `expected_m` cannot (at G = 128, top-8, expected_m is 1 for every M up to
// 16). 0 means "not supplied" and keeps the m_cap-only decode gate.
//
// CompactActs (0.2.1, charter D1) runs the same cascade on the compact activation
// layout: A [act_rows, K] fp8, SFA the one-group K-major words [1, K/128, act_rows],
// D [act_rows, N] (or the fused epilogues' one-group FP8 rows and scale words), with
// expert g's rows starting at row blk0[g] * 128. Every tile pick below reads only
// (m_cap, N, K, expected_m, max_active_groups), which the caller passes exactly as
// for the masked layout, so a compact call runs the instance the masked call of the
// same shape runs -- forced instances included.
template <bool FusedSwiGLU = false, bool ScatterCombine = false, bool CompactActs = false>
inline void gemm_dispatch_sm120_mxfp8_grouped(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int32_t* grouped_layout, uint32_t num_groups, uint32_t m_cap,
    uint32_t shape_n, uint32_t shape_k, uint32_t expected_m, cudaStream_t stream,
    int num_device_sms = kNumDeviceSMs, __nv_fp8_e4m3* out_fp8 = nullptr, int32_t* out_sf = nullptr,
    __nv_bfloat16* out_tokens = nullptr, int32_t const* row_map = nullptr, float const* weight_of_slot = nullptr,
    uint32_t max_active_groups = 0, int32_t const* blk0 = nullptr, uint32_t act_rows = 0)
{
    if (num_device_sms < 0)
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

    int64_t const ld_a = shape_k;
    int64_t const ld_b = shape_k;
    int64_t const ld_d = shape_n;
    // The activation tensors' batch stride: one group of act_rows rows (compact) or
    // m_cap rows per group (masked). The compact descriptors have a batch extent of 1,
    // so their stride is never stepped.
    int64_t const act_group_rows = CompactActs ? static_cast<int64_t>(act_rows) : static_cast<int64_t>(m_cap);
    int64_t const stride_a = act_group_rows * shape_k;
    int64_t const stride_b = static_cast<int64_t>(shape_n) * shape_k;
    int64_t const stride_d = act_group_rows * shape_n;

    // Where the fused combine's scatter runs. The default hands it to the store
    // warp and the otherwise-idle fourth TMA warp, so the weighted atomics overlap
    // the next tile's mainloop the way the TMA store does. Measured on the routed
    // layer (run sm120_scatterwarp_20260928): -6.0 % at Family C M = 8192, -6.3 % at
    // M = 4096, -8.1 % at Family B M = 8192 against scattering from the math warps,
    // and nothing at M = 2048 where the route does not engage. One warp is not
    // enough -- that form costs +6 %, because a single warp cannot keep enough
    // atomic requests in flight. FSO_MOE_SCATTER_WARP=0 restores the math-warp form.
    bool const scatter_in_store_warp = [] {
        char const* e = std::getenv("FSO_MOE_SCATTER_WARP");
        return !(e != nullptr && e[0] == '0');
    }();

#define DISPATCH_GROUPED_TILE_MX(TM, TN, ST)                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if constexpr (ScatterCombine)                                                                                  \
        if (scatter_in_store_warp)                                                                                     \
        {                                                                                                              \
            launch_sm120_mxfp8_gemm_kernel<TM, TN, ST, 1, 16, true, true, FusedSwiGLU, ScatterCombine, true, CompactActs>(           \
                mat_a, ld_a, stride_a, mat_b, ld_b, stride_b, mat_d, ld_d, stride_d, scales_a, 0, scales_b, 0,         \
                num_groups, m_cap, shape_n, shape_k, stream, num_device_sms, grouped_layout, out_fp8, out_sf,          \
                out_tokens, row_map, weight_of_slot, max_active_groups, blk0, act_rows);                                               \
            return;                                                                                                    \
        }                                                                                                              \
        launch_sm120_mxfp8_gemm_kernel<TM, TN, ST, 1, 16, true, true, FusedSwiGLU, ScatterCombine, false, CompactActs>(mat_a, ld_a, stride_a, mat_b,    \
            ld_b, stride_b, mat_d, ld_d, stride_d, scales_a, 0, scales_b, 0, num_groups, m_cap, shape_n, shape_k,      \
            stream, num_device_sms, grouped_layout, out_fp8, out_sf, out_tokens, row_map, weight_of_slot, max_active_groups, blk0, act_rows); \
        return;                                                                                                        \
    } while (0)

    // FSO_FORCE_TILE=TM,TN,ST — A/B knob for the grouped tile sweep.
    // Stream-K / smallm / sched-group force flags are ignored on this path.
    auto forced = read_force_tile();
    if (force_tile_applies(forced, shape_k))
    {
        // FSO_FORCE_MIN_BLOCKS=2: co-schedule 2 CTAs/SM (grid doubles via
        // the launcher). ncu verdict on the decode cells: every pipe < 55%
        // SOL at 1 CTA/SM (latency-bound, achieved occupancy ~23%); only
        // the ~44 KB (16,64,x) instances fit 2 CTAs in the 99 KB budget.
        if (forced.min_blocks_2())
        {
            if (forced.tm == 16 && forced.tn == 64 && forced.st == 4)
            {
                launch_sm120_mxfp8_gemm_kernel<16, 64, 4, 2, 16, true, true, FusedSwiGLU, ScatterCombine, false, CompactActs>(mat_a, ld_a, stride_a,
                    mat_b, ld_b, stride_b, mat_d, ld_d, stride_d, scales_a, 0, scales_b, 0, num_groups, m_cap,
                    shape_n, shape_k, stream, num_device_sms, grouped_layout, out_fp8, out_sf, out_tokens,
                    row_map, weight_of_slot, max_active_groups, blk0, act_rows);
                return;
            }
            if (forced.tm == 16 && forced.tn == 64 && forced.st == 2)
            {
                launch_sm120_mxfp8_gemm_kernel<16, 64, 2, 2, 16, true, true, FusedSwiGLU, ScatterCombine, false, CompactActs>(mat_a, ld_a, stride_a,
                    mat_b, ld_b, stride_b, mat_d, ld_d, stride_d, scales_a, 0, scales_b, 0, num_groups, m_cap,
                    shape_n, shape_k, stream, num_device_sms, grouped_layout, out_fp8, out_sf, out_tokens,
                    row_map, weight_of_slot, max_active_groups, blk0, act_rows);
                return;
            }
            if (forced.tm == 16 && forced.tn == 128 && forced.st == 2)
            {
                launch_sm120_mxfp8_gemm_kernel<16, 128, 2, 2, 16, true, true, FusedSwiGLU, ScatterCombine, false, CompactActs>(mat_a, ld_a, stride_a,
                    mat_b, ld_b, stride_b, mat_d, ld_d, stride_d, scales_a, 0, scales_b, 0, num_groups, m_cap,
                    shape_n, shape_k, stream, num_device_sms, grouped_layout, out_fp8, out_sf, out_tokens,
                    row_map, weight_of_slot, max_active_groups, blk0, act_rows);
                return;
            }
            if (forced.tm == 32 && forced.tn == 64 && forced.st == 2)
            {
                launch_sm120_mxfp8_gemm_kernel<32, 64, 2, 2, 16, true, true, FusedSwiGLU, ScatterCombine, false, CompactActs>(mat_a, ld_a, stride_a,
                    mat_b, ld_b, stride_b, mat_d, ld_d, stride_d, scales_a, 0, scales_b, 0, num_groups, m_cap,
                    shape_n, shape_k, stream, num_device_sms, grouped_layout, out_fp8, out_sf, out_tokens,
                    row_map, weight_of_slot, max_active_groups, blk0, act_rows);
                return;
            }
            if (forced.tm == 64 && forced.tn == 64 && forced.st == 2)
            {
                launch_sm120_mxfp8_gemm_kernel<64, 64, 2, 2, 16, true, true, FusedSwiGLU, ScatterCombine, false, CompactActs>(mat_a, ld_a, stride_a,
                    mat_b, ld_b, stride_b, mat_d, ld_d, stride_d, scales_a, 0, scales_b, 0, num_groups, m_cap,
                    shape_n, shape_k, stream, num_device_sms, grouped_layout, out_fp8, out_sf, out_tokens,
                    row_map, weight_of_slot, max_active_groups, blk0, act_rows);
                return;
            }
            std::fprintf(stderr,
                "[fso] FSO_FORCE_MIN_BLOCKS=2 grouped: wired for (16,64,x)/(16,128,2)/(32,64,2)/(64,64,2); ignoring\n");
        }
        if (forced.tm == 16 && forced.tn == 128 && forced.st == 4) DISPATCH_GROUPED_TILE_MX(16, 128, 4);
        if (forced.tm == 16 && forced.tn == 64 && forced.st == 4)  DISPATCH_GROUPED_TILE_MX(16, 64, 4);
        if (forced.tm == 32 && forced.tn == 128 && forced.st == 4) DISPATCH_GROUPED_TILE_MX(32, 128, 4);
        if (forced.tm == 32 && forced.tn == 64 && forced.st == 4)  DISPATCH_GROUPED_TILE_MX(32, 64, 4);
        if (forced.tm == 64 && forced.tn == 64 && forced.st == 4)  DISPATCH_GROUPED_TILE_MX(64, 64, 4);
        if (forced.tm == 64 && forced.tn == 128 && forced.st == 2) DISPATCH_GROUPED_TILE_MX(64, 128, 2);
        if (forced.tm == 32 && forced.tn == 128 && forced.st == 2) DISPATCH_GROUPED_TILE_MX(32, 128, 2);
        if (forced.tm == 32 && forced.tn == 64 && forced.st == 2)  DISPATCH_GROUPED_TILE_MX(32, 64, 2);
        if (forced.tm == 64 && forced.tn == 64 && forced.st == 2)  DISPATCH_GROUPED_TILE_MX(64, 64, 2);
        if (forced.tm == 96 && forced.tn == 64 && forced.st == 2)  DISPATCH_GROUPED_TILE_MX(96, 64, 2);
        if (forced.tm == 160 && forced.tn == 128 && forced.st == 2) DISPATCH_GROUPED_TILE_MX(160, 128, 2);
        if (forced.tm == 96 && forced.tn == 128 && forced.st == 2) DISPATCH_GROUPED_TILE_MX(96, 128, 2);
        if (forced.tm == 128 && forced.tn == 128 && forced.st == 2) DISPATCH_GROUPED_TILE_MX(128, 128, 2);
        std::fprintf(stderr,
            "[fso] FSO_FORCE_TILE=%d,%d,%d not wired on the grouped path; falling through to cascade\n",
            forced.tm, forced.tn, forced.st);
    }

    uint32_t const tiles_n = (shape_n + 127) / 128;
    uint32_t const em = expected_m == 0 ? 1 : expected_m;

#define DISPATCH_GROUPED_TILE_MX_MB2(TM, TN, ST)                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        launch_sm120_mxfp8_gemm_kernel<TM, TN, ST, 2, 16, true, true, FusedSwiGLU, ScatterCombine, false, CompactActs>(mat_a, ld_a, stride_a, mat_b,    \
            ld_b, stride_b, mat_d, ld_d, stride_d, scales_a, 0, scales_b, 0, num_groups, m_cap, shape_n, shape_k,      \
            stream, num_device_sms, grouped_layout, out_fp8, out_sf, out_tokens, row_map, weight_of_slot, max_active_groups, blk0, act_rows); \
        return;                                                                                                        \
    } while (0)

    // v1 cascade (2026-09-01, FSO_FORCE_TILE sweep on 5090 C1, Qwen3-30B-A3B
    // narrow-N shapes): the grouped kernel is latency/occupancy-bound at
    // 1 CTA/SM (ncu: no pipe > 55% SOL at decode, 76% DRAM at streaming);
    // the 2-CTAs/SM (16,64,4) instance wins or ties every narrow-N cell —
    // -10.8..-12.5% at em=1 decode, -0.2..-2.7% in the streaming band —
    // except short-K em>=16 where (32,128,4) takes -4.4% (down M=512).
    // Known concession: gate_up-like M=4 prefers (16,128,4) by ~4%; kept on
    // the mb2 route for cascade simplicity.
    if (tiles_n < 32)
    {
        // v4 (2026-09-02, dual-GPU 18-config x 14-M sweep, swM_*): the
        // grouped narrow-N cascade has TWO axes, mirroring the dense
        // E30/E31/E33 wave rules:
        //   * expected_m (rows per group): once em grows past ~32 the small
        //     decode tiles collapse — TileM must track em or large-M runs
        //     2-3x slow (gate_up M=4096: (96,128,2) 453.8 us vs the decode
        //     route's 1363.8; down M=2048: 202.6 vs 331.2).
        //   * SF-span partiality (K % 512): partial-span shapes take the
        //     pad-free Stages=2 instances throughout.
        bool const partial_span = ((shape_k + 127) / 128) % 4 != 0;
        // v5 (2026-09-04, Family C / Qwen3.5-35B-A3B, 5090 C1,
        // `sweep_sm120_20260904/`): very short K — one SF span, <= 4 k-tiles
        // (moe.down N=2048 K=512) — at em 16..24 takes the 2-CTA/SM (16,64,4)
        // instance instead of the (32,128,4) the em>=16 short-K rule below
        // picks. Decided on the ROUTED LAYER cell (down forced via
        // FSO_FORCE_TILE + FSO_FORCE_TILE_K=512, gate_up on its cascade pick,
        // 2 passes, µs graph median):
        //   em=16 (M=512):  layer 585.3 vs (32,128,4) 607.7   -3.7%   block+shared 618.0 vs 627.0  -1.4%
        //   em=20 (M=640):  layer 606.4 vs (32,128,4) 615.4   -1.5%
        //   em=24 (M=768):  layer 618.3 vs (32,128,4) 618.0    tie
        //   em=28 (M=896):  layer 632.0 vs (32,128,4) 632.1    tie   -> old pick kept
        //   em=32 (M=1024): layer 657.1 vs (32,128,4) 653.0   +0.6%  -> old pick kept
        // The isolated kernel cell (fso_mxfp8_grouped) ranks tiles
        // differently — it had (16,64,4) at -6.5% for em=32 and (16,128,4) at
        // -7.4% for em=64, yet in the layer those picks cost +2.8% and +0.5%
        // ((16,64,4) at em=64 is +16% in the layer) — so this rule and any
        // future grouped tile change is accepted on the layer cell only (see
        // the layer tables). em=64: (32,64,4) is 0.4% better than the
        // (32,128,4) pick in the layer, inside the noise floor, unchanged.
        // gate_up (K=2048) and the Family B down (K=768, 6 k-tiles) are outside
        // the gate.
        bool const very_short_k = ((shape_k + 127) / 128) <= 4;
        // P3 (2026-10-11, run fso_0.2.1_20261010/P3, RTX 5090 card 22): the Family C geometry (FC1 N = 1024,
        // i.e. tiles_n <= 8, K = 2048; FC2 very short K) at em in (20, 40] takes the two-stage 2-CTA/SM
        // (16,64,2) instance for both GEMMs instead of (16,64,4) 2-CTA/SM (em <= 32) or (32,128,4) (em > 32).
        // Decided on the routed-layer cell (both GEMMs forced, 2 passes, us graph median, cascade -> this rule):
        //   em=22 (M=704) 573.0 -> 569.9  em=24 (M=768) 572.9 -> 570.6  em=28 (M=896) 585.7 -> 583.0
        //   em=30 (M=960) 590.8 -> 584.8  em=32 (M=1024) 596.0 -> 588.8  em=36 (M=1152) 624.8 -> 610.6
        //   em=40 (M=1280) 646.2 -> 629.1
        // and outside the band it loses or ties: em=16 (M=512) 566.0 -> 566.4, em=48 (M=1536) 658.8 -> 674.8,
        // em=64 (M=2048) 665.1 -> 841.2. It is checked before the v5 rule below, which
        // therefore keeps em 16..20 of the FC2. Either GEMM alone gives about half of it at em=32 (FC1 -3.5 us, FC2
        // -3.2 us). The Family B GEMMs (FC1 N = 1536, tiles_n = 12; FC2 K = 768, a partial span) are outside the
        // gate: forced the same way they lose at em=32 (M=512 435.3 -> 438.7) and em=40 (436.3 -> 450.4).
        if (em > 20 && em <= 40 && !partial_span && (very_short_k || tiles_n <= 8))
            DISPATCH_GROUPED_TILE_MX_MB2(16, 64, 2);
        if (very_short_k && em >= 16 && em <= 24)
            DISPATCH_GROUPED_TILE_MX_MB2(16, 64, 4);
        if (em > 128)
        {
            // em=256 sweep point: (96,128,2) beats (64,128,2) by 13%/10.7%
            // (gate_up/down). Larger em unswept (slab sizes) — stays here.
            DISPATCH_GROUPED_TILE_MX(96, 128, 2);
        }
        if (em > 64)
            DISPATCH_GROUPED_TILE_MX(64, 128, 2);
        if (em > 32)
        {
            if (partial_span)
                DISPATCH_GROUPED_TILE_MX(32, 128, 2);
            DISPATCH_GROUPED_TILE_MX(32, 128, 4);
        }
        // em <= 32 decode/mid band: v3 partial-span route, then v2 rules.
        if (partial_span)
        {
            launch_sm120_mxfp8_gemm_kernel<16, 128, 2, 2, 16, true, true, FusedSwiGLU, ScatterCombine, false, CompactActs>(mat_a, ld_a, stride_a, mat_b,
                ld_b, stride_b, mat_d, ld_d, stride_d, scales_a, 0, scales_b, 0, num_groups, m_cap, shape_n, shape_k,
                stream, num_device_sms, grouped_layout, out_fp8, out_sf, out_tokens, row_map, weight_of_slot, max_active_groups, blk0, act_rows);
            return;
        }
        if (em >= 16 && shape_k <= 1024)
            DISPATCH_GROUPED_TILE_MX(32, 128, 4);
        // v2 (sweep3, post scheduler fix): at decode row-caps the wider
        // (16,128,4) single-CTA instance wins big (down M=1 -18.8%, M=4/8
        // -10/-6.8%, gate_up M=4 -10.9%) because its tile count
        // (G_active x tiles_n(128)) stays at or under the SM count, so every
        // CTA runs solo — the K/N partition sweeps measured a ~2x per-k-iter
        // penalty whenever two active CTAs share an SM. v4 refinement
        // (swM sweep): only M=1 (m_cap=4) prefers the solo (16,128,4); M=2
        // (m_cap=8) is +10.9% there and wants the 2-CTA (16,64,4) instead,
        // so gate the solo route at m_cap<=4.
        //
        // v6 (2026-09-28, run sm120_decode_tuning_20260928/s1pre, 5090 C2): that
        // gate was written when M = 2 had m_cap = 8. Since the row cap became
        // round_up(M, 4), M = 1, 2 and 4 all have m_cap = 4 and all took the solo
        // route. With the op's `max_active_groups` bound the solo route's
        // worst-case tile count is known,
        //   tiles_solo = max_active_groups x ceil(m_cap / 16) x tiles_n(128),
        // and the routed-layer sweep (custom routings with 8-20 active experts at
        // m_cap = 4, gate_up forced, GPU span) puts the solo route ahead only while
        //   0.6 x SMs <= tiles_solo <= SMs.
        // Above SMs a second round of 128-wide tiles doubles the solo route's
        // k-loop on the critical path (Family B M = 2, 192 tiles: FC1 18.93 ->
        // 17.44 us on the 2-CTA route). Below 0.6 x SMs the solo route loses in
        // two ways. It leaves most SMs idle (Family C M = 1, 64 tiles: FC1 10.17
        // -> 8.14 us). And its 82 KB CTAs leave no room beside them for the
        // FC2's 46 KB CTAs, which PDL launches while the FC1 still runs, so those
        // pack two per SM onto the few idle SMs (Family B M = 1, 96 tiles: FC1
        // unchanged at 10.65 us, FC2 7.89 -> 5.93 us behind the 47 KB 2-CTA
        // FC1; with PDL disabled the FC2 difference vanishes). Span, 2-CTA
        // against solo: Family B 96 tiles -8.6 %, 108 -1.5 %, 120 +7.6 %,
        // 144 +5.8 %, 192 -4.1 %; Family C 64 -9.6 %, 80-96 within 1.2 %,
        // 104 +4.1 %, 128 +2.7 %, 160 +4.6 %. The two families cross over at
        // different counts (B between 108 and 120, C between 96 and 104); 0.6 x
        // SMs (102 of 170) sides with C at the cost of 1.5 % at a B count of
        // 108, and top-8 routing at these two geometries only produces bounds
        // that are multiples of 64 (C) or 96 (B), none of them strictly inside
        // that window.
        // Very short K (<= 4 k-tiles, the Family C down) keeps the m_cap gate:
        // there the solo route ties or wins at every count measured, 128 to 480
        // tiles (M = 2, 256 tiles: 2-CTA +3.0 %), so the tile bound does not
        // describe it. max_active_groups == 0 (not supplied) and
        // FSO_MOE_DECODE_GATE=0 keep the m_cap gate everywhere (A/B arm).
        static bool const s_decode_gate_tiles = []
        {
            char const* e = std::getenv("FSO_MOE_DECODE_GATE");
            return !(e != nullptr && e[0] == '0');
        }();
        if (s_decode_gate_tiles && max_active_groups > 0 && !very_short_k)
        {
            uint32_t const sms = static_cast<uint32_t>(num_device_sms);
            uint32_t const tiles_solo = max_active_groups * ((m_cap + 15) / 16) * tiles_n;
            if (tiles_solo <= sms && tiles_solo * 5 >= sms * 3)
                DISPATCH_GROUPED_TILE_MX(16, 128, 4);
            DISPATCH_GROUPED_TILE_MX_MB2(16, 64, 4);
        }
        if (m_cap <= 4 && em <= 1)
            DISPATCH_GROUPED_TILE_MX(16, 128, 4);
        DISPATCH_GROUPED_TILE_MX_MB2(16, 64, 4);
    }
    if (em <= 16)
    {
        if (tiles_n >= 48 && shape_k <= 4096)
            DISPATCH_GROUPED_TILE_MX(16, 128, 4);
        DISPATCH_GROUPED_TILE_MX(32, 128, 4);
    }
    if (em <= 32)
    {
        DISPATCH_GROUPED_TILE_MX(32, 128, 4);
    }
    if (em <= 64)
    {
        if (tiles_n >= 96)
            DISPATCH_GROUPED_TILE_MX(64, 128, 2);
        DISPATCH_GROUPED_TILE_MX(64, 64, 4);
    }
    if (em <= 96)
    {
        if (tiles_n >= 32)
            DISPATCH_GROUPED_TILE_MX(96, 128, 2);
        DISPATCH_GROUPED_TILE_MX(64, 128, 2);
    }
    if (tiles_n < 32)
        DISPATCH_GROUPED_TILE_MX(64, 128, 2);
    if (tiles_n <= 48)
        DISPATCH_GROUPED_TILE_MX(96, 128, 2);
    DISPATCH_GROUPED_TILE_MX(128, 128, 2);
#undef DISPATCH_GROUPED_TILE_MX_MB2
#undef DISPATCH_GROUPED_TILE_MX
}

} // namespace kernels::blockscale_gemm
TRTLLM_NAMESPACE_END

#endif // !FSO_SM120_MXFP8_LAUNCHERS_ONLY
