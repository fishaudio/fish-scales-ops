/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pieces of the sm_100/sm_103 dense MXFP8 cascade shared by its four kinds of translation unit: the dispatch TU
// (ops/mxfp8_sm100_kernel.cu, which includes dispatch.cuh and owns the identity-epilogue kernels), the
// dense-epilogue TUs (ops/mxfp8_sm100_epi_kernel_*.cu, which include dense_epilogue_gemm.cuh and own the kernels
// that apply bias / erf-GELU / residual), the split-K dense-epilogue reduce's TU
// (ops/mxfp8_sm100_splitk_epi_kernel.cu) and the quantized-output TUs (ops/mxfp8_sm100_epi_quant_kernel_*.cu,
// which include dense_epilogue_gemm.cuh and own the kernels that write linear_quantized's result, the quantized
// split-K reduce included). The dispatch TU's device code must stay exactly the identity kernels' (see
// launch_sm100_mxfp8_splitk_reduce_epi below).
//
//   * the host knobs both launchers read (FSO_PRINT_TILE_INFO, FSO_FORCE_SWIZZLE, FSO_FORCE_RASTER, PDL);
//   * Sm100DenseWorkspace, the caller-provided device scratch of one call (charter 7.5 of issue #12: per-call
//     scratch comes from the torch caching allocator on the caller's stream, never from a per-thread or
//     process-wide pool);
//   * the list of tile configurations that have a dense-epilogue instantiation, and the declaration of their
//     launcher. The definitions live in dense_epilogue_gemm.cuh and are explicitly instantiated, one group per
//     dense-epilogue TU, so the dispatch TU references them without compiling them a second time.
//
// Including this header instantiates no kernel.

#pragma once

#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#include "blockscale_gemm/common/epilogue.cuh"
#include "blockscale_gemm/arch/sm100/mxfp8/gemm_types.cuh"
#include "blockscale_gemm/arch/sm120/common/env_overrides.cuh" // FSO_FORCE_TILE parsing, fso_pdl_enabled

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace sm100_blockscaled_gemm
{

namespace detail
{

// FSO_FORCE_SWIZZLE: raster swizzle group size for the default (CLC)
// scheduler — sweep knob for L2 locality on large shapes. 0 = CUTLASS
// heuristic default. Read once (function-static), same contract as the
// other FSO_* overrides.
inline int read_force_swizzle() noexcept
{
    static int s_cache = -2;
    if (s_cache == -2)
    {
        char const* env = std::getenv("FSO_FORCE_SWIZZLE");
        s_cache = (env && *env) ? std::atoi(env) : 0;
    }
    return s_cache;
}

// FSO_FORCE_RASTER: 1 = AlongM, 2 = AlongN (default Heuristic). Probe knob
// (2026-07-06): nvjet picks h_bz (horizontal raster) for the narrow-N
// down-class at mid/large M where we lose ~17%. PROBED NEUTRAL
// (raster_probe.txt): ±2% noise on every mid/large-M cell; gate_up AlongN
// −5%. The CLC Heuristic already picks right — do not wire into the cascade.
inline int read_force_raster() noexcept
{
    static int const s_raster = []() {
        char const* e = std::getenv("FSO_FORCE_RASTER");
        return (e && *e) ? std::atoi(e) : 0;
    }();
    return s_raster;
}

// FSO_PRINT_TILE_INFO: print, once per kernel instantiation, the mainloop
// stage count the CollectiveBuilder's StageCountAutoCarveout derived and the
// shared memory the kernel ends up asking for. Host-side, read once, and off
// unless the variable is set, so it costs one int compare on the hot path.
inline bool read_print_tile_info() noexcept
{
    static int s_cache = -1;
    if (s_cache < 0)
    {
        char const* env = std::getenv("FSO_PRINT_TILE_INFO");
        s_cache = (env && *env && env[0] != '0') ? 1 : 0;
    }
    return s_cache != 0;
}

// Programmatic dependent launch of the dense CUTLASS GEMMs (2026-10-11). True when the translation unit that
// includes this header compiled CUTLASS with grid dependency control (FSO_SM100_DENSE_GDC, see
// mxfp8_sm100_kernel.cu and the mxfp8_sm100_epi_kernel_*.cu units, which all define it) and PDL is not switched off
// (FSO_DISABLE_PDL=1, the one global switch). The Stream-K variants (force codes only) keep their plain launch:
// their workspace initialisation is a separate stream-ordered operation in front of the GEMM.
#if defined(FSO_SM100_DENSE_GDC)
inline constexpr bool kDenseGdcCompiled = true;
#else
inline constexpr bool kDenseGdcCompiled = false;
#endif

inline bool dense_gemm_pdl() noexcept
{
    return kDenseGdcCompiled && tensorrt_llm::kernels::blockscale_gemm::fso_pdl_enabled();
}

// The default (CLC) scheduler's knobs: the cascade's swizzle, FSO_FORCE_SWIZZLE over it, FSO_FORCE_RASTER.
template <class SchedulerArgs>
inline void apply_default_scheduler_knobs(SchedulerArgs& scheduler, int swizzle)
{
    int const sw = read_force_swizzle();
    int const eff = sw > 0 ? sw : swizzle;
    if (eff > 0)
        scheduler.max_swizzle_size = eff;
    using cutlass::gemm::kernel::detail::RasterOrderOptions;
    int const raster = read_force_raster();
    if (raster == 1)
        scheduler.raster_order = RasterOrderOptions::AlongM;
    else if (raster == 2)
        scheduler.raster_order = RasterOrderOptions::AlongN;
}

} // namespace detail

// Device scratch of one dense call, provided by the caller: the op bodies (ops/mxfp8.cu, ops/fp8.cu) ask
// launch_sm100_mxfp8_dispatch_workspace_bytes for the size the cascade's choice needs, allocate it with the torch
// caching allocator on the stream they launch on (graph-private while that stream captures), and pass it down.
// Two calls on two streams therefore never share scratch. Today only the two-kernel parallel split-K asks for any
// (its FP32 partial sums); the plain tiles with the default scheduler ask for none. The forced Stream-K variants
// (FSO_FORCE_TILE codes 7 and 8, sweep-only) keep the per-thread Sm100WorkspacePool of dispatch.cuh.
struct Sm100DenseWorkspace
{
    void* ptr = nullptr;
    std::size_t bytes = 0;
};

// Tile configurations (TileM, TileN, ClusterM, ClusterN, TileK, NoSmemEpi) with a dense-epilogue instantiation:
// every configuration the cascade's default rules can pick (gemm_dispatch_sm100_mxfp8 in dispatch.cuh). Each one
// is instantiated twice, without and with the residual (the residual is the epilogue's C source, so it changes the
// epilogue's shared memory and its load pipeline; bias and the activation are run-time fields of one
// instantiation). One group per dense-epilogue translation unit, so the groups compile in parallel.
//
//   1-SM, TMA epilogue:    (128, 64) and (128, 192) wave tiles; (128, 128) and (128, 256) of the M <= 128 band.
//   2-SM, TMA epilogue:    (256, 64) and (256, 192) wave tiles; (256, 128) K256 (mid band, and M <= 128 with
//                          tiles_n <= 60); (256, 256) K128 (M ~ 1024 narrow N, peak band short K); (256, 128) K128
//                          (peak band, N < 4096).
//   direct-store (NoSmem): (128, 128) and (128, 256) of the decode band; (256, 256) K128 of the peak band, K >= 4096.
#define FSO_SM100_DENSE_EPI_CONFIGS_1SM_TMA(X)                                                                         \
    X(128, 64, 1, 1, 128, false)                                                                                       \
    X(128, 192, 1, 1, 128, false)                                                                                      \
    X(128, 128, 1, 1, 128, false)                                                                                      \
    X(128, 256, 1, 1, 128, false)
#define FSO_SM100_DENSE_EPI_CONFIGS_2SM_TMA(X)                                                                         \
    X(256, 64, 2, 1, 128, false)                                                                                       \
    X(256, 192, 2, 1, 128, false)                                                                                      \
    X(256, 128, 2, 1, 256, false)                                                                                      \
    X(256, 256, 2, 1, 128, false)                                                                                      \
    X(256, 128, 2, 1, 128, false)
#define FSO_SM100_DENSE_EPI_CONFIGS_NOSMEM(X)                                                                          \
    X(128, 128, 1, 1, 128, true)                                                                                       \
    X(128, 256, 1, 1, 128, true)                                                                                       \
    X(256, 256, 2, 1, 128, true)

template <int TileM, int TileN, int ClusterM, int ClusterN, int TileK, bool NoSmemEpi>
struct Sm100DenseEpiInstantiated : std::false_type
{
};

#define FSO_SM100_DENSE_EPI_TRAIT(TM_, TN_, CM_, CN_, TK_, NS_)                                                        \
    template <>                                                                                                        \
    struct Sm100DenseEpiInstantiated<TM_, TN_, CM_, CN_, TK_, NS_> : std::true_type                                    \
    {                                                                                                                  \
    };
FSO_SM100_DENSE_EPI_CONFIGS_1SM_TMA(FSO_SM100_DENSE_EPI_TRAIT)
FSO_SM100_DENSE_EPI_CONFIGS_2SM_TMA(FSO_SM100_DENSE_EPI_TRAIT)
FSO_SM100_DENSE_EPI_CONFIGS_NOSMEM(FSO_SM100_DENSE_EPI_TRAIT)
#undef FSO_SM100_DENSE_EPI_TRAIT

// One dense GEMM with the epilogue `epilogue` (bias, erf-GELU, residual; common/epilogue.cuh) applied to the fp32
// accumulator before the one rounding to bf16. kResidual must equal (epilogue.residual != nullptr). With
// `workspace_query` non-null nothing is launched: *workspace_query receives the bytes of `workspace` the call needs
// (zero for every configuration above with the default scheduler). Defined in dense_epilogue_gemm.cuh.
template <int TileM, int TileN, int ClusterM, int ClusterN, int TileK, bool NoSmemEpi, bool kResidual>
cudaError_t launch_sm100_mxfp8_gemm_epi(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int M, int N, int K, cudaStream_t stream,
    fso::DenseEpilogue const& epilogue, Sm100DenseWorkspace const& workspace, int swizzle,
    std::size_t* workspace_query);

// The grid-dependency wait of the split-K reduce kernels (the identity one in dispatch.cuh, the dense-epilogue one
// in ops/mxfp8_sm100_splitk_epi_kernel.cu). With PDL the reduce's CTAs are created while the GEMM in front of it
// still runs; this holds every one of them until that GEMM has completed and its partials are visible.
__device__ __forceinline__ void sm100_pdl_wait(bool pdl)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (pdl)
        cudaGridDependencySynchronize();
#else
    // griddepcontrol needs sm_90 or newer. This header only compiles under
    // CUTLASS_ARCH_MMA_SM100_SUPPORTED so the branch is unreachable today; it
    // keeps the file compilable if that guard is ever widened to an older arch.
    (void) pdl;
#endif
}

// The split-K reduce that applies the dense epilogue once to the sum of the FP32 partials (bias, erf-GELU,
// residual; common/epilogue.cuh), launched as the GEMM's programmatic dependent when `pdl` is set. Defined in
// ops/mxfp8_sm100_splitk_epi_kernel.cu -- its own translation unit, so that the dispatch TU's cubin keeps exactly the
// device code the identity kernels had before the dense epilogue (issue #12, R1: the driver's load-time
// finalisation of a kernel depends on its cubin's layout, and a device function added to the identity TU made the
// unchanged (256,128)K256 identity GEMM 11-18 % slower on the B300).
cudaError_t launch_sm100_mxfp8_splitk_reduce_epi(float const* partials, __nv_bfloat16* out, int splits, long mn4,
    int n4, __nv_bfloat16 const* bias, int act, __nv_bfloat16 const* residual, long ld_residual, bool pdl,
    int blocks, int threads, cudaStream_t stream);

// The quantized-output form of the same configuration (fso.dense.linear_quantized, issue #12 S3b): bias and the
// activation as above, then RN_bf16, then the activation quantizer of fso/activation_contract.h; the e4m3 codes go
// to epilogue.out_q ([M, N] row-major) and the UE8M0 words to epilogue.out_scale (kDenseOutScaleMxSm100: the Sm1xx
// atom layout over align(M, 128) rows, padded rows zero). No residual. Every configuration of the lists above has
// one, so Sm100DenseEpiInstantiated covers both forms. Defined in dense_epilogue_gemm.cuh, instantiated by the
// quantized-output TUs (ops/mxfp8_sm100_epi_quant_kernel_*.cu).
template <int TileM, int TileN, int ClusterM, int ClusterN, int TileK, bool NoSmemEpi>
cudaError_t launch_sm100_mxfp8_gemm_epi_q(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, int32_t* scales_a,
    int32_t* scales_b, int M, int N, int K, cudaStream_t stream, fso::DenseEpilogue const& epilogue,
    Sm100DenseWorkspace const& workspace, int swizzle, std::size_t* workspace_query);

// The reduce of a two-kernel split-K call with the quantized output (the third form of the reduce of
// launch_sm100_mxfp8_gemm_splitk in dispatch.cuh, next to the identity reduce defined there and the bf16-epilogue
// reduce of ops/mxfp8_sm100_splitk_epi_kernel.cu): the FP32 partials [splits, M, N] of `partials` summed in the same
// order as those two, then bias and the activation, RN_bf16, and the activation quantizer of
// fso/activation_contract.h per 32-column block, written to epilogue.out_q / epilogue.out_scale
// (kDenseOutScaleMxSm100, the words of rows M .. align(M, 128) - 1 zero). `pdl` launches it as a programmatic
// dependent of the batched GEMM, exactly as the other two reduces are launched; it reads a partial, the bias, and
// writes anything only after its grid-dependency wait. Defined in ops/mxfp8_sm100_epi_quant_kernel_nosmem.cu (a
// quantized-output TU, so that the dispatch TU's own kernels are not touched by the addition).
cudaError_t launch_sm100_mxfp8_splitk_reduce_q(float const* partials, int splits, int M, int N, cudaStream_t stream,
    fso::DenseEpilogue const& epilogue, bool pdl);

} // namespace sm100_blockscaled_gemm

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
