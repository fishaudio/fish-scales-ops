/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SM100/SM103 grouped (MoE, masked-layout) MXFP8 1x32 BlockScaled GEMM types.
//
// This is the grouped sibling of `gemm_types.cuh`. The dense file instantiates
// the single-problem CUTLASS builder; here we instantiate the *pointer-array*
// (grouped) form: `cutlass::gemm::GroupProblemShape` plus the
// `KernelPtrArrayTmaWarpSpecialized{1,2}SmMxf8f6f4Sm100` mainloop schedules and
// the `PtrArray*` epilogue schedules. The layout tags become pointer tags
// (`layout::RowMajor*`), which is how CUTLASS switches every operand from one
// tensor with one stride to an array of G pointers with an array of G strides.
//
// What the grouped kernel reads from device memory, and why that matters here:
// the tile scheduler walks the per-group problem shapes through
// `GroupProblemShape::get_problem_shape(g)`, which is a plain load from the
// device-resident `problem_shapes` array, and the mainloop rebuilds its TMA
// descriptors per group from the device-resident pointer / stride / SF-layout
// arrays. Nothing about the per-group extents is baked into the host-side
// Params. That is exactly the property the masked MoE contract needs: a single
// captured CUDA graph replays correctly after the routing changes, because the
// per-group row counts are re-read from `masked_m` on every replay.
//
// Constraints inherited from the hardware:
//   * TileM must be 128 or 256. The block-scaled UMMA scale-factor atom works
//     on 128-row blocks and CUTLASS divides MMA_M by that block width without
//     a ceiling (`cutlass/detail/sm100_blockscaled_layout.hpp`), so a 64-row
//     tile collapses the SFA smem layout to zero; the 1SM atom additionally
//     asserts M == 128. A masked group whose valid row count is 1 therefore
//     still runs a 128-row tile; that waste is the granularity of
//     tcgen05.mma.blockscaled, not a dispatch choice. It costs little in the
//     MoE decode band because the cost there is dominated by streaming the
//     expert weights, which the row padding does not change.
//   * TileN may be 64, 128, 192 or 256. On the N axis CUTLASS rounds the
//     scale-factor block up instead of down (the same header uses `ceil_div`
//     there, and the collective builder enumerates N in {64, 128, 192, 256}),
//     so a narrower N tile is served by padding the scale block rather than
//     by a different scale layout — the quantizers and the slab geometry are
//     unchanged. Narrow N tiles exist so the dispatcher can trade weight
//     bytes per CTA against the number of CTAs the routed problem produces;
//     which width wins at which band is decided in grouped_dispatch.cuh.
//     Widths below 64 are rejected by the builder and are not offered here.
//   * 2SM (TileM = 256) requires an even ClusterM.
//   * The scale-factor multicast cannot span more than 4 CTAs, so both
//     cluster dimensions stay <= 4.
//
// Scale-factor layout: identical to the dense sm_100 path, i.e. the CUTLASS
// `Sm1xxBlockScaledConfig<32>` K-major atom, applied *per group*. Group g's
// activation scales occupy a slab describing a (pad(m_cap,128), K) tensor and
// group g's weight scales a slab describing (N, K). See the layout note in
// `quant_kernels.cu` and docs/api/gemm.md.

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

namespace sm100_blockscaled_gemm
{

using namespace cute;

// One kernel instantiation per (TileM, TileN, ClusterM, ClusterN, TileK,
// NoSmemEpi). `NoSmemEpi` selects the direct TMEM->register->global-store
// epilogue instead of the shared-memory-staged TMA store; it is bound per
// call site in the cascade exactly like the dense path, because the dense
// A/B measurements showed the direct store wins only where the store phase
// is a small fraction of the tile's work.
template <int TileM, int TileN, int ClusterM, int ClusterN, int TileK = 128, bool NoSmemEpi = false>
struct Sm100MxFP8GroupedGemmConfig
{
    static_assert(TileM == 128 || TileM == 256, "blockscaled UMMA: TileM must be 128 (1SM) or 256 (2SM)");
    static_assert(TileN == 64 || TileN == 128 || TileN == 192 || TileN == 256,
        "blockscaled UMMA: TileN must be 64, 128, 192 or 256 (the SF block is padded up on the N axis)");
    static_assert(TileK == 128 || TileK == 256, "TileK must be 128 or 256 (K-major mxf8f6f4 TMA constraint)");
    static constexpr bool kIs2Sm = (TileM == 256);
    static_assert(!kIs2Sm || (ClusterM % 2 == 0), "2SM MMA (TileM=256) requires even ClusterM");
    static_assert(ClusterM <= 4 && ClusterN <= 4, "block-scaled SF multicast spans at most 4 CTAs");

    using ElementAPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementBPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = float;

    // Pointer layout tags: the grouped form of the dense path's
    // (A row-major [m_cap, K], B "column-major" [N, K] i.e. K-contiguous,
    // D row-major [m_cap, N]) TN GEMM.
    using LayoutATag = cutlass::layout::RowMajor*;
    using LayoutBTag = cutlass::layout::ColumnMajor*;
    using LayoutDTag = cutlass::layout::RowMajor*;
    static constexpr int AlignmentAB = 16;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using MmaTileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
    using ClusterShape = Shape<Int<ClusterM>, Int<ClusterN>, _1>;

    using KernelSchedule = cute::conditional_t<kIs2Sm,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100>;

    using EpilogueSchedule = cute::conditional_t<NoSmemEpi,
        cute::conditional_t<kIs2Sm,
            cutlass::epilogue::PtrArrayNoSmemWarpSpecialized2Sm,
            cutlass::epilogue::PtrArrayNoSmemWarpSpecialized1Sm>,
        cute::conditional_t<kIs2Sm,
            cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm,
            cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        // No C source: D = alpha * acc with alpha = 1. A void C skips the
        // C-tile TMA load entirely rather than relying on beta == 0.
        void, LayoutDTag, AlignmentD,
        ElementD, LayoutDTag, AlignmentD,
        EpilogueSchedule>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
        ElementAPair, LayoutATag, AlignmentAB,
        ElementBPair, LayoutBTag, AlignmentAB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        ProblemShape,
        CollectiveMainloop,
        CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    using InternalLayoutSFA = typename GemmKernel::CollectiveMainloop::InternalLayoutSFA;
    using InternalLayoutSFB = typename GemmKernel::CollectiveMainloop::InternalLayoutSFB;
    using InternalStrideA = typename GemmKernel::InternalStrideA;
    using InternalStrideB = typename GemmKernel::InternalStrideB;
    using InternalStrideD = typename GemmKernel::InternalStrideD;
};

} // namespace sm100_blockscaled_gemm

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
