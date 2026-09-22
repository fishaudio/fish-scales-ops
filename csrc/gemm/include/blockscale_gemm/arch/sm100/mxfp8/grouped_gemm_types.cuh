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

// The fused-SwiGLU FC1 epilogue (the clone of CUTLASS's pointer-array NoSmem
// EVT epilogue whose store emits MXFP8(silu(gate)*up)). The header is written
// into the BUILD directory by csrc/gemm/tools/make_sm100_fused_swiglu_epilogue.py
// and only the translation unit that compiles the sm_100 grouped kernels gets
// that directory on its include path (see FsoBuildExtension in python/setup.py),
// which is why this header is the only file in the source tree that names it.
#include "blockscale_gemm/arch/sm100/mxfp8/sm100_fused_swiglu_epilogue.hpp"

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

// ---------------------------------------------------------------------------
// The fused-SwiGLU FC1 variant of the same pointer-array kernel
// ---------------------------------------------------------------------------
//
// An MoE layer's first projection (FC1, the gate_up projection) is followed by
// a second kernel that reads the bf16 [G, m_cap, 2*I] result back, computes
// silu(gate) * up and quantises the result to MXFP8 for the second projection.
// The config below removes that second kernel: the GEMM keeps its mainloop and
// its tile scheduler and only its STORE changes, from "write bf16 accumulators"
// to "pair the columns, apply SwiGLU in FP32, take the 32-wide amax, derive the
// UE8M0 byte and write the fp8 bytes plus the scale byte". The measured effect
// on the FC1 stage alone is in run b300_mxfp8_20260917/M-E1 section 4.
//
// Two preconditions, both of which the caller must satisfy because the kernel
// cannot detect their violation:
//
//   * the FC1 weight rows are INTERLEAVED, row 2j being gate_j and row 2j+1
//     being up_j. The reason is the shape of the accumulator fragment a thread
//     holds; the long form is in fused_swiglu_store.cuh. Fed the usual
//     [gate; up] stacking the kernel computes silu(gate_j) * gate_{j+I/2}-ish
//     nonsense and reports no error, which is why the op that drives it says so
//     in its docstring and in docs/api/gemm.md.
//   * the output width I = N/2 is a multiple of 128, so a 1x32 scale block
//     never straddles the atom slab's 128-column block boundary.
//
// Everything else — the argument arrays, the prep kernel, the masked contract,
// the Params cache — is shared with the unfused launcher above.
//
// Why the epilogue OpClass differs from the unfused config. The unfused config
// passes `OpClassTensorOp`, which sends the CUTLASS builder down the legacy
// `thread::LinearCombination` branch and therefore selects the DEFAULT-fusion
// specialisation of the pointer-array NoSmem epilogue — the one that loads the
// WHOLE CTA tile of the accumulator into one thread's registers.
// `OpClassBlockScaledTensorOp` selects the EVT specialisation instead, which
// divides the CTA tile by the epilogue tile that the NoSmem builder pins to
// (TileM, min(64, TileN)). That is the specialisation whose fragment is one
// output row by 64 consecutive N columns, i.e. exactly 32 interleaved gate/up
// pairs, i.e. exactly one 1x32 output scale block — so the pairing and the amax
// are register-local, with no shuffle and no shared memory. The fused store is
// only expressible against that fragment shape.
template <class ET, class EC, class SC, class ED, class SD, class FC, class CT, class AC, class AD>
class FusedSwiGluEpilogueWS
    : public cutlass::epilogue::collective::detail::Sm100TmaWarpSpecializedAdapter<
          cutlass::epilogue::collective::FsoFusedSwiGluPtrArrayNoSmem<ET, EC, SC, ED, SD, FC, CT, AC, AD>>
{
public:
    using cutlass::epilogue::collective::detail::Sm100TmaWarpSpecializedAdapter<
        cutlass::epilogue::collective::FsoFusedSwiGluPtrArrayNoSmem<ET, EC, SC, ED, SD, FC, CT, AC,
            AD>>::Sm100TmaWarpSpecializedAdapter;
};

// Re-emit the builder's OWN nine epilogue template arguments against the
// generated clone.
//
// The builder returns `CollectiveEpilogue<Sm100PtrArrayNoSmemWarpSpecialized,
// EpilogueTile, ElementC, StrideC, ElementD, StrideD, FusionCallbacks,
// CopyOpT2R, AlignmentC, AlignmentD>`. Pattern-matching that type and passing
// its arguments through means not a single template argument of the fused
// epilogue is guessed here: the epilogue tile, the accumulator load op and the
// alignments are whatever the builder computed for this tile shape. If a
// CUTLASS bump changes the builder's return type this fails to compile instead
// of silently binding a differently-partitioned epilogue.
template <class T>
struct RebindFusedSwiGlu;

template <class ET, class EC, class SC, class ED, class SD, class FC, class CT, class AC, class AD>
struct RebindFusedSwiGlu<cutlass::epilogue::collective::CollectiveEpilogue<
    cutlass::epilogue::Sm100PtrArrayNoSmemWarpSpecialized, ET, EC, SC, ED, SD, FC, CT, AC, AD>>
{
    using type = FusedSwiGluEpilogueWS<ET, EC, SC, ED, SD, FC, CT, AC, AD>;
};

// One kernel instantiation per (TileM, TileN, TileK) of the fused FC1. The
// cluster is fixed at (1,1,1) and TileM at 128 because the fused store is
// written against the 1SM NoSmem epilogue's fragment; the 2SM schedules use a
// different epilogue class and are not offered here (run
// b300_mxfp8_20260917/M-E1 section 8.5 records that as untried rather than
// rejected). TileN 64 is excluded because the cascade never selects it and
// because a 64-wide N tile is a single epilogue subtile, which leaves nothing
// for the store loop to amortise.
template <int TileM, int TileN, int TileK = 128>
struct Sm100MxFP8GroupedSwiGluGemmConfig
{
    static_assert(TileM == 128, "the fused FC1 epilogue is written against the 1SM NoSmem epilogue (TileM = 128)");
    static_assert(TileN == 128 || TileN == 192 || TileN == 256,
        "fused FC1: TileN must be 128, 192 or 256 (the widths the pointer-array cascade selects)");
    static_assert(TileN % 64 == 0, "fused FC1: TileN must be a whole number of 64-column epilogue subtiles");
    static_assert(TileK == 128, "fused FC1: TileK 128 only (TileK 256 was not built; see M-E1 section 8.5)");

    using ElementAPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementBPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    // ElementD is still bf16 and D is still declared, because the epilogue's
    // register accumulator is sized from the D tile. Nothing is stored through
    // it — edit 9 of the generator deletes that store — so the D pointer array
    // only feeds address arithmetic (see the launcher's comment on d_base).
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = float;

    using LayoutATag = cutlass::layout::RowMajor*;
    using LayoutBTag = cutlass::layout::ColumnMajor*;
    using LayoutDTag = cutlass::layout::RowMajor*;
    static constexpr int AlignmentAB = 16;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using MmaTileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
    using ClusterShape = Shape<_1, _1, _1>;

    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized1Sm;

    using StockEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        void, LayoutDTag, AlignmentD,
        ElementD, LayoutDTag, AlignmentD,
        EpilogueSchedule>::CollectiveOp;

    using CollectiveEpilogue = typename RebindFusedSwiGlu<StockEpilogue>::type;

    // The fused store writes through plain global stores and stages nothing, so
    // the epilogue's shared-memory footprint is unchanged and therefore
    // `StageCountAutoCarveout` gives the mainloop the same number of stages it
    // gives the stock epilogue. That is the property the whole "free in
    // resources" claim rests on, so it is a compile-time assertion rather than
    // a sentence in a comment.
    static_assert(sizeof(typename CollectiveEpilogue::SharedStorage) == sizeof(typename StockEpilogue::SharedStorage),
        "the fused FC1 epilogue changed the epilogue shared-memory footprint, which would cost a mainloop stage");

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
