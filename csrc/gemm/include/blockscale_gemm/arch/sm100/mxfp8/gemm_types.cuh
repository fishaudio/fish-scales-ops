/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SM100/SM103 (Blackwell datacenter) MXFP8 1×32 BlockScaled GEMM types.
//
// Unlike the sm_120 path (hand-rolled SM120MxFP8BlockScaledBuilder +
// SM120BlockScaledKernel with inline mma.sync mxf8f6f4), the datacenter
// parts have tcgen05.mma.blockscaled with native UE8M0 scale consumption
// from TMEM — CUTLASS's Sm100 BlockScaled CollectiveBuilder path is the
// production-quality implementation, so we instantiate it directly
// instead of hand-rolling.
//
// Arch/codegen note: these kernels carry ArchTag = arch::Sm100 and are
// gated by CUTLASS_ARCH_MMA_SM100_ENABLED (__CUDA_ARCH__ == 1000). To run
// on BOTH B200 (sm_100) and B300/GB300 (sm_103), compile the TU with the
// *family* target `-gencode=arch=compute_100f,code=sm_100f` (CUDA >= 12.9;
// tcgen05.mma.blockscaled kind::mxf8f6f4 is family-portable within 10x).
// sm_103a-only builders in CUTLASS are FP4-Ultra-specific and NOT used here.
//
// Scale factor layout: CUTLASS Sm1xxBlockScaledConfig<32> K-major atom
//   ((32,4),(32,4)) : ((16,4),(0,1))
// i.e. per 128-row × 128-K block of the data tensor there is a 512-byte SF
// block; within it, the 4 consecutive K-block UE8M0 bytes of one row are
// CONTIGUOUS at byte offset (m%32)*16 + ((m/32)%4)*4. Blocks tile K-minor
// (all K SF-blocks of an M-block row are consecutive). This is NOT the
// sm_120 int32-packed [pad(M,4), K/128] K-major layout — quantize kernels
// emit the arch-appropriate layout (see quant_kernels.cu, Sm1xxSfLayout).

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace sm100_blockscaled_gemm
{

using namespace cute;

// One kernel instantiation per (TileM, TileN, ClusterM, ClusterN, TileK,
// UseStreamK).
//
// TileM/TileN are constrained to {128, 256}: the block-scaled SF TMEM
// UTCCP atom works on Blk_MN=128 row/col blocks, and the builder computes
// MMA_M / Blk_MN (integer division) — TileM=64 yields 0 and the SFA smem
// layout degenerates (verified: compile blows up inside
// sm100_blockscaled_mma_warpspecialized.hpp mma_init with Shape<_32,C<0>>).
// Small-M decode shapes therefore run a 128-row tile; that waste is the
// hardware granularity of tcgen05.mma.blockscaled, not a dispatch choice.
//
// TileK ∈ {128, 256}: 128 maximizes pipeline depth (StageCountAutoCarveout),
// 256 halves the mainloop iteration count + SF transactions — the shape
// CUTLASS's own narrow-precision examples use for peak TFLOPS.
//
// UseStreamK selects cutlass::gemm::StreamKScheduler
// (PersistentTileSchedulerSm100StreamK) instead of the default CLC
// persistent scheduler — for small-M long-K shapes where the tile count
// can't fill the machine (e.g. Qwen3 `down` decode: N=2560 → 20 CTAs on
// 148 SMs; every non-split tile config sits at ~17 µs vs cuBLAS ~11 µs).
// Stream-K needs a real workspace (partial-tile accumulators + barriers);
// the launcher's workspace pool provides it.
//
// Is2Sm is derived: TileM == 256 requires the 2SM tcgen05 atom, which in
// turn requires ClusterM % 2 == 0.
//
// NoSmemEpi selects the direct-store epilogue (NoSmemWarpSpecialized:
// TMEM→reg→STG, no smem staging / TMA store — what cuBLAS nvjet does).
// Zero epilogue smem returns the StageCountAutoCarveout carveout to the
// mainloop, deepening the TMA pipeline. ncu A/B 2026-07-06 (b300, best-shot
// cudagraph µs): WINS latency-bound bands — decode (128,128): wqkv M16
// 10.30→9.95, down M32 17.57→17.28; peak (256,256): cubic-8192 352→327 µs
// (3120→3360 TF ≈ cuBLAS parity). LOSES where L2 is near-bound (the TMA
// epilogue's smem-staged 128B bulk stores beat scattered STG sectors):
// mid-band (256,128)K256 gate M2048 40.9→43.0, gate_up M2048 75.0→81.3 µs
// → binding is per-instantiation in dispatch.cuh, NOT global.
// ElementD_ is templated for the parallel split-K path, which writes FP32
// partials to a workspace (reduced to BF16 by a separate kernel) instead of
// BF16 directly.
template <int TileM, int TileN, int ClusterM, int ClusterN, int TileK = 128, bool UseStreamK = false,
    bool NoSmemEpi = false, class ElementD_ = cutlass::bfloat16_t>
struct Sm100MxFP8GemmConfig
{
    static_assert(TileM == 128 || TileM == 256, "blockscaled UMMA: TileM must be 128 (1SM) or 256 (2SM)");
    static_assert(TileN == 128 || TileN == 256, "blockscaled UMMA: TileN must be 128 or 256 (SF Blk_MN=128)");
    static_assert(TileK == 128 || TileK == 256, "TileK must be 128 or 256 (K-major mxf8f6f4 TMA constraint)");
    static constexpr bool kIs2Sm = (TileM == 256);
    static_assert(!kIs2Sm || (ClusterM % 2 == 0), "2SM MMA (TileM=256) requires even ClusterM");

    // MXFP8: e4m3 data + UE8M0 scale, SFVecSize=32 baked into mx_float8_t.
    using ElementAPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementBPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementD = ElementD_;
    using ElementAccumulator = float;

    // A row-major [M, K], B column-major-as-[N, K] (TN GEMM) — matches the
    // linear_mxfp8 surface (x [M,K] row-major, w [N,K] row-major).
    using LayoutATag = cutlass::layout::RowMajor;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    static constexpr int AlignmentAB = 16;  // e4m3: 16 elements = 16B TMA alignment
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using MmaTileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
    using ClusterShape = Shape<Int<ClusterM>, Int<ClusterN>, _1>;

    using KernelSchedule = cute::conditional_t<kIs2Sm,
        cutlass::gemm::KernelTmaWarpSpecialized2SmMxf8f6f4Sm100,
        cutlass::gemm::KernelTmaWarpSpecialized1SmMxf8f6f4Sm100>;

    using TileSchedulerTag = cute::conditional_t<UseStreamK, cutlass::gemm::StreamKScheduler, void>;

    using EpilogueSchedule = cute::conditional_t<NoSmemEpi,
        cute::conditional_t<kIs2Sm,
            cutlass::epilogue::NoSmemWarpSpecialized2Sm,
            cutlass::epilogue::NoSmemWarpSpecialized1Sm>,
        cutlass::epilogue::collective::EpilogueScheduleAuto>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        // No C source: D = alpha * acc (alpha = 1). void C skips the
        // C-tile TMA load entirely instead of relying on beta == 0.
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

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        TileSchedulerTag>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    using LayoutSFA = typename GemmKernel::CollectiveMainloop::LayoutSFA;
    using LayoutSFB = typename GemmKernel::CollectiveMainloop::LayoutSFB;
};

} // namespace sm100_blockscaled_gemm

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
