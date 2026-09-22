/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SM100/SM103 slot-bound grouped (MoE, masked-layout) MXFP8 GEMM — the second
// entry of the grouped cascade, for the decode band only.
//
// Same public contract as `grouped_dispatch.cuh` (A [G, m_cap, K] activations,
// B [G, N, K] weights, D [G, m_cap, N] bf16, masked_m [G] on device, the two
// scale slabs exactly as fso's grouped quantizers emit them). What changes is
// the geometry the GEMM is asked to run, and how the routing reaches it.
//
// Why a second kernel at all
// --------------------------
// The pointer-array kernel puts the routed token rows on the M axis, where the
// block-scaled UMMA atom's granularity is 128 rows. At decode an expert holds
// one to a few rows, so 127 of every 128 rows of every tile are padding: the
// machine issues about 128x more tensor-core work than the problem needs, and
// — the term that actually costs time — the mainloop is too shallow to hide
// the latency of streaming the expert weights, so the weights arrive at about
// 1.7 TB/s where the same bytes can be moved at over 4 TB/s.
//
// This kernel swaps the operands. The expert weight rows go on the M axis,
// which is where a 128-row tile is a perfect fit, and the routed token rows go
// on the N axis, where the tile is 64 wide instead of 128 tall. The activation
// stage in shared memory shrinks with it, and the freed shared memory buys
// mainloop stages: `StageCountAutoCarveout` gives this configuration 8 where
// the pointer-array kernel's 128-wide N tile gets 6 and its 256-wide one gets
// 4. The extra depth is what turns the weight stream from latency-bound into
// bandwidth-bound, and it is the whole mechanism -- the tensor-core work the
// narrower tile saves is worth far less.
//
// Why the operand roles can simply be swapped, with no change to any quantizer
// ---------------------------------------------------------------------------
// `Sm1xxBlockScaledConfig<32>` builds SFA from (M, K, L) and SFB from (N, K, L)
// with the SAME atom, so exchanging which operand is A and which is B exchanges
// which slab is handed to which argument and changes nothing about either
// slab's bytes. The call site therefore hands the weight slab and the weight
// scales to A/SFA, the token slab and the token scales to B/SFB, and passes the
// problem shape (M, N, K, L) = (N_w, m_cap, K, G). With LayoutD = ColumnMajor
// over (N_w, m_cap, G) the epilogue writes the caller's ordinary row-major
// [G, m_cap, N_w] bf16 buffer in place, so no transpose kernel exists and every
// downstream consumer (the SwiGLU quantizer, the combine kernel) reads the
// buffer it already reads today.
//
// How only the routed experts get tiles
// -------------------------------------
// The kernel is CUTLASS's DENSE sm_100 block-scaled kernel, forked by the
// build-time generator `csrc/gemm/tools/make_sm100_slot_kernel.py` into
// `FsoSm100SlotGemm`. fso's masked slab is one contiguous tensor per operand
// with a uniform per-expert stride, so one 3-D TMA descriptor per operand
// already addresses every expert through the batch coordinate — the reason the
// pointer-array kernel's per-CTA descriptor rebuild is not needed here. The
// fork adds two things: the batch coordinate is remapped from a SLOT index to
// an expert id through a device slot list, and a CTA whose slot holds no expert
// (or whose token tile starts past that expert's routed row count) returns
// before it touches any state. The grid is then sized by the host-static bound
// S on the number of experts that can hold rows, so at M = 1 with top-8 routing
// the kernel launches 96 CTAs instead of the 148 the saturated grid would take.
//
// The correctness guard the caller MUST enforce
// ---------------------------------------------
// The swap orientation is numerically correct if and only if the whole row
// capacity fits in ONE token tile, i.e. m_cap <= TileN. The mechanism is in
// CUTLASS: the scale-factor tile is sized as ceil_div(CtaShape_N, 128) * 128
// columns and the global SFB tensor is tiled in whole 128-column scale blocks,
// while the index applied to that mode is the N TILE index. The two
// granularities are reconciled by hand only for the tile widths CUTLASS
// supports; a second token tile therefore addresses a scale block past the end
// of a token extent that has already been rounded up to one 128-row block, the
// out-of-bounds TMA fills the tile with zeros, and a zero UE8M0 byte denotes
// 2^-127. The output is then 2^-127 times the right answer, with no NaN, no
// error and no diagnostic of any kind.
//
// That fault is gated by "more than one token tile", and a top-k router gives
// an expert at most one row per token, so m_cap <= TileN is decidable on the
// host from m_cap alone. `grouped_dispatch.cuh` enforces it as a hard check and
// the op refuses a forced illegal call rather than running it.
//
// Dead tiles: skipped per tile, and only per tile
// -----------------------------------------------
// CUTLASS's static persistent scheduler truncates the grid to the SM count, so
// a problem with more tiles than the machine has SMs is run by CTAs that loop.
// The kernel therefore re-tests liveness at every tile, ahead of the tile's
// first TMA, rather than once for the CTA's first tile: a CTA whose first tile
// is live would otherwise still walk the dead slots' tiles and stream their
// expert weights from HBM for nothing, which was measured on Family B gate_up
// at M = 8 as a DRAM read of 1.341 times the live experts' weight bytes (runs
// b300_mxfp8_20260917/M-V1 and .../M-X1) against 1.039 with the skip in place.
//
// An earlier whole-CTA early exit, which returned before the persistent loops
// began, is gone (run b300_mxfp8_20260917/M-I1). It could only ever be armed
// when the grid was not truncated, because otherwise it dropped every live tile
// assigned to a CTA whose FIRST tile was dead; and even armed it was measured
// (run b300_mxfp8_20260917/M-I5) as never faster beyond the pass-to-pass spread
// and 2 to 7 per cent slower at M <= 4, since every CTA -- live ones included --
// paid for a throwaway tile-scheduler probe and a slot-list read that the
// per-tile test then performed again.
//
// Where the slot list comes from
// ------------------------------
// The caller may hand the list in, and the layer does: it is the fourth output
// of `moe_build_routing(..., with_slots=True)`, which builds it from the
// per-expert histogram it already holds. A call that supplies it launches
// nothing but the GEMM. A call that does not gets the one-block builder below,
// which costs a launch of 1.58 to 2.32 microseconds (run
// b300_mxfp8_20260917/M-P1 section 5.5).

#pragma once

#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Generated at build time from CUTLASS's stock sm_100 dense kernel by
// csrc/gemm/tools/make_sm100_slot_kernel.py, into the build directory — never
// into the source tree. python/setup.py runs the generator and puts the
// directory it wrote into on the include path of this translation unit alone.
// The fork needs everything the stock kernel is normally included after.
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

#include "blockscale_gemm/arch/sm100/mxfp8/sm100_slot_gemm_kernel.hpp"

#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"

// Also generated at build time, by csrc/gemm/tools/make_sm100_fused_swiglu_slot_epilogue.py,
// into the same build directory: the clone of CUTLASS's dense NoSmem EVT
// epilogue whose store emits MXFP8(silu(gate) * up) in this route's swap
// orientation. It is what the fused FC1 variant of the slot kernel binds, and
// it comes after the epilogue collective builder because the CUTLASS header it
// clones is only ever included from there.
#include "blockscale_gemm/arch/sm100/mxfp8/sm100_fused_swiglu_slot_epilogue.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace sm100_blockscaled_gemm
{

namespace slot_detail
{

using namespace cute;

// One kernel instantiation per (TileM, TileN, TileK). The cluster is fixed at
// 1x1x1 and the mainloop stage count is auto-carved against the epilogue's
// shared storage, because both are forced by the design rather than chosen:
//
//   * TileM 128 / cluster 1x1. The expert weight rows on the M axis are a
//     multiple of 128 for every MoE shape the library serves, so a 128-row tile
//     is already exact. A 2SM (TileM 256) atom is untried rather than rejected:
//     the liveness test is a pure function of the tile coordinate, so the two
//     CTAs of a cluster pair would reach the same verdict, but no such
//     configuration has been built or measured on this route.
//   * Auto-carved stages. Pinning the stage count lower fits two CTAs per SM
//     and does buy a little decode-1 latency, but it costs mainloop depth from
//     M = 4 upward and much more above that, which is the opposite of what this
//     kernel is for. The depth IS the mechanism.
template <int TileM, int TileN, int TileK>
struct Sm100MxFP8SlotGemmConfig
{
    static_assert(TileM == 128, "1SM only: cluster 1x1x1 / TileM 128 is the only configuration built and measured");
    static_assert(TileN == 64 || TileN == 128 || TileN == 192 || TileN == 256,
        "block-scaled UMMA: TileN must be 64, 128, 192 or 256 (the SF block is padded up on the N axis)");
    static_assert(TileK == 128 || TileK == 256, "TileK must be 128 or 256 (K-major mxf8f6f4 TMA constraint)");

    using ElementAPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementBPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementSF = cutlass::float_ue8m0_t;
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = float;

    // A = expert weights [N_w, K] row-major, B = routed tokens [m_cap, K] which
    // in CUTLASS's convention is the K-contiguous "column-major" B of a TN GEMM
    // — the same physical buffer the pointer-array path passes as A. D is
    // ColumnMajor over (N_w, m_cap), i.e. the caller's row-major [m_cap, N_w].
    using LayoutATag = cutlass::layout::RowMajor;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    using LayoutDTag = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentAB = 16;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using MmaTileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
    using ClusterShape = Shape<_1, _1, _1>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm100,
        cutlass::arch::OpClassBlockScaledTensorOp, MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
        // No C source: D = alpha * acc with alpha = 1, same as the grouped path.
        void, LayoutDTag, AlignmentD, ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm100,
        cutlass::arch::OpClassBlockScaledTensorOp, ElementAPair, LayoutATag, AlignmentAB, ElementBPair, LayoutBTag,
        AlignmentAB, ElementAccumulator, MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecialized1SmMxf8f6f4Sm100>::CollectiveOp;

    // Swapped: (M, N, K, L) = (N_w, m_cap, K, G).
    using ProblemShape = Shape<int, int, int, int>;

    using GemmKernel = cutlass::gemm::kernel::FsoSm100SlotGemm<ProblemShape, CollectiveMainloop, CollectiveEpilogue,
        cutlass::gemm::StaticPersistentScheduler>;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideD = typename GemmKernel::StrideD;
    using Sm1xxBlkScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
};

// ---------------------------------------------------------------------------
// The fused FC1 variant of the same kernel
// ---------------------------------------------------------------------------
//
// Same geometry, same forked kernel, same slot remap and per-tile skip; the
// only difference is the epilogue, and with it the outputs. Instead of a bf16
// [G, m_cap, 2*I] gate_up tensor it writes the fp8 [G, m_cap, I] tensor and the
// 1x32 UE8M0 scale slab that FC2 consumes as its activation operand, so the
// separate SwiGLU-and-requantise kernel disappears from the decode band. Run
// b300_mxfp8_20260917/M-R1 priced that kernel at 1.8 to 4.5 microseconds of the
// Family B M = 8 layer, on a gap of 2.03; run b300_mxfp8_20260917/M-A2 is the
// measurement of what removing it is actually worth here.
//
// Two configuration knobs differ from the unfused config, and both are forced
// by the fragment the fused store is written against rather than chosen.
//
//   * the epilogue SCHEDULE is the direct store (`NoSmemWarpSpecialized1Sm`)
//     rather than `EpilogueScheduleAuto`, which resolves to the TMA bulk store.
//     The fused store writes two transposed destinations with plain global
//     stores and stages nothing, so a TMA store epilogue would only contribute
//     a descriptor it never uses and the staged D tiles the mainloop then has
//     to give up shared memory for. The NoSmem builder also pins the epilogue
//     tile to (TileM, min(64, TileN)) = (128, 64), so one subtile is the whole
//     CTA tile and a thread's accumulator run is the complete token tile.
//   * the epilogue's D layout tag is N-MAJOR (RowMajor over (M, N)) rather than
//     the caller-facing ColumnMajor. That tag does not describe a tensor any
//     instruction touches — the fused store replaces the D store — it selects
//     the TMEM-to-register copy atom. `sm100_get_tmem_load_op` answers
//     16dp256b (stmatrix_t) for an M-major bf16 D, where a thread owns two rows
//     and two columns and the gate/up partner of a value is not in a lane
//     neighbour, and 32dp32b for an N-major one, where a thread owns ONE row
//     and 64 consecutive columns and lane = row. Only the second shape lets the
//     gate/up pair meet under a `shfl_xor(v, 1)`.
//
// What it costs: `FusedSwiGluSlotArgs` adds four scalars to the epilogue's
// Params, and the cross-warp amax exchange adds 1 KiB to the epilogue's shared
// storage. Both are charged to `StageCountAutoCarveout` like any other epilogue
// storage; the resulting mainloop stage count prints under
// FSO_PRINT_TILE_INFO=1 next to the unfused kernel's.
template <class ET, class EC, class SC, class ED, class SD, class FC, class CT, class AC, class AD>
class FusedSwiGluSlotEpilogueWS
    : public cutlass::epilogue::collective::detail::Sm100TmaWarpSpecializedAdapter<
          cutlass::epilogue::collective::FsoFusedSwiGluSlotNoSmem<ET, EC, SC, ED, SD, FC, CT, AC, AD>>
{
public:
    using cutlass::epilogue::collective::detail::Sm100TmaWarpSpecializedAdapter<
        cutlass::epilogue::collective::FsoFusedSwiGluSlotNoSmem<ET, EC, SC, ED, SD, FC, CT, AC,
            AD>>::Sm100TmaWarpSpecializedAdapter;
};

// Re-emit the builder's OWN ten epilogue template arguments against the
// generated clone.
//
// The NoSmem builder returns `CollectiveEpilogue<Sm100NoSmemWarpSpecialized,
// EpilogueTile, ElementC, StrideC, ElementD, StrideD, FusionCallbacks,
// CopyOpT2R, AlignmentC, AlignmentD>`. Pattern-matching that type and passing
// its arguments through means not a single template argument of the fused
// epilogue is guessed here: the epilogue tile, the accumulator load op and the
// alignments are whatever the builder computed for this tile shape. If a
// CUTLASS bump changes the builder's return type this fails to compile instead
// of silently binding a differently-partitioned epilogue.
template <class T>
struct RebindFusedSwiGluSlot;

template <class ET, class EC, class SC, class ED, class SD, class FC, class CT, class AC, class AD>
struct RebindFusedSwiGluSlot<cutlass::epilogue::collective::CollectiveEpilogue<
    cutlass::epilogue::Sm100NoSmemWarpSpecialized, ET, EC, SC, ED, SD, FC, CT, AC, AD>>
{
    using type = FusedSwiGluSlotEpilogueWS<ET, EC, SC, ED, SD, FC, CT, AC, AD>;
};

template <int TileM, int TileN, int TileK>
struct Sm100MxFP8SlotSwiGluGemmConfig
{
    static_assert(TileM == 128, "1SM only: cluster 1x1x1 / TileM 128 is the only configuration built and measured");
    static_assert(TileN == 64,
        "the fused slot store is written against a 64-column token tile held whole by one thread (one epilogue "
        "subtile); a wider token tile is also outside the route's correctness guard m_cap <= TileN");
    static_assert(TileK == 128, "TileK must be 128 (K-major mxf8f6f4 TMA constraint; TileK 256 was not built)");

    using ElementAPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementBPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementSF = cutlass::float_ue8m0_t;
    // ElementD is still bf16 and D is still declared, because CUTLASS sizes the
    // epilogue's register fragments from the D tile. Nothing is stored through
    // it — edit 10 of the generator deletes that store — so the D pointer and
    // stride only feed address arithmetic no instruction dereferences.
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = float;

    using LayoutATag = cutlass::layout::RowMajor;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    // See the note above: this tag is the copy-atom selector, not a description
    // of a tensor that is written.
    using LayoutDEpiTag = cutlass::layout::RowMajor;
    static constexpr int AlignmentAB = 16;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using MmaTileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
    using ClusterShape = Shape<_1, _1, _1>;

    using StockEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm100,
        cutlass::arch::OpClassBlockScaledTensorOp, MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator, void, LayoutDEpiTag,
        AlignmentD, ElementD, LayoutDEpiTag, AlignmentD,
        cutlass::epilogue::NoSmemWarpSpecialized1Sm>::CollectiveOp;

    using CollectiveEpilogue = typename RebindFusedSwiGluSlot<StockEpilogue>::type;

    // The clone adds exactly the cross-warp amax buffer and nothing else. This
    // is an assertion rather than a sentence in a comment because the whole
    // stage-count argument rests on the size of this struct.
    static_assert(sizeof(typename CollectiveEpilogue::SharedStorage)
            <= sizeof(typename StockEpilogue::SharedStorage)
                + sizeof(float) * cutlass::epilogue::collective::fso_swiglu_slot::kSlotAmaxFloats + 16,
        "the fused slot epilogue grew beyond its cross-warp amax buffer, which would cost mainloop stages");

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm100,
        cutlass::arch::OpClassBlockScaledTensorOp, ElementAPair, LayoutATag, AlignmentAB, ElementBPair, LayoutBTag,
        AlignmentAB, ElementAccumulator, MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecialized1SmMxf8f6f4Sm100>::CollectiveOp;

    // Swapped: (M, N, K, L) = (N_w, m_cap, K, G).
    using ProblemShape = Shape<int, int, int, int>;

    using GemmKernel = cutlass::gemm::kernel::FsoSm100SlotGemm<ProblemShape, CollectiveMainloop, CollectiveEpilogue,
        cutlass::gemm::StaticPersistentScheduler>;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideD = typename GemmKernel::StrideD;
    using Sm1xxBlkScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
};

// Upper bound on experts; matches `grouped_detail::kMaxGroups` and
// moe_glue.cu's kMaxGroups, so the slot-list pool is sized once and never
// reallocated.
constexpr int kMaxSlots = 1024;

// FSO_PRINT_TILE_INFO=1: print, once per instantiation, what
// StageCountAutoCarveout gave the mainloop and what the epilogue cost it. The
// fused and unfused slot kernels print the same line so the two are directly
// comparable, which is the measurement the fused epilogue's shared buffer has
// to justify itself against.
inline bool slot_print_tile_info() noexcept
{
    static bool const on = []
    {
        char const* e = std::getenv("FSO_PRINT_TILE_INFO");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    return on;
}

template <class Config>
inline void slot_report_tile_info(char const* tag, int tile_n)
{
    if (!slot_print_tile_info())
        return;
    static bool s_done = false;
    if (s_done)
        return;
    s_done = true;
    std::fprintf(stderr,
        "[fso] sm100 slot %s: TileM=128 TileN=%d TileK=128 cluster=1x1 stages=%d epi_smem=%zu kernel_smem=%zu\n", tag,
        tile_n, static_cast<int>(Config::CollectiveMainloop::DispatchPolicy::Stages),
        sizeof(typename Config::CollectiveEpilogue::SharedStorage),
        static_cast<std::size_t>(Config::GemmKernel::SharedStorageSize));
}

// Whether this translation unit compiled CUTLASS's sm_100 grid-dependency
// control instructions into the forked kernel. The defining translation unit
// is `csrc/gemm/ops/mxfp8_sm100_slot_kernel.cu`, which sets
// CUTLASS_ENABLE_GDC_FOR_SM100 for the sm_100 / sm_103 device passes only, and
// FSO_SM100_SLOT_GDC as its host-visible twin so host and device agree on
// whether the launch attribute may be set at all.
#if defined(FSO_SM100_SLOT_GDC)
inline constexpr bool kSlotGdcCompiled = true;
#else
inline constexpr bool kSlotGdcCompiled = false;
#endif

// These two mirror `grouped_detail::grouped_pdl_enabled` and
// `grouped_detail::device_sm_count` rather than calling them. They are not
// shared because `grouped_dispatch.cuh` also defines a non-static __global__
// (its pointer-array prep kernel), so a second translation unit cannot include
// that header without a duplicate-symbol link failure. Both read the same
// FSO_DISABLE_PDL switch and the same device attribute, so the two routes
// always agree.
inline bool slot_pdl_enabled() noexcept
{
    static bool const v = []
    {
        char const* e = std::getenv("FSO_DISABLE_PDL");
        return !(e && e[0] == '1');
    }();
    return v;
}

inline int device_sm_count() noexcept
{
    static int s_sms = 0;
    if (s_sms == 0)
    {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&s_sms, cudaDevAttrMultiProcessorCount, dev);
        if (s_sms <= 0)
            s_sms = 148;
    }
    return s_sms;
}

// Compact the experts that hold at least one routed row into the low slots, in
// ascending expert order, and mark the rest -1. One block, no host sync, so a
// captured graph rebuilds the list on every replay from whatever `masked_m`
// holds at replay time.
//
// The order is PACKED — active ids ascending, remainder -1. The packing is now
// a performance property rather than a correctness one: it keeps every dead
// slot at the end of the slot range, so the tiles a CTA skips are contiguous
// and the live tiles are spread evenly over the CTAs. With the liveness test
// moved inside the persistent loops (see the header comment), a reordering
// that put dead tiles at the front would no longer be wrong, but nothing here
// reorders, and the pre-fix exit mode kept for A/B measurement does still
// depend on the packing being a suffix.
//
// This builder now runs only when the caller did not supply a list of its own.
//
// The kernel is launched as a programmatic dependent (so its block starts while
// the kernel ahead of it is still on the machine) but it deliberately does NOT
// call `cudaTriggerProgrammaticLaunchCompletion`. The GEMM that follows reads
// `slot_to_expert` and `masked_m` in its prologue, and although the forked
// kernel now waits on its dependent grids before that read (generator edit 4c),
// withholding the release here costs nothing -- nothing else follows this
// kernel -- and keeps the ordering true even if that wait is ever moved.
__global__ __launch_bounds__(256) void sm100_slot_build_kernel(
    int* __restrict__ slot_to_expert, int32_t const* __restrict__ masked_m, int groups, int num_slots, bool pdl)
{
    __shared__ int s_warp_sum[8];
    __shared__ int s_base;
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;

    for (int s = tid; s < num_slots; s += 256)
        slot_to_expert[s] = -1;
    if (tid == 0)
        s_base = 0;

    // Full wait on the preceding grid before the first read of the routing.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (pdl)
        cudaGridDependencySynchronize();
#endif
    __syncthreads();

    for (int g0 = 0; g0 < groups; g0 += 256)
    {
        int const g = g0 + tid;
        int const active = (g < groups && masked_m[g] > 0) ? 1 : 0;
        int x = active;
        for (int off = 1; off < 32; off <<= 1)
        {
            int const y = __shfl_up_sync(0xffffffffu, x, off);
            if (lane >= off)
                x += y;
        }
        int const warp_excl = x - active;
        if (lane == 31)
            s_warp_sum[warp] = x;
        __syncthreads();
        int warp_off = 0;
        for (int w = 0; w < warp; ++w)
            warp_off += s_warp_sum[w];
        int block_total = 0;
        for (int w = 0; w < 8; ++w)
            block_total += s_warp_sum[w];
        if (active)
        {
            int const slot = s_base + warp_off + warp_excl;
            if (slot < num_slots)
                slot_to_expert[slot] = g;
        }
        __syncthreads();
        if (tid == 0)
            s_base += block_total;
        __syncthreads();
    }
}

// Device-side slot lists for the slot route.
//
// Two slots used alternately, and a capture-time allocation refused with a
// readable message — the same discipline as `grouped_detail::ArgPool`, for the
// same two reasons: consecutive grouped GEMMs in one MoE layer must not write
// each other's buffer, and `cudaMalloc` is forbidden inside a stream capture.
struct SlotPool
{
    static constexpr int kSlots = 2;
    int* base = nullptr;
    int next = 0;

    static SlotPool& instance()
    {
        static thread_local SlotPool p;
        return p;
    }

    // Returns the slot list to use for this call, or nullptr on allocation
    // failure. Allocation happens on the first call only.
    //
    // The capture query has to name the stream the call is being issued on.
    // A torch.cuda.graph capture runs on a side stream, and the legacy default
    // stream reports cudaStreamCaptureStatusNone throughout it, so a guard
    // written against `nullptr` never fires and the cudaMalloc below reaches
    // the driver and comes back as a bare "out of memory".
    int* acquire(cudaStream_t stream)
    {
        if (base == nullptr)
        {
            cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(stream, &cap);
            if (cap == cudaStreamCaptureStatusActive)
            {
                std::fprintf(stderr,
                    "[fish_scales_ops] sm_100 slot-bound grouped MXFP8: the slot-list pool is empty during stream "
                    "capture. Call linear_mxfp8_grouped_masked once eagerly before capturing.\n");
                std::abort();
            }
            if (cudaMalloc(&base, sizeof(int) * kMaxSlots * kSlots) != cudaSuccess)
                return nullptr;
        }
        int* const p = base + static_cast<std::size_t>(next) * kMaxSlots;
        next = (next + 1) % kSlots;
        return p;
    }
};

} // namespace slot_detail

// Launch one (TileM, TileN, TileK) instantiation of the slot-bound kernel.
//
// `num_slots` is the host-static bound S = min(M * topk, G) on the number of
// experts that can hold rows. Everything the grid depends on — S, N_w, m_cap
// and the tile shape — is host-static, so one capture per (shape, M) serves
// every routing draw at that M: only `masked_m` and the slot list change, and
// both are rebuilt on device inside the graph.
// `slot_to_expert` is the caller's own packed active-expert list, or nullptr.
// When it is given no prep kernel is launched at all and the pool is never
// touched, so a graph that only ever takes this form needs no eager warm-up of
// the pool either.
template <class Config>
cudaError_t launch_sm100_mxfp8_slot_gemm(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int32_t* masked_m, int groups, int num_slots, int m_cap, int shape_n,
    int shape_k, int const* slot_to_expert, cudaStream_t stream)
{
    using GemmKernel = typename Config::GemmKernel;
    using Params = typename GemmKernel::Params;
    using Args = typename GemmKernel::Arguments;
    namespace sd = slot_detail;

    if (groups < 1 || groups > sd::kMaxSlots)
        return cudaErrorInvalidValue;
    if (num_slots < 1 || num_slots > groups)
        num_slots = groups;

    bool const pdl = sd::slot_pdl_enabled();
    int const* slots = slot_to_expert;
    if (slots == nullptr)
    {
        int* const own = sd::SlotPool::instance().acquire(stream);
        if (own == nullptr)
            return cudaErrorMemoryAllocation;
        slots = own;
        cudaLaunchConfig_t cfg{};
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
        cfg.gridDim = dim3(1);
        cfg.blockDim = dim3(256);
        cfg.dynamicSmemBytes = 0;
        cfg.stream = stream;
        cfg.attrs = attrs;
        cfg.numAttrs = 1;
        cudaError_t const prep_err = cudaLaunchKernelEx(
            &cfg, sd::sm100_slot_build_kernel, own, static_cast<int32_t const*>(masked_m), groups, num_slots, pdl);
        if (prep_err != cudaSuccess)
            return prep_err;
    }

    // The operand role swap, spelled out. A / SFA are the expert weights and
    // their scales, B / SFB the routed tokens and theirs; both scale slabs are
    // the ones fso's grouped quantizers already return, passed through
    // untouched. `make_cute_packed_stride` fills the dynamic modes of each
    // layout tag's own stride type, so A's RowMajor and B's / D's ColumnMajor
    // do not have to agree on one cute::Stride type.
    int const pm = shape_n;  // expert weight rows, on the M axis
    int const pn = m_cap;    // routed token rows, on the N axis
    auto const stride_a = cutlass::make_cute_packed_stride(typename Config::StrideA{}, cute::make_shape(pm, shape_k, groups));
    auto const stride_b = cutlass::make_cute_packed_stride(typename Config::StrideB{}, cute::make_shape(pn, shape_k, groups));
    auto const stride_d = cutlass::make_cute_packed_stride(typename Config::StrideD{}, cute::make_shape(pm, pn, groups));

    // The batch extent handed to the scale layouts (and to the mainloop and the
    // epilogue) is `groups`, NOT num_slots: the batch coordinate the kernel
    // finally uses is the EXPERT id, which runs over all G. Only the tile
    // scheduler is given L = num_slots, which the generator's edit 5 does.
    auto const layout_sfa
        = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(pm, pn, shape_k, groups));
    auto const layout_sfb
        = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(pm, pn, shape_k, groups));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = sd::device_sm_count();

    Args args{cutlass::gemm::GemmUniversalMode::kGemm, {pm, pn, shape_k, groups},
        {reinterpret_cast<typename Config::ElementA const*>(mat_b), stride_a,
            reinterpret_cast<typename Config::ElementB const*>(mat_a), stride_b,
            reinterpret_cast<typename Config::ElementSF const*>(scales_b), layout_sfa,
            reinterpret_cast<typename Config::ElementSF const*>(scales_a), layout_sfb},
        {{}, nullptr, stride_d, reinterpret_cast<typename Config::ElementD*>(mat_d), stride_d}, hw_info, {}};
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    args.fso_slot_to_expert = slots;
    args.fso_masked_m = masked_m;
    args.fso_num_slots = num_slots;

    // Workspace: the static persistent scheduler needs none today, but ask
    // anyway and keep it in a thread-local pool that never shrinks, so a
    // capture-time growth is refused rather than silently allocating.
    static thread_local void* s_ws = nullptr;
    static thread_local std::size_t s_ws_bytes = 0;
    std::size_t const need = GemmKernel::get_workspace_size(args);
    if (need > s_ws_bytes)
    {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &cap);
        if (cap == cudaStreamCaptureStatusActive)
        {
            std::fprintf(stderr,
                "[fish_scales_ops] sm_100 slot-bound grouped MXFP8: needs %zu workspace bytes during stream capture "
                "but the pool holds %zu. Warm up linear_mxfp8_grouped_masked eagerly before capturing.\n",
                need, s_ws_bytes);
            std::abort();
        }
        if (s_ws)
            cudaFree(s_ws);
        if (need > 0 && cudaMalloc(&s_ws, need) != cudaSuccess)
            return cudaErrorMemoryAllocation;
        s_ws_bytes = need;
        if (GemmKernel::initialize_workspace(args, s_ws, stream) != cutlass::Status::kSuccess)
            return cudaErrorUnknown;
    }

    if (!GemmKernel::can_implement(args))
        return cudaErrorInvalidValue;
    sd::slot_report_tile_info<Config>("unfused", static_cast<int>(cute::size<1>(typename Config::MmaTileShape{})));
    Params const params = GemmKernel::to_underlying_arguments(args, s_ws); // host-only
    dim3 const grid = GemmKernel::get_grid_shape(params);

    // Set max dynamic smem once per instantiation, on the first (eager) call.
    static bool s_smem_configured = false;
    if (!s_smem_configured)
    {
        if (GemmKernel::SharedStorageSize >= (48 << 10))
        {
            cudaError_t const result = cudaFuncSetAttribute(cutlass::device_kernel<GemmKernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, GemmKernel::SharedStorageSize);
            if (result != cudaSuccess)
                return result;
        }
        s_smem_configured = true;
    }

    // PDL on the GEMM itself, the same switch the pointer-array route uses.
    // Compiling CUTLASS's griddepcontrol instructions in is not what arms them:
    // `cudaGridDependencySynchronize` only blocks, and
    // `cudaTriggerProgrammaticLaunchCompletion` only releases, when the kernel
    // was launched through cudaLaunchKernelEx with the programmatic
    // stream-serialisation attribute. After an ordinary <<<>>> launch they are
    // no-ops, so the attribute is what makes the trigger at the end of this
    // kernel release the consumer that follows it.
    dim3 const block = GemmKernel::get_block_shape();
    if (sd::kSlotGdcCompiled && pdl)
    {
        cudaLaunchConfig_t cfg{};
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = GemmKernel::SharedStorageSize;
        cfg.stream = stream;
        cfg.attrs = attrs;
        cfg.numAttrs = 1;
        cudaError_t const r = cudaLaunchKernelEx(&cfg, cutlass::device_kernel<GemmKernel>, params);
        if (r != cudaSuccess)
            return r;
        return cudaGetLastError();
    }
    cutlass::device_kernel<GemmKernel><<<grid, block, GemmKernel::SharedStorageSize, stream>>>(params);
    return cudaGetLastError();
}

// Phase 1 carries exactly one instantiation: swap orientation, 128 x 64 x 128,
// cluster 1x1, auto-carved stages. The kernel template takes no N or K
// parameter — those are runtime extents — so this single instantiation serves
// every published MoE class the route is aimed at (Family B gate_up
// N_w = 1536 K = 2048 and down N_w = 2048 K = 768; Family C gate_up
// N_w = 1024 K = 2048 and down N_w = 2048 K = 512). What the predicate below
// checks is therefore the shape constraint the instantiation actually imposes,
// not a shape whitelist: K is tiled 128 at a time and the ColumnMajor D's
// contiguous extent is N_w, which the 16-byte operand alignment and the
// block-scaled atom together round to 128.
//
// It exists as its own function because `grouped_dispatch.cuh` has to decide
// legality without seeing the generated kernel header, and because a later
// phase that adds narrow token tiles will make it a real table.
inline constexpr int kSlotTileN = 64;

inline bool sm100_mxfp8_slot_instantiated(int shape_n, int shape_k)
{
    return shape_n % 128 == 0 && shape_k % 128 == 0;
}

// Top-level sm_100 / sm_103 slot-bound grouped MXFP8 entry. Legality
// (m_cap <= kSlotTileN, an instantiation for (N_w, K), a known slot bound) is
// decided in `grouped_dispatch.cuh` and refused by the op before the call gets
// here; the check below is the last line of defence, not the guard.
inline cudaError_t gemm_dispatch_sm100_mxfp8_slot(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int32_t* masked_m, int groups, int num_slots, int m_cap, int shape_n,
    int shape_k, int const* slot_to_expert, cudaStream_t stream)
{
    if (m_cap > kSlotTileN || num_slots < 1 || !sm100_mxfp8_slot_instantiated(shape_n, shape_k))
        return cudaErrorInvalidValue;
    return launch_sm100_mxfp8_slot_gemm<slot_detail::Sm100MxFP8SlotGemmConfig<128, kSlotTileN, 128>>(mat_a, mat_b,
        mat_d, scales_a, scales_b, masked_m, groups, num_slots, m_cap, shape_n, shape_k, slot_to_expert, stream);
}

// ---------------------------------------------------------------------------
// The fused FC1 on the slot route
// ---------------------------------------------------------------------------
//
// Everything the unfused launcher above does — the slot list, the prep kernel,
// the programmatic dependent launch, the workspace pool, the shared-memory
// attribute — it does too; only the destinations change. Two things are
// specific to it.
//
// First, D. The generated epilogue's last edit deletes the bf16 store, but the
// D tensor is still declared because CUTLASS sizes the epilogue's register
// fragments from it. The pointer and the stride therefore only ever feed
// address arithmetic that no instruction dereferences. They are pointed at the
// real fp8 output slab rather than at a second allocation so that the address
// the epilogue forms is inside a buffer that exists, which keeps a debugger's
// view of it meaningful; compute-sanitizer confirms nothing is read or written
// through it.
//
// Second, the Params cache. There is none here, for the same reason the unfused
// slot launcher has none: `to_underlying_arguments` on this kernel is cheap
// (the dense kernel builds five TMA descriptors from host-static extents and
// does not walk a per-group argument array), and a captured graph holds the
// Params its node was captured with either way.
template <class Config>
cudaError_t launch_sm100_mxfp8_slot_swiglu_gemm(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_fp8_e4m3* out_h,
    int32_t* out_sfh, int32_t* scales_a, int32_t* scales_b, int32_t* masked_m, int groups, int num_slots, int m_cap,
    int shape_n, int shape_k, int const* slot_to_expert, cudaStream_t stream)
{
    using GemmKernel = typename Config::GemmKernel;
    using Params = typename GemmKernel::Params;
    using Args = typename GemmKernel::Arguments;
    namespace sd = slot_detail;

    if (groups < 1 || groups > sd::kMaxSlots)
        return cudaErrorInvalidValue;
    if (num_slots < 1 || num_slots > groups)
        num_slots = groups;
    // The output width I = N_w / 2 must be a whole number of the scale slab's
    // 128-column blocks, or a 1x32 block would straddle two of them. The caller
    // is refused above this point; this is the last line of defence.
    if ((shape_n & 1) != 0 || ((shape_n / 2) % 128) != 0 || (m_cap % 4) != 0)
        return cudaErrorInvalidValue;

    bool const pdl = sd::slot_pdl_enabled();
    int const* slots = slot_to_expert;
    if (slots == nullptr)
    {
        int* const own = sd::SlotPool::instance().acquire(stream);
        if (own == nullptr)
            return cudaErrorMemoryAllocation;
        slots = own;
        cudaLaunchConfig_t cfg{};
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
        cfg.gridDim = dim3(1);
        cfg.blockDim = dim3(256);
        cfg.dynamicSmemBytes = 0;
        cfg.stream = stream;
        cfg.attrs = attrs;
        cfg.numAttrs = 1;
        cudaError_t const prep_err = cudaLaunchKernelEx(
            &cfg, sd::sm100_slot_build_kernel, own, static_cast<int32_t const*>(masked_m), groups, num_slots, pdl);
        if (prep_err != cudaSuccess)
            return prep_err;
    }

    int const pm = shape_n; // expert weight rows (2*I, gate/up interleaved), on the M axis
    int const pn = m_cap;   // routed token rows, on the N axis
    int64_t const inter = shape_n / 2;
    int64_t const m_pad = (static_cast<int64_t>(m_cap) + 127) / 128 * 128;
    auto const stride_a
        = cutlass::make_cute_packed_stride(typename Config::StrideA{}, cute::make_shape(pm, shape_k, groups));
    auto const stride_b
        = cutlass::make_cute_packed_stride(typename Config::StrideB{}, cute::make_shape(pn, shape_k, groups));
    auto const stride_d
        = cutlass::make_cute_packed_stride(typename Config::StrideD{}, cute::make_shape(pm, pn, groups));

    auto const layout_sfa
        = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(pm, pn, shape_k, groups));
    auto const layout_sfb
        = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(pm, pn, shape_k, groups));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = sd::device_sm_count();

    Args args{cutlass::gemm::GemmUniversalMode::kGemm, {pm, pn, shape_k, groups},
        {reinterpret_cast<typename Config::ElementA const*>(mat_b), stride_a,
            reinterpret_cast<typename Config::ElementB const*>(mat_a), stride_b,
            reinterpret_cast<typename Config::ElementSF const*>(scales_b), layout_sfa,
            reinterpret_cast<typename Config::ElementSF const*>(scales_a), layout_sfb},
        {{}, nullptr, stride_d, reinterpret_cast<typename Config::ElementD*>(out_h), stride_d}, hw_info, {}};
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    // The two real destinations. Base pointers plus per-group element counts,
    // because the MoE slabs are contiguous in the group index and the epilogue
    // already holds the expert id.
    args.epilogue.fused.ptr_h = out_h;
    args.epilogue.fused.ptr_sfh = out_sfh;
    args.epilogue.fused.h_group_elems = static_cast<long long>(m_cap) * inter;
    args.epilogue.fused.sf_group_words = m_pad * (inter / 128);
    args.fso_slot_to_expert = slots;
    args.fso_masked_m = masked_m;
    args.fso_num_slots = num_slots;

    static thread_local void* s_ws = nullptr;
    static thread_local std::size_t s_ws_bytes = 0;
    std::size_t const need = GemmKernel::get_workspace_size(args);
    if (need > s_ws_bytes)
    {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &cap);
        if (cap == cudaStreamCaptureStatusActive)
        {
            std::fprintf(stderr,
                "[fish_scales_ops] sm_100 slot-bound fused-SwiGLU MXFP8: needs %zu workspace bytes during stream "
                "capture but the pool holds %zu. Warm up linear_mxfp8_grouped_masked_swiglu eagerly before "
                "capturing.\n",
                need, s_ws_bytes);
            std::abort();
        }
        if (s_ws)
            cudaFree(s_ws);
        if (need > 0 && cudaMalloc(&s_ws, need) != cudaSuccess)
            return cudaErrorMemoryAllocation;
        s_ws_bytes = need;
        if (GemmKernel::initialize_workspace(args, s_ws, stream) != cutlass::Status::kSuccess)
            return cudaErrorUnknown;
    }

    if (!GemmKernel::can_implement(args))
        return cudaErrorInvalidValue;
    sd::slot_report_tile_info<Config>("fused", static_cast<int>(cute::size<1>(typename Config::MmaTileShape{})));
    Params const params = GemmKernel::to_underlying_arguments(args, s_ws); // host-only
    dim3 const grid = GemmKernel::get_grid_shape(params);

    static bool s_smem_configured = false;
    if (!s_smem_configured)
    {
        if (GemmKernel::SharedStorageSize >= (48 << 10))
        {
            cudaError_t const result = cudaFuncSetAttribute(cutlass::device_kernel<GemmKernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, GemmKernel::SharedStorageSize);
            if (result != cudaSuccess)
                return result;
        }
        s_smem_configured = true;
    }

    dim3 const block = GemmKernel::get_block_shape();
    if (sd::kSlotGdcCompiled && pdl)
    {
        cudaLaunchConfig_t cfg{};
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = GemmKernel::SharedStorageSize;
        cfg.stream = stream;
        cfg.attrs = attrs;
        cfg.numAttrs = 1;
        cudaError_t const r = cudaLaunchKernelEx(&cfg, cutlass::device_kernel<GemmKernel>, params);
        if (r != cudaSuccess)
            return r;
        return cudaGetLastError();
    }
    cutlass::device_kernel<GemmKernel><<<grid, block, GemmKernel::SharedStorageSize, stream>>>(params);
    return cudaGetLastError();
}

// Which (N_w, K) the fused slot variant is instantiated for. The extra clause
// over `sm100_mxfp8_slot_instantiated` is the output width: I = N_w / 2 must be
// a whole number of the scale slab's 128-column blocks, so that a 1x32 output
// block never straddles two of them. Every MoE class this route serves as FC1
// satisfies it (Family B N_w = 1536 -> I = 768, Family C N_w = 1024 -> I = 512).
inline bool sm100_mxfp8_slot_swiglu_instantiated(int shape_n, int shape_k)
{
    return sm100_mxfp8_slot_instantiated(shape_n, shape_k) && (shape_n % 2 == 0) && ((shape_n / 2) % 128 == 0);
}

// Top-level sm_100 / sm_103 slot-bound fused-SwiGLU FC1 entry. Legality
// (m_cap <= kSlotTileN, an instantiation for (N_w, K), a known slot bound) is
// decided in `grouped_dispatch.cuh` and refused by the op before the call gets
// here; the check below is the last line of defence, not the guard.
inline cudaError_t gemm_dispatch_sm100_mxfp8_slot_swiglu(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b,
    __nv_fp8_e4m3* out_h, int32_t* out_sfh, int32_t* scales_a, int32_t* scales_b, int32_t* masked_m, int groups,
    int num_slots, int m_cap, int shape_n, int shape_k, int const* slot_to_expert, cudaStream_t stream)
{
    if (m_cap > kSlotTileN || num_slots < 1 || !sm100_mxfp8_slot_swiglu_instantiated(shape_n, shape_k))
        return cudaErrorInvalidValue;
    return launch_sm100_mxfp8_slot_swiglu_gemm<slot_detail::Sm100MxFP8SlotSwiGluGemmConfig<128, kSlotTileN, 128>>(
        mat_a, mat_b, out_h, out_sfh, scales_a, scales_b, masked_m, groups, num_slots, m_cap, shape_n, shape_k,
        slot_to_expert, stream);
}

} // namespace sm100_blockscaled_gemm

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
