/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SM100/SM103: ONE persistent kernel for a MoE layer's FC1 and FC2 on the compact slot route (issue #13 lever 5,
// run fso_issue13_20261031/I13-L5; design note I13-D45 section 4).
//
// What it replaces
// ----------------
// On the compact layout with the fused FC1 the layer runs routing, gather-quantize, FC1 (`linear_mxfp8_grouped_
// compact_swiglu`: MXFP8(silu(gate) * up) into dq / sd), FC2 (`linear_mxfp8_grouped_compact`: bf16 into dn) and the
// combine. FC2 can start only after every FC1 CTA has triggered its dependents, an FC2 CTA (230,912 B of shared
// memory) cannot share an SM with an FC1 CTA (206,336 B), and an FC2 CTA needs two to three microseconds after it
// lands before its weights stream; the DRAM is idle for 5 to 6.5 microseconds at the boundary (I13-L123, I13a). This
// kernel runs both GEMMs' tiles in one grid: every FC1 tile first, then every FC2 tile, and an FC2 tile waits only
// for the twelve FC1 tiles of its own entry (slot_moe_scheduler.cuh). The FC2 tiles that start while FC1 still runs
// stream their weights before the activation they need exists (slot_moe_mainloop.cuh).
//
// Bytes. Every tile runs the k-loop, the MMA instructions, the TMEM accumulation and the epilogue arithmetic of
// today's kernel for it (the FC1 tiles those of `FsoSm100SlotGemmCompact` with the fused SwiGLU store, the FC2 tiles
// those of the plain bounded store), and the tiles' outputs are disjoint, so dq, sd and dn are the two-kernel path's
// bit for bit. Two things differ and neither touches a value: the FC2 tiles run 8 mainloop stages instead of 9 (the
// shared-memory layout is FC1's), and the token load is the row-bounded one of lever 3 at every row capacity (the
// two-kernel path uses it at m_cap >= 32 only; it is bit-identical to the stock load, slot_rows_mainloop.cuh).
//
// Two modes, as the compact slot route has (grouped_slot_dispatch.cuh, "Tile-list mode"). Slot mode (m_cap <= 64,
// `FsoSm100SlotGemmMoe` with the slot scheduler, translation unit mxfp8_sm100_slot_moe_kernel.cu): an entry is one
// active expert, which owns one 128-row block. Token-tile mode (m_cap > 64, the same class with the scheduler's
// token-tile mode, translation unit mxfp8_sm100_slot_moe_list_kernel.cu; issue #13 lever 5 step 3, run
// fso_issue13_20261031/I13-L5b): an entry is one entry of the routing op's packed list of live 64-row token tiles,
// the token extent per batch is one whole block (pn = 128, half h of block b is N tile h of batch b), the tiles' facts
// come from the list, and the FC1 -> FC2 counters are per block (slot_moe_scheduler.cuh, "The token-tile mode"). The
// FC1 tiles of the token-tile mode are those of `FsoSm100SlotGemmCompactList` with the fused SwiGLU store, the FC2
// tiles those of its plain bounded store. Each mode's kernel lives in its own translation unit (run
// fso_issue12_20261028/R1); FSO_SLOT_MOE_LIST_TU selects the token-tile mode's entry point below.
//
// What the caller must provide (the layer does): the slot route's preconditions (compact layout; slot mode: m_cap <=
// 64, one 128-row block per active expert, the routing op's packed active-expert list; token-tile mode: the routing
// op's live token-tile list), the counters `slot_sched` as moe_build_routing_compact returns it for the same layer
// call (int32 [2 + 2G] in slot mode: 2 + G zero words, then the packed per-slot facts; in token-tile mode
// [2 + 2G + B]: B more words, one per 128-row block, live_halves << 20; the kernel leaves done[] and the block words
// nonzero), and
// a down-projection output `dn` that does NOT alias the gathered activation hq / sh: in one kernel an FC2 tile may
// store while an FC1 tile of another entry still reads its input (D45 section 4.5), which the two-kernel path's
// shared region relied on never happening.
//
// Residency. Every tile is handed out from the counter by a running CTA, so the kernel needs no assumption about which
// of its CTAs land (slot_moe_scheduler.cuh, "Freedom from deadlock"). D45's static FC1 first wave is not offered: it
// deadlocked on the B300 when two of these kernels shared the machine (run fso_issue13_20261031/I13-L5,
// tests/g1_residency).

#pragma once

#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

// The fused kernel's tile scheduler: its TileSchedulerSelector specialisation must be visible before the kernel is
// instantiated with its tag.
#include "blockscale_gemm/arch/sm100/mxfp8/slot_moe_scheduler.cuh"
// slot_work_stealing.cuh is included only because the generated compact classes name its tags; nothing here
// instantiates them.
#include "blockscale_gemm/arch/sm100/mxfp8/slot_work_stealing.cuh"

// Generated at build time into this translation unit's include directory: the slot kernel header (for the compact
// helpers) and, from it, the fused FC1 + FC2 class (make_sm100_slot_kernel.py --moe).
#include "blockscale_gemm/arch/sm100/mxfp8/sm100_slot_moe_kernel.hpp"

#include "blockscale_gemm/arch/sm100/mxfp8/slot_moe_mainloop.cuh"
#include "blockscale_gemm/arch/sm100/mxfp8/slot_rows_mainloop.cuh"

#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"

// Generated at build time too: the slot route's epilogue clones (the fused SwiGLU store and the plain store).
#include "blockscale_gemm/arch/sm100/mxfp8/sm100_fused_swiglu_slot_epilogue.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace sm100_blockscaled_gemm
{

namespace slot_moe_detail
{

using namespace cute;

// The two epilogue adapters, as grouped_slot_dispatch.cuh builds them for today's compact slot kernels
// (`FusedSwiGluSlotEpilogueWSBounded`, `PlainSlotEpilogueWSBounded`); re-declared here under their own names so that
// this translation unit does not include grouped_slot_dispatch.cuh, whose inline entry points would instantiate
// today's slot kernels in it.
template <class ET, class EC, class SC, class ED, class SD, class FC, class CT, class AC, class AD>
class MoeFusedSwiGluEpilogue
    : public cutlass::epilogue::collective::detail::Sm100TmaWarpSpecializedAdapter<
          cutlass::epilogue::collective::FsoFusedSwiGluSlotNoSmemBounded<ET, EC, SC, ED, SD, FC, CT, AC, AD>>
{
public:
    using cutlass::epilogue::collective::detail::Sm100TmaWarpSpecializedAdapter<
        cutlass::epilogue::collective::FsoFusedSwiGluSlotNoSmemBounded<ET, EC, SC, ED, SD, FC, CT, AC,
            AD>>::Sm100TmaWarpSpecializedAdapter;
};

template <class ET, class EC, class SC, class ED, class SD, class FC, class CT, class AC, class AD>
class MoePlainEpilogue
    : public cutlass::epilogue::collective::detail::Sm100TmaWarpSpecializedAdapter<
          cutlass::epilogue::collective::FsoPlainSlotNoSmemBounded<ET, EC, SC, ED, SD, FC, CT, AC, AD>>
{
public:
    using cutlass::epilogue::collective::detail::Sm100TmaWarpSpecializedAdapter<
        cutlass::epilogue::collective::FsoPlainSlotNoSmemBounded<ET, EC, SC, ED, SD, FC, CT, AC,
            AD>>::Sm100TmaWarpSpecializedAdapter;
};

template <class T>
struct RebindMoeEpilogues;

template <class ET, class EC, class SC, class ED, class SD, class FC, class CT, class AC, class AD>
struct RebindMoeEpilogues<cutlass::epilogue::collective::CollectiveEpilogue<
    cutlass::epilogue::Sm100NoSmemWarpSpecialized, ET, EC, SC, ED, SD, FC, CT, AC, AD>>
{
    using Fc1 = MoeFusedSwiGluEpilogue<ET, EC, SC, ED, SD, FC, CT, AC, AD>;
    using Fc2 = MoePlainEpilogue<ET, EC, SC, ED, SD, FC, CT, AC, AD>;
};

// The one configuration: today's compact slot tile (128 x 64 x 128, cluster 1x1x1). Both epilogues rebind the same
// NoSmem builder epilogue, declared N-major (the 32dp32b TMEM atom: one accumulator row and the whole 64-column token
// tile per thread), which is exactly what today's fused FC1 config and today's compact plain config each ask the
// builder for. The mainloop is carved against the larger of the two epilogues' shared storage (FC1's: its cross-warp
// amax buffer), i.e. FC1's 8 stages. `TileList` selects the scheduler's token-tile mode; nothing else differs.
template <bool TileList>
struct Sm100MxFP8SlotMoeConfigT
{
    static constexpr bool kTileList = TileList;
    using ElementAPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementBPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementSF = cutlass::float_ue8m0_t;
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = float;
    using LayoutATag = cutlass::layout::RowMajor;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    using LayoutDEpiTag = cutlass::layout::RowMajor;
    static constexpr int AlignmentAB = 16;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int kTileN = 64;

    using MmaTileShape = Shape<_128, Int<kTileN>, _128>;
    using ClusterShape = Shape<_1, _1, _1>;

    using StockEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm100,
        cutlass::arch::OpClassBlockScaledTensorOp, MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator, void, LayoutDEpiTag,
        AlignmentD, ElementD, LayoutDEpiTag, AlignmentD, cutlass::epilogue::NoSmemWarpSpecialized1Sm>::CollectiveOp;
    using EpilogueFc1 = typename RebindMoeEpilogues<StockEpilogue>::Fc1;
    using EpilogueFc2 = typename RebindMoeEpilogues<StockEpilogue>::Fc2;

    static constexpr int kEpilogueCarveout = static_cast<int>(
        sizeof(typename EpilogueFc1::SharedStorage) > sizeof(typename EpilogueFc2::SharedStorage)
            ? sizeof(typename EpilogueFc1::SharedStorage)
            : sizeof(typename EpilogueFc2::SharedStorage));

    using StockMainloop = typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm100,
        cutlass::arch::OpClassBlockScaledTensorOp, ElementAPair, LayoutATag, AlignmentAB, ElementBPair, LayoutBTag,
        AlignmentAB, ElementAccumulator, MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<kEpilogueCarveout>,
        cutlass::gemm::KernelTmaWarpSpecialized1SmMxf8f6f4Sm100>::CollectiveOp;
    using CollectiveMainloop = slot_detail::FsoMoeMainloop<slot_detail::FsoRowsBoundedMainloop<StockMainloop>>;

    // Swapped: (M, N, K, L) = (N_w, m_cap, K, L), for FC1 and for FC2.
    using ProblemShape = Shape<int, int, int, int>;

    using SchedulerTag = cute::conditional_t<TileList, cutlass::gemm::FsoSlotMoeWorkStealingListScheduler,
        cutlass::gemm::FsoSlotMoeWorkStealingScheduler>;
    using GemmKernel = cutlass::gemm::kernel::FsoSm100SlotGemmMoe<ProblemShape, CollectiveMainloop, EpilogueFc1,
        SchedulerTag, EpilogueFc2>;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideD = typename GemmKernel::StrideD;
    using Sm1xxBlkScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
};

using Sm100MxFP8SlotMoeConfig = Sm100MxFP8SlotMoeConfigT<false>;
using Sm100MxFP8SlotMoeListConfig = Sm100MxFP8SlotMoeConfigT<true>;

// The token-tile list's packing (grouped_slot_dispatch.cuh, `pack_token_tile`; this header cannot include that one):
// a block index below 2^13, an expert below 2^10.
inline constexpr int kMoeListMaxBlocks = 1 << 13;
inline constexpr int kMoeListMaxGroups = 1 << 10;

// The host-visible twin of this translation unit's CUTLASS_ENABLE_GDC_FOR_SM100 (see the TU).
#if defined(FSO_SM100_SLOT_MOE_GDC)
inline constexpr bool kMoeGdcCompiled = true;
#else
inline constexpr bool kMoeGdcCompiled = false;
#endif

// The same FSO_DISABLE_PDL switch and SM count every sm_100 route reads (grouped_slot_dispatch.cuh's
// slot_pdl_enabled / device_sm_count), so the routes always agree.
inline bool moe_pdl_enabled() noexcept
{
    static bool const v = []
    {
        char const* e = std::getenv("FSO_DISABLE_PDL");
        return !(e && e[0] == '1');
    }();
    return v;
}

inline int moe_device_sm_count() noexcept
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

inline bool moe_print_tile_info() noexcept
{
    static bool const on = []
    {
        char const* e = std::getenv("FSO_PRINT_TILE_INFO");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    return on;
}

// The token-tile mode's first-panel L2 prefetch (slot_moe_scheduler.cuh): on unless FSO_MOE_FIRST_PANEL_PREFETCH starts
// with `0`, read once per process. A performance hint only: the outputs are the same bytes either way.
inline bool moe_first_panel_prefetch_enabled() noexcept
{
    static bool const on = []
    {
        char const* e = std::getenv("FSO_MOE_FIRST_PANEL_PREFETCH");
        return !(e != nullptr && e[0] == '0');
    }();
    return on;
}

} // namespace slot_moe_detail

// Which (N1, K1, N2, K2) the fused kernel serves: both GEMMs' weight rows whole 128-row tiles, K whole 128-wide
// k-tiles, and the FC1 output width I = N1 / 2 a whole number of 128-column scale blocks (the fused SwiGLU store's
// rule) equal to FC2's K.
inline bool sm100_mxfp8_slot_moe_instantiated(int n1, int k1, int n2, int k2)
{
    return n1 > 0 && k1 > 0 && n2 > 0 && k2 > 0 && n1 % 256 == 0 && (n1 / 2) % 128 == 0 && k1 % 128 == 0
        && n2 % 128 == 0 && k2 == n1 / 2;
}

// The launch, one instantiation per mode (`Config`: Sm100MxFP8SlotMoeConfig or Sm100MxFP8SlotMoeListConfig; the entry
// points below). `a` [R, K1] fp8 / `sa` its one-group atom slab: the gathered activation; `w13` [G, N1, K1] (gate /
// up rows interleaved) / `sw13`; `w2` [G, N2, K2] / `sw2`; `dq` [R, K2] fp8 / `sd` its atom slab: FC1's output and
// FC2's activation; `dn` [R, N2] bf16: FC2's output. `masked_m` [G], `blk0` [G + 1] (blk0[G] = blocks in use),
// `slot_to_expert` [G] the packed active-expert list, `counters` from the routing op (see the header). Slot mode:
// `num_slots` is the host-static bound min(M * topk, G) the grid is sized from, `m_cap` <= 64 the row capacity.
// Token-tile mode: `slot_tiles` int32 [tiles_cap + 1] the routing op's live token-tile list (its count word last), the
// grid sized from tiles_cap, `m_cap` > 64. `b_alloc` = R / 128.
template <class Config>
cudaError_t launch_sm100_mxfp8_slot_moe_t(__nv_fp8_e4m3 const* a, int32_t const* sa, __nv_fp8_e4m3 const* w13,
    int32_t const* sw13, __nv_fp8_e4m3 const* w2, int32_t const* sw2, __nv_fp8_e4m3* dq, int32_t* sd,
    __nv_bfloat16* dn, int32_t const* masked_m, int32_t const* blk0, int const* slot_to_expert, int* counters,
    int const* slot_tiles, int tiles_cap, int groups, int num_slots, int m_cap, int b_alloc, int n1, int k1, int n2,
    int k2, cudaStream_t stream)
{
    namespace md = slot_moe_detail;
    using GemmKernel = typename Config::GemmKernel;
    using Params = typename GemmKernel::Params;
    using Args = typename GemmKernel::Arguments;
    constexpr bool list_mode = Config::kTileList;

    if (groups < 1 || groups > 1024 || b_alloc < 1 || blk0 == nullptr || counters == nullptr
        || slot_to_expert == nullptr || masked_m == nullptr)
        return cudaErrorInvalidValue;
    if (!sm100_mxfp8_slot_moe_instantiated(n1, k1, n2, k2))
        return cudaErrorInvalidValue;
    if constexpr (list_mode)
    {
        // The list is packed for 64-row token tiles (q | e << 14 | (r - 1) << 24).
        static_assert(Config::kTileN == 64, "the token-tile list is packed for 64-row token tiles");
        if (slot_tiles == nullptr || tiles_cap < 1 || groups > md::kMoeListMaxGroups || b_alloc > md::kMoeListMaxBlocks
            || m_cap <= Config::kTileN)
            return cudaErrorInvalidValue;
        num_slots = tiles_cap;
    }
    else
    {
        if (m_cap < 4 || m_cap > Config::kTileN || (m_cap % 4) != 0)
            return cudaErrorInvalidValue;
        if (num_slots < 1 || num_slots > groups)
            num_slots = groups;
        // Every active expert owns one 128-row block, so the live entries index blocks below b_alloc.
        if (num_slots > b_alloc)
            return cudaErrorInvalidValue;
    }

    bool const pdl = md::moe_pdl_enabled();
    int const L = groups > b_alloc ? groups : b_alloc;
    // Routed token rows per batch, on the N axis: slot mode the row capacity (one token tile), token-tile mode one
    // whole 128-row block, so that the half h of block b is N tile h of batch b.
    int const pn = list_mode ? 128 : m_cap;
    int64_t const inter = n1 / 2;

    // FC1: (M, N, K, L) = (N1, pn, K1, L), A / SFA the expert weights by expert, B / SFB the gathered tokens by
    // block (one compact block per batch), the fused store's destinations by block (launch_sm100_mxfp8_slot_compact's
    // fused slot form, exactly, in either mode).
    auto const stride_a1 = cutlass::make_cute_packed_stride(typename Config::StrideA{}, cute::make_shape(n1, k1, L));
    auto stride_b1 = cutlass::make_cute_packed_stride(typename Config::StrideB{}, cute::make_shape(pn, k1, L));
    auto const stride_d1 = cutlass::make_cute_packed_stride(typename Config::StrideD{}, cute::make_shape(n1, pn, L));
    cute::get<2>(stride_b1) = static_cast<int64_t>(128) * k1;
    auto const layout_sfa1
        = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(n1, pn, k1, L));
    auto const layout_sfb1
        = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(n1, pn, k1, L));
    // FC2: (N2, pn, K2, L), B / SFB = dq / sd by block, the plain store's output by block (the plain compact form).
    auto const stride_a2 = cutlass::make_cute_packed_stride(typename Config::StrideA{}, cute::make_shape(n2, k2, L));
    auto stride_b2 = cutlass::make_cute_packed_stride(typename Config::StrideB{}, cute::make_shape(pn, k2, L));
    auto stride_d2 = cutlass::make_cute_packed_stride(typename Config::StrideD{}, cute::make_shape(n2, pn, L));
    cute::get<2>(stride_b2) = static_cast<int64_t>(128) * k2;
    cute::get<2>(stride_d2) = static_cast<int64_t>(128) * n2;
    auto const layout_sfa2
        = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(n2, pn, k2, L));
    auto const layout_sfb2
        = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(n2, pn, k2, L));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = md::moe_device_sm_count();

    using EA = typename Config::ElementA;
    using EB = typename Config::ElementB;
    using ESF = typename Config::ElementSF;
    using ED = typename Config::ElementD;
    Args args{cutlass::gemm::GemmUniversalMode::kGemm, {n1, pn, k1, L},
        {reinterpret_cast<EA const*>(w13), stride_a1, reinterpret_cast<EB const*>(a), stride_b1,
            reinterpret_cast<ESF const*>(sw13), layout_sfa1, reinterpret_cast<ESF const*>(sa), layout_sfb1},
        {{}, nullptr, stride_d1, reinterpret_cast<ED*>(dq), stride_d1}, hw_info, {}};
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    args.epilogue.fused.ptr_h = dq;
    args.epilogue.fused.ptr_sfh = sd;
    args.epilogue.fused.h_group_elems = static_cast<long long>(128) * inter;
    args.epilogue.fused.sf_group_words = static_cast<long long>(128) * (inter / 128);
    args.problem_shape2 = cute::make_shape(n2, pn, k2, L);
    args.mainloop2 = {reinterpret_cast<EA const*>(w2), stride_a2, reinterpret_cast<EB const*>(dq), stride_b2,
        reinterpret_cast<ESF const*>(sw2), layout_sfa2, reinterpret_cast<ESF const*>(sd), layout_sfb2};
    args.epilogue2 = {{}, nullptr, stride_d2, reinterpret_cast<ED*>(dn), stride_d2};
    args.epilogue2.thread.alpha = 1.0f;
    args.epilogue2.thread.beta = 0.0f;
    args.epilogue2.plain.ptr_d = dn;
    args.epilogue2.plain.col_elems = static_cast<long long>(n2);
    args.epilogue2.plain.batch_elems = static_cast<long long>(128) * n2;
    args.fso_slot_to_expert = slot_to_expert;
    args.fso_masked_m = masked_m;
    args.fso_num_slots = num_slots;
    args.fso_num_groups = groups;
    args.scheduler.counter = counters;
    // The live entries: slot mode blk0[G] (one block per active expert), token-tile mode the list's count word.
    args.scheduler.live_slots = list_mode ? slot_tiles + tiles_cap : blk0 + groups;
    args.scheduler.tiles_m2 = (n2 + 127) / 128;
    args.scheduler.slot_to_expert = slot_to_expert;
    args.scheduler.masked_m = masked_m;
    args.scheduler.num_slots = num_slots;
    args.scheduler.num_groups = groups;
    if constexpr (list_mode)
    {
        // The list, and the per-block words after the 2 + 2G words of the slot mode (slot_moe_scheduler.cuh).
        args.scheduler.tile_list = slot_tiles;
        args.scheduler.block_done = counters + 2 + 2 * groups;
        // The first-panel prefetch (slot_moe_scheduler.cuh): FC1's weights and scales, unless the switch is off or
        // either base is not 16-byte aligned (the bulk prefetch's alignment).
        bool const prefetch = md::moe_first_panel_prefetch_enabled() && reinterpret_cast<uintptr_t>(w13) % 16 == 0
            && reinterpret_cast<uintptr_t>(sw13) % 16 == 0;
        args.scheduler.first_panel_w = prefetch ? static_cast<void const*>(w13) : nullptr;
        args.scheduler.first_panel_sf = sw13;
        args.scheduler.first_panel_k = k1;
    }

    if (GemmKernel::get_workspace_size(args) != 0)
        return cudaErrorNotSupported; // nothing here allocates; the configuration needs no workspace
    if (!GemmKernel::can_implement(args))
        return cudaErrorInvalidValue;
    Params const params = GemmKernel::to_underlying_arguments(args, nullptr); // host-only
    dim3 const grid = GemmKernel::get_grid_shape(params);
    dim3 const block = GemmKernel::get_block_shape();

    if (md::moe_print_tile_info())
    {
        static bool s_done = false;
        if (!s_done)
        {
            s_done = true;
            std::fprintf(stderr,
                "[fso] sm100 slot moe (fused FC1 + FC2, %s mode): TileM=128 TileN=%d TileK=128 cluster=1x1 stages=%d "
                "epi_smem=%d kernel_smem=%zu\n",
                list_mode ? "token-tile" : "slot", Config::kTileN,
                static_cast<int>(Config::CollectiveMainloop::DispatchPolicy::Stages), Config::kEpilogueCarveout,
                static_cast<std::size_t>(GemmKernel::SharedStorageSize));
        }
    }

    static bool s_smem_configured = false;
    if (!s_smem_configured)
    {
        if (GemmKernel::SharedStorageSize >= (48 << 10))
        {
            cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
            cudaError_t const ex = cudaThreadExchangeStreamCaptureMode(&mode);
            if (ex != cudaSuccess)
                return ex;
            cudaError_t const result = cudaFuncSetAttribute(cutlass::device_kernel<GemmKernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, GemmKernel::SharedStorageSize);
            cudaThreadExchangeStreamCaptureMode(&mode);
            if (result != cudaSuccess)
                return result;
        }
        s_smem_configured = true;
    }

    if (md::kMoeGdcCompiled && pdl)
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

// The entry points. Each translation unit defines exactly one, so that each instantiates only its own mode's kernel
// (an inline function's body is compiled, and its kernel instantiated, in every unit that sees it).
#if !defined(FSO_SLOT_MOE_LIST_TU)
// Slot mode (mxfp8_sm100_slot_moe_kernel.cu).
inline cudaError_t gemm_dispatch_sm100_mxfp8_slot_moe(__nv_fp8_e4m3 const* a, int32_t const* sa,
    __nv_fp8_e4m3 const* w13, int32_t const* sw13, __nv_fp8_e4m3 const* w2, int32_t const* sw2, __nv_fp8_e4m3* dq,
    int32_t* sd, __nv_bfloat16* dn, int32_t const* masked_m, int32_t const* blk0, int const* slot_to_expert,
    int* counters, int groups, int num_slots, int m_cap, int b_alloc, int n1, int k1, int n2, int k2,
    cudaStream_t stream)
{
    return launch_sm100_mxfp8_slot_moe_t<slot_moe_detail::Sm100MxFP8SlotMoeConfig>(a, sa, w13, sw13, w2, sw2, dq, sd,
        dn, masked_m, blk0, slot_to_expert, counters, nullptr, 0, groups, num_slots, m_cap, b_alloc, n1, k1, n2, k2,
        stream);
}
#else
// Token-tile mode (mxfp8_sm100_slot_moe_list_kernel.cu).
inline cudaError_t gemm_dispatch_sm100_mxfp8_slot_moe_list(__nv_fp8_e4m3 const* a, int32_t const* sa,
    __nv_fp8_e4m3 const* w13, int32_t const* sw13, __nv_fp8_e4m3 const* w2, int32_t const* sw2, __nv_fp8_e4m3* dq,
    int32_t* sd, __nv_bfloat16* dn, int32_t const* masked_m, int32_t const* blk0, int const* slot_to_expert,
    int* counters, int const* slot_tiles, int tiles_cap, int groups, int m_cap, int b_alloc, int n1, int k1, int n2,
    int k2, cudaStream_t stream)
{
    return launch_sm100_mxfp8_slot_moe_t<slot_moe_detail::Sm100MxFP8SlotMoeListConfig>(a, sa, w13, sw13, w2, sw2, dq,
        sd, dn, masked_m, blk0, slot_to_expert, counters, slot_tiles, tiles_cap, groups, tiles_cap, m_cap, b_alloc, n1,
        k1, n2, k2, stream);
}
#endif

} // namespace sm100_blockscaled_gemm

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
