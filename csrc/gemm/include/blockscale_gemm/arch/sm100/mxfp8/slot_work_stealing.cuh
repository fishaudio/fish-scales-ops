/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// A software work-stealing tile scheduler for the sm_100 / sm_103 slot-bound
// grouped MXFP8 GEMM on the compact activation layout (fso 0.2.1).
//
// Why the slot kernel needs a dynamic scheduler
// ---------------------------------------------
// The slot kernel streams one 128-row panel of an expert's weights per tile.
// Under CUTLASS's static persistent scheduler CTA i takes tiles i, i + grid,
// i + 2 * grid, ... whatever its SM's speed, so the slowest SM sets the end of
// the kernel: run fso_0.2.1_20261010/S10a measured SM-busy fractions of
// 0.83-0.94 and slowest SMs at 0.74-0.90 on this kernel. Handing tiles out on
// demand removed that tail (CUTLASS's cluster-launch-control scheduler over the
// live slots made the same kernel 2.3-4.0 % faster on FC1 and 3.8-9.7 % on FC2).
//
// Why not CUTLASS's cluster-launch-control (CLC) scheduler itself
// ----------------------------------------------------------------
// CLC hands out the CTAs of the launched grid, so its grid must be every tile
// that might be work. In a captured CUDA graph the grid is fixed when the graph
// is captured, while the number of experts that hold rows is known only on the
// device, after the routing kernel ran; so a capturable CLC grid has to be the
// host-static slot bound min(M * topk, G) times the weight tiles, and every dead
// slot's tile then costs a CLC round trip or a launched-and-exiting CTA (S10a:
// Family C M = 32 FC2 14 % slower over the host-static bound than over the live
// slots). This scheduler takes the live tile count from the device instead: the
// grid is the host-static persistent grid min(SMs, bound tiles), and a CTA asks
// a global counter for its next tile, so dead work never reaches it.
//
// The protocol
// ------------
//   * Tiles are numbered slot-major: t = (slot * tiles_n + n) * tiles_m + m, so
//     consecutive tiles share an expert's token panel and the live tiles are the
//     prefix t < live_slots * tiles_n * tiles_m (the slot list is packed: the
//     active experts fill the low slots).
//   * The first `static_waves` waves of the grid are handed out statically: CTA b
//     takes tiles b, b + grid, ... below static_tiles = static_waves * grid, with
//     no global access (a dead slot's tile is skipped by the kernel's own per-tile
//     test, as under the static persistent scheduler). The host picks the waves
//     (slot_static_waves in grouped_slot_dispatch.cuh): one -- draws from the
//     second wave on -- when the bound holds more than two waves, else all of
//     them. Every further tile is static_tiles + atomicAdd(&counter[0], 1)
//     against the live tile count. The scheduler warp issues that
//     draw while the CTA's mainloop is on its current tile -- the CLC throttle
//     handshake paces it exactly as it paces a CLC query -- and publishes the
//     answer in the kernel's CLC response pipeline: it waits for the response
//     slot to be free (the empty barrier, without arming transaction bytes),
//     writes the 16-byte response with an ordinary shared store and arrives on
//     the slot's full barrier, whose arrival has release semantics. So the forked
//     kernel's warp roles consume a drawn tile exactly as they consume a CLC
//     response. (st.async, which would complete transaction bytes instead, is
//     refused outside a cluster of at least two CTAs: compute-sanitizer reports
//     an invalid shared write and the launch fails with an illegal instruction.)
//   * A draw at or past the live tile count is the CTA's last: its scheduler warp
//     reports an invalid tile, and before that it increments counter[1]. The CTA
//     whose increment completes the grid (old value gridDim.x - 1) knows every
//     draw of this launch has happened and puts both counters back to zero. The
//     counter is therefore zero between kernels in stream order, which is what
//     lets the two slot GEMMs of one layer call share it (the second one draws
//     only after its griddepcontrol.wait, i.e. after the first one completed).
//   * The live slot count is the device scalar the routing op already publishes
//     (blk0[G] = 128-row blocks in use = active experts on this route, where every
//     active expert owns exactly one block), read once per CTA after the
//     scheduler warp's griddepcontrol.wait, in flight together with the first
//     draw. In the compact kernel's tile-list mode (row capacities past one token
//     tile, run fso_0.2.1_20261010/S10b-3b) a slot is one entry of the routing
//     op's packed list of live 64-row token tiles (`tiles_n` = 1), and the live
//     count is the list's own count word, which the routing op writes after the
//     last entry. Whether a launch draws at all is the host's verdict for the whole grid
//     (static tiles below the bound tiles), so either every CTA draws -- and the
//     finished-CTA count sees all of them -- or none does. When none does, no warp
//     uses the response pipeline: each role computes its next tile itself and the
//     counter is never touched, so a small call (M <= 2 or 4, at most two waves of
//     bound tiles) runs as the static persistent kernel would, without the
//     scheduler warp's round trips in the way of the dependent launch.
//   * The draw is issued before the scheduler warp waits for a free response
//     slot, and the forked kernel's MMA and epilogue warps ask for the next tile
//     only after the current one's work (generator edits c7 / c8), while the load
//     warp lets the scheduler run two tiles ahead (c9): the atomic's round trip
//     -- about a microsecond while the weights stream -- stays off the path from
//     one tile to the next. (Measured on the first version, which drew one tile
//     ahead and was waited for before each tile's MMA: Family C M = 1 FC1 / FC2
//     20 / 30 per cent slower than the static scheduler.)
//
// Who zeroes the counter: `moe_build_routing_compact(..., with_slots=True)`
// returns it zeroed as its ninth output (`slot_sched`, int32 [2]), a per-call
// tensor, so two layer calls replayed concurrently from two graphs never share
// one. A caller of the compact GEMM op that passes none gets a zeroed one from
// the caching allocator (one fill kernel). The kernel leaves it zero.
//
// Requirements (all enforced by the launcher): cluster 1x1x1 (a 2-SM tile
// would need a cluster-wide broadcast of the response), and a grid of at most
// gridDim.x <= total bound tiles.

#pragma once

#include "cutlass/arch/barrier.h"
#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/pipeline/pipeline.hpp"

#include <cstdint>

namespace cutlass::gemm
{
// The tag the slot launcher hands the forked kernel as its TileSchedulerTag.
struct FsoSlotWorkStealingScheduler
{
};
} // namespace cutlass::gemm

namespace cutlass::gemm::kernel::detail
{

template <class ClusterShape_, uint32_t Stages_>
class FsoSlotWorkStealingTileScheduler
{
public:
    using ClusterShape = ClusterShape_;
    static constexpr bool IsDynamicPersistent = true;
    static constexpr uint32_t Stages = Stages_;
    static_assert(cute::is_static_v<ClusterShape> && cute::size(ClusterShape{}) == 1,
        "the fso slot work-stealing scheduler publishes each drawn tile to its own CTA only (cluster 1x1x1)");

    // Same 16-byte opaque response the kernel reserves per pipeline stage for CLC.
    struct CLCResponse
    {
        uint32_t data[4] = {0};
    };

    struct WorkTileInfo
    {
        int32_t M_idx = 0;
        int32_t N_idx = 0;
        int32_t L_idx = 0;
        bool is_valid_tile = false;

        CUTLASS_HOST_DEVICE bool is_valid() const
        {
            return is_valid_tile;
        }

        CUTLASS_HOST_DEVICE static WorkTileInfo invalid_work_tile()
        {
            return {-1, -1, -1, false};
        }

        CUTLASS_HOST_DEVICE bool is_final_split(uint32_t) const
        {
            return true;
        }

        CUTLASS_HOST_DEVICE int32_t reduction_subtile_idx() const
        {
            return -1;
        }
    };

    struct Arguments
    {
        // int32 [2]: tile draws, finished CTAs. Zero when the kernel starts; zero again when it ends.
        int* counter = nullptr;
        // Device scalar: the number of live slots (a packed prefix of the slot range).
        int const* live_slots = nullptr;
        // Waves of the grid handed out statically (CTA b: tiles b, b + grid, ...) before the draws begin; at
        // least 1. A value that covers every bound tile makes the kernel the static persistent one.
        int static_waves = 1;
        // N tiles per slot when > 0, instead of ceil(N / TileN). The compact kernel's tile-list mode (any row
        // capacity) gives each entry of its live token-tile list one N tile: the slot IS a 64-row token tile, and
        // the kernel rewrites the coordinate to that tile's block and half itself.
        int tiles_n = 0;
    };

    struct Params
    {
        int* counter = nullptr;
        int const* live_slots = nullptr;
        int32_t tiles_m = 1;
        int32_t tiles_n = 1;
        int32_t tiles_l = 1;
        int32_t grid = 1;
        int32_t bound_tiles = 1;  // tiles of the host-static slot bound
        int32_t static_tiles = 1; // tiles below this are handed out statically
    };

    using Pipeline = PipelineCLCFetchAsync<Stages, ClusterShape>;

    //
    // Host side
    //
    template <class ProblemShapeMNKL, class TileShape, class AtomThrShape, class ClusterShapeMNK>
    static Params to_underlying_arguments(ProblemShapeMNKL problem_shape_mnkl, TileShape tile_shape_mnk,
        AtomThrShape, ClusterShapeMNK, KernelHardwareInfo const& hw_info, Arguments const& args,
        [[maybe_unused]] void* workspace = nullptr)
    {
        Params p;
        p.counter = args.counter;
        p.live_slots = args.live_slots;
        p.tiles_m = static_cast<int32_t>(
            (cute::get<0>(problem_shape_mnkl) + cute::size<0>(tile_shape_mnk) - 1) / cute::size<0>(tile_shape_mnk));
        p.tiles_n = args.tiles_n > 0 ? static_cast<int32_t>(args.tiles_n)
                                     : static_cast<int32_t>((cute::get<1>(problem_shape_mnkl)
                                           + cute::size<1>(tile_shape_mnk) - 1) / cute::size<1>(tile_shape_mnk));
        p.tiles_l = static_cast<int32_t>(cute::get<3>(problem_shape_mnkl));
        long long const total = static_cast<long long>(p.tiles_m) * p.tiles_n * p.tiles_l;
        long long const sms = hw_info.sm_count > 0 ? hw_info.sm_count : 1;
        p.grid = static_cast<int32_t>(total < sms ? (total > 0 ? total : 1) : sms);
        p.bound_tiles = static_cast<int32_t>(total > 0 ? total : 1);
        long long const waves = args.static_waves > 0 ? args.static_waves : 1;
        long long const st = waves * p.grid;
        p.static_tiles = static_cast<int32_t>(st < p.bound_tiles ? st : p.bound_tiles);
        return p;
    }

    template <class ProblemShapeMNKL, class TileShape, class AtomThrShape, class ClusterShapeMNK>
    CUTLASS_HOST_DEVICE static dim3 get_grid_shape(Params const& params, ProblemShapeMNKL, TileShape, AtomThrShape,
        ClusterShapeMNK, KernelHardwareInfo)
    {
        return dim3(static_cast<uint32_t>(params.grid), 1u, 1u);
    }

    template <class ProblemShape, class ElementAccumulator>
    static size_t get_workspace_size(
        Arguments const&, ProblemShape, KernelHardwareInfo const&, uint32_t, uint32_t = 1, uint32_t = 1)
    {
        return 0;
    }

    template <class ProblemShape, class ElementAccumulator>
    static cutlass::Status initialize_workspace(Arguments const&, void*, cudaStream_t, ProblemShape,
        KernelHardwareInfo const&, uint32_t, uint32_t = 1, uint32_t = 1, CudaHostAdapter* = nullptr)
    {
        return Status::kSuccess;
    }

    static bool can_implement(Arguments const& args)
    {
        return args.counter != nullptr && args.live_slots != nullptr;
    }

    //
    // Device side
    //
    CUTLASS_DEVICE
    FsoSlotWorkStealingTileScheduler(CLCResponse* clc_response_ptr, Params const& params, dim3)
        : clc_response_ptr_(clc_response_ptr)
        , params_(params)
    {
    }

    CUTLASS_DEVICE WorkTileInfo decode(int32_t t, bool valid) const
    {
        int32_t const mn = params_.tiles_m * params_.tiles_n;
        int32_t const l = t / mn;
        int32_t const r = t - l * mn;
        int32_t const n = r / params_.tiles_m;
        int32_t const m = r - n * params_.tiles_m;
        return {m, n, l, valid};
    }

    // CTA b's first tile is tile b, no draw; the grid never exceeds the bound tiles, so it is in range. Whether it is
    // LIVE is the kernel's own per-tile test (a slot past the live prefix holds -1 in the slot list).
    template <class ClusterShapeMNK>
    CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(ClusterShapeMNK)
    {
        return decode(static_cast<int32_t>(blockIdx.x), true);
    }

    CUTLASS_DEVICE auto work_tile_to_cta_coord(WorkTileInfo const& w) const
    {
        return cute::make_coord(w.M_idx, w.N_idx, cute::_, w.L_idx);
    }

    // Whether the whole launch is static (the host's static waves cover every bound tile): then no warp uses the
    // response pipeline at all -- every role computes its next tile itself, as under the static persistent
    // scheduler, so a call that cannot draw also pays nothing for the machinery that would.
    CUTLASS_DEVICE bool all_static() const
    {
        return params_.static_tiles >= params_.bound_tiles;
    }

    // The scheduler warp: one tile per call, published through the CLC response pipeline.
    CUTLASS_DEVICE
    PipelineState<Stages> advance_to_next_work(Pipeline& pipeline, PipelineState<Stages> state)
    {
        if (all_static())
            return state; // nothing is published, so nothing has to be released before the kernel ends
        bool const leader = cute::elect_one_sync();
        WorkTileInfo w = WorkTileInfo::invalid_work_tile();
        bool last_draw = false;
        if (leader)
        {
            if (next_static_ < 0)
                next_static_ = static_cast<int32_t>(blockIdx.x) + params_.grid;
            if (next_static_ < params_.static_tiles)
            {
                // The static waves: no global access. A tile of a dead slot is skipped by the kernel's own
                // per-tile test, exactly as under the static persistent scheduler.
                w = decode(next_static_, true);
                next_static_ += params_.grid;
            }
            else if (params_.static_tiles < params_.bound_tiles)
            {
                // The draws. Whether there are any is decided on the host for the whole grid, so either every
                // CTA draws (and the finished-CTA count below sees the whole grid) or none does. The live slot
                // count and the draw go out together; the draw does not depend on the count.
                if (live_tiles_ < 0)
                    live_tiles_ = *params_.live_slots * params_.tiles_m * params_.tiles_n;
                int32_t const t = params_.static_tiles + atomicAdd(params_.counter, 1);
                if (t < live_tiles_)
                    w = decode(t, true);
                else
                    last_draw = true;
            }
        }
        __syncwarp();
        // Wait until every consumer released this response slot. `producer_acquire` would also arm 16
        // transaction bytes on the full barrier (what a CLC query completes); the tile is published with a
        // plain arrival instead, so only the wait half is taken.
        while (!(pipeline.producer_try_acquire(state) == BarrierStatus::WaitDone))
        {
        }
        if (leader)
        {
            uint32_t const full = pipeline.producer_get_barrier(state);
            uint32_t const dst = cute::cast_smem_ptr_to_uint(&clc_response_ptr_[state.index()]);
            // The response, then the full barrier's one expected arrival: mbarrier.arrive is a release at CTA
            // scope, and the consumers' wait on the barrier an acquire, so they read the stored response.
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(dst),
                "r"(static_cast<uint32_t>(w.M_idx)), "r"(static_cast<uint32_t>(w.N_idx)),
                "r"(static_cast<uint32_t>(w.L_idx)), "r"(w.is_valid_tile ? 1u : 0u)
                : "memory");
            asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n" ::"r"(full) : "memory");
            if (last_draw)
            {
                // This CTA's last draw, after its answer is out. The release fence orders the draw before the
                // finished-CTA count, so the CTA that completes that count sees every draw of the launch done
                // before it puts both counters back to zero.
                __threadfence();
                if (atomicAdd(params_.counter + 1, 1) == static_cast<int>(gridDim.x) - 1)
                {
                    __threadfence();
                    atomicExch(params_.counter, 0);
                    atomicExch(params_.counter + 1, 0);
                }
            }
        }
        __syncwarp();
        ++state;
        return state;
    }

    // Every warp role: take the next drawn tile from the pipeline.
    template <class TileSchedulerPipeline, class TileSchedulerPipelineState>
    CUTLASS_DEVICE auto fetch_next_work(
        WorkTileInfo current, TileSchedulerPipeline& pipeline, TileSchedulerPipelineState state)
    {
        if (all_static())
        {
            // The next tile of this CTA's static walk, with no pipeline step (the `false` tells the kernel not to
            // advance its pipeline state, and the scheduler warp then stops querying).
            int32_t const t = (current.L_idx * params_.tiles_n + current.N_idx) * params_.tiles_m + current.M_idx
                + params_.grid;
            WorkTileInfo const w = t < params_.bound_tiles ? decode(t, true) : WorkTileInfo::invalid_work_tile();
            return cute::make_tuple(w, false);
        }
        pipeline.consumer_wait(state);
        uint32_t const src = cute::cast_smem_ptr_to_uint(&clc_response_ptr_[state.index()]);
        uint32_t m, n, l, v;
        asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(m), "=r"(n), "=r"(l), "=r"(v)
                     : "r"(src)
                     : "memory");
        cutlass::arch::fence_view_async_shared();
        pipeline.consumer_release(state);
        WorkTileInfo const w{static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(l), v != 0u};
        return cute::make_tuple(w, true);
    }

    //
    // K tile API (every tile covers the whole K extent)
    //
    template <class ProblemShapeMNKL, class TileShape, class Shape>
    CUTLASS_DEVICE auto get_k_tile_iterator(WorkTileInfo const&, ProblemShapeMNKL problem_shape_MNKL,
        TileShape tile_shape, Shape)
    {
        auto k_tiles = cute::ceil_div(cute::get<2>(problem_shape_MNKL), cute::get<2>(tile_shape));
        return cute::make_coord_iterator(k_tiles);
    }

    template <class ProblemShape, class TileShape>
    CUTLASS_HOST_DEVICE static int get_work_k_tile_count(
        WorkTileInfo const&, ProblemShape problem_shape, TileShape tile_shape)
    {
        return cute::size(cute::ceil_div(cute::get<2>(problem_shape), cute::get<2>(tile_shape)));
    }

    CUTLASS_HOST_DEVICE static uint32_t get_work_k_tile_start(WorkTileInfo const&)
    {
        return 0u;
    }

    CUTLASS_HOST_DEVICE static bool compute_epilogue(WorkTileInfo const&, Params const&)
    {
        return true;
    }

    CUTLASS_HOST_DEVICE static bool compute_epilogue(WorkTileInfo const&)
    {
        return true;
    }

    CUTLASS_HOST_DEVICE static bool requires_fixup(Params const&, WorkTileInfo const)
    {
        return false;
    }

    template <bool IsComplex, class TiledMma, class AccEngine, class AccLayout, class AccumulatorPipeline,
        class AccumulatorPipelineState, class CopyOpT2R>
    CUTLASS_DEVICE AccumulatorPipelineState fixup(TiledMma const&, WorkTileInfo const&,
        cute::Tensor<AccEngine, AccLayout>&, AccumulatorPipeline, AccumulatorPipelineState acc_pipe_consumer_state,
        CopyOpT2R) const
    {
        return acc_pipe_consumer_state;
    }

    CUTLASS_DEVICE static bool continue_current_work(WorkTileInfo&)
    {
        return false;
    }

private:
    CLCResponse* clc_response_ptr_ = nullptr;
    Params const& params_;
    int32_t live_tiles_ = -1;
    int32_t next_static_ = -1;
};

// Bind the tag for the sm_100 kernel layer.
template <class TileShape, class ClusterShape, uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<FsoSlotWorkStealingScheduler, arch::Sm100, TileShape, ClusterShape,
    SchedulerPipelineStageCount>
{
    using Scheduler = FsoSlotWorkStealingTileScheduler<ClusterShape, SchedulerPipelineStageCount>;
};

} // namespace cutlass::gemm::kernel::detail
