/*
 * Unified SM100/SM103 128 x 128 x 128 Block-List Persistent MoE Tile Scheduler (v2).
 *
 * ONE single persistent kernel serves ALL M = 1..8192+, padded CUDA Graph
 * buckets, skewed routing, and optional fused Shared Expert (expert E) with
 * ZERO host-side `if M > X` / `if expected_m > X` dispatch:
 *
 *   - 1 compact 128-row block = 1 MoE tile (NO half-block h=0/1, NO live_halves<<20,
 *     NO vertical chunk_list stragglers!).
 *   - `block_list[0..blocks_total-1]` packs:
 *       b (bits 0..12) | expert (bits 13..22) | (rows - 1) (bits 23..29)
 *     and `block_list[blocks_cap] = blocks_total` (ONE scalar read!).
 *   - Shared Expert blocks sit FIRST in `block_list[0..nb_shared-1]` so they run
 *     in parallel from wave 0 -> ZERO straggler on padded buckets (`pad:256/1`!).
 *   - Live-bounded warp register cache (skips register copy when L <= 32 since
 *     all entries sit in 1 L1 cache line; caches 33..256 entries in scheduler warp
 *     registers so `ld.acquire.gpu` L1 invalidation never re-fetches block_list).
 *   - Device-gated first-panel L2 bulk prefetch ONLY when L >= 48.
 *   - Device-gated 2-panel grouping when (T1 + T2) * L >= 14 * grid (halving atomic
 *     draws, acquire latency, and FC1 release traffic at medium/large M).
 *   - Deferred-draw straggler guard (FB5) for 1..2-round launches.
 *   - Multi-stream residency deadlock free (every tile drawn, FC1 before FC2).
 */

#pragma once

#include "blockscale_gemm/arch/sm100/mxfp8/slot_moe_scheduler.cuh"
#include "cutlass/arch/barrier.h"
#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/pipeline/pipeline.hpp"

#include <cstdint>

namespace cutlass::gemm
{
struct FsoUnifiedMoeSchedulerTag
{
};
} // namespace cutlass::gemm

namespace cutlass::gemm::kernel::detail
{

template <class ClusterShape_, uint32_t Stages_>
class FsoUnifiedMoeTileScheduler
{
public:
    using ClusterShape = ClusterShape_;
    using Pipeline = cutlass::PipelineCLCFetchAsync<Stages_, ClusterShape>;
    using PipelineState = typename Pipeline::PipelineState;
    using ThrottlePipeline = cutlass::PipelineAsync<Stages_>;
    using ThrottlePipelineState = typename ThrottlePipeline::PipelineState;

    struct CLCResponse
    {
        uint32_t data[4];
    };

    static constexpr uint32_t NumThreadsPerWarp = 32;
    static constexpr uint32_t NumThreadsPerWarpGroup = 128;
    static constexpr bool IsDynamicPersistent = true;
    static constexpr uint32_t Stages = Stages_;
    static constexpr int32_t kListRegs = 8; // 8 * 32 = 256 block entries in warp registers
    static constexpr int32_t kMaxRegList = kListRegs * 32;
    static constexpr int32_t kGroupMinTilesPerCta = 14;
    static constexpr int32_t kPrefetchMinBlocks = 48;

    struct WorkTileInfo
    {
        int32_t M_idx = 0; // weight panel index (0..T1-1 or 0..T2-1)
        int32_t N_idx = 0; // always 0 (128-row block = 1 TileN=128)
        int32_t L_idx = 0; // compact block index b
        int32_t a_batch = 0; // expert id e
        int32_t rows = 0; // live rows in block b (1..128)
        int32_t phase = 0; // 0 = FC1, 1 = FC2
        uint32_t release = 0u; // FC1 epilogue release count (0, 1, or 2)
        bool ready = false; // FC2: true if done[b] >= T1 at draw time
        bool live = false;
        bool is_valid_tile = false;

        CUTLASS_HOST_DEVICE bool is_valid() const { return is_valid_tile; }
        CUTLASS_HOST_DEVICE static WorkTileInfo invalid_work_tile() { return {}; }
        CUTLASS_HOST_DEVICE int32_t get_linear_idx() const { return L_idx; }
    };

    struct Arguments
    {
        int* counter = nullptr;           // slot_sched[2 + blocks_cap], zeroed by routing kernel
        int const* block_list = nullptr;  // block_list[blocks_cap + 1], last word = live block count
        int32_t blocks_cap = 0;
        int32_t tiles_m2 = 0;
        int32_t num_groups = 0;
        void const* first_panel_w = nullptr;
        int32_t const* first_panel_sf = nullptr;
        int32_t first_panel_k = 0;
    };

    struct Params
    {
        int* counter = nullptr;
        int const* block_list = nullptr;
        int32_t blocks_cap = 0;
        int32_t tiles_m1 = 0;
        int32_t tiles_m2 = 0;
        int32_t num_groups = 0;
        int32_t grid = 0;
        void const* first_panel_w = nullptr;
        int32_t const* first_panel_sf = nullptr;
        int32_t first_panel_k = 0;
    };

    template <class ProblemShapeMNKL, class TileShape, class AtomThrShape, class ClusterShapeMNK>
    static Params to_underlying_arguments(ProblemShapeMNKL problem_shape, TileShape tile_shape, AtomThrShape,
        ClusterShapeMNK, KernelHardwareInfo const& hw_info, Arguments const& args,
        [[maybe_unused]] void* workspace = nullptr)
    {
        int32_t const t1 = static_cast<int32_t>(
            cute::size(cute::ceil_div(cute::get<0>(problem_shape), cute::get<0>(tile_shape))));
        int32_t const t2 = args.tiles_m2;
        int32_t const sm_count = hw_info.sm_count > 0 ? hw_info.sm_count : 1;
        int64_t const max_tiles = static_cast<int64_t>(args.blocks_cap) * static_cast<int64_t>(t1 + t2);
        int32_t const grid = static_cast<int32_t>(
            max_tiles < static_cast<int64_t>(sm_count) ? (max_tiles > 0 ? max_tiles : 1) : sm_count);
        return Params{args.counter, args.block_list, args.blocks_cap, t1, t2, args.num_groups, grid,
            args.first_panel_w, args.first_panel_sf, args.first_panel_k};
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
        return args.counter != nullptr && args.block_list != nullptr && args.blocks_cap > 0 && args.tiles_m2 > 0;
    }

    CUTLASS_DEVICE
    FsoUnifiedMoeTileScheduler(CLCResponse* clc_response_ptr, Params const& params, dim3)
        : clc_response_ptr_(clc_response_ptr)
        , params_(params)
    {
    }

    // Valid, dead placeholder tile so every warp role enters its persistent loop cleanly and
    // draws real work via the atomic counter (guaranteeing multi-stream residency safety).
    CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(ClusterShape)
    {
        WorkTileInfo w{};
        w.is_valid_tile = true;
        w.live = false;
        return w;
    }

    // Called on ALL threads immediately after `wait_on_dependent_grids()`.
    CUTLASS_DEVICE void fso_resolve_first(WorkTileInfo&)
    {
        int32_t live_blocks = params_.block_list[params_.blocks_cap];
        if (live_blocks < 0)
            live_blocks = 0;
        else if (live_blocks > params_.blocks_cap)
            live_blocks = params_.blocks_cap;
        live_blocks_ = live_blocks;
        int32_t const cta = static_cast<int32_t>(blockIdx.x);
        int32_t const t1 = params_.tiles_m1;

        if (live_blocks <= 0)
            return;

        int32_t const guess_idx = cta / t1;
        int32_t const clamped_idx = guess_idx < live_blocks ? guess_idx : (live_blocks - 1);
        uint32_t const entry = fso_ld_facts(params_.block_list + clamped_idx);

        // Device-gated first-panel L2 prefetch: ONLY when live_blocks >= 48 (hides cold weight
        // DRAM latency on medium/large launches without adding TMA queue pressure on decode!).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)
        if (live_blocks >= kPrefetchMinBlocks && guess_idx < live_blocks && params_.first_panel_w != nullptr
            && threadIdx.x == 128)
        {
            int32_t const expert = static_cast<int32_t>((entry >> 13) & 0x3ffu);
            if (expert < params_.num_groups)
            {
                int32_t const panel = cta - guess_idx * t1;
                int64_t const k = static_cast<int64_t>(params_.first_panel_k);
                int64_t const w_expert_bytes = static_cast<int64_t>(t1) * 128 * k;
                int64_t const w_panel_bytes = static_cast<int64_t>(128) * k;
                char const* const wp = reinterpret_cast<char const*>(params_.first_panel_w)
                    + static_cast<int64_t>(expert) * w_expert_bytes + static_cast<int64_t>(panel) * w_panel_bytes;
                constexpr uint32_t kChunk = 16384u;
                for (int64_t off = 0; off < w_panel_bytes; off += kChunk)
                {
                    uint32_t const rem = static_cast<uint32_t>(
                        (w_panel_bytes - off) < kChunk ? (w_panel_bytes - off) : kChunk);
                    fso_prefetch_l2_bulk(wp + off, rem);
                }
                if (params_.first_panel_sf != nullptr)
                {
                    int64_t const sf_panel_bytes = static_cast<int64_t>(128) * (k >> 7) * 4;
                    int64_t const sf_expert_bytes = static_cast<int64_t>(t1) * sf_panel_bytes;
                    char const* const sfp = reinterpret_cast<char const*>(params_.first_panel_sf)
                        + static_cast<int64_t>(expert) * sf_expert_bytes + static_cast<int64_t>(panel) * sf_panel_bytes;
                    for (int64_t off = 0; off < sf_panel_bytes; off += kChunk)
                    {
                        uint32_t const rem = static_cast<uint32_t>(
                            (sf_panel_bytes - off) < kChunk ? (sf_panel_bytes - off) : kChunk);
                        fso_prefetch_l2_bulk(sfp + off, rem);
                    }
                }
            }
        }
        else
#endif
        {
            // Keep L1 line warm for small/decode launches so scheduler warp's first lookup hits L1.
            fso_keep_facts(entry, params_.block_list + clamped_idx);
        }
    }

    CUTLASS_DEVICE auto work_tile_to_cta_coord(WorkTileInfo const& w) const
    {
        return cute::make_coord(w.M_idx, w.N_idx, cute::_, w.L_idx);
    }

    CUTLASS_DEVICE int* done_counter(int block_id) const
    {
        return params_.counter + 2 + block_id;
    }

    CUTLASS_DEVICE uint32_t done_target(WorkTileInfo const&) const
    {
        return static_cast<uint32_t>(params_.tiles_m1);
    }

    CUTLASS_DEVICE uint32_t fso_release_count(WorkTileInfo const& w) const
    {
        return w.release;
    }

    CUTLASS_DEVICE
    PipelineState advance_to_next_work(Pipeline& pipeline, PipelineState state)
    {
        int const lane = static_cast<int>(threadIdx.x & 31);
        int32_t const L = live_blocks_;
        int32_t const t1 = params_.tiles_m1;
        int32_t const t2 = params_.tiles_m2;

        // Lazy one-time init of scheduler-warp register cache (only when 32 < L <= 256).
        if (!sched_init_)
        {
            sched_init_ = true;
            if (L > 32 && L <= kMaxRegList)
            {
                use_reg_list_ = true;
#pragma unroll
                for (int r = 0; r < kListRegs; ++r)
                {
                    int32_t const pos = r * 32 + lane;
                    list_regs_[r] = (pos < L) ? params_.block_list[pos] : 0;
                }
            }
            grouped_ = (static_cast<int64_t>(t1 + t2) * L
                >= static_cast<int64_t>(kGroupMinTilesPerCta) * params_.grid);
            g1_ = grouped_ ? ((t1 + 1) >> 1) : t1;
            g2_ = grouped_ ? ((t2 + 1) >> 1) : t2;
            d1_ = static_cast<int64_t>(g1_) * L;
            d_total_ = static_cast<int64_t>(g1_ + g2_) * L;
            second_round_gate_ = (!grouped_) && (d_total_ > params_.grid)
                && (d_total_ <= params_.grid + (d1_ < params_.grid ? d1_ : params_.grid));
        }

        WorkTileInfo w{};
        if (has_pending_)
        {
            has_pending_ = false;
            w = pending_w_;
        }
        else if (exhausted_ || L <= 0)
        {
            w.is_valid_tile = false;
        }
        else
        {
            // Deferred-draw guard (FB5): avoid a spinning FC2 CTA grabbing a 2nd-round tile
            // before its first FC2 tile clears when the whole launch is <= 2 rounds.
            if (defer_block_ >= 0)
            {
                if (lane == 0)
                {
                    int const* const word = done_counter(defer_block_);
                    uint32_t const target = static_cast<uint32_t>(t1);
                    while (fso_ld_acquire_gpu(word) < target)
                    {
                        if (static_cast<int64_t>(fso_ld_relaxed_gpu(params_.counter)) >= d_total_)
                            break;
                        __nanosleep(256);
                    }
                }
                __syncwarp();
                defer_block_ = -1;
            }

            int64_t draw = 0;
            if (lane == 0)
            {
                uint32_t d32 = 0u;
                asm volatile("atom.global.add.u32 %0, [%1], 1;\n"
                             : "=r"(d32)
                             : "l"(params_.counter)
                             : "memory");
                draw = static_cast<int64_t>(d32);
            }
            draw = static_cast<int64_t>(__shfl_sync(0xFFFFFFFFu, static_cast<uint32_t>(draw), 0));

            if (draw >= d_total_)
            {
                exhausted_ = true;
                w.is_valid_tile = false;
                if (lane == 0)
                {
                    uint32_t done_ctas = 0u;
                    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;\n"
                                 : "=r"(done_ctas)
                                 : "l"(params_.counter + 1)
                                 : "memory");
                    if (done_ctas + 1u == static_cast<uint32_t>(params_.grid))
                    {
                        params_.counter[0] = 0;
                        params_.counter[1] = 0;
                    }
                }
            }
            else if (draw < d1_)
            {
                // FC1 draw
                int32_t const idx = static_cast<int32_t>(draw / g1_);
                int32_t const sub = static_cast<int32_t>(draw - static_cast<int64_t>(idx) * g1_);
                int32_t const panel0 = grouped_ ? (sub << 1) : sub;
                int32_t const cnt = (grouped_ && (panel0 + 1 < t1)) ? 2 : 1;
                int32_t const entry = lookup_entry(idx);
                int32_t const b = entry & 0x1fff;
                int32_t const e = (entry >> 13) & 0x3ff;
                int32_t const r = ((entry >> 23) & 0x7f) + 1;

                w.M_idx = panel0;
                w.N_idx = 0;
                w.L_idx = b;
                w.a_batch = e;
                w.rows = r;
                w.phase = 0;
                w.ready = true;
                w.live = true;
                w.is_valid_tile = true;
                w.release = (cnt == 1) ? 1u : 0u;

                if (cnt == 2)
                {
                    pending_w_ = w;
                    pending_w_.M_idx = panel0 + 1;
                    pending_w_.release = 2u;
                    has_pending_ = true;
                }
            }
            else
            {
                // FC2 draw
                int64_t const u = draw - d1_;
                int32_t const idx = static_cast<int32_t>(u / g2_);
                int32_t const sub = static_cast<int32_t>(u - static_cast<int64_t>(idx) * g2_);
                int32_t const panel0 = grouped_ ? (sub << 1) : sub;
                int32_t const cnt = (grouped_ && (panel0 + 1 < t2)) ? 2 : 1;
                int32_t const entry = lookup_entry(idx);
                int32_t const b = entry & 0x1fff;
                int32_t const e = (entry >> 13) & 0x3ff;
                int32_t const r = ((entry >> 23) & 0x7f) + 1;

                int is_ready = 0;
                if (lane == 0)
                    is_ready = (fso_ld_acquire_gpu(done_counter(b)) >= static_cast<uint32_t>(t1)) ? 1 : 0;
                is_ready = __shfl_sync(0xFFFFFFFFu, is_ready, 0);

                if (!is_ready && second_round_gate_)
                    defer_block_ = b;

                w.M_idx = panel0;
                w.N_idx = 0;
                w.L_idx = b;
                w.a_batch = e;
                w.rows = r;
                w.phase = 1;
                w.ready = (is_ready != 0);
                w.live = true;
                w.is_valid_tile = true;
                w.release = 0u;

                if (cnt == 2)
                {
                    pending_w_ = w;
                    pending_w_.M_idx = panel0 + 1;
                    has_pending_ = true;
                }
            }
        }

        while (!(pipeline.producer_try_acquire(state) == BarrierStatus::WaitDone))
        {
        }
        if (cute::elect_one_sync())
        {
            uint32_t const full = pipeline.producer_get_barrier(state);
            uint32_t const dst = cute::cast_smem_ptr_to_uint(&clc_response_ptr_[state.index()]);
            uint32_t const w0 = (static_cast<uint32_t>(w.M_idx) & 0xffffu)
                | (static_cast<uint32_t>(w.N_idx) << 16);
            uint32_t const r_enc = w.rows > 0 ? static_cast<uint32_t>(w.rows - 1) : 0u;
            uint32_t const w1 = (static_cast<uint32_t>(w.L_idx) & 0x3fffu)
                | ((static_cast<uint32_t>(w.a_batch) & 0x3ffu) << 14)
                | ((r_enc & 0x7fu) << 24);
            uint32_t const w2 = (w.is_valid_tile ? 1u : 0u)
                | (w.live ? 2u : 0u)
                | (w.phase != 0 ? 4u : 0u)
                | (w.ready ? 8u : 0u)
                | ((w.release & 3u) << 4);
            uint32_t const w3 = 0u;
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(dst), "r"(w0), "r"(w1), "r"(w2), "r"(w3)
                : "memory");
            asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n" ::"r"(full) : "memory");
        }
        __syncwarp();
        ++state;
        return state;
    }

    template <class TileSchedulerPipeline, class TileSchedulerPipelineState>
    CUTLASS_DEVICE auto fetch_next_work(
        WorkTileInfo, TileSchedulerPipeline& pipeline, TileSchedulerPipelineState state)
    {
        pipeline.consumer_wait(state);
        uint32_t const src = cute::cast_smem_ptr_to_uint(&clc_response_ptr_[state.index()]);
        uint32_t w0, w1, w2, w3;
        asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(w0), "=r"(w1), "=r"(w2), "=r"(w3)
                     : "r"(src)
                     : "memory");
        (void) w3;
        cutlass::arch::fence_view_async_shared();
        pipeline.consumer_release(state);

        WorkTileInfo w{};
        w.M_idx = static_cast<int32_t>(w0 & 0xffffu);
        w.N_idx = static_cast<int32_t>(w0 >> 16);
        w.L_idx = static_cast<int32_t>(w1 & 0x3fffu);
        w.a_batch = static_cast<int32_t>((w1 >> 14) & 0x3ffu);
        w.rows = static_cast<int32_t>(((w1 >> 24) & 0x7fu) + 1u);
        w.is_valid_tile = (w2 & 1u) != 0u;
        w.live = (w2 & 2u) != 0u;
        w.phase = (w2 & 4u) != 0u ? 1 : 0;
        w.ready = (w2 & 8u) != 0u;
        w.release = (w2 >> 4) & 3u;
        return cute::make_tuple(w, true);
    }

    template <class ProblemShapeMNKL, class TileShape, class Shape>
    CUTLASS_DEVICE auto get_k_tile_iterator(
        WorkTileInfo const&, ProblemShapeMNKL problem_shape_MNKL, TileShape tile_shape, Shape)
    {
        auto k_tiles = cute::ceil_div(cute::get<2>(problem_shape_MNKL), cute::get<2>(tile_shape));
        return cute::make_coord_iterator(k_tiles);
    }

    template <class ProblemShape, class TileShape>
    CUTLASS_HOST_DEVICE static int get_work_k_tile_count(
        WorkTileInfo const&, ProblemShape problem_shape, TileShape tile_shape)
    {
        return static_cast<int>(cute::size(cute::ceil_div(cute::get<2>(problem_shape), cute::get<2>(tile_shape))));
    }

    CUTLASS_HOST_DEVICE static uint32_t get_work_k_tile_start(WorkTileInfo const&) { return 0u; }
    CUTLASS_HOST_DEVICE static bool compute_epilogue(WorkTileInfo const&, Params const&) { return true; }
    CUTLASS_HOST_DEVICE static bool compute_epilogue(WorkTileInfo const&) { return true; }
    CUTLASS_HOST_DEVICE static bool requires_fixup(Params const&, WorkTileInfo const) { return false; }

    template <bool IsComplex, class TiledMma, class AccEngine, class AccLayout, class AccumulatorPipeline,
        class AccumulatorPipelineState, class CopyOpT2R>
    CUTLASS_DEVICE AccumulatorPipelineState fixup(TiledMma const&, WorkTileInfo const&,
        cute::Tensor<AccEngine, AccLayout>&, AccumulatorPipeline, AccumulatorPipelineState acc_pipe_consumer_state,
        CopyOpT2R) const
    {
        return acc_pipe_consumer_state;
    }

    CUTLASS_DEVICE static bool continue_current_work(WorkTileInfo&) { return false; }

private:
    CUTLASS_DEVICE int32_t lookup_entry(int32_t idx) const
    {
        if (use_reg_list_)
        {
            int32_t const reg_idx = idx >> 5;
            int32_t const src_lane = idx & 31;
            int32_t val = list_regs_[0];
#pragma unroll
            for (int r = 1; r < kListRegs; ++r)
            {
                if (reg_idx == r)
                    val = list_regs_[r];
            }
            return __shfl_sync(0xFFFFFFFFu, val, src_lane);
        }
        int32_t val = 0;
        if ((threadIdx.x & 31) == 0)
            val = params_.block_list[idx];
        return __shfl_sync(0xFFFFFFFFu, val, 0);
    }

    CLCResponse* clc_response_ptr_ = nullptr;
    Params const& params_;
    int32_t live_blocks_ = 0;
    bool sched_init_ = false;
    bool use_reg_list_ = false;
    bool grouped_ = false;
    bool second_round_gate_ = false;
    bool exhausted_ = false;
    bool has_pending_ = false;
    int32_t g1_ = 0;
    int32_t g2_ = 0;
    int64_t d1_ = 0;
    int64_t d_total_ = 0;
    int32_t defer_block_ = -1;
    WorkTileInfo pending_w_{};
    int32_t list_regs_[kListRegs]{};
};

template <class TileShape, class ClusterShape, uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<FsoUnifiedMoeSchedulerTag, arch::Sm100, TileShape, ClusterShape,
    SchedulerPipelineStageCount>
{
    using Scheduler = FsoUnifiedMoeTileScheduler<ClusterShape, SchedulerPipelineStageCount>;
};

} // namespace cutlass::gemm::kernel::detail
