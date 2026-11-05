/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The mainloop of the sm_100 / sm_103 persistent FC1 + FC2 slot kernel (issue #13 lever 5, `FsoSm100SlotGemmMoe`,
// run fso_issue13_20261031/I13-L5): the row-bounded compact slot mainloop (slot_rows_mainloop.cuh, lever 3) plus a
// load for the FC2 tiles that waits on the FC1 tiles of its entry.
//
// Why a second load
// -----------------
// An FC2 tile reads two kinds of bytes per stage: the expert's FC2 weights and their scales (A, SFA), which exist
// before the kernel starts, and the entry's FC1 output and its scale words (B, SFB), which the FC1 tiles of the same
// entry write in this kernel. The tile scheduler checks once, when it draws an FC2 tile, whether every FC1 tile of
// the entry has released its count (slot_moe_scheduler.cuh, "the hand-off"), and hands the tile out with that
// verdict. `fso_load_after_entry` then takes one of two paths:
//
//   * ready: `fence.proxy.async.global` and the ordinary load. The fence is needed because the TMA reads through the
//     async proxy what the scheduler warp's acquire ordered in the generic proxy.
//   * not ready: the weights and weight scales of every stage of the tile first, then this warp spins on the entry's
//     done counter (`ld.acquire.gpu`) until it reaches the FC1 tile count, fences, and only then issues the
//     activation and its scales into the same stages. An FC2 tile that starts while FC1 is still running thereby
//     streams its weights at once instead of idling until FC1's last store. (A tile with more k-tiles than the ring
//     has stages -- none of the served FC2 shapes, whose K2 / 128 is 6 or 4 against 8 stages -- waits first and then
//     loads as a ready one.)
//
// The kernel calls this for the FC1 tiles too, with the ready bit forced on and no fence: both phases then share ONE
// inlined copy of the row-bounded load, which keeps the kernel's code (and its instruction-cache footprint) close to
// today's FC1 kernel's.
//
// The stage byte counts of the second path. The stock pipeline's `producer_acquire` arrives on the stage's full
// barrier with the whole tile's transaction bytes, which would let the stage complete as soon as the weights landed
// if the activation were not also counted. The weights-first path therefore never calls it: it waits for the stage to
// be free (`producer_try_acquire`, no arrival), arms the weight bytes with `mbarrier.expect_tx` WITHOUT an arrival
// (so the barrier's one producer arrival is still pending and the phase cannot complete), and after the spin arrives
// with exactly the activation bytes it issues (`mbarrier.arrive.expect_tx`): the stock 64-row box, or the row-bounded
// box of 8, 16 or 32 rows of lever 3. No `complete_tx` is needed on these stages, because only the bytes actually
// loaded are ever armed. The MMA warp waits on the full barrier as for any stage.
//
// Everything else -- the mma, the TMEM set-up, the shared storage, the stage count -- is the row-bounded mainloop's,
// unchanged; this class adds no data member.

#pragma once

#include "blockscale_gemm/arch/sm100/mxfp8/slot_moe_scheduler.cuh"
#include "blockscale_gemm/arch/sm100/mxfp8/slot_rows_mainloop.cuh"

#include <cstdint>

namespace sm100_blockscaled_gemm
{
namespace slot_detail
{

template <class Base>
struct FsoMoeMainloop : public Base
{
    using MainloopPipeline = typename Base::MainloopPipeline;
    using MainloopPipelineState = typename Base::MainloopPipelineState;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementSF = typename Base::ElementSF;
    using Params = typename Base::Params;

    static constexpr int kStages = MainloopPipeline::Stages;
    static constexpr uint32_t kBytesA = cutlass::bits_to_bytes(
        cute::cosize(cute::take<0, 3>(typename Base::SmemLayoutA{})) * cute::sizeof_bits_v<ElementA>);
    static constexpr uint32_t kBytesB = cutlass::bits_to_bytes(
        cute::cosize(cute::take<0, 3>(typename Base::SmemLayoutB{})) * cute::sizeof_bits_v<ElementB>);
    static constexpr uint32_t kBytesSFA = cutlass::bits_to_bytes(
        cute::cosize(cute::take<0, 3>(typename Base::SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSF>);
    static constexpr uint32_t kBytesSFB = cutlass::bits_to_bytes(
        cute::cosize(cute::take<0, 3>(typename Base::SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSF>);
    // The weight half of a stage, armed before the spin.
    static constexpr uint32_t kWeightBytes = kBytesA + kBytesSFA;
    static_assert(kBytesA + kBytesB + kBytesSFA + kBytesSFB == Base::TmaTransactionBytes,
        "the four copies of a stage must account for exactly the transaction bytes the stock pipeline arms");
    static_assert(kBytesB == static_cast<uint32_t>(Base::kTileN * Base::kTileK),
        "the activation stage is the TileN rows of TileK one-byte elements");

    template <class ClusterShapeT>
    CUTLASS_DEVICE FsoMoeMainloop(Params const& params, ClusterShapeT cluster_shape, uint32_t block_rank)
        : Base(params, cluster_shape, block_rank)
    {
    }

    // One tile of either phase (see the header). `ready` false (an FC2 tile whose entry's FC1 tiles had not all
    // released when the scheduler drew it) takes the weights-first path while the tile's k-tiles fit the ring, and
    // otherwise waits on `done` first; `fence` (FC2 tiles) puts the proxy fence between the hand-off and the TMA. Every
    // other case is the row-bounded load, from ONE call site, so the kernel holds one inlined copy of it.
    template <class LoadParams, class TileCoordMNKL, class KTileIterator>
    CUTLASS_DEVICE auto fso_load_after_entry(MainloopPipeline mainloop_pipeline,
        MainloopPipelineState mainloop_pipe_producer_state, LoadParams const& load_inputs,
        TileCoordMNKL const& cta_coord_mnkl, KTileIterator k_tile_iter, int k_tile_count, bool ready, bool fence,
        int const* done, uint32_t target)
    {
        using namespace cute;
        if (!ready && k_tile_count <= kStages)
            return load_weights_first(mainloop_pipeline, mainloop_pipe_producer_state, load_inputs, cta_coord_mnkl,
                k_tile_iter, k_tile_count, done, target);
        if (!ready)
        {
            // More k-tiles than stages: wait for the entry, then load as for any tile (no FC2 of the served shapes
            // takes this path: K2 / 128 <= the ring's stages).
            if (cute::elect_one_sync())
            {
#if !defined(FSO_SLOT_MOE_NEGATIVE_CONTROL_NO_ACQUIRE)
                while (cutlass::gemm::kernel::detail::fso_ld_acquire_gpu(done) < target)
                {
                }
#endif
            }
            __syncwarp();
        }
        if (fence)
        {
            // Every lane: the lane `load` elects to issue the copies is among them.
            cutlass::gemm::kernel::detail::fso_fence_proxy_async_global();
        }
        return this->load(mainloop_pipeline, mainloop_pipe_producer_state, load_inputs, cta_coord_mnkl, k_tile_iter,
            k_tile_count);
    }

private:
    // The weights-first path: all k-tiles of the tile fit the ring (k_tile_count <= kStages).
    template <class LoadParams, class TileCoordMNKL, class KTileIterator>
    CUTLASS_DEVICE auto load_weights_first(MainloopPipeline mainloop_pipeline,
        MainloopPipelineState mainloop_pipe_producer_state, LoadParams const& load_inputs,
        TileCoordMNKL const& cta_coord_mnkl, KTileIterator k_tile_iter, int k_tile_count, int const* done,
        uint32_t target)
    {
        using namespace cute;
        auto [unused_k_tiles, tAgA_mkl, tBgB_nkl, tAsA, tBsB, tAgSFA_mkl, tBgSFB_nkl, tAsSFA, tBsSFB, mcast_mask_a,
            mcast_mask_b, mcast_mask_sfa, mcast_mask_sfb]
            = load_inputs;
        Tensor tAgA = tAgA_mkl(_, get<0>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));
        Tensor tBgB = tBgB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));
        Tensor tAgSFA = tAgSFA_mkl(_, get<0>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));
        Tensor tBgSFB = tBgSFB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;

        int const n_pre = k_tile_count;
        MainloopPipelineState const first_state = mainloop_pipe_producer_state;
        KTileIterator const first_k = k_tile_iter;

        // 1. The weights of every stage of the tile: wait for each stage to be free (no arrival), arm the weight bytes
        //    without an arrival, issue A and SFA.
        CUTLASS_PRAGMA_NO_UNROLL
        for (int i = 0; i < n_pre; ++i)
        {
            while (!(mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state) == cutlass::BarrierStatus::WaitDone))
            {
            }
            BarrierType* tma_barrier = mainloop_pipeline.producer_get_barrier(mainloop_pipe_producer_state);
            int const write_stage = mainloop_pipe_producer_state.index();
            if (cute::elect_one_sync())
            {
                cutlass::arch::ClusterTransactionBarrier::expect_transaction(tma_barrier, kWeightBytes);
                copy(this->observed_tma_load_a_->with(*tma_barrier, mcast_mask_a), tAgA(_, *k_tile_iter),
                    tAsA(_, write_stage));
                copy(this->observed_tma_load_sfa_->with(*tma_barrier, mcast_mask_sfa), tAgSFA(_, *k_tile_iter),
                    tAsSFA(_, write_stage));
            }
            ++mainloop_pipe_producer_state;
            ++k_tile_iter;
        }

        // 2. The hand-off, then the activation of the same stages. One lane spins, fences and issues, so the lane
        //    whose acquire ordered the FC1 stores is the lane whose TMA reads them.
        int const live_rows = this->live_rows_;
        if (cute::elect_one_sync())
        {
#if !defined(FSO_SLOT_MOE_NEGATIVE_CONTROL_NO_ACQUIRE)
            while (cutlass::gemm::kernel::detail::fso_ld_acquire_gpu(done) < target)
            {
            }
#else
            (void) done;
            (void) target;
#endif
            cutlass::gemm::kernel::detail::fso_fence_proxy_async_global();
            MainloopPipelineState st = first_state;
            KTileIterator kt = first_k;
            CUTLASS_PRAGMA_NO_UNROLL
            for (int i = 0; i < n_pre; ++i)
            {
                BarrierType* tma_barrier = mainloop_pipeline.producer_get_barrier(st);
                int const write_stage = st.index();
                if (live_rows > 32)
                {
                    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(tma_barrier, kBytesB + kBytesSFB);
                    copy(this->observed_tma_load_b_->with(*tma_barrier, mcast_mask_b), tBgB(_, *kt),
                        tBsB(_, write_stage));
                }
                else if (live_rows > 16)
                    issue_box<32>(this->box_params_->tma_load_b32, tma_barrier, write_stage, cta_coord_mnkl, kt);
                else if (live_rows > 8)
                    issue_box<16>(this->box_params_->tma_load_b16, tma_barrier, write_stage, cta_coord_mnkl, kt);
                else
                    issue_box<8>(this->box_params_->tma_load_b8, tma_barrier, write_stage, cta_coord_mnkl, kt);
                copy(this->observed_tma_load_sfb_->with(*tma_barrier, mcast_mask_sfb), tBgSFB(_, *kt),
                    tBsSFB(_, write_stage));
                ++st;
                ++kt;
            }
        }
        __syncwarp();
        return cute::make_tuple(mainloop_pipe_producer_state, k_tile_iter);
    }

    // The activation box of `Rows` rows of the tile's first row into stage `write_stage`, with exactly its bytes and
    // the SFB copy's armed by the stage's one producer arrival (slot_rows_mainloop.cuh's `load_box`, split off).
    template <int Rows, class TmaBox, class TileCoordMNKL, class KTileIterator>
    CUTLASS_DEVICE void issue_box(TmaBox const& tma_box, typename MainloopPipeline::ProducerBarrierType* tma_barrier,
        int write_stage, TileCoordMNKL const& cta_coord_mnkl, KTileIterator const& kt)
    {
        using namespace cute;
        Tensor mB = tma_box.get_tma_tensor(make_shape(this->n_extent_, this->k_extent_, this->l_extent_));
        Tensor gB = local_tile(
            mB(_, _, get<3>(cta_coord_mnkl)), make_shape(Int<Rows>{}, Int<Base::kTileK>{}), make_coord(_, _));
        int const box = int(get<1>(cta_coord_mnkl)) * (Base::kTileN / Rows);
        Tensor sB = make_tensor(make_smem_ptr(this->smem_b_ + write_stage * Base::kStageBElems),
            typename Base::template SmemLayoutBox<Rows>{});
        auto [tBgBox, tBsBox] = tma_partition(
            tma_box, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB), group_modes<0, 2>(gB(_, _, box, *kt)));
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
            tma_barrier, static_cast<uint32_t>(Rows * Base::kTileK) + kBytesSFB);
        copy(tma_box.with(*tma_barrier), tBgBox, tBsBox);
    }
};

} // namespace slot_detail
} // namespace sm100_blockscaled_gemm
