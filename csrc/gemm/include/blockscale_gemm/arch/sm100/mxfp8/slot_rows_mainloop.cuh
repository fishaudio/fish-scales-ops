/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The compact slot route's mainloop with a token load bounded by the tile's live rows (issue #13 lever 3).
//
// What it changes, and why
// ------------------------
// The slot kernels put the routed token rows on the N axis of a 128 x 64 x 128 tile: per k-tile stage the mainloop
// loads one 128-row panel of the expert's weights (A, 16 KiB) and one 64-row panel of the expert's tokens (B, 8 KiB).
// At serving M an expert holds far fewer than 64 rows of the panel -- about 1 to 2 at M = 16, 16 at M = 256 -- and
// the rest of the B box is read anyway: in slot mode the rows between the expert's count and the row capacity m_cap
// (the descriptor's N extent cuts the box at m_cap, so rows past it are zero-filled without a memory access), in
// tile-list mode every one of the 64 rows of the half-block (the token extent per batch is a whole 128-row block).
// Those rows belong to no expert: they were never written in this call, so the first weight tile of every expert
// fetches them from DRAM, and every further weight tile of the expert re-reads them through L2 into shared memory.
//
// This mainloop loads B as ONE box of 8, 16 or 32 rows -- the smallest that covers the tile's live rows -- whenever
// the live rows fit in 32, and as the stock 64-row box otherwise. A box of 8 rows is one 128-byte-swizzle atom of the
// K-major operand (8 rows x 128 bytes, TileK = 128 one-byte elements); the 16- and 32-row boxes are two and four such
// atoms. Each box lands where the stock 64-row box puts those rows (the stage's first 8, 16 or 32 rows, atom j at byte
// j * 1024), so the shared-memory image of every live row is the stock one and the MMA reads it unchanged. One copy
// per stage keeps the TMA issue count of the stock load (a first version issued ceil(live / 8) 8-row boxes and was
// measured 1.5 % slower than the stock load on the layer at M = 1024, where most tiles hold 40 to 64 rows). The rows
// not loaded leave their part of the stage holding whatever an earlier stage put there; those rows feed only the
// accumulator columns of rows that are not the expert's, which no epilogue stores (the compact epilogues are bounded
// by the tile's rows, generator edit c6). The live columns are therefore computed from the same bytes in the same
// order, and the output is the stock kernel's bit for bit.
//
// The stage's byte count. The mainloop pipeline arms each stage's full barrier with the whole tile's transaction
// bytes when the producer acquires it (CUTLASS's PipelineTmaUmmaAsync, a compile-time count). The bytes of the rows
// not loaded are taken off that count with mbarrier.complete_tx right after the acquire, by the same thread and
// before any copy of the stage is issued, so the count never goes below the bytes still in flight and the stage
// completes on the last real copy -- the per-stage byte count has become a runtime quantity without forking the
// pipeline class. The A, SFA and SFB copies are the stock ones.
//
// When it is used. Only where the dead rows are actually read: in tile-list mode and in slot mode at a row capacity
// of 32 or more. Below that the descriptor's m_cap extent already zero-fills most of the box at no memory cost, and
// the stock mainloop is kept. The row-bounded kernels are instantiated in their own translation unit
// (mxfp8_sm100_slot_rows_kernel.cu), so the slot translation unit and the kernels it holds -- the ones every launch
// below a row capacity of 32 runs -- are exactly what they were without this class: a first version that compiled
// the bounded load into the one compact kernel cost 0.5 to 1.3 per cent at M <= 16, where it never fires. Inside a
// bounded kernel a tile with more than 32 live rows takes the stock load.
//
// It is a derived class rather than a copy of CUTLASS's collective: Params gains the 8-, 16- and 32-row TMA atoms
// (built from the same B tensor as the stock one, so their global extents and strides are the stock B's), load_init
// remembers the stage buffer and the problem extents, the load warp hands it each tile's live rows
// (`fso_set_tile_rows`, generator edit c12) and load bounds the token copy by them; everything else -- the mma, the
// TMEM set-up, the shared storage, the stage count -- is inherited unchanged. It reads the collective's protected TMA
// atom pointers, so a CUTLASS bump that renames them fails to compile here instead of changing behaviour.

#pragma once

#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cute/atom/mma_traits_sm100.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/kernel_hardware_info.hpp"

#include <cstdint>

namespace sm100_blockscaled_gemm
{
namespace slot_detail
{

template <class Base>
struct FsoRowsBoundedMainloop : public Base
{
    using TileShape = typename Base::TileShape;
    using StrideB = typename Base::StrideB;
    using ElemB = typename Base::TmaInternalElementB;
    using SmemAllocB = typename Base::SmemAllocTypeB;
    using MainloopPipeline = typename Base::MainloopPipeline;
    using MainloopPipelineState = typename Base::MainloopPipelineState;
    // Tells the generated kernel's `fso_set_load_rows` (edit c12) to hand this mainloop each tile's live rows.
    static constexpr bool kFsoRowsBounded = true;

    static constexpr int kTileN = cute::size<1>(TileShape{});
    static constexpr int kTileK = cute::size<2>(TileShape{});
    // One SW128 atom of the K-major 8-bit operand: 8 rows of TileK = 128 one-byte elements; the boxes are 1, 2 or 4
    // of them stacked along the rows, exactly as the stock stage stacks its 8.
    static constexpr int kRowBox = 8;
    using SmemLayoutB8 = decltype(cute::UMMA::Layout_K_SW128_Atom<ElemB>{});
    template <int Rows>
    using SmemLayoutBox = decltype(cute::tile_to_shape(SmemLayoutB8{}, cute::Shape<cute::Int<Rows>, cute::Int<kTileK>>{}));
    static constexpr int kTileBBytes = kTileN * kTileK;
    // Elements (bytes) of one stage of B.
    static constexpr int kStageBElems = int(cute::cosize(cute::take<0, 3>(typename Base::SmemLayoutB{})));
    static_assert(cute::sizeof_bits_v<ElemB> == 8 && cute::sizeof_bits_v<SmemAllocB> == 8,
        "the row-bounded token load is written for a one-byte B operand");
    static_assert(kTileK == 128 && cute::size<0>(SmemLayoutB8{}) == kRowBox && cute::size<1>(SmemLayoutB8{}) == kTileK,
        "TileK 128 of a one-byte operand is exactly one 128-byte-swizzle atom per row: one box = one atom");
    static_assert(kTileN % kRowBox == 0 && kStageBElems == kTileN * kTileK,
        "the stage of B must be the TileN rows of TileK bytes, stacked one 8-row atom after the other");
    static_assert(cute::size(typename Base::AtomThrShapeMNK{}) == 1, "1-SM tiles only (cluster 1x1x1)");

    template <int Rows>
    using TmaBox = decltype(cute::make_tma_atom<ElemB>(cute::SM90_TMA_LOAD{},
        cute::make_tensor(cute::recast_ptr<ElemB>(nullptr), cute::repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutBox<Rows>{}, cute::make_shape(cute::Int<Rows>{}, cute::Int<kTileK>{})));
    static_assert(kTileN == 64, "the boxes are 8, 16 and 32 rows of a 64-row token tile");

    template <int Rows, class TensorB>
    static auto make_box(TensorB const& tensor_b)
    {
        return cute::make_tma_atom<ElemB>(cute::SM90_TMA_LOAD{}, tensor_b, SmemLayoutBox<Rows>{},
            cute::make_shape(cute::Int<Rows>{}, cute::Int<kTileK>{}));
    }

    struct Params : Base::Params
    {
        TmaBox<8> tma_load_b8;
        TmaBox<16> tma_load_b16;
        TmaBox<32> tma_load_b32;
    };

    template <class ProblemShape>
    static Params to_underlying_arguments(ProblemShape const& problem_shape, typename Base::Arguments const& args,
        void* workspace, cutlass::KernelHardwareInfo const& hw_info = cutlass::KernelHardwareInfo{})
    {
        auto problem_shape_MNKL = cute::append<4>(problem_shape, 1);
        auto [M, N, K, L] = problem_shape_MNKL;
        auto tensor_b
            = cute::make_tensor(cute::recast_ptr<ElemB>(args.ptr_B), cute::make_layout(cute::make_shape(N, K, L), args.dB));
        return Params{Base::to_underlying_arguments(problem_shape, args, workspace, hw_info), make_box<8>(tensor_b),
            make_box<16>(tensor_b), make_box<32>(tensor_b)};
    }

    template <class ClusterShapeT>
    CUTLASS_DEVICE FsoRowsBoundedMainloop(Params const& params, ClusterShapeT cluster_shape, uint32_t block_rank)
        : Base(params, cluster_shape, block_rank)
        , box_params_(&params)
    {
    }

    CUTLASS_DEVICE void prefetch_tma_descriptors()
    {
        Base::prefetch_tma_descriptors();
        cute::prefetch_tma_descriptor(box_params_->tma_load_b8.get_tma_descriptor());
        cute::prefetch_tma_descriptor(box_params_->tma_load_b16.get_tma_descriptor());
        cute::prefetch_tma_descriptor(box_params_->tma_load_b32.get_tma_descriptor());
    }

    // The stock partitioning, plus what the bounded load needs: the stage buffer of B and the problem's extents.
    template <class ProblemShape_MNKL>
    CUTLASS_DEVICE auto load_init(ProblemShape_MNKL const& problem_shape_MNKL,
        typename Base::TensorStorage& shared_tensors) const
    {
        smem_b_ = shared_tensors.smem_B.begin();
        n_extent_ = int(cute::get<1>(problem_shape_MNKL));
        k_extent_ = int(cute::get<2>(problem_shape_MNKL));
        l_extent_ = int(cute::get<3>(problem_shape_MNKL));
        return Base::load_init(problem_shape_MNKL, shared_tensors);
    }

    // The load warp, before each live tile's two load calls: the number of the tile's rows (counted from its N
    // tile's first row) that hold the expert's tokens, 1 to TileN.
    CUTLASS_DEVICE void fso_set_tile_rows(int live_rows)
    {
        live_rows_ = live_rows;
    }

    // The stock load when the tile has more than 32 live rows; otherwise the bounded one (see the header).
    template <class LoadParams, class TileCoordMNKL, class KTileIterator>
    CUTLASS_DEVICE auto load(MainloopPipeline mainloop_pipeline, MainloopPipelineState mainloop_pipe_producer_state,
        LoadParams const& load_inputs, TileCoordMNKL const& cta_coord_mnkl, KTileIterator k_tile_iter, int k_tile_count)
    {
        int const live_rows = live_rows_;
        if (live_rows > 32)
            return Base::load(
                mainloop_pipeline, mainloop_pipe_producer_state, load_inputs, cta_coord_mnkl, k_tile_iter, k_tile_count);
        if (live_rows > 16)
            return load_box<32>(box_params_->tma_load_b32, mainloop_pipeline, mainloop_pipe_producer_state, load_inputs,
                cta_coord_mnkl, k_tile_iter, k_tile_count);
        if (live_rows > 8)
            return load_box<16>(box_params_->tma_load_b16, mainloop_pipeline, mainloop_pipe_producer_state, load_inputs,
                cta_coord_mnkl, k_tile_iter, k_tile_count);
        return load_box<8>(box_params_->tma_load_b8, mainloop_pipeline, mainloop_pipe_producer_state, load_inputs,
            cta_coord_mnkl, k_tile_iter, k_tile_count);
    }

protected:
    // (Protected rather than private so that the fused FC1 + FC2 kernel's mainloop, FsoMoeMainloop in
    // slot_moe_mainloop.cuh, can issue the same boxes in its split weights / activation load; an access specifier
    // changes no code.)
    // The stock load with B as one `Rows`-row box at the tile's first row: the A, SFA and SFB copies are the stock
    // ones, and the bytes of the TileN - Rows rows not loaded come off the stage's byte count before any copy of the
    // stage is issued.
    template <int Rows, class TmaAtom, class LoadParams, class TileCoordMNKL, class KTileIterator>
    CUTLASS_DEVICE auto load_box(TmaAtom const& tma_box, MainloopPipeline mainloop_pipeline,
        MainloopPipelineState mainloop_pipe_producer_state, LoadParams const& load_inputs,
        TileCoordMNKL const& cta_coord_mnkl, KTileIterator k_tile_iter, int k_tile_count)
    {
        using namespace cute;
        auto [unused_k_tiles, tAgA_mkl, tBgB_nkl, tAsA, tBsB, tAgSFA_mkl, tBgSFB_nkl, tAsSFA, tBsSFB, mcast_mask_a,
            mcast_mask_b, mcast_mask_sfa, mcast_mask_sfb]
            = load_inputs;

        Tensor tAgA = tAgA_mkl(_, get<0>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));
        Tensor tAgSFA = tAgSFA_mkl(_, get<0>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));
        Tensor tBgSFB = tBgSFB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

        // B of this batch in `Rows`-row boxes: (Rows, TileK, N / Rows, K / TileK); the tile's first box is its N tile
        // times TileN / Rows.
        Tensor mB = tma_box.get_tma_tensor(make_shape(n_extent_, k_extent_, l_extent_));
        Tensor gB = local_tile(mB(_, _, get<3>(cta_coord_mnkl)), make_shape(Int<Rows>{}, Int<kTileK>{}), make_coord(_, _));
        int const box = int(get<1>(cta_coord_mnkl)) * (kTileN / Rows);
        constexpr uint32_t kSkipped = static_cast<uint32_t>(kTileN - Rows) * kTileK;

        auto barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

        CUTLASS_PRAGMA_NO_UNROLL
        while (k_tile_count > 0)
        {
            mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state, barrier_token);

            using BarrierType = typename MainloopPipeline::ProducerBarrierType;
            BarrierType* tma_barrier = mainloop_pipeline.producer_get_barrier(mainloop_pipe_producer_state);

            int write_stage = mainloop_pipe_producer_state.index();
            ++mainloop_pipe_producer_state;
            barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

            if (cute::elect_one_sync())
            {
                // The rows not loaded come off the stage's byte count before any of its copies is issued.
                cutlass::arch::ClusterTransactionBarrier::complete_transaction(
                    tma_barrier, cute::block_rank_in_cluster(), kSkipped);
                copy(this->observed_tma_load_a_->with(*tma_barrier, mcast_mask_a), tAgA(_, *k_tile_iter),
                    tAsA(_, write_stage));
                Tensor sB = make_tensor(make_smem_ptr(smem_b_ + write_stage * kStageBElems), SmemLayoutBox<Rows>{});
                auto [tBgBox, tBsBox] = tma_partition(tma_box, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB),
                    group_modes<0, 2>(gB(_, _, box, *k_tile_iter)));
                copy(tma_box.with(*tma_barrier), tBgBox, tBsBox);
                copy(this->observed_tma_load_sfa_->with(*tma_barrier, mcast_mask_sfa), tAgSFA(_, *k_tile_iter),
                    tAsSFA(_, write_stage));
                copy(this->observed_tma_load_sfb_->with(*tma_barrier, mcast_mask_sfb), tBgSFB(_, *k_tile_iter),
                    tBsSFB(_, write_stage));
            }

            --k_tile_count;
            ++k_tile_iter;
        }

        return cute::make_tuple(mainloop_pipe_producer_state, k_tile_iter);
    }

    Params const* box_params_ = nullptr;
    int live_rows_ = kTileN;
    mutable SmemAllocB* smem_b_ = nullptr;
    mutable int n_extent_ = 0;
    mutable int k_extent_ = 0;
    mutable int l_extent_ = 0;
};

} // namespace slot_detail
} // namespace sm100_blockscaled_gemm
