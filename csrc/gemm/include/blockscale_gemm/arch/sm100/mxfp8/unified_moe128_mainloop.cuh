/*
 * Unified SM100/SM103 128 x 128 x 128 persistent FC1 + FC2 MoE mainloop (v2).
 *
 * Combines:
 *   - 8 / 16 / 32 / 64 / 128 row-bounded TMA activation loading (selected on device)
 *   - Hybrid FC2 weights-first overlap (`fso_load_after_entry`): issues all K2
 *     weight stages (K2=512 -> 4 stages <= 6 ring stages!) before spinning on
 *     the block's FC1 completion counter `done[b] >= T1`.
 */

#pragma once

#include "blockscale_gemm/arch/sm100/mxfp8/slot_moe_scheduler.cuh"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cute/atom/mma_traits_sm100.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/kernel_hardware_info.hpp"

#include <cstdint>

namespace sm100_blockscaled_gemm
{
namespace unified_moe128_detail
{

template <class Base>
struct FsoMoe128Mainloop : public Base
{
    using TileShape = typename Base::TileShape;
    using StrideB = typename Base::StrideB;
    using ElemB = typename Base::TmaInternalElementB;
    using SmemAllocB = typename Base::SmemAllocTypeB;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementSF = typename Base::ElementSF;
    using MainloopPipeline = typename Base::MainloopPipeline;
    using MainloopPipelineState = typename Base::MainloopPipelineState;

    static constexpr bool kFsoRowsBounded = true;
    static constexpr int kStages = MainloopPipeline::Stages;
    static constexpr int kTileM = cute::size<0>(TileShape{});
    static constexpr int kTileN = cute::size<1>(TileShape{});
    static constexpr int kTileK = cute::size<2>(TileShape{});
    static constexpr int kStageBElems = int(cute::cosize(cute::take<0, 3>(typename Base::SmemLayoutB{})));

    static constexpr uint32_t kBytesA = cutlass::bits_to_bytes(
        cute::cosize(cute::take<0, 3>(typename Base::SmemLayoutA{})) * cute::sizeof_bits_v<ElementA>);
    static constexpr uint32_t kBytesB = cutlass::bits_to_bytes(
        cute::cosize(cute::take<0, 3>(typename Base::SmemLayoutB{})) * cute::sizeof_bits_v<ElementB>);
    static constexpr uint32_t kBytesSFA = cutlass::bits_to_bytes(
        cute::cosize(cute::take<0, 3>(typename Base::SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSF>);
    static constexpr uint32_t kBytesSFB = cutlass::bits_to_bytes(
        cute::cosize(cute::take<0, 3>(typename Base::SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSF>);
    static constexpr uint32_t kWeightBytes = kBytesA + kBytesSFA;

    using SmemLayoutB8 = decltype(cute::UMMA::Layout_K_SW128_Atom<ElemB>{});
    template <int Rows>
    using SmemLayoutBox = decltype(cute::tile_to_shape(SmemLayoutB8{}, cute::Shape<cute::Int<Rows>, cute::Int<kTileK>>{}));

    static_assert(kTileM == 128 && kTileN == 128 && kTileK == 128, "FsoMoe128Mainloop requires 128x128x128 tile");
    static_assert(kBytesA + kBytesB + kBytesSFA + kBytesSFB == Base::TmaTransactionBytes,
        "stage copy bytes must match Base::TmaTransactionBytes");
    static_assert(kStageBElems == kTileN * kTileK, "stage B size mismatch");
    static_assert(cute::size(typename Base::AtomThrShapeMNK{}) == 1, "1-SM cluster 1x1x1 required");

    template <int Rows>
    using TmaBox = decltype(cute::make_tma_atom<ElemB>(cute::SM90_TMA_LOAD{},
        cute::make_tensor(cute::recast_ptr<ElemB>(nullptr), cute::repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutBox<Rows>{}, cute::make_shape(cute::Int<Rows>{}, cute::Int<kTileK>{})));

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
        TmaBox<64> tma_load_b64;
    };

    template <class ProblemShape>
    static Params to_underlying_arguments(ProblemShape const& problem_shape, typename Base::Arguments const& args,
        void* workspace, cutlass::KernelHardwareInfo const& hw_info = cutlass::KernelHardwareInfo{})
    {
        auto problem_shape_MNKL = cute::append<4>(problem_shape, 1);
        auto [M, N, K, L] = problem_shape_MNKL;
        auto tensor_b = cute::make_tensor(
            cute::recast_ptr<ElemB>(args.ptr_B), cute::make_layout(cute::make_shape(N, K, L), args.dB));
        return Params{Base::to_underlying_arguments(problem_shape, args, workspace, hw_info), make_box<8>(tensor_b),
            make_box<16>(tensor_b), make_box<32>(tensor_b), make_box<64>(tensor_b)};
    }

    template <class ClusterShapeT>
    CUTLASS_DEVICE FsoMoe128Mainloop(Params const& params, ClusterShapeT cluster_shape, uint32_t block_rank)
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
        cute::prefetch_tma_descriptor(box_params_->tma_load_b64.get_tma_descriptor());
    }

    template <class ProblemShape_MNKL>
    CUTLASS_DEVICE auto load_init(
        ProblemShape_MNKL const& problem_shape_MNKL, typename Base::TensorStorage& shared_tensors) const
    {
        smem_b_ = shared_tensors.smem_B.begin();
        n_extent_ = int(cute::get<1>(problem_shape_MNKL));
        k_extent_ = int(cute::get<2>(problem_shape_MNKL));
        l_extent_ = int(cute::get<3>(problem_shape_MNKL));
        return Base::load_init(problem_shape_MNKL, shared_tensors);
    }

    CUTLASS_DEVICE void fso_set_tile_rows(int rows) { live_rows_ = rows; }

    template <class LoadParams, class TileCoordMNKL, class KTileIterator>
    CUTLASS_DEVICE auto load(MainloopPipeline mainloop_pipeline, MainloopPipelineState mainloop_pipe_producer_state,
        LoadParams const& load_inputs, TileCoordMNKL const& cta_coord_mnkl, KTileIterator k_tile_iter, int k_tile_count)
    {
        int const lr = live_rows_;
        if (lr > 64)
            return Base::load(mainloop_pipeline, mainloop_pipe_producer_state, load_inputs, cta_coord_mnkl,
                k_tile_iter, k_tile_count);
        if (lr > 32)
            return load_box<64>(box_params_->tma_load_b64, mainloop_pipeline, mainloop_pipe_producer_state,
                load_inputs, cta_coord_mnkl, k_tile_iter, k_tile_count);
        if (lr > 16)
            return load_box<32>(box_params_->tma_load_b32, mainloop_pipeline, mainloop_pipe_producer_state,
                load_inputs, cta_coord_mnkl, k_tile_iter, k_tile_count);
        if (lr > 8)
            return load_box<16>(box_params_->tma_load_b16, mainloop_pipeline, mainloop_pipe_producer_state,
                load_inputs, cta_coord_mnkl, k_tile_iter, k_tile_count);
        return load_box<8>(box_params_->tma_load_b8, mainloop_pipeline, mainloop_pipe_producer_state, load_inputs,
            cta_coord_mnkl, k_tile_iter, k_tile_count);
    }

    // Matches FsoSm100SlotGemmMoe's exact call order:
    // (pipeline, state, load_inputs, cta_coord_mnkl, k_tile_iter, k_tile_count, ready, fence, done, target).
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
            if (cute::elect_one_sync())
            {
                while (cutlass::gemm::kernel::detail::fso_ld_acquire_gpu(done) < target)
                {
                }
            }
            __syncwarp();
        }
        if (fence)
            cutlass::gemm::kernel::detail::fso_fence_proxy_async_global();
        return load(mainloop_pipeline, mainloop_pipe_producer_state, load_inputs, cta_coord_mnkl, k_tile_iter,
            k_tile_count);
    }

private:
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

        // 1. Stream all FC2 weight stages before waiting on FC1 completion.
        CUTLASS_PRAGMA_NO_UNROLL
        for (int i = 0; i < n_pre; ++i)
        {
            while (!(mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state)
                == cutlass::BarrierStatus::WaitDone))
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

        // 2. Spin on block done, fence async proxy, then issue row-bounded activation boxes.
        int const lr = live_rows_;
        if (cute::elect_one_sync())
        {
            while (cutlass::gemm::kernel::detail::fso_ld_acquire_gpu(done) < target)
            {
            }
            cutlass::gemm::kernel::detail::fso_fence_proxy_async_global();
            MainloopPipelineState st = first_state;
            KTileIterator kt = first_k;
            CUTLASS_PRAGMA_NO_UNROLL
            for (int i = 0; i < n_pre; ++i)
            {
                BarrierType* tma_barrier = mainloop_pipeline.producer_get_barrier(st);
                int const write_stage = st.index();
                if (lr > 64)
                {
                    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(tma_barrier, kBytesB + kBytesSFB);
                    copy(this->observed_tma_load_b_->with(*tma_barrier, mcast_mask_b), tBgB(_, *kt),
                        tBsB(_, write_stage));
                }
                else if (lr > 32)
                    issue_box<64>(box_params_->tma_load_b64, tma_barrier, write_stage, cta_coord_mnkl, kt);
                else if (lr > 16)
                    issue_box<32>(box_params_->tma_load_b32, tma_barrier, write_stage, cta_coord_mnkl, kt);
                else if (lr > 8)
                    issue_box<16>(box_params_->tma_load_b16, tma_barrier, write_stage, cta_coord_mnkl, kt);
                else
                    issue_box<8>(box_params_->tma_load_b8, tma_barrier, write_stage, cta_coord_mnkl, kt);
                copy(this->observed_tma_load_sfb_->with(*tma_barrier, mcast_mask_sfb), tBgSFB(_, *kt),
                    tBsSFB(_, write_stage));
                ++st;
                ++kt;
            }
        }
        __syncwarp();
        return cute::make_tuple(mainloop_pipe_producer_state, k_tile_iter);
    }

    template <int Rows, class TmaBoxT, class TileCoordMNKL, class KTileIterator>
    CUTLASS_DEVICE void issue_box(TmaBoxT const& tma_box, typename MainloopPipeline::ProducerBarrierType* tma_barrier,
        int write_stage, TileCoordMNKL const& cta_coord_mnkl, KTileIterator const& kt)
    {
        using namespace cute;
        Tensor mB = tma_box.get_tma_tensor(make_shape(n_extent_, k_extent_, l_extent_));
        Tensor gB = local_tile(
            mB(_, _, get<3>(cta_coord_mnkl)), make_shape(Int<Rows>{}, Int<kTileK>{}), make_coord(_, _));
        int const box = int(get<1>(cta_coord_mnkl)) * (kTileN / Rows);
        Tensor sB = make_tensor(make_smem_ptr(smem_b_ + write_stage * kStageBElems), SmemLayoutBox<Rows>{});
        auto [tBgBox, tBsBox] = tma_partition(
            tma_box, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB), group_modes<0, 2>(gB(_, _, box, *kt)));
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
            tma_barrier, static_cast<uint32_t>(Rows * kTileK) + kBytesSFB);
        copy(tma_box.with(*tma_barrier), tBgBox, tBsBox);
    }

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

        Tensor mB = tma_box.get_tma_tensor(make_shape(n_extent_, k_extent_, l_extent_));
        Tensor gB = local_tile(
            mB(_, _, get<3>(cta_coord_mnkl)), make_shape(Int<Rows>{}, Int<kTileK>{}), make_coord(_, _));
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

} // namespace unified_moe128_detail
} // namespace sm100_blockscaled_gemm
