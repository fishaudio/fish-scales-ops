/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// E28 (2026-05-11): Forced-only experimental small-M MXFP8 kernel variant.
//
// **NEGATIVE RESULT — DO NOT REATTEMPT.** Kept in-tree as forced-only
// reference. See `docs/skills/blockscale-gemm-tuning/references/
// sm120-mxfp8-smallm-manual-a-negative.md` for the full analysis +
// measured A/B table. Summary: manual cp.async.bulk for A is 2-13×
// slower than TMA across M ∈ [1, 16] on the validated cells because
// the SW128 swizzle forces 8 × M_actual 16-byte cp.async.bulks per
// stage, vs 1 TMA. The OOB-row cost the hypothesis targeted is
// essentially free in the TMA engine; the cp.async.bulk issue cost
// is not. Default dispatch is unchanged.
//
// Mirror of `SM120BlockScaledKernel<KT>` (fp8/gemm_1d1d.cuh) with two
// modifications:
//   1. Params: drop `TMA_A` + add `ElementA* ptr_A, int64_t ld_a, stride_a`.
//   2. load_ab: replace A's TMA copy with one `SM90_BULK_COPY_G2S` per
//      16-byte chunk × 8 chunks per row × M_actual rows per stage.
//      Smem rows m ∈ [M_in_tile, TileM) are left uninitialised — safe
//      because SFA's TMA-zero-fill (UE8M0=0 → 2^-127) reduces their MMA
//      contribution to ~10^-37 per element, well below float32 ulp.
//      Verified correctness (cos ≥ 0.99927) and bit-exact CUDA Graph
//      replay across M ∈ {1, 2, 4, 8, 16} for wqkv/gate/gate_up shapes.
//
// Scope: sm_120 / sm_121 only, MXFP8, M ≤ 16, K % 128 == 0, batched
// L=1. Forced-only via `BSGEMM_FORCE_SMALLM=1` + `BSGEMM_FORCE_TILE=
// 16,128,4`. Not on the default route.

#pragma once

#include "blockscale_gemm/arch/sm120/mxfp8/utils.cuh"
#include "blockscale_gemm/arch/sm120/fp8/utils.cuh"  // SM120BlockScaledScheduler

#include "cute/arch/copy_sm90_tma.hpp"  // SM90_BULK_COPY_G2S

using namespace cute;

namespace sm120_blockscaled_gemm
{

template <typename KT>
struct SM120MxFP8SmallMKernel
{
    static constexpr int kNumTMAThreads = 128;
    static constexpr int kNumMathThreads = KT::kNumMathThreads;
    static constexpr int MaxThreadsPerBlock = kNumTMAThreads + kNumMathThreads;
    static constexpr int MinBlocksPerMultiprocessor = KT::MinBlocksPerSm;

    using ProblemShape = typename KT::ProblemShape;
    using ElementA = typename KT::ElementA;
    using ElementSFLoad = typename KT::ElementSFLoad;
    using ElementD = typename KT::ElementD;

    struct Params
    {
        // A: raw gmem pointer + row stride (bytes-per-row / sizeof(ElementA))
        ElementA* ptr_A;
        int64_t ld_a;                                   // elements between rows of A
        int64_t stride_a;                               // batch stride (elements)
        typename KT::TMA_B tma_load_b;
        typename KT::TMA_SFA tma_load_sfa;
        typename KT::TMA_SFB tma_load_sfb;
        typename KT::TMA_D tma_store_d;
        typename KT::ProblemShape problem_shape;
        int* grouped_layout = nullptr;
    };

    struct Arguments
    {
        ElementA* ptr_A;
        typename KT::StrideA dA;
        typename KT::ElementB* ptr_B;
        typename KT::StrideB dB;
        ElementSFLoad* ptr_SFA;
        typename KT::StrideSFA dSFA;
        ElementSFLoad* ptr_SFB;
        typename KT::StrideSFB dSFB;
        ElementD* ptr_D;
        typename KT::StrideD dD;
        int* grouped_layout = nullptr;
    };

    static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args)
    {
        auto M = cute::get<0>(problem_shape);
        auto N = cute::get<1>(problem_shape);
        auto K = cute::get<2>(problem_shape);
        auto L = cute::get<3>(problem_shape);

        auto tensor_B = make_tensor(make_gmem_ptr(args.ptr_B), make_layout(make_shape(N, K, L), args.dB));
        typename KT::TMA_B tma_load_b
            = make_tma_copy(SM90_TMA_LOAD{}, tensor_B, typename KT::SmemLayoutB{}(_, _, Int<0>{}),
                make_shape(shape<1>(typename KT::TileShape{}), shape<2>(typename KT::TileShape{})), _1{});

        auto sfa_layout = KT::deduce_sfa_layout(problem_shape);
        auto sfb_layout = KT::deduce_sfb_layout(problem_shape);
        auto tensor_sfa = make_tensor(make_gmem_ptr(args.ptr_SFA), sfa_layout);
        auto tensor_sfb = make_tensor(make_gmem_ptr(args.ptr_SFB), sfb_layout);
        typename KT::TMA_SFA tma_load_sfa
            = make_tma_copy(SM90_TMA_LOAD{}, tensor_sfa, typename KT::SmemLayoutSFA{}(_, _, Int<0>{}),
                make_shape(shape<0>(typename KT::ScaleTileShape{}), shape<2>(typename KT::ScaleTileShape{})), _1{});
        typename KT::TMA_SFB tma_load_sfb
            = make_tma_copy(SM90_TMA_LOAD{}, tensor_sfb, typename KT::SmemLayoutSFB{}(_, _, Int<0>{}),
                make_shape(shape<1>(typename KT::ScaleTileShape{}), shape<2>(typename KT::ScaleTileShape{})), _1{});

        auto tensor_d = make_tensor(make_gmem_ptr(args.ptr_D), make_layout(make_shape(M, N, L), args.dD));
        auto tma_store_d = make_tma_copy_C_sm90(
            typename KT::CopyOpS2G{}, tensor_d, take<0, 2>(typename KT::SmemLayoutD{}), typename KT::EpilogueTile_MN{});

        // ld_a in elements; for FP8 row-major with leading dim K elements, that is just K.
        int64_t ld_a_elems = cute::get<0>(args.dA);
        int64_t stride_a_elems = cute::get<2>(args.dA);

        return {args.ptr_A, ld_a_elems, stride_a_elems,
                tma_load_b, tma_load_sfa, tma_load_sfb, tma_store_d,
                problem_shape, args.grouped_layout};
    }

    static dim3 get_grid_shape(Params const& params)
    {
        int device;
        cudaGetDevice(&device);
        int sm_count;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
        return dim3(sm_count, 1, 1);
    }

    static dim3 get_block_shape()
    {
        return dim3(MaxThreadsPerBlock, 1, 1);
    }

    CUTE_DEVICE
    static void prefetch_tma_descriptors(Params const& params)
    {
        cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_sfa.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_sfb.get_tma_descriptor());
    }

    using TensorStorage = typename KT::TensorStorage;
    using BarrierStorage = typename KT::BarrierStorage;

    struct SharedStorage
    {
        TensorStorage tensors;
        alignas(16) BarrierStorage barriers;
    };

    static constexpr int kSmemSize = int(sizeof(SharedStorage));

    CUTE_DEVICE
    static auto get_mbarriers(SharedStorage& shared_storage)
    {
        using FullBarrier = typename KT::FullBarrier;
        using EmptyBarrier = typename KT::EmptyBarrier;
        auto* ab_full_mbar = recast_ptr<FullBarrier>(&shared_storage.barriers.ab_full_mbar[0]);
        auto* ab_empty_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.ab_empty_mbar[0]);
        auto* sf_full_mbar = recast_ptr<FullBarrier>(&shared_storage.barriers.sf_full_mbar[0]);
        auto* sf_empty_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.sf_empty_mbar[0]);
        auto* store_full_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.store_full_mbar[0]);
        auto* store_empty_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.store_empty_mbar[0]);
        return cute::make_tuple(
            ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_full_mbar, store_empty_mbar);
    }

    template <class BlkCoord>
    CUTE_DEVICE static void load_sf(Params const& params, SharedStorage& shared_storage, BlkCoord const& blk_coord,
        int32_t sf_tile_count, uint32_t& phase, uint32_t& store_phase)
    {
        using X = Underscore;
        auto mSFA_mkl = params.tma_load_sfa.get_tma_tensor(shape(KT::deduce_sfa_layout(params.problem_shape)));
        auto mSFB_nkl = params.tma_load_sfb.get_tma_tensor(shape(KT::deduce_sfb_layout(params.problem_shape)));

        auto gSFA_mkl = local_tile(mSFA_mkl, typename KT::ScaleTileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
        auto gSFB_nkl = local_tile(mSFB_nkl, typename KT::ScaleTileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});

        auto block_tma_sfa = params.tma_load_sfa.get_slice(0);
        auto block_tma_sfb = params.tma_load_sfb.get_slice(0);

        auto m_coord = cute::get<0>(blk_coord);
        auto n_coord = cute::get<1>(blk_coord);
        auto l_coord = cute::get<2>(blk_coord);

        auto gSFA = gSFA_mkl(_, _, m_coord, _, l_coord);
        auto gSFB = gSFB_nkl(_, _, n_coord, _, l_coord);

        auto tAgSFA = block_tma_sfa.partition_S(gSFA);
        auto tBgSFB = block_tma_sfb.partition_S(gSFB);

        auto sSFA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()),
            typename KT::SmemLayoutSFA{});
        auto sSFB_
            = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB.begin()), typename KT::SmemLayoutSFB{});
        auto sSFA = as_position_independent_swizzle_tensor(sSFA_);
        auto sSFB = as_position_independent_swizzle_tensor(sSFB_);

        auto tAsSFA = block_tma_sfa.partition_D(sSFA);
        auto tBsSFB = block_tma_sfb.partition_D(sSFB);

        auto mbarriers = get_mbarriers(shared_storage);
        auto& sf_full_mbar = cute::get<2>(mbarriers);
        auto& sf_empty_mbar = cute::get<3>(mbarriers);
        auto& store_empty_mbar = cute::get<5>(mbarriers);
        store_empty_mbar[0].wait(store_phase);
        store_phase ^= 1;

        for (int32_t sf_tile_idx = 0; sf_tile_idx < sf_tile_count; ++sf_tile_idx)
        {
            sf_empty_mbar[0].wait(phase);
            auto& sf_full_barrier = sf_full_mbar[0];
            auto tma_copy_sfa
                = params.tma_load_sfa.with(*recast_ptr<typename KT::ProducerBarrierType>(&sf_full_barrier));
            cute::copy(tma_copy_sfa, tAgSFA(_, _, _, sf_tile_idx), tAsSFA(_, _, _, Int<0>{}));
            auto tma_copy_sfb
                = params.tma_load_sfb.with(*recast_ptr<typename KT::ProducerBarrierType>(&sf_full_barrier));
            cute::copy(tma_copy_sfb, tBgSFB(_, _, _, sf_tile_idx), tBsSFB(_, _, _, Int<0>{}));
            sf_full_mbar[0].arrive_and_expect_tx(KT::TmaSFTransactionBytes);
            phase ^= 1;
        }
    }

    // load_ab: manual cp.async.bulk for A; TMA for B.
    template <class BlkCoord>
    CUTE_DEVICE static void load_ab(Params const& params, SharedStorage& shared_storage, BlkCoord const& blk_coord,
        int32_t sf_tile_count, uint32_t& phase, uint32_t& store_phase)
    {
        using X = Underscore;
        auto M = cute::get<0>(params.problem_shape);
        auto N = cute::get<1>(params.problem_shape);
        auto K = cute::get<2>(params.problem_shape);
        auto L = cute::get<3>(params.problem_shape);

        auto mB_nkl = params.tma_load_b.get_tma_tensor(make_shape(N, K, L));
        auto gB_nkl = local_tile(
            mB_nkl, typename KT::TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});

        auto block_tma_b = params.tma_load_b.get_slice(0);
        auto m_coord = cute::get<0>(blk_coord);
        auto n_coord = cute::get<1>(blk_coord);
        auto l_coord = cute::get<2>(blk_coord);
        auto gB = gB_nkl(_, _, n_coord, _, l_coord);
        auto tBgB = block_tma_b.partition_S(gB);

        auto sB_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B.begin()),
            typename KT::SmemLayoutB{});
        auto sB = as_position_independent_swizzle_tensor(sB_);
        auto tBsB = block_tma_b.partition_D(sB);

        auto mbarriers = get_mbarriers(shared_storage);
        auto& ab_full_mbar = cute::get<0>(mbarriers);
        auto& ab_empty_mbar = cute::get<1>(mbarriers);
        auto& store_empty_mbar = cute::get<5>(mbarriers);
        store_empty_mbar[0].wait(store_phase);
        store_phase ^= 1;

        // Compute M_in_tile = number of valid rows of A in this CTA's m-block.
        int m_block_base = m_coord * KT::kTileM;
        int M_in_tile = M - m_block_base;
        if (M_in_tile < 0) M_in_tile = 0;
        if (M_in_tile > KT::kTileM) M_in_tile = KT::kTileM;

        constexpr int kTileK = KT::kTileK;
        constexpr int kAtomBytes = 8 * kTileK;           // 8 rows × 128 K-bytes per SW128 atom
        constexpr int kStageBytes = KT::kTileM * kTileK; // 16 × 128 = 2048

        uint32_t const smem_A_base_int
            = cast_smem_ptr_to_uint(shared_storage.tensors.load.smem_A.begin());

        int32_t k_tile_count = sf_tile_count * KT::kNumTileKPerSF;
        for (int32_t k_tile_idx = 0; k_tile_idx < k_tile_count; k_tile_idx += KT::AB_Stages)
        {
            cute::for_each(cute::make_int_sequence<KT::AB_Stages>{},
                [&](auto write_stage_int)
                {
                    constexpr int write_stage = decltype(write_stage_int)::value;
                    ab_empty_mbar[write_stage].wait(phase);
                    auto& ab_full_barrier = ab_full_mbar[write_stage];

                    // Manual A: per-16-byte-chunk cp.async.bulk. The SW128
                    // swizzle XORs bit 4 (= 16-byte stride) based on
                    // bits[9:7] of the byte offset, which flips chunk
                    // positions WITHIN each row's 128-byte span. So one
                    // cp.async.bulk per row is incorrect (writes contiguous
                    // 128 bytes to a non-contiguous swizzled position).
                    // Instead, we issue 8 chunks of 16 bytes per row.
                    //
                    // For chunk c ∈ [0,8) of row m, stage s:
                    //   linear_off    = s*kStageBytes + (m/8)*kAtomBytes
                    //                 + (m%8)*kTileK + c*16
                    //   bits[9:7] of linear_off depend only on (m%8) for
                    //   c*16 < 128 → swizzle mask = (m%8) << 4 affecting
                    //   bits[6:4] (= the c index).
                    //   swizzled_off  = s*kStageBytes + (m/8)*kAtomBytes
                    //                 + (m%8)*kTileK + (c ^ (m%8))*16
                    constexpr int kChunkBytes = 16;
                    constexpr int kChunksPerRow = kTileK / kChunkBytes;  // 8
                    #pragma unroll
                    for (int m = 0; m < KT::kTileM; ++m)
                    {
                        if (m < M_in_tile)
                        {
                            #pragma unroll
                            for (int c = 0; c < kChunksPerRow; ++c)
                            {
                                int64_t gmem_off
                                    = static_cast<int64_t>(m_block_base + m) * params.ld_a
                                      + static_cast<int64_t>(k_tile_idx + write_stage) * kTileK
                                      + static_cast<int64_t>(l_coord) * params.stride_a
                                      + static_cast<int64_t>(c * kChunkBytes);
                                void const* gmem_chunk
                                    = static_cast<void const*>(params.ptr_A + gmem_off);

                                int swizzled_off = write_stage * kStageBytes
                                                   + (m / 8) * kAtomBytes
                                                   + (m % 8) * kTileK
                                                   + (c ^ (m % 8)) * kChunkBytes;
                                uint32_t smem_int = smem_A_base_int + swizzled_off;

                                cute::SM90_BULK_COPY_G2S::copy(
                                    gmem_chunk,
                                    reinterpret_cast<uint64_t*>(&ab_full_barrier),
                                    reinterpret_cast<void*>(static_cast<uintptr_t>(smem_int)),
                                    kChunkBytes);
                            }
                        }
                    }

                    // TMA B
                    auto tma_copy_b
                        = params.tma_load_b.with(*recast_ptr<typename KT::ProducerBarrierType>(&ab_full_barrier));
                    cute::copy(tma_copy_b, tBgB(_, _, _, k_tile_idx + write_stage), tBsB(_, _, _, write_stage));

                    uint32_t const expect_a = static_cast<uint32_t>(M_in_tile) * kTileK;
                    uint32_t const expect_b = KT::TmaTransactionBytesB;
                    ab_full_mbar[write_stage].arrive_and_expect_tx(expect_a + expect_b);
                });
            phase ^= 1;
        }
    }

    CUTE_DEVICE
    static void mma(SharedStorage& shared_storage, int32_t sf_tile_count, uint32_t& sf_phase, uint32_t& ab_phase)
    {
        [[maybe_unused]] int thread_idx = int(threadIdx.x);

        auto sA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_A.begin()),
            typename KT::SmemLayoutA{});
        auto sB_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B.begin()), typename KT::SmemLayoutB{});
        auto sSFA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()),
            typename KT::SmemLayoutSFA{});
        auto sSFB_
            = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB.begin()), typename KT::SmemLayoutSFB{});
        auto sA = as_position_independent_swizzle_tensor(sA_);
        auto sB = as_position_independent_swizzle_tensor(sB_);
        auto sSFA = as_position_independent_swizzle_tensor(sSFA_);
        auto sSFB = as_position_independent_swizzle_tensor(sSFB_);

        typename KT::TiledMma mma;
        auto thr_mma = mma.get_thread_slice(thread_idx);
        auto accum = partition_fragment_C(mma, cute::take<0, 2>(typename KT::TileShape{}));
        auto tCrA = thr_mma.partition_fragment_A(sA(_, _, Int<0>{}));
        auto tCrB = thr_mma.partition_fragment_B(sB(_, _, Int<0>{}));

        auto s2r_copy_A = make_tiled_copy_A(typename KT::SmemCopyAtomA{}, mma);
        auto s2r_thr_copy_A = s2r_copy_A.get_thread_slice(thread_idx);
        auto tXsA = s2r_thr_copy_A.partition_S(sA);
        auto tXrA = s2r_thr_copy_A.retile_D(tCrA);

        auto s2r_copy_B = make_tiled_copy_B(typename KT::SmemCopyAtomB{}, mma);
        auto s2r_thr_copy_B = s2r_copy_B.get_thread_slice(thread_idx);
        auto tXsB = s2r_thr_copy_B.partition_S(sB);
        auto tXrB = s2r_thr_copy_B.retile_D(tCrB);

        auto s2r_copy_SFA = make_tiled_copy_impl(
            typename KT::SmemCopyAtomSF{}, KT::get_layoutSFA_TV(mma), make_shape(size<0>(tile_shape(mma)), _1{}));
        auto s2r_thr_copy_SFA = s2r_copy_SFA.get_thread_slice(thread_idx);
        auto tXsSFA = s2r_thr_copy_SFA.partition_S(sSFA);
        auto tCrSFA = KT::partition_fragment_SFA(sSFA(_, _, Int<0>{}), thr_mma);
        auto tXrSFA = s2r_thr_copy_SFA.retile_D(tCrSFA);
        auto tCrSFA_frg = KT::transform_fragment_for_qmma(tCrSFA);

        auto s2r_copy_SFB = make_tiled_copy_impl(
            typename KT::SmemCopyAtomSF{}, KT::get_layoutSFB_TV(mma), make_shape(size<1>(tile_shape(mma)), _1{}));
        auto s2r_thr_copy_SFB = s2r_copy_SFB.get_thread_slice(thread_idx);
        auto tXsSFB = s2r_thr_copy_SFB.partition_S(sSFB);
        auto tCrSFB = KT::partition_fragment_SFB(sSFB(_, _, Int<0>{}), thr_mma);
        auto tXrSFB = s2r_thr_copy_SFB.retile_D(tCrSFB);
        auto tCrSFB_frg = KT::transform_fragment_for_qmma(tCrSFB);

        cute::clear(accum);
        auto mbarriers = get_mbarriers(shared_storage);
        auto& ab_full_mbar = cute::get<0>(mbarriers);
        auto& ab_empty_mbar = cute::get<1>(mbarriers);
        auto& sf_full_mbar = cute::get<2>(mbarriers);
        auto& sf_empty_mbar = cute::get<3>(mbarriers);
        auto& store_full_mbar = cute::get<4>(mbarriers);

        for (int32_t sf_tile_idx = 0; sf_tile_idx < sf_tile_count - 1; ++sf_tile_idx)
        {
            sf_full_mbar[0].wait(sf_phase);
            cute::copy(s2r_copy_SFA, tXsSFA(_, _, _, Int<0>{}), tXrSFA);
            cute::copy(s2r_copy_SFB, tXsSFB(_, _, _, Int<0>{}), tXrSFB);
            sf_empty_mbar[0].arrive();

            cute::for_each(cute::make_int_sequence<KT::kNumStagePerSF>{},
                [&](auto iter)
                {
                    cute::for_each(cute::make_int_sequence<KT::AB_Stages>{},
                        [&](auto read_stage)
                        {
                            ab_full_mbar[read_stage].wait(ab_phase);
                            cute::copy(s2r_copy_A, tXsA(_, _, _, read_stage), tXrA);
                            cute::copy(s2r_copy_B, tXsB(_, _, _, read_stage), tXrB);
                            ab_empty_mbar[read_stage].arrive();

                            auto tCrSFA_stage = tCrSFA_frg(_, _, _, iter * KT::AB_Stages + read_stage);
                            auto tCrSFB_stage = tCrSFB_frg(_, _, _, iter * KT::AB_Stages + read_stage);
                            cute::gemm(
                                mma, make_zip_tensor(tCrA, tCrSFA_stage), make_zip_tensor(tCrB, tCrSFB_stage), accum);
                        });
                    ab_phase ^= 1;
                });
            sf_phase ^= 1;
        }

        sf_full_mbar[0].wait(sf_phase);
        cute::copy(s2r_copy_SFA, tXsSFA(_, _, _, Int<0>{}), tXrSFA);
        cute::copy(s2r_copy_SFB, tXsSFB(_, _, _, Int<0>{}), tXrSFB);
        sf_empty_mbar[0].arrive();

        cute::for_each(cute::make_int_sequence<KT::kNumStagePerSF>{},
            [&](auto iter)
            {
                cute::for_each(cute::make_int_sequence<KT::AB_Stages>{},
                    [&](auto read_stage)
                    {
                        ab_full_mbar[read_stage].wait(ab_phase);
                        cute::copy(s2r_copy_A, tXsA(_, _, _, read_stage), tXrA);
                        cute::copy(s2r_copy_B, tXsB(_, _, _, read_stage), tXrB);
                        ab_empty_mbar[read_stage].arrive();
                        if constexpr (iter == KT::kNumStagePerSF - 1 && read_stage == KT::AB_Stages - 1)
                        {
                            cutlass::arch::NamedBarrier::sync(KT::kNumMathThreads, 0);
                        }
                        auto tCrSFA_stage = tCrSFA_frg(_, _, _, iter * KT::AB_Stages + read_stage);
                        auto tCrSFB_stage = tCrSFB_frg(_, _, _, iter * KT::AB_Stages + read_stage);
                        cute::gemm(
                            mma, make_zip_tensor(tCrA, tCrSFA_stage), make_zip_tensor(tCrB, tCrSFB_stage), accum);
                    });
                ab_phase ^= 1;
            });
        sf_phase ^= 1;

        // epilogue
        auto accum_frg = recast<Array<typename KT::ElementAccum, 2>>(accum);
        auto epi = make_fragment_like<ElementD>(accum);
        auto epi_frg = recast<Array<ElementD, 2>>(epi);
        cutlass::NumericArrayConverter<ElementD, typename KT::ElementAccum, 2> converter;
        cute::for_each(
            cute::make_int_sequence<cute::size(epi_frg)>{}, [&](auto i) { epi_frg(i) = converter(accum_frg(i)); });

        auto tiled_copy_C_atom = make_tiled_copy_C_atom(typename KT::CopyAtomC{}, mma);
        auto tiled_copy_r2s
            = make_tiled_copy_S(cute::Copy_Atom<typename KT::CopyOpR2S, ElementD>{}, tiled_copy_C_atom);
        auto thr_copy_r2s = tiled_copy_r2s.get_slice(thread_idx);

        auto sD_epi_ = make_tensor(make_smem_ptr(shared_storage.tensors.store.smem_D.begin()),
            typename KT::SmemLayoutD{});
        auto sD_epi = cute::as_position_independent_swizzle_tensor(sD_epi_);
        auto tRS_rD = thr_copy_r2s.retile_S(epi);
        auto tRS_sD = thr_copy_r2s.partition_D(sD_epi);

        copy(tiled_copy_r2s, tRS_rD, tRS_sD(_, _, _, Int<0>{}));
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(KT::kNumMathThreads, 0);
        store_full_mbar[0].arrive();
    }

    template <class BlkCoord>
    CUTE_DEVICE static void store(
        Params const& params, SharedStorage& shared_storage, BlkCoord const& blk_coord, uint32_t& phase)
    {
        auto mbarriers = get_mbarriers(shared_storage);
        auto& store_full_mbar = cute::get<4>(mbarriers);
        auto& store_empty_mbar = cute::get<5>(mbarriers);
        store_full_mbar[0].wait(phase);
        using EpilogueTile = typename KT::EpilogueTile_MN;
        auto M = cute::get<0>(params.problem_shape);
        auto N = cute::get<1>(params.problem_shape);
        auto L = cute::get<3>(params.problem_shape);
        auto mD_mnl = params.tma_store_d.get_tma_tensor(make_shape(M, N, L));
        auto gD_mnl = local_tile(mD_mnl, typename KT::TileShape{}, make_coord(_, _, _), Step<_1, _1, X>{});
        auto m_coord = cute::get<0>(blk_coord);
        auto n_coord = cute::get<1>(blk_coord);
        auto l_coord = cute::get<2>(blk_coord);
        auto gD = gD_mnl(_, _, m_coord, n_coord, l_coord);
        auto gD_epi = flat_divide(gD, EpilogueTile{});

        auto block_tma_d = params.tma_store_d.get_slice(Int<0>{});
        auto sD_epi_ = make_tensor(make_smem_ptr(shared_storage.tensors.store.smem_D.begin()),
            typename KT::SmemLayoutD{});
        auto sD_epi = cute::as_position_independent_swizzle_tensor(sD_epi_);
        auto bSG_sD = block_tma_d.partition_S(sD_epi);
        auto bSG_gD = block_tma_d.partition_D(gD_epi);

        for (int epi_n = 0; epi_n < size<3>(bSG_gD); ++epi_n)
        {
            for (int epi_m = 0; epi_m < size<2>(bSG_gD); ++epi_m)
            {
                cute::copy(params.tma_store_d, bSG_sD(_, _, _, Int<0>{}), bSG_gD(_, _, _, epi_m, epi_n));
            }
        }
        cute::tma_store_arrive();
        cute::tma_store_wait<0>();
        store_empty_mbar[0].arrive();
        phase ^= 1;
    }

    CUTE_DEVICE
    void operator()(Params const& params, char* smem_buf)
    {
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
        int warp_idx = canonical_warp_idx_sync();
        int lane_predicate = cute::elect_one_sync();
        bool is_tma_thread = warp_idx == 0 && lane_predicate;

        if (is_tma_thread)
        {
            prefetch_tma_descriptors(params);
        }
        __syncthreads();

        auto mbarriers = get_mbarriers(shared_storage);
        auto& ab_full_mbar = cute::get<0>(mbarriers);
        auto& ab_empty_mbar = cute::get<1>(mbarriers);
        auto& sf_full_mbar = cute::get<2>(mbarriers);
        auto& sf_empty_mbar = cute::get<3>(mbarriers);
        auto& store_full_mbar = cute::get<4>(mbarriers);
        auto& store_empty_mbar = cute::get<5>(mbarriers);
        if (is_tma_thread)
        {
#pragma unroll
            for (uint32_t i = 0; i < KT::SF_Stages; ++i)
            {
                sf_full_mbar[i].init(1);
                sf_empty_mbar[i].init(KT::kNumMathThreads);
            }

#pragma unroll
            for (uint32_t i = 0; i < KT::AB_Stages; ++i)
            {
                ab_full_mbar[i].init(1);
                ab_empty_mbar[i].init(KT::kNumMathThreads);
            }
            store_full_mbar[0].init(KT::kNumMathThreads);
            store_empty_mbar[0].init(1);
            cutlass::arch::fence_barrier_init();
        }
        __syncthreads();

        auto M = cute::get<0>(params.problem_shape);
        auto N = cute::get<1>(params.problem_shape);
        auto L = cute::get<3>(params.problem_shape);
        int32_t sf_tile_count = (cute::get<2>(params.problem_shape) + 511) / 512;

        using Scheduler = SM120BlockScaledScheduler<KT::kTileM, KT::kTileN, KT::kSchedGroup>;
        if (warp_idx >= KT::kNumMathWarps)
        {
            constexpr int epi_warp_idx = KT::kNumMathWarps;
            constexpr int ab_warp_idx = epi_warp_idx + 1;
            constexpr int sf_warp_idx = ab_warp_idx + 1;
            if (warp_idx == ab_warp_idx)
            {
                uint32_t phase = 1;
                uint32_t store_phase = 1;
                if (lane_predicate)
                {
                    auto scheduler = Scheduler(M, N, L, params.grouped_layout);
                    while (scheduler.get_next_block())
                    {
                        auto blk_coord = cute::make_coord(
                            scheduler.m_block_idx, scheduler.n_block_idx, scheduler.current_group_idx);
                        load_ab(params, shared_storage, blk_coord, sf_tile_count, phase, store_phase);
                    }
                }
                __syncwarp();
            }
            if (warp_idx == sf_warp_idx)
            {
                uint32_t phase = 1;
                uint32_t store_phase = 1;
                if (lane_predicate)
                {
                    auto scheduler = Scheduler(M, N, L, params.grouped_layout);
                    while (scheduler.get_next_block())
                    {
                        auto blk_coord = cute::make_coord(
                            scheduler.m_block_idx, scheduler.n_block_idx, scheduler.current_group_idx);
                        load_sf(params, shared_storage, blk_coord, sf_tile_count, phase, store_phase);
                    }
                }
                __syncwarp();
            }
            if (warp_idx == epi_warp_idx)
            {
                uint32_t phase = 0;
                if (lane_predicate)
                {
                    auto scheduler = Scheduler(M, N, L, params.grouped_layout);
                    while (scheduler.get_next_block())
                    {
                        auto blk_coord = cute::make_coord(
                            scheduler.m_block_idx, scheduler.n_block_idx, scheduler.current_group_idx);
                        store(params, shared_storage, blk_coord, phase);
                    }
                }
                __syncwarp();
            }
        }
        else
        {
            uint32_t sf_phase = 0;
            uint32_t ab_phase = 0;
            auto scheduler = Scheduler(M, N, L, params.grouped_layout);
            while (scheduler.get_next_block())
            {
                mma(shared_storage, sf_tile_count, sf_phase, ab_phase);
            }
        }
    }

  private:
    using X = Underscore;
};

} // namespace sm120_blockscaled_gemm
