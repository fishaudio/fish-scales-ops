/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "blockscale_gemm/arch/sm120/fp8/utils.cuh"
#include "blockscale_gemm/arch/sm120/mxfp8/fused_combine_epi.cuh"
#include "blockscale_gemm/arch/sm120/mxfp8/fused_swiglu_epi.cuh"

#if defined(FSO_PROLOGUE_TRACE)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#endif

using namespace cute;

namespace sm120_blockscaled_gemm
{

#if defined(FSO_PROLOGUE_TRACE)
// FSO_PROLOGUE_TRACE is a compile-time-optional prologue trace, OFF by default.
// Without the macro none of the code it guards exists, and the FSO_PT_* macros
// below expand to nothing, so the kernel is the production kernel. With it the
// kernel stamps %globaltimer and %clock64 at the prologue's phase boundaries into
// a device buffer whose address the launcher reads once from FSO_PROLOGUE_TRACE_PTR
// (unset: no stamps). It exists for scratch builds of the prologue study (run
// sm120_prologue_20260928), e.g. NVCC_APPEND_FLAGS=-DFSO_PROLOGUE_TRACE=1.
//
// The buffer holds kSlots records of kSlotWords uint64 words, one per kernel
// flavour (0 = fused-SwiGLU FC1, 1 = plain epilogue, 2 = fused-combine FC2). Every
// launch of a flavour overwrites its record, so a reader zeroes the buffer, runs,
// and reads the last launch. Record layout:
//   [0, 64)            stamps of CTA 0 (sel 0) and CTA gridDim.x - 1 (sel 1):
//                      word (sel * kNumPts + pt) * 2 is %globaltimer (ns), the
//                      next word %clock64
//   [64, 68)           %smid of sel 0 and sel 1, gridDim.x, blockDim.x
//   [128 + 20 b, +20)  CTA b: entry, PDL wait entered, PDL wait released, first
//                      tile start (math warp 0), %smid, then the exit of warps
//                      0..11, then (math thread 0, first tile) the first scale-
//                      factor stage landing and the end of the k-loop; %globaltimer
//                      ns, 0 where the event did not happen
namespace prologue_trace
{
enum : int
{
    kEntry = 0,     // thread 0, first instruction
    kPrefetchSync,  // thread 0, after the TMA-descriptor prefetch __syncthreads
    kBarrierSync,   // thread 0, after the mbarrier-init __syncthreads
    kStageSync,     // thread 0, after the grouped_layout staging __syncthreads
    kScanSync,      // thread 0, after the prefix-scan __syncthreads
    kPdlWait,       // thread 0, after cudaGridDependencySynchronize
    kMathFirstNext, // thread 0 (math warp 0), first get_next_block return
    kFirstOperands, // thread 0, first tile: the first scale-factor stage has landed
    kMainloopDone,  // thread 0, first tile: k-loop done, epilogue starts
    kEpilogueDone,  // first tile's store issued (store warp) or written (math warps)
    kLoadFirstNext, // A/B load warp, first get_next_block return
    kMathExit,      // thread 0, end of the kernel body
    kNumPts = 16
};
constexpr int kSlots = 3;
constexpr int kMetaBase = 2 * kNumPts * 2;
constexpr int kCtaBase = 128;
constexpr int kCtaWords = 20;
constexpr int kMaxCtas = 512;
constexpr int kSlotWords = kCtaBase + kMaxCtas * kCtaWords;

CUTE_DEVICE uint64_t gtimer()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

CUTE_DEVICE uint64_t cycles()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
    return t;
}

CUTE_DEVICE uint32_t smid()
{
    uint32_t s;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    return s;
}

// `detail` is the stamp row of a detail CTA, nullptr for every other CTA.
CUTE_DEVICE void put(uint64_t* detail, int pt)
{
    if (detail != nullptr)
    {
        detail[2 * pt] = gtimer();
        detail[2 * pt + 1] = cycles();
    }
}

// Wraps one get_next_block() call: on the first call only (and only where `cond`
// holds) stamps `pt`, and the CTA's first-tile word when the call found a tile.
CUTE_DEVICE bool after_next(bool has, int& calls, bool cond, uint64_t* detail, int pt, uint64_t* cta_word)
{
    if (calls++ == 0 && cond)
    {
        put(detail, pt);
        if (has && cta_word != nullptr)
        {
            *cta_word = gtimer();
        }
    }
    return has;
}

inline uint64_t* buffer_from_env()
{
    static uint64_t* const s_buf = []
    {
        char const* e = std::getenv("FSO_PROLOGUE_TRACE_PTR");
        if (e == nullptr || *e == '\0')
        {
            return static_cast<uint64_t*>(nullptr);
        }
        std::fprintf(stderr, "[fso] prologue trace ON, buffer %s\n", e);
        return reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(std::strtoull(e, nullptr, 0)));
    }();
    return s_buf;
}
} // namespace prologue_trace
#define FSO_PT_PARAM , uint64_t* trc_first_tile = nullptr, uint64_t* trc_cta_first = nullptr
#define FSO_PT_ARG(x) , (x)
#define FSO_PT_ARG2(x, y) , (x), (y)
#define FSO_PT_CALLS int trc_calls = 0
#define FSO_PT_NEXT(expr, cond, pt, cta_word)                                                                          \
    prologue_trace::after_next((expr), trc_calls, (cond), trc_det, prologue_trace::pt, (cta_word))
// A role loop passes its trace pointers to the first tile's body only.
#define FSO_PT_FIRST_DECL bool trc_first = true
#define FSO_PT_FIRST(x) (trc_first ? (x) : nullptr)
#define FSO_PT_FIRST_DONE trc_first = false
#define FSO_PT_STAMP(pt)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (threadIdx.x == 0)                                                                                          \
            prologue_trace::put(trc_det, prologue_trace::pt);                                                          \
    } while (0)
#else
#define FSO_PT_PARAM
#define FSO_PT_ARG(x)
#define FSO_PT_ARG2(x, y)
#define FSO_PT_CALLS                                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#define FSO_PT_NEXT(expr, cond, pt, cta_word) (expr)
#define FSO_PT_FIRST_DECL                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#define FSO_PT_FIRST(x) nullptr
#define FSO_PT_FIRST_DONE                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#define FSO_PT_STAMP(pt)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#endif

// FusedSwiGLUEpi turns the epilogue into the MoE layer's FC1 tail: instead of
// staging the bf16 tile for a TMA store into a [G, m_cap, 2*INTER] slab, the math
// warps read the staged tile back, compute silu(gate) * up, requantize to MXFP8
// and write the [G, m_cap, INTER] slab the FC2 consumes, so the gate/up slab and
// the kernel that used to read it both disappear. See
// arch/sm120/mxfp8/fused_swiglu_epi.cuh for the arithmetic and why it goes
// through shared memory. With the flag false every line below is what it was.
// ScatterCombineEpi is the FC2's counterpart: instead of storing the tile into a
// [G, m_cap, HIDDEN] slab for a separate combine kernel to read back and sum, each
// row is scaled by its combine weight and added straight into its token's output
// row with eight-byte atomic reductions. See
// arch/sm120/mxfp8/fused_combine_epi.cuh for the request-width measurement that
// decides the shape, and for what the caller owes (a pre-filled output, and no
// expectation of bit-reproducibility).
template <typename KT, bool FusedSwiGLUEpi = false, bool ScatterCombineEpi = false,
    bool ScatterInStoreWarp = false>
struct SM120BlockScaledKernel
{
    static_assert(!(FusedSwiGLUEpi && ScatterCombineEpi),
        "a GEMM has one epilogue: the fused SwiGLU belongs to the FC1 and the fused combine to the FC2");
    static_assert(!(FusedSwiGLUEpi || ScatterCombineEpi) || KT::kSeparateSmemD,
        "the fused SwiGLU epilogue reads the staged tile after the mainloop has moved on, so the "
        "instance needs its own D buffer (SeparateSmemD) rather than one aliasing the A/B stages");
    static constexpr int kNumTMAThreads = 128;
    static constexpr int kNumScatterThreads = 64;   // store warp + the spare 4th TMA warp
    static constexpr int kScatterBarrierId = 1;     // 0 belongs to the math warps
    static constexpr int kNumMathThreads = KT::kNumMathThreads;
    static constexpr int MaxThreadsPerBlock = kNumTMAThreads + kNumMathThreads;
    // CUTLASS device_kernel applies __launch_bounds__(MaxThreadsPerBlock,
    // MinBlocksPerMultiprocessor); the "min-blocks" hint plans nvcc's register
    // budget per CTA so occupancy can be 1 or 2 CTAs/SM. Most instances run
    // 1 CTA/SM (smem cap), but small-tile instances (e.g. (32, 128, 2) ≈
    // 49 KB) leave room for 2 CTAs/SM if the builder opts in.
    static constexpr int MinBlocksPerMultiprocessor = KT::MinBlocksPerSm;
    // How many leading A/B stages of a launch's first tile issue their B (weight)
    // copy before the PDL wait; the SF-load warp issues the first span's SFB copy
    // there whenever this is nonzero. Grouped instantiations only: the dense ones keep
    // the production order (see the PDL note in operator()). One stage, not
    // all AB_Stages: issuing every stage (run sm120_r2_p2_20260929) gained up to 1.5 %
    // more on the Family C layer at M = 2-128 but lost 1.9 % (Family B) and 1.1 %
    // (Family C) at M = 1, because the extra weight copies land inside the still
    // running predecessor's own weight stream and slow it by about what they save.
    static constexpr int kPreWaitBStages = KT::kGroupedLayoutSmem ? 1 : 0;
    static_assert(kPreWaitBStages >= 0 && kPreWaitBStages <= KT::AB_Stages,
        "the pre-wait B copies belong to the first round of AB_Stages stages");

    using ProblemShape = typename KT::ProblemShape;

    struct Params
    {
        typename KT::TMA_A tma_load_a;
        typename KT::TMA_B tma_load_b;
        typename KT::TMA_SFA tma_load_sfa;
        typename KT::TMA_SFB tma_load_sfb;
        typename KT::TMA_D tma_store_d;
        typename KT::ProblemShape problem_shape;
        int* grouped_layout = nullptr;
        // Read only by the fused SwiGLU epilogue: the MXFP8 output slab, its
        // opaque int32 scale slab, the masked layout's row capacity and the INTER
        // width (half the GEMM's N).
        __nv_fp8_e4m3* out_fp8 = nullptr;
        int32_t* out_sf = nullptr;
        int m_cap = 0;
        int inter = 0;
        // Read only by the fused combine epilogue: the layer output it accumulates
        // into, the slot -> token map and the slot -> combine weight map.
        __nv_bfloat16* out_tokens = nullptr;
        int32_t const* row_map = nullptr;
        float const* weight_of_slot = nullptr;
        // Grouped path only: the slot -> group permutation the persistent
        // scheduler walks -- `group_perm` blocks apart, `group_block` consecutive
        // experts per block. group_perm = 1 is expert-id order (every dense
        // launch, and every grouped launch with the permutation off).
        int group_perm = 1;
        int group_block = 1;
        // Read only by the grouped plain epilogue's partial-tile store (store_rows):
        // the D slab and its row / group strides.
        typename KT::ElementD* ptr_d = nullptr;
        int64_t ld_d = 0;
        int64_t stride_d = 0;
#if defined(FSO_PROLOGUE_TRACE)
        uint64_t* trace = nullptr;
#endif
    };

    struct Arguments
    {
        typename KT::ElementA* ptr_A;
        typename KT::StrideA dA;
        typename KT::ElementB* ptr_B;
        typename KT::StrideB dB;
        typename KT::ElementSFLoad* ptr_SFA;
        typename KT::StrideSFA dSFA;
        typename KT::ElementSFLoad* ptr_SFB;
        typename KT::StrideSFB dSFB;
        typename KT::ElementD* ptr_D;
        typename KT::StrideD dD;
        int* grouped_layout = nullptr;
        __nv_fp8_e4m3* out_fp8 = nullptr;
        int32_t* out_sf = nullptr;
        int m_cap = 0;
        int inter = 0;
        __nv_bfloat16* out_tokens = nullptr;
        int32_t const* row_map = nullptr;
        float const* weight_of_slot = nullptr;
        int group_perm = 1;
        int group_block = 1;
    };

    static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args)
    {
        auto M = cute::get<0>(problem_shape);
        auto N = cute::get<1>(problem_shape);
        auto K = cute::get<2>(problem_shape);
        auto L = cute::get<3>(problem_shape);
        auto tensor_A = make_tensor(make_gmem_ptr(args.ptr_A), make_layout(make_shape(M, K, L), args.dA));
        typename KT::TMA_A tma_load_a
            = make_tma_copy(SM90_TMA_LOAD{}, tensor_A, typename KT::SmemLayoutA{}(_, _, Int<0>{}),
                make_shape(shape<0>(typename KT::TileShape{}), shape<2>(typename KT::TileShape{})), _1{});

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
        return {tma_load_a, tma_load_b, tma_load_sfa, tma_load_sfb, tma_store_d, problem_shape, args.grouped_layout,
            args.out_fp8, args.out_sf, args.m_cap, args.inter, args.out_tokens, args.row_map, args.weight_of_slot,
            args.group_perm, args.group_block, args.ptr_D, cute::get<0>(args.dD), cute::get<2>(args.dD)};
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
        cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_sfa.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_sfb.get_tma_descriptor());
    }

    using TensorStorage = typename KT::TensorStorageSel;
    using BarrierStorage = typename KT::BarrierStorage;

    // Grouped path: the persistent scheduler re-reads grouped_layout[g] while
    // walking groups. Left in global memory that walk is a chain of DEPENDENT
    // global loads (up to L per scheduler pass, ~L2 latency each) executed by
    // every warp of every CTA — ncu showed it as the dominant
    // long_scoreboard stall of the decode cells (45% of issue stalls at
    // G=128, M=1). One cooperative copy into shared memory at kernel start
    // turns the walk into LDS hits. 2 KB reserve; groups beyond the cap fall
    // back to the global pointer.
    static constexpr int kGroupedLayoutSmemCap = 512;

    // The 2 KB prefix array is reserved only when the builder opts in
    // (KT::kGroupedLayoutSmem, grouped MoE instantiations). Dense
    // instantiations keep the smaller layout: with the array, the dense
    // BSFP8 (64,128,4) instance is 102400 B and exceeds the sm_120 per-block
    // limit of 101376 B, so it could not launch (2026-09-03 re-baseline).
    struct SharedStorageDense
    {
        TensorStorage tensors;
        alignas(16) BarrierStorage barriers;
    };

    // The prefix is built by a block-wide scan (see operator()): each thread owns
    // kScanPer consecutive slots, and the warps exchange their totals through
    // grouped_scan_warp_total.
    static constexpr int kScanPer = (kGroupedLayoutSmemCap + MaxThreadsPerBlock - 1) / MaxThreadsPerBlock;
    static constexpr int kScanWarps = MaxThreadsPerBlock / 32;

    struct SharedStorageGrouped
    {
        TensorStorage tensors;
        alignas(16) BarrierStorage barriers;
        int32_t grouped_layout_smem[kGroupedLayoutSmemCap];
        int32_t grouped_scan_warp_total[kScanWarps];
    };

    using SharedStorage = cute::conditional_t<KT::kGroupedLayoutSmem, SharedStorageGrouped, SharedStorageDense>;

    static constexpr int kSmemSize = int(sizeof(SharedStorage));

    CUTE_DEVICE
    static auto get_mbarriers(SharedStorage& shared_storage)
    {
        using FullBarrier = typename KT::FullBarrier;
        using EmptyBarrier = typename KT::EmptyBarrier;
        using ProducerBarrierType = typename FullBarrier::ValueType;
        using ConsumerBarrierType = typename EmptyBarrier::ValueType;
        auto* ab_full_mbar = recast_ptr<FullBarrier>(&shared_storage.barriers.ab_full_mbar[0]);
        auto* ab_empty_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.ab_empty_mbar[0]);
        auto* sf_full_mbar = recast_ptr<FullBarrier>(&shared_storage.barriers.sf_full_mbar[0]);
        auto* sf_empty_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.sf_empty_mbar[0]);
        auto* store_full_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.store_full_mbar[0]);
        auto* store_empty_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.store_empty_mbar[0]);
        return cute::make_tuple(
            ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_full_mbar, store_empty_mbar);
    }

    // One tile's scale-factor spans. `first_tile` is set for the launch's first tile
    // only: load_sfb_first() has then already armed span 0's full barrier for both
    // halves and issued its SFB copy before the PDL wait, so only SFA remains here.
    template <class BlkCoord>
    CUTE_DEVICE static void load_sf(Params const& params, SharedStorage& shared_storage, BlkCoord const& blk_coord,
        int32_t sf_tile_count, uint32_t& phase, uint32_t& store_phase, bool first_tile = false)
    {
        using X = Underscore;
        auto M = cute::get<0>(params.problem_shape);
        auto N = cute::get<1>(params.problem_shape);
        auto K = cute::get<2>(params.problem_shape);
        auto L = cute::get<3>(params.problem_shape);
        auto mSFA_mkl = params.tma_load_sfa.get_tma_tensor(shape(KT::deduce_sfa_layout(params.problem_shape)));
        auto mSFB_nkl = params.tma_load_sfb.get_tma_tensor(shape(KT::deduce_sfb_layout(params.problem_shape)));

        auto gSFA_mkl = local_tile(
            mSFA_mkl, typename KT::ScaleTileShape{}, make_coord(_, _, _), Step<_1, X, _1>{}); // (TILE_M,TILE_K,m,k,l)
        auto gSFB_nkl = local_tile(
            mSFB_nkl, typename KT::ScaleTileShape{}, make_coord(_, _, _), Step<X, _1, _1>{}); // (TILE_N,TILE_K,n,k,l)

        auto block_tma_sfa = params.tma_load_sfa.get_slice(0);
        auto block_tma_sfb = params.tma_load_sfb.get_slice(0);

        auto m_coord = cute::get<0>(blk_coord);
        auto n_coord = cute::get<1>(blk_coord);
        auto l_coord = cute::get<2>(blk_coord);

        auto gSFA = gSFA_mkl(_, _, m_coord, _, l_coord);
        auto gSFB = gSFB_nkl(_, _, n_coord, _, l_coord);

        auto tAgSFA = block_tma_sfa.partition_S(gSFA); // (TMA,TMA_M,TMA_K,k)
        auto tBgSFB = block_tma_sfb.partition_S(gSFB); // (TMA,TMA_N,TMA_K,k)

        auto sSFA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()),
            typename KT::SmemLayoutSFA{}); // (BLK_M,BLK_K,PIPE)
        auto sSFB_
            = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB.begin()), typename KT::SmemLayoutSFB{});
        auto sSFA = as_position_independent_swizzle_tensor(sSFA_); // (BLK_M,BLK_K,PIPE)
        auto sSFB = as_position_independent_swizzle_tensor(sSFB_); // (BLK_N,BLK_K,PIPE)

        auto tAsSFA = block_tma_sfa.partition_D(sSFA);             // (TMA,TMA_M,TMA_K,PIPE)
        auto tBsSFB = block_tma_sfb.partition_D(sSFB);             // (TMA,TMA_N,TMA_K,PIPE)

        auto mbarriers = get_mbarriers(shared_storage);
        auto& ab_full_mbar = cute::get<0>(mbarriers);
        auto& ab_empty_mbar = cute::get<1>(mbarriers);
        auto& sf_full_mbar = cute::get<2>(mbarriers);
        auto& sf_empty_mbar = cute::get<3>(mbarriers);
        auto& store_full_mbar = cute::get<4>(mbarriers);
        auto& store_empty_mbar = cute::get<5>(mbarriers);
        if constexpr (!KT::kSeparateSmemD)
        {
            // Union storage: smem_D aliases smem_A/B, so this tile's loads
            // must wait for the previous tile's TMA store to drain.
            store_empty_mbar[0].wait(store_phase);
            store_phase ^= 1;
        }

        int32_t sf_tile_idx = 0;
        if constexpr (kPreWaitBStages > 0)
        {
            if (first_tile)
            {
                auto tma_copy_sfa
                    = params.tma_load_sfa.with(*recast_ptr<typename KT::ProducerBarrierType>(&sf_full_mbar[0]));
                cute::copy(tma_copy_sfa, tAgSFA(_, _, _, 0), tAsSFA(_, _, _, Int<0>{}));
                phase ^= 1; // flip phase
                sf_tile_idx = 1;
            }
        }
        for (; sf_tile_idx < sf_tile_count; ++sf_tile_idx)
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
            phase ^= 1; // flip phase
        }
    }

    // One tile's A/B stages, AB_Stages k-tiles per round. `first_tile` is set for the
    // launch's first tile only: load_b_first() has then already armed the full
    // barriers of the first kPreWaitBStages stages for both halves and issued their B
    // copy before the PDL wait, so the first round issues only their A copy.
    template <class BlkCoord>
    CUTE_DEVICE static void load_ab(Params const& params, SharedStorage& shared_storage, BlkCoord const& blk_coord,
        int32_t k_tile_count, uint32_t& phase, uint32_t& store_phase, bool first_tile = false)
    {
        using X = Underscore;
        auto M = cute::get<0>(params.problem_shape);
        auto N = cute::get<1>(params.problem_shape);
        auto K = cute::get<2>(params.problem_shape);
        auto L = cute::get<3>(params.problem_shape);

        auto mA_mkl = params.tma_load_a.get_tma_tensor(make_shape(M, K, L));
        auto mB_nkl = params.tma_load_b.get_tma_tensor(make_shape(N, K, L));

        // Make tiled views, defer the slice
        auto gA_mkl = local_tile(
            mA_mkl, typename KT::TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{}); // (BLK_M,BLK_K,m,k,l)
        auto gB_nkl = local_tile(
            mB_nkl, typename KT::TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{}); // (BLK_N,BLK_K,n,k,l)

        auto block_tma_a = params.tma_load_a.get_slice(0);
        auto block_tma_b = params.tma_load_b.get_slice(0);

        auto m_coord = cute::get<0>(blk_coord);
        auto n_coord = cute::get<1>(blk_coord);
        auto l_coord = cute::get<2>(blk_coord);

        auto gA = gA_mkl(_, _, m_coord, _, l_coord);
        auto gB = gB_nkl(_, _, n_coord, _, l_coord);

        auto tAgA = block_tma_a.partition_S(gA); // (TMA,TMA_M,TMA_K,k)
        auto tBgB = block_tma_b.partition_S(gB); // (TMA,TMA_N,TMA_K,k)

        auto sA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_A.begin()),
            typename KT::SmemLayoutA{});                       // (BLK_M,BLK_K,PIPE)
        auto sB_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B.begin()), typename KT::SmemLayoutB{});
        auto sA = as_position_independent_swizzle_tensor(sA_); // (BLK_M,BLK_K,PIPE)
        auto sB = as_position_independent_swizzle_tensor(sB_); // (BLK_N,BLK_K,PIPE)

        auto tAsA = block_tma_a.partition_D(sA);               // (TMA,TMA_M,TMA_K,PIPE)
        auto tBsB = block_tma_b.partition_D(sB);               // (TMA,TMA_N,TMA_K,PIPE)

        auto mbarriers = get_mbarriers(shared_storage);
        auto& ab_full_mbar = cute::get<0>(mbarriers);
        auto& ab_empty_mbar = cute::get<1>(mbarriers);
        auto& sf_full_mbar = cute::get<2>(mbarriers);
        auto& sf_empty_mbar = cute::get<3>(mbarriers);
        auto& store_full_mbar = cute::get<4>(mbarriers);
        auto& store_empty_mbar = cute::get<5>(mbarriers);
        if constexpr (!KT::kSeparateSmemD)
        {
            store_empty_mbar[0].wait(store_phase);
            store_phase ^= 1;
        }

        int32_t k_tile_idx = 0;
        if constexpr (kPreWaitBStages > 0)
        {
            if (first_tile)
            {
                cute::for_each(cute::make_int_sequence<KT::AB_Stages>{},
                    [&](auto write_stage)
                    {
                        auto& ab_full_barrier = ab_full_mbar[write_stage];
                        auto tma_copy_a
                            = params.tma_load_a.with(*recast_ptr<typename KT::ProducerBarrierType>(&ab_full_barrier));
                        constexpr int kStage = decltype(write_stage)::value;
                        if constexpr (kStage < kPreWaitBStages)
                        {
                            cute::copy(tma_copy_a, tAgA(_, _, _, write_stage), tAsA(_, _, _, write_stage));
                        }
                        else
                        {
                            ab_empty_mbar[write_stage].wait(phase);
                            cute::copy(tma_copy_a, tAgA(_, _, _, write_stage), tAsA(_, _, _, write_stage));
                            auto tma_copy_b = params.tma_load_b.with(
                                *recast_ptr<typename KT::ProducerBarrierType>(&ab_full_barrier));
                            cute::copy(tma_copy_b, tBgB(_, _, _, write_stage), tBsB(_, _, _, write_stage));
                            ab_full_mbar[write_stage].arrive_and_expect_tx(KT::TmaABTransactionBytes);
                        }
                    });
                phase ^= 1; // flip phase
                k_tile_idx = KT::AB_Stages;
            }
        }
        for (; k_tile_idx < k_tile_count; k_tile_idx += KT::AB_Stages)
        {
            cute::for_each(cute::make_int_sequence<KT::AB_Stages>{},
                [&](auto write_stage)
                {
                    ab_empty_mbar[write_stage].wait(phase);
                    auto& ab_full_barrier = ab_full_mbar[write_stage];
                    auto tma_copy_a
                        = params.tma_load_a.with(*recast_ptr<typename KT::ProducerBarrierType>(&ab_full_barrier));
                    cute::copy(tma_copy_a, tAgA(_, _, _, k_tile_idx + write_stage), tAsA(_, _, _, write_stage));
                    auto tma_copy_b
                        = params.tma_load_b.with(*recast_ptr<typename KT::ProducerBarrierType>(&ab_full_barrier));
                    cute::copy(tma_copy_b, tBgB(_, _, _, k_tile_idx + write_stage), tBsB(_, _, _, write_stage));
                    ab_full_mbar[write_stage].arrive_and_expect_tx(KT::TmaABTransactionBytes);
                });
            phase ^= 1; // flip phase
        }
    }

    // Before the PDL wait, the launch's first tile only: the B (weight) copy of its
    // first kPreWaitBStages stages. Each stage's full barrier is armed for the whole
    // stage (TmaABTransactionBytes: A and B) before its B copy is issued, so it cannot
    // complete until load_ab() has issued the A copy after the wait and that has
    // landed too. No empty-barrier wait is needed: the barriers were initialised in
    // this launch and the producer's first-round wait (parity 1) passes at once. The
    // same holds for the store_empty wait union-D instances take before a tile's loads
    // (smem D aliases smem A/B there), so writing smem B here is safe for them too.
    template <class BlkCoord>
    CUTE_DEVICE static void load_b_first(
        Params const& params, SharedStorage& shared_storage, BlkCoord const& blk_coord)
    {
        using X = Underscore;
        auto N = cute::get<1>(params.problem_shape);
        auto K = cute::get<2>(params.problem_shape);
        auto L = cute::get<3>(params.problem_shape);
        auto mB_nkl = params.tma_load_b.get_tma_tensor(make_shape(N, K, L));
        auto gB_nkl = local_tile(
            mB_nkl, typename KT::TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{}); // (BLK_N,BLK_K,n,k,l)
        auto block_tma_b = params.tma_load_b.get_slice(0);
        auto gB = gB_nkl(_, _, cute::get<1>(blk_coord), _, cute::get<2>(blk_coord));
        auto tBgB = block_tma_b.partition_S(gB); // (TMA,TMA_N,TMA_K,k)
        auto sB_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B.begin()), typename KT::SmemLayoutB{});
        auto sB = as_position_independent_swizzle_tensor(sB_); // (BLK_N,BLK_K,PIPE)
        auto tBsB = block_tma_b.partition_D(sB);               // (TMA,TMA_N,TMA_K,PIPE)
        auto mbarriers = get_mbarriers(shared_storage);
        auto& ab_full_mbar = cute::get<0>(mbarriers);
        cute::for_each(cute::make_int_sequence<kPreWaitBStages>{},
            [&](auto stage)
            {
                auto& ab_full_barrier = ab_full_mbar[stage];
                ab_full_barrier.arrive_and_expect_tx(KT::TmaABTransactionBytes);
                auto tma_copy_b
                    = params.tma_load_b.with(*recast_ptr<typename KT::ProducerBarrierType>(&ab_full_barrier));
                cute::copy(tma_copy_b, tBgB(_, _, _, stage), tBsB(_, _, _, stage));
            });
    }

    // The SF-load warp's counterpart: span 0's SFB copy, the full barrier armed for
    // SFA and SFB (TmaSFTransactionBytes) before it. SF_Stages is 1, so this is the
    // span the first kNumTileKPerSF k-tiles use.
    template <class BlkCoord>
    CUTE_DEVICE static void load_sfb_first(
        Params const& params, SharedStorage& shared_storage, BlkCoord const& blk_coord)
    {
        using X = Underscore;
        auto mSFB_nkl = params.tma_load_sfb.get_tma_tensor(shape(KT::deduce_sfb_layout(params.problem_shape)));
        auto gSFB_nkl = local_tile(
            mSFB_nkl, typename KT::ScaleTileShape{}, make_coord(_, _, _), Step<X, _1, _1>{}); // (TILE_N,TILE_K,n,k,l)
        auto block_tma_sfb = params.tma_load_sfb.get_slice(0);
        auto gSFB = gSFB_nkl(_, _, cute::get<1>(blk_coord), _, cute::get<2>(blk_coord));
        auto tBgSFB = block_tma_sfb.partition_S(gSFB); // (TMA,TMA_N,TMA_K,k)
        auto sSFB_
            = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB.begin()), typename KT::SmemLayoutSFB{});
        auto sSFB = as_position_independent_swizzle_tensor(sSFB_); // (BLK_N,BLK_K,PIPE)
        auto tBsSFB = block_tma_sfb.partition_D(sSFB);             // (TMA,TMA_N,TMA_K,PIPE)
        auto mbarriers = get_mbarriers(shared_storage);
        auto& sf_full_mbar = cute::get<2>(mbarriers);
        sf_full_mbar[0].arrive_and_expect_tx(KT::TmaSFTransactionBytes);
        auto tma_copy_sfb = params.tma_load_sfb.with(*recast_ptr<typename KT::ProducerBarrierType>(&sf_full_mbar[0]));
        cute::copy(tma_copy_sfb, tBgSFB(_, _, _, 0), tBsSFB(_, _, _, Int<0>{}));
    }

    CUTE_DEVICE
    static void mma(SharedStorage& shared_storage, int32_t sf_tile_count, int32_t k_tile_count, uint32_t& sf_phase,
        uint32_t& ab_phase, uint32_t& store_phase, Params const& params = Params{}, int m_block = 0, int n_block = 0,
        int group = 0 FSO_PT_PARAM)
    {
        [[maybe_unused]] int thread_idx = int(threadIdx.x);

        auto sA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_A.begin()),
            typename KT::SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
        auto sB_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B.begin()), typename KT::SmemLayoutB{});
        auto sSFA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()),
            typename KT::SmemLayoutSFA{}); // (BLK_M,BLK_K,PIPE)
        auto sSFB_
            = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB.begin()), typename KT::SmemLayoutSFB{});
        auto sA = as_position_independent_swizzle_tensor(sA_);     // (BLK_M,BLK_K,PIPE)
        auto sB = as_position_independent_swizzle_tensor(sB_);     // (BLK_N,BLK_K,PIPE)
        auto sSFA = as_position_independent_swizzle_tensor(sSFA_); // (BLK_M,BLK_K,PIPE)
        auto sSFB = as_position_independent_swizzle_tensor(sSFB_); // (BLK_N,BLK_K,PIPE)

        typename KT::TiledMma mma;
        auto tile_shape_mnk = tile_shape(mma);
        auto thr_mma = mma.get_thread_slice(thread_idx);
        auto accum = partition_fragment_C(mma, cute::take<0, 2>(typename KT::TileShape{})); // (MMA,MMA_M,MMA_N)
        // Allocate fragments and descriptors
        auto tCrA = thr_mma.partition_fragment_A(sA(_, _, Int<0>{})); // (MMA,MMA_M,MMA_K)
        auto tCrB = thr_mma.partition_fragment_B(sB(_, _, Int<0>{})); // (MMA,MMA_N,MMA_K)

        // A
        auto s2r_copy_A = make_tiled_copy_A(typename KT::SmemCopyAtomA{}, mma);
        auto s2r_thr_copy_A = s2r_copy_A.get_thread_slice(thread_idx);
        // (((_16,_2,_2,_2),(_16,_1)),(_4,_4,(_1,_2))):(((_128,_16,_2048,_0),(_1,_0)),(_4096,_32,(_0,_16384)))
        auto tXsA = s2r_thr_copy_A.partition_S(sA); // (CPY,CPY_M,CPY_K,PIPE)
        auto tXrA = s2r_thr_copy_A.retile_D(tCrA);  // (CPY,CPY_M,CPY_K)
        // B

        auto s2r_copy_B = make_tiled_copy_B(typename KT::SmemCopyAtomB{}, mma);
        auto s2r_thr_copy_B = s2r_copy_B.get_thread_slice(thread_idx);
        // (((_8,_2,_2,_2,_2),(_16,_1)),(_4,_4,(_1,_2))):(((_128,_16,_2048,_0,_1024),(_1,_0)),(_4096,_32,(_0,_16384)))
        auto tXsB = s2r_thr_copy_B.partition_S(sB); // (CPY,CPY_M,CPY_K,PIPE)
        auto tXrB = s2r_thr_copy_B.retile_D(tCrB);  // (CPY,CPY_M,CPY_K)

        auto s2r_copy_SFA = make_tiled_copy_impl(
            typename KT::SmemCopyAtomSF{}, KT::get_layoutSFA_TV(mma), make_shape(size<0>(tile_shape(mma)), _1{}));
        auto s2r_thr_copy_SFA = s2r_copy_SFA.get_thread_slice(thread_idx);
        auto tXsSFA = s2r_thr_copy_SFA.partition_S(sSFA);                        // (CPY,CPY_M,CPY_K,PIPE)
        auto tCrSFA = KT::partition_fragment_SFA(sSFA(_, _, Int<0>{}), thr_mma); // (MMA,MMA_M,MMA_K)
        auto tXrSFA = s2r_thr_copy_SFA.retile_D(tCrSFA);
        auto tCrSFA_frg = KT::transform_fragment_for_qmma(tCrSFA);

        auto s2r_copy_SFB = make_tiled_copy_impl(
            typename KT::SmemCopyAtomSF{}, KT::get_layoutSFB_TV(mma), make_shape(size<1>(tile_shape(mma)), _1{}));
        auto s2r_thr_copy_SFB = s2r_copy_SFB.get_thread_slice(thread_idx);
        auto tXsSFB = s2r_thr_copy_SFB.partition_S(sSFB);                        // (CPY,CPY_M,CPY_K,PIPE)
        auto tCrSFB = KT::partition_fragment_SFB(sSFB(_, _, Int<0>{}), thr_mma); // (MMA,MMA_N,MMA_K)
        auto tXrSFB = s2r_thr_copy_SFB.retile_D(tCrSFB);
        auto tCrSFB_frg = KT::transform_fragment_for_qmma(tCrSFB);

        cute::clear(accum);
        auto mbarriers = get_mbarriers(shared_storage);
        auto& ab_full_mbar = cute::get<0>(mbarriers);
        auto& ab_empty_mbar = cute::get<1>(mbarriers);
        auto& sf_full_mbar = cute::get<2>(mbarriers);
        auto& sf_empty_mbar = cute::get<3>(mbarriers);
        auto& store_full_mbar = cute::get<4>(mbarriers);
        auto& store_empty_mbar = cute::get<5>(mbarriers);
        for (int32_t sf_tile_idx = 0; sf_tile_idx < sf_tile_count - 1; ++sf_tile_idx)
        {
            sf_full_mbar[0].wait(sf_phase);
#if defined(FSO_PROLOGUE_TRACE)
            if (sf_tile_idx == 0 && thread_idx == 0)
            {
                prologue_trace::put(trc_first_tile, prologue_trace::kFirstOperands);
                if (trc_cta_first != nullptr)
                {
                    trc_cta_first[17] = prologue_trace::gtimer();
                }
            }
#endif
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
                    ab_phase ^= 1; // flip phase
                });
            sf_phase ^= 1;         // flip phase
        }

        sf_full_mbar[0].wait(sf_phase);
#if defined(FSO_PROLOGUE_TRACE)
        if (sf_tile_count == 1 && thread_idx == 0)
        {
            prologue_trace::put(trc_first_tile, prologue_trace::kFirstOperands);
            if (trc_cta_first != nullptr)
            {
                trc_cta_first[17] = prologue_trace::gtimer();
            }
        }
#endif
        cute::copy(s2r_copy_SFA, tXsSFA(_, _, _, Int<0>{}), tXrSFA);
        cute::copy(s2r_copy_SFB, tXsSFB(_, _, _, Int<0>{}), tXrSFB);
        sf_empty_mbar[0].arrive();

        // Last SF span. k_tile_count is aligned to AB_Stages (NOT to the
        // 4-k-tile SF span), so on short-K shapes (e.g. K=768 with
        // Stages=2: 6 real k-tiles) the final span consumes only
        // last_batches * AB_Stages of its 4 slices instead of padding to a
        // full span of zero-fill iterations. Dense shapes (every K a
        // multiple of 512) always take the full-span branch bit-for-bit.
        int32_t const last_batches
            = (k_tile_count - KT::kNumTileKPerSF * (sf_tile_count - 1) + KT::AB_Stages - 1) / KT::AB_Stages;
        auto run_last_span = [&](auto num_batches)
        {
            cute::for_each(cute::make_int_sequence<decltype(num_batches)::value>{},
                [&](auto iter)
                {
                    cute::for_each(cute::make_int_sequence<KT::AB_Stages>{},
                        [&](auto read_stage)
                        {
                            ab_full_mbar[read_stage].wait(ab_phase);
                            cute::copy(s2r_copy_A, tXsA(_, _, _, read_stage), tXrA);
                            cute::copy(s2r_copy_B, tXsB(_, _, _, read_stage), tXrB);
                            ab_empty_mbar[read_stage].arrive();
                            if constexpr (iter == decltype(num_batches)::value - 1
                                && read_stage == KT::AB_Stages - 1)
                            {
                                cutlass::arch::NamedBarrier::sync(
                                    KT::kNumMathThreads, 0); // wait for all threads to finish loading
                            }
                            auto tCrSFA_stage = tCrSFA_frg(_, _, _, iter * KT::AB_Stages + read_stage);
                            auto tCrSFB_stage = tCrSFB_frg(_, _, _, iter * KT::AB_Stages + read_stage);
                            cute::gemm(
                                mma, make_zip_tensor(tCrA, tCrSFA_stage), make_zip_tensor(tCrB, tCrSFB_stage), accum);
                        });
                    ab_phase ^= 1; // flip phase
                });
        };
        if constexpr (KT::kNumStagePerSF == 1)
        {
            run_last_span(cute::Int<1>{});
        }
        else
        {
            if (last_batches == KT::kNumStagePerSF)
            {
                run_last_span(cute::Int<KT::kNumStagePerSF>{});
            }
            else
            {
                run_last_span(cute::Int<1>{});
            }
        }
        sf_phase ^= 1;         // flip phase
#if defined(FSO_PROLOGUE_TRACE)
        if (thread_idx == 0)
        {
            prologue_trace::put(trc_first_tile, prologue_trace::kMainloopDone);
            if (trc_cta_first != nullptr)
            {
                trc_cta_first[18] = prologue_trace::gtimer();
            }
        }
#endif

        // epilogue
        auto accum_frg = recast<Array<typename KT::ElementAccum, 2>>(accum);
        auto epi = make_fragment_like<typename KT::ElementD>(accum);
        auto epi_frg = recast<Array<typename KT::ElementD, 2>>(epi);
        cutlass::NumericArrayConverter<typename KT::ElementD, typename KT::ElementAccum, 2> converter;
        cute::for_each(
            cute::make_int_sequence<cute::size(epi_frg)>{}, [&](auto i) { epi_frg(i) = converter(accum_frg(i)); });

        auto tiled_copy_C_atom = make_tiled_copy_C_atom(typename KT::CopyAtomC{}, mma);

        auto tiled_copy_r2s
            = make_tiled_copy_S(cute::Copy_Atom<typename KT::CopyOpR2S, typename KT::ElementD>{}, tiled_copy_C_atom);
        auto thr_copy_r2s = tiled_copy_r2s.get_slice(thread_idx);

        auto sD_epi_ = make_tensor(make_smem_ptr(shared_storage.tensors.store.smem_D.begin()),
            typename KT::SmemLayoutD{});                                     // (BLK_M,BLK_K,PIPE)
        auto sD_epi = cute::as_position_independent_swizzle_tensor(sD_epi_); // (EPI_TILE_M,EPI_TILE_N,PIPE_D)
        auto tRS_rD = thr_copy_r2s.retile_S(epi);
        auto tRS_sD = thr_copy_r2s.partition_D(sD_epi);

        constexpr bool kScatterWarpOwnsStore = ScatterCombineEpi && ScatterInStoreWarp;
        if constexpr (KT::kSeparateSmemD && !FusedSwiGLUEpi && (!ScatterCombineEpi || kScatterWarpOwnsStore))
        {
            // Dedicated store smem: only this epilogue write must wait for
            // the previous tile's TMA store to drain — the K-mainloop above
            // already overlapped it. The fused epilogue has no TMA store warp at
            // all, so nothing would ever arrive at this barrier; the math warps
            // protect the staging buffer between tiles with their own barrier
            // below instead.
            store_empty_mbar[0].wait(store_phase);
            store_phase ^= 1;
        }
        copy(tiled_copy_r2s, tRS_rD, tRS_sD(_, _, _, Int<0>{}));
        if constexpr (kScatterWarpOwnsStore)
        {
            cutlass::arch::NamedBarrier::sync(KT::kNumMathThreads, 0);
            store_full_mbar[0].arrive();
        }
        else if constexpr (ScatterCombineEpi)
        {
            cutlass::arch::NamedBarrier::sync(KT::kNumMathThreads, 0);
            int const rows_valid = params.grouped_layout != nullptr
                ? params.grouped_layout[group]
                : cute::get<0>(params.problem_shape);
            fused_combine::store_tile<KT::kTileM, KT::kTileN, KT::kNumMathThreads>(sD_epi(_, _, Int<0>{}),
                params.out_tokens, params.row_map, params.weight_of_slot, thread_idx, group,
                m_block * KT::kTileM, n_block * KT::kTileN, rows_valid, params.m_cap,
                cute::get<1>(params.problem_shape));
            cutlass::arch::NamedBarrier::sync(KT::kNumMathThreads, 0);
        }
        else if constexpr (FusedSwiGLUEpi)
        {
            // The staged tile is this tile's whole output, so the math warps
            // consume it themselves: SwiGLU, requantize, store. Two barriers, one
            // to publish the staging and one to protect it from the next tile's
            // staging, and the TMA store warp has nothing to do at all.
            cutlass::arch::NamedBarrier::sync(KT::kNumMathThreads, 0);
            int const rows_valid = params.grouped_layout != nullptr
                ? params.grouped_layout[group]
                : cute::get<0>(params.problem_shape);
            fused_swiglu::store_tile<KT::kTileM, KT::kTileN, KT::kNumMathThreads>(sD_epi(_, _, Int<0>{}),
                params.out_fp8, params.out_sf, thread_idx, group, m_block * KT::kTileM, n_block * KT::kTileN,
                rows_valid, params.m_cap, params.inter);
            cutlass::arch::NamedBarrier::sync(KT::kNumMathThreads, 0);
        }
        else
        {
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(KT::kNumMathThreads, 0); // sync before epilogue
            store_full_mbar[0].arrive();
        }
#if defined(FSO_PROLOGUE_TRACE)
        if constexpr (FusedSwiGLUEpi || (ScatterCombineEpi && !kScatterWarpOwnsStore))
        {
            if (thread_idx == 0)
            {
                prologue_trace::put(trc_first_tile, prologue_trace::kEpilogueDone);
            }
        }
#endif
    }

    template <class BlkCoord>
    CUTE_DEVICE static void scatter(
        Params const& params, SharedStorage& shared_storage, BlkCoord const& blk_coord, uint32_t& phase FSO_PT_PARAM)
    {
        auto mbarriers = get_mbarriers(shared_storage);
        auto& store_full_mbar = cute::get<4>(mbarriers);
        auto& store_empty_mbar = cute::get<5>(mbarriers);
        store_full_mbar[0].wait(phase);
        auto sD_epi_ = make_tensor(
            make_smem_ptr(shared_storage.tensors.store.smem_D.begin()), typename KT::SmemLayoutD{});
        auto sD_epi = cute::as_position_independent_swizzle_tensor(sD_epi_);
        int const group = cute::get<2>(blk_coord);
        int const rows_valid
            = params.grouped_layout != nullptr ? params.grouped_layout[group] : cute::get<0>(params.problem_shape);
        int const scatter_tid
            = ((int(threadIdx.x) / 32 == KT::kNumMathWarps) ? 0 : 32) + (int(threadIdx.x) & 31);
        fused_combine::store_tile<KT::kTileM, KT::kTileN, kNumScatterThreads>(sD_epi(_, _, Int<0>{}),
            params.out_tokens, params.row_map, params.weight_of_slot, scatter_tid, group,
            cute::get<0>(blk_coord) * KT::kTileM, cute::get<1>(blk_coord) * KT::kTileN, rows_valid, params.m_cap,
            cute::get<1>(params.problem_shape));
        cutlass::arch::NamedBarrier::sync(kNumScatterThreads, kScatterBarrierId);
        if (threadIdx.x == KT::kNumMathThreads)
        {
            store_empty_mbar[0].arrive();
#if defined(FSO_PROLOGUE_TRACE)
            prologue_trace::put(trc_first_tile, prologue_trace::kEpilogueDone);
#endif
        }
        phase ^= 1;
    }

    template <class BlkCoord>
    CUTE_DEVICE static void store(
        Params const& params, SharedStorage& shared_storage, BlkCoord const& blk_coord, uint32_t& phase FSO_PT_PARAM)
    {
        auto mbarriers = get_mbarriers(shared_storage);
        auto& ab_full_mbar = cute::get<0>(mbarriers);
        auto& ab_empty_mbar = cute::get<1>(mbarriers);
        auto& sf_full_mbar = cute::get<2>(mbarriers);
        auto& sf_empty_mbar = cute::get<3>(mbarriers);
        auto& store_full_mbar = cute::get<4>(mbarriers);
        auto& store_empty_mbar = cute::get<5>(mbarriers);
        store_full_mbar[0].wait(phase);
        using EpilogueTile = typename KT::EpilogueTile_MN;
        auto M = cute::get<0>(params.problem_shape);
        auto N = cute::get<1>(params.problem_shape);
        auto K = cute::get<2>(params.problem_shape);
        auto L = cute::get<3>(params.problem_shape);
        auto mD_mnl = params.tma_store_d.get_tma_tensor(make_shape(M, N, L));
        auto gD_mnl = local_tile(
            mD_mnl, typename KT::TileShape{}, make_coord(_, _, _), Step<_1, _1, X>{}); // (BLK_M,BLK_N,m,n,l)
        auto m_coord = cute::get<0>(blk_coord);
        auto n_coord = cute::get<1>(blk_coord);
        auto l_coord = cute::get<2>(blk_coord);
        auto gD = gD_mnl(_, _, m_coord, n_coord, l_coord);
        auto gD_epi = flat_divide(gD, EpilogueTile{}); // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

        auto block_tma_d = params.tma_store_d.get_slice(Int<0>{});
        auto sD_epi_ = make_tensor(make_smem_ptr(shared_storage.tensors.store.smem_D.begin()),
            typename KT::SmemLayoutD{});                                     // (BLK_M,BLK_K,PIPE)
        auto sD_epi = cute::as_position_independent_swizzle_tensor(sD_epi_); // (EPI_TILE_M,EPI_TILE_N,PIPE_D)
        auto bSG_sD = block_tma_d.partition_S(sD_epi);                       // (TMA,TMA_M,TMA_K, PIP)
        auto bSG_gD = block_tma_d.partition_D(gD_epi);                       // (TMA,TMA_M,TMA_K, EPI_M, EPI_N)

        for (int epi_n = 0; epi_n < size<3>(bSG_gD); ++epi_n)
        {
            for (int epi_m = 0; epi_m < size<2>(bSG_gD); ++epi_m)
            {
                cute::copy(params.tma_store_d, bSG_sD(_, _, _, Int<0>{}), bSG_gD(_, _, _, epi_m, epi_n));
            }
        }
        cute::tma_store_arrive();
#if defined(FSO_PROLOGUE_TRACE)
        prologue_trace::put(trc_first_tile, prologue_trace::kEpilogueDone);
#endif
        cute::tma_store_wait<0>();
        store_empty_mbar[0].arrive();
        phase ^= 1; // flip phase
    }

    // Grouped plain epilogue: store only the rows that hold output. The masked layout
    // gives every expert m_cap rows and a tile covers TileM of them whatever the
    // expert holds, so the last m-block of every expert is partly padding -- at decode
    // a (16, 64) tile carries one or two routed rows of sixteen -- and the TMA store
    // wrote the whole box. A tile whose rows are all valid keeps the TMA store; a
    // partial tile is written by the whole store warp as 16-byte stores of its valid
    // rows only. Rows at or past masked_m[g] are undefined by the op's contract and are
    // no longer written; the valid rows are bit-identical.
    // Measured (RTX 5090, cold weights, two interleaved passes per arm, run
    // sm120_r2_p3_20260929; valid rows bit-identical under every grouped instance):
    // Family C down (K = 512) -2.1 to -2.7 % at M = 16-64 and -3.4 / -7.4 / -6.4 % at
    // M = 2048 / 4096 / 8192; Family B down (K = 768) -2.5 % at M = 16-64 and
    // -4.2 / -6.2 % at M = 1024 / 2048; the routed layers up to -1.5 % (Family C) and
    // -1.1 % (Family B) at M = 16-128 and up to -2.9 % / -1.7 % from M = 1024, with
    // +0.4 % at M = 512 in both families. Per-tile traces: at decode the TMA store of a
    // mostly-padding tile waited 2.1-2.5 us per tile in the SM's TMA queue behind the
    // loads (0.2 us as direct stores); at M = 4096 each tile's k-loop shortens by 0.4 us
    // because its loads arrive sooner.
    template <class BlkCoord>
    CUTE_DEVICE static void store_rows(Params const& params, SharedStorage& shared_storage,
        BlkCoord const& blk_coord, uint32_t& phase, bool elected FSO_PT_PARAM)
    {
        auto mbarriers = get_mbarriers(shared_storage);
        auto& store_full_mbar = cute::get<4>(mbarriers);
        auto& store_empty_mbar = cute::get<5>(mbarriers);
        auto M = cute::get<0>(params.problem_shape);
        auto N = cute::get<1>(params.problem_shape);
        auto L = cute::get<3>(params.problem_shape);
        int const m_coord = cute::get<0>(blk_coord);
        int const n_coord = cute::get<1>(blk_coord);
        int const l_coord = cute::get<2>(blk_coord);
        // Loaded before the wait so that its latency overlaps it; masked_m is older than
        // the PDL wait every warp has already passed.
        int const rows_valid = params.grouped_layout != nullptr ? params.grouped_layout[l_coord] : int(M);
        int const rows = rows_valid - m_coord * KT::kTileM;
        store_full_mbar[0].wait(phase);
        auto sD_epi_ = make_tensor(make_smem_ptr(shared_storage.tensors.store.smem_D.begin()),
            typename KT::SmemLayoutD{});                                     // (BLK_M,BLK_N,PIPE)
        auto sD_epi = cute::as_position_independent_swizzle_tensor(sD_epi_); // (EPI_TILE_M,EPI_TILE_N,PIPE_D)
        if (rows >= KT::kTileM)
        {
            if (elected)
            {
                using EpilogueTile = typename KT::EpilogueTile_MN;
                auto mD_mnl = params.tma_store_d.get_tma_tensor(make_shape(M, N, L));
                auto gD_mnl = local_tile(
                    mD_mnl, typename KT::TileShape{}, make_coord(_, _, _), Step<_1, _1, X>{}); // (BLK_M,BLK_N,m,n,l)
                auto gD = gD_mnl(_, _, m_coord, n_coord, l_coord);
                auto gD_epi = flat_divide(gD, EpilogueTile{}); // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
                auto block_tma_d = params.tma_store_d.get_slice(Int<0>{});
                auto bSG_sD = block_tma_d.partition_S(sD_epi); // (TMA,TMA_M,TMA_K, PIP)
                auto bSG_gD = block_tma_d.partition_D(gD_epi); // (TMA,TMA_M,TMA_K, EPI_M, EPI_N)
                for (int epi_n = 0; epi_n < size<3>(bSG_gD); ++epi_n)
                {
                    for (int epi_m = 0; epi_m < size<2>(bSG_gD); ++epi_m)
                    {
                        cute::copy(params.tma_store_d, bSG_sD(_, _, _, Int<0>{}), bSG_gD(_, _, _, epi_m, epi_n));
                    }
                }
                cute::tma_store_arrive();
#if defined(FSO_PROLOGUE_TRACE)
                prologue_trace::put(trc_first_tile, prologue_trace::kEpilogueDone);
#endif
                cute::tma_store_wait<0>();
                store_empty_mbar[0].arrive();
            }
        }
        else
        {
            using ElementD = typename KT::ElementD;
            constexpr int kVec = 16 / int(sizeof(ElementD)); // elements per 16-byte store
            constexpr int kVecPerRow = KT::kTileN / kVec;
            static_assert(KT::kTileN % kVec == 0, "a tile row must be a whole number of 16-byte vectors");
            int const lane = int(threadIdx.x) & 31;
            ElementD* const dst = params.ptr_d + int64_t(l_coord) * params.stride_d
                + int64_t(m_coord * KT::kTileM) * params.ld_d + n_coord * KT::kTileN;
            // The swizzle permutes whole 16-byte chunks within a 128-byte row, so the
            // kVec elements from a kVec-aligned column are contiguous in shared memory.
            for (int q = lane; q < rows * kVecPerRow; q += 32)
            {
                int const r = q / kVecPerRow;
                int const c = (q - r * kVecPerRow) * kVec;
                uint4 const v = *reinterpret_cast<uint4 const*>(&sD_epi(r, c, Int<0>{}));
                *reinterpret_cast<uint4*>(dst + int64_t(r) * params.ld_d + c) = v;
            }
            __syncwarp();
            if (elected)
            {
#if defined(FSO_PROLOGUE_TRACE)
                prologue_trace::put(trc_first_tile, prologue_trace::kEpilogueDone);
#endif
                store_empty_mbar[0].arrive();
            }
        }
        phase ^= 1; // flip phase
    }

    CUTE_DEVICE
    void operator()(Params const& params, char* smem_buf)
    {
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
        int warp_idx = canonical_warp_idx_sync();
        int lane_predicate = cute::elect_one_sync();
        bool is_tma_thread = warp_idx == 0 && lane_predicate;
#if defined(FSO_PROLOGUE_TRACE)
        constexpr int kTrcSlot = FusedSwiGLUEpi ? 0 : (ScatterCombineEpi ? 2 : 1);
        uint64_t* const trc
            = params.trace != nullptr ? params.trace + kTrcSlot * prologue_trace::kSlotWords : nullptr;
        int const trc_sel = trc == nullptr ? -1 : (blockIdx.x == 0 ? 0 : (blockIdx.x + 1 == gridDim.x ? 1 : -1));
        uint64_t* const trc_det = trc_sel >= 0 ? trc + trc_sel * prologue_trace::kNumPts * 2 : nullptr;
        uint64_t* const trc_cta = (trc != nullptr && blockIdx.x < prologue_trace::kMaxCtas)
            ? trc + prologue_trace::kCtaBase + blockIdx.x * prologue_trace::kCtaWords
            : nullptr;
        if (threadIdx.x == 0)
        {
            prologue_trace::put(trc_det, prologue_trace::kEntry);
            if (trc_cta != nullptr)
            {
                trc_cta[0] = prologue_trace::gtimer();
                trc_cta[4] = prologue_trace::smid();
            }
            if (trc_det != nullptr)
            {
                trc[prologue_trace::kMetaBase + trc_sel] = prologue_trace::smid();
                trc[prologue_trace::kMetaBase + 2] = gridDim.x;
                trc[prologue_trace::kMetaBase + 3] = blockDim.x;
            }
        }
#endif

        // Grouped path: this thread's grouped_layout counts (the slots it owns in the
        // prefix scan below) are loaded here, at kernel entry, and first used after the
        // barrier setup, so their latency overlaps the descriptor prefetch and the
        // mbarrier init instead of following them. They are grandparent-or-older data,
        // like everything read before cudaGridDependencySynchronize (see the PDL note).
        int prefix_cnt[kScanPer];
#pragma unroll
        for (int j = 0; j < kScanPer; ++j)
        {
            prefix_cnt[j] = 0;
        }
        if constexpr (KT::kGroupedLayoutSmem)
        if (int const L = cute::get<3>(params.problem_shape);
            params.grouped_layout != nullptr && L <= kGroupedLayoutSmemCap)
        {
            int const num_blocks = L / params.group_block;
#pragma unroll
            for (int j = 0; j < kScanPer; ++j)
            {
                int const slot = int(threadIdx.x) * kScanPer + j;
                if (slot < L)
                {
                    int g = slot;
                    if (params.group_perm != 1)
                    {
                        int const block = slot / params.group_block;
                        int const local = slot - block * params.group_block;
                        g = ((block * params.group_perm) % num_blocks) * params.group_block + local;
                    }
                    prefix_cnt[j] = params.grouped_layout[g];
                }
            }
        }

        if (is_tma_thread)
        {
            prefetch_tma_descriptors(params);
        }
        __syncthreads();
        FSO_PT_STAMP(kPrefetchSync);

        auto mbarriers = get_mbarriers(shared_storage);
        auto& ab_full_mbar = cute::get<0>(mbarriers);
        auto& ab_empty_mbar = cute::get<1>(mbarriers);
        auto& sf_full_mbar = cute::get<2>(mbarriers);
        auto& sf_empty_mbar = cute::get<3>(mbarriers);
        auto& store_full_mbar = cute::get<4>(mbarriers);
        auto& store_empty_mbar = cute::get<5>(mbarriers);
        // init barriers
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
        FSO_PT_STAMP(kBarrierSync);

        auto M = cute::get<0>(params.problem_shape);
        auto N = cute::get<1>(params.problem_shape);
        auto K = cute::get<2>(params.problem_shape);
        auto L = cute::get<3>(params.problem_shape);
        // Real k-tiles, padded only to the AB ring width (per-tile constant).
        // The SF span count follows from the padded k count; the final span
        // may be partial (see mma). Every dense K is a multiple of 512, so
        // dense sees identical counts to the old sf-span-aligned padding.
        // K % 128 shapes that are not a multiple of the ring width (K=768 on
        // a Stages=4 instance: 6 real tiles, 8 issued) are correct as they
        // are: the padded k-tiles read out-of-bounds coordinates that TMA
        // zero-fills, and their scale bytes are zero-padded by the repack, so
        // they add nothing. A pad-free variant (partial final round, skipping
        // those TMA loads and MMAs) was tried 2026-09-05 and reverted: it gave
        // no layer-level gain on the K=768 MoE shapes and its extra code path
        // in the mainloop cost the large dense MXFP8 tiles 30-100 % at large M,
        // even though dense shapes never took the new branch. Do not retry
        // without re-running the dense tables on every arch that instantiates
        // this template.
        int32_t const k_real_tiles = (K + KT::kTileK - 1) / KT::kTileK;
        int32_t const k_tile_count = (k_real_tiles + KT::AB_Stages - 1) / KT::AB_Stages * KT::AB_Stages;
        int32_t const sf_tile_count = (k_tile_count + KT::kNumTileKPerSF - 1) / KT::kNumTileKPerSF;

        // Stage grouped_layout into shared memory as an INCLUSIVE per-group
        // m-block prefix: smem[g] = sum_{g' <= g} ceil(masked_m[g'] / TileM).
        // The scheduler then binary-searches its group in O(log L) LDS hits
        // instead of the legacy linear walk, whose per-group dependent load
        // chain (~60 ns/group on 5090, regardless of global-vs-smem
        // residence) dominated the whole decode-band kernel at G=128.
        bool layout_is_cumsum = false;
        int32_t* grouped_layout = params.grouped_layout;
        // Instantiations without the smem array (dense) fall through to the
        // legacy global-memory walk, which they never exercise
        // (grouped_layout == nullptr on the dense path).
        if constexpr (KT::kGroupedLayoutSmem)
        if (grouped_layout != nullptr && L <= kGroupedLayoutSmemCap)
        {
            // The prefix is indexed by SLOT, not by expert id: slot s holds group
            // ((s / w * stride) % (L / w)) * w + s % w, with w = group_block. With
            // group_perm == 1 the two coincide and this is the walk in expert-id
            // order the kernel has always done. With a larger stride the launch
            // still covers every group exactly once (the stride is coprime with the
            // block count, chosen host-side), but the tiles a wave of CTAs runs at
            // the same time come from expert ids spread across the whole set rather
            // than from one contiguous run of them -- which matters when the row
            // counts are correlated with the id, because a contiguous block of
            // experts holding few rows otherwise becomes a phase of the launch in
            // which no CTA has enough arithmetic to cover the weight tile it loads.
            //
            // The scan is block-wide. Every thread owns kScanPer consecutive slots (2
            // at 384 threads and the 512-slot cap), whose counts it loaded at kernel
            // entry. The warps that own slots -- the first ceil(L / (32 * kScanPer))
            // -- scan their threads' sums with five shuffles and publish the warp
            // total; after one barrier each of them adds the totals of the warps below
            // it and writes its slots. The other warps only take part in the two
            // barriers. This replaced warp 0 walking the slots 32 at a time with a
            // serial carry while the other eleven warps waited: at L = 128 / 256 that
            // walk was 4 / 8 dependent rounds of shuffles on the critical path of every
            // CTA (run sm120_prologue_20260928).
            // One side effect, measured in the same run: under nvcc 13.0 the 2-CTA
            // (16,128,2) plain instance went from 64 to 69 registers per thread (72
            // allocated), so two of its CTAs no longer leave room on an SM for a
            // 256-thread, 64-register CTA of the layer's combine kernel, which PDL used to
            // start early beside them. That costs the Family B layer 0.2-0.5 % at M = 4 to
            // 128. Every other form of this change that was compiled raised the count the
            // same way; the only one that kept 64 registers, the serial walk unrolled, was
            // no faster than the serial walk.
            static_assert(kScanPer * MaxThreadsPerBlock >= kGroupedLayoutSmemCap, "scan must cover the cap");
            int const lane = threadIdx.x & 31;
            int const slot0 = int(threadIdx.x) * kScanPer;
            bool const warp_has_slots = warp_idx * 32 * kScanPer < L; // warp-uniform
            int blocks_of[kScanPer];
            int own = 0;
#pragma unroll
            for (int j = 0; j < kScanPer; ++j)
            {
                blocks_of[j] = (prefix_cnt[j] + KT::kTileM - 1) / KT::kTileM; // 0 for slots >= L
                own += blocks_of[j];
            }
            int incl = own;
            if (warp_has_slots)
            {
#pragma unroll
                for (int off = 1; off < 32; off <<= 1)
                {
                    int const n = __shfl_up_sync(0xFFFFFFFFu, incl, off);
                    if (lane >= off)
                    {
                        incl += n;
                    }
                }
                if (lane == 31)
                {
                    shared_storage.grouped_scan_warp_total[warp_idx] = incl;
                }
            }
            __syncthreads();
            FSO_PT_STAMP(kStageSync);
            if (warp_has_slots)
            {
                // The totals of the warps below this one (all of which own slots), read
                // as kScanWarps independent broadcast loads; a loop bounded by warp_idx
                // would chain them.
                int run = incl - own;
#pragma unroll
                for (int w = 0; w < kScanWarps; ++w)
                {
                    run += (w < warp_idx) ? shared_storage.grouped_scan_warp_total[w] : 0;
                }
#pragma unroll
                for (int j = 0; j < kScanPer; ++j)
                {
                    run += blocks_of[j];
                    if (slot0 + j < L)
                    {
                        shared_storage.grouped_layout_smem[slot0 + j] = run;
                    }
                }
            }
            __syncthreads();
            FSO_PT_STAMP(kScanSync);
            grouped_layout = shared_storage.grouped_layout_smem;
            layout_is_cumsum = true;
        }

        // Programmatic dependent launch (PDL). Everything above — TMA
        // descriptor prefetch, mbarrier init, the grouped_layout staging +
        // prefix scan — reads only kernel params and grandparent-or-older
        // data (each kernel in the fso MoE chain triggers its dependents
        // only AFTER its own wait returns, so the pre-wait window overlaps
        // the immediate parent alone). In the grouped instantiations each warp role
        // below takes two more steps before it waits.
        //
        // Step 1, the first tile. The role constructs its scheduler and takes the
        // first get_next_block() before its wait. The call reads only kernel
        // parameters and the smem prefix built above (past the smem cap, masked_m
        // itself, the grandparent-or-older data the prefix is built from), and its
        // binary search and swizzle divisions cost 0.6-0.8 us that used to follow
        // the release on every GEMM of the chain (run sm120_prologue_20260928, Q3).
        //
        // Step 2, the weights (kPreWaitBStages > 0). The A/B-load warp issues the B
        // copy of the first tile's first kPreWaitBStages stages, and the SF-load
        // warp the SFB copy of its first span, before its wait; A and SFA follow
        // after it (load_ab / load_sf, `first_tile`). B and SFB are the expert
        // weights and their scale factors, which no kernel of an fso chain writes:
        // they are quantized at load time, never the output of the kernel launched
        // right before this one. A caller whose B operand IS that kernel's output
        // would race here while PDL is on. Each stage's full barrier expects both halves
        // up front, so it completes only once the copy issued after the wait has
        // landed as well.
        //
        // Every role then waits in its own branch, which orders all of its other
        // global accesses (the parent-produced A/SFA operands, the D output, the
        // fused epilogues' outputs) behind parent completion; warps without a role
        // and the idle lanes of the single-thread roles touch no global memory and
        // do not wait. Thread 0 (math warp 0) triggers right after its wait, which
        // lets OUR dependent start its own prologue while we run. Both are no-ops
        // when the launch chain carries no PDL edges. Each role keeps its own
        // scheduler and its own wait because one scheduler built before a single
        // shared wait kept every role's state live across that wait: +2 to +9
        // registers per instance, and 112-160 bytes of spills on the (160,128,2)
        // instances that sit at the 168-register cap (run sm120_r2_p2_20260929).
        //
        // Dense instantiations (kGroupedLayoutSmem == false) keep the production
        // order below: one wait for every thread, then the roles schedule their first
        // tile. Both steps were measured on them too (run sm120_r2_p2_20260929) and
        // moved the dense tables both ways: -1 to -8 % on the wide shapes, but +4 to
        // +14 % (+18 % with step 1 alone) on the narrow-N ones at M <= 64 (wo,
        // gdn_out_proj, shared_gate_up, the Qwen3-30B-A3B wqkv). There only 40-80
        // of the 170 persistent CTAs have a tile; with the first tile
        // scheduled before the wait the idle CTAs leave their SMs sooner, which moves
        // the chained successor's tile CTAs to a different set of SMs, on which the
        // traced wo GEMM's k-loop ran 40 % longer. The grouped launches are capped at
        // their tile bound, so nearly every CTA has a tile and the effect does not arise.
        //
        // Measured in run sm120_r2_p2_20260929 (RTX 5090 card C2 at 2407 MHz, cold-weight
        // bench, two interleaved passes). Against this file's previous version the routed
        // layer at M = 1 went from 33.8 to 32.5 us (Family B) and from 25.2 to 23.9 us (Family
        // C), every decode cell of the layer and kernel tables got faster (-0.2 to -19 %), and
        // the prefill cells moved -0.22 / -0.20 % in the median (worst +0.2 %). Measured before
        // the partial-tile store (store_rows) existed, step 1 alone took the two M = 1 cells from
        // 34.4 / 25.8 to 33.3 / 25.1 us and both steps to 33.0 / 24.4 us. The trace build shows
        // why: the FC2, whose prologue already hides behind the FC1, now starts its first tile at
        // its PDL release, 0.38-0.42 us after the FC1 ends instead of 0.86-1.06 us, and its first
        // scale stage lands 0.7-1.1 us after the FC1 ends instead of 2.9-6.3 us. The FC1 reaches
        // its wait only after the short gather-quantize has ended, so its first tile starts when
        // it did before; its weight stream now shares DRAM with the FC2's early weight copies.
        if constexpr (KT::kGroupedLayoutSmem)
        {
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
                        auto scheduler = Scheduler(M, N, L, grouped_layout, layout_is_cumsum, params.group_perm, params.group_block);
                        bool has = scheduler.get_next_block();
#if defined(FSO_PROLOGUE_TRACE)
                        prologue_trace::put(trc_det, prologue_trace::kLoadFirstNext);
#endif
                        if constexpr (kPreWaitBStages > 0)
                        {
                            if (has)
                            {
                                load_b_first(params, shared_storage,
                                    cute::make_coord(scheduler.m_block_idx, scheduler.n_block_idx, scheduler.current_group_idx));
                            }
                        }
                        cudaGridDependencySynchronize();
                        for (bool first = true; has; has = scheduler.get_next_block(), first = false)
                        {
                            auto blk_coord = cute::make_coord(
                                scheduler.m_block_idx, scheduler.n_block_idx, scheduler.current_group_idx);
                            load_ab(params, shared_storage, blk_coord, k_tile_count, phase, store_phase, first);
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
                        auto scheduler = Scheduler(M, N, L, grouped_layout, layout_is_cumsum, params.group_perm, params.group_block);
                        bool has = scheduler.get_next_block();
                        if constexpr (kPreWaitBStages > 0)
                        {
                            if (has)
                            {
                                load_sfb_first(params, shared_storage,
                                    cute::make_coord(scheduler.m_block_idx, scheduler.n_block_idx, scheduler.current_group_idx));
                            }
                        }
                        cudaGridDependencySynchronize();
                        for (bool first = true; has; has = scheduler.get_next_block(), first = false)
                        {
                            auto blk_coord = cute::make_coord(
                                scheduler.m_block_idx, scheduler.n_block_idx, scheduler.current_group_idx);
                            load_sf(params, shared_storage, blk_coord, sf_tile_count, phase, store_phase, first);
                        }
                    }
                    __syncwarp();
                }
                constexpr int scatter_warp_b = sf_warp_idx + 1;  // the spare fourth TMA warp
                if ((warp_idx == epi_warp_idx || warp_idx == scatter_warp_b) && ScatterCombineEpi && ScatterInStoreWarp)
                {
                    uint32_t phase = 0;
                    auto scheduler = Scheduler(M, N, L, grouped_layout, layout_is_cumsum, params.group_perm, params.group_block);
                    bool has = scheduler.get_next_block();
                    cudaGridDependencySynchronize();
                    FSO_PT_FIRST_DECL;
                    for (; has; has = scheduler.get_next_block())
                    {
                        auto blk_coord
                            = cute::make_coord(scheduler.m_block_idx, scheduler.n_block_idx, scheduler.current_group_idx);
                        scatter(params, shared_storage, blk_coord, phase FSO_PT_ARG(FSO_PT_FIRST(trc_det)));
                        FSO_PT_FIRST_DONE;
                    }
                    __syncwarp();
                }
                if (warp_idx == epi_warp_idx && !FusedSwiGLUEpi && !ScatterCombineEpi)
                {
                    // The whole warp walks the tiles, so that a partial tile's valid rows are
                    // written by all 32 lanes (store_rows).
                    uint32_t phase = 0;
                    auto scheduler = Scheduler(M, N, L, grouped_layout, layout_is_cumsum, params.group_perm, params.group_block);
                    bool has = scheduler.get_next_block();
                    cudaGridDependencySynchronize();
                    FSO_PT_FIRST_DECL;
                    for (; has; has = scheduler.get_next_block())
                    {
                        auto blk_coord = cute::make_coord(
                            scheduler.m_block_idx, scheduler.n_block_idx, scheduler.current_group_idx);
                        store_rows(params, shared_storage, blk_coord, phase, lane_predicate != 0
                            FSO_PT_ARG(FSO_PT_FIRST(trc_det)));
                        FSO_PT_FIRST_DONE;
                    }
                    __syncwarp();
                }
            }
            else
            {
                uint32_t sf_phase = 0;
                uint32_t ab_phase = 0;
                uint32_t store_phase = 1;
                auto scheduler = Scheduler(M, N, L, grouped_layout, layout_is_cumsum, params.group_perm, params.group_block);
                bool has = scheduler.get_next_block();
                FSO_PT_STAMP(kMathFirstNext);
#if defined(FSO_PROLOGUE_TRACE)
                if (threadIdx.x == 0 && trc_cta != nullptr)
                {
                    trc_cta[1] = prologue_trace::gtimer();
                }
#endif
                cudaGridDependencySynchronize();
                if (threadIdx.x == 0)
                {
                    cudaTriggerProgrammaticLaunchCompletion();
                }
                FSO_PT_STAMP(kPdlWait);
#if defined(FSO_PROLOGUE_TRACE)
                if (threadIdx.x == 0 && trc_cta != nullptr)
                {
                    trc_cta[2] = prologue_trace::gtimer();
                    if (has)
                    {
                        trc_cta[3] = prologue_trace::gtimer();
                    }
                }
#endif
                FSO_PT_FIRST_DECL;
                for (; has; has = scheduler.get_next_block())
                {
                    mma(shared_storage, sf_tile_count, k_tile_count, sf_phase, ab_phase, store_phase, params,
                        scheduler.m_block_idx, scheduler.n_block_idx,
                        scheduler.current_group_idx FSO_PT_ARG2(FSO_PT_FIRST(trc_det), FSO_PT_FIRST(trc_cta)));
                    FSO_PT_FIRST_DONE;
                }
            }
        }
        else
        {
            // Programmatic dependent launch (PDL). Everything above — TMA
            // descriptor prefetch, mbarrier init, the grouped_layout staging +
            // prefix scan — reads only kernel params and grandparent-or-older
            // data (each kernel in the fso MoE chain triggers its dependents
            // only AFTER its own wait returns, so the pre-wait window overlaps
            // the immediate parent alone). The wait below orders every
            // subsequent global access (the parent-produced A/SFA operands and
            // the D output buffer) behind parent completion; the trigger then
            // lets OUR dependent start its own prologue while we run. Both are
            // no-ops when the launch chain carries no PDL edges, so dense
            // callers are unaffected.
#if defined(FSO_PROLOGUE_TRACE)
            if (threadIdx.x == 0 && trc_cta != nullptr)
            {
                trc_cta[1] = prologue_trace::gtimer();
            }
#endif
            cudaGridDependencySynchronize();
            if (threadIdx.x == 0)
            {
                cudaTriggerProgrammaticLaunchCompletion();
            }
            FSO_PT_STAMP(kPdlWait);
#if defined(FSO_PROLOGUE_TRACE)
            if (threadIdx.x == 0 && trc_cta != nullptr)
            {
                trc_cta[2] = prologue_trace::gtimer();
            }
#endif

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
                        auto scheduler = Scheduler(M, N, L, grouped_layout, layout_is_cumsum, params.group_perm, params.group_block);
                        FSO_PT_CALLS;
                        while (FSO_PT_NEXT(scheduler.get_next_block(), true, kLoadFirstNext, nullptr))
                        {
                            auto blk_coord = cute::make_coord(
                                scheduler.m_block_idx, scheduler.n_block_idx, scheduler.current_group_idx);
                            load_ab(params, shared_storage, blk_coord, k_tile_count, phase, store_phase);
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
                        auto scheduler = Scheduler(M, N, L, grouped_layout, layout_is_cumsum, params.group_perm, params.group_block);
                        while (scheduler.get_next_block())
                        {
                            auto blk_coord = cute::make_coord(
                                scheduler.m_block_idx, scheduler.n_block_idx, scheduler.current_group_idx);
                            load_sf(params, shared_storage, blk_coord, sf_tile_count, phase, store_phase);
                        }
                    }
                    __syncwarp();
                }
                constexpr int scatter_warp_b = sf_warp_idx + 1;  // the spare fourth TMA warp
                if ((warp_idx == epi_warp_idx || warp_idx == scatter_warp_b) && ScatterCombineEpi && ScatterInStoreWarp)
                {
                    uint32_t phase = 0;
                    auto scheduler = Scheduler(M, N, L, grouped_layout, layout_is_cumsum, params.group_perm, params.group_block);
                    FSO_PT_CALLS;
                    while (FSO_PT_NEXT(scheduler.get_next_block(), false, kLoadFirstNext, nullptr))
                    {
                        auto blk_coord
                            = cute::make_coord(scheduler.m_block_idx, scheduler.n_block_idx, scheduler.current_group_idx);
                        scatter(params, shared_storage, blk_coord, phase FSO_PT_ARG(trc_calls == 1 ? trc_det : nullptr));
                    }
                    __syncwarp();
                }
                if (warp_idx == epi_warp_idx && !FusedSwiGLUEpi && !ScatterCombineEpi)
                {
                    uint32_t phase = 0;
                    if constexpr (KT::kGroupedLayoutSmem)
                    {
                        // Grouped: the whole warp walks the tiles, so that a partial tile's
                        // valid rows are written by all 32 lanes (store_rows).
                        auto scheduler = Scheduler(M, N, L, grouped_layout, layout_is_cumsum, params.group_perm, params.group_block);
                        FSO_PT_CALLS;
                        while (FSO_PT_NEXT(scheduler.get_next_block(), false, kLoadFirstNext, nullptr))
                        {
                            auto blk_coord = cute::make_coord(
                                scheduler.m_block_idx, scheduler.n_block_idx, scheduler.current_group_idx);
                            store_rows(params, shared_storage, blk_coord, phase, lane_predicate != 0
                                FSO_PT_ARG(trc_calls == 1 ? trc_det : nullptr));
                        }
                    }
                    else if (lane_predicate)
                    {
                        auto scheduler = Scheduler(M, N, L, grouped_layout, layout_is_cumsum, params.group_perm, params.group_block);
                        FSO_PT_CALLS;
                        while (FSO_PT_NEXT(scheduler.get_next_block(), false, kLoadFirstNext, nullptr))
                        {
                            auto blk_coord = cute::make_coord(
                                scheduler.m_block_idx, scheduler.n_block_idx, scheduler.current_group_idx);
                            store(params, shared_storage, blk_coord, phase FSO_PT_ARG(trc_calls == 1 ? trc_det : nullptr));
                        }
                    }
                    __syncwarp();
                }
            }
            else
            {
                uint32_t sf_phase = 0;
                uint32_t ab_phase = 0;
                uint32_t store_phase = 1;
                auto scheduler = Scheduler(M, N, L, grouped_layout, layout_is_cumsum, params.group_perm, params.group_block);
                FSO_PT_CALLS;
                while (FSO_PT_NEXT(scheduler.get_next_block(), threadIdx.x == 0, kMathFirstNext,
                    trc_cta != nullptr ? trc_cta + 3 : nullptr))
                {
                    mma(shared_storage, sf_tile_count, k_tile_count, sf_phase, ab_phase, store_phase, params,
                        scheduler.m_block_idx, scheduler.n_block_idx,
                        scheduler.current_group_idx FSO_PT_ARG(trc_calls == 1 ? trc_det : nullptr));
                }
            }
        }
#if defined(FSO_PROLOGUE_TRACE)
        if (trc_cta != nullptr && (threadIdx.x & 31) == 0)
        {
            trc_cta[5 + warp_idx] = prologue_trace::gtimer();
        }
        FSO_PT_STAMP(kMathExit);
#endif
    }
};

#undef FSO_PT_PARAM
#undef FSO_PT_ARG
#undef FSO_PT_ARG2
#undef FSO_PT_CALLS
#undef FSO_PT_NEXT
#undef FSO_PT_FIRST_DECL
#undef FSO_PT_FIRST
#undef FSO_PT_FIRST_DONE
#undef FSO_PT_STAMP

} // namespace sm120_blockscaled_gemm
