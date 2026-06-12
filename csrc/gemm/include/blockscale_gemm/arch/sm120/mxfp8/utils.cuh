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

// SM120 BlockScaled builder for the **MXFP8 1x32** (true OCP MXFP8) path.
//
// Mirror of `SM120BlockScaledBuilder` in `sm120_utils.cuh` but with
// `kSFVecSize=32` (one UE8M0 byte per 32 K-elements) instead of the 1×128
// software aggregation used by the production `linear_fp8` path. Hardware
// atom is unchanged — `mma.sync ... kind::mxf8f6f4` is intrinsically VS=32.
//
// Only kTileSF (number of int32 SF words per K-tile of 128 elements)
// changes: 1 → 4, since each int32 word now covers 4×32 = 128 K-elements
// instead of 4×128 = 512. Kernel iteration structure (`kNumTileKPerSF`,
// `kNumStagePerSF`) is preserved bit-for-bit so the existing
// SM120BlockScaledKernel<KT> template can host this builder unchanged.

#pragma once
#include "cute/atom/mma_atom.hpp"
#include <cuda_runtime.h>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/config.hpp>
#include <cute/int_tuple.hpp>
#include <cute/layout.hpp>
#include <cutlass/cutlass.h>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/device_kernel.h"
#include "cutlass/numeric_conversion.h"

using namespace cute;
using namespace cutlass;

namespace sm120_blockscaled_gemm
{

// Helper: select PermMmaTileN layout based on TileN.
//
// Earlier (C5/C6) used the size-128 layout for both TileN=128 and TileN=64,
// hoping the extra N-positions 64..127 would be a no-op because the
// scheduler partitions at (kTileM, kTileN) granularity. That assumption is
// wrong: R2S (SM90_U32x2_STSM_N) writes register fragments to smem
// unconditionally, the writes target positions 64..127 which lie OUTSIDE
// SmemLayoutD's TileN=64 allocation, and the resulting smem ptr crosses
// alignment-boundary regions in adjacent SharedStorage members → kernel
// fails with cudaErrorMisalignedAddress at runtime (verified end-to-end on
// 170-SM Blackwell 2026-05-10 across many shapes including FORCE_TILE bench).
//
// Fix (C9 follow-up): TileN=64 uses a true size-64 PermMmaTileN. Each of
// the 4 N-warps now covers 16 N-positions instead of 32 (4 atom calls per
// warp instead of 8 → halved register pressure too). Total positions
// covered = 4×16 = 64, exactly matching SmemLayoutD's N extent. The shape
// is _8,_4,_2 / stride _1,_16,_8 so the per-warp pattern stays
// SW128-atom-aligned (each warp's 16 elements form a contiguous run with
// stride 1).
template <int TileN>
struct PermMmaTileNForTileN;
template <> struct PermMmaTileNForTileN<128> { using type = Layout<Shape<_8, _4, _4>, Stride<_1, _32, _8>>; };
template <> struct PermMmaTileNForTileN<64>  { using type = Layout<Shape<_8, _4, _2>, Stride<_1, _16, _8>>; };

// PermMmaTileN tuned for the 8-warp N-major layout used at TileM=16.
// 8 warps in N, atom_N = 8 → each warp covers TileN / 8 N positions
// (2 atoms per warp at TileN=128, 1 atom per warp at TileN=64).
// Shape encodes (lanes-per-N-atom=8, N-warps, atom-iters-in-N).
// Strides chosen so the 8 warps tile contiguously in N and each warp's
// atom iterations stride past the warp-group's footprint.
template <int TileN>
struct PermMmaTileNForTileN_M16;
template <> struct PermMmaTileNForTileN_M16<128> { using type = Layout<Shape<_8, _8, _2>, Stride<_1, _8, _64>>; };
template <> struct PermMmaTileNForTileN_M16<64>  { using type = Layout<Shape<_8, _8, _1>, Stride<_1, _8,  _0>>; };

// Helper: select SmemLayoutAtomD based on TileN.
// Both TileN=128 and TileN=64 use SW128 for 128-byte swizzle alignment.
template <int TileN, class ElementD>
struct SmemLayoutAtomDForTileN;
template <class T> struct SmemLayoutAtomDForTileN<128, T> { using type = GMMA::Layout_K_SW128_Atom<T>; };
template <class T> struct SmemLayoutAtomDForTileN<64, T>  { using type = GMMA::Layout_K_SW128_Atom<T>; };

// Helper: select R2S copy op based on TileN.
// TileN=128: SM90_U32x2_STSM_N (optimized STSM with 32-bit × 2 val/thread)
// TileN=64:  AutoVectorizingCopy (TileN=64 changes register layout; use
//             auto-vectorizing store as fallback. SmemLayoutAtomD is kept at
//             SW128 for alignment; tile_to_shape truncates to TileN=64.)
template <int TileN>
struct CopyOpR2SForTileN;
template <> struct CopyOpR2SForTileN<128> { using type = SM90_U32x2_STSM_N; };
template <> struct CopyOpR2SForTileN<64>  { using type = SM90_U32x2_STSM_N; };

// Helper: select the per-warp PermMmaTileM and TiledMma thread layout
// based on TileM. The hw atom is 16x8x32 (M,N,K), so atom_M=16 is the
// floor. TileM=16 uses 1 warp in M × 8 warps in N (extreme decode
// variant); all other TileM values use 2 warps in M × 4 warps in N.
template <int TileM>
struct WarpLayoutForTileM
{
    using PermMmaTileM = Int<32>;
    using ThrLayoutVMNK = Layout<Shape<_2, _4, _1>, Stride<_4, _1, _0>>;
};
template <>
struct WarpLayoutForTileM<16>
{
    using PermMmaTileM = Int<16>;
    using ThrLayoutVMNK = Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>;
};

// Helper: select the B-fragment smem→reg copy atom based on (TileM, TileN).
// All standard tiles (TileM >= 32) use SM75_U32x4_LDSM_N: each thread
// loads 4 uint32 per call = 16 vals (e4m3 bytes). The TileM=16 + TileN=64
// extreme-decode variant uses 8 N-warps × TileN=64, so each warp covers
// only 8 N-columns × 128 K = 1024 elements = 32 vals/thread total over
// the K-loop. The first-dim partition lands at 8 vals/thread per call,
// which is not a multiple of LDSM_x4's 16 vals/thread atom →
// `TiledNumVal % AtomNumVal != 0` static_assert in CuTe. Switch to
// SM75_U32x2_LDSM_N (8 vals/thread atom) for this configuration only.
template <int TileM, int TileN>
struct SmemCopyAtomBForTileMN
{
    using type = Copy_Atom<SM75_U32x4_LDSM_N, cute::float_e4m3_t>;
};
template <>
struct SmemCopyAtomBForTileMN<16, 64>
{
    using type = Copy_Atom<SM75_U32x2_LDSM_N, cute::float_e4m3_t>;
};

template <int TileM_ = 32, int TileN_ = 128, int Stages_ = 4, int MinBlocksPerSm_ = 1,
    int SchedGroup_ = 16,
    typename PermMmaTileN_ = typename cute::conditional_t<(TileM_ == 16),
        typename PermMmaTileNForTileN_M16<TileN_>::type,
        typename PermMmaTileNForTileN<TileN_>::type>,
    typename SmemLayoutAtomD_ = typename SmemLayoutAtomDForTileN<TileN_, cute::bfloat16_t>::type,
    typename CopyOpR2S_ = typename CopyOpR2SForTileN<TileN_>::type>
struct SM120MxFP8BlockScaledBuilder
{

    using ElementA = cute::float_e4m3_t;
    using ElementB = cute::float_e4m3_t;
    using ElementSFLoad = int32_t;                // scale load type (4 packed UE8M0 bytes)
    using ElementSFCompute = cute::float_ue8m0_t; // scale mma type
    using ElementAccum = float;
    using ElementD = cute::bfloat16_t;

    static constexpr int AB_Stages = Stages_;
    static constexpr int SF_Stages = 1;
    static constexpr int kTileM = TileM_;
    static constexpr int kTileN = TileN_;
    static constexpr int MinBlocksPerSm = MinBlocksPerSm_;
    // E29: persistent-scheduler swizzle group size. Default 16 matches
    // the historical value baked into SM120BlockScaledScheduler. Forced-
    // only T4 experiment tests values 8 and 32 to see if a different
    // wave-quantization order improves small-M graph_us. Default path
    // is unchanged.
    static constexpr int kSchedGroup = SchedGroup_;

    // VS=32 path: one UE8M0 per 32 K-elements (true OCP MXFP8).
    static constexpr int kSFVecSize = 32;
    // 4 int32 SF words per K-tile of 128 elements (each int32 packs 4 UE8M0
    // bytes; with VS=32 each int32 covers 4×32 = 128 K-elements). Compare:
    // the 1×128 path uses kTileSF=1 (one int32 covers 4×128 = 512 K).
    static constexpr int kTileSF = 4;
    static constexpr int kTileK = 128;
    // Number of K-tiles whose SF data is grouped in one TMA "SF transaction":
    //   total K covered = 4 * kSFVecSize per int32 word × kTileSF int32 words
    //                   = 4 * 32 * 4 = 512
    //   K-tiles per SF tx = 512 / kTileK = 4
    // Identical to the 1×128 path → kernel pipeline structure unchanged.
    static constexpr int kNumTileKPerSF = (4 * kSFVecSize * kTileSF) / kTileK;
    static_assert(kNumTileKPerSF * kTileK == 4 * kSFVecSize * kTileSF, "kTileK must divide SF transaction K span");
    static constexpr int kNumStagePerSF = kNumTileKPerSF / AB_Stages;
    static_assert(kNumStagePerSF > 0 && kNumStagePerSF <= 2, "kNumStagePerSF must be 1 or 2 ");
    static_assert(kNumTileKPerSF % AB_Stages == 0, "kNumTileKPerSF must be divisible by AB_Stages");

    using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
    using ScaleTileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileSF>>;
    using ClusterShape = Shape<_1, _1, _1>;
    using ProblemShape = Shape<int, int, int, int>;

    // ====== mma ======
    // TileM=16 (extreme decode variant) uses 1 warp in M × 8 warps in N
    // with PermMmaTileM=16 (one hw atom in M per warp). All other TileM
    // values use 2 warps in M × 4 warps in N with PermMmaTileM=32. Driven
    // by WarpLayoutForTileM<TileM_> so the assert below still rejects
    // mismatched configurations.
    using PermMmaTileM = typename WarpLayoutForTileM<TileM_>::PermMmaTileM;
    using PermMmaTileN = PermMmaTileN_;
    using PermMmaTileK = Underscore;
    // Hardware atom (already VS=32 — same instruction the 1×128 path uses).
    using MMA_Atom = MMA_Atom<SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e4m3_t, float, float_ue8m0_t,
        32>>;
    using TiledMma = TiledMMA<MMA_Atom, typename WarpLayoutForTileM<TileM_>::ThrLayoutVMNK,
        Tile<PermMmaTileM, PermMmaTileN, PermMmaTileK>>;
    // PermMmaTileN selected via PermMmaTileNForTileN<TileN_> so that the
    // permuted N positions exactly cover [0, kTileN). See helper above for
    // the size-64 specialization rationale.
    static_assert(kTileM % cute::size(PermMmaTileM{}) == 0, "size<0>(TileShape{}) % size(PermMmaTileM{}) == 0");
    static_assert(kTileN % cute::size(PermMmaTileN{}) == 0, "kTileN must be a multiple of size(PermMmaTileN{})");
    using ThrLayoutVMNK = typename TiledMma::ThrLayoutVMNK;
    static constexpr int kNumMathThreads = cute::size(ThrLayoutVMNK{});
    static constexpr int kNumMathWarps = kNumMathThreads / 32;

    CUTE_HOST_DEVICE
    static auto ceil_div(int const& x, int const& y)
    {
        return (x + y - 1) / y;
    }

    CUTE_HOST_DEVICE
    static auto align(int const& x, int const& alignment)
    {
        return ceil_div(x, alignment) * alignment;
    }

    CUTE_HOST_DEVICE
    static auto get_tma_aligned_size(int const& x)
    {
        constexpr int kNumTMAAlignmentBytes = 16;
        CUTE_STATIC_ASSERT(kNumTMAAlignmentBytes % sizeof(ElementSFLoad) == 0, "element_size must be a multiple of 16");
        auto alignment = kNumTMAAlignmentBytes / sizeof(ElementSFLoad);
        return align(x, alignment);
    }

    CUTE_HOST_DEVICE
    static auto deduce_sfa_layout(ProblemShape const& problem_shape)
    {
        auto M = cute::get<0>(problem_shape);
        auto N = cute::get<1>(problem_shape);
        auto K = cute::get<2>(problem_shape);
        auto L = cute::get<3>(problem_shape);
        int64_t scale_m = static_cast<int64_t>(get_tma_aligned_size(M));
        // VS=32: 4× more int32 SF words per K row than the 1×128 path
        // (each int32 covers 4×kSFVecSize = 128 K-elements vs 512).
        int64_t scale_k = static_cast<int64_t>(ceil_div(K, kSFVecSize * 4));
        return make_layout(
            make_shape(scale_m, scale_k, L), make_stride(Int<1>{}, scale_m, scale_m * scale_k));
    }

    CUTE_HOST_DEVICE
    static auto deduce_sfb_layout(ProblemShape const& problem_shape)
    {
        auto M = cute::get<0>(problem_shape);
        auto N = cute::get<1>(problem_shape);
        auto K = cute::get<2>(problem_shape);
        auto L = cute::get<3>(problem_shape);
        int64_t scale_n = static_cast<int64_t>(get_tma_aligned_size(N));
        int64_t scale_k = static_cast<int64_t>(ceil_div(K, kSFVecSize * 4));
        return make_layout(
            make_shape(scale_n, scale_k, L), make_stride(Int<1>{}, scale_n, scale_n * scale_k));
    }

    // ===== Hand-rolled scale-fragment derivation (mirror of the 1×128 path).
    // The atom-level thread→fragment mapping is VS-invariant; what changes
    // for VS=32 is the SF tile K-extent (kTileSF=4), which is consumed by
    // CuTe inside the gemm() inner loop. If this turns out to be subtly
    // tied to kTileSF=1 it will surface as accuracy mismatch vs Triton.
    template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
    CUTE_HOST_DEVICE static constexpr auto thrfrg_SFA(SFATensor&& sfatensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
    {
        CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});

        auto permutation_mnk = TiledPerm{};
        auto t_tile = make_tile(get<0>(permutation_mnk), _1{});
        auto tiled_sfa = logical_divide(sfatensor, t_tile);

        using AtomShape_MNK = typename Atom::Shape_MNK;
        auto atom_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})), make_layout(_1{}));
        auto tiled_atom_sfa = zipped_divide(tiled_sfa, atom_tile);
        using AtomLayoutSFA_TV = Layout<Shape<Shape<_2, _2, _8>, _1>,
            Stride<Stride<_8, _0, _1>, _16>>;
        auto tv_atom_sfa = tiled_atom_sfa.compose(AtomLayoutSFA_TV{}, _);

        auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
        auto thr_tile
            = make_tile(_, make_tile(make_layout(size<1>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
        auto thr_tensor = zipped_divide(tv_atom_sfa, thr_tile);
        return thr_tensor;
    }

    template <class SFATensor, class ThrMma>
    CUTE_HOST_DEVICE static constexpr auto partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma)
    {
        auto thr_tensor
            = make_tensor(static_cast<SFATensor&&>(sfatensor).data(), thrfrg_SFA(sfatensor.layout(), thread_mma));
        auto thr_vmnk = thread_mma.thr_vmnk_;
        auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
        auto partition_SFA = thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
        auto frg_SFA = make_fragment_like<ElementSFLoad>(partition_SFA);
        return frg_SFA;
    }

    template <class TiledMma>
    CUTE_HOST_DEVICE static constexpr auto get_layoutSFA_TV(TiledMma& mma)
    {
        auto tile_shape_mnk = tile_shape(mma);
        auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), _1{}));
        auto thr_tensor = thrfrg_SFA(ref_A, mma);
        auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
        auto atile = make_tile(_,
            make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                          make_stride(Int<1>{}, Int<0>{})),
                _));
        auto tv_sfa = thr_tensor.compose(atile, _);
        auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
        auto tv_layout = tv_sfa.compose(thridx_2_thrid, _);
        return tv_layout;
    }

    template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
    CUTE_HOST_DEVICE static constexpr auto thrfrg_SFB(SFBTensor&& sfbtensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
    {
        CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});

        auto permutation_mnk = TiledPerm{};
        auto t_tile = make_tile(get<1>(permutation_mnk), _1{});
        auto tiled_sfb = logical_divide(sfbtensor, t_tile);

        using AtomShape_MNK = typename Atom::Shape_MNK;
        auto atom_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})), make_layout(_1{}));
        auto tiled_atom_sfb = zipped_divide(tiled_sfb, atom_tile);
        using AtomLayoutSFB_TV = Layout<Shape<Shape<_4, _8>, _1>,
            Stride<Stride<_0, _1>, _8>>;
        auto tv_atom_sfb = tiled_atom_sfb.compose(AtomLayoutSFB_TV{}, _);

        auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
        auto thr_tile
            = make_tile(_, make_tile(make_layout(size<2>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
        auto thr_tensor = zipped_divide(tv_atom_sfb, thr_tile);
        return thr_tensor;
    }

    template <class SFBTensor, class ThrMma>
    CUTE_HOST_DEVICE static constexpr auto partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma)
    {
        auto thr_tensor
            = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(), thrfrg_SFB(sfbtensor.layout(), thread_mma));
        auto thr_vmnk = thread_mma.thr_vmnk_;
        auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
        auto partition_SFB = thr_tensor(thr_vnk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
        auto frg_SFB = make_fragment_like<ElementSFLoad>(partition_SFB);
        return frg_SFB;
    }

    template <class TiledMma>
    CUTE_HOST_DEVICE static constexpr auto get_layoutSFB_TV(TiledMma& mma)
    {
        auto tile_shape_mnk = tile_shape(mma);
        auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), _1{}));
        auto thr_tensor = thrfrg_SFB(ref_B, mma);
        auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
        auto btile = make_tile(_,
            make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                          make_stride(Int<0>{}, Int<1>{})),
                _));
        auto tv_sfb = thr_tensor.compose(btile, _);
        auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
        auto tv_layout = tv_sfb.compose(thridx_2_thrid, _);
        return tv_layout;
    }

    template <class Tensor>
    CUTE_HOST_DEVICE static constexpr auto transform_fragment_for_qmma(Tensor&& tensor)
    {
        // VS=32 variant of the 1×128 transform. The 1×128 path's layout
        //   (_32, num_mn, _4, _4) with strides (_0, _4, _0, _1)
        // means: ONE byte (selected by mode 3) broadcast across 4 hw atoms
        // (mode 2 stride 0) and across 32 K-elements per atom (mode 0 stride 0).
        // For VS=32 each byte covers exactly ONE 32-element hw atom — there
        // is no broadcast across hw atoms — and we have 4× more bytes
        // (kTileSF=4 int32 words per K-tile vs 1). Layout becomes:
        //   (_32, num_mn, _4 hw-atoms-in-K-tile (stride 1), kTileSF K-tile-iter (stride num_mn*4))
        // mode 0 (stride 0): replicate byte across 32 K-elements per atom
        // mode 1 (stride 4 = 1 int32): step M
        // mode 2 (stride 1): the 4 packed bytes of one int32 = 4 hw atoms
        // mode 3 (stride num_mn*4): step to next int32 word for next K-tile
        CUTE_STATIC_ASSERT_V(rank(tensor) == Int<3>{});
        auto old_ptr = tensor.data();
        auto new_ptr = recast_ptr<ElementSFCompute>(old_ptr);
        auto old_layout = tensor.layout();
        auto num_mn = size<1>(shape(old_layout));
        CUTE_STATIC_ASSERT_V(size<2>(shape(old_layout)) == Int<kTileSF>{});
        auto new_layout = make_layout(
            make_shape(_32{}, num_mn, _4{}, Int<kTileSF>{}),
            make_stride(_0{}, _4{}, _1{}, num_mn * Int<4>{}));
        auto new_tensor = make_tensor(new_ptr, new_layout);
        return new_tensor;
    }

    // ====== load smem -> rf ======
    using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
    // SmemCopyAtomB is picked by (TileM, TileN): TileM=16 + TileN=64 needs
    // the smaller LDSM_x2 atom because each of the 8 N-warps covers only
    // 8 N-columns (8 vals/thread per call) — LDSM_x4 wants 16 vals/thread.
    // All other tiles keep LDSM_x4.
    using SmemCopyAtomB = typename SmemCopyAtomBForTileMN<TileM_, TileN_>::type;

    // ====== smem layout ======
    using SmemLayoutAtomA = GMMA::Layout_K_SW128_Atom<ElementA>;
    using SmemLayoutAtomB = GMMA::Layout_K_SW128_Atom<ElementB>;

    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{},
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<AB_Stages>{}), Step<_1, _2, _3>{}));

    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{},
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<AB_Stages>{}), Step<_1, _2, _3>{}));

    // ====== TMA config ======
    using StrideA = Stride<int64_t, Int<1>, int64_t>;
    using StrideB = Stride<int64_t, Int<1>, int64_t>;

    using TMA_A = decltype(make_tma_copy(SM90_TMA_LOAD{},
        make_tensor(recast_ptr<ElementA>(nullptr), repeat_like(StrideA{}, int64_t(0)), StrideA{}),
        SmemLayoutA{}(_, _, Int<0>{}), make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})), _1{}));

    using TMA_B = decltype(make_tma_copy(SM90_TMA_LOAD{},
        make_tensor(recast_ptr<ElementB>(nullptr), repeat_like(StrideB{}, int64_t(0)), StrideB{}),
        SmemLayoutB{}(_, _, Int<0>{}), make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})), _1{}));

    // ====== scale ======
    using SmemCopyAtomSF = Copy_Atom<AutoVectorizingCopy, ElementSFLoad>;

    using SmemLayoutAtomSFA = decltype(make_ordered_layout(select<0, 2>(ScaleTileShape{}), Step<_1, _2>{}));

    using SmemLayoutAtomSFB = decltype(make_ordered_layout(select<1, 2>(ScaleTileShape{}), Step<_1, _2>{}));

    using SmemLayoutSFA = decltype(tile_to_shape(SmemLayoutAtomSFA{},
        make_shape(shape<0>(ScaleTileShape{}), shape<2>(ScaleTileShape{}), Int<SF_Stages>{}), Step<_1, _2, _3>{}));

    using SmemLayoutSFB = decltype(tile_to_shape(SmemLayoutAtomSFB{},
        make_shape(shape<1>(ScaleTileShape{}), shape<2>(ScaleTileShape{}), Int<SF_Stages>{}), Step<_1, _2, _3>{}));

    using StrideSFA = Stride<Int<1>, int64_t, int64_t>;
    using StrideSFB = Stride<Int<1>, int64_t, int64_t>;

    using TMA_SFA = decltype(make_tma_copy(SM90_TMA_LOAD{},
        make_tensor(recast_ptr<ElementSFLoad>(nullptr), repeat_like(StrideSFA{}, int64_t(0)), StrideSFA{}),
        SmemLayoutSFA{}(_, _, cute::Int<0>{}), make_shape(shape<0>(ScaleTileShape{}), shape<2>(ScaleTileShape{})),
        _1{}));

    using TMA_SFB = decltype(make_tma_copy(SM90_TMA_LOAD{},
        make_tensor(recast_ptr<ElementSFLoad>(nullptr), repeat_like(StrideSFB{}, int64_t(0)), StrideSFB{}),
        SmemLayoutSFB{}(_, _, cute::Int<0>{}), make_shape(shape<1>(ScaleTileShape{}), shape<2>(ScaleTileShape{})),
        _1{}));

    static constexpr uint32_t TmaTransactionBytesA = static_cast<uint32_t>(
        cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementA>));
    static constexpr uint32_t TmaTransactionBytesB = static_cast<uint32_t>(
        cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * cute::sizeof_bits_v<ElementB>));
    static constexpr uint32_t TmaABTransactionBytes = TmaTransactionBytesA + TmaTransactionBytesB;
    static constexpr uint32_t TmaTransactionBytesSFA = static_cast<uint32_t>(
        cutlass::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSFLoad>));
    static constexpr uint32_t TmaTransactionBytesSFB = static_cast<uint32_t>(
        cutlass::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSFLoad>));
    static constexpr uint32_t TmaSFTransactionBytes = TmaTransactionBytesSFA + TmaTransactionBytesSFB;

    // ====== TMA store ======
    using StrideD = Stride<int64_t, Int<1>, int64_t>;
    using EpilogueTile_MN = Shape<Int<kTileM>, Int<kTileN>>;

    using CopyAtomC = Copy_Atom<SM90_U32x2_STSM_N, cutlass::half_t>;

    using SmemLayoutAtomD = SmemLayoutAtomD_;

    static constexpr int StagesD = 1;
    using SmemLayoutD = decltype(tile_to_shape(SmemLayoutAtomD{},
        make_shape(size<0>(EpilogueTile_MN{}), size<1>(EpilogueTile_MN{}), Int<StagesD>{}), Step<_1, _2, _3>{}));

    using CopyOpR2S = CopyOpR2S_;
    using CopyOpS2G = SM90_TMA_STORE;
    using TMA_D = decltype(make_tma_copy_C_sm90(CopyOpS2G{},
        make_tensor(make_gmem_ptr(static_cast<ElementD*>(nullptr)), repeat_like(StrideD{}, int64_t(0)), StrideD{}),
        take<0, 2>(SmemLayoutD{}), EpilogueTile_MN{}));

    // ====== moe store ======
    using SmemAtomLayoutO = decltype(composition(
        Swizzle<3, 3, 3>{}, Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{}));

    using SmemLayoutO = decltype(tile_to_shape(SmemAtomLayoutO{}, Shape<Int<kTileM>, Int<kTileN>>{}));

    using SmemCopyAtomR2S = Copy_Atom<AutoVectorizingCopy, ElementD>;

    using SmemCopyAtomS2R = Copy_Atom<UniversalCopy<uint128_t>, ElementD>;
    using GmemCopyAtomR2G = SmemCopyAtomS2R;

    using TiledCopyS2G = decltype(make_tiled_copy(
        SmemCopyAtomS2R{}, Layout<Shape<_32, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{}));

    struct SharedStorageLoad : cute::aligned_struct<128, _0>
    {
        alignas(1024) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> smem_A;
        alignas(1024) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> smem_B;
        cute::ArrayEngine<ElementSFLoad, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
        cute::ArrayEngine<ElementSFLoad, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
    } tensors;

    struct SharedStorageStore : cute::aligned_struct<128, _0>
    {
        alignas(1024) cute::ArrayEngine<ElementD, cute::cosize_v<SmemLayoutD>> smem_D;
    };

    struct SharedStorageMoeStore : cute::aligned_struct<128, _0>
    {
        alignas(1024) cute::ArrayEngine<ElementD, cute::cosize_v<SmemLayoutO>> smem_O;
    };

    union TensorStorage
    {
        SharedStorageLoad load;
        SharedStorageStore store;
    };

    union TensorStorageMoe
    {
        SharedStorageLoad load;
        SharedStorageMoeStore store;
    };

    using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
    using EmptyBarrier = cutlass::arch::ClusterBarrier;
    using ProducerBarrierType = FullBarrier::ValueType;
    using ConsumerBarrierType = EmptyBarrier::ValueType;

    struct BarrierStorage
    {
        FullBarrier ab_full_mbar[AB_Stages];
        EmptyBarrier ab_empty_mbar[AB_Stages];
        FullBarrier sf_full_mbar[SF_Stages];
        EmptyBarrier sf_empty_mbar[SF_Stages];
        EmptyBarrier store_full_mbar[SF_Stages];
        EmptyBarrier store_empty_mbar[SF_Stages];
    };
};

} // namespace sm120_blockscaled_gemm
