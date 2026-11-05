/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The bf16-output dense epilogue of interface v1 (issue #12; common/epilogue.cuh) on the sm_100/sm_103 CUTLASS
// cascade: bias with the -0.0 rule, erf-GELU and the residual, applied to the fp32 accumulator in the fixed order
//
//     v = acc;  v += bias_term(bias[n]);  v = gelu_erf(v) if act;  v += residual[m, n] if residual;  D = RN_bf16(v)
//
// with one rounding. Included only by the dense-epilogue translation units (ops/mxfp8_sm100_epi_kernel_*.cu), each
// of which explicitly instantiates one group of dense_common.cuh's configuration list.
//
// Mechanism: a CUTLASS epilogue visitor tree (EVT) with two fso nodes, plugged into the same sm_100 epilogue
// CollectiveBuilder the identity kernels use, with the same TMA or direct-store schedule as the identity kernel of
// the tile (the 256-wide TMA tiles with a narrower epilogue subtile, see Sm100MxFP8EpiGemmConfig):
//
//     DenseEpiCompute<kResidual>( BiasRowBroadcast(bias), [Sm90SrcFetch<bf16> = the residual as the C source] )
//
//   * BiasRowBroadcast is CUTLASS's Sm90RowBroadcast (bias[n] staged once per tile in shared memory and broadcast
//     down the rows; a null pointer reads as zeros) with one change: its consumer callbacks execute
//     griddepcontrol.wait before the first bias load. Those callbacks run in the four epilogue warps, which in the
//     CUTLASS sm_100 kernel never execute the grid-dependency wait themselves (sm100_gemm_tma_warpspecialized.hpp
//     waits in the mainloop-load, scheduler and epilogue-load warps only, lines 619 / 687 / 809), and the TMA
//     epilogue calls begin() — where the stock node loads the bias — before it waits on the accumulator
//     (sm100_epilogue_tma_warpspecialized.hpp:802 vs 866). Without the wait, a PDL-launched GEMM could read a bias
//     its predecessor is still writing.
//   * The residual is the epilogue's C operand: the TMA epilogue loads it with the epilogue-load warp, after that
//     warp's own grid-dependency wait; the direct-store epilogue loads it in the epilogue warps, inside the
//     subtile loop that follows begin() and therefore the wait above. Every read of bias or residual happens after
//     a griddepcontrol.wait in the reading (or TMA-issuing) thread. In SASS the wait is ACQBULK: the identity
//     kernels have two (mainloop-load and scheduler warps), these kernels three, and four with the residual on the
//     TMA epilogue (the epilogue-load warp's own wait before its C loads) (runs/S1b2/sass/acqbulk_*.txt).
//   * DenseEpiCompute is the root: it receives the fp32 accumulator fragment, the bf16 bias fragment (already the
//     -0.0-ruled bias term: BiasRowBroadcast applies the rule once per column as it stages the bias) and, with
//     kResidual, the bf16 residual fragment, widens both bf16 values exactly to fp32, evaluates the formula of
//     fso::dense_epilogue_value_rt (the shared helper of common/epilogue.cuh: explicitly rounded fp32 operations in
//     the order above) two columns at a time with the packed fp32 operations of sm_100 (dense_epi_math.cuh, bit for
//     bit the scalar helper), and converts once to bf16 with round-to-nearest-even. The activation is a run-time
//     field of its parameters, the bias a run-time pointer (null = no bias, which the -0.0 rule makes the identity
//     on every accumulator value), the residual a compile-time choice (it adds a C load pipeline and C shared
//     memory).
//
// Why not CUTLASS's built-in fusions: LinCombPerColBiasEltAct computes act(alpha * acc + beta * C + bias), i.e.
// the residual inside the activation and alpha / beta multiplies around it, and adds the raw bias (no -0.0 rule);
// the order and the -0.0 corner of interface v1 are not expressible with it. A tagged fusion operation
// (DenseEpiOp below) is used rather than a bare EVT type because the direct-store epilogue allocates an
// EpilogueTile-sized reduction buffer in shared memory for every untagged EVT (sm100_epilogue_nosmem.hpp,
// IsReductionBufferNeeded); the tag tells it no reduction is needed, which keeps that carve-out for the mainloop.
//
// The quantized output (fso.dense.linear_quantized, issue #12 S3b) is a second tree on top of the first:
//
//     DenseEpiQuantStore( DenseEpiCompute<false>( BiasRowBroadcast(bias) ) )
//
// The inner tree is the bf16 epilogue above, unchanged, so the bf16 values it returns are the bytes linear() writes
// on the same tile. DenseEpiQuantStore applies the activation quantizer of fso/activation_contract.h to them (amax of
// each 32-column block floored at 1e-10f, then the byte and the codes of fso_e8m0_from_amax and fso_mx_quantize4,
// the rule of quantize_activation, computed as dense_epi_math.cuh does: the byte from the bits of the amax, the
// products two at a time),
// returns the e4m3 codes as the epilogue's D (an e4m3 [M, N] row-major tensor, stored by the tile's own TMA or
// direct-store path) and writes the UE8M0 bytes itself, into the Sm1xx atom layout over align(M, 128) rows, with
// the bytes of the padded rows zero. That needs every thread's fragment to be one row and 32-aligned runs of
// consecutive columns: the epilogue subtile is 128 x 32 on every configuration (one 32-column block per thread per
// subtile; forced on the TMA tiles, whose automatic 192-wide subtile is 48 columns, and on the direct-store tiles,
// whose automatic subtile is 64 wide), the 32dp32b TMEM load gives each thread one datapath (row) and the subtile's
// 32 columns, and DenseEpiQuantStore checks that layout at compile time.
//
// Why 32 and not 64 columns (S3b-opt, 2026-10-29): the subtile loop of these kernels stays rolled (HeavyTag), so its
// body is the machine code of one subtile, and the erf-GELU alone is about 29 SASS instructions per element. With a
// 64-column subtile the body was about 2,700 instructions (44 KB: a 64-element GELU, the 64-element quantizer and
// the e4m3 store path), against about 1,200 (19 KB) for the bf16 epilogue kernel of the same tile, which CUTLASS
// gives a 32-wide subtile; the quantized kernel then ran far below the issue rate of the bf16 kernel (B300, fc1
// M = 1079: 28.7 us against 16.7 us for the bf16 epilogue and 3.5 us for a separate quantize kernel) with its warps
// waiting for instructions. One block per thread per subtile also makes the scale write one byte per thread per
// subtile, which is the honest granularity anyway: the 64- and 192-wide tiles split a 128-column scale word between
// CTAs.

#pragma once

#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#include "blockscale_gemm/common/epilogue.cuh"
#include "blockscale_gemm/arch/sm100/mxfp8/dense_common.cuh"
#include "blockscale_gemm/arch/sm100/mxfp8/gemm_types.cuh"
#include "fso/activation_contract.h" // the activation quantizer rule (python/fish_scales_ops/include)
#include "blockscale_gemm/arch/sm100/mxfp8/dense_epi_math.cuh" // the pair arithmetic (S3b-opt)

#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"

#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace sm100_blockscaled_gemm
{
namespace dense_epi
{

using namespace cute;

// griddepcontrol.wait with a memory clobber, so the compiler cannot hoist the ordinary loads that follow it.
// (cutlass::arch::wait_on_dependent_grids has no clobber; CUTLASS only orders TMA issues, which are volatile asm
// themselves, after it.) Compiled only where CUTLASS compiles its own waits: the host launches with PDL exactly
// when FSO_SM100_DENSE_GDC is defined, which the translation units define together with
// CUTLASS_ENABLE_GDC_FOR_SM100.
CUTLASS_DEVICE void grid_dependency_wait()
{
#if defined(CUTLASS_GDC_ENABLED)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// bias_term(bias[n]) broadcast down the rows of the tile (CUTLASS Sm90RowBroadcast: no smem pipelining), with the
// grid-dependency wait in front of its gmem loads (see the header comment) and the -0.0 rule of common/epilogue.cuh
// applied as the values are staged into shared memory: +0.0 becomes -0.0 (bf16 0x0000 -> 0x8000), every other value
// is kept, a null bias and the columns beyond N stage -0.0, so DenseEpiCompute adds the staged value as it is
// (acc + -0.0 is the identity on every accumulator value, +0.0 and -0.0 included). Staging applies the rule once per
// column instead of once per element (S3b-opt, 2026-10-29: two SASS instructions per element of the epilogue warps).
// The per-thread copy of the bias stays bf16 (ElementCompute = bf16; DenseEpiCompute widens each value exactly to
// fp32): an fp32 copy costs one register per accumulator column of the epilogue subtile and made the 256-wide-N TMA
// tiles spill.
template <class CtaTileShapeMNK>
struct BiasRowBroadcast : cutlass::epilogue::fusion::Sm90RowBroadcast<0, CtaTileShapeMNK, cutlass::bfloat16_t,
                              cutlass::bfloat16_t, Stride<_0, _1, _0>, 1, true>
{
    using Base = cutlass::epilogue::fusion::Sm90RowBroadcast<0, CtaTileShapeMNK, cutlass::bfloat16_t,
        cutlass::bfloat16_t, Stride<_0, _1, _0>, 1, true>;
    using Base::Base;

    template <class BaseCallbacks>
    struct ConsumerStoreCallbacks : BaseCallbacks
    {
        CUTLASS_DEVICE
        ConsumerStoreCallbacks(BaseCallbacks&& callbacks)
            : BaseCallbacks(cute::forward<BaseCallbacks>(callbacks))
        {
        }

        // Sm90RowBroadcast's begin() (gmem -> smem, one element per copy, the columns beyond N filled) with the
        // -0.0 rule on every staged value. The loads are plain ld.global (the pointer comes from the params, not a
        // __restrict__ argument) and follow the grid-dependency wait.
        CUTLASS_DEVICE void begin()
        {
            grid_dependency_wait();
            bool const is_nullptr = this->params.ptr_row == nullptr;
            Tensor tGS_gRow_flt = filter_zeros(this->tGS_gRow);
            Tensor tGS_sRow_flt = filter_zeros(this->tGS_sRow);
            Tensor tGS_cRow_flt = filter_zeros(this->tGS_cRow, this->tGS_gRow.stride());
            cutlass::bfloat16_t const neg_zero = cutlass::bfloat16_t::bitcast(0x8000u);
            for (int i = 0; i < size(tGS_gRow_flt); ++i)
            {
                if (get<1>(tGS_cRow_flt(i)) >= size<1>(CtaTileShapeMNK{}))
                    continue; // outside the smem row
                cutlass::bfloat16_t v = neg_zero;
                if (!is_nullptr && elem_less(tGS_cRow_flt(i), this->residue_cRow))
                {
                    v = tGS_gRow_flt(i);
                    if (v.raw() == 0u) // +0.0 -> -0.0; -0.0 (0x8000) and every other value stay
                        v = neg_zero;
                }
                tGS_sRow_flt(i) = v;
            }
        }
    };

    template <bool ReferenceSrc, class... Args>
    CUTLASS_DEVICE auto get_consumer_store_callbacks(cutlass::epilogue::fusion::ConsumerStoreArgs<Args...> const& args)
    {
        auto callbacks = Base::template get_consumer_store_callbacks<ReferenceSrc>(args);
        return ConsumerStoreCallbacks<decltype(callbacks)>(cute::move(callbacks));
    }
};

// The root of the tree: v = acc; v += bias_term(b); gelu if act; [v += r]; RN_bf16(v).
template <bool kResidual>
struct DenseEpiCompute
{
    struct SharedStorage
    {
    };

    struct Arguments
    {
        int act = fso::kDenseActNone; // fso::DenseActivation
    };

    using Params = Arguments;

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*)
    {
        return args;
    }

    template <class ProblemShape>
    static bool can_implement(ProblemShape const&, Arguments const& args)
    {
        return args.act == fso::kDenseActNone || args.act == fso::kDenseActGeluErf;
    }

    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const&, Arguments const&)
    {
        return 0;
    }

    template <class ProblemShape>
    static cutlass::Status initialize_workspace(ProblemShape const&, Arguments const&, void*, cudaStream_t,
        cutlass::CudaHostAdapter* = nullptr)
    {
        return cutlass::Status::kSuccess;
    }

    CUTLASS_DEVICE bool is_producer_load_needed() const
    {
        return false;
    }

    CUTLASS_DEVICE bool is_C_load_needed() const
    {
        return false;
    }

    CUTLASS_HOST_DEVICE
    DenseEpiCompute()
        : params()
    {
    }

    CUTLASS_HOST_DEVICE
    DenseEpiCompute(Params const& params_, SharedStorage const&)
        : params(params_)
    {
    }

    Params const params;

    template <class... Args>
    CUTLASS_DEVICE auto get_producer_load_callbacks(cutlass::epilogue::fusion::ProducerLoadArgs<Args...> const&)
    {
        return cutlass::epilogue::fusion::EmptyProducerLoadCallbacks{};
    }

    struct ConsumerStoreCallbacks : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks
    {
        CUTLASS_DEVICE
        ConsumerStoreCallbacks(int act_)
            : act(act_)
        {
        }

        int act;

        // The epilogue value of a fragment, two columns at a time (dense_epi_math.cuh: the packed fp32 operations of
        // sm_100, lane by lane the canonical _rn formula), the activation tested once per fragment, not per element.
        // frg_bias holds bias_term(bias[n]) (the -0.0 rule applied when the bias was staged).
        template <bool kRes, typename ElementAccumulator, typename ElementBias, typename ElementResidual,
            int FragmentSize>
        CUTLASS_DEVICE static cutlass::Array<cutlass::bfloat16_t, FragmentSize> compute(int act,
            cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc,
            cutlass::Array<ElementBias, FragmentSize> const& frg_bias,
            cutlass::Array<ElementResidual, FragmentSize> const* frg_res)
        {
            static_assert(FragmentSize % 2 == 0, "the pair arithmetic needs an even fragment");
            cutlass::Array<float, FragmentSize> v;
            auto pair = [&](int i, int act_c)
            {
                float2 const acc = make_float2(static_cast<float>(frg_acc[2 * i]), static_cast<float>(frg_acc[2 * i + 1]));
                float2 const b = make_float2(static_cast<float>(frg_bias[2 * i]), static_cast<float>(frg_bias[2 * i + 1]));
                float2 r = make_float2(0.f, 0.f);
                if constexpr (kRes)
                    r = make_float2(static_cast<float>((*frg_res)[2 * i]), static_cast<float>((*frg_res)[2 * i + 1]));
                float2 const o = fso::sm100::dense_value_pair<kRes>(acc, b, act_c, r);
                v[2 * i] = o.x;
                v[2 * i + 1] = o.y;
            };
            if (act == fso::kDenseActGeluErf)
            {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < FragmentSize / 2; ++i)
                    pair(i, fso::kDenseActGeluErf);
            }
            else
            {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < FragmentSize / 2; ++i)
                    pair(i, fso::kDenseActNone);
            }
            cutlass::NumericArrayConverter<cutlass::bfloat16_t, float, FragmentSize,
                cutlass::FloatRoundStyle::round_to_nearest>
                convert;
            return convert(v);
        }

        template <typename ElementAccumulator, typename ElementBias, int FragmentSize>
        CUTLASS_DEVICE cutlass::Array<cutlass::bfloat16_t, FragmentSize> visit(
            cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc, int, int, int,
            cutlass::Array<ElementBias, FragmentSize> const& frg_bias)
        {
            static_assert(!kResidual, "the residual tree passes the C fragment");
            return compute<false>(act, frg_acc, frg_bias,
                static_cast<cutlass::Array<cutlass::bfloat16_t, FragmentSize> const*>(nullptr));
        }

        template <typename ElementAccumulator, typename ElementBias, typename ElementResidual, int FragmentSize>
        CUTLASS_DEVICE cutlass::Array<cutlass::bfloat16_t, FragmentSize> visit(
            cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc, int, int, int,
            cutlass::Array<ElementBias, FragmentSize> const& frg_bias,
            cutlass::Array<ElementResidual, FragmentSize> const& frg_res)
        {
            static_assert(kResidual, "only the residual tree passes a C fragment");
            return compute<true>(act, frg_acc, frg_bias, &frg_res);
        }
    };

    template <bool ReferenceSrc, class... Args>
    CUTLASS_DEVICE auto get_consumer_store_callbacks(cutlass::epilogue::fusion::ConsumerStoreArgs<Args...> const&)
    {
        return ConsumerStoreCallbacks(params.act);
    }
};

template <class CtaTileShapeMNK, bool kResidual>
using DenseEpiTree = cute::conditional_t<kResidual,
    cutlass::epilogue::fusion::Sm90EVT<DenseEpiCompute<true>, BiasRowBroadcast<CtaTileShapeMNK>,
        cutlass::epilogue::fusion::Sm90SrcFetch<cutlass::bfloat16_t>>,
    cutlass::epilogue::fusion::Sm90EVT<DenseEpiCompute<false>, BiasRowBroadcast<CtaTileShapeMNK>>>;

// The fusion-operation tag the epilogue CollectiveBuilder is given. Its metadata says: bf16 out, fp32 compute,
// bf16 source when kResidual. Every capability flag of FusionOperation stays false, so no builder path for
// auxiliary tensors, block-scaled outputs or per-row reductions is taken.
template <bool kResidual>
struct DenseEpiOp : cutlass::epilogue::fusion::FusionOperation
{
    using ElementOutput = cutlass::bfloat16_t;
    using ElementCompute = float;
    using ElementSource = cute::conditional_t<kResidual, cutlass::bfloat16_t, void>;
    static constexpr cutlass::FloatRoundStyle RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
    static constexpr bool IsSourceSupported = kResidual;
    // "Heavy" activation: the TMA and direct-store epilogues then keep their subtile loop rolled
    // (UnrollEpiLoop = false). Unrolled, the erf-GELU of every subtile was inlined once per subtile and the
    // kernels reached 6,700 - 7,400 SASS instructions against 2,000 for the identity kernel; on the B300 the
    // epilogue then spent most of its time waiting for instructions (ncu, (256, 128) K256 at M = 364: 70 % of the
    // warp-stall samples "no instruction", the kernel 2.75x the identity's elapsed cycles even with a bias alone).
    // Rolled: 2,800 / 3,300 instructions (runs/S1b2/probes/regx, phaseB/probe).
    struct HeavyTag
    {
        static constexpr bool kIsHeavy = true;
    };
    using ActivationFn = HeavyTag;
};

// ------------------------------------------------------------------------------------- quantized output (S3b)

// FSO_SM100_DENSE_QUANT_CHECK_FRAG=1 (a debug build) makes every visit check at run time what the compile-time check
// below proves from the layout: the fragment is one row and consecutive columns starting on a 32-column boundary.
#ifndef FSO_SM100_DENSE_QUANT_CHECK_FRAG
#define FSO_SM100_DENSE_QUANT_CHECK_FRAG 0
#endif

// True when the per-thread coordinate layout `L` of one epilogue subtile (coordinates relative to the thread's first
// element) coalesces to kFrag elements with unit stride along n (_1@1) and nothing along m: the thread holds one row
// and kFrag consecutive columns. (The 32dp32b TMEM load of a 128 x 32 subtile gives ((_32,_1),_1,_1):((_1@1,_0),_0,_0).)
template <int kFrag, class L>
constexpr bool dense_quant_frag_is_row_run()
{
    return cute::is_same_v<decltype(cute::coalesce(L{})), cute::Layout<cute::Int<kFrag>, cute::ScaledBasis<cute::Int<1>, 1>>>;
}

// The root of the quantized-output tree: its input is the bf16 fragment of the inner tree (RN_bf16 of the epilogue
// value, the bytes linear() writes), its output the e4m3 codes the epilogue stores as D. Per 32-column block of a
// row: amax = max |y| floored at fso_mx_amax_floor(), byte = fso_e8m0_from_amax(amax, &qs), codes =
// fso_mx_quantize4(y * qs) -- the rule of fso/activation_contract.h, which is the rule of quantize_activation,
// evaluated with the division-free, paired forms of dense_epi_math.cuh (bit for bit the contract's functions).
// The bytes go to `out_scale`, int32 words in the Sm1xx atom layout over align(M, 128) rows (fso_mx_sf_word_index_
// sm100): rows below M get their bytes, rows in [M, align(M, 128)) zero bytes (the padded rows of the contract), rows
// beyond align(M, 128) (the far half of a 256-row tile) nothing. Every store follows BiasRowBroadcast's
// griddepcontrol.wait in begin(), so no write reaches memory a still-running predecessor may read.
struct DenseEpiQuantStore
{
    struct SharedStorage
    {
    };

    struct Arguments
    {
        int32_t* out_scale = nullptr; // align(M, 128) * N/128 words
    };

    using Params = Arguments;

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*)
    {
        return args;
    }

    template <class ProblemShape>
    static bool can_implement(ProblemShape const& problem_shape, Arguments const& args)
    {
        auto const N = cute::get<1>(problem_shape);
        return args.out_scale != nullptr && N % 128 == 0;
    }

    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const&, Arguments const&)
    {
        return 0;
    }

    template <class ProblemShape>
    static cutlass::Status initialize_workspace(ProblemShape const&, Arguments const&, void*, cudaStream_t,
        cutlass::CudaHostAdapter* = nullptr)
    {
        return cutlass::Status::kSuccess;
    }

    CUTLASS_DEVICE bool is_producer_load_needed() const
    {
        return false;
    }

    CUTLASS_DEVICE bool is_C_load_needed() const
    {
        return false;
    }

    CUTLASS_HOST_DEVICE
    DenseEpiQuantStore()
        : params()
    {
    }

    CUTLASS_HOST_DEVICE
    DenseEpiQuantStore(Params const& params_, SharedStorage const&)
        : params(params_)
    {
    }

    Params const params;

    template <class... Args>
    CUTLASS_DEVICE auto get_producer_load_callbacks(cutlass::epilogue::fusion::ProducerLoadArgs<Args...> const&)
    {
        return cutlass::epilogue::fusion::EmptyProducerLoadCallbacks{};
    }

    template <class CoordTensor>
    struct ConsumerStoreCallbacks : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks
    {
        CUTLASS_DEVICE
        ConsumerStoreCallbacks(CoordTensor tCcD_, int m0_, int n0_, int M_, int N_, int32_t* out_scale_)
            : tCcD(tCcD_)
            , m0(m0_)
            , n0(n0_)
            , M(M_)
            , N(N_)
            , out_scale(out_scale_)
        {
        }

        CoordTensor tCcD; // (CPY, CPY_M, CPY_N, EPI_M, EPI_N), relative to the thread's first element
        int m0, n0;       // the problem coordinates of the thread's first element
        int M, N;
        int32_t* out_scale;

        template <typename ElementAccumulator, int FragmentSize>
        CUTLASS_DEVICE cutlass::Array<cutlass::float_e4m3_t, FragmentSize> visit(
            cutlass::Array<ElementAccumulator, FragmentSize> const&, int epi_v, int epi_m, int epi_n,
            cutlass::Array<cutlass::bfloat16_t, FragmentSize> const& frg_y)
        {
            static_assert(FragmentSize % 32 == 0, "the quantized output needs whole 32-column blocks");
            constexpr int kBlocks = FragmentSize / 32;
            auto const crd = tCcD(cute::_, cute::_, cute::_, epi_m, epi_n);
            auto const c0 = crd(epi_v * FragmentSize);
            int const m = m0 + static_cast<int>(cute::get<0>(c0));
            int const n = n0 + static_cast<int>(cute::get<1>(c0));
#if FSO_SM100_DENSE_QUANT_CHECK_FRAG
            if ((n & 31) != 0)
                __trap();
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < FragmentSize; ++i)
            {
                auto const ci = crd(epi_v * FragmentSize + i);
                if (m0 + static_cast<int>(cute::get<0>(ci)) != m || n0 + static_cast<int>(cute::get<1>(ci)) != n + i)
                    __trap();
            }
#endif
            cutlass::Array<uint32_t, FragmentSize / 4> codes;
            uint32_t bytes[kBlocks];
            CUTLASS_PRAGMA_UNROLL
            for (int b = 0; b < kBlocks; ++b)
            {
                float y[32];
                float amax = 0.f;
                CUTLASS_PRAGMA_UNROLL
                for (int t = 0; t < 32; ++t)
                {
                    y[t] = static_cast<float>(frg_y[b * 32 + t]);
                    amax = fmaxf(amax, fabsf(y[t]));
                }
                float qs;
                bytes[b] = fso::sm100::dense_e8m0_from_amax(fmaxf(amax, fso_mx_amax_floor()), &qs);
                float2 const qs2 = make_float2(qs, qs);
                CUTLASS_PRAGMA_UNROLL
                for (int p = 0; p < 8; ++p)
                    codes[b * 8 + p] = fso::sm100::dense_quantize4(
                        make_float2(y[4 * p], y[4 * p + 1]), make_float2(y[4 * p + 2], y[4 * p + 3]), qs2);
            }
            // The bytes of blocks n/32 .. n/32 + kBlocks - 1 of row m, one byte store each (the subtile is one block
            // wide, so this is one store per thread per subtile); a padded row gets a zero byte.
            int const m_pad = (M + 127) / 128 * 128;
            if (m < m_pad && n < N)
            {
                bool const live = m < M;
                unsigned char* const sf = reinterpret_cast<unsigned char*>(out_scale);
                CUTLASS_PRAGMA_UNROLL
                for (int b = 0; b < kBlocks; ++b)
                {
                    int const kb = n / 32 + b;
                    int64_t const off = 4 * fso_mx_sf_word_index_sm100(m, kb / 4, N) + (kb & 3);
                    sf[off] = live ? static_cast<unsigned char>(bytes[b]) : static_cast<unsigned char>(0);
                }
            }
            return reinterpret_cast<cutlass::Array<cutlass::float_e4m3_t, FragmentSize> const&>(codes);
        }
    };

    template <bool ReferenceSrc, class... Args>
    CUTLASS_DEVICE auto get_consumer_store_callbacks(cutlass::epilogue::fusion::ConsumerStoreArgs<Args...> const& args)
    {
        auto const [M, N, K, L] = args.problem_shape_mnkl;
        // One epilogue subtile of this thread: one row, consecutive columns (see the header comment).
        using SubtileLayout = decltype(cute::take<0, 3>(args.tCcD.layout()));
        constexpr int kFrag = decltype(cute::size(SubtileLayout{}))::value;
        static_assert(dense_quant_frag_is_row_run<kFrag, SubtileLayout>(),
            "quantized dense epilogue: a thread's epilogue subtile must be one row and consecutive columns");
        int const m0 = static_cast<int>(M) - static_cast<int>(cute::get<0>(args.residue_tCcD));
        int const n0 = static_cast<int>(N) - static_cast<int>(cute::get<1>(args.residue_tCcD));
        return ConsumerStoreCallbacks<decltype(args.tCcD)>(
            args.tCcD, m0, n0, static_cast<int>(M), static_cast<int>(N), params.out_scale);
    }
};

template <class CtaTileShapeMNK>
using DenseEpiQuantTree = cutlass::epilogue::fusion::Sm90EVT<DenseEpiQuantStore, DenseEpiTree<CtaTileShapeMNK, false>>;

// The fusion-operation tag of the quantized-output kernels: e4m3 out, fp32 compute, no source. Every capability flag
// of FusionOperation stays false (in particular IsBlockScaleSupported: CUTLASS's own scale-factor generation rounds
// differently from the contract's rule). Heavy, like DenseEpiOp: the subtile loop stays rolled.
struct DenseEpiQuantOp : cutlass::epilogue::fusion::FusionOperation
{
    using ElementOutput = cutlass::float_e4m3_t;
    using ElementCompute = float;
    using ElementSource = void;
    static constexpr cutlass::FloatRoundStyle RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
    static constexpr bool IsSourceSupported = false;
    using ActivationFn = DenseEpiOp<false>::HeavyTag;
};

} // namespace dense_epi
} // namespace sm100_blockscaled_gemm

// The callbacks of DenseEpiOp for the sm_90-style TMA dispatch policy, which the sm_100 TMA policy and the sm_100
// direct-store policy both alias to (fusion/sm100_callbacks_tma_warpspecialized.hpp).
namespace cutlass::epilogue::fusion
{
template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore, bool kResidual,
    class CtaTileShapeMNK, class EpilogueTile>
struct FusionCallbacks<epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    sm100_blockscaled_gemm::dense_epi::DenseEpiOp<kResidual>, CtaTileShapeMNK, EpilogueTile>
    : sm100_blockscaled_gemm::dense_epi::DenseEpiTree<CtaTileShapeMNK, kResidual>
{
    using Impl = sm100_blockscaled_gemm::dense_epi::DenseEpiTree<CtaTileShapeMNK, kResidual>;
    using Operation = sm100_blockscaled_gemm::dense_epi::DenseEpiOp<kResidual>;

    struct Arguments
    {
        cutlass::bfloat16_t const* bias = nullptr; // [N], or nullptr
        int act = 0;                               // fso::DenseActivation

        operator typename Impl::Arguments() const
        {
            if constexpr (kResidual)
                return {{bias, cutlass::bfloat16_t(0.f), {}}, {}, {act}}; // bias, C, compute
            else
                return {{bias, cutlass::bfloat16_t(0.f), {}}, {act}};     // bias, compute
        }
    };

    using Impl::Impl;
};

// The callbacks of DenseEpiQuantOp (the quantized output), for the same dispatch policies.
template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore, class CtaTileShapeMNK,
    class EpilogueTile>
struct FusionCallbacks<epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    sm100_blockscaled_gemm::dense_epi::DenseEpiQuantOp, CtaTileShapeMNK, EpilogueTile>
    : sm100_blockscaled_gemm::dense_epi::DenseEpiQuantTree<CtaTileShapeMNK>
{
    using Impl = sm100_blockscaled_gemm::dense_epi::DenseEpiQuantTree<CtaTileShapeMNK>;
    using Operation = sm100_blockscaled_gemm::dense_epi::DenseEpiQuantOp;

    struct Arguments
    {
        cutlass::bfloat16_t const* bias = nullptr; // [N], or nullptr
        int act = 0;                               // fso::DenseActivation
        int32_t* out_scale = nullptr;              // the UE8M0 words, Sm1xx atom layout over align(M, 128) rows

        operator typename Impl::Arguments() const
        {
            return {{{bias, cutlass::bfloat16_t(0.f), {}}, {act}}, {out_scale}}; // (bias, compute), quant store
        }
    };

    using Impl::Impl;
};
} // namespace cutlass::epilogue::fusion

namespace sm100_blockscaled_gemm
{

// The CUTLASS types of one dense-epilogue kernel. Mirrors Sm100MxFP8GemmConfig (gemm_types.cuh) — same mainloop
// builder, same kernel schedule, same scheduler — with three differences: the epilogue is built with the DenseEpiOp
// fusion instead of the default alpha * acc; with kResidual the residual is its C source (bf16, row-major, 16-byte
// aligned rows); and the 256-wide-N tiles with the TMA epilogue use a 128 x 64 epilogue subtile instead of the
// 128 x 128 one EpilogueTileAuto forces on block-scaled N = 256 tiles. With 128 x 128 subtiles every thread holds
// 128 accumulators, 128 bias values and, with the residual, 128 residual values at once: ptxas reports 255
// registers and 392 / 516 bytes of spill stores on (128, 256) and (256, 256) with the residual (40 bytes without it
// on (256, 256)), and the residual's C stages cost those tiles a mainloop stage (128x256: 3 -> 2). With 128 x 64
// subtiles: 168 / 239 registers, no spill, and as many mainloop stages as the identity kernel or one more
// (runs/S1b2/probes/regx). The 64-wide subtile on these overlapping-accumulator tiles is the one CUTLASS's own
// direct-store builder picks for them (the shipped (128, 256) and (256, 256) direct-store kernels run with it), so
// the accumulator layout in tensor memory is the production one. The mainloop's StageCountAutoCarveout subtracts
// this epilogue's shared storage, so a tile may get a different mainloop depth than its identity twin (kStages
// below; FSO_PRINT_TILE_INFO prints it).
template <int TileM, int TileN, int ClusterM, int ClusterN, int TileK, bool NoSmemEpi, bool kResidual>
struct Sm100MxFP8EpiGemmConfig
{
    static_assert(TileM == 128 || TileM == 256, "blockscaled UMMA: TileM must be 128 (1SM) or 256 (2SM)");
    static_assert(TileN == 64 || TileN == 128 || TileN == 192 || TileN == 256, "TileN must be 64, 128, 192 or 256");
    static_assert(TileK == 128 || TileK == 256, "TileK must be 128 or 256");
    static constexpr bool kIs2Sm = (TileM == 256);
    static_assert(!kIs2Sm || (ClusterM % 2 == 0), "2SM MMA (TileM=256) requires even ClusterM");

    using ElementAPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementBPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementD = cutlass::bfloat16_t;
    using ElementC = cute::conditional_t<kResidual, cutlass::bfloat16_t, void>;
    using ElementAccumulator = float;

    using LayoutATag = cutlass::layout::RowMajor;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    static constexpr int AlignmentAB = 16;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignmentC = AlignmentD; // 8 bf16 = 16 bytes

    using MmaTileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
    using ClusterShape = Shape<Int<ClusterM>, Int<ClusterN>, _1>;

    using KernelSchedule = cute::conditional_t<kIs2Sm, cutlass::gemm::KernelTmaWarpSpecialized2SmMxf8f6f4Sm100,
        cutlass::gemm::KernelTmaWarpSpecialized1SmMxf8f6f4Sm100>;

    // An explicit epilogue tile needs an explicit schedule (the builder refuses one with EpilogueScheduleAuto); the
    // explicit TMA schedule is the one Auto resolves to for these kernel schedules.
    static constexpr bool kNarrowEpiTile = !NoSmemEpi && TileN == 256;
    using EpilogueSchedule = cute::conditional_t<NoSmemEpi,
        cute::conditional_t<kIs2Sm, cutlass::epilogue::NoSmemWarpSpecialized2Sm,
            cutlass::epilogue::NoSmemWarpSpecialized1Sm>,
        cute::conditional_t<kNarrowEpiTile,
            cute::conditional_t<kIs2Sm, cutlass::epilogue::TmaWarpSpecialized2Sm,
                cutlass::epilogue::TmaWarpSpecialized1Sm>,
            cutlass::epilogue::collective::EpilogueScheduleAuto>>;
    using EpilogueTile = cute::conditional_t<kNarrowEpiTile, Shape<_128, _64>,
        cutlass::epilogue::collective::EpilogueTileAuto>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm100,
        cutlass::arch::OpClassBlockScaledTensorOp, MmaTileShape, ClusterShape, EpilogueTile, ElementAccumulator,
        ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD, EpilogueSchedule,
        dense_epi::DenseEpiOp<kResidual>>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm100,
        cutlass::arch::OpClassBlockScaledTensorOp, ElementAPair, LayoutATag, AlignmentAB, ElementBPair, LayoutBTag,
        AlignmentAB, ElementAccumulator, MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

    using GemmKernel
        = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    static constexpr int kStages = CollectiveMainloop::DispatchPolicy::Stages;
    static constexpr int kSmemBytes = static_cast<int>(GemmKernel::SharedStorageSize);
    static constexpr int kEpiSmemBytes = static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage));
};

// Launch (or, with workspace_query, size) one dense-epilogue GEMM. Same host discipline as the identity launcher
// launch_sm100_mxfp8_gemm in dispatch.cuh: a thread-local cache of initialized Params keyed on everything they
// bake in (operand, bias, residual and output pointers, shape, swizzle, activation), cudaFuncSetAttribute once per
// instantiation on the first (eager) call, PDL when GDC is compiled in. A Params-cache miss inside a capture is
// safe: to_underlying_arguments is host-only work.
template <int TileM, int TileN, int ClusterM, int ClusterN, int TileK, bool NoSmemEpi, bool kResidual>
cudaError_t launch_sm100_mxfp8_gemm_epi(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int M, int N, int K, cudaStream_t stream,
    fso::DenseEpilogue const& epilogue, Sm100DenseWorkspace const& workspace, int swizzle,
    std::size_t* workspace_query)
{
    using Config = Sm100MxFP8EpiGemmConfig<TileM, TileN, ClusterM, ClusterN, TileK, NoSmemEpi, kResidual>;
    using Gemm = typename Config::Gemm;
    using GemmKernel = typename Config::GemmKernel;
    using Args = typename Gemm::Arguments;
    using Params = typename Gemm::Params;

    if ((epilogue.residual != nullptr) != kResidual)
        return cudaErrorInvalidValue; // the dispatcher picks the instantiation from the residual pointer

    if (workspace_query == nullptr && detail::read_print_tile_info())
    {
        static bool s_info_printed = false;
        if (!s_info_printed)
        {
            s_info_printed = true;
            std::fprintf(stderr,
                "[fso sm100 tile] epilogue %dx%dx%d cluster=(%d,%d) nosmem=%d residual=%d "
                "stages=%d smem_bytes=%d epi_smem_bytes=%d\n",
                TileM, TileN, TileK, ClusterM, ClusterN, static_cast<int>(NoSmemEpi), static_cast<int>(kResidual),
                Config::kStages, Config::kSmemBytes, Config::kEpiSmemBytes);
        }
    }

    auto ptr_A = reinterpret_cast<cutlass::float_e4m3_t const*>(mat_a);
    auto ptr_B = reinterpret_cast<cutlass::float_e4m3_t const*>(mat_b);
    auto ptr_SFA = reinterpret_cast<cutlass::float_ue8m0_t const*>(scales_a);
    auto ptr_SFB = reinterpret_cast<cutlass::float_ue8m0_t const*>(scales_b);
    auto ptr_D = reinterpret_cast<cutlass::bfloat16_t*>(mat_d);
    auto ptr_bias = reinterpret_cast<cutlass::bfloat16_t const*>(epilogue.bias);
    auto ptr_C = reinterpret_cast<cutlass::bfloat16_t const*>(epilogue.residual);

    auto problem = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M, N, 1});
    if constexpr (kResidual)
    {
        cute::get<0>(stride_C) = epilogue.ld_residual; // row pitch of the residual, in elements
        cute::get<2>(stride_C) = static_cast<int64_t>(M) * epilogue.ld_residual;
    }
    auto layout_SFA = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem);
    auto layout_SFB = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem);

    Args args{cutlass::gemm::GemmUniversalMode::kGemm, problem,
        {ptr_A, stride_A, ptr_B, stride_B, ptr_SFA, layout_SFA, ptr_SFB, layout_SFB},
        {{ptr_bias, epilogue.act}, ptr_C, stride_C, ptr_D, stride_D}};
    detail::apply_default_scheduler_knobs(args.scheduler, swizzle);

    std::size_t const ws_bytes = Gemm::get_workspace_size(args);
    if (workspace_query != nullptr)
    {
        *workspace_query = ws_bytes;
        return cudaSuccess;
    }
    if (ws_bytes > workspace.bytes)
        return cudaErrorInvalidValue; // the op body sized the workspace with the query above
    void* ws = ws_bytes > 0 ? workspace.ptr : nullptr;

    struct CacheKey
    {
        void const* p[8];
        int64_t dims[6]; // M, N, K, swizzle, act, ld_residual
        bool operator==(CacheKey const& o) const
        {
            return std::memcmp(this, &o, sizeof(o)) == 0;
        }
    };
    struct CacheHash
    {
        std::size_t operator()(CacheKey const& k) const noexcept
        {
            std::size_t h = 1469598103934665603ULL;
            auto const* p = reinterpret_cast<unsigned char const*>(&k);
            for (std::size_t i = 0; i < sizeof(k); ++i)
            {
                h ^= p[i];
                h *= 1099511628211ULL;
            }
            return h;
        }
    };
    static thread_local std::unordered_map<CacheKey, Params, CacheHash> s_params_cache;
    CacheKey key;
    std::memset(&key, 0, sizeof(key));
    key.p[0] = ptr_A;
    key.p[1] = ptr_B;
    key.p[2] = ptr_SFA;
    key.p[3] = ptr_SFB;
    key.p[4] = ptr_D;
    key.p[5] = ptr_bias;
    key.p[6] = ptr_C;
    key.p[7] = ws;
    key.dims[0] = M;
    key.dims[1] = N;
    key.dims[2] = K;
    key.dims[3] = swizzle;
    key.dims[4] = epilogue.act;
    key.dims[5] = epilogue.ld_residual;

    Params kernel_params;
    auto it = s_params_cache.find(key);
    if (it != s_params_cache.end())
        kernel_params = it->second;
    else
    {
        if (Gemm::can_implement(args) != cutlass::Status::kSuccess)
            return cudaErrorInvalidValue;
        if (ws_bytes > 0 && GemmKernel::initialize_workspace(args, ws, stream) != cutlass::Status::kSuccess)
            return cudaErrorUnknown;
        kernel_params = GemmKernel::to_underlying_arguments(args, ws); // host-only
        if (s_params_cache.size() < 64u)
            s_params_cache.emplace(key, kernel_params);
    }

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

    return Gemm::run(kernel_params, stream, nullptr, detail::dense_gemm_pdl()) == cutlass::Status::kSuccess
        ? cudaSuccess
        : cudaErrorLaunchFailure;
}

// Explicit instantiation of one configuration of dense_common.cuh's lists, without and with the residual.
#define FSO_SM100_DENSE_EPI_INSTANTIATE(TM_, TN_, CM_, CN_, TK_, NS_)                                                  \
    template cudaError_t launch_sm100_mxfp8_gemm_epi<TM_, TN_, CM_, CN_, TK_, NS_, false>(__nv_fp8_e4m3*,              \
        __nv_fp8_e4m3*, __nv_bfloat16*, int32_t*, int32_t*, int, int, int, cudaStream_t, fso::DenseEpilogue const&,    \
        Sm100DenseWorkspace const&, int, std::size_t*);                                                                \
    template cudaError_t launch_sm100_mxfp8_gemm_epi<TM_, TN_, CM_, CN_, TK_, NS_, true>(__nv_fp8_e4m3*,               \
        __nv_fp8_e4m3*, __nv_bfloat16*, int32_t*, int32_t*, int, int, int, cudaStream_t, fso::DenseEpilogue const&,    \
        Sm100DenseWorkspace const&, int, std::size_t*);

// ------------------------------------------------------------------------------------- quantized output (S3b)

// The CUTLASS types of one quantized-output kernel (fso.dense.linear_quantized): Sm100MxFP8EpiGemmConfig without the
// residual, with an e4m3 D (the codes, [M, N] row-major, 16-byte aligned rows) and a 128 x 32 epilogue subtile on
// every configuration, with the explicit TMA or direct-store schedule of the tile (an explicit subtile needs an
// explicit schedule). The subtile is what makes a 32-column block register-local and keeps the rolled subtile loop's
// body small (see the header comment); it replaces EpilogueTileAuto, which gives the 192-wide TMA tiles a 48-column
// subtile and the direct-store tiles a 64-column one.
template <int TileM, int TileN, int ClusterM, int ClusterN, int TileK, bool NoSmemEpi>
struct Sm100MxFP8EpiQuantGemmConfig
{
    static_assert(TileM == 128 || TileM == 256, "blockscaled UMMA: TileM must be 128 (1SM) or 256 (2SM)");
    static_assert(TileN % 32 == 0, "the 128 x 32 epilogue subtile must tile TileN");
    static_assert(TileK == 128 || TileK == 256, "TileK must be 128 or 256");
    static constexpr bool kIs2Sm = (TileM == 256);
    static_assert(!kIs2Sm || (ClusterM % 2 == 0), "2SM MMA (TileM=256) requires even ClusterM");

    using ElementAPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementBPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementD = cutlass::float_e4m3_t;
    using ElementC = void;
    using ElementAccumulator = float;

    using LayoutATag = cutlass::layout::RowMajor;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    static constexpr int AlignmentAB = 16;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value; // 16 codes = 16 bytes
    static constexpr int AlignmentC = AlignmentD;

    using MmaTileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
    using ClusterShape = Shape<Int<ClusterM>, Int<ClusterN>, _1>;

    using KernelSchedule = cute::conditional_t<kIs2Sm, cutlass::gemm::KernelTmaWarpSpecialized2SmMxf8f6f4Sm100,
        cutlass::gemm::KernelTmaWarpSpecialized1SmMxf8f6f4Sm100>;
    using EpilogueSchedule = cute::conditional_t<NoSmemEpi,
        cute::conditional_t<kIs2Sm, cutlass::epilogue::NoSmemWarpSpecialized2Sm,
            cutlass::epilogue::NoSmemWarpSpecialized1Sm>,
        cute::conditional_t<kIs2Sm, cutlass::epilogue::TmaWarpSpecialized2Sm,
            cutlass::epilogue::TmaWarpSpecialized1Sm>>;
    using EpilogueTile = Shape<_128, _32>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm100,
        cutlass::arch::OpClassBlockScaledTensorOp, MmaTileShape, ClusterShape, EpilogueTile, ElementAccumulator,
        ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD, EpilogueSchedule,
        dense_epi::DenseEpiQuantOp>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm100,
        cutlass::arch::OpClassBlockScaledTensorOp, ElementAPair, LayoutATag, AlignmentAB, ElementBPair, LayoutBTag,
        AlignmentAB, ElementAccumulator, MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

    using GemmKernel
        = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    static constexpr int kStages = CollectiveMainloop::DispatchPolicy::Stages;
    static constexpr int kSmemBytes = static_cast<int>(GemmKernel::SharedStorageSize);
    static constexpr int kEpiSmemBytes = static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage));
};

// Launch (or, with workspace_query, size) one quantized-output GEMM: epilogue.out_q receives the e4m3 codes [M, N],
// epilogue.out_scale (kDenseOutScaleMxSm100) the UE8M0 words over align(M, 128) rows; bias and act as in the bf16
// epilogue; no residual. The host discipline of launch_sm100_mxfp8_gemm_epi above: a thread-local cache of
// initialized Params keyed on everything they bake in (operands, bias, both outputs, shape, swizzle, activation),
// cudaFuncSetAttribute once per instantiation on the first (eager) call, PDL when GDC is compiled in.
template <int TileM, int TileN, int ClusterM, int ClusterN, int TileK, bool NoSmemEpi>
cudaError_t launch_sm100_mxfp8_gemm_epi_q(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, int32_t* scales_a,
    int32_t* scales_b, int M, int N, int K, cudaStream_t stream, fso::DenseEpilogue const& epilogue,
    Sm100DenseWorkspace const& workspace, int swizzle, std::size_t* workspace_query)
{
    using Config = Sm100MxFP8EpiQuantGemmConfig<TileM, TileN, ClusterM, ClusterN, TileK, NoSmemEpi>;
    using Gemm = typename Config::Gemm;
    using GemmKernel = typename Config::GemmKernel;
    using Args = typename Gemm::Arguments;
    using Params = typename Gemm::Params;

    if (epilogue.residual != nullptr || epilogue.out_scale_layout != fso::kDenseOutScaleMxSm100)
        return cudaErrorInvalidValue; // the dispatcher refuses both before it gets here

    if (workspace_query == nullptr && detail::read_print_tile_info())
    {
        static bool s_info_printed = false;
        if (!s_info_printed)
        {
            s_info_printed = true;
            std::fprintf(stderr,
                "[fso sm100 tile] quantized %dx%dx%d cluster=(%d,%d) nosmem=%d stages=%d smem_bytes=%d "
                "epi_smem_bytes=%d\n",
                TileM, TileN, TileK, ClusterM, ClusterN, static_cast<int>(NoSmemEpi), Config::kStages,
                Config::kSmemBytes, Config::kEpiSmemBytes);
        }
    }

    auto ptr_A = reinterpret_cast<cutlass::float_e4m3_t const*>(mat_a);
    auto ptr_B = reinterpret_cast<cutlass::float_e4m3_t const*>(mat_b);
    auto ptr_SFA = reinterpret_cast<cutlass::float_ue8m0_t const*>(scales_a);
    auto ptr_SFB = reinterpret_cast<cutlass::float_ue8m0_t const*>(scales_b);
    auto ptr_D = reinterpret_cast<cutlass::float_e4m3_t*>(epilogue.out_q);
    auto ptr_bias = reinterpret_cast<cutlass::bfloat16_t const*>(epilogue.bias);
    auto ptr_scale = static_cast<int32_t*>(epilogue.out_scale);

    auto problem = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M, N, 1});
    auto layout_SFA = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem);
    auto layout_SFB = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem);

    Args args{cutlass::gemm::GemmUniversalMode::kGemm, problem,
        {ptr_A, stride_A, ptr_B, stride_B, ptr_SFA, layout_SFA, ptr_SFB, layout_SFB},
        {{ptr_bias, epilogue.act, ptr_scale}, nullptr, stride_C, ptr_D, stride_D}};
    detail::apply_default_scheduler_knobs(args.scheduler, swizzle);

    std::size_t const ws_bytes = Gemm::get_workspace_size(args);
    if (workspace_query != nullptr)
    {
        *workspace_query = ws_bytes;
        return cudaSuccess;
    }
    if (ws_bytes > workspace.bytes)
        return cudaErrorInvalidValue; // the op body sized the workspace with the query above
    void* ws = ws_bytes > 0 ? workspace.ptr : nullptr;

    struct CacheKey
    {
        void const* p[8];
        int64_t dims[5]; // M, N, K, swizzle, act
        bool operator==(CacheKey const& o) const
        {
            return std::memcmp(this, &o, sizeof(o)) == 0;
        }
    };
    struct CacheHash
    {
        std::size_t operator()(CacheKey const& k) const noexcept
        {
            std::size_t h = 1469598103934665603ULL;
            auto const* p = reinterpret_cast<unsigned char const*>(&k);
            for (std::size_t i = 0; i < sizeof(k); ++i)
            {
                h ^= p[i];
                h *= 1099511628211ULL;
            }
            return h;
        }
    };
    static thread_local std::unordered_map<CacheKey, Params, CacheHash> s_params_cache;
    CacheKey key;
    std::memset(&key, 0, sizeof(key));
    key.p[0] = ptr_A;
    key.p[1] = ptr_B;
    key.p[2] = ptr_SFA;
    key.p[3] = ptr_SFB;
    key.p[4] = ptr_D;
    key.p[5] = ptr_bias;
    key.p[6] = ptr_scale;
    key.p[7] = ws;
    key.dims[0] = M;
    key.dims[1] = N;
    key.dims[2] = K;
    key.dims[3] = swizzle;
    key.dims[4] = epilogue.act;

    Params kernel_params;
    auto it = s_params_cache.find(key);
    if (it != s_params_cache.end())
        kernel_params = it->second;
    else
    {
        if (Gemm::can_implement(args) != cutlass::Status::kSuccess)
            return cudaErrorInvalidValue;
        if (ws_bytes > 0 && GemmKernel::initialize_workspace(args, ws, stream) != cutlass::Status::kSuccess)
            return cudaErrorUnknown;
        kernel_params = GemmKernel::to_underlying_arguments(args, ws); // host-only
        if (s_params_cache.size() < 64u)
            s_params_cache.emplace(key, kernel_params);
    }

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

    return Gemm::run(kernel_params, stream, nullptr, detail::dense_gemm_pdl()) == cutlass::Status::kSuccess
        ? cudaSuccess
        : cudaErrorLaunchFailure;
}

// Explicit instantiation of one configuration of dense_common.cuh's lists in its quantized-output form.
#define FSO_SM100_DENSE_EPI_QUANT_INSTANTIATE(TM_, TN_, CM_, CN_, TK_, NS_)                                            \
    template cudaError_t launch_sm100_mxfp8_gemm_epi_q<TM_, TN_, CM_, CN_, TK_, NS_>(__nv_fp8_e4m3*, __nv_fp8_e4m3*,   \
        int32_t*, int32_t*, int, int, int, cudaStream_t, fso::DenseEpilogue const&, Sm100DenseWorkspace const&, int,   \
        std::size_t*);

} // namespace sm100_blockscaled_gemm

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
