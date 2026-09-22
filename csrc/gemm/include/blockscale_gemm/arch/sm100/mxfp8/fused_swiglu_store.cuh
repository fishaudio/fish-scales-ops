/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The dual store of the sm_100 / sm_103 fused FC1 epilogue: it turns one
// epilogue subtile of the FP32 accumulator into MXFP8(silu(gate) * up), i.e.
// the fp8 bytes plus the 1x32 UE8M0 scale bytes that the second grouped GEMM
// (FC2) reads as its activation operand.
//
// Why this exists at all
// ----------------------
// An MoE layer's first projection produces a [G, m_cap, 2*I] bf16 tensor, and a
// separate kernel then reads it back, computes silu(gate) * up and quantises the
// result to MXFP8 for the second projection. That intermediate is four times
// wider than the tensor FC2 actually consumes, and the second kernel is a whole
// extra launch whose only input is a tensor the GEMM had in registers moments
// earlier. Moving the activation and the quantisation into the GEMM's epilogue
// removes both: FC1 writes the fp8 tensor and the scale slab directly, and the
// SwiGLU kernel disappears from the layer.
//
// The precondition that makes it expressible
// ------------------------------------------
// The FC1 weight rows must be INTERLEAVED, i.e. row 2j is gate_j and row 2j+1 is
// up_j, instead of fso's usual [gate; up] stacking. The reason is the shape of
// the fragment a thread holds. On the pointer-array NoSmem epilogue the
// TMEM-to-register copy gives one thread ONE output row and a CONTIGUOUS run of
// that row's N columns (the builder pins the epilogue tile to
// (TileM, min(64, TileN)) and the resulting load atom hands thread t one
// datapath and 64 consecutive columns). With interleaved weight rows those 64
// columns are exactly 32 gate/up PAIRS, which is exactly one 1x32 output scale
// block, so both the pairing and the 32-element amax are register-local: no
// shuffle, no shared memory, no second TMA descriptor. With the [gate; up]
// stacking gate_j and up_j sit N/2 columns apart and never meet in one thread.
//
// That dependency on a CUTLASS-internal partitioning is checked on the device
// rather than assumed: building with FSO_FUSED_SWIGLU_CHECK_FRAG=1 makes the
// store walk its coordinate tensor and __trap() unless every element of the
// fragment really is (same row, n0 + i) with n0 a multiple of 64.
//
// Numerics: FP32 accumulator -> silu in FP32 -> 32-wide amax -> UE8M0 byte ->
// e4m3. No bf16 staging anywhere, which is why the fused output tracks an FP32
// reference slightly BETTER than the two-kernel path does (run
// b300_mxfp8_20260917/M-E1 section 3.3 attributes the whole residual difference
// between the two to the bf16 intermediate the fusion removes).
//
// The three numerical helpers below are vendored from this repository's own
// `csrc/gemm/ops/quant_kernels.cu` (`e8m0_from_amax<true>`, `silu_tanh_approx`,
// `fp8x4_from_floats`), which cannot be included here because they live in that
// translation unit's anonymous namespace. They are copied rather than
// paraphrased so that the byte this epilogue derives is the byte the standalone
// SwiGLU kernel would have derived for the same input; the scale-factor word
// index is vendored from the same file's `sf_word_index_grouped_atom`, which is
// the layout FC2's activation operand (SFA) is built from.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "cute/tensor.hpp"

namespace cutlass
{
namespace epilogue
{
namespace collective
{
namespace fso_swiglu
{

// Arguments of the fused store, carried through the epilogue's Arguments and
// Params by the generated epilogue clone.
//
// The two destinations are plain base pointers plus per-group element counts
// rather than the per-group pointer ARRAYS CUTLASS uses for D, because the MoE
// slabs are contiguous in the group index: group g's fp8 rows start at
// `ptr_h + g * h_group_elems` and its scale slab at
// `ptr_sfh + g * sf_group_words`. That removes two arrays from the prep kernel
// instead of adding them.
struct FusedSwiGluArgs
{
    void* ptr_h = nullptr;         // __nv_fp8_e4m3 base of [G, m_cap, I]
    void* ptr_sfh = nullptr;       // int32 base of [G, pad(m_cap,128) * I/128]
    long long h_group_elems = 0;   // m_cap * I
    long long sf_group_words = 0;  // pad(m_cap,128) * (I/128)
};

// ---- vendored from csrc/gemm/ops/quant_kernels.cu -------------------------
// e8m0_from_amax<true> (UE8M0). Derives the descale via 448/amax -> 1/qs rather
// than amax * (1/448): the two differ by up to one ULP and pick different bytes
// at power-of-two boundaries, so the order matters for byte-for-byte agreement
// with the standalone kernel.
__device__ __forceinline__ void fso_e8m0_from_amax(float amax, float& quant_scale_out, unsigned char& byte_out)
{
    float const quant_scale_raw = 448.f / amax;
    float const dequant_raw = 1.f / quant_scale_raw;
    unsigned int const bits = __float_as_uint(dequant_raw);
    unsigned int const exp_field = (bits >> 23) & 0xFFu;
    unsigned int const has_mantissa = (bits & 0x7FFFFFu) != 0u ? 1u : 0u;
    unsigned int byte = exp_field + has_mantissa;
    byte = byte > 254u ? 254u : byte;
    byte_out = static_cast<unsigned char>(byte);
    unsigned int const ds_bits = byte << 23;
    float const dequant_scale = __uint_as_float(ds_bits);
    quant_scale_out = byte != 0u ? (1.f / dequant_scale) : 1.f;
}

// silu_tanh_approx: silu(x) = 0.5*x*(1 + tanh(x/2)), one MUFU instead of the
// two __expf costs.
__device__ __forceinline__ float fso_silu(float x)
{
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x));
    return 0.5f * x * (1.f + t);
}

// fp8x4_from_floats: two paired cvt.rn.satfinite.e4m3x2.f32.
__device__ __forceinline__ unsigned int fso_fp8x4(float v0, float v1, float v2, float v3)
{
    __nv_fp8x2_storage_t const lo = __nv_cvt_float2_to_fp8x2(make_float2(v0, v1), __NV_SATFINITE, __NV_E4M3);
    __nv_fp8x2_storage_t const hi = __nv_cvt_float2_to_fp8x2(make_float2(v2, v3), __NV_SATFINITE, __NV_E4M3);
    return static_cast<unsigned int>(lo) | (static_cast<unsigned int>(hi) << 16);
}
// ---------------------------------------------------------------------------

#ifndef FSO_FUSED_SWIGLU_CHECK_FRAG
#define FSO_FUSED_SWIGLU_CHECK_FRAG 0
#endif

// The fused store, called once per epilogue subtile in place of the EVT visit
// loop and the bf16 D store.
//
// `acc_frag` is the FP32 accumulator this thread holds for ONE epilogue subtile
// and `coord_frag` the matching coordinate tensor, both exactly as the stock
// epilogue built them. (M, N) are the group's problem extents and `l_coord` the
// group index, which the stock epilogue already uses to pick the per-group D
// pointer.
template <class AccEngine, class AccLayout, class CoordEngine, class CoordLayout>
CUTLASS_DEVICE void fused_swiglu_mxfp8_store(cute::Tensor<AccEngine, AccLayout> const& acc_frag,
    cute::Tensor<CoordEngine, CoordLayout> const& coord_frag, FusedSwiGluArgs const& f, int M, int N, int l_coord)
{
    using namespace cute;
    auto acc = coalesce(acc_frag);
    auto crd = coalesce(coord_frag);
    constexpr int kRun = CUTE_STATIC_V(size(acc));
    static_assert(kRun % 64 == 0,
        "fused SwiGLU epilogue: a thread's accumulator run must be a whole number of 64-column gate/up pairs "
        "(64 interleaved weight rows -> 32 outputs -> exactly one 1x32 scale block)");
    static_assert(kRun == CUTE_STATIC_V(size(crd)), "fused SwiGLU epilogue: coordinate/accumulator size mismatch");

    auto c0 = crd(_0{});
    int const m = get<0>(c0);
    int const n0 = get<1>(c0);

#if FSO_FUSED_SWIGLU_CHECK_FRAG
    // One row per thread, contiguous in n, starting on a 64-column boundary.
    // This is the CUTLASS-internal property the whole fusion rests on, so a
    // debug build turns it into a device-side trap instead of a silent wrong
    // answer.
    if ((n0 & 63) != 0)
        __trap();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kRun; ++i)
    {
        auto ci = crd(i);
        if (get<0>(ci) != m || get<1>(ci) != n0 + i)
            __trap();
    }
#endif

    // Rows at or past the group's masked row count are undefined by the masked
    // contract; the stock epilogue drops them with its predicate, this one by
    // returning before any store.
    if (m >= M)
        return;
    // A partial N tile would put part of a scale block outside the problem. The
    // launcher rejects N % TileN != 0 on the host, so this can only fire if that
    // check was bypassed.
    if (n0 + kRun > N)
        return;

    int const I = N >> 1;      // output width: the GEMM's N is 2*I
    int const num_kp = I >> 7; // scale blocks of 128 output columns

    __nv_fp8_e4m3* hrow = reinterpret_cast<__nv_fp8_e4m3*>(f.ptr_h)
        + static_cast<long long>(l_coord) * f.h_group_elems + static_cast<long long>(m) * I;
    unsigned char* sfbase
        = reinterpret_cast<unsigned char*>(f.ptr_sfh) + 4ll * (static_cast<long long>(l_coord) * f.sf_group_words);

    // sf_word_index_grouped_atom(m, kp, m_cap, num_kp, g) with the group term
    // already folded into sfbase:
    //     word = (m/128 * num_kp + kp) * 128 + (r%32)*4 + r/32,  r = m%128
    // The (r%32)*4 + r/32 term is the row's position inside the atom's 512-byte
    // block; it does not depend on kp, so it is hoisted out of the loop.
    int const r = m & 127;
    long long const word_row
        = static_cast<long long>(m >> 7) * num_kp * 128 + static_cast<long long>((r & 31) * 4 + (r >> 5));

    CUTLASS_PRAGMA_UNROLL
    for (int blk = 0; blk < kRun / 64; ++blk)
    {
        float h[32];
        float ax = 0.f;
        CUTLASS_PRAGMA_UNROLL
        for (int t = 0; t < 32; ++t)
        {
            float const g = acc(blk * 64 + 2 * t);
            float const u = acc(blk * 64 + 2 * t + 1);
            float const v = fso_silu(g) * u;
            h[t] = v;
            ax = fmaxf(ax, fabsf(v));
        }
        ax = fmaxf(ax, 1e-10f); // same floor the standalone kernel applies
        float qs;
        unsigned char byte_v;
        fso_e8m0_from_amax(ax, qs, byte_v);

        unsigned int w[8];
        CUTLASS_PRAGMA_UNROLL
        for (int p = 0; p < 8; ++p)
        {
            w[p] = fso_fp8x4(h[4 * p] * qs, h[4 * p + 1] * qs, h[4 * p + 2] * qs, h[4 * p + 3] * qs);
        }

        int const j = (n0 >> 1) + blk * 32; // output element index, a multiple of 32
        uint4* dst = reinterpret_cast<uint4*>(hrow + j);
        dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
        dst[1] = make_uint4(w[4], w[5], w[6], w[7]);

        int const kp = j >> 7;
        int const b = (j >> 5) & 3;
        sfbase[4ll * (word_row + static_cast<long long>(kp) * 128) + b] = byte_v;
    }
}

} // namespace fso_swiglu
} // namespace collective
} // namespace epilogue
} // namespace cutlass
