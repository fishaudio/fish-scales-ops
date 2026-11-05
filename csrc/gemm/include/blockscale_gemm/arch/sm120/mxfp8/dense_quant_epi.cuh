/*
 * The quantized-output form of the sm_120/sm_121 dense epilogue: fso.dense.linear_quantized (interface v1, issue #12;
 * charter section 7.1).
 *
 * What it computes. linear_quantized(x, w, bias=, activation=) returns, bit for bit,
 * quantize_activation(linear(x, w, bias=, activation=)): the epilogue applies v = act(acc + bias_term) in fp32 with the
 * canonical _rn helpers (common/epilogue_device.cuh), rounds v once to bf16, and then applies this device's activation
 * quantizer to the rounded values: per row and per 32 consecutive output columns, amax = the fp32 maximum of |y|
 * floored at 1e-10f, the UE8M0 byte of fso_e8m0_from_amax, and the codes cvt.rn.satfinite.e4m3(y * 2^(127 - byte)).
 * The output is the next GEMM's activation in activation contract v1 (fso/activation_contract.h): xq float8_e4m3fn
 * [M, N] row-major and the int32 scale words K-major [align(M, 4), N/128] (word(m, kp) at kp * align(M, 4) + m, block
 * 4 * kp + i in byte i), the words of the padded rows zero. Every step uses the contract header's functions
 * (fso_mx_amax_floor, fso_e8m0_from_amax, fso_mx_quantize4, fso_mx_pack_word, fso_mx_sf_word_index_sm120), so the
 * bytes are those of quantize_1x32_packed by construction (test case d of test_dense_v1.py checks the header against
 * that kernel).
 *
 * Two users:
 *   * store_tile: the single-launch tiles (the kernels over SM120DenseQuantBuilder, fp8/gemm_1d1d.cuh). The math
 *     warps stage the tile's bf16 values in smem_D exactly as the bf16 epilogue does (the converter's RN is the one
 *     rounding), synchronise, and then read the staged tile back, as the fused SwiGLU epilogue of the grouped FC1 does
 *     (fused_swiglu_epi.cuh): a 32-column block of a row is spread over several warps' accumulator fragments, while in
 *     shared memory it is 64 contiguous bytes. The dense tiles keep the union storage (smem_D aliases the A/B stages),
 *     so the quantized form needs no shared memory of its own and every tile keeps the shared-memory size of its bf16
 *     form (the (128, 128, 2) tile included, which a separate output buffer would have pushed past the 101,376-byte
 *     limit). A thread releases smem_D to the load warps (the store_empty barrier they wait on before a tile's loads)
 *     as soon as its own reads of it are done, with the generic-to-async proxy fence first (the release_stage rule of
 *     gemm_1d1d.cuh), and then quantizes from registers and stores to global memory while the next tile's loads run.
 *   * sm120_streamk_reduce_quant_kernel: the Stream-K route. The slices run the identity kernels into bf16 partials,
 *     and this reduce sums them exactly as the bf16 reduces do (fp32, slice 0 first, from 0.f), applies the epilogue
 *     once, rounds to bf16 and quantizes.
 *
 * Work split. A thread owns 8 consecutive columns of a row (one 16-byte word of bf16, one 8-byte word of codes), four
 * consecutive lanes own a 32-column block (its amax is two xor shuffles away), sixteen lanes the 128 columns of one
 * scale word (TileN = 128 and the reduce: one 4-byte store by the first of the sixteen) and eight lanes the 64 columns
 * of a TileN = 64 tile, which owns two of the four bytes of each scale word (a 2-byte store; the other half belongs to
 * the neighbouring tile's CTA, a disjoint byte range of the same word).
 */
#pragma once

#include "blockscale_gemm/common/epilogue_device.cuh"
#include "fso/activation_contract.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstddef>
#include <cstdint>

#include <cute/tensor.hpp>

namespace sm120_blockscaled_gemm
{
namespace dense_quant
{

// Eight consecutive bf16 values (one 16-byte word, lowest address first) as fp32. A bf16 is the high half of the
// fp32 with the same value, so the conversion is exact (it is __bfloat162float's).
__device__ __forceinline__ void bf16x8_to_float(uint4 const& w, float (&f)[8])
{
    uint32_t const u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        f[2 * i] = __uint_as_float(u[i] << 16);
        f[2 * i + 1] = __uint_as_float(u[i] & 0xFFFF0000u);
    }
}

// The 8-column chunk of a 32-column block that this lane owns, lanes 4j .. 4j + 3 of the warp owning block j in
// column order: the block's amax (a maximum, so the shuffle order does not matter) floored at the contract's floor,
// its UE8M0 byte, and the lane's eight codes in *codes (column order, lowest byte first). Every lane of the warp
// calls it (full-mask shuffles).
__device__ __forceinline__ uint8_t quantize_chunk(float const (&f)[8], uint2& codes)
{
    float ax = fabsf(f[0]);
#pragma unroll
    for (int i = 1; i < 8; ++i)
        ax = fmaxf(ax, fabsf(f[i]));
    ax = fmaxf(ax, __shfl_xor_sync(0xFFFFFFFFu, ax, 1));
    ax = fmaxf(ax, __shfl_xor_sync(0xFFFFFFFFu, ax, 2));
    ax = fmaxf(ax, fso_mx_amax_floor());
    float qs;
    uint8_t const byte = fso_e8m0_from_amax(ax, &qs);
    codes.x = fso_mx_quantize4(f[0], f[1], f[2], f[3], qs);
    codes.y = fso_mx_quantize4(f[4], f[5], f[6], f[7], qs);
    return byte;
}

// The scale word of the 128 columns that sixteen consecutive lanes own (block j in lanes 4j .. 4j + 3): valid in the
// first lane of the sixteen. Every lane of the warp calls it.
__device__ __forceinline__ uint32_t gather_word16(uint8_t byte)
{
    uint32_t const b0 = byte;
    uint32_t const b1 = __shfl_down_sync(0xFFFFFFFFu, b0, 4);
    uint32_t const b2 = __shfl_down_sync(0xFFFFFFFFu, b0, 8);
    uint32_t const b3 = __shfl_down_sync(0xFFFFFFFFu, b0, 12);
    return fso_mx_pack_word(static_cast<uint8_t>(b0), static_cast<uint8_t>(b1), static_cast<uint8_t>(b2),
        static_cast<uint8_t>(b3));
}

// The two scale bytes of the 64 columns that eight consecutive lanes own (block j in lanes 4j .. 4j + 3), the first
// block in the low byte: valid in the first lane of the eight. Every lane of the warp calls it.
__device__ __forceinline__ uint32_t gather_half8(uint8_t byte)
{
    uint32_t const b0 = byte;
    uint32_t const b1 = __shfl_down_sync(0xFFFFFFFFu, b0, 4);
    return b0 | (b1 << 8);
}

// One tile of the quantized epilogue, run by all math threads of a single-launch tile after the tile's bf16 values
// have been staged in shared memory and the math warps have synchronised on them.
//
// `sD` is the (TileM, TileN) staging tile in its swizzled layout (the 128-byte swizzle of
// GMMA::Layout_K_SW128_Atom<bf16>, 8-row atoms tiled along M first). (m0, n0) is the tile's origin in the [M, N]
// output. `release()` hands smem_D back to the load warps; every thread calls it exactly once, after its last read
// of sD. Rows at or past M are not written to xq; their scale words, for the rows below align(M, 4), are zero (the
// last m-block's tile always covers those rows: TileM is a multiple of 16).
//
// Thread t owns the chunk (row t / kChunksPerRow + p * kRowsPerPass, columns 8 * (t % kChunksPerRow) + [0, 8)) in
// pass p. kRowsPerPass (16 at TileN = 128, 32 at TileN = 64) is a multiple of the swizzle's 8-row period, so a pass
// moves the chunk by whole atoms along M and its shared-memory address by a constant: the passes address sD from one
// base pointer with compile-time offsets. (Computing each pass's swizzled address on its own gave the (160, 128, 2)
// instance one live register per pass across the k-loop, which pushed it past its 168-register cap into spills.)
template <int kTileM, int kTileN, int kNumMathThreads, class STensor, class Release>
CUTE_DEVICE void store_tile(STensor const& sD, __nv_fp8_e4m3* out_q, int32_t* out_scale, int thread_idx, int m0,
    int n0, int M, int N, Release&& release)
{
    static_assert(kTileN == 128 || kTileN == 64, "the quantized epilogue's tiles are 64 or 128 columns wide");
    constexpr int kChunksPerRow = kTileN / 8;
    constexpr int kChunks = kTileM * kChunksPerRow;
    // A warp that takes part has all 32 lanes in it (the shuffles), and its lanes cover whole rows.
    static_assert(kChunks % 32 == 0 && 32 % kChunksPerRow == 0, "a warp's chunks are whole rows of the tile");
    static_assert(kNumMathThreads % kChunksPerRow == 0, "a pass covers whole rows");
    constexpr int kRowsPerPass = kNumMathThreads / kChunksPerRow;
    static_assert(kRowsPerPass % 8 == 0, "a pass moves by whole 8-row swizzle atoms");
    // Every pass complete (all but the (16, 64) tile), or one pass that the first kChunks threads take.
    constexpr bool kFull = kChunks % kNumMathThreads == 0;
    static_assert(kFull || kChunks < kNumMathThreads, "a pass is complete, or the only one");
    constexpr int kPasses = kFull ? kChunks / kNumMathThreads : 1;
    using Elem = typename STensor::value_type;

    bool const active = kFull || thread_idx < kChunks; // warp-uniform: kChunks is a multiple of 32
    int const r0 = thread_idx / kChunksPerRow;
    int const q = thread_idx % kChunksPerRow;
    uint4 v[kPasses];
    if (active)
    {
        Elem const* const base = &sD(r0, q * 8);
        // Elements between a chunk and the same chunk kRowsPerPass rows below (a compile-time constant).
        constexpr auto kPassStride = decltype(cute::layout(sD)(cute::make_coord(cute::Int<kRowsPerPass>{}, cute::Int<0>{})))::value;
        CUTE_UNROLL
        for (int p = 0; p < kPasses; ++p)
        {
            v[p] = *reinterpret_cast<uint4 const*>(base + p * kPassStride);
        }
    }
    release();
    if (!active)
        return;

    int const m_pad = static_cast<int>(fso_align(M, 4));
    int64_t const kp = n0 / 128;
    int const lane = thread_idx & 31;
    CUTE_UNROLL
    for (int p = 0; p < kPasses; ++p)
    {
        int const m = m0 + r0 + p * kRowsPerPass;
        float f[8];
        bf16x8_to_float(v[p], f);
        uint2 codes;
        uint8_t const byte = quantize_chunk(f, codes);
        if (m < M)
        {
            *reinterpret_cast<uint2*>(out_q + static_cast<int64_t>(m) * N + n0 + q * 8) = codes;
        }
        if constexpr (kTileN == 128)
        {
            uint32_t const word = gather_word16(byte);
            if ((lane & 15) == 0 && m < m_pad)
            {
                out_scale[fso_mx_sf_word_index_sm120(m, kp, m_pad)] = m < M ? static_cast<int32_t>(word) : 0;
            }
        }
        else
        {
            uint32_t const half = gather_half8(byte);
            if ((lane & 7) == 0 && m < m_pad)
            {
                // Bytes 2 * (n0 / 64 % 2) and the next of the word: this tile's two 32-column blocks.
                uint8_t* const sf_bytes = reinterpret_cast<uint8_t*>(out_scale);
                *reinterpret_cast<uint16_t*>(
                    sf_bytes + 4 * fso_mx_sf_word_index_sm120(m, kp, m_pad) + 2 * ((n0 / 64) & 1))
                    = m < M ? static_cast<uint16_t>(half) : static_cast<uint16_t>(0);
            }
        }
    }
}

// The Stream-K reduce of the quantized output: the fp32 sum of the K_split bf16 partials in the order of
// sm120_streamk_reduce_kernel and sm120_streamk_reduce_epi_kernel (slice 0 first, from 0.f), then
// v += bias_term(bias[n]); v = act(v) (fso::dense_epilogue_value_rt, no residual), the one rounding to bf16, and the
// quantizer on the rounded values. One thread per 8 columns of the padded [align(M, 4), N] output (N % 128 == 0);
// the threads of the padded rows write their zero scale words and nothing else. Launched without PDL after the
// join of the slices, like the bf16 reduces, so bias (the caller's tensor) and the partials are complete when it
// runs. A template only so that no translation unit carries its code unless it launches it
// (ops/mxfp8_dense_quant_kernel.cu does).
template <int kUnused = 0>
__global__ void sm120_streamk_reduce_quant_kernel(__nv_bfloat16 const* __restrict__ partials,
    __nv_fp8_e4m3* __restrict__ out_q, int32_t* __restrict__ out_scale, int M, int N, int K_split,
    std::size_t partial_stride, __nv_bfloat16 const* bias, int act)
{
    std::size_t const t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int const m_pad = static_cast<int>(fso_align(M, 4));
    std::size_t const chunks = static_cast<std::size_t>(m_pad) * static_cast<std::size_t>(N) / 8;
    // Warp-uniform: chunks is a multiple of 64 (align(M, 4) * N / 8 with N % 128 == 0).
    if (t >= chunks)
        return;
    std::size_t const e = t * 8;
    int const m = static_cast<int>(e / static_cast<std::size_t>(N));
    int const n = static_cast<int>(e - static_cast<std::size_t>(m) * static_cast<std::size_t>(N));
    bool const live = m < M;
    float f[8];
    if (live)
    {
        float acc[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            acc[i] = 0.f;
        for (int s = 0; s < K_split; ++s)
        {
            uint4 const w = *reinterpret_cast<uint4 const*>(partials + static_cast<std::size_t>(s) * partial_stride + e);
            float p[8];
            bf16x8_to_float(w, p);
#pragma unroll
            for (int i = 0; i < 8; ++i)
                acc[i] += p[i];
        }
        bool const has_bias = bias != nullptr;
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            // Plain coherent loads (fso::dense_load_bf16), one element at a time: a bias view may be 2-byte aligned.
            __nv_bfloat16 const b = has_bias ? fso::dense_load_bf16(bias + n + i) : __ushort_as_bfloat16(0);
            float const v = fso::dense_epilogue_value_rt(acc[i], has_bias, b, act, false, __ushort_as_bfloat16(0));
            f[i] = __bfloat162float(__float2bfloat16_rn(v));
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < 8; ++i)
            f[i] = 0.f;
    }
    uint2 codes;
    uint8_t const byte = quantize_chunk(f, codes);
    if (live)
    {
        *reinterpret_cast<uint2*>(out_q + e) = codes;
    }
    uint32_t const word = gather_word16(byte);
    if ((threadIdx.x & 15) == 0)
    {
        out_scale[fso_mx_sf_word_index_sm120(m, n / 128, m_pad)] = live ? static_cast<int32_t>(word) : 0;
    }
}

} // namespace dense_quant
} // namespace sm120_blockscaled_gemm
