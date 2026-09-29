/*
 * Fused SwiGLU + MXFP8 requantize epilogue for the sm_120/121 grouped FC1.
 *
 * What it replaces. The MoE layer's FC1 writes a bf16 [G, m_cap, 2*INTER]
 * gate/up slab, and a separate kernel reads that slab back, computes
 * silu(gate) * up, requantizes to MXFP8 and writes the [G, m_cap, INTER] slab the
 * FC2 consumes. Per layer call that is one launch plus a full round trip of the
 * gate/up slab through memory: 101 MB written and read again at Qwen3-30B-A3B
 * M = 4096, which is past this card's 96 MB L2, and 67 MB at Qwen3.5-35B-A3B
 * (run sm120_fusion_20260927 measured the separate kernel at 56.7 and 14.4 us
 * respectively, and the slab write inside the FC1 on top of that). This epilogue
 * does the same arithmetic on the tile while it is still in shared memory, so the
 * slab never exists.
 *
 * Why it reads shared memory rather than the accumulator registers. The
 * requantize needs an amax over 32 consecutive INTER columns, which is 64
 * consecutive N columns of the GEMM, and those are spread over two warps'
 * accumulator fragments; pairing gate with up additionally depends on the
 * TiledMma's N permutation. Taking the tile from the shared-memory staging buffer
 * the epilogue already fills (the STSM the default epilogue does anyway) makes
 * both trivially local: a warp reads whatever columns it wants. The cost is the
 * smem round trip that was already there, and the instance must carry a
 * dedicated D buffer (SeparateSmemD) so the tile does not alias the mainloop's
 * A/B stages.
 *
 * Bit-exactness. The arithmetic is the unfused kernel's, instruction for
 * instruction: silu(gate)*up in bf16x2 through the same tanh.approx pair, the
 * amax over the same 32 columns (a max reduction, so order does not matter), the
 * same UE8M0 derivation, and the same paired satfinite FP8 converts. The helpers
 * below are copied from csrc/gemm/ops/quant_kernels.cu, whose comments explain
 * each choice; a test asserts the fused FC1's output equals the unfused FC1 plus
 * SwiGLU kernel bit for bit.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#include <cute/tensor.hpp>

namespace sm120_blockscaled_gemm
{
namespace fused_swiglu
{

// tanh.approx.bf16x2 (sm_90+), one MUFU per two elements — the instruction
// quant_kernels.cu's tanh2_approx uses. Going through two fp32 tanh instead
// differs by about one bf16 ULP, which is enough to flip an occasional FP8 byte
// and, more rarely, a block's UE8M0 scale, so the fused and unfused paths would
// no longer be bit-identical (measured: every row differed by a byte or two at
// cosine 0.99993).
__device__ __forceinline__ __nv_bfloat162 tanh2_approx_bf16(__nv_bfloat162 x)
{
    uint32_t r;
    asm("tanh.approx.bf16x2 %0, %1;" : "=r"(r) : "r"(*reinterpret_cast<uint32_t const*>(&x)));
    return *reinterpret_cast<__nv_bfloat162 const*>(&r);
}

// silu(g) * u on two adjacent columns (quant_kernels.cu: silu2_mul).
__device__ __forceinline__ __nv_bfloat162 silu2_mul(__nv_bfloat162 g, __nv_bfloat162 u)
{
    __nv_bfloat162 const half2v = __float2bfloat162_rn(0.5f);
    __nv_bfloat162 const xh = __hmul2(g, half2v);
    __nv_bfloat162 const t = tanh2_approx_bf16(xh);
    return __hmul2(__hfma2(xh, t, xh), u);
}

// UE8M0 byte and quantize scale from an amax (quant_kernels.cu:
// e8m0_from_amax<true>). The derivation goes through 448/amax and its
// reciprocal on purpose: the two orders differ by up to one ULP and pick
// different bytes at powers of two.
__device__ __forceinline__ void e8m0_from_amax(float amax, float& quant_scale_out, uint8_t& byte_out)
{
    float const quant_scale_raw = 448.f / amax;
    float const dequant_raw = 1.f / quant_scale_raw;
    uint32_t const bits = __float_as_uint(dequant_raw);
    uint32_t const exp_field = (bits >> 23) & 0xFFu;
    uint32_t const has_mantissa = (bits & 0x7FFFFFu) != 0u ? 1u : 0u;
    uint32_t byte = exp_field + has_mantissa;
    byte = byte > 254u ? 254u : byte;
    byte_out = static_cast<uint8_t>(byte);
    uint32_t const ds_bits = byte << 23;
    float const dequant_scale = __uint_as_float(ds_bits);
    quant_scale_out = byte != 0u ? (1.f / dequant_scale) : 1.f;
}

// The staging tensor's element type is CUTLASS's bfloat16_t; the arithmetic below
// wants the CUDA type. The two have the same 16-bit storage, so this is a
// reinterpretation rather than a conversion (and works for either type).
template <class T>
__device__ __forceinline__ __nv_bfloat16 as_nv_bf16(T const& v)
{
    static_assert(sizeof(T) == sizeof(__nv_bfloat16), "bf16 storage mismatch");
    __nv_bfloat16 out;
    __builtin_memcpy(&out, &v, sizeof(out));
    return out;
}

// Two paired cvt.rn.satfinite.e4m3x2.f32 (quant_kernels.cu: fp8x4_from_floats,
// halved — this epilogue owns two columns per lane).
__device__ __forceinline__ uint16_t fp8x2_from_floats(float v0, float v1)
{
    return static_cast<uint16_t>(__nv_cvt_float2_to_fp8x2(make_float2(v0, v1), __NV_SATFINITE, __NV_E4M3));
}

// One tile of the fused epilogue, run by all math threads after the tile has been
// staged in shared memory and the math warps have synchronised on it.
//
// `sD` is the (TileM, TileN) shared-memory tile in the epilogue's swizzled
// layout, holding the GEMM's bf16 output for interleaved FC1 weights: N column
// 2j is gate_j and 2j+1 is up_j. Every lane owns two INTER columns, so a
// 32-column scale block is exactly sixteen lanes of one row, and the amax
// reduction never leaves the warp.
//
// Rows at or past this group's `rows_valid` are padding of the masked slab and
// are not written, exactly as the separate kernel leaves them untouched.
template <int kTileM, int kTileN, int kNumMathThreads, class STensor>
CUTE_DEVICE void store_tile(STensor const& sD, __nv_fp8_e4m3* __restrict__ out_fp8,
    int32_t* __restrict__ out_sf, int thread_idx, int group, int m_tile_base, int n_tile_base, int rows_valid,
    int m_cap, int inter)
{
    constexpr int kInterTile = kTileN / 2;          // INTER columns in this tile
    constexpr int kLanesPerRow = kInterTile / 2;    // two INTER columns per lane
    static_assert(kInterTile % 32 == 0, "a scale block is 32 INTER columns");
    static_assert(kLanesPerRow == 16 || kLanesPerRow == 32, "TileN must be 64 or 128");
    constexpr int kRowsPerPass = (kNumMathThreads / 32) * (32 / kLanesPerRow);
    // The row loop's exit must be warp-uniform, because every lane has to reach
    // the amax shuffles below: that holds exactly when a pass covers a whole
    // multiple of the tile's rows.
    static_assert(kTileM % kRowsPerPass == 0, "a pass must cover a whole number of tile rows");

    int const lane = thread_idx & 31;
    int const warp = thread_idx >> 5;
    int const row_in_warp = lane / kLanesPerRow;    // 0, or 0/1 at TileN=64
    int const col_pair = lane % kLanesPerRow;       // which INTER column pair
    int const j_local = 2 * col_pair;               // first INTER column of the pair
    int const num_kp = inter / 128;
    uint8_t* const sf_bytes = reinterpret_cast<uint8_t*>(out_sf);

    for (int row0 = 0; row0 < kTileM; row0 += kRowsPerPass)
    {
        int const m_local = row0 + warp * (32 / kLanesPerRow) + row_in_warp;
        if (m_local >= kTileM)
            break;
        int const m_in = m_tile_base + m_local;
        bool const valid = m_in < rows_valid;
        // gate_j, up_j, gate_{j+1}, up_{j+1} live at N = 2j .. 2j+3.
        int const n0 = 2 * j_local;
        auto const raw_g0 = sD(m_local, n0);
        auto const raw_u0 = sD(m_local, n0 + 1);
        auto const raw_g1 = sD(m_local, n0 + 2);
        auto const raw_u1 = sD(m_local, n0 + 3);
        __nv_bfloat16 const g0 = as_nv_bf16(raw_g0);
        __nv_bfloat16 const u0 = as_nv_bf16(raw_u0);
        __nv_bfloat16 const g1 = as_nv_bf16(raw_g1);
        __nv_bfloat16 const u1 = as_nv_bf16(raw_u1);
        __nv_bfloat162 const g2 = __halves2bfloat162(g0, g1);
        __nv_bfloat162 const u2 = __halves2bfloat162(u0, u1);
        float2 const h = __bfloat1622float2(silu2_mul(g2, u2));
        float ax = fmaxf(fabsf(h.x), fabsf(h.y));
        // Sixteen lanes hold one 32-column scale block of one row.
#pragma unroll
        for (int off = 8; off > 0; off >>= 1)
            ax = fmaxf(ax, __shfl_xor_sync(0xFFFFFFFFu, ax, off));
        ax = fmaxf(ax, 1e-10f);
        float qs;
        uint8_t byte_v;
        e8m0_from_amax(ax, qs, byte_v);
        if (!valid)
            continue;
        int const j = n_tile_base / 2 + j_local;
        int64_t const row_base = (static_cast<int64_t>(group) * m_cap + m_in) * inter;
        uint16_t const packed = fp8x2_from_floats(h.x * qs, h.y * qs);
        *reinterpret_cast<uint16_t*>(&out_fp8[row_base + j]) = packed;
        if ((lane & 15) == 0)
        {
            // One UE8M0 byte per (row, 32 INTER columns). The int32 word it
            // belongs to also holds three blocks that other N tiles own, so the
            // byte is stored on its own: four CTAs writing four distinct bytes of
            // one word is safe, a read-modify-write of the word would not be.
            int const block = j / 32;
            int const kp = block / 4;
            int64_t const word = static_cast<int64_t>(group) * (static_cast<int64_t>(num_kp) * m_cap)
                + static_cast<int64_t>(kp) * m_cap + m_in;
            sf_bytes[4 * word + (block & 3)] = byte_v;
        }
    }
}

} // namespace fused_swiglu
} // namespace sm120_blockscaled_gemm
