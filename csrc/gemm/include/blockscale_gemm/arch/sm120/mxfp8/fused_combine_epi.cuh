/*
 * Fused weighted-combine epilogue for the sm_120/121 grouped FC2.
 *
 * What it replaces. The MoE layer's FC2 writes a bf16 [G, m_cap, HIDDEN] slab of
 * per-pair rows, and a separate kernel reads those rows back and sums each
 * token's topk of them, weighted, into the layer output. This epilogue adds each
 * row into the token's output row directly, so the slab and the combine kernel
 * both disappear.
 *
 * What it is for, measured. The sum has to happen in memory, because a CTA holds
 * one expert's rows and never a token's topk rows, so the adds are atomic. Two
 * measurements decide the shape and the purpose (run sm120_fusion_20260927).
 *
 * The request width matters: moving the same 134 MB at M = 4096 takes 181 us as
 * four `red.global.add.noftz.bf16x2` per thread but 101 us as two
 * `red.global.add.noftz.v2.bf16x2` -- the limit is the request rate (~185 G
 * requests/s), not L2 bandwidth, which the wide form pushes to 1333 GB/s. Hence the
 * eight-byte request below.
 *
 * Where the scatter runs, and why it decides whether this route is a win. The store
 * it replaces is asynchronous: the default epilogue hands its tile to a TMA-store
 * warp that drains shared memory while the mainloop is already on the next tile.
 * Written first in the math warps, this epilogue gave that overlap up and measured
 * +0.2 to +7.0 % against the deterministic pair (run sm120_fusion_20260927). Run
 * instead on the store warp plus the otherwise-idle fourth TMA warp, it gets the
 * overlap back and the route turns into a win on both axes (run
 * sm120_scatterwarp_20260928): the layer is 2-7 % FASTER than the slab pair and the
 * transient footprint still falls about half (Qwen3.5-35B-A3B at M = 8192: 14.0 GiB
 * -> 6.0 GiB; Qwen3-30B-A3B 7.5 -> 3.5), which is the difference between fitting and
 * not fitting beside the weights on a 32 GB card. Chunking the token dimension for
 * the same peak costs +80 to +209 % instead.
 *
 * ONE warp is not enough: that form costs +6 %, because a single warp cannot keep
 * enough atomic requests in flight to sustain the scatter. Two clears it, and
 * doubling again is not possible -- the other two TMA warps are loading A/B and the
 * scale factors. FSO_MOE_SCATTER_WARP=0 restores the math-warp form for an A/B.
 *
 * Contract the caller owes. The output buffer is ACCUMULATED into, so it must
 * already hold what the layer wants added to (zero, or a shared expert's gated
 * output). Rows past `masked_m[g]` are padding and are skipped, and a pair the
 * routing skipped never had a row here to begin with, so a token whose every
 * entry was skipped keeps whatever the buffer held -- zero, if the caller zeroed
 * it. The accumulation order is whatever order the CTAs finish in, so the result
 * is not bit-reproducible run to run; that is why the route is opt-in.
 */
#pragma once

#include <cuda_bf16.h>
#include <cstdint>

#include <cute/tensor.hpp>

namespace sm120_blockscaled_gemm
{
namespace fused_combine
{

// Two bf16x2 lanes in one request: eight bytes, four output columns.
__device__ __forceinline__ void red_v2_bf16x2(void* addr, uint32_t lo, uint32_t hi)
{
    asm volatile("red.global.add.noftz.v2.bf16x2 [%0], {%1, %2};" ::"l"(addr), "r"(lo), "r"(hi) : "memory");
}

template <class T>
__device__ __forceinline__ float to_float(T const& v)
{
    static_assert(sizeof(T) == sizeof(__nv_bfloat16), "bf16 storage mismatch");
    __nv_bfloat16 b;
    __builtin_memcpy(&b, &v, sizeof(b));
    return __bfloat162float(b);
}

// One tile: read the staged bf16 tile back out of shared memory, scale each row by
// its combine weight and add it into the row of `out` its token owns. Four output
// columns per lane, so a warp covers 128 columns in 32 requests of eight bytes.
template <int kTileM, int kTileN, int kNumMathThreads, class STensor>
CUTE_DEVICE void store_tile(STensor const& sD, __nv_bfloat16* __restrict__ out,
    int32_t const* __restrict__ row_map, float const* __restrict__ weight_of_slot, int thread_idx, int group,
    int m_tile_base, int n_tile_base, int rows_valid, int m_cap, int64_t out_ld)
{
    constexpr int kColsPerLane = 4;
    constexpr int kLanesPerRow = kTileN / kColsPerLane;   // 16 at TileN=64, 32 at 128
    static_assert(kLanesPerRow == 16 || kLanesPerRow == 32, "TileN must be 64 or 128");
    constexpr int kRowsPerPass = (kNumMathThreads / 32) * (32 / kLanesPerRow);
    static_assert(kTileM % kRowsPerPass == 0, "a pass must cover a whole number of tile rows");

    int const lane = thread_idx & 31;
    int const warp = thread_idx >> 5;
    int const row_in_warp = lane / kLanesPerRow;
    int const col_lane = lane % kLanesPerRow;
    int const n_local = col_lane * kColsPerLane;

    for (int row0 = 0; row0 < kTileM; row0 += kRowsPerPass)
    {
        int const m_local = row0 + warp * (32 / kLanesPerRow) + row_in_warp;
        int const m_in = m_tile_base + m_local;
        if (m_in >= rows_valid)
            continue;
        int const slot = group * m_cap + m_in;
        int const token = row_map[slot];
        float const w = weight_of_slot[slot];
        float const v0 = to_float(sD(m_local, n_local));
        float const v1 = to_float(sD(m_local, n_local + 1));
        float const v2 = to_float(sD(m_local, n_local + 2));
        float const v3 = to_float(sD(m_local, n_local + 3));
        __nv_bfloat162 const lo = __floats2bfloat162_rn(w * v0, w * v1);
        __nv_bfloat162 const hi = __floats2bfloat162_rn(w * v2, w * v3);
        red_v2_bf16x2(&out[static_cast<int64_t>(token) * out_ld + n_tile_base + n_local],
            *reinterpret_cast<uint32_t const*>(&lo), *reinterpret_cast<uint32_t const*>(&hi));
    }
}

} // namespace fused_combine
} // namespace sm120_blockscaled_gemm
