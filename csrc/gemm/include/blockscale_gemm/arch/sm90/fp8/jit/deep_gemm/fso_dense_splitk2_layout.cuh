/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 fish-scales-ops contributors
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared-memory layout of fp8_gemm_kernel_swapAB_splitk2 (fso_dense_splitk2.cuh), in a header of its own so that the
// host (the configuration of the two-warp-group split-K route in jit_utils.cuh) and the NVRTC-compiled kernel compute
// the same sizes.
//
// One CTA holds one weight tile of kBlockW rows (64 or 128) and one contiguous slice of K, for a token tile of BLOCK_T
// rows that covers all of M. Two math warp-groups share the tile in one of two layouts:
//   * token halves (kWeightHalves = false): 64 weight rows; warp-group g multiplies them by tokens
//     [g * BLOCK_T / 2, (g + 1) * BLOCK_T / 2);
//   * weight halves (kWeightHalves = true): 128 weight rows; warp-group g multiplies rows [64 g, 64 g + 64) by every
//     token of the tile.
// Each of the 8 math warps holds one slab, 16 weight rows by the tokens of its warp-group. Its dynamic shared memory,
// in this order (byte offsets from the 1024-aligned base):
//   [weight boxes: stages x kBlockW x 128 B] [activation boxes: stages x BLOCK_T x 128 B]
//   [D strips: owned slabs x tokens per warp-group x 16 bf16] [mailbox: owned slabs x (S - 1) fragments]
//   [weight scales of the slice: kb_slice fp32] [activation scales of the slice: kb_slice x BLOCK_T fp32]
//   [full and empty barriers: 2 x stages] [mailbox barriers: owned slabs]
// Every 128-byte-swizzled box starts on a 1024-byte boundary (kBlockW x 128 and BLOCK_T x 128 with BLOCK_T a multiple
// of 8 are multiples of 1024), and so does the second warp-group's half of a box (64 weight rows, or BLOCK_T / 2 token
// rows with BLOCK_T a multiple of 16); the D strips and the mailbox start on a 128-byte boundary.
#pragma once

#ifndef NVRTC_JIT_COMPILATION
#include <cstdint>
#endif

namespace deep_gemm
{

namespace fso_dense_splitk2_smem
{
// Threads per CTA: two math warp-groups and one producer warp.
constexpr uint32_t kNumMathThreads = 256;
constexpr uint32_t kNumThreads = kNumMathThreads + 32;
// The slabs of a tile: slab w is math warp w's part of the output.
constexpr uint32_t kNumSlabs = 8;

__device__ __host__ constexpr uint32_t align_up(uint32_t x, uint32_t a)
{
    return (x + a - 1) / a * a;
}

__device__ __host__ constexpr uint32_t block_w(bool weight_halves)
{
    return weight_halves ? 128u : 64u;
}

// The tokens one warp-group multiplies (its WGMMA's N).
__device__ __host__ constexpr uint32_t wg_tokens(bool weight_halves, uint32_t block_t)
{
    return weight_halves ? block_t : block_t / 2;
}

// Slab w belongs to cluster rank w % S, so rank 0 owns the most slabs.
__device__ __host__ constexpr uint32_t owned_slabs(uint32_t splits)
{
    return (kNumSlabs + splits - 1) / splits;
}

__device__ __host__ constexpr uint32_t w_bytes(bool weight_halves)
{
    return block_w(weight_halves) * 128;
}

__device__ __host__ constexpr uint32_t x_bytes(uint32_t block_t)
{
    return block_t * 128;
}

__device__ __host__ constexpr uint32_t x_offset(bool weight_halves, uint32_t stages)
{
    return stages * w_bytes(weight_halves);
}

__device__ __host__ constexpr uint32_t d_offset(bool weight_halves, uint32_t block_t, uint32_t stages)
{
    return align_up(x_offset(weight_halves, stages) + stages * x_bytes(block_t), 128);
}

// The bf16 staging strip of one slab for its TMA store: the warp-group's tokens (rows) by 16 columns (weight rows).
__device__ __host__ constexpr uint32_t d_strip_bytes(bool weight_halves, uint32_t block_t)
{
    return wg_tokens(weight_halves, block_t) * 16 * 2;
}

__device__ __host__ constexpr uint32_t mailbox_offset(bool weight_halves, uint32_t block_t, uint32_t splits,
    uint32_t stages)
{
    return align_up(d_offset(weight_halves, block_t, stages) + owned_slabs(splits) * d_strip_bytes(weight_halves, block_t),
        128);
}

// One pushed fragment: the fp32 partial sums a math warp holds for its slab, 32 lanes x (tokens per warp-group) / 2.
__device__ __host__ constexpr uint32_t fragment_bytes(bool weight_halves, uint32_t block_t)
{
    return 32 * (wg_tokens(weight_halves, block_t) / 2) * 4;
}

__device__ __host__ constexpr uint32_t mailbox_bytes(bool weight_halves, uint32_t block_t, uint32_t splits)
{
    return splits > 1 ? owned_slabs(splits) * (splits - 1) * fragment_bytes(weight_halves, block_t) : 0;
}

// The weight scales of the CTA's K slice: one fp32 per 128-wide K block (a 64- or 128-row tile at a multiple of its
// height lies inside one 128-row scale block), at most kb_slice of them.
__device__ __host__ constexpr uint32_t sw_offset(bool weight_halves, uint32_t block_t, uint32_t splits, uint32_t stages)
{
    return align_up(mailbox_offset(weight_halves, block_t, splits, stages) + mailbox_bytes(weight_halves, block_t, splits),
        128);
}

__device__ __host__ constexpr uint32_t sw_bytes(uint32_t kb_slice)
{
    return align_up(kb_slice * 4, 128);
}

// The activation scales of the CTA's K slice and token tile: [kb_slice][BLOCK_T] fp32.
__device__ __host__ constexpr uint32_t sx_offset(bool weight_halves, uint32_t block_t, uint32_t splits, uint32_t stages,
    uint32_t kb_slice)
{
    return sw_offset(weight_halves, block_t, splits, stages) + sw_bytes(kb_slice);
}

__device__ __host__ constexpr uint32_t sx_bytes(uint32_t block_t, uint32_t kb_slice)
{
    return align_up(kb_slice * block_t * 4, 128);
}

__device__ __host__ constexpr uint32_t barriers_offset(bool weight_halves, uint32_t block_t, uint32_t splits,
    uint32_t stages, uint32_t kb_slice)
{
    return align_up(sx_offset(weight_halves, block_t, splits, stages, kb_slice) + sx_bytes(block_t, kb_slice), 8);
}

__device__ __host__ constexpr uint32_t total_bytes(bool weight_halves, uint32_t block_t, uint32_t splits,
    uint32_t stages, uint32_t kb_slice)
{
    return barriers_offset(weight_halves, block_t, splits, stages, kb_slice) + (2 * stages + owned_slabs(splits)) * 8;
}
} // namespace fso_dense_splitk2_smem

} // namespace deep_gemm
