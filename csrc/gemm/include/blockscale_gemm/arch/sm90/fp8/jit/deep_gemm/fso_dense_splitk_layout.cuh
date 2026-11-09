/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 fish-scales-ops contributors
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared-memory layout of fp8_gemm_kernel_swapAB_splitk (fso_dense_splitk.cuh), in a header of its own so that the
// host (the configuration of the split-K route in jit_utils.cuh) and the NVRTC-compiled kernel compute the same sizes.
//
// One CTA holds one 64-row weight tile and one contiguous slice of K. Its dynamic shared memory, in this order (byte
// offsets from the 1024-aligned base):
//   [weight boxes: stages x 64 x 128 B] [activation boxes: stages x BLOCK_T x 128 B]
//   [D strips: owned slabs x BLOCK_T x 16 bf16] [mailbox: owned slabs x (S - 1) fragments]
//   [weight scales of the slice: kb_slice fp32] [activation scales of the slice: kb_slice x BLOCK_T fp32]
//   [full and empty barriers: 2 x stages] [mailbox barriers: owned slabs]
// Every 128-byte-swizzled box starts on a 1024-byte boundary (64 x 128 and BLOCK_T x 128 with BLOCK_T a multiple of 8
// are multiples of 1024), the D strips and the mailbox on a 128-byte one.
#pragma once

#ifndef NVRTC_JIT_COMPILATION
#include <cstdint>
#endif

namespace deep_gemm
{

namespace fso_dense_splitk_smem
{
// Weight rows per CTA: one math warp-group of four warps, 16 rows each (one m64nBLOCK_Tk32 WGMMA).
constexpr uint32_t kBlockW = 64;
// Threads per CTA: the math warp-group and one producer warp.
constexpr uint32_t kNumThreads = 128 + 32;
// The 16-row slabs of a tile: slab w is math warp w's part of the output.
constexpr uint32_t kNumSlabs = kBlockW / 16;

__device__ __host__ constexpr uint32_t align_up(uint32_t x, uint32_t a)
{
    return (x + a - 1) / a * a;
}

// Slab w belongs to cluster rank w % S, so rank 0 owns the most slabs.
__device__ __host__ constexpr uint32_t owned_slabs(uint32_t splits)
{
    return (kNumSlabs + splits - 1) / splits;
}

__device__ __host__ constexpr uint32_t w_bytes()
{
    return kBlockW * 128;
}

__device__ __host__ constexpr uint32_t x_bytes(uint32_t block_t)
{
    return block_t * 128;
}

__device__ __host__ constexpr uint32_t x_offset(uint32_t stages)
{
    return stages * w_bytes();
}

__device__ __host__ constexpr uint32_t d_offset(uint32_t block_t, uint32_t stages)
{
    return align_up(x_offset(stages) + stages * x_bytes(block_t), 128);
}

// The bf16 staging strip of one slab for its TMA store: BLOCK_T rows (tokens) of 16 columns (weight rows).
__device__ __host__ constexpr uint32_t d_strip_bytes(uint32_t block_t)
{
    return block_t * 16 * 2;
}

__device__ __host__ constexpr uint32_t mailbox_offset(uint32_t block_t, uint32_t splits, uint32_t stages)
{
    return align_up(d_offset(block_t, stages) + owned_slabs(splits) * d_strip_bytes(block_t), 128);
}

// One pushed fragment: the fp32 partial sums a math warp holds for its slab, 32 lanes x BLOCK_T / 2 floats.
__device__ __host__ constexpr uint32_t fragment_bytes(uint32_t block_t)
{
    return 32 * (block_t / 2) * 4;
}

__device__ __host__ constexpr uint32_t mailbox_bytes(uint32_t block_t, uint32_t splits)
{
    return splits > 1 ? owned_slabs(splits) * (splits - 1) * fragment_bytes(block_t) : 0;
}

// The weight scales of the CTA's K slice: one fp32 per 128-wide K block (a 64-row tile lies inside one 128-row scale
// block), at most kb_slice of them.
__device__ __host__ constexpr uint32_t sw_offset(uint32_t block_t, uint32_t splits, uint32_t stages)
{
    return align_up(mailbox_offset(block_t, splits, stages) + mailbox_bytes(block_t, splits), 128);
}

__device__ __host__ constexpr uint32_t sw_bytes(uint32_t kb_slice)
{
    return align_up(kb_slice * 4, 128);
}

// The activation scales of the CTA's K slice and token tile: [kb_slice][BLOCK_T] fp32.
__device__ __host__ constexpr uint32_t sx_offset(uint32_t block_t, uint32_t splits, uint32_t stages, uint32_t kb_slice)
{
    return sw_offset(block_t, splits, stages) + sw_bytes(kb_slice);
}

__device__ __host__ constexpr uint32_t sx_bytes(uint32_t block_t, uint32_t kb_slice)
{
    return align_up(kb_slice * block_t * 4, 128);
}

__device__ __host__ constexpr uint32_t barriers_offset(uint32_t block_t, uint32_t splits, uint32_t stages,
    uint32_t kb_slice)
{
    return align_up(sx_offset(block_t, splits, stages, kb_slice) + sx_bytes(block_t, kb_slice), 8);
}

__device__ __host__ constexpr uint32_t total_bytes(uint32_t block_t, uint32_t splits, uint32_t stages,
    uint32_t kb_slice)
{
    return barriers_offset(block_t, splits, stages, kb_slice) + (2 * stages + owned_slabs(splits)) * 8;
}
} // namespace fso_dense_splitk_smem

} // namespace deep_gemm
