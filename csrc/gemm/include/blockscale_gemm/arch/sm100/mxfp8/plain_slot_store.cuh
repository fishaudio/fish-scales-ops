/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The bf16 store of the PLAIN (unfused) slot GEMM on the compact activation
// layout (fso 0.2.1, run fso_0.2.1_20261010/S10b-1): the slot route's FC2, and
// its FC1 when the FC1 is not fused.
//
// Why the compact slot route has its own store
// --------------------------------------------
// The masked slot kernel stores through CUTLASS's TMA epilogue: the whole
// 64-column token tile of every live expert goes out, so at decode, where an
// expert holds a few of those columns, most of the bytes the GEMM writes are
// rows nobody reads (run S10a: 21-50 MB of writes per GEMM at M >= 64 against
// 2-4 MB of live rows). A TMA store cannot be cut per expert -- its bound is
// the descriptor's extent -- so the compact kernel stores directly from
// registers and only the token columns below the expert's routed row count,
// which the kernel hands its epilogue as the problem extent N (`fso_store_shape`
// in the generated kernel).
//
// Why not CUTLASS's own direct-store epilogue
// -------------------------------------------
// For the column-major D of the swap orientation (token rows contiguous along
// the expert's weight rows) the stock direct store loads the accumulator with
// the 16dp256b TMEM atom, which leaves each thread two rows eight apart and two
// columns, so every store is a scalar 2-byte write and one warp instruction
// touches eight columns. Measured on the slot FC2 (S10b-1 probe): 10 per cent
// slower than the TMA store at Family C M = 1 and 8, where the epilogue of a
// short tile is on the critical path. This store instead is bound into a clone
// of the NoSmem epilogue that declares D N-major, which selects the 32dp32b TMEM
// atom: thread (warp, lane) holds accumulator ROW m = 32 * warp + lane of the
// tile (one expert weight row) and the tile's 64 token columns of it, as for the
// fused SwiGLU slot store (fused_swiglu_slot_store.cuh, which documents the
// mapping and its debug trap). A token column of the tile is then 128
// consecutive bf16 of that token's output row, and each warp writes 32 of them
// -- 64 contiguous bytes -- per column, one column per instruction.
//
// Numerics are the stock epilogue's: D = alpha * acc with alpha = 1 and no
// source, i.e. the FP32 accumulator rounded to bf16 to nearest-even, so the
// bytes of every live row are the TMA epilogue's.

#pragma once

#include <cuda_bf16.h>

#include "cute/tensor.hpp"

namespace cutlass
{
namespace epilogue
{
namespace collective
{
namespace fso_plain_slot
{

// Where the bf16 output lives: element (weight row m, token column c, batch l)
// at ptr_d + l * batch_elems + c * col_elems + m. On the compact slot route l is
// the expert's first block, col_elems = N_w and batch_elems = 128 * N_w.
struct PlainSlotArgs
{
    void* ptr_d = nullptr;
    long long col_elems = 0;
    long long batch_elems = 0;
};

constexpr int kSlotEpiTokens = 64;

template <class AccEngine, class AccLayout, class CoordEngine, class CoordLayout>
CUTLASS_DEVICE void plain_slot_bf16_store(cute::Tensor<AccEngine, AccLayout> const& acc_frag,
    cute::Tensor<CoordEngine, CoordLayout> const& coord_frag, PlainSlotArgs const& f, int M, int N, int l_coord)
{
    using namespace cute;
    auto acc = coalesce(acc_frag);
    auto crd = coalesce(coord_frag);
    constexpr int kRun = CUTE_STATIC_V(size(acc));
    static_assert(kRun == kSlotEpiTokens,
        "plain slot store: a thread's accumulator run must be the whole 64-column token tile (TileN 64, one "
        "epilogue subtile, 32dp32b TMEM load); a different epilogue tile breaks the lane = row mapping");
    static_assert(kRun == CUTE_STATIC_V(size(crd)), "plain slot store: coordinate/accumulator size mismatch");

    auto c0 = crd(_0{});
    int const m = get<0>(c0);  // accumulator row = expert weight row
    int const n0 = get<1>(c0); // first token column of this thread's run
    if (m >= M)
        return; // no cross-lane step follows, so a thread outside the problem simply stores nothing
    // N is the expert's routed row count (the compact kernel's store bound): the columns below it are live.
    int const live = (N - n0) < kRun ? (N - n0) : kRun;
    __nv_bfloat16* const base = reinterpret_cast<__nv_bfloat16*>(f.ptr_d)
        + static_cast<long long>(l_coord) * f.batch_elems + static_cast<long long>(n0) * f.col_elems + m;
    CUTLASS_PRAGMA_UNROLL
    for (int t = 0; t < kRun; ++t)
    {
        if (t >= live)
            break;
        base[static_cast<long long>(t) * f.col_elems] = __float2bfloat16_rn(acc(t));
    }
}

} // namespace fso_plain_slot
} // namespace collective
} // namespace epilogue
} // namespace cutlass
