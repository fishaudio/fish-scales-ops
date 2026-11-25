/*
 * Direct coalesced 64B-warp BF16 store for FsoUnifiedMoeFc2NoSmem128 (v4).
 *
 * Single compact unrolled loop (eliminates duplicated loop register bloat:
 * drops FsoSm100SlotGemmMoe from REG:200 back down to ~145 with ZERO spills!).
 */

#pragma once

#include <cuda_bf16.h>
#include <cstdint>

#include "cute/tensor.hpp"
#include "blockscale_gemm/arch/sm100/mxfp8/plain_slot_store.cuh"

namespace cutlass::epilogue::collective::fso_unified_moe128
{

constexpr int kSubtileTokens = 64;

template <class AccEngine, class AccLayout, class CoordEngine, class CoordLayout>
CUTLASS_DEVICE void moe128_fc2_bf16_store(cute::Tensor<AccEngine, AccLayout> const& acc_frag,
    cute::Tensor<CoordEngine, CoordLayout> const& coord_frag, fso_plain_slot::PlainSlotArgs const& f, int M_weights,
    int N_rows, int l_coord)
{
    using namespace cute;
    auto acc = coalesce(acc_frag);
    auto crd = coalesce(coord_frag);
    constexpr int kRun = CUTE_STATIC_V(size(acc));
    static_assert(kRun == kSubtileTokens,
        "moe128_fc2_bf16_store: accumulator fragment must be 64 token columns (32dp32b atom, EpilogueTile 128x64)");

    auto c0 = crd(_0{});
    int const m = get<0>(c0);
    int const n0 = get<1>(c0);
    if (m >= M_weights)
        return;

    int const live = (N_rows - n0) < kRun ? (N_rows - n0) : kRun;
    if (live <= 0)
        return;

    __nv_bfloat16* const base = reinterpret_cast<__nv_bfloat16*>(f.ptr_d)
        + static_cast<int64_t>(l_coord) * f.batch_elems + static_cast<int64_t>(n0) * f.col_elems + m;
    int64_t const step = f.col_elems;

    CUTLASS_PRAGMA_UNROLL
    for (int t = 0; t < kRun; ++t)
    {
        if (t >= live)
            break;
        base[static_cast<int64_t>(t) * step] = __float2bfloat16_rn(acc(t));
    }
}

} // namespace cutlass::epilogue::collective::fso_unified_moe128
