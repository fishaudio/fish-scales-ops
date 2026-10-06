/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared env-var override parsing for the sm_120 dispatchers (fp8 + mxfp8).
//
// The cascades themselves (and their per-precision tile tables) stay in
// `arch/sm120/fp8/dispatch.cuh` and `arch/sm120/mxfp8/dispatch.cuh` — those
// tile choices are tuned per-precision (e.g. FP8 uses NS=4 at TileM=64 while
// MXFP8 uses NS=2 because the 4× SF traffic at kSFVecSize=32 hits the 99 KB
// SMEM budget). What IS shared between the two paths is the env-var contract:
//
//   FSO_FORCE_TILE="TM,TN,ST"      e.g. "32,128,4"
//   FSO_FORCE_TILE_K="K"           optional: apply FSO_FORCE_TILE only to GEMMs
//                                  whose K equals this value, so a layer-level
//                                  sweep can vary one projection's tile while
//                                  the others keep their cascade picks
//   FSO_FORCE_KSPLIT="N"           Stream-K split, optional
//   FSO_DISABLE_OVERRIDES=1        skip K-aware single-launch overrides
//
// Same wire format on both paths so a single sweep harness can drive both.
// See docs/skills/blockscale-gemm-tuning/references/dispatcher-overrides.md.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace tensorrt_llm
{
namespace kernels::blockscale_gemm
{

// Parsed `FSO_FORCE_TILE` + `FSO_FORCE_KSPLIT`. Sentinel values:
//   tm == -1 → no override (or unparseable env var)
//   tm  > 0  → use (tm, tn, st); ks > 0 forces Stream-K split
struct ForcedTile
{
    int tm;
    int tn;
    int st;
    int ks;
    int mb;  // FSO_FORCE_MIN_BLOCKS: MinBlocksPerSm hint; sentinel -1 = unset.
    int sm;  // FSO_FORCE_SMALLM: 1 → use experimental SmallM kernel variant.
    int sg;  // FSO_FORCE_SCHED_GROUP: scheduler 1-D blocks-per-group; sentinel -1.

    constexpr bool active() const noexcept
    {
        return tm > 0;
    }
    constexpr bool stream_k() const noexcept
    {
        return ks > 1;
    }
    constexpr bool min_blocks_2() const noexcept
    {
        return mb == 2;
    }
    constexpr bool smallm() const noexcept
    {
        return sm == 1;
    }
    constexpr bool sched_group_forced() const noexcept
    {
        return sg > 0;
    }
};

// Reads FSO_FORCE_TILE / FSO_FORCE_KSPLIT once (function-static cache).
// Subsequent calls return the cached struct — one int compare on the hot
// path, same shape as the inline statics the dispatchers used before.
inline ForcedTile read_force_tile() noexcept
{
    static int s_tm = -2, s_tn = -2, s_st = -2, s_ks = -2, s_mb = -2, s_sm = -2, s_sg = -2;
    if (s_tm == -2)
    {
        char const* tile_env = std::getenv("FSO_FORCE_TILE");
        if (tile_env && *tile_env)
        {
            int tm = 0, tn = 0, st = 0;
            if (std::sscanf(tile_env, "%d,%d,%d", &tm, &tn, &st) == 3)
            {
                s_tm = tm;
                s_tn = tn;
                s_st = st;
            }
            else
            {
                s_tm = -1;
            }
        }
        else
        {
            s_tm = -1;
        }
        char const* ks_env = std::getenv("FSO_FORCE_KSPLIT");
        s_ks = (ks_env && *ks_env) ? std::atoi(ks_env) : -1;
        char const* mb_env = std::getenv("FSO_FORCE_MIN_BLOCKS");
        s_mb = (mb_env && *mb_env) ? std::atoi(mb_env) : -1;
        char const* sm_env = std::getenv("FSO_FORCE_SMALLM");
        s_sm = (sm_env && *sm_env && std::atoi(sm_env) == 1) ? 1 : 0;
        char const* sg_env = std::getenv("FSO_FORCE_SCHED_GROUP");
        s_sg = (sg_env && *sg_env) ? std::atoi(sg_env) : -1;
    }
    return ForcedTile{s_tm, s_tn, s_st, s_ks, s_mb, s_sm, s_sg};
}

// FSO_FORCE_TILE_K: when set (> 0), FSO_FORCE_TILE applies only to GEMMs whose
// K equals this value. Lets a layer-level sweep force one projection's tile
// (e.g. moe.down, K=512) while the other GEMMs in the captured graph keep
// their cascade picks. Added 2026-09-04 after the Family C sweep showed that
// kernel-bench tile rankings do not transfer to the layer for the grouped
// down GEMM, so the layer cell is the measure. Read once per process.
inline bool force_tile_applies(ForcedTile const& forced, uint32_t shape_k) noexcept
{
    static int const s_k = []
    {
        char const* e = std::getenv("FSO_FORCE_TILE_K");
        return (e && *e) ? std::atoi(e) : 0;
    }();
    return forced.active() && (s_k <= 0 || s_k == static_cast<int>(shape_k));
}

// FSO_MOE_GROUP_ORDER -- the order the grouped MoE scheduler visits experts in.
//   unset / 0 / 1   expert-id order (the default, and what every measurement
//                   before 2026-09-27 was taken with)
//   "auto"          a stride of about a quarter of the walk, so four consecutive
//                   steps come from four different quarters of the expert set
//   <n>             that exact stride, if it is coprime with the number of steps
// FSO_MOE_GROUP_BLOCK=<w> -- how many CONSECUTIVE experts one step of that walk
// carries (default 1). The permutation applies to blocks of w experts and the
// experts inside a block stay in id order, so the weight and activation streams
// stay sequential for w slots at a time while the blocks themselves are spread
// across the set. w must divide the expert count.
//
// The stride must be coprime with the number of blocks or the walk would repeat
// some of them and miss others; `sm120_group_perm_for` below enforces that and
// falls back to expert-id order (with a warning) when it cannot.
//
// Why this knob exists: the layer is bound by expert weight traffic, and an
// expert costs its whole weight matrix however few rows it holds. A contiguous
// run of low-row experts therefore becomes a phase of the launch with too little
// arithmetic to cover those loads. Measured on Family C at M=1024, moving the
// high-row experts from a leading block to every fourth id took the skew penalty
// from +4.46 % to +1.37 % with the row counts unchanged. This knob is the kernel
// side of the same idea: it decorrelates schedule position from expert id, so it
// helps exactly when the counts are correlated with the id and is a no-op
// otherwise.
inline int read_group_order_env() noexcept
{
    static int const s_cache = []
    {
        char const* e = std::getenv("FSO_MOE_GROUP_ORDER");
        if (!e || !*e)
            return 1;
        if (std::strcmp(e, "auto") == 0)
            return -1;
        int const v = std::atoi(e);
        return v <= 0 ? 1 : v;
    }();
    return s_cache;
}

inline int read_group_block_env() noexcept
{
    static int const s_cache = []
    {
        char const* e = std::getenv("FSO_MOE_GROUP_BLOCK");
        int const v = (e && *e) ? std::atoi(e) : 1;
        return v < 1 ? 1 : v;
    }();
    return s_cache;
}

// The resolved walk: `block` consecutive experts per step, `stride` steps apart.
// stride == 1 means expert-id order, whatever the block size.
struct GroupOrder
{
    int stride{1};
    int block{1};
};

inline GroupOrder sm120_group_perm_for(int num_groups) noexcept
{
    int const req = read_group_order_env();
    if (req == 1 || num_groups < 8)
        return {};
    int const w = read_group_block_env();
    if (w < 1 || num_groups % w != 0)
    {
        std::fprintf(stderr, "[fso] FSO_MOE_GROUP_BLOCK=%d does not divide %d experts; using expert-id order\n", w,
            num_groups);
        return {};
    }
    int const nb = num_groups / w; // the permutation runs over blocks, not experts
    if (nb < 8)
    {
        std::fprintf(stderr, "[fso] FSO_MOE_GROUP_BLOCK=%d leaves only %d blocks of %d experts; using expert-id order\n",
            w, nb, num_groups);
        return {};
    }
    auto coprime = [](int a, int b)
    {
        while (b)
        {
            int const t = a % b;
            a = b;
            b = t;
        }
        return a == 1;
    };
    if (req > 1)
    {
        if (req < nb && coprime(req, nb))
            return {req, w};
        std::fprintf(stderr, "[fso] FSO_MOE_GROUP_ORDER=%d is not coprime with %d blocks; using expert-id order\n", req,
            nb);
        return {};
    }
    for (int a = nb / 4 + 1; a < nb; ++a)
        if (coprime(a, nb))
            return {a, w};
    return {};
}

// Reads FSO_DISABLE_OVERRIDES once. Returns true when the dispatcher
// should skip its K-aware short-K / small-M overrides.
inline bool read_disable_overrides() noexcept
{
    static int s_cache = -1;
    if (s_cache == -1)
    {
        char const* env = std::getenv("FSO_DISABLE_OVERRIDES");
        s_cache = (env && std::atoi(env) == 1) ? 1 : 0;
    }
    return s_cache != 0;
}


// FSO_DISABLE_PDL=1 turns off programmatic dependent launch attributes on
// every fso MoE-chain launch (kernel-side griddepcontrol wait/trigger then
// degrade to no-ops). Read once per process.
inline bool fso_pdl_enabled()
{
    static bool const v = []
    {
        char const* e = std::getenv("FSO_DISABLE_PDL");
        return !(e && e[0] == '1');
    }();
    return v;
}

} // namespace kernels::blockscale_gemm
} // namespace tensorrt_llm
