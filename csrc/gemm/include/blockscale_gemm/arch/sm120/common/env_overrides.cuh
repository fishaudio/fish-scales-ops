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
//   FSO_FORCE_KSPLIT="N"           Stream-K split, optional
//   FSO_DISABLE_OVERRIDES=1        skip K-aware single-launch overrides
//
// Same wire format on both paths so a single sweep harness can drive both.
// See docs/skills/blockscale-gemm-tuning/references/dispatcher-overrides.md.

#pragma once

#include <cstdio>
#include <cstdlib>

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
