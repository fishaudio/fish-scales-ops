/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SM100/SM103 MXFP8 (1×32, SFVecSize=32) dispatch and kernel launcher.
//
// Mirrors the sm_120 dispatcher's host-overhead discipline:
//   * E9-equivalent: a thread-local cache of *initialized*
//     GemmUniversalAdapter instances keyed by (ptrs, shape). On a hit the
//     call is just adapter.run(stream) — no cuTensorMapEncodeTiled, no
//     cudaFuncSetAttribute (the adapter does both only in initialize()).
//   * E8-equivalent: cudaFuncSetAttribute happens inside
//     Gemm::initialize(), i.e. only on cache miss. CUDA-graph capture is
//     therefore safe as long as callers warm up eagerly first — the same
//     contract test_cuda_graph.py enforces for sm_120.
//   * Workspace: the dense BlockScaled kernel with the default tile
//     scheduler needs no global workspace on the shapes we serve; if
//     CUTLASS ever asks for one we grow a thread-local pool OUTSIDE
//     capture (same contract as Sm120BfPackPool) and abort with a clear
//     message if a capture-mode call needs more.
//
// Cascade status: bring-up heuristic only (C1). The (M, tiles_n) rules
// below are placeholders pending the b300 tile sweep (FSO_FORCE_TILE grid
// + cudagraph-µs bench per docs/perf/README.md methodology). Retune before
// trusting any cell.
//
// Env overrides: FSO_FORCE_TILE=TM,TN,ST reuses the sm_120 wire format so
// the shared sweep harness drives this path unchanged. For sm_100 the
// third field selects the (SM-count, TileK, cluster, scheduler) variant
// instead of the stage count (stages are StageCountAutoCarveout-derived):
//   ST=1 → 1SM cluster(1,1) K128          ST=2 → 2SM cluster(2,1) K128
//   ST=3 → 1SM cluster(1,1) K256          ST=4 → 2SM cluster(2,1) K256
//   ST=5 → 2SM cluster(2,2) K128          ST=6 → 2SM cluster(2,2) K256
//   ST=7 → 1SM cluster(1,1) K128 StreamK  ST=8 → 1SM cluster(1,1) K256 StreamK
//   ST=9 → 1SM parallel split-K (two-kernel; splits from FSO_FORCE_KSPLIT,
//          else auto pick_splits) — (128,128) only
//   ST=10 → 1SM cluster(2,2) K128 (TMA multicast ×2 on both A and B — the
//           nvjet mid-band shape; ncu F3)      ST=11 → 2SM cluster(2,2) K256
//
// Narrow / 192-wide N tiles (added 2026-09-17). Codes 1..11 above bake a
// particular epilogue choice into each row, which is fine for the tiles the
// cascade already ships but useless for sweeping a new tile, where the
// epilogue is one of the things being decided. Codes 20..25 therefore name
// the (SM-count, TileK, cluster) variant AND the epilogue explicitly, and
// accept any TileN the kernel supports (64, 128, 192, 256):
//   ST=20 → 1SM cluster(1,1) K128, NoSmem (direct-store) epilogue
//   ST=21 → 1SM cluster(1,1) K128, TMA (smem-staged) epilogue
//   ST=22 → 1SM cluster(1,1) K256, NoSmem epilogue
//   ST=23 → 1SM cluster(1,1) K256, TMA epilogue
//   ST=24 → 2SM cluster(2,1) K128, NoSmem epilogue
//   ST=25 → 2SM cluster(2,1) K128, TMA epilogue
// Instantiated TileN per code (the set the 2026-09-17 sweep needed; adding
// one costs a full CUTLASS kernel's compile time, so the list is explicit):
//   ST=20/21: TileN 64, 128, 192      ST=22/23: TileN 64
//   ST=24/25: TileN 64, 192
//
// FSO_PRINT_TILE_INFO=1 makes every instantiation print, once, the mainloop
// stage count StageCountAutoCarveout derived for it and its shared-memory
// footprint — the quantity that decides whether a narrower TileN bought
// pipeline depth or only CTAs.
//
// FSO_FORCE_TILE_K=<K> restricts the override to GEMMs with that K, so a
// layer-level A/B can move one projection's tile while the others keep their
// cascade picks (same contract as the sm_120 dispatchers).
//
// Unknown combos fall through to the cascade (same behavior as sm_120).

#pragma once

#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#include "blockscale_gemm/arch/sm100/mxfp8/gemm_types.cuh"
#include "blockscale_gemm/arch/sm120/common/env_overrides.cuh" // FSO_FORCE_TILE parsing (arch-agnostic)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace sm100_blockscaled_gemm
{

namespace detail
{

// FSO_FORCE_SWIZZLE: raster swizzle group size for the default (CLC)
// scheduler — sweep knob for L2 locality on large shapes. 0 = CUTLASS
// heuristic default. Read once (function-static), same contract as the
// other FSO_* overrides.
inline int read_force_swizzle() noexcept
{
    static int s_cache = -2;
    if (s_cache == -2)
    {
        char const* env = std::getenv("FSO_FORCE_SWIZZLE");
        s_cache = (env && *env) ? std::atoi(env) : 0;
    }
    return s_cache;
}

// FSO_PRINT_TILE_INFO: print, once per kernel instantiation, the mainloop
// stage count the CollectiveBuilder's StageCountAutoCarveout derived and the
// shared memory the kernel ends up asking for. Host-side, read once, and off
// unless the variable is set, so it costs one int compare on the hot path.
inline bool read_print_tile_info() noexcept
{
    static int s_cache = -1;
    if (s_cache < 0)
    {
        char const* env = std::getenv("FSO_PRINT_TILE_INFO");
        s_cache = (env && *env && env[0] != '0') ? 1 : 0;
    }
    return s_cache != 0;
}

// Multiprocessor count of the current device, read once. The rule below is
// written against the count the driver reports rather than the 148 of a whole
// B300, so a MIG slice gets its own answer.
inline int sm_count() noexcept
{
    static int s_sms = 0;
    if (s_sms == 0)
    {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&s_sms, cudaDevAttrMultiProcessorCount, dev);
        if (s_sms <= 0)
            s_sms = 148;
    }
    return s_sms;
}

// CTAs a (cluster row count, CTAs per cluster, TileN) choice launches for an
// (M, N) problem. A 1-SM tile covers 128 rows with one CTA; a 2-SM tile covers
// 256 rows with a two-CTA cluster.
inline int grid_ctas(int M, int N, int cluster_rows, int ctas_per_cluster, int tile_n) noexcept
{
    return ((M + cluster_rows - 1) / cluster_rows) * ((N + tile_n - 1) / tile_n) * ctas_per_cluster;
}

// N-tile chosen by wave arithmetic. Returns TileM * 1000 + TileN, or 0 when
// this (M, N) is not a cell the rule owns.
//
// Why the CTA count is the quantity that decides: the block-scaled mainloop
// stages 200-230 KB of shared memory per CTA (printed per instantiation by
// FSO_PRINT_TILE_INFO), more than half of the SM's shared memory, so exactly
// one CTA is resident per SM. The CTA count is therefore the number of SMs
// that do any work at all, and in the decode band -- where the problem is one
// or two 128-row M-tiles -- it is decided by the N tile alone. Halving TileN
// doubles the CTA count and costs about 1.47x in per-SM work rate (ncu fit,
// 2026-09-17), a net 1.36x, but only while SMs are idle: past one wave the
// extra CTAs queue behind the first and the narrower tile is pure loss. So:
// take the tile that puts the most CTAs on the machine without spilling into a
// second wave.
//
// Four bounds, each from a measured counter-example rather than from the model:
//   * M >= 64. Below it the A-operand tile is almost entirely out-of-bounds
//     fill and every extra CTA repeats that fill, so the same kernel gets
//     slower the more CTAs it launches (gate_up M = 1, isolated GEMM: 14.46 us
//     on 76 CTAs, 14.85 on 102, 24.63 on 304), and at M = 32 the wave-picked
//     tile is 25 % slower than tier 1 on wqkv and 18 % on gate_up.
//   * At least half the SMs must end up busy. Every measured win puts 80 to
//     144 CTAs on the machine; the very narrow Family C projections
//     (N = 1024, N = 2048) would get a 32- or 64-CTA grid, outside the range
//     this rule was calibrated on.
//   * TileN = 64 only while the problem is a single 128-row M-tile (M <= 128).
//     At M = 256 the 64-wide tile is a tie with tier 1 on the isolated GEMM
//     (`wo` 8.27 against 8.26, `down` 14.39 against 14.40) and 3.5 % SLOWER on
//     the Family A MLP layer cell, where `down` runs straight after the SwiGLU
//     quantize. At M = 512 with N = 2048 it doubles the CTA count to 128 and is
//     still 10 % slower than tier 1, so the rate penalty of the narrow atom is
//     not always repaid. The 192-wide tile has no such counter-example up to
//     M = 1024.
//   * The 2-SM 192-wide form is not shipped. The only cells where it is the
//     CTA-maximising pick are Family C `gdn.in_proj` (N = 12288) at M = 64 and
//     M = 128, and there it measured 1.3x SLOWER than tier 1 even though it
//     puts 128 CTAs on the machine against cuBLAS's 96; every 192-wide win in
//     the sweep came from the 1-SM form.
//
// Ties (the same CTA count from a 1-SM and a 2-SM tile) go to the 2-SM tile.
// At an equal CTA count the two forms keep the same number of SMs busy, but
// the 2-SM form covers 256 rows per tile instead of 128, so the kernel makes
// half as many passes over the B operand -- the weight matrix. Timed on their
// own the two forms agree to within the replay tick (`wo` and `down` at
// M = 1024: 12.40 / 22.57 us for 1-SM against 12.37 / 22.55 for 2-SM), because
// an isolated dense cell replays one GEMM 150 times and its whole working set
// (10 MB of activations plus a 25 MB weight at `down` M = 1024) stays resident
// in this part's 132 MB L2, which makes the extra passes free. In a fused
// layer they are not free: the Family A MLP block runs `down` straight after
// `gate_up`, whose 50 MB weight and 40 MB BF16 output have just swept L2, so
// `down` starts with a cold weight and every extra pass over it is HBM
// traffic. Measured at M = 1024, N = 2560 with nsys inside the captured block
// (per-launch median): the 1-SM 192-wide tile costs 21.38 us on its own but
// 25.44 us in the block, while the 2-SM 192-wide tile costs 20.90 us on its
// own and 21.10 us in the block (run b300_head_vs_wt_20260921/R2).
inline int pick_wave_tile(int M, int N) noexcept
{
    if (M < 64 || M > 1024)
        return 0;
    int const sms = sm_count();
    int const cand_n[4] = {256, 192, 128, 64};
    int best_m = 0, best_n = 0, best_ctas = 0;
    for (int i = 0; i < 4; ++i)
    {
        int const ctas = grid_ctas(M, N, 128, 1, cand_n[i]);
        if (ctas <= sms && ctas > best_ctas) { best_ctas = ctas; best_m = 128; best_n = cand_n[i]; }
    }
    for (int i = 0; i < 4; ++i)
    {
        int const ctas = grid_ctas(M, N, 256, 2, cand_n[i]);
        if (ctas <= sms && ctas > best_ctas) { best_ctas = ctas; best_m = 256; best_n = cand_n[i]; }
    }
    if (best_n != 64 && best_n != 192)
        return 0;                 // the pick is a tile the cascade already had
    if (2 * best_ctas < sms)
        return 0;                 // narrowing would still leave the machine half idle
    if (best_n == 64 && M > 128)
        return 0;                 // tie at M = 256, loses at M = 512
    if (best_m == 256 && best_n == 192)
        return 0;                 // measured counter-example at gdn.in_proj M = 64 / 128
    // Equal-CTA tie-break, applied after the guards above so that each of their
    // rejections keeps its exact meaning: when the 2-SM form of the SAME N
    // width launches the same number of CTAs, take it, for the halved number of
    // passes over the weight matrix explained in the comment above. The test is
    // an equality on CTA counts, so it can only fire above M = 128 (below it a
    // 2-SM tile always doubles the count) and it never changes which cells the
    // rule owns -- only which tile an already-owned cell gets.
    if (best_m == 128 && grid_ctas(M, N, 256, 2, best_n) == best_ctas)
        best_m = 256;
    return best_m * 1000 + best_n;
}

// Thread-local CUTLASS workspace pool. Expected to stay empty (dense
// BlockScaled + default scheduler needs no workspace); exists so a future
// scheduler change degrades to a clear runtime contract instead of UB.
struct Sm100WorkspacePool
{
    void* ptr = nullptr;
    std::size_t bytes = 0;

    static Sm100WorkspacePool& instance()
    {
        static thread_local Sm100WorkspacePool p;
        return p;
    }

    void* ensure(std::size_t needed)
    {
        if (needed == 0)
            return nullptr;
        if (bytes >= needed)
            return ptr;
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(nullptr, &cap);
        if (cap == cudaStreamCaptureStatusActive)
        {
            std::fprintf(stderr,
                "[blockscale_gemm] Sm100WorkspacePool: needs %zu bytes during stream capture but pool has %zu. "
                "Warm up linear_mxfp8 on the largest shape before capture.\n",
                needed, bytes);
            std::abort();
        }
        if (ptr)
            cudaFree(ptr);
        cudaMalloc(&ptr, needed);
        bytes = needed;
        return ptr;
    }
};

} // namespace detail

// Launch one (TileM, TileN, ClusterM, ClusterN) instantiation. Returns
// cudaSuccess on success; kernel runtime errors surface on the next sync
// (post-launch cudaGetLastError check intentionally omitted — see the E8
// note in the sm_120 launcher).
//
// Capture-safety split (mirror of the sm_120 E8/E9 discipline):
//   * cudaFuncSetAttribute → per-instantiation static guard, fires on the
//     FIRST call only. Callers must warm up eagerly before capture.
//   * Params (TMA descriptors) → thread-local cache. A cache MISS during
//     capture is safe: to_underlying_arguments is host-only work, and
//     GemmUniversalAdapter's *static* run(params, stream) issues nothing
//     but the (cluster-)launch itself.
template <int TileM, int TileN, int ClusterM, int ClusterN, int TileK = 128, bool UseStreamK = false,
    bool NoSmemEpi = false>
cudaError_t launch_sm100_mxfp8_gemm(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int M, int N, int K, cudaStream_t stream, int swizzle = 0)
{
    using Config = Sm100MxFP8GemmConfig<TileM, TileN, ClusterM, ClusterN, TileK, UseStreamK, NoSmemEpi>;
    using Gemm = typename Config::Gemm;
    using GemmKernel = typename Config::GemmKernel;
    using Args = typename Gemm::Arguments;
    using Params = typename Gemm::Params;

    if (detail::read_print_tile_info())
    {
        static bool s_info_printed = false;
        if (!s_info_printed)
        {
            s_info_printed = true;
            std::fprintf(stderr,
                "[fso sm100 tile] %dx%dx%d cluster=(%d,%d) streamk=%d nosmem=%d "
                "stages=%d smem_bytes=%d epi_smem_bytes=%d\n",
                TileM, TileN, TileK, ClusterM, ClusterN, static_cast<int>(UseStreamK),
                static_cast<int>(NoSmemEpi), Config::kStages, Config::kSmemBytes,
                Config::kEpiSmemBytes);
        }
    }

    auto ptr_A = reinterpret_cast<cutlass::float_e4m3_t const*>(mat_a);
    auto ptr_B = reinterpret_cast<cutlass::float_e4m3_t const*>(mat_b);
    auto ptr_SFA = reinterpret_cast<cutlass::float_ue8m0_t const*>(scales_a);
    auto ptr_SFB = reinterpret_cast<cutlass::float_ue8m0_t const*>(scales_b);
    auto ptr_D = reinterpret_cast<cutlass::bfloat16_t*>(mat_d);

    auto problem = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M, N, 1});
    auto layout_SFA = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem);
    auto layout_SFB = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem);

    Args args{cutlass::gemm::GemmUniversalMode::kGemm, problem,
        {ptr_A, stride_A, ptr_B, stride_B, ptr_SFA, layout_SFA, ptr_SFB, layout_SFB},
        {{1.0f, 0.0f}, nullptr, stride_D, ptr_D, stride_D}};

    if constexpr (UseStreamK)
    {
        // splits > 1 bypasses the stream-K heuristic in favour of an
        // explicit split-K decomposition (the heuristic mode measured
        // WORSE than no split at all on down-class decode shapes —
        // 30.9 µs vs 17.5 µs at M=1, 2026-07-05 sweep). FSO_FORCE_KSPLIT
        // drives the sweep; the cascade sets nothing yet.
        int const ks = tensorrt_llm::kernels::blockscale_gemm::read_force_tile().ks;
        if (ks > 1)
            args.scheduler.splits = ks;
        // Probe knobs (2026-07-06, both read once into static caches):
        //   FSO_SK_NDET=1   → ReductionMode::Nondeterministic (atomic
        //                     accumulation; the 2026-07-05 losses were all
        //                     under the default Deterministic turnstile
        //                     lock, which serializes the fixup).
        //   FSO_SK_DECOMP   → 1=DataParallel 2=SplitK 3=StreamK (default
        //                     heuristic).
        static int const s_ndet = []() {
            char const* e = std::getenv("FSO_SK_NDET");
            return (e && *e) ? std::atoi(e) : 0;
        }();
        static int const s_decomp = []() {
            char const* e = std::getenv("FSO_SK_DECOMP");
            return (e && *e) ? std::atoi(e) : 0;
        }();
        using cutlass::gemm::kernel::detail::ReductionMode;
        using cutlass::gemm::kernel::detail::DecompositionMode;
        if (s_ndet)
            args.scheduler.reduction_mode = ReductionMode::Nondeterministic;
        if (s_decomp == 1)
            args.scheduler.decomposition_mode = DecompositionMode::DataParallel;
        else if (s_decomp == 2)
            args.scheduler.decomposition_mode = DecompositionMode::SplitK;
        else if (s_decomp == 3)
            args.scheduler.decomposition_mode = DecompositionMode::StreamK;
    }
    else
    {
        // Cascade-provided value; FSO_FORCE_SWIZZLE overrides for sweeps.
        int const sw = detail::read_force_swizzle();
        int const eff = sw > 0 ? sw : swizzle;
        if (eff > 0)
            args.scheduler.max_swizzle_size = eff;
        // FSO_FORCE_RASTER: 1 = AlongM, 2 = AlongN (default Heuristic).
        // Probe knob (2026-07-06): nvjet picks h_bz (horizontal raster) for
        // the narrow-N down-class at mid/large M where we lose ~17%.
        // PROBED NEUTRAL (raster_probe.txt): ±2% noise on every mid/large-M
        // cell; gate_up AlongN −5%. The CLC Heuristic already picks right —
        // do not wire into the cascade.
        static int const s_raster = []() {
            char const* e = std::getenv("FSO_FORCE_RASTER");
            return (e && *e) ? std::atoi(e) : 0;
        }();
        using cutlass::gemm::kernel::detail::RasterOrderOptions;
        if (s_raster == 1)
            args.scheduler.raster_order = RasterOrderOptions::AlongM;
        else if (s_raster == 2)
            args.scheduler.raster_order = RasterOrderOptions::AlongN;
    }

    // E9-style cache: key on everything the initialized Params bake in.
    struct CacheKey
    {
        void const* p[5];
        int dims[4];  // M, N, K, swizzle
        bool operator==(CacheKey const& o) const { return std::memcmp(this, &o, sizeof(o)) == 0; }
    };
    struct CacheHash
    {
        std::size_t operator()(CacheKey const& k) const noexcept
        {
            std::size_t h = 1469598103934665603ULL;
            auto const* p = reinterpret_cast<unsigned char const*>(&k);
            for (std::size_t i = 0; i < sizeof(k); ++i) { h ^= p[i]; h *= 1099511628211ULL; }
            return h;
        }
    };
    static thread_local std::unordered_map<CacheKey, Params, CacheHash> s_params_cache;
    CacheKey key;
    std::memset(&key, 0, sizeof(key));
    key.p[0] = ptr_A; key.p[1] = ptr_B; key.p[2] = ptr_SFA; key.p[3] = ptr_SFB; key.p[4] = ptr_D;
    key.dims[0] = M; key.dims[1] = N; key.dims[2] = K; key.dims[3] = swizzle;

    Params kernel_params;
    auto it = s_params_cache.find(key);
    if (it != s_params_cache.end())
    {
        kernel_params = it->second;
        if constexpr (UseStreamK)
        {
            // Stream-K reduction barriers must be clean before every launch;
            // re-enqueue the (stream-ordered, capture-legal) workspace init
            // even on a Params-cache hit. A captured init simply re-runs on
            // every replay — exactly the semantics the barriers need.
            std::size_t const ws_bytes = Gemm::get_workspace_size(args);
            void* ws = detail::Sm100WorkspacePool::instance().ensure(ws_bytes);
            if (ws_bytes > 0
                && GemmKernel::initialize_workspace(args, ws, stream) != cutlass::Status::kSuccess)
                return cudaErrorUnknown;
        }
    }
    else
    {
        if (Gemm::can_implement(args) != cutlass::Status::kSuccess)
            return cudaErrorInvalidValue;
        std::size_t const ws_bytes = Gemm::get_workspace_size(args);
        void* ws = detail::Sm100WorkspacePool::instance().ensure(ws_bytes);
        if (ws_bytes > 0)
        {
            // Stream-ordered workspace init (capture-legal; a captured init
            // simply re-runs on every replay, which is the semantics the
            // scheduler workspace wants anyway).
            if (GemmKernel::initialize_workspace(args, ws, stream) != cutlass::Status::kSuccess)
                return cudaErrorUnknown;
        }
        kernel_params = GemmKernel::to_underlying_arguments(args, ws); // host-only
        if (s_params_cache.size() < 64u)
            s_params_cache.emplace(key, kernel_params);
    }

    // E8-equivalent: set max dynamic smem once per instantiation, on the
    // first (eager) call. GemmUniversalAdapter::initialize would do this on
    // every cache miss — including mid-capture — so we do it ourselves.
    static bool s_smem_configured = false;
    if (!s_smem_configured)
    {
        if (GemmKernel::SharedStorageSize >= (48 << 10))
        {
            cudaError_t result = cudaFuncSetAttribute(cutlass::device_kernel<GemmKernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, GemmKernel::SharedStorageSize);
            if (result != cudaSuccess)
                return result;
        }
        s_smem_configured = true;
    }

    return Gemm::run(kernel_params, stream) == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorLaunchFailure;
}

// ---------------------------------------------------------------------------
// Parallel split-K (two-kernel) decode path.
//
// ncu 2026-07-06 (profile/ncu_sm100_2026-07-06/REPORT.md, F1): at down-class
// decode shapes (N=2560 → 20 tile-CTAs on 148 SMs) cuBLAS runs the SAME
// 128×128 tile as a data-parallel split-K×3 (grid 60) plus a trivial
// splitKreduce kernel, and wins ~0.75×. CUTLASS's StreamK scheduler is the
// wrong kind of split-K here (in-kernel semaphore fixup: heuristic 30.9 µs,
// splits=2 18.7 µs vs 16.7 plain). This path reproduces the cuBLAS scheme:
//   1. ONE batched GEMM launch (L = splits) writing FP32 partials to the
//      workspace pool — batch b covers K-slice [b*K/S, (b+1)*K/S). A/B are
//      addressed in-place via batch strides (K/S along K); the SF tensors
//      keep the full-K quantize layout via stride surgery (see below).
//   2. A float4 grid-stride reduce kernel: partials [S, M, N] → BF16 D.
// Both launches are capture-safe (workspace from the pool, eager-warmup
// contract unchanged).
//
// SF layout stride surgery: tile_atom_to_shape_SFA((M,N,K/S,S)) has the
// right SHAPE for a K-slice batch, but its strides describe S independent
// contiguous tensors. In the full-K quantize layout (K-minor blocks per
// 128-row block), the correct strides are:
//   * M-block-row stride = 512 * (K/128)      = default * S
//   * batch (L) stride   = 512 * (K/(128*S))  = the DEFAULT M-block-row stride
// so the fix is: stride(Mrest) *= S; stride(L) = old stride(Mrest).

// Programmatic dependent launch (PDL) for the two-kernel split-K pair.
//
// The reduce kernel below reads the FP32 partials the batched split-K GEMM
// writes, so it must not touch them before that GEMM has finished. Without
// PDL the driver enforces that by serialising the two launches outright: the
// reduce's CTAs are only created after the GEMM's last CTA retires, so the
// reduce's entire launch and prologue latency sits exposed between the two
// kernels. cuBLAS's own split-K pair does not pay that cost -- its reduce is
// observed starting while its GEMM is still running.
//
// With `cudaLaunchAttributeProgrammaticStreamSerialization` set on the reduce
// launch the driver may create and schedule the reduce's CTAs while the GEMM's
// tail is still draining, and `cudaGridDependencySynchronize()` inside the
// kernel then holds every CTA until the GEMM has actually completed. The
// ordering the data dependency requires is therefore unchanged; only the
// launch and the prologue overlap.
//
// The producer is a CUTLASS kernel we do not modify, so its completion signal
// is the implicit one at kernel exit. That is the safe case: fso does not
// build CUTLASS with CUTLASS_ENABLE_GDC_FOR_SM100, so the
// `cutlass::arch::launch_dependent_grids()` call that sits between the CUTLASS
// mainloop and its epilogue stores compiles to nothing and cannot release the
// reduce before the partials have landed.
//
// The reduce deliberately does not call cudaTriggerProgrammaticLaunchCompletion():
// it writes the caller's D buffer throughout its own execution, so there is no
// earlier point at which it could safely signal a future dependent.
//
// FSO_DISABLE_PDL=1 restores the plain serialised launch (the same variable
// the sm_120 MoE chain reads, so one knob disables PDL everywhere).
__device__ __forceinline__ void sm100_pdl_wait(bool pdl)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (pdl)
        cudaGridDependencySynchronize();
#else
    // griddepcontrol needs sm_90 or newer. This header only compiles under
    // CUTLASS_ARCH_MMA_SM100_SUPPORTED so the branch is unreachable today; it
    // keeps the file compilable if that guard is ever widened to an older arch.
    (void) pdl;
#endif
}

static __global__ void sm100_mxfp8_splitk_reduce_kernel(
    float4 const* __restrict__ partials, ushort4* __restrict__ out, int splits, long mn4, bool pdl)
{
    sm100_pdl_wait(pdl);
    long const stride = static_cast<long>(gridDim.x) * blockDim.x;
    for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; i < mn4; i += stride)
    {
        float4 acc = partials[i];
        for (int s = 1; s < splits; ++s)
        {
            float4 const p = partials[s * mn4 + i];
            acc.x += p.x; acc.y += p.y; acc.z += p.z; acc.w += p.w;
        }
        ushort4 o;
        o.x = __bfloat16_as_ushort(__float2bfloat16(acc.x));
        o.y = __bfloat16_as_ushort(__float2bfloat16(acc.y));
        o.z = __bfloat16_as_ushort(__float2bfloat16(acc.z));
        o.w = __bfloat16_as_ushort(__float2bfloat16(acc.w));
        out[i] = o;
    }
}

namespace detail
{

// Largest split count in [2, min(8, sm_count / tiles)] that divides the SF
// K-block count evenly (SF blocks are 128-K-wide and cannot straddle a
// slice). Returns 1 when no useful split exists — including when the tile
// grid already fills the machine, which self-limits the route to decode.
inline int pick_splits(int tiles, int kblks) noexcept
{
    static int s_sms = 0;
    if (s_sms == 0)
    {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&s_sms, cudaDevAttrMultiProcessorCount, dev);
        if (s_sms <= 0)
            s_sms = 148;
    }
    int const budget = s_sms / (tiles > 0 ? tiles : 1);
    for (int s = budget < 8 ? budget : 8; s >= 2; --s)
        if (kblks % s == 0)
            return s;
    return 1;
}

} // namespace detail

// Batched-slice GEMM into FP32 partials + reduce. Caller guarantees
// K % 128 == 0 and (K/128) % splits == 0 (pick_splits does).
template <int TileM, int TileN>
cudaError_t launch_sm100_mxfp8_gemm_splitk(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int M, int N, int K, int splits, cudaStream_t stream)
{
    // NoSmemEpi: decode band, partial stores are a sliver (see the M ≤ 32
    // NOSMEM rationale in the cascade). ElementD float → FP32 partials.
    using Config = Sm100MxFP8GemmConfig<TileM, TileN, 1, 1, 128, false, true, float>;
    using Gemm = typename Config::Gemm;
    using GemmKernel = typename Config::GemmKernel;
    using Args = typename Gemm::Arguments;
    using Params = typename Gemm::Params;

    if (detail::read_print_tile_info())
    {
        static bool s_info_printed = false;
        if (!s_info_printed)
        {
            s_info_printed = true;
            std::fprintf(stderr,
                "[fso sm100 tile] splitk %dx%dx128 cluster=(1,1) fp32-partials "
                "stages=%d smem_bytes=%d epi_smem_bytes=%d\n",
                TileM, TileN, Config::kStages, Config::kSmemBytes, Config::kEpiSmemBytes);
        }
    }

    auto ptr_A = reinterpret_cast<cutlass::float_e4m3_t const*>(mat_a);
    auto ptr_B = reinterpret_cast<cutlass::float_e4m3_t const*>(mat_b);
    auto ptr_SFA = reinterpret_cast<cutlass::float_ue8m0_t const*>(scales_a);
    auto ptr_SFB = reinterpret_cast<cutlass::float_ue8m0_t const*>(scales_b);

    int const kslice = K / splits;
    long const mn = static_cast<long>(M) * N;

    // Workspace: FP32 partials [splits, M, N], plus whatever CUTLASS asks
    // for (expected 0 with the default CLC scheduler) appended behind.
    std::size_t const partial_bytes = static_cast<std::size_t>(splits) * mn * sizeof(float);

    auto problem = cute::make_shape(M, N, kslice, splits);
    // In-place K-slices: leading strides stay full-K, batch stride = kslice.
    auto stride_A = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, kslice, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, kslice, 1});
    cute::get<0>(stride_A) = K;
    cute::get<2>(stride_A) = kslice;
    cute::get<0>(stride_B) = K;
    cute::get<2>(stride_B) = kslice;
    // Partials are genuinely batched [S, M, N] — packed stride.
    auto stride_D = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M, N, splits});

    // SF layouts: shape from the sliced problem, strides fixed up per the
    // header comment (Mrest *= S; L = old Mrest). In-place mutation of a
    // by-value copy keeps the exact LayoutSFA/LayoutSFB types.
    // LayoutSF{A,B} structure: shape (((32,4),Mrest), ((32,4),Krest), (1,L))
    // stride (((16,4),sMrest), ((0,1),512), (0,sL)) — the scalars to patch
    // are get<0,1> (M-block-row) and get<2,1> (batch).
    auto fix_sf = [splits](auto l) {
        auto const row = cute::get<0, 1>(l.stride());
        cute::get<0, 1>(l.stride()) = row * splits;
        cute::get<2, 1>(l.stride()) = row;
        return l;
    };
    typename Config::LayoutSFA layout_SFA
        = fix_sf(Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem));
    typename Config::LayoutSFB layout_SFB
        = fix_sf(Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem));

    // Params cache (E9 discipline; ws ptr participates via ptr_D).
    struct CacheKey
    {
        void const* p[5];
        int dims[4];  // M, N, K, splits
        bool operator==(CacheKey const& o) const { return std::memcmp(this, &o, sizeof(o)) == 0; }
    };
    struct CacheHash
    {
        std::size_t operator()(CacheKey const& k) const noexcept
        {
            std::size_t h = 1469598103934665603ULL;
            auto const* p = reinterpret_cast<unsigned char const*>(&k);
            for (std::size_t i = 0; i < sizeof(k); ++i) { h ^= p[i]; h *= 1099511628211ULL; }
            return h;
        }
    };
    static thread_local std::unordered_map<CacheKey, Params, CacheHash> s_params_cache;

    // The pool ptr is stable unless it grows; growth only happens outside
    // capture (ensure() aborts otherwise), and the ptr is part of the key.
    Args args{cutlass::gemm::GemmUniversalMode::kGemm, problem,
        {ptr_A, stride_A, ptr_B, stride_B, ptr_SFA, layout_SFA, ptr_SFB, layout_SFB},
        {{1.0f, 0.0f}, nullptr, stride_D, nullptr /* ptr_D patched below */, stride_D}};

    std::size_t const ws_bytes = Gemm::get_workspace_size(args);
    void* base = detail::Sm100WorkspacePool::instance().ensure(partial_bytes + ws_bytes);
    if (base == nullptr)
        return cudaErrorMemoryAllocation;
    float* partials = static_cast<float*>(base);
    void* cutlass_ws = static_cast<char*>(base) + partial_bytes;
    args.epilogue.ptr_D = partials;

    CacheKey key;
    std::memset(&key, 0, sizeof(key));
    key.p[0] = ptr_A; key.p[1] = ptr_B; key.p[2] = ptr_SFA; key.p[3] = ptr_SFB; key.p[4] = partials;
    key.dims[0] = M; key.dims[1] = N; key.dims[2] = K; key.dims[3] = splits;

    Params kernel_params;
    auto it = s_params_cache.find(key);
    if (it != s_params_cache.end())
        kernel_params = it->second;
    else
    {
        // NOTE: Gemm::can_implement is intentionally SKIPPED. The mainloop's
        // check literally recomputes tile_atom_to_shape_SFA(problem) and
        // compares strides, so the split-K stride surgery always "fails" it
        // even though to_underlying_arguments consumes args.layout_SFA/SFB
        // verbatim (verified: make_tensor(args.ptr_SFA, args.layout_SFA) →
        // TMA descriptor). The constraints it would enforce hold by
        // construction here: K % 128 == 0, kslice % 128 == 0 (16B TMA
        // alignment on every batch base), N % 4 == 0 for the FP32 partials.
        if (ws_bytes > 0
            && GemmKernel::initialize_workspace(args, cutlass_ws, stream) != cutlass::Status::kSuccess)
            return cudaErrorUnknown;
        kernel_params = GemmKernel::to_underlying_arguments(args, cutlass_ws); // host-only
        if (s_params_cache.size() < 64u)
            s_params_cache.emplace(key, kernel_params);
    }

    static bool s_smem_configured = false;
    if (!s_smem_configured)
    {
        if (GemmKernel::SharedStorageSize >= (48 << 10))
        {
            cudaError_t result = cudaFuncSetAttribute(cutlass::device_kernel<GemmKernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, GemmKernel::SharedStorageSize);
            if (result != cudaSuccess)
                return result;
        }
        s_smem_configured = true;
    }

    if (Gemm::run(kernel_params, stream) != cutlass::Status::kSuccess)
        return cudaErrorLaunchFailure;

    // Reduce: N is a multiple of 128 on this surface, so M*N % 4 == 0.
    long const mn4 = mn / 4;
    int const threads = 256;
    int const blocks = static_cast<int>((mn4 + threads - 1) / threads) < 1024
        ? static_cast<int>((mn4 + threads - 1) / threads)
        : 1024;
    // PDL on the reduce only (see the comment above the reduce kernel): the
    // GEMM keeps its ordinary launch, the reduce becomes its programmatic
    // dependent and waits on the implicit end-of-kernel trigger.
    bool const pdl = tensorrt_llm::kernels::blockscale_gemm::fso_pdl_enabled();
    if (pdl)
    {
        cudaLaunchConfig_t cfg{};
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.gridDim = dim3(blocks);
        cfg.blockDim = dim3(threads);
        cfg.dynamicSmemBytes = 0;
        cfg.stream = stream;
        cfg.attrs = attrs;
        cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, sm100_mxfp8_splitk_reduce_kernel,
            reinterpret_cast<float4 const*>(partials), reinterpret_cast<ushort4*>(mat_d), splits, mn4, true);
    }
    else
    {
        sm100_mxfp8_splitk_reduce_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float4 const*>(partials), reinterpret_cast<ushort4*>(mat_d), splits, mn4, false);
    }
    return cudaGetLastError();
}

// Top-level sm_100/sm_103 MXFP8 dispatcher.
//
// Bring-up cascade (C1, placeholder — retune on b300 hardware):
//   * decode / small M          → (64, 128) 1SM
//   * mid M                     → (128, 128) 1SM;  wide N → (128, 256) 1SM
//   * large M (> 256)           → (256, 128) 2SM;  wide N → (256, 256) 2SM
// No Stream-K: the CLC-based persistent scheduler already load-balances
// tail waves; revisit only if the b300 sweep shows narrow-N long-K decode
// shapes losing to a K-split (compare against sm_120's K≥9728 gate).
inline cudaError_t gemm_dispatch_sm100_mxfp8(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int M, int N, int K, cudaStream_t stream)
{
    // Epilogue binding (ncu A/B 2026-07-06, see gemm_types.cuh): NOSMEM
    // (direct-store) on the latency-bound decode tiles and the L2-headroom
    // peak tile; TMA epilogue everywhere L2 is near-bound (mid-band
    // (256,128)K256) or unmeasured (StreamK, forced-only variants, and the
    // (256,256)K128 *mid-band* narrow-N route — same tile as peak but bound
    // TMA until probed, hence the separate NOSMEM/TMA call sites below).
#define DISPATCH_SM100_MX(TM_, TN_, CM_, CN_, TK_, SK_) \
    launch_sm100_mxfp8_gemm<TM_, TN_, CM_, CN_, TK_, SK_>(mat_a, mat_b, mat_d, scales_a, scales_b, M, N, K, stream)
#define DISPATCH_SM100_MX_SW(TM_, TN_, CM_, CN_, TK_, SK_, SW_) \
    launch_sm100_mxfp8_gemm<TM_, TN_, CM_, CN_, TK_, SK_>(mat_a, mat_b, mat_d, scales_a, scales_b, M, N, K, stream, SW_)
#define DISPATCH_SM100_MX_NOSMEM(TM_, TN_, CM_, CN_, TK_, SK_) \
    launch_sm100_mxfp8_gemm<TM_, TN_, CM_, CN_, TK_, SK_, true>(mat_a, mat_b, mat_d, scales_a, scales_b, M, N, K, stream)
#define DISPATCH_SM100_MX_NOSMEM_SW(TM_, TN_, CM_, CN_, TK_, SK_, SW_) \
    launch_sm100_mxfp8_gemm<TM_, TN_, CM_, CN_, TK_, SK_, true>(mat_a, mat_b, mat_d, scales_a, scales_b, M, N, K, stream, SW_)

    // FSO_FORCE_TILE=TM,TN,ST — ST picks the variant (see header comment).
    // Valid tiles: TileM ∈ {128, 256} (SF Blk_MN=128 granularity on the M
    // axis); TileN ∈ {64, 128, 192, 256} (CUTLASS pads the SF block up on the
    // N axis), but only the (TileM, TileN, ST) rows written out below are
    // actually instantiated.
    auto forced = tensorrt_llm::kernels::blockscale_gemm::read_force_tile();
    if (forced.active() && forced.tm > 0
        && tensorrt_llm::kernels::blockscale_gemm::force_tile_applies(
            forced, static_cast<uint32_t>(K)))
    {
        int const tm = forced.tm, tn = forced.tn, st = forced.st;
        // Codes 20..25: explicit (SM count, TileK, cluster, epilogue) with a
        // free TileN — the sweep surface for the 64- and 192-wide tiles.
        if (st == 20 && tm == 128)
        {
            if (tn == 64) return DISPATCH_SM100_MX_NOSMEM(128, 64, 1, 1, 128, false);
            if (tn == 128) return DISPATCH_SM100_MX_NOSMEM(128, 128, 1, 1, 128, false);
            if (tn == 192) return DISPATCH_SM100_MX_NOSMEM(128, 192, 1, 1, 128, false);
        }
        if (st == 21 && tm == 128)
        {
            if (tn == 64) return DISPATCH_SM100_MX(128, 64, 1, 1, 128, false);
            if (tn == 128) return DISPATCH_SM100_MX(128, 128, 1, 1, 128, false);
            if (tn == 192) return DISPATCH_SM100_MX(128, 192, 1, 1, 128, false);
        }
        if (st == 22 && tm == 128 && tn == 64) return DISPATCH_SM100_MX_NOSMEM(128, 64, 1, 1, 256, false);
        if (st == 23 && tm == 128 && tn == 64) return DISPATCH_SM100_MX(128, 64, 1, 1, 256, false);
        if (st == 24 && tm == 256)
        {
            if (tn == 64) return DISPATCH_SM100_MX_NOSMEM(256, 64, 2, 1, 128, false);
            if (tn == 192) return DISPATCH_SM100_MX_NOSMEM(256, 192, 2, 1, 128, false);
        }
        if (st == 25 && tm == 256)
        {
            if (tn == 64) return DISPATCH_SM100_MX(256, 64, 2, 1, 128, false);
            if (tn == 192) return DISPATCH_SM100_MX(256, 192, 2, 1, 128, false);
        }
        // 1SM rows (TileM=128) — K128 rows carry the cascade's NOSMEM binding
        // so FSO_FORCE_TILE sweeps exercise the production instantiation.
        if (tm == 128 && tn == 128 && st == 1) return DISPATCH_SM100_MX_NOSMEM(128, 128, 1, 1, 128, false);
        if (tm == 128 && tn == 256 && st == 1) return DISPATCH_SM100_MX_NOSMEM(128, 256, 1, 1, 128, false);
        if (tm == 128 && tn == 128 && st == 3) return DISPATCH_SM100_MX(128, 128, 1, 1, 256, false);
        if (tm == 128 && tn == 256 && st == 3) return DISPATCH_SM100_MX(128, 256, 1, 1, 256, false);
        if (tm == 128 && tn == 128 && st == 7) return DISPATCH_SM100_MX(128, 128, 1, 1, 128, true);
        if (tm == 128 && tn == 128 && st == 8) return DISPATCH_SM100_MX(128, 128, 1, 1, 256, true);
        if (tm == 128 && tn == 128 && st == 9 && (K % 128) == 0)
        {
            int const tiles = ((M + 127) / 128) * ((N + 127) / 128);
            int const ks = forced.ks > 1 ? forced.ks : detail::pick_splits(tiles, K / 128);
            if (ks > 1 && (K / 128) % ks == 0)
                return launch_sm100_mxfp8_gemm_splitk<128, 128>(
                    mat_a, mat_b, mat_d, scales_a, scales_b, M, N, K, ks, stream);
            // No valid split → plain decode tile.
            return DISPATCH_SM100_MX_NOSMEM(128, 128, 1, 1, 128, false);
        }
        // 1SM cluster(2,2) probes (ncu F3: nvjet's gate M=2048 config is
        // 128×256/CTA with a 4-CTA cluster → half our L2 traffic).
        // NEGATIVE RESULT (2026-07-06 c22_probe): all three lose across the
        // whole mid-band — (128,256)c22 +15..28% (gate M2048 52.6 vs 41.0),
        // (128,128)c22 +9..20%, (256,128)c22-K256 +1..8%. The L2-traffic
        // saving does not survive the CLC scheduler's coarser 4-CTA
        // allocation granularity + cluster barriers. nvjet's mid-band edge
        // is scheduler/pipeline-level, not reachable by cluster shape alone.
        // Kept as force codes for future sweeps; do NOT wire into the
        // cascade.
        if (tm == 128 && tn == 128 && st == 10) return DISPATCH_SM100_MX(128, 128, 2, 2, 128, false);
        if (tm == 128 && tn == 256 && st == 10) return DISPATCH_SM100_MX(128, 256, 2, 2, 128, false);
        if (tm == 256 && tn == 128 && st == 11) return DISPATCH_SM100_MX(256, 128, 2, 2, 256, false);
        // 2SM rows (TileM=256)
        if (tm == 256 && tn == 128 && st == 2) return DISPATCH_SM100_MX(256, 128, 2, 1, 128, false);
        if (tm == 256 && tn == 256 && st == 2) return DISPATCH_SM100_MX_NOSMEM(256, 256, 2, 1, 128, false);
        if (tm == 256 && tn == 128 && st == 4) return DISPATCH_SM100_MX(256, 128, 2, 1, 256, false);
        if (tm == 256 && tn == 256 && st == 4) return DISPATCH_SM100_MX(256, 256, 2, 1, 256, false);
        if (tm == 256 && tn == 256 && st == 5) return DISPATCH_SM100_MX(256, 256, 2, 2, 128, false);
        if (tm == 256 && tn == 256 && st == 6) return DISPATCH_SM100_MX(256, 256, 2, 2, 256, false);
        // Unknown forced combo → fall through to the cascade (sm_120 contract).
    }

    // Cascade v2 — rules distilled from the 2026-07-05 b300 round-2 tile
    // sweep (runs/b300_tile_sweep_v2.jsonl: 12 variants × Qwen3 grid,
    // best-shot cudagraph µs). Key findings baked in:
    //   * (256,128) 2SM c(2,1) K256 is the mid-band workhorse
    //     (128 < M ≤ 2048: best-or-tie on wqkv/wo/gate/down).
    //   * StreamK in heuristic mode LOSES to plain tiles everywhere
    //     (down M=1: 30.9 vs 17.5 µs) — removed from the default route
    //     until explicit-splits (FSO_FORCE_KSPLIT) proves out.
    //   * c(2,2) and K256 do NOT help the peak band (cubic-8192:
    //     c22 410 µs / K256 435 µs vs c21-K128 370 µs).
    //   * down-class decode floor is (256,128)K256 ≈ 16.7 µs vs cuBLAS
    //     ~10.4 µs — the remaining gap needs split-K or a custom path,
    //     tracked separately.
    int const tiles_n = (N + 127) / 128;

    // Parallel split-K for down-class decode (ncu F1): narrow-N long-K
    // shapes whose tile grid can't fill the machine. pick_splits returns 1
    // (→ fall through) once the grid fills, so the route self-limits to
    // decode. Band from the 2026-07-06 sk_probe (best-shot µs):
    //   * M ≤ 32, K ≥ 4096: always wins — down 17.2→11.7 (BEATS cuBLAS
    //     12.3), wo 10.6→9.9. wqkv/gate K=2560 excluded (forced-SK loses,
    //     11.4 vs 10.1: split ×2 + reduce isn't amortized at short K).
    //   * 32 < M ≤ 128: only long-K — down K=9728 wins (16.4→12.9 M=64,
    //     15.7 M=128) but wo K=4096 loses (10.3→13.9) → K ≥ 8192.
    //   * Deeper splits lose (down M=1: S=19 14.9 vs S=4 12.9) — the ≤ 8
    //     cap plus largest-divisor pick in pick_splits is the sweet spot.
    if (tiles_n <= 32 && (K % 128) == 0
        && ((M <= 32 && K >= 4096) || (M <= 128 && K >= 8192)))
    {
        int const tiles = ((M + 127) / 128) * tiles_n;
        int const ks = detail::pick_splits(tiles, K / 128);
        if (ks > 1)
            return launch_sm100_mxfp8_gemm_splitk<128, 128>(
                mat_a, mat_b, mat_d, scales_a, scales_b, M, N, K, ks, stream);
    }

    // --- N-tile chosen by wave arithmetic (2026-09-17) -----------------
    // detail::pick_wave_tile carries the rule and its measured bounds; it
    // returns 0 for every cell this round did not measure a win on, and those
    // fall through to the rules distilled in b300_mlp_tune_20260915 unchanged.
    // The parallel split-K branch above keeps precedence, so the narrow-N
    // long-K decode cells it owns are untouched.
    {
        int const wave_tile = detail::pick_wave_tile(M, N);
        switch (wave_tile)
        {
        case 128 * 1000 + 64:  return DISPATCH_SM100_MX(128, 64, 1, 1, 128, false);
        case 128 * 1000 + 192: return DISPATCH_SM100_MX(128, 192, 1, 1, 128, false);
        case 256 * 1000 + 64:  return DISPATCH_SM100_MX(256, 64, 2, 1, 128, false);
        case 256 * 1000 + 192: return DISPATCH_SM100_MX(256, 192, 2, 1, 128, false);
        default: break;
        }
    }

    if (M <= 8)
    {
        // Extreme decode: 1SM (128,128); very wide N amortizes the per-CTA
        // prologue with a 256-wide tile (gate_up 15.7 vs 19.9, gate 12.5
        // vs 13.5 µs). NOSMEM: latency-bound, L2 SOL ≈ 12% — direct store
        // wins (wqkv M1 10.75→10.50, down M1 17.60→17.15).
        if (N >= 9728)
            return DISPATCH_SM100_MX_NOSMEM(128, 256, 1, 1, 128, false);
        return DISPATCH_SM100_MX_NOSMEM(128, 128, 1, 1, 128, false);
    }
    if (M <= 128)
    {
        // NOSMEM only while the M-tile is mostly empty (M ≤ 32: stores are
        // a sliver, latency wins — gate M1-32 −9..−16%). At fuller M-tiles
        // with this band's short-K shapes the epilogue dominates and the
        // TMA bulk store wins big (v23 bench: gate M=128 8.2→10.3 µs,
        // gate_up M=128 12.3→14.4 µs under NOSMEM → reverted to TMA).
        bool const nosmem = (M <= 32);
        // gate_up-class stays 1SM 256-wide through M=128 (13.3 vs 15.6+).
        if (N >= 16384)
            return nosmem ? DISPATCH_SM100_MX_NOSMEM(128, 256, 1, 1, 128, false)
                          : DISPATCH_SM100_MX(128, 256, 1, 1, 128, false);
        // gate M ≤ 16 also prefers the 256-wide 1SM tile (12.4 vs 12.7).
        if (M <= 16 && N >= 9728)
            return DISPATCH_SM100_MX_NOSMEM(128, 256, 1, 1, 128, false);
        // One 128-row M-tile: 2SM (256,128)K256 wins when the wave still
        // fits (wqkv tn=48: 9.3, down tn=20: 16.7); wide N overflows with
        // 2× CTAs and loses (gate tn=76: 12.1 vs 9.4 1SM) → stay 1SM.
        // M ≤ 32 is still latency-dominated and prefers 1SM (wqkv M=32:
        // 8.3 vs 10.3 µs under the M>16 2SM rule — bench 2026-07-05).
        if (M > 32 && tiles_n <= 60)
            return DISPATCH_SM100_MX(256, 128, 2, 1, 256, false);
        return nosmem ? DISPATCH_SM100_MX_NOSMEM(128, 128, 1, 1, 128, false)
                      : DISPATCH_SM100_MX(128, 128, 1, 1, 128, false);
    }
    if (M <= 2048)
    {
        // Narrow N at M≈1024 halves the wave count with a 256-wide tile
        // (down/wo M=1024) — long-K only (cubic-1024/1536 regressed) and
        // not at M=2048 (wo regressed 18%).
        if (tiles_n <= 20 && M >= 1024 && M < 2048 && K >= 4096)
            return DISPATCH_SM100_MX(256, 256, 2, 1, 128, false);
        // Wide-N many-wave cells also benefit from raster swizzle
        // (gate_up M=2048 K256: sw8 75.3 vs sw0 79.3 µs p50, warm probe
        // 2026-07-05); narrow-N mid cells are flat → gate on tiles_n.
        int const mid_swizzle = (M >= 1024 && tiles_n >= 32) ? 8 : 0;
        return DISPATCH_SM100_MX_SW(256, 128, 2, 1, 256, false, mid_swizzle);
    }
    // Peak band: 256-wide once N supplies enough tiles; K128 (K256 loses
    // ~6% at cubic-4096+, deeper pipeline wins over fewer iterations).
    // Raster swizzle 8 recovers L2 locality on many-wave problems
    // (cubic-8192: 370 → 349 µs +6%, cubic-12288/16384 +11/16%) — but
    // ONLY when N supplies at least a full swizzle group of tiles
    // (down M=4096, tiles_n=20: swizzle=8 cost 24%).
    // NOSMEM on the peak tile only when the mainloop is long enough to
    // amortize the slower scattered-STG store (K ≥ 4096: cubic-8192
    // 352→327 µs = 3360 TF ≈ cuBLAS parity, cubic-4096 −3.8%, L2 SOL 49%
    // has headroom). Short-K epilogue-heavy peak cells keep the TMA bulk
    // store (v23 bench: gate M=4096 K=2560 74.3→85.0 µs under NOSMEM).
    // Narrow-N (256,128) stays TMA (unmeasured).
    int const peak_swizzle = (tiles_n >= 32) ? 8 : 0;
    if (N >= 4096)
    {
        if (K >= 4096)
            return DISPATCH_SM100_MX_NOSMEM_SW(256, 256, 2, 1, 128, false, peak_swizzle);
        return DISPATCH_SM100_MX_SW(256, 256, 2, 1, 128, false, peak_swizzle);
    }
    return DISPATCH_SM100_MX_SW(256, 128, 2, 1, 128, false, peak_swizzle);

#undef DISPATCH_SM100_MX
#undef DISPATCH_SM100_MX_SW
}

} // namespace sm100_blockscaled_gemm

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
