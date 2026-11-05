/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Stream-K (concurrent K-split) wrapper for the SM120 BlockScaled fp8 GEMM.
 *
 * Targets shapes where tiles_M * tiles_N severely under-fills the GPU
 * (e.g. 170-SM Blackwell: tiles_n=20 -> only 12% of 170 SMs busy). Splits the K
 * dimension into k_split chunks; each chunk runs on its own CUDA stream
 * with gridDim = sm_count / k_split, so the chunks co-schedule onto disjoint
 * SM subsets. A reduction kernel then sums the per-chunk BF16 partials in
 * FP32 and casts to BF16.
 *
 * Empirically validated on 170-SM Blackwell, M in [1, 128]:
 *   - tiles_n=20 (N=2560), K=4096:  K_split=2 -> +34% vs K=1
 *   - tiles_n=20 (N=2560), K=9728:  K_split=4 -> +114% vs K=1
 *   - tiles_n=48 (N=6144):           marginal (+0..16%)
 *   - tiles_n>=96 (N>=12288):        K_split=1 wins (no under-fill to fix)
 *
 * Sweet spot: k_split ~= max(1, K / 2048), bounded by tile_n underfill.
 */

#pragma once

#include "blockscale_gemm/common/epilogue.cuh"
#include "tensorrt_llm/common/assert.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdlib>

namespace tensorrt_llm::kernels::blockscale_gemm
{

// Forward decl of the kernel-level launcher (defined in sm120_dispatch.cuh).
// The 4th param (MinBlocksPerSm) gets its default (=1) from the definition;
// repeating it here would be a C++ "redefinition of default argument" error.
template <int TileM, int TileN, int Stages, int MinBlocksPerSm>
void launch_sm120_gemm_kernel(__nv_fp8_e4m3* mat_a, int64_t ld_a, int64_t stride_a, __nv_fp8_e4m3* mat_b, int64_t ld_b,
    int64_t stride_b, __nv_bfloat16* mat_d, int64_t ld_d, int64_t stride_d, float* scales_a, int64_t stride_scales_a,
    float* scales_b, int64_t stride_scales_b, uint32_t num_problems, uint32_t shape_m, uint32_t shape_n,
    uint32_t shape_k, cudaStream_t stream, int num_device_sms);

namespace detail
{

inline int sk_ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

inline int sk_round_up(int a, int b)
{
    return sk_ceil_div(a, b) * b;
}

// Persistent pool of side streams + events (process-wide). Cap at 8 side streams (matches max k_split).
//
// The bf16 partial scratch is NOT here any more (issue #12, charter 7.5: no workspace shared by two streams). It
// used to be one process-wide buffer, allocated at the first Stream-K call and shared by every caller stream, so two
// Stream-K calls on two streams (or two graphs replayed concurrently) wrote the same partials. Every caller now
// passes its own scratch, allocated by the op body from the torch caching allocator on the caller's current stream
// (graph-private inside a capture): Sm120DenseRoute::streamk_scratch_bytes gives its size. The side streams stay
// shared: inside a graph they become nodes, and outside one each slice is ordered after the caller's prior work by
// the fork event below.
//
// CUDA-graph compatibility:
//   1. Side streams + events are created lazily — the host-side `cudaStreamCreate` / `cudaEventCreateWithFlags`
//      calls happen at the first Stream-K launch, which must be OUTSIDE any capture (one eager call first).
//   2. Fork-join: every launch records `fork_event` on the caller's stream and has each side stream wait on it
//      before its slice, then joins the side streams back with `side_done`. Capture sees the cross-stream
//      dependencies and stitches them into the graph.
struct StreamKPool
{
    static constexpr int kMaxKSplit = 8;
    cudaStream_t side_streams[kMaxKSplit - 1];
    cudaEvent_t side_done[kMaxKSplit - 1];
    cudaEvent_t fork_event = nullptr;  // recorded on main, waited on side
    bool streams_initialized = false;

    static StreamKPool& instance()
    {
        static StreamKPool pool;
        return pool;
    }

    void ensure_streams_initialized()
    {
        if (streams_initialized)
            return;
        for (int i = 0; i < kMaxKSplit - 1; ++i)
        {
            cudaStreamCreate(&side_streams[i]);
            cudaEventCreateWithFlags(&side_done[i], cudaEventDisableTiming);
        }
        cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming);
        streams_initialized = true;
    }
};

} // namespace detail

// Reduction kernel: accumulate K_split BF16 partials per (m, n) in FP32, cast
// back to BF16. partial_stride is the per-slice stride in elements.
__global__ inline void sm120_streamk_reduce_kernel(__nv_bfloat16 const* __restrict__ partials,
    __nv_bfloat16* __restrict__ out, int M, int N, int K_split, std::size_t partial_stride)
{
    // Use size_t for tid/total: large M*N (~>INT_MAX) overflows int.
    std::size_t const tid
        = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t const total = static_cast<std::size_t>(M) * static_cast<std::size_t>(N);
    if (tid >= total)
        return;
    float acc = 0.f;
    for (int s = 0; s < K_split; ++s)
    {
        acc += __bfloat162float(partials[static_cast<std::size_t>(s) * partial_stride + tid]);
    }
    out[tid] = __float2bfloat16_rn(acc);
}

// The epilogue twin of the reduce (interface v1, common/epilogue.cuh): the same fp32 sum of the K_split partials, in
// the same order, then v += bias_term(bias[n]); v = act(v); [v += residual[m, n]] and the one rounding. The slices
// keep the identity epilogue, so every term is applied once, after the reduction. Launched only for a non-identity
// epilogue; the identity reduce above is unchanged. Launched without PDL after the join, so bias and residual (the
// caller's tensors, possibly produced by the kernel before the GEMM) are complete when it runs.
// A template only so that no translation unit carries its code unless it launches it.
template <int kUnused = 0>
__global__ void sm120_streamk_reduce_epi_kernel(__nv_bfloat16 const* __restrict__ partials,
    __nv_bfloat16* __restrict__ out, int M, int N, int K_split, std::size_t partial_stride,
    __nv_bfloat16 const* __restrict__ bias, __nv_bfloat16 const* __restrict__ residual, int64_t ld_residual, int act)
{
    std::size_t const tid
        = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t const total = static_cast<std::size_t>(M) * static_cast<std::size_t>(N);
    if (tid >= total)
        return;
    float acc = 0.f;
    for (int s = 0; s < K_split; ++s)
    {
        acc += __bfloat162float(partials[static_cast<std::size_t>(s) * partial_stride + tid]);
    }
    std::size_t const m = tid / static_cast<std::size_t>(N);
    std::size_t const n = tid - m * static_cast<std::size_t>(N);
    bool const has_bias = bias != nullptr;
    bool const has_residual = residual != nullptr;
    __nv_bfloat16 const b = has_bias ? bias[n] : __ushort_as_bfloat16(0);
    __nv_bfloat16 const r = has_residual ? residual[m * static_cast<std::size_t>(ld_residual) + n] : __ushort_as_bfloat16(0);
    out[tid] = __float2bfloat16_rn(fso::dense_epilogue_value_rt(acc, has_bias, b, act, has_residual, r));
}

// The route of one dense sm_120 GEMM call, as its cascade decides it (sm120_mxfp8_dense_route in
// mxfp8/dispatch.cuh, sm120_fp8_dense_route in fp8/dispatch.cuh): the kernel instance and the K split. The
// dispatchers launch exactly this, the op bodies size the Stream-K scratch from it, and the host-side route queries
// report it.
struct Sm120DenseRoute
{
    int tile_m = 0;
    int tile_n = 0;
    int stages = 0;
    int min_blocks = 1;   // MinBlocksPerSm; 2 only under the FSO_FORCE_MIN_BLOCKS=2 diagnostic
    int sched_group = 16; // persistent-scheduler group; 8 / 32 only under FSO_FORCE_SCHED_GROUP
    int k_split = 1;      // > 1: Stream-K over this many K slices (the launcher's clamps applied); 1: single launch
    bool smallm = false;  // the FSO_FORCE_SMALLM diagnostic kernel (MXFP8 only)

    // An instance that exists only behind a diagnostic knob (no dense-epilogue instantiation).
    bool forced_only_instance() const
    {
        return smallm || min_blocks != 1 || sched_group != 16;
    }

    // Bytes of the bf16 partial scratch the route needs: k_split slices of [M, N], or 0 for a single launch.
    std::size_t streamk_scratch_bytes(int M, int N) const
    {
        return k_split > 1 ? static_cast<std::size_t>(k_split) * static_cast<std::size_t>(M)
                * static_cast<std::size_t>(N) * sizeof(__nv_bfloat16)
                           : 0;
    }
};

// The K split a Stream-K launcher runs for a requested `k_split`: at most one slice per `sf_block`-element K chunk
// (one int32 scale word: 128 for MXFP8, 512 for block-FP8 1x128), and at most StreamKPool::kMaxKSplit slices; 1 means
// the launcher falls back to a single launch of the same tile. The same clamps the launchers have always applied, in
// the same order.
inline int sm120_streamk_effective_split(int k_split, int K, int sf_block)
{
    int const k_chunks_total = detail::sk_ceil_div(K, sf_block);
    if (k_split > k_chunks_total)
        k_split = k_chunks_total;
    if (k_split <= 1)
        return 1;
    if (k_split > detail::StreamKPool::kMaxKSplit)
        k_split = detail::StreamKPool::kMaxKSplit;
    return k_split;
}

// The caller-provided scratch of one Stream-K call, checked against the route's need.
inline __nv_bfloat16* sm120_streamk_scratch(void* scratch, std::size_t scratch_bytes, std::size_t needed)
{
    TLLM_CHECK_WITH_INFO(scratch != nullptr && scratch_bytes >= needed,
        "sm_120 Stream-K: the caller's partial scratch holds %zu bytes at %p, the route needs %zu (the op body "
        "allocates it from the route's streamk_scratch_bytes)",
        scratch_bytes, scratch, needed);
    return reinterpret_cast<__nv_bfloat16*>(scratch);
}

// Concurrent-stream Stream-K launcher (block-FP8 1x128). `scratch` / `scratch_bytes` is the caller's bf16 partial
// scratch (Sm120DenseRoute::streamk_scratch_bytes of the route: k_split slices of [M, N]), allocated by the op body
// from the torch caching allocator on `stream`.
template <int TileM, int TileN, int Stages>
void launch_sm120_streamk_gemm(__nv_fp8_e4m3* mat_a, int ld_a, __nv_fp8_e4m3* mat_b, int ld_b, __nv_bfloat16* mat_d,
    int ld_d, int M, int N, int K, float* scales_a, float* scales_b, int k_split, int sm_count, cudaStream_t stream,
    void* scratch, std::size_t scratch_bytes)
{
    constexpr int kSfBlock = 512; // SF granularity per int32 word

    int const k_chunks_total = detail::sk_ceil_div(K, kSfBlock);
    k_split = sm120_streamk_effective_split(k_split, K, kSfBlock);
    if (k_split <= 1)
    {
        // Single launch using the full GPU.
        launch_sm120_gemm_kernel<TileM, TileN, Stages>(mat_a, ld_a, 0, mat_b, ld_b, 0, mat_d, ld_d, 0, scales_a, 0,
            scales_b, 0, /*num_problems=*/1u, static_cast<uint32_t>(M), static_cast<uint32_t>(N),
            static_cast<uint32_t>(K), stream, sm_count);
        return;
    }

    auto& pool = detail::StreamKPool::instance();
    pool.ensure_streams_initialized();
    std::size_t const needed = static_cast<std::size_t>(k_split) * M * N * sizeof(__nv_bfloat16);
    __nv_bfloat16* d_partial = sm120_streamk_scratch(scratch, scratch_bytes, needed);

    int const sfa_pad_m = detail::sk_round_up(M, 4);
    int const sfb_pad_n = detail::sk_round_up(N, 4);
    int const k_chunks_per_split = detail::sk_ceil_div(k_chunks_total, k_split);
    int const sm_per_slice = std::max(1, sm_count / k_split);
    std::size_t const partial_stride = static_cast<std::size_t>(M) * N;

    // Fork point. Every side-stream slice waits on an event recorded on the caller's stream, eager or capturing.
    // Under capture this is what stitches the side streams into the graph. Outside capture it orders each slice
    // after the caller's prior work: the slices read A / SFA, which the caller's stream may still be producing, and
    // write the caller's scratch, which the caching allocator may just have handed back from a tensor an earlier
    // kernel of that stream still reads. (Until issue #12 the fork was skipped outside capture to save host time,
    // which raced on any non-default caller stream: side streams only synchronise implicitly with the legacy
    // default stream.)
    cudaEventRecord(pool.fork_event, stream);

    int actual_splits = 0;
    for (int s = 0; s < k_split; ++s)
    {
        int const k_chunk_start = s * k_chunks_per_split;
        if (k_chunk_start >= k_chunks_total)
            break;
        int const k_chunk_end = std::min(k_chunk_start + k_chunks_per_split, k_chunks_total);
        int const k_off = k_chunk_start * kSfBlock;
        int const k_size = std::min((k_chunk_end - k_chunk_start) * kSfBlock, K - k_off);
        if (k_size <= 0)
            break;

        __nv_fp8_e4m3* a_off = mat_a + k_off;
        __nv_fp8_e4m3* b_off = mat_b + k_off;
        // SFA[m, k_chunk] col-major stride pad(M,4) -> int32 offset.
        float* sa_off = scales_a + static_cast<std::size_t>(k_chunk_start) * sfa_pad_m;
        float* sb_off = scales_b + static_cast<std::size_t>(k_chunk_start) * sfb_pad_n;
        __nv_bfloat16* part_d = d_partial + static_cast<std::size_t>(actual_splits) * partial_stride;

        cudaStream_t this_stream = (s == 0) ? stream : pool.side_streams[s - 1];
        if (s != 0)
            cudaStreamWaitEvent(this_stream, pool.fork_event, 0);

        launch_sm120_gemm_kernel<TileM, TileN, Stages>(a_off, ld_a, 0, b_off, ld_b, 0, part_d, /*ld_d=*/N, 0, sa_off, 0,
            sb_off, 0, /*num_problems=*/1u, static_cast<uint32_t>(M), static_cast<uint32_t>(N),
            static_cast<uint32_t>(k_size), this_stream, sm_per_slice);
        ++actual_splits;
    }

    // Fence side streams back into main stream.
    for (int s = 1; s < actual_splits; ++s)
    {
        cudaEventRecord(pool.side_done[s - 1], pool.side_streams[s - 1]);
        cudaStreamWaitEvent(stream, pool.side_done[s - 1], 0);
    }

    // Reduce on the main stream.
    // M*N must be computed in 64-bit: int overflows for M=N>~46340 (sqrt(INT_MAX)).
    std::size_t const total = static_cast<std::size_t>(M) * static_cast<std::size_t>(N);
    int const reduce_threads = 256;
    int const reduce_blocks = static_cast<int>((total + reduce_threads - 1) / reduce_threads);
    sm120_streamk_reduce_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
        d_partial, mat_d, M, N, actual_splits, partial_stride);
}

// Heuristic: pick k_split given problem shape and SM count.
//   tiles_n >= 96       -> never split (already saturating SMs)
//   tiles_n in [32, 96) -> K-dependent split for small M
//   tiles_n < 32        -> split proportional to K, capped at 4
//
// P2 update (2026-05-05): the previous "K/2048 capped at 8" heuristic was
// over-aggressive on long K. Empirical sweep on 170-SM Blackwell (170 SMs) shows
// k_split=4 is the global optimum across K ∈ {8192, 12288, 16384} for both
// (M=128, tiles_n<32) and (M=32, tiles_n<32) — k_split=8 over-reduces the
// per-stream K work relative to the partial-D BF16 reduce overhead. The
// previous cap of 8 cost up to 80 % on M=32 N=2560 K=16384 (was 92 TF, now
// 165 TF). For tiles_n in [32, 96), M ≤ 32 + K=16384 wants k_split=4 too
// (165 → 230 TF, +40 %).
//
// Source data: `docs/skills/blockscale-gemm-tuning/references/baselines/p2_kSplit_sweep.txt`.
inline int sm120_streamk_choose_k_split(int M, int N, int K, int sm_count)
{
    // Diagnostic: FSO_DISABLE_STREAMK=1 forces single-launch everywhere
    // (used to measure CUTLASS BlockScaled latency without Stream-K). Read
    // once and cached in a function-local static.
    static int disable_streamk = -1;
    if (disable_streamk == -1)
    {
        char const* s = std::getenv("FSO_DISABLE_STREAMK");
        disable_streamk = (s && std::atoi(s) == 1) ? 1 : 0;
    }
    if (disable_streamk == 1)
        return 1;

    int const tiles_n = detail::sk_ceil_div(N, 128);
    if (tiles_n >= 96)
        return 1;
    if (tiles_n >= 32)
    {
        if (M > 32)
            return 1;
        // Mid-N + small-M: scale split with K. K=16384 wants k=4.
        return (K <= 8192) ? 2 : 4;
    }
    // Narrow N: split proportional to K, capped at 4.
    //
    // 2026-05-08 K lower-bound: empirical 170-SM Blackwell sweep at M ∈ {32, 128},
    // N ∈ {2560, 3072} exposed a blind spot where short-K shapes lose to
    // single-launch even though the heuristic still picks k_split>=2:
    //   M=128 N=2560 K=4096 split=2 (slice=2048): 126 TF vs 145 OFF (-13%)
    //   M=128 N=2560 K=6144 split=3 (slice=2048): 121 TF vs 154 OFF (-21%)
    //   M=128 N=2560 K=8192 split=4 (slice=2048): 54  TF vs 162 OFF (-67%)
    // Yet K=9728 split=4 (slice=2432) wins at 207 TF (+27%), and K>=12288
    // wins by 40-80%. The cliff sits at K=8192 specifically — a slice_K
    // floor doesn't capture it cleanly because slice_K=2432@split=4 (K=9728)
    // wins while slice_K=2730@split=3 (K=8192 forced down) doesn't. Gate
    // on the measured K threshold instead.
    constexpr int min_k_for_split = 9728;
    if (K < min_k_for_split)
        return 1;
    // E6 (2026-05-10): M-saturation gate. Stream-K is a remedy for SM
    // under-fill caused by `tiles_m * tiles_n < sm_count`. When M is large
    // enough that (TileM=128) M-tiles × tiles_n already saturate the GPU,
    // splitting K just adds reduce-kernel overhead for no parallelism win.
    //   M=2048 N=2560 K=9728 (tn=20): w/o gate streamk(k=4) ~539 TF vs (128)
    //                                   single-launch direct 691 TF (+28%)
    //   M=4096 N=2560 K=9728 (tn=20): w/o gate streamk(k=4) ~526 TF vs (128)
    //                                   single-launch direct 683 TF (+30%)
    // E32 (2026-05-28): lower threshold from 100% to 90% saturation. The
    // hard `>= sm_count` gate let qwen-down M=1024 N=2560 K=9728 (tiles_m=8,
    // tn=20 → 160 CTAs, 94% of 170 SMs) fire Stream-K k=4 even though
    // single-launch wins by +19% (564→672 TF). Above ~90% the K-split
    // reduce overhead dominates the marginal parallelism gain.
    int const tiles_m_at_128 = detail::sk_ceil_div(M, 128);
    if (tiles_m_at_128 * tiles_n * 10 >= sm_count * 9)
        return 1;
    int desired = std::max(1, K / 2048);
    int max_useful = std::max(1, sm_count / std::max(1, tiles_n));
    desired = std::min(desired, max_useful);
    desired = std::min(desired, 4);
    return desired;
}

} // namespace tensorrt_llm::kernels::blockscale_gemm
