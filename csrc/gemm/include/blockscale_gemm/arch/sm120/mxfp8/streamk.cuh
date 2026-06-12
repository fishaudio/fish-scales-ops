/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Stream-K (concurrent K-split) wrapper for the SM120 MXFP8 (1×32) GEMM.
// Mirrors `sm120_streamk.cuh::launch_sm120_streamk_gemm` but routes
// through `launch_sm120_mxfp8_gemm_kernel` and takes int32-packed UE8M0
// scales. Reuses the same `detail::StreamKPool` (BF16 partial scratch
// + side streams + fork_event) and the same
// `sm120_streamk_choose_k_split` heuristic — the only VS-specific bit is
// `kSfBlock = 4 * kSFVecSize = 128` (the K-extent of one int32 SF word).

#pragma once

#include "blockscale_gemm/arch/sm120/mxfp8/dispatch.cuh"  // launch_sm120_mxfp8_gemm_kernel
#include "blockscale_gemm/arch/sm120/common/streamk.cuh"         // detail::StreamKPool, sm120_streamk_choose_k_split, sk_ceil_div, sk_round_up, sm120_streamk_reduce_kernel

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN
namespace kernels::blockscale_gemm
{

template <int TileM, int TileN, int Stages>
void launch_sm120_mxfp8_streamk_gemm(__nv_fp8_e4m3* mat_a, int ld_a, __nv_fp8_e4m3* mat_b, int ld_b,
    __nv_bfloat16* mat_d, int ld_d, int M, int N, int K, int32_t* scales_a, int32_t* scales_b, int k_split,
    int sm_count, cudaStream_t stream)
{
    // VS=32: each int32 SF word covers 4 UE8M0 bytes × 32 K-elements = 128 K.
    // VS=128 path uses kSfBlock=512 (4 × 128). The K-chunk arithmetic is
    // identical otherwise.
    constexpr int kSfBlock = 128;

    int const k_chunks_total = detail::sk_ceil_div(K, kSfBlock);
    if (k_split > k_chunks_total)
        k_split = k_chunks_total;
    if (k_split <= 1)
    {
        launch_sm120_mxfp8_gemm_kernel<TileM, TileN, Stages>(mat_a, ld_a, 0, mat_b, ld_b, 0, mat_d, ld_d, 0,
            scales_a, 0, scales_b, 0, /*num_problems=*/1u, static_cast<uint32_t>(M), static_cast<uint32_t>(N),
            static_cast<uint32_t>(K), stream, sm_count);
        return;
    }

    auto& pool = detail::StreamKPool::instance();
    pool.ensure_streams_initialized();
    if (k_split > detail::StreamKPool::kMaxKSplit)
        k_split = detail::StreamKPool::kMaxKSplit;
    std::size_t const needed = static_cast<std::size_t>(k_split) * M * N * sizeof(__nv_bfloat16);
    __nv_bfloat16* d_partial = pool.ensure_partial_capacity(needed);

    int const sfa_pad_m = detail::sk_round_up(M, 4);
    int const sfb_pad_n = detail::sk_round_up(N, 4);
    int const k_chunks_per_split = detail::sk_ceil_div(k_chunks_total, k_split);
    int const sm_per_slice = std::max(1, sm_count / k_split);
    std::size_t const partial_stride = static_cast<std::size_t>(M) * N;

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &capture_status);
    bool const is_capturing = (capture_status == cudaStreamCaptureStatusActive);
    if (is_capturing)
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
        // SFA[m, k_chunk] col-major stride pad(M,4) -> int32 offset (1 word
        // per kSfBlock=128 K-elements per row in the VS=32 layout).
        int32_t* sa_off = scales_a + static_cast<std::size_t>(k_chunk_start) * sfa_pad_m;
        int32_t* sb_off = scales_b + static_cast<std::size_t>(k_chunk_start) * sfb_pad_n;
        __nv_bfloat16* part_d = d_partial + static_cast<std::size_t>(actual_splits) * partial_stride;

        cudaStream_t this_stream = (s == 0) ? stream : pool.side_streams[s - 1];
        if (s != 0 && is_capturing)
            cudaStreamWaitEvent(this_stream, pool.fork_event, 0);

        launch_sm120_mxfp8_gemm_kernel<TileM, TileN, Stages>(a_off, ld_a, 0, b_off, ld_b, 0, part_d, /*ld_d=*/N, 0,
            sa_off, 0, sb_off, 0, /*num_problems=*/1u, static_cast<uint32_t>(M), static_cast<uint32_t>(N),
            static_cast<uint32_t>(k_size), this_stream, sm_per_slice);
        ++actual_splits;
    }

    for (int s = 1; s < actual_splits; ++s)
    {
        cudaEventRecord(pool.side_done[s - 1], pool.side_streams[s - 1]);
        cudaStreamWaitEvent(stream, pool.side_done[s - 1], 0);
    }

    // M*N must be computed in 64-bit: int overflows for M=N>~46340 (sqrt(INT_MAX)).
    std::size_t const total = static_cast<std::size_t>(M) * static_cast<std::size_t>(N);
    int const reduce_threads = 256;
    int const reduce_blocks = static_cast<int>((total + reduce_threads - 1) / reduce_threads);
    sm120_streamk_reduce_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
        d_partial, mat_d, M, N, actual_splits, partial_stride);
}

} // namespace kernels::blockscale_gemm
TRTLLM_NAMESPACE_END
