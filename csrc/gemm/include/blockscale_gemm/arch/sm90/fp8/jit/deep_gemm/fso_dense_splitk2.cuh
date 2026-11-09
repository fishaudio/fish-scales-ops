/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 fish-scales-ops contributors
 * SPDX-License-Identifier: Apache-2.0
 */

// Dense block-FP8 GEMM above the decode band on sm_90 (M up to 128), swap-AB orientation, with K split across the CTAs
// of a thread-block cluster and two math warp-groups per CTA (fp8_gemm_kernel_swapAB_splitk2): the two-warp-group
// split-K route of gemm_dispatch_sm90 (dispatch.cuh). deep_gemm::jit::get_dense_splitk2_config (jit_utils.cuh) says
// where the dispatcher takes it.
//
// Why a second split-K kernel. Above the decode band the other sm_90 routes give every output element to one CTA that
// streams the whole of K for its weight tile, and they tile M into two to five token tiles or M blocks, each of which
// streams the same weight tile again with the same full-K chain: on a narrow weight (N = 1280 or 2560) only 20 to 54
// independent K chains run, each at about 130-155 ns per 128-wide K block (issue #12, FH7 step 1). FH5's split-K kernel
// (fso_dense_splitk.cuh) splits K but carries the whole token tile on one warp-group, and at a 64-token tile that
// warp-group's chain (four m64n64k32 WGMMAs, the wait for the previous block, the release and the promotion of 32
// accumulators per thread) sets 229 ns per block with the producer waiting on free stages. This kernel shares each CTA's
// tile between two math warp-groups:
//   * token halves (kWeightHalves = false): one 64-row weight tile; warp-group g multiplies it by tokens
//     [g * BLOCK_T / 2, (g + 1) * BLOCK_T / 2), BLOCK_T a multiple of 16 from 32 to 128 (at most n64 per warp-group);
//   * weight halves (kWeightHalves = true): one 128-row weight tile; warp-group g multiplies rows [64 g, 64 g + 64) by
//     every token, BLOCK_T a multiple of 8 from 16 to 128 (twice the CTAs' weight bytes per K block, so that a wide
//     weight reaches enough clusters of S CTAs that are resident at once).
// Everything else is FH5's kernel: grid x = tiles * kNumSplits, the kNumSplits CTAs of one weight tile form one cluster
// and CTA r takes K blocks [r * kb / S + min(r, kb % S), ...); one token tile covers all of M; the stage ring runs once
// through the slice.
//
// Programmatic dependent launch, arithmetic, dense epilogue, reduction, cluster barrier and pipeline stages: as in
// fso_dense_splitk.cuh, with 8 math warps instead of 4. Every thread executes griddepcontrol.wait before its first
// global read and the kernel does not trigger its dependents early. Each math warp promotes every 128-wide K block in K
// order with the scale product weight scale first (accum, then final += (sw * sx) * accum), so with kNumSplits = 1 every
// output element is the bf16 rounding of the fp32 value fp8_gemm_kernel_swapAB computes; with S > 1 the cluster adds
// the S partials of every element in rank order (p_0 + p_1 + ... + p_{S-1}) and rounds once. The slab of math warp w
// (16 weight rows by its warp-group's tokens) is owned by cluster rank w % S, so S can be 1 to 8; the owner warp applies
// bias, erf-GELU and residual to the reduced sum, rounds to bf16 and stores its slab with one TMA store. Warp-groups
// whose WGMMA is at most 64 wide keep two accumulator sets (the WGMMAs of K block kb + 1 run while kb is promoted); the
// 128-wide one of the weight-halves layout keeps one set, for registers.
//
// Three differences from fso_dense_splitk.cuh, each measured on the H200 (issue #12 FH7, runs/FH7/data/diag2-13):
//   * each K block's scales are loaded into registers before the wait for that block's WGMMAs (loaded after it, their
//     shared-memory latency sat in the chain of every block: 1.1 us per call on 1280 x 7680 at M = 78), and drained
//     right before the stage is released (the stage-release audit);
//   * the mailbox fragments are vector-major: vector i of lane l at (32 i + l) * 16 bytes of the slot, so that the 32
//     lanes of one vector write and read 512 consecutive bytes (lane-major fragments are 128 or 256 bytes apart per lane
//     at 64 or 128 tokens per warp-group, every lane in the same banks);
//   * the owner adds the rank-order sum without choosing per element between its registers and a slot: it starts from
//     -0.0f (-0 + x == x for every x, signed zeros included), adds the lower ranks' slots, its own partial, then the
//     higher ranks' slots. The bytes are those of p_0 + p_1 + ... + p_{S-1}.
//
// Shared memory: fso_dense_splitk2_layout.cuh. The host computes the same sizes from the same functions.
#pragma once

#include "fso_dense_epilogue.cuh"
#include "fso_dense_splitk2_layout.cuh"
#include "mma_utils.cuh"
#include "tma_utils.cuh"
#include "utils.cuh"

namespace deep_gemm
{

namespace fso_dense_splitk2
{
__device__ __forceinline__ uint32_t cta_rank_in_cluster()
{
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

// The shared::cluster address of `local_smem_addr` (a shared::cta address) in the CTA of rank `rank`.
__device__ __forceinline__ uint32_t map_to_rank(uint32_t local_smem_addr, uint32_t rank)
{
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(local_smem_addr), "r"(rank));
    return r;
}

// 16 bytes into a peer CTA's shared memory, completing 16 transaction bytes on the peer's mbarrier.
__device__ __forceinline__ void st_async_v4(uint32_t remote_addr, float a, float b, float c, float d,
    uint32_t remote_mbar)
{
    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(
            remote_addr),
        "r"(__float_as_uint(a)), "r"(__float_as_uint(b)), "r"(__float_as_uint(c)), "r"(__float_as_uint(d)),
        "r"(remote_mbar)
        : "memory");
}

__device__ __forceinline__ float4 lds_v4_f32(uint32_t addr)
{
    float4 v;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(addr));
    return v;
}

__device__ __forceinline__ uint32_t smem_u32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// A coherent global load (ld.global, not the non-coherent ld.global.nc), written out so that the compiler can neither
// turn it into a read-only-path load nor move it above the PDL wait (both statements are volatile).
__device__ __forceinline__ float ld_global_f32(float const* p)
{
    float v;
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}

// Programmatic dependent launch: wait for the preceding kernel of the stream (a no-op without a programmatic
// dependency).
__device__ __forceinline__ void pdl_wait()
{
    asm volatile("griddepcontrol.wait;" ::: "memory");
}
} // namespace fso_dense_splitk2

// SHAPE_K: K (a multiple of 128). kWeightHalves: the layout (fso_dense_splitk2_layout.cuh). BLOCK_T: the token tile, at
// least M (a multiple of 16 from 32 to 128 for token halves, of 8 from 16 to 128 for weight halves). kNumSplits: the
// CTAs of a cluster (1 to 8). kNumStages: the stage ring. kBias, kAct, kResidual: the dense epilogue variant (all false
// / kDenseActNone: the identity).
template <uint32_t SHAPE_K, bool kWeightHalves, uint32_t BLOCK_T, uint32_t kNumSplits, uint32_t kNumStages, bool kBias,
    int kAct, bool kResidual>
__global__ void __launch_bounds__(fso_dense_splitk2_smem::kNumThreads, 1)
    fp8_gemm_kernel_swapAB_splitk2(float const* scales_w, // weight scales [ceil(N/128), K/128] fp32, row-major
        float const* scales_x,                            // activation scales [K/128, align4(M)] fp32 (M fastest)
        uint32_t shape_m,                                 // M (tokens), at most BLOCK_T
        uint32_t shape_n,                                 // N (weight rows)
        const __grid_constant__ CUtensorMap tensor_map_w, // weight [N, K] fp8: kBlockW x 128 boxes, 128-byte swizzle
        const __grid_constant__ CUtensorMap tensor_map_x, // activation [M, K] fp8: BLOCK_T x 128 boxes
        const __grid_constant__ CUtensorMap tensor_map_d, // D [M, N] bf16: min(tokens per warp-group, M) x 16 boxes
        const fso::DenseEpilogue epilogue)                // bias / residual pointers; unused by the identity
{
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
    namespace L = fso_dense_splitk2_smem;
    namespace X = fso_dense_splitk2;

    constexpr uint32_t BLOCK_W = L::block_w(kWeightHalves);
    constexpr uint32_t BLOCK_K = 128;
    constexpr uint32_t kWgTokens = L::wg_tokens(kWeightHalves, BLOCK_T); // the WGMMA's N
    constexpr uint32_t kNumMathWarps = 8;
    constexpr uint32_t kNumMathThreads = L::kNumMathThreads;
    constexpr uint32_t kNumKBlocks = SHAPE_K / BLOCK_K;
    constexpr uint32_t kSlice = (kNumKBlocks + kNumSplits - 1) / kNumSplits; // the longest K slice of the cluster
    constexpr bool kRingWraps = kSlice > kNumStages; // a stage is filled more than once: the consumer releases it
    constexpr uint32_t kOwned = L::owned_slabs(kNumSplits);
    using WGMMA = typename FP8MMASelector<kWgTokens>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    constexpr uint32_t kNumAccum = WGMMA::kNumAccum; // kWgTokens / 2 per thread
    constexpr uint32_t kNumQuads = kNumAccum / 4;
    constexpr bool kTwoSets = kWgTokens <= 64;
    constexpr bool kEpilogue = kBias or kAct != fso::kDenseActNone or kResidual;
    using Epi = DenseEpilogueOf<Sm90DenseEpilogueOp<kBias, kAct, kResidual>>;
    // Registers per math thread: the WGMMA accumulator sets and the promoted sum, then the bias pair and the residual.
    using Policy = DenseOperandPolicy<kBias, kResidual, (kTwoSets ? 3 : 2) * kNumAccum, kNumQuads, 4>;
    DG_STATIC_ASSERT(kWeightHalves ? (BLOCK_T % 8 == 0 and BLOCK_T >= 16 and BLOCK_T <= 128)
                                   : (BLOCK_T % 16 == 0 and BLOCK_T >= 32 and BLOCK_T <= 128),
        "token tile of the layout");
    DG_STATIC_ASSERT(SHAPE_K % BLOCK_K == 0, "K must be a multiple of 128");
    DG_STATIC_ASSERT(kNumSplits >= 1 and kNumSplits <= L::kNumSlabs and kNumSplits <= kNumKBlocks, "1 .. 8 K slices");
    DG_STATIC_ASSERT(kNumStages >= 1, "at least one stage");
    DG_STATIC_ASSERT(kNumAccum % 4 == 0, "accumulator layout");
    DG_STATIC_ASSERT(L::x_bytes(BLOCK_T) % 1024 == 0 and L::w_bytes(kWeightHalves) % 1024 == 0,
        "every 128-byte-swizzled box starts on a 1024-byte boundary");
    DG_STATIC_ASSERT(kWeightHalves or (kWgTokens * 128) % 1024 == 0, "the second token half starts on 1024 bytes");
    DG_STATIC_ASSERT(L::fragment_bytes(kWeightHalves, BLOCK_T) % 16 == 0, "16-byte pushes");

    uint32_t const warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t const lane_idx = get_lane_id();
    uint32_t const rank = kNumSplits > 1 ? X::cta_rank_in_cluster() : 0u;
    uint32_t const tile = blockIdx.x / kNumSplits;
    uint32_t const row0 = tile * BLOCK_W; // first weight row (output column) of the tile
    uint32_t const kb_begin = rank * (kNumKBlocks / kNumSplits) + min(rank, kNumKBlocks % kNumSplits);
    uint32_t const kb_count = kNumKBlocks / kNumSplits + (rank < kNumKBlocks % kNumSplits ? 1u : 0u);

    if (threadIdx.x == kNumMathThreads)
    {
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_w));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_x));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_d));
    }
    __syncwarp();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    if (threadIdx.x == 0)
        DG_DEVICE_ASSERT((X::smem_u32(smem_buffer) & 1023u) == 0u and "split-K smem base must be 1024-byte aligned");
    auto smem_w = [&](uint32_t s) { return smem_buffer + s * L::w_bytes(kWeightHalves); };
    auto smem_x
        = [&](uint32_t s) { return smem_buffer + L::x_offset(kWeightHalves, kNumStages) + s * L::x_bytes(BLOCK_T); };
    uint8_t* const smem_d = smem_buffer + L::d_offset(kWeightHalves, BLOCK_T, kNumStages);
    uint8_t* const smem_mailbox = smem_buffer + L::mailbox_offset(kWeightHalves, BLOCK_T, kNumSplits, kNumStages);
    float* const smem_sw
        = reinterpret_cast<float*>(smem_buffer + L::sw_offset(kWeightHalves, BLOCK_T, kNumSplits, kNumStages));
    float* const smem_sx = reinterpret_cast<float*>(
        smem_buffer + L::sx_offset(kWeightHalves, BLOCK_T, kNumSplits, kNumStages, kSlice));
    auto barriers = reinterpret_cast<Barrier*>(
        smem_buffer + L::barriers_offset(kWeightHalves, BLOCK_T, kNumSplits, kNumStages, kSlice));
    Barrier* const full_barriers = barriers;
    Barrier* const empty_barriers = barriers + kNumStages;
    Barrier* const mailbox_barriers = barriers + 2 * kNumStages;

    // Barriers. A stage is full after its two TMA boxes; it is empty after one arrival per math warp. The mailbox
    // barrier of an owned slab completes when the S - 1 peers' fragments have arrived: one arrival (made here, with the
    // expected byte count) plus the transaction bytes the peers' st.async complete.
    if (threadIdx.x == kNumMathThreads)
    {
#pragma unroll
        for (uint32_t s = 0; s < kNumStages; ++s)
        {
            full_barriers[s].init(1);
            empty_barriers[s].init(kNumMathWarps);
        }
        if constexpr (kNumSplits > 1)
        {
#pragma unroll
            for (uint32_t j = 0; j < kOwned; ++j)
            {
                mailbox_barriers[j].init(1);
                if (j * kNumSplits + rank < L::kNumSlabs)
                    mailbox_barriers[j].arrive_and_expect_tx(
                        (kNumSplits - 1) * L::fragment_bytes(kWeightHalves, BLOCK_T));
            }
        }
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::fence_barrier_init();
    }
    // The CTA barrier publishes the initialisation to this CTA's threads; the cluster barrier's arrive (after the
    // initialising thread's fence.mbarrier_init.release.cluster) publishes it to the peers, which wait on it before
    // their first push.
    __syncthreads();
    if constexpr (kNumSplits > 1)
        cute::cluster_arrive_relaxed();

    constexpr uint32_t kTxBytes = L::w_bytes(kWeightHalves) + L::x_bytes(BLOCK_T);
    constexpr auto kNormalHint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);

    if (warp_idx == kNumMathWarps)
    {
        // Producer: one thread issues every TMA load of the CTA's K slice, all of them after the PDL wait.
        X::pdl_wait();
        if (lane_idx == 0)
        {
            for (uint32_t kb = 0; kb < kb_count; ++kb)
            {
                uint32_t const s = kb % kNumStages;
                uint32_t const phase = (kb / kNumStages) & 1u;
                if constexpr (kRingWraps)
                    empty_barriers[s].wait(phase ^ 1u);
                int32_t const k_idx = static_cast<int32_t>((kb_begin + kb) * BLOCK_K);
                cute::SM90_TMA_LOAD_2D::copy(&tensor_map_w, reinterpret_cast<uint64_t*>(&full_barriers[s]),
                    kNormalHint, smem_w(s), k_idx, static_cast<int32_t>(row0));
                cute::SM90_TMA_LOAD_2D::copy(&tensor_map_x, reinterpret_cast<uint64_t*>(&full_barriers[s]),
                    kNormalHint, smem_x(s), k_idx, 0);
                full_barriers[s].arrive_and_expect_tx(kTxBytes);
            }
        }
        if constexpr (kNumSplits > 1)
        {
            __syncwarp();
            cute::cluster_wait();
        }
        return;
    }

    // Math warp-groups. Every global read below comes after the PDL wait.
    X::pdl_wait();

    uint32_t const wg = warp_idx / 4;
    // This warp-group's operands inside a stage: the weight half (weight halves) or the token half (token halves).
    uint32_t const w_off = kWeightHalves ? wg * 64u * 128u : 0u;
    uint32_t const tok_base = kWeightHalves ? 0u : wg * kWgTokens;
    uint32_t const x_off = tok_base * 128u;

    // The scales of the CTA's K slice, loaded into registers now and stored to shared memory once K block 0's WGMMAs
    // are issued, so that their latency overlaps the first stage's data: the weight scales (the tile lies in one
    // 128-row scale block, so one float per K block) and the activation scales (BLOCK_T per K block, M fastest as the
    // quantizer writes them, row stride align4(M); the token rows from M on read as 0, so that the scales the quantize
    // inside linear_qx leaves unwritten there are never read).
    constexpr uint32_t kSwPerThread = (kSlice + kNumMathThreads - 1) / kNumMathThreads;
    constexpr uint32_t kSxPerThread = (kSlice * BLOCK_T + kNumMathThreads - 1) / kNumMathThreads;
    float sw_reg[kSwPerThread], sx_reg[kSxPerThread];
    {
        float const* const sw_ptr = scales_w + (row0 / 128) * kNumKBlocks + kb_begin;
#pragma unroll
        for (uint32_t j = 0; j < kSwPerThread; ++j)
        {
            uint32_t const i = threadIdx.x + j * kNumMathThreads;
            sw_reg[j] = i < kb_count ? X::ld_global_f32(sw_ptr + i) : 0.f;
        }
        uint32_t const m_ld = (shape_m + 3u) / 4u * 4u;
#pragma unroll
        for (uint32_t j = 0; j < kSxPerThread; ++j)
        {
            uint32_t const i = threadIdx.x + j * kNumMathThreads;
            uint32_t const kb = i / BLOCK_T, t = i % BLOCK_T;
            sx_reg[j] = (i < kb_count * BLOCK_T and t < shape_m)
                ? X::ld_global_f32(scales_x + (kb_begin + kb) * m_ld + t)
                : 0.f;
        }
    }

    // The slab of this warp, and the dense epilogue's operands of the slabs this CTA owns (registers only).
    uint32_t const owner = warp_idx % kNumSplits;
    uint32_t const local = warp_idx / kNumSplits; // index among the owner's slabs
    uint32_t const wrow_off = (kWeightHalves ? wg * 64u : 0u) + (warp_idx % 4) * 16u; // the slab's first row in the tile
    uint32_t const wrow0 = row0 + wrow_off + lane_idx / 4;
    uint32_t const tok0 = tok_base + 2 * (lane_idx % 4);
    DenseOperandRegsSwapAB<kBias, Policy::kResInRegs, kNumQuads> epi_regs;
    if constexpr (kEpilogue)
    {
        if (owner == rank)
            dense_stage_tile_swapab<Epi, Policy, 64, kWgTokens, kNumMathThreads, kNumQuads>(epi_regs, epilogue,
                row0 + (kWeightHalves ? wg * 64u : 0u), wrow0, shape_n, tok_base, tok0, shape_m, threadIdx.x);
    }

    uint32_t const scale_offset = tok_base + (lane_idx % 4) * 2;
    float final_accum[kNumAccum];
#pragma unroll
    for (uint32_t i = 0; i < kNumAccum; ++i)
        final_accum[i] = 0.f;

    // Issue the four WGMMAs of K block kb into `acc` once its stage is full.
    auto issue = [&](float* acc, uint32_t kb)
    {
        uint32_t const s = kb % kNumStages;
        full_barriers[s].wait((kb / kNumStages) & 1u);
#pragma unroll
        for (uint32_t i = 0; i < kNumAccum; ++i)
            warpgroup_fence_operand(acc[i]);
        warpgroup_arrive();
#pragma unroll
        for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++k)
        {
            auto desc_a = make_smem_desc(smem_w(s) + w_off + k * WGMMA::K, 1);
            auto desc_b = make_smem_desc(smem_x(s) + x_off + k * WGMMA::K, 1);
            WGMMA::wgmma(desc_a, desc_b, acc, k);
        }
        warpgroup_commit_batch();
    };
    // The scales of K block kb (the weight scale and this thread's token scales), loaded from shared memory into
    // registers before the wait for kb's WGMMAs, so that the load latency overlaps them: loaded after the wait they
    // cost about 1.1 us per call on 1280 x 7680 at M = 78 (FH7, runs/FH7/data/diag2.jsonl), where the shared memory is
    // busy with both warp-groups' WGMMA operand reads and the TMA writes.
    struct BlockScales
    {
        float w;
        float2 x[kNumQuads];
    };
    auto load_scales = [&](uint32_t kb)
    {
        BlockScales sc;
        sc.w = ld_shared(smem_sw + kb);
        float const* const sx = smem_sx + kb * BLOCK_T + scale_offset;
#pragma unroll
        for (uint32_t i = 0; i < kNumQuads; ++i)
            sc.x[i] = ld_shared(reinterpret_cast<float2 const*>(sx + i * 8));
        return sc;
    };
    // Once the WGMMAs of K block kb have completed: release its stage, and add (sw * sx) * acc to the sum, the scale
    // product weight scale first, in K order, as fp8_gemm_kernel_swapAB promotes. The release comes first (FH5: the ring
    // is latency-bound). It needs no proxy fence for the bytes: the ring holds only the two TMA boxes, which the math
    // threads read only through the WGMMAs (async proxy, complete at warpgroup_wait), and the scale loads read the
    // staging area outside the ring, which no TMA writes. They are drained before the release anyway (scales_done
    // above), so that no ld.shared of the thread is outstanding when its lane releases a stage, which
    // test_stage_release_audit_sm90 checks for every release. A slice of at most kNumStages K blocks fills every stage
    // once and never waits for a release, so its instantiations issue none (kRingWraps).
    // The thread's scale loads are drained before its lane releases a stage (fence.proxy.async orders the thread's
    // generic shared-memory reads before its later async-proxy operations, and __syncwarp orders the lanes' fences
    // before lane 0's release, as fso_fence_stage_reads does in the kernels that read scales from a stage). Placed
    // right before the release, the loads issued ahead of the WGMMA wait have usually completed (draining them right
    // after their issue cost 0.4-1.9 %, FH7 runs/FH7/data/diag13.jsonl).
    auto scales_done = [&]()
    {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
    };
    auto retire = [&](float* acc, uint32_t kb, BlockScales const& sc)
    {
#pragma unroll
        for (uint32_t i = 0; i < kNumAccum; ++i)
            warpgroup_fence_operand(acc[i]);
        if constexpr (kRingWraps)
        {
            scales_done();
            if (lane_idx == 0)
                empty_barriers[kb % kNumStages].arrive();
        }
#pragma unroll
        for (uint32_t i = 0; i < kNumQuads; ++i)
        {
            float const s0 = sc.w * sc.x[i].x, s1 = sc.w * sc.x[i].y;
            final_accum[i * 4 + 0] += s0 * acc[i * 4 + 0];
            final_accum[i * 4 + 1] += s1 * acc[i * 4 + 1];
            final_accum[i * 4 + 2] += s0 * acc[i * 4 + 2];
            final_accum[i * 4 + 3] += s1 * acc[i * 4 + 3];
        }
    };
    auto stage_scales = [&]()
    {
#pragma unroll
        for (uint32_t j = 0; j < kSwPerThread; ++j)
        {
            uint32_t const i = threadIdx.x + j * kNumMathThreads;
            if (i < kb_count)
                smem_sw[i] = sw_reg[j];
        }
#pragma unroll
        for (uint32_t j = 0; j < kSxPerThread; ++j)
        {
            uint32_t const i = threadIdx.x + j * kNumMathThreads;
            if (i < kb_count * BLOCK_T)
                smem_sx[i] = sx_reg[j];
        }
        asm volatile("bar.sync 1, %0;" ::"r"(kNumMathThreads) : "memory");
    };

    if constexpr (kTwoSets)
    {
        // Two accumulator sets: the WGMMAs of K block kb + 1 run while K block kb is promoted.
        float acc0[kNumAccum], acc1[kNumAccum];
        issue(acc0, 0);
        stage_scales();
        uint32_t kb = 0;
        while (true)
        {
            // acc0 holds K block kb in flight
            if (kb + 1 < kb_count)
            {
                issue(acc1, kb + 1);
                BlockScales const sc = load_scales(kb);
                warpgroup_wait<1>();
                retire(acc0, kb, sc);
                ++kb;
            }
            else
            {
                BlockScales const sc = load_scales(kb);
                warpgroup_wait<0>();
                retire(acc0, kb, sc);
                break;
            }
            // acc1 holds K block kb in flight
            if (kb + 1 < kb_count)
            {
                issue(acc0, kb + 1);
                BlockScales const sc = load_scales(kb);
                warpgroup_wait<1>();
                retire(acc1, kb, sc);
                ++kb;
            }
            else
            {
                BlockScales const sc = load_scales(kb);
                warpgroup_wait<0>();
                retire(acc1, kb, sc);
                break;
            }
        }
    }
    else
    {
        // One accumulator set: every block is promoted after its own WGMMAs complete.
        float acc[kNumAccum];
        issue(acc, 0);
        stage_scales();
        for (uint32_t kb = 0; kb < kb_count; ++kb)
        {
            if (kb > 0)
                issue(acc, kb);
            BlockScales const sc = load_scales(kb);
            warpgroup_wait<0>();
            retire(acc, kb, sc);
        }
    }

    // Reduction: slab warp_idx belongs to rank warp_idx % S.
    if constexpr (kNumSplits > 1)
    {
        cute::cluster_wait();
        if (owner != rank)
        {
            uint32_t const slot = rank < owner ? rank : rank - 1u;
            // Fragment layout: 16-byte vector i of lane l at (i * 32 + l) * 16 of the slot, so that the 32 lanes of one
            // vector index write (and the owner reads) 512 consecutive bytes, free of bank conflicts. (A lane-major
            // layout puts the lanes kNumAccum * 4 bytes apart, 128 or 256 bytes for 64- or 128-token warp-groups: every
            // lane of a vector index in the same banks; FH7 measured 1.9-4.4 us of sum and store on such slabs.)
            uint32_t const base = X::smem_u32(smem_mailbox)
                + (local * (kNumSplits - 1) + slot) * L::fragment_bytes(kWeightHalves, BLOCK_T) + lane_idx * 16;
            uint32_t const remote = X::map_to_rank(base, owner);
            uint32_t const remote_mbar = X::map_to_rank(X::smem_u32(&mailbox_barriers[local]), owner);
#pragma unroll
            for (uint32_t i = 0; i < kNumAccum / 4; ++i)
                X::st_async_v4(remote + i * 32 * 16, final_accum[i * 4 + 0], final_accum[i * 4 + 1],
                    final_accum[i * 4 + 2], final_accum[i * 4 + 3], remote_mbar);
            return;
        }
        // Owner: the S - 1 peers' fragments of this slab, then p_0 + p_1 + ... + p_{S-1} in rank order.
        mailbox_barriers[local].wait(0);
        // p_0 + p_1 + ... + p_{S-1} in rank order, without a per-element choice between this CTA's registers and a
        // mailbox slot (a select per element and peer cost 0.5-1.6 us per call on the 64- and 128-token warp-groups,
        // FH7 runs/FH7/data/diag11.jsonl): the sum starts at -0.0f (-0 + x == x for every x, signed zeros included),
        // the peers below this rank are added from their slots, then this CTA's own partial, then the peers above.
        float sum[kNumAccum];
#pragma unroll
        for (uint32_t i = 0; i < kNumAccum; ++i)
            sum[i] = -0.0f;
        uint32_t const slots = X::smem_u32(smem_mailbox)
            + local * (kNumSplits - 1) * L::fragment_bytes(kWeightHalves, BLOCK_T) + lane_idx * 16;
        auto add_slot = [&](uint32_t slot)
        {
#pragma unroll
            for (uint32_t i = 0; i < kNumAccum / 4; ++i)
            {
                float4 const v
                    = X::lds_v4_f32(slots + slot * L::fragment_bytes(kWeightHalves, BLOCK_T) + i * 32 * 16);
                sum[i * 4 + 0] = __fadd_rn(sum[i * 4 + 0], v.x);
                sum[i * 4 + 1] = __fadd_rn(sum[i * 4 + 1], v.y);
                sum[i * 4 + 2] = __fadd_rn(sum[i * 4 + 2], v.z);
                sum[i * 4 + 3] = __fadd_rn(sum[i * 4 + 3], v.w);
            }
        };
        for (uint32_t q = 0; q < rank; ++q)
            add_slot(q);
#pragma unroll
        for (uint32_t i = 0; i < kNumAccum; ++i)
            sum[i] = __fadd_rn(sum[i], final_accum[i]);
        for (uint32_t q = rank + 1; q < kNumSplits; ++q)
            add_slot(q - 1);
#pragma unroll
        for (uint32_t i = 0; i < kNumAccum; ++i)
            final_accum[i] = sum[i];
    }

    // The dense epilogue on the reduced sum (the transposed tile: the bias by weight row, the residual by
    // (token, weight row)), before the one rounding to bf16 below.
    if constexpr (kEpilogue)
        dense_epilogue_fragment_swapab<kBias, kAct, kResidual, kNumQuads, Policy::kResInRegs>(final_accum, epilogue,
            epi_regs.bias.ptr(), epi_regs.res.ptr(), wrow0, shape_n, tok0, shape_m);

    // Store: STSM (transposed) into this slab's [tokens][16] bf16 strip, then one TMA store of the strip at
    // (column row0 + wrow_off, row tok_base). Matrix j of the x4 store holds accumulators 8i + 2j, 8i + 2j + 1:
    // tokens 16i .. 16i + 7 (j = 0, 1) or 16i + 8 .. 16i + 15 (j = 2, 3), weight rows 0 .. 7 (j even) or 8 .. 15.
    uint8_t* const strip = smem_d + local * L::d_strip_bytes(kWeightHalves, BLOCK_T);
    uint32_t tid = 0;
    if (lane_idx < 8)
        tid = lane_idx * 16;
    else if (lane_idx < 16)
        tid = (lane_idx - 8) * 16 + 8;
    else if (lane_idx < 24)
        tid = (lane_idx - 8) * 16;
    else
        tid = (lane_idx - 16) * 16 + 8;
    auto strip_bf16 = reinterpret_cast<__nv_bfloat16*>(strip);
#pragma unroll
    for (uint32_t i = 0; i < kNumAccum / 8; ++i)
    {
        SM90_U32x4_STSM_T<nv_bfloat162>::copy(__float22bfloat162_rn({final_accum[i * 8 + 0], final_accum[i * 8 + 1]}),
            __float22bfloat162_rn({final_accum[i * 8 + 2], final_accum[i * 8 + 3]}),
            __float22bfloat162_rn({final_accum[i * 8 + 4], final_accum[i * 8 + 5]}),
            __float22bfloat162_rn({final_accum[i * 8 + 6], final_accum[i * 8 + 7]}), strip_bf16 + i * 16 * 16 + tid);
    }
    if constexpr (kNumAccum % 8 != 0)
    {
        SM90_U32x2_STSM_T<nv_bfloat162>::copy(
            __float22bfloat162_rn({final_accum[kNumAccum / 8 * 8 + 0], final_accum[kNumAccum / 8 * 8 + 1]}),
            __float22bfloat162_rn({final_accum[kNumAccum / 8 * 8 + 2], final_accum[kNumAccum / 8 * 8 + 3]}),
            strip_bf16 + kNumAccum / 8 * 16 * 16 + tid);
    }
    cute::tma_store_fence();
    __syncwarp();
    if (lane_idx == 0)
    {
        cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, strip, static_cast<int32_t>(row0 + wrow_off),
            static_cast<int32_t>(tok_base));
        cute::tma_store_arrive();
        cute::tma_store_wait<0>();
    }
    __syncwarp();
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_90a");
#endif
}

} // namespace deep_gemm
