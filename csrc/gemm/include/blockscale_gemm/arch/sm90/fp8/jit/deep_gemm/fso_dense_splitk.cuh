/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 fish-scales-ops contributors
 * SPDX-License-Identifier: Apache-2.0
 */

// Dense block-FP8 GEMM for small token counts on sm_90, swap-AB orientation, with K split across the CTAs of a
// thread-block cluster (fp8_gemm_kernel_swapAB_splitk): the in-cluster split-K route of gemm_dispatch_sm90
// (dispatch.cuh). deep_gemm::jit::get_dense_splitk_config (jit_utils.cuh) says where the dispatcher takes it.
//
// Why a separate kernel. fp8_gemm_kernel_swapAB (fp8_gemm_impl.cuh) gives every output element to one CTA, which
// streams the whole of K for its weight tile. On a narrow weight at decode sizes that leaves most of the machine idle:
// N = 2560 rows make 40 tiles of 64 rows, so 40 of 132 SMs stream the whole weight, and a CTA's stream runs at the rate
// one SM sustains (about 130 ns per 128-wide K block, against 103 ns for a bare TMA ring on the same grid; issue #12,
// FH3). This kernel gives each CTA one 64-row weight tile and one contiguous slice of K:
//   * grid x = tiles * kNumSplits; the kNumSplits CTAs of one tile form one cluster (1-D, consecutive blockIdx.x), and
//     CTA r of the cluster takes K blocks [r * kb / S + min(r, kb % S), ...) of the tile, the first kb % S CTAs one
//     more; with kNumSplits = 1 there is no cluster and the CTA takes all of K;
//   * one token tile of BLOCK_T rows (a multiple of 8 from 16 to 64) covers all of M;
//   * the stage ring runs once through the CTA's K slice, with up to kNumStages stages in flight.
//
// Programmatic dependent launch. Every thread executes griddepcontrol.wait before its first global read: the producer
// before its first TMA load, the math threads before they read the scales, the bias and the residual. Before the
// wait the CTA only carves its shared memory, initialises its barriers and prefetches its tensor-map descriptors
// (kernel parameters), as the other kernels of this directory do. The weight is read after the wait too: the kernel
// launched just before this one may be the one that wrote it (fso.dense.prepare_weight followed by linear), and the
// activation quantize ahead of a bf16-x call may trigger its dependents before its own wait. The kernel does not
// trigger its own dependents early; they launch when its CTAs have exited, as after the other dense GEMMs. The scales
// are read with coherent loads (ld.global, never the non-coherent path), because under programmatic launch the
// kernel's lifetime overlaps the kernel that writes them.
//
// Arithmetic. The math warp-group multiplies its 64 weight rows by the token tile with m64nBLOCK_Tk32 WGMMAs and
// promotes every 128-wide K block exactly as fp8_gemm_kernel_swapAB does (accum, then final += (sw * sx) * accum in K
// order, the scale product weight scale first), so with kNumSplits = 1 every output element is the bf16 rounding of
// the fp32 value fp8_gemm_kernel_swapAB computes. With kNumSplits = S > 1 each CTA holds the partial sum of its slice;
// the cluster then adds the S partials of every element in rank order (p_0 + p_1 + ... + p_{S-1}, one fp32 addition
// each, the first partial taken as it is) and rounds once. That is deterministic (no atomics, a fixed order, the same
// bytes on every call and replay), but it is not bitwise equal to the unsplit sum, whose additions follow K order: an
// element's fp32 value can differ from the one-CTA route's by about one rounding of the partial sums, which after the
// bf16 rounding is one unit in the last place on almost every element that differs, and a few more units only where
// the sum nearly cancels.
//
// Dense epilogue (interface v1; fso_dense_epilogue.cuh). With kBias, kAct or kResidual, the owner warp of a slab
// applies bias, erf-GELU and residual to the reduced fp32 sum, once, in the fixed order of common/epilogue_device.cuh,
// before the one rounding to bf16. Its bias and residual are loaded into registers right after the PDL wait (only by
// the warps that own a slab), so the main loop hides the loads. The identity instantiation compiles none of it.
//
// Reduction (no global workspace, no second launch). The 16-row slab of math warp w (w = 0 .. 3) is owned by cluster
// rank w % S. Every other CTA of the cluster pushes its fp32 fragment of that slab into the owner's shared memory with
// st.async.shared::cluster ... mbarrier::complete_tx, 16 bytes per instruction, into a mailbox slot reserved for it,
// and completes the owner's per-slab mbarrier by the bytes it wrote. The owner armed that barrier with
// arrive.expect_tx for (S - 1) fragments when it initialised it, before the cluster barrier arrive that publishes the
// initialisation, and every pusher waits on that cluster barrier before its first push, so a push never reaches an
// uninitialised barrier. The owner warp waits on the barrier, sums, applies the epilogue, rounds, and stores its slab:
// STSM (transposed) into a [BLOCK_T][16] bf16 staging strip and one TMA store of that strip into D [M, N] (rows past M
// and columns past N are clipped by the tensor map). A pushing CTA does not touch its peers' shared memory after its
// pushes, and every owner waits for all the bytes addressed to it before it exits, so no CTA's shared memory is
// written after its exit.
//
// Cluster barrier. Every thread arrives on it (relaxed) right after the CTA-wide barrier that publishes this CTA's
// mbarrier initialisation, and waits on it only before its first access to a peer (a pusher's first st.async) or, for
// the threads that never access a peer, before they exit. The prologue therefore does not wait for the slowest CTA
// of the cluster to start, which FH3 measured at about 0.3 us.
//
// Pipeline stages. A stage holds two TMA boxes (weight and activation) that the math threads read only through the
// WGMMAs; the scales are staged once per CTA in shared memory outside the ring, with plain stores. So a stage is
// released (empty-barrier arrive) as soon as its WGMMAs have completed (warpgroup_wait), with no generic-proxy load of
// the stage to order against the producer's next TMA write of it (the hazard fso_fence_stage_reads covers in the
// kernels that read scales from a stage). A slice of at most kNumStages K blocks fills every stage once; its
// instantiations neither release nor wait for releases (kRingWraps), so no release can run ahead of a load there.
//
// Shared memory: fso_dense_splitk_layout.cuh. The host computes the same sizes from the same functions.
#pragma once

#include "fso_dense_epilogue.cuh"
#include "fso_dense_splitk_layout.cuh"
#include "mma_utils.cuh"
#include "tma_utils.cuh"
#include "utils.cuh"

namespace deep_gemm
{

namespace fso_dense_splitk
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

// Programmatic dependent launch: wait for the preceding kernel of the stream (fso_pdl_wait in fp8_gemm_impl.cuh, which
// this header does not include; a no-op without a programmatic dependency).
__device__ __forceinline__ void pdl_wait()
{
    asm volatile("griddepcontrol.wait;" ::: "memory");
}
} // namespace fso_dense_splitk

static_assert(sizeof(fso::DenseEpilogue) == 56, "fso::DenseEpilogue layout (the host passes it by value)");

// SHAPE_K: K (a multiple of 128). BLOCK_T: the token tile, a multiple of 8 from 16 to 64, at least M. kNumSplits: the
// CTAs of a cluster (1 to 4: every rank owns at least one of the four slabs). kNumStages: the stage ring.
// kBias, kAct, kResidual: the dense epilogue variant (all false / kDenseActNone: the identity).
template <uint32_t SHAPE_K, uint32_t BLOCK_T, uint32_t kNumSplits, uint32_t kNumStages, bool kBias, int kAct,
    bool kResidual>
__global__ void __launch_bounds__(fso_dense_splitk_smem::kNumThreads, 1)
    fp8_gemm_kernel_swapAB_splitk(float const* scales_w, // weight scales [ceil(N/128), K/128] fp32, row-major
        float const* scales_x,                           // activation scales [K/128, align4(M)] fp32 (M fastest)
        uint32_t shape_m,                                // M (tokens), at most BLOCK_T
        uint32_t shape_n,                                // N (weight rows)
        const __grid_constant__ CUtensorMap tensor_map_w, // weight [N, K] fp8: 64 x 128 boxes, 128-byte swizzle
        const __grid_constant__ CUtensorMap tensor_map_x, // activation [M, K] fp8: BLOCK_T x 128 boxes
        const __grid_constant__ CUtensorMap tensor_map_d, // D [M, N] bf16: min(BLOCK_T, M) x 16 boxes
        const fso::DenseEpilogue epilogue)                // bias / residual pointers; unused by the identity
{
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
    namespace L = fso_dense_splitk_smem;
    namespace X = fso_dense_splitk;

    constexpr uint32_t BLOCK_W = L::kBlockW;
    constexpr uint32_t BLOCK_K = 128;
    constexpr uint32_t kNumMathWarps = 4;
    constexpr uint32_t kNumMathThreads = 32 * kNumMathWarps;
    constexpr uint32_t kNumKBlocks = SHAPE_K / BLOCK_K;
    constexpr uint32_t kSlice = (kNumKBlocks + kNumSplits - 1) / kNumSplits; // the longest K slice of the cluster
    constexpr bool kRingWraps = kSlice > kNumStages; // a stage is filled more than once: the consumer releases it
    constexpr uint32_t kOwned = L::owned_slabs(kNumSplits);
    using WGMMA = typename FP8MMASelector<BLOCK_T>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    constexpr uint32_t kNumAccum = WGMMA::kNumAccum; // BLOCK_T / 2 per thread
    constexpr uint32_t kNumQuads = kNumAccum / 4;
    constexpr bool kEpilogue = kBias or kAct != fso::kDenseActNone or kResidual;
    using Epi = DenseEpilogueOf<Sm90DenseEpilogueOp<kBias, kAct, kResidual>>;
    // Registers per math thread: two WGMMA accumulator sets and the promoted sum, then the bias pair and the residual.
    using Policy = DenseOperandPolicy<kBias, kResidual, 3 * kNumAccum, kNumQuads, 4>;
    DG_STATIC_ASSERT(BLOCK_T % 8 == 0 and BLOCK_T >= 16 and BLOCK_T <= 64, "token tile of 16 .. 64, a multiple of 8");
    DG_STATIC_ASSERT(SHAPE_K % BLOCK_K == 0, "K must be a multiple of 128");
    DG_STATIC_ASSERT(kNumSplits >= 1 and kNumSplits <= L::kNumSlabs and kNumSplits <= kNumKBlocks, "1 .. 4 K slices");
    DG_STATIC_ASSERT(kNumStages >= 1, "at least one stage");
    DG_STATIC_ASSERT(kNumAccum % 4 == 0, "accumulator layout");
    DG_STATIC_ASSERT(L::x_bytes(BLOCK_T) % 1024 == 0 and L::w_bytes() % 1024 == 0,
        "every 128-byte-swizzled box starts on a 1024-byte boundary");
    DG_STATIC_ASSERT(L::fragment_bytes(BLOCK_T) % 16 == 0, "16-byte pushes");
    DG_STATIC_ASSERT(not kResidual or Policy::kResInRegs, "the residual of a slab is held in registers");

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
    auto smem_w = [&](uint32_t s) { return smem_buffer + s * L::w_bytes(); };
    auto smem_x = [&](uint32_t s) { return smem_buffer + L::x_offset(kNumStages) + s * L::x_bytes(BLOCK_T); };
    uint8_t* const smem_d = smem_buffer + L::d_offset(BLOCK_T, kNumStages);
    uint8_t* const smem_mailbox = smem_buffer + L::mailbox_offset(BLOCK_T, kNumSplits, kNumStages);
    float* const smem_sw = reinterpret_cast<float*>(smem_buffer + L::sw_offset(BLOCK_T, kNumSplits, kNumStages));
    float* const smem_sx
        = reinterpret_cast<float*>(smem_buffer + L::sx_offset(BLOCK_T, kNumSplits, kNumStages, kSlice));
    auto barriers
        = reinterpret_cast<Barrier*>(smem_buffer + L::barriers_offset(BLOCK_T, kNumSplits, kNumStages, kSlice));
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
                    mailbox_barriers[j].arrive_and_expect_tx((kNumSplits - 1) * L::fragment_bytes(BLOCK_T));
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

    constexpr uint32_t kTxBytes = L::w_bytes() + L::x_bytes(BLOCK_T);
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

    // Math warp-group. Every global read below comes after the PDL wait.
    X::pdl_wait();

    // The scales of the CTA's K slice, loaded into registers now and stored to shared memory once K block 0's WGMMAs
    // are issued, so that their latency overlaps the first stage's data: the weight scales (the 64-row tile lies in one
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
    uint32_t const wrow0 = row0 + warp_idx * 16 + lane_idx / 4;
    uint32_t const tok0 = 2 * (lane_idx % 4);
    DenseOperandRegsSwapAB<kBias, Policy::kResInRegs, kNumQuads> epi_regs;
    if constexpr (kEpilogue)
    {
        if (owner == rank)
            dense_stage_tile_swapab<Epi, Policy, BLOCK_W, BLOCK_T, kNumMathThreads, kNumQuads>(
                epi_regs, epilogue, row0, wrow0, shape_n, 0u, tok0, shape_m, threadIdx.x);
    }

    uint32_t const scale_offset = (lane_idx % 4) * 2;
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
            auto desc_a = make_smem_desc(smem_w(s) + k * WGMMA::K, 1);
            auto desc_b = make_smem_desc(smem_x(s) + k * WGMMA::K, 1);
            WGMMA::wgmma(desc_a, desc_b, acc, k);
        }
        warpgroup_commit_batch();
    };
    // Once the WGMMAs of K block kb have completed: release its stage, and add (sw * sx) * acc to the sum, the scale
    // product weight scale first, in K order, as fp8_gemm_kernel_swapAB promotes. The release comes first, because the
    // ring is latency-bound and every nanosecond it waits delays the refill (promoting first cost 1-2 %, FH5 hold H).
    // It needs no proxy fence: the ring holds only the two TMA boxes, which the math threads read only through the
    // WGMMAs (async proxy, complete at warpgroup_wait), while the scale loads below read the staging area outside the
    // ring, which no TMA writes. A slice of at most kNumStages K blocks fills every stage once and never waits for a
    // release, so its instantiations issue none (kRingWraps).
    auto retire = [&](float* acc, uint32_t kb)
    {
#pragma unroll
        for (uint32_t i = 0; i < kNumAccum; ++i)
            warpgroup_fence_operand(acc[i]);
        if constexpr (kRingWraps)
        {
            if (lane_idx == 0)
                empty_barriers[kb % kNumStages].arrive();
        }
        float const scale_w = ld_shared(smem_sw + kb);
        float const* const sx = smem_sx + kb * BLOCK_T + scale_offset;
#pragma unroll
        for (uint32_t i = 0; i < kNumQuads; ++i)
        {
            float2 const v = ld_shared(reinterpret_cast<float2 const*>(sx + i * 8));
            float const s0 = scale_w * v.x, s1 = scale_w * v.y;
            final_accum[i * 4 + 0] += s0 * acc[i * 4 + 0];
            final_accum[i * 4 + 1] += s1 * acc[i * 4 + 1];
            final_accum[i * 4 + 2] += s0 * acc[i * 4 + 2];
            final_accum[i * 4 + 3] += s1 * acc[i * 4 + 3];
        }
    };

    // Two accumulator sets: the WGMMAs of K block kb + 1 run while K block kb is promoted.
    float acc0[kNumAccum], acc1[kNumAccum];
    issue(acc0, 0);
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
    }
    uint32_t kb = 0;
    while (true)
    {
        // acc0 holds K block kb in flight
        if (kb + 1 < kb_count)
        {
            issue(acc1, kb + 1);
            warpgroup_wait<1>();
            retire(acc0, kb);
            ++kb;
        }
        else
        {
            warpgroup_wait<0>();
            retire(acc0, kb);
            break;
        }
        // acc1 holds K block kb in flight
        if (kb + 1 < kb_count)
        {
            issue(acc0, kb + 1);
            warpgroup_wait<1>();
            retire(acc1, kb);
            ++kb;
        }
        else
        {
            warpgroup_wait<0>();
            retire(acc1, kb);
            break;
        }
    }

    // Reduction: slab warp_idx (16 weight rows x BLOCK_T tokens) belongs to rank warp_idx % S.
    if constexpr (kNumSplits > 1)
    {
        cute::cluster_wait();
        if (owner != rank)
        {
            uint32_t const slot = rank < owner ? rank : rank - 1u;
            uint32_t const base = X::smem_u32(smem_mailbox)
                + ((local * (kNumSplits - 1) + slot) * 32 + lane_idx) * (kNumAccum * 4);
            uint32_t const remote = X::map_to_rank(base, owner);
            uint32_t const remote_mbar = X::map_to_rank(X::smem_u32(&mailbox_barriers[local]), owner);
#pragma unroll
            for (uint32_t i = 0; i < kNumAccum / 4; ++i)
                X::st_async_v4(remote + i * 16, final_accum[i * 4 + 0], final_accum[i * 4 + 1],
                    final_accum[i * 4 + 2], final_accum[i * 4 + 3], remote_mbar);
            return;
        }
        // Owner: the S - 1 peers' fragments of this slab, then p_0 + p_1 + ... + p_{S-1} in rank order.
        mailbox_barriers[local].wait(0);
        float sum[kNumAccum];
#pragma unroll
        for (uint32_t q = 0; q < kNumSplits; ++q)
        {
#pragma unroll
            for (uint32_t i = 0; i < kNumAccum / 4; ++i)
            {
                float4 v;
                if (q == rank)
                {
                    v = make_float4(final_accum[i * 4 + 0], final_accum[i * 4 + 1], final_accum[i * 4 + 2],
                        final_accum[i * 4 + 3]);
                }
                else
                {
                    uint32_t const slot = q < rank ? q : q - 1u;
                    v = X::lds_v4_f32(X::smem_u32(smem_mailbox)
                        + ((local * (kNumSplits - 1) + slot) * 32 + lane_idx) * (kNumAccum * 4) + i * 16);
                }
                if (q == 0)
                {
                    sum[i * 4 + 0] = v.x;
                    sum[i * 4 + 1] = v.y;
                    sum[i * 4 + 2] = v.z;
                    sum[i * 4 + 3] = v.w;
                }
                else
                {
                    sum[i * 4 + 0] = __fadd_rn(sum[i * 4 + 0], v.x);
                    sum[i * 4 + 1] = __fadd_rn(sum[i * 4 + 1], v.y);
                    sum[i * 4 + 2] = __fadd_rn(sum[i * 4 + 2], v.z);
                    sum[i * 4 + 3] = __fadd_rn(sum[i * 4 + 3], v.w);
                }
            }
        }
#pragma unroll
        for (uint32_t i = 0; i < kNumAccum; ++i)
            final_accum[i] = sum[i];
    }

    // The dense epilogue on the reduced sum (the transposed tile: the bias by weight row, the residual by
    // (token, weight row)), before the one rounding to bf16 below.
    if constexpr (kEpilogue)
        dense_epilogue_fragment_swapab<kBias, kAct, kResidual, kNumQuads, Policy::kResInRegs>(final_accum, epilogue,
            epi_regs.bias.ptr(), epi_regs.res.ptr(), wrow0, shape_n, tok0, shape_m);

    // Store: STSM (transposed) into this slab's [BLOCK_T][16] bf16 strip, then one TMA store of the strip at
    // (column row0 + 16 * warp_idx, row 0). Matrix j of the x4 store holds accumulators 8i + 2j, 8i + 2j + 1:
    // tokens 16i .. 16i + 7 (j = 0, 1) or 16i + 8 .. 16i + 15 (j = 2, 3), weight rows 0 .. 7 (j even) or 8 .. 15.
    uint8_t* const strip = smem_d + local * L::d_strip_bytes(BLOCK_T);
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
        cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, strip, static_cast<int32_t>(row0 + warp_idx * 16), 0);
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
