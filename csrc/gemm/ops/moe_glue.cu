/*
 * MoE routing/combine glue kernels for the sm_120 grouped MXFP8 path (M1).
 *
 * The M1 layer composition initially built its masked-layout index tensors
 * with ~8 torch ops (argsort / scatter_add / cumsum / index fills) and its
 * weighted combine with ~4 more. Under CUDA-graph replay each tiny kernel
 * still costs its GPU-side gap, and the measured glue (~45 us/layer at M=1
 * on RTX 5090) exceeded the grouped GEMMs themselves — the case-05
 * launch-economics failure reproduced inside our own pipeline. These two
 * kernels replace all of it:
 *
 *   moe_build_routing : topk_ids -> (masked_m, row_map, slot_of_flat, and
 *     optionally slot_to_expert) in ONE
 *     launch. Single CTA, shared-memory histogram (G <= kMaxGroups), atomic
 *     rank assignment. Slot order within a group is atomic-arrival order —
 *     a permutation of the sorted recipe, semantically equivalent (every
 *     consumer goes through row_map / slot_of_flat consistently). row_map
 *     slots at or beyond masked_m[g] are left uninitialised on purpose: no
 *     consumer reads them, and skipping the G*m_cap clear matters at decode.
 *   moe_combine : out[t] = sum_j topk_w[t,j] * dn_flat[slot_of_flat[t*topk+j]]
 *     vectorised 8 bf16 per thread.
 *
 * Both are capture-safe: fixed shapes, no host reads, masked_m content is
 * produced on device. Layer composition after this file: routing(1) +
 * gather-quant(1) + grouped gate_up(1) + silu-quant(1) + grouped down(1) +
 * combine(1) = 6 kernels, zero torch-op glue.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace blockscale_gemm
{
namespace
{

constexpr int kMaxGroups = 1024;
constexpr int kRoutingThreads = 512;

// The (N, K) pairs of the grouped GEMMs a layer will run on this routing, for
// which the routing kernels also emit CUTLASS-style per-group problem shapes
// (run b300_round3_20260922/M-A4). Passed to the kernel by value; four is
// more than any MoE layer of this library runs (two routed GEMMs), and the
// op refuses more rather than sizing the struct for an unbounded list.
constexpr int kMaxProblemShapeSets = 4;

struct ProblemShapeList
{
    int32_t n[kMaxProblemShapeSets];
    int32_t k[kMaxProblemShapeSets];
    int32_t count;
};

// Write group g's (rows, N, K) triple for every requested GEMM. Called by the
// same thread, in the same pass, that publishes masked_m[g], so the shapes
// and the counts a GEMM reads are published together. The row count is
// clamped to m_cap exactly as the sm_100 preparation kernel clamps it, so the
// two ways of producing the array agree byte for byte on a malformed routing
// as well as on a legal one.
__device__ __forceinline__ void fso_emit_problem_shapes(
    int32_t* __restrict__ problem_shapes, ProblemShapeList const& ps, int g, int num_groups, int rows, int m_cap)
{
    if (problem_shapes == nullptr)
        return;
    int const m = rows < 0 ? 0 : (rows > m_cap ? m_cap : rows);
    for (int p = 0; p < ps.count; ++p)
    {
        int32_t* dst = problem_shapes + (static_cast<size_t>(p) * num_groups + g) * 3;
        dst[0] = m;
        dst[1] = ps.n[p];
        dst[2] = ps.k[p];
    }
}

// Emit the packed active-expert list that the sm_100 slot-bound grouped MXFP8
// GEMM's grid is indexed by: the ids of the experts holding at least one routed
// row, ascending, in the low slots, and -1 in every slot after them.
//
// Why it lives here. That GEMM sizes its grid by a host-static bound on the
// number of experts that can hold rows and remaps each CTA's batch coordinate
// from a slot to an expert id through this list, so the list has to be rebuilt
// on device on every call (and on every graph replay). Building it takes the
// per-expert routed row count and nothing else, and this kernel already holds
// that count in shared memory, so a caller that asks for the list here pays a
// block-wide scan instead of a second one-block launch of its own
// (run b300_mxfp8_20260917/M-P1 §5.5 measured that launch at 1.58-2.32 us).
//
// Preconditions: `counts` holds the FINAL per-expert routed row count, the
// block has synchronised on it, and every slot of `slot_to_expert` has already
// been set to -1 (the scan only writes the active prefix). Block-uniform: every
// thread of the block must reach this call.
__device__ inline void fso_emit_slot_list(
    int32_t* __restrict__ slot_to_expert, int32_t const* __restrict__ counts, int num_groups)
{
    // ONE warp does the whole scan, and that is the point: a block-wide scan
    // needs two `__syncthreads()` -- one to publish the per-warp totals, one to
    // publish their prefixes -- and two barriers across the routing kernel's 512
    // threads cost about as much again as the entire kernel without the list
    // (0.69 against 0.68 microseconds at Family B M = 1, run
    // b300_mxfp8_20260917/M-I1, stage slopes). A single warp carries the base in
    // a register instead and needs no barrier at all, covering kMaxGroups in 32
    // iterations of a 32-lane shuffle scan. Every thread of the block may call
    // this; the warps that do not participate return immediately, which is safe
    // precisely because there is no barrier inside.
    if ((static_cast<int>(threadIdx.x) >> 5) != 0)
        return;
    int const lane = static_cast<int>(threadIdx.x) & 31;
    int base = 0;
    for (int g0 = 0; g0 < num_groups; g0 += 32)
    {
        int const g = g0 + lane;
        int const active = (g < num_groups && counts[g] > 0) ? 1 : 0;
        int x = active;
        for (int off = 1; off < 32; off <<= 1)
        {
            int const y = __shfl_up_sync(0xffffffffu, x, off);
            if (lane >= off)
                x += y;
        }
        if (active)
            slot_to_expert[base + x - active] = g;
        base += __shfl_sync(0xffffffffu, x, 31);
    }
}

// The body of the single-CTA routing builder, shared by the two kernels below
// it. `kShapes` selects whether the per-group problem shapes are emitted (run
// b300_round3_20260922/M-A4); with it false the body is the pre-existing
// kernel's, instruction for instruction, so callers that never ask for the
// shapes keep the kernel they had, on every architecture.
template <bool kShapes, bool kWeights>
__device__ __forceinline__ void moe_build_routing_body(
    int32_t const* __restrict__ topk_ids, // [M * topk]
    int32_t* __restrict__ masked_m,       // [G]
    int32_t* __restrict__ row_map,        // [G * m_cap] (valid slots only)
    int32_t* __restrict__ slot_of_flat,   // [M * topk]
    int32_t* __restrict__ slot_to_expert, // [G] or nullptr (see fso_emit_slot_list)
    int32_t* __restrict__ problem_shapes, // [P * G * 3] (see fso_emit_problem_shapes), read iff kShapes
    ProblemShapeList const& ps,
    float const* __restrict__ topk_w,     // [M * topk], read iff kWeights
    float* __restrict__ weight_of_slot,   // [G * m_cap] at valid slots, written iff kWeights
    int num_pairs, int topk, int num_groups, int m_cap, bool pdl)
{
    // PDL: order the topk_ids read behind the parent (upstream layer /
    // router), then release the dependent's prologue.
    //
    // The release stays where it is when this call also emits the slot list.
    // The consumer of that list -- the sm_100 slot-bound GEMM -- reads it in
    // its own prologue, so the ordering it needs is enforced on its side, by a
    // `griddepcontrol.wait` placed ahead of that read; see the generator
    // csrc/gemm/tools/make_sm100_slot_kernel.py, edit 4c. Withholding the
    // release here instead would have cost every OTHER consumer of this kernel
    // its prologue overlap, on every architecture, for a hazard that belongs to
    // one route.
    if (pdl)
    {
        cudaGridDependencySynchronize();
        if (threadIdx.x == 0)
        {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }
    // +1: the dummy histogram slot that a routed pair with an expert id outside
    // [0, num_groups) counts into, so the pair loop below keeps one atomic per
    // pair with no divergent path around it. The prefix over the real groups
    // never reads it.
    __shared__ int32_t cnt[kMaxGroups + 1];
    for (int g = threadIdx.x; g < num_groups; g += blockDim.x)
        cnt[g] = 0;
    if (threadIdx.x == 0)
        cnt[kMaxGroups] = 0;
    if (slot_to_expert != nullptr)
    {
        for (int s = threadIdx.x; s < num_groups; s += blockDim.x)
            slot_to_expert[s] = -1;
    }
    __syncthreads();

    // Expert ids outside [0, num_groups) are skipped, which is what a serving
    // engine needs from this builder on two counts. A CUDA-graph or
    // piecewise-graph bucket is captured for a fixed token count and padded
    // with rows that carry no real token; sglang gives those rows the expert id
    // `num_experts` (its moe_align overflow slot) or -1. Under expert
    // parallelism the dispatcher also rewrites every expert this rank does not
    // own to -1, so a rank sees ids outside its own local range on real tokens
    // too. Such a pair must cost no expert compute and must contribute nothing
    // to the token's output. It therefore gets no slot: the histogram entry
    // goes to the dummy slot, `row_map` is left alone, and `slot_of_flat` is set
    // to -1, which the gather, the SwiGLU-requantize and the combine all read as
    // "no routed row". A token whose every entry is skipped produces a zero
    // output row. Without this the pair indexed cnt[e] and wrote
    // row_map[e * m_cap + r] with e negative or past the last group, which is a
    // shared-memory and a global out-of-bounds write (the same fault the sm_90
    // sorted builder was fixed for in b1bbda2).
    for (int i = threadIdx.x; i < num_pairs; i += blockDim.x)
    {
        unsigned const ue = static_cast<unsigned>(topk_ids[i]);
        bool const valid = ue < static_cast<unsigned>(num_groups);
        int const e = valid ? static_cast<int>(ue) : kMaxGroups; // dummy slot
        int const r = atomicAdd(&cnt[e], 1);
        if (valid)
        {
            int const slot = e * m_cap + r;
            row_map[slot] = i / topk; // source token row
            slot_of_flat[i] = slot;
            if constexpr (kWeights)
            {
                // The combine weight indexed by SLOT rather than by pair: the
                // fused FC2 epilogue knows which row it holds, not which pair
                // produced it. Padding slots are left alone, like row_map's.
                weight_of_slot[slot] = topk_w[i];
            }
        }
        else
        {
            slot_of_flat[i] = -1; // skipped: padded row, or an expert another rank owns
        }
    }
    __syncthreads();

    for (int g = threadIdx.x; g < num_groups; g += blockDim.x)
    {
        masked_m[g] = cnt[g];
        if constexpr (kShapes)
            fso_emit_problem_shapes(problem_shapes, ps, g, num_groups, cnt[g], m_cap);
    }

    if (slot_to_expert != nullptr)
        fso_emit_slot_list(slot_to_expert, cnt, num_groups);
}

__global__ void moe_build_routing_kernel(
    int32_t const* __restrict__ topk_ids, // [M * topk]
    int32_t* __restrict__ masked_m,       // [G]
    int32_t* __restrict__ row_map,        // [G * m_cap] (valid slots only)
    int32_t* __restrict__ slot_of_flat,   // [M * topk]
    int32_t* __restrict__ slot_to_expert, // [G] or nullptr (see fso_emit_slot_list)
    int num_pairs, int topk, int num_groups, int m_cap, bool pdl)
{
    moe_build_routing_body<false, false>(topk_ids, masked_m, row_map, slot_of_flat, slot_to_expert, nullptr,
        ProblemShapeList{}, nullptr, nullptr, num_pairs, topk, num_groups, m_cap, pdl);
}

// The same builder, also publishing each slot's combine weight, which is what the
// fused FC2 epilogue needs to add its rows straight into the layer output. Its own
// instantiation, so the two kernels above stay byte-identical for every caller
// that does not ask for it.
__global__ void moe_build_routing_w_kernel(
    int32_t const* __restrict__ topk_ids, int32_t* __restrict__ masked_m, int32_t* __restrict__ row_map,
    int32_t* __restrict__ slot_of_flat, int32_t* __restrict__ slot_to_expert,
    float const* __restrict__ topk_w, float* __restrict__ weight_of_slot,
    int num_pairs, int topk, int num_groups, int m_cap, bool pdl)
{
    moe_build_routing_body<false, true>(topk_ids, masked_m, row_map, slot_of_flat, slot_to_expert, nullptr,
        ProblemShapeList{}, topk_w, weight_of_slot, num_pairs, topk, num_groups, m_cap, pdl);
}

// The same builder, also emitting the per-group problem shapes of every GEMM
// the caller named (run b300_round3_20260922/M-A4).
__global__ void moe_build_routing_ps_kernel(
    int32_t const* __restrict__ topk_ids, // [M * topk]
    int32_t* __restrict__ masked_m,       // [G]
    int32_t* __restrict__ row_map,        // [G * m_cap] (valid slots only)
    int32_t* __restrict__ slot_of_flat,   // [M * topk]
    int32_t* __restrict__ slot_to_expert, // [G] or nullptr (see fso_emit_slot_list)
    int32_t* __restrict__ problem_shapes, // [P * G * 3] (see fso_emit_problem_shapes)
    ProblemShapeList ps,
    int num_pairs, int topk, int num_groups, int m_cap, bool pdl)
{
    moe_build_routing_body<true, false>(topk_ids, masked_m, row_map, slot_of_flat, slot_to_expert, problem_shapes,
        ps, nullptr, nullptr, num_pairs, topk, num_groups, m_cap, pdl);
}

// kBias adds a per-token row on top of the weighted expert sum, optionally
// scaled by a per-token factor: that is a MoE block's shared expert and its
// sigmoid gate, which every other layout pays a separate elementwise pass for.
// With kBias false the body is what it was, instruction for instruction, so the
// layers that have no shared expert keep the kernel they had on every arch.
template <bool kBias>
__global__ void moe_combine_kernel(
    __nv_bfloat16 const* __restrict__ dn, // [G * m_cap, H]
    int32_t const* __restrict__ slot_of_flat, // [M * topk]
    float const* __restrict__ topk_w,     // [M, topk]
    __nv_bfloat16* __restrict__ out,      // [M, H]
    __nv_bfloat16 const* __restrict__ bias,   // [M, H] or nullptr (read iff kBias)
    float const* __restrict__ bias_scale,     // [M] or nullptr: per-token factor on bias
    int M, int topk, int H, bool pdl)
{
    constexpr int kVec = 8; // 8 bf16 = 16 bytes per thread
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    int const num_hvec = H / kVec;
    // A thread past the end has nothing to load and nothing to wait for. Thread 0
    // of every CTA is in range (the grid is ceil(total / blockDim)), so every CTA
    // still reaches the PDL trigger below.
    if (tid >= M * num_hvec)
        return;
    int const t = tid / num_hvec;
    int const h = (tid % num_hvec) * kVec;

    // Prefetch all topk routed rows before reducing so the scattered LDG.128
    // latencies overlap. At M=1 this kernel is a single latency-bound CTA and
    // the old load->use-per-iteration chain serialised topk dependent loads.
    // Fast path unrolls a fixed bound (covers the common MoE topk); larger topk
    // falls back to the sequential loop.
    constexpr int kMaxTopk = 8;
    bool const fast = topk <= kMaxTopk;
    float ws[kMaxTopk];
    int slots[kMaxTopk];
    if (fast)
    {
        // Three unrolled phases so the topk slot / weight loads, then the topk LDG.128 row loads, issue
        // back-to-back (two memory round trips per thread, as before masked entries were supported: a
        // branch or select inside a single loop cost +1.4-1.7 us at M = 8). A masked padded-row entry
        // (sorted layout, flat_to_sorted = -1) reads row 0, always allocated, and is zeroed afterwards.
        //
        // The first phase runs BEFORE the PDL wait below. slot_of_flat and topk_w are routing outputs,
        // written four launches upstream of this kernel in every MoE chain of this library, so they are
        // grandparent-or-older data like everything a kernel of the chain reads before its wait; only
        // the dn rows come from the parent (the FC2). A CTA that is resident before the FC2 ends, which
        // the 128-thread launch in moe_combine() makes the common case at decode, has its first round
        // trip done when the wait returns. Direct callers inherit the chain's precondition: neither
        // input may be written by the kernel launched immediately before this one. Measured in run
        // sm120_r2_p1_20260929 (REPORT.md, part 2).
#pragma unroll
        for (int j = 0; j < kMaxTopk; ++j)
        {
            if (j < topk)
            {
                slots[j] = slot_of_flat[t * topk + j];
                ws[j] = topk_w[t * topk + j];
            }
        }
    }

    // PDL entry (see moe_build_routing_kernel): the wait orders every dn read behind the parent,
    // and the trigger follows the wait as in every kernel of the chain.
    if (pdl)
    {
        cudaGridDependencySynchronize();
        if (threadIdx.x == 0)
        {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }

    float acc[kVec];
#pragma unroll
    for (int v = 0; v < kVec; ++v)
        acc[v] = 0.f;

    if (fast)
    {
        float4 raws[kMaxTopk];
#pragma unroll
        for (int j = 0; j < kMaxTopk; ++j)
        {
            if (j < topk)
            {
                int const src = slots[j] < 0 ? 0 : slots[j];
                raws[j] = *reinterpret_cast<float4 const*>(&dn[static_cast<int64_t>(src) * H + h]);
            }
        }
#pragma unroll
        for (int j = 0; j < kMaxTopk; ++j)
        {
            if (j < topk && slots[j] < 0)
            {
                ws[j] = 0.f;
                raws[j].x = 0.f;
                raws[j].y = 0.f;
                raws[j].z = 0.f;
                raws[j].w = 0.f;
            }
        }
#pragma unroll
        for (int j = 0; j < kMaxTopk; ++j)
        {
            if (j < topk)
            {
                __nv_bfloat162 const* v2 = reinterpret_cast<__nv_bfloat162 const*>(&raws[j]);
#pragma unroll
                for (int p = 0; p < kVec / 2; ++p)
                {
                    float2 const f = __bfloat1622float2(v2[p]);
                    acc[2 * p] += ws[j] * f.x;
                    acc[2 * p + 1] += ws[j] * f.y;
                }
            }
        }
    }
    else
    {
        for (int j = 0; j < topk; ++j)
        {
            int const slot = slot_of_flat[t * topk + j];
            bool const valid = slot >= 0; // masked padded-row entry (sorted layout): no routed row
            float const wl = topk_w[t * topk + j];
            float const w = valid ? wl : 0.f;
            float4 const rawl = *reinterpret_cast<float4 const*>(
                &dn[static_cast<int64_t>(valid ? slot : 0) * H + h]);
            float4 const raw = valid ? rawl : make_float4(0.f, 0.f, 0.f, 0.f);
            __nv_bfloat162 const* v2 = reinterpret_cast<__nv_bfloat162 const*>(&raw);
#pragma unroll
            for (int p = 0; p < kVec / 2; ++p)
            {
                float2 const f = __bfloat1622float2(v2[p]);
                acc[2 * p] += w * f.x;
                acc[2 * p + 1] += w * f.y;
            }
        }
    }

    if constexpr (kBias)
    {
        float const bs = (bias_scale != nullptr) ? bias_scale[t] : 1.f;
        float4 const raw = *reinterpret_cast<float4 const*>(&bias[static_cast<int64_t>(t) * H + h]);
        __nv_bfloat162 const* v2 = reinterpret_cast<__nv_bfloat162 const*>(&raw);
#pragma unroll
        for (int p = 0; p < kVec / 2; ++p)
        {
            float2 const f = __bfloat1622float2(v2[p]);
            acc[2 * p] += bs * f.x;
            acc[2 * p + 1] += bs * f.y;
        }
    }

    __nv_bfloat162 packed[kVec / 2];
#pragma unroll
    for (int p = 0; p < kVec / 2; ++p)
        packed[p] = __floats2bfloat162_rn(acc[2 * p], acc[2 * p + 1]);
    *reinterpret_cast<float4*>(&out[static_cast<int64_t>(t) * H + h])
        = *reinterpret_cast<float4 const*>(&packed[0]);
}

// -----------------------------------------------------------------------
// Multi-CTA routing builder — sm_100 mid / prefill band (T5, 2026-09-17).
//
// Why a second kernel exists. `moe_build_routing_kernel` above is a single
// CTA, which is the right shape at decode: the work is a few hundred routed
// pairs and the cost is the launch, not the arithmetic. From roughly M = 256
// upwards it is the wrong shape: at Family B M = 1024 the kernel has to read
// 8192 routed pairs, write 8192 scattered `row_map` entries and 8192
// sequential `slot_of_flat` entries, and it does all of that from ONE
// streaming multiprocessor out of 148, which measured 10.7 us on the B300 —
// more than the whole SwiGLU-quantize kernel next to it.
//
// What this kernel does differently. The pairs are split over many CTAs and
// the per-expert rank assignment is done in two steps so that the number of
// GLOBAL atomics is one per (CTA, expert) instead of one per routed pair:
//
//   phase 1  each CTA histograms its own chunk into shared memory;
//   phase 2  each CTA reserves a contiguous block of rows for every expert it
//            saw, with one atomicAdd on the global counter per expert, and
//            keeps the returned offset as that expert's base;
//   phase 3  each CTA re-reads its chunk and assigns each pair a row inside
//            its own reserved block using shared-memory atomics;
//   phase 4  the CTA that arrives last (a global arrival counter plus a
//            threadfence) publishes `masked_m` from the global counters, emits
//            the packed active-expert list when one was asked for, and
//            re-zeroes both scratch arrays for the next call.
//
// Slot order inside a group therefore changes from single-CTA arrival order to
// (CTA, arrival) order. That is the same kind of permutation the single-CTA
// kernel already documents as semantically free: every consumer addresses rows
// through `row_map` / `slot_of_flat`, the GEMM treats the rows of a group
// independently, and `moe_combine` sums over the token's topk slots in the
// same order as before, so the layer output is unchanged.
//
// Capture safety. The two scratch arrays are owned by a process-wide pool that
// allocates and zeroes them on the first (eager) call and never again; the
// kernel restores them to zero before it exits, so a captured graph that
// replays this kernel any number of times always finds them zero. A first call
// made inside a capture is refused with a message, exactly as the grouped
// GEMM's argument pool does.
// Body shared by the two multi-CTA kernels below, on the same `kShapes`
// footing as the single-CTA body above.
template <bool kShapes, bool kWeights>
__device__ __forceinline__ void moe_build_routing_multi_body(
    int32_t const* __restrict__ topk_ids, // [M * topk]
    int32_t* __restrict__ masked_m,       // [G]
    int32_t* __restrict__ row_map,        // [G * m_cap] (valid slots only)
    int32_t* __restrict__ slot_of_flat,   // [M * topk]
    int32_t* __restrict__ gcnt,           // [G]  scratch, zero in, zero out
    int32_t* __restrict__ gdone,          // [1]  scratch, zero in, zero out
    int32_t* __restrict__ slot_to_expert, // [G] or nullptr (see fso_emit_slot_list)
    int32_t* __restrict__ problem_shapes, // [P * G * 3] (see fso_emit_problem_shapes), read iff kShapes
    ProblemShapeList const& ps,
    float const* __restrict__ topk_w,     // [M * topk], read iff kWeights
    float* __restrict__ weight_of_slot,   // [G * m_cap] at valid slots, written iff kWeights
    int num_pairs, int topk, int num_groups, int m_cap, bool pdl)
{
    if (pdl)
    {
        cudaGridDependencySynchronize();
        if (threadIdx.x == 0)
        {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }

    // +1 in all three arrays: the dummy slot that pairs with an expert id
    // outside [0, num_groups) count and reserve into, so both pair loops keep
    // one atomic per pair. See the single-CTA body above for why those pairs
    // exist (graph padding rows and, under expert parallelism, experts another
    // rank owns) and what the -1 in `slot_of_flat` means to the consumers.
    __shared__ int32_t cnt[kMaxGroups + 1];
    __shared__ int32_t base[kMaxGroups + 1];
    __shared__ int32_t fill[kMaxGroups + 1];
    __shared__ int32_t s_last;

    for (int g = threadIdx.x; g < num_groups; g += blockDim.x)
    {
        cnt[g] = 0;
        fill[g] = 0;
    }
    if (threadIdx.x == 0)
    {
        cnt[kMaxGroups] = 0;
        base[kMaxGroups] = 0;
        fill[kMaxGroups] = 0;
    }
    __syncthreads();

    int const stride = gridDim.x * blockDim.x;
    int const start = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start; i < num_pairs; i += stride)
    {
        unsigned const ue = static_cast<unsigned>(topk_ids[i]);
        atomicAdd(&cnt[ue < static_cast<unsigned>(num_groups) ? static_cast<int>(ue) : kMaxGroups], 1);
    }
    __syncthreads();

    for (int g = threadIdx.x; g < num_groups; g += blockDim.x)
        base[g] = (cnt[g] > 0) ? atomicAdd(&gcnt[g], cnt[g]) : 0;
    __syncthreads();

    for (int i = start; i < num_pairs; i += stride)
    {
        unsigned const ue = static_cast<unsigned>(topk_ids[i]);
        bool const valid = ue < static_cast<unsigned>(num_groups);
        int const e = valid ? static_cast<int>(ue) : kMaxGroups; // dummy slot
        int const r = atomicAdd(&fill[e], 1);
        if (valid)
        {
            int const slot = e * m_cap + base[e] + r;
            row_map[slot] = i / topk;
            slot_of_flat[i] = slot;
            if constexpr (kWeights)
                weight_of_slot[slot] = topk_w[i]; // see the single-CTA body
        }
        else
        {
            slot_of_flat[i] = -1; // skipped: padded row, or an expert another rank owns
        }
    }

    __threadfence();
    if (threadIdx.x == 0)
        s_last = (atomicAdd(gdone, 1) == gridDim.x - 1) ? 1 : 0;
    __syncthreads();
    if (s_last)
    {
        // The slot list is emitted in this same final pass, by the one CTA that
        // already publishes `masked_m`, because it is the only CTA that can see
        // the finished per-expert totals. Those totals live in `gcnt`, which the
        // same loop has to zero for the next call, so they are parked in the
        // shared `base` array -- dead since phase 3 -- and the scan reads them
        // from there.
        for (int g = threadIdx.x; g < num_groups; g += blockDim.x)
        {
            int32_t const total = gcnt[g];
            masked_m[g] = total;
            if constexpr (kShapes)
                fso_emit_problem_shapes(problem_shapes, ps, g, num_groups, total, m_cap);
            base[g] = total;
            gcnt[g] = 0;
        }
        if (slot_to_expert != nullptr)
        {
            for (int s = threadIdx.x; s < num_groups; s += blockDim.x)
                slot_to_expert[s] = -1;
        }
        __syncthreads();
        if (threadIdx.x == 0)
            *gdone = 0;
        if (slot_to_expert != nullptr)
            fso_emit_slot_list(slot_to_expert, base, num_groups);
    }
}

__global__ void moe_build_routing_multi_kernel(
    int32_t const* __restrict__ topk_ids, // [M * topk]
    int32_t* __restrict__ masked_m,       // [G]
    int32_t* __restrict__ row_map,        // [G * m_cap] (valid slots only)
    int32_t* __restrict__ slot_of_flat,   // [M * topk]
    int32_t* __restrict__ gcnt,           // [G]  scratch, zero in, zero out
    int32_t* __restrict__ gdone,          // [1]  scratch, zero in, zero out
    int32_t* __restrict__ slot_to_expert, // [G] or nullptr (see fso_emit_slot_list)
    int num_pairs, int topk, int num_groups, int m_cap, bool pdl)
{
    moe_build_routing_multi_body<false, false>(topk_ids, masked_m, row_map, slot_of_flat, gcnt, gdone,
        slot_to_expert, nullptr, ProblemShapeList{}, nullptr, nullptr, num_pairs, topk, num_groups, m_cap, pdl);
}

// The multi-CTA twin of moe_build_routing_w_kernel.
__global__ void moe_build_routing_multi_w_kernel(
    int32_t const* __restrict__ topk_ids, int32_t* __restrict__ masked_m, int32_t* __restrict__ row_map,
    int32_t* __restrict__ slot_of_flat, int32_t* __restrict__ gcnt, int32_t* __restrict__ gdone,
    int32_t* __restrict__ slot_to_expert, float const* __restrict__ topk_w, float* __restrict__ weight_of_slot,
    int num_pairs, int topk, int num_groups, int m_cap, bool pdl)
{
    moe_build_routing_multi_body<false, true>(topk_ids, masked_m, row_map, slot_of_flat, gcnt, gdone,
        slot_to_expert, nullptr, ProblemShapeList{}, topk_w, weight_of_slot, num_pairs, topk, num_groups, m_cap,
        pdl);
}

__global__ void moe_build_routing_multi_ps_kernel(
    int32_t const* __restrict__ topk_ids, // [M * topk]
    int32_t* __restrict__ masked_m,       // [G]
    int32_t* __restrict__ row_map,        // [G * m_cap] (valid slots only)
    int32_t* __restrict__ slot_of_flat,   // [M * topk]
    int32_t* __restrict__ gcnt,           // [G]  scratch, zero in, zero out
    int32_t* __restrict__ gdone,          // [1]  scratch, zero in, zero out
    int32_t* __restrict__ slot_to_expert, // [G] or nullptr (see fso_emit_slot_list)
    int32_t* __restrict__ problem_shapes, // [P * G * 3] (see fso_emit_problem_shapes)
    ProblemShapeList ps,
    int num_pairs, int topk, int num_groups, int m_cap, bool pdl)
{
    moe_build_routing_multi_body<true, false>(topk_ids, masked_m, row_map, slot_of_flat, gcnt, gdone,
        slot_to_expert, problem_shapes, ps, nullptr, nullptr, num_pairs, topk, num_groups, m_cap, pdl);
}


// -----------------------------------------------------------------------
// Contiguous (triton-style) sorted layout builder — H4a.
//
// The masked layout above scales the GEMM's work with the expert capacity E
// (the DeepGEMM GroupedMasked scheduler scans all E groups per block fetch).
// The contiguous layout instead sorts the M*topk routed pairs by expert into
// one compact list, padding each active expert's run up to BLOCK_M, so the
// GroupedContiguous scheduler enumerates only active padded blocks. This
// single CTA emits, capture-safe (fixed P_max buffers, device length scalar):
//   sorted_expert_ids [P_max] : owning expert id per sorted row across an
//     active expert's whole padded run; -1 for the trailing slack past the
//     actual padded length (never enumerated, thanks to the scheduler's
//     device length gate). = the scheduler's grouped_layout.
//   sorted_src [P_max] : source flat pair index (t*topk+s) for real rows; -1
//     for both intra-expert pad rows and trailing slack (gather/combine skip).
//   num_padded_dev [1] : the actual padded length P_actual (triton's
//     num_tokens_post_padded); the GEMM's device length gate reads it.
// Expert ids outside [0, num_groups) are skipped: sglang masks the rows past
// num_token_non_padded of a CUDA-graph / piecewise-graph bucket to -1 or to
// num_experts (its moe_align overflow slot), and those pairs must cost no
// expert compute. They get no sorted row (flat_to_sorted = -1); the gather,
// SwiGLU-quant and combine kernels skip them, so a token whose entries are
// all masked produces a zero output row.
// The first row of every block is always real (nb=ceil(cnt/BLOCK_M) implies
// (nb-1)*BLOCK_M < cnt), so grouped_layout read at a block's first row is
// always a valid expert even before the pad rows are labelled.
__global__ void moe_build_sorted_kernel(
    int32_t const* __restrict__ topk_ids, // [M * topk]
    int32_t* __restrict__ sorted_expert_ids, // [P_max] (block-start labels)
    int32_t* __restrict__ flat_to_sorted,     // [M * topk]: pair -> sorted row
    int32_t* __restrict__ num_padded_dev,     // [1]
    int num_pairs, int topk, int num_groups, int block_m, int p_max, bool pdl)
{
    if (pdl)
    {
        cudaGridDependencySynchronize();
        if (threadIdx.x == 0)
            cudaTriggerProgrammaticLaunchCompletion();
    }
    // +1: a dummy slot (index kMaxGroups) that masked padded-row entries (expert id outside
    // [0, num_groups)) count and scatter into, keeping both pair loops branch-free; the prefix
    // scan never reads it.
    __shared__ int32_t cnt[kMaxGroups + 1];  // per-expert routed count
    __shared__ int32_t off[kMaxGroups + 1];  // exclusive prefix of padded_e
    __shared__ int32_t fill[kMaxGroups + 1]; // per-expert scatter cursor
    __shared__ int32_t s_p_actual;

    // Phase 1: histogram routed pairs per expert.
    for (int g = threadIdx.x; g < num_groups; g += blockDim.x)
        cnt[g] = 0;
    if (threadIdx.x == 0)
    {
        cnt[kMaxGroups] = 0;
        off[kMaxGroups] = 0;
        fill[kMaxGroups] = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < num_pairs; i += blockDim.x)
    {
        // One unsigned min: a negative id lands on the dummy slot, an id in [num_groups, kMaxGroups)
        // on a slot the prefix scan never reads (it only covers [0, num_groups)).
        unsigned const ue = static_cast<unsigned>(topk_ids[i]);
        atomicAdd(&cnt[min(ue, static_cast<unsigned>(kMaxGroups))], 1);
    }
    __syncthreads();

    // Phase 2: exclusive prefix of padded_e = ceil(cnt/BM)*BM, done as a single
    // warp-0 shuffle scan over chunks of 32 experts (no internal syncs, no
    // serial 128-iter dependency chain). block_m is a power of 2, so the ceil
    // is a bitwise AND — avoiding 128 runtime integer divisions, which were the
    // dominant single-CTA cost (each ~tens of cycles, serialized).
    int const bm_mask = block_m - 1;
    if (threadIdx.x < 32)
    {
        int const lane = threadIdx.x;
        int base = 0;
        for (int chunk = 0; chunk < num_groups; chunk += 32)
        {
            int const e = chunk + lane;
            int const c = (e < num_groups) ? cnt[e] : 0;
            int const padded = c > 0 ? ((c + bm_mask) & ~bm_mask) : 0;
            int incl = padded;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1)
            {
                int const v = __shfl_up_sync(0xFFFFFFFFu, incl, d);
                if (lane >= d)
                    incl += v;
            }
            if (e < num_groups)
            {
                off[e] = incl - padded + base; // exclusive prefix
                fill[e] = 0;
            }
            base += __shfl_sync(0xFFFFFFFFu, incl, 31); // chunk total
        }
        if (lane == 0)
        {
            s_p_actual = base;
            num_padded_dev[0] = base;
        }
    }
    __syncthreads();
    int const p_actual = s_p_actual;

    // Phases 3+4 (independent, no sync between): label block-START rows only
    // (the scheduler reads grouped_layout at m_block*BLOCK_M) via an O(log E)
    // search over off[]; and scatter each routed pair to its sorted row. Both
    // only read off[]/fill[] from phase 2. Slack block starts past p_actual get
    // -1 for the scheduler length gate.
    int const num_blocks = p_max / block_m;
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x)
    {
        int const r0 = b * block_m;
        int eid = -1;
        if (r0 < p_actual)
        {
            int lo = 0, hi = num_groups - 1;
            while (lo < hi)
            {
                int mid = (lo + hi + 1) >> 1;
                if (off[mid] <= r0)
                    lo = mid;
                else
                    hi = mid - 1;
            }
            eid = lo;
        }
        sorted_expert_ids[r0] = eid;
    }
    for (int i = threadIdx.x; i < num_pairs; i += blockDim.x)
    {
        unsigned const ue = static_cast<unsigned>(topk_ids[i]);
        int const es = static_cast<int>(min(ue, static_cast<unsigned>(kMaxGroups))); // masked -> dummy slot
        int const r = atomicAdd(&fill[es], 1);
        // -1 for a masked padded-row entry (id outside [0, num_groups)): gather / combine skip it.
        flat_to_sorted[i] = (ue < static_cast<unsigned>(num_groups)) ? off[es] + r : -1;
    }
}

// -----------------------------------------------------------------------
// Router top-k: softmax selection, padded-row sentinel, expert-parallel remap
// and the shared expert's gate, in one launch.
//
// What a serving MoE block runs ahead of the experts. sglang's block computes
// the router logits with a replicated bf16 GEMM, takes a softmax over the
// expert dimension, selects the top k, renormalises the k probabilities, masks
// the rows past `num_token_non_padded` of a CUDA-graph bucket to a sentinel
// expert id, gathers the ids through a global-to-local expert map under expert
// parallelism, and -- for a model with a shared expert -- computes that
// expert's sigmoid gate from a second replicated weight. That is four or five
// launches ahead of a layer whose own kernels cost a few microseconds each at
// decode, which is the launch economics this file exists to fix.
//
// This kernel takes everything except the GEMM. The GEMM stays with the caller
// because cuBLAS already does it in one launch and at the DRAM roofline at
// every token count: the gate matrix is E*H*2 bytes (1 MB at E=256, H=2048)
// and a hand-written per-token form would have one CTA pull all of it, which
// measured far worse than the library GEMM at every M worth having.
//
// The shared expert's gate rides along as one extra logit column: a caller that
// concatenates its shared-gate weight row onto the router weight once at load
// time gets `sigmoid(hidden . shared_gate_w)` per token for free here, and the
// selection still runs over the first E columns only.
//
// Numerics. The renormalised softmax over the selected k does not need the full
// softmax denominator: dividing the k selected exp(l_j - max) by their own sum
// is algebraically the renormalised softmax whatever the shift, which is what
// sglang's `renormalize=True` path produces. Without renormalisation the full
// denominator is summed over all E experts first. The selection is a descending
// sequence of warp-wide maxima, so the ids come out in torch.topk's order
// (descending weight, and on an exact tie the lower expert id first) and the
// first selected value is the row maximum the softmax shifts by.
constexpr int kRouterMaxTopk = 8;
constexpr int kRouterSlotsSmall = 8;  // E <= 256:  8 experts per lane
constexpr int kRouterSlotsLarge = 32; // E <= 1024: 32 experts per lane
constexpr int kRouterThreads = 128;   // 4 warps = 4 tokens per CTA

// Explicit widening, so the kernel does not depend on whether the build leaves
// the __CUDA_NO_*_CONVERSIONS__ guards defined.
__device__ __forceinline__ float router_to_float(__nv_bfloat16 v)
{
    return __bfloat162float(v);
}

__device__ __forceinline__ float router_to_float(__half v)
{
    return __half2float(v);
}

__device__ __forceinline__ float router_to_float(float v)
{
    return v;
}

// Select the top `topk` of the logits this warp holds and write the ids and
// their softmax weights. `v[s]` is the logit of expert `lane + 32*s` (-inf past
// E); the array is consumed, winners being masked out as they are selected.
// `full_denom` is read only when `renormalize` is false, where it must be
// sum(exp(l - max)) over all E experts.
template <int kSlots>
__device__ __forceinline__ void router_select_topk(float (&v)[kSlots], int E, int topk, bool renormalize,
    float full_denom, int lane, int32_t const* __restrict__ expert_map, bool masked, int sentinel,
    int32_t* __restrict__ out_ids, float* __restrict__ out_w)
{
    if (masked)
    {
        // A padded row of a graph bucket: no expert runs for it. The sentinel is
        // the id the caller's engine gives those rows (its num_experts), passed
        // through the same expert map as a real id so an expert-parallel rank
        // sees the local out-of-range value its layer skips.
        if (lane == 0)
        {
            int const id = expert_map != nullptr ? expert_map[sentinel] : sentinel;
            for (int j = 0; j < topk; ++j)
            {
                out_ids[j] = id;
                out_w[j] = 0.f;
            }
        }
        return;
    }
    float best_v[kRouterMaxTopk];
    int best_e[kRouterMaxTopk];
    for (int j = 0; j < topk; ++j)
    {
        float bv = -INFINITY;
        int be = 0x7FFFFFFF;
#pragma unroll
        for (int s = 0; s < kSlots; ++s)
        {
            int const e = lane + 32 * s;
            // Strictly greater, scanned in ascending expert order, so a tie
            // inside one lane keeps the lower id.
            if (e < E && v[s] > bv)
            {
                bv = v[s];
                be = e;
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
        {
            float const ov = __shfl_xor_sync(0xFFFFFFFFu, bv, off);
            int const oe = __shfl_xor_sync(0xFFFFFFFFu, be, off);
            if (ov > bv || (ov == bv && oe < be))
            {
                bv = ov;
                be = oe;
            }
        }
        best_v[j] = bv;
        best_e[j] = be;
        // Mask the winner out of the owning lane's slots for the next pass.
        if ((be & 31) == lane)
        {
            int const sw = be >> 5;
#pragma unroll
            for (int s = 0; s < kSlots; ++s)
            {
                if (s == sw)
                    v[s] = -INFINITY;
            }
        }
    }
    if (lane != 0)
        return;
    float const shift = best_v[0]; // the row maximum: the selection is descending
    float ex[kRouterMaxTopk];
    float denom = 0.f;
    for (int j = 0; j < topk; ++j)
    {
        ex[j] = expf(best_v[j] - shift);
        denom += ex[j]; // summed in the order torch sums its topk output
    }
    if (!renormalize)
        denom = full_denom;
    float const inv = 1.f / denom;
    for (int j = 0; j < topk; ++j)
    {
        int const id = best_e[j];
        out_ids[j] = expert_map != nullptr ? expert_map[id] : id;
        out_w[j] = ex[j] * inv;
    }
}

// sum(exp(l - max)) over all E experts, needed only when the caller does not
// renormalise over the selected k.
template <int kSlots>
__device__ __forceinline__ float router_full_denom(float const (&v)[kSlots], int E, int lane)
{
    float m = -INFINITY;
#pragma unroll
    for (int s = 0; s < kSlots; ++s)
    {
        int const e = lane + 32 * s;
        if (e < E)
            m = fmaxf(m, v[s]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, off));
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < kSlots; ++s)
    {
        int const e = lane + 32 * s;
        if (e < E)
            acc += expf(v[s] - m);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
    return acc;
}

template <typename TLogit, int kSlots>
__global__ void moe_topk_from_logits_kernel(TLogit const* __restrict__ logits, // [M, ld]
    int32_t const* __restrict__ n_valid,                                       // [1] or nullptr
    int32_t const* __restrict__ expert_map,                                    // [E + 1] or nullptr
    int32_t* __restrict__ topk_ids,                                            // [M, topk]
    float* __restrict__ topk_w,                                                // [M, topk]
    float* __restrict__ shared_gate,                                           // [M] or nullptr
    int M, int E, int ld, int topk, int sentinel, bool renormalize, bool pdl)
{
    if (pdl)
    {
        cudaGridDependencySynchronize();
        if (threadIdx.x == 0)
        {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }
    int const lane = threadIdx.x & 31;
    int const token = static_cast<int>(blockIdx.x) * (kRouterThreads >> 5) + static_cast<int>(threadIdx.x >> 5);
    if (token >= M)
        return;
    int64_t const row = static_cast<int64_t>(token) * ld;
    // The shared expert's gate is logit column E, when the caller concatenated
    // that weight row onto the router's. A padded row still gets a value (the
    // block multiplies it by a zero-valued shared output), so it is written
    // before the mask check.
    if (shared_gate != nullptr && lane == 0)
    {
        float const l = router_to_float(logits[row + E]);
        shared_gate[token] = 1.f / (1.f + expf(-l));
    }
    bool const masked = (n_valid != nullptr) && (token >= *n_valid);
    float v[kSlots];
    float full_denom = 0.f;
    if (!masked)
    {
#pragma unroll
        for (int s = 0; s < kSlots; ++s)
        {
            int const e = lane + 32 * s;
            v[s] = (e < E) ? router_to_float(logits[row + e]) : -INFINITY;
        }
        if (!renormalize)
            full_denom = router_full_denom<kSlots>(v, E, lane);
    }
    router_select_topk<kSlots>(v, E, topk, renormalize, full_denom, lane, expert_map, masked, sentinel,
        topk_ids + static_cast<int64_t>(token) * topk, topk_w + static_cast<int64_t>(token) * topk);
}

} // anonymous namespace


// PDL launch helper (mirrors quant_kernels.cu): programmatic-stream-
// serialization attribute so downstream launches overlap this kernel.
// FSO_DISABLE_PDL=1 restores plain serialised launches.
static inline bool fso_pdl_enabled()
{
    static bool v = []
    {
        char const* e = std::getenv("FSO_DISABLE_PDL");
        return !(e && e[0] == '1');
    }();
    return v;
}

// See quant_kernels.cu: huge programmatic dependents flood the SMs and
// throttle the parent's tail, so large grids launch plain.
constexpr unsigned kFsoPdlMaxGridCtas = 4096;

template <typename KernelT, typename... Args>
static inline void fso_pdl_launch(KernelT kernel, dim3 grid, dim3 block, cudaStream_t stream, bool pdl, Args... args)
{
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernel, args..., pdl);
}


// Current device's SM major version (10 = sm_100 B200 / sm_103 B300), cached.
// Mirrors the helper in mxfp8.cu; the multi-CTA routing builder is selected on
// this device family only, so sm_120 and sm_90 keep the single-CTA kernel and
// its exact behaviour.
static bool moe_glue_is_sm100_family()
{
    static bool const v = []
    {
        auto const* prop = at::cuda::getCurrentDeviceProperties();
        return prop->major == 10;
    }();
    return v;
}

// Does this launch take the multi-CTA routing builder below instead of the
// single-CTA one?
//
// The multi-CTA kernel is arch-neutral C++ (a shared-memory histogram, one
// global atomic per (CTA, expert), and a last-CTA publish behind a threadfence),
// but until 2026-09-28 it was ON for sm_100/103 only, where it was written. The
// single-CTA builder does all of an M = 4096 Family C launch -- 32768 routed
// pairs, as many scattered row_map writes -- from one SM out of the 5090's 170,
// and an nsys timeline of the layer measured it at 30.4 us there, 3.3 % of the
// whole layer and more than the gather-quantize next to it. With the multi-CTA
// builder that kernel becomes 3.6 us and the Family C layer drops 0.49 % at
// M = 512, 1.13 % at M = 1024 and 4.36 % at M = 4096, so sm_120 now takes it too.
// The decode band is untouched either way: the path needs n_pairs >= 4096, which
// at top-8 means M >= 512, and below that the single CTA is the right shape
// because the cost there is the launch and not the arithmetic.
// FSO_MOE_ROUTING_MULTI=0 restores the single-CTA builder on any arch.
static bool moe_glue_routing_multi_enabled()
{
    static int const forced = []
    {
        char const* e = std::getenv("FSO_MOE_ROUTING_MULTI");
        return (e != nullptr && *e != '\0') ? (std::atoi(e) != 0 ? 1 : 0) : -1;
    }();
    if (forced >= 0)
    {
        return forced != 0;
    }
    auto const* prop = at::cuda::getCurrentDeviceProperties();
    return prop->major == 10 || prop->major == 12;
}

// Scratch for moe_build_routing_multi_kernel: per expert-group global counters
// plus one arrival counter, zero on entry and zero again on exit because the
// last CTA restores them.
//
// What the buffer has to guarantee, and what that implies
// ------------------------------------------------------
// The kernel's contract is that between the instant one launch reads a buffer
// and the instant the same launch restores it to zero, no other launch may
// touch it. Two launches that are ordered with respect to each other can
// therefore share a buffer; two launches that may execute concurrently must
// not. The pool below hands out one buffer per key, where the key is
//
//   * while the stream is capturing, the capture sequence id (every capture
//     gets its own id, so two graphs captured on the SAME stream - which is
//     what `torch.cuda.graph` does by default, since it reuses one class-level
//     capture stream - still get different buffers and may be replayed
//     concurrently on different streams);
//   * otherwise the stream handle (two launches on one stream are serialised
//     by stream ordering, so one buffer is enough for all of them, including
//     the 48 calls a many-layer model makes inside one captured graph).
//
// The map itself is `thread_local`, so two host threads never share a buffer
// and never race on the first-call allocation. Within a thread the map is
// only ever read and written from that thread.
//
// The one case this does not cover is the same captured `cudaGraph_t`
// instantiated twice into two `cudaGraphExec_t` and replayed concurrently;
// CUDA orders repeated launches of a single graphExec, but not two execs built
// from one graph. PyTorch instantiates once per `torch.cuda.CUDAGraph`, so the
// case does not arise through the supported API.
//
// Capture safety is unchanged from the previous single-buffer form: the whole
// ring is allocated and zeroed once, on the first eager call, and a first call
// that happens inside a capture aborts with the same message rather than
// allocating. A capture on a stream the pool has not seen before needs no
// allocation - it takes an already-zeroed slot - so the eager warm-up contract
// is no stricter than the grouped GEMM's ArgPool: one eager call per thread.
// If a thread runs out of slots the caller falls back to the single-CTA
// builder, which needs no scratch and is always correct.
struct RoutingScratch
{
    static constexpr int kSlots = 16;
    static constexpr std::size_t kWords = static_cast<std::size_t>(kMaxGroups) + 1;

    int32_t* base = nullptr;                          // kSlots * kWords int32
    int used = 0;                                     // slots handed out
    std::unordered_map<unsigned long long, int32_t*> by_key;
    bool warned = false;

    static RoutingScratch& instance()
    {
        static thread_local RoutingScratch s;
        return s;
    }

    int32_t* acquire(cudaStream_t stream)
    {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        unsigned long long cap_id = 0;
        if (cudaStreamGetCaptureInfo(stream, &cap, &cap_id) != cudaSuccess)
        {
            cap = cudaStreamCaptureStatusNone;
            cap_id = 0;
        }
        // Capture ids and stream handles live in different key spaces; the top
        // bit separates them so a capture id can never alias a stream handle.
        unsigned long long const key = (cap == cudaStreamCaptureStatusActive)
            ? (cap_id | (1ULL << 63))
            : static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(stream));

        auto it = by_key.find(key);
        if (it != by_key.end())
            return it->second;

        if (base == nullptr)
        {
            if (cap == cudaStreamCaptureStatusActive)
            {
                std::fprintf(stderr,
                    "[fish_scales_ops] moe_build_routing: the routing scratch pool is empty during stream capture. "
                    "Call moe_build_routing once eagerly before capturing.\n");
                std::abort();
            }
            std::size_t const bytes = sizeof(int32_t) * kWords * kSlots;
            if (cudaMalloc(&base, bytes) != cudaSuccess)
            {
                base = nullptr;
                return nullptr;
            }
            // Synchronous zeroing, once per thread and never inside a capture:
            // a later capture on a different stream takes an already-zeroed
            // slot, and stream ordering alone would not order that eager
            // memset before the captured kernel.
            if (cudaMemsetAsync(base, 0, bytes, stream) != cudaSuccess)
                return nullptr;
            if (cudaStreamSynchronize(stream) != cudaSuccess)
                return nullptr;
        }

        if (used >= kSlots)
        {
            if (!warned)
            {
                warned = true;
                std::fprintf(stderr,
                    "[fish_scales_ops] moe_build_routing: more than %d distinct streams/captures on one host "
                    "thread; falling back to the single-CTA routing builder for the rest. Results are unaffected.\n",
                    kSlots);
            }
            return nullptr;
        }
        int32_t* const p = base + static_cast<std::size_t>(used) * kWords;
        ++used;
        by_key.emplace(key, p);
        return p;
    }
};

// Routed-pair count from which the multi-CTA builder wins on sm_100. Below it
// the single CTA is the faster shape because the kernel is launch-bound, and
// the crossover was measured on the layer cell: at 2048 routed pairs (Family B
// and Family C at M = 256, top-8) the multi-CTA form is 0.2 us SLOWER because
// it only gets two CTAs and pays an extra pass over the pairs, while from 4096
// pairs (M = 512) on it is faster and the gap grows with M.
constexpr int kRoutingMultiMinPairs = 4096;
constexpr int kRoutingMultiThreads = 256;
constexpr int kRoutingMultiPairsPerCta = 256;
constexpr int kRoutingMultiMaxCtas = 132;


// moe_build_routing: topk_ids [M, topk] int32 -> (masked_m [G] int32,
// row_map [G * m_cap] int32, slot_of_flat [M * topk] int32), one launch.
//
// An expert id outside [0, num_groups) is skipped: no group counts it, it
// occupies no slot, its row_map entry is not written and its slot_of_flat entry
// is -1, which the gather-quantize, the SwiGLU requantize and the combine all
// read as "no routed row". The two kernel bodies above say why a serving engine
// produces such ids (graph padding rows, and experts another rank owns under
// expert parallelism) and what it costs to skip them.
//
// With `with_slots` it also returns `slot_to_expert` [G] int32: the ids of the
// experts that hold at least one routed row, ascending, in the low entries, and
// -1 in the rest. That is exactly the list the sm_100 slot-bound grouped MXFP8
// GEMM's grid is indexed by, and it is derived from the per-expert histogram
// this kernel already holds, so producing it here replaces a separate one-block
// launch per grouped GEMM (two per MoE layer). Without the flag the returned
// tensor is empty and no slot work is done at all, so every existing caller is
// unaffected. The list is architecture-independent glue -- sm_120 and sm_90
// produce it too, and simply have no route that consumes it.
//
// `problem_shapes_nk` (run b300_round3_20260922/M-A4) is a flat list of
// (N, K) pairs -- [N0, K0, N1, K1, ...] -- one per grouped GEMM the caller will
// run on this routing. For each pair the kernel also writes the per-group
// CUTLASS problem shape, the int32 triple (rows, N, K) with rows clamped to
// m_cap, into the fifth output, int32 [P, G, 3], in the same pass that writes
// `masked_m`. That triple is the only routing-dependent argument the sm_100
// pointer-array grouped GEMM has, so a caller that hands `problem_shapes[i]`
// to GEMM i lets it launch without its per-call preparation kernel. An empty
// list (the default) writes nothing and returns an empty [0, G, 3] tensor, so
// existing callers are unaffected. Like the slot list this is
// architecture-independent glue: every architecture produces it and only the
// sm_100/103 cascade reads it.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> moe_build_routing(
    at::Tensor topk_ids, int64_t num_groups, int64_t m_cap, bool with_slots, std::vector<int64_t> problem_shapes_nk,
    std::optional<at::Tensor> topk_w)
{
    TORCH_CHECK(topk_ids.is_cuda() && topk_ids.dtype() == at::kInt,
        "topk_ids must be CUDA int32");
    TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be [M, topk]");
    TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");
    TORCH_CHECK(num_groups >= 1 && num_groups <= kMaxGroups,
        "num_groups must be in [1, ", kMaxGroups, "]");
    TORCH_CHECK(m_cap >= 1 && m_cap % 4 == 0, "m_cap must be a positive multiple of 4");
    int const M = topk_ids.size(0);
    int const topk = topk_ids.size(1);
    // Caller contract: with no-replacement topk routing a group receives at
    // most M rows, so m_cap >= M guarantees no slot overflow.
    TORCH_CHECK(m_cap >= M, "m_cap must be >= M (per-expert count can reach M)");
    TORCH_CHECK(problem_shapes_nk.size() % 2 == 0,
        "problem_shapes_nk must be a flat list of (N, K) pairs, got ", problem_shapes_nk.size(), " ints");
    int const num_ps = static_cast<int>(problem_shapes_nk.size() / 2);
    TORCH_CHECK(num_ps <= kMaxProblemShapeSets,
        "problem_shapes_nk names ", num_ps, " GEMMs; at most ", kMaxProblemShapeSets, " are supported");
    ProblemShapeList ps{};
    for (int p = 0; p < num_ps; ++p)
    {
        int64_t const n = problem_shapes_nk[2 * p], k = problem_shapes_nk[2 * p + 1];
        TORCH_CHECK(n >= 1 && k >= 1 && n <= INT32_MAX && k <= INT32_MAX,
            "problem_shapes_nk pair ", p, " = (", n, ", ", k, ") must be positive int32");
        ps.n[p] = static_cast<int32_t>(n);
        ps.k[p] = static_cast<int32_t>(k);
    }
    ps.count = num_ps;

    auto opts = topk_ids.options();
    auto masked_m = at::empty({num_groups}, opts);
    auto row_map = at::empty({num_groups * m_cap}, opts);
    auto slot_of_flat = at::empty({static_cast<int64_t>(M) * topk}, opts);
    // One entry per expert, not per active expert: the caller's grid bound is a
    // host-static min(M * topk, G) that it does not have to agree with here,
    // and a [G] list is legal for any bound it chooses.
    auto slot_to_expert = at::empty({with_slots ? num_groups : 0}, opts);
    int32_t* const slot_ptr = with_slots ? reinterpret_cast<int32_t*>(slot_to_expert.data_ptr()) : nullptr;
    auto problem_shapes = at::empty({num_ps, num_groups, 3}, opts);
    int32_t* const ps_ptr = num_ps > 0 ? reinterpret_cast<int32_t*>(problem_shapes.data_ptr()) : nullptr;
    // Per-slot combine weights, asked for by passing topk_w: the fused FC2
    // epilogue adds its rows into the layer output itself and needs the weight of
    // the row it holds, which the pair-indexed topk_w cannot give it. Off by
    // default; the returned tensor is then empty and the builders that do not
    // publish it are the ones they always were.
    bool const want_w = topk_w.has_value() && topk_w->numel() > 0;
    float const* w_in = nullptr;
    if (want_w)
    {
        auto const& tw = topk_w.value();
        TORCH_CHECK(tw.is_cuda() && tw.dtype() == at::kFloat && tw.is_contiguous(),
            "topk_w must be a contiguous CUDA fp32 tensor");
        TORCH_CHECK(tw.numel() == static_cast<int64_t>(M) * topk,
            "topk_w must have M * topk entries, matching topk_ids");
        TORCH_CHECK(num_ps == 0,
            "moe_build_routing: the per-group problem shapes (sm_100/103 pointer-array route) and the per-slot "
            "combine weights (sm_120 fused FC2 epilogue) are never both consumed; ask for one");
        w_in = reinterpret_cast<float const*>(tw.data_ptr());
    }
    auto weight_of_slot = at::empty({want_w ? num_groups * m_cap : 0}, topk_ids.options().dtype(at::kFloat));
    float* const w_ptr = want_w ? reinterpret_cast<float*>(weight_of_slot.data_ptr()) : nullptr;

    auto stream = at::cuda::getCurrentCUDAStream();
    int const n_pairs = M * topk;
    if (moe_glue_routing_multi_enabled() && n_pairs >= kRoutingMultiMinPairs)
    {
        int32_t* scratch = RoutingScratch::instance().acquire(stream);
        if (scratch != nullptr)
        {
            int blocks = (n_pairs + kRoutingMultiPairsPerCta - 1) / kRoutingMultiPairsPerCta;
            if (blocks > kRoutingMultiMaxCtas)
                blocks = kRoutingMultiMaxCtas;
            if (blocks < 1)
                blocks = 1;
            // The pre-existing kernel when no shapes were asked for, so such
            // callers launch exactly the code they launched before.
            if (want_w)
                fso_pdl_launch(moe_build_routing_multi_w_kernel, dim3(blocks), dim3(kRoutingMultiThreads), stream,
                    fso_pdl_enabled(), reinterpret_cast<int32_t const*>(topk_ids.data_ptr()),
                    reinterpret_cast<int32_t*>(masked_m.data_ptr()),
                    reinterpret_cast<int32_t*>(row_map.data_ptr()),
                    reinterpret_cast<int32_t*>(slot_of_flat.data_ptr()), scratch, scratch + kMaxGroups, slot_ptr,
                    w_in, w_ptr, n_pairs, topk, static_cast<int>(num_groups), static_cast<int>(m_cap));
            else if (ps_ptr == nullptr)
                fso_pdl_launch(moe_build_routing_multi_kernel, dim3(blocks), dim3(kRoutingMultiThreads), stream,
                    fso_pdl_enabled(), reinterpret_cast<int32_t const*>(topk_ids.data_ptr()),
                    reinterpret_cast<int32_t*>(masked_m.data_ptr()),
                    reinterpret_cast<int32_t*>(row_map.data_ptr()),
                    reinterpret_cast<int32_t*>(slot_of_flat.data_ptr()), scratch, scratch + kMaxGroups, slot_ptr,
                    n_pairs, topk, static_cast<int>(num_groups), static_cast<int>(m_cap));
            else
                fso_pdl_launch(moe_build_routing_multi_ps_kernel, dim3(blocks), dim3(kRoutingMultiThreads), stream,
                    fso_pdl_enabled(), reinterpret_cast<int32_t const*>(topk_ids.data_ptr()),
                    reinterpret_cast<int32_t*>(masked_m.data_ptr()),
                    reinterpret_cast<int32_t*>(row_map.data_ptr()),
                    reinterpret_cast<int32_t*>(slot_of_flat.data_ptr()), scratch, scratch + kMaxGroups, slot_ptr,
                    ps_ptr, ps, n_pairs, topk, static_cast<int>(num_groups), static_cast<int>(m_cap));
            return {masked_m, row_map, slot_of_flat, slot_to_expert, problem_shapes, weight_of_slot};
        }
    }
    if (want_w)
        fso_pdl_launch(moe_build_routing_w_kernel, dim3(1), dim3(kRoutingThreads), stream, fso_pdl_enabled(),
            reinterpret_cast<int32_t const*>(topk_ids.data_ptr()),
            reinterpret_cast<int32_t*>(masked_m.data_ptr()),
            reinterpret_cast<int32_t*>(row_map.data_ptr()),
            reinterpret_cast<int32_t*>(slot_of_flat.data_ptr()), slot_ptr, w_in, w_ptr,
            n_pairs, topk, static_cast<int>(num_groups), static_cast<int>(m_cap));
    else if (ps_ptr == nullptr)
        fso_pdl_launch(moe_build_routing_kernel, dim3(1), dim3(kRoutingThreads), stream, fso_pdl_enabled(),
            reinterpret_cast<int32_t const*>(topk_ids.data_ptr()),
            reinterpret_cast<int32_t*>(masked_m.data_ptr()),
            reinterpret_cast<int32_t*>(row_map.data_ptr()),
            reinterpret_cast<int32_t*>(slot_of_flat.data_ptr()), slot_ptr,
            n_pairs, topk, static_cast<int>(num_groups), static_cast<int>(m_cap));
    else
        fso_pdl_launch(moe_build_routing_ps_kernel, dim3(1), dim3(kRoutingThreads), stream, fso_pdl_enabled(),
            reinterpret_cast<int32_t const*>(topk_ids.data_ptr()),
            reinterpret_cast<int32_t*>(masked_m.data_ptr()),
            reinterpret_cast<int32_t*>(row_map.data_ptr()),
            reinterpret_cast<int32_t*>(slot_of_flat.data_ptr()), slot_ptr, ps_ptr, ps,
            n_pairs, topk, static_cast<int>(num_groups), static_cast<int>(m_cap));
    return {masked_m, row_map, slot_of_flat, slot_to_expert, problem_shapes, weight_of_slot};
}


// moe_build_sorted: topk_ids [M, topk] int32 -> (sorted_expert_ids [P_max],
// sorted_src [P_max], num_padded_dev [1]), one launch. P_max = M*topk +
// num_groups*(block_m-1) is the worst-case padded length (fixed for capture).
std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_build_sorted(
    at::Tensor topk_ids, int64_t num_groups, int64_t block_m)
{
    TORCH_CHECK(topk_ids.is_cuda() && topk_ids.dtype() == at::kInt,
        "topk_ids must be CUDA int32");
    TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.is_contiguous(),
        "topk_ids must be contiguous [M, topk]");
    TORCH_CHECK(num_groups >= 1 && num_groups <= kMaxGroups,
        "num_groups must be in [1, ", kMaxGroups, "]");
    TORCH_CHECK(block_m >= 1 && (block_m & (block_m - 1)) == 0,
        "block_m must be a positive power of 2 (bitwise-ceil padding)");
    int const M = topk_ids.size(0);
    int const topk = topk_ids.size(1);
    int const R = M * topk;
    // Worst case: every active expert wastes up to block_m-1 rows; at most
    // min(R, num_groups) experts are active.
    int const active_max = R < num_groups ? R : static_cast<int>(num_groups);
    int const bm = static_cast<int>(block_m);
    // Round P_max up to a block_m multiple so every block-start row is in range
    // and num_blocks = P_max/block_m is exact.
    int const p_max = (R + active_max * (bm - 1) + bm - 1) / bm * bm;

    auto opts = topk_ids.options();
    auto sorted_expert_ids = at::empty({p_max}, opts);
    auto flat_to_sorted = at::empty({R}, opts);
    auto num_padded_dev = at::empty({1}, opts);

    auto stream = at::cuda::getCurrentCUDAStream();
    fso_pdl_launch(moe_build_sorted_kernel, dim3(1), dim3(kRoutingThreads), stream, fso_pdl_enabled(),
        reinterpret_cast<int32_t const*>(topk_ids.data_ptr()),
        reinterpret_cast<int32_t*>(sorted_expert_ids.data_ptr()),
        reinterpret_cast<int32_t*>(flat_to_sorted.data_ptr()),
        reinterpret_cast<int32_t*>(num_padded_dev.data_ptr()),
        R, topk, static_cast<int>(num_groups), static_cast<int>(block_m), p_max);
    return {sorted_expert_ids, flat_to_sorted, num_padded_dev};
}


// moe_combine: dn [G, m_cap, H] bf16 + slot_of_flat [M*topk] + topk_w [M,topk]
// fp32 -> out [M, H] bf16 (weighted sum over the topk routed rows).
//
// Three optional arguments let a whole MoE block end in this one kernel.
// `bias` is a per-token row added on top of the weighted expert sum -- a
// model's shared-expert output -- and `bias_scale` a per-token factor on it,
// which is where a sigmoid shared-expert gate goes. Adding them here replaces
// an elementwise pass that reads and writes the whole [M, H] block twice.
// `out` is a caller-supplied destination: a data-parallel or reduce-scatter
// path already owns the buffer the result has to land in, and writing straight
// into it saves the copy.
at::Tensor moe_combine(at::Tensor dn, at::Tensor slot_of_flat, at::Tensor topk_w,
    std::optional<at::Tensor> bias, std::optional<at::Tensor> bias_scale, std::optional<at::Tensor> out_opt)
{
    TORCH_CHECK(dn.is_cuda() && dn.dtype() == at::kBFloat16, "dn must be CUDA bf16");
    TORCH_CHECK(dn.dim() == 3 && dn.is_contiguous(), "dn must be contiguous [G, m_cap, H]");
    TORCH_CHECK(slot_of_flat.dtype() == at::kInt && slot_of_flat.is_contiguous(),
        "slot_of_flat must be contiguous int32");
    TORCH_CHECK(topk_w.dtype() == at::kFloat && topk_w.dim() == 2 && topk_w.is_contiguous(),
        "topk_w must be contiguous fp32 [M, topk]");
    int const M = topk_w.size(0);
    int const topk = topk_w.size(1);
    int const H = dn.size(2);
    TORCH_CHECK(H % 8 == 0, "H must be a multiple of 8 (16-byte vectorised combine)");
    TORCH_CHECK(slot_of_flat.numel() == static_cast<int64_t>(M) * topk,
        "slot_of_flat must have M*topk entries");

    __nv_bfloat16 const* bias_ptr = nullptr;
    if (bias.has_value() && bias->numel() > 0)
    {
        auto const& b = bias.value();
        TORCH_CHECK(b.is_cuda() && b.dtype() == at::kBFloat16 && b.is_contiguous(),
            "bias must be a contiguous CUDA bf16 tensor");
        TORCH_CHECK(b.dim() == 2 && b.size(0) == M && b.size(1) == H, "bias must be [M, H]");
        bias_ptr = reinterpret_cast<__nv_bfloat16 const*>(b.data_ptr());
    }
    float const* bias_scale_ptr = nullptr;
    if (bias_scale.has_value() && bias_scale->numel() > 0)
    {
        auto const& bs = bias_scale.value();
        TORCH_CHECK(bs.is_cuda() && bs.dtype() == at::kFloat && bs.is_contiguous(),
            "bias_scale must be a contiguous CUDA fp32 tensor");
        TORCH_CHECK(bs.numel() == M, "bias_scale must have one entry per token");
        TORCH_CHECK(bias_ptr != nullptr, "bias_scale needs a bias to scale");
        bias_scale_ptr = reinterpret_cast<float const*>(bs.data_ptr());
    }
    at::Tensor out;
    if (out_opt.has_value() && out_opt->numel() > 0)
    {
        out = out_opt.value();
        TORCH_CHECK(out.is_cuda() && out.dtype() == at::kBFloat16 && out.is_contiguous(),
            "out must be a contiguous CUDA bf16 tensor");
        TORCH_CHECK(out.dim() == 2 && out.size(0) == M && out.size(1) == H, "out must be [M, H]");
    }
    else
    {
        out = at::empty({M, H}, dn.options());
    }
    int64_t const total = static_cast<int64_t>(M) * (H / 8);
    // Block size: 128 threads on sm_120 while the grid is small, 256 otherwise.
    // Under PDL the combine starts early only if one of its CTAs fits on an SM
    // beside the FC2's resident CTAs. The Family B decode FC2, the 2-CTA
    // (16,128,2) grouped instance, has used 69 registers (72 allocated) since the
    // prologue round: two of its 384-thread CTAs hold 55,296 of the SM's 65,536
    // registers, and the 10,240 left take a 128-thread combine CTA at 64 registers
    // (8,192) but not a 256-thread one (16,384). Run sm120_r2_p1_20260929 measured
    // this launch: the combine enters at the FC2's release again, its critical-path
    // increment falls 0.4-1.4 us at Family B M = 4-128, and the layer is 0.1-0.8 %
    // faster at Family B M = 1-128 and up to 1.0 % at Family C, bit-identically
    // (each thread reduces its own 8 columns over top-k in the same order). Grids
    // above 256 CTAs of 128 threads (M > 128 at H = 2048, the prefill band) keep
    // the 256-thread launch, and so do the other arches, whose FC2 kernels have
    // other register footprints and were not measured with this launch.
    constexpr int kSmallGridThreads = 128;
    constexpr int64_t kSmallGridMaxCtas = 256;
    bool const small_grid = at::cuda::getCurrentDeviceProperties()->major == 12
        && total <= kSmallGridMaxCtas * kSmallGridThreads;
    int const threads = small_grid ? kSmallGridThreads : 256;
    int const grid = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    bool const pdl = fso_pdl_enabled() && static_cast<unsigned>(grid) <= kFsoPdlMaxGridCtas;
    if (bias_ptr != nullptr)
    {
        fso_pdl_launch(moe_combine_kernel<true>, dim3(grid), dim3(threads), stream, pdl,
            reinterpret_cast<__nv_bfloat16 const*>(dn.data_ptr()),
            reinterpret_cast<int32_t const*>(slot_of_flat.data_ptr()),
            reinterpret_cast<float const*>(topk_w.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), bias_ptr, bias_scale_ptr, M, topk, H);
    }
    else
    {
        fso_pdl_launch(moe_combine_kernel<false>, dim3(grid), dim3(threads), stream, pdl,
            reinterpret_cast<__nv_bfloat16 const*>(dn.data_ptr()),
            reinterpret_cast<int32_t const*>(slot_of_flat.data_ptr()),
            reinterpret_cast<float const*>(topk_w.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), nullptr, nullptr, M, topk, H);
    }
    return out;
}


// moe_combine_sorted: contiguous-layout combine. dn [P_max, H] bf16 (the
// GroupedContiguous GEMM output in expert-sorted rows) + flat_to_sorted
// [M*topk] (pair -> sorted row) + topk_w [M, topk] -> out [M, H]. Reuses
// moe_combine_kernel — dn is already indexed flat as dn[slot*H + h], and the
// sorted row index from flat_to_sorted plays the role of the masked slot.
at::Tensor moe_combine_sorted(at::Tensor dn, at::Tensor flat_to_sorted, at::Tensor topk_w)
{
    TORCH_CHECK(dn.is_cuda() && dn.dtype() == at::kBFloat16, "dn must be CUDA bf16");
    TORCH_CHECK(dn.dim() == 2 && dn.is_contiguous(), "dn must be contiguous [P_max, H]");
    TORCH_CHECK(flat_to_sorted.dtype() == at::kInt && flat_to_sorted.is_contiguous(),
        "flat_to_sorted must be contiguous int32");
    TORCH_CHECK(topk_w.dtype() == at::kFloat && topk_w.dim() == 2 && topk_w.is_contiguous(),
        "topk_w must be contiguous fp32 [M, topk]");
    int const M = topk_w.size(0);
    int const topk = topk_w.size(1);
    int const H = dn.size(1);
    TORCH_CHECK(H % 8 == 0, "H must be a multiple of 8 (16-byte vectorised combine)");
    TORCH_CHECK(flat_to_sorted.numel() == static_cast<int64_t>(M) * topk,
        "flat_to_sorted must have M*topk entries");

    auto out = at::empty({M, H}, dn.options());
    int const threads = 256;
    int64_t const total = static_cast<int64_t>(M) * (H / 8);
    int const grid = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    bool const pdl = fso_pdl_enabled() && static_cast<unsigned>(grid) <= kFsoPdlMaxGridCtas;
    fso_pdl_launch(moe_combine_kernel<false>, dim3(grid), dim3(threads), stream, pdl,
        reinterpret_cast<__nv_bfloat16 const*>(dn.data_ptr()),
        reinterpret_cast<int32_t const*>(flat_to_sorted.data_ptr()),
        reinterpret_cast<float const*>(topk_w.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), nullptr, nullptr, M, topk, H);
    return out;
}

// moe_topk_from_logits: router logits -> (topk_ids, topk_w [, shared_gate]),
// one launch. See the kernel above for what it replaces and why the GEMM that
// produces the logits stays with the caller.
//
// `logits` is [M, E] bf16 / fp16 / fp32, or [M, E+1] with `with_shared_gate`,
// where column E is the shared expert's gate logit and the returned third
// tensor is its sigmoid. Only the first E columns are selected over, and E is
// taken from the tensor's width, so the sentinel this kernel writes into padded
// rows is the caller's own `num_experts` -- the value sglang's
// `_mask_topk_ids_padded_region` uses.
//
// `num_token_non_padded` is the device int32 [1] a graph bucket carries: the
// rows at or past it get the sentinel id and a zero weight, which the grouped
// layer skips at no expert cost. `expert_map` is the int32 [E+1] global-to-local
// expert table an expert-parallel rank builds once (its own experts mapped to
// [0, E_local), every other expert and the sentinel to a value outside that
// range); applying it here saves the gather launch the dispatcher would run.
std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_topk_from_logits(at::Tensor logits, int64_t topk,
    bool renormalize, bool with_shared_gate, std::optional<at::Tensor> num_token_non_padded,
    std::optional<at::Tensor> expert_map)
{
    TORCH_CHECK(logits.is_cuda() && logits.dim() == 2, "logits must be a CUDA [M, E] tensor");
    TORCH_CHECK(logits.stride(1) == 1, "logits must have a contiguous expert dimension");
    TORCH_CHECK(logits.dtype() == at::kBFloat16 || logits.dtype() == at::kHalf || logits.dtype() == at::kFloat,
        "logits must be bf16, fp16 or fp32");
    int const M = static_cast<int>(logits.size(0));
    int const width = static_cast<int>(logits.size(1));
    int const E = width - (with_shared_gate ? 1 : 0);
    TORCH_CHECK(E >= 1 && E <= kMaxGroups, "num_experts must be in [1, ", kMaxGroups, "], got ", E);
    TORCH_CHECK(topk >= 1 && topk <= kRouterMaxTopk && topk <= E, "topk must be in [1, min(",
        kRouterMaxTopk, ", num_experts)], got ", topk);
    int32_t const* n_valid = nullptr;
    if (num_token_non_padded.has_value() && num_token_non_padded->numel() > 0)
    {
        auto const& nv = num_token_non_padded.value();
        TORCH_CHECK(nv.is_cuda() && nv.dtype() == at::kInt && nv.numel() == 1,
            "num_token_non_padded must be a CUDA int32 tensor with one element");
        n_valid = reinterpret_cast<int32_t const*>(nv.data_ptr());
    }
    int32_t const* map_ptr = nullptr;
    if (expert_map.has_value() && expert_map->numel() > 0)
    {
        auto const& em = expert_map.value();
        TORCH_CHECK(em.is_cuda() && em.dtype() == at::kInt && em.is_contiguous(),
            "expert_map must be a contiguous CUDA int32 tensor");
        TORCH_CHECK(em.numel() >= static_cast<int64_t>(E) + 1,
            "expert_map must have num_experts + 1 entries (the last one maps the padded-row sentinel)");
        map_ptr = reinterpret_cast<int32_t const*>(em.data_ptr());
    }

    auto ids = at::empty({M, topk}, logits.options().dtype(at::kInt));
    auto weights = at::empty({M, topk}, logits.options().dtype(at::kFloat));
    auto shared_gate = at::empty({with_shared_gate ? M : 0}, logits.options().dtype(at::kFloat));
    if (M == 0)
        return {ids, weights, shared_gate};
    float* gate_ptr = with_shared_gate ? reinterpret_cast<float*>(shared_gate.data_ptr()) : nullptr;
    int const ld = static_cast<int>(logits.stride(0));
    int const tokens_per_cta = kRouterThreads / 32;
    int const grid = (M + tokens_per_cta - 1) / tokens_per_cta;
    auto stream = at::cuda::getCurrentCUDAStream();
    bool const pdl = fso_pdl_enabled() && static_cast<unsigned>(grid) <= kFsoPdlMaxGridCtas;

#define FSO_ROUTER_LAUNCH(TLOGIT, SLOTS, PTR)                                                                          \
    fso_pdl_launch(moe_topk_from_logits_kernel<TLOGIT, SLOTS>, dim3(grid), dim3(kRouterThreads), stream, pdl, PTR,      \
        n_valid, map_ptr, reinterpret_cast<int32_t*>(ids.data_ptr()), reinterpret_cast<float*>(weights.data_ptr()),     \
        gate_ptr, M, E, ld, static_cast<int>(topk), E, renormalize)

    if (logits.dtype() == at::kBFloat16)
    {
        auto const* p = reinterpret_cast<__nv_bfloat16 const*>(logits.data_ptr());
        if (E <= 256)
            FSO_ROUTER_LAUNCH(__nv_bfloat16, kRouterSlotsSmall, p);
        else
            FSO_ROUTER_LAUNCH(__nv_bfloat16, kRouterSlotsLarge, p);
    }
    else if (logits.dtype() == at::kHalf)
    {
        auto const* p = reinterpret_cast<__half const*>(logits.data_ptr());
        if (E <= 256)
            FSO_ROUTER_LAUNCH(__half, kRouterSlotsSmall, p);
        else
            FSO_ROUTER_LAUNCH(__half, kRouterSlotsLarge, p);
    }
    else
    {
        auto const* p = reinterpret_cast<float const*>(logits.data_ptr());
        if (E <= 256)
            FSO_ROUTER_LAUNCH(float, kRouterSlotsSmall, p);
        else
            FSO_ROUTER_LAUNCH(float, kRouterSlotsLarge, p);
    }
#undef FSO_ROUTER_LAUNCH
    return {ids, weights, shared_gate};
}

} // namespace blockscale_gemm
