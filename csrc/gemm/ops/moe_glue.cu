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
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include <cuda_bf16.h>

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
template <bool kShapes>
__device__ __forceinline__ void moe_build_routing_body(
    int32_t const* __restrict__ topk_ids, // [M * topk]
    int32_t* __restrict__ masked_m,       // [G]
    int32_t* __restrict__ row_map,        // [G * m_cap] (valid slots only)
    int32_t* __restrict__ slot_of_flat,   // [M * topk]
    int32_t* __restrict__ slot_to_expert, // [G] or nullptr (see fso_emit_slot_list)
    int32_t* __restrict__ problem_shapes, // [P * G * 3] (see fso_emit_problem_shapes), read iff kShapes
    ProblemShapeList const& ps,
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
    __shared__ int32_t cnt[kMaxGroups];
    for (int g = threadIdx.x; g < num_groups; g += blockDim.x)
        cnt[g] = 0;
    if (slot_to_expert != nullptr)
    {
        for (int s = threadIdx.x; s < num_groups; s += blockDim.x)
            slot_to_expert[s] = -1;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_pairs; i += blockDim.x)
    {
        int const e = topk_ids[i];
        int const r = atomicAdd(&cnt[e], 1);
        int const slot = e * m_cap + r;
        row_map[slot] = i / topk; // source token row
        slot_of_flat[i] = slot;
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
    moe_build_routing_body<false>(topk_ids, masked_m, row_map, slot_of_flat, slot_to_expert, nullptr,
        ProblemShapeList{}, num_pairs, topk, num_groups, m_cap, pdl);
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
    moe_build_routing_body<true>(topk_ids, masked_m, row_map, slot_of_flat, slot_to_expert, problem_shapes, ps,
        num_pairs, topk, num_groups, m_cap, pdl);
}

__global__ void moe_combine_kernel(
    __nv_bfloat16 const* __restrict__ dn, // [G * m_cap, H]
    int32_t const* __restrict__ slot_of_flat, // [M * topk]
    float const* __restrict__ topk_w,     // [M, topk]
    __nv_bfloat16* __restrict__ out,      // [M, H]
    int M, int topk, int H, bool pdl)
{
    // PDL entry (see moe_build_routing_kernel).
    if (pdl)
    {
        cudaGridDependencySynchronize();
        if (threadIdx.x == 0)
        {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }
    constexpr int kVec = 8; // 8 bf16 = 16 bytes per thread
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    int const num_hvec = H / kVec;
    if (tid >= M * num_hvec)
        return;
    int const t = tid / num_hvec;
    int const h = (tid % num_hvec) * kVec;

    float acc[kVec];
#pragma unroll
    for (int v = 0; v < kVec; ++v)
        acc[v] = 0.f;

    // Prefetch all topk routed rows before reducing so the scattered LDG.128
    // latencies overlap. At M=1 this kernel is a single latency-bound CTA and
    // the old load->use-per-iteration chain serialised topk dependent loads.
    // Fast path unrolls a fixed bound (covers the common MoE topk); larger topk
    // falls back to the sequential loop.
    constexpr int kMaxTopk = 8;
    if (topk <= kMaxTopk)
    {
        float4 raws[kMaxTopk];
        float ws[kMaxTopk];
#pragma unroll
        for (int j = 0; j < kMaxTopk; ++j)
        {
            if (j < topk)
            {
                int const slot = slot_of_flat[t * topk + j];
                ws[j] = topk_w[t * topk + j];
                raws[j] = *reinterpret_cast<float4 const*>(&dn[static_cast<int64_t>(slot) * H + h]);
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
            float const w = topk_w[t * topk + j];
            float4 const raw = *reinterpret_cast<float4 const*>(
                &dn[static_cast<int64_t>(slot) * H + h]);
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
template <bool kShapes>
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

    __shared__ int32_t cnt[kMaxGroups];
    __shared__ int32_t base[kMaxGroups];
    __shared__ int32_t fill[kMaxGroups];
    __shared__ int32_t s_last;

    for (int g = threadIdx.x; g < num_groups; g += blockDim.x)
    {
        cnt[g] = 0;
        fill[g] = 0;
    }
    __syncthreads();

    int const stride = gridDim.x * blockDim.x;
    int const start = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start; i < num_pairs; i += stride)
        atomicAdd(&cnt[topk_ids[i]], 1);
    __syncthreads();

    for (int g = threadIdx.x; g < num_groups; g += blockDim.x)
        base[g] = (cnt[g] > 0) ? atomicAdd(&gcnt[g], cnt[g]) : 0;
    __syncthreads();

    for (int i = start; i < num_pairs; i += stride)
    {
        int const e = topk_ids[i];
        int const slot = e * m_cap + base[e] + atomicAdd(&fill[e], 1);
        row_map[slot] = i / topk;
        slot_of_flat[i] = slot;
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
    moe_build_routing_multi_body<false>(topk_ids, masked_m, row_map, slot_of_flat, gcnt, gdone, slot_to_expert,
        nullptr, ProblemShapeList{}, num_pairs, topk, num_groups, m_cap, pdl);
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
    moe_build_routing_multi_body<true>(topk_ids, masked_m, row_map, slot_of_flat, gcnt, gdone, slot_to_expert,
        problem_shapes, ps, num_pairs, topk, num_groups, m_cap, pdl);
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
    __shared__ int32_t cnt[kMaxGroups];  // per-expert routed count
    __shared__ int32_t off[kMaxGroups];  // exclusive prefix of padded_e
    __shared__ int32_t fill[kMaxGroups]; // per-expert scatter cursor
    __shared__ int32_t s_p_actual;

    // Phase 1: histogram routed pairs per expert.
    for (int g = threadIdx.x; g < num_groups; g += blockDim.x)
        cnt[g] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < num_pairs; i += blockDim.x)
        atomicAdd(&cnt[topk_ids[i]], 1);
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
        int const e = topk_ids[i];
        int const r = atomicAdd(&fill[e], 1);
        flat_to_sorted[i] = off[e] + r;
    }
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
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> moe_build_routing(
    at::Tensor topk_ids, int64_t num_groups, int64_t m_cap, bool with_slots, std::vector<int64_t> problem_shapes_nk)
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

    auto stream = at::cuda::getCurrentCUDAStream();
    int const n_pairs = M * topk;
    if (moe_glue_is_sm100_family() && n_pairs >= kRoutingMultiMinPairs)
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
            if (ps_ptr == nullptr)
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
            return {masked_m, row_map, slot_of_flat, slot_to_expert, problem_shapes};
        }
    }
    if (ps_ptr == nullptr)
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
    return {masked_m, row_map, slot_of_flat, slot_to_expert, problem_shapes};
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
at::Tensor moe_combine(at::Tensor dn, at::Tensor slot_of_flat, at::Tensor topk_w)
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

    auto out = at::empty({M, H}, dn.options());
    int const threads = 256;
    int64_t const total = static_cast<int64_t>(M) * (H / 8);
    int const grid = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    bool const pdl = fso_pdl_enabled() && static_cast<unsigned>(grid) <= kFsoPdlMaxGridCtas;
    fso_pdl_launch(moe_combine_kernel, dim3(grid), dim3(threads), stream, pdl,
        reinterpret_cast<__nv_bfloat16 const*>(dn.data_ptr()),
        reinterpret_cast<int32_t const*>(slot_of_flat.data_ptr()),
        reinterpret_cast<float const*>(topk_w.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        M, topk, H);
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
    fso_pdl_launch(moe_combine_kernel, dim3(grid), dim3(threads), stream, pdl,
        reinterpret_cast<__nv_bfloat16 const*>(dn.data_ptr()),
        reinterpret_cast<int32_t const*>(flat_to_sorted.data_ptr()),
        reinterpret_cast<float const*>(topk_w.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        M, topk, H);
    return out;
}

} // namespace blockscale_gemm
