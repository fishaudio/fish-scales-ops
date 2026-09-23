/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SM100/SM103 grouped (MoE, masked-layout) MXFP8 dispatch and kernel launcher.
//
// Contract implemented here (the sm_120 masked-slab contract, reproduced on
// the datacenter Blackwell parts):
//
//   A   [G, m_cap, K]   fp8 e4m3, rows >= masked_m[g] are undefined
//   B   [G, N, K]       fp8 e4m3, per-expert weights
//   SFA [G, slab_a]     int32, per group the CUTLASS Sm1xx atom layout for a
//                       (pad(m_cap,128), K) tensor
//   SFB [G, slab_b]     int32, per group the atom layout for an (N, K) tensor
//   D   [G, m_cap, N]   bf16, rows >= masked_m[g] are undefined
//   masked_m [G]        int32 ON DEVICE, read only by kernels
//   expected_m          host int, tile selection only
//
// How the masked semantics are expressed to CUTLASS
// -------------------------------------------------
// CUTLASS's grouped GEMM takes a device array of per-group problem shapes, a
// device array of per-group base pointers, a device array of per-group strides
// and (for block-scaled kernels) a device array of per-group scale-factor
// layouts. The tile scheduler reads the problem shapes from device memory to
// decide how many tiles each group contributes, and the mainloop rebuilds its
// TMA descriptors per group from the pointer / stride / layout arrays. None of
// that is baked into the host-side Params.
//
// So the whole masked behaviour reduces to: set group g's problem shape to
// (masked_m[g], N, K). A group with zero valid rows then contributes zero
// tiles and is skipped; a group with fewer valid rows than the 128-row tile
// granularity gets exactly one tile whose out-of-range rows are handled by the
// TMA bounds (loads read zero, stores are dropped). `sm100_grouped_prep_kernel`
// below builds all those arrays on device from `masked_m` in a single tiny
// launch, which is what makes the call CUDA-Graph capture-safe: the arrays are
// rebuilt on every replay, so one capture serves changing routing.
//
// Why a device-built scale-factor layout and not one shared layout: the
// activation scale slab is written for pad(m_cap,128) rows, and the atom
// layout's strides depend only on K, so a layout built for masked_m[g] rows
// addresses exactly the same words as one built for m_cap rows. Building it
// from masked_m[g] keeps the scale-factor TMA descriptor's row extent equal to
// the data descriptor's, so a tile's padding rows read zero data *and* zero
// scale bytes instead of a live scale byte against zeroed data.
//
// Host-overhead discipline (same as the dense sm_100 launcher):
//   * A thread-local cache of initialized Params keyed by (slot, groups,
//     scheduler knobs). For grouped mode CUTLASS builds its initial TMA
//     descriptors from the *tile* shape, not the problem shape, so one Params
//     serves every (m_cap, N, K) that shares an argument slot.
//   * cudaFuncSetAttribute behind a per-instantiation static guard, so it
//     fires on the first (eager) call only and never during capture.
//   * The CUTLASS workspace (per-SM tensormap scratch) and the argument arrays
//     come from thread-local pools that only ever allocate outside capture; a
//     capture-time allocation aborts with a message telling the caller to warm
//     up eagerly first.
//
// Routing-supplied problem shapes: the launch without a prep kernel
// -----------------------------------------------------------------------
// Of the eleven per-group arrays only the problem shapes depend on the routing
// (see the note on `sm100_grouped_prep_kernel` for why `layout_sfa` does not
// have to). `moe_build_routing` already holds every group's row count when it
// publishes `masked_m`, so it can write the `(rows, N, K)` triples for each
// GEMM of the layer in the same pass, and a caller that hands that tensor in
// as `problem_shapes` lets the launcher skip the prep launch altogether (run
// b300_round3_20260922/M-A4; the two prep launches were the whole of the
// per-layer kernel-time budget item L4 of run b300_remeasure_20260922). The
// ten remaining arrays are then a pure function of the tensor set -- the five
// base addresses, the per-group byte strides, m_cap, N and K -- and are built
// ONCE per distinct tensor set into a key-addressed block of `StaticArena`
// below, and reused by every later call that presents the same key. Two
// consequences a caller can observe, both explained at `StaticArena`: a block
// bound while a stream is capturing is written OUTSIDE the capture (a side
// stream plus a host wait, under the relaxed capture mode) so the captured
// graph carries no writer of its arrays, and such a block is pinned for the
// life of the process because the graph's kernel node keeps reading it.
//
// A second route, not a tile of this one, lives in
// `grouped_slot_dispatch.cuh`: the slot-bound swap-orientation kernel for the
// decode band, selected by `grouped_detail::slot_route` below and switched with
// FSO_GROUPED_SLOT (0 never, unset/1 the rule, `force` whenever legal).
//
// Env overrides: FSO_FORCE_TILE=TM,TN,ST reuses the shared wire format. On the
// sm_100 grouped path the third field selects the (SM-count, cluster, TileK,
// epilogue) variant:
//   ST=1 -> 1SM cluster(1,1) K128 TMA epilogue
//   ST=2 -> 2SM cluster(2,1) K128 TMA epilogue
//   ST=3 -> 1SM cluster(1,1) K256 TMA epilogue
//   ST=4 -> 2SM cluster(2,1) K256 TMA epilogue
//   ST=5 -> 1SM cluster(1,1) K128 direct-store (NoSmem) epilogue
//   ST=6 -> 2SM cluster(2,1) K128 direct-store (NoSmem) epilogue
//   ST=7 -> 1SM cluster(1,1) K256 direct-store (NoSmem) epilogue
//   ST=8 -> 2SM cluster(2,1) K256 direct-store (NoSmem) epilogue
//   ST=9 -> 1SM cluster(1,1) K128 direct store, EVT-partitioned epilogue tile
//   ST=11-> 1SM cluster(2,1) K128 TMA epilogue (weight tile multicast to the
//           two CTAs of the cluster, which hold neighbouring M tiles)
// ST 9 and ST 11 were added in run b300_mxfp8_20260917/M-A1 and are wired at
// TN 128/192/256 (ST 9) and TN 192/256 (ST 11). ST=10 is deliberately unused:
// the 2SM form of ST 9 would need the 2Sm NoSmem epilogue class, which the
// direct-store fragment this variant exists to fix is not written against.
//
// The TN field carries the N-tile width. Until 2026-09-17 only 128 and 256
// were instantiated; the narrow widths the block-scaled builder also accepts
// are now wired on the two variants the cascade itself uses (1SM, TileK 128,
// TMA store = ST 1 and direct store = ST 5), which adds these four force
// codes:
//   128,64,1    1SM cluster(1,1) K128 TMA epilogue,          64-wide N tile
//   128,192,1   1SM cluster(1,1) K128 TMA epilogue,         192-wide N tile
//   128,64,5    1SM cluster(1,1) K128 direct-store epilogue,  64-wide N tile
//   128,192,5   1SM cluster(1,1) K128 direct-store epilogue, 192-wide N tile
// The 2SM and TileK 256 variants are deliberately not extended to the narrow
// widths: both lost everywhere in the 2026-09-15 sweep, so a narrow-N version
// of them would only enlarge the binary. Run b300_mxfp8_20260917/M-A1 re-swept
// all twenty pre-existing codes on both Qwen3 MoE families at M = 64 to 4096
// and reproduced that verdict at every cell, so the omission stands.

#pragma once

#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#include "blockscale_gemm/arch/sm100/mxfp8/grouped_gemm_types.cuh"
#include "blockscale_gemm/arch/sm120/common/env_overrides.cuh" // FSO_FORCE_TILE parsing (arch-agnostic)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace sm100_blockscaled_gemm
{

namespace grouped_detail
{

// Every tile instantiation shares the same argument-array element types: the
// strides and the scale-factor layouts depend only on the operand layout tags
// and on SFVecSize, not on the tile shape. We therefore build the arrays once,
// with the types taken from a reference config, and static_assert in each
// launcher that its own types match.
using RefConfig = Sm100MxFP8GroupedGemmConfig<128, 128, 1, 1, 128, false>;
using GProblemShape = cute::Shape<int, int, int>;
using GStrideA = typename RefConfig::InternalStrideA;
using GStrideB = typename RefConfig::InternalStrideB;
using GStrideD = typename RefConfig::InternalStrideD;
using GLayoutSFA = typename RefConfig::InternalLayoutSFA;
using GLayoutSFB = typename RefConfig::InternalLayoutSFB;
using GSfConfig = typename RefConfig::Sm1xxBlkScaledConfig;

// Upper bound on experts; matches moe_glue.cu's kMaxGroups so the argument
// pool can be sized once and never reallocated (a reallocation would strand
// the cached Params, which hold the array pointers).
constexpr int kMaxGroups = 1024;

// FSO_PRINT_TILE_INFO: print, once per kernel instantiation, the mainloop stage
// count StageCountAutoCarveout derived and the shared memory the kernel asks
// for. The dense path has had this since the sm_100 port; the grouped path had
// not, which is why the picked tile could only be read out of an nsys trace's
// mangled name. Host-side, read once, and off unless the variable is set.
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

// Builds every per-group argument array from `masked_m`. One thread per group.
//
// Runs immediately before the GEMM on the same stream, so a captured graph
// re-executes it on every replay and the GEMM always sees the routing that is
// live at replay time.
//
// Split around the grid-dependency barrier (T5, 2026-09-17)
// ---------------------------------------------------------
// Only two of the eleven arrays depend on the routing: `problem[g]`, which
// carries masked_m[g] as the group's row count, and `layout_sfa[g]`, whose row
// extent is derived from it. The five base pointers, the three strides and
// `layout_sfb[g]` are functions of the tensor set alone (base address, m_cap,
// N, K); `tile_atom_to_shape_SFB` does not even look at the M field of the
// problem shape it is handed (cutlass/detail/sm100_blockscaled_layout.hpp
// destructures the shape and uses only N, K and L).
//
// The kernel therefore does all of that work FIRST, then executes the
// programmatic-dependent-launch barrier, and only afterwards loads masked_m
// and writes the two routing-dependent arrays. When the caller launches this
// kernel with `cudaLaunchAttributeProgrammaticStreamSerialization` (see
// launch_prep_kernel below) the blocks are scheduled as soon as the preceding
// kernel signals `cudaTriggerProgrammaticLaunchCompletion`, so the whole static
// half executes while that kernel is still running and only the short dynamic
// tail is left on the critical path.
//
// `cudaGridDependencySynchronize()` is a full wait on the preceding grid's
// completion, not merely on its trigger. That is what makes the split safe
// whatever precedes the call, including a caller whose immediately preceding
// kernel is the producer of `masked_m`. (CUTLASS depends on the same property:
// its sm_100 pointer-array kernel calls `launch_dependent_grids()` before its
// epilogue stores have landed and documents the placement as affecting
// "performance, not functional correctness".)
//
// One further constant: `tile_atom_to_shape_SFA` tiles a 128-row scale-factor
// atom over the row extent, so its result depends on masked_m[g] only through
// ceil_div(m, 128). Whenever the row capacity is at most 128 rows — every
// Family B / Family C cell up to M = 128 — every legal masked_m[g] and the
// m = 0 fallback round to the same single-tile layout, so SFA is static too
// and the dynamic tail is one load plus the three-int problem shape.
__global__ void sm100_grouped_prep_kernel(GProblemShape* __restrict__ problem, void const** __restrict__ ptr_a,
    void const** __restrict__ ptr_b, void const** __restrict__ ptr_sfa, void const** __restrict__ ptr_sfb,
    void** __restrict__ ptr_d, GStrideA* __restrict__ stride_a, GStrideB* __restrict__ stride_b,
    GStrideD* __restrict__ stride_d, GLayoutSFA* __restrict__ layout_sfa, GLayoutSFB* __restrict__ layout_sfb,
    int32_t const* __restrict__ masked_m, int groups, int m_cap, int shape_n, int shape_k, char* a_base, char* b_base,
    char* sfa_base, char* sfb_base, char* d_base, int64_t a_bytes, int64_t b_bytes, int64_t sfa_bytes,
    int64_t sfb_bytes, int64_t d_bytes, bool pdl)
{
    int const g = blockIdx.x * blockDim.x + threadIdx.x;
    bool const live = (g < groups);

    // A group with no rows is never scheduled, so its layout is never read;
    // still give it a legal (non-zero) row extent so no descriptor can ever be
    // built from a zero dimension. 128 is that fallback, and it is also the
    // row extent every m in [1, 128] rounds up to.
    bool const sfa_is_static = (m_cap <= 128);

    // ---- routing-independent half (executes before the barrier) ----------
    if (live)
    {
        ptr_a[g] = a_base + static_cast<int64_t>(g) * a_bytes;
        ptr_b[g] = b_base + static_cast<int64_t>(g) * b_bytes;
        ptr_sfa[g] = sfa_base + static_cast<int64_t>(g) * sfa_bytes;
        ptr_sfb[g] = sfb_base + static_cast<int64_t>(g) * sfb_bytes;
        ptr_d[g] = d_base + static_cast<int64_t>(g) * d_bytes;

        stride_a[g] = cute::make_stride(static_cast<int64_t>(shape_k), cute::Int<1>{}, cute::Int<0>{});
        stride_b[g] = cute::make_stride(static_cast<int64_t>(shape_k), cute::Int<1>{}, cute::Int<0>{});
        stride_d[g] = cute::make_stride(static_cast<int64_t>(shape_n), cute::Int<1>{}, cute::Int<0>{});

        layout_sfb[g] = GSfConfig::tile_atom_to_shape_SFB(cute::make_shape(128, shape_n, shape_k, 1));
        if (sfa_is_static)
            layout_sfa[g] = GSfConfig::tile_atom_to_shape_SFA(cute::make_shape(128, shape_n, shape_k, 1));
    }

    // ---- barrier on the preceding grid -----------------------------------
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (pdl)
        cudaGridDependencySynchronize();
#endif

    // ---- routing-dependent half ------------------------------------------
    if (live)
    {
        int m = masked_m[g];
        m = m < 0 ? 0 : (m > m_cap ? m_cap : m);
        problem[g] = cute::make_shape(m, shape_n, shape_k);
        if (!sfa_is_static)
        {
            int const m_desc = m > 0 ? m : 128;
            layout_sfa[g] = GSfConfig::tile_atom_to_shape_SFA(cute::make_shape(m_desc, shape_n, shape_k, 1));
        }
    }

    // Release the consumer (the grouped GEMM, launched with PDL when this
    // translation unit compiled CUTLASS's GDC instructions) as soon as this
    // block's stores are issued.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (pdl && threadIdx.x == 0)
        cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// The routing-independent arrays on their own, for the launch that takes its
// problem shapes from `moe_build_routing` (run b300_round3_20260922/M-A4).
//
// Same ten arrays, same values as the first half of the prep kernel, with one
// deliberate difference: `layout_sfa` is derived from `m_cap` at every row
// capacity, not from `masked_m[g]` above 128 rows. That is safe for the
// reason the prep kernel's note gives for the sub-128 case, extended: the
// scale-factor descriptor is read one 128-row atom per M tile, and the grouped
// scheduler only ever issues the ceil(masked_m[g] / 128) tiles the problem
// shape allows, so every tile that runs finds its atom in bounds under either
// row extent and loads the same bytes. The row extent past the last live tile
// is never addressed. The launch is plain (no programmatic attribute): it
// happens only on the eager path, immediately before the GEMM, and the GEMM's
// own `griddepcontrol.wait` then orders it.
__global__ void sm100_grouped_static_kernel(void const** __restrict__ ptr_a, void const** __restrict__ ptr_b,
    void const** __restrict__ ptr_sfa, void const** __restrict__ ptr_sfb, void** __restrict__ ptr_d,
    GStrideA* __restrict__ stride_a, GStrideB* __restrict__ stride_b, GStrideD* __restrict__ stride_d,
    GLayoutSFA* __restrict__ layout_sfa, GLayoutSFB* __restrict__ layout_sfb, int groups, int m_cap, int shape_n,
    int shape_k, char* a_base, char* b_base, char* sfa_base, char* sfb_base, char* d_base, int64_t a_bytes,
    int64_t b_bytes, int64_t sfa_bytes, int64_t sfb_bytes, int64_t d_bytes)
{
    int const g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups)
        return;
    ptr_a[g] = a_base + static_cast<int64_t>(g) * a_bytes;
    ptr_b[g] = b_base + static_cast<int64_t>(g) * b_bytes;
    ptr_sfa[g] = sfa_base + static_cast<int64_t>(g) * sfa_bytes;
    ptr_sfb[g] = sfb_base + static_cast<int64_t>(g) * sfb_bytes;
    ptr_d[g] = d_base + static_cast<int64_t>(g) * d_bytes;
    stride_a[g] = cute::make_stride(static_cast<int64_t>(shape_k), cute::Int<1>{}, cute::Int<0>{});
    stride_b[g] = cute::make_stride(static_cast<int64_t>(shape_k), cute::Int<1>{}, cute::Int<0>{});
    stride_d[g] = cute::make_stride(static_cast<int64_t>(shape_n), cute::Int<1>{}, cute::Int<0>{});
    layout_sfb[g] = GSfConfig::tile_atom_to_shape_SFB(cute::make_shape(128, shape_n, shape_k, 1));
    // 128 is the row extent every m_cap in [1, 128] rounds up to (and the
    // prep kernel's fallback), so the two launches agree exactly there.
    int const m_desc = m_cap < 128 ? 128 : m_cap;
    layout_sfa[g] = GSfConfig::tile_atom_to_shape_SFA(cute::make_shape(m_desc, shape_n, shape_k, 1));
}

// Device-side argument arrays for the grouped launch.
//
// Two slots, used alternately: consecutive grouped GEMMs in one MoE layer
// (gate_up then down) are ordered by the stream, so one slot would already be
// correct, but alternating removes any dependence on that ordering and keeps
// each captured graph node pointing at arrays no other node writes.
struct ArgSlot
{
    GProblemShape* problem = nullptr;
    void const** ptr_a = nullptr;
    void const** ptr_b = nullptr;
    void const** ptr_sfa = nullptr;
    void const** ptr_sfb = nullptr;
    void** ptr_d = nullptr;
    GStrideA* stride_a = nullptr;
    GStrideB* stride_b = nullptr;
    GStrideD* stride_d = nullptr;
    GLayoutSFA* layout_sfa = nullptr;
    GLayoutSFB* layout_sfb = nullptr;
};

struct ArgPool
{
    static constexpr int kSlots = 2;
    void* base = nullptr;
    ArgSlot slots[kSlots];
    int next = 0;

    static ArgPool& instance()
    {
        static thread_local ArgPool p;
        return p;
    }

    static std::size_t align_up(std::size_t v, std::size_t a)
    {
        return (v + a - 1) / a * a;
    }

    // Byte size of one slot's arrays, all 16-byte aligned. Sized for
    // kMaxGroups so the allocation happens exactly once per thread.
    static std::size_t slot_bytes()
    {
        std::size_t off = 0;
        auto take = [&](std::size_t bytes) { off = align_up(off, 16) + bytes; };
        take(sizeof(GProblemShape) * kMaxGroups);
        take(sizeof(void*) * kMaxGroups); // ptr_a
        take(sizeof(void*) * kMaxGroups); // ptr_b
        take(sizeof(void*) * kMaxGroups); // ptr_sfa
        take(sizeof(void*) * kMaxGroups); // ptr_sfb
        take(sizeof(void*) * kMaxGroups); // ptr_d
        take(sizeof(GStrideA) * kMaxGroups);
        take(sizeof(GStrideB) * kMaxGroups);
        take(sizeof(GStrideD) * kMaxGroups);
        take(sizeof(GLayoutSFA) * kMaxGroups);
        take(sizeof(GLayoutSFB) * kMaxGroups);
        return align_up(off, 256);
    }

    static void carve(char* raw, std::size_t& off, ArgSlot& s)
    {
        auto take = [&](std::size_t bytes) -> char*
        {
            off = align_up(off, 16);
            char* p = raw + off;
            off += bytes;
            return p;
        };
        s.problem = reinterpret_cast<GProblemShape*>(take(sizeof(GProblemShape) * kMaxGroups));
        s.ptr_a = reinterpret_cast<void const**>(take(sizeof(void*) * kMaxGroups));
        s.ptr_b = reinterpret_cast<void const**>(take(sizeof(void*) * kMaxGroups));
        s.ptr_sfa = reinterpret_cast<void const**>(take(sizeof(void*) * kMaxGroups));
        s.ptr_sfb = reinterpret_cast<void const**>(take(sizeof(void*) * kMaxGroups));
        s.ptr_d = reinterpret_cast<void**>(take(sizeof(void*) * kMaxGroups));
        s.stride_a = reinterpret_cast<GStrideA*>(take(sizeof(GStrideA) * kMaxGroups));
        s.stride_b = reinterpret_cast<GStrideB*>(take(sizeof(GStrideB) * kMaxGroups));
        s.stride_d = reinterpret_cast<GStrideD*>(take(sizeof(GStrideD) * kMaxGroups));
        s.layout_sfa = reinterpret_cast<GLayoutSFA*>(take(sizeof(GLayoutSFA) * kMaxGroups));
        s.layout_sfb = reinterpret_cast<GLayoutSFB*>(take(sizeof(GLayoutSFB) * kMaxGroups));
    }

    // Returns the slot index to use for this call, or -1 on allocation
    // failure. Allocation happens on the first call only and is refused
    // during stream capture.
    //
    // The capture query has to name the stream the call is being issued on.
    // A torch.cuda.graph capture runs on a side stream, and the legacy default
    // stream reports cudaStreamCaptureStatusNone throughout it, so the guard
    // as originally written against `nullptr` never fired: a capture whose
    // first grouped call had not been warmed up eagerly reached the cudaMalloc
    // below and surfaced as a bare "out of memory" instead of the message.
    int acquire(cudaStream_t stream)
    {
        if (base == nullptr)
        {
            cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(stream, &cap);
            if (cap == cudaStreamCaptureStatusActive)
            {
                std::fprintf(stderr,
                    "[fish_scales_ops] sm_100 grouped MXFP8: the argument-array pool is empty during stream "
                    "capture. Call linear_mxfp8_grouped_masked once eagerly before capturing.\n");
                std::abort();
            }
            std::size_t const per_slot = slot_bytes();
            if (cudaMalloc(&base, per_slot * kSlots) != cudaSuccess)
                return -1;
            for (int i = 0; i < kSlots; ++i)
            {
                std::size_t off = 0;
                carve(static_cast<char*>(base) + per_slot * i, off, slots[i]);
            }
        }
        int const idx = next;
        next = (next + 1) % kSlots;
        return idx;
    }
};

// ---------------------------------------------------------------------------
// Key-addressed static argument blocks (run b300_round3_20260922/M-A4)
// ---------------------------------------------------------------------------
//
// What a block is. When the caller supplies the problem shapes, the ten
// remaining per-group arrays are a function of nothing but this key: the five
// base addresses, the per-group byte strides, the group count, m_cap, N and K.
// Two calls with the same key need the same bytes in those arrays, so a block
// is written once when it is bound to a key and read by every later call that
// presents the key again. The key is compared on the host; nothing here reads
// device memory.
//
// Why a capture binds its block outside the capture. A `torch.cuda.graph`
// capture allocates its tensors from the graph's private pool, so the
// addresses the captured GEMM will read are not the addresses the eager
// warm-up bound, and the key misses. Writing the block through the capturing
// stream would put the writer back into the graph as a node, which is the
// launch this route exists to remove. The block is therefore written on the
// arena's own non-blocking stream and waited for on the host before the call
// returns, with the calling thread switched to `cudaStreamCaptureModeRelaxed`
// for the duration -- the same exchange the PyTorch caching allocator makes
// around its own capture-time cudaMalloc -- so that the launch and the wait
// are neither captured nor treated as capture-invalidating calls. By the time
// the graph is instantiated the arrays are resident, and the graph's kernel
// node reads them on every replay with no writer ahead of it.
//
// Why a capture-bound block is pinned. That kernel node keeps reading the
// block for as long as the graph exists, and the pool has no way to learn when
// a graph is destroyed, so the block can never be rebound: eviction is
// restricted to blocks that only eager calls have used, least recently used
// first. Memory therefore grows with the number of distinct (capture, GEMM)
// pairs a host thread creates, at `block_bytes(groups)` each (about 13 KB at
// 128 experts, 106 KB at the 1024 cap); the arena is one allocation of
// FSO_GROUPED_ARG_POOL_MB megabytes (default 16), made on the first eager
// call of the thread and never grown. A capture that finds the arena full
// aborts with a message naming the variable, because the alternative is to
// fall back to the prep launch inside the graph and silently give back the
// kernel this change removed; an eager call that finds it full falls back to
// exactly that prep launch, which is always correct.
//
// What is NOT protected, and was not before either. An eager-bound block may
// be evicted and rewritten while a GEMM issued earlier on another stream is
// still reading it. The two-slot alternation the prep path uses has the same
// exposure with a two-call window; the arena's window is the number of
// evictable blocks, so it is no worse. Within one stream, order alone makes
// the rewrite safe.
struct StaticKey
{
    void const* a;
    void const* b;
    void const* sfa;
    void const* sfb;
    void const* d;
    int64_t a_bytes;
    int64_t b_bytes;
    int64_t sfa_bytes;
    int64_t sfb_bytes;
    int64_t d_bytes;
    int groups;
    int m_cap;
    int shape_n;
    int shape_k;
};

struct StaticBlock
{
    StaticKey key;
    ArgSlot arrays; // `problem` stays nullptr: the shapes come from the caller
    std::size_t bytes = 0;
    bool pinned = false;
    unsigned long long last_use = 0;
};

struct StaticArena
{
    char* base = nullptr;
    std::size_t capacity = 0;
    std::size_t used = 0;
    std::vector<StaticBlock> blocks;
    unsigned long long tick = 0;
    cudaStream_t side = nullptr;
    bool failed = false;
    bool warned_full = false;

    static StaticArena& instance()
    {
        static thread_local StaticArena a;
        return a;
    }

    static std::size_t pool_bytes()
    {
        static std::size_t const v = []
        {
            char const* e = std::getenv("FSO_GROUPED_ARG_POOL_MB");
            long mb = (e && *e) ? std::atol(e) : 16;
            if (mb < 1)
                mb = 1;
            return static_cast<std::size_t>(mb) << 20;
        }();
        return v;
    }

    // Byte size of one block's ten arrays for `groups` groups, each array
    // 16-byte aligned and the whole rounded to 256 so blocks never share a
    // cache line.
    static std::size_t block_bytes(int groups)
    {
        std::size_t off = 0;
        auto take = [&](std::size_t bytes) { off = ArgPool::align_up(off, 16) + bytes; };
        for (int i = 0; i < 5; ++i)
            take(sizeof(void*) * groups);
        take(sizeof(GStrideA) * groups);
        take(sizeof(GStrideB) * groups);
        take(sizeof(GStrideD) * groups);
        take(sizeof(GLayoutSFA) * groups);
        take(sizeof(GLayoutSFB) * groups);
        return ArgPool::align_up(off, 256);
    }

    static void carve(char* raw, int groups, ArgSlot& s)
    {
        std::size_t off = 0;
        auto take = [&](std::size_t bytes) -> char*
        {
            off = ArgPool::align_up(off, 16);
            char* p = raw + off;
            off += bytes;
            return p;
        };
        s.problem = nullptr;
        s.ptr_a = reinterpret_cast<void const**>(take(sizeof(void*) * groups));
        s.ptr_b = reinterpret_cast<void const**>(take(sizeof(void*) * groups));
        s.ptr_sfa = reinterpret_cast<void const**>(take(sizeof(void*) * groups));
        s.ptr_sfb = reinterpret_cast<void const**>(take(sizeof(void*) * groups));
        s.ptr_d = reinterpret_cast<void**>(take(sizeof(void*) * groups));
        s.stride_a = reinterpret_cast<GStrideA*>(take(sizeof(GStrideA) * groups));
        s.stride_b = reinterpret_cast<GStrideB*>(take(sizeof(GStrideB) * groups));
        s.stride_d = reinterpret_cast<GStrideD*>(take(sizeof(GStrideD) * groups));
        s.layout_sfa = reinterpret_cast<GLayoutSFA*>(take(sizeof(GLayoutSFA) * groups));
        s.layout_sfb = reinterpret_cast<GLayoutSFB*>(take(sizeof(GLayoutSFB) * groups));
    }

    // First use: the arena and its side stream. Refused inside a capture, as
    // the prep path's pool is, and for the same reason (cudaMalloc).
    bool init(bool capturing)
    {
        if (base != nullptr)
            return true;
        if (failed)
            return false;
        if (capturing)
        {
            std::fprintf(stderr,
                "[fish_scales_ops] sm_100 grouped MXFP8: the static argument arena is empty during stream "
                "capture. Call the grouped GEMM once eagerly (with problem_shapes) before capturing.\n");
            std::abort();
        }
        capacity = pool_bytes();
        if (cudaMalloc(reinterpret_cast<void**>(&base), capacity) != cudaSuccess)
        {
            base = nullptr;
            failed = true;
            return false;
        }
        if (cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking) != cudaSuccess)
        {
            side = nullptr;
            failed = true;
            return false;
        }
        blocks.reserve(64);
        return true;
    }

    // The block for `key`: an existing one on a hit, otherwise the least
    // recently used evictable block of the same size, otherwise a fresh carve.
    // Returns -1 when none of those is available. `need_build` tells the
    // caller whether the block's arrays still have to be written.
    int bind(StaticKey const& key, bool capturing, bool& need_build)
    {
        need_build = false;
        ++tick;
        for (std::size_t i = 0; i < blocks.size(); ++i)
        {
            if (std::memcmp(&blocks[i].key, &key, sizeof(key)) == 0)
            {
                blocks[i].last_use = tick;
                // A graph is about to reference it: from now on it may never
                // be rebound, whoever bound it first. If an eager call bound
                // it, its write was a launch on that call's stream and nothing
                // here proves it has landed, so the block is written again
                // through the capture path (same bytes, host-waited) before
                // the graph can be instantiated.
                if (capturing && !blocks[i].pinned)
                {
                    blocks[i].pinned = true;
                    need_build = true;
                }
                return static_cast<int>(i);
            }
        }
        std::size_t const need = block_bytes(key.groups);
        int victim = -1;
        for (std::size_t i = 0; i < blocks.size(); ++i)
        {
            if (blocks[i].pinned || blocks[i].bytes != need)
                continue;
            if (victim < 0 || blocks[i].last_use < blocks[static_cast<std::size_t>(victim)].last_use)
                victim = static_cast<int>(i);
        }
        if (victim < 0)
        {
            if (used + need > capacity)
                return -1;
            StaticBlock b{};
            carve(base + used, key.groups, b.arrays);
            b.bytes = need;
            used += need;
            blocks.push_back(b);
            victim = static_cast<int>(blocks.size()) - 1;
        }
        StaticBlock& b = blocks[static_cast<std::size_t>(victim)];
        std::memcpy(&b.key, &key, sizeof(key));
        b.pinned = capturing;
        b.last_use = tick;
        need_build = true;
        return victim;
    }
};

// Build a key from the call's operands. Zero-filled first so the padding
// bytes compare equal under memcmp.
inline StaticKey make_static_key(void const* a, void const* b, void const* sfa, void const* sfb, void const* d,
    int64_t a_bytes, int64_t b_bytes, int64_t sfa_bytes, int64_t sfb_bytes, int64_t d_bytes, int groups, int m_cap,
    int shape_n, int shape_k)
{
    StaticKey k;
    std::memset(&k, 0, sizeof(k));
    k.a = a;
    k.b = b;
    k.sfa = sfa;
    k.sfb = sfb;
    k.d = d;
    k.a_bytes = a_bytes;
    k.b_bytes = b_bytes;
    k.sfa_bytes = sfa_bytes;
    k.sfb_bytes = sfb_bytes;
    k.d_bytes = d_bytes;
    k.groups = groups;
    k.m_cap = m_cap;
    k.shape_n = shape_n;
    k.shape_k = shape_k;
    return k;
}

// Write a block's arrays. Eager: a plain launch on the caller's stream, ahead
// of the GEMM. Capturing: on the arena's side stream, waited for on the host,
// under the relaxed capture mode -- see the note on `StaticArena` for why.
inline cudaError_t build_static_block(StaticArena& arena, StaticBlock const& blk, cudaStream_t stream,
    bool capturing, char* a_base, char* b_base, char* sfa_base, char* sfb_base, char* d_base)
{
    StaticKey const& k = blk.key;
    ArgSlot const& s = blk.arrays;
    int const threads = 32;
    int const grid = (k.groups + threads - 1) / threads;
    cudaError_t err = cudaSuccess;
    if (!capturing)
    {
        sm100_grouped_static_kernel<<<grid, threads, 0, stream>>>(s.ptr_a, s.ptr_b, s.ptr_sfa, s.ptr_sfb, s.ptr_d,
            s.stride_a, s.stride_b, s.stride_d, s.layout_sfa, s.layout_sfb, k.groups, k.m_cap, k.shape_n,
            k.shape_k, a_base, b_base, sfa_base, sfb_base, d_base, k.a_bytes, k.b_bytes, k.sfa_bytes, k.sfb_bytes,
            k.d_bytes);
        return cudaGetLastError();
    }
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    err = cudaThreadExchangeStreamCaptureMode(&mode); // `mode` now holds the caller's
    if (err != cudaSuccess)
        return err;
    sm100_grouped_static_kernel<<<grid, threads, 0, arena.side>>>(s.ptr_a, s.ptr_b, s.ptr_sfa, s.ptr_sfb, s.ptr_d,
        s.stride_a, s.stride_b, s.stride_d, s.layout_sfa, s.layout_sfb, k.groups, k.m_cap, k.shape_n, k.shape_k,
        a_base, b_base, sfa_base, sfb_base, d_base, k.a_bytes, k.b_bytes, k.sfa_bytes, k.sfb_bytes, k.d_bytes);
    err = cudaGetLastError();
    if (err == cudaSuccess)
        err = cudaStreamSynchronize(arena.side);
    cudaError_t const restore = cudaThreadExchangeStreamCaptureMode(&mode);
    return err != cudaSuccess ? err : restore;
}

// The caller's problem-shape tensor is int32 [G, 3] holding (rows, N, K) per
// group, and the launcher hands it to CUTLASS as a `cute::Shape<int, int, int>`
// array. cute's tuple stores its three ints in declaration order with no
// padding on every compiler this library builds with, but that is a property
// of the ABI and not of the language, so it is checked once at run time
// instead of assumed: a mismatch refuses the route with a clear error rather
// than feeding the scheduler transposed shapes.
inline bool problem_shape_is_three_plain_ints()
{
    static bool const ok = []
    {
        if (sizeof(GProblemShape) != 3 * sizeof(int))
            return false;
        GProblemShape s = cute::make_shape(0x11223344, 0x55667788, 0x0a0b0c0d);
        int v[3];
        std::memcpy(v, &s, sizeof(v));
        return v[0] == 0x11223344 && v[1] == 0x55667788 && v[2] == 0x0a0b0c0d;
    }();
    return ok;
}

// The static-array block for one call on the routing-supplied-shapes path,
// bound and, if needed, written. Returns nullptr when the arena is exhausted
// (eager only: a capture aborts inside `init`/here instead), in which case the
// caller takes the prep launch.
inline ArgSlot const* acquire_static_arrays(cudaStream_t stream, StaticKey const& key, char* a_base, char* b_base,
    char* sfa_base, char* sfb_base, char* d_base, cudaError_t& err)
{
    err = cudaSuccess;
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap);
    bool const capturing = (cap == cudaStreamCaptureStatusActive);
    StaticArena& arena = StaticArena::instance();
    if (!arena.init(capturing))
    {
        err = cudaErrorMemoryAllocation;
        return nullptr;
    }
    bool need_build = false;
    int const idx = arena.bind(key, capturing, need_build);
    if (idx < 0)
    {
        if (capturing)
        {
            std::fprintf(stderr,
                "[fish_scales_ops] sm_100 grouped MXFP8: the static argument arena (%zu MB) is full during "
                "stream capture; every captured (graph, GEMM) pair pins one block. Raise "
                "FSO_GROUPED_ARG_POOL_MB before the first grouped call.\n",
                arena.capacity >> 20);
            std::abort();
        }
        if (!arena.warned_full)
        {
            arena.warned_full = true;
            std::fprintf(stderr,
                "[fish_scales_ops] sm_100 grouped MXFP8: the static argument arena (%zu MB) is full of "
                "capture-pinned blocks; eager calls fall back to the per-launch prep kernel. Raise "
                "FSO_GROUPED_ARG_POOL_MB to keep them on the launch-free path.\n",
                arena.capacity >> 20);
        }
        return nullptr;
    }
    StaticBlock const& blk = arena.blocks[static_cast<std::size_t>(idx)];
    if (need_build)
    {
        err = build_static_block(arena, blk, stream, capturing, a_base, b_base, sfa_base, sfb_base, d_base);
        if (err != cudaSuccess)
            return nullptr;
    }
    return &blk.arrays;
}

// Whether this translation unit compiled CUTLASS's sm_100 grid-dependency
// control (GDC) instructions into the grouped kernel. The defining translation
// unit is `csrc/gemm/ops/mxfp8_sm100_grouped_kernel.cu`, which sets
// CUTLASS_ENABLE_GDC_FOR_SM100 for the sm_100 / sm_103 device passes only.
//
// This is a correctness gate, not a tuning knob. With GDC compiled in, the
// CUTLASS pointer-array kernel executes `griddepcontrol.wait` before its tile
// scheduler reads the device-resident per-group problem shapes — the call in
// cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp whose comment
// names exactly this case ("For the static grouped scheduler, the problem
// shapes might be produced by a previous kernel in global memory") — and again
// in each load participant before its first global access. Without GDC those
// calls compile to nothing, so a PDL-launched GEMM would have no barrier at
// all and could read the prep kernel's arrays before they are written. That is
// why the dense sm_100 split-K work refused `launch_with_pdl` on the CUTLASS
// GEMM, and why enabling it here is paired with the macro.
#if defined(FSO_SM100_GROUPED_GDC)
inline constexpr bool kGroupedGdcCompiled = true;
#else
inline constexpr bool kGroupedGdcCompiled = false;
#endif

// Programmatic dependent launch, the same switch the MoE glue kernels use
// (quant_kernels.cu / moe_glue.cu): on by default, FSO_DISABLE_PDL=1 restores
// plain stream-serialised launches for A/B measurement.
inline bool grouped_pdl_enabled() noexcept
{
    static bool const v = []
    {
        char const* e = std::getenv("FSO_DISABLE_PDL");
        return !(e && e[0] == '1');
    }();
    return v;
}

// Launch the prep kernel with the programmatic-stream-serialization attribute
// so its routing-independent half overlaps the kernel that precedes it.
inline cudaError_t launch_prep_kernel(dim3 grid, dim3 block, cudaStream_t stream, bool pdl, GProblemShape* problem,
    void const** ptr_a, void const** ptr_b, void const** ptr_sfa, void const** ptr_sfb, void** ptr_d,
    GStrideA* stride_a, GStrideB* stride_b, GStrideD* stride_d, GLayoutSFA* layout_sfa, GLayoutSFB* layout_sfb,
    int32_t const* masked_m, int groups, int m_cap, int shape_n, int shape_k, char* a_base, char* b_base,
    char* sfa_base, char* sfb_base, char* d_base, int64_t a_bytes, int64_t b_bytes, int64_t sfa_bytes,
    int64_t sfb_bytes, int64_t d_bytes)
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
    return cudaLaunchKernelEx(&cfg, sm100_grouped_prep_kernel, problem, ptr_a, ptr_b, ptr_sfa, ptr_sfb, ptr_d,
        stride_a, stride_b, stride_d, layout_sfa, layout_sfb, masked_m, groups, m_cap, shape_n, shape_k, a_base,
        b_base, sfa_base, sfb_base, d_base, a_bytes, b_bytes, sfa_bytes, sfb_bytes, d_bytes, pdl);
}

// Device SM count, queried once.
inline int device_sm_count() noexcept
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

// ---------------------------------------------------------------------------
// The second cascade entry: the slot-bound, swap-orientation decode kernel
// ---------------------------------------------------------------------------
// Defined in `csrc/gemm/ops/mxfp8_sm100_slot_kernel.cu`, which is the only
// translation unit that sees the build-generated kernel header. The dispatcher
// therefore decides the route through these three plain functions and never
// includes `grouped_slot_dispatch.cuh` (that header is included by exactly one
// translation unit, and the pointer-array prep kernel in this one is a
// non-static __global__, so the two headers cannot be mixed).
int slot_kernel_tile_n();
bool slot_kernel_instantiated(int shape_n, int shape_k);
cudaError_t slot_kernel_launch(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d, int32_t* scales_a,
    int32_t* scales_b, int32_t* masked_m, int groups, int num_slots, int m_cap, int shape_n, int shape_k,
    int const* slot_to_expert, cudaStream_t stream);

// The same kernel with the SwiGLU and the MXFP8 requantise folded into its
// epilogue: it writes the fp8 [G, m_cap, I] tensor and its 1x32 UE8M0 scale
// slab instead of the bf16 [G, m_cap, 2*I] gate_up tensor, so the decode band
// no longer launches the separate SwiGLU kernel either (run
// b300_mxfp8_20260917/M-A2). Same two-function shape as above, for the same
// translation-unit reason.
bool slot_swiglu_kernel_instantiated(int shape_n, int shape_k);
cudaError_t slot_swiglu_kernel_launch(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_fp8_e4m3* out_h,
    int32_t* out_sfh, int32_t* scales_a, int32_t* scales_b, int32_t* masked_m, int groups, int num_slots, int m_cap,
    int shape_n, int shape_k, int const* slot_to_expert, cudaStream_t stream);
// The token-column chunk the fused epilogue walks (`kSlotChunk` in
// fused_swiglu_slot_store.cuh: 32 columns, one warp width). `slot_route`
// admits the fused slot kernel only while the whole row capacity fits ONE such
// chunk (run b300_round3_20260922/M-A3, see the rule below).
int slot_swiglu_epilogue_chunk();

// What the route decision came out as. Anything from kSlotRouteRefusedMcap
// onwards means the caller forced the slot route on a configuration its
// correctness guard rejects; the op turns each of those into a TORCH_CHECK
// before any kernel is launched, because outside the guard the kernel does not
// fail — it returns 2^-127 times the right answer (see the mechanism note at
// the top of grouped_slot_dispatch.cuh).
enum SlotRouteDecision
{
    kSlotRouteCascade = 0,
    kSlotRouteSlot = 1,
    kSlotRouteRefusedMcap = 2,
    kSlotRouteRefusedShape = 3,
    kSlotRouteRefusedBound = 4,
};

// FSO_GROUPED_SLOT: 0 never takes the slot route, unset or 1 applies the rule,
// `force` takes it whenever it is legal, and `force@<N>` takes it whenever it
// is legal AND shape_n == N while sending every other shape to the cascade.
//
// The last form is the per-GEMM A/B knob. A MoE layer runs two grouped GEMMs
// whose N_w differ — gate_up has N_w = 2 * moe_inter, down has N_w = hidden —
// so naming one N_w routes exactly one of the two through the slot kernel and
// leaves the other on the cascade. That is what separates the two GEMMs'
// contributions in a layer measurement, where only the sum is timed; the
// mixed-variant columns of the run directory b300_mxfp8_20260917/M-I3 were
// produced with it, and the rule below was fitted to those columns.
//
// Read once into a function-local static so nothing can change between capture
// and replay.
struct SlotRouteKnob
{
    int mode;   // 0 never, 1 rule, 2 force, 3 force only for shape_n == only_n
    int only_n;
};

inline SlotRouteKnob const& slot_route_knob() noexcept
{
    static SlotRouteKnob const knob = []
    {
        SlotRouteKnob k{1, 0};
        char const* e = std::getenv("FSO_GROUPED_SLOT");
        if (e == nullptr)
            return k;
        if (std::strncmp(e, "force@", 6) == 0)
        {
            k.mode = 3;
            k.only_n = std::atoi(e + 6);
            return k;
        }
        if (std::strcmp(e, "force") == 0)
        {
            k.mode = 2;
            return k;
        }
        k.mode = (e[0] == '0') ? 0 : 1;
        return k;
    }();
    return knob;
}

inline int slot_route_mode() noexcept
{
    return slot_route_knob().mode;
}

// The rule and the hard guard.
//
// Legality, which applies to `force` as well: the whole row capacity must fit
// in one token tile (m_cap <= TileN), an instantiation must cover (N_w, K), and
// the caller must have supplied the host-static bound on how many groups can
// hold rows. That bound cannot be recovered from `expected_m`, which is
// ceil(rows / G) and is 1 for every decode M at G = 128, so the op takes it as
// its own argument and passes 0 when it is unknown.
//
// The performance rule is narrower than legality. It was re-derived in run
// b300_mxfp8_20260917/M-I5, after the per-tile skip, and it is NOT the
// active-fraction rule that preceded it.
//
// What changed and why the old rule no longer describes the route. Before the
// per-tile skip, a CTA of the slot kernel walked the tiles of every slot the
// host-static bound provided, including the slots that held no expert, and
// streamed those experts' weight panels from HBM. The route's advantage was
// therefore whatever remained after paying for that waste, and it shrank as the
// dead fraction (G - S) / G shrank -- which is why the boundary then sat at a
// fixed active fraction, S <= 3G/8. With the dead tiles skipped ahead of their
// first TMA, the waste is gone: on Family B gate_up at M = 8 the kernel's DRAM
// read fell from 1.296 to 1.040 times the live experts' weight bytes. What is
// left is the route's real mechanism, which has nothing to do with the dead
// fraction: the swap puts the expert weight rows on the 128-row M axis and the
// routed tokens on a 64-wide N axis, the activation stage in shared memory
// shrinks with it, and StageCountAutoCarveout spends the freed memory on
// mainloop stages -- eight here against the pointer-array route's six and four.
// That depth is what turns the expert weight stream from latency-bound into
// bandwidth-bound, and it is present whether or not any slot is dead. The
// measurement says so directly: on Family B at M = 16 to 48 EVERY expert can
// hold a row (S = G = 128, no dead slots at all) and the route is still 6 to
// 9 per cent faster than the cascade.
//
// Layer microseconds relative to the pointer-array control, with the route
// forced on one GEMM at a time so the two GEMMs separate (run directory
// b300_mxfp8_20260917/M-I5, B300, 148 SMs, better of two passes, worst
// pass-to-pass spread 0.84 %):
//
//   Family B, G = 128   (M = 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
//     gate_up    0.71  0.80  0.81  0.91  0.92  0.94  0.94  0.95  0.98  0.99  0.99  1.00
//     down       0.88  0.90  0.90  0.93  0.94  0.93  0.94  0.95  0.94  0.96  0.96  0.98
//   Family C, G = 256   (same M grid)
//     gate_up    0.77  0.76  0.76  0.80  0.88  0.92  0.91  0.94  0.97  0.97  0.99  1.01
//     down       0.83  0.81  0.82  0.90  0.95  0.96  0.98  0.98  1.01  1.03  1.03  1.05
//
// Three of the four classes are faster everywhere in the legal band; only
// Family C `down` turns over, and it turns over at a place the dispatcher can
// name. Ordering the classes by the number of tiles the slot grid is built
// from -- S * ceil(N_w / 128), the LIVE tiles -- every cell at or below 2048
// live tiles is faster than the control (by 0.8 to 6 per cent at the top of
// that range) with a single exception that the second clause below excludes,
// while the first cell above 2048, Family C `down` at S = 192 and 3072 live
// tiles, is 1.4 per cent slower, rising to 5 per cent at 4096. The live-tile
// count is expressed below as a number of waves of the machine rather than as a
// bare constant, because the quantity it stands for is how many times the
// narrow token tile has to push the live work past the SMs: the mainloop-depth
// advantage is a rate, so it is paid back on every wave, while whatever the
// wider pointer-array tile saves per issue accumulates with the wave count too
// and eventually overtakes it. On this 148-SM part fourteen waves is 2072
// tiles, which sits between the measured 2048 (still winning) and 3072 (losing).
//
// The dead-tile wave guard that the previous rule carried has been REMOVED, and
// removing it is not a simplification but a consequence of the measurement
// above. It required the pointer-array grid's dead tiles to be worth at least
// one wave, on the reasoning that deleting them was the whole advantage. The
// Family B cells at S = G = 128 have no dead tiles whatsoever and the route wins
// 6 to 9 per cent there, so that premise is false and the guard would have
// switched the route off exactly where it pays best.
//
// The second clause, re-derived once the FC1 was fused on BOTH routes (run
// b300_round3_20260922/M-A3). The M-I5 table above was measured with an unfused
// FC1 and a separate SwiGLU kernel on both routes, and it stopped one step
// below the full token tile because Family C `gate_up` lost 1.3 per cent at
// m_cap = 64. Since then the pointer-array FC1 gained the fused SwiGLU EVT
// direct-store epilogue (runs b300_mxfp8_20260917/M-E2 and M-A1) and the slot
// FC1 its own fused epilogue (run .../M-A2), and the two fusions do not pay
// equally: the pointer-array store writes one thread's whole row with vector
// stores, while the swap orientation puts one token's output element in each
// lane, so the slot epilogue's fp8 bytes and scale bytes go out transposed, one
// byte per lane per token, in 32-column chunks that each cost a warp reduction,
// a shared-memory exchange and a named barrier. That cost grows with the row
// capacity, and it overtakes the mainloop-depth advantage between m_cap = 32
// (one chunk) and m_cap = 40 (two chunks). Layer microseconds relative to the
// pointer-array control, the route forced on ONE GEMM at a time, best of two
// passes, worst spread 0.55 per cent:
//
//   Family B, G = 128                  (M = 24, 32, 40, 48, 56, 64)
//     FC1 fused slot / fused cascade   0.972 0.981 1.006 1.013 1.025 1.041
//     FC2 slot / cascade               0.978 0.976 0.977 0.981 0.986 0.984
//   Family C, G = 256
//     FC1 fused slot / fused cascade   0.962 0.972 1.020 1.032 1.033 1.066
//     FC2 slot / cascade               1.011 1.043 1.046 1.057 1.046 1.059
//
// The K-copies stage slopes say the same about the kernels themselves: at
// M = 48 / 64 the fused slot FC1 costs 3.2 / 5.6 microseconds MORE than the
// fused pointer-array FC1 on Family B and 0.9 / 8.5 more on Family C, while
// the plain slot FC2 costs 3.0 / 1.3 LESS on Family B (Family C's FC2 sits
// above the wave clause and loses 3.2 / 4.6, as the clause says). The plain
// kernel cells agree: forced onto the slot route at m_cap = 48 / 64 the
// unfused gate_up is 4.9 / 2.5 per cent faster on Family B and 2.4 / 0.3 (+)
// on Family C, and down is 9.2 / 5.3 per cent faster on Family B.
//
// So the second clause is a property of the KERNEL the call lands on, which
// the dispatcher knows because the fused FC1 has its own entry point: the
// fused-SwiGLU slot kernel is taken only while the whole row capacity fits one
// epilogue chunk (m_cap <= 32), and the plain slot kernel is taken up to the
// full token tile (m_cap <= 64, which is also the legality bound). The
// live-tile clause applies to both. What this changes on the published layers:
// Family B keeps the slot FC2 at M = 40..64 with the FC1 back on the fused
// pointer-array kernel (M = 64: 123.9 -> 122.2 microseconds, and M = 48 / 56
// gain 1.0 / 1.7 per cent), and Family C's `gate_up` leaves the slot route at
// M = 40..56, where the previous clause had it 1.9 to 3.3 per cent behind the
// pointer-array kernel (Family C routed M = 40 / 48 / 56: 123.3 / 135.5 / 142.1
// -> 121.1 / 131.7 / 137.6). The crossover between m_cap = 32 and 40 is not
// resolved finer than the M grid (36 was not measured), so the fused clause
// stops at the chunk width rather than at a fitted value.
inline SlotRouteDecision slot_route(
    int m_cap, int shape_n, int shape_k, int groups, int max_active_groups, bool fused_swiglu = false) noexcept
{
    SlotRouteKnob const& knob = slot_route_knob();
    int const mode = knob.mode;
    if (mode == 0)
        return kSlotRouteCascade;
    // `force@<N>` is the per-GEMM A/B form: a shape it does not name behaves as
    // if the route did not exist, so the other GEMM of the same layer stays on
    // the cascade instead of falling through to the rule.
    if (mode == 3 && shape_n != knob.only_n)
        return kSlotRouteCascade;
    bool const forced = (mode == 2 || mode == 3);
    if (max_active_groups <= 0)
        return forced ? kSlotRouteRefusedBound : kSlotRouteCascade;
    if (m_cap > slot_kernel_tile_n())
        return forced ? kSlotRouteRefusedMcap : kSlotRouteCascade;
    if (!slot_kernel_instantiated(shape_n, shape_k))
        return forced ? kSlotRouteRefusedShape : kSlotRouteCascade;
    if (forced)
        return kSlotRouteSlot;
    // The fused-SwiGLU slot kernel only while the row capacity fits one
    // epilogue chunk (see the note above); the plain kernel runs up to the full
    // token tile, which the legality check has already admitted.
    if (fused_swiglu && m_cap > slot_swiglu_epilogue_chunk())
        return kSlotRouteCascade;
    // The live tiles the slot grid is built from must be at most fourteen waves
    // of this machine. `groups` is unused by this clause and stays in the
    // signature because the legality checks and the refusal messages need it.
    long long const live_tiles
        = static_cast<long long>(max_active_groups) * ((shape_n + 127) / 128);
    if (live_tiles > 14LL * static_cast<long long>(device_sm_count()))
        return kSlotRouteCascade;
    (void) groups;
    return kSlotRouteSlot;
}

// Whether a grouped call of this shape would READ a caller-supplied
// `problem_shapes` tensor (run b300_round3_20260922/M-A4). The pointer-array
// cascade is the only reader; the slot route sizes its grid from `masked_m`
// and the slot list. A caller uses this to ask `moe_build_routing` for the
// shapes exactly where a GEMM will consume them, the same discipline
// `slot_route` gives the slot list. A refused forced call answers "consumed"
// too, which is harmless: the op raises before any kernel runs.
// `fused_swiglu` names the kernel the call lands on, exactly as for
// `slot_route`: the fused-SwiGLU FC1 leaves the slot route one step earlier
// than the plain grouped GEMM (one epilogue chunk against the full token
// tile), so the two queries are complements only when asked with the same
// flag, and a layer asks both for its FC1 with the flag set.
inline bool problem_shapes_consumed(
    int m_cap, int shape_n, int shape_k, int groups, int max_active_groups, bool fused_swiglu = false) noexcept
{
    return slot_route(m_cap, shape_n, shape_k, groups, max_active_groups, fused_swiglu) != kSlotRouteSlot;
}

// The tile the pointer-array cascade picks, as data rather than as a sequence
// of returns.
//
// Rules 1 to 4 are written out in the long comment above
// `gemm_dispatch_sm100_mxfp8_grouped`; this function is only their mechanical
// form. It exists because there are now TWO launchers that must agree on the
// tile — the ordinary grouped GEMM and the fused-SwiGLU FC1 of
// `gemm_dispatch_sm100_mxfp8_grouped_swiglu` — and a second hand-written copy
// of the rules would be a second thing to keep in step. The fused route reads
// `tile_n` and ignores `nosmem`, because its epilogue is always the direct
// store (that is the specialisation the fused store is written against).
struct CascadePick
{
    int tile_n;        // 128, 192 or 256
    bool nosmem;       // true = direct TMEM->register->global store, false = TMA bulk store
    bool evt = false;  // with nosmem: partition the CTA tile by the epilogue tile
                       // (the EVT form, see `EvtEpi` in grouped_gemm_types.cuh)
                       // instead of holding the whole CTA tile in one thread's
                       // registers. Ignored when nosmem is false.
};

// Rule 4's wave arithmetic: how full is the last wave of the grid a given
// N-tile width produces? `groups * ceil(N / tile_n)` is the tile count with
// every expert active (an upper bound on the real, routing-dependent count);
// the grouped scheduler runs one CTA per SM, so the remainder against the SM
// count is the tail wave's width. Returned scaled by 1024 to stay integral.
// A free function because the fused FC1's dispatcher asks the same question.
inline int tail_fill_q10(int shape_n, int tile_n, int groups) noexcept
{
    int const sms = device_sm_count();
    long long const tiles = static_cast<long long>(groups) * ((shape_n + tile_n - 1) / tile_n);
    long long const tail = tiles % sms;
    return tail == 0 ? 1024 : static_cast<int>(tail * 1024 / sms);
}

inline CascadePick cascade_pick(int m_cap, int shape_n, int shape_k, int groups, int expected_m) noexcept
{
    int const em = expected_m < 1 ? 1 : expected_m;

    // Long mainloop: the 256-wide N tile with the TMA bulk-store epilogue,
    // except at a tiny row capacity where the tile count matters more, and
    // except where the 192-wide tile fills the last wave better (rule 4).
    if (shape_k >= 1024)
    {
        if (m_cap <= 4)
            return CascadePick{128, false};
        if (em <= 16 && shape_n % 192 == 0
            && tail_fill_q10(shape_n, 192, groups) > tail_fill_q10(shape_n, 256, groups))
            return CascadePick{192, true};
        return CascadePick{256, false};
    }
    // Short mainloop (rule 3, re-derived in run b300_mxfp8_20260917/M-A1; the
    // long comment on rules 1-5 below carries the measurement). The question
    // the branch asks is whether the 192-wide tile's five mainloop buffers
    // cover the whole K loop.
    //
    // Four K-tiles or fewer: they do, and that tile with the EVT-partitioned
    // direct store was the fastest configuration measured at every expected_m,
    // eleven consecutive M without a crossover. The cap at 256 is not a
    // measured boundary -- expected_m only reached 128 on the grid -- it is
    // there so an unmeasured extrapolation cannot run away.
    if (shape_k <= 512)
    {
        if (em <= 256)
            return CascadePick{shape_n >= 192 ? 192 : 128, true, true};
        return CascadePick{256, false};
    }
    // More K-tiles than the narrower tile can buffer: take the 256-wide tile,
    // with the EVT direct store while the store is still a sliver and the TMA
    // bulk store once the rows pile up.
    if (em <= 64)
        return CascadePick{shape_n >= 256 ? 256 : 128, true, true};
    return CascadePick{256, false};
}

// ---------------------------------------------------------------------------
// Which FC1 form a caller should use: the fused one, or the old pair
// ---------------------------------------------------------------------------
//
// FSO_FC1_FUSED: `0` never uses the fused FC1, unset applies the rule below,
// `1` uses it wherever it is legal. Read once into a function-local static, so
// nothing can change between a CUDA-graph capture and its replays.
//
// The rule. BOTH routes now have a fused FC1, so the answer is simply "yes
// wherever an instantiation exists". The two fusions are different pieces of
// code because the two routes hold the accumulator differently — the
// pointer-array route has a thread own one output row and a contiguous run of
// that row's N columns, so the gate/up pairing and the 32-wide amax are
// register-local, while the slot route's swap puts the weight rows on M, so the
// pair sits in two lanes and one 1x32 block spans two warps (run
// b300_mxfp8_20260917/M-E1 section 1.8 stated the problem, run
// b300_mxfp8_20260917/M-A2 solved it with a `shfl_xor` for the pair, one
// `redux.sync` per warp and one shared-memory step across the warp pair). What
// the dispatcher has to decide is therefore not WHETHER to fuse but WHICH fused
// kernel to launch, and that question is answered by `slot_route` exactly as it
// is for the unfused FC1, so the two can never drift apart.
//
// `shape_n` here is the FC1 weight-row count N = 2 * I, the same quantity
// `slot_route` is asked about for the unfused FC1.
inline int fused_fc1_knob() noexcept
{
    static int const mode = []
    {
        char const* e = std::getenv("FSO_FC1_FUSED");
        if (e == nullptr)
            return 1; // the rule
        if (e[0] == '0')
            return 0; // never
        return 2;     // force wherever legal
    }();
    return mode;
}

// The load-time half of the same decision: may this shape use the fused FC1
// AT ALL, on any M?
//
// A caller has to answer that before it loads its weights, because the fused
// op needs the interleaved row order and the unfused fallback then needs the
// pairwise SwiGLU kernel to match — one layout for the whole model, chosen
// once. `fused_fc1_route` answers the other half, per call. Both read the same
// knob through the same static, so "the feature is off" and "this call does
// not take it" cannot disagree.
inline bool fused_fc1_available(int shape_n, int shape_k) noexcept
{
    if (fused_fc1_knob() == 0)
        return false;
    // The fused epilogue halves N and writes one 1x32 scale block per 32
    // output columns, so the output width must be a whole number of the scale
    // slab's 128-column blocks, and K must satisfy the usual MXFP8 constraint.
    return (shape_n % 2 == 0) && ((shape_n / 2) % 128 == 0) && (shape_k % 128 == 0);
}

inline bool fused_fc1_route(int m_cap, int shape_n, int shape_k, int groups, int max_active_groups) noexcept
{
    if (!fused_fc1_available(shape_n, shape_k))
        return false;
    if (fused_fc1_knob() == 2)
        return true;
    // Where the dispatcher takes the slot route FOR THE FUSED KERNEL, the fused
    // FC1 is the slot route's own fused variant, so the caller may still use
    // the fused op — but only if that variant is instantiated for this shape.
    // If it is not, the caller must keep the old pair, because the fused op
    // would otherwise have to fall back to the pointer-array fused kernel on a
    // band where the slot route is 2 to 4 per cent faster. The verdict is asked
    // with `fused_swiglu = true` because that is the kernel the fused op lands
    // on: its clause stops at one epilogue chunk, one step short of the plain
    // kernel's (run b300_round3_20260922/M-A3).
    if (slot_route(m_cap, shape_n, shape_k, groups, max_active_groups, /*fused_swiglu=*/true) == kSlotRouteSlot)
        return slot_swiglu_kernel_instantiated(shape_n, shape_k);
    return true;
}

} // namespace grouped_detail

// Launch one (TileM, TileN, ClusterM, ClusterN, TileK, NoSmemEpi)
// instantiation of the grouped kernel.
//
// `problem_shapes`, when not null, is the caller's int32 [groups, 3] array of
// (rows, N, K) per group, written by `moe_build_routing`; the launcher then
// issues no prep kernel and takes its static arrays from `StaticArena`. Null
// keeps the prep launch of the original design.
template <int TileM, int TileN, int ClusterM, int ClusterN, int TileK = 128, bool NoSmemEpi = false,
    bool EvtEpi = false>
cudaError_t launch_sm100_mxfp8_grouped_gemm(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int32_t* masked_m, int groups, int m_cap, int shape_n, int shape_k,
    cudaStream_t stream, int32_t const* problem_shapes = nullptr, int swizzle = 0)
{
    using Config = Sm100MxFP8GroupedGemmConfig<TileM, TileN, ClusterM, ClusterN, TileK, NoSmemEpi, EvtEpi>;
    using Gemm = typename Config::Gemm;
    using GemmKernel = typename Config::GemmKernel;
    using Args = typename Gemm::Arguments;
    using Params = typename Gemm::Params;
    namespace gd = grouped_detail;

    static_assert(cute::is_same_v<typename Config::InternalStrideA, gd::GStrideA>, "stride type drift (A)");
    static_assert(cute::is_same_v<typename Config::InternalStrideB, gd::GStrideB>, "stride type drift (B)");
    static_assert(cute::is_same_v<typename Config::InternalStrideD, gd::GStrideD>, "stride type drift (D)");
    static_assert(cute::is_same_v<typename Config::InternalLayoutSFA, gd::GLayoutSFA>, "SFA layout type drift");
    static_assert(cute::is_same_v<typename Config::InternalLayoutSFB, gd::GLayoutSFB>, "SFB layout type drift");

    if (groups < 1 || groups > gd::kMaxGroups)
        return cudaErrorInvalidValue;

    if (gd::read_print_tile_info())
    {
        static bool s_info_printed = false;
        if (!s_info_printed)
        {
            s_info_printed = true;
            std::fprintf(stderr,
                "[fso sm100 grouped tile] %dx%dx%d cluster=(%d,%d) nosmem=%d evt=%d "
                "stages=%d smem_bytes=%d epi_smem_bytes=%d\n",
                TileM, TileN, TileK, ClusterM, ClusterN, static_cast<int>(NoSmemEpi),
                static_cast<int>(EvtEpi), Config::kStages, Config::kSmemBytes, Config::kEpiSmemBytes);
        }
    }

    // Per-group byte strides of the caller's packed slabs.
    int64_t const kp = shape_k / 128;
    int64_t const m_pad = (static_cast<int64_t>(m_cap) + 127) / 128 * 128;
    int64_t const a_bytes = static_cast<int64_t>(m_cap) * shape_k;                 // fp8
    int64_t const b_bytes = static_cast<int64_t>(shape_n) * shape_k;               // fp8
    int64_t const sfa_bytes = m_pad * kp * static_cast<int64_t>(sizeof(int32_t));  // atom slab
    int64_t const sfb_bytes = static_cast<int64_t>(shape_n) * kp * static_cast<int64_t>(sizeof(int32_t));
    int64_t const d_bytes = static_cast<int64_t>(m_cap) * shape_n * static_cast<int64_t>(sizeof(__nv_bfloat16));

    bool const pdl = gd::grouped_pdl_enabled();

    // Which arrays the kernel reads: the caller's problem shapes plus a
    // key-bound static block (no launch), or the prep kernel's slot (one
    // launch). `arrays` and `problem` are what the two paths have in common.
    gd::ArgSlot const* arrays = nullptr;
    gd::GProblemShape* problem = nullptr;
    if (problem_shapes != nullptr && gd::problem_shape_is_three_plain_ints())
    {
        cudaError_t bind_err = cudaSuccess;
        arrays = gd::acquire_static_arrays(stream,
            gd::make_static_key(mat_a, mat_b, scales_a, scales_b, mat_d, a_bytes, b_bytes, sfa_bytes, sfb_bytes,
                d_bytes, groups, m_cap, shape_n, shape_k),
            reinterpret_cast<char*>(mat_a), reinterpret_cast<char*>(mat_b), reinterpret_cast<char*>(scales_a),
            reinterpret_cast<char*>(scales_b), reinterpret_cast<char*>(mat_d), bind_err);
        if (bind_err != cudaSuccess)
            return bind_err;
        if (arrays != nullptr)
            problem = reinterpret_cast<gd::GProblemShape*>(const_cast<int32_t*>(problem_shapes));
    }
    else if (problem_shapes != nullptr)
    {
        std::fprintf(stderr,
            "[fish_scales_ops] sm_100 grouped MXFP8: cute::Shape<int,int,int> is not three plain ints on this "
            "toolchain; refusing the routing-supplied problem shapes.\n");
        return cudaErrorInvalidValue;
    }
    if (arrays == nullptr)
    {
        auto& pool = gd::ArgPool::instance();
        int const slot_idx = pool.acquire(stream);
        if (slot_idx < 0)
            return cudaErrorMemoryAllocation;
        gd::ArgSlot const& slot = pool.slots[slot_idx];
        // One thread per group, spread over 32-thread blocks: the prep is a
        // latency problem, not a throughput one, and a single 128-thread block
        // puts every group's layout arithmetic on one SM.
        int const threads = 32;
        int const blocks = (groups + threads - 1) / threads;
        cudaError_t const prep_err = gd::launch_prep_kernel(dim3(blocks), dim3(threads), stream, pdl, slot.problem,
            slot.ptr_a, slot.ptr_b, slot.ptr_sfa, slot.ptr_sfb, slot.ptr_d, slot.stride_a, slot.stride_b,
            slot.stride_d, slot.layout_sfa, slot.layout_sfb, masked_m, groups, m_cap, shape_n, shape_k,
            reinterpret_cast<char*>(mat_a), reinterpret_cast<char*>(mat_b), reinterpret_cast<char*>(scales_a),
            reinterpret_cast<char*>(scales_b), reinterpret_cast<char*>(mat_d), a_bytes, b_bytes, sfa_bytes,
            sfb_bytes, d_bytes);
        if (prep_err != cudaSuccess)
            return prep_err;
        arrays = &slot;
        problem = slot.problem;
    }
    gd::ArgSlot const& slot = *arrays;

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = gd::device_sm_count();

    typename GemmKernel::TileSchedulerArguments scheduler{};
    if (swizzle > 0)
        scheduler.max_swizzle_size = swizzle;

    using MainloopArgs = typename GemmKernel::MainloopArguments;
    using EpilogueArgs = typename GemmKernel::EpilogueArguments;

    MainloopArgs mainloop_args{};
    mainloop_args.ptr_A = reinterpret_cast<decltype(mainloop_args.ptr_A)>(slot.ptr_a);
    mainloop_args.dA = slot.stride_a;
    mainloop_args.ptr_B = reinterpret_cast<decltype(mainloop_args.ptr_B)>(slot.ptr_b);
    mainloop_args.dB = slot.stride_b;
    mainloop_args.ptr_SFA = reinterpret_cast<decltype(mainloop_args.ptr_SFA)>(slot.ptr_sfa);
    mainloop_args.layout_SFA = slot.layout_sfa;
    mainloop_args.ptr_SFB = reinterpret_cast<decltype(mainloop_args.ptr_SFB)>(slot.ptr_sfb);
    mainloop_args.layout_SFB = slot.layout_sfb;

    EpilogueArgs epilogue_args{};
    epilogue_args.thread.alpha = 1.0f;
    epilogue_args.thread.beta = 0.0f;
    epilogue_args.ptr_C = nullptr;
    epilogue_args.dC = nullptr;
    epilogue_args.ptr_D = reinterpret_cast<decltype(epilogue_args.ptr_D)>(slot.ptr_d);
    epilogue_args.dD = slot.stride_d;

    Args args{cutlass::gemm::GemmUniversalMode::kGrouped, typename Config::ProblemShape{groups, problem, nullptr},
        mainloop_args, epilogue_args, hw_info, scheduler};

    // Workspace: per-SM tensormap scratch. Its size depends only on the SM
    // count, so it is allocated once per instantiation and never grows — which
    // is what lets the Params cache hold a stable pointer.
    static thread_local void* s_ws = nullptr;
    static thread_local std::size_t s_ws_bytes = 0;
    std::size_t const need = Gemm::get_workspace_size(args);
    if (need > s_ws_bytes)
    {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &cap);
        if (cap == cudaStreamCaptureStatusActive)
        {
            std::fprintf(stderr,
                "[fish_scales_ops] sm_100 grouped MXFP8: needs %zu workspace bytes during stream capture but the "
                "pool holds %zu. Warm up linear_mxfp8_grouped_masked eagerly before capturing.\n",
                need, s_ws_bytes);
            std::abort();
        }
        if (s_ws)
            cudaFree(s_ws);
        if (cudaMalloc(&s_ws, need) != cudaSuccess)
            return cudaErrorMemoryAllocation;
        s_ws_bytes = need;
        if (GemmKernel::initialize_workspace(args, s_ws, stream) != cutlass::Status::kSuccess)
            return cudaErrorUnknown;
    }

    // Params cache. For grouped mode CUTLASS builds the initial TMA
    // descriptors from the tile shape, so Params depends on the argument-array
    // pointers, the problem-shape pointer, the group count and the scheduler
    // knobs — never on (m_cap, N, K). The two pointers stand for the slot on
    // the prep path (both fixed per slot) and for the static block plus the
    // caller's tensor on the routing-supplied path.
    struct CacheKey
    {
        void const* arrays;
        void const* problem;
        int groups;
        int swizzle;
        bool operator==(CacheKey const& o) const { return std::memcmp(this, &o, sizeof(o)) == 0; }
    };
    struct CacheHash
    {
        std::size_t operator()(CacheKey const& k) const noexcept
        {
            std::size_t h = 1469598103934665603ULL;
            auto const* p = reinterpret_cast<unsigned char const*>(&k);
            for (std::size_t i = 0; i < sizeof(k); ++i)
            {
                h ^= p[i];
                h *= 1099511628211ULL;
            }
            return h;
        }
    };
    static thread_local std::unordered_map<CacheKey, Params, CacheHash> s_params_cache;
    CacheKey key;
    std::memset(&key, 0, sizeof(key));
    key.arrays = slot.ptr_a;
    key.problem = problem;
    key.groups = groups;
    key.swizzle = swizzle;

    Params kernel_params;
    auto it = s_params_cache.find(key);
    if (it != s_params_cache.end())
    {
        kernel_params = it->second;
    }
    else
    {
        if (Gemm::can_implement(args) != cutlass::Status::kSuccess)
            return cudaErrorInvalidValue;
        kernel_params = GemmKernel::to_underlying_arguments(args, s_ws); // host-only
        // Each capture on the routing-supplied path brings a new key (its
        // own block and its own shape tensor), so the cap is wide enough for
        // a bench process's captures; past it the host recomputes Params,
        // which is correct and costs microseconds of host time only.
        if (s_params_cache.size() < 256u)
            s_params_cache.emplace(key, kernel_params);
    }

    // Set max dynamic smem once per instantiation, on the first (eager) call.
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

    // PDL on the GEMM itself. The kernel's own `griddepcontrol.wait` — present
    // only when this translation unit compiled GDC, see kGroupedGdcCompiled —
    // keeps the ordering against whatever precedes the GEMM in the stream: the
    // prep kernel on that path, and on the routing-supplied path the kernel
    // ahead of the GEMM in the layer (gather-quantize or the other GEMM),
    // whose completion in turn implies the routing kernel's, because every
    // kernel of the chain executes its own wait before it can retire. The
    // wait sits at cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp
    // line 839 (v4.4.2), and the first read of the problem shapes is the tile
    // scheduler's constructor at line 842 (sm90_tile_scheduler_group.hpp lines
    // 283-289). Meanwhile the CTAs, their cluster launch and the per-group
    // tensormap initialisation still start early.
    bool const gemm_pdl = gd::kGroupedGdcCompiled && pdl;
    return Gemm::run(kernel_params, stream, nullptr, gemm_pdl) == cutlass::Status::kSuccess
        ? cudaSuccess
        : cudaErrorLaunchFailure;
}

// Top-level sm_100 / sm_103 grouped MXFP8 dispatcher.
//
// Cascade v1 (2026-09-15, first tuning pass). Every rule below was decided on
// the LAYER cell of bench_moe_qwen3_30a3.py — the whole MoE block in one CUDA
// graph — for the two Qwen3 MoE families, with one projection forced through
// FSO_FORCE_TILE + FSO_FORCE_TILE_K at a time while the other kept its cascade
// pick. `expected_m` is the host-side hint ceil(total_rows / G); the real
// per-group row counts live on device and never reach the host.
//
// Rule 1 — the M tile stays at 128 rows (1SM). The 2SM 256-row tile was swept
// at every measured band and lost everywhere, by 3 % at the largest
// expected_m measured and by 15-46 % in the decode band: a masked MoE group
// holds far fewer rows than 256, so the second SM's half of the tile is
// mostly padding, and the tile count drops below what the machine needs to
// stay busy. The 2SM instantiations stay reachable through FSO_FORCE_TILE for
// future sweeps but are not on the cascade.
//
// Rule 2 — the N tile follows K, not N. A 256-wide N tile halves the number of
// CTAs and doubles the weight bytes each CTA pulls per K-tile. With a long
// mainloop (K >= 1024, i.e. at least 8 K-tiles) there is enough pipelining to
// hide that and the wider tile wins on both families' gate_up projection
// (K = 2048) by 4-6 % through the decode and mid bands. With a short mainloop
// (both families' down projection, K = 512 and K = 768, i.e. 4 and 6 K-tiles)
// the wider tile has nothing to hide behind and loses 4-9 % at small
// expected_m.
//
// Rule 2a — the long-K route falls back to the 128-wide tile when the row
// capacity itself is tiny (m_cap <= 4). m_cap bounds the rows per group and
// therefore the token count, so at four rows the whole problem is a handful of
// experts; halving the tile count there leaves the machine with less work in
// flight than it can run, and the narrower tile measured 2.5-4.6 % faster on
// the layer cell at M = 2 and M = 4 while being a tie at M = 1.
//
// Rule 3 — on a short mainloop the direct store beats the TMA bulk store for
// most of the band, it has to be the EVT-partitioned form of that epilogue, and
// which N tile it is paired with is decided by whether that tile's mainloop
// buffers cover the K loop.
//
// Re-derived in run b300_mxfp8_20260917/M-A1. That run forced every
// instantiation the pointer-array family can express — four N widths x two
// TileK x 1SM/2SM x two epilogues, the twenty codes ST 1-8 — through
// FSO_FORCE_TILE on the FC2 of both Qwen3 MoE families at M = 64 to 4096, on
// the LAYER cell, two interleaved passes whose median pass-to-pass
// disagreement was 0.12 per cent and whose worst was 1.04 per cent, against a
// control whose process timed the same list of cells. (An unmatched control is
// worth up to 2.3 per cent on this bench, because a process that has drawn and
// freed more tensors meets the next M with a different allocator state; that
// had to be fixed before any of the numbers below meant anything.)
//
// What the twenty said, and what was wrong with them. The shipped pick — the
// 256-wide tile with the TMA bulk store — was the best of the twenty from
// expected_m = 32 upwards on both families, and below that the 192-wide tile
// with the DIRECT store was 0.2 to 3.7 per cent faster. But the direct store
// was not monotone in the N tile: at expected_m = 4 on Family B it cost 49.00
// microseconds at TileN = 128, 46.70 at 192 and 69.00 at 256. No argument about
// store volume produces that shape.
//
// The cause, and the instantiation that removes it. The unfused config built
// the pointer-array NoSmem epilogue through the CUTLASS builder's
// DEFAULT-fusion branch, which brings the whole CTA tile of the accumulator
// into one thread's registers, so the register cost grows with TileN until the
// tile stops fitting. `EvtEpi` in grouped_gemm_types.cuh selects the EVT
// specialisation instead, whose fragment is one output row by 64 columns
// whatever TileN is. That the epilogue is the whole of the difference can be
// read off FSO_PRINT_TILE_INFO, which now reports the grouped path too: the two
// direct-store forms produce the SAME mainloop — 6 / 5 / 4 stages at TileN
// 128 / 192 / 256 and the same 206976 / 216576 / 206848 bytes of shared memory
// — and differ only in a 1-byte against a 32-byte epilogue carve-out. Nothing
// about the loads changed; only how the accumulator reaches global memory.
// With that one template argument the 256-wide direct store stops collapsing
// and becomes the fastest configuration measured anywhere in the band: on
// Family B it takes the whole layer cell from 128.14 to 124.04 microseconds at
// M = 64, 133.16 to 128.80 at 128, 137.74 to 133.45 at 256, 145.20 to 141.19
// at 512 and 158.22 to 154.85 at 1024 — 2.1 to 3.3 per cent of the layer — and
// the isolated FC2 from 46.81 to 44.56 microseconds at M = 64.
//
// Why the branch is on `shape_k <= 512` rather than on a single width. The
// direct store's shared-memory carve-out is one byte where the TMA store's is
// 17408, so StageCountAutoCarveout gives the direct store's 192-wide tile FIVE
// mainloop buffers against the 256-wide tile's four. Family C's FC2 has
// K = 512, four K-tiles at TileK 128, so the fifth buffer covers the WHOLE
// mainloop and no K-tile load is ever waited on; Family B's has K = 768, six
// K-tiles, so it does not, and the wider tile's smaller CTA count wins instead.
// Measured, Family C prefers 192 at every M of the grid by 1.1 to 2.1 per cent
// of the layer and Family B prefers 256 at every M by 1.0 to 1.8 per cent; both
// runs are seven consecutive M wide and reproduce on both passes.
//
// Why only the long-K side of the branch carries a cut-off. Where the fifth
// buffer covers the loop there is no crossover to find: on Family C the
// 192-wide EVT direct store is ahead of the shipped pick at expected_m = 1, 2,
// 3, 4, 8, 16, 32, 48, 64, 96 and 128 — eleven consecutive M, by 0.6 to 6.7 per
// cent of the layer — and the grid ends before it turns over. Where it does
// not, Family B crosses over between expected_m = 96 (the EVT direct store
// still wins the layer by 0.8 per cent) and expected_m = 128 (it loses by 0.9,
// and the isolated FC2 by 5.6). The cut-off is written at 64 rather than 96
// because 64 is the last expected_m at which the win is decisive (2.1 per cent)
// rather than inside the noise band; the 0.8 per cent at expected_m = 96 is
// knowingly left on the table.
//
// What did NOT change, with the numbers that say so. The 2SM (256-row) tile
// lost at every M of both families, by 36 per cent at M = 64 and 7 per cent at
// M = 4096 on the isolated FC2, so rule 1 stands. TileK 256 lost at every M of
// both families, by 13 per cent at M = 64 and 18 per cent at M = 4096, so the K
// tile stays at 128. A two-CTA cluster on the 1SM tile, which multicasts the
// weight tile to the two CTAs holding neighbouring M tiles and is the one axis
// of this family no earlier sweep had touched, lost at every M of both families
// — its best width runs 1.10 to 1.86 times the cascade's own pick on the
// isolated FC2 and 1.03 to 1.30 times on the layer — which is the same verdict
// the dense path recorded for cluster multicast in the mid band. It stays
// reachable as ST 11 and is not on the cascade.
//
// Rule 4 below keeps the default-fusion direct store it was tuned with. The EVT
// form is very likely right there too, but rule 4 governs the LONG-mainloop
// route, and neither family's layer ever reaches the unfused long-K cascade
// (above m_cap = 32 the fused FC1 takes that shape and below it the slot route
// does), so there is no layer cell to decide it on and it was left alone.
//
// Cascade v2 adds rule 4 (2026-09-17, second tuning pass, subtask T6). Nothing
// in rules 1-3 changed.
//
// Rule 4 — a 192-wide N tile with the direct store, but only where the tile
// count it produces fits the machine better than the 256-wide tile's does.
// CUTLASS accepts N tiles of 64, 128, 192 and 256 on this block-scaled path
// (fso's own static_assert used to reject 64 and 192, which is what this round
// relaxed), and the N tile is the dispatcher's only lever on how many CTAs the
// routed problem produces.
//
// The mechanism is wave arithmetic against the 148 SMs of a B300, and it is
// visible in the tile counts. The grouped scheduler launches one CTA per SM and
// walks the tile list, so what matters is how full the last wave is. Family B's
// gate_up projection (N = 1536, 128 experts, all of them active from M = 128 on)
// produces 128 x ceil(1536/256) = 768 tiles at the 256-wide tile: five full
// waves and a sixth holding 28 of 148 CTAs, so the tail wave runs at 19 %
// occupancy. The same problem at the 192-wide tile produces 128 x 8 = 1024
// tiles: six full waves and a seventh holding 136 of 148, a 92 % tail. That is
// worth 1.7-3.4 % on the whole layer cell at M = 16...256, reproduced on two
// independent passes.
//
// The same arithmetic is why the rule must not fire anywhere else that was
// measured. Family C's gate_up (N = 1024, 256 experts) already has a 92 % tail
// at the 256-wide tile (1024 tiles) and drops to a 38 % tail at the 192-wide
// tile (1536 tiles), and 1024 is not a multiple of 192 so a sixth of the last
// tile's columns are padding as well; measured, the 192-wide tile is a wash
// there (-1.9 % to +1.3 %). Both families' down projections (N = 2048) likewise
// go from a 92 % / 84 % tail at 256 wide to a 51 % / 3 % tail at 192 wide, and
// measure as a wash. So the rule is gated on the arithmetic itself rather than
// on a shape whitelist: it fires only when 192 divides N exactly and when the
// last wave of the 192-wide grid is fuller than the last wave of the 256-wide
// grid. `groups` is used as the tile-count proxy because the host cannot see
// how many experts the routing actually activated; it is an upper bound, so the
// rule only fires where the grid is close to expert-saturated.
//
// Two further facts about the 192-wide tile shape this rule. First, the tile is
// paired with the direct store rather than the TMA store: `StageCountAutoCarveout`
// gives the 192-wide tile 4 mainloop stages with the TMA epilogue's 17 KB of
// shared memory carved out and 5 stages with the direct store's, and the extra
// stage is most of the difference (192-wide with the TMA store measures -0.3 %
// to -1.8 %, with the direct store -1.7 % to -3.4 %). Second, the rule is capped
// at expected_m <= 16 for the same reason rule 3 is: past that the direct
// store's scattered global writes stop being a sliver and the configuration
// turns into a 4 % loss at expected_m = 64 and a 30 % loss at expected_m = 128.
//
// Cascade v2a adds a second ENTRY, not a rule (run b300_mxfp8_20260917/M-I2).
// The slot-bound swap-orientation kernel of `grouped_slot_dispatch.cuh` is a
// different kernel with a different geometry, so it is selected before the
// cascade rather than by them; `slot_route` above holds the rule and the hard
// correctness guard. It sits after the FSO_FORCE_TILE block on purpose:
// FSO_FORCE_TILE names a pointer-array tile and is the A/B knob for rules 1-4,
// so an explicit tile request must keep reaching the kernel it names. The two
// knobs are not meant to be combined.
//
// `problem_shapes` (run b300_round3_20260922/M-A4) is the caller's int32
// [groups, 3] array from `moe_build_routing`, or null. Only the pointer-array
// cascade reads it; the slot route derives its grid from `masked_m` and the
// slot list and ignores it.
inline cudaError_t gemm_dispatch_sm100_mxfp8_grouped(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int32_t* masked_m, int groups, int m_cap, int shape_n, int shape_k,
    int expected_m, int max_active_groups, int const* slot_to_expert, int32_t const* problem_shapes,
    cudaStream_t stream)
{
#define DISPATCH_SM100_GROUPED(TM_, TN_, CM_, CN_, TK_, NOSMEM_)                                                       \
    launch_sm100_mxfp8_grouped_gemm<TM_, TN_, CM_, CN_, TK_, NOSMEM_>(                                                 \
        mat_a, mat_b, mat_d, scales_a, scales_b, masked_m, groups, m_cap, shape_n, shape_k, stream, problem_shapes)
// The same launcher with the EVT-partitioned direct store (see `EvtEpi` in
// grouped_gemm_types.cuh). Separate macro rather than a seventh argument on the
// one above, so the twenty pre-existing lines of the force table stay
// character-for-character what they were.
#define DISPATCH_SM100_GROUPED_EVT(TM_, TN_, CM_, CN_, TK_)                                                            \
    launch_sm100_mxfp8_grouped_gemm<TM_, TN_, CM_, CN_, TK_, true, true>(                                              \
        mat_a, mat_b, mat_d, scales_a, scales_b, masked_m, groups, m_cap, shape_n, shape_k, stream, problem_shapes)

    auto forced = tensorrt_llm::kernels::blockscale_gemm::read_force_tile();
    if (tensorrt_llm::kernels::blockscale_gemm::force_tile_applies(forced, static_cast<uint32_t>(shape_k))
        && forced.tm > 0)
    {
        int const tm = forced.tm, tn = forced.tn, st = forced.st;
        if (tm == 128 && tn == 64 && st == 1) return DISPATCH_SM100_GROUPED(128, 64, 1, 1, 128, false);
        if (tm == 128 && tn == 128 && st == 1) return DISPATCH_SM100_GROUPED(128, 128, 1, 1, 128, false);
        if (tm == 128 && tn == 192 && st == 1) return DISPATCH_SM100_GROUPED(128, 192, 1, 1, 128, false);
        if (tm == 128 && tn == 256 && st == 1) return DISPATCH_SM100_GROUPED(128, 256, 1, 1, 128, false);
        if (tm == 256 && tn == 128 && st == 2) return DISPATCH_SM100_GROUPED(256, 128, 2, 1, 128, false);
        if (tm == 256 && tn == 256 && st == 2) return DISPATCH_SM100_GROUPED(256, 256, 2, 1, 128, false);
        if (tm == 128 && tn == 128 && st == 3) return DISPATCH_SM100_GROUPED(128, 128, 1, 1, 256, false);
        if (tm == 128 && tn == 256 && st == 3) return DISPATCH_SM100_GROUPED(128, 256, 1, 1, 256, false);
        if (tm == 256 && tn == 128 && st == 4) return DISPATCH_SM100_GROUPED(256, 128, 2, 1, 256, false);
        if (tm == 256 && tn == 256 && st == 4) return DISPATCH_SM100_GROUPED(256, 256, 2, 1, 256, false);
        if (tm == 128 && tn == 64 && st == 5) return DISPATCH_SM100_GROUPED(128, 64, 1, 1, 128, true);
        if (tm == 128 && tn == 128 && st == 5) return DISPATCH_SM100_GROUPED(128, 128, 1, 1, 128, true);
        if (tm == 128 && tn == 192 && st == 5) return DISPATCH_SM100_GROUPED(128, 192, 1, 1, 128, true);
        if (tm == 128 && tn == 256 && st == 5) return DISPATCH_SM100_GROUPED(128, 256, 1, 1, 128, true);
        if (tm == 256 && tn == 128 && st == 6) return DISPATCH_SM100_GROUPED(256, 128, 2, 1, 128, true);
        if (tm == 256 && tn == 256 && st == 6) return DISPATCH_SM100_GROUPED(256, 256, 2, 1, 128, true);
        if (tm == 128 && tn == 128 && st == 7) return DISPATCH_SM100_GROUPED(128, 128, 1, 1, 256, true);
        if (tm == 128 && tn == 256 && st == 7) return DISPATCH_SM100_GROUPED(128, 256, 1, 1, 256, true);
        if (tm == 256 && tn == 128 && st == 8) return DISPATCH_SM100_GROUPED(256, 128, 2, 1, 256, true);
        if (tm == 256 && tn == 256 && st == 8) return DISPATCH_SM100_GROUPED(256, 256, 2, 1, 256, true);
        // ST 9 (run b300_mxfp8_20260917/M-A1): the direct store again, but with
        // the EVT partitioning of the CTA tile instead of the builder's
        // default-fusion whole-tile-in-registers form. Same store destination,
        // same bytes; what changes is how many registers the epilogue needs,
        // which is what made the default-fusion direct store collapse at
        // TileN = 256.
        if (tm == 128 && tn == 128 && st == 9) return DISPATCH_SM100_GROUPED_EVT(128, 128, 1, 1, 128);
        if (tm == 128 && tn == 192 && st == 9) return DISPATCH_SM100_GROUPED_EVT(128, 192, 1, 1, 128);
        if (tm == 128 && tn == 256 && st == 9) return DISPATCH_SM100_GROUPED_EVT(128, 256, 1, 1, 128);
        // ST 11: the 1-SM 128-row tile in a TWO-CTA cluster. The cluster's two
        // CTAs hold neighbouring M tiles of the same N tile, so the weight tile
        // is TMA-multicast to both instead of being fetched twice. It is the
        // one axis of the pointer-array family the M-A1 sweep had not touched,
        // and it is aimed at the only band the sweep left open: prefill, where
        // an expert holds more than one 128-row tile and the FC2 therefore
        // streams each expert's weights more than once.
        if (tm == 128 && tn == 192 && st == 11) return DISPATCH_SM100_GROUPED(128, 192, 2, 1, 128, false);
        if (tm == 128 && tn == 256 && st == 11) return DISPATCH_SM100_GROUPED(128, 256, 2, 1, 128, false);
        std::fprintf(stderr,
            "[fso] FSO_FORCE_TILE=%d,%d,%d not wired on the sm_100 grouped path; falling through to cascade\n", tm, tn,
            st);
    }

    switch (grouped_detail::slot_route(m_cap, shape_n, shape_k, groups, max_active_groups, /*fused_swiglu=*/false))
    {
    case grouped_detail::kSlotRouteSlot:
        // The slot list and the grid are sized by the host-static bound on how
        // many groups can hold rows. When the caller hands the list in (it is
        // `moe_build_routing(..., with_slots=True)`'s fourth output, built from
        // the histogram that op already has) the route launches nothing but the
        // GEMM; otherwise it builds the list itself with a one-block kernel, in
        // place of the pointer-array prep kernel, because it needs a compacted
        // expert list rather than eleven per-group arrays.
        return grouped_detail::slot_kernel_launch(mat_a, mat_b, mat_d, scales_a, scales_b, masked_m, groups,
            max_active_groups < groups ? max_active_groups : groups, m_cap, shape_n, shape_k, slot_to_expert,
            stream);
    case grouped_detail::kSlotRouteCascade: break;
    default:
        // A forced illegal configuration. The op raises before reaching here,
        // so this is only the last line of defence against a caller that
        // bypassed it — refuse rather than return a silently wrong answer.
        return cudaErrorInvalidValue;
    }

    // Rules 1-4 in mechanical form, shared with the fused-SwiGLU FC1 route so
    // the two cannot drift apart.
    grouped_detail::CascadePick const pick
        = grouped_detail::cascade_pick(m_cap, shape_n, shape_k, groups, expected_m);
    if (pick.nosmem && pick.evt)
    {
        if (pick.tile_n == 128) return DISPATCH_SM100_GROUPED_EVT(128, 128, 1, 1, 128);
        if (pick.tile_n == 192) return DISPATCH_SM100_GROUPED_EVT(128, 192, 1, 1, 128);
        return DISPATCH_SM100_GROUPED_EVT(128, 256, 1, 1, 128);
    }
    if (pick.nosmem)
    {
        if (pick.tile_n == 128) return DISPATCH_SM100_GROUPED(128, 128, 1, 1, 128, true);
        if (pick.tile_n == 192) return DISPATCH_SM100_GROUPED(128, 192, 1, 1, 128, true);
        return DISPATCH_SM100_GROUPED(128, 256, 1, 1, 128, true);
    }
    if (pick.tile_n == 128) return DISPATCH_SM100_GROUPED(128, 128, 1, 1, 128, false);
    if (pick.tile_n == 192) return DISPATCH_SM100_GROUPED(128, 192, 1, 1, 128, false);
    return DISPATCH_SM100_GROUPED(128, 256, 1, 1, 128, false);

#undef DISPATCH_SM100_GROUPED
#undef DISPATCH_SM100_GROUPED_EVT
}

// ---------------------------------------------------------------------------
// The fused-SwiGLU FC1: one grouped GEMM that also emits MXFP8(silu(gate)*up)
// ---------------------------------------------------------------------------
//
// Same pointer-array geometry, same prep kernel, same masked contract and the
// same argument-array pool as `launch_sm100_mxfp8_grouped_gemm` above; the only
// difference is the epilogue, and with it the outputs: instead of one bf16
// [G, m_cap, N] tensor this writes an fp8 [G, m_cap, N/2] tensor and its 1x32
// UE8M0 scale slab, which is precisely the pair the SECOND grouped GEMM (FC2)
// consumes as its activation operand. The separate SwiGLU-and-requantise kernel
// therefore disappears from the layer.
//
// Precondition the kernel cannot check: the weight rows are gate/up
// INTERLEAVED. See the comment on `Sm100MxFP8GroupedSwiGluGemmConfig`.
template <int TileM, int TileN, int TileK = 128>
cudaError_t launch_sm100_mxfp8_grouped_swiglu_gemm(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b,
    __nv_fp8_e4m3* out_h, int32_t* out_sfh, int32_t* scales_a, int32_t* scales_b, int32_t* masked_m, int groups,
    int m_cap, int shape_n, int shape_k, cudaStream_t stream, int32_t const* problem_shapes = nullptr)
{
    using Config = Sm100MxFP8GroupedSwiGluGemmConfig<TileM, TileN, TileK>;
    using Gemm = typename Config::Gemm;
    using GemmKernel = typename Config::GemmKernel;
    using Args = typename Gemm::Arguments;
    using Params = typename Gemm::Params;
    namespace gd = grouped_detail;

    static_assert(cute::is_same_v<typename Config::InternalStrideA, gd::GStrideA>, "stride type drift (A)");
    static_assert(cute::is_same_v<typename Config::InternalStrideB, gd::GStrideB>, "stride type drift (B)");
    static_assert(cute::is_same_v<typename Config::InternalStrideD, gd::GStrideD>, "stride type drift (D)");
    static_assert(cute::is_same_v<typename Config::InternalLayoutSFA, gd::GLayoutSFA>, "SFA layout type drift");
    static_assert(cute::is_same_v<typename Config::InternalLayoutSFB, gd::GLayoutSFB>, "SFB layout type drift");

    if (groups < 1 || groups > gd::kMaxGroups)
        return cudaErrorInvalidValue;
    // A partial N tile would leave part of an output row unwritten: the store
    // drops any 64-column run that is not wholly inside N, and nothing else
    // writes those columns. The caller's N is always a multiple of 256 (it is
    // 2*I with I a multiple of 128), so this only fires if that contract broke.
    if (shape_n % TileN != 0 || (shape_n / 2) % 128 != 0)
        return cudaErrorInvalidValue;

    if (gd::read_print_tile_info())
    {
        static bool s_info_printed = false;
        if (!s_info_printed)
        {
            s_info_printed = true;
            std::fprintf(stderr,
                "[fso sm100 grouped tile] %dx%dx%d cluster=(1,1) fused-swiglu "
                "stages=%d smem_bytes=%d epi_smem_bytes=%d\n",
                TileM, TileN, TileK, Config::kStages, Config::kSmemBytes, Config::kEpiSmemBytes);
        }
    }

    int64_t const kp = shape_k / 128;
    int64_t const m_pad = (static_cast<int64_t>(m_cap) + 127) / 128 * 128;
    int64_t const inter = shape_n / 2;
    int64_t const a_bytes = static_cast<int64_t>(m_cap) * shape_k;
    int64_t const b_bytes = static_cast<int64_t>(shape_n) * shape_k;
    int64_t const sfa_bytes = m_pad * kp * static_cast<int64_t>(sizeof(int32_t));
    int64_t const sfb_bytes = static_cast<int64_t>(shape_n) * kp * static_cast<int64_t>(sizeof(int32_t));
    // D is declared but never stored through (generator edit 9 deletes the bf16
    // store), so this base pointer and this per-group stride only ever feed
    // address arithmetic that no instruction dereferences. They are set to the
    // real fp8 output slab rather than to a second allocation so that the
    // pointer the epilogue forms for group g is the group's own output base,
    // which keeps the arrays the prep kernel writes meaningful to read in a
    // debugger.
    int64_t const d_bytes = static_cast<int64_t>(m_cap) * inter;

    bool const pdl = gd::grouped_pdl_enabled();

    // Same two paths as the unfused launcher: the caller's shapes plus a
    // key-bound static block, or the prep kernel's slot.
    gd::ArgSlot const* arrays = nullptr;
    gd::GProblemShape* problem = nullptr;
    if (problem_shapes != nullptr && gd::problem_shape_is_three_plain_ints())
    {
        cudaError_t bind_err = cudaSuccess;
        arrays = gd::acquire_static_arrays(stream,
            gd::make_static_key(mat_a, mat_b, scales_a, scales_b, out_h, a_bytes, b_bytes, sfa_bytes, sfb_bytes,
                d_bytes, groups, m_cap, shape_n, shape_k),
            reinterpret_cast<char*>(mat_a), reinterpret_cast<char*>(mat_b), reinterpret_cast<char*>(scales_a),
            reinterpret_cast<char*>(scales_b), reinterpret_cast<char*>(out_h), bind_err);
        if (bind_err != cudaSuccess)
            return bind_err;
        if (arrays != nullptr)
            problem = reinterpret_cast<gd::GProblemShape*>(const_cast<int32_t*>(problem_shapes));
    }
    else if (problem_shapes != nullptr)
    {
        std::fprintf(stderr,
            "[fish_scales_ops] sm_100 fused-SwiGLU grouped MXFP8: cute::Shape<int,int,int> is not three plain "
            "ints on this toolchain; refusing the routing-supplied problem shapes.\n");
        return cudaErrorInvalidValue;
    }
    if (arrays == nullptr)
    {
        auto& pool = gd::ArgPool::instance();
        int const slot_idx = pool.acquire(stream);
        if (slot_idx < 0)
            return cudaErrorMemoryAllocation;
        gd::ArgSlot const& slot = pool.slots[slot_idx];
        int const threads = 32;
        int const blocks = (groups + threads - 1) / threads;
        cudaError_t const prep_err = gd::launch_prep_kernel(dim3(blocks), dim3(threads), stream, pdl, slot.problem,
            slot.ptr_a, slot.ptr_b, slot.ptr_sfa, slot.ptr_sfb, slot.ptr_d, slot.stride_a, slot.stride_b,
            slot.stride_d, slot.layout_sfa, slot.layout_sfb, masked_m, groups, m_cap, shape_n, shape_k,
            reinterpret_cast<char*>(mat_a), reinterpret_cast<char*>(mat_b), reinterpret_cast<char*>(scales_a),
            reinterpret_cast<char*>(scales_b), reinterpret_cast<char*>(out_h), a_bytes, b_bytes, sfa_bytes,
            sfb_bytes, d_bytes);
        if (prep_err != cudaSuccess)
            return prep_err;
        arrays = &slot;
        problem = slot.problem;
    }
    gd::ArgSlot const& slot = *arrays;

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = gd::device_sm_count();

    typename GemmKernel::TileSchedulerArguments scheduler{};

    using MainloopArgs = typename GemmKernel::MainloopArguments;
    using EpilogueArgs = typename GemmKernel::EpilogueArguments;

    MainloopArgs mainloop_args{};
    mainloop_args.ptr_A = reinterpret_cast<decltype(mainloop_args.ptr_A)>(slot.ptr_a);
    mainloop_args.dA = slot.stride_a;
    mainloop_args.ptr_B = reinterpret_cast<decltype(mainloop_args.ptr_B)>(slot.ptr_b);
    mainloop_args.dB = slot.stride_b;
    mainloop_args.ptr_SFA = reinterpret_cast<decltype(mainloop_args.ptr_SFA)>(slot.ptr_sfa);
    mainloop_args.layout_SFA = slot.layout_sfa;
    mainloop_args.ptr_SFB = reinterpret_cast<decltype(mainloop_args.ptr_SFB)>(slot.ptr_sfb);
    mainloop_args.layout_SFB = slot.layout_sfb;

    EpilogueArgs epilogue_args{};
    epilogue_args.thread.alpha = 1.0f;
    epilogue_args.thread.beta = 0.0f;
    epilogue_args.ptr_C = nullptr;
    epilogue_args.dC = nullptr;
    epilogue_args.ptr_D = reinterpret_cast<decltype(epilogue_args.ptr_D)>(slot.ptr_d);
    epilogue_args.dD = slot.stride_d;
    // The two real destinations. They are base pointers plus per-group element
    // counts rather than per-group pointer arrays because the MoE slabs are
    // contiguous in the group index, so the prep kernel does not have to build
    // two more arrays.
    epilogue_args.fused.ptr_h = out_h;
    epilogue_args.fused.ptr_sfh = out_sfh;
    epilogue_args.fused.h_group_elems = static_cast<long long>(m_cap) * inter;
    epilogue_args.fused.sf_group_words = m_pad * (inter / 128);

    Args args{cutlass::gemm::GemmUniversalMode::kGrouped, typename Config::ProblemShape{groups, problem, nullptr},
        mainloop_args, epilogue_args, hw_info, scheduler};

    static thread_local void* s_ws = nullptr;
    static thread_local std::size_t s_ws_bytes = 0;
    std::size_t const need = Gemm::get_workspace_size(args);
    if (need > s_ws_bytes)
    {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &cap);
        if (cap == cudaStreamCaptureStatusActive)
        {
            std::fprintf(stderr,
                "[fish_scales_ops] sm_100 fused-SwiGLU grouped MXFP8: needs %zu workspace bytes during stream "
                "capture but the pool holds %zu. Warm up linear_mxfp8_grouped_masked_swiglu eagerly before "
                "capturing.\n",
                need, s_ws_bytes);
            std::abort();
        }
        if (s_ws)
            cudaFree(s_ws);
        if (cudaMalloc(&s_ws, need) != cudaSuccess)
            return cudaErrorMemoryAllocation;
        s_ws_bytes = need;
        if (GemmKernel::initialize_workspace(args, s_ws, stream) != cutlass::Status::kSuccess)
            return cudaErrorUnknown;
    }

    // Params cache, keyed exactly as the unfused launcher's. The fused
    // destinations DO enter Params (they are epilogue arguments), so unlike the
    // unfused cache the key has to carry them: a layer that ran two different
    // FC1 output buffers through the same instantiation would otherwise replay
    // the first buffer's pointers. They are host-side pointer values, so
    // keying on them costs nothing and keeps the cache correct.
    struct CacheKey
    {
        void const* arrays;
        void const* problem;
        int groups;
        void const* ptr_h;
        void const* ptr_sfh;
        long long h_group_elems;
        long long sf_group_words;
        bool operator==(CacheKey const& o) const { return std::memcmp(this, &o, sizeof(o)) == 0; }
    };
    struct CacheHash
    {
        std::size_t operator()(CacheKey const& k) const noexcept
        {
            std::size_t h = 1469598103934665603ULL;
            auto const* p = reinterpret_cast<unsigned char const*>(&k);
            for (std::size_t i = 0; i < sizeof(k); ++i)
            {
                h ^= p[i];
                h *= 1099511628211ULL;
            }
            return h;
        }
    };
    static thread_local std::unordered_map<CacheKey, Params, CacheHash> s_params_cache;
    CacheKey key;
    std::memset(&key, 0, sizeof(key));
    key.arrays = slot.ptr_a;
    key.problem = problem;
    key.groups = groups;
    key.ptr_h = out_h;
    key.ptr_sfh = out_sfh;
    key.h_group_elems = epilogue_args.fused.h_group_elems;
    key.sf_group_words = epilogue_args.fused.sf_group_words;

    Params kernel_params;
    auto it = s_params_cache.find(key);
    if (it != s_params_cache.end())
    {
        kernel_params = it->second;
    }
    else
    {
        if (Gemm::can_implement(args) != cutlass::Status::kSuccess)
            return cudaErrorInvalidValue;
        kernel_params = GemmKernel::to_underlying_arguments(args, s_ws); // host-only
        if (s_params_cache.size() < 256u)
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

    bool const gemm_pdl = gd::kGroupedGdcCompiled && pdl;
    return Gemm::run(kernel_params, stream, nullptr, gemm_pdl) == cutlass::Status::kSuccess
        ? cudaSuccess
        : cudaErrorLaunchFailure;
}

// Dispatcher for the fused FC1.
//
// The tile comes from the same `cascade_pick` rules 1-4 the ordinary grouped
// GEMM uses, so the fused FC1 runs on the width the cascade was tuned to for
// this (m_cap, N, K, G, expected_m) — the fusion changes the store, not the
// shape of the problem. Three deliberate differences, each explained where it
// is applied below:
//
//   * the epilogue is ALWAYS the direct store, never the TMA bulk store,
//     because the fused store is written against the NoSmem EVT epilogue's
//     fragment. `pick.nosmem` is therefore read and discarded.
//   * a pick of 64 (which no rule produces today, and which only
//     FSO_FORCE_TILE can reach on the unfused path) is promoted to 128,
//     because a 64-wide N tile is a single epilogue subtile: the fused store's
//     per-thread loop would run once, so the tile buys nothing and the fused
//     configuration is not instantiated at that width.
//   * rule 4's `expected_m <= 16` cap does not apply, because that cap exists
//     to stop a wide BF16 direct store from dominating and the fused store
//     writes about a quarter of those bytes.
//
// Whether the layer should call this at all rather than the old pair is a
// separate question, answered by `grouped_detail::fused_fc1_route`, because
// the caller has to choose the SwiGLU kernel to match.
inline cudaError_t gemm_dispatch_sm100_mxfp8_grouped_swiglu(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b,
    __nv_fp8_e4m3* out_h, int32_t* out_sfh, int32_t* scales_a, int32_t* scales_b, int32_t* masked_m, int groups,
    int m_cap, int shape_n, int shape_k, int expected_m, int max_active_groups, int const* slot_to_expert,
    int32_t const* problem_shapes, cudaStream_t stream)
{
#define DISPATCH_SM100_GROUPED_SWIGLU(TN_)                                                                            \
    launch_sm100_mxfp8_grouped_swiglu_gemm<128, TN_, 128>(mat_a, mat_b, out_h, out_sfh, scales_a, scales_b,         \
        masked_m, groups, m_cap, shape_n, shape_k, stream, problem_shapes)

    // The route first, the tile second. `slot_route` is the SAME function the
    // unfused FC1 asks, so the fused FC1 changes which kernel a call lands on
    // in exactly the places the unfused one does; the fusion is a property of
    // the epilogue, not of the routing. On the decode band that means the
    // swap-orientation slot kernel with the fused epilogue, which is the one
    // path where the separate SwiGLU kernel used to survive the fusion (run
    // b300_mxfp8_20260917/M-A2).
    //
    // FSO_FORCE_TILE is an N-tile knob of the pointer-array cascade and says
    // nothing about the slot route, so it is not consulted here; FSO_GROUPED_SLOT
    // is the knob that steers this decision, through `slot_route` itself. The
    // verdict is asked for the FUSED kernel, whose clause stops at one epilogue
    // chunk (m_cap <= 32) where the plain kernel's runs to the full tile (run
    // b300_round3_20260922/M-A3).
    if (grouped_detail::slot_route(m_cap, shape_n, shape_k, groups, max_active_groups, /*fused_swiglu=*/true)
            == grouped_detail::kSlotRouteSlot
        && grouped_detail::slot_swiglu_kernel_instantiated(shape_n, shape_k))
    {
        return grouped_detail::slot_swiglu_kernel_launch(mat_a, mat_b, out_h, out_sfh, scales_a, scales_b, masked_m,
            groups, max_active_groups < groups ? max_active_groups : groups, m_cap, shape_n, shape_k, slot_to_expert,
            stream);
    }

    // FSO_FORCE_TILE stays the A/B knob it is on the unfused path: the TN field
    // names the N-tile width and the ST field is ignored here, because the
    // fused route has exactly one epilogue and one TileK.
    auto forced = tensorrt_llm::kernels::blockscale_gemm::read_force_tile();
    if (tensorrt_llm::kernels::blockscale_gemm::force_tile_applies(forced, static_cast<uint32_t>(shape_k))
        && forced.tm == 128 && shape_n % forced.tn == 0)
    {
        if (forced.tn == 128) return DISPATCH_SM100_GROUPED_SWIGLU(128);
        if (forced.tn == 192) return DISPATCH_SM100_GROUPED_SWIGLU(192);
        if (forced.tn == 256) return DISPATCH_SM100_GROUPED_SWIGLU(256);
        std::fprintf(stderr,
            "[fso] FSO_FORCE_TILE TN=%d is not instantiated on the sm_100 fused-SwiGLU FC1 path; falling through "
            "to the cascade\n",
            forced.tn);
    }

    grouped_detail::CascadePick const pick
        = grouped_detail::cascade_pick(m_cap, shape_n, shape_k, groups, expected_m);
    int tile_n = pick.tile_n < 128 ? 128 : pick.tile_n;

    // The one place the fused route deviates from the cascade, and it is a
    // consequence of the fusion rather than a fit to a shape.
    //
    // Rule 4 prefers the 192-wide N tile over the 256-wide one whenever 192
    // divides N and the 192-wide grid's last wave is fuller — pure wave
    // arithmetic against the SM count — but it caps that at
    // `expected_m <= 16`. The cap is there because rule 4's 192-wide tile is
    // paired with the DIRECT store, and a direct store of a bf16
    // [m_cap, 2*I] tile stops being a sliver once the rows pile up: past
    // expected_m = 16 the unfused configuration turned into a 4 per cent loss
    // at 64 and a 30 per cent loss at 128.
    //
    // The fused store writes an fp8 [m_cap, I] tile plus its scale bytes,
    // which is about a quarter of those bytes, so the premise of the cap does
    // not hold on this route and the wave arithmetic should be allowed to
    // decide at every expected_m. Measured on the layer cell (run
    // b300_mxfp8_20260917/M-E2, TABLES_tileprobe.txt, two passes): on Family B
    // gate_up, where 192 divides N = 1536 and the 192-wide grid's tail wave is
    // 92 per cent full against the 256-wide grid's 19 per cent, forcing 192
    // above the cap is 3.9 us faster at M = 512, 4.7 at M = 1024 and 2.2 at
    // M = 2048, and a wash (-0.3 us, 0.1 per cent) at M = 4096. Family C's
    // N = 1024 is not a multiple of 192, so the clause cannot fire there and
    // that family is untouched.
    if (tile_n == 256 && shape_n % 192 == 0
        && grouped_detail::tail_fill_q10(shape_n, 192, groups)
            > grouped_detail::tail_fill_q10(shape_n, 256, groups))
        tile_n = 192;

    // A width that does not divide N would leave the last tile's columns
    // unwritten; 128 divides every legal N (= 2*I with I a multiple of 128).
    if (shape_n % tile_n != 0)
        tile_n = 128;
    if (tile_n == 192) return DISPATCH_SM100_GROUPED_SWIGLU(192);
    if (tile_n == 256) return DISPATCH_SM100_GROUPED_SWIGLU(256);
    return DISPATCH_SM100_GROUPED_SWIGLU(128);

#undef DISPATCH_SM100_GROUPED_SWIGLU
}

} // namespace sm100_blockscaled_gemm

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
