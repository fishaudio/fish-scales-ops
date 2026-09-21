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
// of them would only enlarge the binary.

#pragma once

#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#include "blockscale_gemm/arch/sm100/mxfp8/grouped_gemm_types.cuh"
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
    int acquire()
    {
        if (base == nullptr)
        {
            cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(nullptr, &cap);
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

} // namespace grouped_detail

// Launch one (TileM, TileN, ClusterM, ClusterN, TileK, NoSmemEpi)
// instantiation of the grouped kernel.
template <int TileM, int TileN, int ClusterM, int ClusterN, int TileK = 128, bool NoSmemEpi = false>
cudaError_t launch_sm100_mxfp8_grouped_gemm(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int32_t* masked_m, int groups, int m_cap, int shape_n, int shape_k,
    cudaStream_t stream, int swizzle = 0)
{
    using Config = Sm100MxFP8GroupedGemmConfig<TileM, TileN, ClusterM, ClusterN, TileK, NoSmemEpi>;
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

    auto& pool = gd::ArgPool::instance();
    int const slot_idx = pool.acquire();
    if (slot_idx < 0)
        return cudaErrorMemoryAllocation;
    gd::ArgSlot const& slot = pool.slots[slot_idx];

    // Per-group byte strides of the caller's packed slabs.
    int64_t const kp = shape_k / 128;
    int64_t const m_pad = (static_cast<int64_t>(m_cap) + 127) / 128 * 128;
    int64_t const a_bytes = static_cast<int64_t>(m_cap) * shape_k;                 // fp8
    int64_t const b_bytes = static_cast<int64_t>(shape_n) * shape_k;               // fp8
    int64_t const sfa_bytes = m_pad * kp * static_cast<int64_t>(sizeof(int32_t));  // atom slab
    int64_t const sfb_bytes = static_cast<int64_t>(shape_n) * kp * static_cast<int64_t>(sizeof(int32_t));
    int64_t const d_bytes = static_cast<int64_t>(m_cap) * shape_n * static_cast<int64_t>(sizeof(__nv_bfloat16));

    bool const pdl = gd::grouped_pdl_enabled();
    {
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
    }

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

    Args args{cutlass::gemm::GemmUniversalMode::kGrouped,
        typename Config::ProblemShape{groups, slot.problem, nullptr}, mainloop_args, epilogue_args, hw_info,
        scheduler};

    // Workspace: per-SM tensormap scratch. Its size depends only on the SM
    // count, so it is allocated once per instantiation and never grows — which
    // is what lets the Params cache hold a stable pointer.
    static thread_local void* s_ws = nullptr;
    static thread_local std::size_t s_ws_bytes = 0;
    std::size_t const need = Gemm::get_workspace_size(args);
    if (need > s_ws_bytes)
    {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(nullptr, &cap);
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
    // pointers, the group count and the scheduler knobs — never on
    // (m_cap, N, K).
    struct CacheKey
    {
        int slot;
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
    key.slot = slot_idx;
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
        if (s_params_cache.size() < 64u)
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
    // keeps the ordering against the prep kernel, while the CTAs, their cluster
    // launch and the per-group tensormap initialisation start while the prep
    // kernel is still running.
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
// Rule 3 — the direct-store (NoSmem) epilogue wins exactly while the store is
// a sliver. The epilogue writes a 128-row tile but only expected_m of those
// rows are inside the problem, so the store volume grows with expected_m while
// the mainloop does not, and the shorter the mainloop the sooner the store
// dominates. That is why the cut-off is K-dependent: at K = 512 (4 K-tiles)
// the direct store is 4-9 % faster all the way to expected_m = 16 and only
// turns into a 14 % loss at expected_m = 64, while at K = 768 (6 K-tiles) it
// is ahead by 0.6-1.7 % at expected_m = 1 and already 1.3-1.7 % behind at
// expected_m = 4. On the long-K route the 256-wide tile with a direct store was
// the worst config measured anywhere (+9 to +46 %), so the wide route keeps the
// TMA store; rule 4 below is the one narrow exception.
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
inline cudaError_t gemm_dispatch_sm100_mxfp8_grouped(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int32_t* masked_m, int groups, int m_cap, int shape_n, int shape_k,
    int expected_m, cudaStream_t stream)
{
#define DISPATCH_SM100_GROUPED(TM_, TN_, CM_, CN_, TK_, NOSMEM_)                                                       \
    launch_sm100_mxfp8_grouped_gemm<TM_, TN_, CM_, CN_, TK_, NOSMEM_>(                                                 \
        mat_a, mat_b, mat_d, scales_a, scales_b, masked_m, groups, m_cap, shape_n, shape_k, stream)

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
        std::fprintf(stderr,
            "[fso] FSO_FORCE_TILE=%d,%d,%d not wired on the sm_100 grouped path; falling through to cascade\n", tm, tn,
            st);
    }

    int const em = expected_m < 1 ? 1 : expected_m;

    // Rule 4's wave arithmetic: how full is the last wave of the grid a given
    // N-tile width produces? `groups * ceil(N / tile_n)` is the tile count with
    // every expert active (an upper bound on the real, routing-dependent count);
    // the grouped scheduler runs one CTA per SM, so the remainder against the SM
    // count is the tail wave's width. Returned scaled by 1024 to stay integral.
    auto tail_fill_q10 = [&](int tile_n) -> int
    {
        int const sms = grouped_detail::device_sm_count();
        long long const tiles = static_cast<long long>(groups) * ((shape_n + tile_n - 1) / tile_n);
        long long const tail = tiles % sms;
        return tail == 0 ? 1024 : static_cast<int>(tail * 1024 / sms);
    };

    // Long mainloop: the 256-wide N tile with the TMA bulk-store epilogue,
    // except at a tiny row capacity where the tile count matters more, and
    // except where the 192-wide tile fills the last wave better (rule 4).
    if (shape_k >= 1024)
    {
        if (m_cap <= 4)
            return DISPATCH_SM100_GROUPED(128, 128, 1, 1, 128, false);
        if (em <= 16 && shape_n % 192 == 0 && tail_fill_q10(192) > tail_fill_q10(256))
            return DISPATCH_SM100_GROUPED(128, 192, 1, 1, 128, true);
        return DISPATCH_SM100_GROUPED(128, 256, 1, 1, 128, false);
    }
    // Short mainloop, sliver store: 128-wide N tile with the direct store. The
    // shorter the mainloop, the further up expected_m the direct store stays
    // ahead.
    int const nosmem_em_max = (shape_k <= 512) ? 16 : 1;
    if (em <= nosmem_em_max)
        return DISPATCH_SM100_GROUPED(128, 128, 1, 1, 128, true);
    // Short mainloop, full store: back to the 256-wide tile and the TMA store.
    return DISPATCH_SM100_GROUPED(128, 256, 1, 1, 128, false);

#undef DISPATCH_SM100_GROUPED
}

} // namespace sm100_blockscaled_gemm

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
