// MXFP8 paged-KV decode v3 — cross-batch packed CTA.
//
// Packing structure: 4 warps per CTA, each warp serves a different
// (b, h_kv) work unit independently. Per-warp K/V/P smem slabs; per-warp
// cp.async pipeline; per-warp QK / softmax / PV. The CTA has no
// cross-warp coordination — warps act like 4 mini-CTAs.
//
// Compared to v1 MVP (1 work unit per CTA, 4 warps along M with only
// gqa_group / 64 useful M-rows):
//   - v1's mma issues 4 warps × 16 M-rows = 64 M-rows per atom, but only
//     gqa_group are real (e.g. 4/64 = 6.25 % for LLaMA-3 8B gqa=4).
//   - v3's mma issues 4 warps × 16 M-rows = 64 M-rows per atom,
//     each warp owns gqa_group real (= up to 16 useful per warp).
//     For gqa_group=4: 16/16 useful per warp × 4 warps = 64 useful
//     per CTA, vs 4 useful per CTA in v1. 16× better useful-mma /
//     atom issued.
//
// Same per-D Bc as v1 wouldn't fit (4 × 33 KB > 99 KB cap), so v3
// uses Bc=32 across all Ds. This means each kv-tile is 32 K-tokens
// (half a v1 page for D∈{64,128}, quarter for D=32). The kernel reads
// from the same KV cache page layout the caller built for v1 — each
// "v3 kv-tile" corresponds to half a page (or quarter for D=32).
//
// Per-D tile config:
//   D = 32  : Br=64, Bc=32, kStages=2, kCtasPerSm=2  (smem ~18 KB/CTA)
//   D = 64  : Br=64, Bc=32, kStages=2, kCtasPerSm=2  (smem ~34 KB/CTA)
//   D = 128 : Br=64, Bc=32, kStages=2, kCtasPerSm=1  (smem ~66 KB/CTA)
//
// Public ABI mirrors v1 (same partial-output scratch + reduce path)
// so the wrapper can route v1 ↔ v3 per shape without changing the
// caller's tensor layouts.
//
// Caveats:
//   - gqa_group must fit in M=16 (1 warp's M-rows). Most modern models
//     satisfy this; for gqa_group > 16 fall back to v1.
//   - page_size is still 64 for D∈{64,128} (caller-side layout); v3
//     internally reads half a page per kv-tile via a (kv_tile/2, kv_tile%2)
//     index split. For D=32 the page_size is 128 (caller side) so each
//     kv-tile is a quarter-page.
//   - Each warp issues its own cp.async — no benefit from CTA-wide
//     coalescing. The trade-off is independent work units.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <math.h>
#include <cstdint>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/atom/copy_atom.hpp>

namespace flash_attn_sm120_decode_paged {

using namespace cute;

using ElementA   = float_e4m3_t;
using ElementB   = float_e4m3_t;
using ElementSF  = float_ue8m0_t;
using ElementAcc = float;
using ElementOut = bfloat16_t;

constexpr int kBlock3 = 32;            // K-block size for UE8M0 scales (sf_vec_size)

using MmaAtom = MMA_Atom<SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    ElementA, ElementB, ElementAcc, ElementSF, /*sf_vec_size=*/32>>;

// 1-warp TiledMma — each warp uses TiledMma<1,1,1> on its OWN smem region.
using TiledMma = TiledMMA<MmaAtom, Layout<Shape<_1, _1, _1>>>;

template <int K> struct SwizzledKAtom;
template <> struct SwizzledKAtom< 32> { using type = GMMA::Layout_K_SW32_Atom<ElementA>; };
template <> struct SwizzledKAtom< 64> { using type = GMMA::Layout_K_SW64_Atom<ElementA>; };
template <> struct SwizzledKAtom<128> { using type = GMMA::Layout_K_SW128_Atom<ElementA>; };
template <> struct SwizzledKAtom<256> { using type = GMMA::Layout_K_SW128_Atom<ElementA>; };

template <int kBc, int kD, int kStages>
struct SmemLayouts {
    using AtomK = typename SwizzledKAtom<kD>::type;
    using K = decltype(tile_to_shape(
        AtomK{},
        make_shape(Int<kBc>{}, Int<kD>{}, Int<kStages>{}),
        Step<_1, _2, _3>{}));

    using AtomV = typename SwizzledKAtom<kBc>::type;
    using V = decltype(tile_to_shape(
        AtomV{},
        make_shape(Int<kD>{}, Int<kBc>{}, Int<kStages>{}),
        Step<_1, _2, _3>{}));

    using AtomP = typename SwizzledKAtom<kBc>::type;
    using P = decltype(tile_to_shape(
        AtomP{},
        make_shape(Int<16>{}, Int<kBc>{}),                  // M=16 (1 warp), N=kBc
        Step<_1, _2>{}));
};

using GmemCopyAtomAB = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, ElementA>;
using SmemCopyAtomA  = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
using SmemCopyAtomB  = Copy_Atom<SM75_U32x2_LDSM_N, ElementB>;

__device__ __forceinline__ uint32_t smem_to_uint(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// v19 micro-opt: pack 2 fp32 → 2 fp8 (e4m3) in one PTX. Paired with STS.U16
// in the P-quant inner loop to halve conversion + store counts.
__device__ __forceinline__ uint16_t cvt_e4m3x2_fp32(float a, float b) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890)
    uint16_t r;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\n"
                 : "=h"(r) : "f"(a), "f"(b));
    return r;
#else
    auto a8 = (uint8_t)(uint32_t)cute::float_e4m3_t(a);
    auto b8 = (uint8_t)(uint32_t)cute::float_e4m3_t(b);
    return uint16_t(a8) | (uint16_t(b8) << 8);
#endif
}

// Pack 2 fp32 → 2 bf16 in one PTX. Paired with STG.B32 in the epilogue
// to halve fp32→bf16 conversions and global stores.
__device__ __forceinline__ uint32_t cvt_bf16x2_fp32(float a, float b) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
    uint32_t r;
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;\n"
                 : "=r"(r) : "f"(a), "f"(b));
    return r;
#else
    uint16_t a16 = __bfloat16_as_short(__float2bfloat16(a));
    uint16_t b16 = __bfloat16_as_short(__float2bfloat16(b));
    return uint32_t(a16) | (uint32_t(b16) << 16);
#endif
}

__device__ __forceinline__ void mma_mxf8(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint8_t  sa, uint8_t  sb)
{
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 1200)
    asm volatile(
        "{\n"
        "  .reg .b32 sa, sb;\n"
        "  cvt.u32.u8 sa, %10;\n"
        "  cvt.u32.u8 sb, %11;\n"
        "  mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale"
                ".scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0\n"
        "    {%0,%1,%2,%3},\n"
        "    {%4,%5,%6,%7},\n"
        "    {%8,%9},\n"
        "    {%0,%1,%2,%3},\n"
        "    sa, {0,0},\n"
        "    sb, {0,0};\n"
        "}\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "h"((unsigned short)sa),
          "h"((unsigned short)sb));
#else
    (void)d0; (void)d1; (void)d2; (void)d3;
    (void)a0; (void)a1; (void)a2; (void)a3;
    (void)b0; (void)b1; (void)sa; (void)sb;
#endif
}

// Regular (non-block-scaled) FP8 mma m16n8k32. Used for PV in the
// "channel-V" design where V's quantisation lives in a per-(h_kv, D)
// fp32 scale tensor outside the paged cache — see ::decode_paged_kernel
// epilogue. Output is in *quant* units; caller applies the post-scale
// FFMA when writing O.
__device__ __forceinline__ void mma_fp8_plain(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 1200)
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\n"
        "    {%0,%1,%2,%3},\n"
        "    {%4,%5,%6,%7},\n"
        "    {%8,%9},\n"
        "    {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
#else
    (void)d0; (void)d1; (void)d2; (void)d3;
    (void)a0; (void)a1; (void)a2; (void)a3;
    (void)b0; (void)b1;
#endif
}

// ---------- v3 kernel ------------------------------------------------------
//
// page_bc_ratio = page_size / kBc — how many v3 kv-tiles fit in one caller
// page. Runtime parameter. Caller's page_size MUST satisfy
// page_size % kBc == 0 for K (block-scaled). With the "channel-V" design
// V no longer has a per-page scale layout, so V tolerates any page_size
// (1, 16, 32, 64, ...) — the practical lower bound is set by K's
// sf_vec_size=32. To support sglang page_size=1, either drop K block_scale
// too or pad K scale storage to 32-token groups internally.
// The compiler turns the division/modulo by page_bc_ratio into shift+mask
// for power-of-2 values.

template <int kHeadDim, int kBc, int kStages, int kCtasPerSm>
__global__
__launch_bounds__(4 * 32, kCtasPerSm)
void decode_paged_kernel(
    const __nv_fp8_e4m3* __restrict__ Q,
    const uint8_t*       __restrict__ Qs,
    const __nv_fp8_e4m3* __restrict__ K_pool,
    const uint8_t*       __restrict__ K_chan_scale,   // [H_kv, D/32] UE8M0 — channel-K scale
    const __nv_fp8_e4m3* __restrict__ V_pool,
    const float*         __restrict__ V_chan_scale,   // [H_kv, D] fp32 — channel-V scale
    const int32_t*       __restrict__ block_table,
    const int32_t*       __restrict__ seq_lens,
    __nv_bfloat16*       __restrict__ O,
    float*               __restrict__ M_partial,
    float*               __restrict__ L_partial,
    float*               __restrict__ O_partial,
    int32_t*             __restrict__ sync_counter,    // [B, H_q] int32, persistent or zeroed
    int batch, int num_q_heads, int num_kv_heads,
    int max_blocks, int num_splits, int target_counter,
    int page_size,
    float softmax_scale)
{
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 1200)
    using Layouts = SmemLayouts<kBc, kHeadDim, kStages>;
    using SmemLayoutK = typename Layouts::K;
    using SmemLayoutV = typename Layouts::V;
    using SmemLayoutP = typename Layouts::P;

    constexpr int kD             = kHeadDim;
    constexpr int kNumWarps      = 4;
    constexpr int kThreads       = kNumWarps * 32;
    constexpr int n_kblk         = kD / kBlock3;
    constexpr int n_v_kblk_per_tile = kBc / kBlock3;        // = 1 for kBc=32
    constexpr int n_qk_n_tiles   = kBc / 8;
    constexpr int n_pv_n_tiles   = kD / 8;
    // K and V both block-scale-free; K_chan_scale[h_kv, D/32] is loaded once
    // into registers and reused across every kv-tile. page_size is the
    // caller's logical block granularity (independent of kBc).
    const int page_bc_ratio      = page_size / kBc;          // ≥ 1 when page_size ≥ kBc
    const int kPageSize          = page_size;                // alias for legacy lambda code

    const int tid    = threadIdx.x;
    const int warp   = tid / 32;
    const int lane   = tid % 32;
    const int row    = lane / 4;
    const int tid_g  = lane % 4;

    // Total work units = batch × num_kv_heads. Each CTA serves 4 of them
    // (one per warp). Splits along kv-tile axis stack in grid.y.
    const int wu_per_cta   = kNumWarps;
    const int total_wu     = batch * num_kv_heads;
    const int packed_idx   = blockIdx.x;
    const int split_id     = blockIdx.y;
    const int wu_id        = packed_idx * wu_per_cta + warp;
    if (wu_id >= total_wu) return;
    if (split_id >= num_splits) return;

    const int b    = wu_id / num_kv_heads;
    const int h_kv = wu_id % num_kv_heads;
    const int seq_len   = seq_lens[b];
    const int gqa_group = num_q_heads / num_kv_heads;

    // Don't early-return on empty/zero-seq: the fused reduce needs EVERY
    // CTA's atomic to count toward num_splits. Empty/zero work units just
    // skip the prologue + kv-tile loop and fall through to the epilogue
    // with neutral pv_acc/m/l.
    const int n_kv_tiles_total = (seq_len <= 0) ? 0 : (seq_len + kBc - 1) / kBc;
    int       kv_tile_start    = 0;
    int       kv_tile_end      = 0;
    int       n_kv_tiles       = 0;
    if (n_kv_tiles_total > 0) {
        const int tiles_per_split = (n_kv_tiles_total + num_splits - 1) / num_splits;
        kv_tile_start = split_id * tiles_per_split;
        kv_tile_end   = kv_tile_start + tiles_per_split;
        if (kv_tile_end > n_kv_tiles_total) kv_tile_end = n_kv_tiles_total;
        if (kv_tile_start > n_kv_tiles_total) kv_tile_start = n_kv_tiles_total;
        n_kv_tiles = kv_tile_end - kv_tile_start;
    }

    // ---------- Per-warp smem slabs ---------------------------------------
    //
    // Each warp's K / V / P slabs must satisfy the swizzle base-alignment
    // requirement (Layout_K_SW128_Atom uses Swizzle<3,4,3> which asserts
    // the base pointer's low (B+M+S=10) bits are zero, i.e. 1024-byte
    // aligned). The simplest fix is to allocate all four warps' K slabs
    // contiguously (so each warp's K starts at warp*K_size, which is
    // ≥1024 aligned since each K is ≥1024 bytes itself and a multiple
    // thereof for the shapes we use), then V slabs, then P, then scales.
    extern __shared__ unsigned char smem_raw[];
    constexpr int kKWarpBytes  = sizeof(ElementA) * (int)cosize(SmemLayoutK{});
    constexpr int kVWarpBytes  = sizeof(ElementA) * (int)cosize(SmemLayoutV{});
    constexpr int kPWarpBytes  = sizeof(ElementA) * (int)cosize(SmemLayoutP{});
    // K and V scales are both channel tensors loaded into registers below.
    // No per-tile scale smem slab is needed.

    auto* smem_K_all  = reinterpret_cast<ElementA*>(smem_raw);
    auto* smem_V_all  = smem_K_all  + 4 * cosize(SmemLayoutK{});
    auto* smem_P_all  = smem_V_all  + 4 * cosize(SmemLayoutV{});

    auto* smem_K_ptr  = smem_K_all  + warp * cosize(SmemLayoutK{});
    auto* smem_V_ptr  = smem_V_all  + warp * cosize(SmemLayoutV{});
    auto* smem_P_ptr  = smem_P_all  + warp * cosize(SmemLayoutP{});

    constexpr auto layoutK = SmemLayoutK{};
    constexpr auto layoutV = SmemLayoutV{};
    constexpr auto layoutP = SmemLayoutP{};

    Tensor sK_pi = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(smem_K_ptr), SmemLayoutK{}));
    Tensor sV_pi = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(smem_V_ptr), SmemLayoutV{}));
    Tensor sP_pi = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(smem_P_ptr), SmemLayoutP{}));

    TiledMma mma;
    auto thr_mma = mma.get_thread_slice(lane);

    auto s2r_copy_K = make_tiled_copy_B(SmemCopyAtomB{}, mma);
    auto thr_s2r_K  = s2r_copy_K.get_thread_slice(lane);
    auto tCrK = thr_mma.partition_fragment_B(sK_pi(_, _, Int<0>{}));
    auto tXsK = thr_s2r_K.partition_S(sK_pi);
    auto tXrK = thr_s2r_K.retile_D(tCrK);

    auto s2r_copy_V = make_tiled_copy_B(SmemCopyAtomB{}, mma);
    auto thr_s2r_V  = s2r_copy_V.get_thread_slice(lane);
    auto tCrV = thr_mma.partition_fragment_B(sV_pi(_, _, Int<0>{}));
    auto tXsV = thr_s2r_V.partition_S(sV_pi);
    auto tXrV = thr_s2r_V.retile_D(tCrV);

    auto s2r_copy_P = make_tiled_copy_A(SmemCopyAtomA{}, mma);
    auto thr_s2r_P  = s2r_copy_P.get_thread_slice(lane);
    auto tCrP = thr_mma.partition_fragment_A(sP_pi);
    auto tXsP = thr_s2r_P.partition_S(sP_pi);
    auto tXrP = thr_s2r_P.retile_D(tCrP);

    // ---------- Q frag (decode-shaped, packed into 1 warp's M=16) ---------
    const int m_row_lo = row + 0;
    const int m_row_hi = row + 8;
    const bool lo_valid = (m_row_lo < gqa_group);
    const bool hi_valid = (m_row_hi < gqa_group);

    const __nv_fp8_e4m3* Q_base = Q + ((int64_t)b * num_q_heads) * kD;
    const uint8_t* Qs_base = Qs + ((int64_t)b * num_kv_heads + h_kv) * n_kblk;

    uint32_t qfrag[n_kblk][4];
    #pragma unroll
    for (int kb = 0; kb < n_kblk; ++kb) {
        const int k0 = kb * kBlock3;
        const int a_col0 = tid_g * 4 + k0;
        const int a_col1 = a_col0 + 16;
        const int h_q_lo = h_kv * gqa_group + m_row_lo;
        const int h_q_hi = h_kv * gqa_group + m_row_hi;
        qfrag[kb][0] = lo_valid
            ? *reinterpret_cast<const uint32_t*>(Q_base + (int64_t)h_q_lo * kD + a_col0) : 0u;
        qfrag[kb][1] = hi_valid
            ? *reinterpret_cast<const uint32_t*>(Q_base + (int64_t)h_q_hi * kD + a_col0) : 0u;
        qfrag[kb][2] = lo_valid
            ? *reinterpret_cast<const uint32_t*>(Q_base + (int64_t)h_q_lo * kD + a_col1) : 0u;
        qfrag[kb][3] = hi_valid
            ? *reinterpret_cast<const uint32_t*>(Q_base + (int64_t)h_q_hi * kD + a_col1) : 0u;
    }
    uint8_t qs[n_kblk];
    #pragma unroll
    for (int kb = 0; kb < n_kblk; ++kb) qs[kb] = Qs_base[kb];

    // ---------- K channel-scale (one UE8M0 byte per (h_kv, D-block)) -------
    // Constant across the entire kv-tile loop (block_scale operand for QK
    // mma). Loaded once into registers; replaces the per-tile Ks_pool slab.
    uint8_t ks[n_kblk];
    {
        const uint8_t* Ksc_h = K_chan_scale + (int64_t)h_kv * n_kblk;
        #pragma unroll
        for (int kb = 0; kb < n_kblk; ++kb) ks[kb] = Ksc_h[kb];
    }

    // ---------- V channel-scale (one fp32 per (h_kv, d)) -------------------
    //
    // Per-lane preload of the 2*n_pv_n_tiles V scales this lane will
    // multiply into pv_acc at the epilogue. The factor 1/256 folds in
    // here (P_quant = P_real * 256, so output picks up 1/256 to get back
    // to real units).
    constexpr float kInvP = 1.0f / 256.0f;
    float vsc[n_pv_n_tiles][2];
    {
        const float* Vsc_h = V_chan_scale + (int64_t)h_kv * kD;
        #pragma unroll
        for (int nt = 0; nt < n_pv_n_tiles; ++nt) {
            const int n_base = nt * 8 + tid_g * 2;
            vsc[nt][0] = Vsc_h[n_base + 0] * kInvP;
            vsc[nt][1] = Vsc_h[n_base + 1] * kInvP;
        }
    }

    float m_lo = -INFINITY, m_hi = -INFINITY;
    float l_lo = 0.f, l_hi = 0.f;
    float pv_acc[n_pv_n_tiles][4];
    #pragma unroll
    for (int nt = 0; nt < n_pv_n_tiles; ++nt) {
        pv_acc[nt][0] = pv_acc[nt][1] = pv_acc[nt][2] = pv_acc[nt][3] = 0.f;
    }

    // ---------- Issue cp.async lambdas — page_size ≥ kBc only -------------
    //
    // page_size must be a multiple of kBc=32 (= MXFP8 sf_vec_size). One page
    // covers ≥1 kv-tile; the per-tile page lookup hoists out of the inner
    // loop. The page_size < 32 slow path was removed for code clarity — see
    // git history for the gather variant if a future revisit is needed.

    auto issue_cp_async_K = [&](int stage, int kv_tile) {
        const int page_outer = kv_tile / page_bc_ratio;
        const int page_inner = kv_tile - page_outer * page_bc_ratio;
        const int32_t slot = block_table[(int64_t)b * max_blocks + page_outer];
        const auto* K_page = K_pool
            + (int64_t)slot * kPageSize * num_kv_heads * kD
            + (int64_t)page_inner * kBc * num_kv_heads * kD
            + (int64_t)h_kv * kD;
        const int kv_row_stride = num_kv_heads * kD;
        constexpr int kOps = (kBc * kD) / 16;
        constexpr int kChunksPerRow = kD / 16;
        #pragma unroll
        for (int i = lane; i < kOps; i += 32) {
            const int s_idx  = i / kChunksPerRow;
            const int d_offs = (i % kChunksPerRow) * 16;
            const auto* src = K_page + (int64_t)s_idx * kv_row_stride + d_offs;
            int off = layoutK(s_idx, d_offs, stage);
            uint32_t dst = smem_to_uint(smem_K_ptr + off);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                          :: "r"(dst), "l"(src));
        }
    };

    auto issue_cp_async_V = [&](int stage, int kv_tile) {
        const int page_outer = kv_tile / page_bc_ratio;
        const int page_inner = kv_tile - page_outer * page_bc_ratio;
        const int32_t page_idx = block_table[(int64_t)b * max_blocks + page_outer];
        const auto* V_page = V_pool
            + (int64_t)page_idx * kD * num_kv_heads * kPageSize
            + (int64_t)h_kv * kPageSize
            + (int64_t)page_inner * kBc;
        const int v_d_stride = num_kv_heads * kPageSize;
        constexpr int kOps = (kD * kBc) / 16;
        constexpr int kChunksPerRow = kBc / 16;
        #pragma unroll
        for (int i = lane; i < kOps; i += 32) {
            const int d_idx    = i / kChunksPerRow;
            const int s_offset = (i % kChunksPerRow) * 16;
            const auto* src = V_page + (int64_t)d_idx * v_d_stride + s_offset;
            int off = layoutV(d_idx, s_offset, stage);
            uint32_t dst = smem_to_uint(smem_V_ptr + off);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                          :: "r"(dst), "l"(src));
        }
    };

    // K and V scales are channel tensors loaded once into registers above
    // (ks[kb], vsc[nt][i]) — no per-tile scale load lambda needed.
    auto load_page = [&](int stage, int kv_tile) {
        issue_cp_async_K(stage, kv_tile);
        issue_cp_async_V(stage, kv_tile);
    };

    // ---------- Prologue: prefetch first stage -----------------------------
    // kStages=2: ping-pong (curr/next swap). kStages=1: single-buffer
    // (D=256 path) — cp.async fully serial, no overlap.
    int curr = 0, next = (kStages > 1) ? 1 : 0;
    if (n_kv_tiles > 0) {
        load_page(0, kv_tile_start);
        cp_async_commit_group();
    }

    // ---------- KV tile loop (per-warp) ------------------------------------
    // Empty splits (n_kv_tiles == 0) skip this loop entirely.
    for (int iter = 0; iter < n_kv_tiles; ++iter) {
        const int kv_tile = kv_tile_start + iter;

        if constexpr (kStages == 1) {
            cp_async_wait_group<0>();
        } else {
            if (iter + 1 < n_kv_tiles) {
                load_page(next, kv_tile + 1);
                cp_async_commit_group();
                cp_async_wait_group<1>();
            } else {
                cp_async_wait_group<0>();
            }
        }
        __syncwarp();        // warp-scoped — other warps are independent

        cute::copy(s2r_copy_K, tXsK(_, _, _, curr), tXrK);
        auto tCrK_u32 = recast<uint32_t>(tCrK);

        // ---------- QK ----------------------------------------------------
        float s_frag[n_qk_n_tiles][4];
        #pragma unroll
        for (int nt = 0; nt < n_qk_n_tiles; ++nt) {
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
            #pragma unroll
            for (int kb = 0; kb < n_kblk; ++kb) {
                uint32_t b0 = tCrK_u32(0, nt, kb);
                uint32_t b1 = tCrK_u32(1, nt, kb);
                mma_mxf8(d0, d1, d2, d3,
                         qfrag[kb][0], qfrag[kb][1], qfrag[kb][2], qfrag[kb][3],
                         b0, b1, qs[kb], ks[kb]);
            }
            s_frag[nt][0] = d0 * softmax_scale;
            s_frag[nt][1] = d1 * softmax_scale;
            s_frag[nt][2] = d2 * softmax_scale;
            s_frag[nt][3] = d3 * softmax_scale;

            if (kv_tile == n_kv_tiles_total - 1) {
                const int kc0 = kv_tile * kBc + nt * 8 + tid_g * 2 + 0;
                const int kc1 = kv_tile * kBc + nt * 8 + tid_g * 2 + 1;
                if (kc0 >= seq_len) { s_frag[nt][0] = -INFINITY; s_frag[nt][2] = -INFINITY; }
                if (kc1 >= seq_len) { s_frag[nt][1] = -INFINITY; s_frag[nt][3] = -INFINITY; }
            }
        }

        // ---------- Online softmax (warp-scoped shfl reduction) ----------
        float lmax_lo = -INFINITY, lmax_hi = -INFINITY;
        #pragma unroll
        for (int nt = 0; nt < n_qk_n_tiles; ++nt) {
            lmax_lo = fmaxf(lmax_lo, fmaxf(s_frag[nt][0], s_frag[nt][1]));
            lmax_hi = fmaxf(lmax_hi, fmaxf(s_frag[nt][2], s_frag[nt][3]));
        }
        #pragma unroll
        for (int o = 2; o > 0; o >>= 1) {
            lmax_lo = fmaxf(lmax_lo, __shfl_xor_sync(0xffffffffu, lmax_lo, o));
            lmax_hi = fmaxf(lmax_hi, __shfl_xor_sync(0xffffffffu, lmax_hi, o));
        }
        const float new_max_lo = fmaxf(m_lo, lmax_lo);
        const float new_max_hi = fmaxf(m_hi, lmax_hi);
        const float alpha_lo = (m_lo == -INFINITY) ? 1.0f : exp2f(m_lo - new_max_lo);
        const float alpha_hi = (m_hi == -INFINITY) ? 1.0f : exp2f(m_hi - new_max_hi);

        float lsum_lo = 0.f, lsum_hi = 0.f;
        #pragma unroll
        for (int nt = 0; nt < n_qk_n_tiles; ++nt) {
            s_frag[nt][0] = exp2f(s_frag[nt][0] - new_max_lo); lsum_lo += s_frag[nt][0];
            s_frag[nt][1] = exp2f(s_frag[nt][1] - new_max_lo); lsum_lo += s_frag[nt][1];
            s_frag[nt][2] = exp2f(s_frag[nt][2] - new_max_hi); lsum_hi += s_frag[nt][2];
            s_frag[nt][3] = exp2f(s_frag[nt][3] - new_max_hi); lsum_hi += s_frag[nt][3];
        }
        #pragma unroll
        for (int o = 2; o > 0; o >>= 1) {
            lsum_lo += __shfl_xor_sync(0xffffffffu, lsum_lo, o);
            lsum_hi += __shfl_xor_sync(0xffffffffu, lsum_hi, o);
        }

        l_lo = l_lo * alpha_lo + lsum_lo;
        l_hi = l_hi * alpha_hi + lsum_hi;
        m_lo = new_max_lo;
        m_hi = new_max_hi;

        #pragma unroll
        for (int nt = 0; nt < n_pv_n_tiles; ++nt) {
            pv_acc[nt][0] *= alpha_lo;
            pv_acc[nt][1] *= alpha_lo;
            pv_acc[nt][2] *= alpha_hi;
            pv_acc[nt][3] *= alpha_hi;
        }

        // ---------- P quant (v19: pack via cvt.e4m3x2 + STS.U16) ---------
        // P_quant = P_real * 256. 1/256 inverse folded into the epilogue
        // FFMA with per-channel V scale; PV mma is plain f32.e4m3.e4m3.
        constexpr float kPFp8Scale = 256.0f;
        const int row_lo = row + 0;
        const int row_hi = row + 8;
        #pragma unroll
        for (int nt = 0; nt < n_qk_n_tiles; ++nt) {
            const int n_base = nt * 8 + tid_g * 2;
            const int o_lo = layoutP(row_lo, n_base + 0);
            const int o_hi = layoutP(row_hi, n_base + 0);
            uint16_t pack_lo = cvt_e4m3x2_fp32(
                s_frag[nt][0] * kPFp8Scale, s_frag[nt][1] * kPFp8Scale);
            uint16_t pack_hi = cvt_e4m3x2_fp32(
                s_frag[nt][2] * kPFp8Scale, s_frag[nt][3] * kPFp8Scale);
            *reinterpret_cast<uint16_t*>(smem_P_ptr + o_lo) = pack_lo;
            *reinterpret_cast<uint16_t*>(smem_P_ptr + o_hi) = pack_hi;
        }
        __syncwarp();

        cute::copy(s2r_copy_P, tXsP, tXrP);
        cute::copy(s2r_copy_V, tXsV(_, _, _, curr), tXrV);
        auto tCrP_u32 = recast<uint32_t>(tCrP);
        auto tCrV_u32 = recast<uint32_t>(tCrV);

        #pragma unroll
        for (int kb = 0; kb < n_v_kblk_per_tile; ++kb) {
            uint32_t a0 = tCrP_u32(0, 0, kb);
            uint32_t a1 = tCrP_u32(1, 0, kb);
            uint32_t a2 = tCrP_u32(2, 0, kb);
            uint32_t a3 = tCrP_u32(3, 0, kb);
            #pragma unroll
            for (int nt = 0; nt < n_pv_n_tiles; ++nt) {
                uint32_t b0 = tCrV_u32(0, nt, kb);
                uint32_t b1 = tCrV_u32(1, nt, kb);
                mma_fp8_plain(pv_acc[nt][0], pv_acc[nt][1], pv_acc[nt][2], pv_acc[nt][3],
                              a0, a1, a2, a3, b0, b1);
            }
        }

        if constexpr (kStages == 1) {
            // K and V are both consumed above — reload stage 0 for the next
            // iter; wait_group<0> at the top of the next iter blocks on it.
            if (iter + 1 < n_kv_tiles) {
                load_page(0, kv_tile + 1);
                cp_async_commit_group();
            }
        } else {
            const int tmp = curr; curr = next; next = tmp;
        }
        __syncwarp();
    }

    // ---------- Epilogue --------------------------------------------------
    //
    // Fused-reduce path uses ALL 32 lanes of the warp for the D-stride
    // reduce loop — early-returning invalid M-row lanes would leave most
    // D-cols unwritten. Gate ONLY the partial-write on lo/hi_valid.
    const int h_q_lo = h_kv * gqa_group + m_row_lo;
    const int h_q_hi = h_kv * gqa_group + m_row_hi;

    if (num_splits == 1) {
        if (!lo_valid && !hi_valid) return;
        const float inv_l_lo = (l_lo > 0.f) ? (1.f / l_lo) : 0.f;
        const float inv_l_hi = (l_hi > 0.f) ? (1.f / l_hi) : 0.f;
        __nv_bfloat16* O_lo = O + ((int64_t)b * num_q_heads + h_q_lo) * kD;
        __nv_bfloat16* O_hi = O + ((int64_t)b * num_q_heads + h_q_hi) * kD;
        // bf16x2 pack: adjacent (n_base+0, n_base+1) outputs collapse into
        // one STG.B32 + one cvt.rn.bf16x2.f32 (halves both store and convert
        // counts).
        #pragma unroll
        for (int nt = 0; nt < n_pv_n_tiles; ++nt) {
            const int n_base = nt * 8 + tid_g * 2;
            // vsc[nt][*] already folds in (1/256) for the P quant.
            const float lo0 = inv_l_lo * vsc[nt][0];
            const float lo1 = inv_l_lo * vsc[nt][1];
            const float hi0 = inv_l_hi * vsc[nt][0];
            const float hi1 = inv_l_hi * vsc[nt][1];
            if (lo_valid) {
                uint32_t pack_lo = cvt_bf16x2_fp32(
                    pv_acc[nt][0] * lo0, pv_acc[nt][1] * lo1);
                *reinterpret_cast<uint32_t*>(O_lo + n_base) = pack_lo;
            }
            if (hi_valid) {
                uint32_t pack_hi = cvt_bf16x2_fp32(
                    pv_acc[nt][2] * hi0, pv_acc[nt][3] * hi1);
                *reinterpret_cast<uint32_t*>(O_hi + n_base) = pack_hi;
            }
        }
    } else {
        const int64_t off_lo = (((int64_t)b * num_q_heads) + h_q_lo) * num_splits + split_id;
        const int64_t off_hi = (((int64_t)b * num_q_heads) + h_q_hi) * num_splits + split_id;
        if (lo_valid && tid_g == 0) {
            M_partial[off_lo] = m_lo;
            L_partial[off_lo] = l_lo;
        }
        if (hi_valid && tid_g == 0) {
            M_partial[off_hi] = m_hi;
            L_partial[off_hi] = l_hi;
        }
        float* O_part_lo = O_partial + off_lo * kD;
        float* O_part_hi = O_partial + off_hi * kD;
        #pragma unroll
        for (int nt = 0; nt < n_pv_n_tiles; ++nt) {
            const int n_base = nt * 8 + tid_g * 2;
            if (lo_valid) {
                O_part_lo[n_base + 0] = pv_acc[nt][0];
                O_part_lo[n_base + 1] = pv_acc[nt][1];
            }
            if (hi_valid) {
                O_part_hi[n_base + 0] = pv_acc[nt][2];
                O_part_hi[n_base + 1] = pv_acc[nt][3];
            }
        }

        // ---------- Fused reduce per-warp (last CTA writes final O) -----
        //
        // In v3 each warp serves its own (b, h_kv) work unit so the
        // sync + reduce stays warp-scoped: no __syncthreads needed. The
        // warp atomicAdds its gqa_group counters and, on the triggering
        // counter, the warp itself does the reduce for that h_q.
        if (sync_counter != nullptr) {
            __syncwarp();              // ensure all this warp's partial writes are visible
            __threadfence();           // release: other CTAs/warps see our partials

            #pragma unroll
            for (int g = 0; g < 16; ++g) {
                if (g >= gqa_group) break;
                const int h_q = h_kv * gqa_group + g;
                // Lane 0 atomic-adds; broadcast result to whole warp.
                int prev;
                if (lane == 0) {
                    prev = atomicAdd(
                        &sync_counter[(int64_t)b * num_q_heads + h_q], 1);
                } else {
                    prev = 0;
                }
                prev = __shfl_sync(0xffffffffu, prev, 0);
                if (prev + 1 != target_counter) continue;

                __threadfence();         // acquire: see all writers' partials
                const int64_t off_bhq = ((int64_t)b * num_q_heads + h_q) * num_splits;
                float m_final = -INFINITY;
                for (int k = 0; k < num_splits; ++k) {
                    float mk = M_partial[off_bhq + k];
                    if (mk > m_final) m_final = mk;
                }
                constexpr int kPerLane = (kD + 31) / 32;
                float o_acc[kPerLane];
                #pragma unroll
                for (int i = 0; i < kPerLane; ++i) o_acc[i] = 0.f;
                float l_final = 0.f;
                for (int k = 0; k < num_splits; ++k) {
                    float mk = M_partial[off_bhq + k];
                    float lk = L_partial[off_bhq + k];
                    if (lk == 0.f) continue;
                    float alpha = exp2f(mk - m_final);
                    l_final += alpha * lk;
                    const float* O_k = O_partial + (off_bhq + k) * kD;
                    #pragma unroll
                    for (int i = 0; i < kPerLane; ++i) {
                        int d = lane + i * 32;
                        if (d < kD) o_acc[i] += alpha * O_k[d];
                    }
                }
                const float inv_l = (l_final > 0.f) ? (1.f / l_final) : 0.f;
                const float* Vsc_h = V_chan_scale + (int64_t)h_kv * kD;
                __nv_bfloat16* O_row = O + ((int64_t)b * num_q_heads + h_q) * kD;
                #pragma unroll
                for (int i = 0; i < kPerLane; ++i) {
                    int d = lane + i * 32;
                    if (d < kD) {
                        const float factor = inv_l * Vsc_h[d] * kInvP;
                        O_row[d] = __float2bfloat16(o_acc[i] * factor);
                    }
                }
            }
        }
    }
#else
    // sm < 1200 stub.
    (void)Q; (void)Qs; (void)K_pool; (void)K_chan_scale; (void)V_pool; (void)V_chan_scale;
    (void)block_table; (void)seq_lens; (void)O;
    (void)M_partial; (void)L_partial; (void)O_partial; (void)sync_counter;
    (void)batch; (void)num_q_heads; (void)num_kv_heads;
    (void)max_blocks; (void)num_splits; (void)target_counter;
    (void)page_size;
    (void)softmax_scale;
#endif
}

// ---------- Launch helper --------------------------------------------------

template <int kHeadDim, int kBc, int kStages, int kCtasPerSm>
cudaError_t launch_decode_paged_impl(
    const void* Q, const void* Qs,
    const void* K_pool, const void* K_chan_scale,
    const void* V_pool, const void* V_chan_scale,
    const void* block_table, const void* seq_lens,
    void* O,
    void* M_partial, void* L_partial, void* O_partial,
    void* sync_counter,
    int batch, int num_q_heads, int num_kv_heads,
    int max_blocks, int num_splits, int target_counter,
    int page_size,
    float softmax_scale,
    cudaStream_t stream)
{
    if (num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0) return cudaErrorInvalidValue;
    if (num_splits <= 0) return cudaErrorInvalidValue;
    if (num_splits > 1 && (M_partial == nullptr || L_partial == nullptr || O_partial == nullptr))
        return cudaErrorInvalidValue;
    // page_size must be ≥ kBc=32 and a multiple of kBc. The slow per-byte
    // path was removed (see git history); use page_size ∈ {32, 64, ...}.
    if (page_size <= 0 || page_size % kBc != 0)
        return cudaErrorInvalidValue;
    if (K_chan_scale == nullptr || V_chan_scale == nullptr)
        return cudaErrorInvalidValue;
    const int gqa_group = num_q_heads / num_kv_heads;
    if (gqa_group > 16) return cudaErrorNotSupported;        // v3 packs gqa into 1 warp
    const int total_wu = batch * num_kv_heads;
    const int packed   = (total_wu + 3) / 4;

    constexpr int kThreads = 4 * 32;
    dim3 grid(packed, num_splits, 1);
    dim3 block(kThreads);

    using Layouts = SmemLayouts<kBc, kHeadDim, kStages>;
    // K and V scales are both gmem tensors loaded into registers; no
    // per-tile scale smem slab.
    const int per_warp_bytes =
        sizeof(ElementA) * (int)cosize(typename Layouts::K{})
      + sizeof(ElementA) * (int)cosize(typename Layouts::V{})
      + sizeof(ElementA) * (int)cosize(typename Layouts::P{});
    const int smem_bytes = 4 * per_warp_bytes;

    auto* kfn = decode_paged_kernel<kHeadDim, kBc, kStages, kCtasPerSm>;
    cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    kfn<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(Q),
        reinterpret_cast<const uint8_t*>(Qs),
        reinterpret_cast<const __nv_fp8_e4m3*>(K_pool),
        reinterpret_cast<const uint8_t*>(K_chan_scale),
        reinterpret_cast<const __nv_fp8_e4m3*>(V_pool),
        reinterpret_cast<const float*>(V_chan_scale),
        reinterpret_cast<const int32_t*>(block_table),
        reinterpret_cast<const int32_t*>(seq_lens),
        reinterpret_cast<__nv_bfloat16*>(O),
        reinterpret_cast<float*>(M_partial),
        reinterpret_cast<float*>(L_partial),
        reinterpret_cast<float*>(O_partial),
        reinterpret_cast<int32_t*>(sync_counter),
        batch, num_q_heads, num_kv_heads,
        max_blocks, num_splits, target_counter,
        page_size,
        softmax_scale * 1.4426950408889634f);    // × log2(e)
    return cudaGetLastError();
}

}  // namespace flash_attn_sm120_decode_paged
