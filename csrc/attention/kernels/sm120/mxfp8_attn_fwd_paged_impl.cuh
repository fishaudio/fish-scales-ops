// MXFP8 paged-KV prefill (extend mode) implementation header.
//
// Variant of mxfp8_attn_fwd_impl.cuh that consumes a paged KV cache
// (block_table-driven) and ragged Q via qo_indptr — the same surface
// sglang's `FlashInferAttnBackend` consumes from flashinfer's
// `BatchPrefillWithPagedKVCache`.
//
// Key differences vs the contiguous prefill:
//   - K / V live in a global page pool keyed by paged_kv_indices.
//   - Q is ragged: q_seq_len[b] = qo_indptr[b+1] - qo_indptr[b].
//   - Causal mask compares the q-token's *absolute* KV position
//     kv_offset_history[b] + q_row to the kv column.
//   - Each CTA serves one (b, h_q, q_tile_in_batch) work unit; the
//     host packs work_units[] so we avoid an in-kernel batch search.
//   - kBc is fixed at 32 (= MXFP8 sf_vec_size) so every kv-tile lives
//     inside exactly one page, regardless of page_size ∈ {32,64,128,256}.
//
// Per-D tile config (initial cut; performance tuning is a separate pass):
//   D = 32  : Br=64, Bc=32, kStages=2, kCtasPerSm=2  (smem ~14 KB / CTA)
//   D = 64  : Br=64, Bc=32, kStages=2, kCtasPerSm=2  (smem ~22 KB / CTA)
//   D = 128 : Br=64, Bc=32, kStages=2, kCtasPerSm=1  (smem ~38 KB / CTA)
//
// This file is included by the per-D TUs (mxfp8_attn_fwd_paged_d{32,64,128}.cu)
// so each D specialisation gets its own ptxas pass — mirroring the
// non-paged prefill's structural codegen isolation.
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

namespace flash_attn_sm120_paged_prefill {

using namespace cute;

using ElementA   = float_e4m3_t;
using ElementB   = float_e4m3_t;
using ElementSF  = float_ue8m0_t;
using ElementAcc = float;
using ElementOut = bfloat16_t;

constexpr int kBlock3 = 32;            // K-block size for UE8M0 scales (sf_vec_size)

using MmaAtom = MMA_Atom<SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    ElementA, ElementB, ElementAcc, ElementSF, /*sf_vec_size=*/32>>;

// kNumWarps × atom_M(16) = Br. kNumWarps = Br / 16. Br=64 → 4 warps.
template <int kNumWarps>
using TiledMmaT = TiledMMA<MmaAtom, Layout<Shape<Int<kNumWarps>, _1, _1>>>;

template <int K> struct SwizzledKAtom;
template <> struct SwizzledKAtom< 32> { using type = GMMA::Layout_K_SW32_Atom<ElementA>; };
template <> struct SwizzledKAtom< 64> { using type = GMMA::Layout_K_SW64_Atom<ElementA>; };
template <> struct SwizzledKAtom<128> { using type = GMMA::Layout_K_SW128_Atom<ElementA>; };
template <> struct SwizzledKAtom<256> { using type = GMMA::Layout_K_SW128_Atom<ElementA>; };

template <int kBr, int kBc, int kD, int kStages>
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
        make_shape(Int<kBr>{}, Int<kBc>{}),
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

// v19 micro-opt (shared with v15a impl): pack 2 fp32 → 2 fp8 (e4m3) in one
// PTX. d[15:8] = e4m3(b), d[7:0] = e4m3(a). Paired with STS.U16 in the
// P-quant inner loop to halve conversion + store counts.
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

// Pack 2 fp32 → 2 bf16 in one PTX instruction. d[31:16]=bf16(b), d[15:0]=bf16(a).
// Used in the epilogue, paired with STG.B32 to halve global stores.
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

// Regular FP8 mma (no block_scale) for PV. See decode kernel header.
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

// ---------- Kernel template ------------------------------------------------
//
// Specialised on:
//   kHeadDim ∈ {32, 64, 128}
//   kBr      = 64                    (M-tile, 4 warps × atom_M=16)
//   kBc      = 32                    (= MXFP8 sf_vec_size; one page lookup per kv-tile)
//   kStages  = 2                     (cp.async pipeline depth)
//   kCtasPerSm                       (per-D: 2 at D≤64, 1 at D=128)
//   kIsCausal                        (compile-time mask)
//
// Each CTA processes one work unit = (b, h_q, q_tile_in_batch). Host packs
// work_units[N] so the kernel never binary-searches qo_indptr.
//
// page_size is a runtime kernel arg; the only constraint is page_size %
// kBc == 0 (i.e. page_size ∈ {32, 64, 128, 256}). page_size=32 is the
// recommended layout for sglang integration.

template <int kHeadDim, int kBr, int kBc, int kStages, int kCtasPerSm, bool kIsCausal>
__global__
__launch_bounds__((kBr / 16) * 32, kCtasPerSm)
void paged_prefill_kernel(
    const __nv_fp8_e4m3* __restrict__ Q,                 // [total_q, H_q, D]
    const uint8_t*       __restrict__ Qs,                // [total_q/16, H_q, D/32]
    const __nv_fp8_e4m3* __restrict__ K_pool,            // [num_pages, page_size, H_kv, D]
    const uint8_t*       __restrict__ K_chan_scale,      // [H_kv, D/32] UE8M0 — channel-K scale
    const __nv_fp8_e4m3* __restrict__ V_pool,            // [num_pages, D, H_kv, page_size]
    const float*         __restrict__ V_chan_scale,      // [H_kv, D] fp32 — channel-V scale
    const int32_t*       __restrict__ qo_indptr,         // [B+1]
    const int32_t*       __restrict__ paged_kv_indices,  // [total_pages_used]
    const int32_t*       __restrict__ paged_kv_indptr,   // [B+1]
    const int32_t*       __restrict__ paged_kv_last_page_len, // [B]
    const int32_t*       __restrict__ work_units,        // [total_work*3] = (b, h_q, q_tile_in_b) tuples
    __nv_bfloat16*       __restrict__ O,                 // [total_q, H_q, D]
    int total_work, int num_q_heads, int num_kv_heads,
    int page_size,
    float softmax_scale)
{
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 1200)
    using Layouts = SmemLayouts<kBr, kBc, kHeadDim, kStages>;
    using SmemLayoutK = typename Layouts::K;
    using SmemLayoutV = typename Layouts::V;
    using SmemLayoutP = typename Layouts::P;

    constexpr int kD             = kHeadDim;
    constexpr int kNumWarps      = kBr / 16;
    constexpr int kThreads       = kNumWarps * 32;
    using TiledMma = TiledMmaT<kNumWarps>;
    constexpr int n_kblk         = kD / kBlock3;
    constexpr int n_v_kblk_per_tile = kBc / kBlock3;
    constexpr int n_qk_n_tiles   = kBc / 8;
    constexpr int n_pv_n_tiles   = kD / 8;

    const int tid    = threadIdx.x;
    const int warp   = tid / 32;
    const int lane   = tid % 32;
    const int row    = lane / 4;
    const int tid_g  = lane % 4;

    const int gqa_group = num_q_heads / num_kv_heads;

    // ---------- Smem slabs -------------------------------------------------
    // K and V scales are both channel tensors loaded into registers (ks[kb],
    // vsc[nt][i]) below; no per-tile scale smem slab.
    extern __shared__ unsigned char smem_raw[];
    auto* smem_K_ptr  = reinterpret_cast<ElementA*>(smem_raw);
    auto* smem_V_ptr  = smem_K_ptr + cosize(SmemLayoutK{});
    auto* smem_P_ptr  = smem_V_ptr + cosize(SmemLayoutV{});

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
    auto thr_mma = mma.get_thread_slice(tid);

    auto s2r_copy_K = make_tiled_copy_B(SmemCopyAtomB{}, mma);
    auto thr_s2r_K  = s2r_copy_K.get_thread_slice(tid);
    auto tCrK = thr_mma.partition_fragment_B(sK_pi(_, _, Int<0>{}));
    auto tXsK = thr_s2r_K.partition_S(sK_pi);
    auto tXrK = thr_s2r_K.retile_D(tCrK);

    auto s2r_copy_V = make_tiled_copy_B(SmemCopyAtomB{}, mma);
    auto thr_s2r_V  = s2r_copy_V.get_thread_slice(tid);
    auto tCrV = thr_mma.partition_fragment_B(sV_pi(_, _, Int<0>{}));
    auto tXsV = thr_s2r_V.partition_S(sV_pi);
    auto tXrV = thr_s2r_V.retile_D(tCrV);

    auto s2r_copy_P = make_tiled_copy_A(SmemCopyAtomA{}, mma);
    auto thr_s2r_P  = s2r_copy_P.get_thread_slice(tid);
    auto tCrP = thr_mma.partition_fragment_A(sP_pi);
    auto tXsP = thr_s2r_P.partition_S(sP_pi);
    auto tXrP = thr_s2r_P.retile_D(tCrP);

    // ---------- Persistent CTA grid-stride loop over work units ------------
    for (int wu = blockIdx.x; wu < total_work; wu += gridDim.x) {
        // Work unit unpacking: (b, h_q, q_tile_in_b).
        const int b        = work_units[wu * 3 + 0];
        const int h        = work_units[wu * 3 + 1];
        const int q_tile_b = work_units[wu * 3 + 2];
        const int h_kv     = h / gqa_group;

        // Per-batch ranges.
        const int q_lo      = qo_indptr[b];
        const int q_hi      = qo_indptr[b + 1];
        const int q_seq_len = q_hi - q_lo;
        const int kv_page_lo = paged_kv_indptr[b];
        const int kv_page_hi = paged_kv_indptr[b + 1];
        const int kv_n_pages = kv_page_hi - kv_page_lo;
        const int kv_last_len = paged_kv_last_page_len[b];
        const int kv_seq_len  = (kv_n_pages > 0)
            ? ((kv_n_pages - 1) * page_size + kv_last_len)
            : 0;
        // History prefix length = KV length already in the cache before
        // this extend slot's S_q new tokens.
        const int kv_offset_history = kv_seq_len - q_seq_len;

        // q_row_base_in_b is the first M-row this warp owns.
        // Each lane's M-rows in the 16-row warp tile are:
        //   lo half: row + 0  (where row = lane / 4 ∈ [0, 8))
        //   hi half: row + 8
        // So q_row_lo_b / q_row_hi_b are per-lane absolute Q positions; both
        // the causal mask and the OOB validity check must include `row`.
        const int q_row_base_in_b = q_tile_b * kBr + warp * 16;
        const int q_row_lo_b      = q_row_base_in_b + row + 0;
        const int q_row_hi_b      = q_row_base_in_b + row + 8;
        const bool lo_valid = (q_row_lo_b < q_seq_len);
        const bool hi_valid = (q_row_hi_b < q_seq_len);

        const int q_row_stride  = num_q_heads  * kD;
        const int kv_row_stride = num_kv_heads * kD;   // within a page, K is row-major (page_size, H_kv, D)

        // Absolute base addresses for Q / O for this work unit's warp.
        const auto* Q_base  = Q  + ((int64_t)(q_lo + q_row_base_in_b)) * q_row_stride + h * kD;
        auto*       O_base  = O  + ((int64_t)(q_lo + q_row_base_in_b)) * q_row_stride + h * kD;
        // Qs base: one byte per (16-Q-row tile, H_q, D-block). Each warp
        // owns 16 M-rows, so Qs_base picks the Qs-row matching this warp's
        // 16-row block. The Qs tile index within the batch is `q_lo/16 +
        // q_tile_b * kNumWarps + warp` (note: q_lo must be a multiple of
        // 16, enforced caller-side).
        const auto* Qs_base = Qs + ((int64_t)(q_lo / 16) + q_tile_b * kNumWarps + warp) * num_q_heads * n_kblk
                                 + h * n_kblk;

        // ---------- Q frag + Q scales: register-resident -------------------
        // Out-of-bound Q rows (q_row_*_b >= q_seq_len) load zeros — masked
        // against -INFINITY in the softmax via lo_valid / hi_valid below.
        uint32_t qfrag[n_kblk][4];
        #pragma unroll
        for (int kb = 0; kb < n_kblk; ++kb) {
            const int k0 = kb * kBlock3;
            const int a_col0 = tid_g * 4 + k0;
            const int a_col1 = a_col0 + 16;
            qfrag[kb][0] = lo_valid
                ? *reinterpret_cast<const uint32_t*>(Q_base + (row + 0) * q_row_stride + a_col0) : 0u;
            qfrag[kb][1] = hi_valid
                ? *reinterpret_cast<const uint32_t*>(Q_base + (row + 8) * q_row_stride + a_col0) : 0u;
            qfrag[kb][2] = lo_valid
                ? *reinterpret_cast<const uint32_t*>(Q_base + (row + 0) * q_row_stride + a_col1) : 0u;
            qfrag[kb][3] = hi_valid
                ? *reinterpret_cast<const uint32_t*>(Q_base + (row + 8) * q_row_stride + a_col1) : 0u;
        }
        uint8_t qs[n_kblk];
        #pragma unroll
        for (int kb = 0; kb < n_kblk; ++kb) qs[kb] = Qs_base[kb];

        // ---------- K channel-scale preload (UE8M0 byte per D-block) --------
        uint8_t ks[n_kblk];
        {
            const uint8_t* Ksc_h = K_chan_scale + (int64_t)h_kv * n_kblk;
            #pragma unroll
            for (int kb = 0; kb < n_kblk; ++kb) ks[kb] = Ksc_h[kb];
        }

        // ---------- V channel-scale preload --------------------------------
        // 2*n_pv_n_tiles fp32 scales per lane, already folded with 1/256
        // for the P quant inverse.
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

        // ---------- Per-row softmax / PV state -----------------------------
        float m_lo = -INFINITY, m_hi = -INFINITY;
        float l_lo = 0.f,        l_hi = 0.f;
        float pv_acc[n_pv_n_tiles][4];
        #pragma unroll
        for (int nt = 0; nt < n_pv_n_tiles; ++nt) {
            pv_acc[nt][0] = pv_acc[nt][1] = pv_acc[nt][2] = pv_acc[nt][3] = 0.f;
        }

        // ---------- KV tile bound (causal short-circuit) -------------------
        int n_kv_tiles = (kv_seq_len + kBc - 1) / kBc;
        if constexpr (kIsCausal) {
            // The last Q-row in this CTA's tile (q_row_base + kBr-1) attends
            // to KV positions [0, kv_offset_history + q_row_base + kBr - 1].
            // Tiles strictly above the diagonal are skippable.
            const int q_row_max_global = kv_offset_history + q_tile_b * kBr + kBr - 1;
            const int n_eff = (q_row_max_global / kBc) + 1;
            if (n_eff < n_kv_tiles) n_kv_tiles = n_eff;
        }

        // ---------- Paged K/V load lambdas — page_size ≥ kBc only --------
        // Caller-side page_size must be a multiple of kBc=32. One page
        // covers ≥1 kv-tile; per-tile page lookup hoists out of the inner
        // loop. The page_size < 32 slow-gather path was removed for code
        // clarity (see git history for the variant).
        auto issue_cp_async_K = [&](int stage, int kv_tile) {
            const int kv_token_start = kv_tile * kBc;
            const int page_outer = kv_token_start / page_size;
            const int page_inner = kv_token_start - page_outer * page_size;
            const int32_t page_idx = paged_kv_indices[kv_page_lo + page_outer];
            const auto* K_page = K_pool
                + (int64_t)page_idx * page_size * num_kv_heads * kD
                + (int64_t)page_inner * num_kv_heads * kD
                + (int64_t)h_kv * kD;
            const int kv_stride = num_kv_heads * kD;
            constexpr int kOps = (kBc * kD) / 16;
            constexpr int kChunksPerRow = kD / 16;
            #pragma unroll
            for (int i = tid; i < kOps; i += kThreads) {
                const int s_idx  = i / kChunksPerRow;
                const int d_offs = (i % kChunksPerRow) * 16;
                const auto* src = K_page + (int64_t)s_idx * kv_stride + d_offs;
                int off = layoutK(s_idx, d_offs, stage);
                uint32_t dst = smem_to_uint(smem_K_ptr + off);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                              :: "r"(dst), "l"(src));
            }
        };
        auto issue_cp_async_V = [&](int stage, int kv_tile) {
            const int kv_token_start = kv_tile * kBc;
            const int page_outer = kv_token_start / page_size;
            const int page_inner = kv_token_start - page_outer * page_size;
            const int32_t page_idx = paged_kv_indices[kv_page_lo + page_outer];
            const auto* V_page = V_pool
                + (int64_t)page_idx * kD * num_kv_heads * page_size
                + (int64_t)h_kv * page_size
                + (int64_t)page_inner;
            const int v_d_stride = num_kv_heads * page_size;
            constexpr int kOps = (kD * kBc) / 16;
            constexpr int kChunksPerRow = kBc / 16;
            #pragma unroll
            for (int i = tid; i < kOps; i += kThreads) {
                const int d_idx    = i / kChunksPerRow;
                const int s_offset = (i % kChunksPerRow) * 16;
                const auto* src = V_page + (int64_t)d_idx * v_d_stride + s_offset;
                int off = layoutV(d_idx, s_offset, stage);
                uint32_t dst = smem_to_uint(smem_V_ptr + off);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                              :: "r"(dst), "l"(src));
            }
        };
        // K and V scales are channel tensors in registers; no per-tile load.
        auto load_kv_tile = [&](int stage, int kv_tile) {
            issue_cp_async_K(stage, kv_tile);
            issue_cp_async_V(stage, kv_tile);
        };

        // ---------- Prologue: prefetch first stage -------------------------
        int curr = 0, next = 1;
        if (n_kv_tiles > 0) {
            load_kv_tile(0, 0);
            cp_async_commit_group();
        }

        // ---------- KV tile loop -------------------------------------------
        for (int kv_tile = 0; kv_tile < n_kv_tiles; ++kv_tile) {
            if (kv_tile + 1 < n_kv_tiles) {
                load_kv_tile(next, kv_tile + 1);
                cp_async_commit_group();
                cp_async_wait_group<1>();
            } else {
                cp_async_wait_group<0>();
            }
            __syncthreads();

            cute::copy(s2r_copy_K, tXsK(_, _, _, curr), tXrK);
            auto tCrK_u32 = recast<uint32_t>(tCrK);

            // ---------- QK -----------------------------------------------
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

                // ---------- Masking ---------------------------------------
                // KV tail: kv col >= kv_seq_len → -INFINITY.
                // Causal: kv col > absolute_q_pos → -INFINITY.
                // Q-row OOB: lo_valid / hi_valid is gating Q-fragment to 0;
                //            also gate the row's masked elements to -INF
                //            so softmax produces 0 (== identity in epilogue).
                {
                    const int q_abs_lo = kv_offset_history + q_row_lo_b;
                    const int q_abs_hi = kv_offset_history + q_row_hi_b;
                    const int kc0 = kv_tile * kBc + nt * 8 + tid_g * 2 + 0;
                    const int kc1 = kv_tile * kBc + nt * 8 + tid_g * 2 + 1;

                    bool mask00 = (kc0 >= kv_seq_len) || !lo_valid;
                    bool mask01 = (kc1 >= kv_seq_len) || !lo_valid;
                    bool mask10 = (kc0 >= kv_seq_len) || !hi_valid;
                    bool mask11 = (kc1 >= kv_seq_len) || !hi_valid;
                    if constexpr (kIsCausal) {
                        mask00 = mask00 || (kc0 > q_abs_lo);
                        mask01 = mask01 || (kc1 > q_abs_lo);
                        mask10 = mask10 || (kc0 > q_abs_hi);
                        mask11 = mask11 || (kc1 > q_abs_hi);
                    }
                    if (mask00) s_frag[nt][0] = -INFINITY;
                    if (mask01) s_frag[nt][1] = -INFINITY;
                    if (mask10) s_frag[nt][2] = -INFINITY;
                    if (mask11) s_frag[nt][3] = -INFINITY;
                }
            }

            // ---------- Online softmax (Bc -> n_qk_n_tiles N-tiles) -------
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

            // ---------- P quant -> smem_P slab (v19: pack via STS.U16) ---
            // 1/256 is folded into the epilogue FFMA alongside the per-channel
            // V scale; PV mma is plain f32.e4m3.e4m3 (no block_scale operand).
            // Pack 2 fp32 → 2 fp8 in one cvt.e4m3x2.f32 and write 2 bytes via
            // STS.U16. Halves both conversions and stores vs the byte-by-byte
            // path. Adjacent cols at o_lo+0/+1 are contiguous bytes in the
            // SW64-swizzled P layout (Swizzle<*,4,3> doesn't touch bit 0).
            constexpr float kPFp8Scale = 256.0f;
            const int row_lo = warp * 16 + (row + 0);
            const int row_hi = warp * 16 + (row + 8);
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

            const int tmp = curr; curr = next; next = tmp;
            __syncthreads();
        }

        // ---------- Epilogue (bf16x2 pack: cvt.rn.bf16x2.f32 + STG.B32) -------
        // OOB rows skip the O-write (their pv_acc is garbage from masked
        // softmax, but we never write it). Valid rows divide by l and write.
        // vsc[nt][*] = scale_V[h_kv, n_base + *] * (1/256), preloaded above.
        // Pack adjacent (n_base+0, n_base+1) bf16 outputs into one uint32
        // store to halve epilogue store count.
        const float inv_l_lo = (l_lo > 0.f) ? (1.f / l_lo) : 0.f;
        const float inv_l_hi = (l_hi > 0.f) ? (1.f / l_hi) : 0.f;
        #pragma unroll
        for (int nt = 0; nt < n_pv_n_tiles; ++nt) {
            const int n_base = nt * 8 + tid_g * 2;
            const float lo0 = inv_l_lo * vsc[nt][0];
            const float lo1 = inv_l_lo * vsc[nt][1];
            const float hi0 = inv_l_hi * vsc[nt][0];
            const float hi1 = inv_l_hi * vsc[nt][1];
            if (lo_valid) {
                uint32_t pack_lo = cvt_bf16x2_fp32(
                    pv_acc[nt][0] * lo0, pv_acc[nt][1] * lo1);
                *reinterpret_cast<uint32_t*>(O_base + (row + 0) * q_row_stride + n_base) = pack_lo;
            }
            if (hi_valid) {
                uint32_t pack_hi = cvt_bf16x2_fp32(
                    pv_acc[nt][2] * hi0, pv_acc[nt][3] * hi1);
                *reinterpret_cast<uint32_t*>(O_base + (row + 8) * q_row_stride + n_base) = pack_hi;
            }
        }
        __syncthreads();
    }   // for wu
#else
    // sm < 1200 stub.
    (void)Q; (void)Qs; (void)K_pool; (void)K_chan_scale; (void)V_pool; (void)V_chan_scale;
    (void)qo_indptr; (void)paged_kv_indices; (void)paged_kv_indptr;
    (void)paged_kv_last_page_len; (void)work_units; (void)O;
    (void)total_work; (void)num_q_heads; (void)num_kv_heads;
    (void)page_size; (void)softmax_scale;
#endif
}

// ---------- Per-(D, causal) launch helper + dispatch -----------------------

template <int kHeadDim, int kBr, int kBc, int kStages, int kCtasPerSm, bool kIsCausal>
cudaError_t launch_paged_prefill_impl(
    const void* Q, const void* Qs,
    const void* K_pool, const void* K_chan_scale,
    const void* V_pool, const void* V_chan_scale,
    const void* qo_indptr,
    const void* paged_kv_indices,
    const void* paged_kv_indptr,
    const void* paged_kv_last_page_len,
    const void* work_units,
    void* O,
    int total_work, int num_q_heads, int num_kv_heads,
    int page_size,
    float softmax_scale, cudaStream_t stream)
{
    if (num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0)
        return cudaErrorInvalidValue;
    // page_size must be ≥ kBc=32 and a multiple of kBc. The slow gather
    // path was removed (see git history); use page_size ∈ {32, 64, 128, ...}.
    if (page_size <= 0 || page_size % kBc != 0)
        return cudaErrorInvalidValue;
    if (K_chan_scale == nullptr || V_chan_scale == nullptr)
        return cudaErrorInvalidValue;
    if (total_work <= 0)
        return cudaSuccess;                  // nothing to do

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    const int max_grid = sm_count * kCtasPerSm;
    const int grid_x   = (total_work < max_grid) ? total_work : max_grid;

    constexpr int kThreads = (kBr / 16) * 32;
    dim3 grid(grid_x);
    dim3 block(kThreads);

    using Layouts = SmemLayouts<kBr, kBc, kHeadDim, kStages>;
    // K and V scales both in registers; no per-tile scale smem slab.
    const int smem_bytes =
        cosize(typename Layouts::K{})
      + cosize(typename Layouts::V{})
      + cosize(typename Layouts::P{});

    auto* kfn = paged_prefill_kernel<kHeadDim, kBr, kBc, kStages, kCtasPerSm, kIsCausal>;
    cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    kfn<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(Q),
        reinterpret_cast<const uint8_t*>(Qs),
        reinterpret_cast<const __nv_fp8_e4m3*>(K_pool),
        reinterpret_cast<const uint8_t*>(K_chan_scale),
        reinterpret_cast<const __nv_fp8_e4m3*>(V_pool),
        reinterpret_cast<const float*>(V_chan_scale),
        reinterpret_cast<const int32_t*>(qo_indptr),
        reinterpret_cast<const int32_t*>(paged_kv_indices),
        reinterpret_cast<const int32_t*>(paged_kv_indptr),
        reinterpret_cast<const int32_t*>(paged_kv_last_page_len),
        reinterpret_cast<const int32_t*>(work_units),
        reinterpret_cast<__nv_bfloat16*>(O),
        total_work, num_q_heads, num_kv_heads, page_size,
        softmax_scale * 1.4426950408889634f);    // × log2(e)
    return cudaGetLastError();
}

}  // namespace flash_attn_sm120_paged_prefill
