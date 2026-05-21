// Implementation header for the MXFP8 FlashAttention forward kernel.
// Included by per-D translation units (d32.cu, d64.cu,
// d128.cu) so each D specialisation gets its own ptxas pass —
// avoiding the shared-codegen-state problem that blocked R6 / R9 / R10 /
// v16 P3 (any inner-loop branch perturbed D=128 ptxas allocator).
//
// See docs (kept under blockscale_attention legacy) for the full algorithm and tuning history.
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

namespace flash_attn_sm120 {

using namespace cute;

// ---------- Element types (atom-invariant) ---------------------------------

using ElementA   = float_e4m3_t;
using ElementB   = float_e4m3_t;
using ElementSF  = float_ue8m0_t;
using ElementAcc = float;
using ElementOut = bfloat16_t;

constexpr int kBlock3 = 32;            // K-block size for UE8M0 scales (sf_vec_size)

using MmaAtom = MMA_Atom<SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    ElementA, ElementB, ElementAcc, ElementSF, /*sf_vec_size=*/32>>;

// kNumWarps × atom_M(16) = Br. kNumWarps = Br / 16.
// (R5 used a hardcoded 4-warp TiledMma; v16 lifts it to a template so per-D
// dispatch can pick Br ∈ {64, 128} → 4 / 8 warp configurations.)
template <int kNumWarps>
using TiledMmaT = TiledMMA<MmaAtom, Layout<Shape<Int<kNumWarps>, _1, _1>>>;

// ---------- SwizzledKAtom selector -----------------------------------------
// Pick the GMMA::Layout_K_SW{32,64,128}_Atom whose K-extent equals the K-dim
// (FP8 = 1 byte/elem so byte count == element count). Used for K, V, P.

template <int K>
struct SwizzledKAtom;

template <>
struct SwizzledKAtom< 32> { using type = GMMA::Layout_K_SW32_Atom<ElementA>; };
template <>
struct SwizzledKAtom< 64> { using type = GMMA::Layout_K_SW64_Atom<ElementA>; };
template <>
struct SwizzledKAtom<128> { using type = GMMA::Layout_K_SW128_Atom<ElementA>; };
// D=256 reuses SW128 — tile_to_shape replicates twice along the K-dim.
template <>
struct SwizzledKAtom<256> { using type = GMMA::Layout_K_SW128_Atom<ElementA>; };

// ---------- Per-instance smem layouts --------------------------------------
// K is the QK B operand: shape [Bc, kD] with kD as the K-dim — Layout_K_SW{kD}.
// V is the PV B operand: shape [kD, Bc] with Bc as the K-dim — Layout_K_SW{Bc}.
// P is the PV A operand: shape [Br, Bc] with Bc as the K-dim — same as V.
// Swizzle<*,4,3>: XOR only on offset bits 4..6, so 16-byte cp.async chunks
// (bits 0..3) and 4-byte uint32 reads (bits 0..1) survive at the same physical
// byte alignment.

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

// ---------- Copy atoms (atom-invariant) ------------------------------------
// gmem->smem cp.async (16 B per thread).
using GmemCopyAtomAB = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, ElementA>;
// smem->reg ldmatrix.b16. Atom granularity must match the per-thread fragment
// the SM120 mxf8 m16n8k32 atom delivers per atom-call:
//   A-frag = 16 FP8 = 4 uint32 -> SM75_U32x4_LDSM_N
//   B-frag =  8 FP8 = 2 uint32 -> SM75_U32x2_LDSM_N
// Using x4 for B trips a CuTe static_assert ("TiledCopy uses too few vals").
using SmemCopyAtomA  = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
using SmemCopyAtomB  = Copy_Atom<SM75_U32x2_LDSM_N, ElementB>;

// ---------- Inline PTX helpers (mma + cp.async wait) ------------------------

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

// ---------- Software 2^x via FFMA pipe (no MUFU.EX2) -----------------------
//
// Specialised for x ∈ (-30, 0]: the post-softmax-max domain. Cubic minimax
// approximation over [0, 1) for the fractional part, integer part via bit
// reinterpret of the IEEE-754 exponent field.
//
// All ops lie on the FFMA / INT pipe — no SFU. Used in pair with hardware
// `exp2f` (MUFU.EX2) inside the softmax loop so the warp scheduler can
// dual-issue MUFU + FFMA on the same cycle, hiding the scarce-SFU pipe
// latency. R6 experiment, gated on FSO_ATTN_SOFTMAX_DUAL.
//
// Accuracy: ~12-bit relative (max abs error ~3e-4 over the range). After
// softmax sum normalisation the error budget is comparable to FP8 quant
// noise, so cos similarity is empirically unchanged at the 0.9995 threshold
// (validated below in tests/attention/test_sm120_mxfp8_attn.py).

__device__ __forceinline__ float exp2f_fma(float x) {
    int xi = __float2int_rd(x);                 // floor → INT pipe
    if (xi <= -30) return 0.0f;
    float xf = x - __int2float_rn(xi);          // fractional ∈ [0, 1) → FFMA
    float p;
    p = fmaf(xf, 0.0555041f, 0.2402265f);
    p = fmaf(xf, p,          0.6931472f);
    p = fmaf(xf, p,          1.0f);             // 2^xf cubic minimax
    int e = (xi + 127) << 23;                   // 2^xi via IEEE-754 exponent
    return __int_as_float(e) * p;
}

// v19 micro-opt: pack 2 fp32 → 2 fp8 (e4m3) in one PTX instruction.
//   cvt.rn.satfinite.e4m3x2.f32 d, high, low
//     d[15:8] = e4m3(high), d[7:0] = e4m3(low)
// Returns uint16 with `a` at byte 0 (low) and `b` at byte 1 (high). Used by
// the P-quant inner loop, paired with STS.U16 to halve both conversions and
// stores vs the original per-byte path.
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

// Pack 2 fp32 → 2 bf16 in one PTX instruction.
//   cvt.rn.bf16x2.f32 d, high, low
//     d[31:16] = bf16(high), d[15:0] = bf16(low)
// Returns uint32 with `a` in low 16 bits and `b` in high 16 bits. Used in
// the epilogue, paired with STG.B32 to halve fp32→bf16 conversions and
// global stores vs the per-element __float2bfloat16 + STG.B16 path.
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
    // sm < 1200 stub: kind::mxf8f6f4 mma is not encodable. Empty body so
    // nvcc can lower the kernel template for sm_90a even though the host
    // dispatcher will never launch it there.
    (void)d0; (void)d1; (void)d2; (void)d3;
    (void)a0; (void)a1; (void)a2; (void)a3;
    (void)b0; (void)b1; (void)sa; (void)sb;
#endif
}

// ---------- Kernel template ------------------------------------------------
//
// Specialised on:
//   kHeadDim ∈ {32, 64, 128}   — head dim D (atom-K = 32 so D % 32 == 0)
//   kBr, kBc                   — Q-tile rows / KV-tile cols (multiples of 32)
//   kStages                    — cp.async pipeline depth (typically 2)
//   kIsCausal                  — compile-time causal mask (eliminates per-(nt,kb) branches)
// kBr is fixed at 64 in all current instances (4 warps × atom_M=16). Bigger
// per-D specialisations (e.g. D=32, Bc=128) shrink V/K smem and let us trade
// kv-tile count for per-tile work without breaking 2-CTA/SM occupancy.

// R7 experiment: nvcc / ptxas's register allocator and instruction scheduler
// are empirically sensitive to function symbol-name length and namespace
// hashing. CUTLASS folklore: putting `cutlass_kernel` (or similar
// CUTLASS-style tags) in the symbol can flip ptxas onto a different codegen
// path. R4→R5 (320 → 369 TF, +15 %) was driven by exactly this kind of
// codegen-path shift after template cleanup; this experiment tests whether
// adding a `cutlass_kernel`-tagged wrapper triggers another such shift.
//
// Compile with -DFSO_ATTN_CUTLASS_NAME to rebuild the kernel under the
// `mxfp8_attn_fwd_cutlass_kernel` symbol; default keeps the original
// name. Bench A/B with and without to measure the codegen-path delta.
#ifdef FSO_ATTN_CUTLASS_NAME
#define KERNEL_NAME mxfp8_attn_fwd_cutlass_kernel
#else
#define KERNEL_NAME mxfp8_attn_fwd_kernel
#endif

template <int kHeadDim, int kBr, int kBc, int kStages, int kCtasPerSm, bool kIsCausal>
__global__
#ifndef FSO_ATTN_NO_LAUNCH_BOUNDS
__launch_bounds__((kBr / 16) * 32, kCtasPerSm)
#endif
void KERNEL_NAME(
    const __nv_fp8_e4m3* __restrict__ Q,         // [B, Sq, H_q,  D]
    const uint8_t*       __restrict__ Qs,        // [B, Sq/16, H_q, D/32]
    const __nv_fp8_e4m3* __restrict__ K,         // [B, Sk, H_kv, D]
    const uint8_t*       __restrict__ Ks,        // [B, Sk/Bc, H_kv, D/32]
    const __nv_fp8_e4m3* __restrict__ V,         // [B, D,  H_kv, Sk] FP8
    const uint8_t*       __restrict__ Vs,        // [B, Sk/Bc, H_kv, Bc/32]
    __nv_bfloat16*       __restrict__ O,         // [B, Sq, H_q, D]
    int batch, int num_q_heads, int num_kv_heads,
    int seq_q, int seq_k,
    float softmax_scale,
    int total_work)
{
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 1200)
    using Layouts = SmemLayouts<kBr, kBc, kHeadDim, kStages>;
    using SmemLayoutK = typename Layouts::K;
    using SmemLayoutV = typename Layouts::V;
    using SmemLayoutP = typename Layouts::P;

    constexpr int kD             = kHeadDim;
    constexpr int kNumWarps      = kBr / 16;              // kBr = kNumWarps × atom_M(16)
    constexpr int kThreads       = kNumWarps * 32;        // CTA size (= __launch_bounds__'s 1st arg)
    using TiledMma = TiledMmaT<kNumWarps>;
    constexpr int n_kblk         = kD / kBlock3;          // QK K-blocks
    constexpr int n_v_kblk_per_tile = kBc / kBlock3;      // PV K-blocks per kv tile
    constexpr int n_qk_n_tiles   = kBc / 8;               // QK atom-N tiles (atom_N=8)
    constexpr int n_pv_n_tiles   = kD / 8;                // PV atom-N tiles

    const int tid    = threadIdx.x;
    const int warp   = tid / 32;
    const int lane   = tid % 32;
    const int row    = lane / 4;
    const int tid_g  = lane % 4;

    const int q_tiles_per_head = seq_q / kBr;
    const int wu_per_batch     = num_q_heads * q_tiles_per_head;
    const int q_row_stride     = num_q_heads  * kD;       // gmem stride along Sq
    const int kv_row_stride    = num_kv_heads * kD;       // gmem stride along Sk for K
    const int H_kv_times_S     = num_kv_heads * seq_k;    // V stride for D-axis
    const int gqa_group        = num_q_heads / num_kv_heads;  // q-heads per kv-head

    // ---------- Smem layout: swizzled K/V/P slabs --------------------------
    extern __shared__ unsigned char smem_raw[];
    auto* smem_K_ptr = reinterpret_cast<ElementA*>(smem_raw);
    auto* smem_V_ptr = smem_K_ptr + cosize(SmemLayoutK{});
    auto* smem_P_ptr = smem_V_ptr + cosize(SmemLayoutV{});
    auto* smem_Ks_ptr = reinterpret_cast<uint8_t*>(smem_P_ptr + cosize(SmemLayoutP{}));
    auto* smem_Vs_ptr = smem_Ks_ptr + kStages * n_kblk;

    // smem→reg reads go through cute::copy driven by Copy_Atom<SM75_U32x{2,4}_LDSM_N>;
    // for cp.async writes we still call the raw layout offset (Tensor.operator()
    // proxy returns by value on swizzled ComposedLayout — bug caught in step-A).
    constexpr auto layoutK = SmemLayoutK{};
    constexpr auto layoutV = SmemLayoutV{};
    constexpr auto layoutP = SmemLayoutP{};

    Tensor sK_pi = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(smem_K_ptr), SmemLayoutK{}));      // (Bc, D, S)
    Tensor sV_pi = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(smem_V_ptr), SmemLayoutV{}));      // (D, Bc, S)
    Tensor sP_pi = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(smem_P_ptr), SmemLayoutP{}));      // (Br, Bc)

    TiledMma mma;
    auto thr_mma = mma.get_thread_slice(tid);

    // QK: K is the B operand (N=Bc, K=D contraction).
    auto s2r_copy_K = make_tiled_copy_B(SmemCopyAtomB{}, mma);
    auto thr_s2r_K  = s2r_copy_K.get_thread_slice(tid);
    auto tCrK = thr_mma.partition_fragment_B(sK_pi(_, _, Int<0>{}));   // (MMA, MMA_N, MMA_K)
    auto tXsK = thr_s2r_K.partition_S(sK_pi);                          // (CPY, CPY_N, CPY_K, S)
    auto tXrK = thr_s2r_K.retile_D(tCrK);                              // (CPY, CPY_N, CPY_K)

    // PV: V is the B operand (N=D, K=Bc contraction).
    auto s2r_copy_V = make_tiled_copy_B(SmemCopyAtomB{}, mma);
    auto thr_s2r_V  = s2r_copy_V.get_thread_slice(tid);
    auto tCrV = thr_mma.partition_fragment_B(sV_pi(_, _, Int<0>{}));   // (MMA, MMA_N=16, MMA_K=2)
    auto tXsV = thr_s2r_V.partition_S(sV_pi);                          // (CPY, CPY_N, CPY_K, S)
    auto tXrV = thr_s2r_V.retile_D(tCrV);

    // PV: P is the A operand (M=Br, K=Bc contraction). One stage (no pipeline).
    auto s2r_copy_P = make_tiled_copy_A(SmemCopyAtomA{}, mma);
    auto thr_s2r_P  = s2r_copy_P.get_thread_slice(tid);
    auto tCrP = thr_mma.partition_fragment_A(sP_pi);                   // (MMA, MMA_M, MMA_K)
    auto tXsP = thr_s2r_P.partition_S(sP_pi);
    auto tXrP = thr_s2r_P.retile_D(tCrP);

    // For inline-PTX mma we feed uint32 operands. recast<uint32_t> on the
    // fragments below collapses the per-atom-element axis (FP8 packed 4-per
    // -uint32) so that tCr*_u32(i, n_idx, k_idx) yields the i-th uint32 of
    // the (n_idx, k_idx) atom instance. The static shape of MMA depends on
    // the atom; for SM120_16x8x32_TN_VS, B-frag has 8 FP8 = 2 uint32 per
    // thread per atom, A-frag has 16 FP8 = 4 uint32 per thread per atom.

    // ---------- Persistent CTA grid-stride loop -----------------------------
    for (int wu = blockIdx.x; wu < total_work; wu += gridDim.x) {
        const int b      = wu / wu_per_batch;
        const int h      = (wu / q_tiles_per_head) % num_q_heads;
        const int q_tile = wu % q_tiles_per_head;

        // GQA: integer-divide Q-head into its KV-head group. For MHA the group
        // size is 1 so h_kv == h. K/V/Ks/Vs all index by h_kv; Q/Qs/O index by h.
        const int h_kv = h / gqa_group;

        const int q_row_base = q_tile * kBr + warp * 16;

        const auto* Q_base   = Q  + ((int64_t)b * seq_q + q_row_base) * q_row_stride + h * kD;
        const auto* Qs_base  = Qs + ((int64_t)b * (seq_q / 16) + q_tile * kNumWarps + warp) * num_q_heads * n_kblk + h * n_kblk;
        const auto* K_base_b = K  + ((int64_t)b * seq_k * num_kv_heads * kD) + h_kv * kD;
        const auto* Ks_base_b= Ks + ((int64_t)b * (seq_k / kBc) * num_kv_heads * n_kblk) + h_kv * n_kblk;
        const auto* V_base_bh= V  + ((int64_t)b * kD * H_kv_times_S) + h_kv * seq_k;
        const auto* Vs_base_b= Vs + ((int64_t)b * (seq_k / kBc) * num_kv_heads * n_v_kblk_per_tile) + h_kv * n_v_kblk_per_tile;
        auto* O_base = O + ((int64_t)b * seq_q + q_row_base) * q_row_stride + h * kD;

        // ---------- Q frag + Q scales: register-resident -------------------
        uint32_t qfrag[n_kblk][4];
        #pragma unroll
        for (int kb = 0; kb < n_kblk; ++kb) {
            const int k0 = kb * kBlock3;
            const int a_col0 = tid_g * 4 + k0;
            const int a_col1 = a_col0 + 16;
            qfrag[kb][0] = *reinterpret_cast<const uint32_t*>(Q_base + (row + 0) * q_row_stride + a_col0);
            qfrag[kb][1] = *reinterpret_cast<const uint32_t*>(Q_base + (row + 8) * q_row_stride + a_col0);
            qfrag[kb][2] = *reinterpret_cast<const uint32_t*>(Q_base + (row + 0) * q_row_stride + a_col1);
            qfrag[kb][3] = *reinterpret_cast<const uint32_t*>(Q_base + (row + 8) * q_row_stride + a_col1);
        }
        uint8_t qs[n_kblk];
        #pragma unroll
        for (int kb = 0; kb < n_kblk; ++kb) qs[kb] = Qs_base[kb];

        // ---------- Per-row softmax / PV state -----------------------------
        float m_lo = -INFINITY, m_hi = -INFINITY;
        float l_lo = 0.f,        l_hi = 0.f;
        float pv_acc[n_pv_n_tiles][4];
        #pragma unroll
        for (int nt = 0; nt < n_pv_n_tiles; ++nt) {
            pv_acc[nt][0] = pv_acc[nt][1] = pv_acc[nt][2] = pv_acc[nt][3] = 0.f;
        }

        // ---------- KV tile bound (causal short-circuit) -------------------
        int n_kv_tiles = seq_k / kBc;
        if constexpr (kIsCausal) {
            const int q_row_max_cta = q_tile * kBr + kBr - 1;
            const int n_eff = (q_row_max_cta / kBc) + 1;
            if (n_eff < n_kv_tiles) n_kv_tiles = n_eff;
        }

        // ---------- Issue cp.async lambdas ---------------------------------
        // K tile: load Bc rows × D cols of FP8 into smem_K[stage].
        //   Bc*D / 16 cp.async ops distributed over kThreads (= kNumWarps * 32).
        auto issue_cp_async_K = [&](int stage, int kv_tile) {
            constexpr int kOps = (kBc * kD) / 16;
            constexpr int kChunksPerRow = kD / 16;
            #pragma unroll
            for (int i = tid; i < kOps; i += kThreads) {
                const int s_idx  = i / kChunksPerRow;
                const int d_offs = (i % kChunksPerRow) * 16;
                const auto* src = K_base_b + ((int64_t)kv_tile * kBc + s_idx) * kv_row_stride + d_offs;
                // Layout function returns the *swizzled* byte offset for the
                // ElementA (1-byte FP8). The 16 consecutive K bytes are
                // contiguous in physical smem because Swizzle<3,4,3> only
                // affects bits 4..6 of the offset.
                int off = layoutK(s_idx, d_offs, stage);
                uint32_t dst = smem_to_uint(smem_K_ptr + off);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                              :: "r"(dst), "l"(src));
            }
        };
        // V tile: load D rows × Bc cols of FP8 into smem_V[stage].
        //   D*Bc / 16 cp.async ops distributed over kThreads.
        auto issue_cp_async_V = [&](int stage, int kv_tile) {
            constexpr int kOps = (kD * kBc) / 16;
            constexpr int kChunksPerRow = kBc / 16;
            #pragma unroll
            for (int i = tid; i < kOps; i += kThreads) {
                const int d_idx    = i / kChunksPerRow;
                const int s_offset = (i % kChunksPerRow) * 16;
                const auto* src = V_base_bh + (int64_t)d_idx * H_kv_times_S + kv_tile * kBc + s_offset;
                int off = layoutV(d_idx, s_offset, stage);
                uint32_t dst = smem_to_uint(smem_V_ptr + off);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                              :: "r"(dst), "l"(src));
            }
        };

        auto issue_scales = [&](int stage, int kv_tile_idx) {
            if (tid < n_kblk)
                smem_Ks_ptr[stage * n_kblk + tid] =
                    Ks_base_b[(int64_t)kv_tile_idx * num_kv_heads * n_kblk + tid];
            if (tid < n_v_kblk_per_tile)
                smem_Vs_ptr[stage * n_v_kblk_per_tile + tid] =
                    Vs_base_b[(int64_t)kv_tile_idx * num_kv_heads * n_v_kblk_per_tile + tid];
        };

        // ---------- Prologue: prefetch first (kStages-1) tiles -------------
        // For kStages == 2 we keep the exact R5 inline form (single prefetch
        // without the `if (s < n_kv_tiles)` guard) so the SASS stays
        // byte-identical to the regression contract. For kStages >= 3 we use
        // a guarded loop so the prologue handles the n_kv_tiles < kStages-1
        // edge case too.
        if constexpr (kStages == 2) {
            issue_cp_async_K(0, 0);
            issue_cp_async_V(0, 0);
            issue_scales(0, 0);
            cp_async_commit_group();
        } else {
            #pragma unroll
            for (int s = 0; s < kStages - 1; ++s) {
                if (s < n_kv_tiles) {
                    issue_cp_async_K(s, s);
                    issue_cp_async_V(s, s);
                    issue_scales(s, s);
                }
                cp_async_commit_group();
            }
        }

        // 2-stage swap state (used only when kStages == 2; DCE'd otherwise).
        int curr = 0, next = 1;

        // ---------- KV tile loop -------------------------------------------
        for (int kv_tile = 0; kv_tile < n_kv_tiles; ++kv_tile) {

            if constexpr (kStages == 2) {
                // R5 byte-identical 2-stage path: per-iter prefetch + curr/next swap.
                if (kv_tile + 1 < n_kv_tiles) {
                    issue_cp_async_K(next, kv_tile + 1);
                    issue_cp_async_V(next, kv_tile + 1);
                    issue_scales(next, kv_tile + 1);
                    cp_async_commit_group();
                    cp_async_wait_group<1>();
                } else {
                    cp_async_wait_group<0>();
                }
            } else {
                // General kStages>=3 modular ring: prefetch ahead by (kStages-1),
                // always commit (empty group when out of tiles), wait<kStages-1>.
                // `curr` is recomputed from kv_tile each iter so the shared compute
                // body below uses the right stage in both paths.
                curr = kv_tile % kStages;
                const int prefetch_tile = kv_tile + (kStages - 1);
                if (prefetch_tile < n_kv_tiles) {
                    const int prefetch_stage = prefetch_tile % kStages;
                    issue_cp_async_K(prefetch_stage, prefetch_tile);
                    issue_cp_async_V(prefetch_stage, prefetch_tile);
                    issue_scales(prefetch_stage, prefetch_tile);
                }
                cp_async_commit_group();
                cp_async_wait_group<kStages - 1>();
            }
            __syncthreads();

            // ---------- ldmatrix K for stage `curr` (cooperative b16) -----
            cute::copy(s2r_copy_K, tXsK(_, _, _, curr), tXrK);
            auto tCrK_u32 = recast<uint32_t>(tCrK);   // (2, n_qk_n_tiles, n_kblk)

            // v18 micro-opt: hoist Ks scales out of the nt×kb loop. ptxas
            // didn't auto-hoist `smem_Ks_ptr[curr*n_kblk+kb]` (runtime curr)
            // → 32 redundant LDS.U8 per warp/kv-tile. Pre-loading into a
            // local register array saves ~28 LDS.U8. PV side already used
            // this pattern for v_scale; v18 mirrors it for QK.
            uint8_t ks_local[n_kblk];
            #pragma unroll
            for (int kb = 0; kb < n_kblk; ++kb) {
                ks_local[kb] = smem_Ks_ptr[curr * n_kblk + kb];
            }

            // ---------- QK ------------------------------------------------
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
                             b0, b1, qs[kb], ks_local[kb]);   // v18: hoisted
                }
                s_frag[nt][0] = d0 * softmax_scale;
                s_frag[nt][1] = d1 * softmax_scale;
                s_frag[nt][2] = d2 * softmax_scale;
                s_frag[nt][3] = d3 * softmax_scale;

                if constexpr (kIsCausal) {
                    // v16 P1: D-specific mask path. D=32 has reg headroom
                    // (baseline 244/0 spill vs D=128 232/0) so it can absorb
                    // the runtime branch's codegen perturbation AND capture
                    // R10's algorithmic save (skip mask check on tiles
                    // strictly below the diagonal). D=64 / D=128 stay on
                    // R5's "always predicate" path — the if-constexpr false
                    // branch is fully DCE'd in the corresponding SASS, so
                    // their cuobjdump reg/spill is unchanged from R5.
                    if constexpr (kHeadDim == 32) {
                        if (kv_tile == n_kv_tiles - 1) {     // diagonal-only
                            const int q0 = q_row_base + row + 0;
                            const int q8 = q_row_base + row + 8;
                            const int kc0 = kv_tile * kBc + nt * 8 + tid_g * 2 + 0;
                            const int kc1 = kv_tile * kBc + nt * 8 + tid_g * 2 + 1;
                            if (kc0 > q0) s_frag[nt][0] = -INFINITY;
                            if (kc1 > q0) s_frag[nt][1] = -INFINITY;
                            if (kc0 > q8) s_frag[nt][2] = -INFINITY;
                            if (kc1 > q8) s_frag[nt][3] = -INFINITY;
                        }
                    } else {
                        const int q0 = q_row_base + row + 0;
                        const int q8 = q_row_base + row + 8;
                        const int kc0 = kv_tile * kBc + nt * 8 + tid_g * 2 + 0;
                        const int kc1 = kv_tile * kBc + nt * 8 + tid_g * 2 + 1;
                        if (kc0 > q0) s_frag[nt][0] = -INFINITY;
                        if (kc1 > q0) s_frag[nt][1] = -INFINITY;
                        if (kc0 > q8) s_frag[nt][2] = -INFINITY;
                        if (kc1 > q8) s_frag[nt][3] = -INFINITY;
                    }
                }
            }

            // ---------- Online softmax (Bc -> n_qk_n_tiles N-tiles) ------
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
#ifdef FSO_ATTN_SKIP_RESCALE
            // FA4-style rescale skip (per Dao-AILab/flash-attention cute/softmax.py
            // SoftmaxSm100::update_row_max). When the new max grows by at most
            // `kRescaleThreshold` (log2 domain) over the running max, REVERT
            // the max to its previous value and pretend alpha = 1, skipping
            // the per-(nt, kb) pv_acc rescale. Trades up to exp2(threshold)
            // alpha error per skipped iter for ~64 FFMA + 1 exp2f saved per
            // kv tile (D=128, n_pv_n_tiles=16). The warp-uniform `__all_sync`
            // gate is what actually shaves the FFMAs — per-lane skip would
            // be predicated and still consume issue slots.
#ifndef FSO_ATTN_RESCALE_THRESHOLD
#define FSO_ATTN_RESCALE_THRESHOLD 0.0625f
#endif
            constexpr float kRescaleThreshold = FSO_ATTN_RESCALE_THRESHOLD;  // alpha ∈ [2^-thr, 1]
            float new_max_lo = fmaxf(m_lo, lmax_lo);
            float new_max_hi = fmaxf(m_hi, lmax_hi);
            const bool can_skip_lo = (m_lo != -INFINITY) && (lmax_lo - m_lo <= kRescaleThreshold);
            const bool can_skip_hi = (m_hi != -INFINITY) && (lmax_hi - m_hi <= kRescaleThreshold);
            if (can_skip_lo) new_max_lo = m_lo;
            if (can_skip_hi) new_max_hi = m_hi;
            const float alpha_lo = can_skip_lo ? 1.0f : ((m_lo == -INFINITY) ? 1.0f : exp2f(m_lo - new_max_lo));
            const float alpha_hi = can_skip_hi ? 1.0f : ((m_hi == -INFINITY) ? 1.0f : exp2f(m_hi - new_max_hi));
            const bool warp_skip_rescale = __all_sync(0xffffffffu, can_skip_lo & can_skip_hi);
#else
            const float new_max_lo = fmaxf(m_lo, lmax_lo);
            const float new_max_hi = fmaxf(m_hi, lmax_hi);
            const float alpha_lo = (m_lo == -INFINITY) ? 1.0f : exp2f(m_lo - new_max_lo);
            const float alpha_hi = (m_hi == -INFINITY) ? 1.0f : exp2f(m_hi - new_max_hi);
#endif

            float lsum_lo = 0.f, lsum_hi = 0.f;
#if defined(FSO_ATTN_SOFTMAX_DUAL)
            // Dual-issue: lo half stays on MUFU.EX2, hi half on FFMA via the
            // software cubic. Compiler should interleave so warp scheduler
            // co-issues MUFU + FFMA per cycle rather than sequentialising the
            // SFU pipe.
            #pragma unroll
            for (int nt = 0; nt < n_qk_n_tiles; ++nt) {
                s_frag[nt][0] = exp2f    (s_frag[nt][0] - new_max_lo); lsum_lo += s_frag[nt][0];
                s_frag[nt][1] = exp2f    (s_frag[nt][1] - new_max_lo); lsum_lo += s_frag[nt][1];
                s_frag[nt][2] = exp2f_fma(s_frag[nt][2] - new_max_hi); lsum_hi += s_frag[nt][2];
                s_frag[nt][3] = exp2f_fma(s_frag[nt][3] - new_max_hi); lsum_hi += s_frag[nt][3];
            }
#else
            #pragma unroll
            for (int nt = 0; nt < n_qk_n_tiles; ++nt) {
                s_frag[nt][0] = exp2f(s_frag[nt][0] - new_max_lo); lsum_lo += s_frag[nt][0];
                s_frag[nt][1] = exp2f(s_frag[nt][1] - new_max_lo); lsum_lo += s_frag[nt][1];
                s_frag[nt][2] = exp2f(s_frag[nt][2] - new_max_hi); lsum_hi += s_frag[nt][2];
                s_frag[nt][3] = exp2f(s_frag[nt][3] - new_max_hi); lsum_hi += s_frag[nt][3];
            }
#endif
            #pragma unroll
            for (int o = 2; o > 0; o >>= 1) {
                lsum_lo += __shfl_xor_sync(0xffffffffu, lsum_lo, o);
                lsum_hi += __shfl_xor_sync(0xffffffffu, lsum_hi, o);
            }

            l_lo = l_lo * alpha_lo + lsum_lo;
            l_hi = l_hi * alpha_hi + lsum_hi;
            m_lo = new_max_lo;
            m_hi = new_max_hi;
#ifdef FSO_ATTN_SKIP_RESCALE
            // Warp-uniform skip: when ALL lanes can defer the max update
            // (alpha == 1.0 for both halves), branch around the 64-FFMA
            // rescale entirely. When any lane needs a real rescale, fall
            // through and multiply (lanes with alpha = 1 are still touched
            // but the FFMA is no-op for them).
            if (!warp_skip_rescale) {
                #pragma unroll
                for (int nt = 0; nt < n_pv_n_tiles; ++nt) {
                    pv_acc[nt][0] *= alpha_lo;
                    pv_acc[nt][1] *= alpha_lo;
                    pv_acc[nt][2] *= alpha_hi;
                    pv_acc[nt][3] *= alpha_hi;
                }
            }
#else
            #pragma unroll
            for (int nt = 0; nt < n_pv_n_tiles; ++nt) {
                pv_acc[nt][0] *= alpha_lo;
                pv_acc[nt][1] *= alpha_lo;
                pv_acc[nt][2] *= alpha_hi;
                pv_acc[nt][3] *= alpha_hi;
            }
#endif

            // ---------- P quant -> smem_P slab (v19: pack via STS.U16) -----
            // v15a path: 4 separate fp32→fp8 conversions + 4 STS.B8 per nt.
            // v19 fuses to 2 PTX `cvt.e4m3x2.f32` (each packs 2 fp32→2 fp8)
            // and 2 STS.U16 per nt. Halves both conversion and store counts.
            // Adjacent columns (n_base+0, n_base+1) live at consecutive byte
            // offsets in the SW64-swizzled P layout — bit 0 doesn't get
            // perturbed by the Swizzle<*,4,3> XOR (bits 4..6 only).
            constexpr float   kPFp8Scale = 256.0f;
            constexpr uint8_t kPScaleByte = 119;
            const int row_lo = warp * 16 + (row + 0);
            const int row_hi = warp * 16 + (row + 8);
            #pragma unroll
            for (int nt = 0; nt < n_qk_n_tiles; ++nt) {
                const int n_base = nt * 8 + tid_g * 2;
                const int o_lo = layoutP(row_lo, n_base + 0);   // o_lo+1 == layoutP(row_lo, n_base+1)
                const int o_hi = layoutP(row_hi, n_base + 0);
                uint16_t pack_lo = cvt_e4m3x2_fp32(
                    s_frag[nt][0] * kPFp8Scale, s_frag[nt][1] * kPFp8Scale);
                uint16_t pack_hi = cvt_e4m3x2_fp32(
                    s_frag[nt][2] * kPFp8Scale, s_frag[nt][3] * kPFp8Scale);
                *reinterpret_cast<uint16_t*>(smem_P_ptr + o_lo) = pack_lo;
                *reinterpret_cast<uint16_t*>(smem_P_ptr + o_hi) = pack_hi;
            }
            __syncwarp();

            // ---------- ldmatrix P (intra-warp, after the warp-local quant) and V ---
            cute::copy(s2r_copy_P, tXsP, tXrP);                  // (MMA, MMA_M, MMA_K)
            cute::copy(s2r_copy_V, tXsV(_, _, _, curr), tXrV);   // (MMA, MMA_N, MMA_K)
            auto tCrP_u32 = recast<uint32_t>(tCrP);              // (4, MMA_M, n_v_kblk_per_tile)
            auto tCrV_u32 = recast<uint32_t>(tCrV);              // (2, n_pv_n_tiles, n_v_kblk_per_tile)

            // ---------- PV: pv_acc += P_fp8 @ V_fp8 (kb-outer, nt-inner) --
            #pragma unroll
            for (int kb = 0; kb < n_v_kblk_per_tile; ++kb) {
                uint32_t a0 = tCrP_u32(0, 0, kb);
                uint32_t a1 = tCrP_u32(1, 0, kb);
                uint32_t a2 = tCrP_u32(2, 0, kb);
                uint32_t a3 = tCrP_u32(3, 0, kb);
                const uint8_t v_scale = smem_Vs_ptr[curr * n_v_kblk_per_tile + kb];

                #pragma unroll
                for (int nt = 0; nt < n_pv_n_tiles; ++nt) {
                    uint32_t b0 = tCrV_u32(0, nt, kb);
                    uint32_t b1 = tCrV_u32(1, nt, kb);
                    mma_mxf8(pv_acc[nt][0], pv_acc[nt][1], pv_acc[nt][2], pv_acc[nt][3],
                             a0, a1, a2, a3, b0, b1,
                             kPScaleByte, v_scale);
                }
            }

            if constexpr (kStages == 2) {
                // R5 verbatim 2-stage swap. For kStages>=3 `curr` is recomputed
                // each iter at the loop head, so no swap is needed.
                const int tmp = curr; curr = next; next = tmp;
            }
            __syncthreads();
        }

        // ---------- Epilogue ----------------------------------------------
        // Epilogue (v19 micro-opt: bf16x2 pack via cvt.rn.bf16x2.f32 + STG.B32).
        // The two adjacent output cols at (n_base+0, n_base+1) are 2-byte-stride
        // in bf16, so they pack into one uint32. Per nt: 4 byte stores collapse
        // to 2 uint32 stores, and the 4 fp32→bf16 conversions collapse to 2.
        const float inv_l_lo = (l_lo > 0.f) ? (1.f / l_lo) : 0.f;
        const float inv_l_hi = (l_hi > 0.f) ? (1.f / l_hi) : 0.f;
        #pragma unroll
        for (int nt = 0; nt < n_pv_n_tiles; ++nt) {
            const int n_base = nt * 8 + tid_g * 2;
            uint32_t pack_lo = cvt_bf16x2_fp32(
                pv_acc[nt][0] * inv_l_lo, pv_acc[nt][1] * inv_l_lo);
            uint32_t pack_hi = cvt_bf16x2_fp32(
                pv_acc[nt][2] * inv_l_hi, pv_acc[nt][3] * inv_l_hi);
            *reinterpret_cast<uint32_t*>(O_base + (row + 0) * q_row_stride + n_base) = pack_lo;
            *reinterpret_cast<uint32_t*>(O_base + (row + 8) * q_row_stride + n_base) = pack_hi;
        }

        __syncthreads();
    }   // for wu

    // R8 experiment: removing the dead __deadbeef instantiation block.
    // The block was defensive (force static_asserts in CuTe templates to fire
    // even if the body doesn't reference the types). All five types
    // (TiledMma, SmemCopyAtomA/B, GmemCopyAtomAB, layout aliases) are now
    // referenced live in the kernel body, so the block is redundant. Removing
    // it lets ptxas not track those types' state across the dead branch.
#ifdef FSO_ATTN_KEEP_DEAD_BLOCK
    if (tid == 0xdeadbeef) {
        TiledMma _mma{};
        SmemCopyAtomA _ca{}; SmemCopyAtomB _cb{}; GmemCopyAtomAB _cg{};
        (void)_mma; (void)_ca; (void)_cb; (void)_cg;
    }
#endif
#else
    // sm < 1200 stub: SM120 BlockScaled mxf8f6f4 mma is not encodable on
    // earlier arches. The host dispatcher (mxfp8_attn_fwd_launch) returns
    // cudaErrorNotSupported on sm < 12, so this body is unreachable at
    // runtime; nvcc still needs a legal lowering for the sm_90a SASS pass.
    (void)Q; (void)Qs; (void)K; (void)Ks; (void)V; (void)Vs; (void)O;
    (void)batch; (void)num_q_heads; (void)num_kv_heads;
    (void)seq_q; (void)seq_k; (void)softmax_scale; (void)total_work;
#endif
}

// ---------- Per-(D, causal) launch helper + dispatch -----------------------

template <int kHeadDim, int kBr, int kBc, int kStages, int kCtasPerSm, bool kIsCausal>
cudaError_t launch_impl(
    const void* Q, const void* Qs,
    const void* K, const void* Ks,
    const void* V, const void* Vs,
    void* O,
    int batch, int num_q_heads, int num_kv_heads, int seq_q, int seq_k,
    float softmax_scale, cudaStream_t stream)
{
    if (seq_q % kBr != 0 || seq_k % kBc != 0) return cudaErrorInvalidValue;
    if (num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0) return cudaErrorInvalidValue;
    const int total_work = batch * num_q_heads * (seq_q / kBr);

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    const int max_grid = sm_count * kCtasPerSm;
    const int grid_x   = (total_work < max_grid) ? total_work : max_grid;

    constexpr int kThreads = (kBr / 16) * 32;
    dim3 grid(grid_x);
    dim3 block(kThreads);

    using Layouts = SmemLayouts<kBr, kBc, kHeadDim, kStages>;
    const int smem_bytes =
        cosize(typename Layouts::K{})                  // K[stages]
      + cosize(typename Layouts::V{})                  // V[stages]
      + cosize(typename Layouts::P{})                  // P slab
      + kStages * (kHeadDim / kBlock3)                 // Ks
      + kStages * (kBc      / kBlock3);                // Vs

    auto* kfn = KERNEL_NAME<kHeadDim, kBr, kBc, kStages, kCtasPerSm, kIsCausal>;
    cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    kfn<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(Q),
        reinterpret_cast<const uint8_t*>(Qs),
        reinterpret_cast<const __nv_fp8_e4m3*>(K),
        reinterpret_cast<const uint8_t*>(Ks),
        reinterpret_cast<const __nv_fp8_e4m3*>(V),
        reinterpret_cast<const uint8_t*>(Vs),
        reinterpret_cast<__nv_bfloat16*>(O),
        batch, num_q_heads, num_kv_heads, seq_q, seq_k,
        softmax_scale * 1.4426950408889634f,    // × log2(e)
        total_work);
    return cudaGetLastError();
}

}  // namespace flash_attn_sm120
