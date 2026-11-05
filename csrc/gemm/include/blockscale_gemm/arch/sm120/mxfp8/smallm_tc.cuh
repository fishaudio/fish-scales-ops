/*
 * The sm_120 / sm_121 small-M dense MXFP8 route ("sm120.smallm", issue #12 decision 3): a tensor-core GEMV-like
 * kernel for the decode band, M <= 64, in two forms of one template.
 *
 *   G1 (kSmallmFormQ):    x is the pre-quantized activation of contract v1 (float8_e4m3fn [M, K] and its K-major
 *                         int32 scale words [align(M, 4), K/128]) -- the 2b call linear(xq, w, x_scale=sx).
 *   G2 (kSmallmFormBf16): x is bf16 [M, K]; every CTA quantizes the activation it needs into shared memory with the
 *                         functions of fso/activation_contract.h (the rule of quantize_activation, bit for bit), so the
 *                         bf16-x call is one launch. G2 and quantize_activation + G1 give the same bytes: the staged
 *                         codes and scale bytes are the same, and the accumulation below does not depend on the form.
 *
 * The weight is the prepared sm_120 MXFP8 weight: float8_e4m3fn [N, K] and int32 scale words [N, K/128] with strides
 * (1, N) (word (n, kp) at kp * N + n). The output is bf16 [M, N]; the dense epilogue of interface v1 (bias, erf-GELU,
 * residual; common/epilogue_device.cuh) is applied once, after the cross-warp reduction, by dense_epilogue_value_rt,
 * in its own instantiation (kEpi), so the identity calls run the identity kernel.
 *
 * The arithmetic. One mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32 per (16 weight rows, one
 * 32-element K block, 8 activation rows): the weight rows are the MMA's A (row-major, K contiguous), the activation
 * rows its B columns (M sits on the 8-wide side; up to kNc column groups, so M <= 8 * kNc and the columns M .. 8 kNc-1
 * are zero), the UE8M0 bytes of both operands go straight into the MMA's scale operands, and the tensor core
 * accumulates in fp32. Inside a 32-block the K order is permuted the same way for A and B (MMA slot 4t + i <->
 * element 8t + i, slot 16 + 4t + i <-> element 8t + 4 + i, t = lane % 4), so each lane reads 8 contiguous bytes of
 * each of its two weight rows per MMA (a quad covers a row's 32-byte block: 8 full sectors per warp load).
 * Scale operands (the CUTLASS SM120 SFALayout / SFBLayout for VS = 32 with selectors {0, 0}): lane l supplies the
 * scale of A row (l / 4) + 8 * (l % 2) and of B column l / 4, in byte 0 of its register.
 *
 * The work decomposition. A CTA has kWk (4 or 8) warps; warp wk owns the K blocks wk, wk + kWk, wk + 2 kWk, ... (all
 * warps walk K together, so one step of the CTA reads kWk consecutive 32-byte blocks of each weight row). A tile group is kRt
 * 16-row weight tiles. The grid is persistent: CTA i owns the tile groups i, i + gridDim.x, ... (the host sizes the
 * grid from the SM count, one CTA per SM at most, balanced so that every CTA owns the same number of tile groups
 * +- 1). The activation is staged in shared memory: once per CTA when all of it fits (cj == 0), otherwise in K
 * chunks of cj blocks per warp (cj * kWk blocks), re-staged for every tile group at the step that first needs them.
 * The kWk partial tiles of the K-split warps are summed in a fixed order through shared memory: the result is
 * deterministic and does not depend on the grid, on kNc, on kRt or on the chunking (it does depend on kWk, which
 * the plan derives from the shape alone, the same for G1 and G2).
 *
 * The weight loads go through a register ring of 8 steps, issued ahead of their use and across tile-group boundaries,
 * and the first 8 are issued before griddepcontrol.wait (PDL): the weight is not produced by the previous kernel. x,
 * its scale, bias and residual are read only after the wait; the output is written after it. Nothing is allocated but
 * the output, and no global workspace is used.
 *
 * The step loop (Q120c, 2026-10-29). A tile group's steps are consumed in ring groups of 8. For the plans the rules
 * route (the activation resident in shared memory, cj == 0) the steady-state ring groups run with no runtime bounds:
 * the group's weight / scale / activation bases are advanced by compile-time strides, every step's offsets are
 * immediates, the scale shift is per warp, and the issue of the next group is unconditional; only the last full group
 * and the partial tail (at most two groups per tile group) carry warp-uniform predicates for the tile group's end and
 * the next tile group's prefetch. The chunked plans (cj > 0, FSO_SMALLM_ROUTE=1 only) are a separate instantiation
 * with a generic checked step loop and a countdown over the chunk boundary (no modulo). The per-warp accumulation
 * order is unchanged in both, so the bytes are those of the first library version.
 */
#pragma once

#include "blockscale_gemm/arch/sm120/mxfp8/smallm_route.h"
#include "blockscale_gemm/common/epilogue_device.cuh"
#include "fso/activation_contract.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tensorrt_llm::kernels::blockscale_gemm::sm120_smallm
{

enum : int
{
    kSmallmFormQ = 0,    // G1: pre-quantized x (contract v1)
    kSmallmFormBf16 = 1, // G2: bf16 x, quantized in the kernel
};

constexpr int kRing = fso::sm120_smallm::kRing;           // weight-load ring depth (steps)
constexpr int kRedStride = fso::sm120_smallm::kRedStride; // floats per (warp, tile, column) slab of the reduction buffer
constexpr int kMaxNc = 8;                                 // M <= 64

struct SmallmParams
{
    uint8_t const* w;         // fp8 [N, K]
    int32_t const* ws;        // weight scale words, (kp, n) at kp * N + n
    void const* x;            // G1: fp8 [M, K]; G2: bf16 [M, K]
    int32_t const* sx;        // G1: activation scale words, (kp, m) at kp * align(M, 4) + m
    __nv_bfloat16* y;         // [M, N]
    fso::DenseEpilogue epi;   // bias / residual / act (kEpi instantiations only)
    int M, N, K;
    int cj;                   // 0: x staged once per CTA (resident); > 0: K chunks of cj blocks per warp
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
#define FSO_SMALLM_DEVICE 1
#endif

__device__ __forceinline__ uint2 smallm_ldg64_stream(void const* p)
{
    uint2 r;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
    return r;
}

__device__ __forceinline__ uint32_t smallm_ldg32(void const* p)
{
    uint32_t r;
    asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(r) : "l"(p));
    return r;
}

__device__ __forceinline__ void smallm_cp_async16(void* smem_dst, void const* gsrc, int src_bytes)
{
    uint32_t const d = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(gsrc), "r"(src_bytes) : "memory");
}

__device__ __forceinline__ void smallm_mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb)
{
#if defined(FSO_SMALLM_DEVICE)
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "h"(uint16_t(0)), "h"(uint16_t(0)),
          "r"(sb), "h"(uint16_t(0)), "h"(uint16_t(0)));
#else
    (void) d, (void) a0, (void) a1, (void) a2, (void) a3, (void) b0, (void) b1, (void) sa, (void) sb;
#endif
}

__device__ __forceinline__ float smallm_bf_lo(uint32_t u)
{
    return __uint_as_float(u << 16);
}

__device__ __forceinline__ float smallm_bf_hi(uint32_t u)
{
    return __uint_as_float(u & 0xFFFF0000u);
}

struct SmallmX4
{
    uint4 a, b, c, d;
};

// The 32 bf16 values of item `it` of a G2 chunk (activation row it / nbc, block b0 + it % nbc), or zeros past the
// items / past M.
__device__ __forceinline__ SmallmX4 smallm_load_item(SmallmParams const& p, int it, int items, int nbc, int b0)
{
    SmallmX4 v = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    int const m = it / nbc, bl = it % nbc;
    if (it < items && m < p.M)
    {
        uint4 const* src = reinterpret_cast<uint4 const*>(
            static_cast<uint8_t const*>(p.x) + (size_t(m) * p.K + size_t(b0 + bl) * 32) * 2);
        v.a = src[0];
        v.b = src[1];
        v.c = src[2];
        v.d = src[3];
    }
    return v;
}

// Stage the activation blocks [b0, b0 + nbc) into shared memory: codes [8 nc][rs] (rows >= M zero) and scale bytes
// [8 nc][cb] (rows >= M: 127, the scale 1, against zero codes). Every load of a thread is in flight at once (cp.async
// for the G1 codes, an unrolled batch for the scale words, a one-item-ahead pipeline for the G2 quantize).
template <int kForm, int kNc, int kThreads>
__device__ __forceinline__ void smallm_stage(
    SmallmParams const& p, uint8_t* xs_q, uint8_t* xs_b, int rs, int cb, int b0, int nbc)
{
    constexpr int kMp8 = 8 * kNc;
    int const M = p.M;
    int const K = p.K;
    int const mpc = (M + 3) & ~3;   // contract row padding of the activation scale words
    if constexpr (kForm == kSmallmFormQ)
    {
        int const units = kMp8 * nbc * 2;   // 16-byte units of codes
        for (int it = threadIdx.x; it < units; it += kThreads)
        {
            int const m = it / (nbc * 2), u = it % (nbc * 2);
            uint8_t const* src = static_cast<uint8_t const*>(p.x) + size_t(m < M ? m : 0) * K + size_t(b0) * 32 + u * 16;
            smallm_cp_async16(xs_q + m * rs + u * 16, src, m < M ? 16 : 0);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        int const wpr = (nbc + 3) / 4;   // scale words per row of the chunk (b0 is a multiple of 4)
        int const words = kMp8 * wpr;
        for (int it = threadIdx.x; it < words; it += 2 * kThreads)
        {
            int const it2 = it + kThreads;
            int const m1 = it / wpr, q1 = it % wpr;
            int const m2 = it2 / wpr, q2 = it2 % wpr;
            uint32_t w1 = 0x7F7F7F7Fu, w2 = 0x7F7F7F7Fu;
            if (m1 < M)
                w1 = static_cast<uint32_t>(p.sx[size_t(b0 / 4 + q1) * mpc + m1]);
            if (it2 < words && m2 < M)
                w2 = static_cast<uint32_t>(p.sx[size_t(b0 / 4 + q2) * mpc + m2]);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                if (q1 * 4 + i < nbc)
                    xs_b[m1 * cb + q1 * 4 + i] = static_cast<uint8_t>(w1 >> (8 * i));
                if (it2 < words && q2 * 4 + i < nbc)
                    xs_b[m2 * cb + q2 * 4 + i] = static_cast<uint8_t>(w2 >> (8 * i));
            }
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
    }
    else
    {
        int const items = kMp8 * nbc;
        SmallmX4 cur = smallm_load_item(p, threadIdx.x, items, nbc, b0);
        for (int it = threadIdx.x; it < items; it += kThreads)
        {
            SmallmX4 const nxt = smallm_load_item(p, it + kThreads, items, nbc, b0);
            int const m = it / nbc, bl = it % nbc;
            uint4* dst = reinterpret_cast<uint4*>(xs_q + m * rs + bl * 32);
            if (m >= M)
            {
                dst[0] = make_uint4(0, 0, 0, 0);
                dst[1] = make_uint4(0, 0, 0, 0);
                xs_b[m * cb + bl] = 127;
            }
            else
            {
                uint32_t const xw[16] = {cur.a.x, cur.a.y, cur.a.z, cur.a.w, cur.b.x, cur.b.y, cur.b.z, cur.b.w,
                                         cur.c.x, cur.c.y, cur.c.z, cur.c.w, cur.d.x, cur.d.y, cur.d.z, cur.d.w};
                float a = 0.f;
#pragma unroll
                for (int e = 0; e < 16; ++e)
                {
                    a = fmaxf(a, fabsf(smallm_bf_lo(xw[e])));
                    a = fmaxf(a, fabsf(smallm_bf_hi(xw[e])));
                }
                float qs;
                uint8_t const byte = fso_e8m0_from_amax(fmaxf(a, fso_mx_amax_floor()), &qs);
                uint32_t q[8];
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    q[e] = fso_mx_quantize4(smallm_bf_lo(xw[2 * e]), smallm_bf_hi(xw[2 * e]),
                        smallm_bf_lo(xw[2 * e + 1]), smallm_bf_hi(xw[2 * e + 1]), qs);
                dst[0] = make_uint4(q[0], q[1], q[2], q[3]);
                dst[1] = make_uint4(q[4], q[5], q[6], q[7]);
                xs_b[m * cb + bl] = byte;
            }
            cur = nxt;
        }
    }
}

// Shared-memory loads at a compile-time immediate offset from a 32-bit shared address (the x fragments of a step of
// the resident kernel; the offset is the step's block within its ring group).
template <int kImm>
__device__ __forceinline__ uint2 smallm_lds64(uint32_t addr)
{
    uint2 r;
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2+%3];" : "=r"(r.x), "=r"(r.y) : "r"(addr), "n"(kImm));
    return r;
}

template <int kImm>
__device__ __forceinline__ uint32_t smallm_lds8(uint32_t addr)
{
    uint32_t r;
    asm volatile("ld.shared.u8 %0, [%1+%2];" : "=r"(r) : "r"(addr), "n"(kImm));
    return r;
}

// The per-thread state of a tile group's step stream: the weight register ring, the accumulators, and the x
// fragments of the current K block.
template <int kNc, int kRt>
struct SmallmStream
{
    uint2 ra[kRing], rb[kRing];
    uint32_t rs[kRing];
    float acc[kRt][kNc][4];
    uint2 bf[kNc];
    uint32_t bs[kNc];
};

// Step u (0 .. kRing - 1) of a ring group covers tile t = u % kRt and the warp's block jj = u / kRt of the group.
template <int kRt, int u>
struct SmallmStep
{
    static constexpr int t = u % kRt;
    static constexpr int jj = u / kRt;
};

// Issue the weight loads of step u of the ring group whose bases are (wg, sg) into slot u: this lane's 8 bytes of
// rows g and g + 8 of tile t (tile t is w_tile = 16 K bytes from tile 0, row g + 8 is w_row8 = 8 K bytes from row g),
// block jj of the group (32 kWk bytes per block of this warp), and the scale word of the block's quad for row srow of
// tile t (s_j words per block of this warp, 16 words per tile).
template <int kWk, int kRt, int u, int kNc>
__device__ __forceinline__ void smallm_issue(SmallmStream<kNc, kRt>& st, uint8_t const* wg, int32_t const* sg,
    size_t w_tile, size_t w_row8, int s_j)
{
    constexpr int t = SmallmStep<kRt, u>::t, jj = SmallmStep<kRt, u>::jj;
    uint8_t const* a = wg + (t ? w_tile : size_t(0)) + jj * (kWk * 32);
    st.ra[u] = smallm_ldg64_stream(a);
    st.rb[u] = smallm_ldg64_stream(a + w_row8);
    st.rs[u] = smallm_ldg32(sg + t * 16 + jj * s_j);
}

// Consume slot u: at t == 0 the x fragments of block jj (xg / bg: 32-bit shared addresses of this lane's codes and of
// the scale bytes of rows 8 jc + g, at this warp's first block of the group, wk + kWk kJGrp grp; block jj of the
// group is kWk jj blocks further), then kNc MMAs into acc[t]. The accumulation order per warp is the K order of its
// blocks, as before: the bytes do not change.
template <int kWk, int kRt, int u, int kNc>
__device__ __forceinline__ void smallm_consume(SmallmStream<kNc, kRt>& st, uint2 A0, uint2 A1, uint32_t S,
    uint32_t const (&xg)[kNc], uint32_t const (&bg)[kNc])
{
    constexpr int t = SmallmStep<kRt, u>::t, jj = SmallmStep<kRt, u>::jj;
    if constexpr (t == 0)
    {
#pragma unroll
        for (int jc = 0; jc < kNc; ++jc)
        {
            st.bf[jc] = smallm_lds64<jj * kWk * 32>(xg[jc]);
            st.bs[jc] = smallm_lds8<jj * kWk>(bg[jc]);
        }
    }
#pragma unroll
    for (int jc = 0; jc < kNc; ++jc)
        smallm_mma(st.acc[t][jc], A0.x, A1.x, A0.y, A1.y, st.bf[jc].x, st.bf[jc].y, S, st.bs[jc]);
}

// One ring group of the steady state: every step valid, every slot re-issued for the ring group kRing steps later of
// the same tile group (bases wn / sn), no bounds checks and no predicates. (The recursion over u unrolls the group.)
template <int kWk, int kRt, int kNc, int u = 0>
__device__ __forceinline__ void smallm_group_full(SmallmStream<kNc, kRt>& st, uint32_t const (&xg)[kNc],
    uint32_t const (&bg)[kNc], uint32_t shift, uint8_t const* wn, int32_t const* sn, size_t w_tile, size_t w_row8,
    int s_j)
{
    if constexpr (u < kRing)
    {
        uint2 const A0 = st.ra[u], A1 = st.rb[u];
        uint32_t const S = st.rs[u] >> shift;
        smallm_issue<kWk, kRt, u>(st, wn, sn, w_tile, w_row8, s_j);
        smallm_consume<kWk, kRt, u>(st, A0, A1, S, xg, bg);
        smallm_group_full<kWk, kRt, kNc, u + 1>(st, xg, bg, shift, wn, sn, w_tile, w_row8, s_j);
    }
}

// A ring group at an edge of the tile group (the last full group, the partial tail, or a group whose successor holds
// blocks past nb), with warp-uniform bounds (every operand below is the same for the whole warp):
//   step u is consumed when u < consume_lt and its block (kWk jj from the group's first block) is below nb_rem;
//   its slot is re-issued from this tile group's next ring group (wn / sn) when u < this_lt and kWk jj < nb_rem_next,
//   else from the next tile group's first ring group (wo / so) when u >= other_from, u < other_lt and
//   kWk jj < nb_rem_other, else left alone (a padding step: nothing was issued there before either).
template <int kWk, int kRt, int kNc, int u = 0>
__device__ __forceinline__ void smallm_group_edge(SmallmStream<kNc, kRt>& st, uint32_t const (&xg)[kNc],
    uint32_t const (&bg)[kNc], uint32_t shift, int consume_lt, int nb_rem, uint8_t const* wn, int32_t const* sn,
    int this_lt, int nb_rem_next, uint8_t const* wo, int32_t const* so, int other_from, int other_lt,
    int nb_rem_other, size_t w_tile, size_t w_row8, int s_j)
{
    if constexpr (u < kRing)
    {
        constexpr int jj = SmallmStep<kRt, u>::jj;
        uint2 const A0 = st.ra[u], A1 = st.rb[u];
        uint32_t const S = st.rs[u] >> shift;
        if (u < this_lt && kWk * jj < nb_rem_next)
            smallm_issue<kWk, kRt, u>(st, wn, sn, w_tile, w_row8, s_j);
        else if (u >= other_from && u < other_lt && kWk * jj < nb_rem_other)
            smallm_issue<kWk, kRt, u>(st, wo, so, w_tile, w_row8, s_j);
        if (u < consume_lt && kWk * jj < nb_rem)
            smallm_consume<kWk, kRt, u>(st, A0, A1, S, xg, bg);
        smallm_group_edge<kWk, kRt, kNc, u + 1>(st, xg, bg, shift, consume_lt, nb_rem, wn, sn, this_lt, nb_rem_next,
            wo, so, other_from, other_lt, nb_rem_other, w_tile, w_row8, s_j);
    }
}

// The kernel. kChunked selects the plan family: false = the activation is staged once per CTA (cj == 0, the only
// plans the rules route; the step loop above runs without runtime bounds in its steady state); true = K chunks of
// p.cj blocks per warp, re-staged for every tile group (FSO_SMALLM_ROUTE=1 at large M x K; wk == 8 only, as config()
// produces them; a generic checked step loop with a countdown over the chunk boundary).
template <int kForm, int kNc, int kRt, int kWk, bool kEpi, bool kChunked>
__global__ void __launch_bounds__(32 * kWk, 1) sm120_smallm_kernel(SmallmParams p)
{
#if defined(FSO_SMALLM_DEVICE)
    constexpr int kThreads = 32 * kWk;
    constexpr int kMp8 = 8 * kNc;
    constexpr int kRows = 16 * kRt;
    constexpr int kJGrp = kRing / kRt;        // K blocks per warp consumed by one ring group
    constexpr int kBGrp = kWk * kJGrp;        // weight blocks a ring group advances
    static_assert(kRing % kRt == 0, "kRt must divide the ring depth");
    static_assert(kWk % 4 == 0, "a scale word holds the four blocks of a quad");

    extern __shared__ __align__(16) uint8_t smem[];

    int const tid = threadIdx.x;
    int const wk = tid / 32;
    int const lane = tid & 31;
    int const g = lane >> 2;
    int const tig = lane & 3;
    int const M = p.M, N = p.N, K = p.K;
    int const nb = K / 32;
    int const nbw = (nb + kWk - 1) / kWk;            // K blocks per warp (the last ones may be past nb)
    int const L = nbw * kRt;                          // steps per tile group
    int const LP = (L + kRing - 1) / kRing * kRing;   // padded to the ring depth
    int const G = L / kRing;                          // full ring groups
    int const tail = L - G * kRing;                   // steps of the partial last group (0 .. kRing - 1)
    int const ntg = N / kRows;
    int const srow = g + 8 * (lane & 1);              // the A row whose scale this lane supplies
    int const cj = kChunked ? p.cj : 0;

    fso::sm120_smallm::Smem const sl = fso::sm120_smallm::smem_layout(kNc, kRt, kWk, K, cj);
    uint8_t* xs_q = smem;
    uint8_t* xs_b = smem + sl.xq_bytes;
    float* red = reinterpret_cast<float*>(kChunked ? smem : smem + sl.xq_bytes + sl.xb_bytes);
    int const chunk_blocks = sl.cb;

    SmallmStream<kNc, kRt> st;
    uint32_t const shift = 8u * uint32_t(wk & 3);    // (b & 3) == (wk & 3) for every block b of this warp

    // Per-thread bases. Weight: this lane's 8 bytes of row g of tile 0 of tile group 0 at block wk; tile t is
    // w_tile = 16 K bytes further, row g + 8 is w_row8 = 8 K bytes further, tile group tg is tg kRows K bytes
    // further, block j of this warp is 32 kWk j bytes further. Scale words: (b >> 2) N + n with
    // b >> 2 = (wk >> 2) + (kWk / 4) j and n = tg kRows + 16 t + srow.
    uint8_t const* const w0 = p.w + size_t(g) * K + size_t(wk) * 32 + tig * 8;
    size_t const w_tile = size_t(16) * K;
    size_t const w_row8 = size_t(8) * K;
    size_t const w_tg = size_t(kRows) * K;
    int32_t const* const s0 = p.ws + size_t(wk >> 2) * N + srow;
    int const s_j = (kWk / 4) * N;                    // scale words between consecutive blocks of this warp
    // The staged activation (resident kernel): this lane's 8 bytes of row m = 8 jc + g at this warp's first block wk,
    // and row m's scale byte at block wk (32-bit shared addresses; rows 8 jc + g are sl.rs / chunk_blocks bytes apart,
    // so the per-row bases are recomputed from these two per tile group rather than kept in registers). The block
    // term wk matters: the steps' immediates only add the warp's block stride kWk jj.
    uint32_t const xq_g = static_cast<uint32_t>(__cvta_generic_to_shared(xs_q + g * sl.rs + wk * 32 + tig * 8));
    uint32_t const xb_g = static_cast<uint32_t>(__cvta_generic_to_shared(xs_b + g * chunk_blocks + wk));
    uint32_t const xq_row8 = 8u * uint32_t(sl.rs);
    uint32_t const xb_row8 = 8u * uint32_t(chunk_blocks);

    // Checked issue of step s of tile group tg_ into slot u (the prologue, and every step of the chunked kernel).
    auto issue_at = [&](int tg_, int s, int u)
    {
        int const j = s / kRt, t = s % kRt;
        int const b = wk + kWk * j;
        if (s < L && b < nb)
        {
            uint8_t const* a = w0 + size_t(tg_) * w_tg + size_t(t) * w_tile + size_t(j) * (kWk * 32);
            st.ra[u] = smallm_ldg64_stream(a);
            st.rb[u] = smallm_ldg64_stream(a + w_row8);
            st.rs[u] = smallm_ldg32(s0 + size_t(tg_) * kRows + t * 16 + j * s_j);
        }
    };

    int tg = blockIdx.x;
    if (tg < ntg)
    {
#pragma unroll
        for (int u = 0; u < kRing; ++u)
            issue_at(tg, u, u);
    }

    // PDL: everything above reads the weight only; x, its scale, bias and residual are read below, the output is
    // written below.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    if constexpr (!kChunked)
    {
        smallm_stage<kForm, kNc, kThreads>(p, xs_q, xs_b, sl.rs, chunk_blocks, 0, nb);
        __syncthreads();
    }

    for (; tg < ntg; tg += gridDim.x)
    {
        int const tg_next = tg + gridDim.x;
        bool const has_next = tg_next < ntg;
#pragma unroll
        for (int t = 0; t < kRt; ++t)
#pragma unroll
            for (int jc = 0; jc < kNc; ++jc)
                st.acc[t][jc][0] = st.acc[t][jc][1] = st.acc[t][jc][2] = st.acc[t][jc][3] = 0.f;

        if constexpr (!kChunked)
        {
            uint8_t const* wg = w0 + size_t(tg) * w_tg;              // this tile group, current ring group
            int32_t const* sg = s0 + size_t(tg) * kRows;
            uint8_t const* const wo = w0 + size_t(tg_next) * w_tg;   // the next tile group, ring group 0
            int32_t const* const so = s0 + size_t(tg_next) * kRows;
            uint32_t xg[kNc], bg[kNc];
#pragma unroll
            for (int jc = 0; jc < kNc; ++jc)
            {
                xg[jc] = xq_g + jc * xq_row8;
                bg[jc] = xb_g + jc * xb_row8;
            }
            int b0 = wk;                                             // this warp's first block of the ring group
            auto advance = [&]()
            {
                wg += kBGrp * 32;
                sg += kJGrp * s_j;
                b0 += kBGrp;
#pragma unroll
                for (int jc = 0; jc < kNc; ++jc)
                {
                    xg[jc] += kBGrp * 32;
                    bg[jc] += kBGrp;
                }
            };
            // A full group may issue its successor unchecked only if every block of the successor is below nb; that
            // fails for the last full group when it holds the last j and nb % kWk != 0 (K = 128 mod 256 at wk = 8).
            int steady = G - 1 - ((tail == 0 && (nb % kWk) != 0) ? 1 : 0);
            if (steady < 0)
                steady = 0;
            for (int grp = 0; grp < steady; ++grp)
            {
                smallm_group_full<kWk, kRt, kNc>(st, xg, bg, shift, wg + kBGrp * 32, sg + kJGrp * s_j, w_tile,
                    w_row8, s_j);
                advance();
            }
            // The remaining groups (the last full group(s) and the partial tail) with warp-uniform bounds: a step is
            // consumed while it is below L and its block below nb; a slot is re-issued from this tile group while the
            // step kRing later is below L (and its block below nb), else from the next tile group's first group once
            // the step kRing later reaches LP (the padding); one call site keeps the code small.
            int const last = tail > 0 ? G : G - 1;
#pragma unroll 1
            for (int grp = steady; grp <= last; ++grp)
            {
                int const s0g = grp * kRing;
                int const consume_lt = (L - s0g) < kRing ? (L - s0g) : kRing;
                int const this_lt = (L - s0g - kRing) > 0 ? ((L - s0g - kRing) < kRing ? (L - s0g - kRing) : kRing) : 0;
                int const other_from = (LP - s0g - kRing) > 0 ? (LP - s0g - kRing) : 0;
                int const other_lt = has_next ? (L < kRing ? L : kRing) : 0;
                smallm_group_edge<kWk, kRt, kNc>(st, xg, bg, shift, consume_lt, nb - b0, wg + kBGrp * 32,
                    sg + kJGrp * s_j, this_lt, nb - (b0 + kBGrp), wo, so, other_from, other_lt, nb - wk, w_tile,
                    w_row8, s_j);
                advance();
            }
        }
        else
        {
            // K chunks: the chunk holding block b is staged at the first step that needs it (tile 0 of the first j of
            // the chunk, every tile of a j reads the same block), counted down over j instead of a modulo. Uniform
            // across the CTA.
            int chunk_b0 = 0;
            int j_to_chunk = 0;
            for (int s0g = 0; s0g < LP; s0g += kRing)
            {
#pragma unroll
                for (int u = 0; u < kRing; ++u)
                {
                    int const s = s0g + u;
                    int const t = u % kRt;   // s0g is a multiple of the ring depth, which kRt divides
                    int const j = s / kRt;
                    int const b = wk + kWk * j;
                    if (s < L && t == 0)
                    {
                        if (j_to_chunk == 0)
                        {
                            chunk_b0 = kWk * j;
                            int const nbc = (nb - chunk_b0) < chunk_blocks ? (nb - chunk_b0) : chunk_blocks;
                            __syncthreads();   // every warp is done with the previous chunk (or the previous reduction)
                            smallm_stage<kForm, kNc, kThreads>(p, xs_q, xs_b, sl.rs, chunk_blocks, chunk_b0, nbc);
                            __syncthreads();
                            j_to_chunk = cj;
                        }
                        --j_to_chunk;
                    }
                    uint2 const A0 = st.ra[u], A1 = st.rb[u];
                    uint32_t const S = st.rs[u] >> shift;
                    if (s + kRing < LP)
                        issue_at(tg, s + kRing, u);
                    else if (has_next)
                        issue_at(tg_next, s + kRing - LP, u);
                    if (s < L && b < nb)
                    {
                        if (t == 0)
                        {
                            int const bl = b - chunk_b0;
#pragma unroll
                            for (int jc = 0; jc < kNc; ++jc)
                            {
                                int const m = 8 * jc + g;
                                st.bf[jc] = *reinterpret_cast<uint2 const*>(xs_q + m * sl.rs + bl * 32 + tig * 8);
                                st.bs[jc] = xs_b[m * chunk_blocks + bl];
                            }
                        }
#pragma unroll
                        for (int jc = 0; jc < kNc; ++jc)
                            smallm_mma(st.acc[t][jc], A0.x, A1.x, A0.y, A1.y, st.bf[jc].x, st.bf[jc].y, S, st.bs[jc]);
                    }
                }
            }
        }

        // The kWk warps' partial tiles, summed in a fixed order. Fragment d0 = (row g, column 2 tig), d1 = (row g,
        // column 2 tig + 1), d2 / d3 = row g + 8; columns are activation rows.
        if constexpr (kChunked)
            __syncthreads();   // red aliases the x chunk buffer
#pragma unroll
        for (int t = 0; t < kRt; ++t)
#pragma unroll
            for (int jc = 0; jc < kNc; ++jc)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    int const col = 8 * jc + 2 * tig + (e & 1);
                    int const row = g + 8 * (e >> 1);
                    red[((wk * kRt + t) * kMp8 + col) * kRedStride + row] = st.acc[t][jc][e];
                }
        __syncthreads();
        // Consecutive threads write consecutive output rows n of one activation row (coalesced stores).
        for (int o = tid; o < kRows * M; o += kThreads)
        {
            int const col = o / kRows, r = o % kRows;
            int const t = r / 16, row = r % 16;
            float v = red[((0 * kRt + t) * kMp8 + col) * kRedStride + row];
#pragma unroll
            for (int k2 = 1; k2 < kWk; ++k2)
                v += red[((k2 * kRt + t) * kMp8 + col) * kRedStride + row];
            int const n = tg * kRows + r;
            if constexpr (kEpi)
            {
                fso::DenseEpilogue const& e = p.epi;
                __nv_bfloat16 const zero = __float2bfloat16_rn(0.f);
                __nv_bfloat16 const bn = e.bias != nullptr ? fso::dense_load_bf16(e.bias + n) : zero;
                __nv_bfloat16 const rmn
                    = e.residual != nullptr ? fso::dense_load_bf16(e.residual + size_t(col) * e.ld_residual + n) : zero;
                v = fso::dense_epilogue_value_rt(v, e.bias != nullptr, bn, e.act, e.residual != nullptr, rmn);
            }
            p.y[size_t(col) * N + n] = __float2bfloat16_rn(v);
        }
        __syncthreads();   // red is rewritten by the next tile group (or the next chunk overwrites it)
    }
#else
    (void) p;
#endif
}

} // namespace tensorrt_llm::kernels::blockscale_gemm::sm120_smallm

namespace tensorrt_llm::kernels::blockscale_gemm::sm120_smallm
{

// One launch of the instantiation (kForm, nc, rt, wk, kEpi, chunked): grid CTAs of 32 * wk threads with `smem` bytes
// of dynamic shared memory, PDL as `pdl` says (the process-wide switch, fso_pdl_enabled). `chunked` = p.cj > 0 (wk == 8
// only). Defined in ops/mxfp8_smallm_kernels.inc and instantiated by one translation unit per (kForm, kEpi).
template <int kForm, bool kEpi>
cudaError_t smallm_launch(
    SmallmParams const& p, int nc, int rt, int wk, bool chunked, int grid, int smem, bool pdl, cudaStream_t s);

// The opt-in shared-memory attribute of every kernel of the (kForm, kEpi) set, set once per process (the first call
// of the route, which the capture contract makes an eager one).
template <int kForm, bool kEpi>
cudaError_t smallm_init();

} // namespace tensorrt_llm::kernels::blockscale_gemm::sm120_smallm
