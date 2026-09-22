/*
 * Wrappers around the standalone library's host-side quantize launchers, with
 * plain function signatures (no ATen). Lives in its own TU so we can include
 * the heavy CUTLASS / cute headers without the `cute::Layout` symbol leaking
 * into the ATen-aware ops.cu (which would conflict with `at::Layout`).
 */

#include "blockscale_gemm/common/scale_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace blockscale_gemm
{
namespace detail
{

// Forward decl (defined further below — fused uint64 LDG.64 BSFP8 quantize).
void fp8bs_quantize_1x128_fp32_fast(__nv_fp8_e4m3* x_q, float* fp32_scales,
    __nv_bfloat16 const* x, int M, int K, cudaStream_t stream, bool use_ue8m0);

void fp8bs_quantize_1x128(__nv_fp8_e4m3* x_q, float* scales, __nv_bfloat16 const* x, int M, int K, cudaStream_t stream,
    bool use_ue8m0)
{
    // Fast path: uint64 LDG.64 + e8m0 bit-manip + larger per-warp work. Requires
    // K % 512 == 0 (4 K-blocks/warp × 128 elem/K-block). Qwen3-4B shapes
    // (K ∈ {2560, 4096, 9728, 19456}) all satisfy this.
    // Fallback: the legacy persistent-grid `fp8_1x128_cs` kernel for K only a
    // multiple of 128.
    if (K % 512 == 0)
        fp8bs_quantize_1x128_fp32_fast(x_q, scales, x, M, K, stream, use_ue8m0);
    else
        tensorrt_llm::kernels::blockscale_gemm::fp8_1x128_cs(x_q, scales, x, K, M, stream, use_ue8m0);
}

void fp8bs_quantize_128x128(
    __nv_fp8_e4m3* w_q, float* scales, __nv_bfloat16 const* w, int N, int K, cudaStream_t stream, bool use_ue8m0)
{
    // The runner-internal weight quant path. NOT fp8_128x128_cs, which is a
    // cast-only placeholder that fills scales with 1.0.
    tensorrt_llm::kernels::blockscale_gemm::fp8_128x128_quant(w_q, scales, w, K, N, stream, use_ue8m0);
}

// MXFP8 1×32 quantize — restored 2026-05-21 for the fish-scales-ops MXFP8
// GEMM path. The repack helpers below are VS-agnostic and shared with the
// 1×128 FP8 path.

namespace
{

constexpr int kMxFp8VecSize = 32;  // OCP MXFP8 hardware-native granularity

template <bool USE_UE8M0>
__global__ void fp8bs_quantize_1x32_kernel(
    __nv_fp8_e4m3* __restrict__ out, float* __restrict__ scales,
    __nv_bfloat16 const* __restrict__ input, int M, int K)
{
    int const warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int const lane_id = threadIdx.x & 31;
    int const k_blocks = K / kMxFp8VecSize;
    int const total_warps = M * k_blocks;
    if (warp_id >= total_warps) return;

    int const m  = warp_id / k_blocks;
    int const kb = warp_id % k_blocks;
    int const k  = kb * kMxFp8VecSize + lane_id;

    __nv_bfloat16 const x = input[m * K + k];
    float const ax = fabsf(__bfloat162float(x));

    float amax = ax;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float const other = __shfl_xor_sync(0xFFFFFFFFu, amax, off);
        amax = fmaxf(amax, other);
    }
    amax = fmaxf(amax, 1e-10f);

    float quant_scale = 448.f / amax;
    float dequant_scale;
    if constexpr (USE_UE8M0) {
        float const dequant_raw = 1.f / quant_scale;
        __nv_fp8_e8m0 ue8m0_scale;
        ue8m0_scale.__x = __nv_cvt_float_to_e8m0(dequant_raw, __NV_SATFINITE, cudaRoundPosInf);
        dequant_scale = static_cast<float>(ue8m0_scale);
        quant_scale = dequant_scale != 0.f ? 1.f / dequant_scale : 1.f;
    } else {
        dequant_scale = 1.f / quant_scale;
    }

    int const m_pad = ((M + 3) / 4) * 4;
    if (lane_id == 0) {
        scales[kb * m_pad + m] = dequant_scale;
    }

    float const v = __bfloat162float(x) * quant_scale;
    float const v_sat = fmaxf(-448.f, fminf(448.f, v));
    out[m * K + k] = __nv_fp8_e4m3(v_sat);
}

} // anonymous namespace

void fp8bs_quantize_1x32(__nv_fp8_e4m3* x_q, float* scales, __nv_bfloat16 const* x, int M, int K, cudaStream_t stream,
    bool use_ue8m0)
{
    int const k_blocks = K / kMxFp8VecSize;
    int const total_warps = M * k_blocks;
    constexpr int kThreadsPerBlock = 256;
    constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;
    int const grid = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;

    int const m_pad = ((M + 3) / 4) * 4;
    if (m_pad != M) {
        cudaMemsetAsync(scales, 0, sizeof(float) * m_pad * k_blocks, stream);
    }

    if (use_ue8m0)
        fp8bs_quantize_1x32_kernel<true><<<grid, kThreadsPerBlock, 0, stream>>>(x_q, scales, x, M, K);
    else
        fp8bs_quantize_1x32_kernel<false><<<grid, kThreadsPerBlock, 0, stream>>>(x_q, scales, x, M, K);
}

// ----- Fused quantize + scale-repack kernel (2026-05-28) -------------------
//
// Replaces the legacy `quantize_1x32` + `repack_mxfp8_scales` two-kernel
// pipeline with a single fused kernel:
//   * 1 warp = K_BLOCKS_PER_WARP K-blocks of one M row (e.g. 4 K-blocks =
//     128 K-elements = exactly one packed int32 output for K_BLOCKS_PER_WARP=4)
//   * The warp loops over K-blocks; for each block it does a 32-lane shfl
//     amax reduction, derives the UE8M0 byte, casts the 32 elements to FP8
//     and stores. Lane 0 accumulates the K_BLOCKS_PER_WARP bytes into a
//     local int32 and writes it directly to the packed output buffer.
//   * Eliminates: (a) the FP32 scale round-trip through global memory,
//     (b) one full kernel launch, (c) the cudaMemsetAsync of FP32 scales.
//
// Output `packed` layout: K-major int32 [pad(M,4), K/(32 * K_BLOCKS_PER_WARP)]
// with stride (1 along M, M_pad along K). For K_BLOCKS_PER_WARP=4 this is
// the exact layout CUTLASS Sm120BlockScaledKernel expects (4 UE8M0 bytes
// per int32, byte i packed at bit `i*8`).
namespace
{

// Helper: convert per-block amax to UE8M0 byte + recovered quant scale.
//
// USE_UE8M0=true does RCEIL (ceil(log2(amax/448)) + 127) via direct
// IEEE-754 bit manipulation — equivalent to `__nv_cvt_float_to_e8m0(...,
// SATFINITE, RoundPosInf)` but shorter SASS (no library call).
//   For amax > 0 finite:
//     descale = amax * (1/448); bits = float_as_uint(descale)
//     exp_field = (bits >> 23) & 0xFF
//     byte = exp_field + ((bits & 0x7FFFFF) != 0 ? 1 : 0)
//     clamp to [0, 254]; 255 is NaN.
//   Dequant scale recovered from byte via `byte << 23` reinterpret as fp32.
// USE_UE8M0=false: take amax/448 as-is, extract IEEE biased exp from
// the FP32 dequant (matches repack_mxfp8_scales).
template <bool USE_UE8M0>
__device__ __forceinline__ void e8m0_from_amax(
    float amax, float& quant_scale_out, uint8_t& byte_out)
{
    // Match the legacy IEEE-754 pipeline byte-exactly: derive descale via
    // 448/amax → 1/quant_scale (NOT amax*(1/448) — the two paths differ by
    // up to 1 ULP and produce different byte values at pow-2 boundaries).
    float const quant_scale_raw = 448.f / amax;
    float const dequant_raw = 1.f / quant_scale_raw;
    if constexpr (USE_UE8M0) {
        uint32_t const bits = __float_as_uint(dequant_raw);
        uint32_t const exp_field = (bits >> 23) & 0xFFu;
        uint32_t const has_mantissa = (bits & 0x7FFFFFu) != 0u ? 1u : 0u;
        uint32_t byte = exp_field + has_mantissa;
        byte = byte > 254u ? 254u : byte;
        byte_out = static_cast<uint8_t>(byte);
        // dequant_scale = 2^(byte - 127); fp32 with biased exp = byte, mantissa = 0.
        uint32_t const ds_bits = byte << 23;
        float const dequant_scale = __uint_as_float(ds_bits);
        quant_scale_out = byte != 0u ? (1.f / dequant_scale) : 1.f;
    } else {
        uint32_t const bits = __float_as_uint(dequant_raw);
        byte_out = static_cast<uint8_t>((bits >> 23) & 0xFFu);
        quant_scale_out = quant_scale_raw;
    }
}

// silu via tanh.approx.f32 (sm_75+): silu(x) = 0.5·x·(1 + tanh(x/2)).
// One MUFU op instead of __expf's two (MUFU.EX2 + MUFU.RCP); the fused
// silu+quantize kernel is MUFU/convert instruction-bound at large M (ncu
// 2026-07-06: 37.8 µs @ M=2048 vs ~16 µs memory roofline, SM 76% / MEM 29%).
// |tanh.approx err| ≲ 2^-10.8 — far below the bf16-input / fp8-output
// precision floor.
__device__ __forceinline__ float silu_tanh_approx(float x)
{
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x));
    return 0.5f * x * (1.f + t);
}

// bf16x2 SIMD variant: one MUFU per TWO elements (tanh.approx.bf16x2,
// sm_90+). Intermediate h is bf16 (~2^-8 rel err) — still 16× finer than
// the fp8-e4m3 output (2^-4); 0.5·x is exact in bf16 (exponent decrement).
__device__ __forceinline__ __nv_bfloat162 tanh2_approx(__nv_bfloat162 x)
{
    uint32_t r;
    asm("tanh.approx.bf16x2 %0, %1;" : "=r"(r) : "r"(*reinterpret_cast<uint32_t const*>(&x)));
    return *reinterpret_cast<__nv_bfloat162 const*>(&r);
}

// h01 = silu(g01) * u01 for a bf16x2 pair: xh = g/2; h = (xh + xh·tanh(xh))·u.
__device__ __forceinline__ __nv_bfloat162 silu2_mul(__nv_bfloat162 g, __nv_bfloat162 u)
{
    __nv_bfloat162 const half2v = __float2bfloat162_rn(0.5f);
    __nv_bfloat162 const xh = __hmul2(g, half2v);
    __nv_bfloat162 const t = tanh2_approx(xh);
    return __hmul2(__hfma2(xh, t, xh), u);
}


// Two paired cvt.rn.satfinite.e4m3x2.f32 (sm_89+) instead of four scalar
// converts + byte inserts. satfinite saturates to ±448, replacing the
// explicit fmaxf/fminf clamps. (NaN input becomes fp8 NaN instead of the
// clamps' ±448 — inputs are finite by the amax ≥ 1e-10 contract.)
__device__ __forceinline__ uint32_t fp8x4_from_floats(float v0, float v1, float v2, float v3)
{
    __nv_fp8x2_storage_t const lo
        = __nv_cvt_float2_to_fp8x2(make_float2(v0, v1), __NV_SATFINITE, __NV_E4M3);
    __nv_fp8x2_storage_t const hi
        = __nv_cvt_float2_to_fp8x2(make_float2(v2, v3), __NV_SATFINITE, __NV_E4M3);
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

// Scale-factor output layout selector (2026-07-05, sm_100/sm_103 support).
//
// SM120_KMAJOR: the sm_120 int32-packed K-major layout — word (m, kp) at
//   int32 index `kp * M_pad + m`, M_pad = align(M, 4). This is what
//   CUTLASS Sm120BlockScaledKernel's TMA descriptor reads.
//
// SM1XX_ATOM: the CUTLASS Sm1xxBlockScaledConfig<32> K-major atom layout
//   used by the tcgen05.mma.blockscaled collectives on sm_100/sm_103:
//     SfKMajorAtom ((32,4),(32,4)) : ((16,4),(0,1))
//   Per 128-row × 128-K block of data there is one 512-byte SF block; the
//   4 consecutive K-block bytes of a row are contiguous, so each (m, kp)
//   still maps to exactly ONE int32 word:
//     block   = (m / 128) * (K/128) + kp          (blocks tile K-minor)
//     in-blk  = (m % 32) * 4 + (m % 128) / 32     (int32 units)
//     index   = block * 128 + in-blk
//   M_pad = align(M, 128); padded rows must write zero words (the kernels'
//   m >= M branch handles that).
enum class MxSfLayout { SM120_KMAJOR, SM1XX_ATOM };

template <MxSfLayout SF_LAYOUT>
__device__ __forceinline__ int sf_word_index(int m, int kp, int M_pad, int num_kp)
{
    if constexpr (SF_LAYOUT == MxSfLayout::SM120_KMAJOR) {
        return kp * M_pad + m;
    } else {
        int const r = m % 128;
        return ((m / 128) * num_kp + kp) * 128 + (r % 32) * 4 + r / 32;
    }
}

// Vectorized fused quantize + scale-pack (E34, 2026-05-28).
//
// Each inner iteration handles 4 K-blocks (128 K-elements) at once via a
// single uint64 (LDG.64 = 4 BF16) per lane:
//   * 32 lanes × 4 BF16 = 128 elements ⇒ exactly 4 K-blocks per iter.
//   * Lanes 0..7 own K-block 0; 8..15 own K-block 1; 16..23 own K-block 2;
//     24..31 own K-block 3. Per-K-block amax is reduced inside each 8-lane
//     group via shfl_xor offsets 4, 2, 1.
//   * Each lane writes its 4 FP8 bytes as one uint32 store (STG.32). Adjacent
//     lanes coalesce into a 128-byte STG per warp iter.
//   * Lane 0/8/16/24 each hold their group's UE8M0 byte; lane 0 gathers them
//     via 3× __shfl and packs into a single int32. With K_BLOCKS_PER_WARP=8
//     the warp does 2 iters and writes 2 packed int32 outputs.
//
// Cost vs Fix-B (bfloat162):
//   * LDG: 8 × LDG.32 → 2 × LDG.64 per warp at K_BLOCKS_PER_WARP=8 (4× fewer
//     load instructions, same bytes).
//   * STG: 8 × STG.16 → 2 × STG.32 per warp (4× fewer store instructions).
//   * Shfl: 8 × (2 full-warp shfl_xor) → 2 × (3 sub-warp shfl_xor + 3 shfl).
template <bool USE_UE8M0, int K_BLOCKS_PER_WARP, MxSfLayout SF_LAYOUT = MxSfLayout::SM120_KMAJOR>
__global__ void fp8bs_quantize_1x32_packed_kernel(
    __nv_fp8_e4m3* __restrict__ out_fp8, int32_t* __restrict__ out_packed,
    __nv_bfloat16 const* __restrict__ input, int M, int K, int M_pad)
{
    static_assert(K_BLOCKS_PER_WARP == 4 || K_BLOCKS_PER_WARP == 8,
        "K_BLOCKS_PER_WARP must be 4 (1 int32/warp) or 8 (2 int32/warp).");
    static_assert(K_BLOCKS_PER_WARP % 4 == 0,
        "uint64 vectorised inner iter handles 4 K-blocks at a time");

    constexpr int kVec = kMxFp8VecSize;                 // 32 K-elements per scale block
    constexpr int kIterKBlocks = 4;                      // 4 K-blocks / inner iter via uint64 load
    constexpr int kIterElems = kIterKBlocks * kVec;      // 128 K-elements / iter
    constexpr int kNumIters = K_BLOCKS_PER_WARP / kIterKBlocks;

    int const warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int const lane_id = threadIdx.x & 31;
    int const k_groups = K / (kVec * K_BLOCKS_PER_WARP);
    int const total_warps = M_pad * k_groups;
    if (warp_id >= total_warps) return;

    int const m  = warp_id / k_groups;
    int const kg = warp_id % k_groups;
    int const kp_base = kg * (K_BLOCKS_PER_WARP / 4);
    int const num_kp = K / 128;  // int32 SF words per row

    if (m >= M) {
        if (lane_id == 0) {
#pragma unroll
            for (int p = 0; p < K_BLOCKS_PER_WARP / 4; ++p) {
                out_packed[sf_word_index<SF_LAYOUT>(m, kp_base + p, M_pad, num_kp)] = 0;
            }
        }
        return;
    }

    int packed_words[K_BLOCKS_PER_WARP / 4] = {};

#pragma unroll
    for (int it = 0; it < kNumIters; ++it) {
        int const kb_base = kg * K_BLOCKS_PER_WARP + it * kIterKBlocks;
        int const k_base  = kb_base * kVec;

        // LDG.64: 4 BF16 per lane. Lane L covers k_base + L*4 .. L*4+3.
        uint64_t const xword = *reinterpret_cast<uint64_t const*>(
            &input[m * K + k_base + lane_id * 4]);
        __nv_bfloat16 const* xv = reinterpret_cast<__nv_bfloat16 const*>(&xword);
        float const x0 = __bfloat162float(xv[0]);
        float const x1 = __bfloat162float(xv[1]);
        float const x2 = __bfloat162float(xv[2]);
        float const x3 = __bfloat162float(xv[3]);

        float my_ax = fmaxf(fmaxf(fabsf(x0), fabsf(x1)),
                            fmaxf(fabsf(x2), fabsf(x3)));

        // 8-lane group reduction via full-warp shfl_xor (offsets 4, 2, 1 stay
        // within each 8-lane group, so the 4 K-blocks reduce in parallel).
        // (redux.sync.max.u32 on fabs bits was tried 2026-07-06 and is
        // SLOWER on sm_103: silu kernel 27.3 → 31.0 µs — subset-mask redux
        // costs more than the 3-shfl chain. Do not retry.)
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            my_ax = fmaxf(my_ax, __shfl_xor_sync(0xFFFFFFFFu, my_ax, off));
        }
        my_ax = fmaxf(my_ax, 1e-10f);

        float qs;
        uint8_t byte_v;
        e8m0_from_amax<USE_UE8M0>(my_ax, qs, byte_v);

        // STG.32: 4 FP8 bytes per lane (paired satfinite converts).
        uint32_t const fp_word = fp8x4_from_floats(x0 * qs, x1 * qs, x2 * qs, x3 * qs);
        *reinterpret_cast<uint32_t*>(&out_fp8[m * K + k_base + lane_id * 4]) = fp_word;

        // Gather the 4 K-blocks' UE8M0 bytes to lane 0 and pack into int32.
        // After the 8-lane reduction, lanes 0/8/16/24 each hold the byte for
        // K-block (it*4 + 0/1/2/3).
        uint32_t const b0 = byte_v;
        uint32_t const b1 = __shfl_sync(0xFFFFFFFFu, byte_v, 8);
        uint32_t const b2 = __shfl_sync(0xFFFFFFFFu, byte_v, 16);
        uint32_t const b3 = __shfl_sync(0xFFFFFFFFu, byte_v, 24);
        if (lane_id == 0) {
            packed_words[it] = static_cast<int>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
        }
    }

    if (lane_id == 0) {
#pragma unroll
        for (int p = 0; p < K_BLOCKS_PER_WARP / 4; ++p) {
            out_packed[sf_word_index<SF_LAYOUT>(m, kp_base + p, M_pad, num_kp)] = packed_words[p];
        }
    }
}

} // anonymous namespace

void fp8bs_quantize_1x32_packed(__nv_fp8_e4m3* x_q, int32_t* packed_scales,
    __nv_bfloat16 const* x, int M, int K, cudaStream_t stream, bool use_ue8m0, bool sm1xx_sf_layout)
{
    // K_BLOCKS_PER_WARP = 8 (256 K-elements/warp, 2 int32 outputs/warp) when
    // K is a multiple of 256 — fewer CTAs, better amortisation of launch
    // overhead. Falls back to 4 when K % 256 != 0 (e.g. shapes where K is
    // only a multiple of 128).
    //
    // sm1xx_sf_layout=true → Sm1xxBlockScaledConfig atom layout for the
    // sm_100/sm_103 tcgen05 kernels; M pads to 128 so the padded warps
    // zero the full SF atom block (the m >= M branch).
    int const m_pad = sm1xx_sf_layout ? ((M + 127) / 128) * 128 : ((M + 3) / 4) * 4;
    constexpr int kThreadsPerBlock = 256;
    constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;

    auto launch = [&](auto kBlocksPerWarpT, auto ue8m0T, auto sfLayoutT) {
        constexpr int kBlocksPerWarp = decltype(kBlocksPerWarpT)::value;
        int const k_groups = K / (kMxFp8VecSize * kBlocksPerWarp);
        int const total_warps = m_pad * k_groups;
        int const grid = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
        fp8bs_quantize_1x32_packed_kernel<decltype(ue8m0T)::value, kBlocksPerWarp, decltype(sfLayoutT)::value>
            <<<grid, kThreadsPerBlock, 0, stream>>>(x_q, packed_scales, x, M, K, m_pad);
    };

    auto launch_k = [&](auto ue8m0T, auto sfLayoutT) {
        if (K % 256 == 0) launch(std::integral_constant<int, 8>{}, ue8m0T, sfLayoutT);
        else              launch(std::integral_constant<int, 4>{}, ue8m0T, sfLayoutT);
    };

    using kSm120 = std::integral_constant<MxSfLayout, MxSfLayout::SM120_KMAJOR>;
    using kSm1xx = std::integral_constant<MxSfLayout, MxSfLayout::SM1XX_ATOM>;
    if (use_ue8m0) {
        if (sm1xx_sf_layout) launch_k(std::true_type{}, kSm1xx{});
        else                 launch_k(std::true_type{}, kSm120{});
    } else {
        if (sm1xx_sf_layout) launch_k(std::false_type{}, kSm1xx{});
        else                 launch_k(std::false_type{}, kSm120{});
    }
}

// ----- Fused silu·chunk + quantize 1×32 → packed (E35, 2026-05-28) ---------
//
// Input is `gu` [M, 2*INTER] bf16 (the GEMM output of gate_up). Each output
// row's first INTER cols are `gate`, second INTER cols are `up`. The kernel
// computes `h = silu(gate) * up` and immediately quantizes `h` to FP8 +
// packed UE8M0 scale, **without materialising `h` in global memory** —
// saving an intermediate write+read pass of size M·INTER·2 bytes.
//
// Replaces the two-kernel chain:
//   compiled silu*mul (Inductor)    [M·INTER·2 bytes write of h]
//   fp8bs_quantize_1x32_packed       [M·INTER·2 bytes read of h]
//
// Kernel structure identical to `fp8bs_quantize_1x32_packed_kernel`: one
// warp = K_BLOCKS_PER_WARP K-blocks of one M row, uint64 LDG.64 (4 BF16
// per lane). The only deltas vs the plain quantize kernel are:
//   * Each lane issues TWO LDG.64 (one for gate, one for up) instead of one.
//   * Per-element compute: silu(gate) = gate · sigmoid(gate); h = silu·up
//     replaces the simple BF16→FP32 conversion. Reduction + e8m0 + FP8
//     cast + packed-scale gather all unchanged.
namespace
{

template <bool USE_UE8M0, int K_BLOCKS_PER_WARP, MxSfLayout SF_LAYOUT = MxSfLayout::SM120_KMAJOR>
__global__ void silu_chunk_mul_quantize_1x32_packed_kernel(
    __nv_fp8_e4m3* __restrict__ out_fp8, int32_t* __restrict__ out_packed,
    __nv_bfloat16 const* __restrict__ gu, int M, int K, int M_pad)
{
    static_assert(K_BLOCKS_PER_WARP == 4 || K_BLOCKS_PER_WARP == 8,
        "K_BLOCKS_PER_WARP must be 4 (1 int32/warp) or 8 (2 int32/warp).");
    static_assert(K_BLOCKS_PER_WARP % 4 == 0,
        "uint64 vectorised inner iter handles 4 K-blocks at a time");

    constexpr int kVec = kMxFp8VecSize;
    constexpr int kIterKBlocks = 4;
    constexpr int kNumIters = K_BLOCKS_PER_WARP / kIterKBlocks;

    int const warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int const lane_id = threadIdx.x & 31;
    int const k_groups = K / (kVec * K_BLOCKS_PER_WARP);
    int const total_warps = M_pad * k_groups;
    if (warp_id >= total_warps) return;

    int const m  = warp_id / k_groups;
    int const kg = warp_id % k_groups;
    int const kp_base = kg * (K_BLOCKS_PER_WARP / 4);
    int const num_kp = K / 128;  // int32 SF words per row
    int const stride_m_gu = 2 * K;  // gu has 2*K cols (gate then up)

    if (m >= M) {
        if (lane_id == 0) {
#pragma unroll
            for (int p = 0; p < K_BLOCKS_PER_WARP / 4; ++p) {
                out_packed[sf_word_index<SF_LAYOUT>(m, kp_base + p, M_pad, num_kp)] = 0;
            }
        }
        return;
    }

    int packed_words[K_BLOCKS_PER_WARP / 4] = {};

    // NOTE(ncu 2026-07-06): IMAD (address arithmetic) is this kernel's #1
    // opcode with the issue port at ~75%, but hoisting base pointers out of
    // the loop measured NEUTRAL-to-negative (27.4 → 28.1 µs) — the compiler
    // already CSEs the bases; the residual IMADs are irreducible 64-bit
    // address forms. Kernel is issue-bound at ~27 µs vs ~16 µs roofline
    // (M=2048); remaining cost is the shfl chain + e8m0 + converts.
#pragma unroll
    for (int it = 0; it < kNumIters; ++it) {
        int const kb_base = kg * K_BLOCKS_PER_WARP + it * kIterKBlocks;
        int const k_base  = kb_base * kVec;

        // Two LDG.64 per lane: gate[m, k_base + lane*4..lane*4+3]
        //                     up  [m, k_base + lane*4..lane*4+3]
        // gate lives at gu[m, 0..K), up at gu[m, K..2K). Same M, different col offset.
        uint64_t const gate_word = *reinterpret_cast<uint64_t const*>(
            &gu[m * stride_m_gu + 0 + k_base + lane_id * 4]);
        uint64_t const up_word = *reinterpret_cast<uint64_t const*>(
            &gu[m * stride_m_gu + K + k_base + lane_id * 4]);
        __nv_bfloat162 const* gv2 = reinterpret_cast<__nv_bfloat162 const*>(&gate_word);
        __nv_bfloat162 const* uv2 = reinterpret_cast<__nv_bfloat162 const*>(&up_word);

        // h_i = silu(gate_i) * up_i, bf16x2 SIMD (1 MUFU per 2 elems; see
        // silu2_mul).
        float2 const f01 = __bfloat1622float2(silu2_mul(gv2[0], uv2[0]));
        float2 const f23 = __bfloat1622float2(silu2_mul(gv2[1], uv2[1]));
        float const h0 = f01.x, h1 = f01.y, h2 = f23.x, h3 = f23.y;

        float my_ax = fmaxf(fmaxf(fabsf(h0), fabsf(h1)), fmaxf(fabsf(h2), fabsf(h3)));

#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            my_ax = fmaxf(my_ax, __shfl_xor_sync(0xFFFFFFFFu, my_ax, off));
        }
        my_ax = fmaxf(my_ax, 1e-10f);

        float qs;
        uint8_t byte_v;
        e8m0_from_amax<USE_UE8M0>(my_ax, qs, byte_v);

        uint32_t const fp_word = fp8x4_from_floats(h0 * qs, h1 * qs, h2 * qs, h3 * qs);
        *reinterpret_cast<uint32_t*>(&out_fp8[m * K + k_base + lane_id * 4]) = fp_word;

        uint32_t const b0 = byte_v;
        uint32_t const b1 = __shfl_sync(0xFFFFFFFFu, byte_v, 8);
        uint32_t const b2 = __shfl_sync(0xFFFFFFFFu, byte_v, 16);
        uint32_t const b3 = __shfl_sync(0xFFFFFFFFu, byte_v, 24);
        if (lane_id == 0) {
            packed_words[it] = static_cast<int>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
        }
    }

    if (lane_id == 0) {
#pragma unroll
        for (int p = 0; p < K_BLOCKS_PER_WARP / 4; ++p) {
            out_packed[sf_word_index<SF_LAYOUT>(m, kp_base + p, M_pad, num_kp)] = packed_words[p];
        }
    }
}

} // anonymous namespace

void fp8bs_silu_chunk_mul_quantize_1x32_packed(
    __nv_fp8_e4m3* x_q, int32_t* packed_scales,
    __nv_bfloat16 const* gu, int M, int K, cudaStream_t stream, bool use_ue8m0, bool sm1xx_sf_layout)
{
    // See fp8bs_quantize_1x32_packed for the sm1xx_sf_layout contract.
    int const m_pad = sm1xx_sf_layout ? ((M + 127) / 128) * 128 : ((M + 3) / 4) * 4;
    constexpr int kThreadsPerBlock = 256;
    constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;

    auto launch = [&](auto kBlocksPerWarpT, auto ue8m0T, auto sfLayoutT) {
        constexpr int kBlocksPerWarp = decltype(kBlocksPerWarpT)::value;
        int const k_groups = K / (kMxFp8VecSize * kBlocksPerWarp);
        int const total_warps = m_pad * k_groups;
        int const grid = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
        silu_chunk_mul_quantize_1x32_packed_kernel<decltype(ue8m0T)::value, kBlocksPerWarp, decltype(sfLayoutT)::value>
            <<<grid, kThreadsPerBlock, 0, stream>>>(x_q, packed_scales, gu, M, K, m_pad);
    };

    auto launch_k = [&](auto ue8m0T, auto sfLayoutT) {
        if (K % 256 == 0) launch(std::integral_constant<int, 8>{}, ue8m0T, sfLayoutT);
        else              launch(std::integral_constant<int, 4>{}, ue8m0T, sfLayoutT);
    };

    using kSm120 = std::integral_constant<MxSfLayout, MxSfLayout::SM120_KMAJOR>;
    using kSm1xx = std::integral_constant<MxSfLayout, MxSfLayout::SM1XX_ATOM>;
    if (use_ue8m0) {
        if (sm1xx_sf_layout) launch_k(std::true_type{}, kSm1xx{});
        else                 launch_k(std::true_type{}, kSm120{});
    } else {
        if (sm1xx_sf_layout) launch_k(std::false_type{}, kSm1xx{});
        else                 launch_k(std::false_type{}, kSm120{});
    }
}

// ----- Grouped (MoE, masked) BLOCK-SCALE FP8 1x128 FP32-K-major quantize (H2)
//
// The sm_90 (H200) layout-native quantizer: fused token-gather + 1x128
// block-scale FP8 quantize writing FP32 K-major scales DIRECTLY in the
// GroupedMasked kernel's SFA layout — SFA[(g*Kb + kb)*m_cap + m] = dequant
// scale (per-token amax/448, matching per_token_cast(use_ue8m0=False)).
// This is exactly what the revived sm_90 GroupedMasked kernel's TMA
// descriptor reads (ColMajor [pad(m_cap,4),(K/128)*G]), so it eliminates the
// stock deep_gemm pipeline's per_token_cast + tma_align_input_scale two-step
// (that transform is the launch/pass the fso layout-native quant removes).
// Full-warp amax per 128-K-block (32 lanes x 4 bf16). m_cap must be a
// multiple of 4 (== pad(m_cap,4)); on sm_90 the masked slab uses m_cap >= 64.

namespace
{

__global__ void fp8bs_quantize_1x128_fp32_grouped_gather_kernel(
    __nv_fp8_e4m3* __restrict__ out_fp8, float* __restrict__ out_sfa,
    __nv_bfloat16 const* __restrict__ input, int32_t const* __restrict__ slot_of_flat,
    int n_pairs, int topk, int m_cap, int K)
{
    int const warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int const lane_id = threadIdx.x & 31;
    int const Kb = K / 128;
    int const i = warp_id / Kb;   // routed pair
    int const kb = warp_id % Kb;  // 128-K-block
    if (i >= n_pairs)
        return;
    int const slot = slot_of_flat[i];
    int const g = slot / m_cap;
    int const m_in = slot % m_cap;
    int const src = i / topk;
    int const k_base = kb * 128;

    uint64_t const xword = *reinterpret_cast<uint64_t const*>(
        &input[static_cast<int64_t>(src) * K + k_base + lane_id * 4]);
    __nv_bfloat16 const* xv = reinterpret_cast<__nv_bfloat16 const*>(&xword);
    float const x0 = __bfloat162float(xv[0]);
    float const x1 = __bfloat162float(xv[1]);
    float const x2 = __bfloat162float(xv[2]);
    float const x3 = __bfloat162float(xv[3]);
    float my_ax = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        my_ax = fmaxf(my_ax, __shfl_xor_sync(0xFFFFFFFFu, my_ax, off));
    my_ax = fmaxf(my_ax, 1e-10f);
    float const qs = 448.f / my_ax;         // quant scale
    float const dequant = my_ax * (1.f / 448.f);
    uint32_t const fp_word = fp8x4_from_floats(x0 * qs, x1 * qs, x2 * qs, x3 * qs);
    *reinterpret_cast<uint32_t*>(
        &out_fp8[static_cast<int64_t>(slot) * K + k_base + lane_id * 4]) = fp_word;
    if (lane_id == 0)
        out_sfa[(static_cast<int64_t>(g) * Kb + kb) * m_cap + m_in] = dequant;
}

__global__ void silu_chunk_mul_quantize_1x128_fp32_grouped_kernel(
    __nv_fp8_e4m3* __restrict__ out_fp8, float* __restrict__ out_sfa,
    __nv_bfloat16 const* __restrict__ gu, int32_t const* __restrict__ slot_of_flat,
    int n_pairs, int m_cap, int K)
{
    int const warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int const lane_id = threadIdx.x & 31;
    int const Kb = K / 128;
    int const i = warp_id / Kb;
    int const kb = warp_id % Kb;
    if (i >= n_pairs)
        return;
    int const slot = slot_of_flat[i];
    int const g = slot / m_cap;
    int const m_in = slot % m_cap;
    int64_t const stride_m_gu = 2 * static_cast<int64_t>(K);
    int const k_base = kb * 128;

    uint64_t const gate_word = *reinterpret_cast<uint64_t const*>(
        &gu[slot * stride_m_gu + 0 + k_base + lane_id * 4]);
    uint64_t const up_word = *reinterpret_cast<uint64_t const*>(
        &gu[slot * stride_m_gu + K + k_base + lane_id * 4]);
    __nv_bfloat162 const* gv2 = reinterpret_cast<__nv_bfloat162 const*>(&gate_word);
    __nv_bfloat162 const* uv2 = reinterpret_cast<__nv_bfloat162 const*>(&up_word);
    float2 const f01 = __bfloat1622float2(silu2_mul(gv2[0], uv2[0]));
    float2 const f23 = __bfloat1622float2(silu2_mul(gv2[1], uv2[1]));
    float const h0 = f01.x, h1 = f01.y, h2 = f23.x, h3 = f23.y;
    float my_ax = fmaxf(fmaxf(fabsf(h0), fabsf(h1)), fmaxf(fabsf(h2), fabsf(h3)));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        my_ax = fmaxf(my_ax, __shfl_xor_sync(0xFFFFFFFFu, my_ax, off));
    my_ax = fmaxf(my_ax, 1e-10f);
    float const qs = 448.f / my_ax;
    float const dequant = my_ax * (1.f / 448.f);
    uint32_t const fp_word = fp8x4_from_floats(h0 * qs, h1 * qs, h2 * qs, h3 * qs);
    *reinterpret_cast<uint32_t*>(
        &out_fp8[static_cast<int64_t>(slot) * K + k_base + lane_id * 4]) = fp_word;
    if (lane_id == 0)
        out_sfa[(static_cast<int64_t>(g) * Kb + kb) * m_cap + m_in] = dequant;
}

// ----- Contiguous (triton-style sorted layout) 1x128 quant variants (H4) ----
//
// Companions of the sm_90 GroupedContiguous GEMM. The activation is a flat
// expert-sorted matrix [P_max, K] and the scale is the DENSE K-major layout the
// contiguous SFA TMA descriptor expects: ColMajor [align(P_max,4), K/128], i.e.
//   sfa(r, kb) = kb * sfa_ld + r,  sfa_ld = align(P_max, 4).
// Both iterate the R real routed pairs (i in [0, M*topk)) and place each at its
// sorted row via flat_to_sorted[i] — so the work scales with R, not P_max, and
// there is no O(P_max) memset. Padding rows of the output are left untouched:
// the GroupedContiguous GEMM is row-independent, so a padding row only produces
// a padding output row, which the combine drops (it reads only real rows via
// flat_to_sorted). Each warp owns one (pair i, 128-K-block kb).
__global__ void fp8bs_quantize_1x128_fp32_sorted_gather_kernel(
    __nv_fp8_e4m3* __restrict__ out_fp8,  // [P_max, K]
    float* __restrict__ out_sfa,          // [K/128, sfa_ld] = ColMajor[sfa_ld, K/128]
    __nv_bfloat16 const* __restrict__ input,   // [M, K]
    int32_t const* __restrict__ flat_to_sorted, // [R]: pair -> sorted row
    int n_pairs, int topk, int sfa_ld, int K)
{
    int const warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int const lane_id = threadIdx.x & 31;
    int const Kb = K / 128;
    int const i = warp_id / Kb;   // routed pair
    int const kb = warp_id % Kb;  // 128-K-block
    if (i >= n_pairs)
        return;
    int const r = flat_to_sorted[i]; // sorted row
    int const k_base = kb * 128;
    int const token = i / topk;
    uint64_t const xword = *reinterpret_cast<uint64_t const*>(
        &input[static_cast<int64_t>(token) * K + k_base + lane_id * 4]);
    __nv_bfloat16 const* xv = reinterpret_cast<__nv_bfloat16 const*>(&xword);
    float const x0 = __bfloat162float(xv[0]);
    float const x1 = __bfloat162float(xv[1]);
    float const x2 = __bfloat162float(xv[2]);
    float const x3 = __bfloat162float(xv[3]);
    float my_ax = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        my_ax = fmaxf(my_ax, __shfl_xor_sync(0xFFFFFFFFu, my_ax, off));
    my_ax = fmaxf(my_ax, 1e-10f);
    float const qs = 448.f / my_ax;
    float const dequant = my_ax * (1.f / 448.f);
    uint32_t const fp_word = fp8x4_from_floats(x0 * qs, x1 * qs, x2 * qs, x3 * qs);
    *reinterpret_cast<uint32_t*>(&out_fp8[static_cast<int64_t>(r) * K + k_base + lane_id * 4]) = fp_word;
    if (lane_id == 0)
        out_sfa[static_cast<int64_t>(kb) * sfa_ld + r] = dequant;
}

__global__ void silu_chunk_mul_quantize_1x128_fp32_sorted_kernel(
    __nv_fp8_e4m3* __restrict__ out_fp8,  // [P_max, K] (K = INTER)
    float* __restrict__ out_sfa,          // ColMajor[sfa_ld, K/128]
    __nv_bfloat16 const* __restrict__ gu, // [P_max, 2*K] gate_up output, sorted
    int32_t const* __restrict__ flat_to_sorted, // [R]: pair -> sorted row
    int n_pairs, int sfa_ld, int K)
{
    int const warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int const lane_id = threadIdx.x & 31;
    int const Kb = K / 128;
    int const i = warp_id / Kb;
    int const kb = warp_id % Kb;
    if (i >= n_pairs)
        return;
    int const r = flat_to_sorted[i];
    int const k_base = kb * 128;
    int64_t const stride_m_gu = 2 * static_cast<int64_t>(K);
    uint64_t const gate_word = *reinterpret_cast<uint64_t const*>(
        &gu[static_cast<int64_t>(r) * stride_m_gu + 0 + k_base + lane_id * 4]);
    uint64_t const up_word = *reinterpret_cast<uint64_t const*>(
        &gu[static_cast<int64_t>(r) * stride_m_gu + K + k_base + lane_id * 4]);
    __nv_bfloat162 const* gv2 = reinterpret_cast<__nv_bfloat162 const*>(&gate_word);
    __nv_bfloat162 const* uv2 = reinterpret_cast<__nv_bfloat162 const*>(&up_word);
    float2 const f01 = __bfloat1622float2(silu2_mul(gv2[0], uv2[0]));
    float2 const f23 = __bfloat1622float2(silu2_mul(gv2[1], uv2[1]));
    float const h0 = f01.x, h1 = f01.y, h2 = f23.x, h3 = f23.y;
    float my_ax = fmaxf(fmaxf(fabsf(h0), fabsf(h1)), fmaxf(fabsf(h2), fabsf(h3)));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        my_ax = fmaxf(my_ax, __shfl_xor_sync(0xFFFFFFFFu, my_ax, off));
    my_ax = fmaxf(my_ax, 1e-10f);
    float const qs = 448.f / my_ax;
    float const dequant = my_ax * (1.f / 448.f);
    uint32_t const fp_word = fp8x4_from_floats(h0 * qs, h1 * qs, h2 * qs, h3 * qs);
    *reinterpret_cast<uint32_t*>(&out_fp8[static_cast<int64_t>(r) * K + k_base + lane_id * 4]) = fp_word;
    if (lane_id == 0)
        out_sfa[static_cast<int64_t>(kb) * sfa_ld + r] = dequant;
}

} // anonymous namespace

void fp8bs_quantize_1x128_fp32_sorted_gather(__nv_fp8_e4m3* x_q, float* sfa,
    __nv_bfloat16 const* x, int32_t const* flat_to_sorted, int n_pairs, int topk,
    int sfa_ld, int K, cudaStream_t stream)
{
    constexpr int kThreads = 256, kWarps = kThreads / 32;
    int const Kb = K / 128;
    int64_t const total_warps = static_cast<int64_t>(n_pairs) * Kb;
    int const grid = static_cast<int>((total_warps + kWarps - 1) / kWarps);
    fp8bs_quantize_1x128_fp32_sorted_gather_kernel<<<grid, kThreads, 0, stream>>>(
        x_q, sfa, x, flat_to_sorted, n_pairs, topk, sfa_ld, K);
}

void fp8bs_silu_chunk_mul_quantize_1x128_fp32_sorted(__nv_fp8_e4m3* x_q, float* sfa,
    __nv_bfloat16 const* gu, int32_t const* flat_to_sorted, int n_pairs, int sfa_ld, int K,
    cudaStream_t stream)
{
    constexpr int kThreads = 256, kWarps = kThreads / 32;
    int const Kb = K / 128;
    int64_t const total_warps = static_cast<int64_t>(n_pairs) * Kb;
    int const grid = static_cast<int>((total_warps + kWarps - 1) / kWarps);
    silu_chunk_mul_quantize_1x128_fp32_sorted_kernel<<<grid, kThreads, 0, stream>>>(
        x_q, sfa, gu, flat_to_sorted, n_pairs, sfa_ld, K);
}

void fp8bs_quantize_1x128_fp32_grouped_gather(__nv_fp8_e4m3* x_q, float* sfa,
    __nv_bfloat16 const* x, int32_t const* slot_of_flat, int n_pairs, int topk,
    int m_cap, int K, cudaStream_t stream)
{
    constexpr int kThreads = 256, kWarps = kThreads / 32;
    int const Kb = K / 128;
    int64_t const total_warps = static_cast<int64_t>(n_pairs) * Kb;
    int const grid = static_cast<int>((total_warps + kWarps - 1) / kWarps);
    fp8bs_quantize_1x128_fp32_grouped_gather_kernel<<<grid, kThreads, 0, stream>>>(
        x_q, sfa, x, slot_of_flat, n_pairs, topk, m_cap, K);
}

void fp8bs_silu_chunk_mul_quantize_1x128_fp32_grouped(__nv_fp8_e4m3* x_q, float* sfa,
    __nv_bfloat16 const* gu, int32_t const* slot_of_flat, int n_pairs, int m_cap, int K,
    cudaStream_t stream)
{
    constexpr int kThreads = 256, kWarps = kThreads / 32;
    int const Kb = K / 128;
    int64_t const total_warps = static_cast<int64_t>(n_pairs) * Kb;
    int const grid = static_cast<int>((total_warps + kWarps - 1) / kWarps);
    silu_chunk_mul_quantize_1x128_fp32_grouped_kernel<<<grid, kThreads, 0, stream>>>(
        x_q, sfa, gu, slot_of_flat, n_pairs, m_cap, K);
}

// ----- Grouped (MoE, masked layout) MXFP8 quantize variants (M1, M3) -------
//
// Companions of the grouped GEMMs. Each writes the per-group packed-scale
// layout the target architecture's SFA TMA descriptor expects, selected by the
// `sm1xx_sf_layout` argument:
//
//   sm_120/121 (M1): per-group K-major words,
//     sf_word(g, m_in, kp) = g * (num_kp * m_cap) + kp * m_cap + m_in
//   sm_100/103 (M3): per-group CUTLASS Sm1xx atom slab of
//     pad(m_cap,128) * num_kp words — see sf_word_index_grouped_atom below.
//
// with m_cap % 4 == 0. Both index the FLAT ROUTED-PAIR space
// (i in [0, M*topk), host-static) and derive (g, m_in) from
// slot_of_flat[i] — iterating the padded G*m_cap space cost a ~94%-dead
// warp scan at M=512. Rows of the grouped outputs that no pair maps to
// stay undefined; the GEMM's masked contract already ignores them.
//
//   * ..._grouped_gather: fused token-gather + quantize (src row = i/topk).
//   * ..._grouped_scatter: the same result computed once per TOKEN and
//     scattered to that token's topk slots — see its own comment below.
//   * silu_chunk_mul_quantize_1x32_packed_grouped: SwiGLU fused quant for
//     the grouped down-GEMM input (row = slot, in place in the pair space).
//     This one has no fan-out — each destination slot has exactly one source
//     row — so there is nothing for a scatter form to amortise here.

namespace
{

__device__ __forceinline__ int sf_word_index_grouped(int m_in, int kp, int m_cap, int num_kp, int g)
{
    return g * (num_kp * m_cap) + kp * m_cap + m_in;
}

// sm_100 / sm_103 grouped scale-factor addressing.
//
// Each group owns a slab in the CUTLASS Sm1xxBlockScaledConfig<32> atom layout
// (see MxSfLayout::SM1XX_ATOM above) describing a (pad(m_cap,128), K) tensor:
// per 128-row x 128-K block there is one 512-byte scale-factor block, and a
// row's 4 consecutive K-block bytes stay contiguous, so each (m_in, kp) is
// still exactly one int32 word. Group slabs are laid end to end, so group g's
// words start at g * pad(m_cap,128) * (K/128).
//
// Rows of a group between masked_m[g] and pad(m_cap,128) are never written
// here and never read by the GEMM: the grouped launcher sets group g's problem
// shape to (masked_m[g], N, K), so the scale-factor TMA descriptor's row
// extent stops at masked_m[g] and every row past it is an out-of-bounds read
// that the TMA replaces with zero.
__device__ __forceinline__ int64_t sf_word_index_grouped_atom(int m_in, int kp, int m_cap, int num_kp, int g)
{
    int const m_pad = (m_cap + 127) / 128 * 128;
    int const r = m_in % 128;
    return static_cast<int64_t>(g) * (static_cast<int64_t>(num_kp) * m_pad)
        + (static_cast<int64_t>(m_in / 128) * num_kp + kp) * 128 + (r % 32) * 4 + r / 32;
}

template <bool USE_UE8M0, int K_BLOCKS_PER_WARP, bool SM1XX_SF = false>
__global__ void fp8bs_quantize_1x32_packed_grouped_gather_kernel(
    __nv_fp8_e4m3* __restrict__ out_fp8, int32_t* __restrict__ out_packed,
    __nv_bfloat16 const* __restrict__ input, int32_t const* __restrict__ slot_of_flat,
    int n_pairs, int topk, int m_cap, int K, bool pdl)
{
    static_assert(K_BLOCKS_PER_WARP == 4 || K_BLOCKS_PER_WARP == 8,
        "K_BLOCKS_PER_WARP must be 4 (1 int32/warp) or 8 (2 int32/warp).");

    constexpr int kVec = kMxFp8VecSize;
    constexpr int kIterKBlocks = 4;
    constexpr int kNumIters = K_BLOCKS_PER_WARP / kIterKBlocks;

    // PDL: order the slot_of_flat / input reads behind the parent, then
    // release the dependent's prologue. Armed only for small grids.
    if (pdl)
    {
        cudaGridDependencySynchronize();
        if (threadIdx.x == 0)
        {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }
    // Flat-pair indexing: one warp per (routed pair, K-chunk). The padded
    // G*m_cap iteration space this kernel used at first cost a 94%-dead
    // warp scan at M=512 (65k CTAs, ~27 us of pure early-outs); the pair
    // space is exactly the valid work and is host-static.
    int const warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int const lane_id = threadIdx.x & 31;
    int const k_groups = K / (kVec * K_BLOCKS_PER_WARP);
    int const i = warp_id / k_groups;   // routed pair index
    int const kg = warp_id % k_groups;
    if (i >= n_pairs)
        return;
    int const slot = slot_of_flat[i];   // g * m_cap + m_in
    int const g = slot / m_cap;
    int const m_in = slot % m_cap;
    int const src = i / topk;           // source token row

    int const kp_base = kg * (K_BLOCKS_PER_WARP / 4);
    int const num_kp = K / 128;

    int packed_words[K_BLOCKS_PER_WARP / 4] = {};

#pragma unroll
    for (int it = 0; it < kNumIters; ++it) {
        int const kb_base = kg * K_BLOCKS_PER_WARP + it * kIterKBlocks;
        int const k_base  = kb_base * kVec;

        uint64_t const xword = *reinterpret_cast<uint64_t const*>(
            &input[static_cast<int64_t>(src) * K + k_base + lane_id * 4]);
        __nv_bfloat16 const* xv = reinterpret_cast<__nv_bfloat16 const*>(&xword);
        float const x0 = __bfloat162float(xv[0]);
        float const x1 = __bfloat162float(xv[1]);
        float const x2 = __bfloat162float(xv[2]);
        float const x3 = __bfloat162float(xv[3]);

        float my_ax = fmaxf(fmaxf(fabsf(x0), fabsf(x1)),
                            fmaxf(fabsf(x2), fabsf(x3)));
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            my_ax = fmaxf(my_ax, __shfl_xor_sync(0xFFFFFFFFu, my_ax, off));
        }
        my_ax = fmaxf(my_ax, 1e-10f);

        float qs;
        uint8_t byte_v;
        e8m0_from_amax<USE_UE8M0>(my_ax, qs, byte_v);

        uint32_t const fp_word = fp8x4_from_floats(x0 * qs, x1 * qs, x2 * qs, x3 * qs);
        *reinterpret_cast<uint32_t*>(
            &out_fp8[static_cast<int64_t>(slot) * K + k_base + lane_id * 4]) = fp_word;

        uint32_t const b0 = byte_v;
        uint32_t const b1 = __shfl_sync(0xFFFFFFFFu, byte_v, 8);
        uint32_t const b2 = __shfl_sync(0xFFFFFFFFu, byte_v, 16);
        uint32_t const b3 = __shfl_sync(0xFFFFFFFFu, byte_v, 24);
        if (lane_id == 0) {
            packed_words[it] = static_cast<int>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
        }
    }

    if (lane_id == 0) {
#pragma unroll
        for (int p = 0; p < K_BLOCKS_PER_WARP / 4; ++p) {
            if constexpr (SM1XX_SF) {
                out_packed[sf_word_index_grouped_atom(m_in, kp_base + p, m_cap, num_kp, g)] = packed_words[p];
            } else {
                out_packed[sf_word_index_grouped(m_in, kp_base + p, m_cap, num_kp, g)] = packed_words[p];
            }
        }
    }
}

// Token-space form of the kernel above: quantize each token ONCE and scatter
// the result to all of its topk destination slots.
//
// Precondition. `slot_of_flat` is dense and token-major: moe_build_routing
// writes exactly one entry per routed pair at index i = token * topk + j, so
// the topk destinations of token `src` are the contiguous entries
// slot_of_flat[src * topk .. src * topk + topk - 1], and every one of them is
// a valid slot (there is no "unrouted pair" sentinel in this space).
//
// What the pair-space kernel above does. It gives one warp to each
// (routed pair, K-chunk), so for a token routed to topk experts the SAME
// source row is loaded from `input`, amax-reduced across the warp, turned into
// a UE8M0 byte and converted to FP8 topk separate times, and the launch grid
// is topk times the number of distinct token rows.
//
// Why that is wasted work. The FP8 bytes and the scale byte a pair writes are
// functions of the token row alone -- the destination slot only selects WHERE
// they are stored. topk - 1 of the topk loads, reductions and conversions
// therefore recompute a value the kernel already holds.
//
// Consequence. The measurements in run b300_mxfp8_20260917/M-Q1 show the
// pair-space kernel running at a very low fraction of memory speed-of-light
// while issuing M * topk CTAs, i.e. it is latency- and occupancy-bound on
// redundant loads rather than bandwidth-bound on the bytes it owes. This form
// issues M CTAs, reads each token row once and performs the same number of
// stores, so the loads and the float work fall by a factor of topk while the
// store traffic is unchanged.
//
// The trade it makes. Parallelism: the grid is topk times smaller, so below
// the point where the token-space grid still fills the device the pair-space
// form has more CTAs to hide latency with and stays faster. The launcher's
// rule (see fso_gather_quant_use_scatter below) is what decides between them.
//
// TOPK_STATIC > 0 compiles the destination loop for a fixed topk so the slot
// loads can be hoisted above the arithmetic and the store loop fully unrolled;
// TOPK_STATIC == 0 keeps the runtime loop for any other topk.
//
// Architecture. This is an sm_100/103 path: the launcher only selects it when
// the sm_100 scale-factor layout was requested, and the body below is compiled
// out on every other architecture so the sm_90 and sm_120 SASS passes emit an
// empty stub (the same technique the sm_90 fatbin uses for sm_120-only code).
template <bool USE_UE8M0, int K_BLOCKS_PER_WARP, bool SM1XX_SF, int TOPK_STATIC>
__global__ void fp8bs_quantize_1x32_packed_grouped_scatter_kernel(
    __nv_fp8_e4m3* __restrict__ out_fp8, int32_t* __restrict__ out_packed,
    __nv_bfloat16 const* __restrict__ input, int32_t const* __restrict__ slot_of_flat,
    int num_tokens, int topk, int m_cap, int K, bool pdl)
{
    static_assert(K_BLOCKS_PER_WARP == 4 || K_BLOCKS_PER_WARP == 8,
        "K_BLOCKS_PER_WARP must be 4 (1 int32/warp) or 8 (2 int32/warp).");
    static_assert(TOPK_STATIC >= 0, "TOPK_STATIC must be 0 (runtime) or a positive topk.");

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)
    constexpr int kVec = kMxFp8VecSize;
    constexpr int kIterKBlocks = 4;
    constexpr int kNumIters = K_BLOCKS_PER_WARP / kIterKBlocks;

    // PDL entry (see the pair-space kernel above).
    if (pdl)
    {
        cudaGridDependencySynchronize();
        if (threadIdx.x == 0)
        {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }
    // Token-space indexing: one warp per (token row, K-chunk).
    int const warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int const lane_id = threadIdx.x & 31;
    int const k_groups = K / (kVec * K_BLOCKS_PER_WARP);
    int const src = warp_id / k_groups;   // source token row
    int const kg = warp_id % k_groups;
    if (src >= num_tokens)
        return;

    int const kp_base = kg * (K_BLOCKS_PER_WARP / 4);
    int const num_kp = K / 128;

    // Destination slots are pure address material: issue their loads before
    // the arithmetic so the scattered-store addresses are already resolved
    // when the amax reduction finishes.
    constexpr int kSlotRegs = TOPK_STATIC > 0 ? TOPK_STATIC : 1;
    int slots[kSlotRegs];
    if constexpr (TOPK_STATIC > 0)
    {
#pragma unroll
        for (int t = 0; t < TOPK_STATIC; ++t)
        {
            slots[t] = slot_of_flat[src * TOPK_STATIC + t];
        }
    }

    uint32_t fp_words[kNumIters] = {};
    int packed_words[K_BLOCKS_PER_WARP / 4] = {};

#pragma unroll
    for (int it = 0; it < kNumIters; ++it) {
        int const kb_base = kg * K_BLOCKS_PER_WARP + it * kIterKBlocks;
        int const k_base  = kb_base * kVec;

        uint64_t const xword = *reinterpret_cast<uint64_t const*>(
            &input[static_cast<int64_t>(src) * K + k_base + lane_id * 4]);
        __nv_bfloat16 const* xv = reinterpret_cast<__nv_bfloat16 const*>(&xword);
        float const x0 = __bfloat162float(xv[0]);
        float const x1 = __bfloat162float(xv[1]);
        float const x2 = __bfloat162float(xv[2]);
        float const x3 = __bfloat162float(xv[3]);

        float my_ax = fmaxf(fmaxf(fabsf(x0), fabsf(x1)),
                            fmaxf(fabsf(x2), fabsf(x3)));
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            my_ax = fmaxf(my_ax, __shfl_xor_sync(0xFFFFFFFFu, my_ax, off));
        }
        my_ax = fmaxf(my_ax, 1e-10f);

        float qs;
        uint8_t byte_v;
        e8m0_from_amax<USE_UE8M0>(my_ax, qs, byte_v);

        fp_words[it] = fp8x4_from_floats(x0 * qs, x1 * qs, x2 * qs, x3 * qs);

        uint32_t const b0 = byte_v;
        uint32_t const b1 = __shfl_sync(0xFFFFFFFFu, byte_v, 8);
        uint32_t const b2 = __shfl_sync(0xFFFFFFFFu, byte_v, 16);
        uint32_t const b3 = __shfl_sync(0xFFFFFFFFu, byte_v, 24);
        if (lane_id == 0) {
            packed_words[it] = static_cast<int>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
        }
    }

    // One destination: the same stores the pair-space kernel performs for the
    // pair that owns this slot, from values that were computed once.
    auto store_slot = [&](int slot) {
#pragma unroll
        for (int it = 0; it < kNumIters; ++it) {
            int const k_base = (kg * K_BLOCKS_PER_WARP + it * kIterKBlocks) * kVec;
            *reinterpret_cast<uint32_t*>(
                &out_fp8[static_cast<int64_t>(slot) * K + k_base + lane_id * 4]) = fp_words[it];
        }
        if (lane_id == 0) {
            int const g = slot / m_cap;
            int const m_in = slot % m_cap;
#pragma unroll
            for (int p = 0; p < K_BLOCKS_PER_WARP / 4; ++p) {
                if constexpr (SM1XX_SF) {
                    out_packed[sf_word_index_grouped_atom(m_in, kp_base + p, m_cap, num_kp, g)] = packed_words[p];
                } else {
                    out_packed[sf_word_index_grouped(m_in, kp_base + p, m_cap, num_kp, g)] = packed_words[p];
                }
            }
        }
    };

    if constexpr (TOPK_STATIC > 0)
    {
#pragma unroll
        for (int t = 0; t < TOPK_STATIC; ++t)
        {
            store_slot(slots[t]);
        }
    }
    else
    {
        for (int t = 0; t < topk; ++t)
        {
            store_slot(slot_of_flat[src * topk + t]);
        }
    }
#else
    (void) out_fp8; (void) out_packed; (void) input; (void) slot_of_flat;
    (void) num_tokens; (void) topk; (void) m_cap; (void) K; (void) pdl;
#endif
}

// PAIRWISE selects where the kernel finds gate_i and up_i inside a row of `gu`.
//
// false (the layout every caller used before the sm_100 fused FC1 existed): the
// row is [gate_0 .. gate_{K-1}, up_0 .. up_{K-1}], so up_i sits K columns after
// gate_i and the two halves are loaded as two separate 8-byte words.
//
// true: the row is [gate_0, up_0, gate_1, up_1, ...], the order a GEMM produces
// when its weight rows are gate/up interleaved. The four outputs a lane owns
// then need eight consecutive bf16 values, which is one 16-byte load instead of
// two 8-byte ones, followed by a de-interleave into the (gate, gate) and
// (up, up) pairs that `silu2_mul` takes. Everything after that point — the
// amax reduction, the UE8M0 byte, the fp8 conversion and both stores — is
// shared, so the two forms produce bit-identical results on the same values.
//
// Why a template flag rather than a second kernel: the two differ only in the
// load and the shuffle that follows it, and a compile-time flag keeps the
// PAIRWISE=false instantiation's machine code exactly what it was (verified by
// disassembly, see run b300_mxfp8_20260917/M-E2).
template <bool USE_UE8M0, int K_BLOCKS_PER_WARP, bool SM1XX_SF = false, bool PAIRWISE = false>
__global__ void silu_chunk_mul_quantize_1x32_packed_grouped_kernel(
    __nv_fp8_e4m3* __restrict__ out_fp8, int32_t* __restrict__ out_packed,
    __nv_bfloat16 const* __restrict__ gu, int32_t const* __restrict__ slot_of_flat,
    int n_pairs, int m_cap, int K, bool pdl)
{
    static_assert(K_BLOCKS_PER_WARP == 4 || K_BLOCKS_PER_WARP == 8,
        "K_BLOCKS_PER_WARP must be 4 (1 int32/warp) or 8 (2 int32/warp).");

    constexpr int kVec = kMxFp8VecSize;
    constexpr int kIterKBlocks = 4;
    constexpr int kNumIters = K_BLOCKS_PER_WARP / kIterKBlocks;

    // PDL entry (see gather kernel above).
    if (pdl)
    {
        cudaGridDependencySynchronize();
        if (threadIdx.x == 0)
        {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }
    // Flat-pair indexing over the valid routed rows (see gather kernel).
    int const warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int const lane_id = threadIdx.x & 31;
    int const k_groups = K / (kVec * K_BLOCKS_PER_WARP);
    int const i = warp_id / k_groups;
    int const kg = warp_id % k_groups;
    if (i >= n_pairs)
        return;
    int const slot = slot_of_flat[i];
    int const g = slot / m_cap;
    int const m_in = slot % m_cap;

    int const kp_base = kg * (K_BLOCKS_PER_WARP / 4);
    int const num_kp = K / 128;
    int64_t const stride_m_gu = 2 * static_cast<int64_t>(K);

    int packed_words[K_BLOCKS_PER_WARP / 4] = {};

#pragma unroll
    for (int it = 0; it < kNumIters; ++it) {
        int const kb_base = kg * K_BLOCKS_PER_WARP + it * kIterKBlocks;
        int const k_base  = kb_base * kVec;

        // Both branches end at the same two float2s, and the PAIRWISE=false
        // branch holds the original expressions unchanged so its machine code
        // is unchanged.
        float2 f01, f23;
        if constexpr (PAIRWISE)
        {
            // Eight consecutive bf16 = [g0,u0,g1,u1,g2,u2,g3,u3], 16-byte
            // aligned because the element offset is a multiple of 8 (k_base is
            // a multiple of 32 and the lane stride is 8).
            uint4 const w = *reinterpret_cast<uint4 const*>(
                &gu[slot * stride_m_gu + 2 * (k_base + lane_id * 4)]);
            // Each 32-bit word is one (gate, up) pair, gate in the low half.
            // __byte_perm picks bytes from two sources: nibble values 0-3 index
            // the first operand's bytes, 4-7 the second's, least significant
            // nibble first. 0x5410 therefore builds (a.lo, b.lo) = (gate, gate)
            // and 0x7632 builds (a.hi, b.hi) = (up, up).
            uint32_t const p0 = __byte_perm(w.x, w.y, 0x5410);
            uint32_t const p1 = __byte_perm(w.x, w.y, 0x7632);
            uint32_t const p2 = __byte_perm(w.z, w.w, 0x5410);
            uint32_t const p3 = __byte_perm(w.z, w.w, 0x7632);
            f01 = __bfloat1622float2(silu2_mul(*reinterpret_cast<__nv_bfloat162 const*>(&p0),
                *reinterpret_cast<__nv_bfloat162 const*>(&p1)));
            f23 = __bfloat1622float2(silu2_mul(*reinterpret_cast<__nv_bfloat162 const*>(&p2),
                *reinterpret_cast<__nv_bfloat162 const*>(&p3)));
        }
        else
        {
            uint64_t const gate_word = *reinterpret_cast<uint64_t const*>(
                &gu[slot * stride_m_gu + 0 + k_base + lane_id * 4]);
            uint64_t const up_word = *reinterpret_cast<uint64_t const*>(
                &gu[slot * stride_m_gu + K + k_base + lane_id * 4]);
            __nv_bfloat162 const* gv2 = reinterpret_cast<__nv_bfloat162 const*>(&gate_word);
            __nv_bfloat162 const* uv2 = reinterpret_cast<__nv_bfloat162 const*>(&up_word);

            f01 = __bfloat1622float2(silu2_mul(gv2[0], uv2[0]));
            f23 = __bfloat1622float2(silu2_mul(gv2[1], uv2[1]));
        }
        float const h0 = f01.x, h1 = f01.y, h2 = f23.x, h3 = f23.y;

        float my_ax = fmaxf(fmaxf(fabsf(h0), fabsf(h1)), fmaxf(fabsf(h2), fabsf(h3)));
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            my_ax = fmaxf(my_ax, __shfl_xor_sync(0xFFFFFFFFu, my_ax, off));
        }
        my_ax = fmaxf(my_ax, 1e-10f);

        float qs;
        uint8_t byte_v;
        e8m0_from_amax<USE_UE8M0>(my_ax, qs, byte_v);

        uint32_t const fp_word = fp8x4_from_floats(h0 * qs, h1 * qs, h2 * qs, h3 * qs);
        *reinterpret_cast<uint32_t*>(
            &out_fp8[static_cast<int64_t>(slot) * K + k_base + lane_id * 4]) = fp_word;

        uint32_t const b0 = byte_v;
        uint32_t const b1 = __shfl_sync(0xFFFFFFFFu, byte_v, 8);
        uint32_t const b2 = __shfl_sync(0xFFFFFFFFu, byte_v, 16);
        uint32_t const b3 = __shfl_sync(0xFFFFFFFFu, byte_v, 24);
        if (lane_id == 0) {
            packed_words[it] = static_cast<int>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
        }
    }

    if (lane_id == 0) {
#pragma unroll
        for (int p = 0; p < K_BLOCKS_PER_WARP / 4; ++p) {
            if constexpr (SM1XX_SF) {
                out_packed[sf_word_index_grouped_atom(m_in, kp_base + p, m_cap, num_kp, g)] = packed_words[p];
            } else {
                out_packed[sf_word_index_grouped(m_in, kp_base + p, m_cap, num_kp, g)] = packed_words[p];
            }
        }
    }
}

} // anonymous namespace


// PDL launch helper for the grouped MoE glue chain: launches with the
// programmatic-stream-serialization attribute so a downstream kernel's
// launch/prologue overlaps this one (both sides carry griddepcontrol
// wait/trigger). FSO_DISABLE_PDL=1 restores plain serialised launches.
static inline bool fso_pdl_enabled()
{
    static bool v = []
    {
        char const* e = std::getenv("FSO_DISABLE_PDL");
        return !(e && e[0] == '1');
    }();
    return v;
}

// Grids beyond this many CTAs launch without PDL: a huge programmatic
// dependent floods every SM with waiting CTAs and throttles the parent's
// tail (measured +46 us on the M=512 layer before this gate).
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

// ----- Token-scatter selection for the grouped gather-quantize -------------
//
// FSO_GATHER_QUANT_ONCE selects between the two forms of the grouped
// activation gather-quantize: `0` always uses the pair-space kernel (one warp
// per routed pair), `1` always uses the token-scatter kernel (one warp per
// token, result scattered to the topk slots), and leaving it unset uses the
// rule below. Read once into a function-local static, so a captured graph
// never re-reads the environment and every replay launches the same kernel.
static inline int fso_gather_quant_once_mode()
{
    static int const v = []
    {
        char const* e = std::getenv("FSO_GATHER_QUANT_ONCE");
        if (e == nullptr)
            return -1;
        if (e[0] == '0')
            return 0;
        if (e[0] == '1')
            return 1;
        return -1;
    }();
    return v;
}

// Measurement knob, not a tuning knob: `0` forbids the compile-time-topk
// instantiation of the scatter kernel so the runtime-topk loop can be timed
// against it inside one build (run b300_mxfp8_20260917/M-Q1).
static inline bool fso_gather_quant_topk_static()
{
    static bool const v = []
    {
        char const* e = std::getenv("FSO_GATHER_QUANT_TOPK_STATIC");
        return !(e && e[0] == '0');
    }();
    return v;
}

// Multiprocessor count of the current device, queried once. This translation
// unit has no ATen, so the CUDA runtime attribute is the available source; the
// query happens at the first launch (eager warm-up) and never inside a graph
// capture. Single-device assumption, the same one fso_pdl_enabled makes.
static inline int fso_device_sm_count()
{
    static int const v = []
    {
        int dev = 0;
        int n = 0;
        if (cudaGetDevice(&dev) == cudaSuccess
            && cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && n > 0)
        {
            return n;
        }
        return 1;
    }();
    return v;
}

// Where the two forms cross.
//
// The token-scatter form removes (topk - 1) / topk of the global loads and of
// the amax / conversion work, and issues exactly the same stores, but its grid
// is topk times smaller. Below the point where the token-space grid alone
// covers the device, the pair-space form's extra CTAs are what hides the
// scattered-store latency and it stays ahead despite the redundant loads;
// above it the machine is full either way and only the removed work is left.
// The rule therefore asks one question: does the token-space grid fill at
// least this many waves of CTAs across the device's multiprocessors? Stated in
// (num_tokens, k_groups, warps per block, SM count) rather than in M, it
// carries across families and K without a per-family constant. The crossover
// measured in run b300_mxfp8_20260917/M-Q1 sits at one wave.
constexpr int kFsoScatterMinWaves = 1;

static inline bool fso_gather_quant_use_scatter(int num_tokens, int topk, int k_groups, int warps_per_block)
{
    int const mode = fso_gather_quant_once_mode();
    if (mode == 0)
        return false;
    if (topk < 2)
        return false; // one destination per token: nothing to amortise
    if (mode == 1)
        return true;
    int64_t const ctas
        = (static_cast<int64_t>(num_tokens) * k_groups + warps_per_block - 1) / warps_per_block;
    return ctas >= static_cast<int64_t>(kFsoScatterMinWaves) * fso_device_sm_count();
}

// `sm1xx_sf_layout` picks the sm_100/sm_103 per-group Sm1xx atom slab instead
// of the sm_120 per-group K-major slab. The sm_120 instantiations are
// unchanged (SM1XX_SF defaults to false and the generated store is the same
// expression it always was), so sm_120 output stays byte-identical.
//
// Two kernels implement this call and produce byte-identical output on every
// defined row: the pair-space kernel (one warp per routed pair) and the
// token-scatter kernel (one warp per token, result scattered to the topk
// slots). The scatter form is offered on the sm_100/103 layout only and is
// selected by fso_gather_quant_use_scatter; sm_120 and sm_90 always take the
// pair-space kernel, exactly as before. Applying the same idea on sm_120 is a
// separate exercise: the arch is first-class, it is deferred, not dismissed.
void fp8bs_quantize_1x32_packed_grouped_gather(__nv_fp8_e4m3* x_q, int32_t* packed_scales,
    __nv_bfloat16 const* x, int32_t const* slot_of_flat, int n_pairs, int topk,
    int m_cap, int K, cudaStream_t stream, bool use_ue8m0, bool sm1xx_sf_layout)
{
    constexpr int kThreadsPerBlock = 256;
    constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;

    auto launch = [&](auto kBlocksPerWarpT, auto ue8m0T, auto sfAtomT) {
        constexpr int kBlocksPerWarp = decltype(kBlocksPerWarpT)::value;
        int const k_groups = K / (kMxFp8VecSize * kBlocksPerWarp);
        // Token-scatter form, sm_100/103 scale layout only. The pair space is
        // dense and token-major (n_pairs = num_tokens * topk), which is what
        // lets a token's topk destinations be read as one contiguous run of
        // slot_of_flat; if a caller ever passes a ragged pair space the
        // equality below fails and the pair-space kernel is used.
        if constexpr (decltype(sfAtomT)::value) {
            int const num_tokens = topk > 0 ? n_pairs / topk : 0;
            bool const dense = topk > 0 && num_tokens * topk == n_pairs;
            if (dense && fso_gather_quant_use_scatter(num_tokens, topk, k_groups, kWarpsPerBlock)) {
                int64_t const warps = static_cast<int64_t>(num_tokens) * k_groups;
                int const grid = static_cast<int>((warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
                bool const pdl = fso_pdl_enabled() && static_cast<unsigned>(grid) <= kFsoPdlMaxGridCtas;
                if (topk == 8 && fso_gather_quant_topk_static()) {
                    fso_pdl_launch(fp8bs_quantize_1x32_packed_grouped_scatter_kernel<decltype(ue8m0T)::value,
                                       kBlocksPerWarp, true, 8>,
                        dim3(grid), dim3(kThreadsPerBlock), stream, pdl, x_q, packed_scales, x, slot_of_flat,
                        num_tokens, topk, m_cap, K);
                } else {
                    fso_pdl_launch(fp8bs_quantize_1x32_packed_grouped_scatter_kernel<decltype(ue8m0T)::value,
                                       kBlocksPerWarp, true, 0>,
                        dim3(grid), dim3(kThreadsPerBlock), stream, pdl, x_q, packed_scales, x, slot_of_flat,
                        num_tokens, topk, m_cap, K);
                }
                return;
            }
        }
        int64_t const total_warps = static_cast<int64_t>(n_pairs) * k_groups;
        int const grid = static_cast<int>((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
        bool const pdl = fso_pdl_enabled() && static_cast<unsigned>(grid) <= kFsoPdlMaxGridCtas;
        fso_pdl_launch(fp8bs_quantize_1x32_packed_grouped_gather_kernel<decltype(ue8m0T)::value, kBlocksPerWarp,
                           decltype(sfAtomT)::value>,
            dim3(grid), dim3(kThreadsPerBlock), stream, pdl, x_q, packed_scales, x, slot_of_flat,
            n_pairs, topk, m_cap, K);
    };
    auto launch_k = [&](auto ue8m0T, auto sfAtomT) {
        if (K % 256 == 0) launch(std::integral_constant<int, 8>{}, ue8m0T, sfAtomT);
        else              launch(std::integral_constant<int, 4>{}, ue8m0T, sfAtomT);
    };
    auto launch_sf = [&](auto ue8m0T) {
        if (sm1xx_sf_layout) launch_k(ue8m0T, std::true_type{});
        else                 launch_k(ue8m0T, std::false_type{});
    };
    if (use_ue8m0) launch_sf(std::true_type{});
    else           launch_sf(std::false_type{});
}

void fp8bs_silu_chunk_mul_quantize_1x32_packed_grouped(__nv_fp8_e4m3* x_q, int32_t* packed_scales,
    __nv_bfloat16 const* gu, int32_t const* slot_of_flat, int n_pairs, int m_cap, int K,
    cudaStream_t stream, bool use_ue8m0, bool sm1xx_sf_layout, bool pairwise)
{
    constexpr int kThreadsPerBlock = 256;
    constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;

    auto launch = [&](auto kBlocksPerWarpT, auto ue8m0T, auto sfAtomT, auto pairT) {
        constexpr int kBlocksPerWarp = decltype(kBlocksPerWarpT)::value;
        int const k_groups = K / (kMxFp8VecSize * kBlocksPerWarp);
        int64_t const total_warps = static_cast<int64_t>(n_pairs) * k_groups;
        int const grid = static_cast<int>((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
        bool const pdl = fso_pdl_enabled() && static_cast<unsigned>(grid) <= kFsoPdlMaxGridCtas;
        fso_pdl_launch(silu_chunk_mul_quantize_1x32_packed_grouped_kernel<decltype(ue8m0T)::value, kBlocksPerWarp,
                           decltype(sfAtomT)::value, decltype(pairT)::value>,
            dim3(grid), dim3(kThreadsPerBlock), stream, pdl, x_q, packed_scales, gu, slot_of_flat,
            n_pairs, m_cap, K);
    };
    auto launch_k = [&](auto ue8m0T, auto sfAtomT, auto pairT) {
        if (K % 256 == 0) launch(std::integral_constant<int, 8>{}, ue8m0T, sfAtomT, pairT);
        else              launch(std::integral_constant<int, 4>{}, ue8m0T, sfAtomT, pairT);
    };
    // The interleaved (pairwise) row order only arises from the sm_100 fused
    // FC1's weight layout, so it is instantiated in the sm_100 scale-layout
    // branch alone. Nesting it here rather than as a fourth top-level
    // dimension is what keeps the sm_120 branch's instantiation set — and
    // therefore its device code — exactly what it was. The ATen op refuses
    // pairwise=true on any other architecture before reaching this point.
    auto launch_sf = [&](auto ue8m0T) {
        if (sm1xx_sf_layout)
        {
            if (pairwise) launch_k(ue8m0T, std::true_type{}, std::true_type{});
            else          launch_k(ue8m0T, std::true_type{}, std::false_type{});
        }
        else
        {
            launch_k(ue8m0T, std::false_type{}, std::false_type{});
        }
    };
    if (use_ue8m0) launch_sf(std::true_type{});
    else           launch_sf(std::false_type{});
}

// ----- Fused BSFP8 1×128 quantize (E36, 2026-05-28) ------------------------
//
// Drop-in replacement for the legacy `fp8_1x128_cs` (scale_1x128_kernel)
// kernel from blockscale_gemm/common/scale_kernels.cuh. Mirrors the design
// of `fp8bs_quantize_1x32_packed_kernel`:
//   * Each warp processes K_BLOCKS_PER_WARP K-blocks (each = 128 K-elements)
//     of one M row. For BSFP8 a single K-block exactly fills a warp at
//     uint64 LDG.64 (4 BF16/lane × 32 lanes = 128 BF16). We do
//     K_BLOCKS_PER_WARP=4 inner iters per warp ⇒ 512 K-elements/warp.
//   * Full-warp shfl_xor amax reduction inside each K-block.
//   * Two output formats selected by template:
//       - SCALE_OUTPUT_PACKED: 4 K-block bytes packed into one int32 K-major.
//         Used by sm_120 (CUTLASS Sm120BlockScaledKernel expects this).
//       - SCALE_OUTPUT_FP32  : 4 FP32 dequant scales written K-major.
//         Used by sm_90 (deep_gemm WGMMA path consumes FP32 scales).
//   * Eliminates: (a) the FP32 → packed scale round-trip on sm_120, (b) the
//     persistent-kernel grid-stride loop overhead of the legacy launcher.

// PACKED_INT32_SM1XX_ATOM (2026-09-05, B200/B300 block-FP8): the sm_100/sm_103
// tcgen05 BlockScaled collective consumes one UE8M0 byte per 32 K-elements in
// the Sm1xxBlockScaledConfig<32> atom layout (see MxSfLayout::SM1XX_ATOM). A
// 1x128 scale is expressed by writing the same byte into the 4 consecutive
// 32-element slots of its K-block, i.e. one int32 word `byte * 0x01010101`
// per (row, K-block) at sf_word_index<SM1XX_ATOM>(m, kb, M_pad, K/128). The
// GEMM then runs unchanged through the MXFP8 tiers — same trick the sm_120
// builder uses (kSFVecSize=128 is a software aggregation of the VS=32 atom).
enum class BSFp8ScaleFormat { FP32_KMAJOR, PACKED_INT32_KMAJOR, PACKED_INT32_SM1XX_ATOM };

namespace
{

template <bool USE_UE8M0, BSFp8ScaleFormat OUT_FMT, int K_BLOCKS_PER_WARP = 4>
__global__ void fp8bs_quantize_1x128_packed_kernel(
    __nv_fp8_e4m3* __restrict__ out_fp8, void* __restrict__ out_scale,
    __nv_bfloat16 const* __restrict__ input, int M, int K, int M_pad)
{
    static_assert(K_BLOCKS_PER_WARP == 4, "K_BLOCKS_PER_WARP=4 for 1 packed int32 / warp");

    constexpr int kBlockSize = 128;                          // 1×128 BSFP8
    constexpr int kIterElems = kBlockSize;                    // 128 K-elem / iter
    int const warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int const lane_id = threadIdx.x & 31;
    int const k_groups = K / (kBlockSize * K_BLOCKS_PER_WARP);
    int const total_warps = M_pad * k_groups;
    if (warp_id >= total_warps) return;

    int const m  = warp_id / k_groups;
    int const kg = warp_id % k_groups;

    int const num_kp = K / kBlockSize;  // words per row in the SM1XX atom layout (one per K-block)

    // Padded M rows: write zero (packed) or skip (fp32).
    if (m >= M) {
        if (lane_id == 0) {
            if constexpr (OUT_FMT == BSFp8ScaleFormat::PACKED_INT32_KMAJOR) {
                static_cast<int32_t*>(out_scale)[kg * M_pad + m] = 0;
            } else if constexpr (OUT_FMT == BSFp8ScaleFormat::PACKED_INT32_SM1XX_ATOM) {
#pragma unroll
                for (int p = 0; p < K_BLOCKS_PER_WARP; ++p) {
                    static_cast<int32_t*>(out_scale)[sf_word_index<MxSfLayout::SM1XX_ATOM>(
                        m, kg * K_BLOCKS_PER_WARP + p, M_pad, num_kp)] = 0;
                }
            } else {
#pragma unroll
                for (int p = 0; p < K_BLOCKS_PER_WARP; ++p) {
                    static_cast<float*>(out_scale)[(kg * K_BLOCKS_PER_WARP + p) * M_pad + m] = 0.f;
                }
            }
        }
        return;
    }

    uint32_t packed = 0;

#pragma unroll
    for (int it = 0; it < K_BLOCKS_PER_WARP; ++it) {
        int const kb = kg * K_BLOCKS_PER_WARP + it;
        int const k_base = kb * kBlockSize;

        // LDG.64: 4 BF16 per lane. 32 lanes × 4 = 128 elements = 1 K-block.
        uint64_t const xword = *reinterpret_cast<uint64_t const*>(
            &input[m * K + k_base + lane_id * 4]);
        __nv_bfloat16 const* xv = reinterpret_cast<__nv_bfloat16 const*>(&xword);
        float const x0 = __bfloat162float(xv[0]);
        float const x1 = __bfloat162float(xv[1]);
        float const x2 = __bfloat162float(xv[2]);
        float const x3 = __bfloat162float(xv[3]);

        float my_ax = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));

        // Full-warp amax reduction (whole warp covers one K-block).
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            my_ax = fmaxf(my_ax, __shfl_xor_sync(0xFFFFFFFFu, my_ax, off));
        }
        my_ax = fmaxf(my_ax, 1e-10f);

        float qs;
        uint8_t byte_v;
        e8m0_from_amax<USE_UE8M0>(my_ax, qs, byte_v);

        // STG.32: 4 FP8 bytes per lane (paired satfinite converts).
        uint32_t const fp_word = fp8x4_from_floats(x0 * qs, x1 * qs, x2 * qs, x3 * qs);
        *reinterpret_cast<uint32_t*>(&out_fp8[m * K + k_base + lane_id * 4]) = fp_word;

        // Emit scale (one per K-block).
        if constexpr (OUT_FMT == BSFp8ScaleFormat::PACKED_INT32_KMAJOR) {
            if (lane_id == 0) {
                packed |= static_cast<uint32_t>(byte_v) << (it * 8);
            }
        } else if constexpr (OUT_FMT == BSFp8ScaleFormat::PACKED_INT32_SM1XX_ATOM) {
            if (lane_id == 0) {
                // 1x128 block scale replicated into the 4 VS=32 slots of this K-block.
                static_cast<int32_t*>(out_scale)[sf_word_index<MxSfLayout::SM1XX_ATOM>(m, kb, M_pad, num_kp)]
                    = static_cast<int32_t>(static_cast<uint32_t>(byte_v) * 0x01010101u);
            }
        } else {
            // FP32 path: write dequant scale at K-major position (kb, m).
            if (lane_id == 0) {
                float dequant_scale;
                if constexpr (USE_UE8M0) {
                    uint32_t const ds_bits = static_cast<uint32_t>(byte_v) << 23;
                    dequant_scale = __uint_as_float(ds_bits);
                } else {
                    dequant_scale = 1.f / qs;
                }
                static_cast<float*>(out_scale)[kb * M_pad + m] = dequant_scale;
            }
        }
    }

    if constexpr (OUT_FMT == BSFp8ScaleFormat::PACKED_INT32_KMAJOR) {
        if (lane_id == 0) {
            // packed scale layout: K-major [pad(M,4), K/512] int32 (4 bytes / int32).
            static_cast<int32_t*>(out_scale)[kg * M_pad + m] = static_cast<int>(packed);
        }
    }
}

} // anonymous namespace

void fp8bs_quantize_1x128_packed(__nv_fp8_e4m3* x_q, int32_t* packed_scales,
    __nv_bfloat16 const* x, int M, int K, cudaStream_t stream, bool use_ue8m0, bool sm1xx_sf_layout)
{
    // sm_120: K-major words, rows padded to 4; sm_100/103: Sm1xx atom layout, rows padded to 128.
    int const m_pad = sm1xx_sf_layout ? ((M + 127) / 128) * 128 : ((M + 3) / 4) * 4;
    constexpr int kBlockSize = 128;
    constexpr int kBlocksPerWarp = 4;
    int const k_groups = K / (kBlockSize * kBlocksPerWarp);
    int const total_warps = m_pad * k_groups;
    constexpr int kThreadsPerBlock = 256;
    constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;
    int const grid = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;

    if (sm1xx_sf_layout) {
        if (use_ue8m0) {
            fp8bs_quantize_1x128_packed_kernel<true, BSFp8ScaleFormat::PACKED_INT32_SM1XX_ATOM>
                <<<grid, kThreadsPerBlock, 0, stream>>>(x_q, packed_scales, x, M, K, m_pad);
        } else {
            fp8bs_quantize_1x128_packed_kernel<false, BSFp8ScaleFormat::PACKED_INT32_SM1XX_ATOM>
                <<<grid, kThreadsPerBlock, 0, stream>>>(x_q, packed_scales, x, M, K, m_pad);
        }
        return;
    }
    if (use_ue8m0) {
        fp8bs_quantize_1x128_packed_kernel<true, BSFp8ScaleFormat::PACKED_INT32_KMAJOR>
            <<<grid, kThreadsPerBlock, 0, stream>>>(x_q, packed_scales, x, M, K, m_pad);
    } else {
        fp8bs_quantize_1x128_packed_kernel<false, BSFp8ScaleFormat::PACKED_INT32_KMAJOR>
            <<<grid, kThreadsPerBlock, 0, stream>>>(x_q, packed_scales, x, M, K, m_pad);
    }
}

void fp8bs_quantize_1x128_fp32_fast(__nv_fp8_e4m3* x_q, float* fp32_scales,
    __nv_bfloat16 const* x, int M, int K, cudaStream_t stream, bool use_ue8m0)
{
    int const m_pad = ((M + 3) / 4) * 4;
    constexpr int kBlockSize = 128;
    constexpr int kBlocksPerWarp = 4;
    int const k_groups = K / (kBlockSize * kBlocksPerWarp);
    int const total_warps = m_pad * k_groups;
    constexpr int kThreadsPerBlock = 256;
    constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;
    int const grid = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;

    if (use_ue8m0) {
        fp8bs_quantize_1x128_packed_kernel<true, BSFp8ScaleFormat::FP32_KMAJOR>
            <<<grid, kThreadsPerBlock, 0, stream>>>(x_q, fp32_scales, x, M, K, m_pad);
    } else {
        fp8bs_quantize_1x128_packed_kernel<false, BSFp8ScaleFormat::FP32_KMAJOR>
            <<<grid, kThreadsPerBlock, 0, stream>>>(x_q, fp32_scales, x, M, K, m_pad);
    }
}

// Repack UE8M0-rounded FP32 dequant scales into the int32-packed layout
// CUTLASS Sm120BlockScaledKernel expects: 4 K-consecutive UE8M0 bytes per
// int32 word, **per-row** along the M (or N) dim (so the SF tensor has
// shape [scale_outer = align(M_or_N, 4), K/512] int32 in K-major).
//
// Two modes:
//   1. SFA (activations): input is already per-row (1 scale per M row,
//      K-major shape [pad(M,4), K/128] from quantize_1x128). Just K-pack.
//   2. SFB (weights): our `quantize_128x128` produces per-128-block
//      scales (1 per (N/128, K/128) tuple). CUTLASS expects per-N-row.
//      We repack with an EXPANSION: each output row n reads from
//      input row (n / 128) and packs 4 K-blocks of that scale.

// Mode 1: per-row pack. src shape [outer_pad, K/128] K-major FP32; out
// [outer_pad, ceil(K/512)]. A K/128 count that is not a multiple of 4 leaves
// the last word's missing bytes zero (never consumed: the sm_120 kernel's K
// loop stops at the real k-tile count, 2026-09-05).
__global__ void repack_ue8m0_per_row_kernel(int32_t* __restrict__ dst,
    float const* __restrict__ src, int outer_pad, int K_blocks_in, int K_blocks_out)
{
    int const o = blockIdx.x * blockDim.x + threadIdx.x;
    int const kp = blockIdx.y * blockDim.y + threadIdx.y;
    if (o >= outer_pad || kp >= K_blocks_out) return;
    int const stride_outer = outer_pad;  // K-major: data[kb * stride + outer]
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int const kb = kp * 4 + i;
        if (kb < K_blocks_in)
        {
            float const s = src[kb * stride_outer + o];
            uint32_t const fbits = __float_as_uint(s);
            uint32_t const e8m0 = (fbits >> 23) & 0xFFu;
            packed |= (e8m0 << (i * 8));
        }
    }
    dst[kp * stride_outer + o] = static_cast<int32_t>(packed);
}

// Mode 2: SFB expand-and-pack. src shape [N/128, K/128] FP32 row-major.
// Out shape [N_pad, K/512] int32 K-major. Out row n reads from
// src[n/128, kp*4..kp*4+3] (the same 128-N-block scale repeats across
// 128 consecutive N rows in the output).
__global__ void repack_ue8m0_expand_n_kernel(int32_t* __restrict__ dst,
    float const* __restrict__ src, int N_pad, int N_blocks_in, int K_blocks_in_per_row)
{
    int const n = blockIdx.x * blockDim.x + threadIdx.x;
    int const kp = blockIdx.y * blockDim.y + threadIdx.y;
    int const K_blocks_out = (K_blocks_in_per_row + 3) / 4;
    if (n >= N_pad || kp >= K_blocks_out) return;
    int const nb = n / 128;
    uint32_t packed = 0;
    if (nb < N_blocks_in)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int const kb = kp * 4 + i;
            if (kb >= K_blocks_in_per_row)
                break;
            // src is row-major: data[nb * K_blocks_in_per_row + kb]
            float const s = src[nb * K_blocks_in_per_row + kb];
            uint32_t const fbits = __float_as_uint(s);
            uint32_t const e8m0 = (fbits >> 23) & 0xFFu;
            packed |= (e8m0 << (i * 8));
        }
    }
    int const stride_n = N_pad;  // K-major: dst[kp * stride_n + n]
    dst[kp * stride_n + n] = static_cast<int32_t>(packed);
}

// ---- sm_100 / sm_103 block-FP8 (1x128 / 128x128) scale repack into the Sm1xx atom layout --------
// One int32 word per (row, K-block) = the UE8M0 exponent byte replicated into the 4 VS=32 slots.
// Rows are padded to 128 (zero words) as the tcgen05 collective's SF tile expects.

// SFA: src is the 1x128 FP32 scale tensor [pad(M,4), K/128] with K-major physical layout
// (data[kb * M_pad4 + m]); dst is the 1-D atom buffer [pad(M,128) * K/128] int32.
__global__ void repack_ue8m0_1x128_atom_kernel(int32_t* __restrict__ dst, float const* __restrict__ src,
    int M, int M_pad4, int M_pad128, int K_blocks)
{
    int const m = blockIdx.x * blockDim.x + threadIdx.x;
    int const kb = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M_pad128 || kb >= K_blocks) return;
    uint32_t word = 0;
    if (m < M)
    {
        uint32_t const fbits = __float_as_uint(src[kb * M_pad4 + m]);
        word = ((fbits >> 23) & 0xFFu) * 0x01010101u;
    }
    dst[sf_word_index<MxSfLayout::SM1XX_ATOM>(m, kb, M_pad128, K_blocks)] = static_cast<int32_t>(word);
}

// SFB: src is the 128x128 FP32 scale tensor [N/128, K/128] row-major; dst is the per-N-row atom
// buffer [pad(N,128) * K/128] int32 (all 128 rows of an N-block share the block's scale).
__global__ void repack_ue8m0_128x128_atom_kernel(int32_t* __restrict__ dst, float const* __restrict__ src,
    int N, int N_blocks, int N_pad128, int K_blocks)
{
    int const n = blockIdx.x * blockDim.x + threadIdx.x;
    int const kb = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= N_pad128 || kb >= K_blocks) return;
    uint32_t word = 0;
    if (n < N && n / 128 < N_blocks)
    {
        uint32_t const fbits = __float_as_uint(src[(n / 128) * K_blocks + kb]);
        word = ((fbits >> 23) & 0xFFu) * 0x01010101u;
    }
    dst[sf_word_index<MxSfLayout::SM1XX_ATOM>(n, kb, N_pad128, K_blocks)] = static_cast<int32_t>(word);
}

void repack_ue8m0_scales_1x128_for_sm1xx(int32_t* dst, float const* src, int M, int M_pad4, int K_blocks,
    cudaStream_t stream)
{
    int const M_pad128 = (M + 127) / 128 * 128;
    dim3 const block(32, 4, 1);
    dim3 const grid((M_pad128 + block.x - 1) / block.x, (K_blocks + block.y - 1) / block.y, 1);
    repack_ue8m0_1x128_atom_kernel<<<grid, block, 0, stream>>>(dst, src, M, M_pad4, M_pad128, K_blocks);
}

void repack_ue8m0_scales_128x128_for_sm1xx(int32_t* dst, float const* src, int N, int N_blocks, int K_blocks,
    cudaStream_t stream)
{
    int const N_pad128 = (N + 127) / 128 * 128;
    dim3 const block(32, 4, 1);
    dim3 const grid((N_pad128 + block.x - 1) / block.x, (K_blocks + block.y - 1) / block.y, 1);
    repack_ue8m0_128x128_atom_kernel<<<grid, block, 0, stream>>>(dst, src, N, N_blocks, N_pad128, K_blocks);
}

void repack_ue8m0_scales_for_sm120(int32_t* dst, float const* src, int M_pad, int K_blocks_in,
    cudaStream_t stream)
{
    int const K_blocks_out = (K_blocks_in + 3) / 4;
    dim3 const block(32, 4, 1);
    dim3 const grid((M_pad + block.x - 1) / block.x,
                    (K_blocks_out + block.y - 1) / block.y, 1);
    repack_ue8m0_per_row_kernel<<<grid, block, 0, stream>>>(dst, src, M_pad, K_blocks_in, K_blocks_out);
}

void repack_ue8m0_scales_sfb_for_sm120(int32_t* dst, float const* src, int N_pad,
    int N_blocks_in, int K_blocks_in_per_row, cudaStream_t stream)
{
    int const K_blocks_out = (K_blocks_in_per_row + 3) / 4;
    dim3 const block(32, 4, 1);
    dim3 const grid((N_pad + block.x - 1) / block.x,
                    (K_blocks_out + block.y - 1) / block.y, 1);
    repack_ue8m0_expand_n_kernel<<<grid, block, 0, stream>>>(dst, src, N_pad, N_blocks_in,
        K_blocks_in_per_row);
}

} // namespace detail
} // namespace blockscale_gemm
