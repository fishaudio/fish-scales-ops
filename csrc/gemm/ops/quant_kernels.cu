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
    __nv_fp8_e4m3* w_q, float* scales, __nv_bfloat16 const* w, int N, int K, cudaStream_t stream)
{
    // The runner-internal weight quant path. NOT fp8_128x128_cs, which is a
    // cast-only placeholder that fills scales with 1.0.
    tensorrt_llm::kernels::blockscale_gemm::fp8_128x128_quant(w_q, scales, w, K, N, stream);
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
template <bool USE_UE8M0, int K_BLOCKS_PER_WARP>
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

    if (m >= M) {
        if (lane_id == 0) {
#pragma unroll
            for (int p = 0; p < K_BLOCKS_PER_WARP / 4; ++p) {
                out_packed[(kp_base + p) * M_pad + m] = 0;
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
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            my_ax = fmaxf(my_ax, __shfl_xor_sync(0xFFFFFFFFu, my_ax, off));
        }
        my_ax = fmaxf(my_ax, 1e-10f);

        float qs;
        uint8_t byte_v;
        e8m0_from_amax<USE_UE8M0>(my_ax, qs, byte_v);

        // STG.32: 4 FP8 bytes per lane.
        float const v0 = fmaxf(-448.f, fminf(448.f, x0 * qs));
        float const v1 = fmaxf(-448.f, fminf(448.f, x1 * qs));
        float const v2 = fmaxf(-448.f, fminf(448.f, x2 * qs));
        float const v3 = fmaxf(-448.f, fminf(448.f, x3 * qs));
        uint32_t fp_word;
        reinterpret_cast<uint8_t*>(&fp_word)[0] = __nv_fp8_e4m3(v0).__x;
        reinterpret_cast<uint8_t*>(&fp_word)[1] = __nv_fp8_e4m3(v1).__x;
        reinterpret_cast<uint8_t*>(&fp_word)[2] = __nv_fp8_e4m3(v2).__x;
        reinterpret_cast<uint8_t*>(&fp_word)[3] = __nv_fp8_e4m3(v3).__x;
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
            out_packed[(kp_base + p) * M_pad + m] = packed_words[p];
        }
    }
}

} // anonymous namespace

void fp8bs_quantize_1x32_packed(__nv_fp8_e4m3* x_q, int32_t* packed_scales,
    __nv_bfloat16 const* x, int M, int K, cudaStream_t stream, bool use_ue8m0)
{
    // K_BLOCKS_PER_WARP = 8 (256 K-elements/warp, 2 int32 outputs/warp) when
    // K is a multiple of 256 — fewer CTAs, better amortisation of launch
    // overhead. Falls back to 4 when K % 256 != 0 (e.g. shapes where K is
    // only a multiple of 128).
    int const m_pad = ((M + 3) / 4) * 4;
    constexpr int kThreadsPerBlock = 256;
    constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;

    auto launch = [&](auto kBlocksPerWarpT) {
        constexpr int kBlocksPerWarp = decltype(kBlocksPerWarpT)::value;
        int const k_groups = K / (kMxFp8VecSize * kBlocksPerWarp);
        int const total_warps = m_pad * k_groups;
        int const grid = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
        if (use_ue8m0)
            fp8bs_quantize_1x32_packed_kernel<true, kBlocksPerWarp>
                <<<grid, kThreadsPerBlock, 0, stream>>>(x_q, packed_scales, x, M, K, m_pad);
        else
            fp8bs_quantize_1x32_packed_kernel<false, kBlocksPerWarp>
                <<<grid, kThreadsPerBlock, 0, stream>>>(x_q, packed_scales, x, M, K, m_pad);
    };

    if (K % 256 == 0) launch(std::integral_constant<int, 8>{});
    else              launch(std::integral_constant<int, 4>{});
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

template <bool USE_UE8M0, int K_BLOCKS_PER_WARP>
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
    int const stride_m_gu = 2 * K;  // gu has 2*K cols (gate then up)

    if (m >= M) {
        if (lane_id == 0) {
#pragma unroll
            for (int p = 0; p < K_BLOCKS_PER_WARP / 4; ++p) {
                out_packed[(kp_base + p) * M_pad + m] = 0;
            }
        }
        return;
    }

    int packed_words[K_BLOCKS_PER_WARP / 4] = {};

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
        __nv_bfloat16 const* gv = reinterpret_cast<__nv_bfloat16 const*>(&gate_word);
        __nv_bfloat16 const* uv = reinterpret_cast<__nv_bfloat16 const*>(&up_word);

        // h_i = silu(gate_i) * up_i = gate_i / (1 + exp(-gate_i)) * up_i
        float const g0 = __bfloat162float(gv[0]);
        float const g1 = __bfloat162float(gv[1]);
        float const g2 = __bfloat162float(gv[2]);
        float const g3 = __bfloat162float(gv[3]);
        float const u0 = __bfloat162float(uv[0]);
        float const u1 = __bfloat162float(uv[1]);
        float const u2 = __bfloat162float(uv[2]);
        float const u3 = __bfloat162float(uv[3]);
        // silu via fast __expf (good enough for activation, bf16 input limits precision anyway)
        float const h0 = g0 * (1.f / (1.f + __expf(-g0))) * u0;
        float const h1 = g1 * (1.f / (1.f + __expf(-g1))) * u1;
        float const h2 = g2 * (1.f / (1.f + __expf(-g2))) * u2;
        float const h3 = g3 * (1.f / (1.f + __expf(-g3))) * u3;

        float my_ax = fmaxf(fmaxf(fabsf(h0), fabsf(h1)), fmaxf(fabsf(h2), fabsf(h3)));

#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            my_ax = fmaxf(my_ax, __shfl_xor_sync(0xFFFFFFFFu, my_ax, off));
        }
        my_ax = fmaxf(my_ax, 1e-10f);

        float qs;
        uint8_t byte_v;
        e8m0_from_amax<USE_UE8M0>(my_ax, qs, byte_v);

        float const v0 = fmaxf(-448.f, fminf(448.f, h0 * qs));
        float const v1 = fmaxf(-448.f, fminf(448.f, h1 * qs));
        float const v2 = fmaxf(-448.f, fminf(448.f, h2 * qs));
        float const v3 = fmaxf(-448.f, fminf(448.f, h3 * qs));
        uint32_t fp_word;
        reinterpret_cast<uint8_t*>(&fp_word)[0] = __nv_fp8_e4m3(v0).__x;
        reinterpret_cast<uint8_t*>(&fp_word)[1] = __nv_fp8_e4m3(v1).__x;
        reinterpret_cast<uint8_t*>(&fp_word)[2] = __nv_fp8_e4m3(v2).__x;
        reinterpret_cast<uint8_t*>(&fp_word)[3] = __nv_fp8_e4m3(v3).__x;
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
            out_packed[(kp_base + p) * M_pad + m] = packed_words[p];
        }
    }
}

} // anonymous namespace

void fp8bs_silu_chunk_mul_quantize_1x32_packed(
    __nv_fp8_e4m3* x_q, int32_t* packed_scales,
    __nv_bfloat16 const* gu, int M, int K, cudaStream_t stream, bool use_ue8m0)
{
    int const m_pad = ((M + 3) / 4) * 4;
    constexpr int kThreadsPerBlock = 256;
    constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;

    auto launch = [&](auto kBlocksPerWarpT) {
        constexpr int kBlocksPerWarp = decltype(kBlocksPerWarpT)::value;
        int const k_groups = K / (kMxFp8VecSize * kBlocksPerWarp);
        int const total_warps = m_pad * k_groups;
        int const grid = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
        if (use_ue8m0)
            silu_chunk_mul_quantize_1x32_packed_kernel<true, kBlocksPerWarp>
                <<<grid, kThreadsPerBlock, 0, stream>>>(x_q, packed_scales, gu, M, K, m_pad);
        else
            silu_chunk_mul_quantize_1x32_packed_kernel<false, kBlocksPerWarp>
                <<<grid, kThreadsPerBlock, 0, stream>>>(x_q, packed_scales, gu, M, K, m_pad);
    };

    if (K % 256 == 0) launch(std::integral_constant<int, 8>{});
    else              launch(std::integral_constant<int, 4>{});
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

enum class BSFp8ScaleFormat { FP32_KMAJOR, PACKED_INT32_KMAJOR };

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

    // Padded M rows: write zero (packed) or skip (fp32).
    if (m >= M) {
        if (lane_id == 0) {
            if constexpr (OUT_FMT == BSFp8ScaleFormat::PACKED_INT32_KMAJOR) {
                static_cast<int32_t*>(out_scale)[kg * M_pad + m] = 0;
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

        // STG.32: 4 FP8 bytes per lane.
        float const v0 = fmaxf(-448.f, fminf(448.f, x0 * qs));
        float const v1 = fmaxf(-448.f, fminf(448.f, x1 * qs));
        float const v2 = fmaxf(-448.f, fminf(448.f, x2 * qs));
        float const v3 = fmaxf(-448.f, fminf(448.f, x3 * qs));
        uint32_t fp_word;
        reinterpret_cast<uint8_t*>(&fp_word)[0] = __nv_fp8_e4m3(v0).__x;
        reinterpret_cast<uint8_t*>(&fp_word)[1] = __nv_fp8_e4m3(v1).__x;
        reinterpret_cast<uint8_t*>(&fp_word)[2] = __nv_fp8_e4m3(v2).__x;
        reinterpret_cast<uint8_t*>(&fp_word)[3] = __nv_fp8_e4m3(v3).__x;
        *reinterpret_cast<uint32_t*>(&out_fp8[m * K + k_base + lane_id * 4]) = fp_word;

        // Emit scale (one per K-block).
        if constexpr (OUT_FMT == BSFp8ScaleFormat::PACKED_INT32_KMAJOR) {
            if (lane_id == 0) {
                packed |= static_cast<uint32_t>(byte_v) << (it * 8);
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

// Mode 1: per-row pack. src shape [outer_pad, K/128] K-major FP32.
__global__ void repack_ue8m0_per_row_kernel(int32_t* __restrict__ dst,
    float const* __restrict__ src, int outer_pad, int K_blocks_out)
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
        float const s = src[kb * stride_outer + o];
        uint32_t const fbits = __float_as_uint(s);
        uint32_t const e8m0 = (fbits >> 23) & 0xFFu;
        packed |= (e8m0 << (i * 8));
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
    int const K_blocks_out = K_blocks_in_per_row / 4;
    if (n >= N_pad || kp >= K_blocks_out) return;
    int const nb = n / 128;
    uint32_t packed = 0;
    if (nb < N_blocks_in)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int const kb = kp * 4 + i;
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

void repack_ue8m0_scales_for_sm120(int32_t* dst, float const* src, int M_pad, int K_blocks_in,
    cudaStream_t stream)
{
    int const K_blocks_out = K_blocks_in / 4;
    dim3 const block(32, 4, 1);
    dim3 const grid((M_pad + block.x - 1) / block.x,
                    (K_blocks_out + block.y - 1) / block.y, 1);
    repack_ue8m0_per_row_kernel<<<grid, block, 0, stream>>>(dst, src, M_pad, K_blocks_out);
}

void repack_ue8m0_scales_sfb_for_sm120(int32_t* dst, float const* src, int N_pad,
    int N_blocks_in, int K_blocks_in_per_row, cudaStream_t stream)
{
    int const K_blocks_out = K_blocks_in_per_row / 4;
    dim3 const block(32, 4, 1);
    dim3 const grid((N_pad + block.x - 1) / block.x,
                    (K_blocks_out + block.y - 1) / block.y, 1);
    repack_ue8m0_expand_n_kernel<<<grid, block, 0, stream>>>(dst, src, N_pad, N_blocks_in,
        K_blocks_in_per_row);
}

} // namespace detail
} // namespace blockscale_gemm
