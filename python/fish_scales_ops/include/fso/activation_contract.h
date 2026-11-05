/*
 * fish_scales_ops activation contract, version 1.
 *
 * The layout and the rounding rule of the pre-quantized activation that fso.dense.linear(xq, weight, x_scale=sx)
 * accepts and fso.dense.quantize_activation(x) produces, stated so that a caller can write xq and sx from its own
 * kernel (a fused norm + quantize, say) and get exactly the bytes quantize_activation would give. The functions are
 * __host__ __device__ and reproduce the library's quantizers bit for bit: the 1x32 MXFP8 rule of
 * csrc/gemm/ops/quant_kernels.cu (e8m0_from_amax<true> and fp8x4_from_floats), and the 1x128 block-FP8 rule of the
 * sm_90 activation quantize that linear_qx runs inside the op (scale_1x128_lanes_kernel<kBf16AmaxFloor = true> of
 * csrc/gemm/include/blockscale_gemm/common/scale_kernels.cuh, which the raw op quantize_1x128_contract behind
 * quantize_activation runs with the padded rows zeroed). fso.dense.include_dir() is the directory that holds this
 * file (fso/activation_contract.h). It needs only the CUDA headers.
 *
 * The contract a device follows is fso.dense.activation_contract()["format"]:
 *
 * MXFP8 1x32 ("mxfp8_1x32"; sm_100 / sm_103 and sm_120 / sm_121)
 *   xq:      float8_e4m3fn [M, K], row-major, contiguous, 16-byte aligned; K % 128 == 0.
 *   x_scale: int32 words. word(m, kp) = b0 | b1 << 8 | b2 << 16 | b3 << 24, where b_i is the UE8M0 byte of the
 *            32-element block 4 * kp + i of row m (kp in [0, K/128)); byte b means the scale 2^(b - 127).
 *            sm_100 / sm_103: 1-D, align(M, 128) * (K/128) words, word(m, kp) at fso_mx_sf_word_index_sm100(m, kp, K)
 *              (the CUTLASS Sm1xxBlockScaledConfig atom layout).
 *            sm_120 / sm_121: [align(M, 4), K/128] with strides (1, align(M, 4)) (M fastest), word(m, kp) at
 *              fso_mx_sf_word_index_sm120(m, kp, align(M, 4)) = kp * align(M, 4) + m; x_scale[m, kp] in torch
 *              indexing is word(m, kp).
 *            The words of the padded rows (M <= m < the padded M) are zero.
 *   Rule, per 32-element block of a row: amax = fp32 max |x| over the block, floored at fso_mx_amax_floor() (1e-10f);
 *            b = fso_e8m0_from_amax(amax, &qs), which computes q = 448.f / amax, d = 1.f / q,
 *            b = exp_field(d) + (mantissa(d) != 0) capped at 254, and qs = 2^(127 - b);
 *            code(x) = cvt.rn.satfinite.e4m3(x * qs), four at a time with fso_mx_quantize4.
 *
 * Block-FP8 1x128 ("bsfp8_1x128"; sm_90)
 *   xq:      float8_e4m3fn [M, K], row-major, contiguous, 16-byte aligned; K % 128 == 0.
 *   x_scale: fp32, a torch tensor of shape [align(M, 4), K/128] with strides (1, align(M, 4)) (M fastest): the
 *            scale of row m, 128-element group kb is element fso_bsfp8_scale_index(m, kb, align(M, 4)) =
 *            kb * align(M, 4) + m of the buffer, and x_scale[m, kb] in torch indexing is that scale.
 *            The scales of the padded rows (M <= m < align(M, 4)) are 0.f. The GEMM never reads a padded row into
 *            an output row.
 *   Rule, per 128-element group of a row: amax = fp32 max |x| over the group, floored at fso_bsfp8_amax_floor(),
 *            the bf16 value nearest to 1e-10f, 1.00044e-10 (fp32 bits 0x2EDC0000), for every K;
 *            qs = fso_bsfp8_quant_scale(amax, &scale) = 448.f / amax and scale = 1.f / qs (not a power of two);
 *            code(x) = cvt.rn.satfinite.e4m3(x * qs), four at a time with fso_bsfp8_quantize4.
 *            This is the quantize of linear_qx (and so of fso.dense.linear(x, w) with a bf16 x). The public
 *            fso.compat.quantize_1x128_fp8 floors at 1e-10f instead when K % 512 == 0 (and leaves the padded rows
 *            unwritten when K % 512 != 0), so it differs from this contract on a group whose amax is below
 *            1.00044e-10; it is not the contract's quantizer.
 *
 * Common to both: x is the bf16 activation converted exactly to fp32; amax is an exact maximum (any order); every
 * division and product is a single IEEE fp32 operation rounded to nearest (on the device these functions use
 * __fdiv_rn / __fmul_rn, so --use_fast_math does not change them); NaN inputs are outside the contract.
 *
 * The version changes whenever the layout or the rule does; check kFsoActivationContractVersion against
 * fso.dense.activation_contract()["version"] at load time.
 */
#pragma once

#include <cstdint>
#include <cstring>

#include <cuda_fp8.h>

#if defined(__CUDACC__)
#define FSO_AC_HD __host__ __device__ __forceinline__
#else
#define FSO_AC_HD inline
#endif

constexpr int kFsoActivationContractVersion = 1;

// ------------------------------------------------------------------------------------------------- scalar helpers

FSO_AC_HD uint32_t fso_float_as_uint(float f)
{
#if defined(__CUDA_ARCH__)
    return __float_as_uint(f);
#else
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
#endif
}

FSO_AC_HD float fso_uint_as_float(uint32_t u)
{
#if defined(__CUDA_ARCH__)
    return __uint_as_float(u);
#else
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
#endif
}

// a / b and a * b as single fp32 operations rounded to nearest, whatever the device compile flags.
FSO_AC_HD float fso_div_rn(float a, float b)
{
#if defined(__CUDA_ARCH__)
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}

FSO_AC_HD float fso_mul_rn(float a, float b)
{
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}

FSO_AC_HD int64_t fso_align(int64_t v, int64_t a)
{
    return (v + a - 1) / a * a;
}

// Four fp32 values -> four e4m3 codes in one word (lowest byte first), cvt.rn.satfinite.e4m3x2.f32 twice: the
// fp8x4_from_floats of quant_kernels.cu. Values beyond +-448 saturate to +-448.
FSO_AC_HD uint32_t fso_fp8x4_from_floats(float v0, float v1, float v2, float v3)
{
    __nv_fp8x2_storage_t const lo = __nv_cvt_float2_to_fp8x2(make_float2(v0, v1), __NV_SATFINITE, __NV_E4M3);
    __nv_fp8x2_storage_t const hi = __nv_cvt_float2_to_fp8x2(make_float2(v2, v3), __NV_SATFINITE, __NV_E4M3);
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

// ------------------------------------------------------------------------------------------------- MXFP8 1x32

// The floor of a block's amax.
FSO_AC_HD float fso_mx_amax_floor()
{
    return 1e-10f;
}

// The UE8M0 byte of a 32-element block whose amax (already floored at fso_mx_amax_floor()) is `amax`, and in
// *quant_scale the factor its elements are multiplied by before the conversion, 2^(127 - byte): the
// e8m0_from_amax<true> of quant_kernels.cu (q = 448.f / amax, d = 1.f / q, byte = exp_field(d) + (mantissa(d) != 0)
// capped at 254; the descale is derived through q and not as amax / 448, which differs by an ulp at powers of two).
FSO_AC_HD uint8_t fso_e8m0_from_amax(float amax, float* quant_scale)
{
    float const q_raw = fso_div_rn(448.f, amax);
    float const d_raw = fso_div_rn(1.f, q_raw);
    uint32_t const bits = fso_float_as_uint(d_raw);
    uint32_t const exp_field = (bits >> 23) & 0xFFu;
    uint32_t const has_mantissa = (bits & 0x7FFFFFu) != 0u ? 1u : 0u;
    uint32_t byte = exp_field + has_mantissa;
    byte = byte > 254u ? 254u : byte;
    float const dequant_scale = fso_uint_as_float(byte << 23);
    *quant_scale = byte != 0u ? fso_div_rn(1.f, dequant_scale) : 1.f;
    return static_cast<uint8_t>(byte);
}

// Four elements of a block, x * quant_scale each, as one word of e4m3 codes.
FSO_AC_HD uint32_t fso_mx_quantize4(float x0, float x1, float x2, float x3, float quant_scale)
{
    return fso_fp8x4_from_floats(fso_mul_rn(x0, quant_scale), fso_mul_rn(x1, quant_scale),
        fso_mul_rn(x2, quant_scale), fso_mul_rn(x3, quant_scale));
}

// The scale word of four consecutive blocks: block 4 * kp + i in byte i.
FSO_AC_HD uint32_t fso_mx_pack_word(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3)
{
    return static_cast<uint32_t>(b0) | (static_cast<uint32_t>(b1) << 8) | (static_cast<uint32_t>(b2) << 16)
        | (static_cast<uint32_t>(b3) << 24);
}

// sm_100 / sm_103: index of word(m, kp) in the 1-D atom-layout buffer of an [M, k] activation.
FSO_AC_HD int64_t fso_mx_sf_word_index_sm100(int64_t m, int64_t kp, int64_t k)
{
    int64_t const num_kp = k / 128;
    int64_t const r = m % 128;
    return ((m / 128) * num_kp + kp) * 128 + (r % 32) * 4 + r / 32;
}

// sm_100 / sm_103: words in the buffer (padded rows included).
FSO_AC_HD int64_t fso_mx_sf_words_sm100(int64_t m, int64_t k)
{
    return fso_align(m, 128) * (k / 128);
}

// sm_120 / sm_121: index of word(m, kp) in the M-fastest buffer; m_pad = align(M, 4).
FSO_AC_HD int64_t fso_mx_sf_word_index_sm120(int64_t m, int64_t kp, int64_t m_pad)
{
    return kp * m_pad + m;
}

// sm_120 / sm_121: words in the buffer (padded rows included).
FSO_AC_HD int64_t fso_mx_sf_words_sm120(int64_t m, int64_t k)
{
    return fso_align(m, 4) * (k / 128);
}

// ------------------------------------------------------------------------------------------------- block-FP8 1x128 (sm_90)

// The floor of a 128-element group's amax: the bf16 value nearest to 1e-10f (1.00044e-10, fp32 bits 0x2EDC0000), for
// every K.
FSO_AC_HD float fso_bsfp8_amax_floor()
{
    return fso_uint_as_float(0x2EDC0000u);
}

// The factor a group's elements are multiplied by, qs = 448.f / amax (amax already floored), and in *scale the
// fp32 dequant scale stored for the group, 1.f / qs.
FSO_AC_HD float fso_bsfp8_quant_scale(float amax, float* scale)
{
    float const qs = fso_div_rn(448.f, amax);
    *scale = fso_div_rn(1.f, qs);
    return qs;
}

// Four elements of a group, x * qs each, as one word of e4m3 codes.
FSO_AC_HD uint32_t fso_bsfp8_quantize4(float x0, float x1, float x2, float x3, float qs)
{
    return fso_fp8x4_from_floats(fso_mul_rn(x0, qs), fso_mul_rn(x1, qs), fso_mul_rn(x2, qs), fso_mul_rn(x3, qs));
}

// Element index of the scale of row m, group kb in the fp32 buffer; m_pad = align(M, 4).
FSO_AC_HD int64_t fso_bsfp8_scale_index(int64_t m, int64_t kb, int64_t m_pad)
{
    return kb * m_pad + m;
}

// fp32 elements in the buffer (padded rows included; their scales are 0.f).
FSO_AC_HD int64_t fso_bsfp8_scales(int64_t m, int64_t k)
{
    return fso_align(m, 4) * (k / 128);
}

#undef FSO_AC_HD
