/*
 * The arithmetic of the sm_100 / sm_103 dense epilogue, two elements at a time (issue #12, S3b-opt, 2026-10-29).
 *
 * The CUTLASS sm_100 kernels compute their epilogue in four warps, one per SM sub-partition, so a tile's epilogue
 * costs about one issue slot per SASS instruction per element and the instruction count per element is the lever
 * (ncu, runs/S3bo/ncu: the erf-GELU epilogue issues about 29 instructions per element and runs at the issue rate of
 * one warp per sub-partition). The functions here give the bytes of the shared helpers of
 * common/epilogue_device.cuh (the canonical _rn formula of interface v1) and of fso/activation_contract.h (the
 * activation quantizer) with fewer instructions:
 *
 *   * two elements at a time with the packed fp32 operations of sm_100 (fma.rn.f32x2, add.rn.f32x2, mul.rn.f32x2,
 *     written as PTX so that they are never contracted: dense_fma2 / dense_add2 / dense_mul2 below): one instruction
 *     gives two IEEE round-to-nearest results, each one identical to the scalar __fmaf_rn / __fadd_rn / __fmul_rn of
 *     the same operands, so the fma chain of erff, the bias add, the products of the GELU and the products of the
 *     quantizer cost half the instructions;
 *   * erff written out as the sequence CUDA 13.0's libdevice computes (read from the PTX of a probe kernel,
 *     runs/S3bo/scripts/erf_probe.ptx: one branch-free polynomial with per-element coefficient selects and an
 *     ex2.approx.ftz step for |x| >= 1.00296), so that its fma chain can be packed; dense_erff_pair and
 *     dense_gelu_erf_pair are checked bit for bit against erff and fso::dense_gelu_erf on every fp32 input
 *     (runs/S3bo/erf/erf_exhaustive.cu, run on the B300);
 *   * the UE8M0 byte of a block read off the bits of its amax, with no division (dense_e8m0_from_amax): the
 *     reference computes q = RN(448 / amax), d = RN(1 / q), byte = exp_field(d) + (mantissa(d) != 0) capped at 254,
 *     and qs = RN(1 / 2^(byte - 127)). With amax = (1 + f) 2^(E - 127), 448 / amax = (1.75 / (1 + f)) 2^(135 - E) has
 *     the exponent field 262 - E when f <= 0.75 (mantissa bits <= 0x600000) and 261 - E otherwise, RN(1 / q) then
 *     the exponent field 254 - exp_field(q) with a zero mantissa exactly when q is a power of two, so the byte is
 *     E - 8 or E - 7, and qs = 2^(127 - byte) is exact (2^-127 for byte 254 is the subnormal 0x00400000). Checked
 *     equal to fso_e8m0_from_amax on every non-NaN fp32 pattern floored at 1e-10f (runs/S3bo/e8m0/e8m0_exhaustive.c,
 *     host, and the device check in runs/S3bo/erf/erf_exhaustive.cu).
 *
 * The bias term's -0.0 rule (common/epilogue_device.cuh) is not here: the sm_100 epilogue applies it once per column
 * when it stages the bias into shared memory (dense_epilogue_gemm.cuh, BiasRowBroadcast), not once per element.
 *
 * For the sm_100 / sm_103 kernels; included by dense_epilogue_gemm.cuh and by the exhaustive check. Needs only the CUDA
 * headers, epilogue_device.cuh and fso/activation_contract.h. The sm_100 dense TUs are compiled for every arch of a
 * multi-arch build (the release wheel's 9.0a;10.0f;12.0a: CUTLASS compiles the sm_100 kernel bodies out of the
 * other device passes), so this header must compile in every device pass: the packed PTX is used where it exists
 * (__CUDA_ARCH__ >= 1000, and the host pass, which compiles no device code), and the scalar _rn operations, lane by
 * lane the same bits, in the device passes below sm_100, where nothing calls them.
 */
#pragma once

#include "blockscale_gemm/common/epilogue_device.cuh"
#include "fso/activation_contract.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// 1: the packed fp32 PTX (fma.rn.f32x2 / add.rn.f32x2 / mul.rn.f32x2, sm_100 and newer); 0: two scalar _rn
// operations (the sm_90 device pass of a multi-arch build, which compiles these functions but never calls them).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
#define FSO_SM100_DENSE_PACKED_F32 0
#else
#define FSO_SM100_DENSE_PACKED_F32 1
#endif

namespace fso
{
namespace sm100
{

// The packed fp32 operations as PTX with an explicit rounding modifier (the CUDA intrinsics __ffma2_rn / __fadd2_rn /
// __fmul2_rn lower to the same PTX; the wrappers make the operation and its rounding explicit in the source). One
// caveat holds for both forms: ptxas contracts a mul.rn.f32x2 whose result feeds an add.rn.f32x2 into one
// fma.rn.f32x2 (seen in SASS as an FFMA2 replacing an FMUL2 + FADD2 pair), which rounds once instead of twice; the
// scalar mul.rn.f32 + add.rn.f32 are never contracted. So no packed add below ever takes a packed product as its
// operand (the one place where the formula does that, the residual add after the GELU, uses two scalar adds). The
// probe runs/S3bo/erf/contract_probe.cu and fadd2_probe.cu record the behaviour; add.rn.f32x2 and mul.rn.f32x2 on
// their own give the scalar bits on every fp32 pattern (fadd2_probe.out).
__device__ __forceinline__ unsigned long long dense_pack2(float2 v)
{
    unsigned long long u;
    asm("mov.b64 %0, {%1, %2};" : "=l"(u) : "f"(v.x), "f"(v.y));
    return u;
}

__device__ __forceinline__ float2 dense_unpack2(unsigned long long u)
{
    float2 v;
    asm("mov.b64 {%0, %1}, %2;" : "=f"(v.x), "=f"(v.y) : "l"(u));
    return v;
}

// (a.x + b.x, a.y + b.y), each lane one IEEE round-to-nearest fp32 addition, never contracted.
__device__ __forceinline__ float2 dense_add2(float2 a, float2 b)
{
#if FSO_SM100_DENSE_PACKED_F32
    unsigned long long d;
    asm("add.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(dense_pack2(a)), "l"(dense_pack2(b)));
    return dense_unpack2(d);
#else
    return make_float2(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y));
#endif
}

// (a.x * b.x, a.y * b.y), one rounding per lane, never contracted.
__device__ __forceinline__ float2 dense_mul2(float2 a, float2 b)
{
#if FSO_SM100_DENSE_PACKED_F32
    unsigned long long d;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(dense_pack2(a)), "l"(dense_pack2(b)));
    return dense_unpack2(d);
#else
    return make_float2(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y));
#endif
}

// (fma(a.x, b.x, c.x), fma(a.y, b.y, c.y)), one rounding per lane.
__device__ __forceinline__ float2 dense_fma2(float2 a, float2 b, float2 c)
{
#if FSO_SM100_DENSE_PACKED_F32
    unsigned long long d;
    asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(dense_pack2(a)), "l"(dense_pack2(b)), "l"(dense_pack2(c)));
    return dense_unpack2(d);
#else
    return make_float2(__fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y));
#endif
}

// ex2.approx.ftz.f32: the exp2 of the large-argument path of libdevice's erff.
__device__ __forceinline__ float dense_ex2_approx_ftz(float x)
{
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// erff of two values, the libdevice __nv_erff sequence of CUDA 13.0 lane by lane:
//   t = |a|; large = t >= 1.0029599667f; x = large ? t : RN(a * a);
//   r = fma(c0, x, c1); r = fma(r, x, c2); ... r = fma(r, x, c6)   (one coefficient set per path);
//   s = large ? -x : a; r = fma(r, s, s);
//   large: r = copysign(RN(1 - ex2.approx.ftz(r)), a).
// Both paths are evaluated for both lanes with constant coefficients (immediate operands of the packed fma), and
// the lane's path is selected at the end: a lane's result is exactly the sequence libdevice runs for it, the other
// path's value is dropped. This costs 8 extra packed fmas per pair but saves the 18 per-lane coefficient and operand
// selects that a shared chain needs (the predicate differs per lane), which the four epilogue warps issue one by one.
__device__ __forceinline__ float2 dense_erff_pair(float2 a)
{
    constexpr float kLarge = 1.0029599666595458984f; // 0x3F8060FE
    auto const k = [](uint32_t u) { return __uint_as_float(u); };
    auto const k2 = [&](uint32_t u) { return make_float2(__uint_as_float(u), __uint_as_float(u)); };
    float2 const t = make_float2(fabsf(a.x), fabsf(a.y));
    bool const lx = t.x >= kLarge, ly = t.y >= kLarge;
    // the small path: x = a * a
    float2 const xs = dense_mul2(a, a);
    float2 rs = dense_fma2(k2(0x38B1E96Au), xs, k2(0xBA574D20u));
    rs = dense_fma2(rs, xs, k2(0x3BAAD5EAu));
    rs = dense_fma2(rs, xs, k2(0xBCDC1BE7u));
    rs = dense_fma2(rs, xs, k2(0x3DE718AFu));
    rs = dense_fma2(rs, xs, k2(0xBEC093ACu));
    rs = dense_fma2(rs, xs, k2(0x3E0375D3u));
    rs = dense_fma2(rs, a, a);
    // the large path: x = |a|, s = -|a|
    float2 rl = dense_fma2(k2(0x38EB4C3Au), t, k2(0xBAAE005Bu));
    rl = dense_fma2(rl, t, k2(0x3C09919Fu));
    rl = dense_fma2(rl, t, k2(0xBD24D99Au));
    rl = dense_fma2(rl, t, k2(0x3E235519u));
    rl = dense_fma2(rl, t, k2(0x3F69B4F9u));
    rl = dense_fma2(rl, t, k2(0x3F210A14u));
    float2 const nt = make_float2(-t.x, -t.y);
    rl = dense_fma2(rl, nt, nt);
    float2 const e = make_float2(dense_ex2_approx_ftz(rl.x), dense_ex2_approx_ftz(rl.y));
    float2 const one_minus_e = dense_add2(make_float2(1.f, 1.f), make_float2(-e.x, -e.y));
    (void) k;
    return make_float2(lx ? copysignf(one_minus_e.x, a.x) : rs.x, ly ? copysignf(one_minus_e.y, a.y) : rs.y);
}

// erf GELU of two values: 0.5f * v * (1.f + erff(v * kDenseSqrt1_2)) with every operation rounded on its own, lane
// by lane the dense_gelu_erf of common/epilogue_device.cuh.
__device__ __forceinline__ float2 dense_gelu_erf_pair(float2 v)
{
    float2 const half = make_float2(0.5f, 0.5f);
    float2 const one = make_float2(1.f, 1.f);
    float2 const k = make_float2(kDenseSqrt1_2, kDenseSqrt1_2);
    return dense_mul2(dense_mul2(half, v), dense_add2(one, dense_erff_pair(dense_mul2(v, k))));
}

// v = acc + bias_term; gelu if act; [v += residual], two columns at a time: lane by lane the
// dense_epilogue_value_rt<kResidual>(acc, bias_n, act, residual) of common/epilogue_device.cuh, with bias_term
// already the -0.0-ruled bias of the column (the staging applies the rule).
template <bool kResidual>
__device__ __forceinline__ float2 dense_value_pair(float2 acc, float2 bias_term, int act, float2 residual)
{
    float2 v = dense_add2(acc, bias_term);
    if (act == kDenseActGeluErf)
        v = dense_gelu_erf_pair(v);
    if constexpr (kResidual)
    {
        // Two scalar adds, not a packed one: ptxas contracts a mul.rn.f32x2 whose result feeds an add.rn.f32x2 into
        // one fma.rn.f32x2 even with the explicit rounding modifiers (checked on CUDA 13.0: the PTX keeps mul.rn +
        // add.rn, the SASS shows one FFMA2; the scalar mul.rn.f32 + add.rn.f32 stay two instructions), and the
        // GELU's last operation is a packed multiply. One rounding instead of two changed 1-120 elements per cell of
        // test_dense_v1 case p in the bias + GELU + residual form; the scalar add.rn.f32 is never contracted.
        v = make_float2(__fadd_rn(v.x, residual.x), __fadd_rn(v.y, residual.y));
    }
    return v;
}

// The UE8M0 byte of a block from its amax (already floored at fso_mx_amax_floor(), never NaN) and the factor its
// elements are multiplied by, 2^(127 - byte): fso_e8m0_from_amax of fso/activation_contract.h without its three
// divisions (see the header comment for the derivation and the exhaustive check).
__device__ __forceinline__ uint32_t dense_e8m0_from_amax(float amax, float* quant_scale)
{
    uint32_t const bits = __float_as_uint(amax);
    uint32_t const e = (bits >> 23) & 0xFFu;
    uint32_t const f = bits & 0x7FFFFFu;
    int byte = e == 255u ? 254 : static_cast<int>(e) - (f <= 0x600000u ? 8 : 7);
    byte = byte < 0 ? 0 : (byte > 254 ? 254 : byte);
    *quant_scale = byte == 0 ? 1.f
                             : (byte == 254 ? __uint_as_float(0x00400000u)
                                            : __uint_as_float(static_cast<uint32_t>(254 - byte) << 23));
    return static_cast<uint32_t>(byte);
}

// Four elements of a block, y * quant_scale each, as one word of e4m3 codes: fso_mx_quantize4 with the products two
// at a time.
__device__ __forceinline__ uint32_t dense_quantize4(float2 y01, float2 y23, float2 quant_scale2)
{
    float2 const p01 = dense_mul2(y01, quant_scale2);
    float2 const p23 = dense_mul2(y23, quant_scale2);
    return fso_fp8x4_from_floats(p01.x, p01.y, p23.x, p23.y);
}

} // namespace sm100
} // namespace fso
