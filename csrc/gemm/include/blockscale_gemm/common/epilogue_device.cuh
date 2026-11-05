/*
 * The dense epilogue of interface v1 (issue #12; charter section 7.4): the one POD struct every dense route receives
 * and the device helpers that apply it, in the fixed order
 *
 *     v = acc;  v += bias_term(bias[n]);  v = act(v);  [v += residual[m, n]];  out = RN_bf16(v)  [-> quantize]
 *
 * This header is the single definition of the formula for every architecture. It is included by
 * common/epilogue.cuh (the AOT translation units: the ATen ops, the CUTLASS routes of sm_100 / sm_103 and sm_120 /
 * sm_121) and by the sm_90 kernels that NVRTC compiles at run time (arch/sm90/fp8/jit/deep_gemm), which find it
 * through the JIT include path as <blockscale_gemm/common/epilogue_device.cuh>. So it must stay NVRTC-safe: it
 * includes nothing but the CUDA scalar types (and <cstdint> outside NVRTC, where the JIT programs' own prelude
 * provides the fixed-width integer types), and it holds no host-only code (no <string>, no exceptions). The host
 * side of the contract (the routes' capability masks, the refusal) is in common/epilogue.cuh.
 *
 * Semantics (fso.dense.linear's contract):
 *   * bias_term(b) = float(b), except that +0.0 becomes -0.0, so that a zero bias is the identity on every
 *     accumulator value (-0.0 + -0.0 = -0.0, +0.0 + -0.0 = +0.0) and bias = 0 gives the bytes of bias = None.
 *   * act is the identity or erf-GELU in fp32, 0.5f * v * (1.f + erff(v * kDenseSqrt1_2)), with the constant rounded
 *     to fp32 first (torch's own erf GELU multiplies by float(M_SQRT1_2) in fp32).
 *   * the residual is added only when given (no + 0.0).
 *   * each term is one IEEE fp32 operation rounded to nearest, applied once, and every one of them is written as an
 *     _rn intrinsic (__fadd_rn / __fmul_rn), which the compiler never contracts: nvcc and NVRTC both contract
 *     a * b + c into an fma by default, which would fuse the GELU's last product with the residual addition and
 *     skip the fp32 rounding of act(v) before + r. The caller rounds the result once to bf16. (This _rn form is the
 *     canonical text for every architecture, the main agent's note of 2026-10-28; S1c wrote it for sm_120.)
 */
#pragma once

#if !defined(__CUDACC_RTC__)
#include <cstdint>
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Namespace fso (not blockscale_gemm): the dispatch headers live in tensorrt_llm::kernels::blockscale_gemm and in
// sm100_blockscaled_gemm, where an unqualified blockscale_gemm:: would name the enclosing namespace.
namespace fso
{

// The activation applied by the epilogue: the `act` argument of the raw ops and the `act` field below.
enum DenseActivation : int
{
    kDenseActNone = 0,
    kDenseActGeluErf = 1, // 0.5f * v * (1.f + erff(v * kDenseSqrt1_2)), fp32
};

// Layout of out_scale when the epilogue writes the quantized form (fso/activation_contract.h, version 1).
enum DenseOutScaleLayout : int
{
    kDenseOutScaleNone = 0,
    kDenseOutScaleMxSm100 = 1,   // MXFP8 1x32, int32 words in the Sm1xx atom layout (sm_100 / sm_103)
    kDenseOutScaleMxSm120 = 2,   // MXFP8 1x32, int32 words K-major [align(M, 4), K/128] (sm_120 / sm_121)
    kDenseOutScaleBsfp8Sm90 = 3, // block-FP8 1x128, fp32 K-major [align(M, 4), K/128] (sm_90)
};

// The fields of a DenseEpilogue, as bits, for the capability masks of the routes.
enum DenseEpilogueField : unsigned
{
    kDenseEpiBias = 1u << 0,
    kDenseEpiResidual = 1u << 1,
    kDenseEpiAct = 1u << 2,
    kDenseEpiQuantOut = 1u << 3,
};

// fp32 value of M_SQRT1_2, the constant of the erf GELU.
constexpr float kDenseSqrt1_2 = 0.70710678118654752440f;

struct DenseEpilogue
{
    __nv_bfloat16 const* bias = nullptr;     // [N], or nullptr
    __nv_bfloat16 const* residual = nullptr; // [M, ld_residual] row-major, or nullptr
    int64_t ld_residual = 0;                 // elements between rows of residual
    int act = kDenseActNone;                 // DenseActivation
    __nv_fp8_e4m3* out_q = nullptr;          // quantized output [M, N] (linear_quantized), or nullptr
    void* out_scale = nullptr;               // its scale, in out_scale_layout
    int out_scale_layout = kDenseOutScaleNone;

    __host__ __device__ constexpr unsigned fields() const
    {
        return (bias != nullptr ? unsigned(kDenseEpiBias) : 0u) | (residual != nullptr ? unsigned(kDenseEpiResidual) : 0u)
            | (act != kDenseActNone ? unsigned(kDenseEpiAct) : 0u) | (out_q != nullptr ? unsigned(kDenseEpiQuantOut) : 0u);
    }

    __host__ __device__ constexpr bool is_identity() const
    {
        return fields() == 0u;
    }
};

// ---------------------------------------------------------------------------------------------- device helpers
// The fixed order of the epilogue, for the routes that implement it. Nothing below is instantiated by a route that
// applies no field, so it adds no machine code there.

// The bias term of column n: +0.0 becomes -0.0, every other value (-0.0 included) is kept. The float overload takes
// float(bias[n]) (the sm_100 cascade's run-time form below); both apply the same rule.
__device__ __forceinline__ float dense_bias_term(float b)
{
    return b == 0.f ? -0.f : b;
}

__device__ __forceinline__ float dense_bias_term(__nv_bfloat16 b)
{
    float const f = __bfloat162float(b);
    return f == 0.f ? -0.f : f;
}

// erf GELU in fp32: 0.5f * v * (1.f + erff(v * kDenseSqrt1_2)), every operation rounded to nearest on its own.
__device__ __forceinline__ float dense_gelu_erf(float v)
{
    return __fmul_rn(__fmul_rn(0.5f, v), __fadd_rn(1.f, erff(__fmul_rn(v, kDenseSqrt1_2))));
}

// v = acc; v += bias_term; v = act(v); [v += residual]. kAct is a DenseActivation. The caller rounds once:
// __float2bfloat16_rn(dense_epilogue_value<...>(...)).
template <bool kBias, int kAct, bool kResidual>
__device__ __forceinline__ float dense_epilogue_value(float acc, __nv_bfloat16 bias_n, __nv_bfloat16 residual_mn)
{
    float v = acc;
    if constexpr (kBias)
        v = __fadd_rn(v, dense_bias_term(bias_n));
    if constexpr (kAct == kDenseActGeluErf)
        v = dense_gelu_erf(v);
    if constexpr (kResidual)
        v = __fadd_rn(v, __bfloat162float(residual_mn));
    return v;
}

// The same with the fields as run-time flags, for a route that selects them at run time.
__device__ __forceinline__ float dense_epilogue_value_rt(
    float acc, bool has_bias, __nv_bfloat16 bias_n, int act, bool has_residual, __nv_bfloat16 residual_mn)
{
    float v = acc;
    if (has_bias)
        v = __fadd_rn(v, dense_bias_term(bias_n));
    if (act == kDenseActGeluErf)
        v = dense_gelu_erf(v);
    if (has_residual)
        v = __fadd_rn(v, __bfloat162float(residual_mn));
    return v;
}

// The same order with the bias and the activation chosen at run time and the residual at compile time, for kernels
// that serve every combination from one instantiation per residual form (the sm_100 cascade's EVT node and its
// split-K reduce). `bias_n` is float(bias[n]) (0 when there is no bias: the -0.0 rule makes acc + bias_term(0) the
// identity on every accumulator value, so a missing bias and a zero bias give the same bits); `act` is a
// DenseActivation; the residual is added only when kResidual.
template <bool kResidual>
__device__ __forceinline__ float dense_epilogue_value_rt(float acc, float bias_n, int act, float residual_mn)
{
    float v = __fadd_rn(acc, dense_bias_term(bias_n));
    if (act == kDenseActGeluErf)
        v = dense_gelu_erf(v);
    if constexpr (kResidual)
        v = __fadd_rn(v, residual_mn);
    return v;
}

// Loads of a bias or residual tensor: plain coherent global loads (ld.global), never the non-coherent read-only path
// (__ldg / ld.global.nc), whose contract wants the data read-only for the whole of the kernel's lifetime; under PDL
// that lifetime overlaps the producer that writes the tensor. The routes call them only after their
// griddepcontrol.wait (its "memory" clobber keeps the compiler from moving them above it), because the producer of
// the tensor may be the kernel launched before them. The pointers reach the kernels inside the struct above, never
// as __restrict__ parameters, so the compiler has no licence to turn them into .nc loads.
__device__ __forceinline__ __nv_bfloat16 dense_load_bf16(__nv_bfloat16 const* p)
{
    return *p;
}

// Two adjacent bf16 as one 4-byte load; p must be 4-byte aligned.
__device__ __forceinline__ __nv_bfloat162 dense_load_bf16x2(__nv_bfloat16 const* p)
{
    return *reinterpret_cast<__nv_bfloat162 const*>(p);
}

} // namespace fso
