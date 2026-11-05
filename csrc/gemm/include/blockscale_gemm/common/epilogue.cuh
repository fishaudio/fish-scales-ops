/*
 * Kernel-side contract of the dense epilogue (interface v1, issue #12; charter section 7.4).
 *
 * Every dense GEMM route receives one DenseEpilogue by value or by const reference. The fields name what the
 * epilogue applies to the fp32 accumulator after the GEMM (and after any split-K or Stream-K reduction) and before
 * the one rounding to bf16:
 *
 *     v = acc;  v += bias_term(bias[n]);  v = act(v);  [v += residual[m, n]];  out = RN_bf16(v)  [-> quantize]
 *
 * with bias_term(b) = float(b), except that +0.0 becomes -0.0, so that a zero bias is the identity on every
 * accumulator value (-0.0 + -0.0 = -0.0, +0.0 + -0.0 = +0.0) and bias = 0 gives the bytes of bias = None; act is the
 * identity or erf-GELU in fp32, 0.5f * v * (1.f + erff(v * kDenseSqrt1_2)), with the constant rounded to fp32 first
 * (torch's own erf GELU multiplies by float(M_SQRT1_2) in fp32); the residual is added only when given (no + 0.0).
 * Each term is applied once. A route that writes the quantized form (out_q / out_scale, for linear_quantized) rounds
 * to bf16 first and then applies the activation quantizer rule of fso/activation_contract.h to the rounded values.
 *
 * Rules for a route:
 *   * bias and residual are read only after griddepcontrol.wait: their producer may be the kernel launched before.
 *   * a split route (two-kernel split-K, Stream-K) applies the epilogue once, after its reduction.
 *   * a route that cannot honour a field refuses the call (require_dense_epilogue below) before it launches
 *     anything; it never drops a field silently.
 *   * all fields zero / nullptr is today's epilogue, and today's machine code: a route that serves a field compiles
 *     the non-identity epilogue into a separate kernel (or kernel variant), so its identity calls run the kernels
 *     they had before.
 *
 * The struct, the enums and the device helpers are in epilogue_device.cuh, which the sm_90 JIT kernels include too
 * (NVRTC); this header adds the host side: the capability masks of the routes and the refusal. It is included by
 * the ATen translation units (ops/fp8.cu, ops/mxfp8.cu, through runner.h) as well as by the CUTLASS ones, so it
 * pulls in nothing but the CUDA scalar types: no CUTLASS, no cute, no torch.
 */
#pragma once

#include "blockscale_gemm/common/epilogue_device.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace fso
{

// What each route applies today, as DenseEpilogueField masks. Interface v1 landed the plumbing with every route
// serving the identity alone; the phases that add an epilogue to a route widen its mask in the same change. The ATen
// op bodies check the mask of the route they are about to reach before they launch anything (a repack, a quantize),
// and the route's dispatch checks it again for callers that reach it directly.
// sm_90: bias, residual and erf-GELU on all three dense kernels of gemm_dispatch_sm90 (DeepGEMM v2's dense kernel,
// fp8_gemm_kernel and its swap-AB form); the quantized output (linear_quantized) is not served there yet.
constexpr unsigned kDenseEpiServedSm90DeepGemm = kDenseEpiBias | kDenseEpiResidual | kDenseEpiAct;
// gemm_dispatch_sm100_mxfp8 (C++ CUTLASS cascade, issue #12 S1b2): bias, residual and erf-GELU on every tile the
// cascade rules pick and on the two-kernel split-K (applied once, in the reduce). The quantized output
// (linear_quantized, S3b): every configuration the rules pick has a quantized-output instantiation
// (launch_sm100_mxfp8_gemm_epi_q) and the split-K applies the quantizer once, in its own reduce; it takes bias and
// the activation, no residual (the dispatch refuses out_q together with a residual).
constexpr unsigned kDenseEpiServedSm100Cascade = kDenseEpiBias | kDenseEpiResidual | kDenseEpiAct | kDenseEpiQuantOut;
// gemm_dispatch_sm120_mxfp8: every single-launch tile of its cascade has an epilogue instantiation
// (SM120DenseEpiBuilder) and a quantized-output one (SM120DenseQuantBuilder, linear_quantized: bias and activation,
// no residual), and its Stream-K route applies the fields once in the reduce kernel (the quantized output in its own
// reduce). The forced-only diagnostic instances (FSO_FORCE_SMALLM, FSO_FORCE_MIN_BLOCKS=2, FSO_FORCE_SCHED_GROUP)
// refuse a non-identity one.
constexpr unsigned kDenseEpiServedSm120Mxfp8Cascade
    = unsigned(kDenseEpiBias) | unsigned(kDenseEpiResidual) | unsigned(kDenseEpiAct) | unsigned(kDenseEpiQuantOut);
constexpr unsigned kDenseEpiServedSm120Fp8Cascade = 0u;  // gemm_dispatch_sm120 (block-FP8 1x128)
// The sm_120/sm_121 small-M route (arch/sm120/mxfp8/smallm_tc.cuh, issue #12 decision 3: the decode band M <= 64,
// G1 for the 2b call and G2, quantize inside, for the bf16-x call): bias, residual and erf-GELU, applied once after
// its cross-warp reduction. The quantized output is not served.
constexpr unsigned kDenseEpiServedSm120Smallm
    = unsigned(kDenseEpiBias) | unsigned(kDenseEpiResidual) | unsigned(kDenseEpiAct);

// Thrown by a route that cannot apply a field of the epilogue it was given. The ATen op bodies turn it into
// c10::NotImplementedError (Python NotImplementedError); a direct C++ caller sees a std::runtime_error.
class DenseEpilogueNotImplemented : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// The names of the fields of `mask`, for messages.
inline std::string dense_epilogue_field_names(unsigned mask)
{
    std::string s;
    auto add = [&](unsigned bit, char const* name)
    {
        if ((mask & bit) == 0u)
            return;
        if (!s.empty())
            s += ", ";
        s += name;
    };
    add(kDenseEpiBias, "bias");
    add(kDenseEpiResidual, "residual");
    add(kDenseEpiAct, "activation 'gelu_erf'");
    add(kDenseEpiQuantOut, "quantized output");
    return s;
}

// Refuse `e` unless every field it sets is in `served`. `where` names the architecture and the route.
inline void require_dense_epilogue(DenseEpilogue const& e, unsigned served, char const* where)
{
    unsigned const missing = e.fields() & ~served;
    if (missing == 0u)
        return;
    throw DenseEpilogueNotImplemented(std::string(where) + ": the dense epilogue (" + dense_epilogue_field_names(missing)
        + ") is not implemented on this route yet; call without it and apply it to the bf16 result");
}

} // namespace fso
