/*
 * The epilogue arguments of the dense raw ops (linear_fp8, linear_qx, linear_mxfp8_raw; interface v1, issue #12):
 * validation of `Tensor? bias, Tensor? residual, int act` from metadata only, the fso::DenseEpilogue they fill, and
 * the translation of a route's refusal into NotImplementedError. ATen translation units only (ops/fp8.cu,
 * ops/mxfp8.cu); the struct itself is in blockscale_gemm/common/epilogue.cuh.
 */
#pragma once

#include "blockscale_gemm/common/epilogue.cuh"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <optional>

namespace blockscale_gemm::detail
{

// bias: bf16 [N], contiguous, on `device`. residual: bf16 [M, N], contiguous, on `device`. act: 0 (none) or 1
// (gelu_erf). Every check reads metadata only (dtype, sizes, strides, device); a malformed argument is a ValueError.
inline fso::DenseEpilogue make_dense_epilogue(char const* op, std::optional<at::Tensor> const& bias,
    std::optional<at::Tensor> const& residual, int64_t act, int64_t M, int64_t N, at::Device device)
{
    fso::DenseEpilogue e;
    if (bias.has_value())
    {
        at::Tensor const& b = *bias;
        TORCH_CHECK_VALUE(b.scalar_type() == at::kBFloat16, op, ": bias must be bf16, got ", b.scalar_type());
        TORCH_CHECK_VALUE(b.dim() == 1 && b.size(0) == N, op, ": bias must be [N] = [", N, "], got ", b.sizes());
        TORCH_CHECK_VALUE(b.device() == device, op, ": bias is on ", b.device(), " and the operands on ", device);
        TORCH_CHECK_VALUE(b.is_contiguous(), op, ": bias must be contiguous, got strides ", b.strides());
        e.bias = reinterpret_cast<__nv_bfloat16 const*>(b.data_ptr());
    }
    if (residual.has_value())
    {
        at::Tensor const& r = *residual;
        TORCH_CHECK_VALUE(r.scalar_type() == at::kBFloat16, op, ": residual must be bf16, got ", r.scalar_type());
        TORCH_CHECK_VALUE(r.dim() == 2 && r.size(0) == M && r.size(1) == N, op, ": residual must be the output's shape [M, N] = [",
            M, ", ", N, "], got ", r.sizes());
        TORCH_CHECK_VALUE(r.device() == device, op, ": residual is on ", r.device(), " and the operands on ", device);
        TORCH_CHECK_VALUE(r.is_contiguous(), op, ": residual must be contiguous, got strides ", r.strides());
        e.residual = reinterpret_cast<__nv_bfloat16 const*>(r.data_ptr());
        e.ld_residual = N;
    }
    TORCH_CHECK_VALUE(act == fso::kDenseActNone || act == fso::kDenseActGeluErf, op,
        ": act must be 0 (no activation) or 1 (gelu_erf), got ", act);
    e.act = static_cast<int>(act);
    return e;
}

// The route's refusal (fso::DenseEpilogueNotImplemented) as c10::NotImplementedError, which Python sees as
// NotImplementedError. `where` names the architecture and the route.
inline void require_dense_epilogue_op(fso::DenseEpilogue const& e, unsigned served, char const* where)
{
    try
    {
        fso::require_dense_epilogue(e, served, where);
    }
    catch (fso::DenseEpilogueNotImplemented const& ex)
    {
        TORCH_CHECK_NOT_IMPLEMENTED(false, ex.what());
    }
}

// f(), with a route's refusal thrown from inside it (fso::DenseEpilogueNotImplemented: a forced configuration
// without a dense-epilogue instantiation, a residual the route cannot read) turned into NotImplementedError.
template <class F>
inline auto dense_epilogue_refusing(F&& f) -> decltype(f())
{
    try
    {
        return f();
    }
    catch (fso::DenseEpilogueNotImplemented const& ex)
    {
        TORCH_CHECK_NOT_IMPLEMENTED(false, ex.what());
        throw; // not reached: TORCH_CHECK_NOT_IMPLEMENTED(false, ...) throws
    }
}

} // namespace blockscale_gemm::detail
