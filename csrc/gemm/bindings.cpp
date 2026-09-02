/*
 * Torch op registration for fish-scales-ops GEMM domain.
 * The C++ namespace stays `blockscale_gemm` (internal); the torch.ops
 * registration uses `fish_scales_ops`. The single PYBIND11_MODULE for
 * the combined extension lives in csrc/attention/csrc/flash_attn_ext.cpp.
 *
 * Surface:
 *   FP8 1×128 / 128×128 block-scaled — sm_90 deep_gemm JIT + sm_120
 *     CUTLASS Sm120BlockScaledKernel + Stream-K.
 *   MXFP8 1×32 (sm_120 only)         — CUTLASS Sm120MxFP8BlockScaledKernel.
 *   BF16 convenience wrappers        — linear_bf16, linear_qx (FP8 path).
 */

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <tuple>

namespace blockscale_gemm
{
at::Tensor linear_bf16(at::Tensor x, at::Tensor w);
at::Tensor linear_fp8(at::Tensor x_fp8, at::Tensor w_fp8, at::Tensor sx, at::Tensor sw);
at::Tensor linear_qx(at::Tensor x_bf16, at::Tensor w_fp8, at::Tensor sw);
std::tuple<at::Tensor, at::Tensor> quantize_1x128(at::Tensor x, bool use_ue8m0);
std::tuple<at::Tensor, at::Tensor> quantize_1x128_packed(at::Tensor x, bool use_ue8m0);
std::tuple<at::Tensor, at::Tensor> quantize_128x128(at::Tensor w);
at::Tensor repack_fp8_act_scales(at::Tensor sx_f32);
at::Tensor repack_fp8_wgt_scales(at::Tensor sw_f32);
// MXFP8 1×32 (sm_120 only).
std::tuple<at::Tensor, at::Tensor> quantize_1x32(at::Tensor x, bool use_ue8m0);
std::tuple<at::Tensor, at::Tensor> quantize_1x32_packed(at::Tensor x, bool use_ue8m0);
std::tuple<at::Tensor, at::Tensor> silu_chunk_mul_quantize_1x32(at::Tensor gu, bool use_ue8m0);
at::Tensor repack_mxfp8_scales(at::Tensor scales_f32);
at::Tensor linear_mxfp8_raw(at::Tensor x_fp8, at::Tensor w_fp8,
                            at::Tensor sx_int32, at::Tensor sw_int32);
// Grouped (MoE, masked layout) MXFP8 — sm_120 only (M1).
at::Tensor linear_mxfp8_grouped_masked(at::Tensor a_fp8, at::Tensor w_fp8, at::Tensor sa_int32,
    at::Tensor sw_int32, at::Tensor masked_m, int64_t expected_m);
std::tuple<at::Tensor, at::Tensor> quantize_1x32_grouped_gather(
    at::Tensor x, at::Tensor slot_of_flat, int64_t topk, int64_t num_groups,
    int64_t m_cap, bool use_ue8m0);
std::tuple<at::Tensor, at::Tensor> silu_chunk_mul_quantize_1x32_grouped(
    at::Tensor gu, at::Tensor slot_of_flat, bool use_ue8m0);
std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_build_routing(
    at::Tensor topk_ids, int64_t num_groups, int64_t m_cap);
at::Tensor moe_combine(at::Tensor dn, at::Tensor slot_of_flat, at::Tensor topk_w);
} // namespace blockscale_gemm

TORCH_LIBRARY_FRAGMENT(fish_scales_ops, m)
{
    m.def("linear_bf16(Tensor x, Tensor w) -> Tensor");
    m.def("linear_fp8(Tensor x_fp8, Tensor w_fp8, Tensor sx, Tensor sw) -> Tensor");
    m.def("linear_qx(Tensor x_bf16, Tensor w_fp8, Tensor sw) -> Tensor");
    m.def("quantize_1x128(Tensor x, bool use_ue8m0=False) -> (Tensor, Tensor)");
    m.def("quantize_1x128_packed(Tensor x, bool use_ue8m0=True) -> (Tensor, Tensor)");
    m.def("quantize_128x128(Tensor w) -> (Tensor, Tensor)");
    m.def("repack_fp8_act_scales(Tensor sx_f32) -> Tensor");
    m.def("repack_fp8_wgt_scales(Tensor sw_f32) -> Tensor");
    m.def("quantize_1x32(Tensor x, bool use_ue8m0=True) -> (Tensor, Tensor)");
    m.def("quantize_1x32_packed(Tensor x, bool use_ue8m0=True) -> (Tensor, Tensor)");
    m.def("silu_chunk_mul_quantize_1x32(Tensor gu, bool use_ue8m0=True) -> (Tensor, Tensor)");
    m.def("repack_mxfp8_scales(Tensor scales_f32) -> Tensor");
    m.def("linear_mxfp8_raw(Tensor x_fp8, Tensor w_fp8, "
                            "Tensor sx_int32, Tensor sw_int32) -> Tensor");
    m.def("linear_mxfp8_grouped_masked(Tensor a_fp8, Tensor w_fp8, Tensor sa_int32, "
                                       "Tensor sw_int32, Tensor masked_m, int expected_m) -> Tensor");
    m.def("quantize_1x32_grouped_gather(Tensor x, Tensor slot_of_flat, int topk, "
                                        "int num_groups, int m_cap, bool use_ue8m0=True) -> (Tensor, Tensor)");
    m.def("silu_chunk_mul_quantize_1x32_grouped(Tensor gu, Tensor slot_of_flat, "
                                                "bool use_ue8m0=True) -> (Tensor, Tensor)");
    m.def("moe_build_routing(Tensor topk_ids, int num_groups, int m_cap) "
          "-> (Tensor, Tensor, Tensor)");
    m.def("moe_combine(Tensor dn, Tensor slot_of_flat, Tensor topk_w) -> Tensor");
}

TORCH_LIBRARY_IMPL(fish_scales_ops, CUDA, m)
{
    m.impl("linear_bf16", &blockscale_gemm::linear_bf16);
    m.impl("linear_fp8", &blockscale_gemm::linear_fp8);
    m.impl("linear_qx", &blockscale_gemm::linear_qx);
    m.impl("quantize_1x128", &blockscale_gemm::quantize_1x128);
    m.impl("quantize_1x128_packed", &blockscale_gemm::quantize_1x128_packed);
    m.impl("quantize_128x128", &blockscale_gemm::quantize_128x128);
    m.impl("repack_fp8_act_scales", &blockscale_gemm::repack_fp8_act_scales);
    m.impl("repack_fp8_wgt_scales", &blockscale_gemm::repack_fp8_wgt_scales);
    m.impl("quantize_1x32", &blockscale_gemm::quantize_1x32);
    m.impl("quantize_1x32_packed", &blockscale_gemm::quantize_1x32_packed);
    m.impl("silu_chunk_mul_quantize_1x32", &blockscale_gemm::silu_chunk_mul_quantize_1x32);
    m.impl("repack_mxfp8_scales", &blockscale_gemm::repack_mxfp8_scales);
    m.impl("linear_mxfp8_raw", &blockscale_gemm::linear_mxfp8_raw);
    m.impl("linear_mxfp8_grouped_masked", &blockscale_gemm::linear_mxfp8_grouped_masked);
    m.impl("quantize_1x32_grouped_gather", &blockscale_gemm::quantize_1x32_grouped_gather);
    m.impl("silu_chunk_mul_quantize_1x32_grouped", &blockscale_gemm::silu_chunk_mul_quantize_1x32_grouped);
    m.impl("moe_build_routing", &blockscale_gemm::moe_build_routing);
    m.impl("moe_combine", &blockscale_gemm::moe_combine);
}
