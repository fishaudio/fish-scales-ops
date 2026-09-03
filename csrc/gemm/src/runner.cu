/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "blockscale_gemm/runner.h"
#include "blockscale_gemm/dispatch.cuh"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/logger.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::blockscale_gemm
{

template <typename ElementA, typename ElementB, typename ElementD>
void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::gemm(void* mat_d, void const* mat_a,
    void const* mat_b, int shape_m, int shape_n, int shape_k, cudaStream_t stream, float const* scales_a,
    float const* scales_b)
{
    constexpr bool internal_quantize_a = !std::is_same_v<ElementA, __nv_fp8_e4m3>;
    constexpr bool internal_quantize_b = !std::is_same_v<ElementB, __nv_fp8_e4m3>;
    __nv_fp8_e4m3* fp8_mat_a;
    __nv_fp8_e4m3* fp8_mat_b;
    float* per_token_per_128c_scales;
    float* per_block_scales;

    auto* ws_ptr = workspace_;
    if constexpr (internal_quantize_a || internal_quantize_b)
    {
        TLLM_CHECK(ws_ptr != nullptr);
    }

    if constexpr (internal_quantize_a)
    {
        fp8_mat_a = reinterpret_cast<__nv_fp8_e4m3*>(ws_ptr);
        ws_ptr += max_shape_m_4_align_ * shape_k * sizeof(__nv_fp8_e4m3);
        per_token_per_128c_scales = reinterpret_cast<float*>(ws_ptr);
        ws_ptr += max_shape_m_4_align_ * div_up(shape_k, 128) * sizeof(float);
    }

    if constexpr (internal_quantize_b)
    {
        fp8_mat_b = reinterpret_cast<__nv_fp8_e4m3*>(ws_ptr);
        ws_ptr += shape_n * shape_k * sizeof(__nv_fp8_e4m3);
        per_block_scales = reinterpret_cast<float*>(ws_ptr);
        ws_ptr += div_up(shape_n, 128) * div_up(shape_k, 128) * sizeof(float);
    }

#ifdef COMPILE_HOPPER_TMA_GEMMS
    if constexpr (internal_quantize_a && internal_quantize_b)
    {
        fp8_gemm_run(reinterpret_cast<__nv_bfloat16 const*>(mat_a), fp8_mat_a, shape_k, per_token_per_128c_scales,
            reinterpret_cast<__nv_bfloat16 const*>(mat_b), fp8_mat_b, shape_k, per_block_scales,
            reinterpret_cast<__nv_bfloat16*>(mat_d), shape_n, shape_m, shape_n, shape_k, stream, internal_quantize_a,
            internal_quantize_b);
    }

    if constexpr (internal_quantize_a && !internal_quantize_b)
    {
        fp8_gemm_run(reinterpret_cast<__nv_bfloat16 const*>(mat_a), fp8_mat_a, shape_k, per_token_per_128c_scales,
            nullptr, reinterpret_cast<__nv_fp8_e4m3*>(const_cast<void*>(mat_b)), shape_k, const_cast<float*>(scales_b),
            reinterpret_cast<__nv_bfloat16*>(mat_d), shape_n, shape_m, shape_n, shape_k, stream, internal_quantize_a,
            internal_quantize_b);
    }
#else  // COMPILE_HOPPER_TMA_GEMMS
    TLLM_THROW("fp8 blockscale gemm only support Hopper.");
#endif // COMPILE_HOPPER_TMA_GEMMS
}

template <typename ElementA, typename ElementB, typename ElementD>
void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::gemm(__nv_fp8_e4m3 const* mat_a, int ld_a,
    __nv_fp8_e4m3 const* mat_b, int ld_b, __nv_bfloat16* mat_d, int ld_d, int shape_m, int shape_n, int shape_k,
    float const* scales_a, float const* scales_b, cudaStream_t stream)
{

    fp8_gemm_run(const_cast<__nv_fp8_e4m3*>(mat_a), ld_a, const_cast<__nv_fp8_e4m3*>(mat_b), ld_b, mat_d, ld_d, shape_m,
        shape_n, shape_k, const_cast<float*>(scales_a), const_cast<float*>(scales_b), stream);
}

template <typename ElementA, typename ElementB, typename ElementD>
void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::fp8CS1x128(__nv_fp8_e4m3* mat_quant, float* scales,
    __nv_bfloat16 const* mat, int shape_x, int shape_y, cudaStream_t stream, bool use_ue8m0)
{
    fp8_1x128_cs(mat_quant, scales, mat, shape_x, shape_y, stream, use_ue8m0);
}

template <typename ElementA, typename ElementB, typename ElementD>
void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::fp8CS1x128Reshape(__nv_fp8_e4m3* mat_quant,
    float* scales, __nv_bfloat16 const* mat, int shape_x, int shape_h, int shape_y, int stride_x, cudaStream_t stream)
{
    fp8_1x128_cs_reshape(mat_quant, scales, mat, shape_x, shape_h, shape_y, stride_x, stream);
}

template <typename ElementA, typename ElementB, typename ElementD>
void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::fp8CS128x128(
    __nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x, int shape_y, cudaStream_t stream)
{
    fp8_128x128_cs(mat_quant, scales, mat, shape_x, shape_y, stream);
}

template <typename ElementA, typename ElementB, typename ElementD>
size_t CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::getWorkspaceSizeBase(
    size_t max_shape_m, size_t shape_n, size_t shape_k, size_t num_problems)
{
    // num_problems is retained for ABI back-compat but is always 1 since the
    // MoE/grouped-GEMM path was removed (2026-05-21).
    (void) num_problems;
    max_shape_m_4_align_ = std::max(max_shape_m_4_align_, int64_t(div_up(max_shape_m, 4) * 4));
    // Single-problem path: 32-aligned ceil of max_shape_m (was
    // compute_padded_offset(max_shape_m, 1) from the deleted moe_padding.cuh).
    max_shape_m_32_align_padded_ = int64_t((max_shape_m + 31) / 32 * 32);

    constexpr bool internal_quantize_a = !std::is_same_v<ElementA, __nv_fp8_e4m3>;
    constexpr bool internal_quantize_b = !std::is_same_v<ElementB, __nv_fp8_e4m3>;
    size_t total_workspace_size = 0;
    if constexpr (internal_quantize_a)
    {
        // fp8_mat_a
        total_workspace_size += max_shape_m_4_align_ * shape_k * sizeof(__nv_fp8_e4m3);
        // scales_a
        total_workspace_size += max_shape_m_32_align_padded_ * div_up(shape_k, 128) * sizeof(float);
    }

    if constexpr (internal_quantize_b)
    {
        // fp8_mat_b
        total_workspace_size += shape_n * shape_k * sizeof(__nv_fp8_e4m3);
        // scales_b
        total_workspace_size += div_up(shape_k, 128) * div_up(shape_n, 128) * sizeof(float);
    }

    return total_workspace_size;
}

template <typename ElementA, typename ElementB, typename ElementD>
size_t CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::getWorkspaceSize(
    size_t shape_m, size_t shape_n, size_t shape_k, size_t top_k, size_t num_problems)
{
    return getWorkspaceSizeBase(shape_m * top_k, shape_n, shape_k, num_problems);
}

template <typename ElementA, typename ElementB, typename ElementD>
size_t CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::getFP8DataSize(
    int shape_m, int shape_n, bool is_act)
{
    int shape_m_4_align = div_up(shape_m, 4) * 4;
    constexpr bool internal_quantize_a = !std::is_same_v<ElementA, __nv_fp8_e4m3>;
    constexpr bool internal_quantize_b = !std::is_same_v<ElementB, __nv_fp8_e4m3>;
    if (is_act && internal_quantize_a)
    {
        return div_up(shape_m_4_align * shape_n * sizeof(__nv_fp8_e4m3), 128) * 128;
    }

    if ((!is_act) && internal_quantize_b)
    {
        return div_up(shape_m * shape_n * sizeof(__nv_fp8_e4m3), 128) * 128;
    }
    return 0;
}

template <typename ElementA, typename ElementB, typename ElementD>
size_t CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::getActScaleSize(int shape_m, int shape_k)
{
    int shape_m_4_align = div_up(shape_m, 4) * 4;
    constexpr bool internal_quantize_a = !std::is_same_v<ElementA, __nv_fp8_e4m3>;
    size_t total_workspace_size = 0;
    if constexpr (internal_quantize_a)
    {
        // scales_a
        total_workspace_size += div_up(shape_m_4_align * div_up(shape_k, 128) * sizeof(float), 128) * 128;
    }
    return total_workspace_size;
}

template <typename ElementA, typename ElementB, typename ElementD>
size_t CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::getWeightScaleSize(int shape_n, int shape_k)
{
    constexpr bool internal_quantize_b = !std::is_same_v<ElementB, __nv_fp8_e4m3>;
    size_t total_workspace_size = 0;
    if constexpr (internal_quantize_b)
    {
        // scales_b
        total_workspace_size += div_up(div_up(shape_k, 128) * div_up(shape_n, 128) * sizeof(float), 128) * 128;
    }

    return total_workspace_size;
}

template <typename ElementA, typename ElementB, typename ElementD>
size_t CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::getActWorkspaceSize(int shape_m, int shape_k)
{
    return getFP8DataSize(shape_m, shape_k, true) + getActScaleSize(shape_m, shape_k);
}

template <typename ElementA, typename ElementB, typename ElementD>
size_t CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::getWeightWorkspaceSize(int shape_n, int shape_k)
{
    return getFP8DataSize(shape_n, shape_k, false) + getWeightScaleSize(shape_n, shape_k);
}

template class CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>;
template class CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>;
template class CutlassFp8BlockScaleGemmRunner<__nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat16>;
template class CutlassFp8BlockScaleGemmRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>;

} // namespace kernels::blockscale_gemm

TRTLLM_NAMESPACE_END

// C-ABI entry for the sm_90 (H200) grouped block-scale FP8 masked GEMM.
// Lives here (not a standalone TU) because runner.cu is the single owner of
// the vendored DeepGEMM JIT symbols — a second TU that includes dispatch.cuh
// double-defines the JIT's non-inline helpers at link time.
namespace blockscale_gemm::detail
{
cudaError_t launch_sm90_fp8_grouped_masked_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    float* SFA, float* SFB, int32_t* masked_m, int num_groups, int m_cap, int N, int K, int expected_m,
    cudaStream_t stream)
{
#ifdef COMPILE_HOPPER_TMA_GEMMS
    tensorrt_llm::kernels::blockscale_gemm::gemm_dispatch_sm90_grouped_masked(
        reinterpret_cast<void*>(A), reinterpret_cast<void*>(B), reinterpret_cast<void*>(D), SFA, SFB, masked_m,
        static_cast<uint32_t>(num_groups), static_cast<uint32_t>(m_cap), static_cast<uint32_t>(N),
        static_cast<uint32_t>(K), static_cast<uint32_t>(expected_m), stream);
    return cudaGetLastError();
#else
    (void) A; (void) B; (void) D; (void) SFA; (void) SFB; (void) masked_m;
    (void) num_groups; (void) m_cap; (void) N; (void) K; (void) expected_m; (void) stream;
    return cudaErrorNotSupported;
#endif
}

// C-ABI entry for the sm_90 grouped block-scale FP8 contiguous (sorted) GEMM.
cudaError_t launch_sm90_fp8_grouped_contiguous_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    float* SFA, float* SFB, int32_t* sorted_expert_ids, int num_groups, int p_max, int N, int K, int block_m,
    int expected_m, cudaStream_t stream)
{
#ifdef COMPILE_HOPPER_TMA_GEMMS
    tensorrt_llm::kernels::blockscale_gemm::gemm_dispatch_sm90_grouped_contiguous(
        reinterpret_cast<void*>(A), reinterpret_cast<void*>(B), reinterpret_cast<void*>(D), SFA, SFB,
        sorted_expert_ids, static_cast<uint32_t>(num_groups), static_cast<uint32_t>(p_max),
        static_cast<uint32_t>(N), static_cast<uint32_t>(K), static_cast<uint32_t>(block_m),
        static_cast<uint32_t>(expected_m), stream);
    return cudaGetLastError();
#else
    (void) A; (void) B; (void) D; (void) SFA; (void) SFB; (void) sorted_expert_ids;
    (void) num_groups; (void) p_max; (void) N; (void) K; (void) block_m; (void) expected_m; (void) stream;
    return cudaErrorNotSupported;
#endif
}

// C-ABI entry for the sm_90 grouped contiguous swap-AB GEMM (block_n=16
// activation tiling for M>=8). A = activation [P_max,K], B = weights [G,N,K],
// SFA = activation K-major scales, SFB = weight scales [G,N/128,K/128].
cudaError_t launch_sm90_fp8_grouped_contiguous_swapab_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    float* SFA, float* SFB, int32_t* sorted_expert_ids, int num_groups, int p_max, int N, int K, int block_n,
    int expected_m, cudaStream_t stream)
{
#ifdef COMPILE_HOPPER_TMA_GEMMS
    // Weight is the swap-AB A matrix, activation the B matrix.
    tensorrt_llm::kernels::blockscale_gemm::gemm_dispatch_sm90_grouped_contiguous_swapab(
        reinterpret_cast<void*>(B), reinterpret_cast<void*>(A), reinterpret_cast<void*>(D), SFB, SFA,
        sorted_expert_ids, static_cast<uint32_t>(num_groups), static_cast<uint32_t>(p_max),
        static_cast<uint32_t>(N), static_cast<uint32_t>(K), static_cast<uint32_t>(block_n),
        static_cast<uint32_t>(expected_m), stream);
    return cudaGetLastError();
#else
    (void) A; (void) B; (void) D; (void) SFA; (void) SFB; (void) sorted_expert_ids;
    (void) num_groups; (void) p_max; (void) N; (void) K; (void) block_n; (void) expected_m; (void) stream;
    return cudaErrorNotSupported;
#endif
}
} // namespace blockscale_gemm::detail

