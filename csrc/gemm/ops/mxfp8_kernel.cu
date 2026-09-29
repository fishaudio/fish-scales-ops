/*
 * cute-using TU for the SM120 MXFP8 (1×32, kSFVecSize=32) BlockScaled GEMM.
 * Plain-C interface so the ATen wrapper TU doesn't get cute::Layout vs
 * at::Layout name collision.
 *
 * C2: routes through the full (M, tiles_n) cascade in
 * `sm120_mxfp8_dispatch.cuh`. Reuses the same Stream-K wrapper +
 * sm120_streamk_choose_k_split heuristic as the 1×128 path.
 */
#include "blockscale_gemm/arch/sm120/mxfp8/dispatch.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace blockscale_gemm::detail
{

// The launcher's own default for "ask the driver".
static constexpr int kNumDeviceSMsUnset = -1;

cudaError_t launch_sm120_mxfp8_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int M, int N, int K, cudaStream_t stream)
{
    tensorrt_llm::kernels::blockscale_gemm::gemm_dispatch_sm120_mxfp8(
        A, B, D, SFA, SFB,
        static_cast<uint32_t>(M), static_cast<uint32_t>(N), static_cast<uint32_t>(K), stream);
    return cudaGetLastError();
}

// Grouped (MoE, masked layout) entry — see gemm_dispatch_sm120_mxfp8_grouped
// in mxfp8/dispatch.cuh for the tensor contracts. `max_active_groups` is the
// op's host-static bound min(M * topk, G) (0 = not supplied); the decode tile
// rule reads it.
cudaError_t launch_sm120_mxfp8_grouped_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int32_t* masked_m, int num_groups, int m_cap, int N, int K,
    int expected_m, int max_active_groups, cudaStream_t stream)
{
    tensorrt_llm::kernels::blockscale_gemm::gemm_dispatch_sm120_mxfp8_grouped(
        A, B, D, SFA, SFB, masked_m,
        static_cast<uint32_t>(num_groups), static_cast<uint32_t>(m_cap),
        static_cast<uint32_t>(N), static_cast<uint32_t>(K),
        static_cast<uint32_t>(expected_m), stream, tensorrt_llm::kernels::blockscale_gemm::kNumDeviceSMs,
        nullptr, nullptr, nullptr, nullptr, nullptr, static_cast<uint32_t>(max_active_groups));
    return cudaGetLastError();
}

// Fused-SwiGLU grouped entry (sm_120/121): the FC1 with its SwiGLU + MXFP8
// requantize epilogue, writing the [G, m_cap, N/2] FP8 slab and its K-major
// scale words instead of a bf16 [G, m_cap, N] slab. `D` is unused (the launcher
// points the never-read D descriptor at the FP8 slab).
cudaError_t launch_sm120_mxfp8_grouped_swiglu_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_fp8_e4m3* out_fp8,
    int32_t* out_sf, int32_t* SFA, int32_t* SFB, int32_t* masked_m, int num_groups, int m_cap, int N, int K,
    int expected_m, int max_active_groups, cudaStream_t stream)
{
    tensorrt_llm::kernels::blockscale_gemm::gemm_dispatch_sm120_mxfp8_grouped<true>(
        A, B, reinterpret_cast<__nv_bfloat16*>(out_fp8), SFA, SFB, masked_m,
        static_cast<uint32_t>(num_groups), static_cast<uint32_t>(m_cap),
        static_cast<uint32_t>(N), static_cast<uint32_t>(K),
        static_cast<uint32_t>(expected_m), stream, kNumDeviceSMsUnset, out_fp8, out_sf, nullptr, nullptr, nullptr,
        static_cast<uint32_t>(max_active_groups));
    return cudaGetLastError();
}

// Fused-combine grouped entry (sm_120/121): the FC2 with its weighted-combine
// epilogue, adding each row into its token's row of `out_tokens` instead of
// storing a [G, m_cap, HIDDEN] slab for a separate combine kernel. `out_tokens` is
// accumulated into, so the caller fills it first.
cudaError_t launch_sm120_mxfp8_grouped_combine_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B,
    __nv_bfloat16* out_tokens, int32_t* SFA, int32_t* SFB, int32_t* masked_m, int32_t const* row_map,
    float const* weight_of_slot, int num_groups, int m_cap, int N, int K, int expected_m, int max_active_groups,
    cudaStream_t stream)
{
    tensorrt_llm::kernels::blockscale_gemm::gemm_dispatch_sm120_mxfp8_grouped<false, true>(
        A, B, out_tokens, SFA, SFB, masked_m,
        static_cast<uint32_t>(num_groups), static_cast<uint32_t>(m_cap),
        static_cast<uint32_t>(N), static_cast<uint32_t>(K),
        static_cast<uint32_t>(expected_m), stream, kNumDeviceSMsUnset, nullptr, nullptr,
        out_tokens, row_map, weight_of_slot, static_cast<uint32_t>(max_active_groups));
    return cudaGetLastError();
}

} // namespace blockscale_gemm::detail
