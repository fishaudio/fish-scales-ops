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

cudaError_t launch_sm120_mxfp8_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int M, int N, int K, cudaStream_t stream)
{
    tensorrt_llm::kernels::blockscale_gemm::gemm_dispatch_sm120_mxfp8(
        A, B, D, SFA, SFB,
        static_cast<uint32_t>(M), static_cast<uint32_t>(N), static_cast<uint32_t>(K), stream);
    return cudaGetLastError();
}

// Grouped (MoE, masked layout) entry — see gemm_dispatch_sm120_mxfp8_grouped
// in mxfp8/dispatch.cuh for the tensor contracts.
cudaError_t launch_sm120_mxfp8_grouped_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int32_t* masked_m, int num_groups, int m_cap, int N, int K,
    int expected_m, cudaStream_t stream)
{
    tensorrt_llm::kernels::blockscale_gemm::gemm_dispatch_sm120_mxfp8_grouped(
        A, B, D, SFA, SFB, masked_m,
        static_cast<uint32_t>(num_groups), static_cast<uint32_t>(m_cap),
        static_cast<uint32_t>(N), static_cast<uint32_t>(K),
        static_cast<uint32_t>(expected_m), stream);
    return cudaGetLastError();
}

} // namespace blockscale_gemm::detail
