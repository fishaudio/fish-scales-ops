/*
 * cute-using TU for the sm_120/121 grouped MXFP8 GEMMs on the COMPACT activation
 * layout (0.2.1, charter decision D1). Plain-C interface, like mxfp8_kernel.cu, so the
 * ATen wrapper TU does not see cute's Layout.
 *
 * The compact layout replaces the masked slab's G groups of m_cap rows with ONE group
 * of R rows made of 128-row blocks: expert g's routed rows are rows
 * [blk0[g] * 128, blk0[g] * 128 + masked_m[g]) of A [R, K] and of the output, and its
 * scale words are the same rows of the one-group K-major words [1, K/128, R]. The
 * weights keep the expert index. The tile cascade is the masked layout's own
 * (gemm_dispatch_sm120_mxfp8_grouped with CompactActs = true), fed the same host-side
 * hints (m_cap, expected_m, max_active_groups), so a compact call runs the instance
 * the masked call of the same shape runs. These instantiations live in their own TU
 * only so that they compile in parallel with mxfp8_kernel.cu.
 */
#include "blockscale_gemm/arch/sm120/mxfp8/dispatch.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace blockscale_gemm::detail
{

// The launcher's own default for "ask the driver".
static constexpr int kNumDeviceSMsUnsetCompact = -1;

// Plain grouped GEMM on the compact layout: D [act_rows, N] bf16. Rows of D that no
// expert's live range covers are not written.
cudaError_t launch_sm120_mxfp8_grouped_compact_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int32_t* masked_m, int32_t const* blk0, int num_groups, int m_cap, int act_rows,
    int N, int K, int expected_m, int max_active_groups, cudaStream_t stream)
{
    tensorrt_llm::kernels::blockscale_gemm::gemm_dispatch_sm120_mxfp8_grouped<false, false, true>(
        A, B, D, SFA, SFB, masked_m,
        static_cast<uint32_t>(num_groups), static_cast<uint32_t>(m_cap),
        static_cast<uint32_t>(N), static_cast<uint32_t>(K),
        static_cast<uint32_t>(expected_m), stream, tensorrt_llm::kernels::blockscale_gemm::kNumDeviceSMs,
        nullptr, nullptr, nullptr, nullptr, nullptr, static_cast<uint32_t>(max_active_groups),
        blk0, static_cast<uint32_t>(act_rows));
    return cudaGetLastError();
}

// Fused-SwiGLU FC1 on the compact layout: writes the FP8 rows [act_rows, N/2] and the
// one-group K-major scale words [1, (N/2)/128, act_rows] of the live rows. `D` is not
// passed: as in the masked entry, the never-read D descriptor points at the FP8 rows.
cudaError_t launch_sm120_mxfp8_grouped_compact_swiglu_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B,
    __nv_fp8_e4m3* out_fp8, int32_t* out_sf, int32_t* SFA, int32_t* SFB, int32_t* masked_m, int32_t const* blk0,
    int num_groups, int m_cap, int act_rows, int N, int K, int expected_m, int max_active_groups, cudaStream_t stream)
{
    tensorrt_llm::kernels::blockscale_gemm::gemm_dispatch_sm120_mxfp8_grouped<true, false, true>(
        A, B, reinterpret_cast<__nv_bfloat16*>(out_fp8), SFA, SFB, masked_m,
        static_cast<uint32_t>(num_groups), static_cast<uint32_t>(m_cap),
        static_cast<uint32_t>(N), static_cast<uint32_t>(K),
        static_cast<uint32_t>(expected_m), stream, kNumDeviceSMsUnsetCompact, out_fp8, out_sf, nullptr, nullptr,
        nullptr, static_cast<uint32_t>(max_active_groups), blk0, static_cast<uint32_t>(act_rows));
    return cudaGetLastError();
}

// Fused-combine FC2 on the compact layout: adds each live row, scaled by
// weight_of_slot[row], into out_tokens[row_map[row]] (both indexed by compact row).
cudaError_t launch_sm120_mxfp8_grouped_compact_combine_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B,
    __nv_bfloat16* out_tokens, int32_t* SFA, int32_t* SFB, int32_t* masked_m, int32_t const* blk0,
    int32_t const* row_map, float const* weight_of_slot, int num_groups, int m_cap, int act_rows, int N, int K,
    int expected_m, int max_active_groups, cudaStream_t stream)
{
    tensorrt_llm::kernels::blockscale_gemm::gemm_dispatch_sm120_mxfp8_grouped<false, true, true>(
        A, B, out_tokens, SFA, SFB, masked_m,
        static_cast<uint32_t>(num_groups), static_cast<uint32_t>(m_cap),
        static_cast<uint32_t>(N), static_cast<uint32_t>(K),
        static_cast<uint32_t>(expected_m), stream, kNumDeviceSMsUnsetCompact, nullptr, nullptr,
        out_tokens, row_map, weight_of_slot, static_cast<uint32_t>(max_active_groups), blk0,
        static_cast<uint32_t>(act_rows));
    return cudaGetLastError();
}

} // namespace blockscale_gemm::detail
