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
#include <cstddef>
#include <cstdint>

namespace blockscale_gemm::detail
{

// The launcher's own default for "ask the driver".
static constexpr int kNumDeviceSMsUnset = -1;

// The route of one dense call (sm120_mxfp8_dense_route), with the SM count resolved as the dispatch resolves it.
static tensorrt_llm::kernels::blockscale_gemm::Sm120DenseRoute sm120_mxfp8_route_of(int M, int N, int K)
{
    namespace bg = tensorrt_llm::kernels::blockscale_gemm;
    if (bg::kNumDeviceSMs < 0)
        bg::kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    return bg::sm120_mxfp8_dense_route(
        static_cast<uint32_t>(M), static_cast<uint32_t>(N), static_cast<uint32_t>(K), bg::kNumDeviceSMs);
}

// Host only, before anything is allocated or launched: refuses `epilogue` where the route cannot apply it (a field
// outside kDenseEpiServedSm120Mxfp8Cascade, or a forced diagnostic instance; fso::DenseEpilogueNotImplemented) and
// returns the bytes of bf16 partial scratch the route needs (0 for a single launch). The op body allocates that from
// the torch caching allocator on the caller's stream and passes it to launch_sm120_mxfp8_dispatch.
std::size_t sm120_mxfp8_dense_prepare(int M, int N, int K, fso::DenseEpilogue const& epilogue)
{
    namespace bg = tensorrt_llm::kernels::blockscale_gemm;
    fso::require_dense_epilogue(epilogue, fso::kDenseEpiServedSm120Mxfp8Cascade,
        "sm_120/sm_121 dense MXFP8 GEMM (gemm_dispatch_sm120_mxfp8, the CUTLASS cascade)");
    bg::Sm120DenseRoute const route = sm120_mxfp8_route_of(M, N, K);
    bg::sm120_mxfp8_dense_require_epilogue(route, epilogue);
    bg::sm120_mxfp8_dense_require_quant_out(epilogue, static_cast<uint32_t>(N));
    return route.streamk_scratch_bytes(M, N);
}

// The route as eight integers, for the host-side query op (mxfp8_dense_route_sm120): tile_m, tile_n, stages,
// min_blocks, sched_group, k_split, smallm, Stream-K scratch bytes.
void sm120_mxfp8_dense_route_query(int M, int N, int K, int64_t* out8)
{
    auto const r = sm120_mxfp8_route_of(M, N, K);
    out8[0] = r.tile_m;
    out8[1] = r.tile_n;
    out8[2] = r.stages;
    out8[3] = r.min_blocks;
    out8[4] = r.sched_group;
    out8[5] = r.k_split;
    out8[6] = r.smallm ? 1 : 0;
    out8[7] = static_cast<int64_t>(r.streamk_scratch_bytes(M, N));
}

// `epilogue` (common/epilogue.cuh): the dispatch throws fso::DenseEpilogueNotImplemented for a field its route does
// not apply, before it launches anything; the op body (ops/mxfp8.cu) turns that into NotImplementedError. With
// epilogue.out_q (linear_quantized) D is out_q and is not written as bf16 (gemm_dispatch_sm120_mxfp8).
// `streamk_scratch` / `streamk_scratch_bytes`: the caller's partial scratch (sm120_mxfp8_dense_prepare's size).
cudaError_t launch_sm120_mxfp8_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int M, int N, int K, cudaStream_t stream, fso::DenseEpilogue const& epilogue,
    void* streamk_scratch, std::size_t streamk_scratch_bytes)
{
    tensorrt_llm::kernels::blockscale_gemm::gemm_dispatch_sm120_mxfp8(
        A, B, D, SFA, SFB,
        static_cast<uint32_t>(M), static_cast<uint32_t>(N), static_cast<uint32_t>(K), stream,
        tensorrt_llm::kernels::blockscale_gemm::kNumDeviceSMs, epilogue, streamk_scratch, streamk_scratch_bytes);
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
