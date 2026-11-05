/*
 * The dense-epilogue instances of the sm_120/sm_121 MXFP8 GEMM (interface v1, issue #12; common/epilogue.cuh).
 *
 * One instantiation per single-launch tile of the dense cascade (mxfp8/dispatch.cuh, sm120_mxfp8_dense_route: the
 * main cascade, its overrides and the TileM = 16 routes), each the kernel over
 * SM120DenseEpiBuilder<SM120MxFP8BlockScaledBuilder<TileM, TileN, Stages>>, which applies bias, the erf GELU and the
 * residual to the fp32 accumulator before the one rounding to bf16. The identity instances stay in
 * ops/mxfp8_kernel.cu, unchanged; this translation unit exists so that the epilogue instances compile in parallel
 * with them rather than lengthening that one.
 *
 * A Stream-K route needs none of these: its slices run the identity instances and the reduce applies the epilogue
 * (mxfp8/streamk.cuh).
 */
// The launchers only: the dispatchers' bodies would instantiate the identity kernels here too (see the macro's note
// in mxfp8/dispatch.cuh).
#define FSO_SM120_MXFP8_LAUNCHERS_ONLY 1
#include "blockscale_gemm/arch/sm120/mxfp8/dispatch.cuh"

TRTLLM_NAMESPACE_BEGIN
namespace kernels::blockscale_gemm
{

bool launch_sm120_mxfp8_dense_epi_single(int tile_m, int tile_n, int stages, __nv_fp8_e4m3* mat_a,
    __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d, int32_t* scales_a, int32_t* scales_b, uint32_t shape_m,
    uint32_t shape_n, uint32_t shape_k, cudaStream_t stream, int num_device_sms, fso::DenseEpilogue const& epilogue)
{
    int64_t const ld_a = shape_k;
    int64_t const ld_b = shape_k;
    int64_t const ld_d = shape_n;
    constexpr int64_t stride = 0;
    constexpr uint32_t num_problems = 1;
#define EPI_MX(TM_, TN_, ST_)                                                                                          \
    if (tile_m == TM_ && tile_n == TN_ && stages == ST_)                                                               \
    {                                                                                                                  \
        launch_sm120_mxfp8_gemm_kernel<TM_, TN_, ST_, 1, 16, false, false, false, false, false, false, true>(mat_a,    \
            ld_a, stride, mat_b, ld_b, stride, mat_d, ld_d, stride, scales_a, stride, scales_b, stride, num_problems,  \
            shape_m, shape_n, shape_k, stream, num_device_sms, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,  \
            0, nullptr, 0, &epilogue);                                                                                 \
        return true;                                                                                                   \
    }
    // The TileM = 16 routes (tilem16_band, tilem16_narrow_single).
    EPI_MX(16, 128, 4)
    EPI_MX(16, 64, 4)
    // The overrides and the main cascade.
    EPI_MX(32, 128, 4)
    EPI_MX(32, 64, 4)
    EPI_MX(64, 64, 4)
    EPI_MX(64, 128, 2)
    EPI_MX(96, 128, 2)
    EPI_MX(128, 64, 2)
    EPI_MX(128, 128, 2)
    EPI_MX(160, 128, 2)
#undef EPI_MX
    return false;
}

} // namespace kernels::blockscale_gemm
TRTLLM_NAMESPACE_END
