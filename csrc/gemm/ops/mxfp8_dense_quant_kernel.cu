/*
 * The quantized-output instances of the sm_120/sm_121 MXFP8 GEMM: fso.dense.linear_quantized (interface v1, issue #12;
 * charter 7.1; mxfp8/dense_quant_epi.cuh).
 *
 * One instantiation per single-launch tile of the dense cascade (mxfp8/dispatch.cuh, sm120_mxfp8_dense_route: the main
 * cascade, its overrides and the TileM = 16 routes; the ten tiles of ops/mxfp8_dense_epi_kernel.cu), each the kernel
 * over SM120DenseQuantBuilder<SM120MxFP8BlockScaledBuilder<TileM, TileN, Stages>>: bias and the erf GELU on the fp32
 * accumulator, the one rounding to bf16, and the activation quantizer of contract v1 on the rounded tile, written as
 * xq [M, N] and its K-major scale words. Every tile keeps the shared-memory size of its bf16 form: the quantized form
 * reads the bf16 staging buffer smem_D, which aliases the A/B stages, and needs no buffer of its own.
 *
 * Also the Stream-K route's reduce (sm120_streamk_reduce_quant_kernel): its slices run the identity instances into
 * bf16 partials, and the reduce applies the epilogue and the quantizer once, after the sum.
 *
 * Their own translation unit, so that they compile in parallel with the identity and bf16-epilogue instances and so
 * that no other unit carries their code; the dispatch reaches them through the two functions below.
 */
// The launchers only: the dispatchers' bodies would instantiate the identity kernels here too (see the macro's note
// in mxfp8/dispatch.cuh).
#define FSO_SM120_MXFP8_LAUNCHERS_ONLY 1
#include "blockscale_gemm/arch/sm120/mxfp8/dispatch.cuh"

TRTLLM_NAMESPACE_BEGIN
namespace kernels::blockscale_gemm
{

bool launch_sm120_mxfp8_dense_quant_single(int tile_m, int tile_n, int stages, __nv_fp8_e4m3* mat_a,
    __nv_fp8_e4m3* mat_b, int32_t* scales_a, int32_t* scales_b, uint32_t shape_m, uint32_t shape_n,
    uint32_t shape_k, cudaStream_t stream, int num_device_sms, fso::DenseEpilogue const& epilogue)
{
    TLLM_CHECK_WITH_INFO(epilogue.out_q != nullptr && epilogue.out_scale != nullptr,
        "sm120 mxfp8 quantized-output launch without its output buffers");
    int64_t const ld_a = shape_k;
    int64_t const ld_b = shape_k;
    int64_t const ld_d = shape_n;
    constexpr int64_t stride = 0;
    constexpr uint32_t num_problems = 1;
    // The D descriptor is built over out_q and never issued (launch_sm120_mxfp8_gemm_kernel, DenseQuant).
    auto* const mat_d = reinterpret_cast<__nv_bfloat16*>(epilogue.out_q);
#define QUANT_MX(TM_, TN_, ST_)                                                                                        \
    if (tile_m == TM_ && tile_n == TN_ && stages == ST_)                                                               \
    {                                                                                                                  \
        launch_sm120_mxfp8_gemm_kernel<TM_, TN_, ST_, 1, 16, false, false, false, false, false, false, true, true>(    \
            mat_a, ld_a, stride, mat_b, ld_b, stride, mat_d, ld_d, stride, scales_a, stride, scales_b, stride,         \
            num_problems, shape_m, shape_n, shape_k, stream, num_device_sms, nullptr, nullptr, nullptr, nullptr,       \
            nullptr, nullptr, 0, nullptr, 0, &epilogue);                                                               \
        return true;                                                                                                   \
    }
    // The TileM = 16 routes (tilem16_band, tilem16_narrow_single).
    QUANT_MX(16, 128, 4)
    QUANT_MX(16, 64, 4)
    // The overrides and the main cascade.
    QUANT_MX(32, 128, 4)
    QUANT_MX(32, 64, 4)
    QUANT_MX(64, 64, 4)
    QUANT_MX(64, 128, 2)
    QUANT_MX(96, 128, 2)
    QUANT_MX(128, 64, 2)
    QUANT_MX(128, 128, 2)
    QUANT_MX(160, 128, 2)
#undef QUANT_MX
    return false;
}

void launch_sm120_mxfp8_streamk_reduce_quant(__nv_bfloat16 const* partials, int M, int N, int K_split,
    std::size_t partial_stride, fso::DenseEpilogue const& epilogue, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(epilogue.out_q != nullptr && epilogue.out_scale != nullptr && N % 128 == 0,
        "sm120 mxfp8 quantized Stream-K reduce: output buffers %p / %p, N = %d (a multiple of 128)",
        static_cast<void*>(epilogue.out_q), epilogue.out_scale, N);
    // One thread per 8 columns of the padded [align(M, 4), N] output.
    std::size_t const chunks = static_cast<std::size_t>((M + 3) / 4 * 4) * static_cast<std::size_t>(N) / 8;
    int const threads = 256;
    int const blocks = static_cast<int>((chunks + threads - 1) / threads);
    sm120_blockscaled_gemm::dense_quant::sm120_streamk_reduce_quant_kernel<><<<blocks, threads, 0, stream>>>(
        partials, epilogue.out_q, reinterpret_cast<int32_t*>(epilogue.out_scale), M, N, K_split, partial_stride,
        epilogue.bias, epilogue.act);
}

} // namespace kernels::blockscale_gemm
TRTLLM_NAMESPACE_END
