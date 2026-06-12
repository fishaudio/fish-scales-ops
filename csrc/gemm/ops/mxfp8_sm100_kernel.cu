/*
 * cute-using TU for the SM100/SM103 MXFP8 (1×32, SFVecSize=32) BlockScaled
 * GEMM. Plain-C interface so the ATen wrapper TU doesn't get cute::Layout
 * vs at::Layout name collision — same split as mxfp8_kernel.cu (sm_120).
 *
 * Device code materialises only in the sm_100f (family) gencode pass; on
 * toolchains without SM100 support (< CUDA 12.8) the launcher degrades to
 * cudaErrorNotSupported so the extension still links.
 */
#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#include "blockscale_gemm/arch/sm100/mxfp8/dispatch.cuh"
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace blockscale_gemm::detail
{

bool sm100_mxfp8_compiled()
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return true;
#else
    return false;
#endif
}

cudaError_t launch_sm100_mxfp8_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int M, int N, int K, cudaStream_t stream)
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return sm100_blockscaled_gemm::gemm_dispatch_sm100_mxfp8(A, B, D, SFA, SFB, M, N, K, stream);
#else
    (void) A; (void) B; (void) D; (void) SFA; (void) SFB;
    (void) M; (void) N; (void) K; (void) stream;
    return cudaErrorNotSupported;
#endif
}

} // namespace blockscale_gemm::detail
