/*
 * Isolated translation unit for the SM100/SM103 unified persistent FC1 + FC2
 * MoE kernel (drama3 unified architecture).
 *
 * Separate TU preserves upstream cubins byte-for-byte (R1 Mercury rule).
 */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)
#define CUTLASS_ENABLE_GDC_FOR_SM100 1
#endif
#define FSO_SM100_MOE_GDC 1

#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#include "blockscale_gemm/arch/sm100/mxfp8/unified_moe_dispatch.cuh"
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace blockscale_gemm::detail
{

cudaError_t launch_sm100_mxfp8_unified_moe(__nv_fp8_e4m3 const* a, int32_t const* sa, __nv_fp8_e4m3 const* w13,
    int32_t const* sw13, __nv_fp8_e4m3 const* w2, int32_t const* sw2, __nv_fp8_e4m3* dq, int32_t* sd,
    __nv_bfloat16* dn, int32_t const* masked_m, int32_t const* blk0, int const* slot_to_expert, int* counters,
    int const* slot_tiles, int tiles_cap, int groups, int b_alloc, int n1, int k1, int n2, int k2,
    cudaStream_t stream)
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return sm100_blockscaled_gemm::gemm_dispatch_sm100_mxfp8_unified_moe(a, sa, w13, sw13, w2, sw2, dq, sd, dn,
        masked_m, blk0, slot_to_expert, counters, slot_tiles, tiles_cap, groups, b_alloc, n1, k1, n2, k2, stream);
#else
    (void) a; (void) sa; (void) w13; (void) sw13; (void) w2; (void) sw2; (void) dq; (void) sd; (void) dn;
    (void) masked_m; (void) blk0; (void) slot_to_expert; (void) counters; (void) slot_tiles; (void) tiles_cap;
    (void) groups; (void) b_alloc; (void) n1; (void) k1; (void) n2; (void) k2;
    (void) stream;
    return cudaErrorNotSupported;
#endif
}

} // namespace blockscale_gemm::detail
