/*
 * cute-using TU for the SM100/SM103 persistent FC1 + FC2 slot kernel of the MoE layer (issue #13 lever 5,
 * `FsoSm100SlotGemmMoe`; blockscale_gemm/arch/sm100/mxfp8/grouped_slot_moe_dispatch.cuh): one grid that runs the
 * fused-SwiGLU FC1 tiles and then the FC2 tiles of one layer call on the compact slot route.
 *
 * Its own translation unit, for the reason of run fso_issue12_20261028/R1: the driver re-finalises an sm_100f kernel
 * for sm_103 from the cubin's Mercury form, and the result depends on the device-code set of the unit, so a new
 * kernel family added to mxfp8_sm100_slot_kernel.cu or mxfp8_sm100_slot_rows_kernel.cu would not leave those units'
 * kernels executing the code they executed before. It includes neither grouped_slot_dispatch.cuh (whose inline entry
 * points would instantiate today's slot kernels here) nor anything but the fused kernel's own headers and the two
 * generated slot headers. GDC is scoped exactly as in the slot units: CUTLASS_ENABLE_GDC_FOR_SM100 only in the device
 * passes for sm_100 / sm_103, FSO_SM100_SLOT_MOE_GDC as its host-visible twin.
 *
 * Device code materialises only in the sm_100f (family) gencode pass.
 */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)
#define CUTLASS_ENABLE_GDC_FOR_SM100 1
#endif
#define FSO_SM100_SLOT_MOE_GDC 1

#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#include "blockscale_gemm/arch/sm100/mxfp8/grouped_slot_moe_dispatch.cuh"
#endif

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace blockscale_gemm::detail
{

// Whether this build carries the fused FC1 + FC2 slot kernel (CUDA >= 12.8 with the sm_100 family target).
bool sm100_mxfp8_slot_moe_compiled()
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return true;
#else
    return false;
#endif
}

bool sm100_mxfp8_slot_moe_shape_ok(int n1, int k1, int n2, int k2)
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return sm100_blockscaled_gemm::sm100_mxfp8_slot_moe_instantiated(n1, k1, n2, k2);
#else
    (void) n1; (void) k1; (void) n2; (void) k2;
    return false;
#endif
}

cudaError_t launch_sm100_mxfp8_slot_moe(__nv_fp8_e4m3 const* a, int32_t const* sa, __nv_fp8_e4m3 const* w13,
    int32_t const* sw13, __nv_fp8_e4m3 const* w2, int32_t const* sw2, __nv_fp8_e4m3* dq, int32_t* sd,
    __nv_bfloat16* dn, int32_t const* masked_m, int32_t const* blk0, int const* slot_to_expert, int* counters,
    int groups, int num_slots, int m_cap, int b_alloc, int n1, int k1, int n2, int k2, cudaStream_t stream)
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return sm100_blockscaled_gemm::gemm_dispatch_sm100_mxfp8_slot_moe(a, sa, w13, sw13, w2, sw2, dq, sd, dn, masked_m,
        blk0, slot_to_expert, counters, groups, num_slots, m_cap, b_alloc, n1, k1, n2, k2, stream);
#else
    (void) a; (void) sa; (void) w13; (void) sw13; (void) w2; (void) sw2; (void) dq; (void) sd; (void) dn;
    (void) masked_m; (void) blk0; (void) slot_to_expert; (void) counters; (void) groups; (void) num_slots;
    (void) m_cap; (void) b_alloc; (void) n1; (void) k1; (void) n2; (void) k2; (void) stream;
    return cudaErrorNotSupported;
#endif
}

} // namespace blockscale_gemm::detail
