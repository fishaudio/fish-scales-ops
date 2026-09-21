/*
 * cute-using TU for the SM100/SM103 grouped (MoE, masked-layout) MXFP8 1x32
 * BlockScaled GEMM. Plain-C interface so the ATen wrapper TU does not get the
 * cute::Layout vs at::Layout name collision — same split as
 * mxfp8_sm100_kernel.cu (dense) and mxfp8_kernel.cu (sm_120).
 *
 * The grouped kernels are a separate translation unit from the dense ones
 * because each CUTLASS pointer-array instantiation is expensive to compile;
 * keeping them apart lets the two sets rebuild independently.
 *
 * Device code materialises only in the sm_100f (family) gencode pass; on
 * toolchains without SM100 support the launcher degrades to
 * cudaErrorNotSupported so the extension still links for sm_90-only and
 * sm_120-only builds.
 */
// Grid dependency control (GDC) for this translation unit only.
//
// CUTLASS guards every `griddepcontrol` instruction behind CUTLASS_GDC_ENABLED,
// which cutlass/arch/grid_dependency_control.h derives from
// CUTLASS_ENABLE_GDC_FOR_SM100. Turning it on is what makes the grouped GEMM
// safe to launch with a programmatic dependent launch: the pointer-array
// kernel then executes `griddepcontrol.wait` before its tile scheduler reads
// the device-resident per-group problem shapes that sm100_grouped_prep_kernel
// writes, and before every load participant's first global access.
//
// Two scoping rules matter and both are enforced here:
//   * the macro is defined only in the DEVICE passes for sm_100 / sm_103. The
//     second enabling block in grid_dependency_control.h would otherwise also
//     switch GDC on for __CUDA_ARCH__ 1200 and 1210, i.e. for the sm_120 /
//     sm_121 device pass of this file, changing code this subtask must leave
//     byte-identical.
//   * FSO_SM100_GROUPED_GDC is the host-visible twin, read by
//     grouped_dispatch.cuh to decide whether `launch_with_pdl` may be handed
//     to CUTLASS at all, so host and device always agree.
// Both macros are local to this file; no other translation unit sees either.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)
#define CUTLASS_ENABLE_GDC_FOR_SM100 1
#endif
#define FSO_SM100_GROUPED_GDC 1

#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#include "blockscale_gemm/arch/sm100/mxfp8/grouped_dispatch.cuh"
#endif

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace blockscale_gemm::detail
{

bool sm100_mxfp8_grouped_compiled()
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return true;
#else
    return false;
#endif
}

cudaError_t launch_sm100_mxfp8_grouped_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D, int32_t* SFA,
    int32_t* SFB, int32_t* masked_m, int num_groups, int m_cap, int N, int K, int expected_m, cudaStream_t stream)
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return sm100_blockscaled_gemm::gemm_dispatch_sm100_mxfp8_grouped(
        A, B, D, SFA, SFB, masked_m, num_groups, m_cap, N, K, expected_m, stream);
#else
    (void) A; (void) B; (void) D; (void) SFA; (void) SFB; (void) masked_m;
    (void) num_groups; (void) m_cap; (void) N; (void) K; (void) expected_m; (void) stream;
    return cudaErrorNotSupported;
#endif
}

} // namespace blockscale_gemm::detail
