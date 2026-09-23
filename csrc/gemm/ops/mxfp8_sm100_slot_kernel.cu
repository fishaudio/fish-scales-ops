/*
 * cute-using TU for the SM100/SM103 slot-bound grouped (MoE, masked-layout)
 * MXFP8 1x32 BlockScaled GEMM — the decode-band second entry of the grouped
 * cascade. Split from `mxfp8_sm100_grouped_kernel.cu` for two reasons: it
 * compiles a different CUTLASS kernel (the DENSE block-scaled kernel, forked by
 * the build-time generator, rather than the pointer-array one), and that forked
 * kernel comes from a header generated into the build directory, whose include
 * path is added for this translation unit only.
 *
 * Device code materialises only in the sm_100f (family) gencode pass; on
 * toolchains without SM100 support the whole route is absent and
 * grouped_dispatch.cuh never refers to it.
 */
// Grid dependency control (GDC) for this translation unit only, scoped exactly
// as in mxfp8_sm100_grouped_kernel.cu.
//
// CUTLASS guards every `griddepcontrol` instruction behind CUTLASS_GDC_ENABLED,
// which cutlass/arch/grid_dependency_control.h derives from
// CUTLASS_ENABLE_GDC_FOR_SM100. The forked kernel then executes
// `griddepcontrol.wait` in each load participant before its first global access
// and `griddepcontrol.launch_dependents` at the end of the mainloop, which is
// what lets the consumer that follows the GEMM start early.
//
// Two scoping rules matter and both are enforced here:
//   * the macro is defined only in the DEVICE passes for sm_100 / sm_103. The
//     second enabling block in grid_dependency_control.h would otherwise also
//     switch GDC on for __CUDA_ARCH__ 1200 and 1210, i.e. for the sm_120 /
//     sm_121 device pass of this file.
//   * FSO_SM100_SLOT_GDC is the host-visible twin, read by
//     grouped_slot_dispatch.cuh to decide whether the programmatic
//     stream-serialisation launch attribute may be set at all, so host and
//     device always agree.
// Both macros are local to this file; no other translation unit sees either.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)
#define CUTLASS_ENABLE_GDC_FOR_SM100 1
#endif
#define FSO_SM100_SLOT_GDC 1

#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#include "blockscale_gemm/arch/sm100/mxfp8/grouped_slot_dispatch.cuh"
#endif

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

namespace sm100_blockscaled_gemm
{
namespace grouped_detail
{

// The three entry points `grouped_dispatch.cuh` declares. They are defined here
// rather than in a header because the slot route's kernel header only exists in
// the build directory of this translation unit; the dispatcher has to be able
// to decide legality and route the call without ever seeing it.

int slot_kernel_tile_n()
{
    return kSlotTileN;
}

bool slot_kernel_instantiated(int shape_n, int shape_k)
{
    return sm100_mxfp8_slot_instantiated(shape_n, shape_k);
}

cudaError_t slot_kernel_launch(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d, int32_t* scales_a,
    int32_t* scales_b, int32_t* masked_m, int groups, int num_slots, int m_cap, int shape_n, int shape_k,
    int const* slot_to_expert, cudaStream_t stream)
{
    return gemm_dispatch_sm100_mxfp8_slot(mat_a, mat_b, mat_d, scales_a, scales_b, masked_m, groups, num_slots, m_cap,
        shape_n, shape_k, slot_to_expert, stream);
}

// The fused-SwiGLU variant of the same route: the GEMM's epilogue emits
// MXFP8(silu(gate) * up) directly, so the decode band stops launching the
// separate SwiGLU-and-requantise kernel and stops writing the bf16
// [G, m_cap, 2*I] intermediate (run b300_mxfp8_20260917/M-A2).

bool slot_swiglu_kernel_instantiated(int shape_n, int shape_k)
{
    return sm100_mxfp8_slot_swiglu_instantiated(shape_n, shape_k);
}

// The fused epilogue's token-column chunk, read from the store itself so the
// dispatcher's rule cannot drift from the kernel: `slot_route` admits the fused
// slot kernel only while m_cap fits one chunk (run b300_round3_20260922/M-A3).
int slot_swiglu_epilogue_chunk()
{
    return cutlass::epilogue::collective::fso_swiglu_slot::kSlotChunk;
}

cudaError_t slot_swiglu_kernel_launch(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_fp8_e4m3* out_h,
    int32_t* out_sfh, int32_t* scales_a, int32_t* scales_b, int32_t* masked_m, int groups, int num_slots, int m_cap,
    int shape_n, int shape_k, int const* slot_to_expert, cudaStream_t stream)
{
    return gemm_dispatch_sm100_mxfp8_slot_swiglu(mat_a, mat_b, out_h, out_sfh, scales_a, scales_b, masked_m, groups,
        num_slots, m_cap, shape_n, shape_k, slot_to_expert, stream);
}

} // namespace grouped_detail
} // namespace sm100_blockscaled_gemm

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
