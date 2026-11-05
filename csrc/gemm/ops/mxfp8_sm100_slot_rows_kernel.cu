/*
 * cute-using TU for the SM100/SM103 compact slot GEMMs with the row-bounded token load (issue #13 lever 3):
 * FsoSm100SlotGemmCompact / FsoSm100SlotGemmCompactList instantiated with FsoRowsBoundedMainloop
 * (blockscale_gemm/arch/sm100/mxfp8/slot_rows_mainloop.cuh), which loads a tile's token operand as one box of 8, 16 or
 * 32 rows covering the expert's rows in the tile instead of the whole 64-row token tile.
 *
 * Its own translation unit, for the reason of run fso_issue12_20261028/R1: the driver re-finalises an sm_100f kernel
 * for sm_103 from the cubin's Mercury form, and the result depends on the device-code set of the unit, so adding
 * these kernels to mxfp8_sm100_slot_kernel.cu would not leave that unit's kernels -- the ones every launch below a
 * row capacity of 32 runs -- executing the code they executed before. This unit defines FSO_SLOT_ROWS_TU, under which
 * grouped_slot_dispatch.cuh compiles only the row-bounded compact entries and declares (but does not define) the
 * list builders; the slot unit's compact entries hand a launch here when sm100_mxfp8_slot_rows_applies (tile-list
 * mode, and slot mode at m_cap >= 32, with the routing op's lists given). Same GDC scoping as the slot unit.
 *
 * Device code materialises only in the sm_100f (family) gencode pass.
 */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)
#define CUTLASS_ENABLE_GDC_FOR_SM100 1
#endif
#define FSO_SM100_SLOT_GDC 1
#define FSO_SLOT_ROWS_TU 1

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

cudaError_t slot_kernel_launch_compact_rows(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int32_t* masked_m, int32_t const* blk0, int groups, int num_slots,
    int m_cap, int b_alloc, int shape_n, int shape_k, int const* slot_to_expert, int* slot_sched, int* slot_tiles,
    int tiles_cap, bool build_tiles, cudaStream_t stream)
{
    return gemm_dispatch_sm100_mxfp8_slot_compact_rows(mat_a, mat_b, mat_d, scales_a, scales_b, masked_m, blk0, groups,
        num_slots, m_cap, b_alloc, shape_n, shape_k, slot_to_expert, slot_sched, slot_tiles, tiles_cap, build_tiles,
        stream);
}

cudaError_t slot_swiglu_kernel_launch_compact_rows(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_fp8_e4m3* out_h,
    int32_t* out_sfh, int32_t* scales_a, int32_t* scales_b, int32_t* masked_m, int32_t const* blk0, int groups,
    int num_slots, int m_cap, int b_alloc, int shape_n, int shape_k, int const* slot_to_expert, int* slot_sched,
    int* slot_tiles, int tiles_cap, bool build_tiles, cudaStream_t stream)
{
    return gemm_dispatch_sm100_mxfp8_slot_swiglu_compact_rows(mat_a, mat_b, out_h, out_sfh, scales_a, scales_b,
        masked_m, blk0, groups, num_slots, m_cap, b_alloc, shape_n, shape_k, slot_to_expert, slot_sched, slot_tiles,
        tiles_cap, build_tiles, stream);
}

} // namespace grouped_detail
} // namespace sm100_blockscaled_gemm

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
