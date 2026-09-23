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
    int32_t* SFB, int32_t* masked_m, int num_groups, int m_cap, int N, int K, int expected_m, int max_active_groups,
    int const* slot_to_expert, int32_t const* problem_shapes, cudaStream_t stream)
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return sm100_blockscaled_gemm::gemm_dispatch_sm100_mxfp8_grouped(A, B, D, SFA, SFB, masked_m, num_groups, m_cap,
        N, K, expected_m, max_active_groups, slot_to_expert, problem_shapes, stream);
#else
    (void) A; (void) B; (void) D; (void) SFA; (void) SFB; (void) masked_m;
    (void) num_groups; (void) m_cap; (void) N; (void) K; (void) expected_m;
    (void) max_active_groups; (void) slot_to_expert; (void) problem_shapes; (void) stream;
    return cudaErrorNotSupported;
#endif
}

// The fused-SwiGLU FC1: one grouped GEMM whose epilogue writes
// MXFP8(silu(gate) * up) instead of the bf16 gate_up tensor. See the comment
// on `Sm100MxFP8GroupedSwiGluGemmConfig` for the interleaved-weight
// precondition; the ATen op documents it for callers.
cudaError_t launch_sm100_mxfp8_grouped_swiglu_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_fp8_e4m3* H,
    int32_t* SFH, int32_t* SFA, int32_t* SFB, int32_t* masked_m, int num_groups, int m_cap, int N, int K,
    int expected_m, int max_active_groups, int const* slot_to_expert, int32_t const* problem_shapes,
    cudaStream_t stream)
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return sm100_blockscaled_gemm::gemm_dispatch_sm100_mxfp8_grouped_swiglu(A, B, H, SFH, SFA, SFB, masked_m,
        num_groups, m_cap, N, K, expected_m, max_active_groups, slot_to_expert, problem_shapes, stream);
#else
    (void) A; (void) B; (void) H; (void) SFH; (void) SFA; (void) SFB; (void) masked_m;
    (void) num_groups; (void) m_cap; (void) N; (void) K; (void) expected_m;
    (void) max_active_groups; (void) slot_to_expert; (void) problem_shapes; (void) stream;
    return cudaErrorNotSupported;
#endif
}

// Whether a grouped call of this shape would READ a caller-supplied
// `problem_shapes` tensor (run b300_round3_20260922/M-A4): 1 on the
// pointer-array cascade, 0 on the slot route, which derives its grid from
// `masked_m` and the slot list. The ATen layer exposes it so a caller asks
// `moe_build_routing` for the shapes exactly where a GEMM will consume them.
// `fused_swiglu` names the kernel, as for `sm100_mxfp8_grouped_slot_taken`
// below: the two answers are complements only under the same flag.
int sm100_mxfp8_grouped_problem_shapes_consumed(
    int m_cap, int N, int K, int groups, int max_active_groups, int fused_swiglu)
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return sm100_blockscaled_gemm::grouped_detail::problem_shapes_consumed(
               m_cap, N, K, groups, max_active_groups, fused_swiglu != 0)
        ? 1
        : 0;
#else
    (void) m_cap; (void) N; (void) K; (void) groups; (void) max_active_groups; (void) fused_swiglu;
    return 0;
#endif
}

// Whether a caller whose FC1 is this (m_cap, N_w, K, G, max_active_groups)
// should use the fused FC1 or keep the old pair (unfused FC1 + the separate
// SwiGLU kernel).
//
// The choice is not the caller's taste: the fused FC1 exists only on the
// pointer-array route, so wherever the dispatcher would take the slot route the
// layer must keep the old pair. The rule lives with `slot_route` in the
// dispatcher and this only exposes its verdict to the ATen layer, so the
// bench, the tests and any future layer op all read the same decision.
// Returns 1 for "use the fused FC1", 0 for "keep the old pair".
int sm100_mxfp8_grouped_fused_fc1_route(int m_cap, int N, int K, int groups, int max_active_groups)
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return sm100_blockscaled_gemm::grouped_detail::fused_fc1_route(m_cap, N, K, groups, max_active_groups) ? 1 : 0;
#else
    (void) m_cap; (void) N; (void) K; (void) groups; (void) max_active_groups;
    return 0;
#endif
}

// The load-time half: may this (N_w, K) use the fused FC1 on any M? A caller
// needs it before it quantises its weights, because the interleaved row order
// is one decision for the whole model while the route is a per-call one.
int sm100_mxfp8_grouped_fused_fc1_available(int N, int K)
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return sm100_blockscaled_gemm::grouped_detail::fused_fc1_available(N, K) ? 1 : 0;
#else
    (void) N; (void) K;
    return 0;
#endif
}

// Host-side guard query for `linear_mxfp8_grouped_masked`.
//
// The op has to be able to refuse a forced-but-illegal slot-route call with a
// readable message BEFORE any kernel runs, because outside the guard the slot
// kernel does not fail: it returns 2^-127 times the right answer, or NaN, with
// no error of any kind. The decision itself stays in one place (the
// dispatcher's `slot_route`); this only exposes its verdict to the ATen layer.
// The return value is `sm100_blockscaled_gemm::grouped_detail::SlotRouteDecision`
// as an int: 0/1 mean the call may proceed, 2 the row capacity exceeds the
// token tile, 3 no instantiation covers (N, K), 4 no slot bound was supplied.
int sm100_mxfp8_grouped_slot_refusal(int m_cap, int N, int K, int groups, int max_active_groups)
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return static_cast<int>(
        sm100_blockscaled_gemm::grouped_detail::slot_route(m_cap, N, K, groups, max_active_groups));
#else
    (void) m_cap; (void) N; (void) K; (void) groups; (void) max_active_groups;
    return 0;
#endif
}

// Whether the dispatcher WOULD take the slot-bound decode route for this
// (m_cap, N, K, G, max_active_groups).
//
// The slot route is the only consumer of the packed active-expert list that
// `moe_build_routing(..., with_slots=True)` emits, so a caller that knows the
// answer in advance can leave the list unbuilt wherever nothing would read it.
// The list is not free: the routing kernel pays a block-wide scan over the
// per-expert histogram to compact it, and outside the decode band that scan
// produces a tensor every kernel ignores.
//
// Returns 1 only for `kSlotRouteSlot`. Every refusal code answers 0 as well,
// because a refused forced call does not reach the slot kernel either. The
// enum lives in the dispatcher header, which only this translation unit
// includes, so the comparison stays here rather than in the ATen layer.
// `fused_swiglu` names the kernel the call would land on — the fused-SwiGLU FC1
// (`linear_mxfp8_grouped_masked_swiglu`) or the plain grouped GEMM — because
// the two have different row-capacity clauses (run b300_round3_20260922/M-A3).
int sm100_mxfp8_grouped_slot_taken(int m_cap, int N, int K, int groups, int max_active_groups, int fused_swiglu)
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return sm100_blockscaled_gemm::grouped_detail::slot_route(
               m_cap, N, K, groups, max_active_groups, fused_swiglu != 0)
            == sm100_blockscaled_gemm::grouped_detail::kSlotRouteSlot
        ? 1
        : 0;
#else
    (void) m_cap; (void) N; (void) K; (void) groups; (void) max_active_groups; (void) fused_swiglu;
    return 0;
#endif
}

// The token-tile width of the slot route's phase-1 instantiation, for the
// message the op prints when the guard refuses a call.
int sm100_mxfp8_grouped_slot_tile_n()
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return sm100_blockscaled_gemm::grouped_detail::slot_kernel_tile_n();
#else
    return 0;
#endif
}

} // namespace blockscale_gemm::detail
