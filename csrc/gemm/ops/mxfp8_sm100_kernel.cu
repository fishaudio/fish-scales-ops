/*
 * cute-using TU for the SM100/SM103 MXFP8 (1×32, SFVecSize=32) BlockScaled
 * GEMM. Plain-C interface so the ATen wrapper TU doesn't get cute::Layout
 * vs at::Layout name collision — same split as mxfp8_kernel.cu (sm_120).
 *
 * Device code materialises only in the sm_100f (family) gencode pass; on
 * toolchains without SM100 support (< CUDA 12.8) the launcher degrades to
 * cudaErrorNotSupported so the extension still links.
 */
// Grid dependency control (GDC) for the dense GEMMs of this translation unit (2026-10-11).
//
// CUTLASS guards every `griddepcontrol` instruction behind CUTLASS_GDC_ENABLED, which
// cutlass/arch/grid_dependency_control.h derives from CUTLASS_ENABLE_GDC_FOR_SM100. With it compiled in, the
// dense sm_100 kernel executes `griddepcontrol.wait` in its load and scheduler participants before their
// first global access, and `griddepcontrol.launch_dependents` once its MMA warp has issued the last MMA, so
// the dense GEMM may be launched as a programmatic dependent of the kernel before it (the activation
// quantize in front of it) and may release the kernel after it (the SwiGLU quantize, the split-K reduce)
// before its epilogue drains. That is what makes `launch_with_pdl` safe on these kernels; without the macro
// the instructions compile to nothing and a PDL launch would read its input early.
//
// Scoping, as in mxfp8_sm100_grouped_kernel.cu: the macro is defined only in the DEVICE passes for sm_100 /
// sm_103 (the second enabling block in grid_dependency_control.h would otherwise also switch GDC on for the
// sm_120 / sm_121 passes of this file), and FSO_SM100_DENSE_GDC is the host-visible twin dispatch.cuh reads
// before it hands `launch_with_pdl` to CUTLASS. Both macros are local to this file.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)
#define CUTLASS_ENABLE_GDC_FOR_SM100 1
#endif
#define FSO_SM100_DENSE_GDC 1

#include "cutlass/arch/config.h"
#include "blockscale_gemm/common/epilogue.cuh"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#include "blockscale_gemm/arch/sm100/mxfp8/dispatch.cuh"
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstddef>
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

// `epilogue` (common/epilogue.cuh): the dispatch throws fso::DenseEpilogueNotImplemented for a field or a forced
// configuration its route does not serve, before it launches anything; the op bodies (ops/mxfp8.cu, ops/fp8.cu)
// turn that into NotImplementedError. `workspace` / `workspace_bytes`: the caller's device scratch for this call,
// at least launch_sm100_mxfp8_dispatch_workspace_bytes(M, N, K, epilogue) bytes, allocated on `stream` (issue #12,
// charter 7.5: the torch caching allocator on the caller's stream).
cudaError_t launch_sm100_mxfp8_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int M, int N, int K, cudaStream_t stream, fso::DenseEpilogue const& epilogue,
    void* workspace, std::size_t workspace_bytes)
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    sm100_blockscaled_gemm::Sm100DenseWorkspace ws;
    ws.ptr = workspace;
    ws.bytes = workspace_bytes;
    return sm100_blockscaled_gemm::gemm_dispatch_sm100_mxfp8(A, B, D, SFA, SFB, M, N, K, stream, epilogue, ws);
#else
    (void) A; (void) B; (void) D; (void) SFA; (void) SFB;
    (void) M; (void) N; (void) K; (void) stream; (void) epilogue; (void) workspace; (void) workspace_bytes;
    return cudaErrorNotSupported;
#endif
}

// The device scratch launch_sm100_mxfp8_dispatch needs for this call, in bytes (0 for every plain tile; the FP32
// partials for the two-kernel split-K). Host-only. Throws fso::DenseEpilogueNotImplemented exactly when the
// dispatch would, so an op body that asks first refuses before it allocates or launches anything.
std::size_t launch_sm100_mxfp8_dispatch_workspace_bytes(int M, int N, int K, fso::DenseEpilogue const& epilogue)
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    return sm100_blockscaled_gemm::gemm_workspace_sm100_mxfp8(M, N, K, epilogue);
#else
    (void) M; (void) N; (void) K; (void) epilogue;
    return 0;
#endif
}

// Guidance appended to the op's error message for the one launch error a
// caller can act on: a forced Stream-K variant (FSO_FORCE_TILE codes 7 / 8,
// sweep-only) needed its per-thread workspace to grow inside a CUDA-graph
// capture (Sm100WorkspacePool::ensure in arch/sm100/mxfp8/dispatch.cuh refuses
// that). The dense split-K of the cascade takes its workspace from the caller
// and never raises this. Empty for every other error.
char const* sm100_mxfp8_error_hint(cudaError_t err)
{
    if (err == cudaErrorStreamCaptureUnsupported)
        return ": the sm_100/103 dense GEMM forced to a Stream-K variant (FSO_FORCE_TILE codes 7 / 8) needed a "
               "larger workspace inside a CUDA-graph capture, where it cannot allocate. Run the same call once "
               "eagerly on this thread before capturing it; graphs captured earlier stay valid when the workspace "
               "grows.";
    return "";
}

} // namespace blockscale_gemm::detail
