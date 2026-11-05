/*
 * The reduce kernel of the sm_100/sm_103 dense two-kernel split-K when the call carries the dense epilogue (bias,
 * erf-GELU, residual; issue #12 interface v1), and its host launcher. The identity-epilogue reduce stays next to
 * the identity GEMMs in mxfp8_sm100_kernel.cu (dispatch.cuh); this one lives in its own translation unit on purpose:
 *
 * The dispatch TU's cubin must keep exactly the device code it had before the dense epilogue (issue #12, R1):
 * on sm_100/sm_103 with CUDA 13.0 the driver re-finalises a kernel from the cubin's Mercury form when it loads the
 * module, and the machine code it emits depends on the cubin's layout (symbol table, relocation targets, the
 * kernel's position), not only on the kernel's own SASS. Adding this kernel to the identity TU re-finalised the
 * unchanged (256,128)K256 identity GEMM with sixteen extra NOPs and a schedule that runs 11-18 % slower on the
 * B300 (runs/R1 of fso_issue12_20261028). A separate TU leaves the identity cubin alone.
 *
 * Declared in blockscale_gemm/arch/sm100/mxfp8/dense_common.cuh (launch_sm100_mxfp8_splitk_reduce_epi).
 */
#include "cutlass/arch/config.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#include "blockscale_gemm/common/epilogue.cuh"
#include "blockscale_gemm/arch/sm100/mxfp8/dense_common.cuh"

namespace sm100_blockscaled_gemm
{

// The reduce of a split-K call that carries the dense epilogue (issue #12; common/epilogue.cuh): the same sum of
// the partials, in the same order, as sm100_mxfp8_splitk_reduce_kernel (dispatch.cuh), then the epilogue applied
// once to that sum (v = acc; v += bias_term(bias[n]); gelu if act; v += residual[m, n] if residual) and one rounding
// to bf16. The batched GEMM in front of it still writes plain FP32 partials, so no split ever sees the bias or the
// residual. Bias and residual are read only after the grid-dependency wait (sm100_pdl_wait, the first statement):
// with PDL their producer may be the kernel before the GEMM, and the wait returns only once the GEMM, which itself
// waited on that kernel, has completed. A null bias is no bias; a null residual is no residual.
// N % 4 == 0 and every element group of four lies in one row (N is a multiple of 128 on this surface).
//
// Every global read here is a coherent one (__ldcg: ld.global.cg, cached in L2 only), never the non-coherent
// read-only path: under PDL this grid is created while the GEMM in front of it (the partials' producer) still runs,
// and bias / residual may come from the kernel before that, so the read-only contract (no writer during the
// kernel's lifetime) does not hold even though every read follows the grid-dependency wait.
static __global__ void sm100_mxfp8_splitk_reduce_epi_kernel(float4 const* partials, ushort4* __restrict__ out,
    int splits, long mn4, int n4, __nv_bfloat16 const* bias, int act, __nv_bfloat16 const* residual,
    long ld_residual, bool pdl)
{
    sm100_pdl_wait(pdl);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)
    // As in sm100_mxfp8_splitk_reduce_kernel: a programmatic dependent still waits for this grid's completion.
    if (pdl && threadIdx.x == 0)
        cudaTriggerProgrammaticLaunchCompletion();
#endif
    long const stride = static_cast<long>(gridDim.x) * blockDim.x;
    for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; i < mn4; i += stride)
    {
        float4 acc = __ldcg(partials + i);
        for (int s = 1; s < splits; ++s)
        {
            float4 const p = __ldcg(partials + s * mn4 + i);
            acc.x += p.x; acc.y += p.y; acc.z += p.z; acc.w += p.w;
        }
        long const m = i / n4;
        int const n = static_cast<int>(i - m * n4) * 4;
        float b[4] = {0.f, 0.f, 0.f, 0.f};
        if (bias != nullptr)
        {
            b[0] = __bfloat162float(__ldcg(bias + n + 0));
            b[1] = __bfloat162float(__ldcg(bias + n + 1));
            b[2] = __bfloat162float(__ldcg(bias + n + 2));
            b[3] = __bfloat162float(__ldcg(bias + n + 3));
        }
        float const a[4] = {acc.x, acc.y, acc.z, acc.w};
        float r[4] = {0.f, 0.f, 0.f, 0.f};
        if (residual != nullptr)
        {
            ushort4 const rv = __ldcg(reinterpret_cast<ushort4 const*>(residual + m * ld_residual + n));
            r[0] = __bfloat162float(__ushort_as_bfloat16(rv.x));
            r[1] = __bfloat162float(__ushort_as_bfloat16(rv.y));
            r[2] = __bfloat162float(__ushort_as_bfloat16(rv.z));
            r[3] = __bfloat162float(__ushort_as_bfloat16(rv.w));
        }
        // The activation and the residual are tested once per group of four, not inside every element's formula.
        float v[4];
        if (act == fso::kDenseActGeluErf)
        {
            if (residual != nullptr)
                for (int j = 0; j < 4; ++j)
                    v[j] = fso::dense_epilogue_value_rt<true>(a[j], b[j], fso::kDenseActGeluErf, r[j]);
            else
                for (int j = 0; j < 4; ++j)
                    v[j] = fso::dense_epilogue_value_rt<false>(a[j], b[j], fso::kDenseActGeluErf, 0.f);
        }
        else
        {
            if (residual != nullptr)
                for (int j = 0; j < 4; ++j)
                    v[j] = fso::dense_epilogue_value_rt<true>(a[j], b[j], fso::kDenseActNone, r[j]);
            else
                for (int j = 0; j < 4; ++j)
                    v[j] = fso::dense_epilogue_value_rt<false>(a[j], b[j], fso::kDenseActNone, 0.f);
        }
        ushort4 o;
        o.x = __bfloat16_as_ushort(__float2bfloat16_rn(v[0]));
        o.y = __bfloat16_as_ushort(__float2bfloat16_rn(v[1]));
        o.z = __bfloat16_as_ushort(__float2bfloat16_rn(v[2]));
        o.w = __bfloat16_as_ushort(__float2bfloat16_rn(v[3]));
        out[i] = o;
    }
}

// The launch, with the same programmatic-dependent attribute the identity reduce uses (dispatch.cuh): the reduce
// becomes the GEMM's programmatic dependent and waits for the GEMM's completion before it reads a partial, a bias
// or a residual. `pdl` false launches it plainly (FSO_DISABLE_PDL=1).
cudaError_t launch_sm100_mxfp8_splitk_reduce_epi(float const* partials, __nv_bfloat16* out, int splits, long mn4,
    int n4, __nv_bfloat16 const* bias, int act, __nv_bfloat16 const* residual, long ld_residual, bool pdl,
    int blocks, int threads, cudaStream_t stream)
{
    if (pdl)
    {
        cudaLaunchConfig_t cfg{};
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.gridDim = dim3(blocks);
        cfg.blockDim = dim3(threads);
        cfg.dynamicSmemBytes = 0;
        cfg.stream = stream;
        cfg.attrs = attrs;
        cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, sm100_mxfp8_splitk_reduce_epi_kernel, reinterpret_cast<float4 const*>(partials),
            reinterpret_cast<ushort4*>(out), splits, mn4, n4, bias, act, residual, ld_residual, true);
    }
    else
    {
        sm100_mxfp8_splitk_reduce_epi_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<float4 const*>(partials),
            reinterpret_cast<ushort4*>(out), splits, mn4, n4, bias, act, residual, ld_residual, false);
    }
    return cudaGetLastError();
}

} // namespace sm100_blockscaled_gemm

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
