/*
 * The instantiations of the sm_120 small-M route's kernel (blockscale_gemm/arch/sm120/mxfp8/smallm_tc.cuh) for one
 * (form, epilogue) pair: FSO_SMALLM_FORM (kSmallmFormQ = 0, kSmallmFormBf16 = 1) and FSO_SMALLM_EPI (0, 1), set by
 * the including translation unit. Every (nc, rt, wk) the plan can pick: nc 1..8, rt 1 or 2, wk 4 or 8, as the
 * resident kernel; the chunked kernel (cj > 0) for wk == 8 only, the only width config() produces chunked.
 */
#include "blockscale_gemm/arch/sm120/mxfp8/smallm_tc.cuh"

#include <cuda_runtime.h>

#ifndef FSO_SMALLM_FORM
#error "FSO_SMALLM_FORM must be defined by the including translation unit"
#endif
#ifndef FSO_SMALLM_EPI
#error "FSO_SMALLM_EPI must be defined by the including translation unit"
#endif

namespace tensorrt_llm::kernels::blockscale_gemm::sm120_smallm
{
namespace
{

using KernelFn = void (*)(SmallmParams);

template <int kForm, bool kEpi, int kNc, int kRt, int kWk, bool kChunked>
constexpr KernelFn kernel_of()
{
    return &sm120_smallm_kernel<kForm, kNc, kRt, kWk, kEpi, kChunked>;
}

template <int kForm, bool kEpi, int kRt, int kWk, bool kChunked>
KernelFn pick_nc(int nc)
{
    switch (nc)
    {
    case 1: return kernel_of<kForm, kEpi, 1, kRt, kWk, kChunked>();
    case 2: return kernel_of<kForm, kEpi, 2, kRt, kWk, kChunked>();
    case 3: return kernel_of<kForm, kEpi, 3, kRt, kWk, kChunked>();
    case 4: return kernel_of<kForm, kEpi, 4, kRt, kWk, kChunked>();
    case 5: return kernel_of<kForm, kEpi, 5, kRt, kWk, kChunked>();
    case 6: return kernel_of<kForm, kEpi, 6, kRt, kWk, kChunked>();
    case 7: return kernel_of<kForm, kEpi, 7, kRt, kWk, kChunked>();
    case 8: return kernel_of<kForm, kEpi, 8, kRt, kWk, kChunked>();
    default: return nullptr;
    }
}

template <int kForm, bool kEpi>
KernelFn pick(int nc, int rt, int wk, bool chunked)
{
    if (chunked)
    {
        if (rt == 1 && wk == 8)
            return pick_nc<kForm, kEpi, 1, 8, true>(nc);
        if (rt == 2 && wk == 8)
            return pick_nc<kForm, kEpi, 2, 8, true>(nc);
        return nullptr;
    }
    if (rt == 1 && wk == 8)
        return pick_nc<kForm, kEpi, 1, 8, false>(nc);
    if (rt == 2 && wk == 8)
        return pick_nc<kForm, kEpi, 2, 8, false>(nc);
    if (rt == 1 && wk == 4)
        return pick_nc<kForm, kEpi, 1, 4, false>(nc);
    if (rt == 2 && wk == 4)
        return pick_nc<kForm, kEpi, 2, 4, false>(nc);
    return nullptr;
}

} // namespace

template <int kForm, bool kEpi>
cudaError_t smallm_init()
{
    for (int chunked = 0; chunked < 2; ++chunked)
        for (int wk : {4, 8})
            for (int rt : {1, 2})
                for (int nc = 1; nc <= kMaxNc; ++nc)
                {
                    KernelFn const k = pick<kForm, kEpi>(nc, rt, wk, chunked != 0);
                    if (k == nullptr)
                        continue;   // no chunked kernel at wk == 4
                    cudaError_t const e = cudaFuncSetAttribute(reinterpret_cast<void const*>(k),
                        cudaFuncAttributeMaxDynamicSharedMemorySize, fso::sm120_smallm::kSmemPerBlockMax);
                    if (e != cudaSuccess)
                        return e;
                }
    return cudaSuccess;
}

template <int kForm, bool kEpi>
cudaError_t smallm_launch(
    SmallmParams const& p, int nc, int rt, int wk, bool chunked, int grid, int smem, bool pdl, cudaStream_t s)
{
    KernelFn const k = pick<kForm, kEpi>(nc, rt, wk, chunked);
    if (k == nullptr)
        return cudaErrorInvalidValue;
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.gridDim = dim3(unsigned(grid), 1, 1);
    cfg.blockDim = dim3(unsigned(32 * wk), 1, 1);
    cfg.dynamicSmemBytes = size_t(smem);
    cfg.stream = s;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, k, p);
}

template cudaError_t smallm_init<FSO_SMALLM_FORM, bool(FSO_SMALLM_EPI)>();
template cudaError_t smallm_launch<FSO_SMALLM_FORM, bool(FSO_SMALLM_EPI)>(
    SmallmParams const&, int, int, int, bool, int, int, bool, cudaStream_t);

} // namespace tensorrt_llm::kernels::blockscale_gemm::sm120_smallm
