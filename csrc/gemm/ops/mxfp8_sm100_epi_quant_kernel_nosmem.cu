/*
 * cute-using TU for the sm_100/sm_103 dense MXFP8 GEMMs that write the quantized output of fso.dense.linear_quantized
 * (issue #12 S3b: bias and erf-GELU, RN_bf16, then the activation quantizer of fso/activation_contract.h; e4m3 codes
 * and UE8M0 words in the sm_100 activation contract): the tiles with the direct-store (NoSmem) epilogue. One of three such units
 * (mxfp8_sm100_epi_quant_kernel_{1sm,2sm,nosmem}.cu), each instantiating one group of the configuration list in
 * blockscale_gemm/arch/sm100/mxfp8/dense_common.cuh, so the groups compile in parallel with each other and with the
 * other sm_100 units. The dispatcher in mxfp8_sm100_kernel.cu references these instantiations and, from this unit
 * only, the quantized-output reduce of its two-kernel split-K (launch_sm100_mxfp8_splitk_reduce_q, declared in
 * dense_common.cuh); nothing else here is callable.
 *
 * Grid dependency control, as in mxfp8_sm100_kernel.cu: CUTLASS_ENABLE_GDC_FOR_SM100 only in the sm_100 / sm_103
 * device passes (it would otherwise also switch GDC on in the sm_120 / sm_121 passes), FSO_SM100_DENSE_GDC as the
 * host-visible twin that lets the launcher pass launch_with_pdl. The epilogue's bias loads and every one of its
 * stores follow the grid-dependency wait (dense_epilogue_gemm.cuh), so the PDL launch is safe.
 */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)
#define CUTLASS_ENABLE_GDC_FOR_SM100 1
#endif
#define FSO_SM100_DENSE_GDC 1

#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#include "blockscale_gemm/arch/sm100/mxfp8/dense_epilogue_gemm.cuh"

namespace sm100_blockscaled_gemm
{
FSO_SM100_DENSE_EPI_CONFIGS_NOSMEM(FSO_SM100_DENSE_EPI_QUANT_INSTANTIATE)

// ------------------------------------------------ the quantized-output reduce of the two-kernel split-K (S3b)
//
// The third form of the split-K reduce (dispatch.cuh defines the identity and the bf16-epilogue forms next to the
// launcher that picks between them; this one lives here so that the dispatch translation unit's own kernels stay
// untouched). One warp per scale word, i.e. per (row m < align(M, 128), 128-column group kp): lane l holds the four
// columns kp * 128 + 4 l .. 4 l + 3, so the eight lanes 8 j .. 8 j + 7 hold 32-column block 4 kp + j and the block's
// amax is three xor-shuffles away. Per element: the partials summed in the order of the other two reduces (p0, then
// + p1, + p2, ...: the same fp32 accumulator, so the bf16 value is the byte linear() writes on this route), bias
// and the activation through fso::dense_epilogue_value_rt, RN_bf16; then per block the activation quantizer of
// fso/activation_contract.h (amax floored at fso_mx_amax_floor(), fso_e8m0_from_amax, fso_mx_quantize4), the four
// codes of a lane as one 32-bit store, the four bytes of the word gathered to lane 0 and stored once. A row in
// [M, align(M, 128)) gets a zero word (the padded rows of the contract) and no codes. Every global read is a
// coherent __ldcg and follows the grid-dependency wait, as in the two reduces of dispatch.cuh (under PDL this grid
// is created while the batched GEMM in front of it still runs).
namespace
{

__global__ void sm100_mxfp8_splitk_reduce_quant_kernel(float4 const* partials, uint32_t* out_q, int32_t* out_scale,
    int splits, int M, int M_pad, int N, __nv_bfloat16 const* bias, int act, bool pdl)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (pdl)
        cudaGridDependencySynchronize();
#endif
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)
    // As in the reduces of dispatch.cuh: a programmatic dependent still waits for this grid's completion.
    if (pdl && threadIdx.x == 0)
        cudaTriggerProgrammaticLaunchCompletion();
#endif
    int const lane = threadIdx.x & 31;
    int const num_kp = N / 128;
    long const words = static_cast<long>(M_pad) * num_kp;
    long const mn4 = static_cast<long>(M) * N / 4; // float4s of one partial
    long const warp_stride = static_cast<long>(gridDim.x) * (blockDim.x >> 5);
    for (long w = (static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5; w < words; w += warp_stride)
    {
        int const m = static_cast<int>(w / num_kp);
        int const kp = static_cast<int>(w - static_cast<long>(m) * num_kp);
        int32_t* const sf = out_scale + fso_mx_sf_word_index_sm100(m, kp, N);
        if (m >= M)
        {
            if (lane == 0)
                *sf = 0;
            continue;
        }
        int const n = kp * 128 + lane * 4;
        long const i = (static_cast<long>(m) * N + n) / 4;
        float4 acc = __ldcg(partials + i);
        for (int s = 1; s < splits; ++s)
        {
            float4 const p = __ldcg(partials + s * mn4 + i);
            acc.x += p.x;
            acc.y += p.y;
            acc.z += p.z;
            acc.w += p.w;
        }
        float b[4] = {0.f, 0.f, 0.f, 0.f};
        if (bias != nullptr)
        {
            b[0] = __bfloat162float(__ldcg(bias + n + 0));
            b[1] = __bfloat162float(__ldcg(bias + n + 1));
            b[2] = __bfloat162float(__ldcg(bias + n + 2));
            b[3] = __bfloat162float(__ldcg(bias + n + 3));
        }
        float const a[4] = {acc.x, acc.y, acc.z, acc.w};
        float y[4];
        if (act == fso::kDenseActGeluErf)
        {
            for (int j = 0; j < 4; ++j)
                y[j] = __bfloat162float(__float2bfloat16_rn(
                    fso::dense_epilogue_value_rt<false>(a[j], b[j], fso::kDenseActGeluErf, 0.f)));
        }
        else
        {
            for (int j = 0; j < 4; ++j)
                y[j] = __bfloat162float(
                    __float2bfloat16_rn(fso::dense_epilogue_value_rt<false>(a[j], b[j], fso::kDenseActNone, 0.f)));
        }
        float amax = fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
        float qs;
        uint32_t const byte = fso_e8m0_from_amax(fmaxf(amax, fso_mx_amax_floor()), &qs);
        out_q[i] = fso_mx_quantize4(y[0], y[1], y[2], y[3], qs);
        uint32_t const b0 = __shfl_sync(0xffffffffu, byte, 0);
        uint32_t const b1 = __shfl_sync(0xffffffffu, byte, 8);
        uint32_t const b2 = __shfl_sync(0xffffffffu, byte, 16);
        uint32_t const b3 = __shfl_sync(0xffffffffu, byte, 24);
        if (lane == 0)
            *sf = static_cast<int32_t>(fso_mx_pack_word(static_cast<uint8_t>(b0), static_cast<uint8_t>(b1),
                static_cast<uint8_t>(b2), static_cast<uint8_t>(b3)));
    }
}

} // namespace

cudaError_t launch_sm100_mxfp8_splitk_reduce_q(float const* partials, int splits, int M, int N, cudaStream_t stream,
    fso::DenseEpilogue const& epilogue, bool pdl)
{
    if (epilogue.out_q == nullptr || epilogue.out_scale == nullptr
        || epilogue.out_scale_layout != fso::kDenseOutScaleMxSm100 || epilogue.residual != nullptr || N % 128 != 0)
        return cudaErrorInvalidValue; // the dispatch refuses all of these before anything is launched
    int const M_pad = (M + 127) / 128 * 128;
    long const words = static_cast<long>(M_pad) * (N / 128);
    int const threads = 256; // eight warps, one word each per iteration
    long const want = (words + 7) / 8;
    int const blocks = static_cast<int>(want < 1024 ? want : 1024);
    auto* const out_q = reinterpret_cast<uint32_t*>(epilogue.out_q);
    auto* const out_scale = static_cast<int32_t*>(epilogue.out_scale);
    auto const* const p4 = reinterpret_cast<float4 const*>(partials);
    if (pdl)
    {
        // The reduce as the batched GEMM's programmatic dependent (the launch attribute of the two reduces of
        // dispatch.cuh): its CTAs may be created while the GEMM drains; it waits for the GEMM's completion before
        // it reads a partial or the bias, and before it writes a code or a scale word.
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
        cudaLaunchKernelEx(&cfg, sm100_mxfp8_splitk_reduce_quant_kernel, p4, out_q, out_scale, splits, M, M_pad, N,
            epilogue.bias, epilogue.act, true);
    }
    else
    {
        sm100_mxfp8_splitk_reduce_quant_kernel<<<blocks, threads, 0, stream>>>(
            p4, out_q, out_scale, splits, M, M_pad, N, epilogue.bias, epilogue.act, false);
    }
    return cudaGetLastError();
}

} // namespace sm100_blockscaled_gemm
#endif
