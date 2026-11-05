/*
 * cute-using TU for the sm_100/sm_103 dense MXFP8 GEMMs that write the quantized output of fso.dense.linear_quantized
 * (issue #12 S3b: bias and erf-GELU, RN_bf16, then the activation quantizer of fso/activation_contract.h; e4m3 codes
 * and UE8M0 words in the sm_100 activation contract): the 2-SM tiles with the TMA (smem-staged) epilogue. One of three such units
 * (mxfp8_sm100_epi_quant_kernel_{1sm,2sm,nosmem}.cu), each instantiating one group of the configuration list in
 * blockscale_gemm/arch/sm100/mxfp8/dense_common.cuh, so the groups compile in parallel with each other and with the
 * other sm_100 units. The dispatcher in mxfp8_sm100_kernel.cu references these instantiations; nothing else here is
 * callable.
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
FSO_SM100_DENSE_EPI_CONFIGS_2SM_TMA(FSO_SM100_DENSE_EPI_QUANT_INSTANTIATE)
} // namespace sm100_blockscaled_gemm
#endif
