/*
 * Audio-tower fused Blackwell (sm_100 / sm_103) Conv2d Stem epilogue kernels:
 *   1. gelu_nhwc_to_mxfp8_sm100: takes conv3 output [B, C=480, F=16, T=13] in channels_last
 *      memory order (physical [B, F=16, T=13, C=480]), applies exact erf-GELU (packed f32x2
 *      bit-identical to F.gelu(..., approximate="none")), permutes (b, t, f, c) -> [B*T, F*C],
 *      and quantizes directly to MXFP8 1x32 SM100 atom layout in ONE pass with PDL.
 *      (Pairs with static load-time column reorder of conv_out weight (C,F) -> (F,C).)
 *   2. gelu_erf_bf16_pdl_sm100: fast packed-f32x2 exact erf-GELU on BF16 tensor with PDL
 *      for conv1 & conv2 activations.
 */

#include "blockscale_gemm/arch/sm100/mxfp8/dense_epi_math.cuh"
#include "fso/activation_contract.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

namespace fso_audio
{
namespace
{

static inline bool stem_is_sm100_device(c10::Device dev)
{
    auto const* prop = at::cuda::getDeviceProperties(dev.index());
    return prop != nullptr && prop->major == 10;
}

static inline bool stem_pdl_enabled()
{
    static bool const v = [] {
        char const* e = std::getenv("FSO_DISABLE_PDL");
        return !(e && e[0] == '1');
    }();
    return v;
}

template <typename KernelT, typename... Args>
static inline cudaError_t launch_stem_pdl(
    KernelT kernel, dim3 grid, dim3 block, cudaStream_t stream, bool pdl, Args... args)
{
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, args..., pdl);
}

// One CTA (256 threads = 16 groups of 16 threads = 16 kp groups per iter) per output row m = b*T + t.
// Physical input layout: [B, F, T, C] contiguous (PyTorch channels_last [B, C, F, T]), C % 8 == 0, (F*C) % 128 == 0.
__global__ void __launch_bounds__(256) gelu_nhwc_to_mxfp8_kernel(
    __nv_fp8_e4m3* __restrict__ out_q,
    int32_t* __restrict__ out_scale,
    __nv_bfloat16 const* __restrict__ x_nhwc,
    int B,
    int F_dim,
    int T_dim,
    int C_dim,
    int M,
    int M_pad,
    bool pdl)
{
    if (pdl)
    {
#if __CUDA_ARCH__ >= 900
        asm volatile("griddepcontrol.wait;\n" ::: "memory");
        if (threadIdx.x == 0)
            asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
    }

    int const tid = threadIdx.x;
    int const row = blockIdx.x;
    int const K = F_dim * C_dim;
    int const num_kp = K >> 7; // K / 128

    if (row >= M)
    {
        int const total_words = (M_pad - M) * num_kp;
        for (int idx = tid; idx < total_words; idx += 256)
        {
            int const r = M + idx / num_kp;
            int const kp = idx % num_kp;
            out_scale[fso_mx_sf_word_index_sm100(r, kp, K)] = 0;
        }
        return;
    }

    int const b = row / T_dim;
    int const t = row - b * T_dim;
    int const lane16 = tid & 15;
    int const kp_in_cta = tid >> 4; // 0..15

    int64_t const b_base = static_cast<int64_t>(b) * F_dim * T_dim * C_dim;
    __nv_fp8_e4m3* const q_row = out_q + static_cast<int64_t>(row) * K;

    for (int kp_base = 0; kp_base < num_kp; kp_base += 16)
    {
        int const kp = kp_base + kp_in_cta;
        if (kp < num_kp)
        {
            int const col = kp * 128 + lane16 * 8;
            int const f = col / C_dim;
            int const c = col - f * C_dim;
            int64_t const src_off = b_base + (static_cast<int64_t>(f) * T_dim + t) * C_dim + c;

            uint4 const raw = *reinterpret_cast<uint4 const*>(x_nhwc + src_off);
            __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&raw);

            float y[8];
            float amax = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float2 const xf = __bfloat1622float2(xp[i]);
                float2 const gf = fso::sm100::dense_gelu_erf_pair(xf);
                y[2 * i] = gf.x;
                y[2 * i + 1] = gf.y;
                amax = fmaxf(amax, fmaxf(fabsf(gf.x), fabsf(gf.y)));
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1, 4));
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2, 4));
            amax = fmaxf(amax, 1e-10f);

            float qs;
            uint32_t const byte_v = fso::sm100::dense_e8m0_from_amax(amax, &qs);
            float2 const qs2 = make_float2(qs, qs);
            uint2 q8;
            q8.x = fso::sm100::dense_quantize4(make_float2(y[0], y[1]), make_float2(y[2], y[3]), qs2);
            q8.y = fso::sm100::dense_quantize4(make_float2(y[4], y[5]), make_float2(y[6], y[7]), qs2);
            *reinterpret_cast<uint2*>(q_row + col) = q8;

            uint32_t const b0 = byte_v;
            uint32_t const b1 = __shfl_sync(0xFFFFFFFFu, b0, 4, 16);
            uint32_t const b2 = __shfl_sync(0xFFFFFFFFu, b0, 8, 16);
            uint32_t const b3 = __shfl_sync(0xFFFFFFFFu, b0, 12, 16);
            if (lane16 == 0)
            {
                out_scale[fso_mx_sf_word_index_sm100(row, kp, K)] =
                    static_cast<int32_t>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
            }
        }
    }
}

__global__ void __launch_bounds__(256) gelu_erf_bf16_pdl_kernel(
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16 const* __restrict__ x,
    int64_t num_vec8,
    bool pdl)
{
    if (pdl)
    {
#if __CUDA_ARCH__ >= 900
        asm volatile("griddepcontrol.wait;\n" ::: "memory");
        if (threadIdx.x == 0)
            asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
    }
    int64_t const idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_vec8)
        return;
    uint4 const raw = reinterpret_cast<uint4 const*>(x)[idx];
    __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&raw);
    uint4 out_raw;
    __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&out_raw);
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        float2 const xf = __bfloat1622float2(xp[i]);
        float2 const gf = fso::sm100::dense_gelu_erf_pair(xf);
        op[i] = __float22bfloat162_rn(gf);
    }
    reinterpret_cast<uint4*>(out)[idx] = out_raw;
}

} // namespace

std::tuple<at::Tensor, at::Tensor> gelu_nhwc_to_mxfp8_sm100(at::Tensor x_nhwc)
{
    TORCH_CHECK(x_nhwc.is_cuda() && x_nhwc.dtype() == at::kBFloat16 && x_nhwc.dim() == 4,
        "gelu_nhwc_to_mxfp8_sm100: x_nhwc must be 4-D CUDA bf16 [B, C, F, T] in channels_last memory format");
    TORCH_CHECK(stem_is_sm100_device(x_nhwc.device()), "gelu_nhwc_to_mxfp8_sm100: requires sm_100 / sm_103");
    TORCH_CHECK(x_nhwc.is_contiguous(at::MemoryFormat::ChannelsLast),
        "gelu_nhwc_to_mxfp8_sm100: x_nhwc must be channels_last contiguous");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(x_nhwc.data_ptr()) & 15u) == 0u,
        "gelu_nhwc_to_mxfp8_sm100: x_nhwc must be 16-byte aligned");
    int const B = static_cast<int>(x_nhwc.size(0));
    int const C = static_cast<int>(x_nhwc.size(1));
    int const F_dim = static_cast<int>(x_nhwc.size(2));
    int const T_dim = static_cast<int>(x_nhwc.size(3));
    int const M = B * T_dim;
    int const K = F_dim * C;
    TORCH_CHECK(C > 0 && F_dim > 0 && C % 8 == 0 && K % 128 == 0,
        "gelu_nhwc_to_mxfp8_sm100: C>0 (multiple of 8), F>0, and F*C multiple of 128 required");

    int const M_pad = (M + 127) / 128 * 128;
    auto out_q = at::empty({M, K}, x_nhwc.options().dtype(at::kFloat8_e4m3fn));
    auto out_scale = at::empty({static_cast<int64_t>(M_pad) * (K / 128)}, x_nhwc.options().dtype(at::kInt));
    if (M == 0)
        return {out_q, out_scale};

    c10::cuda::CUDAGuard guard(x_nhwc.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    bool const pdl = stem_pdl_enabled();
    dim3 grid(M + (M_pad > M ? 1 : 0), 1, 1);
    cudaError_t err = launch_stem_pdl(gelu_nhwc_to_mxfp8_kernel, grid, dim3(256), stream, pdl,
        reinterpret_cast<__nv_fp8_e4m3*>(out_q.data_ptr()),
        reinterpret_cast<int32_t*>(out_scale.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(x_nhwc.data_ptr()),
        B, F_dim, T_dim, C, M, M_pad);
    TORCH_CHECK(err == cudaSuccess, "gelu_nhwc_to_mxfp8_sm100 launch failed: ", cudaGetErrorString(err));
    return {out_q, out_scale};
}

at::Tensor gelu_erf_bf16_pdl_sm100(at::Tensor x)
{
    TORCH_CHECK(x.is_cuda() && x.dtype() == at::kBFloat16 && x.numel() % 8 == 0,
        "gelu_erf_bf16_pdl_sm100: x must be CUDA bf16 with numel % 8 == 0");
    TORCH_CHECK(stem_is_sm100_device(x.device()), "gelu_erf_bf16_pdl_sm100: requires sm_100 / sm_103");
    bool const is_contig = x.is_contiguous() || (x.dim() == 4 && x.is_contiguous(at::MemoryFormat::ChannelsLast));
    TORCH_CHECK(is_contig, "gelu_erf_bf16_pdl_sm100: x must be contiguous or 4-D channels_last contiguous");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) & 15u) == 0u,
        "gelu_erf_bf16_pdl_sm100: x must be 16-byte aligned");
    auto out = at::empty_like(x);
    int64_t const n8 = x.numel() / 8;
    if (n8 == 0)
        return out;
    TORCH_CHECK((reinterpret_cast<uintptr_t>(out.data_ptr()) & 15u) == 0u,
        "gelu_erf_bf16_pdl_sm100: out must be 16-byte aligned");
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    bool const pdl = stem_pdl_enabled();
    dim3 grid(static_cast<unsigned int>((n8 + 255) / 256), 1, 1);
    cudaError_t err = launch_stem_pdl(gelu_erf_bf16_pdl_kernel, grid, dim3(256), stream, pdl,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr()),
        n8);
    TORCH_CHECK(err == cudaSuccess, "gelu_erf_bf16_pdl_sm100 launch failed: ", cudaGetErrorString(err));
    return out;
}

} // namespace fso_audio
