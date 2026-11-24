/*
 * Audio-tower fused Blackwell (sm_100 / sm_103) LayerNorm -> MXFP8 1x32 kernels.
 *
 * Emits (x_q [M, D] float8_e4m3fn, x_scale [align(M, 128) * (D / 128)] int32)
 * matching fso/activation_contract.h bit for bit (SM100 1-D atom layout,
 * UE8M0 1x32 scales, zeroed padded rows [M, align(M, 128))), with PDL
 * (Programmatic Dependent Launch) wait + early trigger so upstream GEMMs and
 * downstream weights-first GEMMs overlap seamlessly.
 */

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

static inline bool pdl_enabled()
{
    static bool const v = [] {
        char const* e = std::getenv("FSO_DISABLE_PDL");
        return !(e && e[0] == '1');
    }();
    return v;
}

template <typename KernelT, typename... Args>
static inline void launch_pdl(
    KernelT kernel, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, bool pdl, Args... args)
{
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernel, args..., pdl);
}

// Fast division-free UE8M0 helper bit-identical to fso_e8m0_from_amax
// (same derivation as dense_epi_math.cuh::dense_e8m0_from_amax).
__device__ __forceinline__ uint32_t fast_e8m0_from_amax(float amax, float* quant_scale)
{
    uint32_t const bits = __float_as_uint(amax);
    uint32_t const e = (bits >> 23) & 0xFFu;
    uint32_t const f = bits & 0x7FFFFFu;
    int byte = e == 255u ? 254 : static_cast<int>(e) - (f <= 0x600000u ? 8 : 7);
    byte = byte < 0 ? 0 : (byte > 254 ? 254 : byte);
    *quant_scale = byte == 0 ? 1.f
                             : (byte == 254 ? __uint_as_float(0x00400000u)
                                            : __uint_as_float(static_cast<uint32_t>(254 - byte) << 23));
    return static_cast<uint32_t>(byte);
}

__device__ __forceinline__ float warp_sum_32(float v)
{
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off, 32);
    return v;
}

// ---------------------------------------------------------------------------
// Variant 0 (5-warp CTA = 160 threads, 1 row per CTA) for D == 1280.
// Each thread owns 8 contiguous elements (1x uint4 load, 1x uint2 fp8 store).
// 4 threads = 32 elements (1 MXFP8 block); 16 threads = 128 elements (1 int32 scale word).
// Extra CTA (blockIdx.x == M) zeroes padded rows [M, align(M, 128)) scale words.
// ---------------------------------------------------------------------------
template <bool kRoundBf16, bool kHasResidual>
__global__ void __launch_bounds__(160, 8) ln1280_mxfp8_cta160_kernel(
    __nv_fp8_e4m3* __restrict__ out_q,
    int32_t* __restrict__ out_scale,
    __nv_bfloat16* __restrict__ out_res, // written iff kHasResidual
    __nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ residual, // read iff kHasResidual
    __nv_bfloat16 const* __restrict__ gamma,
    __nv_bfloat16 const* __restrict__ beta,
    int M,
    int M_pad,
    float eps,
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

    // Last CTA zeroes all padded rows' scale words [M, M_pad) * 10 words.
    if (row >= M)
    {
        int const pad_rows = M_pad - M;
        int const total_words = pad_rows * 10;
        for (int idx = tid; idx < total_words; idx += 160)
        {
            int const r = M + idx / 10;
            int const kp = idx % 10;
            out_scale[fso_mx_sf_word_index_sm100(r, kp, 1280)] = 0;
        }
        return;
    }

    int const col = tid * 8;
    uint4 const x_raw = *reinterpret_cast<uint4 const*>(x + static_cast<int64_t>(row) * 1280 + col);
    uint4 const g_raw = *reinterpret_cast<uint4 const*>(gamma + col);
    uint4 const b_raw = *reinterpret_cast<uint4 const*>(beta + col);

    float v[8];
    __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&x_raw);

    if constexpr (kHasResidual)
    {
        uint4 const r_raw = *reinterpret_cast<uint4 const*>(residual + static_cast<int64_t>(row) * 1280 + col);
        __nv_bfloat162 const* rp = reinterpret_cast<__nv_bfloat162 const*>(&r_raw);
        uint4 res_out;
        __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&res_out);
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float2 const xf = __bfloat1622float2(xp[i]);
            float2 const rf = __bfloat1622float2(rp[i]);
            float2 const sf = make_float2(xf.x + rf.x, xf.y + rf.y);
            __nv_bfloat162 const sb = __float22bfloat162_rn(sf);
            op[i] = sb;
            float2 const sbf = __bfloat1622float2(sb);
            v[2 * i] = sbf.x;
            v[2 * i + 1] = sbf.y;
        }
        *reinterpret_cast<uint4*>(out_res + static_cast<int64_t>(row) * 1280 + col) = res_out;
    }
    else
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float2 const xf = __bfloat1622float2(xp[i]);
            v[2 * i] = xf.x;
            v[2 * i + 1] = xf.y;
        }
    }

    // Two-pass exact fp32 mean & variance (v[0..7] stay in registers, zero extra memory reads).
    float lsum = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        lsum += v[i];
    lsum = warp_sum_32(lsum);

    __shared__ float smem_sum[5];
    __shared__ float smem_var[5];
    int const warp_id = tid >> 5;
    int const lane_id = tid & 31;
    if (lane_id == 0)
        smem_sum[warp_id] = lsum;
    __syncthreads();

    float const mean = (smem_sum[0] + smem_sum[1] + smem_sum[2] + smem_sum[3] + smem_sum[4]) * (1.f / 1280.f);

    float lvar = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        float const d = v[i] - mean;
        v[i] = d;
        lvar += d * d;
    }
    lvar = warp_sum_32(lvar);
    if (lane_id == 0)
        smem_var[warp_id] = lvar;
    __syncthreads();

    float const var = (smem_var[0] + smem_var[1] + smem_var[2] + smem_var[3] + smem_var[4]) * (1.f / 1280.f);
    float const rstd = rsqrtf(var + eps);

    __nv_bfloat162 const* gp = reinterpret_cast<__nv_bfloat162 const*>(&g_raw);
    __nv_bfloat162 const* bp = reinterpret_cast<__nv_bfloat162 const*>(&b_raw);
    float y[8];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        float2 const gf = __bfloat1622float2(gp[i]);
        float2 const bf = __bfloat1622float2(bp[i]);
        float y0 = v[2 * i] * rstd * gf.x + bf.x;
        float y1 = v[2 * i + 1] * rstd * gf.y + bf.y;
        if constexpr (kRoundBf16)
        {
            __nv_bfloat162 const yb = __float22bfloat162_rn(make_float2(y0, y1));
            float2 const ybf = __bfloat1622float2(yb);
            y0 = ybf.x;
            y1 = ybf.y;
        }
        y[2 * i] = y0;
        y[2 * i + 1] = y1;
        amax = fmaxf(amax, fmaxf(fabsf(y0), fabsf(y1)));
    }

    // 4 consecutive threads = 32 elements (1 MXFP8 block)
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2, 4));
    amax = fmaxf(amax, 1e-10f);

    float qs;
    uint32_t const byte_v = fast_e8m0_from_amax(amax, &qs);

    uint2 q8;
    q8.x = fso_fp8x4_from_floats(y[0] * qs, y[1] * qs, y[2] * qs, y[3] * qs);
    q8.y = fso_fp8x4_from_floats(y[4] * qs, y[5] * qs, y[6] * qs, y[7] * qs);
    *reinterpret_cast<uint2*>(out_q + static_cast<int64_t>(row) * 1280 + col) = q8;

    // 16 consecutive threads = 128 elements (1 int32 scale word); lanes 0,4,8,12 of each 16-lane group hold b0..b3.
    uint32_t const b0 = byte_v;
    uint32_t const b1 = __shfl_sync(0xFFFFFFFFu, b0, 4, 16);
    uint32_t const b2 = __shfl_sync(0xFFFFFFFFu, b0, 8, 16);
    uint32_t const b3 = __shfl_sync(0xFFFFFFFFu, b0, 12, 16);
    if ((tid & 15) == 0)
    {
        int const kp = tid >> 4; // 0..9
        out_scale[fso_mx_sf_word_index_sm100(row, kp, 1280)] =
            static_cast<int32_t>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
    }
}

// ---------------------------------------------------------------------------
// Variant 1 (Warp-per-row, ZERO __syncthreads(), ZERO shared memory!) for D == 1280.
// Each warp (32 lanes) owns 1 row: 5 chunks of 256 elements (8 elems = 1 uint4 per lane per chunk).
// Launches WarpsPerCta warps per CTA (1 warp/CTA for small M, 4 warps/CTA for larger M).
// ---------------------------------------------------------------------------
template <int kWarpsPerCta, bool kRoundBf16>
__global__ void __launch_bounds__(kWarpsPerCta * 32) ln1280_mxfp8_warp_kernel(
    __nv_fp8_e4m3* __restrict__ out_q,
    int32_t* __restrict__ out_scale,
    __nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ gamma,
    __nv_bfloat16 const* __restrict__ beta,
    int M,
    int M_pad,
    float eps,
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

    int const warp_in_cta = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int const row = blockIdx.x * kWarpsPerCta + warp_in_cta;

    if (row >= M)
    {
        if (row < M_pad)
        {
            if (lane < 10)
                out_scale[fso_mx_sf_word_index_sm100(row, lane, 1280)] = 0;
        }
        return;
    }

    __nv_bfloat16 const* x_row = x + static_cast<int64_t>(row) * 1280;
    __nv_fp8_e4m3* q_row = out_q + static_cast<int64_t>(row) * 1280;

    float v[5][8];
    float lsum = 0.f;
#pragma unroll
    for (int c = 0; c < 5; ++c)
    {
        int const col = c * 256 + lane * 8;
        uint4 const x_raw = *reinterpret_cast<uint4 const*>(x_row + col);
        __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&x_raw);
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float2 const xf = __bfloat1622float2(xp[i]);
            v[c][2 * i] = xf.x;
            v[c][2 * i + 1] = xf.y;
            lsum += xf.x + xf.y;
        }
    }
    float const mean = warp_sum_32(lsum) * (1.f / 1280.f);

    float lvar = 0.f;
#pragma unroll
    for (int c = 0; c < 5; ++c)
    {
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            float const d = v[c][i] - mean;
            v[c][i] = d;
            lvar += d * d;
        }
    }
    float const rstd = rsqrtf(warp_sum_32(lvar) * (1.f / 1280.f) + eps);

#pragma unroll
    for (int c = 0; c < 5; ++c)
    {
        int const col = c * 256 + lane * 8;
        uint4 const g_raw = *reinterpret_cast<uint4 const*>(gamma + col);
        uint4 const b_raw = *reinterpret_cast<uint4 const*>(beta + col);
        __nv_bfloat162 const* gp = reinterpret_cast<__nv_bfloat162 const*>(&g_raw);
        __nv_bfloat162 const* bp = reinterpret_cast<__nv_bfloat162 const*>(&b_raw);
        float y[8];
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float2 const gf = __bfloat1622float2(gp[i]);
            float2 const bf = __bfloat1622float2(bp[i]);
            float y0 = v[c][2 * i] * rstd * gf.x + bf.x;
            float y1 = v[c][2 * i + 1] * rstd * gf.y + bf.y;
            if constexpr (kRoundBf16)
            {
                __nv_bfloat162 const yb = __float22bfloat162_rn(make_float2(y0, y1));
                float2 const ybf = __bfloat1622float2(yb);
                y0 = ybf.x;
                y1 = ybf.y;
            }
            y[2 * i] = y0;
            y[2 * i + 1] = y1;
            amax = fmaxf(amax, fmaxf(fabsf(y0), fabsf(y1)));
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1, 4));
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2, 4));
        amax = fmaxf(amax, 1e-10f);

        float qs;
        uint32_t const byte_v = fast_e8m0_from_amax(amax, &qs);
        uint2 q8;
        q8.x = fso_fp8x4_from_floats(y[0] * qs, y[1] * qs, y[2] * qs, y[3] * qs);
        q8.y = fso_fp8x4_from_floats(y[4] * qs, y[5] * qs, y[6] * qs, y[7] * qs);
        *reinterpret_cast<uint2*>(q_row + col) = q8;

        uint32_t const b0 = byte_v;
        uint32_t const b1 = __shfl_sync(0xFFFFFFFFu, b0, 4, 16);
        uint32_t const b2 = __shfl_sync(0xFFFFFFFFu, b0, 8, 16);
        uint32_t const b3 = __shfl_sync(0xFFFFFFFFu, b0, 12, 16);
        if ((lane & 15) == 0)
        {
            int const kp = c * 2 + (lane >> 4);
            out_scale[fso_mx_sf_word_index_sm100(row, kp, 1280)] =
                static_cast<int32_t>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
        }
    }
}

// ---------------------------------------------------------------------------
// General D % 128 == 0 fallback (1 CTA of 256 threads per row, 2-pass global).
// ---------------------------------------------------------------------------
template <bool kRoundBf16>
__global__ void __launch_bounds__(256) ln_general_mxfp8_kernel(
    __nv_fp8_e4m3* __restrict__ out_q,
    int32_t* __restrict__ out_scale,
    __nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ gamma,
    __nv_bfloat16 const* __restrict__ beta,
    int M,
    int D,
    int M_pad,
    float eps,
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
    int const num_kp = D / 128;

    if (row >= M)
    {
        int const total_words = (M_pad - M) * num_kp;
        for (int idx = tid; idx < total_words; idx += 256)
        {
            int const r = M + idx / num_kp;
            int const kp = idx % num_kp;
            out_scale[fso_mx_sf_word_index_sm100(r, kp, D)] = 0;
        }
        return;
    }

    __nv_bfloat16 const* x_row = x + static_cast<int64_t>(row) * D;
    __nv_fp8_e4m3* q_row = out_q + static_cast<int64_t>(row) * D;

    __shared__ float smem_sum[8];
    __shared__ float smem_var[8];
    float lsum = 0.f;
    for (int col = tid * 8; col < D; col += 256 * 8)
    {
        uint4 const xr = *reinterpret_cast<uint4 const*>(x_row + col);
        __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&xr);
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float2 const xf = __bfloat1622float2(xp[i]);
            lsum += xf.x + xf.y;
        }
    }
    lsum = warp_sum_32(lsum);
    if ((tid & 31) == 0)
        smem_sum[tid >> 5] = lsum;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w)
        tot += smem_sum[w];
    float const mean = tot / static_cast<float>(D);

    float lvar = 0.f;
    for (int col = tid * 8; col < D; col += 256 * 8)
    {
        uint4 const xr = *reinterpret_cast<uint4 const*>(x_row + col);
        __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&xr);
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float2 const xf = __bfloat1622float2(xp[i]);
            float const d0 = xf.x - mean;
            float const d1 = xf.y - mean;
            lvar += d0 * d0 + d1 * d1;
        }
    }
    lvar = warp_sum_32(lvar);
    if ((tid & 31) == 0)
        smem_var[tid >> 5] = lvar;
    __syncthreads();
    float vtot = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w)
        vtot += smem_var[w];
    float const rstd = rsqrtf(vtot / static_cast<float>(D) + eps);

    // 256 threads * 8 elems = 2048 elems per chunk (16 scale words per chunk).
    for (int base = 0; base < D; base += 256 * 8)
    {
        int const col = base + tid * 8;
        if (col < D)
        {
            uint4 const xr = *reinterpret_cast<uint4 const*>(x_row + col);
            uint4 const gr = *reinterpret_cast<uint4 const*>(gamma + col);
            uint4 const br = *reinterpret_cast<uint4 const*>(beta + col);
            __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&xr);
            __nv_bfloat162 const* gp = reinterpret_cast<__nv_bfloat162 const*>(&gr);
            __nv_bfloat162 const* bp = reinterpret_cast<__nv_bfloat162 const*>(&br);
            float y[8];
            float amax = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float2 const xf = __bfloat1622float2(xp[i]);
                float2 const gf = __bfloat1622float2(gp[i]);
                float2 const bf = __bfloat1622float2(bp[i]);
                float y0 = (xf.x - mean) * rstd * gf.x + bf.x;
                float y1 = (xf.y - mean) * rstd * gf.y + bf.y;
                if constexpr (kRoundBf16)
                {
                    __nv_bfloat162 const yb = __float22bfloat162_rn(make_float2(y0, y1));
                    float2 const ybf = __bfloat1622float2(yb);
                    y0 = ybf.x;
                    y1 = ybf.y;
                }
                y[2 * i] = y0;
                y[2 * i + 1] = y1;
                amax = fmaxf(amax, fmaxf(fabsf(y0), fabsf(y1)));
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1, 4));
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2, 4));
            amax = fmaxf(amax, 1e-10f);
            float qs;
            uint32_t const byte_v = fast_e8m0_from_amax(amax, &qs);
            uint2 q8;
            q8.x = fso_fp8x4_from_floats(y[0] * qs, y[1] * qs, y[2] * qs, y[3] * qs);
            q8.y = fso_fp8x4_from_floats(y[4] * qs, y[5] * qs, y[6] * qs, y[7] * qs);
            *reinterpret_cast<uint2*>(q_row + col) = q8;

            uint32_t const b0 = byte_v;
            uint32_t const b1 = __shfl_sync(0xFFFFFFFFu, b0, 4, 16);
            uint32_t const b2 = __shfl_sync(0xFFFFFFFFu, b0, 8, 16);
            uint32_t const b3 = __shfl_sync(0xFFFFFFFFu, b0, 12, 16);
            if ((tid & 15) == 0)
            {
                int const kp = col >> 7;
                out_scale[fso_mx_sf_word_index_sm100(row, kp, D)] =
                    static_cast<int32_t>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
            }
        }
    }
}

} // namespace

std::tuple<at::Tensor, at::Tensor> layernorm_mxfp8_sm100(
    at::Tensor x,
    at::Tensor gamma,
    at::Tensor beta,
    double eps,
    bool round_bf16,
    int64_t variant)
{
    TORCH_CHECK(x.is_cuda() && x.dtype() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2,
        "x must be contiguous CUDA bf16 [M, D]");
    TORCH_CHECK(gamma.is_cuda() && gamma.dtype() == at::kBFloat16 && gamma.is_contiguous() && gamma.dim() == 1,
        "gamma must be contiguous CUDA bf16 [D]");
    TORCH_CHECK(beta.is_cuda() && beta.dtype() == at::kBFloat16 && beta.is_contiguous() && beta.dim() == 1,
        "beta must be contiguous CUDA bf16 [D]");
    int const M = static_cast<int>(x.size(0));
    int const D = static_cast<int>(x.size(1));
    TORCH_CHECK(gamma.size(0) == D && beta.size(0) == D, "gamma/beta size mismatch");
    TORCH_CHECK(D % 128 == 0, "D must be a multiple of 128");

    int const M_pad = (M + 127) / 128 * 128;
    auto x_q = at::empty({M, D}, x.options().dtype(at::kFloat8_e4m3fn));
    auto x_scale = at::empty({static_cast<int64_t>(M_pad) * (D / 128)}, x.options().dtype(at::kInt));
    if (M == 0)
        return {x_q, x_scale};

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    bool const pdl = pdl_enabled();
    float const eps_f = static_cast<float>(eps);

    auto* q_ptr = reinterpret_cast<__nv_fp8_e4m3*>(x_q.data_ptr());
    auto* s_ptr = reinterpret_cast<int32_t*>(x_scale.data_ptr());
    auto const* x_ptr = reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr());
    auto const* g_ptr = reinterpret_cast<__nv_bfloat16 const*>(gamma.data_ptr());
    auto const* b_ptr = reinterpret_cast<__nv_bfloat16 const*>(beta.data_ptr());

    if (D == 1280)
    {
        // variant: 0 = auto (warp1 for small M<=192, cta160 for mid, warp4 for large),
        //          1 = cta160 (5 warps/row), 2 = warp1 (1 warp/CTA), 3 = warp4 (4 warps/CTA)
        int v = static_cast<int>(variant);
        if (v == 0)
            v = (M <= 256) ? 1 : 3;

        if (v == 1)
        {
            dim3 grid(M + (M_pad > M ? 1 : 0), 1, 1);
            if (round_bf16)
                launch_pdl(ln1280_mxfp8_cta160_kernel<true, false>, grid, dim3(160), 0, stream, pdl,
                    q_ptr, s_ptr, nullptr, x_ptr, nullptr, g_ptr, b_ptr, M, M_pad, eps_f);
            else
                launch_pdl(ln1280_mxfp8_cta160_kernel<false, false>, grid, dim3(160), 0, stream, pdl,
                    q_ptr, s_ptr, nullptr, x_ptr, nullptr, g_ptr, b_ptr, M, M_pad, eps_f);
        }
        else if (v == 2)
        {
            dim3 grid(M_pad, 1, 1);
            if (round_bf16)
                launch_pdl(ln1280_mxfp8_warp_kernel<1, true>, grid, dim3(32), 0, stream, pdl,
                    q_ptr, s_ptr, x_ptr, g_ptr, b_ptr, M, M_pad, eps_f);
            else
                launch_pdl(ln1280_mxfp8_warp_kernel<1, false>, grid, dim3(32), 0, stream, pdl,
                    q_ptr, s_ptr, x_ptr, g_ptr, b_ptr, M, M_pad, eps_f);
        }
        else
        {
            dim3 grid((M_pad + 3) / 4, 1, 1);
            if (round_bf16)
                launch_pdl(ln1280_mxfp8_warp_kernel<4, true>, grid, dim3(128), 0, stream, pdl,
                    q_ptr, s_ptr, x_ptr, g_ptr, b_ptr, M, M_pad, eps_f);
            else
                launch_pdl(ln1280_mxfp8_warp_kernel<4, false>, grid, dim3(128), 0, stream, pdl,
                    q_ptr, s_ptr, x_ptr, g_ptr, b_ptr, M, M_pad, eps_f);
        }
        return {x_q, x_scale};
    }

    dim3 grid(M + (M_pad > M ? 1 : 0), 1, 1);
    if (round_bf16)
        launch_pdl(ln_general_mxfp8_kernel<true>, grid, dim3(256), 0, stream, pdl,
            q_ptr, s_ptr, x_ptr, g_ptr, b_ptr, M, D, M_pad, eps_f);
    else
        launch_pdl(ln_general_mxfp8_kernel<false>, grid, dim3(256), 0, stream, pdl,
            q_ptr, s_ptr, x_ptr, g_ptr, b_ptr, M, D, M_pad, eps_f);
    return {x_q, x_scale};
}

TORCH_LIBRARY_FRAGMENT(fish_scales_ops, m)
{
    m.def("layernorm_mxfp8_sm100(Tensor x, Tensor gamma, Tensor beta, float eps=1e-5, "
          "bool round_bf16=False, int variant=0) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fish_scales_ops, CUDA, m)
{
    m.impl("layernorm_mxfp8_sm100", &layernorm_mxfp8_sm100);
}

} // namespace fso_audio
