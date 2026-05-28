/*
 * ATen wrapper for the MXFP8 (1×32, true OCP MXFP8) blockwise GEMM.
 * SM120+ only (CUTLASS BlockScaled with kSFVecSize=32). C1 stage:
 * single tile (128,128,2), takes already-packed int32 scales; quantize
 * + repack lands in C3.
 *
 * Forward-decl of the launcher so cute headers don't leak into this TU
 * and clash with at::Layout.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace blockscale_gemm
{
namespace detail
{
cudaError_t launch_sm120_mxfp8_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int M, int N, int K, cudaStream_t stream);
// In quant_kernels.cu.
void fp8bs_quantize_1x32(__nv_fp8_e4m3* x_q, float* scales, __nv_bfloat16 const* x, int M, int K, cudaStream_t stream,
    bool use_ue8m0);
void fp8bs_quantize_1x32_packed(__nv_fp8_e4m3* x_q, int32_t* packed_scales,
    __nv_bfloat16 const* x, int M, int K, cudaStream_t stream, bool use_ue8m0);
void fp8bs_silu_chunk_mul_quantize_1x32_packed(__nv_fp8_e4m3* x_q,
    int32_t* packed_scales, __nv_bfloat16 const* gu, int M, int K,
    cudaStream_t stream, bool use_ue8m0);
void repack_ue8m0_scales_for_sm120(int32_t* dst, float const* src, int M_pad, int K_blocks_in,
    cudaStream_t stream);
} // namespace detail


// quantize_1x32: BF16 [M, K] -> FP8 (E4M3) [M, K] + FP32 dequant scales
// in K-major shape [pad(M,4), K/32]. Mirrors quantize_1x128's storage
// convention so existing callers can pattern-match the layout. Scales
// are UE8M0-quantized when use_ue8m0=True (mandatory for sm_120).
std::tuple<at::Tensor, at::Tensor> quantize_1x32(at::Tensor x, bool use_ue8m0)
{
    TORCH_CHECK(x.is_cuda() && x.dtype() == at::kBFloat16, "x must be CUDA bf16");
    TORCH_CHECK(x.dim() >= 2, "x must be at least 2D");
    int const K = x.size(-1);
    TORCH_CHECK(K % 32 == 0, "K must be a multiple of 32 (MXFP8 vec size)");

    auto x_flat = x.contiguous().view({-1, K});
    int const M = x_flat.size(0);
    int const M_pad = (M + 3) / 4 * 4;
    int const k_blocks = K / 32;

    auto x_q = at::empty({M, K}, x.options().dtype(at::kFloat8_e4m3fn));
    auto scales = at::empty({M_pad, k_blocks}, x.options().dtype(at::kFloat));

    auto stream = at::cuda::getCurrentCUDAStream();
    detail::fp8bs_quantize_1x32(reinterpret_cast<__nv_fp8_e4m3*>(x_q.data_ptr()),
        reinterpret_cast<float*>(scales.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(x_flat.data_ptr()), M, K, stream, use_ue8m0);

    return {x_q.view(x.sizes()), scales};
}


// quantize_1x32_packed: fused BF16 [M, K] -> FP8 (E4M3) [M, K] + int32-packed
// UE8M0 scales [pad(M,4), K/128] K-major (4 UE8M0 bytes per int32). Single
// kernel — replaces the legacy `quantize_1x32` + `repack_mxfp8_scales`
// sequence. Output layout is bit-exact with the legacy two-kernel path.
//
// Requires K % 128 == 0 (same as legacy). Use this for production sm_120
// MXFP8 paths; the legacy two-step path is kept for the (rare) callers
// that need to introspect the FP32 dequant scale.
std::tuple<at::Tensor, at::Tensor> quantize_1x32_packed(at::Tensor x, bool use_ue8m0)
{
    TORCH_CHECK(x.is_cuda() && x.dtype() == at::kBFloat16, "x must be CUDA bf16");
    TORCH_CHECK(x.dim() >= 2, "x must be at least 2D");
    int const K = x.size(-1);
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128 (packed scale needs 4 K-blocks per int32)");

    auto x_flat = x.contiguous().view({-1, K});
    int const M = x_flat.size(0);
    int const M_pad = (M + 3) / 4 * 4;
    int const K_blocks_out = K / 128;     // packed int32 scale columns

    auto x_q = at::empty({M, K}, x.options().dtype(at::kFloat8_e4m3fn));
    // K-major (stride 1 along M, M_pad along K). Matches `repack_mxfp8_scales`.
    auto packed = at::empty_strided({M_pad, K_blocks_out}, {1, M_pad},
        x.options().dtype(at::kInt));

    auto stream = at::cuda::getCurrentCUDAStream();
    detail::fp8bs_quantize_1x32_packed(
        reinterpret_cast<__nv_fp8_e4m3*>(x_q.data_ptr()),
        reinterpret_cast<int32_t*>(packed.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(x_flat.data_ptr()),
        M, K, stream, use_ue8m0);

    return {x_q.view(x.sizes()), packed};
}


// silu_chunk_mul_quantize_1x32: fused SwiGLU activation prologue + MXFP8
// quantize. Takes BF16 `gu` shape [..., 2*INTER] (gate concatenated with up
// along last dim), computes `h = silu(gate) * up`, and quantizes to FP8 +
// packed UE8M0 scale — without materialising `h` in global memory. Use
// before `linear_mxfp8(hq, w_down, sh_packed, sw)` in SwiGLU MLP forward.
//
// Requires last-dim % 256 == 0 (so INTER % 128 == 0). Output shape:
//   x_q     : FP8 [..., INTER]
//   packed  : int32 [pad(M_flat,4), INTER/128] K-major (stride 1 along M)
std::tuple<at::Tensor, at::Tensor> silu_chunk_mul_quantize_1x32(
    at::Tensor gu, bool use_ue8m0)
{
    TORCH_CHECK(gu.is_cuda() && gu.dtype() == at::kBFloat16, "gu must be CUDA bf16");
    TORCH_CHECK(gu.dim() >= 2, "gu must be at least 2D");
    int const TWO_INTER = gu.size(-1);
    TORCH_CHECK(TWO_INTER % 2 == 0, "last dim must be even (gate || up)");
    int const INTER = TWO_INTER / 2;
    TORCH_CHECK(INTER % 128 == 0,
        "INTER must be a multiple of 128 (packed scale needs 4 K-blocks per int32)");

    auto gu_flat = gu.contiguous().view({-1, TWO_INTER});
    int const M = gu_flat.size(0);
    int const M_pad = (M + 3) / 4 * 4;
    int const K_blocks_out = INTER / 128;

    auto out_sizes = gu.sizes().vec();
    out_sizes.back() = INTER;
    auto x_q = at::empty(out_sizes, gu.options().dtype(at::kFloat8_e4m3fn));
    auto packed = at::empty_strided({M_pad, K_blocks_out}, {1, M_pad},
        gu.options().dtype(at::kInt));

    auto stream = at::cuda::getCurrentCUDAStream();
    detail::fp8bs_silu_chunk_mul_quantize_1x32_packed(
        reinterpret_cast<__nv_fp8_e4m3*>(x_q.data_ptr()),
        reinterpret_cast<int32_t*>(packed.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(gu_flat.data_ptr()),
        M, INTER, stream, use_ue8m0);

    return {x_q, packed};
}


// repack_mxfp8_scales: convert FP32 1x32 dequant scales [pad(M,4), K/32]
// (output of quantize_1x32) to the int32-packed layout the CUTLASS
// Sm120BlockScaledKernel expects: [pad(M,4), K/128] int32 K-major, with
// 4 UE8M0 bytes per int32 word (each byte covers 32 K-elements). Same
// kernel as the 1x128 path's repack — the byte-pack-4 logic is
// VS-agnostic; what differs is the input tensor's K-extent.
at::Tensor repack_mxfp8_scales(at::Tensor scales_f32)
{
    TORCH_CHECK(scales_f32.is_cuda() && scales_f32.dtype() == at::kFloat,
        "scales must be CUDA fp32");
    TORCH_CHECK(scales_f32.dim() == 2, "scales must be 2D [pad(M,4), K/32]");
    int const M_pad = scales_f32.size(0);
    int const K_blocks_in = scales_f32.size(1);
    TORCH_CHECK(M_pad % 4 == 0, "scales.size(0) must be a multiple of 4");
    TORCH_CHECK(K_blocks_in % 4 == 0, "K/32 must be a multiple of 4 (= K%128 == 0)");

    int const K_blocks_out = K_blocks_in / 4;
    // K-major (M-fastest) memory layout matches both the kernel TMA descriptor
    // and the repack write pattern `dst[kp * M_pad + m]`. The 1×128 linear_fp8
    // path stores row-major and silently relies on the kernel ignoring torch
    // metadata; we set the strides correctly so downstream Python tools that
    // index `[m, kp]` actually read the right bytes.
    auto packed = at::empty_strided({M_pad, K_blocks_out}, {1, M_pad},
        scales_f32.options().dtype(at::kInt));
    auto stream = at::cuda::getCurrentCUDAStream();
    detail::repack_ue8m0_scales_for_sm120(
        reinterpret_cast<int32_t*>(packed.data_ptr()),
        reinterpret_cast<float const*>(scales_f32.contiguous().data_ptr()),
        M_pad, K_blocks_in, stream);
    return packed;
}


// linear_mxfp8_raw: take int32-packed UE8M0 scales directly. Routes
// through the full (M, tiles_n) cascade + Stream-K (gemm_dispatch_sm120_mxfp8).
// Production users should call linear_mxfp8 (C4) which adds quantize +
// repack on top.
at::Tensor linear_mxfp8_raw(at::Tensor x_fp8, at::Tensor w_fp8, at::Tensor sx_int32, at::Tensor sw_int32)
{
    TORCH_CHECK(x_fp8.is_cuda() && w_fp8.is_cuda(), "x/w must be on CUDA");
    TORCH_CHECK(x_fp8.dtype() == at::kFloat8_e4m3fn && w_fp8.dtype() == at::kFloat8_e4m3fn,
        "x_fp8 / w_fp8 must be float8_e4m3fn");
    TORCH_CHECK(sx_int32.dtype() == at::kInt && sw_int32.dtype() == at::kInt,
        "sx / sw must be int32 (packed UE8M0)");
    TORCH_CHECK(x_fp8.dim() == 2 && w_fp8.dim() == 2, "x_fp8 / w_fp8 must be 2D");

    int const M = x_fp8.size(0);
    int const K = x_fp8.size(1);
    int const N = w_fp8.size(0);
    TORCH_CHECK(w_fp8.size(1) == K, "w.size(1) must match x.size(1)");
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128 (BlockScaled atom requires it)");

    auto y = at::empty({M, N}, x_fp8.options().dtype(at::kBFloat16));
    auto stream = at::cuda::getCurrentCUDAStream();
    auto err = detail::launch_sm120_mxfp8_dispatch(
        reinterpret_cast<__nv_fp8_e4m3*>(x_fp8.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(w_fp8.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        reinterpret_cast<int32_t*>(sx_int32.data_ptr()),
        reinterpret_cast<int32_t*>(sw_int32.data_ptr()),
        M, N, K, stream);
    TORCH_CHECK(err == cudaSuccess, "sm120 mxfp8 kernel error: ", cudaGetErrorString(err));
    return y;
}

} // namespace blockscale_gemm
