/*
 * PyTorch ATen ops for blockscale_gemm.
 *
 * Kept ATen-only: includes just the runner declaration and forward decls for
 * the quantize functions defined in quant_kernels.cu. The CUTLASS / cute
 * headers (which `using namespace cute`) are NOT pulled into this TU because
 * they collide with at::Layout.
 *
 * Public ops registered as torch.ops.blockscale_gemm.* in bindings.cpp.
 */

#include "blockscale_gemm/runner.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdlib>
#include <torch/torch.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace blockscale_gemm
{
namespace detail
{
// Defined in quant_kernels.cu.
void fp8bs_quantize_1x128(__nv_fp8_e4m3* x_q, float* scales, __nv_bfloat16 const* x, int M, int K, cudaStream_t stream,
    bool use_ue8m0);
void fp8bs_quantize_1x128_packed(__nv_fp8_e4m3* x_q, int32_t* packed_scales,
    __nv_bfloat16 const* x, int M, int K, cudaStream_t stream, bool use_ue8m0);
void fp8bs_quantize_128x128(
    __nv_fp8_e4m3* w_q, float* scales, __nv_bfloat16 const* w, int N, int K, cudaStream_t stream);
} // namespace detail

namespace
{
namespace runtime = tensorrt_llm::kernels::blockscale_gemm;

inline int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

void check_cuda_bf16(at::Tensor const& t, char const* name)
{
    TORCH_CHECK(t.is_cuda(), name, " must be on CUDA");
    TORCH_CHECK(t.dtype() == at::kBFloat16, name, " must be bf16");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_cuda_fp8(at::Tensor const& t, char const* name)
{
    TORCH_CHECK(t.is_cuda(), name, " must be on CUDA");
    TORCH_CHECK(t.dtype() == at::kFloat8_e4m3fn, name, " must be fp8_e4m3");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}
} // namespace

at::Tensor linear_bf16(at::Tensor x, at::Tensor w)
{
    check_cuda_bf16(x, "x");
    check_cuda_bf16(w, "w");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(w.dim() == 2, "w must be 2D");
    int const M = x.size(0);
    int const K = x.size(1);
    int const N = w.size(0);
    TORCH_CHECK(w.size(1) == K, "w.size(1) must match x.size(1)");

    auto y = at::empty({M, N}, x.options());

    using Runner = runtime::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>;
    Runner runner;
    std::size_t ws_bytes = runner.getWorkspaceSize(M, N, K);
    auto ws = at::empty({static_cast<int64_t>(ws_bytes)}, x.options().dtype(at::kByte));
    runner.configureWorkspace(reinterpret_cast<char*>(ws.data_ptr()));

    auto stream = at::cuda::getCurrentCUDAStream();
    runner.gemm(reinterpret_cast<void*>(y.data_ptr()), reinterpret_cast<void const*>(x.data_ptr()),
        reinterpret_cast<void const*>(w.data_ptr()), M, N, K, stream);
    return y;
}

// Forward decls: defined in quant_kernels.cu. Repack the
// UE8M0-rounded FP32 scales into the int32-packed layout CUTLASS
// Sm120BlockScaledKernel expects (4 K-consecutive UE8M0 bytes per int32,
// per-row along M or N).
namespace detail
{
// Per-row M-dim activation scales: in shape [M_pad, K/128] K-major FP32.
// Out shape [M_pad, K/512] K-major int32.
void repack_ue8m0_scales_for_sm120(int32_t* dst, float const* src, int M_pad, int K_blocks_in,
    cudaStream_t stream);
// Per-128-N-block weight scales -> per-N-row packed: in shape
// [N/128, K/128] FP32 row-major. Out shape [N_pad, K/512] K-major int32
// where rows 0..127 share scales from input row 0, etc.
void repack_ue8m0_scales_sfb_for_sm120(int32_t* dst, float const* src, int N_pad,
    int N_blocks_in, int K_blocks_in_per_row, cudaStream_t stream);
}

at::Tensor linear_fp8(at::Tensor x_fp8, at::Tensor w_fp8, at::Tensor sx, at::Tensor sw)
{
    check_cuda_fp8(x_fp8, "x_fp8");
    check_cuda_fp8(w_fp8, "w_fp8");
    TORCH_CHECK(x_fp8.dim() == 2 && w_fp8.dim() == 2, "x_fp8 / w_fp8 must be 2D");
    TORCH_CHECK(sx.is_cuda() && sw.is_cuda(), "scales must be on CUDA");

    int const M = x_fp8.size(0);
    int const K = x_fp8.size(1);
    int const N = w_fp8.size(0);
    TORCH_CHECK(w_fp8.size(1) == K, "w.size(1) must match x.size(1)");

    auto y = at::empty({M, N}, x_fp8.options().dtype(at::kBFloat16));

    // Single FP8 path per arch via CUTLASS BlockScaled (sm_120) or
    // deep_gemm JIT (sm_90). On sm_120 the kernel reads scales as int32
    // with 4 UE8M0 bytes packed per word; our `quantize_1x128(use_ue8m0=
    // True)` produces UE8M0-quantised values stored in FP32. Repack here
    // before calling the runner.
    // Cache sm_major in a function-local static — cudaDeviceGetAttribute
    // is forbidden during CUDA graph stream capture, so we resolve it
    // once at first call and reuse. Assumes a single GPU per process
    // (multi-GPU callers must use one process per device).
    static int sm_major = []() {
        int dev = -1;
        cudaGetDevice(&dev);
        int v = -1;
        cudaDeviceGetAttribute(&v, cudaDevAttrComputeCapabilityMajor, dev);
        return v;
    }();

    at::Tensor sx_use = sx;
    at::Tensor sw_use = sw;
    if (sm_major == 12)
    {
        // Fast path: scales are already in the int32-packed layout
        // (output of repack_fp8_act_scales / repack_fp8_wgt_scales).
        // Skip the per-call repack — at small M this saves ~9 μs of
        // alloc + 2 repack-kernel launches per call.
        bool const sx_packed_in = sx.dtype() == at::kInt;
        bool const sw_packed_in = sw.dtype() == at::kInt;
        TORCH_CHECK(sx_packed_in == sw_packed_in,
            "sx and sw must both be FP32 (auto-pack each call) or both be int32 (pre-packed); got mixed dtypes");
        if (!sx_packed_in)
        {
            TORCH_CHECK(sx.dtype() == at::kFloat && sw.dtype() == at::kFloat,
                "sm_120 FP8 scales must be FP32 (use quantize_1x128(use_ue8m0=True)) "
                "or int32 (pre-packed via repack_fp8_act_scales / repack_fp8_wgt_scales)");
            // sx layout (deep_gemm convention): K-major in [pad(M,4), K/128]
            // metadata. Physical: data[kb * pad_M + m]. Repack to int32
            // [pad_M, K/512] with 4 K-consecutive UE8M0 bytes per int32.
            int const M_pad_quant = sx.size(0);
            int const K_blocks_in = sx.size(1);
            TORCH_CHECK(K_blocks_in % 4 == 0,
                "sm_120 FP8 requires K_blocks divisible by 4 (K%512==0); got K=", K);
            auto stream = at::cuda::getCurrentCUDAStream();
            // Output: physical layout [K/512, pad_M] int32 (K-major), tensor
            // metadata [pad_M, K/512] for downstream consistency.
            auto sx_packed = at::empty({M_pad_quant, K_blocks_in / 4},
                sx.options().dtype(at::kInt));
            detail::repack_ue8m0_scales_for_sm120(
                reinterpret_cast<int32_t*>(sx_packed.data_ptr()),
                reinterpret_cast<float const*>(sx.data_ptr()),
                M_pad_quant, K_blocks_in, stream);
            sx_use = sx_packed;
            // sw is [N/128, K/128] FP32 row-major (1 scale per 128x128
            // weight block). CUTLASS expects [align(N,4), K/512] int32
            // K-major (1 scale per N row). Expand 128-block scale across
            // 128 N rows + pack 4 K-consecutive bytes per int32.
            int const N_blocks_in = sw.size(0);
            TORCH_CHECK(sw.size(1) == K_blocks_in,
                "sw must have same K/128 as sx; got ", sw.size(1));
            // align(N, 4): we need scale_n = align(N, 4) per CUTLASS
            // get_tma_aligned_size(N).
            int const N_pad = (N + 3) / 4 * 4;
            auto sw_packed = at::empty({N_pad, K_blocks_in / 4},
                sw.options().dtype(at::kInt));
            detail::repack_ue8m0_scales_sfb_for_sm120(
                reinterpret_cast<int32_t*>(sw_packed.data_ptr()),
                reinterpret_cast<float const*>(sw.data_ptr()),
                N_pad, N_blocks_in, K_blocks_in, stream);
            sw_use = sw_packed;
        }
        // else: sx/sw already int32 (caller pre-packed). Pass through.
    }

    using Runner = runtime::CutlassFp8BlockScaleGemmRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>;
    Runner runner;
    runner.configureWorkspace(nullptr);

    auto stream = at::cuda::getCurrentCUDAStream();
    runner.gemm(reinterpret_cast<__nv_fp8_e4m3 const*>(x_fp8.data_ptr()), /*ld_a=*/K,
        reinterpret_cast<__nv_fp8_e4m3 const*>(w_fp8.data_ptr()), /*ld_b=*/K,
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), /*ld_d=*/N, M, N, K,
        reinterpret_cast<float const*>(sx_use.data_ptr()),
        reinterpret_cast<float const*>(sw_use.data_ptr()), stream);
    return y;
}

std::tuple<at::Tensor, at::Tensor> quantize_1x128(at::Tensor x, bool use_ue8m0)
{
    check_cuda_bf16(x, "x");
    TORCH_CHECK(x.dim() >= 2, "x must be at least 2D");
    int const K = x.size(-1);
    int const M = x.numel() / K;
    auto x2 = x.view({M, K});

    auto x_q = at::empty({M, K}, x.options().dtype(at::kFloat8_e4m3fn));
    int const m_pad = ceil_div(M, 4) * 4;
    int const k_blocks = ceil_div(K, 128);
    auto scales = at::empty({m_pad, k_blocks}, x.options().dtype(at::kFloat));

    auto stream = at::cuda::getCurrentCUDAStream();
    detail::fp8bs_quantize_1x128(reinterpret_cast<__nv_fp8_e4m3*>(x_q.data_ptr()),
        reinterpret_cast<float*>(scales.data_ptr()), reinterpret_cast<__nv_bfloat16 const*>(x2.data_ptr()), M, K,
        stream, use_ue8m0);
    return {x_q.view(x.sizes()), scales};
}


// quantize_1x128_packed: fused BSFP8 1×128 quantize + scale-pack.
// Returns (fp8 [..., K], packed int32 [pad(M,4), K/512] K-major).
// Replaces `quantize_1x128` + `repack_fp8_act_scales` two-step path on
// sm_120 — eliminates the FP32 scale round-trip through global memory.
// Requires K % 512 == 0 (4 K-blocks × 128 elements per warp).
std::tuple<at::Tensor, at::Tensor> quantize_1x128_packed(at::Tensor x, bool use_ue8m0)
{
    check_cuda_bf16(x, "x");
    TORCH_CHECK(x.dim() >= 2, "x must be at least 2D");
    int const K = x.size(-1);
    TORCH_CHECK(K % 512 == 0, "K must be a multiple of 512 (4 K-blocks per packed int32)");
    int const M = x.numel() / K;
    auto x2 = x.view({M, K});

    auto x_q = at::empty({M, K}, x.options().dtype(at::kFloat8_e4m3fn));
    int const m_pad = ceil_div(M, 4) * 4;
    int const k_blocks_out = K / 512;  // packed int32 columns
    // Allocate as row-major to match `repack_fp8_act_scales`. The kernel still
    // writes physical layout [k_blocks_out, m_pad] K-major (dst[kp*M_pad+m]);
    // PyTorch metadata is `at::empty({M_pad, k_blocks_out})` row-major and the
    // GEMM reads via raw pointer. This is the FSO 1×128 BSFP8 path convention.
    auto packed = at::empty({m_pad, k_blocks_out}, x.options().dtype(at::kInt));

    auto stream = at::cuda::getCurrentCUDAStream();
    detail::fp8bs_quantize_1x128_packed(reinterpret_cast<__nv_fp8_e4m3*>(x_q.data_ptr()),
        reinterpret_cast<int32_t*>(packed.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(x2.data_ptr()), M, K, stream, use_ue8m0);
    return {x_q.view(x.sizes()), packed};
}

std::tuple<at::Tensor, at::Tensor> quantize_128x128(at::Tensor w)
{
    check_cuda_bf16(w, "w");
    TORCH_CHECK(w.dim() == 2, "w must be 2D");
    int const N = w.size(0);
    int const K = w.size(1);

    auto w_q = at::empty({N, K}, w.options().dtype(at::kFloat8_e4m3fn));
    auto scales = at::empty({ceil_div(N, 128), ceil_div(K, 128)}, w.options().dtype(at::kFloat));

    auto stream = at::cuda::getCurrentCUDAStream();
    detail::fp8bs_quantize_128x128(reinterpret_cast<__nv_fp8_e4m3*>(w_q.data_ptr()),
        reinterpret_cast<float*>(scales.data_ptr()), reinterpret_cast<__nv_bfloat16 const*>(w.data_ptr()), N, K,
        stream);
    return {w_q, scales};
}

// repack_fp8_act_scales / repack_fp8_wgt_scales: explicit pre-pack of FP8
// dequant scales into the int32-packed UE8M0 format CUTLASS Sm120BlockScaled
// expects. Mirror of what `linear_fp8` does internally on sm_120, exposed as
// public ops so callers with cached weights can repack ONCE outside the
// inference loop and skip the per-call repack overhead. Pass the resulting
// int32 tensors back into `linear_fp8` and the wrapper takes the fast path.
//
// Output layouts (matched to the kernel's deduce_sf*_layout):
//   act:  [pad(M, 4), K/512] int32 K-major (4 UE8M0 bytes per word)
//   wgt:  [pad(N, 4), K/512] int32 K-major (per-N-row, expanded from the
//         per-128×128-block FP32 input)
at::Tensor repack_fp8_act_scales(at::Tensor sx_f32)
{
    TORCH_CHECK(sx_f32.is_cuda() && sx_f32.dtype() == at::kFloat,
        "sx_f32 must be CUDA fp32 (output of quantize_1x128(use_ue8m0=True))");
    TORCH_CHECK(sx_f32.dim() == 2, "sx_f32 must be 2D [pad(M,4), K/128]");
    int const M_pad = sx_f32.size(0);
    int const K_blocks_in = sx_f32.size(1);
    TORCH_CHECK(M_pad % 4 == 0, "sx_f32.size(0) must be a multiple of 4");
    TORCH_CHECK(K_blocks_in % 4 == 0, "K/128 must be a multiple of 4 (= K%512 == 0)");

    int const K_blocks_out = K_blocks_in / 4;
    auto packed = at::empty({M_pad, K_blocks_out}, sx_f32.options().dtype(at::kInt));
    auto stream = at::cuda::getCurrentCUDAStream();
    detail::repack_ue8m0_scales_for_sm120(
        reinterpret_cast<int32_t*>(packed.data_ptr()),
        reinterpret_cast<float const*>(sx_f32.contiguous().data_ptr()),
        M_pad, K_blocks_in, stream);
    return packed;
}


at::Tensor repack_fp8_wgt_scales(at::Tensor sw_f32)
{
    TORCH_CHECK(sw_f32.is_cuda() && sw_f32.dtype() == at::kFloat,
        "sw_f32 must be CUDA fp32 (output of quantize_128x128)");
    TORCH_CHECK(sw_f32.dim() == 2, "sw_f32 must be 2D [N/128, K/128]");
    int const N_blocks_in = sw_f32.size(0);
    int const K_blocks_in = sw_f32.size(1);
    TORCH_CHECK(K_blocks_in % 4 == 0, "K/128 must be a multiple of 4 (= K%512 == 0)");

    int const N = N_blocks_in * 128;
    int const N_pad = (N + 3) / 4 * 4;  // matches CUTLASS get_tma_aligned_size(N)
    int const K_blocks_out = K_blocks_in / 4;
    auto packed = at::empty({N_pad, K_blocks_out}, sw_f32.options().dtype(at::kInt));
    auto stream = at::cuda::getCurrentCUDAStream();
    detail::repack_ue8m0_scales_sfb_for_sm120(
        reinterpret_cast<int32_t*>(packed.data_ptr()),
        reinterpret_cast<float const*>(sw_f32.contiguous().data_ptr()),
        N_pad, N_blocks_in, K_blocks_in, stream);
    return packed;
}


// linear_qx: y = x @ w.T with x quantized inside the op and a *pre-quantized*
// FP8 weight + per-128x128 scales. This is the typical inference path: weight
// is fixed, only the activation needs per-call quantization.
//
// `sw` layout (SM90 path): [ceil(N,128), ceil(K,128)] float32, row-major.
// SM120 layouts differ (UE8M0-packed int32) and are not validated through
// this op yet.
at::Tensor linear_qx(at::Tensor x_bf16, at::Tensor w_fp8, at::Tensor sw)
{
    check_cuda_bf16(x_bf16, "x_bf16");
    check_cuda_fp8(w_fp8, "w_fp8");
    TORCH_CHECK(x_bf16.dim() == 2 && w_fp8.dim() == 2, "x and w must be 2D");
    TORCH_CHECK(sw.is_cuda() && sw.dtype() == at::kFloat, "sw must be CUDA float32");
    int const M = x_bf16.size(0);
    int const K = x_bf16.size(1);
    int const N = w_fp8.size(0);
    TORCH_CHECK(w_fp8.size(1) == K, "w.size(1) must match x.size(1)");

    auto y = at::empty({M, N}, x_bf16.options());

    using Runner = runtime::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>;
    Runner runner;
    std::size_t ws_bytes = runner.getWorkspaceSize(M, N, K);
    auto ws = at::empty({static_cast<int64_t>(ws_bytes)}, x_bf16.options().dtype(at::kByte));
    runner.configureWorkspace(reinterpret_cast<char*>(ws.data_ptr()));

    auto stream = at::cuda::getCurrentCUDAStream();
    runner.gemm(reinterpret_cast<void*>(y.data_ptr()), reinterpret_cast<void const*>(x_bf16.data_ptr()),
        reinterpret_cast<void const*>(w_fp8.data_ptr()), M, N, K, stream,
        /*scales_a=*/nullptr, // produced internally
        reinterpret_cast<float const*>(sw.data_ptr()));
    return y;
}

} // namespace blockscale_gemm
