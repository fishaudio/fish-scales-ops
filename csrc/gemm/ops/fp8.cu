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
    __nv_bfloat16 const* x, int M, int K, cudaStream_t stream, bool use_ue8m0, bool sm1xx_sf_layout);
void fp8bs_quantize_128x128(
    __nv_fp8_e4m3* w_q, float* scales, __nv_bfloat16 const* w, int N, int K, cudaStream_t stream, bool use_ue8m0);
// sm_100 / sm_103 block-FP8: 1x128 / 128x128 UE8M0 FP32 scales -> Sm1xx atom layout, each byte replicated
// into its 4 VS=32 slots (quant_kernels.cu). The GEMM is the MXFP8 tcgen05 path.
void repack_ue8m0_scales_1x128_for_sm1xx(int32_t* dst, float const* src, int M, int M_pad4, int K_blocks,
    cudaStream_t stream);
void repack_ue8m0_scales_128x128_for_sm1xx(int32_t* dst, float const* src, int N, int N_blocks, int K_blocks,
    cudaStream_t stream);
// In mxfp8_sm100_kernel.cu (sm_100/sm_103 tcgen05 BlockScaled path).
cudaError_t launch_sm100_mxfp8_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int M, int N, int K, cudaStream_t stream);
bool sm100_mxfp8_compiled();
// sm_90 (H200) grouped block-scale FP8 masked — fp8_grouped_sm90_kernel.cu.
cudaError_t launch_sm90_fp8_grouped_masked_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    float* SFA, float* SFB, int32_t* masked_m, int num_groups, int m_cap, int N, int K, int expected_m,
    cudaStream_t stream);
// H2 layout-native grouped 1x128 FP32-K-major quantizers (quant_kernels.cu).
void fp8bs_quantize_1x128_fp32_grouped_gather(__nv_fp8_e4m3* x_q, float* sfa, __nv_bfloat16 const* x,
    int32_t const* slot_of_flat, int n_pairs, int topk, int m_cap, int K, cudaStream_t stream);
void fp8bs_silu_chunk_mul_quantize_1x128_fp32_grouped(__nv_fp8_e4m3* x_q, float* sfa, __nv_bfloat16 const* gu,
    int32_t const* slot_of_flat, int n_pairs, int m_cap, int K, cudaStream_t stream);
// H4 contiguous (sorted-layout) siblings.
cudaError_t launch_sm90_fp8_grouped_contiguous_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    float* SFA, float* SFB, int32_t* sorted_expert_ids, int num_groups, int p_max, int N, int K, int block_m,
    int expected_m, cudaStream_t stream);
cudaError_t launch_sm90_fp8_grouped_contiguous_swapab_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    float* SFA, float* SFB, int32_t* sorted_expert_ids, int num_groups, int p_max, int N, int K, int block_n,
    int expected_m, cudaStream_t stream);
void fp8bs_quantize_1x128_fp32_sorted_gather(__nv_fp8_e4m3* x_q, float* sfa, __nv_bfloat16 const* x,
    int32_t const* flat_to_sorted, int n_pairs, int topk, int sfa_ld, int K, cudaStream_t stream);
void fp8bs_silu_chunk_mul_quantize_1x128_fp32_sorted(__nv_fp8_e4m3* x_q, float* sfa, __nv_bfloat16 const* gu,
    int32_t const* flat_to_sorted, int n_pairs, int sfa_ld, int K, cudaStream_t stream);
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

// sm_100 (B200) / sm_103 (B300): the block-FP8 1x128 path runs on the MXFP8
// tcgen05 collective with each UE8M0 scale byte replicated into its four 32-wide
// slots (2026-09-05). Cached per process; cudaDeviceGetAttribute is forbidden
// during CUDA graph capture.
static bool is_sm100_family_cached()
{
    static bool const v = []() {
        int dev = -1;
        cudaGetDevice(&dev);
        int major = -1;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        return major == 10;
    }();
    return v;
}

std::tuple<at::Tensor, at::Tensor> quantize_1x128_packed(at::Tensor x, bool use_ue8m0);
std::tuple<at::Tensor, at::Tensor> quantize_128x128(at::Tensor w, bool use_ue8m0);
at::Tensor repack_fp8_wgt_scales(at::Tensor sw_f32);
at::Tensor linear_fp8(at::Tensor x_fp8, at::Tensor w_fp8, at::Tensor sx, at::Tensor sw);

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

    if (is_sm100_family_cached())
    {
        // No runner path on datacenter Blackwell: compose the public ops
        // (UE8M0 quantize of both operands, atom-layout scales, MXFP8 GEMM).
        auto [xq, sx] = quantize_1x128_packed(x.contiguous(), /*use_ue8m0=*/true);
        auto [wq, sw_f32] = quantize_128x128(w.contiguous(), /*use_ue8m0=*/true);
        return linear_fp8(xq, wq, sx, repack_fp8_wgt_scales(sw_f32));
    }

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
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128 (1x128 activation scale blocks); got K=", K);

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
    if (sm_major == 10)
    {
        // sm_100 / sm_103: block-FP8 through the MXFP8 tcgen05 path. Scales
        // must be UE8M0 (power-of-two); int32 inputs are the pre-packed atom
        // layout from repack_fp8_{act,wgt}_scales / quantize_1x128_packed, FP32
        // inputs are packed here per call.
        TORCH_CHECK(N % 128 == 0, "sm_100/sm_103 block-FP8 requires N % 128 == 0; got N=", N);
        bool const sx_packed_in = sx.dtype() == at::kInt;
        bool const sw_packed_in = sw.dtype() == at::kInt;
        TORCH_CHECK(sx_packed_in == sw_packed_in,
            "sx and sw must both be FP32 (auto-pack each call) or both be int32 (pre-packed); got mixed dtypes");
        auto stream = at::cuda::getCurrentCUDAStream();
        if (!sx_packed_in)
        {
            TORCH_CHECK(sx.dtype() == at::kFloat && sw.dtype() == at::kFloat,
                "sm_100 block-FP8 scales must be FP32 UE8M0 (quantize with use_ue8m0=True) or int32 (pre-packed)");
            int const M_pad4 = sx.size(0);
            int const K_blocks_in = sx.size(1);
            TORCH_CHECK(K_blocks_in == K / 128, "sx.size(1) must be K/128; got ", K_blocks_in, " for K=", K);
            TORCH_CHECK(sw.size(1) == K_blocks_in, "sw must have the same K/128 as sx; got ", sw.size(1));
            int const M_pad128 = (M + 127) / 128 * 128;
            int const N_pad128 = (N + 127) / 128 * 128;
            auto sx_packed = at::empty({static_cast<int64_t>(M_pad128) * K_blocks_in}, sx.options().dtype(at::kInt));
            auto sw_packed = at::empty({static_cast<int64_t>(N_pad128) * K_blocks_in}, sw.options().dtype(at::kInt));
            detail::repack_ue8m0_scales_1x128_for_sm1xx(reinterpret_cast<int32_t*>(sx_packed.data_ptr()),
                reinterpret_cast<float const*>(sx.contiguous().data_ptr()), M_pad4, M_pad4, K_blocks_in, stream);
            detail::repack_ue8m0_scales_128x128_for_sm1xx(reinterpret_cast<int32_t*>(sw_packed.data_ptr()),
                reinterpret_cast<float const*>(sw.contiguous().data_ptr()), N, sw.size(0), K_blocks_in, stream);
            sx_use = sx_packed;
            sw_use = sw_packed;
        }
        TORCH_CHECK(detail::sm100_mxfp8_compiled(),
            "block-FP8 on sm_100/sm_103 requires the extension to be built with TORCH_CUDA_ARCH_LIST including 10.0f");
        cudaError_t const err = detail::launch_sm100_mxfp8_dispatch(
            reinterpret_cast<__nv_fp8_e4m3*>(x_fp8.data_ptr()), reinterpret_cast<__nv_fp8_e4m3*>(w_fp8.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), reinterpret_cast<int32_t*>(sx_use.data_ptr()),
            reinterpret_cast<int32_t*>(sw_use.data_ptr()), M, N, K, stream);
        TORCH_CHECK(err == cudaSuccess, "sm100 block-FP8 kernel error: ", cudaGetErrorString(err));
        return y;
    }
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
            // [pad_M, ceil(K/512)] with 4 K-consecutive UE8M0 bytes per int32;
            // a K/128 count that is not a multiple of 4 (K=768: 6 blocks, 2
            // words) leaves the last word's tail bytes zero — the kernel's K
            // loop stops at the real k-tile count, so they are never read
            // (K % 512 requirement lifted 2026-09-05; K % 128 remains).
            int const M_pad_quant = sx.size(0);
            int const K_blocks_in = sx.size(1);
            TORCH_CHECK(K_blocks_in == K / 128, "sx.size(1) must be K/128; got ", K_blocks_in, " for K=", K);
            int const K_words = ceil_div(K_blocks_in, 4);
            auto stream = at::cuda::getCurrentCUDAStream();
            // Output: physical layout [ceil(K/512), pad_M] int32 (K-major),
            // tensor metadata [pad_M, ceil(K/512)] for downstream consistency.
            auto sx_packed = at::empty({M_pad_quant, K_words},
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
            auto sw_packed = at::empty({N_pad, K_words},
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
// The fused kernel handles K % 512 == 0 (4 K-blocks × 128 elements per
// warp); other K % 128 == 0 shapes take the two-step path (FP32 quantize +
// repack, tail word zero-padded) so every K the GEMM accepts is reachable
// from this op (2026-09-05).
std::tuple<at::Tensor, at::Tensor> quantize_1x128_packed(at::Tensor x, bool use_ue8m0)
{
    check_cuda_bf16(x, "x");
    TORCH_CHECK(x.dim() >= 2, "x must be at least 2D");
    int const K = x.size(-1);
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128 (1x128 scale blocks); got K=", K);
    bool const sm1xx = is_sm100_family_cached();
    if (K % 512 != 0)
    {
        // Two-step path; the repack is called directly (no UE8M0 sanity sync)
        // so the op stays capturable inside a CUDA graph.
        auto [x_q, sx_f32] = quantize_1x128(x, use_ue8m0);
        int const M_pad = sx_f32.size(0);
        int const K_blocks_in = sx_f32.size(1);
        if (sm1xx)
        {
            int const M_pad128 = (x.numel() / K + 127) / 128 * 128;
            auto packed = at::empty({static_cast<int64_t>(M_pad128) * K_blocks_in}, sx_f32.options().dtype(at::kInt));
            detail::repack_ue8m0_scales_1x128_for_sm1xx(reinterpret_cast<int32_t*>(packed.data_ptr()),
                reinterpret_cast<float const*>(sx_f32.data_ptr()), M_pad, M_pad, K_blocks_in,
                at::cuda::getCurrentCUDAStream());
            return {x_q, packed};
        }
        auto packed = at::empty({M_pad, ceil_div(K_blocks_in, 4)}, sx_f32.options().dtype(at::kInt));
        detail::repack_ue8m0_scales_for_sm120(reinterpret_cast<int32_t*>(packed.data_ptr()),
            reinterpret_cast<float const*>(sx_f32.data_ptr()), M_pad, K_blocks_in,
            at::cuda::getCurrentCUDAStream());
        return {x_q, packed};
    }
    int const M = x.numel() / K;
    auto x2 = x.view({M, K});

    auto x_q = at::empty({M, K}, x.options().dtype(at::kFloat8_e4m3fn));
    if (sm1xx)
    {
        // Sm1xx atom layout: opaque 1-D [pad(M,128) * K/128] int32, one word per (row, K-block).
        int const M_pad128 = ceil_div(M, 128) * 128;
        auto packed = at::empty({static_cast<int64_t>(M_pad128) * (K / 128)}, x.options().dtype(at::kInt));
        detail::fp8bs_quantize_1x128_packed(reinterpret_cast<__nv_fp8_e4m3*>(x_q.data_ptr()),
            reinterpret_cast<int32_t*>(packed.data_ptr()), reinterpret_cast<__nv_bfloat16 const*>(x2.data_ptr()),
            M, K, at::cuda::getCurrentCUDAStream(), use_ue8m0, /*sm1xx_sf_layout=*/true);
        return {x_q.view(x.sizes()), packed};
    }
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
        reinterpret_cast<__nv_bfloat16 const*>(x2.data_ptr()), M, K, stream, use_ue8m0, /*sm1xx_sf_layout=*/false);
    return {x_q.view(x.sizes()), packed};
}

// use_ue8m0: quantize with power-of-two block scales (stored as UE8M0-exact
// FP32). Mandatory for the sm_120 GEMM, whose scale repack keeps only the
// exponent byte of each FP32 scale: an amax/448 scale is truncated to the
// power of two below it and the block dequantises 0.5-1.0x too small. Found
// 2026-09-05 (kernel output vs its own dequantised inputs: cos 0.991-0.995,
// per-block gain 0.59-0.70 on random weights). The Python wrapper defaults
// this to true on sm_120; the sm_90 deep_gemm path keeps FP32 scales.
std::tuple<at::Tensor, at::Tensor> quantize_128x128(at::Tensor w, bool use_ue8m0)
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
        stream, use_ue8m0);
    return {w_q, scales};
}

// The sm_120 int32-packed scale format holds UE8M0 exponents only. A scale
// that is not an exact power of two cannot be represented and would be
// truncated silently, so the explicit pre-pack ops reject it (one device
// -> host sync; these ops run at weight-load time, never inside a captured
// graph). The per-call auto-repack inside linear_fp8 cannot afford the sync
// and relies on the quantizers' use_ue8m0 defaults.
static void check_scales_are_ue8m0(at::Tensor const& s_f32, char const* what)
{
    auto const mant = at::bitwise_and(s_f32.contiguous().view(at::kInt), 0x7FFFFF);
    TORCH_CHECK(!(mant != 0).any().item<bool>(), what,
        " must be UE8M0-exact (powers of two): quantize with use_ue8m0=True (sm_120 default). "
        "A non-power-of-two FP32 scale would be truncated to its exponent and the block dequantised "
        "0.5-1.0x too small.");
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
    check_scales_are_ue8m0(sx_f32, "repack_fp8_act_scales: activation scales");
    if (is_sm100_family_cached())
    {
        // Sm1xx atom layout [pad(M,128) * K/128] int32, byte replicated x4 (see quant_kernels.cu).
        int const M_pad128 = ceil_div(M_pad, 128) * 128;
        auto packed = at::empty({static_cast<int64_t>(M_pad128) * K_blocks_in}, sx_f32.options().dtype(at::kInt));
        detail::repack_ue8m0_scales_1x128_for_sm1xx(reinterpret_cast<int32_t*>(packed.data_ptr()),
            reinterpret_cast<float const*>(sx_f32.contiguous().data_ptr()), M_pad, M_pad, K_blocks_in,
            at::cuda::getCurrentCUDAStream());
        return packed;
    }

    int const K_blocks_out = ceil_div(K_blocks_in, 4);  // tail word zero-padded when K/128 % 4 != 0
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
    check_scales_are_ue8m0(sw_f32, "repack_fp8_wgt_scales: weight scales");
    if (is_sm100_family_cached())
    {
        int const N = N_blocks_in * 128;   // N_pad128 == N here
        auto packed = at::empty({static_cast<int64_t>(N) * K_blocks_in}, sw_f32.options().dtype(at::kInt));
        detail::repack_ue8m0_scales_128x128_for_sm1xx(reinterpret_cast<int32_t*>(packed.data_ptr()),
            reinterpret_cast<float const*>(sw_f32.contiguous().data_ptr()), N, N_blocks_in, K_blocks_in,
            at::cuda::getCurrentCUDAStream());
        return packed;
    }

    int const N = N_blocks_in * 128;
    int const N_pad = (N + 3) / 4 * 4;  // matches CUTLASS get_tma_aligned_size(N)
    int const K_blocks_out = ceil_div(K_blocks_in, 4);  // tail word zero-padded when K/128 % 4 != 0
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
    TORCH_CHECK(!is_sm100_family_cached(),
        "linear_qx (runner path) is not available on sm_100/sm_103; use quantize_1x128_fp8_packed + linear_fp8 "
        "with UE8M0 weight scales (repack_fp8_wgt_scales) instead");
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


// ---------------------------------------------------------------------------
// Grouped (MoE, masked) block-scale FP8 on sm_90 (H200). Revives the in-tree
// DeepGEMM GroupedMasked kernel. Scales are FP32 (sm_90 convention), passed
// through to the same DeepGEMM WGMMA kernel the dense linear_fp8 uses — so
// the SFA/SFB layout is exactly what that kernel's TMA descriptors expect
// (H1: caller prepares them; H2 gives fso a layout-native fused quantizer).
//   a_fp8 [G, m_cap, K], w_fp8 [G, N, K], out [G, m_cap, N] bf16
//   sa/sw FP32; masked_m [G] int32 on device (capture-safe).
at::Tensor linear_fp8_grouped_masked(at::Tensor a_fp8, at::Tensor w_fp8, at::Tensor sa, at::Tensor sw,
    at::Tensor masked_m, int64_t expected_m)
{
    TORCH_CHECK(a_fp8.is_cuda() && a_fp8.dtype() == at::kFloat8_e4m3fn && a_fp8.dim() == 3,
        "a_fp8 must be CUDA float8_e4m3fn [G, m_cap, K]");
    TORCH_CHECK(w_fp8.is_cuda() && w_fp8.dtype() == at::kFloat8_e4m3fn && w_fp8.dim() == 3,
        "w_fp8 must be CUDA float8_e4m3fn [G, N, K]");
    TORCH_CHECK(sa.is_cuda() && sa.dtype() == at::kFloat, "sa must be CUDA float32");
    TORCH_CHECK(sw.is_cuda() && sw.dtype() == at::kFloat, "sw must be CUDA float32");
    TORCH_CHECK(masked_m.is_cuda() && masked_m.dtype() == at::kInt && masked_m.is_contiguous(),
        "masked_m must be contiguous CUDA int32 [G]");
    TORCH_CHECK(a_fp8.is_contiguous() && w_fp8.is_contiguous(), "a_fp8 / w_fp8 must be contiguous");
    int const G = a_fp8.size(0);
    int const m_cap = a_fp8.size(1);
    int const K = a_fp8.size(2);
    int const N = w_fp8.size(1);
    TORCH_CHECK(w_fp8.size(0) == G && w_fp8.size(2) == K, "w_fp8 shape must match a_fp8");
    TORCH_CHECK(masked_m.numel() == G, "masked_m must have G entries");
    TORCH_CHECK(K % 128 == 0 && N % 128 == 0, "N and K must be multiples of 128");
    TORCH_CHECK(expected_m >= 1, "expected_m must be >= 1");

    auto y = at::empty({G, m_cap, N}, a_fp8.options().dtype(at::kBFloat16));
    auto stream = at::cuda::getCurrentCUDAStream();
    auto err = detail::launch_sm90_fp8_grouped_masked_dispatch(
        reinterpret_cast<__nv_fp8_e4m3*>(a_fp8.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(w_fp8.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        reinterpret_cast<float*>(sa.data_ptr()),
        reinterpret_cast<float*>(sw.data_ptr()),
        reinterpret_cast<int32_t*>(masked_m.data_ptr()),
        G, m_cap, N, K, static_cast<int>(expected_m), stream);
    TORCH_CHECK(err == cudaSuccess, "sm90 fp8 grouped masked kernel error: ", cudaGetErrorString(err));
    return y;
}


// H2 — layout-native grouped 1x128 quantizers for the sm_90 grouped path.
// Fused token-gather + block-scale FP8 quantize writing the SFA layout the
// GroupedMasked kernel's TMA descriptor reads directly ([G, K/128, m_cap]
// FP32 K-major), so the stock deep_gemm per_token_cast + tma_align two-step
// is gone. slot_of_flat[i] = g*m_cap + m_in (from moe_build_routing); src
// row = i/topk.
std::tuple<at::Tensor, at::Tensor> quantize_1x128_grouped_gather_sm90(
    at::Tensor x, at::Tensor slot_of_flat, int64_t topk, int64_t num_groups, int64_t m_cap)
{
    TORCH_CHECK(x.is_cuda() && x.dtype() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(),
        "x must be contiguous CUDA bf16 [M, K]");
    TORCH_CHECK(slot_of_flat.dtype() == at::kInt && slot_of_flat.is_cuda() && slot_of_flat.is_contiguous(),
        "slot_of_flat must be contiguous CUDA int32");
    int const M = x.size(0);
    int const K = x.size(1);
    int const G = static_cast<int>(num_groups);
    int const Kb = K / 128;
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    TORCH_CHECK(m_cap >= 1 && m_cap % 4 == 0, "m_cap must be a positive multiple of 4");
    TORCH_CHECK(slot_of_flat.numel() == static_cast<int64_t>(M) * topk, "slot_of_flat must be [M*topk]");

    auto x_q = at::empty({G, m_cap, K}, x.options().dtype(at::kFloat8_e4m3fn));
    auto sfa = at::empty({G, Kb, m_cap}, x.options().dtype(at::kFloat));  // K-major
    auto stream = at::cuda::getCurrentCUDAStream();
    detail::fp8bs_quantize_1x128_fp32_grouped_gather(
        reinterpret_cast<__nv_fp8_e4m3*>(x_q.data_ptr()),
        reinterpret_cast<float*>(sfa.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr()),
        reinterpret_cast<int32_t const*>(slot_of_flat.data_ptr()),
        static_cast<int>(M * topk), static_cast<int>(topk), static_cast<int>(m_cap), K, stream);
    return {x_q, sfa};
}

std::tuple<at::Tensor, at::Tensor> silu_chunk_mul_quantize_1x128_grouped_sm90(
    at::Tensor gu, at::Tensor slot_of_flat)
{
    TORCH_CHECK(gu.is_cuda() && gu.dtype() == at::kBFloat16 && gu.dim() == 3 && gu.is_contiguous(),
        "gu must be contiguous CUDA bf16 [G, m_cap, 2*INTER]");
    TORCH_CHECK(slot_of_flat.dtype() == at::kInt && slot_of_flat.is_cuda() && slot_of_flat.is_contiguous(),
        "slot_of_flat must be contiguous CUDA int32");
    int const G = gu.size(0);
    int const m_cap = gu.size(1);
    int const TWO_INTER = gu.size(2);
    TORCH_CHECK(TWO_INTER % 2 == 0, "last dim must be even (gate || up)");
    int const INTER = TWO_INTER / 2;
    int const Kb = INTER / 128;
    TORCH_CHECK(INTER % 128 == 0, "INTER must be a multiple of 128");
    TORCH_CHECK(m_cap % 4 == 0, "m_cap must be a multiple of 4");

    auto x_q = at::empty({G, m_cap, INTER}, gu.options().dtype(at::kFloat8_e4m3fn));
    auto sfa = at::empty({G, Kb, m_cap}, gu.options().dtype(at::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream();
    detail::fp8bs_silu_chunk_mul_quantize_1x128_fp32_grouped(
        reinterpret_cast<__nv_fp8_e4m3*>(x_q.data_ptr()),
        reinterpret_cast<float*>(sfa.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(gu.data_ptr()),
        reinterpret_cast<int32_t const*>(slot_of_flat.data_ptr()),
        static_cast<int>(slot_of_flat.numel()), m_cap, INTER, stream);
    return {x_q, sfa};
}


// ===== H4 contiguous (triton-style sorted layout) sm_90 grouped ops =========

static inline int align_up4(int x) { return (x + 3) / 4 * 4; }

// linear_fp8_grouped_contiguous: contiguous grouped GEMM on the expert-sorted
// activation. a_fp8 [P_max, K], w_fp8 [G, N, K], sa = SFA ColMajor
// [K/128, align(P_max,4)], sw = SFB [G, N/128, K/128],
// sorted_expert_ids [P_max] (grouped_layout, -1 past the real length).
// Returns D [P_max, N] bf16 in sorted order.
at::Tensor linear_fp8_grouped_contiguous(at::Tensor a_fp8, at::Tensor w_fp8, at::Tensor sa, at::Tensor sw,
    at::Tensor sorted_expert_ids, int64_t block_m, int64_t expected_m)
{
    TORCH_CHECK(a_fp8.is_cuda() && a_fp8.dtype() == at::kFloat8_e4m3fn && a_fp8.dim() == 2,
        "a_fp8 must be CUDA float8_e4m3fn [P_max, K]");
    TORCH_CHECK(w_fp8.is_cuda() && w_fp8.dtype() == at::kFloat8_e4m3fn && w_fp8.dim() == 3,
        "w_fp8 must be CUDA float8_e4m3fn [G, N, K]");
    TORCH_CHECK(sa.is_cuda() && sa.dtype() == at::kFloat, "sa must be CUDA float32");
    TORCH_CHECK(sw.is_cuda() && sw.dtype() == at::kFloat, "sw must be CUDA float32");
    TORCH_CHECK(sorted_expert_ids.is_cuda() && sorted_expert_ids.dtype() == at::kInt
            && sorted_expert_ids.is_contiguous(),
        "sorted_expert_ids must be contiguous CUDA int32 [P_max]");
    TORCH_CHECK(a_fp8.is_contiguous() && w_fp8.is_contiguous(), "a_fp8 / w_fp8 must be contiguous");
    int const P_max = a_fp8.size(0);
    int const K = a_fp8.size(1);
    int const G = w_fp8.size(0);
    int const N = w_fp8.size(1);
    TORCH_CHECK(w_fp8.size(2) == K, "w_fp8 K must match a_fp8");
    TORCH_CHECK(sorted_expert_ids.numel() == P_max, "sorted_expert_ids must have P_max entries");
    TORCH_CHECK(K % 128 == 0 && N % 128 == 0, "N and K must be multiples of 128");
    TORCH_CHECK(block_m == 64, "H4a supports block_m = 64 only");

    auto y = at::empty({P_max, N}, a_fp8.options().dtype(at::kBFloat16));
    auto stream = at::cuda::getCurrentCUDAStream();
    auto err = detail::launch_sm90_fp8_grouped_contiguous_dispatch(
        reinterpret_cast<__nv_fp8_e4m3*>(a_fp8.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(w_fp8.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        reinterpret_cast<float*>(sa.data_ptr()),
        reinterpret_cast<float*>(sw.data_ptr()),
        reinterpret_cast<int32_t*>(sorted_expert_ids.data_ptr()),
        G, P_max, N, K, static_cast<int>(block_m), static_cast<int>(expected_m), stream);
    TORCH_CHECK(err == cudaSuccess, "sm90 fp8 grouped contiguous kernel error: ", cudaGetErrorString(err));
    return y;
}

// Swap-AB contiguous grouped GEMM (block_n = 16 activation tiling, for M>=8).
// Same buffer layouts as linear_fp8_grouped_contiguous; the activation is the
// swap-AB B matrix and the weight the A matrix. sorted_expert_ids must be built
// with padding = block_n.
at::Tensor linear_fp8_grouped_contiguous_swapab(at::Tensor a_fp8, at::Tensor w_fp8, at::Tensor sa, at::Tensor sw,
    at::Tensor sorted_expert_ids, int64_t block_n, int64_t expected_m)
{
    TORCH_CHECK(a_fp8.is_cuda() && a_fp8.dtype() == at::kFloat8_e4m3fn && a_fp8.dim() == 2,
        "a_fp8 must be CUDA float8_e4m3fn [P_max, K]");
    TORCH_CHECK(w_fp8.is_cuda() && w_fp8.dtype() == at::kFloat8_e4m3fn && w_fp8.dim() == 3,
        "w_fp8 must be CUDA float8_e4m3fn [G, N, K]");
    TORCH_CHECK(sa.is_cuda() && sa.dtype() == at::kFloat, "sa must be CUDA float32");
    TORCH_CHECK(sw.is_cuda() && sw.dtype() == at::kFloat, "sw must be CUDA float32");
    TORCH_CHECK(sorted_expert_ids.is_cuda() && sorted_expert_ids.dtype() == at::kInt
            && sorted_expert_ids.is_contiguous(),
        "sorted_expert_ids must be contiguous CUDA int32 [P_max]");
    TORCH_CHECK(a_fp8.is_contiguous() && w_fp8.is_contiguous(), "a_fp8 / w_fp8 must be contiguous");
    int const P_max = a_fp8.size(0);
    int const K = a_fp8.size(1);
    int const G = w_fp8.size(0);
    int const N = w_fp8.size(1);
    TORCH_CHECK(w_fp8.size(2) == K, "w_fp8 K must match a_fp8");
    TORCH_CHECK(sorted_expert_ids.numel() == P_max, "sorted_expert_ids must have P_max entries");
    TORCH_CHECK(K % 128 == 0 && N % 128 == 0, "N and K must be multiples of 128");
    TORCH_CHECK(block_n == 16, "H4b swap-AB supports block_n = 16 only");

    auto y = at::empty({P_max, N}, a_fp8.options().dtype(at::kBFloat16));
    auto stream = at::cuda::getCurrentCUDAStream();
    auto err = detail::launch_sm90_fp8_grouped_contiguous_swapab_dispatch(
        reinterpret_cast<__nv_fp8_e4m3*>(a_fp8.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(w_fp8.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        reinterpret_cast<float*>(sa.data_ptr()),
        reinterpret_cast<float*>(sw.data_ptr()),
        reinterpret_cast<int32_t*>(sorted_expert_ids.data_ptr()),
        G, P_max, N, K, static_cast<int>(block_n), static_cast<int>(expected_m), stream);
    TORCH_CHECK(err == cudaSuccess, "sm90 fp8 grouped contiguous swapAB kernel error: ", cudaGetErrorString(err));
    return y;
}

// Fused token-gather + 1x128 quantize into the contiguous sorted layout. x
// [M, K] bf16, flat_to_sorted [R] (pair -> sorted row), p_max (output rows).
// Iterates the R real pairs; padding rows of the output are left uninitialised
// (the GroupedContiguous GEMM is row-independent and the combine reads only
// real rows). Returns (x_q [P_max, K] fp8, sfa [K/128, align(P_max,4)] fp32).
std::tuple<at::Tensor, at::Tensor> quantize_1x128_sorted_gather_sm90(
    at::Tensor x, at::Tensor flat_to_sorted, int64_t p_max, int64_t topk)
{
    TORCH_CHECK(x.is_cuda() && x.dtype() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(),
        "x must be contiguous CUDA bf16 [M, K]");
    TORCH_CHECK(flat_to_sorted.dtype() == at::kInt && flat_to_sorted.is_cuda() && flat_to_sorted.is_contiguous(),
        "flat_to_sorted must be contiguous CUDA int32");
    int const K = x.size(1);
    int const P_max = static_cast<int>(p_max);
    int const R = flat_to_sorted.numel();
    int const Kb = K / 128;
    int const sfa_ld = align_up4(P_max);
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");

    auto x_q = at::empty({P_max, K}, x.options().dtype(at::kFloat8_e4m3fn));
    auto sfa = at::empty({Kb, sfa_ld}, x.options().dtype(at::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream();
    detail::fp8bs_quantize_1x128_fp32_sorted_gather(
        reinterpret_cast<__nv_fp8_e4m3*>(x_q.data_ptr()),
        reinterpret_cast<float*>(sfa.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr()),
        reinterpret_cast<int32_t const*>(flat_to_sorted.data_ptr()),
        R, static_cast<int>(topk), sfa_ld, K, stream);
    return {x_q, sfa};
}

// SwiGLU + 1x128 requantize into the contiguous sorted layout. gu [P_max,
// 2*INTER] bf16 (gate||up, sorted), flat_to_sorted [R]. Iterates the R real
// pairs. Returns (x_q [P_max, INTER] fp8, sfa [INTER/128, align(P_max,4)]).
std::tuple<at::Tensor, at::Tensor> silu_chunk_mul_quantize_1x128_sorted_sm90(
    at::Tensor gu, at::Tensor flat_to_sorted)
{
    TORCH_CHECK(gu.is_cuda() && gu.dtype() == at::kBFloat16 && gu.dim() == 2 && gu.is_contiguous(),
        "gu must be contiguous CUDA bf16 [P_max, 2*INTER]");
    TORCH_CHECK(flat_to_sorted.dtype() == at::kInt && flat_to_sorted.is_cuda() && flat_to_sorted.is_contiguous(),
        "flat_to_sorted must be contiguous CUDA int32");
    int const P_max = gu.size(0);
    int const TWO_INTER = gu.size(1);
    TORCH_CHECK(TWO_INTER % 2 == 0, "last dim must be even (gate || up)");
    int const INTER = TWO_INTER / 2;
    int const R = flat_to_sorted.numel();
    int const Kb = INTER / 128;
    int const sfa_ld = align_up4(P_max);
    TORCH_CHECK(INTER % 128 == 0, "INTER must be a multiple of 128");

    auto x_q = at::empty({P_max, INTER}, gu.options().dtype(at::kFloat8_e4m3fn));
    auto sfa = at::empty({Kb, sfa_ld}, gu.options().dtype(at::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream();
    detail::fp8bs_silu_chunk_mul_quantize_1x128_fp32_sorted(
        reinterpret_cast<__nv_fp8_e4m3*>(x_q.data_ptr()),
        reinterpret_cast<float*>(sfa.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(gu.data_ptr()),
        reinterpret_cast<int32_t const*>(flat_to_sorted.data_ptr()),
        R, sfa_ld, INTER, stream);
    return {x_q, sfa};
}

} // namespace blockscale_gemm
