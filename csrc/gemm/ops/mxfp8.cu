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
// In mxfp8_sm100_kernel.cu (sm_100/sm_103 tcgen05 BlockScaled path).
cudaError_t launch_sm100_mxfp8_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int M, int N, int K, cudaStream_t stream);
bool sm100_mxfp8_compiled();
// In quant_kernels.cu. sm1xx_sf_layout selects the Sm1xxBlockScaledConfig
// atom scale layout (sm_100/sm_103) instead of the sm_120 int32 K-major.
void fp8bs_quantize_1x32(__nv_fp8_e4m3* x_q, float* scales, __nv_bfloat16 const* x, int M, int K, cudaStream_t stream,
    bool use_ue8m0);
void fp8bs_quantize_1x32_packed(__nv_fp8_e4m3* x_q, int32_t* packed_scales,
    __nv_bfloat16 const* x, int M, int K, cudaStream_t stream, bool use_ue8m0, bool sm1xx_sf_layout = false);
void fp8bs_silu_chunk_mul_quantize_1x32_packed(__nv_fp8_e4m3* x_q,
    int32_t* packed_scales, __nv_bfloat16 const* gu, int M, int K,
    cudaStream_t stream, bool use_ue8m0, bool sm1xx_sf_layout = false);
void repack_ue8m0_scales_for_sm120(int32_t* dst, float const* src, int M_pad, int K_blocks_in,
    cudaStream_t stream);
// Grouped (MoE, masked layout) variants — see quant_kernels.cu and
// mxfp8_kernel.cu for the layout contracts.
cudaError_t launch_sm120_mxfp8_grouped_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int32_t* masked_m, int num_groups, int m_cap, int N, int K,
    int expected_m, cudaStream_t stream);
// In mxfp8_sm100_grouped_kernel.cu (sm_100/sm_103 tcgen05 pointer-array path).
cudaError_t launch_sm100_mxfp8_grouped_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int32_t* masked_m, int num_groups, int m_cap, int N, int K,
    int expected_m, cudaStream_t stream);
bool sm100_mxfp8_grouped_compiled();
void fp8bs_quantize_1x32_packed_grouped_gather(__nv_fp8_e4m3* x_q, int32_t* packed_scales,
    __nv_bfloat16 const* x, int32_t const* slot_of_flat, int n_pairs, int topk,
    int m_cap, int K, cudaStream_t stream, bool use_ue8m0, bool sm1xx_sf_layout = false);
void fp8bs_silu_chunk_mul_quantize_1x32_packed_grouped(__nv_fp8_e4m3* x_q, int32_t* packed_scales,
    __nv_bfloat16 const* gu, int32_t const* slot_of_flat, int n_pairs, int m_cap, int K,
    cudaStream_t stream, bool use_ue8m0, bool sm1xx_sf_layout = false);
} // namespace detail

namespace
{

// Current device's SM major version, cached per device id. MXFP8 routes:
//   10 (sm_100 B200 / sm_103 B300) → tcgen05 BlockScaled (CUTLASS builder)
//   12 (sm_120/121 consumer)       → Sm120BlockScaledKernel
bool is_sm100_family()
{
    auto const* prop = at::cuda::getCurrentDeviceProperties();
    return prop->major == 10;
}

} // anonymous namespace


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
    int const K_blocks_out = K / 128;     // packed int32 scale columns
    bool const sm1xx = is_sm100_family();

    auto x_q = at::empty({M, K}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor packed;
    if (sm1xx)
    {
        // Sm1xxBlockScaledConfig atom layout (sm_100/sm_103): opaque 1-D
        // buffer, 128 int32 words per (128-row × 128-K) block, blocks
        // K-minor. No 2-D stride annotation can describe it — treat as an
        // opaque handle sized [ceil(M/128)*128 * K/128] int32.
        int const M_pad = (M + 127) / 128 * 128;
        packed = at::empty({static_cast<int64_t>(M_pad) * K_blocks_out}, x.options().dtype(at::kInt));
    }
    else
    {
        int const M_pad = (M + 3) / 4 * 4;
        // K-major (stride 1 along M, M_pad along K). Matches `repack_mxfp8_scales`.
        packed = at::empty_strided({M_pad, K_blocks_out}, {1, M_pad}, x.options().dtype(at::kInt));
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    detail::fp8bs_quantize_1x32_packed(
        reinterpret_cast<__nv_fp8_e4m3*>(x_q.data_ptr()),
        reinterpret_cast<int32_t*>(packed.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(x_flat.data_ptr()),
        M, K, stream, use_ue8m0, sm1xx);

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
    int const K_blocks_out = INTER / 128;
    bool const sm1xx = is_sm100_family();

    auto out_sizes = gu.sizes().vec();
    out_sizes.back() = INTER;
    auto x_q = at::empty(out_sizes, gu.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor packed;
    if (sm1xx)
    {
        // Opaque Sm1xx atom layout — see quantize_1x32_packed.
        int const M_pad = (M + 127) / 128 * 128;
        packed = at::empty({static_cast<int64_t>(M_pad) * K_blocks_out}, gu.options().dtype(at::kInt));
    }
    else
    {
        int const M_pad = (M + 3) / 4 * 4;
        packed = at::empty_strided({M_pad, K_blocks_out}, {1, M_pad}, gu.options().dtype(at::kInt));
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    detail::fp8bs_silu_chunk_mul_quantize_1x32_packed(
        reinterpret_cast<__nv_fp8_e4m3*>(x_q.data_ptr()),
        reinterpret_cast<int32_t*>(packed.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(gu_flat.data_ptr()),
        M, INTER, stream, use_ue8m0, sm1xx);

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
    cudaError_t err;
    if (is_sm100_family())
    {
        TORCH_CHECK(detail::sm100_mxfp8_compiled(),
            "linear_mxfp8 on sm_100/sm_103 requires the extension to be built with CUDA >= 12.8 "
            "and TORCH_CUDA_ARCH_LIST including 10.0f (family target for B200 + B300).");
        // Scales must be in the Sm1xx atom layout (quantize_1x32_packed
        // emits it automatically on sm_100-family devices).
        err = detail::launch_sm100_mxfp8_dispatch(
            reinterpret_cast<__nv_fp8_e4m3*>(x_fp8.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(w_fp8.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
            reinterpret_cast<int32_t*>(sx_int32.data_ptr()),
            reinterpret_cast<int32_t*>(sw_int32.data_ptr()),
            M, N, K, stream);
        TORCH_CHECK(err == cudaSuccess, "sm100 mxfp8 kernel error: ", cudaGetErrorString(err));
        return y;
    }
    err = detail::launch_sm120_mxfp8_dispatch(
        reinterpret_cast<__nv_fp8_e4m3*>(x_fp8.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(w_fp8.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        reinterpret_cast<int32_t*>(sx_int32.data_ptr()),
        reinterpret_cast<int32_t*>(sw_int32.data_ptr()),
        M, N, K, stream);
    TORCH_CHECK(err == cudaSuccess, "sm120 mxfp8 kernel error: ", cudaGetErrorString(err));
    return y;
}


// ---------------------------------------------------------------------------
// Grouped (MoE, masked layout) surface — sm_120/121 (M1) and sm_100/103 (M3).
//
// Layout contracts (G = num experts/groups, m_cap = per-group row capacity,
// m_cap % 4 == 0):
//   a_fp8    [G, m_cap, K]        rows >= masked_m[g] undefined
//   w_fp8    [G, N, K]            per-expert weights
//   out      [G, m_cap, N] bf16   rows >= masked_m[g] undefined
// The two scale tensors are opaque per-group slabs whose byte layout differs
// by architecture, because the two GEMM engines read scale factors through
// different descriptors:
//   sm_120/121: sa [G, K/128, m_cap], sw [G, K/128, N] — int32 K-major words
//               per group (N % 128 == 0, so pad(N,4) == N).
//   sm_100/103: sa [G, pad(m_cap,128) * K/128], sw [G, N * K/128] — per group
//               one CUTLASS Sm1xxBlockScaledConfig<32> atom slab, the same
//               layout the dense sm_100 path uses, applied per group.
// Always produce them with the grouped quantize ops on the device the GEMM
// will run on; the layouts are not interchangeable.
//
// masked_m is int32 [G] on device and is read ONLY by the kernel — never on
// the host — so calls are CUDA-Graph capture-safe with dynamic routing.
// expected_m is a host-side static tile-selection hint (ceil(rows/G)).

at::Tensor linear_mxfp8_grouped_masked(at::Tensor a_fp8, at::Tensor w_fp8, at::Tensor sa_int32,
    at::Tensor sw_int32, at::Tensor masked_m, int64_t expected_m)
{
    TORCH_CHECK(a_fp8.is_cuda() && w_fp8.is_cuda(), "a/w must be on CUDA");
    TORCH_CHECK(a_fp8.dtype() == at::kFloat8_e4m3fn && w_fp8.dtype() == at::kFloat8_e4m3fn,
        "a_fp8 / w_fp8 must be float8_e4m3fn");
    TORCH_CHECK(sa_int32.dtype() == at::kInt && sw_int32.dtype() == at::kInt,
        "sa / sw must be int32 (packed UE8M0)");
    TORCH_CHECK(masked_m.dtype() == at::kInt && masked_m.is_cuda(), "masked_m must be CUDA int32");
    TORCH_CHECK(a_fp8.dim() == 3 && w_fp8.dim() == 3, "a_fp8 [G,m_cap,K] / w_fp8 [G,N,K] must be 3D");
    TORCH_CHECK(a_fp8.is_contiguous() && w_fp8.is_contiguous() && masked_m.is_contiguous(),
        "a/w/masked_m must be contiguous");

    int const G = a_fp8.size(0);
    int const m_cap = a_fp8.size(1);
    int const K = a_fp8.size(2);
    int const N = w_fp8.size(1);
    TORCH_CHECK(w_fp8.size(0) == G, "w_fp8.size(0) must equal a_fp8.size(0)");
    TORCH_CHECK(w_fp8.size(2) == K, "w_fp8.size(2) must match a_fp8.size(2)");
    TORCH_CHECK(masked_m.numel() == G, "masked_m must have G entries");
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128 (BlockScaled atom requires it)");
    TORCH_CHECK(m_cap % 4 == 0, "m_cap must be a multiple of 4 (per-group scale padding)");
    TORCH_CHECK(expected_m >= 1, "expected_m must be >= 1");
    int64_t const kp = K / 128;
    bool const sm1xx = is_sm100_family();
    int64_t const sa_words = sm1xx ? (static_cast<int64_t>((m_cap + 127) / 128 * 128) * kp)
                                   : (static_cast<int64_t>(m_cap) * kp);
    TORCH_CHECK(sa_int32.numel() == static_cast<int64_t>(G) * sa_words,
        sm1xx ? "sa_int32 must be [G, pad(m_cap,128) * K/128] (per-group Sm1xx atom slab)"
              : "sa_int32 must be [G, K/128, m_cap] (per-group K-major packed scales)");
    TORCH_CHECK(sw_int32.numel() == static_cast<int64_t>(G) * kp * N,
        sm1xx ? "sw_int32 must be [G, N * K/128] (per-group Sm1xx atom slab)"
              : "sw_int32 must be [G, K/128, N] (per-group K-major packed scales)");
    TORCH_CHECK(sa_int32.is_contiguous() && sw_int32.is_contiguous(), "sa/sw must be contiguous");

    auto y = at::empty({G, m_cap, N}, a_fp8.options().dtype(at::kBFloat16));
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err;
    if (sm1xx)
    {
        TORCH_CHECK(detail::sm100_mxfp8_grouped_compiled(),
            "linear_mxfp8_grouped_masked on sm_100/sm_103 requires the extension to be built with CUDA >= 12.8 "
            "and TORCH_CUDA_ARCH_LIST including 10.0f (family target for B200 + B300).");
        // The sm_100 launcher builds its per-group CUTLASS argument arrays in a
        // pool sized once for this many groups (same bound moe_build_routing
        // enforces), so refuse anything larger rather than overrun it.
        TORCH_CHECK(G <= 1024, "sm_100/sm_103 grouped MXFP8 supports at most 1024 experts, got ", G);
        err = detail::launch_sm100_mxfp8_grouped_dispatch(
            reinterpret_cast<__nv_fp8_e4m3*>(a_fp8.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(w_fp8.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
            reinterpret_cast<int32_t*>(sa_int32.data_ptr()),
            reinterpret_cast<int32_t*>(sw_int32.data_ptr()),
            reinterpret_cast<int32_t*>(masked_m.data_ptr()),
            G, m_cap, N, K, static_cast<int>(expected_m), stream);
        TORCH_CHECK(err == cudaSuccess, "sm100 mxfp8 grouped kernel error: ", cudaGetErrorString(err));
        return y;
    }
    err = detail::launch_sm120_mxfp8_grouped_dispatch(
        reinterpret_cast<__nv_fp8_e4m3*>(a_fp8.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(w_fp8.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        reinterpret_cast<int32_t*>(sa_int32.data_ptr()),
        reinterpret_cast<int32_t*>(sw_int32.data_ptr()),
        reinterpret_cast<int32_t*>(masked_m.data_ptr()),
        G, m_cap, N, K, static_cast<int>(expected_m), stream);
    TORCH_CHECK(err == cudaSuccess, "sm120 mxfp8 grouped kernel error: ", cudaGetErrorString(err));
    return y;
}


// quantize_1x32_grouped_gather: fused token-gather + MXFP8 quantize into the
// masked grouped layout, indexed over the flat routed-pair space. x is the
// flat [M, K] bf16 activation; slot_of_flat (int32 [M * topk], from
// moe_build_routing) maps pair i -> destination slot g*m_cap + m_in; the
// source row is i / topk. num_groups is implied by the output shape:
// pass G explicitly so the scale buffer can be sized without a device read.
std::tuple<at::Tensor, at::Tensor> quantize_1x32_grouped_gather(
    at::Tensor x, at::Tensor slot_of_flat, int64_t topk, int64_t num_groups,
    int64_t m_cap, bool use_ue8m0)
{
    TORCH_CHECK(x.is_cuda() && x.dtype() == at::kBFloat16, "x must be CUDA bf16");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K]");
    TORCH_CHECK(slot_of_flat.dtype() == at::kInt && slot_of_flat.is_cuda()
            && slot_of_flat.is_contiguous(),
        "slot_of_flat must be contiguous CUDA int32");
    int const M = x.size(0);
    int const K = x.size(1);
    int const G = static_cast<int>(num_groups);
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    TORCH_CHECK(topk >= 1, "topk must be >= 1");
    TORCH_CHECK(m_cap >= 1 && m_cap % 4 == 0, "m_cap must be a positive multiple of 4");
    TORCH_CHECK(slot_of_flat.numel() == static_cast<int64_t>(M) * topk,
        "slot_of_flat must have M * topk entries");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int const kp = K / 128;
    bool const sm1xx = is_sm100_family();
    auto x_q = at::empty({G, m_cap, K}, x.options().dtype(at::kFloat8_e4m3fn));
    // sm_120: per-group K-major [K/128, m_cap]. sm_100/103: one opaque Sm1xx
    // atom slab per group, sized for pad(m_cap,128) rows.
    auto packed = sm1xx
        ? at::empty({G, static_cast<int64_t>((m_cap + 127) / 128 * 128) * kp}, x.options().dtype(at::kInt))
        : at::empty({G, kp, static_cast<int64_t>(m_cap)}, x.options().dtype(at::kInt));

    auto stream = at::cuda::getCurrentCUDAStream();
    detail::fp8bs_quantize_1x32_packed_grouped_gather(
        reinterpret_cast<__nv_fp8_e4m3*>(x_q.data_ptr()),
        reinterpret_cast<int32_t*>(packed.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr()),
        reinterpret_cast<int32_t const*>(slot_of_flat.data_ptr()),
        static_cast<int>(M * topk), static_cast<int>(topk),
        static_cast<int>(m_cap), K, stream, use_ue8m0, sm1xx);
    return {x_q, packed};
}


// silu_chunk_mul_quantize_1x32_grouped: SwiGLU fused quant for the grouped
// down-GEMM input, indexed over the flat routed-pair space (row = slot).
// gu is the grouped gate_up output (bf16 [G, m_cap, 2*INTER]).
std::tuple<at::Tensor, at::Tensor> silu_chunk_mul_quantize_1x32_grouped(
    at::Tensor gu, at::Tensor slot_of_flat, bool use_ue8m0)
{
    TORCH_CHECK(gu.is_cuda() && gu.dtype() == at::kBFloat16, "gu must be CUDA bf16");
    TORCH_CHECK(gu.dim() == 3, "gu must be 3D [G, m_cap, 2*INTER]");
    TORCH_CHECK(gu.is_contiguous(), "gu must be contiguous");
    TORCH_CHECK(slot_of_flat.dtype() == at::kInt && slot_of_flat.is_cuda()
            && slot_of_flat.is_contiguous(),
        "slot_of_flat must be contiguous CUDA int32");
    int const G = gu.size(0);
    int const m_cap = gu.size(1);
    int const TWO_INTER = gu.size(2);
    TORCH_CHECK(TWO_INTER % 2 == 0, "last dim must be even (gate || up)");
    int const INTER = TWO_INTER / 2;
    TORCH_CHECK(INTER % 128 == 0, "INTER must be a multiple of 128");
    TORCH_CHECK(m_cap % 4 == 0, "m_cap must be a multiple of 4");

    int const kp = INTER / 128;
    bool const sm1xx = is_sm100_family();
    auto x_q = at::empty({G, m_cap, INTER}, gu.options().dtype(at::kFloat8_e4m3fn));
    auto packed = sm1xx
        ? at::empty({G, static_cast<int64_t>((m_cap + 127) / 128 * 128) * kp}, gu.options().dtype(at::kInt))
        : at::empty({G, kp, static_cast<int64_t>(m_cap)}, gu.options().dtype(at::kInt));

    auto stream = at::cuda::getCurrentCUDAStream();
    detail::fp8bs_silu_chunk_mul_quantize_1x32_packed_grouped(
        reinterpret_cast<__nv_fp8_e4m3*>(x_q.data_ptr()),
        reinterpret_cast<int32_t*>(packed.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(gu.data_ptr()),
        reinterpret_cast<int32_t const*>(slot_of_flat.data_ptr()),
        static_cast<int>(slot_of_flat.numel()), m_cap, INTER, stream, use_ue8m0, sm1xx);
    return {x_q, packed};
}

} // namespace blockscale_gemm
