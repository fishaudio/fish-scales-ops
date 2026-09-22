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
#include <optional>

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
// In mxfp8_sm100_grouped_kernel.cu (sm_100/sm_103 tcgen05 pointer-array path
// plus the slot-bound decode route it dispatches to).
cudaError_t launch_sm100_mxfp8_grouped_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int32_t* masked_m, int num_groups, int m_cap, int N, int K,
    int expected_m, int max_active_groups, int const* slot_to_expert, cudaStream_t stream);
bool sm100_mxfp8_grouped_compiled();
int sm100_mxfp8_grouped_slot_refusal(int m_cap, int N, int K, int groups, int max_active_groups);
int sm100_mxfp8_grouped_slot_taken(int m_cap, int N, int K, int groups, int max_active_groups);
int sm100_mxfp8_grouped_slot_tile_n();
// Fused-SwiGLU FC1 (sm_100/sm_103) and the host-side router that says whether a
// call should take it. Both grouped routes carry the fusion now -- the
// pointer-array cascade and the swap-orientation slot kernel -- so the
// dispatcher needs the slot route's two extra arguments as well (run
// b300_mxfp8_20260917/M-A2).
cudaError_t launch_sm100_mxfp8_grouped_swiglu_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_fp8_e4m3* H,
    int32_t* SFH, int32_t* SFA, int32_t* SFB, int32_t* masked_m, int num_groups, int m_cap, int N, int K,
    int expected_m, int max_active_groups, int const* slot_to_expert, cudaStream_t stream);
int sm100_mxfp8_grouped_fused_fc1_route(int m_cap, int N, int K, int groups, int max_active_groups);
int sm100_mxfp8_grouped_fused_fc1_available(int N, int K);
void fp8bs_quantize_1x32_packed_grouped_gather(__nv_fp8_e4m3* x_q, int32_t* packed_scales,
    __nv_bfloat16 const* x, int32_t const* slot_of_flat, int n_pairs, int topk,
    int m_cap, int K, cudaStream_t stream, bool use_ue8m0, bool sm1xx_sf_layout = false);
void fp8bs_silu_chunk_mul_quantize_1x32_packed_grouped(__nv_fp8_e4m3* x_q, int32_t* packed_scales,
    __nv_bfloat16 const* gu, int32_t const* slot_of_flat, int n_pairs, int m_cap, int K,
    cudaStream_t stream, bool use_ue8m0, bool sm1xx_sf_layout = false, bool pairwise = false);
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

// Validate an optional caller-supplied slot list and return the device pointer
// the dispatcher wants, or nullptr when no list was given.
//
// The list is read by a kernel that has no way to notice that it is malformed:
// the slot-bound route indexes `masked_m` with whatever expert id it finds and
// would drop or duplicate an expert's output rows rather than fail. The checks
// are therefore all on this side. One entry per group is required because the
// grid bound the caller passes separately may be anything up to G, and the
// device must match `masked_m`'s so the two are read by the same kernel.
int const* check_slot_to_expert(std::optional<at::Tensor> const& slot_to_expert, int G, at::Tensor const& masked_m)
{
    if (!slot_to_expert.has_value())
        return nullptr;
    at::Tensor const& t = *slot_to_expert;
    TORCH_CHECK(t.is_cuda() && t.dtype() == at::kInt, "slot_to_expert must be a CUDA int32 tensor");
    TORCH_CHECK(t.is_contiguous(), "slot_to_expert must be contiguous");
    TORCH_CHECK(t.numel() == G, "slot_to_expert must have one entry per group (G=", G, "), got ", t.numel(),
        "; it is moe_build_routing(..., with_slots=True)'s fourth output");
    TORCH_CHECK(t.device() == masked_m.device(), "slot_to_expert must be on the same device as masked_m");
    return reinterpret_cast<int const*>(t.data_ptr());
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
//
// max_active_groups is a second, optional host-side static hint: an upper
// bound on how many groups can hold at least one row, i.e. min(M * topk, G).
// It cannot be recovered from expected_m — at G = 128 with top-8 routing
// expected_m is 1 for every M up to 16, while the number of groups that can
// hold rows runs from 8 to 128 — and the sm_100 slot-bound decode route needs
// it to size its slot list and its grid. Default 0 means "not supplied", which
// keeps that route switched off. sm_120/121 ignore the argument.
//
// slot_to_expert is the third optional argument and it is a device tensor, not
// a hint. The sm_100 slot-bound route needs the packed list of experts that
// hold at least one routed row — ascending ids in the low entries, -1 after
// them — and when it is not given it builds that list itself with a one-block
// kernel before the GEMM. `moe_build_routing(..., with_slots=True)` produces
// exactly the same list from the histogram it already has, so a caller that
// passes it here removes one launch per grouped GEMM. It is read only by that
// route: on sm_120/121, and on any call the cascade serves, it is validated and
// otherwise unused.

at::Tensor linear_mxfp8_grouped_masked(at::Tensor a_fp8, at::Tensor w_fp8, at::Tensor sa_int32,
    at::Tensor sw_int32, at::Tensor masked_m, int64_t expected_m, int64_t max_active_groups,
    std::optional<at::Tensor> slot_to_expert)
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
    TORCH_CHECK(max_active_groups >= 0 && max_active_groups <= G,
        "max_active_groups must be in [0, G]; got ", max_active_groups, " with G=", G);
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
    int const* const slot_ptr = check_slot_to_expert(slot_to_expert, G, masked_m);

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
        // The slot-bound decode route is correct only while the whole row
        // capacity fits one token tile. Outside that bound it does not fail —
        // it returns 2^-127 times the right answer, or NaN, with no error — so
        // a FSO_GROUPED_SLOT=force that lands on an illegal configuration has
        // to be refused here, before any kernel is launched. The rule itself
        // lives in the dispatcher; this reads its verdict.
        int const slot_refusal = detail::sm100_mxfp8_grouped_slot_refusal(
            m_cap, N, K, G, static_cast<int>(max_active_groups));
        TORCH_CHECK(slot_refusal != 2,
            "FSO_GROUPED_SLOT=force: the sm_100 slot-bound grouped MXFP8 route needs the whole row capacity to "
            "fit one token tile, but m_cap=", m_cap, " exceeds TileN=", detail::sm100_mxfp8_grouped_slot_tile_n(),
            ". Outside that bound the kernel silently scales its output by 2^-127. Unset FSO_GROUPED_SLOT or "
            "call with m_cap <= ", detail::sm100_mxfp8_grouped_slot_tile_n(), ".");
        TORCH_CHECK(slot_refusal != 3,
            "FSO_GROUPED_SLOT=force: no sm_100 slot-bound grouped MXFP8 instantiation covers N=", N, " K=", K, ".");
        TORCH_CHECK(slot_refusal != 4,
            "FSO_GROUPED_SLOT=force: the sm_100 slot-bound grouped MXFP8 route needs max_active_groups > 0 "
            "(= min(M * topk, G)) to size its slot list and grid; the op was called with max_active_groups=0.");
        err = detail::launch_sm100_mxfp8_grouped_dispatch(
            reinterpret_cast<__nv_fp8_e4m3*>(a_fp8.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(w_fp8.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
            reinterpret_cast<int32_t*>(sa_int32.data_ptr()),
            reinterpret_cast<int32_t*>(sw_int32.data_ptr()),
            reinterpret_cast<int32_t*>(masked_m.data_ptr()),
            G, m_cap, N, K, static_cast<int>(expected_m), static_cast<int>(max_active_groups), slot_ptr, stream);
        TORCH_CHECK(err == cudaSuccess, "sm100 mxfp8 grouped kernel error: ", cudaGetErrorString(err));
        return y;
    }
    // sm_120/121 has one grouped route, so neither max_active_groups nor the
    // slot list carries information it can use; both are validated above and
    // then ignored.
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


// linear_mxfp8_grouped_masked_swiglu: the grouped FC1 (gate_up projection) with
// the SwiGLU activation and the MXFP8 requantise folded into its epilogue.
//
// Precondition the kernel CANNOT detect: `w13_fp8` must have its gate and up
// rows INTERLEAVED, row 2j being gate_j and row 2j+1 being up_j, not fso's
// usual [gate; up] stacking. Use `quantize_moe_weights_1x32_fp8(w13,
// w13_interleave=True)` or `interleave_w13_fp8` to produce it. Fed the
// [gate; up] order this op returns a silently wrong answer and raises nothing,
// because the two orders are the same bytes in a different sequence.
//
// What it replaces: `linear_mxfp8_grouped_masked` on the FC1 weights followed
// by `silu_chunk_mul_quantize_1x32_grouped`. The intermediate bf16
// [G, m_cap, 2*I] tensor is never written and the second kernel disappears.
// Both sm_100/sm_103 grouped routes carry the fusion -- the pointer-array
// cascade and the swap-orientation slot kernel of the decode band -- and the
// call is routed between them by the same `slot_route` rule the unfused FC1
// uses, so `max_active_groups` and `slot_to_expert` matter here exactly as they
// do for `linear_mxfp8_grouped_masked`.
//
// Returns (h_fp8 [G, m_cap, I], sh) where sh is the same opaque per-group Sm1xx
// atom slab `silu_chunk_mul_quantize_1x32_grouped` produces, so FC2 consumes
// the pair unchanged. Rows at or beyond masked_m[g] are undefined in both, as
// in every other op of this surface.
std::tuple<at::Tensor, at::Tensor> linear_mxfp8_grouped_masked_swiglu(at::Tensor a_fp8, at::Tensor w13_fp8,
    at::Tensor sa_int32, at::Tensor sw13_int32, at::Tensor masked_m, int64_t expected_m, int64_t max_active_groups,
    std::optional<at::Tensor> slot_to_expert)
{
    TORCH_CHECK(is_sm100_family(),
        "linear_mxfp8_grouped_masked_swiglu is implemented only on sm_100/sm_103 (B200 / B300). The two fused "
        "FC1 epilogues are clones of CUTLASS sm_100 NoSmem epilogues, which exist only on that architecture; "
        "sm_120/121 drives a different grouped kernel and sm_90 has no MXFP8 hardware at all. Use "
        "linear_mxfp8_grouped_masked + silu_chunk_mul_quantize_1x32_grouped there.");
    TORCH_CHECK(detail::sm100_mxfp8_grouped_compiled(),
        "linear_mxfp8_grouped_masked_swiglu on sm_100/sm_103 requires the extension to be built with CUDA >= 12.8 "
        "and TORCH_CUDA_ARCH_LIST including 10.0f (family target for B200 + B300).");
    TORCH_CHECK(a_fp8.is_cuda() && w13_fp8.is_cuda(), "a/w13 must be on CUDA");
    TORCH_CHECK(a_fp8.dtype() == at::kFloat8_e4m3fn && w13_fp8.dtype() == at::kFloat8_e4m3fn,
        "a_fp8 / w13_fp8 must be float8_e4m3fn");
    TORCH_CHECK(sa_int32.dtype() == at::kInt && sw13_int32.dtype() == at::kInt,
        "sa / sw13 must be int32 (packed UE8M0)");
    TORCH_CHECK(masked_m.dtype() == at::kInt && masked_m.is_cuda(), "masked_m must be CUDA int32");
    TORCH_CHECK(a_fp8.dim() == 3 && w13_fp8.dim() == 3, "a_fp8 [G,m_cap,K] / w13_fp8 [G,2*I,K] must be 3D");
    TORCH_CHECK(a_fp8.is_contiguous() && w13_fp8.is_contiguous() && masked_m.is_contiguous(),
        "a/w13/masked_m must be contiguous");

    int const G = a_fp8.size(0);
    int const m_cap = a_fp8.size(1);
    int const K = a_fp8.size(2);
    int const N = w13_fp8.size(1);
    int const INTER = N / 2;
    TORCH_CHECK(w13_fp8.size(0) == G, "w13_fp8.size(0) must equal a_fp8.size(0)");
    TORCH_CHECK(w13_fp8.size(2) == K, "w13_fp8.size(2) must match a_fp8.size(2)");
    TORCH_CHECK(masked_m.numel() == G, "masked_m must have G entries");
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    TORCH_CHECK(N % 2 == 0 && INTER % 128 == 0,
        "w13_fp8.size(1) must be 2*I with I a multiple of 128 (a 1x32 output scale block must not straddle the "
        "atom slab's 128-column block), got N=", N);
    TORCH_CHECK(m_cap % 4 == 0, "m_cap must be a multiple of 4 (per-group scale padding)");
    TORCH_CHECK(expected_m >= 1, "expected_m must be >= 1");
    TORCH_CHECK(max_active_groups >= 0 && max_active_groups <= G,
        "max_active_groups must be in [0, G]; got ", max_active_groups, " with G=", G);
    TORCH_CHECK(G <= 1024, "sm_100/sm_103 grouped MXFP8 supports at most 1024 experts, got ", G);
    int64_t const kp = K / 128;
    int64_t const m_pad = (static_cast<int64_t>(m_cap) + 127) / 128 * 128;
    TORCH_CHECK(sa_int32.numel() == static_cast<int64_t>(G) * m_pad * kp,
        "sa_int32 must be [G, pad(m_cap,128) * K/128] (per-group Sm1xx atom slab)");
    TORCH_CHECK(sw13_int32.numel() == static_cast<int64_t>(G) * kp * N,
        "sw13_int32 must be [G, 2*I * K/128] (per-group Sm1xx atom slab)");
    TORCH_CHECK(sa_int32.is_contiguous() && sw13_int32.is_contiguous(), "sa/sw13 must be contiguous");
    int const* const slot_ptr = check_slot_to_expert(slot_to_expert, G, masked_m);

    auto h = at::empty({G, m_cap, INTER}, a_fp8.options().dtype(at::kFloat8_e4m3fn));
    auto sh = at::empty({G, m_pad * (INTER / 128)}, a_fp8.options().dtype(at::kInt));

    auto stream = at::cuda::getCurrentCUDAStream();
    cudaError_t const err = detail::launch_sm100_mxfp8_grouped_swiglu_dispatch(
        reinterpret_cast<__nv_fp8_e4m3*>(a_fp8.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(w13_fp8.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(h.data_ptr()),
        reinterpret_cast<int32_t*>(sh.data_ptr()),
        reinterpret_cast<int32_t*>(sa_int32.data_ptr()),
        reinterpret_cast<int32_t*>(sw13_int32.data_ptr()),
        reinterpret_cast<int32_t*>(masked_m.data_ptr()),
        G, m_cap, N, K, static_cast<int>(expected_m), static_cast<int>(max_active_groups), slot_ptr, stream);
    TORCH_CHECK(err == cudaSuccess, "sm100 fused-SwiGLU grouped mxfp8 kernel error: ", cudaGetErrorString(err));
    return {h, sh};
}


// mxfp8_grouped_swiglu_fused_route: the host-side router a caller uses to
// decide, ONCE per shape, whether to build its FC1 out of the fused op or out
// of the old pair.
//
// The decision cannot be left to the GEMM, because the two forms need
// different weights (interleaved vs [gate; up]) and different follow-on
// kernels, so it has to be made before the layer is composed. It is the
// negation of the sm_100 slot-route verdict: the fused FC1 lives only on the
// pointer-array route. Every non-sm_100 device answers false, which is the
// same answer as "this architecture has no fused FC1".
bool mxfp8_grouped_swiglu_fused_route(
    int64_t m_cap, int64_t n_w, int64_t k, int64_t num_groups, int64_t max_active_groups)
{
    if (!is_sm100_family() || !detail::sm100_mxfp8_grouped_compiled())
        return false;
    return detail::sm100_mxfp8_grouped_fused_fc1_route(static_cast<int>(m_cap), static_cast<int>(n_w),
               static_cast<int>(k), static_cast<int>(num_groups), static_cast<int>(max_active_groups))
        != 0;
}


// mxfp8_grouped_swiglu_available: the load-time half of the same decision.
//
// The fused FC1 needs interleaved weight rows and the unfused fallback then
// needs the pairwise SwiGLU kernel to match, so the layout is one decision for
// the whole model and has to be taken before the weights are quantised. This
// says whether the fused FC1 can be used at all for a given (N_w, K) on this
// device and under this FSO_FC1_FUSED setting;
// `mxfp8_grouped_swiglu_fused_route` then says whether a particular call takes
// it. Both read the same knob out of the same static.
bool mxfp8_grouped_swiglu_available(int64_t n_w, int64_t k)
{
    if (!is_sm100_family() || !detail::sm100_mxfp8_grouped_compiled())
        return false;
    return detail::sm100_mxfp8_grouped_fused_fc1_available(static_cast<int>(n_w), static_cast<int>(k)) != 0;
}


// mxfp8_grouped_slot_possible: would a grouped GEMM of this shape take the
// slot-bound decode route?
//
// The question a caller actually needs answered is "will anything read the
// packed active-expert list if I ask the routing kernel for it?", and the
// slot route is the only reader. Building the list is not free: the routing
// kernel adds a block-wide compaction of the histogram it already holds, which
// costs the sm_120 layer a quarter to half a microsecond per call and the
// sm_100 layer up to three quarters of one above the decode band -- in both
// cases for a tensor no kernel then looks at. A caller that asks this first
// and passes `with_slots=` accordingly pays for the list exactly where it is
// used. Every architecture but sm_100/103 answers false, because no other
// architecture has the slot route at all.
//
// The verdict comes from the dispatcher's own `slot_route`, not from a
// restatement of its rule, so the two cannot drift apart; FSO_GROUPED_SLOT
// therefore steers this answer exactly as it steers the route.
bool mxfp8_grouped_slot_possible(
    int64_t m_cap, int64_t n_w, int64_t k, int64_t num_groups, int64_t max_active_groups)
{
    if (!is_sm100_family() || !detail::sm100_mxfp8_grouped_compiled())
        return false;
    return detail::sm100_mxfp8_grouped_slot_taken(static_cast<int>(m_cap), static_cast<int>(n_w),
               static_cast<int>(k), static_cast<int>(num_groups), static_cast<int>(max_active_groups))
        != 0;
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
//
// `pairwise` selects how the gate and the up halves are found inside a row.
// With pairwise=false (the default, and the only behaviour before the fused
// FC1 existed) the row is [gate_0..gate_{I-1}, up_0..up_{I-1}], so gate_i and
// up_i are I columns apart. With pairwise=true the row is
// [gate_0, up_0, gate_1, up_1, ...], i.e. gate_i is at column 2i and up_i at
// column 2i+1. The second layout exists because a layer that carries
// interleaved FC1 weights — which the fused FC1 requires — produces its
// gate_up tensor in that order, and on the decode band the layer still runs
// the unfused FC1 through the slot route and therefore still needs this
// kernel. Passing the wrong value multiplies the wrong pairs together and
// raises nothing; the two layouts are the same bytes in a different order.
std::tuple<at::Tensor, at::Tensor> silu_chunk_mul_quantize_1x32_grouped(
    at::Tensor gu, at::Tensor slot_of_flat, bool use_ue8m0, bool pairwise)
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
    // The pairwise form is instantiated only in the sm_100 scale-layout
    // combination, because the layout it serves only arises from the sm_100
    // fused FC1's interleaved weights. Refusing it elsewhere is what keeps the
    // sm_90 and sm_120 device code byte-identical to what it was.
    TORCH_CHECK(!pairwise || sm1xx,
        "silu_chunk_mul_quantize_1x32_grouped(pairwise=True) exists only on sm_100/sm_103: the interleaved "
        "gate/up row order it reads is produced by the sm_100 fused FC1 weight layout, which no other "
        "architecture has.");
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
        static_cast<int>(slot_of_flat.numel()), m_cap, INTER, stream, use_ue8m0, sm1xx, pairwise);
    return {x_q, packed};
}

} // namespace blockscale_gemm
