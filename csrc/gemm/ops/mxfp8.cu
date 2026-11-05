/*
 * ATen ops for the MXFP8 (1×32, true OCP MXFP8) block-scaled GEMMs, dense and
 * grouped (MoE), on sm_100/sm_103 (CUTLASS tcgen05 BlockScaled) and
 * sm_120/sm_121 (CUTLASS Sm120 BlockScaled, kSFVecSize=32), with the 1×32
 * quantizers, the grouped gather / SwiGLU requantize ops and the host-side
 * route queries. Every op takes the arch-native packed int32 scales the
 * quantizers of the same architecture produce.
 *
 * Forward-decl of the launcher so cute headers don't leak into this TU
 * and clash with at::Layout.
 */

#include "input_alignment.h"
#include "dense_epilogue_args.h"
#include "blockscale_gemm/arch/sm120/mxfp8/smallm_route.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace blockscale_gemm
{
namespace detail
{
cudaError_t launch_sm120_mxfp8_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int M, int N, int K, cudaStream_t stream, fso::DenseEpilogue const& epilogue,
    void* streamk_scratch, std::size_t streamk_scratch_bytes);
// In mxfp8_kernel.cu: the route's epilogue check and its Stream-K scratch size (host only), and the route query.
std::size_t sm120_mxfp8_dense_prepare(int M, int N, int K, fso::DenseEpilogue const& epilogue);
void sm120_mxfp8_dense_route_query(int M, int N, int K, int64_t* out8);
// The sm_120/sm_121 small-M route (ops/mxfp8_smallm_kernel.cu; rules and plan in arch/sm120/mxfp8/smallm_route.h).
bool sm120_smallm_takes(int M, int N, int K, bool bf16x);
void sm120_smallm_route_query(int M, int N, int K, bool bf16x, int64_t* out10);
cudaError_t launch_sm120_mxfp8_smallm(int form, void const* x, int32_t const* sx, __nv_fp8_e4m3 const* w,
    int32_t const* ws, __nv_bfloat16* y, int M, int N, int K, fso::DenseEpilogue const& epi, cudaStream_t stream);
// In mxfp8_sm100_kernel.cu (sm_100/sm_103 tcgen05 BlockScaled path). The dense cascade takes its per-call
// scratch from the caller: ask for the size, allocate it on the launch stream, pass it down.
cudaError_t launch_sm100_mxfp8_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int M, int N, int K, cudaStream_t stream, fso::DenseEpilogue const& epilogue,
    void* workspace, std::size_t workspace_bytes);
std::size_t launch_sm100_mxfp8_dispatch_workspace_bytes(int M, int N, int K, fso::DenseEpilogue const& epilogue);
bool sm100_mxfp8_compiled();
char const* sm100_mxfp8_error_hint(cudaError_t err);
// In quant_kernels.cu. sm1xx_sf_layout selects the Sm1xxBlockScaledConfig
// atom scale layout (sm_100/sm_103) instead of the sm_120 int32 K-major.
void fp8bs_quantize_1x32(__nv_fp8_e4m3* x_q, float* scales, __nv_bfloat16 const* x, int M, int K, cudaStream_t stream,
    bool use_ue8m0);
void fp8bs_quantize_1x32_packed(__nv_fp8_e4m3* x_q, int32_t* packed_scales,
    __nv_bfloat16 const* x, int M, int K, cudaStream_t stream, bool use_ue8m0, bool sm1xx_sf_layout = false);
void fp8bs_silu_chunk_mul_quantize_1x32_packed(__nv_fp8_e4m3* x_q,
    int32_t* packed_scales, __nv_bfloat16 const* gu, int M, int K,
    cudaStream_t stream, bool use_ue8m0, bool sm1xx_sf_layout = false);
// The width of the vector loads with which the kernel each of the two launchers above selects reads its bf16
// input (16 or 8 bytes; x_q nullptr stands for a fresh allocation). The op bodies refuse an input whose rows that
// kernel cannot read (input_alignment.h), before anything is allocated or launched. fp8bs_quantize_1x32 reads
// one element per load and needs no alignment.
int fp8bs_quantize_1x32_packed_x_vector_bytes(__nv_fp8_e4m3 const* x_q, __nv_bfloat16 const* x, bool sm1xx_sf_layout);
int fp8bs_silu_chunk_mul_quantize_1x32_packed_x_vector_bytes(
    __nv_fp8_e4m3 const* x_q, __nv_bfloat16 const* gu, bool sm1xx_sf_layout);
void repack_ue8m0_scales_for_sm120(int32_t* dst, float const* src, int M_pad, int K_blocks_in,
    cudaStream_t stream);
// Grouped (MoE, masked layout) variants — see quant_kernels.cu and
// mxfp8_kernel.cu for the layout contracts.
cudaError_t launch_sm120_mxfp8_grouped_combine_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B,
    __nv_bfloat16* out_tokens, int32_t* SFA, int32_t* SFB, int32_t* masked_m, int32_t const* row_map,
    float const* weight_of_slot, int num_groups, int m_cap, int N, int K, int expected_m, int max_active_groups,
    cudaStream_t stream);
cudaError_t launch_sm120_mxfp8_grouped_swiglu_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_fp8_e4m3* out_fp8,
    int32_t* out_sf, int32_t* SFA, int32_t* SFB, int32_t* masked_m, int num_groups, int m_cap, int N, int K,
    int expected_m, int max_active_groups, cudaStream_t stream);
cudaError_t launch_sm120_mxfp8_grouped_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int32_t* masked_m, int num_groups, int m_cap, int N, int K,
    int expected_m, int max_active_groups, cudaStream_t stream);
// In mxfp8_sm100_grouped_kernel.cu (sm_100/sm_103 tcgen05 pointer-array path
// plus the slot-bound decode route it dispatches to).
cudaError_t launch_sm100_mxfp8_grouped_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int32_t* masked_m, int num_groups, int m_cap, int N, int K,
    int expected_m, int max_active_groups, int const* slot_to_expert, int32_t const* problem_shapes,
    cudaStream_t stream);
bool sm100_mxfp8_grouped_compiled();
int sm100_mxfp8_grouped_slot_refusal(int m_cap, int N, int K, int groups, int max_active_groups, int compact);
int sm100_mxfp8_grouped_slot_taken(
    int m_cap, int N, int K, int groups, int max_active_groups, int fused_swiglu, int compact, int expected_m);
int sm100_mxfp8_grouped_slot_tile_n();
int sm100_mxfp8_grouped_problem_shapes_consumed(
    int m_cap, int N, int K, int groups, int max_active_groups, int fused_swiglu, int compact, int expected_m);
// Fused-SwiGLU FC1 (sm_100/sm_103) and the host-side router that says whether a
// call should take it. Both grouped routes carry the fusion now -- the
// pointer-array cascade and the swap-orientation slot kernel -- so the
// dispatcher needs the slot route's two extra arguments as well (run
// b300_mxfp8_20260917/M-A2).
cudaError_t launch_sm100_mxfp8_grouped_swiglu_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_fp8_e4m3* H,
    int32_t* SFH, int32_t* SFA, int32_t* SFB, int32_t* masked_m, int num_groups, int m_cap, int N, int K,
    int expected_m, int max_active_groups, int const* slot_to_expert, int32_t const* problem_shapes,
    cudaStream_t stream);
int sm100_mxfp8_grouped_fused_fc1_route(
    int m_cap, int N, int K, int groups, int max_active_groups, int compact, int expected_m);
// The compact activation layout (fso 0.2.1), same translation unit.
cudaError_t launch_sm100_mxfp8_grouped_compact_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int32_t* masked_m, int32_t const* blk0, int num_groups, int b_alloc, int m_cap, int N,
    int K, int expected_m, int max_active_groups, int const* slot_to_expert, int32_t const* problem_shapes,
    int64_t const* ptr_arrays, int* slot_sched, int* slot_tiles, int tiles_cap, bool build_tiles, cudaStream_t stream);
cudaError_t launch_sm100_mxfp8_grouped_swiglu_compact_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B,
    __nv_fp8_e4m3* H, int32_t* SFH, int32_t* SFA, int32_t* SFB, int32_t* masked_m, int32_t const* blk0,
    int num_groups, int b_alloc, int m_cap, int N, int K, int expected_m, int max_active_groups,
    int const* slot_to_expert, int32_t const* problem_shapes, int64_t const* ptr_arrays, int* slot_sched,
    int* slot_tiles, int tiles_cap, bool build_tiles, cudaStream_t stream);
int sm100_mxfp8_grouped_compact_sfa_rows();
int sm100_mxfp8_grouped_compact_fits(long long rows, int K);
// The persistent FC1 + FC2 slot kernel of the MoE layer (issue #13 lever 5), in mxfp8_sm100_slot_moe_kernel.cu.
bool sm100_mxfp8_slot_moe_compiled();
bool sm100_mxfp8_slot_moe_shape_ok(int n1, int k1, int n2, int k2);
cudaError_t launch_sm100_mxfp8_slot_moe(__nv_fp8_e4m3 const* a, int32_t const* sa, __nv_fp8_e4m3 const* w13,
    int32_t const* sw13, __nv_fp8_e4m3 const* w2, int32_t const* sw2, __nv_fp8_e4m3* dq, int32_t* sd,
    __nv_bfloat16* dn, int32_t const* masked_m, int32_t const* blk0, int const* slot_to_expert, int* counters,
    int groups, int num_slots, int m_cap, int b_alloc, int n1, int k1, int n2, int k2, cudaStream_t stream);
// Its token-tile mode (issue #13 lever 5 step 3), in mxfp8_sm100_slot_moe_list_kernel.cu.
cudaError_t launch_sm100_mxfp8_slot_moe_list(__nv_fp8_e4m3 const* a, int32_t const* sa, __nv_fp8_e4m3 const* w13,
    int32_t const* sw13, __nv_fp8_e4m3 const* w2, int32_t const* sw2, __nv_fp8_e4m3* dq, int32_t* sd,
    __nv_bfloat16* dn, int32_t const* masked_m, int32_t const* blk0, int const* slot_to_expert, int* counters,
    int const* slot_tiles, int tiles_cap, int groups, int m_cap, int b_alloc, int n1, int k1, int n2, int k2,
    cudaStream_t stream);
int sm100_mxfp8_grouped_fused_fc1_available(int N, int K);
void fp8bs_quantize_1x32_packed_grouped_gather(__nv_fp8_e4m3* x_q, int32_t* packed_scales,
    __nv_bfloat16 const* x, int32_t const* slot_of_flat, int n_pairs, int topk,
    int m_cap, int K, cudaStream_t stream, bool use_ue8m0, bool sm1xx_sf_layout = false);
void fp8bs_silu_chunk_mul_quantize_1x32_packed_grouped(__nv_fp8_e4m3* x_q, int32_t* packed_scales,
    __nv_bfloat16 const* gu, int32_t const* slot_of_flat, int n_pairs, int m_cap, int K,
    cudaStream_t stream, bool use_ue8m0, bool sm1xx_sf_layout = false, bool pairwise = false);
// The width of the vector loads with which the kernels of the two grouped quantizers above read their bf16 input
// (8 bytes; 16 for the pairwise SwiGLU form); the op bodies refuse an input whose rows those kernels cannot read
// (input_alignment.h), before anything is allocated or launched.
int fp8bs_quantize_1x32_packed_grouped_gather_x_vector_bytes();
int fp8bs_silu_chunk_mul_quantize_1x32_packed_grouped_x_vector_bytes(bool pairwise);
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

// FSO_FC1_FUSED: 0 never uses the fused FC1 (so a caller driven by
// mxfp8_grouped_swiglu_available also keeps the stacked weight layout), unset
// applies the arch's rule, 1 uses it wherever legal. Read once per process; the
// sm_100/103 path reads the same variable inside its own dispatcher.
bool fso_fc1_fused_enabled()
{
    static bool const v = []
    {
        char const* e = std::getenv("FSO_FC1_FUSED");
        return !(e != nullptr && e[0] == '0');
    }();
    return v;
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

// Validate an optional caller-supplied per-group problem-shape array (run
// b300_round3_20260922/M-A4) and return the device pointer the sm_100
// dispatcher wants, or nullptr when none was given.
//
// The array is int32 [G, 3], one (rows, N, K) triple per group, and it is
// `moe_build_routing(..., problem_shapes_for=[...])`'s output for THIS GEMM's
// (N, K). The kernel that reads it -- CUTLASS's grouped tile scheduler -- has
// no way to notice a triple built for another GEMM: a wrong N or K there
// walks tiles that do not exist and reads past the operands, and a row count
// above m_cap does the same on the activation slab. Shape, dtype, layout and
// device are therefore checked here on every call, and the CONTENT can be
// checked as well, at the cost of a device-to-host copy, by setting
// FSO_CHECK_PROBLEM_SHAPES=1; that is a debugging aid, read on every call so
// it can be switched on for one process without a rebuild, and it is never on
// in a timed path.
int32_t const* check_problem_shapes(std::optional<at::Tensor> const& problem_shapes, int G, int m_cap, int N, int K,
    at::Tensor const& masked_m)
{
    if (!problem_shapes.has_value())
        return nullptr;
    at::Tensor const& t = *problem_shapes;
    TORCH_CHECK(t.is_cuda() && t.dtype() == at::kInt, "problem_shapes must be a CUDA int32 tensor");
    TORCH_CHECK(t.is_contiguous(), "problem_shapes must be contiguous");
    TORCH_CHECK(t.dim() == 2 && t.size(0) == G && t.size(1) == 3,
        "problem_shapes must be [G, 3] with G=", G, " (one (rows, N, K) triple per group), got ", t.sizes(),
        "; it is one entry of moe_build_routing(..., problem_shapes_for=[(N, K), ...])'s last output");
    TORCH_CHECK(t.device() == masked_m.device(), "problem_shapes must be on the same device as masked_m");
    char const* check = std::getenv("FSO_CHECK_PROBLEM_SHAPES");
    if (check && check[0] == '1')
    {
        at::Tensor const h = t.to(at::kCPU);
        auto const acc = h.accessor<int32_t, 2>();
        for (int g = 0; g < G; ++g)
        {
            TORCH_CHECK(acc[g][1] == N && acc[g][2] == K,
                "FSO_CHECK_PROBLEM_SHAPES: problem_shapes[", g, "] = (", acc[g][0], ", ", acc[g][1], ", ",
                acc[g][2], ") was built for another GEMM; this call has N=", N, " K=", K);
            TORCH_CHECK(acc[g][0] >= 0 && acc[g][0] <= m_cap,
                "FSO_CHECK_PROBLEM_SHAPES: problem_shapes[", g, "] has ", acc[g][0], " rows, outside [0, m_cap=",
                m_cap, "]");
        }
    }
    return reinterpret_cast<int32_t const*>(t.data_ptr());
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
    detail::require_bf16_vector_aligned(x_flat, M, K,
        detail::fp8bs_quantize_1x32_packed_x_vector_bytes(
            nullptr, reinterpret_cast<__nv_bfloat16 const*>(x_flat.data_ptr()), sm1xx),
        "fish_scales_ops::quantize_1x32_packed", "x");

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
    detail::require_bf16_vector_aligned(gu_flat, M, TWO_INTER,
        detail::fp8bs_silu_chunk_mul_quantize_1x32_packed_x_vector_bytes(
            nullptr, reinterpret_cast<__nv_bfloat16 const*>(gu_flat.data_ptr()), sm1xx),
        "fish_scales_ops::silu_chunk_mul_quantize_1x32", "gu");

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
//
// bias / residual / act: the dense epilogue of interface v1 (dense_epilogue_args.h). The sm_100/sm_103 cascade
// applies bias, erf-GELU and the residual, and sm_120/sm_121 applies them on every route of its cascade
// (kDenseEpiServedSm120Mxfp8Cascade). A field the route does not serve (a forced configuration without a
// dense-epilogue instantiation on sm_100/sm_103, a forced diagnostic instance on sm_120) is refused with
// NotImplementedError before anything is launched or allocated.
//
// Workspace (sm_100/sm_103, issue #12 charter 7.5): the bytes the cascade's choice needs (the two-kernel split-K's
// FP32 partials) are allocated here with the torch caching allocator on the current stream, the stream the GEMM is
// launched on, so two streams never share it and a capture takes graph-private memory.
//
// The Stream-K scratch (sm_120): when the route splits K, its bf16 partial scratch is allocated here, from the torch
// caching allocator on the current stream (graph-private inside a capture), and lives until the op returns; the
// launches that write and read it are all ordered before the op's last kernel on that stream (the side-stream slices
// join it before the reduce). No buffer is shared between two calls or two streams.
at::Tensor linear_mxfp8_raw(at::Tensor x_fp8, at::Tensor w_fp8, at::Tensor sx_int32, at::Tensor sw_int32,
    std::optional<at::Tensor> bias, std::optional<at::Tensor> residual, int64_t act)
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
    // TMA reads the activation and its scale (input_alignment.h); K % 128 == 0 keeps every row aligned.
    detail::require_tma_aligned(x_fp8, "fish_scales_ops::linear_mxfp8_raw", "x_fp8");
    detail::require_tma_aligned(sx_int32, "fish_scales_ops::linear_mxfp8_raw", "sx_int32");
    // ... and the weight and its scale (K % 128 == 0 keeps every weight row aligned as well); with M == 0 nothing is
    // read.
    if (M > 0)
    {
        detail::require_tma_aligned(w_fp8, "fish_scales_ops::linear_mxfp8_raw", "w_fp8");
        detail::require_tma_aligned(sw_int32, "fish_scales_ops::linear_mxfp8_raw", "sw_int32");
    }
    fso::DenseEpilogue const epilogue
        = detail::make_dense_epilogue("fish_scales_ops::linear_mxfp8_raw", bias, residual, act, M, N, x_fp8.device());
    // sm_120/sm_121, the decode band: the small-M route (G1) where its rule R1 says it beats the cascade
    // (arch/sm120/mxfp8/smallm_route.h). It reads the same operands and applies the same epilogue.
    if (at::cuda::getCurrentDeviceProperties()->major == 12 && detail::sm120_smallm_takes(M, N, K, /*bf16x=*/false))
    {
        detail::require_dense_epilogue_op(epilogue, fso::kDenseEpiServedSm120Smallm,
            "fish_scales_ops::linear_mxfp8_raw (sm_120/sm_121, the small-M route sm120.smallm)");
        auto y = at::empty({M, N}, x_fp8.options().dtype(at::kBFloat16));
        cudaError_t const e = detail::launch_sm120_mxfp8_smallm(0, x_fp8.data_ptr(),
            reinterpret_cast<int32_t const*>(sx_int32.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3 const*>(w_fp8.data_ptr()),
            reinterpret_cast<int32_t const*>(sw_int32.data_ptr()), reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
            M, N, K, epilogue, at::cuda::getCurrentCUDAStream());
        TORCH_CHECK(e == cudaSuccess, "sm120 mxfp8 small-M kernel error: ", cudaGetErrorString(e));
        return y;
    }
    if (!epilogue.is_identity())
    {
        if (is_sm100_family())
            detail::require_dense_epilogue_op(epilogue, fso::kDenseEpiServedSm100Cascade,
                "fish_scales_ops::linear_mxfp8_raw (sm_100/sm_103, the C++ CUTLASS cascade gemm_dispatch_sm100_mxfp8)");
        else
            detail::require_dense_epilogue_op(epilogue, fso::kDenseEpiServedSm120Mxfp8Cascade,
                "fish_scales_ops::linear_mxfp8_raw (sm_120/sm_121, the CUTLASS MXFP8 cascade gemm_dispatch_sm120_mxfp8)");
    }
    bool const sm100 = is_sm100_family();
    std::size_t ws_bytes = 0;
    if (sm100)
    {
        TORCH_CHECK(detail::sm100_mxfp8_compiled(),
            "linear_mxfp8 on sm_100/sm_103 requires the extension to be built with CUDA >= 12.8 "
            "and TORCH_CUDA_ARCH_LIST including 10.0f (family target for B200 + B300).");
        // Host-only; refuses (NotImplementedError) before anything is allocated when the configuration the cascade
        // or FSO_FORCE_TILE picks cannot apply the epilogue.
        ws_bytes = detail::dense_epilogue_refusing(
            [&] { return detail::launch_sm100_mxfp8_dispatch_workspace_bytes(M, N, K, epilogue); });
    }

    auto y = at::empty({M, N}, x_fp8.options().dtype(at::kBFloat16));
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err;
    if (sm100)
    {
        at::Tensor workspace;
        if (ws_bytes > 0)
            workspace = at::empty({static_cast<int64_t>(ws_bytes)}, x_fp8.options().dtype(at::kByte));
        // Scales must be in the Sm1xx atom layout (quantize_1x32_packed
        // emits it automatically on sm_100-family devices).
        err = detail::dense_epilogue_refusing([&] {
            return detail::launch_sm100_mxfp8_dispatch(
                reinterpret_cast<__nv_fp8_e4m3*>(x_fp8.data_ptr()),
                reinterpret_cast<__nv_fp8_e4m3*>(w_fp8.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
                reinterpret_cast<int32_t*>(sx_int32.data_ptr()),
                reinterpret_cast<int32_t*>(sw_int32.data_ptr()),
                M, N, K, stream, epilogue, ws_bytes > 0 ? workspace.data_ptr() : nullptr, ws_bytes);
        });
        TORCH_CHECK(err == cudaSuccess, "sm100 mxfp8 kernel error: ", cudaGetErrorString(err),
            detail::sm100_mxfp8_error_hint(err));
        return y;
    }
    std::size_t scratch_bytes = 0;
    try
    {
        scratch_bytes = detail::sm120_mxfp8_dense_prepare(M, N, K, epilogue);
    }
    catch (fso::DenseEpilogueNotImplemented const& ex)
    {
        TORCH_CHECK_NOT_IMPLEMENTED(false, "fish_scales_ops::linear_mxfp8_raw: ", ex.what());
    }
    at::Tensor scratch;
    if (scratch_bytes > 0)
        scratch = at::empty({static_cast<int64_t>(scratch_bytes)}, x_fp8.options().dtype(at::kByte));
    err = detail::launch_sm120_mxfp8_dispatch(
        reinterpret_cast<__nv_fp8_e4m3*>(x_fp8.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(w_fp8.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        reinterpret_cast<int32_t*>(sx_int32.data_ptr()),
        reinterpret_cast<int32_t*>(sw_int32.data_ptr()),
        M, N, K, stream, epilogue, scratch_bytes > 0 ? scratch.data_ptr() : nullptr, scratch_bytes);
    TORCH_CHECK(err == cudaSuccess, "sm120 mxfp8 kernel error: ", cudaGetErrorString(err));
    return y;
}

// linear_mxfp8_quantized_raw: the GEMM of linear_mxfp8_raw with its result written in activation contract v1 of this
// device instead of bf16 (fso.dense.linear_quantized, interface v1, issue #12; charter 7.1): (xq [M, N]
// float8_e4m3fn row-major, x_scale int32 in this device's contract layout, the padded rows' words zero), bit for bit
// quantize_1x32_packed(linear_mxfp8_raw(x, w, sx, sw, bias, act)). `bias` / `act` are those of linear_mxfp8_raw;
// there is no residual. N % 128 == 0 (the result is the next GEMM's activation, K % 128 == 0).
// sm_100/sm_103 (S3b): the C++ CUTLASS cascade writes it in the same launch(es) as the bf16 call (the quantized-output
// instantiation of the configuration the cascade picks, or the batched split-K GEMM and its quantized reduce): the
// epilogue rounds to bf16 and quantizes the rounded values with fso/activation_contract.h, so no quantize kernel and
// no bf16 intermediate exist. The scale is the 1-D atom-layout tensor of align(M, 128) * N/128 words that
// quantize_1x32_packed returns on this architecture, every word written by the kernel. The Python op refuses the call
// first when the router's plan names a tier that does not serve it (decode row, cuBLAS, DSL); here the cascade's own
// refusals (a forced row without an instantiation, a scale layout it cannot write) are NotImplementedError before
// anything is launched. Unlike linear_mxfp8_raw, the two output buffers are allocated before that check: the cascade
// selects the quantized form from their pointers (epilogue.out_q), so they must exist when it is asked; the split-K
// workspace is still sized by the query and allocated only when the call is accepted.
// sm_120/sm_121 (S3c): x_scale int32 [align(M, 4), N/128] with strides (1, align(M, 4)); every route of the cascade
// writes it in the same launch(es) as the bf16 call (the quantized-output instance of the tile, or the Stream-K slices
// and the quantized reduce), the scale's padded words included, so no fill or quantize kernel is added.
// A refused call (a forced diagnostic instance on sm_120, a field outside the route's mask) launches nothing.
std::tuple<at::Tensor, at::Tensor> linear_mxfp8_quantized_raw(at::Tensor x_fp8, at::Tensor w_fp8, at::Tensor sx_int32,
    at::Tensor sw_int32, std::optional<at::Tensor> bias, int64_t act)
{
    char const* const op = "fish_scales_ops::linear_mxfp8_quantized_raw";
    TORCH_CHECK(x_fp8.is_cuda() && w_fp8.is_cuda(), op, ": x/w must be on CUDA");
    TORCH_CHECK(x_fp8.dtype() == at::kFloat8_e4m3fn && w_fp8.dtype() == at::kFloat8_e4m3fn,
        op, ": x_fp8 / w_fp8 must be float8_e4m3fn");
    TORCH_CHECK(sx_int32.dtype() == at::kInt && sw_int32.dtype() == at::kInt, op, ": sx / sw must be int32 (packed UE8M0)");
    TORCH_CHECK(x_fp8.dim() == 2 && w_fp8.dim() == 2, op, ": x_fp8 / w_fp8 must be 2D");
    int const M = x_fp8.size(0);
    int const K = x_fp8.size(1);
    int const N = w_fp8.size(0);
    TORCH_CHECK(w_fp8.size(1) == K, op, ": w.size(1) must match x.size(1)");
    TORCH_CHECK(K % 128 == 0, op, ": K must be a multiple of 128");
    TORCH_CHECK_VALUE(N % 128 == 0, op, ": N = ", N, " must be a multiple of 128 (the quantized result is the next "
        "GEMM's activation, whose K is a multiple of 128)");
    // TMA reads the activation and its scale (input_alignment.h); checked before the outputs are allocated.
    detail::require_tma_aligned(x_fp8, op, "x_fp8");
    detail::require_tma_aligned(sx_int32, op, "sx_int32");
    if (M > 0) // with M == 0 nothing is read
    {
        detail::require_tma_aligned(w_fp8, op, "w_fp8");
        detail::require_tma_aligned(sw_int32, op, "sw_int32");
    }
    fso::DenseEpilogue epilogue = detail::make_dense_epilogue(op, bias, std::nullopt, act, M, N, x_fp8.device());
    if (is_sm100_family())
    {
        TORCH_CHECK(detail::sm100_mxfp8_compiled(),
            "linear_mxfp8 on sm_100/sm_103 requires the extension to be built with CUDA >= 12.8 "
            "and TORCH_CUDA_ARCH_LIST including 10.0f (family target for B200 + B300).");
        // The scale as quantize_1x32_packed returns it on sm_100/sm_103: a 1-D int32 buffer of align(M, 128) *
        // N/128 words in the Sm1xx atom layout (fso_mx_sf_words_sm100); the kernels write every word, the padded
        // rows' as zero.
        int64_t const M_pad = (static_cast<int64_t>(M) + 127) / 128 * 128;
        auto xq = at::empty({M, N}, x_fp8.options().dtype(at::kFloat8_e4m3fn));
        auto sq = at::empty({M_pad * (N / 128)}, x_fp8.options().dtype(at::kInt));
        if (M == 0)
            return {xq, sq};
        epilogue.out_q = reinterpret_cast<__nv_fp8_e4m3*>(xq.data_ptr());
        epilogue.out_scale = sq.data_ptr();
        epilogue.out_scale_layout = fso::kDenseOutScaleMxSm100;
        detail::require_dense_epilogue_op(epilogue, fso::kDenseEpiServedSm100Cascade,
            "fish_scales_ops::linear_mxfp8_quantized_raw (sm_100/sm_103, the C++ CUTLASS cascade "
            "gemm_dispatch_sm100_mxfp8)");
        // Host-only; refuses (NotImplementedError) before anything is launched when the configuration the cascade
        // or FSO_FORCE_TILE picks cannot write the quantized output (charter 7.5: the split-K's FP32 partials come
        // from the torch caching allocator on the current stream, graph-private inside a capture, allocated only past
        // this point).
        std::size_t const ws_bytes = detail::dense_epilogue_refusing(
            [&] { return detail::launch_sm100_mxfp8_dispatch_workspace_bytes(M, N, K, epilogue); });
        at::Tensor workspace;
        if (ws_bytes > 0)
            workspace = at::empty({static_cast<int64_t>(ws_bytes)}, x_fp8.options().dtype(at::kByte));
        auto stream = at::cuda::getCurrentCUDAStream();
        // D is nullptr: with epilogue.out_q set the cascade writes the codes and the scale words and no bf16 tile.
        cudaError_t const err = detail::dense_epilogue_refusing([&] {
            return detail::launch_sm100_mxfp8_dispatch(
                reinterpret_cast<__nv_fp8_e4m3*>(x_fp8.data_ptr()),
                reinterpret_cast<__nv_fp8_e4m3*>(w_fp8.data_ptr()),
                nullptr,
                reinterpret_cast<int32_t*>(sx_int32.data_ptr()),
                reinterpret_cast<int32_t*>(sw_int32.data_ptr()),
                M, N, K, stream, epilogue, ws_bytes > 0 ? workspace.data_ptr() : nullptr, ws_bytes);
        });
        TORCH_CHECK(err == cudaSuccess, "sm100 mxfp8 kernel error: ", cudaGetErrorString(err),
            detail::sm100_mxfp8_error_hint(err));
        return {xq, sq};
    }
    TORCH_CHECK_NOT_IMPLEMENTED(at::cuda::getCurrentDeviceProperties()->major == 12, op, ": MXFP8 needs sm_120/sm_121 "
        "(or sm_100/sm_103); this device is sm_", at::cuda::getCurrentDeviceProperties()->major,
        at::cuda::getCurrentDeviceProperties()->minor);
    int const M_pad = (M + 3) / 4 * 4;
    int const kb = N / 128;
    auto xq = at::empty({M, N}, x_fp8.options().dtype(at::kFloat8_e4m3fn));
    // K-major, as quantize_1x32_packed returns it on sm_120 (stride 1 along M, M_pad along the scale columns).
    auto sq = at::empty_strided({M_pad, kb}, {1, M_pad}, x_fp8.options().dtype(at::kInt));
    if (M == 0)
        return {xq, sq};
    epilogue.out_q = reinterpret_cast<__nv_fp8_e4m3*>(xq.data_ptr());
    epilogue.out_scale = sq.data_ptr();
    epilogue.out_scale_layout = fso::kDenseOutScaleMxSm120;
    detail::require_dense_epilogue_op(epilogue, fso::kDenseEpiServedSm120Mxfp8Cascade,
        "fish_scales_ops::linear_mxfp8_quantized_raw (sm_120/sm_121, the CUTLASS MXFP8 cascade gemm_dispatch_sm120_mxfp8)");
    std::size_t scratch_bytes = 0;
    try
    {
        scratch_bytes = detail::sm120_mxfp8_dense_prepare(M, N, K, epilogue);
    }
    catch (fso::DenseEpilogueNotImplemented const& ex)
    {
        TORCH_CHECK_NOT_IMPLEMENTED(false, op, ": ", ex.what());
    }
    at::Tensor scratch;
    if (scratch_bytes > 0)
        scratch = at::empty({static_cast<int64_t>(scratch_bytes)}, x_fp8.options().dtype(at::kByte));
    auto stream = at::cuda::getCurrentCUDAStream();
    // D is out_q: the launches write xq and its scale words, never a bf16 tile (gemm_dispatch_sm120_mxfp8).
    cudaError_t const err = detail::launch_sm120_mxfp8_dispatch(
        reinterpret_cast<__nv_fp8_e4m3*>(x_fp8.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(w_fp8.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(xq.data_ptr()),
        reinterpret_cast<int32_t*>(sx_int32.data_ptr()),
        reinterpret_cast<int32_t*>(sw_int32.data_ptr()),
        M, N, K, stream, epilogue, scratch_bytes > 0 ? scratch.data_ptr() : nullptr, scratch_bytes);
    TORCH_CHECK(err == cudaSuccess, "sm120 mxfp8 kernel error: ", cudaGetErrorString(err));
    return {xq, sq};
}

// mxfp8_dense_route_sm120(M, N, K): the route the sm_120/sm_121 dense MXFP8 cascade takes for one [M, K] x [N, K]
// call in this process (the FSO_FORCE_* knobs included), as [tile_m, tile_n, stages, min_blocks, sched_group,
// k_split, smallm, streamk_scratch_bytes]; k_split > 1 is Stream-K. Host only, launches nothing; for tests and
// benches. Refuses other architectures.
std::vector<int64_t> mxfp8_dense_route_sm120(int64_t M, int64_t N, int64_t K)
{
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 12,
        "mxfp8_dense_route_sm120: the current device is not sm_120/sm_121");
    TORCH_CHECK(M >= 0 && N > 0 && K > 0, "mxfp8_dense_route_sm120: M >= 0, N > 0, K > 0, got ", M, ", ", N, ", ", K);
    std::vector<int64_t> out(8, 0);
    detail::sm120_mxfp8_dense_route_query(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), out.data());
    return out;
}

// linear_mxfp8_bf16x: the bf16-activation dense MXFP8 GEMM of sm_120/sm_121 as one op, so that the dense layer makes
// one call for every M. x: bf16 [M, K], contiguous, K % 128 == 0; w_fp8 / sw_int32: the prepared sm_120 weight and its
// int32 K-major scale words. In the decode band, where the small-M route's rule R2 says so and x is 16-byte aligned,
// one kernel (G2) quantizes x by activation contract v1 inside itself; everywhere else the op runs quantize_1x32_packed
// (the contract's quantizer, quantize_activation's bytes) and then linear_mxfp8_raw, the 2b call (G1 where R1 says so,
// else the cascade). R2 implies R1, and G2 equals quantize + G1 bit for bit, so the result is always the bytes of
// linear_mxfp8_raw(quantize(x), ...). bias / residual / act: the dense epilogue of interface v1, as linear_mxfp8_raw.
at::Tensor linear_mxfp8_bf16x(at::Tensor x, at::Tensor w_fp8, at::Tensor sw_int32, std::optional<at::Tensor> bias,
    std::optional<at::Tensor> residual, int64_t act)
{
    TORCH_CHECK(x.is_cuda() && w_fp8.is_cuda(), "linear_mxfp8_bf16x: x / w must be on CUDA");
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 12,
        "linear_mxfp8_bf16x: the current device is not sm_120/sm_121 (it is the bf16-x entry of the sm_120 dense "
        "MXFP8 GEMM; other architectures quantize with quantize_1x32_packed and call linear_mxfp8_raw)");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "linear_mxfp8_bf16x: x must be bf16, got ", x.scalar_type());
    TORCH_CHECK(w_fp8.dtype() == at::kFloat8_e4m3fn, "linear_mxfp8_bf16x: w_fp8 must be float8_e4m3fn");
    TORCH_CHECK(sw_int32.dtype() == at::kInt, "linear_mxfp8_bf16x: sw must be int32 (packed UE8M0)");
    TORCH_CHECK(x.dim() == 2 && w_fp8.dim() == 2, "linear_mxfp8_bf16x: x / w_fp8 must be 2D");
    TORCH_CHECK(x.is_contiguous(), "linear_mxfp8_bf16x: x must be contiguous");
    int const M = x.size(0);
    int const K = x.size(1);
    int const N = w_fp8.size(0);
    TORCH_CHECK(w_fp8.size(1) == K, "linear_mxfp8_bf16x: w.size(1) must match x.size(1)");
    TORCH_CHECK(K % 128 == 0, "linear_mxfp8_bf16x: K must be a multiple of 128");
    TORCH_CHECK(N % 128 == 0, "linear_mxfp8_bf16x: N must be a multiple of 128 (BlockScaled atom requires it)");
    // The weight and its scale are read in place on both forms of the op (the small-M kernel G2, or linear_mxfp8_raw),
    // so the flat 16-byte rule of linear_mxfp8_raw applies here too, before either form is chosen: a misaligned one
    // is refused with ValueError and nothing is launched (input_alignment.h). With M == 0 nothing is read.
    if (M > 0)
    {
        detail::require_tma_aligned(w_fp8, "fish_scales_ops::linear_mxfp8_bf16x", "w_fp8");
        detail::require_tma_aligned(sw_int32, "fish_scales_ops::linear_mxfp8_bf16x", "sw_int32");
    }
    fso::DenseEpilogue const epilogue
        = detail::make_dense_epilogue("fish_scales_ops::linear_mxfp8_bf16x", bias, residual, act, M, N, x.device());
    if (M == 0)
        return at::empty({0, N}, x.options().dtype(at::kBFloat16));
    // G2 reads x in 16-byte vectors from its data pointer (the row pitch, 2 K bytes with K % 128 == 0, keeps every row
    // aligned with it). An x whose pointer is not 16-byte aligned takes quantize_1x32_packed + linear_mxfp8_raw below,
    // which give the same bytes: the quantizer then picks its narrower kernel or refuses what that kernel cannot read
    // (input_alignment.h), so this op never launches a kernel on an x it cannot read.
    bool const x_vec16 = reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0;
    if (x_vec16 && detail::sm120_smallm_takes(M, N, K, /*bf16x=*/true))
    {
        detail::require_dense_epilogue_op(epilogue, fso::kDenseEpiServedSm120Smallm,
            "fish_scales_ops::linear_mxfp8_bf16x (sm_120/sm_121, the small-M route sm120.smallm)");
        auto y = at::empty({M, N}, x.options().dtype(at::kBFloat16));
        cudaError_t const e = detail::launch_sm120_mxfp8_smallm(1, x.data_ptr(), nullptr,
            reinterpret_cast<__nv_fp8_e4m3 const*>(w_fp8.data_ptr()),
            reinterpret_cast<int32_t const*>(sw_int32.data_ptr()), reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
            M, N, K, epilogue, at::cuda::getCurrentCUDAStream());
        TORCH_CHECK(e == cudaSuccess, "sm120 mxfp8 small-M kernel error: ", cudaGetErrorString(e));
        return y;
    }
    auto [x_q, sx] = quantize_1x32_packed(x, true);
    return linear_mxfp8_raw(x_q, w_fp8, sx, sw_int32, bias, residual, act);
}

// mxfp8_dense_smallm_route_sm120(M, N, K, bf16x): the small-M route of one sm_120/sm_121 dense call in this process
// (FSO_SMALLM_ROUTE and the force knobs included), as [use, nc, rt, wk, cj, layout_smem, launch_smem, grid, r1, r2]:
// use = 1 when the call (the bf16-x call when bf16x, else the 2b call) runs the small-M kernel; nc / rt / wk / cj
// its configuration (cj = K blocks per warp per chunk, 0 = the activation resident); r1 / r2 are the rules'
// verdicts on their own. Host only (reads the device's SM count), launches nothing.
std::vector<int64_t> mxfp8_dense_smallm_route_sm120(int64_t M, int64_t N, int64_t K, bool bf16x)
{
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 12,
        "mxfp8_dense_smallm_route_sm120: the current device is not sm_120/sm_121");
    TORCH_CHECK(M >= 0 && N > 0 && K > 0, "mxfp8_dense_smallm_route_sm120: M >= 0, N > 0, K > 0, got ", M, ", ", N,
        ", ", K);
    std::vector<int64_t> out(10, 0);
    detail::sm120_smallm_route_query(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), bf16x, out.data());
    return out;
}

// sm120_smallm_rules(M, N, K, sms): the rules and the configuration of the small-M route alone, on a device of `sms`
// multiprocessors (the rules' one-round plan guard depends on it), as [takes_2b, takes_bf16x, r1, r2, nc, rt, wk, cj,
// layout_smem]; takes_* include FSO_SMALLM_ROUTE and the force knobs; every field is 0 for a shape the kernel does not
// serve (M outside 1..64, K not a multiple of 128, N not a multiple of 32). Pure host arithmetic: it needs no device
// (the host unit test of the rules runs it on any machine with a fixed SM count; fso.dense passes the device's).
std::vector<int64_t> sm120_smallm_rules(int64_t M, int64_t N, int64_t K, int64_t sms)
{
    namespace sr = fso::sm120_smallm;
    TORCH_CHECK(M >= 0 && N >= 0 && K >= 0 && M <= (1 << 30) && N <= (1 << 30) && K <= (1 << 30),
        "sm120_smallm_rules: M, N, K must be in [0, 2^30], got ", M, ", ", N, ", ", K);
    TORCH_CHECK(sms >= 1 && sms <= (1 << 16), "sm120_smallm_rules: sms (the device's SM count) must be in [1, 65536], got ",
        sms);
    int const m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K), d = static_cast<int>(sms);
    if (!sr::shape_served(m, n, k))
        return std::vector<int64_t>(9, 0);
    sr::Plan const c = sr::config(m, n, k);
    return {sr::plan(m, n, k, false, d).use ? 1 : 0, sr::plan(m, n, k, true, d).use ? 1 : 0,
        sr::rule_r1(m, n, k, d) ? 1 : 0, sr::rule_r2(m, n, k, d) ? 1 : 0, c.nc, c.rt, c.wk, c.cj, c.smem};
}


// The four operands every grouped MXFP8 GEMM reads with TMA: the fp8 activation, its int32 scale words, the fp8
// weights and their int32 scale words (input_alignment.h). Without the check a misaligned one aborts the process at
// CuTe's host assertion (the sm_100/sm_103 slot route, sm_120/sm_121) or faults on the device (the sm_100/sm_103
// pointer-array route, whose descriptors are built on the device). Without an activation row nothing is read.
static void require_grouped_tma_operands(at::Tensor const& a, at::Tensor const& sa, at::Tensor const& b,
    at::Tensor const& sb, char const* op, char const* a_name, char const* sa_name, char const* b_name,
    char const* sb_name)
{
    if (a.numel() == 0)
        return;
    detail::require_tma_aligned(a, op, a_name);
    detail::require_tma_aligned(sa, op, sa_name);
    detail::require_tma_aligned(b, op, b_name);
    detail::require_tma_aligned(sb, op, sb_name);
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
// keeps that route switched off. sm_120/121 read it for the decode tile rule
// (the solo route's tile count, see gemm_dispatch_sm120_mxfp8_grouped); 0 keeps
// that cascade's m_cap-only gate.
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
//
// problem_shapes is the fourth optional argument and the pointer-array route's
// counterpart of the slot list (run b300_round3_20260922/M-A4): int32 [G, 3]
// on device, the (rows, N, K) triple of every group for THIS GEMM's (N, K),
// as `moe_build_routing(..., problem_shapes_for=[(N, K), ...])` writes it in
// the pass that publishes `masked_m`. With it the sm_100/103 cascade launches
// nothing but the GEMM: its routing-dependent argument was the only thing the
// per-launch preparation kernel still had to compute, and the tensor-set
// arrays it also rebuilt are now bound once per tensor set. Without it the
// preparation kernel runs as before. sm_120/121 validates and ignores it, and
// so does the sm_100 slot route.

at::Tensor linear_mxfp8_grouped_masked(at::Tensor a_fp8, at::Tensor w_fp8, at::Tensor sa_int32,
    at::Tensor sw_int32, at::Tensor masked_m, int64_t expected_m, int64_t max_active_groups,
    std::optional<at::Tensor> slot_to_expert, std::optional<at::Tensor> problem_shapes)
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
    require_grouped_tma_operands(a_fp8, sa_int32, w_fp8, sw_int32, "fish_scales_ops::linear_mxfp8_grouped_masked",
        "a_fp8", "sa_int32", "w_fp8", "sw_int32");
    int const* const slot_ptr = check_slot_to_expert(slot_to_expert, G, masked_m);
    int32_t const* const ps_ptr = check_problem_shapes(problem_shapes, G, m_cap, N, K, masked_m);

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
            m_cap, N, K, G, static_cast<int>(max_active_groups), /*compact=*/0);
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
            G, m_cap, N, K, static_cast<int>(expected_m), static_cast<int>(max_active_groups), slot_ptr, ps_ptr,
            stream);
        TORCH_CHECK(err == cudaSuccess, "sm100 mxfp8 grouped kernel error: ", cudaGetErrorString(err));
        return y;
    }
    // sm_120/121 has one grouped route, so neither the slot list nor the
    // problem shapes carry information it can use; both are validated above
    // and then ignored. max_active_groups bounds the decode route's tile count
    // and is passed on to the cascade.
    err = detail::launch_sm120_mxfp8_grouped_dispatch(
        reinterpret_cast<__nv_fp8_e4m3*>(a_fp8.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(w_fp8.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        reinterpret_cast<int32_t*>(sa_int32.data_ptr()),
        reinterpret_cast<int32_t*>(sw_int32.data_ptr()),
        reinterpret_cast<int32_t*>(masked_m.data_ptr()),
        G, m_cap, N, K, static_cast<int>(expected_m), static_cast<int>(max_active_groups), stream);
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
    std::optional<at::Tensor> slot_to_expert, std::optional<at::Tensor> problem_shapes)
{
    // Shape checks common to both architectures first, then the arch's own
    // scale-slab layout and launch.
    auto const* prop = at::cuda::getCurrentDeviceProperties();
    bool const sm120 = prop->major == 12;
    TORCH_CHECK(is_sm100_family() || sm120,
        "linear_mxfp8_grouped_masked_swiglu needs sm_100/sm_103 (B200 / B300) or sm_120/121; sm_90 has no MXFP8 "
        "hardware at all. Use linear_mxfp8_grouped_masked + silu_chunk_mul_quantize_1x32_grouped there.");
    if (sm120)
    {
        TORCH_CHECK(a_fp8.is_cuda() && w13_fp8.is_cuda(), "a/w13 must be on CUDA");
        TORCH_CHECK(a_fp8.dtype() == at::kFloat8_e4m3fn && w13_fp8.dtype() == at::kFloat8_e4m3fn,
            "a_fp8 / w13_fp8 must be float8_e4m3fn");
        TORCH_CHECK(sa_int32.dtype() == at::kInt && sw13_int32.dtype() == at::kInt,
            "sa / sw13 must be int32 (packed UE8M0)");
        TORCH_CHECK(masked_m.dtype() == at::kInt && masked_m.is_cuda(), "masked_m must be CUDA int32");
        TORCH_CHECK(a_fp8.dim() == 3 && w13_fp8.dim() == 3, "a_fp8 [G,m_cap,K] / w13_fp8 [G,2*I,K] must be 3D");
        TORCH_CHECK(a_fp8.is_contiguous() && w13_fp8.is_contiguous() && masked_m.is_contiguous(),
            "a/w13/masked_m must be contiguous");
        TORCH_CHECK(!slot_to_expert.has_value() || slot_to_expert->numel() == 0,
            "the sm_120/121 grouped path has no slot-bound route, so it reads no active-expert list");
        TORCH_CHECK(!problem_shapes.has_value() || problem_shapes->numel() == 0,
            "the sm_120/121 grouped path has no pointer-array route, so it reads no per-group problem shapes");
        int const G = a_fp8.size(0);
        int const m_cap = a_fp8.size(1);
        int const K = a_fp8.size(2);
        int const N = w13_fp8.size(1);
        int const INTER = N / 2;
        TORCH_CHECK(w13_fp8.size(0) == G && w13_fp8.size(2) == K, "w13_fp8 must be [G, 2*I, K] matching a_fp8");
        TORCH_CHECK(masked_m.numel() == G, "masked_m must have G entries");
        TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
        TORCH_CHECK(N % 2 == 0 && INTER % 128 == 0,
            "w13_fp8.size(1) must be 2*I with I a multiple of 128 (a 1x32 output scale block must not straddle "
            "the packed int32 word), got N=", N);
        TORCH_CHECK(m_cap % 4 == 0, "m_cap must be a multiple of 4 (per-group scale padding)");
        TORCH_CHECK(expected_m >= 1, "expected_m must be >= 1");
        TORCH_CHECK(G <= 1024, "the grouped MXFP8 path supports at most 1024 experts, got ", G);
        TORCH_CHECK(max_active_groups >= 0 && max_active_groups <= G,
            "max_active_groups must be in [0, G]; got ", max_active_groups, " with G=", G);
        int64_t const kp = K / 128;
        TORCH_CHECK(sa_int32.numel() == static_cast<int64_t>(G) * m_cap * kp,
            "sa_int32 must be [G, K/128, m_cap] (per-group K-major packed scales)");
        TORCH_CHECK(sw13_int32.numel() == static_cast<int64_t>(G) * kp * N,
            "sw13_int32 must be [G, K/128, 2*I] (per-group K-major packed scales)");
        TORCH_CHECK(sa_int32.is_contiguous() && sw13_int32.is_contiguous(), "sa/sw13 must be contiguous");
        require_grouped_tma_operands(a_fp8, sa_int32, w13_fp8, sw13_int32,
            "fish_scales_ops::linear_mxfp8_grouped_masked_swiglu", "a_fp8", "sa_int32", "w13_fp8", "sw13_int32");
        auto h = at::empty({G, m_cap, INTER}, a_fp8.options().dtype(at::kFloat8_e4m3fn));
        auto sh = at::empty({G, INTER / 128, m_cap}, a_fp8.options().dtype(at::kInt));
        auto stream = at::cuda::getCurrentCUDAStream();
        cudaError_t const err = detail::launch_sm120_mxfp8_grouped_swiglu_dispatch(
            reinterpret_cast<__nv_fp8_e4m3*>(a_fp8.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(w13_fp8.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(h.data_ptr()),
            reinterpret_cast<int32_t*>(sh.data_ptr()),
            reinterpret_cast<int32_t*>(sa_int32.data_ptr()),
            reinterpret_cast<int32_t*>(sw13_int32.data_ptr()),
            reinterpret_cast<int32_t*>(masked_m.data_ptr()),
            G, m_cap, N, K, static_cast<int>(expected_m), static_cast<int>(max_active_groups), stream);
        TORCH_CHECK(err == cudaSuccess, "sm120 fused-SwiGLU grouped mxfp8 kernel error: ", cudaGetErrorString(err));
        return {h, sh};
    }
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
    require_grouped_tma_operands(a_fp8, sa_int32, w13_fp8, sw13_int32,
        "fish_scales_ops::linear_mxfp8_grouped_masked_swiglu", "a_fp8", "sa_int32", "w13_fp8", "sw13_int32");
    int const* const slot_ptr = check_slot_to_expert(slot_to_expert, G, masked_m);
    // The fused FC1 is a GEMM of N = 2*I weight rows, so its triple carries
    // that N, not the I columns it finally stores.
    int32_t const* const ps_ptr = check_problem_shapes(problem_shapes, G, m_cap, N, K, masked_m);

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
        G, m_cap, N, K, static_cast<int>(expected_m), static_cast<int>(max_active_groups), slot_ptr, ps_ptr,
        stream);
    TORCH_CHECK(err == cudaSuccess, "sm100 fused-SwiGLU grouped mxfp8 kernel error: ", cudaGetErrorString(err));
    return {h, sh};
}



// ---------------------------------------------------------------------------
// The compact activation layout (fso 0.2.1), sm_100/sm_103 half of
// linear_mxfp8_grouped_compact and linear_mxfp8_grouped_compact_swiglu. The op
// entries live in ops/mxfp8_compact.cu and call these on sm_100/sm_103.
// ---------------------------------------------------------------------------
//
// Layout (charter D1, the one-group view). The activations are ONE group of
// R = 128 * B_alloc rows: a fp8 [R, K], sa the arch's one-group scale tensor of
// R rows (sm_100/103: the Sm1xx atom slab, R * K/128 int32 words, one 128-row
// atom per 128 rows), d bf16 [R, N] / dq fp8 [R, I] with sd the one-group atom
// slab of R rows. Expert g owns the 128-row blocks starting at blk0[g] (int32
// [G + 1], the exclusive prefix of ceil(masked_m / 128)); its routed rows are
// rows blk0[g] * 128 + r, r < masked_m[g]. The weights b [G, N, K] and their
// scales keep the batch index g. Every output row of an expert at or past
// masked_m[g] is left untouched, on both routes (the slot route's stores are
// bounded by each expert's routed row count, unlike the masked slot route's,
// which writes its slab's first m_cap rows).
//
// `m_cap` is the masked op's row capacity align(M, 4): a host-static HINT the
// route rule and the tile cascade key on, so the compact op runs the kernel the
// masked op runs for the same (m_cap, N, K, G, expected_m, max_active_groups).
// It is not a tensor dimension. The slot route additionally relies on
// masked_m[g] <= m_cap, which no-replacement top-k routing guarantees (one row
// per token per expert), exactly as the masked layout does.
//
// The three optional tensors are moe_build_routing_compact's: the packed
// active-expert list (slot route), and for the pointer-array route this GEMM's
// problem shapes int32 [G, 3] = (masked_m[g], N, K) together with its per-call
// pointer arrays int64 [3, G] (row 0 = a + blk0[g] * 128 * K, row 1 = sa +
// blk0[g] * 4K bytes, row 2 = d + blk0[g] * 128 * row bytes of d). Give both or
// neither; without them the op launches its own preparation kernel. The weight
// side of the pointer-array arguments lives in a block keyed by the weights and
// (G, N, K), bound on first sight and never per capture.
//
// FSO_CHECK_PROBLEM_SHAPES=1 also checks the content of problem_shapes and
// ptr_arrays against masked_m, blk0 and the operand addresses (a device-to-host
// copy; a debugging aid, never on in a timed path).

namespace
{

struct CompactDims
{
    int G, R, K, N, b_alloc;
};

CompactDims check_compact_operands(char const* op, at::Tensor const& a, at::Tensor const& b, at::Tensor const& sa,
    at::Tensor const& sb, at::Tensor const& masked_m, at::Tensor const& blk0, int64_t m_cap, int64_t expected_m,
    int64_t max_active_groups, char const* b_name = "b", char const* sb_name = "sb")
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), op, ": a/b must be on CUDA");
    TORCH_CHECK(a.dtype() == at::kFloat8_e4m3fn && b.dtype() == at::kFloat8_e4m3fn,
        op, ": a / b must be float8_e4m3fn");
    TORCH_CHECK(sa.dtype() == at::kInt && sb.dtype() == at::kInt, op, ": sa / sb must be int32 (packed UE8M0)");
    TORCH_CHECK(masked_m.dtype() == at::kInt && masked_m.is_cuda(), op, ": masked_m must be CUDA int32");
    TORCH_CHECK(blk0.dtype() == at::kInt && blk0.is_cuda(), op, ": blk0 must be CUDA int32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 3, op, ": a must be [R, K] (one group of R rows) and b [G, N, K]");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && sa.is_contiguous() && sb.is_contiguous()
            && masked_m.is_contiguous() && blk0.is_contiguous(),
        op, ": a, b, sa, sb, masked_m and blk0 must be contiguous");
    CompactDims d{};
    d.R = static_cast<int>(a.size(0));
    d.K = static_cast<int>(a.size(1));
    d.G = static_cast<int>(b.size(0));
    d.N = static_cast<int>(b.size(1));
    TORCH_CHECK(b.size(2) == d.K, op, ": b.size(2) must equal a.size(1) (K)");
    TORCH_CHECK(d.R >= 128 && d.R % 128 == 0, op, ": a.size(0) must be a positive multiple of 128 (R = 128 * B_alloc), "
        "got ", d.R);
    d.b_alloc = d.R / 128;
    TORCH_CHECK(masked_m.numel() == d.G, op, ": masked_m must have G = ", d.G, " entries");
    TORCH_CHECK(blk0.numel() == d.G + 1, op, ": blk0 must have G + 1 = ", d.G + 1, " entries");
    TORCH_CHECK(d.K % 128 == 0, op, ": K must be a multiple of 128");
    TORCH_CHECK(d.N % 128 == 0, op, ": N must be a multiple of 128 (BlockScaled atom requires it)");
    TORCH_CHECK(m_cap >= 4 && m_cap % 4 == 0, op, ": m_cap (the masked op's align(M, 4)) must be a positive multiple "
        "of 4, got ", m_cap);
    TORCH_CHECK(expected_m >= 1, op, ": expected_m must be >= 1");
    TORCH_CHECK(max_active_groups >= 0 && max_active_groups <= d.G,
        op, ": max_active_groups must be in [0, G]; got ", max_active_groups, " with G=", d.G);
    TORCH_CHECK(d.G >= 1 && d.G <= 1024, op, ": the grouped MXFP8 path supports 1 to 1024 experts, got ", d.G);
    int64_t const kp = d.K / 128;
    TORCH_CHECK(sa.numel() == static_cast<int64_t>(d.R) * kp,
        op, ": sa must be the one-group Sm1xx atom slab of R rows, R * K/128 = ", static_cast<int64_t>(d.R) * kp,
        " int32 words, got ", sa.numel());
    TORCH_CHECK(sb.numel() == static_cast<int64_t>(d.G) * d.N * kp,
        op, ": sb must be [G, N * K/128] (per-group Sm1xx atom slab)");
    require_grouped_tma_operands(a, sa, b, sb, op, "a", "sa", b_name, sb_name);
    TORCH_CHECK(detail::sm100_mxfp8_grouped_compact_fits(d.R, d.K) != 0,
        op, ": R = ", d.R, " rows at K = ", d.K, " exceeds the compact scale-factor layout's constant row extent (",
        detail::sm100_mxfp8_grouped_compact_sfa_rows(), " rows, K < 65536)");
    return d;
}

// The route refusals the masked op makes, made on the compact op's m_cap hint.
void check_compact_slot_refusal(char const* op, CompactDims const& d, int m_cap, int max_active_groups)
{
    int const r = detail::sm100_mxfp8_grouped_slot_refusal(m_cap, d.N, d.K, d.G, max_active_groups, /*compact=*/1);
    TORCH_CHECK(r != 2, op, ": FSO_GROUPED_SLOT=force: the sm_100 slot-bound route needs the whole row capacity to fit "
        "one token tile, but m_cap=", m_cap, " exceeds TileN=", detail::sm100_mxfp8_grouped_slot_tile_n(),
        ". Outside that bound the kernel silently scales its output by 2^-127. Unset FSO_GROUPED_SLOT or call with "
        "m_cap <= ", detail::sm100_mxfp8_grouped_slot_tile_n(), ".");
    TORCH_CHECK(r != 3, op, ": FSO_GROUPED_SLOT=force: no sm_100 slot-bound instantiation covers N=", d.N, " K=", d.K,
        ".");
    TORCH_CHECK(r != 4, op, ": FSO_GROUPED_SLOT=force: the sm_100 slot-bound route needs max_active_groups > 0 "
        "(= min(M * topk, G)) to size its slot list and grid; the op was called with max_active_groups=0.");
}

// problem_shapes [G, 3] int32 and ptr_arrays [3, G] int64: both or neither.
// Returns the two device pointers (or two nullptrs).
std::pair<int32_t const*, int64_t const*> check_compact_arrays(char const* op, CompactDims const& d,
    std::optional<at::Tensor> const& problem_shapes, std::optional<at::Tensor> const& ptr_arrays,
    at::Tensor const& masked_m, at::Tensor const& blk0, void const* a_base, void const* sa_base, void const* d_base,
    int64_t d_row_bytes)
{
    bool const has_ps = problem_shapes.has_value() && problem_shapes->defined();
    bool const has_pa = ptr_arrays.has_value() && ptr_arrays->defined();
    TORCH_CHECK(has_ps == has_pa, op, ": problem_shapes and ptr_arrays come together from "
        "moe_build_routing_compact; pass both or neither");
    if (!has_ps)
        return {nullptr, nullptr};
    at::Tensor const& ps = *problem_shapes;
    at::Tensor const& pa = *ptr_arrays;
    TORCH_CHECK(ps.is_cuda() && ps.dtype() == at::kInt && ps.is_contiguous() && ps.dim() == 2 && ps.size(0) == d.G
            && ps.size(1) == 3,
        op, ": problem_shapes must be a contiguous CUDA int32 [G, 3] tensor (G=", d.G, "), got ", ps.sizes());
    TORCH_CHECK(pa.is_cuda() && pa.dtype() == at::kLong && pa.is_contiguous() && pa.dim() == 2 && pa.size(0) == 3
            && pa.size(1) == d.G,
        op, ": ptr_arrays must be a contiguous CUDA int64 [3, G] tensor (G=", d.G, "), got ", pa.sizes());
    TORCH_CHECK(ps.device() == masked_m.device() && pa.device() == masked_m.device(),
        op, ": problem_shapes / ptr_arrays must be on masked_m's device");
    char const* check = std::getenv("FSO_CHECK_PROBLEM_SHAPES");
    if (check && check[0] == '1')
    {
        at::Tensor const hps = ps.to(at::kCPU);
        at::Tensor const hpa = pa.to(at::kCPU);
        at::Tensor const hm = masked_m.to(at::kCPU);
        at::Tensor const hb = blk0.to(at::kCPU);
        auto const aps = hps.accessor<int32_t, 2>();
        auto const apa = hpa.accessor<int64_t, 2>();
        auto const am = hm.accessor<int32_t, 1>();
        auto const ab = hb.accessor<int32_t, 1>();
        int64_t const sfa_blk = static_cast<int64_t>(d.K / 128) * 512;
        for (int g = 0; g < d.G; ++g)
        {
            TORCH_CHECK(aps[g][0] == am[g] && aps[g][1] == d.N && aps[g][2] == d.K, "FSO_CHECK_PROBLEM_SHAPES: ",
                op, ": problem_shapes[", g, "] = (", aps[g][0], ", ", aps[g][1], ", ", aps[g][2],
                ") does not match (masked_m[g]=", am[g], ", N=", d.N, ", K=", d.K, ")");
            int64_t const b0 = ab[g];
            TORCH_CHECK(b0 >= 0 && (b0 + (am[g] + 127) / 128) <= d.b_alloc, "FSO_CHECK_PROBLEM_SHAPES: ", op,
                ": expert ", g, " (blk0=", b0, ", rows=", am[g], ") does not fit B_alloc=", d.b_alloc);
            TORCH_CHECK(apa[0][g] == reinterpret_cast<int64_t>(a_base) + b0 * 128 * d.K
                    && apa[1][g] == reinterpret_cast<int64_t>(sa_base) + b0 * sfa_blk
                    && apa[2][g] == reinterpret_cast<int64_t>(d_base) + b0 * 128 * d_row_bytes,
                "FSO_CHECK_PROBLEM_SHAPES: ", op, ": ptr_arrays[:, ", g, "] do not point at expert ", g,
                "'s first block of this call's a / sa / output");
        }
    }
    return {reinterpret_cast<int32_t const*>(ps.data_ptr()), reinterpret_cast<int64_t const*>(pa.data_ptr())};
}

// The slot route runs the token side on the slot index, which equals blk0[e] for
// the expert e in that slot when every active expert holds at most m_cap <= 64 rows
// (one block each) and the slot list is the packed ascending list of active
// experts. Under FSO_CHECK_PROBLEM_SHAPES=1 the op checks that on the host.
void check_compact_slot_identity(char const* op, CompactDims const& d, at::Tensor const& masked_m,
    at::Tensor const& blk0, std::optional<at::Tensor> const& slot_to_expert, int m_cap)
{
    char const* check = std::getenv("FSO_CHECK_PROBLEM_SHAPES");
    if (!(check && check[0] == '1'))
        return;
    at::Tensor const hm = masked_m.to(at::kCPU);
    at::Tensor const hb = blk0.to(at::kCPU);
    auto const am = hm.accessor<int32_t, 1>();
    auto const ab = hb.accessor<int32_t, 1>();
    int rank = 0;
    for (int g = 0; g < d.G; ++g)
    {
        TORCH_CHECK(am[g] <= m_cap, "FSO_CHECK_PROBLEM_SHAPES: ", op, ": the slot route needs masked_m[g] <= m_cap, "
            "but expert ", g, " holds ", am[g], " rows with m_cap=", m_cap);
        if (am[g] > 0)
        {
            TORCH_CHECK(ab[g] == rank, "FSO_CHECK_PROBLEM_SHAPES: ", op, ": blk0[", g, "] = ", ab[g],
                " but expert ", g, " is active expert number ", rank, " in ascending order");
            ++rank;
        }
    }
    if (slot_to_expert.has_value() && slot_to_expert->defined())
    {
        at::Tensor const hs = slot_to_expert->to(at::kCPU);
        auto const as = hs.accessor<int32_t, 1>();
        int s = 0;
        for (int g = 0; g < d.G; ++g)
            if (am[g] > 0)
            {
                TORCH_CHECK(as[s] == g, "FSO_CHECK_PROBLEM_SHAPES: ", op, ": slot_to_expert[", s, "] = ", as[s],
                    " is not the packed ascending list of active experts (expected ", g, ")");
                ++s;
            }
    }
}

// The slot route's work-stealing counter (int32 [2]: tile draws, finished CTAs).
// The kernel needs it zero when it starts and leaves it zero, so one counter
// serves any number of slot-route GEMMs issued one after another on a stream;
// moe_build_routing_compact(..., with_slots=True) returns a zeroed one per call
// (`slot_sched`). A call that lands on the slot route without one gets a zeroed
// counter from the caching allocator (one fill kernel on the stream); a call
// that does not land there ignores it. Returns the device pointer or nullptr
// when the route will not run.
int* compact_slot_sched(char const* op, std::optional<at::Tensor> const& slot_sched, at::Tensor const& masked_m,
    bool slot_taken, at::Tensor& owned)
{
    if (slot_sched.has_value() && slot_sched->defined())
    {
        at::Tensor const& t = *slot_sched;
        TORCH_CHECK(t.is_cuda() && t.dtype() == at::kInt && t.is_contiguous() && t.numel() >= 2,
            op, ": slot_sched must be a contiguous CUDA int32 tensor of at least 2 entries (moe_build_routing_compact's "
            "`slot_sched` output), got ", t.sizes(), " ", t.dtype());
        TORCH_CHECK(t.device() == masked_m.device(), op, ": slot_sched must be on masked_m's device");
        return slot_taken ? reinterpret_cast<int*>(t.data_ptr()) : nullptr;
    }
    if (!slot_taken)
        return nullptr;
    owned = at::zeros({2}, masked_m.options());
    return reinterpret_cast<int*>(owned.data_ptr());
}

// The compact slot route's packed list of live 64-row token tiles (run
// fso_0.2.1_20261010/S10b-3b), int32 [T + 1]: entries [0, count) the live tiles,
// entries [count, T) -1, entry T the count (moe_build_routing_compact(...,
// with_tiles=True)'s `slot_tiles`; packing in grouped_slot_dispatch.cuh,
// `pack_token_tile`). The slot route reads it past a row capacity of one token tile
// (m_cap > 64), where an expert spans several token tiles and blocks; at m_cap <= 64
// it keeps its slot mode and ignores the list. A call that needs the list and was
// handed none gets one from the caching allocator, of 2 * B_alloc entries (every
// 128-row block holds at most two token tiles), which the launch fills from
// masked_m and blk0 first (one small kernel). Returns the device pointer, or nullptr
// when the route reads no list; `cap` gets T and `build` whether the launch must
// fill it.
int* compact_slot_tiles(char const* op, std::optional<at::Tensor> const& slot_tiles, at::Tensor const& masked_m,
    at::Tensor const& blk0, CompactDims const& d, bool list_mode, at::Tensor& owned, int& cap, bool& build)
{
    cap = 0;
    build = false;
    bool const given = slot_tiles.has_value() && slot_tiles->defined();
    if (given)
    {
        at::Tensor const& t = *slot_tiles;
        TORCH_CHECK(t.is_cuda() && t.dtype() == at::kInt && t.is_contiguous() && t.dim() == 1 && t.numel() >= 2,
            op, ": slot_tiles must be a contiguous 1-D CUDA int32 tensor of T + 1 >= 2 entries "
            "(moe_build_routing_compact's `slot_tiles` output), got ", t.sizes(), " ", t.dtype());
        TORCH_CHECK(t.device() == masked_m.device(), op, ": slot_tiles must be on masked_m's device");
        TORCH_CHECK(t.numel() - 1 <= int64_t{2} * d.b_alloc, op, ": slot_tiles holds ", t.numel() - 1,
            " entries, more than the 2 * B_alloc = ", int64_t{2} * d.b_alloc, " token tiles the compact buffer has");
    }
    if (!list_mode)
        return nullptr;
    TORCH_CHECK(d.b_alloc <= 8192, op, ": the slot route past a row capacity of one token tile packs a 128-row "
        "block index into 13 bits, so it needs R <= 8192 * 128 rows; got B_alloc = ", d.b_alloc,
        ". Unset FSO_GROUPED_SLOT.");
    if (given)
    {
        cap = static_cast<int>(slot_tiles->numel() - 1);
        char const* check = std::getenv("FSO_CHECK_PROBLEM_SHAPES");
        if (check && check[0] == '1')
        {
            // The list against masked_m and blk0: every expert's token tiles, in order, then -1, then the count.
            at::Tensor const ht = slot_tiles->to(at::kCPU);
            at::Tensor const hm = masked_m.to(at::kCPU);
            at::Tensor const hb = blk0.to(at::kCPU);
            auto const at_ = ht.accessor<int32_t, 1>();
            auto const am = hm.accessor<int32_t, 1>();
            auto const ab = hb.accessor<int32_t, 1>();
            int t = 0;
            for (int g = 0; g < d.G; ++g)
                for (int k = 0; k * 64 < am[g]; ++k, ++t)
                {
                    int const r = std::min(64, am[g] - 64 * k);
                    int const want = (2 * ab[g] + k) | (g << 14) | ((r - 1) << 24);
                    TORCH_CHECK(t < cap && at_[t] == want, "FSO_CHECK_PROBLEM_SHAPES: ", op, ": slot_tiles[", t,
                        "] does not hold token tile ", k, " of expert ", g, " (blk0=", ab[g], ", rows=", am[g], ")");
                }
            TORCH_CHECK(at_[cap] == t, "FSO_CHECK_PROBLEM_SHAPES: ", op, ": slot_tiles[T] = ", at_[cap],
                " but the routing has ", t, " live token tiles");
            for (int u = t; u < cap; ++u)
                TORCH_CHECK(at_[u] == -1, "FSO_CHECK_PROBLEM_SHAPES: ", op, ": slot_tiles[", u,
                    "] past the live tiles is not -1");
        }
        return reinterpret_cast<int*>(slot_tiles->data_ptr());
    }
    cap = 2 * d.b_alloc;
    owned = at::empty({cap + 1}, masked_m.options());
    build = true;
    return reinterpret_cast<int*>(owned.data_ptr());
}

// The op entries (ops/mxfp8_compact.cu) send sm_100/sm_103 here and serve sm_120/121 themselves, so this only
// guards the build.
void compact_arch_gate(char const* op)
{
    auto const* prop = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(prop->major == 10, op, ": the sm_100/sm_103 half was called on sm_", prop->major, "x");
    TORCH_CHECK(detail::sm100_mxfp8_grouped_compiled(),
        op, " on sm_100/sm_103 requires the extension to be built with CUDA >= 12.8 and TORCH_CUDA_ARCH_LIST "
        "including 10.0f (family target for B200 + B300).");
}

} // anonymous namespace

void linear_mxfp8_grouped_compact_sm100(at::Tensor a, at::Tensor b, at::Tensor sa, at::Tensor sb,
    at::Tensor masked_m, at::Tensor blk0, int64_t m_cap, int64_t expected_m, int64_t max_active_groups,
    std::optional<at::Tensor> slot_to_expert, std::optional<at::Tensor> problem_shapes,
    std::optional<at::Tensor> ptr_arrays, at::Tensor d, std::optional<at::Tensor> slot_sched,
    std::optional<at::Tensor> slot_tiles)
{
    char const* const op = "linear_mxfp8_grouped_compact";
    compact_arch_gate(op);
    CompactDims const dims
        = check_compact_operands(op, a, b, sa, sb, masked_m, blk0, m_cap, expected_m, max_active_groups);
    TORCH_CHECK(d.is_cuda() && d.dtype() == at::kBFloat16 && d.is_contiguous() && d.dim() == 2 && d.size(0) == dims.R
            && d.size(1) == dims.N,
        op, ": d must be a contiguous CUDA bf16 [R, N] = [", dims.R, ", ", dims.N, "] tensor, got ", d.sizes());
    check_compact_slot_refusal(op, dims, static_cast<int>(m_cap), static_cast<int>(max_active_groups));
    int const* const slot_ptr = check_slot_to_expert(slot_to_expert, dims.G, masked_m);
    bool const slot_taken = detail::sm100_mxfp8_grouped_slot_taken(static_cast<int>(m_cap), dims.N, dims.K, dims.G,
        static_cast<int>(max_active_groups), /*fused_swiglu=*/0, /*compact=*/1, static_cast<int>(expected_m));
    // The pointer-array route writes d through the CUTLASS epilogue's TMA descriptors (16-byte vectors); the compact
    // slot route stores it one bf16 at a time (plain_slot_store.cuh), so only the former needs an aligned d
    // (input_alignment.h). Checked before anything is allocated or launched.
    if (!slot_taken)
        detail::require_tma_store_aligned(d, op, "d");
    bool const list_mode = slot_taken && m_cap > detail::sm100_mxfp8_grouped_slot_tile_n();
    if (slot_taken && !list_mode)
        check_compact_slot_identity(op, dims, masked_m, blk0, slot_to_expert, static_cast<int>(m_cap));
    at::Tensor sched_owned;
    int* const sched_ptr = compact_slot_sched(op, slot_sched, masked_m, slot_taken, sched_owned);
    at::Tensor tiles_owned;
    int tiles_cap = 0;
    bool build_tiles = false;
    int* const tiles_ptr
        = compact_slot_tiles(op, slot_tiles, masked_m, blk0, dims, list_mode, tiles_owned, tiles_cap, build_tiles);
    auto const arrays = check_compact_arrays(op, dims, problem_shapes, ptr_arrays, masked_m, blk0, a.data_ptr(),
        sa.data_ptr(), d.data_ptr(), static_cast<int64_t>(dims.N) * 2);
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaError_t const err = detail::launch_sm100_mxfp8_grouped_compact_dispatch(
        reinterpret_cast<__nv_fp8_e4m3*>(a.data_ptr()), reinterpret_cast<__nv_fp8_e4m3*>(b.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), reinterpret_cast<int32_t*>(sa.data_ptr()),
        reinterpret_cast<int32_t*>(sb.data_ptr()), reinterpret_cast<int32_t*>(masked_m.data_ptr()),
        reinterpret_cast<int32_t const*>(blk0.data_ptr()), dims.G, dims.b_alloc, static_cast<int>(m_cap), dims.N,
        dims.K, static_cast<int>(expected_m), static_cast<int>(max_active_groups), slot_ptr, arrays.first,
        arrays.second, sched_ptr, tiles_ptr, tiles_cap, build_tiles, stream);
    TORCH_CHECK(err == cudaSuccess, "sm100 compact mxfp8 grouped kernel error: ", cudaGetErrorString(err));
}

void linear_mxfp8_grouped_compact_swiglu_sm100(at::Tensor a, at::Tensor b, at::Tensor sa, at::Tensor sb,
    at::Tensor masked_m, at::Tensor blk0, int64_t m_cap, int64_t expected_m, int64_t max_active_groups,
    std::optional<at::Tensor> slot_to_expert, std::optional<at::Tensor> problem_shapes,
    std::optional<at::Tensor> ptr_arrays, at::Tensor dq, at::Tensor sd, std::optional<at::Tensor> slot_sched,
    std::optional<at::Tensor> slot_tiles)
{
    char const* const op = "linear_mxfp8_grouped_compact_swiglu";
    compact_arch_gate(op);
    CompactDims const dims
        = check_compact_operands(op, a, b, sa, sb, masked_m, blk0, m_cap, expected_m, max_active_groups);
    int const inter = dims.N / 2;
    TORCH_CHECK(dims.N % 2 == 0 && inter % 128 == 0,
        op, ": b.size(1) must be 2*I with I a multiple of 128 (a 1x32 output scale block must not straddle the atom "
        "slab's 128-column block), got N=", dims.N);
    TORCH_CHECK(dq.is_cuda() && dq.dtype() == at::kFloat8_e4m3fn && dq.is_contiguous() && dq.dim() == 2
            && dq.size(0) == dims.R && dq.size(1) == inter,
        op, ": dq must be a contiguous CUDA float8_e4m3fn [R, I] = [", dims.R, ", ", inter, "] tensor, got ",
        dq.sizes());
    TORCH_CHECK(sd.is_cuda() && sd.dtype() == at::kInt && sd.is_contiguous()
            && sd.numel() == static_cast<int64_t>(dims.R) * (inter / 128),
        op, ": sd must be the contiguous one-group Sm1xx atom slab of R rows, R * I/128 = ",
        static_cast<int64_t>(dims.R) * (inter / 128), " int32 words");
    check_compact_slot_refusal(op, dims, static_cast<int>(m_cap), static_cast<int>(max_active_groups));
    int const* const slot_ptr = check_slot_to_expert(slot_to_expert, dims.G, masked_m);
    bool const slot_taken = detail::sm100_mxfp8_grouped_slot_taken(static_cast<int>(m_cap), dims.N, dims.K, dims.G,
        static_cast<int>(max_active_groups), /*fused_swiglu=*/1, /*compact=*/1, static_cast<int>(expected_m));
    // The pointer-array route's fused store writes dq in 16-byte vectors (fused_swiglu_store.cuh); the compact slot
    // route's writes it one code at a time (fused_swiglu_slot_store.cuh). Both write sd one byte at a time.
    if (!slot_taken)
        detail::require_out_aligned(dq, dims.R, inter, 16, op, "dq");
    bool const list_mode = slot_taken && m_cap > detail::sm100_mxfp8_grouped_slot_tile_n();
    if (slot_taken && !list_mode)
        check_compact_slot_identity(op, dims, masked_m, blk0, slot_to_expert, static_cast<int>(m_cap));
    at::Tensor sched_owned;
    int* const sched_ptr = compact_slot_sched(op, slot_sched, masked_m, slot_taken, sched_owned);
    at::Tensor tiles_owned;
    int tiles_cap = 0;
    bool build_tiles = false;
    int* const tiles_ptr
        = compact_slot_tiles(op, slot_tiles, masked_m, blk0, dims, list_mode, tiles_owned, tiles_cap, build_tiles);
    // The fused FC1 is a GEMM of N = 2*I weight rows, so its triples carry that
    // N; its D rows are the fp8 output's (I bytes each).
    auto const arrays = check_compact_arrays(op, dims, problem_shapes, ptr_arrays, masked_m, blk0, a.data_ptr(),
        sa.data_ptr(), dq.data_ptr(), static_cast<int64_t>(inter));
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaError_t const err = detail::launch_sm100_mxfp8_grouped_swiglu_compact_dispatch(
        reinterpret_cast<__nv_fp8_e4m3*>(a.data_ptr()), reinterpret_cast<__nv_fp8_e4m3*>(b.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(dq.data_ptr()), reinterpret_cast<int32_t*>(sd.data_ptr()),
        reinterpret_cast<int32_t*>(sa.data_ptr()), reinterpret_cast<int32_t*>(sb.data_ptr()),
        reinterpret_cast<int32_t*>(masked_m.data_ptr()), reinterpret_cast<int32_t const*>(blk0.data_ptr()), dims.G,
        dims.b_alloc, static_cast<int>(m_cap), dims.N, dims.K, static_cast<int>(expected_m),
        static_cast<int>(max_active_groups), slot_ptr, arrays.first, arrays.second, sched_ptr, tiles_ptr, tiles_cap,
        build_tiles, stream);
    TORCH_CHECK(err == cudaSuccess, "sm100 compact fused-SwiGLU mxfp8 grouped kernel error: ",
        cudaGetErrorString(err));
}

// linear_mxfp8_grouped_compact_fc1_fc2 (internal; sm_100/sm_103): the MoE layer's FC1 and FC2 on the compact slot
// route as ONE persistent kernel (issue #13 lever 5, runs fso_issue13_20261031/I13-L5 and I13-L5b;
// blockscale_gemm/arch/sm100/mxfp8/grouped_slot_moe_dispatch.cuh). It writes exactly what
// linear_mxfp8_grouped_compact_swiglu (dq, sd) followed by linear_mxfp8_grouped_compact (dn) write for the same
// arguments, bit for bit, at every live row; rows no expert covers are not written.
//
// Only `_moe_layer_mxfp8_compact` calls it, on the plan's fused route (the compact layout, the fused FC1, both GEMMs
// on the compact slot route), in the slot route's two modes: slot mode at m_cap <= 64, token-tile mode past it (the
// routing op's live token-tile list, `slot_tiles`, required there). Its contract is the layer's, and the op checks
// what it can on the host:
//   * `slot_to_expert` is the routing op's packed active-expert list (required: the kernel has no list builder);
//   * `slot_sched` is the routing op's counter tensor of the SAME layer call: int32 [2 + 2G] in slot mode, words
//     0 .. 1 + G zero (the draw counters and the per-entry FC1 counters, which the kernel leaves nonzero, so a counter
//     tensor serves one call), words 2 + G .. the packed per-slot facts expert | rows << 16 the routing op writes; in
//     token-tile mode B more words after those, one per 128-row block, live_halves << 20 (the kernel adds its FC1
//     completions into them; at least B_alloc - 1 of them, the routing op's num_blocks with the layer's one block of
//     slack);
//   * `slot_tiles` (token-tile mode) is the routing op's list of the same call, int32 [T + 1];
//   * `dn` must not overlap `a` or `sa` (in one kernel an FC2 tile can store while an FC1 tile of another entry still
//     reads its input), which the op checks on the pointers.
// Every tile is handed out from the counter (slot_moe_scheduler.cuh): that needs no assumption about which CTAs of
// the grid are resident, which a static FC1 first wave does and which two of these kernels running at the same time
// on two streams can break (a deadlock, measured: run fso_issue13_20261031/I13-L5, tests/g1_residency).
namespace
{
bool byte_ranges_overlap(void const* p, int64_t pn, void const* q, int64_t qn)
{
    auto const a = reinterpret_cast<uintptr_t>(p);
    auto const b = reinterpret_cast<uintptr_t>(q);
    return a < b + static_cast<uintptr_t>(qn) && b < a + static_cast<uintptr_t>(pn);
}
} // anonymous namespace

void linear_mxfp8_grouped_compact_fc1_fc2(at::Tensor a, at::Tensor w13, at::Tensor sa, at::Tensor sw13, at::Tensor w2,
    at::Tensor sw2, at::Tensor masked_m, at::Tensor blk0, int64_t m_cap, int64_t expected_m, int64_t max_active_groups,
    at::Tensor slot_to_expert, at::Tensor slot_sched, at::Tensor dq, at::Tensor sd, at::Tensor dn,
    std::optional<at::Tensor> slot_tiles)
{
    char const* const op = "linear_mxfp8_grouped_compact_fc1_fc2";
    compact_arch_gate(op);
    TORCH_CHECK(detail::sm100_mxfp8_slot_moe_compiled(), op, ": this build carries no fused FC1 + FC2 slot kernel");
    CompactDims const d1 = check_compact_operands(op, a, w13, sa, sw13, masked_m, blk0, m_cap, expected_m,
        max_active_groups, "w13", "sw13");
    int const inter = d1.N / 2;
    TORCH_CHECK(d1.N % 2 == 0 && inter % 128 == 0, op, ": w13.size(1) must be 2*I with I a multiple of 128, got N=",
        d1.N);
    TORCH_CHECK(w2.is_cuda() && w2.dtype() == at::kFloat8_e4m3fn && w2.is_contiguous() && w2.dim() == 3
            && w2.size(0) == d1.G && w2.size(2) == inter,
        op, ": w2 must be the contiguous fp8 per-expert weights [G, N2, I] = [", d1.G, ", N2, ", inter, "], got ",
        w2.sizes());
    int const n2 = static_cast<int>(w2.size(1));
    TORCH_CHECK(n2 > 0 && n2 % 128 == 0, op, ": w2.size(1) must be a positive multiple of 128, got ", n2);
    TORCH_CHECK(sw2.is_cuda() && sw2.dtype() == at::kInt && sw2.is_contiguous()
            && sw2.numel() == static_cast<int64_t>(d1.G) * n2 * (inter / 128),
        op, ": sw2 must be [G, N2 * I/128] int32 (per-group Sm1xx atom slab)");
    // The FC2 half reads w2 and its scale words with TMA as well (input_alignment.h).
    detail::require_tma_aligned(w2, op, "w2");
    detail::require_tma_aligned(sw2, op, "sw2");
    // The slot route's two modes (the same split as linear_mxfp8_grouped_compact's): slot mode while the row capacity
    // fits one token tile, the token-tile mode past it.
    bool const list_mode = m_cap > detail::sm100_mxfp8_grouped_slot_tile_n();
    TORCH_CHECK(max_active_groups >= 1, op, ": max_active_groups (= min(M * topk, G)) must be >= 1");
    TORCH_CHECK(detail::sm100_mxfp8_slot_moe_shape_ok(d1.N, d1.K, n2, inter), op,
        ": no fused FC1 + FC2 slot instantiation covers N1=", d1.N, " K1=", d1.K, " N2=", n2, " K2=", inter);
    TORCH_CHECK(dq.is_cuda() && dq.dtype() == at::kFloat8_e4m3fn && dq.is_contiguous() && dq.dim() == 2
            && dq.size(0) == d1.R && dq.size(1) == inter,
        op, ": dq must be a contiguous CUDA float8_e4m3fn [R, I] = [", d1.R, ", ", inter, "] tensor, got ", dq.sizes());
    TORCH_CHECK(sd.is_cuda() && sd.dtype() == at::kInt && sd.is_contiguous()
            && sd.numel() == static_cast<int64_t>(d1.R) * (inter / 128),
        op, ": sd must be the contiguous one-group Sm1xx atom slab of R rows, R * I/128 = ",
        static_cast<int64_t>(d1.R) * (inter / 128), " int32 words");
    TORCH_CHECK(dn.is_cuda() && dn.dtype() == at::kBFloat16 && dn.is_contiguous() && dn.dim() == 2
            && dn.size(0) == d1.R && dn.size(1) == n2,
        op, ": dn must be a contiguous CUDA bf16 [R, N2] = [", d1.R, ", ", n2, "] tensor, got ", dn.sizes());
    // The FC1 half writes dq and sd one element at a time, but the FC2 half reads them back with TMA
    // (input_alignment.h); dn is written one bf16 at a time (the plain slot store) and is not checked.
    detail::require_tma_aligned(dq, op, "dq");
    detail::require_tma_aligned(sd, op, "sd");
    int64_t const dn_bytes = dn.numel() * 2;
    TORCH_CHECK(!byte_ranges_overlap(dn.data_ptr(), dn_bytes, a.data_ptr(), a.numel())
            && !byte_ranges_overlap(dn.data_ptr(), dn_bytes, sa.data_ptr(), sa.numel() * 4)
            && !byte_ranges_overlap(dn.data_ptr(), dn_bytes, dq.data_ptr(), dq.numel())
            && !byte_ranges_overlap(dn.data_ptr(), dn_bytes, sd.data_ptr(), sd.numel() * 4),
        op, ": dn must not overlap a, sa, dq or sd: in one kernel an FC2 tile may store while an FC1 tile of another "
        "entry still reads its input");
    std::optional<at::Tensor> const slots_opt = slot_to_expert;
    int const* const slot_ptr = check_slot_to_expert(slots_opt, d1.G, masked_m);
    // Token-tile mode: the per-block words follow the 2 + 2G words, one per 128-row block the routing op can use
    // (its num_blocks, which the layer's buffer exceeds by one block of slack: B_alloc - 1).
    int64_t const sched_need = 2 + 2 * static_cast<int64_t>(d1.G) + (list_mode ? std::max(1, d1.b_alloc - 1) : 0);
    TORCH_CHECK(slot_sched.is_cuda() && slot_sched.dtype() == at::kInt && slot_sched.is_contiguous()
            && slot_sched.numel() >= sched_need && slot_sched.device() == masked_m.device(),
        op, ": slot_sched must be this layer call's counter tensor from the routing op, contiguous CUDA int32 of at "
        "least ", list_mode ? "2 + 2G + B_alloc - 1 = " : "2 + 2G = ", sched_need,
        " entries (moe_build_routing_compact(..., with_slots=True", list_mode ? ", with_tiles=True" : "",
        ")'s `slot_sched`), got ", slot_sched.sizes(), " ", slot_sched.dtype());
    if (list_mode)
    {
        TORCH_CHECK(slot_tiles.has_value() && slot_tiles->defined(), op, ": past a row capacity of one token tile "
            "(m_cap=", m_cap, ") the fused kernel walks the routing op's live token-tile list; pass `slot_tiles` "
            "(moe_build_routing_compact(..., with_tiles=True)'s output)");
        at::Tensor const& t = *slot_tiles;
        TORCH_CHECK(t.is_cuda() && t.dtype() == at::kInt && t.is_contiguous() && t.dim() == 1 && t.numel() >= 2
                && t.device() == masked_m.device(),
            op, ": slot_tiles must be a contiguous 1-D CUDA int32 tensor of T + 1 >= 2 entries on masked_m's device "
            "(moe_build_routing_compact's `slot_tiles` output), got ", t.sizes(), " ", t.dtype());
        TORCH_CHECK(t.numel() - 1 <= int64_t{2} * d1.b_alloc, op, ": slot_tiles holds ", t.numel() - 1,
            " entries, more than the 2 * B_alloc = ", int64_t{2} * d1.b_alloc, " token tiles the compact buffer has");
        TORCH_CHECK(d1.b_alloc <= 8192 && d1.G <= 1024, op, ": the token-tile list packs a 128-row block index into 13 "
            "bits and an expert id into 10, so it needs B_alloc <= 8192 and G <= 1024; got B_alloc = ", d1.b_alloc,
            ", G = ", d1.G);
    }
    else
        check_compact_slot_identity(op, d1, masked_m, blk0, slots_opt, static_cast<int>(m_cap));
    auto stream = at::cuda::getCurrentCUDAStream();
    if (list_mode)
    {
        cudaError_t const err = detail::launch_sm100_mxfp8_slot_moe_list(
            reinterpret_cast<__nv_fp8_e4m3 const*>(a.data_ptr()), reinterpret_cast<int32_t const*>(sa.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3 const*>(w13.data_ptr()), reinterpret_cast<int32_t const*>(sw13.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3 const*>(w2.data_ptr()), reinterpret_cast<int32_t const*>(sw2.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(dq.data_ptr()), reinterpret_cast<int32_t*>(sd.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(dn.data_ptr()), reinterpret_cast<int32_t const*>(masked_m.data_ptr()),
            reinterpret_cast<int32_t const*>(blk0.data_ptr()), slot_ptr, reinterpret_cast<int*>(slot_sched.data_ptr()),
            reinterpret_cast<int const*>(slot_tiles->data_ptr()), static_cast<int>(slot_tiles->numel() - 1), d1.G,
            static_cast<int>(m_cap), d1.b_alloc, d1.N, d1.K, n2, inter, stream);
        TORCH_CHECK(err == cudaSuccess, "sm100 fused FC1 + FC2 kernel (token-tile mode) error: ", cudaGetErrorString(err));
        return;
    }
    int const num_slots = static_cast<int>(max_active_groups < d1.G ? max_active_groups : d1.G);
    cudaError_t const err = detail::launch_sm100_mxfp8_slot_moe(
        reinterpret_cast<__nv_fp8_e4m3 const*>(a.data_ptr()), reinterpret_cast<int32_t const*>(sa.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3 const*>(w13.data_ptr()), reinterpret_cast<int32_t const*>(sw13.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3 const*>(w2.data_ptr()), reinterpret_cast<int32_t const*>(sw2.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(dq.data_ptr()), reinterpret_cast<int32_t*>(sd.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dn.data_ptr()), reinterpret_cast<int32_t const*>(masked_m.data_ptr()),
        reinterpret_cast<int32_t const*>(blk0.data_ptr()), slot_ptr, reinterpret_cast<int*>(slot_sched.data_ptr()),
        d1.G, num_slots, static_cast<int>(m_cap), d1.b_alloc, d1.N, d1.K, n2, inter, stream);
    TORCH_CHECK(err == cudaSuccess, "sm100 fused FC1 + FC2 slot kernel error: ", cudaGetErrorString(err));
}

// linear_mxfp8_grouped_masked_combine: the grouped FC2 (down projection) with the
// weighted combine in its epilogue (sm_120/121).
//
// Instead of storing a bf16 [G, m_cap, HIDDEN] slab for moe_combine to read back
// and sum, each row is scaled by its combine weight and added into its token's row
// of `out`. The slab and the combine launch both disappear; the cost is that the
// adds are atomic, so the accumulation order -- and therefore the last bits of
// each output element -- depends on how the CTAs interleave.
//
// `out` is ACCUMULATED into: the caller fills it first (zero, or a shared
// expert's gated output), which is also what makes a token whose every routed
// entry was skipped come out as whatever the caller put there. `row_map` and
// `weight_of_slot` come from moe_build_routing (the latter by passing topk_w).
//
// Size is the reason this is a separate op rather than the default: run
// sm120_fusion_20260927 measured the scatter at 101 us against 261 us for the
// store-plus-read pair at M = 4096, where the slab is 134 MB and exceeds the
// card's 96 MB L2, but at M <= 1024 both slabs are L2-resident and the pair is
// cheaper. The host decides; this op does what it is told.
at::Tensor linear_mxfp8_grouped_masked_combine(at::Tensor a_fp8, at::Tensor w2_fp8, at::Tensor sa_int32,
    at::Tensor sw2_int32, at::Tensor masked_m, at::Tensor row_map, at::Tensor weight_of_slot, at::Tensor out,
    int64_t expected_m, int64_t max_active_groups)
{
    auto const* prop = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(prop->major == 12,
        "linear_mxfp8_grouped_masked_combine is the sm_120/121 fused-combine FC2; sm_100/103 serves this band "
        "from its own routes and sm_90 has no MXFP8 hardware.");
    TORCH_CHECK(a_fp8.is_cuda() && w2_fp8.is_cuda() && out.is_cuda(), "a/w2/out must be on CUDA");
    TORCH_CHECK(a_fp8.dtype() == at::kFloat8_e4m3fn && w2_fp8.dtype() == at::kFloat8_e4m3fn,
        "a_fp8 / w2_fp8 must be float8_e4m3fn");
    TORCH_CHECK(sa_int32.dtype() == at::kInt && sw2_int32.dtype() == at::kInt, "sa / sw2 must be int32");
    TORCH_CHECK(masked_m.dtype() == at::kInt && masked_m.is_cuda(), "masked_m must be CUDA int32");
    TORCH_CHECK(row_map.dtype() == at::kInt && row_map.is_cuda() && row_map.is_contiguous(),
        "row_map must be contiguous CUDA int32");
    TORCH_CHECK(weight_of_slot.dtype() == at::kFloat && weight_of_slot.is_cuda() && weight_of_slot.is_contiguous(),
        "weight_of_slot must be contiguous CUDA fp32 (moe_build_routing with topk_w)");
    TORCH_CHECK(out.dtype() == at::kBFloat16 && out.dim() == 2 && out.is_contiguous(),
        "out must be a contiguous bf16 [M, HIDDEN] tensor, pre-filled with what the layer adds to");
    TORCH_CHECK(a_fp8.dim() == 3 && w2_fp8.dim() == 3 && a_fp8.is_contiguous() && w2_fp8.is_contiguous(),
        "a_fp8 [G,m_cap,INTER] / w2_fp8 [G,HIDDEN,INTER] must be contiguous 3D");
    int const G = a_fp8.size(0);
    int const m_cap = a_fp8.size(1);
    int const K = a_fp8.size(2);
    int const N = w2_fp8.size(1);
    TORCH_CHECK(w2_fp8.size(0) == G && w2_fp8.size(2) == K, "w2_fp8 must be [G, HIDDEN, INTER] matching a_fp8");
    TORCH_CHECK(masked_m.numel() == G, "masked_m must have G entries");
    TORCH_CHECK(out.size(1) == N, "out must have HIDDEN columns matching w2_fp8.size(1)");
    TORCH_CHECK(K % 128 == 0 && N % 128 == 0, "K and N must be multiples of 128");
    TORCH_CHECK(m_cap % 4 == 0, "m_cap must be a multiple of 4");
    TORCH_CHECK(expected_m >= 1, "expected_m must be >= 1");
    TORCH_CHECK(row_map.numel() >= static_cast<int64_t>(G) * m_cap && weight_of_slot.numel() >= static_cast<int64_t>(G) * m_cap,
        "row_map and weight_of_slot must have G * m_cap entries");
    int64_t const kp = K / 128;
    TORCH_CHECK(sa_int32.numel() == static_cast<int64_t>(G) * m_cap * kp,
        "sa_int32 must be [G, K/128, m_cap] (per-group K-major packed scales)");
    TORCH_CHECK(sw2_int32.numel() == static_cast<int64_t>(G) * kp * N,
        "sw2_int32 must be [G, K/128, HIDDEN]");
    require_grouped_tma_operands(a_fp8, sa_int32, w2_fp8, sw2_int32,
        "fish_scales_ops::linear_mxfp8_grouped_masked_combine", "a_fp8", "sa_int32", "w2_fp8", "sw2_int32");
    // The epilogue adds into out with 8-byte reductions, and the GEMM's TMA store descriptor is built on out.
    detail::require_tma_store_aligned(out, "fish_scales_ops::linear_mxfp8_grouped_masked_combine", "out");
    TORCH_CHECK(max_active_groups >= 0 && max_active_groups <= G,
        "max_active_groups must be in [0, G]; got ", max_active_groups, " with G=", G);
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaError_t const err = detail::launch_sm120_mxfp8_grouped_combine_dispatch(
        reinterpret_cast<__nv_fp8_e4m3*>(a_fp8.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(w2_fp8.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<int32_t*>(sa_int32.data_ptr()),
        reinterpret_cast<int32_t*>(sw2_int32.data_ptr()),
        reinterpret_cast<int32_t*>(masked_m.data_ptr()),
        reinterpret_cast<int32_t const*>(row_map.data_ptr()),
        reinterpret_cast<float const*>(weight_of_slot.data_ptr()),
        G, m_cap, N, K, static_cast<int>(expected_m), static_cast<int>(max_active_groups), stream);
    TORCH_CHECK(err == cudaSuccess, "sm120 fused-combine grouped mxfp8 kernel error: ", cudaGetErrorString(err));
    return out;
}


// mxfp8_grouped_swiglu_fused_route: the host-side router a caller uses to
// decide, ONCE per shape, whether to build its FC1 out of the fused op or out
// of the old pair.
//
// The decision cannot be left to the GEMM, because the two forms need
// different weights (interleaved vs [gate; up]) and different follow-on
// kernels, so it has to be made before the layer is composed. On sm_100/103
// both grouped routes carry a fused FC1 (the pointer-array cascade and the
// swap-orientation slot kernel), so with FSO_FC1_FUSED unset the answer is
// true wherever an instantiation covers the shape, and the slot-route verdict
// only decides which fused kernel runs. sm_120/121 has one grouped route, so
// the per-call answer is the load-time one (mxfp8_grouped_swiglu_available).
// Every other device answers false: it has no fused FC1. `compact` names the
// layout of the fused op the call will be (the compact and the masked ops have
// different slot-route rules on sm_100/103); other architectures ignore it.
bool mxfp8_grouped_swiglu_available(int64_t n_w, int64_t k); // defined below

bool mxfp8_grouped_swiglu_fused_route(int64_t m_cap, int64_t n_w, int64_t k, int64_t num_groups,
    int64_t max_active_groups, bool compact, int64_t expected_m)
{
    auto const* prop = at::cuda::getCurrentDeviceProperties();
    if (prop->major == 12)
    {
        // One grouped route on this architecture, so every call that may use the
        // fused FC1 does: the per-call answer is the load-time one.
        (void) m_cap;
        (void) num_groups;
        (void) max_active_groups;
        (void) compact;
        (void) expected_m;
        return mxfp8_grouped_swiglu_available(n_w, k);
    }
    if (!is_sm100_family() || !detail::sm100_mxfp8_grouped_compiled())
        return false;
    return detail::sm100_mxfp8_grouped_fused_fc1_route(static_cast<int>(m_cap), static_cast<int>(n_w),
               static_cast<int>(k), static_cast<int>(num_groups), static_cast<int>(max_active_groups),
               compact ? 1 : 0, static_cast<int>(expected_m))
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
    auto const* prop = at::cuda::getCurrentDeviceProperties();
    if (prop->major == 12)
    {
        // sm_120/121: the fused FC1 is the grouped kernel's own epilogue, so it
        // is available for every shape the grouped path serves whose INTER is a
        // multiple of 128 (a 1x32 output scale block must not straddle the packed
        // int32 word) and whose K is a multiple of 128. The knob is the same one.
        if (!fso_fc1_fused_enabled())
            return false;
        return n_w % 2 == 0 && (n_w / 2) % 128 == 0 && k % 128 == 0;
    }
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
// therefore steers this answer exactly as it steers the route. `fused_swiglu`
// says which kernel the call will land on: the fused-SwiGLU FC1
// (`linear_mxfp8_grouped_masked_swiglu`) takes the slot route one step short
// of the plain grouped GEMM — one 32-column epilogue chunk instead of the full
// 64-wide token tile (run b300_round3_20260922/M-A3) — so a layer asks for its
// FC1 with the flag set and for its FC2 with it clear. `compact` says which
// layout's op the call will be: the compact ops (`linear_mxfp8_grouped_compact`,
// `_compact_swiglu`) are routed by their own rule (`slot_route_compact` in the
// dispatcher, run fso_0.2.1_20261010/S10b-3a), the masked ops by 0.2.0's, which
// is also the answer with the flag clear. `expected_m` is the expected rows per
// expert, ceil(M * topk / G), which the compact rule bounds since run
// fso_0.2.1_20261010/S16b; 0 (unknown, the default) keeps the compact rule's
// previous bound, m_cap <= 512, so a caller that does not pass it keeps its
// answers. The compact ops always know it (it is their `expected_m` argument), so
// a caller that composes them should pass it to get the route the op takes.
bool mxfp8_grouped_slot_possible(int64_t m_cap, int64_t n_w, int64_t k, int64_t num_groups,
    int64_t max_active_groups, bool fused_swiglu, bool compact, int64_t expected_m)
{
    if (!is_sm100_family() || !detail::sm100_mxfp8_grouped_compiled())
        return false;
    return detail::sm100_mxfp8_grouped_slot_taken(static_cast<int>(m_cap), static_cast<int>(n_w),
               static_cast<int>(k), static_cast<int>(num_groups), static_cast<int>(max_active_groups),
               fused_swiglu ? 1 : 0, compact ? 1 : 0, static_cast<int>(expected_m))
        != 0;
}


// mxfp8_grouped_problem_shapes_consumed: would a grouped GEMM of this shape
// read a caller-supplied `problem_shapes` tensor? (run b300_round3_20260922/M-A4)
//
// The complement of `mxfp8_grouped_slot_possible` on sm_100/103: the
// pointer-array cascade consumes the per-group (rows, N, K) triples and the
// slot route does not. Asking `moe_build_routing` for the triples where no
// kernel reads them costs the routing kernel a few stores per group for
// nothing, so a layer asks this per GEMM and requests the shapes exactly
// where they are consumed. Every architecture but sm_100/103 answers false.
// sm_120/121 has no pointer-array route: linear_mxfp8_grouped_masked validates
// such a tensor and ignores it, while linear_mxfp8_grouped_masked_swiglu
// refuses a non-empty one, so a caller driven by this query never passes one
// there. `fused_swiglu`
// names the kernel the call will land on, exactly as for
// `mxfp8_grouped_slot_possible`: the fused-SwiGLU FC1 leaves the slot route
// one step earlier than the plain grouped GEMM, so the two queries are
// complements only when both are asked with the same flags (a layer asks both
// for its FC1 with the flag set and for its FC2 with it clear, and both under
// the `compact` flag of the layout it runs).
bool mxfp8_grouped_problem_shapes_consumed(int64_t m_cap, int64_t n_w, int64_t k, int64_t num_groups,
    int64_t max_active_groups, bool fused_swiglu, bool compact, int64_t expected_m)
{
    if (!is_sm100_family() || !detail::sm100_mxfp8_grouped_compiled())
        return false;
    return detail::sm100_mxfp8_grouped_problem_shapes_consumed(static_cast<int>(m_cap), static_cast<int>(n_w),
               static_cast<int>(k), static_cast<int>(num_groups), static_cast<int>(max_active_groups),
               fused_swiglu ? 1 : 0, compact ? 1 : 0, static_cast<int>(expected_m))
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
    detail::require_bf16_vector_aligned(x, M, K, detail::fp8bs_quantize_1x32_packed_grouped_gather_x_vector_bytes(),
        "fish_scales_ops::quantize_1x32_grouped_gather", "x");

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
    // The interleaved row order arises from a fused FC1's weight layout, which
    // sm_100/103 and sm_120/121 both have; sm_90 has neither MXFP8 nor a fused
    // FC1, so its device code keeps exactly the instantiations it had.
    auto const* prop_pw = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(!pairwise || sm1xx || prop_pw->major == 12,
        "silu_chunk_mul_quantize_1x32_grouped(pairwise=True) reads the interleaved gate/up row order a fused "
        "FC1 produces, which exists on sm_100/sm_103 and sm_120/121 only.");
    if (slot_of_flat.numel() > 0) // no routed pair: the kernel reads nothing
        detail::require_bf16_vector_aligned(gu, static_cast<int64_t>(G) * m_cap, TWO_INTER,
            detail::fp8bs_silu_chunk_mul_quantize_1x32_packed_grouped_x_vector_bytes(pairwise),
            "fish_scales_ops::silu_chunk_mul_quantize_1x32_grouped", "gu");
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
