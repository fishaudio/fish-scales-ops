/*
 * ATen ops of the compact activation layout of the MoE layer (0.2.1, charter decisions D1 and D4).
 *
 * The masked slab gives every local expert m_cap rows of every activation-side tensor. The compact layout packs
 * the routed rows into ONE group of R rows made of 128-row blocks: expert g owns the blocks from blk0[g] on (blk0
 * is the routing's exclusive prefix of ceil(masked_m[g] / 128)), so its rows are [blk0[g] * 128,
 * blk0[g] * 128 + masked_m[g]). A compact row index is therefore exactly the slab slot of a masked slab with one
 * group of m_cap = R rows, which is why the glue kernels (gather-quantize, SwiGLU requantize, combine) run
 * unchanged on it with num_groups = 1, m_cap = R and a slot_of_flat that holds compact rows.
 *
 * Every op here is out-style: the layer allocates all of a call's buffers before the routing op runs (charter
 * D3), and the producers write into them.
 *
 *   quantize_1x32_grouped_gather_out          quantize_1x32_grouped_gather into given buffers
 *   silu_chunk_mul_quantize_1x32_grouped_out  silu_chunk_mul_quantize_1x32_grouped into given buffers
 *   linear_mxfp8_grouped_compact              the grouped GEMM: a [R, K] x b [G, N, K] -> d [R, N]
 *   linear_mxfp8_grouped_compact_swiglu       the fused-SwiGLU FC1 -> dq [R, I] + its one-group scale words
 *   linear_mxfp8_grouped_compact_combine      the fused-combine FC2 (sm_120/121, opt-in, not bit-reproducible)
 *
 * The two glue ops are arch-generic: they write exactly what their allocating twins return, in the layout those
 * twins use on the device (sm_100/103 the Sm1xx atom slab, every other device the K-major words). The three GEMM
 * ops serve sm_120/121 here; their sm_100/103 half is a separate piece of work (charter S5), and sm_90 has no
 * MXFP8 hardware.
 *
 * Rows of an output that no expert's live range covers are not written by any op here (in particular, every
 * store of the compact GEMMs is bounded by masked_m[g]); they hold whatever the caller's buffer held.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <torch/torch.h>

#include <cstdint>
#include <limits>
#include <optional>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace blockscale_gemm
{
namespace detail
{
// In mxfp8_compact_kernel.cu.
cudaError_t launch_sm120_mxfp8_grouped_compact_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_bfloat16* D,
    int32_t* SFA, int32_t* SFB, int32_t* masked_m, int32_t const* blk0, int num_groups, int m_cap, int act_rows,
    int N, int K, int expected_m, int max_active_groups, cudaStream_t stream);
cudaError_t launch_sm120_mxfp8_grouped_compact_swiglu_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B,
    __nv_fp8_e4m3* out_fp8, int32_t* out_sf, int32_t* SFA, int32_t* SFB, int32_t* masked_m, int32_t const* blk0,
    int num_groups, int m_cap, int act_rows, int N, int K, int expected_m, int max_active_groups,
    cudaStream_t stream);
cudaError_t launch_sm120_mxfp8_grouped_compact_combine_dispatch(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B,
    __nv_bfloat16* out_tokens, int32_t* SFA, int32_t* SFB, int32_t* masked_m, int32_t const* blk0,
    int32_t const* row_map, float const* weight_of_slot, int num_groups, int m_cap, int act_rows, int N, int K,
    int expected_m, int max_active_groups, cudaStream_t stream);
// In quant_kernels.cu (the launchers of the allocating twins).
void fp8bs_quantize_1x32_packed_grouped_gather(__nv_fp8_e4m3* x_q, int32_t* packed_scales, __nv_bfloat16 const* x,
    int32_t const* slot_of_flat, int n_pairs, int topk, int m_cap, int K, cudaStream_t stream, bool use_ue8m0,
    bool sm1xx_sf_layout);
void fp8bs_silu_chunk_mul_quantize_1x32_packed_grouped(__nv_fp8_e4m3* x_q, int32_t* packed_scales,
    __nv_bfloat16 const* gu, int32_t const* slot_of_flat, int n_pairs, int m_cap, int K, cudaStream_t stream,
    bool use_ue8m0, bool sm1xx_sf_layout, bool pairwise);
} // namespace detail

namespace
{

int current_sm_major()
{
    return at::cuda::getCurrentDeviceProperties()->major;
}

// The int32 scale words a grouped activation slab of `groups` groups of `m_cap` rows and `kp` K-blocks of 128
// carries on this device: one atom slab per group padded to 128 rows on sm_100/103, K-major words elsewhere.
int64_t grouped_scale_words(bool sm1xx, int64_t groups, int64_t m_cap, int64_t kp)
{
    int64_t const rows = sm1xx ? (m_cap + 127) / 128 * 128 : m_cap;
    return groups * rows * kp;
}

// The three compact GEMMs: sm_120/121 is served here, sm_100/103 by ops/mxfp8.cu (the combine is sm_120/121
// only), and every other device has no MXFP8 hardware.
void require_compact_gemm_arch(char const* op)
{
    int const major = current_sm_major();
    if (major == 12)
    {
        return;
    }
    TORCH_CHECK_NOT_IMPLEMENTED(false, op,
        ": the grouped MXFP8 (MoE) path needs sm_100/sm_103 (B200/B300) or sm_120/121 (RTX 5090 / RTX PRO 6000); "
        "this device is sm_", major, "x. On sm_90 (Hopper) use the block-FP8 grouped surface "
        "(linear_fp8_grouped_masked / moe_layer_fp8_sm90) -- Hopper has no MXFP8 hardware.");
}

bool given(std::optional<at::Tensor> const& t)
{
    return t.has_value() && t->numel() > 0;
}

struct CompactDims
{
    int64_t R;
    int64_t K;
    int64_t N;
    int64_t G;
};

// Shapes, dtypes, layouts and devices of the operands every compact GEMM shares. The host cannot check the one
// contract that matters most -- 128 * blk0[G] <= R, i.e. that the routing placed every row inside the buffer --
// without a device read; the routing op guarantees it for num_blocks <= R / 128.
CompactDims check_compact_gemm(char const* op, at::Tensor const& a, at::Tensor const& b, at::Tensor const& sa,
    at::Tensor const& sb, at::Tensor const& masked_m, at::Tensor const& blk0, int64_t m_cap, int64_t expected_m,
    int64_t max_active_groups)
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), op, ": a and b must be CUDA tensors");
    TORCH_CHECK(a.dtype() == at::kFloat8_e4m3fn && b.dtype() == at::kFloat8_e4m3fn, op,
        ": a and b must be float8_e4m3fn");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), op, ": a and b must be contiguous");
    TORCH_CHECK(a.dim() == 2 || (a.dim() == 3 && a.size(0) == 1), op,
        ": a must be the compact activation [R, K] (or its one-group view [1, R, K]), got ", a.sizes());
    TORCH_CHECK(b.dim() == 3, op, ": b must be the per-expert weights [G, N, K], got ", b.sizes());
    int64_t const R = a.size(-2);
    int64_t const K = a.size(-1);
    int64_t const G = b.size(0);
    int64_t const N = b.size(1);
    TORCH_CHECK(b.size(2) == K, op, ": b [G, N, K] must have a's K = ", K, ", got ", b.sizes());
    TORCH_CHECK(G >= 1 && G <= 1024, op, ": the grouped MXFP8 path supports 1 to 1024 experts, got G = ", G);
    TORCH_CHECK(K % 128 == 0, op, ": K must be a multiple of 128, got ", K);
    TORCH_CHECK(N % 128 == 0, op, ": N must be a multiple of 128 (BlockScaled atom requires it), got ", N);
    TORCH_CHECK(R >= 128 && R % 128 == 0 && R <= std::numeric_limits<int>::max(), op,
        ": the compact activation has R = 128 * B rows (B >= 1 blocks of 128 rows), got R = ", R);
    TORCH_CHECK(sa.is_cuda() && sb.is_cuda() && sa.dtype() == at::kInt && sb.dtype() == at::kInt, op,
        ": sa / sb must be CUDA int32 (packed UE8M0)");
    TORCH_CHECK(sa.is_contiguous() && sb.is_contiguous(), op, ": sa / sb must be contiguous");
    int64_t const kp = K / 128;
    TORCH_CHECK(sa.numel() == R * kp, op, ": sa must be the one-group K-major scale words [1, K/128, R] (",
        R * kp, " int32: quantize_1x32_grouped_gather_out with num_groups = 1, m_cap = R), got ", sa.numel());
    TORCH_CHECK(sb.numel() == G * kp * N, op, ": sb must be the per-expert scale words [G, K/128, N] (",
        G * kp * N, " int32), got ", sb.numel());
    TORCH_CHECK(masked_m.is_cuda() && masked_m.dtype() == at::kInt && masked_m.is_contiguous()
            && masked_m.numel() == G,
        op, ": masked_m must be contiguous CUDA int32 [G] = [", G, "], got ", masked_m.sizes());
    TORCH_CHECK(blk0.is_cuda() && blk0.dtype() == at::kInt && blk0.is_contiguous() && blk0.numel() == G + 1, op,
        ": blk0 must be contiguous CUDA int32 [G + 1] = [", G + 1,
        "] (the routing's exclusive prefix of ceil(masked_m / 128)), got ", blk0.sizes());
    TORCH_CHECK(m_cap >= 4 && m_cap % 4 == 0 && m_cap <= std::numeric_limits<int>::max(), op,
        ": m_cap is the masked layout's row capacity align(M, 4), the host-static hint the tile cascade and the "
        "grid bound read; it must be a positive multiple of 4, got ", m_cap);
    TORCH_CHECK(expected_m >= 1 && expected_m <= std::numeric_limits<int>::max(), op,
        ": expected_m must be >= 1, got ", expected_m);
    TORCH_CHECK(max_active_groups >= 0 && max_active_groups <= G, op,
        ": max_active_groups must be in [0, G]; got ", max_active_groups, " with G = ", G);
    auto const dev = a.device();
    TORCH_CHECK(b.device() == dev && sa.device() == dev && sb.device() == dev && masked_m.device() == dev
            && blk0.device() == dev,
        op, ": every tensor must be on ", dev);
    return {R, K, N, G};
}

// The sm_100/103 routes' per-call tensors. sm_120/121 has neither the slot-bound route nor the pointer-array
// route, so nothing would read them; a caller that passes one has built routing tensors for the wrong device.
void refuse_sm100_route_tensors(char const* op, std::optional<at::Tensor> const& slot_to_expert,
    std::optional<at::Tensor> const& problem_shapes, std::optional<at::Tensor> const& ptr_arrays)
{
    TORCH_CHECK(!given(slot_to_expert), op,
        ": the sm_120/121 grouped path has no slot-bound route, so it reads no active-expert list (pass None)");
    TORCH_CHECK(!given(problem_shapes), op,
        ": the sm_120/121 grouped path has no pointer-array route, so it reads no per-group problem shapes "
        "(pass None)");
    TORCH_CHECK(!given(ptr_arrays), op,
        ": the sm_120/121 grouped path has no pointer-array route, so it reads no per-call pointer arrays "
        "(pass None)");
}

} // anonymous namespace


// quantize_1x32_grouped_gather_out: quantize_1x32_grouped_gather writing into the caller's buffers. Called with
// num_groups = 1, m_cap = R and a compact slot_of_flat it fills the compact activation [R, K] and its one-group
// scale words; with the masked layout's arguments it fills a masked slab. x_q holds num_groups * m_cap * K fp8,
// packed the arch's scale words (sm_120 and sm_90: num_groups * K/128 * m_cap K-major; sm_100/103:
// num_groups * pad(m_cap, 128) * K/128 in Sm1xx atoms). Only the rows routed pairs map to are written.
void quantize_1x32_grouped_gather_out(at::Tensor x, at::Tensor slot_of_flat, int64_t topk, int64_t num_groups,
    int64_t m_cap, bool use_ue8m0, at::Tensor x_q, at::Tensor packed)
{
    TORCH_CHECK(x.is_cuda() && x.dtype() == at::kBFloat16, "x must be CUDA bf16");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(slot_of_flat.dtype() == at::kInt && slot_of_flat.is_cuda() && slot_of_flat.is_contiguous(),
        "slot_of_flat must be contiguous CUDA int32");
    int64_t const M = x.size(0);
    int64_t const K = x.size(1);
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    TORCH_CHECK(topk >= 1, "topk must be >= 1");
    TORCH_CHECK(num_groups >= 1, "num_groups must be >= 1");
    TORCH_CHECK(m_cap >= 1 && m_cap % 4 == 0, "m_cap must be a positive multiple of 4");
    TORCH_CHECK(slot_of_flat.numel() == M * topk, "slot_of_flat must have M * topk entries");
    TORCH_CHECK(num_groups * m_cap <= std::numeric_limits<int>::max() && M * topk <= std::numeric_limits<int>::max(),
        "num_groups * m_cap and M * topk must fit int32 (the slots are int32)");
    bool const sm1xx = current_sm_major() == 10;
    int64_t const kp = K / 128;
    int64_t const want_q = num_groups * m_cap * K;
    int64_t const want_sf = grouped_scale_words(sm1xx, num_groups, m_cap, kp);
    TORCH_CHECK(x_q.is_cuda() && x_q.dtype() == at::kFloat8_e4m3fn && x_q.is_contiguous() && x_q.numel() == want_q,
        "x_q must be a contiguous CUDA float8_e4m3fn tensor of num_groups * m_cap * K = ", want_q,
        " elements, got ", x_q.dtype(), " ", x_q.sizes());
    TORCH_CHECK(packed.is_cuda() && packed.dtype() == at::kInt && packed.is_contiguous() && packed.numel() == want_sf,
        "packed must be a contiguous CUDA int32 tensor of ", want_sf, " words (",
        sm1xx ? "num_groups * pad(m_cap, 128) * K/128, Sm1xx atoms" : "num_groups * K/128 * m_cap, K-major",
        "), got ", packed.dtype(), " ", packed.sizes());
    auto const dev = x.device();
    TORCH_CHECK(slot_of_flat.device() == dev && x_q.device() == dev && packed.device() == dev,
        "every tensor must be on ", dev);
    if (M * topk == 0)
    {
        return; // no routed pair: nothing to write
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    detail::fp8bs_quantize_1x32_packed_grouped_gather(reinterpret_cast<__nv_fp8_e4m3*>(x_q.data_ptr()),
        reinterpret_cast<int32_t*>(packed.data_ptr()), reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr()),
        reinterpret_cast<int32_t const*>(slot_of_flat.data_ptr()), static_cast<int>(M * topk),
        static_cast<int>(topk), static_cast<int>(m_cap), static_cast<int>(K), stream, use_ue8m0, sm1xx);
}


// silu_chunk_mul_quantize_1x32_grouped_out: silu_chunk_mul_quantize_1x32_grouped writing into the caller's
// buffers. gu is the FC1 output, the masked slab [G, m_cap, 2*INTER] or the compact one-group [R, 2*INTER]
// (G = 1, m_cap = R); dq holds G * m_cap * INTER fp8 and sd the arch's scale words for (G, m_cap, INTER/128).
// Only the rows routed pairs map to are written.
void silu_chunk_mul_quantize_1x32_grouped_out(
    at::Tensor gu, at::Tensor slot_of_flat, bool use_ue8m0, bool pairwise, at::Tensor dq, at::Tensor sd)
{
    TORCH_CHECK(gu.is_cuda() && gu.dtype() == at::kBFloat16, "gu must be CUDA bf16");
    TORCH_CHECK(gu.dim() == 3 || gu.dim() == 2,
        "gu must be the masked [G, m_cap, 2*INTER] or the compact one-group [R, 2*INTER], got ", gu.sizes());
    TORCH_CHECK(gu.is_contiguous(), "gu must be contiguous");
    TORCH_CHECK(slot_of_flat.dtype() == at::kInt && slot_of_flat.is_cuda() && slot_of_flat.is_contiguous(),
        "slot_of_flat must be contiguous CUDA int32");
    int64_t const G = gu.dim() == 3 ? gu.size(0) : 1;
    int64_t const m_cap = gu.size(-2);
    int64_t const TWO_INTER = gu.size(-1);
    TORCH_CHECK(TWO_INTER % 2 == 0, "last dim must be even (gate || up)");
    int64_t const INTER = TWO_INTER / 2;
    TORCH_CHECK(INTER % 128 == 0, "INTER must be a multiple of 128");
    TORCH_CHECK(m_cap % 4 == 0, "m_cap must be a multiple of 4");
    TORCH_CHECK(G * m_cap <= std::numeric_limits<int>::max()
            && slot_of_flat.numel() <= std::numeric_limits<int>::max(),
        "G * m_cap and the pair count must fit int32 (the slots are int32)");
    int const major = current_sm_major();
    bool const sm1xx = major == 10;
    // The interleaved row order arises from a fused FC1's weight layout, which sm_100/103 and sm_120/121 have;
    // the allocating twin refuses it elsewhere, and so does this op.
    TORCH_CHECK(!pairwise || sm1xx || major == 12,
        "silu_chunk_mul_quantize_1x32_grouped_out(pairwise=True) reads the interleaved gate/up row order a fused "
        "FC1 produces, which exists on sm_100/sm_103 and sm_120/121 only.");
    int64_t const kp = INTER / 128;
    int64_t const want_q = G * m_cap * INTER;
    int64_t const want_sf = grouped_scale_words(sm1xx, G, m_cap, kp);
    TORCH_CHECK(dq.is_cuda() && dq.dtype() == at::kFloat8_e4m3fn && dq.is_contiguous() && dq.numel() == want_q,
        "dq must be a contiguous CUDA float8_e4m3fn tensor of G * m_cap * INTER = ", want_q, " elements, got ",
        dq.dtype(), " ", dq.sizes());
    TORCH_CHECK(sd.is_cuda() && sd.dtype() == at::kInt && sd.is_contiguous() && sd.numel() == want_sf,
        "sd must be a contiguous CUDA int32 tensor of ", want_sf, " words (",
        sm1xx ? "G * pad(m_cap, 128) * INTER/128, Sm1xx atoms" : "G * INTER/128 * m_cap, K-major", "), got ",
        sd.dtype(), " ", sd.sizes());
    auto const dev = gu.device();
    TORCH_CHECK(slot_of_flat.device() == dev && dq.device() == dev && sd.device() == dev,
        "every tensor must be on ", dev);
    if (slot_of_flat.numel() == 0)
    {
        return; // no routed pair: nothing to write
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    detail::fp8bs_silu_chunk_mul_quantize_1x32_packed_grouped(reinterpret_cast<__nv_fp8_e4m3*>(dq.data_ptr()),
        reinterpret_cast<int32_t*>(sd.data_ptr()), reinterpret_cast<__nv_bfloat16 const*>(gu.data_ptr()),
        reinterpret_cast<int32_t const*>(slot_of_flat.data_ptr()), static_cast<int>(slot_of_flat.numel()),
        static_cast<int>(m_cap), static_cast<int>(INTER), stream, use_ue8m0, sm1xx, pairwise);
}


// The sm_100/sm_103 halves of the two GEMM ops (ops/mxfp8.cu): the CUTLASS pointer-array and slot routes on the
// compact layout, with their own argument checks. The entries below send sm_100/sm_103 calls there.
void linear_mxfp8_grouped_compact_sm100(at::Tensor a, at::Tensor b, at::Tensor sa, at::Tensor sb,
    at::Tensor masked_m, at::Tensor blk0, int64_t m_cap, int64_t expected_m, int64_t max_active_groups,
    std::optional<at::Tensor> slot_to_expert, std::optional<at::Tensor> problem_shapes,
    std::optional<at::Tensor> ptr_arrays, at::Tensor d);
void linear_mxfp8_grouped_compact_swiglu_sm100(at::Tensor a, at::Tensor b, at::Tensor sa, at::Tensor sb,
    at::Tensor masked_m, at::Tensor blk0, int64_t m_cap, int64_t expected_m, int64_t max_active_groups,
    std::optional<at::Tensor> slot_to_expert, std::optional<at::Tensor> problem_shapes,
    std::optional<at::Tensor> ptr_arrays, at::Tensor dq, at::Tensor sd);

// linear_mxfp8_grouped_compact: the grouped MXFP8 GEMM on the compact layout. a fp8 [R, K] (expert g's rows from
// blk0[g] * 128), b fp8 [G, N, K], sa the one-group scale words [1, K/128, R], sb [G, K/128, N], masked_m [G],
// blk0 [G + 1]; writes d bf16 [R, N] at the live rows only. m_cap / expected_m / max_active_groups are the masked
// layout's host-static hints (align(M, 4), ceil(M * topk / G), min(M * topk, G)), so the call runs the instance
// the masked call of the same shape runs. The three optional tensors belong to the sm_100/103 routes.
void linear_mxfp8_grouped_compact(at::Tensor a, at::Tensor b, at::Tensor sa, at::Tensor sb, at::Tensor masked_m,
    at::Tensor blk0, int64_t m_cap, int64_t expected_m, int64_t max_active_groups,
    std::optional<at::Tensor> slot_to_expert, std::optional<at::Tensor> problem_shapes,
    std::optional<at::Tensor> ptr_arrays, at::Tensor d)
{
    char const* const op = "linear_mxfp8_grouped_compact";
    if (current_sm_major() == 10)
    {
        linear_mxfp8_grouped_compact_sm100(a, b, sa, sb, masked_m, blk0, m_cap, expected_m, max_active_groups,
            slot_to_expert, problem_shapes, ptr_arrays, d);
        return;
    }
    require_compact_gemm_arch(op);
    CompactDims const s = check_compact_gemm(op, a, b, sa, sb, masked_m, blk0, m_cap, expected_m, max_active_groups);
    refuse_sm100_route_tensors(op, slot_to_expert, problem_shapes, ptr_arrays);
    TORCH_CHECK(d.is_cuda() && d.dtype() == at::kBFloat16 && d.is_contiguous(), op,
        ": d must be a contiguous CUDA bf16 tensor");
    TORCH_CHECK(d.numel() == s.R * s.N && d.size(-1) == s.N, op, ": d must be [R, N] = [", s.R, ", ", s.N,
        "] (or its one-group view [1, R, N]), got ", d.sizes());
    TORCH_CHECK(d.device() == a.device(), op, ": d must be on ", a.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaError_t const err = detail::launch_sm120_mxfp8_grouped_compact_dispatch(
        reinterpret_cast<__nv_fp8_e4m3*>(a.data_ptr()), reinterpret_cast<__nv_fp8_e4m3*>(b.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), reinterpret_cast<int32_t*>(sa.data_ptr()),
        reinterpret_cast<int32_t*>(sb.data_ptr()), reinterpret_cast<int32_t*>(masked_m.data_ptr()),
        reinterpret_cast<int32_t const*>(blk0.data_ptr()), static_cast<int>(s.G), static_cast<int>(m_cap),
        static_cast<int>(s.R), static_cast<int>(s.N), static_cast<int>(s.K), static_cast<int>(expected_m),
        static_cast<int>(max_active_groups), stream);
    TORCH_CHECK(err == cudaSuccess, op, ": sm120 kernel error: ", cudaGetErrorString(err));
}


// linear_mxfp8_grouped_compact_swiglu: the fused-SwiGLU FC1 on the compact layout. b is the FC1 weight [G, 2*I, K]
// with gate and up rows INTERLEAVED (row 2j gate_j, row 2j+1 up_j; the op cannot detect the stacked order, which
// gives a silently wrong result), I a multiple of 128. Writes dq fp8 [R, I] and sd, the one-group K-major scale
// words [1, I/128, R], at the live rows only -- the pair silu_chunk_mul_quantize_1x32_grouped_out writes from the
// unfused FC1's output, bit for bit.
void linear_mxfp8_grouped_compact_swiglu(at::Tensor a, at::Tensor b, at::Tensor sa, at::Tensor sb,
    at::Tensor masked_m, at::Tensor blk0, int64_t m_cap, int64_t expected_m, int64_t max_active_groups,
    std::optional<at::Tensor> slot_to_expert, std::optional<at::Tensor> problem_shapes,
    std::optional<at::Tensor> ptr_arrays, at::Tensor dq, at::Tensor sd)
{
    char const* const op = "linear_mxfp8_grouped_compact_swiglu";
    if (current_sm_major() == 10)
    {
        linear_mxfp8_grouped_compact_swiglu_sm100(a, b, sa, sb, masked_m, blk0, m_cap, expected_m,
            max_active_groups, slot_to_expert, problem_shapes, ptr_arrays, dq, sd);
        return;
    }
    require_compact_gemm_arch(op);
    CompactDims const s = check_compact_gemm(op, a, b, sa, sb, masked_m, blk0, m_cap, expected_m, max_active_groups);
    refuse_sm100_route_tensors(op, slot_to_expert, problem_shapes, ptr_arrays);
    int64_t const INTER = s.N / 2;
    TORCH_CHECK(s.N % 2 == 0 && INTER % 128 == 0, op,
        ": b must be [G, 2*I, K] with I a multiple of 128 (a 1x32 output scale block must not straddle the packed "
        "int32 word), got N = ", s.N);
    TORCH_CHECK(dq.is_cuda() && dq.dtype() == at::kFloat8_e4m3fn && dq.is_contiguous(), op,
        ": dq must be a contiguous CUDA float8_e4m3fn tensor");
    TORCH_CHECK(dq.numel() == s.R * INTER && dq.size(-1) == INTER, op, ": dq must be [R, I] = [", s.R, ", ", INTER,
        "] (or its one-group view [1, R, I]), got ", dq.sizes());
    TORCH_CHECK(sd.is_cuda() && sd.dtype() == at::kInt && sd.is_contiguous() && sd.numel() == s.R * (INTER / 128),
        op, ": sd must be the one-group K-major scale words [1, I/128, R] (", s.R * (INTER / 128),
        " contiguous CUDA int32), got ", sd.dtype(), " ", sd.sizes());
    TORCH_CHECK(dq.device() == a.device() && sd.device() == a.device(), op, ": dq and sd must be on ", a.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaError_t const err = detail::launch_sm120_mxfp8_grouped_compact_swiglu_dispatch(
        reinterpret_cast<__nv_fp8_e4m3*>(a.data_ptr()), reinterpret_cast<__nv_fp8_e4m3*>(b.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(dq.data_ptr()), reinterpret_cast<int32_t*>(sd.data_ptr()),
        reinterpret_cast<int32_t*>(sa.data_ptr()), reinterpret_cast<int32_t*>(sb.data_ptr()),
        reinterpret_cast<int32_t*>(masked_m.data_ptr()), reinterpret_cast<int32_t const*>(blk0.data_ptr()),
        static_cast<int>(s.G), static_cast<int>(m_cap), static_cast<int>(s.R), static_cast<int>(s.N),
        static_cast<int>(s.K), static_cast<int>(expected_m), static_cast<int>(max_active_groups), stream);
    TORCH_CHECK(err == cudaSuccess, op, ": sm120 fused-SwiGLU kernel error: ", cudaGetErrorString(err));
}


// linear_mxfp8_grouped_compact_combine: the fused-combine FC2 on the compact layout (sm_120/121 only). Each live
// row r of expert g (r in [blk0[g] * 128, blk0[g] * 128 + masked_m[g])) is scaled by weight_of_slot[r] and ADDED
// into out[row_map[r]] with atomic reductions, so `out` [M, N] must be pre-filled (zero, or a shared expert's
// output) and the result is not bit-reproducible run to run. row_map / weight_of_slot are indexed by compact row:
// the routing op writes the first 128 * B_max rows, and a layer allocates R = 128 * (B_max + 1) (one block of
// slack), so they must cover at least R - 128 rows.
void linear_mxfp8_grouped_compact_combine(at::Tensor a, at::Tensor b, at::Tensor sa, at::Tensor sb,
    at::Tensor masked_m, at::Tensor blk0, at::Tensor row_map, at::Tensor weight_of_slot, int64_t m_cap,
    int64_t expected_m, int64_t max_active_groups, at::Tensor out)
{
    char const* const op = "linear_mxfp8_grouped_compact_combine";
    int const major = current_sm_major();
    TORCH_CHECK_NOT_IMPLEMENTED(major == 12, op,
        " is the sm_120/121 fused-combine FC2; sm_100/103 serves this band from its own routes and sm_90 has no "
        "MXFP8 hardware (this device is sm_", major, "x).");
    CompactDims const s = check_compact_gemm(op, a, b, sa, sb, masked_m, blk0, m_cap, expected_m, max_active_groups);
    TORCH_CHECK(row_map.is_cuda() && row_map.dtype() == at::kInt && row_map.is_contiguous(), op,
        ": row_map must be contiguous CUDA int32 (compact row -> token)");
    TORCH_CHECK(weight_of_slot.is_cuda() && weight_of_slot.dtype() == at::kFloat && weight_of_slot.is_contiguous(),
        op, ": weight_of_slot must be contiguous CUDA fp32 (compact row -> combine weight)");
    TORCH_CHECK(row_map.numel() >= s.R - 128 && weight_of_slot.numel() >= s.R - 128, op,
        ": row_map and weight_of_slot are indexed by compact row and must cover the routed rows, at least R - 128 = ",
        s.R - 128, " entries (the routing's 128 * B_max rows; the buffers' last block is allocation slack), got ",
        row_map.numel(), " and ", weight_of_slot.numel());
    TORCH_CHECK(out.is_cuda() && out.dtype() == at::kBFloat16 && out.dim() == 2 && out.is_contiguous(), op,
        ": out must be a contiguous CUDA bf16 [M, N] tensor, pre-filled with what the layer adds to");
    TORCH_CHECK(out.size(1) == s.N, op, ": out must have N = ", s.N, " columns (b.size(1)), got ", out.sizes());
    auto const dev = a.device();
    TORCH_CHECK(row_map.device() == dev && weight_of_slot.device() == dev && out.device() == dev, op,
        ": every tensor must be on ", dev);
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaError_t const err = detail::launch_sm120_mxfp8_grouped_compact_combine_dispatch(
        reinterpret_cast<__nv_fp8_e4m3*>(a.data_ptr()), reinterpret_cast<__nv_fp8_e4m3*>(b.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), reinterpret_cast<int32_t*>(sa.data_ptr()),
        reinterpret_cast<int32_t*>(sb.data_ptr()), reinterpret_cast<int32_t*>(masked_m.data_ptr()),
        reinterpret_cast<int32_t const*>(blk0.data_ptr()), reinterpret_cast<int32_t const*>(row_map.data_ptr()),
        reinterpret_cast<float const*>(weight_of_slot.data_ptr()), static_cast<int>(s.G), static_cast<int>(m_cap),
        static_cast<int>(s.R), static_cast<int>(s.N), static_cast<int>(s.K), static_cast<int>(expected_m),
        static_cast<int>(max_active_groups), stream);
    TORCH_CHECK(err == cudaSuccess, op, ": sm120 fused-combine kernel error: ", cudaGetErrorString(err));
}

} // namespace blockscale_gemm
