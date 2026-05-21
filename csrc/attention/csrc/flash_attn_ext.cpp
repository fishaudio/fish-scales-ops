// torch.ops + PYBIND11_MODULE entry for the fish_scales_ops attention domain.
//
// Registers three torch.ops in the `fish_scales_ops` namespace:
//   mxfp8_attn_fwd        — SM120 MXFP8 prefill, contiguous K/V (per-D dispatch).
//   mxfp8_decode_paged    — SM120 MXFP8 paged-KV decode (S_q=1).
//   mxfp8_attn_fwd_paged  — SM120 MXFP8 paged-KV prefill / extend (S_q > 1),
//                           flashinfer-style ragged + paged signature.
//
// Both host dispatchers return cudaErrorNotSupported on devices below
// sm_120; the kernel bodies are __CUDA_ARCH__-guarded to compile as
// empty stubs under sm_90a.
//
// The single PYBIND11_MODULE for the combined extension lives here. The
// GEMM bindings register their own TORCH_LIBRARY_FRAGMENT in
// csrc/gemm/bindings.cpp (no pybind11 surface — torch.ops only).

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// SM120 MXFP8 forward — top-level dispatcher, defined in
// kernels/sm120/mxfp8_attn_fwd.cu. Routes to per-D launchers.
extern "C" cudaError_t mxfp8_attn_fwd_launch(
    const void* Q, const void* Qs,
    const void* K, const void* Ks,
    const void* V, const void* Vs,
    void* O,
    int batch, int num_q_heads, int num_kv_heads,
    int seq_q, int seq_k, int head_dim,
    float softmax_scale, int causal,
    cudaStream_t stream);

// SM120 MXFP8 paged-KV decode — top-level dispatcher, defined in
// kernels/sm120/mxfp8_decode_paged.cu. Routes to per-D launchers.
extern "C" cudaError_t mxfp8_decode_paged_launch(
    const void* Q, const void* Qs,
    const void* K_pool, const void* K_chan_scale,
    const void* V_pool, const void* V_chan_scale,
    const void* block_table, const void* seq_lens,
    void* O,
    void* M_partial, void* L_partial, void* O_partial,
    void* sync_counter,
    int batch, int num_q_heads, int num_kv_heads,
    int max_blocks, int head_dim, int num_splits, int target_counter,
    int page_size,
    float softmax_scale,
    cudaStream_t stream);

// kernels/sm120/mxfp8_attn_fwd_paged.cu.
extern "C" cudaError_t mxfp8_attn_fwd_paged_launch(
    const void* Q, const void* Qs,
    const void* K_pool, const void* K_chan_scale,
    const void* V_pool, const void* V_chan_scale,
    const void* qo_indptr,
    const void* paged_kv_indices,
    const void* paged_kv_indptr,
    const void* paged_kv_last_page_len,
    const void* work_units,
    void* O,
    int total_work, int num_q_heads, int num_kv_heads,
    int head_dim, int page_size, int causal,
    float softmax_scale,
    cudaStream_t stream);

namespace fish_scales_ops_attn {

at::Tensor mxfp8_attn_fwd(at::Tensor q_fp8, at::Tensor q_scales,
                          at::Tensor k_fp8, at::Tensor k_scales,
                          at::Tensor v_fp8, at::Tensor v_scales,
                          double softmax_scale, bool causal)
{
    TORCH_CHECK(q_fp8.is_cuda() && k_fp8.is_cuda() && v_fp8.is_cuda(),
                "q/k/v must be CUDA tensors");
    TORCH_CHECK(q_scales.is_cuda() && k_scales.is_cuda() && v_scales.is_cuda(),
                "q/k/v scales must be CUDA tensors");
    TORCH_CHECK(q_fp8.dtype() == at::kFloat8_e4m3fn &&
                k_fp8.dtype() == at::kFloat8_e4m3fn &&
                v_fp8.dtype() == at::kFloat8_e4m3fn,
                "q_fp8 / k_fp8 / v_fp8 must be float8_e4m3fn");
    TORCH_CHECK(q_scales.dtype() == at::kByte &&
                k_scales.dtype() == at::kByte &&
                v_scales.dtype() == at::kByte,
                "q/k/v scales must be uint8 (UE8M0)");
    TORCH_CHECK(q_fp8.dim() == 4 && k_fp8.dim() == 4 && v_fp8.dim() == 4,
                "q/k/v must be 4D");

    // Q: [B, Sq, H_q, D]   K: [B, Sk, H_kv, D]   V: [B, D, H_kv, Sk] (pre-transposed)
    const int64_t B    = q_fp8.size(0);
    const int64_t Sq   = q_fp8.size(1);
    const int64_t H_q  = q_fp8.size(2);
    const int64_t D    = q_fp8.size(3);
    const int64_t Sk   = k_fp8.size(1);
    const int64_t H_kv = k_fp8.size(2);
    TORCH_CHECK(k_fp8.size(0) == B && k_fp8.size(3) == D,
                "K shape mismatch: expected [B,Sk,H_kv,D]");
    TORCH_CHECK(v_fp8.size(0) == B && v_fp8.size(1) == D &&
                v_fp8.size(2) == H_kv && v_fp8.size(3) == Sk,
                "V must be pre-transposed [B,D,H_kv,Sk]");
    TORCH_CHECK(H_q % H_kv == 0, "H_q must be a multiple of H_kv (GQA)");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto o = at::empty({B, Sq, H_q, D},
                       q_fp8.options().dtype(at::kBFloat16));

    cudaError_t err = ::mxfp8_attn_fwd_launch(
        q_fp8.data_ptr(), q_scales.data_ptr(),
        k_fp8.data_ptr(), k_scales.data_ptr(),
        v_fp8.data_ptr(), v_scales.data_ptr(),
        o.data_ptr(),
        static_cast<int>(B), static_cast<int>(H_q), static_cast<int>(H_kv),
        static_cast<int>(Sq), static_cast<int>(Sk), static_cast<int>(D),
        static_cast<float>(softmax_scale),
        causal ? 1 : 0,
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "mxfp8_attn_fwd launch failed: ",
                cudaGetErrorString(err));
    return o;
}

// mxfp8_decode_paged: paged-KV decode forward.
//   q_fp8        : [B, 1, H_q, D] FP8 — single decode token per batch entry
//   q_scales     : [B, 1, H_q, D/32] UE8M0
//   k_pool       : [P, page_size, H_kv, D] FP8           (P = total pages in the pool)
//   k_pool_scales: [P, page_size, H_kv, D/32] UE8M0
//   v_pool       : [P, D, H_kv, page_size] FP8 (pre-transposed)
//   v_chan_scale : [H_kv, D] fp32 — per-channel V scale (no per-page layout)
//   block_table  : [B, max_blocks] int32  per-sequence list of page ids
//   seq_lens     : [B] int32              actual token counts per sequence
//   m_partial    : [B, H_q, num_splits] fp32 scratch (re-zeroed per call)
//   l_partial    : [B, H_q, num_splits] fp32 scratch
//   o_partial    : [B, H_q, num_splits, D] bf16 scratch
//   sync_counter : [B, H_q] int32          inter-split sync flags (callers
//                                          managing kv_split_k must increment
//                                          target_counter accordingly)
//   num_splits   : int                     kv split factor
//   target_counter: int                    expected sync_counter value
//   softmax_scale: float
// Returns: bf16 [B, H_q, D]
at::Tensor mxfp8_decode_paged(
    at::Tensor q_fp8, at::Tensor q_scales,
    at::Tensor k_pool, at::Tensor k_chan_scale,
    at::Tensor v_pool, at::Tensor v_chan_scale,
    at::Tensor block_table, at::Tensor seq_lens,
    at::Tensor m_partial, at::Tensor l_partial, at::Tensor o_partial,
    at::Tensor sync_counter,
    int64_t num_splits, int64_t target_counter,
    double softmax_scale)
{
    TORCH_CHECK(q_fp8.is_cuda() && k_pool.is_cuda() && v_pool.is_cuda(),
                "q / k_pool / v_pool must be CUDA");
    TORCH_CHECK(q_fp8.dtype() == at::kFloat8_e4m3fn &&
                k_pool.dtype() == at::kFloat8_e4m3fn &&
                v_pool.dtype() == at::kFloat8_e4m3fn,
                "q/k/v must be float8_e4m3fn");
    TORCH_CHECK(q_scales.dtype() == at::kByte &&
                k_chan_scale.dtype() == at::kByte,
                "q_scales / k_chan_scale must be uint8 (UE8M0)");
    TORCH_CHECK(v_chan_scale.dtype() == at::kFloat,
                "v_chan_scale must be fp32 [H_kv, D]");
    TORCH_CHECK(block_table.dtype() == at::kInt && seq_lens.dtype() == at::kInt,
                "block_table / seq_lens must be int32");
    TORCH_CHECK(q_fp8.dim() == 3, "q must be 3D [B, H_q, D] for single-token decode");

    const int64_t B    = q_fp8.size(0);
    const int64_t H_q  = q_fp8.size(1);
    const int64_t D    = q_fp8.size(2);
    const int64_t page_size = k_pool.size(1);
    const int64_t H_kv = k_pool.size(2);
    const int64_t max_blocks = block_table.size(1);
    TORCH_CHECK(k_pool.size(3) == D, "k_pool head_dim mismatch");
    TORCH_CHECK(v_pool.size(1) == D, "v_pool head_dim mismatch");
    TORCH_CHECK(v_pool.size(2) == H_kv, "v_pool H_kv mismatch");
    TORCH_CHECK(v_pool.size(3) == page_size,
                "v_pool page_size mismatch: K and V pools must share page_size");
    TORCH_CHECK(k_chan_scale.dim() == 2 &&
                k_chan_scale.size(0) == H_kv &&
                k_chan_scale.size(1) == D / 32,
                "k_chan_scale must be [H_kv, D/32] UE8M0");
    TORCH_CHECK(v_chan_scale.dim() == 2 &&
                v_chan_scale.size(0) == H_kv &&
                v_chan_scale.size(1) == D,
                "v_chan_scale must be [H_kv, D] fp32");
    TORCH_CHECK(page_size > 0 && page_size % 32 == 0,
                "page_size must be a positive multiple of 32 (MXFP8 sf_vec_size)");
    TORCH_CHECK(block_table.size(0) == B && seq_lens.size(0) == B,
                "block_table/seq_lens batch mismatch");
    TORCH_CHECK(H_q % H_kv == 0, "H_q must be a multiple of H_kv (GQA)");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto o = at::empty({B, H_q, D},
                       q_fp8.options().dtype(at::kBFloat16));

    cudaError_t err = ::mxfp8_decode_paged_launch(
        q_fp8.data_ptr(), q_scales.data_ptr(),
        k_pool.data_ptr(), k_chan_scale.data_ptr(),
        v_pool.data_ptr(), v_chan_scale.data_ptr(),
        block_table.data_ptr(), seq_lens.data_ptr(),
        o.data_ptr(),
        m_partial.data_ptr(), l_partial.data_ptr(), o_partial.data_ptr(),
        sync_counter.data_ptr(),
        static_cast<int>(B), static_cast<int>(H_q), static_cast<int>(H_kv),
        static_cast<int>(max_blocks), static_cast<int>(D),
        static_cast<int>(num_splits), static_cast<int>(target_counter),
        static_cast<int>(page_size),
        static_cast<float>(softmax_scale),
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "mxfp8_decode_paged launch failed: ",
                cudaGetErrorString(err));
    return o;
}

// mxfp8_attn_fwd_paged: paged-KV prefill / extend forward.
//   q_fp8                  : [total_q_tokens, H_q, D] FP8
//   q_scales               : [total_q_tokens / 16, H_q, D/32] UE8M0
//   k_pool                 : [num_pages, page_size, H_kv, D] FP8
//   k_pool_scales          : [num_pages, H_kv, D/32] UE8M0  (max-pooled across S)
//   v_pool                 : [num_pages, D, H_kv, page_size] FP8  (pre-transposed)
//   v_pool_scales          : [num_pages, page_size/32, H_kv] UE8M0
//   qo_indptr              : [B+1] int32  cumulative Q-token counts
//   paged_kv_indices       : [total_pages_used] int32  flat page-id list
//   paged_kv_indptr        : [B+1] int32  per-req slice into paged_kv_indices
//   paged_kv_last_page_len : [B]   int32  valid tokens in the last page
//   work_units             : [total_work*3] int32  (b, h_q, q_tile_in_b) tuples,
//                            packed by the wrapper so the kernel skips a binary search.
//   total_work             : int   number of work units
//   causal                 : bool
//   softmax_scale          : float
// Returns: bf16 [total_q_tokens, H_q, D]
at::Tensor mxfp8_attn_fwd_paged(
    at::Tensor q_fp8, at::Tensor q_scales,
    at::Tensor k_pool, at::Tensor k_chan_scale,
    at::Tensor v_pool, at::Tensor v_chan_scale,
    at::Tensor qo_indptr,
    at::Tensor paged_kv_indices,
    at::Tensor paged_kv_indptr,
    at::Tensor paged_kv_last_page_len,
    at::Tensor work_units,
    int64_t total_work,
    bool causal,
    double softmax_scale)
{
    TORCH_CHECK(q_fp8.is_cuda() && k_pool.is_cuda() && v_pool.is_cuda(),
                "q / k_pool / v_pool must be CUDA");
    TORCH_CHECK(q_fp8.dtype() == at::kFloat8_e4m3fn &&
                k_pool.dtype() == at::kFloat8_e4m3fn &&
                v_pool.dtype() == at::kFloat8_e4m3fn,
                "q/k/v must be float8_e4m3fn");
    TORCH_CHECK(q_scales.dtype() == at::kByte &&
                k_chan_scale.dtype() == at::kByte,
                "q_scales / k_chan_scale must be uint8 (UE8M0)");
    TORCH_CHECK(v_chan_scale.dtype() == at::kFloat,
                "v_chan_scale must be fp32 [H_kv, D]");
    TORCH_CHECK(qo_indptr.dtype() == at::kInt &&
                paged_kv_indices.dtype() == at::kInt &&
                paged_kv_indptr.dtype() == at::kInt &&
                paged_kv_last_page_len.dtype() == at::kInt &&
                work_units.dtype() == at::kInt,
                "index/length tensors must be int32");
    TORCH_CHECK(q_fp8.dim() == 3, "q must be 3D [total_q, H_q, D]");
    TORCH_CHECK(k_pool.dim() == 4, "k_pool must be 4D [num_pages, page_size, H_kv, D]");
    TORCH_CHECK(v_pool.dim() == 4, "v_pool must be 4D [num_pages, D, H_kv, page_size]");

    const int64_t total_q   = q_fp8.size(0);
    const int64_t H_q       = q_fp8.size(1);
    const int64_t D         = q_fp8.size(2);
    const int64_t page_size = k_pool.size(1);
    const int64_t H_kv      = k_pool.size(2);
    TORCH_CHECK(k_pool.size(3) == D, "k_pool head_dim mismatch");
    TORCH_CHECK(v_pool.size(1) == D, "v_pool head_dim mismatch");
    TORCH_CHECK(v_pool.size(2) == H_kv, "v_pool H_kv mismatch");
    TORCH_CHECK(v_pool.size(3) == page_size,
                "v_pool page_size mismatch: K and V pools must share page_size");
    TORCH_CHECK(k_chan_scale.dim() == 2 &&
                k_chan_scale.size(0) == H_kv &&
                k_chan_scale.size(1) == D / 32,
                "k_chan_scale must be [H_kv, D/32] UE8M0");
    TORCH_CHECK(v_chan_scale.dim() == 2 &&
                v_chan_scale.size(0) == H_kv &&
                v_chan_scale.size(1) == D,
                "v_chan_scale must be [H_kv, D] fp32");
    TORCH_CHECK(page_size > 0 && page_size % 32 == 0,
                "page_size must be a positive multiple of 32 (MXFP8 sf_vec_size)");
    TORCH_CHECK(H_q % H_kv == 0, "H_q must be a multiple of H_kv (GQA)");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto o = at::empty({total_q, H_q, D},
                       q_fp8.options().dtype(at::kBFloat16));

    cudaError_t err = ::mxfp8_attn_fwd_paged_launch(
        q_fp8.data_ptr(), q_scales.data_ptr(),
        k_pool.data_ptr(), k_chan_scale.data_ptr(),
        v_pool.data_ptr(), v_chan_scale.data_ptr(),
        qo_indptr.data_ptr(),
        paged_kv_indices.data_ptr(),
        paged_kv_indptr.data_ptr(),
        paged_kv_last_page_len.data_ptr(),
        work_units.data_ptr(),
        o.data_ptr(),
        static_cast<int>(total_work),
        static_cast<int>(H_q), static_cast<int>(H_kv),
        static_cast<int>(D), static_cast<int>(page_size),
        causal ? 1 : 0,
        static_cast<float>(softmax_scale),
        stream.stream());
    TORCH_CHECK(err == cudaSuccess,
                "mxfp8_attn_fwd_paged launch failed: ",
                cudaGetErrorString(err));
    return o;
}

} // namespace fish_scales_ops_attn

TORCH_LIBRARY_FRAGMENT(fish_scales_ops, m) {
    m.def("mxfp8_attn_fwd(Tensor q_fp8, Tensor q_scales, "
                          "Tensor k_fp8, Tensor k_scales, "
                          "Tensor v_fp8, Tensor v_scales, "
                          "float softmax_scale, bool causal) -> Tensor");
    m.def("mxfp8_decode_paged(Tensor q_fp8, Tensor q_scales, "
                              "Tensor k_pool, Tensor k_chan_scale, "
                              "Tensor v_pool, Tensor v_chan_scale, "
                              "Tensor block_table, Tensor seq_lens, "
                              "Tensor m_partial, Tensor l_partial, Tensor o_partial, "
                              "Tensor sync_counter, "
                              "int num_splits, int target_counter, "
                              "float softmax_scale) -> Tensor");
    m.def("mxfp8_attn_fwd_paged(Tensor q_fp8, Tensor q_scales, "
                                "Tensor k_pool, Tensor k_chan_scale, "
                                "Tensor v_pool, Tensor v_chan_scale, "
                                "Tensor qo_indptr, "
                                "Tensor paged_kv_indices, "
                                "Tensor paged_kv_indptr, "
                                "Tensor paged_kv_last_page_len, "
                                "Tensor work_units, "
                                "int total_work, bool causal, "
                                "float softmax_scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(fish_scales_ops, CUDA, m) {
    m.impl("mxfp8_attn_fwd", &fish_scales_ops_attn::mxfp8_attn_fwd);
    m.impl("mxfp8_decode_paged", &fish_scales_ops_attn::mxfp8_decode_paged);
    m.impl("mxfp8_attn_fwd_paged", &fish_scales_ops_attn::mxfp8_attn_fwd_paged);
}

// The combined extension needs at least one PYBIND11_MODULE definition for
// the .so to load as a Python module. The torch.ops registrations above
// handle the public surface; this block is intentionally doc-only.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "fish_scales_ops C++ extension (GEMM + MXFP8 attention).";
}
