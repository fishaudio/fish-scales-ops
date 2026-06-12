// Paged-prefill (extend mode) MXFP8 attention forward — top-level dispatch.
//
// Routes by head_dim into per-D launchers compiled in mxfp8_attn_fwd_paged_d{32,64,128,256}.cu.
// SM120-only at runtime; below sm_120 the dispatcher returns cudaErrorNotSupported
// (the per-D TUs still build for sm_90a with an empty kernel-body stub).

#include <cuda_runtime.h>

extern "C" cudaError_t mxfp8_attn_fwd_paged_launch_d32(
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
    int page_size, int causal,
    float softmax_scale,
    cudaStream_t stream);

extern "C" cudaError_t mxfp8_attn_fwd_paged_launch_d64(
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
    int page_size, int causal,
    float softmax_scale,
    cudaStream_t stream);

extern "C" cudaError_t mxfp8_attn_fwd_paged_launch_d128(
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
    int page_size, int causal,
    float softmax_scale,
    cudaStream_t stream);

extern "C" cudaError_t mxfp8_attn_fwd_paged_launch_d256(
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
    int page_size, int causal,
    float softmax_scale,
    cudaStream_t stream);

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
    cudaStream_t stream)
{
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return cudaErrorNotSupported;
    int sm_major = 0;
    if (cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)
        return cudaErrorNotSupported;
    if (sm_major < 12) return cudaErrorNotSupported;

    switch (head_dim) {
        case 32:
            return mxfp8_attn_fwd_paged_launch_d32(
                Q, Qs, K_pool, K_chan_scale, V_pool, V_chan_scale,
                qo_indptr, paged_kv_indices, paged_kv_indptr,
                paged_kv_last_page_len, work_units, O,
                total_work, num_q_heads, num_kv_heads,
                page_size, causal, softmax_scale, stream);
        case 64:
            return mxfp8_attn_fwd_paged_launch_d64(
                Q, Qs, K_pool, K_chan_scale, V_pool, V_chan_scale,
                qo_indptr, paged_kv_indices, paged_kv_indptr,
                paged_kv_last_page_len, work_units, O,
                total_work, num_q_heads, num_kv_heads,
                page_size, causal, softmax_scale, stream);
        case 128:
            return mxfp8_attn_fwd_paged_launch_d128(
                Q, Qs, K_pool, K_chan_scale, V_pool, V_chan_scale,
                qo_indptr, paged_kv_indices, paged_kv_indptr,
                paged_kv_last_page_len, work_units, O,
                total_work, num_q_heads, num_kv_heads,
                page_size, causal, softmax_scale, stream);
        case 256:
            return mxfp8_attn_fwd_paged_launch_d256(
                Q, Qs, K_pool, K_chan_scale, V_pool, V_chan_scale,
                qo_indptr, paged_kv_indices, paged_kv_indptr,
                paged_kv_last_page_len, work_units, O,
                total_work, num_q_heads, num_kv_heads,
                page_size, causal, softmax_scale, stream);
        default:
            return cudaErrorNotSupported;
    }
}
