// Paged-prefill per-D TU for kHeadDim = 256.
// Per-D tile config: Br=64, Bc=32, kStages=2, kCtasPerSm=1 (smem ~34 KB).
// 1 CTA/SM is forced by register pressure (n_pv_n_tiles=32) regardless of smem.
#include "mxfp8_attn_fwd_paged_impl.cuh"

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
    cudaStream_t stream)
{
    using namespace flash_attn_sm120_paged_prefill;
    if (causal) {
        return launch_paged_prefill_impl<256, 64, 32, 2, 1, true>(
            Q, Qs, K_pool, K_chan_scale, V_pool, V_chan_scale,
            qo_indptr, paged_kv_indices, paged_kv_indptr,
            paged_kv_last_page_len, work_units, O,
            total_work, num_q_heads, num_kv_heads, page_size,
            softmax_scale, stream);
    } else {
        return launch_paged_prefill_impl<256, 64, 32, 2, 1, false>(
            Q, Qs, K_pool, K_chan_scale, V_pool, V_chan_scale,
            qo_indptr, paged_kv_indices, paged_kv_indptr,
            paged_kv_last_page_len, work_units, O,
            total_work, num_q_heads, num_kv_heads, page_size,
            softmax_scale, stream);
    }
}
