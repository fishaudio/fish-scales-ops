// v3 per-D TU for kHeadDim = 256. Single-buffer (kStages=1) — the 4 per-warp
// slabs at stages=2 would be 130 KB > 99 KB cap; at stages=1 they fit in 66 KB.
// cp.async overlap is lost but decode is memory-bound at D=256 anyway.
// The kStages=1 dependency pattern (load at end-of-iter, wait at top-of-iter)
// lives behind an `if constexpr (kStages == 1)` branch in the impl.
#include "mxfp8_decode_paged_impl.cuh"

extern "C" cudaError_t mxfp8_decode_paged_launch_d256(
    const void* Q, const void* Qs,
    const void* K_pool, const void* K_chan_scale,
    const void* V_pool, const void* V_chan_scale,
    const void* block_table, const void* seq_lens,
    void* O,
    void* M_partial, void* L_partial, void* O_partial,
    void* sync_counter,
    int batch, int num_q_heads, int num_kv_heads,
    int max_blocks, int num_splits, int target_counter,
    int page_size,
    float softmax_scale,
    cudaStream_t stream)
{
    using namespace flash_attn_sm120_decode_paged;
    return launch_decode_paged_impl<256, 32, 1, 1>(
        Q, Qs, K_pool, K_chan_scale, V_pool, V_chan_scale,
        block_table, seq_lens, O,
        M_partial, L_partial, O_partial, sync_counter,
        batch, num_q_heads, num_kv_heads, max_blocks, num_splits, target_counter,
        page_size, softmax_scale, stream);
}
