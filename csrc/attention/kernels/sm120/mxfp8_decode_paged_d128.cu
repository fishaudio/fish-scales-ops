// v3 per-D TU for kHeadDim = 128. page_size is a runtime parameter;
// kBc=32 so page_bc_ratio = page_size / 32.
// smem per warp: K(32×128×2=8KB) + V(128×32×2=8KB) + P(16×32=0.5KB) + scales ≈ 16.5 KB
// 4 warps × 16.5 = 66 KB. At kCtasPerSm=1 (smem cap), 1 CTA/SM.
#include "mxfp8_decode_paged_impl.cuh"

extern "C" cudaError_t mxfp8_decode_paged_launch_d128(
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
    return launch_decode_paged_impl<128, 32, 2, 1>(
        Q, Qs, K_pool, K_chan_scale, V_pool, V_chan_scale,
        block_table, seq_lens, O,
        M_partial, L_partial, O_partial, sync_counter,
        batch, num_q_heads, num_kv_heads, max_blocks, num_splits, target_counter,
        page_size, softmax_scale, stream);
}
