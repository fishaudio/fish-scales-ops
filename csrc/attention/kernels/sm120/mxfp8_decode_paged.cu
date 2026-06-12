// MXFP8 paged-KV decode forward — top-level dispatch.
//
// Mirrors the prefill `mxfp8_attn_fwd.cu` shape: switches on head_dim and
// forwards to the per-D translation unit. Per-D launch ABI is identical to
// the per-D `extern "C"` symbols emitted by
// mxfp8_decode_paged_d{32,64,128,256}.cu.
//
// At runtime, refuses to launch on sm < 12.0 (the kernels' inline PTX
// mma.sync.kind::mxf8f6f4 + CUTLASS SM120 atoms are sm_120a-specific; the
// kernel body is `#if __CUDA_ARCH__ >= 1200`-guarded so sm_90a SASS pass
// emits an empty stub).

#include <cuda_runtime.h>

extern "C" cudaError_t mxfp8_decode_paged_launch_d32(
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
    cudaStream_t stream);

extern "C" cudaError_t mxfp8_decode_paged_launch_d64(
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
    cudaStream_t stream);

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
    cudaStream_t stream);

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
    cudaStream_t stream);

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
    cudaStream_t stream)
{
    // sm_120-only at runtime.
    int dev = -1; cudaGetDevice(&dev);
    int sm_major = -1;
    cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, dev);
    if (sm_major < 12)
        return cudaErrorNotSupported;

    switch (head_dim) {
        case 32:
            return mxfp8_decode_paged_launch_d32(
                Q, Qs, K_pool, K_chan_scale, V_pool, V_chan_scale,
                block_table, seq_lens, O,
                M_partial, L_partial, O_partial, sync_counter,
                batch, num_q_heads, num_kv_heads, max_blocks, num_splits, target_counter,
                page_size, softmax_scale, stream);
        case 64:
            return mxfp8_decode_paged_launch_d64(
                Q, Qs, K_pool, K_chan_scale, V_pool, V_chan_scale,
                block_table, seq_lens, O,
                M_partial, L_partial, O_partial, sync_counter,
                batch, num_q_heads, num_kv_heads, max_blocks, num_splits, target_counter,
                page_size, softmax_scale, stream);
        case 128:
            return mxfp8_decode_paged_launch_d128(
                Q, Qs, K_pool, K_chan_scale, V_pool, V_chan_scale,
                block_table, seq_lens, O,
                M_partial, L_partial, O_partial, sync_counter,
                batch, num_q_heads, num_kv_heads, max_blocks, num_splits, target_counter,
                page_size, softmax_scale, stream);
        case 256:
            return mxfp8_decode_paged_launch_d256(
                Q, Qs, K_pool, K_chan_scale, V_pool, V_chan_scale,
                block_table, seq_lens, O,
                M_partial, L_partial, O_partial, sync_counter,
                batch, num_q_heads, num_kv_heads, max_blocks, num_splits, target_counter,
                page_size, softmax_scale, stream);
        default:
            return cudaErrorNotSupported;
    }
}
