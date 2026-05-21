// Per-D translation unit, kHeadDim = 256.
//
// Per-D tile config: Br=64, Bc=32, kStages=2, kCtasPerSm=1 (smem ~34 KB).
// Bc=32 (not 64) because pv_acc[D/8][4] = 128 fp32 already pushes reg/thread
// past 256 → 1 CTA/SM whether smem fits or not; the smaller Bc preserves the
// kStages=2 cp.async overlap that disappears at Bc=64 + 1 CTA.
// SwizzledKAtom<256> aliases Layout_K_SW128_Atom (tile_to_shape replicates).

#include "mxfp8_attn_fwd_impl.cuh"

extern "C" cudaError_t mxfp8_attn_fwd_launch_d256(
    const void* Q, const void* Qs,
    const void* K, const void* Ks,
    const void* V, const void* Vs,
    void* O,
    int batch, int num_q_heads, int num_kv_heads,
    int seq_q, int seq_k,
    float softmax_scale, int causal,
    cudaStream_t stream)
{
    using namespace flash_attn_sm120;
    if (causal) {
        return launch_impl<256, 64, 32, 2, 1, true>(
            Q, Qs, K, Ks, V, Vs, O,
            batch, num_q_heads, num_kv_heads, seq_q, seq_k, softmax_scale, stream);
    } else {
        return launch_impl<256, 64, 32, 2, 1, false>(
            Q, Qs, K, Ks, V, Vs, O,
            batch, num_q_heads, num_kv_heads, seq_q, seq_k, softmax_scale, stream);
    }
}
