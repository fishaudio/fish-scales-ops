// Per-D translation unit, kHeadDim = 64.
//
// Independent ptxas pass — see mxfp8_attn_fwd_d32.cu header for why.
//
// Per-D tile config (preserved from prior tuningstep-2):
//   Br = 64, Bc = 64, kStages = 2, kCtasPerSm = 2  (smem 20 KB)
//
// v16 P2 attempted Bc=128 for non-causal here (+1.4% NC, -14% causal) and
// reverted because the change couldn't trade across kIsCausal cleanly.
// With per-D split, a follow-up can re-attempt per-(D, causal) tile picks
// without disturbing D=128 / D=32 codegen — but production stays on
// Bc=64 for now (cuobjdump regression contract: REG 219 causal / 225 NC).

#include "mxfp8_attn_fwd_impl.cuh"

extern "C" cudaError_t mxfp8_attn_fwd_launch_d64(
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
    // D = 64 : Br=64, Bc=64, S=2, 2 CTA/SM
    if (causal) {
        return launch_impl<64, 64, 64, 2, 2, true>(
            Q, Qs, K, Ks, V, Vs, O,
            batch, num_q_heads, num_kv_heads, seq_q, seq_k, softmax_scale, stream);
    } else {
        return launch_impl<64, 64, 64, 2, 2, false>(
            Q, Qs, K, Ks, V, Vs, O,
            batch, num_q_heads, num_kv_heads, seq_q, seq_k, softmax_scale, stream);
    }
}
