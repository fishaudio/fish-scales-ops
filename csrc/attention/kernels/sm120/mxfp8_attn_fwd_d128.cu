// Per-D translation unit, kHeadDim = 128.
//
// Independent ptxas pass — see mxfp8_attn_fwd_d32.cu header for why.
// This is the *production headline* D — Qwen3 / LLaMA-3 / Qwen2-72B all
// run here. R5 codegen state (REG 232/238, 0 spill) is the regression
// contract for this instance; any change that drifts it must be matched
// by a bench win.
//
// Per-D tile config (preserved from prior tuning):
//   Br = 64, Bc = 64, kStages = 2, kCtasPerSm = 2  (smem 36 KB)
//
// Br=128 was tried in v16 step-2 (smem 40 KB → 1 CTA/SM trap, -10 to -23%);
// production stays at Br=64.

#include "mxfp8_attn_fwd_impl.cuh"

extern "C" cudaError_t mxfp8_attn_fwd_launch_d128(
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
    // D = 128 : Br=64, Bc=64, S=2, 2 CTA/SM
    if (causal) {
        return launch_impl<128, 64, 64, 2, 2, true>(
            Q, Qs, K, Ks, V, Vs, O,
            batch, num_q_heads, num_kv_heads, seq_q, seq_k, softmax_scale, stream);
    } else {
        return launch_impl<128, 64, 64, 2, 2, false>(
            Q, Qs, K, Ks, V, Vs, O,
            batch, num_q_heads, num_kv_heads, seq_q, seq_k, softmax_scale, stream);
    }
}
