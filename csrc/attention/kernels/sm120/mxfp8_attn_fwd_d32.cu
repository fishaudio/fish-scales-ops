// Per-D translation unit, kHeadDim = 32.
//
// Independent ptxas pass: only the (D=32, causal) and (D=32, non-causal)
// kernel template instances are emitted here, so per-instance register
// allocation does not share state with D=64 / D=128. This is the structural
// fix for the v16 R6/R9/R10/P3 wall — algorithmic patches that helped D=32
// regressed D=128 because the same ptxas pass had to schedule both.
//
// Per-D tile config (preserved from prior tuningP1):
//   Br = 64, Bc = 128, kStages = 2, kCtasPerSm = 2  (smem 24 KB)
//
// The v16 P1 `if constexpr (kHeadDim == 32)` mask-hoist branch in the
// kernel body fires here, so cuobjdump for D=32 causal stays at 255+24
// stack spill (bench wins +5-7% over R5 baseline anyway — algorithmic
// save beats the spill cost on small D).

#include "mxfp8_attn_fwd_impl.cuh"

extern "C" cudaError_t mxfp8_attn_fwd_launch_d32(
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
    // D = 32 : Br=64, Bc=128, S=2, 2 CTA/SM
    if (causal) {
        return launch_impl<32, 64, 128, 2, 2, true>(
            Q, Qs, K, Ks, V, Vs, O,
            batch, num_q_heads, num_kv_heads, seq_q, seq_k, softmax_scale, stream);
    } else {
        return launch_impl<32, 64, 128, 2, 2, false>(
            Q, Qs, K, Ks, V, Vs, O,
            batch, num_q_heads, num_kv_heads, seq_q, seq_k, softmax_scale, stream);
    }
}
