// MXFP8 FlashAttention forward — top-level dispatch.
//
// The kernel implementation lives in `mxfp8_attn_fwd_impl.cuh` and is
// instantiated by per-D translation units
// (`mxfp8_attn_fwd_d{32,64,128,256}.cu`).
// This file only contains the runtime dispatch that forwards into the
// per-D entry points. Each per-D TU compiles independently — ptxas sees
// one D's template instances at a time, so register allocation for D=32
// can no longer disturb D=128's codegen state (the wall the v16-class
// rewrite attempts kept hitting; see docs/design/attn_v16_rewrite.md).
//
// The production impl includes the v18 (Ks-scale hoist) + v19 (P-quant
// pack via cvt.e4m3x2.f32 + STS.U16) micro-opts merged 2026-05-26.
// Earlier negative-result variants (v16/v17/v20/v21pre/v21) and probe
// instrumentation live in `experiments/`; the env-var routing that used
// to dispatch into them has been removed.

#include <cuda_runtime.h>

extern "C" cudaError_t mxfp8_attn_fwd_launch_d32(
    const void* Q, const void* Qs,
    const void* K, const void* Ks,
    const void* V, const void* Vs,
    void* O,
    int batch, int num_q_heads, int num_kv_heads,
    int seq_q, int seq_k,
    float softmax_scale, int causal,
    cudaStream_t stream);

extern "C" cudaError_t mxfp8_attn_fwd_launch_d64(
    const void* Q, const void* Qs,
    const void* K, const void* Ks,
    const void* V, const void* Vs,
    void* O,
    int batch, int num_q_heads, int num_kv_heads,
    int seq_q, int seq_k,
    float softmax_scale, int causal,
    cudaStream_t stream);

extern "C" cudaError_t mxfp8_attn_fwd_launch_d128(
    const void* Q, const void* Qs,
    const void* K, const void* Ks,
    const void* V, const void* Vs,
    void* O,
    int batch, int num_q_heads, int num_kv_heads,
    int seq_q, int seq_k,
    float softmax_scale, int causal,
    cudaStream_t stream);

extern "C" cudaError_t mxfp8_attn_fwd_launch_d256(
    const void* Q, const void* Qs,
    const void* K, const void* Ks,
    const void* V, const void* Vs,
    void* O,
    int batch, int num_q_heads, int num_kv_heads,
    int seq_q, int seq_k,
    float softmax_scale, int causal,
    cudaStream_t stream);

extern "C" cudaError_t mxfp8_attn_fwd_launch(
    const void* Q, const void* Qs,
    const void* K, const void* Ks,
    const void* V, const void* Vs,
    void* O,
    int batch, int num_q_heads, int num_kv_heads,
    int seq_q, int seq_k, int head_dim,
    float softmax_scale, int causal,
    cudaStream_t stream)
{
    // SM120 BlockScaled mxf8f6f4 mma is not encodable on earlier arches.
    // The per-D translation units still build for sm_90a (with an empty
    // kernel-body stub) so the combined extension links cleanly; this
    // dispatcher just refuses to launch on anything below sm_120.
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return cudaErrorNotSupported;
    int sm_major = 0;
    if (cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)
        return cudaErrorNotSupported;
    if (sm_major < 12) return cudaErrorNotSupported;

    switch (head_dim) {
        case 32:
            return mxfp8_attn_fwd_launch_d32(
                Q, Qs, K, Ks, V, Vs, O,
                batch, num_q_heads, num_kv_heads, seq_q, seq_k,
                softmax_scale, causal, stream);
        case 64:
            return mxfp8_attn_fwd_launch_d64(
                Q, Qs, K, Ks, V, Vs, O,
                batch, num_q_heads, num_kv_heads, seq_q, seq_k,
                softmax_scale, causal, stream);
        case 128:
            return mxfp8_attn_fwd_launch_d128(
                Q, Qs, K, Ks, V, Vs, O,
                batch, num_q_heads, num_kv_heads, seq_q, seq_k,
                softmax_scale, causal, stream);
        case 256:
            return mxfp8_attn_fwd_launch_d256(
                Q, Qs, K, Ks, V, Vs, O,
                batch, num_q_heads, num_kv_heads, seq_q, seq_k,
                softmax_scale, causal, stream);
        default:
            return cudaErrorNotSupported;
    }
}
