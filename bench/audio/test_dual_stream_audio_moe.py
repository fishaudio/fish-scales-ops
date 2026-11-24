"""Dual-stream concurrency stress test: Full Audio Tower (3x Conv2d Stem + 32L Encoder + Tail)
CUDA Graph on Stream A running concurrently with LLM MXFP8 MoE layer (`fso.moe.layer`) on Stream B
from a single host thread.

Verifies:
  - 0 bitwise mismatches on Full Audio Tower outputs over 200 concurrent iterations
  - 0 bitwise mismatches on MoE layer outputs over 200 concurrent iterations
  - Capture-safety on non-default streams & zero shared cross-stream scratch hazards
  - Zero mutation of global dec.WF_PRE_WAIT_STAGES_* (stays 2/1)
"""
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import fish_scales_ops as fso
import fish_scales_ops.audio as audio
import fish_scales_ops.gemm._sm100_decode as dec
from bench_stem_and_e2e import make_tower_weights, make_stem_weights


def run_full_audio_tower(mel_nhwc, pos, S, W, dw_co_perm):
    (w1, b1), (w2, b2), (w3, b3) = S["nhwc"]
    h = audio.gelu_erf_bf16_pdl(F.conv2d(mel_nhwc, w1, b1, stride=2, padding=1))
    h = audio.gelu_erf_bf16_pdl(F.conv2d(h, w2, b2, stride=2, padding=1))
    h = F.conv2d(h, w3, b3, stride=2, padding=1)
    xc_q, xc_s = audio.gelu_nhwc_to_mxfp8(h)
    x = audio.linear_audio_conv_out(xc_q, dw_co_perm, xc_s, residual=pos)
    M, _ = x.shape
    for L in W["layers"]:
        hq, hs = audio.layernorm_mxfp8(x, L["ln1"][0], L["ln1"][1])
        qkv = audio.linear_audio_qkv(hq, L["dqkv"], hs, bias=L["qkv"][1])
        q, k, v = qkv.view(M, 3, 20, 64).unbind(1)
        aq, as_ = audio.attn_fa4_mxfp8(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
        x = audio.linear_audio_out_proj(aq, L["do"], as_, bias=L["o"][1], residual=x)
        h2q, h2s = audio.layernorm_mxfp8(x, L["ln2"][0], L["ln2"][1])
        uq, us = fso.dense.linear_quantized(h2q, L["dfc1"], x_scale=h2s, bias=L["fc1"][1], activation="gelu_erf")
        x = audio.linear_audio_fc2(uq, L["dfc2"], us, bias=L["fc2"][1], residual=x)
    xq, xs = audio.layernorm_mxfp8(x, W["ln_post"][0], W["ln_post"][1])
    xq2, xs2 = fso.dense.linear_quantized(xq, W["p1"][2], x_scale=xs, bias=W["p1"][1], activation="gelu_erf")
    return audio.linear_audio_proj2(xq2, W["p2"][2], xs2, bias=W["p2"][1])


def main():
    torch.backends.cudnn.benchmark = True
    assert dec.WF_PRE_WAIT_STAGES_ONE_CTA == 2 and dec.WF_PRE_WAIT_STAGES_TWO_CTA == 1
    torch.manual_seed(12345)

    S = make_stem_weights()
    W = make_tower_weights(n_layers=32)
    dw_co_perm = audio.prepare_conv_out_weight(W["conv"][0], channels=480, freq=16)

    # LLM MXFP8 MoE layer (B300 fused slot-MoE path, G=64 experts, topk=8, H=2048, I=768)
    G, topk, H, I = 64, 8, 2048, 768
    w13 = (torch.randn(G, 2 * I, H, device="cuda", dtype=torch.bfloat16) / math.sqrt(H)).contiguous()
    w2 = (torch.randn(G, H, I, device="cuda", dtype=torch.bfloat16) / math.sqrt(I)).contiguous()
    experts = fso.moe.prepare_experts(w13, w2, format="mxfp8")

    for M_audio, M_moe in [(182, 16), (533, 128), (1079, 64)]:
        B = M_audio // 13
        mel_nhwc = (torch.randn(B, 1, 128, 100, device="cuda", dtype=torch.bfloat16) * 0.5).to(memory_format=torch.channels_last)
        pos = (torch.randn(M_audio, 1280, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
        x_moe = (torch.randn(M_moe, H, device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
        logits = torch.randn(M_moe, G, device="cuda", dtype=torch.float32)
        topk_w, topk_ids = torch.topk(logits, topk, dim=-1)
        topk_w = torch.softmax(topk_w, dim=-1).to(torch.float32)
        topk_ids = topk_ids.to(torch.int32)

        # Warm up cuDNN Conv2d selection + CuTe-DSL JIT before golden ref
        for _ in range(2):
            _ = run_full_audio_tower(mel_nhwc, pos, S, W, dw_co_perm)
            _ = fso.moe.layer(x_moe, experts, topk_ids, topk_w)
        torch.cuda.synchronize()
        ref_audio = run_full_audio_tower(mel_nhwc, pos, S, W, dw_co_perm).clone()
        ref_moe = fso.moe.layer(x_moe, experts, topk_ids, topk_w).clone()
        torch.cuda.synchronize()

        s_audio = torch.cuda.Stream()
        s_moe = torch.cuda.Stream()

        with torch.cuda.stream(s_audio):
            for _ in range(2):
                run_full_audio_tower(mel_nhwc, pos, S, W, dw_co_perm)
        s_audio.synchronize()
        g_audio = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_audio, stream=s_audio):
            out_audio = run_full_audio_tower(mel_nhwc, pos, S, W, dw_co_perm)
        s_audio.synchronize()

        with torch.cuda.stream(s_moe):
            for _ in range(2):
                fso.moe.layer(x_moe, experts, topk_ids, topk_w)
        s_moe.synchronize()
        g_moe = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_moe, stream=s_moe):
            out_moe = fso.moe.layer(x_moe, experts, topk_ids, topk_w)
        s_moe.synchronize()

        audio_mismatches = 0
        moe_mismatches = 0
        for _ in range(200):
            out_audio.fill_(float("nan"))
            out_moe.fill_(float("nan"))
            torch.cuda.synchronize()
            with torch.cuda.stream(s_audio):
                g_audio.replay()
            with torch.cuda.stream(s_moe):
                g_moe.replay()
            torch.cuda.synchronize()
            if not torch.equal(out_audio.view(torch.uint16), ref_audio.view(torch.uint16)):
                audio_mismatches += 1
            if not torch.equal(out_moe.view(torch.uint16), ref_moe.view(torch.uint16)):
                moe_mismatches += 1

        print(
            f"  Dual-stream stress (Full Stem+32L M_audio={M_audio}, MoE M_moe={M_moe}, 200 iters): "
            f"audio_mismatches={audio_mismatches}, moe_mismatches={moe_mismatches}",
            flush=True,
        )
        assert audio_mismatches == 0 and moe_mismatches == 0, "Dual-stream mismatch detected!"

    assert dec.WF_PRE_WAIT_STAGES_ONE_CTA == 2 and dec.WF_PRE_WAIT_STAGES_TWO_CTA == 1
    print("ALL DUAL-STREAM CONCURRENCY STRESS TESTS PASSED (0 MISMATCHES)!", flush=True)


if __name__ == "__main__":
    main()
