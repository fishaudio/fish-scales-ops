#!/usr/bin/env python3
"""Minimal CUDA-Graph nsys profile driver comparing fso v0.3.0 vs drama3_unified on s3-drama."""
import argparse
import torch
import torch.nn.functional as F
import fish_scales_ops as fso
from fish_scales_ops import drama3_unified as d3u

E, TOPK, H, I = 256, 8, 2048, 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fso030", "unified"], required=True)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--live", type=int, default=0, help=">0 for padded bucket (M, live)")
    args = ap.parse_args()

    M = args.M
    torch.manual_seed(42)
    h = (torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
    a_out = (torch.randn(M, 4096, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()

    if args.live > 0:
        ids = torch.full((M, TOPK), -1, device="cuda", dtype=torch.int32)
        tw = torch.zeros((M, TOPK), device="cuda", dtype=torch.float32)
        ids[:args.live] = torch.rand(args.live, E, device="cuda").topk(TOPK, dim=-1).indices.to(torch.int32)
        tw[:args.live] = torch.softmax(torch.randn(args.live, TOPK, device="cuda"), dim=-1).to(torch.float32)
    else:
        ids = torch.rand(M, E, device="cuda").topk(TOPK, dim=-1).indices.to(torch.int32).contiguous()
        tw = torch.softmax(torch.randn(M, TOPK, device="cuda"), dim=-1).to(torch.float32).contiguous()

    w_in = (torch.randn(12288, H, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    w_out = (torch.randn(H, 4096, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    w13 = (torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    w2 = (torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    sw13 = (torch.randn(2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    sw2 = (torch.randn(H, I, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()

    if args.mode == "fso030":
        f_in = fso.dense.prepare_weight(w_in, format="mxfp8")
        f_out = fso.dense.prepare_weight(w_out, format="mxfp8")
        f_moe = fso.moe.prepare_experts(w13, w2, format="mxfp8")
        f_s1 = fso.dense.prepare_weight(sw13, format="mxfp8")
        f_s2 = fso.dense.prepare_weight(sw2, format="mxfp8")

        def step():
            _ = fso.dense.linear(h, f_in)
            _ = fso.dense.linear(a_out, f_out)
            gu = fso.dense.linear(h, f_s1)
            g, u = gu.chunk(2, dim=-1)
            ys = fso.dense.linear(F.silu(g) * u, f_s2)
            return fso.moe.layer(h, f_moe, ids, tw, bias=ys)
    else:
        u_in = d3u.prepare_dense_weight_unified(w_in)
        u_out = d3u.prepare_dense_weight_unified(w_out)
        u_moe = d3u.prepare_moe_unified(w13, w2, shared_w13=sw13, shared_w2=sw2)

        def step():
            _ = d3u.linear_unified(h, *u_in)
            _ = d3u.linear_unified(a_out, *u_out)
            return d3u.moe_layer_unified(h, u_moe, ids, tw)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        step()
        with torch.cuda.graph(g, stream=s):
            step()
    torch.cuda.synchronize()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
