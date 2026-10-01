#!/usr/bin/env python3
"""Skipped expert ids through moe_layer_mxfp8_sm120. A serving engine hands this
layer two kinds of entry that name no expert of this rank. The rows past
num_token_non_padded of a CUDA-graph or piecewise-graph bucket carry the id
num_experts (sglang's moe_align overflow slot) or -1, and under expert
parallelism the dispatcher rewrites every expert another rank owns to -1, which
happens on real tokens and on individual top-k entries rather than on whole
trailing rows. Both must take no expert compute, leave every other row bit
identical to the call without them, and give a fully skipped token an all-zero
row.

Four cases per cell: trailing fills with num_experts and with -1 at several real
row counts; an expert-parallel draw, where ids come from the full expert set and
everything this rank does not own becomes -1, checked against a reference that
keeps those entries but zeroes their combine weight (the same value by
construction, since the grouped GEMM is row-independent and a zero-weight term
adds exactly zero); and a graph captured with one real-row count and replayed
with another. The sm_120 twin of test_moe_layer_padded_ids_sm90.py, which was
written after the sm_90 sorted builder wrote through a garbage row on the first
padded prefill bucket of a serving run; the masked-layout builders this layer
uses carried the same defect until the sm_120 layer was added."""
import sys
import torch
import fish_scales_ops as fso

FAMILIES = {
    # label: (E_local, topk, hidden, inter, E_global)
    "C_35a3_ep2": (128, 8, 2048, 512, 256),  # the drama artifact at tp2/ep2
    "B_30a3": (128, 8, 2048, 768, 128),
}


def substitute_distinct(ids_global, local, e):
    """Replace every non-local entry with a local expert the token does not
    already route to, keeping each token's ids distinct (see case 3)."""
    rows = ids_global.cpu().tolist()
    keep = local.cpu().tolist()
    for t, row in enumerate(rows):
        used = {row[j] for j in range(len(row)) if keep[t][j]}
        nxt = 0
        for j in range(len(row)):
            if not keep[t][j]:
                while nxt in used:
                    nxt += 1
                used.add(nxt)
                row[j] = nxt
    return torch.tensor(rows, dtype=torch.int32, device=ids_global.device)


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        where = "no CUDA device" if not torch.cuda.is_available() \
            else f"device is sm_{torch.cuda.get_device_capability()[0]}x"
        print(f"SKIP: test_moe_layer_padded_ids_sm120 is sm_120/121 (RTX 5090) only ({where})")
        return 0
    torch.manual_seed(0)
    dev = "cuda"
    for fam, (E, TOPK, HIDDEN, INTER, E_GLOBAL) in FAMILIES.items():
        w13 = torch.randn(E, 2 * INTER, HIDDEN, device=dev, dtype=torch.bfloat16) * 0.02
        w2 = torch.randn(E, HIDDEN, INTER, device=dev, dtype=torch.bfloat16) * 0.02
        w13f, sw13 = fso.compat.quantize_moe_weights_1x32_fp8(w13)
        w2f, sw2 = fso.compat.quantize_moe_weights_1x32_fp8(w2)
        del w13, w2
        layer = lambda h, ids, w: fso.compat.moe_layer_mxfp8_sm120(
            h, w13f, sw13, w2f, sw2, ids, w)
        for M in (1, 8, 64, 512, 1024):
            g = torch.Generator(device=dev)
            g.manual_seed(1234 + M)
            hidden = torch.randn(M, HIDDEN, device=dev, dtype=torch.bfloat16, generator=g)
            ids = torch.rand(M, E, device=dev, generator=g).topk(TOPK, dim=1).indices.to(torch.int32)
            topk_w = torch.softmax(torch.rand(M, TOPK, device=dev, generator=g), dim=1).float()
            ref = layer(hidden, ids, topk_w)
            torch.cuda.synchronize()
            # 1 + 2: trailing padding rows, both fills the engine uses.
            for fill in (E, -1):
                for n_real in (max(1, M // 2), 1, M - 1, 0):
                    masked = ids.clone()
                    masked[n_real:] = fill
                    out = layer(hidden, masked, topk_w)
                    torch.cuda.synchronize()
                    assert torch.equal(out[:n_real], ref[:n_real]), \
                        f"{fam} M={M} fill={fill} n_real={n_real}: real rows differ from the unmasked call"
                    assert bool((out[n_real:] == 0).all()), \
                        f"{fam} M={M} fill={fill} n_real={n_real}: masked rows are not zero"
            # 3: an expert-parallel draw. Ids come from the whole model's expert
            # set; this rank owns [0, E), everything else becomes -1. The
            # reference replaces each -1 with a local expert this token does not
            # already route to and gives it combine weight zero: the surviving
            # entries then produce the same rows (the grouped GEMM treats a
            # group's rows independently) and the substituted ones contribute an
            # exact zero to the sum, so the two calls must agree bit for bit. The
            # substitutes have to be distinct within a token because the layer
            # sizes each expert's capacity at one row per token; two entries of
            # one token naming the same expert would overflow that capacity,
            # which is the caller contract of moe_build_routing rather than
            # something the layer can absorb.
            ep_ids_global = torch.rand(M, E_GLOBAL, device=dev, generator=g).topk(TOPK, dim=1).indices
            local = ep_ids_global < E
            ep_ids = torch.where(local, ep_ids_global, torch.full_like(ep_ids_global, -1)).to(torch.int32)
            ep_w = torch.softmax(torch.rand(M, TOPK, device=dev, generator=g), dim=1).float()
            ep_out = layer(hidden, ep_ids, ep_w)
            ep_ref = layer(hidden, substitute_distinct(ep_ids_global, local, E),
                           torch.where(local, ep_w, torch.zeros_like(ep_w)))
            torch.cuda.synchronize()
            n_dead = int((~local).all(dim=1).sum())
            assert torch.equal(ep_out, ep_ref), \
                f"{fam} M={M}: expert-parallel draw differs from the zero-weight reference"
            if n_dead:
                dead = (~local).all(dim=1)
                assert bool((ep_out[dead] == 0).all()), \
                    f"{fam} M={M}: a token with no local expert did not get a zero row"
            # 4: capture with M//2 real rows, replay with M-1 real rows written
            # into the same id buffer.
            ids_buf = ids.clone()
            ids_buf[M // 2:] = E
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    layer(hidden, ids_buf, topk_w)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=s):
                out_g = layer(hidden, ids_buf, topk_w)
            torch.cuda.synchronize()
            ids_buf.copy_(ids)
            ids_buf[M - 1:] = -1
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out_g[:M - 1], ref[:M - 1]), f"{fam} M={M}: graph replay real rows differ"
            assert bool((out_g[M - 1:] == 0).all()), f"{fam} M={M}: graph replay masked row is not zero"
            print(f"{fam:12s} M={M:5d}: fills ({E}, -1) x real rows (M/2, 1, M-1, 0) bit-identical and "
                  f"masked rows zero; EP draw ({n_dead} fully remote tokens) == zero-weight reference; "
                  f"graph replay with a different real-row count == eager  OK")
    print("\nmoe_layer_mxfp8_sm120 skipped expert ids: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
