#!/usr/bin/env python3
"""The composed sm_90 MoE layer entry moe_layer_fp8_sm90 with its internal
dispatch (swap-AB with block_n = 16 / 32 chosen from the routed rows per active
expert, the non-swap contiguous block_m=64 path above 24 rows per expert, whose
FC2 runs the swap-AB GEMM with block_n 64 on the same 64-row layout).
Validates correctness at every cascade step and across the last threshold,
CUDA-graph capture safety for a decode M and a prefill M, and, in both expert
shapes, that the FC2 route of the non-swap path (the default, FSO_FC2_SWAP=0
and FSO_FC2_SWAP=1) leaves the layer output bit-identical."""
import os, sys, math
import torch
import fish_scales_ops as fso
from fish_scales_ops.gemm import fp8 as gemm_fp8  # the implementation module, for the private FC2 rule

E, TOPK, HIDDEN, INTER = 128, 8, 2048, 768


def moe_ref(hidden, w13, w2, topk_ids, topk_w):
    M = hidden.shape[0]
    out = torch.zeros(M, HIDDEN, device="cuda", dtype=torch.float32)
    for t in range(M):
        for j in range(TOPK):
            e = int(topk_ids[t, j])
            gu = hidden[t].float() @ w13[e].T.float()
            g, u = gu.chunk(2, dim=-1)
            out[t] += float(topk_w[t, j]) * ((torch.nn.functional.silu(g) * u) @ w2[e].T.float())
    return out.to(torch.bfloat16)


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


# Both measured expert shapes of the sm_90 retune: (E, top-k, HIDDEN, INTER).
FC2_FAMILIES = {"E128_I768": (128, 8, 2048, 768), "E256_I512": (256, 8, 2048, 512)}


def fc2_route_equality():
    """At the first non-swap M of each expert shape, the default dispatch, FSO_FC2_SWAP=0 (non-swap FC2) and
    FSO_FC2_SWAP=1 (swap-AB FC2 with block_n 64) must give bit-identical layer outputs. The knob is read on every
    call; the check asserts that it does switch the FC2 route, so the equality is not vacuous."""
    print("\n=== non-swap path: FC2 route default / FSO_FC2_SWAP=0 / =1, bit-identical layer outputs ===")
    saved = os.environ.get("FSO_FC2_SWAP")
    try:
        for name, (E, TOPK, H, I) in FC2_FAMILIES.items():
            torch.manual_seed(11)
            w13 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) / math.sqrt(H)
            w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) / math.sqrt(I)
            w13f, sw13 = fso.compat.quantize_moe_weights_1x128_fp8_sm90(w13)
            w2f, sw2 = fso.compat.quantize_moe_weights_1x128_fp8_sm90(w2)
            del w13, w2
            M = fso.compat.moe_swap_ab_max_m(E, TOPK) + 1  # the first M on the non-swap path
            assert fso.compat.moe_swap_ab_block_n(M, E, TOPK) is None, f"{name} M={M} is not a non-swap cell"
            hidden = torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 0.1
            topk_ids = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(M)]).to(torch.int32)
            topk_w = torch.softmax(torch.rand(M, TOPK, device="cuda"), dim=1)
            outs, fc2 = {}, {}
            for mode in ("default", "0", "1"):
                if mode == "default":
                    os.environ.pop("FSO_FC2_SWAP", None)
                else:
                    os.environ["FSO_FC2_SWAP"] = mode
                fc2[mode] = gemm_fp8._sm90_fc2_swap(False)
                outs[mode] = fso.compat.moe_layer_fp8_sm90(hidden, w13f, sw13, w2f, sw2, topk_ids, topk_w)
                torch.cuda.synchronize()
            assert fc2["0"] is False and fc2["1"] is True, f"{name}: FSO_FC2_SWAP did not switch the FC2 ({fc2})"
            sane = bool(torch.isfinite(outs["default"].float()).all()) and bool((outs["default"] != 0).any())
            eq0 = torch.equal(outs["default"].view(torch.int16), outs["0"].view(torch.int16))
            eq1 = torch.equal(outs["default"].view(torch.int16), outs["1"].view(torch.int16))
            ok = sane and eq0 and eq1
            print(f"{name} M={M:4d} non-swap, default FC2 {'swap-AB 64' if fc2['default'] else 'non-swap'}: "
                  f"== FC2 non-swap {eq0}, == FC2 swap-AB 64 {eq1}  {'OK' if ok else 'FAIL'}")
            assert ok, f"{name} M={M}: the FC2 route changed the layer output (or the output is not sane)"
            del w13f, sw13, w2f, sw2, outs
            torch.cuda.empty_cache()
    finally:
        if saved is None:
            os.environ.pop("FSO_FC2_SWAP", None)
        else:
            os.environ["FSO_FC2_SWAP"] = saved


def main():
    # This suite drives the composed sm_90 (H200) layer entry only; on any other
    # device the first op raises NotImplementedError, which used to exit 1 and
    # read as a failure. Skip explicitly instead, in the form
    # test_fp8_grouped_sm90.py uses (run b300_round3_20260922/M-A3).
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        where = "no CUDA device" if not torch.cuda.is_available() \
            else f"device is sm_{torch.cuda.get_device_capability()[0]}x"
        print(f"SKIP: test_moe_layer_dispatch_sm90 is sm_90 (H200) only ({where})")
        return 0

    torch.manual_seed(0)
    w13 = torch.randn(E, 2 * INTER, HIDDEN, device="cuda", dtype=torch.bfloat16) / math.sqrt(HIDDEN)
    w2 = torch.randn(E, HIDDEN, INTER, device="cuda", dtype=torch.bfloat16) / math.sqrt(INTER)
    w13f, sw13 = fso.compat.quantize_moe_weights_1x128_fp8_sm90(w13)
    w2f, sw2 = fso.compat.quantize_moe_weights_1x128_fp8_sm90(w2)
    thr = fso.compat.moe_swap_ab_max_m(E, TOPK)  # inclusive upper M of the swap-AB path
    print(f"moe_swap_ab_max_m(E={E}, topk={TOPK}) = {thr}")
    # E=128/top-8: rows per expert = M/16 -> block_n 16 up to M=192, 32 up to 384 (= thr), non-swap above

    print("=== dispatch cos across the threshold ===")
    for M in (1, 8, 64, 192, 193, thr, thr + 1, 1024):
        hidden = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1
        topk_ids = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(M)]).to(torch.int32)
        topk_w = torch.rand(M, TOPK, device="cuda", dtype=torch.float32)
        y = fso.compat.moe_layer_fp8_sm90(hidden, w13f, sw13, w2f, sw2, topk_ids, topk_w)
        bn = fso.compat.moe_swap_ab_block_n(M, E, TOPK)
        path = f"swap-AB/{bn}" if bn is not None else "contig"
        c = cos(y, moe_ref(hidden, w13, w2, topk_ids, topk_w))
        print(f"M={M:4d} -> {path:10s} cos={c:.4f}  {'OK' if c > 0.995 else 'FAIL'}")
        assert c > 0.995, f"cos {c} at M={M}"

    print("\n=== CUDA-graph capture + reroute (decode M=8, prefill M=256) ===")
    for M in (8, 256):
        hidden = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1
        topk_ids = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(M)]).to(torch.int32)
        topk_w = torch.rand(M, TOPK, device="cuda", dtype=torch.float32)
        fn = lambda: fso.compat.moe_layer_fp8_sm90(hidden, w13f, sw13, w2f, sw2, topk_ids, topk_w)
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = fn()
        topk_ids.copy_(torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(M)]).to(torch.int32))
        topk_w.copy_(torch.rand(M, TOPK, device="cuda"))
        hidden.copy_(torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1)
        g.replay(); torch.cuda.synchronize()
        c = cos(y, moe_ref(hidden, w13, w2, topk_ids, topk_w))
        print(f"M={M:4d} graph reroute cos={c:.4f}  {'OK' if c > 0.995 else 'FAIL'}")
        assert c > 0.995
    del w13, w2, w13f, sw13, w2f, sw2
    torch.cuda.empty_cache()
    fc2_route_equality()
    print("\nmoe_layer_fp8_sm90 dispatch: ALL PASS")


if __name__ == "__main__":
    sys.exit(main())
