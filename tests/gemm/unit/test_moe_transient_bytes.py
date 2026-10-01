#!/usr/bin/env python3
"""fso.moe.transient_bytes against what one fso.moe.layer call really allocates.

The layer runs every bucket as one call, so a serving stack reserves
fso.moe.transient_bytes(experts, tokens, topk) for its largest bucket. That only
works if the figure is never below the real peak and not far above it. Per cell
this test measures the peak allocated bytes of one eager call above the bytes
held before it (the inputs, the weights and every persistent pool are already
resident: the first call of each shape runs before the measured one) and asserts

    measured <= reserved  and  reserved - measured <= 12 MiB + 1 % of reserved.

The query charges every tensor above 1 MiB one extra MiB, because the caching
allocator may serve it from a cached block without splitting off a remainder of up
to 1 MiB (seen on sm_90: +1024 KiB and +896 KiB on the two GEMM outputs); a call has
at most ten such tensors, hence the 12 MiB of allowed slack.

It runs on whichever architecture the device is: sm_90 builds block-FP8 experts
and exercises the expert-sorted layer, sm_100/103 and sm_120/121 build bf16
experts in the mxfp8 format and exercise the masked slab (with the fused FC1,
which prepare_experts selects by default). On sm_90 it also checks that the
query's padded row count equals what moe_build_sorted allocates.
"""
import sys

import torch

import fish_scales_ops as fso
from fish_scales_ops.gemm import fp8 as gemm_fp8  # the implementation module, for the private sm_90 plan

FAMILIES = {
    # label: (E_local, topk, hidden, inter)
    "C_35a3_tp1": (256, 8, 2048, 512),
    "C_35a3_ep2": (128, 8, 2048, 512),
    "B_30a3": (128, 8, 2048, 768),
}
TOKENS = (1, 2, 8, 33, 64, 256, 1024, 2048)
MIB = 1 << 20


def _experts(E, H, I, major, dev):
    w13 = torch.randn(E, 2 * I, H, device=dev, dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(E, H, I, device=dev, dtype=torch.bfloat16) * 0.02
    if major == 9:
        w13q, s13 = fso.compat.quantize_moe_weights_1x128_fp8_sm90(w13)
        w2q, s2 = fso.compat.quantize_moe_weights_1x128_fp8_sm90(w2)
        return fso.moe.prepare_experts(w13q, w2q, format="bsfp8", sw13=s13, sw2=s2)
    return fso.moe.prepare_experts(w13, w2, format="mxfp8")


def main():
    if not torch.cuda.is_available():
        print("SKIP: test_moe_transient_bytes needs a CUDA device")
        return 0
    major = torch.cuda.get_device_capability(0)[0]
    if major not in (9, 10, 12):
        print(f"SKIP: fso.moe serves no format on sm_{major}x")
        return 0
    dev = "cuda"
    torch.manual_seed(0)
    worst_over = 0
    for fam, (E, TOPK, H, I) in FAMILIES.items():
        experts = _experts(E, H, I, major, dev)
        free = torch.cuda.mem_get_info()[0]
        for M in TOKENS:
            reserved = fso.moe.transient_bytes(experts, M, TOPK)
            if reserved > 0.8 * free:
                print(f"{fam:12s} M={M:5d}: skipped, {reserved / MIB:.0f} MiB does not fit beside the weights")
                continue
            g = torch.Generator(device=dev)
            g.manual_seed(7 + M)
            hidden = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
            ids = torch.rand(M, E, device=dev, generator=g).topk(TOPK, dim=1).indices.to(torch.int32).contiguous()
            wts = torch.softmax(torch.rand(M, TOPK, device=dev, generator=g), dim=1).float().contiguous()
            y = fso.moe.layer(hidden, experts, ids, wts)  # first call of the shape: pools, JIT
            del y
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            y = fso.moe.layer(hidden, experts, ids, wts)
            torch.cuda.synchronize()
            measured = torch.cuda.max_memory_allocated() - base
            del y
            slack = reserved - measured
            ok = measured <= reserved and slack <= 12 * MIB + 0.01 * reserved
            worst_over = max(worst_over, measured - reserved)
            print(f"{fam:12s} M={M:5d}: reserved {reserved / MIB:10.3f} MiB, measured {measured / MIB:10.3f} MiB, "
                  f"slack {slack / 1024:9.1f} KiB  {'OK' if ok else 'FAIL'}")
            assert measured <= reserved, f"{fam} M={M}: the call allocated {measured} B, above the {reserved} B reserved"
            assert slack <= 12 * MIB + 0.01 * reserved, f"{fam} M={M}: reserved {reserved} B but only {measured} B used"
            if major == 9:
                _, _, block, p_max, _ = gemm_fp8._sm90_layer_plan(M, E, TOPK, H, I)
                se, _, _ = torch.ops.fish_scales_ops.moe_build_sorted(ids, E, block)
                assert int(se.shape[0]) == p_max, (
                    f"{fam} M={M}: moe_build_sorted allocated {int(se.shape[0])} rows, the plan says {p_max}")
        del experts
        torch.cuda.empty_cache()
    print(f"\nfso.moe.transient_bytes on sm_{major}x: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
