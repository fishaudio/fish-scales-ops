#!/usr/bin/env python3
"""Write the MoE layer's outputs for fixed inputs, so that two builds can be compared bitwise.

    python tests/tools/dump_layer_outputs.py OUT.pt

The inputs are deterministic (fixed seeds, the same generator calls in the same order), so two builds that compute the
same layer produce byte-identical tensors, and `scripts/ci/run_suite.py --layer-ref REF.pt` compares a build's dump with
a reference build's. The reference dumps of the 0.2.0 release were written by 537efa5-era builds with these exact
inputs; do not change the seeds, the shapes, the order of the random draws or the keys, or those references stop
applying.

Per architecture:

* sm_90 (H200): block-FP8 experts of Family C (E=256, I=512) and Family B (E=128, I=768), eleven token counts from 1
  to 8192, which cover every route (swap-AB 16 / 32, non-swap). Keys ``<fam>/M<M>/entry`` (`moe_layer_fp8_sm90`) and
  ``<fam>/M<M>/unified`` (`fso.moe.layer`).
* sm_120/121 (RTX 5090): MXFP8 experts of three geometries, six token counts. Keys ``entry`` and ``entry_bias``
  (`moe_layer_mxfp8_sm120` without and with a bias) and ``unified`` (`fso.moe.layer`).
* sm_100/103 (B200/B300): the same MXFP8 inputs. Keys ``unified_bias`` and ``unified`` (the per-arch sm_120 entry
  refuses this device, so the unified op is the entry).

Every id tensor carries skipped ids (-1 and E), as a serving stack's padded rows and expert-parallel entries do.
"""
import sys

import torch
import fish_scales_ops as fso


def dump_sm90(dev):
    res = {}
    for fam, (E, TOPK, H, I) in {"C_35a3": (256, 8, 2048, 512), "B_30a3": (128, 8, 2048, 768)}.items():
        torch.manual_seed(11)
        w13 = torch.randn(E, 2 * I, H, device=dev, dtype=torch.bfloat16) * 0.02
        w2 = torch.randn(E, H, I, device=dev, dtype=torch.bfloat16) * 0.02
        w13q, s13 = fso.compat.quantize_moe_weights_1x128_fp8_sm90(w13)
        w2q, s2 = fso.compat.quantize_moe_weights_1x128_fp8_sm90(w2)
        experts = fso.moe.prepare_experts(w13q, w2q, format="bsfp8", sw13=s13, sw2=s2)
        del w13, w2
        for M in (1, 2, 8, 64, 256, 512, 1024, 2048, 3072, 4096, 8192):
            g = torch.Generator(device=dev); g.manual_seed(1000 + M)
            hidden = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
            ids = torch.rand(M, E, device=dev, generator=g).topk(TOPK, dim=1).indices.to(torch.int32)
            ids[::7, 0] = -1          # an expert-parallel remote entry
            ids[::11, 1] = E          # a padded-row sentinel
            wts = torch.softmax(torch.rand(M, TOPK, device=dev, generator=g), dim=1).float()
            a = fso.compat.moe_layer_fp8_sm90(hidden, w13q, s13, w2q, s2, ids, wts)
            c = fso.moe.layer(hidden, experts, ids, wts)
            torch.cuda.synchronize()
            res[f"{fam}/M{M}/entry"] = a.cpu()
            res[f"{fam}/M{M}/unified"] = c.cpu()
        del experts, w13q, w2q
        torch.cuda.empty_cache()
    return res


def dump_mxfp8(dev, major):
    res = {}
    geometries = {"C_35a3_tp1": (256, 8, 2048, 512), "C_35a3_ep2": (128, 8, 2048, 512), "B_30a3": (128, 8, 2048, 768)}
    for fam, (E, TOPK, H, I) in geometries.items():
        torch.manual_seed(11)
        w13 = torch.randn(E, 2 * I, H, device=dev, dtype=torch.bfloat16) * 0.02
        w2 = torch.randn(E, H, I, device=dev, dtype=torch.bfloat16) * 0.02
        w13q, s13 = fso.compat.quantize_moe_weights_1x32_fp8(w13, w13_interleave=True)
        w2q, s2 = fso.compat.quantize_moe_weights_1x32_fp8(w2)
        experts = fso.moe.prepare_experts(w13, w2, format="mxfp8")
        del w13, w2
        for M in (1, 8, 64, 512, 2048, 4096):
            g = torch.Generator(device=dev); g.manual_seed(1000 + M)
            hidden = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
            ids = torch.rand(M, E, device=dev, generator=g).topk(TOPK, dim=1).indices.to(torch.int32)
            ids[::7, 0] = -1          # an expert-parallel remote entry
            ids[::11, 1] = E          # a padded-row sentinel
            wts = torch.softmax(torch.rand(M, TOPK, device=dev, generator=g), dim=1).float()
            bias = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.1
            bscale = torch.rand(M, device=dev, generator=g)
            if major == 12:
                a = fso.compat.moe_layer_mxfp8_sm120(hidden, w13q, s13, w2q, s2, ids, wts, w13_interleaved=True)
                b = fso.compat.moe_layer_mxfp8_sm120(hidden, w13q, s13, w2q, s2, ids, wts, w13_interleaved=True,
                                                     bias=bias, bias_scale=bscale)
                res[f"{fam}/M{M}/entry"] = a.cpu(); res[f"{fam}/M{M}/entry_bias"] = b.cpu()
            else:  # sm_100/103: the per-arch sm_120 entry refuses this device; the unified op is the entry
                b = fso.moe.layer(hidden, experts, ids, wts, bias=bias, bias_scale=bscale)
                res[f"{fam}/M{M}/unified_bias"] = b.cpu()
            c = fso.moe.layer(hidden, experts, ids, wts)
            torch.cuda.synchronize()
            res[f"{fam}/M{M}/unified"] = c.cpu()
        del experts, w13q, w2q
        torch.cuda.empty_cache()
    return res


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    major = torch.cuda.get_device_capability(0)[0]
    if major == 9:
        res = dump_sm90("cuda")
    elif major in (10, 12):
        res = dump_mxfp8("cuda", major)
    else:
        raise SystemExit(f"dump_layer_outputs: no MoE layer on sm_{major}x")
    torch.save(res, sys.argv[1])
    print(f"wrote {len(res)} tensors to {sys.argv[1]}")


if __name__ == "__main__":
    main()
