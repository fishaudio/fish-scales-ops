#!/usr/bin/env python3
"""Machine-code audit of the sm_90 JIT GEMMs: no pipeline stage may be released while a generic shared-memory load
of that stage can still be outstanding (issue #12, SW1, 2026-11-03).

The math warps of the DeepGEMM v1 kernels (fp8_gemm_impl.cuh) and of the vendored DeepGEMM v2 kernel
(impls/sm90_fp8_gemm_1d2d.cuh) read their stage's TMA-written activation scales with ld.shared, and lane 0 releases the
stage with an mbarrier arrive. An arrive does not wait for the warp's outstanding loads, and ptxas places the first use
of the loaded registers where it likes: in the base kernels it put it after the arrive in 66 of 150 default-route
cubins, and the producer's TMA refill then overwrote the scales before the load was performed (wrong output blocks on
multi-wave launches). fso_fence_stage_reads() orders the loads before the release. ptxas is free to change its schedule
in a later version, so this test compiles a set of instantiations that covers every kernel family and every route
(dense main path and swap-AB, 64- and 128-row weight tiles, token tiles 16-80, NotDivisibleK and DivisibleK stage
counts, DeepGEMM v2, the grouped MoE kernels on the swap-AB, SwiGLU-fused, cluster-split and two-warp-group routes),
dumps their cubins (FSO_JIT_DUMP_CUBIN), disassembles them and checks every per-thread release
(SYNCS.ARRIVE.TRANS64.A1T0): each LDS / LDSM scoreboard set in the 80 instructions before it must be waited on by an
instruction in between or by the arrive itself, or be drained by a MEMBAR / FENCE.VIEW.ASYNC in between (the control
bits carry the scoreboard numbers). sm_90 only, and it needs cuobjdump (CUDA_HOME/bin or PATH); prints SKIP otherwise.
usage: python tests/gemm/unit/test_stage_release_audit_sm90.py
"""
import glob
import itertools
import os
import re
import shutil
import subprocess
import sys
import tempfile

DENSE = [  # (N, K, M): default routes, chosen for kernel-family and stage-count coverage
    (6144, 2560, 1), (6144, 2560, 32), (19456, 2560, 1), (19456, 2560, 20), (19456, 2560, 28), (19456, 2560, 64),
    (19456, 2560, 128), (2560, 9728, 1), (2560, 9728, 96), (2048, 4096, 40), (2048, 4096, 80), (5120, 2048, 1),
    (5120, 2048, 96), (5120, 2048, 768), (12288, 2048, 256), (1280, 1280, 78), (1280, 7680, 1079), (2048, 512, 16),
    (2560, 2048, 2048), (19456, 2560, 2048),
    # the swap-AB routes above M = 96 (issue 12, H4): 64-row tiles with 32 / 48 tokens, 128-row tiles with 48 / 80
    # tokens on two-wave launches
    (2048, 4096, 97), (2560, 4096, 136), (9728, 2560, 144), (12288, 2048, 152),
    # the in-cluster split-K route (issue 12, FH5): two slices with 16 / 24 / 48 / 56 / 64 tokens, three with 16 / 32
    # (and sharing SMs on 40 tiles), four with 16 / 24 / 56, and the 4 K blocks of 2048 x 512 in two slices
    (2560, 4096, 8), (2560, 4096, 24), (2048, 4096, 16), (2048, 4096, 32), (2560, 9216, 8), (2560, 9216, 48),
    (2560, 9728, 64), (1024, 2048, 16), (1024, 4096, 24), (2048, 512, 8), (2560, 4096, 56), (1280, 5120, 56),
    # the v2 kernel's fourfold-unrolled main loop on its 160-wide tiles (issue 12, FH6): 19456 x 2560 at M <= 128 above
    # runs the 160-wide one-wave tiles, 12352 x 2560 at M = 264 a 128 x 160 tile
    (12352, 2560, 264),
]
FORCED = [(12288, 2048, 136)]  # the swap-AB 128 x 24 tile of the failure, FSO_SM90_SWAP_MAX_M=1024 FSO_SM90_SWAP_BN=24
MOE = [("C", 1), ("C", 2), ("C", 64), ("C", 1024), ("B", 1), ("B", 512)]


def worker(which):
    import torch
    import fish_scales_ops as fso
    cells = FORCED if which == "forced" else DENSE
    if which in ("dense", "forced"):
        for n, k, m in cells:
            torch.manual_seed(m * 1009 + n * 17 + k)
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.5
            w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / (k ** 0.5)
            dw = fso.dense.prepare_weight(w, format="bsfp8")
            fso.dense.linear(x, dw)
            fso.dense.linear(x, dw, bias=torch.randn(n, dtype=torch.bfloat16, device="cuda"))
    else:
        fam = {"C": (256, 8, 2048, 512), "B": (128, 8, 2048, 768)}
        for f, m in MOE:
            E, topk, hidden, inter = fam[f]
            torch.manual_seed(0)
            w13f, sw13 = fso.compat.quantize_moe_weights_1x128_fp8_sm90(
                torch.randn(E, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.02)
            w2f, sw2 = fso.compat.quantize_moe_weights_1x128_fp8_sm90(
                torch.randn(E, hidden, inter, device="cuda", dtype=torch.bfloat16) * 0.02)
            h = torch.randn(m, hidden, device="cuda", dtype=torch.bfloat16)
            ids = torch.rand(m, E, device="cuda").topk(topk, dim=1).indices.to(torch.int32)
            wts = torch.softmax(torch.rand(m, topk, device="cuda"), dim=1).float()
            fso.compat.moe_layer_fp8_sm90(h, w13f, sw13, w2f, sw2, ids, wts)
    torch.cuda.synchronize()
    print("WORKER OK", flush=True)


INS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?);\s*/\* (0x[0-9a-f]+) \*/")
HI = re.compile(r"^\s+/\* (0x[0-9a-f]+) \*/")
FN = re.compile(r"Function : (\S+)")


def unwaited_releases(sass_text):
    """(function, arrive offset, pending loads) for every release with an LDS / LDSM scoreboard still pending."""
    funcs, cur, pend = {}, None, None
    for line in sass_text.splitlines():
        m = FN.search(line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            pend = None
            continue
        if cur is None:
            continue
        m = INS.search(line)
        if m:
            pend = (int(m.group(1), 16), m.group(2).strip())
            continue
        m = HI.match(line)
        if m and pend is not None:
            hi = int(m.group(1), 16)
            funcs[cur].append((pend[0], pend[1], (hi >> 46) & 7, (hi >> 52) & 0x3f))  # write barrier, wait mask
            pend = None
    bad = []
    for name, ins in funcs.items():
        for i, (off, txt, _, wait) in enumerate(ins):
            if "SYNCS.ARRIVE.TRANS64.A1T0" not in txt or ".RED" in txt:
                continue
            out = {}
            for o2, t2, wbar, wt2 in ins[max(0, i - 80):i]:
                op = t2.split()[1] if t2.startswith("@") else t2.split()[0]
                for b in range(6):
                    if wt2 >> b & 1:
                        out.pop(b, None)
                if op.startswith(("MEMBAR", "FENCE.VIEW.ASYNC", "EXIT")):
                    out.clear()
                if op.startswith("LDS") and wbar != 7:
                    out[wbar] = t2
            left = [t for b, t in out.items() if not (wait >> b & 1)]
            if left:
                bad.append((name, off, left))
    return bad


def main():
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 9:
        print("SKIP: sm_90 (H200) only")
        return 0
    cuobjdump = None
    for cand in ([os.path.join(os.environ["CUDA_HOME"], "bin", "cuobjdump")] if os.environ.get("CUDA_HOME") else []) \
            + [shutil.which("cuobjdump")]:
        if cand and os.path.isfile(cand):
            cuobjdump = cand
            break
    if cuobjdump is None:
        print("SKIP: cuobjdump not found (set CUDA_HOME)")
        return 0
    with tempfile.TemporaryDirectory() as cache:
        for which in ("dense", "forced", "moe"):
            env = {k: v for k, v in os.environ.items() if not k.startswith("FSO_")}
            env.update({"FSO_JIT_DUMP_CUBIN": "1", "FSO_JIT_CACHE_DIR": cache})
            if which == "forced":
                env.update({"FSO_SM90_SWAP_MAX_M": "1024", "FSO_SM90_SWAP_BN": "24"})
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", which], env=env,
                               capture_output=True, text=True, timeout=3000)
            if p.returncode != 0 or "WORKER OK" not in p.stdout:
                raise SystemExit(f"worker {which} failed, exit {p.returncode}:\n{p.stderr[-3000:]}")
        cubins = sorted(glob.glob(os.path.join(cache, "cache", "*", "*.cubin")))
        assert cubins, "no cubin was dumped (FSO_JIT_DUMP_CUBIN)"
        bad, kinds = [], set()
        for c in cubins:
            name = os.path.basename(os.path.dirname(c)).split("_", 1)[1]
            kinds.add("_".join(itertools.takewhile(lambda t: not re.fullmatch(r"\d+(x\d+)?", t), name.split("_"))))
            sass = subprocess.run([cuobjdump, "-sass", c], capture_output=True, text=True, check=True).stdout
            bad += [(name, off, left) for _, off, left in unwaited_releases(sass)]
    print(f"{len(cubins)} JIT cubins audited ({', '.join(sorted(kinds))}); releases with a pending shared-memory load: "
          f"{len(bad)}")
    for name, off, left in bad[:10]:
        print(f"  {name}: arrive at {off:#06x} with {left}")
    assert not bad, f"{len(bad)} stage releases can run ahead of the warp's ld.shared of that stage"
    print("sm_90 stage release audit: PASS")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        sys.exit(main())
