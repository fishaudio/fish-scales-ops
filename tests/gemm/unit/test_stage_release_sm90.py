#!/usr/bin/env python3
"""Run-to-run byte stability of the sm_90 dense swap-AB GEMM on a multi-wave launch, the configuration that exposed
the stage-release race of the DeepGEMM v1 kernels (issue #12, SW1, 2026-11-03).

The race: every math warp reads its pipeline stage's TMA-written activation scales with ld.shared before the WGMMA, and
lane 0 releases the stage after warpgroup_wait. Without a fence nothing makes the warp wait for those loads before the
release, and in the last real stage of a NotDivisibleK pass ptxas uses the loaded registers only after it. When the
producer refills the stage at once for the CTA's next tile and that TMA write lands before the load is performed, the
warp promotes the tile's last k-block with the next tile's scales: one 64-row x BLOCK_N block of the output is wrong.
A forced 128 x 24 swap-AB tile on N = 12288, K = 2048 (6 stages, so a 4-stage tail pass) at M = 136 (576 tiles, five
waves on 132 SMs) returned wrong bytes on 127 of 10,000 calls before the fix (fso_fence_stage_reads in
fp8_gemm_impl.cuh) and on none after it. This test runs that configuration in a child process (FSO_SM90_SWAP_MAX_M and
FSO_SM90_SWAP_BN are read once per process) and requires every call to equal the first call and the default route's
output computed here (the main path; both routes compute each element in one CTA in K order, so the bytes are equal).
sm_90 only (prints SKIP elsewhere). usage: python tests/gemm/unit/test_stage_release_sm90.py
"""
import os
import subprocess
import sys
import tempfile

N, K, M = 12288, 2048, 136
CALLS = 4000


def inputs():
    import torch
    import fish_scales_ops as fso
    torch.manual_seed(M * 1009 + N * 17 + K)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.5
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.5)
    dw = fso.dense.prepare_weight(w, format="bsfp8")
    xq, sx = fso.compat.quantize_1x128_fp8(x)
    return fso, x, dw, xq, sx


def worker(ref_path):
    import torch
    fso, x, dw, xq, sx = inputs()
    ref = torch.load(ref_path).cuda().view(torch.int16)
    first = fso.compat.linear_fp8(xq, dw.weight, sx, dw.scale).view(torch.int16).clone()
    bad_first = int(not torch.equal(first, ref))
    bad = 0
    done = 0
    while done < CALLS:
        ys = []
        for i in range(50):  # 50 calls back to back, the quantize -> GEMM chain on every other one
            ys.append(fso.dense.linear(x, dw) if (done + i) % 2 else fso.compat.linear_fp8(xq, dw.weight, sx, dw.scale))
        torch.cuda.synchronize()
        bad += sum(int(not torch.equal(y.view(torch.int16), ref)) for y in ys)
        done += 50
    print(f"RESULT {bad_first} {bad}", flush=True)


def main():
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 9:
        print("SKIP: sm_90 (H200) only")
        return 0
    fso, x, dw, xq, sx = inputs()
    ref = fso.compat.linear_fp8(xq, dw.weight, sx, dw.scale)  # default route: the main path at M = 136
    torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = os.path.join(tmp, "ref.pt")
        torch.save(ref.cpu(), ref_path)
        env = {k: v for k, v in os.environ.items() if not k.startswith("FSO_")}
        env["FSO_SM90_SWAP_MAX_M"] = "1024"
        env["FSO_SM90_SWAP_BN"] = "24"
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", ref_path], env=env,
                           capture_output=True, text=True, timeout=1800)
    res = [line for line in p.stdout.splitlines() if line.startswith("RESULT ")]
    if p.returncode != 0 or not res:
        raise SystemExit(f"worker failed, exit {p.returncode}:\n{p.stderr[-3000:]}")
    bad_first, bad = (int(v) for v in res[-1].split()[1:])
    print(f"forced swap-AB 128 x 24, {N} x {K}, M = {M} (5 waves): first call "
          f"{'!=' if bad_first else '=='} default route; {bad} of {CALLS} calls differ from it")
    assert bad_first == 0, "the forced swap-AB route's first call differs from the default route's output"
    assert bad == 0, f"{bad} of {CALLS} calls of the forced multi-wave swap-AB launch returned different bytes"
    print("sm_90 stage release order: PASS")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        sys.exit(main())
