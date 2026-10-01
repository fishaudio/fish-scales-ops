"""sm_120/121 `linear_bf16` packed-scale scratch: growth after a capture, and growth inside one.

On sm_120/121 the bf16-input path (`fso.compat.linear_bf16` and
`fso.compat.linear_qx`) repacks its FP32 block scales into int32 UE8M0 words in a
per-thread scratch pool (`Sm120BfPackPool` in
csrc/gemm/include/blockscale_gemm/dispatch.cuh). A captured CUDA graph keeps the
scratch address it saw at capture time, so the pool must never free a buffer a
graph may still use. Two cases, each in its own process so the thread-local pool
starts empty:

* ``grow``: capture `linear_bf16` at M = 4, which sizes the activation-scale
  scratch for that call (the pool allocates at least 1 MiB); make an eager call
  at M = 32768, K = 8192, whose 2 MiB of packed activation scales need a larger
  buffer; NaN-fill the captured output and replay. The replay must be
  bit-exact against the eager M = 4 result, and the eager M = 32768 result must
  be unchanged afterwards (a pool that freed the captured buffer lets the
  replays write into memory that now belongs to someone else).
* ``capture_grow``: one eager call at M = 4, then a capture at M = 32768 straight
  away. The pool cannot allocate inside a capture, so the op must raise
  RuntimeError naming the remedy, and the process must stay usable: an eager
  M = 32768 call afterwards must succeed and match a fresh one.

Run: ``PYTHONPATH=python python tests/gemm/unit/test_sm120_pack_pool_growth.py``.
On any architecture other than sm_120/121 the file exits 0 without testing.
"""
from __future__ import annotations

import subprocess
import sys

import torch

import fish_scales_ops as fso

N, K = 256, 8192
M_SMALL, M_LARGE = 4, 32768   # packed activation scales: 4 * 16 * 4 B vs 32768 * 16 * 4 B = 2 MiB


def _inputs(M):
    torch.manual_seed(M * 1009 + N * 17 + K)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.5)
    return x, w


def case_grow():
    xs, w = _inputs(M_SMALL)
    xl, _ = _inputs(M_LARGE)
    ref_small = fso.compat.linear_bf16(xs, w).clone()
    assert torch.isfinite(ref_small).all(), "eager M=4 reference has NaN/Inf"
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fso.compat.linear_bf16(xs, w)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        captured = fso.compat.linear_bf16(xs, w)
    torch.cuda.synchronize()

    ref_large = fso.compat.linear_bf16(xl, w).clone()     # grows the activation-scale scratch
    torch.cuda.synchronize()
    ref_large_host = ref_large.cpu()
    assert torch.isfinite(ref_large).all(), "eager M=32768 result has NaN/Inf"

    for _ in range(20):
        captured.fill_(float("nan"))
        g.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(captured).all(), "replay after the scratch grew produced NaN/Inf"
    assert torch.equal(captured, ref_small), "replay after the scratch grew differs from the eager M=4 result"
    assert torch.equal(ref_large.cpu(), ref_large_host), "the replays overwrote the eager M=32768 result"
    print("  grow: 20 replays bit-exact after the scratch grew; the later eager result is untouched")


def case_capture_grow():
    xs, w = _inputs(M_SMALL)
    xl, _ = _inputs(M_LARGE)
    fso.compat.linear_bf16(xs, w)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    raised = None
    try:
        with torch.cuda.graph(g, stream=s):
            fso.compat.linear_bf16(xl, w)
    except RuntimeError as exc:
        raised = str(exc)
    assert raised is not None, "a capture that needs a larger scratch did not raise"
    assert "eagerly" in raised, f"the error does not name the remedy: {raised[:300]}"
    torch.cuda.synchronize()
    a = fso.compat.linear_bf16(xl, w)
    b = fso.compat.linear_bf16(xl, w)
    torch.cuda.synchronize()
    assert torch.isfinite(a).all() and torch.equal(a, b), "eager calls after the refused capture are wrong"
    print("  capture_grow: RuntimeError naming the remedy; eager calls afterwards are correct")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        {"grow": case_grow, "capture_grow": case_capture_grow}[sys.argv[2]]()
        return
    major = torch.cuda.get_device_capability()[0]
    if major != 12:
        print(f"SKIP: test_sm120_pack_pool_growth is sm_120/121 only (device is sm_{major}x)")
        return
    for case in ("grow", "capture_grow"):
        r = subprocess.run([sys.executable, __file__, "--case", case], capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stdout.write(r.stderr[-3000:])
            raise SystemExit(f"case {case} failed (exit {r.returncode})")
    print("sm_120 linear_bf16 packed-scale scratch growth: ALL PASS")


if __name__ == "__main__":
    main()
