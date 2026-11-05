"""sm_100/103 dense split-K workspace: per call, from the caller's stream (issue #12, charter 7.5).

The dense C++ cascade of sm_100/103 runs the narrow-N long-K decode cells
(N <= 4096, K >= 4096 at M <= 32, K >= 8192 up to M = 104) as a two-kernel
parallel split-K. Until the issue-12 round its FP32 partial sums lived in a
per-thread pool that grew on eager calls, never freed a buffer, refused to grow
inside a capture, and was shared by every stream and graph of the thread. Now
the op body asks the cascade for the bytes the call needs and allocates them
with the torch caching allocator on the stream it launches on (graph-private
memory inside a capture), so nothing has to be sized eagerly and no two
streams share a workspace. The pool remains for the sweep-only Stream-K force
codes, which this file does not touch. Two cases, each in its own process:

* ``grow``: capture the down projection (N = 2560, K = 9728) at M = 1; make an
  eager call at M = 104 (the largest M of the band), which needs a larger
  workspace; NaN-fill the captured output and replay. The replay must be
  bit-exact against the eager M = 1 result, and the eager M = 104 result must be
  unchanged afterwards. When the driver API of `cuda-python` is importable
  (nvidia-cutlass-dsl depends on it), the case also allocates a few sentinel
  buffers of exactly the M = 1 workspace size after the eager call, fills them
  with a byte pattern and checks the pattern after the replays: a graph that
  wrote memory it does not own would be likely to hit them. ``--no-sentinel``
  leaves them out, which is how the case runs under
  ``compute-sanitizer --tool memcheck``.
* ``capture_grow``: one eager call at M = 1, then a capture at M = 104 straight
  away. That capture used to raise (the pool could not grow inside it); it must
  now succeed, its replay into a NaN-filled output must equal an eager M = 104
  call, and the process must stay usable.

The children run with FSO_PRINT_TILE_INFO=1, so their stderr shows that the
split-K kernel ran (``[fso sm100 tile] splitk``), the workspace each capture
baked in (``[fso sm100 splitk] captured on stream ...``: exactly one line per
capture), and that the old pool never grew (no ``[fso sm100 workspace] grew``
line). The op is ``torch.ops.fish_scales_ops.linear_mxfp8_raw``, which goes
straight to the C++ cascade; the public ``linear_mxfp8`` would hand M <= 64 to
the CuTe-DSL decode row.

Run: ``PYTHONPATH=python python tests/gemm/unit/test_sm100_workspace_growth.py``.
On any architecture other than sm_100/103 the file exits 0 without testing.
"""
from __future__ import annotations

import os
import subprocess
import sys

import torch

import fish_scales_ops as fso

N, K = 2560, 9728            # Qwen3-4B `down`: 20 N tiles, 76 K blocks -> 4 K slices
M_SMALL, M_LARGE = 1, 104     # 104: the largest M the K >= 8192 split-K band takes
SPLITS = 4                   # pick_splits(20 tiles, 76 K blocks) on a 148-SM part


def _operands(M):
    torch.manual_seed(M * 1009 + N * 17 + K)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    return fso.compat.quantize_1x32_fp8(x)


def _weight():
    torch.manual_seed(N * 17 + K)
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.5)
    return fso.compat.quantize_1x32_fp8(w)


def _gemm(xq, sx, wq, sw):
    return torch.ops.fish_scales_ops.linear_mxfp8_raw(xq, wq, sx, sw)


class _Sentinels:
    """Raw driver allocations of one size, filled with a byte pattern."""

    PATTERN = 0xA5

    def __init__(self, nbytes, count):
        self.nbytes, self.bufs, self.cu = nbytes, [], None
        try:
            from cuda.bindings import driver as cu
        except Exception as exc:                     # cuda-python not installed
            print(f"  sentinel buffers skipped ({type(exc).__name__}: {exc})")
            return
        self.cu = cu
        for _ in range(count):
            err, ptr = cu.cuMemAlloc(nbytes)
            if err != cu.CUresult.CUDA_SUCCESS:
                print(f"  sentinel buffers: cuMemAlloc failed ({err}); using {len(self.bufs)}")
                break
            (err,) = cu.cuMemsetD8(ptr, self.PATTERN, nbytes)
            assert err == cu.CUresult.CUDA_SUCCESS, f"cuMemsetD8 failed: {err}"
            self.bufs.append(ptr)
        torch.cuda.synchronize()

    def clobbered(self):
        """Indices of the sentinels whose pattern changed."""
        bad = []
        for i, ptr in enumerate(self.bufs):
            copy = torch.empty(self.nbytes, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            (err,) = self.cu.cuMemcpyDtoD(copy.data_ptr(), ptr, self.nbytes)
            assert err == self.cu.CUresult.CUDA_SUCCESS, f"cuMemcpyDtoD failed: {err}"
            torch.cuda.synchronize()
            if not bool((copy == self.PATTERN).all()):
                bad.append(i)
        return bad

    def free(self):
        for ptr in self.bufs:
            self.cu.cuMemFree(ptr)
        self.bufs = []


def case_grow(sentinel=True):
    wq, sw = _weight()
    xs, sxs = _operands(M_SMALL)
    xl, sxl = _operands(M_LARGE)

    # Eager call, then warm-up on the capture stream: the pool is sized for M = 1.
    ref_small = _gemm(xs, sxs, wq, sw).clone()
    assert torch.isfinite(ref_small).all(), "eager M=1 reference has NaN/Inf"
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _gemm(xs, sxs, wq, sw)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        captured = _gemm(xs, sxs, wq, sw)
    torch.cuda.synchronize()

    # Eager call that needs a larger split-K workspace than the capture saw. Its
    # result is a tensor allocated after the capture; a host copy tells a later
    # overwrite of that tensor apart from a wrong recomputation.
    ref_large = _gemm(xl, sxl, wq, sw).clone()
    torch.cuda.synchronize()
    ref_large_host = ref_large.cpu()
    assert torch.isfinite(ref_large).all(), f"eager M={M_LARGE} result has NaN/Inf"

    sent = _Sentinels(SPLITS * M_SMALL * N * 4, count=8) if sentinel else None

    for _ in range(20):
        captured.fill_(float("nan"))
        g.replay()
    torch.cuda.synchronize()

    assert torch.isfinite(captured).all(), "replay after a larger eager call produced NaN/Inf"
    if not torch.equal(captured, ref_small):
        diff = (captured.float() - ref_small.float()).abs().max().item()
        raise AssertionError(f"replay after a larger eager call != eager (max abs diff {diff})")
    assert torch.equal(ref_large.cpu(), ref_large_host), (
        "the replays overwrote a tensor allocated after the capture (the eager "
        f"M={M_LARGE} result): the graph writes memory it does not own")
    again = _gemm(xl, sxl, wq, sw)
    torch.cuda.synchronize()
    assert torch.equal(again.cpu(), ref_large_host), f"eager M={M_LARGE} changed after the replays"
    if sent is not None and sent.cu is not None:
        bad, total = sent.clobbered(), len(sent.bufs)
        sent.free()
        assert not bad, (f"the replays overwrote {len(bad)} of {total} buffers allocated after "
                         "the capture: the graph writes memory it does not own")
        print(f"  sentinels         {total} x {SPLITS * M_SMALL * N * 4} B allocated after the "
              "capture, untouched by the replays  OK")
    print(f"  grow              capture M={M_SMALL}, eager M={M_LARGE}, 20 replays bit-exact  OK")


def case_capture_grow():
    wq, sw = _weight()
    xs, sxs = _operands(M_SMALL)
    xl, sxl = _operands(M_LARGE)
    _gemm(xs, sxs, wq, sw)                 # an eager call at M = 1 only: nothing is sized for M = 104
    torch.cuda.synchronize()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):    # the workspace comes from the graph's private pool
        captured = _gemm(xl, sxl, wq, sw)
    torch.cuda.synchronize()
    captured.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    ref = _gemm(xl, sxl, wq, sw)
    torch.cuda.synchronize()
    assert torch.isfinite(captured).all() and torch.equal(captured, ref), \
        f"capture at M={M_LARGE} without an eager call of that shape: replay != eager"
    print(f"  capture_grow      capture at M={M_LARGE} after an eager M={M_SMALL} only: no error, "
          "replay into NaN == eager  OK")

    # The process is still usable.
    y1 = _gemm(xl, sxl, wq, sw).clone()
    y2 = _gemm(xl, sxl, wq, sw)
    torch.cuda.synchronize()
    assert torch.isfinite(y1).all() and torch.equal(y1, y2), f"eager M={M_LARGE} after the capture is wrong"
    print(f"  capture_grow      eager M={M_LARGE} after the capture consistent  OK")


def _child(case, *extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("FSO_")}
    env["FSO_PRINT_TILE_INFO"] = "1"
    return subprocess.run([sys.executable, os.path.abspath(__file__), "--case", case, *extra],
                          env=env, capture_output=True, text=True)


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--case":
        if sys.argv[2] == "grow":
            case_grow(sentinel="--no-sentinel" not in sys.argv)
        elif sys.argv[2] == "capture_grow":
            case_capture_grow()
        else:
            raise SystemExit(f"unknown case {sys.argv[2]}")
        return 0

    assert torch.cuda.is_available(), "CUDA required"
    sm = torch.cuda.get_device_capability(0)
    print(f"Device: {torch.cuda.get_device_name(0)} sm_{sm[0]}{sm[1]}\n")
    if sm[0] != 10:
        print("SKIP: the dense split-K is sm_100/sm_103-only")
        return 0

    failed = []
    for case in ("grow", "capture_grow"):
        r = _child(case)
        sys.stdout.write(r.stdout)
        grew = [ln for ln in r.stderr.splitlines() if ln.startswith("[fso sm100 workspace] grew")]
        caps = [ln for ln in r.stderr.splitlines() if ln.startswith("[fso sm100 splitk] captured")]
        problems = []
        if r.returncode != 0:
            problems.append(f"exit {r.returncode}")
        if "[fso sm100 tile] splitk" not in r.stderr:
            problems.append("the split-K kernel never ran")
        # The per-thread pool is not used by the split-K any more: it never grows.
        if grew:
            problems.append(f"the per-thread workspace pool grew {len(grew)} time(s)")
        # One capture per case, one workspace line per capture.
        if len(caps) != 1:
            problems.append(f"expected 1 captured-workspace line, got {len(caps)}")
        if problems:
            failed.append(case)
            print(f"  {case:<17} FAILED: {'; '.join(problems)}\n{r.stderr[-3000:]}")
        else:
            for ln in caps:
                print(f"    {ln}")
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    print("\nsm_100 split-K workspace (per call, caller's stream): ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
