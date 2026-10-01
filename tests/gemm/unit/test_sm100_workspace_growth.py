"""sm_100/103 dense split-K workspace: growth after a capture, and growth inside one.

The dense C++ cascade of sm_100/103 runs the narrow-N long-K decode cells
(N <= 4096, K >= 4096 at M <= 32, K >= 8192 up to M = 128) as a two-kernel
parallel split-K, and keeps the FP32 partial sums in a per-thread workspace
pool (`Sm100WorkspacePool` in
csrc/gemm/include/blockscale_gemm/arch/sm100/mxfp8/dispatch.cuh). A captured
CUDA graph keeps the workspace address it saw at capture time, so the pool must
never free a buffer a graph may still use. Two cases, each in its own process so
the thread-local pool starts empty:

* ``grow``: capture the down projection (N = 2560, K = 9728) at M = 1, which
  sizes the pool for that call; make an eager call at M = 128, which needs a
  larger workspace; NaN-fill the captured output and replay. The replay must be
  bit-exact against the eager M = 1 result, and the eager M = 128 result must be
  unchanged afterwards. When the driver API of `cuda-python` is importable
  (nvidia-cutlass-dsl depends on it), the case also allocates a few sentinel
  buffers of exactly the M = 1 workspace size after the growth, fills them with a
  byte pattern and checks the pattern after the replays: a pool that freed the
  captured buffer hands its address back to the driver, the sentinels are
  likely to land on it, and the replays then overwrite memory the process owns
  elsewhere. ``--no-sentinel`` leaves them out, which is how the case runs under
  ``compute-sanitizer --tool memcheck`` (a sentinel would make the freed range
  valid memory again and hide the access).
* ``capture_grow``: one eager call at M = 1, then a capture at M = 128 straight
  away. The pool cannot allocate inside a capture, so the op must raise
  RuntimeError naming the remedy, and the process must stay usable: an eager
  M = 128 call afterwards must succeed and match a fresh one.

The children run with FSO_PRINT_TILE_INFO=1, so their stderr shows that the
split-K kernel ran (``[fso sm100 tile] splitk``) and when the pool grew
(``[fso sm100 workspace] grew``). The op is
``torch.ops.fish_scales_ops.linear_mxfp8_raw``, which goes straight to the C++
cascade; the public ``linear_mxfp8`` would hand M <= 64 to the CuTe-DSL decode
row, which does not use this pool.

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
M_SMALL, M_LARGE = 1, 128
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
    # result is a tensor allocated after the growth; a host copy tells a later
    # overwrite of that tensor apart from a wrong recomputation.
    ref_large = _gemm(xl, sxl, wq, sw).clone()
    torch.cuda.synchronize()
    ref_large_host = ref_large.cpu()
    assert torch.isfinite(ref_large).all(), "eager M=128 result has NaN/Inf"

    sent = _Sentinels(SPLITS * M_SMALL * N * 4, count=8) if sentinel else None

    for _ in range(20):
        captured.fill_(float("nan"))
        g.replay()
    torch.cuda.synchronize()

    assert torch.isfinite(captured).all(), "replay after the workspace grew produced NaN/Inf"
    if not torch.equal(captured, ref_small):
        diff = (captured.float() - ref_small.float()).abs().max().item()
        raise AssertionError(f"replay after the workspace grew != eager (max abs diff {diff})")
    assert torch.equal(ref_large.cpu(), ref_large_host), (
        "the replays overwrote a tensor allocated after the workspace grew (the eager "
        "M=128 result): the graph still writes the workspace the pool freed")
    again = _gemm(xl, sxl, wq, sw)
    torch.cuda.synchronize()
    assert torch.equal(again.cpu(), ref_large_host), "eager M=128 changed after the replays"
    if sent is not None and sent.cu is not None:
        bad, total = sent.clobbered(), len(sent.bufs)
        sent.free()
        assert not bad, (f"the replays overwrote {len(bad)} of {total} buffers allocated after "
                         "the workspace grew: the graph still writes the workspace the pool freed")
        print(f"  sentinels         {total} x {SPLITS * M_SMALL * N * 4} B allocated after the "
              "growth, untouched by the replays  OK")
    print(f"  grow              capture M={M_SMALL}, eager M={M_LARGE}, 20 replays bit-exact  OK")


def case_capture_grow():
    wq, sw = _weight()
    xs, sxs = _operands(M_SMALL)
    xl, sxl = _operands(M_LARGE)
    _gemm(xs, sxs, wq, sw)                 # sizes the pool for M = 1 only
    torch.cuda.synchronize()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    raised = None
    try:
        with torch.cuda.graph(g, stream=s):
            _gemm(xl, sxl, wq, sw)
    except RuntimeError as exc:
        raised = exc
    assert raised is not None, "a capture that needed a larger workspace did not raise"
    assert "split-K workspace" in str(raised) and "eagerly" in str(raised), \
        f"RuntimeError without the remedy: {raised}"
    print(f"  capture_grow      RuntimeError: ...{str(raised).splitlines()[0][-120:]}")

    # The process is still usable and the pool still works.
    torch.cuda.synchronize()
    y1 = _gemm(xl, sxl, wq, sw).clone()
    y2 = _gemm(xl, sxl, wq, sw)
    torch.cuda.synchronize()
    assert torch.isfinite(y1).all() and torch.equal(y1, y2), "eager M=128 after the refusal is wrong"
    print("  capture_grow      process usable after the refusal, eager M=128 consistent  OK")


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
        print("SKIP: the split-K workspace pool is sm_100/sm_103-only")
        return 0

    failed = []
    for case in ("grow", "capture_grow"):
        r = _child(case)
        sys.stdout.write(r.stdout)
        grew = [ln for ln in r.stderr.splitlines() if ln.startswith("[fso sm100 workspace] grew")]
        problems = []
        if r.returncode != 0:
            problems.append(f"exit {r.returncode}")
        if "[fso sm100 tile] splitk" not in r.stderr:
            problems.append("the split-K kernel never ran")
        # grow: the M = 1 call and the M = 128 call. capture_grow: the M = 1
        # call and the eager M = 128 call after the refusal (none inside it).
        if len(grew) != 2:
            problems.append(f"expected 2 workspace growth lines, got {len(grew)}")
        if problems:
            failed.append(case)
            print(f"  {case:<17} FAILED: {'; '.join(problems)}\n{r.stderr[-3000:]}")
        else:
            for ln in grew:
                print(f"    {ln}")
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    print("\nsm_100 split-K workspace growth: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
