"""sm_100/sm_103 MXFP8 decode row (M ≤ 32) — accuracy, routing, capture.

The decode row (`python/fish_scales_ops/gemm/_sm100_decode.py`) puts two
vendored NVIDIA CuTe-DSL kernels in front of the three tiers of the sm_100
MXFP8 router for the band M ≤ 32. This file checks the three things that can
go wrong when a kernel is swapped in under an existing public op:

1. **Accuracy.** For every Family A dense shape at M ∈ {1, 2, 4, 8, 16, 32} the
   decode row's output is compared with what the router returned before the row
   existed (cuBLAS `scaled_mm` on the wide-N shapes, the C++ two-kernel split-K
   cascade on the narrow-N ones) and with an FP32 reference. The cosine against
   the reference must agree with the old path's to five decimals, and the two
   outputs must agree elementwise to within one BF16 ulp — in most cells they
   are bit-identical, because the two designs sum the same FP32 K-slice
   partials in the same order.

2. **Routing.** The decode row must fire exactly where it is meant to: M ≤ 32
   with a tactic its own kernels accept, and nowhere else. The M > 32 band, the
   shapes whose K does not divide into the split-K tiling, and every
   architecture other than sm_100/103 must be untouched. The version gate is
   checked by mocking an old `nvidia-cutlass-dsl` in a subprocess and
   confirming that `route` then returns bit-identical results to the shipped
   path and that `pick_config` declines every cell.

3. **Capture.** After an eager warmup, capturing the call must not compile,
   must not allocate and must not synchronise; a replay after the activation
   and its scale buffer are rewritten in place must be bit-exact against an
   eager call on the same rewritten buffers.

Run: `PYTHONPATH=python python tests/gemm/unit/test_mxfp8_decode_sm100.py`.
On any architecture other than sm_100/103 the file exits 0 without testing.
"""
from __future__ import annotations

import os
import subprocess
import sys

import torch
import torch.nn.functional as F

import fish_scales_ops as fso

# Family A dense projections (docs/perf/README.md §3) plus the two Family B/C
# widths that share their classes.
SHAPES = [
    ("wqkv",     6144, 2560),   # wide N  (tiles_n = 48)  → persistent kernel
    ("wo",       2560, 4096),   # narrow N (tiles_n = 20) → split-K, K = 4096
    ("gate_up", 19456, 2560),   # wide N  (tiles_n = 152) → persistent kernel
    ("down",     2560, 9728),   # narrow N (tiles_n = 20) → split-K, K = 9728
]
M_GRID = [1, 2, 4, 8, 16, 32]


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.double().flatten(), b.double().flatten(),
                               dim=0).item()


def _bf16_tolerance(t: torch.Tensor) -> torch.Tensor:
    """Per-element "one BF16 ulp" tolerance for comparing against ``t``.

    BF16 keeps seven stored mantissa bits, so one ulp at a value is
    2^(exponent - 7). That is the right tolerance for an element of ordinary
    magnitude and a meaningless one for an element that rounded to zero or to a
    tiny residue, where a relative comparison has nothing to hold on to. The
    tolerance is therefore floored at one ulp of the LARGEST element scaled by
    2^-8, i.e. at a magnitude two BF16 resolutions below anything the output
    can express at its own scale.
    """
    f = t.float().abs()
    peak = f.max().clamp(min=2.0 ** -126)
    ulp = torch.pow(2.0, torch.floor(torch.log2(f.clamp(min=2.0 ** -126))) - 7.0)
    floor = torch.pow(2.0, torch.floor(torch.log2(peak)) - 7.0) * 2.0 ** -8
    return torch.clamp(ulp, min=float(floor))


def _inputs(M: int, N: int, K: int):
    """The bench draw, so a failure here is reproducible from a bench cell."""
    torch.manual_seed(M * 1009 + N * 17 + K)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.5)
    xq, sx = fso.gemm.quantize_1x32_fp8(x)
    wq, sw = fso.gemm.quantize_1x32_fp8(w)
    torch.cuda.synchronize()
    return x, w, xq, sx, wq, sw


def _baseline(xq, sx, wq, sw):
    """What `route` returned for this cell before the decode row existed.

    Mirrors the tier order of `_sm100_dispatch.route` with the decode row
    removed: tier 1 where `should_route` claims the cell, otherwise the raw C++
    cascade (tier 2's `pick_config` declines the whole M < 256 band, and the
    `wave_tile_owns` guard cannot fire below M = 64).
    """
    from fish_scales_ops.gemm import _sm100_smm
    m, k = xq.shape
    n = wq.shape[0]
    if _sm100_smm.should_route(m, n, k):
        y = _sm100_smm.linear_mxfp8_smm(xq, wq, sx, sw)
        if y is not None:
            return y, "tier1_cublas_scaled_mm"
    return torch.ops.fish_scales_ops.linear_mxfp8_raw(xq, wq, sx, sw), \
        "tier3_cxx_cascade"


def test_accuracy() -> None:
    from fish_scales_ops.gemm import _sm100_decode
    bit_exact = 0
    total = 0
    for tag, N, K in SHAPES:
        for M in M_GRID:
            x, w, xq, sx, wq, sw = _inputs(M, N, K)
            cfg = _sm100_decode.pick_config(M, N, K)
            assert cfg is not None, \
                f"{tag} M={M}: the decode row declined a cell it should own"
            y_new = fso.gemm.linear_mxfp8(xq, wq, sx, sw)
            y_old, tier = _baseline(xq, sx, wq, sw)
            ref = (x.float() @ w.float().t())
            torch.cuda.synchronize()

            c_new, c_old = _cos(y_new, ref), _cos(y_old, ref)
            assert c_new >= 0.999, \
                f"{tag} M={M}: decode cos {c_new:.6f} < 0.999"
            assert round(c_new, 5) == round(c_old, 5), (
                f"{tag} M={M}: decode cos {c_new:.6f} differs from the shipped "
                f"path's {c_old:.6f} at the fifth decimal")
            diff = (y_new.float() - y_old.float()).abs()
            ulp = _bf16_tolerance(y_old)
            assert bool((diff <= ulp).all()), (
                f"{tag} M={M}: decode output differs from {tier} by more than "
                f"one BF16 ulp (max {diff.max().item():.3e}, "
                f"largest allowed {ulp.max().item():.3e})")
            total += 1
            if bool(torch.equal(y_new, y_old)):
                bit_exact += 1
            kind = "splitk" if cfg[3] > 1 else "persistent"
            print(f"  acc  {tag:>8} M={M:>2}  {kind:<10} split_k={cfg[3]} "
                  f"tile={cfg[0]}  cos={c_new:.6f} (was {c_old:.6f}) "
                  f"maxdiff={diff.max().item():.3e}  vs {tier}  OK")
    print(f"  {bit_exact} of {total} cells bit-identical to the shipped path")


def test_routing() -> None:
    """The decode row's predicate must fire only where it is meant to."""
    from fish_scales_ops.gemm import _sm100_decode, _sm100_dsl, _sm100_smm

    # In the band, on every Family A shape.
    for tag, N, K in SHAPES:
        for M in M_GRID:
            cfg = _sm100_decode.pick_config(M, N, K)
            assert cfg is not None, f"{tag} M={M}: expected a decode tactic"
            tile, cluster, swap_ab, split_k = cfg
            tiles_n = (N + 127) // 128
            assert tile[0] == 128 and tile[1] == (8 if M <= 8 else
                                                  16 if M <= 16 else 32), \
                f"{tag} M={M}: unexpected MMA tile {tile}"
            assert cluster == (1, 1) and swap_ab is True
            if tiles_n <= 32:
                assert split_k in (2, 4), f"{tag} M={M}: split_k {split_k}"
                assert K % (128 * split_k) == 0, \
                    f"{tag} M={M}: split_k {split_k} does not divide K"
                # K ≥ 8192 keeps four slices at every M; a 4096-deep shape
                # drops to two once the token tile widens past M = 8.
                assert split_k == (4 if (K >= 8192 or M <= 8) else 2)
                assert K // split_k >= _sm100_decode.MIN_SLICE_K
            else:
                # Wide N: two slices while the token tile is 8 wide and the
                # doubled grid still fits one wave, one slice above that.
                sms = _sm100_smm._sm_count()
                want = 2 if (M <= 8 and tiles_n * 2 <= sms) else 1
                assert split_k == want, (
                    f"{tag} M={M}: wide-N split {split_k}, expected {want} "
                    f"(tiles_n={tiles_n}, SMs={sms})")
            if split_k > 1:
                # A split is only taken when the resulting grid still fits one
                # wave. One slice has no such constraint: its grid is the tile
                # count, which a wide shape exceeds by construction.
                assert tiles_n * split_k <= _sm100_smm._sm_count(), \
                    f"{tag} M={M}: the split pushes the grid past one wave"
    print("  routing  every Family A decode cell has the expected tactic  OK")

    # A narrow-N shape too shallow to split must fall through to one K slice
    # rather than being forced into a split whose slices are a tile or two
    # deep. The Family C shared-expert `down` projection (N = 2048, K = 512) is
    # the live case: four slices would leave one mainloop K-tile each.
    for M in M_GRID:
        cfg = _sm100_decode.pick_config(M, 2048, 512)
        assert cfg is not None and cfg[3] == 1, (
            f"shared_down M={M}: expected one K slice on a K = 512 shape, "
            f"got {cfg}")
    # ... while the sibling gate_up projection (K = 2048) is deep enough.
    for M in M_GRID:
        cfg = _sm100_decode.pick_config(M, 1024, 2048)
        assert cfg is not None and cfg[3] == (4 if M <= 8 else 2), (
            f"shared_gate_up M={M}: unexpected split {cfg}")
    print("  routing  a K = 512 narrow-N shape falls through to one slice  OK")

    # Out of the band.
    for tag, N, K in SHAPES:
        for M in (33, 48, 64, 128, 256, 1024, 4096):
            assert _sm100_decode.pick_config(M, N, K) is None, \
                f"{tag} M={M}: the decode row took a cell above M = 32"
    print("  routing  nothing above M = 32 is taken  OK")

    # Shapes the kernels cannot serve are declined, not forced.
    assert _sm100_decode.pick_config(1, 2560, 2560 + 64) is None, \
        "a K that is not a multiple of 128 must be declined"
    cfg = _sm100_decode.pick_config(1, 2560, 1152)
    assert cfg is not None and cfg[3] == 1, (
        "K = 1152 divides neither 512 nor 256, so no split depth is valid and "
        f"the cell must fall through to one slice, not be forced: got {cfg}")
    assert _sm100_decode.pick_config(1, 2560 + 64, 4096) is None, \
        "an N that is not a multiple of 128 must be declined"
    print("  routing  invalid tactics decline instead of forcing  OK")

    # The tiers below are untouched: their own predicates still answer the way
    # they did, because the decode row is a separate module and does not edit
    # them. (This is the regression guard for `should_route` / `pick_config`.)
    assert _sm100_smm.should_route(1, 6144, 2560) is True
    assert _sm100_smm.should_route(1, 2560, 9728) is False
    assert _sm100_dsl.pick_config(1, 6144, 2560) is None
    assert _sm100_dsl.pick_config(4096, 19456, 2560) is not None
    print("  routing  tier 1 and tier 2 predicates unchanged  OK")


def test_capture() -> None:
    """Capture must not compile, allocate or synchronise; replay is bit-exact."""
    from fish_scales_ops.gemm import _sm100_decode
    for tag, N, K in (("wo", 2560, 4096), ("gate_up", 19456, 2560)):
        for M in (1, 32):
            x, w, xq0, sx0, wq, sw = _inputs(M, N, K)
            xq, sx = xq0.clone(), sx0.clone()

            def call():
                return fso.gemm.linear_mxfp8(xq, wq, sx, sw)

            for _ in range(5):                      # eager warmup: JIT + pools
                call()
            torch.cuda.synchronize()

            st = _sm100_decode._init()
            assert st is not None
            compiled_before = len(st["compiled"])

            torch.cuda.synchronize()
            a0, r0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
            torch.cuda.set_sync_debug_mode("error")
            try:
                for _ in range(50):
                    call()
            finally:
                torch.cuda.set_sync_debug_mode("default")
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == a0, \
                f"{tag} M={M}: steady-state calls leaked device memory"
            assert torch.cuda.memory_reserved() == r0, \
                f"{tag} M={M}: steady-state calls grew the allocator's reserve"

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    call()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                captured = call()
            torch.cuda.synchronize()
            assert len(st["compiled"]) == compiled_before, \
                f"{tag} M={M}: a kernel compiled inside the capture"

            # Rewrite the activation AND its scale buffer in place, the way a
            # decode loop does, then replay.
            torch.manual_seed(987654 + M)
            x2 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
            xq2, sx2 = fso.gemm.quantize_1x32_fp8(x2)
            xq.copy_(xq2)
            sx.copy_(sx2)
            captured.fill_(float("nan"))
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            replayed = captured.clone()
            assert not bool(torch.isnan(replayed).any()), \
                f"{tag} M={M}: replay did not overwrite the output"
            eager = call()
            torch.cuda.synchronize()
            assert torch.equal(replayed, eager), \
                f"{tag} M={M}: replay after an in-place rewrite is not bit-exact"
            assert len(st["compiled"]) == compiled_before, \
                f"{tag} M={M}: a kernel compiled during replay"
            c = _cos(replayed, x2.float() @ w.float().t())
            assert c >= 0.999, f"{tag} M={M}: replayed cos {c:.6f} < 0.999"
            del g
            print(f"  capture {tag:>8} M={M:>2}  no compile, no alloc, no sync, "
                  f"replay bit-exact, cos={c:.6f}  OK")


def _assert_inert(label: str) -> None:
    """The decode row must be inert: `enabled()` answers False, `pick_config`
    declines every cell, and `route` / `linear_mxfp8` return exactly what the
    shipped tiers return. There are two ways the row is switched off — a DSL
    below the floor and the kill knobs — and both must look like this.
    """
    from fish_scales_ops.gemm import _sm100_decode, _sm100_dispatch
    assert _sm100_decode.enabled() is False, \
        f"{label}: the decode row claims to be enabled"
    for tag, N, K in SHAPES:
        for M in M_GRID:
            assert _sm100_decode.pick_config(M, N, K) is None, \
                f"{label} {tag} M={M}: the decode row took a cell"
    for tag, N, K in SHAPES:
        for M in (1, 32):
            x, w, xq, sx, wq, sw = _inputs(M, N, K)
            y_route = _sm100_dispatch.route(xq, wq, sx, sw)
            y_old, tier = _baseline(xq, sx, wq, sw)
            y_pub = fso.gemm.linear_mxfp8(xq, wq, sx, sw)
            torch.cuda.synchronize()
            if tier == "tier1_cublas_scaled_mm":
                assert y_route is not None and torch.equal(y_route, y_old), (
                    f"{label} {tag} M={M}: the router no longer returns what "
                    "tier 1 returns")
            else:
                # Narrow-N decode is exactly the band where every tier
                # declines and `linear_mxfp8` falls through to the raw C++ op,
                # which is what `route` returning None means.
                assert y_route is None, (
                    f"{label} {tag} M={M}: some tier claimed a cell that used "
                    "to fall through to the C++ cascade")
            assert torch.equal(y_pub, y_old), (
                f"{label} {tag} M={M}: `linear_mxfp8` no longer matches the "
                "shipped tier order")
    print(f"  {label:<8} decode row inert, route bit-identical to the shipped "
          "tiers  OK")


def _old_dsl_child() -> None:
    """Subprocess body: pretend the installed CuTe DSL predates the floor.

    `_sm100_decode._init` reads `cutlass.__version__` before it imports the
    vendored kernels, so setting the attribute before the first call is enough
    to reproduce what an old install does, without needing one.
    """
    import cutlass
    cutlass.__version__ = "4.4.2"
    _assert_inert("old-DSL")


def run_old_dsl_case() -> None:
    env = dict(os.environ)
    env["FSO_LOG"] = "1"
    # This case is about the version floor and nothing else. `_init` checks
    # the kill knobs BEFORE the version and prints no floor notice when one is
    # set, so a knob inherited from the parent (the FSO_DISABLE_DECODE_DSL=1
    # arm of the suite matrix) would shadow exactly what is being tested.
    env.pop("FSO_DISABLE_DSL", None)
    env.pop("FSO_DISABLE_DECODE_DSL", None)
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--old-dsl"],
                       capture_output=True, text=True, env=env)
    sys.stdout.write(r.stdout)
    assert r.returncode == 0, \
        f"old-DSL case failed (exit {r.returncode}):\n{r.stderr[-2000:]}"
    notices = [ln for ln in r.stdout.splitlines()
               if "nvidia-cutlass-dsl" in ln and ln.startswith("fso: ")]
    assert len(notices) == 1, (
        "expected exactly one FSO_LOG notice naming the DSL floor, got "
        f"{len(notices)}:\n{r.stdout}")
    print("  old-DSL  FSO_LOG printed the reason exactly once  OK")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--old-dsl":
        _old_dsl_child()
        return 0

    assert torch.cuda.is_available(), "CUDA required"
    sm = torch.cuda.get_device_capability(0)
    print(f"Device: {torch.cuda.get_device_name(0)} sm_{sm[0]}{sm[1]}\n")
    if sm[0] != 10:
        print("The MXFP8 decode row is sm_100/sm_103-only — skipping.")
        return 0

    from fish_scales_ops.gemm import _sm100_decode
    if not _sm100_decode.enabled():
        knob = next((k for k in ("FSO_DISABLE_DECODE_DSL", "FSO_DISABLE_DSL")
                     if os.getenv(k)), None)
        if knob is not None:
            # The row is off because of the knob, not the DSL. Check that the
            # knob makes it inert in the same way, then run the version-gate
            # case with the knob cleared so the floor itself is exercised.
            print(f"{knob} is set — the decode row is switched off by the "
                  "knob, so only the inertness cases can run.\n")
            print("== knob-off inertness (this process) ==")
            _assert_inert("knob-off")
            print("\n== version gate (subprocess with a mocked old DSL, knob "
                  "cleared) ==")
            run_old_dsl_case()
            print("\nAll sm_100 MXFP8 decode-row inertness cases passed.")
            return 0
        print("nvidia-cutlass-dsl is missing or below 4.5.0 — the decode row "
              "is inert here, so only the inertness case can run.")
        run_old_dsl_case()
        return 0
    print(f"CuTe DSL: {_sm100_decode._init()['dsl_version']}\n")

    print("== accuracy vs the shipped path ==")
    test_accuracy()
    print("\n== routing ==")
    test_routing()
    print("\n== CUDA-graph capture ==")
    test_capture()
    print("\n== version gate (subprocess with a mocked old DSL) ==")
    run_old_dsl_case()
    print("\nAll sm_100 MXFP8 decode-row tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
