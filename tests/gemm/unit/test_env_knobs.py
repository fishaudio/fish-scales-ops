"""Boolean FSO_* switches read in Python, and the sm_100/103 CuTe-DSL tier notices.

1. **One parsing rule.** The extension reads a boolean switch as off when it is
   unset, empty or starts with ``0`` (``read_print_tile_info`` in
   csrc/gemm/include/blockscale_gemm/arch/sm100/mxfp8/dispatch.cuh). The Python
   package used to read its switches as "any non-empty value", so
   ``FSO_DISABLE_DSL=0`` disabled the tier it names. Every boolean switch the
   package reads now goes through ``fish_scales_ops._env.env_flag``; this file
   checks the helper's table and, in child processes, that ``=0`` really means
   off for each switch (the sm_100/103 tier switches on sm_100/103, the two MoE
   switches anywhere).

2. **No silent tier loss (sm_100/103).** The mid-band CuTe-DSL tier compiles a
   CUTLASS example it finds in the source tree or at ``FSO_DSL_KERNEL_PATH``.
   When that file is missing the tier used to vanish without a word. It must now
   print exactly one ``fso:`` line on stderr per process naming the tier, the
   reason and ``FSO_DSL_KERNEL_PATH``, while the router still returns the right
   result from the remaining tiers; in the normal layout it must stay silent and
   load. The decode row's equivalent: an nvidia-cutlass-dsl below its 4.5.0
   floor must produce one ``fso:`` line without ``FSO_LOG`` (the row used to
   print it only under ``FSO_LOG=1``).

Run: ``PYTHONPATH=python python tests/gemm/unit/test_env_knobs.py``.
"""
from __future__ import annotations

import os
import subprocess
import sys

import torch

MID_TAG = "fso: sm_100/103 CuTe-DSL mid-band tier unavailable"
DECODE_TAG = "fso: sm_100/103 CuTe-DSL decode row"


def test_env_flag_table():
    from fish_scales_ops._env import env_flag

    name = "FSO__TEST_ENV_FLAG"
    table = [(None, False), ("", False), ("0", False), ("00", False), ("0x1", False),
             ("1", True), ("2", True), ("true", True), ("yes", True), ("on", True)]
    saved = os.environ.pop(name, None)
    try:
        for raw, want in table:
            if raw is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = raw
            assert env_flag(name) is want, f"env_flag({raw!r}) = {env_flag(name)}, want {want}"
        os.environ.pop(name, None)
        assert env_flag(name, default=True) is True
        os.environ[name] = ""
        assert env_flag(name, default=True) is True
        os.environ[name] = "0"
        assert env_flag(name, default=True) is False
    finally:
        os.environ.pop(name, None)
        if saved is not None:
            os.environ[name] = saved
    print("  env_flag          unset/empty/'0…' off, anything else on, default honoured  OK")


# --------------------------------------------------------------------------- #
# child-process bodies
# --------------------------------------------------------------------------- #
def _child_sm100_knobs():
    """Print what each sm_100/103 tier switch resolved to in this process."""
    from fish_scales_ops.gemm import _sm100_decode, _sm100_dsl, _sm100_smm
    print("SMM", int(bool(_sm100_smm._init())))
    print("DSL", int(_sm100_dsl._init() is not None))
    print("DECODE", int(_sm100_decode._init() is not None))
    # FSO_PRINT_TILE_INFO: the decode row prints its pick on stderr when it is on.
    _sm100_decode.pick_config(1, 2560, 4096)


def _child_moe_knobs():
    from fish_scales_ops import moe
    from fish_scales_ops.gemm import mxfp8
    moe._fused_combine_allowed()
    print("FUSED_COMBINE", int(bool(moe._fused_combine_env)))
    print("BLOCK_OVERLAP", int(bool(mxfp8._moe_block_overlap_enabled())))


def _child_mid_band():
    """Call the mid-band tier's init twice, then run a GEMM the tier would own."""
    import fish_scales_ops as fso
    from fish_scales_ops.gemm import _sm100_dsl
    st1 = _sm100_dsl._init()
    st2 = _sm100_dsl._init()
    print("STATE", int(st1 is not None), int(st2 is not None))
    M = N = K = 4096                       # a cubic cell the tier's table names
    cfg = _sm100_dsl.pick_config(M, N, K)
    print("PICKED", int(cfg is not None))
    torch.manual_seed(5)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.5)
    xq, sx = fso.compat.quantize_1x32_fp8(x)
    wq, sw = fso.compat.quantize_1x32_fp8(w)
    y = fso.compat.linear_mxfp8(xq, wq, sx, sw)
    y2 = fso.compat.linear_mxfp8(xq, wq, sx, sw)
    ref = x.float() @ w.float().t()
    cos = torch.nn.functional.cosine_similarity(y.float().flatten(), ref.flatten(), dim=0).item()
    print("COS", f"{cos:.6f}", int(torch.equal(y, y2)))


def _child_old_decode_dsl():
    import cutlass
    cutlass.__version__ = "4.4.2"
    from fish_scales_ops.gemm import _sm100_decode
    print("DECODE", int(_sm100_decode._init() is not None), int(_sm100_decode.enabled()))


CHILDREN = {
    "sm100_knobs": _child_sm100_knobs,
    "moe_knobs": _child_moe_knobs,
    "mid_band": _child_mid_band,
    "old_decode_dsl": _child_old_decode_dsl,
}


def _run(case, **env_set):
    env = {k: v for k, v in os.environ.items() if not k.startswith("FSO_")}
    env.update(env_set)
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", case],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"child {case} {env_set} failed (exit {r.returncode}):\n{r.stderr[-3000:]}"
    vals = {}
    for ln in r.stdout.splitlines():
        parts = ln.split()
        if parts and parts[0].isupper():
            vals[parts[0]] = parts[1:]
    return vals, r.stderr


def test_moe_knobs():
    base, _ = _run("moe_knobs")
    assert base["FUSED_COMBINE"] == ["0"] and base["BLOCK_OVERLAP"] == ["1"], base
    for raw, fused, overlap in (("0", "0", "0"), ("1", "1", "1"), ("true", "1", "1"), ("00", "0", "0")):
        v, _ = _run("moe_knobs", FSO_MOE_FUSED_COMBINE=raw, FSO_MOE_BLOCK_OVERLAP=raw)
        assert v["FUSED_COMBINE"] == [fused] and v["BLOCK_OVERLAP"] == [overlap], (raw, v)
    print("  moe switches      FSO_MOE_FUSED_COMBINE / FSO_MOE_BLOCK_OVERLAP follow env_flag  OK")


def test_sm100_tier_knobs():
    base, _ = _run("sm100_knobs")
    print(f"  sm100 tiers       default: smm={base['SMM'][0]} dsl={base['DSL'][0]} "
          f"decode={base['DECODE'][0]}")
    for knob, keys in (("FSO_DISABLE_SMM", ("SMM",)), ("FSO_DISABLE_DSL", ("DSL", "DECODE")),
                       ("FSO_DISABLE_DECODE_DSL", ("DECODE",))):
        off, _ = _run("sm100_knobs", **{knob: "1"})
        zero, _ = _run("sm100_knobs", **{knob: "0"})
        for key in keys:
            assert off[key] == ["0"], f"{knob}=1 left {key} on"
            assert zero[key] == base[key], f"{knob}=0 changed {key} ({base[key]} -> {zero[key]})"
    if base["DECODE"] == ["1"]:
        _, err_on = _run("sm100_knobs", FSO_PRINT_TILE_INFO="1")
        _, err_zero = _run("sm100_knobs", FSO_PRINT_TILE_INFO="0")
        assert "[fso sm100 decode]" in err_on, err_on[-1000:]
        assert "[fso sm100 decode]" not in err_zero, "FSO_PRINT_TILE_INFO=0 still printed"
    print("  sm100 tiers       =1 switches each tier off, =0 leaves it as unset  OK")


def test_mid_band_notice():
    # The default kernel is the copy a wheel carries in _dsl/, else the CUTLASS example in the source tree;
    # _default_kernel_path() is the tier's own answer.
    from fish_scales_ops.gemm import _sm100_dsl
    default_present = os.path.exists(_sm100_dsl._default_kernel_path())

    v, err = _run("mid_band", FSO_DSL_KERNEL_PATH="/nonexistent/dense_blockscaled_gemm_persistent.py")
    lines = [ln for ln in err.splitlines() if ln.startswith(MID_TAG)]
    assert v["STATE"] == ["0", "0"], v
    assert len(lines) == 1, f"expected one mid-band notice, got {len(lines)}:\n{err[-2000:]}"
    assert "FSO_DSL_KERNEL_PATH" in lines[0] and "/nonexistent/" in lines[0], lines[0]
    assert float(v["COS"][0]) > 0.99 and v["COS"][1] == "1", v
    print(f"  mid-band missing  one notice on stderr, fallback result cos={v['COS'][0]}  OK")

    v, err = _run("mid_band")
    lines = [ln for ln in err.splitlines() if ln.startswith(MID_TAG)]
    if default_present:
        assert v["STATE"] == ["1", "1"] and v["PICKED"] == ["1"], v
        assert not lines, f"the tier loaded and still printed a notice:\n{err[-2000:]}"
        assert float(v["COS"][0]) > 0.99 and v["COS"][1] == "1", v
        print(f"  mid-band present  tier loaded silently, cos={v['COS'][0]}  OK")
    else:
        assert v["STATE"] == ["0", "0"] and len(lines) == 1, (v, err[-2000:])
        print("  mid-band present  (no kernel file in this layout) one notice  OK")


def test_old_decode_dsl_notice():
    v, err = _run("old_decode_dsl")
    lines = [ln for ln in err.splitlines() if ln.startswith(DECODE_TAG)]
    assert v["DECODE"] == ["0", "0"], v
    assert len(lines) == 1 and "nvidia-cutlass-dsl" in lines[0], \
        f"expected one decode-row notice without FSO_LOG, got {len(lines)}:\n{err[-2000:]}"
    print("  decode row        DSL below the floor: one notice without FSO_LOG  OK")


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        CHILDREN[sys.argv[2]]()
        return 0
    cases = [("env_flag", test_env_flag_table)]
    if not torch.cuda.is_available():
        print("no CUDA device: child-process cases skipped")
    else:
        sm = torch.cuda.get_device_capability(0)
        print(f"Device: {torch.cuda.get_device_name(0)} sm_{sm[0]}{sm[1]}")
        cases.append(("MoE switches", test_moe_knobs))
        try:
            import cutlass  # noqa: F401
            have_dsl = True
        except ImportError:
            have_dsl = False
        if sm[0] != 10:
            print("sm_100/103 tier cases skipped on this architecture")
        elif not have_dsl:
            print("nvidia-cutlass-dsl not installed: the DSL tier cases are skipped")
        else:
            cases += [("sm_100/103 tier switches", test_sm100_tier_knobs),
                      ("sm_100/103 mid-band tier notice", test_mid_band_notice),
                      ("sm_100/103 decode row notice", test_old_decode_dsl_notice)]
    failed = []
    for title, fn in cases:
        print(f"\n== {title} ==")
        try:
            fn()
        except Exception as exc:            # report every case, not just the first
            failed.append(title)
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[-1500:]}")
    if failed:
        print(f"\nenv knobs: FAILED: {', '.join(failed)}")
        return 1
    print("\nenv knobs: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
