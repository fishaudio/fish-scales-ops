#!/usr/bin/env python3
"""The public surface of fish_scales_ops, pinned: the exported names of every
namespace, the deprecated fso.gemm path of the 45 names that now live in
fso.compat, the removal of fso.moe.lowlevel, the torch ops the two stable
entries register, the arguments removed on 2026-09-30, and the architecture
refusals of the entries that exist on some architectures only. Runs on any
machine, a CPU-only one included; the refusal checks that need a particular
device say when they are skipped.

The deprecation checks run first, because each fso.gemm name warns only on its
first lookup in a process: nothing before them may touch fso.gemm.<name>."""
import importlib
import inspect
import os
import re
import subprocess
import sys
import warnings

import torch
import fish_scales_ops as fso
from fish_scales_ops._arch import sm_major

TOP = ["dense", "moe", "attention", "compat"]
DENSE = {"FORMATS", "DenseWeight", "prepare_weight", "linear", "supported", "describe"}
MOE = {"FORMATS", "MoeExperts", "prepare_experts", "layer", "transient_bytes", "supported",
       "describe"}
# The eleven explicit dense ops, in the order of fso.gemm's former __all__.
COMPAT_DENSE = ["quantize_1x128_fp8", "quantize_1x128_fp8_packed", "quantize_128x128_fp8",
                "repack_fp8_act_scales", "repack_fp8_wgt_scales", "linear_fp8", "linear_qx",
                "quantize_1x32_fp8", "silu_chunk_mul_quantize_1x32_fp8", "linear_mxfp8", "linear_bf16"]
COMPAT_MOE = {
    "moe_layer_fp8_sm90", "moe_layer_mxfp8_sm120", "moe_block_mxfp8_sm120",
    "moe_layer_transient_bytes_sm90", "moe_layer_transient_bytes_mxfp8",
    "moe_layer_fused_combine_engages_sm120", "moe_router_topk", "moe_topk_from_logits",
    "moe_build_routing", "moe_build_sorted", "moe_combine", "moe_combine_sorted",
    "linear_fp8_grouped_masked", "linear_fp8_grouped_contiguous",
    "linear_fp8_grouped_contiguous_swapab", "quantize_1x128_grouped_gather_sm90",
    "quantize_1x128_sorted_gather_sm90", "silu_chunk_mul_quantize_1x128_grouped_sm90",
    "silu_chunk_mul_quantize_1x128_sorted_sm90", "quantize_moe_weights_1x128_fp8_sm90",
    "MOE_SWAP_BLOCK_N_CASCADE", "moe_swap_ab_block_n", "moe_swap_ab_max_m",
    "linear_mxfp8_grouped_masked", "linear_mxfp8_grouped_masked_swiglu",
    "linear_mxfp8_grouped_masked_combine", "quantize_1x32_grouped_gather_fp8",
    "silu_chunk_mul_quantize_1x32_grouped_fp8", "quantize_moe_weights_1x32_fp8",
    "interleave_w13_fp8", "mxfp8_grouped_swiglu_fused_route", "mxfp8_grouped_swiglu_available",
    "mxfp8_grouped_slot_possible", "mxfp8_grouped_problem_shapes_consumed"}
COMPAT = set(COMPAT_DENSE) | COMPAT_MOE
ATTENTION = {"flash_attn_fwd", "FlashAttnDispatch", "mxfp8_fwd", "pre_quantize_q",
             "pre_quantize_k", "pre_quantize_v", "mxfp8_paged_prefill_fwd", "plan_paged_prefill",
             "PrefillPagedPlan", "quantize_q_ragged", "mxfp8_decode_paged_fwd", "plan_decode_paged",
             "DecodePagedPlan", "quantize_q_grouped", "compute_k_chan_scale",
             "compute_v_chan_scale", "quantize_kv_to_paged"}
# The existing raw ops serving stacks register their own fake implementations for
# (apex's _fake_ops.py does exactly these). fish_scales_ops must register none of its
# own for them, or the caller's registration raises.
CALLER_FAKED = ["linear_qx", "linear_fp8", "linear_mxfp8_raw", "quantize_1x128", "quantize_1x32",
                "repack_mxfp8_scales", "repack_fp8_act_scales", "repack_fp8_wgt_scales"]


def check(cond, what, failures):
    print(f"  {'OK  ' if cond else 'FAIL'} {what}")
    if not cond:
        failures.append(what)


def _first_lookup(name, use_import):
    """Look ``name`` up on fso.gemm for the first time; return (object, warnings)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if use_import:
            ns = {}
            exec(f"from fish_scales_ops.gemm import {name}", ns)
            obj = ns[name]
        else:
            obj = getattr(fso.gemm, name)
    return obj, [x for x in w if issubclass(x.category, DeprecationWarning)]


def deprecation_checks(failures):
    print("== fso.gemm: the deprecated path of the 45 compat names ==")
    assert COMPAT_MOE.isdisjoint(COMPAT_DENSE) and len(COMPAT) == 45
    wrong = []
    for i, name in enumerate(sorted(COMPAT)):
        use_import = i % 2 == 1  # every other name through `from fish_scales_ops.gemm import name`
        obj, dep = _first_lookup(name, use_import)
        msg = str(dep[0].message).replace("fish_scales_ops", "fso") if len(dep) == 1 else ""
        good = (len(dep) == 1 and obj is getattr(fso.compat, name) and f"fso.compat.{name}" in msg
                and "0.3.0" in msg and (("fso.dense" in msg) if name in COMPAT_DENSE else ("fso.moe" in msg)))
        _, again = _first_lookup(name, not use_import)
        if not good or again:
            wrong.append(f"{name} ({len(dep)} warnings first, {len(again)} after)")
    check(not wrong, f"each of the {len(COMPAT)} names resolves to the fso.compat object with exactly one "
          f"DeprecationWarning naming fso.compat.<name>, fso.dense or fso.moe and 0.3.0, and none on a "
          f"second lookup, half of them through `from ... import`" + (f" (wrong: {wrong})" if wrong else ""),
          failures)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ns = {}
        exec("from fish_scales_ops.gemm import *", ns)
    star = sorted(k for k in ns if not k.startswith("__"))
    check(star == sorted(COMPAT_DENSE) and not w,
          f"`from fish_scales_ops.gemm import *` still imports the eleven dense names "
          f"({len(star)} names, {len(w)} new warnings)", failures)
    check(fso.gemm.__all__ == COMPAT_DENSE, "fso.gemm.__all__ is the eleven dense names, in their former order",
          failures)
    check(COMPAT <= set(dir(fso.gemm)), "dir(fso.gemm) lists the 45 deprecated names", failures)
    try:
        fso.gemm.no_such_name
        check(False, "an unknown fso.gemm name raises AttributeError", failures)
    except AttributeError:
        check(True, "an unknown fso.gemm name raises AttributeError", failures)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mods = [importlib.import_module(f"fish_scales_ops.gemm.{m}") for m in ("fp8", "mxfp8", "bf16")]
        same = fso.gemm.fp8 is mods[0]
    check(same and not [x for x in w if issubclass(x.category, DeprecationWarning)],
          "the implementation modules fish_scales_ops.gemm.{fp8,mxfp8,bf16} import without a warning", failures)


def main():
    failures = []
    major = sm_major()
    print(f"device: {'sm_%dx' % major if major else 'no CUDA device'}; torch {torch.__version__}")

    deprecation_checks(failures)

    print("== exported names ==")
    check(fso.__all__ == TOP, f"fish_scales_ops.__all__ == {TOP}", failures)
    check(hasattr(fso, "gemm") and "gemm" not in fso.__all__,
          "fso.gemm is still an attribute of the package and is not in __all__", failures)
    pyproject = os.path.join(os.path.dirname(os.path.abspath(fso.__file__)), os.pardir, "pyproject.toml")
    if os.path.exists(pyproject):
        text = open(pyproject).read()
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
        check(m is not None and fso.__version__ == m.group(1),
              f"fish_scales_ops.__version__ {fso.__version__} matches python/pyproject.toml "
              f"({m.group(1) if m else 'no version line'})", failures)
        for pkg in ("fish_scales_ops.dense", "fish_scales_ops.compat"):
            check(f'"{pkg}"' in text, f"python/pyproject.toml packages {pkg}", failures)
        check('"fish_scales_ops.moe.lowlevel"' not in text, "python/pyproject.toml does not list lowlevel",
              failures)
    else:
        print(f"  skip: no pyproject.toml next to the package (installed copy); __version__ {fso.__version__}")
    for mod, want in ((fso.dense, DENSE), (fso.moe, MOE), (fso.compat, COMPAT), (fso.attention, ATTENTION)):
        got = set(mod.__all__)
        check(got == want and len(mod.__all__) == len(got),
              f"{mod.__name__}.__all__ ({len(got)} names)"
              + ("" if got == want else f": extra {sorted(got - want)}, missing {sorted(want - got)}"),
              failures)
        missing = [n for n in mod.__all__ if not hasattr(mod, n)]
        check(not missing, f"{mod.__name__}: every exported name resolves", failures)
    check(fso.compat.__all__[:11] == COMPAT_DENSE, "fso.compat lists the dense ops first, in their former order",
          failures)

    print("== fso.moe.lowlevel is gone ==")
    check(not hasattr(fso.moe, "lowlevel") and "lowlevel" not in dir(fso.moe), "fso.moe has no lowlevel attribute",
          failures)
    try:
        importlib.import_module("fish_scales_ops.moe.lowlevel")
        check(False, "import fish_scales_ops.moe.lowlevel raises ModuleNotFoundError", failures)
    except ModuleNotFoundError:
        check(True, "import fish_scales_ops.moe.lowlevel raises ModuleNotFoundError", failures)

    print("== importing the package raises no DeprecationWarning ==")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""  # a non-GPU process: a card in exclusive-process mode cannot refuse it
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(fso.__file__)))
    env["PYTHONPATH"] = pkg_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    r = subprocess.run([sys.executable, "-W", "error::DeprecationWarning", "-c",
                        "import fish_scales_ops as f; assert f.__file__.startswith(%r)" % pkg_parent],
                       env=env, capture_output=True, text=True, timeout=600)
    check(r.returncode == 0, "python -W error::DeprecationWarning -c 'import fish_scales_ops' (CPU-only process)"
          + ("" if r.returncode == 0 else f": exit {r.returncode}, {r.stderr.strip().splitlines()[-1:]}"),
          failures)

    print("== torch ops ==")
    ops = torch.ops.fish_scales_ops
    for name in ("dense_linear", "moe_layer"):
        try:
            getattr(ops, name)
            ok = True
        except (AttributeError, RuntimeError):
            ok = False
        check(ok, f"torch.ops.fish_scales_ops.{name} is registered", failures)
    schema = str(ops.dense_linear.default._schema)
    check(schema == "fish_scales_ops::dense_linear(Tensor x, Tensor weight, Tensor scale, str kind) -> Tensor",
          f"dense_linear schema: {schema}", failures)

    print("== arguments removed on 2026-09-30 ==")
    for fn, gone in ((fso.compat.moe_layer_mxfp8_sm120, ("on_output", "chunk_tokens")),
                     (fso.compat.moe_block_mxfp8_sm120, ("on_output", "chunk_tokens")),
                     (fso.attention.flash_attn_fwd, ("force_kernel",))):
        params = inspect.signature(fn).parameters
        check(not any(g in params for g in gone), f"{fn.__name__} has no {', '.join(gone)}", failures)
    check(inspect.signature(fso.compat.quantize_1x128_fp8).parameters["use_ue8m0"].default is None,
          "quantize_1x128_fp8 resolves use_ue8m0 from the architecture by default", failures)

    print("== architecture refusals ==")
    if major != 12:
        for name, call in (
                ("mxfp8_fwd", lambda: fso.attention.mxfp8_fwd(None, None, None, None, None, None)),
                ("mxfp8_decode_paged_fwd", lambda: fso.attention.mxfp8_decode_paged_fwd(
                    None, None, None, None, None, None, None, None)),
                ("mxfp8_paged_prefill_fwd", lambda: fso.attention.mxfp8_paged_prefill_fwd(
                    None, None, None, None, None, None, None, None, None, None)),
                ("plan_decode_paged", lambda: fso.attention.plan_decode_paged(
                    B=1, H_q=8, H_kv=1, D=128, max_blocks=4)),
                ("plan_paged_prefill", lambda: fso.attention.plan_paged_prefill(
                    qo_indptr_cpu=torch.tensor([0, 16], dtype=torch.int32), num_q_heads=8))):
            try:
                call()
                ok = False
            except NotImplementedError as e:
                ok = "flash_attn_fwd" in str(e)
            check(ok, f"attention.{name} refuses sm_{major}x and names flash_attn_fwd", failures)
    else:
        print("  skip: the MXFP8 attention entries run on this device")
    if major and major < 10:
        for name, call in (("repack_fp8_act_scales", lambda: fso.compat.repack_fp8_act_scales(torch.ones(4, 1))),
                           ("repack_fp8_wgt_scales", lambda: fso.compat.repack_fp8_wgt_scales(torch.ones(1, 1))),
                           ("quantize_1x128_fp8_packed", lambda: fso.compat.quantize_1x128_fp8_packed(
                               torch.ones(4, 128, dtype=torch.bfloat16)))):
            try:
                call()
                ok = False
            except NotImplementedError:
                ok = True
            check(ok, f"compat.{name} refuses sm_{major}x", failures)
        try:
            fso.compat.linear_fp8(torch.empty(1, 128), torch.empty(1, 128),
                                  torch.ones(4, 1, dtype=torch.int32), torch.ones(1, 1, dtype=torch.int32))
            ok = False
        except ValueError:
            ok = True
        check(ok, "compat.linear_fp8 on sm_90 refuses int32 pre-packed scales", failures)
    else:
        print("  skip: the Blackwell scale layout is this device's own (or there is no device)")

    # A caller that registers its own fakes for the raw ops it calls (apex does, for
    # CALLER_FAKED) must not meet a registration of fish_scales_ops. Recent torch lets a
    # second register_fake override the first, so the check reads the registrations
    # instead: only the two Python custom ops, dense_linear and moe_layer, may carry a
    # Meta kernel or a fake, and they must (the positive control of the check).
    print("== fake implementations: only the two Python custom ops ==")
    names = sorted(n for n in torch._C._dispatch_get_all_op_names() if n.startswith("fish_scales_ops::"))
    with_meta = [n for n in names if torch._C._dispatch_has_kernel_for_dispatch_key(n, "Meta")]
    try:
        from torch._library import simple_registry
        with_fake = [n for n in names if simple_registry.singleton.find(n).fake_impl.kernel is not None]
    except Exception as e:  # noqa: BLE001 - an internal registry that moved is reported, not fatal
        print(f"  note: torch's fake registry is not readable here ({type(e).__name__}); Meta kernels only")
        with_fake = with_meta
    want = ["fish_scales_ops::dense_linear", "fish_scales_ops::moe_layer"]
    check(with_meta == want and with_fake == want and all(f"fish_scales_ops::{n}" in names for n in CALLER_FAKED),
          f"of the {len(names)} fish_scales_ops ops only dense_linear and moe_layer carry a Meta kernel or a "
          f"fake, so a caller's fakes for {', '.join(CALLER_FAKED)} meet no registration of ours"
          + ("" if with_meta == want and with_fake == want else f" (Meta: {with_meta}, fake: {with_fake})"),
          failures)

    print("public surface: " + ("ALL PASS" if not failures else f"{len(failures)} FAILED"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
