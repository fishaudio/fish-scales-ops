"""``fish_scales_ops.moe`` — one MoE layer surface for every architecture.

A serving stack holds the routed experts of its MoE layers in one of two
checkpoint dialects and needs two things from this library: a load-time
preparation of the rank's local experts on the device that will serve them, and a
per-call layer that runs those experts for the ids the router chose. The
architecture dispatch lives inside the library, in a torch custom op, so the
caller writes one code path and never names an architecture::

    import fish_scales_ops as fso

    experts = fso.moe.prepare_experts(w13, w2, format="bsfp8", sw13=sw13, sw2=sw2)  # load time
    out = fso.moe.layer(hidden, experts, topk_ids, topk_w)                          # per call
    fso.moe.supported("bsfp8")    # can this device serve the format?
    print(fso.moe.describe())     # the architecture matrix, for logs and errors
    reserve = fso.moe.transient_bytes(experts, max_tokens_per_forward, topk)       # memory to reserve

``format`` names what the caller holds. ``"bsfp8"`` is float8_e4m3fn experts with
fp32 128x128 block scales (the ``bsgemm-moe`` dialect); sm_90 serves them as they
are through the expert-sorted block-FP8 layer, and sm_100/sm_103 and sm_120/sm_121
dequantize them block by block at load time and requantize them to MXFP8 1x32,
because the grouped MXFP8 kernels are the only grouped path those architectures
have. ``"mxfp8"`` is either bf16 experts (the ``mxfp8`` dialect that keeps its experts
bf16 in the checkpoint), quantized to MXFP8 1x32 at load time, or an MXFP8 checkpoint
as it is stored: float8_e4m3fn experts with uint8 UE8M0 scales, one byte per 32
elements along K, whose bytes are kept and whose scales are only repacked into this
device's layout. Both are served on sm_100/sm_103 and sm_120/sm_121; sm_90 has no
MXFP8 hardware and refuses the format. Every other format, bf16 experts included,
raises: there is no fallback of any kind.

Every bucket runs as one layer call. The transient memory a call needs is the
caller's to reserve, and :func:`transient_bytes` gives the exact figure for the
largest bucket a forward can carry on this device.

This module is the stable MoE interface. The per-step pieces behind
:func:`layer` — the per-architecture layer entries, the grouped GEMMs, the
routing builders, the combines, the router and the route queries — are in
:mod:`fish_scales_ops.compat`. Their names keep working there, but they are
building blocks whose arguments follow the kernels and change when a kernel or
a route changes.
"""
import dataclasses

import torch

from .._arch import sm_major
from .._env import env_flag
from ..gemm.fp8 import moe_layer_fp8_sm90, moe_layer_transient_bytes_sm90
from ..gemm.mxfp8 import (
    _moe_layer_mxfp8,
    _ue8m0_scale_handle,
    _w13_interleave_perm,
    moe_layer_transient_bytes_mxfp8,
    mxfp8_grouped_swiglu_available,
    quantize_moe_weights_1x32_fp8,
)

__all__ =["FORMATS", "MoeExperts", "prepare_experts", "layer", "transient_bytes", "supported",
           "describe"]

# The formats prepare_experts accepts, and the architecture majors that serve each.
FORMATS = ("bsfp8", "mxfp8")
_SERVED = {"bsfp8": (9, 10, 12), "mxfp8": (10, 12)}

# Checkpoint dialects that exist but that no architecture serves through this
# module. They are refused with NotImplementedError, and anything not listed here
# or in FORMATS with ValueError, so a caller can tell "not served" from "misspelt".
_UNSERVED = ("bf16", "bfloat16", "fp16", "float16", "fp32", "float32", "fp8", "int8", "w8a8",
             "int4", "w4a16", "w4a8", "mxfp4", "nvfp4")

# Largest fp32 scratch the load-time conversion holds at once, in bytes: the
# experts are dequantized and requantized a few at a time so that preparing a
# layer never needs a second full copy of its weights.
_CONVERT_BUDGET_BYTES = 256 * 1024 * 1024


_device_arch_cache: int = -1


def _device_arch() -> int:
    """Compute capability of device 0 as ``major * 10 + minor`` (90, 100, 103, 120,
    121), 0 without a CUDA device; resolved once, like :func:`sm_major`, whose
    device it describes."""
    global _device_arch_cache
    if _device_arch_cache < 0:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            _device_arch_cache = major * 10 + minor
        else:
            _device_arch_cache = 0
    return _device_arch_cache


def _arch_label(arch: int) -> str:
    return f"sm_{arch}" if arch > 0 else "no CUDA device"


def _sm90_jit_compiler_line() -> str:
    """The line that :func:`describe` and :func:`fish_scales_ops.dense.describe` add
    on sm_90: the compiler of the deep_gemm JIT in this process
    (``torch.ops.fish_scales_ops.jit_compiler_sm90``), which loads the bundled NVRTC
    on first use, or the error that keeps it from loading. Never raises."""
    try:
        compiler = torch.ops.fish_scales_ops.jit_compiler_sm90()
    except Exception as e:  # noqa: BLE001 - the description must never fail
        compiler = "unavailable: " + " ".join(str(e).splitlines())
    return f"sm_90 JIT compiler: {compiler}"


def _build_line() -> str:
    """The line that :func:`describe` and :func:`fish_scales_ops.dense.describe`
    add after their first: the package version, the commit and the torch the
    extension was built against (:func:`fish_scales_ops.build_info`), or, for an
    in-place build, the running torch. Never raises."""
    try:
        from .. import build_info

        info = build_info()
        if info.get("source_build"):
            return (f"build: fish-scales-ops {info.get('version')}, an in-place build of the source tree, "
                    f"running torch {info.get('torch')}")
        commit = str(info.get("commit") or "unknown")[:12] + (" (dirty)" if info.get("dirty") else "")
        return (f"build: fish-scales-ops {info.get('version')}, commit {commit}, built against torch "
                f"{info.get('torch')}")
    except Exception as e:  # noqa: BLE001 - the description must never fail
        return "build: unavailable: " + " ".join(str(e).splitlines())


def _arch_major(arch) -> int:
    """The compute-capability major of an ``arch`` argument: an int major (9, 10,
    12), an int ``major * 10 + minor`` (90, 100, 103, 120, 121), a ``(major, minor)``
    pair, or a string such as ``"sm_90"``, ``"sm_100a"``, ``"sm_103"``, ``"120"`` or
    ``"12.0"``."""
    if isinstance(arch, bool):
        raise ValueError(f"arch must name a compute capability, got {arch!r}")
    if isinstance(arch, (tuple, list)) and len(arch) == 2 and all(isinstance(v, int) for v in arch):
        return int(arch[0])
    if isinstance(arch, str):
        s = arch.strip().lower()
        if s.startswith("sm_"):
            s = s[3:]
        elif s.startswith("sm"):
            s = s[2:]
        if "." in s:
            head = s.split(".", 1)[0]
            if head.isdigit():
                return int(head)
        s = s.rstrip("af")
        if not s.isdigit():
            raise ValueError(f"arch {arch!r} does not name a compute capability (e.g. 'sm_90', 'sm_120')")
        arch = int(s)
    if isinstance(arch, int):
        if 0 < arch < 20:
            return arch
        if 20 <= arch < 1000:
            return arch // 10
    raise ValueError(f"arch {arch!r} does not name a compute capability (e.g. 'sm_90', 'sm_120')")


def supported(format, arch=None) -> bool:
    """Whether ``format`` can be served on ``arch`` (this device when ``None``).

    This is the capability query a model-level resolver asks before it picks this
    layer: ``"bsfp8"`` is served on sm_90, sm_100/sm_103 and sm_120/sm_121,
    ``"mxfp8"`` on sm_100/sm_103 and sm_120/sm_121, and nothing else anywhere. It
    answers from the architecture matrix alone and never raises for a format it
    does not know; an ``arch`` that names no compute capability raises
    ``ValueError``. Without a CUDA device it answers ``False`` for every format.
    """
    if not isinstance(format, str):
        return False
    majors = _SERVED.get(format.strip().lower())
    if majors is None:
        return False
    major = sm_major() if arch is None else _arch_major(arch)
    return major in majors


def describe() -> str:
    """The architecture matrix as text, with this device's rows marked, for logs
    and for the messages of a caller that refuses a configuration. The second
    line names the build (version, commit and the torch the extension was built
    against, from :func:`fish_scales_ops.build_info`). On sm_90 the last line
    names the compiler of the deep_gemm JIT (the bundled NVRTC and its path), or
    the error that keeps it from loading; describe() itself never raises."""
    arch = _device_arch()
    major = arch // 10
    name = ""
    if arch > 0:
        try:
            name = f" ({torch.cuda.get_device_name(0)})"
        except Exception:  # noqa: BLE001 - the description must never fail
            name = ""
    families = (
        (9, "sm_90 (H200)"),
        (10, "sm_100/sm_103 (B200, B300)"),
        (12, "sm_120/sm_121 (RTX 5090, RTX PRO 6000)"),
    )
    rows = {
        "bsfp8": {
            9: "served: the checkpoint's block-FP8 tensors are kept as they are and run "
               "through the expert-sorted block-FP8 layer (moe_layer_fp8_sm90)",
            10: "served: dequantized block by block at load time, requantized to MXFP8 1x32 "
                "with the FC1 rows interleaved, run through the masked-slab MXFP8 layer",
            12: "served: dequantized block by block at load time, requantized to MXFP8 1x32 "
                "with the FC1 rows interleaved, run through the masked-slab MXFP8 layer",
        },
        "mxfp8": {
            9: "not served: sm_90 has no MXFP8 hardware; serve block-FP8 experts with "
               "format 'bsfp8' instead",
            10: "served: bf16 experts are quantized to MXFP8 1x32 at load time, an MXFP8 "
                "checkpoint keeps its bytes and has its scales repacked; the FC1 rows are "
                "interleaved, run through the masked-slab MXFP8 layer",
            12: "served: bf16 experts are quantized to MXFP8 1x32 at load time, an MXFP8 "
                "checkpoint keeps its bytes and has its scales repacked; the FC1 rows are "
                "interleaved, run through the masked-slab MXFP8 layer",
        },
    }
    heads = {
        "bsfp8": "format 'bsfp8': float8_e4m3fn experts with fp32 128x128 block scales "
                 "(the bsgemm-moe dialect)",
        "mxfp8": "format 'mxfp8': bf16 experts quantized to MXFP8 1x32 at load time (the mxfp8 "
                 "dialect), or float8_e4m3fn experts with uint8 UE8M0 scales [E, N, K/32] as an "
                 "MXFP8 checkpoint stores them",
    }
    lines = [f"fish_scales_ops.moe: one MoE layer for every architecture; this device is "
             f"{_arch_label(arch)}{name}.", _build_line()]
    for fmt in FORMATS:
        lines.append(heads[fmt])
        for fam, label in families:
            mark = "   <- this device" if fam == major else ""
            lines.append(f"  {label:40s} {rows[fmt][fam]}{mark}")
    lines.append("Any other format (bf16 experts, int8, fp4, ...) is served on no architecture: "
                 "prepare_experts raises and nothing falls back.")
    if major not in (9, 10, 12):
        lines.append(f"This device ({_arch_label(arch)}) serves no format.")
    lines.append("layer(): top-k ids outside [0, E_local) are skipped at no expert cost, and the "
                 "result is this rank's partial sum, which the caller reduces across ranks.")
    if major == 9:
        lines.append(_sm90_jit_compiler_line())
    return "\n".join(lines)


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class MoeExperts:
    """The routed experts of one MoE layer, prepared for this device by
    :func:`prepare_experts`. Treat it as an opaque handle: the weight and scale
    tensors are in the layout the architecture's kernels read, and the scale layouts
    are not portable between architectures.

    Attributes:
        kind: ``"bsfp8"`` (block-FP8 weights with fp32 128x128 scales, the sm_90
            layer) or ``"mxfp8"`` (MXFP8 1x32 weights with opaque int32 scale
            handles, the sm_100/sm_103 and sm_120/sm_121 layer).
        arch: compute capability of the device the handle was prepared on, as
            ``major * 10 + minor`` (90, 100, 103, 120, 121).
        num_experts: the local expert count E_local.
        hidden: the hidden size H.
        inter: the local intermediate size I_local.
        w13, sw13: the gate/up weights ``[E_local, 2 * I_local, H]`` and their scales.
        w2, sw2: the down weights ``[E_local, H, I_local]`` and their scales.
        w13_interleaved: the ``w13`` rows alternate gate and up (the fused-FC1
            layout of the MXFP8 layer) rather than stacking ``[gate; up]``.
    """

    kind: str
    arch: int
    num_experts: int
    hidden: int
    inter: int
    w13: torch.Tensor
    sw13: torch.Tensor
    w2: torch.Tensor
    sw2: torch.Tensor
    w13_interleaved: bool

    def __repr__(self) -> str:
        return (f"MoeExperts(kind={self.kind!r}, arch=sm_{self.arch}, num_experts={self.num_experts}, "
                f"hidden={self.hidden}, inter={self.inter}, w13_interleaved={self.w13_interleaved}, "
                f"w13={tuple(self.w13.shape)} {self.w13.dtype}, sw13={tuple(self.sw13.shape)} "
                f"{self.sw13.dtype}, w2={tuple(self.w2.shape)} {self.w2.dtype}, "
                f"sw2={tuple(self.sw2.shape)} {self.sw2.dtype})")


def _format_error(format, where: str) -> Exception:
    """The exception for a format this module does not serve on any architecture."""
    served_here = [f for f in FORMATS if supported(f)]
    here = (f"this device ({where}) serves {', '.join(repr(f) for f in served_here)}"
            if served_here else f"this device ({where}) serves no format")
    if isinstance(format, str) and format.strip().lower() in _UNSERVED:
        return NotImplementedError(
            f"fso.moe.prepare_experts: format {format!r} is not served by fish_scales_ops on any "
            f"architecture, and {here}. bf16 experts can be quantized at load time with "
            f"format='mxfp8' on sm_100/sm_103 and sm_120/sm_121, which also takes MXFP8 "
            f"experts as stored (float8_e4m3fn with uint8 UE8M0 scales); block-FP8 experts are "
            f"format='bsfp8'. Otherwise serve this checkpoint with another MoE backend.")
    return ValueError(
        f"fso.moe.prepare_experts: unknown format {format!r} on {where}; the formats are "
        f"'bsfp8' (float8_e4m3fn experts with fp32 128x128 block scales) and 'mxfp8' "
        f"(bf16 experts quantized to MXFP8 1x32 at load time, or float8_e4m3fn experts with "
        f"uint8 UE8M0 scales as an MXFP8 checkpoint stores them), and {here}.")


def _arch_error(fmt: str, arch: int) -> NotImplementedError:
    """The exception for a known format that this device's architecture cannot serve."""
    where = _arch_label(arch)
    major = arch // 10
    if fmt == "mxfp8" and major == 9:
        return NotImplementedError(
            f"fso.moe.prepare_experts: format 'mxfp8' needs MXFP8 hardware (sm_100/sm_103 or "
            f"sm_120/sm_121) and this device is {where}, which has none. On sm_90 serve "
            f"block-FP8 experts with format='bsfp8', or serve this checkpoint with another MoE "
            f"backend.")
    return NotImplementedError(
        f"fso.moe.prepare_experts: fish_scales_ops has no MoE layer for {where}; format "
        f"'bsfp8' is served on sm_90, sm_100/sm_103 and sm_120/sm_121 and format 'mxfp8' on "
        f"sm_100/sm_103 and sm_120/sm_121. Serve this checkpoint with another MoE backend here.")


def _check_expert_tensors(w13, w2, fmt: str, where: str):
    """Shape, device and dtype checks shared by both formats; returns (E, H, I)."""
    for name, t in (("w13", w13), ("w2", w2)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"fso.moe.prepare_experts (format {fmt!r}, {where}): {name} must be a "
                             f"tensor, got {type(t).__name__}")
        if t.dim() != 3:
            raise ValueError(
                f"fso.moe.prepare_experts (format {fmt!r}, {where}): {name} must be 3-D "
                f"({'[E_local, 2*I_local, H]' if name == 'w13' else '[E_local, H, I_local]'}), "
                f"got shape {tuple(t.shape)}")
        if not t.is_cuda:
            raise ValueError(
                f"fso.moe.prepare_experts (format {fmt!r}, {where}): {name} must be on the CUDA "
                f"device that will serve the layer, got {t.device}")
    if w2.device != w13.device:
        raise ValueError(f"fso.moe.prepare_experts (format {fmt!r}, {where}): w13 is on {w13.device} "
                         f"and w2 on {w2.device}; both must be on the serving device")
    e, two_i, h = (int(v) for v in w13.shape)
    e2, h2, i = (int(v) for v in w2.shape)
    if e2 != e:
        raise ValueError(f"fso.moe.prepare_experts (format {fmt!r}, {where}): w13 holds {e} experts "
                         f"and w2 {e2}; both must hold the rank's local experts")
    if not 1 <= e <= 1024:
        raise ValueError(f"fso.moe.prepare_experts (format {fmt!r}, {where}): {e} local experts; "
                         f"the routing kernels take 1 to 1024")
    if h2 != h:
        raise ValueError(f"fso.moe.prepare_experts (format {fmt!r}, {where}): w13 is [E, 2*I, {h}] "
                         f"but w2 is [E, {h2}, I]; both must span the same hidden size")
    if two_i != 2 * i:
        raise ValueError(
            f"fso.moe.prepare_experts (format {fmt!r}, {where}): w13 has {two_i} rows per expert "
            f"and w2 contracts over {i}; w13 must stack gate and up, [2 * I_local, H] per expert")
    if h % 128 != 0:
        raise ValueError(f"fso.moe.prepare_experts (format {fmt!r}, {where}): hidden size {h} is not "
                         f"a multiple of 128, which every block-scaled kernel here requires")
    if i % 128 != 0:
        raise ValueError(
            f"fso.moe.prepare_experts (format {fmt!r}, {where}): intermediate size per rank "
            f"I_local = {i} is not a multiple of 128, which the 128-element scale blocks require; "
            f"under tensor parallelism choose a tp that keeps intermediate_size / tp a multiple "
            f"of 128")
    return e, h, i


def _quantize_mxfp8_chunked(num_experts: int, source, per_expert_elems: int, interleave: bool):
    """Quantize ``num_experts`` experts to MXFP8 1x32 a few at a time.

    ``source(e0, e1)`` returns the bf16 weights of experts ``[e0, e1)``. The grouped
    weight quantizer works expert by expert, so quantizing in chunks produces the
    same bytes as one call over all experts while the transient memory stays at one
    chunk's worth.
    """
    step = max(1, _CONVERT_BUDGET_BYTES // max(1, per_expert_elems * 4))
    w_out = None
    s_out = None
    for e0 in range(0, num_experts, step):
        e1 = min(e0 + step, num_experts)
        chunk = source(e0, e1)
        q, s = quantize_moe_weights_1x32_fp8(chunk, w13_interleave=interleave)
        if w_out is None:
            w_out = torch.empty((num_experts,) + tuple(q.shape[1:]), dtype=q.dtype, device=q.device)
            s_out = torch.empty((num_experts,) + tuple(s.shape[1:]), dtype=s.dtype, device=s.device)
        w_out[e0:e1].copy_(q)
        s_out[e0:e1].copy_(s)
        del chunk, q, s
    return w_out, s_out


def _dequantize_block_fp8(wq: torch.Tensor, scale: torch.Tensor, e0: int, e1: int) -> torch.Tensor:
    """bf16 values of block-FP8 experts ``[e0, e1)``: each fp8 value times the fp32
    scale of its 128x128 block, computed in fp32 and rounded once to bf16."""
    c = e1 - e0
    n, k = int(wq.shape[1]), int(wq.shape[2])
    w = wq[e0:e1].to(torch.float32)
    w.view(c, n // 128, 128, k // 128, 128).mul_(scale[e0:e1].view(c, n // 128, 1, k // 128, 1))
    return w.to(torch.bfloat16)


def _mxfp8_checkpoint_form(w13, w2, sw13, sw2, e: int, h: int, i: int, where: str) -> bool:
    """Which of the two ``"mxfp8"`` forms the caller holds, after validating it.

    ``False``: bf16 experts and no scales, quantized at load time. ``True``: an MXFP8
    checkpoint as it is stored, float8_e4m3fn experts with uint8 UE8M0 scales
    ``sw13 [E, 2I, H/32]`` and ``sw2 [E, H, I/32]``. Any mixture of the two (bf16 with
    scales, float8_e4m3fn without them, one weight of each dtype) and any scale that is
    not the uint8 tensor of that shape on the weights' device raises ``ValueError``.
    """
    ctx = f"fso.moe.prepare_experts (format 'mxfp8', {where})"
    want = {"sw13": (e, 2 * i, h // 32), "sw2": (e, h, i // 32)}
    forms = (f"bf16 experts and no scales (quantized here), or float8_e4m3fn experts with "
             f"uint8 UE8M0 scales sw13 {list(want['sw13'])} and sw2 {list(want['sw2'])} "
             f"(one byte per 32 elements along K, byte b = 2^(b - 127), as an MXFP8 checkpoint "
             f"stores them)")
    fp8 = torch.float8_e4m3fn
    for name, t in (("w13", w13), ("w2", w2)):
        if t.dtype not in (torch.bfloat16, fp8):
            raise ValueError(f"{ctx}: {name} is {t.dtype}; format 'mxfp8' takes {forms}. Block-FP8 "
                             f"experts with fp32 128x128 scales are format='bsfp8'")
    if w13.dtype != w2.dtype:
        raise ValueError(f"{ctx}: w13 is {w13.dtype} and w2 {w2.dtype}; format 'mxfp8' takes {forms}, "
                         f"the same dtype for both")
    if w13.dtype == torch.bfloat16:
        if sw13 is not None or sw2 is not None:
            raise ValueError(f"{ctx}: bf16 experts are quantized here and take no scales; sw13/sw2 go "
                             f"with float8_e4m3fn experts (uint8 sw13 {list(want['sw13'])}, sw2 "
                             f"{list(want['sw2'])}), and fp32 128x128 block scales with format='bsfp8'")
        return False
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    for name, t in (("sw13", sw13), ("sw2", sw2)):
        need = (f"uint8 {list(want[name])}, one UE8M0 byte per 32 elements along K, row-major, "
                f"byte b = 2^(b - 127)")
        if t is None:
            raise ValueError(f"{ctx}: float8_e4m3fn experts need their scales; {name} is missing and "
                             f"must be {need}")
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{ctx}: {name} must be a tensor, {need}; got {type(t).__name__}")
        if t.dtype != torch.uint8 or tuple(t.shape) != want[name]:
            hint = ""
            if e8m0 is not None and t.dtype == e8m0:
                hint = f"; a float8_e8m0fnu tensor holds the same bytes, pass {name}.view(torch.uint8)"
            elif t.dtype == torch.int32:
                hint = ("; the int32 scale handle quantize_moe_weights_1x32_fp8 returns is "
                        "architecture-specific and not taken here, pass the checkpoint's uint8 bytes")
            elif t.dtype == torch.float32:
                hint = "; fp32 128x128 block scales are format='bsfp8'"
            raise ValueError(f"{ctx}: {name} must be {need}; got {t.dtype} {list(t.shape)}{hint}")
        if t.device != w13.device:
            raise ValueError(f"{ctx}: {name} is on {t.device} and the weights on {w13.device}; both "
                             f"must be on the serving device")
    return True


def _prepare_mxfp8_checkpoint(w13, w2, sw13, sw2, arch: int, e: int, h: int, i: int) -> MoeExperts:
    """The handle of an MXFP8 checkpoint as it is stored (validated by
    :func:`_mxfp8_checkpoint_form`): the weight bytes are kept, the FC1 rows re-ordered
    when the handle is interleaved (the rule :func:`prepare_experts` applies to bf16
    experts), and the uint8 scales packed into this device's int32 layout by
    ``_ue8m0_scale_handle``, with the FC1 scale rows following the weight rows. Nothing
    is quantized."""
    interleave = bool(mxfp8_grouped_swiglu_available(2 * i, h))
    perm = _w13_interleave_perm(2 * i, w13.device) if interleave else None
    w13_d = w13.detach()
    w13_q = w13_d.index_select(1, perm) if interleave else w13_d.contiguous()
    s13_q = _ue8m0_scale_handle(sw13.detach(), row_perm=perm)
    w2_q = w2.detach().contiguous()
    s2_q = _ue8m0_scale_handle(sw2.detach())
    return MoeExperts("mxfp8", arch, e, h, i, w13_q, s13_q, w2_q, s2_q, interleave)


def prepare_experts(w13, w2, *, format, sw13=None, sw2=None) -> MoeExperts:
    """Prepare one MoE layer's local experts for this device, once, at load time.

    Args:
        w13: the rank's local gate/up weights ``[E_local, 2 * I_local, H]``, gate
            rows first (``[gate; up]``), on the CUDA device that will serve them.
        w2: the local down weights ``[E_local, H, I_local]``.
        format: what the caller holds. ``"bsfp8"``: float8_e4m3fn weights with fp32
            128x128 block scales ``sw13 [E_local, 2 * I_local / 128, H / 128]`` and
            ``sw2 [E_local, H / 128, I_local / 128]`` (value = fp8 * scale).
            ``"mxfp8"``: bf16 weights and no scales; or an MXFP8 checkpoint as it is
            stored, float8_e4m3fn weights with uint8 UE8M0 scales
            ``sw13 [E_local, 2 * I_local, H / 32]`` and ``sw2 [E_local, H, I_local / 32]``,
            one byte per 32 elements along K, row-major, byte ``b`` meaning the scale
            ``2^(b - 127)`` (value = fp8 * scale), the same bytes on every architecture.
        sw13, sw2: the scales of a float8_e4m3fn layer, in the form ``format`` names;
            omitted for bf16 experts.

    What happens per architecture:

    * sm_90 with ``"bsfp8"``: the tensors are kept as they are (made contiguous
      if they are not), and the handle's kind is ``"bsfp8"``.
    * sm_100/sm_103 and sm_120/sm_121 with ``"bsfp8"``: each expert is dequantized
      block by block (fp8 value times its block scale in fp32, rounded to bf16) and
      requantized to MXFP8 1x32, which is a second quantization of already
      quantized weights; the handle's kind is ``"mxfp8"``.
    * sm_100/sm_103 and sm_120/sm_121 with ``"mxfp8"``: bf16 experts are quantized to
      MXFP8 1x32. float8_e4m3fn experts with uint8 scales are not quantized at all:
      their weight bytes are kept (only the FC1 rows are re-ordered when they are
      interleaved) and their scale bytes are packed into this device's int32 scale
      layout. Kind ``"mxfp8"`` either way.
    * sm_90 with ``"mxfp8"``, any other architecture, and any other format raise
      (``NotImplementedError`` for what is not served, ``ValueError`` for a
      malformed argument), naming the architecture; nothing falls back.

    The MXFP8 FC1 rows are interleaved (gate_j, up_j adjacent) whenever the
    library's load-time query ``mxfp8_grouped_swiglu_available`` says the fused
    FC1 can serve the shape, which it does unless ``FSO_FC1_FUSED=0``; the
    handle's ``w13_interleaved`` records the choice. The conversion runs a few
    experts at a time, so its transient memory is bounded by a fixed budget
    rather than by a second copy of the layer. An MXFP8 checkpoint needs no
    conversion: the handle keeps the caller's ``w2`` (made contiguous if it is not),
    and the caller's ``w13`` too unless the FC1 rows are interleaved, in which case
    the interleaved ``w13`` is a new tensor; the scale handles are always new.

    Constraints: ``H % 128 == 0``, ``I_local % 128 == 0``, ``1 <= E_local <= 1024``,
    and the router draws its top-k without replacement.

    Returns:
        :class:`MoeExperts`.
    """
    arch = _device_arch()
    where = _arch_label(arch)
    if not isinstance(format, str) or format.strip().lower() not in FORMATS:
        raise _format_error(format, where)
    fmt = format.strip().lower()
    if not supported(fmt):
        raise _arch_error(fmt, arch)
    e, h, i = _check_expert_tensors(w13, w2, fmt, where)
    dev_major = torch.cuda.get_device_capability(w13.device)[0]
    if dev_major != arch // 10:
        raise ValueError(
            f"fso.moe.prepare_experts (format {fmt!r}): the experts are on {w13.device}, an "
            f"sm_{dev_major}x device, while fish_scales_ops dispatches on device 0 ({where}); "
            f"mixed architectures in one process are not supported")

    if fmt == "bsfp8":
        for name, t in (("w13", w13), ("w2", w2)):
            if t.dtype != torch.float8_e4m3fn:
                raise ValueError(
                    f"fso.moe.prepare_experts (format 'bsfp8', {where}): {name} must be "
                    f"float8_e4m3fn, got {t.dtype}; bf16 experts are format='mxfp8'")
        want = {"sw13": (e, 2 * i // 128, h // 128), "sw2": (e, h // 128, i // 128)}
        for name, t in (("sw13", sw13), ("sw2", sw2)):
            if not isinstance(t, torch.Tensor):
                raise ValueError(
                    f"fso.moe.prepare_experts (format 'bsfp8', {where}): {name} is required, "
                    f"fp32 {list(want[name])} (one scale per 128x128 block)")
            if t.dtype != torch.float32 or tuple(t.shape) != want[name]:
                raise ValueError(
                    f"fso.moe.prepare_experts (format 'bsfp8', {where}): {name} must be fp32 "
                    f"{list(want[name])} (one scale per 128x128 block), got {t.dtype} "
                    f"{list(t.shape)}")
            if t.device != w13.device:
                raise ValueError(
                    f"fso.moe.prepare_experts (format 'bsfp8', {where}): {name} is on {t.device}, "
                    f"the weights on {w13.device}")
        w13_c = w13.detach().contiguous()
        w2_c = w2.detach().contiguous()
        s13_c = sw13.detach().contiguous()
        s2_c = sw2.detach().contiguous()
        if arch // 10 == 9:
            return MoeExperts("bsfp8", arch, e, h, i, w13_c, s13_c, w2_c, s2_c, False)

        def src13(e0, e1):
            return _dequantize_block_fp8(w13_c, s13_c, e0, e1)

        def src2(e0, e1):
            return _dequantize_block_fp8(w2_c, s2_c, e0, e1)
    else:
        if _mxfp8_checkpoint_form(w13, w2, sw13, sw2, e, h, i, where):
            return _prepare_mxfp8_checkpoint(w13, w2, sw13, sw2, arch, e, h, i)
        w13_d = w13.detach()
        w2_d = w2.detach()

        def src13(e0, e1):
            return w13_d[e0:e1]

        def src2(e0, e1):
            return w2_d[e0:e1]

    interleave = bool(mxfp8_grouped_swiglu_available(2 * i, h))
    w13_q, s13_q = _quantize_mxfp8_chunked(e, src13, 2 * i * h, interleave)
    w2_q, s2_q = _quantize_mxfp8_chunked(e, src2, h * i, False)
    return MoeExperts("mxfp8", arch, e, h, i, w13_q, s13_q, w2_q, s2_q, interleave)


# The fused-combine FC2 (sm_120/121) adds each routed row into its token's output
# row with atomic reductions instead of storing the per-expert down-projection slab
# for a separate combine kernel. Where `moe_layer_fused_combine_engages_sm120` would
# take it, the accumulation order follows the order in which the CTAs finish, so
# those buckets are not bit-reproducible run to run; the library therefore keeps it
# opt-in here, the same default as moe_layer_mxfp8_sm120: FSO_MOE_FUSED_COMBINE=1
# allows it and the engagement rule then decides per bucket (read once per process).
_fused_combine_env: bool | None = None


def _fused_combine_allowed() -> bool:
    global _fused_combine_env
    if _fused_combine_env is None:
        _fused_combine_env = env_flag("FSO_MOE_FUSED_COMBINE")
    return _fused_combine_env and sm_major() == 12


@torch.library.custom_op(
    "fish_scales_ops::moe_layer",
    mutates_args=(),
    schema="(Tensor hidden, Tensor w13, Tensor sw13, Tensor w2, Tensor sw2, Tensor topk_ids, "
           "Tensor topk_w, Tensor? bias, Tensor? bias_scale, str kind, bool w13_interleaved) -> Tensor",
)
def _moe_layer_op(hidden, w13, sw13, w2, sw2, topk_ids, topk_w, bias, bias_scale, kind, w13_interleaved):
    """The body of ``torch.ops.fish_scales_ops.moe_layer``: dispatch on the handle's
    kind and on this device's architecture, then run that architecture's chain."""
    major = sm_major()
    arch = _device_arch()
    # layer() already hands over int32 ids, fp32 weights and a [M] bias_scale; a caller
    # of the registered op may not, and each of these is a no-op when it does.
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        topk_ids = topk_ids.to(torch.int32).contiguous()
    if topk_w.dtype != torch.float32 or not topk_w.is_contiguous():
        topk_w = topk_w.to(torch.float32).contiguous()
    if bias is not None:
        bias = bias.contiguous()
    if bias_scale is not None:
        if bias is None:
            raise ValueError(f"fish_scales_ops::moe_layer ({_arch_label(arch)}): bias_scale scales bias "
                             f"and needs it")
        bias_scale = bias_scale.reshape(-1).to(torch.float32).contiguous()
    if kind == "bsfp8":
        if major != 9:
            raise NotImplementedError(
                f"fish_scales_ops::moe_layer: kind 'bsfp8' is the sm_90 block-FP8 layer and this "
                f"device is {_arch_label(arch)}; prepare the experts on this device with "
                f"fso.moe.prepare_experts, which requantizes block-FP8 experts to MXFP8 on "
                f"sm_100/sm_103 and sm_120/sm_121")
        if w13_interleaved:
            raise ValueError("fish_scales_ops::moe_layer: the sm_90 block-FP8 layer takes w13 in the "
                             "[gate; up] order; w13_interleaved must be False for kind 'bsfp8'")
        if int(hidden.shape[0]) == 0:
            return hidden.new_empty(hidden.shape)
        y = moe_layer_fp8_sm90(hidden, w13, sw13, w2, sw2, topk_ids, topk_w)
        # The sm_90 combine kernel has no bias input yet, so the bias is folded with one
        # fused elementwise pass over the result: out + bias, or out + bias_scale * bias
        # evaluated as one fp32 multiply-add (the product is not rounded on its own) and
        # rounded once to bf16.
        if bias is not None:
            if bias_scale is None:
                y.add_(bias)
            else:
                y.addcmul_(bias, bias_scale.unsqueeze(1))
        return y
    if kind == "mxfp8":
        if major not in (10, 12):
            raise NotImplementedError(
                f"fish_scales_ops::moe_layer: kind 'mxfp8' needs MXFP8 hardware (sm_100/sm_103 or "
                f"sm_120/sm_121) and this device is {_arch_label(arch)}; on sm_90 prepare "
                f"block-FP8 experts with format='bsfp8'")
        x = hidden if hidden.is_contiguous() else hidden.contiguous()
        return _moe_layer_mxfp8(
            x, w13, sw13, w2, sw2, topk_ids, topk_w, bias=bias, bias_scale=bias_scale,
            w13_interleaved=w13_interleaved, fused_combine=_fused_combine_allowed())
    raise ValueError(f"fish_scales_ops::moe_layer: unknown kind {kind!r} on {_arch_label(arch)}; "
                     f"the kinds are 'bsfp8' and 'mxfp8', produced by fso.moe.prepare_experts")


@_moe_layer_op.register_fake
def _moe_layer_fake(hidden, w13, sw13, w2, sw2, topk_ids, topk_w, bias, bias_scale, kind, w13_interleaved):
    return hidden.new_empty(hidden.shape)


def layer(hidden, experts, topk_ids, topk_w, *, bias=None, bias_scale=None) -> torch.Tensor:
    """Run the routed experts of one MoE layer for one batch of tokens.

    ``out[t] = sum_j topk_w[t, j] * expert_{topk_ids[t, j]}(hidden[t])``, plus
    ``bias_scale[t] * bias[t]`` when ``bias`` is given, where
    ``expert_e(x) = (silu(x W_gate^T) * (x W_up^T)) W_down^T`` over the rank's local
    experts. Every id outside ``[0, E_local)`` is skipped at no expert cost on every
    architecture: that covers the padded rows of a CUDA-graph bucket (sglang labels
    them ``num_experts``, or ``-1``), the entries an expert-parallel dispatcher maps
    to ``-1`` because another rank owns the expert, and the padded-row sentinel it
    maps to ``E_local``. A token whose ids are all skipped comes out as its bias
    term, or as zero without a bias. The result is this rank's partial sum; the
    caller reduces it across tensor- and expert-parallel ranks.

    Args:
        hidden: bf16 ``[M, H]`` on the device the experts were prepared on.
        experts: the :class:`MoeExperts` handle from :func:`prepare_experts`.
        topk_ids: ``[M, topk]`` int32 or int64 local expert ids, drawn without
            replacement per token (narrowed to int32 here).
        topk_w: ``[M, topk]`` combine weights (taken to fp32 here).
        bias: optional bf16 ``[M, H]`` added to the result, e.g. a shared expert's
            output. On sm_100/sm_103 and sm_120/sm_121 it is folded into the
            combine kernel; on sm_90 it is one fused elementwise add on the result.
        bias_scale: optional ``[M]`` per-token factor on ``bias`` (e.g. a sigmoid
            shared-expert gate), taken to fp32; requires ``bias``.

    The call is ``torch.ops.fish_scales_ops.moe_layer``, a torch custom op: opaque
    to ``torch.compile`` (its fake implementation returns an empty ``[M, H]``) and
    capturable into a CUDA graph after one eager call of the same shape, which
    creates the library's per-thread pools and, on sm_90, JIT-compiles the kernels.
    Every host-side decision is a function of the argument shapes, so one capture
    per token bucket replays correctly for any routing of that bucket.

    Returns:
        bf16 ``[M, H]``, a new tensor.
    """
    arch = _device_arch()
    where = _arch_label(arch)
    if not isinstance(experts, MoeExperts):
        raise TypeError(f"fso.moe.layer ({where}): experts must be the MoeExperts handle from "
                        f"fso.moe.prepare_experts, got {type(experts).__name__}")
    if experts.arch // 10 != arch // 10:
        raise ValueError(
            f"fso.moe.layer: the experts were prepared on sm_{experts.arch} and this device is "
            f"{where}; the weight and scale layouts are architecture-specific, so prepare them "
            f"with fso.moe.prepare_experts on the serving device")
    if not isinstance(hidden, torch.Tensor) or hidden.dim() != 2 or hidden.dtype != torch.bfloat16:
        raise ValueError(f"fso.moe.layer ({where}, kind {experts.kind!r}): hidden must be bf16 [M, H]")
    m = int(hidden.shape[0])
    if int(hidden.shape[1]) != experts.hidden:
        raise ValueError(f"fso.moe.layer ({where}, kind {experts.kind!r}): hidden is "
                         f"[{m}, {int(hidden.shape[1])}] but the experts span H = {experts.hidden}")
    if topk_ids.dim() != 2 or topk_w.dim() != 2 or tuple(topk_ids.shape) != tuple(topk_w.shape) \
            or int(topk_ids.shape[0]) != m:
        raise ValueError(
            f"fso.moe.layer ({where}, kind {experts.kind!r}): topk_ids and topk_w must both be "
            f"[M, topk] with M = {m}, got {tuple(topk_ids.shape)} and {tuple(topk_w.shape)}")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"fso.moe.layer ({where}, kind {experts.kind!r}): topk_ids must be int32 or "
                         f"int64, got {topk_ids.dtype}")
    if not topk_w.is_floating_point():
        raise ValueError(f"fso.moe.layer ({where}, kind {experts.kind!r}): topk_w must be floating "
                         f"point, got {topk_w.dtype}")
    ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
    ids = ids.contiguous()
    wts = topk_w if topk_w.dtype == torch.float32 else topk_w.to(torch.float32)
    wts = wts.contiguous()
    x = hidden if hidden.is_contiguous() else hidden.contiguous()
    if bias is not None:
        if bias.dim() != 2 or tuple(bias.shape) != (m, experts.hidden) or bias.dtype != torch.bfloat16:
            raise ValueError(f"fso.moe.layer ({where}, kind {experts.kind!r}): bias must be bf16 "
                             f"[{m}, {experts.hidden}], got {bias.dtype} {tuple(bias.shape)}")
        bias = bias.contiguous()
    if bias_scale is not None:
        if bias is None:
            raise ValueError(f"fso.moe.layer ({where}, kind {experts.kind!r}): bias_scale scales "
                             f"bias and needs it")
        if bias_scale.numel() != m or not bias_scale.is_floating_point():
            raise ValueError(f"fso.moe.layer ({where}, kind {experts.kind!r}): bias_scale must hold "
                             f"one floating-point factor per token ({m}), got {bias_scale.dtype} "
                             f"{tuple(bias_scale.shape)}")
        bias_scale = bias_scale.reshape(m)
        bias_scale = bias_scale if bias_scale.dtype == torch.float32 else bias_scale.to(torch.float32)
        bias_scale = bias_scale.contiguous()
    return torch.ops.fish_scales_ops.moe_layer(
        x, experts.w13, experts.sw13, experts.w2, experts.sw2, ids, wts, bias, bias_scale,
        experts.kind, experts.w13_interleaved)


def transient_bytes(experts, tokens, topk) -> int:
    """Device memory, in bytes, that one :func:`layer` call with ``tokens`` rows
    allocates on this device for these experts: the routing tensors, the gathered
    and quantized activation, the intermediates of the two grouped GEMMs and the
    ``[tokens, hidden]`` result, each charged as the caching allocator can charge
    it (512-byte blocks, and up to 1 MiB more for a tensor above 1 MiB that a cached
    block serves unsplit). It is an upper bound on what the call occupies, tight to
    about 1 MiB per large tensor.

    :func:`layer` runs every bucket as one call, so this is the memory a serving
    stack reserves, once per process, for the largest bucket a forward can carry
    (the layer's intermediates are freed
    when the call returns, so the layers of a model reuse one reservation). It is
    exact for the route the layer takes on this device at that token count,
    because the layer and this function read the same host-side plan:

    * sm_90, kind ``"bsfp8"``: the expert-sorted layout, sized by the routed rows
      (``tokens * topk`` plus per-expert padding), see
      :func:`fish_scales_ops.compat.moe_layer_transient_bytes_sm90`;
    * sm_100/103 and sm_120/121, kind ``"mxfp8"``: the masked slab, sized by
      ``num_experts * align(tokens, 4)`` rows, with the fused FC1 when the handle's
      ``w13_interleaved`` and the route query allow it and, on sm_120/121, the fused
      combine where ``FSO_MOE_FUSED_COMBINE=1`` allows it and its engagement rule
      takes it, see :func:`fish_scales_ops.compat.moe_layer_transient_bytes_mxfp8`.

    Inputs, weights, a ``bias`` and the library's persistent pools (routing scratch,
    the sm_100/103 argument arena) are not counted.

    Args:
        experts: the :class:`MoeExperts` handle from :func:`prepare_experts`.
        tokens: rows in the bucket (``M``), ``>= 0``.
        topk: experts per token.
    """
    arch = _device_arch()
    where = _arch_label(arch)
    if not isinstance(experts, MoeExperts):
        raise TypeError(f"fso.moe.transient_bytes ({where}): experts must be the MoeExperts handle from "
                        f"fso.moe.prepare_experts, got {type(experts).__name__}")
    if experts.arch // 10 != arch // 10:
        raise ValueError(
            f"fso.moe.transient_bytes: the experts were prepared on sm_{experts.arch} and this device is "
            f"{where}; the layout, and so the memory it needs, is architecture-specific")
    for name, v in (("tokens", tokens), ("topk", topk)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise ValueError(f"fso.moe.transient_bytes ({where}): {name} must be a non-negative int, got {v!r}")
    if tokens == 0:
        return 0
    if topk < 1:
        raise ValueError(f"fso.moe.transient_bytes ({where}): topk must be >= 1, got {topk}")
    if experts.kind == "bsfp8":
        return moe_layer_transient_bytes_sm90(tokens, experts.num_experts, topk, experts.hidden, experts.inter)
    return moe_layer_transient_bytes_mxfp8(
        tokens, experts.num_experts, topk, experts.hidden, experts.inter,
        w13_interleaved=experts.w13_interleaved, fused_combine=_fused_combine_allowed())
