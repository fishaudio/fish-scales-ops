"""``fish_scales_ops.dense`` — one dense linear surface for every architecture.

A serving stack holds each dense projection (attention, MLP, lm_head) in one of
two checkpoint dialects and needs two things from this library: a load-time
preparation of the weight on the device that will serve it, and a per-call
linear that quantizes the activation and runs the GEMM. The architecture
dispatch lives inside the library, in a torch custom op, so the caller writes
one code path and never names an architecture or a scale layout::

    import fish_scales_ops as fso

    weight = fso.dense.prepare_weight(w, format="bsfp8", scale=w_scale)   # load time
    y = fso.dense.linear(x, weight)                                        # per call
    fso.dense.supported("mxfp8")     # can this device serve the format?
    print(fso.dense.describe())      # the architecture matrix, for logs and errors

Interface v1 (issue #12) adds, on the same handle:

    xq, sx = fso.dense.quantize_activation(x)          # the activation contract of this device
    y = fso.dense.linear(xq, weight, x_scale=sx)       # pre-quantized activation: no quantize inside
    y = fso.dense.linear(x, weight, bias=b, residual=r, activation="gelu_erf")   # epilogue arguments
    xq2, sx2 = fso.dense.linear_quantized(x, weight, bias=b, activation="gelu_erf")
    fso.dense.activation_contract()                    # the layout and rounding rule, versioned
    fso.dense.include_dir()                            # holds fso/activation_contract.h

``x_scale`` is complete on every architecture. The epilogue arguments (``bias``, ``residual``, ``activation``)
and ``linear_quantized`` are validated and passed down to the architecture's GEMM. In this build sm_90 (all three
of its dense kernels), the sm_100/sm_103 C++ cascade (tier 3) and sm_120/sm_121 (every route of its MXFP8 cascade)
apply ``bias``, ``residual`` and ``activation`` (one launch, one rounding), the sm_100/sm_103 swap-AB row (the
M <= 64 decode band and the narrow-N long-K band above it) applies them too and tier 1 (cuBLAS ``scaled_mm``) the
``bias``, and the sm_100/sm_103 router sends a call to the first tier of its shape that applies every field it
carries. ``linear_quantized`` is served by every route of the
sm_120/sm_121 MXFP8 cascade and by the sm_100/sm_103 C++ cascade, which every such call reaches through that
fall-through (the GEMM writes the quantized result in its own launch). The routes that apply none of the fields
(sm_120/sm_121 block-FP8 1x128), and ``linear_quantized`` on sm_90, refuse them with ``NotImplementedError`` naming
the architecture and the route, before anything is launched.

``format`` names what the caller holds. ``"bsfp8"`` is a float8_e4m3fn weight
with fp32 128x128 block scales (value = fp8 * scale), or a bf16 weight that is
quantized to block-FP8 at load time. ``"mxfp8"`` is a bf16 weight that is
quantized to MXFP8 1x32 at load time, or a float8_e4m3fn weight with its MXFP8
scales: either the uint8 UE8M0 bytes ``[N, K/32]`` an MXFP8 checkpoint stores (the
same bytes on every architecture, packed here into this device's layout), or the
int32 scales that :func:`fish_scales_ops.compat.quantize_1x32_fp8` produces on
this architecture.

sm_90 serves ``"bsfp8"`` as block-FP8: every call runs ``linear_qx``, which
quantizes the activation to 1x128 inside the GEMM op. sm_90 has no MXFP8
hardware and refuses ``"mxfp8"``. sm_100/sm_103 and sm_120/sm_121 serve both
formats as MXFP8 1x32, because the MXFP8 GEMM is the dense path those
architectures are built around: a block-FP8 checkpoint is dequantized block by
block at load time and requantized, the same rule as
:func:`fish_scales_ops.moe.prepare_experts`, and every call runs the fused 1x32
activation quantize followed by :func:`fish_scales_ops.compat.linear_mxfp8`.
Every other format and every other architecture raises; nothing falls back.

:func:`linear` is ``torch.ops.fish_scales_ops.dense_linear``, a torch custom op
that ships its own fake implementation, so ``torch.compile`` keeps it as one
opaque node and the caller registers nothing. The explicit, format-specific ops
it is built from are in :mod:`fish_scales_ops.compat`.
"""
import dataclasses
import math
import os
import warnings

import torch

from .._arch import sm_major
from .._env import env_flag
from ..gemm import _epilogue
from ..gemm.fp8 import _sm90_act_scale, linear_fp8, quantize_128x128_fp8
from ..gemm.mxfp8 import _sm_count, _ue8m0_scale_handle, linear_mxfp8, quantize_1x32_fp8
# One architecture probe and one parser of the ``arch`` argument, shared with fso.moe,
# and the sm_90 JIT compiler line both describe() functions end with.
from ..moe import _arch_label, _arch_major, _build_line, _device_arch, _sm90_jit_compiler_line

__all__ = ["FORMATS", "ACTIVATIONS", "DenseWeight", "prepare_weight", "linear", "linear_quantized",
           "quantize_activation", "activation_contract", "include_dir", "supported", "describe"]

# The activations the epilogue knows (interface v1): None, or one of these names.
ACTIVATIONS = _epilogue.ACTIVATIONS

# FSO_BATCH_INVARIANT (docs/api/dense.md, "Determinism and batch invariance"): read once per process, at the first
# dense call. On sm_90 the extension acts on it (no route whose summation order depends on M); on the other
# architectures it changes nothing in this release, and the first call that finds it set says so once.
_BATCH_INVARIANT_SEEN = False


def _batch_invariant_note(major: int) -> None:
    global _BATCH_INVARIANT_SEEN
    if _BATCH_INVARIANT_SEEN:
        return
    _BATCH_INVARIANT_SEEN = True
    if major != 9 and env_flag("FSO_BATCH_INVARIANT"):
        warnings.warn("fish_scales_ops: FSO_BATCH_INVARIANT=1 makes the dense GEMM batch-invariant on sm_90 only in "
                      "this release; on this architecture it has no effect (docs/api/dense.md, \"Determinism and "
                      "batch invariance\")", RuntimeWarning, stacklevel=3)

# The version of the activation contract (the xq / x_scale layout and the quantizer rule) that
# quantize_activation produces, linear(x_scale=...) accepts and fso/activation_contract.h states.
_CONTRACT_VERSION = 1

# The formats prepare_weight accepts, and the architecture majors that serve each.
FORMATS = ("bsfp8", "mxfp8")
_SERVED = {"bsfp8": (9, 10, 12), "mxfp8": (10, 12)}

# Dialects that exist but that no architecture serves through this module. They are
# refused with NotImplementedError, and anything not listed here or in FORMATS with
# ValueError, so a caller can tell "not served" from "misspelt".
_UNSERVED = ("bf16", "bfloat16", "fp16", "float16", "fp32", "float32", "fp8", "int8", "w8a8",
             "int4", "w4a16", "w4a8", "mxfp4", "nvfp4")

# Largest fp32 scratch, in bytes, that the load-time conversion of a block-FP8
# checkpoint to MXFP8 holds at once. The conversion dequantizes and requantizes the
# weight a whole number of 128-row blocks at a time, so preparing a large projection
# (an lm_head, say) never needs an fp32 or bf16 copy of the whole weight.
_CONVERT_BUDGET_BYTES = 256 * 1024 * 1024


def supported(format, arch=None) -> bool:
    """Whether ``format`` can be served on ``arch`` (this device when ``None``).

    ``"bsfp8"`` is served on sm_90, sm_100/sm_103 and sm_120/sm_121, ``"mxfp8"``
    on sm_100/sm_103 and sm_120/sm_121, and nothing else anywhere. The query
    answers from the architecture matrix alone and never raises for a format it
    does not know (it answers ``False``); an ``arch`` that names no compute
    capability raises ``ValueError``. ``arch`` takes the forms
    :func:`fish_scales_ops.moe.supported` takes: an int major (9, 10, 12), an int
    ``major * 10 + minor`` (90, 103, 120), a ``(major, minor)`` pair, or a string
    such as ``"sm_90"``, ``"sm_100a"`` or ``"12.0"``. Without a CUDA device it
    answers ``False`` for every format.
    """
    if not isinstance(format, str):
        return False
    majors = _SERVED.get(format.strip().lower())
    if majors is None:
        return False
    major = sm_major() if arch is None else _arch_major(arch)
    return major in majors


def describe() -> str:
    """The architecture matrix of the dense routes as text, with this device's
    rows marked, for logs and for the messages of a caller that refuses a
    configuration. The second line names the build (version, commit and the
    torch the extension was built against, from
    :func:`fish_scales_ops.build_info`). On sm_90 the last line names the
    compiler of the deep_gemm JIT (the bundled NVRTC and its path), or the error
    that keeps it from loading; describe() itself never raises."""
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
            9: "served: the block-FP8 weight is kept as it is (a bf16 weight is quantized with "
               "quantize_128x128_fp8); every call runs linear_qx, which quantizes the activation "
               "to 1x128 inside the GEMM op",
            10: "served: dequantized block by block at load time and requantized to MXFP8 1x32 (a bf16 "
                "weight is quantized to MXFP8 directly); every call runs the 1x32 activation quantize "
                "and the MXFP8 router (the swap-AB row: the M <= 64 decode band and the narrow-N long-K "
                "band above it; cuBLAS scaled_mm, the CuTe-DSL tier, "
                "the C++ cascade)",
            12: "served: dequantized block by block at load time and requantized to MXFP8 1x32 (a bf16 "
                "weight is quantized to MXFP8 directly); a call runs the MXFP8 routes of format 'mxfp8' "
                "below",
        },
        "mxfp8": {
            9: "not served: sm_90 has no MXFP8 hardware; serve a block-FP8 checkpoint or a bf16 "
               "weight with format 'bsfp8' instead",
            10: "served: a bf16 weight is quantized to MXFP8 1x32 at load time and an MXFP8 checkpoint "
                "keeps its bytes, uint8 UE8M0 scales packed into the atom layout; every call runs the "
                "1x32 activation quantize and the MXFP8 router",
            12: "served: a bf16 weight is quantized to MXFP8 1x32 at load time and an MXFP8 checkpoint "
                "keeps its bytes, its scale packed or restored to the N-fastest layout the GEMM reads; "
                "in the decode band (M <= 64) the small-M route sm120.smallm serves the shapes its "
                "rules pick (one launch for a bf16 x, the activation quantized inside the kernel; the "
                "2b call likewise), every other call runs the 1x32 activation quantize and the "
                "CUTLASS MXFP8 cascade",
        },
    }
    heads = {
        "bsfp8": "format 'bsfp8': a float8_e4m3fn weight with fp32 128x128 block scales, or a bf16 "
                 "weight quantized at load time",
        "mxfp8": "format 'mxfp8': a bf16 weight quantized to MXFP8 1x32 at load time, or a "
                 "float8_e4m3fn weight with uint8 UE8M0 scales [N, K/32] as an MXFP8 checkpoint "
                 "stores them, or with the int32 MXFP8 scales of quantize_1x32_fp8 on this "
                 "architecture",
    }
    lines = [f"fish_scales_ops.dense: one dense linear for every architecture; this device is "
             f"{_arch_label(arch)}{name}.", _build_line()]
    for fmt in FORMATS:
        lines.append(heads[fmt])
        for fam, label in families:
            mark = "   <- this device" if fam == major else ""
            lines.append(f"  {label:40s} {rows[fmt][fam]}{mark}")
    lines.append("Any other format (a bf16-precision GEMM, int8, fp4, ...) is served on no architecture: "
                 "prepare_weight raises and nothing falls back.")
    if major not in (9, 10, 12):
        lines.append(f"This device ({_arch_label(arch)}) serves no format.")
    lines.append("linear(): x is bf16 [..., K], quantized inside the op, or the float8_e4m3fn [M, K] of "
                 "quantize_activation with its x_scale (activation contract v1, no quantize inside); the result "
                 "is bf16 [..., N]. The epilogue arguments bias / residual / activation and linear_quantized are "
                 "validated and passed to the architecture's GEMM: sm_90, the sm_100/sm_103 swap-AB row and C++ "
                 "cascade and the sm_120/sm_121 MXFP8 cascade apply bias, residual and gelu_erf inside their GEMM, "
                 "sm_100/sm_103 tier 1 (cuBLAS scaled_mm) the bias, and the sm_100/sm_103 router sends a call to "
                 "the first tier of its shape that applies every field it carries; the sm_120/sm_121 MXFP8 cascade "
                 "and the sm_100/sm_103 C++ cascade (where every such call lands) also write linear_quantized's "
                 "result in their own launch; the routes that apply none (sm_120/sm_121 block-FP8 1x128), and "
                 "linear_quantized on sm_90, refuse them with NotImplementedError before anything is launched.")
    if major == 9:
        lines.append(_sm90_jit_compiler_line())
    return "\n".join(lines)


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class DenseWeight:
    """One dense weight, prepared for this device by :func:`prepare_weight`.
    Treat it as an opaque handle: the weight and scale tensors are in the layout
    the architecture's GEMM reads, and the scale layouts are not portable between
    architectures.

    Attributes:
        kind: the execution kind, ``"bsfp8"`` (block-FP8 weight with fp32 128x128
            scales, run by ``linear_qx`` on sm_90) or ``"mxfp8"`` (MXFP8 1x32
            weight with an opaque int32 scale handle, run by the MXFP8 GEMM of
            sm_100/sm_103 and sm_120/sm_121).
        format: the ``format`` the caller passed to :func:`prepare_weight`.
        arch: compute capability of the device the handle was prepared on, as
            ``major * 10 + minor`` (90, 100, 103, 120, 121).
        in_features: ``K``, the contraction size.
        out_features: ``N``, the output size.
        weight: float8_e4m3fn ``[N, K]``.
        scale: the architecture-native scale tensor of ``weight``: fp32
            ``[ceil(N/128), K/128]`` for kind ``"bsfp8"``; for kind ``"mxfp8"``
            int32 ``[N, K/128]`` with strides ``(1, N)`` (N fastest) on sm_120/sm_121, and a
            1-D int32 tensor in the CUTLASS block-scaled atom layout on
            sm_100/sm_103.
    """

    kind: str
    format: str
    arch: int
    in_features: int
    out_features: int
    weight: torch.Tensor
    scale: torch.Tensor

    def __repr__(self) -> str:
        return (f"DenseWeight(kind={self.kind!r}, format={self.format!r}, arch=sm_{self.arch}, "
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"weight={tuple(self.weight.shape)} {self.weight.dtype}, "
                f"scale={tuple(self.scale.shape)} {self.scale.dtype} stride={tuple(self.scale.stride())})")


def _format_error(format, where: str) -> Exception:
    """The exception for a format this module does not serve on any architecture."""
    served_here = [f for f in FORMATS if supported(f)]
    here = (f"this device ({where}) serves {', '.join(repr(f) for f in served_here)}"
            if served_here else f"this device ({where}) serves no format")
    if isinstance(format, str) and format.strip().lower() in _UNSERVED:
        return NotImplementedError(
            f"fso.dense.prepare_weight: format {format!r} is not served by fish_scales_ops on any "
            f"architecture, and {here}. A bf16 weight is quantized at load time with format='bsfp8' "
            f"or, on sm_100/sm_103 and sm_120/sm_121, format='mxfp8'; a block-FP8 checkpoint is "
            f"format='bsfp8' and an MXFP8 checkpoint (float8_e4m3fn with uint8 UE8M0 scales) "
            f"format='mxfp8'. Otherwise serve this weight with another GEMM.")
    return ValueError(
        f"fso.dense.prepare_weight: unknown format {format!r} on {where}; the formats are 'bsfp8' "
        f"(a float8_e4m3fn weight with fp32 128x128 block scales, or a bf16 weight) and 'mxfp8' (a "
        f"bf16 weight, or a float8_e4m3fn weight with uint8 UE8M0 scales or the MXFP8 scales of "
        f"quantize_1x32_fp8), and {here}.")


def _arch_error(fmt: str, arch: int) -> NotImplementedError:
    """The exception for a known format that this device's architecture cannot serve."""
    where = _arch_label(arch)
    if fmt == "mxfp8" and arch // 10 == 9:
        return NotImplementedError(
            f"fso.dense.prepare_weight: format 'mxfp8' needs MXFP8 hardware (sm_100/sm_103 or "
            f"sm_120/sm_121) and this device is {where}, which has none. On sm_90 serve a block-FP8 "
            f"checkpoint, or a bf16 weight, with format='bsfp8'.")
    return NotImplementedError(
        f"fso.dense.prepare_weight: fish_scales_ops has no dense GEMM for {where}; format 'bsfp8' "
        f"is served on sm_90, sm_100/sm_103 and sm_120/sm_121 and format 'mxfp8' on sm_100/sm_103 "
        f"and sm_120/sm_121. Serve this weight with another GEMM here.")


def _aligned_storage(t: torch.Tensor, align: int = 16) -> torch.Tensor:
    """``t``, or a copy of it with the same shape, strides and bytes in an allocation of its own when its data pointer
    is not a multiple of ``align`` bytes. :func:`prepare_weight` keeps a caller's checkpoint tensor in the handle where
    it can, and the GEMMs read the handle's tensors in place (TMA needs 16 bytes; the CuTe-DSL tier of sm_100/sm_103
    takes a weight scale only at 32 bytes), so a kept tensor that is a view at such an offset is copied once here, at
    load time, and never per call. Every other tensor of a handle is a fresh allocation, which the caching allocator
    aligns far beyond both."""
    if t.numel() == 0 or t.data_ptr() % align == 0:
        return t
    out = torch.empty_strided(tuple(t.shape), tuple(t.stride()), dtype=t.dtype, device=t.device)
    out.copy_(t)
    return out


def _block_scale(scale, n: int, k: int, device, ctx: str) -> torch.Tensor:
    """Validate the fp32 128x128 block scales of a block-FP8 [N, K] weight; returns
    them contiguous (the row-major layout every block-FP8 consumer reads)."""
    want = ((n + 127) // 128, k // 128)
    if not isinstance(scale, torch.Tensor) or scale.dtype != torch.float32 or tuple(scale.shape) != want:
        got = (f"{scale.dtype} {list(scale.shape)}" if isinstance(scale, torch.Tensor)
               else type(scale).__name__)
        raise ValueError(
            f"{ctx}: scale must be fp32 {list(want)}, one scale per 128x128 block of the [N, K] = "
            f"[{n}, {k}] weight (value = fp8 * scale), got {got}")
    if scale.device != device:
        raise ValueError(f"{ctx}: scale is on {scale.device} and the weight on {device}; both must be "
                         f"on the serving device")
    return scale.detach().contiguous()


def _mxfp8_scale(scale, n: int, k: int, device, major: int, ctx: str) -> torch.Tensor:
    """Validate the MXFP8 scale of an MXFP8 [N, K] checkpoint and return it in the
    layout the GEMM reads.

    On sm_120/sm_121 ``quantize_1x32_fp8`` returns int32 ``[N, K/128]`` with the
    strides ``(1, N)`` (N fastest: the words of one 128-wide K block are adjacent
    across rows), and the GEMM reads those bytes by raw pointer. A safetensors
    round trip stores ``.contiguous()``, which keeps every element's value but
    rewrites the bytes row-major, strides ``(K/128, 1)``; handed to the GEMM as it
    is, that tensor would be read in the wrong order and the result would be wrong
    without an error. Both forms are accepted here, and the row-major one is
    rewritten to the N-fastest layout. With ``K/128 == 1`` the two forms are the
    same bytes. On sm_100/sm_103 the scale is a 1-D tensor in the CUTLASS
    block-scaled atom layout, which a round trip leaves unchanged.
    """
    kb = k // 128
    if not isinstance(scale, torch.Tensor):
        raise ValueError(f"{ctx}: scale must be the checkpoint's uint8 UE8M0 scale [N, K/32] = [{n}, "
                         f"{k // 32}] or the int32 tensor fso.compat.quantize_1x32_fp8 returns for the "
                         f"weight on this architecture, got {type(scale).__name__}")
    if scale.dtype != torch.int32:
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        hint = ("; a float8_e8m0fnu tensor holds the same bytes, pass scale.view(torch.uint8)"
                if e8m0 is not None and scale.dtype == e8m0 else "")
        raise ValueError(f"{ctx}: the MXFP8 scale must be the checkpoint's uint8 UE8M0 scale [N, K/32] = "
                         f"[{n}, {k // 32}] (one byte per 32 elements along K, row-major, byte b = "
                         f"2^(b - 127)) or the int32 tensor fso.compat.quantize_1x32_fp8 returns for the "
                         f"weight on this architecture (four UE8M0 bytes per word), got {scale.dtype}{hint}")
    if scale.device != device:
        raise ValueError(f"{ctx}: scale is on {scale.device} and the weight on {device}; both must be "
                         f"on the serving device")
    s = scale.detach()
    if major == 12:
        if tuple(s.shape) != (n, kb):
            raise ValueError(
                f"{ctx}: on sm_120/sm_121 the MXFP8 scale of an [N, K] = [{n}, {k}] weight is int32 "
                f"[N, K/128] = [{n}, {kb}], got {list(s.shape)}; a scale produced on another "
                f"architecture is not valid here")
        if kb == 1:
            return s.contiguous()  # one word per row: both stride forms are the same bytes
        if s.stride() == (1, n):
            return s  # the N-fastest form quantize_1x32_fp8 returns
        if s.stride() == (kb, 1):
            return s.t().contiguous().t()  # the .contiguous() form of a round trip, restored to N fastest
        raise ValueError(
            f"{ctx}: the MXFP8 scale has strides {tuple(s.stride())}; it must be the N-fastest tensor "
            f"quantize_1x32_fp8 returns (strides (1, {n})) or its .contiguous() copy (strides ({kb}, 1)). "
            f"Any other view (a transpose, a slice) does not describe the scale bytes the GEMM reads")
    want = n * kb
    if s.dim() != 1 or s.numel() != want:
        raise ValueError(
            f"{ctx}: on sm_100/sm_103 the MXFP8 scale of an [N, K] = [{n}, {k}] weight is a 1-D int32 "
            f"tensor of N * K/128 = {want} words in the CUTLASS block-scaled atom layout, got shape "
            f"{list(s.shape)}; a scale produced on another architecture is not valid here")
    if s.stride() != (1,):
        raise ValueError(f"{ctx}: the MXFP8 scale has stride {tuple(s.stride())}; it must be the "
                         f"contiguous 1-D tensor quantize_1x32_fp8 returns")
    return s


def _ue8m0_scale(scale: torch.Tensor, n: int, k: int, device, major: int, ctx: str) -> torch.Tensor:
    """Validate the uint8 UE8M0 scale of an MXFP8 [N, K] checkpoint as it is stored and
    pack it into the layout the GEMM reads.

    The checkpoint form is one byte per 32 elements along K, row-major, ``[N, K/32]``,
    byte ``b`` meaning the scale ``2^(b - 127)``: the same bytes on every architecture.
    Any strides are accepted, because a uint8 tensor's element order says which byte is
    which. The bytes are packed by ``_ue8m0_scale_handle`` (the packer of
    ``fso.moe.prepare_experts``) into exactly the tensor ``quantize_1x32_fp8`` returns
    for the same scales on this architecture: int32 ``[N, K/128]`` with strides
    ``(1, N)`` on sm_120/sm_121, a 1-D int32 tensor in the CUTLASS block-scaled atom
    layout on sm_100/sm_103. Nothing is requantized.
    """
    want = (n, k // 32)
    if tuple(scale.shape) != want:
        raise ValueError(f"{ctx}: the uint8 UE8M0 scale of an [N, K] = [{n}, {k}] weight is [N, K/32] = "
                         f"{list(want)} (one byte per 32 elements along K, row-major, byte b = "
                         f"2^(b - 127)), got uint8 {list(scale.shape)}")
    if scale.device != device:
        raise ValueError(f"{ctx}: scale is on {scale.device} and the weight on {device}; both must be "
                         f"on the serving device")
    packed = _ue8m0_scale_handle(scale.detach().unsqueeze(0))[0]
    if major != 12:
        return packed
    # sm_120/sm_121: allocate exactly the [N, K/128] tensor with strides (1, N) that
    # quantize_1x32_fp8 allocates. Taking the transpose of the packed words instead gives
    # the same bytes, but at K/128 == 1 torch treats the stride of the size-1 dimension as
    # free and the result would not carry the quantizer's strides.
    out = torch.empty_strided((n, k // 128), (1, n), dtype=torch.int32, device=packed.device)
    out.copy_(packed.t())
    return out


def _dequantize_block_fp8(wq: torch.Tensor, scale: torch.Tensor, r0: int, r1: int) -> torch.Tensor:
    """bf16 values of rows ``[r0, r1)`` of a block-FP8 weight: each fp8 value times the
    fp32 scale of its 128x128 block, computed in fp32 and rounded once to bf16, the
    rule of ``fso.moe.prepare_experts``. ``r0`` and ``r1`` are multiples of 128."""
    rows, k = r1 - r0, int(wq.shape[1])
    w = wq[r0:r1].to(torch.float32)
    w.view(rows // 128, 128, k // 128, 128).mul_(scale[r0 // 128:r1 // 128].view(rows // 128, 1, k // 128, 1))
    return w.to(torch.bfloat16)


def _requantize_block_fp8(wq: torch.Tensor, scale: torch.Tensor, major: int):
    """MXFP8 1x32 weight and scale of a block-FP8 checkpoint ``wq`` [N, K] (N and K
    multiples of 128), converted a whole number of 128-row blocks at a time.

    The 1x32 quantizer works row by row along K, so converting in row chunks gives
    the bytes of one call over the whole weight. The chunks' scales are placed in
    the layout one call would produce: on sm_120/sm_121 row ``r`` of the N-fastest
    ``[N, K/128]`` tensor (strides ``(1, N)``), on sm_100/sm_103 the atom layout's words of rows
    ``[r0, r1)``, which are the contiguous range ``[r0 * K/128, r1 * K/128)``
    because the atom layout stores each 128-row block in one contiguous run.
    """
    n, k = int(wq.shape[0]), int(wq.shape[1])
    kb = k // 128
    rows = max(128, (_CONVERT_BUDGET_BYTES // (k * 4)) // 128 * 128)
    if rows >= n:
        return quantize_1x32_fp8(_dequantize_block_fp8(wq, scale, 0, n))
    q_out = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=wq.device)
    if major == 12:
        s_out = torch.empty_strided((n, kb), (1, n), dtype=torch.int32, device=wq.device)
    else:
        s_out = torch.empty((n * kb,), dtype=torch.int32, device=wq.device)
    for r0 in range(0, n, rows):
        r1 = min(r0 + rows, n)
        q, s = quantize_1x32_fp8(_dequantize_block_fp8(wq, scale, r0, r1))
        q_out[r0:r1].copy_(q)
        if major == 12:
            s_out[r0:r1].copy_(s)
        else:
            s_out[r0 * kb:r1 * kb].copy_(s)
        del q, s
    return q_out, s_out


def prepare_weight(w, *, format, scale=None) -> DenseWeight:
    """Prepare one dense weight for this device, once, at load time.

    Args:
        w: the weight ``[N, K]`` (out_features, in_features) on the CUDA device
            that will serve it: float8_e4m3fn with its ``scale``, or bf16 without
            one.
        format: what the caller holds.

            * ``"bsfp8"``: a float8_e4m3fn weight with fp32 128x128 block scales
              ``scale [ceil(N/128), K/128]`` (value = fp8 * scale), the block-FP8
              checkpoint dialect; or a bf16 weight and no scale, quantized here.
            * ``"mxfp8"``: a bf16 weight and no scale, quantized here to MXFP8
              1x32; or a float8_e4m3fn weight with its MXFP8 scale, in one of two
              forms. The portable form is the one an MXFP8 checkpoint stores:
              uint8 ``[N, K/32]``, one UE8M0 byte per 32 elements along K,
              row-major, byte ``b`` meaning the scale ``2^(b - 127)``, the same
              bytes on every architecture; it is packed here into this device's
              layout. The arch-native form is the int32 scale that
              :func:`fish_scales_ops.compat.quantize_1x32_fp8` returns for the
              weight on this architecture, accepted in the N-fastest form (strides
              ``(1, N)``) the quantizer returns and in the ``.contiguous()`` form a safetensors
              round trip produces; the latter is rewritten to the layout the GEMM
              reads.
        scale: the scale of a float8_e4m3fn weight; omitted for a bf16 one.

    What happens per architecture:

    * sm_90, ``"bsfp8"``: a block-FP8 checkpoint is kept as it is (made
      contiguous if it is not), and a bf16 weight is quantized with
      ``quantize_128x128_fp8`` and its sm_90 default, fp32 scales ``amax / 448``.
      The handle's kind is ``"bsfp8"``.
    * sm_100/sm_103 and sm_120/sm_121, ``"bsfp8"``: a block-FP8 checkpoint is
      dequantized block by block (fp8 value times its block scale in fp32,
      rounded to bf16) and requantized to MXFP8 1x32, the rule of
      ``fso.moe.prepare_experts``; that is a second quantization of already
      quantized weights. A bf16 weight is quantized to MXFP8 directly. Kind
      ``"mxfp8"``.
    * sm_100/sm_103 and sm_120/sm_121, ``"mxfp8"``: served natively; a bf16 weight
      is quantized, and a float8_e4m3fn weight keeps its bytes, with only its scale
      packed or restored to the layout the GEMM reads. Kind ``"mxfp8"``.
    * sm_90 with ``"mxfp8"``, any other architecture and any other format raise
      (``NotImplementedError`` for what is not served, ``ValueError`` for a
      malformed argument), naming the architecture; nothing falls back.

    Constraints: ``K % 128 == 0`` on every architecture, and ``N % 128 == 0`` on
    sm_100/sm_103 and sm_120/sm_121, whose MXFP8 GEMMs require it.

    Memory: a block-FP8 checkpoint on sm_100/sm_103 and sm_120/sm_121 is
    converted a whole number of 128-row blocks at a time, holding at most about
    256 MiB of fp32 scratch plus the matching bf16 and fp8 rows, so the
    conversion never holds a second full-precision copy of the weight. The other
    routes allocate only the prepared weight and its scale. On sm_90 a
    block-FP8 checkpoint is kept as the caller's own tensors, so freeing the
    source parameters afterwards releases no memory there. An MXFP8 checkpoint on
    sm_100/sm_103 and sm_120/sm_121 keeps the caller's weight tensor as well; its
    scale is a new tensor when it arrives as uint8 bytes or in the row-major int32
    form.

    Alignment: every tensor of the returned handle starts on a 16-byte boundary
    (the int32 scale of sm_100/sm_103 on a 32-byte one), which the GEMMs need. A
    kept checkpoint tensor that is a view at another offset is copied into a
    tensor of its own here, once; :func:`linear` never copies a weight.

    Returns:
        :class:`DenseWeight`.
    """
    arch = _device_arch()
    where = _arch_label(arch)
    if not isinstance(format, str) or format.strip().lower() not in FORMATS:
        raise _format_error(format, where)
    fmt = format.strip().lower()
    if not supported(fmt):
        raise _arch_error(fmt, arch)
    major = arch // 10
    ctx = f"fso.dense.prepare_weight (format {fmt!r}, {where})"
    if not isinstance(w, torch.Tensor):
        raise ValueError(f"{ctx}: w must be a tensor, got {type(w).__name__}")
    if w.dim() != 2:
        raise ValueError(f"{ctx}: w must be 2-D [N, K] (out_features, in_features), got shape "
                         f"{tuple(w.shape)}")
    if not w.is_cuda:
        raise ValueError(f"{ctx}: w must be on the CUDA device that will serve it, got {w.device}")
    dev_major = torch.cuda.get_device_capability(w.device)[0]
    if dev_major != major:
        raise ValueError(
            f"{ctx}: the weight is on {w.device}, an sm_{dev_major}x device, while fish_scales_ops "
            f"dispatches on device 0 ({where}); mixed architectures in one process are not supported")
    n, k = int(w.shape[0]), int(w.shape[1])
    if n < 1 or k < 1:
        raise ValueError(f"{ctx}: w is empty, shape {tuple(w.shape)}")
    if k % 128 != 0:
        raise ValueError(f"{ctx}: in_features K = {k} is not a multiple of 128, which every "
                         f"block-scaled GEMM here requires (one scale per 128 elements along K)")
    kind = "bsfp8" if major == 9 else "mxfp8"
    if kind == "mxfp8" and n % 128 != 0:
        raise ValueError(f"{ctx}: out_features N = {n} is not a multiple of 128, which the MXFP8 "
                         f"GEMMs of sm_100/sm_103 and sm_120/sm_121 require")
    if w.dtype not in (torch.float8_e4m3fn, torch.bfloat16):
        raise ValueError(f"{ctx}: w must be float8_e4m3fn (with its scale) or bf16 (quantized here), "
                         f"got {w.dtype}")
    is_fp8 = w.dtype == torch.float8_e4m3fn
    if is_fp8 and scale is None:
        need = ("fp32 128x128 block scales [ceil(N/128), K/128]" if fmt == "bsfp8"
                else f"the checkpoint's uint8 UE8M0 scale [N, K/32] = [{n}, {k // 32}] or the int32 "
                     f"MXFP8 scale quantize_1x32_fp8 returns for it")
        raise ValueError(f"{ctx}: a float8_e4m3fn weight needs its scale, {need}; a bf16 weight is "
                         f"quantized here and takes none")
    if not is_fp8 and scale is not None:
        takes = ("fp32 128x128 block scales [ceil(N/128), K/128]" if fmt == "bsfp8"
                 else f"its uint8 UE8M0 scale [N, K/32] = [{n}, {k // 32}] or the int32 scale of "
                      f"quantize_1x32_fp8")
        raise ValueError(f"{ctx}: a bf16 weight is quantized here and takes no scale; pass scale "
                         f"only with a float8_e4m3fn weight, which takes {takes}")
    wd = w.detach()

    if fmt == "bsfp8":
        if is_fp8:
            s = _block_scale(scale, n, k, w.device, ctx)
            if kind == "bsfp8":
                return DenseWeight("bsfp8", fmt, arch, k, n, _aligned_storage(wd.contiguous()), _aligned_storage(s))
            q, sq = _requantize_block_fp8(wd.contiguous(), s, major)
        elif kind == "bsfp8":
            q, sq = quantize_128x128_fp8(wd)  # the sm_90 default: fp32 scales amax / 448
            return DenseWeight("bsfp8", fmt, arch, k, n, q, sq)
        else:
            q, sq = quantize_1x32_fp8(wd)
        return _complete_handle(DenseWeight("mxfp8", fmt, arch, k, n, q, sq))

    if is_fp8:
        if isinstance(scale, torch.Tensor) and scale.dtype == torch.uint8:
            sq = _ue8m0_scale(scale, n, k, w.device, major, ctx)
        else:
            sq = _mxfp8_scale(scale, n, k, w.device, major, ctx)
        return _complete_handle(DenseWeight("mxfp8", fmt, arch, k, n, _aligned_storage(wd.contiguous()),
                                            _aligned_storage(sq, 32 if major == 10 else 16)))
    q, sq = quantize_1x32_fp8(wd)
    return _complete_handle(DenseWeight("mxfp8", fmt, arch, k, n, q, sq))


def _complete_handle(dw: DenseWeight) -> DenseWeight:
    """Return ``dw`` once its tensors are complete for every kernel launched after this point (sm_100/sm_103).

    Why. On sm_100/sm_103 the GEMMs of the decode row read the weight and its scale before their
    programmatic-dependent-launch wait (the weights-first prologue, issue #12 FB1). Such a GEMM can therefore
    overlap the kernel launched right before the call on the same stream, and a weight written by that kernel could
    be read before it is complete. This function's own last write is such a kernel: the quantize of a
    bf16 weight (a programmatic-dependent launch itself, which releases its dependents early), or the scale repack
    of a checkpoint. So the handle ends with one plain launch (a one-word fill, no programmatic serialisation) on the
    current stream: a plain launch starts only once every kernel in front of it on the stream has completed and its
    writes are visible, and anything launched after it can overlap at most that fill. ``prepare_weight`` followed
    directly by ``linear`` is then safe (test_dense_v1, case x). The fill costs one tiny launch per prepared weight,
    at load time; nothing synchronises the host."""
    if dw.arch // 10 == 10:
        torch.zeros(1, dtype=torch.int32, device=dw.weight.device)
    return dw




# ---------------------------------------------------------------------------------------------------------------------
# The activation contract (version 1): the xq / x_scale form quantize_activation produces and linear(x_scale=...) takes.
# ---------------------------------------------------------------------------------------------------------------------

def _align(v, a: int):
    return (v + a - 1) // a * a


def _act_scale_meta(major: int, m, k: int):
    """(dtype, shape, strides) of the activation scale of an [m, k] activation on this architecture family: exactly
    the tensor quantize_activation returns (quantize_1x128_contract on sm_90, quantize_1x32_fp8 on sm_100/sm_103
    and sm_120/sm_121). ``m`` may be a SymInt (the fakes)."""
    kb = k // 128
    if major == 9:
        mp = _align(m, 4)
        return torch.float32, (mp, kb), (1, mp)
    if major == 12:
        mp = _align(m, 4)
        return torch.int32, (mp, kb), (1, mp)
    return torch.int32, (_align(m, 128) * kb,), (1,)


def _empty_act_scale(major: int, m, k: int, device) -> torch.Tensor:
    dtype, shape, strides = _act_scale_meta(major, m, k)
    return torch.empty_strided(shape, strides, dtype=dtype, device=device)


def _contract_layout(major: int) -> dict:
    """The x_scale part of activation contract v1 on this architecture family, as data."""
    if major == 9:
        return {
            "format": "bsfp8_1x128",
            "x_scale_dtype": "float32",
            "x_scale_shape": "[align(M, 4), K/128]",
            "x_scale_strides": "(1, align(M, 4)): M fastest, so torch indexing is exact (x_scale[m, kb] is the scale "
                               "of row m, group kb)",
            "x_scale_index": "scale(m, kb) = x_scale[m, kb] = x_scale.data_ptr()[kb * align(M, 4) + m]",
            "padded_rows": "zero (0.f)",
            "rule": "per 128-element group of a row: amax = max |x| in fp32, floored at bf16(1e-10f) = 1.00044e-10 "
                    "(fp32 bits 0x2EDC0000) for every K; qs = 448.f / amax; scale = 1.f / qs (fp32, not a power of "
                    "two); code = cvt.rn.satfinite.e4m3(x * qs). This is the quantize linear(x, w) runs inside "
                    "linear_qx; fso.compat.quantize_1x128_fp8 differs from it for K % 512 == 0 (it floors at 1e-10f) "
                    "on a group whose amax is below 1.00044e-10, and leaves the padded rows unwritten when "
                    "K % 512 != 0",
            "header_functions": ["fso_bsfp8_amax_floor()", "fso_bsfp8_quant_scale(amax, &scale)",
                                 "fso_bsfp8_quantize4(x0, x1, x2, x3, qs)", "fso_bsfp8_scale_index(m, kb, m_pad)"],
        }
    common_rule = ("per 32-element block of a row: amax = max |x| in fp32, floored at 1e-10f; q = 448.f / amax; "
                   "d = 1.f / q; b = exp_field(d) + (mantissa(d) != 0), capped at 254; code = "
                   "cvt.rn.satfinite.e4m3(x * 2^(127 - b)); word(m, kp) = b0 | b1 << 8 | b2 << 16 | b3 << 24 with b_i "
                   "the byte of block 4 * kp + i; byte b means the scale 2^(b - 127)")
    if major == 12:
        return {
            "format": "mxfp8_1x32",
            "x_scale_dtype": "int32",
            "x_scale_shape": "[align(M, 4), K/128]",
            "x_scale_strides": "(1, align(M, 4)): M fastest, so torch indexing is exact (x_scale[m, kp] is word "
                               "(m, kp))",
            "x_scale_index": "word(m, kp) = x_scale[m, kp] = x_scale.data_ptr()[kp * align(M, 4) + m]",
            "padded_rows": "zero words",
            "rule": common_rule,
            "header_functions": ["fso_mx_amax_floor()", "fso_e8m0_from_amax(amax, &quant_scale)",
                                 "fso_mx_quantize4(x0, x1, x2, x3, quant_scale)", "fso_mx_pack_word(b0, b1, b2, b3)",
                                 "fso_mx_sf_word_index_sm120(m, kp, m_pad)"],
        }
    return {
        "format": "mxfp8_1x32",
        "x_scale_dtype": "int32",
        "x_scale_shape": "[align(M, 128) * K/128] (1-D)",
        "x_scale_strides": "(1,)",
        "x_scale_index": "word(m, kp) = x_scale.data_ptr()[((m / 128) * (K/128) + kp) * 128 + (m % 32) * 4 + "
                         "(m % 128) / 32] (the CUTLASS Sm1xxBlockScaledConfig atom layout)",
        "padded_rows": "zero words",
        "rule": common_rule,
        "header_functions": ["fso_mx_amax_floor()", "fso_e8m0_from_amax(amax, &quant_scale)",
                             "fso_mx_quantize4(x0, x1, x2, x3, quant_scale)", "fso_mx_pack_word(b0, b1, b2, b3)",
                             "fso_mx_sf_word_index_sm100(m, kp, k)"],
    }


def _contract_want(major: int, m: int, k: int) -> str:
    dtype, shape, strides = _act_scale_meta(major, m, k)
    dt = "fp32" if dtype == torch.float32 else "int32"
    if major in (9, 12):
        form = "M fastest: element (m, j) at j * align(M, 4) + m"
    else:
        form = "the 1-D Sm1xx atom layout"
    return f"{dt} {list(shape)} with strides {tuple(strides)} ({form}) for an [M, K] = [{m}, {k}] activation"


def activation_contract() -> dict:
    """The activation contract of this device, as data: what :func:`quantize_activation` returns and
    ``linear(xq, weight, x_scale=sx)`` accepts, and the quantizer rule that produces it, bit for bit. The same
    contract is stated in C++ by ``fso/activation_contract.h`` under :func:`include_dir`, for a caller that writes the
    quantized activation from its own kernel (``kFsoActivationContractVersion``). The version changes whenever the
    layout or the rule does.

    Keys: ``version`` (1), ``arch``, ``format`` (``"mxfp8_1x32"`` on sm_100/sm_103 and sm_120/sm_121,
    ``"bsfp8_1x128"`` on sm_90), ``xq`` (dtype and layout of the quantized activation), ``k_multiple`` (128), the
    ``x_scale_*`` entries (dtype, shape, strides, physical index of a scale word, padded rows), ``rule``,
    ``header`` and ``header_functions``, ``include_dir``. Raises NotImplementedError on an architecture with no
    dense route.
    """
    arch = _device_arch()
    major = arch // 10
    if major not in (9, 10, 12):
        raise NotImplementedError(f"fso.dense.activation_contract: this device ({_arch_label(arch)}) has no dense "
                                  f"route and therefore no activation contract; the contract exists on sm_90, "
                                  f"sm_100/sm_103 and sm_120/sm_121")
    out = {"version": _CONTRACT_VERSION, "arch": _arch_label(arch)}
    layout = _contract_layout(major)
    out["format"] = layout.pop("format")
    out["xq"] = {"dtype": "float8_e4m3fn", "shape": "[M, K]", "layout": "row-major, contiguous, 16-byte aligned"}
    out["k_multiple"] = 128
    out.update(layout)
    out["header"] = "fso/activation_contract.h"
    out["include_dir"] = include_dir()
    return out


def include_dir() -> str:
    """The directory that holds ``fso/activation_contract.h`` (pass it as an include path, e.g. ``-I`` or
    ``extra_include_paths`` of ``torch.utils.cpp_extension.load_inline``). It ships inside the package."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "include")


def _check_x_scale(op: str, major: int, x_scale, m: int, k: int, device) -> None:
    """Validate ``x_scale`` against activation contract v1 from its metadata (dtype, device, shape, strides,
    alignment); a mismatch is a ValueError naming the contract version and the expected layout."""
    want = _contract_want(major, m, k)
    head = f"{op}: x_scale does not follow activation contract v1 of this device (expected {want})"
    if not isinstance(x_scale, torch.Tensor):
        raise ValueError(f"{head}; got {type(x_scale).__name__}")
    dtype, shape, strides = _act_scale_meta(major, m, k)
    if x_scale.dtype != dtype:
        raise ValueError(f"{head}; got dtype {x_scale.dtype}. Produce it with fso.dense.quantize_activation on "
                         f"this device")
    if x_scale.device != device:
        raise ValueError(f"{head}; got a tensor on {x_scale.device}, the weight is on {device}")
    if x_scale.dim() != len(shape) or tuple(x_scale.shape) != tuple(shape):
        raise ValueError(f"{head}; got shape {list(x_scale.shape)}. A scale produced on another architecture or "
                         f"for another M is not valid here")
    if x_scale.numel() > 0:
        # The [align(M, 4), K/128] layouts (sm_90, sm_120/sm_121) are M fastest: the strides are compared on the dims
        # of size > 1, which with K/128 == 1 leaves only the unit stride of the M dim (is_contiguous).
        if major in (9, 12) and k // 128 > 1:
            ok = x_scale.stride() == tuple(strides)
        else:
            ok = x_scale.is_contiguous()
        if not ok:
            hint = ""
            if major in (9, 12) and x_scale.is_contiguous():
                hint = (". These are row-major strides (K/128, 1), the strides of a .contiguous() copy of the "
                        "M-fastest tensor" + (" and of fso.compat.quantize_1x128_fp8's scale" if major == 9 else "")
                        + "; pass the tensor quantize_activation returned")
            raise ValueError(f"{head}; got strides {tuple(x_scale.stride())}{hint}")
        if x_scale.data_ptr() % 16 != 0:
            raise ValueError(f"{head}; its data pointer is not 16-byte aligned (offset {x_scale.data_ptr() % 16})")


def _check_xq(op: str, x, device) -> None:
    """The pre-quantized activation: float8_e4m3fn [M, K], 2-D, contiguous, on the weight's device, 16-byte aligned."""
    if x.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{op}: with x_scale, x must be the float8_e4m3fn [M, K] activation of "
                         f"fso.dense.quantize_activation (activation contract v1), got {x.dtype}")
    if x.device != device:
        raise ValueError(f"{op}: x is on {x.device} and the weight on {device}")
    if not x.is_contiguous():
        raise ValueError(f"{op}: the float8_e4m3fn activation must be contiguous (row-major [M, K], activation "
                         f"contract v1), got strides {tuple(x.stride())}")
    if x.numel() > 0 and x.data_ptr() % 16 != 0:
        raise ValueError(f"{op}: the float8_e4m3fn activation's data pointer is not 16-byte aligned (offset "
                         f"{x.data_ptr() % 16}); activation contract v1 requires it")


def _x_bf16_storage(x):
    """The bf16 activation (x_scale None, and quantize_activation's x) in storage the quantize kernels can read.
    They read it in vectors from its data pointer, and a contiguous view at an element offset that is not a multiple
    of 8 can start below the 16-byte boundary their widest loads need (the raw ops underneath refuse what their kernel
    cannot read, input_alignment.h). Such an x is copied into a tensor of its own (one copy kernel; the caching
    allocator returns blocks aligned far beyond 16 bytes), so every bf16 x the interface accepts runs; a
    non-contiguous x is copied by .contiguous() further down anyway. Called in the op bodies, which run on real
    tensors: under torch.compile the fakes stand in for them, and inside a CUDA-graph capture the copy is captured
    with the call."""
    if x.numel() > 0 and x.is_contiguous() and x.data_ptr() % 16 != 0:
        return x.clone()
    return x


def _check_operands(op: str, x, weight, x_scale) -> None:
    """Shapes and dtypes of x and weight, and the pairing of an fp8 x with its x_scale."""
    if x.dim() != 2 or weight.dim() != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError(f"{op}: x must be [M, K] and weight [N, K] with the same K, got "
                         f"{tuple(x.shape)} and {tuple(weight.shape)}; fso.dense.linear flattens the "
                         f"leading dimensions of x")
    if x_scale is None:
        if x.dtype == torch.float8_e4m3fn:
            raise ValueError(f"{op}: x is float8_e4m3fn but no x_scale was given; a pre-quantized activation comes "
                             f"with its scale (xq, x_scale = fso.dense.quantize_activation(x)), and a bf16 x is "
                             f"quantized inside the op")
        if x.dtype != torch.bfloat16 or weight.dtype != torch.float8_e4m3fn:
            raise ValueError(f"{op}: x must be bf16 and weight float8_e4m3fn, got {x.dtype} and "
                             f"{weight.dtype}")
    else:
        if x.dtype == torch.bfloat16:
            raise ValueError(f"{op}: x_scale was given with a bf16 x; x_scale belongs to the float8_e4m3fn x that "
                             f"fso.dense.quantize_activation returns, and a bf16 x is quantized inside the op")
        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError(f"{op}: weight must be float8_e4m3fn, got {weight.dtype}")


def _check_weight_scale(op: str, major: int, kind: str, scale, n: int, k: int) -> None:
    """The handle's kind against this architecture, and its scale's metadata."""
    if kind == "bsfp8":
        if major != 9:
            raise NotImplementedError(
                f"{op}: kind 'bsfp8' is the sm_90 block-FP8 GEMM; prepare the weight on this device "
                f"with fso.dense.prepare_weight, which serves block-FP8 checkpoints as MXFP8 on "
                f"sm_100/sm_103 and sm_120/sm_121")
        if scale.dtype != torch.float32 or tuple(scale.shape) != ((n + 127) // 128, k // 128):
            raise ValueError(f"{op}: scale must be the fp32 [ceil(N/128), K/128] block scales, got "
                             f"{scale.dtype} {list(scale.shape)}")
        return
    if kind == "mxfp8":
        if major not in (10, 12):
            raise NotImplementedError(
                f"{op}: kind 'mxfp8' needs MXFP8 hardware (sm_100/sm_103 or sm_120/sm_121); on sm_90 "
                f"prepare the weight with format='bsfp8'")
        kb = k // 128
        if major == 12:
            ok = (scale.dtype == torch.int32 and tuple(scale.shape) == (n, kb)
                  and (kb == 1 or scale.stride() == (1, n)))
            want = f"int32 [{n}, {kb}] with strides (1, {n}) (N fastest)"
        else:
            ok = scale.dtype == torch.int32 and scale.dim() == 1 and scale.numel() == n * kb \
                and scale.stride() == (1,)
            want = f"a contiguous 1-D int32 tensor of {n * kb} words (the atom layout)"
        if not ok:
            raise ValueError(f"{op}: scale must be {want}, got {scale.dtype} {list(scale.shape)} "
                             f"strides {tuple(scale.stride())}; fso.dense.prepare_weight produces it")
        return
    raise ValueError(f"{op}: unknown kind; the kinds are 'bsfp8' and 'mxfp8', produced by "
                     f"fso.dense.prepare_weight")


def _check_weight_aligned(op: str, kind: str, weight, scale) -> None:
    """The weight, and on kind ``"mxfp8"`` its int32 scale, must start on a 16-byte boundary: every route reads them in
    place with TMA, and a weight is never copied per call. Every tensor :func:`prepare_weight` returns does. The fp32
    block scales of kind ``"bsfp8"`` are read with plain loads, at any address. Asked before the activation quantize,
    so a refused call launches nothing."""
    for name, t in (("weight", weight), ("scale", scale)) if kind == "mxfp8" else (("weight", weight),):
        if t.numel() > 0 and t.data_ptr() % 16 != 0:
            raise ValueError(f"{op}: the {name}'s data pointer is {t.data_ptr() % 16} bytes past a 16-byte boundary; "
                             f"the GEMM reads it in place with TMA, which needs a 16-byte aligned address. "
                             f"fso.dense.prepare_weight returns aligned tensors; a handle built from views at other "
                             f"offsets is refused, and a weight is never copied per call")


def _sm120_route(m: int, n: int, k: int, bf16x: bool, quantized: bool = False, sms=None, device=None) -> str:
    """The key of :data:`_epilogue.SERVED` of the sm_120/sm_121 route this call takes: ``"sm120.smallm"`` where the
    small-M route's rule picks it (the bf16-x call's R2 or the 2b call's R1; host arithmetic in the extension,
    ``sm120_smallm_rules``, on a device of ``sms`` multiprocessors, by default the SM count of ``device`` or of the
    current device), else ``"sm120.mxfp8"``, the cascade. A ``linear_quantized`` call (``quantized``) always runs on
    the cascade: the small-M route does not write the quantized output."""
    if quantized:
        return "sm120.mxfp8"
    r = torch.ops.fish_scales_ops.sm120_smallm_rules(m, n, k, _sm_count(device) if sms is None else int(sms))
    return "sm120.smallm" if r[1 if bf16x else 0] else "sm120.mxfp8"


def _route_require(major: int, m: int, n: int, k: int, epilogue, op: str, bf16x: bool = False, device=None) -> None:
    """Refuse ``epilogue`` (NotImplementedError naming the architecture and the route) unless the route this call
    takes applies it. Host-only, and asked before the activation quantize, so a refused call launches nothing.
    ``bf16x``: the call carries a bf16 activation (the routes of a 2b call and of a bf16-x call can differ on
    sm_120/sm_121)."""
    if epilogue is None:
        return
    if major == 9:
        _epilogue.require("sm90", epilogue, op)
    elif major == 12:
        _epilogue.require(_sm120_route(m, n, k, bf16x, epilogue.quantized, device=device), epilogue, op)
    else:
        from ..gemm import _sm100_dispatch
        _sm100_dispatch.require(m, n, k, epilogue, op)


def _gemm(major: int, kind: str, x, weight, scale, x_scale, epilogue):
    """The GEMM of one validated dense call. With x_scale the activation is already quantized; otherwise it is
    quantized here (sm_100/sm_103, sm_120/sm_121) or inside linear_qx (sm_90)."""
    kw = {}
    if epilogue is not None:
        kw = {"bias": epilogue.bias, "residual": epilogue.residual, "activation": epilogue.activation}
    if kind == "bsfp8":
        if x_scale is not None:
            # sm_90: the checked M-fastest scale, relabelled in place as the buffer the GEMM reads (never permuted)
            return linear_fp8(x, weight.contiguous(), _sm90_act_scale(x_scale), scale.contiguous(), **kw)
        return torch.ops.fish_scales_ops.linear_qx(x.contiguous(), weight.contiguous(), scale.contiguous(),
                                                   *(() if epilogue is None else _epilogue.raw_args(epilogue)))
    if x_scale is not None:
        return linear_mxfp8(x, weight, x_scale, scale, **kw)
    if major == 12:
        # One op for every M: the small-M route's one-launch form (the activation quantized inside the kernel) where
        # its rule picks it, else quantize_1x32_packed + linear_mxfp8_raw inside the same op; the bytes of
        # linear_mxfp8(quantize_1x32_fp8(x)...) either way.
        return torch.ops.fish_scales_ops.linear_mxfp8_bf16x(
            x.contiguous(), weight, scale, *(() if epilogue is None else _epilogue.raw_args(epilogue)))
    xq, sx = quantize_1x32_fp8(x)
    return linear_mxfp8(xq, weight, sx, scale, **kw)


def _gemm_quantized(major: int, kind: str, x, weight, scale, x_scale, epilogue, op: str):
    """The GEMM of one validated linear_quantized call, its result in this device's activation contract: a bf16 x is
    quantized first (as for linear), then ``linear_mxfp8_quantized_raw`` writes (xq, x_scale) of the [M, N] result in
    the GEMM's own launch(es). sm_100/sm_103: on the C++ cascade (tier 3). The call goes to the raw op directly, not
    through ``_sm100_dispatch.route``: no other tier serves the quantized output, so the feature-aware router's plan
    for it is always the cascade (``_route_require`` has checked that plan before this point). sm_120/sm_121: on every
    route of the MXFP8 cascade. sm_90 does not serve it in this build (``_route_require`` refuses it before this
    point; the raise below is the backstop)."""
    if major == 10 and kind == "mxfp8":
        if x_scale is None:
            x, x_scale = quantize_1x32_fp8(x)
        # Both scales are the 1-D atom-layout tensors of this architecture (contiguous already).
        return torch.ops.fish_scales_ops.linear_mxfp8_quantized_raw(
            x.contiguous(), weight.contiguous(), x_scale.contiguous(), scale.contiguous(), epilogue.bias,
            epilogue.act_code)
    if major == 12 and kind == "mxfp8":
        if x_scale is None:
            x, x_scale = quantize_1x32_fp8(x)
        # The M- and N-fastest scales keep their byte order (see linear_mxfp8): never .contiguous() such a view.
        return torch.ops.fish_scales_ops.linear_mxfp8_quantized_raw(
            x.contiguous(), weight.contiguous(), x_scale.contiguous() if x_scale.is_contiguous() else x_scale,
            scale.contiguous() if scale.is_contiguous() else scale, epilogue.bias, epilogue.act_code)
    raise NotImplementedError(f"{op}: no route writes the quantized output on this architecture in this build "
                              f"(interface v1)")


# The schemas are literal strings in the decorators: scripts/gen_op_schemas.py reads them from the source.
@torch.library.custom_op(
    "fish_scales_ops::dense_linear",
    mutates_args=(),
    schema="(Tensor x, Tensor weight, Tensor scale, str kind, Tensor? bias=None, Tensor? x_scale=None, "
           "Tensor? residual=None, str? activation=None) -> Tensor",
)
def _dense_linear_op(x, weight, scale, kind, bias=None, x_scale=None, residual=None, activation=None):
    """The body of ``torch.ops.fish_scales_ops.dense_linear``: check the arguments'
    metadata, then dispatch on the handle's kind and on this device's
    architecture. Every check reads shapes, strides and dtypes only, so the op
    neither synchronizes the device nor depends on tensor values. The four-argument
    call is today's op, unchanged."""
    major = sm_major()
    _batch_invariant_note(major)
    where = _arch_label(_device_arch())
    op = f"fish_scales_ops::dense_linear ({where}, kind {kind!r})"
    _check_operands(op, x, weight, x_scale)
    m, n, k = int(x.shape[0]), int(weight.shape[0]), int(x.shape[1])
    _check_weight_scale(op, major, kind, scale, n, k)
    epilogue = _epilogue.make(op, m=m, n=n, device=weight.device, bias=bias, residual=residual,
                              activation=activation)
    if m > 0:  # an empty x reads no weight
        _check_weight_aligned(op, kind, weight, scale)
    if x_scale is not None:
        _check_xq(op, x, weight.device)
        _check_x_scale(op, major, x_scale, m, k, weight.device)
    else:
        x = _x_bf16_storage(x)
    if m == 0:
        return x.new_empty((0, n), dtype=torch.bfloat16)
    _route_require(major, m, n, k, epilogue, op, bf16x=x_scale is None, device=weight.device)
    return _gemm(major, kind, x, weight, scale, x_scale, epilogue)


@_dense_linear_op.register_fake
def _dense_linear_fake(x, weight, scale, kind, bias=None, x_scale=None, residual=None, activation=None):
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError(f"fish_scales_ops::dense_linear: x must be [M, K] and weight [N, K], got "
                         f"{x.dim()}-D and {weight.dim()}-D tensors")
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op(
    "fish_scales_ops::dense_linear_quantized",
    mutates_args=(),
    schema="(Tensor x, Tensor weight, Tensor scale, str kind, Tensor? bias=None, Tensor? x_scale=None, "
           "str? activation=None) -> (Tensor, Tensor)",
)
def _dense_linear_quantized_op(x, weight, scale, kind, bias=None, x_scale=None, activation=None):
    """The body of ``torch.ops.fish_scales_ops.dense_linear_quantized``: the arguments of ``dense_linear`` (no
    residual), and the result written in this device's activation contract instead of bf16. sm_120/sm_121 write it on
    every route of the MXFP8 cascade and sm_100/sm_103 on the C++ cascade, where the feature-aware router sends every
    such call; elsewhere (sm_90), after the checks, the op raises NotImplementedError naming the architecture and the
    route, before anything is launched."""
    major = sm_major()
    _batch_invariant_note(major)
    where = _arch_label(_device_arch())
    op = f"fish_scales_ops::dense_linear_quantized ({where}, kind {kind!r})"
    _check_operands(op, x, weight, x_scale)
    m, n, k = int(x.shape[0]), int(weight.shape[0]), int(x.shape[1])
    _check_weight_scale(op, major, kind, scale, n, k)
    if n % 128 != 0:
        raise ValueError(f"{op}: out_features N = {n} is not a multiple of 128; the quantized result is the next "
                         f"GEMM's activation, whose K must be a multiple of 128")
    epilogue = _epilogue.make(op, m=m, n=n, device=weight.device, bias=bias, activation=activation,
                              quantized=True)
    if m > 0:  # an empty x reads no weight
        _check_weight_aligned(op, kind, weight, scale)
    if x_scale is not None:
        _check_xq(op, x, weight.device)
        _check_x_scale(op, major, x_scale, m, k, weight.device)
    else:
        x = _x_bf16_storage(x)
    if m == 0:
        return (x.new_empty((0, n), dtype=torch.float8_e4m3fn), _empty_act_scale(major, 0, n, x.device))
    _route_require(major, m, n, k, epilogue, op)
    return _gemm_quantized(major, kind, x, weight, scale, x_scale, epilogue, op)


@_dense_linear_quantized_op.register_fake
def _dense_linear_quantized_fake(x, weight, scale, kind, bias=None, x_scale=None, activation=None):
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError(f"fish_scales_ops::dense_linear_quantized: x must be [M, K] and weight [N, K], got "
                         f"{x.dim()}-D and {weight.dim()}-D tensors")
    m, n = x.shape[0], weight.shape[0]
    return (x.new_empty((m, n), dtype=torch.float8_e4m3fn), _empty_act_scale(sm_major(), m, int(n), x.device))


def _quantize_activation_checks(x, op: str) -> int:
    """bf16 [..., K] on CUDA with K % 128 == 0 on an architecture with a dense route; returns this major."""
    major = sm_major()
    arch = _device_arch()
    if major not in (9, 10, 12):
        raise NotImplementedError(f"{op}: this device ({_arch_label(arch)}) has no dense route and no activation "
                                  f"contract; quantize_activation exists on sm_90, sm_100/sm_103 and sm_120/sm_121")
    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16 or x.dim() < 1:
        got = f"{x.dtype} {tuple(x.shape)}" if isinstance(x, torch.Tensor) else type(x).__name__
        raise ValueError(f"{op}: x must be bf16 [..., K], got {got}")
    if not x.is_cuda:
        raise ValueError(f"{op}: x must be on the CUDA device, got {x.device}")
    if x.shape[-1] % 128 != 0:
        raise ValueError(f"{op}: K = {x.shape[-1]} is not a multiple of 128, which activation contract v1 requires "
                         f"(one scale word per 128 elements along K)")
    return major


@torch.library.custom_op(
    "fish_scales_ops::dense_quantize_activation",
    mutates_args=(),
    schema="(Tensor x) -> (Tensor, Tensor)",
)
def _dense_quantize_activation_op(x):
    """The body of ``torch.ops.fish_scales_ops.dense_quantize_activation``: this device's activation quantizer under
    a stable name, on x flattened to [M, K]. sm_90: ``quantize_1x128_contract``, the activation quantize linear_qx
    runs inside the op (fp32 scales, the bf16 amax floor for every K), with the padded rows' scales zeroed.
    sm_100/sm_103 and sm_120/sm_121: ``quantize_1x32_fp8``."""
    op = f"fish_scales_ops::dense_quantize_activation ({_arch_label(_device_arch())})"
    major = _quantize_activation_checks(x, op)
    k = int(x.shape[-1])
    x2 = _x_bf16_storage(x.reshape(-1, k))
    m = int(x2.shape[0])
    if m == 0:
        return (x.new_empty((0, k), dtype=torch.float8_e4m3fn), _empty_act_scale(major, 0, k, x.device))
    if major == 9:
        xq, sx = torch.ops.fish_scales_ops.quantize_1x128_contract(x2.contiguous())
        # The raw op writes element (m, kb) at kb * align(M, 4) + m under row-major metadata; the contract's tensor
        # is the M-fastest view of the same buffer (strides (1, align(M, 4))), so that x_scale[m, kb] is exact.
        sx = sx.view(k // 128, sx.shape[0]).t()
    else:
        xq, sx = quantize_1x32_fp8(x2)
    return xq.view(m, k), sx


@_dense_quantize_activation_op.register_fake
def _dense_quantize_activation_fake(x):
    k = x.shape[-1]
    m = math.prod(x.shape[:-1]) if x.dim() > 1 else 1
    return (x.new_empty((m, k), dtype=torch.float8_e4m3fn), _empty_act_scale(sm_major(), m, int(k), x.device))


def _trim_none(*args) -> tuple:
    """The optional trailing arguments of an op call without its trailing Nones, so that a call without them is
    the old four-argument call, node for node."""
    args = list(args)
    while args and args[-1] is None:
        args.pop()
    return tuple(args)


def _check_handle(weight, where: str, ctx: str) -> None:
    if not isinstance(weight, DenseWeight):
        raise TypeError(f"{ctx} ({where}): weight must be the DenseWeight handle from "
                        f"fso.dense.prepare_weight, got {type(weight).__name__}")
    arch = _device_arch()
    if weight.arch // 10 != arch // 10:
        raise ValueError(
            f"{ctx}: the weight was prepared on sm_{weight.arch} and this device is "
            f"{where}; the weight and scale layouts are architecture-specific, so prepare it with "
            f"fso.dense.prepare_weight on the serving device")


def _flatten_x(x, weight, x_scale, where: str, ctx: str):
    """x as the op's 2-D [M, K] and the leading shape of the result."""
    if x_scale is None:
        if isinstance(x, torch.Tensor) and x.dtype == torch.float8_e4m3fn:
            raise ValueError(f"{ctx} ({where}, kind {weight.kind!r}): x is float8_e4m3fn but no x_scale was given; "
                             f"pass the scale fso.dense.quantize_activation returned with it, or a bf16 x")
        if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16 or x.dim() < 1 \
                or x.shape[-1] != weight.in_features:
            got = (f"{x.dtype} {tuple(x.shape)}" if isinstance(x, torch.Tensor) else type(x).__name__)
            raise ValueError(f"{ctx} ({where}, kind {weight.kind!r}): x must be bf16 [..., K] "
                             f"with K = in_features = {weight.in_features}, got {got}")
    else:
        if not isinstance(x, torch.Tensor) or x.dtype != torch.float8_e4m3fn or x.dim() != 2 \
                or x.shape[-1] != weight.in_features or not x.is_contiguous():
            got = (f"{x.dtype} {tuple(x.shape)} strides {tuple(x.stride())}" if isinstance(x, torch.Tensor)
                   else type(x).__name__)
            raise ValueError(f"{ctx} ({where}, kind {weight.kind!r}): with x_scale, x must be the float8_e4m3fn "
                             f"[M, K] activation of fso.dense.quantize_activation (2-D, contiguous, K = in_features = "
                             f"{weight.in_features}; activation contract v1), got {got}")
    if x.device != weight.weight.device:
        raise ValueError(f"{ctx} ({where}, kind {weight.kind!r}): x is on {x.device} and the "
                         f"weight on {weight.weight.device}")
    lead = x.shape[:-1]
    if x_scale is not None:
        return x, lead  # already the op's 2-D contiguous [M, K]; nothing is copied
    # (A misaligned x is copied in the op body, _x_bf16_storage: under torch.compile this wrapper is traced on fake
    # tensors, which have no data pointer; reshape copies a non-contiguous x into a buffer of its own.)
    return x.reshape(-1, weight.in_features), lead


def linear(x, weight, *, bias=None, x_scale=None, residual=None, activation=None) -> torch.Tensor:
    """``y = act(x @ W^T + bias) + residual`` for a weight prepared by :func:`prepare_weight`, with one rounding.

    Args:
        x: the activation, on the device the weight was prepared on, in one of two forms:

            * bf16 ``[..., K]`` with ``x_scale=None``: quantized inside the op on every call, to 1x128 block-FP8
              inside ``linear_qx`` on sm_90 and to MXFP8 1x32 by the fused quantize kernel on sm_100/sm_103 and
              sm_120/sm_121 (two launches, today's path);
            * the float8_e4m3fn ``[M, K]`` activation that :func:`quantize_activation` returns (2-D, contiguous),
              with its ``x_scale``: the GEMM alone runs, no quantize.
        weight: the :class:`DenseWeight` handle.
        bias: bf16 ``[N]``, contiguous, on the weight's device, or None.
        x_scale: the scale of a float8_e4m3fn ``x``, in this device's activation contract
            (:func:`activation_contract`, version 1), or None for a bf16 ``x``.
        residual: bf16 of the result's shape ``[..., N]``, contiguous, or None.
        activation: None or ``"gelu_erf"`` (:data:`ACTIVATIONS`); another name is a ValueError, a name that exists
            but is not served (``"gelu_tanh"``, ``"silu"``, ...) a NotImplementedError.

    Returns:
        bf16 ``[..., N]``, a new tensor: ``RN_bf16(act(acc + b) + r)`` where ``acc`` is the fp32 accumulator after
        any split reduction, ``b`` is the bias with +0.0 replaced by -0.0 (a zero bias gives the bytes of no bias),
        ``act`` is the identity or erf-GELU in fp32 and ``r`` is added only when given. Each term is applied once.

    Interface v1 status: ``x_scale`` is served on every architecture. ``bias``, ``residual`` and ``activation`` are
    validated (ValueError for a malformed argument) and passed to the architecture's GEMM, which applies them in
    its own launch where the route serves them: sm_90 on all three of its dense kernels, sm_100/sm_103 on its C++
    cascade (tier 3), sm_120/sm_121 on every route of its MXFP8 cascade. A route that does not serve a field yet
    raises NotImplementedError naming the architecture and the route, before anything is launched. Without them the
    call is today's, with today's bytes.

    The call is ``torch.ops.fish_scales_ops.dense_linear`` on the activation
    flattened to ``[M, K]``, with the result reshaped back. The op is a torch
    custom op with a fake implementation, so ``torch.compile`` keeps it as one
    opaque node, without a graph break and without fake registrations by the
    caller. It can be captured into a CUDA graph after one eager call of the
    same shape on the capturing thread, which creates the library's per-thread
    pools and, on sm_90, JIT-compiles the GEMM; nothing inside the op
    synchronizes the device.
    """
    where = _arch_label(_device_arch())
    _check_handle(weight, where, "fso.dense.linear")
    x2, lead = _flatten_x(x, weight, x_scale, where, "fso.dense.linear")
    r2 = None
    if residual is not None:
        want = tuple(lead) + (weight.out_features,)
        if not isinstance(residual, torch.Tensor) or tuple(residual.shape) != want:
            got = list(residual.shape) if isinstance(residual, torch.Tensor) else type(residual).__name__
            raise ValueError(f"fso.dense.linear ({where}, kind {weight.kind!r}): residual must have the result's "
                             f"shape {list(want)}, got {got}")
        if not residual.is_contiguous():
            raise ValueError(f"fso.dense.linear ({where}, kind {weight.kind!r}): residual must be contiguous, got "
                             f"strides {tuple(residual.stride())}")
        r2 = residual.view(-1, weight.out_features)
    y = torch.ops.fish_scales_ops.dense_linear(
        x2, weight.weight, weight.scale, weight.kind, *_trim_none(bias, x_scale, r2, activation))
    return y.reshape(*lead, weight.out_features)


def linear_quantized(x, weight, *, bias=None, x_scale=None, activation=None):
    """``(xq_next, scale_next)``: the result of :func:`linear` written in this device's activation contract instead
    of bf16, ready to be the next GEMM's ``x`` / ``x_scale``.

    Bit for bit ``quantize_activation(linear(x, weight, bias=bias, x_scale=x_scale, activation=activation))`` when
    the bf16 call runs on the same route: the epilogue rounds to bf16 first and then applies this device's activation
    quantizer rule to the rounded values, so the fused path is checkable against the unfused one on every route. On
    sm_90 and sm_120/sm_121 the two calls always take the same route. On sm_100/sm_103 this call always runs on the
    C++ cascade (no other tier writes the quantized output) while the bf16 call of a decode-row, tier-1 or tier-2
    cell runs on that tier, so there it equals ``quantize_activation`` of the cascade's bf16 result
    (``torch.ops.fish_scales_ops.linear_mxfp8_raw``) and can differ from ``quantize_activation(linear(...))`` in
    individual codes where that tier sums the ``K`` products in another order. ``N`` must be a multiple of 128 (it
    is the next GEMM's ``K``). There is no residual argument. ``x``, ``bias``, ``x_scale`` and ``activation`` are
    those of :func:`linear`; the leading dimensions of a bf16 ``x`` are flattened, and ``xq_next`` is ``[M, N]``.

    Interface v1 status: sm_120/sm_121 serve it on every route of the MXFP8 cascade, and sm_100/sm_103 on the C++
    cascade (tier 3), on every configuration its rules pick and on its two-kernel split-K; the sm_100/sm_103 router
    sends every ``linear_quantized`` call to the cascade, also on shapes whose bf16 call takes the decode row, cuBLAS
    or the DSL tier, since no other tier writes the quantized output. The GEMM writes ``xq_next`` and ``scale_next``
    (padded rows' words zero) in its own launch(es), with no bf16 intermediate in memory and no quantize kernel; a
    bf16 ``x`` is quantized first, as for :func:`linear`. sm_90 validates the arguments and raises
    NotImplementedError naming the architecture and the route, before anything is launched. The op is
    ``torch.ops.fish_scales_ops.dense_linear_quantized`` (a custom op with a fake that returns this device's output
    shapes and dtypes).
    """
    where = _arch_label(_device_arch())
    _check_handle(weight, where, "fso.dense.linear_quantized")
    x2, _ = _flatten_x(x, weight, x_scale, where, "fso.dense.linear_quantized")
    return torch.ops.fish_scales_ops.dense_linear_quantized(
        x2, weight.weight, weight.scale, weight.kind, *_trim_none(bias, x_scale, activation))


def quantize_activation(x):
    """``(xq, x_scale)``: the activation ``x`` (bf16 ``[..., K]``, ``K % 128 == 0``) quantized by this device's
    activation quantizer, in the form ``linear(xq, weight, x_scale=x_scale)`` takes.

    The quantizer is the one ``linear(x, weight)`` runs inside the op, so that ``linear(xq, weight, x_scale=x_scale)``
    gives the bytes of ``linear(x, weight)``: on sm_90 the 1x128 block-FP8 quantize of ``linear_qx`` (fp32 scales
    ``1 / (448 / amax)``, amax floored at the bf16 value of 1e-10 for every K), with the padded rows' scales zeroed;
    it differs from ``fso.compat.quantize_1x128_fp8`` only where that op floors at 1e-10f (K % 512 == 0) on a group
    whose amax is below 1.00044e-10, and in the padded rows. On sm_100/sm_103 and sm_120/sm_121 it is
    ``fso.compat.quantize_1x32_fp8`` (MXFP8, UE8M0 scales), with the same bytes. The leading dimensions of ``x`` are
    flattened: ``xq`` is float8_e4m3fn ``[M, K]``, and
    ``x_scale`` is the tensor of activation contract v1 for ``M`` rows (:func:`activation_contract`,
    ``fso/activation_contract.h``). An empty ``x`` gives empty tensors and launches nothing.

    The op is ``torch.ops.fish_scales_ops.dense_quantize_activation``, a custom op with a fake implementation, so
    ``torch.compile`` keeps it as one opaque node. Nothing in it synchronizes the device.
    """
    return torch.ops.fish_scales_ops.dense_quantize_activation(x)
