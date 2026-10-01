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

``format`` names what the caller holds. ``"bsfp8"`` is a float8_e4m3fn weight
with fp32 128x128 block scales (value = fp8 * scale), or a bf16 weight that is
quantized to block-FP8 at load time. ``"mxfp8"`` is a bf16 weight that is
quantized to MXFP8 1x32 at load time, or a float8_e4m3fn weight together with
the int32 MXFP8 scales that :func:`fish_scales_ops.compat.quantize_1x32_fp8`
produces on this architecture.

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

import torch

from .._arch import sm_major
from ..gemm.fp8 import quantize_128x128_fp8
from ..gemm.mxfp8 import linear_mxfp8, quantize_1x32_fp8
# One architecture probe and one parser of the ``arch`` argument, shared with fso.moe.
from ..moe import _arch_label, _arch_major, _device_arch

__all__ = ["FORMATS", "DenseWeight", "prepare_weight", "linear", "supported", "describe"]

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
    configuration."""
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
                "and the MXFP8 router (the M <= 64 decode row, cuBLAS scaled_mm, the CuTe-DSL tier, "
                "the C++ cascade)",
            12: "served: dequantized block by block at load time and requantized to MXFP8 1x32 (a bf16 "
                "weight is quantized to MXFP8 directly); every call runs the 1x32 activation quantize "
                "and the CUTLASS MXFP8 cascade",
        },
        "mxfp8": {
            9: "not served: sm_90 has no MXFP8 hardware; serve a block-FP8 checkpoint or a bf16 "
               "weight with format 'bsfp8' instead",
            10: "served: a bf16 weight is quantized to MXFP8 1x32 at load time and an MXFP8 checkpoint "
                "is kept as it is; every call runs the 1x32 activation quantize and the MXFP8 router",
            12: "served: a bf16 weight is quantized to MXFP8 1x32 at load time and an MXFP8 checkpoint "
                "is kept as it is, its scale restored to the K-major layout the GEMM reads; every call "
                "runs the 1x32 activation quantize and the CUTLASS MXFP8 cascade",
        },
    }
    heads = {
        "bsfp8": "format 'bsfp8': a float8_e4m3fn weight with fp32 128x128 block scales, or a bf16 "
                 "weight quantized at load time",
        "mxfp8": "format 'mxfp8': a bf16 weight quantized to MXFP8 1x32 at load time, or a "
                 "float8_e4m3fn weight with the int32 MXFP8 scales of quantize_1x32_fp8 on this "
                 "architecture",
    }
    lines = [f"fish_scales_ops.dense: one dense linear for every architecture; this device is "
             f"{_arch_label(arch)}{name}."]
    for fmt in FORMATS:
        lines.append(heads[fmt])
        for fam, label in families:
            mark = "   <- this device" if fam == major else ""
            lines.append(f"  {label:40s} {rows[fmt][fam]}{mark}")
    lines.append("Any other format (a bf16-precision GEMM, int8, fp4, ...) is served on no architecture: "
                 "prepare_weight raises and nothing falls back.")
    if major not in (9, 10, 12):
        lines.append(f"This device ({_arch_label(arch)}) serves no format.")
    lines.append("linear(): x is bf16 [..., K] and the result bf16 [..., N]; the activation is quantized "
                 "inside the op on every call, and there is no bias argument.")
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
            int32 ``[N, K/128]`` in the K-major layout on sm_120/sm_121, and a
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
            f"format='bsfp8' and an MXFP8 checkpoint format='mxfp8'. Otherwise serve this weight "
            f"with another GEMM.")
    return ValueError(
        f"fso.dense.prepare_weight: unknown format {format!r} on {where}; the formats are 'bsfp8' "
        f"(a float8_e4m3fn weight with fp32 128x128 block scales, or a bf16 weight) and 'mxfp8' (a "
        f"bf16 weight, or a float8_e4m3fn weight with the MXFP8 scales of quantize_1x32_fp8), and "
        f"{here}.")


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
    strides ``(1, N)`` (K-major: the words of one 128-wide K block are adjacent
    across rows), and the GEMM reads those bytes by raw pointer. A safetensors
    round trip stores ``.contiguous()``, which keeps every element's value but
    rewrites the bytes row-major, strides ``(K/128, 1)``; handed to the GEMM as it
    is, that tensor would be read in the wrong order and the result would be wrong
    without an error. Both forms are accepted here, and the row-major one is
    rewritten to the K-major layout. With ``K/128 == 1`` the two forms are the
    same bytes. On sm_100/sm_103 the scale is a 1-D tensor in the CUTLASS
    block-scaled atom layout, which a round trip leaves unchanged.
    """
    kb = k // 128
    if not isinstance(scale, torch.Tensor):
        raise ValueError(f"{ctx}: scale must be the int32 tensor fso.compat.quantize_1x32_fp8 returns "
                         f"for the weight on this architecture, got {type(scale).__name__}")
    if scale.dtype != torch.int32:
        raise ValueError(f"{ctx}: the MXFP8 scale must be the int32 tensor fso.compat.quantize_1x32_fp8 "
                         f"returns for the weight on this architecture (four UE8M0 bytes per word), "
                         f"got {scale.dtype}")
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
            return s  # the K-major form quantize_1x32_fp8 returns
        if s.stride() == (kb, 1):
            return s.t().contiguous().t()  # the .contiguous() form of a round trip, restored to K-major
        raise ValueError(
            f"{ctx}: the MXFP8 scale has strides {tuple(s.stride())}; it must be the K-major tensor "
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
    the layout one call would produce: on sm_120/sm_121 row ``r`` of the K-major
    ``[N, K/128]`` tensor, on sm_100/sm_103 the atom layout's words of rows
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
              1x32; or a float8_e4m3fn weight with the int32 scale that
              :func:`fish_scales_ops.compat.quantize_1x32_fp8` returns for it on
              this architecture. That scale is accepted in the K-major form the
              quantizer returns and in the ``.contiguous()`` form a safetensors
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
    * sm_100/sm_103 and sm_120/sm_121, ``"mxfp8"``: served natively. Kind
      ``"mxfp8"``.
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
    source parameters afterwards releases no memory there.

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
                else "the int32 MXFP8 scale quantize_1x32_fp8 returns for it")
        raise ValueError(f"{ctx}: a float8_e4m3fn weight needs its scale, {need}; a bf16 weight is "
                         f"quantized here and takes none")
    if not is_fp8 and scale is not None:
        raise ValueError(f"{ctx}: a bf16 weight is quantized here and takes no scale; pass scale "
                         f"only with a float8_e4m3fn weight")
    wd = w.detach()

    if fmt == "bsfp8":
        if is_fp8:
            s = _block_scale(scale, n, k, w.device, ctx)
            if kind == "bsfp8":
                return DenseWeight("bsfp8", fmt, arch, k, n, wd.contiguous(), s)
            q, sq = _requantize_block_fp8(wd.contiguous(), s, major)
        elif kind == "bsfp8":
            q, sq = quantize_128x128_fp8(wd)  # the sm_90 default: fp32 scales amax / 448
            return DenseWeight("bsfp8", fmt, arch, k, n, q, sq)
        else:
            q, sq = quantize_1x32_fp8(wd)
        return DenseWeight("mxfp8", fmt, arch, k, n, q, sq)

    if is_fp8:
        sq = _mxfp8_scale(scale, n, k, w.device, major, ctx)
        return DenseWeight("mxfp8", fmt, arch, k, n, wd.contiguous(), sq)
    q, sq = quantize_1x32_fp8(wd)
    return DenseWeight("mxfp8", fmt, arch, k, n, q, sq)


@torch.library.custom_op(
    "fish_scales_ops::dense_linear",
    mutates_args=(),
    schema="(Tensor x, Tensor weight, Tensor scale, str kind) -> Tensor",
)
def _dense_linear_op(x, weight, scale, kind):
    """The body of ``torch.ops.fish_scales_ops.dense_linear``: check the arguments'
    metadata, then dispatch on the handle's kind and on this device's
    architecture. Every check reads shapes, strides and dtypes only, so the op
    neither synchronizes the device nor depends on tensor values."""
    major = sm_major()
    where = _arch_label(_device_arch())
    op = f"fish_scales_ops::dense_linear ({where}, kind {kind!r})"
    if x.dim() != 2 or weight.dim() != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError(f"{op}: x must be [M, K] and weight [N, K] with the same K, got "
                         f"{tuple(x.shape)} and {tuple(weight.shape)}; fso.dense.linear flattens the "
                         f"leading dimensions of x")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{op}: x must be bf16 and weight float8_e4m3fn, got {x.dtype} and "
                         f"{weight.dtype}")
    m, n, k = int(x.shape[0]), int(weight.shape[0]), int(x.shape[1])
    if kind == "bsfp8":
        if major != 9:
            raise NotImplementedError(
                f"{op}: kind 'bsfp8' is the sm_90 block-FP8 GEMM; prepare the weight on this device "
                f"with fso.dense.prepare_weight, which serves block-FP8 checkpoints as MXFP8 on "
                f"sm_100/sm_103 and sm_120/sm_121")
        if scale.dtype != torch.float32 or tuple(scale.shape) != ((n + 127) // 128, k // 128):
            raise ValueError(f"{op}: scale must be the fp32 [ceil(N/128), K/128] block scales, got "
                             f"{scale.dtype} {list(scale.shape)}")
        if m == 0:
            return x.new_empty((0, n))
        return torch.ops.fish_scales_ops.linear_qx(x.contiguous(), weight.contiguous(), scale.contiguous())
    if kind == "mxfp8":
        if major not in (10, 12):
            raise NotImplementedError(
                f"{op}: kind 'mxfp8' needs MXFP8 hardware (sm_100/sm_103 or sm_120/sm_121); on sm_90 "
                f"prepare the weight with format='bsfp8'")
        kb = k // 128
        if major == 12:
            ok = (scale.dtype == torch.int32 and tuple(scale.shape) == (n, kb)
                  and (kb == 1 or scale.stride() == (1, n)))
            want = f"int32 [{n}, {kb}] with strides (1, {n}) (the K-major layout)"
        else:
            ok = scale.dtype == torch.int32 and scale.dim() == 1 and scale.numel() == n * kb \
                and scale.stride() == (1,)
            want = f"a contiguous 1-D int32 tensor of {n * kb} words (the atom layout)"
        if not ok:
            raise ValueError(f"{op}: scale must be {want}, got {scale.dtype} {list(scale.shape)} "
                             f"strides {tuple(scale.stride())}; fso.dense.prepare_weight produces it")
        if m == 0:
            return x.new_empty((0, n))
        xq, sx = quantize_1x32_fp8(x)
        return linear_mxfp8(xq, weight, sx, scale)
    raise ValueError(f"{op}: unknown kind; the kinds are 'bsfp8' and 'mxfp8', produced by "
                     f"fso.dense.prepare_weight")


@_dense_linear_op.register_fake
def _dense_linear_fake(x, weight, scale, kind):
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError(f"fish_scales_ops::dense_linear: x must be [M, K] and weight [N, K], got "
                         f"{x.dim()}-D and {weight.dim()}-D tensors")
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)


def linear(x, weight) -> torch.Tensor:
    """``y = x @ W^T`` for a weight prepared by :func:`prepare_weight`.

    Args:
        x: bf16 ``[..., K]`` on the device the weight was prepared on. The
            activation is quantized inside the op on every call: to 1x128
            block-FP8 inside ``linear_qx`` on sm_90, and to MXFP8 1x32 by the
            fused quantize kernel on sm_100/sm_103 and sm_120/sm_121.
        weight: the :class:`DenseWeight` handle.

    Returns:
        bf16 ``[..., N]``, a new tensor. There is no bias argument; a caller with
        a bias adds it to the result.

    The call is ``torch.ops.fish_scales_ops.dense_linear`` on the activation
    flattened to ``[M, K]``, with the result reshaped back. The op is a torch
    custom op with a fake implementation, so ``torch.compile`` keeps it as one
    opaque node, without a graph break and without fake registrations by the
    caller. It can be captured into a CUDA graph after one eager call of the
    same shape on the capturing thread, which creates the library's per-thread
    pools and, on sm_90, JIT-compiles the GEMM; nothing inside the op
    synchronizes the device.
    """
    arch = _device_arch()
    where = _arch_label(arch)
    if not isinstance(weight, DenseWeight):
        raise TypeError(f"fso.dense.linear ({where}): weight must be the DenseWeight handle from "
                        f"fso.dense.prepare_weight, got {type(weight).__name__}")
    if weight.arch // 10 != arch // 10:
        raise ValueError(
            f"fso.dense.linear: the weight was prepared on sm_{weight.arch} and this device is "
            f"{where}; the weight and scale layouts are architecture-specific, so prepare it with "
            f"fso.dense.prepare_weight on the serving device")
    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16 or x.dim() < 1 \
            or x.shape[-1] != weight.in_features:
        got = (f"{x.dtype} {tuple(x.shape)}" if isinstance(x, torch.Tensor) else type(x).__name__)
        raise ValueError(f"fso.dense.linear ({where}, kind {weight.kind!r}): x must be bf16 [..., K] "
                         f"with K = in_features = {weight.in_features}, got {got}")
    if x.device != weight.weight.device:
        raise ValueError(f"fso.dense.linear ({where}, kind {weight.kind!r}): x is on {x.device} and the "
                         f"weight on {weight.weight.device}")
    lead = x.shape[:-1]
    y = torch.ops.fish_scales_ops.dense_linear(
        x.reshape(-1, weight.in_features), weight.weight, weight.scale, weight.kind)
    return y.reshape(*lead, weight.out_features)
