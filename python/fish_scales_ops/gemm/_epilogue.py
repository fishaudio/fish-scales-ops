"""The dense epilogue of interface v1 (issue #12): its arguments, their validation, and what each dense GEMM route
applies today.

A dense GEMM call may carry ``bias`` (bf16 ``[N]``), ``residual`` (bf16 ``[M, N]``), ``activation`` (``None`` or
``"gelu_erf"``) and, for ``fso.dense.linear_quantized``, a quantized output. The result is
``RN_bf16(act(acc + b) + r)`` with one rounding, where ``b`` is the bias with +0.0 replaced by -0.0 (so a zero bias
is the identity) and ``r`` is added only when given. The C++ side of the same contract is
``csrc/gemm/include/blockscale_gemm/common/epilogue.cuh``.

Every check here reads metadata only (dtype, device, shape, strides), so it never synchronizes the device.

``SERVED`` names, per route, the fields the route applies; a route refuses every other field with
``NotImplementedError`` naming the architecture and the route, before it launches or allocates anything. Interface v1
landed the plumbing with every route serving the identity alone; the phases that add an epilogue to a route widen its
entry here (for the C++ routes, in the same change as the matching ``kDenseEpiServed*`` mask of epilogue.cuh). Served
today: bias, residual and erf-GELU on all three sm_90 dense kernels (``"sm90"``), on the sm_100/sm_103 M <= 64
CuTe-DSL decode row (``"sm100.decode"``, both vendored kernels) and C++ cascade (``"sm100.cascade"``: tier 3, every
configuration its rules pick and its two-kernel split-K) and on every sm_120/sm_121 MXFP8 route (``"sm120.mxfp8"``, the
cascade, and ``"sm120.smallm"``, the small-M route of the decode band); the bias alone on sm_100/sm_103 tier 1
(``"sm100.smm"``, cuBLAS scaled_mm's own bias epilogue); nothing on tier 2 (``"sm100.dsl"``); the quantized output
(``fso.dense.linear_quantized``) on every route of the sm_120/sm_121 MXFP8 cascade, which every sm_120/sm_121
``linear_quantized`` call takes (the small-M route does not write it), and on the sm_100/sm_103 C++ cascade. On
sm_100/sm_103 the router is feature-aware (``_sm100_dispatch``): a tier that owns a cell by shape but does not serve
every field of the call hands the cell to the next tier that does, and the cascade serves every field, so on that
architecture every ``linear_quantized`` call runs on the cascade and a well-formed call is refused only for a cascade
residual whose rows are not 16-byte aligned.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import torch

# The activations the epilogue knows, and the ``act`` code of the raw ops (epilogue.cuh, DenseActivation).
ACTIVATIONS = ("gelu_erf",)
ACT_CODES = {None: 0, "gelu_erf": 1}
# Names that exist but that no route serves: refused with NotImplementedError, and any other string with
# ValueError, so a caller can tell "not served" from "misspelt". "gelu" is refused rather than guessed: torch's
# F.gelu defaults to the erf form, which is spelt "gelu_erf" here, and its tanh form is "gelu_tanh".
UNSERVED_ACTIVATIONS = ("gelu", "gelu_tanh", "quick_gelu", "silu", "swish", "relu", "relu2", "swiglu", "geglu")

# Route keys -> (label for messages, frozenset of served field names). Field names: "bias", "residual",
# "activation", "quantized". The sm_100 keys are the tiers of _sm100_dispatch.route.
SERVED: dict[str, frozenset] = {
    # gemm_dispatch_sm90 (DeepGEMM JIT), C++ mask kDenseEpiServedSm90DeepGemm
    "sm90": frozenset({"bias", "residual", "activation"}),
    # the CuTe-DSL swap-AB row (_sm100_decode: the M <= 64 decode band and the narrow-N long-K band above it):
    # both vendored kernels, in fp32 after the reduction (S1b1; the band above M = 64: P5)
    "sm100.decode": frozenset({"bias", "residual", "activation"}),
    # tier 1, cuBLAS scaled_mm (_sm100_smm): scaled_mm(..., bias=), cuBLAS's own fused bias epilogue (probe P-A)
    "sm100.smm": frozenset({"bias"}),
    "sm100.dsl": frozenset(),       # tier 2, the CuTe-DSL persistent kernel (_sm100_dsl): declines every field
    # tier 3, gemm_dispatch_sm100_mxfp8, C++ mask kDenseEpiServedSm100Cascade (issue #12 S1b2; "quantized" is
    # linear_quantized, S3b: every configuration the rules pick through its quantized-output instantiation and the
    # two-kernel split-K through its quantized reduce; it takes bias and the activation, no residual).
    "sm100.cascade": frozenset({"bias", "residual", "activation", "quantized"}),
    # gemm_dispatch_sm120_mxfp8, C++ mask kDenseEpiServedSm120Mxfp8Cascade: every route of the cascade (the
    # single-launch tiles through their dense-epilogue and quantized-output instances, Stream-K through its
    # reduces); the forced diagnostic instances (FSO_FORCE_SMALLM, FSO_FORCE_MIN_BLOCKS=2, FSO_FORCE_SCHED_GROUP)
    # refuse in C++. "quantized" is linear_quantized (bias and activation; it takes no residual).
    "sm120.mxfp8": frozenset({"bias", "residual", "activation", "quantized"}),
    # the sm_120/sm_121 small-M route (decode band M <= 64, issue #12 decision 3; arch/sm120/mxfp8/smallm_tc.cuh),
    # C++ mask kDenseEpiServedSm120Smallm: the epilogue applied once after its cross-warp reduction. It does not write
    # the quantized output: linear_quantized runs on the cascade on every sm_120/sm_121 call.
    "sm120.smallm": frozenset({"bias", "residual", "activation"}),
    "sm120.fp8": frozenset(),       # gemm_dispatch_sm120 (block-FP8 1x128), kDenseEpiServedSm120Fp8Cascade
}

ROUTE_LABELS = {
    "sm90": "sm_90, the DeepGEMM JIT route gemm_dispatch_sm90",
    "sm100.decode": "sm_100/sm_103, the M <= 64 CuTe-DSL decode row",
    "sm100.smm": "sm_100/sm_103, tier 1, cuBLAS scaled_mm",
    "sm100.dsl": "sm_100/sm_103, tier 2, the CuTe-DSL persistent kernel",
    "sm100.cascade": "sm_100/sm_103, tier 3, the C++ CUTLASS cascade gemm_dispatch_sm100_mxfp8",
    "sm120.mxfp8": "sm_120/sm_121, the CUTLASS MXFP8 cascade gemm_dispatch_sm120_mxfp8",
    "sm120.smallm": "sm_120/sm_121, the small-M route of the decode band (M <= 64), sm120.smallm",
    "sm120.fp8": "sm_120/sm_121, the block-FP8 1x128 CUTLASS cascade gemm_dispatch_sm120",
}


@dataclasses.dataclass(frozen=True, eq=False)
class Epilogue:
    """A validated, non-identity dense epilogue. ``None`` stands for the identity everywhere."""

    bias: Optional[torch.Tensor] = None
    residual: Optional[torch.Tensor] = None
    activation: Optional[str] = None
    quantized: bool = False

    def fields(self) -> tuple[str, ...]:
        out = []
        if self.bias is not None:
            out.append("bias")
        if self.residual is not None:
            out.append("residual")
        if self.activation is not None:
            out.append("activation")
        if self.quantized:
            out.append("quantized")
        return tuple(out)

    @property
    def act_code(self) -> int:
        return ACT_CODES[self.activation]


def describe_fields(fields) -> str:
    names = {"bias": "bias", "residual": "residual", "activation": "activation 'gelu_erf'",
             "quantized": "quantized output (linear_quantized)"}
    return ", ".join(names[f] for f in fields)


def check_activation(activation, ctx: str) -> Optional[str]:
    """``None`` or a name of :data:`ACTIVATIONS`. A name of :data:`UNSERVED_ACTIVATIONS` is a NotImplementedError,
    anything else a ValueError."""
    if activation is None:
        return None
    if isinstance(activation, str) and activation in ACTIVATIONS:
        return activation
    if isinstance(activation, str) and activation.strip().lower() in UNSERVED_ACTIVATIONS:
        raise NotImplementedError(
            f"{ctx}: activation {activation!r} is not served by fish_scales_ops; the served activations are "
            f"{ACTIVATIONS} ('gelu_erf' is 0.5 * v * (1 + erf(v / sqrt(2))) in fp32, torch's F.gelu with "
            f"approximate='none'). Apply other activations to the bf16 result")
    raise ValueError(f"{ctx}: activation must be None or one of {ACTIVATIONS}, got {activation!r}")


def make(ctx: str, *, m: int, n: int, device, bias=None, residual=None, activation=None,
         quantized: bool = False) -> Optional[Epilogue]:
    """Validate the epilogue arguments of one dense call of output ``[m, n]`` on ``device`` and return the
    :class:`Epilogue`, or ``None`` when nothing is applied. Metadata only; a malformed argument is a ValueError."""
    act = check_activation(activation, ctx)
    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            raise ValueError(f"{ctx}: bias must be a bf16 tensor [N] = [{n}] or None, got {type(bias).__name__}")
        if bias.dtype != torch.bfloat16:
            raise ValueError(f"{ctx}: bias must be bf16 [N] = [{n}], got {bias.dtype}")
        if bias.dim() != 1 or int(bias.shape[0]) != n:
            raise ValueError(f"{ctx}: bias must be 1-D [N] = [{n}], got shape {list(bias.shape)}")
        if bias.device != device:
            raise ValueError(f"{ctx}: bias is on {bias.device} and the weight on {device}; both must be on the "
                             f"serving device")
        if not bias.is_contiguous():
            raise ValueError(f"{ctx}: bias must be contiguous, got strides {tuple(bias.stride())}")
    if residual is not None:
        if not isinstance(residual, torch.Tensor):
            raise ValueError(f"{ctx}: residual must be a bf16 tensor of the output's shape or None, got "
                             f"{type(residual).__name__}")
        if residual.dtype != torch.bfloat16:
            raise ValueError(f"{ctx}: residual must be bf16, got {residual.dtype}")
        if residual.dim() != 2 or tuple(residual.shape) != (m, n):
            raise ValueError(f"{ctx}: residual must have the output's shape [M, N] = [{m}, {n}], got "
                             f"{list(residual.shape)}")
        if residual.device != device:
            raise ValueError(f"{ctx}: residual is on {residual.device} and the weight on {device}; both must be "
                             f"on the serving device")
        if not residual.is_contiguous():
            raise ValueError(f"{ctx}: residual must be contiguous, got strides {tuple(residual.stride())}")
    if bias is None and residual is None and act is None and not quantized:
        return None
    return Epilogue(bias=bias, residual=residual, activation=act, quantized=quantized)


def not_served(route: str, epilogue: Epilogue, ctx: str = "") -> Optional[NotImplementedError]:
    """The NotImplementedError for ``epilogue`` on ``route`` (a key of :data:`SERVED`), or ``None`` when the route
    applies every field of it."""
    missing = [f for f in epilogue.fields() if f not in SERVED[route]]
    head = f"{ctx}: " if ctx else ""
    if not missing and route == "sm100.cascade" and epilogue.residual is not None \
            and epilogue.residual.data_ptr() % 16 != 0:
        # The cascade's kernels read the residual in 16-byte rows (the TMA epilogue's C descriptor, the direct-store
        # epilogue's vector loads); the C++ dispatch refuses the same call (require_sm100_residual_layout), but only
        # after the activation quantize of a bf16 x would have run, so the check is repeated here, from metadata.
        return NotImplementedError(
            f"{head}the residual starts at an address that is not a multiple of 16 bytes (data_ptr % 16 = "
            f"{epilogue.residual.data_ptr() % 16}); the route this call takes [{ROUTE_LABELS[route]}] reads it in "
            f"16-byte rows. Pass a residual that starts on a 16-byte boundary (any fresh tensor does)")
    if not missing:
        return None
    return NotImplementedError(
        f"{head}the dense epilogue ({describe_fields(missing)}) is not implemented yet on the route this call takes "
        f"[{ROUTE_LABELS[route]}] (fish_scales_ops interface v1); call without it and apply it to the bf16 result")


def require(route: str, epilogue: Optional[Epilogue], ctx: str = "") -> None:
    """Raise :func:`not_served`'s error unless ``route`` applies every field of ``epilogue``."""
    if epilogue is None:
        return
    err = not_served(route, epilogue, ctx)
    if err is not None:
        raise err


def raw_args(epilogue: Optional[Epilogue]) -> tuple:
    """The trailing ``(bias, residual, act)`` arguments of the raw ops (linear_fp8, linear_qx, linear_mxfp8_raw)."""
    if epilogue is None:
        return (None, None, 0)
    return (epilogue.bias, epilogue.residual, epilogue.act_code)
