"""Top-level three-tier dispatcher for the sm_100/sm_103 MXFP8 GEMM.

Every ``linear_mxfp8`` call on Blackwell datacenter (SM100/SM103) enters
here. The tiers, in order:

0. **Swap-AB kernels** (see :mod:`._sm100_decode`) — the M ≤ 64 decode band
   and the narrow-N long-K band above it. Two vendored NVIDIA/FlashInfer
   CuTe-DSL kernels in swap-AB orientation with an 8/16/32-wide token tile up
   to M = 32 — an in-cluster split-K kernel for narrow-N shapes and the plain
   persistent kernel for wide-N shapes — and, from M = 33 to 64, the
   persistent kernel with a 64-wide token tile and the 2-CTA 256-row weight
   tile in a cluster of two; the long-K narrow-N cells of that upper band stay
   with the C++ cascade. Above M = 64 the row takes the narrow-N (N ≤ 4096)
   long-K (K ≥ 4096) cells up to where tier 1's grid fills the machine: the
   split-K kernel with 64-wide token tiles where its clusters fit one wave,
   the persistent 2-CTA form where they do not, outside the cascade's
   two-kernel split-K and wave-tile rules (issue #12, P5).
   This row exists only when the installed
   ``nvidia-cutlass-dsl`` is at least 4.5.0; below that it is inert and the
   three tiers below behave exactly as they did before it was added.

1. **cuBLAS ``scaled_mm``** (see :mod:`._sm100_smm`) — the band
   ``_sm100_smm.should_route`` assigns to it, chiefly the wide-N shapes.
   cuBLAS's ``nvjet_128x128_128x6_4x1_v_bz`` kernel uses a 4-CTA multicast
   cluster and a 6-stage pipeline there; the band boundaries come from
   measurements on the B300 (runs cited in ``should_route``), whose numbers
   are not restated here. Our fused-quantize output is byte-identical to what
   ``scaled_mm`` reads under ``SWIZZLE_32_4_4`` (both use CUTLASS
   ``Sm1xxBlockScaledConfig<32>``), so cuBLAS gets the same payload —
   no extra kernel launches, no re-permute.

2. **CuTe DSL persistent block-scaled kernel** (see :mod:`._sm100_dsl`)
   — the cells ``_sm100_dsl.pick_config`` names (the square cubic shapes,
   and the wide-N rows when tier 1 is unavailable). It applies an explicit L2
   raster swizzle and a cluster shape per configuration, which the
   CollectiveBuilder does not pick by default.

3. **C++ cascade** (``dispatch_sm100_mxfp8`` in ``dispatch.cuh``) —
   fall-through catch-all. Owns the ``down``/``wo`` narrow-N band above the
   decode band, which uses a two-kernel parallel split-K scheme that neither
   cuBLAS nor the DSL kernel offers, plus a NoSmem-epilogue
   overlapping-accumulator path for square cubic shapes. On short-K shapes
   (K ≤ 2560) its tiles come from a wave rule of their own
   (``pick_short_k_tile``), and tier 1 declines the bias calls of the
   narrow-N cells of that rule just above the decode row
   (``_sm100_smm.short_k_owns``, issue #12 T1).

Env kills: ``FSO_DISABLE_SMM=1`` skips tier 1, ``FSO_DISABLE_DSL=1``
skips tier 2 and the decode row, ``FSO_DISABLE_DECODE_DSL=1`` skips the decode
row alone, all together = pure C++.

Epilogue (interface v1, ``_epilogue.py``) and feature-aware routing (issue
#12, S1b1): the shape picks the route, as above, among the tiers that serve
every field of the call. A tier that owns a cell by shape but does not apply
one of its fields (``_epilogue.SERVED``: the swap-AB row applies bias, residual
and erf-GELU; tier 1 the bias alone, through cuBLAS's own epilogue; tier 2
nothing; the cascade all three) hands the cell to the next tier in the same
order that does (on the narrow-N long-K cells tier 1 owns, the swap-AB row's
persistent form first, while its clusters fit one wave), so the feature is
never absent and never refused for want of a kernel — the C++ cascade, the
last tier, serves everything, the quantized output of ``linear_quantized``
included (issue #12 S3b); no other tier writes the quantized output, so every
``linear_quantized`` call lands on the cascade.
:func:`route` runs the call on the tier it lands on; :func:`plan` names that
tier for a shape and an epilogue without running anything, so a caller that
launches work before the GEMM (the activation quantize of
``fso.dense.linear``, the scale repack of ``linear_fp8``) can refuse first.
The two walk the same candidate list (:func:`_candidates`) and agree by
construction (test_dense_v1, case q, checks it on the device).
"""
from __future__ import annotations

from typing import Iterator, Optional, Tuple

import torch

from . import _epilogue

# The route keys of _epilogue.SERVED in the order the tiers are asked.
DECODE, SMM, DSL, CASCADE = "sm100.decode", "sm100.smm", "sm100.dsl", "sm100.cascade"


def _serves(key: str, epilogue: Optional[_epilogue.Epilogue]) -> bool:
    """True when the route ``key`` applies every field of ``epilogue`` (None: the identity, served everywhere).
    Field names only; the cascade's residual-alignment rule stays with ``_epilogue.not_served``, which the caller
    applies to the tier :func:`plan` returns."""
    if epilogue is None:
        return True
    return all(f in _epilogue.SERVED[key] for f in epilogue.fields())


def _candidates(m: int, n: int, k: int, epilogue: Optional[_epilogue.Epilogue]) -> Iterator[Tuple[str, object]]:
    """The tiers that take a contiguous ``[m, k] x [n, k]`` call with ``epilogue``, in order, each with the
    configuration its picker chose (None for tier 1, which has no configuration). The C++ cascade is not yielded:
    it is the fall-through when the iterator is exhausted. Host-only: the pickers and the availability probes run,
    nothing on the device."""
    # Swap-AB row: the decode band M <= 64 and the narrow-N long-K band above
    # it. It is asked first because every tier below loses those cells — tier
    # 1 and tier 2 cannot narrow their N tile below 128 and tier 3 answers a
    # narrow-N decode cell with two serialised launches or, from M = 33, with a
    # 256-row token tile that is three quarters empty; above M = 64 on narrow
    # N with long K every tier below runs 10-64 CTAs over the whole K.
    # ``pick_config`` returns None for every cell it does not own, so the tiers
    # below are reached exactly as before on those.
    from . import _sm100_decode
    if _serves(DECODE, epilogue):
        cfg = _sm100_decode.pick_config(m, n, k)
        if cfg is not None and _sm100_decode.enabled() and \
                (epilogue is None or _sm100_decode.epilogue_available()):
            yield DECODE, cfg

    # Tier 1: cuBLAS scaled_mm for the small-M wide-N decode band.
    from . import _sm100_smm
    smm_owns = _sm100_smm.should_route(m, n, k)
    # A bias call of a short-K narrow-N cell just above the decode row runs on
    # the C++ cascade's short-K tile instead (``_sm100_smm.short_k_owns``,
    # issue #12 T1); the identity call of the same cell stays on tier 1.
    if smm_owns and epilogue is not None and _sm100_smm.short_k_owns(m, n, k):
        smm_owns = False
    if smm_owns and _serves(SMM, epilogue):
        yield SMM, None
    # A call tier 1 owns by shape but cannot serve (a residual or erf-GELU):
    # on the narrow-N long-K class the swap-AB row's persistent form takes it
    # ahead of the C++ cascade while its clusters fit one wave
    # (``_sm100_decode.pick_fallthrough_config``); the identity and bias-only
    # calls of the same cells stay on tier 1.
    if smm_owns and not _serves(SMM, epilogue) and _serves(DECODE, epilogue):
        cfg = _sm100_decode.pick_fallthrough_config(m, n, k)
        if cfg is not None and _sm100_decode.enabled() and _sm100_decode.epilogue_available():
            yield DECODE, cfg

    # The C++ cascade's wave rule owns the cells where narrowing its N tile
    # fills the machine (see ``_sm100_smm.wave_tile_owns``). Tier 2's kernel is
    # persistent and built on a fixed 128- or 256-wide N tile, so on those cells
    # it leaves the same SMs idle that tier 1 does; without this guard its
    # wide-N rows would swallow the cell before the cascade ever sees it.
    if _sm100_smm.wave_tile_owns(m, n):
        return

    # Tier 2: CuTe DSL kernel for mid-band + peak.
    if _serves(DSL, epilogue):
        from . import _sm100_dsl
        cfg = _sm100_dsl.pick_config(m, n, k)
        if cfg is not None and _sm100_dsl._init() is not None:
            yield DSL, cfg


def route(x_fp8: torch.Tensor, w_fp8: torch.Tensor,
          sx: torch.Tensor, sw: torch.Tensor,
          epilogue: Optional[_epilogue.Epilogue] = None) -> Optional[torch.Tensor]:
    """Return the MXFP8 GEMM result via the best-per-shape tier that serves
    ``epilogue``, or None if all tiers decline (caller falls to the C++ op,
    which applies or refuses ``epilogue`` itself)."""
    if x_fp8.dim() != 2:
        return None
    m, k = x_fp8.shape
    n = w_fp8.shape[0]
    for key, cfg in _candidates(int(m), int(n), int(k), epilogue):
        if key == DECODE:
            from . import _sm100_decode
            y = _sm100_decode.linear_mxfp8_decode(x_fp8, w_fp8, sx, sw, cfg, epilogue)
        elif key == SMM:
            from . import _sm100_smm
            y = _sm100_smm.linear_mxfp8_smm(x_fp8, w_fp8, sx, sw, epilogue)
        else:
            from . import _sm100_dsl
            y = _sm100_dsl.linear_mxfp8_dsl(x_fp8, w_fp8, sx, sw, cfg, epilogue)
        if y is not None:
            return y
    return None


def plan(m: int, n: int, k: int, epilogue: Optional[_epilogue.Epilogue] = None) -> str:
    """The :data:`_epilogue.SERVED` key of the tier :func:`route` takes for a
    contiguous ``[m, k] x [n, k]`` call carrying ``epilogue``: ``"sm100.decode"``,
    ``"sm100.smm"``, ``"sm100.dsl"`` or ``"sm100.cascade"``. Host-only: it asks
    the same pickers and availability probes ``route`` asks, in the same order,
    and runs nothing on the device. (``route`` additionally falls through when
    the decode row is handed an operand whose last dimension is strided, and
    when the CuTe-DSL tier is handed an activation or weight scale that is not
    32-byte aligned, which it cannot take; the cascade that then runs a declined
    DSL call serves every field, while the DSL tier serves none. ``linear_mxfp8``
    copies a row-strided activation before routing.) For an epilogue no earlier
    tier serves (the quantized output of
    ``linear_quantized``, which only the cascade writes) the answer is the
    cascade; :func:`_epilogue.require` names the refusal if the cascade lacks a
    field (none today)."""
    for key, _cfg in _candidates(m, n, k, epilogue):
        return key
    return CASCADE


def served_fields(m: int, n: int, k: int) -> frozenset:
    """The epilogue fields a call of this shape can carry on this device: the union over the tiers of the
    fall-through path, which is the cascade's set, since the cascade is the last tier and serves a superset of
    every other tier's fields (the quantized output of linear_quantized included, on the cascade alone)."""
    return frozenset().union(*(_epilogue.SERVED[k] for k in (DECODE, SMM, DSL, CASCADE)))


def require(m: int, n: int, k: int, epilogue: Optional[_epilogue.Epilogue], ctx: str = "") -> None:
    """Refuse ``epilogue`` with NotImplementedError unless the tier :func:`plan`
    names for ``(m, n, k, epilogue)`` applies every field of it (with the
    feature-aware fall-through only a cascade residual that is not 16-byte
    aligned is refused; ``linear_quantized`` lands on the cascade). Host-only."""
    if epilogue is None:
        return
    _epilogue.require(plan(m, n, k, epilogue), epilogue, ctx)
