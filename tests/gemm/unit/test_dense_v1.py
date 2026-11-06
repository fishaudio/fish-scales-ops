#!/usr/bin/env python3
"""Interface v1 of fso.dense (issue #12): the pre-quantized activation (form 2b), the activation contract and its
header, and the epilogue arguments, which this build validates, applies on the routes that serve them (sm_90, the
sm_100/sm_103 C++ cascade and sm_120/sm_121 MXFP8) and refuses elsewhere.

The cases, each run where the hardware exists and skipped with a message elsewhere:

  a.  2b bytes: with xq, sx = quantize_activation(x), linear(xq, w, x_scale=sx) is torch.equal to linear(x, w) for
      N x K in {1280x1280, 5120x1280, 1280x5120, 1280x7680, 2048x1280, 5120x2048, 2048x4096} (the issue-12 tower,
      stem and LLM projections) and M in {1, 7, 78, 129, 1079}, on an input with an all-zero row. quantize_activation
      has the contract's shape and strides and, on random input, the bytes of the compat quantizer (quantize_1x32_fp8
      on sm_100/sm_103 and sm_120/sm_121; on sm_90 quantize_1x128_fp8 on the live rows, because the contract zeroes
      the padded rows that quantize_1x128_fp8 leaves unwritten when K % 512 != 0); on sm_90 its padded rows are zero.
  a2. the sm_90 floor (main-agent decision of 2026-10-28): blocks whose amax lies in (0, 1.00044e-10), at K = 2048
      and the other K of the table: linear(xq, w, x_scale=sx) is torch.equal to linear(x, w), because the contract
      quantizes like linear_qx (the bf16 floor for every K). For K % 512 == 0 the codes of such a block differ from
      fso.compat.quantize_1x128_fp8 (floor 1e-10f), which is asserted so that the case cannot pass vacuously.
  b.  refusals: every malformed argument (x_scale of the wrong dtype, ndim, shape, strides, alignment or device; an
      fp8 x without x_scale and an x_scale with a bf16 x; a non-contiguous or 3-D fp8 x; bias of the wrong dtype,
      length or device; residual of the wrong shape; an unknown activation) raises ValueError; a known epilogue that
      the route of the call does not apply (the fields of _epilogue.SERVED for that route; linear_quantized
      everywhere) and a known-but-unserved activation name raise NotImplementedError naming the architecture and the
      route, for a bf16 x and for the 2b form. None of them launches anything on the device: torch.profiler records
      no CUDA kernel, memcpy or memset during the call (a positive control shows that it records the kernels of a
      served call). The bias / residual / gelu_erf calls whose fields the route serves run and launch instead.
  c.  torch.compile(fullgraph=True) of f(x) = linear(*quantize_activation(x)...): a recording backend sees exactly
      one dense_quantize_activation node and one dense_linear node; inductor's output equals eager.
  d.  fso/activation_contract.h: a small CUDA extension built with torch.utils.cpp_extension.load_inline (cached in
      a temp directory), which includes the header from fso.dense.include_dir() and quantizes a bf16 [M, K] into
      fp8 codes and scale words in this device's layout, gives exactly the bytes of quantize_activation(x) (codes and
      scale, padded words included)
      for M in {1, 7, 78, 129} and K in {1280, 5120}. The input has rows whose amax is below the floor, so the
      floor of each rule is exercised. The same kernel built with --use_fast_math gives the same bytes, and the
      header's host path (run on the CPU) gives the same bytes as its device path.
  e.  CUDA graph: the 2b call (and quantize_activation + the 2b call) captured on a non-default stream after one
      eager call; replay into a NaN-filled output equals eager.
  f.  the fakes: under FakeTensorMode the three dense ops return the shapes, strides and dtypes of the real outputs
      (dense_linear_quantized: the activation contract of an [M, N] activation); quantize_activation, the 2b call and
      two refused calls run under torch.cuda.set_sync_debug_mode("error") without synchronizing the device; the old
      four-argument dense_linear call is unchanged; activation_contract() and include_dir() describe this device and
      the shipped header.

  The sm_90 dense epilogue (charter section 10), each case sm_90 only:
  g.  order-free exact accumulator on every route of gemm_dispatch_sm90 (swap-AB at M = 1, 7, 32 and, above M = 32,
      on four cells of 32 < M < 64, eight of 64 < M <= 96 and five of 96 < M <= 100 / 128 < M <= 160 (one- and
      two-wave launches) where the dispatcher takes it (the kernel name must show the rule's tile) and four plus six
      plus eight neighbours it keeps on the main path; in the default mode, FH1's routes and neighbours: DeepGEMM v2's
      tiles of use_dense_v2_override, single and paired over M or N, the two swap-AB bands with three token tiles and
      swap-AB's 64-row weight tiles at 8 <= M <= 32, the kernel name showing the tile and the cluster, among them two
      former swap-AB cells of 128 < M <= 160 (9728 x
      2560 at M = 144, 12288 x 2048 at M = 152), which now take the v2 kernel's pair over M; DeepGEMM v2's
      kernel and fp8_gemm_kernel at M = 78, 129, 1079, forced in child processes with FSO_SM90_DENSE_DG2=all and =0;
      the default mode in this process; at M = 78 the three N = 1280 shapes take swap-AB in every mode, and at
      M = 129 1280x5120 and 1280x7680 do),
      N x K in {1280x1280, 5120x1280, 1280x5120, 1280x7680}, plus the two-wave
      256 x 128 DG2 tile (1280x1280 at M = 16384 by default, M = 3072 with =all; the kernel name is checked) and, in
      the default mode, the 15 issue-12 cells the picker's rule 5 re-routes to the v2 kernel (a mostly empty last M
      block below M = 1024; the kernel name must show the rule's tile and stage count): x holds
      bf16 integers in
      [-7, 7] with a +-7 in every 128-element group (exact 1x128 codes, scale 1/64), the weight e4m3 integers in
      [-4, 4] with block scale 1 or 2^-8, so the fp32 accumulator is exact in any order and its value is known from
      an fp64 product. Bias (multiples of 1/8), residual (multiples of 1/8, |r| < 2^7) and erf-GELU, in the six
      combinations bias, bias+residual, bias+gelu, gelu, residual, bias+gelu+residual: the output is byte-equal to
      RN_bf16(act(acc + b) + r), with the GELU of a load_inline kernel written from the formula in its _rn form
      (every fp32 operation rounded on its own; checked bit-equal to the plain formula built with -fmad=false), for
      a bf16 x and for the 2b form, through the epilogue variant of the expected kernel (torch.profiler: kernel name; one kernel in
      the 2b form, quantize + GEMM for a bf16 x). The double-rounding variant RN_bf16(RN_bf16(acc) + b) differs on a
      nonzero fraction (the case discriminates). bias = 0 == bias = None, with and without a residual, including the
      -0.0 corner (a zero x row times an all-negative weight row), and on random data.
  h.  fused calls on each route captured on a non-default stream after one eager call; replay into a NaN-filled
      output equals eager (in this process and in the two child modes).
  i.  FSO_DISABLE_PDL=1 (a child process) gives the bytes of the default for every call of (h).
  j.  two different captured fused calls, one per stream, replayed concurrently from one thread 200 times: every
      replay equals the serial result.
  k.  torch.compile(fullgraph=True) of linear(x, w, bias=, residual=, activation=) and of the 2b form: bias, residual
      and activation stay arguments of the one dense_linear node (no add or gelu node); inductor equals eager.
  l.  a fused call allocates exactly what the identity call allocates and does not synchronize the device.
  n.  bias / residual views at an odd element offset (2-byte aligned only, the epilogue's scalar load path) give the
      bytes of aligned copies, on each route (also in the two child modes), for a bf16 x and the 2b form.
  m.  PDL: a producer kernel releases its dependents at once and writes the bias and residual 2 ms later; the fused
      call launched right behind it (both entries, each route; also in the two child modes) gives the bytes of the
      call on the final tensors. Control: a dependent that reads before its griddepcontrol.wait sees stale values.

  The sm_120/sm_121 dense epilogue (S1c), sm_120/sm_121 only:
  o.  sm_120/sm_121, the dense epilogue (bias, erf-GELU, residual) on every route of the MXFP8 cascade, each in its
      own subprocess (the FSO_FORCE_* knobs are read once per process): the natural cascade on the issue-12 shapes,
      the M x K grid {1, 7, 78, 129, 1079} x {1280, 5120, 7680} and shapes on which it splits K (narrow N with
      K >= 9728, tiles_n >= 32 at M <= 32); each of the ten single-launch tiles (FSO_FORCE_TILE) and the seven
      Stream-K tiles (FSO_FORCE_TILE + FSO_FORCE_KSPLIT) on the grid. On the order-free exact construction of charter
      section 10 every variant (bias, residual, gelu_erf and their combinations) equals RN_bf16 of the exact fp32
      value, once rounded, on the bf16-x and the 2b path (on Stream-K the accumulator is the fp32 sum of the slices'
      bf16 partials); the double-rounding variant differs, so the cells discriminate it; bias = 0 is the identity
      (the -0.0 corner and random data); bias / residual written by the kernels right before the GEMM are read.
      FSO_DISABLE_PDL=1 gives the same bytes. The forced diagnostic instances (SmallM, FSO_FORCE_MIN_BLOCKS=2,
      FSO_FORCE_SCHED_GROUP) refuse every field with NotImplementedError naming the instance. Capture on a
      non-default stream (replay into NaN == eager), concurrent replay of two captured calls on two streams from one
      thread (200 x), two Stream-K calls on two streams with their own scratch (memory history: one allocation per
      stream, different addresses; peak torch allocation covers it) replayed and run eagerly concurrently (200 x),
      torch.compile (recording backend and inductor) with the new arguments, and the reference GELU's two spellings.

  The sm_120/sm_121 small-M route (Q120b), sm_120/sm_121 only:
  w.  the decode-band route sm120.smallm (M <= 64: G1 for the 2b call where rule R1 picks it, G2 -- one launch, the
      activation quantized inside the kernel -- for the bf16-x call where R2 picks it), one subprocess per knob
      setting. Per cell the host route query, the rules op and dense._sm120_route agree; the launched kernel names
      show the small-M kernel exactly where the plan says, in the (form, nc, rt, wk, epilogue) instantiation it
      names, and the bf16-x call is one launch where R2 picks it; the identity call and the six epilogue variants
      equal the exact reference on both entries and the bf16-x call equals the 2b call byte for byte (G2 ==
      quantize_activation + G1). Cells: the twelve Q120a shapes at M in {1 .. 64} with the last M each rule routes
      and the first it does not, five held-out (N, K), bias = 0 corners, linear_quantized on routed cells served
      on the cascade with no small-M kernel (both entries equal; case s compares its bytes);
      FSO_SMALLM_ROUTE=1 on every M 1..64 of four shapes (every nc, both tile counts, both warp counts, resident
      and chunked staging); FSO_SMALLM_ROUTE=0 (never the route; the exact identity outputs equal the route's);
      FSO_DISABLE_PDL=1 gives the same bytes; capture on a non-default stream, two captured calls replayed
      concurrently on two streams (200 x), eager calls on two streams; a producer that writes x / xq + sx / bias +
      residual 2 ms after releasing its dependents (PDL) is read correctly; torch.compile keeps one dense_linear
      node and inductor equals eager. `--case w` runs this case alone.

  The sm_100/sm_103 dense epilogue on the C++ cascade (S1b2), sm_100/sm_103 only:
  p.  sm_100/sm_103 only: bias, erf-GELU and residual on the C++ cascade (tier 3), forced (FSO_DISABLE_SMM=1,
      FSO_DISABLE_DSL=1), one subprocess per family: the cascade's own picks on the issue-12 shapes (which include
      narrow-N long-K decode cells that take the two-kernel split-K), every tile family forced under its
      FSO_FORCE_TILE code (21 / 25 with TileN 64-256, 4/128, 20/128, 20/256, 24/256; the parent checks from the
      FSO_PRINT_TILE_INFO lines that exactly the forced dense-epilogue instantiation ran, with and without the
      residual), and the split-K forced (code 9, auto split and FSO_FORCE_KSPLIT=2). Per cell, M in {1, 7, 78, 129,
      1079} (split-K: M <= 129) and N x K over the tower / LLM shapes with K in {1280, 5120, 7680}: the order-free
      exact accumulator of the charter (integer codes, UE8M0 127 weights, a +-4 in every 32-block of x) gives the
      expected bytes RN_bf16(acc + b), RN_bf16(acc + r), RN_bf16(acc + b + r) (both entry paths), and for erf-GELU
      a reference kernel built with load_inline for the extension's target; the double-rounding variant
      RN_bf16(RN_bf16(acc) + b) must differ on a nonzero count of elements. Also: bias = 0 == bias = None on random
      data and on the -0.0 corner (zero x row times an all-negative weight row; the observed byte is printed);
      capture on a non-default stream after one eager call, replay into a NaN-filled output == eager; FSO_DISABLE_PDL
      =1 gives the same bytes (digest of every output of the family); two different split-K calls captured on two
      streams of one thread and replayed concurrently 200 times equal their serial results, and their captures baked
      in two different workspaces (the [fso sm100 splitk] captured lines); torch.compile(fullgraph=True) keeps one
      dense_linear node and equals eager; a residual not 16-byte aligned is refused (NotImplementedError, no device
      event) on every entry; the force rows without a dense-epilogue kernel (7, 8: Stream-K; 3) refuse an epilogue
      call naming the code, with no device event, and keep serving identity calls.

  The sm_100/sm_103 dense epilogue on the decode row and tier 1, and the feature-aware router (S1b1), sm_100/sm_103
  only; one subprocess per family, the card's natural routing (no FSO_DISABLE_* switch):
  q.  decode: section-10 exactness on the M <= 64 CuTe-DSL decode row in its split-K form (M in {1, 7, 8, 16, 32}
      on the narrow-N shapes 1280x5120, 1280x7680, 2048x4096, 1280x1280) and its persistent form (M in {12, 33, 48,
      64}: the 8/16/32-wide token tile on wide N and the 64-wide token tile with the 2-CTA weight tile), both entry
      paths, with the kernel that ran checked by name (one fused kernel for the 2b form), bias = 0 == None, the -0.0
      corner on both forms; capture on a non-default stream, two different fused calls replayed concurrently on two
      streams (200 x), torch.compile. smm: tier 1 (cuBLAS scaled_mm) applies the bias alone on the cells it owns
      (one nvjet kernel; bias = 0 == None and the -0.0 corner through cuBLAS's own epilogue), and a residual / GELU
      call on those cells falls through to the C++ cascade with the exact bytes; plan() names the tier each call
      ran on. agree: plan() == the tier route() ran on, over the issue-12 shapes (tower, stem, LLM tp1 / tp2) and
      the Family A / B / C dense projections at M in {1, 7, 8, 12, 16, 32, 33, 48, 64, 78, 129, 182, 288, 533,
      640, 1079} for the identity, bias, residual and bias + GELU + residual. pdl: a producer that writes the bias
      / residual 2 ms after releasing its dependents, directly ahead of a fused call on the decode row (both
      forms), tier 1, the cascade fall-through and (P5) the narrow-N long-K band and its fall-through from tier 1,
      gives the bytes of the call on the final tensors. wide (issue #12, P5): the swap-AB row above M = 64 on the
      narrow-N long-K cells (o 2048x4096, fc2 1280x5120, conv_out 1280x7680): section-10 exactness on every form
      the rule picks (split-K with 64-wide token tiles in two and four slices; the persistent 2-CTA form with 64-
      and 128-wide token tiles), checked against pick_config and the kernel that ran, both entry paths; the
      fall-through of residual / GELU calls from the tier-1 cell fc2 M = 1079 onto the row; every tactic of the new
      tile list forced through linear_mxfp8_decode; bias = 0 == None, the -0.0 corner; capture on a non-default
      stream; 200 replays of each captured new tactic bit-identical; two-stream concurrent replay (200 x).
      shortk (issue #12, T1): the C++ cascade's short-K tile rule on natural routing: section-10 exactness on the
      narrow-N short-K cells whose bias calls tier 1 declines (64 < M <= 256: the identity stays on tier 1, every other
      call runs on the cascade) and on cells tier 1 keeps
      whose residual / GELU calls take the rule's tiles, both entry paths; every cascade call of those cells
      (identity, bias, residual, bias + GELU, linear_quantized) runs the tile a test-local mirror of the rule names
      (checked on the MMA atom of the kernel that ran) and plan() names the tier it ran on; the -0.0 corner;
      capture on a non-default stream; two-stream concurrent replay (200 x) of two short-K tiles.
      fb2 (the B300 swap-AB route rules): section-10 exactness, both entry paths, every call checked for the tier it
      ran on, on the cells of each rule: the narrow-N long-K class at 32 < M <= 64 on the split-K form with the
      64-wide token tile (1280 x 5120 in four slices, 2048 / 2560 x 4096 in two); the narrow-N short-K band (two
      slices while the split-K grid has at most 80 CTAs: 1280 x 1280 at M = 65 / 129 / 256, 2048 x 1280 at 117,
      1024 x 2048 at 320, 2560 x 2048 at 96, 2048 x 2048 at 80); the identity call of wide-N short-K cells on the
      1-SM persistent 128 x 64 form, their bias / residual calls on the tier they had; the erf-GELU calls the row
      takes where the cascade's short-K rule would leave its 64-wide tile; each rule's picker against the tactic and
      on the cells one step outside it; linear_quantized planned on the cascade; the -0.0 corner; capture on a
      non-default stream (both entries); four captured calls replayed concurrently on four streams (200 x).
      FSO_DISABLE_PDL=1 (children) gives the digests of the decode, smm, wide, shortk and fb2 families. `--case q`
      runs this case alone.

  The sm_120/sm_121 quantized output, fso.dense.linear_quantized (S3c), sm_120/sm_121 only:
  r.  sm_120/sm_121, linear_quantized on every route of the MXFP8 cascade, each in its own subprocess (the natural
      families with FSO_SMALLM_ROUTE=0, so that the bf16 call of a decode-band cell runs on the cascade as well): on
      random data, linear_quantized(x, w, bias=, activation=) gives, bit for bit, quantize_activation(linear(x, w,
      bias=, activation=)) -- xq and every scale word of the contract layout, the padded rows' words included -- for the
      identity, bias, erf-GELU and bias + erf-GELU, on the bf16-x and the 2b entry, on M in {1, 7, 78, 129, 533,
      1079} x N x K in {5120 x 1280, 1280 x 1280, 2048 x 1280, 1280 x 5120}, on shapes the cascade splits K on (the
      quantized Stream-K reduce), and on each of the ten single-launch tiles and the seven Stream-K tiles forced; then
      the chained GEMM linear(xq_next, w2, x_scale=scale_next) == linear(y, w2); bias = 0 == bias = None.
      FSO_DISABLE_PDL=1 gives the same bytes. The forced diagnostic instances refuse it with NotImplementedError naming
      the instance and the quantized output (no device work on the 2b entry). Capture on a non-default stream (replay
      into NaN-filled xq and all-ones scale words == eager), concurrent replay of two captured calls on two streams
      from one thread (200 x), torch.compile (recording backend: one dense_linear_quantized node; inductor == eager).
      `--case r` runs this case alone.

  The sm_100/sm_103 quantized output, fso.dense.linear_quantized on the C++ cascade (S3b), sm_100/sm_103 only:
  s.  sm_100/sm_103 only (S3b): linear_quantized(...) == quantize_activation(linear(...)) bit for bit (the codes and
      every scale word, the padded rows' words included) on the C++ cascade, forced as in (p), one subprocess per
      family (the cascade's own picks, every forced tile of (p) on one M per band, the split-K with the automatic
      and a forced split, the refusing rows): identity, bias, bias + erf-GELU and erf-GELU alone from both entry
      paths on N x K in {5120 x 1280, 1280 x 1280, 2048 x 1280, 1280 x 5120} and M in {1, 7, 78, 129, 533, 1079};
      the chain linear(xq_next, w2, x_scale=scale_next) == linear(y, w2); the fused call launches as many kernels as
      the bf16 call and no quantize kernel; capture on a non-default stream (replay into a clobbered output ==
      eager); FSO_DISABLE_PDL=1 gives the same digests; two different split-K quantized calls captured on two
      streams and replayed concurrently 200 times; torch.compile(fullgraph=True) keeps one dense_linear_quantized
      node; the real op's output metadata equals its fake's. On every forced TMA tile narrower than 256 the
      erf-GELU calls run the epilogue in two warpgroups and the identity / bias calls in one; the 256-wide and
      direct-store tiles run one for every call (FB3). mwg (FB3): the two-warpgroup kernels of the cascade's own
      picks at fc1 M = 533 / 1079 and proj1 M = 533, each erf-GELU call captured and replayed 200 times
      bit-identically, two of them replayed concurrently on two streams 200 times.
      natural: on the card's own routing, on cells whose bf16 call takes the decode row (its narrow-N long-K band above
      M = 64 and that band's fall-through from tier 1 included, P5), tier 1 or tier 2, the quantized call is planned on
      and runs on the cascade (no other tier's kernel) and equals quantize_activation of the bf16 cascade call
      (linear_mxfp8_raw) from both entries; the codes that differ from quantize_activation(linear(...)) on the bf16
      call's own route are counted and printed.
      sm_120/sm_121 (the small-M route of case w does not write the quantized output, so
      linear_quantized runs on the cascade while the bf16 call of a routed decode-band cell runs on the small-M
      route): two subprocesses on the same cells and data, FSO_SMALLM_ROUTE=0 and the rules, on the issue's TP2 qkv /
      o shards across the decode band, the tower shapes at decode M and K = 9728 at M <= 8 (a Stream-K cascade): the
      quantized call launches no small-M kernel, gives the same bytes from both entries and in both processes, and
      equals quantize_activation(linear(...)) bit for bit with the route off; with the rules the bf16 elements that
      differ from the cascade's and the codes and scale words that differ from quantize_activation(linear(...)) on
      the small-M route are counted and printed. (b) and (f) treat linear_quantized as served where the route of the
      call applies it. `--case s` runs this case alone.

  t.  sm_90 only (FH5): the in-cluster split-K route (fp8_gemm_kernel_swapAB_splitk) of the decode band. The cells
      the rule takes run the split-K kernel at the slice count and token tile the dispatcher's rule gives (the mirror
      _sm90_splitk, with the H200's occupancy at one and at two CTAs per SM), their neighbours past each K class's band
      edge and past 66 weight tiles do not; on the rule's cells every epilogue variant and the identity give section
      10's bytes (exact integer accumulators) from the bf16-x call and the 2b call, bias = 0 gives the bytes of
      bias = None with the -0.0 corner; on random data the bf16-x call equals the 2b call and 20 repeated calls give
      the same bytes; child processes compare the route with the route off (FSO_SM90_DENSE_SPLITK=0: the share of
      elements that differ and the largest difference in bf16 ulps are printed) and the one-slice kernel
      (FSO_SM90_SPLITK_S=1) with the swap-AB route (bit for bit), and run section 10 on forced cells with 2, 3 and 4
      slices (token tiles up to 64, an N that is not a multiple of 64). Cases h, i, j and m carry split-K cells too
      (capture on a side stream, PDL off == on, two streams, and a producer that writes bias / residual, or the weight
      and its scale, 2 ms after releasing its dependents). `--case t` runs this case alone.

  The weight handle and the weights-first GEMMs (issue #12, FB1), sm_100/sm_103 only:
  x.  prepare_weight -> linear back to back, both entries, behind a producer that releases its dependents at once and
      finishes 2 ms later: equal to the synchronized call (prepare_weight ends with a plain launch). A weight written
      by the kernel launched right before a pre-quantized call: on the decode row with weights first it is read before
      it is complete (the control: the call differs), on every other route after (the guard: the call is equal).
      prepare_weight with its plain launch dropped: the differing calls are counted (data). A captured graph whose
      weight bytes are rewritten in place (memcpy, gather kernel) and replayed equals the eager call on the new
      weight. fso.compat.linear_fp8 packing FP32 scales inside the call equals the call on pre-packed scales.
      `--case x` runs this case alone.

usage: python tests/gemm/unit/test_dense_v1.py [--case a|b|c|d|e|f|g|o|w|p|q|r|s|t|x]
"""
import os
import re
import sys
import tempfile

import torch

import fish_scales_ops as fso

C = fso.compat
D = fso.dense

SHAPES = ((1280, 1280), (5120, 1280), (1280, 5120), (1280, 7680), (2048, 1280), (5120, 2048), (2048, 4096))
MS = (1079, 129, 78, 7, 1)

failures: list[str] = []


def _check(ok: bool, msg: str) -> bool:
    if not ok:
        failures.append(msg)
    return ok


def _ok(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def _arch() -> tuple[int, int]:
    return torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)


def _arch_label() -> str:
    major, minor = _arch()
    return f"sm_{major}{minor}" if major else "no CUDA device"


def make_x(M: int, K: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return (torch.randn(M, K, device="cuda") * 0.1).bfloat16()


def make_w(N: int, K: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return (torch.randn(N, K, device="cuda") / K ** 0.5).bfloat16()


def prepared(major: int, N: int, K: int, seed: int):
    return D.prepare_weight(make_w(N, K, seed), format="bsfp8" if major == 9 else "mxfp8")


_SAME_SIZE_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32}


def bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same dtype, shape and strides, and the same bits element by element (compared through an integer view of the
    same element size, which keeps the strides, so NaN codes and -0.0 compare as bits)."""
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape) or a.stride() != b.stride():
        return False
    it = _SAME_SIZE_INT[a.element_size()]
    return torch.equal(a.view(it), b.view(it))


def sm90_scale_rows(sx: torch.Tensor, M: int, K: int, live_only: bool) -> torch.Tensor:
    """The sm_90 scale buffer as [K/128, align(M, 4)] (its physical order: element (m, kb) at kb * align(M, 4) + m),
    fp32 bits as int32; with live_only, the rows m < M only. sx is either the contract's M-fastest tensor (strides
    (1, align(M, 4)), whose transpose is the buffer in memory order) or fso.compat.quantize_1x128_fp8's scale (the
    same buffer under row-major metadata); never a .contiguous() copy, which would read the buffer in another order."""
    kb, mp = K // 128, (M + 3) // 4 * 4
    assert sx.is_contiguous() or sx.stride() == (1, mp), f"not an sm_90 scale layout: strides {tuple(sx.stride())}"
    t = (sx if sx.is_contiguous() else sx.t()).reshape(-1).view(torch.int32).view(kb, mp)
    return t[:, :M] if live_only else t


_SM90_AMAX_FLOOR = torch.tensor(0x2EDC0000, dtype=torch.int32).view(torch.float32).item()  # bf16(1e-10f)


def sm90_reference_scale(x: torch.Tensor) -> torch.Tensor:
    """The contract's fp32 scale of every (row m, 128-element group kb) of the bf16 [M, K] x, computed here from x
    with torch ops (not by the library): amax = max |x| in fp32, floored at bf16(1e-10f); qs = 448.f / amax;
    scale = 1.f / qs, each a single fp32 operation. [M, K/128], row m, column kb."""
    M, K = x.shape
    amax = x.float().view(M, K // 128, 128).abs().amax(dim=-1)
    amax = torch.maximum(amax, torch.full_like(amax, _SM90_AMAX_FLOOR))
    qs = torch.full_like(amax, 448.0) / amax
    return torch.ones_like(qs) / qs


def scales_equal(major: int, a: torch.Tensor, b: torch.Tensor, M: int, K: int) -> tuple[bool, str]:
    """Scale tensors equal in metadata and bytes; on sm_90 with K % 512 != 0 the padded rows are not compared,
    because the quantizer does not write them."""
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape) or a.stride() != b.stride():
        return False, "metadata differs"
    if major == 9 and K % 512 != 0:
        return torch.equal(sm90_scale_rows(a, M, K, True), sm90_scale_rows(b, M, K, True)), "live rows"
    return bytes_equal(a, b), "all words"


def expected_scale_meta(major: int, M: int, K: int):
    kb = K // 128
    if major == 9:
        mp = (M + 3) // 4 * 4
        return torch.float32, (mp, kb), (1, mp)
    if major == 12:
        mp = (M + 3) // 4 * 4
        return torch.int32, (mp, kb), (1, mp)
    return torch.int32, (((M + 127) // 128 * 128) * kb,), (1,)


def meta_ok(t: torch.Tensor, major: int, M: int, K: int) -> bool:
    dtype, shape, strides = expected_scale_meta(major, M, K)
    if t.dtype != dtype or tuple(t.shape) != shape:
        return False
    return t.numel() == 0 or K // 128 == 1 and major != 10 or t.stride() == strides


def compat_quantize(major: int, x: torch.Tensor):
    return C.quantize_1x128_fp8(x) if major == 9 else C.quantize_1x32_fp8(x)


# --- (a) 2b bytes ---------------------------------------------------------------------------------------------------

def case_a(major: int) -> None:
    for N, K in SHAPES:
        dw = prepared(major, N, K, seed=N * 17 + K)
        for M in MS:
            x = make_x(M, K, seed=M * 1009 + N * 17 + K)
            xq, sx = D.quantize_activation(x)
            cq, cs = compat_quantize(major, x)
            q_ok = bytes_equal(xq, cq.view(M, K)) and meta_ok(sx, major, M, K)
            if major == 9:
                s_ok = torch.equal(sm90_scale_rows(sx, M, K, True), sm90_scale_rows(cs, M, K, True))
                pad = sm90_scale_rows(sx, M, K, False)[:, M:]
                pad_ok = torch.equal(pad, torch.zeros_like(pad)) and torch.equal(sx[M:], torch.zeros_like(sx[M:]))
                # torch indexing of the contract tensor is exact: x_scale[m, kb] is the scale of row m, group kb
                idx_ok = torch.equal(sx[:M].contiguous().view(torch.int32),
                                     sm90_reference_scale(x).view(torch.int32))
                _check(idx_ok, f"a N={N} K={K} M={M}: x_scale[m, kb] != the scale of row m, group kb computed from x")
                s_ok = s_ok and idx_ok
                how = "live rows; padded rows zero; x_scale[m, kb] == the scale of (m, kb) computed from x"
            else:
                s_ok, how = scales_equal(major, sx, cs, M, K)
                pad_ok = True
            xz = x.clone()
            if M >= 7:
                xz[3].zero_()  # an all-zero row: every group of it is floored
            xzq, xzs = D.quantize_activation(xz)
            y_ref = D.linear(xz, dw)
            y = D.linear(xzq, dw, x_scale=xzs)
            y_ok = tuple(y.shape) == (M, N) and y.dtype == torch.bfloat16 and bool(torch.isfinite(y).all()) \
                and torch.equal(y, y_ref)
            ok = q_ok and s_ok and pad_ok and y_ok
            _check(q_ok and s_ok and pad_ok, f"a N={N} K={K} M={M}: quantize_activation != the compat quantizer "
                                             f"(codes {q_ok}, scale {s_ok}, padded rows {pad_ok}, {how})")
            _check(y_ok, f"a N={N} K={K} M={M}: linear(xq, w, x_scale=sx) != linear(x, w)")
            print(f"  a  N={N:5d} K={K:5d} M={M:5d}: quantize_activation == compat quantizer (codes, scale {how}, "
                  f"contract metadata); linear(xq, w, x_scale=sx) == linear(x, w) (all-zero row included)  "
                  f"{_ok(ok)}")
        del dw
    torch.cuda.empty_cache()


def _tiny_blocks(M: int, K: int, seed: int) -> torch.Tensor:
    """Random rows with tiny blocks: row 1 entirely at |x| < 1e-10, and in row 2 one 128-element group (and so its
    four 32-element blocks) at |x| < 1e-10 with the rest normal. Every tiny group has 0 < amax < 1.00044e-10."""
    x = make_x(M, K, seed)
    t = (torch.rand(K, device="cuda") * 2 - 1) * 9e-11
    x[1] = t.bfloat16()
    x[2, 128:256] = t[:128].bfloat16()
    amax = x[1].float().abs().max().item()
    assert 0 < amax < 1.00044e-10, amax
    return x


def case_a2(major: int) -> None:
    for N, K in ((5120, 2048), (2048, 4096), (5120, 1280), (1280, 5120)):
        dw = prepared(major, N, K, seed=N + K)
        for M in (7, 78, 1079):
            x = _tiny_blocks(M, K, seed=M + K)
            xq, sx = D.quantize_activation(x)
            y_ref = D.linear(x, dw)
            y = D.linear(xq, dw, x_scale=sx)
            ok = torch.equal(y, y_ref)
            _check(ok, f"a2 N={N} K={K} M={M}: 2b != linear(x, w) on blocks with amax in (0, 1.00044e-10)")
            note = ""
            if major == 9:
                cq, cs = C.quantize_1x128_fp8(x)
                differs = not torch.equal(cq.view(torch.uint8)[1], xq.view(torch.uint8)[1])
                y_compat = C.linear_fp8(cq, dw.weight, cs, dw.scale)
                if K % 512 == 0:
                    _check(differs, f"a2 N={N} K={K} M={M}: compat.quantize_1x128_fp8 gave the contract's codes on a "
                                    f"tiny row at K % 512 == 0; the case does not exercise the floor")
                note = (f"; compat.quantize_1x128_fp8 codes of the tiny row {'differ' if differs else 'are equal'}, "
                        f"compat 2-step GEMM {'==' if torch.equal(y_compat, y_ref) else '!='} linear(x, w)")
            print(f"  a2 N={N:5d} K={K:5d} M={M:5d}: tiny blocks (amax < 1.00044e-10): linear(xq, w, x_scale=sx) == "
                  f"linear(x, w)  {_ok(ok)}{note}")
        del dw
    torch.cuda.empty_cache()


# --- (b) refusals without a launch ----------------------------------------------------------------------------------

def _device_events(fn):
    """(exception or None, number of CUDA-side events: kernels, memcpys, memsets) for one call of fn."""
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    exc = None
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - the caller judges the type
            exc = e
        torch.cuda.synchronize()
    n = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    return exc, n


def _expect(label, exc_type, fn, name_arch=True, route_word=None) -> str:
    exc, n = _device_events(fn)
    arch = _arch_label()
    if exc is None:
        _check(False, f"b {label}: did not raise")
        return "did not raise"
    if type(exc) is not exc_type:
        _check(False, f"b {label}: raised {type(exc).__name__} instead of {exc_type.__name__}: {exc}")
        return f"wrong exception {type(exc).__name__}: {exc}"
    named = (not name_arch) or (arch in str(exc)) or (f"sm_{_arch()[0]}" in str(exc))
    routed = route_word is None or route_word in str(exc)
    _check(named, f"b {label}: the message does not name {arch}: {exc}")
    _check(routed, f"b {label}: the message does not name the route ({route_word!r}): {exc}")
    _check(n == 0, f"b {label}: {n} CUDA events (kernels / copies) ran before the {exc_type.__name__}")
    return f"{exc_type.__name__}, {n} device events{'' if named else ' (arch NOT named)'}" \
           f"{'' if routed else ' (route NOT named)'}"


def _expect_served(label, fn) -> str:
    """A served epilogue call: no exception, and at least one device launch."""
    exc, n = _device_events(fn)
    ok = exc is None and n >= 1
    _check(ok, f"b {label}: a served call raised {type(exc).__name__ if exc else None} ({exc}) or launched nothing "
               f"({n} device events)")
    return f"served, {n} device events" if ok else f"FAILED: {type(exc).__name__ if exc else ''} {exc} ({n} events)"


def _served_fields(major: int, M: int, N: int, K: int, quantized: bool = False) -> set:
    """The epilogue fields (_epilogue.SERVED) a dense call of [M, K] x [N, K] can carry on this device: the key
    fso.dense.linear asks (dense._route_require): "sm90"; on sm_120/sm_121 "sm120.smallm" or "sm120.mxfp8" as
    dense._sm120_route names for the shape (the two routes serve the same bias / residual / activation fields; a
    linear_quantized call, `quantized`, always takes the cascade, the only one of the two that writes the quantized
    output); on sm_100/sm_103 the union over the tiers of the feature-aware fall-through
    (_sm100_dispatch.served_fields: a field some tier serves lands on that tier, so only a field no tier serves is
    refused)."""
    from fish_scales_ops.gemm import _epilogue as E
    if major == 9:
        key = "sm90"
    elif major == 12:
        key = D._sm120_route(M, N, K, bf16x=True, quantized=quantized)
    else:
        from fish_scales_ops.gemm import _sm100_dispatch
        return set(_sm100_dispatch.served_fields(M, N, K))
    return set(E.SERVED[key])


def case_b(major: int) -> None:
    N, K, M = 1280, 1280, 7
    dw = prepared(major, N, K, seed=41)
    x = make_x(M, K, seed=42)
    xq, sx = D.quantize_activation(x)
    L = D.linear
    bias = torch.randn(N, device="cuda").bfloat16()
    res = torch.randn(M, N, device="cuda").bfloat16()
    torch.cuda.synchronize()

    # Positive control: a served call is seen launching.
    exc, n = _device_events(lambda: L(xq, dw, x_scale=sx))
    ctrl = exc is None and n >= 1
    _check(ctrl, f"b: the profiler control saw {n} device events for a served 2b call (exception {exc})")
    print(f"  b  control: a served linear(xq, w, x_scale=sx) shows {n} device event(s) under the profiler  "
          f"{_ok(ctrl)}")

    kb = K // 128
    compat_scale = None
    if major == 9:
        wrong_dtype = sx.view(torch.int32)
        wrong_ndim = sx.reshape(-1)
        wrong_strides = sx.contiguous()  # the row-major copy of the M-fastest tensor
        compat_scale = C.quantize_1x128_fp8(x)[1]  # the same buffer layout under row-major (K/128, 1) metadata
    elif major == 12:
        wrong_dtype = sx.view(torch.float32)
        wrong_ndim = sx.t().contiguous().view(-1)
        wrong_strides = sx.contiguous()  # the row-major copy of the M-fastest tensor
    else:
        wrong_dtype = sx.view(torch.float32)
        wrong_ndim = sx.view(-1, kb)
        wrong_strides = torch.empty(sx.numel() * 2, dtype=torch.int32, device="cuda")[::2]
    # M + 128 changes the scale's shape on every family (on sm_100/sm_103 the 1-D atom layout holds
    # align(M, 128) rows, so an M in the same 128-row block has the same shape and cannot be told apart).
    _, other_m = D.quantize_activation(make_x(M + 128, K, seed=43))
    buf = torch.empty(sx.numel() + 4, dtype=sx.dtype, device="cuda")
    misaligned = buf[1:1 + sx.numel()].view(sx.shape) if sx.dim() == 1 else \
        torch.as_strided(buf, sx.shape, sx.stride(), storage_offset=1)
    # Every argument is built here, before the profiled calls, so that a launch the profiler sees is the op's.
    sx_cpu, bias_f32, bias_cpu, res_f32, x_fp16 = sx.cpu(), bias.float(), bias.cpu(), res.float(), x.half()
    bias_zero = torch.zeros_like(bias)
    x_k200 = make_x(4, 200, seed=44)
    xq_strided = torch.empty(M, 2 * K, dtype=torch.float8_e4m3fn, device="cuda")[:, :K]
    xq_e5m2 = xq.view(torch.uint8).view(torch.float8_e5m2)
    torch.cuda.synchronize()
    results = {
        "x_scale of the wrong dtype": _expect("x_scale dtype", ValueError, lambda: L(xq, dw, x_scale=wrong_dtype)),
        "x_scale of the wrong ndim": _expect("x_scale ndim", ValueError, lambda: L(xq, dw, x_scale=wrong_ndim)),
        "x_scale for another M": _expect("x_scale shape", ValueError, lambda: L(xq, dw, x_scale=other_m)),
        "x_scale with foreign strides": _expect("x_scale strides", ValueError,
                                                lambda: L(xq, dw, x_scale=wrong_strides)),
        **({"x_scale of fso.compat.quantize_1x128_fp8 (strides (K/128, 1))": _expect(
            "x_scale compat strides", ValueError, lambda: L(xq, dw, x_scale=compat_scale))}
           if compat_scale is not None else {}),
        "x_scale not 16-byte aligned": _expect("x_scale alignment", ValueError,
                                               lambda: L(xq, dw, x_scale=misaligned)),
        "x_scale on the CPU": _expect("x_scale device", ValueError, lambda: L(xq, dw, x_scale=sx_cpu)),
        "fp8 x without x_scale": _expect("fp8 x no x_scale", ValueError, lambda: L(xq, dw)),
        "x_scale with a bf16 x": _expect("bf16 x with x_scale", ValueError, lambda: L(x, dw, x_scale=sx)),
        "fp8 x, 3-D": _expect("fp8 x 3-D", ValueError, lambda: L(xq.view(1, M, K), dw, x_scale=sx)),
        "fp8 x, non-contiguous": _expect("fp8 x strided", ValueError, lambda: L(xq_strided, dw, x_scale=sx)),
        "fp8 x, float8_e5m2": _expect("fp8 e5m2", ValueError,
                                      lambda: L(xq_e5m2, dw, x_scale=sx)),
        "bias of the wrong dtype": _expect("bias dtype", ValueError, lambda: L(x, dw, bias=bias_f32)),
        "bias of the wrong length": _expect("bias length", ValueError, lambda: L(x, dw, bias=bias[:N - 128])),
        "bias 2-D": _expect("bias 2-D", ValueError, lambda: L(x, dw, bias=bias.view(1, N))),
        "bias on the CPU": _expect("bias device", ValueError, lambda: L(x, dw, bias=bias_cpu)),
        "residual of the wrong shape": _expect("residual shape", ValueError, lambda: L(x, dw, residual=res[:, :128])),
        "residual of the wrong dtype": _expect("residual dtype", ValueError, lambda: L(x, dw, residual=res_f32)),
        "unknown activation": _expect("unknown activation", ValueError, lambda: L(x, dw, activation="gelu_fast")),
        "linear_quantized, unknown activation": _expect(
            "lq unknown activation", ValueError, lambda: D.linear_quantized(x, dw, activation="nope")),
        "quantize_activation, K % 128 != 0": _expect(
            "qa K", ValueError, lambda: D.quantize_activation(x_k200)),
        "quantize_activation, fp16 x": _expect(
            "qa fp16", ValueError, lambda: D.quantize_activation(x_fp16)),
    }
    route = {9: "DeepGEMM", 12: "cascade", 10: "sm_100/sm_103"}[major]

    def route_word(bf16x: bool) -> str:
        """The word a refusal message must carry: on sm_120/sm_121 the route the call takes, which in the decode
        band may be the small-M route sm120.smallm (Q120b) instead of the cascade."""
        if major == 12 and D._sm120_route(M, N, K, bf16x) == "sm120.smallm":
            return "sm120.smallm"
        return route
    ni = NotImplementedError
    # The epilogue fields the route of this call applies (_epilogue.SERVED, for the route the call takes): a call
    # whose fields are all served must run and launch, any other must be refused without a launch.
    served = _served_fields(major, M, N, K)
    epi_calls = {
        "bias (bf16 x)": (("bias",), "bias", lambda: L(x, dw, bias=bias)),
        "bias (2b)": (("bias",), "bias 2b", lambda: L(xq, dw, x_scale=sx, bias=bias)),
        "zero bias (2b)": (("bias",), "zero bias 2b", lambda: L(xq, dw, x_scale=sx, bias=bias_zero)),
        "residual (bf16 x)": (("residual",), "residual", lambda: L(x, dw, residual=res)),
        "residual (2b)": (("residual",), "residual 2b", lambda: L(xq, dw, x_scale=sx, residual=res)),
        "activation gelu_erf (bf16 x)": (("activation",), "gelu", lambda: L(x, dw, activation="gelu_erf")),
        "activation gelu_erf (2b)": (("activation",), "gelu 2b",
                                     lambda: L(xq, dw, x_scale=sx, activation="gelu_erf")),
        "bias + gelu_erf + residual (2b)": (
            ("bias", "activation", "residual"), "all 2b",
            lambda: L(xq, dw, x_scale=sx, bias=bias, activation="gelu_erf", residual=res)),
    }
    for k, (fields, label, fn) in epi_calls.items():
        results[k] = _expect_served(label, fn) if set(fields) <= served else \
            _expect(label, ni, fn, route_word=route_word("2b" not in label))
    results["activation 'silu' (known, unserved)"] = _expect("silu", ni, lambda: L(x, dw, activation="silu"))
    # linear_quantized: served where the route writes the quantized output (the "quantized" field of _epilogue.SERVED;
    # the sm_120/sm_121 MXFP8 cascade since S3c, which every sm_120/sm_121 linear_quantized call takes, also where the
    # bf16 call of the shape takes the small-M route; the sm_100/sm_103 C++ cascade since S3b, which the feature-aware
    # router sends every such call to), refused with NotImplementedError naming the route elsewhere (sm_90).
    served_q = _served_fields(major, M, N, K, quantized=True)
    lq_x = lambda: D.linear_quantized(x, dw)  # noqa: E731
    lq_2b = lambda: D.linear_quantized(xq, dw, x_scale=sx, bias=bias, activation="gelu_erf")  # noqa: E731
    results["linear_quantized (bf16 x)"] = _expect_served("lq", lq_x) if {"quantized"} <= served_q else \
        _expect("lq", ni, lq_x, route_word=route)
    results["linear_quantized (2b, bias, gelu_erf)"] = _expect_served("lq 2b", lq_2b) \
        if {"quantized", "bias", "activation"} <= served_q else _expect("lq 2b", ni, lq_2b, route_word=route)
    # The raw ops refuse too, before anything runs (sm_90: linear_fp8 and linear_qx; Blackwell: linear_mxfp8_raw).
    ops = torch.ops.fish_scales_ops
    if major == 9:  # served on sm_90 (the DeepGEMM route applies the epilogue)
        results["raw linear_qx(bias=...)"] = _expect_served(
            "raw qx", lambda: ops.linear_qx(x, dw.weight, dw.scale, bias))
        results["raw linear_fp8(act=1)"] = _expect_served(
            "raw fp8", lambda: ops.linear_fp8(xq, dw.weight, sx, dw.scale, None, None, 1))
        results["raw linear_qx(act=7)"] = _expect(
            "raw qx act", ValueError, lambda: ops.linear_qx(x, dw.weight, dw.scale, None, None, 7), name_arch=False)
        results["raw linear_fp8(bias of length N-1)"] = _expect(
            "raw fp8 bias", ValueError, lambda: ops.linear_fp8(xq, dw.weight, sx, dw.scale, bias[1:]),
            name_arch=False)
    else:
        if major == 10:
            # sm_100/sm_103: the raw op goes straight to the C++ cascade, which applies the residual since S1b2 (case
            # p); it still refuses a residual whose rows it cannot read (not 16-byte aligned) before anything runs.
            res_buf = torch.zeros(M * N + 8, dtype=torch.bfloat16, device="cuda")
            res_mis = res_buf[1:1 + M * N].view(M, N)
            torch.cuda.synchronize()
            results["raw linear_mxfp8_raw(misaligned residual)"] = _expect(
                "raw mx", ni, lambda: ops.linear_mxfp8_raw(xq, dw.weight, sx, dw.scale, None, res_mis),
                name_arch=False)
        results["raw linear_mxfp8_raw(act=7)"] = _expect(
            "raw mx act", ValueError, lambda: ops.linear_mxfp8_raw(xq, dw.weight, sx, dw.scale, None, None, 7),
            name_arch=False)
    for k, v in results.items():
        print(f"  b  {k:42s} -> {v}")
    # The messages, once each, for the record (the bias message only where the route refuses a bias).
    msgs = [("x_scale dtype", lambda: L(xq, dw, x_scale=wrong_dtype))]
    if "quantized" not in served:
        msgs.append(("linear_quantized", lambda: D.linear_quantized(x, dw)))
    if "bias" not in served:
        msgs.insert(0, ("bias", lambda: L(x, dw, bias=bias)))
    for label, fn in msgs:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  b  message ({label}): {type(e).__name__}: {e}")


# --- (c) torch.compile ----------------------------------------------------------------------------------------------

def case_c(major: int) -> None:
    N, K = 5120, 1280
    dw = prepared(major, N, K, seed=21)

    def f(x):
        xq, sx = D.quantize_activation(x)
        return D.linear(xq, dw, x_scale=sx)

    xs = {M: make_x(M, K, seed=22 + M) for M in (78, 1079, 7)}
    want = {M: f(x) for M, x in xs.items()}
    graphs = []

    def record(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    rec = torch.compile(f, backend=record, fullgraph=True)
    rec_ok = all(torch.equal(rec(x), want[M]) for M, x in xs.items())
    per_graph = []
    for g in graphs:
        ours = [str(n.target) for n in g.graph.nodes if n.op == "call_function" and "fish_scales_ops" in str(n.target)]
        per_graph.append(ours)
    node_ok = bool(graphs) and all(
        sum("dense_quantize_activation" in t for t in ours) == 1 and sum("dense_linear" in t for t in ours) == 1
        and len(ours) == 2 for ours in per_graph)
    _check(rec_ok and node_ok, f"c: recording backend: {len(graphs)} graphs, fish_scales_ops nodes {per_graph}, "
                               f"equal {rec_ok}")
    print(f"  c  recording backend, fullgraph=True: {len(graphs)} graph(s) for M = 78, 1079, 7; nodes per graph "
          f"{per_graph[0] if per_graph else []} (one dense_quantize_activation, one dense_linear); == eager  "
          f"{_ok(rec_ok and node_ok)}")
    torch._dynamo.reset()
    ind = torch.compile(f, fullgraph=True)
    for M, x in xs.items():
        got = ind(x)
        torch.cuda.synchronize()
        ok = torch.equal(got, want[M])
        _check(ok, f"c: inductor M={M}: compiled != eager")
        print(f"  c  inductor, fullgraph=True, M={M:5d}: compiled == eager  {_ok(ok)}")
    torch._dynamo.reset()


# --- (d) the contract header ----------------------------------------------------------------------------------------

_CONTRACT_CPP = r"""
#include <torch/extension.h>
torch::Tensor contract_quantize(torch::Tensor x, int64_t layout, torch::Tensor q, torch::Tensor s);
torch::Tensor contract_quantize_host(torch::Tensor x, int64_t layout, torch::Tensor q, torch::Tensor s);
int64_t contract_version();
"""

_CONTRACT_CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include "fso/activation_contract.h"

// layout: 0 = BSFP8 1x128 (sm_90), 1 = MXFP8 sm_100/sm_103 atom layout, 2 = MXFP8 sm_120/sm_121 M fastest.
// One thread per scale entry: an (m, 128-element group) for BSFP8, an (m, kp) word of four 32-element blocks for
// MXFP8. Rows m >= M write the padded scale (zero words; for BSFP8 0.f only when the contract zeroes them).
__host__ __device__ inline void quantize_entry(const __nv_bfloat16* x, uint8_t* q, void* s, int64_t M, int64_t K,
                                               int64_t layout, int64_t m, int64_t g) {
  const int64_t K_blocks = K / 128;
  if (layout == 0) {
    const int64_t mp = fso_align(M, 4);
    float* sc = static_cast<float*>(s);
    if (m >= M) { sc[fso_bsfp8_scale_index(m, g, mp)] = 0.f; return; }
    const __nv_bfloat16* row = x + m * K + g * 128;
    float amax = 0.f;
    for (int i = 0; i < 128; ++i) amax = fmaxf(amax, fabsf(__bfloat162float(row[i])));
    amax = fmaxf(amax, fso_bsfp8_amax_floor());
    float scale;
    const float qs = fso_bsfp8_quant_scale(amax, &scale);
    uint32_t* out = reinterpret_cast<uint32_t*>(q + m * K + g * 128);
    for (int i = 0; i < 32; ++i)
      out[i] = fso_bsfp8_quantize4(__bfloat162float(row[4 * i]), __bfloat162float(row[4 * i + 1]),
                                   __bfloat162float(row[4 * i + 2]), __bfloat162float(row[4 * i + 3]), qs);
    sc[fso_bsfp8_scale_index(m, g, mp)] = scale;
    return;
  }
  int32_t* words = static_cast<int32_t*>(s);
  const int64_t idx = layout == 1 ? fso_mx_sf_word_index_sm100(m, g, K) : fso_mx_sf_word_index_sm120(m, g, fso_align(M, 4));
  if (m >= M) { words[idx] = 0; return; }
  uint8_t b[4];
  for (int blk = 0; blk < 4; ++blk) {
    const __nv_bfloat16* row = x + m * K + g * 128 + blk * 32;
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(__bfloat162float(row[i])));
    amax = fmaxf(amax, fso_mx_amax_floor());
    float qs;
    b[blk] = fso_e8m0_from_amax(amax, &qs);
    uint32_t* out = reinterpret_cast<uint32_t*>(q + m * K + g * 128 + blk * 32);
    for (int i = 0; i < 8; ++i)
      out[i] = fso_mx_quantize4(__bfloat162float(row[4 * i]), __bfloat162float(row[4 * i + 1]),
                                __bfloat162float(row[4 * i + 2]), __bfloat162float(row[4 * i + 3]), qs);
  }
  words[idx] = static_cast<int32_t>(fso_mx_pack_word(b[0], b[1], b[2], b[3]));
}

__global__ void contract_kernel(const __nv_bfloat16* x, uint8_t* q, void* s, int64_t M, int64_t M_pad, int64_t K,
                                int64_t layout) {
  const int64_t K_blocks = K / 128;
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= M_pad * K_blocks) return;
  quantize_entry(x, q, s, M, K, layout, t / K_blocks, t % K_blocks);
}

static int64_t padded_rows(int64_t M, int64_t layout) { return layout == 1 ? fso_align(M, 128) : fso_align(M, 4); }

torch::Tensor contract_quantize(torch::Tensor x, int64_t layout, torch::Tensor q, torch::Tensor s) {
  const int64_t M = x.size(0), K = x.size(1), mp = padded_rows(M, layout);
  const int64_t n = mp * (K / 128);
  contract_kernel<<<(n + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<uint8_t*>(q.data_ptr()), s.data_ptr(),
      M, mp, K, layout);
  return q;
}

torch::Tensor contract_quantize_host(torch::Tensor x, int64_t layout, torch::Tensor q, torch::Tensor s) {
  const int64_t M = x.size(0), K = x.size(1), mp = padded_rows(M, layout);
  for (int64_t m = 0; m < mp; ++m)
    for (int64_t g = 0; g < K / 128; ++g)
      quantize_entry(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<uint8_t*>(q.data_ptr()),
                     s.data_ptr(), M, K, layout, m, g);
  return q;
}

int64_t contract_version() { return kFsoActivationContractVersion; }
"""


_FILL = -1515870811  # 0xA5A5A5A5: the scale buffers' fill before the header kernel writes them


def _build_contract_ext(fast_math: bool):
    from torch.utils.cpp_extension import load_inline
    major, minor = _arch()
    arch = f"{major}.{minor}"
    name = f"fso_contract_v1_sm{major}{minor}{'_fm' if fast_math else ''}"
    build_dir = os.path.join(tempfile.gettempdir(), f"fso_test_dense_v1_{os.getuid()}", name)
    os.makedirs(build_dir, exist_ok=True)
    saved = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load_inline(name=name, cpp_sources=_CONTRACT_CPP, cuda_sources=_CONTRACT_CU,
                           functions=["contract_quantize", "contract_quantize_host", "contract_version"],
                           extra_include_paths=[D.include_dir()],
                           extra_cuda_cflags=["-O3"] + (["--use_fast_math"] if fast_math else []),
                           build_directory=build_dir, verbose=False)
    finally:
        if saved is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = saved


def _contract_input(M: int, K: int, seed: int) -> torch.Tensor:
    """Random rows, plus rows whose amax is below the floor (1e-12 scale), an all-zero row and a large row."""
    x = make_x(M, K, seed)
    if M >= 7:
        x[1] = (x[1].float() * 1e-11).bfloat16()
        x[2].zero_()
        x[4] = (x[4].float() * 1e4).bfloat16()
    elif M >= 1:
        x[0, :K // 2] = (x[0, :K // 2].float() * 1e-11).bfloat16()
    return x


def case_d(major: int) -> None:
    layout = {9: 0, 10: 1, 12: 2}[major]
    exts = {}
    for fm in (False, True):
        try:
            exts[fm] = _build_contract_ext(fm)
        except Exception as e:  # noqa: BLE001 - a build failure is a test failure, reported with its reason
            _check(False, f"d: building the contract extension (fast_math={fm}) failed: {type(e).__name__}: {e}")
            print(f"  d  build (fast_math={fm}) FAILED: {type(e).__name__}: {str(e)[-400:]}")
            return
    ver_ok = exts[False].contract_version() == 1 == D.activation_contract()["version"]
    _check(ver_ok, "d: kFsoActivationContractVersion != activation_contract()['version'] == 1")
    print(f"  d  header kFsoActivationContractVersion == activation_contract()['version'] == 1  {_ok(ver_ok)}")
    for K in (1280, 5120):
        for M in (1, 7, 78, 129):
            x = _contract_input(M, K, seed=M * 31 + K)
            xq, sx = D.quantize_activation(x)
            outs = {}
            for fm, ext in exts.items():
                q = torch.full((M, K), 0x7F, dtype=torch.uint8, device="cuda")  # 0x7F: an e4m3 NaN, never produced
                s = torch.empty_strided(sx.shape, sx.stride(), dtype=sx.dtype, device="cuda")
                s.view(torch.int32).fill_(_FILL)
                ext.contract_quantize(x, layout, q, s)
                torch.cuda.synchronize()
                outs[fm] = (q, s)
            qh = torch.full((M, K), 0x7F, dtype=torch.uint8)
            sh = torch.empty_strided(sx.shape, sx.stride(), dtype=sx.dtype)
            sh.view(torch.int32).fill_(_FILL)
            exts[False].contract_quantize_host(x.cpu(), layout, qh, sh)
            q, s = outs[False]
            codes_ok = torch.equal(q, xq.view(torch.uint8))
            scale_ok = bytes_equal(s, sx)
            fm_ok = torch.equal(outs[True][0], q) and bytes_equal(outs[True][1], s)
            host_ok = torch.equal(qh, q.cpu()) and torch.equal(sh.view(torch.int32), s.cpu().view(torch.int32))
            ok = codes_ok and scale_ok and fm_ok and host_ok
            _check(ok, f"d K={K} M={M}: header kernel vs quantize_activation: codes {codes_ok}, scale {scale_ok}, "
                       f"fast-math build {fm_ok}, host path {host_ok}")
            print(f"  d  K={K:5d} M={M:4d}: header kernel == quantize_activation (codes; scale words incl. padded "
                  f"rows); --use_fast_math build == default build; host path == device path  {_ok(ok)}")


# --- (e) CUDA graph -------------------------------------------------------------------------------------------------

def _graph_case(label, fn) -> None:
    ref = fn().clone()  # one eager call on the default stream
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = fn()
    out.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    ok = bool(torch.isfinite(out).all()) and torch.equal(out, ref)
    _check(ok, f"e {label}: replay != eager")
    print(f"  e  {label}: one eager call, capture on a non-default stream, replay into a NaN-filled output == eager  "
          f"{_ok(ok)}")


def case_e(major: int) -> None:
    for N, K in ((1280, 1280), (5120, 1280), (2048, 4096)):
        dw = prepared(major, N, K, seed=51 + N)
        for M in (78, 1079, 7):
            x = make_x(M, K, seed=52 + M)
            xq, sx = D.quantize_activation(x)
            _graph_case(f"2b N={N:5d} K={K:5d} M={M:5d}", lambda: D.linear(xq, dw, x_scale=sx))

            def both():
                q, s = D.quantize_activation(x)
                return D.linear(q, dw, x_scale=s)

            _graph_case(f"quantize_activation + 2b N={N:5d} K={K:5d} M={M:5d}", both)


# --- (f) fakes, the old call, the contract description --------------------------------------------------------------

def _no_sync(label: str, fn) -> None:
    """fn() under torch.cuda.set_sync_debug_mode("error"): nothing in it may synchronize the device. A
    NotImplementedError of a refused call is the expected outcome, not a failure."""
    import warnings
    torch.cuda.synchronize()
    why = ""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # torch labels the mode a prototype
            torch.cuda.set_sync_debug_mode("error")
        try:
            fn()
        except NotImplementedError:
            pass
        ok = True
    except RuntimeError as e:
        ok, why = False, str(e).splitlines()[0]
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    _check(ok, f"f {label}: synchronized the device: {why}")
    print(f"  f  {label}: no device synchronization under set_sync_debug_mode('error')  {_ok(ok)}")


def case_f(major: int) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode
    N, K, M = 2048, 1280, 7
    dw = prepared(major, N, K, seed=61)
    x = make_x(M, K, seed=62)
    xq, sx = D.quantize_activation(x)
    ops = torch.ops.fish_scales_ops
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fx = mode.from_tensor(x)
        fxq, fsx = ops.dense_quantize_activation(fx)
        fy = ops.dense_linear(mode.from_tensor(xq), mode.from_tensor(dw.weight), mode.from_tensor(dw.scale), dw.kind,
                              None, mode.from_tensor(sx))
        fq2, fs2 = ops.dense_linear_quantized(fx, mode.from_tensor(dw.weight), mode.from_tensor(dw.scale), dw.kind)
    meta = lambda t: (t.dtype, tuple(t.shape), t.stride())  # noqa: E731
    qa_ok = meta(fxq) == meta(xq) and meta(fsx) == meta(sx)
    y = D.linear(xq, dw, x_scale=sx)
    ly_ok = meta(fy) == meta(y)
    dt, shp, st = expected_scale_meta(major, M, N)
    lq_ok = meta(fq2) == (torch.float8_e4m3fn, (M, N), (N, 1)) and fs2.dtype == dt and tuple(fs2.shape) == shp \
        and (fs2.stride() == st or (N // 128 == 1 and major != 10))
    _check(qa_ok and ly_ok and lq_ok, f"f: fakes: quantize_activation {qa_ok}, dense_linear {ly_ok}, "
                                      f"dense_linear_quantized {lq_ok} ({meta(fq2)}, {meta(fs2)})")
    print(f"  f  fakes under FakeTensorMode: dense_quantize_activation {meta(fsx)} == real {_ok(qa_ok)}; "
          f"dense_linear (2b) == real {_ok(ly_ok)}; dense_linear_quantized {meta(fq2)}, {meta(fs2)} == the "
          f"contract of an [M, N] activation {_ok(lq_ok)}")
    bias = torch.randn(N, device="cuda").bfloat16()
    D.quantize_activation(x)
    D.linear(xq, dw, x_scale=sx)
    _no_sync("quantize_activation(x)", lambda: D.quantize_activation(x))
    _no_sync("2b linear(xq, w, x_scale=sx)", lambda: D.linear(xq, dw, x_scale=sx))
    _no_sync("quantize_activation + 2b linear", lambda: D.linear(*D.quantize_activation(x)[:1], dw,
                                                                 x_scale=sx))
    _no_sync("linear(x, w, bias=b) (served or refused by this route)", lambda: D.linear(x, dw, bias=bias))
    _no_sync("linear_quantized(xq, w, x_scale=sx) (served or refused by this route)",
             lambda: D.linear_quantized(xq, dw, x_scale=sx))
    old = ops.dense_linear(x, dw.weight, dw.scale, dw.kind)
    old_ok = torch.equal(old, D.linear(x, dw))
    _check(old_ok, "f: the old four-argument dense_linear call differs from linear(x, w)")
    print(f"  f  old four-argument dense_linear(x, w, s, kind) == linear(x, w)  {_ok(old_ok)}")
    c = D.activation_contract()
    hdr = os.path.join(D.include_dir(), "fso", "activation_contract.h")
    want_fmt = "bsfp8_1x128" if major == 9 else "mxfp8_1x32"
    c_ok = (c["version"] == 1 and c["format"] == want_fmt and c["arch"] == _arch_label() and os.path.isfile(hdr)
            and c["header"] == "fso/activation_contract.h" and c["include_dir"] == D.include_dir())
    _check(c_ok, f"f: activation_contract() / include_dir() wrong: {c}")
    print(f"  f  activation_contract(): version 1, format {c['format']}, x_scale {c['x_scale_dtype']} "
          f"{c['x_scale_shape']}; include_dir() holds {hdr}  {_ok(c_ok)}")
    e0 = D.quantize_activation(x[:0])
    e_ok = tuple(e0[0].shape) == (0, K) and meta_ok(e0[1], major, 0, K)
    y0 = D.linear(e0[0], dw, x_scale=e0[1])
    e_ok = e_ok and tuple(y0.shape) == (0, N)
    _check(e_ok, f"f: empty M: quantize_activation {tuple(e0[0].shape)} {tuple(e0[1].shape)}, linear {tuple(y0.shape)}")
    print(f"  f  empty M: quantize_activation -> {tuple(e0[0].shape)}, {tuple(e0[1].shape)}; 2b linear -> "
          f"{tuple(y0.shape)}  {_ok(e_ok)}")
    x3 = make_x(6, K, seed=63).view(2, 3, K)
    q3, s3 = D.quantize_activation(x3)
    q2, s2 = D.quantize_activation(x3.reshape(6, K))
    nd_ok = tuple(q3.shape) == (6, K) and bytes_equal(q3, q2) and bytes_equal(s3, s2)
    _check(nd_ok, "f: quantize_activation of [2, 3, K] is not the [6, K] result")
    print(f"  f  quantize_activation([2, 3, K]) == quantize_activation([6, K]) (flattened to [M, K])  {_ok(nd_ok)}")


# --- (g) .. (k) the sm_90 dense epilogue: section-10 exactness, capture, streams, PDL, compile --------------------

# The routes of gemm_dispatch_sm90 and how a call reaches each: M < 32 (or M = 32 with N < 16384) runs the swap-AB
# fp8_gemm_kernel_swapAB, and so do 32 < M < 64 and 64 < M <= 96 where the main path's tile is empty enough and the
# swap-AB launch fits one wave, and 96 < M <= 100 and 128 < M <= 160 on the shapes of the H4 rule (one or two waves;
# deep_gemm::jit::use_dense_swap_ab, mirrored by _sm90_swap_ab below), and in FH1's two bands, 144 < M <= 216 and
# 256 < M <= 288 (not mirrored: their cells are listed in _SM90_EPI_FH1); in the default mode the tiles of
# deep_gemm::jit::use_dense_v2_override come next (_SM90_EPI_FH1); otherwise
# FSO_SM90_DENSE_DG2 decides between DeepGEMM v2's sm90_fp8_gemm_1d2d_impl ("all": every call; default: its tile
# classes) and fp8_gemm_kernel ("0": every call). The variable is read once per process, so the "0" and "all" passes
# run in child processes of this script (--sm90-epilogue-mode).
_SM90_EPI_SHAPES = ((1280, 1280), (5120, 1280), (1280, 5120), (1280, 7680))
_SM90_EPI_SWAP_MS = (1, 7, 32)  # M = 32 (N < 16384) is the swap-AB route's full 32-token tile
# Above M = 32 (issue-12 subtask H2, /data/bench-runs/fso_issue12_20261028/runs/H2), default mode, (N, K, M): the
# swap-AB route with 64-row weight tiles and a 16-token tile (2048 x 1280 at M = 40 and at M = 46, the last M with 18
# empty rows), with 128-row weight tiles and a 24-token tile (5120 x 2048 at M = 36) and a 40-token tile (9728 x 2560
# at M = 33), and the neighbours the rule keeps on the main path: 2048 x 1280 at M = 47 (17 empty rows), 5120 x 2048
# at M = 37 (27 empty rows), 1280 x 1280 at M = 40 (60 swap-AB tiles, fewer than half the SMs), 5120 x 1280 at M = 36
# (K < 2048). A swap-AB cell's kernel name must show the tile _sm90_swap_ab_tile names.
# At 64 < M <= 96 (issue-12 subtask H3, /data/bench-runs/fso_issue12_20261028/runs/H3), default mode: the swap-AB route
# with 64-row weight tiles and the smallest one-wave token tile (2048 x 4096 and 2048 x 1280 at M = 80 / 78: 24 tokens,
# which takes both off rule 5's v2 kernel; 1280 x 1280 at M = 72 and 1280 x 7680 at M = 96: 16; 2560 x 4096 at
# M = 96: 32), with 128-row weight tiles and two token tiles (5120 x 2048 at M = 72: 40, from rule 5's v2 kernel; at
# M = 96: 48, from fp8_gemm_kernel) and one (9728 x 2560 at M = 72: 72), and the neighbours it keeps on the main path,
# with the kernel they run (_SM90_EPI_H3_MAIN): 5120 x 2048 at M = 76 (52 empty rows: rule 5's v2 kernel), 9728 x 2560
# at M = 84 (one token tile below the rule-5 band), 2048 x 4096 at M = 100 (28 empty rows), 5120 x 1280 at M = 72
# (K < 2048), 19456 x 2560 at M = 72 (no one-wave swap-AB tile) and 2048 x 4096 at M = 64 (no empty rows).
_SM90_EPI_REROUTE_CELLS = ((2048, 1280, 40), (2048, 1280, 46), (5120, 2048, 36), (9728, 2560, 33),
                           (2048, 1280, 47), (5120, 2048, 37), (1280, 1280, 40), (5120, 1280, 36),
                           (2048, 4096, 80), (2048, 1280, 78), (1280, 7680, 96), (2560, 4096, 96),
                           (5120, 2048, 72), (5120, 2048, 96), (9728, 2560, 72), (1280, 1280, 72),
                           (5120, 2048, 76), (9728, 2560, 84), (2048, 4096, 100), (5120, 1280, 72),
                           (19456, 2560, 72), (2048, 4096, 64)) + (
    # Above M = 96 (issue-12 subtask H4, /data/bench-runs/fso_issue12_20261028/runs/H4), default mode: the swap-AB
    # route with 64-row weight tiles and K >= 4096 at 96 < M <= 100 (2048 x 4096 at M = 97: 32 tokens; 2560 x 9728 at
    # M = 100: 40) and at 128 < M <= 144 (2560 x 4096 at M = 136: 48; 1280 x 7680 at M = 144: 24), with 128-row weight
    # tiles at 128 < M <= 144 with fewer CTAs than the main path (9728 x 2560 at M = 144: 228 tiles of 128 x 48, two
    # waves) and up to M = 160 with at most three quarters of them (6144 x 2560 at M = 160: 96 of 128 x 80, one wave;
    # 12288 x 2048 at M = 152: 192 of 128 x 80, two waves), and the neighbours the rule keeps on the main path, with the
    # kernel they run (_SM90_EPI_H4_MAIN): 2560 x 4096 at M = 104 and 152, 1280 x 5120 at M = 100 (N < 2048),
    # 2560 x 2048 at M = 136 (K < 4096), 5120 x 2048 at M = 136 (as many CTAs as the main path), 9728 x 2560 at
    # M = 160, 6144 x 2560 at M = 176 and 19456 x 2560 at M = 144 (more CTAs than the main path).
    (2048, 4096, 97), (2560, 9728, 100), (2560, 4096, 136), (1280, 7680, 144), (9728, 2560, 144), (6144, 2560, 160),
    (12288, 2048, 152), (2560, 4096, 104), (2560, 4096, 152), (1280, 5120, 100), (2560, 2048, 136),
    (5120, 2048, 136), (9728, 2560, 160), (6144, 2560, 176), (19456, 2560, 144))
_SM90_EPI_H3_MAIN = {(5120, 2048, 76): "dg2", (9728, 2560, 84): "v1", (2048, 4096, 100): "v1",
                     (5120, 1280, 72): "dg2", (19456, 2560, 72): "dg2", (2048, 4096, 64): "v1"}
_SM90_EPI_H4_MAIN = {(2560, 4096, 104): "v1", (2560, 4096, 152): "dg2", (1280, 5120, 100): "v1",
                     (2560, 2048, 136): "dg2", (5120, 2048, 136): "dg2", (9728, 2560, 160): "dg2",
                     (6144, 2560, 176): "dg2", (19456, 2560, 144): "dg2"}
# The main path's CTA count (deep_gemm::jit::dense_main_path_tiles on 132 SMs: DeepGEMM v2's configuration where the
# dispatcher takes it, else fp8_gemm_kernel's tile) of the cells whose swap-AB decision above M = 96 compares the
# 128-row-weight-tile launch with it, as the dispatcher computes it (runs/H4/tmp/rulecheck).
_SM90_H4_MAIN_TILES = {(5120, 1280, 129, 132): 120, (6144, 2560, 160, 132): 129, (9728, 2560, 144, 132): 261,
                       (12288, 2048, 152, 132): 258, (5120, 2048, 136, 132): 120, (9728, 2560, 160, 132): 261,
                       (19456, 2560, 144, 132): 244}
_SM90_EPI_MAIN_MS = (78, 129, 1079)


def _sm90_sms() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def _sm90_swap_ab_tile(M: int, N: int, sms: int):
    """(weight rows, token tile) of the swap-AB route at 32 < M < 64 and 64 < M <= 96, or None when no tile fits one
    wave: the mirror of deep_gemm::jit::dense_swap_ab_token_tile on get_dense_gemm_config's swap-AB weight tile (rule 2:
    64 rows when ceil(N / 128) * 4 <= SMs, else 128); above M = 96 the mirror of dense_swap_ab_token_tile_above_96 (the
    fewest waves, then the fullest last wave, then the smaller tile)."""
    cdiv = lambda a, b: -(-a // b)  # noqa: E731
    rows = 64 if (N <= 64 or cdiv(N, 128) * 4 <= sms) else 128
    tiles = cdiv(N, rows)
    if M > 96:
        best = None
        for t in range(16, 129, 8):
            n = tiles * cdiv(M, t)
            waves = cdiv(n, sms)
            key = (waves, -(n - (waves - 1) * sms))
            if best is None or key < best[0]:
                best = (key, t)
        return rows, best[1]
    if rows == 64:
        if M < 64:
            return (64, 16) if tiles * cdiv(M, 16) <= sms else None
        return next(((64, t) for t in range(16, 129, 8) if tiles * cdiv(M, t) <= sms), None)
    if 2 * tiles <= sms:
        return 128, cdiv(cdiv(M, 2), 8) * 8
    if tiles <= sms:
        return 128, cdiv(M, 8) * 8
    return None


def _sm90_swap_ab(M: int, N: int, K: int, sms: int) -> bool:
    """Whether gemm_dispatch_sm90 takes the swap-AB route (mirror of deep_gemm::jit::use_dense_swap_ab, without FH1's
    bands at 144 < M <= 216 and 256 < M <= 288, whose cells _SM90_EPI_FH1 lists with their route)."""
    if M < 32 or (M == 32 and N < 16384):
        return True
    if M > 96:  # mirror of use_dense_swap_ab_above_96
        if not (M <= 100 or 128 < M <= 160):
            return False
        rows, tok = _sm90_swap_ab_tile(M, N, sms)
        tiles = -(-N // rows) * (-(-M // tok))
        if 2 * tiles < sms:
            return False
        if rows == 64:
            return K >= 4096 and (N >= 2048 if M <= 100 else M <= 144)
        if M <= 100:
            return False
        main = _SM90_H4_MAIN_TILES[(N, K, M, sms)]  # a 128-row cell of this test without its main-path CTA count
        return (M <= 144 and tiles < main) or 4 * tiles <= 3 * main
    if M <= 32 or M == 64:
        return False
    tile = _sm90_swap_ab_tile(M, N, sms)
    if tile is None:
        return False
    rows, tok = tile
    weight_tiles = -(-N // rows)
    if 2 * weight_tiles * (-(-M // tok)) < sms:  # fewer than half the SMs would get a tile
        return False
    if M < 64:
        return 64 - M >= 18 if rows == 64 else (64 - M >= 28 and K >= 2048)
    empty = 128 - M  # 64 < M <= 96: the main path's last M block leaves 128 - M >= 32 rows empty
    if rows == 64:
        return True
    if K < 2048:
        return False
    return empty >= 56 or (empty < 48 and 2 * weight_tiles <= sms)


def _swap_tile_re(N: int, K: int, bm: int, bn: int) -> str:
    """A regular expression on the swap-AB kernel's name for the tile: <SHAPE_M = N, K, BLOCK_M, BLOCK_N, ...>."""
    return rf"fp8_gemm_kernel_swapAB<{N}u, {K}u, {bm}u, {bn}u, "
# The two-wave DG2 tile (BLOCK_M = 256: the epilogue's second wave, its residual registers and prefetch rows) is
# picked by none of the cells above: (N, K, M, mode). Default mode: the 256 x 128 tile with multicast on A (c1x2);
# FSO_SM90_DENSE_DG2=all: the one without (c1x1). The kernel name is checked (_dg2_tile_re).
_SM90_EPI_TWO_WAVE = ((1280, 1280, 16384, "default"), (1280, 1280, 3072, "all"))
# Rule 5 of the sm_90 picker (jit_utils.cuh use_dg2_dense_partial_block; H1 of the issue-12 round,
# /data/bench-runs/fso_issue12_20261028/runs/H1): below M = 1024 a cell whose fp8_gemm_kernel tile leaves at least 48
# rows of its last M block empty runs the v2 kernel, at DeepGEMM's own configuration when that is a single-CTA one or
# at fp8_gemm_kernel's tile when DeepGEMM picks a two-CTA cluster (5120 x 2048 at M = 288 / 320: 128 x 128 with 5
# stages, a configuration no DeepGEMM pick produces). The 17 cells of the issue-12 grid the rule re-routes,
# (N, K, M, block_m, block_n, stages), default mode only: every epilogue variant must run on the v2 kernel at that
# configuration (the kernel name) and give section 10's bytes. 5120 x 1280 at M = 78 is also a cell of the grid above.
# Two more cells of the grid, 2048 x 1280 at M = 78 and 2048 x 4096 at M = 80, take the swap-AB route since H3
# (_SM90_EPI_REROUTE_CELLS).
_SM90_EPI_RULE5 = (
    (1280, 1280, 533, 64, 96, 10), (5120, 1280, 78, 64, 80, 11), (1280, 5120, 533, 64, 96, 10),
    (1280, 7680, 533, 64, 96, 10), (2048, 1280, 533, 128, 80, 7),
    (5120, 2048, 80, 64, 80, 11), (5120, 2048, 288, 128, 128, 5), (5120, 2048, 320, 128, 128, 5),
    (5120, 2048, 416, 128, 160, 5), (5120, 2048, 448, 128, 160, 5),
    (2048, 4096, 288, 64, 80, 11), (2048, 4096, 320, 64, 80, 11),
    (2048, 4096, 416, 128, 64, 8), (2048, 4096, 448, 128, 64, 8), (2048, 4096, 576, 128, 80, 7))


# FH1 of the issue-12 round (/data/bench-runs/fso_issue12_20261028/runs/FH1), default mode, (N, K, M) -> (route,
# block_m, block_n, cluster), as the dispatcher computes them (runs/FH1/tmp/finalinc): DeepGEMM v2's kernel in one wave
# where the route needs two or more at M <= 256 (deep_gemm::jit::use_dense_v2_override: the fewest rows, then the
# narrowest tile, up to M = 128 a width with one main loop; 128-row tiles at 128 < M <= 192 paired over M, "c2x1"), its
# 128 x 160 / 192 pair over N ("c1x2") in a quarter fewer waves for the wide outputs at 256 < M <= 672 (19456 x 2560 at
# M = 64 keeps fp8_gemm_kernel: its one-wave tiles are 160 or 192 wide), its 64 x 128 pair over M for the narrow
# outputs at 256 < M <= 384 with K >= 4096; fp8_gemm_kernel_swapAB with three token tiles for the narrow outputs at
# 144 < M <= 216 (64-row weight tiles, K >= 8192 or against fp8_gemm_kernel's tile) and the wide outputs at
# 256 < M <= 288 (128-row weight tiles), and at 8 <= M <= 32 with 64-row weight tiles and one token tile where they fit
# one wave; and neighbours each rule keeps (cluster None: the kernel class only). Two cells of _SM90_EPI_REROUTE_CELLS
# move here from swap-AB (9728 x 2560 at M = 144, 12288 x 2048 at M = 152: one wave of the v2 pair instead of two of
# swap-AB); _sm90_swap_ab does not mirror FH1's bands, so every cell in them is listed here.
_SM90_EPI_FH1 = {
    (18432, 2560, 8): ("dg2", 16, 144, "c1x1"), (18432, 2560, 128): ("dg2", 128, 144, "c1x1"),
    (19456, 2560, 64): ("v1", None, None, None),
    (9728, 2560, 144): ("dg2", 128, 160, "c2x1"), (12288, 2048, 152): ("dg2", 128, 192, "c2x1"),
    (9728, 2560, 160): ("dg2", 128, 160, "c2x1"), (10240, 2560, 256): ("dg2", 128, 160, "c1x1"),
    (2560, 9216, 152): ("swapab", 64, 56, None), (2560, 4096, 216): ("swapab", 64, 72, None),
    (10240, 2560, 272): ("swapab", 128, 96, None),
    (18432, 2560, 288): ("dg2", 128, 160, "c1x2"), (18432, 2560, 400): ("dg2", 128, 192, "c1x2"),
    (2560, 4096, 288): ("dg2", 64, 128, "c2x1"), (2560, 9216, 344): ("dg2", 64, 128, "c2x1"),
    (2560, 4096, 200): ("dg2", 64, 80, "c1x1"), (10240, 2560, 296): ("v1", None, None, None),
    (2560, 4096, 328): ("dg2", 64, 128, "c1x1"),
    # swap-AB at 8 <= M <= 32 with 64-row weight tiles and one token tile up to N = 8448 (get_dense_swap_ab_config);
    # M < 8 keeps rule 2's 128-row tile
    (5120, 2048, 16): ("swapab", 64, 16, None), (6144, 2560, 32): ("swapab", 64, 32, None),
    (6144, 2560, 4): ("swapab", 128, 16, None),
}


def _dg2_cluster_re(bm: int, bn: int, cluster: str) -> str:
    """A regular expression on the v2 kernel's name for the tile and its cluster: after the stage count come
    <128, threads, cluster size, multicast on A, ...>; "c2x1" pairs two M blocks (multicast on B), "c1x2" two N
    blocks (multicast on A), "c1x1" is a single CTA."""
    size, on_a = {"c1x1": (1, "false"), "c2x1": (2, "false"), "c1x2": (2, "true")}[cluster]
    return rf"0u, 1u, {bm}u, {bn}u, 128u, 128u, 128u, \d+u, \d+u, 128u, \d+u, {size}u, {on_a}, "


def _dg2_tile_re(bm: int, bn: int, stages=None) -> str:
    """A regular expression on the v2 kernel's name for the tile (and stage count): the template arguments run
    <major, 0, N, 0, 1, BLOCK_M, BLOCK_N, 128, 128, 128, swizzle, stages, 128, threads, ...> (compiler.cuh
    generateKernelDg2); the leading "0u, 1u, " keeps a stage count from matching as a BLOCK_M."""
    return rf"0u, 1u, {bm}u, {bn}u, 128u, 128u, 128u, \d+u, {stages if stages is not None else r'\d+'}u, "


# (label, bias, activation, residual)
_SM90_EPI_VARIANTS = (("bias", True, None, False), ("bias+residual", True, None, True),
                      ("bias+gelu", True, "gelu_erf", False), ("gelu", False, "gelu_erf", False),
                      ("residual", False, None, True), ("bias+gelu+residual", True, "gelu_erf", True))

_GELU_CPP = r"""
#include <torch/extension.h>
torch::Tensor gelu_ref(torch::Tensor v);
torch::Tensor gelu_plain(torch::Tensor v);
"""

_GELU_CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// The reference of the contract: 0.5f * v * (1.f + erff(v * 0.70710678118654752440f)) in fp32 with every operation
// rounded to nearest on its own (the _rn form, which no compiler setting contracts), written out here and not taken
// from the library; and the same formula written plainly, which equals it only where the compiler does not contract
// (built with -fmad=false for the cross-check).
__global__ void gelu_ref_kernel(const float* v, float* y, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float x = v[i];
    y[i] = __fmul_rn(__fmul_rn(0.5f, x), __fadd_rn(1.f, erff(__fmul_rn(x, 0.70710678118654752440f))));
  }
}

__global__ void gelu_plain_kernel(const float* v, float* y, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float x = v[i];
    y[i] = 0.5f * x * (1.f + erff(x * 0.70710678118654752440f));
  }
}

static torch::Tensor launch(void (*k)(const float*, float*, int64_t), torch::Tensor v) {
  TORCH_CHECK(v.is_cuda() && v.dtype() == torch::kFloat32 && v.is_contiguous());
  auto y = torch::empty_like(v);
  const int64_t n = v.numel();
  k<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(v.data_ptr<float>(), y.data_ptr<float>(), n);
  return y;
}

torch::Tensor gelu_ref(torch::Tensor v) { return launch(gelu_ref_kernel, v); }
torch::Tensor gelu_plain(torch::Tensor v) { return launch(gelu_plain_kernel, v); }
"""

_GELU_NVRTC_SRC = r"""
// _GELU_CU's two kernels, for the library's own sm_90 JIT compiler.
extern "C" __global__ void gelu_ref_kernel(const float* v, float* y, long long n) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float x = v[i];
    y[i] = __fmul_rn(__fmul_rn(0.5f, x), __fadd_rn(1.f, erff(__fmul_rn(x, 0.70710678118654752440f))));
  }
}

extern "C" __global__ void gelu_plain_kernel(const float* v, float* y, long long n) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float x = v[i];
    y[i] = 0.5f * x * (1.f + erff(x * 0.70710678118654752440f));
  }
}
"""


class _NvrtcGelu:
    """sm_90: gelu_ref and gelu_plain of _GELU_CU built by the compiler of the library's sm_90 kernels, the NVRTC
    bundled with fish_scales_ops, and launched through the driver API on torch's current stream. The library's
    kernels take erff from that NVRTC's libdevice; load_inline would take it from CUDA_HOME's nvcc, and CUDA 13.2's
    erff differs from 13.0's in the last bit of some values, which the exact comparison would report as a mismatch."""

    def __init__(self, nofma: bool):
        import ctypes
        d = os.path.join(os.path.dirname(fso.__file__), "_nvrtc")
        ctypes.CDLL(os.path.join(d, "libnvrtc-builtins.so.13.0"), mode=ctypes.RTLD_GLOBAL)
        nv = ctypes.CDLL(os.path.join(d, "libnvrtc.so.13"))
        major, minor = _arch()
        opts = [f"--gpu-architecture=sm_{major}{minor}a"] + (["--fmad=false"] if nofma else [])
        prog = ctypes.c_void_p()
        rc = nv.nvrtcCreateProgram(ctypes.byref(prog), _GELU_NVRTC_SRC.encode(), b"fso_gelu_ref.cu", 0, None, None)
        assert rc == 0, f"nvrtcCreateProgram: {rc}"
        rc = nv.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts]))
        if rc != 0:
            size = ctypes.c_size_t()
            nv.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            nv.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(f"NVRTC could not build the reference GELU ({rc}): {log.value.decode()[-1000:]}")
        size = ctypes.c_size_t()
        nv.nvrtcGetCUBINSize(prog, ctypes.byref(size))
        cubin = ctypes.create_string_buffer(size.value)
        nv.nvrtcGetCUBIN(prog, cubin)
        nv.nvrtcDestroyProgram(ctypes.byref(prog))
        torch.cuda.synchronize()  # torch's primary context is current on this thread
        self._ct, self._cu = ctypes, ctypes.CDLL("libcuda.so.1")
        self._mod = ctypes.c_void_p()
        rc = self._cu.cuModuleLoadData(ctypes.byref(self._mod), cubin)
        assert rc == 0, f"cuModuleLoadData: {rc}"
        self._fn = {}
        for name in ("gelu_ref_kernel", "gelu_plain_kernel"):
            fn = ctypes.c_void_p()
            rc = self._cu.cuModuleGetFunction(ctypes.byref(fn), self._mod, name.encode())
            assert rc == 0, f"cuModuleGetFunction {name}: {rc}"
            self._fn[name] = fn

    def _launch(self, name: str, v: torch.Tensor) -> torch.Tensor:
        assert v.is_cuda and v.dtype == torch.float32 and v.is_contiguous()
        ct = self._ct
        y = torch.empty_like(v)
        n = v.numel()
        if n:
            pv, py, pn = ct.c_void_p(v.data_ptr()), ct.c_void_p(y.data_ptr()), ct.c_longlong(n)
            args = (ct.c_void_p * 3)(ct.addressof(pv), ct.addressof(py), ct.addressof(pn))
            rc = self._cu.cuLaunchKernel(self._fn[name], ct.c_uint((n + 255) // 256), ct.c_uint(1), ct.c_uint(1),
                                         ct.c_uint(256), ct.c_uint(1), ct.c_uint(1), ct.c_uint(0),
                                         ct.c_void_p(torch.cuda.current_stream().cuda_stream), args, None)
            assert rc == 0, f"cuLaunchKernel {name}: {rc}"
        return y

    def gelu_ref(self, v: torch.Tensor) -> torch.Tensor:
        return self._launch("gelu_ref_kernel", v)

    def gelu_plain(self, v: torch.Tensor) -> torch.Tensor:
        return self._launch("gelu_plain_kernel", v)


_gelu_exts = {}


def _gelu_ext(nofma: bool):
    """The reference GELU, built with the default contraction or with -fmad=false: on sm_90 by the library's own
    JIT compiler (_NvrtcGelu), elsewhere by load_inline with this toolkit's nvcc."""
    if nofma not in _gelu_exts and _arch()[0] == 9:
        _gelu_exts[nofma] = _NvrtcGelu(nofma)
    if nofma not in _gelu_exts:
        from torch.utils.cpp_extension import load_inline
        major, minor = _arch()
        name = f"fso_gelu_ref_sm{major}{minor}{'_nofma' if nofma else ''}"
        build_dir = os.path.join(tempfile.gettempdir(), f"fso_test_dense_v1_{os.getuid()}", name)
        os.makedirs(build_dir, exist_ok=True)
        saved = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        try:
            _gelu_exts[nofma] = load_inline(
                name=name, cpp_sources=_GELU_CPP, cuda_sources=_GELU_CU, functions=["gelu_ref", "gelu_plain"],
                extra_cuda_cflags=["-O3"] + (["-fmad=false"] if nofma else []), build_directory=build_dir,
                verbose=False)
        finally:
            if saved is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = saved
    return _gelu_exts[nofma]


def _gelu_reference_checked(v32: torch.Tensor) -> torch.Tensor:
    """The _rn reference of v32 (default build), after checking that it equals the _rn kernel built with -fmad=false
    and the plain formula built with -fmad=false, bit for bit (the _rn form is the plain formula without
    contraction)."""
    g = _gelu_ext(False).gelu_ref(v32)
    for what, other in (("_rn, -fmad=false", _gelu_ext(True).gelu_ref(v32)),
                        ("plain, -fmad=false", _gelu_ext(True).gelu_plain(v32))):
        same = torch.equal(g.view(torch.int32), other.view(torch.int32))
        _check(same, f"g: the _rn reference GELU != the {what} build on {v32.numel()} values")
        if not same:
            print(f"  g  reference GELU: _rn (default build) != {what}  FAIL")
    return g


def _kernels_of(fn):
    """(result of fn(), names of the CUDA kernels it launched, in order)."""
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
             and "memset" not in e.name.lower() and "memcpy" not in e.name.lower()]
    return out, names


# Clusters of 2, 3 and 4 CTAs the occupancy calculator places at once, at one CTA per SM (the split-K kernel requests
# at least 114 KB of shared memory) and at the kernel's own size where two CTAs fit an SM, by SM count: the H200's 132
# SMs (FH5 runs/FH5/scripts/occ_probe.py). Elsewhere the route mirror is not exact.
_SM90_SPLITK_CLUSTERS = {132: {1: {2: 66, 3: 39, 4: 30}, 2: {2: 132, 3: 79, 4: 62}}}


def _sm90_splitk_pref(N: int, K: int, sms: int) -> int:
    """The split-K route's preferred slice count (mirror of deep_gemm::jit::dense_splitk_preferred_slices)."""
    if not N or not K or K % 128:
        return 0
    tiles, kb = -(-N // 64), K // 128
    for s in (4, 3):
        if 4 * s * tiles <= 3 * sms and 2 * s <= kb:
            return s
    if K > 4096 and 3 * tiles <= sms and 6 <= kb:
        return 3
    if 2 * tiles <= sms and 4 <= kb:
        return 2
    return 0


def _sm90_splitk_max_m(s: int, K: int) -> int:
    """The largest M the route takes at s slices (mirror of deep_gemm::jit::dense_splitk_max_m)."""
    if K >= 8192:
        return 64
    if K >= 4096:
        return 56
    return 40 if (K >= 2048 and s >= 3) else 24


def _sm90_splitk(M: int, N: int, K: int, sms: int) -> int:
    """The slice count of the in-cluster split-K route (fp8_gemm_kernel_swapAB_splitk) for an M x N x K dense GEMM on
    `sms` SMs, 0 when gemm_dispatch_sm90 does not take the route: the preferred count at one CTA per SM, or at the
    kernel's own size with a token tile of at most 24 rows, lowered one step while one cluster per 64-row weight tile
    does not fit at once (the mirror of sm90_dense_splitk_route in dispatch.cuh)."""
    if M == 0 or M > 64 or K % 128 or N % 8:
        return 0
    cap = _SM90_SPLITK_CLUSTERS.get(sms)
    tiles, tok = -(-N // 64), max(16, -(-M // 8) * 8)
    s = _sm90_splitk_pref(N, K, sms)
    while s >= 2:
        if M > _sm90_splitk_max_m(s, K):
            return 0
        if cap is None or tiles <= cap[1][s] or (tok <= 24 and tiles <= cap[2][s]):
            return s
        s -= 1
    return 0


def _splitk_re(K: int, M: int, S: int) -> str:
    """A regular expression on the split-K kernel's name: <SHAPE_K, BLOCK_T = max(16, M rounded up to 8), S, ...>."""
    return rf"fp8_gemm_kernel_swapAB_splitk<{K}u, {max(16, -(-M // 8) * 8)}u, {S}u, "


def _sm90_route(names) -> str:
    """The sm_90 dense kernel among `names` and whether it is an epilogue variant: 'dg2', 'v1', 'swapab', 'splitk'
    (+ '+epi')."""
    gemm = [n for n in names if "fp8_gemm_kernel" in n or "sm90_fp8_gemm_1d2d_impl" in n]
    if len(gemm) != 1:
        return f"?{len(gemm)} GEMM kernels"
    n = gemm[0]
    if "fp8_gemm_kernel_swapAB_splitk" in n:
        m = re.search(r"fp8_gemm_kernel_swapAB_splitk<\d+u, \d+u, \d+u, \d+u, (true|false), (\d+), (true|false)>", n)
        epi = m is not None and (m.group(1) == "true" or m.group(2) != "0" or m.group(3) == "true")
        return "splitk" + ("+epi" if epi else "")
    if "sm90_fp8_gemm_1d2d_impl" in n:
        return "dg2" + ("+epi" if "sm90_fp8_gemm_1d2d_impl_fso_epilogue" in n else "")
    if "fp8_gemm_kernel_swapAB" in n:
        return "swapab" + ("+epi" if "NormalSchedulerInputSwapABEpi" in n else "")
    return "v1" + ("+epi" if "NormalSchedulerInputEpi" in n else "")


def _exact_inputs(M: int, N: int, K: int, seed: int):
    """Section 10's order-free exact construction, for the sm_90 1x128 block-FP8 quantizer: x holds bf16 integers in
    [-7, 7] with a +-7 at the head of every 128-element group, so amax = 7, qs = 448 / 7 = 64 and the scale 1 / 64 are
    exact and every code x * 64 is an exact e4m3 value (the MXFP8 form of section 10, integers in [-4, 4] with b = 121,
    is not exact here: 448 / 4 = 112 makes 3 * 112 = 336 inexact in e4m3); row r0 is all zeros. The weight holds e4m3
    codes of integers in [-4, 4], row n0 all negative ([-4, -1]): the -0.0 corner at (r0, n0)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randint(-7, 8, (M, K), device="cuda", generator=g)
    sign = torch.randint(0, 2, (M, K // 128), device="cuda", generator=g) * 2 - 1
    x.view(M, K // 128, 128)[:, :, 0] = 7 * sign
    r0 = 3 if M >= 7 else None  # M = 1 keeps its one row (the corner is exercised from M = 7 on)
    if r0 is not None:
        x[r0].zero_()
    w = torch.randint(-4, 5, (N, K), device="cuda", generator=g)
    n0 = 5
    w[n0] = torch.randint(-4, 0, (K,), device="cuda", generator=g)
    acc = x.double() @ w.double().t()  # exact: integers below 2^53
    b = (torch.randint(-64, 65, (N,), device="cuda", generator=g).double() / 8).bfloat16()
    r = (torch.randint(-1023, 1024, (M, N), device="cuda", generator=g).double() / 8).bfloat16()
    return x.bfloat16(), w.float().to(torch.float8_e4m3fn), acc, b, r, r0, n0


def _rn_bf16(v64: torch.Tensor) -> torch.Tensor:
    """RN_bf16 of a float64 tensor whose values are exact fp32 values (checked)."""
    v32 = v64.float()
    assert torch.equal(v32.double(), v64), "the reference value is not exact in fp32"
    return v32.bfloat16()


def _expected(acc_s: torch.Tensor, b, r, act, has_bias: bool, has_res: bool) -> torch.Tensor:
    """RN_bf16(act(acc + b) + r) from the exact fp64 accumulator acc_s (already scaled)."""
    v = acc_s + (b.double()[None, :] if has_bias else 0.0)
    if act is None:
        if has_res:
            v = v + r.double()
        return _rn_bf16(v)
    v32 = v.float()
    assert torch.equal(v32.double(), v), "acc + b is not exact in fp32"
    g = _gelu_reference_checked(v32.contiguous())
    if has_res:
        g = g + r.float()  # one fp32 addition, rounded to nearest
    return g.bfloat16()


def case_g(major: int, mode: str) -> None:
    """Section-10 exactness of every epilogue variant on the routes this process's FSO_SM90_DENSE_DG2 mode reaches."""
    if major != 9:
        print(f"  g  sm_90 only; skipped on {_arch_label()}")
        return
    want_main = {"0": "v1", "all": "dg2"}.get(mode)
    sms = _sm90_sms()
    ms = (_SM90_EPI_SWAP_MS if mode == "default" else ()) + _SM90_EPI_MAIN_MS
    cells = [(N, K, M, None, "") for N, K in _SM90_EPI_SHAPES for M in ms]
    for N, K, M in (_SM90_EPI_REROUTE_CELLS if mode == "default" else ()):
        if (N, K, M) in _SM90_EPI_FH1:  # moved by FH1's rules: listed below with its new route
            continue
        tile = _sm90_swap_ab_tile(M, N, sms) if _sm90_swap_ab(M, N, K, sms) else None
        cells.append((N, K, M, None, "") if tile is None else
                     (N, K, M, _swap_tile_re(N, K, *tile), f"swap {tile[0]}x{tile[1]}"))
    cells += [(N, K, M, _dg2_tile_re(256, 128), "256x128") for N, K, M, m in _SM90_EPI_TWO_WAVE if m == mode]
    if mode == "default":  # the cells rule 5 re-routes: the v2 kernel at the rule's configuration
        cells += [(N, K, M, _dg2_tile_re(bm, bn, ns), f"{bm}x{bn} s{ns}") for N, K, M, bm, bn, ns in _SM90_EPI_RULE5]
        for (N, K, M), (route, bm, bn, cluster) in _SM90_EPI_FH1.items():  # FH1's routes and their neighbours
            if route == "dg2":
                cells.append((N, K, M, _dg2_cluster_re(bm, bn, cluster), f"{bm}x{bn} {cluster}"))
            elif route == "swapab":
                cells.append((N, K, M, _swap_tile_re(N, K, bm, bn), f"swap {bm}x{bn}"))
            else:
                cells.append((N, K, M, None, ""))
    routes_seen = set()
    for N, K, M, want_tile, tile_label in cells:
        x, wq, acc, b, r, r0, n0 = _exact_inputs(M, N, K, seed=M * 131 + N * 7 + K)
        nb = (N + 127) // 128
        # Weight scale 1 (acc = the integer sums: bf16 keeps 8 bits of a |acc| ~ 400, so double rounding is
        # visible) for the variants without the activation, 2^-8 (|acc| ~ 1.5, the curved part of the GELU) with.
        dws = {s: D.prepare_weight(wq, format="bsfp8",
                                   scale=torch.full((nb, K // 128), 2.0 ** -s, device="cuda")) for s in (0, 8)}
        xq, sx = D.quantize_activation(x)
        if _sm90_splitk(M, N, K, sms):
            want_route = "splitk"  # the in-cluster split-K route (FH5), taken before every route below
        elif mode == "default" and (N, K, M) in _SM90_EPI_FH1:
            want_route = _SM90_EPI_FH1[(N, K, M)][0]  # FH1's route (bands _sm90_swap_ab does not mirror)
        else:
            want_route = "swapab" if _sm90_swap_ab(M, N, K, sms) else want_main
        if want_route is None and 32 < M < 64:  # a main-path neighbour of the swap-AB band (no DG2 class there)
            want_route = "v1"
        if want_route is None and mode == "default" and (N, K, M) in _SM90_EPI_H3_MAIN:
            want_route = _SM90_EPI_H3_MAIN[(N, K, M)]  # a main-path neighbour of the band above M = 64
        if want_route is None and mode == "default" and (N, K, M) in _SM90_EPI_H4_MAIN:
            want_route = _SM90_EPI_H4_MAIN[(N, K, M)]  # a main-path neighbour of the bands above M = 96
        for label, hb, act, hr in _SM90_EPI_VARIANTS:
            s = 8 if act else 0
            dw = dws[s]
            acc_s = acc * 2.0 ** -s
            exp = _expected(acc_s, b, r, act, hb, hr)
            kw = dict(bias=b if hb else None, activation=act, residual=r if hr else None)
            y1, k1 = _kernels_of(lambda: D.linear(x, dw, **kw))
            y2, k2 = _kernels_of(lambda: D.linear(xq, dw, x_scale=sx, **kw))
            route1, route2 = _sm90_route(k1), _sm90_route(k2)
            routes_seen.add(route1.split("+")[0])
            ok1, ok2 = bytes_equal(y1, exp), bytes_equal(y2, exp)
            route_ok = route1 == route2 and route1.endswith("+epi") and len(k2) == 1 and len(k1) == 2 \
                and (want_route is None or route1.split("+")[0] == want_route)
            if want_tile is not None:  # the two-wave and rule-5 cells (the v2 kernel), and the swap-AB cells above
                # M = 32 (fp8_gemm_kernel_swapAB), at the named configuration
                tile_ok = all(any(re.search(want_tile, k) for k in ks) for ks in (k1, k2))
                route_ok = route_ok and route1.startswith("swapab" if tile_label.startswith("swap") else "dg2") \
                    and tile_ok
                route1 += " " + tile_label

            extra = ""
            if label == "bias":
                dbl = (_rn_bf16(acc_s).double() + b.double()[None, :]).float().bfloat16()  # RN(RN(acc) + b)
                frac = float((dbl.view(torch.int16) != exp.view(torch.int16)).double().mean())
                disc_ok = frac > 0
                _check(disc_ok, f"g {mode} N={N} K={K} M={M}: the double-rounding variant equals the expected "
                                f"bytes everywhere; the case does not discriminate")
                extra = f"; double rounding differs on {100 * frac:.1f} % {_ok(disc_ok)}"
            ok = ok1 and ok2 and route_ok
            _check(ok, f"g {mode} N={N} K={K} M={M} {label}: bf16-x {ok1}, 2b {ok2}, route {route1} / {route2} "
                       f"({len(k1)} / {len(k2)} kernels, want {want_route})")
            print(f"  g  [{mode:7s}] N={N:5d} K={K:5d} M={M:5d} {label:19s}: route {route1:10s} "
                  f"bf16-x {_ok(ok1)}, 2b {_ok(ok2)} == RN_bf16(act(acc + b) + r){extra}  {_ok(ok)}")
        # bias = 0 is the identity: the -0.0 corner (x row r0 zero, weight row n0 all negative) and every other
        # element, without and with a residual, bf16 x and 2b.
        dw = dws[0]
        z = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
        for entry, call in (("bf16-x", lambda **kw: D.linear(x, dw, **kw)),
                            ("2b", lambda **kw: D.linear(xq, dw, x_scale=sx, **kw))):
            y_none, y_zero = call(), call(bias=z)
            yr_none, yr_zero = call(residual=r), call(bias=z, residual=r)
            corner = "" if r0 is None else \
                f"; the -0.0 corner's no-bias bytes 0x{int(y_none.view(torch.int16)[r0, n0]) & 0xFFFF:04x}"
            ok = bytes_equal(y_none, y_zero) and bytes_equal(yr_none, yr_zero)
            _check(ok, f"g {mode} N={N} K={K} M={M} {entry}: bias = 0 != bias = None")
            print(f"  g  [{mode:7s}] N={N:5d} K={K:5d} M={M:5d} {entry:6s} bias=0 == bias=None (with and without "
                  f"residual{corner})  {_ok(ok)}")
        del dws
        torch.cuda.empty_cache()
    print(f"  g  [{mode}] kernels reached: {sorted(routes_seen)}")
    if mode == "default":
        # bias = 0 == bias = None on random data, every route of the default mode.
        for N, K in _SM90_EPI_SHAPES:
            dw = prepared(major, N, K, seed=N + 3 * K)
            for M in (1, 7, 78, 129, 1079):
                x = make_x(M, K, seed=M + N + K)
                z = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
                ok = bytes_equal(D.linear(x, dw), D.linear(x, dw, bias=z))
                _check(ok, f"g random N={N} K={K} M={M}: bias = 0 != bias = None")
                print(f"  g  random data N={N:5d} K={K:5d} M={M:5d}: linear(x, w, bias=0) == linear(x, w)  {_ok(ok)}")
            del dw


def _sm90_mode_child(mode: str) -> None:
    """Run case_g (and the capture case) in a child process with FSO_SM90_DENSE_DG2=mode; its lines are printed with
    a prefix and its failures become ours."""
    import subprocess
    env = dict(os.environ)
    env["FSO_SM90_DENSE_DG2"] = mode
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--sm90-epilogue-mode", mode], env=env,
                       capture_output=True, text=True, timeout=3600)
    for line in p.stdout.splitlines():
        print(f"    | {line}")
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-5:]
        _check(False, f"g child FSO_SM90_DENSE_DG2={mode}: exit status {p.returncode}; {' / '.join(tail)}")


# --- (h) capture on a non-default stream, (i) PDL off == on, (j) two streams, (k) torch.compile ----------------------

def _graph_calls(major: int):
    """[(label, fn)]: fused calls on each sm_90 route, both entries."""
    calls = []
    for N, K, M in ((1280, 1280, 7), (5120, 1280, 78), (1280, 5120, 1079), (1280, 7680, 129), (5120, 1280, 1),
                    (2048, 1280, 40), (5120, 2048, 36),  # swap-AB above M = 32 (64- and 128-row tiles)
                    (2048, 4096, 80), (5120, 2048, 96),  # swap-AB above M = 64 (64- and 128-row tiles)
                    (2560, 4096, 136), (9728, 2560, 144),  # above M = 128 (64-row; 128-row, two waves)
                    (2560, 4096, 8), (2560, 9216, 24)):  # the in-cluster split-K route (S = 2, S = 3)
        dw = prepared(major, N, K, seed=71 + N + K)
        x = make_x(M, K, seed=72 + M)
        xq, sx = D.quantize_activation(x)
        b = (torch.randn(N, device="cuda") * 0.1).bfloat16()
        r = (torch.randn(M, N, device="cuda")).bfloat16()
        calls.append((f"N={N:5d} K={K:5d} M={M:5d} bias+gelu, bf16-x",
                      lambda dw=dw, x=x, b=b: D.linear(x, dw, bias=b, activation="gelu_erf")))
        calls.append((f"N={N:5d} K={K:5d} M={M:5d} bias+residual, 2b",
                      lambda dw=dw, xq=xq, sx=sx, b=b, r=r: D.linear(xq, dw, x_scale=sx, bias=b, residual=r)))
        calls.append((f"N={N:5d} K={K:5d} M={M:5d} bias+gelu+residual, 2b",
                      lambda dw=dw, xq=xq, sx=sx, b=b, r=r: D.linear(xq, dw, x_scale=sx, bias=b, residual=r,
                                                                    activation="gelu_erf")))
    return calls


def case_h(major: int, mode: str) -> None:
    if major != 9:
        print(f"  h  sm_90 only; skipped on {_arch_label()}")
        return
    for label, fn in _graph_calls(major):
        _graph_case(f"[{mode}] epilogue {label}", fn)


_PDL_FILE_ENV = "FSO_TEST_DENSE_V1_PDL_OUT"


def _pdl_outputs(major: int) -> dict:
    return {label: fn().clone() for label, fn in _graph_calls(major)}


def case_i(major: int) -> None:
    """FSO_DISABLE_PDL=1 gives the bytes of the default (PDL on) for every fused call of _graph_calls."""
    if major != 9:
        print(f"  i  sm_90 only; skipped on {_arch_label()}")
        return
    import subprocess
    out = os.path.join(tempfile.mkdtemp(prefix="fso_test_dense_v1_pdl_"), "pdl_off.pt")
    env = dict(os.environ)
    env["FSO_DISABLE_PDL"] = "1"
    env[_PDL_FILE_ENV] = out
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--pdl-off-child"], env=env, capture_output=True,
                       text=True, timeout=1800)
    if p.returncode != 0 or not os.path.isfile(out):
        _check(False, f"i: the FSO_DISABLE_PDL=1 child failed ({p.returncode}): {(p.stderr or '')[-300:]}")
        return
    off = torch.load(out)
    on = _pdl_outputs(major)
    for label, y in on.items():
        ok = label in off and bytes_equal(off[label].cuda(), y)
        _check(ok, f"i {label}: FSO_DISABLE_PDL=1 != default")
        print(f"  i  {label}: FSO_DISABLE_PDL=1 == default (PDL on)  {_ok(ok)}")


def case_j(major: int) -> None:
    """Two captured fused calls replayed concurrently on two streams from one thread, 200 times, equal their serial
    (eager) results every time; each replay writes into an output filled with NaN on its own stream first."""
    if major != 9:
        print(f"  j  sm_90 only; skipped on {_arch_label()}")
        return
    calls = dict(_graph_calls(major))
    pairs = (("N= 5120 K= 1280 M=   78 bias+gelu, bf16-x", "N= 1280 K= 1280 M=    7 bias+residual, 2b"),
             ("N= 1280 K= 5120 M= 1079 bias+gelu+residual, 2b", "N= 1280 K= 7680 M=  129 bias+gelu, bf16-x"),
             # swap-AB above M = 32, 64- and 128-row weight tiles
             ("N= 2048 K= 1280 M=   40 bias+gelu, bf16-x", "N= 5120 K= 2048 M=   36 bias+gelu+residual, 2b"),
             # swap-AB above M = 64, 64- and 128-row weight tiles
             ("N= 2048 K= 4096 M=   80 bias+gelu, bf16-x", "N= 5120 K= 2048 M=   96 bias+gelu+residual, 2b"),
             # swap-AB above M = 128, 64-row tiles and 128-row tiles on two waves
             ("N= 2560 K= 4096 M=  136 bias+residual, 2b", "N= 9728 K= 2560 M=  144 bias+gelu+residual, 2b"),
             # the in-cluster split-K route, S = 2 and S = 3: two cluster launches side by side
             ("N= 2560 K= 4096 M=    8 bias+gelu, bf16-x", "N= 2560 K= 9216 M=   24 bias+gelu+residual, 2b"))
    for la, lb in pairs:
        fa, fb = calls[la], calls[lb]
        ra, rb = fa().clone(), fb().clone()
        torch.cuda.synchronize()
        sa, sb = torch.cuda.Stream(), torch.cuda.Stream()
        ga, gb = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        for s in (sa, sb):
            s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(ga, stream=sa):
            oa = fa()
        with torch.cuda.graph(gb, stream=sb):
            ob = fb()
        torch.cuda.synchronize()
        bad = 0
        for _ in range(200):
            with torch.cuda.stream(sa):
                oa.fill_(float("nan"))
                ga.replay()
            with torch.cuda.stream(sb):
                ob.fill_(float("nan"))
                gb.replay()
            torch.cuda.synchronize()
            bad += int(not torch.equal(oa, ra)) + int(not torch.equal(ob, rb))
        ok = bad == 0
        _check(ok, f"j {la} || {lb}: {bad} of 400 concurrent replays differ from the serial results")
        print(f"  j  [{la}] on stream 1 || [{lb}] on stream 2, 200 concurrent replays == serial: {bad} mismatches  "
              f"{_ok(ok)}")
        del ga, gb


def case_k(major: int) -> None:
    """torch.compile(fullgraph=True): bias / residual / activation stay arguments of the one dense_linear node (no
    separate add or gelu node), and inductor's output equals eager."""
    if major != 9:
        print(f"  k  sm_90 only; skipped on {_arch_label()}")
        return
    N, K = 5120, 1280
    dw = prepared(major, N, K, seed=81)
    b = (torch.randn(N, device="cuda") * 0.1).bfloat16()

    def f_bf16(x, r):
        return D.linear(x, dw, bias=b, residual=r, activation="gelu_erf")

    def f_2b(x, r):
        xq, sx = D.quantize_activation(x)
        return D.linear(xq, dw, x_scale=sx, bias=b, residual=r)

    for name, f, want in (("bf16 x, bias + gelu_erf + residual", f_bf16, {"dense_linear": 1}),
                          ("2b, bias + residual", f_2b, {"dense_quantize_activation": 1, "dense_linear": 1})):
        xs = {M: (make_x(M, K, seed=82 + M), torch.randn(M, N, device="cuda").bfloat16()) for M in (78, 1079, 7)}
        eager = {M: f(*a) for M, a in xs.items()}
        graphs = []

        def record(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        torch._dynamo.reset()
        rec = torch.compile(f, backend=record, fullgraph=True)
        rec_ok = all(torch.equal(rec(*a), eager[M]) for M, a in xs.items())
        per_graph = []
        for g in graphs:
            calls = [str(n.target) for n in g.graph.nodes if n.op == "call_function"]
            per_graph.append(calls)
        node_ok = bool(graphs)
        import operator
        for g, calls in zip(graphs, per_graph):
            ours = [c for c in calls if "fish_scales_ops" in c]
            # Any other node that produces a tensor (an add, a gelu, ...) would be a part of the epilogue left
            # outside the op; the tuple getitem of quantize_activation is allowed, and so are the symbolic-shape
            # nodes (eq, not_, ...) of a recompiled dynamic graph, which produce no tensor.
            others = [str(n.target) for n in g.graph.nodes if n.op == "call_function"
                      and "fish_scales_ops" not in str(n.target) and n.target is not operator.getitem
                      and isinstance(n.meta.get("example_value"), torch.Tensor)]
            counts = {k: sum(k in c and not (k == "dense_linear" and "quantized" in c) for c in ours) for k in want}
            node_ok = node_ok and counts == want and len(ours) == sum(want.values()) and not others
        _check(rec_ok and node_ok, f"k {name}: {len(graphs)} graphs, nodes {per_graph}, == eager {rec_ok}")
        print(f"  k  {name}, fullgraph=True: {len(graphs)} graph(s); call_function nodes "
              f"{per_graph[0] if per_graph else []}; == eager  {_ok(rec_ok and node_ok)}")
        torch._dynamo.reset()
        ind = torch.compile(f, fullgraph=True)
        for M, a in xs.items():
            got = ind(*a)
            torch.cuda.synchronize()
            ok = torch.equal(got, eager[M])
            _check(ok, f"k {name} inductor M={M}: compiled != eager")
            print(f"  k  {name}, inductor, M={M:5d}: compiled == eager  {_ok(ok)}")
        torch._dynamo.reset()


def case_l(major: int) -> None:
    """A fused call allocates what the identity call allocates (its output; on the bf16-x path the quantize
    buffers) and nothing more, and does not synchronize the device."""
    if major != 9:
        print(f"  l  sm_90 only; skipped on {_arch_label()}")
        return
    for N, K, M in ((1280, 1280, 7), (5120, 1280, 78), (1280, 5120, 1079)):
        dw = prepared(major, N, K, seed=91 + N)
        x = make_x(M, K, seed=92 + M)
        xq, sx = D.quantize_activation(x)
        b = (torch.randn(N, device="cuda") * 0.1).bfloat16()
        r = torch.randn(M, N, device="cuda").bfloat16()
        for entry, ident, fused in (
                ("bf16-x", lambda: D.linear(x, dw), lambda: D.linear(x, dw, bias=b, residual=r, activation="gelu_erf")),
                ("2b", lambda: D.linear(xq, dw, x_scale=sx),
                 lambda: D.linear(xq, dw, x_scale=sx, bias=b, residual=r, activation="gelu_erf"))):
            ident(), fused()  # warm: JIT compile and caches
            torch.cuda.synchronize()

            def allocs(fn):
                a0 = torch.cuda.memory_stats()["allocation.all.allocated"]
                y = fn()
                torch.cuda.synchronize()
                return torch.cuda.memory_stats()["allocation.all.allocated"] - a0, y

            n_ident, _ = allocs(ident)
            n_fused, _ = allocs(fused)
            ok = n_ident == n_fused
            _check(ok, f"l N={N} K={K} M={M} {entry}: the fused call made {n_fused} allocations, the identity "
                       f"call {n_ident}")
            print(f"  l  N={N:5d} K={K:5d} M={M:5d} {entry:6s}: allocations per call, fused {n_fused} == identity "
                  f"{n_ident}  {_ok(ok)}")
            _no_sync(f"fused linear N={N} M={M} {entry}", fused)


# --- (n) bias / residual views at an odd element offset (2-byte aligned only) -------------------------------------

def case_n(major: int, mode: str) -> None:
    """A bias or residual that starts at an odd element offset (a contiguous view such as t[1:], which the op
    accepts) takes the epilogue's 2-byte load path instead of the paired 4-byte one: its bytes equal those of the same
    call on aligned copies, on each route this mode reaches, for a bf16 x and the 2b form."""
    if major != 9:
        print(f"  n  sm_90 only; skipped on {_arch_label()}")
        return
    ms = ((7, 32) if mode == "default" else ()) + (78, 1079)
    cells = [(N, K, ms) for N, K in ((1280, 1280), (5120, 1280))]
    if mode == "default":  # swap-AB above M = 32, 64 and 128 (case g): 64- and 128-row weight tiles; FH1's routes
        cells += [(2048, 1280, (40,)), (5120, 2048, (36, 96)), (2048, 4096, (80,)), (2560, 4096, (136,)),
                  (6144, 2560, (160,)), (2560, 4096, (216, 288)), (18432, 2560, (8, 288))]
    for N, K, cell_ms in cells:
        dw = prepared(major, N, K, seed=111 + N)
        for M in cell_ms:
            x = make_x(M, K, seed=112 + M)
            xq, sx = D.quantize_activation(x)
            b = (torch.randn(N, device="cuda") * 0.1).bfloat16()
            r = torch.randn(M, N, device="cuda").bfloat16()
            bo = torch.empty(N + 1, dtype=torch.bfloat16, device="cuda")[1:]
            bo.copy_(b)
            ro = torch.empty(M * N + 1, dtype=torch.bfloat16, device="cuda")[1:].view(M, N)
            ro.copy_(r)
            assert bo.is_contiguous() and ro.is_contiguous() and bo.data_ptr() % 4 == 2 and ro.data_ptr() % 4 == 2
            for label, kw, kwo in (("bias+residual", dict(bias=b, residual=r), dict(bias=bo, residual=ro)),
                                   ("bias+gelu+residual", dict(bias=b, residual=r, activation="gelu_erf"),
                                    dict(bias=bo, residual=ro, activation="gelu_erf"))):
                (y_a, kn), y_o = _kernels_of(lambda: D.linear(xq, dw, x_scale=sx, **kw)), \
                    D.linear(xq, dw, x_scale=sx, **kwo)
                ok2 = bytes_equal(y_a, y_o)
                ok1 = bytes_equal(D.linear(x, dw, **kw), D.linear(x, dw, **kwo))
                ok = ok1 and ok2
                _check(ok, f"n [{mode}] N={N} K={K} M={M} {label}: odd-offset views != aligned (bf16-x {ok1}, 2b {ok2})")
                print(f"  n  [{mode:7s}] N={N:5d} K={K:5d} M={M:5d} {label:18s} route {_sm90_route(kn):10s}: bias / "
                      f"residual at an odd element offset == aligned copies (bf16-x {_ok(ok1)}, 2b {_ok(ok2)})  "
                      f"{_ok(ok)}")
        del dw


# --- (m) PDL: the epilogue reads bias / residual only after griddepcontrol.wait ------------------------------------

_PDL_CPP = r"""
#include <torch/extension.h>
void slow_copy2(torch::Tensor src1, torch::Tensor dst1, torch::Tensor src2, torch::Tensor dst2, int64_t blocks,
                int64_t spin_ns);
void early_read(torch::Tensor p, torch::Tensor stale);
"""

_PDL_CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// The producer: lets its programmatic dependents launch at once (griddepcontrol.launch_dependents), spins for
// spin_ns on the global timer, then copies src -> dst. A dependent that read dst before its griddepcontrol.wait
// would see the old contents.
__global__ void slow_copy2_kernel(const uint16_t* s1, uint16_t* d1, int64_t n1, const uint16_t* s2, uint16_t* d2,
                                  int64_t n2, unsigned long long spin_ns) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < spin_ns);
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n1; i += stride) d1[i] = s1[i];
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n2; i += stride) d2[i] = s2[i];
}

// The positive control: a programmatic dependent that reads p BEFORE its griddepcontrol.wait and counts the
// elements still zero (stale), then waits.
__global__ void early_read_kernel(const volatile uint16_t* p, int64_t n, int* stale) {
  int c = 0;
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) c += (p[i] == 0);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  atomicAdd(stale, c);
}

void slow_copy2(torch::Tensor src1, torch::Tensor dst1, torch::Tensor src2, torch::Tensor dst2, int64_t blocks,
                int64_t spin_ns) {
  slow_copy2_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint16_t*>(src1.data_ptr()), reinterpret_cast<uint16_t*>(dst1.data_ptr()), src1.numel(),
      reinterpret_cast<const uint16_t*>(src2.data_ptr()), reinterpret_cast<uint16_t*>(dst2.data_ptr()), src2.numel(),
      static_cast<unsigned long long>(spin_ns));
}

void early_read(torch::Tensor p, torch::Tensor stale) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = 1;
  cfg.blockDim = 256;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, early_read_kernel, reinterpret_cast<const volatile uint16_t*>(p.data_ptr()),
                                 p.numel(), stale.data_ptr<int>()) == cudaSuccess);
}
"""

_pdl_ext = None


def _pdl_probe():
    global _pdl_ext
    if _pdl_ext is None:
        from torch.utils.cpp_extension import load_inline
        major, minor = _arch()
        name = f"fso_pdl_probe_sm{major}{minor}"
        build_dir = os.path.join(tempfile.gettempdir(), f"fso_test_dense_v1_{os.getuid()}", name)
        os.makedirs(build_dir, exist_ok=True)
        saved = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        try:
            _pdl_ext = load_inline(name=name, cpp_sources=_PDL_CPP, cuda_sources=_PDL_CU,
                                   functions=["slow_copy2", "early_read"], extra_cuda_cflags=["-O3"],
                                   build_directory=build_dir, verbose=False)
        finally:
            if saved is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = saved
    return _pdl_ext


def case_m(major: int, mode: str) -> None:
    """PDL: a producer that lets its dependents launch at once and writes the bias and residual 2 ms later, directly
    ahead of a fused call; the call gives the bytes of the same call with the final tensors (every epilogue load is
    after the GEMM's griddepcontrol.wait). The control shows the window is real: a dependent that reads before its
    wait sees the stale (zero) tensor. Skipped with FSO_DISABLE_PDL=1 (nothing overlaps then)."""
    if major != 9:
        print(f"  m  sm_90 only; skipped on {_arch_label()}")
        return
    if os.environ.get("FSO_DISABLE_PDL") == "1":
        print("  m  FSO_DISABLE_PDL=1: no programmatic launches; skipped")
        return
    ext = _pdl_probe()
    spin = 2_000_000  # ns
    # Load both probe kernels once (lazy module loading would otherwise delay the first control's reader past the
    # producer's spin, and the control would see no overlap).
    t0, t1 = torch.ones(256, dtype=torch.int16, device="cuda"), torch.zeros(256, dtype=torch.int16, device="cuda")
    ext.slow_copy2(t0, t1, t0, t1, 1, 1000)
    ext.early_read(t1, torch.zeros(1, dtype=torch.int32, device="cuda"))
    torch.cuda.synchronize()
    for N, K, M in ((1280, 1280, 7), (5120, 1280, 78), (1280, 5120, 1079), (1280, 7680, 129),
                    (2560, 4096, 8), (2560, 9216, 24)):  # the last two: the in-cluster split-K route (S = 2, 3)
        dw = prepared(major, N, K, seed=101 + N)
        x = make_x(M, K, seed=102 + M)
        xq, sx = D.quantize_activation(x)
        b = (torch.randn(N, device="cuda") * 0.1 + 0.5).bfloat16()  # no zero element (the control counts zeros)
        r = (torch.randn(M, N, device="cuda") + 4.0).bfloat16()
        db, dr = torch.zeros_like(b), torch.zeros_like(r)
        for entry, call in (("bf16-x", lambda bb, rr: D.linear(x, dw, bias=bb, residual=rr, activation="gelu_erf")),
                            ("2b", lambda bb, rr: D.linear(xq, dw, x_scale=sx, bias=bb, residual=rr,
                                                           activation="gelu_erf"))):
            want = call(b, r)
            call(db, dr)  # warm (JIT) with the destination tensors
            stale = torch.zeros(1, dtype=torch.int32, device="cuda")
            db.zero_(), dr.zero_()
            torch.cuda.synchronize()
            ext.slow_copy2(b.view(torch.int16), db.view(torch.int16), r.view(torch.int16), dr.view(torch.int16), 8,
                           spin)
            ext.early_read(db.view(torch.int16), stale)
            torch.cuda.synchronize()
            n_stale = int(stale.item())
            db.zero_(), dr.zero_()
            torch.cuda.synchronize()
            ext.slow_copy2(b.view(torch.int16), db.view(torch.int16), r.view(torch.int16), dr.view(torch.int16), 8,
                           spin)
            got = call(db, dr)
            torch.cuda.synchronize()
            ok = bytes_equal(got, want)
            ctrl = n_stale > 0
            _check(ok, f"m [{mode}] N={N} K={K} M={M} {entry}: the fused call behind a late producer != the call on "
                       f"the final tensors")
            _check(ctrl, f"m [{mode}] N={N} K={K} M={M} {entry}: the control saw no stale element (no overlap)")
            print(f"  m  [{mode:7s}] N={N:5d} K={K:5d} M={M:5d} {entry:6s}: behind a producer that writes bias / "
                  f"residual 2 ms after releasing its dependents == on the final tensors {_ok(ok)}; control: a "
                  f"pre-wait reader saw {n_stale} of {N} bias elements stale {_ok(ctrl)}")
    if mode != "default":
        return
    # The weight and its scale written late: the split-K kernel reads them only after its griddepcontrol.wait too
    # (fso_dense_splitk.cuh), so a call right behind a producer that rewrites the handle's tensors in place (a reload,
    # or prepare_weight's kernels just before) gives the bytes of the call on the final tensors. The 2b call follows
    # the producer directly; the bf16-x call has the activation quantize in between. 5120 x 2048 (the swap-AB route)
    # is the reference.
    for N, K, M in ((2560, 4096, 8), (2560, 9216, 24), (5120, 2048, 8)):
        dw = prepared(major, N, K, seed=105 + N)
        x = make_x(M, K, seed=106 + M)
        xq, sx = D.quantize_activation(x)
        w_src, s_src = dw.weight.clone(), dw.scale.clone()
        for entry, call in (("bf16-x", lambda: D.linear(x, dw)), ("2b", lambda: D.linear(xq, dw, x_scale=sx))):
            want = call()
            torch.cuda.synchronize()
            stale = torch.zeros(1, dtype=torch.int32, device="cuda")
            dw.weight.zero_(), dw.scale.zero_()
            torch.cuda.synchronize()
            ext.slow_copy2(w_src.view(torch.int16), dw.weight.view(torch.int16), s_src.view(torch.int16),
                           dw.scale.view(torch.int16), 8, spin)
            ext.early_read(dw.scale.view(torch.int16), stale)
            torch.cuda.synchronize()
            n_stale = int(stale.item())
            dw.weight.zero_(), dw.scale.zero_()
            torch.cuda.synchronize()
            ext.slow_copy2(w_src.view(torch.int16), dw.weight.view(torch.int16), s_src.view(torch.int16),
                           dw.scale.view(torch.int16), 8, spin)
            got = call()
            torch.cuda.synchronize()
            ok = bytes_equal(got, want)
            ctrl = n_stale > 0
            _check(ok, f"m N={N} K={K} M={M} {entry}: the call behind a producer that writes the weight late != the "
                       f"call on the final weight")
            _check(ctrl, f"m N={N} K={K} M={M} {entry}: the weight control saw no stale element (no overlap)")
            print(f"  m  [{mode:7s}] N={N:5d} K={K:5d} M={M:5d} {entry:6s}: behind a producer that writes the weight "
                  f"and its scale 2 ms after releasing its dependents == on the final weight {_ok(ok)}; control: a "
                  f"pre-wait reader saw {n_stale} scale halves stale {_ok(ctrl)}")


# --- (t) the sm_90 in-cluster split-K route (FH5) --------------------------------------------------------------------
# fp8_gemm_kernel_swapAB_splitk (deep_gemm/fso_dense_splitk.cuh): one 64-row weight tile per cluster of S CTAs that
# divide K and add their fp32 partial sums in rank order (p_0 + p_1 + ... + p_{S-1}), then apply the epilogue once and
# round. The cells the rule takes (N, K, M; _sm90_splitk mirrors the dispatcher): S = 2 with token tiles 16, 48 and 56
# on 2560 x 4096, S = 3 on 2048 x 4096 (token tile 32), S = 4 on 1280 x 5120 at M = 56, S = 3 sharing SMs on the 40
# tiles of 2560 x 9216 at M = 8 and S = 2 one CTA per SM from M = 25 (token tiles 40 and 64), S = 4 on 1024 x 2048, S = 2
# on the 4 K blocks of 2048 x 512 and on the 64 tiles of 4096 x 2048; and neighbours it leaves on the routes below it
# (one row past each K class's band edge, 80 and 67 weight tiles). The forced cells (FSO_SM90_DENSE_SPLITK=force with
# FSO_SM90_SPLITK_S = 2, 3, 4 in child processes) cover token tiles up to 64 and an N that is not a multiple of 64,
# each with clusters that fit one CTA per SM.
_SM90_SPLITK_CELLS = ((2560, 4096, 1), (2560, 4096, 16), (2560, 4096, 48), (2560, 4096, 56), (2048, 4096, 1),
                      (2048, 4096, 32), (1280, 5120, 56), (2560, 9216, 8), (2560, 9216, 40), (2560, 9728, 64),
                      (1024, 2048, 16), (2048, 512, 16), (4096, 2048, 8))
_SM90_SPLITK_OFF = ((2560, 4096, 57), (2048, 7680, 57), (1024, 2048, 41), (2048, 512, 25), (5120, 2048, 8),
                    (4288, 4096, 16))
_SM90_SPLITK_FORCED = {2: ((2560, 4096, 40), (2528, 9216, 24), (4096, 4096, 7)),
                       3: ((2048, 4096, 64), (2496, 9216, 7), (1280, 5120, 33)),
                       4: ((1024, 4096, 7), (1920, 9216, 48), (1240, 2048, 64))}
_T_FILE_ENV = "FSO_TEST_DENSE_V1_T_OUT"


def _t_random_outputs() -> dict:
    """{(N, K, M, call): output} on random data for the cells of the rule: the identity bf16-x call and the bias + GELU
    + residual 2b call."""
    outs = {}
    for N, K, M in _SM90_SPLITK_CELLS:
        dw = prepared(9, N, K, seed=301 + N + K)
        x = make_x(M, K, seed=302 + M)
        xq, sx = D.quantize_activation(x)
        g = torch.Generator(device="cuda").manual_seed(303 + M + N)
        b = (torch.randn(N, device="cuda", generator=g) * 0.1).bfloat16()
        r = torch.randn(M, N, device="cuda", generator=g).bfloat16()
        outs[(N, K, M, "bf16-x")] = D.linear(x, dw).clone()
        outs[(N, K, M, "2b+epi")] = D.linear(xq, dw, x_scale=sx, bias=b, residual=r, activation="gelu_erf").clone()
    return outs


def _t_exact_cell(N: int, K: int, M: int, want_s: int, tag: str) -> None:
    """Section 10 on one split-K cell: every epilogue variant and the identity give RN_bf16(act(acc + b) + r) from both
    entries on the split-K kernel with `want_s` slices (the kernel name), and bias = 0 gives the bytes of bias = None."""
    x, wq, acc, b, r, r0, n0 = _exact_inputs(M, N, K, seed=M * 131 + N * 7 + K)
    nb = (N + 127) // 128
    dws = {s: D.prepare_weight(wq, format="bsfp8", scale=torch.full((nb, K // 128), 2.0 ** -s, device="cuda"))
           for s in (0, 8)}
    xq, sx = D.quantize_activation(x)
    want_re = _splitk_re(K, M, want_s)
    for label, hb, act, hr in (("identity", False, None, False),) + _SM90_EPI_VARIANTS:
        s = 8 if act else 0
        exp = _expected(acc * 2.0 ** -s, b, r, act, hb, hr)
        kw = dict(bias=b if hb else None, activation=act, residual=r if hr else None)
        y1, k1 = _kernels_of(lambda: D.linear(x, dws[s], **kw))
        y2, k2 = _kernels_of(lambda: D.linear(xq, dws[s], x_scale=sx, **kw))
        route1, route2 = _sm90_route(k1), _sm90_route(k2)
        want_route = "splitk" + ("" if label == "identity" else "+epi")
        route_ok = route1 == route2 == want_route and all(any(re.search(want_re, k) for k in ks) for ks in (k1, k2))
        ok1, ok2 = bytes_equal(y1, exp), bytes_equal(y2, exp)
        ok = ok1 and ok2 and route_ok
        _check(ok, f"t {tag} N={N} K={K} M={M} {label}: bf16-x {ok1}, 2b {ok2}, route {route1} / {route2} (want "
                   f"{want_route} S={want_s})")
        print(f"  t  {tag} N={N:5d} K={K:5d} M={M:3d} S={want_s} {label:19s}: route {route1:11s} bf16-x {_ok(ok1)}, "
              f"2b {_ok(ok2)} == RN_bf16(act(acc + b) + r)  {_ok(ok)}")
    dw = dws[0]
    z = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
    for entry, call in (("bf16-x", lambda **kw: D.linear(x, dw, **kw)),
                        ("2b", lambda **kw: D.linear(xq, dw, x_scale=sx, **kw))):
        y_none, y_zero = call(), call(bias=z)
        yr_none, yr_zero = call(residual=r), call(bias=z, residual=r)
        corner = "" if r0 is None else \
            f"; the -0.0 corner's no-bias bytes 0x{int(y_none.view(torch.int16)[r0, n0]) & 0xFFFF:04x}"
        ok = bytes_equal(y_none, y_zero) and bytes_equal(yr_none, yr_zero)
        _check(ok, f"t {tag} N={N} K={K} M={M} {entry}: bias = 0 != bias = None")
        print(f"  t  {tag} N={N:5d} K={K:5d} M={M:3d} {entry:6s} bias=0 == bias=None (with and without residual"
              f"{corner})  {_ok(ok)}")


def _t_worker(mode: str) -> int:
    """A child of case (t): "off" / "s1" save _t_random_outputs() (FSO_SM90_DENSE_SPLITK=0, or FSO_SM90_SPLITK_S=1);
    "force<s>" runs section 10 on the forced cells with s slices."""
    major, _ = _arch()
    if mode in ("off", "s1"):
        torch.save({k: v.cpu() for k, v in _t_random_outputs().items()}, os.environ[_T_FILE_ENV])
        return 0
    s = int(mode[len("force"):])
    for N, K, M in _SM90_SPLITK_FORCED[s]:
        _t_exact_cell(N, K, M, s, f"[force S={s}]")
    return _finish(f"t child {mode}: ALL PASS")


def _t_child(mode: str, env_extra: dict, out: str = "") -> bool:
    import subprocess
    env = {k: v for k, v in os.environ.items() if not k.startswith("FSO_SM90_")}
    env.update(env_extra)
    if out:
        env[_T_FILE_ENV] = out
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--t-worker", mode], env=env, capture_output=True,
                       text=True, timeout=3600)
    for line in p.stdout.splitlines():
        print(f"    | {line}")
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-5:]
        _check(False, f"t child {mode}: exit status {p.returncode}; {' / '.join(tail)}")
    return p.returncode == 0


def _bf16_ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|a - b| in units in the last place of bf16, on the monotonic integer image of the bit patterns (+0 and -0 both
    map to 0)."""
    def ordered(t):
        v = t.view(torch.int16).to(torch.int32)
        m = v & 0x7FFF
        return torch.where(v < 0, -m, m)
    return (ordered(a) - ordered(b)).abs()


def case_t(major: int) -> None:
    """The sm_90 in-cluster split-K route: its routes, section-10 exactness on every epilogue variant from both entries,
    bias = 0, bf16-x == 2b, determinism, the one-slice kernel against the swap-AB route, and the difference of the
    split route from the one-CTA-per-element route it replaces."""
    if major != 9:
        print(f"  t  sm_90 only; skipped on {_arch_label()}")
        return
    sms = _sm90_sms()
    # Routes: the rule's cells run the split-K kernel at the mirror's slice count and token tile; the neighbours do not.
    for N, K, M in _SM90_SPLITK_CELLS + _SM90_SPLITK_OFF:
        s = _sm90_splitk(M, N, K, sms)
        dw = prepared(major, N, K, seed=311 + N)
        x = make_x(M, K, seed=312 + M)
        _, ks = _kernels_of(lambda: D.linear(x, dw))
        route = _sm90_route(ks)
        ok = (route == "splitk" and any(re.search(_splitk_re(K, M, s), k) for k in ks)) if s else \
            (route in ("swapab", "v1", "dg2"))
        _check(ok, f"t route N={N} K={K} M={M}: {route}, want {'splitk S=' + str(s) if s else 'not splitk'}")
        print(f"  t  route N={N:5d} K={K:5d} M={M:3d}: {route:8s} (rule: {'S=' + str(s) if s else 'off'})  {_ok(ok)}")
    for N, K, M in _SM90_SPLITK_CELLS:
        _t_exact_cell(N, K, M, _sm90_splitk(M, N, K, sms), "[rule]")
    # Random data: bf16-x == 2b, and 20 repeated calls give the same bytes (a fixed summation order).
    outs = _t_random_outputs()
    for N, K, M in _SM90_SPLITK_CELLS:
        dw = prepared(major, N, K, seed=301 + N + K)
        x = make_x(M, K, seed=302 + M)
        xq, sx = D.quantize_activation(x)
        y1, y2 = D.linear(x, dw), D.linear(xq, dw, x_scale=sx)
        same = bytes_equal(y1, y2) and bytes_equal(y1, outs[(N, K, M, "bf16-x")])
        rep = all(bytes_equal(D.linear(x, dw), y1) for _ in range(20))
        ok = same and rep
        _check(ok, f"t random N={N} K={K} M={M}: bf16-x == 2b {same}, 20 calls identical {rep}")
        print(f"  t  random N={N:5d} K={K:5d} M={M:3d}: bf16-x == 2b {_ok(same)}, 20 repeated calls identical "
              f"{_ok(rep)}")
    # Children: the route off (the one-CTA-per-element swap-AB route) and the split-K kernel with one slice.
    tmp = tempfile.mkdtemp(prefix="fso_test_dense_v1_t_")
    off_file, s1_file = os.path.join(tmp, "off.pt"), os.path.join(tmp, "s1.pt")
    if _t_child("off", {"FSO_SM90_DENSE_SPLITK": "0"}, off_file) and \
            _t_child("s1", {"FSO_SM90_SPLITK_S": "1"}, s1_file):
        off, s1 = torch.load(off_file), torch.load(s1_file)
        for key, y in outs.items():
            N, K, M, call = key
            yo, y1 = off[key].cuda(), s1[key].cuda()
            one_ok = bytes_equal(y1, yo)
            _check(one_ok, f"t N={N} K={K} M={M} {call}: the one-slice split-K kernel != the swap-AB route")
            ulps = _bf16_ulps(y, yo)
            n_diff = int((ulps > 0).sum())
            print(f"  t  N={N:5d} K={K:5d} M={M:3d} {call:6s}: one-slice split-K == swap-AB route {_ok(one_ok)}; the "
                  f"rule's split route differs from it in {n_diff} of {y.numel()} elements "
                  f"({100.0 * n_diff / y.numel():.3f} %), at most {int(ulps.max())} bf16 ulp")
    for s in (2, 3, 4):
        _t_child(f"force{s}", {"FSO_SM90_DENSE_SPLITK": "force", "FSO_SM90_SPLITK_S": str(s)})


# --- (o) the sm_120/sm_121 dense epilogue (S1c) ---------------------------------------------------------------------
#
# Charter section 10 on every route of the sm_120 MXFP8 cascade. Every route is run in a subprocess of its own,
# because the FSO_FORCE_* knobs are read once per process: the natural cascade (the issue-12 shapes, the charter's
# M x K grid and shapes on which the cascade itself splits K), each of the ten single-launch tiles forced with
# FSO_FORCE_TILE, each of the seven Stream-K tiles forced with FSO_FORCE_TILE + FSO_FORCE_KSPLIT, the three forced
# diagnostic instances (which must refuse), FSO_DISABLE_PDL=1 reruns, and the graph / stream / compile cases.
#
# The exact construction: weights are e4m3 codes of integers in [-4, 4] with UE8M0 byte 127 (value = the integer);
# activations are bf16 integers in [-4, 4] with a +-4 in every 32-block, so the quantizer picks byte 121 (scale 2^-6)
# and the codes x * 64 are exact. Every product is an integer and |acc| <= 16 K < 2^24, so the fp32 accumulator is
# exact in any order and known from an fp64 matmul. Bias and residual are multiples of 1/8 (|r| < 2^7), so acc + b
# and + r are exact in fp32. On a Stream-K route acc is the fp32 sum of the slices' bf16-rounded partials (what the
# reduce sums), which the reference forms from the slices' exact fp64 partials. The zeros of acc take their sign
# from the identity call's output (RN_bf16 keeps it), so the expected bytes are exact including signed zeros.
# erf-GELU is applied by a CUDA reference kernel (load_inline, the library's compiler and flags, the same erff).

S1C_TOWER = ((1280, 1280), (5120, 1280), (1280, 5120), (1280, 7680), (2048, 1280))
S1C_TOWER_MS = (78, 117, 182, 364, 533, 1079)
S1C_LLM = ((2560, 2048), (2048, 2048))
S1C_LLM_MS = (1, 8, 32, 64, 128, 512, 2048)
S1C_GRID_KS = (1280, 5120, 7680)
S1C_GRID_MS = (1, 7, 78, 129, 1079)
# Shapes on which the natural cascade splits K: narrow N with K >= 9728 at several M (TileM 16 / 32 / 64 / 128
# Stream-K tiles), and tiles_n >= 32 at M <= 32 (K split 2).
S1C_STREAMK_NATURAL = ((1, 2560, 9728), (7, 2560, 10240), (16, 2560, 9728), (32, 2560, 9728), (64, 2560, 9728),
                       (100, 2560, 9728), (129, 2560, 9728), (300, 2560, 9728), (8, 4096, 4096), (20, 4096, 4096),
                       (32, 6144, 5120))
S1C_SINGLE_TILES = ((16, 128, 4), (16, 64, 4), (32, 128, 4), (32, 64, 4), (64, 64, 4), (64, 128, 2), (96, 128, 2),
                    (128, 64, 2), (128, 128, 2), (160, 128, 2))
S1C_SK_TILES = (((16, 128, 4), 4), ((32, 128, 4), 2), ((32, 64, 4), 4), ((64, 128, 2), 3), ((64, 64, 4), 4),
                ((128, 128, 2), 2), ((128, 64, 2), 4))
# The epilogue variants: (label, bias, activation, residual).
S1C_VARIANTS = (("bias", True, None, False), ("residual", False, None, True), ("gelu", False, "gelu_erf", False),
                ("bias+gelu", True, "gelu_erf", False), ("bias+residual", True, None, True),
                ("bias+gelu+residual", True, "gelu_erf", True))
S1C_KNOBS = ("FSO_FORCE_TILE", "FSO_FORCE_KSPLIT", "FSO_FORCE_MIN_BLOCKS", "FSO_FORCE_SMALLM", "FSO_FORCE_SCHED_GROUP",
             "FSO_FORCE_TILE_K", "FSO_DISABLE_PDL", "FSO_DISABLE_STREAMK", "FSO_DISABLE_OVERRIDES",
             "BSGEMM_FORCE_TILE", "BSGEMM_FORCE_KSPLIT")

_GELU_REF_CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
// The contract's erf GELU, (0.5f * v) * (1.f + erff(v * 0.70710678118654752440f)) in fp32: `rn` spells every
// operation with an _rn intrinsic (never contracted into an fma), `plain` is the expression as written.
__global__ void gelu_rn_kernel(const float* v, float* o, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float x = v[i];
    o[i] = __fmul_rn(__fmul_rn(0.5f, x), __fadd_rn(1.f, erff(__fmul_rn(x, 0.70710678118654752440f))));
  }
}
__global__ void gelu_plain_kernel(const float* v, float* o, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float x = v[i];
    o[i] = 0.5f * x * (1.f + erff(x * 0.70710678118654752440f));
  }
}
torch::Tensor gelu_ref(torch::Tensor v, bool plain) {
  TORCH_CHECK(v.is_cuda() && v.scalar_type() == at::kFloat && v.is_contiguous());
  auto o = torch::empty_like(v);
  const int64_t n = v.numel();
  const int threads = 256;
  const int64_t blocks = (n + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (n > 0) {
    if (plain) gelu_plain_kernel<<<blocks, threads, 0, stream>>>(v.data_ptr<float>(), o.data_ptr<float>(), n);
    else gelu_rn_kernel<<<blocks, threads, 0, stream>>>(v.data_ptr<float>(), o.data_ptr<float>(), n);
  }
  return o;
}
"""
_GELU_REF_CPP = "#include <torch/extension.h>\ntorch::Tensor gelu_ref(torch::Tensor v, bool plain);\n"
_GELU_EXTS: dict = {}


def _s1c_gelu_ext(fmad: bool):
    """The reference GELU extension, built with the library's nvcc flags (-O3, fma contraction on) or with
    -fmad=false (the expression's every operation rounded, the literal reading of the formula)."""
    if fmad in _GELU_EXTS:
        return _GELU_EXTS[fmad]
    from torch.utils.cpp_extension import load_inline
    major, minor = _arch()
    name = f"fso_s1c_gelu_ref_sm{major}{minor}{'' if fmad else '_nofma'}"
    build_dir = os.path.join(tempfile.gettempdir(), f"fso_test_dense_v1_{os.getuid()}", name)
    os.makedirs(build_dir, exist_ok=True)
    saved = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    try:
        ext = load_inline(name=name, cpp_sources=_GELU_REF_CPP, cuda_sources=_GELU_REF_CU, functions=["gelu_ref"],
                          extra_cuda_cflags=["-O3"] + ([] if fmad else ["-fmad=false"]), build_directory=build_dir,
                          verbose=False)
    finally:
        if saved is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = saved
    _GELU_EXTS[fmad] = ext
    return ext


def _s1c_route(M: int, N: int, K: int) -> dict:
    r = torch.ops.fish_scales_ops.mxfp8_dense_route_sm120(M, N, K)
    return {"tile": (r[0], r[1], r[2]), "min_blocks": r[3], "sched_group": r[4], "k_split": r[5],
            "smallm": bool(r[6]), "scratch": r[7]}


def _s1c_smallm_taken(M: int, N: int, K: int) -> bool:
    """Whether the small-M route sm120.smallm (Q120b) takes the 2b call of this cell in this process (rules, or the
    FSO_SMALLM_ROUTE / force-knob setting); the cascade's route query above does not know about it."""
    return bool(int(torch.ops.fish_scales_ops.mxfp8_dense_smallm_route_sm120(M, N, K, False)[0]))


def _s1c_acc_ksplit(M: int, N: int, K: int, route: dict) -> int:
    """The K split whose accumulator the identity call rounds: 1 when the small-M route takes the cell (its
    fixed-order fp32 sums of integers below 2^24 are exact in every configuration, and the bf16-x call is then G2 or
    quantize + G1, exact as well), else the cascade's k_split (a Stream-K route rounds its slices' partials)."""
    return 1 if _s1c_smallm_taken(M, N, K) else route["k_split"]


def _s1c_exact_operands(M: int, N: int, K: int, seed: int):
    """Integer activations (a +-4 in every 32-block) and integer weights in [-4, 4], on the CPU."""
    g = torch.Generator().manual_seed(seed)
    xi = torch.randint(-4, 5, (M, K), generator=g)
    blk = xi.view(M, K // 32, 32)
    pos = torch.randint(0, 32, (M, K // 32, 1), generator=g)
    sign = torch.randint(0, 2, (M, K // 32, 1), generator=g) * 2 - 1
    blk.scatter_(2, pos, 4 * sign)
    wi = torch.randint(-4, 5, (N, K), generator=g)
    return xi, wi


def _s1c_weight(wi: torch.Tensor):
    """The DenseWeight of an integer weight: e4m3 codes of the integers, UE8M0 byte 127 everywhere."""
    N, K = wi.shape
    wq = wi.to(torch.float32).to(torch.float8_e4m3fn).cuda()
    sc = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda")
    return D.prepare_weight(wq, format="mxfp8", scale=sc)


def _s1c_acc(xi: torch.Tensor, wi: torch.Tensor, k_split: int) -> torch.Tensor:
    """The accumulator the route rounds, fp32 on the GPU: the exact product (single launch), or the fp32 sum of the
    K slices' bf16-rounded exact partials, slices as launch_sm120_mxfp8_streamk_gemm cuts them (128-element chunks,
    ceil(chunks / k) per slice)."""
    K = xi.shape[1]
    xd, wd = xi.double(), wi.double()
    if k_split <= 1:
        return (xd @ wd.T).float().cuda()
    chunks = K // 128
    per = (chunks + k_split - 1) // k_split
    acc = torch.zeros(xi.shape[0], wi.shape[0], dtype=torch.float64)
    s = 0
    while s * per < chunks:
        k0, k1 = s * per * 128, min((s + 1) * per, chunks) * 128
        part = (xd[:, k0:k1] @ wd[:, k0:k1].T).float().to(torch.bfloat16)  # exact fp32, then the slice's rounding
        acc += part.double()
        s += 1
    return acc.float().cuda()  # integers < 2^24: the fp32 sum of the partials in any order


def _s1c_bias_term(b: torch.Tensor) -> torch.Tensor:
    f = b.float()
    return torch.where(f == 0, torch.full_like(f, -0.0), f)


def _s1c_expected(acc: torch.Tensor, bias, act, res) -> torch.Tensor:
    v = acc
    if bias is not None:
        v = v + _s1c_bias_term(bias)
    if act == "gelu_erf":
        v = _s1c_gelu_ext(True).gelu_ref(v.contiguous(), False)
    if res is not None:
        v = v + res.float()
    return v.to(torch.bfloat16)


def _s1c_digest(t: torch.Tensor) -> str:
    import hashlib
    return hashlib.sha256(t.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]


class _S1cWorker:
    def __init__(self):
        self.failures: list[str] = []
        self.digests: dict[str, str] = {}
        self.cells = 0
        self.double_differs = 0
        self.double_total = 0

    def check(self, ok: bool, msg: str) -> bool:
        if not ok:
            self.failures.append(msg)
        return ok

    def exact_cell(self, M: int, N: int, K: int, want_tile=None, want_split=None, label: str = "") -> None:
        """Every epilogue variant on one cell, both entry paths, against the exact reference."""
        route = _s1c_route(M, N, K)
        taken = _s1c_smallm_taken(M, N, K)
        tag = f"{label} M={M} N={N} K={K} route {route['tile']} k_split {route['k_split']}" \
            + (" -> sm120.smallm" if taken else "")
        if want_tile is not None:
            self.check(route["tile"] == tuple(want_tile), f"o {tag}: the route is not the forced tile {want_tile}")
            self.check(not taken, f"o {tag}: the small-M route took a forced-tile cell")
        if want_split is not None:
            ok = route["k_split"] > 1 if want_split == "split" else route["k_split"] == want_split
            self.check(ok, f"o {tag}: K split {route['k_split']}, expected {want_split}")
        seed = M * 1009 + N * 17 + K
        xi, wi = _s1c_exact_operands(M, N, K, seed)
        dw = _s1c_weight(wi)
        x = xi.to(torch.bfloat16).cuda()
        xq, sx = D.quantize_activation(x)
        words = sx.view(torch.int32)[:M] if sx.dim() == 2 else sx
        self.check(bool((words == 0x79797979).all()), f"o {tag}: the activation scale bytes are not all 121 "
                                                      f"(the exact construction failed)")
        g = torch.Generator().manual_seed(seed + 1)
        bias = (torch.randint(-64, 65, (N,), generator=g).float() / 8).to(torch.bfloat16).cuda()
        res = (torch.randint(-1000, 1001, (M, N), generator=g).float() / 8).to(torch.bfloat16).cuda()
        acc_exact = _s1c_acc(xi, wi, _s1c_acc_ksplit(M, N, K, route))
        y_id = D.linear(xq, dw, x_scale=sx)
        y_id_x = D.linear(x, dw)
        id_ok = torch.equal(y_id, acc_exact.to(torch.bfloat16)) and bytes_equal(y_id, y_id_x)
        self.check(id_ok, f"o {tag}: the identity call != RN_bf16(exact accumulator) or bf16-x != 2b")
        acc = torch.where(acc_exact == 0, y_id.float(), acc_exact)  # the route's signed zeros
        res_ok = True
        for vlabel, use_b, act, use_r in S1C_VARIANTS:
            b = bias if use_b else None
            r = res if use_r else None
            want = _s1c_expected(acc, b, act, r)
            got = D.linear(xq, dw, x_scale=sx, bias=b, activation=act, residual=r)
            got_x = D.linear(x, dw, bias=b, activation=act, residual=r)
            ok = bytes_equal(got, want) and bytes_equal(got_x, got)
            if not ok:
                nbad = int((got.view(torch.int16) != want.view(torch.int16)).sum())
                self.check(False, f"o {tag} {vlabel}: {nbad} of {got.numel()} elements differ from the exact "
                                  f"reference (bf16-x == 2b: {bytes_equal(got_x, got)})")
            res_ok &= ok
            self.digests[f"{label}|{M}x{N}x{K}|{vlabel}"] = _s1c_digest(got)
            if vlabel == "bias":
                double = (acc.to(torch.bfloat16).float() + _s1c_bias_term(bias)).to(torch.bfloat16)
                self.double_differs += int((double.view(torch.int16) != want.view(torch.int16)).sum())
                self.double_total += want.numel()
        self.cells += 1
        print(f"  o  {tag}: identity exact, {len(S1C_VARIANTS)} variants x (bf16 x, 2b) exact  "
              f"{_ok(id_ok and res_ok)}")

    def zero_corner(self, M: int, N: int, K: int, label: str = "") -> None:
        """bias = 0 is the identity: an all-zero x row against an all-negative weight column (the -0.0 corner),
        and bias = 0 == bias = None on random data."""
        route = _s1c_route(M, N, K)
        tag = f"{label} M={M} N={N} K={K} route {route['tile']} k_split {route['k_split']}"
        xi, wi = _s1c_exact_operands(M, N, K, M + N + K)
        xi[0].zero_()
        wi[0] = -(torch.arange(K) % 4 + 1)
        dw = _s1c_weight(wi)
        x = xi.to(torch.bfloat16).cuda()
        zero = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
        y0 = D.linear(x, dw)
        yb = D.linear(x, dw, bias=zero)
        corner = int(y0.view(torch.int16)[0, 0]) & 0xFFFF
        ok = bytes_equal(y0, yb)
        xr = make_x(M, K, seed=K + 3)
        dwr = prepared(12, N, K, seed=N + 5)
        ok_r = bytes_equal(D.linear(xr, dwr), D.linear(xr, dwr, bias=zero))
        self.check(ok and ok_r, f"o {tag}: bias = 0 differs from bias = None (corner byte {corner:#06x}; random "
                                f"data {ok_r})")
        self.digests[f"{label}|{M}x{N}x{K}|zero"] = _s1c_digest(yb)
        print(f"  o  {tag}: -0.0 corner byte (no bias) {corner:#06x}; bias = 0 == bias = None there and on random "
              f"data  {_ok(ok and ok_r)}")

    def producer_then_gemm(self, M: int, N: int, K: int) -> None:
        """bias and residual written by the kernels launched right before the GEMM (PDL edge): the GEMM reads
        the new values, not the buffers' previous NaNs."""
        route = _s1c_route(M, N, K)
        xi, wi = _s1c_exact_operands(M, N, K, 7 * M + K)
        dw = _s1c_weight(wi)
        xq, sx = D.quantize_activation(xi.to(torch.bfloat16).cuda())
        g = torch.Generator().manual_seed(M + N)
        bias = (torch.randint(-64, 65, (N,), generator=g).float() / 8).to(torch.bfloat16).cuda()
        res = (torch.randint(-1000, 1001, (M, N), generator=g).float() / 8).to(torch.bfloat16).cuda()
        y_id = D.linear(xq, dw, x_scale=sx)
        acc_exact = _s1c_acc(xi, wi, _s1c_acc_ksplit(M, N, K, route))
        want = _s1c_expected(torch.where(acc_exact == 0, y_id.float(), acc_exact), bias, "gelu_erf", res)
        bias_buf = torch.full_like(bias, float("nan"))
        res_buf = torch.full_like(res, float("nan"))
        ok = True
        for _ in range(20):
            torch.cuda.synchronize()
            bias_buf.fill_(float("nan"))
            res_buf.fill_(float("nan"))
            torch.mul(bias, 1, out=bias_buf)  # kernels (x * 1 is exact, -0.0 included), not memcpys, so that the
            torch.mul(res, 1, out=res_buf)    # second one is the GEMM's immediate predecessor (a PDL edge)
            got = D.linear(xq, dw, x_scale=sx, bias=bias_buf, activation="gelu_erf", residual=res_buf)
            ok &= bytes_equal(got, want)
        self.check(ok, f"o producer M={M} N={N} K={K}: the GEMM did not see the bias / residual its predecessors "
                       f"wrote")
        print(f"  o  producer M={M} N={N} K={K} route {route['tile']} k_split {route['k_split']}: bias and residual "
              f"written by the two kernels before the GEMM, 20 times: exact  {_ok(ok)}")


def _s1c_worker_exact(w: _S1cWorker, spec: dict) -> None:
    for cell in spec["cells"]:
        w.exact_cell(*cell["mnk"], want_tile=cell.get("tile"), want_split=cell.get("split"), label=spec["label"])
    for mnk in spec.get("zero", []):
        w.zero_corner(*mnk, label=spec["label"])
    for mnk in spec.get("producer", []):
        w.producer_then_gemm(*mnk)
    if w.double_total:
        frac = w.double_differs / w.double_total
        w.check(w.double_differs > 0, f"o {spec['label']}: RN_bf16(RN_bf16(acc) + b) never differs from "
                                      f"RN_bf16(acc + b); the cells do not discriminate a double rounding")
        print(f"  o  {spec['label']}: the double-rounding variant differs on {frac:.1%} of "
              f"{w.double_total} biased elements (the test discriminates it)")


def _s1c_worker_refuse(w: _S1cWorker, spec: dict) -> None:
    """A forced diagnostic instance: the identity call runs, and every epilogue field is refused with
    NotImplementedError naming the instance; on the 2b path before any device work. (A bf16 x is quantized before
    the C++ route, which only the C++ side knows under these knobs, refuses: one quantize kernel runs there.)"""
    M, N, K = spec["mnk"]
    route = _s1c_route(M, N, K)
    x = make_x(M, K, seed=11)
    dw = prepared(12, N, K, seed=12)
    xq, sx = D.quantize_activation(x)
    bias = torch.randn(N, device="cuda").bfloat16()
    y = D.linear(x, dw)
    w.check(bool(torch.isfinite(y).all()), f"o {spec['label']}: the identity call on the forced instance failed")
    for vlabel, fn, no_events in (("bias (2b)", lambda: D.linear(xq, dw, x_scale=sx, bias=bias), True),
                                  ("gelu (2b)", lambda: D.linear(xq, dw, x_scale=sx, activation="gelu_erf"), True),
                                  ("bias (bf16 x)", lambda: D.linear(x, dw, bias=bias), False)):
        exc, n = _device_events(fn)
        ok = isinstance(exc, NotImplementedError) and spec["word"] in str(exc) and (n == 0 or not no_events)
        w.check(ok, f"o {spec['label']} {vlabel}: expected NotImplementedError naming {spec['word']!r} with no "
                    f"device work, got {type(exc).__name__ if exc else None} ({n} events): {exc}")
        print(f"  o  {spec['label']} route {route}: {vlabel} -> {type(exc).__name__ if exc else 'no exception'}, "
              f"{n} device events  {_ok(ok)}")
        if exc is not None and vlabel == "bias (2b)":
            print(f"  o  message: {exc}")


def _s1c_capture(fn):
    """One eager call, capture on a non-default stream, replay into a NaN-filled output; (graph, out, eager)."""
    ref = fn().clone()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = fn()
    torch.cuda.synchronize()
    return g, out, ref


def _s1c_worker_graphs(w: _S1cWorker, spec: dict) -> None:
    calls = {}
    for name, (M, N, K), use_b, act, use_r in (
            ("fc1 bias+gelu", (78, 5120, 1280), True, "gelu_erf", False),
            ("out_proj bias+residual", (533, 1280, 1280), True, None, True),
            ("fc2 bias+residual", (1079, 1280, 5120), True, None, True),
            ("streamk16 bias+gelu+residual", (16, 2560, 9728), True, "gelu_erf", True),
            ("streamk20 bias", (20, 4096, 4096), True, None, False),
            ("streamk129 bias+residual", (129, 2560, 9728), True, None, True)):
        route = _s1c_route(M, N, K)
        x = make_x(M, K, seed=M + 1)
        dw = prepared(12, N, K, seed=N + 2)
        xq, sx = D.quantize_activation(x)
        b = torch.randn(N, device="cuda").bfloat16() if use_b else None
        r = torch.randn(M, N, device="cuda").bfloat16() if use_r else None
        for path, fn in (("bf16 x", lambda: D.linear(x, dw, bias=b, activation=act, residual=r)),
                         ("2b", lambda: D.linear(xq, dw, x_scale=sx, bias=b, activation=act, residual=r))):
            g, out, ref = _s1c_capture(fn)
            out.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            ok = bool(torch.isfinite(out).all()) and bytes_equal(out, ref)
            w.check(ok, f"o graph {name} ({path}): replay != eager")
            print(f"  o  graph {name} M={M} N={N} K={K} route {route['tile']} k_split {route['k_split']} ({path}): "
                  f"capture on a non-default stream, replay into NaN == eager  {_ok(ok)}")
            calls[(name, path)] = (g, out, ref, (x, dw, xq, sx, b, r))  # the graph reads these: keep them alive
    # Two different captured calls replayed concurrently from one thread, one per stream, 200 times.
    for a, b2 in ((("fc1 bias+gelu", "2b"), ("streamk16 bias+gelu+residual", "2b")),
                  (("streamk20 bias", "bf16 x"), ("streamk129 bias+residual", "2b")),
                  (("streamk16 bias+gelu+residual", "bf16 x"), ("streamk16 bias+gelu+residual", "2b"))):
        (g1, o1, r1, _), (g2, o2, r2, _) = calls[a], calls[b2]
        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
        bad1 = torch.zeros((), dtype=torch.int64, device="cuda")
        bad2 = torch.zeros((), dtype=torch.int64, device="cuda")
        torch.cuda.synchronize()
        for _ in range(200):
            with torch.cuda.stream(s1):
                g1.replay()
                bad1 += (o1.view(torch.int16) != r1.view(torch.int16)).sum()
            with torch.cuda.stream(s2):
                g2.replay()
                bad2 += (o2.view(torch.int16) != r2.view(torch.int16)).sum()
        torch.cuda.synchronize()
        ok = int(bad1) == 0 and int(bad2) == 0
        w.check(ok, f"o concurrent replay {a} || {b2}: {int(bad1)} / {int(bad2)} differing elements")
        print(f"  o  concurrent replay, two streams, one thread, 200 x: {a[0]} ({a[1]}) || {b2[0]} ({b2[1]}) == "
              f"serial  {_ok(ok)}")
    # Two concurrent Stream-K calls on two streams no longer share scratch: each capture allocates its own partial
    # buffer from the torch caching allocator on its own stream (memory history), and the scratch is a torch
    # allocation at all (peak allocated). Eager calls on two streams, interleaved without synchronization, stay exact.
    M, N, K = 16, 2560, 9728
    route = _s1c_route(M, N, K)
    want_bytes = (route["scratch"] + 511) // 512 * 512
    xa, xb = make_x(M, K, seed=31), make_x(M, K, seed=32)
    dw = prepared(12, N, K, seed=33)
    ba = torch.randn(N, device="cuda").bfloat16()
    ra = torch.randn(M, N, device="cuda").bfloat16()
    fa = lambda: D.linear(xa, dw, bias=ba, activation="gelu_erf")  # noqa: E731
    fb = lambda: D.linear(xb, dw, residual=ra)  # noqa: E731
    fa(), fb()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fa()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    ok_peak = route["k_split"] > 1 and peak >= route["scratch"] + M * N * 2
    w.check(ok_peak, f"o scratch: a Stream-K call's peak torch allocation {peak} B < scratch {route['scratch']} + out")
    print(f"  o  scratch: route {route['tile']} k_split {route['k_split']} needs {route['scratch']} B; the call's peak "
          f"torch allocation is {peak} B (scratch + output come from the caching allocator)  {_ok(ok_peak)}")
    torch.cuda.memory._record_memory_history(max_entries=200000)
    try:
        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
        ga, gb = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        s1.wait_stream(torch.cuda.current_stream())
        s2.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(ga, stream=s1):
            oa = fa()
        with torch.cuda.graph(gb, stream=s2):
            ob = fb()
        torch.cuda.synchronize()
        snap = torch.cuda.memory._snapshot()
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
    traces = snap.get("device_traces", [[]])
    ev = traces[torch.cuda.current_device()] if len(traces) > torch.cuda.current_device() else []
    allocs = [(e.get("stream"), e.get("addr")) for e in ev if e.get("action") == "alloc" and e.get("size") == want_bytes]
    by_stream = {s1.cuda_stream: [a for st, a in allocs if st == s1.cuda_stream],
                 s2.cuda_stream: [a for st, a in allocs if st == s2.cuda_stream]}
    ok_trace = all(len(v) >= 1 for v in by_stream.values()) and \
        not (set(by_stream[s1.cuda_stream]) & set(by_stream[s2.cuda_stream]))
    w.check(ok_trace, f"o scratch: captures on two streams did not each allocate their own {want_bytes} B scratch "
                      f"(allocations of that size by stream: {by_stream})")
    print(f"  o  scratch: the two captures allocate their {want_bytes} B partial buffers on their own streams at "
          f"different addresses: {[[hex(a) for a in v] for v in by_stream.values()]}  {_ok(ok_trace)}")
    ra_ref, rb_ref = fa().clone(), fb().clone()
    bad = torch.zeros(2, dtype=torch.int64, device="cuda")
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    for _ in range(200):
        with torch.cuda.stream(s1):
            ga.replay()
            bad[0] += (oa.view(torch.int16) != ra_ref.view(torch.int16)).sum()
        with torch.cuda.stream(s2):
            gb.replay()
            bad[1] += (ob.view(torch.int16) != rb_ref.view(torch.int16)).sum()
    torch.cuda.synchronize()
    ok_rep = int(bad.sum()) == 0
    w.check(ok_rep, f"o scratch: concurrent replay of the two Stream-K graphs differs ({bad.tolist()})")
    print(f"  o  scratch: the two Stream-K graphs replayed concurrently on their streams, 200 x == serial  "
          f"{_ok(ok_rep)}")
    bad.zero_()
    torch.cuda.synchronize()
    for _ in range(200):
        with torch.cuda.stream(s1):
            ya = fa()
            bad[0] += (ya.view(torch.int16) != ra_ref.view(torch.int16)).sum()
        with torch.cuda.stream(s2):
            yb = fb()
            bad[1] += (yb.view(torch.int16) != rb_ref.view(torch.int16)).sum()
    torch.cuda.synchronize()
    ok_eager = int(bad.sum()) == 0
    w.check(ok_eager, f"o scratch: eager Stream-K calls on two streams differ ({bad.tolist()})")
    print(f"  o  scratch: eager Stream-K calls on two streams, interleaved without synchronization, 200 x == serial  "
          f"{_ok(ok_eager)}")


def _s1c_worker_compile(w: _S1cWorker, spec: dict) -> None:
    N, K = 5120, 1280
    dw = prepared(12, N, K, seed=71)
    bias = torch.randn(N, device="cuda").bfloat16()

    def f_x(x, r):
        return D.linear(x, dw, bias=bias, activation="gelu_erf", residual=r)

    def f_2b(x, r):
        xq, sx = D.quantize_activation(x)
        return D.linear(xq, dw, x_scale=sx, bias=bias, activation="gelu_erf", residual=r)

    for fname, f, want_nodes in (("bf16 x", f_x, {"dense_linear": 1}),
                                 ("2b", f_2b, {"dense_quantize_activation": 1, "dense_linear": 1})):
        ins = {M: (make_x(M, K, seed=72 + M), torch.randn(M, N, device="cuda").bfloat16()) for M in (78, 1079, 7)}
        want = {M: f(*a) for M, a in ins.items()}
        graphs = []

        def record(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        torch._dynamo.reset()
        rec = torch.compile(f, backend=record, fullgraph=True)
        rec_ok = all(bytes_equal(rec(*a), want[M]) for M, a in ins.items())
        per_graph = [[str(n.target) for n in g.graph.nodes if n.op == "call_function" and "fish_scales_ops" in
                      str(n.target)] for g in graphs]
        node_ok = bool(graphs) and all(len(ours) == sum(want_nodes.values()) and all(
            sum(k in t and (k != "dense_linear" or "quantized" not in t) for t in ours) == v
            for k, v in want_nodes.items()) for ours in per_graph)
        w.check(rec_ok and node_ok, f"o compile ({fname}): recording backend nodes {per_graph}, equal {rec_ok}")
        print(f"  o  torch.compile ({fname}, bias + gelu_erf + residual), recording backend, fullgraph: "
              f"{len(graphs)} graph(s), nodes {per_graph[0] if per_graph else []}; == eager  "
              f"{_ok(rec_ok and node_ok)}")
        torch._dynamo.reset()
        ind = torch.compile(f, fullgraph=True)
        ok = all(bytes_equal(ind(*a), want[M]) for M, a in ins.items())
        torch.cuda.synchronize()
        w.check(ok, f"o compile ({fname}): inductor != eager")
        print(f"  o  torch.compile ({fname}), inductor, fullgraph, M = 78, 1079, 7: == eager  {_ok(ok)}")
        torch._dynamo.reset()


def _s1c_worker_gelu_ref(w: _S1cWorker, spec: dict) -> None:
    """The reference GELU's two spellings agree: the _rn form under the library's flags and the plain expression
    with no fma contraction (-fmad=false), on every fp32 value the exact cells produce and a random sweep."""
    v = torch.cat([torch.arange(-300000, 300001, dtype=torch.float32) / 8,
                   torch.randn(1 << 20) * 8, torch.tensor([0.0, -0.0, 1e-30, -1e-30, 10.0, -10.0])]).cuda()
    a = _s1c_gelu_ext(True).gelu_ref(v, False)
    b = _s1c_gelu_ext(False).gelu_ref(v, True)
    c = _s1c_gelu_ext(True).gelu_ref(v, True)
    ok = torch.equal(a.view(torch.int32), b.view(torch.int32))
    n_c = int((a.view(torch.int32) != c.view(torch.int32)).sum())
    w.check(ok, "o reference GELU: the _rn spelling differs from the uncontracted plain expression")
    print(f"  o  reference GELU on {v.numel()} fp32 values: _rn spelling == plain expression built with -fmad=false "
          f"{_ok(ok)}; the plain expression built with fma contraction differs on {n_c} values (why the library "
          f"spells the operations with _rn intrinsics)")


def _s1c_worker(spec_json: str) -> int:
    import json
    spec = json.loads(spec_json)
    w = _S1cWorker()
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        {"exact": _s1c_worker_exact, "refuse": _s1c_worker_refuse, "graphs": _s1c_worker_graphs,
         "compile": _s1c_worker_compile, "gelu_ref": _s1c_worker_gelu_ref}[spec["kind"]](w, spec)
    except Exception as e:  # noqa: BLE001 - reported to the parent as a failure with its reason
        import traceback
        traceback.print_exc()
        w.failures.append(f"o {spec.get('label', spec['kind'])}: {type(e).__name__}: {e}")
    print("S1C_RESULT " + json.dumps({"failures": w.failures, "digests": w.digests, "cells": w.cells}))
    return 0


def _s1c_specs() -> list:
    """(label, env, spec) of every worker of case o."""
    grid = [{"mnk": (M, 1280, K)} for K in S1C_GRID_KS for M in S1C_GRID_MS] + [{"mnk": (129, 5120, 1280)}]
    natural = ([{"mnk": (M, N, K)} for N, K in S1C_TOWER for M in S1C_TOWER_MS]
               + [{"mnk": (M, N, K)} for N, K in S1C_LLM for M in S1C_LLM_MS]
               + grid
               + [{"mnk": mnk, "split": "split"} for mnk in S1C_STREAMK_NATURAL])
    zero = [(7, 1280, 1280), (1079, 1280, 5120), (16, 2560, 9728), (20, 4096, 4096)]
    producer = [(78, 5120, 1280), (16, 2560, 9728)]
    specs = [("natural", {}, {"kind": "exact", "label": "natural", "cells": natural, "zero": zero,
                              "producer": producer}),
             ("natural, FSO_DISABLE_PDL=1", {"FSO_DISABLE_PDL": "1"},
              {"kind": "exact", "label": "natural", "cells": natural, "zero": zero, "producer": producer})]
    for t in S1C_SINGLE_TILES:
        lab = f"tile {t}"
        cells = [dict(c, tile=t, split=1) for c in grid]
        specs.append((lab, {"FSO_FORCE_TILE": ",".join(map(str, t))},
                      {"kind": "exact", "label": lab, "cells": cells, "zero": [(7, 1280, 1280)]}))
    for t, ks in S1C_SK_TILES:
        lab = f"tile {t} ksplit {ks}"
        cells = [dict(c, tile=t, split="split") for c in grid]
        specs.append((lab, {"FSO_FORCE_TILE": ",".join(map(str, t)), "FSO_FORCE_KSPLIT": str(ks)},
                      {"kind": "exact", "label": lab, "cells": cells, "zero": [(7, 1280, 5120)]}))
    for t, ks in (((128, 128, 2), 0), ((32, 64, 4), 4)):
        lab = f"tile {t}" + (f" ksplit {ks}" if ks else "")
        env = {"FSO_FORCE_TILE": ",".join(map(str, t)), "FSO_DISABLE_PDL": "1"}
        if ks:
            env["FSO_FORCE_KSPLIT"] = str(ks)
        cells = [dict(c, tile=t) for c in grid]
        specs.append((lab + ", FSO_DISABLE_PDL=1", env, {"kind": "exact", "label": lab, "cells": cells,
                                                        "zero": [(7, 1280, 5120 if ks else 1280)]}))
    specs += [("SmallM", {"FSO_FORCE_TILE": "16,128,4", "FSO_FORCE_SMALLM": "1"},
               {"kind": "refuse", "label": "SmallM", "mnk": (4, 2048, 2048), "word": "SmallM"}),
              ("MinBlocksPerSm=2", {"FSO_FORCE_TILE": "16,128,4", "FSO_FORCE_MIN_BLOCKS": "2"},
               {"kind": "refuse", "label": "MinBlocksPerSm=2", "mnk": (4, 2048, 2048), "word": "MinBlocksPerSm=2"}),
              ("SchedGroup=8", {"FSO_FORCE_TILE": "16,64,4", "FSO_FORCE_SCHED_GROUP": "8"},
               {"kind": "refuse", "label": "SchedGroup=8", "mnk": (4, 2048, 2048), "word": "FSO_FORCE_SCHED_GROUP"}),
              ("graphs and streams", {}, {"kind": "graphs", "label": "graphs"}),
              ("torch.compile", {}, {"kind": "compile", "label": "compile"}),
              ("reference GELU", {}, {"kind": "gelu_ref", "label": "gelu_ref"})]
    return specs


def case_o(major: int) -> None:
    import json
    import subprocess
    if major != 12:
        print(f"  o  SKIP: the sm_120/sm_121 dense epilogue cases need sm_120/sm_121; this is {_arch_label()}")
        return
    here = os.path.abspath(__file__)
    digests = {}
    for label, env_add, spec in _s1c_specs():
        env = {k: v for k, v in os.environ.items() if k not in S1C_KNOBS}
        env.update(env_add)
        r = subprocess.run([sys.executable, here, "--s1c-worker", json.dumps(spec)], env=env, capture_output=True,
                           text=True)
        result = None
        for line in r.stdout.splitlines():
            if line.startswith("S1C_RESULT "):
                result = json.loads(line[len("S1C_RESULT "):])
            else:
                print(line)
        if result is None or r.returncode != 0:
            _check(False, f"o [{label}]: the worker failed (rc {r.returncode}): {r.stderr[-2000:]}")
            print(f"  o  [{label}] worker FAILED rc {r.returncode}\n{r.stderr[-2000:]}")
            continue
        failures.extend(f"[{label}] {f}" for f in result["failures"])
        digests[label] = result["digests"]
        print(f"  o  [{label}] ({', '.join(f'{k}={v}' for k, v in env_add.items()) or 'no knobs'}): "
              f"{result['cells']} exact cells, {len(result['failures'])} failures  {_ok(not result['failures'])}")
    # FSO_DISABLE_PDL=1 gives the bytes of the default, cell for cell.
    for on, off in (("natural", "natural, FSO_DISABLE_PDL=1"),
                    ("tile (128, 128, 2)", "tile (128, 128, 2), FSO_DISABLE_PDL=1"),
                    ("tile (32, 64, 4) ksplit 4", "tile (32, 64, 4) ksplit 4, FSO_DISABLE_PDL=1")):
        a, b = digests.get(on), digests.get(off)
        common = sorted(set(a or {}) & set(b or {}))
        ok = bool(common) and all(a[k] == b[k] for k in common)
        _check(ok, f"o PDL: [{on}] and [{off}] differ on {sum(a[k] != b[k] for k in common) if common else 'all'} "
                   f"of {len(common)} outputs")
        print(f"  o  PDL on vs off: [{on}] == [{off}] on {len(common)} outputs  {_ok(ok)}")


# --- (w) the sm_120/sm_121 small-M route sm120.smallm (Q120b) ------------------------------------------------------
#
# The decode-band route of issue #12 decision 3 (arch/sm120/mxfp8/smallm_tc.cuh, rules in smallm_route.h): the 2b
# call runs G1 where R1 picks it, the bf16-x call runs G2 (one launch, the activation quantized inside the kernel)
# where R2 picks it, the cascade elsewhere. Every worker is a subprocess (FSO_SMALLM_ROUTE and the force knobs are
# read once per process). Per cell: the host route query, the rules op and dense._sm120_route agree; the kernel
# names show the small-M kernel exactly where the plan says, in the instantiation (form, nc, rt, wk, epilogue) the
# plan names, and the bf16-x call is ONE launch where R2 picks it; the identity call and every epilogue variant
# equal the exact reference on both entries; the bf16-x call equals the 2b call byte for byte (G2 == quantize + G1).
# The exact construction is case o's; on the small-M route the accumulator is the exact product whatever the
# configuration (fixed-order fp32 sums of integers below 2^24), on the cascade it is case o's (Stream-K: the
# slices' bf16 partials).

W_SHAPES = {  # tag: (N, K), the twelve Q120a decode shapes the rules were fitted on
    "llm_qkv_tp2": (2560, 2048), "llm_o_tp2": (2048, 2048), "llm_qkv": (5120, 2048), "llm_o": (2048, 4096),
    "A_wqkv": (6144, 2560), "A_wo": (2560, 4096), "A_gate_up": (19456, 2560), "A_down": (2560, 9728),
    "C_wqkv_gated": (9216, 2048), "C_gdn_in_proj": (12288, 2048), "C_shared_gate_up": (1024, 2048),
    "C_shared_down": (2048, 512),
}
W_MS = (1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64)
W_HELD_OUT = ((1280, 1280), (1280, 3072), (4096, 5120), (2560, 7680), (1280, 7680))   # (N, K) outside the fit
# every M 1..64 under FSO_SMALLM_ROUTE=1. Q120c (the resident kernel's unchecked steady ring groups and its edge
# groups): K = 1920, 3968 and 6016 have nb = 60, 124, 188 K blocks, not a multiple of the 8 warps, so the last block
# index of half the warps is past nb. Resident plans with 8 warps and one tile: (1280, 1920) at M <= 32 runs one edge
# group per tile group; (2560, 3968) at M <= 16 two edge groups (the last full group issues a successor with blocks
# past nb); (4096, 6016) at M <= 8 one steady group then two edge groups, with two tile groups per CTA (the last edge
# group prefetches the next tile group). Above those M the plans are chunked (8 warps, cj 5 .. 24: the chunk
# countdown), with two tile groups per CTA at N = 4096.
W_ON_SHAPES = ((2048, 512), (1024, 2048), (6144, 2560), (2560, 9728), (1280, 1920), (2560, 3968), (4096, 6016))
W_KNOBS = S1C_KNOBS + ("FSO_SMALLM_ROUTE",)
_W_KERNEL_RE = None


def _sms() -> int:
    """The current device's SM count: the small-M rules' one-round plan guard depends on it (sm120_smallm_rules)."""
    return int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)


def _w_route(M: int, N: int, K: int, bf16x: bool) -> dict:
    r = [int(v) for v in torch.ops.fish_scales_ops.mxfp8_dense_smallm_route_sm120(M, N, K, bf16x)]
    return {"use": bool(r[0]), "nc": r[1], "rt": r[2], "wk": r[3], "cj": r[4], "smem": r[5], "launch_smem": r[6],
            "grid": r[7], "r1": bool(r[8]), "r2": bool(r[9])}


def _w_smallm_kernels(names) -> list:
    """(form, nc, rt, wk, epi) of every small-M kernel among the launched kernel names."""
    global _W_KERNEL_RE
    import re
    if _W_KERNEL_RE is None:
        # (form, nc, rt, wk, epi[, chunked]): the sixth argument (Q120c, the resident / chunked instantiation) is
        # not part of the plan the test checks
        _W_KERNEL_RE = re.compile(r"sm120_smallm_kernel<(\d+), (\d+), (\d+), (\d+), (true|false)(?:, (?:true|false))?>")
    out = []
    for n in names:
        m = _W_KERNEL_RE.search(n)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5) == "true"))
    return out


class _WWorker(_S1cWorker):
    def route_cell(self, M: int, N: int, K: int, label: str, expect_use=None) -> None:
        r2b, rx = _w_route(M, N, K, False), _w_route(M, N, K, True)
        rules = [int(v) for v in torch.ops.fish_scales_ops.sm120_smallm_rules(M, N, K, _sms())]
        tag = f"{label} M={M} N={N} K={K}"
        self.check(rules[0] == int(r2b["use"]) and rules[1] == int(rx["use"]),
                   f"w {tag}: sm120_smallm_rules {rules[:2]} != the route query {r2b['use']}, {rx['use']}")
        self.check(not rx["use"] or r2b["use"], f"w {tag}: the bf16-x call takes the route but the 2b call does not")
        self.check((r2b["nc"], r2b["rt"], r2b["wk"], r2b["cj"]) == (rx["nc"], rx["rt"], rx["wk"], rx["cj"]),
                   f"w {tag}: the two forms get different configurations")
        for bf16x, r in ((False, r2b), (True, rx)):
            key = D._sm120_route(M, N, K, bf16x)
            self.check(key == ("sm120.smallm" if r["use"] else "sm120.mxfp8"),
                       f"w {tag}: dense._sm120_route(bf16x={bf16x}) = {key}, the plan says use = {r['use']}")
        if expect_use is not None:
            self.check(r2b["use"] == expect_use and rx["use"] == expect_use,
                       f"w {tag}: expected use = {expect_use} on both entries, got {r2b['use']} / {rx['use']}")
        cascade = _s1c_route(M, N, K)
        cfg = f"nc {r2b['nc']} rt {r2b['rt']} wk {r2b['wk']} cj {r2b['cj']} grid {r2b['grid']}"
        route_txt = (f"smallm[{cfg}]" if r2b["use"] else f"cascade {cascade['tile']} k_split {cascade['k_split']}") \
            + (" / bf16-x " + ("smallm" if rx["use"] else "quantize + " + ("smallm" if r2b["use"] else "cascade")))
        seed = M * 1009 + N * 17 + K
        xi, wi = _s1c_exact_operands(M, N, K, seed)
        dw = _s1c_weight(wi)
        x = xi.to(torch.bfloat16).cuda()
        xq, sx = D.quantize_activation(x)
        words = sx.view(torch.int32)[:M] if sx.dim() == 2 else sx
        self.check(bool((words == 0x79797979).all()), f"w {tag}: the activation scale bytes are not all 121")
        # the identity calls: which kernels run, and the bytes
        y_id, names_2b = _kernels_of(lambda: D.linear(xq, dw, x_scale=sx))
        y_id_x, names_x = _kernels_of(lambda: D.linear(x, dw))
        sm_2b, sm_x = _w_smallm_kernels(names_2b), _w_smallm_kernels(names_x)
        self.check(bool(sm_2b) == r2b["use"] and len(sm_2b) <= 1,
                   f"w {tag}: the 2b call launched {len(sm_2b)} small-M kernels, the plan says use = {r2b['use']}")
        # The bf16-x call is G2 (one launch) where R2 picks it; elsewhere it is the activation quantize followed by
        # the 2b route (dense.md, smallm_route.h), that is quantize + G1 where R1 picks it (one small-M kernel after
        # the quantize kernel) and quantize + the cascade otherwise.
        self.check(bool(sm_x) == (rx["use"] or r2b["use"]) and len(sm_x) <= 1,
                   f"w {tag}: the bf16-x call launched {len(sm_x)} small-M kernels, the plan says G2 = {rx['use']}, "
                   f"G1 (2b) = {r2b['use']}")
        if r2b["use"]:
            want = (0, r2b["nc"], r2b["rt"], r2b["wk"], False)
            self.check(sm_2b == [want], f"w {tag}: the 2b call ran {sm_2b}, the plan names {want}")
            self.check(len(names_2b) == 1, f"w {tag}: the 2b call launched {len(names_2b)} kernels: {names_2b}")
        if rx["use"]:
            want = (1, rx["nc"], rx["rt"], rx["wk"], False)
            self.check(sm_x == [want], f"w {tag}: the bf16-x call ran {sm_x}, the plan names {want}")
            self.check(len(names_x) == 1, f"w {tag}: the bf16-x call is not one launch: {names_x}")
        else:
            self.check(len(names_x) >= 2, f"w {tag}: the bf16-x call off the route launched {names_x}")
            if r2b["use"]:
                want = (0, r2b["nc"], r2b["rt"], r2b["wk"], False)
                self.check(sm_x == [want], f"w {tag}: the bf16-x call off G2 ran {sm_x}; quantize + G1 expects {want}")
        acc_2b = _s1c_acc(xi, wi, 1 if r2b["use"] else cascade["k_split"])
        id_ok = torch.equal(y_id, acc_2b.to(torch.bfloat16))
        self.check(id_ok, f"w {tag}: the 2b identity call != RN_bf16(exact accumulator)")
        self.check(bytes_equal(y_id, y_id_x), f"w {tag}: bf16-x identity != 2b identity (G2 != quantize + G1)")
        acc = torch.where(acc_2b == 0, y_id.float(), acc_2b)
        g = torch.Generator().manual_seed(seed + 1)
        bias = (torch.randint(-64, 65, (N,), generator=g).float() / 8).to(torch.bfloat16).cuda()
        res = (torch.randint(-1000, 1001, (M, N), generator=g).float() / 8).to(torch.bfloat16).cuda()
        res_ok = True
        for vlabel, use_b, act, use_r in S1C_VARIANTS:
            b = bias if use_b else None
            r = res if use_r else None
            want = _s1c_expected(acc, b, act, r)
            got, n2 = _kernels_of(lambda: D.linear(xq, dw, x_scale=sx, bias=b, activation=act, residual=r))
            got_x, nx = _kernels_of(lambda: D.linear(x, dw, bias=b, activation=act, residual=r))
            ok = bytes_equal(got, want) and bytes_equal(got_x, got)
            if r2b["use"]:
                ok &= self.check(_w_smallm_kernels(n2) == [(0, r2b["nc"], r2b["rt"], r2b["wk"], True)],
                                 f"w {tag} {vlabel}: the 2b call did not run the epilogue instantiation: {n2}")
            if rx["use"]:
                ok &= self.check(_w_smallm_kernels(nx) == [(1, rx["nc"], rx["rt"], rx["wk"], True)] and len(nx) == 1,
                                 f"w {tag} {vlabel}: the bf16-x call is not one epilogue launch: {nx}")
            elif r2b["use"]:
                ok &= self.check(_w_smallm_kernels(nx) == [(0, r2b["nc"], r2b["rt"], r2b["wk"], True)] and len(nx) >= 2,
                                 f"w {tag} {vlabel}: the bf16-x call off G2 is not quantize + the G1 epilogue "
                                 f"instantiation: {nx}")
            if not (bytes_equal(got, want) and bytes_equal(got_x, got)):
                nbad = int((got.view(torch.int16) != want.view(torch.int16)).sum())
                self.check(False, f"w {tag} {vlabel}: {nbad} of {got.numel()} elements differ from the exact "
                                  f"reference (bf16-x == 2b: {bytes_equal(got_x, got)})")
            res_ok &= ok
            self.digests[f"{label}|{M}x{N}x{K}|{vlabel}"] = _s1c_digest(got)
            if vlabel == "bias":
                double = (acc.to(torch.bfloat16).float() + _s1c_bias_term(bias)).to(torch.bfloat16)
                self.double_differs += int((double.view(torch.int16) != want.view(torch.int16)).sum())
                self.double_total += want.numel()
        self.digests[f"{label}|{M}x{N}x{K}|identity"] = _s1c_digest(y_id)
        self.cells += 1
        print(f"  w  {tag}: {route_txt}: identity exact, {len(S1C_VARIANTS)} variants x (bf16 x, 2b) exact  "
              f"{_ok(id_ok and res_ok)}")

    def quantized_on_cascade(self, M: int, N: int, K: int) -> None:
        """linear_quantized on a cell whose bf16 calls the small-M route takes: the small-M route does not write the
        quantized output, so the call runs on the MXFP8 cascade (S3c), from both entries, with no small-M kernel among
        its launches, and the two entries give the same bytes. Its bytes against the bf16 call on each route are case
        s's (the codes that differ from quantize_activation(linear(...)) on the small-M route are counted there)."""
        x = make_x(M, K, seed=5)
        dw = prepared(12, N, K, seed=6)
        xq, sx = D.quantize_activation(x)
        b = torch.randn(N, device="cuda").bfloat16()
        self.check(_w_route(M, N, K, False)["use"], f"w linear_quantized M={M} N={N} K={K}: not a small-M cell")
        outs = {}
        for lab, fn in (("2b", lambda: D.linear_quantized(xq, dw, x_scale=sx, bias=b, activation="gelu_erf")),
                        ("bf16 x", lambda: D.linear_quantized(x, dw, bias=b, activation="gelu_erf"))):
            fn()  # eager once
            (q, sc), names = _kernels_of(fn)
            small = _w_smallm_kernels(names)
            ok = bool(names) and not small
            self.check(ok, f"w linear_quantized M={M} N={N} K={K} ({lab}): launched {names}")
            outs[lab] = (q, sc)
            print(f"  w  linear_quantized (bias + gelu_erf) on a small-M cell M={M} N={N} K={K} ({lab}): served on "
                  f"the cascade, {len(names)} kernel(s), no small-M kernel  {_ok(ok)}")
        same = bytes_equal(outs["2b"][0], outs["bf16 x"][0]) and bytes_equal(outs["2b"][1], outs["bf16 x"][1])
        self.check(same, f"w linear_quantized M={M} N={N} K={K}: the bf16-x entry differs from the 2b entry")
        print(f"  w  linear_quantized M={M} N={N} K={K}: bf16-x entry == 2b entry (codes and scale words)  "
              f"{_ok(same)}")


def _w_worker_exact(w: _WWorker, spec: dict) -> None:
    for cell in spec["cells"]:
        w.route_cell(*cell["mnk"], label=spec["label"], expect_use=spec.get("expect_use"))
    for mnk in spec.get("zero", []):
        w.zero_corner(*mnk, label=spec["label"])
    for mnk in spec.get("quantized", []):
        w.quantized_on_cascade(*mnk)
    if w.double_total:
        w.check(w.double_differs > 0, f"w {spec['label']}: RN_bf16(RN_bf16(acc) + b) never differs from "
                                      f"RN_bf16(acc + b); the cells do not discriminate a double rounding")
        print(f"  w  {spec['label']}: the double-rounding variant differs on "
              f"{w.double_differs / w.double_total:.1%} of {w.double_total} biased elements")


def _w_worker_graphs(w: _WWorker, spec: dict) -> None:
    calls = {}
    for name, (M, N, K), use_b, act, use_r in (
            ("qkv_tp2 M=4 bias+gelu", (4, 2560, 2048), True, "gelu_erf", False),
            ("o M=8 bias+residual", (8, 2048, 4096), True, None, True),
            ("o_tp2 M=24 bias+gelu+residual", (24, 2048, 2048), True, "gelu_erf", True),
            ("down M=1 residual", (1, 2560, 9728), False, None, True),
            ("shared_down M=64 bias", (64, 2048, 512), True, None, False),
            ("o M=16 bias", (16, 2048, 4096), True, None, False),
            ("shared_gate_up M=32 bias (2b route only)", (32, 1024, 2048), True, None, False)):
        r2b, rx = _w_route(M, N, K, False), _w_route(M, N, K, True)
        x = make_x(M, K, seed=M + 1)
        dw = prepared(12, N, K, seed=N + 2)
        xq, sx = D.quantize_activation(x)
        b = torch.randn(N, device="cuda").bfloat16() if use_b else None
        r = torch.randn(M, N, device="cuda").bfloat16() if use_r else None
        for path, fn, used in (("bf16 x", lambda: D.linear(x, dw, bias=b, activation=act, residual=r), rx["use"]),
                               ("2b", lambda: D.linear(xq, dw, x_scale=sx, bias=b, activation=act, residual=r),
                                r2b["use"])):
            g, out, ref = _s1c_capture(fn)
            out.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            ok = bool(torch.isfinite(out).all()) and bytes_equal(out, ref)
            w.check(ok, f"w graph {name} ({path}): replay != eager")
            print(f"  w  graph {name} M={M} N={N} K={K} ({path}, {'smallm' if used else 'cascade'}): capture on a "
                  f"non-default stream, replay into NaN == eager  {_ok(ok)}")
            calls[(name, path)] = (g, out, ref, (x, dw, xq, sx, b, r))
    for a, b2 in ((("qkv_tp2 M=4 bias+gelu", "bf16 x"), ("o M=8 bias+residual", "2b")),
                  (("down M=1 residual", "bf16 x"), ("shared_down M=64 bias", "bf16 x")),
                  (("o_tp2 M=24 bias+gelu+residual", "bf16 x"), ("o_tp2 M=24 bias+gelu+residual", "2b"))):
        (g1, o1, r1, _), (g2, o2, r2, _) = calls[a], calls[b2]
        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
        bad1 = torch.zeros((), dtype=torch.int64, device="cuda")
        bad2 = torch.zeros((), dtype=torch.int64, device="cuda")
        torch.cuda.synchronize()
        for _ in range(200):
            with torch.cuda.stream(s1):
                g1.replay()
                bad1 += (o1.view(torch.int16) != r1.view(torch.int16)).sum()
            with torch.cuda.stream(s2):
                g2.replay()
                bad2 += (o2.view(torch.int16) != r2.view(torch.int16)).sum()
        torch.cuda.synchronize()
        ok = int(bad1) == 0 and int(bad2) == 0
        w.check(ok, f"w concurrent replay {a} || {b2}: {int(bad1)} / {int(bad2)} differing elements")
        print(f"  w  concurrent replay, two streams, one thread, 200 x: {a[0]} ({a[1]}) || {b2[0]} ({b2[1]}) == "
              f"serial  {_ok(ok)}")
    # Eager calls on two streams, interleaved without synchronization (no workspace: nothing to share).
    (ga, oa, ra_ref, (xa, dwa, _, _, ba, _)) = calls[("qkv_tp2 M=4 bias+gelu", "bf16 x")]
    (gb, ob, rb_ref, (xb, dwb, _, _, bb, rb)) = calls[("o M=8 bias+residual", "bf16 x")]
    bad = torch.zeros(2, dtype=torch.int64, device="cuda")
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    for _ in range(200):
        with torch.cuda.stream(s1):
            ya = D.linear(xa, dwa, bias=ba, activation="gelu_erf")
            bad[0] += (ya.view(torch.int16) != ra_ref.view(torch.int16)).sum()
        with torch.cuda.stream(s2):
            yb = D.linear(xb, dwb, bias=bb, residual=rb)
            bad[1] += (yb.view(torch.int16) != rb_ref.view(torch.int16)).sum()
    torch.cuda.synchronize()
    ok_eager = int(bad.sum()) == 0
    w.check(ok_eager, f"w eager two streams: {bad.tolist()} differing elements")
    print(f"  w  eager calls on two streams, interleaved, 200 x: == serial  {_ok(ok_eager)}")


def _w_worker_pdl(w: _WWorker, spec: dict) -> None:
    """A producer that releases its dependents at once and writes its outputs 2 ms later, directly ahead of the
    call: the activation (bf16 x of G2; xq and sx of G1), then the bias and the residual. The call gives the bytes of
    the call on the final tensors (every read of x / sx / bias / residual is after griddepcontrol.wait; the weight
    ring before it reads the weight only). Control: a dependent reading before its wait sees stale values."""
    if os.environ.get("FSO_DISABLE_PDL") == "1":
        print("  w  FSO_DISABLE_PDL=1: no programmatic launches; skipped")
        return
    ext = _pdl_probe()
    spin = 2_000_000
    t0, t1 = torch.ones(256, dtype=torch.int16, device="cuda"), torch.zeros(256, dtype=torch.int16, device="cuda")
    ext.slow_copy2(t0, t1, t0, t1, 1, 1000)
    ext.early_read(t1, torch.zeros(1, dtype=torch.int32, device="cuda"))
    torch.cuda.synchronize()
    for M, N, K in ((4, 2560, 2048), (8, 2048, 4096), (24, 2048, 2048), (1, 1280, 7680), (40, 2048, 512)):
        r2b, rx = _w_route(M, N, K, False), _w_route(M, N, K, True)
        w.check(r2b["use"] and rx["use"], f"w pdl M={M} N={N} K={K}: not a small-M cell on both entries")
        dw = prepared(12, N, K, seed=201 + N)
        x = (make_x(M, K, seed=202 + M) + 0.25).contiguous()  # no zero element: the control counts zeros
        xq, sx = D.quantize_activation(x)
        b = (torch.randn(N, device="cuda") * 0.1 + 0.5).bfloat16()
        r = (torch.randn(M, N, device="cuda") + 4.0).bfloat16()
        want_x = D.linear(x, dw, bias=b, residual=r, activation="gelu_erf")
        want_2b = D.linear(xq, dw, x_scale=sx, bias=b, residual=r, activation="gelu_erf")
        dx, dxq, dsx = torch.zeros_like(x), torch.zeros_like(xq), torch.zeros_like(sx)  # dsx keeps sx's strides
        db, dr = torch.zeros_like(b), torch.zeros_like(r)
        flat16 = lambda t: t.as_strided((t.numel(),), (1,)).view(torch.int16)  # noqa: E731  (K-major words, flat)
        rounds = (
            ("bf16-x, x + residual late", lambda: D.linear(dx, dw, bias=b, residual=dr, activation="gelu_erf"),
             (x.view(torch.int16), dx.view(torch.int16), r.view(torch.int16), dr.view(torch.int16)), want_x),
            ("bf16-x, bias + residual late", lambda: D.linear(x, dw, bias=db, residual=dr, activation="gelu_erf"),
             (b.view(torch.int16), db.view(torch.int16), r.view(torch.int16), dr.view(torch.int16)), want_x),
            ("2b, xq + sx late", lambda: D.linear(dxq, dw, x_scale=dsx, bias=b, residual=r, activation="gelu_erf"),
             (xq.view(torch.int16), dxq.view(torch.int16), flat16(sx), flat16(dsx)), want_2b),
            ("2b, bias + residual late", lambda: D.linear(xq, dw, x_scale=sx, bias=db, residual=dr,
                                                          activation="gelu_erf"),
             (b.view(torch.int16), db.view(torch.int16), r.view(torch.int16), dr.view(torch.int16)), want_2b),
        )
        for label, call, (s1, d1, s2, d2), want in rounds:
            d1.zero_(), d2.zero_()
            call()  # warm with the destination tensors
            stale = torch.zeros(1, dtype=torch.int32, device="cuda")
            d1.zero_(), d2.zero_()
            torch.cuda.synchronize()
            ext.slow_copy2(s1, d1, s2, d2, 8, spin)
            ext.early_read(d1, stale)
            torch.cuda.synchronize()
            n_stale = int(stale.item())
            d1.zero_(), d2.zero_()
            torch.cuda.synchronize()
            ext.slow_copy2(s1, d1, s2, d2, 8, spin)
            got = call()
            torch.cuda.synchronize()
            ok = bytes_equal(got, want)
            w.check(ok, f"w pdl M={M} N={N} K={K} {label}: the call behind a late producer != the final tensors")
            w.check(n_stale > 0, f"w pdl M={M} N={N} K={K} {label}: the control saw no stale element (no overlap)")
            print(f"  w  pdl M={M:2d} N={N:5d} K={K:4d} {label:28s}: behind a producer writing 2 ms after releasing "
                  f"its dependents == final tensors {_ok(ok)}; control: pre-wait reader saw {n_stale} stale  "
                  f"{_ok(n_stale > 0)}")


def _w_worker_compile(w: _WWorker, spec: dict) -> None:
    N, K = 2560, 2048
    dw = prepared(12, N, K, seed=71)
    bias = torch.randn(N, device="cuda").bfloat16()

    def f_x(x, r):
        return D.linear(x, dw, bias=bias, activation="gelu_erf", residual=r)

    def f_2b(x, r):
        xq, sx = D.quantize_activation(x)
        return D.linear(xq, dw, x_scale=sx, bias=bias, activation="gelu_erf", residual=r)

    for fname, f, want_nodes in (("bf16 x", f_x, {"dense_linear": 1}),
                                 ("2b", f_2b, {"dense_quantize_activation": 1, "dense_linear": 1})):
        ins = {M: (make_x(M, K, seed=72 + M), torch.randn(M, N, device="cuda").bfloat16()) for M in (4, 16, 40, 64)}
        want = {M: f(*a) for M, a in ins.items()}
        graphs = []

        def record(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        torch._dynamo.reset()
        rec = torch.compile(f, backend=record, fullgraph=True)
        rec_ok = all(bytes_equal(rec(*a), want[M]) for M, a in ins.items())
        per_graph = [[str(n.target) for n in g.graph.nodes if n.op == "call_function" and "fish_scales_ops" in
                      str(n.target)] for g in graphs]
        node_ok = bool(graphs) and all(len(ours) == sum(want_nodes.values()) and all(
            sum(k in t and (k != "dense_linear" or "quantized" not in t) for t in ours) == v
            for k, v in want_nodes.items()) for ours in per_graph)
        w.check(rec_ok and node_ok, f"w compile ({fname}): recording backend nodes {per_graph}, equal {rec_ok}")
        print(f"  w  torch.compile ({fname}, bias + gelu_erf + residual, M = 4, 16, 40, 64 on {N}x{K}), recording "
              f"backend, fullgraph: {len(graphs)} graph(s), nodes {per_graph[0] if per_graph else []}; == eager  "
              f"{_ok(rec_ok and node_ok)}")
        torch._dynamo.reset()
        ind = torch.compile(f, fullgraph=True)
        ok = all(bytes_equal(ind(*a), want[M]) for M, a in ins.items())
        torch.cuda.synchronize()
        w.check(ok, f"w compile ({fname}): inductor != eager")
        print(f"  w  torch.compile ({fname}), inductor, fullgraph: == eager  {_ok(ok)}")
        torch._dynamo.reset()


def _w_worker(spec_json: str) -> int:
    import json
    spec = json.loads(spec_json)
    w = _WWorker()
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        {"exact": _w_worker_exact, "graphs": _w_worker_graphs, "pdl": _w_worker_pdl,
         "compile": _w_worker_compile}[spec["kind"]](w, spec)
    except Exception as e:  # noqa: BLE001 - reported to the parent as a failure with its reason
        import traceback
        traceback.print_exc()
        w.failures.append(f"w {spec.get('label', spec['kind'])}: {type(e).__name__}: {e}")
    print("W_RESULT " + json.dumps({"failures": w.failures, "digests": w.digests, "cells": w.cells}))
    return 0


def _w_specs() -> list:
    """(label, env, spec) of every worker of case w. The boundary cells of the rules are read from the rules op in
    the parent (the same extension the workers load)."""
    sms = _sms()
    op = lambda M, N, K: torch.ops.fish_scales_ops.sm120_smallm_rules(M, N, K, sms)  # noqa: E731
    cells = []
    for tag, (N, K) in W_SHAPES.items():
        ms = set(W_MS)
        for idx in (0, 1):  # the last M each rule routes and the first it does not
            top = max((M for M in range(1, 65) if int(op(M, N, K)[idx])), default=0)
            ms.update(m for m in (top, top + 1) if 1 <= m <= 64)
        cells += [{"mnk": (M, N, K)} for M in sorted(ms)]
    held = [{"mnk": (M, N, K)} for N, K in W_HELD_OUT for M in (1, 8, 32, 64)]
    zero = [(4, 2560, 2048), (16, 2048, 4096), (64, 2048, 512), (1, 2560, 9728)]
    quantized = [(4, 2560, 2048), (1, 1280, 7680)]
    rules_spec = {"kind": "exact", "label": "rules", "cells": cells + held, "zero": zero, "quantized": quantized}
    on_cells = [{"mnk": (M, N, K)} for N, K in W_ON_SHAPES for M in range(1, 65)]
    off_cells = [{"mnk": (M, N, K)} for tag, (N, K) in W_SHAPES.items() for M in (1, 8, 32)]
    return [("rules", {}, rules_spec),
            ("rules, FSO_DISABLE_PDL=1", {"FSO_DISABLE_PDL": "1"}, dict(rules_spec, zero=[], quantized=[])),
            ("FSO_SMALLM_ROUTE=1", {"FSO_SMALLM_ROUTE": "1"},
             {"kind": "exact", "label": "on", "cells": on_cells, "expect_use": True, "zero": zero}),
            ("FSO_SMALLM_ROUTE=0", {"FSO_SMALLM_ROUTE": "0"},
             {"kind": "exact", "label": "off", "cells": off_cells, "expect_use": False}),
            ("graphs and streams", {}, {"kind": "graphs", "label": "graphs"}),
            ("PDL late producer", {}, {"kind": "pdl", "label": "pdl"}),
            ("torch.compile", {}, {"kind": "compile", "label": "compile"})]


def case_w(major: int) -> None:
    import json
    import subprocess
    if major != 12:
        print(f"  w  SKIP: the sm_120/sm_121 small-M route needs sm_120/sm_121; this is {_arch_label()}")
        return
    if os.environ.get("FSO_SMALLM_ROUTE") not in (None, "", "rules") or any(os.environ.get(k) for k in S1C_KNOBS):
        print("  w  SKIP: FSO_SMALLM_ROUTE or a force knob is set in this process; the workers set their own")
        return
    here = os.path.abspath(__file__)
    digests = {}
    for label, env_add, spec in _w_specs():
        env = {k: v for k, v in os.environ.items() if k not in W_KNOBS}
        env.update(env_add)
        r = subprocess.run([sys.executable, here, "--w-worker", json.dumps(spec)], env=env, capture_output=True,
                           text=True)
        result = None
        for line in r.stdout.splitlines():
            if line.startswith("W_RESULT "):
                result = json.loads(line[len("W_RESULT "):])
            else:
                print(line)
        if result is None or r.returncode != 0:
            _check(False, f"w [{label}]: the worker failed (rc {r.returncode}): {r.stderr[-2000:]}")
            print(f"  w  [{label}] worker FAILED rc {r.returncode}\n{r.stderr[-2000:]}")
            continue
        failures.extend(f"[{label}] {f}" for f in result["failures"])
        digests[label] = result["digests"]
        print(f"  w  [{label}] ({', '.join(f'{k}={v}' for k, v in env_add.items()) or 'no knobs'}): "
              f"{result['cells']} exact cells, {len(result['failures'])} failures  {_ok(not result['failures'])}")
    a, b = digests.get("rules"), digests.get("rules, FSO_DISABLE_PDL=1")
    common = sorted(set(a or {}) & set(b or {}))
    ok = bool(common) and all(a[k] == b[k] for k in common)
    _check(ok, f"w PDL: [rules] and [rules, FSO_DISABLE_PDL=1] differ on "
               f"{sum(a[k] != b[k] for k in common) if common else 'all'} of {len(common)} outputs")
    print(f"  w  PDL on vs off: [rules] == [rules, FSO_DISABLE_PDL=1] on {len(common)} outputs  {_ok(ok)}")
    # Informational: how many identity outputs the route changes against the 0.2.2 routes (a K-split order
    # difference; the integer construction makes both exact, so on these cells they must agree).
    on, off = digests.get("rules"), digests.get("FSO_SMALLM_ROUTE=0")
    if on and off:
        strip = lambda d: {k.split("|", 1)[1]: v for k, v in d.items() if k.endswith("|identity")}  # noqa: E731
        on_i, off_i = strip(on), strip(off)
        keys = sorted(set(on_i) & set(off_i))
        # Where the cascade is a single launch both outputs are RN_bf16 of the exact product and must agree; where it
        # splits K (Stream-K) the cascade rounds its partials first, so a difference there is expected.
        single = [k for k in keys if _s1c_route(*map(int, k.split("|")[0].split("x")))["k_split"] <= 1]
        split = [k for k in keys if k not in single]
        same = sum(on_i[k] == off_i[k] for k in single)
        same_split = sum(on_i[k] == off_i[k] for k in split)
        _check(same == len(single), f"w identity: the route and the single-launch cascade differ on "
                                    f"{len(single) - same} of {len(single)} exact cells")
        print(f"  w  identity outputs: route == single-launch cascade on {same} of {len(single)} cells {_ok(same == len(single))}; "
              f"== Stream-K cascade on {same_split} of {len(split)} (informational: Stream-K rounds its partials)")


# --- (p) sm_100/sm_103: the dense epilogue on the C++ cascade (tier 3) ----------------------------------------------
#
# Every family runs in its own process: FSO_FORCE_TILE, FSO_FORCE_KSPLIT, FSO_DISABLE_PDL and the tier switches are
# read once per process. Every family forces the cascade (FSO_DISABLE_SMM=1, FSO_DISABLE_DSL=1, which also switches
# the decode row off) and prints its kernels (FSO_PRINT_TILE_INFO=1), so the parent can see which dense-epilogue
# instantiation each family ran.

P_COMMON_ENV = {"FSO_DISABLE_SMM": "1", "FSO_DISABLE_DSL": "1", "FSO_PRINT_TILE_INFO": "1"}
P_SHAPES = ((1280, 1280), (5120, 1280), (1280, 5120), (1280, 7680), (2048, 5120))   # N x K
P_MS = (1, 7, 78, 129, 1079)
# The cascade's own picks: the issue-12 tower / LLM shapes plus a narrow-N long-K decode cell (split-K by the rules).
P_RULE_SHAPES = ((1280, 1280), (5120, 1280), (1280, 5120), (1280, 7680), (2048, 1280), (5120, 2048), (2048, 4096))
# label -> (FSO_FORCE_TILE, the dense-epilogue instantiation it must run: TM x TN x TK, nosmem)
P_FORCED = {
    "21/64": ("128,64,21", "128x64x128", 0), "21/128": ("128,128,21", "128x128x128", 0),
    "21/192": ("128,192,21", "128x192x128", 0), "21/256": ("128,256,21", "128x256x128", 0),
    "25/64": ("256,64,25", "256x64x128", 0), "25/128": ("256,128,25", "256x128x128", 0),
    "25/192": ("256,192,25", "256x192x128", 0), "25/256": ("256,256,25", "256x256x128", 0),
    "4/128": ("256,128,4", "256x128x256", 0),
    "20/128": ("128,128,20", "128x128x128", 1), "20/256": ("128,256,20", "128x256x128", 1),
    "24/256": ("256,256,24", "256x256x128", 1),
}
# The two-kernel split-K, forced (auto split and a forced split of 2); M <= 129 keeps a split on every shape but
# N = 5120 at M = 129, whose grid already fills the machine (that cell takes the plain (128, 128) tile).
P_SPLIT = {"9": {"FSO_FORCE_TILE": "128,128,9"}, "9/ks2": {"FSO_FORCE_TILE": "128,128,9", "FSO_FORCE_KSPLIT": "2"}}
P_SPLIT_MS = (1, 7, 78, 129)
# Rows without a dense-epilogue instantiation: they refuse an epilogue call and keep serving identity calls.
P_REFUSE = {"refuse/7": ("128,128,7", "Stream-K"), "refuse/8": ("128,128,8", "Stream-K"),
            "refuse/3": ("128,128,3", "no dense-epilogue instantiation")}

_SM100_GELU_CU = r"""
#include <torch/extension.h>
// The erf-GELU of interface v1, written as the charter states it, in a kernel of its own: the reference the fused
// epilogue must match bit for bit (same compiler, same erff, same target as the extension).
__global__ void gelu_ref_kernel(float const* __restrict__ v, float* __restrict__ o, long n)
{
    long const i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float const x = v[i];
        o[i] = 0.5f * x * (1.f + erff(x * 0.70710678118654752440f));
    }
}
torch::Tensor gelu_ref(torch::Tensor v)
{
    TORCH_CHECK(v.is_cuda() && v.dtype() == torch::kFloat32 && v.is_contiguous());
    auto o = torch::empty_like(v);
    long const n = v.numel();
    if (n > 0)
        gelu_ref_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            v.data_ptr<float>(), o.data_ptr<float>(), n);
    return o;
}
"""


def _sm100_gelu_ext():
    from torch.utils.cpp_extension import load_inline
    major, minor = _arch()
    name = f"fso_gelu_ref_sm{major}{minor}"
    build_dir = os.path.join(tempfile.gettempdir(), f"fso_test_dense_v1_{os.getuid()}", name)
    os.makedirs(build_dir, exist_ok=True)
    # The extension's own target (an arch flag in extra_cuda_cflags stops torch from adding its own).
    return load_inline(name=name, cpp_sources="torch::Tensor gelu_ref(torch::Tensor v);",
                       cuda_sources="#include <ATen/cuda/CUDAContext.h>\n" + _SM100_GELU_CU, functions=["gelu_ref"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_100f,code=sm_100f"],
                       build_directory=build_dir, verbose=False)


def _p_ints(M: int, N: int, K: int, seed: int, sparse: bool):
    """bf16 integer activations [M, K] with a +-4 at position 0 of every 32-block (the quantizer then picks
    UE8M0 121 and codes x * 64, exact) and integer weights [N, K]. Dense: both uniform in [-4, 4], so |acc| reaches
    the thousands and a double rounding is visible. Sparse (for erf-GELU): 3 % of the other activations are +-1,
    weights are in [-1, 1] and zero at position 0 of every block, so |acc| stays where erf is not saturated."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    if sparse:
        mask = torch.rand(M, K, generator=g, device="cuda") < 0.03
        x = torch.where(mask, torch.randint(0, 2, (M, K), generator=g, device="cuda") * 2 - 1, 0)
        w = torch.randint(-1, 2, (N, K), generator=g, device="cuda")
        w.view(N, K // 32, 32)[:, :, 0] = 0
    else:
        x = torch.randint(-4, 5, (M, K), generator=g, device="cuda")
        w = torch.randint(-4, 5, (N, K), generator=g, device="cuda")
    x.view(M, K // 32, 32)[:, :, 0] = 4 * (torch.randint(0, 2, (M, K // 32), generator=g, device="cuda") * 2 - 1)
    return x.to(torch.bfloat16), w


def _p_weight(w: torch.Tensor):
    """The MXFP8 checkpoint entry: fp8 codes = the integers, UE8M0 byte 127 (scale 1) everywhere."""
    N, K = w.shape
    return D.prepare_weight(w.to(torch.float32).to(torch.float8_e4m3fn), format="mxfp8",
                            scale=torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda"))


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _p_eq(label: str, got: torch.Tensor, want: torch.Tensor, local: list) -> bool:
    ok = got.dtype == want.dtype and tuple(got.shape) == tuple(want.shape) and torch.equal(_bits(got), _bits(want))
    if not ok:
        bad = int((_bits(got) != _bits(want)).sum()) if tuple(got.shape) == tuple(want.shape) else -1
        local.append(f"{label}: {bad} element(s) differ in their bits")
    return ok


class _PDigest:
    def __init__(self):
        import hashlib
        self.h = hashlib.sha256()

    def add(self, t: torch.Tensor):
        self.h.update(_bits(t).cpu().numpy().tobytes())

    def hex(self) -> str:
        return self.h.hexdigest()


def _p_cell(M: int, N: int, K: int, seed: int, ext, dig: _PDigest, local: list, stats: dict) -> None:
    """The exactness cases of charter section 10 on one (M, N, K): bias, residual, bias + residual (both entry
    paths), zero bias, erf-GELU with and without bias / residual; expected bytes from the exact fp64 accumulator."""
    L = D.linear
    tag = f"M={M} N={N} K={K}"
    x, w = _p_ints(M, N, K, seed, sparse=False)
    dw = _p_weight(w)
    xq, sx = D.quantize_activation(x)
    if not torch.equal(xq.float(), x.float() * 64):
        local.append(f"{tag}: the activation quantizer did not give the exact codes x * 64 (test construction)")
        return
    acc = x.double() @ w.double().t()  # exact: integer products, |acc| <= 16 K < 2^24
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    b = (torch.randint(-64, 65, (N,), generator=g, device="cuda").double() / 8).bfloat16()
    r = (torch.randint(-1023, 1024, (M, N), generator=g, device="cuda").double() / 8).bfloat16()
    bd, rd = b.double()[None, :], r.double()

    y0 = L(xq, dw, x_scale=sx)  # the identity: today's bytes, and the sign the route gives an exact-zero acc
    # Where acc == 0 and b == 0 the result is the accumulator's zero, whose sign the route decides; there the
    # expected bytes are the identity's (the -0.0 rule makes bias = 0 the identity).
    zpos = (acc == 0) & (bd == 0)
    rn = lambda v: v.float().bfloat16()  # noqa: E731 - fp64 -> fp32 is exact here, then one RN to bf16
    _p_eq(f"{tag} identity == RN_bf16(acc)", y0, torch.where(acc == 0, y0, rn(acc)), local)
    want_b = torch.where(zpos, y0, rn(acc + bd))
    yb = L(xq, dw, x_scale=sx, bias=b)
    _p_eq(f"{tag} bias", yb, want_b, local)
    stats["double_rounding_differs"] += int((rn(rn(acc).double() + bd) != want_b).sum())
    stats["elements"] += M * N
    yr = L(xq, dw, x_scale=sx, residual=r)
    _p_eq(f"{tag} residual", yr, rn(acc + rd), local)
    want_br = rn(acc + bd + rd)
    ybr = L(xq, dw, x_scale=sx, bias=b, residual=r)
    _p_eq(f"{tag} bias + residual", ybr, want_br, local)
    ybr_x = L(x, dw, bias=b, residual=r)  # the bf16 entry path: the quantize inside gives the same xq
    _p_eq(f"{tag} bias + residual, bf16 x", ybr_x, want_br, local)
    yz = L(xq, dw, x_scale=sx, bias=torch.zeros_like(b))
    _p_eq(f"{tag} bias = 0 == bias = None", yz, y0, local)
    for t in (y0, yb, yr, ybr, ybr_x, yz):
        dig.add(t)

    xs, ws = _p_ints(M, N, K, seed + 7, sparse=True)
    dws = _p_weight(ws)
    xsq, ssx = D.quantize_activation(xs)
    accs = xs.double() @ ws.double().t()
    y0s = L(xsq, dws, x_scale=ssx)
    gl = ext.gelu_ref((accs + bd).float().contiguous())   # acc + b is exact in fp32
    gl0 = ext.gelu_ref(accs.float().contiguous())
    zs = (accs == 0) & (bd == 0)
    yg = L(xsq, dws, x_scale=ssx, bias=b, activation="gelu_erf")
    _p_eq(f"{tag} bias + gelu_erf", yg, torch.where(zs, y0s, gl.bfloat16()), local)
    ygr = L(xsq, dws, x_scale=ssx, bias=b, residual=r, activation="gelu_erf")
    _p_eq(f"{tag} bias + gelu_erf + residual", ygr, (gl + r.float()).bfloat16(), local)
    yg0 = L(xs, dws, activation="gelu_erf")
    _p_eq(f"{tag} gelu_erf, bf16 x", yg0, torch.where(accs == 0, y0s, gl0.bfloat16()), local)
    stats["gelu_active"] += int(((accs + bd).abs() < 4).sum())
    for t in (y0s, yg, ygr, yg0):
        dig.add(t)
    stats["cells"] += 1


def _p_zero_corner(local: list, stats: dict) -> None:
    """x row 0 all zeros and weight row (output column) 0 with all-negative codes: every product is -0.0. The
    bias = 0 output must have the identity's bytes, whatever sign the route gives that accumulator."""
    M, N, K = 7, 1280, 1280
    x, w = _p_ints(M, N, K, 991, sparse=False)
    x[0] = 0
    w[0] = -torch.randint(1, 5, (K,), device="cuda")
    dw = _p_weight(w)
    xq, sx = D.quantize_activation(x)
    y0 = D.linear(xq, dw, x_scale=sx)
    yz = D.linear(xq, dw, x_scale=sx, bias=torch.zeros(N, dtype=torch.bfloat16, device="cuda"))
    stats["zero_byte"] = f"0x{int(_bits(y0)[0, 0]) & 0xFFFF:04x}"
    _p_eq("-0.0 corner: bias = 0 == bias = None", yz, y0, local)
    xr, dr = make_x(78, 2048, 993), prepared(10, 1280, 2048, 994)
    _p_eq("random data: bias = 0 == bias = None", D.linear(xr, dr, bias=torch.zeros(1280, dtype=torch.bfloat16,
                                                                                         device="cuda")),
          D.linear(xr, dr), local)


def _p_capture(label: str, fn, local: list) -> None:
    ref = fn().clone()  # one eager call on the default stream
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = fn()
    out.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    if not (bool(torch.isfinite(out).all()) and torch.equal(_bits(out), _bits(ref))):
        local.append(f"capture {label}: replay into a NaN-filled output != eager")


def _p_compile(local: list) -> None:
    N, K, M = 5120, 1280, 78
    dw = prepared(10, N, K, seed=71)
    x = make_x(M, K, seed=72)
    b = torch.randn(N, device="cuda").bfloat16()
    r = torch.randn(M, N, device="cuda").bfloat16()

    def f(x):
        return D.linear(x, dw, bias=b, residual=r, activation="gelu_erf")

    want = f(x)
    graphs = []

    def record(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    got = torch.compile(f, backend=record, fullgraph=True)(x)
    nodes = [str(n.target) for gm in graphs for n in gm.graph.nodes
             if n.op == "call_function" and "fish_scales_ops" in str(n.target)]
    if not (torch.equal(got, want) and len(nodes) == 1 and "dense_linear" in nodes[0]):
        local.append(f"torch.compile recording backend: nodes {nodes}, equal {torch.equal(got, want)}")
    torch._dynamo.reset()
    got = torch.compile(f, fullgraph=True)(x)
    torch.cuda.synchronize()
    if not torch.equal(got, want):
        local.append("torch.compile inductor: compiled != eager")
    torch._dynamo.reset()


def _p_misaligned_residual(local: list) -> None:
    N, K, M = 1280, 1280, 78
    dw = prepared(10, N, K, seed=81)
    x = make_x(M, K, seed=82)
    buf = torch.zeros(M * N + 8, dtype=torch.bfloat16, device="cuda")
    r_mis = buf[1:1 + M * N].view(M, N)
    xq, sx = D.quantize_activation(x)
    torch.cuda.synchronize()
    for label, fn in (("bf16 x", lambda: D.linear(x, dw, residual=r_mis)),
                      ("2b", lambda: D.linear(xq, dw, x_scale=sx, residual=r_mis)),
                      ("raw linear_mxfp8_raw", lambda: torch.ops.fish_scales_ops.linear_mxfp8_raw(
                          xq, dw.weight, sx, dw.scale, None, r_mis))):
        exc, n = _device_events(fn)
        if type(exc) is not NotImplementedError or "16" not in str(exc) or n != 0:
            local.append(f"misaligned residual ({label}): {type(exc).__name__ if exc else 'no exception'}, "
                         f"{n} device events: {exc}")


def _p_two_streams(local: list) -> None:
    """Two different split-K calls captured on two streams of one thread, replayed concurrently 200 times; each
    must equal its serial result every time. The parent checks that the two captures baked in different
    workspaces ([fso sm100 splitk] captured ... lines)."""
    L = D.linear
    calls = []
    for (N, K, M, seed) in ((1280, 5120, 7, 101), (2048, 7680, 1, 102)):
        dw = prepared(10, N, K, seed=seed)
        x = make_x(M, K, seed=seed + 1)
        xq, sx = D.quantize_activation(x)
        b = torch.randn(N, device="cuda").bfloat16()
        r = torch.randn(M, N, device="cuda").bfloat16()
        calls.append(lambda xq=xq, dw=dw, sx=sx, b=b, r=r: L(xq, dw, x_scale=sx, bias=b, residual=r,
                                                             activation="gelu_erf"))
    refs = [c().clone() for c in calls]
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs, outs = [], []
    for c, s in zip(calls, streams):
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            outs.append(c())
        graphs.append(g)
    torch.cuda.synchronize()
    bad = [torch.zeros((), dtype=torch.int64, device="cuda") for _ in calls]
    for _ in range(200):
        for i in range(2):
            with torch.cuda.stream(streams[i]):
                graphs[i].replay()
                bad[i] += (_bits(outs[i]) != _bits(refs[i])).sum()
    torch.cuda.synchronize()
    nbad = [int(t) for t in bad]
    if any(nbad):
        local.append(f"two-stream concurrent replay: {nbad} mismatching elements over 200 iterations")


def _p_refuse(force: str, word: str, local: list) -> None:
    N, K, M = 1280, 1280, 7
    dw = prepared(10, N, K, seed=91)
    x = make_x(M, K, seed=92)
    xq, sx = D.quantize_activation(x)
    b = torch.randn(N, device="cuda").bfloat16()
    torch.cuda.synchronize()
    for label, fn in (("2b, bias", lambda: D.linear(xq, dw, x_scale=sx, bias=b)),
                      ("2b, gelu_erf", lambda: D.linear(xq, dw, x_scale=sx, activation="gelu_erf")),
                      ("raw, bias", lambda: torch.ops.fish_scales_ops.linear_mxfp8_raw(xq, dw.weight, sx, dw.scale,
                                                                                        b))):
        exc, n = _device_events(fn)
        msg = str(exc)
        if type(exc) is not NotImplementedError or f"FSO_FORCE_TILE={force}" not in msg or word not in msg or n:
            local.append(f"FSO_FORCE_TILE={force} ({label}): {type(exc).__name__ if exc else 'no exception'}, "
                         f"{n} device events: {msg}")
    y = D.linear(xq, dw, x_scale=sx)  # the identity keeps working on the row
    torch.cuda.synchronize()
    if not bool(torch.isfinite(y).all()):
        local.append(f"FSO_FORCE_TILE={force}: the identity call gave non-finite values")


def p_worker(label: str) -> int:
    """One family of case p, in this process; the last stdout line is a JSON summary for the parent."""
    import json
    local: list[str] = []
    stats = {"cells": 0, "elements": 0, "double_rounding_differs": 0, "gelu_active": 0}
    dig = _PDigest()
    torch.backends.cuda.matmul.allow_tf32 = False
    if label.startswith("refuse/"):
        _p_refuse(*P_REFUSE[label], local)
    elif label == "streams":
        _p_two_streams(local)
    else:
        ext = _sm100_gelu_ext()
        if label in P_SPLIT:
            cells = [(M, N, K) for (N, K) in P_SHAPES for M in P_SPLIT_MS]
        elif label in P_FORCED:
            cells = [(M, N, K) for (N, K) in P_SHAPES for M in P_MS]
        else:  # "rules": the cascade's own picks
            cells = [(M, N, K) for (N, K) in P_RULE_SHAPES for M in P_MS]
        for (M, N, K) in cells:
            _p_cell(M, N, K, M * 1009 + N * 17 + K, ext, dig, local, stats)
        _p_zero_corner(local, stats)
        M, N, K = cells[min(2, len(cells) - 1)]
        x, w = _p_ints(M, N, K, 7, sparse=False)
        dw = _p_weight(w)
        xq, sx = D.quantize_activation(x)
        b = torch.randn(N, device="cuda").bfloat16()
        r = torch.randn(M, N, device="cuda").bfloat16()
        _p_capture(f"M={M} N={N} K={K} bias + gelu_erf + residual",
                   lambda: D.linear(xq, dw, x_scale=sx, bias=b, residual=r, activation="gelu_erf"), local)
        _p_capture(f"M={M} N={N} K={K} bias, bf16 x", lambda: D.linear(x, dw, bias=b), local)
        if label == "rules":
            _p_compile(local)
            _p_misaligned_residual(local)
    print(json.dumps({"label": label, "failures": local, "digest": dig.hex(), **stats}))
    return 0


def _p_spawn(label: str, extra_env: dict):
    import json
    import subprocess
    env = {k: v for k, v in os.environ.items() if not k.startswith("FSO_")}
    env.update(P_COMMON_ENV)
    env.update(extra_env)
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--p-worker", label], env=env,
                       capture_output=True, text=True, timeout=7200)
    summary = None
    for line in reversed(r.stdout.splitlines()):
        if line.startswith("{"):
            summary = json.loads(line)
            break
    return r, summary


def case_p(major: int) -> None:
    import re
    if major != 10:
        print("  p  SKIP: the C++ cascade's dense epilogue is sm_100/sm_103-only")
        return
    runs = {"rules": {}}
    runs.update({k: {"FSO_FORCE_TILE": v[0]} for k, v in P_FORCED.items()})
    runs.update(P_SPLIT)
    runs.update({k: {"FSO_FORCE_TILE": v[0]} for k, v in P_REFUSE.items()})
    runs["streams"] = {"FSO_FORCE_TILE": "128,128,9"}
    nopdl = {"rules": "rules/nopdl", "9": "9/nopdl"}
    digests = {}
    epi_re = re.compile(r"^\[fso sm100 tile\] epilogue (\d+x\d+x\d+) cluster=\(\d+,\d+\) nosmem=(\d) residual=(\d)")
    for label, env in list(runs.items()) + [(v, {**runs[k], "FSO_DISABLE_PDL": "1"}) for k, v in nopdl.items()]:
        worker = label.split("/nopdl")[0]
        r, summary = _p_spawn(worker, env)
        ok = r.returncode == 0 and summary is not None and not summary["failures"]
        epi = sorted({m.groups() for m in map(epi_re.match, r.stderr.splitlines()) if m})
        note = ""
        if worker in P_FORCED:
            _, tile, ns = P_FORCED[worker]
            want = {(tile, str(ns), "0"), (tile, str(ns), "1")}
            cov = set(epi) == want
            ok = ok and cov
            note = f"; ran {['%s nosmem=%s residual=%s' % e for e in epi]}{'' if cov else ' (expected ' + str(sorted(want)) + ')'}"
        elif worker in P_SPLIT:
            split_ran = "[fso sm100 tile] splitk 128x128x128" in r.stderr
            ok = ok and split_ran
            note = f"; split-K ran {split_ran}"
        elif worker == "rules":
            split_ran = "[fso sm100 tile] splitk 128x128x128" in r.stderr
            ok = ok and split_ran and len(epi) > 0
            note = f"; rules ran {len(epi)} epilogue kernels {['%s ns=%s r=%s' % e for e in epi]}, split-K {split_ran}"
        elif worker == "streams":
            caps = re.findall(r"\[fso sm100 splitk\] captured on stream (0x[0-9a-f]+): workspace (0x[0-9a-f]+)",
                              r.stderr)
            two = len(caps) == 2 and caps[0][0] != caps[1][0] and caps[0][1] != caps[1][1]
            ok = ok and two
            note = f"; captures {caps} (two streams, two workspaces: {two})"
        if summary is not None and summary.get("cells"):
            note += (f"; {summary['cells']} cells, double rounding differs on {summary['double_rounding_differs']} of "
                     f"{summary['elements']} elements, {summary['gelu_active']} GELU inputs in |v| < 4, -0.0 corner "
                     f"byte {summary.get('zero_byte')}")
            if summary["double_rounding_differs"] == 0:
                ok = False
                note += " (the test does not discriminate a double rounding)"
            digests[label] = summary["digest"]
        if not ok:
            detail = summary["failures"][:8] if summary else r.stdout[-1500:] + r.stderr[-1500:]
            _check(False, f"p {label}: {detail}")
        print(f"  p  {label:12s} {_ok(ok)}{note}")
        if not ok and summary is not None:
            for f in summary["failures"][:8]:
                print(f"       {f}")
    for k, v in nopdl.items():
        same = k in digests and digests.get(k) == digests.get(v)
        _check(same, f"p: FSO_DISABLE_PDL=1 changed the bytes of family {k}")
        print(f"  p  PDL on / off, family {k}: identical bytes  {_ok(same)}")


# --- (q) sm_100/sm_103: the dense epilogue on the decode row and tier 1, feature-aware fall-through (S1b1) --------
#
# Families, one subprocess each (FSO_DISABLE_PDL is read once per process; the DSL compiles are per process too):
#   decode   exactness of charter section 10 on the M <= 64 CuTe-DSL decode row, split-K form (M in Q_SPLIT_MS on
#            the narrow-N shapes) and persistent form (M in Q_PERSIST_MS), both entry paths, bias = 0 == None, the
#            -0.0 corner; plus capture, two-stream replay and torch.compile on the row.
#   smm      tier 1 (cuBLAS scaled_mm) with the bias alone on cells it owns, bias = 0 == None, the -0.0 corner, and
#            the fall-through: residual / GELU calls on those cells land on the C++ cascade with the exact bytes.
#   agree    plan() names the tier that route() runs on, over the issue-12 grid and the Family A / B / C dense grids,
#            for the identity, bias, residual and bias + GELU + residual.
#   pdl      a producer that writes the bias / residual 2 ms after releasing its dependents, ahead of a fused call
#            on each route (decode split-K and persistent, tier 1, the cascade fall-through; and, from P5, the
#            narrow-N long-K band's split-K form and its fall-through from tier 1).
#   wide     (issue #12, P5) the swap-AB row above M = 64, the narrow-N long-K band: exactness of section 10 on every
#            form the rule picks (split-K with 64-wide token tiles in two and four slices, the persistent 2-CTA form
#            with 64- and 128-wide token tiles) on o / fc2 / conv_out at partial and several token tiles, both entry
#            paths, the kernel checked by name and plan(); the fall-through of a residual / GELU call from a tier-1
#            cell (fc2 at M = 1079) onto the row; every tactic of the kernels' new tile list forced through
#            linear_mxfp8_decode (the 128-wide split-K tiles the rule does not pick included); bias = 0 == None, the
#            -0.0 corner on the split-K and persistent forms; capture on a non-default stream with a NaN-filled
#            replay; 200 replays of each captured new tactic bit-identical (determinism); two different fused calls
#            replayed concurrently on two streams (200 x); a linear_quantized call of every band cell is planned on
#            the cascade (the row does not write the quantized output; case s runs such calls on the device).
#   shortk   (issue #12, T1) the cascade's short-K tile rule (pick_short_k_tile): exactness on the cells whose bias
#            calls tier 1 declines and on the tier-1 cells whose residual / GELU calls take the rule's tiles, the tile each cascade
#            call ran checked against a test-local mirror of the rule, the -0.0 corner, capture, two streams.
#   fb2      (FB2) the swap-AB route rules of the B300 adoption round: exactness on each rule's cells with every call's
#            tier (Q_FB2_CELLS), the rules' tactics and edges (Q_FB2_TACTICS, Q_FB2_EDGES), linear_quantized on the
#            cascade, the -0.0 corner, capture, four streams.
#   */nopdl  the decode, smm, wide, shortk and fb2 families again with FSO_DISABLE_PDL=1; their output digests must
#            match.

Q_ENV = {"FSO_PRINT_TILE_INFO": "1"}
# Narrow-N shapes whose decode cells split K in-cluster (tiles_n <= 32; K % 256 == 0 and K / split >= 512).
Q_SPLIT_SHAPES = ((1280, 5120), (1280, 7680), (2048, 4096), (1280, 1280))
Q_SPLIT_MS = (1, 7, 8, 16, 32)
# The persistent form: wide N at M <= 32 (one K slice) and the 64-wide token tile from M = 33, on wide N and on the
# narrow-N cells with K < 4096. (Since FB2 the narrow-N long-K cells of 32 < M <= 64, 1280 x 5120 among them, run on the
# split-K form with the 64-wide token tile: Q_FB2_CELLS.)
Q_PERSIST_CELLS = tuple((5120, 1280, M) for M in (12, 33, 48, 64)) + ((1280, 1280, 48), (2048, 2048, 64))
# Cells tier 1 owns (should_route, outside the decode band and the wave rule). fc1 [5120 x 1280] at the tower rungs
# M = 78 / 117 / 533 is NOT one of them: there the cascade's wave rule (`_sm100_smm.wave_tile_owns`, 148 SMs: a
# 128x64 tile puts 80 CTAs on the machine) owns the cell by shape, identity included, so fc1 enters tier 1 at M = 129.
# Nor, for the bias, are the narrow-N short-K cells at 64 < M <= 256 (2048 x 1280 at M = 129 / 182:
# `_sm100_smm.short_k_owns`, issue #12 T1), whose bias calls the shortk family below runs on the cascade; the bias stays
# on tier 1 from M = 257. Nor, since FB2, fc1 at M = 129-192, whose identity and erf-GELU calls the swap-AB row takes
# (Q_FB2_CELLS); from M = 193 the row's 1-SM grid no longer fits one wave and every call keeps its tier.
Q_SMM_CELLS = ((5120, 1280, 208), (5120, 1280, 256), (5120, 1280, 1079), (1280, 1280, 364), (1280, 1280, 257),
               (2048, 1280, 364), (2048, 1280, 1079))
# (issue #12, T1) the cascade's short-K tile rule (pick_short_k_tile in dispatch.cuh), on the card's own routing.
# Cells whose bias calls tier 1 declines (narrow N, 1024 <= K <= 2560, 64 < M <= 256): the identity stays on tier 1,
# every other call runs on the cascade, on the 1-SM 128 x 64 tile, except the bias and residual calls at 128 < M <= 256,
# which take the 2-SM 256 x 64 tile. Since FB2 the swap-AB row's narrow-N short-K band takes every call of the cells
# whose split-K grid (2 x tiles_n x ceil(M / 64) CTAs) has at most 80 CTAs (1280 x 1280 to M = 256, 2048-wide to
# M = 128, 1024 x 2048 to M = 320: Q_FB2_CELLS), so the cells left here are 1536- and 2048-wide above M = 128.
Q_SHORTK_DECLINE_CELLS = ((2048, 1280, 129), (2048, 1280, 182), (2048, 1280, 256), (2048, 2048, 160), (1536, 1280, 200),
                          (2048, 2560, 200))
# Cells tier 1 keeps for the identity and the bias, whose residual / GELU / quantized calls take the rule's tiles on
# the cascade: 5120 x 1280 at M = 256 (2-SM 256 x 128, TileK 128) and 364 (1-SM 128 x 128), 1280 x 1280 at M = 364
# (1-SM 128 x 64) and 1079 (2-SM 256 x 128), 2048 x 1280 at M = 600 (2-SM 256 x 128) and 1079 (1-SM 128 x 128). (At
# 5120 x 1280 M = 129-192 and 2048 x 1280 M = 513-576 the erf-GELU calls run on the swap-AB row since FB2:
# Q_FB2_CELLS.)
Q_SHORTK_EPI_CELLS = ((5120, 1280, 256), (5120, 1280, 364), (1280, 1280, 364), (1280, 1280, 1079), (2048, 1280, 600),
                      (2048, 1280, 1079))
# (issue #12, T1b) cells the wave rule owns (2368 <= N <= 4736 at M <= 128: tier 1 declines every call), so every call
# runs on the cascade: the 1-SM 128 x 64 tile where the activation is under 128 KiB (the empty-pair guard of
# pick_wave_tile: the fused q / k / v 3840 x 1280 at M = 78 and 102, 4096 x 1280 at M = 65, 3840 x 1536 at M = 85), the
# 2-SM 256 x 64 tile from 128 KiB on (3840 x 1280 at M = 103 and 117, 3840 x 1536 at M = 86). (The tp2 qkv shard
# 2560 x 2048 at M = 96 runs on the swap-AB row's narrow-N short-K band since FB2.)
Q_SHORTK_WAVE_CELLS = ((3840, 1280, 78), (3840, 1280, 102), (3840, 1280, 103), (3840, 1280, 117), (4096, 1280, 65),
                       (3840, 1536, 85), (3840, 1536, 86))
# plan / route agreement grids: the issue-12 shapes (tower, stem, LLM tp1 and tp2) and the Family A / B / C dense
# projections of docs/perf/README.md.
Q_AGREE_SHAPES = tuple(dict.fromkeys(SHAPES + ((2560, 2048), (2048, 2048),
                                               (6144, 2560), (2560, 4096), (19456, 2560), (2560, 9728),
                                               (5120, 2048), (9216, 2048), (12288, 2048), (1024, 2048), (2048, 512))))
Q_AGREE_MS = (1, 7, 8, 12, 16, 32, 33, 48, 64, 78, 129, 182, 288, 533, 640, 1079)
# (issue #12, P5) the narrow-N long-K band above M = 64: (N, K, M, expected tactic) on natural routing. o [2048 x 4096]:
# split-K T64 in two slices (K / 4 = 1024 is below the four-way floor) at a partial tile (65), two token tiles (78,
# 129: three), then the persistent 2-CTA form with T64 (384, and 576: 72 of the 74 resident clusters of two); fc2
# [1280 x 5120] and conv_out [1280 x 7680]: four slices while the clusters of four fit (78, 129), two above (364 / 384),
# the persistent form at 533. (The 128-wide persistent form runs on natural routing only as the fall-through below.)
Q_WIDE_CELLS = ((2048, 4096, 65, ((128, 64), (1, 1), True, 2)), (2048, 4096, 78, ((128, 64), (1, 1), True, 2)),
                (2048, 4096, 129, ((128, 64), (1, 1), True, 2)), (2048, 4096, 384, ((256, 64), (2, 1), True, 1)),
                (2048, 4096, 576, ((256, 64), (2, 1), True, 1)),
                (1280, 5120, 78, ((128, 64), (1, 1), True, 4)), (1280, 5120, 129, ((128, 64), (1, 1), True, 4)),
                (1280, 5120, 364, ((128, 64), (1, 1), True, 2)), (1280, 5120, 533, ((256, 64), (2, 1), True, 1)),
                (1280, 7680, 78, ((128, 64), (1, 1), True, 4)), (1280, 7680, 129, ((128, 64), (1, 1), True, 4)),
                (1280, 7680, 384, ((128, 64), (1, 1), True, 2)))
# A tier-1 cell of the class (fc2 at M = 1079): identity and bias stay on cuBLAS, residual / GELU fall through onto the
# row's persistent form (T128) instead of the C++ cascade.
Q_WIDE_FALLTHROUGH = ((1280, 5120, 1079, ((256, 128), (2, 1), True, 1)),)
# Every tactic of the kernels' new tile list, forced through linear_mxfp8_decode on (N, K, M).
Q_WIDE_FORCED_TACTICS = (((128, 64), (1, 1), True, 2), ((128, 64), (1, 1), True, 4), ((128, 128), (1, 1), True, 2),
                         ((128, 128), (1, 1), True, 4), ((256, 64), (2, 1), True, 1), ((256, 128), (2, 1), True, 1))
Q_WIDE_FORCED_CELLS = ((2048, 4096, 129), (1280, 5120, 200), (1280, 7680, 65))
# (FB2) the route rules of the B300 swap-AB adoption round: (N, K, M, identity tier, bias tier, residual tier,
# activation tier) on natural routing. The narrow-N long-K class at 32 < M <= 64 on the split-K form with the 64-wide
# token tile (four slices on 1280 x 5120, two on 2048 / 2560 x 4096); the narrow-N short-K band (two slices while the
# split-K grid has at most 80 CTAs); the identity call of wide-N short-K cells on the 1-SM persistent 128 x 64 form
# (its bias / residual calls keep their tier); the erf-GELU call where the cascade's short-K rule would leave its
# 64-wide tile (fc1 5120 x 1280 at M = 129-192, 2048 x 1280 at 513-576, 2560 x 1280 at 385-448, 6144 x 2560 at
# 129-192), whose identity / bias / residual calls keep tier 1 or the cascade.
_DEC, _SMM, _CAS = "sm100.decode", "sm100.smm", "sm100.cascade"
Q_FB2_CELLS = tuple((1280, 5120, M, _DEC, _DEC, _DEC, _DEC) for M in (33, 48, 64)) + (
    (2048, 4096, 48, _DEC, _DEC, _DEC, _DEC), (2560, 4096, 40, _DEC, _DEC, _DEC, _DEC),
    (1280, 1280, 65, _DEC, _DEC, _DEC, _DEC), (1280, 1280, 129, _DEC, _DEC, _DEC, _DEC),
    (1280, 1280, 256, _DEC, _DEC, _DEC, _DEC), (2048, 1280, 117, _DEC, _DEC, _DEC, _DEC),
    (1024, 2048, 320, _DEC, _DEC, _DEC, _DEC), (2560, 2048, 96, _DEC, _DEC, _DEC, _DEC),
    (2048, 2048, 80, _DEC, _DEC, _DEC, _DEC),
    (5120, 2048, 96, _DEC, _CAS, _CAS, _CAS), (9216, 2048, 128, _DEC, _CAS, _CAS, _CAS),
    (5120, 1280, 78, _DEC, _CAS, _CAS, _CAS), (5120, 1280, 182, _DEC, _SMM, _CAS, _DEC),
    (6144, 2560, 160, _DEC, _SMM, _CAS, _DEC), (2048, 1280, 533, _SMM, _SMM, _CAS, _DEC),
    (2560, 1280, 448, _SMM, _SMM, _CAS, _DEC))


def _q_tier(names) -> str:
    """The sm_100 route that ran, from the CUDA kernel names of one call (the activation quantize of a bf16 x and
    torch's own kernels are ignored)."""
    names = [n for n in names if "quant" not in n.lower() and "elementwise" not in n.lower()
             and "fill" not in n.lower() and "copy" not in n.lower()]
    if any("nvjet" in n for n in names):
        return "sm100.smm"
    if any("flashinfer_sm100_blockscaled" in n for n in names):
        return "sm100.decode"
    if any("cutlass_wrapper" in n for n in names):
        return "sm100.dsl"
    return "sm100.cascade"


def _q_mma_tile(names):
    """(tile rows, tile columns) of the CUTLASS cascade GEMM among `names`, from its MMA atom
    (SM100_MMA_MXF8F6F4_SS: 1-SM, rows 128; SM100_MMA_MXF8F6F4_2x1SM_SS: 2-SM, rows 256), or None."""
    for n in names:
        m = re.search(r"SM100_MMA_MXF8F6F4_(2x1SM_)?SS\w*?Li(\d+)ELi(\d+)E", n)
        if m:
            return (int(m.group(2)), int(m.group(3)))
    return None


def _shortk_tile(M: int, N: int, K: int, fused: bool):
    """Test-local mirror of pick_short_k_tile (dispatch.cuh, issue #12 T1): the (rows, columns) of the cascade's
    short-K tile, or None where the rule does not answer. `fused`: an activation or the quantized output."""
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    def ctas(rows, per, tn):
        return -(-M // rows) * -(-N // tn) * per
    if K > 2560 or M < 65 or M > 2048:
        return None
    if ctas(128, 1, 64) <= sms:
        if not fused:
            from fish_scales_ops.gemm import _sm100_smm
            if _sm100_smm.wave_tile_owns(M, N):     # pick_wave_tile's 64-wide pick: the 2-SM form when it fits,
                # except below M = 129 with an activation under 128 KiB and K >= 1024 (the empty-pair guard, #12 T1b)
                small_act = M <= 128 and K >= 1024 and M * K < 128 * 1024
                return (256, 64) if ctas(256, 2, 64) <= sms and not small_act else (128, 64)
            if 128 < M <= 256 and ctas(256, 2, 64) <= sms:
                return (256, 64)
        return (128, 64)
    if ctas(256, 2, 128) <= sms:
        return (256, 128)
    if ctas(128, 1, 128) <= sms:
        return (128, 128)
    return None


def _q_shortk_tiles(M: int, N: int, K: int, local: list, stats: dict) -> None:
    """Every call of one cell that plans on the cascade must run the short-K tile the rule names: the identity,
    the bias, the residual, bias + erf-GELU (both entries) and linear_quantized(bias + erf-GELU)."""
    from fish_scales_ops.gemm import _sm100_dispatch
    dw = prepared(10, N, K, seed=401 + N + M)
    x = make_x(M, K, seed=402 + M)
    xq, sx = D.quantize_activation(x)
    b = (torch.randn(N, device="cuda") * 0.1).bfloat16()
    r = torch.randn(M, N, device="cuda").bfloat16()
    calls = (("identity", dict(), False, lambda: D.linear(xq, dw, x_scale=sx)),
             ("bias", dict(bias=b), False, lambda: D.linear(xq, dw, x_scale=sx, bias=b)),
             ("bias (bf16 x)", dict(bias=b), False, lambda: D.linear(x, dw, bias=b)),
             ("residual", dict(residual=r), False, lambda: D.linear(xq, dw, x_scale=sx, residual=r)),
             ("bias + gelu_erf", dict(bias=b, activation="gelu_erf"), True,
              lambda: D.linear(xq, dw, x_scale=sx, bias=b, activation="gelu_erf")),
             ("bias + gelu_erf (bf16 x)", dict(bias=b, activation="gelu_erf"), True,
              lambda: D.linear(x, dw, bias=b, activation="gelu_erf")),
             ("linear_quantized bias + gelu_erf", None, True,
              lambda: D.linear_quantized(xq, dw, x_scale=sx, bias=b, activation="gelu_erf")))
    for label, kw, fused, fn in calls:
        _, names = _q_kernels(fn)
        tier = _q_tier(names)
        if kw is not None:
            planned = _sm100_dispatch.plan(M, N, K, _q_epilogue(M, N, **kw) if kw else None)
            if planned != tier:
                local.append(f"shortk M={M} N={N} K={K} {label}: ran on {tier}, plan() says {planned}")
        if tier != "sm100.cascade":
            continue
        want, got = _shortk_tile(M, N, K, fused), _q_mma_tile(names)
        if want is None or got != want:
            local.append(f"shortk M={M} N={N} K={K} {label}: the cascade ran tile {got}, the rule names {want}")
        stats["shortk_tiles"] = stats.get("shortk_tiles", 0) + 1


def _q_kernels(fn):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = fn()
        torch.cuda.synchronize()
    return out, [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]


def _q_epilogue(M: int, N: int, bias=None, residual=None, activation=None):
    from fish_scales_ops.gemm import _epilogue
    # The device the weight would carry: an explicit index, because _epilogue.make compares it with the bias's and
    # the residual's device, and a tensor made with device="cuda" reports "cuda:<index>".
    return _epilogue.make("test", m=M, n=N, device=torch.device("cuda", torch.cuda.current_device()), bias=bias,
                          residual=residual, activation=activation)


def _q_cell(M: int, N: int, K: int, seed: int, ext, dig: _PDigest, local: list, stats: dict, want_tier: str,
            fallthrough_tier: str = None, bias_tier: str = None, act_tier: str = None) -> None:
    """The exactness cases of section 10 on one cell, each call checked for the route it ran on: `want_tier` for
    the identity, `bias_tier or want_tier` for the bias (issue #12 T1: the bias calls of the short-K narrow-N cells
    leave tier 1), `fallthrough_tier or want_tier` for residual calls (tier 1 serves the bias alone, so those fall
    through to the cascade there), and `act_tier or fallthrough_tier or want_tier` for erf-GELU calls (FB2: the
    swap-AB row takes them where the cascade's short-K rule would leave its 64-wide tile)."""
    from fish_scales_ops.gemm import _sm100_dispatch
    L = D.linear
    tag = f"M={M} N={N} K={K}"
    ft = fallthrough_tier or want_tier
    bt = bias_tier or want_tier
    at = act_tier or ft
    x, w = _p_ints(M, N, K, seed, sparse=False)
    dw = _p_weight(w)
    xq, sx = D.quantize_activation(x)
    if not torch.equal(xq.float(), x.float() * 64):
        local.append(f"{tag}: the activation quantizer did not give the exact codes x * 64 (test construction)")
        return
    acc = x.double() @ w.double().t()
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    b = (torch.randint(-64, 65, (N,), generator=g, device="cuda").double() / 8).bfloat16()
    r = (torch.randint(-1023, 1024, (M, N), generator=g, device="cuda").double() / 8).bfloat16()
    bd, rd = b.double()[None, :], r.double()
    rn = lambda v: v.float().bfloat16()  # noqa: E731

    def run(label, fn, want, tier, one_kernel=False):
        got, names = _q_kernels(fn)
        ok = _p_eq(f"{tag} {label}", got, want, local)
        ran = _q_tier(names)
        if ran != tier:
            local.append(f"{tag} {label}: ran on {ran} ({names}), expected {tier}")
            ok = False
        if one_kernel and len(names) != 1:
            local.append(f"{tag} {label}: {len(names)} kernels {names}, expected one fused kernel")
            ok = False
        stats["calls"] += 1
        dig.add(got)
        return got, ok

    y0, names0 = _q_kernels(lambda: L(xq, dw, x_scale=sx))
    if _q_tier(names0) != want_tier:
        local.append(f"{tag} identity: ran on {_q_tier(names0)} ({names0}), expected {want_tier}")
    _p_eq(f"{tag} identity == RN_bf16(acc)", y0, torch.where(acc == 0, y0, rn(acc)), local)
    dig.add(y0)
    zpos = (acc == 0) & (bd == 0)
    want_b = torch.where(zpos, y0, rn(acc + bd))
    run("bias (2b)", lambda: L(xq, dw, x_scale=sx, bias=b), want_b, bt, one_kernel=True)
    run("bias (bf16 x)", lambda: L(x, dw, bias=b), want_b, bt)
    stats["double_rounding_differs"] += int((rn(rn(acc).double() + bd) != want_b).sum())
    stats["elements"] += M * N
    run("bias = 0 == bias = None", lambda: L(xq, dw, x_scale=sx, bias=torch.zeros_like(b)), y0, bt)
    run("residual", lambda: L(xq, dw, x_scale=sx, residual=r), rn(acc + rd), ft)
    want_br = rn(acc + bd + rd)
    run("bias + residual (2b)", lambda: L(xq, dw, x_scale=sx, bias=b, residual=r), want_br, ft,
        one_kernel=(ft == "sm100.decode"))
    run("bias + residual (bf16 x)", lambda: L(x, dw, bias=b, residual=r), want_br, ft)
    # plan() must name the tier each call ran on.
    for label, epi, tier in (("bias", _q_epilogue(M, N, bias=b), bt),
                             ("residual", _q_epilogue(M, N, residual=r), ft),
                             ("bias+residual", _q_epilogue(M, N, bias=b, residual=r), ft),
                             ("bias+gelu", _q_epilogue(M, N, bias=b, activation="gelu_erf"), at)):
        p = _sm100_dispatch.plan(M, N, K, epi)
        if p != tier:
            local.append(f"{tag} plan({label}) = {p}, but the call ran on {tier}")

    xs, ws = _p_ints(M, N, K, seed + 7, sparse=True)
    dws = _p_weight(ws)
    xsq, ssx = D.quantize_activation(xs)
    accs = xs.double() @ ws.double().t()
    y0s = L(xsq, dws, x_scale=ssx)
    gl = ext.gelu_ref((accs + bd).float().contiguous())
    gl0 = ext.gelu_ref(accs.float().contiguous())
    zs = (accs == 0) & (bd == 0)
    run("bias + gelu_erf", lambda: L(xsq, dws, x_scale=ssx, bias=b, activation="gelu_erf"),
        torch.where(zs, y0s, gl.bfloat16()), at, one_kernel=(at == "sm100.decode"))
    run("bias + gelu_erf + residual", lambda: L(xsq, dws, x_scale=ssx, bias=b, residual=r, activation="gelu_erf"),
        (gl + r.float()).bfloat16(), at)
    run("gelu_erf (bf16 x)", lambda: L(xs, dws, activation="gelu_erf"), torch.where(accs == 0, y0s, gl0.bfloat16()),
        at)
    stats["gelu_active"] += int(((accs + bd).abs() < 4).sum())
    dig.add(y0s)
    stats["cells"] += 1


def _q_zero_corner(M: int, N: int, K: int, want_tier: str, local: list, stats: dict, key: str,
                   bias_tier: str = None) -> None:
    """x row 0 all zeros and output column 0 with all-negative weight codes: every product is -0.0. bias = 0 must
    give the identity's bytes on the route that owns the cell (the bias call on `bias_tier or want_tier`: issue #12
    T1 sends the bias calls of the short-K narrow-N cells to the cascade while their identity stays on tier 1); the
    observed identity byte is recorded."""
    x, w = _p_ints(M, N, K, 991 + M, sparse=False)
    x[0] = 0
    w[0] = -torch.randint(1, 5, (K,), device="cuda")
    dw = _p_weight(w)
    xq, sx = D.quantize_activation(x)
    y0, names0 = _q_kernels(lambda: D.linear(xq, dw, x_scale=sx))
    yz, namesz = _q_kernels(lambda: D.linear(xq, dw, x_scale=sx, bias=torch.zeros(N, dtype=torch.bfloat16,
                                                                                    device="cuda")))
    stats[key] = f"0x{int(_bits(y0)[0, 0]) & 0xFFFF:04x} / bias=0 0x{int(_bits(yz)[0, 0]) & 0xFFFF:04x}"
    if _q_tier(names0) != want_tier or _q_tier(namesz) != (bias_tier or want_tier):
        local.append(f"-0.0 corner M={M} N={N} K={K}: ran on {_q_tier(names0)} / {_q_tier(namesz)}, expected "
                     f"{want_tier} / {bias_tier or want_tier}")
    _p_eq(f"-0.0 corner M={M} N={N} K={K} ({want_tier}): bias = 0 == bias = None", yz, y0, local)


def _q_two_streams(local: list) -> None:
    """Two different decode-row fused calls (split-K and persistent forms) captured on two streams of one thread and
    replayed concurrently 200 times; each must equal its serial result every time."""
    L = D.linear
    calls = []
    for (N, K, M, seed) in ((1280, 5120, 7, 201), (5120, 1280, 48, 202)):
        dw = prepared(10, N, K, seed=seed)
        x = make_x(M, K, seed=seed + 1)
        xq, sx = D.quantize_activation(x)
        b = torch.randn(N, device="cuda").bfloat16()
        r = torch.randn(M, N, device="cuda").bfloat16()
        calls.append(lambda xq=xq, dw=dw, sx=sx, b=b, r=r: L(xq, dw, x_scale=sx, bias=b, residual=r,
                                                             activation="gelu_erf"))
    refs = [c().clone() for c in calls]
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs, outs = [], []
    for c, s in zip(calls, streams):
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            outs.append(c())
        graphs.append(g)
    torch.cuda.synchronize()
    bad = [torch.zeros((), dtype=torch.int64, device="cuda") for _ in calls]
    for _ in range(200):
        for i in range(2):
            with torch.cuda.stream(streams[i]):
                graphs[i].replay()
                bad[i] += (_bits(outs[i]) != _bits(refs[i])).sum()
    torch.cuda.synchronize()
    nbad = [int(t) for t in bad]
    if any(nbad):
        local.append(f"two-stream concurrent replay (decode row): {nbad} mismatching elements over 200 iterations")


def _q_shortk_two_streams(local: list) -> None:
    """(issue #12, T1) different fused calls on the cascade's short-K tiles (2-SM 256 x 64 with bias + residual,
    1-SM 128 x 128 with bias + erf-GELU, and (T1b) the fused q / k / v 3840 x 1280 at M = 78 with bias on the 1-SM
    128 x 64 tile the empty-pair guard picks) captured on one stream each of one thread and replayed concurrently 200
    times; each must equal its serial result every time."""
    L = D.linear
    calls = []
    for (N, K, M, seed, kw) in ((2048, 1280, 129, 211, dict(residual=True)), (5120, 1280, 364, 212,
                                                                             dict(activation="gelu_erf")),
                                (3840, 1280, 78, 213, dict())):
        dw = prepared(10, N, K, seed=seed)
        x = make_x(M, K, seed=seed + 1)
        xq, sx = D.quantize_activation(x)
        b = torch.randn(N, device="cuda").bfloat16()
        r = torch.randn(M, N, device="cuda").bfloat16() if kw.get("residual") else None
        act = kw.get("activation")
        calls.append(lambda xq=xq, dw=dw, sx=sx, b=b, r=r, act=act: L(xq, dw, x_scale=sx, bias=b, residual=r,
                                                                       activation=act))
    refs = [c().clone() for c in calls]
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream() for _ in calls]
    graphs, outs = [], []
    for c, s in zip(calls, streams):
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            outs.append(c())
        graphs.append(g)
    torch.cuda.synchronize()
    bad = [torch.zeros((), dtype=torch.int64, device="cuda") for _ in calls]
    for _ in range(200):
        for i in range(len(calls)):
            with torch.cuda.stream(streams[i]):
                graphs[i].replay()
                bad[i] += (_bits(outs[i]) != _bits(refs[i])).sum()
    torch.cuda.synchronize()
    nbad = [int(t) for t in bad]
    if any(nbad):
        local.append(f"two-stream concurrent replay (short-K tiles): {nbad} mismatching elements over 200 iterations")


def _q_compile(local: list) -> None:
    N, K, M = 1280, 5120, 16     # a decode-row split-K cell
    dw = prepared(10, N, K, seed=171)
    x = make_x(M, K, seed=172)
    b = torch.randn(N, device="cuda").bfloat16()
    r = torch.randn(M, N, device="cuda").bfloat16()

    def f(x):
        return D.linear(x, dw, bias=b, residual=r, activation="gelu_erf")

    want = f(x)
    graphs = []

    def record(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    got = torch.compile(f, backend=record, fullgraph=True)(x)
    nodes = [str(n.target) for gm in graphs for n in gm.graph.nodes
             if n.op == "call_function" and "fish_scales_ops" in str(n.target)]
    if not (torch.equal(got, want) and len(nodes) == 1 and "dense_linear" in nodes[0]):
        local.append(f"torch.compile recording backend (decode row): nodes {nodes}, equal {torch.equal(got, want)}")
    torch._dynamo.reset()
    got = torch.compile(f, fullgraph=True)(x)
    torch.cuda.synchronize()
    if not torch.equal(got, want):
        local.append("torch.compile inductor (decode row): compiled != eager")
    torch._dynamo.reset()


def _q_agree(local: list, stats: dict) -> None:
    """plan() must name the tier route() runs on, for every cell of the grids and four epilogues."""
    from fish_scales_ops.gemm import _sm100_dispatch
    L = D.linear
    seen = {}
    for (N, K) in Q_AGREE_SHAPES:
        dw = prepared(10, N, K, seed=N + K)
        for M in Q_AGREE_MS:
            x = make_x(M, K, seed=M + 3)
            xq, sx = D.quantize_activation(x)
            b = torch.randn(N, device="cuda").bfloat16()
            r = torch.randn(M, N, device="cuda").bfloat16()
            variants = (("identity", None, lambda: L(xq, dw, x_scale=sx)),
                        ("bias", _q_epilogue(M, N, bias=b), lambda: L(xq, dw, x_scale=sx, bias=b)),
                        ("residual", _q_epilogue(M, N, residual=r), lambda: L(xq, dw, x_scale=sx, residual=r)),
                        ("bias+gelu+residual", _q_epilogue(M, N, bias=b, residual=r, activation="gelu_erf"),
                         lambda: L(xq, dw, x_scale=sx, bias=b, residual=r, activation="gelu_erf")))
            for label, epi, fn in variants:
                planned = _sm100_dispatch.plan(M, N, K, epi)
                y, names = _q_kernels(fn)
                ran = _q_tier(names)
                stats["agree_calls"] += 1
                seen[(planned, label)] = seen.get((planned, label), 0) + 1
                if ran != planned:
                    local.append(f"agree M={M} N={N} K={K} {label}: plan {planned}, ran {ran} ({names})")
                if not bool(torch.isfinite(y).all()):
                    local.append(f"agree M={M} N={N} K={K} {label}: non-finite output")
    stats["agree_tiers"] = {f"{k[0]}/{k[1]}": v for k, v in sorted(seen.items())}


def _q_pdl(local: list, stats: dict) -> None:
    """A producer that releases its dependents at once and writes the bias and residual 2 ms later, directly ahead
    of a fused call on each route; the call must give the bytes of the same call on the final tensors."""
    ext = _pdl_probe()
    spin = 2_000_000
    t0, t1 = torch.ones(256, dtype=torch.int16, device="cuda"), torch.zeros(256, dtype=torch.int16, device="cuda")
    ext.slow_copy2(t0, t1, t0, t1, 1, 1000)
    ext.early_read(t1, torch.zeros(1, dtype=torch.int32, device="cuda"))
    torch.cuda.synchronize()
    # (N, K, M, epilogue kwargs, expected tier)
    cells = ((1280, 5120, 7, dict(bias=True, residual=True, activation="gelu_erf"), "sm100.decode"),
             (5120, 1280, 48, dict(bias=True, residual=True, activation="gelu_erf"), "sm100.decode"),
             (5120, 1280, 182, dict(bias=True, residual=False, activation=None), "sm100.smm"),
             (2048, 1280, 129, dict(bias=True, residual=True, activation=None), "sm100.cascade"),
             # (T1) a bias-only call of a narrow-N short-K cell tier 1 declines: the cascade's short-K tile
             (2048, 1280, 129, dict(bias=True, residual=False, activation=None), "sm100.cascade"),
             # (FB2) the narrow-N short-K band's split-K form, and an erf-GELU call the row takes from the cascade
             (1280, 1280, 78, dict(bias=True, residual=True, activation=None), "sm100.decode"),
             (5120, 1280, 182, dict(bias=True, residual=False, activation="gelu_erf"), "sm100.decode"),
             # (P5) the narrow-N long-K band: its split-K form, and the fall-through of a tier-1 cell onto the row
             (2048, 4096, 129, dict(bias=True, residual=True, activation="gelu_erf"), "sm100.decode"),
             (1280, 5120, 1079, dict(bias=True, residual=True, activation=None), "sm100.decode"))
    for N, K, M, spec, tier in cells:
        dw = prepared(10, N, K, seed=301 + N + M)
        x = make_x(M, K, seed=302 + M)
        xq, sx = D.quantize_activation(x)
        b = (torch.randn(N, device="cuda") * 0.1 + 0.5).bfloat16()
        r = (torch.randn(M, N, device="cuda") + 4.0).bfloat16()
        db, dr = torch.zeros_like(b), torch.zeros_like(r)

        def call(bb, rr, entry):
            kw = dict(bias=bb if spec["bias"] else None, residual=rr if spec["residual"] else None,
                      activation=spec["activation"])
            return D.linear(x, dw, **kw) if entry == "bf16-x" else D.linear(xq, dw, x_scale=sx, **kw)

        for entry in ("bf16-x", "2b"):
            want, names = _q_kernels(lambda: call(b, r, entry))
            if _q_tier(names) != tier:
                local.append(f"pdl M={M} N={N} K={K} {entry}: ran on {_q_tier(names)}, expected {tier}")
            call(db, dr, entry)
            stale = torch.zeros(1, dtype=torch.int32, device="cuda")
            db.zero_(), dr.zero_()
            torch.cuda.synchronize()
            ext.slow_copy2(b.view(torch.int16), db.view(torch.int16), r.view(torch.int16), dr.view(torch.int16), 8,
                           spin)
            ext.early_read(db.view(torch.int16), stale)
            torch.cuda.synchronize()
            n_stale = int(stale.item())
            db.zero_(), dr.zero_()
            torch.cuda.synchronize()
            ext.slow_copy2(b.view(torch.int16), db.view(torch.int16), r.view(torch.int16), dr.view(torch.int16), 8,
                           spin)
            got = call(db, dr, entry)
            torch.cuda.synchronize()
            if not bytes_equal(got, want):
                local.append(f"pdl M={M} N={N} K={K} {entry} ({tier}): behind a late producer != on the final "
                             f"tensors")
            if n_stale == 0:
                local.append(f"pdl M={M} N={N} K={K} {entry}: the control saw no stale element (no overlap)")
            stats["pdl_cells"] += 1
            stats["pdl_stale_min"] = min(stats.get("pdl_stale_min", 1 << 30), n_stale)


def _q_wide_forced(M: int, N: int, K: int, cfg, ext, dig: _PDigest, local: list, stats: dict) -> None:
    """(P5) One tactic of the swap-AB kernels' tile list forced through linear_mxfp8_decode on the exact construction:
    identity, bias, residual, bias + residual, bias = 0, bias + erf-GELU (+ residual) against RN_bf16 of the exact
    accumulator (GELU through the load_inline reference)."""
    from fish_scales_ops.gemm import _sm100_decode
    tag = f"forced {cfg} M={M} N={N} K={K}"
    x, w = _p_ints(M, N, K, M * 1009 + N * 17 + K + 3, sparse=False)
    dw = _p_weight(w)
    xq, sx = D.quantize_activation(x)
    acc = x.double() @ w.double().t()
    g = torch.Generator(device="cuda").manual_seed(M + N + 1)
    b = (torch.randint(-64, 65, (N,), generator=g, device="cuda").double() / 8).bfloat16()
    r = (torch.randint(-1023, 1024, (M, N), generator=g, device="cuda").double() / 8).bfloat16()
    bd, rd = b.double()[None, :], r.double()
    rn = lambda v: v.float().bfloat16()  # noqa: E731

    def run(epi):
        y = _sm100_decode.linear_mxfp8_decode(xq, dw.weight, sx, dw.scale, cfg, epi)
        if y is None:
            local.append(f"{tag}: linear_mxfp8_decode declined")
            return torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        stats["calls"] += 1
        dig.add(y)
        return y

    y0 = run(None)
    _p_eq(f"{tag} identity == RN_bf16(acc)", y0, torch.where(acc == 0, y0, rn(acc)), local)
    zpos = (acc == 0) & (bd == 0)
    _p_eq(f"{tag} bias", run(_q_epilogue(M, N, bias=b)), torch.where(zpos, y0, rn(acc + bd)), local)
    _p_eq(f"{tag} residual", run(_q_epilogue(M, N, residual=r)), rn(acc + rd), local)
    _p_eq(f"{tag} bias + residual", run(_q_epilogue(M, N, bias=b, residual=r)), rn(acc + bd + rd), local)
    _p_eq(f"{tag} bias = 0 == bias = None", run(_q_epilogue(M, N, bias=torch.zeros_like(b))), y0, local)
    stats["double_rounding_differs"] += int((rn(rn(acc).double() + bd) != rn(acc + bd)).sum())
    stats["elements"] += M * N
    xs, ws = _p_ints(M, N, K, M * 1009 + N * 17 + K + 10, sparse=True)
    dws = _p_weight(ws)
    xsq, ssx = D.quantize_activation(xs)
    accs = xs.double() @ ws.double().t()
    gl = ext.gelu_ref((accs + bd).float().contiguous())
    yg = _sm100_decode.linear_mxfp8_decode(xsq, dws.weight, ssx, dws.scale, cfg,
                                           _q_epilogue(M, N, bias=b, activation="gelu_erf"))
    y0s = _sm100_decode.linear_mxfp8_decode(xsq, dws.weight, ssx, dws.scale, cfg)
    zs = (accs == 0) & (bd == 0)
    _p_eq(f"{tag} bias + gelu_erf", yg, torch.where(zs, y0s, gl.bfloat16()), local)
    ygr = _sm100_decode.linear_mxfp8_decode(xsq, dws.weight, ssx, dws.scale, cfg,
                                            _q_epilogue(M, N, bias=b, residual=r, activation="gelu_erf"))
    _p_eq(f"{tag} bias + gelu_erf + residual", ygr, (gl + r.float()).bfloat16(), local)
    stats["gelu_active"] += int(((accs + bd).abs() < 4).sum())
    for t in (yg, y0s, ygr):
        dig.add(t)
    stats["forced"] += 1


def _q_wide_determinism(local: list, stats: dict) -> None:
    """(P5) Each new tactic of the band (and the fall-through form) captured once with a fused call, then replayed 200
    times: every replay is bit-identical to the first (the fixed-order cluster sum and the persistent schedule)."""
    for (N, K, M, cfg) in Q_WIDE_CELLS[1:2] + Q_WIDE_CELLS[3:4] + Q_WIDE_CELLS[5:6] + Q_WIDE_FALLTHROUGH:
        dw = prepared(10, N, K, seed=401 + M)
        x = make_x(M, K, seed=402 + M)
        xq, sx = D.quantize_activation(x)
        b = torch.randn(N, device="cuda").bfloat16()
        r = torch.randn(M, N, device="cuda").bfloat16()
        fn = lambda: D.linear(xq, dw, x_scale=sx, bias=b, residual=r)  # noqa: E731
        ref, names = _q_kernels(fn)
        if _q_tier(names) != "sm100.decode":
            local.append(f"determinism M={M} N={N} K={K}: ran on {_q_tier(names)} ({names})")
        ref = ref.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            out = fn()
        torch.cuda.synchronize()
        bad = torch.zeros((), dtype=torch.int64, device="cuda")
        for _ in range(200):
            g.replay()
            bad += (_bits(out) != _bits(ref)).sum()
        torch.cuda.synchronize()
        if int(bad):
            local.append(f"determinism {cfg} M={M} N={N} K={K}: {int(bad)} mismatching elements over 200 replays")
        stats["determinism_cells"] += 1


def _q_wide_two_streams(local: list) -> None:
    """(P5) Two different fused calls of the band (split-K form and persistent form) captured on two streams of one
    thread and replayed concurrently 200 times; each must equal its serial result every time."""
    L = D.linear
    calls = []
    for (N, K, M, seed) in ((1280, 5120, 129, 501), (2048, 4096, 384, 502)):
        dw = prepared(10, N, K, seed=seed)
        x = make_x(M, K, seed=seed + 1)
        xq, sx = D.quantize_activation(x)
        b = torch.randn(N, device="cuda").bfloat16()
        r = torch.randn(M, N, device="cuda").bfloat16()
        calls.append(lambda xq=xq, dw=dw, sx=sx, b=b, r=r: L(xq, dw, x_scale=sx, bias=b, residual=r,
                                                             activation="gelu_erf"))
    refs = [c().clone() for c in calls]
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs, outs = [], []
    for c, st in zip(calls, streams):
        st.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=st):
            outs.append(c())
        graphs.append(g)
    torch.cuda.synchronize()
    bad = [torch.zeros((), dtype=torch.int64, device="cuda") for _ in calls]
    for _ in range(200):
        for i in range(2):
            with torch.cuda.stream(streams[i]):
                graphs[i].replay()
                bad[i] += (_bits(outs[i]) != _bits(refs[i])).sum()
    torch.cuda.synchronize()
    nbad = [int(t) for t in bad]
    if any(nbad):
        local.append(f"two-stream concurrent replay (M > 64 band): {nbad} mismatching elements over 200 iterations")


def _q_wide(ext, dig: _PDigest, local: list, stats: dict) -> None:
    """(P5) The wide family: see the family list above case q."""
    from fish_scales_ops.gemm import _sm100_decode
    from fish_scales_ops.gemm import _epilogue as E, _sm100_dispatch
    for (N, K, M, cfg) in Q_WIDE_CELLS + Q_WIDE_FALLTHROUGH:
        # linear_quantized is the cascade's alone on sm_100: the row does not claim the quantized output, so a
        # quantized call of the band (and of its fall-through cell) is planned on the cascade (case s runs it).
        for kw in ({}, {"activation": "gelu_erf"}):
            eq = E.make("case q", m=M, n=N, device=torch.device("cuda", torch.cuda.current_device()), quantized=True,
                        bias=torch.zeros(N, dtype=torch.bfloat16, device="cuda"), **kw)
            p = _sm100_dispatch.plan(M, N, K, eq)
            if p != "sm100.cascade":
                local.append(f"wide M={M} N={N} K={K}: plan(linear_quantized {kw}) = {p}, expected sm100.cascade")
    for (N, K, M, cfg) in Q_WIDE_CELLS:
        got = _sm100_decode.pick_config(M, N, K)
        if got != cfg:
            local.append(f"wide M={M} N={N} K={K}: pick_config {got}, expected {cfg}")
        _q_cell(M, N, K, M * 1009 + N * 17 + K, ext, dig, local, stats, "sm100.decode")
    # The band's top: tier 1 keeps o at M = 640 (the 128-wide persistent form is not used on the band, see
    # _narrow_long_k_config) and fc2 at M = 1079 (90 cuBLAS CTAs); a K below 4096 and a wide N are not the band.
    for (N, K, M) in ((2048, 4096, 640), (1280, 5120, 1079), (2560, 4096, 512), (3840, 1280, 182), (5120, 4096, 182)):
        got = _sm100_decode.pick_config(M, N, K)
        if got is not None:
            local.append(f"wide M={M} N={N} K={K}: pick_config {got}, expected None (not the band)")
    for (N, K, M, cfg) in Q_WIDE_FALLTHROUGH:
        got, ft = _sm100_decode.pick_config(M, N, K), _sm100_decode.pick_fallthrough_config(M, N, K)
        if got is not None or ft != cfg:
            local.append(f"fall-through M={M} N={N} K={K}: pick_config {got} (expected None), "
                         f"pick_fallthrough_config {ft} (expected {cfg})")
        _q_cell(M, N, K, M * 1009 + N * 17 + K, ext, dig, local, stats, "sm100.smm", "sm100.decode")
    for (N, K, M) in Q_WIDE_FORCED_CELLS:
        for cfg in Q_WIDE_FORCED_TACTICS:
            _q_wide_forced(M, N, K, cfg, ext, dig, local, stats)
    _q_zero_corner(78, 1280, 5120, "sm100.decode", local, stats, "zero_byte_wide_splitk")
    _q_zero_corner(384, 2048, 4096, "sm100.decode", local, stats, "zero_byte_wide_persistent")
    for (N, K, M, seed) in ((1280, 5120, 129, 31), (2048, 4096, 576, 32), (1280, 5120, 1079, 33)):
        x, w = _p_ints(M, N, K, seed, sparse=False)
        dw = _p_weight(w)
        xq, sx = D.quantize_activation(x)
        b = torch.randn(N, device="cuda").bfloat16()
        r = torch.randn(M, N, device="cuda").bfloat16()
        _p_capture(f"wide M={M} N={N} K={K} bias + gelu_erf + residual",
                   lambda: D.linear(xq, dw, x_scale=sx, bias=b, residual=r, activation="gelu_erf"), local)
        _p_capture(f"wide M={M} N={N} K={K} bias + residual, bf16 x", lambda: D.linear(x, dw, bias=b, residual=r),
                   local)
    _q_wide_determinism(local, stats)
    _q_wide_two_streams(local)


# (FB2) the tactic each Q_FB2_CELLS cell's swap-AB calls must run: (N, K, M) -> (picker, tactic). "config" is
# pick_config (the cell is the row's by shape, every call); "identity" pick_identity_config and "activation"
# pick_activation_config (the row takes only those calls of the cell).
_FB2_T64 = lambda s: ((128, 64), (1, 1), True, s)  # noqa: E731
_FB2_P128 = ((128, 64), (1, 1), True, 1)
Q_FB2_TACTICS = {(1280, 5120, 33): ("config", _FB2_T64(4)), (1280, 5120, 48): ("config", _FB2_T64(4)),
                 (1280, 5120, 64): ("config", _FB2_T64(4)), (2048, 4096, 48): ("config", _FB2_T64(2)),
                 (2560, 4096, 40): ("config", _FB2_T64(2)), (1280, 1280, 65): ("config", _FB2_T64(2)),
                 (1280, 1280, 129): ("config", _FB2_T64(2)), (1280, 1280, 256): ("config", _FB2_T64(2)),
                 (2048, 1280, 117): ("config", _FB2_T64(2)), (1024, 2048, 320): ("config", _FB2_T64(2)),
                 (2560, 2048, 96): ("config", _FB2_T64(2)), (2048, 2048, 80): ("config", _FB2_T64(2)),
                 (5120, 2048, 96): ("identity", _FB2_P128), (9216, 2048, 128): ("identity", _FB2_P128),
                 (5120, 1280, 78): ("identity", _FB2_P128), (5120, 1280, 182): ("identity", _FB2_P128),
                 (6144, 2560, 160): ("identity", _FB2_P128), (2048, 1280, 533): ("activation", _FB2_P128),
                 (2560, 1280, 448): ("activation", _FB2_P128)}
# (FB2) the rules' edges: (N, K, M, picker, expected) on cells one step outside (None) and inside each rule.
Q_FB2_EDGES = (
    # the narrow-N short-K band: 80 CTAs of the split-K grid at most
    (1280, 1280, 256, "config", _FB2_T64(2)), (1280, 1280, 257, "config", None),
    (2048, 1280, 128, "config", _FB2_T64(2)), (2048, 1280, 129, "config", None),
    (1024, 2048, 320, "config", _FB2_T64(2)), (1024, 2048, 321, "config", None),
    (3840, 1280, 65, "config", None), (2048, 512, 128, "config", None), (2048, 1152, 128, "config", None),
    # the narrow-N long-K class at 32 < M <= 64: the cascade's two-kernel split-K keeps K >= 8192; a K no split divides
    # keeps the wide-token form
    (2560, 9728, 48, "config", None), (2560, 8064, 40, "config", ((256, 64), (2, 1), True, 1)),
    # identity calls of wide-N short-K cells: the 1-SM grid fits one wave; K <= 2560; narrow N is not the rule's
    (5120, 2048, 192, "identity", _FB2_P128), (5120, 2048, 193, "identity", None),
    (9216, 2048, 128, "identity", _FB2_P128), (9216, 2048, 129, "identity", None),
    (12288, 2048, 96, "identity", None), (5120, 4096, 96, "identity", None), (4096, 2048, 96, "identity", None),
    (5120, 896, 96, "identity", None),
    # erf-GELU calls: the cascade's 1-SM 128 x 64 grid overflows one wave while the row's grid fits
    (5120, 1280, 128, "activation", None), (5120, 1280, 129, "activation", _FB2_P128),
    (5120, 1280, 192, "activation", _FB2_P128), (5120, 1280, 193, "activation", None),
    (2048, 1280, 512, "activation", None), (2048, 1280, 513, "activation", _FB2_P128),
    (2048, 1280, 576, "activation", _FB2_P128), (2048, 1280, 577, "activation", None),
    (1280, 1280, 1079, "activation", None), (5120, 4096, 182, "activation", None))


def _q_fb2(ext, dig: _PDigest, local: list, stats: dict) -> None:
    """(FB2) The route rules of the B300 swap-AB adoption: see Q_FB2_CELLS and the family list above case q."""
    from fish_scales_ops.gemm import _sm100_decode
    from fish_scales_ops.gemm import _epilogue as E, _sm100_dispatch
    pickers = {"config": _sm100_decode.pick_config, "identity": _sm100_decode.pick_identity_config,
               "activation": _sm100_decode.pick_activation_config}
    for (N, K, M, picker, want) in Q_FB2_EDGES:
        got = pickers[picker](M, N, K)
        if got != want:
            local.append(f"fb2 edge M={M} N={N} K={K}: {picker} picker {got}, expected {want}")
    for (N, K, M, t_id, t_bias, t_res, t_act) in Q_FB2_CELLS:
        picker, want = Q_FB2_TACTICS[(N, K, M)]
        got = pickers[picker](M, N, K)
        if got != want:
            local.append(f"fb2 M={M} N={N} K={K}: {picker} picker {got}, expected {want}")
        if picker != "config" and _sm100_decode.pick_config(M, N, K) is not None:
            local.append(f"fb2 M={M} N={N} K={K}: pick_config owns a cell the rule takes only by call")
        # linear_quantized stays on the cascade, which alone writes the quantized output
        eq = E.make("case q", m=M, n=N, device=torch.device("cuda", torch.cuda.current_device()), quantized=True,
                    bias=torch.zeros(N, dtype=torch.bfloat16, device="cuda"), activation="gelu_erf")
        if _sm100_dispatch.plan(M, N, K, eq) != "sm100.cascade":
            local.append(f"fb2 M={M} N={N} K={K}: linear_quantized not planned on the cascade")
        _q_cell(M, N, K, M * 1009 + N * 17 + K, ext, dig, local, stats, t_id, t_res, bias_tier=t_bias, act_tier=t_act)
    _q_zero_corner(78, 1280, 1280, "sm100.decode", local, stats, "zero_byte_fb2_shortk")
    _q_zero_corner(96, 5120, 2048, "sm100.decode", local, stats, "zero_byte_fb2_identity", bias_tier="sm100.cascade")
    _q_zero_corner(48, 2048, 4096, "sm100.decode", local, stats, "zero_byte_fb2_split64")
    calls = []
    for (N, K, M, seed, kw) in ((1280, 1280, 182, 51, dict(residual=True)), (5120, 1280, 182, 52,
                                                                            dict(activation="gelu_erf")),
                                (2048, 4096, 48, 53, dict(residual=True, activation="gelu_erf"))):
        x, w = _p_ints(M, N, K, seed, sparse=False)
        dw = _p_weight(w)
        xq, sx = D.quantize_activation(x)
        b = torch.randn(N, device="cuda").bfloat16()
        r = torch.randn(M, N, device="cuda").bfloat16() if kw.get("residual") else None
        act = kw.get("activation")
        _p_capture(f"fb2 M={M} N={N} K={K} {kw} (2b)",
                   lambda: D.linear(xq, dw, x_scale=sx, bias=b, residual=r, activation=act), local)
        _p_capture(f"fb2 M={M} N={N} K={K} {kw} (bf16 x)", lambda: D.linear(x, dw, bias=b, residual=r, activation=act),
                   local)
        calls.append(lambda xq=xq, dw=dw, sx=sx, b=b, r=r, act=act: D.linear(xq, dw, x_scale=sx, bias=b, residual=r,
                                                                             activation=act))
    # identity on the 1-SM persistent form (no epilogue tensors)
    x, w = _p_ints(160, 6144, 2560, 54, sparse=False)
    dw = _p_weight(w)
    xq, sx = D.quantize_activation(x)
    _p_capture("fb2 M=160 N=6144 K=2560 identity (bf16 x)", lambda: D.linear(x, dw), local)
    calls.append(lambda xq=xq, dw=dw, sx=sx: D.linear(xq, dw, x_scale=sx))
    # 200 concurrent replays of the four captured calls on four streams, each against its serial result
    refs = [c().clone() for c in calls]
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream() for _ in calls]
    graphs, outs = [], []
    for c, s in zip(calls, streams):
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            outs.append(c())
        graphs.append(g)
    torch.cuda.synchronize()
    bad = [torch.zeros((), dtype=torch.int64, device="cuda") for _ in calls]
    for _ in range(200):
        for i in range(len(calls)):
            with torch.cuda.stream(streams[i]):
                graphs[i].replay()
                bad[i] += (_bits(outs[i]) != _bits(refs[i])).sum()
    torch.cuda.synchronize()
    nbad = [int(t) for t in bad]
    stats["determinism_cells"] += len(calls)
    if any(nbad):
        local.append(f"fb2 concurrent replay on four streams: {nbad} mismatching elements over 200 iterations")


def q_worker(label: str) -> int:
    import json
    local: list[str] = []
    stats = {"cells": 0, "calls": 0, "elements": 0, "double_rounding_differs": 0, "gelu_active": 0,
             "agree_calls": 0, "pdl_cells": 0, "forced": 0, "determinism_cells": 0}
    dig = _PDigest()
    torch.backends.cuda.matmul.allow_tf32 = False
    if label == "decode":
        ext = _sm100_gelu_ext()
        for (N, K) in Q_SPLIT_SHAPES:
            for M in Q_SPLIT_MS:
                _q_cell(M, N, K, M * 1009 + N * 17 + K, ext, dig, local, stats, "sm100.decode")
        for (N, K, M) in Q_PERSIST_CELLS:
            _q_cell(M, N, K, M * 1009 + N * 17 + K, ext, dig, local, stats, "sm100.decode")
        _q_zero_corner(7, 1280, 5120, "sm100.decode", local, stats, "zero_byte_splitk")
        _q_zero_corner(48, 5120, 1280, "sm100.decode", local, stats, "zero_byte_persistent")
        for (N, K, M, seed) in ((1280, 5120, 7, 11), (5120, 1280, 48, 12), (2048, 4096, 32, 13)):
            x, w = _p_ints(M, N, K, seed, sparse=False)
            dw = _p_weight(w)
            xq, sx = D.quantize_activation(x)
            b = torch.randn(N, device="cuda").bfloat16()
            r = torch.randn(M, N, device="cuda").bfloat16()
            _p_capture(f"decode M={M} N={N} K={K} bias + gelu_erf + residual",
                       lambda: D.linear(xq, dw, x_scale=sx, bias=b, residual=r, activation="gelu_erf"), local)
            _p_capture(f"decode M={M} N={N} K={K} bias, bf16 x", lambda: D.linear(x, dw, bias=b), local)
        _q_two_streams(local)
        _q_compile(local)
    elif label == "smm":
        ext = _sm100_gelu_ext()
        for (N, K, M) in Q_SMM_CELLS:
            _q_cell(M, N, K, M * 1009 + N * 17 + K, ext, dig, local, stats, "sm100.smm", "sm100.cascade")
        _q_zero_corner(364, 1280, 1280, "sm100.smm", local, stats, "zero_byte_smm")
        _q_zero_corner(208, 5120, 1280, "sm100.smm", local, stats, "zero_byte_smm_wide")
        for (N, K, M, seed) in ((5120, 1280, 182, 21), (1280, 1280, 364, 22)):
            x, w = _p_ints(M, N, K, seed, sparse=False)
            dw = _p_weight(w)
            xq, sx = D.quantize_activation(x)
            b = torch.randn(N, device="cuda").bfloat16()
            r = torch.randn(M, N, device="cuda").bfloat16()
            _p_capture(f"smm M={M} N={N} K={K} bias", lambda: D.linear(xq, dw, x_scale=sx, bias=b), local)
            _p_capture(f"fall-through M={M} N={N} K={K} bias + residual, bf16 x",
                       lambda: D.linear(x, dw, bias=b, residual=r), local)
    elif label == "shortk":
        ext = _sm100_gelu_ext()
        for (N, K, M) in Q_SHORTK_DECLINE_CELLS:
            _q_cell(M, N, K, M * 1009 + N * 17 + K, ext, dig, local, stats, "sm100.smm", "sm100.cascade",
                    bias_tier="sm100.cascade")
            _q_shortk_tiles(M, N, K, local, stats)
        for (N, K, M) in Q_SHORTK_EPI_CELLS:
            _q_cell(M, N, K, M * 1009 + N * 17 + K, ext, dig, local, stats, "sm100.smm", "sm100.cascade")
            _q_shortk_tiles(M, N, K, local, stats)
        for (N, K, M) in Q_SHORTK_WAVE_CELLS:
            _q_cell(M, N, K, M * 1009 + N * 17 + K, ext, dig, local, stats, "sm100.cascade")
            _q_shortk_tiles(M, N, K, local, stats)
        _q_zero_corner(129, 2048, 1280, "sm100.smm", local, stats, "zero_byte_shortk", bias_tier="sm100.cascade")
        for (N, K, M, seed) in ((2048, 1280, 129, 41), (5120, 1280, 364, 42), (1280, 1280, 1079, 43),
                                (3840, 1280, 78, 44)):
            x, w = _p_ints(M, N, K, seed, sparse=False)
            dw = _p_weight(w)
            xq, sx = D.quantize_activation(x)
            b = torch.randn(N, device="cuda").bfloat16()
            r = torch.randn(M, N, device="cuda").bfloat16()
            _p_capture(f"shortk M={M} N={N} K={K} bias + gelu_erf",
                       lambda: D.linear(xq, dw, x_scale=sx, bias=b, activation="gelu_erf"), local)
            _p_capture(f"shortk M={M} N={N} K={K} bias + residual, bf16 x", lambda: D.linear(x, dw, bias=b, residual=r),
                       local)
            _p_capture(f"shortk M={M} N={N} K={K} bias, 2b", lambda: D.linear(xq, dw, x_scale=sx, bias=b), local)
        _q_shortk_two_streams(local)
    elif label == "wide":
        ext = _sm100_gelu_ext()
        _q_wide(ext, dig, local, stats)
    elif label == "fb2":
        ext = _sm100_gelu_ext()
        _q_fb2(ext, dig, local, stats)
    elif label == "agree":
        _q_agree(local, stats)
    elif label == "pdl":
        _q_pdl(local, stats)
    print(json.dumps({"label": label, "failures": local, "digest": dig.hex(), **stats}))
    return 0


def _q_spawn(label: str, extra_env: dict):
    import json
    import subprocess
    env = {k: v for k, v in os.environ.items() if not k.startswith("FSO_")}
    env.update(Q_ENV)
    env.update(extra_env)
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--q-worker", label], env=env,
                       capture_output=True, text=True, timeout=7200)
    summary = None
    for line in reversed(r.stdout.splitlines()):
        if line.startswith("{"):
            summary = json.loads(line)
            break
    return r, summary


def case_q(major: int) -> None:
    import re
    if major != 10:
        print("  q  SKIP: the decode row and tier 1 are sm_100/sm_103-only")
        return
    from fish_scales_ops.gemm import _sm100_decode
    if not _sm100_decode.enabled():
        print("  q  SKIP: the CuTe-DSL decode row is not available in this environment")
        return
    runs = [("decode", {}), ("smm", {}), ("wide", {}), ("shortk", {}), ("fb2", {}), ("agree", {}), ("pdl", {}),
            ("decode/nopdl", {"FSO_DISABLE_PDL": "1"}), ("smm/nopdl", {"FSO_DISABLE_PDL": "1"}),
            ("wide/nopdl", {"FSO_DISABLE_PDL": "1"}), ("shortk/nopdl", {"FSO_DISABLE_PDL": "1"}),
            ("fb2/nopdl", {"FSO_DISABLE_PDL": "1"})]
    digests = {}
    tactic_re = re.compile(r"^\[fso sm100 decode(?: fall-through| identity| activation)?\] M=(\d+) N=(\d+) K=(\d+) "
                           r"-> (\S+) tiler=\((\d+), (\d+)\)")
    for label, env in runs:
        worker = label.split("/")[0]
        r, summary = _q_spawn(worker, env)
        ok = r.returncode == 0 and summary is not None and not summary["failures"]
        note = ""
        if summary is not None:
            if summary.get("cells"):
                tactics = sorted({(m.group(4), m.group(5), m.group(6)) for m in map(tactic_re.match,
                                                                                    r.stderr.splitlines()) if m})
                note = (f"; {summary['cells']} cells, {summary['calls']} fused calls, double rounding differs on "
                        f"{summary['double_rounding_differs']} of {summary['elements']} elements, "
                        f"{summary['gelu_active']} GELU inputs in |v| < 4; decode tactics "
                        f"{['%s %sx%s' % t for t in tactics]}; -0.0 corner "
                        f"{[f'{k}: {v}' for k, v in summary.items() if k.startswith('zero_byte')]}")
                if summary.get("shortk_tiles"):
                    note += f"; {summary['shortk_tiles']} cascade calls on the short-K tile the rule names"
                if summary.get("forced") or summary.get("determinism_cells"):
                    note += (f"; {summary['forced']} forced tactic cells, {summary['determinism_cells']} captured "
                             f"tactics replayed 200 x bit-identically")
                if summary["double_rounding_differs"] == 0:
                    ok = False
                    note += " (the test does not discriminate a double rounding)"
            if summary.get("agree_calls"):
                note = f"; {summary['agree_calls']} calls, tiers {summary.get('agree_tiers')}"
            if summary.get("pdl_cells"):
                note = f"; {summary['pdl_cells']} cells x 2 entries, min stale elements seen by the control {summary.get('pdl_stale_min')}"
            digests[label] = summary["digest"]
        if not ok:
            detail = summary["failures"][:10] if summary else r.stdout[-1500:] + r.stderr[-1500:]
            _check(False, f"q {label}: {detail}")
        print(f"  q  {label:12s} {_ok(ok)}{note}")
        if not ok and summary is not None:
            for f in summary["failures"][:10]:
                print(f"       {f}")
    for k in ("decode", "smm", "wide", "shortk", "fb2"):
        same = k in digests and digests.get(k) == digests.get(f"{k}/nopdl")
        _check(same, f"q: FSO_DISABLE_PDL=1 changed the bytes of family {k}")
        print(f"  q  PDL on / off, family {k}: identical bytes  {_ok(same)}")
# --- (r) sm_120/sm_121: linear_quantized, the quantized-output epilogue (S3c) --------------------------------------
#
# One subprocess per route, like case (o): FSO_FORCE_TILE / FSO_FORCE_KSPLIT / FSO_DISABLE_PDL are read once per
# process. Random data: the fused call must reproduce, bit for bit, the unfused pair it replaces,
# quantize_activation(linear(x, w, bias=, activation=)), on xq and on every scale word of the sm_120 contract layout
# (the padded rows' words included), on the bf16-x and on the 2b entry; then the chained GEMM
# linear(xq_next, w2, x_scale=scale_next) must equal linear(y, w2), y being the unfused bf16 result. The natural
# families run with FSO_SMALLM_ROUTE=0, so that the bf16 call of a decode-band cell runs on the cascade too, the route
# linear_quantized always takes (the forced families switch the small-M route off through their force knobs); case s
# compares linear_quantized with the bf16 call on the small-M route and counts the codes that differ.

S3C_SHAPES = ((5120, 1280), (1280, 1280), (2048, 1280), (1280, 5120))  # N x K: fc1, proj1, 2048 x 1280, fc2
S3C_MS = (1, 7, 78, 129, 533, 1079)
# Shapes the natural cascade splits K on (narrow N with long K; tiles_n >= 32 at M <= 32): the quantized reduce.
S3C_STREAMK_NATURAL = ((1, 2560, 9728), (16, 2560, 9728), (129, 2560, 9728), (20, 4096, 4096), (32, 6144, 5120))
# The forced tiles' grid: the M set on two N x K and one wider cell.
S3C_FORCED_GRID = [(M, 1280, K) for K in (1280, 5120) for M in S3C_MS] + [(129, 5120, 1280)]
# (label, bias, activation): linear_quantized takes no residual.
S3C_VARIANTS = (("identity", False, None), ("bias", True, None), ("gelu", False, "gelu_erf"),
                ("bias+gelu", True, "gelu_erf"))
S3C_N2 = 1280  # out_features of the chained second GEMM (its K is the first GEMM's N)


def _s3c_digest(*ts: torch.Tensor) -> str:
    import hashlib
    h = hashlib.sha256()
    for t in ts:
        h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


class _S3cWorker:
    def __init__(self):
        self.failures: list[str] = []
        self.digests: dict[str, str] = {}
        self.cells = 0

    def check(self, ok: bool, msg: str) -> bool:
        if not ok:
            self.failures.append(msg)
        return ok

    def cell(self, M: int, N: int, K: int, want_tile=None, want_split=None, label: str = "") -> None:
        """Every variant on one cell, both entries: the fused call == quantize_activation(the unfused call) on xq and
        on every scale word; the chained GEMM on the fused outputs == the chained GEMM on the unfused bf16 result."""
        route = _s1c_route(M, N, K)
        tag = f"{label} M={M} N={N} K={K} route {route['tile']} k_split {route['k_split']}"
        if want_tile is not None:
            self.check(route["tile"] == tuple(want_tile), f"r {tag}: the route is not the forced tile {want_tile}")
        if want_split is not None:
            ok = route["k_split"] > 1 if want_split == "split" else route["k_split"] == want_split
            self.check(ok, f"r {tag}: K split {route['k_split']}, expected {want_split}")
        seed = M * 1009 + N * 17 + K
        x = make_x(M, K, seed=seed)
        dw = prepared(12, N, K, seed=seed + 1)
        dw2 = prepared(12, S3C_N2, N, seed=seed + 2)
        xq, sx = D.quantize_activation(x)
        torch.manual_seed(seed + 3)
        bias = torch.randn(N, device="cuda").bfloat16()
        all_ok = True
        rows = 0
        for vlabel, use_b, act in S3C_VARIANTS:
            b = bias if use_b else None
            y = D.linear(xq, dw, x_scale=sx, bias=b, activation=act)  # the unfused bf16 result ...
            want_q, want_s = D.quantize_activation(y)                  # ... quantized by this device's rule
            got_q, got_s = D.linear_quantized(xq, dw, x_scale=sx, bias=b, activation=act)
            got_qx, got_sx = D.linear_quantized(x, dw, bias=b, activation=act)
            rows = int(got_s.shape[0])
            q_ok = bytes_equal(got_q, want_q)
            s_ok = bytes_equal(got_s, want_s) and meta_ok(got_s, 12, M, N)  # every word, the padded rows included
            pad_ok = bool((got_s[M:] == 0).all())
            x_ok = bytes_equal(got_qx, got_q) and bytes_equal(got_sx, got_s)
            y2_fused = D.linear(got_q, dw2, x_scale=got_s)
            y2_ref = D.linear(y, dw2)
            chain_ok = bytes_equal(y2_fused, y2_ref)
            ok = q_ok and s_ok and pad_ok and x_ok and chain_ok
            if not ok:
                nq = int((got_q.view(torch.uint8) != want_q.view(torch.uint8)).sum()) \
                    if tuple(got_q.shape) == tuple(want_q.shape) else -1
                ns = int((got_s != want_s).sum()) if tuple(got_s.shape) == tuple(want_s.shape) else -1
                q_msg = "==" if q_ok else f"differs on {nq} codes"
                s_msg = "==" if s_ok else f"differ on {ns} words (or metadata)"
                self.check(False, f"r {tag} {vlabel}: xq {q_msg}, scale words {s_msg}, padded words zero {pad_ok}, "
                                  f"bf16-x entry == 2b entry {x_ok}, chained GEMM == unfused chain {chain_ok}")
            all_ok &= ok
            self.digests[f"{label}|{M}x{N}x{K}|{vlabel}"] = _s3c_digest(got_q, got_s)
        zq, zs = D.linear_quantized(xq, dw, x_scale=sx, bias=torch.zeros_like(bias))
        iq, is_ = D.linear_quantized(xq, dw, x_scale=sx)
        zero_ok = bytes_equal(zq, iq) and bytes_equal(zs, is_)
        self.check(zero_ok, f"r {tag}: bias = 0 differs from bias = None")
        self.cells += 1
        print(f"  r  {tag}: {len(S3C_VARIANTS)} variants x (bf16 x, 2b): xq, all {rows} scale rows, chained GEMM "
              f"exact; bias = 0 == None  {_ok(all_ok and zero_ok)}")

    def refuse_case(self, spec: dict) -> None:
        """A forced diagnostic instance: the identity call runs; linear_quantized is refused with NotImplementedError
        naming the instance and the quantized output, on the 2b entry before any device work (on the bf16-x entry the
        activation quantize runs first: the C++ route is the one that knows the knobs)."""
        M, N, K = spec["mnk"]
        route = _s1c_route(M, N, K)
        x = make_x(M, K, seed=11)
        dw = prepared(12, N, K, seed=12)
        xq, sx = D.quantize_activation(x)
        bias = torch.randn(N, device="cuda").bfloat16()
        y = D.linear(xq, dw, x_scale=sx)
        self.check(bool(torch.isfinite(y).all()), f"r {spec['label']}: the identity call on the forced instance failed")
        for vlabel, fn, no_events in (
                ("linear_quantized (2b)", lambda: D.linear_quantized(xq, dw, x_scale=sx), True),
                ("linear_quantized (2b, bias, gelu)",
                 lambda: D.linear_quantized(xq, dw, x_scale=sx, bias=bias, activation="gelu_erf"), True),
                ("linear_quantized (bf16 x)", lambda: D.linear_quantized(x, dw), False)):
            exc, n = _device_events(fn)
            ok = isinstance(exc, NotImplementedError) and spec["word"] in str(exc) \
                and "quantized output" in str(exc) and (n == 0 or not no_events)
            self.check(ok, f"r {spec['label']} {vlabel}: expected NotImplementedError naming {spec['word']!r} and "
                           f"the quantized output with no device work, got {type(exc).__name__ if exc else None} "
                           f"({n} events): {exc}")
            print(f"  r  {spec['label']} route {route}: {vlabel} -> {type(exc).__name__ if exc else 'no exception'}, "
                  f"{n} device events  {_ok(ok)}")
            if exc is not None and vlabel == "linear_quantized (2b)":
                print(f"  r  message: {exc}")

    def graphs_case(self) -> None:
        calls = {}
        for name, (M, N, K), use_b, act in (("fc1 bias+gelu", (78, 5120, 1280), True, "gelu_erf"),
                                            ("proj1 identity", (533, 1280, 1280), False, None),
                                            ("fc2 bias", (1079, 1280, 5120), True, None),
                                            ("streamk16 bias+gelu", (16, 2560, 9728), True, "gelu_erf"),
                                            ("streamk129 gelu", (129, 2560, 9728), False, "gelu_erf")):
            route = _s1c_route(M, N, K)
            x = make_x(M, K, seed=M + 1)
            dw = prepared(12, N, K, seed=N + 2)
            xq, sx = D.quantize_activation(x)
            b = torch.randn(N, device="cuda").bfloat16() if use_b else None
            for path, fn in (("bf16 x", lambda: D.linear_quantized(x, dw, bias=b, activation=act)),
                             ("2b", lambda: D.linear_quantized(xq, dw, x_scale=sx, bias=b, activation=act))):
                ref = tuple(t.clone() for t in fn())
                torch.cuda.synchronize()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    out = fn()
                torch.cuda.synchronize()
                out[0].view(torch.uint8).fill_(0xFF)  # every code a NaN
                out[1].fill_(-1)                       # every scale word 0xFFFFFFFF, the padded rows' included
                g.replay()
                torch.cuda.synchronize()
                ok = bytes_equal(out[0], ref[0]) and bytes_equal(out[1], ref[1])
                self.check(ok, f"r graph {name} ({path}): replay != eager")
                print(f"  r  graph {name} M={M} N={N} K={K} route {route['tile']} k_split {route['k_split']} ({path}): "
                      f"capture on a non-default stream, replay into NaN codes / all-ones words == eager  {_ok(ok)}")
                calls[(name, path)] = (g, out, ref, (x, dw, xq, sx, b))  # the graph reads these: keep them alive
        # Two different captured calls replayed concurrently from one thread, one per stream, 200 times.
        for a, b2 in ((("fc1 bias+gelu", "2b"), ("streamk16 bias+gelu", "2b")),
                      (("proj1 identity", "bf16 x"), ("streamk129 gelu", "2b")),
                      (("fc2 bias", "bf16 x"), ("fc2 bias", "2b"))):
            (g1, o1, r1, _), (g2, o2, r2, _) = calls[a], calls[b2]
            s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
            bad1 = torch.zeros((), dtype=torch.int64, device="cuda")
            bad2 = torch.zeros((), dtype=torch.int64, device="cuda")
            torch.cuda.synchronize()
            for _ in range(200):
                with torch.cuda.stream(s1):
                    g1.replay()
                    bad1 += (o1[0].view(torch.uint8) != r1[0].view(torch.uint8)).sum() + (o1[1] != r1[1]).sum()
                with torch.cuda.stream(s2):
                    g2.replay()
                    bad2 += (o2[0].view(torch.uint8) != r2[0].view(torch.uint8)).sum() + (o2[1] != r2[1]).sum()
            torch.cuda.synchronize()
            ok = int(bad1) == 0 and int(bad2) == 0
            self.check(ok, f"r concurrent replay {a} || {b2}: {int(bad1)} / {int(bad2)} differing elements")
            print(f"  r  concurrent replay, two streams, one thread, 200 x: {a[0]} ({a[1]}) || {b2[0]} ({b2[1]}) == "
                  f"serial  {_ok(ok)}")

    def compile_case(self) -> None:
        N, K = 5120, 1280
        dw = prepared(12, N, K, seed=71)
        bias = torch.randn(N, device="cuda").bfloat16()

        def f_x(x):
            return D.linear_quantized(x, dw, bias=bias, activation="gelu_erf")

        def f_2b(x):
            xq, sx = D.quantize_activation(x)
            return D.linear_quantized(xq, dw, x_scale=sx, bias=bias, activation="gelu_erf")

        def same(got, want, strict: bool) -> bool:
            if strict:
                return bytes_equal(got[0], want[0]) and bytes_equal(got[1], want[1])
            # inductor may hand the M-fastest scale back in another stride order: compare the words logically
            return bytes_equal(got[0], want[0]) and torch.equal(got[1], want[1])

        for fname, f, want_nodes in (("bf16 x", f_x, ["dense_linear_quantized"]),
                                     ("2b", f_2b, ["dense_linear_quantized", "dense_quantize_activation"])):
            ins = {M: make_x(M, K, seed=72 + M) for M in (78, 1079, 7)}
            want = {M: f(a) for M, a in ins.items()}
            graphs = []

            def record(gm, example_inputs):
                graphs.append(gm)
                return gm.forward

            torch._dynamo.reset()
            rec = torch.compile(f, backend=record, fullgraph=True)
            rec_ok = all(same(rec(a), want[M], True) for M, a in ins.items())
            per_graph = [sorted(str(n.target).split("fish_scales_ops.")[-1].split(".")[0] for n in g.graph.nodes
                                if n.op == "call_function" and "fish_scales_ops" in str(n.target)) for g in graphs]
            node_ok = bool(graphs) and all(ours == sorted(want_nodes) for ours in per_graph)
            self.check(rec_ok and node_ok, f"r compile ({fname}): recording backend nodes {per_graph}, equal {rec_ok}")
            print(f"  r  torch.compile ({fname}, bias + gelu_erf), recording backend, fullgraph: {len(graphs)} "
                  f"graph(s), nodes {per_graph[0] if per_graph else []}; == eager  {_ok(rec_ok and node_ok)}")
            torch._dynamo.reset()
            ind = torch.compile(f, fullgraph=True)
            ok = all(same(ind(a), want[M], False) for M, a in ins.items())
            torch.cuda.synchronize()
            self.check(ok, f"r compile ({fname}): inductor != eager")
            print(f"  r  torch.compile ({fname}), inductor, fullgraph, M = 78, 1079, 7: == eager  {_ok(ok)}")
            torch._dynamo.reset()


def _s3c_worker(spec_json: str) -> int:
    import json
    spec = json.loads(spec_json)
    w = _S3cWorker()
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        kind = spec["kind"]
        if kind == "exact":
            for cell in spec["cells"]:
                w.cell(*cell["mnk"], want_tile=cell.get("tile"), want_split=cell.get("split"), label=spec["label"])
        elif kind == "refuse":
            w.refuse_case(spec)
        elif kind == "graphs":
            w.graphs_case()
        elif kind == "compile":
            w.compile_case()
        else:
            raise ValueError(f"unknown worker kind {kind!r}")
    except Exception as e:  # noqa: BLE001 - reported to the parent as a failure with its reason
        import traceback
        traceback.print_exc()
        w.failures.append(f"r {spec.get('label', spec['kind'])}: {type(e).__name__}: {e}")
    print("S3C_RESULT " + json.dumps({"failures": w.failures, "digests": w.digests, "cells": w.cells}))
    return 0


def _s3c_specs() -> list:
    """(label, env, spec) of every worker of case r."""
    grid = [{"mnk": (M, N, K)} for N, K in S3C_SHAPES for M in S3C_MS]
    natural = grid + [{"mnk": mnk, "split": "split"} for mnk in S3C_STREAMK_NATURAL]
    forced = [{"mnk": mnk} for mnk in S3C_FORCED_GRID]
    specs = [("natural", {"FSO_SMALLM_ROUTE": "0"}, {"kind": "exact", "label": "natural", "cells": natural}),
             ("natural, FSO_DISABLE_PDL=1", {"FSO_SMALLM_ROUTE": "0", "FSO_DISABLE_PDL": "1"},
              {"kind": "exact", "label": "natural", "cells": natural})]
    for t in S1C_SINGLE_TILES:
        lab = f"tile {t}"
        specs.append((lab, {"FSO_FORCE_TILE": ",".join(map(str, t))},
                      {"kind": "exact", "label": lab, "cells": [dict(c, tile=t, split=1) for c in forced]}))
    for t, ks in S1C_SK_TILES:
        lab = f"tile {t} ksplit {ks}"
        specs.append((lab, {"FSO_FORCE_TILE": ",".join(map(str, t)), "FSO_FORCE_KSPLIT": str(ks)},
                      {"kind": "exact", "label": lab, "cells": [dict(c, tile=t, split="split") for c in forced]}))
    for t, ks in (((160, 128, 2), 0), ((32, 64, 4), 4)):
        lab = f"tile {t}" + (f" ksplit {ks}" if ks else "")
        env = {"FSO_FORCE_TILE": ",".join(map(str, t)), "FSO_DISABLE_PDL": "1"}
        if ks:
            env["FSO_FORCE_KSPLIT"] = str(ks)
        specs.append((lab + ", FSO_DISABLE_PDL=1", env,
                      {"kind": "exact", "label": lab, "cells": [dict(c, tile=t) for c in forced]}))
    specs += [("SmallM", {"FSO_FORCE_TILE": "16,128,4", "FSO_FORCE_SMALLM": "1"},
               {"kind": "refuse", "label": "SmallM", "mnk": (4, 2048, 2048), "word": "SmallM"}),
              ("MinBlocksPerSm=2", {"FSO_FORCE_TILE": "16,128,4", "FSO_FORCE_MIN_BLOCKS": "2"},
               {"kind": "refuse", "label": "MinBlocksPerSm=2", "mnk": (4, 2048, 2048), "word": "MinBlocksPerSm=2"}),
              ("SchedGroup=8", {"FSO_FORCE_TILE": "16,64,4", "FSO_FORCE_SCHED_GROUP": "8"},
               {"kind": "refuse", "label": "SchedGroup=8", "mnk": (4, 2048, 2048), "word": "FSO_FORCE_SCHED_GROUP"}),
              ("graphs and streams", {}, {"kind": "graphs", "label": "graphs"}),
              ("torch.compile", {}, {"kind": "compile", "label": "compile"})]
    return specs


def case_r(major: int) -> None:
    import json
    import subprocess
    if major != 12:
        print(f"  r  SKIP: case r is the sm_120/sm_121 MXFP8 cascade (sm_100/sm_103: case s); this is {_arch_label()}")
        return
    here = os.path.abspath(__file__)
    digests = {}
    for label, env_add, spec in _s3c_specs():
        env = {k: v for k, v in os.environ.items() if k not in W_KNOBS}
        env.update(env_add)
        r = subprocess.run([sys.executable, here, "--s3c-worker", json.dumps(spec)], env=env, capture_output=True,
                           text=True)
        result = None
        for line in r.stdout.splitlines():
            if line.startswith("S3C_RESULT "):
                result = json.loads(line[len("S3C_RESULT "):])
            else:
                print(line)
        if result is None or r.returncode != 0:
            _check(False, f"r [{label}]: the worker failed (rc {r.returncode}): {r.stderr[-2000:]}")
            print(f"  r  [{label}] worker FAILED rc {r.returncode}\n{r.stderr[-2000:]}")
            continue
        failures.extend(f"[{label}] {f}" for f in result["failures"])
        digests[label] = result["digests"]
        print(f"  r  [{label}] ({', '.join(f'{k}={v}' for k, v in env_add.items()) or 'no knobs'}): "
              f"{result['cells']} cells, {len(result['failures'])} failures  {_ok(not result['failures'])}")
    # FSO_DISABLE_PDL=1 gives the bytes of the default, cell for cell.
    for on, off in (("natural", "natural, FSO_DISABLE_PDL=1"),
                    ("tile (160, 128, 2)", "tile (160, 128, 2), FSO_DISABLE_PDL=1"),
                    ("tile (32, 64, 4) ksplit 4", "tile (32, 64, 4) ksplit 4, FSO_DISABLE_PDL=1")):
        a, b = digests.get(on), digests.get(off)
        common = sorted(set(a or {}) & set(b or {}))
        ok = bool(common) and all(a[k] == b[k] for k in common)
        _check(ok, f"r PDL: [{on}] and [{off}] differ on {sum(a[k] != b[k] for k in common) if common else 'all'} "
                   f"of {len(common)} outputs")
        print(f"  r  PDL on vs off: [{on}] == [{off}] on {len(common)} outputs  {_ok(ok)}")


# --- (s) sm_100/sm_103: linear_quantized on the C++ cascade (S3b) ----------------------------------------------------
#
# fso.dense.linear_quantized(...) == quantize_activation(linear(...)) bit for bit: the e4m3 codes and every word of the
# scale (the padded rows' words included), on the cascade's own picks and on every forced family of case p (the same
# FSO_FORCE_TILE rows, the split-K with the automatic and a forced split, the rows without an instantiation, which
# must refuse), one subprocess per family. Those families force the cascade (FSO_DISABLE_SMM=1, FSO_DISABLE_DSL=1),
# so the bf16 reference runs on the same route. The "natural" family runs on the card's own routing (no
# FSO_DISABLE_* switch): on cells whose bf16 call takes the decode row, tier 1 (cuBLAS) or tier 2 (the DSL kernel),
# the feature-aware router (S1b1) must send linear_quantized to the cascade, the only tier that writes the quantized
# output; see _s_natural_cell. Per cell: identity, bias, bias + erf-GELU and erf-GELU alone, from the
# pre-quantized x and from a bf16 x; the chain linear(xq_next, w2, x_scale=scale_next) == linear(y, w2); the fused
# call launches as many kernels as the bf16 call of the same cell and none of them is a quantize kernel. Per family:
# capture on a non-default stream after one eager call (replay into a clobbered output == eager), FSO_DISABLE_PDL=1
# reruns with equal digests, two different split-K calls captured on two streams and replayed concurrently, the
# torch.compile node, and the output metadata of the real op == its fake's.

S_SHAPES = ((5120, 1280), (1280, 1280), (2048, 1280), (1280, 5120))  # N x K: fc1, proj1, 2048 x 1280, fc2
S_MS = (1, 7, 78, 129, 533, 1079)
S_FORCED_MS = (7, 129, 1079)  # the forced tiles: one M per band (the exact rows are case p's job)
S_SPLIT_MS = (1, 7, 78, 129)
# label -> the keyword arguments of the call (True for the bias stands for the cell's bias tensor)
S_VARIANTS = (("identity", {}), ("bias", {"bias": True}),
              ("bias + gelu_erf", {"bias": True, "activation": "gelu_erf"}), ("gelu_erf", {"activation": "gelu_erf"}))


def _s_pair_eq(label: str, got, want, local: list) -> bool:
    """(xq, scale) pairs equal in metadata and in every byte (the scale's padded words included)."""
    gq, gs = got
    wq, ws = want
    ok_q, ok_s = bytes_equal(gq, wq), bytes_equal(gs, ws)
    if not (ok_q and ok_s):
        nq = int((gq.view(torch.uint8) != wq.view(torch.uint8)).sum()) if tuple(gq.shape) == tuple(wq.shape) else -1
        ns = int((gs.view(torch.int32) != ws.view(torch.int32)).sum()) if tuple(gs.shape) == tuple(ws.shape) else -1
        local.append(f"{label}: codes {'equal' if ok_q else f'{nq} differ'} ({tuple(gq.shape)} vs {tuple(wq.shape)}), "
                     f"scale words {'equal' if ok_s else f'{ns} differ'} ({tuple(gs.shape)} vs {tuple(ws.shape)})")
    return ok_q and ok_s


def _s_kwargs(spec: dict, b: torch.Tensor) -> dict:
    return {k: (b if k == "bias" else v) for k, v in spec.items()}


def _s_cell(M: int, N: int, K: int, seed: int, dig: _PDigest, local: list, stats: dict) -> None:
    """One (M, N, K): the four variants from both entry paths against quantize_activation(linear(...)), then the
    chain into a second GEMM."""
    L, LQ, QA = D.linear, D.linear_quantized, D.quantize_activation
    tag = f"M={M} N={N} K={K}"
    x = make_x(M, K, seed)
    dw = prepared(10, N, K, seed + 1)
    xq, sx = QA(x)
    torch.manual_seed(seed + 2)
    b = (torch.randn(N, device="cuda") * 0.5).bfloat16()
    for label, spec in S_VARIANTS:
        kw = _s_kwargs(spec, b)
        y = L(xq, dw, x_scale=sx, **kw)
        want = QA(y)
        got = LQ(xq, dw, x_scale=sx, **kw)
        _s_pair_eq(f"{tag} {label} (2b)", got, want, local)
        got_x = LQ(x, dw, **kw)  # the bf16 entry path: the quantize inside gives the same xq / sx
        _s_pair_eq(f"{tag} {label} (bf16 x)", got_x, want, local)
        dig.add(got[0].view(torch.uint8))
        dig.add(got[1])
        stats["elements"] += M * N
        stats["nonzero_codes"] += int((got[0].view(torch.uint8) != 0).sum())
        stats["scale_words"] += int(got[1].numel())
    y = L(xq, dw, x_scale=sx, bias=b, activation="gelu_erf")
    q, s = LQ(xq, dw, x_scale=sx, bias=b, activation="gelu_erf")
    dw2 = prepared(10, 1280, N, seed + 3)  # the next GEMM: K2 = N
    _p_eq(f"{tag} chain linear(xq_next, w2, x_scale=scale_next) == linear(y, w2)", L(q, dw2, x_scale=s), L(y, dw2),
          local)
    stats["cells"] += 1


def _s_launches(M: int, N: int, K: int, local: list, stats: dict) -> None:
    """The fused call launches as many kernels as the bf16 call of the same cell, none of them a quantize kernel."""
    x = make_x(M, K, 5)
    dw = prepared(10, N, K, 6)
    xq, sx = D.quantize_activation(x)
    b = torch.randn(N, device="cuda").bfloat16()
    D.linear_quantized(xq, dw, x_scale=sx, bias=b, activation="gelu_erf")  # eager once (attributes, params cache)
    _, names_y = _kernels_of(lambda: D.linear(xq, dw, x_scale=sx, bias=b, activation="gelu_erf"))
    _, names_q = _kernels_of(lambda: D.linear_quantized(xq, dw, x_scale=sx, bias=b, activation="gelu_erf"))
    quant = [n for n in names_q if "quantize" in n.lower()]
    if len(names_q) != len(names_y) or quant or not names_q:
        local.append(f"launches M={M} N={N} K={K}: fused {names_q} vs bf16 {names_y}")
    stats["launches"] = f"{len(names_q)} fused / {len(names_y)} bf16 (M={M} N={N} K={K})"


def _s_capture(label: str, fn, local: list) -> None:
    ref = tuple(t.clone() for t in fn())  # one eager call on the default stream
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = fn()
    out[0].view(torch.uint8).fill_(0xFF)  # the e4m3 NaN code
    out[1].fill_(-1)
    g.replay()
    torch.cuda.synchronize()
    if not (bytes_equal(out[0], ref[0]) and bytes_equal(out[1], ref[1])):
        local.append(f"capture {label}: replay into a clobbered output != eager")


def _s_compile(local: list) -> None:
    N, K, M = 5120, 1280, 78
    dw = prepared(10, N, K, seed=71)
    x = make_x(M, K, seed=72)
    b = torch.randn(N, device="cuda").bfloat16()

    def f(x):
        return D.linear_quantized(x, dw, bias=b, activation="gelu_erf")

    want = f(x)
    graphs = []

    def record(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    got = torch.compile(f, backend=record, fullgraph=True)(x)
    nodes = [str(n.target) for gm in graphs for n in gm.graph.nodes
             if n.op == "call_function" and "fish_scales_ops" in str(n.target)]
    eq = bytes_equal(got[0], want[0]) and bytes_equal(got[1], want[1])
    if not (eq and len(nodes) == 1 and "dense_linear_quantized" in nodes[0]):
        local.append(f"torch.compile recording backend: nodes {nodes}, equal {eq}")
    torch._dynamo.reset()
    got = torch.compile(f, fullgraph=True)(x)
    torch.cuda.synchronize()
    if not (bytes_equal(got[0], want[0]) and bytes_equal(got[1], want[1])):
        local.append("torch.compile inductor: compiled != eager")
    torch._dynamo.reset()


def _s_fake_meta(local: list) -> None:
    """The real op's outputs have the metadata of its fake (the contract of an [M, N] activation)."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    N, K, M = 1280, 1280, 129
    dw = prepared(10, N, K, seed=73)
    x = make_x(M, K, seed=74)
    ops = torch.ops.fish_scales_ops
    q, s = ops.dense_linear_quantized(x, dw.weight, dw.scale, dw.kind)
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fq, fs = ops.dense_linear_quantized(mode.from_tensor(x), mode.from_tensor(dw.weight),
                                            mode.from_tensor(dw.scale), dw.kind)
    meta = lambda t: (t.dtype, tuple(t.shape), t.stride())  # noqa: E731
    if meta(q) != meta(fq) or meta(s) != meta(fs) or not meta_ok(s, 10, M, N):
        local.append(f"fake metadata: real {meta(q)}, {meta(s)} vs fake {meta(fq)}, {meta(fs)}")


def _s_two_streams(local: list) -> None:
    """Two different split-K quantized calls captured on two streams of one thread, replayed concurrently 200
    times; each must equal its serial result every time (codes and scale words)."""
    calls = []
    for (N, K, M, seed) in ((1280, 5120, 7, 101), (2048, 7680, 1, 102)):
        dw = prepared(10, N, K, seed=seed)
        x = make_x(M, K, seed=seed + 1)
        xq, sx = D.quantize_activation(x)
        b = torch.randn(N, device="cuda").bfloat16()
        calls.append(lambda xq=xq, dw=dw, sx=sx, b=b: D.linear_quantized(xq, dw, x_scale=sx, bias=b,
                                                                         activation="gelu_erf"))
    refs = [tuple(t.clone() for t in c()) for c in calls]
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs, outs = [], []
    for c, s in zip(calls, streams):
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            outs.append(c())
        graphs.append(g)
    torch.cuda.synchronize()
    bad = [torch.zeros((), dtype=torch.int64, device="cuda") for _ in calls]
    for _ in range(200):
        for i in range(2):
            with torch.cuda.stream(streams[i]):
                graphs[i].replay()
                bad[i] += (outs[i][0].view(torch.uint8) != refs[i][0].view(torch.uint8)).sum()
                bad[i] += (outs[i][1] != refs[i][1]).sum()
    torch.cuda.synchronize()
    nbad = [int(t) for t in bad]
    if any(nbad):
        local.append(f"two-stream concurrent replay: {nbad} mismatching elements over 200 iterations")


# The quantized output's two-warpgroup kernels (issue #12 FB3) on the cascade's own picks, N x K x M: the tower's fc1 at
# M = 533 (the wave rule's 1-SM 128 x 192 tile) and 1079 (the band rule's 2-SM 256 x 128 TileK 256 tile) and proj1 at
# M = 533 (the short-K rule's 1-SM 128 x 64 tile), all with the TMA epilogue and two accumulator stages.
S_MWG_CELLS = ((5120, 1280, 533), (5120, 1280, 1079), (1280, 1280, 533))


def _s_multiwg(local: list, stats: dict) -> None:
    """Determinism of the two-warpgroup kernels (dense_epilogue_gemm.cuh, kDenseQuantEpiWarpgroups): every call of
    S_MWG_CELLS (bias + erf-GELU, the 2b entry) is captured on a stream of its own and replayed 200 times, each replay
    bit-identical to the eager call (codes and scale words); then the first two are replayed concurrently on their two
    streams 200 times, each against its serial result."""
    calls = []
    for (N, K, M) in S_MWG_CELLS:
        dw = prepared(10, N, K, seed=M + N)
        x = make_x(M, K, seed=M + N + 1)
        xq, sx = D.quantize_activation(x)
        b = (torch.randn(N, device="cuda") * 0.5).bfloat16()
        calls.append((f"M={M} N={N} K={K}", lambda xq=xq, dw=dw, sx=sx, b=b: D.linear_quantized(
            xq, dw, x_scale=sx, bias=b, activation="gelu_erf")))
    refs = [tuple(t.clone() for t in c()) for _, c in calls]
    torch.cuda.synchronize()
    streams, graphs, outs = [], [], []
    for _, c in calls:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            outs.append(c())
        streams.append(s)
        graphs.append(g)
    torch.cuda.synchronize()

    def mismatches(i):
        return (outs[i][0].view(torch.uint8) != refs[i][0].view(torch.uint8)).sum() + (outs[i][1] != refs[i][1]).sum()

    for i, (label, _) in enumerate(calls):
        bad = torch.zeros((), dtype=torch.int64, device="cuda")
        torch.cuda.synchronize()  # the counter's fill (default stream) before the replay stream adds to it
        with torch.cuda.stream(streams[i]):
            for _ in range(200):
                graphs[i].replay()
                bad += mismatches(i)
        torch.cuda.synchronize()
        if int(bad):
            local.append(f"two warpgroups {label}: {int(bad)} mismatching elements over 200 replays")
        stats["mwg_replayed"] = stats.get("mwg_replayed", 0) + 1
    bad = [torch.zeros((), dtype=torch.int64, device="cuda") for _ in range(2)]
    torch.cuda.synchronize()
    for _ in range(200):
        for i in range(2):
            with torch.cuda.stream(streams[i]):
                graphs[i].replay()
                bad[i] += mismatches(i)
    torch.cuda.synchronize()
    nbad = [int(t) for t in bad]
    if any(nbad):
        local.append(f"two warpgroups, two-stream concurrent replay: {nbad} mismatching elements over 200 iterations")


def _s_refuse(force: str, word: str, local: list) -> None:
    N, K, M = 1280, 1280, 7
    dw = prepared(10, N, K, seed=91)
    x = make_x(M, K, seed=92)
    xq, sx = D.quantize_activation(x)
    b = torch.randn(N, device="cuda").bfloat16()
    torch.cuda.synchronize()
    # Pre-quantized entries only (as in _p_refuse): with a bf16 x the activation quantize runs before the cascade's
    # refusal, because the forced row is only known to the C++ dispatch, not to the router's plan.
    for label, fn in (("2b", lambda: D.linear_quantized(xq, dw, x_scale=sx)),
                      ("2b, bias + gelu_erf", lambda: D.linear_quantized(xq, dw, x_scale=sx, bias=b,
                                                                         activation="gelu_erf")),
                      ("raw", lambda: torch.ops.fish_scales_ops.linear_mxfp8_quantized_raw(xq, dw.weight, sx,
                                                                                            dw.scale, b))):
        exc, n = _device_events(fn)
        msg = str(exc)
        if type(exc) is not NotImplementedError or f"FSO_FORCE_TILE={force}" not in msg or word not in msg or n:
            local.append(f"FSO_FORCE_TILE={force} ({label}): {type(exc).__name__ if exc else 'no exception'}, "
                         f"{n} device events: {msg}")
    y = D.linear(xq, dw, x_scale=sx)  # the identity keeps working on the row
    torch.cuda.synchronize()
    if not bool(torch.isfinite(y).all()):
        local.append(f"FSO_FORCE_TILE={force}: the identity call gave non-finite values")


# The natural-routing family: cells (M, N, K) whose bf16 call takes another tier on the card's own routing -- the
# decode row in its in-cluster split-K form (1 x 2048 x 4096, 16 x 1280 x 5120, 32 x 1280 x 1280) and its persistent
# form (12 and 48 x 5120 x 1280), tier 1 (182 x 5120 x 1280, 78 x 1280 x 1280 -- its bias call on the cascade's
# short-K tile since issue #12 T1 --, 1079 x 2048 x 1280) and tier 2 (the cubic 4096 cell) -- with the variants the
# decode row and tier 1 serve in bf16 (the identity, the bias, bias + GELU).
S_NATURAL_CELLS = ((1, 2048, 4096), (16, 1280, 5120), (32, 1280, 1280), (12, 5120, 1280), (48, 5120, 1280),
                   (182, 5120, 1280), (78, 1280, 1280), (1079, 2048, 1280), (4096, 4096, 4096),
                   # (P5) the swap-AB row's narrow-N long-K band above M = 64 (its split-K form in two and four
                   # slices, its persistent form) and the tier-1 cell whose residual / GELU calls fall through
                   # onto the row: the quantized call stays on the cascade.
                   (129, 2048, 4096), (182, 1280, 5120), (384, 2048, 4096), (78, 1280, 7680), (1079, 1280, 5120),
                   # (issue #12, T1) the bias call on the cascade's 2-SM 256 x 64 short-K tile, the quantized call on
                   # its 1-SM 128 x 64 tile: the bit-for-bit chain across two tiles of the one route.
                   (182, 1280, 1280))
S_NATURAL_VARIANTS = (("identity", {}), ("bias", {"bias": True}),
                      ("bias + gelu_erf", {"bias": True, "activation": "gelu_erf"}))


def _s_natural_cell(M: int, N: int, K: int, seed: int, local: list, stats: dict) -> None:
    """One cell on the card's own routing: the router's plan for the quantized call is the cascade while the plan
    of the same call without the quantized output may name another tier; the kernels the fused call runs are the
    cascade's (no cuBLAS, decode-row or DSL kernel); its bytes equal quantize_activation of the bf16 cascade call
    (the raw op linear_mxfp8_raw, which goes straight to the cascade), from both entries. The difference to
    quantize_activation(linear(...)) on the bf16 call's own route (another tier, possibly another accumulation
    order) is counted, not required to be zero: stats["natural"] holds it per cell and variant."""
    from fish_scales_ops.gemm import _epilogue as E
    from fish_scales_ops.gemm import _sm100_dispatch
    L, LQ, QA = D.linear, D.linear_quantized, D.quantize_activation
    ops = torch.ops.fish_scales_ops
    x = make_x(M, K, seed)
    dw = prepared(10, N, K, seed + 1)
    xq, sx = QA(x)
    torch.manual_seed(seed + 2)
    b = (torch.randn(N, device="cuda") * 0.5).bfloat16()
    for label, spec in S_NATURAL_VARIANTS:
        kw = _s_kwargs(spec, b)
        tag = f"natural M={M} N={N} K={K} {label}"
        e_q = E.make("case s", m=M, n=N, device=dw.weight.device, quantized=True, **kw)
        e_y = E.make("case s", m=M, n=N, device=dw.weight.device, **kw)
        plan_q, plan_y = _sm100_dispatch.plan(M, N, K, e_q), _sm100_dispatch.plan(M, N, K, e_y)
        stats["bf16_tiers"][plan_y] = stats["bf16_tiers"].get(plan_y, 0) + 1
        if plan_q != "sm100.cascade":
            local.append(f"{tag}: the plan of the quantized call is {plan_q}, not sm100.cascade")
        got, names = _q_kernels(lambda: LQ(xq, dw, x_scale=sx, **kw))
        foreign = [n for n in names if "nvjet" in n or "flashinfer_sm100_blockscaled" in n or "cutlass_wrapper" in n]
        if foreign or not names:
            local.append(f"{tag}: the fused call ran {names} (expected only cascade kernels)")
        act = E.ACT_CODES[kw.get("activation")]
        y_cascade = ops.linear_mxfp8_raw(xq, dw.weight, sx, dw.scale, kw.get("bias"), None, act)
        _s_pair_eq(f"{tag} (2b) vs quantize_activation(linear_mxfp8_raw), the cascade", got, QA(y_cascade), local)
        _s_pair_eq(f"{tag} (bf16 x) vs the 2b call", LQ(x, dw, **kw), got, local)
        y_route = L(xq, dw, x_scale=sx, **kw)  # the bf16 call on its own route (plan_y)
        wq, ws = QA(y_route)
        n_codes = int((got[0].view(torch.uint8) != wq.view(torch.uint8)).sum())
        n_words = int((got[1] != ws).sum())
        n_bf16 = int((y_route.view(torch.int16) != y_cascade.view(torch.int16)).sum())
        stats["natural"].append([M, N, K, label, plan_y, n_bf16, n_codes, n_words, M * N])
        stats["elements"] += M * N
        stats["nonzero_codes"] += int((got[0].view(torch.uint8) != 0).sum())
        stats["scale_words"] += int(got[1].numel())
    stats["cells"] += 1


def s_worker(label: str) -> int:
    """One family of case s, in this process; the last stdout line is a JSON summary for the parent."""
    import json
    local: list[str] = []
    stats = {"cells": 0, "elements": 0, "nonzero_codes": 0, "scale_words": 0}
    dig = _PDigest()
    torch.backends.cuda.matmul.allow_tf32 = False
    if label == "natural":
        from fish_scales_ops.gemm import _sm100_decode, _sm100_dsl
        stats.update(bf16_tiers={}, natural=[], decode_enabled=bool(_sm100_decode.enabled()),
                     dsl_available=_sm100_dsl._init() is not None)
        for (M, N, K) in S_NATURAL_CELLS:
            _s_natural_cell(M, N, K, M * 1009 + N * 17 + K, local, stats)
    elif label.startswith("refuse/"):
        _s_refuse(*P_REFUSE[label], local)
    elif label == "streams":
        _s_two_streams(local)
    elif label == "mwg":
        _s_multiwg(local, stats)
    else:
        if label in P_SPLIT:
            cells = [(M, N, K) for (N, K) in S_SHAPES for M in S_SPLIT_MS]
        elif label in P_FORCED:
            cells = [(M, N, K) for (N, K) in S_SHAPES for M in S_FORCED_MS]
        else:  # "rules": the cascade's own picks
            cells = [(M, N, K) for (N, K) in S_SHAPES for M in S_MS]
        for (M, N, K) in cells:
            _s_cell(M, N, K, M * 1009 + N * 17 + K, dig, local, stats)
        M, N, K = cells[min(2, len(cells) - 1)]
        _s_launches(M, N, K, local, stats)
        x = make_x(M, K, 7)
        dw = prepared(10, N, K, 8)
        xq, sx = D.quantize_activation(x)
        b = torch.randn(N, device="cuda").bfloat16()
        _s_capture(f"M={M} N={N} K={K} bias + gelu_erf",
                   lambda: D.linear_quantized(xq, dw, x_scale=sx, bias=b, activation="gelu_erf"), local)
        _s_capture(f"M={M} N={N} K={K} bias, bf16 x", lambda: D.linear_quantized(x, dw, bias=b), local)
        if label == "rules":
            _s_compile(local)
            _s_fake_meta(local)
    print(json.dumps({"label": label, "failures": local, "digest": dig.hex(), **stats}))
    return 0


def _s_spawn(label: str, extra_env: dict):
    import json
    import subprocess
    env = {k: v for k, v in os.environ.items() if not k.startswith("FSO_")}
    # every family but "natural" forces the cascade (P_COMMON_ENV); "natural" keeps the card's own routing
    env.update(Q_ENV if label == "natural" else P_COMMON_ENV)
    env.update(extra_env)
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--s-worker", label], env=env,
                       capture_output=True, text=True, timeout=7200)
    summary = None
    for line in reversed(r.stdout.splitlines()):
        if line.startswith("{"):
            summary = json.loads(line)
            break
    return r, summary


def case_s(major: int) -> None:
    import re
    if major == 12:
        _case_s_sm120()  # linear_quantized beside the small-M route (defined below)
        return
    if major != 10:
        print("  s  SKIP: linear_quantized beside another route: sm_100/sm_103 (the C++ cascade beside the other "
              "tiers) and sm_120/sm_121 (the cascade beside the small-M route)")
        return
    runs = {"rules": {}}
    runs.update({k: {"FSO_FORCE_TILE": v[0]} for k, v in P_FORCED.items()})
    runs.update(P_SPLIT)
    runs.update({k: {"FSO_FORCE_TILE": v[0]} for k, v in P_REFUSE.items()})
    runs["streams"] = {"FSO_FORCE_TILE": "128,128,9"}
    runs["mwg"] = {}
    runs["natural"] = {}
    nopdl = {"rules": "rules/nopdl", "9": "9/nopdl"}
    digests = {}
    s_re = re.compile(r"^\[fso sm100 tile\] quantized (\d+x\d+x\d+) cluster=\(\d+,\d+\) nosmem=(\d) .* warpgroups=(\d)")
    for label, env in list(runs.items()) + [(v, {**runs[k], "FSO_DISABLE_PDL": "1"}) for k, v in nopdl.items()]:
        worker = label.split("/nopdl")[0]
        r, summary = _s_spawn(worker, env)
        ok = r.returncode == 0 and summary is not None and not summary["failures"]
        ran = sorted({m.groups() for m in map(s_re.match, r.stderr.splitlines()) if m})
        bf16_epi = sum(1 for line in r.stderr.splitlines() if line.startswith("[fso sm100 tile] epilogue "))
        split_ran = "[fso sm100 tile] splitk 128x128x128" in r.stderr
        note = ""
        if worker in P_FORCED:
            _, tile, ns = P_FORCED[worker]
            # the quantized output's epilogue warpgroups (dense_epilogue_gemm.cuh, kDenseQuantGeluEpiWarpgroups): on
            # the TMA epilogue of the tiles narrower than 256 the erf-GELU calls run two and the identity / bias calls
            # one (CUTLASS's own class); every other tile runs one for every call
            two = ns == 0 and tile.split("x")[1] != "256"
            want = {(tile, str(ns), "1"), (tile, str(ns), "2")} if two else {(tile, str(ns), "1")}
            cov = set(ran) == want
            ok = ok and cov
            note = (f"; ran {['%s nosmem=%s warpgroups=%s' % e for e in ran]}"
                    f"{'' if cov else ' (expected ' + str(sorted(want)) + ')'}")
        elif worker in P_SPLIT:
            ok = ok and split_ran
            note = f"; split-K ran {split_ran}"
        elif worker == "rules":
            ok = ok and split_ran and len(ran) > 0
            note = f"; rules ran {len(ran)} quantized kernels {['%s ns=%s wg=%s' % e for e in ran]}, split-K {split_ran}"
        elif worker == "streams":
            caps = re.findall(r"\[fso sm100 splitk\] captured on stream (0x[0-9a-f]+): workspace (0x[0-9a-f]+)",
                              r.stderr)
            two = len(caps) == 2 and caps[0][0] != caps[1][0] and caps[0][1] != caps[1][1]
            ok = ok and two
            note = f"; captures {caps} (two streams, two workspaces: {two})"
        elif worker == "mwg":
            two = len(ran) > 0 and all(e[2] == "2" for e in ran)
            ok = ok and two
            n_rep = summary.get("mwg_replayed") if summary else None
            note = (f"; the cascade's picks ran {['%s ns=%s wg=%s' % e for e in ran]} (two warpgroups on each: {two}), "
                    f"{n_rep} captured calls replayed 200 x bit-identically, two of them concurrently on two streams")
        elif worker == "natural" and summary is not None:
            tiers = summary.get("bf16_tiers", {})
            want = {"sm100.smm"} | ({"sm100.decode"} if summary.get("decode_enabled") else set()) \
                | ({"sm100.dsl"} if summary.get("dsl_available") else set())
            seen = want <= set(tiers)
            ran_q = len(ran) > 0
            ok = ok and seen and ran_q
            note = (f"; bf16 calls planned on {tiers} (needs {sorted(want)}: {seen}), every quantized call planned "
                    f"on and ran on the cascade ({len(ran)} quantized kernels {['%s ns=%s wg=%s' % e for e in ran]})")
            for row in summary.get("natural", []):
                M, N, K, var, tier, n_bf16, n_codes, n_words, n_el = row
                print(f"  s  natural M={M} N={N} K={K} {var:15s} bf16 route {tier:13s}: vs quantize_activation("
                      f"linear()) on that route {n_codes} / {n_el} codes and {n_words} scale words differ "
                      f"(bf16 results differ on {n_bf16} elements)")
        if summary is not None and summary.get("cells"):
            note += (f"; {summary['cells']} cells, {summary['elements']} codes x 2 paths ({summary['nonzero_codes']} "
                     f"nonzero), {summary['scale_words']} scale words; launches {summary.get('launches')}; the bf16 "
                     f"reference ran {bf16_epi} epilogue instantiation(s)")
            if summary["nonzero_codes"] == 0:
                ok = False
                note += " (every code is zero: the data does not exercise the quantizer)"
            digests[label] = summary["digest"]
        if not ok:
            detail = summary["failures"][:8] if summary else r.stdout[-1500:] + r.stderr[-1500:]
            _check(False, f"s {label}: {detail}")
        print(f"  s  {label:12s} {_ok(ok)}{note}")
        if not ok and summary is not None:
            for f in summary["failures"][:8]:
                print(f"       {f}")
    for k, v in nopdl.items():
        same = k in digests and digests.get(k) == digests.get(v)
        _check(same, f"s: FSO_DISABLE_PDL=1 changed the bytes of family {k}")
        print(f"  s  PDL on / off, family {k}: identical bytes  {_ok(same)}")

# --- (s), sm_120/sm_121: linear_quantized beside the small-M route ---------------------------------------------------
#
# The small-M route (case w) does not write the quantized output, so every sm_120/sm_121 linear_quantized call runs on
# the MXFP8 cascade, while the bf16 call of a decode-band cell the route takes runs G1 (the 2b call, rule R1) or G2 /
# quantize + G1 (the bf16-x call). The contract (dense.md): linear_quantized(...) is bit for bit
# quantize_activation(linear(...)) when the bf16 call runs on the same route. Two subprocesses on the same cells and
# the same data: "off" (FSO_SMALLM_ROUTE=0: every bf16 call on the cascade) and "rules" (the shipped rules).
# Required: in each process linear_quantized launches no small-M kernel and gives the same bytes from both entries;
# it gives the same bytes in both processes (the switch does not move it); in the off process it equals
# quantize_activation(linear(...)) bit for bit, codes and every scale word; in the rules process the bf16 call runs the
# small-M kernel where the plan says so, and its bf16-x entry equals its 2b entry. Counted per cell and variant, not
# required to be zero: the bf16 elements in which the small-M route's result differs from the cascade's, and the codes
# and scale words in which linear_quantized differs from quantize_activation of the bf16 call on the small-M route.

# Candidate cells (M, N, K); the ones a rule routes (the 2b call's R1 or the bf16-x call's R2) are run: the issue's
# RTX 5090 TP2 shards qkv [2560 x 2048] and o [2048 x 2048] across the decode band, the tower shapes of case r at
# decode M, and K = 9728 at M <= 8, where the cascade splits K (Stream-K) and rounds its slices' partials (on the RTX
# 5090 the rules route none of the K = 9728 cells; they run where a rule takes them).
S120_CANDIDATES = tuple((M, N, K) for (N, K) in ((2560, 2048), (2048, 2048)) for M in (1, 4, 8, 16, 32, 48, 64)) \
    + tuple((M, N, K) for (N, K) in ((5120, 1280), (1280, 1280), (2048, 1280), (1280, 5120)) for M in (1, 7, 16)) \
    + ((1, 2560, 9728), (4, 2560, 9728), (8, 2560, 9728))


def _s120_worker(spec_json: str) -> int:
    """One process of the sm_120 family of case s (spec: mode "off" | "rules", the cells, the output file)."""
    import json
    spec = json.loads(spec_json)
    mode = spec["mode"]
    L, LQ, QA = D.linear, D.linear_quantized, D.quantize_activation
    local: list[str] = []
    saved, rows = {}, []
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for (M, N, K) in [tuple(c) for c in spec["cells"]]:
            seed = M * 1009 + N * 17 + K
            x = make_x(M, K, seed)
            dw = prepared(12, N, K, seed + 1)
            xq, sx = QA(x)
            torch.manual_seed(seed + 2)
            b = (torch.randn(N, device="cuda") * 0.5).bfloat16()
            r2b, rx = _w_route(M, N, K, False), _w_route(M, N, K, True)
            for label, vspec in S_VARIANTS:
                kw = _s_kwargs(vspec, b)
                tag = f"sm120 [{mode}] M={M} N={N} K={K} {label}"
                L(xq, dw, x_scale=sx, **kw)  # eager once
                y, names_y = _kernels_of(lambda: L(xq, dw, x_scale=sx, **kw))
                want_small = mode == "rules" and r2b["use"]
                if bool(_w_smallm_kernels(names_y)) != want_small:
                    local.append(f"{tag}: the 2b bf16 call ran {names_y}; a small-M kernel expected: {want_small}")
                if not bytes_equal(L(x, dw, **kw), y):
                    local.append(f"{tag}: the bf16-x call differs from the 2b call")
                LQ(xq, dw, x_scale=sx, **kw)  # eager once
                (q, sc), names_q = _kernels_of(lambda: LQ(xq, dw, x_scale=sx, **kw))
                if _w_smallm_kernels(names_q) or not names_q:
                    local.append(f"{tag}: linear_quantized ran {names_q} (expected the cascade alone)")
                q_x, sc_x = LQ(x, dw, **kw)
                if not (bytes_equal(q_x, q) and bytes_equal(sc_x, sc)):
                    local.append(f"{tag}: linear_quantized from the bf16-x entry differs from the 2b entry")
                wq, ws = QA(y)
                n_codes = int((q.view(torch.uint8) != wq.view(torch.uint8)).sum())
                n_words = int((sc != ws).sum())
                if mode == "off" and (n_codes or n_words or not bytes_equal(sc, ws)):
                    local.append(f"{tag}: on the cascade linear_quantized != quantize_activation(linear(...)): "
                                 f"{n_codes} codes, {n_words} scale words differ")
                saved[f"{M}x{N}x{K}|{label}"] = {"y": y.cpu(), "q": q.view(torch.uint8).cpu(), "s": sc.cpu()}
                rows.append([M, N, K, label, bool(r2b["use"]), bool(rx["use"]), n_codes, n_words, M * N,
                             int(sc.numel())])
    except Exception as e:  # noqa: BLE001 - reported to the parent as a failure with its reason
        import traceback
        traceback.print_exc()
        local.append(f"sm120 [{mode}]: {type(e).__name__}: {e}")
    torch.save(saved, spec["out"])
    print("S120_RESULT " + json.dumps({"failures": local, "rows": rows}))
    return 0


def _case_s_sm120() -> None:
    import json
    import shutil
    import subprocess
    if os.environ.get("FSO_SMALLM_ROUTE") not in (None, "", "rules") or any(os.environ.get(k) for k in S1C_KNOBS):
        print("  s  SKIP: FSO_SMALLM_ROUTE or a force knob is set in this process; the workers set their own")
        return
    sms = _sms()
    op = lambda M, N, K: torch.ops.fish_scales_ops.sm120_smallm_rules(M, N, K, sms)  # noqa: E731
    cells = [c for c in S120_CANDIDATES if int(op(*c)[0]) or int(op(*c)[1])]
    _check(bool(cells), "s sm120: no candidate cell takes the small-M route under the rules")
    tmp = tempfile.mkdtemp(prefix="dense_v1_s120_")
    here = os.path.abspath(__file__)
    res = {}
    for mode, env_add in (("off", {"FSO_SMALLM_ROUTE": "0"}), ("rules", {})):
        out = os.path.join(tmp, f"{mode}.pt")
        env = {k: v for k, v in os.environ.items() if k not in W_KNOBS}
        env.update(env_add)
        r = subprocess.run([sys.executable, here, "--s120-worker", json.dumps({"mode": mode, "cells": cells,
                                                                              "out": out})],
                           env=env, capture_output=True, text=True, timeout=3600)
        result = None
        for line in r.stdout.splitlines():
            if line.startswith("S120_RESULT "):
                result = json.loads(line[len("S120_RESULT "):])
            else:
                print(line)
        if result is None or r.returncode != 0:
            _check(False, f"s sm120 [{mode}]: the worker failed (rc {r.returncode}): {r.stderr[-2000:]}")
            print(f"  s  sm120 [{mode}] worker FAILED rc {r.returncode}\n{r.stderr[-2000:]}")
            shutil.rmtree(tmp, ignore_errors=True)
            return
        failures.extend(f"s {f}" for f in result["failures"])
        res[mode] = (result, torch.load(out))
        print(f"  s  sm120 [{mode}] ({', '.join(f'{k}={v}' for k, v in env_add.items()) or 'the rules'}): "
              f"{len(result['rows'])} cell-variants, {len(result['failures'])} failures  {_ok(not result['failures'])}")
    shutil.rmtree(tmp, ignore_errors=True)
    (_, t_off), (r_rules, t_rules) = res["off"], res["rules"]
    tot = {"rows": 0, "rows_bf16": 0, "bf16": 0, "codes": 0, "words": 0, "rows_codes": 0, "el": 0, "sw": 0}
    worst = (0, None)
    for M, N, K, label, use2b, usex, n_codes, n_words, n_el, n_sw in r_rules["rows"]:
        key = f"{M}x{N}x{K}|{label}"
        a, c = t_rules[key], t_off[key]
        same_q = torch.equal(a["q"], c["q"]) and torch.equal(a["s"], c["s"])
        _check(same_q, f"s sm120 {key}: linear_quantized differs between FSO_SMALLM_ROUTE=0 and the rules")
        n_bf16 = int((a["y"].view(torch.int16) != c["y"].view(torch.int16)).sum())
        route = (("G1 / " + ("G2" if usex else "quantize + G1")) if use2b else
                 ("cascade / G2" if usex else "cascade"))
        print(f"  s  sm120 M={M:2d} N={N:5d} K={K:5d} {label:15s} bf16 call (2b / bf16 x) on {route:20s}: "
              f"bf16 results differ from the cascade's on {n_bf16} / {n_el} elements; linear_quantized vs "
              f"quantize_activation(linear()) on that route: {n_codes} / {n_el} codes, {n_words} / {n_sw} scale "
              f"words differ; linear_quantized the same with the route off  {_ok(same_q)}")
        tot["rows"] += 1
        tot["rows_bf16"] += int(n_bf16 > 0)
        tot["rows_codes"] += int(n_codes > 0)
        tot["bf16"] += n_bf16
        tot["codes"] += n_codes
        tot["words"] += n_words
        tot["el"] += n_el
        tot["sw"] += n_sw
        if n_codes > worst[0]:
            worst = (n_codes, key)
    print(f"  s  sm120 summary: {len(cells)} small-M cells x {len(S_VARIANTS)} variants = {tot['rows']}; the bf16 "
          f"results of the small-M route differ from the cascade's on {tot['bf16']} of {tot['el']} elements "
          f"({tot['rows_bf16']} cell-variants with a difference); linear_quantized differs from "
          f"quantize_activation(linear()) on the small-M route on {tot['codes']} of {tot['el']} codes "
          f"({tot['rows_codes']} cell-variants, most {worst[0]} at {worst[1]}) and {tot['words']} of {tot['sw']} "
          f"scale words; on the cascade (route off) it equals quantize_activation(linear()) bit for bit")


# --- (x) sm_100/sm_103: the weight handle is complete before the next launch (issue #12, FB1) ------------------------
#
# The GEMMs of the sm_100/sm_103 decode row read the weight and its scale before their PDL wait (the weights-first
# prologue, fish_scales_ops.gemm._sm100_decode._WEIGHTS_FIRST); the other routes read nothing before it. A weight
# written by the kernel launched right before a decode-row GEMM would therefore be read before it is complete, and
# prepare_weight ends with one plain launch (fso.dense._complete_handle). The producer in front of each call below is
# the probe of case (m): it releases its programmatic dependents at entry and finishes 2 ms later, so a programmatic
# launch behind it can start at once.

def _x_dst_handle(dw):
    """A handle of dw's shapes, strides and dtypes with zeroed tensors (the destination the producer fills)."""
    w2 = torch.zeros_like(dw.weight)
    s2 = torch.zeros_like(dw.scale)
    return type(dw)(dw.kind, dw.format, dw.arch, dw.in_features, dw.out_features, w2, s2)


def case_x(major: int) -> None:
    """sm_100/sm_103 (issue #12, FB1): the weight handle and the weights-first GEMMs.
      x1  prepare_weight -> linear back to back, both entries (a bf16 x: quantize + GEMM; the pre-quantized x), behind
          a producer that releases its dependents at once and finishes 2 ms later (the weight quantize, a
          programmatic dependent of it, runs only then): every call equals the same call made after a device
          synchronization. prepare_weight ends with one plain launch, which starts only when the weight quantize has
          completed, so neither entry's GEMM can start before the handle is complete.
      x2  a weight written by the kernel launched right before a pre-quantized call (the producer copies the codes
          and the scale words into a zeroed handle 2 ms after releasing its dependents). On the decode row with
          weights first, the control: the weight is read before it is written, and the call differs from the call on
          the final handle (the overlap the plain launch closes). On every other route, and with weights first off,
          the guard: the GEMM reads nothing before its wait, and the call equals the call on the final handle. The
          bf16-x entry behind the same writer is printed (its activation quantize stands between the writer and the
          GEMM; whether the GEMM can overlap the writer depends on where the quantize releases it).
      x2b prepare_weight with the plain launch dropped (fso.dense._complete_handle patched to the identity), behind the
          producer, both entries, eight trials each: the calls that differ from the synchronized call are counted and
          printed (data; how wide this window is depends on where the activation and weight quantizes release their
          dependents).
      x3  a captured graph of linear (both entries) whose weight bytes are rewritten in place by an ordinary torch copy
          (a copy-engine memcpy, and a gather kernel), then replayed: equals the eager call on the new weight. A graph
          launch depends on everything launched before it on the stream (a full dependency: a programmatic edge exists
          only between kernel nodes inside a graph), so the GEMM's early weight reads come after the copy.
      x4  fso.compat.linear_fp8 with FP32 scales, which it packs inside the call (both FP32, and each of the two
          mixes with a pre-packed scale), behind the producer: equal to the call on scales packed beforehand. The
          call writes the weight side first and keeps a plain launch between that write and the GEMM.
    Skipped with FSO_DISABLE_PDL=1 for x1 / x2 / x2b / x4 (nothing overlaps then)."""
    if major != 10:
        print(f"  x  sm_100/sm_103 only; skipped on {_arch_label()}")
        return
    from fish_scales_ops.gemm import _sm100_decode, _sm100_dispatch
    pdl = os.environ.get("FSO_DISABLE_PDL") != "1"
    weights_first = pdl and _sm100_decode._weights_first()
    ext = _pdl_probe() if pdl else None
    spin = 2_000_000  # ns
    if pdl:
        t0, t1 = torch.ones(256, dtype=torch.int16, device="cuda"), torch.zeros(256, dtype=torch.int16, device="cuda")
        ext.slow_copy2(t0, t1, t0, t1, 1, 1000)
        torch.cuda.synchronize()
    tiny_s, tiny_d = torch.ones(64, dtype=torch.int16, device="cuda"), torch.zeros(64, dtype=torch.int16, device="cuda")
    # (N, K, M): the decode row's split-K and persistent forms and its two-CTA tile at M = 64 (the kernels that read
    # the weight before their wait), and short-K cells with a bias: the 1280 x 1280 ones on the row's narrow short-K
    # split-K form, the 5120 x 1280 one on tier 1 (cuBLAS scaled_mm, a plain launch, which reads nothing before the
    # kernel in front has finished). Each cell's route is asked from the router, so the case follows the routes.
    cells = ((5120, 2048, 1), (2048, 4096, 16), (5120, 2048, 64), (1280, 5120, 78), (1280, 1280, 78),
             (1280, 1280, 182), (5120, 1280, 182))
    for N, K, M in cells:
        w = make_w(N, K, seed=1301 + N + K)
        x = make_x(M, K, seed=1302 + M)
        b = (torch.randn(N, device="cuda") * 0.1).bfloat16() if K == 1280 else None
        dw_ref = D.prepare_weight(w, format="mxfp8")
        xq, sx = D.quantize_activation(x)
        torch.cuda.synchronize()
        want = {"bf16-x": D.linear(x, dw_ref, bias=b), "2b": D.linear(xq, dw_ref, x_scale=sx, bias=b)}
        torch.cuda.synchronize()
        calls = {"bf16-x": lambda d: D.linear(x, d, bias=b), "2b": lambda d: D.linear(xq, d, x_scale=sx, bias=b)}
        route = _sm100_dispatch.plan(M, N, K, _q_epilogue(M, N, bias=b) if b is not None else None)
        reads_early = weights_first and route == "sm100.decode"
        tag = f"N={N:5d} K={K:5d} M={M:5d} {route}"
        if pdl:
            # x1: producer -> prepare_weight -> linear
            for entry, call in calls.items():
                ok = True
                for _ in range(4):
                    torch.cuda.synchronize()
                    ext.slow_copy2(tiny_s, tiny_d, tiny_s, tiny_d, 1, spin)
                    dw = D.prepare_weight(w, format="mxfp8")
                    got = call(dw)
                    torch.cuda.synchronize()
                    ok &= bytes_equal(got, want[entry])
                _check(ok, f"x1 {tag} {entry}: prepare_weight -> linear behind a late producer != the synchronized call")
                print(f"  x1 {tag} {entry:6s}: producer -> prepare_weight -> linear == synchronized call {_ok(ok)}")
            # x2: a weight written by the kernel right before the call
            for entry, call in calls.items():
                dst = _x_dst_handle(dw_ref)
                call(dst)  # warm with the destination handle (descriptors, JIT)
                dst.weight.zero_(), dst.scale.zero_()
                torch.cuda.synchronize()
                ext.slow_copy2(dw_ref.weight.view(torch.int16), dst.weight.view(torch.int16),
                               dw_ref.scale.view(torch.int16), dst.scale.view(torch.int16), 8, spin)
                got = call(dst)
                torch.cuda.synchronize()
                differs = not bytes_equal(got, want[entry])
                if entry == "2b" and reads_early:
                    _check(differs, f"x2 {tag} 2b: a weight written by the kernel right before the call was read after "
                                    f"it was complete (the control saw no overlap)")
                    print(f"  x2 {tag} 2b    : weight written 2 ms after its writer released the GEMM: the call "
                          f"differs from the call on the final weight {_ok(differs)} (the overlap is real)")
                elif entry == "2b":
                    _check(not differs, f"x2 {tag} 2b: a GEMM that reads nothing before its wait read a weight written "
                                        f"by the kernel right before the call before it was complete")
                    print(f"  x2 {tag} 2b    : weight written 2 ms after its writer released the GEMM: the call "
                          f"equals the call on the final weight {_ok(not differs)} (nothing read before the wait)")
                else:
                    print(f"  x2 {tag} bf16-x: the same writer in front of the activation quantize: the call "
                          f"{'differs' if differs else 'equals the call on the final weight'} (data)")
            # x2b: prepare_weight without its plain launch
            saved = getattr(D, "_complete_handle", None)
            if saved is None:
                print(f"  x2b {tag}: prepare_weight has no trailing plain launch in this build; skipped")
            else:
                D._complete_handle = lambda dw_: dw_
                try:
                    for entry, call in calls.items():
                        n_diff = 0
                        for _ in range(8):
                            torch.cuda.synchronize()
                            ext.slow_copy2(tiny_s, tiny_d, tiny_s, tiny_d, 1, spin)
                            dw = D.prepare_weight(w, format="mxfp8")
                            got = call(dw)
                            torch.cuda.synchronize()
                            n_diff += 0 if bytes_equal(got, want[entry]) else 1
                        print(f"  x2b {tag} {entry:6s}: prepare_weight WITHOUT its plain launch -> linear: {n_diff} "
                              f"of 8 calls differ from the synchronized call (data)")
                finally:
                    D._complete_handle = saved
        # x3: in-place rewrite, then graph replay
        w2 = make_w(N, K, seed=1303 + N + K)
        dw2 = D.prepare_weight(w2, format="mxfp8")
        torch.cuda.synchronize()
        want2 = {"bf16-x": D.linear(x, dw2, bias=b), "2b": D.linear(xq, dw2, x_scale=sx, bias=b)}
        for entry, call in calls.items():
            dw = D.prepare_weight(w, format="mxfp8")
            call(dw)
            torch.cuda.synchronize()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                call(dw)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                out = call(dw)
            g.replay()
            torch.cuda.synchronize()
            ok0 = bytes_equal(out, want[entry])
            dw.weight.copy_(dw2.weight)  # same dtype, contiguous: a copy-engine memcpy
            dw.scale.copy_(dw2.scale)
            g.replay()
            torch.cuda.synchronize()
            ok1 = bytes_equal(out, want2[entry])
            src_w, dst_w = dw.weight.view(torch.uint8).view(-1), dw.weight.view(torch.uint8).view(-1)
            idx = torch.arange(dw_ref.weight.numel(), device="cuda")
            torch.index_select(dw_ref.weight.view(torch.uint8).view(-1), 0, idx, out=dst_w)  # a gather kernel
            idx_s = torch.arange(dw_ref.scale.numel(), device="cuda")
            torch.index_select(dw_ref.scale.view(-1), 0, idx_s, out=dw.scale.view(-1))
            g.replay()
            torch.cuda.synchronize()
            ok2 = bytes_equal(out, want[entry])
            del src_w
            ok = ok0 and ok1 and ok2
            _check(ok, f"x3 {tag} {entry}: graph replay after an in-place weight rewrite != eager on the new weight "
                       f"(capture {ok0}, memcpy {ok1}, gather kernel {ok2})")
            print(f"  x3 {tag} {entry:6s}: graph replay after an in-place rewrite (memcpy, then a gather kernel) == "
                  f"eager on the new weight {_ok(ok)}")
            del g
        # x4: linear_fp8 packing FP32 scales inside the call, behind the producer
        if pdl:
            wq, sw32 = C.quantize_128x128_fp8(w)
            xq1, sx32 = C.quantize_1x128_fp8(x)
            # the reference packs with the kernels linear_fp8 itself runs (the unchecked torch ops), ahead of the call
            ops = torch.ops.fish_scales_ops
            sxp, swp = ops.repack_fp8_act_scales(sx32.contiguous(), False), ops.repack_fp8_wgt_scales(sw32, False)
            torch.cuda.synchronize()
            ref = C.linear_fp8(xq1, wq, sxp, swp)
            torch.cuda.synchronize()
            ok = True
            for sx_, sw_ in ((sx32, sw32), (sxp, sw32), (sx32, swp)):
                for _ in range(3):
                    torch.cuda.synchronize()
                    ext.slow_copy2(tiny_s, tiny_d, tiny_s, tiny_d, 1, spin)
                    got = C.linear_fp8(xq1, wq, sx_, sw_)
                    torch.cuda.synchronize()
                    ok &= bytes_equal(got, ref)
            _check(ok, f"x4 {tag}: linear_fp8 with scales packed inside the call != the call on pre-packed scales")
            print(f"  x4 {tag}: linear_fp8 packing its FP32 scales in the call (three scale mixes) == pre-packed "
                  f"{_ok(ok)}")


def _finish(ok_line: str) -> int:
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(ok_line)
    return 0


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--s1c-worker":  # a worker of case (o)
        return _s1c_worker(sys.argv[2])
    if len(sys.argv) > 2 and sys.argv[1] == "--p-worker":  # a worker of case (p)
        return p_worker(sys.argv[2])
    if len(sys.argv) > 2 and sys.argv[1] == "--q-worker":  # a worker of case (q)
        return q_worker(sys.argv[2])
    if len(sys.argv) > 2 and sys.argv[1] == "--s3c-worker":  # a worker of case (r)
        return _s3c_worker(sys.argv[2])
    if len(sys.argv) > 2 and sys.argv[1] == "--s-worker":  # a worker of case (s)
        return s_worker(sys.argv[2])
    if len(sys.argv) > 2 and sys.argv[1] == "--s120-worker":  # a worker of case (s), sm_120/sm_121
        return _s120_worker(sys.argv[2])
    if len(sys.argv) > 2 and sys.argv[1] == "--w-worker":  # a worker of case (w)
        return _w_worker(sys.argv[2])
    if len(sys.argv) > 2 and sys.argv[1] == "--t-worker":  # a worker of case (t)
        return _t_worker(sys.argv[2])
    # --case X runs one case alone (a shared card whose lock windows are bounded; the default runs every case).
    only = sys.argv[sys.argv.index("--case") + 1] if "--case" in sys.argv else None

    def wanted(case: str) -> bool:
        return only is None or only == case

    major, minor = _arch()
    print(f"device: {torch.cuda.get_device_name(0) + ' ' if major else ''}{_arch_label()}; torch {torch.__version__}")
    if major not in (9, 10, 12):
        print(f"SKIP: interface v1 needs a dense route (sm_90, sm_100/sm_103, sm_120/sm_121); this is {_arch_label()}")
        return 0
    torch.backends.cuda.matmul.allow_tf32 = False
    if "--sm90-epilogue-mode" in sys.argv:  # child of case (g): one FSO_SM90_DENSE_DG2 mode
        mode = sys.argv[sys.argv.index("--sm90-epilogue-mode") + 1]
        case_g(major, mode)
        case_h(major, mode)
        case_m(major, mode)
        case_n(major, mode)
        return _finish(f"sm_90 epilogue, FSO_SM90_DENSE_DG2={mode}: ALL PASS")
    if "--pdl-off-child" in sys.argv:  # child of case (i): FSO_DISABLE_PDL=1
        torch.save({k: v.cpu() for k, v in _pdl_outputs(major).items()}, os.environ[_PDL_FILE_ENV])
        return 0
    if wanted("a"):
        print("== (a) 2b bytes: linear(xq, w, x_scale=sx) == linear(x, w); quantize_activation == compat ==")
        case_a(major)
        print("== (a2) blocks with amax in (0, 1.00044e-10) ==")
        case_a2(major)
    if wanted("b"):
        print("== (b) refusals, without a device launch ==")
        case_b(major)
    if wanted("c"):
        print("== (c) torch.compile ==")
        case_c(major)
    if wanted("d"):
        print("== (d) the contract header fso/activation_contract.h ==")
        case_d(major)
    if wanted("e"):
        print("== (e) CUDA graph capture on a non-default stream ==")
        case_e(major)
    if wanted("f"):
        print("== (f) fakes, the old four-argument call, activation_contract() ==")
        case_f(major)
    mode = os.environ.get("FSO_SM90_DENSE_DG2") or "default"
    if wanted("g"):
        print(f"== (g) sm_90 epilogue exactness (section 10), FSO_SM90_DENSE_DG2={mode} here, 0 and all in child "
              f"processes ==")
        case_g(major, mode)
        if major == 9 and mode == "default":
            for child in ("0", "all"):
                print(f"  g  child process, FSO_SM90_DENSE_DG2={child}:")
                _sm90_mode_child(child)
        print("== (h) sm_90 epilogue: capture on a non-default stream ==")
        case_h(major, mode)
        print("== (i) sm_90 epilogue: FSO_DISABLE_PDL=1 == default ==")
        case_i(major)
        print("== (j) sm_90 epilogue: two captured calls replayed concurrently on two streams ==")
        case_j(major)
        print("== (k) sm_90 epilogue: torch.compile ==")
        case_k(major)
        print("== (l) sm_90 epilogue: no extra allocation, no synchronization ==")
        case_l(major)
        print("== (m) sm_90 epilogue: bias / residual read only after griddepcontrol.wait (PDL) ==")
        case_m(major, mode)
        print("== (n) sm_90 epilogue: bias / residual at an odd element offset ==")
        case_n(major, mode)
    if wanted("t"):
        print("== (t) sm_90 in-cluster split-K route: routes, section-10 exactness, bf16-x == 2b, determinism, the "
              "one-slice kernel, the split route against the one-CTA route (subprocesses per knob) ==")
        case_t(major)
    if wanted("o"):
        print("== (o) the sm_120/sm_121 dense epilogue on every route (one subprocess per route) ==")
        case_o(major)
    if wanted("w"):
        print("== (w) the sm_120/sm_121 small-M route sm120.smallm: routes, kernels, bytes (one subprocess per knob) ==")
        case_w(major)
    if wanted("p"):
        print("== (p) sm_100/sm_103: bias / erf-GELU / residual on the C++ cascade, every tile family forced ==")
        case_p(major)
    if wanted("q"):
        print("== (q) sm_100/sm_103: the epilogue on the decode row and tier 1, feature-aware fall-through ==")
        case_q(major)
    if wanted("r"):
        print("== (r) sm_120/sm_121: linear_quantized on every route of the MXFP8 cascade (one subprocess per "
              "route) ==")
        case_r(major)
    if wanted("s"):
        print("== (s) linear_quantized beside another route: sm_100/sm_103 on the C++ cascade (every tile family "
              "forced, and the natural routing); sm_120/sm_121 on the cascade beside the small-M route ==")
        case_s(major)
    if wanted("x"):
        print("== (x) sm_100/sm_103: the weight handle is complete before the next launch; weights-first GEMMs ==")
        case_x(major)
    return _finish("fso.dense interface v1: ALL PASS")


if __name__ == "__main__":
    sys.exit(main())
