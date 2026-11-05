# `fish_scales_ops.dense` — public API

The stable dense linear interface: one load-time call that prepares a weight
for this device, and one per-call op that quantizes the activation and runs the
GEMM. The architecture dispatch happens inside a torch custom op, so a caller
writes one code path for sm_90, sm_100/sm_103 and sm_120/sm_121, never names an
architecture, and never handles a scale layout. Changes to these names and
arguments go through a deprecation release.

```python
weight = fso.dense.prepare_weight(w, format="bsfp8", scale=w_scale)  # load time
y      = fso.dense.linear(x, weight)                                 # per call
ok     = fso.dense.supported("mxfp8")   # this device, or arch="sm_90", "sm_103", 120, (12, 0)
text   = fso.dense.describe()           # the architecture matrix, for logs and error messages

# interface v1
xq, sx = fso.dense.quantize_activation(x)        # this device's activation contract
y      = fso.dense.linear(xq, weight, x_scale=sx) # pre-quantized activation: the GEMM alone
y      = fso.dense.linear(x, weight, bias=b, residual=r, activation="gelu_erf")
xq2, sx2 = fso.dense.linear_quantized(x, weight, bias=b, activation="gelu_erf")
info   = fso.dense.activation_contract()          # layout and rounding rule, versioned
inc    = fso.dense.include_dir()                  # holds fso/activation_contract.h
```

`fso.dense` exports `FORMATS`, `ACTIVATIONS`, `DenseWeight`, `prepare_weight`,
`linear`, `linear_quantized`, `quantize_activation`, `activation_contract`,
`include_dir`, `supported` and `describe`.

**Interface v1 status in this build.** The pre-quantized activation
(`quantize_activation` and `linear(xq, weight, x_scale=sx)`) is served on every
architecture. The epilogue arguments (`bias`, `residual`, `activation`) and
`linear_quantized` are validated and passed down to the architecture's GEMM. On
sm_90 the GEMM applies `bias`, `residual` and `activation="gelu_erf"` itself, in
the same launch, on every shape, for a bf16 `x` and for the pre-quantized form.
On sm_100/sm_103 the CuTe-DSL swap-AB row (the M ≤ 64 decode band, the
narrow-N long-K and short-K bands above it, and the identity and erf-GELU
calls it takes on some other short-K cells) and the C++ cascade (tier 3 of the sm_100
router) apply all three, tier 1 (cuBLAS `scaled_mm`) applies the bias, and the
router sends a call to the first tier of its shape that applies every field it
carries; every route of the sm_120/sm_121 MXFP8 cascade and the sm_120/sm_121
small-M route of the decode band (`sm120.smallm`, below) apply all three too.
`linear_quantized`'s result (`xq_next`, `scale_next`) is written in the GEMM's
own launch, with no bf16 intermediate and no quantize kernel, by every route of
the sm_120/sm_121 MXFP8 cascade, which every sm_120/sm_121 `linear_quantized`
call takes (the small-M route does not write the quantized output), and by the
sm_100/sm_103 C++ cascade, to which the router sends every `linear_quantized`
call (no other sm_100/sm_103 tier writes the quantized output). The route that
applies none of the fields (the sm_120/sm_121 block-FP8 1×128 cascade), and
`linear_quantized` on sm_90, raise `NotImplementedError` naming the architecture
and the route, before anything is launched on the device. Without them, every
call is today's, with today's bytes.

The explicit, format-specific ops it is built from
(`quantize_1x128_fp8`, `linear_fp8`, `linear_qx`, `quantize_1x32_fp8`,
`linear_mxfp8` and the rest) stay available in `fso.compat`, which also holds the
reference sections shared by every GEMM op: the scale layouts, the constraints,
the generated `torch.ops` schemas, the CUDA-graph rules and the environment
variables ([`compat.md`](compat.md)).

**No performance numbers in this file**; reference numbers live in
[`../perf/`](../perf/README.md). Code is the source of truth for signatures:
`python/fish_scales_ops/dense/__init__.py`, which also registers
`fish_scales_ops::dense_linear`, `fish_scales_ops::dense_linear_quantized` and
`fish_scales_ops::dense_quantize_activation`.

## Quick start

```python
import torch, fish_scales_ops as fso

# ---- once per weight, at load time, on the serving device ----------------------
# A block-FP8 checkpoint: float8_e4m3fn [N, K] with fp32 [ceil(N/128), K/128]
# block scales (value = fp8 * scale), served on sm_90, sm_100/103 and sm_120/121.
weight = fso.dense.prepare_weight(w_fp8, format="bsfp8", scale=w_scale)
# A bf16 weight, quantized here: block-FP8 on sm_90, MXFP8 1x32 on sm_100/103
# and sm_120/121.
weight = fso.dense.prepare_weight(w_bf16, format="bsfp8")
# An MXFP8 checkpoint (sm_100/103, sm_120/121): float8_e4m3fn [N, K] with uint8
# UE8M0 scales [N, K/32], one byte per 32 elements along K, as the checkpoint
# stores them; the same bytes load on every MXFP8 architecture.
weight = fso.dense.prepare_weight(w_fp8, format="mxfp8", scale=w_scale_u8)
# Or with the int32 scale fso.compat.quantize_1x32_fp8 returned for it on this
# architecture, as it came out of the quantizer or after a safetensors round trip.
weight = fso.dense.prepare_weight(w_fp8, format="mxfp8", scale=w_scale)

# ---- per call ------------------------------------------------------------------
y = fso.dense.linear(x, weight)    # bf16 [..., K] -> bf16 [..., N]
if bias is not None:               # bias= is part of interface v1, but no route of
    y = y + bias                   # this build applies it yet: add it outside

# ---- or with the activation quantized ahead of the call (interface v1) ---------
xq, sx = fso.dense.quantize_activation(x)          # fp8 [M, K] + this device's scale
y = fso.dense.linear(xq, weight, x_scale=sx)       # the GEMM alone; == linear(x, weight)
```

## Formats and the architecture matrix

`format` names what the caller holds, as in `fso.moe.prepare_experts`. The table
says what each architecture does with it. Every refusal names the architecture
and what to use instead, and nothing falls back.

| `format` | what the caller holds | sm_90 (H200) | sm_100 / sm_103 (B200, B300) | sm_120 / sm_121 (RTX 5090) |
|---|---|---|---|---|
| `"bsfp8"` | a float8_e4m3fn weight with fp32 128×128 block scales `[ceil(N/128), K/128]`, weight = fp8 value × block scale; or a bf16 weight and no scale | a checkpoint is kept as it is (made contiguous if it is not); a bf16 weight is quantized with `quantize_128x128_fp8` and its sm_90 default, fp32 scales `amax / 448`. Kind `"bsfp8"` | a checkpoint is dequantized block by block (fp8 value × its block scale in fp32, rounded once to bf16) and requantized to MXFP8 1×32; a bf16 weight is quantized to MXFP8 1×32 directly. Kind `"mxfp8"` | same as sm_100/103 |
| `"mxfp8"` | a bf16 weight and no scale; or a float8_e4m3fn weight with its scale, either the uint8 UE8M0 bytes `[N, K/32]` an MXFP8 checkpoint stores (one byte per 32 elements along K, row-major, byte `b` meaning 2^(b − 127)) or the int32 scale `quantize_1x32_fp8` returned for it on this architecture | `NotImplementedError`: sm_90 has no MXFP8 hardware, whatever the dtypes; use `"bsfp8"` | a bf16 weight is quantized to MXFP8 1×32; a checkpoint keeps its weight bytes, and a uint8 scale is packed into the atom layout. Kind `"mxfp8"` | same as sm_100/103; a uint8 scale is packed into, and an int32 one restored to, the N-fastest layout the GEMM reads (strides `(1, N)`, below) |
| anything else (`"bf16"`, `"int8"`, `"nvfp4"`, …) | — | `NotImplementedError` for a known dialect, `ValueError` for an unknown string | same | same |

Constraints: `K % 128 == 0` on every architecture (one scale per 128 elements
along K). `N % 128 == 0` on sm_100/sm_103 and sm_120/sm_121, whose MXFP8 GEMMs
require it. sm_90 accepts other `N`; `tests/gemm/unit/test_dense.py` covers
`N = 320` there.

`"bsfp8"` on sm_100/sm_103 and sm_120/sm_121 is a double quantization: every
weight is already an fp8 value times its block scale, and requantizing it to
1×32 rounds it a second time. It is the rule `fso.moe.prepare_experts` applies to
block-FP8 experts, and it is the price of serving a block-FP8 checkpoint on the
architectures whose dense path is the MXFP8 GEMM. A checkpoint that still has
its bf16 master can take the single-rounding route by preparing the master with
`"mxfp8"` (or `"bsfp8"`, which quantizes a bf16 weight to MXFP8 directly there).

## `prepare_weight`

`prepare_weight(w, *, format, scale=None)` is called once per weight, at load
time, on the CUDA device that will serve the weight. It validates the weight
and its scale, converts them as the table says, and returns an
`fso.dense.DenseWeight`, a frozen handle with these fields:

| field | meaning |
|---|---|
| `kind` | the execution kind: `"bsfp8"` (block-FP8 weight, fp32 128×128 scales, sm_90) or `"mxfp8"` (MXFP8 1×32 weight, int32 scale handle, sm_100/103 and sm_120/121) |
| `format` | the `format` the caller passed |
| `arch` | the compute capability the handle was prepared on, as `major * 10 + minor` (90, 100, 103, 120, 121) |
| `in_features`, `out_features` | `K` and `N` |
| `weight` | float8_e4m3fn `[N, K]` |
| `scale` | the architecture-native scale tensor: fp32 `[ceil(N/128), K/128]` for kind `"bsfp8"`; for kind `"mxfp8"`, int32 `[N, K/128]` with strides `(1, N)` on sm_120/121 and a 1-D int32 tensor in the CUTLASS block-scaled atom layout on sm_100/103 |

A handle is valid only on the architecture family it was prepared on, and
`linear` refuses a handle from another family. The scale layouts are listed
under *Scale layouts* in [`compat.md`](compat.md).

**MXFP8 checkpoints: the uint8 scale.** The portable form of an MXFP8 scale is
the one an MXFP8 checkpoint stores: uint8 `[N, K/32]`, one UE8M0 byte per 32
elements along K, in the row-major order of the weight, byte `b` meaning the
scale 2^(b − 127). The bytes are the same on every MXFP8 architecture.
`prepare_weight` packs each row's bytes four to an int32 word (the lowest K block
in the lowest byte) and writes the words in this architecture's layout: the
`[N, K/128]` tensor with strides `(1, N)` (N fastest: word `(n, kp)` at
`kp * N + n`) on sm_120/sm_121, the 1-D atom
layout on sm_100/sm_103. The result is exactly the tensor `quantize_1x32_fp8`
returns for the same scales there. Only bytes move, so codes and scales that no
load-time rule would produce are served as written. A uint8 tensor's element
order says which byte is which, so any strides are accepted (a transposed copy,
a slice of a larger buffer) and there is no stride hazard of the kind the int32
form has. A `float8_e8m0fnu` tensor holds the same bytes and is accepted after
`.view(torch.uint8)`.

**MXFP8 checkpoints and the int32 scale's stride.** On sm_120/sm_121,
`quantize_1x32_fp8` returns the weight scale as an int32 `[N, K/128]` tensor with
the strides `(1, N)`: the scale words of one 128-wide K block are adjacent
across rows, which is the order in which the GEMM reads the bytes through a raw
pointer. A safetensors file holds tensors in contiguous order, so a writer
passes the tensor through `.contiguous()`, which keeps every element's value but
rewrites the bytes in row-major order, with the strides `(K/128, 1)`, and a
reader gets that row-major tensor back. A GEMM handed the row-major
tensor reads the scale words in the wrong order, and the result is wrong
without an error. `prepare_weight` accepts both forms and rewrites the
row-major one to the N-fastest layout (strides `(1, N)`); any other strides (a transpose, a slice)
are refused with a `ValueError`. With `K/128 == 1` the two forms are the same
bytes. On sm_100/sm_103 the scale is a 1-D tensor, which a round trip leaves
unchanged, and it must have exactly `N * K/128` words.

**Memory.** A block-FP8 checkpoint on sm_100/sm_103 and sm_120/sm_121 is
converted a whole number of 128-row blocks at a time, holding at most about
256 MiB of fp32 scratch plus the matching bf16 and fp8 rows, so the conversion
of a large projection never holds a second full-precision copy of the weight.
The 1×32 quantizer works row by row along K, so the converted bytes do not
depend on the chunking (`test_dense.py` compares a forced ragged chunking
against one call). A bf16 weight is quantized into a new weight and scale, and
nothing else is allocated. A checkpoint that needs no conversion is kept as the
caller's own tensors (made contiguous if it is not): the block-FP8 checkpoint on
sm_90, and the MXFP8 checkpoint on sm_100/sm_103 and sm_120/sm_121, whose scale
alone is new when it arrives as uint8 bytes or in the row-major int32 form. Freeing the source
parameters of such a checkpoint afterwards releases no memory; a converted or
quantized weight is a new tensor.

**Alignment.** Every tensor of a handle starts on a 16-byte boundary, and the
int32 scale of sm_100/sm_103 on a 32-byte one, which the GEMMs need (the TMA
descriptors, and the CuTe-DSL tier of sm_100/sm_103 for the scale). A checkpoint
tensor that the handle would keep but that is a view at another offset is copied
into a tensor of its own here, once.

## `linear`

`linear(x, weight)` computes `x @ W^T` for a prepared weight. `x` is bf16
`[..., K]` on the device the weight was prepared on; the result is a new bf16
`[..., N]` tensor. The leading dimensions are flattened into `M` for the op and
restored on the result, so a `[B, S, K]` activation gives `[B, S, N]`, a 1-D
`[K]` activation gives `[N]`, and an empty `M` gives an empty result. A bf16
`x` that is not contiguous, or whose data pointer is not 16-byte aligned (a view
that starts at an element offset that is not a multiple of 8, such as
`buf[3:].view(M, K)`), is copied into a tensor of its own first, at the cost of
one extra copy kernel, on every architecture: the activation quantize kernels
read `x` in vectors from its data pointer. The raw ops underneath
(`fso.compat`, `torch.ops.fish_scales_ops`) do not copy; they refuse with
`ValueError` what the kernel they launch cannot read ([`compat.md`](compat.md),
*Constraints*; `tests/gemm/unit/test_misaligned_x.py`). The copy that `linear`
makes of a bf16 `x` that is not contiguous or not 16-byte aligned is an
allocation of that input's size, from the caching allocator on the current
stream, and no transient figure of this library counts it. The weight is never
copied: every route reads it in place and needs it 16-byte aligned (the TMA
descriptors of the C++ GEMMs, the pointer wrappers of the CuTe-DSL and cuBLAS
tiers of sm_100/sm_103), and `linear` refuses, with `ValueError` naming the
tensor and before anything is launched, a handle whose `weight` (or, on kind
`"mxfp8"`, whose int32 `scale`) does not start on a 16-byte boundary. A handle from
`prepare_weight` always does. The fp32 block scales of kind `"bsfp8"` are read
with plain loads and are accepted at any address. The full signature is
`linear(x, weight, *, bias=None, x_scale=None, residual=None, activation=None)`;
`x_scale` is described under *Pre-quantized activation* below and the other three
under *Epilogue arguments*.

The activation is quantized inside the op on every call. The call is
`torch.ops.fish_scales_ops.dense_linear(x2d, weight.weight, weight.scale,
weight.kind)`, a `torch.library.custom_op` whose body dispatches on the
handle's kind and on the device's architecture:

| kind | architecture | what runs |
|---|---|---|
| `"bsfp8"` | sm_90 | `linear_qx`, which quantizes the activation to 1×128 block-FP8 inside the GEMM op and runs the deep_gemm WGMMA kernel (the in-cluster split-K kernel in its band, below). The result is bit-identical to `quantize_activation` + `linear(xq, weight, x_scale=sx)` everywhere, the split-K band included, and to `quantize_1x128_fp8` + `linear_fp8` except on a 128-element group whose amax is below 1.00044e-10 when `K % 512 == 0` (where `quantize_1x128_fp8` floors at `1e-10f` and `linear_qx` at the bf16 value of it) |
| `"mxfp8"` | sm_120 / sm_121 | one op, `linear_mxfp8_bf16x`: in the decode band (`M ≤ 64`), where the small-M route's rule R2 picks it, one kernel (`sm120.smallm`, form G2) that quantizes the activation to 1×32 MXFP8 inside itself and runs the GEMM; elsewhere the fused 1×32 activation quantize (`quantize_1x32_fp8`'s kernel) followed by `linear_mxfp8_raw`, that is the small-M route's form G1 where its rule R1 picks it and the CUTLASS MXFP8 cascade with Stream-K otherwise. The bytes are those of `quantize_activation` + `linear(xq, weight, x_scale=sx)` on every shape |
| `"mxfp8"` | sm_100 / sm_103 | the fused 1×32 activation quantize, then `linear_mxfp8`, the Python router: the CuTe-DSL swap-AB row (the M ≤ 64 decode band and, above it, the narrow-N long-K band: N ≤ 4096, K ≥ 4096; the narrow-N short-K band: N ≤ 4096, 1024 ≤ K ≤ 2560, while its split-K grid has at most 80 CTAs; the identity call of a wide-N short-K cell while the row's 1-SM grid fits one wave; the erf-GELU call of a short-K cell where the cascade's short-K tile rule would leave its 64-wide tile), cuBLAS `scaled_mm`, the CuTe-DSL tier and the C++ cascade (which also takes the bias calls of the narrow-N short-K cells just above the decode band that the swap-AB row leaves: N ≤ 2048, 1024 ≤ K ≤ 2560, 64 < M ≤ 256). The bf16-x call and the pre-quantized call of a shape and epilogue take the same route |
| anything else | any | `NotImplementedError` naming the architecture for a kind of another architecture, `ValueError` for an unknown kind |

The op also checks the metadata of its arguments, so a caller of the raw op
gets an error instead of a silent misread: `x` must be bf16 `[M, K]` and
`weight` float8_e4m3fn `[N, K]`; for kind `"bsfp8"` the scale must be fp32
`[ceil(N/128), K/128]`; for kind `"mxfp8"` it must be int32 `[N, K/128]` with the
strides `(1, N)` on sm_120/121 (the row-major copy of a round trip is refused
here, because only `prepare_weight` restores it), and a contiguous 1-D tensor of
`N * K/128` words on sm_100/103. These checks read shapes, strides and dtypes
only, so they neither synchronize the device nor depend on tensor values.

**The sm_120/sm_121 small-M route (`sm120.smallm`).** The decode band of
sm_120/sm_121 (`M ≤ 64`) has a tensor-core kernel of its own next to the CUTLASS
cascade, deterministic (a fixed summation order) and the same for both of its
forms. The form G1 takes the pre-quantized activation of the contract (the 2b
call `linear(xq, weight, x_scale=sx)`); the form G2 takes the bf16 `x` and
quantizes it inside the kernel by the activation contract, so the bf16-x call is
one launch. G2 gives the bytes of `quantize_activation` + G1. Two rules in
`(M, N, K)` pick the route, with guards on the configuration the kernel takes
for the shape (the host query
`torch.ops.fish_scales_ops.sm120_smallm_rules(M, N, K, sms)` returns their
verdicts and the configuration for a device of `sms` multiprocessors;
`mxfp8_dense_smallm_route_sm120(M, N, K, bf16x)` the route of one call on this
device, launch parameters included): R1 routes the 2b call to G1 when
`pad8(M) · K · (1 + N / 16384) ≤ 73728`, the activation stays resident in shared
memory, the kernel covers `M` with at most five 8-row groups (`M ≤ 40`), and the
`N / 16` (or `N / 32`) tile groups of the kernel's persistent grid fit one round
on the device (at most its SM count); R2 routes the bf16-x call to G2 when
`f(M) ≤ 65536` and, above `M = 8`, `f(pad8(M)) ≤ 65536` as well, with
`f(m) = m · K · (1 + N / 24576) · (K / 2048)^¼`, and R1 holds (the kernel's
configuration changes every 8 rows, so a rule takes a whole 8-row bucket or
none of it; within the first bucket the row count decides). Above `M = 64`
nothing changes. On the RTX 5090 (170 SMs) the rules serve the issue's TP2
projections, `2560 × 2048` up to `M = 24` on both calls and `2048 × 2048` up to
`M = 32` for the 2b call and `M = 24` for the bf16-x call; shapes with more than 2720 output features (5440
for `K ≤ 1152`, where a tile group holds two 16-row tiles) stay on the cascade at
every `M`. The decode cells outside the band stay on the activation quantize +
the cascade.
`FSO_SMALLM_ROUTE=0` turns the
route off (the routes of 0.2.2), `FSO_SMALLM_ROUTE=1` takes it on every call with
`M ≤ 64` the kernel can serve; a forced cascade configuration (`FSO_FORCE_TILE`,
…) turns it off. The route applies `bias`, `residual` and `activation` after its
reduction. It does not write the quantized output: `linear_quantized` runs on
the cascade on every sm_120/sm_121 call (see `linear_quantized` below for what
that means for its bytes).

**The sm_90 in-cluster split-K route.** On sm_90 the decode band of the narrower
weights has a GEMM of its own (`fp8_gemm_kernel_swapAB_splitk`), which the
dispatcher takes before every other sm_90 route. At small `M` the other routes give
every output element to one CTA, so a narrow weight is streamed by one CTA per
weight tile only (40 of the H200's 132 SMs for `N = 2560` at 64-row tiles). The
split-K route gives
each 64-row weight tile a thread-block cluster of `S` CTAs. Each CTA streams one
contiguous slice of `K`, and the cluster adds the `S` fp32 partial sums of every
output element in rank order (`p_0 + p_1 + … + p_{S−1}`) in shared memory, applies
`bias`, `activation` and `residual` once to the sum and rounds once to bf16: one
launch, no workspace. The rule, in `M`, `N`, `K` and the SM count, with
`tiles = ⌈N / 64⌉`:

- `S` is the largest of 4 and 3 whose `S × tiles` CTAs take at most three quarters
  of the SMs and whose slices hold at least 2 blocks of 128 along `K`; else 3 when
  `K > 4096` and `3 × tiles` CTAs fit the SMs; else 2 when `2 × tiles` CTAs fit them
  and the slices hold at least 2 blocks.
- The kernel runs one CTA per SM. When the occupancy calculator cannot place one
  cluster per tile at once, a launch whose token tile has at most 24 rows keeps `S`
  with two CTAs to an SM if that places them, and any other launch drops one slice
  and checks the band again.
- The band: `M ≤ 64` for `K ≥ 8192`, `M ≤ 56` for `4096 ≤ K < 8192`, `M ≤ 40` for
  `2048 ≤ K < 4096` with 3 or 4 slices, and `M ≤ 24` otherwise.

On the H200 that takes every weight up to `N = 4224` in its band: `2560 × 4096`
with 2 slices up to `M = 56`, `2048 × 4096` with 3, `1280 × 5120` with 4,
`2560 × 9216` and `2560 × 9728` with 3 up to `M = 24` and 2 from `M = 25` to 64,
`1024 × 2048` with 4 up to `M = 40`, and `2048 × 512` with 2 up to `M = 24`. Wider weights (`5120 × 2048` and
up) keep their routes. The decision depends on the shape and the device only, so
the identity call, every epilogue call, the bf16-x call and the pre-quantized call
of a shape take the same route.

The route's result is deterministic: the same bytes on every call, replay, stream
and process, and the bf16-x call equals `linear(xq, weight, x_scale=sx)` bit for
bit. It is not bit-identical to the one-CTA routes it replaces, because an
element's 128-wide blocks along `K` are summed per slice and the slices are then
added in rank order, not in `K` order. The fp32 sums therefore differ by about one
rounding of the partial sums. After the rounding to bf16 almost every element is
equal, and the others differ by one unit in the last place. An element whose sum
nearly cancels (a result far below the partial sums) can differ by a few more
units, which is still far below the output's scale. On the 94 cells of the H200
A/B that the route takes (6.0 million bf16-x outputs), 566 elements (0.0094 %)
differed from the routes it replaces. 545 of them differed by one unit in the last
place. The other 21 differed by 2 to 28 units, all on values below 1.6 × 10⁻⁵ in
outputs whose largest magnitude was about 2. The largest absolute difference was
one unit in the last place of a value near 1.3. `FSO_SM90_DENSE_SPLITK=0` turns the
route off and gives the routes and bytes the dispatcher has without it. With
`FSO_SM90_SPLITK_S=1` the kernel runs one slice and sums the blocks in `K` order, as
the one-CTA routes do; on the test's cells its bytes equal those of the route it
replaces.

`tests/gemm/unit/test_dense.py` asserts that `linear` is `torch.equal` to the
explicit composition from `fso.compat` on every architecture: on sm_90 to
`quantize_1x128_fp8` + `linear_fp8` and to `linear_qx`; on sm_100/103 and
sm_120/121 to `quantize_1x32_fp8` + `linear_mxfp8`, for a bf16 weight, for an
MXFP8 checkpoint in both scale stride forms, and for a block-FP8 checkpoint
against the same dequantize and requantize steps done by hand. It covers the
Qwen3-4B projections at `M` in {1, 7, 64, 1000}.
`tests/gemm/unit/test_prequant_mxfp8.py` adds the uint8 scale: the handle it
gives equals the handles of both int32 forms and of the bf16 weight for the same
bytes, random codes and scale bytes reach the handle unchanged, and `linear` is
`torch.equal` across all of them.

## Pre-quantized activation (interface v1)

`quantize_activation(x)` quantizes a bf16 `[..., K]` activation (`K % 128 == 0`)
with this device's activation quantizer and returns `(xq, x_scale)`: `xq` is
float8_e4m3fn `[M, K]` (the leading dimensions flattened into `M`), and `x_scale`
is the scale tensor of the activation contract below. The quantizer is the one
`linear(x, weight)` runs inside the op. On sm_100/sm_103 and sm_120/sm_121 that is
`fso.compat.quantize_1x32_fp8` (MXFP8, UE8M0 scales), with the same bytes. On
sm_90 it is the 1×128 block-FP8 quantize of `linear_qx` (fp32 scales, amax floored
at the bf16 value of 1e-10 for every `K`), with the scales of the padded rows
written as zero; `fso.compat.quantize_1x128_fp8` differs from it for `K % 512 ==
0`, where it floors at `1e-10f`, on a group whose amax is below 1.00044e-10, and
in the padded rows, which it leaves unwritten when `K % 512 != 0`. An empty `x`
gives empty tensors and launches nothing. Like `linear`, it copies a bf16 `x`
that is not contiguous or not 16-byte aligned into a tensor of its own first.

`linear(xq, weight, x_scale=x_scale)` then runs the GEMM alone, without a quantize
launch, and its result is `torch.equal` to `linear(x, weight)` (`tests/gemm/unit/
test_dense_v1.py`, case a, on the issue-12 projections at `M` in {1, 7, 78, 129,
1079}; case a2 adds blocks whose amax lies below the floor). The pair can be produced by another kernel of the caller's, for example a
fused norm and quantize, as long as it follows the contract; the header
`fso/activation_contract.h` states the rule in C++.

The op checks the pre-quantized form from metadata only and raises `ValueError`
for: an fp8 `x` without `x_scale`, an `x_scale` with a bf16 `x`, an `x` that is
not float8_e4m3fn, not 2-D, not contiguous or not 16-byte aligned, and an
`x_scale` whose dtype, device, shape, strides or alignment (16 bytes) differ from
the contract's for this `M` and `K`. On sm_90 and sm_120/sm_121 that includes the
row-major strides `(K/128, 1)` of a `.contiguous()` copy of `x_scale` (and, on
sm_90, of the scale `fso.compat.quantize_1x128_fp8` returns), which would index the
buffer in another order. The message names the contract version and
the expected layout. Nothing reads the device, so a malformed call launches
nothing. Unlike a bf16 `x`, the pre-quantized pair is never copied: its layout
is the caller's.

## Activation contract v1

The contract is the layout and the rounding rule of `(xq, x_scale)`.
`fso.dense.activation_contract()` returns it as a dict (`version`, `arch`,
`format`, `xq`, `x_scale_dtype`, `x_scale_shape`, `x_scale_strides`,
`x_scale_index`, `padded_rows`, `rule`, `header`, `header_functions`,
`include_dir`), and `fso/activation_contract.h` under `fso.dense.include_dir()`
states it in C++ with `kFsoActivationContractVersion = 1` and `__host__
__device__` functions that reproduce the library's quantizers bit for bit. The
header ships inside the package. A caller that writes the pair from its own
kernel adds `include_dir()` to its include path and checks
`kFsoActivationContractVersion` against `activation_contract()["version"]` at
load time; the version changes whenever the layout or the rule does.

In every format `xq` is float8_e4m3fn `[M, K]`, row-major, contiguous and 16-byte
aligned, and `K % 128 == 0`.

| | MXFP8 1×32 (`"mxfp8_1x32"`) | block-FP8 1×128 (`"bsfp8_1x128"`) |
|---|---|---|
| architectures | sm_100/sm_103, sm_120/sm_121 | sm_90 |
| `x_scale` | int32 words; word `(m, kp)` = `b0 \| b1 << 8 \| b2 << 16 \| b3 << 24`, `b_i` the UE8M0 byte of the 32-element block `4 * kp + i` of row `m`, byte `b` meaning 2^(b − 127) | fp32, one scale per 128-element group |
| sm_100/sm_103 layout | 1-D, `align(M, 128) * K/128` words; word `(m, kp)` at `((m / 128) * (K/128) + kp) * 128 + (m % 32) * 4 + (m % 128) / 32` (`fso_mx_sf_word_index_sm100`) | — |
| sm_120/sm_121 layout | `[align(M, 4), K/128]` with strides `(1, align(M, 4))` (M fastest); word `(m, kp)` at `kp * align(M, 4) + m` (`fso_mx_sf_word_index_sm120`); torch indexing is exact: `x_scale[m, kp]` is word `(m, kp)` | — |
| sm_90 layout | — | `[align(M, 4), K/128]` with strides `(1, align(M, 4))` (M fastest); the scale of row `m`, group `kb` at `kb * align(M, 4) + m` (`fso_bsfp8_scale_index`). Torch indexing is exact: `x_scale[m, kb]` is the scale of row `m`, group `kb` |
| padded rows | zero words | `0.f` |
| rule | per 32-element block: `amax` = fp32 max \|x\|, floored at `1e-10f`; `q = 448.f / amax`; `d = 1.f / q`; `b = exp_field(d) + (mantissa(d) != 0)`, capped at 254; code = `cvt.rn.satfinite.e4m3(x * 2^(127 − b))` (`fso_e8m0_from_amax`, `fso_mx_quantize4`) | per 128-element group: `amax` = fp32 max \|x\|, floored at the bf16 value nearest to `1e-10f`, 1.00044e-10 (fp32 bits `0x2EDC0000`), for every `K` (`fso_bsfp8_amax_floor`); `qs = 448.f / amax`; stored scale `1.f / qs` (not a power of two); code = `cvt.rn.satfinite.e4m3(x * qs)` (`fso_bsfp8_quant_scale`, `fso_bsfp8_quantize4`). This is the quantize of `linear_qx`, not `fso.compat.quantize_1x128_fp8` (see above) |

Common to both rules: `x` is the bf16 value converted exactly to fp32, `amax` is
an exact maximum, and each division and product is one IEEE fp32 operation
rounded to nearest; the header's device code uses `__fdiv_rn` / `__fmul_rn`, so a
kernel built with `--use_fast_math` gets the same bytes. NaN inputs are outside
the contract. `test_dense_v1.py` (case d) builds a kernel from the header with
`torch.utils.cpp_extension.load_inline` and checks that its codes and scale words
equal `quantize_activation` exactly, the padded words included, with and without
`--use_fast_math`, and that the header's host path gives the same bytes.

## Epilogue arguments

`linear(x, weight, *, bias=None, x_scale=None, residual=None, activation=None)`
defines the result as

`y[m, n] = RN_bf16( act( acc[m, n] + b[n] ) + r[m, n] )`

with exactly one rounding. `acc` is the fp32 value today's GEMM rounds, after any
split reduction. `b[n]` is `float(bias[n])` with +0.0 replaced by −0.0, so a zero
bias is the identity and gives the bytes of `bias=None`, a −0.0 accumulator
included. `act` is the identity or, for `activation="gelu_erf"`, `0.5f * v * (1.f
+ erff(v * (float)M_SQRT1_2))` in fp32 (torch's `F.gelu` with
`approximate="none"`). `r` is added only when `residual` is given. Every term is
applied once, after the reduction, on every route.

- `bias`: bf16 `[N]`, contiguous, on the weight's device.
- `residual`: bf16 of the result's shape `[..., N]`, contiguous.
- `activation`: `None` or a name of `fso.dense.ACTIVATIONS` (`("gelu_erf",)`).
  Another string is a `ValueError`; a name that exists but that no route serves
  (`"gelu"`, `"gelu_tanh"`, `"silu"`, `"relu"`, …) is a `NotImplementedError`.

`linear_quantized(x, weight, *, bias=None, x_scale=None, activation=None)`
returns `(xq_next, scale_next)`, bit for bit `quantize_activation(linear(x,
weight, bias=bias, x_scale=x_scale, activation=activation))` when the bf16 call
runs on the same route: the epilogue rounds to bf16 first and then applies this
device's activation quantizer rule to the rounded values. `N % 128 == 0` (it is
the next GEMM's `K`); there is no residual argument. On sm_90 the two calls
always take the same route. On sm_120/sm_121 `linear_quantized` always runs on
the MXFP8 cascade, the only sm_120/sm_121 route that writes the quantized
output, while the bf16 call of a decode-band cell that the small-M route takes
(`sm120.smallm`: the 2b call where its rule R1 picks it, the bf16-x call as G2
or as quantize + G1) runs on that route; the result is then bit for bit
`quantize_activation` of the cascade's bf16 result (the bf16 call with
`FSO_SMALLM_ROUTE=0`), and it can differ from `quantize_activation(linear(...))`
in individual codes, because the small-M route sums the `K` products in another
order (each warp sums its own `K` blocks and the warps' sums are then added in a
fixed order). On sm_100/sm_103 `linear_quantized` always runs on the C++
cascade, the only tier that writes the quantized output, while the bf16 call of
a cell owned by the decode row, tier 1 or tier 2 runs on that tier; the result
is then bit for bit `quantize_activation` of the cascade's bf16 result (the raw
op `linear_mxfp8_raw`, which goes straight to the cascade on that architecture),
and it can differ from `quantize_activation(linear(...))` in individual codes
where the other tier sums the `K` products in another order (the decode row's
split-K does). `test_dense_v1.py` case s counts those codes on both
architectures. It copies a misaligned or non-contiguous bf16 `x` and refuses a
handle with a misaligned weight or scale exactly as `linear` does.

Malformed epilogue arguments raise `ValueError`. In this build sm_90 applies
`bias`, `residual` and `activation` on all four of its dense GEMM kernels, each
in its one launch (on the in-cluster split-K kernel of the decode band once,
after the rank-order sum of its slices); on sm_100/sm_103 the CuTe-DSL swap-AB row (the M ≤ 64 decode
band and the bands and calls it takes above it) applies all three inside both of its
kernels (in the in-cluster split-K kernel once, after the fp32 sum), tier 1 (cuBLAS `scaled_mm`) applies the bias through cuBLAS's own
fused epilogue, tier 2 (the CuTe-DSL persistent kernel) applies none, and the
C++ cascade applies all three on every configuration its rules pick and on its
two-kernel split-K (once, in the reduce); the sm_120/sm_121 MXFP8 cascade applies
them on every route, its Stream-K route included (once, in the reduce), and the
sm_120/sm_121 small-M route of the decode band (`sm120.smallm`, both forms)
applies them once after its cross-warp reduction. The fields are applied to the
fp32 accumulator in the order above and the result is rounded to bf16 once. The
sm_100/sm_103 router is feature-aware: the shape picks the route among the tiers
that apply every field of the call, so a tier that owns a cell by shape but lacks
one of its fields hands the cell to the next tier that has it (a residual or GELU
call on a tier-1 cell runs on the swap-AB row's persistent form when the cell is
narrow-N long-K and that form's clusters fit one wave, on the cascade otherwise;
the bias alone stays on tier 1; on short-K cells the router also chooses by the
fields: the swap-AB row takes the identity call of a wide-N cell and the
erf-GELU call where the cascade's short-K tile rule would leave its 64-wide
tile, while the bias and residual calls of those cells keep their tiers), and a
field is never dropped and never refused
for want of a kernel on that architecture. `linear_quantized` is served on
sm_120/sm_121 by every route of the MXFP8 cascade (the single-launch tiles write
`xq_next` and `scale_next` from the GEMM's staged tile; the Stream-K route writes
them in its reduce, after the sum); the small-M route does not write the
quantized output, so every sm_120/sm_121 `linear_quantized` call runs on the
cascade, also on decode-band shapes whose bf16 call takes the small-M route. On
sm_100/sm_103 it is served by the C++ cascade, on every configuration its rules
pick (the GEMM's epilogue writes them) and on its two-kernel split-K (its reduce
writes them, after the sum). No other sm_100/sm_103 tier writes the quantized
output, so the router sends every `linear_quantized` call to the cascade, also on
shapes whose bf16 call takes the swap-AB row, cuBLAS or the DSL tier. On sm_90 it
is not served yet. A well-formed call whose route does not apply a field it
carries (`linear_quantized` on sm_90 among them) raises `NotImplementedError`
that names the architecture and the route (on sm_100/sm_103 the tier the router
would take for that shape and epilogue), before anything is launched. On
sm_100/sm_103 the cascade also refuses, the same way, a residual whose rows are
not 16-byte aligned and an `FSO_FORCE_TILE` row whose configuration has no
dense-epilogue kernel (the sweep-only Stream-K codes 7 and 8 among them); on
sm_120/sm_121 the forced diagnostic instances (`FSO_FORCE_SMALLM`,
`FSO_FORCE_MIN_BLOCKS=2`, `FSO_FORCE_SCHED_GROUP`) refuse them. The per-route
tables come with the phases that implement them.

The explicit ops take the same keyword-only arguments: `fso.compat.linear_fp8`,
`linear_qx` and `linear_mxfp8` accept `bias=`, `residual=` and `activation=`,
apply them on the routes above (sm_90; the sm_100/sm_103 cascade; on
sm_120/sm_121 the MXFP8 cascade and the small-M route of `linear_mxfp8`, not the
block-FP8 1×128 route of `linear_fp8` and `linear_qx`) and refuse them the same
way elsewhere, and the
raw ops `linear_fp8`, `linear_qx` and
`linear_mxfp8_raw` gained `Tensor? bias=None, Tensor? residual=None, int act=0`
as trailing, defaulted arguments (`act` 0 = none, 1 = gelu_erf). Their old
positional calls are unchanged.

### torch.compile

`fish_scales_ops::dense_linear`, `dense_linear_quantized` and
`dense_quantize_activation` ship their own fake implementations, so
`torch.compile` keeps each as one opaque node, without a graph break, and the
caller registers no fake of its own. The fakes of the two tuple-returning ops
return this architecture's real output shapes, strides and dtypes. A function
that calls `fso.dense.linear` compiles under `fullgraph=True`; the graph holds
exactly one `dense_linear` node, and a second `M` recompiles once with a
symbolic `M` that later `M` reuse. The compiled call runs the same body and is
`torch.equal` to the eager one (`test_dense.py`, case e); a function that calls
`quantize_activation` and `linear(xq, weight, x_scale=sx)` holds one node of each
(`test_dense_v1.py`, case c). The existing raw ops (`linear_qx`,
`linear_mxfp8_raw`, the quantizers and the scale repacks) still have no fake
implementation registered by this library, so a caller that registers its own
fakes for them keeps working: torch drops trailing arguments that equal their
defaults before it calls a fake, so a fake written for the old four-argument
schema keeps serving the old calls.

### CUDA graphs

`linear` can be captured into a CUDA graph after one eager call of the same
shape on the capturing thread, like every op in this library: that call creates
the library's pools (among them the sm_120/121 Stream-K side streams and the
per-thread workspaces), JIT-compiles the GEMM on sm_90 and the CuTe-DSL kernels
on sm_100/103. The partial sums of the split routes (the sm_120/121 Stream-K
route, the sm_100/103 two-kernel split-K) come from the torch caching allocator
on the caller's stream, once per call and graph-private inside a capture, so
they have no size for the warm-up to fix and two streams never share them.
Nothing inside the op synchronizes the device;
`test_dense.py` checks an eager call under `torch.cuda.set_sync_debug_mode
("error")` and replays captured calls into NaN-filled outputs against eager.
The shape-dependent decisions (the route, the tile, the empty-`M` case) are
functions of the argument shapes, so one capture per token bucket replays
correctly. The pre-quantized call `linear(xq, weight, x_scale=sx)`, alone or after
`quantize_activation`, captures on a non-default stream after one eager call
(`test_dense_v1.py`, case e).

A captured graph reads the weight and scale tensors of the handle it was
captured with, at the addresses they had at capture time. On sm_100/103 the
decode row binds those addresses as raw pointers, which is the general rule of
*CUDA-graph compatibility* in [`compat.md`](compat.md). Rewriting the weight
bytes in place (reloading into the same tensors) and replaying is correct;
preparing a new handle means capturing again.

On sm_100/103 the GEMM of a dense call may read the handle's weight and scale
before its programmatic-launch wait (the weights-first prologue of the swap-AB
row), and the activation quantize of a call with a bf16 `x` releases the GEMM at
its own entry, before it waits for the kernel in front of it. The kernel
launched immediately before a dense call on the same stream must therefore not
write `weight.weight` or `weight.scale`, in either form of the call; `x`, `xq`,
`sx`, the bias and the residual may come from that kernel. `prepare_weight`
ends with one plain launch, so the very next call may use the handle it
returns. Rewriting the weight bytes in place with an ordinary copy and then
replaying a captured graph stays correct, because a graph launch waits for
everything launched before it on the stream (`test_dense_v1.py`, case x).

## `supported` and `describe`

`supported(format, arch=None)` answers from the architecture matrix: `"bsfp8"`
on sm_90, sm_100/sm_103 and sm_120/sm_121, `"mxfp8"` on sm_100/sm_103 and
sm_120/sm_121, and nothing else anywhere. `arch` defaults to this device and
takes the forms `fso.moe.supported` takes: an int major (9, 10, 12), an int
`major * 10 + minor` (90, 103, 120), a `(major, minor)` pair, or a string such as
`"sm_90"`, `"sm_100a"` or `"12.0"`. The query never raises for a format it does
not know (it answers `False`), so a model-level resolver can ask it before it
chooses this path; an `arch` that names no compute capability raises
`ValueError`. Without a CUDA device it answers `False` for every format.
`describe()` returns the matrix as text, with this device's rows marked. On
sm_90 its last line names the compiler of the JIT that builds the sm_90 kernels
(`torch.ops.fish_scales_ops.jit_compiler_sm90()`: the bundled NVRTC 13.0 and its
path), or the error that keeps that compiler from loading; `describe()` itself
never raises.

## Migrating from the explicit ops and the raw ops

Every existing name keeps working: the explicit ops are in `fso.compat` (and,
until 0.3.0, under their deprecated `fso.gemm` path), and every
`torch.ops.fish_scales_ops.*` op keeps its name and behaviour. Interface v1
appended defaulted arguments to the schemas of `dense_linear`, `linear_fp8`,
`linear_qx` and `linear_mxfp8_raw`; their old positional calls are unchanged. What
follows maps the common call sequences onto `prepare_weight` and `linear`, and
states what changes when a caller makes that move.

**`linear_qx(x, w_fp8, sw)` with the fp32 128×128 scales of a block-FP8
checkpoint.** `prepare_weight(w_fp8, format="bsfp8", scale=sw)` and
`linear(x, weight)` run the same `linear_qx` op on sm_90, and the result is
bit-identical (`test_dense.py`, case a). The same two calls serve that checkpoint
on sm_100/sm_103 and sm_120/sm_121 as well, through MXFP8 after the load-time
requantization described above. `linear_qx` itself takes fp32 scales only, on
every architecture where it exists: it raises a `RuntimeError` for the int32
scales `repack_fp8_wgt_scales` produces, and it raises on sm_100/sm_103, where
it does not exist.

**`quantize_1x32` + `repack_mxfp8_scales` + `linear_mxfp8_raw` with an MXFP8
checkpoint.** `prepare_weight(w_fp8, format="mxfp8", scale=sw)` and
`linear(x, weight)` take the same weight and scale. The activation quantize is
the fused `quantize_1x32_packed`, whose output is bit-exact with the two-step
`quantize_1x32` + `repack_mxfp8_scales` form, and the GEMM is `linear_mxfp8`. On
sm_120/sm_121 that is `linear_mxfp8_raw` with the same tensors, and the result is
bit-identical to the raw sequence on the N-fastest scale (`test_dense.py`, case
c). On sm_100/sm_103 `linear_mxfp8` puts the swap-AB row and the cuBLAS and
CuTe-DSL tiers in front of the C++ cascade that `linear_mxfp8_raw` runs, so a
call can land on a different kernel and its result can differ from the raw op's
in rounding. The scale may be passed in the `.contiguous()` form a safetensors
round trip produces; `prepare_weight` restores the N-fastest layout that
`linear_mxfp8_raw` would otherwise read in the wrong order.

**Fake registrations.** `fish_scales_ops::dense_linear` carries its own fake
implementation, so a caller of `fso.dense.linear` under `torch.compile` needs
none. A caller that keeps calling the raw ops keeps registering its own fakes
for them, as before.

**Bias and activation shapes.** `linear` accepts any number of leading
dimensions for a bf16 activation and takes the epilogue arguments of interface v1
(applied on the routes listed under *Epilogue arguments*); the explicit ops take
a 2-D activation.

**Holding the tensors directly.** A caller that wants to keep the prepared
tensors in its own module can store `weight.weight`, `weight.scale` and
`weight.kind` and call `torch.ops.fish_scales_ops.dense_linear(x2d, w, s, kind)`
with a 2-D activation, or `dense_linear(xq, w, s, kind, None, x_scale)` with a
pre-quantized one; that is exactly what `linear` does after its argument checks.
