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
```

`fso.dense` exports `FORMATS`, `DenseWeight`, `prepare_weight`, `linear`,
`supported` and `describe`. The explicit, format-specific ops it is built from
(`quantize_1x128_fp8`, `linear_fp8`, `linear_qx`, `quantize_1x32_fp8`,
`linear_mxfp8` and the rest) stay available in `fso.compat`, which also holds the
reference sections shared by every GEMM op: the scale layouts, the constraints,
the generated `torch.ops` schemas, the CUDA-graph rules and the environment
variables ([`compat.md`](compat.md)).

**No performance numbers in this file**; reference numbers live in
[`../perf/`](../perf/README.md). Code is the source of truth for signatures:
`python/fish_scales_ops/dense/__init__.py`, which also registers
`fish_scales_ops::dense_linear`.

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
# An MXFP8 checkpoint (sm_100/103, sm_120/121): float8_e4m3fn [N, K] with the
# int32 scale fso.compat.quantize_1x32_fp8 returned for it on this architecture,
# as it came out of the quantizer or after a safetensors round trip.
weight = fso.dense.prepare_weight(w_fp8, format="mxfp8", scale=w_scale)

# ---- per call ------------------------------------------------------------------
y = fso.dense.linear(x, weight)    # bf16 [..., K] -> bf16 [..., N]
if bias is not None:               # there is no bias argument; add it outside
    y = y + bias
```

## Formats and the architecture matrix

`format` names what the caller holds, as in `fso.moe.prepare_experts`. The table
says what each architecture does with it. Every refusal names the architecture
and what to use instead, and nothing falls back.

| `format` | what the caller holds | sm_90 (H200) | sm_100 / sm_103 (B200, B300) | sm_120 / sm_121 (RTX 5090) |
|---|---|---|---|---|
| `"bsfp8"` | a float8_e4m3fn weight with fp32 128×128 block scales `[ceil(N/128), K/128]`, weight = fp8 value × block scale; or a bf16 weight and no scale | a checkpoint is kept as it is (made contiguous if it is not); a bf16 weight is quantized with `quantize_128x128_fp8` and its sm_90 default, fp32 scales `amax / 448`. Kind `"bsfp8"` | a checkpoint is dequantized block by block (fp8 value × its block scale in fp32, rounded once to bf16) and requantized to MXFP8 1×32; a bf16 weight is quantized to MXFP8 1×32 directly. Kind `"mxfp8"` | same as sm_100/103 |
| `"mxfp8"` | a bf16 weight and no scale; or a float8_e4m3fn weight with the int32 scale `quantize_1x32_fp8` returned for it on this architecture | `NotImplementedError`: sm_90 has no MXFP8 hardware; use `"bsfp8"` | a bf16 weight is quantized to MXFP8 1×32; a checkpoint is kept as it is. Kind `"mxfp8"` | same as sm_100/103; a checkpoint's scale is restored to the K-major layout the GEMM reads (below) |
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

**MXFP8 checkpoints and the scale's stride.** On sm_120/sm_121,
`quantize_1x32_fp8` returns the weight scale as an int32 `[N, K/128]` tensor with
the strides `(1, N)`: the scale words of one 128-wide K block are adjacent
across rows, which is the order in which the GEMM reads the bytes through a raw
pointer. A safetensors file holds tensors in contiguous order, so a writer
passes the tensor through `.contiguous()`, which keeps every element's value but
rewrites the bytes in row-major order, with the strides `(K/128, 1)`, and a
reader gets that row-major tensor back. A GEMM handed the row-major
tensor reads the scale words in the wrong order, and the result is wrong
without an error. `prepare_weight` accepts both forms and rewrites the
row-major one to the K-major layout; any other strides (a transpose, a slice)
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
alone is copied when it arrives in the row-major form. Freeing the source
parameters of such a checkpoint afterwards releases no memory; a converted or
quantized weight is a new tensor.

## `linear`

`linear(x, weight)` computes `x @ W^T` for a prepared weight. `x` is bf16
`[..., K]` on the device the weight was prepared on; the result is a new bf16
`[..., N]` tensor. The leading dimensions are flattened into `M` for the op and
restored on the result, so a `[B, S, K]` activation gives `[B, S, N]`, a 1-D
`[K]` activation gives `[N]`, and an empty `M` gives an empty result. A
non-contiguous `x` is made contiguous inside the op. There is no bias argument:
a caller with a bias adds it to the result.

The activation is quantized inside the op on every call. The call is
`torch.ops.fish_scales_ops.dense_linear(x2d, weight.weight, weight.scale,
weight.kind)`, a `torch.library.custom_op` whose body dispatches on the
handle's kind and on the device's architecture:

| kind | architecture | what runs |
|---|---|---|
| `"bsfp8"` | sm_90 | `linear_qx`, which quantizes the activation to 1×128 block-FP8 inside the GEMM op and runs the deep_gemm WGMMA kernel. The result is bit-identical to `quantize_1x128_fp8` + `linear_fp8` |
| `"mxfp8"` | sm_120 / sm_121 | the fused 1×32 activation quantize (`quantize_1x32_fp8`), then `linear_mxfp8`, which runs the CUTLASS MXFP8 cascade with Stream-K |
| `"mxfp8"` | sm_100 / sm_103 | the fused 1×32 activation quantize, then `linear_mxfp8`, the Python router: the M ≤ 64 CuTe-DSL decode row, cuBLAS `scaled_mm`, the CuTe-DSL tier and the C++ cascade |
| anything else | any | `NotImplementedError` naming the architecture for a kind of another architecture, `ValueError` for an unknown kind |

The op also checks the metadata of its arguments, so a caller of the raw op
gets an error instead of a silent misread: `x` must be bf16 `[M, K]` and
`weight` float8_e4m3fn `[N, K]`; for kind `"bsfp8"` the scale must be fp32
`[ceil(N/128), K/128]`; for kind `"mxfp8"` it must be int32 `[N, K/128]` with the
strides `(1, N)` on sm_120/121 (the row-major copy of a round trip is refused
here, because only `prepare_weight` restores it), and a contiguous 1-D tensor of
`N * K/128` words on sm_100/103. These checks read shapes, strides and dtypes
only, so they neither synchronize the device nor depend on tensor values.

`tests/gemm/unit/test_dense.py` asserts that `linear` is `torch.equal` to the
explicit composition from `fso.compat` on every architecture: on sm_90 to
`quantize_1x128_fp8` + `linear_fp8` and to `linear_qx`; on sm_100/103 and
sm_120/121 to `quantize_1x32_fp8` + `linear_mxfp8`, for a bf16 weight, for an
MXFP8 checkpoint in both scale stride forms, and for a block-FP8 checkpoint
against the same dequantize and requantize steps done by hand. It covers the
Qwen3-4B projections at `M` in {1, 7, 64, 1000}.

### torch.compile

`fish_scales_ops::dense_linear` ships its own fake implementation, which returns
an empty bf16 `[M, N]` tensor, so `torch.compile` keeps the op as one opaque
node, without a graph break, and the caller registers no fake of its own. A
function that calls `fso.dense.linear` compiles under `fullgraph=True`; the
graph holds exactly one `dense_linear` node, and a second `M` recompiles once
with a symbolic `M` that later `M` reuse. The compiled call runs the same body
and is `torch.equal` to the eager one (`test_dense.py`, case e). The existing
raw ops (`linear_qx`, `linear_mxfp8_raw`, the quantizers and the scale
repacks) still have no fake implementation registered by this library, so a
caller that registers its own fakes for them keeps working.

### CUDA graphs

`linear` can be captured into a CUDA graph after one eager call of the same
shape on the capturing thread, like every op in this library: that call creates
the library's pools (among them the sm_120/121 Stream-K scratch, which the first
Stream-K call of the process sizes for good, and the per-thread workspaces),
JIT-compiles the GEMM on sm_90 and the CuTe-DSL kernels on sm_100/103. Nothing inside the op synchronizes the device;
`test_dense.py` checks an eager call under `torch.cuda.set_sync_debug_mode
("error")` and replays captured calls into NaN-filled outputs against eager.
The shape-dependent decisions (the route, the tile, the empty-`M` case) are
functions of the argument shapes, so one capture per token bucket replays
correctly.

A captured graph reads the weight and scale tensors of the handle it was
captured with, at the addresses they had at capture time. On sm_100/103 the
decode row binds those addresses as raw pointers, which is the general rule of
*CUDA-graph compatibility* in [`compat.md`](compat.md). Rewriting the weight
bytes in place (reloading into the same tensors) and replaying is correct;
preparing a new handle means capturing again.

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
(`torch.ops.fish_scales_ops.jit_compiler_sm90()`: the bundled NVRTC 13.2 and its
path), or the error that keeps that compiler from loading; `describe()` itself
never raises.

## Migrating from the explicit ops and the raw ops

Every existing name keeps working: the explicit ops are in `fso.compat` (and,
until 0.3.0, under their deprecated `fso.gemm` path), and every
`torch.ops.fish_scales_ops.*` op keeps its name, schema and behaviour. What
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
bit-identical to the raw sequence on the K-major scale (`test_dense.py`, case
c). On sm_100/sm_103 `linear_mxfp8` puts the decode row and the cuBLAS and
CuTe-DSL tiers in front of the C++ cascade that `linear_mxfp8_raw` runs, so a
call can land on a different kernel and its result can differ from the raw op's
in rounding. The scale may be passed in the `.contiguous()` form a safetensors
round trip produces; `prepare_weight` restores the K-major layout that
`linear_mxfp8_raw` would otherwise read in the wrong order.

**Fake registrations.** `fish_scales_ops::dense_linear` carries its own fake
implementation, so a caller of `fso.dense.linear` under `torch.compile` needs
none. A caller that keeps calling the raw ops keeps registering its own fakes
for them, as before.

**Bias and activation shapes.** `linear` takes no bias and accepts any number
of leading dimensions; the explicit ops take a 2-D activation and no bias
either.

**Holding the tensors directly.** A caller that wants to keep the prepared
tensors in its own module can store `weight.weight`, `weight.scale` and
`weight.kind` and call `torch.ops.fish_scales_ops.dense_linear(x2d, w, s, kind)`
with a 2-D activation; that is exactly what `linear` does after its argument
checks.
