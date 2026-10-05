# Integrating fish-scales-ops 0.2.0

This page is for a serving engine that calls fish-scales-ops (fso). It is ordered by task:
- what to call;
- how to build;
- how to serve dense and MoE layers;
- how capture and compile behave;
- what a deployment may set;
- how existing call sites map onto the 0.2.0 interface.

The full contracts are in the per-namespace references:
- [`api/dense.md`](api/dense.md)
- [`api/moe.md`](api/moe.md)
- [`api/attention.md`](api/attention.md)
- [`api/compat.md`](api/compat.md)

[`../CHANGELOG.md`](../CHANGELOG.md) lists what changed. Performance numbers are in [`perf/`](perf/README.md).

## 1. What to call

```python
import fish_scales_ops as fso
```

| task | load time (once per weight or layer) | per call |
|---|---|---|
| dense linear layer, including the LM head | `w = fso.dense.prepare_weight(weight, format=..., scale=...)` | `y = fso.dense.linear(x, w)` |
| routed MoE layer | `e = fso.moe.prepare_experts(w13, w2, format=..., sw13=..., sw2=...)` | `out = fso.moe.layer(hidden, e, topk_ids, topk_w)` |
| shared expert of an MoE block | as a dense layer | its output enters `fso.moe.layer` as `bias=` / `bias_scale=` |
| memory to reserve for the MoE layer | `fso.moe.transient_bytes(e, max_tokens, topk)` | — |
| which formats this device serves | `fso.dense.supported(fmt)`, `fso.moe.supported(fmt)`, and `describe()` for logs | — |
| how the installed copy was built | `fso.build_info()`: the wheel's commit, toolchain and the torch it was built against, or `source_build` | — |
| attention | — | `fso.attention.flash_attn_fwd(q, k, v, ...)` (torch SDPA on every architecture); the sm_120/121 MXFP8 kernels in [`api/attention.md`](api/attention.md) |

**Stable names.** These are the 0.2.0 stable names: `fso.dense`, `fso.moe` and `fso.attention`, and the
one top-level function `fso.build_info()`.
- **Same code on every architecture.** The same calls run on sm_90 (H200), sm_100/sm_103 (B200/B300) and
  sm_120/sm_121 (RTX 5090). The architecture and shape dispatch happen inside one torch custom op per
  layer: `fish_scales_ops::dense_linear` and `fish_scales_ops::moe_layer`. A caller never names an
  architecture, never handles a scale layout and never picks a tile. Nothing falls back silently; a
  combination the device cannot serve raises and names what to use instead.
- **`fso.compat`.** It keeps every explicit, format-specific building block under its existing name: the
  dense quantizers, scale repacks and GEMMs, and the MoE per-step pieces. Code that needs one of them can
  use it. New integration code should not need it.

## 2. Build and install

There are two ways in: a wheel, built once and installed on every serving machine, or an in-place build of
the source tree.

- **The wheel.** `scripts/build_wheel.sh` builds one wheel in a fixed container (CUDA 13.2.1, glibc 2.35,
  Python 3.12, torch 2.13.0+cu130) with the kernels of every serving architecture (`9.0a;10.0f;12.0a`).
  The wheel carries every file the extension reads at run time: the bundled NVRTC, the headers the sm_90
  kernels compile with and the sm_100/sm_103 CuTe-DSL kernel. No source tree is needed where it runs.

  ```bash
  pip install --no-deps fish_scales_ops-0.2.0-cp312-cp312-linux_x86_64.whl
  python -c "import fish_scales_ops as fso; print(fso.build_info())"
  ```

  The serving venv must hold the torch the wheel was built against (`fso.build_info()["torch"]`).
  Importing the package under another torch raises `ImportError`, which names both versions.
- **An in-place build**, for development or for a torch other than the wheel's:

  ```bash
  git clone --recursive https://github.com/fishaudio/fish-scales-ops
  cd fish-scales-ops
  ARCH="9.0a;10.0f;12.0a" EDITABLE=1 ./scripts/build.sh   # every serving architecture in one extension
  # or one machine only: ARCH=9.0a (H200), ARCH=10.0f (B200/B300), ARCH=12.0a (RTX 5090)
  ```

- **Runtime.** torch 2.11 or newer with a CUDA 13 runtime. Build in the serving venv: the extension
  links against the venv's own torch (`build.sh` installs with `--no-build-isolation`).
- **An in-place build keeps reading the source tree.** Two kinds of file come from it:
  - sm_90 compiles its kernels at run time from the deep_gemm and CUDA headers whose directories the
    build recorded. `FSO_JIT_INCLUDE_DIRS` points elsewhere if the tree moves.
  - sm_100/sm_103 load a CuTe-DSL kernel file from `3rdparty/cutlass`. `FSO_DSL_KERNEL_PATH` overrides
    its location.

  So build in place (`EDITABLE=1` or the default in-place build) and keep the tree. A container image
  that keeps the cloned tree, as an image that installs with `pip install -e` does, satisfies this. A
  wheel has neither dependency.
- **The sm_90 compiler is bundled.** The sm_90 kernels are compiled at run time by NVRTC 13.2.78. The
  library bundles that NVRTC and loads it privately, so the NVRTC that torch ships (13.0 in the torch 2.13
  cu130 wheel) and the CUDA toolkit of the image play no part.
  - For an `ARCH` with `9.0a`, `build.sh` downloads the `nvidia-cuda-nvrtc==13.2.78` wheel with pip, checks
    its pinned sha256 and unpacks it into `python/fish_scales_ops/_nvrtc/` (about 120 MB), before it
    compiles.
  - An offline build passes the wheel with `FSO_NVRTC_WHEEL=/path/to/wheel`.
  - `torch.ops.fish_scales_ops.jit_compiler_sm90()` and both `describe()` texts report the compiler in use.
  - Driver: the bundled compiler is validated on driver 595 (the CUDA 13.2 driver). CUDA's minor-version
    compatibility should let a CUDA 13.0 driver (580) load its cubins, but that combination was not
    tested. On such a host, check one sm_90 GEMM call before rollout; `FSO_JIT_NVRTC_LIB` can point at
    torch's own `libnvrtc.so.13` as a fallback.
- **B200/B300 extra.** The CuTe-DSL tiers need `nvidia-cutlass-dsl` (the `sm100` extra in
  `python/pyproject.toml`: `pip install "nvidia-cutlass-dsl>=4.8.0,<5"`). Without it those tiers stay off
  and the other tiers serve every shape. Any other reason a tier cannot load prints one `fso:` line on
  stderr.
- **Check after install.** `print(fso.dense.describe()); print(fso.moe.describe())` on the serving
  device. Both list the formats this device serves and the path each one takes, and their second line
  names the build (version, commit and the torch it was built against).

## 3. Dense layers: `fso.dense`

### Load time

`format` names what the caller holds, not what the device runs:

| the checkpoint holds | call | sm_90 | sm_100/103, sm_120/121 |
|---|---|---|---|
| block-FP8: float8_e4m3fn `[N, K]` + fp32 `[ceil(N/128), K/128]` scales (value = fp8 × scale) | `prepare_weight(w, format="bsfp8", scale=s)` | served as it is (block-FP8, 1×128 activations) | dequantized and requantized to MXFP8 1×32 at load (a second rounding of the weights) |
| bf16 `[N, K]` | `prepare_weight(w, format="bsfp8")` | quantized to block-FP8 | quantized to MXFP8 1×32 |
| bf16 `[N, K]` | `prepare_weight(w, format="mxfp8")` | refused: sm_90 has no MXFP8 | quantized to MXFP8 1×32 |
| MXFP8: float8_e4m3fn `[N, K]` + the int32 scale `fso.compat.quantize_1x32_fp8` returned for it on this architecture | `prepare_weight(w, format="mxfp8", scale=s)` | refused | served as it is |

- **Constraints.** `K % 128 == 0` everywhere. On sm_100/103 and sm_120/121, also `N % 128 == 0`.
- **MXFP8 checkpoints written through safetensors.** Such a checkpoint holds its scale in row-major order
  instead of the K-major order the quantizer produced. `prepare_weight` recognises both forms and
  restores the K-major one. Do not reorder the scale by hand: a GEMM handed the row-major scale computes
  a wrong result without an error.
- **The handle.** It is valid only on the architecture family it was prepared on. Prepare on the serving
  device.

### Per call

```python
y = fso.dense.linear(x, w)   # x: bf16 [..., K]  ->  y: bf16 [..., N]
if bias is not None:
    y = y + bias             # there is no bias argument
```

The activation is quantized inside the op on every call:
- 1×128 block-FP8 on sm_90, through `linear_qx`;
- MXFP8 1×32 on the Blackwell parts, followed by the architecture's MXFP8 GEMM router.

On B200/B300 the router includes the M ≤ 64 decode row and the cuBLAS and CuTe-DSL tiers.

**Tensor parallelism.** A column-parallel shard is `[N/tp, K]` and a row-parallel shard is
`[N, K/tp]`. Each rank prepares its own shard, and the per-rank `K` and, on Blackwell, `N` must stay
multiples of 128. The all-reduce after a row-parallel layer is the caller's, as for any linear layer.

## 4. MoE layers: `fso.moe`

### Load time

```python
experts = fso.moe.prepare_experts(w13, w2, format="bsfp8", sw13=sw13, sw2=sw2)  # block-FP8 experts
experts = fso.moe.prepare_experts(w13, w2, format="mxfp8")                        # bf16 experts -> MXFP8 (Blackwell)
```

`w13 [E_local, 2 * I_local, H]` has the gate rows first. `w2` is `[E_local, H, I_local]`. These are this
rank's local experts, on the serving device.
- **`"bsfp8"`.** Served as it is on sm_90. On sm_100/103 and sm_120/121 each expert is requantized to
  MXFP8 at load, a few experts at a time.
- **`"mxfp8"`.** Takes bf16 experts. It is refused on sm_90.
- **Constraints.** `H % 128 == 0` and `I_local % 128 == 0`. Under tensor parallelism, choose a tp that
  keeps `I_local` a multiple of 128. At most 1024 local experts.

### Per call

```python
out = fso.moe.layer(hidden, experts, topk_ids, topk_w, bias=shared_out, bias_scale=shared_gate)
```

| argument | form |
|---|---|
| `hidden` | bf16 `[M, H]` |
| `topk_ids` | int32 or int64 `[M, topk]` |
| `topk_w` | `[M, topk]`, taken as fp32 |
| `bias` | optional bf16 `[M, H]` |
| `bias_scale` | optional, one factor per token; requires `bias` |

- **Ids outside `[0, E_local)` are skipped at no expert cost.** That covers the padded rows of a graph
  bucket (labelled `num_experts` or −1) and the entries an expert-parallel dispatcher maps away because
  another rank owns the expert. A token whose ids are all skipped comes out as its bias term, or as zero.
- **The result is this rank's partial sum.** The caller reduces it across tensor- and expert-parallel
  ranks, as for any MoE runner.
- **Shared expert.** Run it with `fso.dense` and pass its output as `bias` and its sigmoid gate as
  `bias_scale`. On sm_100/103 and sm_120/121 the add happens inside the combine kernel. On sm_90 it is
  one fused pass after the combine.
- **One call per bucket.** Every bucket is one call; there is no chunking.
  `fso.moe.transient_bytes(experts, tokens, topk)` is a tight upper bound on the memory one call
  allocates for `tokens` rows. It does not count the inputs, the weights or the library's persistent
  pools. Reserve it once for the largest bucket a forward can carry; every MoE layer of the model reuses
  that reservation.

## 5. CUDA graphs and `torch.compile`

- **Eager warm-up before capture.** Make one eager call of each shape on the capturing thread before you
  capture it. That call creates the library's per-thread pools and, on sm_90, JIT-compiles the kernels.
  Capture one graph per token bucket. Every route and tile decision is a function of the shapes, so a
  graph replays correctly for any routing of its bucket.
- **Captured addresses.** A captured graph keeps the addresses of the tensors it captured. Rewriting a
  prepared weight in place and replaying is correct. Preparing a new handle means capturing again.
- **No synchronisation.** Neither op synchronizes the device. Both are captured into graphs in the
  tests, and `test_dense.py` also runs `fso.dense.linear` under
  `torch.cuda.set_sync_debug_mode("error")`.
- **`torch.compile`.** Both ops ship their own fake implementations. `torch.compile`, including
  `fullgraph=True`, keeps each op as one opaque node, and the caller registers no fake of its own. Do not
  register fakes for `fish_scales_ops::dense_linear` or `fish_scales_ops::moe_layer`: torch 2.13 silently
  replaces an existing fake with a second registration.
- **Raw ops.** The older raw ops (`linear_qx`, `linear_mxfp8_raw`, the quantizers and repacks) still have
  no fake from this library. A caller that keeps using them keeps registering its own.

## 6. What a deployment may set

Leave everything unset unless one of these applies. [`api/compat.md`](api/compat.md) lists every
variable.

| variable | when to set it |
|---|---|
| `FSO_MOE_FUSED_COMBINE=1` | sm_120/121: allows the fused-combine FC2 on the buckets whose down-projection slab would dominate the layer's memory. Those buckets are then not bit-reproducible run to run. `transient_bytes` follows the setting |
| `FSO_JIT_INCLUDE_DIRS=a:b:c` | sm_90, an in-place build: the source tree that built the extension moved. A wheel carries its headers |
| `FSO_JIT_NVRTC_LIB=/path/to/libnvrtc.so` | sm_90: compile with another NVRTC library than the bundled 13.2.78. A version other than 13.2 prints one notice. A path that cannot be loaded raises; nothing falls back to torch's NVRTC |
| `FSO_STREAMK_POOL_MB=<n>` | sm_120/121: the dense GEMM's Stream-K scratch is allocated once. A later shape that needs more aborts the process. Size it for the largest dense shape, or make that shape's call the first |
| `FSO_GROUPED_ARG_POOL_MB=<n>` | sm_100/103: the per-thread arena that every captured MoE GEMM pins a block of. Raise it if the process captures very many graphs |
| `FSO_FC1_FUSED=0` | A/B runs only: the unfused FC1 on every architecture. Set it before `prepare_experts`, because it decides the weight layout |

Boolean variables are off when unset, empty or starting with `0`, and on otherwise.

## 7. Migrating existing call sites

Every name a 0.1 caller used keeps working in 0.2.0. Every `torch.ops.fish_scales_ops.*` op keeps its name,
its schema and its behaviour. `fso.gemm.<name>` resolves to the same object as `fso.compat.<name>`, warns
once per name, and is removed in 0.3.0.

| today | 0.2.0 | what changes |
|---|---|---|
| `torch.ops.fish_scales_ops.linear_qx(x, w_fp8, w_scale)` for a block-FP8 checkpoint, with `repack_fp8_wgt_scales` at load on sm_120 | `w = fso.dense.prepare_weight(w_fp8, format="bsfp8", scale=w_scale)` at load; `fso.dense.linear(x, w)` per call | Same kernel and bit-identical result on sm_90. The same two calls also serve the checkpoint on sm_100/103 and sm_120/121, through MXFP8. This removes two failures of the raw call: `linear_qx` takes fp32 scales only, so it raises for the int32 output of `repack_fp8_wgt_scales`, and it does not exist on sm_100/103 |
| `quantize_1x32(x)` + `repack_mxfp8_scales(sx)` + `linear_mxfp8_raw(xq, w, sx, w_scale)` for an MXFP8 checkpoint | `w = fso.dense.prepare_weight(w_fp8, format="mxfp8", scale=w_scale)`; `fso.dense.linear(x, w)` | Bit-identical on sm_120/121. On sm_100/103 the call now goes through the full MXFP8 router; `linear_mxfp8_raw` runs only its last tier, the C++ cascade. A call can therefore land on another kernel there, and its result can differ from the raw op's in rounding. The scale's stride fix-up after a safetensors load moves into `prepare_weight` |
| `fso.gemm.moe_layer_fp8_sm90(hidden, w13, sw13, w2, sw2, topk_ids, topk_w)` | `e = fso.moe.prepare_experts(w13, w2, format="bsfp8", sw13=sw13, sw2=sw2)` at load; `fso.moe.layer(hidden, e, topk_ids, topk_w)` | Bit-identical on sm_90 (`test_moe_unified.py`), and the same calls serve sm_100/103 and sm_120/121. `topk_ids` may stay int64, and the shared expert can enter as `bias` / `bias_scale` |
| `fso.gemm.quantize_128x128_fp8`, `fso.gemm.quantize_1x32_fp8` and the other explicit ops, for example in offline quantization scripts | `fso.compat.<same name>` | Same function objects; only the import path changes |
| the caller's own fake registrations for the raw ops | none for `dense_linear` and `moe_layer` | Keep the fakes only for raw ops the caller still calls |
| `moe_layer_mxfp8_sm120(..., chunk_tokens=...)` / `moe_block_mxfp8_sm120(...)` with `chunk_tokens` or `on_output` (the 0.1 sm_120 entries) | `fso.moe.layer` (one call per bucket), with `fso.moe.transient_bytes` for the reservation | The chunking arguments and `on_output` are gone; the per-arch entries remain in `fso.compat` without them |
| `fso.attention.flash_attn_fwd(..., force_kernel=...)` | drop `force_kernel` | Dispatch is internal |

Behaviour a 0.1 caller can observe is listed in [`../CHANGELOG.md`](../CHANGELOG.md):
- the quantizers' scale-form default;
- sm_90 refusing the Blackwell int32 scale layout;
- the `check` argument of the repack ops;
- the boolean parsing of `FSO_*` variables.

## 8. Checking an integration

- **Run the tests on each serving machine** after installing:

  ```bash
  python tests/gemm/unit/test_dense.py
  python tests/gemm/unit/test_moe_unified.py
  python tests/gemm/unit/test_cuda_graph.py
  python tests/gemm/unit/test_public_surface.py
  ```

  They check the stable entries against the explicit compositions bit for bit. Each skips what the device
  cannot run.
- **Compare against the old path.** When moving a call site, compare the old path and the new one on
  real weights with `torch.equal`. The pairs listed as bit-identical in section 7 must match exactly. A
  difference there means a wrong argument, such as a scale from another architecture or a weight
  prepared on another device.
- **Log `describe()` at start-up.** `fso.dense.describe()` and `fso.moe.describe()` record which path each
  format takes on that machine.
