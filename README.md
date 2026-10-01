# fish-scales-ops

Block-scaled FP8 / MXFP8 GEMM, MoE layer and FlashAttention kernels for LLM
serving on NVIDIA Hopper (H200, sm_90a) and Blackwell (RTX 5090 sm_120a;
B200 / B300 sm_100 / sm_103), as a PyTorch extension. Current version: 0.2.0;
the changes of each release are in [`CHANGELOG.md`](CHANGELOG.md).

<!-- Status: the tables in the Performance section are rendered from
     tests/baselines/ by bench/gemm/python/render_perf_docs.py and change only
     when a baseline does; `render_perf_docs.py --check` reports any drift.
     Every other section follows the placement rules in docs/README.md. -->

## Supported hardware and dtypes

| SM | device | GEMM | attention |
|---|---|---|---|
| sm_90 | H200 | block-FP8 1×128 (dense + grouped MoE); `linear_bf16` (bf16 in and out, block-FP8 inside) | torch SDPA fallback |
| sm_120 | RTX 5090 | block-FP8 1×128, MXFP8 1×32 (dense + grouped MoE); `linear_bf16` (bf16 in and out, block-FP8 inside) | MXFP8 prefill / paged decode / paged prefill |
| sm_100 / sm_103 | B300 | MXFP8 1×32 (dense + grouped MoE since 2026-09-15), block-FP8 1×128 (dense; runs on the MXFP8 tcgen05 tiers with replicated scales, added 2026-09-05); `linear_bf16` (bf16 in and out, block-FP8 inside) | torch SDPA fallback |

`linear_bf16` takes and returns bf16 but quantizes both operands to block-FP8
on every call, so its precision is block-FP8's; there is no BF16-precision GEMM
in this library.

## Performance

<!-- POLICY (docs/README.md rule 2): this section covers sm_90 and sm_120 only,
     lists a few hot shapes with absolute µs and TFLOPS, and contains NO
     comparison against any other library. Every row is copied from
     docs/perf/<domain>/<sm>.md at the same commit; docs/perf is the source. -->

Numbers below are CUDA-graph replay medians with cold weights, taken on the
H200 at its natural clock and on the RTX 5090 under its clock lock
(`docs/perf/README.md` §5), and they measure the fish-scales-ops 0.2.0 release
code: the H200 `gate_up` rows and every RTX 5090 row come from the release
perf run of 2026-10-01 on the release build (commit `633f5ca`), and the H200
MoE rows from a run of the same code earlier that day, which the release run
re-measured and matched. The run behind each row is named in the provenance
section of the `docs/perf/` file it comes from. On the RTX 5090 the Family A
and Family B rows ran on one card and the Family C MoE rows on another, so
their absolute prefill numbers are not comparable across families
(`docs/perf/README.md` §5). The protocol, shape families
and full tables are in [`docs/perf/`](docs/perf/README.md). The rows are
generated from `tests/baselines/` by `bench/gemm/python/render_perf_docs.py`
and change only when a baseline does.
Attention rows are absent until `docs/perf/attention/sm120.md` has an accepted
baseline.

### sm_90 — NVIDIA H200 (132 SMs, natural clock)

| family            | op                                             | shape                                                        | M / S  | dtype           |     µs | TFLOPS |
|-------------------|------------------------------------------------|--------------------------------------------------------------|--------|-----------------|-------:|-------:|
| A Qwen3-4B        | `gate_up`                                      | 19456×2560                                                   | M=1    | block-FP8 1×128 |  18.00 |      6 |
| A Qwen3-4B        | `gate_up`                                      | 19456×2560                                                   | M=4096 | block-FP8 1×128 | 412.16 |    990 |
| B Qwen3-30B-A3B   | MoE layer (routed, E=128, top-8)               | 1536×2048 + 2048×768 per expert                              | M=1    | block-FP8 1×128 |   21.9 |    3.4 |
| B Qwen3-30B-A3B   | MoE layer (routed, E=128, top-8)               | 1536×2048 + 2048×768 per expert                              | M=2048 | block-FP8 1×128 |  369.6 |  418.4 |
| C Qwen3.5-35B-A3B | MoE block (routed E=256 top-8 + shared expert) | 1024×2048 + 2048×512 per expert, shared 1024×2048 + 2048×512 | M=1    | block-FP8 1×128 |   35.6 |    1.6 |
| C Qwen3.5-35B-A3B | MoE block (routed E=256 top-8 + shared expert) | 1024×2048 + 2048×512 per expert, shared 1024×2048 + 2048×512 | M=2048 | block-FP8 1×128 |  419.0 |  276.7 |

### sm_120 — NVIDIA RTX 5090 (170 SMs, 2400 MHz locked)

| family            | op                                             | shape                                                        | M / S  | dtype           |     µs | TFLOPS |
|-------------------|------------------------------------------------|--------------------------------------------------------------|--------|-----------------|-------:|-------:|
| A Qwen3-4B        | `gate_up`                                      | 19456×2560                                                   | M=1    | MXFP8 1×32      |  35.68 |      3 |
| A Qwen3-4B        | `gate_up`                                      | 19456×2560                                                   | M=4096 | MXFP8 1×32      | 663.12 |    615 |
| A Qwen3-4B        | `gate_up`                                      | 19456×2560                                                   | M=4096 | block-FP8 1×128 | 647.57 |    630 |
| B Qwen3-30B-A3B   | MoE layer (routed, E=128, top-8)               | 1536×2048 + 2048×768 per expert                              | M=1    | MXFP8 1×32      |   32.4 |    2.3 |
| B Qwen3-30B-A3B   | MoE layer (routed, E=128, top-8)               | 1536×2048 + 2048×768 per expert                              | M=2048 | MXFP8 1×32      |  564.2 |  274.0 |
| C Qwen3.5-35B-A3B | MoE block (routed E=256 top-8 + shared expert) | 1024×2048 + 2048×512 per expert, shared 1024×2048 + 2048×512 | M=1    | MXFP8 1×32      |   38.7 |    1.5 |
| C Qwen3.5-35B-A3B | MoE block (routed E=256 top-8 + shared expert) | 1024×2048 + 2048×512 per expert, shared 1024×2048 + 2048×512 | M=2048 | MXFP8 1×32      |  717.4 |  161.6 |

Row selection rule: per SM, one decode point and one prefill point per shape
family for GEMM, one prefill and one decode row for attention where a native
kernel exists; at most about ten rows per SM.

## Install

```bash
git clone <repo> fish-scales-ops
cd fish-scales-ops
git submodule update --init --recursive      # 3rdparty/cutlass (v4.4.2)

EDITABLE=1 ./scripts/build.sh                # pip install -e python/ (development)
ARCH=12.0a ./scripts/build.sh                # single-arch build, fastest turnaround
ARCH="9.0a;10.0f;12.0a" ./scripts/build.sh   # every deployment arch in one extension
```

`scripts/build.sh` probes `CUDA_HOME`, CUTLASS and the Python headers and
stops early with a clear message when one is missing.

| variable | purpose | default |
|---|---|---|
| `ARCH` | `TORCH_CUDA_ARCH_LIST`; `10.0f` is the sm_100f family target that serves B200 and B300 with one cubin (CUDA ≥ 12.9) | `9.0a;12.0a` |
| `CUDA_HOME` | CUDA toolkit root | `nvcc` on `PATH`, else `/usr/local/cuda` |
| `CUTLASS_DIR` (`BSGEMM_CUTLASS_DIR` accepted) | CUTLASS 4.x root | `3rdparty/cutlass` |
| `PYTHON_INCLUDE`, `PYTHON_INCLUDE_MULTIARCH` | override for `Python.h` / `pyconfig.h` | `sysconfig` |
| `MAX_JOBS` | ninja parallelism | `nproc` |
| `EDITABLE` | `pip install -e` instead of `build_ext --inplace` | `0` |

Two things the build depends on: `ninja` must be on `PATH` (without it
setuptools falls back to distutils, which does not track header dependencies,
so an edited `.cuh` is not recompiled), and `--no-build-isolation` is
deliberate so the extension links against the active venv's torch instead of
a fresh one pulled into a build sandbox (c10 ABI mismatch otherwise). Runtime
requirement: torch 2.11+cu130 or newer with a CUDA 13 runtime. On sm_90 the
GEMM kernels are NVRTC-compiled in the process at first call and bind the
`libnvrtc.so.13` already loaded by torch; the NVRTC build affects kernel speed
(see `docs/perf/README.md` §8).

The built extension keeps reading files from the source tree that built it,
so install in place (`EDITABLE=1`, or the default `build_ext --inplace`) and
keep that tree where it is:

- On sm_90 the JIT compiles each kernel from the CUTLASS headers and the
  vendored deep_gemm headers. Their directories are recorded in the extension
  at build time. If the tree moves, set `FSO_JIT_INCLUDE_DIRS` to the new
  colon-separated list.
- On sm_100/sm_103 the CuTe-DSL tier loads
  `3rdparty/cutlass/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py`
  from the tree. `FSO_DSL_KERNEL_PATH` points it elsewhere.

A wheel installed without the tree does not have these files.

## Usage

```python
import torch
import fish_scales_ops as fso

# ----- Dense linear, every arch: prepare once per weight, then one call per batch ---
weight = fso.dense.prepare_weight(w_fp8, format="bsfp8", scale=w_scale)  # block-FP8 checkpoint
# weight = fso.dense.prepare_weight(w_bf16, format="mxfp8")             # bf16 weight -> MXFP8 (Blackwell)
y = fso.dense.linear(x, weight)     # bf16 [..., K] -> bf16 [..., N]; the arch dispatch is inside the torch op

# ----- MoE layer, every arch: prepare once per layer, then one call per batch ---
experts = fso.moe.prepare_experts(w13, w2, format="bsfp8", sw13=sw13, sw2=sw2)  # or format="mxfp8" (bf16 experts)
out = fso.moe.layer(hidden, experts, topk_ids, topk_w)    # the arch dispatch is inside the torch op
reserve = fso.moe.transient_bytes(experts, max_tokens, topk)   # bytes one call allocates

# ----- Attention -------------------------------------------------------------
o = fso.attention.flash_attn_fwd(q, k, v, causal=True)      # torch SDPA on every arch
o = fso.attention.mxfp8_fwd(q_fp8, q_sc, k_fp8, k_sc, v_fp8, v_sc, causal=True)   # sm_120/121 MXFP8, pre-quantized inputs

# ----- Explicit, format-specific GEMM ops (fso.compat) -------------------------
sm = torch.cuda.get_device_capability(0)[0]          # 9 = H200, 10 = B200/B300, 12 = RTX 5090
wq, sw = fso.compat.quantize_128x128_fp8(w_bf16)      # UE8M0 scales on Blackwell, FP32 on Hopper
if sm >= 10:
    sw = fso.compat.repack_fp8_wgt_scales(sw)         # pre-pack once at load time
    xq, sx = fso.compat.quantize_1x128_fp8_packed(x_bf16)
else:
    xq, sx = fso.compat.quantize_1x128_fp8(x_bf16)
y = fso.compat.linear_fp8(xq, wq, sx, sw)             # bf16 [M, N]
if sm >= 10:                                          # MXFP8 1x32 (sm_100/103, sm_120/121)
    wqm, swm = fso.compat.quantize_1x32_fp8(w_bf16)
    xqm, sxm = fso.compat.quantize_1x32_fp8(x_bf16)
    ym = fso.compat.linear_mxfp8(xqm, wqm, sxm, swm)
```

`fso.dense`, `fso.moe` and `fso.attention` are the three stable namespaces,
independent of each other and without top-level re-exports. `fso.dense` is the
dense linear interface for every architecture: `prepare_weight` converts a
weight for this device at load time, and `linear` is one torch custom op whose
body quantizes the activation and runs the architecture's GEMM. `fso.moe` is the
MoE layer interface for every architecture, built the same way: `prepare_experts`
converts a layer's local experts at load time, and `layer` is one torch custom op
whose body runs the architecture's chain. Both ops ship their own fake
implementations, so `torch.compile` needs nothing from the caller. `fso.attention`
holds torch SDPA on every architecture and the sm_120/121 MXFP8 kernels, which
refuse other architectures.

`fso.compat` keeps the names existing callers use: the explicit, format-specific
dense ops (`linear_fp8`, `linear_qx`, `linear_mxfp8`, their quantizers and scale
repacks, `linear_bf16`), which keep their signatures, and the MoE per-step pieces
(the per-arch layer entries, the sm_120 block, the grouped GEMMs, the routing
builders, the combines and the router), which are building blocks whose
arguments follow the kernels. Before 0.2.0 these names were exported from
`fso.gemm`, where they still resolve with a `DeprecationWarning` until 0.3.0
removes `fso.gemm`. The contracts, scale layouts and constraints are in
[`docs/api/dense.md`](docs/api/dense.md), [`docs/api/moe.md`](docs/api/moe.md),
[`docs/api/attention.md`](docs/api/attention.md) and
[`docs/api/compat.md`](docs/api/compat.md).

## Status

| path | Hopper sm_90 | Blackwell sm_120 | Blackwell datacenter sm_100 / sm_103 |
|---|---|---|---|
| block-FP8 1×128 GEMM | ✓ deep_gemm WGMMA, NVRTC JIT, FP32 scales | ✓ CUTLASS `Sm120BlockScaledKernel`, UE8M0 scales | ✓ since 2026-09-05, on the MXFP8 tcgen05 path with replicated scales |
| `linear_qx` (bf16 activation quantized inside the op, block-FP8 weight) | ✓ bit-identical to `quantize_1x128_fp8` + `linear_fp8` (`tests/gemm/unit/test_linear_qx.py`) | ✓ bit-identical to `quantize_1x128_fp8` + `linear_fp8` (same test) | — refused with a `RuntimeError`; use `quantize_1x128_fp8_packed` + `linear_fp8` |
| MXFP8 1×32 GEMM | — | ✓ CUTLASS block-scaled | ✓ an M ≤ 64 CuTe-DSL decode row in front of three tiers: cuBLAS `scaled_mm`, CuTe DSL, C++ cascade |
| dense linear surface `fso.dense` | ✓ `format="bsfp8"` (block-FP8 checkpoints and bf16 weights), run by `linear_qx` | ✓ `format="bsfp8"` (block-FP8 checkpoints, requantized to MXFP8 at load, and bf16 weights) and `format="mxfp8"` (bf16 weights and MXFP8 checkpoints) | ✓ the same formats and the same code path as sm_120, through the MXFP8 router; validated on the B300 by `test_dense.py` |
| MoE layer surface `fso.moe` | ✓ block-FP8 experts (`format="bsfp8"`) | ✓ block-FP8 experts, requantized to MXFP8 at load, and bf16 experts (`format="mxfp8"`) | ✓ the same formats and the same code path as sm_120, with the sm_100 routing extras; validated on the B300 by the unit tests below (`test_moe_unified.py`, `test_moe_transient_bytes.py`, `test_mxfp8_grouped.py` among them) |
| grouped MoE layer (the `fso.compat` per-step pieces) | ✓ expert-sorted contiguous layout with swap-AB decode path (`moe_layer_fp8_sm90`) | ✓ masked slab layout, MXFP8, composed layer and whole-block entries (`moe_layer_mxfp8_sm120`, `moe_block_mxfp8_sm120`) | ✓ since 2026-09-15 (milestone M3): masked slab layout, MXFP8, CUTLASS pointer-array kernel |
| BF16 attention | torch SDPA | torch SDPA | torch SDPA |
| MXFP8 attention prefill | — | ✓ D ∈ {32, 64, 128, 256}, native GQA | — |
| MXFP8 paged prefill (extend) | — | ✓ page_size a multiple of 32 | — |
| MXFP8 paged decode | — | ✓ split-K across CTAs, plan/run API | — |

Constraints worth knowing up front:

- Blackwell GEMMs consume UE8M0 (power-of-two) scales on both operands; the
  quantizers default to them there and the pre-pack ops reject anything else.
  Scale tensors are arch-native handles — quantize on the arch that runs the
  GEMM.
- `K % 128 == 0` everywhere; `N % 128 == 0` for the Blackwell GEMMs and every
  grouped GEMM.
- Every op can be captured into a CUDA graph after one eager call of the same
  shape on the capturing thread, which creates the library's pools and, on
  sm_90, JIT-compiles the kernels
  ([`docs/api/compat.md`](docs/api/compat.md#cuda-graph-compatibility)). That
  includes `linear_fp8` given FP32 scales and `linear_bf16` on sm_100/103,
  which pack their scales inside the call without a device sync, as on
  sm_120/121. The explicit pre-pack ops `repack_fp8_act_scales` /
  `repack_fp8_wgt_scales` check their scales with a device sync and belong at
  weight-load time, outside any capture.
- MXFP8 attention is sm_120-only; the paged KV cache needs `page_size ≥ 32`
  (the MXFP8 scale vector), matching sglang's `--page-size 32`; the decode
  kernel caps `H_q / H_kv` at 64.
- B200 / B300 builds use `ARCH="10.0f"`. B300 numbers are published only in
  [`docs/perf/gemm/sm100.md`](docs/perf/gemm/sm100.md) and
  [`docs/perf/layer/sm100.md`](docs/perf/layer/sm100.md), never in the section
  above, because that section covers sm_90 and sm_120 only
  (`docs/README.md` rule 2). Like the H200, the B300 is measured at its
  natural clock (`docs/perf/README.md` §5 and §8).
- Performance work follows [`docs/perf/README.md`](docs/perf/README.md): same
  device, the clock policy of its §5 (the RTX 5090 locked, datacenter cards at
  their natural clock with a clock sampler), CUDA-graph replay median with cold
  weights, every cell traceable to a baseline jsonl, and a change is accepted
  only if every affected cell is faster or within ±1 % of the committed
  baseline.

## Repository layout

```
fish-scales-ops/
├── 3rdparty/cutlass/                    submodule, v4.4.2
├── csrc/
│   ├── common/compat/                   vendored TensorRT-LLM-style compat headers
│   ├── gemm/                            block-FP8 1x128 + MXFP8 1x32, dense and grouped
│   │   ├── include/blockscale_gemm/arch/{sm90,sm100,sm120}/   per-arch kernels and cascades
│   │   └── ops/                         PyTorch ops (fp8.cu, mxfp8.cu, quant_kernels.cu, moe_glue.cu)
│   └── attention/                       sm_120 MXFP8 prefill, paged prefill, paged decode
├── python/fish_scales_ops/
│   ├── dense/                           fso.dense: prepare_weight, linear (the dense_linear custom op)
│   ├── moe/                             fso.moe: prepare_experts, layer (the moe_layer custom op)
│   ├── compat/                          fso.compat: the explicit dense ops and the MoE per-step pieces
│   ├── gemm/                            implementation: fp8.py, mxfp8.py, bf16.py, sm_100 tier router (fso.gemm: deprecated path of fso.compat)
│   └── attention/                       fso.attention: SDPA wrapper and the sm_120 MXFP8 entries (backends/)
├── tests/{gemm/unit,attention}/         correctness, two-reference FP8 gates, CUDA-graph replay
├── tests/baselines/                     accepted perf runs (jsonl), the source of every table
├── bench/{gemm/python,attention}/       benches, tile sweeps, render_perf_docs.py
├── docs/                                docs/README.md is the map; api/, perf/{gemm,layer,attention}/
└── scripts/                             build.sh, gen_op_schemas.py (the generated torch.ops blocks of docs/api/)
```

## Tests and benches

```bash
export PYTHONPATH=python
# GEMM correctness (dense BF16/FP8, MXFP8, block-FP8 two-reference gate, grouped MoE, CUDA-graph replay)
# One script per command: `python a.py b.py` runs a.py only.
python tests/gemm/unit/test_public_surface.py          # any machine: the exported names, the deprecated fso.gemm path, the arch refusals
python tests/gemm/unit/test_dense.py                   # every arch: fso.dense against the explicit fso.compat composition, compile, graphs, refusals
python tests/gemm/unit/test_correctness.py
python tests/gemm/unit/test_mxfp8_correctness.py
python tests/gemm/unit/test_fp8_quantization.py
python tests/gemm/unit/test_linear_qx.py             # sm_90 / sm_120: bf16-in GEMM against quantize + linear_fp8; sm_100 refuses
python tests/gemm/unit/test_fp8_k128_sm120.py          # sm_100 / sm_120
python tests/gemm/unit/test_cuda_graph.py
python tests/gemm/unit/test_env_knobs.py               # every arch: the boolean FSO_* switches; sm_100/103: the CuTe-DSL tier notices
python tests/gemm/unit/test_sm100_workspace_growth.py  # sm_100 / sm_103: split-K workspace growth around a captured graph
python tests/gemm/unit/test_sm120_pack_pool_growth.py  # sm_120 / sm_121: linear_bf16 scale-scratch growth around a captured graph
python tests/gemm/unit/test_mxfp8_grouped.py           # sm_100 / sm_120
python tests/gemm/unit/test_mxfp8_decode_sm100.py      # sm_100 / sm_103: the M <= 64 decode row
python tests/gemm/unit/test_moe_routing_threads.py     # every arch: the single-CTA builder; sm_100/103 and sm_120/121: the multi-CTA builder under two threads / two streams
python tests/gemm/unit/test_moe_routing_masked_ids.py  # every arch: expert ids outside [0, E) through the builders and the gather
python tests/gemm/unit/test_moe_router_topk.py         # every arch: fused router top-k vs the torch reference
python tests/gemm/unit/test_moe_unified.py             # every arch: fso.moe against the per-arch entries, the bsfp8 double quantization, the refusals
python tests/gemm/unit/test_moe_transient_bytes.py     # every arch: fso.moe.transient_bytes against the measured peak
python tests/gemm/unit/test_moe_block_sm120.py         # RTX 5090: the composed block and its tp / ep / dp contracts
python tests/gemm/unit/test_moe_layer_determinism_sm120.py   # RTX 5090
python tests/gemm/unit/test_moe_layer_padded_ids_sm120.py    # RTX 5090
python tests/gemm/unit/test_mxfp8_fused_fc1_sm120.py         # RTX 5090
python tests/gemm/unit/test_mxfp8_fused_combine_sm120.py     # RTX 5090
python tests/gemm/unit/test_fp8_grouped_sm90.py              # H200
python tests/gemm/unit/test_fp8_contiguous_sm90.py           # H200
python tests/gemm/unit/test_fp8_contiguous_swapab_sm90.py    # H200
python tests/gemm/unit/test_moe_sorted_sm90.py               # H200
python tests/gemm/unit/test_moe_layer_dispatch_sm90.py       # H200
python tests/gemm/unit/test_moe_layer_determinism_sm90.py    # H200
python tests/gemm/unit/test_moe_layer_padded_ids_sm90.py     # H200
python tests/gemm/unit/test_fp8_contiguous_2wg_sm90.py      # H200: the two-warp-group FC1
python tests/gemm/unit/test_fp8_fused_fc1_sm90.py           # H200: the fused SwiGLU FC1 against the unfused chain
python tests/gemm/unit/test_fp8_fused_fc1_swapab_sm90.py    # H200: the fused swap-AB SwiGLU FC1 against the unfused chain
python tests/gemm/unit/test_jit_cache_sm90.py               # H200: the JIT disk cache (subprocesses, temp dirs)
# Attention (the MXFP8 kernels run on sm_120; elsewhere the tests check the refusals)
python tests/attention/test_smoke.py
python -m pytest tests/attention/

# Benches (write jsonl outside the git tree; one process per cell, graph-replay median)
python bench/gemm/python/bench_qwen3_4b_mlp.py --family {qwen3-4b,qwen3-30a3,qwen3.5-35a3} --run --out <dir>/x.jsonl
python bench/gemm/python/bench_qwen3_4b_mlp_forward.py --run --out <dir>/x.jsonl
python bench/gemm/python/bench_moe_qwen3_30a3.py --run --out <dir>/x.jsonl        # Family B MoE layer + kernels
python bench/gemm/python/bench_moe_qwen3_35a3.py --run --out <dir>/x.jsonl        # Family C (routed + shared expert)
python bench/attention/bench_qwen3_llama3.py
python bench/gemm/python/render_perf_docs.py --check   # tables in docs/ and this README match tests/baselines/
python scripts/gen_op_schemas.py --check               # the generated torch.ops schema blocks of docs/api/ match the source
```

The measurement protocol, shape families, clock policy and the acceptance
rule for a performance change are in
[`docs/perf/README.md`](docs/perf/README.md).

## Environment variables

Every variable the library reads is listed with its scope, default, read time
and consequence in
[`docs/api/compat.md`](docs/api/compat.md#environment-variables). They come in two
kinds:

- **Production switches and capacities**, which a deployment may set on
  purpose: `FSO_MOE_FUSED_COMBINE=1` (the only way to enable the fused combine
  of `fso.moe.layer` on sm_120/121), `FSO_FC1_FUSED` (the fused FC1, and with
  it the weight layout `fso.moe.prepare_experts` writes), `FSO_MOE_BLOCK_OVERLAP`,
  `FSO_DISABLE_DECODE_DSL`, `FSO_JIT_INCLUDE_DIRS`, and the two pool sizes
  `FSO_STREAMK_POOL_MB` and `FSO_GROUPED_ARG_POOL_MB`, whose exhaustion aborts
  the process.
- **Debugging and A/B knobs** (`FSO_FORCE_TILE`, `FSO_FORCE_KSPLIT`,
  `FSO_DISABLE_STREAMK`, `FSO_DISABLE_PDL`, `FSO_PRINT_TILE_INFO`, the sm_100
  tier switches, the deep_gemm JIT diagnostics and the rest), which a correct
  deployment does not need.

Most are read once per process; `FSO_SWAP_BN`, `FSO_SWAP_STAGES`,
`FSO_SWAPAB_CTAS_PER_SM`, `FSO_MOE_SCATTER_WARP` and
`FSO_CHECK_PROBLEM_SHAPES` are read on every call. Set them before the first
call and keep them unchanged between a CUDA-graph capture and its replays.

## License

Apache-2.0 (declared in `python/pyproject.toml`). A top-level `LICENSE` file
lands before the first tagged release. Vendored upstream files under
`csrc/gemm/.../jit/deep_gemm/` keep their original Apache-2.0 NVIDIA copyright
headers.

## Acknowledgments

fish-scales-ops stands on a lot of open-source work; several parts are direct
ports or close adaptations of the projects below, and files that carry
upstream copyright headers keep them.

- **[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)** — the sm_90 block-FP8
  NVRTC-JIT path under `csrc/gemm/include/blockscale_gemm/arch/sm90/fp8/jit/deep_gemm/`
  (compiler driver, JIT utilities, TMA utilities, schedulers, the NVRTC
  C++ / CUTLASS shim) derives from DeepGEMM's WGMMA kernels; the grouped MoE
  schedulers were revived from the same tree.
- **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)** — the compat shim
  under `csrc/common/compat/include/tensorrt_llm/` is a minimal vendored subset
  of TRT-LLM's common headers; the block-scaled GEMM runner shape was extracted
  from its `CutlassFp8BlockScaleGemmRunner` interface.
- **[CUTLASS](https://github.com/NVIDIA/cutlass)** — submodule at
  `3rdparty/cutlass`; the sm_120 and sm_100 kernels build on `cute`, the
  block-scaled MMA atoms, `Sm120BlockScaledKernel` and the tcgen05 collectives,
  and the sm_100 DSL tier on the CuTe DSL block-scaled example.
- **[FlashAttention](https://github.com/Dao-AILab/flash-attention)** — the
  online softmax, KV-tile pipelining and rescale-skip in the sm_120 MXFP8
  attention kernel follow the FlashAttention v2 / v3 algorithm.
- **[SGLang](https://github.com/sgl-project/sglang)** and
  **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** — the paged
  prefill / decode APIs follow their extend-phase and paged-KV `indptr`
  conventions (`BatchPrefillWithPagedKVCache` planner shape), so integration
  into an sglang attention backend stays mechanical; their triton and
  FlashInfer kernels are the same-day comparators in `docs/perf/layer/`.

If you spot upstream code that is not credited correctly, please open an issue.
