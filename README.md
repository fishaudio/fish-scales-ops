# fish-scales-ops

Block-scaled FP8 / MXFP8 GEMM, MoE layer and FlashAttention kernels for LLM
serving on NVIDIA Hopper (H200, sm_90) and Blackwell (B200 / B300, sm_100 /
sm_103; RTX 5090, sm_120), as a PyTorch extension. Current version: 0.2.0.
The changes of each release are in [`CHANGELOG.md`](CHANGELOG.md); a serving
engine starts with the integration guide, [`docs/guide.md`](docs/guide.md).

<!-- The tables in the Performance section are rendered from tests/baselines/ by
     bench/gemm/python/render_perf_docs.py; `render_perf_docs.py --check` reports
     any drift. Every other section follows the placement rules in docs/README.md. -->

## Supported hardware

| feature | H200 (sm_90) | B200 / B300 (sm_100 / sm_103) | RTX 5090 (sm_120) |
|---|---|---|---|
| dense linear, `fso.dense` | block-FP8 1×128 | MXFP8 1×32; MXFP8 checkpoints load as stored, block-FP8 checkpoints are requantized to MXFP8 at load | MXFP8 1×32; MXFP8 checkpoints load as stored, block-FP8 checkpoints are requantized to MXFP8 at load |
| MoE layer, `fso.moe` | block-FP8 experts | MXFP8 (MXFP8, block-FP8 or bf16 experts) | MXFP8 (MXFP8, block-FP8 or bf16 experts) |
| attention, `fso.attention` | torch SDPA | torch SDPA | MXFP8 prefill, paged prefill and paged decode; torch SDPA |

`fso.dense.linear` and `fso.moe.layer` take and return bf16 and quantize the
activations inside; there is no BF16-precision GEMM in this library.

## Install

Requirements: Linux x86_64, Python 3.12, torch **2.13.0+cu130** (the wheel is
built against it and its import refuses another torch), glibc 2.35 or newer,
and an NVIDIA driver for CUDA 13.0 (the tables were measured with driver 595).

```bash
pip install --no-deps fish_scales_ops-0.2.0-cp312-cp312-linux_x86_64.whl   # the wheel of the GitHub release
python -c "import fish_scales_ops as fso; print(fso.__version__, fso.build_info()['commit'])"
```

One wheel serves all three architectures. It is compiled with the same CUDA
13.0 that torch 2.13.0+cu130 is built with, and it carries the NVRTC 13.0.88
that compiles the sm_90 kernels at their first call. `fso.build_info()` returns
how the installed copy was built (commit, toolchain, torch, architectures).

To build the wheel yourself (Docker, about 40 minutes on a large machine):

```bash
git submodule update --init --recursive      # 3rdparty/cutlass
scripts/build_wheel.sh --out dist/           # the wheel, its sha256, BUILD_INFO.json and a test kit
```

[`docs/harness.md`](docs/harness.md) covers building, the GPU test suite and
the performance harness; [`docs/guide.md`](docs/guide.md) §2 covers the
in-place development build.

## Usage

```python
import torch
import fish_scales_ops as fso

# Dense linear, every architecture: prepare once per weight at load time, then one call per batch.
weight = fso.dense.prepare_weight(w_fp8, format="bsfp8", scale=w_scale)  # a block-FP8 checkpoint
# weight = fso.dense.prepare_weight(w_fp8, format="mxfp8", scale=w_ue8m0)  # an MXFP8 checkpoint, as stored (Blackwell)
# weight = fso.dense.prepare_weight(w_bf16, format="mxfp8")             # a bf16 weight, as MXFP8 (Blackwell)
y = fso.dense.linear(x, weight)              # bf16 [..., K] -> bf16 [..., N]

# MoE layer, every architecture: prepare once per layer at load time, then one call per batch.
experts = fso.moe.prepare_experts(w13, w2, format="bsfp8", sw13=sw13, sw2=sw2)  # a block-FP8 checkpoint
# experts = fso.moe.prepare_experts(w13, w2, format="mxfp8", sw13=sw13, sw2=sw2)  # an MXFP8 checkpoint (Blackwell)
# experts = fso.moe.prepare_experts(w13, w2, format="mxfp8")                      # bf16 experts, as MXFP8 (Blackwell)
out = fso.moe.layer(hidden, experts, topk_ids, topk_w)
reserve = fso.moe.transient_bytes(experts, max_tokens, topk)   # bytes one call allocates

# Attention.
o = fso.attention.flash_attn_fwd(q, k, v, causal=True)                              # torch SDPA, every architecture
o = fso.attention.mxfp8_fwd(q_fp8, q_sc, k_fp8, k_sc, v_fp8, v_sc, causal=True)    # RTX 5090, pre-quantized inputs
```

`fso.dense`, `fso.moe` and `fso.attention` are the stable interface. The
architecture dispatch happens inside the library: `fso.dense.linear` and
`fso.moe.layer` are torch custom ops with their own fake implementations, so
they work under `torch.compile` and inside CUDA graphs without help from the
caller. `fso.compat` keeps the explicit, format-specific ops that existing
callers use (`linear_fp8`, the quantizers, the MoE per-step pieces); the old
path `fso.gemm` still resolves to them with a `DeprecationWarning` until 0.3.0.
The contracts, scale layouts and constraints are in
[`docs/api/dense.md`](docs/api/dense.md), [`docs/api/moe.md`](docs/api/moe.md),
[`docs/api/attention.md`](docs/api/attention.md) and
[`docs/api/compat.md`](docs/api/compat.md).

Rules worth knowing up front:

- `K` must be a multiple of 128 everywhere, and `N` a multiple of 128 for the
  Blackwell GEMMs and for every MoE GEMM.
- Capture a call into a CUDA graph only after one eager call of the same shape
  on the capturing thread. That first call creates the library's pools and, on
  sm_90, compiles the kernels.
- On Blackwell the GEMMs consume power-of-two (UE8M0) scales; the quantizers
  default to them there. Quantize on the architecture that runs the GEMM.
- The MXFP8 attention kernels run on the RTX 5090 only. The paged KV cache
  needs a page size that is a multiple of 32, and the decode kernel supports a
  query-to-KV head ratio of at most 64.
- A deployment normally sets no environment variable.
  [`docs/guide.md`](docs/guide.md) §6 lists the few it may set, and
  [`docs/api/compat.md`](docs/api/compat.md#environment-variables) lists every
  variable the library reads.

## Performance

<!-- POLICY (docs/README.md rule 2): this section covers sm_90 and sm_120 only,
     lists a few hot shapes with absolute µs and TFLOPS, and contains NO
     comparison against any other library. -->

CUDA-graph replay medians with cold weights, for the 0.2.0 release wheel; the
H200 runs at its natural clock and the RTX 5090 under its clock lock. The
B300 tables, the comparisons against vLLM, SGLang, TensorRT-LLM and
torch/cuBLAS, and a per-machine summary of where fso is faster or slower are
in [`docs/perf/`](docs/perf/README.md).

### sm_90 — NVIDIA H200 (132 SMs, natural clock)

| family            | op                                             | shape                                                        | M / S  | dtype           |     µs | TFLOPS |
|-------------------|------------------------------------------------|--------------------------------------------------------------|--------|-----------------|-------:|-------:|
| A Qwen3-4B        | `gate_up`                                      | 19456×2560                                                   | M=1    | block-FP8 1×128 |  17.38 |      6 |
| A Qwen3-4B        | `gate_up`                                      | 19456×2560                                                   | M=4096 | block-FP8 1×128 | 333.39 |   1224 |
| B Qwen3-30B-A3B   | MoE layer (routed, E=128, top-8)               | 1536×2048 + 2048×768 per expert                              | M=1    | block-FP8 1×128 |   20.1 |    3.8 |
| B Qwen3-30B-A3B   | MoE layer (routed, E=128, top-8)               | 1536×2048 + 2048×768 per expert                              | M=2048 | block-FP8 1×128 |  331.7 |  466.1 |
| C Qwen3.5-35B-A3B | MoE block (routed E=256 top-8 + shared expert) | 1024×2048 + 2048×512 per expert, shared 1024×2048 + 2048×512 | M=1    | block-FP8 1×128 |   27.6 |    2.1 |
| C Qwen3.5-35B-A3B | MoE block (routed E=256 top-8 + shared expert) | 1024×2048 + 2048×512 per expert, shared 1024×2048 + 2048×512 | M=2048 | block-FP8 1×128 |  360.4 |  321.8 |

### sm_120 — NVIDIA RTX 5090 (170 SMs, 2400 MHz locked)

| family            | op                                             | shape                                                        | M / S  | dtype           |     µs | TFLOPS |
|-------------------|------------------------------------------------|--------------------------------------------------------------|--------|-----------------|-------:|-------:|
| A Qwen3-4B        | `gate_up`                                      | 19456×2560                                                   | M=1    | MXFP8 1×32      |  35.69 |      3 |
| A Qwen3-4B        | `gate_up`                                      | 19456×2560                                                   | M=4096 | MXFP8 1×32      | 661.10 |    617 |
| A Qwen3-4B        | `gate_up`                                      | 19456×2560                                                   | M=4096 | block-FP8 1×128 | 645.76 |    632 |
| B Qwen3-30B-A3B   | MoE layer (routed, E=128, top-8)               | 1536×2048 + 2048×768 per expert                              | M=1    | MXFP8 1×32      |   32.7 |    2.3 |
| B Qwen3-30B-A3B   | MoE layer (routed, E=128, top-8)               | 1536×2048 + 2048×768 per expert                              | M=2048 | MXFP8 1×32      |  546.4 |  283.0 |
| C Qwen3.5-35B-A3B | MoE block (routed E=256 top-8 + shared expert) | 1024×2048 + 2048×512 per expert, shared 1024×2048 + 2048×512 | M=1    | MXFP8 1×32      |   39.4 |    1.4 |
| C Qwen3.5-35B-A3B | MoE block (routed E=256 top-8 + shared expert) | 1024×2048 + 2048×512 per expert, shared 1024×2048 + 2048×512 | M=2048 | MXFP8 1×32      |  703.0 |  165.0 |

## TODO

Open items after 0.2.1: the known issues of [`CHANGELOG.md`](CHANGELOG.md),
whose measured gaps are in the summaries of [`docs/perf/`](docs/perf/README.md),
two planned items, and two build items.

- **H200 dense MLP block at M = 96, 160 and 192 (6–12 %) and at M = 64, 128, 256, 1024 and 8192 (1–3 %)**, against
  sglang's block-FP8 linear (DeepGEMM) ([`docs/perf/layer/sm90.md`](docs/perf/layer/sm90.md), Family A).
  Candidates: a fused SwiGLU and 1×128 quantize kernel, and a DeepGEMM tile order whose L2 traffic does not depend
  on the SMs that the early-started GEMM's thread blocks land on.
- **H200 MoE layer under one-EP-rank routing at M = 2048 (Family B) and M = 8192 (Family C)**, against 0.2.0
  ([`CHANGELOG.md`](CHANGELOG.md), 0.2.1 known issues).
- **RTX 5090 Family C MoE layer at M = 64**, against vLLM's Triton FP8 MoE layer, and at M = 1024 under skewed
  routing against sglang's ([`docs/perf/layer/sm120.md`](docs/perf/layer/sm120.md), Family C).
- **RTX 5090 Family C MoE at M = 1 and 2 against 0.2.0**: the routed layer at M = 2 and the block with the shared
  expert added after the routed layer at M = 1 ([`CHANGELOG.md`](CHANGELOG.md), 0.2.1 known issues).
- **Blackwell dense MLP block at one batch size each**: M = 96 on the B300 against sglang's DeepGEMM linear, M = 1024
  on the RTX 5090 against vLLM's ([`docs/perf/layer/sm100.md`](docs/perf/layer/sm100.md),
  [`docs/perf/layer/sm120.md`](docs/perf/layer/sm120.md), Family A). On the B300 the 2560-wide down projection at
  96–112 tokens needs a swap-AB kernel with small token tiles, or a split-K that reduces inside the cluster.
- **The direct block-FP8 GEMM on Blackwell** (`fso.compat.linear_fp8`), against sglang's DeepGEMM linear on the
  B300 (16–21 % from M = 32 to M = 192) and vLLM's on the RTX 5090 (5–14 % from M = 512). `fso.dense` serves
  block-FP8 checkpoints through the MXFP8 path there; this item concerns callers who pass block-FP8 operands
  themselves. Candidates on the B300: the path's own 1×128 activation quantizer and its cuBLAS launches.
- **MoE layer transient memory at decode batch sizes.** On Blackwell the compact layout reserves a whole 128-row
  block for every expert that can receive a row, so at decode-band M one call's transient
  (`fso.moe.transient_bytes`) is several times that of trtllm-gen's and Triton's MoE layers. Candidates: store the
  down projection's rows at their token positions without block padding, and gather the activation rows inside the
  first grouped GEMM instead of keeping a gathered copy.
- **Cross-rank reduction fused with the MoE GEMM / combine (#6)**, for tensor- and expert-parallel serving over
  PCIe or NVLink: all-reduce and reduce-scatter forms behind a communicator prepared once. Planned for the next
  release.
- **Pin the PTX at build time.** Compile the kernels to PTX once with a pinned front end and turn that PTX into SASS
  with one pinned `ptxas`, so that a toolkit update changes only that last step. 0.2.1 compiles everything with the
  CUDA 13.0 that torch 2.13.0+cu130 uses.
- **Byte-identical rebuilds.** nvcc puts temporary file names into the extension's local symbol names, so two builds
  of one commit hold the same code but are different files ([`docs/harness.md`](docs/harness.md) §1).

## Development

```bash
EDITABLE=1 ./scripts/build.sh                                      # in-place build for development (docs/guide.md §2)
python scripts/ci/run_suite.py --wheel <whl> --testkit <kit> --machine <h200|b300|5090> --work <dir>   # the GPU test suite against a wheel
python bench/run_perf.py --machine <h200|b300|5090> --out <dir>   # the tables of record, under the machine's environment lock
python bench/gemm/python/perf_report.py install --run <dir>       # install a run's tables and render the docs
```

[`docs/harness.md`](docs/harness.md) describes each step;
[`docs/perf/README.md`](docs/perf/README.md) describes how the tables are
measured.

## License

Apache-2.0 (declared in `python/pyproject.toml`). Vendored upstream files under
`csrc/gemm/.../jit/deep_gemm/` keep their original Apache-2.0 NVIDIA copyright
headers. Its subdirectories `comm/`, `common/`, `epilogue/`, `impls/`, `mma/`,
`ptx/` and `scheduler/` are DeepGEMM v2's dense sm_90 kernel from
sgl-project/DeepGEMM v0.2.0 under the MIT License (`deep_gemm/LICENSE.DeepGEMM`).

## Acknowledgments

fish-scales-ops stands on a lot of open-source work; several parts are direct
ports or close adaptations of the projects below, and files that carry
upstream copyright headers keep them.

- **[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)** — the sm_90 block-FP8
  path derives from DeepGEMM's WGMMA kernels and its JIT driver, and the sm_90
  dense GEMM runs DeepGEMM v2's `sm90_fp8_gemm_1d2d` kernel (vendored from
  sgl-project/DeepGEMM v0.2.0) at DeepGEMM's own configuration for the tiles
  fish-scales-ops' kernel cannot run.
- **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)** — the compat shim
  under `csrc/common/compat/include/tensorrt_llm/` is a minimal vendored subset
  of TRT-LLM's common headers, and the block-scaled GEMM runner follows its
  `CutlassFp8BlockScaleGemmRunner` interface.
- **[CUTLASS](https://github.com/NVIDIA/cutlass)** — submodule at
  `3rdparty/cutlass`; the sm_120 and sm_100 kernels build on CuTe, its
  block-scaled MMA atoms and collectives, and the sm_100 DSL tier on the CuTe
  DSL block-scaled example.
- **[FlashAttention](https://github.com/Dao-AILab/flash-attention)** — the
  sm_120 MXFP8 attention kernel follows the FlashAttention v2 / v3 algorithm.
- **[SGLang](https://github.com/sgl-project/sglang)** and
  **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** — the paged
  prefill / decode APIs follow their paged-KV conventions, so integration into
  an sglang attention backend stays mechanical.

If you spot upstream code that is not credited correctly, please open an issue.
