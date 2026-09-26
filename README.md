# fish-scales-ops

Block-scaled FP8 / MXFP8 GEMM and FlashAttention kernels for LLM serving on
NVIDIA Hopper (H200, sm_90a) and Blackwell (RTX 5090 sm_120a; B300 sm_100 /
sm_103 for MXFP8 GEMM). PyTorch extension plus a standalone C++ library.

<!-- Status (2026-09-05): the Performance section is generated from
     tests/baselines/ by bench/gemm/python/render_perf_docs.py (GEMM baselines
     frozen at the 2026-09-05 pause). All other sections were migrated from
     the previous README on 2026-09-13 under the placement rules in
     docs/README.md; the previous README is archived outside the repository
     (../fso-doc_review-backup-20260915/repo/README.md). -->

## Supported hardware and dtypes

| SM | device | GEMM | attention |
|---|---|---|---|
| sm_90 | H200 | BF16, block-FP8 1×128 (dense + grouped MoE) | torch SDPA fallback |
| sm_120 | RTX 5090 | BF16, block-FP8 1×128, MXFP8 1×32 (dense + grouped MoE) | MXFP8 prefill / paged decode / paged prefill |
| sm_100 / sm_103 | B300 | BF16, MXFP8 1×32 (dense + grouped MoE since 2026-09-15), block-FP8 1×128 (dense; runs on the MXFP8 tcgen05 tiers with replicated scales, added 2026-09-05) | torch SDPA fallback |

## Performance

<!-- POLICY (docs/README.md rule 2): this section covers sm_90 and sm_120 only,
     lists a few hot shapes with absolute µs and TFLOPS, and contains NO
     comparison against any other library. Every row is copied from
     docs/perf/<domain>/<sm>.md at the same commit; docs/perf is the source. -->

Numbers below are CUDA-graph replay medians at locked clocks (H200 GPU 0 runs
of 2026-09-03; RTX 5090 runs of 2026-09-04 / 2026-09-05); the protocol, shape
families and full tables are in [`docs/perf/`](docs/perf/README.md). The rows
are generated from `tests/baselines/` by
`bench/gemm/python/render_perf_docs.py` and change only when a baseline does.
Attention rows are absent until `docs/perf/attention/sm120.md` has an accepted
baseline.

### sm_90 — NVIDIA H200 (132 SMs, 1830 MHz locked)

| family | op | shape | M / S | dtype | µs | TFLOPS |
|---|---|---|---|---|---:|---:|
| A Qwen3-4B | `gate_up` | 19456×2560 | M=1 | block-FP8 1×128 | 17.85 | 6 |
| A Qwen3-4B | `gate_up` | 19456×2560 | M=4096 | block-FP8 1×128 | 360.42 | 1132 |
| B Qwen3-30B-A3B | MoE layer (routed, E=128, top-8) | 1536×2048 + 2048×768 per expert | M=1 | block-FP8 1×128 | 20.0 | 3.8 |
| B Qwen3-30B-A3B | MoE layer (routed, E=128, top-8) | 1536×2048 + 2048×768 per expert | M=2048 | block-FP8 1×128 | 422.2 | 366.2 |
| C Qwen3.5-35B-A3B | MoE block (routed E=256 top-8 + shared expert) | 1024×2048 + 2048×512 per expert, shared 1024×2048 + 2048×512 | M=1 | block-FP8 1×128 | 33.0 | 1.7 |
| C Qwen3.5-35B-A3B | MoE block (routed E=256 top-8 + shared expert) | 1024×2048 + 2048×512 per expert, shared 1024×2048 + 2048×512 | M=2048 | block-FP8 1×128 | 425.9 | 272.3 |

### sm_120 — NVIDIA RTX 5090 (170 SMs, 2400 MHz locked)

| family | op | shape | M / S | dtype | µs | TFLOPS |
|---|---|---|---|---|---:|---:|
| A Qwen3-4B | `gate_up` | 19456×2560 | M=1 | MXFP8 1×32 | 18.52 | 5 |
| A Qwen3-4B | `gate_up` | 19456×2560 | M=4096 | MXFP8 1×32 | 704.21 | 579 |
| A Qwen3-4B | `gate_up` | 19456×2560 | M=4096 | block-FP8 1×128 | 682.94 | 597 |
| B Qwen3-30B-A3B | MoE layer (routed, E=128, top-8) | 1536×2048 + 2048×768 per expert | M=1 | MXFP8 1×32 | 22.6 | 3.3 |
| B Qwen3-30B-A3B | MoE layer (routed, E=128, top-8) | 1536×2048 + 2048×768 per expert | M=2048 | MXFP8 1×32 | 643.3 | 240.3 |
| C Qwen3.5-35B-A3B | MoE block (routed E=256 top-8 + shared expert) | 1024×2048 + 2048×512 per expert, shared 1024×2048 + 2048×512 | M=1 | MXFP8 1×32 | 34.9 | 1.6 |
| C Qwen3.5-35B-A3B | MoE block (routed E=256 top-8 + shared expert) | 1024×2048 + 2048×512 per expert, shared 1024×2048 + 2048×512 | M=2048 | MXFP8 1×32 | 794.6 | 145.9 |

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

## Usage

```python
import torch
import fish_scales_ops as fso

sm = torch.cuda.get_device_capability(0)[0]        # 9 = H200, 10 = B200/B300, 12 = RTX 5090

# ----- GEMM, block-FP8 1x128 / 128x128 (every arch) -----------------------------
wq, sw = fso.gemm.quantize_128x128_fp8(w_bf16)      # UE8M0 scales on Blackwell, FP32 on Hopper
if sm >= 10:
    sw = fso.gemm.repack_fp8_wgt_scales(sw)         # pre-pack once at load time
    xq, sx = fso.gemm.quantize_1x128_fp8_packed(x_bf16)
else:
    xq, sx = fso.gemm.quantize_1x128_fp8(x_bf16)
y = fso.gemm.linear_fp8(xq, wq, sx, sw)             # bf16 [M, N]

# ----- GEMM, MXFP8 1x32 (Blackwell: sm_100/103 and sm_120) ---------------------
if sm >= 10:
    wqm, swm = fso.gemm.quantize_1x32_fp8(w_bf16)
    xqm, sxm = fso.gemm.quantize_1x32_fp8(x_bf16)
    ym = fso.gemm.linear_mxfp8(xqm, wqm, sxm, swm)

# ----- MoE layer, sm_90 (per-expert weights quantized offline) ------------------
out = fso.gemm.moe_layer_fp8_sm90(hidden, w13_fp8, sw13, w2_fp8, sw2, topk_ids, topk_w)

# ----- Attention -------------------------------------------------------------
o = fso.attention.flash_attn_fwd(q, k, v, causal=True)      # torch SDPA on every arch
from fish_scales_ops.attention.backends import sm120_mxfp8   # sm_120 MXFP8 kernels take
o = sm120_mxfp8.mxfp8_fwd(q_fp8, q_sc, k_fp8, k_sc, v_fp8, v_sc, causal=True)   # pre-quantized inputs
```

`fso.gemm` and `fso.attention` are independent namespaces with no top-level
re-exports. The sm_120 MoE layer is composed from six masked-layout ops; the
contracts, scale layouts and constraints of every op are in
[`docs/api/gemm.md`](docs/api/gemm.md) and
[`docs/api/attention.md`](docs/api/attention.md).

## Status

| path | Hopper sm_90 | Blackwell sm_120 | Blackwell datacenter sm_100 / sm_103 |
|---|---|---|---|
| block-FP8 1×128 GEMM | ✓ deep_gemm WGMMA, NVRTC JIT, FP32 scales | ✓ CUTLASS `Sm120BlockScaledKernel`, UE8M0 scales | ✓ since 2026-09-05, on the MXFP8 tcgen05 path with replicated scales |
| MXFP8 1×32 GEMM | — | ✓ CUTLASS block-scaled | ✓ three tiers: cuBLAS `scaled_mm`, CuTe DSL, C++ cascade |
| grouped MoE layer | ✓ expert-sorted contiguous layout with swap-AB decode path (`moe_layer_fp8_sm90`) | ✓ masked slab layout, MXFP8, six ops | ✓ since 2026-09-15 (milestone M3): masked slab layout, MXFP8, CUTLASS pointer-array kernel |
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
- MXFP8 attention is sm_120-only; the paged KV cache needs `page_size ≥ 32`
  (the MXFP8 scale vector), matching sglang's `--page-size 32`; the decode
  kernel caps `H_q / H_kv` at 64.
- B200 / B300 builds use `ARCH="10.0f"`; the B300 has no clock lock, so its
  numbers are measured unlocked and are published only in
  [`docs/perf/gemm/sm100.md`](docs/perf/gemm/sm100.md) and
  [`docs/perf/layer/sm100.md`](docs/perf/layer/sm100.md), never in the section
  above (`docs/perf/README.md` §5 and §8).
- Performance work follows [`docs/perf/README.md`](docs/perf/README.md): same
  device, locked clock, CUDA-graph replay median, every cell traceable to a
  baseline jsonl, and a change is accepted only if every affected cell is
  faster or within ±1 % of the committed baseline.

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
│   ├── gemm/                            fp8.py, mxfp8.py, bf16.py, sm_100 tier router
│   └── attention/                       flash_attn_func.py (SDPA dispatch), backends/sm120_mxfp8*.py
├── tests/{gemm/unit,attention}/         correctness, two-reference FP8 gates, CUDA-graph replay
├── tests/baselines/                     accepted perf runs (jsonl), the source of every table
├── bench/{gemm/python,attention}/       benches, tile sweeps, render_perf_docs.py
├── docs/                                docs/README.md is the map; api/, perf/{gemm,layer,attention}/
└── scripts/build.sh
```

## Tests and benches

```bash
export PYTHONPATH=python
# GEMM correctness (dense BF16/FP8, MXFP8, block-FP8 two-reference gate, grouped MoE, CUDA-graph replay)
python tests/gemm/unit/test_correctness.py
python tests/gemm/unit/test_mxfp8_correctness.py
python tests/gemm/unit/test_fp8_k128_sm120.py          # sm_100 / sm_120
python tests/gemm/unit/test_mxfp8_grouped.py           # sm_120
python tests/gemm/unit/test_moe_routing_threads.py     # sm_100 / sm_103: multi-CTA routing builder under two threads / two streams
python tests/gemm/unit/test_fp8_grouped_sm90.py tests/gemm/unit/test_moe_layer_dispatch_sm90.py   # H200
python tests/gemm/unit/test_cuda_graph.py
# Attention (sm_120)
python -m pytest tests/attention/

# Benches (write jsonl outside the git tree; one process per cell, graph-replay median)
python bench/gemm/python/bench_qwen3_4b_mlp.py --family {qwen3-4b,qwen3-30a3,qwen3.5-35a3} --run --out <dir>/x.jsonl
python bench/gemm/python/bench_qwen3_4b_mlp_forward.py --run --out <dir>/x.jsonl
python bench/gemm/python/bench_moe_qwen3_30a3.py --run --out <dir>/x.jsonl        # Family B MoE layer + kernels
python bench/gemm/python/bench_moe_qwen3_35a3.py --run --out <dir>/x.jsonl        # Family C (routed + shared expert)
python bench/attention/bench_qwen3_llama3.py
python bench/gemm/python/render_perf_docs.py --check   # tables in docs/ and this README match tests/baselines/
```

The measurement protocol, shape families, clock locks and the acceptance
rule for a performance change are in
[`docs/perf/README.md`](docs/perf/README.md).

## Tuning knobs

Debugging and A/B environment variables (`FSO_FORCE_TILE`,
`FSO_FORCE_TILE_K`, `FSO_FORCE_KSPLIT`, `FSO_DISABLE_STREAMK`,
`FSO_DISABLE_PDL`, `FSO_PRINT_TILE_INFO`, the sm_100 tier switches, the
deep_gemm JIT diagnostics and the rest) are listed with their scope in
[`docs/api/gemm.md`](docs/api/gemm.md#env-var-overrides-debugging--a-b-only).
All are read once per process and none is part of the production contract.

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
