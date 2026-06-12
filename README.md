# fish-scales-ops

A production-grade block-scaled FP8 GEMM + FlashAttention library for NVIDIA
**Hopper** (H100 / H200, sm_90a), **Blackwell consumer** (sm_120a) and
**Blackwell datacenter** (B200 / B300, sm_100 / sm_103 — MXFP8 GEMM) GPUs.
Beats PyTorch / cuBLAS / FlashInfer on the shapes that matter for LLM serving.

* **GEMM** — FP8 1×128 (act) × 128×128 (wgt) on sm_90a/sm_120a, plus MXFP8
  1×32 (OCP UE8M0) on sm_120a and on Blackwell datacenter (sm_100/sm_103,
  tcgen05 BlockScaled). CUDA Graph capture-safe. The sm_120a MXFP8 path
  matches or beats `torch.nn.functional.scaled_mm` (cuBLAS / cuBLASLt MXFP8,
  cu13) on **8 of 10** square cubic shapes from 1024³ to 16384³, by up to **+21%**.
  In **end-to-end Qwen3-4B SwiGLU MLP forward**, FSO MXFP8 beats raw cuBLAS
  `scaled_mm` by **1.7-6.2× at every M from 1 to 4096**, and the
  `torch.compile`-fused cuBLAS reference by **2.1-4.3×** at decode shapes
  (M ≤ 128) while staying within **±8%** at prefill (M ≥ 512). The lift comes from two fused
  custom CUDA kernels: one for activation quantize + UE8M0 scale-pack, and
  one that fuses the SwiGLU prologue directly into the down-GEMM activation
  quantize — saving an entire BF16 intermediate write+read pass. The 1×128
  BSFP8 path's quantize kernel was rewritten with the same uint64 LDG.64
  pipeline; **H200 (sm_90) BSFP8 MLP forward is 15-23% faster at prefill
  shapes** (M ≥ 512) on top of the cascade fixes.
* **Attention** — MXFP8 FlashAttention forward on sm_120a: contiguous
  prefill, paged-prefill (extend), and paged decode. Native GQA via
  stride-0 K/V broadcast. **1.4× over torch SDPA prefill**, **7-19× over
  SDPA decode**, **1.2-2.7× over FlashInfer BF16** on extend/decode.

## Headline perf

All numbers are CUDA-Graph capture-and-replay (median of 50 iters × 3 reps),
measured with the **GPU clock locked to a no-boost frequency** so they
reproduce run-to-run and across units (boost clocks vary by thermal / power /
silicon bin). The baseline part is the **RTX PRO 6000 Blackwell Server
Edition** (sm_120a, 188 SMs) locked to **2400 MHz** (`nvidia-smi -lgc
2430,2430`), **CUDA 13.0 / torch 2.11+cu130**. The full matrix — plus the
170-SM Blackwell GPU and H200 (sm_90) tables, all no-boost cu13 — lives in
[`docs/perf.md`](docs/perf.md). Reproduce: lock the clock, then
`bench/gemm/python/bench_qwen3_4b_mlp.py --run`.

### GEMM — square M=N=K, MXFP8 vs cuBLAS `scaled_mm` (RTX PRO 6000, 2400 MHz no-boost, cu13)

Apples-to-apples: same UE8M0 1×32 FP8 numerics + RCEIL, same CUDA-Graph
harness. `sMM` = `F.scaled_mm` (cuBLAS, `BlockWise1x32` + `SWIZZLE_32_4_4`).

|   M=N=K |  BSFP8 TF |  MXFP8 TF | sMM (cuBLAS) TF | MXFP8 vs cuBLAS | MX cos |
|    ---: |      ---: |      ---: |            ---: |            ---: |   ---: |
|    1024 |       149 |       208 |             174 |       **+20%** | 0.9993 |
|    1536 |       236 |       392 |             392 |             +0% | 0.9993 |
|    2048 |       281 |       420 |             441 |             −5% | 0.9993 |
|    2560 |       420 |   **565** |             467 |       **+21%** | 0.9993 |
|    3072 |       506 |       554 |             533 |             +4% | 0.9993 |
|    4096 |       569 |       624 |             655 |             −5% | 0.9993 |
|    6144 |       588 |   **746** |             705 |             +6% | 0.9993 |
|    8192 |       601 |   **750** |             743 |             +1% | 0.9993 |
|   12288 |       744 |       726 |             707 |             +3% | 0.9993 |
| **16384** | **765** | **742** |         **706** |       **+5%** | 0.9993 |

**Peak 750 TF MXFP8 (8192³) / 765 TF BSFP8 (16384³)** — ≈ 81-83% of this
part's 2400 MHz dense-FP8 ceiling (~923 TF; the 1007 TF whitepaper figure is
the 2617 MHz boost spec, unreachable at no-boost). **fso MXFP8 matches or
beats the cu13 cuBLAS `scaled_mm` on 8 of 10 cubic sizes** (7 wins up to
**+21%** at 2560³, +20% at 1024³, +5% at 16384³; a tie at 1536³); only
2048³/4096³ trail by ~5% where cuBLAS keeps a hand-tuned tile cluster. cos = 0.9993 across the sweep. *(The same
MXFP8-vs-cuBLAS comparison on the 170-SM Blackwell GPU — also 2400 MHz no-boost,
cu13 — is in [`docs/perf.md`](docs/perf.md); fso wins there too, by up to
+20%.)*

**Why BSFP8 cos dips at M=N=K=1024.** On sm_120 the BSFP8 path uses
**UE8M0** scales (pure 8-bit exponent, 0 mantissa), so every per-block
scale is rounded **up to the next power of two** — worst case wastes
nearly 2× of E4M3's headroom on that block. BSFP8's K-block is 128
elements wide, so at K=1024 each row of `x` has only **8 scale groups**
and unlucky `amax`-just-above-pow2 blocks aren't statistically averaged
out in the dot product. At K=2048 (16 blocks) cos recovers to 0.9937;
by K≥8192 (64+ blocks) it settles at 0.994-0.996. MXFP8 uses 1×32
groups — 4× more scales per row and a tighter `amax`/element ratio per
group, so pow-2 rounding overhead stays small and cos is flat at
0.9993 across all sizes. The 1024 cell also runs the same `(64,128,4)`
kernel as 2048/4096 (Stream-K is gated at `K ≥ 9728`), so the dip is
purely a scale-quantisation artifact, not a different numerical path.

### MLP block forward (end-to-end SwiGLU, Qwen3-4B — RTX PRO 6000, 2400 MHz no-boost, cu13)

Full activation-quantize → `gate_up` → `silu·gate` → activation-quantize →
`down` closure, captured into one CUDA-Graph and replayed. This is the
per-token MLP cost a production decode loop actually pays. `sMM` = cuBLAS
`scaled_mm` with the reference `to_mxfp`+`to_blocked` quantize; `sMM-c` = the
same with the quantize fused by `torch.compile`.

|     M | FSO MXFP8 µs | cuBLAS sMM µs | sMM-c (compiled) µs | FSO vs sMM | FSO vs sMM-c |
|  ---: |         ---: |          ---: |                ---: |       ---: |          ---: |
|     1 |     **36.9** |         200.5 |               139.5 |   **5.4×** |     **3.8×** |
|    16 |     **32.8** |         204.8 |               139.5 |   **6.2×** |     **4.3×** |
|    64 |     **39.0** |         182.5 |               113.1 |   **4.7×** |     **2.9×** |
|   128 |     **52.0** |         189.5 |               108.6 |   **3.6×** |     **2.1×** |
|   512 |    **179.5** |         342.4 |               180.6 |   **1.9×** |      ≈ tie |
|  1024 |    **299.2** |         509.9 |               277.7 |   **1.7×** |       −7.2% |
|  2048 |    **512.5** |         970.1 |               512.3 |   **1.9×** |      ≈ tie |
|  4096 |   **1041.1** |        2233.2 |              1095.7 |   **2.1×** |     **+5.0%** |

**FSO MXFP8 beats raw cuBLAS `scaled_mm` 1.7–6.2× at every M from 1 to
4096.** Against the `torch.compile`-fused cuBLAS (`sMM-c` — `to_mxfp` +
`to_blocked` fused into one Inductor kernel) FSO is **2.1–4.3× faster at
decode shapes (M ≤ 128)** and within **±8% at prefill** (sMM-c edges FSO by
7% only at M=1024; FSO is ahead at M ≥ 2048).

The wins come from two custom CUDA kernels:

* **`fp8bs_quantize_1x32_packed_kernel`** — fused BF16 → FP8 + packed UE8M0
  int32 scale, replacing the legacy two-step `quantize_1x32` +
  `repack_mxfp8_scales`. uint64 LDG.64 loads (4 BF16/lane), 8-lane
  sub-warp amax reductions via `shfl_xor`, direct IEEE-754 bit
  manipulation for E8M0 RCEIL (skipping `__nv_cvt_float_to_e8m0`).
  Kernel-level microbench: 0.86-1.00× of Inductor's fused quantize
  across all shapes — 10-14% faster at M ∈ {1024, 2048}, ties at
  GDDR7-bandwidth-bound 4096×9728+.
* **`silu_chunk_mul_quantize_1x32_packed_kernel`** — fused SwiGLU
  prologue: takes `gu = [gate || up]` (the gate_up GEMM output) and
  emits FP8 + packed scale directly, **without materialising the BF16
  `h = silu(gate) * up` intermediate in global memory**. Saves
  M·INTER·2 bytes of read+write per call — ~160 µs at M=4096
  INTER=9728. cuBLAS can't replicate this because `torch.scaled_mm`
  requires pre-quantised inputs, breaking the fusion chain.

cos = 0.9979 for all FP8 paths, bit-equivalent to the unfused
reference to 6 decimal places.

### Attention — MXFP8 forward vs torch SDPA (causal prefill)

| model | B | S | H_q / H_kv | mxfp8 µs | mxfp8 TF | sdpa µs | sdpa TF | speedup | cos |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Llama3-8B  | 4 | 8192 | 32 / 8  | 4735 | 464 | 6610  | 333 | 1.40× | 0.9993 |
| Llama3-70B | 4 | 8192 | 64 / 8  | 9391 | **468** | 13091 | 336 | 1.39× | 0.9993 |
| Qwen3-32B  | 4 | 8192 | 64 / 8  | 9397 | **468** | 13077 | 336 | 1.39× | 0.9993 |
| Qwen2-72B  | 4 | 8192 | 64 / 8  | 9388 | **468** | 13101 | 336 | 1.40× | 0.9993 |

**Peak prefill: 468 TF (46.5% MFU) on Llama3-70B / Qwen3-32B / Qwen2-72B
B=4 S=8192, ~1.4× over torch SDPA, cos 0.9993.**

### Attention — paged decode (S_q = 1) vs torch SDPA

| model | B | KV | mxfp8 µs | sdpa µs | speedup | cos |
|---|--:|--:|--:|--:|--:|--:|
| Llama3-8B  | 4 |  4096 | 26.7 |  188 | **7.05×** | 0.9993 |
| Llama3-8B  | 4 | 16384 | 65.6 |  712 | **10.85×** | 0.9992 |
| Llama3-70B | 4 | 16384 | 73.4 | 1407 | **19.17×** | 0.9992 |
| Qwen3-32B  | 4 | 16384 | 73.7 | 1407 | **19.09×** | 0.9993 |
| Qwen2-72B  | 4 | 16384 | 75.4 | 1407 | **18.65×** | 0.9992 |

### Attention — extend (paged) vs FlashInfer BF16

| Shape | FI µs | FSO µs | FI TF | FSO TF | speedup | cos |
|---|--:|--:|--:|--:|--:|--:|
| Llama3-8B decode B=8 kv=8192   |  186 |   68 |   5.8 |  15.7 | **2.72×** | 0.9993 |
| Qwen2-72B decode B=8 kv=4096   |   67 |   47 |  16.0 |  23.0 | **1.44×** | 0.9992 |
| Llama3-8B extend64 B=4 kv=1024 |   39 |   32 | 115.1 | 137.0 | **1.19×** | 0.9993 |
| Llama3-8B prefill B=1 S=8192   | 1875 | 2475 | 293.3 | 222.2 |  0.76× | 0.9994 |

FlashInfer wins on long-prefill at small B (the workstation card favours
FI's larger-tile schedule there). FSO wins decisively on decode and
short-context extend.

## Install

```bash
git clone <repo> fish-scales-ops
cd fish-scales-ops
git submodule update --init --recursive   # 3rdparty/cutlass (v4.4.2)

EDITABLE=1 ./scripts/build.sh             # pip install -e python/
```

The build script probes `CUDA_HOME`, CUTLASS, and Python headers from
standard locations and fails early with a clear message if anything is
missing. Common overrides:

| Var | Purpose | Default |
|---|---|---|
| `ARCH` | `TORCH_CUDA_ARCH_LIST` | `9.0a;12.0a` |
| `CUDA_HOME` | CUDA toolkit root | `nvcc` on PATH → `/usr/local/cuda` |
| `CUTLASS_DIR` | CUTLASS 4.x root | `<repo>/3rdparty/cutlass` |
| `PYTHON_INCLUDE` | dir containing `Python.h` | `sysconfig.get_path("include")` |
| `MAX_JOBS` | ninja parallelism | `$(nproc)` |
| `EDITABLE` | `pip install -e` vs `build_ext --inplace` | `0` |

`--no-build-isolation` inside `build.sh` is deliberate: it forces the
`_C.so` to link against the active venv's torch and avoids the c10 ABI
mismatch you get when pip pulls a fresh torch into a build sandbox.

## Usage

```python
import torch
import fish_scales_ops as fso

sm120 = torch.cuda.get_device_capability(0)[0] >= 12

# ----- GEMM, FP8 1×128 (Hopper + Blackwell) -----
wq, sw = fso.gemm.quantize_128x128_fp8(w_bf16)
xq, sx = fso.gemm.quantize_1x128_fp8(x_bf16, use_ue8m0=sm120)
if sm120:
    sw = fso.gemm.repack_fp8_wgt_scales(sw)
    sx = fso.gemm.repack_fp8_act_scales(sx)
y = fso.gemm.linear_fp8(xq, wq, sx, sw)                  # → bf16 [M, N]

# ----- GEMM, MXFP8 1×32 (sm_100/103 + sm_120a — 80% MFU at M=N=K=16384 on RTX PRO 6000) -----
if mxfp8_ok:  # torch.cuda.get_device_capability(0)[0] in (10, 12)
    wqm, swm = fso.gemm.quantize_1x32_fp8(w_bf16)
    xqm, sxm = fso.gemm.quantize_1x32_fp8(x_bf16)
    ym = fso.gemm.linear_mxfp8(xqm, wqm, sxm, swm)

# ----- Attention -----
o = fso.attention.flash_attn_fwd(q, k, v, causal=True)   # BF16/FP16 → SDPA

# Pre-quantised MXFP8 fast path (sm_120a only):
from fish_scales_ops.attention.backends import sm120_mxfp8 as bk
q_fp8, q_sc = bk.pre_quantize_q(q_bf16)
k_fp8, k_sc = bk.pre_quantize_k(k_bf16)
v_fp8, v_sc = bk.pre_quantize_v(v_bf16)
o = bk.mxfp8_fwd(q_fp8, q_sc, k_fp8, k_sc, v_fp8, v_sc, causal=True)
```

`fso.gemm` and `fso.attention` are independent; there are no top-level
re-exports.

## Status

| Path | Hopper (sm_90a) | Blackwell (sm_120a) | Blackwell DC (sm_100/103) |
|---|---|---|---|
| FP8 1×128 GEMM | ✓ (NVRTC JIT, DeepGEMM-derived WGMMA) | ✓ (CUTLASS BlockScaled) | — |
| MXFP8 1×32 GEMM | — | ✓ (CUTLASS BlockScaled mxf8f6f4) | ✓ (CUTLASS tcgen05 BlockScaled) |
| BF16 FlashAttention | torch SDPA | torch SDPA | torch SDPA |
| **MXFP8 FlashAttention prefill** | — | ✓ (D ∈ {32, 64, 128, 256}) | — |
| **MXFP8 paged-prefill (extend)** | — | ✓ (page_size ∈ {32, 64, 128, 256}) | — |
| **MXFP8 paged decode** | — | ✓ (cross-batch packed CTAs) | — |

Constraints worth knowing up front:

* MXFP8 attention is **sm_120a-only**. The host dispatcher returns
  `cudaErrorNotSupported` on Hopper and Blackwell datacenter.
* B200 / B300 (sm_100 / sm_103): build with `ARCH="10.0f"` — the sm_100f
  *family* target (CUDA ≥ 12.9), one cubin serves both. The MXFP8 scale
  tensor is an **opaque handle in the arch-native layout**: always quantize
  on the same device arch the GEMM runs on (the sm_120 int32-K-major and
  sm_100/103 `Sm1xxBlockScaledConfig` atom layouts are not interchangeable).
* Paged KV cache requires `page_size ≥ 32` (MXFP8 `sf_vec_size=32` is a
  spec-level constraint, not an impl limitation). Match SGLang's
  `--page-size 32` at minimum.
* On sm_120a, FP8 activation quantize must use `use_ue8m0=True`; the
  wrapper repacks scales into the int32-packed UE8M0 layout the CUTLASS
  kernel expects.
* GQA is supported natively (stride-0 K/V broadcast). `H_q / H_kv ≤ 64`
  for the decode kernel; prefill has no such cap.
* The `linear_bf16` fused BF16→FP8 path is known to NaN on CUDA 13.0 +
  sm_120a; production callers should pre-quantise (`linear_fp8`).

## Layout

```
fish-scales-ops/
├── 3rdparty/cutlass/             # submodule, v4.4.2
├── csrc/
│   ├── common/compat/            # vendored TRT-LLM-style compat headers
│   ├── gemm/                     # FP8 1×128 + MXFP8 1×32
│   └── attention/                # sm_120a MXFP8 forward + paged decode
├── python/fish_scales_ops/
│   ├── gemm/                     # fp8.py, mxfp8.py, bf16.py
│   └── attention/
│       ├── flash_attn_func.py    # dispatch (SDPA fallback)
│       └── backends/sm120_mxfp8{,_decode,_paged_prefill}.py
├── tests/{gemm,attention}/
├── bench/{gemm,attention}/
└── scripts/build.sh
```

## Tests + bench

```bash
# GEMM correctness + CUDA-graph compat
PYTHONPATH=python python tests/gemm/unit/test_correctness.py
PYTHONPATH=python python tests/gemm/unit/test_cuda_graph.py

# Attention (46 tests, includes D=256)
PYTHONPATH=python python -m pytest tests/attention/

# GEMM bench — square M=N=K (peak MFU) and Qwen3-4B MLP shapes
PYTHONPATH=python python bench/gemm/python/bench_cubic.py
PYTHONPATH=python python bench/gemm/python/bench_qwen3_4b_mlp.py \
    --run --out runs/<tag>.jsonl

# Attention bench vs SDPA + vs FlashInfer
PYTHONPATH=python python bench/attention/bench_qwen3_llama3.py
PYTHONPATH=python python bench/attention/bench_extend_vs_flashinfer.py
```

## Tuning knobs

`FSO_FORCE_TILE`, `FSO_FORCE_KSPLIT`, `FSO_FORCE_MIN_BLOCKS`,
`FSO_FORCE_SMALLM`, `FSO_FORCE_SCHED_GROUP`, `FSO_DISABLE_OVERRIDES`,
`FSO_DISABLE_STREAMK`, `FSO_JIT_INCLUDE_DIRS`, `FSO_STREAMK_POOL_MB`.

## License

Apache-2.0 (declared in `python/pyproject.toml`). A top-level `LICENSE`
file will land before the first tagged release. Vendored upstream files
under `csrc/gemm/.../jit/deep_gemm/` retain their original Apache-2.0
NVIDIA copyright headers.

## Acknowledgments

`fish-scales-ops` stands on a lot of open-source work. Several portions
of the codebase are direct ports or close adaptations from the projects
below. We are grateful to their authors, and where individual files
preserve upstream copyright headers we have kept them intact.

* **[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)** —
  the sm_90a FP8 1×128 NVRTC-JIT path under
  `csrc/gemm/include/blockscale_gemm/arch/sm90/fp8/jit/deep_gemm/` is
  derived from DeepGEMM's WGMMA kernels (compiler driver, JIT utils,
  TMA utils, scheduler, NVRTC C++/CUTLASS stdlib shim). Upstream
  copyright is preserved per file.
* **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)** —
  the compat shim under `csrc/common/compat/include/tensorrt_llm/` is a
  minimal vendored subset of TRT-LLM's `common/` headers so the JIT-loaded
  kernel sources can keep their original include paths. The block-scaled
  FP8 GEMM runner shape was originally extracted from TRT-LLM's
  `CutlassFp8BlockScaleGemmRunner` interface.
* **[FlashAttention](https://github.com/Dao-AILab/flash-attention)** —
  the online-softmax pattern, KV-tile pipelining, and "FA4-style"
  rescale-skip in the sm_120a MXFP8 attention kernel follow Tri Dao's
  FlashAttention v2/v3 algorithm. The Hopper / Ampere BF16 backends are
  thin CuTe-DSL adaptations of the FA-v2/v3 reference.
* **[SGLang](https://github.com/sgl-project/sglang)** —
  the paged-prefill / paged-decode API mirrors SGLang's extend-phase
  expectations (qo_indptr / kv_page_indptr / kv_page_indices /
  kv_last_page_len), to keep integration into SGLang's attention backend
  layer mechanical. `--page-size 32` is the minimum SGLang launch flag
  for this library.
* **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** —
  the paged-KV indptr convention and the `BatchPrefillWithPagedKVCache` /
  `BatchDecodeWithPagedKVCache` planner shape are flashinfer-compatible.
  The extend bench uses flashinfer as the BF16 baseline.
* **[CUTLASS](https://github.com/NVIDIA/cutlass)** —
  bundled as a submodule at `3rdparty/cutlass`. The sm_120a kernels build
  on `cute`, `GMMA::Layout_K_SW*_Atom`, and `Sm120BlockScaledKernel`.

If you spot upstream code we have not credited correctly, please open
an issue.
