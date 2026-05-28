# fish-scales-ops — frozen reference perf (GEMM)

## Optimisation summary — what got faster (2026-05)

A tuning pass over the sm_120 MXFP8/BSFP8 stack. Everything below is
bit-exact (or cos-unchanged) vs its pre-change baseline; numbers are the
same CUDA-graph harness used in the detail tables.

| # | Area | Change | Speedup | Where |
|---|---|---|---|---|
| 1 | **MXFP8 GEMM dispatch** | Fixed 5 cascade tile-selection bugs (E28–E32): `short_k` M-cap, TileM=160 M-gate, (128,64,2) wave-fill rescue, M∈(128,512] tile choice, Stream-K 90% saturation gate | **+15% … +104%** on 8 previously-losing cells (cubic-1024/1536/2048, Qwen MLP M=256/1024); cubic MXFP8 750 TF peak (8192³, PRO 6000 no-boost) | `arch/sm120/{fp8,mxfp8}/dispatch.cuh`, `common/streamk.cuh` |
| 2 | **MXFP8 1×32 quantize** | Fused quantize+scale-pack single kernel (was 2 kernels + FP32 scale roundtrip): uint64 LDG.64, 8-lane sub-warp reduce, IEEE-754 E8M0 bit-manip | **2.4×** (142→58 µs @ M=4096 K=9728); **matches/beats** Inductor-fused `to_mxfp+to_blocked` at every shape | `ops/quant_kernels.cu` |
| 3 | **BSFP8 1×128 quantize** | Same uint64 LDG.64 fused rewrite, sm_120 packed + sm_90 FP32 output | sm_120 **+13–15%**, **H200 +15–23%** on BSFP8 MLP-fwd prefill | `ops/quant_kernels.cu`, `ops/fp8.cu` |
| 4 | **SwiGLU MLP forward** | `silu(gate)*up` fused into the down-GEMM activation quantize — no BF16 `h` intermediate written | MXFP8 MLP-fwd **1.7–6.2× vs raw cuBLAS `scaled_mm`** at every M (1→4096); **2.1–4.3×** vs the `torch.compile`-fused cuBLAS at decode (M≤128), within ±8% at prefill | `ops/quant_kernels.cu` (`silu_chunk_mul_quantize_1x32`) |

Scope: GEMM dispatch + GEMM/MLP-forward quantize. The attention
forward kernels are unchanged (see `docs/design/attn_micro_opts.md`
for the ceiling analysis).

## Methodology — no-boost clock lock (reproducible defaults)

Every number in this document is measured with the GPU **clock locked to a
fixed no-boost frequency** (`nvidia-smi -lgc`), so results are reproducible
run-to-run and across units — boost clocks vary with thermal / power / silicon
bin and are *not* reproducible. Reset afterwards with `nvidia-smi -rgc`.

| GPU | lock command | achieved SM clock |
|---|---|---|
| RTX PRO 6000 Blackwell Server Edition | `nvidia-smi -lgc 2430,2430` | **2400 MHz** |
| Blackwell GPU (170 SMs) | `nvidia-smi -lgc 2407,2407` | **2400 MHz** |
| NVIDIA H200 (sm_90) | `nvidia-smi -lgc 1980,1980` | **1830 MHz** |

`-lgc V,V` snaps to the nearest supported P-state; the achieved clock is what
the tables reflect. **Toolkit is identical on all three: CUDA 13.0 / torch
2.11+cu130.** The `sMM` (cuBLAS `scaled_mm` MXFP8) comparison therefore uses
the **cu13 cuBLAS** on *both* Blackwell parts — cu13 fixes the large-square
MXFP8 cuBLAS regression that cu12.8 had (16384³: cu12.8 ≈ 527 TF → cu13 ≈ 706
TF), so the fso-vs-cuBLAS deltas below are the honest cu13 numbers, not the
inflated cu12.8 ones.

The two Blackwell parts are locked to the **same 2400 MHz**, so PRO 6000 vs
Blackwell GPU(170 SM) is a clean SM-count + per-SM-FP8-rate comparison (188 vs 170 SMs; the
PRO 6000's server-class FP8 tensor cores are ~38 % faster per SM-clock).

## Environment

Software stack (identical across all three machines):

| component | version |
|---|---|
| PyTorch | `2.11.0+cu130` |
| CUDA runtime (torch) | 13.0 |
| Build toolkit (nvcc) | 13.0 (sm_120a parts) · 13.2 (H200) |
| CUTLASS | 4.4.2 (`3rdparty/cutlass` submodule) |
| Python | 3.12.3 |
| OS | Linux x86_64 |

Hardware + per-GPU clock lock (see Methodology above for the `-lgc` commands):

| GPU | arch | SMs | driver | no-boost SM clock |
|---|---|---|---|---|
| RTX PRO 6000 Blackwell Server Edition | sm_120a | 188 | 580.126.09 | 2400 MHz |
| Blackwell GPU  | sm_120a | 170 | 595.58.03 | 2400 MHz |
| NVIDIA H200 | sm_90 | 132 | 595.58.03 | 1830 MHz |

Build: `TORCH_CUDA_ARCH_LIST="12.0a" python setup.py build_ext --inplace`
(sm_120a parts) / `"9.0"` (H200). The sm_120 `.so` links cu13 NVRTC, so set
`LD_LIBRARY_PATH=$CUDA_HOME/lib64` (cu13 toolkit) if the wheel's NVRTC isn't
on the loader path.

All `µs` numbers below are **CUDA-Graph capture-and-replay** timings — each `linear_*` / `scaled_mm` call is captured into a `torch.cuda.CUDAGraph` and replayed in a tight loop, then the median over 50 iters × 3 reps is reported. Single-call eager timing was removed: it conflates kernel time with PyTorch op dispatch + `cudaLaunchKernelEx` overhead, both of which a graph-captured production decode loop pays exactly **once** (at capture), so an eager baseline misleads kernel-side tuning toward host-bound shapes that are not host-bound in production. `TF` (TFLOPS = `2·M·N·K / (µs · 1e6)`) is computed from these graph numbers. BF16 baseline is `torch.nn.functional.linear`. Pre-quantize + pre-pack happen outside the timing loop. `cos` is cosine similarity vs BF16.

Columns:
- `BSFP8` — fish-scales-ops `linear_fp8` (1×128 act / 128×128 wgt, UE8M0 scales on sm_120).
- `MXFP8` — fish-scales-ops `linear_mxfp8` (1×32, UE8M0). sm_120 only.
- `sMM`   — `torch.nn.functional.scaled_mm` with the same 1×32 UE8M0 block-scaled FP8 inputs (cuBLAS / cuBLASLt path); sm_120+ only. Apples-to-apples kernel comparison vs `MXFP8`.

Shapes are Qwen3-4B (hidden=2560, 32 Q-heads + 8 KV-heads, head_dim=128, intermediate=9728):
- `wqkv`     N=6144  K=2560 (fused Q+K+V projection)
- `wo`       N=2560  K=4096 (attention output projection)
- `gate_up`  N=19456 K=2560 (fused gate+up)
- `gate`     N=9728  K=2560 (single gate or up)
- `down`     N=2560  K=9728
Plus a `cubic` (M=N=K) sweep ∈ {1024, 1536, 2048, 2560, 3072, 4096, 6144, 8192, 12288, 16384} for the peak-MFU regime.

Reproduce: `PYTHONPATH=python python bench/gemm/python/bench_qwen3_4b_mlp.py --run --out <jsonl>`.

## 188 SMs · 2400 MHz no-boost · cu13 — NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120)

### `wqkv` (N=6144, K=2560)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | MXFP8 µs | MXFP8 TF | sMM µs | sMM TF | BSFP8 cos | MXFP8 cos | sMM cos |
|  ---: |    ---: |     ---: |     ---: |     ---: |     ---: |   ---: |   ---: |      ---: |      ---: |    ---: |
|     1 |   14.43 |    14.40 |        2 |    14.40 |        2 |    24.60 |        1 |    0.9947 |    0.9993 |    0.9993 |
|     2 |   14.44 |    12.40 |        5 |    14.38 |        4 |    23.37 |        3 |    0.9947 |    0.9993 |    0.9993 |
|     4 |   15.12 |    14.31 |        9 |    14.39 |        9 |    23.69 |        5 |    0.9943 |    0.9993 |    0.9993 |
|     8 |   16.09 |    14.36 |       18 |    14.36 |       18 |    23.71 |       11 |    0.9945 |    0.9993 |    0.9993 |
|    16 |   18.50 |    12.40 |       41 |    14.39 |       35 |    22.63 |       22 |    0.9946 |    0.9993 |    0.9993 |
|    32 |   20.22 |    12.40 |       81 |    14.32 |       70 |    22.61 |       45 |    0.9949 |    0.9993 |    0.9993 |
|    64 |   16.43 |    14.39 |      140 |    14.37 |      140 |    22.59 |       89 |    0.9944 |    0.9993 |    0.9993 |
|   128 |   22.72 |    18.46 |      218 |    18.49 |      218 |    22.68 |      178 |    0.9947 |    0.9993 |    0.9993 |
|   512 |   67.71 |    43.12 |      374 |    43.10 |      374 |    47.11 |      342 |    0.9946 |    0.9993 |    0.9993 |
|  1024 |  122.97 |    73.78 |      437 |    58.66 |      549 |    70.07 |      460 |    0.9944 |    0.9993 |    0.9993 |
|  2048 |  213.14 |   134.89 |      478 |   120.62 |      534 |   113.04 |      570 |    0.9945 |    0.9993 |    0.9993 |
|  4096 |  383.73 |   255.84 |      504 |   202.65 |      636 |   204.38 |      630 |    0.9949 |    0.9993 |    0.9993 |

### `wo` (N=2560, K=4096)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | MXFP8 µs | MXFP8 TF | sMM µs | sMM TF | BSFP8 cos | MXFP8 cos | sMM cos |
|  ---: |    ---: |     ---: |     ---: |     ---: |     ---: |   ---: |   ---: |      ---: |      ---: |    ---: |
|     1 |    8.27 |    14.40 |        1 |    12.31 |        2 |    34.03 |        1 |    0.9878 |    0.9992 |    0.9992 |
|     2 |   10.32 |    14.37 |        3 |    12.36 |        3 |    33.82 |        1 |    0.9891 |    0.9993 |    0.9993 |
|     4 |   10.40 |    14.38 |        6 |    12.34 |        7 |    34.03 |        2 |    0.9882 |    0.9993 |    0.9993 |
|     8 |   10.46 |    14.05 |       12 |    12.35 |       14 |    34.29 |        5 |    0.9879 |    0.9993 |    0.9993 |
|    16 |   12.36 |    12.48 |       27 |    12.31 |       27 |    34.29 |       10 |    0.9887 |    0.9993 |    0.9993 |
|    32 |   14.36 |    12.35 |       54 |    14.27 |       47 |    33.56 |       20 |    0.9876 |    0.9993 |    0.9993 |
|    64 |   12.34 |    14.39 |       93 |    14.40 |       93 |    33.80 |       40 |    0.9869 |    0.9993 |    0.9993 |
|   128 |   16.42 |    14.41 |      186 |    14.39 |      186 |    34.53 |       78 |    0.9894 |    0.9993 |    0.9993 |
|   512 |   39.08 |    26.21 |      410 |    26.76 |      401 |    34.83 |      308 |    0.9874 |    0.9993 |    0.9993 |
|  1024 |   69.79 |    49.21 |      436 |    37.29 |      576 |    37.38 |      574 |    0.9887 |    0.9993 |    0.9993 |
|  2048 |  119.86 |    95.96 |      448 |    73.92 |      581 |    72.53 |      592 |    0.9885 |    0.9993 |    0.9993 |
|  4096 |  240.74 |   160.21 |      536 |   147.46 |      583 |   142.52 |      603 |    0.9879 |    0.9993 |    0.9993 |

### `gate_up` (N=19456, K=2560)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | MXFP8 µs | MXFP8 TF | sMM µs | sMM TF | BSFP8 cos | MXFP8 cos | sMM cos |
|  ---: |    ---: |     ---: |     ---: |     ---: |     ---: |   ---: |   ---: |      ---: |      ---: |    ---: |
|     1 |   29.82 |    24.59 |        4 |    16.44 |        6 |    61.47 |        2 |    0.9947 |    0.9993 |    0.9993 |
|     2 |   28.06 |    22.55 |        9 |    16.45 |       12 |    61.49 |        3 |    0.9942 |    0.9993 |    0.9993 |
|     4 |   30.74 |    22.57 |       18 |    16.45 |       24 |    57.41 |        7 |    0.9948 |    0.9993 |    0.9993 |
|     8 |   33.21 |    20.55 |       39 |    14.41 |       55 |    55.35 |       14 |    0.9947 |    0.9993 |    0.9993 |
|    16 |   39.80 |    18.46 |       86 |    14.42 |      111 |    55.32 |       29 |    0.9947 |    0.9993 |    0.9993 |
|    32 |   24.87 |    14.65 |      218 |    16.43 |      194 |    51.24 |       62 |    0.9947 |    0.9993 |    0.9993 |
|    64 |   28.89 |    18.49 |      345 |    20.51 |      311 |    27.30 |      234 |    0.9943 |    0.9993 |    0.9993 |
|   128 |   47.22 |    26.65 |      478 |    26.85 |      475 |    26.83 |      475 |    0.9948 |    0.9993 |    0.9993 |
|   512 |  155.00 |   114.40 |      446 |   100.50 |      507 |    94.40 |      540 |    0.9947 |    0.9993 |    0.9993 |
|  1024 |  301.83 |   168.38 |      606 |   170.96 |      597 |   160.49 |      636 |    0.9946 |    0.9993 |    0.9993 |
|  2048 |  552.91 |   298.53 |      683 |   320.61 |      636 |   299.63 |      681 |    0.9945 |    0.9993 |    0.9993 |
|  4096 |  987.15 |   577.64 |      706 |   577.90 |      706 |   575.18 |      709 |    0.9947 |    0.9993 |    0.9993 |

### `gate` (N=9728, K=2560)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | MXFP8 µs | MXFP8 TF | sMM µs | sMM TF | BSFP8 cos | MXFP8 cos | sMM cos |
|  ---: |    ---: |     ---: |     ---: |     ---: |     ---: |   ---: |   ---: |      ---: |      ---: |    ---: |
|     1 |   16.46 |    16.46 |        3 |    14.38 |        3 |    34.90 |        1 |    0.9950 |    0.9993 |    0.9993 |
|     2 |   16.54 |    14.38 |        7 |    14.39 |        7 |    34.91 |        3 |    0.9949 |    0.9993 |    0.9993 |
|     4 |   16.54 |    14.92 |       13 |    14.39 |       14 |    34.98 |        6 |    0.9945 |    0.9993 |    0.9993 |
|     8 |   17.61 |    16.27 |       24 |    14.36 |       28 |    32.83 |       12 |    0.9944 |    0.9993 |    0.9993 |
|    16 |   20.55 |    14.37 |       55 |    14.36 |       55 |    32.74 |       24 |    0.9944 |    0.9993 |    0.9993 |
|    32 |   16.63 |    12.56 |      127 |    14.41 |      111 |    29.07 |       55 |    0.9945 |    0.9993 |    0.9993 |
|    64 |   20.57 |    15.61 |      204 |    14.41 |      221 |    23.83 |      134 |    0.9948 |    0.9993 |    0.9993 |
|   128 |   28.75 |    18.48 |      345 |    18.54 |      344 |    22.71 |      281 |    0.9945 |    0.9993 |    0.9993 |
|   512 |   78.02 |    64.28 |      397 |    50.07 |      509 |    48.34 |      528 |    0.9946 |    0.9993 |    0.9993 |
|  1024 |  151.88 |   108.17 |      472 |   100.13 |      509 |    94.20 |      541 |    0.9945 |    0.9993 |    0.9993 |
|  2048 |  299.00 |   195.39 |      522 |   178.47 |      572 |   159.88 |      638 |    0.9947 |    0.9993 |    0.9993 |
|  4096 |  549.13 |   365.70 |      558 |   319.17 |      639 |   299.09 |      682 |    0.9947 |    0.9993 |    0.9993 |

### `down` (N=2560, K=9728)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | MXFP8 µs | MXFP8 TF | sMM µs | sMM TF | BSFP8 cos | MXFP8 cos | sMM cos |
|  ---: |    ---: |     ---: |     ---: |     ---: |     ---: |   ---: |   ---: |      ---: |      ---: |    ---: |
|     1 |   18.55 |    16.02 |        3 |    15.53 |        3 |    73.81 |        1 |    0.9952 |    0.9993 |    0.9993 |
|     2 |   18.92 |    16.45 |        6 |    15.47 |        6 |    73.76 |        1 |    0.9952 |    0.9993 |    0.9993 |
|     4 |   20.31 |    16.08 |       12 |    15.47 |       13 |    73.76 |        3 |    0.9958 |    0.9993 |    0.9993 |
|     8 |   20.52 |    16.49 |       24 |    15.46 |       26 |    73.74 |        5 |    0.9954 |    0.9993 |    0.9993 |
|    16 |   20.92 |    16.37 |       49 |    15.47 |       52 |    73.75 |       11 |    0.9955 |    0.9993 |    0.9993 |
|    32 |   20.59 |    16.38 |       97 |    14.41 |      111 |    73.78 |       22 |    0.9957 |    0.9993 |    0.9993 |
|    64 |   18.51 |    18.47 |      173 |    17.48 |      182 |    73.74 |       43 |    0.9954 |    0.9993 |    0.9993 |
|   128 |   28.78 |    21.99 |      290 |    23.08 |      276 |    73.75 |       86 |    0.9955 |    0.9993 |    0.9993 |
|   512 |   90.21 |    68.04 |      375 |    73.99 |      345 |    74.17 |      344 |    0.9957 |    0.9993 |    0.9993 |
|  1024 |  154.76 |   112.28 |      454 |   107.07 |      476 |    85.98 |      593 |    0.9954 |    0.9993 |    0.9993 |
|  2048 |  277.04 |   214.73 |      475 |   172.04 |      593 |   165.61 |      616 |    0.9957 |    0.9993 |    0.9993 |
|  4096 |  554.59 |   354.03 |      576 |   342.62 |      595 |   332.94 |      613 |    0.9957 |    0.9993 |    0.9993 |

### cubic (M=N=K) — peak-MFU sweep

|  M=N=K | BF16 µs | BSFP8 µs | BSFP8 TF | MXFP8 µs | MXFP8 TF | sMM µs | sMM TF | BSFP8 cos | MXFP8 cos | sMM cos |
|   ---: |    ---: |     ---: |     ---: |     ---: |     ---: |   ---: |   ---: |      ---: |      ---: |    ---: |
|   1024 |   12.38 |    14.38 |      149 |    10.31 |      208 |    12.34 |      174 |    0.9678 |    0.9993 |    0.9993 |
|   1536 |   26.83 |    30.75 |      236 |    18.47 |      392 |    18.47 |      392 |    0.9857 |    0.9993 |    0.9993 |
|   2048 |   61.75 |    61.24 |      281 |    40.95 |      420 |    38.92 |      441 |    0.9937 |    0.9993 |    0.9993 |
|   2560 |  122.99 |    79.84 |      420 |    59.41 |      565 |    71.90 |      467 |    0.9943 |    0.9993 |    0.9993 |
|   3072 |  176.82 |   114.64 |      506 |   104.72 |      554 |   108.77 |      533 |    0.9945 |    0.9993 |    0.9993 |
|   4096 |  360.69 |   241.46 |      569 |   220.33 |      624 |   209.75 |      655 |    0.9884 |    0.9993 |    0.9993 |
|   6144 | 1235.53 |   788.50 |      588 |   622.12 |      746 |   658.05 |      705 |    0.9674 |    0.9993 |    0.9993 |
|   8192 | 2660.51 |  1829.38 |      601 |  1466.79 |      750 |  1479.45 |      743 |    0.9941 |    0.9993 |    0.9993 |
|  12288 | 9088.28 |  4987.65 |      744 |  5110.93 |      726 |  5251.77 |      707 |    0.9959 |    0.9993 |    0.9993 |
|  16384 | 21507.78 | 11492.56 |      765 | 11847.67 |      742 | 12460.43 |      706 |    0.9950 |    0.9993 |    0.9993 |

## 170 SMs · 2400 MHz no-boost · cu13 — NVIDIA GeForce Blackwell GPU  (sm_120)

### `wqkv` (N=6144, K=2560)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | MXFP8 µs | MXFP8 TF | sMM µs | sMM TF | BSFP8 cos | MXFP8 cos | sMM cos |
|  ---: |    ---: |     ---: |     ---: |     ---: |     ---: |   ---: |   ---: |      ---: |      ---: |    ---: |
|     1 |   16.48 |    14.42 |        2 |    12.39 |        3 |    24.64 |        1 |    0.9951 |    0.9993 |    0.9993 |
|     2 |   16.00 |    12.38 |        5 |    12.36 |        5 |    22.81 |        3 |    0.9948 |    0.9993 |    0.9993 |
|     4 |   16.46 |    12.53 |       10 |    12.39 |       10 |    23.75 |        5 |    0.9945 |    0.9993 |    0.9993 |
|     8 |   16.51 |    12.87 |       20 |    12.37 |       20 |    23.29 |       11 |    0.9947 |    0.9993 |    0.9993 |
|    16 |   18.55 |    12.39 |       41 |    12.38 |       41 |    22.96 |       22 |    0.9943 |    0.9993 |    0.9993 |
|    32 |   22.04 |    12.39 |       81 |    14.36 |       70 |    22.63 |       44 |    0.9942 |    0.9993 |    0.9993 |
|    64 |   20.79 |    14.39 |      140 |    12.59 |      160 |    22.62 |       89 |    0.9946 |    0.9993 |    0.9993 |
|   128 |   39.04 |    16.51 |      244 |    18.51 |      218 |    22.66 |      178 |    0.9943 |    0.9993 |    0.9993 |
|   512 |  113.44 |    45.13 |      357 |    41.18 |      391 |    46.76 |      344 |    0.9940 |    0.9993 |    0.9993 |
|  1024 |  186.58 |    76.24 |      422 |    57.77 |      558 |    67.31 |      479 |    0.9946 |    0.9993 |    0.9993 |
|  2048 |  361.54 |   135.60 |      475 |   111.44 |      578 |   110.53 |      583 |    0.9948 |    0.9993 |    0.9993 |
|  4096 |  693.71 |   261.23 |      493 |   222.07 |      580 |   216.76 |      594 |    0.9948 |    0.9993 |    0.9993 |

### `wo` (N=2560, K=4096)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | MXFP8 µs | MXFP8 TF | sMM µs | sMM TF | BSFP8 cos | MXFP8 cos | sMM cos |
|  ---: |    ---: |     ---: |     ---: |     ---: |     ---: |   ---: |   ---: |      ---: |      ---: |    ---: |
|     1 |   12.41 |    14.43 |        1 |    12.38 |        2 |    33.07 |        1 |    0.9885 |    0.9993 |    0.9993 |
|     2 |   20.54 |    14.45 |        3 |    12.35 |        3 |    33.12 |        1 |    0.9884 |    0.9993 |    0.9993 |
|     4 |   20.55 |    15.50 |        5 |    12.45 |        7 |    33.31 |        3 |    0.9881 |    0.9993 |    0.9993 |
|     8 |   20.54 |    12.38 |       14 |    12.34 |       14 |    33.40 |        5 |    0.9880 |    0.9993 |    0.9993 |
|    16 |   20.59 |    14.40 |       23 |    12.36 |       27 |    33.48 |       10 |    0.9897 |    0.9993 |    0.9993 |
|    32 |   12.77 |    12.37 |       54 |    12.54 |       54 |    32.92 |       20 |    0.9888 |    0.9993 |    0.9993 |
|    64 |   14.42 |    14.40 |       93 |    14.42 |       93 |    33.18 |       40 |    0.9876 |    0.9993 |    0.9993 |
|   128 |   20.83 |    14.42 |      186 |    14.68 |      183 |    33.52 |       80 |    0.9890 |    0.9993 |    0.9993 |
|   512 |   59.71 |    26.70 |      402 |    26.80 |      401 |    34.89 |      308 |    0.9877 |    0.9993 |    0.9993 |
|  1024 |  114.98 |    49.30 |      436 |    37.64 |      571 |    37.39 |      574 |    0.9888 |    0.9993 |    0.9993 |
|  2048 |  229.31 |    94.83 |      453 |    71.85 |      598 |    69.96 |      614 |    0.9888 |    0.9993 |    0.9993 |
|  4096 |  454.34 |   183.92 |      467 |   141.79 |      606 |   138.92 |      618 |    0.9897 |    0.9993 |    0.9993 |

### `gate_up` (N=19456, K=2560)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | MXFP8 µs | MXFP8 TF | sMM µs | sMM TF | BSFP8 cos | MXFP8 cos | sMM cos |
|  ---: |    ---: |     ---: |     ---: |     ---: |     ---: |   ---: |   ---: |      ---: |      ---: |    ---: |
|     1 |   53.35 |    24.63 |        4 |    18.51 |        5 |    59.51 |        2 |    0.9944 |    0.9993 |    0.9993 |
|     2 |   34.84 |    22.59 |        9 |    16.47 |       12 |    59.51 |        3 |    0.9943 |    0.9993 |    0.9993 |
|     4 |   35.06 |    22.69 |       18 |    16.51 |       24 |    56.32 |        7 |    0.9945 |    0.9993 |    0.9993 |
|     8 |   37.92 |    20.60 |       39 |    16.34 |       49 |    54.84 |       15 |    0.9945 |    0.9993 |    0.9993 |
|    16 |   44.75 |    18.53 |       86 |    14.41 |      111 |    55.45 |       29 |    0.9948 |    0.9993 |    0.9993 |
|    32 |   39.12 |    16.44 |      194 |    16.44 |      194 |    49.26 |       65 |    0.9947 |    0.9993 |    0.9993 |
|    64 |   40.98 |    18.72 |      341 |    20.56 |      310 |    26.77 |      238 |    0.9947 |    0.9993 |    0.9993 |
|   128 |   77.91 |    26.69 |      478 |    26.91 |      474 |    27.62 |      462 |    0.9948 |    0.9993 |    0.9993 |
|   512 |  292.67 |   109.71 |      465 |    94.56 |      539 |    89.08 |      573 |    0.9946 |    0.9993 |    0.9993 |
|  1024 |  579.78 |   190.13 |      537 |   182.80 |      558 |   173.49 |      588 |    0.9945 |    0.9993 |    0.9993 |
|  2048 | 1132.14 |   321.20 |      635 |   329.83 |      619 |   324.65 |      628 |    0.9946 |    0.9993 |    0.9993 |
|  4096 | 2114.39 |   627.19 |      651 |   638.29 |      639 |   655.62 |      622 |    0.9948 |    0.9993 |    0.9993 |

### `gate` (N=9728, K=2560)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | MXFP8 µs | MXFP8 TF | sMM µs | sMM TF | BSFP8 cos | MXFP8 cos | sMM cos |
|  ---: |    ---: |     ---: |     ---: |     ---: |     ---: |   ---: |   ---: |      ---: |      ---: |    ---: |
|     1 |   22.60 |    16.46 |        3 |    14.41 |        3 |    32.88 |        2 |    0.9951 |    0.9993 |    0.9993 |
|     2 |   16.75 |    14.43 |        7 |    14.39 |        7 |    32.88 |        3 |    0.9944 |    0.9993 |    0.9993 |
|     4 |   18.54 |    14.85 |       13 |    14.41 |       14 |    32.88 |        6 |    0.9945 |    0.9993 |    0.9993 |
|     8 |   18.52 |    16.03 |       25 |    14.42 |       28 |    30.84 |       13 |    0.9947 |    0.9993 |    0.9993 |
|    16 |   20.60 |    14.44 |       55 |    12.71 |       63 |    30.79 |       26 |    0.9947 |    0.9993 |    0.9993 |
|    32 |   24.67 |    14.37 |      111 |    14.45 |      110 |    28.79 |       55 |    0.9948 |    0.9993 |    0.9993 |
|    64 |   24.67 |    14.44 |      221 |    14.43 |      221 |    22.62 |      141 |    0.9944 |    0.9993 |    0.9993 |
|   128 |   39.06 |    18.51 |      344 |    18.98 |      336 |    22.79 |      280 |    0.9943 |    0.9993 |    0.9993 |
|   512 |  147.30 |    62.26 |      410 |    49.16 |      519 |    47.46 |      537 |    0.9948 |    0.9993 |    0.9993 |
|  1024 |  290.96 |   112.34 |      454 |    94.75 |      538 |    88.80 |      574 |    0.9945 |    0.9993 |    0.9993 |
|  2048 |  550.15 |   214.68 |      475 |   164.71 |      619 |   174.34 |      585 |    0.9946 |    0.9993 |    0.9993 |
|  4096 | 1130.49 |   407.83 |      500 |   328.22 |      622 |   319.30 |      639 |    0.9946 |    0.9993 |    0.9993 |

### `down` (N=2560, K=9728)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | MXFP8 µs | MXFP8 TF | sMM µs | sMM TF | BSFP8 cos | MXFP8 cos | sMM cos |
|  ---: |    ---: |     ---: |     ---: |     ---: |     ---: |   ---: |   ---: |      ---: |      ---: |    ---: |
|     1 |   18.50 |    16.24 |        3 |    14.92 |        3 |    73.82 |        1 |    0.9955 |    0.9993 |    0.9993 |
|     2 |   18.53 |    16.32 |        6 |    15.28 |        7 |    73.81 |        1 |    0.9955 |    0.9993 |    0.9993 |
|     4 |   20.54 |    16.44 |       12 |    14.52 |       14 |    73.84 |        3 |    0.9957 |    0.9993 |    0.9993 |
|     8 |   43.15 |    16.48 |       24 |    14.52 |       27 |    74.02 |        5 |    0.9959 |    0.9993 |    0.9993 |
|    16 |   45.22 |    15.63 |       51 |    14.84 |       54 |    73.84 |       11 |    0.9956 |    0.9993 |    0.9993 |
|    32 |   22.64 |    15.91 |      100 |    14.44 |      110 |    73.82 |       22 |    0.9956 |    0.9993 |    0.9993 |
|    64 |   23.26 |    18.49 |      172 |    17.06 |      187 |    73.82 |       43 |    0.9952 |    0.9993 |    0.9993 |
|   128 |   41.07 |    22.02 |      290 |    22.67 |      281 |    73.84 |       86 |    0.9958 |    0.9993 |    0.9993 |
|   512 |  136.25 |    64.16 |      397 |    69.74 |      366 |    73.86 |      345 |    0.9957 |    0.9993 |    0.9993 |
|  1024 |  270.50 |   108.23 |      471 |    84.06 |      607 |    86.41 |      590 |    0.9953 |    0.9993 |    0.9993 |
|  2048 |  530.77 |   214.84 |      475 |   162.02 |      630 |   163.65 |      623 |    0.9954 |    0.9993 |    0.9993 |
|  4096 | 1071.06 |   422.76 |      483 |   327.63 |      623 |   322.50 |      633 |    0.9957 |    0.9993 |    0.9993 |

### cubic (M=N=K) — peak-MFU sweep

|  M=N=K | BF16 µs | BSFP8 µs | BSFP8 TF | MXFP8 µs | MXFP8 TF | sMM µs | sMM TF | BSFP8 cos | MXFP8 cos | sMM cos |
|   ---: |    ---: |     ---: |     ---: |     ---: |     ---: |   ---: |   ---: |      ---: |      ---: |    ---: |
|   1024 |   18.55 |    14.41 |      149 |    10.34 |      208 |    12.39 |      173 |    0.9712 |    0.9993 |    0.9993 |
|   1536 |   47.22 |    33.15 |      219 |    18.52 |      391 |    18.53 |      391 |    0.9870 |    0.9993 |    0.9993 |
|   2048 |  118.92 |    65.46 |      262 |    40.98 |      419 |    38.99 |      441 |    0.9952 |    0.9993 |    0.9993 |
|   2560 |  186.66 |    76.58 |      438 |    57.56 |      583 |    67.66 |      496 |    0.9945 |    0.9993 |    0.9993 |
|   3072 |  307.78 |   125.41 |      462 |    98.74 |      587 |   106.58 |      544 |    0.9943 |    0.9993 |    0.9993 |
|   4096 |  747.41 |   272.32 |      505 |   208.63 |      659 |   241.71 |      569 |    0.9897 |    0.9993 |    0.9993 |
|   6144 | 2380.68 |   896.72 |      517 |   756.76 |      613 |   750.76 |      618 |    0.9672 |    0.9993 |    0.9993 |
|   8192 | 5505.54 |  2140.87 |      514 |  1712.68 |      642 |  1813.96 |      606 |    0.9942 |    0.9993 |    0.9993 |
|  12288 | 18248.84 |  5546.84 |      669 |  5666.60 |      655 |  6021.08 |      616 |    0.9960 |    0.9993 |    0.9993 |
|  16384 | 43009.94 | 13087.37 |      672 | 13436.04 |      655 | 14393.90 |      611 |    0.9949 |    0.9993 |    0.9993 |

## sm_90 · 1830 MHz no-boost · cu13 — NVIDIA H200 (sm_90)

### `wqkv` (N=6144, K=2560)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | BSFP8 cos |
|  ---: |    ---: |     ---: |     ---: |      ---: |
|     1 |    9.91 |     7.63 |        4 |    0.9993 |
|     2 |    9.41 |     7.38 |        9 |    0.9993 |
|     4 |    9.13 |     7.61 |       17 |    0.9993 |
|     8 |    7.90 |     7.09 |       35 |    0.9993 |
|    16 |    8.23 |     6.90 |       73 |    0.9993 |
|    32 |    8.56 |     9.26 |      109 |    0.9993 |
|    64 |    8.85 |     6.31 |      319 |    0.9993 |
|   128 |    8.72 |     7.54 |      534 |    0.9993 |
|   512 |   23.23 |    18.41 |      875 |    0.9993 |
|  1024 |   44.06 |    30.44 |     1058 |    0.9993 |
|  2048 |   84.55 |    58.32 |     1105 |    0.9993 |
|  4096 |  163.08 |   113.33 |     1137 |    0.9993 |

### `wo` (N=2560, K=4096)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | BSFP8 cos |
|  ---: |    ---: |     ---: |     ---: |      ---: |
|     1 |    9.51 |     8.09 |        3 |    0.9993 |
|     2 |    9.10 |     8.09 |        5 |    0.9993 |
|     4 |    8.67 |     8.08 |       10 |    0.9993 |
|     8 |    7.90 |     7.70 |       22 |    0.9993 |
|    16 |    8.19 |     7.64 |       44 |    0.9993 |
|    32 |    8.64 |     9.88 |       68 |    0.9993 |
|    64 |    9.16 |     7.27 |      185 |    0.9993 |
|   128 |   10.16 |     8.53 |      315 |    0.9993 |
|   512 |   18.02 |    12.68 |      847 |    0.9993 |
|  1024 |   30.61 |    23.14 |      928 |    0.9993 |
|  2048 |   54.99 |    42.01 |     1022 |    0.9993 |
|  4096 |  107.88 |    71.66 |     1199 |    0.9993 |

### `gate_up` (N=19456, K=2560)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | BSFP8 cos |
|  ---: |    ---: |     ---: |     ---: |      ---: |
|     1 |   30.17 |    17.70 |        6 |    0.9993 |
|     2 |   30.41 |    17.70 |       11 |    0.9993 |
|     4 |   30.71 |    18.00 |       22 |    0.9993 |
|     8 |   30.80 |    17.92 |       44 |    0.9993 |
|    16 |   31.20 |    17.90 |       89 |    0.9993 |
|    32 |   31.64 |    18.79 |      170 |    0.9993 |
|    64 |   32.85 |    16.92 |      377 |    0.9993 |
|   128 |   30.83 |    19.92 |      640 |    0.9993 |
|   512 |   72.02 |    53.28 |      957 |    0.9993 |
|  1024 |  139.71 |   101.74 |     1003 |    0.9993 |
|  2048 |  272.49 |   178.72 |     1142 |    0.9993 |
|  4096 |  519.08 |   350.40 |     1164 |    0.9993 |

### `gate` (N=9728, K=2560)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | BSFP8 cos |
|  ---: |    ---: |     ---: |     ---: |      ---: |
|     1 |   13.40 |     8.72 |        6 |    0.9993 |
|     2 |   13.36 |     8.42 |       12 |    0.9993 |
|     4 |   13.36 |     8.80 |       23 |    0.9993 |
|     8 |   13.52 |     8.17 |       49 |    0.9993 |
|    16 |   13.54 |     7.76 |      103 |    0.9993 |
|    32 |   13.83 |    10.66 |      149 |    0.9993 |
|    64 |   13.99 |     7.51 |      425 |    0.9993 |
|   128 |   14.58 |     9.31 |      685 |    0.9993 |
|   512 |   36.88 |    28.94 |      881 |    0.9993 |
|  1024 |   70.37 |    52.12 |      979 |    0.9993 |
|  2048 |  138.31 |   101.11 |     1009 |    0.9993 |
|  4096 |  270.61 |   179.33 |     1138 |    0.9993 |

### `down` (N=2560, K=9728)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | BSFP8 cos |
|  ---: |    ---: |     ---: |     ---: |      ---: |
|     1 |   15.73 |    14.58 |        3 |    0.9994 |
|     2 |   15.63 |    14.76 |        7 |    0.9993 |
|     4 |   15.78 |    14.57 |       14 |    0.9993 |
|     8 |   15.93 |    14.20 |       28 |    0.9993 |
|    16 |   16.29 |    14.30 |       56 |    0.9993 |
|    32 |   17.08 |    19.77 |       81 |    0.9993 |
|    64 |   17.95 |    13.74 |      232 |    0.9993 |
|   128 |   21.56 |    15.66 |      407 |    0.9993 |
|   512 |   36.02 |    25.33 |     1007 |    0.9993 |
|  1024 |   67.76 |    49.74 |     1025 |    0.9993 |
|  2048 |  124.61 |    95.01 |     1074 |    0.9993 |
|  4096 |  247.78 |   156.49 |     1304 |    0.9993 |

### cubic (M=N=K) — peak-MFU sweep

|  M=N=K | BF16 µs | BSFP8 µs | BSFP8 TF | BSFP8 cos |
|   ---: |    ---: |     ---: |     ---: |      ---: |
|   1024 |    6.39 |     6.32 |      340 |    0.9993 |
|   1536 |   13.62 |    11.54 |      628 |    0.9993 |
|   2048 |   24.14 |    18.95 |      907 |    0.9993 |
|   2560 |   47.14 |    36.83 |      911 |    0.9993 |
|   3072 |   73.63 |    56.62 |     1024 |    0.9993 |
|   4096 |  172.00 |   113.57 |     1210 |    0.9993 |
|   6144 |  605.15 |   363.91 |     1275 |    0.9993 |
|   8192 | 1544.18 |   950.97 |     1156 |    0.9993 |
|  12288 | 5067.04 |  3167.36 |     1172 |    0.9993 |
|  16384 | 12993.08 |  8176.30 |     1076 |    0.9993 |

## MLP block forward (end-to-end, CUDA-Graph replay)

Full Qwen3-4B SwiGLU MLP forward as one timed closure: activation
quantize → `gate_up` FP8 GEMM → SwiGLU (chunk + silu·gate) → activation
quantize → `down` FP8 GEMM. Weights are pre-quantized + pre-packed at
setup; activations quantize **inside** the timed region. BF16 path is
two `F.linear` calls with the SwiGLU in between, no quantize. The
whole closure is captured into a `torch.cuda.CUDAGraph` and
`g.replay()` is timed — this is the per-block decode budget a
production graph-captured loop pays. `cos` is cosine similarity vs
the BF16 path's output.

The `sMM` column (sm_120 only) is the same closure with
`torch.nn.functional.scaled_mm` (cuBLAS / cuBLASLt MXFP8) in place of
`linear_mxfp8`, and PyTorch's reference `to_mxfp` + `to_blocked` (in
`torch.testing._internal.common_quantized`) doing the activation
quantize. Both quantize calls run inside the graph so their Python
overhead is paid once at capture, not per replay.

The `sMM-c` column is the same path with the activation quantize
wrapped in `torch.compile(mode="default")` so Inductor fuses the ~10
small ops in `to_mxfp` + the pad/permute/reshape of `to_blocked` into
a single kernel. (`default` mode is used not `reduce-overhead`
because the latter does its own inner cudagraph capture that
conflicts with the outer `_time_graph` capture.)

FSO's MXFP8 path uses two custom CUDA kernels that fuse multiple
operations into single passes:

* `fp8bs_quantize_1x32_packed_kernel` — fused BF16 → FP8 + packed
  int32 UE8M0 scale, replacing the legacy `quantize_1x32` + separate
  `repack_mxfp8_scales` two-step. uint64 LDG.64 loads (4 BF16/lane),
  8-lane sub-warp amax reductions, direct IEEE-754 bit manipulation
  for E8M0 RCEIL (skipping `__nv_cvt_float_to_e8m0`).
* `silu_chunk_mul_quantize_1x32_packed_kernel` — fused SwiGLU
  activation prologue: takes BF16 `gu = [gate || up]` from the
  `gate_up` GEMM and emits FP8 + packed scale directly, **without
  materialising the BF16 `h = silu(gate) * up` intermediate in
  global memory** (saves M·INTER·2 bytes of read+write per call —
  ~160 µs at M=4096 INTER=9728 on a 188-SM RTX PRO 6000).

The combination beats the equivalent Inductor-compiled chain
(`silu*mul` compiled + `to_mxfp+to_blocked` compiled, called
separately) by **2.1–4.3× at decode shapes** (M ≤ 128) and stays
within **±8% at prefill** (M ≥ 512) in the integrated MLP forward
closure — the compiled cuBLAS chain (`sMM-c`) edges fso only at
M=1024 (−7%), while fso is ahead again at M ≥ 2048. Against the
*unfused* cuBLAS reference (`sMM`) fso is **1.7–6.2× faster at
every M**.

Reproduce: `PYTHONPATH=python python bench/gemm/python/bench_qwen3_4b_mlp_forward.py --run --out <jsonl>`.

### 188 SMs · 2400 MHz no-boost · cu13 — NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120)

|       M | BF16 µs | BSFP8 µs | MXFP8 µs |  sMM µs | sMM-c µs | BSFP8 cos | MXFP8 cos | sMM cos | sMM-c cos |
|    ---: |    ---: |    ---: |    ---: |    ---: |    ---: |    ---: |    ---: |    ---: |    ---: |
|     1 |  112.74 |    45.07 |    36.93 |   200.54 |   139.45 |    0.9842 |    0.9979 |    0.9979 |    0.9979 |
|     2 |  116.16 |    43.11 |    34.89 |   200.90 |   143.47 |    0.9854 |    0.9980 |    0.9980 |    0.9980 |
|     4 |  115.43 |    43.12 |    34.88 |   204.93 |   143.94 |    0.9848 |    0.9979 |    0.9979 |    0.9979 |
|     8 |  114.91 |    43.00 |    34.56 |   202.74 |   137.30 |    0.9850 |    0.9979 |    0.9979 |    0.9979 |
|    16 |  116.83 |    38.97 |    32.81 |   204.78 |   139.49 |    0.9846 |    0.9979 |    0.9979 |    0.9979 |
|    32 |  126.80 |    36.90 |    30.95 |   202.87 |   133.16 |    0.9849 |    0.9979 |    0.9979 |    0.9979 |
|    64 |  126.72 |    41.08 |    38.98 |   182.49 |   113.08 |    0.9847 |    0.9979 |    0.9979 |    0.9979 |
|   128 |  143.44 |    53.40 |    51.95 |   189.47 |   108.64 |    0.9847 |    0.9979 |    0.9979 |    0.9979 |
|   512 |  274.79 |   195.62 |   179.54 |   342.41 |   180.64 |    0.9848 |    0.9979 |    0.9979 |    0.9979 |
|  1024 |  489.67 |   312.55 |   299.21 |   509.87 |   277.74 |    0.9848 |    0.9979 |    0.9979 |    0.9979 |
|  2048 |  906.54 |   575.25 |   512.45 |   970.14 |   512.33 |    0.9848 |    0.9979 |    0.9979 |    0.9979 |
|  4096 | 1795.68 |  1119.20 |  1041.06 |  2233.18 |  1095.69 |    0.9848 |    0.9979 |    0.9979 |    0.9979 |

### 170 SMs · 2400 MHz no-boost · cu13 — NVIDIA GeForce Blackwell GPU  (sm_120)

|       M | BF16 µs | BSFP8 µs | MXFP8 µs |  sMM µs | sMM-c µs | BSFP8 cos | MXFP8 cos | sMM cos | sMM-c cos |
|    ---: |    ---: |    ---: |    ---: |    ---: |    ---: |    ---: |    ---: |    ---: |    ---: |
|     1 |   91.47 |    43.16 |    34.94 |   188.61 |   136.92 |    0.9844 |    0.9980 |    0.9980 |    0.9980 |
|     2 |  109.02 |    41.47 |    33.15 |   190.99 |   139.57 |    0.9843 |    0.9978 |    0.9978 |    0.9978 |
|     4 |  109.04 |    43.09 |    34.89 |   194.74 |   139.67 |    0.9848 |    0.9979 |    0.9979 |    0.9979 |
|     8 |  132.86 |    41.09 |    32.87 |   192.28 |   135.41 |    0.9844 |    0.9979 |    0.9979 |    0.9979 |
|    16 |  133.56 |    39.00 |    30.84 |   194.25 |   136.00 |    0.9841 |    0.9979 |    0.9979 |    0.9979 |
|    32 |  116.38 |    34.90 |    30.84 |   192.30 |   129.84 |    0.9844 |    0.9979 |    0.9979 |    0.9979 |
|    64 |  119.28 |    39.60 |    38.32 |   174.33 |   110.74 |    0.9844 |    0.9979 |    0.9979 |    0.9979 |
|   128 |  132.53 |    53.42 |    51.67 |   186.25 |   106.39 |    0.9843 |    0.9979 |    0.9979 |    0.9979 |
|   512 |  445.88 |   189.57 |   174.70 |   336.59 |   187.78 |    0.9844 |    0.9979 |    0.9979 |    0.9979 |
|  1024 |  897.07 |   325.10 |   290.10 |   531.09 |   288.78 |    0.9843 |    0.9979 |    0.9979 |    0.9979 |
|  2048 | 1781.77 |   607.80 |   541.74 |  1123.17 |   571.40 |    0.9844 |    0.9979 |    0.9979 |    0.9979 |
|  4096 | 3442.88 |  1266.95 |  1096.08 |  2337.86 |  1176.31 |    0.9844 |    0.9979 |    0.9979 |    0.9979 |

### sm_90 · 1830 MHz no-boost · cu13 — NVIDIA H200 (sm_90)

|       M | BF16 µs | BSFP8 µs | BSFP8 cos |
|    ---: |    ---: |    ---: |    ---: |
|     1 |   50.93 |    42.10 |    0.9979 |
|     2 |   52.79 |    42.11 |    0.9978 |
|     4 |   53.35 |    42.53 |    0.9979 |
|     8 |   53.69 |    42.94 |    0.9979 |
|    16 |   54.60 |    43.16 |    0.9979 |
|    32 |   55.24 |    48.91 |    0.9979 |
|    64 |   58.14 |    44.24 |    0.9979 |
|   128 |   62.82 |    50.01 |    0.9979 |
|   512 |  137.74 |    98.04 |    0.9979 |
|  1024 |  265.96 |   181.05 |    0.9979 |
|  2048 |  528.25 |   334.18 |    0.9979 |
|  4096 | 1042.77 |   663.31 |    0.9979 |


# fish-scales-ops — frozen reference perf (attention)

CUDA-Graph capture-and-replay timings; median over 50 iters × 3 reps.
Three kernels exposed via `fso.attention.backends`:
- `sm120_mxfp8.mxfp8_fwd` — contiguous-K/V causal prefill (v15a).
- `sm120_mxfp8_decode.mxfp8_decode_paged_fwd` — single-token paged decode
  (v3, channel-K + channel-V scales).
- `sm120_mxfp8_paged_prefill.mxfp8_paged_prefill_fwd` — flashinfer-style
  ragged-Q + paged-KV extend (S_q > 1 with kv_history).

The decode and paged-prefill kernels share the same channel-scale layout:
K scale `[H_kv, D/32]` UE8M0 (block-scaled mma), V scale `[H_kv, D]` fp32
(applied post-mma in the epilogue). Both require `page_size ≥ 32` and
`page_size % 32 == 0`.

`sdpa` is `torch.nn.functional.scaled_dot_product_attention` on the same
BF16 inputs (with `repeat_interleave` for GQA expansion).
`flashinfer` is `BatchPrefillWithPagedKVCacheWrapper` / `BatchDecodeWithPagedKVCacheWrapper`
on BF16 K/V at the same page_size=32.
`cos` is cosine similarity of the MXFP8 output vs the reference.
Prefill / extend TFLOPS = `4·total_q·sum_kv·H_q·D / (µs · 1e6)` halved
for causal; decode TFLOPS uses `4·B·KV·H_q·D`.

Reproduce:
- `PYTHONPATH=python python bench/attention/bench_qwen3_llama3.py --out <jsonl>` — prefill + decode vs SDPA.
- `PYTHONPATH=python python bench/attention/bench_extend_vs_flashinfer.py` — paged-prefill / decode vs flashinfer BF16.

Outlier stress: append `--outliers` (5% channels × 20, 0.3% elements × 8).

## Blackwell GPU (170 SMs, sm_120a)

### Prefill (contiguous K/V, causal)

Three micro-opts stacked into v15a's main impl (in chronological order):
- **v18** (Ks-scale hoist) — pre-load Ks scales into a register array
  before the nt loop; ptxas didn't auto-hoist the runtime-`curr` index.
- **v19** (P-quant pack) — `cvt.rn.satfinite.e4m3x2.f32` + STS.U16 halves
  fp32→fp8 conversions and smem byte stores in P-quant.
- **bf16x2** (epilogue pack) — `cvt.rn.bf16x2.f32` + STG.B32 halves
  fp32→bf16 conversions and global stores in the output epilogue.

Cumulative gain vs original v15a: **~+10-18% TFLOPS** (shorter seqs gain
more because the epilogue is a bigger fraction of total). All cos checks
unchanged at ≥ 0.9993.

|     model |  B |    S | H_q/H_kv | mxfp8 µs | mxfp8 TF | sdpa µs | sdpa TF | speedup |    cos |
|       --: | -: |  --: |      --: |     --: |     ---: |    --: |    ---: |    ---: |   ---: |
| Llama3-8B  |  1 |  1024 | 32/8     |     33.0 |      260 |    84.0 |     102 |   2.54x | 0.9994 |
| Llama3-8B  |  1 |  2048 | 32/8     |     93.9 |      366 |   241.8 |     142 |   2.57x | 0.9994 |
| Llama3-8B  |  1 |  4096 | 32/8     |    361.9 |      380 |   789.3 |     174 |   2.18x | 0.9993 |
| Llama3-8B  |  1 |  8192 | 32/8     |   1491.3 |      369 |  2803.2 |     196 |   1.88x | 0.9993 |
| Llama3-8B  |  4 |  1024 | 32/8     |     95.9 |      358 |   231.6 |     148 |   2.41x | 0.9994 |
| Llama3-8B  |  4 |  2048 | 32/8     |    383.6 |      358 |   761.7 |     180 |   1.99x | 0.9993 |
| Llama3-8B  |  4 |  4096 | 32/8     |   1457.8 |      377 |  2733.5 |     201 |   1.88x | 0.9993 |
| Llama3-8B  |  4 |  8192 | 32/8     |   5522.5 |      398 | 10455.4 |     210 |   1.89x | 0.9993 |
| Llama3-70B |  1 |  1024 | 64/8     |     49.9 |      344 |   131.2 |     131 |   2.63x | 0.9994 |
| Llama3-70B |  1 |  2048 | 64/8     |    171.3 |      401 |   422.0 |     163 |   2.46x | 0.9994 |
| Llama3-70B |  1 |  4096 | 64/8     |    754.4 |      364 |  1438.2 |     191 |   1.91x | 0.9994 |
| Llama3-70B |  1 |  8192 | 64/8     |   2780.2 |      395 |  5383.4 |     204 |   1.94x | 0.9993 |
| Llama3-70B |  4 |  1024 | 64/8     |    195.7 |      351 |   422.4 |     163 |   2.16x | 0.9994 |
| Llama3-70B |  4 |  2048 | 64/8     |    787.3 |      349 |  1440.7 |     191 |   1.83x | 0.9994 |
| Llama3-70B |  4 |  4096 | 64/8     |   2799.5 |      393 |  5388.2 |     204 |   1.92x | 0.9993 |
| Llama3-70B |  4 |  8192 | 64/8     |  10871.6 |      405 | 20685.9 |     213 |   1.90x | 0.9993 |
| Qwen3-4B   |  1 |  1024 | 32/8     |     32.9 |      261 |    84.1 |     102 |   2.55x | 0.9994 |
| Qwen3-4B   |  1 |  2048 | 32/8     |     94.5 |      363 |   242.0 |     142 |   2.56x | 0.9994 |
| Qwen3-4B   |  1 |  4096 | 32/8     |    362.6 |      379 |   792.1 |     174 |   2.18x | 0.9993 |
| Qwen3-4B   |  1 |  8192 | 32/8     |   1505.4 |      365 |  2799.5 |     196 |   1.86x | 0.9993 |
| Qwen3-4B   |  4 |  1024 | 32/8     |     96.6 |      356 |   232.7 |     148 |   2.41x | 0.9994 |
| Qwen3-4B   |  4 |  2048 | 32/8     |    372.9 |      369 |   762.1 |     180 |   2.04x | 0.9993 |
| Qwen3-4B   |  4 |  4096 | 32/8     |   1472.4 |      373 |  2766.7 |     199 |   1.88x | 0.9993 |
| Qwen3-4B   |  4 |  8192 | 32/8     |   5563.2 |      395 | 10479.7 |     210 |   1.88x | 0.9993 |
| Qwen3-32B  |  1 |  1024 | 64/8     |     49.7 |      346 |   131.2 |     131 |   2.64x | 0.9994 |
| Qwen3-32B  |  1 |  2048 | 64/8     |    172.7 |      398 |   422.9 |     163 |   2.45x | 0.9994 |
| Qwen3-32B  |  1 |  4096 | 64/8     |    771.1 |      356 |  1444.5 |     190 |   1.87x | 0.9994 |
| Qwen3-32B  |  1 |  8192 | 64/8     |   2799.4 |      393 |  5377.9 |     204 |   1.92x | 0.9993 |
| Qwen3-32B  |  4 |  1024 | 64/8     |    196.7 |      349 |   422.8 |     163 |   2.15x | 0.9994 |
| Qwen3-32B  |  4 |  2048 | 64/8     |    783.8 |      351 |  1458.2 |     189 |   1.86x | 0.9994 |
| Qwen3-32B  |  4 |  4096 | 64/8     |   2817.2 |      390 |  5383.9 |     204 |   1.91x | 0.9993 |
| Qwen3-32B  |  4 |  8192 | 64/8     |  10920.3 |      403 | 20573.7 |     214 |   1.88x | 0.9993 |
| Qwen2-72B  |  1 |  1024 | 64/8     |     49.7 |      346 |   131.2 |     131 |   2.64x | 0.9994 |
| Qwen2-72B  |  1 |  2048 | 64/8     |    172.3 |      399 |   422.6 |     163 |   2.45x | 0.9994 |
| Qwen2-72B  |  1 |  4096 | 64/8     |    768.4 |      358 |  1442.6 |     191 |   1.88x | 0.9994 |
| Qwen2-72B  |  1 |  8192 | 64/8     |   2801.7 |      392 |  5324.3 |     207 |   1.90x | 0.9993 |
| Qwen2-72B  |  4 |  1024 | 64/8     |    196.6 |      350 |   422.9 |     162 |   2.15x | 0.9994 |
| Qwen2-72B  |  4 |  2048 | 64/8     |    773.9 |      355 |  1446.3 |     190 |   1.87x | 0.9994 |
| Qwen2-72B  |  4 |  4096 | 64/8     |   2816.0 |      390 |  5381.8 |     204 |   1.91x | 0.9993 |
| Qwen2-72B  |  4 |  8192 | 64/8     |  10892.7 |      404 | 20627.1 |     213 |   1.89x | 0.9993 |

### Decode (single-token, paged-KV, page_size=32)

Decode kernel is v3 with channel-K + channel-V; cross-batch packing
(4 warps × 4 work units per CTA) hides per-batch GQA-pack overhead.

|     model |  B |    KV | H_q/H_kv | mxfp8 µs | mxfp8 TF | sdpa µs | sdpa TF | speedup |    cos |
|       --: | -: |   --: |      --: |     --: |     ---: |    --: |    ---: |    ---: |   ---: |
| Llama3-8B  |  1 |  1024 | 32/8     |     22.6 |        1 |    15.3 |       1 |   0.68x | 0.9993 |
| Llama3-8B  |  1 |  4096 | 32/8     |     24.7 |        3 |    41.1 |       2 |   1.67x | 0.9993 |
| Llama3-8B  |  1 | 16384 | 32/8     |     34.9 |        8 |   170.3 |       2 |   4.88x | 0.9992 |
| Llama3-8B  |  4 |  1024 | 32/8     |     14.4 |        5 |    34.9 |       2 |   2.42x | 0.9993 |
| Llama3-8B  |  4 |  4096 | 32/8     |     24.7 |       11 |   174.4 |       2 |   7.07x | 0.9993 |
| Llama3-8B  |  4 | 16384 | 32/8     |     93.6 |       11 |   647.0 |       2 |   6.91x | 0.9993 |
| Llama3-70B |  1 |  1024 | 64/8     |     39.2 |        1 |    21.0 |       2 |   0.54x | 0.9993 |
| Llama3-70B |  1 |  4096 | 64/8     |     41.9 |        3 |    90.0 |       1 |   2.15x | 0.9992 |
| Llama3-70B |  1 | 16384 | 64/8     |     51.6 |       10 |   331.0 |       2 |   6.42x | 0.9993 |
| Llama3-70B |  4 |  1024 | 64/8     |     21.3 |        6 |   110.7 |       1 |   5.19x | 0.9992 |
| Llama3-70B |  4 |  4096 | 64/8     |     30.8 |       17 |   332.7 |       2 |  10.79x | 0.9993 |
| Llama3-70B |  4 | 16384 | 64/8     |    100.3 |       21 |  1278.8 |       2 |  12.75x | 0.9993 |
| Qwen3-4B   |  1 |  1024 | 32/8     |     22.6 |        1 |    15.5 |       1 |   0.68x | 0.9993 |
| Qwen3-4B   |  1 |  4096 | 32/8     |     24.7 |        3 |    41.1 |       2 |   1.66x | 0.9993 |
| Qwen3-4B   |  1 | 16384 | 32/8     |     34.9 |        8 |   170.2 |       2 |   4.88x | 0.9992 |
| Qwen3-4B   |  4 |  1024 | 32/8     |     14.4 |        5 |    34.9 |       2 |   2.42x | 0.9993 |
| Qwen3-4B   |  4 |  4096 | 32/8     |     24.7 |       11 |   174.4 |       2 |   7.07x | 0.9993 |
| Qwen3-4B   |  4 | 16384 | 32/8     |     93.1 |       12 |   647.4 |       2 |   6.95x | 0.9993 |
| Qwen3-32B  |  1 |  1024 | 64/8     |     39.4 |        1 |    21.1 |       2 |   0.53x | 0.9993 |
| Qwen3-32B  |  1 |  4096 | 64/8     |     41.7 |        3 |    89.3 |       2 |   2.14x | 0.9992 |
| Qwen3-32B  |  1 | 16384 | 64/8     |     51.8 |       10 |   331.1 |       2 |   6.39x | 0.9993 |
| Qwen3-32B  |  4 |  1024 | 64/8     |     21.1 |        6 |   110.8 |       1 |   5.24x | 0.9992 |
| Qwen3-32B  |  4 |  4096 | 64/8     |     30.8 |       17 |   332.6 |       2 |  10.80x | 0.9993 |
| Qwen3-32B  |  4 | 16384 | 64/8     |    100.4 |       21 |  1278.8 |       2 |  12.73x | 0.9993 |
| Qwen2-72B  |  1 |  1024 | 64/8     |     39.4 |        1 |    21.4 |       2 |   0.54x | 0.9993 |
| Qwen2-72B  |  1 |  4096 | 64/8     |     41.7 |        3 |    89.5 |       1 |   2.15x | 0.9992 |
| Qwen2-72B  |  1 | 16384 | 64/8     |     51.5 |       10 |   331.2 |       2 |   6.43x | 0.9993 |
| Qwen2-72B  |  4 |  1024 | 64/8     |     21.0 |        6 |   110.7 |       1 |   5.27x | 0.9992 |
| Qwen2-72B  |  4 |  4096 | 64/8     |     30.8 |       17 |   332.7 |       2 |  10.80x | 0.9993 |
| Qwen2-72B  |  4 | 16384 | 64/8     |    100.5 |       21 |  1278.9 |       2 |  12.72x | 0.9993 |

### Paged prefill (extend, page_size=32) — vs flashinfer BF16

Same micro-opt stack as the contiguous prefill (v18 Ks-hoist + v19
P-pack + bf16x2 epilogue pack). The paged kernel sees an additional
**−2-6% µs** on extend cells from the bf16x2 stage on top of the v19
gains. Decode is unaffected (S_q=1, epilogue is negligible).

Same paged KV cache layout as decode (channel-K + channel-V). Mma
configuration: kBr=64 Q-tile rows, kBc=32 K-tile cols, 4 warps along M.
flashinfer baseline: `BatchPrefillWithPagedKVCacheWrapper` (BF16, same
page_size) or `BatchDecodeWithPagedKVCacheWrapper` for the S_q=1 cells.
The bench routes S_q=1 cells through `mxfp8_decode_paged_fwd` to mirror
how sglang's `FlashInferAttnBackend` picks decode vs prefill ops.

| shape                                | mode   |  B |   tQ |   FI µs | mxfp8 µs |  FI TF | mxfp8 TF | speedup |    cos |
| :----------------------------------- | :----- | -: |  --: |    ---: |     ---: |   ---: |     ---: |    ---: |   ---: |
| Llama3-8B  decode  B=8 kv=1024       | decode |  8 |    8 |    31.0 |     28.1 |    4.3 |      4.8 |   1.10x | 0.9992 |
| Llama3-8B  decode  B=8 kv=8192       | decode |  8 |    8 |   167.6 |     91.8 |    6.4 |     11.7 |   1.83x | 0.9992 |
| Llama3-8B  extend64  B=4 kv=1024     | extend |  4 |  256 |    44.2 |     38.1 |  100.3 |    116.4 |   1.16x | 0.9993 |
| Llama3-8B  extend256 B=4 kv=4096     | extend |  4 | 1024 |   451.2 |    375.9 |  157.1 |    188.5 |   1.20x | 0.9992 |
| Llama3-8B  prefill   B=2 S=2048      | extend |  2 | 4096 |   425.6 |    335.5 |  161.5 |    204.9 |   1.27x | 0.9994 |
| Llama3-8B  prefill   B=1 S=8192      | extend |  1 | 8192 |  2876.1 |   2597.5 |  191.2 |    211.7 |   1.11x | 0.9993 |
| Qwen3-32B  decode  B=8 kv=4096       | decode |  8 |    8 |    91.2 |     36.1 |    5.9 |     14.9 |   2.53x | 0.9993 |
| Qwen3-32B  extend64  B=8 kv=4096     | extend |  8 |  512 |   247.5 |    184.0 |  140.0 |    188.3 |   1.35x | 0.9993 |
| Qwen2-72B  decode  B=8 kv=4096       | decode |  8 |    8 |    91.8 |     42.2 |   11.7 |     25.4 |   2.18x | 0.9992 |
| Qwen2-72B  extend64  B=8 kv=4096     | extend |  8 |  512 |   440.0 |    365.8 |  157.4 |    189.4 |   1.20x | 0.9993 |
| Qwen3-32B  mixed  B=4                | extend |  4 |  338 |   167.6 |    143.0 |  135.1 |    158.3 |   1.17x | 0.9985 |

Takeaways:
- Decode beats flashinfer BF16 by **1.1× – 2.5×** (MXFP8 K/V is 2× the
  BW of BF16; v3 cross-batch packing handles GQA cleanly).
- Short extend (S_q=64 with kv_history) beats flashinfer by **1.16× – 1.35×**.
- Mid extend (S_q=256, 1024) beats flashinfer by **1.17× – 1.27×**.
- Long single-batch prefill (S_q=8192) is **+11% over** flashinfer's FA3
  prefill (was −8% pre-v19, +8% pre-bf16x2).
