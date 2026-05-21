# fish-scales-ops — frozen reference perf (GEMM)

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

## Blackwell GPU (170 SMs, sm_120a)

### `wqkv` (N=6144, K=2560)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | MXFP8 µs | MXFP8 TF | sMM µs | sMM TF | BSFP8 cos | MXFP8 cos | sMM cos |
|  ---: |    ---: |     ---: |     ---: |     ---: |     ---: |   ---: |   ---: |      ---: |      ---: |    ---: |
|     1 |   14.43 |    12.36 |        3 |    10.34 |        3 |    20.54 |        2 |    0.9951 |    0.9993 |    0.9993 |
|     2 |   12.60 |    10.34 |        6 |    10.37 |        6 |    20.60 |        3 |    0.9948 |    0.9993 |    0.9993 |
|     4 |   14.41 |    12.35 |       10 |    12.35 |       10 |    20.56 |        6 |    0.9945 |    0.9993 |    0.9993 |
|     8 |   14.41 |    12.33 |       20 |    10.33 |       24 |    20.51 |       12 |    0.9947 |    0.9993 |    0.9993 |
|    16 |   16.46 |    10.76 |       47 |    10.33 |       49 |    20.54 |       25 |    0.9943 |    0.9993 |    0.9993 |
|    32 |   18.54 |    10.34 |       97 |    12.31 |       82 |    18.91 |       53 |    0.9942 |    0.9993 |    0.9993 |
|    64 |   18.51 |    12.39 |      162 |    12.34 |      163 |    18.50 |      109 |    0.9946 |    0.9993 |    0.9993 |
|   128 |   32.85 |    14.54 |      277 |    16.46 |      245 |    18.55 |      217 |    0.9943 |    0.9993 |    0.9993 |
|   512 |   94.39 |    39.31 |      410 |    35.90 |      449 |    39.12 |      412 |    0.9940 |    0.9993 |    0.9993 |
|  1024 |  154.30 |    69.40 |      464 |    51.40 |      627 |    58.22 |      553 |    0.9946 |    0.9993 |    0.9993 |
|  2048 |  302.64 |   126.25 |      510 |    98.71 |      653 |   100.62 |      640 |    0.9948 |    0.9993 |    0.9993 |
|  4096 |  583.38 |   250.14 |      515 |   197.75 |      652 |   194.89 |      661 |    0.9948 |    0.9993 |    0.9993 |

### `wo` (N=2560, K=4096)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | MXFP8 µs | MXFP8 TF | sMM µs | sMM TF | BSFP8 cos | MXFP8 cos | sMM cos |
|  ---: |    ---: |     ---: |     ---: |     ---: |     ---: |   ---: |   ---: |      ---: |      ---: |    ---: |
|     1 |   10.54 |    12.38 |        2 |    12.37 |        2 |    28.62 |        1 |    0.9885 |    0.9993 |    0.9993 |
|     2 |   16.48 |    12.38 |        3 |    10.32 |        4 |    28.61 |        1 |    0.9884 |    0.9993 |    0.9993 |
|     4 |   16.47 |    14.39 |        6 |    10.34 |        8 |    28.74 |        3 |    0.9881 |    0.9993 |    0.9993 |
|     8 |   16.48 |    12.33 |       14 |    10.34 |       16 |    28.75 |        6 |    0.9880 |    0.9993 |    0.9993 |
|    16 |   18.50 |    12.38 |       27 |    10.30 |       33 |    28.74 |       12 |    0.9897 |    0.9993 |    0.9993 |
|    32 |   10.34 |    10.33 |       65 |    12.35 |       54 |    28.50 |       24 |    0.9888 |    0.9993 |    0.9993 |
|    64 |   12.39 |    12.35 |      109 |    12.36 |      109 |    28.29 |       47 |    0.9876 |    0.9993 |    0.9993 |
|   128 |   18.52 |    13.01 |      206 |    14.40 |      186 |    28.72 |       93 |    0.9890 |    0.9993 |    0.9993 |
|   512 |   49.44 |    24.12 |      445 |    24.63 |      436 |    28.76 |      373 |    0.9877 |    0.9993 |    0.9993 |
|  1024 |   96.10 |    45.22 |      475 |    34.55 |      622 |    35.19 |      610 |    0.9888 |    0.9993 |    0.9993 |
|  2048 |  188.89 |    87.01 |      494 |    65.77 |      653 |    65.48 |      656 |    0.9888 |    0.9993 |    0.9993 |
|  4096 |  377.05 |   170.63 |      503 |   130.81 |      657 |   128.81 |      667 |    0.9897 |    0.9993 |    0.9993 |

### `gate_up` (N=19456, K=2560)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | MXFP8 µs | MXFP8 TF | sMM µs | sMM TF | BSFP8 cos | MXFP8 cos | sMM cos |
|  ---: |    ---: |     ---: |     ---: |     ---: |     ---: |   ---: |   ---: |      ---: |      ---: |    ---: |
|     1 |   49.24 |    20.57 |        5 |    16.45 |        6 |    52.39 |        2 |    0.9944 |    0.9993 |    0.9993 |
|     2 |   30.37 |    18.53 |       11 |    14.44 |       14 |    52.41 |        4 |    0.9943 |    0.9993 |    0.9993 |
|     4 |   31.06 |    20.29 |       20 |    14.53 |       27 |    49.25 |        8 |    0.9945 |    0.9993 |    0.9993 |
|     8 |   33.71 |    18.51 |       43 |    14.42 |       55 |    51.27 |       16 |    0.9945 |    0.9993 |    0.9993 |
|    16 |   38.86 |    16.48 |       97 |    12.39 |      129 |    49.16 |       32 |    0.9948 |    0.9993 |    0.9993 |
|    32 |   32.92 |    14.42 |      221 |    14.42 |      221 |    43.39 |       73 |    0.9947 |    0.9993 |    0.9993 |
|    64 |   35.05 |    16.45 |      388 |    18.49 |      345 |    23.51 |      271 |    0.9947 |    0.9993 |    0.9993 |
|   128 |   64.68 |    22.93 |      556 |    24.67 |      517 |    24.81 |      514 |    0.9948 |    0.9993 |    0.9993 |
|   512 |  244.41 |    95.70 |      533 |    84.92 |      601 |    82.15 |      621 |    0.9946 |    0.9993 |    0.9993 |
|  1024 |  493.49 |   164.22 |      621 |   160.39 |      636 |   155.47 |      656 |    0.9945 |    0.9993 |    0.9993 |
|  2048 |  942.43 |   297.14 |      687 |   308.81 |      661 |   305.90 |      667 |    0.9946 |    0.9993 |    0.9993 |
|  4096 | 1764.07 |   606.78 |      672 |   611.72 |      667 |   601.54 |      678 |    0.9948 |    0.9993 |    0.9993 |

### `gate` (N=9728, K=2560)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | MXFP8 µs | MXFP8 TF | sMM µs | sMM TF | BSFP8 cos | MXFP8 cos | sMM cos |
|  ---: |    ---: |     ---: |     ---: |     ---: |     ---: |   ---: |   ---: |      ---: |      ---: |    ---: |
|     1 |   18.86 |    14.43 |        3 |    12.38 |        4 |    28.76 |        2 |    0.9951 |    0.9993 |    0.9993 |
|     2 |   14.45 |    12.35 |        8 |    12.37 |        8 |    28.78 |        3 |    0.9944 |    0.9993 |    0.9993 |
|     4 |   14.66 |    14.36 |       14 |    12.37 |       16 |    28.79 |        7 |    0.9945 |    0.9993 |    0.9993 |
|     8 |   16.49 |    14.40 |       28 |    12.34 |       32 |    26.72 |       15 |    0.9947 |    0.9993 |    0.9993 |
|    16 |   18.54 |    12.37 |       64 |    12.39 |       64 |    26.70 |       30 |    0.9947 |    0.9993 |    0.9993 |
|    32 |   20.81 |    12.36 |      129 |    12.36 |      129 |    24.68 |       65 |    0.9948 |    0.9993 |    0.9993 |
|    64 |   22.61 |    13.37 |      238 |    12.39 |      257 |    18.54 |      172 |    0.9944 |    0.9993 |    0.9993 |
|   128 |   32.87 |    16.43 |      388 |    16.47 |      387 |    18.64 |      342 |    0.9943 |    0.9993 |    0.9993 |
|   512 |  121.14 |    56.01 |      455 |    43.60 |      585 |    42.60 |      599 |    0.9948 |    0.9993 |    0.9993 |
|  1024 |  241.95 |   102.12 |      499 |    83.62 |      610 |    80.55 |      633 |    0.9945 |    0.9993 |    0.9993 |
|  2048 |  460.08 |   202.52 |      504 |   150.38 |      678 |   156.61 |      651 |    0.9946 |    0.9993 |    0.9993 |
|  4096 |  939.77 |   393.32 |      519 |   303.40 |      672 |   302.07 |      675 |    0.9946 |    0.9993 |    0.9993 |

### `down` (N=2560, K=9728)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | MXFP8 µs | MXFP8 TF | sMM µs | sMM TF | BSFP8 cos | MXFP8 cos | sMM cos |
|  ---: |    ---: |     ---: |     ---: |     ---: |     ---: |   ---: |   ---: |      ---: |      ---: |    ---: |
|     1 |   14.69 |    14.35 |        3 |    12.58 |        4 |    59.49 |        1 |    0.9955 |    0.9993 |    0.9993 |
|     2 |   16.47 |    14.35 |        7 |    13.71 |        7 |    59.52 |        2 |    0.9955 |    0.9993 |    0.9993 |
|     4 |   16.52 |    14.34 |       14 |    12.45 |       16 |    59.50 |        3 |    0.9957 |    0.9993 |    0.9993 |
|     8 |   36.96 |    14.40 |       28 |    12.39 |       32 |    59.49 |        7 |    0.9959 |    0.9993 |    0.9993 |
|    16 |   39.00 |    13.69 |       58 |    12.34 |       65 |    59.52 |       13 |    0.9956 |    0.9993 |    0.9993 |
|    32 |   18.53 |    13.62 |      117 |    12.37 |      129 |    59.51 |       27 |    0.9956 |    0.9993 |    0.9993 |
|    64 |   20.53 |    16.46 |      194 |    14.56 |      219 |    59.49 |       54 |    0.9952 |    0.9993 |    0.9993 |
|   128 |   33.02 |    19.16 |      333 |    20.54 |      310 |    59.52 |      107 |    0.9958 |    0.9993 |    0.9993 |
|   512 |  112.88 |    57.93 |      440 |    62.28 |      409 |    61.61 |      414 |    0.9957 |    0.9993 |    0.9993 |
|  1024 |  223.48 |   101.24 |      504 |    77.13 |      661 |    79.31 |      643 |    0.9953 |    0.9993 |    0.9993 |
|  2048 |  437.19 |   200.05 |      510 |   151.38 |      674 |   153.01 |      667 |    0.9954 |    0.9993 |    0.9993 |
|  4096 |  907.89 |   407.77 |      500 |   303.84 |      671 |   300.46 |      679 |    0.9957 |    0.9993 |    0.9993 |

### cubic (M=N=K) — peak-MFU sweep

|  M=N=K | BF16 µs | BSFP8 µs | BSFP8 TF | MXFP8 µs | MXFP8 TF | sMM µs | sMM TF | BSFP8 cos | MXFP8 cos | sMM cos |
|   ---: |    ---: |     ---: |     ---: |     ---: |     ---: |   ---: |   ---: |      ---: |      ---: |    ---: |
|   1024 |   14.44 |    12.38 |      174 |     8.28 |      259 |    10.35 |      208 |    0.9712 |    0.9993 |    0.9993 |
|   1536 |   39.04 |    30.10 |      241 |    14.80 |      490 |    14.96 |      485 |    0.9870 |    0.9993 |    0.9993 |
|   2048 |   98.41 |    57.49 |      299 |    35.08 |      490 |    33.24 |      517 |    0.9952 |    0.9993 |    0.9993 |
|   2560 |  155.03 |    70.12 |      479 |    51.43 |      652 |    59.32 |      566 |    0.9945 |    0.9993 |    0.9993 |
|   3072 |  254.81 |   116.33 |      498 |    89.35 |      649 |    94.44 |      614 |    0.9943 |    0.9993 |    0.9993 |
|   4096 |  631.74 |   256.26 |      536 |   193.38 |      711 |   210.08 |      654 |    0.9897 |    0.9993 |    0.9993 |
|   6144 | 2098.14 |   900.96 |      515 |   704.32 |      659 |   710.00 |      653 |    0.9672 |    0.9993 |    0.9993 |
|   8192 | 4992.52 |  2109.80 |      521 |  1672.82 |      657 |  1711.64 |      642 |    0.9942 |    0.9993 |    0.9993 |
|  12288 | 16645.66 |  5470.00 |      678 |  5605.42 |      662 |  5984.61 |      620 |    0.9960 |    0.9993 |    0.9993 |
|  16384 | 39409.32 | 12910.20 |      681 | 13305.34 |      661 | 14257.21 |      617 |    0.9949 |    0.9993 |    0.9993 |

## H200 — NVIDIA H200 (sm_90)

### `wqkv` (N=6144, K=2560)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | BSFP8 cos |
|  ---: |    ---: |     ---: |     ---: |      ---: |
|     1 |    9.81 |     7.62 |        4 |    0.9993 |
|     2 |    9.30 |     7.26 |        9 |    0.9993 |
|     4 |    9.09 |     7.63 |       16 |    0.9993 |
|     8 |    8.01 |     7.22 |       35 |    0.9993 |
|    16 |    8.24 |     6.84 |       74 |    0.9993 |
|    32 |    8.41 |     9.16 |      110 |    0.9993 |
|    64 |    8.80 |     6.29 |      320 |    0.9993 |
|   128 |    8.71 |     7.58 |      531 |    0.9993 |
|   512 |   23.57 |    18.18 |      886 |    0.9993 |
|  1024 |   44.14 |    30.36 |     1061 |    0.9993 |
|  2048 |   84.63 |    58.20 |     1107 |    0.9993 |
|  4096 |  163.31 |   113.30 |     1137 |    0.9993 |

### `wo` (N=2560, K=4096)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | BSFP8 cos |
|  ---: |    ---: |     ---: |     ---: |      ---: |
|     1 |    8.88 |     8.03 |        3 |    0.9993 |
|     2 |    9.05 |     7.93 |        5 |    0.9993 |
|     4 |    8.69 |     8.06 |       10 |    0.9993 |
|     8 |    7.93 |     7.75 |       22 |    0.9993 |
|    16 |    8.15 |     7.77 |       43 |    0.9993 |
|    32 |    8.55 |    10.01 |       67 |    0.9993 |
|    64 |    9.26 |     7.28 |      184 |    0.9993 |
|   128 |   10.25 |     8.52 |      315 |    0.9993 |
|   512 |   18.26 |    12.64 |      850 |    0.9993 |
|  1024 |   30.06 |    23.11 |      929 |    0.9993 |
|  2048 |   55.02 |    42.41 |     1013 |    0.9993 |
|  4096 |  107.64 |    71.25 |     1206 |    0.9993 |

### `gate_up` (N=19456, K=2560)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | BSFP8 cos |
|  ---: |    ---: |     ---: |     ---: |      ---: |
|     1 |   30.28 |    17.88 |        6 |    0.9993 |
|     2 |   30.48 |    17.67 |       11 |    0.9993 |
|     4 |   30.73 |    17.99 |       22 |    0.9993 |
|     8 |   30.71 |    17.93 |       44 |    0.9993 |
|    16 |   31.18 |    18.01 |       88 |    0.9993 |
|    32 |   31.67 |    18.73 |      170 |    0.9993 |
|    64 |   32.61 |    16.52 |      386 |    0.9993 |
|   128 |   30.77 |    19.98 |      638 |    0.9993 |
|   512 |   72.01 |    53.29 |      957 |    0.9993 |
|  1024 |  140.03 |   101.61 |     1004 |    0.9993 |
|  2048 |  274.80 |   178.07 |     1146 |    0.9993 |
|  4096 |  540.03 |   369.60 |     1104 |    0.9993 |

### `gate` (N=9728, K=2560)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | BSFP8 cos |
|  ---: |    ---: |     ---: |     ---: |      ---: |
|     1 |   13.44 |     8.70 |        6 |    0.9993 |
|     2 |   13.40 |     8.38 |       12 |    0.9993 |
|     4 |   12.65 |     8.73 |       23 |    0.9993 |
|     8 |   13.51 |     8.14 |       49 |    0.9993 |
|    16 |   13.53 |     7.73 |      103 |    0.9993 |
|    32 |   14.28 |    10.59 |      150 |    0.9993 |
|    64 |   15.40 |     7.70 |      414 |    0.9993 |
|   128 |   14.70 |     9.27 |      688 |    0.9993 |
|   512 |   36.82 |    28.92 |      882 |    0.9993 |
|  1024 |   70.25 |    51.91 |      983 |    0.9993 |
|  2048 |  138.49 |   100.83 |     1012 |    0.9993 |
|  4096 |  272.01 |   178.52 |     1143 |    0.9993 |

### `down` (N=2560, K=9728)

|     M | BF16 µs | BSFP8 µs | BSFP8 TF | BSFP8 cos |
|  ---: |    ---: |     ---: |     ---: |      ---: |
|     1 |   15.56 |    14.63 |        3 |    0.9994 |
|     2 |   15.71 |    14.63 |        7 |    0.9993 |
|     4 |   15.68 |    14.52 |       14 |    0.9993 |
|     8 |   15.74 |    14.21 |       28 |    0.9993 |
|    16 |   15.97 |    14.51 |       55 |    0.9993 |
|    32 |   17.01 |    19.85 |       80 |    0.9993 |
|    64 |   17.64 |    13.95 |      229 |    0.9993 |
|   128 |   21.26 |    15.59 |      409 |    0.9993 |
|   512 |   36.12 |    25.22 |     1011 |    0.9993 |
|  1024 |   67.86 |    49.66 |     1027 |    0.9993 |
|  2048 |  124.59 |    95.18 |     1072 |    0.9993 |
|  4096 |  247.68 |   156.47 |     1304 |    0.9993 |

### cubic (M=N=K) — peak-MFU sweep

|  M=N=K | BF16 µs | BSFP8 µs | BSFP8 TF | BSFP8 cos |
|   ---: |    ---: |     ---: |     ---: |      ---: |
|   1024 |    6.30 |     6.11 |      352 |    0.9993 |
|   1536 |   13.53 |    11.50 |      630 |    0.9993 |
|   2048 |   23.95 |    18.74 |      917 |    0.9993 |
|   2560 |   46.95 |    36.78 |      912 |    0.9993 |
|   3072 |   74.06 |    56.64 |     1024 |    0.9993 |
|   4096 |  172.27 |   113.53 |     1211 |    0.9993 |
|   6144 |  571.07 |   370.42 |     1252 |    0.9993 |
|   8192 | 1596.31 |   940.06 |     1170 |    0.9993 |
|  12288 | 5288.53 |  3254.59 |     1140 |    0.9993 |
|  16384 | 13021.68 |  7904.02 |     1113 |    0.9993 |

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
a single kernel. This closes most of the gap vs `MXFP8` — and at
M ≥ 1024 actually **beats** FSO's `MXFP8` closure by ~10-13%, because
cuBLAS's MXFP8 GEMM kernel at large M has a slight edge over FSO's
hand-tuned cascade and the fused quantize is no longer the
bottleneck. `default` mode is used (not `reduce-overhead`) because
the latter does its own inner cudagraph capture that conflicts with
the outer `_time_graph` capture.

Reproduce: `PYTHONPATH=python python bench/gemm/python/bench_qwen3_4b_mlp_forward.py --run --out <jsonl>`.

### Blackwell GPU (170 SMs, sm_120a)

|       M | BF16 µs | BSFP8 µs | MXFP8 µs |  sMM µs | sMM-c µs | BSFP8 cos | MXFP8 cos | sMM cos | sMM-c cos |
|    ---: |    ---: |    ---: |    ---: |    ---: |    ---: |    ---: |    ---: |    ---: |    ---: |
|     1 |   84.13 |    39.42 |    34.89 |   160.03 |   116.96 |    0.9844 |    0.9980 |    0.9980 |    0.9980 |
|     2 |  102.56 |    39.02 |    34.90 |   164.04 |   120.99 |    0.9843 |    0.9978 |    0.9978 |    0.9978 |
|     4 |  102.69 |    39.05 |    32.89 |   167.14 |   121.01 |    0.9848 |    0.9979 |    0.9979 |    0.9979 |
|     8 |  121.26 |    39.02 |    32.86 |   164.19 |   116.87 |    0.9844 |    0.9979 |    0.9979 |    0.9979 |
|    16 |  122.53 |    36.93 |    30.83 |   166.47 |   117.52 |    0.9841 |    0.9979 |    0.9979 |    0.9979 |
|    32 |  108.80 |    32.88 |    30.83 |   164.49 |   112.77 |    0.9844 |    0.9979 |    0.9979 |    0.9979 |
|    64 |  111.85 |    37.00 |    38.91 |   149.62 |    94.39 |    0.9844 |    0.9979 |    0.9979 |    0.9979 |
|   128 |  124.90 |    51.24 |    53.88 |   160.12 |    92.26 |    0.9843 |    0.9979 |    0.9979 |    0.9979 |
|   512 |  374.38 |   185.70 |   187.70 |   289.75 |   170.13 |    0.9844 |    0.9979 |    0.9979 |    0.9979 |
|  1024 |  759.18 |   324.70 |   323.93 |   468.41 |   289.78 |    0.9843 |    0.9979 |    0.9979 |    0.9979 |
|  2048 | 1497.46 |   655.17 |   661.92 |  1069.77 |   593.00 |    0.9844 |    0.9979 |    0.9979 |    0.9979 |
|  4096 | 2951.57 |  1366.73 |  1385.67 |  2277.85 |  1227.02 |    0.9844 |    0.9979 |    0.9979 |    0.9979 |

### H200 — NVIDIA H200 (sm_90)

|       M | BF16 µs | BSFP8 µs | BSFP8 cos |
|    ---: |    ---: |    ---: |    ---: |
|     1 |   50.54 |    41.43 |    0.9980 |
|     2 |   52.44 |    43.48 |    0.9978 |
|     4 |   53.02 |    44.02 |    0.9979 |
|     8 |   53.37 |    44.09 |    0.9979 |
|    16 |   54.18 |    44.40 |    0.9979 |
|    32 |   54.84 |    49.50 |    0.9979 |
|    64 |   57.17 |    46.63 |    0.9979 |
|   128 |   61.84 |    55.20 |    0.9979 |
|   512 |  136.90 |   121.68 |    0.9979 |
|  1024 |  265.42 |   234.33 |    0.9979 |
|  2048 |  528.43 |   433.60 |    0.9979 |
|  4096 | 1061.84 |   801.83 |    0.9979 |



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
