"""Trimmed copy of NVIDIA's SM100 block-scaled CuTe-DSL GEMM kernels.

Source: flashinfer-python 0.6.18.post1,
``flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100{,_splitk,_common}.py``
(Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES, SPDX-License-Identifier:
BSD-3-Clause). See ``README.md`` for the exact provenance, the md5 table and
the list of trims, and ``LICENSE`` for the licence text.

Nothing is imported here: the kernel modules import ``cutlass`` at module
scope, and on an nvidia-cutlass-dsl older than 4.5.0 that import raises. The
only importer is :mod:`fish_scales_ops.gemm._sm100_decode`, which checks the
DSL version first.
"""
