"""Third-party sources vendored into fish-scales-ops.

Each subpackage here is a trimmed copy of code from another project, kept
under its own licence with its own ``LICENSE`` and ``README.md`` recording the
source package, version, file paths, date and what was trimmed. Nothing in
this package is part of the public API, and nothing here is imported unless a
platform-specific path needs it.

  ``flashinfer_sm100_blockscaled``
      NVIDIA SM100 block-scaled CuTe-DSL GEMM kernels (BSD-3-Clause), used by
      the sm_100/sm_103 MXFP8 decode row in
      :mod:`fish_scales_ops.gemm._sm100_decode`.
"""
