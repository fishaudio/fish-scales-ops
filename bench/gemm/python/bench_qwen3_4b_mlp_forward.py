"""End-to-end MLP forward bench for Qwen3-4B.

The per-linear `bench_qwen3_4b_mlp.py` doesn't capture the host overhead
of chaining 5 ops back-to-back in a real forward (quantize → gate_up →
SwiGLU → quantize → down). This bench runs the full SwiGLU MLP block as
one closure, both eagerly and inside a CUDA-graph capture, so we see the
**actual** per-token MLP cost a production decode loop pays.

Pipeline (per call):

    BF16:
      gate_up = F.linear(x_bf16,  W_gate_up_bf16)
      h_bf16  = F.silu(gate) * up      where gate, up = gate_up.chunk(2, -1)
      y       = F.linear(h_bf16, W_down_bf16)

    BSFP8 (1×128):
      xq, sx     = quantize_1x128_fp8(x_bf16)         (+ repack on sm_120)
      gate_up_bf = linear_fp8(xq, Wgu_fp8, sx_p, sgu_p)
      h_bf16     = silu(gate) * up
      hq, sh     = quantize_1x128_fp8(h_bf16)          (+ repack)
      y          = linear_fp8(hq, Wd_fp8, sh_p, sd_p)

    MXFP8 (1×32):
      xq, sx     = quantize_1x32_fp8(x_bf16)
      gate_up_bf = linear_mxfp8(xq, Wgu_fp8, sx, sgu)
      h_bf16     = silu(gate) * up
      hq, sh     = quantize_1x32_fp8(h_bf16)
      y          = linear_mxfp8(hq, Wd_fp8, sh, sd)

Weights are pre-quantized + pre-packed at startup (production pattern —
model init does this once). Activations are quantised inside the timed
region.

Serving-library comparators (``--dtypes``; never in the default set, so the
default run's output is unchanged). Each takes the same BF16 weights,
block-quantized once at load to the layout a block-FP8 checkpoint stores
(weight_block_size [128, 128]: FP8 E4M3 weights, fp32 `weight_scale_inv` =
128x128-block amax / 448), and runs the whole block -- both activation
quantizations included -- inside the timed graph:

    sgl_fp8b     sglang's `Fp8LinearMethod` for both projections and its
                 `SiluAndMul` between them, exactly as sglang's Qwen3 MLP calls
                 them; the block-FP8 GEMM backend is the one sglang dispatches
                 to on this device (`initialize_fp8_gemm_config` then
                 `dispatch_w8a8_block_fp8_linear`, as the scheduler does:
                 DeepGEMM on sm_90 and sm_100, CUTLASS on sm_120), including
                 the weight post-processing of `process_weights_after_loading`
                 (the UE8M0 scale requantization for DeepGEMM on sm_100).
    vllm_fp8b    vLLM's `Fp8LinearMethod` (the block-FP8 kernel its
                 `init_fp8_linear_kernel` selects: DeepGEMM with UE8M0 scales
                 on sm_120) and its `SiluAndMul`, both in their CUDA custom-op
                 form (vLLM's eager-mode ops).
    cublas_fp8b  `torch.nn.functional.scaled_mm` with BlockWise1x128
                 activation and BlockWise128x128 weight scales (cuBLASLt), the
                 1x128 activation quantize and silu*mul each a
                 `torch.compile`d function, like the `smm_fast` closure; rows
                 are zero-padded to a multiple of 4, which cuBLASLt requires.
                 torch 2.13 takes this recipe on sm_90 only (n/a elsewhere).

Every comparator record names the backend that ran (`backend`, plus the
library's own dispatch object) and its library version; a comparator that is
not available on the device or in the environment is recorded as
`{"error": "n/a: <reason>"}`. FSO_BENCH_TRACE_KERNELS=1 additionally records
the CUDA kernels of one eager call (`kernels`), for smoke evidence only.

  python bench_qwen3_4b_mlp_forward.py --run --out <jsonl> [--dtypes sgl_fp8b,cublas_fp8b] [--Ms 1,64]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import math
import torch
import torch.nn.functional as F


# Qwen3-4B
HIDDEN = 2560
INTERMEDIATE = 9728
M_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]  # docs/perf/README.md §4

# The default set: the fso rows and their torch / cuBLAS columns, what a run without --dtypes measures.
FSO_DTYPES = ("bf16", "bsfp8", "mxfp8", "smm", "smm_fast")
# Serving-library comparators, measured only when --dtypes names them (see the module docstring).
CMP_DTYPES = ("sgl_fp8b", "vllm_fp8b", "cublas_fp8b")
# Every name --dtypes accepts; bench/run_perf.py validates the suite's --dtypes values against it.
DTYPES = FSO_DTYPES + CMP_DTYPES


def set_m_grid(ms):
    """--Ms: these M values replace the M grid."""
    global M_GRID
    M_GRID = list(ms)


def _make_bf16_weights(device="cuda"):
    torch.manual_seed(0xc0ffee)
    w_gate_up = (torch.randn(2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device=device) / (HIDDEN ** 0.5)).contiguous()
    w_down = (torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.bfloat16, device=device) / (INTERMEDIATE ** 0.5)).contiguous()
    return w_gate_up, w_down


@torch.compile(mode="default", dynamic=False)
def _silu_chunk_mul(gu):
    """SwiGLU: split gu in two halves along last dim, return silu(gate) * up.

    Compiled (Inductor) — the eager F.silu(gate) * up path materialises a
    silu_out intermediate (~M*INTER bf16 of memory traffic) that Inductor
    can fuse away, cutting total mem traffic from 5/8 → 3/8 of the M*INTER
    BF16 tensor count. Save: ~160 µs at M=4096 INTER=9728 on a 170-SM
    Blackwell GPU.
    """
    gate, up = gu.chunk(2, dim=-1)
    return F.silu(gate) * up


def _build_mlp_fn(dtype, sm_major):
    if dtype in CMP_DTYPES:
        w_gate_up_bf, w_down_bf = _make_bf16_weights()
        return _CMP_BUILDERS[dtype](w_gate_up_bf, w_down_bf, sm_major)
    import fish_scales_ops as fso
    w_gate_up_bf, w_down_bf = _make_bf16_weights()

    if dtype == "bf16":
        def fn(x_bf):
            gu = F.linear(x_bf, w_gate_up_bf)
            gate, up = gu.chunk(2, dim=-1)
            h = F.silu(gate) * up
            return F.linear(h, w_down_bf)
        return fn

    if dtype == "bsfp8":
        wgu_q, sgu = fso.compat.quantize_128x128_fp8(w_gate_up_bf)
        wd_q, sd = fso.compat.quantize_128x128_fp8(w_down_bf)
        if sm_major >= 10:
            sgu = fso.compat.repack_fp8_wgt_scales(sgu)
            sd = fso.compat.repack_fp8_wgt_scales(sd)

        if sm_major >= 10:
            # sm_120 / sm_100: use the fused single-kernel `quantize_1x128_fp8_packed`
            # which emits the int32-packed UE8M0 K-major scale layout directly,
            # skipping the separate `repack_fp8_act_scales` pass.
            def fn(x_bf):
                xq, sx = fso.compat.quantize_1x128_fp8_packed(x_bf)
                gu = fso.compat.linear_fp8(xq, wgu_q, sx, sgu)
                h = _silu_chunk_mul(gu)
                hq, sh = fso.compat.quantize_1x128_fp8_packed(h)
                return fso.compat.linear_fp8(hq, wd_q, sh, sd)
        else:
            # sm_90: deep_gemm path consumes FP32 scales directly. The
            # underlying `fp8bs_quantize_1x128` already routes to the fast
            # uint64 LDG.64 kernel when K%512==0.
            def fn(x_bf):
                xq, sx = fso.compat.quantize_1x128_fp8(x_bf, use_ue8m0=False)
                gu = fso.compat.linear_fp8(xq, wgu_q, sx, sgu)
                h = _silu_chunk_mul(gu)
                hq, sh = fso.compat.quantize_1x128_fp8(h, use_ue8m0=False)
                return fso.compat.linear_fp8(hq, wd_q, sh, sd)
        return fn

    if dtype == "mxfp8":
        if sm_major not in (10, 12):
            return None
        wgu_q, sgu = fso.compat.quantize_1x32_fp8(w_gate_up_bf)
        wd_q, sd = fso.compat.quantize_1x32_fp8(w_down_bf)

        def fn(x_bf):
            xq, sx = fso.compat.quantize_1x32_fp8(x_bf)
            gu = fso.compat.linear_mxfp8(xq, wgu_q, sx, sgu)
            # Fused silu(gate) * up + quantize → fp8 + packed scale,
            # no `h` intermediate.
            hq, sh = fso.compat.silu_chunk_mul_quantize_1x32_fp8(gu)
            return fso.compat.linear_mxfp8(hq, wd_q, sh, sd)
        return fn

    if dtype in ("smm", "smm_fast"):
        # cuBLAS MXFP8 path via torch.nn.functional.scaled_mm. Same closure
        # shape as the MXFP8 path so the comparison is apples-to-apples at
        # the MLP-block level (activation quantize → gate_up → SwiGLU →
        # activation quantize → down). Two variants:
        #   "smm"      — reference quantize via `to_mxfp` + `to_blocked`
        #                (~10 small kernels per call).
        #   "smm_fast" — same closure but wrapped in `torch.compile(default)`
        #                so Inductor fuses the quantize + blocked-layout
        #                permute into a single kernel. ~5-19× faster on the
        #                quantize step alone; works under outer CUDA-Graph
        #                capture because mode="default" suppresses Inductor's
        #                own cudagraph layer.
        if sm_major not in (10, 12):
            return None
        from torch.testing._internal.common_quantized import to_mxfp, to_blocked
        from torch._C import _ScalingType as ST, _SwizzleType as SW

        sgu_un, wgu_q = to_mxfp(w_gate_up_bf.contiguous(), 32, "mxfp8")
        sd_un,  wd_q  = to_mxfp(w_down_bf.contiguous(),    32, "mxfp8")
        sgu_b = to_blocked(sgu_un)
        sd_b  = to_blocked(sd_un)
        wgu_t = wgu_q.t()
        wd_t  = wd_q.t()

        def _quantize_to_blocked(t):
            s_un, q = to_mxfp(t.contiguous(), 32, "mxfp8")
            return q, to_blocked(s_un)

        if dtype == "smm_fast":
            _quantize_to_blocked = torch.compile(_quantize_to_blocked, mode="default", dynamic=False)

        def fn(x_bf):
            xq, sx_b = _quantize_to_blocked(x_bf)
            gu = F.scaled_mm(xq, wgu_t, sx_b, ST.BlockWise1x32, sgu_b, ST.BlockWise1x32,
                             swizzle_a=SW.SWIZZLE_32_4_4, swizzle_b=SW.SWIZZLE_32_4_4,
                             output_dtype=torch.bfloat16)
            h = _silu_chunk_mul(gu)
            hq, sh_b = _quantize_to_blocked(h)
            return F.scaled_mm(hq, wd_t, sh_b, ST.BlockWise1x32, sd_b, ST.BlockWise1x32,
                               swizzle_a=SW.SWIZZLE_32_4_4, swizzle_b=SW.SWIZZLE_32_4_4,
                               output_dtype=torch.bfloat16)
        return fn

    raise ValueError(f"unknown dtype {dtype!r}")


# ----------------------------------------------------------------------------- serving-library comparators
FP8_MAX = 448.0
# The quantization_config of a block-FP8 Qwen3 checkpoint (that of Qwen/Qwen3-4B-FP8). Both libraries parse it into
# their Fp8Config, and the config.json the serving-argument resolution reads is Qwen3-4B's around it.
FP8_BLOCK_QCFG = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}


class NotAvailable(Exception):
    """A comparator that cannot run on this device or in this environment; the cell records it as n/a."""


def _block_quant_128x128(w):
    """BF16 [N, K] -> (FP8 E4M3 [N, K], fp32 [N/128, K/128]): what a block-FP8 checkpoint stores for a linear
    (weight_block_size [128, 128], `weight_scale_inv` = the 128x128 block's amax / 448). Every comparator loads
    its weights in this form and post-processes them the way its library does at load."""
    N, K = w.shape
    b = w.float().view(N // 128, 128, K // 128, 128)
    s = b.abs().amax(dim=(1, 3)).clamp(min=1e-10) / FP8_MAX
    q = (b / s[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(N, K)
    return q.contiguous(), s.contiguous()


def _model_dir():
    """A directory holding only a config.json: Qwen3-4B's architecture with the block-FP8 quantization_config. The
    libraries resolve their serving arguments against it; no weights or tokenizer are read from it."""
    import atexit
    import shutil
    import tempfile
    d = tempfile.mkdtemp(prefix="fso-mlp-qwen3-4b-fp8-")
    atexit.register(shutil.rmtree, d, True)
    cfg = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": HIDDEN,
           "intermediate_size": INTERMEDIATE, "num_hidden_layers": 36, "num_attention_heads": 32,
           "num_key_value_heads": 8, "head_dim": 128, "hidden_act": "silu", "vocab_size": 151936,
           "max_position_embeddings": 40960, "rms_norm_eps": 1e-6, "rope_theta": 1000000,
           "tie_word_embeddings": True, "torch_dtype": "bfloat16", "quantization_config": dict(FP8_BLOCK_QCFG)}
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    return d


def _versions(*dists):
    from importlib import metadata
    out = {"torch": torch.__version__}
    for d in dists:
        try:
            out[d] = metadata.version(d)
        except Exception:
            pass
    return out


def _err(e, limit=300):
    return f"{type(e).__name__}: {' '.join(str(e).split())[:limit]}"


class _model_init:
    """What both libraries' model loaders hold while they build a model's layers: the model dtype (BF16) as torch's
    default dtype and the GPU as the default device. vLLM's Fp8LinearMethod takes its GEMM output dtype from the
    default dtype, and its DeepGEMM kernel accepts only a BF16 output, so the kernel choice depends on it."""

    def __enter__(self):
        self.old = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        self.dev = torch.device("cuda")
        self.dev.__enter__()

    def __exit__(self, *exc):
        self.dev.__exit__(*exc)
        torch.set_default_dtype(self.old)


def _scale_desc(t):
    return f"{str(t.dtype).replace('torch.', '')} {list(t.shape)}"


_SGL = {}


def _sglang_init():
    """Publish sglang's serving arguments for a block-FP8 Qwen3-4B checkpoint and initialize its FP8 GEMM
    configuration the way the scheduler does at startup (`initialize_fp8_gemm_config`, which maps `auto` to
    `cutlass` on sm_120), before any Fp8LinearMethod exists: the method picks its block-FP8 linear in its
    constructor. DeepGEMM's start-up pre-compile of every M from 1 to 16384 is turned off
    (SGLANG_JIT_DEEPGEMM_PRECOMPILE=0): it only fills the JIT cache ahead of time, the kernel DeepGEMM picks
    for an M is the same either way, and the cell's eager warmup compiles the one it needs."""
    if _SGL:
        return _SGL
    os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"     # read when sglang's DeepGEMM wrapper is imported
    try:
        import dataclasses
        from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    except Exception as e:
        raise NotAvailable(f"sglang does not import here ({_err(e, 160)})")
    if dataclasses.is_dataclass(ServerArgs):
        raise NotAvailable("this sglang predates 0.5.20 (dataclass ServerArgs); the cell drives 0.5.20's runtime context")
    set_global_server_args_for_scheduler(ServerArgs(model_path=_model_dir()))
    from sglang.srt.layers.quantization.fp8_utils import get_fp8_gemm_runner_backend, initialize_fp8_gemm_config
    initialize_fp8_gemm_config()
    _SGL["fp8_gemm_runner_backend"] = get_fp8_gemm_runner_backend().value
    return _SGL


def _sgl_backend(linear_name):
    """sglang's block-FP8 linear function -> the GEMM backend it runs (for shapes it does not hand to its
    Triton fallback; both projections here are multiples of 128 in N and K, so none is)."""
    n = linear_name.lower()
    for key, label in (("flashinfer_deepgemm", "FlashInfer DeepGEMM"), ("deepgemm", "DeepGEMM"),
                       ("flashinfer", "FlashInfer"), ("cutlass", "CUTLASS"), ("aiter", "AITER"),
                       ("triton", "Triton")):
        if key in n:
            return label
    return linear_name


def _build_sgl_fp8b(w_gu, w_d, sm_major):
    """sglang's dense block-FP8 MLP: the Qwen3 MLP's gate_up_proj -> SiluAndMul -> down_proj, each projection
    an Fp8LinearMethod over its own layer, as `Fp8Config.get_quant_method` gives every linear one."""
    ctx = _sglang_init()
    import torch.nn as nn
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
    qc = Fp8Config.from_config(dict(FP8_BLOCK_QCFG))

    def linear(w, partitions):
        N, K = w.shape
        wq, ws = _block_quant_128x128(w)
        with _model_init():
            method, layer = Fp8LinearMethod(qc), nn.Module()
            method.create_weights(layer, input_size_per_partition=K, output_partition_sizes=list(partitions),
                                  input_size=K, output_size=N, params_dtype=torch.bfloat16, weight_loader=None)
        layer.weight.data.copy_(wq)
        layer.weight_scale_inv.data.copy_(ws)
        method.process_weights_after_loading(layer)
        return method, layer

    m_gu, l_gu = linear(w_gu, (INTERMEDIATE, INTERMEDIATE))
    m_d, l_d = linear(w_d, (HIDDEN,))
    act = SiluAndMul()

    def fn(x_bf):
        gu = m_gu.apply(l_gu, x_bf)
        return m_d.apply(l_d, act(gu))

    name = m_gu.w8a8_block_fp8_linear.__name__
    sc = l_gu.weight_scale_inv
    fn.info = {"backend": _sgl_backend(name), "linear": f"sglang Fp8LinearMethod -> {name}",
               "fp8_gemm_runner_backend": ctx["fp8_gemm_runner_backend"],
               "weight_scale": ("UE8M0, requantized at load for DeepGEMM" if getattr(sc, "format_ue8m0", False)
                                else "fp32") + f" ({_scale_desc(sc)})",
               "act": f"sglang SiluAndMul.{act.dispatch_forward().__name__}",
               "deepgemm_precompile": os.environ.get("SGLANG_JIT_DEEPGEMM_PRECOMPILE"),
               "versions": _versions("sglang", "sglang-kernel", "sgl-deep-gemm", "flashinfer-python")}
    if fn.info["backend"] == "Triton":   # not the case on sm_90 / sm_100 / sm_120; stated if it ever is
        fn.info["config_source"] = _triton_config_source("sglang.kernels.ops.quantization.fp8_kernel")
    return fn


def _triton_config_source(module):
    """Whether a library's block-FP8 Triton GEMM finds a tuned config file for the two projections on this device."""
    try:
        import importlib
        get = importlib.import_module(module).get_w8a8_block_fp8_configs
        return {f"{N}x{K}": ("tuned json" if get(N, K, 128, 128) else "default")
                for N, K in ((2 * INTERMEDIATE, HIDDEN), (HIDDEN, INTERMEDIATE))}
    except Exception as e:
        return f"unknown ({_err(e, 120)})"


_VLLM = {}


def _vllm_init():
    """A vLLM config for the block-FP8 Qwen3-4B checkpoint, entered once for the process. Compilation mode NONE
    makes `custom_ops` default to "all", so QuantFP8 and SiluAndMul dispatch to their CUDA ops (vLLM's eager-mode
    kernels); under the default -O2 compile vLLM would instead fuse silu_and_mul with the down projection's group
    quantize (`silu_and_mul_per_block_quant`), which this cell does not model: it pays one kernel and one BF16
    round trip of the M x 9728 intermediate more than that compiled graph."""
    if _VLLM:
        return _VLLM
    try:
        from vllm.config import CompilationConfig, ModelConfig, VllmConfig, set_current_vllm_config
        from vllm.config.compilation import CompilationMode
    except Exception as e:
        raise NotAvailable(f"vllm does not import here ({_err(e, 160)})")
    d = _model_dir()
    mc = ModelConfig(model=d, tokenizer=d, skip_tokenizer_init=True, dtype="bfloat16")
    vc = VllmConfig(model_config=mc, compilation_config=CompilationConfig(mode=CompilationMode.NONE))
    ctx = set_current_vllm_config(vc)
    ctx.__enter__()                      # held for the life of the worker process
    # vLLM's weight parameters record their tensor-parallel rank, so the world=1 groups must exist (gloo, one rank,
    # a free port, as vLLM's own single-process layer tests set them up); nothing is communicated.
    import socket
    from vllm.distributed import parallel_state as ps
    if not ps.model_parallel_is_initialized():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        ps.init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                        distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo")
        ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    _VLLM.update(config=vc, ctx=ctx)
    return _VLLM


def _build_vllm_fp8b(w_gu, w_d, sm_major):
    """vLLM's dense block-FP8 MLP: Fp8LinearMethod for both projections (the block-FP8 kernel its
    init_fp8_linear_kernel selects on this device, with that kernel's activation quantize) and SiluAndMul."""
    v = _vllm_init()
    import torch.nn as nn
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
    qc = Fp8Config.from_config(dict(FP8_BLOCK_QCFG))

    def linear(w, partitions):
        N, K = w.shape
        wq, ws = _block_quant_128x128(w)
        with _model_init():
            method, layer = Fp8LinearMethod(qc), nn.Module()
            method.create_weights(layer, input_size_per_partition=K, output_partition_sizes=list(partitions),
                                  input_size=K, output_size=N, params_dtype=torch.bfloat16,
                                  weight_loader=lambda *a, **k: None)
        layer.weight.data.copy_(wq)
        layer.weight_scale_inv.data.copy_(ws)
        method.process_weights_after_loading(layer)
        return method, layer

    m_gu, l_gu = linear(w_gu, (INTERMEDIATE, INTERMEDIATE))
    m_d, l_d = linear(w_d, (HIDDEN,))
    act = SiluAndMul()

    def fn(x_bf):
        gu = m_gu.apply(l_gu, x_bf)
        return m_d.apply(l_d, act(gu))

    kern = m_gu.fp8_linear
    kname = type(kern).__name__
    label = next((lab for key, lab in (("FlashInferFp8DeepGEMM", "FlashInfer/DeepGEMM"), ("DeepGemm", "DeepGEMM"),
                                        ("FlashInfer", "FlashInfer"), ("Cutlass", "CUTLASS"), ("B12x", "B12X"),
                                        ("Marlin", "Marlin"), ("Humming", "Humming"), ("Triton", "Triton"),
                                        ("Torch", "torch scaled_mm")) if key in kname), kname)
    quant = getattr(kern, "quant_fp8", None)
    fn.info = {"backend": label, "linear": f"vLLM Fp8LinearMethod -> {kname}",
               "act_quant": (f"QuantFP8.{quant._forward_method.__name__}" + (" (UE8M0 scales)" if quant.use_ue8m0 else "")
                             if quant is not None else "inside the GEMM op"),
               "weight_scale": _scale_desc(l_gu.weight_scale_inv),
               "act": f"vLLM SiluAndMul.{act._forward_method.__name__}",
               "custom_ops": list(v["config"].compilation_config.custom_ops),
               "versions": _versions("vllm", "flashinfer-python", "deep-gemm", "deep_gemm")}
    if "DeepGEMM" in label:             # vLLM imports deep_gemm from site-packages, else its vendored copy
        dg = sys.modules.get("deep_gemm") or sys.modules.get("vllm.third_party.deep_gemm")
        fn.info["deep_gemm"] = f"{getattr(dg, '__name__', '?')} {getattr(dg, '__version__', '')} {getattr(dg, '__file__', '')}"
    if label == "Triton":
        fn.info["config_source"] = _triton_config_source("vllm.model_executor.layers.quantization.utils.fp8_utils")
    return fn


# cuBLASLt has no block-scaled FP8 algorithm when the row count is not a multiple of 4 (its heuristic returns
# CUBLAS_STATUS_NOT_SUPPORTED for M = 1 on sm_90): the activation scales' leading dimension must be 16-byte aligned.
CUBLAS_ROW_ALIGN = 4


def _quant_1x128_fp32(t, rows):
    """BF16 [M, K] -> (FP8 [rows, K], fp32 scales [K/128, rows]) for the cublas_fp8b cell (compiled by Inductor),
    the rows beyond M zero. The scales come back K-block-major so that their transpose is the (rows, K/128)
    operand with strides (1, rows) that scaled_mm's BlockWise1x128 recipe requires."""
    M, K = t.shape
    if rows > M:
        t = F.pad(t, (0, 0, 0, rows - M))
    g = t.float().view(rows, K // 128, 128)
    s = g.abs().amax(dim=-1).clamp(min=1e-10) / FP8_MAX
    q = (g / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(rows, K)
    return q, s.t().contiguous()


_QUANT_1x128 = []


def _build_cublas_fp8b(w_gu, w_d, sm_major):
    """cuBLAS block-FP8 via torch: F.scaled_mm with BlockWise1x128 activation scales and BlockWise128x128
    weight scales (fp32), the activation quantize and silu*mul each a torch.compile'd function."""
    from torch._C import _ScalingType as ST
    if not _QUANT_1x128:
        _QUANT_1x128.append(torch.compile(_quant_1x128_fp32, mode="default", dynamic=False))
    quant = _QUANT_1x128[0]
    wgu_q, sgu = _block_quant_128x128(w_gu)
    wd_q, sd = _block_quant_128x128(w_d)
    wgu_t, sgu_t, wd_t, sd_t = wgu_q.t(), sgu.t(), wd_q.t(), sd.t()    # (K, N) column-major, (K/128, N/128)

    def mm(q, s_t, w_t, sw_t):
        # The (M, K/128) operand over the K-block-major scales, with the strides (1, M) scaled_mm checks literally;
        # spelled out because Inductor may hand back a size-1 dim with any stride (at M = 1 the transpose of its
        # (K/128, 1) output came back as (20, 1) strides and was refused).
        kb, m = s_t.shape
        return F.scaled_mm(q, w_t, s_t.as_strided((m, kb), (1, m)), ST.BlockWise1x128, sw_t, ST.BlockWise128x128,
                           output_dtype=torch.bfloat16)

    try:   # does this torch / cuBLAS take the recipe on this device at all?
        qp, sp = _quant_1x128_fp32(torch.zeros(CUBLAS_ROW_ALIGN, w_d.shape[1], dtype=torch.bfloat16, device="cuda"),
                                   CUBLAS_ROW_ALIGN)
        mm(qp, sp, wd_t, sd_t)
        torch.cuda.synchronize()
    except Exception as e:
        cap = torch.cuda.get_device_capability(0)
        raise NotAvailable(f"torch {torch.__version__} / cuBLAS rejects BlockWise1x128 x BlockWise128x128 "
                           f"scaled_mm on sm_{cap[0]}{cap[1]}: {_err(e, 220)}")

    def fn(x_bf):
        M = x_bf.shape[0]
        rows = -(-M // CUBLAS_ROW_ALIGN) * CUBLAS_ROW_ALIGN    # zero rows ride through both GEMMs and are dropped
        xq, sx_t = quant(x_bf, rows)
        gu = mm(xq, sx_t, wgu_t, sgu_t)
        h = _silu_chunk_mul(gu)
        hq, sh_t = quant(h, rows)
        y = mm(hq, sh_t, wd_t, sd_t)
        return y if rows == M else y[:M]

    fn.info = {"backend": "cuBLASLt", "linear": "torch.nn.functional.scaled_mm BlockWise1x128 x BlockWise128x128",
               "act_quant": f"torch.compile (Inductor) 1x128 fp32-scale quantize, rows zero-padded to a multiple of "
                            f"{CUBLAS_ROW_ALIGN} inside the graph", "act": "torch.compile silu*mul",
               "versions": _versions("nvidia-cublas")}
    return fn


_CMP_BUILDERS = {"sgl_fp8b": _build_sgl_fp8b, "vllm_fp8b": _build_vllm_fp8b, "cublas_fp8b": _build_cublas_fp8b}


def _trace_kernels(fn, x_bf):
    """The CUDA kernels (and copies) of one eager call, in launch order (FSO_BENCH_TRACE_KERNELS=1; smoke
    evidence only)."""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(x_bf)
        torch.cuda.synchronize()
    evs = sorted((e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
                 key=lambda e: e.time_range.start)
    return [e.name[:200] for e in evs]


def _busy_warm():
    """Optional DVFS settle before a cell's timing (same knob as
    bench_qwen3_4b_mlp.py / bench_moe_*.py): on machines without a clock
    lock (B300: idles at 120 MHz between subprocess cells, ramps to the
    flat 2032 MHz max under load) the 15-iteration eager warmup of a
    small-M cell is a few hundred µs of GPU work and does not finish the
    ramp. FSO_BENCH_WARM_MS=<ms> spins a dummy matmul for that long first.
    Default 0 → protocol identical to the locked-clock machines."""
    import time
    ms = int(os.environ.get("FSO_BENCH_WARM_MS", "0"))
    if ms <= 0:
        return
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    t0 = time.monotonic()
    while (time.monotonic() - t0) * 1000.0 < ms:
        a = a @ a * 1e-3  # keep values bounded; result reused to defeat DCE
    torch.cuda.synchronize()


def _time_graph(call, iters=50, warmup=15, repeats=3):
    _busy_warm()
    # Eager warmup (sets static cudaFuncSetAttribute guards, Params cache,
    # Stream-K pool — all must be hot before stream capture starts).
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        call()

    samples = []
    for _ in range(repeats):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        samples.append(e0.elapsed_time(e1) / iters)
    return statistics.median(samples)


def bench_worker(M, dtypes=None):
    """Run one M cell over the default dtypes, or over `dtypes` (--dtypes). Designed for subprocess use."""
    sm_major = torch.cuda.get_device_capability(0)[0]
    device = "cuda"
    torch.manual_seed(M * 1009)
    x_bf = (torch.randn(M, HIDDEN, dtype=torch.bfloat16, device=device) * 0.1).contiguous()

    out = {"M": M, "hidden": HIDDEN, "intermediate": INTERMEDIATE}

    # BF16 reference for cosine check
    fn_bf = _build_mlp_fn("bf16", sm_major)
    y_ref = fn_bf(x_bf)

    # CUDA-Graph capture-and-replay is the only metric we report. Eager
    # is dropped because it conflates kernel cost with PyTorch op dispatch
    # + cudaLaunchKernelEx overhead — both elided in production by graph
    # capture. torch.compile is also dropped: on top of cudagraph it adds
    # at most a few percent (Inductor can fuse silu·gate and pick a
    # Triton matmul autotune), and that win does not change kernel-side
    # tuning decisions, which is what this bench is meant to inform.
    # Cold-L2 protocol (2026-09-28): the captured graph runs the block over R
    # independent weight copies (`_build_mlp_fn` seeds the same values into fresh
    # memory each time, quantized forms included), R * weight bytes >= 2 x L2,
    # and reports replay time / R -- each copy's weights are evicted by the
    # others before the graph returns to them, as a serving step evicts every
    # layer's weights. One warm copy served the M <= 128 rows from L2.
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    for dtype in (dtypes or FSO_DTYPES):
        cmp = dtype in CMP_DTYPES
        if cmp:
            # A comparator that cannot be built here (library missing, recipe rejected by this device) is an n/a
            # cell with its reason; any other failure while building it is an error cell. Neither stops the row.
            try:
                fn = _build_mlp_fn(dtype, sm_major)
            except NotAvailable as e:
                out[dtype] = {"error": f"n/a: {' '.join(str(e).split())[:300]}"}
                continue
            except Exception as e:
                out[dtype] = {"error": _err(e)}
                continue
            record = dict(fn.info)
        else:
            fn = _build_mlp_fn(dtype, sm_major)
            if fn is None:
                continue
            record = {}
        try:
            y = fn(x_bf)
            record["cos"] = float(F.cosine_similarity(y.float().flatten(), y_ref.float().flatten(), dim=0).item())
            if cmp and os.environ.get("FSO_BENCH_TRACE_KERNELS") == "1":
                record["kernels"] = _trace_kernels(fn, x_bf)
            wbytes = (2 * INTERMEDIATE * HIDDEN + HIDDEN * INTERMEDIATE) * (2 if dtype == "bf16" else 1)
            mult = float(os.environ.get("FSO_BENCH_L2_MULT", "2"))
            R = max(2, min(16, int(math.ceil(mult * l2 / wbytes))))
            if os.environ.get("FSO_BENCH_WEIGHT_COPIES"):
                R = max(1, int(os.environ["FSO_BENCH_WEIGHT_COPIES"]))
            fns = [fn] + [_build_mlp_fn(dtype, sm_major) for _ in range(R - 1)]

            def call_all(fns=fns):
                for f in fns:
                    f(x_bf)
            record["weight_copies"] = R
            record["graph_us"] = _time_graph(call_all) * 1000.0 / R
        except Exception as e:
            record["error"] = _err(e) if cmp else f"{type(e).__name__}: {str(e)[:120]}"
        out[dtype] = record

    return out


def _worker_record(stdout, M):
    """The worker's result: the last stdout line that is a JSON row for this M (a library may log to stdout)."""
    for line in reversed(stdout.decode(errors="replace").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and rec.get("M") == M:
                return rec
    return None


def run_grid(out_path, dtypes=None):
    """One worker subprocess per M. A run of comparators only (every --dtypes name in CMP_DTYPES) records a
    worker that dies as an error cell of each dtype and goes on; a run with an fso dtype stops at it, as before."""
    import subprocess
    sm_major = torch.cuda.get_device_capability(0)[0]
    name = torch.cuda.get_device_name(0)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        f = open(out_path, "w")
    else:
        f = sys.stdout
    sm_cap = torch.cuda.get_device_capability(0)
    meta = {"_device": name, "_sm": sm_cap[0] * 10 + sm_cap[1]}
    if dtypes:
        meta.update(dtypes=list(dtypes), torch=torch.__version__)
    f.write(json.dumps(meta) + "\n")
    f.flush()
    strict = not dtypes or not set(dtypes) <= set(CMP_DTYPES)
    worker = [sys.executable, __file__, "--worker"] + (["--dtypes", ",".join(dtypes)] if dtypes else [])

    total = len(M_GRID)
    t_start = __import__("time").time()
    for i, M in enumerate(M_GRID):
        result = subprocess.run(
            worker,
            input=str(M).encode(),
            capture_output=True,
            check=strict,
        )
        rec = _worker_record(result.stdout, M)
        if rec is None:
            tail = (result.stderr.decode(errors="replace").strip().splitlines() or [""])[-1]
            if strict:
                raise RuntimeError(f"M={M}: the worker printed no result row (exit {result.returncode}): {tail[:300]}")
            err = f"worker exited {result.returncode} without a result: {' '.join(tail.split())[:300]}"
            rec = {"M": M, "hidden": HIDDEN, "intermediate": INTERMEDIATE, **{d: {"error": err} for d in dtypes}}
        f.write(json.dumps(rec) + "\n")
        f.flush()
        elapsed = __import__("time").time() - t_start
        eta = elapsed / (i + 1) * (total - i - 1)
        cells = [(k, rec[k]) for k in (dtypes or FSO_DTYPES) if k in rec and "graph_us" in rec[k]]
        summary = "  ".join(
            f"{k}: graph={v['graph_us']:.1f}" for k, v in cells
        )
        if dtypes:   # comparator cells also say what they are, or why they are missing
            summary += "".join(
                f"  {k}: cos={rec[k]['cos']:.4f} backend={rec[k].get('backend')}" if "graph_us" in rec.get(k, {})
                else f"  {k}=n/a ({rec[k]['error'][5:90]})" if rec.get(k, {}).get("error", "").startswith("n/a: ")
                else f"  {k}=ERR ({rec.get(k, {}).get('error', 'no record')[:90]})"
                for k in dtypes if k in CMP_DTYPES)
        print(f"[{i+1:>2}/{total}] M={M:>4}  {summary}  (t={elapsed:.0f}s eta={eta:.0f}s)",
              flush=True)
    if out_path:
        f.close()


def render_md(jsonl_path, label):
    recs = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    meta = recs[0] if "_device" in recs[0] else {"_device": "?", "_sm": "?"}
    rows = [r for r in recs if "M" in r]

    lines = []
    lines.append(f"### {label} — {meta['_device']} (sm_{meta['_sm']})")
    lines.append("")
    has_mx       = any("mxfp8"    in r and "graph_us" in r.get("mxfp8",    {}) for r in rows)
    has_smm      = any("smm"      in r and "graph_us" in r.get("smm",      {}) for r in rows)
    has_smm_fast = any("smm_fast" in r and "graph_us" in r.get("smm_fast", {}) for r in rows)

    # Single `µs` column per dtype = CUDA-Graph capture-and-replay timing
    # for the full SwiGLU MLP forward (5 ops for FP8 paths, 2 for BF16).
    # `sMM` is the cuBLAS MXFP8 path with reference `to_mxfp + to_blocked`
    # quantize. `sMM-c` is the same path with the quantize wrapped in
    # `torch.compile(mode="default")` so Inductor fuses it into one kernel.
    cols_us  = ["BF16 µs", "BSFP8 µs"]
    cos_keys = [("BSFP8 cos", "bsfp8")]
    if has_mx:
        cols_us.append("MXFP8 µs");  cos_keys.append(("MXFP8 cos", "mxfp8"))
    if has_smm:
        cols_us.append("sMM µs");    cos_keys.append(("sMM cos", "smm"))
    if has_smm_fast:
        cols_us.append("sMM-c µs");  cos_keys.append(("sMM-c cos", "smm_fast"))
    head_cells = ["M"] + cols_us + [c for c, _ in cos_keys]
    sep_cells  = ["---:"] * len(head_cells)
    lines.append("| " + " | ".join(f"{c:>7}" for c in head_cells) + " |")
    lines.append("| " + " | ".join(f"{c:>7}" for c in sep_cells)  + " |")

    def cell(r, k, field):
        v = r.get(k, {}).get(field)
        if not isinstance(v, float):
            return "—"
        return f"{v:.4f}" if field == "cos" else f"{v:.2f}"

    for r in sorted(rows, key=lambda x: x["M"]):
        M = r["M"]
        parts = [f"{M:>5}"]
        parts += [f"{cell(r, 'bf16',  'graph_us'):>7}",
                  f"{cell(r, 'bsfp8', 'graph_us'):>8}"]
        if has_mx:       parts.append(f"{cell(r, 'mxfp8',    'graph_us'):>8}")
        if has_smm:      parts.append(f"{cell(r, 'smm',      'graph_us'):>8}")
        if has_smm_fast: parts.append(f"{cell(r, 'smm_fast', 'graph_us'):>8}")
        for _, k in cos_keys:
            parts.append(f"{cell(r, k, 'cos'):>9}")
        lines.append("| " + " | ".join(parts) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--out")
    p.add_argument("--format", action="store_true")
    p.add_argument("--from", dest="srcs", nargs="+",
                   help="LABEL=jsonl pairs for format mode")
    p.add_argument("--md-out", default="-")
    p.add_argument("--Ms", type=str, default=None,
                   help="comma list of M values that replaces the M grid, e.g. 1,64 (with --run)")
    p.add_argument("--dtypes", type=str, default=None,
                   help=f"comma list of the dtypes to measure, from {','.join(DTYPES)} (default: "
                        f"{','.join(FSO_DTYPES)}); the comparators run only when named here")
    args = p.parse_args()
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()] if args.dtypes else None
    if dtypes is not None and (not dtypes or set(dtypes) - set(DTYPES)):
        p.error(f"--dtypes: unknown {sorted(set(dtypes) - set(DTYPES)) or 'empty list'}; known: {','.join(DTYPES)}")
    # Run by an absolute interpreter path, a venv's console scripts are not on PATH the way activation puts them,
    # and the sglang / deep_gemm imports shell out to `ninja` (bench_moe_qwen3_30a3.py does the same).
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

    if args.worker:
        M = int(sys.stdin.read().strip())
        rec = bench_worker(M, dtypes)
        sys.stdout.write(json.dumps(rec) + "\n")
        return

    if args.run:
        if args.Ms:
            set_m_grid([int(m) for m in args.Ms.split(",")])
        run_grid(args.out, dtypes)
        return

    if args.format:
        out = ""
        for src in args.srcs or []:
            label, path = src.split("=", 1)
            out += render_md(path, label) + "\n"
        if args.md_out == "-":
            sys.stdout.write(out)
        else:
            with open(args.md_out, "w") as f:
                f.write(out)
        return

    p.error("specify --run or --format or --worker")


if __name__ == "__main__":
    main()
