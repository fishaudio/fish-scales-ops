"""The tables of record, per machine: every bench invocation that bench/run_perf.py runs.

Each step is one bench invocation in one environment of the machine's lock (bench/env/<machine>.lock.json):

  name     a short label, unique per machine; the step's log is logs/<name>.log in the run directory
  env      the environment of the lock whose interpreter runs the step
  script   the bench script, relative to the tree root
  args     the bench arguments; run_perf.py appends `--out <run_dir>/<out>`, and `--Ms 1,64` for a smoke run
  out      the raw file the step writes, named as bench/gemm/python/perf_report.py's manifest expects
  group    the table group: dense (the dense GEMM and MLP-block tables), moe (the fso rows of the MoE layer and
           grouped-kernel tables), moe_ref (layer-level comparators), moe_kern (kernel-level comparators),
           shared (the routed + shared-expert block of Family C, its fso rows and its comparators), mlp_ref (the
           serving-library comparators of the Family A MLP block)

The steps encode the chains that produced the 2026-10-01 tables, with the same benches, implementations, grids
(the benches' defaults) and file names; nothing here is a new invocation:

  h200  /data/bench-runs/fso_release_perf_20261001/h200/scripts/run_stage1.sh (every fso row), then the two
        comparator steps of /data/bench-runs/h200_moe_tables_20261001/scripts/run_chain.sh (sglang Triton BF16 and
        FP8 block, deep_gemm masked pipeline); that chain's three fso steps are the same invocations as the stage-1
        ones and are not repeated.
  5090  /data/bench-runs/fso_release_perf_20261001/5090_stage2/chain.sh, which ran on two cards at once: card 21
        took Family B and the dense tables, card 22 Family C and the routed + shared block. The lock names one
        card for every table, so the two halves run one after the other: card 21's steps, then card 22's, with the
        four max-autotune comparator steps of both halves moved to the end. The chain kept them last on each card
        so that their CPU-heavy Inductor compile could not overlap the eager-timed torch cells, and the same order
        keeps that true here. The ref_kern_c1_* names are the manifest's patterns, not a card.
  b300  /data/bench-runs/fso_release_perf_20261001/b300/run_stage1.sh (fso MoE rows), run_dense.sh (the dense
        tables, with the expecttest overlay that the lock now gives every step) and stage2/run_stage2.sh (the
        torch-native comparators), one pass. The 2026-10-01 tables of record took pass 1 of two.

The mlp_ref steps at the end of the file are new invocations, added for the 0.2.0 tables (2026-10-05): the
serving libraries' block-FP8 dense linear (sglang on every machine, vLLM on the RTX 5090) and, on the H200 only,
cuBLAS's block-FP8 scaled_mm, each driving the same Family A MLP block as the fso BSFP8 cell, in the environment
where that library lives.
"""

GROUPS = ("dense", "moe", "moe_ref", "moe_kern", "shared", "mlp_ref")

B = "bench/gemm/python"
DENSE = B + "/bench_qwen3_4b_mlp.py"
MLP = B + "/bench_qwen3_4b_mlp_forward.py"
MOE_B = B + "/bench_moe_qwen3_30a3.py"
MOE_C = B + "/bench_moe_qwen3_35a3.py"
MOE = {"30a3": MOE_B, "35a3": MOE_C}


def step(name, env, script, args, out, group):
    assert group in GROUPS, group
    return {"name": name, "env": env, "script": script, "args": list(args), "out": out, "group": group}


def dense_steps(sm, env="main"):
    """The four dense files of one device: Family A per projection, the Family A MLP block, and the dense
    projections of Families B and C. The per-machine lists put them in the order of that machine's chain."""
    return {
        "A": step("dense_A", env, DENSE, ["--run", "--family", "qwen3-4b"], f"gemm_sm{sm}_qwen3_4b.jsonl", "dense"),
        "mlp": step("mlp_fwd", env, MLP, ["--run"], f"gemm_sm{sm}_qwen3_4b_mlp_fwd.jsonl", "dense"),
        "B": step("dense_B", env, DENSE, ["--run", "--family", "qwen3-30a3"], f"gemm_sm{sm}_qwen3_30a3_dense.jsonl", "dense"),
        "C": step("dense_C", env, DENSE, ["--run", "--family", "qwen3.5-35a3"], f"gemm_sm{sm}_qwen3_35a3_dense.jsonl", "dense"),
    }


# ----------------------------------------------------------------------------- H200 (sm_90)
_h200_dense = dense_steps(90)
H200 = [
    # run_stage1.sh: dense tables in its order (A, MLP block, B, C), then the fso MoE rows.
    _h200_dense["A"], _h200_dense["mlp"], _h200_dense["B"], _h200_dense["C"],
    step("fso_30a3", "main", MOE_B, ["--run", "--impls", "fso_bsfp8_layer"], "perf_moe_qwen3_30a3_h200.jsonl", "moe"),
    step("fso_35a3", "main", MOE_C, ["--run", "--impls", "fso_bsfp8_layer"], "perf_moe_qwen3_35a3_h200.jsonl", "moe"),
    step("fso_35a3_shared", "main", MOE_C, ["--run", "--impls", "fso_bsfp8_layer_shared"],
         "perf_moe_qwen3_35a3_shared_h200.jsonl", "shared"),
    # run_chain.sh: the same-day layer comparators (sglang 0.5.20 Triton BF16 / FP8 block, deep_gemm masked pipeline).
    step("ref_30a3", "main", MOE_B, ["--run", "--impls", "triton_bf16,triton_fp8b,dg_fp8_layer"], "ref_new_30a3_h200.jsonl", "moe_ref"),
    step("ref_35a3", "main", MOE_C, ["--run", "--impls", "triton_bf16,triton_fp8b,dg_fp8_layer"], "ref_new_35a3_h200.jsonl", "moe_ref"),
]


# ----------------------------------------------------------------------------- RTX 5090 (sm_120)
def _moe_family_5090(f):
    """chain.sh moe_family(): the family's fso rows, layer comparators and kernel-level comparators."""
    s = MOE[f]
    return [
        step(f"fso_{f}", "main", s, ["--run", "--impls", "fso_mxfp8_grouped,fso_mxfp8_layer"], f"perf_moe_qwen3_{f}_5090.jsonl", "moe"),
        step(f"new_{f}", "main", s, ["--run", "--impls", "vllm_bf16,vllm_fp8b,torch_grouped_bf16_layer,torch_grouped_bf16_layer_compiled,"
                                     "torch_smm_mxfp8_layer,torch_smm_mxfp8_layer_compiled"], f"ref_new_{f}_5090.jsonl", "moe_ref"),
        step(f"trt_{f}", "main", s, ["--run", "--impls", "trtllm_cutlass_bf16,trtllm_cutlass_fp8"], f"ref_trt_{f}_5090.jsonl", "moe_ref"),
        step(f"sgl020_{f}", "sglang", s, ["--run", "--impls", "triton_bf16,triton_fp8b"], f"ref_sgl020_{f}_5090.jsonl", "moe_ref"),
        step(f"kern_vllm_{f}", "main", s, ["--run", "--impls", "vllm_triton_grouped_fp8b,vllm_triton_grouped_bf16", "--projs", "gate_up,down"],
             f"ref_kern_c1_vllm_{f}_5090.jsonl", "moe_kern"),
        step(f"kern_sgl_{f}", "sglang", s, ["--run", "--impls", "sgl_triton_grouped_fp8b,sgl_triton_grouped_bf16", "--projs", "gate_up,down"],
             f"ref_kern_c1_sgl_{f}_5090.jsonl", "moe_kern"),
        step(f"kern_fi_{f}", "fi", s, ["--run", "--impls", "fi_cudnn_grouped_mxfp8,fi_cudnn_grouped_bf16", "--projs", "gate_up,down"],
             f"ref_kern_c1_fi_{f}_5090.jsonl", "moe_kern"),
    ]


def _torch_ma_5090(f):
    """chain.sh torch_ma(): the max-autotune comparators, last."""
    s = MOE[f]
    return [
        step(f"torchma_{f}", "main", s, ["--run", "--impls", "torch_grouped_bf16_layer_maxautotune,torch_smm_mxfp8_layer_maxautotune"],
             f"ref_torchma_{f}_5090.jsonl", "moe_ref"),
        step(f"torchma_smm_{f}", "main", s, ["--run", "--impls", "torch_smm_mxfp8_layer_maxautotune"],
             f"ref_torchma_smm_{f}_5090.jsonl", "moe_ref"),
    ]


_5090_dense = dense_steps(120)
RTX5090 = (
    # card 21's half of chain.sh: Family B, then the dense tables (A, B, C, then the MLP block)
    _moe_family_5090("30a3")
    + [_5090_dense["A"], _5090_dense["B"], _5090_dense["C"], _5090_dense["mlp"]]
    # card 22's half: Family C and the routed + shared block
    + _moe_family_5090("35a3")
    + [step("shared_35a3", "main", MOE_C, ["--run", "--impls", "fso_mxfp8_layer_shared"], "perf_moe_qwen3_35a3_shared_5090.jsonl", "shared")]
    # both halves' max-autotune comparators, last
    + _torch_ma_5090("30a3") + _torch_ma_5090("35a3")
)


# ----------------------------------------------------------------------------- B300 (sm_103; tables tagged sm100)
_b300_dense = dense_steps(100)
B300 = [
    # run_stage1.sh: the fso MoE rows
    step("fso_30a3", "main", MOE_B, ["--run", "--impls", "fso_mxfp8_grouped,fso_mxfp8_layer"], "perf_moe_qwen3_30a3_b300.jsonl", "moe"),
    step("fso_35a3", "main", MOE_C, ["--run", "--impls", "fso_mxfp8_grouped,fso_mxfp8_layer"], "perf_moe_qwen3_35a3_b300.jsonl", "moe"),
    step("fso_35a3_shared", "main", MOE_C, ["--run", "--impls", "fso_mxfp8_layer_shared"], "perf_moe_qwen3_35a3_shared_b300.jsonl", "shared"),
    # run_dense.sh: the dense tables (A, B, C, then the MLP block)
    _b300_dense["A"], _b300_dense["B"], _b300_dense["C"], _b300_dense["mlp"],
    # stage2/run_stage2.sh: the torch-native comparators of the routed layer and of the routed + shared block
    step("ref_30a3", "main", MOE_B, ["--run", "--impls", "torch_grouped_bf16_layer,torch_smm_mxfp8_layer"], "ref_new_30a3_b300.jsonl", "moe_ref"),
    step("ref_35a3", "main", MOE_C, ["--run", "--impls", "torch_grouped_bf16_layer,torch_smm_mxfp8_layer"], "ref_new_35a3_b300.jsonl", "moe_ref"),
    # the sglang 0.5.20 Triton fused_experts layers (BF16 and w8a8 block FP8), the backend apex uses on this card with
    # expert parallelism; the same env0520 venv (sglang 0.5.20), on the same card as the fso rows
    step("sgl020_30a3", "main", MOE_B, ["--run", "--impls", "triton_bf16,triton_fp8b"], "ref_sgl020_30a3_b300.jsonl", "moe_ref"),
    step("sgl020_35a3", "main", MOE_C, ["--run", "--impls", "triton_bf16,triton_fp8b"], "ref_sgl020_35a3_b300.jsonl", "moe_ref"),
    step("ref_35a3_shared", "main", MOE_C, ["--run", "--impls", "torch_grouped_bf16_layer_shared,torch_smm_mxfp8_layer_shared"],
         "ref_new_35a3_shared_b300.jsonl", "shared"),
]

# ----------------------------------------------------------------------------- Family A MLP-block comparators
# bench_qwen3_4b_mlp_forward.py --dtypes: the dense MLP block of the fso BSFP8 cell (gate_up -> silu*mul -> down,
# both activation quantizations in the graph, block-FP8 weights quantized at load, the same cold-weight rotation)
# run by sglang's Fp8LinearMethod + SiluAndMul (sgl_fp8b), vLLM's (vllm_fp8b) and torch's cuBLAS block-FP8
# scaled_mm (cublas_fp8b), each in the lock environment that has the library: sglang 0.5.20 is apex-0520 on the
# H200, env0520 on the B300 and the separate sglang venv on the RTX 5090; vLLM 0.29 only the RTX 5090's main venv.
# torch 2.13 takes the cuBLAS block recipe on sm_90 only: elsewhere it refuses with "DeepSeek style (1x128, 128x128)
# scaling only supported in CUDA for SM90" (the n/a of docs/perf/layer/sm{100,120}.md), so only the H200 has a cuBLAS
# step. perf_report.py merges the raw files into ref_mlp_qwen3_4b_<dev>.jsonl, which render_perf_docs.py joins to
# the Family A table by M.
def mlp_ref(lib, env, dtypes, dev):
    return step(f"mlp_ref_{lib}", env, MLP, ["--run", "--dtypes", dtypes], f"ref_mlp_{lib}_{dev}.jsonl", "mlp_ref")


H200 += [mlp_ref("sgl", "main", "sgl_fp8b", "h200"), mlp_ref("cublas", "main", "cublas_fp8b", "h200")]
RTX5090 += [mlp_ref("sgl", "sglang", "sgl_fp8b", "5090"), mlp_ref("vllm", "main", "vllm_fp8b", "5090")]
B300 += [mlp_ref("sgl", "main", "sgl_fp8b", "b300")]

SUITES = {"h200": H200, "5090": RTX5090, "b300": B300}
