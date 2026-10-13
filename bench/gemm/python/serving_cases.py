"""The serving-shaped MoE layer cases (issue #8 ask A): three routing families beside the uniform draw.

One definition, read by the MoE layer bench (bench_moe_qwen3_30a3.py: which cells a family has, which comparator
cells are n/a and why), by bench/perf_suite.py (the steps of the tables of record) and by render_perf_docs.py (the
"Serving-shaped cases" section of docs/perf/layer/sm{90,100,120}.md). Standard library only. docs/perf/README.md
("Routing") defines the families for readers; this file is where the numbers behind those words live.

  uniform  the draw of every table of record: each token takes TOPK distinct experts uniformly at random
           (bench_moe_qwen3_30a3.make_routing, seed M * 7919 + 3). It is not changed by anything here.
  skew     each token takes TOPK distinct experts, sampled without replacement in proportion to a Zipf popularity
           p_e ~ (rank_e + 1) ** -s over a seeded random ranking of the experts (seed M * 7919 + 3). SKEW_S holds
           s per family, calibrated on the CPU so that the hottest expert receives h = 3.8 times the rows of the
           uniform draw's hottest expert at the same M (geometric mean of the expectation over M = 1024, 2048,
           4096, 8192); issue #8 measured h = 3.4-4.3 on apex's prefill calls. h grows with M under any
           independent per-token draw, because the uniform maximum approaches its mean as M grows, so the
           achieved h is recorded per cell.
  padded   a CUDA-graph bucket of B executed rows of which the first L are live (PADDED_CELLS). The live rows carry
           exactly the uniform draw of M = L, so a padded cell is an A/B against the uniform cell at M = L; the
           B - L padded rows carry -1 in every top-k slot and a zero weight. -1 is the padding id sglang 0.5.20
           writes into the padded rows of a captured bucket when expert parallelism is on (topk.py,
           _mask_topk_ids_padded_region; forward_batch_info.py keeps num_token_non_padded only for moe_ep_size > 1);
           fso skips it, and any id outside [0, E), on every architecture.
  ep       the per-rank call of expert parallelism over EP_SIZE ranks: the uniform draw over the model's E experts
           (the same tokens as the uniform cell at that M), every id of an expert that rank 0 does not own
           (rank 0 owns [0, E / EP_SIZE)) mapped to -1, and a layer that holds the E / EP_SIZE local experts only.
"""

ROUTINGS = ("uniform", "skew", "padded", "ep")
EP_SIZE = 2

# The output-file tag of each serving-shaped family: perf_moe_qwen3_<fam>_<tag>_<dev>.jsonl (fso) and
# ref_moe_qwen3_<fam>_<tag>_<dev>.jsonl (comparators).
TAGS = {"skew": "skew", "padded": "padded", "ep": f"ep{EP_SIZE}"}

# Zipf exponent of the skewed draw per model (the bench's MODELS keys), calibrated 2026-10-10 on 16 seeds per M
# (/data/bench-runs/fso_0.2.1_20261010/S9/calib/calibration.md): expected h at M = 1024 / 2048 / 4096 / 8192 is
# 3.49 / 3.69 / 3.91 / 4.15 for Family B (E = 128) and 3.29 / 3.64 / 4.02 / 4.31 for Family C (E = 256).
SKEW_S = {"qwen3-30a3": 0.451, "qwen3.5-35a3": 0.386}
SKEW_TARGET_H = 3.8

# (bucket rows B, live rows L) of the padded family. The unpadded reference of a cell is the uniform table's row at
# M = L (the same live rows) and at M = B (a full bucket).
PADDED_CELLS = ((8, 1), (32, 1), (64, 1), (64, 16), (128, 1), (256, 1), (256, 8), (256, 64), (512, 1), (512, 128))

# fso's layer rows run in every family: ids outside [0, E_local) are skipped (docs/api/moe.md, Skipped ids).
FSO_IMPLS = ("fso_bsfp8_layer", "fso_mxfp8_layer")

# The comparators of the serving-shaped cases: the block-FP8 layer comparators of each machine's uniform table,
# each driven through the call a serving engine makes for the case. routing -> None when the comparator serves the
# case natively, else the reason its cell is n/a (recorded in the cell and printed under the table).
NA_TRITON_PADDED = (
    "sglang 0.5.20 writes -1 into padded rows only under expert parallelism, and only then does its Triton "
    "fused_experts skip -1 ids (filter_expert needs num_experts != num_local_experts); a single-rank sglang routes "
    "padded rows like live ones, and its Triton layer given -1 ids addresses expert -1 outside the weight tensor "
    "and faults with an illegal memory access")
NA_VLLM_EP = (
    "vLLM's expert-parallel call takes the global ids with an expert_map (fused_experts(global_num_experts=, "
    "expert_map=)), not ids already mapped to -1; this bench does not drive that form yet")
NA_VLLM_PADDED = (
    "vLLM writes no padding id into the padded rows of a captured bucket; its kernels expect ids in [0, E) without "
    "an expert_map")
COMPARATORS = {
    "triton_fp8b": {"skew": None, "ep": None, "padded": NA_TRITON_PADDED},
    "dg_fp8_layer": {"skew": None, "ep": None, "padded": None},
    "trtllm_gen_fp8b": {"skew": None, "ep": None, "padded": None},
    "vllm_fp8b": {"skew": None, "ep": NA_VLLM_EP, "padded": NA_VLLM_PADDED},
}
# Each machine's comparators, in column order (the block-FP8 layer comparators of its uniform table).
DEVICE_COMPARATORS = {"h200": ("triton_fp8b", "dg_fp8_layer"), "b300": ("trtllm_gen_fp8b", "triton_fp8b"),
                      "5090": ("triton_fp8b", "vllm_fp8b")}
# The deep_gemm masked pipeline sizes its slab from the bucket (m_max = (M // 256 + 1) * 256 rows per expert), and
# the uniform table runs it in the decode band only; the serving-shaped cases keep that band.
DECODE_ONLY = ("dg_fp8_layer",)

NA_OTHER = ("the serving-shaped cases run fso's layer and the block-FP8 serving comparators only "
            "(bench/gemm/python/serving_cases.py); BF16, per-tensor FP8 and torch-native layers are reference "
            "columns of the uniform tables")


def na_reason(impl, routing):
    """None when `impl` has cells under `routing`, else the reason its cells are n/a."""
    if routing == "uniform" or impl in FSO_IMPLS:
        return None
    if impl in COMPARATORS:
        return COMPARATORS[impl][routing]
    return NA_OTHER
