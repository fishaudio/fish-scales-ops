#!/usr/bin/env python3
"""Qwen3.5-35B-A3B routed-MoE geometry (Family C) — front-end for
bench_moe_qwen3_30a3.py --model qwen3.5-35a3.

Geometry (Qwen3.5-35B-A3B-Base config.json): E=256 routed experts, top-8,
hidden=2048, moe_intermediate=512, 40 layers.
  gate_up : [8M, 2048] x [E, 1024, 2048] -> [8M, 1024]
  down    : [8M,  512] x [E, 2048,  512] -> [8M, 2048]
The shared expert (moe_intermediate 512, every token) is a dense GEMM pair and
is benched by bench_qwen3_4b_mlp.py --family qwen3.5-35a3 (shared_gate_up /
shared_down), not here. Same protocol, cells, impls and jsonl fields as the
30a3 script; see its docstring. Typical Family C fill on H200:

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python <python> \\
      bench/gemm/python/bench_moe_qwen3_35a3.py --run --impls fso_bsfp8_layer \\
      --Ms 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192 --out <jsonl>
"""
import os
import runpy
import sys

if "--model" not in sys.argv:
    sys.argv[1:1] = ["--model", "qwen3.5-35a3"]
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "bench_moe_qwen3_30a3.py"), run_name="__main__")
