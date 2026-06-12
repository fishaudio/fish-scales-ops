#!/usr/bin/env bash
# Run as: sudo bash bench/prof/run_5090_ncu.sh
# Profiles the differentiated sm_120 kernels (MXFP8 GEMM + attention prefill)
# with ncu --set full, then prints a compact SOL / pipe / occupancy / stall
# summary for each. Paste the whole "================ SUMMARY" block back.
set -u
cd /home/stone/workspace/standalone/fish-scales-ops
export PYTHONPATH=python
PY=/home/stone/workspace/standalone/.dev/bin/python
NCU=/tmp/cu13_extract/nsight_compute/ncu
GPU=${GPU:-0}                      # idle, clock-locked GPU to profile on
OUT=/tmp/fso_ncu; mkdir -p $OUT

# label | driver-args | ncu skip | ncu count
run() {
  local label="$1"; shift
  local drv="$1"; shift
  local skip="$1"; shift
  local cnt="$1"; shift
  echo ">>> profiling $label ..."
  CUDA_VISIBLE_DEVICES=$GPU $NCU --target-processes all -c "$cnt" -s "$skip" \
      --set full -o "$OUT/$label" --force-overwrite \
      $PY $drv >/dev/null 2>"$OUT/$label.err" \
    && echo "    ok" || { echo "    FAILED:"; tail -3 "$OUT/$label.err"; }
}

# compute-bound MXFP8 GEMM (peak regime) — "~70% of peak" claim
run gemm_cubic4096 "bench/prof/prof_gemm.py mxfp8 4096 4096 4096 30" 30 2
# prefill-shape MXFP8 GEMM (gate_up M=512, streamk regime)
run gemm_gateup_m512 "bench/prof/prof_gemm.py mxfp8 512 19456 2560 30" 30 2
# attention prefill D=128 S=4096 causal — the "40% mma-peak ceiling" claim
run attn_d128_s4096_causal "bench/prof/prof_attn.py 1 4096 32 8 128 1 30" 20 2
# attention prefill D=128 S=4096 non-causal
run attn_d128_s4096_noncausal "bench/prof/prof_attn.py 1 4096 32 8 128 0 30" 20 2

echo
echo "================ SUMMARY ================"
for rep in $OUT/*.ncu-rep; do
  [ -e "$rep" ] || continue
  label=$(basename "$rep" .ncu-rep)
  echo
  echo "######## $label ########"
  # pick the single longest-duration kernel in the report (the GEMM/attn kernel)
  $NCU -i "$rep" --page details \
     --section SpeedOfLight --section Occupancy \
     --section ComputeWorkloadAnalysis --section WarpStateStats 2>/dev/null \
   | grep -iE "void |Duration|Compute \(SM\)|Memory Throughput|DRAM Throughput|L1/TEX|L2 Cache|Achieved Occupancy|Theoretical Occupancy|Registers Per Thread|Shared Memory|Block Limit|highest-utilized|Tensor|FMA|ALU|LSU|Issue Slots Busy|Issued Ipc|Warp Cycles Per Issued|stall|Stall" \
   | head -60
done
echo "================ END ================"
