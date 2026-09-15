#!/usr/bin/env bash
# Phase 5 JIT-isolation lint — fails if any AOT code reaches into the
# deep_gemm JIT subtree (`arch/sm90/fp8/jit/`) other than the dedicated
# sm_90 entry point. See arch/sm90/fp8/jit/README.md.
#
# Run from anywhere:
#   bash tests/gemm/unit/lint_jit_isolation.sh

set -euo pipefail

cd "$(dirname "$0")/../../.."

# Files that are allowed to include from the JIT subtree:
#   - anything inside arch/sm90/fp8/jit/ itself (sibling references)
#   - arch/sm90/fp8/dispatch.cuh, the single sm_90 entry point (dense and grouped)
ALLOW='^csrc/gemm/include/blockscale_gemm/arch/sm90/fp8/(jit/|dispatch\.cuh)'

violations=$(
  grep -rn '#include.*\(deep_gemm/\|arch/sm90/fp8/jit/\)' \
    csrc/ 2>/dev/null \
  | grep -Ev "$ALLOW" \
  || true
)

if [[ -n "$violations" ]]; then
  echo "JIT-isolation violations:" >&2
  echo "$violations" >&2
  exit 1
fi

echo "JIT isolation OK"
