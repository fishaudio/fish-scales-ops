#!/usr/bin/env bash
# Stable build script for fish-scales-ops.
#
# Usage:
#   ./scripts/build.sh                                # full clean build, sm_90a + sm_120a
#   ARCH=12.0a ./scripts/build.sh                     # single-arch dev build (fastest)
#   CLEAN=0 ./scripts/build.sh                        # incremental, keep build/
#   EDITABLE=1 ./scripts/build.sh                     # pip install -e python/  (recommended for dev)
#   CUDA_HOME=/opt/cuda-13.0 ./scripts/build.sh       # pick a specific CUDA toolkit
#
# Environment (override anything you want; defaults try to be portable):
#
#   ARCH         TORCH_CUDA_ARCH_LIST value. Default "9.0a;12.0a".
#                For B200/B300 (sm_100/sm_103) add "10.0f" — the FAMILY
#                target (CUDA >= 12.9) so one cubin serves both. e.g.
#                  ARCH="10.0f" ./scripts/build.sh        # datacenter-Blackwell-only dev build
#                  ARCH="9.0a;10.0f;12.0a" ./scripts/build.sh
#   CLEAN        1 (default) → wipe python/build + stale .so before building.
#   EDITABLE     1 → pip install -e . --no-build-isolation
#                0 (default) → setup.py build_ext --inplace
#   MAX_JOBS     ninja parallelism. Default $(nproc). Lower if you OOM
#                (nvcc + CUTLASS templates eat ~2-4 GB per job).
#
#   CUDA_HOME    Path to the CUDA toolkit (must contain bin/nvcc). Auto-detect:
#                  1. CUDA_HOME env var (you set it)
#                  2. nvcc on PATH
#                  3. /usr/local/cuda
#                On hosts with multiple toolkits we DO NOT guess — set CUDA_HOME
#                yourself to keep the build reproducible.
#
#   CUTLASS_DIR  CUTLASS 4.x checkout root. Default <repo>/3rdparty/cutlass
#                (the submodule). `BSGEMM_CUTLASS_DIR` accepted as legacy alias.
#
#   PYTHON_INCLUDE   Optional override for the Python.h search path. The
#                    script tries `sysconfig.get_path("include")` first; if
#                    that doesn't have Python.h (some venvs / standalone
#                    Pythons don't ship the headers), set this to an explicit
#                    directory containing Python.h.
#                    Sites with multiarch headers (e.g. Debian/Ubuntu python-dev)
#                    may also need PYTHON_INCLUDE_MULTIARCH for pyconfig.h.
#
# Probes only standard system locations. No hard-coded user paths.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

ARCH="${ARCH:-9.0a;12.0a}"
CLEAN="${CLEAN:-1}"
EDITABLE="${EDITABLE:-0}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

# ---- Resolve CUDA_HOME --------------------------------------------------------
# Order: user-supplied env → nvcc on PATH → /usr/local/cuda.
# Refuse to guess beyond that — explicit is better than picking a toolkit
# the user didn't intend.
if [[ -z "${CUDA_HOME:-}" ]]; then
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        CUDA_HOME=/usr/local/cuda
    fi
fi
if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "[build] ERROR: cannot locate nvcc." >&2
    echo "[build]   set CUDA_HOME=/path/to/cuda, or put nvcc on PATH." >&2
    exit 1
fi
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"

# ---- Resolve CUTLASS ----------------------------------------------------------
CUTLASS_DIR="${CUTLASS_DIR:-${BSGEMM_CUTLASS_DIR:-${REPO}/3rdparty/cutlass}}"
if [[ ! -d "${CUTLASS_DIR}/include/cutlass" ]]; then
    echo "[build] ERROR: CUTLASS not at ${CUTLASS_DIR}/include/cutlass" >&2
    echo "[build]   run: git submodule update --init --recursive" >&2
    echo "[build]   or:  CUTLASS_DIR=/path/to/cutlass ./scripts/build.sh" >&2
    exit 1
fi
export CUTLASS_DIR

# ---- Resolve Python headers ---------------------------------------------------
# venv / standalone-Python interactions:
#   * Standard CPython sysconfig usually returns a real Python.h.
#   * `python -m venv` doesn't copy headers; it relies on the base Python's
#     include dir, which sysconfig.get_path('include') still points at — OK
#     if the base Python is system CPython or a uv-managed standalone build.
#   * Debian/Ubuntu split Python.h (in /usr/include/python3.X) from
#     pyconfig.h (in /usr/include/<multiarch>/python3.X). Both must be
#     reachable. We probe by actually trying to find both files.
#   * If you're on a host with a venv whose base Python has no headers
#     installed (e.g. minimal container with python3 but no python3-dev),
#     set PYTHON_INCLUDE explicitly to a dir that contains Python.h.
#
# We never bake user paths into the script — the only thing this block does
# is probe `sysconfig` outputs + standard system locations, and let
# PYTHON_INCLUDE override.
declare -a _py_extra_includes=()
PY_HAS_HEADER="$(python - <<'PY' 2>/dev/null || true
import os, sysconfig
inc = sysconfig.get_path("include")
plat = sysconfig.get_path("platinclude")
have_h = os.path.exists(os.path.join(inc, "Python.h"))
have_cfg = os.path.exists(os.path.join(inc, "pyconfig.h")) or os.path.exists(os.path.join(plat, "pyconfig.h"))
print(int(have_h and have_cfg))
PY
)"

if [[ -n "${PYTHON_INCLUDE:-}" ]]; then
    if [[ ! -f "${PYTHON_INCLUDE}/Python.h" ]]; then
        echo "[build] ERROR: PYTHON_INCLUDE=${PYTHON_INCLUDE} but Python.h missing" >&2
        exit 1
    fi
    _py_extra_includes+=("${PYTHON_INCLUDE}")
    if [[ -n "${PYTHON_INCLUDE_MULTIARCH:-}" ]]; then
        _py_extra_includes+=("${PYTHON_INCLUDE_MULTIARCH}")
    fi
elif [[ "${PY_HAS_HEADER:-0}" != "1" ]]; then
    # Auto-probe: ask the active Python where its own header lives. This
    # catches uv-installed standalone Pythons that ship their headers
    # alongside the interpreter (e.g. .../include/pythonX.Y/Python.h).
    while IFS= read -r _candidate; do
        if [[ -n "${_candidate}" && -f "${_candidate}/Python.h" ]]; then
            _py_extra_includes+=("${_candidate}")
            break
        fi
    done < <(python - <<'PY' 2>/dev/null || true
import os, sys, sysconfig
candidates = [
    sysconfig.get_path("include"),
    sysconfig.get_path("platinclude"),
    # Standalone CPython conventionally puts headers next to bin/.
    os.path.join(sys.base_prefix, "include", f"python{sys.version_info.major}.{sys.version_info.minor}"),
]
seen = set()
for c in candidates:
    if c and c not in seen:
        seen.add(c)
        print(c)
PY
)
    if [[ "${#_py_extra_includes[@]}" -eq 0 ]]; then
        echo "[build] ERROR: Python.h not found in sysconfig include dirs." >&2
        echo "[build]   Install python headers (e.g. apt install python3-dev)," >&2
        echo "[build]   or set PYTHON_INCLUDE=/path/containing/Python.h" >&2
        echo "[build]   (and PYTHON_INCLUDE_MULTIARCH=... if pyconfig.h is split out)." >&2
        exit 1
    fi
fi

if [[ "${#_py_extra_includes[@]}" -gt 0 ]]; then
    _joined="$(IFS=:; echo "${_py_extra_includes[*]}")"
    export CPATH="${CPATH:+${CPATH}:}${_joined}"
    echo "[build] CPATH += ${_joined}"
fi

# ---- Clean stale .so + build/ -------------------------------------------------
if [[ "${CLEAN}" == "1" ]]; then
    rm -rf python/build python/fish_scales_ops.egg-info
    rm -f  python/fish_scales_ops/_C.*.so
    echo "[build] cleaned python/build + stale .so"
fi

# ---- Build --------------------------------------------------------------------
export TORCH_CUDA_ARCH_LIST="${ARCH}"
echo "[build] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "[build] CUDA_HOME=${CUDA_HOME}"
echo "[build] CUTLASS_DIR=${CUTLASS_DIR}"
echo "[build] MAX_JOBS=${MAX_JOBS}  EDITABLE=${EDITABLE}"

cd python
if [[ "${EDITABLE}" == "1" ]]; then
    pip install -e . --no-build-isolation --verbose 2>&1 | tail -10
else
    python setup.py build_ext --inplace 2>&1 | tail -10
fi

# ---- Verify -------------------------------------------------------------------
so=$(ls fish_scales_ops/_C*.so 2>/dev/null | head -1)
if [[ -z "${so}" ]]; then
    echo "[build] ERROR: _C.so was not produced" >&2
    exit 1
fi
size=$(stat -c '%s' "${so}")
echo "[build] OK  ${so}  ($((size / 1024 / 1024)) MB)"

PYTHONPATH="${REPO}/python" python -c "
import torch, fish_scales_ops as fso
print(f'[smoke] gemm ops:      {sorted(n for n in dir(fso.gemm) if not n.startswith(\"_\"))}')
print(f'[smoke] attention ops: {sorted(n for n in dir(fso.attention) if not n.startswith(\"_\"))}')
print(f'[smoke] torch.ops:     {[n for n in dir(torch.ops.fish_scales_ops) if not n.startswith(\"_\")]}')
"
