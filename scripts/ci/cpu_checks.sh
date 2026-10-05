#!/usr/bin/env bash
# The checks of a built wheel that need no GPU, run inside the build image.
#
# Usage:
#   scripts/ci/cpu_checks.sh [--dist DIR]        # on the host: runs itself in the fso-build-wheel image
#
#   --dist DIR   the directory scripts/build_wheel.sh wrote (default: dist/ at the repository root). It must hold
#                exactly one wheel and its test kit.
#
# What runs (any failure fails the script):
#   1. Source-tree checks of the repository: the JIT-isolation lint, `gen_op_schemas.py --check`,
#      `render_perf_docs.py --check`, and that every Python file parses.
#   2. The wheel, installed with `pip install --no-deps --target` into a scratch directory, with the test kit unpacked
#      next to it and no source tree on PYTHONPATH:
#        - `import fish_scales_ops` under `-W error::DeprecationWarning`, printing build_info();
#        - tests/gemm/unit/test_public_surface.py, test_wheel_layout.py and test_jit_nvrtc_pin_sm90.py (their
#          device-independent parts; the device parts skip themselves without a GPU);
#        - tests/bench/test_run_perf_plan.py (the performance harness's plan and lock files).
#
# The image has the CUDA toolkit but no driver, so libcuda.so.1 is taken from the toolkit's stub library. Nothing here
# launches a kernel; the GPU tests run on the three machines through scripts/ci/run_suite.py.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST="${REPO}/dist"
INNER=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dist) DIST="${2:?--dist needs a directory}"; shift 2 ;;
        --inner) INNER=1; shift ;;
        -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        *) echo "[cpu_checks] unknown argument $1" >&2; exit 2 ;;
    esac
done

if [[ "${INNER}" == "0" ]]; then
    DIST="$(cd "${DIST}" && pwd)"
    tag="fso-build-wheel:$(sha256sum "${REPO}/docker/build-wheel.Dockerfile" | cut -c1-12)"
    docker image inspect "${tag}" >/dev/null 2>&1 || {
        echo "[cpu_checks] image ${tag} not found; scripts/build_wheel.sh builds it" >&2; exit 1; }
    exec docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -v "${REPO}:/src:ro" -v "${DIST}:/dist:ro" "${tag}" bash /src/scripts/ci/cpu_checks.sh --inner --dist /dist
fi

# ---- inside the image ---------------------------------------------------------
PY=/opt/venv/bin/python
SRC=/src
fail=0
run() {  # run <name> <command...>
    local name="$1"; shift
    if "$@" > "/tmp/${name}.log" 2>&1; then
        echo "[cpu_checks] PASS ${name}: $(grep -v '^\s*$' "/tmp/${name}.log" | tail -1 | cut -c1-140)"
    else
        echo "[cpu_checks] FAIL ${name} (exit $?)"; tail -25 "/tmp/${name}.log"; fail=1
    fi
}

cd "${SRC}"
run lint_jit_isolation bash tests/gemm/unit/lint_jit_isolation.sh
run gen_op_schemas "${PY}" scripts/gen_op_schemas.py --check
run render_perf_docs "${PY}" bench/gemm/python/render_perf_docs.py --check
run python_parse "${PY}" - <<'PYEOF'
import ast, pathlib, sys
bad = []
for top in ("python", "tests", "bench", "scripts"):
    for p in pathlib.Path(top).rglob("*.py"):
        if "build" in p.parts or "__pycache__" in p.parts or "_nvrtc" in p.parts or "_jit_include" in p.parts:
            continue
        try:
            ast.parse(p.read_text())
        except SyntaxError as e:
            bad.append(f"{p}: {e}")
print(f"{len(bad)} files fail to parse" if bad else "every Python file parses")
sys.exit(1 if bad else 0)
PYEOF

wheels=("${DIST}"/fish_scales_ops-*.whl)
kits=("${DIST}"/fish_scales_ops-*-testkit-*.tar.gz)
[[ ${#wheels[@]} -eq 1 && -f "${wheels[0]}" ]] || { echo "[cpu_checks] ${DIST} must hold exactly one wheel"; exit 1; }
[[ ${#kits[@]} -eq 1 && -f "${kits[0]}" ]] || { echo "[cpu_checks] ${DIST} must hold exactly one test kit"; exit 1; }
( cd "${DIST}" && sha256sum -c "$(basename "${wheels[0]}").sha256" ) || { echo "[cpu_checks] wheel sha256 mismatch"; exit 1; }

WORK="$(mktemp -d)"
"${PY}" -m pip install --quiet --no-deps --no-compile --disable-pip-version-check --target "${WORK}/site" "${wheels[0]}"
mkdir -p "${WORK}/kit" "${WORK}/stubs"
tar -xzf "${kits[0]}" -C "${WORK}/kit"
KIT="${WORK}/kit"; [[ -d "${KIT}/tests" ]] || KIT="$(dirname "$(find "${WORK}/kit" -maxdepth 2 -type d -name tests | head -1)")"
# No driver in the image: the extension's libcuda.so.1 resolves to the toolkit's stub.
ln -s "${CUDA_HOME:-/usr/local/cuda}/lib64/stubs/libcuda.so" "${WORK}/stubs/libcuda.so.1"
export LD_LIBRARY_PATH="${WORK}/stubs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" PYTHONPATH="${WORK}/site" PYTHONDONTWRITEBYTECODE=1
for v in $(env | grep -o '^FSO_[A-Za-z0-9_]*' || true); do unset "$v"; done

cd "${KIT}"
run wheel_import "${PY}" -W error::DeprecationWarning -c "
import json, fish_scales_ops as f
assert f.__file__.startswith('${WORK}/site'), f.__file__
print('fish_scales_ops', f.__version__, json.dumps(f.build_info(), sort_keys=True))"
run test_public_surface "${PY}" tests/gemm/unit/test_public_surface.py
run test_wheel_layout "${PY}" tests/gemm/unit/test_wheel_layout.py
run test_jit_nvrtc_pin_sm90 "${PY}" tests/gemm/unit/test_jit_nvrtc_pin_sm90.py
run test_run_perf_plan "${PY}" tests/bench/test_run_perf_plan.py

rm -rf "${WORK}"
[[ "${fail}" == "0" ]] && echo "[cpu_checks] ALL PASS" || { echo "[cpu_checks] FAILED"; exit 1; }
