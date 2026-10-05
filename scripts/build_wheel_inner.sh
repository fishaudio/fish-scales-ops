#!/usr/bin/env bash
# The part of scripts/build_wheel.sh that runs inside the fso-build-wheel container (docker/build-wheel.Dockerfile).
# It is not meant for a host: it builds under /opt/fso-wheel-build and reads what the host script mounts.
#
# Inputs:   /opt/fso-wheel-build/src    the source, unpacked from /in/src.tar before this script starts
#           /in/cutlass.tar             CUTLASS at the submodule commit
#           FSO_BUILD_COMMIT, FSO_BUILD_DIRTY, FSO_BUILD_IMAGE, FSO_BUILD_IMAGE_ID, FSO_BUILD_BASE_IMAGE,
#           FSO_BUILD_CUTLASS_COMMIT, FSO_BUILD_ARCH, MAX_JOBS; FSO_NVRTC_WHEEL when the host passes the NVRTC wheel
# Outputs:  /out/<wheel>, /out/<wheel>.sha256, /out/BUILD_INFO.json

set -euo pipefail

W=/opt/fso-wheel-build
SRC="${W}/src"
say() { echo "[build_wheel_inner] $*"; }

for v in FSO_BUILD_COMMIT FSO_BUILD_DIRTY FSO_BUILD_IMAGE FSO_BUILD_IMAGE_ID FSO_BUILD_BASE_IMAGE \
         FSO_BUILD_CUTLASS_COMMIT FSO_BUILD_ARCH MAX_JOBS; do
    [[ -n "${!v:-}" ]] || { echo "[build_wheel_inner] ERROR: ${v} is not set" >&2; exit 1; }
done
mkdir -p "${SRC}/3rdparty/cutlass" "${W}/dist" "${HOME}"
tar -xf /in/cutlass.tar -C "${SRC}/3rdparty/cutlass"
cd "${SRC}"

export CUTLASS_DIR="${SRC}/3rdparty/cutlass"
export TORCH_CUDA_ARCH_LIST="${FSO_BUILD_ARCH}"
export MAX_JOBS
# The nvidia/cuda image puts /usr/local/cuda/lib64 on LD_LIBRARY_PATH, which the dynamic loader searches before a
# library's RUNPATH. The bundled libnvrtc.so.13 finds its libnvrtc-builtins.so.13.2 through RUNPATH $ORIGIN, so with
# that variable set it would load the toolkit's copy instead, and scripts/vendor_nvrtc.py rightly refuses that.
# Nothing in the build needs the variable.
unset LD_LIBRARY_PATH
# The container has no GPU and no driver: the link takes libcuda.so from the toolkit's stubs. The extension records
# libcuda.so.1 as NEEDED, which the serving machine's driver provides.
export LIBRARY_PATH="${CUDA_HOME}/lib64/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}"

say "toolchain: $(nvcc --version | grep '^Cuda compilation tools'); $(gcc --version | head -1); $(python --version);" \
    "torch $(python -c 'import torch; print(torch.__version__)'); $(ldd --version | head -1)"
say "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS} CUTLASS_DIR=${CUTLASS_DIR}"

# 1. The bundled NVRTC (honours FSO_NVRTC_WHEEL).
python scripts/vendor_nvrtc.py

# 2. The packaged JIT include tree, the CuTe-DSL kernel and the build information.
python scripts/build_wheel_package.py stage --commit "${FSO_BUILD_COMMIT}" --dirty "${FSO_BUILD_DIRTY}" \
    --image "${FSO_BUILD_IMAGE}" --image-id "${FSO_BUILD_IMAGE_ID}" --base-image "${FSO_BUILD_BASE_IMAGE}" \
    --cutlass-commit "${FSO_BUILD_CUTLASS_COMMIT}" --cutlass "${CUTLASS_DIR}" --cuda-home "${CUDA_HOME}" \
    --arch-list "${FSO_BUILD_ARCH}"

# 3 and 4. The extension and the wheel, in one setuptools build.
start=$(date +%s)
python -m pip wheel ./python --no-build-isolation --no-deps --wheel-dir "${W}/dist" -v
say "pip wheel took $(( $(date +%s) - start )) s"
wheels=("${W}"/dist/fish_scales_ops-*.whl)
[[ ${#wheels[@]} -eq 1 && -f "${wheels[0]}" ]] || { echo "[build_wheel_inner] ERROR: ${#wheels[@]} wheels" >&2; exit 1; }
whl="${wheels[0]}"

# 5. glibc_required, read from the extension in the wheel.
python scripts/build_wheel_package.py finalize "${whl}"

# 6. The post-build checks.
python scripts/build_wheel_package.py check "${whl}" --arch-list "${FSO_BUILD_ARCH}"

cp "${whl}" /out/
(cd /out && sha256sum "$(basename "${whl}")" > "$(basename "${whl}").sha256")
cp python/fish_scales_ops/BUILD_INFO.json /out/BUILD_INFO.json
say "wrote /out/$(basename "${whl}")"
