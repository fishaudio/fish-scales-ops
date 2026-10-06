#!/usr/bin/env bash
# Build the fish-scales-ops wheel: one artifact, built once in a fixed container, that the H200 (sm_90), the B300
# (sm_100/sm_103) and the RTX 5090 (sm_120) install with pip and run without the source tree.
#
# Usage:
#   scripts/build_wheel.sh [--out DIR] [--jobs N] [--cutlass DIR] [--allow-dirty]
#                          [--ccache DIR [--ccache-bin FILE]] [--cpus N] [--memory SIZE]
#
#   --out DIR       where the outputs go (default: dist/ at the repository root, which git ignores)
#   --jobs N        parallel compile jobs (default 32). One job needs up to 16 GB of memory (each architecture pass
#                   of csrc/gemm/ops/mxfp8_kernel.cu peaks there), and four jobs together peaked near 29 GB.
#   --cutlass DIR   the CUTLASS checkout (default: the 3rdparty/cutlass submodule). It must be checked out at the
#                   commit the repository records for the submodule (git ls-tree HEAD 3rdparty/cutlass); any other
#                   commit is refused.
#   --allow-dirty   build a working tree that differs from HEAD, as it is, instead of refusing it. The wheel then
#                   records "dirty": true and the test kit's name ends in -dirty.
#   --ccache DIR    compile the CUDA translation units through ccache, with the cache in DIR (created when missing; it
#                   must lie outside the repository). Off by default. See "Compiler cache" below.
#   --ccache-bin F  the ccache executable that --ccache mounts into the container (default: $FSO_CCACHE_BIN). It has to
#                   run there, so use a static build, such as ccache-4.14.1-linux-x86_64-musl-static of the ccache
#                   releases.
#   --cpus N        limit the build container to N CPUs (docker run --cpus), for example 4 to reproduce a GitHub-hosted
#                   runner. Choose --jobs to match.
#   --memory SIZE   limit the build container's memory, with no swap (docker run --memory and --memory-swap), for
#                   example 16g.
#   --arch LIST     development only: build these TORCH_CUDA_ARCH_LIST entries instead of "9.0a;10.0f;12.0a".
#
# Environment:
#   FSO_NVRTC_WHEEL  the nvidia-cuda-nvrtc 13.0.88 wheel to bundle. The container then runs without a network.
#                    Otherwise the container downloads the wheel with pip. scripts/vendor_nvrtc.py checks its pinned
#                    sha256 either way.
#   FSO_CCACHE_BIN   the default of --ccache-bin.
#   FSO_CCACHE_MAXSIZE  the size limit of the --ccache directory (default 2GiB); ccache evicts the least recently used
#                    objects beyond it.
#
# Steps:
#   1. The source is `git archive HEAD`, the committed tree only. A working tree with uncommitted changes or
#      untracked files is refused unless --allow-dirty is given; the source is then the working tree's files, tracked
#      and untracked, except what .gitignore excludes.
#   2. CUTLASS is exported from the --cutlass checkout at the recorded commit: include/, tools/util/include/,
#      LICENSE.txt and the CuTe-DSL kernel of the sm_100 mid-band tier.
#   3. Unless it exists, the image fso-build-wheel:<first 12 hex digits of the Dockerfile's sha256> is built from
#      docker/build-wheel.Dockerfile: Ubuntu 22.04 (glibc 2.35, gcc 11), CUDA 13.0.3, Python 3.12, torch 2.13.0+cu130.
#      It holds the toolchain only, no fish-scales-ops source.
#   4. scripts/build_wheel_inner.sh runs in a container of that image, as the invoking user, without a GPU or a
#      driver. It bundles NVRTC (scripts/vendor_nvrtc.py), stages _jit_include/, _dsl/ and BUILD_INFO.json into the
#      package (scripts/build_wheel_package.py stage), builds the extension and the wheel with
#      `pip wheel python/ --no-build-isolation --no-deps` for 9.0a;10.0f;12.0a (libcuda.so from the toolkit's
#      lib64/stubs), records the newest GLIBC version the extension needs, and runs the post-build checks
#      (scripts/build_wheel_package.py check): the packaged files, at most GLIBC_2.35 and GLIBCXX_3.4.30, no
#      link-time NVRTC, and cubins for sm_90a, sm_100f and sm_120a. Any failure fails this script.
#   5. Outputs in --out: the wheel, <wheel>.sha256, a copy of BUILD_INFO.json, build.log, and the test kit
#      fish_scales_ops-<version>-testkit-<commit>.tar.gz, which holds tests/, bench/, scripts/, docs/, README.md and
#      CHANGELOG.md of the same source.
#
# Compiler cache (--ccache DIR):
#   torch's cpp_extension starts nvcc with the command in the PYTORCH_NVCC environment variable when it is set. The
#   build sets it to "<ccache> $CUDA_HOME/bin/nvcc", so every nvcc command keeps its arguments and paths and only gains
#   ccache in front of it. ccache runs in depend mode: it keys each object on the compiler, the arguments, the source
#   file and every header in nvcc's -MD dependency list, which covers the host pass and every device pass. ccache's
#   default mode is not safe here: it hashes the output of `nvcc -E`, which holds only the first architecture's pass,
#   so after a change that only the sm_100 pass, the sm_120 pass or the host code sees it would return the old object.
#   ccache silently leaves depend mode for a compile without -MD, so its preprocessor command is set to `false`: such a
#   compile then runs without the cache instead of in the default mode. The two host C++ translation units are compiled
#   without the cache. A hit returns the exact bytes of an earlier compile, and a miss runs the compile unchanged. The
#   cache's statistics for this build are appended to build.log.
#
# Installing: `pip install --no-deps <wheel>` into an environment that has the torch the wheel was built against
# (fish_scales_ops.build_info()["torch"]); importing the package under another torch raises ImportError.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${REPO}/dist"
JOBS=32
CUTLASS="${REPO}/3rdparty/cutlass"
ALLOW_DIRTY=0
ARCH="9.0a;10.0f;12.0a"
COMPILER_CACHE=""
COMPILER_CACHE_BIN="${FSO_CCACHE_BIN:-}"
COMPILER_CACHE_MAX="${FSO_CCACHE_MAXSIZE:-2GiB}"
CPUS=""
MEMORY=""

say() { echo "[build_wheel] $*"; }
die() { echo "[build_wheel] ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT="${2:?--out needs a directory}"; shift 2 ;;
        --jobs) JOBS="${2:?--jobs needs a number}"; shift 2 ;;
        --cutlass) CUTLASS="${2:?--cutlass needs a directory}"; shift 2 ;;
        --allow-dirty) ALLOW_DIRTY=1; shift ;;
        --arch) ARCH="${2:?--arch needs a list}"; shift 2 ;;
        --ccache) COMPILER_CACHE="${2:?--ccache needs a directory}"; shift 2 ;;
        --ccache-bin) COMPILER_CACHE_BIN="${2:?--ccache-bin needs a file}"; shift 2 ;;
        --cpus) CPUS="${2:?--cpus needs a number}"; shift 2 ;;
        --memory) MEMORY="${2:?--memory needs a size}"; shift 2 ;;
        -h|--help) sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument $1 (see --help)" ;;
    esac
done
[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || die "--jobs ${JOBS} is not a positive number"
[[ -z "${CPUS}" || "${CPUS}" =~ ^([1-9][0-9]*|[0-9]+[.][0-9]+)$ ]] || die "--cpus ${CPUS} is not a positive number"
[[ -z "${MEMORY}" || "${MEMORY}" =~ ^[1-9][0-9]*[bkmgBKMG]?$ ]] || die "--memory ${MEMORY} is not a size such as 16g"
for tool in docker git tar sha256sum; do
    command -v "${tool}" >/dev/null 2>&1 || die "${tool} is not on PATH"
done
mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"
# ccache with a configuration that only these variables define: CCACHE_CONFIGPATH=/dev/null keeps a ccache.conf in the
# cache directory, or a system one, from changing it.
ccache_host() {
    CCACHE_CONFIGPATH=/dev/null CCACHE_MAXSIZE="${COMPILER_CACHE_MAX}" \
        "${COMPILER_CACHE_BIN}" --dir "${COMPILER_CACHE}" "$@"
}
if [[ -n "${COMPILER_CACHE}" ]]; then
    [[ -n "${COMPILER_CACHE_BIN}" ]] \
        || die "--ccache needs the ccache executable: pass --ccache-bin FILE or set FSO_CCACHE_BIN"
    [[ -f "${COMPILER_CACHE_BIN}" && -x "${COMPILER_CACHE_BIN}" ]] \
        || die "--ccache-bin ${COMPILER_CACHE_BIN} is not an executable file"
    COMPILER_CACHE_BIN="$(realpath "${COMPILER_CACHE_BIN}")"
    # The container runs the same binary, so one that runs here (a static build for this architecture) runs there.
    COMPILER_CACHE_VERSION="$("${COMPILER_CACHE_BIN}" --print-version 2>/dev/null)" || true
    [[ "${COMPILER_CACHE_VERSION}" =~ ^[0-9]+[.][0-9] ]] \
        || die "--ccache-bin ${COMPILER_CACHE_BIN} does not run as ccache on this host; use a static ccache build"
    COMPILER_CACHE="$(realpath -m "${COMPILER_CACHE}")"
    case "${COMPILER_CACHE}/" in
        "$(cd "${REPO}" && pwd -P)/"*)
            die "the compiler cache ${COMPILER_CACHE} is inside the repository; put it outside" ;;
    esac
    mkdir -p "${COMPILER_CACHE}"
    ccache_host --zero-stats >/dev/null   # the statistics at the end then count this build only
fi
START=$(date +%s)

# ---- 1. Source --------------------------------------------------------------------------------------------------
cd "${REPO}"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "${REPO} is not a git working tree"
COMMIT="$(git rev-parse HEAD)"
SHORT="$(git rev-parse --short HEAD)"
STATUS="$(git status --porcelain --untracked-files=normal --ignore-submodules=all)"
DIRTY=0
if [[ -n "${STATUS}" ]]; then
    if [[ "${ALLOW_DIRTY}" != "1" ]]; then
        echo "${STATUS}" | sed 's/^/  /' >&2
        die "the working tree differs from HEAD ${SHORT} (above). Commit the changes, or pass --allow-dirty to build the working tree as it is."
    fi
    DIRTY=1
    say "--allow-dirty: building the working tree of ${SHORT} as it is, with these differences:"
    echo "${STATUS}" | sed 's/^/  /'
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/fso_build_wheel.XXXXXX")"
CONTAINER="fso-build-wheel-$$-$(date +%s)"
cleanup() {
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
    rm -rf "${TMP}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
mkdir -p "${TMP}/in" "${TMP}/out" "${TMP}/kit"

if [[ "${DIRTY}" == "1" ]]; then
    # Tracked files that still exist (a deleted one is left out; a submodule, mode 160000, is not a file), then the
    # untracked files git does not ignore.
    {
        git ls-files -z --stage | while IFS= read -r -d '' entry; do
            mode="${entry%% *}"
            path="${entry#*$'\t'}"
            if [[ "${mode}" != "160000" ]] && [[ -e "${path}" || -L "${path}" ]]; then
                printf '%s\0' "${path}"
            fi
        done
        git ls-files -z --others --exclude-standard
    } | tar --null -T - -cf "${TMP}/in/src.tar"
else
    git archive --format=tar HEAD > "${TMP}/in/src.tar"
fi
VERSION="$(tar -xOf "${TMP}/in/src.tar" python/pyproject.toml | sed -n 's/^version *= *"\([^"]*\)".*/\1/p' | head -1)"
[[ -n "${VERSION}" ]] || die "python/pyproject.toml of the source has no version line"
KIT="fish_scales_ops-${VERSION}-testkit-${SHORT}$([[ "${DIRTY}" == "1" ]] && echo -dirty || true)"
say "source: ${COMMIT}$([[ "${DIRTY}" == "1" ]] && echo ' plus the working-tree changes' || true), version ${VERSION}"

# ---- 2. CUTLASS ---------------------------------------------------------------------------------------------------
WANT_CUTLASS="$(git ls-tree HEAD 3rdparty/cutlass | awk '$2 == "commit" { print $3 }')"
[[ -n "${WANT_CUTLASS}" ]] || die "HEAD records no commit for the submodule 3rdparty/cutlass"
# The directory must be the top of a checkout of its own: an uninitialised submodule is an empty directory inside
# this repository, where git would answer with this repository's HEAD.
CUTLASS_TOP="$(git -C "${CUTLASS}" rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "${CUTLASS_TOP}" && "$(cd "${CUTLASS}" && pwd -P)" == "$(cd "${CUTLASS_TOP}" && pwd -P)" ]] \
    || die "${CUTLASS} is not a git checkout of CUTLASS (an uninitialised submodule is an empty directory): run git submodule update --init, or pass --cutlass DIR"
HAVE_CUTLASS="$(git -C "${CUTLASS}" rev-parse HEAD)"
[[ "${HAVE_CUTLASS}" == "${WANT_CUTLASS}" ]] \
    || die "CUTLASS at ${CUTLASS} is at ${HAVE_CUTLASS}, but the repository records ${WANT_CUTLASS} for 3rdparty/cutlass; check that commit out, or pass --cutlass DIR"
git -C "${CUTLASS}" archive --format=tar "${WANT_CUTLASS}" include tools/util/include LICENSE.txt \
    examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py > "${TMP}/in/cutlass.tar"
say "CUTLASS: ${CUTLASS} at ${WANT_CUTLASS}"

# ---- 3. Image ---------------------------------------------------------------------------------------------------
DOCKERFILE="${REPO}/docker/build-wheel.Dockerfile"
IMAGE="fso-build-wheel:$(sha256sum "${DOCKERFILE}" | cut -c1-12)"
BASE_IMAGE="$(sed -n 's/^FROM[[:space:]]\{1,\}\([^[:space:]]*\).*/\1/p' "${DOCKERFILE}" | head -1)"
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    say "building the image ${IMAGE} from ${DOCKERFILE}"
    mkdir -p "${TMP}/docker"
    cp "${DOCKERFILE}" "${TMP}/docker/Dockerfile"
    docker build -t "${IMAGE}" "${TMP}/docker"
fi
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE}")"
say "image: ${IMAGE} (${IMAGE_ID}), from ${BASE_IMAGE}"
if [[ -n "${COMPILER_CACHE}" ]]; then
    # The nvcc torch would start: $CUDA_HOME/bin/nvcc, with the CUDA_HOME the image sets.
    IMAGE_CUDA_HOME="$(docker image inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "${IMAGE}" \
        | sed -n 's/^CUDA_HOME=//p' | head -1)"
    [[ -n "${IMAGE_CUDA_HOME}" ]] \
        || die "the image ${IMAGE} sets no CUDA_HOME, so the compiler cache cannot name its nvcc"
fi

# ---- 4. Build in the container ----------------------------------------------------------------------------------
RUN_ARGS=(--rm --name "${CONTAINER}" --user "$(id -u):$(id -g)"
          -e HOME=/opt/fso-wheel-build/home
          -e FSO_BUILD_COMMIT="${COMMIT}" -e FSO_BUILD_DIRTY="${DIRTY}"
          -e FSO_BUILD_IMAGE="${IMAGE}" -e FSO_BUILD_IMAGE_ID="${IMAGE_ID}" -e FSO_BUILD_BASE_IMAGE="${BASE_IMAGE}"
          -e FSO_BUILD_CUTLASS_COMMIT="${WANT_CUTLASS}" -e FSO_BUILD_ARCH="${ARCH}" -e MAX_JOBS="${JOBS}"
          -v "${TMP}/in:/in:ro" -v "${TMP}/out:/out")
if [[ -n "${CPUS}" ]]; then
    RUN_ARGS+=(--cpus "${CPUS}")
fi
if [[ -n "${MEMORY}" ]]; then
    RUN_ARGS+=(--memory "${MEMORY}" --memory-swap "${MEMORY}")
fi
if [[ -n "${COMPILER_CACHE}" ]]; then
    # The binary is mounted off PATH on purpose: torch wraps every compiler with a ccache it finds on PATH, the host C++
    # compiler included, and only nvcc is meant to go through the cache.
    RUN_ARGS+=(-v "${COMPILER_CACHE}:/opt/fso-ccache/cache" -v "${COMPILER_CACHE_BIN}:/opt/fso-ccache/bin/ccache:ro"
               -e CCACHE_CONFIGPATH=/dev/null -e CCACHE_DIR=/opt/fso-ccache/cache -e CCACHE_DEPEND=1
               -e CCACHE_PREFIX_CPP=false -e CCACHE_MAXSIZE="${COMPILER_CACHE_MAX}"
               -e PYTORCH_NVCC="/opt/fso-ccache/bin/ccache ${IMAGE_CUDA_HOME}/bin/nvcc")
    say "compiler cache: ccache ${COMPILER_CACHE_VERSION} in depend mode, ${COMPILER_CACHE}" \
        "(at most ${COMPILER_CACHE_MAX})"
fi
if [[ -n "${FSO_NVRTC_WHEEL:-}" ]]; then
    [[ -f "${FSO_NVRTC_WHEEL}" ]] || die "FSO_NVRTC_WHEEL=${FSO_NVRTC_WHEEL} does not exist"
    NVRTC_NAME="$(basename "${FSO_NVRTC_WHEEL}")"
    RUN_ARGS+=(--network none -v "$(realpath "${FSO_NVRTC_WHEEL}"):/in-nvrtc/${NVRTC_NAME}:ro"
               -e FSO_NVRTC_WHEEL="/in-nvrtc/${NVRTC_NAME}")
fi
say "building in ${CONTAINER} (${JOBS} jobs, arch ${ARCH}${CPUS:+, ${CPUS} CPUs}${MEMORY:+, ${MEMORY} memory});" \
    "the log goes to ${OUT}/build.log"
set +e
docker run "${RUN_ARGS[@]}" "${IMAGE}" bash -c \
    'set -euo pipefail; mkdir -p /opt/fso-wheel-build/src; tar -xf /in/src.tar -C /opt/fso-wheel-build/src;
     exec bash /opt/fso-wheel-build/src/scripts/build_wheel_inner.sh' 2>&1 | tee "${TMP}/out/build.log"
RC=${PIPESTATUS[0]}
set -e
if [[ -n "${COMPILER_CACHE}" ]]; then
    {
        echo "[build_wheel] compiler cache statistics of this build (${COMPILER_CACHE}):"
        ccache_host --show-stats --verbose || true
        stats="$(ccache_host --print-stats 2>/dev/null)" || stats=""
        counter() { awk -v k="$1" '$1 == k { n += $2 } END { print n + 0 }' <<< "${stats}"; }
        hits=$(( $(counter direct_cache_hit) + $(counter preprocessed_cache_hit) ))
        misses="$(counter cache_miss)"
        uncached="$(counter preprocessor_error)"
        echo "[build_wheel] compiler cache: ${hits} CUDA compiles came from the cache," \
            "${misses} were compiled and stored"
        # None at all means torch ignored PYTORCH_NVCC; uncached ones had no -MD. Either way the wheel is right, but
        # those compiles were not cached.
        if [[ -n "${stats}" && $(( hits + misses + uncached )) == 0 ]]; then
            echo "[build_wheel] WARNING: no compile went through ccache; torch did not use PYTORCH_NVCC"
        fi
        if [[ "${uncached}" != "0" ]]; then
            echo "[build_wheel] WARNING: ${uncached} compiles had no -MD dependency list and ran without the cache"
        fi
    } 2>&1 | tee -a "${TMP}/out/build.log"
fi
cp "${TMP}/out/build.log" "${OUT}/build.log"
[[ "${RC}" == "0" ]] || die "the build in the container failed (exit ${RC}); see ${OUT}/build.log"

# ---- 5. Outputs -------------------------------------------------------------------------------------------------
WHEELS=("${TMP}"/out/*.whl)
[[ ${#WHEELS[@]} -eq 1 && -f "${WHEELS[0]}" ]] || die "the container produced ${#WHEELS[@]} wheels"
WHEEL="$(basename "${WHEELS[0]}")"
(cd "${TMP}/out" && sha256sum -c "${WHEEL}.sha256" >/dev/null) || die "the sha256 of ${WHEEL} does not match"

mkdir -p "${TMP}/kit/${KIT}"
tar -xf "${TMP}/in/src.tar" -C "${TMP}/kit/${KIT}" tests bench scripts docs README.md CHANGELOG.md
tar -czf "${TMP}/out/${KIT}.tar.gz" -C "${TMP}/kit" "${KIT}"

for f in "${WHEEL}" "${WHEEL}.sha256" BUILD_INFO.json "${KIT}.tar.gz"; do
    cp "${TMP}/out/${f}" "${OUT}/${f}"
done
say "done in $(( $(date +%s) - START )) s:"
say "  wheel     ${OUT}/${WHEEL} ($(( $(stat -c %s "${OUT}/${WHEEL}") / 1024 / 1024 )) MiB)"
say "  sha256    $(cut -d' ' -f1 "${OUT}/${WHEEL}.sha256")"
say "  test kit  ${OUT}/${KIT}.tar.gz"
say "  build     ${OUT}/BUILD_INFO.json, ${OUT}/build.log"
