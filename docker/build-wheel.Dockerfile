# The fixed environment in which scripts/build_wheel.sh builds the fish-scales-ops wheel.
#
# One wheel serves the H200 (sm_90), the B300 (sm_100/sm_103) and the RTX 5090 (sm_120), so it is built once here
# rather than on each machine:
#   - Ubuntu 22.04 has glibc 2.35 and gcc 11, so the extension needs no glibc newer than the oldest serving host has
#     (the B300 pod runs Debian 12 with glibc 2.36 and GLIBCXX_3.4.30). An extension built on Ubuntu 24.04 needs
#     GLIBC_2.38 and does not load there.
#   - The CUDA 13.2.1 toolkit (nvcc and NVRTC 13.2.78) is the one the kernels are validated with. Its headers also
#     become the packaged sm_90 JIT include tree.
#   - Python 3.12 and torch 2.13.0+cu130 match the three serving environments.
# The image holds the toolchain only. scripts/build_wheel.sh mounts the source when it runs the build; it builds
# this file into the image fso-build-wheel:<first 12 hex digits of this file's sha256>.

FROM nvidia/cuda:13.2.1-devel-ubuntu22.04@sha256:3805e62773832c404db8785c63190aaba49bff02bb199b0d187ab897170c6cd1

ARG DEBIAN_FRONTEND=noninteractive

# Python 3.12 with its headers comes from the deadsnakes PPA, whose signing key is checked against its fingerprint.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg git; \
    curl -fsSL 'https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776' \
        -o /tmp/deadsnakes.asc; \
    fpr="$(gpg --show-keys --with-colons /tmp/deadsnakes.asc | awk -F: '$1 == "fpr" { print $10; exit }')"; \
    test "${fpr}" = F23C5A6CF475977595C89F51BA6932366A755776; \
    gpg --dearmor -o /usr/share/keyrings/deadsnakes.gpg /tmp/deadsnakes.asc; \
    echo 'deb [signed-by=/usr/share/keyrings/deadsnakes.gpg] https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main' \
        > /etc/apt/sources.list.d/deadsnakes.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends python3.12 python3.12-dev python3.12-venv; \
    rm -rf /var/lib/apt/lists/* /tmp/deadsnakes.asc /root/.gnupg

# One virtual environment with pinned build tools and the torch the wheel is built against. setuptools is installed
# before torch so that torch's own dependency on it keeps the pinned version.
RUN set -eux; \
    python3.12 -m venv /opt/venv; \
    /opt/venv/bin/python -m pip install --no-cache-dir pip==26.2.1; \
    /opt/venv/bin/python -m pip install --no-cache-dir setuptools==84.0.0 wheel==0.48.0 ninja==1.13.2; \
    /opt/venv/bin/python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0; \
    /opt/venv/bin/python -c "import torch; assert torch.__version__ == '2.13.0+cu130', torch.__version__"

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# scripts/build_wheel.sh runs the build as the invoking user, so the build directory is writable by everyone.
RUN mkdir -m 1777 /opt/fso-wheel-build
