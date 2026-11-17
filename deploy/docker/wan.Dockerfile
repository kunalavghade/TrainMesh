# WAN test image: the Python runtime + the tmcomm extension + meshd/tm-agent/trainmesh + tc.
# Built by scripts/e2e/wan-docker.sh from the repository root; the Go binaries are
# cross-compiled on the host into deploy/docker/.wan-bin first.

# Declared before the first FROM so the second stage's FROM can use it.
ARG BASE_IMAGE=trainmesh-runtime:dev

# Stage 1: build the tmcomm wheel for this architecture (abi3, Python >= 3.11).
FROM rust:1.90-slim-bookworm AS tmcomm
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /maturin && /maturin/bin/pip install --no-cache-dir "maturin==1.16.0"
COPY native /src/native
RUN cd /src/native/tmcomm && /maturin/bin/maturin build --release --compatibility linux --out /wheels

# Stage 2: runtime image + tmcomm + iproute2 (tc netem) + Go binaries.
FROM ${BASE_IMAGE}
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends iproute2 procps \
    && rm -rf /var/lib/apt/lists/*
COPY --from=tmcomm /wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels
COPY python /opt/trainmesh/python
RUN pip install --no-cache-dir /opt/trainmesh/python
COPY deploy/docker/.wan-bin/ /usr/local/bin/
ENV TRAINMESH_PYTHON=/usr/local/bin/python \
    TRAINMESH_RUNTIME_DIR=/opt/trainmesh/python
WORKDIR /opt/trainmesh
