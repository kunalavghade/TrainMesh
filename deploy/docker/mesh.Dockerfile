# TrainMesh mesh image: the Python runtime plus meshd, tm-agent and the trainmesh CLI.
# The Go binaries are cross-compiled on the host and passed in the build context
# (see scripts/e2e/minikube.sh mesh).
ARG BASE_IMAGE=trainmesh-runtime:dev
FROM ${BASE_IMAGE}

COPY trainmesh meshd tm-agent /usr/local/bin/
ENV TRAINMESH_PYTHON=/usr/local/bin/python \
    TRAINMESH_RUNTIME_DIR=/opt/trainmesh/python
