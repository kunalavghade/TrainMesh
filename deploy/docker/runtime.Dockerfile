# CPU image of the TrainMesh Python runtime, used by the multi-node dev tests.
FROM python:3.12-slim

ARG TORCH_VERSION=2.15.0
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1

RUN pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu

COPY python /opt/trainmesh/python
RUN pip install /opt/trainmesh/python

WORKDIR /opt/trainmesh
