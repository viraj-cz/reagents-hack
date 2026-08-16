FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG LEAN_VERSION=v4.30.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git python3 python3-pip python3-venv z3 zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain "${LEAN_VERSION}"
ENV PATH="/root/.elan/bin:${PATH}"

RUN git clone --depth 1 --branch "${LEAN_VERSION}" \
    https://github.com/leanprover-community/mathlib4.git /opt/mathlib \
    && cd /opt/mathlib \
    && lake exe cache get

RUN python3 -m pip install --break-system-packages --no-cache-dir \
    "numpy>=2,<3" "scipy>=1.14,<2" "sympy>=1.13,<2" \
    "networkx>=3.3,<4" "pint>=0.24,<1" "cvxpy>=1.6,<2" \
    "control>=0.10,<1"

# The operations module now lives inside the `reagents` package so the Modal
# broker executors can IMPORT it (see broker/service.py). It deliberately has no
# `reagents` imports, so copying the single file here still works standalone.
COPY src/reagents/tools/tool_runtime.py /opt/reagents/tool_runtime.py
WORKDIR /opt/mathlib
