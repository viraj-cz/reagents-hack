FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PROTO_REF=edf64afbcf84cc7c5e4e1404418c8ef1f16c34ce

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates cmake curl gcc g++ git make python3 python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --no-cache-dir \
    "git+https://github.com/evo-design/proto-language.git@${PROTO_REF}"

COPY tooling/runtime/tool_runtime.py /opt/reagents/tool_runtime.py
WORKDIR /work
