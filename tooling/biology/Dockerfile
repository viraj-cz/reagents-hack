FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates hmmer iqtree mafft ncbi-blast+ \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --no-cache-dir \
    "numpy>=2,<3" "scipy>=1.14,<2" "biopython>=1.84,<2" \
    "rdkit>=2025.3,<2027" "cobra>=0.29,<1" "scanpy>=1.11,<2" \
    "pymc>=5.20,<6" "statsmodels>=0.14,<1" "scikit-learn>=1.6,<2" \
    "openmm>=8.2,<9" "cantera>=3.1,<4"

# See tooling/reasoning/Dockerfile: the file moved into src/reagents/tools/ so
# the Modal executors can import it, and stays import-free so this still works.
COPY src/reagents/tools/tool_runtime.py /opt/reagents/tool_runtime.py
COPY benchmarks/perturbseq_norman/v2/public/training_data.json \
    /opt/reagents/norman_v2_training.json
ENV REAGENTS_NORMAN_TRAINING_PATH=/opt/reagents/norman_v2_training.json
WORKDIR /work
