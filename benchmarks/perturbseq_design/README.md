# Perturb-seq diverse batch design

This benchmark asks for a ten-combination follow-up batch from 63 held-out
two-gene conditions in the real Norman K562 CRISPRa Perturb-seq screen. The
objective rewards both large non-additive residuals and coverage of distinct
training-defined response programs, subject to a maximum of two uses per gene.

The source H5AD hash, expression-independent split salt, 64-feature panel,
training-only PCA/program centers, strength normalization, coverage bonus, and
exact mixed-integer oracle are frozen before any model call. Candidate outcomes,
utilities, and the oracle live only in `private/expected.json`; broker tools and
demigods receive `public/training_data.json`.

```bash
UV_CACHE_DIR=/tmp/reagents-uv-cache uv run --extra biology --extra reasoning \
  python -m benchmarks.perturbseq_design.prepare
UV_CACHE_DIR=/tmp/reagents-uv-cache uv run --extra biology --extra reasoning \
  python -m benchmarks.perturbseq_design.run_baselines \
  --output benchmarks/perturbseq_design/runs/baselines.json
UV_CACHE_DIR=/tmp/reagents-uv-cache uv run --extra llm python scripts/e2e_live.py \
  --problem perturbdesign --domains 4 --turns 28 --broker \
  --result-out benchmarks/perturbseq_design/runs/pipeline.json
```
