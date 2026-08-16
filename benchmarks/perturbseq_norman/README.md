# Norman Perturb-seq held-out combinations

This benchmark predicts 12 real two-gene CRISPRa pseudobulk responses from the
Norman et al. K562 Perturb-seq screen. Both constituent singles and every
non-held-out condition are training evidence. The task therefore isolates
nonlinear interaction prediction rather than zero-shot gene annotation.

## Leakage controls

- Test pairs are selected by SHA-256 rank of condition labels with the public
  salt `reagents-norman-v1`. Expression is not read by the selection rule.
- Named paper/repository demos are excluded before ranking to reduce model
  memorization.
- The 64-gene panel, all candidate models, CV errors, class threshold, ensemble
  weights, and uncertainty summaries are fitted using training conditions only.
- Raw cells remain in a gitignored local cache. No raw matrix or private fixture
  is mounted into a sandbox.
- Broker tools contain only the generated training artifact and use opaque
  `T01`–`T12` / `F001`–`F064` identifiers.
- `public/freeze_manifest.json` commits to the source, split, public fixtures,
  preparation script, and private truth before any model call.
- Public verification checks only structure. Private scoring starts only after
  every pipeline/control sandbox has terminated.

Source: the processed Norman 2019 H5AD indexed by scPerturb, ultimately derived
from GEO accession GSE133344.

## Run

```bash
python -m benchmarks.perturbseq_norman.safety_probe \
  --output benchmarks/perturbseq_norman/runs/safety-probe.json

python scripts/e2e_live.py --problem perturbseq --domains 4 --turns 12 --broker \
  --result-out benchmarks/perturbseq_norman/runs/pipeline.json

python -m benchmarks.perturbseq_norman.run_controls --samples 4 --turns 12 \
  --output benchmarks/perturbseq_norman/runs/controls.json

python -m benchmarks.perturbseq_norman.run_base \
  --output benchmarks/perturbseq_norman/runs/base.json

python -m benchmarks.perturbseq_norman.run_code_base \
  --output benchmarks/perturbseq_norman/runs/code-base.json

python -m benchmarks.perturbseq_norman.compare \
  benchmarks/perturbseq_norman/runs/pipeline.json \
  benchmarks/perturbseq_norman/runs/controls.json \
  --base benchmarks/perturbseq_norman/runs/base.json \
  --code-base benchmarks/perturbseq_norman/runs/code-base.json \
  --output benchmarks/perturbseq_norman/runs/comparison.json
```

The direct base call receives the frozen public problem but no tools or agent
loop. The Claude Code base receives the same problem and standard local
file/shell utilities, but no Broker lease or domain tools. The first native
control is the prespecified single-Claude
tool-augmented baseline. The arithmetic mean / majority-class result across all
valid independent controls is the deployable multi-Claude comparison. The best
individual score is labeled oracle-only and is never used as a deployable
selector.
