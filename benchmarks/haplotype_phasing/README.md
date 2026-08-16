# HG004 long-read phasing benchmark

This benchmark asks the system to infer a complete diploid phase across 49
heterozygous SNVs from 25 noisy PacBio reads (477 allele calls). The fixture is
real Genome in a Bottle HG004 / NA24143 data, cropped and downsampled in the
official WhatsHap test suite.

The benchmark is designed to test a change of representation, not work
decomposition. Every demigod must solve all 49 binary variables. God sees the
native biological problem, then sends each worker only a sealed alternative
coordinate system. The worker-facing `binary.*` tools expose opaque symbols
and words in four forms:

- a signed weighted constraint graph;
- a zero-field Ising energy;
- a sparse weighted error-correcting code;
- weighted XOR logic.

The tools also provide public objective evaluation, local flip deltas, and
connectivity checks. They never return a candidate assignment and never expose
sample names, genomic coordinates, bases, read/variant labels, provenance, or
the held-out phase. God alone owns the inverse map back to `V001..V049`.

## Freeze and scoring

`public/freeze_manifest.json` commits to the question, observations, and source
files. The answer commitment is kept separately in `private/freeze_seal.json`;
even its digest is not put in an agent prompt. `private/expected.json` contains
the official WhatsHap fixture phase plus an independently encoded exact optimum
of the public weighted-discordance objective. All files are frozen before model
calls. Global haplotype complement is scored identically.

The private score combines:

- 45% exact-objective optimality;
- 35% switch accuracy against the fixture reference;
- 20% orientation-invariant phase accuracy.

The official fixture reference and the exact public optimum agree on all 49
positions up to global orientation.

## Reproduce the fixture

From a checkout of <https://github.com/whatshap/whatshap>:

```bash
uv run python benchmarks/haplotype_phasing/prepare.py \
  --source-dir /path/to/whatshap/tests/data/pacbio
```

Preparation requires the `biology` and `reasoning` extras (`pysam` and Z3).
Runtime workers do not need either package: the deterministic Broker owns the
generated public evidence and computations.

## Matched run

```bash
mkdir -p benchmarks/haplotype_phasing/runs

uv run python scripts/e2e_live.py \
  --problem phasing --domains 4 --broker --require-broker \
  --result-out benchmarks/haplotype_phasing/runs/pipeline.json

uv run python -m benchmarks.haplotype_phasing.run_base \
  --output benchmarks/haplotype_phasing/runs/standard-claude.json

uv run python -m benchmarks.haplotype_phasing.run_native_agent \
  --output benchmarks/haplotype_phasing/runs/native-code-agent.json

uv run python -m benchmarks.haplotype_phasing.compare \
  benchmarks/haplotype_phasing/runs/pipeline.json \
  benchmarks/haplotype_phasing/runs/standard-claude.json \
  benchmarks/haplotype_phasing/runs/native-code-agent.json \
  --output benchmarks/haplotype_phasing/runs/comparison.json
```

The benchmark contract raises the phasing run to 30 turns, 64 Broker calls,
and a one-hour tool lease per worker. Generic smoke-test defaults cannot
silently shorten it. A caller can still choose a larger budget.

The standard control is one Claude API call with the full native observation
matrix and no tools. The second control is one native-field Claude Code agent
with the same matrix and local Python, but no God, inverse transform, Broker,
or alternative representation.
