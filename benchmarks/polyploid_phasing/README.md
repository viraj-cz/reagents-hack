# Hard benchmark: real-read tetraploid phasing

This is the differentiation benchmark that replaces the small diploid fixture
as the headline run. It reconstructs four unordered haplotypes at 42 sites
from 18 informative real PacBio reads pooled from HG00514 and NA19240. The
fixture is the official 12 kb WhatsHap polyphase integration case used to test
the algorithm described in *Haplotype threading: accurate polyploid phasing
from long reads*.

The difficulty is structural, not just input length. A candidate must jointly:

- cluster each noisy sparse read among four latent sources;
- reconstruct four complete sequences;
- obey the known alternate-allele dosage at every site;
- handle row-label permutation symmetry;
- cut phase blocks where evidence does not support continuity.

Every demigod solves the complete problem. God keeps the biological problem
and inverse maps; workers receive only opaque `S...`/`W...` symbols through
alternative full representations: signed compatibility clustering, pair-state
tensors, constrained layered flow, or sparse latent words. The `latent.*`
tools expose computations and objective checks, never a candidate solution or
native metadata.

The private fixture includes WhatsHap 2.8 output and an independently encoded
exact optimum of the public latent-factor discordance objective. Scoring is 40%
objective quality, 40% permutation-invariant phase accuracy within reference
blocks, and 20% block-boundary accuracy.

## Rebuild

```bash
python benchmarks/polyploid_phasing/prepare.py \
  --source-dir /path/to/whatshap/tests/data \
  --whatshap /path/to/whatshap
```

## Run

```bash
python scripts/e2e_live.py --problem polyphase --domains 4 --require-broker \
  --result-out benchmarks/polyploid_phasing/runs/pipeline.json

python -m benchmarks.polyploid_phasing.run_base \
  --output benchmarks/polyploid_phasing/runs/standard-claude.json

python -m benchmarks.polyploid_phasing.run_native_agent \
  --output benchmarks/polyploid_phasing/runs/native-code-agent.json

python -m benchmarks.polyploid_phasing.compare \
  benchmarks/polyploid_phasing/runs/pipeline.json \
  benchmarks/polyploid_phasing/runs/standard-claude.json \
  benchmarks/polyploid_phasing/runs/native-code-agent.json \
  --output benchmarks/polyploid_phasing/runs/comparison.json
```

The contract grants each pipeline worker up to 40 turns, 96 Broker calls, and a
90-minute lease. Generic smoke-test limits cannot shorten this benchmark.

The saved comparison is computed only after all three response files exist.
The public objective and private reference never enter a baseline response after
generation, and the held-out phase labels never enter the pipeline, Broker, or
demigod sandboxes.
