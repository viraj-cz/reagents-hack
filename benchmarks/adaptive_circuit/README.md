# Adaptive synthetic circuit workflow benchmark

This benchmark is split into an agent-visible input and an evaluator-only key.

- `public/question.json` is the native problem God receives.
- `public/observations.csv` is the only file uploaded to the shared Modal volume.
- `private/expected.json` is read only after the run. Never upload `private/`,
  mention its contents in a prompt, or derive domain success criteria from it.

The withheld `Z_KO` minute-8 measurement makes answer leakage detectable: an
agent must infer it from the other interventions rather than read it from the
input table.
