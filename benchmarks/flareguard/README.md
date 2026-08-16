# FlareGuard: complete-objective biological design benchmark

FlareGuard asks the system to design a research-only living diagnostic that
recognizes sustained chronic intestinal inflammation without activating on
healthy, transient, injury, or pathogen-associated signals.

It is deliberately a **change-of-coordinates** benchmark. Every demigod must
solve the complete design objective in an alternative representation such as
temporal logic, circuit synthesis, control theory, coding theory, or reachability.
Agents are not assigned separate biological subtasks.

## Visibility boundary

- `public/question.json` defines the native problem and required answer schema.
- `public/trajectories.csv`, `parts.json`, and `compatibility.json` are hydrated
  into God's `NativeProblem.inputs`.
- God transforms every input into each invented representation. The raw public
  files are **not mounted** in demigod sandboxes, because that would bypass the
  semantic seal and expose the native representation.
- `private/heldout_trajectories.csv` and `private/expected.json` are evaluator
  only. God's image excludes every `private/` benchmark directory; `grade.py`
  loads these files locally only after the run.

## Complete artifact contract

Each demigod artifact must contain:

- `candidate_solution`: one full circuit in invented notation;
- `constraint_results`: a result for every projected obligation;
- `certificate`: enough structure to check feasibility, robustness, and
  minimality after inverse mapping;
- `conclusion`: the complete decision in the invented language.

God compares these alternative complete candidates, inverse-maps the selected
design, and runs the public deterministic verifier in the original biological
domain. The private grader then repeats the simulation on withheld trajectories
and checks global minimality.

## Commands

Freeze the model and test the unchanged native wording with one short call
before starting sandboxes:

```bash
uv run python -m benchmarks.flareguard.safety_probe \
  --model claude-opus-4-8 \
  --output benchmarks/flareguard/runs/safety-probe.json
```

The full pipeline should only run after the offline suite and both live
preflights pass:

```bash
uv run modal deploy src/broker/service.py
uv run python scripts/preflight_toolbox.py
uv run python scripts/preflight_live.py
uv run python scripts/e2e_live.py \
  --problem flareguard \
  --domains 4 \
  --turns 12 \
  --model claude-opus-4-8 \
  --broker \
  --approve-high-risk-tool reasoning.python \
  --approve-high-risk-tool engineering.python \
  --approve-high-risk-tool biology.python \
  --approve-high-risk-tool design.proto_run \
  --result-out benchmarks/flareguard/runs/flareguard-live.json
```

Those approvals are exact capability IDs, not a blanket grant. God may bind
only a subset to any one demigod, and the Broker still enforces that demigod's
lease, call budget, lifetime, and access mode.

Run four native-representation controls with the same model, turn cap, Modal
sandbox, public inputs, egress restriction, and exact Broker capabilities:

```bash
uv run python -m benchmarks.flareguard.run_controls \
  --samples 4 \
  --turns 12 \
  --model claude-opus-4-8 \
  --tool reasoning.python \
  --tool engineering.python \
  --tool biology.python \
  --tool design.proto_run \
  --output benchmarks/flareguard/runs/flareguard-controls.json
```

The first control is the single-agent baseline. The exact-design majority is
the deployable independent-sampling control; the evaluator-only best sample is
reported only as a diagnostic upper bound and is never used for selection.

Score the saved record without another model call:

```bash
uv run python -m benchmarks.flareguard.grade \
  benchmarks/flareguard/runs/flareguard-live.json
uv run python -m benchmarks.flareguard.compare \
  benchmarks/flareguard/runs/flareguard-live.json \
  benchmarks/flareguard/runs/flareguard-controls.json
```

The records include the public-input hash, pinned models, elapsed time, God
token usage, demigod SDK usage/cost, projected envelopes, exact grants, and
Broker-authored tool traces. They contain no private fixture contents.

Do not place a copy of `private/` under `public/`, `shared/`, a prompt, or a run
record.
