# Reagents: how the whole system works

Reagents reasons over biology problems by projecting them into invented
representation domains, solving inside those domains with isolated demigods, and
translating artifacts back into the original field.

This file is the system context for the three branches that make up the
hackathon stack. It describes the assembled whole and the remaining explicit
boundaries between its components.

## The three layers

```
TxBench-PP eval  -->  God (orchestrate)  -->  Demigods (isolated workers)
viraj/env             God-spawn               spawn-agents-environment
     |                      |                         |
     |                      |                         |
 run_agent(task, work_dir)  DomainSpec -> DemiGodSpec  Modal sandbox
     |                      |                         |
     |<----- EVAL_ANSWER ---+<----- result.json ------+
```

| Branch | Role | Integration surface |
|---|---|---|
| `viraj/env` | Benchmark harness (TxBench-PP). Scores JSON answers. | `run_agent(task, work_dir) -> dict \| str` in `agent.py` |
| `God-spawn` | God. Invents domains, transforms, binds tools, integrates. | `God.solve(NativeProblem) -> NativeSolution` |
| `spawn-agents-environment` | One demigod in one Modal sandbox. No God here. | `spawn_demigod(DemiGodSpec, run_id) -> DemiGodResult` |

God owns projection and integration. The spawn branch owns one isolated worker.
The env branch owns scoring. Do not collapse those jobs.

Intended eval path:

1. `viraj/env` calls `run_agent(task, work_dir)`.
2. God turns that into a `NativeProblem` (statement = task, files = `work_dir/data`).
3. God invents orthogonal domains and projects the complete problem into each.
   Inverse maps stay on God.
4. God emits `DemiGodSpec`s and calls `spawn_demigod` N times with one `run_id`, seeding `shared/` from `work_dir/data`.
5. Each sandbox writes `out/<name>/result.json`.
6. God reads those manifests, inverse-maps, integrates, and returns the eval JSON inside `<EVAL_ANSWER>`.

God supports both runtimes: `DemigodRuntime` is the deterministic local/test
path, while `SandboxDemigodRuntime` adapts the sealed envelope into a
`DemiGodSpec` and calls `spawn_demigod` in Modal.

## Core idea

God does not solve the native biology problem. It treats reasoning like a
Fourier transform: project into a foreign language, work with limited tools, invert.

- **God** sees the original problem, invents domains, transforms, leases tools, integrates.
- **Demigods** never see the original writeup, other demigods, credentials, or the rest of the catalog. Each one lives in a single representation and returns artifacts.
- **Orthogonality** is about representation language (proof, spectral, rewrite, information, geometry, constraint), not biology subfields or task slices. Every artifact is a complete alternative solution to the same objective.
- **Sponsor tools** (Paperclip, Phylo/Biomni, Proto, Benchling) are native-field I/O. God may use them to ingest or emit. A demigod should not get a biology vendor pack unless that *is* the invented language, and even then only exact IDs.

## God loop (`God-spawn`)

Entry: `God.solve(problem)` in `src/reagents/god/orchestrator.py`.

```
NativeProblem
  -> Planner.plan          invent DomainSpecs, critic, regenerate collisions
  -> Transformer.forward   DomainProblem + InverseMap (God-only)
  -> build_envelope        sealed ContextEnvelope (no transform_prompt)
  -> _spawn                policy gates + bind lease + DemigodRuntime
  -> Integrator.integrate  InverseMap + artifacts -> NativeSolution
```

God’s functions (it has no other jobs):

| Function | Module | Job |
|---|---|---|
| `Planner.invent` | `god/planner.py` | Invent *n* domains: name, 1–2 axes, language, 2–4 catalog tool IDs, artifact schema, abstract forbidden rules. |
| `structural_critic` / `llm_critic` | same | Reject shared primary axes, tool Jaccard > 0.3, paraphrased languages, unknown tools. |
| `Planner.plan` | same | `registry.load_deferred()` then invent/critic up to 4 rounds. |
| `Transformer.forward` | `god/transformer.py` | Project into domain symbols only. Retry on native-name leaks. |
| `God.build_envelope` | `god/orchestrator.py` | Entire demigod-visible world. Strips God’s transform notes. |
| `_spawn` | same | Write / high-risk approval, `registry.bind`, run the worker. |
| `Integrator.integrate` | `god/integrator.py` | Translate artifacts back. Conflict resolution, not concatenation. |

Public surface: `God.__init__`, `build_envelope`, `solve`. After a run, `god.last_trace` holds specs, envelopes, inverse maps, artifacts, failures, leaks.

Axes (seating labels, not domains): `topology`, `conservation`, `dynamics`, `geometry`, `information`, `causality`, `scale`, `symmetry`, `stochasticity`. No two accepted specs may share a primary axis.

### Objects

- `NativeProblem` — original field: statement, entities, constraints, question.
- `DomainSpec` — invented world. Language is free; axes and tool IDs are not.
- `DomainProblem` — complete problem already in that language. Its projection manifest accounts for every source input, objective, constraint, and required output. No native-field text.
- `InverseMap` — `s1 -> glucose`. God only. Never put in an envelope.
- `ContextEnvelope` — spawn payload: sealed spec, domain problem, tool *schemas*, artifact schema, budget, forbidden rules.
- `CapabilityLease` — immutable authority for one run: exact tool IDs, max calls, wall time, write flag.
- `DomainArtifact` — complete candidate output (payload, justification, certificate, tool_trace).
- `NativeSolution` — translated answer, per-domain contributions, conflicts, gaps.

Isolation: `IsolationGuard` holds native terms on God’s side and scans envelope + artifact. Those terms are not written into the envelope (that would leak). Demigods start a **fresh** message list. No shared memory. No demigod-to-demigod channel.

## How tools are given to subagents

God sees the catalog. A demigod gets a snapshot of 2–4 IDs, then a lease. From the demigod, that set cannot grow.

1. Catalog must already contain the IDs. Local builtins are always registered. Containers load if `REAGENTS_ENABLE_CONTAINERS=1`. Paperclip/Biomni are **deferred** until `plan()` calls `load_deferred()`.
2. Planner writes `DomainSpec.tool_ids` from `registry.ids()`. God does not invent executables.
3. Envelope gets `ToolSpec` schemas only: id, description, JSON schema, access, risk. No callables, no secrets, no URLs.
4. `_spawn` refuses write tools unless `God(..., approved_write_tools={...})` contains those exact IDs. High-risk interpreters need `approved_high_risk_tools`.
5. `registry.bind(...)` mints a `CapabilityLease` and `BoundToolPack`. Broker is the only path to an executor.
6. Unknown id → `UnboundToolError`. Over budget / expired / write without flag → `ToolPolicyError`.

Example: Phylo plus a local tool is not “give them Phylo.” Discovery registers `biomni.<remote_name>`. God lists `["biomni.run_workflow", "simplify"]` on the spec. Operator allowlists the write ID. Bind freezes that pair. The MCP client uses God’s env header; the demigod only sees JSON.

Lease is immutable **by broker policy**. `CapabilityLease` is documented as immutable; the type is not `frozen`. The demigod cannot `register` or expand `tool_ids`. God may grow the catalog before planning (`load_deferred`); it does not re-bind mid-run.

### Catalog (God-spawn)

Always local: `build_graph`, `rewrite_edge`, `find_cycles`, `cut`, `match_motif`, `simplify`, `solve`, `dimensional_check`, `simulate`, `sample`, `entropy`, `compress`, `embed`, `distance`.

If containers enabled: `formal.lean_check`, `formal.z3_solve`, `biology.sequence_stats`, `chemistry.rdkit_descriptors`, `design.proto_check`, `reasoning.python`, `biology.python`, `engineering.python`, `design.proto_run`.

If MCP enabled: `paperclip.*` from `https://paperclip.gxl.ai/mcp`, `biomni.*` from `https://mcp.phylo.bio/mcp` (cap 24 each, optional allowlists). Cursor OAuth for those servers does **not** populate this catalog. Headless God still needs `PAPERCLIP_API_KEY` / `BIOMNI_MCP_AUTHORIZATION` to discover, or planning continues with local tools only.

The spawn-agent package registry is separate and closed for packages such as
`pandas`. Callable tools take the Broker path instead: God publishes the exact
capability lease, the demigod receives only its URL and lease ID, and the Broker
executes and audits every call. Free-text names fail before a billed sandbox.

## Demigod spawn (`spawn-agents-environment`)

There is no God on that branch. A human or God calls:

```python
from demigod import DemiGodSpec, spawn_demigod

result = spawn_demigod(spec, run_id="run-1")
```

One spec → one `modal.Sandbox` → artifacts on disk. Isolation is filesystem:

- `demigod-run-<run_id>-shared` — read-only input, same view for every demigod.
- `demigod-run-<run_id>-out` mounted at `out/<name>/` — that agent cannot see siblings.

`DemiGodSpec`: `name`, `domain`, `tools` (registry keys), `problem.{context,goal,success_criteria}`, `files` (relative to `shared/`), resource knobs.

`DemiGodResult` (`out/<name>/result.json`): `claim`, `confidence`, `evidence`, `method`, `unknowns`, `blockers`, `files`. Runner overwrites `name`, `domain`, `status`, `error` so the agent cannot fake success.

Recombination reads **files**, not transcripts. The runner seam (`inside` vs `outside` sandbox) is not final; nothing outside `runner/` should import `claude_agent_sdk`.

## Benchmark (`viraj/env`)

[TxBench-PP](https://benchmarks.bio/txbench): verifiable small-molecule preclinical pharmacology. Each public eval is a workflow snapshot plus a question whose answer is a small JSON object. No partial credit.

```bash
uv sync
uv run latch login
uv run python main.py --list
uv run python main.py CTRL01_no_cc1_gate_for_crizotinib_hits
uv run python main.py --all --out results
```

`load_eval` splits public `Task` from private grader spec. Only the task reaches `run_agent`. Workspace is a temp dir with eval files hardlinked into `data/`. Ground truth stays in the runner process.

Return the answer dict, or text containing `<EVAL_ANSWER>...</EVAL_ANSWER>`. This repo has 12 public evals; the leaderboard is 100, withheld. Public scores are harness checks, not official numbers.

God plugs in by replacing `agent.py`’s `run_agent` body. Do not rewrite `runner.py` / `grader.py`.

## Assembled integration surfaces

| | God | Demigod / Broker | Bench |
|---|---|---|---|
| Input | `NativeProblem` + complete projection manifest | sealed `DemiGodSpec` | public task and files only |
| Worker | local or `SandboxDemigodRuntime` | one Modal sandbox per demigod | evaluator process |
| Tools | catalog, approval policy, immutable lease | leased HTTPS Broker calls plus closed image packages | no evaluator key in prompts |
| Isolation | semantic seal + inverse maps retained by God | private output volume, scoped egress, no Modal token | private files remain evaluator-only |
| Output | compared complete candidates + native verification | files + `DemiGodResult` + Broker-authored trace | structured answer and private score |

## How to run this branch

```bash
./scripts/bootstrap.sh
uv run reagents                 # scripted God, toy pathway, no API key
uv run reagents --live          # needs ANTHROPIC_API_KEY
uv run reagents tools list
uv run reagents tools doctor
uv run pytest
```

`--live` uses Anthropic for invent/transform/demigod/integrate. Scripted mode proves envelopes seal and axes stay distinct.

Write / high-risk tools stay denied unless constructed explicitly:

```python
God(
    llm,
    approved_write_tools={"biomni.run_workflow"},
    approved_high_risk_tools={"biology.python"},
)
```

Paperclip/Biomni in **Cursor** are OAuth (`https://paperclip.gxl.ai/mcp`, `https://mcp.phylo.bio/mcp`). That is not the same socket God binds.

## Non-goals

- Demigods talking to each other.
- God inventing new executable tools.
- Demigods installing packages or holding credentials.
- Treating Paperclip/Biomni as a default demigod pack.
- Using public TxBench-PP scores as a leaderboard result.
- Mixing the spawn runner seam into God or the registry.
