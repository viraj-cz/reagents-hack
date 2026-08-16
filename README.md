# Reagents

> **Fourier Transform for Agentic Reasoning:** decompose a problem into
> orthogonal invented domains, solve each independently in isolation, recombine.

A GOD/DEMI_GOD reasoning runtime. GOD invents domain representations, projects
the problem into each, spawns a DEMI_GOD per domain in its own isolated
environment, and maps validated artifacts back to the original problem.

## Two halves, one seam

This repo is the consolidation of two independently-built pieces.

| | `src/reagents/` — the brain | `src/demigod/` — the body |
|---|---|---|
| Owns | plan, transform, seal, integrate | spawn, isolate, execute, collect |
| Key types | `DomainSpec`, `ContextEnvelope`, `InverseMap` | `DemiGodSpec`, image catalog, volume layout |
| Isolation | **semantic** — no native terms reach a demigod | **execution** — one Modal sandbox per demigod |
| Tools | authorization: leases, brokers, risk tiers | installation: pinned images, secrets, smoke tests |

Neither subsumes the other, and the split is deliberate: `reagents` decides
*what* a demigod should reason about, `demigod` decides *where and how* it runs.

**The seam is one call** — `reagents/god/orchestrator.py`, in `_spawn()`:

```python
return await god.runtime.run(envelope, pack, guard=guard)
```

Everything upstream (plan → transform → seal-check → operator approval →
lease minting) is GOD-side and independent of how a demigod executes.

## Layout

```
src/reagents/
  contracts.py     DomainSpec, ContextEnvelope, DomainArtifact, InverseMap, ...
  isolation.py     semantic sealing: native_terms, find_leaks, assert_sealed
  god/             planner, transformer, orchestrator, integrator
  demigod/         in-process runtime (the toy/local execution path)
  tools/           registry, broker + capability leases, MCP, containers
  llm/             LLMClient protocol; anthropic + scripted implementations
src/demigod/
  spec.py          DemiGodSpec: name, domain, tools, problem, files, misc
  result.py        the output contract + result.json manifest
  registry/        the CLOSED tool set + agent-facing docs
  images.py        pre-baked Modal image catalog + resolve_image()
  layout.py        two volumes per run: shared/ read-only, out/<name>/ writable
  runner/          >>> THE SEAM <<< who drives the agent loop
  spawn.py         spawn_demigod() / the CLI
scripts/
  bake.py  smoke_test.py  preflight_live.py  bootstrap.sh
```

## Quickstart

```bash
uv sync
```

Spawn a single DEMI_GOD (no GOD involved):

```bash
uv run spawn-demigod --spec examples/smoke-demigod.json --shared examples/data/transactions.csv
```

Run the GOD loop (defaults to the scripted LLM; `--live` for real inference):

```bash
uv run reagents
```

Offline checks — no Modal account or API key needed:

```bash
uv run pytest && uv run ruff check . && uv run ruff format --check .
```

Before the first live spawn of the day, and after any Modal SDK bump:

```bash
uv run python scripts/preflight_live.py
```

**This project uses `uv` exclusively.** `uv.lock` is committed — do not add it
to `.gitignore`. Python 3.12 is pinned in `.python-version` to match
`PYTHON_VERSION` in the sandbox image, so you develop against the same
interpreter the agent runs on.

## Setup

```bash
uv run modal token new                             # caller-side auth
uv run modal secret create demigod-anthropic ANTHROPIC_API_KEY=sk-ant-...
uv run python scripts/bake.py                      # pre-bake images
```

Copy `.env.example` to `.env` for GOD-side and sponsor-tool credentials. Note
the scope rule: **a DEMI_GOD receives none of these.** Its Anthropic key arrives
as a mounted Modal Secret; tool credentials belong to the broker.

## Isolation, in both senses

**Semantic** (`reagents/isolation.py`): a `DomainProblem` must contain no
native-field text. `assert_sealed()` runs after transform and before spawn;
violations are recorded in `OrchestrationTrace.leaks`.

**Execution** (`demigod/layout.py`): two Modal Volumes per run. `shared/` is
mounted read-only everywhere; each sandbox mounts only its own `<name>` subpath
of the out volume, so no agent has a path to a sibling's output. Verified live.

> Modal forbids mounting one Volume at two locations in a single sandbox
> (`InvalidError: The same Volume cannot be mounted in multiple locations for
> the same function`), which is why there are two.

## Constraints learned the hard way

Each of these cost a failed run; they are recorded so the next person meets
them as documentation.

- **The Agent SDK does not bundle the `claude` CLI.** It shells out to a binary
  found via `shutil.which("claude")`. Every image installs Node 22 (NodeSource —
  Debian's apt nodejs is 18.x, and `@anthropic-ai/claude-code` needs >= 22) plus
  the pinned CLI, and runs `claude --version` as a build step so a broken image
  fails the *bake*.
- **`sandbox.open()` is retired server-side.** Use `sandbox.filesystem`
  (`write_text`/`read_text`) — note data comes first.
- **`IS_SANDBOX=1` is required.** Modal sandboxes run as root, and the CLI
  refuses `--dangerously-skip-permissions` under root. The refusal surfaces as a
  bare `ProcessError` naming neither the flag nor root.
- **`claude --version` passing does not mean the CLI works.** It passed on an
  image where every real query failed. `preflight_live.py` runs `claude -p`.

## Cost

Modal compute for these sandboxes is cents. **The dominant cost is Anthropic
tokens**, one call per agent turn. `max_turns` is the lever that matters;
`examples/smoke-demigod.json` is the deliberately cheap spec (12 turns).

## Adding a tool

Don't freehand it — run `.claude/skills/add-tool-to-registry/SKILL.md`. It
walks: check it isn't already registered → write the pinned entry → decide
whether it fits an existing image → write the agent-facing usage doc → add and
**run** the smoke test.
