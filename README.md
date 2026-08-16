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

Two further packages complete the picture. `src/broker/` — the
**TOOLBOX_BROKER** — exists because those two columns disagreed about what a
tool is. `src/godbox/` answers the remaining question — *where and how does GOD
itself run?* — by putting the orchestrator in its own long-lived Modal sandbox.
See [Three sandbox types](#three-sandbox-types).

Both depend on the two halves; neither half depends on them. That direction is
a security boundary, not a style preference: `demigod` is the only package
shipped into a DEMI_GOD's container, and the moment it can reach `reagents`,
the agent's own container holds the planner prompts and inverse maps that
constrain it.

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
  spec.py          DemiGodSpec: name, domain, tools, toolbox, problem, files
  result.py        the output contract + result.json manifest
  registry/        the CLOSED tool set + agent-facing docs
  images.py        pre-baked Modal image catalog + resolve_image()
  layout.py        two volumes per run: shared/ read-only, out/<name>/ writable
  egress.py        the sandbox's outbound domain allowlist
  runner/          >>> THE SEAM <<< who drives the agent loop
  toolbox/         the DEMI_GOD half of the broker: protocol, client, `toolbox` CLI
  spawn.py         spawn_demigod() / the CLI
src/broker/        >>> THE BROKER SANDBOX <<<
  grants.py        durable lease state; the call counter IS the audit log
  router.py        the request path. Plain ASGI, no Modal, tested offline
  dispatch.py      which tools run inline and which get their own container
  modal_store.py   GrantStore over modal.Dict + modal.Queue
  service.py       the modal.App: router endpoint + one executor per tool class
  session.py       GOD's API: grant -> collect_trace -> revoke
src/godbox/        >>> THE GOD SANDBOX <<<
  status.py        modal.Dict status channel: phase, heartbeat, per-demigod rows
  images.py        god_image() -- modal client + Anthropic SDK + all 3 packages
  layout.py        GodRequest + where things live in GOD's sandbox
  launch.py        OUTSIDE: create the sandbox, hand over the task, detach
  entrypoint.py    INSIDE: drive God.solve(), report, self-terminate
  cli.py           `god launch|status|watch|list|logs|followup|terminate`
scripts/
  bake.py  smoke_test.py  bootstrap.sh
  preflight_live.py  preflight_toolbox.py  preflight_god.py  preflight_nested.py
```

## Three sandbox types

`GOD` spawns. `DEMI_GOD` reasons in one invented domain. `BROKER` executes tools
on behalf of a lease, and audits every call.

The broker exists because `reagents` tool ids resolve to in-process Python
callables behind a capability lease, while a DEMI_GOD is a different process on
a different machine. Those cannot be reconciled by renaming, so before the
broker every demigod was told its whole toolset was unreachable and reasoned
with nothing.

```
demigod sandbox                     broker (modal.Function, autoscaled)
+----------------------+            +--------------------------------+
| toolbox call X -i f  | --HTTPS--> | router  (asgi, cheap tools)    |
|   Bearer lease_...   |            |   |-- inline: LOCAL tools      |
| NO modal token       |            |   '-- remote: exec_* per class |
+----------------------+            +--------------------------------+
```

**Trust asymmetry is the whole design.** Modal tokens are workspace-wide: a
demigod holding one could spawn sandboxes and read every sibling's output
volume. So it holds a URL and a lease id, and nothing else — and it cannot be
given more by accident, because nothing shipped into its image imports Modal.
That is asserted, not assumed: `tests/test_toolbox_isolation.py`.

**The lease is the credential.** `Authorization: Bearer <lease_id>`, where the
id is `CapabilityLease.lease_id`. The existing `ToolBroker` already enforces
max-calls, wall time and write permission, and it remains the only code path
from a request to an executor — the router just seeds its two counters from
durable state so they survive an autoscaled replica.

**The broker authors `tool_trace`.** It sees every call, including the ones it
refused, so `DemiGodResult.tool_trace` cannot be under-reported by the thing
being audited.

Deploy it, then point GOD at it:

```bash
uv run modal deploy src/broker/service.py
uv run python scripts/preflight_toolbox.py    # cheap live check, no tokens
```

```python
from broker.session import modal_session
from reagents.demigod.sandbox_runtime import SandboxDemigodRuntime

runtime = SandboxDemigodRuntime(
    run_id="run-1",
    toolbox=modal_session(),  # grant -> collect_trace -> revoke, per demigod
    restrict_egress=True,  # pin sandbox egress to the API + the broker
)
```

During development, `uv run modal serve src/broker/service.py` and pass its URL
as `TOOLBOX_BROKER_URL` instead of deploying.

## Quickstart

```bash
uv sync
```

Spawn a single DEMI_GOD (no GOD involved):

```bash
uv run spawn-demigod --spec examples/smoke-demigod.json --shared examples/data/transactions.csv
```

Run the GOD loop in this process (defaults to the scripted LLM; `--live` for
real inference):

```bash
uv run reagents
```

Or run GOD in **its own long-lived Modal sandbox**, so the run survives closing
the laptop:

```bash
uv run god launch --problem simple --domains 2 --turns 12   # returns in seconds
uv run god status <run_id>      # a snapshot, from anywhere
uv run god watch  <run_id>      # follow it to completion
uv run god list                 # what did I leave running?
uv run god logs   <run_id>      # GOD's stdout, out of the live sandbox
uv run god terminate <run_id>
```

GOD spawns its DEMI_GODs from inside that sandbox (nested spawning: see
`scripts/preflight_nested.py`). Progress goes to a `modal.Dict`, which is
readable the instant it is written; artifacts and `solution.json` are uploaded
to the run's out volume at the end. **The two channels are not
interchangeable** -- see below.

Offline checks — no Modal account or API key needed:

```bash
uv run pytest && uv run ruff check . && uv run ruff format --check .
```

Before the first live spawn of the day, and after any Modal SDK bump:

```bash
uv run python scripts/preflight_live.py      # one DEMI_GOD sandbox, no agent
uv run python scripts/preflight_toolbox.py   # Dict, Queue, asgi_app, egress
uv run python scripts/preflight_god.py       # one GOD sandbox, no agent
uv run python scripts/preflight_nested.py    # can a sandbox spawn a sandbox?
```

All four are cheap — one small sandbox each, no agent loop, zero Anthropic
tokens. They exist because the offline suite cannot catch a server-side API
change: `sandbox.open()` passed every local check right up until the server
retired it.

**This project uses `uv` exclusively.** `uv.lock` is committed — do not add it
to `.gitignore`. Python 3.12 is pinned in `.python-version` to match
`PYTHON_VERSION` in the sandbox image, so you develop against the same
interpreter the agent runs on.

## Setup

```bash
uv run modal token new                             # caller-side auth
uv run modal secret create demigod-anthropic ANTHROPIC_API_KEY=sk-ant-...
# Only for `god launch`: lets GOD spawn DEMI_GODs from inside its own sandbox.
uv run modal secret create demigod-modal-token \
    MODAL_TOKEN_ID=ak-... MODAL_TOKEN_SECRET=as-...
uv run python scripts/bake.py                      # pre-bake images
```

Modal credentials are workspace-wide, so `demigod-modal-token` goes to GOD and
**never** to a DEMI_GOD. That is enforced twice: a demigod does not mount the
secret, and the demigod images do not install the `modal` client at all, so a
leaked token would still be unusable. `tests/test_package_boundary.py` fails if
either barrier drifts.

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
- **`Volume.commit()` does not work in a Sandbox.** It raises `RuntimeError:
  commit() can only be called on a mounted volume inside a container` -- it is
  a Modal *Function* API. A Sandbox's mount writes are flushed only when the
  sandbox terminates, so an artifact written through a mount is invisible for
  the entire run. `Volume.batch_upload()` works from inside a Sandbox and is
  visible immediately, which is why GOD does not mount the out volume at all.
- **A `modal.Dict` is the only live cross-container channel.** Status, phase,
  and heartbeat go there precisely because a volume cannot carry them. Polling
  `out/` mid-run shows an empty directory, which reads as "the agents produced
  nothing" -- the most expensive misdiagnosis in this repo's history.
- **`idle_timeout` does not fire while a command is running**, even a detached
  one with both streams sent to `DEVNULL`. Verified: a sandbox with
  `idle_timeout=30` ran a 50s unattended process to completion and was
  collected ~30s after it *ended*. That is what makes a detached GOD safe.
- **A detached exec must not use the default `StreamType.PIPE`.** An unread
  pipe applies backpressure and eventually hangs the process nobody is
  watching. GOD redirects to a file in the sandbox and uses `DEVNULL`.
- **Modal's blocking API inside `async def` warns and stalls the loop.** Use
  `.aio` (`await d.put.aio(...)`), or keep the call outside `asyncio.run`.

## Cost

Modal compute for these sandboxes is cents. **The dominant cost is Anthropic
tokens**, one call per agent turn. `max_turns` is the lever that matters;
`examples/smoke-demigod.json` is the deliberately cheap spec (12 turns).

## Adding a tool

Don't freehand it — run `.claude/skills/add-tool-to-registry/SKILL.md`. It
walks: check it isn't already registered → write the pinned entry → decide
whether it fits an existing image → write the agent-facing usage doc → add and
**run** the smoke test.

### God and subagent progress stream

Use `--trace` for concise, user-facing updates while the analysis runs:

```bash
uv run reagents --trace
```

The terminal separates God's orchestration from each subagent's scoped work. It shows
the selected representations, allowed capabilities and budgets, bounded tool activity,
artifact-level reasoning summaries, and the translation back to the original problem.
It does not dump subprocess logs, raw tool payloads, or private model chain-of-thought.

```text
◆ God is analyzing the problem
  Checking capabilities — 14 local capabilities available

◇ Choosing useful representations
  Selected 3 complementary approaches:
    • stoichiometric_flow — conservation; tools: simplify, dimensional_check
    • catalytic_dag — topology; tools: build_graph, cut
    • rate_orbit — dynamics; tools: simulate, sample

◇ Running scoped analyses
  ┌─ Subagent catalytic_dag started
  │ catalytic_dag · Scope — axis=topology; language=directed acyclic graph
  │ catalytic_dag · Allowed tools — build_graph, cut
  │ catalytic_dag · → Using build_graph (inputs: edges, nodes)
  │ catalytic_dag · ← build_graph returned fields: edges, nodes
  │ catalytic_dag · Reasoning summary — The graph is a single chain; removing v3 disconnects v4.
  │ catalytic_dag · Finding — Inflating e1 cannot increase flow into v4 while e3 remains the cut.
  └─ Subagent catalytic_dag complete — artifact validated; tool_calls=2
  ✓ God accepted catalytic_dag's artifact

◇ Synthesizing the findings
  Translating 3 domain artifacts back into the original biology problem

◆ Analysis complete
  The integrated answer is ready at 86% confidence
```

Messages are length-bounded and redact recognized credential fields. Tool calls show
argument names, and results show their shape rather than dumping their contents.
Programmatic callers can pass a `TerminalTracer` or another `TraceSink` to
`God(..., tracer=...)`; `TerminalTracer(show_subagents=False)` restores a God-only
view.

Live LLM execution additionally requires the `llm` extra and `ANTHROPIC_API_KEY`.
