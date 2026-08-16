# DEMI_GOD spawning mechanism

> Fourier Transform for Agentic Reasoning: decompose a problem into orthogonal
> domain components, solve each independently, recombine.

**Scope: this repo builds the spawn mechanism only.** There is no GOD agent
here — no decomposition, no fan-out, no synthesis. The GOD is, for now,
*whatever calls the spawn script with a valid spec*. A human writes the spec
today; a GOD writes it later. Nothing below changes when that happens.

One DEMI_GOD = one `modal.Sandbox`, one domain, one output directory.

## The one command

```bash
uv sync
uv run spawn-demigod --spec examples/pandas-demigod.json --dry-run
```

Then, once Modal auth and the images are set up (see Setup):

```bash
uv run spawn-demigod --spec examples/pandas-demigod.json
```

`--dry-run` validates the spec, resolves the image, and stops before creating a
sandbox. Use it first; it catches every error that is cheap to catch.

**This project uses `uv` exclusively.** `uv.lock` is committed — do not add it
to `.gitignore`. It is what makes every collaborator, and every image bake,
resolve the identical dependency set. Python 3.12 is pinned in `.python-version`
to match `PYTHON_VERSION` in the sandbox image, so you develop against the same
interpreter the agent runs on. `uv sync` fetches that interpreter for you.

Programmatically, the whole integration surface a future GOD needs:

```python
from demigod import DemiGodSpec, spawn_demigod

result = spawn_demigod(spec, run_id="run-1")  # -> DemiGodResult
```

## Layout

```
src/demigod/
  spec.py        DemiGodSpec: name, domain, tools, problem, files, miscellaneous
  result.py      DemiGodResult: the output contract + result.json manifest
  registry/      the CLOSED tool set (entries.py) + agent-facing docs (docs/)
  images.py      pre-baked image catalog + resolve_image()
  layout.py      volume layout: shared/ read-only, out/<name>/ writable
  prompt.py      builds the DEMI_GOD system prompt from a spec
  runner/        >>> THE SEAM <<< who drives the agent loop
  entrypoint.py  what runs inside the sandbox today
  spawn.py       the CLI / spawn_demigod()
scripts/
  bake.py        pre-bake the image catalog (run after registry changes)
  smoke_test.py  prove each registry tool works in its image
.claude/skills/add-tool-to-registry/   the procedure for adding a tool
```

## The five locked-in pieces

**1. Input spec.** `DemiGodSpec` — name, domain, tools, problem
(context/goal/success criteria), files, miscellaneous. A DEMI_GOD is told
nothing about its siblings or about recombination; that ignorance is what keeps
the components orthogonal.

**2. Output is files.** A DEMI_GOD's product is artifacts written to
`out/<name>/`. The structured result is a *manifest* (`result.json`) indexing
them: `claim`, `confidence`, `evidence`, `method`, `unknowns`, `blockers`,
`files`, `miscellaneous`. The transcript is discarded. Recombination reads
files.

The runner owns the envelope fields (`name`, `domain`, `status`, `error`) and
overwrites them on read-back, so an agent cannot self-report success on a run
that crashed.

**3. Closed tool registry.** The GOD selects tool keys from
`demigod.registry`. Free-text names are rejected at spec validation with the
list of valid keys — not as an ImportError inside a live sandbox. Each entry
carries key, display name, pinned install spec, required secrets, an
agent-facing usage doc, and a smoke test. One real entry today: `pandas`.
Add more via the `add-tool-to-registry` skill.

**4. Pre-baked images.** `resolve_image(tool_keys)` picks the smallest catalog
image covering the requested tools and **hard-errors if none does** — it never
falls back to building on the fly, because a per-spawn `pip install` costs the
agent 30-90s and makes two nominally-identical DEMI_GODs non-identical. With
one tool registered there is effectively one image; the seam is what matters.

**5. One sandbox per DEMI_GOD.** Alive for that agent's whole task — not
per-call, not pooled. Bounded by `timeout` (wall clock) and `idle_timeout`
(the runaway-cost guarantee). Terminated unconditionally in a `finally`.

## Volume layout

**Two** Modal Volumes per run:

```
demigod-run-<run_id>-shared      read-only input, same view for every DEMI_GOD
demigod-run-<run_id>-out
  <name>/                        that agent's writable output dir
```

Enforced at mount time, not by prompt instruction:

```python
volumes = {
    "/run/shared": shared_vol.with_mount_options(read_only=True),
    "/run/out": out_vol.with_mount_options(sub_path=name),
}
```

Because each sandbox mounts only its own `<name>` subpath of the out volume, an
agent has no path by which to reach a sibling's output. Isolation is a
filesystem property. **Verified live**: `shared/` is populated and unwritable,
`out/` is writable and shows only that agent's directory.

> **Why two volumes and not one with two sub_paths?** Modal forbids it:
> `InvalidError: The same Volume cannot be mounted in multiple locations for the
> same function`. The original one-volume design failed on first contact. All
> the properties survived the fix; only the names changed.

Input data reaches a DEMI_GOD **only** via `--shared`, because `shared/` is
read-only at every mount and so cannot be populated from inside:

```bash
uv run spawn-demigod --spec examples/smoke-demigod.json --shared examples/data/transactions.csv
```

`spawn` warns if the spec's `files` name anything that was not uploaded — a
guard against paying for a sandbox whose agent spends every turn hunting for a
CSV that was never there.

## The runner seam (read this before restructuring anything)

**The inside-vs-outside-sandbox decision is NOT final.** Today the agent loop
runs *inside* the sandbox. It may move to the caller.

Everything that defines *what a DEMI_GOD is* is deliberately independent of
that choice:

| Module | Runner-independent? |
|---|---|
| `spec.py`, `result.py` | yes — the contracts |
| `registry/` | yes — what tools are, how to use them |
| `images.py`, `layout.py` | yes — environment and paths |
| `prompt.py` | yes — what the agent is told |
| `runner/`, `entrypoint.py` | **no — this is the seam** |

The invariant: **nothing outside `runner/` imports `claude_agent_sdk`, and
nothing outside `runner/` knows where the loop runs.** An `if inside:` branch
appearing in `prompt.py` or `images.py` means the design is breaking.

Both runners satisfy one contract:

```python
class Runner(Protocol):
    def run(self, spec: DemiGodSpec, run_id: str) -> DemiGodResult: ...
```

`runner/outside.py` is a documented stub. Its docstring lists exactly what the
swap buys (API key never enters the sandbox; loop survives sandbox death) and
costs (every tool call becomes a network round-trip; `can_use_tool` has to
reimplement Read/Write/Edit against `sandbox.open`). Implementing it should not
require editing any other module. Switch with `--runner outside`.

## Setup

```bash
uv sync                                            # env + pinned deps
uv run modal token new                             # caller-side auth
uv run modal secret create demigod-anthropic ANTHROPIC_API_KEY=sk-ant-...
uv run python scripts/bake.py                      # pre-bake images
uv run python scripts/smoke_test.py                # prove the registry
```

Checks (no Modal account or API key needed — these run offline):

```bash
uv run pytest && uv run ruff check . && uv run ruff format --check .
```

The `demigod-anthropic` secret exists **only because the loop runs inside the
sandbox**. An outside runner reads the key from the caller's environment and the
secret disappears.

### What every image must contain

The Agent SDK does **not** bundle the `claude` CLI — `claude_agent_sdk` shells
out to a `claude` executable it finds via `shutil.which("claude")`. So
`pip install claude-agent-sdk` alone produces an image where every DEMI_GOD dies
on its first turn with `CLINotFoundError`. `images._install_agent_runtime`
therefore installs, in this order:

1. Node.js 22 from NodeSource — `@anthropic-ai/claude-code` declares
   `engines.node >= 22`, and Debian bookworm's apt `nodejs` is 18.x, so apt
   alone is **not** enough.
2. `@anthropic-ai/claude-code`, pinned.
3. `claude --version`, so a broken image fails the *bake* rather than a live,
   already-billed sandbox.
4. The SDK and pydantic, pinned to the versions in `uv.lock`.

All of this is a cost of running the loop *inside* the sandbox. Under an outside
runner the entire function is deleted and the sandbox needs no Node, no CLI, no
SDK, and no API key.

### Cost control

Modal compute for these sandboxes is cents; **the dominant cost is Anthropic API
tokens**, one call per agent turn against your key. So `max_turns` is the lever
that matters. `examples/smoke-demigod.json` is the deliberately cheap spec:
600s wall, 120s idle, 12 turns, 1 cpu, 2 GiB. Use it before the full
`pandas-demigod.json` (1800s / 30 turns).

Every sandbox is terminated in a `finally`, and `idle_timeout` is the backstop
if the caller dies.

### Status — verified live on Modal

- `demigod-data` image bakes clean; inside it: node v22.23.2, `claude` 2.1.233
  at `/usr/bin/claude`, pandas 2.2.3, SDK 0.2.139, registry docs present
- Volume mounts verified: shared read-only + populated, out writable + isolated
- 24 offline tests, ruff clean

Not yet exercised: **a full agent run.** That needs the `demigod-anthropic`
secret.

## Adding a tool

Don't freehand it. Run the skill:

```
.claude/skills/add-tool-to-registry/SKILL.md
```

It walks: check it isn't already registered → write the pinned entry → decide
whether it fits an existing pre-baked image or needs a new one → write the
agent-facing usage doc (including the traps) → add and **run** the smoke test.
