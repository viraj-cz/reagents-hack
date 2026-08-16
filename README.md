# Reagents

Reagents is a God/demigod reasoning runtime for translating biology problems into
orthogonal invented domains, solving within those constrained domains, and mapping
validated artifacts back to the original problem.

## Tool architecture

Tools are installed or connected once at the God layer. A demigod never installs a
package and never receives a credential. Immediately before planning, God may lazily
discover remote MCP schemas. It then issues each demigod an exact, time-limited
capability lease containing only the selected tool IDs and a maximum call count.
Write-capable tools are denied unless the lease explicitly permits writes.
The application operator grants that authority by passing exact audited IDs through
`God(..., approved_write_tools={"provider.tool_name"})`; neither the planner nor a
demigod can add to this set.
General-purpose container interpreters are separately gated with
`approved_high_risk_tools={"biology.python"}` (or another exact ID).

The default catalog includes small local symbolic, graph, geometry, simulation, and
information tools. Optional providers add:

- Paperclip and Phylo/Biomni through deferred MCP discovery.
- Lean 4, Z3, NumPy/SciPy/SymPy, NetworkX, Pint, CVXPY, and python-control.
- BLAST+, HMMER, MAFFT, IQ-TREE, Biopython, RDKit, COBRApy, Scanpy, PyMC, and OpenMM.
- Cantera for engineering/physical-system reasoning.
- Proto in a pinned, isolated design-language image.

Heavy or conflict-prone tools live in containers. Remote tools remain on their MCP
servers. Lightweight Python dependencies may also be installed on the host.

## Bootstrap

Requirements are Python 3.11+, `uv`, and optionally Docker or Podman. The project
pins Python 3.12 because its scientific wheels have the broadest macOS compatibility.

```bash
./scripts/bootstrap.sh
```

To install the scientific Python stack on the host as well:

```bash
./scripts/bootstrap.sh --host-science
```

To build all isolated images (this is large, chiefly because of mathlib):

```bash
./scripts/bootstrap.sh --containers
```

Inspect what is available without making network connections:

```bash
uv run reagents tools doctor
uv run reagents tools list
```

## Sponsor MCP credentials

Copy `.env.example` to `.env` and fill secrets locally. `.env` is ignored by Git.
Export the values into the process environment before starting Reagents; the program
does not automatically source the file.

Paperclip uses `PAPERCLIP_API_KEY`. Phylo/Biomni uses OAuth for Codex; authenticate
the project entry with:

```bash
codex mcp login biomni
codex mcp list
```

For the standalone God Python process, set `BIOMNI_MCP_AUTHORIZATION` to the complete
Authorization header value supplied by Phylo, such as `Bearer <token>`. This value is
resolved only at connection time and is never placed in an envelope or prompt.

Enable deferred discovery and test the configured connections with:

```bash
REAGENTS_ENABLE_MCP=1 uv run reagents tools doctor --connect
```

Use `PAPERCLIP_MCP_ALLOWED_TOOLS` and `BIOMNI_MCP_ALLOWED_TOOLS` as comma-separated
exact remote names to reduce discovery to an audited allowlist. Exact names can be
copied from the connected doctor output.

## Isolated tools

`compose.yaml` builds three versioned images: `reagents/reasoning-core`,
`reagents/biology-core`, and `reagents/proto-design`. At runtime, containers have no
network, use a read-only root filesystem, get only a temporary `/tmp`, receive JSON
on stdin, and return JSON on stdout. Capabilities are dropped, privilege escalation
is disabled, and CPU, memory, and process counts are bounded.

```bash
REAGENTS_ENABLE_CONTAINERS=1 uv run reagents tools list
```

## Run and test

```bash
uv run reagents
uv run pytest
```

Live LLM execution additionally requires the `llm` extra and `ANTHROPIC_API_KEY`.
