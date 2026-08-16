"""The image GOD runs in. NOT the demigod image, and that is the point.

THE ASYMMETRY IS A SAFETY PROPERTY, not an oversight:

                        modal client   reagents   Anthropic SDK   claude CLI
    god_image()              yes          yes          yes            no
    demigod images.CATALOG   NO           NO           no             yes

* **No modal client in a demigod.** Modal credentials are workspace-wide:
  whoever holds them can spawn sandboxes, list volumes, and read any output in
  the workspace. GOD holds a token because GOD is the trusted orchestrator. A
  DEMI_GOD does not -- and without the client installed, a demigod that somehow
  *obtained* a token still could not spawn a sandbox or mount a sibling's
  volume. Two independent barriers, not one.
* **No `reagents` in a demigod.** `reagents` contains the planner prompts, the
  transform prompts, and the inverse maps -- the machinery that decides what an
  agent is allowed to know and how to translate its answer back. Shipping it
  into the container would hand the agent the key to the seal it is under.
  `demigod/images.py` ships `add_local_python_source("demigod")` and only that;
  `tests/test_package_boundary.py` fails if that ever drifts.
* **No `claude` CLI in GOD's image.** GOD's loop is plain Anthropic API calls
  (`reagents/llm/anthropic_client.py`), not an agent loop. Skipping Node 22 and
  the CLI is the difference between a ~200MB image that builds in seconds and
  the multi-minute NodeSource bake every demigod image pays.

Everything here is PINNED to what `uv.lock` resolved, for the same reason
`demigod/images.py` pins: two GOD sandboxes launched a week apart must run the
same software, or the lockfile is decoration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from demigod.images import PYTHON_VERSION

if TYPE_CHECKING:  # keep `modal` out of the import path for pure-logic tests
    import modal

MODAL_CLIENT = "modal==1.5.4"
"""GOD spawns DEMI_GOD sandboxes from inside its own sandbox. Proven to work by
`scripts/preflight_nested.py`; without this line it fails as
`ModuleNotFoundError: No module named 'modal'`, which reads misleadingly like
nested spawning being unsupported."""

ANTHROPIC_SDK = "anthropic==0.122.0"
"""GOD's own planner/transformer/integrator calls. Lives in the optional `llm`
extra locally, so a plain `uv sync` omits it -- but inside the image it is not
optional, it IS the loop."""

PYDANTIC = "pydantic==2.13.4"
"""Every contract on both sides of the seam. Matches demigod's AGENT_RUNTIME
pin, so a DemiGodResult serialized by a demigod validates in GOD."""

MCP_CLIENT = ("mcp==1.29.0", "httpx==0.28.1")
"""Sponsor MCP catalogs, for DISCOVERY only.

`reagents.tools.mcp` is imported by `default_registry()` whenever
REAGENTS_ENABLE_MCP is set, so without these a GOD sandbox with the flag on
would fail the import and plan against the local tools alone -- which is what a
GOD sandbox was silently doing: "no remote catalogs configured", while the same
code on a laptop reported one. GOD needs the catalog to plan with; the calls
themselves are the broker's sponsor tier, which pins these separately."""

GOD_PIP = (MODAL_CLIENT, ANTHROPIC_SDK, PYDANTIC, *MCP_CLIENT)

GOD_LOCAL_SOURCES = ("reagents", "demigod", "broker", "godbox", "benchmarks")
"""All five packages, because GOD is the one place they meet: it reasons
    with `reagents`, spawns with `demigod`, grants tools through `broker`,
    reports with `godbox`, and loads closed benchmark verifiers. None of these
    additional packages enter a DEMI_GOD image."""

GOD_SOURCE_IGNORE = ["**/private/**", "**/data/source/**"]
"""Evaluator-only benchmark fixtures must not exist in GOD's filesystem.

A LIST, and the type is load-bearing. Modal converts ignore patterns with
`elif isinstance(ignore, list): ignore = FilePatternMatcher(*ignore)`
(modal/mount.py). A tuple fails that check and is passed through as though it
were already the predicate, so the first file Modal tests raises
`TypeError: 'tuple' object is not callable` -- from inside mount resolution,
naming neither this constant nor the image. Every GOD sandbox creation failed
that way, 1.4s in, with a message that points at Modal's internals."""

FORBIDDEN_IN_DEMIGOD_IMAGE = ("reagents", "broker", "godbox", "benchmarks")
"""Asserted offline in tests/test_package_boundary.py. Named here so the rule
sits next to the reasoning for it rather than only in a test file."""


def god_image() -> modal.Image:
    """Materialize GOD's image.

    Modal content-hashes image definitions, so repeated calls with an unchanged
    definition are a cache hit rather than a rebuild.

    `ignore=[]` matches `demigod/images.py`: `add_local_python_source` defaults
    to `ignore=NON_PYTHON_FILES`, which silently drops the registry's
    agent-facing `.md` docs. GOD does not read those, but `demigod` is shipped
    here too and a half-populated copy of it is a trap for whoever next changes
    the runner.
    """
    import modal

    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .pip_install(*GOD_PIP)
        .env(
            {
                "REAGENTS_ENABLE_CONTAINERS": "1",
                "REAGENTS_ENABLE_NORMAN_BENCHMARK": "1",
            }
        )
        .add_local_python_source(*GOD_LOCAL_SOURCES, ignore=GOD_SOURCE_IGNORE)
    )
