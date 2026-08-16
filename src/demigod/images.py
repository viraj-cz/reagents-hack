"""Pre-baked image catalog and resolver.

Design decision (locked): images are **pre-baked**, never built per spawn.

The reason is latency and determinism. A per-spawn `pip install` costs 30-90s of
the agent's wall clock, can fail on a network blip, and means two DEMI_GODs
nominally holding the same tool can hold different versions of it. Instead a
small catalog of images is baked ahead of time; the resolver's only job is to
pick one that covers the requested tool keys, and to hard-error if none does.

With only `pandas` in the registry there is essentially one image. That is
fine -- the SEAM is the point. When the catalog grows, `resolve_image` starts
making a real choice (smallest covering image) and nothing else has to change.

RUNNER-INDEPENDENT: an image is a description of an execution environment. It is
identical whether the agent loop runs inside it or merely shells into it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from demigod.registry import REGISTRY, ToolEntry, resolve

if TYPE_CHECKING:  # keep `modal` out of the import path for pure-logic tests
    import modal

PYTHON_VERSION = "3.12"
"""Matches .python-version, so the sandbox interpreter is the one collaborators
develop against."""

# --- Agent runtime -----------------------------------------------------------
#
# Present in every image because the agent loop currently runs *inside* the
# sandbox. If the loop moves outside, delete this whole section and the
# `_install_agent_runtime` call below -- that is the only image-level
# consequence of the inside/outside decision.
#
# Versions are PINNED, and pinned to whatever `uv.lock` resolved. An unpinned
# image spec means two DEMI_GODs baked a week apart run different agent
# software, which quietly breaks the reproducibility the lockfile exists to
# guarantee. TODO: derive these from uv.lock at bake time instead of by hand;
# `scripts/bake.py` should fail loudly if they drift.
AGENT_RUNTIME = (
    "claude-agent-sdk==0.2.139",
    "pydantic==2.13.4",
)

CLAUDE_CODE_VERSION = "2.1.233"
"""The `claude` CLI, installed via npm.

NOT optional and NOT bundled with the SDK. `claude_agent_sdk` shells out to a
`claude` executable found with `shutil.which("claude")` (see the SDK's
_internal/transport/subprocess_cli.py). Without this the very first `query()`
raises CLINotFoundError and every DEMI_GOD dies on startup.
"""

NODE_MAJOR = "22"
"""@anthropic-ai/claude-code declares engines.node >= 22. Debian bookworm's apt
`nodejs` is 18.x, so apt alone is NOT enough -- hence the NodeSource repo."""


def _install_agent_runtime(image: modal.Image) -> modal.Image:
    """Install Node, the `claude` CLI, and the Agent SDK onto an image.

    Ordered before tool installs because these layers are slow and change
    rarely -- a registry edit should not re-download Node.

    This is the single image-level cost of running the loop INSIDE the sandbox.
    Under an outside-the-sandbox runner the whole function is deleted: the
    sandbox would need no Node, no CLI, no SDK, and no API key.
    """
    return (
        image.apt_install("curl", "ca-certificates")
        .run_commands(
            # Debian bookworm's apt nodejs is 18.x; claude-code needs >= 22.
            f"curl -fsSL https://deb.nodesource.com/setup_{NODE_MAJOR}.x | bash -",
            "apt-get install -y --no-install-recommends nodejs",
            "apt-get clean && rm -rf /var/lib/apt/lists/*",
            f"npm install -g @anthropic-ai/claude-code@{CLAUDE_CODE_VERSION}",
            # Fail the BAKE, not the run. Without this a missing CLI surfaces as
            # a CLINotFoundError inside a live, already-billed sandbox.
            "claude --version",
        )
        .pip_install(*AGENT_RUNTIME)
    )


TOOLBOX_BIN = "/usr/local/bin/toolbox"
"""Where the brokered-tool CLI lands in every image.

A shell command, not a Python module invocation, because the caller is an agent
composing bash: `toolbox call x -i args.json` is something it can build up,
echo, pipe and retry, whereas `python -m demigod.toolbox call ...` invites it to
`import demigod.toolbox` and reason about the transport instead of the tool.

Two lines of shim rather than a console-script entry point: images get the
package via `add_local_python_source`, which copies source and never runs pip,
so `[project.scripts]` would produce nothing here.
"""


TOOLBOX_SMOKE_TEST = (TOOLBOX_BIN, "--help")
"""Proves the shim resolves AND that `demigod.toolbox` imports in the image.

Run by `scripts/smoke_test.py` against a built image, NOT during the build.
The build cannot check it: the shim has to be written before
`add_local_python_source`, so at build time there is no `demigod` package for
`--help` to import.

WHY IT IS WORTH RUNNING AT ALL. `claude --version` passing on an image where
every real query died is this repo's own precedent -- a binary existing is not a
binary working. Here the failure mode is `toolbox: command not found` inside a
live, already-billed sandbox, which an agent reads as "I have no tools".
"""


def _install_toolbox_cli(image: modal.Image) -> modal.Image:
    """Put `toolbox` on PATH. One tiny layer, no dependencies, never changes.

    Applied FIRST, before apt and pip and before the source is added. Two
    reasons, and the second is not optional:

    * The shim is two static lines, so as the earliest layer it is a cache hit
      forever -- adding a registry tool does not rewrite it.
    * Modal REJECTS a build step after `add_local_*`:
          InvalidError: An image tried to run a build step after using
          `image.add_local_*` to include local files.
      Verified live, by trying it. The alternative it suggests (`copy=True`)
      would make every edit to our own source rebuild the whole layer instead of
      being mounted at container startup -- a real cost, to move a check that
      belongs in the smoke test anyway.

    Installed in EVERY image, including `demigod-base`. The CLI is inert without
    a grant -- it exits 2 with an explanation -- and an image that lacks it
    cannot be given brokered tools later without a re-bake.
    """
    import shlex

    shim = ("#!/bin/sh", 'exec python -m demigod.toolbox "$@"')
    write = "printf '%s\\n' " + " ".join(shlex.quote(line) for line in shim)
    return image.run_commands(
        f"{write} > {TOOLBOX_BIN}",
        f"chmod +x {TOOLBOX_BIN}",
    )


@dataclass(frozen=True)
class PrebakedImage:
    """One image in the catalog.

    `tool_keys` is the contract: the set of registry tools this image is
    guaranteed to have baked in, verified by their smoke tests.
    """

    name: str
    tool_keys: frozenset[str]
    base_apt: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""

    @property
    def size_rank(self) -> int:
        """Proxy for image size. Fewer tools == smaller == preferred.

        TODO: replace with a measured size in MB once images are actually built;
        tool count is a stand-in that happens to be correct for a nested catalog.
        """
        return len(self.tool_keys)

    def covers(self, requested: set[str]) -> bool:
        return requested.issubset(self.tool_keys)

    def build(self) -> modal.Image:
        """Materialize the modal.Image.

        Modal content-hashes image definitions, so calling this repeatedly with
        an unchanged catalog is a cache hit, not a rebuild. The first call after
        a registry change pays the build cost once -- run `scripts/bake.py` to
        pay it deliberately rather than on a caller's critical path.
        """
        import modal

        entries: list[ToolEntry] = [REGISTRY[k] for k in sorted(self.tool_keys)]

        apt = list(self.base_apt)
        for e in entries:
            apt.extend(e.apt)

        # Tool pip specs only. The agent runtime is installed separately, and
        # first, so that adding a tool to the registry does not invalidate the
        # (slow) Node + CLI layers.
        pip: list[str] = []
        for e in entries:
            pip.extend(e.install)

        image = modal.Image.debian_slim(python_version=PYTHON_VERSION)
        # First: two static lines that never change, so this layer is a cache
        # hit forever. It must also precede add_local_python_source -- Modal
        # rejects any build step after that. See _install_toolbox_cli.
        image = _install_toolbox_cli(image)
        if apt:
            image = image.apt_install(*sorted(set(apt)))
        image = _install_agent_runtime(image)
        if pip:
            image = image.pip_install(*sorted(set(pip)))
        # Ship our own package last, so edits to it do not bust the (expensive)
        # pip layer.
        #
        # `ignore=[]` is load-bearing: add_local_python_source defaults to
        # ignore=NON_PYTHON_FILES, which would silently drop
        # registry/docs/*.md -- the agent-facing tool usage docs. The failure
        # mode is a live agent with no idea how to use its tools, and a
        # traceback pointing at a missing file rather than at this line.
        return image.add_local_python_source("demigod", ignore=[])


# --- The catalog. Ordered by nothing; the resolver sorts. -------------------

BASE = PrebakedImage(
    name="demigod-base",
    tool_keys=frozenset(),
    description="Agent runtime only. For DEMI_GODs that reason without tools.",
)

DATA = PrebakedImage(
    name="demigod-data",
    tool_keys=frozenset({"pandas"}),
    description="Tabular analysis. pandas + numpy + pyarrow.",
)

CATALOG: tuple[PrebakedImage, ...] = (BASE, DATA)


class ImageResolutionError(RuntimeError):
    """No pre-baked image covers the requested tool set."""


def resolve_image(tool_keys: list[str]) -> PrebakedImage:
    """Smallest pre-baked image covering `tool_keys`. Hard-errors otherwise.

    Deliberately does NOT fall back to building an image on the fly. A missing
    combination is a catalog bug that a human should fix once, in the catalog,
    rather than a cost every caller silently pays forever.
    """
    resolve(tool_keys)  # raises RegistryError on unknown keys, with the valid set
    requested = set(tool_keys)

    candidates = [img for img in CATALOG if img.covers(requested)]
    if not candidates:
        raise ImageResolutionError(
            f"no pre-baked image covers tools {sorted(requested)}.\n"
            f"Catalog: "
            + ", ".join(f"{i.name}{sorted(i.tool_keys)}" for i in CATALOG)
            + "\nEither add these tools to an existing image's tool_keys and "
            "re-bake, or add a new PrebakedImage to CATALOG. See "
            ".claude/skills/add-tool-to-registry/SKILL.md step 3."
        )
    return min(candidates, key=lambda i: (i.size_rank, i.name))


def required_secret_names(tool_keys: list[str]) -> list[str]:
    """Env vars the selected tools need, for mapping onto Modal Secrets."""
    names: set[str] = set()
    for entry in resolve(tool_keys):
        names.update(entry.secrets)
    return sorted(names)
