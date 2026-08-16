"""The tool registry: a CLOSED set of tools a DEMI_GOD may be given.

Why closed: the GOD is a language model choosing tools. If tool names were free
text it would confidently ask for `polars` or `pandas2` or `dataframes`, and the
failure would surface deep inside the sandbox as an ImportError with no
attribution. Here it surfaces at spec-validation time, before anything is
spawned, with a list of what actually exists.

Adding a tool is a documented procedure, not an edit: see
`.claude/skills/add-tool-to-registry/SKILL.md`. Author once, reuse forever.

RUNNER-INDEPENDENT. A registry entry describes what the tool *is* -- how it is
installed, what secrets it needs, how to tell the agent to use it, how to prove
it works. None of that changes if the agent loop moves outside the sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "docs"


@dataclass(frozen=True)
class ToolEntry:
    """One tool in the closed set.

    The five fields are the five things that go wrong with a tool, in order:
    it isn't installed, it has no credentials, the agent doesn't know how to
    call it, and nobody checked that it works.
    """

    key: str
    """Stable identifier. What the GOD writes in `DemiGodSpec.tools`."""

    display_name: str
    """Human label for logs and prompts."""

    install: tuple[str, ...]
    """pip requirement specifiers, pinned. Consumed by `demigod.images` when
    baking an image -- NOT installed per-spawn."""

    secrets: tuple[str, ...] = ()
    """Env var names the tool needs at runtime. Resolved to Modal Secrets by
    `demigod.images.required_secret_names`. Empty for offline tools."""

    apt: tuple[str, ...] = ()
    """System packages, if any. Also baked into the image."""

    smoke_test: tuple[str, ...] = ()
    """A command, as argv, that exits 0 iff the tool works in the image. Run
    against a candidate image by `scripts/smoke_test.py`; the point is that a
    broken image fails on a machine you control rather than mid-run."""

    doc_file: str = ""
    """Filename in registry/docs/. Its contents are spliced into the DEMI_GOD
    system prompt when this tool is selected. Written for the *agent*, not for
    a human: what it can do, the idioms, the traps."""

    tags: tuple[str, ...] = field(default_factory=tuple)
    """Free-form, for the GOD's tool-selection heuristics later. Not validated."""

    @cached_property
    def usage_doc(self) -> str:
        """The agent-facing usage doc. Empty string if none authored."""
        if not self.doc_file:
            return ""
        path = DOCS_DIR / self.doc_file
        if not path.exists():
            raise RegistryError(
                f"tool {self.key!r} declares doc_file={self.doc_file!r} but "
                f"{path} does not exist"
            )
        return path.read_text(encoding="utf-8").strip()


class RegistryError(ValueError):
    """Raised for unknown tool keys or malformed entries."""


# The closed set lives in entries.py so this module stays pure interface.
from demigod.registry.entries import REGISTRY  # noqa: E402


def all_keys() -> list[str]:
    """Every valid tool key, sorted. Show this in every error message."""
    return sorted(REGISTRY)


def get(key: str) -> ToolEntry:
    """Look up one tool. Raises RegistryError with the full valid set."""
    try:
        return REGISTRY[key]
    except KeyError:
        raise RegistryError(
            f"unknown tool {key!r}. The registry is a closed set; free-text tool "
            f"names are not accepted. Valid keys: {all_keys()}. To add a tool, "
            f"follow .claude/skills/add-tool-to-registry/SKILL.md"
        ) from None


def validate_tool_keys(keys: list[str]) -> list[str]:
    """Reject unknown keys and duplicates. Returns the keys unchanged.

    Called from `DemiGodSpec` validation, so a bad tool name fails before a
    sandbox is ever created.
    """
    unknown = [k for k in keys if k not in REGISTRY]
    if unknown:
        raise RegistryError(
            f"unknown tool key(s): {unknown}. The registry is a closed set; "
            f"free-text tool names are not accepted. Valid keys: {all_keys()}. "
            f"To add a tool, follow .claude/skills/add-tool-to-registry/SKILL.md"
        )
    if len(set(keys)) != len(keys):
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        raise RegistryError(f"duplicate tool key(s): {dupes}")
    return keys


def resolve(keys: list[str]) -> list[ToolEntry]:
    """Validated keys -> entries, in the order requested."""
    validate_tool_keys(keys)
    return [REGISTRY[k] for k in keys]


__all__ = [
    "REGISTRY",
    "RegistryError",
    "ToolEntry",
    "all_keys",
    "get",
    "resolve",
    "validate_tool_keys",
]
