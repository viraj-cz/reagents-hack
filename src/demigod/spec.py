"""The DEMI_GOD input spec: everything the GOD supplies to describe one agent.

RUNNER-INDEPENDENT. Nothing here knows or cares whether the agent loop runs
inside or outside the sandbox. See `demigod/runner/__init__.py`.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from demigod.toolbox.protocol import ToolboxGrant

NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,30}[a-z0-9]$")


class Problem(BaseModel):
    """The problem statement, split into the three things an agent actually needs.

    The GOD writes this per-domain. It is the *component* of the decomposed
    problem, not the whole problem -- a DEMI_GOD should never be told about its
    siblings (that is what makes the components orthogonal).
    """

    context: str = Field(..., description="What the agent needs to know to start.")
    goal: str = Field(..., description="One sentence: what to produce.")
    success_criteria: list[str] = Field(
        default_factory=list,
        description="Checkable conditions. The agent self-assesses against these.",
    )


class DemiGodSpec(BaseModel):
    """Complete description of one DEMI_GOD. Serialized to JSON and handed over.

    This is the ONLY thing the GOD needs to construct. Everything downstream
    (image, volume mounts, system prompt, sandbox lifetime) is derived from it.
    """

    name: str = Field(
        ...,
        description="Slug, unique within a run. Names the output dir out/<name>/.",
    )
    domain: str = Field(
        ...,
        description=(
            "The orthogonal axis this agent reasons along, e.g. 'quantitative "
            "time-series analysis'. Shapes the system prompt; the agent is told "
            "to stay inside it and to report anything outside it as an unknown."
        ),
    )
    domain_name: str | None = Field(
        None,
        description=(
            "The domain's own identifier, when one exists upstream (reagents' "
            "DomainSpec.name). Kept alongside `name` because the two differ: "
            "`name` is the slugified infrastructure identity (no underscores -- "
            "it becomes a directory and a sandbox name), while reagents' domain "
            "names are snake_case identifiers. Defaults to `name` when absent."
        ),
    )
    artifact_schema: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "JSON Schema the agent's `payload` must satisfy. Invented per-domain "
            "by the planner; embedded in the system prompt and checked on the way "
            "out. Empty means no structural requirement beyond the fixed contract."
        ),
    )
    tools: list[str] = Field(
        default_factory=list,
        description=(
            "Tool KEYS from the closed registry (demigod.registry). Free text is "
            "rejected at validation time, not at runtime in the sandbox."
        ),
    )
    toolbox: ToolboxGrant | None = Field(
        None,
        description=(
            "Capability lease on the TOOLBOX_BROKER: a URL and a lease id, and "
            "nothing else. Orthogonal to `tools` -- that field names pip "
            "packages baked into this agent's image, while this one names "
            "callables that run on someone else's machine. A demigod can hold "
            "both, either, or neither.\n\n"
            "This is the ONLY credential a DEMI_GOD is given besides its "
            "Anthropic key. In particular it is never given a Modal token: "
            "Modal tokens are workspace-wide, so one would let it spawn "
            "sandboxes and read every sibling's output volume."
        ),
    )
    egress_domains: list[str] | None = Field(
        None,
        description=(
            "Domains this sandbox may reach, passed to Modal as "
            "`outbound_domain_allowlist`. None means unrestricted, which is "
            "Modal's default and this repo's prior behaviour. Build one with "
            "`demigod.egress.allowlist(broker_url)`."
        ),
    )
    problem: Problem
    files: list[str] = Field(
        default_factory=list,
        description=(
            "Paths, relative to shared/, that this agent should look at first. "
            "Advisory: the whole of shared/ is readable regardless."
        ),
    )
    miscellaneous: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Escape hatch. Passed through verbatim into the system prompt as "
            "JSON. Use for hints, constraints, priors, anything not yet modeled."
        ),
    )

    # --- resource knobs (have sane defaults; the GOD rarely sets these) ---

    max_lifetime_s: int = Field(
        3600,
        description="Hard wall-clock cap on the sandbox. Modal `timeout`.",
    )
    idle_timeout_s: int = Field(
        300,
        description=(
            "Hard idle timeout. Modal terminates the sandbox after this long with "
            "no active exec/stdin/TCP. This is the anti-runaway-cost guarantee."
        ),
    )
    max_turns: int = Field(
        40,
        description=(
            "Agent loop turn cap. THE dominant cost lever: Modal compute for a "
            "small sandbox is cents/hour, while each turn is an Anthropic API "
            "call against your key. Lower this first when budgeting."
        ),
    )
    cpu: float = Field(1.0, description="Cores. Sandbox billing is per core-second.")
    memory_mb: int = Field(2048, description="RAM in MiB.")

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        if not NAME_RE.match(v):
            raise ValueError(
                f"name {v!r} must be a lowercase slug (a-z0-9-), 3-32 chars, "
                "starting with a letter -- it becomes a directory name and a "
                "Modal sandbox name."
            )
        return v

    @field_validator("tools")
    @classmethod
    def _known_tools(cls, v: list[str]) -> list[str]:
        # Imported here to keep the module import graph shallow and avoid any
        # chance of a cycle if the registry ever wants to reference the spec.
        from demigod.registry import validate_tool_keys

        return validate_tool_keys(v)

    @field_validator("files")
    @classmethod
    def _relative_files(cls, v: list[str]) -> list[str]:
        for p in v:
            if p.startswith("/") or ".." in p.split("/"):
                raise ValueError(
                    f"files entry {p!r} must be relative to shared/ and must not "
                    "escape it"
                )
        return v
