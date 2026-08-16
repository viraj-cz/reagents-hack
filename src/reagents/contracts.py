"""Shared contracts between God, the transform, and isolated demigods."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from demigod.result import DemiGodResult


class Axis(str, Enum):
    """Fixed vocabulary of orthogonal reasoning axes.

    God invents languages at runtime but must seat each domain on 1–2 of these.
    """

    TOPOLOGY = "topology"
    CONSERVATION = "conservation"
    DYNAMICS = "dynamics"
    GEOMETRY = "geometry"
    INFORMATION = "information"
    CAUSALITY = "causality"
    SCALE = "scale"
    SYMMETRY = "symmetry"
    STOCHASTICITY = "stochasticity"


class Budget(BaseModel):
    """Per-demigod ceilings. Deliberately generous -- see below.

    THESE WERE SILENTLY COSTING ARTIFACTS. Measured live on the glycolysis
    problem at the old values: GOD planned three domains and only ONE produced
    a usable artifact. One demigod died with "exhausted its step budget without
    an artifact" at `max_steps=8`, and a second returned a complete `payload`
    whose top-level `claim` was missing -- the shape a run truncated mid-answer
    leaves behind. Re-run with these values, every planned domain produced an
    artifact and both failures disappeared, at roughly equal total token spend.

    A cap that stops a demigod mid-answer does not save the run's cost; it
    spends the whole budget and throws the result away. These are set high
    enough that a healthy run never reaches them, which makes hitting one
    informative -- it now means something is genuinely wrong rather than that
    the problem was slightly larger than the default.

    `max_tokens` is the one that is NOT generous, and not by choice: Opus 4.8
    allows 128K output tokens, but only on a streaming request, and
    `AnthropicLLM` calls `messages.create` non-streaming. 16K is the ceiling
    the SDK permits without converting that call to `messages.stream()`.
    """

    max_tokens: int = 16_000
    max_steps: int = 100
    wall_time_s: float = 3600.0
    max_tool_calls: int = 500


class ToolProvider(str, Enum):
    LOCAL = "local"
    MCP = "mcp"
    CONTAINER = "container"


class ToolAccess(str, Enum):
    READ = "read"
    COMPUTE = "compute"
    WRITE = "write"


class ToolSpec(BaseModel):
    """Schema a demigod is allowed to see for a bound tool."""

    id: str
    description: str
    parameters_schema: dict[str, Any]
    output_schema: dict[str, Any] = Field(default_factory=dict)
    namespace: str = "generic"
    provider: ToolProvider = ToolProvider.LOCAL
    access: ToolAccess = ToolAccess.COMPUTE
    side_effects: list[str] = Field(default_factory=list)
    cost_class: str = "free"
    latency_class: str = "interactive"
    defer_loading: bool = False


class CapabilityLease(BaseModel):
    """Immutable authority minted by God for one demigod run."""

    lease_id: str
    subject_id: str
    tool_ids: list[str]
    max_calls: int = 16
    wall_time_s: float = 60.0
    allow_write: bool = False


class NativeProblem(BaseModel):
    """The original problem in the initial representative field."""

    id: str
    statement: str
    entities: list[str] = Field(default_factory=list)
    sensitive_terms: list[str] = Field(
        default_factory=list,
        description=(
            "Additional native labels that must not cross the semantic seal, "
            "including dataset categories and trajectory identifiers."
        ),
    )
    constraints: list[str] = Field(default_factory=list)
    question: str
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured native-field evidence available to God. The transformer "
            "must project every top-level input; raw inputs never enter a demigod "
            "envelope."
        ),
    )
    required_outputs: list[str] = Field(
        default_factory=list,
        description="Native deliverables every complete candidate must provide.",
    )
    answer_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional JSON schema for NativeSolution.structured_answer.",
    )


class DomainSpec(BaseModel):
    """An invented representation space. Language is free; axes and tools are not."""

    name: str
    axes: list[Axis] = Field(min_length=1, max_length=2)
    language: str
    transform_prompt: str
    tool_ids: list[str] = Field(min_length=2, max_length=4)
    artifact_schema: dict[str, Any]
    forbidden: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _identifier_name(cls, value: str) -> str:
        if not value or not value[0].isalpha() or not all(c.isalnum() or c == "_" for c in value):
            raise ValueError("domain name must be an identifier like stoichiometric_flow")
        return value.lower()

    @field_validator("tool_ids")
    @classmethod
    def _unique_tools(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("tool_ids must be unique")
        return value

    @property
    def primary_axis(self) -> Axis:
        return self.axes[0]


class ProjectionManifest(BaseModel):
    """Machine-checkable proof that a projection retained the whole problem.

    IDs are deliberately semantic-free (``source:input:02``, ``objective:03``),
    so this manifest can enter the sealed envelope. The transformer validates
    the sets against God's native problem before any demigod is spawned.
    """

    source_ids: list[str] = Field(default_factory=list)
    objective_ids: list[str] = Field(default_factory=list)
    output_ids: list[str] = Field(default_factory=list)
    source_map: dict[str, str] = Field(default_factory=dict)
    objective_map: dict[str, str] = Field(default_factory=dict)
    output_map: dict[str, str] = Field(default_factory=dict)
    information_losses: list[str] = Field(default_factory=list)


class DomainProblem(BaseModel):
    """Problem already projected into a domain. Must contain no native-field text."""

    domain_name: str
    representation: dict[str, Any]
    task: str
    notation_guide: str
    projection_manifest: ProjectionManifest = Field(
        default_factory=ProjectionManifest
    )


class InverseMap(BaseModel):
    """God-only bookkeeping: domain symbols back to native entities."""

    domain_name: str
    symbol_to_native: dict[str, str]


class ContextEnvelope(BaseModel):
    """Everything a demigod is allowed to see. Nothing else."""

    domain: DomainSpec
    problem: DomainProblem
    tools: list[ToolSpec]
    artifact_schema: dict[str, Any]
    budget: Budget = Field(default_factory=Budget)
    forbidden: list[str] = Field(default_factory=list)


# DomainArtifact and DemigodFailure used to live here. They are now one type,
# `demigod.result.DemiGodResult`, re-exported so `reagents.contracts` remains
# the single place to look for the orchestration contracts.
#
# Why they merged: returning a union forced every consumer to isinstance-branch
# before it could read anything, and the two types shared most of their meaning.
# A failed demigod is now the same shape with `status != "ok"`, `confidence`
# 0.0, and the reason in `blockers` (plus `isolation_violations` when a seal
# was breached). One parse path, success or failure.
#
# The direction of the import matters: `demigod` never imports `reagents`, so
# only `demigod` is shipped into the sandbox image and GOD's planner prompts and
# inverse maps stay unreadable by the agent they constrain.
__all__ = [
    "Axis",
    "Budget",
    "CapabilityLease",
    "ContextEnvelope",
    "DemiGodResult",
    "DomainProblem",
    "DomainSpec",
    "InverseMap",
    "NativeProblem",
    "NativeSolution",
    "OrchestrationTrace",
    "ProjectionManifest",
    "ToolAccess",
    "ToolProvider",
    "ToolSpec",
]


class NativeSolution(BaseModel):
    problem_id: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    domain_contributions: dict[str, str] = Field(default_factory=dict)
    conflicts: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    structured_answer: dict[str, Any] = Field(default_factory=dict)
    verification: dict[str, Any] = Field(default_factory=dict)


class OrchestrationTrace(BaseModel):
    """God-side record of a run. Never sent to a demigod."""

    specs: list[DomainSpec] = Field(default_factory=list)
    inverse_maps: list[InverseMap] = Field(default_factory=list)
    envelopes: list[ContextEnvelope] = Field(default_factory=list)
    # Both are DemiGodResult now; they are split by `status`, not by type.
    artifacts: list[DemiGodResult] = Field(default_factory=list)
    failures: list[DemiGodResult] = Field(default_factory=list)
    leaks: list[str] = Field(default_factory=list)
    solution: NativeSolution | None = None
    direct: bool = False
    """God answered without inventing a domain or spawning anything.

    Every other field is empty on such a run, which is indistinguishable by
    shape from a run whose planning collapsed -- and the two must not be
    confused, because one has an answer and the other does not. Callers that
    treat "no artifacts" as failure (exit codes, run records) check this."""
