"""THE DEMI_GOD output contract. One shape, every run, success or failure.

This is the consolidation of what were three types:

    demigod.DemiGodResult        claim/confidence/evidence/method/unknowns/...
    reagents.DomainArtifact      domain_name/payload/justification/tool_trace
    reagents.DemigodFailure      domain_name/reason/isolation_violations

Two rules make the merged shape usable by an orchestrator:

1. **Failure is a status, not a type.** `reagents` returned a union
   (`DomainArtifact | DemigodFailure`), so every consumer had to branch on
   `isinstance` before it could read anything. Here a failed DEMI_GOD returns
   the *same* shape with `status != "ok"` and `confidence == 0.0`. One parse
   path, always -- which is what makes runs diffable and reproducible.

2. **The envelope is runner-owned.** `demigod_name`, `domain_name`, `run_id`,
   `status`, `error` are stripped from the schema the agent is shown and are
   overwritten on read-back, so a model cannot self-report success on a run
   that crashed.

The output is still FILES. `claim` and `payload` are a summary and an index
over artifacts written to `out/<name>/`; recombination reads files, not
transcripts. A crashed agent still leaves whatever it managed to write, plus a
failure manifest written by the runner.

RUNNER-INDEPENDENT: nothing here knows where the agent loop ran.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from demigod.schema import validate_payload

RESULT_FILENAME = "result.json"

Status = Literal["ok", "failed", "timeout"]

ENVELOPE_FIELDS = ("demigod_name", "domain_name", "run_id", "status", "error")
"""Runner-owned. Never authored by the agent, never shown in its schema."""


class DemiGodResult(BaseModel):
    """Manifest written to `out/<name>/result.json`."""

    # --- the finding (agent-authored) -------------------------------------

    claim: str = Field(..., description="The finding, stated plainly, in prose.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured result conforming to the domain's artifact_schema. The "
            "machine-readable half of the finding; `claim` is the prose half."
        ),
    )
    method: str = Field(
        ...,
        description="HOW the result was produced -- enough for someone to re-run it.",
    )
    justification: str = Field(
        "",
        description=(
            "WHY the claim follows, argued in the domain's own language. Distinct "
            "from `method` on purpose: an integrator weighing conflicting claims "
            "needs the argument; a human reproducing the work needs the procedure."
        ),
    )

    # --- provenance -------------------------------------------------------

    evidence: list[str] = Field(
        default_factory=list,
        description="Paths under out/<name>/ that specifically support `claim`.",
    )
    files: list[str] = Field(
        default_factory=list, description="Every artifact produced."
    )
    tool_trace: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "One entry per tool call: {tool, input, result}. Authored by the "
            "BROKER once it lands, not by the agent -- the broker sees every call, "
            "so the trace cannot be under-reported by the thing being audited."
        ),
    )

    # --- honesty ----------------------------------------------------------

    unknowns: list[str] = Field(
        default_factory=list,
        description=(
            "What it could not determine, including anything outside its domain. "
            "This is the field an orchestrator reads to plan the next round."
        ),
    )
    blockers: list[str] = Field(
        default_factory=list, description="What actively stopped it."
    )
    isolation_violations: list[str] = Field(
        default_factory=list,
        description="Native-field terms detected in the output. Empty is the norm.",
    )
    miscellaneous: dict[str, Any] = Field(default_factory=dict)

    # --- identity envelope (runner-owned; see module docstring) -----------

    demigod_name: str | None = Field(
        None, description="Infrastructure identity: the slug, sandbox, and out dir."
    )
    domain_name: str | None = Field(
        None,
        description=(
            "Domain identity: DomainSpec.name. Differs from demigod_name, which is "
            "its slugified form -- reagents uses underscores, demigod names forbid "
            "them because the name becomes a directory and a sandbox name."
        ),
    )
    run_id: str | None = None
    status: Status = "ok"
    error: str | None = None

    # --- io ---------------------------------------------------------------

    def write(self, out_dir: str | Path) -> Path:
        """Write this manifest to `<out_dir>/result.json`."""
        path = Path(out_dir) / RESULT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def read(cls, out_dir: str | Path) -> DemiGodResult:
        """Read and validate the manifest from `<out_dir>/result.json`."""
        path = Path(out_dir) / RESULT_FILENAME
        if not path.exists():
            raise ResultMissingError(
                f"no {RESULT_FILENAME} at {path} -- the agent finished without "
                "writing its manifest"
            )
        try:
            return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            raise ResultInvalidError(f"{path} is not a valid DemiGodResult: {e}") from e

    @classmethod
    def failure(
        cls,
        *,
        status: Status,
        error: str,
        demigod_name: str | None = None,
        domain_name: str | None = None,
        run_id: str | None = None,
        isolation_violations: list[str] | None = None,
    ) -> DemiGodResult:
        """The manifest the runner writes when the agent produced none.

        Confidence 0.0 and the error echoed into `blockers`, so a consumer
        reading only the contract fields still sees the failure without having
        to check `status`.
        """
        return cls(
            claim="",
            confidence=0.0,
            method="",
            blockers=[error],
            isolation_violations=list(isolation_violations or []),
            demigod_name=demigod_name,
            domain_name=domain_name,
            run_id=run_id,
            status=status,
            error=error,
        )

    def validate_against(self, artifact_schema: dict[str, Any]) -> list[str]:
        """Check `payload` against a domain's schema. Empty list means valid."""
        return validate_payload(self.payload, artifact_schema)


class ResultMissingError(RuntimeError):
    """The agent exited without writing result.json."""


class ResultInvalidError(RuntimeError):
    """result.json exists but does not match the contract."""


def result_json_schema(artifact_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Schema of the agent-authored subset, for embedding in the system prompt.

    Envelope fields are removed so the agent is never invited to set them. A
    caller-supplied `artifact_schema` replaces the generic `payload` definition
    -- which is how a per-domain output shape rides inside an otherwise fixed
    contract.
    """
    schema = DemiGodResult.model_json_schema()
    for field in ENVELOPE_FIELDS:
        schema["properties"].pop(field, None)
    if artifact_schema:
        schema["properties"]["payload"] = dict(artifact_schema)
    return schema
