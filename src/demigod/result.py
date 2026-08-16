"""The DEMI_GOD output contract.

The load-bearing idea: **a DEMI_GOD's output is artifact files on disk.** The
structured result is a *manifest* that points at those files. It is not the
product; it is the index and the self-assessment.

Concretely, after a successful run the volume contains:

    out/<name>/result.json     <- DemiGodResult, written by the agent
    out/<name>/<artifacts...>  <- the actual work

Consequences worth keeping:
  * Recombination (the GOD's job, later) reads files, not transcripts.
  * A crashed agent still leaves whatever artifacts it managed to write, plus a
    failure manifest written by the runner.
  * Nothing about this depends on where the loop ran. RUNNER-INDEPENDENT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

RESULT_FILENAME = "result.json"

Status = Literal["ok", "failed", "timeout"]


class DemiGodResult(BaseModel):
    """Manifest written to `out/<name>/result.json`.

    Field-by-field, this is the locked contract:

      claim         the answer, in this agent's domain, in prose
      confidence    0.0-1.0 self-assessment of `claim`
      evidence      paths backing the claim -- these are what the GOD reads
      method        how it got there, enough for another agent to re-run
      unknowns      what it could not determine (incl. out-of-domain questions)
      blockers      what actively stopped it (missing tool, bad input, ...)
      files         every artifact it produced
      miscellaneous free-form passthrough, mirrors the spec field
    """

    claim: str = Field(..., description="The finding, stated plainly.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list[str] = Field(
        default_factory=list,
        description="Paths relative to out/<name>/, each supporting `claim`.",
    )
    method: str = Field(..., description="How the claim was reached.")
    unknowns: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    files: list[str] = Field(
        default_factory=list,
        description="All artifacts produced, relative to out/<name>/.",
    )
    miscellaneous: dict[str, Any] = Field(default_factory=dict)

    # --- runner-populated envelope. The agent never writes these; the runner
    # overwrites them on read-back so they cannot be faked from inside. ---
    name: str | None = None
    domain: str | None = None
    status: Status = "ok"
    error: str | None = None

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
        cls, *, name: str, domain: str, status: Status, error: str
    ) -> DemiGodResult:
        """Manifest the runner writes when the agent never produced one.

        Confidence 0.0 and the error in `blockers` so that a downstream consumer
        that only looks at the contract fields still sees the failure.
        """
        return cls(
            claim="",
            confidence=0.0,
            evidence=[],
            method="",
            unknowns=[],
            blockers=[error],
            files=[],
            name=name,
            domain=domain,
            status=status,
            error=error,
        )


class ResultMissingError(RuntimeError):
    """The agent exited without writing result.json."""


class ResultInvalidError(RuntimeError):
    """result.json exists but does not match the contract."""


# JSON Schema for the contract, embedded in the system prompt so the agent
# writes a manifest that validates on the first try.
def result_json_schema() -> dict[str, Any]:
    """Schema of the agent-authored subset (envelope fields excluded)."""
    schema = DemiGodResult.model_json_schema()
    for envelope_field in ("name", "domain", "status", "error"):
        schema["properties"].pop(envelope_field, None)
    return schema
