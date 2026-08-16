"""There is exactly one shape a demigod returns, and both runtimes use it."""

from __future__ import annotations

import ast
import pathlib

from demigod.result import DemiGodResult, result_json_schema
from reagents.toy import ARTIFACT_SCHEMA

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def test_no_second_result_model_exists():
    """A rival model describing a demigod's output is a drift generator.

    `DemigodDraft` was one: `{payload, justification}`, used only by the
    in-process runtime. Because it had no `confidence` field, every in-process
    artifact was hardcoded to 0.5 while sandbox artifacts reported a real
    number -- so the integrator weighed a confident proof exactly like a hedged
    guess. The scripted demigods wrote `confidence` inside the payload, where
    nothing read it. Two shapes, silently disagreeing.

    Asserted structurally rather than by grep for the old name, so a NEW rival
    under a different name fails too.
    """
    contract_fields = {"claim", "confidence", "payload", "justification", "method"}
    offenders = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {
                b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                for b in node.bases
            }
            if "BaseModel" not in bases or node.name == "DemiGodResult":
                continue
            fields = {
                t.target.id
                for t in node.body
                if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)
            }
            # A model carrying the finding-shaped pair is a second contract.
            if {"payload", "justification"} <= fields or (
                len(fields & contract_fields) >= 3
            ):
                offenders.append(
                    f"{path.relative_to(SRC)}:{node.name} {sorted(fields)}"
                )
    assert not offenders, (
        "these models duplicate the DemiGodResult contract; extend it or nest "
        f"inside `payload` instead: {offenders}"
    )


def test_both_runtimes_describe_output_with_the_same_schema_builder():
    """One builder, so the two runtimes cannot describe output differently."""
    schema = result_json_schema(ARTIFACT_SCHEMA)
    assert schema["properties"]["payload"] == ARTIFACT_SCHEMA
    # Envelope fields are runner-owned and must never be shown to the agent.
    for field in ("demigod_name", "domain_name", "run_id", "status", "error"):
        assert field not in schema["properties"], field


def test_confidence_is_agent_authored_not_hardcoded():
    """The contract requires it, so no runtime can substitute a constant."""
    assert DemiGodResult.model_fields["confidence"].is_required()
