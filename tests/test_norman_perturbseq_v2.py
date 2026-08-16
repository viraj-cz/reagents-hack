"""The stronger benchmark exposes evidence/compute, never answer candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.perturbseq_norman.v2_benchmark import load_problem
from reagents.tools.registry import default_registry
from reagents.tools.tool_runtime import norman_training_python

CASE_DIR = Path(__file__).parents[1] / "benchmarks" / "perturbseq_norman" / "v2"
PUBLIC = CASE_DIR / "public"
PRIVATE = CASE_DIR / "private"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_freeze_commits_to_public_and_private_boundaries():
    manifest = json.loads((PUBLIC / "freeze_manifest.json").read_text())
    assert manifest["frozen_before_model_calls"] is True
    assert manifest["prior_heldout_pairs_excluded"] is True
    assert manifest["test_pair_selection_uses_expression"] is False
    private_fixture = PRIVATE / "expected.json"
    if private_fixture.exists():
        assert manifest["private_expected_sha256"] == _sha(private_fixture)
    else:
        # CI and published source intentionally omit the held-out payload; the
        # evaluator injects it and verifies this public commitment before use.
        assert len(manifest["private_expected_sha256"]) == 64
    assert manifest["public_question_sha256"] == _sha(PUBLIC / "question.json")
    assert manifest["public_training_data_sha256"] == _sha(
        PUBLIC / "training_data.json"
    )


def test_v2_public_problem_has_components_but_no_target_outcomes():
    problem = load_problem()
    assert problem.id == "norman-perturbseq-primitives-heldout-v2"
    training = json.loads((PUBLIC / "training_data.json").read_text())
    assert len(training["training_pairs"]) == 119
    assert len(training["target_components"]) == 12
    for target_id in training["target_ids"]:
        assert target_id not in training["training_pairs"]
    target_catalog = json.dumps(problem.inputs["target_catalog"])
    assert "observed_delta" not in target_catalog
    assert "interaction_class" not in target_catalog


def test_v2_registry_exposes_primitives_and_four_labs(monkeypatch):
    monkeypatch.delenv("REAGENTS_ENABLE_NORMAN_V2_BENCHMARK", raising=False)
    assert not [
        tool for tool in default_registry().ids() if tool.startswith("screen2.")
    ]
    monkeypatch.setenv("REAGENTS_ENABLE_NORMAN_V2_BENCHMARK", "1")
    registry = default_registry()
    tools = [tool for tool in registry.ids() if tool.startswith("screen2.")]
    assert set(tools) == {
        "screen2.training_manifest",
        "screen2.single_effects",
        "screen2.training_pairs",
        "screen2.algebra_lab",
        "screen2.geometry_lab",
        "screen2.graph_lab",
        "screen2.information_lab",
    }
    manifest = registry.get("screen2.training_manifest").call_sync()
    assert manifest["heldout_outcomes_present"] is False


def test_training_lab_executes_agent_code_without_heldout_outcomes(monkeypatch):
    monkeypatch.setenv(
        "REAGENTS_NORMAN_TRAINING_PATH", str(PUBLIC / "training_data.json")
    )
    result = norman_training_python(
        {
            "source": (
                "print(len(DATA['training_pairs']), len(DATA['single_effects']))\n"
                "print(sorted(DATA['target_components'])[:2])"
            )
        }
    )
    assert result["ok"] is True
    assert "119 105" in result["stdout"]
    assert "T01" in result["stdout"]
    assert result["heldout_outcomes_present"] is False
