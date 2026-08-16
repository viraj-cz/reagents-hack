from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.perturbseq_design import load_problem, score_solution
from benchmarks.perturbseq_design.benchmark import PerturbSeqDesignVerifier
from reagents.contracts import NativeSolution
from reagents.tools.registry import default_registry
from reagents.tools.tool_runtime import perturbseq_design_python

CASE = Path(__file__).parents[1] / "benchmarks" / "perturbseq_design"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_freeze_and_private_boundary():
    manifest = json.loads((CASE / "public/freeze_manifest.json").read_text())
    assert manifest["frozen_before_model_calls"] is True
    assert manifest["candidate_selection_uses_expression"] is False
    assert manifest["feature_selection_uses_candidates"] is False
    assert manifest["private_expected_sha256"] == _sha(CASE / "private/expected.json")
    public = json.loads((CASE / "public/training_data.json").read_text())
    assert len(public["training_pair_ids"]) == 63
    assert len(public["candidate_ids"]) == 63
    assert "oracle_selection" not in public
    assert "candidates" not in public


def test_registry_and_training_lab_are_candidate_blind(monkeypatch):
    monkeypatch.setenv("REAGENTS_ENABLE_PERTURBSEQ_DESIGN", "1")
    tools = {item for item in default_registry().ids() if item.startswith("portfolio.")}
    assert tools == {
        "portfolio.manifest",
        "portfolio.single_effects",
        "portfolio.training_pairs",
        "portfolio.algebra_lab",
        "portfolio.geometry_lab",
        "portfolio.graph_lab",
        "portfolio.optimization_lab",
    }
    monkeypatch.setenv(
        "REAGENTS_PERTURBSEQ_DESIGN_PATH", str(CASE / "public/training_data.json")
    )
    result = perturbseq_design_python(
        {"source": "print(len(DATA['candidate_ids']), 'candidates' in DATA)"}
    )
    assert result["ok"] is True
    assert "63 False" in result["stdout"]
    assert result["candidate_outcomes_present"] is False


def test_oracle_is_valid_and_scores_ten():
    expected = json.loads((CASE / "private/expected.json").read_text())
    report = score_solution({"selected_candidate_ids": expected["oracle_selection"]})
    assert report["passed"] is True
    assert report["score_10"] == 10.0
    assert report["metrics"]["regret"] == 0.0


def test_verifier_rejects_component_cap_violation():
    public = json.loads((CASE / "public/training_data.json").read_text())
    counts = {}
    violating = None
    for candidate, components in public["candidate_components"].items():
        for component in components:
            counts.setdefault(component, []).append(candidate)
            if len(counts[component]) >= 3:
                violating = counts[component][:3]
                break
        if violating:
            break
    selection = (
        violating
        + [item for item in public["candidate_ids"] if item not in violating][:7]
    )
    solution = NativeSolution(
        problem_id=load_problem().id,
        answer="",
        confidence=0.5,
        structured_answer={"selected_candidate_ids": selection},
    )
    report = PerturbSeqDesignVerifier().verify(load_problem(), solution)
    assert report.passed is False
    assert any("cap exceeded" in error for error in report.errors)
