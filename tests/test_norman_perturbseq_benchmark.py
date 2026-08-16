from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.perturbseq_norman.benchmark import (
    PRIVATE_DIR,
    PUBLIC_DIR,
    NormanPerturbSeqVerifier,
    load_problem,
    score_solution,
)
from benchmarks.perturbseq_norman.compare import _artifact_answer
from reagents.contracts import NativeSolution
from reagents.tools.registry import default_registry


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _answer(method: str = "ensemble") -> dict:
    data = json.loads((PUBLIC_DIR / "tool_data.json").read_text())
    return {
        "predictions": [
            {
                "target_id": target_id,
                "predicted_delta": data["targets"][target_id]["methods"][method][
                    "predicted_delta"
                ],
                "interaction_class": data["targets"][target_id]["methods"][method][
                    "predicted_interaction_class"
                ],
                "confidence": 0.6,
                "falsifier": "The held-out vector disagrees in direction.",
            }
            for target_id in data["target_ids"]
        ],
        "method_summary": f"Frozen training-only {method} candidate.",
    }


def test_freeze_manifest_commits_to_all_model_visible_and_private_fixtures():
    manifest = json.loads((PUBLIC_DIR / "freeze_manifest.json").read_text())
    assert manifest["frozen_before_model_calls"] is True
    assert manifest["test_pair_selection_uses_expression"] is False
    assert manifest["private_expected_sha256"] == _sha(PRIVATE_DIR / "expected.json")
    assert manifest["public_question_sha256"] == _sha(PUBLIC_DIR / "question.json")
    assert manifest["public_tool_data_sha256"] == _sha(PUBLIC_DIR / "tool_data.json")


def test_problem_contains_commitment_but_no_heldout_expression():
    problem = load_problem()
    dumped = json.dumps(problem.inputs)
    assert problem.id == "norman-perturbseq-heldout-combinations-v1"
    assert "private_expected_sha256" in dumped
    assert "observed_delta" not in dumped
    assert "cell_count" not in dumped


def test_registry_gates_eight_training_only_screen_tools(monkeypatch):
    monkeypatch.delenv("REAGENTS_ENABLE_NORMAN_BENCHMARK", raising=False)
    assert not [tool for tool in default_registry().ids() if tool.startswith("screen.")]
    monkeypatch.setenv("REAGENTS_ENABLE_NORMAN_BENCHMARK", "1")
    registry = default_registry()
    tools = [tool for tool in registry.ids() if tool.startswith("screen.")]
    assert len(tools) == 8
    result = registry.get("screen.information_candidate").call_sync(
        target_ids=["T01", "T12"]
    )
    rendered = json.dumps(result)
    assert set(result["targets"]) == {"T01", "T12"}
    assert "observed_delta" not in rendered
    assert "cell_count" not in rendered


def test_public_verifier_checks_shape_without_scoring_truth():
    solution = NativeSolution(
        problem_id=load_problem().id,
        answer="candidate",
        confidence=0.6,
        structured_answer=_answer(),
    )
    report = NormanPerturbSeqVerifier().verify(load_problem(), solution)
    assert report.passed and report.score == 1.0
    assert report.checks["heldout_truth_consulted"] is False


def test_private_scorer_reports_real_metrics_only_after_valid_submission():
    score = score_solution({"structured_answer": _answer()}, private=True)
    assert score["passed"]
    assert 0 <= score["score_10"] <= 10
    assert "additive_relative_skill" in score["metrics"]
    assert len(score["per_target"]) == 12


def test_missing_target_fails_before_private_scoring():
    answer = _answer()
    answer["predictions"].pop()
    score = score_solution({"structured_answer": answer}, private=True)
    assert not score["passed"] and score["score_10"] == 0
    assert any("missing targets" in error for error in score["errors"])


def test_finalizer_copies_public_artifact_vectors_without_private_truth():
    from reagents.contracts import DemiGodResult

    answer = _answer("ridge")
    candidates = {
        item["target_id"]: {
            "delta_vector": item["predicted_delta"],
            "interaction_class": item["interaction_class"],
            "confidence": item["confidence"],
            "falsifier": item["falsifier"],
        }
        for item in answer["predictions"]
    }
    artifact = DemiGodResult(
        domain_name="linear_superposition_residual",
        claim="The residual algebra candidate is the primary vector source.",
        payload={"candidate_solution": candidates},
        confidence=0.6,
        method="Copied the public candidate vectors returned by the Broker.",
    )
    incomplete = NativeSolution(
        problem_id=load_problem().id,
        answer="selected residual algebra",
        structured_answer={
            "selected_primary_vector_source": "linear_superposition_residual",
            "predictions": [
                {
                    "target_id": item["target_id"],
                    "interaction_class": item["interaction_class"],
                    "confidence": item["confidence"],
                    "falsifier": item["falsifier"],
                }
                for item in answer["predictions"]
            ],
        },
        confidence=0.6,
    )
    finalized = NormanPerturbSeqVerifier().finalize(
        load_problem(), incomplete, [artifact]
    )
    report = NormanPerturbSeqVerifier().verify(load_problem(), finalized)
    assert report.passed
    assert len(finalized.structured_answer["predictions"][0]["predicted_delta"]) == 64


def test_post_run_artifact_diagnostic_maps_opaque_ids_positionally():
    answer = _answer("ridge")
    artifact = {
        "confidence": 0.6,
        "payload": {
            "candidate_solution": {
                f"q{position:02d}": {
                    "delta_vector": item["predicted_delta"],
                    "interaction_class": item["interaction_class"],
                    "confidence": item["confidence"],
                    "falsifier": item["falsifier"],
                }
                for position, item in enumerate(answer["predictions"], start=1)
            }
        },
    }
    projected = _artifact_answer(artifact)
    predictions = projected["structured_answer"]["predictions"]
    assert [item["target_id"] for item in predictions] == [
        f"T{position:02d}" for position in range(1, 13)
    ]
    assert len(predictions[0]["predicted_delta"]) == 64
