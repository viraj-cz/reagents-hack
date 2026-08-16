from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.haplotype_phasing.benchmark import (
    PRIVATE_DIR,
    PUBLIC_DIR,
    VARIANT_IDS,
    HaplotypePhasingVerifier,
    load_problem,
    normalize_phase,
    score_solution,
)
from demigod.result import DemiGodResult
from reagents.contracts import NativeSolution
from reagents.tools.registry import default_registry


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_freeze_commits_to_public_private_and_real_source_boundaries():
    manifest = json.loads((PUBLIC_DIR / "freeze_manifest.json").read_text())
    assert manifest["frozen_before_model_calls"] is True
    assert manifest["heldout_reference_available_to_agents"] is False
    assert manifest["public_question_sha256"] == _sha(PUBLIC_DIR / "question.json")
    assert manifest["public_observations_sha256"] == _sha(
        PUBLIC_DIR / "observations.json"
    )
    assert "private_expected_sha256" not in manifest
    private_seal = json.loads((PRIVATE_DIR / "freeze_seal.json").read_text())
    assert private_seal["private_expected_sha256"] == _sha(
        PRIVATE_DIR / "expected.json"
    )
    assert private_seal["disclosed_to_agents"] is False
    assert manifest["counts"] == {
        "variants": 49,
        "reads": 25,
        "allele_calls": 477,
    }


def test_god_problem_has_provenance_but_not_private_phase():
    problem = load_problem()
    assert problem.id == "giab-hg004-long-read-phasing-v1"
    serialized = json.dumps(problem.model_dump(mode="json"))
    assert "HG004" in serialized
    assert "exact_public_objective_optimum" not in serialized
    assert '"phase"' not in serialized
    assert "private_expected_sha256" not in serialized


def test_binary_registry_is_gated_and_worker_outputs_are_semantically_sealed(
    monkeypatch,
):
    monkeypatch.delenv("REAGENTS_ENABLE_HAPLOTYPE_BENCHMARK", raising=False)
    assert not [tool for tool in default_registry().ids() if tool.startswith("binary.")]
    monkeypatch.setenv("REAGENTS_ENABLE_HAPLOTYPE_BENCHMARK", "1")
    registry = default_registry()
    tool_ids = [tool for tool in registry.ids() if tool.startswith("binary.")]
    assert set(tool_ids) == {
        "binary.graph_compute",
        "binary.energy_compute",
        "binary.code_compute",
        "binary.logic_compute",
        "binary.observe",
        "binary.score",
        "binary.flip_delta",
        "binary.components",
    }
    zeros = {f"S{index:03d}": 0 for index in range(1, 50)}
    visible = json.dumps(
        {
            "specs": [
                registry.get(tool_id).spec().model_dump() for tool_id in tool_ids
            ],
            "graph": registry.get("binary.graph_compute").call_sync(),
            "energy": registry.get("binary.energy_compute").call_sync(),
            "code": registry.get("binary.code_compute").call_sync(),
            "logic": registry.get("binary.logic_compute").call_sync(),
            "observations": registry.get("binary.observe").call_sync(word_ids=["W001"]),
            "score": registry.get("binary.score").call_sync(assignment=zeros),
            "delta": registry.get("binary.flip_delta").call_sync(
                assignment=zeros, symbols=["S001"]
            ),
            "components": registry.get("binary.components").call_sync(),
        }
    ).lower()
    for forbidden in (
        "hg004",
        "pacbio",
        "genome",
        "chromosome",
        "variant",
        "read",
        "allele",
        "haplotype",
        "reference",
        "alternate",
    ):
        assert forbidden not in visible


def test_private_reference_and_global_complement_both_score_perfectly():
    expected = json.loads((PRIVATE_DIR / "expected.json").read_text())
    phase = expected["phase"]
    answer = {
        "phase_assignment": phase,
        "uncertain_variants": [],
        "method_summary": "fixture reference",
    }
    assert score_solution(answer)["score_10"] == 10.0
    answer["phase_assignment"] = {key: 1 - value for key, value in phase.items()}
    assert score_solution(answer)["score_10"] == 10.0


def test_public_schema_requires_native_ids_not_worker_symbols():
    assignment = {f"S{index:03d}": 0 for index in range(1, 50)}
    _, errors = normalize_phase(
        {
            "phase_assignment": assignment,
            "uncertain_variants": [],
            "method_summary": "wrong coordinate system",
        }
    )
    assert errors


def test_god_side_finalizer_can_inverse_translate_complete_artifact():
    expected = json.loads((PRIVATE_DIR / "expected.json").read_text())
    symbols = {
        f"S{int(key[1:]):03d}": value for key, value in expected["phase"].items()
    }
    artifact = DemiGodResult(
        claim="complete abstract assignment",
        confidence=0.9,
        method="weighted constraint optimization",
        payload={
            "candidate_solution": {
                "assignment": symbols,
                "uncertain_symbols": [],
                "method_summary": "abstract system optimum",
            }
        },
        demigod_name="abstract-agent",
        domain_name="abstract_domain",
        run_id="test",
    )
    solution = NativeSolution(
        problem_id=load_problem().id,
        answer="",
        structured_answer={},
        confidence=0.5,
    )
    finalized = HaplotypePhasingVerifier().finalize(
        load_problem(), solution, [artifact]
    )
    assert set(finalized.structured_answer["phase_assignment"]) == set(VARIANT_IDS)
    assert score_solution(finalized)["score_10"] == 10.0
