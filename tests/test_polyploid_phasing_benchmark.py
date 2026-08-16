from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.polyploid_phasing.benchmark import (
    PRIVATE_DIR,
    PUBLIC_DIR,
    PolyploidPhasingVerifier,
    load_problem,
    score_solution,
)
from reagents.contracts import NativeSolution
from reagents.tools.registry import default_registry


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_answer() -> dict:
    expected = json.loads((PRIVATE_DIR / "expected.json").read_text())
    blocks: dict[str, list[str]] = {}
    for variant_id, block in expected["phase_set"].items():
        blocks.setdefault(str(block), []).append(variant_id)
    covered = set(expected["phase_set"])
    for index in range(1, 43):
        variant_id = f"V{index:03d}"
        if variant_id not in covered:
            blocks[f"unsupported-{variant_id}"] = [variant_id]
    return {
        "haplotypes": [
            {"label": f"H{index + 1}", "alleles": row}
            for index, row in enumerate(expected["canonical_reference"]["haplotypes"])
        ],
        "phase_blocks": [
            {"variant_ids": ids, "confidence": 1.0} for ids in blocks.values()
        ],
        "uncertain_variants": [],
        "method_summary": "frozen reference",
    }


def test_polyploid_freeze_and_private_boundary():
    manifest = json.loads((PUBLIC_DIR / "freeze_manifest.json").read_text())
    assert manifest["counts"] == {
        "ploidy": 4,
        "variants": 42,
        "informative_reads": 18,
        "calls": 385,
    }
    assert manifest["public_question_sha256"] == _sha(PUBLIC_DIR / "question.json")
    assert manifest["public_observations_sha256"] == _sha(
        PUBLIC_DIR / "observations.json"
    )
    assert "private_expected_sha256" not in manifest
    seal = json.loads((PRIVATE_DIR / "freeze_seal.json").read_text())
    assert seal["private_expected_sha256"] == _sha(PRIVATE_DIR / "expected.json")
    assert seal["disclosed_to_agents"] is False


def test_reference_scores_ten_and_exact_objective_is_recorded():
    score = score_solution(_reference_answer())
    assert score["score_10"] == 10.0
    assert score["exact_optimum"] <= score["reference_objective_ceiling"]


def test_latent_tools_are_gated_and_semantically_sealed(monkeypatch):
    monkeypatch.delenv("REAGENTS_ENABLE_POLYPLOID_BENCHMARK", raising=False)
    assert not [tool for tool in default_registry().ids() if tool.startswith("latent.")]
    monkeypatch.setenv("REAGENTS_ENABLE_POLYPLOID_BENCHMARK", "1")
    registry = default_registry()
    ids = [tool for tool in registry.ids() if tool.startswith("latent.")]
    assert set(ids) == {
        "latent.words_compute",
        "latent.similarity_compute",
        "latent.tensor_compute",
        "latent.layered_compute",
        "latent.score",
        "latent.swap_delta",
        "latent.align",
    }
    expected = json.loads((PRIVATE_DIR / "expected.json").read_text())
    factors = [
        {f"S{int(key[1:]):03d}": value for key, value in row.items()}
        for row in expected["exact_public_objective_optimum"]["haplotypes"]
    ]
    words = registry.get("latent.words_compute").call_sync()
    similarity = registry.get("latent.similarity_compute").call_sync()
    tensor = registry.get("latent.tensor_compute").call_sync()
    layers = registry.get("latent.layered_compute").call_sync()
    assert len(words["words"]) == 18
    assert len(similarity["nodes"]) == 18
    assert all(node["observations"] for node in similarity["nodes"])
    assert len(similarity["column_multiplicity"]) == 42
    assert len(tensor["column_multiplicity"]) == 42
    assert len(tensor["cells"]) == 861
    assert len(layers["layers"]) == 42
    assert len(layers["path_constraints"]) == 18
    visible = json.dumps(
        {
            "specs": [registry.get(tool_id).spec().model_dump() for tool_id in ids],
            "words": words,
            "similarity": similarity,
            "tensor": tensor,
            "layers": layers,
            "score": registry.get("latent.score").call_sync(factors=factors),
        }
    ).lower()
    assert "candidate_solution" not in visible
    for forbidden in (
        "hg00514",
        "na19240",
        "pacbio",
        "human",
        "chromosome",
        "variant",
        "read",
        "allele",
        "haplotype",
        "genome",
    ):
        assert forbidden not in visible


def test_public_verifier_accepts_only_complete_dosage_consistent_answer():
    answer = _reference_answer()
    solution = NativeSolution(
        problem_id=load_problem().id,
        answer="complete",
        structured_answer=answer,
        confidence=0.8,
    )
    report = PolyploidPhasingVerifier().verify(load_problem(), solution)
    assert report.passed
    alleles = answer["haplotypes"][0]["alleles"]
    alleles["V001"] = 1 - alleles["V001"]
    bad = solution.model_copy(update={"structured_answer": answer})
    assert not PolyploidPhasingVerifier().verify(load_problem(), bad).passed


def test_public_verifier_accepts_god_json_answer_with_positional_rows():
    answer = _reference_answer()
    positional = {
        **answer,
        "haplotypes": [
            {
                "label": row["label"],
                "alleles": [
                    row["alleles"][variant_id]
                    for variant_id in sorted(row["alleles"])
                ],
            }
            for row in answer["haplotypes"]
        ],
    }
    solution = NativeSolution(
        problem_id=load_problem().id,
        answer=json.dumps(positional),
        structured_answer={"consensus_objective": 604},
        confidence=0.8,
    )
    report = PolyploidPhasingVerifier().verify(load_problem(), solution)
    assert report.passed
    assert report.checks["weighted_discordance"] <= 933
