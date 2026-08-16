"""Public verifier and private scorer for the HG004 phasing benchmark."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

from reagents.contracts import DemiGodResult, NativeProblem, NativeSolution
from reagents.tools import haplotype as binary_tools
from reagents.verification import VerificationReport

CASE_DIR = Path(__file__).parent
PUBLIC_DIR = CASE_DIR / "public"
PRIVATE_DIR = CASE_DIR / "private"
VARIANT_IDS = tuple(f"V{index:03d}" for index in range(1, 50))
SYMBOL_IDS = tuple(f"S{index:03d}" for index in range(1, 50))


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_problem() -> NativeProblem:
    raw = _load(PUBLIC_DIR / "question.json")
    raw["inputs"]["freeze_manifest"] = _load(PUBLIC_DIR / "freeze_manifest.json")
    return NativeProblem.model_validate(raw)


def _structured(raw: dict[str, Any] | NativeSolution) -> dict[str, Any]:
    if isinstance(raw, NativeSolution):
        return raw.structured_answer
    if "solution" in raw and isinstance(raw["solution"], dict):
        return raw["solution"].get("structured_answer", {})
    return raw.get("structured_answer", raw)


def normalize_phase(
    raw: dict[str, Any] | NativeSolution,
) -> tuple[dict[str, int], list[str]]:
    structured = _structured(raw)
    assignment = (
        structured.get("phase_assignment") if isinstance(structured, dict) else None
    )
    if not isinstance(assignment, dict):
        return {}, ["structured_answer.phase_assignment must be an object"]
    errors: list[str] = []
    actual = set(assignment)
    expected = set(VARIANT_IDS)
    if actual != expected:
        errors.append(
            f"phase_assignment IDs must be exactly V001..V049; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    normalized: dict[str, int] = {}
    for variant_id in VARIANT_IDS:
        value = assignment.get(variant_id)
        if isinstance(value, bool):
            value = int(value)
        if value not in (0, 1):
            errors.append(f"{variant_id} must be 0 or 1")
            continue
        normalized[variant_id] = int(value)
    uncertain = (
        structured.get("uncertain_variants", []) if isinstance(structured, dict) else []
    )
    if not isinstance(uncertain, list) or any(
        item not in expected for item in uncertain
    ):
        errors.append("uncertain_variants must contain only public variant IDs")
    summary = structured.get("method_summary") if isinstance(structured, dict) else None
    if not isinstance(summary, str) or not summary.strip():
        errors.append("method_summary must be a non-empty string")
    return normalized, errors


def _symbols_to_variants(assignment: Any) -> dict[str, int] | None:
    if not isinstance(assignment, dict):
        return None
    keys = set(assignment)
    if keys == set(SYMBOL_IDS):
        return {f"V{int(key[1:]):03d}": value for key, value in assignment.items()}
    if keys == set(VARIANT_IDS):
        return dict(assignment)
    return None


def _artifact_candidate(artifact: DemiGodResult) -> dict[str, Any] | None:
    payload = artifact.payload
    candidate = payload.get("candidate_solution", payload)
    if not isinstance(candidate, dict):
        return None
    raw_assignment = candidate.get("assignment", candidate.get("phase_assignment"))
    assignment = _symbols_to_variants(raw_assignment)
    if assignment is None:
        return None
    return {
        "phase_assignment": assignment,
        "uncertain_variants": [
            f"V{int(item[1:]):03d}"
            if isinstance(item, str) and item.startswith("S")
            else item
            for item in candidate.get(
                "uncertain_symbols", candidate.get("uncertain_variants", [])
            )
        ],
        "method_summary": candidate.get(
            "method_summary",
            f"Complete assignment translated from the {artifact.domain_name} artifact.",
        ),
        "source_domain": artifact.domain_name,
    }


def _public_cost(native_assignment: dict[str, int]) -> int:
    symbols = {
        f"S{int(key[1:]):03d}": value for key, value in native_assignment.items()
    }
    return int(binary_tools.score(symbols)["weighted_discordance"])


class HaplotypePhasingVerifier:
    """Schema and public-objective checks; never reads private expected phase."""

    def finalize(
        self,
        problem: NativeProblem,
        solution: NativeSolution,
        artifacts: list[DemiGodResult],
    ) -> NativeSolution:
        candidates: list[dict[str, Any]] = []
        integrated, integrated_errors = normalize_phase(solution)
        if not integrated_errors:
            candidates.append(
                {
                    **solution.structured_answer,
                    "phase_assignment": integrated,
                    "source_domain": "god_integration",
                }
            )
        for artifact in artifacts:
            candidate = _artifact_candidate(artifact)
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            return solution
        selected = min(
            candidates, key=lambda item: _public_cost(item["phase_assignment"])
        )
        objective = binary_tools.score(
            {
                f"S{int(key[1:]):03d}": value
                for key, value in selected["phase_assignment"].items()
            }
        )
        structured = {
            "phase_assignment": selected["phase_assignment"],
            "uncertain_variants": selected.get("uncertain_variants", []),
            "method_summary": selected.get(
                "method_summary", "Public-objective selection."
            ),
            "public_objective": {
                "weighted_discordance": objective["weighted_discordance"],
                "total_weight": objective["total_weight"],
                "heldout_reference_used": False,
            },
            "selected_source": selected.get("source_domain"),
        }
        return solution.model_copy(update={"structured_answer": structured})

    def verify(
        self, problem: NativeProblem, solution: NativeSolution
    ) -> VerificationReport:
        assignment, errors = normalize_phase(solution)
        objective = _public_cost(assignment) if not errors else None
        return VerificationReport(
            passed=not errors,
            score=1.0 if not errors else 0.0,
            checks={
                "variant_count": len(assignment),
                "variant_ids_exact": set(assignment) == set(VARIANT_IDS),
                "binary_values": all(value in (0, 1) for value in assignment.values()),
                "weighted_discordance": objective,
                "heldout_reference_consulted": False,
            },
            errors=errors,
        )


def _orientation_hamming(left: dict[str, int], right: dict[str, int]) -> int:
    direct = sum(left[key] != right[key] for key in VARIANT_IDS)
    return min(direct, len(VARIANT_IDS) - direct)


def _switch_accuracy(left: dict[str, int], right: dict[str, int]) -> float:
    errors = 0
    for previous, current in pairwise(VARIANT_IDS):
        if (left[previous] ^ left[current]) != (right[previous] ^ right[current]):
            errors += 1
    return 1.0 - errors / (len(VARIANT_IDS) - 1)


def score_solution(
    raw: dict[str, Any] | NativeSolution, *, private: bool = True
) -> dict[str, Any]:
    assignment, errors = normalize_phase(raw)
    if errors:
        return {"passed": False, "score_10": 0.0, "errors": errors}
    cost = _public_cost(assignment)
    if not private:
        return {
            "passed": True,
            "score_10": None,
            "weighted_discordance": cost,
            "errors": [],
        }
    expected = _load(PRIVATE_DIR / "expected.json")
    reference = {key: int(value) for key, value in expected["phase"].items()}
    optimum_cost = int(
        expected["exact_public_objective_optimum"]["weighted_discordance"]
    )
    relative_excess = max(0.0, (cost - optimum_cost) / max(optimum_cost, 1))
    objective_optimality = 1.0 / (1.0 + relative_excess)
    hamming = _orientation_hamming(assignment, reference)
    phase_accuracy = 1.0 - hamming / len(VARIANT_IDS)
    switch_accuracy = _switch_accuracy(assignment, reference)
    score_10 = 10.0 * (
        0.45 * objective_optimality + 0.35 * switch_accuracy + 0.20 * phase_accuracy
    )
    return {
        "passed": True,
        "score_10": round(score_10, 4),
        "weighted_discordance": cost,
        "exact_optimum": optimum_cost,
        "objective_optimality": round(objective_optimality, 6),
        "orientation_invariant_phase_accuracy": round(phase_accuracy, 6),
        "switch_accuracy": round(switch_accuracy, 6),
        "orientation_invariant_hamming_errors": hamming,
        "errors": [],
    }
