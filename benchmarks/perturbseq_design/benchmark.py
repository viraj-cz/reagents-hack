"""Public verifier and evaluator-private scorer for batch design."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from reagents.contracts import DemiGodResult, NativeProblem, NativeSolution
from reagents.verification import VerificationReport

CASE_DIR = Path(__file__).parent
PUBLIC_DIR = CASE_DIR / "public"
PRIVATE_DIR = CASE_DIR / "private"
BATCH_SIZE = 10


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_problem() -> NativeProblem:
    raw = _load(PUBLIC_DIR / "question.json")
    raw["inputs"]["freeze_manifest"] = _load(PUBLIC_DIR / "freeze_manifest.json")
    return NativeProblem.model_validate(raw)


def _structured(raw: dict[str, Any] | NativeSolution) -> dict[str, Any]:
    if isinstance(raw, NativeSolution):
        return raw.structured_answer
    if isinstance(raw.get("solution"), dict):
        return raw["solution"].get("structured_answer", {})
    return raw.get("structured_answer", raw)


def normalize_selection(
    raw: dict[str, Any] | NativeSolution,
) -> tuple[list[str], list[str]]:
    structured = _structured(raw)
    selection = structured.get("selected_candidate_ids")
    if selection is None and isinstance(structured.get("candidate_solution"), dict):
        candidate = structured["candidate_solution"]
        selection = candidate.get("selection", candidate.get("selected_candidate_ids"))
    errors: list[str] = []
    if not isinstance(selection, list):
        return [], ["selected_candidate_ids must be an array"]
    values: list[str] = []
    for index, item in enumerate(selection):
        if isinstance(item, dict):
            item = item.get("candidate_id", item.get("id"))
        if not isinstance(item, str):
            errors.append(f"selection[{index}] is not a candidate identifier")
        else:
            values.append(item)
    public = _load(PUBLIC_DIR / "training_data.json")
    allowed = set(public["candidate_ids"])
    if len(values) != BATCH_SIZE:
        errors.append(f"selection must contain exactly {BATCH_SIZE} candidates")
    if len(set(values)) != len(values):
        errors.append("selection contains duplicate candidates")
    unknown = sorted(set(values) - allowed)
    if unknown:
        errors.append(f"unknown candidates: {unknown}")
    counts: Counter[str] = Counter()
    for candidate in values:
        for gene in public["candidate_components"].get(candidate, []):
            counts[gene] += 1
    over = {gene: count for gene, count in counts.items() if count > public["gene_cap"]}
    if over:
        errors.append(f"component-use cap exceeded: {over}")
    return values, errors


def score_solution(
    raw: dict[str, Any] | NativeSolution, *, private: bool = True
) -> dict[str, Any]:
    selection, errors = normalize_selection(raw)
    if errors:
        return {"passed": False, "score_10": 0.0, "errors": errors}
    if not private:
        return {"passed": True, "score_10": None, "errors": []}
    expected = _load(PRIVATE_DIR / "expected.json")
    candidates = expected["candidates"]
    covered = {
        candidates[item]["program"] for item in selection if candidates[item]["strong"]
    }
    strength_sum = sum(candidates[item]["strength"] for item in selection)
    utility = strength_sum + 1.5 * len(covered)
    oracle_utility = float(expected["oracle_utility"])
    score = 10 * utility / oracle_utility if oracle_utility else 0.0
    oracle = set(expected["oracle_selection"])
    return {
        "passed": True,
        "score_10": round(min(10.0, score), 4),
        "metrics": {
            "utility": round(utility, 6),
            "oracle_utility": round(oracle_utility, 6),
            "regret": round(oracle_utility - utility, 6),
            "strength_sum": round(strength_sum, 6),
            "mean_strength": round(strength_sum / BATCH_SIZE, 6),
            "strong_hit_rate": round(
                sum(bool(candidates[item]["strong"]) for item in selection)
                / BATCH_SIZE,
                6,
            ),
            "program_coverage": len(covered),
            "oracle_overlap": len(set(selection) & oracle),
        },
        "selection": selection,
        "errors": [],
    }


class PerturbSeqDesignVerifier:
    def verify(
        self, problem: NativeProblem, solution: NativeSolution
    ) -> VerificationReport:
        del problem
        selection, errors = normalize_selection(solution)
        return VerificationReport(
            passed=not errors,
            score=1.0 if not errors else 0.0,
            checks={"selection": selection, "batch_size": len(selection)},
            errors=errors,
        )

    def finalize(
        self,
        problem: NativeProblem,
        solution: NativeSolution,
        artifacts: list[DemiGodResult],
    ) -> NativeSolution:
        del problem
        _, errors = normalize_selection(solution)
        if not errors:
            return solution
        # Schema transport only: choose the highest-confidence already-complete
        # artifact. No private truth or utility is consulted here.
        for artifact in sorted(artifacts, key=lambda item: -item.confidence):
            candidate = {"structured_answer": artifact.payload}
            selection, artifact_errors = normalize_selection(candidate)
            if not artifact_errors:
                structured = dict(solution.structured_answer)
                structured["selected_candidate_ids"] = selection
                structured.setdefault(
                    "method_summary",
                    "Transported from the highest-confidence complete artifact.",
                )
                return solution.model_copy(update={"structured_answer": structured})
        return solution


__all__ = [
    "PerturbSeqDesignVerifier",
    "load_problem",
    "normalize_selection",
    "score_solution",
]
