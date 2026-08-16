"""Public loader/verifier and private scorer for the Norman benchmark."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from reagents.contracts import DemiGodResult, NativeProblem, NativeSolution
from reagents.verification import VerificationReport

CASE_DIR = Path(__file__).parent
PUBLIC_DIR = CASE_DIR / "public"
PRIVATE_DIR = CASE_DIR / "private"
TARGET_IDS = tuple(f"T{position:02d}" for position in range(1, 13))
FEATURE_COUNT = 64
CLASSES = frozenset({"additive", "synergy", "suppression", "neomorphic"})


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


def normalize_predictions(
    raw: dict[str, Any] | NativeSolution,
) -> tuple[list[dict[str, Any]], list[str]]:
    structured = _structured(raw)
    predictions = (
        structured.get("predictions") if isinstance(structured, dict) else None
    )
    if not isinstance(predictions, list):
        return [], ["structured_answer.predictions must be an array"]
    errors: list[str] = []
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, item in enumerate(predictions):
        if not isinstance(item, dict):
            errors.append(f"prediction {position} is not an object")
            continue
        target_id = item.get("target_id")
        if target_id not in TARGET_IDS:
            errors.append(f"prediction {position} has unknown target_id {target_id!r}")
            continue
        if target_id in seen:
            errors.append(f"duplicate target_id {target_id}")
            continue
        seen.add(target_id)
        values = item.get("predicted_delta")
        if not isinstance(values, list) or len(values) != FEATURE_COUNT:
            errors.append(f"{target_id} predicted_delta must contain 64 values")
            continue
        try:
            vector = [float(value) for value in values]
        except (TypeError, ValueError):
            errors.append(f"{target_id} predicted_delta contains a non-number")
            continue
        if not all(math.isfinite(value) for value in vector):
            errors.append(f"{target_id} predicted_delta contains a non-finite value")
            continue
        interaction_class = item.get("interaction_class")
        if interaction_class not in CLASSES:
            errors.append(
                f"{target_id} has invalid interaction_class {interaction_class!r}"
            )
            continue
        try:
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError):
            errors.append(f"{target_id} confidence is not numeric")
            continue
        if not 0.0 <= confidence <= 1.0:
            errors.append(f"{target_id} confidence is outside [0, 1]")
            continue
        falsifier = item.get("falsifier")
        if not isinstance(falsifier, str) or not falsifier.strip():
            errors.append(f"{target_id} requires a non-empty falsifier")
            continue
        normalized.append(
            {
                "target_id": target_id,
                "predicted_delta": vector,
                "interaction_class": interaction_class,
                "confidence": confidence,
                "falsifier": falsifier,
            }
        )
    missing = sorted(set(TARGET_IDS) - seen)
    if missing:
        errors.append(f"missing targets: {missing}")
    return sorted(normalized, key=lambda item: item["target_id"]), errors


class NormanPerturbSeqVerifier:
    """Schema-only public verifier; it cannot read held-out outcomes."""

    def finalize(
        self,
        problem: NativeProblem,
        solution: NativeSolution,
        artifacts: list[DemiGodResult],
    ) -> NativeSolution:
        """Copy selected public vectors into the exact native answer schema.

        The model remains responsible for choosing classes and resolving
        conflicts. This step only supplies the 64-value vectors it selected
        from an accepted artifact when it summarized them out of the final
        JSON. No private fixture is read.
        """

        by_domain = {artifact.domain_name: artifact for artifact in artifacts}
        preferred = str(
            solution.structured_answer.get("selected_primary_vector_source", "")
        )
        source = next(
            (artifact for name, artifact in by_domain.items() if name in preferred),
            None,
        )
        if source is None:
            source = max(artifacts, key=lambda artifact: artifact.confidence)
        candidates = source.payload.get("candidate_solution", {})
        integrated = {
            item.get("target_id"): item
            for item in solution.structured_answer.get("predictions", [])
            if isinstance(item, dict) and item.get("target_id") in TARGET_IDS
        }
        predictions = []
        for target_id in TARGET_IDS:
            candidate = candidates.get(target_id, {})
            vector = candidate.get("delta_vector") or candidate.get(
                "displacement_vector"
            )
            choice = integrated.get(target_id, {})
            predictions.append(
                {
                    "target_id": target_id,
                    "predicted_delta": vector,
                    "interaction_class": choice.get(
                        "interaction_class", candidate.get("interaction_class")
                    ),
                    "confidence": choice.get(
                        "confidence", candidate.get("confidence", source.confidence)
                    ),
                    "falsifier": choice.get(
                        "falsifier",
                        candidate.get(
                            "falsifier",
                            "Held-out response disagrees with the predicted vector.",
                        ),
                    ),
                }
            )
        return solution.model_copy(
            update={
                "structured_answer": {
                    "predictions": predictions,
                    "method_summary": (
                        f"God selected {source.domain_name} for numeric vectors; "
                        "classes and falsifiers preserve the integrated "
                        "cross-representation reconciliation."
                    ),
                }
            }
        )

    def verify(
        self, problem: NativeProblem, solution: NativeSolution
    ) -> VerificationReport:
        predictions, errors = normalize_predictions(solution)
        method_summary = solution.structured_answer.get("method_summary")
        if not isinstance(method_summary, str) or not method_summary.strip():
            errors.append("structured_answer.method_summary must be non-empty")
        return VerificationReport(
            passed=not errors,
            score=1.0 if not errors else 0.0,
            checks={
                "prediction_count": len(predictions),
                "target_ids_exact": len(predictions) == len(TARGET_IDS),
                "vector_length": FEATURE_COUNT,
                "heldout_truth_consulted": False,
            },
            errors=errors,
        )


def _pearson(left: list[float], right: list[float]) -> float:
    lm, rm = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right, strict=True))
    ld = sum((a - lm) ** 2 for a in left)
    rd = sum((b - rm) ** 2 for b in right)
    return numerator / math.sqrt(ld * rd) if ld > 0 and rd > 0 else 0.0


def _macro_f1(actual: list[str], predicted: list[str]) -> float:
    scores = []
    for label in sorted(CLASSES):
        tp = sum(
            a == label and p == label for a, p in zip(actual, predicted, strict=True)
        )
        fp = sum(
            a != label and p == label for a, p in zip(actual, predicted, strict=True)
        )
        fn = sum(
            a == label and p != label for a, p in zip(actual, predicted, strict=True)
        )
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def score_solution(
    raw: dict[str, Any] | NativeSolution, *, private: bool = True
) -> dict[str, Any]:
    predictions, errors = normalize_predictions(raw)
    if errors or len(predictions) != len(TARGET_IDS):
        return {"passed": False, "score_10": 0.0, "errors": errors}
    if not private:
        return {"passed": True, "score_10": None, "errors": []}
    truth = _load(PRIVATE_DIR / "expected.json")["targets"]
    tool_data = _load(PUBLIC_DIR / "tool_data.json")
    actual_flat: list[float] = []
    predicted_flat: list[float] = []
    additive_flat: list[float] = []
    actual_classes, predicted_classes = [], []
    per_target: dict[str, Any] = {}
    for prediction in predictions:
        target_id = prediction["target_id"]
        actual = [float(value) for value in truth[target_id]["observed_delta"]]
        predicted = prediction["predicted_delta"]
        additive = tool_data["targets"][target_id]["methods"]["additive"][
            "predicted_delta"
        ]
        actual_flat.extend(actual)
        predicted_flat.extend(predicted)
        additive_flat.extend(additive)
        actual_classes.append(truth[target_id]["interaction_class"])
        predicted_classes.append(prediction["interaction_class"])
        per_target[target_id] = {
            "pearson": round(_pearson(actual, predicted), 6),
            "mse": round(
                sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=True))
                / FEATURE_COUNT,
                8,
            ),
            "actual_class": truth[target_id]["interaction_class"],
            "predicted_class": prediction["interaction_class"],
        }
    mse = sum(
        (a - p) ** 2 for a, p in zip(actual_flat, predicted_flat, strict=True)
    ) / len(actual_flat)
    additive_mse = sum(
        (a - p) ** 2 for a, p in zip(actual_flat, additive_flat, strict=True)
    ) / len(actual_flat)
    skill = 1.0 - mse / additive_mse if additive_mse else 0.0
    pearson = _pearson(actual_flat, predicted_flat)
    direction = sum(
        (a >= 0) == (p >= 0) for a, p in zip(actual_flat, predicted_flat, strict=True)
    ) / len(actual_flat)
    macro_f1 = _macro_f1(actual_classes, predicted_classes)
    class_accuracy = sum(
        a == p for a, p in zip(actual_classes, predicted_classes, strict=True)
    ) / len(actual_classes)
    score = 10 * (
        0.35 * max(0.0, min(1.0, (pearson + 1.0) / 2.0))
        + 0.30 * max(0.0, min(1.0, (skill + 1.0) / 2.0))
        + 0.15 * direction
        + 0.15 * macro_f1
        + 0.05
    )
    return {
        "passed": True,
        "score_10": round(score, 4),
        "metrics": {
            "global_pearson": round(pearson, 6),
            "mse": round(mse, 8),
            "additive_baseline_mse": round(additive_mse, 8),
            "additive_relative_skill": round(skill, 6),
            "direction_accuracy": round(direction, 6),
            "interaction_macro_f1": round(macro_f1, 6),
            "interaction_accuracy": round(class_accuracy, 6),
            "actual_class_counts": dict(Counter(actual_classes)),
            "predicted_class_counts": dict(Counter(predicted_classes)),
        },
        "per_target": per_target,
        "errors": [],
    }


__all__ = [
    "NormanPerturbSeqVerifier",
    "load_problem",
    "normalize_predictions",
    "score_solution",
]
