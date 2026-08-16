"""Public loader/verifier and private scorer for the primitive-tool benchmark."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.perturbseq_norman.benchmark import (
    FEATURE_COUNT,
    TARGET_IDS,
    NormanPerturbSeqVerifier,
    _macro_f1,
    _pearson,
    normalize_predictions,
)
from reagents.contracts import NativeSolution

CASE_DIR = Path(__file__).parent / "v2"
PUBLIC_DIR = CASE_DIR / "public"
PRIVATE_DIR = CASE_DIR / "private"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_problem():
    """Load public inputs only; held-out responses remain evaluator-private."""

    from reagents.contracts import NativeProblem

    raw = _load(PUBLIC_DIR / "question.json")
    raw["inputs"]["freeze_manifest"] = _load(PUBLIC_DIR / "freeze_manifest.json")
    return NativeProblem.model_validate(raw)


class NormanPerturbSeqV2Verifier(NormanPerturbSeqVerifier):
    """The existing public schema/finalization checks apply to the v2 answer."""


def score_solution(
    raw: dict[str, Any] | NativeSolution, *, private: bool = True
) -> dict[str, Any]:
    """Score after a run; never called by God, demigods, tools, or the Broker."""

    predictions, errors = normalize_predictions(raw)
    if errors or len(predictions) != len(TARGET_IDS):
        return {"passed": False, "score_10": 0.0, "errors": errors}
    if not private:
        return {"passed": True, "score_10": None, "errors": []}

    truth = _load(PRIVATE_DIR / "expected.json")["targets"]
    training = _load(PUBLIC_DIR / "training_data.json")
    components = training["target_components"]
    singles = training["single_effects"]
    actual_flat: list[float] = []
    predicted_flat: list[float] = []
    additive_flat: list[float] = []
    actual_classes: list[str] = []
    predicted_classes: list[str] = []
    per_target: dict[str, Any] = {}
    for prediction in predictions:
        target_id = prediction["target_id"]
        actual = [float(value) for value in truth[target_id]["observed_delta"]]
        predicted = prediction["predicted_delta"]
        left, right = components[target_id]
        additive = [
            float(a) + float(b)
            for a, b in zip(singles[left], singles[right], strict=True)
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
    "NormanPerturbSeqV2Verifier",
    "load_problem",
    "score_solution",
]
