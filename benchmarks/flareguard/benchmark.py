"""Loader, deterministic simulator, oracle search, and grader for FlareGuard.

The public files define the biological design problem. The private directory is
read only by ``score_solution(private=True)`` after a run; it is never included
in ``NativeProblem.inputs`` or a Modal shared volume.
"""

from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from reagents.contracts import NativeProblem, NativeSolution
from reagents.schema import validate_payload
from reagents.verification import VerificationReport

CASE_DIR = Path(__file__).parent
PUBLIC_DIR = CASE_DIR / "public"
PRIVATE_DIR = CASE_DIR / "private"

DESIGN_KEYS = {
    "positive_sensor_ids",
    "positive_quorum",
    "exclusion_sensor_ids",
    "persistence_part_id",
    "reset_part_id",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            row["time_h"] = float(raw["time_h"])
            row["desired_reporter"] = int(raw["desired_reporter"])
            for key in (
                "nitrate",
                "thiosulfate",
                "calprotectin",
                "quorum_signal",
                "siderophore",
                "butyrate",
            ):
                row[key] = float(raw[key])
            rows.append(row)
    return rows


def load_public_inputs() -> dict[str, Any]:
    return {
        "trajectory_table": _load_rows(PUBLIC_DIR / "trajectories.csv"),
        "part_library": _load_json(PUBLIC_DIR / "parts.json"),
        "compatibility_rules": _load_json(PUBLIC_DIR / "compatibility.json"),
    }


def load_problem() -> NativeProblem:
    """Hydrate public files into God's native problem before projection.

    Returning no mount paths is intentional: every demigod receives these data
    only after its transformer has re-encoded them. Mounting the native CSV next
    to a sealed envelope would bypass the semantic isolation boundary.
    """

    problem = NativeProblem.model_validate_json(
        (PUBLIC_DIR / "question.json").read_text(encoding="utf-8")
    )
    return problem.model_copy(update={"inputs": load_public_inputs()})


def demigod_mount_paths() -> list[Path]:
    """FlareGuard has no raw native files visible inside demigod sandboxes."""

    return []


def _parts_by_id(parts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for section in ("sensors", "timers", "reset_modules", "logic_modules"):
        entries.extend(parts.get(section, []))
    entries.append(parts["reporter"])
    return {entry["id"]: entry for entry in entries}


def _predicate(sensor: dict[str, Any], row: dict[str, Any]) -> bool:
    value = float(row[sensor["signal"]])
    threshold = float(sensor["threshold"])
    if sensor["comparator"] == "gte":
        return value >= threshold
    if sensor["comparator"] == "lte":
        return value <= threshold
    raise ValueError(f"unsupported comparator {sensor['comparator']!r}")


def normalize_design(raw: dict[str, Any]) -> dict[str, Any]:
    design = raw.get("design", raw)
    if not isinstance(design, dict):
        raise ValueError("structured_answer.design must be an object")
    missing = sorted(DESIGN_KEYS - set(design))
    if missing:
        raise ValueError(f"design is missing {missing}")
    return {
        "positive_sensor_ids": sorted(set(design["positive_sensor_ids"])),
        "positive_quorum": int(design["positive_quorum"]),
        "exclusion_sensor_ids": sorted(set(design["exclusion_sensor_ids"])),
        "persistence_part_id": str(design["persistence_part_id"]),
        "reset_part_id": str(design["reset_part_id"]),
    }


def validate_design(
    design: dict[str, Any], parts: dict[str, Any], compatibility: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    by_id = _parts_by_id(parts)
    positive_allowed = set(compatibility["grammar"]["positive_sensor_ids"])
    exclusion_allowed = set(compatibility["grammar"]["exclusion_sensor_ids"])
    positives = set(design["positive_sensor_ids"])
    exclusions = set(design["exclusion_sensor_ids"])
    selected_parts = {
        design["persistence_part_id"],
        design["reset_part_id"],
    }
    unknown = (positives | exclusions | selected_parts) - set(by_id)
    if unknown:
        errors.append(f"unknown part ids: {sorted(unknown)}")
    if not positives or not positives <= positive_allowed:
        errors.append("positive sensors violate the compatible voting-gate interface")
    if not exclusions <= exclusion_allowed:
        errors.append("exclusion sensors violate the compatible repression interface")
    quorum = design["positive_quorum"]
    if not 1 <= quorum <= len(positives):
        errors.append("positive_quorum must be between 1 and the positive sensor count")
    if design["persistence_part_id"] not in {p["id"] for p in parts["timers"]}:
        errors.append("persistence_part_id is not a timer")
    if design["reset_part_id"] not in {p["id"] for p in parts["reset_modules"]}:
        errors.append("reset_part_id is not a reset module")
    for pair in compatibility.get("incompatible_pairs", []):
        if set(pair) <= positives | exclusions:
            errors.append(f"incompatible pair selected: {pair}")
    return errors


def burden(design: dict[str, Any], parts: dict[str, Any]) -> float:
    by_id = _parts_by_id(parts)
    selected = [
        *design["positive_sensor_ids"],
        *design["exclusion_sensor_ids"],
        design["persistence_part_id"],
        design["reset_part_id"],
        parts["reporter"]["id"],
    ]
    total = sum(float(by_id[part_id]["burden"]) for part_id in selected)
    positive_count = len(design["positive_sensor_ids"])
    if positive_count > 1:
        gate_id = "G_VOTE" if positive_count >= 3 else "G_AND_OR"
        total += float(by_id[gate_id]["burden"])
    if design["exclusion_sensor_ids"]:
        total += float(by_id["G_EXCLUDE"]["burden"])
    return round(total, 4)


def simulate_design(
    design: dict[str, Any],
    rows: Iterable[dict[str, Any]],
    parts: dict[str, Any],
    *,
    dropped_sensor: str | None = None,
) -> list[dict[str, Any]]:
    by_id = _parts_by_id(parts)
    persistence_h = float(by_id[design["persistence_part_id"]]["hours"])
    reset_h = float(by_id[design["reset_part_id"]]["hours"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["trajectory"])].append(row)

    predictions: list[dict[str, Any]] = []
    for trajectory, points in sorted(grouped.items()):
        true_since: float | None = None
        false_since: float | None = None
        output = False
        for row in sorted(points, key=lambda item: float(item["time_h"])):
            time_h = float(row["time_h"])

            positive_count = sum(
                sensor != dropped_sensor and _predicate(by_id[sensor], row)
                for sensor in design["positive_sensor_ids"]
            )
            excluded = any(
                sensor != dropped_sensor and _predicate(by_id[sensor], row)
                for sensor in design["exclusion_sensor_ids"]
            )
            base = positive_count >= design["positive_quorum"] and not excluded

            if base:
                if true_since is None:
                    true_since = time_h
                false_since = None
                if time_h - true_since >= persistence_h:
                    output = True
            else:
                true_since = None
                if output:
                    if false_since is None:
                        false_since = time_h
                    if time_h - false_since >= reset_h:
                        output = False
                else:
                    false_since = None

            predictions.append(
                {
                    "trajectory": trajectory,
                    "time_h": time_h,
                    "predicted_reporter": int(output),
                    "desired_reporter": int(row["desired_reporter"]),
                    "dropped_sensor": dropped_sensor,
                }
            )
    return predictions


def _prediction_errors(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in predictions
        if row["predicted_reporter"] != row["desired_reporter"]
    ]


def evaluate_design(
    raw_design: dict[str, Any],
    *,
    include_private: bool = False,
) -> dict[str, Any]:
    inputs = load_public_inputs()
    rows = list(inputs["trajectory_table"])
    if include_private:
        rows.extend(_load_rows(PRIVATE_DIR / "heldout_trajectories.csv"))
    parts = inputs["part_library"]
    compatibility = inputs["compatibility_rules"]
    try:
        design = normalize_design(raw_design)
    except (TypeError, ValueError, KeyError) as exc:
        return {
            "passed": False,
            "score": 0.0,
            "errors": [str(exc)],
            "checks": {},
        }

    errors = validate_design(design, parts, compatibility)
    if errors:
        return {"passed": False, "score": 0.0, "errors": errors, "checks": {}}

    baseline = simulate_design(design, rows, parts)
    baseline_errors = _prediction_errors(baseline)
    dropout_errors: dict[str, int] = {}
    for sensor in parts["sensors"]:
        predictions = simulate_design(design, rows, parts, dropped_sensor=sensor["id"])
        count = len(_prediction_errors(predictions))
        if count:
            dropout_errors[sensor["id"]] = count

    design_burden = burden(design, parts)
    burden_ok = design_burden <= float(compatibility["burden_limit"])
    checks = {
        "all_native_trajectories": not baseline_errors,
        "single_sensor_dropout": not dropout_errors,
        "burden_limit": burden_ok,
        "burden": design_burden,
        "baseline_mismatches": len(baseline_errors),
        "dropout_mismatches": dropout_errors,
    }
    passed = not baseline_errors and not dropout_errors and burden_ok
    score = (
        0.45 * float(not baseline_errors)
        + 0.35 * float(not dropout_errors)
        + 0.20 * float(burden_ok)
    )
    result_errors = []
    if baseline_errors:
        result_errors.append(f"{len(baseline_errors)} phenotype predictions fail")
    if dropout_errors:
        result_errors.append(f"sensor-dropout failures: {dropout_errors}")
    if not burden_ok:
        result_errors.append(
            f"burden {design_burden} exceeds {compatibility['burden_limit']}"
        )
    return {
        "passed": passed,
        "score": round(score, 4),
        "errors": result_errors,
        "checks": checks,
        "design": design,
    }


def enumerate_valid_designs(*, include_private: bool = False) -> list[dict[str, Any]]:
    inputs = load_public_inputs()
    parts = inputs["part_library"]
    grammar = inputs["compatibility_rules"]["grammar"]
    positive_ids = grammar["positive_sensor_ids"]
    exclusion_ids = grammar["exclusion_sensor_ids"]
    candidates: list[dict[str, Any]] = []
    for positive_count in range(1, len(positive_ids) + 1):
        for positives in itertools.combinations(positive_ids, positive_count):
            for quorum in range(1, positive_count + 1):
                for exclusion_count in range(0, len(exclusion_ids) + 1):
                    for exclusions in itertools.combinations(
                        exclusion_ids, exclusion_count
                    ):
                        for timer in parts["timers"]:
                            for reset in parts["reset_modules"]:
                                design = {
                                    "positive_sensor_ids": list(positives),
                                    "positive_quorum": quorum,
                                    "exclusion_sensor_ids": list(exclusions),
                                    "persistence_part_id": timer["id"],
                                    "reset_part_id": reset["id"],
                                }
                                result = evaluate_design(
                                    design, include_private=include_private
                                )
                                if result["passed"]:
                                    candidates.append(result)
    return sorted(
        candidates,
        key=lambda item: (
            item["checks"]["burden"],
            len(item["design"]["positive_sensor_ids"])
            + len(item["design"]["exclusion_sensor_ids"]),
            json.dumps(item["design"], sort_keys=True),
        ),
    )


def score_solution(
    solution: NativeSolution | dict[str, Any], *, private: bool = True
) -> dict[str, Any]:
    if isinstance(solution, NativeSolution):
        structured = solution.structured_answer
    else:
        raw = solution.get("solution", solution)
        structured = raw.get("structured_answer", raw)
    result = evaluate_design(structured, include_private=private)
    schema_errors = validate_payload(structured, load_problem().answer_schema)
    oracle = enumerate_valid_designs(include_private=private)
    optimal_burden = oracle[0]["checks"]["burden"] if oracle else None
    submitted_burden = result.get("checks", {}).get("burden")
    minimal = bool(result["passed"] and submitted_burden == optimal_burden)
    result["checks"]["globally_minimal"] = minimal
    result["checks"]["optimal_burden"] = optimal_burden
    result["score_10"] = round(result["score"] * 9 + float(minimal), 2)
    result["checks"]["complete_native_answer"] = not schema_errors
    if schema_errors:
        result["errors"].extend(
            f"native answer schema: {error}" for error in schema_errors
        )
        result["score_10"] = min(result["score_10"], 8.0)
    result["passed"] = bool(result["passed"] and minimal and not schema_errors)
    if result["score_10"] < 10 and not minimal:
        result["errors"].append("design is not the minimum-burden valid circuit")
    return result


class FlareGuardVerifier:
    """God-side public checker; the private grader remains a separate boundary."""

    def verify(
        self, problem: NativeProblem, solution: NativeSolution
    ) -> VerificationReport:
        if problem.id != "flareguard-living-diagnostic":
            return VerificationReport(
                passed=False,
                score=0.0,
                errors=[f"wrong verifier for problem {problem.id!r}"],
            )
        result = score_solution(solution, private=False)
        return VerificationReport(
            passed=result["passed"],
            score=result["score_10"] / 10,
            checks=result["checks"],
            errors=result["errors"],
        )


__all__ = [
    "FlareGuardVerifier",
    "burden",
    "demigod_mount_paths",
    "enumerate_valid_designs",
    "evaluate_design",
    "load_problem",
    "load_public_inputs",
    "normalize_design",
    "score_solution",
    "simulate_design",
]
