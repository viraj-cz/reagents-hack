"""Run deterministic public-data baselines and score only after committing."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor

from benchmarks.perturbseq_design.benchmark import score_solution

CASE_DIR = Path(__file__).parent
PUBLIC = CASE_DIR / "public" / "training_data.json"


def _solve(
    data: dict[str, Any], scores: np.ndarray, programs: np.ndarray | None = None
) -> list[str]:
    candidates = data["candidate_ids"]
    count = len(candidates)
    program_count = data["program_count"] if programs is not None else 0
    objective = np.zeros(count + program_count)
    objective[:count] = -scores
    if programs is not None:
        objective[count:] = -data["coverage_bonus"]
    rows, lower, upper = [], [], []
    row = np.zeros_like(objective)
    row[:count] = 1
    rows.append(row)
    lower.append(data["batch_size"])
    upper.append(data["batch_size"])
    for gene in data["gene_ids"]:
        row = np.zeros_like(objective)
        for index, candidate in enumerate(candidates):
            if gene in data["candidate_components"][candidate]:
                row[index] = 1
        rows.append(row)
        lower.append(0)
        upper.append(data["gene_cap"])
    if programs is not None:
        for program in range(program_count):
            row = np.zeros_like(objective)
            row[:count] = -(programs == program).astype(float)
            row[count + program] = 1
            rows.append(row)
            lower.append(-np.inf)
            upper.append(0)
    result = milp(
        c=objective,
        integrality=np.ones_like(objective),
        bounds=Bounds(np.zeros_like(objective), np.ones_like(objective)),
        constraints=LinearConstraint(np.stack(rows), lower, upper),
    )
    if not result.success or result.x is None:
        raise RuntimeError(result.message)
    return sorted(
        candidate
        for candidate, value in zip(candidates, result.x[:count], strict=True)
        if value > 0.5
    )


def _matrices(data: dict[str, Any]):
    singles = {key: np.asarray(value) for key, value in data["single_effects"].items()}
    train_ids = data["training_pair_ids"]
    candidate_ids = data["candidate_ids"]

    def features(components: list[str]) -> np.ndarray:
        a, b = (singles[item] for item in components)
        return np.concatenate(
            [
                a + b,
                np.abs(a - b),
                a * b,
                [np.linalg.norm(a), np.linalg.norm(b), float(a @ b)],
            ]
        )

    train_x = np.stack(
        [features(data["training_pairs"][item]["components"]) for item in train_ids]
    )
    candidate_x = np.stack(
        [features(data["candidate_components"][item]) for item in candidate_ids]
    )
    train_observed = np.stack(
        [data["training_pairs"][item]["observed_delta"] for item in train_ids]
    )
    train_additive = np.stack(
        [
            sum(
                (singles[g] for g in data["training_pairs"][item]["components"]),
                np.zeros(64),
            )
            for item in train_ids
        ]
    )
    residual = train_observed - train_additive
    scale = np.median(np.linalg.norm(residual, axis=1))
    strength = np.clip(np.linalg.norm(residual, axis=1) / scale, 0, 2)
    folds = np.asarray([data["training_pairs"][item]["fold"] for item in train_ids])
    additive_candidates = np.stack(
        [
            sum((singles[g] for g in data["candidate_components"][item]), np.zeros(64))
            for item in candidate_ids
        ]
    )
    return singles, train_x, candidate_x, residual, strength, folds, additive_candidates


def run(output: Path) -> dict[str, Any]:
    data = json.loads(PUBLIC.read_text(encoding="utf-8"))
    singles, train_x, candidate_x, residual, strength, _folds, additive = _matrices(
        data
    )
    candidate_ids = data["candidate_ids"]
    predictions: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}

    predictions["additive_magnitude"] = (np.linalg.norm(additive, axis=1), None)
    predictions["strong_singles"] = (
        np.asarray(
            [
                sum(
                    np.linalg.norm(singles[g])
                    for g in data["candidate_components"][item]
                )
                for item in candidate_ids
            ]
        ),
        None,
    )
    by_gene: dict[str, list[float]] = defaultdict(list)
    for pair_id, value in zip(data["training_pair_ids"], strength, strict=True):
        for gene in data["training_pairs"][pair_id]["components"]:
            by_gene[gene].append(float(value))
    prior = np.asarray(
        [
            np.mean(
                [
                    np.mean(by_gene[g]) if by_gene[g] else np.mean(strength)
                    for g in data["candidate_components"][item]
                ]
            )
            for item in candidate_ids
        ]
    )
    predictions["gene_interaction_prior"] = (prior, None)

    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(train_x, strength)
    predictions["ridge_strength"] = (np.clip(ridge.predict(candidate_x), 0, 2), None)
    knn = KNeighborsRegressor(n_neighbors=8, weights="distance").fit(train_x, strength)
    predictions["geometry_knn"] = (np.clip(knn.predict(candidate_x), 0, 2), None)
    forest = RandomForestRegressor(
        n_estimators=400, min_samples_leaf=3, random_state=1907, n_jobs=-1
    ).fit(train_x, strength)
    predictions["random_forest"] = (np.clip(forest.predict(candidate_x), 0, 2), None)

    additive_latent = PCA(n_components=6, random_state=0).fit_transform(
        np.vstack([additive, residual])
    )[: len(additive)]
    diversity_program = KMeans(
        n_clusters=data["program_count"], random_state=1, n_init=20
    ).fit_predict(additive_latent)
    predictions["diversity_only"] = (
        np.full(len(candidate_ids), 0.001),
        diversity_program,
    )

    committed = {
        name: _solve(data, scores, programs)
        for name, (scores, programs) in predictions.items()
    }
    rng = np.random.default_rng(20260816)
    random_batches = []
    for _ in range(250):
        random_batches.append(_solve(data, rng.random(len(candidate_ids)), None))
    scores = {
        name: score_solution({"selected_candidate_ids": batch})
        for name, batch in committed.items()
    }
    random_scores = [
        score_solution({"selected_candidate_ids": batch}) for batch in random_batches
    ]
    random_values = np.asarray([item["score_10"] for item in random_scores])
    record = {
        "kind": "deterministic_public_baselines",
        "responses_committed_before_private_scoring": True,
        "selections": committed,
        "scores": scores,
        "random_250": {
            "mean_score_10": round(float(random_values.mean()), 4),
            "p10_score_10": round(float(np.quantile(random_values, 0.1)), 4),
            "median_score_10": round(float(np.median(random_values)), 4),
            "p90_score_10": round(float(np.quantile(random_values, 0.9)), 4),
            "best_score_10": round(float(random_values.max()), 4),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {"scores": result["scores"], "random_250": result["random_250"]}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
