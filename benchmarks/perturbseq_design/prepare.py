"""Freeze a leakage-resistant Norman Perturb-seq batch-design benchmark.

The split and all metric hyperparameters are fixed without consulting a model.
Candidate outcomes are used only to create the evaluator-private utility table
and exact oracle.  Agents see controls, singles, training doubles, and opaque
candidate component identities; never candidate responses or utilities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from benchmarks.perturbseq_norman.prepare import (
    MEMORIZATION_EXCLUSIONS,
    PANEL_SIZE,
    SOURCE,
    SOURCE_SHA256,
    SOURCE_URL,
    _condition_means,
    _decode,
    _round_vector,
    _sha256,
)

CASE_DIR = Path(__file__).parent
PUBLIC = CASE_DIR / "public"
PRIVATE = CASE_DIR / "private"
TOOL_MODULE = (
    Path(__file__).parents[2]
    / "src"
    / "reagents"
    / "tools"
    / "_perturbseq_design_public.py"
)
BENCHMARK_ID = "norman-perturbseq-diverse-batch-design-v1"
SPLIT_SALT = "reagents-norman-diverse-batch-v1"
BATCH_SIZE = 10
PROGRAM_COUNT = 6
GENE_CAP = 2
COVERAGE_BONUS = 1.5
FOLD_COUNT = 5


def _hash(label: str) -> str:
    return hashlib.sha256(f"{SPLIT_SALT}:{label}".encode()).hexdigest()


def _fold(label: str) -> int:
    return int(_hash(f"fold:{label}"), 16) % FOLD_COUNT


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _kmeans(values: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic farthest-first k-means used only during freezing."""

    centers = [values[int(np.argmax(np.linalg.norm(values, axis=1)))]]
    while len(centers) < count:
        distances = np.min(
            np.stack([np.sum((values - center) ** 2, axis=1) for center in centers]),
            axis=0,
        )
        centers.append(values[int(np.argmax(distances))])
    center_array = np.stack(centers)
    labels = np.zeros(len(values), dtype=int)
    for _ in range(100):
        next_labels = np.argmin(
            np.sum((values[:, None, :] - center_array[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        next_centers = center_array.copy()
        for label in range(count):
            members = values[next_labels == label]
            if len(members):
                next_centers[label] = members.mean(axis=0)
        if np.array_equal(next_labels, labels) and np.allclose(
            next_centers, center_array
        ):
            break
        labels, center_array = next_labels, next_centers
    return center_array, labels


def _exact_oracle(
    candidates: list[str],
    components: dict[str, list[str]],
    strengths: dict[str, float],
    programs: dict[str, int],
    strong: dict[str, bool],
) -> tuple[list[str], float]:
    genes = sorted({gene for pair in components.values() for gene in pair})
    n, k = len(candidates), PROGRAM_COUNT
    objective = np.zeros(n + k)
    objective[:n] = [-strengths[candidate] for candidate in candidates]
    objective[n:] = -COVERAGE_BONUS
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    row = np.zeros(n + k)
    row[:n] = 1
    rows.append(row)
    lower.append(BATCH_SIZE)
    upper.append(BATCH_SIZE)
    for gene in genes:
        row = np.zeros(n + k)
        for index, candidate in enumerate(candidates):
            if gene in components[candidate]:
                row[index] = 1
        rows.append(row)
        lower.append(0)
        upper.append(GENE_CAP)
    for program in range(PROGRAM_COUNT):
        # z_program <= sum selected strong candidates assigned to program.
        row = np.zeros(n + k)
        for index, candidate in enumerate(candidates):
            if strong[candidate] and programs[candidate] == program:
                row[index] = -1
        row[n + program] = 1
        rows.append(row)
        lower.append(-np.inf)
        upper.append(0)

    result = milp(
        c=objective,
        integrality=np.ones(n + k),
        bounds=Bounds(np.zeros(n + k), np.ones(n + k)),
        constraints=LinearConstraint(np.stack(rows), lower, upper),
        options={"time_limit": 60},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"oracle optimization failed: {result.message}")
    chosen = [
        candidate
        for candidate, value in zip(candidates, result.x[:n], strict=True)
        if value > 0.5
    ]
    covered = {programs[candidate] for candidate in chosen if strong[candidate]}
    utility = sum(strengths[candidate] for candidate in chosen) + COVERAGE_BONUS * len(
        covered
    )
    return sorted(chosen), float(utility)


def prepare(source: Path = SOURCE) -> dict[str, Any]:
    if _sha256(source) != SOURCE_SHA256:
        raise ValueError("source H5AD hash does not match the pinned Norman fixture")

    with h5py.File(source, "r") as handle:
        conditions = _decode(handle["obs/__categories/perturbation_name"][:])
        codes = handle["obs/perturbation_name"][:].astype(int)
        means, cell_counts = _condition_means(handle, codes, len(conditions))

    eligible = sorted(
        condition
        for condition in conditions
        if "+" in condition and condition not in MEMORIZATION_EXCLUSIONS
    )
    ordered = sorted(eligible, key=_hash)
    split = len(ordered) // 2
    candidate_pairs = sorted(ordered[:split])
    training_pairs = sorted(ordered[split:])
    index = {condition: position for position, condition in enumerate(conditions)}
    single_names = sorted(
        condition
        for condition in conditions
        if "+" not in condition and condition != "control"
    )
    feature_rows = means[
        [index["control"]]
        + [index[name] for name in single_names]
        + [index[name] for name in training_pairs]
    ]
    feature_indices = np.argsort(-np.var(feature_rows, axis=0), kind="stable")[
        :PANEL_SIZE
    ]
    reduced = means[:, feature_indices]
    control = reduced[index["control"]]
    gene_id_of = {
        gene: f"G{position:03d}" for position, gene in enumerate(single_names, 1)
    }
    single_effects = {
        gene_id_of[gene]: reduced[index[gene]] - control for gene in single_names
    }
    train_ids = {
        pair: f"D{position:03d}" for position, pair in enumerate(training_pairs, 1)
    }
    candidate_ids = {
        pair: f"C{position:03d}" for position, pair in enumerate(candidate_pairs, 1)
    }

    def additive(pair: str) -> np.ndarray:
        return sum(
            (single_effects[gene_id_of[gene]] for gene in pair.split("+")),
            np.zeros(PANEL_SIZE),
        )

    train_observed = np.stack(
        [reduced[index[pair]] - control for pair in training_pairs]
    )
    train_additive = np.stack([additive(pair) for pair in training_pairs])
    train_residuals = train_observed - train_additive
    scale = float(np.median(np.linalg.norm(train_residuals, axis=1)))
    if scale <= 0:
        raise RuntimeError("training residual scale is zero")
    normalized = train_residuals / scale
    _, _, vt = np.linalg.svd(normalized - normalized.mean(axis=0), full_matrices=False)
    basis = vt[: min(8, len(vt))]
    train_latent = normalized @ basis.T
    centers, train_labels = _kmeans(train_latent, PROGRAM_COUNT)
    train_strengths = np.clip(np.linalg.norm(train_residuals, axis=1) / scale, 0, 2)
    strong_threshold = float(np.quantile(train_strengths, 0.75))

    candidate_components = {
        candidate_ids[pair]: [gene_id_of[gene] for gene in pair.split("+")]
        for pair in candidate_pairs
    }
    candidate_residuals = {
        candidate_ids[pair]: reduced[index[pair]] - control - additive(pair)
        for pair in candidate_pairs
    }
    candidate_strengths = {
        candidate: float(np.clip(np.linalg.norm(residual) / scale, 0, 2))
        for candidate, residual in candidate_residuals.items()
    }
    candidate_programs = {
        candidate: int(
            np.argmin(
                np.sum((residual[None, :] / scale @ basis.T - centers) ** 2, axis=1)
            )
        )
        for candidate, residual in candidate_residuals.items()
    }
    candidate_strong = {
        candidate: strength >= strong_threshold
        for candidate, strength in candidate_strengths.items()
    }
    candidate_list = sorted(candidate_components)
    oracle, oracle_utility = _exact_oracle(
        candidate_list,
        candidate_components,
        candidate_strengths,
        candidate_programs,
        candidate_strong,
    )

    public_data = {
        "version": 1,
        "benchmark_id": BENCHMARK_ID,
        "feature_ids": [f"F{position:03d}" for position in range(1, PANEL_SIZE + 1)],
        "gene_ids": sorted(single_effects),
        "training_pair_ids": [train_ids[pair] for pair in training_pairs],
        "candidate_ids": candidate_list,
        "fold_count": FOLD_COUNT,
        "batch_size": BATCH_SIZE,
        "gene_cap": GENE_CAP,
        "program_count": PROGRAM_COUNT,
        "coverage_bonus": COVERAGE_BONUS,
        "strength_definition": (
            "clipped L2 norm of hidden residual divided by the frozen median "
            "training-residual norm"
        ),
        "strong_threshold": round(strong_threshold, 8),
        "candidate_components": candidate_components,
        "single_effects": {
            gene: _round_vector(vector) for gene, vector in single_effects.items()
        },
        "training_pairs": {
            train_ids[pair]: {
                "components": [gene_id_of[gene] for gene in pair.split("+")],
                "observed_delta": _round_vector(train_observed[row]),
                "fold": _fold(pair),
                "cell_count": int(cell_counts[index[pair]]),
                "training_program": int(train_labels[row]),
            }
            for row, pair in enumerate(training_pairs)
        },
    }
    private_truth = {
        "version": 1,
        "source_sha256": SOURCE_SHA256,
        "scale": scale,
        "strong_threshold": strong_threshold,
        "candidates": {
            candidate_ids[pair]: {
                "condition": pair,
                "components": candidate_components[candidate_ids[pair]],
                "strength": candidate_strengths[candidate_ids[pair]],
                "program": candidate_programs[candidate_ids[pair]],
                "strong": candidate_strong[candidate_ids[pair]],
                "cell_count": int(cell_counts[index[pair]]),
            }
            for pair in candidate_pairs
        },
        "oracle_selection": oracle,
        "oracle_utility": oracle_utility,
    }

    reasoning_contract = {
        "artifact_required_keys": [
            "candidate_solution",
            "constraint_results",
            "certificate",
            "conclusion",
            "hypotheses",
            "experiments",
            "model_comparison",
            "validation",
        ],
        "minimum_broker_calls": 4,
        "minimum_model_configurations": 3,
        "required_tool_prefix": "portfolio.",
        "required_compute_suffix": "_lab",
        "runtime_budget": {
            "max_demigod_turns": 28,
            "max_tokens_per_model_turn": 8192,
            "demigod_wall_time_s": 2400,
            "max_broker_calls": 48,
        },
        "requirements": [
            "Every worker must solve the full ten-item portfolio.",
            "Use frozen training folds for model selection and report validation.",
            "Respect the two-use cap for every component symbol.",
            "Return exactly ten distinct C identifiers in "
            "candidate_solution.selection.",
            "Include a robustness ablation and a machine-checkable "
            "selection certificate.",
        ],
    }
    public_question = {
        "id": BENCHMARK_ID,
        "statement": (
            "Choose a ten-combination CRISPRa Perturb-seq follow-up batch from "
            "real unmeasured Norman K562 combinations to maximize discovery of "
            "strong, diverse non-additive genetic interactions."
        ),
        "entities": sorted(
            set(
                single_names
                + [gene for pair in candidate_pairs for gene in pair.split("+")]
            )
        ),
        "sensitive_terms": ["K562", "CRISPRa", "Perturb-seq", "Norman"],
        "constraints": [
            "Select exactly ten distinct candidate combinations.",
            "No component gene may appear in more than two selected combinations.",
            "Candidate outcomes and utilities are evaluator-private until scoring.",
            "Use only frozen public training evidence for fitting and model selection.",
            "Every demigod must return an independently complete batch, "
            "not a partial list.",
        ],
        "question": (
            "Which ten candidate pairs should be run next to maximize normalized "
            "non-additive interaction strength plus coverage of distinct strong "
            "response programs?"
        ),
        "inputs": {
            "study": {
                "assay": "single-cell Perturb-seq",
                "perturbation": "two-gene CRISPR activation",
                "cell_context": "K562",
                "source_accession": "GSE133344",
            },
            "split_protocol": {
                "salt": SPLIT_SALT,
                "selection_uses_expression": False,
                "memorization_exclusions": sorted(MEMORIZATION_EXCLUSIONS),
                "training_pair_count": len(training_pairs),
                "candidate_pair_count": len(candidate_pairs),
                "feature_selection": "training-only top pseudobulk variance",
            },
            "candidate_catalog": [
                {
                    "candidate_id": candidate_ids[pair],
                    "genes": pair.split("+"),
                    "component_ids": candidate_components[candidate_ids[pair]],
                }
                for pair in candidate_pairs
            ],
            "scoring_rule": {
                "batch_size": BATCH_SIZE,
                "gene_cap": GENE_CAP,
                "program_count": PROGRAM_COUNT,
                "coverage_bonus": COVERAGE_BONUS,
                "utility": (
                    "sum clipped normalized interaction strengths + "
                    "coverage_bonus * number of distinct strong programs"
                ),
                "primary_score": "10 * submitted_utility / exact_oracle_utility",
            },
            "broker_interfaces": [
                "portfolio.manifest",
                "portfolio.single_effects",
                "portfolio.training_pairs",
                "portfolio.algebra_lab",
                "portfolio.geometry_lab",
                "portfolio.graph_lab",
                "portfolio.optimization_lab",
            ],
            "reasoning_contract": reasoning_contract,
        },
        "required_outputs": [
            "exactly ten candidate identifiers",
            "constraint certificate for uniqueness and component-use caps",
            "training-fold model comparison",
            "robustness ablation and rationale for strength-diversity tradeoff",
        ],
        "answer_schema": {
            "type": "object",
            "required": ["selected_candidate_ids", "method_summary"],
            "properties": {
                "selected_candidate_ids": {
                    "type": "array",
                    "minItems": BATCH_SIZE,
                    "maxItems": BATCH_SIZE,
                    "uniqueItems": True,
                    "items": {"type": "string", "pattern": "^C[0-9]{3}$"},
                },
                "method_summary": {"type": "string"},
            },
        },
    }
    public_question["inputs"]["reasoning_contract"] = reasoning_contract

    _write_json(PUBLIC / "training_data.json", public_data)
    _write_json(PUBLIC / "question.json", public_question)
    _write_json(PRIVATE / "expected.json", private_truth)
    manifest = {
        "benchmark_id": BENCHMARK_ID,
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "split_salt": SPLIT_SALT,
        "frozen_before_model_calls": True,
        "candidate_selection_uses_expression": False,
        "feature_selection_uses_candidates": False,
        "program_basis_and_centers_use_training_only": True,
        "metric_hyperparameters_fixed": True,
        "public_question_sha256": _sha256(PUBLIC / "question.json"),
        "public_training_data_sha256": _sha256(PUBLIC / "training_data.json"),
        "private_expected_sha256": _sha256(PRIVATE / "expected.json"),
    }
    _write_json(PUBLIC / "freeze_manifest.json", manifest)
    TOOL_MODULE.write_text(
        '"""Generated training-only batch-design data. Do not edit."""\n\n'
        f"DATA = {public_data!r}\n",
        encoding="utf-8",
    )
    return {
        "training_pairs": len(training_pairs),
        "candidates": len(candidate_pairs),
        "oracle": oracle,
        "oracle_utility": oracle_utility,
        "strong_threshold": strong_threshold,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
