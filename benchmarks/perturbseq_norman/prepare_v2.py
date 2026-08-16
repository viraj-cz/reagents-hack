"""Freeze a fresh Norman split with raw training primitives, not answers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from benchmarks.perturbseq_norman.prepare import (
    MEMORIZATION_EXCLUSIONS,
    PANEL_SIZE,
    SOURCE,
    SOURCE_SHA256,
    SOURCE_URL,
    TEST_SIZE,
    _condition_means,
    _decode,
    _interaction_class,
    _round_vector,
    _select_test_pairs,
    _sha256,
)

CASE_DIR = Path(__file__).parent / "v2"
PUBLIC = CASE_DIR / "public"
PRIVATE = CASE_DIR / "private"
TOOL_MODULE = (
    Path(__file__).parents[2] / "src" / "reagents" / "tools" / "_norman_v2_public.py"
)
SPLIT_SALT = "reagents-norman-v2-primitives"
BENCHMARK_ID = "norman-perturbseq-primitives-heldout-v2"


def _select_v2_pairs(conditions: list[str]) -> list[str]:
    previous = set(_select_test_pairs(conditions))
    candidates = sorted(
        condition
        for condition in conditions
        if "+" in condition
        and condition not in MEMORIZATION_EXCLUSIONS
        and condition not in previous
    )
    return sorted(
        candidates,
        key=lambda condition: hashlib.sha256(
            f"{SPLIT_SALT}:{condition}".encode()
        ).hexdigest(),
    )[:TEST_SIZE]


def _fold(name: str, count: int = 5) -> int:
    return (
        int(hashlib.sha256(f"{SPLIT_SALT}:fold:{name}".encode()).hexdigest(), 16)
        % count
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def prepare(source: Path = SOURCE) -> dict[str, Any]:
    if _sha256(source) != SOURCE_SHA256:
        raise ValueError("source H5AD hash does not match the pinned Norman fixture")

    with h5py.File(source, "r") as handle:
        conditions = _decode(handle["obs/__categories/perturbation_name"][:])
        codes = handle["obs/perturbation_name"][:].astype(int)
        genes = _decode(handle["var/gene_symbols"][:])
        test_pairs = _select_v2_pairs(conditions)
        means, cell_counts = _condition_means(handle, codes, len(conditions))

    index = {condition: position for position, condition in enumerate(conditions)}
    train_conditions = [
        condition for condition in conditions if condition not in test_pairs
    ]
    train_rows = means[[index[condition] for condition in train_conditions]]
    feature_indices = np.argsort(-np.var(train_rows, axis=0), kind="stable")[
        :PANEL_SIZE
    ]
    feature_genes = [genes[position] for position in feature_indices]
    feature_ids = [f"F{position:03d}" for position in range(1, PANEL_SIZE + 1)]
    reduced = means[:, feature_indices]
    control = reduced[index["control"]]
    single_names = sorted(
        condition
        for condition in conditions
        if "+" not in condition and condition != "control"
    )
    gene_id_of = {
        gene: f"G{position:03d}" for position, gene in enumerate(single_names, start=1)
    }
    singles = {
        gene_id_of[gene]: reduced[index[gene]] - control for gene in single_names
    }
    train_pairs = sorted(
        condition for condition in train_conditions if "+" in condition
    )
    pair_id_of = {
        pair: f"D{position:03d}" for position, pair in enumerate(train_pairs, start=1)
    }
    target_ids = [f"T{position:02d}" for position in range(1, TEST_SIZE + 1)]

    train_observed = np.stack([reduced[index[pair]] - control for pair in train_pairs])
    train_additive = np.stack(
        [
            sum(
                (singles[gene_id_of[gene]] for gene in pair.split("+")),
                np.zeros(PANEL_SIZE),
            )
            for pair in train_pairs
        ]
    )
    ratios = np.linalg.norm(train_observed - train_additive, axis=1) / np.maximum(
        np.linalg.norm(train_additive, axis=1), 1e-8
    )
    class_threshold = float(np.quantile(ratios, 0.25))

    public_data = {
        "version": 2,
        "benchmark_id": BENCHMARK_ID,
        "feature_ids": feature_ids,
        "gene_ids": sorted(singles),
        "target_ids": target_ids,
        "training_pair_ids": [pair_id_of[pair] for pair in train_pairs],
        "fold_count": 5,
        "interaction_classes": [
            "additive",
            "synergy",
            "suppression",
            "neomorphic",
        ],
        "class_rule": {
            "residual_ratio_threshold": round(class_threshold, 8),
            "alignment_threshold": 0.25,
            "definition": (
                "residual=observed-additive; below ratio threshold is additive; "
                "otherwise cosine >0.25 synergy, <-0.25 suppression, else "
                "neomorphic"
            ),
        },
        "target_components": {
            target_id: [gene_id_of[gene] for gene in pair.split("+")]
            for target_id, pair in zip(target_ids, test_pairs, strict=True)
        },
        "single_effects": {
            gene_id: _round_vector(vector) for gene_id, vector in singles.items()
        },
        "training_pairs": {
            pair_id_of[pair]: {
                "components": [gene_id_of[gene] for gene in pair.split("+")],
                "observed_delta": _round_vector(train_observed[row]),
                "fold": _fold(pair),
                "cell_count": int(cell_counts[index[pair]]),
            }
            for row, pair in enumerate(train_pairs)
        },
    }

    target_additive = {
        target_id: sum(
            (singles[gene_id_of[gene]] for gene in pair.split("+")),
            np.zeros(PANEL_SIZE),
        )
        for target_id, pair in zip(target_ids, test_pairs, strict=True)
    }
    private_truth = {
        "version": 2,
        "source_sha256": SOURCE_SHA256,
        "targets": {
            target_id: {
                "condition": pair,
                "observed_delta": _round_vector(reduced[index[pair]] - control),
                "interaction_class": _interaction_class(
                    reduced[index[pair]] - control,
                    target_additive[target_id],
                    class_threshold,
                ),
                "cell_count": int(cell_counts[index[pair]]),
            }
            for target_id, pair in zip(target_ids, test_pairs, strict=True)
        },
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
        "minimum_model_configurations": 2,
        "required_tool_prefix": "screen2.",
        "required_compute_suffix": "_lab",
        "requirements": [
            "Fit or construct predictions during the run; no target candidate is "
            "available from a data tool.",
            "Compare at least two configurations using only frozen training folds.",
            "Perform at least one ablation, perturbation, or counterexample check.",
            "Return one independently complete candidate for all targets.",
        ],
    }
    public_question = {
        "id": BENCHMARK_ID,
        "statement": (
            "Predict a fresh set of 12 held-out two-gene CRISPRa responses from "
            "the Norman K562 Perturb-seq screen. Raw training pseudobulk effects, "
            "opaque component relationships, and deterministic compute labs are "
            "available, but no target prediction is precomputed."
        ),
        "entities": sorted(set(feature_genes + single_names)),
        "sensitive_terms": ["K562", "CRISPRa", "Norman"],
        "constraints": [
            "Held-out double-condition measurements are unavailable until private "
            "scoring.",
            "Use only T01-T12, F001-F064, G identifiers, and D identifiers across "
            "the Broker boundary.",
            "Every target requires 64 ordered values, one allowed interaction "
            "class, confidence, and a falsifier.",
            "All fitting, model selection, and validation must use the frozen "
            "training folds only.",
            "Do not claim a target prediction was supplied by a tool; agents must "
            "construct predictions through their representation.",
        ],
        "question": (
            "For every target T01-T12, construct its control-relative pseudobulk "
            "delta on F001-F064 and genetic-interaction class. Explore the complete "
            "problem in a transformed representation, compare models on training "
            "folds, perform a robustness check, and state a falsifier."
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
                "heldout_count": TEST_SIZE,
                "prior_heldout_pairs_excluded": True,
                "selection_uses_expression": False,
                "feature_selection": "training-only top pseudobulk variance",
            },
            "target_catalog": [
                {
                    "target_id": target_id,
                    "genes": pair.split("+"),
                    "component_ids": [gene_id_of[gene] for gene in pair.split("+")],
                }
                for target_id, pair in zip(target_ids, test_pairs, strict=True)
            ],
            "feature_catalog": [
                {"feature_id": feature_id, "gene": gene}
                for feature_id, gene in zip(feature_ids, feature_genes, strict=True)
            ],
            "gene_alias_catalog": [
                {"gene_id": gene_id_of[gene], "gene": gene} for gene in single_names
            ],
            "broker_interfaces": {
                "screen2.training_manifest": "training IDs, folds, and shapes",
                "screen2.single_effects": "raw training single-condition vectors",
                "screen2.training_pairs": "raw observed training double vectors",
                "screen2.algebra_lab": "execute agent-authored training-only models",
                "screen2.geometry_lab": "execute agent-authored training-only models",
                "screen2.graph_lab": "execute agent-authored training-only models",
                "screen2.information_lab": (
                    "execute agent-authored training-only models"
                ),
            },
            "reasoning_contract": reasoning_contract,
        },
        "required_outputs": [
            "complete candidate_solution for T01-T12 with 64 values each",
            "hypotheses considered in the invented representation",
            "experiments backed by the Broker call ledger",
            "comparison of at least two training-validated model configurations",
            "ablation, counterexample, or robustness validation",
            "constraint results, certificate, conclusion, confidence, and falsifiers",
        ],
        "answer_schema": {
            "type": "object",
            "required": ["predictions", "method_summary"],
            "properties": {
                "predictions": {
                    "type": "array",
                    "minItems": TEST_SIZE,
                    "maxItems": TEST_SIZE,
                    "items": {
                        "type": "object",
                        "required": [
                            "target_id",
                            "predicted_delta",
                            "interaction_class",
                            "confidence",
                            "falsifier",
                        ],
                    },
                },
                "method_summary": {"type": "string"},
            },
        },
    }

    _write_json(PUBLIC / "training_data.json", public_data)
    _write_json(PUBLIC / "question.json", public_question)
    private_path = PRIVATE / "expected.json"
    _write_json(private_path, private_truth)
    manifest = {
        "benchmark_id": BENCHMARK_ID,
        "frozen_before_model_calls": True,
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "split_salt": SPLIT_SALT,
        "prepare_script_sha256": _sha256(Path(__file__)),
        "prior_heldout_pairs_excluded": True,
        "test_pair_selection_uses_expression": False,
        "test_pairs_sha256": hashlib.sha256(
            json.dumps(test_pairs).encode()
        ).hexdigest(),
        "private_expected_sha256": _sha256(private_path),
        "public_question_sha256": _sha256(PUBLIC / "question.json"),
        "public_training_data_sha256": _sha256(PUBLIC / "training_data.json"),
    }
    _write_json(PUBLIC / "freeze_manifest.json", manifest)
    TOOL_MODULE.write_text(
        '"""Generated training-only Norman v2 data. Do not edit."""\n\n'
        f"DATA = {public_data!r}\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
