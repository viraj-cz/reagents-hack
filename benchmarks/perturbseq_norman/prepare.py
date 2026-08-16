"""Freeze a real Norman Perturb-seq split without consulting test outcomes.

The source is the processed Norman 2019 H5AD indexed by scPerturb. Pair
selection depends only on condition labels and a public salt. Feature selection,
model fitting, calibration, and every diagnostic exposed to agents use training
conditions only. Held-out pseudobulk values are written solely to ``private``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import sparse

CASE_DIR = Path(__file__).parent
SOURCE = CASE_DIR / "data" / "source" / "Norman_2019.h5ad"
PUBLIC = CASE_DIR / "public"
PRIVATE = CASE_DIR / "private"
TOOL_MODULE = (
    Path(__file__).parents[2] / "src" / "reagents" / "tools" / "_norman_public.py"
)

SOURCE_URL = "https://ndownloader.figshare.com/files/34027562"
SOURCE_SHA256 = "b679c157fea550ef5be4dad91da9dff1f5d8287313ad3f3537f6cf14d4bcd434"
SPLIT_SALT = "reagents-norman-v1"
PANEL_SIZE = 64
TEST_SIZE = 12
CHUNK_ROWS = 2_000

# Excluded before hashing because these pairs are named in the Norman/GEARS
# repositories, paper examples, or widely repeated demos. This is a
# memorization guard, not an outcome-based filter.
MEMORIZATION_EXCLUSIONS = frozenset(
    {
        "CBL+CNN1",
        "CEBPB+FOSB",
        "DUSP9+ETS2",
        "DUSP9+MAPK1",
        "PTPN12+ZBTB25",
    }
)

METHODS = ("additive", "ridge", "latent", "geometry", "graph")
CLASSES = ("additive", "synergy", "suppression", "neomorphic")


def _decode(values: Any) -> list[str]:
    return [
        value.decode() if isinstance(value, bytes) else str(value) for value in values
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_test_pairs(conditions: list[str]) -> list[str]:
    candidates = sorted(
        condition
        for condition in conditions
        if "+" in condition and condition not in MEMORIZATION_EXCLUSIONS
    )
    return sorted(
        candidates,
        key=lambda condition: hashlib.sha256(
            f"{SPLIT_SALT}:{condition}".encode()
        ).hexdigest(),
    )[:TEST_SIZE]


def _condition_means(
    handle: h5py.File, codes: np.ndarray, n_conditions: int
) -> tuple[np.ndarray, np.ndarray]:
    matrix = handle["X"]
    shape = tuple(int(value) for value in matrix.attrs["shape"])
    indptr = matrix["indptr"][:]
    sums = np.zeros((n_conditions, shape[1]), dtype=np.float64)
    counts = np.bincount(codes, minlength=n_conditions).astype(np.int64)
    for start in range(0, shape[0], CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, shape[0])
        low, high = int(indptr[start]), int(indptr[stop])
        chunk = sparse.csr_matrix(
            (
                matrix["data"][low:high],
                matrix["indices"][low:high],
                indptr[start : stop + 1] - low,
            ),
            shape=(stop - start, shape[1]),
        )
        rows = np.arange(stop - start)
        groups = sparse.csr_matrix(
            (np.ones(stop - start), (rows, codes[start:stop])),
            shape=(stop - start, n_conditions),
        )
        sums += (groups.T @ chunk).toarray()
    return sums / counts[:, None], counts


def _pair_features(single: dict[str, np.ndarray], condition: str) -> np.ndarray:
    left, right = condition.split("+")
    a, b = single[left], single[right]
    return np.concatenate((a + b, np.abs(a - b), a * b, np.maximum(a, b)))


def _standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (train - mean) / scale, (other - mean) / scale


def _ridge_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float
) -> np.ndarray:
    xm, ym = x_train.mean(axis=0), y_train.mean(axis=0)
    xc, yc = x_train - xm, y_train - ym
    dual = np.linalg.solve(xc @ xc.T + alpha * np.eye(len(xc)), yc)
    return ym + (x_test - xm) @ xc.T @ dual


def _folds(names: list[str], count: int = 5) -> np.ndarray:
    return np.array(
        [
            int(hashlib.sha256(f"{SPLIT_SALT}:fold:{name}".encode()).hexdigest(), 16)
            % count
            for name in names
        ]
    )


def _mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((actual - predicted) ** 2))


def _ridge_oof(
    x: np.ndarray, y: np.ndarray, names: list[str]
) -> tuple[float, np.ndarray, float]:
    fold_ids = _folds(names)
    best: tuple[float, float, np.ndarray] | None = None
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
        out = np.zeros_like(y)
        for fold in range(5):
            train, test = fold_ids != fold, fold_ids == fold
            out[test] = _ridge_predict(x[train], y[train], x[test], alpha)
        score = _mse(y, out)
        if best is None or score < best[0]:
            best = (score, alpha, out)
    assert best is not None
    return best


def _knn_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, k: int
) -> tuple[np.ndarray, list[list[int]], list[list[float]]]:
    train_norm = np.linalg.norm(x_train, axis=1)
    test_norm = np.linalg.norm(x_test, axis=1)
    similarity = (
        x_test @ x_train.T / np.maximum(test_norm[:, None] * train_norm[None, :], 1e-8)
    )
    indices = np.argsort(-similarity, axis=1)[:, :k]
    predictions, weights_out = [], []
    for row, neighbors in enumerate(indices):
        weights = np.maximum(similarity[row, neighbors], 0.0) + 1e-3
        weights /= weights.sum()
        predictions.append(weights @ y_train[neighbors])
        weights_out.append(weights.tolist())
    return np.asarray(predictions), indices.tolist(), weights_out


def _knn_oof(
    x: np.ndarray, y: np.ndarray, names: list[str]
) -> tuple[float, int, np.ndarray]:
    folds = _folds(names)
    best: tuple[float, int, np.ndarray] | None = None
    for k in (3, 5, 8, 12):
        out = np.zeros_like(y)
        for fold in range(5):
            train, test = folds != fold, folds == fold
            xt, xv = _standardize(x[train], x[test])
            out[test] = _knn_predict(xt, y[train], xv, min(k, int(train.sum())))[0]
        score = _mse(y, out)
        if best is None or score < best[0]:
            best = (score, k, out)
    assert best is not None
    return best


def _latent_oof(
    x: np.ndarray, y: np.ndarray, names: list[str]
) -> tuple[float, tuple[int, float], np.ndarray]:
    folds = _folds(names)
    best: tuple[float, tuple[int, float], np.ndarray] | None = None
    for rank in (4, 8, 12):
        for alpha in (1.0, 10.0, 100.0):
            out = np.zeros_like(y)
            for fold in range(5):
                train, test = folds != fold, folds == fold
                center = y[train].mean(axis=0)
                _, _, vt = np.linalg.svd(y[train] - center, full_matrices=False)
                basis = vt[:rank]
                coords = (y[train] - center) @ basis.T
                predicted = _ridge_predict(x[train], coords, x[test], alpha)
                out[test] = center + predicted @ basis
            score = _mse(y, out)
            if best is None or score < best[0]:
                best = (score, (rank, alpha), out)
    assert best is not None
    return best


def _graph_features(
    single: dict[str, np.ndarray], pairs: list[str], targets: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    genes = sorted(single)
    vectors = np.stack([single[gene] for gene in genes])
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    lookup = {gene: index for index, gene in enumerate(genes)}

    def features(condition: str) -> np.ndarray:
        a, b = condition.split("+")
        return np.concatenate((vectors[lookup[a]], vectors[lookup[b]]))

    return np.stack([features(pair) for pair in pairs]), np.stack(
        [features(pair) for pair in targets]
    )


def _interaction_class(
    observed: np.ndarray, additive: np.ndarray, threshold: float
) -> str:
    residual = observed - additive
    ratio = np.linalg.norm(residual) / max(np.linalg.norm(additive), 1e-8)
    if ratio < threshold:
        return "additive"
    cosine = float(
        residual
        @ additive
        / max(np.linalg.norm(residual) * np.linalg.norm(additive), 1e-8)
    )
    if cosine > 0.25:
        return "synergy"
    if cosine < -0.25:
        return "suppression"
    return "neomorphic"


def _round_vector(values: np.ndarray) -> list[float]:
    return np.round(values.astype(float), 6).tolist()


def prepare(source: Path = SOURCE) -> dict[str, Any]:
    if _sha256(source) != SOURCE_SHA256:
        raise ValueError("source H5AD hash does not match the pinned Norman fixture")

    with h5py.File(source, "r") as handle:
        conditions = _decode(handle["obs/__categories/perturbation_name"][:])
        codes = handle["obs/perturbation_name"][:].astype(int)
        genes = _decode(handle["var/gene_symbols"][:])
        test_pairs = _select_test_pairs(conditions)
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
    reduced = means[:, feature_indices]
    control = reduced[index["control"]]
    singles = {
        condition: reduced[index[condition]] - control
        for condition in conditions
        if "+" not in condition and condition != "control"
    }
    train_pairs = sorted(
        condition for condition in train_conditions if "+" in condition
    )
    train_observed = np.stack([reduced[index[pair]] - control for pair in train_pairs])
    train_additive = np.stack(
        [
            sum((singles[gene] for gene in pair.split("+")), np.zeros(PANEL_SIZE))
            for pair in train_pairs
        ]
    )
    residual = train_observed - train_additive

    x_train_raw = np.stack([_pair_features(singles, pair) for pair in train_pairs])
    x_target_raw = np.stack([_pair_features(singles, pair) for pair in test_pairs])
    x_train, x_target = _standardize(x_train_raw, x_target_raw)

    ridge_mse, ridge_alpha, _ = _ridge_oof(x_train, residual, train_pairs)
    ridge_residual = _ridge_predict(x_train, residual, x_target, ridge_alpha)

    geometry_mse, geometry_k, _ = _knn_oof(x_train_raw, residual, train_pairs)
    geometry_residual, geometry_neighbors, geometry_weights = _knn_predict(
        x_train, residual, x_target, geometry_k
    )

    latent_mse, (latent_rank, latent_alpha), _ = _latent_oof(
        x_train, residual, train_pairs
    )
    center = residual.mean(axis=0)
    _, _, vt = np.linalg.svd(residual - center, full_matrices=False)
    basis = vt[:latent_rank]
    latent_coords = (residual - center) @ basis.T
    latent_residual = (
        center + _ridge_predict(x_train, latent_coords, x_target, latent_alpha) @ basis
    )

    graph_train_raw, graph_target_raw = _graph_features(
        singles, train_pairs, test_pairs
    )
    graph_train, graph_target = _standardize(graph_train_raw, graph_target_raw)
    graph_mse, graph_k, _ = _knn_oof(graph_train_raw, residual, train_pairs)
    graph_residual, graph_neighbors, graph_weights = _knn_predict(
        graph_train, residual, graph_target, graph_k
    )

    additive_mse = _mse(residual, np.zeros_like(residual))
    cv_mse = {
        "additive": additive_mse,
        "ridge": ridge_mse,
        "latent": latent_mse,
        "geometry": geometry_mse,
        "graph": graph_mse,
    }
    target_additive = np.stack(
        [
            sum((singles[gene] for gene in pair.split("+")), np.zeros(PANEL_SIZE))
            for pair in test_pairs
        ]
    )
    candidates = {
        "additive": target_additive,
        "ridge": target_additive + ridge_residual,
        "latent": target_additive + latent_residual,
        "geometry": target_additive + geometry_residual,
        "graph": target_additive + graph_residual,
    }
    weights = {method: 1.0 / max(score, 1e-8) for method, score in cv_mse.items()}
    total_weight = sum(weights.values())
    weights = {method: value / total_weight for method, value in weights.items()}
    ensemble = sum(
        (weights[method] * candidates[method] for method in METHODS),
        np.zeros_like(target_additive),
    )

    ratios = np.linalg.norm(residual, axis=1) / np.maximum(
        np.linalg.norm(train_additive, axis=1), 1e-8
    )
    class_threshold = float(np.quantile(ratios, 0.25))
    train_classes = [
        _interaction_class(observed, additive, class_threshold)
        for observed, additive in zip(train_observed, train_additive, strict=True)
    ]
    feature_ids = [f"F{position:03d}" for position in range(1, PANEL_SIZE + 1)]
    target_ids = [f"T{position:02d}" for position in range(1, TEST_SIZE + 1)]

    tool_targets: dict[str, Any] = {}
    for row, (target_id, _pair) in enumerate(zip(target_ids, test_pairs, strict=True)):
        method_records = {}
        for method in METHODS:
            prediction = candidates[method][row]
            method_records[method] = {
                "predicted_delta": _round_vector(prediction),
                "predicted_interaction_class": _interaction_class(
                    prediction, target_additive[row], class_threshold
                ),
                "cv_mse": round(cv_mse[method], 8),
            }
        method_records["ensemble"] = {
            "predicted_delta": _round_vector(ensemble[row]),
            "predicted_interaction_class": _interaction_class(
                ensemble[row], target_additive[row], class_threshold
            ),
            "cv_mse": round(sum(weights[m] * cv_mse[m] for m in METHODS), 8),
        }
        geometry_analogs = [
            {
                "pair_id": f"D{train_pairs.index(train_pairs[i]) + 1:03d}",
                "weight": round(float(weight), 6),
                "interaction_class": train_classes[i],
            }
            for i, weight in zip(
                geometry_neighbors[row], geometry_weights[row], strict=True
            )
        ]
        graph_analogs = [
            {
                "pair_id": f"D{train_pairs.index(train_pairs[i]) + 1:03d}",
                "weight": round(float(weight), 6),
                "interaction_class": train_classes[i],
            }
            for i, weight in zip(graph_neighbors[row], graph_weights[row], strict=True)
        ]
        tool_targets[target_id] = {
            "methods": method_records,
            "ensemble_weights": {
                key: round(value, 6) for key, value in weights.items()
            },
            "geometry_analogs": geometry_analogs,
            "graph_analogs": graph_analogs,
            "cross_method_spread": round(
                float(
                    np.mean(
                        np.std(np.stack([candidates[m][row] for m in METHODS]), axis=0)
                    )
                ),
                8,
            ),
        }

    public_data = {
        "version": 1,
        "feature_ids": feature_ids,
        "target_ids": target_ids,
        "methods": list(METHODS),
        "interaction_classes": list(CLASSES),
        "class_threshold": round(class_threshold, 8),
        "cv_mse": {key: round(value, 8) for key, value in cv_mse.items()},
        "targets": tool_targets,
    }
    private_truth = {
        "version": 1,
        "source_sha256": SOURCE_SHA256,
        "targets": {
            target_id: {
                "condition": pair,
                "observed_delta": _round_vector(reduced[index[pair]] - control),
                "interaction_class": _interaction_class(
                    reduced[index[pair]] - control,
                    target_additive[row],
                    class_threshold,
                ),
                "cell_count": int(cell_counts[index[pair]]),
            }
            for row, (target_id, pair) in enumerate(
                zip(target_ids, test_pairs, strict=True)
            )
        },
    }
    public_question = {
        "id": "norman-perturbseq-heldout-combinations-v1",
        "statement": (
            "Predict the real transcriptomic response of 12 held-out two-gene CRISPRa "
            "conditions from the Norman et al. K562 Perturb-seq screen. All controls, "
            "single perturbations, and remaining double perturbations are training "
            "evidence."
        ),
        "entities": sorted(
            set(
                feature_genes
                + [gene for pair in test_pairs for gene in pair.split("+")]
            )
        ),
        "sensitive_terms": ["K562", "CRISPRa", "Norman"],
        "constraints": [
            "The held-out double-condition expression measurements are unavailable "
            "until private scoring.",
            "Use only T01-T12 and F001-F064 as Broker and prediction interface "
            "identifiers.",
            "Every target requires 64 ordered delta-expression values, one interaction "
            "class, confidence, and a falsifier.",
            "Allowed interaction classes are additive, synergy, suppression, and "
            "neomorphic under the frozen training-derived rule.",
            "Do not claim access to raw cells or held-out outcomes; deterministic "
            "Broker tools expose training-only candidates and diagnostics.",
        ],
        "question": (
            "For every target T01-T12, predict its control-relative pseudobulk "
            "expression delta on F001-F064 and its genetic-interaction class. "
            "Explain which transformed evidence supports each complete candidate and "
            "state what result would falsify it."
        ),
        "inputs": {
            "study": {
                "assay": "single-cell Perturb-seq",
                "perturbation": "two-gene CRISPR activation",
                "cell_context": "K562",
                "source_accession": "GSE133344",
                "normalization": (
                    "curated per-cell normalized log expression from scPerturb"
                ),
            },
            "split_protocol": {
                "salt": SPLIT_SALT,
                "selection": "SHA-256 rank of eligible condition labels only",
                "heldout_count": TEST_SIZE,
                "feature_selection": (
                    "top pseudobulk variance using training conditions only"
                ),
                "memorization_exclusions": sorted(MEMORIZATION_EXCLUSIONS),
            },
            "target_catalog": [
                {"target_id": target_id, "genes": pair.split("+")}
                for target_id, pair in zip(target_ids, test_pairs, strict=True)
            ],
            "feature_catalog": [
                {"feature_id": feature_id, "gene": gene}
                for feature_id, gene in zip(feature_ids, feature_genes, strict=True)
            ],
            "broker_interfaces": {
                "screen.algebra_candidate": (
                    "additive and supervised residual-algebra candidates"
                ),
                "screen.algebra_certificate": (
                    "training-only residual-algebra cross-validation audit"
                ),
                "screen.geometry_candidate": (
                    "nearest-neighbor interaction-manifold candidate"
                ),
                "screen.geometry_certificate": (
                    "anonymous manifold-analog and cross-validation audit"
                ),
                "screen.graph_candidate": (
                    "single-effect similarity-graph analog candidate"
                ),
                "screen.graph_certificate": (
                    "anonymous graph-edge-match and cross-validation audit"
                ),
                "screen.information_candidate": (
                    "cross-representation ensemble and uncertainty diagnostics"
                ),
                "screen.information_certificate": (
                    "ensemble-weight and disagreement audit"
                ),
            },
        },
        "required_outputs": [
            "exactly one prediction for every target T01-T12",
            "64 numeric predicted_delta values in F001-F064 order per target",
            "one allowed interaction_class per target",
            "confidence from 0 to 1 and a concrete falsifier per target",
            "method summary distinguishing data-backed inference from uncertainty",
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

    PUBLIC.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    (PUBLIC / "tool_data.json").write_text(
        json.dumps(public_data, indent=2), encoding="utf-8"
    )
    (PUBLIC / "question.json").write_text(
        json.dumps(public_question, indent=2), encoding="utf-8"
    )
    private_path = PRIVATE / "expected.json"
    private_path.write_text(json.dumps(private_truth, indent=2), encoding="utf-8")
    manifest = {
        "benchmark_id": public_question["id"],
        "frozen_before_model_calls": True,
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "split_salt": SPLIT_SALT,
        "prepare_script_sha256": _sha256(Path(__file__)),
        "test_pair_selection_uses_expression": False,
        "test_pairs_sha256": hashlib.sha256(
            json.dumps(test_pairs).encode()
        ).hexdigest(),
        "private_expected_sha256": _sha256(private_path),
        "public_question_sha256": _sha256(PUBLIC / "question.json"),
        "public_tool_data_sha256": _sha256(PUBLIC / "tool_data.json"),
    }
    (PUBLIC / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    TOOL_MODULE.write_text(
        '"""Generated training-only Norman benchmark data. Do not edit."""\n\n'
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
