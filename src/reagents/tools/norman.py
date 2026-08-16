"""Training-only Broker views for the frozen Norman perturbation benchmark."""

from __future__ import annotations

from typing import Any

from reagents.contracts import RiskTier, ToolAccess
from reagents.tools._norman_public import DATA
from reagents.tools.registry import Tool


def _targets(target_ids: list[str]) -> list[tuple[str, dict[str, Any]]]:
    if not target_ids:
        raise ValueError("target_ids must contain at least one target")
    if len(target_ids) > len(DATA["target_ids"]):
        raise ValueError("too many target ids")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("target_ids must be unique")
    unknown = sorted(set(target_ids) - set(DATA["target_ids"]))
    if unknown:
        raise ValueError(f"unknown target ids: {unknown}")
    return [(target_id, DATA["targets"][target_id]) for target_id in target_ids]


def algebra_candidate(target_ids: list[str]) -> dict[str, Any]:
    """Return additive and supervised residual-algebra candidates."""

    return {
        "coordinate_system": "interaction_residual_algebra",
        "feature_ids": DATA["feature_ids"],
        "training_only_cv_mse": DATA["cv_mse"],
        "targets": {
            target_id: {
                "additive": record["methods"]["additive"],
                "ridge": record["methods"]["ridge"],
            }
            for target_id, record in _targets(target_ids)
        },
    }


def geometry_candidate(target_ids: list[str]) -> dict[str, Any]:
    """Return interaction-manifold and low-rank latent candidates."""

    return {
        "coordinate_system": "latent_interaction_geometry",
        "feature_ids": DATA["feature_ids"],
        "training_only_cv_mse": DATA["cv_mse"],
        "targets": {
            target_id: {
                "geometry": record["methods"]["geometry"],
                "latent": record["methods"]["latent"],
                "nearest_training_analogs": record["geometry_analogs"],
            }
            for target_id, record in _targets(target_ids)
        },
    }


def graph_candidate(target_ids: list[str]) -> dict[str, Any]:
    """Return single-effect similarity-graph analog candidates."""

    return {
        "coordinate_system": "signed_similarity_graph",
        "feature_ids": DATA["feature_ids"],
        "training_only_cv_mse": DATA["cv_mse"],
        "targets": {
            target_id: {
                "graph": record["methods"]["graph"],
                "matched_training_edges": record["graph_analogs"],
            }
            for target_id, record in _targets(target_ids)
        },
    }


def information_candidate(target_ids: list[str]) -> dict[str, Any]:
    """Return a train-calibrated ensemble and cross-method uncertainty."""

    return {
        "coordinate_system": "information_weighted_consensus",
        "feature_ids": DATA["feature_ids"],
        "training_only_cv_mse": DATA["cv_mse"],
        "targets": {
            target_id: {
                "ensemble": record["methods"]["ensemble"],
                "ensemble_weights": record["ensemble_weights"],
                "cross_method_spread": record["cross_method_spread"],
            }
            for target_id, record in _targets(target_ids)
        },
    }


def algebra_certificate(target_ids: list[str]) -> dict[str, Any]:
    """Return train-CV evidence for residual algebra without held-out values."""

    return {
        "certificate": "training_only_cross_validation",
        "candidate_family": "interaction_residual_algebra",
        "cv_mse": {method: DATA["cv_mse"][method] for method in ("additive", "ridge")},
        "target_ids": [target_id for target_id, _ in _targets(target_ids)],
        "heldout_outcomes_used": False,
    }


def geometry_certificate(target_ids: list[str]) -> dict[str, Any]:
    """Return anonymous manifold-neighbor evidence and train-CV errors."""

    return {
        "certificate": "anonymous_training_analogs",
        "candidate_family": "latent_interaction_geometry",
        "cv_mse": {method: DATA["cv_mse"][method] for method in ("geometry", "latent")},
        "targets": {
            target_id: record["geometry_analogs"]
            for target_id, record in _targets(target_ids)
        },
        "heldout_outcomes_used": False,
    }


def graph_certificate(target_ids: list[str]) -> dict[str, Any]:
    """Return anonymous matched-edge evidence and train-CV errors."""

    return {
        "certificate": "signed_graph_edge_matches",
        "candidate_family": "signed_similarity_graph",
        "cv_mse": {"graph": DATA["cv_mse"]["graph"]},
        "targets": {
            target_id: record["graph_analogs"]
            for target_id, record in _targets(target_ids)
        },
        "heldout_outcomes_used": False,
    }


def information_certificate(target_ids: list[str]) -> dict[str, Any]:
    """Return ensemble weights and disagreement as an uncertainty certificate."""

    return {
        "certificate": "cross_representation_disagreement",
        "candidate_family": "information_weighted_consensus",
        "targets": {
            target_id: {
                "weights": record["ensemble_weights"],
                "spread": record["cross_method_spread"],
            }
            for target_id, record in _targets(target_ids)
        },
        "heldout_outcomes_used": False,
    }


_PARAMETERS = {
    "type": "object",
    "required": ["target_ids"],
    "properties": {
        "target_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "uniqueItems": True,
            "items": {"type": "string", "pattern": "^T(0[1-9]|1[0-2])$"},
        }
    },
    "additionalProperties": False,
}


def all_tools() -> list[Tool]:
    common = {
        "parameters_schema": _PARAMETERS,
        "namespace": "screen",
        "access": ToolAccess.READ,
        "risk_tier": RiskTier.LOW,
        "side_effects": (),
    }
    return [
        Tool(
            id="screen.algebra_candidate",
            description=(
                "For opaque targets T01-T12, return training-only additive and "
                "supervised interaction-residual predictions over F001-F064."
            ),
            fn=algebra_candidate,
            **common,
        ),
        Tool(
            id="screen.geometry_candidate",
            description=(
                "For opaque targets T01-T12, return training-only manifold-neighbor "
                "and low-rank latent predictions plus anonymous analog diagnostics."
            ),
            fn=geometry_candidate,
            **common,
        ),
        Tool(
            id="screen.graph_candidate",
            description=(
                "For opaque targets T01-T12, return training-only signed similarity-"
                "graph predictions and anonymous matched edges."
            ),
            fn=graph_candidate,
            **common,
        ),
        Tool(
            id="screen.information_candidate",
            description=(
                "For opaque targets T01-T12, return the train-CV-weighted consensus "
                "prediction and cross-representation uncertainty."
            ),
            fn=information_candidate,
            **common,
        ),
        Tool(
            id="screen.algebra_certificate",
            description=(
                "Audit training-only cross-validation evidence for additive and "
                "residual-algebra candidates; exposes no held-out outcomes."
            ),
            fn=algebra_certificate,
            **common,
        ),
        Tool(
            id="screen.geometry_certificate",
            description=(
                "Audit anonymous training-manifold analogs and latent/geometry "
                "cross-validation errors; exposes no held-out outcomes."
            ),
            fn=geometry_certificate,
            **common,
        ),
        Tool(
            id="screen.graph_certificate",
            description=(
                "Audit anonymous signed-graph edge matches and graph cross-validation "
                "error; exposes no held-out outcomes."
            ),
            fn=graph_certificate,
            **common,
        ),
        Tool(
            id="screen.information_certificate",
            description=(
                "Audit train-derived ensemble weights and cross-method disagreement "
                "for uncertainty calibration; exposes no held-out outcomes."
            ),
            fn=information_certificate,
            **common,
        ),
    ]


__all__ = ["all_tools"]
