"""Low-level training primitives for the exploratory Norman v2 benchmark."""

from __future__ import annotations

from typing import Any

from reagents.contracts import RiskTier, ToolAccess, ToolProvider
from reagents.tools._norman_v2_public import DATA
from reagents.tools.container import ContainerExecutor, ContainerToolConfig
from reagents.tools.registry import Tool


def training_manifest() -> dict[str, Any]:
    """Return identifiers, folds, shapes, and target component structure."""

    return {
        "benchmark_id": DATA["benchmark_id"],
        "feature_ids": DATA["feature_ids"],
        "gene_ids": DATA["gene_ids"],
        "target_ids": DATA["target_ids"],
        "training_pair_ids": DATA["training_pair_ids"],
        "fold_count": DATA["fold_count"],
        "class_rule": DATA["class_rule"],
        "target_components": DATA["target_components"],
        "heldout_outcomes_present": False,
    }


def single_effects(gene_ids: list[str]) -> dict[str, Any]:
    """Return raw 64-coordinate training effects for opaque single causes."""

    if not gene_ids or len(gene_ids) > 32:
        raise ValueError("gene_ids must contain 1-32 identifiers")
    if len(set(gene_ids)) != len(gene_ids):
        raise ValueError("gene_ids must be unique")
    unknown = sorted(set(gene_ids) - set(DATA["gene_ids"]))
    if unknown:
        raise ValueError(f"unknown gene ids: {unknown}")
    return {
        "feature_ids": DATA["feature_ids"],
        "single_effects": {
            gene_id: DATA["single_effects"][gene_id] for gene_id in gene_ids
        },
        "heldout_outcomes_used": False,
    }


def training_pairs(pair_ids: list[str]) -> dict[str, Any]:
    """Return raw observed training-double responses and frozen fold IDs."""

    if not pair_ids or len(pair_ids) > 16:
        raise ValueError("pair_ids must contain 1-16 identifiers")
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("pair_ids must be unique")
    unknown = sorted(set(pair_ids) - set(DATA["training_pair_ids"]))
    if unknown:
        raise ValueError(f"unknown pair ids: {unknown}")
    return {
        "feature_ids": DATA["feature_ids"],
        "training_pairs": {
            pair_id: DATA["training_pairs"][pair_id] for pair_id in pair_ids
        },
        "heldout_outcomes_used": False,
    }


_EMPTY_PARAMETERS = {"type": "object", "properties": {}, "additionalProperties": False}
_GENE_PARAMETERS = {
    "type": "object",
    "required": ["gene_ids"],
    "properties": {
        "gene_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 32,
            "uniqueItems": True,
            "items": {"type": "string", "pattern": "^G[0-9]{3}$"},
        }
    },
    "additionalProperties": False,
}
_PAIR_PARAMETERS = {
    "type": "object",
    "required": ["pair_ids"],
    "properties": {
        "pair_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 16,
            "uniqueItems": True,
            "items": {"type": "string", "pattern": "^D[0-9]{3}$"},
        }
    },
    "additionalProperties": False,
}
_LAB_PARAMETERS = {
    "type": "object",
    "required": ["source"],
    "properties": {
        "source": {
            "type": "string",
            "maxLength": 100000,
            "description": (
                "Python program. A read-only DATA dict containing all training "
                "singles/doubles, folds, target components, and no held-out "
                "outcomes is preloaded. Print JSON or concise diagnostics."
            ),
        }
    },
    "additionalProperties": False,
}


def all_tools() -> list[Tool]:
    local = {
        "namespace": "screen2",
        "access": ToolAccess.READ,
        "risk_tier": RiskTier.LOW,
        "side_effects": (),
    }
    tools = [
        Tool(
            id="screen2.training_manifest",
            description=(
                "[evidence primitive] Inspect opaque training IDs, fixed folds, "
                "target component relationships, class rule, and tensor shapes; "
                "returns no target outcomes or predictions."
            ),
            parameters_schema=_EMPTY_PARAMETERS,
            fn=training_manifest,
            **local,
        ),
        Tool(
            id="screen2.single_effects",
            description=(
                "[evidence primitive] Read selected raw training single-condition "
                "64-vectors by opaque G identifier; returns no target outcomes."
            ),
            parameters_schema=_GENE_PARAMETERS,
            fn=single_effects,
            **local,
        ),
        Tool(
            id="screen2.training_pairs",
            description=(
                "[evidence primitive] Read selected observed training-double "
                "64-vectors, component IDs, cell counts, and fixed folds by opaque "
                "D identifier; returns no target outcomes."
            ),
            parameters_schema=_PAIR_PARAMETERS,
            fn=training_pairs,
            **local,
        ),
    ]
    lab_descriptions = {
        "algebra_lab": (
            "[training compute lab] Execute agent-authored NumPy/scikit-learn code "
            "against the complete read-only training bundle to construct and "
            "cross-validate algebraic, residual, tensor, or symmetry models."
        ),
        "geometry_lab": (
            "[training compute lab] Execute agent-authored NumPy/SciPy/scikit-learn "
            "code against the complete read-only training bundle to construct and "
            "validate latent, metric, neighbor, or manifold models."
        ),
        "graph_lab": (
            "[training compute lab] Execute agent-authored NumPy/NetworkX/"
            "scikit-learn code against the complete read-only training bundle to "
            "construct and ablate signed relational models."
        ),
        "information_lab": (
            "[training compute lab] Execute agent-authored NumPy/SciPy/scikit-learn "
            "code against the complete read-only training bundle to compare models, "
            "estimate uncertainty, or build information-theoretic gates."
        ),
    }
    for suffix, description in lab_descriptions.items():
        tools.append(
            Tool(
                id=f"screen2.{suffix}",
                namespace="screen2",
                description=description,
                parameters_schema=_LAB_PARAMETERS,
                executor=ContainerExecutor(
                    ContainerToolConfig(
                        "reagents/norman-training:latest",
                        "norman_training_python",
                        timeout_s=180.0,
                    )
                ),
                provider=ToolProvider.CONTAINER,
                access=ToolAccess.COMPUTE,
                risk_tier=RiskTier.HIGH,
                latency_class="batch",
            )
        )
    return tools


__all__ = ["all_tools", "single_effects", "training_manifest", "training_pairs"]
