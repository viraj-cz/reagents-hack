"""Opaque primitives for the held-out experiment-portfolio benchmark."""

from __future__ import annotations

from typing import Any

from reagents.contracts import ToolAccess, ToolProvider
from reagents.tools._perturbseq_design_public import DATA
from reagents.tools.container import ContainerExecutor, ContainerToolConfig
from reagents.tools.registry import Tool


def manifest() -> dict[str, Any]:
    return {
        key: DATA[key]
        for key in (
            "benchmark_id",
            "feature_ids",
            "gene_ids",
            "training_pair_ids",
            "candidate_ids",
            "candidate_components",
            "fold_count",
            "batch_size",
            "gene_cap",
            "program_count",
            "coverage_bonus",
            "strong_threshold",
        )
    } | {"candidate_outcomes_present": False}


def single_effects(gene_ids: list[str]) -> dict[str, Any]:
    if not gene_ids or len(gene_ids) > 40 or len(gene_ids) != len(set(gene_ids)):
        raise ValueError("gene_ids must contain 1-40 unique identifiers")
    unknown = sorted(set(gene_ids) - set(DATA["gene_ids"]))
    if unknown:
        raise ValueError(f"unknown gene ids: {unknown}")
    return {
        "feature_ids": DATA["feature_ids"],
        "single_effects": {gene: DATA["single_effects"][gene] for gene in gene_ids},
        "candidate_outcomes_used": False,
    }


def training_pairs(pair_ids: list[str]) -> dict[str, Any]:
    if not pair_ids or len(pair_ids) > 20 or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("pair_ids must contain 1-20 unique identifiers")
    unknown = sorted(set(pair_ids) - set(DATA["training_pair_ids"]))
    if unknown:
        raise ValueError(f"unknown pair ids: {unknown}")
    return {
        "feature_ids": DATA["feature_ids"],
        "training_pairs": {pair: DATA["training_pairs"][pair] for pair in pair_ids},
        "candidate_outcomes_used": False,
    }


EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}
GENES = {
    "type": "object",
    "required": ["gene_ids"],
    "properties": {
        "gene_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 40,
            "uniqueItems": True,
            "items": {"type": "string", "pattern": "^G[0-9]{3}$"},
        }
    },
    "additionalProperties": False,
}
PAIRS = {
    "type": "object",
    "required": ["pair_ids"],
    "properties": {
        "pair_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 20,
            "uniqueItems": True,
            "items": {"type": "string", "pattern": "^D[0-9]{3}$"},
        }
    },
    "additionalProperties": False,
}
LAB = {
    "type": "object",
    "required": ["source"],
    "properties": {
        "source": {
            "type": "string",
            "maxLength": 120000,
            "description": (
                "Python program. Read-only DATA contains training observations, "
                "fixed folds, candidate components, and no candidate outcomes. "
                "Print concise JSON or diagnostics."
            ),
        }
    },
    "additionalProperties": False,
}


def all_tools() -> list[Tool]:
    common = {"namespace": "portfolio", "access": ToolAccess.READ, "side_effects": ()}
    tools = [
        Tool(
            id="portfolio.manifest",
            description=(
                "Inspect opaque candidates, components, fixed folds, portfolio "
                "constraints, and scoring constants; returns no candidate outcome "
                "or utility."
            ),
            parameters_schema=EMPTY,
            fn=manifest,
            **common,
        ),
        Tool(
            id="portfolio.single_effects",
            description=(
                "Read raw coordinate vectors for selected opaque single causes; "
                "returns no candidate outcome."
            ),
            parameters_schema=GENES,
            fn=single_effects,
            **common,
        ),
        Tool(
            id="portfolio.training_pairs",
            description=(
                "Read selected observed training-pair vectors, component IDs, "
                "counts, and folds; returns no candidate outcome."
            ),
            parameters_schema=PAIRS,
            fn=training_pairs,
            **common,
        ),
    ]
    descriptions = {
        "algebra_lab": (
            "Fit and cross-validate residual algebra or tensor models against "
            "frozen folds."
        ),
        "geometry_lab": (
            "Fit and validate latent, metric, neighbor, or diversity representations."
        ),
        "graph_lab": "Fit and ablate relational or message-passing representations.",
        "optimization_lab": (
            "Construct and verify constrained ten-item portfolios from "
            "agent-authored public-data predictions."
        ),
    }
    for suffix, description in descriptions.items():
        tools.append(
            Tool(
                id=f"portfolio.{suffix}",
                namespace="portfolio",
                description=f"[training-only compute] {description}",
                parameters_schema=LAB,
                executor=ContainerExecutor(
                    ContainerToolConfig(
                        "reagents/perturbseq-design:latest",
                        "perturbseq_design_python",
                        timeout_s=240.0,
                    )
                ),
                provider=ToolProvider.CONTAINER,
                access=ToolAccess.COMPUTE,
                latency_class="batch",
            )
        )
    return tools


__all__ = ["all_tools", "manifest", "single_effects", "training_pairs"]
