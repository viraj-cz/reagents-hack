"""Representation-neutral binary primitives over the frozen public evidence.

The Broker may import biological source data, but no function below returns
native labels, provenance, coordinates, bases, or domain terminology. Workers
see only abstract symbols, noisy binary words, constraints, and objective values.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

from reagents.contracts import ToolAccess
from reagents.tools._haplotype_public import DATA
from reagents.tools.registry import Tool


def _symbol(variant_id: str) -> str:
    return f"S{int(variant_id[1:]):03d}"


def _word(read_id: str) -> str:
    return f"W{int(read_id[1:4]):03d}"


def _symbol_ids() -> list[str]:
    return [_symbol(item["variant_id"]) for item in DATA["variants"]]


def _normalize_assignment(assignment: dict[str, Any]) -> dict[str, int]:
    expected = set(_symbol_ids())
    if set(assignment) != expected:
        raise ValueError(
            f"assignment must contain exactly {len(expected)} symbols; "
            f"missing={sorted(expected - set(assignment))}, "
            f"extra={sorted(set(assignment) - expected)}"
        )
    normalized: dict[str, int] = {}
    for symbol, value in assignment.items():
        if isinstance(value, bool):
            value = int(value)
        if value not in (0, 1):
            raise ValueError(f"{symbol} must be 0 or 1")
        normalized[symbol] = int(value)
    return normalized


def observe(word_ids: list[str] | None = None) -> dict[str, Any]:
    """Return selected noisy binary words in opaque notation."""

    available = {_word(item["read_id"]): item for item in DATA["reads"]}
    selected = word_ids or sorted(available)
    if len(selected) > len(available) or len(set(selected)) != len(selected):
        raise ValueError("word_ids must be unique public word identifiers")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"unknown word identifiers: {unknown}")
    return {
        "coordinate_system": "noisy_binary_words",
        "symbol_ids": _symbol_ids(),
        "words": [
            {
                "word_id": word_id,
                "observations": [
                    {
                        "symbol": _symbol(call["variant_id"]),
                        "bit": call["allele"],
                        "weight": call["quality"],
                    }
                    for call in available[word_id]["calls"]
                ],
            }
            for word_id in selected
        ],
        "native_metadata_present": False,
    }


def _pair_totals() -> dict[tuple[str, str], dict[str, int]]:
    totals: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"same": 0, "opposite": 0, "support": 0}
    )
    for record in DATA["reads"]:
        calls = record["calls"]
        for left, right in combinations(calls, 2):
            a, b = _symbol(left["variant_id"]), _symbol(right["variant_id"])
            key = tuple(sorted((a, b)))
            weight = min(int(left["quality"]), int(right["quality"]))
            relation = "same" if left["allele"] == right["allele"] else "opposite"
            totals[key][relation] += weight
            totals[key]["support"] += 1
    return dict(totals)


def graph_compute(min_support: int = 2) -> dict[str, Any]:
    """Project all evidence into a signed weighted constraint graph."""

    edges = []
    for (left, right), values in sorted(_pair_totals().items()):
        if values["support"] < min_support:
            continue
        same, opposite = values["same"], values["opposite"]
        edges.append(
            {
                "left": left,
                "right": right,
                "preferred_xor": int(opposite > same),
                "agreement_weight": max(same, opposite),
                "conflict_weight": min(same, opposite),
                "margin": abs(same - opposite),
                "support": values["support"],
            }
        )
    return {
        "coordinate_system": "signed_constraint_graph",
        "nodes": _symbol_ids(),
        "edges": edges,
        "complete_problem": True,
    }


def energy_compute(min_support: int = 2) -> dict[str, Any]:
    """Project all evidence into a zero-field Ising energy system."""

    couplings = []
    for (left, right), values in sorted(_pair_totals().items()):
        if values["support"] < min_support:
            continue
        # E = -J*s_i*s_j: positive J favors equal binary states.
        couplings.append(
            {
                "i": left,
                "j": right,
                "J": values["same"] - values["opposite"],
                "support": values["support"],
            }
        )
    return {
        "coordinate_system": "ising_energy",
        "spin_map": "spin(Si)=1-2*bit(Si)",
        "energy": "E=-sum(J_ij*spin_i*spin_j)",
        "zero_field": True,
        "couplings": couplings,
        "complete_problem": True,
    }


def code_compute() -> dict[str, Any]:
    """Project every noisy word into sparse weighted parity-check rows."""

    rows = []
    for record in DATA["reads"]:
        calls = record["calls"]
        anchor = calls[0]
        for call in calls[1:]:
            rows.append(
                {
                    "row_id": f"{_word(record['read_id'])}:{len(rows) + 1:04d}",
                    "symbols": [
                        _symbol(anchor["variant_id"]),
                        _symbol(call["variant_id"]),
                    ],
                    "syndrome": anchor["allele"] ^ call["allele"],
                    "weight": min(anchor["quality"], call["quality"]),
                }
            )
    return {
        "coordinate_system": "weighted_error_correcting_code",
        "symbol_ids": _symbol_ids(),
        "parity_rows": rows,
        "complete_problem": True,
    }


def logic_compute(min_support: int = 2) -> dict[str, Any]:
    """Project all evidence into weighted XOR clauses with contradictions."""

    clauses = []
    for (left, right), values in sorted(_pair_totals().items()):
        if values["support"] < min_support:
            continue
        clauses.append(
            {
                "clause": f"{left} XOR {right}",
                "preferred_rhs": int(values["opposite"] > values["same"]),
                "preferred_weight": max(values["same"], values["opposite"]),
                "contradiction_weight": min(values["same"], values["opposite"]),
                "support": values["support"],
            }
        )
    return {
        "coordinate_system": "weighted_xor_logic",
        "symbol_ids": _symbol_ids(),
        "clauses": clauses,
        "global_complement_symmetry": True,
        "complete_problem": True,
    }


def score(assignment: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a complete bit assignment against noisy words, up to complement."""

    bits = _normalize_assignment(assignment)
    word_costs = []
    for record in DATA["reads"]:
        direct = sum(
            call["quality"]
            for call in record["calls"]
            if bits[_symbol(call["variant_id"])] != call["allele"]
        )
        mass = sum(call["quality"] for call in record["calls"])
        word_costs.append(
            {
                "word_id": _word(record["read_id"]),
                "discordance": min(direct, mass - direct),
                "orientation": int(direct > mass - direct),
                "weight": mass,
            }
        )
    return {
        "objective": "minimum_weighted_word_discordance",
        "weighted_discordance": sum(item["discordance"] for item in word_costs),
        "total_weight": sum(item["weight"] for item in word_costs),
        "word_costs": word_costs,
        "complete": True,
        "external_target_used": False,
    }


def flip_delta(assignment: dict[str, Any], symbols: list[str]) -> dict[str, Any]:
    """Return the exact objective change from flipping a proposed symbol set."""

    bits = _normalize_assignment(assignment)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be a non-empty unique list")
    unknown = sorted(set(symbols) - set(bits))
    if unknown:
        raise ValueError(f"unknown symbols: {unknown}")
    before = score(bits)["weighted_discordance"]
    changed = dict(bits)
    for symbol in symbols:
        changed[symbol] = 1 - changed[symbol]
    after = score(changed)["weighted_discordance"]
    return {
        "flipped": symbols,
        "before": before,
        "after": after,
        "delta": after - before,
        "improves": after < before,
    }


def components() -> dict[str, Any]:
    """Return connected components and symbol coverage of the constraint system."""

    adjacency = {symbol: set() for symbol in _symbol_ids()}
    for left, right in _pair_totals():
        adjacency[left].add(right)
        adjacency[right].add(left)
    pending = set(adjacency)
    groups = []
    while pending:
        root = min(pending)
        stack = [root]
        group = set()
        while stack:
            node = stack.pop()
            if node in group:
                continue
            group.add(node)
            stack.extend(adjacency[node] - group)
        pending -= group
        groups.append(sorted(group))
    return {
        "coordinate_system": "constraint_connectivity",
        "components": groups,
        "component_count": len(groups),
        "symbol_count": len(adjacency),
    }


_NO_ARGS = {"type": "object", "properties": {}, "additionalProperties": False}
_SUPPORT_ARGS = {
    "type": "object",
    "properties": {"min_support": {"type": "integer", "minimum": 1, "default": 2}},
    "additionalProperties": False,
}
_ASSIGNMENT = {
    "type": "object",
    "required": ["assignment"],
    "properties": {"assignment": {"type": "object"}},
    "additionalProperties": False,
}


def all_tools() -> list[Tool]:
    common = {
        "namespace": "binary",
        "access": ToolAccess.COMPUTE,
        "side_effects": (),
    }
    return [
        Tool(
            id="binary.graph_compute",
            description=(
                "Project the complete noisy binary system into signed weighted edges."
            ),
            parameters_schema=_SUPPORT_ARGS,
            fn=graph_compute,
            **common,
        ),
        Tool(
            id="binary.energy_compute",
            description=(
                "Project the complete noisy binary system into Ising couplings."
            ),
            parameters_schema=_SUPPORT_ARGS,
            fn=energy_compute,
            **common,
        ),
        Tool(
            id="binary.code_compute",
            description=(
                "Project the complete noisy binary system into weighted parity checks."
            ),
            parameters_schema=_NO_ARGS,
            fn=code_compute,
            **common,
        ),
        Tool(
            id="binary.logic_compute",
            description=(
                "Project the complete noisy binary system into weighted XOR clauses."
            ),
            parameters_schema=_SUPPORT_ARGS,
            fn=logic_compute,
            **common,
        ),
        Tool(
            id="binary.observe",
            description=(
                "Inspect selected opaque noisy binary words without native metadata."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "word_ids": {"type": "array", "items": {"type": "string"}}
                },
                "additionalProperties": False,
            },
            fn=observe,
            **common,
        ),
        Tool(
            id="binary.score",
            description=(
                "Evaluate a complete bit assignment under the public word objective."
            ),
            parameters_schema=_ASSIGNMENT,
            fn=score,
            **common,
        ),
        Tool(
            id="binary.flip_delta",
            description="Evaluate the objective delta from flipping selected symbols.",
            parameters_schema={
                "type": "object",
                "required": ["assignment", "symbols"],
                "properties": {
                    "assignment": {"type": "object"},
                    "symbols": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string"},
                    },
                },
                "additionalProperties": False,
            },
            fn=flip_delta,
            **common,
        ),
        Tool(
            id="binary.components",
            description=(
                "Inspect connectivity and coverage of the abstract constraint system."
            ),
            parameters_schema=_NO_ARGS,
            fn=components,
            **common,
        ),
    ]
