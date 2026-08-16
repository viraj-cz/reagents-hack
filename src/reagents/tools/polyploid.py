"""Sealed latent-factor primitives for the tetraploid benchmark."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations, pairwise, permutations, product
from typing import Any

from reagents.contracts import ToolAccess
from reagents.tools._polyploid_public import DATA
from reagents.tools.registry import Tool


def _symbol(native_id: str) -> str:
    return f"S{int(native_id[1:]):03d}"


def _word(native_id: str) -> str:
    return f"W{int(native_id[1:4]):03d}"


def _symbols() -> list[str]:
    return [_symbol(item["variant_id"]) for item in DATA["variants"]]


def _dosages() -> dict[str, int]:
    return {
        _symbol(item["variant_id"]): int(item["alternate_dosage"])
        for item in DATA["variants"]
    }


def _normalize(factors: list[dict[str, Any]]) -> list[dict[str, int]]:
    if not isinstance(factors, list) or len(factors) != 4:
        raise ValueError("factors must contain exactly four complete rows")
    expected = set(_symbols())
    normalized = []
    for index, row in enumerate(factors):
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError(f"factor {index} must contain every public symbol exactly")
        parsed = {}
        for symbol, value in row.items():
            if isinstance(value, bool):
                value = int(value)
            if value not in (0, 1):
                raise ValueError(f"{symbol} in factor {index} must be 0 or 1")
            parsed[symbol] = int(value)
        normalized.append(parsed)
    violations = [
        symbol
        for symbol, dosage in _dosages().items()
        if sum(row[symbol] for row in normalized) != dosage
    ]
    if violations:
        raise ValueError(f"column multiplicities violated for {violations}")
    return normalized


def words_compute() -> dict[str, Any]:
    """Return the complete sparse noisy-word matrix and column multiplicities."""

    return {
        "coordinate_system": "sparse_noisy_factor_words",
        "factor_count": 4,
        "symbol_ids": _symbols(),
        "column_multiplicity": _dosages(),
        "words": [
            {
                "word_id": _word(read["read_id"]),
                "observations": [
                    {
                        "symbol": _symbol(call["variant_id"]),
                        "bit": call["bit"],
                        "weight": call["weight"],
                    }
                    for call in read["calls"]
                ],
            }
            for read in DATA["reads"]
        ],
        "native_metadata_present": False,
        "complete_problem": True,
    }


def similarity_compute(min_overlap: int = 2) -> dict[str, Any]:
    """Project all words into a signed compatibility graph."""

    calls = [
        {
            _symbol(call["variant_id"]): (call["bit"], call["weight"])
            for call in read["calls"]
        }
        for read in DATA["reads"]
    ]
    edges = []
    for left, right in combinations(range(len(calls)), 2):
        shared = sorted(set(calls[left]) & set(calls[right]))
        if len(shared) < min_overlap:
            continue
        agree = sum(
            min(calls[left][s][1], calls[right][s][1])
            for s in shared
            if calls[left][s][0] == calls[right][s][0]
        )
        conflict = sum(
            min(calls[left][s][1], calls[right][s][1])
            for s in shared
            if calls[left][s][0] != calls[right][s][0]
        )
        edges.append(
            {
                "left": f"W{left + 1:03d}",
                "right": f"W{right + 1:03d}",
                "overlap": len(shared),
                "agreement": agree,
                "conflict": conflict,
                "signed_margin": agree - conflict,
            }
        )
    opaque_words = words_compute()["words"]
    by_word = {item["word_id"]: item["observations"] for item in opaque_words}
    return {
        "coordinate_system": "signed_word_compatibility",
        "symbol_ids": _symbols(),
        "column_multiplicity": _dosages(),
        "nodes": [
            {
                "id": f"W{i + 1:03d}",
                "observations": by_word[f"W{i + 1:03d}"],
            }
            for i in range(len(calls))
        ],
        "edges": edges,
        "target_clusters": 4,
        "complete_problem": True,
    }


def tensor_compute() -> dict[str, Any]:
    """Project all observations into weighted pair-state tensors."""

    cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    support: dict[tuple[str, str], int] = defaultdict(int)
    for read in DATA["reads"]:
        for left, right in combinations(read["calls"], 2):
            a, b = _symbol(left["variant_id"]), _symbol(right["variant_id"])
            if a > b:
                a, b, left, right = b, a, right, left
            state = 2 * int(left["bit"]) + int(right["bit"])
            cells[(a, b)][state] += min(left["weight"], right["weight"])
            support[(a, b)] += 1
    return {
        "coordinate_system": "pair_state_tensor",
        "factor_count": 4,
        "column_multiplicity": _dosages(),
        "cells": [
            {
                "left": left,
                "right": right,
                "state_weights_00_01_10_11": values,
                "support": support[(left, right)],
            }
            for (left, right), values in sorted(cells.items())
        ],
        "complete_problem": True,
    }


def layered_compute() -> dict[str, Any]:
    """Project the system into dosage-constrained layer states and bridges."""

    dosages = _dosages()
    layers = []
    for symbol in _symbols():
        states = [
            list(bits)
            for bits in product((0, 1), repeat=4)
            if sum(bits) == dosages[symbol]
        ]
        layers.append({"symbol": symbol, "allowed_states": states})
    bridges = []
    for left, right in pairwise(_symbols()):
        support = 0
        weight = 0
        for read in DATA["reads"]:
            by_symbol = {_symbol(call["variant_id"]): call for call in read["calls"]}
            if left in by_symbol and right in by_symbol:
                support += 1
                weight += min(by_symbol[left]["weight"], by_symbol[right]["weight"])
        bridges.append(
            {"left": left, "right": right, "support": support, "weight": weight}
        )
    return {
        "coordinate_system": "layered_state_flow",
        "layers": layers,
        "bridges": bridges,
        "path_constraints": words_compute()["words"],
        "row_permutation_symmetry": True,
        "complete_problem": True,
    }


def score(factors: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate four complete factors under nearest-word weighted discordance."""

    rows = _normalize(factors)
    word_costs = []
    for read in DATA["reads"]:
        costs = [
            sum(
                call["weight"]
                for call in read["calls"]
                if row[_symbol(call["variant_id"])] != call["bit"]
            )
            for row in rows
        ]
        word_costs.append(
            {
                "word_id": _word(read["read_id"]),
                "costs": costs,
                "minimum": min(costs),
                "nearest_factor": min(range(4), key=costs.__getitem__),
            }
        )
    return {
        "objective": "minimum_weighted_latent_factor_discordance",
        "weighted_discordance": sum(item["minimum"] for item in word_costs),
        "word_costs": word_costs,
        "multiplicities_satisfied": True,
        "external_target_used": False,
    }


def swap_delta(
    factors: list[dict[str, Any]], row_a: int, row_b: int, symbols: list[str]
) -> dict[str, Any]:
    """Evaluate swapping selected columns between two factor rows."""

    rows = _normalize(factors)
    if row_a == row_b or not 0 <= row_a < 4 or not 0 <= row_b < 4:
        raise ValueError("row_a and row_b must be distinct indices in 0..3")
    unknown = sorted(set(symbols) - set(_symbols()))
    if not symbols or unknown:
        raise ValueError(f"symbols must be non-empty and public; unknown={unknown}")
    before = score(rows)["weighted_discordance"]
    changed = [dict(row) for row in rows]
    for symbol in symbols:
        changed[row_a][symbol], changed[row_b][symbol] = (
            changed[row_b][symbol],
            changed[row_a][symbol],
        )
    after = score(changed)["weighted_discordance"]
    return {"before": before, "after": after, "delta": after - before}


def align(
    factors_a: list[dict[str, Any]], factors_b: list[dict[str, Any]]
) -> dict[str, Any]:
    """Find the row permutation minimizing disagreement between two candidates."""

    left, right = _normalize(factors_a), _normalize(factors_b)
    best = None
    for permutation in permutations(range(4)):
        disagreements = sum(
            left[row][symbol] != right[permutation[row]][symbol]
            for row in range(4)
            for symbol in _symbols()
        )
        candidate = (disagreements, permutation)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return {"disagreements": best[0], "permutation": list(best[1])}


_NO_ARGS = {"type": "object", "properties": {}, "additionalProperties": False}
_FACTOR_ARGS = {
    "type": "object",
    "required": ["factors"],
    "properties": {"factors": {"type": "array", "minItems": 4, "maxItems": 4}},
    "additionalProperties": False,
}


def all_tools() -> list[Tool]:
    common = {
        "namespace": "latent",
        "access": ToolAccess.COMPUTE,
        "side_effects": (),
    }
    return [
        Tool(
            id="latent.words_compute",
            description=(
                "Project the complete system into sparse weighted words and "
                "column multiplicities."
            ),
            parameters_schema=_NO_ARGS,
            fn=words_compute,
            **common,
        ),
        Tool(
            id="latent.similarity_compute",
            description=(
                "Project all words into a signed four-cluster compatibility graph."
            ),
            parameters_schema={
                "type": "object",
                "properties": {"min_overlap": {"type": "integer", "minimum": 1}},
                "additionalProperties": False,
            },
            fn=similarity_compute,
            **common,
        ),
        Tool(
            id="latent.tensor_compute",
            description="Project all evidence into weighted pair-state tensors.",
            parameters_schema=_NO_ARGS,
            fn=tensor_compute,
            **common,
        ),
        Tool(
            id="latent.layered_compute",
            description=(
                "Project the complete system into constrained layer states and bridges."
            ),
            parameters_schema=_NO_ARGS,
            fn=layered_compute,
            **common,
        ),
        Tool(
            id="latent.score",
            description=(
                "Score four complete factor rows under the public discordance "
                "objective."
            ),
            parameters_schema=_FACTOR_ARGS,
            fn=score,
            **common,
        ),
        Tool(
            id="latent.swap_delta",
            description="Evaluate a dosage-preserving row swap over selected columns.",
            parameters_schema={
                "type": "object",
                "required": ["factors", "row_a", "row_b", "symbols"],
                "properties": {
                    "factors": {"type": "array", "minItems": 4, "maxItems": 4},
                    "row_a": {"type": "integer", "minimum": 0, "maximum": 3},
                    "row_b": {"type": "integer", "minimum": 0, "maximum": 3},
                    "symbols": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
            fn=swap_delta,
            **common,
        ),
        Tool(
            id="latent.align",
            description=(
                "Align two complete candidates under four-row permutation symmetry."
            ),
            parameters_schema={
                "type": "object",
                "required": ["factors_a", "factors_b"],
                "properties": {
                    "factors_a": {"type": "array", "minItems": 4, "maxItems": 4},
                    "factors_b": {"type": "array", "minItems": 4, "maxItems": 4},
                },
                "additionalProperties": False,
            },
            fn=align,
            **common,
        ),
    ]
