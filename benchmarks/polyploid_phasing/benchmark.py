"""Public verifier and private scorer for tetraploid long-read phasing."""

from __future__ import annotations

import json
from itertools import pairwise, permutations
from pathlib import Path
from typing import Any

from reagents.contracts import DemiGodResult, NativeProblem, NativeSolution
from reagents.tools import polyploid as latent_tools
from reagents.verification import VerificationReport

CASE_DIR = Path(__file__).parent
PUBLIC_DIR = CASE_DIR / "public"
PRIVATE_DIR = CASE_DIR / "private"
VARIANT_IDS = tuple(f"V{index:03d}" for index in range(1, 43))
SYMBOL_IDS = tuple(f"S{index:03d}" for index in range(1, 43))


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_problem() -> NativeProblem:
    raw = _load(PUBLIC_DIR / "question.json")
    raw["inputs"]["freeze_manifest"] = _load(PUBLIC_DIR / "freeze_manifest.json")
    return NativeProblem.model_validate(raw)


def _structured(raw: dict[str, Any] | NativeSolution) -> dict[str, Any]:
    if isinstance(raw, NativeSolution):
        structured = raw.structured_answer
        if isinstance(structured, dict) and "haplotypes" in structured:
            return structured
        try:
            parsed = json.loads(raw.answer)
        except (json.JSONDecodeError, TypeError):
            return structured
        return parsed if isinstance(parsed, dict) else structured
    if "solution" in raw and isinstance(raw["solution"], dict):
        return _structured(NativeSolution.model_validate(raw["solution"]))
    structured = raw.get("structured_answer")
    if isinstance(structured, dict) and "haplotypes" in structured:
        return structured
    answer = raw.get("answer")
    if isinstance(answer, str):
        try:
            parsed = json.loads(answer)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    return structured if isinstance(structured, dict) else raw


def normalize_solution(
    raw: dict[str, Any] | NativeSolution,
) -> tuple[list[dict[str, int]], list[list[str]], list[str]]:
    structured = _structured(raw)
    if not isinstance(structured, dict):
        return [], [], ["structured answer must be an object"]
    raw_haplotypes = structured.get("haplotypes")
    errors: list[str] = []
    if not isinstance(raw_haplotypes, list) or len(raw_haplotypes) != 4:
        return [], [], ["haplotypes must contain exactly four objects"]
    expected = set(VARIANT_IDS)
    haplotypes = []
    for row_index, raw_row in enumerate(raw_haplotypes):
        row = raw_row.get("alleles", raw_row) if isinstance(raw_row, dict) else raw_row
        if isinstance(row, list) and len(row) == len(VARIANT_IDS):
            row = dict(zip(VARIANT_IDS, row, strict=True))
        if not isinstance(row, dict) or set(row) != expected:
            errors.append(f"haplotype {row_index} must contain exactly V001..V042")
            continue
        parsed = {}
        for variant_id, value in row.items():
            if isinstance(value, bool):
                value = int(value)
            if value not in (0, 1):
                errors.append(f"{variant_id} in haplotype {row_index} must be 0 or 1")
            else:
                parsed[variant_id] = int(value)
        haplotypes.append(parsed)
    public = _load(PUBLIC_DIR / "observations.json")
    dosage = {
        item["variant_id"]: item["alternate_dosage"] for item in public["variants"]
    }
    if len(haplotypes) == 4:
        violations = [
            variant_id
            for variant_id in VARIANT_IDS
            if sum(row.get(variant_id, -10) for row in haplotypes) != dosage[variant_id]
        ]
        if violations:
            errors.append(f"allele dosage violated at {violations}")
    raw_blocks = structured.get("phase_blocks")
    blocks: list[list[str]] = []
    if not isinstance(raw_blocks, list) or not raw_blocks:
        errors.append("phase_blocks must be a non-empty array")
    else:
        flat = []
        for index, block in enumerate(raw_blocks):
            if not isinstance(block, dict) or not isinstance(
                block.get("variant_ids"), list
            ):
                errors.append(f"phase block {index} must contain variant_ids")
                continue
            ids = block["variant_ids"]
            if not ids or any(item not in expected for item in ids):
                errors.append(f"phase block {index} contains invalid IDs")
                continue
            blocks.append(list(ids))
            flat.extend(ids)
        if len(flat) != len(set(flat)) or set(flat) != expected:
            errors.append("phase_blocks must partition V001..V042 exactly once")
    uncertain = structured.get("uncertain_variants", [])
    if not isinstance(uncertain, list) or any(
        item not in expected for item in uncertain
    ):
        errors.append("uncertain_variants must contain only public variant IDs")
    summary = structured.get("method_summary")
    if not isinstance(summary, str) or not summary.strip():
        errors.append("method_summary must be non-empty")
    return haplotypes, blocks, errors


def _symbols_to_native(rows: Any) -> list[dict[str, int]] | None:
    if not isinstance(rows, list) or len(rows) != 4:
        return None
    rows = [row.get("alleles", row) if isinstance(row, dict) else row for row in rows]
    if all(isinstance(row, list) and len(row) == len(VARIANT_IDS) for row in rows):
        return [dict(zip(VARIANT_IDS, row, strict=True)) for row in rows]
    if all(isinstance(row, dict) and set(row) == set(SYMBOL_IDS) for row in rows):
        return [
            {f"V{int(symbol[1:]):03d}": value for symbol, value in row.items()}
            for row in rows
        ]
    if all(isinstance(row, dict) and set(row) == set(VARIANT_IDS) for row in rows):
        return [dict(row) for row in rows]
    return None


def _artifact_candidate(artifact: DemiGodResult) -> dict[str, Any] | None:
    candidate = artifact.payload.get("candidate_solution", artifact.payload)
    if not isinstance(candidate, dict):
        return None
    rows = None
    for key in ("factors", "haplotypes", "assignments", "words", "strands", "rows"):
        if key in candidate:
            rows = _symbols_to_native(candidate[key])
            if rows is not None:
                break
    if rows is None:
        return None
    raw_blocks = candidate.get("blocks", candidate.get("phase_blocks", []))
    blocks = []
    for block in raw_blocks if isinstance(raw_blocks, list) else []:
        ids = (
            block.get("symbols", block.get("variant_ids", []))
            if isinstance(block, dict)
            else block
        )
        if isinstance(ids, list) and ids:
            blocks.append(
                {
                    "variant_ids": [
                        f"V{int(item[1:]):03d}"
                        if isinstance(item, str) and item.startswith("S")
                        else item
                        for item in ids
                    ],
                    "confidence": block.get("confidence", 0.5)
                    if isinstance(block, dict)
                    else 0.5,
                }
            )
    if not blocks:
        blocks = [{"variant_ids": list(VARIANT_IDS), "confidence": 0.2}]
    uncertain = candidate.get(
        "uncertain_symbols", candidate.get("uncertain_variants", [])
    )
    return {
        "haplotypes": rows,
        "phase_blocks": blocks,
        "uncertain_variants": [
            f"V{int(item[1:]):03d}"
            if isinstance(item, str) and item.startswith("S")
            else item
            for item in uncertain
        ],
        "method_summary": candidate.get(
            "method_summary", f"Translated complete {artifact.domain_name} candidate."
        ),
        "selected_source": artifact.domain_name,
    }


def _public_cost(rows: list[dict[str, int]]) -> int:
    factors = [
        {f"S{int(key[1:]):03d}": value for key, value in row.items()} for row in rows
    ]
    return int(latent_tools.score(factors)["weighted_discordance"])


class PolyploidPhasingVerifier:
    """Public-only finalization and validation."""

    def finalize(
        self,
        problem: NativeProblem,
        solution: NativeSolution,
        artifacts: list[DemiGodResult],
    ) -> NativeSolution:
        candidates = []
        rows, _, errors = normalize_solution(solution)
        if not errors:
            candidates.append({**solution.structured_answer, "haplotypes": rows})
        candidates.extend(
            candidate
            for artifact in artifacts
            if (candidate := _artifact_candidate(artifact)) is not None
        )
        valid = []
        for candidate in candidates:
            parsed, _, candidate_errors = normalize_solution(candidate)
            if not candidate_errors:
                valid.append((candidate, parsed))
        if not valid:
            return solution
        selected, rows = min(valid, key=lambda item: _public_cost(item[1]))
        selected = {
            **selected,
            "haplotypes": [
                {"label": f"H{index + 1}", "alleles": row}
                for index, row in enumerate(rows)
            ],
            "public_objective": {
                "weighted_discordance": _public_cost(rows),
                "external_target_used": False,
            },
        }
        return solution.model_copy(update={"structured_answer": selected})

    def verify(
        self, problem: NativeProblem, solution: NativeSolution
    ) -> VerificationReport:
        rows, blocks, errors = normalize_solution(solution)
        return VerificationReport(
            passed=not errors,
            score=1.0 if not errors else 0.0,
            checks={
                "haplotype_count": len(rows),
                "variant_count": len(rows[0]) if rows else 0,
                "phase_block_count": len(blocks),
                "weighted_discordance": _public_cost(rows) if not errors else None,
                "heldout_target_consulted": False,
            },
            errors=errors,
        )


def _block_phase_accuracy(
    rows: list[dict[str, int]], expected: dict[str, Any]
) -> float:
    by_block: dict[str, list[str]] = {}
    for variant_id, block in expected["phase_set"].items():
        by_block.setdefault(str(block), []).append(variant_id)
    correct = total = 0
    for variant_ids in by_block.values():
        best_errors = None
        for permutation in permutations(range(4)):
            errors = sum(
                rows[row][variant_id]
                != expected["phased"][variant_id][permutation[row]]
                for row in range(4)
                for variant_id in variant_ids
            )
            best_errors = errors if best_errors is None else min(best_errors, errors)
        total += 4 * len(variant_ids)
        correct += 4 * len(variant_ids) - int(best_errors or 0)
    return correct / total if total else 0.0


def _boundary_accuracy(blocks: list[list[str]], expected: dict[str, Any]) -> float:
    predicted = {
        variant_id: index for index, block in enumerate(blocks) for variant_id in block
    }
    actual = expected["phase_set"]
    matches = 0
    for left, right in pairwise(VARIANT_IDS):
        predicted_joined = predicted[left] == predicted[right]
        actual_joined = (
            left in actual and right in actual and actual[left] == actual[right]
        )
        matches += predicted_joined == actual_joined
    return matches / (len(VARIANT_IDS) - 1)


def score_solution(
    raw: dict[str, Any] | NativeSolution, *, private: bool = True
) -> dict[str, Any]:
    rows, blocks, errors = normalize_solution(raw)
    if errors:
        return {"passed": False, "score_10": 0.0, "errors": errors}
    cost = _public_cost(rows)
    if not private:
        return {"passed": True, "score_10": None, "weighted_discordance": cost}
    expected = _load(PRIVATE_DIR / "expected.json")
    optimum = int(expected["exact_public_objective_optimum"]["weighted_discordance"])
    reference_ceiling = int(expected["canonical_reference"]["weighted_discordance"])
    objective_score = min(1.0, reference_ceiling / max(cost, 1))
    phase_accuracy = _block_phase_accuracy(rows, expected)
    boundary_accuracy = _boundary_accuracy(blocks, expected)
    score_10 = 10 * (
        0.4 * objective_score + 0.4 * phase_accuracy + 0.2 * boundary_accuracy
    )
    return {
        "passed": True,
        "score_10": round(score_10, 4),
        "weighted_discordance": cost,
        "exact_optimum": optimum,
        "reference_objective_ceiling": reference_ceiling,
        "objective_optimality": round(objective_score, 6),
        "permutation_invariant_block_phase_accuracy": round(phase_accuracy, 6),
        "phase_boundary_accuracy": round(boundary_accuracy, 6),
        "errors": [],
    }
