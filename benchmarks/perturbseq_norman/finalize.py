"""Deterministically finalize an already completed pipeline record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.perturbseq_norman.benchmark import (
    NormanPerturbSeqVerifier,
    load_problem,
)
from reagents.contracts import DemiGodResult, NativeSolution


def finalize_record(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    verifier = NormanPerturbSeqVerifier()
    solution = NativeSolution.model_validate(record["solution"])
    artifacts = [DemiGodResult.model_validate(raw) for raw in record["artifacts"]]
    finalized = verifier.finalize(load_problem(), solution, artifacts)
    finalized.gaps = [
        gap for gap in finalized.gaps if not gap.startswith("native verification:")
    ]
    report = verifier.verify(load_problem(), finalized)
    finalized.verification = report.model_dump(mode="json")
    record["integration_before_finalization"] = record["solution"]
    record["solution"] = finalized.model_dump(mode="json")
    record["finalization"] = {
        "kind": "deterministic_public_schema_projection",
        "heldout_truth_consulted": False,
        "passed": report.passed,
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    record = finalize_record(args.record)
    print(json.dumps(record["finalization"], indent=2))
    return 0 if record["finalization"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
