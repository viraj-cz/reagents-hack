"""Score a saved FlareGuard run against the evaluator-only trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.flareguard.benchmark import score_solution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_record", type=Path)
    args = parser.parse_args()
    record = json.loads(args.run_record.read_text(encoding="utf-8"))
    score = score_solution(record, private=True)
    print(json.dumps(score, indent=2))
    return 0 if score["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
