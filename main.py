"""Run TxBench-PP evals against the agent in agent.py.

python main.py --list
python main.py CTRL01_no_cc1_gate_for_crizotinib_hits
python main.py --all --out results
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from agent import run_agent
from runner import list_evals, run_one

MARK = {True: "PASS", False: "FAIL"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval", nargs="?", help="eval id, or path to an eval.json")
    parser.add_argument("--all", action="store_true", help="run every public eval")
    parser.add_argument("--list", action="store_true", help="list available eval ids")
    parser.add_argument("--out", type=Path, help="directory to write summary.json into")
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="leave the scratch directory on disk for debugging",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.list:
        print("\n".join(list_evals()))
        return 0

    if args.all:
        eval_ids = list_evals()
    elif args.eval:
        eval_ids = [args.eval]
    else:
        build_parser().print_help()
        return 2

    results = []
    for eval_id in eval_ids:
        print(f"\n=== {eval_id}")
        result = run_one(eval_id, run_agent, keep_workspace=args.keep_workspace)
        print(f"  {MARK[result.grade.passed]}  {result.grade.reasoning.strip()[:200]}")
        results.append(result.as_dict())

    passed = sum(1 for r in results if r["passed"])

    print("\n" + "=" * 70)
    for result in results:
        print(f"{MARK[result['passed']]}  {result['eval_id']}")
    print(f"\n{passed}/{len(results)} passed")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        summary = {
            "benchmark": "txbench-pp",
            "generated_at": datetime.now(UTC).isoformat(),
            "n_evals": len(results),
            "n_passed": passed,
            "pass_rate": round(100 * passed / len(results), 2) if results else None,
            "results": results,
        }
        target = args.out / "summary.json"
        target.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
