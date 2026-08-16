"""Plug your model in here. This is the only file you need to edit.

    run_agent(task, work_dir) -> dict | str

  task      The full question text. It ends with the exact JSON schema your
            answer must use. Nothing else about the eval is passed to you —
            the ground truth stays on the other side of this boundary.

  work_dir  An empty scratch directory outside the repo. The eval's input
            files are in `work_dir / "data"`. Write whatever you like here;
            it is deleted after the run unless you pass --keep-workspace.

Return either the answer dict, or your model's final text with an
<EVAL_ANSWER> block in it — the runner parses both. To attach run metadata
(cost, tokens, model id) return {"answer": ..., "metadata": {...}} instead.

Anything you raise is caught, recorded on the result, and scored as a failure,
so a crash on one eval will not abort a full sweep.

The default below shells out to the Claude Code CLI. Replace the body.
"""

import json
import subprocess
from pathlib import Path

HARNESS_NOTES = """

---
Notes from the harness:
- The data for this task is in the `data/` directory of your working directory.
- Derive the answer from that data. Do not answer from prior knowledge.
- When done, print the answer wrapped in <EVAL_ANSWER> tags exactly as
  specified above, and also write the same JSON to ./eval_answer.json
"""


def run_agent(task: str, work_dir: Path) -> dict | str:
    proc = subprocess.run(
        [
            "claude",
            "-p",
            task + HARNESS_NOTES,
            "--output-format",
            "json",
            "--allowedTools",
            "Bash,Read,Write,Edit,Glob,Grep",
            "--max-turns",
            "80",
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr[-2000:]}")

    payload = json.loads(proc.stdout)
    return {
        "answer": payload.get("result", ""),
        "metadata": {
            "cost_usd": payload.get("total_cost_usd"),
            "duration_ms": payload.get("duration_ms"),
            "num_turns": payload.get("num_turns"),
        },
    }
