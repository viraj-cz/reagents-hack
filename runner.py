"""Loads an eval, runs the agent against it, and grades the result.

Answer isolation
----------------
`eval.json` holds both the question and the answer: `grader.config.ground_truth`
plus a `notes` field spelling out the full derivation. `load_eval` splits those
apart, and only `Task` is ever handed to the agent — the grader spec stays in
this process and never crosses the boundary.

The agent also runs in a temp directory outside this repo, so walking up from
its working directory reaches /tmp rather than `txbench-pp/evals/`.

This is isolation by construction, not a security sandbox: an agent that opens
`txbench-pp/evals/<id>/eval.json` by absolute path will still find the answer.
To make cheating impossible rather than merely deliberate, run the agent in a
container — the official harness uses public.ecr.aws/p5z7v3z8/benchmark_agent.
"""

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from latch_eval_tools import download_data

from grader import Grade, extract_answer, grade

REPO_ROOT = Path(__file__).resolve().parent
EVALS_DIR = REPO_ROOT / "txbench-pp" / "evals"
CACHE_NAME = ".eval_cache"


@dataclass
class Task:
    """The public half of an eval. Safe to hand to an agent."""

    id: str
    prompt: str
    data_node: list[str] = field(default_factory=list)


@dataclass
class Result:
    eval_id: str
    answer: dict | None
    grade: Grade
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "eval_id": self.eval_id,
            "passed": self.grade.passed,
            "score": self.grade.score,
            "answer": self.answer,
            "reasoning": self.grade.reasoning,
            "metrics": self.grade.metrics,
            "metadata": self.metadata,
        }


def list_evals() -> list[str]:
    return sorted(p.parent.name for p in EVALS_DIR.glob("*/eval.json"))


def eval_path(eval_id: str) -> Path:
    direct = Path(eval_id)
    if direct.is_file():
        return direct
    by_id = EVALS_DIR / eval_id / "eval.json"
    if by_id.is_file():
        return by_id
    raise SystemExit(f"no eval matching {eval_id!r} — try --list")


def load_eval(path: Path) -> tuple[Task, dict]:
    """Split an eval into (public task, private grader spec)."""
    raw = json.loads(path.read_text())

    data_node = raw.get("data_node") or []
    if isinstance(data_node, str):
        data_node = [data_node]

    task = Task(id=raw["id"], prompt=raw["task"], data_node=data_node)

    grader_spec = raw.get("grader")
    if grader_spec is None:
        graders = raw.get("graders") or []
        grader_spec = graders[0] if graders else None
    if grader_spec is None:
        raise SystemExit(f"{path} has no grader spec")

    return task, grader_spec


def stage_workspace(task: Task) -> Path:
    """Create an isolated working directory with the eval's data in ./data."""
    root = Path(tempfile.mkdtemp(prefix=f"txbench-{task.id}-"))
    if REPO_ROOT in root.parents:
        raise RuntimeError(f"workspace {root} is inside the repo; refusing to run")

    download_data(task.data_node, root, CACHE_NAME)
    return root / "agent_workspace"


def run_one(eval_id: str, agent_fn, keep_workspace: bool = False) -> Result:
    task, grader_spec = load_eval(eval_path(eval_id))
    work_dir = stage_workspace(task)

    metadata: dict = {}
    output = None
    try:
        output = agent_fn(task.prompt, work_dir)
        if isinstance(output, dict) and "answer" in output:
            metadata = output.get("metadata", {})
            output = output["answer"]
    except Exception as exc:
        metadata = {"agent_error": f"{type(exc).__name__}: {exc}"}

    answer = extract_answer(output)
    if answer is None:
        fallback = work_dir / "eval_answer.json"
        if fallback.exists():
            answer = extract_answer(fallback.read_text())

    result = Result(task.id, answer, grade(answer, grader_spec), metadata)

    if keep_workspace:
        print(f"  workspace kept at {work_dir}")
    else:
        shutil.rmtree(work_dir.parent, ignore_errors=True)

    return result
