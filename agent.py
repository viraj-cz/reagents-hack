"""TxBench plug-in: God plans, sealed demigods run, God integrates.

    run_agent(task, work_dir) -> dict | str

This is the only file the harness edits. `runner.py` / `grader.py` stay
untouched. The body used to shell out to `claude`; it now drives
`God.solve` with one Modal sandbox per demigod and the eval's input files
seeded into the run's read-only `shared/` volume.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from pathlib import Path

from reagents.contracts import NativeProblem, NativeSolution
from reagents.demigod.sandbox_runtime import SandboxDemigodRuntime, seed_shared_files
from reagents.god.orchestrator import God
from reagents.llm.client import make_llm
from reagents.tracing import TerminalTracer

REPO_ROOT = Path(__file__).resolve().parent
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# Interpreters the planner is allowed to name. Without this list a live plan
# that picks `reasoning.python` is dropped before spawn. The sandbox still
# cannot call them unless a toolbox lease is published; they are approved so
# the demigod is allowed to exist and fall back to pandas + shared/.
APPROVED_HIGH_RISK_TOOLS = {
    "reasoning.python",
    "biology.python",
    "engineering.python",
}


def load_env_file() -> None:
    """Load a gitignored `.env` if present. Existing env vars win."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
        return
    load_dotenv(env_path, override=False)


def data_files(work_dir: Path) -> list[Path]:
    data = work_dir / "data"
    if not data.is_dir():
        return []
    return sorted(path for path in data.rglob("*") if path.is_file())


def problem_from_task(task: str, work_dir: Path) -> NativeProblem:
    """Build the native problem God sees. Entities stay empty on purpose.

    Isolation treats `entities` as leak terms. Filling them from a TxBench
    prompt would drop domains that mention assay names. The staged files are
    the native field; they are seeded into `shared/` unsigned, same as the
    sandbox runtime's documented exception for GOD-supplied data.
    """
    names = [path.relative_to(work_dir).as_posix() for path in data_files(work_dir)]
    staged = ", ".join(names) if names else "none"
    return NativeProblem(
        id="txbench",
        statement=task,
        question=task,
        constraints=[
            "The final `answer` must be the exact JSON object the question's "
            "schema requires: no extra keys, no markdown, no surrounding prose.",
            "Derive values only from the staged input files, "
            "never from prior knowledge.",
            f"Input files are mounted read-only under shared/: {staged}",
        ],
    )


def parse_answer_object(text: str) -> dict | None:
    """Pull a JSON object out of God's answer string, if one is there."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        loaded = json.loads(stripped)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass
    match = JSON_OBJECT_RE.search(stripped)
    if match is None:
        return None
    try:
        loaded = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def format_solution(solution: NativeSolution, work_dir: Path) -> dict:
    """Harness-shaped return: parsed object when possible, else EVAL_ANSWER text."""
    parsed = parse_answer_object(solution.answer)
    if parsed is not None:
        payload = parsed
        rendered = json.dumps(parsed, indent=2)
    else:
        rendered = solution.answer
        payload = f"<EVAL_ANSWER>\n{rendered}\n</EVAL_ANSWER>"
    (work_dir / "eval_answer.json").write_text(
        rendered if parsed is not None else json.dumps({"raw": solution.answer}),
        encoding="utf-8",
    )
    return {
        "answer": payload,
        "metadata": {
            "confidence": solution.confidence,
            "conflicts": solution.conflicts,
            "gaps": solution.gaps,
            "domain_contributions": solution.domain_contributions,
        },
    }


def run_agent(task: str, work_dir: Path) -> dict | str:
    load_env_file()
    return asyncio.run(_solve(task, Path(work_dir)))


async def _solve(task: str, work_dir: Path) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is required for God's loop. Put it in a "
            "gitignored .env at the repo root, or export it in the shell. "
            "Demigods read a separate key from the Modal secret demigod-anthropic."
        )

    # Sponsor MCP/container catalogs are native-field biology tools. They must
    # not be the default pack for a TxBench demigod — orthogonal domains are
    # invented languages, not "biology as a graph".
    os.environ.pop("REAGENTS_ENABLE_MCP", None)
    os.environ.pop("REAGENTS_ENABLE_CONTAINERS", None)

    domains = int(os.environ.get("REAGENTS_TXBENCH_DOMAINS", "2"))
    turns = int(os.environ.get("REAGENTS_TXBENCH_TURNS", "10"))
    run_id = os.environ.get("REAGENTS_TXBENCH_RUN_ID") or f"tx{uuid.uuid4().hex[:8]}"

    problem = problem_from_task(task, work_dir)
    paths = data_files(work_dir)
    shared: list[str] = []
    if paths:
        shared = await asyncio.to_thread(
            seed_shared_files, run_id, [str(path) for path in paths]
        )

    god = God(
        make_llm(),
        domain_count=domains,
        runtime=SandboxDemigodRuntime(
            run_id=run_id,
            max_turns=turns,
            shared_files=shared,
        ),
        tracer=TerminalTracer(),
        approved_high_risk_tools=APPROVED_HIGH_RISK_TOOLS,
    )
    solution = await god.solve(problem)
    return format_solution(solution, work_dir)
