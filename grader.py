"""Answer extraction and scoring.

`grade()` delegates to the official latch-eval-tools graders so scores stay
comparable with the published TxBench leaderboard.
"""

import json
import re
from dataclasses import dataclass, field

from latch_eval_tools import GRADER_REGISTRY

ANSWER_RE = re.compile(r"<EVAL_ANSWER>\s*(.*?)\s*</EVAL_ANSWER>", re.DOTALL)


@dataclass
class Grade:
    passed: bool
    score: float
    reasoning: str
    metrics: dict = field(default_factory=dict)


def extract_answer(output: str | dict | None) -> dict | None:
    """Normalize whatever the agent returned into an answer dict.

    Accepts a dict as-is, or pulls the last <EVAL_ANSWER> block out of text.
    Models often emit the block twice (once while reasoning, once for real),
    so the last one wins.
    """
    if output is None or isinstance(output, dict):
        return output

    for raw in reversed(ANSWER_RE.findall(output)):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    return None


def grade(answer: dict | None, grader_spec: dict) -> Grade:
    if answer is None:
        return Grade(False, 0.0, "agent produced no parseable <EVAL_ANSWER> block")

    grader_type = grader_spec.get("type")
    grader_cls = GRADER_REGISTRY.get(grader_type)
    if grader_cls is None:
        raise ValueError(f"unknown grader type {grader_type!r}")

    try:
        result = grader_cls().evaluate_answer(answer, grader_spec.get("config", {}))
    except Exception as exc:
        return Grade(False, 0.0, f"grader rejected the answer: {exc}")

    metrics = {
        k: v
        for k, v in (result.metrics or {}).items()
        if not isinstance(v, (list, dict))
    }
    return Grade(result.passed, result.score, result.reasoning, metrics)
