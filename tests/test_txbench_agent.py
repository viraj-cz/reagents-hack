"""Offline tests for the TxBench God plug-in. No Modal, no API key."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from reagents.contracts import NativeSolution

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import format_solution, parse_answer_object, problem_from_task  # noqa: E402


def test_parse_answer_object_accepts_raw_json_and_fenced_json():
    assert parse_answer_object('{"responsive": 16, "removed": 0}') == {
        "responsive": 16,
        "removed": 0,
    }
    fenced = """```json
{"responsive": 16, "removed": 0}
```"""
    assert parse_answer_object(fenced) == {"responsive": 16, "removed": 0}


def test_parse_answer_object_pulls_json_out_of_prose():
    text = 'The count is {"responsive": 16, "removed": 0} after dropping the gate.'
    assert parse_answer_object(text) == {"responsive": 16, "removed": 0}


def test_problem_from_task_keeps_entities_empty_and_names_staged_files(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "hits.csv").write_text("id,value\n1,2\n", encoding="utf-8")
    problem = problem_from_task("How many hits? Return {\"n\": int}.", tmp_path)
    assert problem.entities == []
    assert any("shared/" in item for item in problem.constraints)
    assert "data/hits.csv" in problem.constraints[-1]


def test_format_solution_writes_eval_answer_json(tmp_path):
    solution = NativeSolution(
        problem_id="txbench",
        answer='{"responsive": 16, "removed": 0}',
        confidence=0.8,
    )
    payload = format_solution(solution, tmp_path)
    assert payload["answer"] == {"responsive": 16, "removed": 0}
    written = json.loads((tmp_path / "eval_answer.json").read_text(encoding="utf-8"))
    assert written == {"responsive": 16, "removed": 0}
