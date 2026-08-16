"""Run one direct, no-tool Claude call on the frozen Norman problem."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from benchmarks.perturbseq_norman.benchmark import load_problem, score_solution
from reagents.llm.anthropic_client import DEFAULT_MODEL, _extract_json_object

ROOT = Path(__file__).parents[2]


class BaselinePrediction(BaseModel):
    target_id: str
    predicted_delta: list[float] = Field(min_length=64, max_length=64)
    interaction_class: Literal["additive", "synergy", "suppression", "neomorphic"]
    confidence: float = Field(ge=0.0, le=1.0)
    falsifier: str = Field(min_length=1)


class BaselineAnswer(BaseModel):
    predictions: list[BaselinePrediction] = Field(min_length=12, max_length=12)
    method_summary: str = Field(min_length=1)


def _load_env() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.removeprefix("export ").strip(), value.strip())


def public_prompt() -> tuple[str, str]:
    problem = load_problem()
    public = problem.model_dump(mode="json")
    digest = hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
    prompt = (
        "Solve this frozen benchmark directly from the supplied public problem. "
        "No tools, files, code execution, retrieval, subagents, or additional "
        "messages are available. Broker interface descriptions are contextual "
        "metadata only: you cannot call them and must not claim their outputs. "
        "Make your best biologically informed estimates and complete every field.\n\n"
        f"Public problem:\n{json.dumps(public, indent=2)}"
    )
    return prompt, digest


async def run(model: str, output: Path) -> dict:
    import anthropic

    prompt, public_input_sha256 = public_prompt()
    schema = json.dumps(BaselineAnswer.model_json_schema())
    started = time.time()
    # Deliberately one API call with no `tools` parameter and no retry loop.
    message = await anthropic.AsyncAnthropic().messages.create(
        model=model,
        max_tokens=16000,
        system=(
            "You are the no-tool base-model control for a computational biology "
            "benchmark. Work independently and return JSON only matching this "
            f"schema:\n{schema}"
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    raw_response = "\n".join(
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text"
    )
    record = {
        "kind": "single_claude_no_tools",
        "model": model,
        "api_calls": 1,
        "tools_provided": [],
        "agent_loop": False,
        "public_input_sha256": public_input_sha256,
        "stop_reason": message.stop_reason,
        "elapsed_s": round(time.time() - started, 3),
        "usage": {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        },
        "raw_response": raw_response,
        "answer": None,
        "parse_error": None,
        "score": None,
        "response_committed_before_private_scoring": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")

    try:
        candidate = _extract_json_object(raw_response)
        if candidate is None:
            raise ValueError("response did not contain a complete JSON object")
        answer = BaselineAnswer.model_validate_json(candidate).model_dump(mode="json")
        record["answer"] = answer
        record["score"] = score_solution({"structured_answer": answer}, private=True)
    except Exception as exc:
        record["parse_error"] = str(exc)
        record["score"] = {
            "passed": False,
            "score_10": 0.0,
            "errors": [f"invalid base-model response: {exc}"],
        }
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is missing from the process and .env")
    record = asyncio.run(run(args.model, args.output))
    print(
        json.dumps(
            {
                key: value
                for key, value in record.items()
                if key not in {"raw_response", "answer"}
            },
            indent=2,
        )
    )
    return 0 if record["score"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
