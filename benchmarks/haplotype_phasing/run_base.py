"""Run one standard Claude call on the native biological problem, with no tools."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

from pydantic import BaseModel, Field

from benchmarks.haplotype_phasing.benchmark import (
    PUBLIC_DIR,
    VARIANT_IDS,
    load_problem,
    score_solution,
)
from reagents.llm.anthropic_client import DEFAULT_MODEL, _extract_json_object

ROOT = Path(__file__).parents[2]


class BaselineAnswer(BaseModel):
    phase_assignment: dict[str, int]
    uncertain_variants: list[str] = Field(default_factory=list)
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
    problem = load_problem().model_dump(mode="json")
    observations = json.loads((PUBLIC_DIR / "observations.json").read_text())
    public = {"problem": problem, "observations": observations}
    digest = hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
    prompt = (
        "Solve this frozen real-data problem directly in its original biological "
        "representation. No tools, code execution, files, retrieval, subagents, "
        "or follow-up messages are available. Infer a complete answer from the "
        "supplied evidence and return JSON only. Do not omit any variant.\n\n"
        f"Public input:\n{json.dumps(public, indent=2)}"
    )
    return prompt, digest


async def run(model: str, output: Path) -> dict:
    import anthropic

    prompt, digest = public_prompt()
    schema = json.dumps(BaselineAnswer.model_json_schema())
    started = time.time()
    message = await anthropic.AsyncAnthropic().messages.create(
        model=model,
        max_tokens=12000,
        system=(
            "You are the single-call no-tool control for a computational biology "
            "benchmark. Work independently. Return JSON only matching this schema; "
            f"phase_assignment must contain exactly {list(VARIANT_IDS)}:\n{schema}"
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    raw_response = "\n".join(
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text"
    )
    record = {
        "kind": "single_claude_native_no_tools",
        "model": model,
        "api_calls": 1,
        "tools_provided": [],
        "agent_loop": False,
        "public_input_sha256": digest,
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
        output.write_text(json.dumps(record, indent=2), encoding="utf-8")
        record["score"] = score_solution({"structured_answer": answer}, private=True)
    except Exception as exc:
        record["parse_error"] = str(exc)
        record["score"] = {
            "passed": False,
            "score_10": 0.0,
            "errors": [f"invalid standard-Claude response: {exc}"],
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
            {k: v for k, v in record.items() if k not in {"raw_response", "answer"}},
            indent=2,
        )
    )
    return 0 if record["score"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
