"""One standard Claude call on the native tetraploid problem, without tools."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

from benchmarks.polyploid_phasing.benchmark import (
    PUBLIC_DIR,
    load_problem,
    score_solution,
)
from reagents.llm.anthropic_client import DEFAULT_MODEL, _extract_json_object

ROOT = Path(__file__).parents[2]


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
    public = {
        "problem": load_problem().model_dump(mode="json"),
        "observations": json.loads((PUBLIC_DIR / "observations.json").read_text()),
    }
    digest = hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
    return (
        "Solve this frozen real-data tetraploid phasing problem directly in its "
        "native biological representation. No tools, code, files, retrieval, "
        "subagents, or follow-up messages are available. Return JSON only in the "
        "problem's exact answer schema, with four complete 42-site haplotypes and "
        "a partition of all sites into phase blocks.\n\n"
        f"{json.dumps(public, indent=2)}",
        digest,
    )


async def run(model: str, output: Path) -> dict:
    import anthropic

    prompt, digest = public_prompt()
    started = time.time()
    message = await anthropic.AsyncAnthropic().messages.create(
        model=model,
        max_tokens=16000,
        system="You are the no-tool control. Return one JSON object and no prose.",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "\n".join(
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text"
    )
    record = {
        "kind": "single_claude_native_no_tools_polyploid",
        "model": model,
        "api_calls": 1,
        "tools_provided": [],
        "public_input_sha256": digest,
        "elapsed_s": round(time.time() - started, 3),
        "usage": {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        },
        "raw_response": raw,
        "answer": None,
        "score": None,
        "parse_error": None,
        "response_committed_before_private_scoring": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        candidate = _extract_json_object(raw)
        if candidate is None:
            raise ValueError("response did not contain a complete JSON object")
        answer = json.loads(candidate)
        record["answer"] = answer
        output.write_text(json.dumps(record, indent=2), encoding="utf-8")
        record["score"] = score_solution(answer, private=True)
    except Exception as exc:
        record["parse_error"] = str(exc)
        record["score"] = {
            "passed": False,
            "score_10": 0.0,
            "errors": [f"invalid response: {exc}"],
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
        raise SystemExit("ANTHROPIC_API_KEY is missing")
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
