"""Run one direct Claude call on the public benchmark inputs.

This is the no-pipeline control: no invented representations, tools, subagents,
private evaluator data, or iterative agent loop.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from reagents.llm.anthropic_client import DEFAULT_MODEL

CASE_DIR = Path(__file__).parent
ROOT = CASE_DIR.parent.parent


def load_env_file() -> None:
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
    question_path = CASE_DIR / "public" / "question.json"
    observations_path = CASE_DIR / "public" / "observations.csv"
    question = json.loads(question_path.read_text(encoding="utf-8"))
    observations = observations_path.read_text(encoding="utf-8")
    prompt = (
        f"Experimental setup:\n{question['statement']}\n\n"
        f"Constraints:\n- "
        + "\n- ".join(question["constraints"])
        + f"\n\nQuestion:\n{question['question']}\n\n"
        f"observations.csv:\n```csv\n{observations}```\n"
    )
    digest = hashlib.sha256(
        question_path.read_bytes() + b"\0" + observations_path.read_bytes()
    ).hexdigest()
    return prompt, digest


async def run(model: str, output: Path) -> None:
    import anthropic

    prompt, public_input_sha256 = public_prompt()
    client = anthropic.AsyncAnthropic()
    message = await client.messages.create(
        model=model,
        max_tokens=6000,
        system=(
            "You are a scientific reasoning assistant. Solve the supplied task "
            "directly from the public observations. Do not delegate or assume "
            "facts not present in the input. Give a concise but adequately "
            "justified answer to every numbered request."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    answer = "\n".join(
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text"
    )
    record = {
        "kind": "single_claude_no_pipeline",
        "model": model,
        "public_input_sha256": public_input_sha256,
        "answer": answer,
        "stop_reason": message.stop_reason,
        "usage": {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(answer)
    print(f"\nSaved baseline record to {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_env_file()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is missing from the process and .env")
    asyncio.run(run(args.model, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
