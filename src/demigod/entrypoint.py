"""Runs INSIDE the sandbox. Invoked by `InsideSandboxRunner` as:

    python -m demigod.entrypoint --spec /run/spec.json

This module is part of the runner seam even though it does not live in
`runner/` -- it cannot, because it has to be importable inside the image while
`runner/inside.py` runs on the caller. If the loop moves outside the sandbox,
this file is deleted along with `runner/inside.py` and nothing else changes.

It is deliberately thin. It reads the spec, asks `prompt.py` for the prompt,
runs the loop, and gets out of the way. Every decision of substance was made by
a runner-independent module before this process started.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from demigod.layout import OUT_MOUNT, SHARED_MOUNT
from demigod.prompt import build_system_prompt, build_task_prompt
from demigod.result import RESULT_FILENAME, DemiGodResult
from demigod.spec import DemiGodSpec

# The agent gets file and shell tools and nothing else. No WebSearch: a
# DEMI_GOD reasons over what is in shared/ using the tools it was granted. If it
# needs the internet, that is a registry entry, not an ambient capability.
ALLOWED_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep", "Bash"]


async def run_agent(spec: DemiGodSpec) -> int:
    """Run the DEMI_GOD loop to completion. Returns a process exit code."""
    from claude_agent_sdk import ClaudeAgentOptions, query

    options = ClaudeAgentOptions(
        system_prompt=build_system_prompt(spec),
        allowed_tools=ALLOWED_TOOLS,
        # The sandbox IS the isolation boundary, so in-loop permission prompts
        # buy nothing and would deadlock a non-interactive run.
        permission_mode="bypassPermissions",
        cwd=OUT_MOUNT,
        # shared/ is outside cwd; without this the agent cannot read its inputs.
        add_dirs=[SHARED_MOUNT],
        max_turns=spec.max_turns,
        # Do not pick up any .claude/ settings that happen to be on the image.
        # A DEMI_GOD's behavior must come from its spec alone, or runs stop
        # being reproducible.
        setting_sources=[],
    )

    print(f"[entrypoint] {spec.name}: domain={spec.domain!r} tools={spec.tools}")

    async for message in query(prompt=build_task_prompt(spec), options=options):
        # Coarse but useful: this is what streams back to the caller's console.
        # TODO: structured logging + token/cost accounting once the GOD needs to
        # budget across many DEMI_GODs.
        print(f"[agent] {_summarize(message)}", flush=True)

    return _verify_manifest(spec)


def _summarize(message: object) -> str:
    """One line per SDK message. Defensive: message types vary by SDK version."""
    text = getattr(message, "content", None)
    if isinstance(text, list):
        parts = [getattr(b, "text", "") for b in text]
        joined = " ".join(p for p in parts if p).strip()
        if joined:
            return joined[:500]
    return type(message).__name__


def _verify_manifest(spec: DemiGodSpec) -> int:
    """Confirm the agent wrote a valid result.json before we exit 0.

    Checked here, inside, as well as by the runner outside. Duplication is
    intentional: in here we can still say something useful about *which* field
    is wrong, and the exit code gives the runner an unambiguous signal.
    """
    try:
        result = DemiGodResult.read(OUT_MOUNT)
    except Exception as e:
        print(f"[entrypoint] FAIL: {e}", file=sys.stderr)
        return 2

    missing = [
        p for p in result.evidence + result.files if not (Path(OUT_MOUNT) / p).exists()
    ]
    if missing:
        # Not fatal. The manifest is the agent's own index and a stale path is a
        # quality signal, not a reason to throw away the run.
        print(
            f"[entrypoint] WARNING: {RESULT_FILENAME} references files that do "
            f"not exist: {missing}",
            file=sys.stderr,
        )

    print(
        f"[entrypoint] {spec.name}: ok, confidence={result.confidence}, "
        f"{len(result.files)} artifact(s)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="DEMI_GOD in-sandbox entrypoint")
    parser.add_argument("--spec", required=True, help="Path to spec.json")
    args = parser.parse_args()

    spec = DemiGodSpec.model_validate_json(Path(args.spec).read_text(encoding="utf-8"))
    return asyncio.run(run_agent(spec))


if __name__ == "__main__":
    raise SystemExit(main())
