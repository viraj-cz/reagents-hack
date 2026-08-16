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

    truncated = False
    try:
        async for message in query(prompt=build_task_prompt(spec), options=options):
            # Coarse but useful: this is what streams back to the caller's
            # console. TODO: structured logging + token/cost accounting once the
            # GOD needs to budget across many DEMI_GODs.
            print(f"[agent] {_summarize(message)}", flush=True)
    except Exception as e:
        if not _is_turn_limit(e):
            raise
        # Running out of turns is a BUDGET event, not a crash. The SDK raises
        # on the cap, which -- before this was handled -- killed the process
        # before the manifest was ever verified, so a demigod that had done six
        # turns of real work returned literally nothing. Cap and keep whatever
        # exists instead.
        truncated = True
        print(
            f"[entrypoint] turn limit ({spec.max_turns}) reached -- keeping "
            f"partial work",
            file=sys.stderr,
            flush=True,
        )

    return _verify_manifest(spec, truncated=truncated)


_TURN_LIMIT_MARKERS = ("maximum number of turns", "max_turns")


def _is_turn_limit(exc: Exception) -> bool:
    """Is this the SDK's turn-cap error rather than a genuine failure?

    Matched on message text because the SDK raises a bare `Exception` carrying
    the CLI's error string ("Claude Code returned an error result: Reached
    maximum number of turns (N)") with no typed subclass to catch. If a future
    SDK adds one, match on that instead -- this is deliberately narrow so any
    other error still propagates and fails the run loudly.
    """
    return any(m in str(exc).lower() for m in _TURN_LIMIT_MARKERS)


def _summarize(message: object) -> str:
    """One line per content block, so a live run is actually readable.

    The previous version printed text blocks and fell through to
    `type(message).__name__` for everything else -- so every tool call, tool
    result, and file write appeared as a bare `AssistantMessage`. You could see
    THAT the agent acted, never WHAT it did, which makes a streamed run useless
    for diagnosis.

    Defensive throughout: block shapes vary by SDK version, and a logging helper
    must never be the thing that kills a run.
    """
    blocks = getattr(message, "content", None)
    if not isinstance(blocks, list):
        return type(message).__name__

    lines: list[str] = []
    for b in blocks:
        kind = getattr(b, "type", None) or type(b).__name__
        text = getattr(b, "text", None)
        if text:
            lines.append(f"text: {text.strip()[:400]}")
            continue
        if kind == "tool_use" or hasattr(b, "input"):
            name = getattr(b, "name", "?")
            args = getattr(b, "input", {})
            # File writes are the interesting ones -- show the path and size
            # rather than dumping the whole file back into the log.
            if isinstance(args, dict):
                detail = args.get("file_path") or args.get("path") or ""
                if not detail:
                    detail = str(args.get("command", ""))[:160]
                if "content" in args:
                    detail = f"{detail} ({len(str(args['content']))} chars)"
            else:
                detail = str(args)[:160]
            lines.append(f"TOOL {name}: {detail}"[:400])
            continue
        result = getattr(b, "content", None)
        if result is not None and kind == "tool_result":
            flat = str(result).replace("\n", " ")[:200]
            err = " ERROR" if getattr(b, "is_error", False) else ""
            lines.append(f"  ->{err} {flat}")
            continue
        if getattr(b, "thinking", None):
            lines.append(f"thinking: {str(b.thinking).strip()[:200]}")
            continue
        lines.append(kind)

    return " | ".join(lines) if lines else type(message).__name__


def _verify_manifest(spec: DemiGodSpec, *, truncated: bool = False) -> int:
    """Confirm the agent wrote a valid result.json before we exit 0.

    Checked here, inside, as well as by the runner outside. Duplication is
    intentional: in here we can still say something useful about *which* field
    is wrong, and the exit code gives the runner an unambiguous signal.

    `truncated` means the turn budget ran out. In that case a missing manifest
    is not a failure to report -- it is work to salvage: whatever files the
    agent wrote are still on the volume, so we synthesize a manifest indexing
    them rather than discarding the run.
    """
    try:
        result = DemiGodResult.read(OUT_MOUNT)
    except Exception as e:
        if truncated:
            return _salvage(spec, reason=str(e))
        print(f"[entrypoint] FAIL: {e}", file=sys.stderr)
        return 2

    if truncated:
        # The agent wrote a manifest early (as instructed) but never got to
        # finalize it. Keep its content; mark it as incomplete so a consumer
        # weights it accordingly rather than treating it as a finished answer.
        result.status = "timeout"
        result.blockers = [
            *result.blockers,
            f"turn budget ({spec.max_turns}) exhausted before the agent finished",
        ]
        result.write(OUT_MOUNT)
        print(
            f"[entrypoint] {spec.name}: TRUNCATED but manifest kept "
            f"(confidence={result.confidence}, {len(result.files)} artifact(s))"
        )
        return 0

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


def _salvage(spec: DemiGodSpec, *, reason: str) -> int:
    """Synthesize a manifest for a run that ran out of turns before writing one.

    The agent's files are still on the volume. Indexing them turns a total loss
    into a low-confidence partial result that an orchestrator can still weigh,
    and that a human can still read.
    """
    out = Path(OUT_MOUNT)
    produced = sorted(
        p.name for p in out.iterdir() if p.is_file() and p.name != RESULT_FILENAME
    )
    result = DemiGodResult(
        claim="",
        confidence=0.0,
        method=(
            "Run was cut off by the turn budget before the agent wrote its own "
            "manifest. This manifest was synthesized by the entrypoint and "
            "indexes whatever files survived."
        ),
        files=produced,
        blockers=[
            f"turn budget ({spec.max_turns}) exhausted before any manifest was "
            f"written ({reason})"
        ],
        status="timeout",
    )
    result.write(out)
    print(
        f"[entrypoint] {spec.name}: TRUNCATED, salvaged {len(produced)} file(s) "
        f"into a synthesized manifest",
        file=sys.stderr,
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
