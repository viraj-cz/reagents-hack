"""Runner: agent loop OUTSIDE the sandbox. NOT IMPLEMENTED -- exists to hold
the seam open and to make the cost of the swap concrete.

Shape it would take:

    create Sandbox (same image, same mounts, same timeouts)
      -> run the Agent SDK loop HERE, on the caller
      -> intercept Bash/Read/Write tool calls via ClaudeAgentOptions.can_use_tool
         and forward them to sandbox.exec / sandbox.open
      -> read back out/<name>/result.json
      -> terminate

What it buys:
  * The Anthropic API key never enters the sandbox.
  * The loop survives sandbox death -- a crashed sandbox becomes a retryable
    tool error instead of a lost run.
  * Transcripts, token accounting and interrupts are on the caller, where the
    future GOD can see them.

What it costs:
  * Every tool call becomes a network round-trip. For a Bash-heavy agent this
    is the dominant latency.
  * `can_use_tool` interception has to faithfully reimplement Read/Write/Edit
    against `sandbox.open`, including globbing and encoding. This is the actual
    work, and it is where the bugs will be.

Note what is NOT on that list: the prompt, the registry, the images, the volume
layout, the result contract. That is the seam paying off -- implementing this
file should not require editing any other module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from demigod.result import DemiGodResult
from demigod.spec import DemiGodSpec


class OutsideSandboxRunner:
    """Placeholder. See module docstring for the intended implementation."""

    name = "outside"

    def run(self, spec: DemiGodSpec, run_id: str) -> DemiGodResult:
        raise NotImplementedError(
            "The outside-the-sandbox runner is not implemented. The "
            "inside/outside decision is still open -- use `--runner inside`. "
            "If you are here to implement it, read the docstring in "
            "demigod/runner/outside.py first; it should be the only file you "
            "need to touch."
        )


if TYPE_CHECKING:
    from demigod.runner import Runner

    _: Runner = OutsideSandboxRunner()
