"""THE SWAPPABLE SEAM. Read this before changing anything in this package.

Everything else in `demigod/` answers "what is a DEMI_GOD?":

    spec.py      what the GOD supplies
    result.py    what comes back
    registry/    which tools exist, and how to use them
    images.py    what environment they run in
    layout.py    where files live
    prompt.py    what the agent is told

This package answers a different and much less settled question: **who drives
the agent loop?**

That decision is PROVISIONAL. Today the loop runs inside the sandbox
(`InsideSandboxRunner`): we exec `python -m demigod.entrypoint` and the Agent
SDK runs next to the tools, with the model's Bash/Write calls hitting the
sandbox filesystem directly. The alternative is to run the loop on the caller
and use the sandbox purely as an execution target for tool calls
(`OutsideSandboxRunner`, stubbed).

The invariant that makes the swap cheap: **nothing outside this package imports
`claude_agent_sdk`, and nothing outside this package knows where the loop
runs.** If you find yourself wanting `if inside:` in prompt.py or images.py,
that is the design breaking -- push the branch back in here.

Both runners satisfy the same three-line contract:

    run(spec, run_id) -> DemiGodResult

taking a spec and returning a validated manifest, having left artifacts in
`out/<name>/` on the run volume. Everything else -- image resolution, mounting,
prompt construction, result validation -- is shared and lives outside.
"""

from __future__ import annotations

from typing import Protocol

from demigod.result import DemiGodResult
from demigod.spec import DemiGodSpec


class Runner(Protocol):
    """Drives one DEMI_GOD from spec to manifest.

    Implementations MUST:
      * leave artifacts in `out/<spec.name>/` on the run volume;
      * return a validated `DemiGodResult`, never raise for agent-level failure
        (a failed agent is a `status="failed"` manifest, not an exception);
      * respect `spec.max_lifetime_s` and `spec.idle_timeout_s` as hard caps;
      * tear down the sandbox on every path, including exceptions.

    Implementations MUST NOT own: the prompt, the tool registry, the image
    catalog, the volume layout, or the shape of the result.
    """

    name: str

    def run(self, spec: DemiGodSpec, run_id: str) -> DemiGodResult:
        """Execute one DEMI_GOD to completion."""
        ...


def get_runner(kind: str = "inside") -> Runner:
    """Select a runner. The single place the inside/outside choice is made.

    Threaded through from `spawn.py --runner`, so flipping the default is a
    one-word change and A/B-ing the two is a flag.
    """
    if kind == "inside":
        from demigod.runner.inside import InsideSandboxRunner

        return InsideSandboxRunner()
    if kind == "outside":
        from demigod.runner.outside import OutsideSandboxRunner

        return OutsideSandboxRunner()
    raise ValueError(f"unknown runner {kind!r}; expected 'inside' or 'outside'")


__all__ = ["Runner", "get_runner"]
