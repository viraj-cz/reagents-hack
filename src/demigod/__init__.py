"""Spawn one DEMI_GOD agent, reliably, in an isolated Modal sandbox.

Scope: this package builds the *spawning mechanism* only. There is no GOD here
-- no decomposition, no fan-out, no synthesis. The GOD is, for now, whatever
calls `spawn_demigod()` with a valid spec.

Module map, by whether the inside/outside-sandbox decision can touch it:

  RUNNER-INDEPENDENT (must never import claude_agent_sdk, must never branch on
  where the loop runs):
    spec.py       DemiGodSpec        -- the input contract
    result.py     DemiGodResult      -- the output contract (files + manifest)
    registry/     ToolEntry          -- the closed tool set
    images.py     PrebakedImage      -- pre-baked image catalog + resolver
    layout.py     RunLayout          -- volume layout (shared/ ro, out/<name>/ rw)
    prompt.py     build_system_prompt -- what the agent is told

  RUNNER-DEPENDENT (the seam; swapping inside->outside touches only these):
    runner/       Runner protocol, InsideSandboxRunner, OutsideSandboxRunner
    entrypoint.py what executes inside the sandbox today

  CALLER-FACING:
    spawn.py      spawn_demigod()    -- the one function/command
"""

from demigod.result import DemiGodResult
from demigod.spec import DemiGodSpec, Problem

__all__ = ["DemiGodResult", "DemiGodSpec", "Problem", "spawn_demigod"]


def __getattr__(name: str):
    # Lazy so that `import demigod` does not pull in `modal`.
    if name == "spawn_demigod":
        from demigod.spawn import spawn_demigod

        return spawn_demigod
    raise AttributeError(name)
