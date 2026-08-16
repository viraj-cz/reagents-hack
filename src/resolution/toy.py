"""Problems the UI can start from, and which modes each one actually supports.

`ScriptedLLM.for_toy_pathway()` is keyed by phase name -- `transform:catalytic_dag`,
`demigod:rate_orbit` and so on -- so it can only replay the one problem those
scripts were written against. Any other prompt raises `no script for phase`.
That is a property of the offline stand-in, not a limitation worth hiding: the
preset carries `modes`, and the UI refuses to offer a replay of a problem no
recording exists for.
"""

from __future__ import annotations

from typing import Any

from reagents.contracts import NativeProblem
from reagents.toy import arithmetic_problem, schedule_problem, toy_problem

_BUILDERS = {
    "shop-change": arithmetic_problem,
    "release-schedule": schedule_problem,
    "pfk-bottleneck": toy_problem,
}


def _describe(
    problem: NativeProblem,
    *,
    label: str,
    modes: list[str],
    blurb: str,
    tier: str,
) -> dict[str, Any]:
    return {
        "id": problem.id,
        "label": label,
        "blurb": blurb,
        "tier": tier,
        "modes": modes,
        "prompt": f"{problem.statement}\n\n{problem.question}",
        "entities": list(problem.entities),
        "constraints": list(problem.constraints),
    }


PRESETS: list[dict[str, Any]] = [
    # ORDERED BY HOW MUCH A SECOND OPINION IS WORTH, which is the only axis the
    # dynamic option responds to. The counts in these blurbs are observed, not
    # promised: they are what GOD chose on a live Opus 4.8 run with the count
    # left to it, and a pinned 2/3/4 overrides them by definition.
    _describe(
        arithmetic_problem(),
        label="Change from a hundred",
        modes=["live"],
        tier="easy",
        blurb="Arithmetic. Left to itself GOD answers this one, spawning nothing.",
    ),
    _describe(
        schedule_problem(),
        label="Release critical path",
        modes=["live"],
        tier="medium",
        blurb="Real dependencies, one obvious representation. It chose two domains.",
    ),
    _describe(
        toy_problem(),
        label="Glycolytic bottleneck",
        modes=["scripted", "live"],
        tier="hard",
        blurb="The recorded run. Three domains, seven tool calls, no API key.",
    ),
]


def preset_problem(preset_id: str) -> NativeProblem | None:
    builder = _BUILDERS.get(preset_id)
    return builder() if builder else None
