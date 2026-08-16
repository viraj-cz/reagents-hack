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
from reagents.toy import simple_problem, toy_problem

_BUILDERS = {
    "pfk-bottleneck": toy_problem,
    "valve-bottleneck": simple_problem,
}


def _describe(
    problem: NativeProblem, *, label: str, modes: list[str], blurb: str
) -> dict[str, Any]:
    return {
        "id": problem.id,
        "label": label,
        "blurb": blurb,
        "modes": modes,
        "prompt": f"{problem.statement}\n\n{problem.question}",
        "entities": list(problem.entities),
        "constraints": list(problem.constraints),
    }


PRESETS: list[dict[str, Any]] = [
    _describe(
        toy_problem(),
        label="Glycolytic bottleneck",
        modes=["scripted", "live"],
        blurb="The recorded run. Three domains, seven tool calls, no API key.",
    ),
    _describe(
        simple_problem(),
        label="Three tanks in series",
        modes=["live"],
        blurb="Same shape, plain vocabulary. Needs live inference.",
    ),
]


def preset_problem(preset_id: str) -> NativeProblem | None:
    builder = _BUILDERS.get(preset_id)
    return builder() if builder else None
