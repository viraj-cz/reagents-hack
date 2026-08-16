"""Builds the DEMI_GOD system prompt from a spec.

RUNNER-INDEPENDENT -- and this is the least obvious of the runner-independent
modules, so it is worth stating why. The prompt describes *the agent's job*: its
domain, its tools, its paths, its output contract. None of that is a statement
about which process is calling the model. An outside-the-sandbox runner passes
this exact string to the same SDK; only the transport differs.

The one thing that must never leak in here is orchestration. A DEMI_GOD is told
nothing about siblings, about the GOD, or about how its output will be
recombined. That ignorance is what keeps the components orthogonal -- an agent
that knows another agent is covering statistics will defer to it and leave a
hole in its own decomposition.
"""

from __future__ import annotations

import json

from demigod.layout import OUT_MOUNT, SHARED_MOUNT
from demigod.registry import resolve
from demigod.result import RESULT_FILENAME, result_json_schema
from demigod.spec import DemiGodSpec

_TEMPLATE = """\
You are a DEMI_GOD agent. You reason in exactly one domain and produce files.

# Your domain

{domain}

Stay inside it. If solving your goal seems to require reasoning outside this
domain, do NOT reason outside it -- record the question in `unknowns` and
continue. Something else is covering that ground. Work that strays outside your
domain is worse than useless: it is confidently wrong in a domain you were not
selected for, and it will be recombined as if it were authoritative.

# Your problem

## Context
{context}

## Goal
{goal}

## Success criteria
{success_criteria}

# Your filesystem

- `{shared}` -- READ-ONLY input. Shared, identical for every agent. You cannot
  write here; the mount will reject it.
- `{out}` -- YOUR output directory. Everything you produce goes here. It is
  private to you.

{files_section}

# Your tools

{tools_section}

You have Read, Write, Edit, Glob, Grep and Bash. Anything not listed above is
not installed, and installing things is not your job -- if you need a tool you
do not have, that is a `blocker`, not a detour.

# Your output IS files

Your transcript is discarded. Only what you write to `{out}` survives, and it is
read by something that never saw you work. So:

- Every claim must be backed by a file. A number in prose with no artifact
  behind it will not be believed.
- Scripts you wrote are artifacts. Keep them. They are how your method is
  re-run.
- Write intermediate results as you go, not in one pass at the end. If you are
  cut off, whatever is on disk is what you produced.

# Finishing

Your LAST action is to write `{out}/{result_filename}`, matching this schema:

```json
{schema}
```

Field notes, in the order people get them wrong:

- `claim`: your finding, stated plainly, in your domain. Not a summary of what
  you did -- the answer.
- `confidence`: 0.0-1.0, calibrated. Low confidence honestly reported is
  useful. High confidence wrongly reported poisons the recombination.
- `evidence`: paths relative to `{out}` that back `claim` specifically.
- `method`: enough that someone could reproduce your result without you.
- `unknowns`: what you could not determine, including everything you deferred
  as out-of-domain. An empty `unknowns` on a hard problem reads as a failure to
  notice, not as thoroughness.
- `blockers`: what actively stopped you. Missing data, missing tool, ambiguous
  spec.
- `files`: everything you wrote, relative to `{out}`.

Write it even if you failed. A manifest with an empty `claim`, confidence 0.0
and a populated `blockers` is a useful result. Silence is not.
{misc_section}"""


def build_system_prompt(spec: DemiGodSpec) -> str:
    """Render the full DEMI_GOD system prompt for `spec`."""
    criteria = (
        "\n".join(f"- {c}" for c in spec.problem.success_criteria)
        or "- None specified. Use your judgment and state it in `method`."
    )

    if spec.files:
        listed = "\n".join(f"- `{SHARED_MOUNT}/{f}`" for f in spec.files)
        files_section = (
            f"Start with these (the rest of `{SHARED_MOUNT}` is also readable):\n"
            f"{listed}"
        )
    else:
        files_section = f"No files were called out. Explore `{SHARED_MOUNT}` yourself."

    entries = resolve(spec.tools)
    if entries:
        docs = "\n\n".join(e.usage_doc or f"## {e.display_name}" for e in entries)
        tools_section = (
            "These are installed and verified in your environment.\n\n" + docs
        )
    else:
        tools_section = (
            "No domain tools. You have the standard file and shell tools only; "
            "reason from what is in `shared/`."
        )

    misc_section = ""
    if spec.miscellaneous:
        misc_section = (
            "\n# Additional context from your caller\n\n```json\n"
            + json.dumps(spec.miscellaneous, indent=2)
            + "\n```\n"
        )

    return _TEMPLATE.format(
        domain=spec.domain.strip(),
        context=spec.problem.context.strip(),
        goal=spec.problem.goal.strip(),
        success_criteria=criteria,
        shared=SHARED_MOUNT,
        out=OUT_MOUNT,
        files_section=files_section,
        tools_section=tools_section,
        result_filename=RESULT_FILENAME,
        schema=json.dumps(result_json_schema(), indent=2),
        misc_section=misc_section,
    )


def build_task_prompt(spec: DemiGodSpec) -> str:
    """The opening user turn. The system prompt holds the standing rules; this
    is just the kick-off, kept short so it does not restate and contradict it."""
    return (
        f"Begin. Your goal: {spec.problem.goal.strip()}\n\n"
        f"Work in {OUT_MOUNT}. When you are done, write "
        f"{OUT_MOUNT}/{RESULT_FILENAME}."
    )
