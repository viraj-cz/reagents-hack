"""Builds the DEMI_GOD system prompt from a spec.

RUNNER-INDEPENDENT -- and this is the least obvious of the runner-independent
modules, so it is worth stating why. The prompt describes *the agent's job*: its
domain, its tools, its paths, its output contract. None of that is a statement
about which process is calling the model. An outside-the-sandbox runner passes
this exact string to the same SDK; only the transport differs.

The one thing that must never leak in here is orchestration. A DEMI_GOD is told
nothing about siblings, about the GOD, or about how its output will be
recombined. That ignorance keeps the representations independent -- an agent
that knows another candidate exists may defer an obligation instead of solving
the complete projected objective itself.
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
domain, do NOT switch back to the original field. Express the missing operation
inside your representation or record a genuine information loss in `unknowns`.

You are not solving one slice of a larger task. Your representation contains the
complete objective, and your artifact must be an independently complete candidate
solution: satisfy every listed constraint, produce every required output, analyze
robustness or counterexamples, and provide a certificate that can be checked after
your symbols are translated back. Nothing else is assigned to fill a gap for you.

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
{toolbox_section}
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

# Your manifest: write it EARLY, then keep it current

`{out}/{result_filename}` is the one file that must exist. Write it **within
your first few actions** — as soon as you have any claim at all, even a bad one
— and rewrite it each time your answer improves.

Do NOT save it for the end. You have a limited number of turns and you will not
be warned before they run out. A manifest written early and refined twice beats
a perfect one you never got to write: if you are cut off, whatever is on disk at
that moment is your entire contribution.

A first pass with `confidence` 0.1 and a rough `claim` is a good use of an early
turn. Overwrite it as you learn more.

It must match this schema:

```json
{schema}
```

Field notes, in the order people get them wrong:

- `claim`: your finding, stated plainly, in your domain. Not a summary of what
  you did -- the answer.
- `confidence`: 0.0-1.0, calibrated. Low confidence honestly reported is
  useful. High confidence wrongly reported poisons the recombination.
- `payload`: the same finding, machine-readable, in exactly the shape the
  schema above gives. `claim` is read by a person; `payload` is read by code.
  They must agree.
- A successful `payload` is a complete candidate, not a partial observation for
  another agent to finish. Populate every required solution, constraint, and
  certificate field even when some values are explicitly unknown.
- `method`: enough that someone could reproduce your result without you. The
  procedure.
- `justification`: why the claim actually follows, argued in your domain's own
  terms. The argument, not the procedure -- whatever reads your output has to
  weigh it against other findings, and it can only do that if you show why.
- `evidence`: paths relative to `{out}` that back `claim` specifically.
- `unknowns`: what you could not determine, including everything you deferred
  as out-of-domain. An empty `unknowns` on a hard problem reads as a failure to
  notice, not as thoroughness.
- `blockers`: what actively stopped you. Missing data, missing tool, ambiguous
  spec.
- `files`: everything you wrote, relative to `{out}`.

Do not set `demigod_name`, `domain_name`, `run_id`, `status` or `error`. They
are not yours to write and will be overwritten.

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
    elif spec.toolbox is not None:
        # A brokered lease is a real toolset. Saying "no tools" here and then
        # describing `toolbox` two lines later reads as a contradiction, and an
        # agent that believes the first sentence never runs the command.
        tools_section = (
            "Nothing extra is installed in your environment, but you hold a "
            "toolbox lease -- see below."
        )
    else:
        tools_section = (
            "No domain tools. You have the standard file and shell tools only; "
            "reason from what is in `shared/`."
        )

    toolbox_section = build_toolbox_section(spec)

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
        toolbox_section=toolbox_section,
        result_filename=RESULT_FILENAME,
        # The domain's artifact_schema replaces the generic `payload` slot, so
        # the agent is shown the exact structure it must produce rather than a
        # bare "object". Envelope fields are excluded either way.
        schema=json.dumps(result_json_schema(spec.artifact_schema or None), indent=2),
        misc_section=misc_section,
    )


_TOOLBOX_TEMPLATE = """
## Brokered tools -- the `toolbox` command

These run somewhere else, on a broker, and you reach them over the network with
one command. They are NOT Python imports; there is no library to import.

```
toolbox list                            # what you may call, and how many calls remain
toolbox describe {example}              # the input schema for one tool
toolbox call {example} -i args.json     # run it; the result prints as JSON
toolbox call {example} -i - <<'EOF'     # or pass arguments on stdin
{{"some_argument": 1}}
EOF
```

Start with `toolbox list`. Read `toolbox describe <tool>` before the first call
to a tool -- the input must match its schema exactly or the broker rejects it,
and a rejected call tells you what was wrong.

Granted to you: {ids}

Rules that will otherwise cost you the run:

- **Your calls are metered.** `toolbox list` shows how many remain. When they
  are gone the broker refuses, and there is no way to ask for more. Spend them
  on the calls that decide your claim, not on exploring.
- **A refusal is not a retry.** `[lease_exhausted]`, `[unbound_tool]` and
  `[write_denied]` are permanent. Only `[invalid_input]` is worth another
  attempt, with corrected input.
- **Save what you get.** `toolbox call ... -o result.json` writes the output
  into your output directory, where it becomes evidence. A number that appears
  only in your transcript did not survive.
- **Never invent a result.** If a tool you need refuses, record it in
  `blockers`. A fabricated tool output is the single worst thing you can
  produce: it is indistinguishable from a real one to whatever reads you next.
"""


def build_toolbox_section(spec: DemiGodSpec) -> str:
    """The brokered-tool half of the tools section. Empty when no lease exists.

    Deliberately a CLI paragraph rather than N tool schemas. Schemas are
    permanent context -- four brokered tools would put four JSON Schemas in
    every turn of this agent's window -- and `toolbox describe` fetches one on
    demand, at the moment it is about to be used.
    """
    grant = spec.toolbox
    if grant is None:
        return ""
    ids = grant.tool_ids
    if not ids:
        # A lease with no tools is a configuration mistake worth surfacing to
        # the agent rather than a silent empty section it reasons around.
        return (
            "\n## Brokered tools\n\nA toolbox lease was issued to you but it "
            "grants no tools. Run `toolbox list` to confirm, then record it in "
            "`blockers`.\n"
        )
    example = ids[0]
    return _TOOLBOX_TEMPLATE.format(
        example=example,
        ids=", ".join(f"`{i}`" for i in ids),
    )


def build_task_prompt(spec: DemiGodSpec) -> str:
    """The opening user turn. The system prompt holds the standing rules; this
    is just the kick-off, kept short so it does not restate and contradict it."""
    return (
        f"Begin. Your goal: {spec.problem.goal.strip()}\n\n"
        f"Work in {OUT_MOUNT}. Write an initial {OUT_MOUNT}/{RESULT_FILENAME} "
        f"early -- a rough claim at low confidence is fine -- then refine it as "
        f"you go. You have {spec.max_turns} turns and will not be warned before "
        f"they run out."
    )
