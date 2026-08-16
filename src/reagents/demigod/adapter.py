"""GOD-side glue: turn a sealed `ContextEnvelope` into a `DemiGodSpec`.

This is the whole translation between the two halves of the system. `reagents`
decides WHAT a demigod reasons about; `demigod` decides WHERE and HOW it runs.
Everything that has to be reconciled between those two vocabularies is here and
nowhere else.

Three reconciliations are worth knowing about:

**Names.** `reagents` domain names are snake_case identifiers
(`stoichiometric_flow`). `demigod` names become a directory and a Modal sandbox
name, so its validator forbids underscores. `slugify_domain_name` bridges them,
and the result keeps BOTH: `demigod_name` (infrastructure) and `domain_name`
(domain identity).

**The problem.** `reagents` hands over a structured `DomainProblem`
(representation + notation guide); `demigod`'s prompt builder wants prose. The
representation is rendered as JSON under its notation guide rather than
flattened away, so the agent still sees symbols rather than a paraphrase.

**Tools -- and there are now TWO answers, because there were always two kinds of
tool.** `reagents` tool ids are dotted and resolve to in-process callables behind
a capability lease (`formal.z3_solve`); `demigod` tool keys are flat and resolve
to pip packages baked into a Modal image (`pandas`). These do not line up by
renaming and never will, because they are different things: one is a function
someone else owns, the other is a library on your own disk.

- A pip package is an IMAGE concern. `tool_map` still handles it: name the
  demigod-registry key and it gets baked in. That path is unchanged.
- A callable is a BROKER concern. `toolbox` handles it: GOD publishes the
  capability lease to the TOOLBOX_BROKER and hands the demigod a URL and a lease
  id, and the demigod calls the real function over HTTP with `toolbox call`.
  The ids do not change: `formal.z3_solve` is `formal.z3_solve` on both sides.

Anything in neither -- no image key, no broker lease -- is still reported to the
agent as unavailable, loudly, in `miscellaneous`. A silently-dropped tool
produces an agent that invents results it had no way to compute.
"""

from __future__ import annotations

import json
import re
from typing import Any

from demigod.egress import allowlist
from demigod.spec import DemiGodSpec, Problem
from demigod.toolbox.protocol import ToolboxGrant
from reagents.contracts import ContextEnvelope

_NON_SLUG = re.compile(r"[^a-z0-9]+")

MAX_NAME_LEN = 32
MIN_NAME_LEN = 3


def slugify_domain_name(name: str) -> str:
    """`stoichiometric_flow` -> `stoichiometric-flow`.

    Must satisfy demigod's `^[a-z][a-z0-9-]{1,30}[a-z0-9]$`: lowercase, hyphens
    only, 3-32 chars, starts with a letter, ends alphanumeric. Every domain name
    reagents' planner produces fails that regex as-is, so this is not optional
    polish -- without it the first spawn dies in spec validation.
    """
    slug = _NON_SLUG.sub("-", name.strip().lower()).strip("-")
    if not slug or not slug[0].isalpha():
        slug = f"d-{slug}" if slug else "domain"
    slug = slug[:MAX_NAME_LEN].rstrip("-")
    if len(slug) < MIN_NAME_LEN:
        slug = f"{slug}-dg"[:MAX_NAME_LEN]
    return slug


def render_domain(envelope: ContextEnvelope) -> str:
    """The `domain` prose demigod's prompt builder expects.

    Built from the envelope's structured DomainSpec: the invented language, the
    axes it is seated on, and anything explicitly forbidden. `transform_prompt`
    is NOT included -- the orchestrator already blanks it, because it is GOD's
    instruction to itself and would leak the native problem.
    """
    domain = envelope.domain
    parts = [domain.language.strip()]
    axes = ", ".join(a.value for a in domain.axes)
    parts.append(f"You reason along these axes and no others: {axes}.")
    forbidden = list(envelope.forbidden) or list(domain.forbidden)
    if forbidden:
        rules = "\n".join(f"- {f}" for f in forbidden)
        parts.append(f"Hard constraints on your reasoning:\n{rules}")
    return "\n\n".join(parts)


def render_context(envelope: ContextEnvelope) -> str:
    """The problem, still in domain representation. Never the native problem."""
    problem = envelope.problem
    return (
        f"{problem.notation_guide.strip()}\n\n"
        f"Representation:\n```json\n"
        f"{json.dumps(problem.representation, indent=2)}\n```\n\n"
        f"Complete-objective manifest:\n```json\n"
        f"{problem.projection_manifest.model_dump_json(indent=2)}\n```"
    )


def envelope_to_spec(
    envelope: ContextEnvelope,
    *,
    tool_map: dict[str, str] | None = None,
    toolbox: ToolboxGrant | None = None,
    files: list[str] | None = None,
    max_turns: int | None = None,
    model: str | None = None,
    cpu: float = 1.0,
    memory_mb: int = 2048,
    restrict_egress: bool = False,
) -> DemiGodSpec:
    """Sealed envelope -> a spawnable DemiGodSpec.

    `files` are paths under the run's shared volume. GOD chooses them; the
    caller is responsible for having seeded them (see RunLayout.seed_shared),
    because shared/ is read-only at every mount and cannot be filled from inside.

    `toolbox` is a lease already published to the broker (see
    `broker.session.ToolboxSession.grant`). Publishing it is the caller's job,
    not this function's: minting authority and translating vocabulary are
    different responsibilities, and an adapter that could mint would be a second
    path around GOD's operator-approval checks.

    `restrict_egress` pins the sandbox's outbound network to the agent API plus
    the broker. Off by default -- turning a prompt instruction into a firewall
    rule is a change in behaviour, and it should be one someone opted into.
    """
    tool_map = tool_map or {}
    brokered = set(toolbox.tool_ids) if toolbox else set()

    mapped = list(
        dict.fromkeys(
            tool_map[t] for t in envelope.domain.tool_ids if t in tool_map
        )
    )
    # Shared input files are tables the demigod can only read if pandas is in
    # the image. Without this, a plan that named only in-process tools spawned
    # a sandbox that could see shared/ and had no way to open it.
    if files and "pandas" not in mapped:
        mapped.append("pandas")
    unmapped = [
        t
        for t in envelope.domain.tool_ids
        if t not in tool_map and t not in brokered
    ]

    misc: dict[str, Any] = {
        "axes": [a.value for a in envelope.domain.axes],
        "domain_name": envelope.domain.name,
    }
    if unmapped:
        # Loud on purpose. A silently-dropped tool produces an agent that
        # invents results it had no way to compute.
        misc["unavailable_tools"] = unmapped
        misc["unavailable_tools_note"] = (
            "These tools were selected for your domain but are NOT reachable "
            "from your environment yet. Do not simulate, approximate, or "
            "pretend to call them. If your goal requires one, record it in "
            "`blockers` and produce whatever partial result you honestly can."
        )

    budget = envelope.budget
    return DemiGodSpec(
        name=slugify_domain_name(envelope.domain.name),
        domain_name=envelope.domain.name,
        domain=render_domain(envelope),
        tools=mapped,
        toolbox=toolbox,
        egress_domains=(
            allowlist(toolbox.base if toolbox else None) if restrict_egress else None
        ),
        problem=Problem(
            context=render_context(envelope),
            goal=envelope.problem.task,
            success_criteria=_criteria_from_schema(envelope.artifact_schema),
        ),
        files=files or [],
        artifact_schema=envelope.artifact_schema,
        miscellaneous=misc,
        # Budget.wall_time_s defaults to 60 and doubles as the capability-lease
        # expiry; max_lifetime_s is the sandbox kill timer and defaults to 3600.
        # Mapping them 1:1 would kill every sandbox after a minute, so the
        # sandbox is given headroom over the lease rather than matching it.
        max_lifetime_s=max(int(budget.wall_time_s * 1.25), 600),
        idle_timeout_s=120,
        max_turns=max_turns or budget.max_steps,
        # None means "keep the spec's own default" rather than "no model" --
        # DemiGodSpec.model is non-optional and pinned there.
        **({"model": model} if model else {}),
        cpu=cpu,
        memory_mb=memory_mb,
    )


def _criteria_from_schema(artifact_schema: dict[str, Any]) -> list[str]:
    """Turn the domain's required payload keys into checkable success criteria.

    reagents expresses "what a good answer contains" structurally, as an
    artifact_schema; demigod expresses it as prose criteria in the prompt. The
    schema is enforced on the way out either way -- this just means the agent is
    told up front rather than discovering it at validation time.
    """
    required = artifact_schema.get("required") or []
    criteria = [f"`payload` includes {key!r}" for key in required]
    criteria.append(
        "the artifact is one complete candidate satisfying every projected objective"
    )
    criteria.append("`payload` validates against the schema shown below")
    return criteria
