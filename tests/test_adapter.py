"""The GOD->DEMI_GOD translation. No Modal, no network, no API key.

Every test here guards a place where the two halves of the system disagree
about vocabulary. Those disagreements are silent by nature -- they surface as a
validation error on the first live spawn, or worse, as an agent quietly missing
a tool -- so they are pinned here instead.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from demigod.spec import DemiGodSpec
from reagents.contracts import (
    Axis,
    Budget,
    ContextEnvelope,
    DomainProblem,
    DomainSpec,
)
from reagents.demigod.adapter import envelope_to_spec, slugify_domain_name

ARTIFACT_SCHEMA = {
    "type": "object",
    "required": ["findings", "conclusion"],
    "properties": {
        "findings": {"type": "array", "items": {"type": "string"}},
        "conclusion": {"type": "string"},
    },
}


def make_envelope(**domain_overrides) -> ContextEnvelope:
    fields = {
        "name": "stoichiometric_flow",
        "axes": [Axis.CONSERVATION, Axis.TOPOLOGY],
        "language": "hypergraph over token multisets; edges conserve token counts",
        "transform_prompt": (
            "SECRET: God's own instruction, must never reach a demigod"
        ),
        "tool_ids": ["formal.z3_solve", "graph.build"],
        "artifact_schema": ARTIFACT_SCHEMA,
        "forbidden": ["Use only symbols defined in the representation."],
    }
    domain = DomainSpec(**{**fields, **domain_overrides})
    return ContextEnvelope(
        domain=domain,
        problem=DomainProblem(
            domain_name=domain.name,
            representation={"nodes": ["t1", "t2"], "edges": [["t1", "t2"]]},
            task="Determine whether token count is conserved along every edge.",
            notation_guide="t<i> denotes a token species; edges are directed.",
        ),
        tools=[],
        artifact_schema=ARTIFACT_SCHEMA,
        budget=Budget(max_steps=6, wall_time_s=60.0),
    )


# --- names ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("domain_name", "expected"),
    [
        ("stoichiometric_flow", "stoichiometric-flow"),
        ("catalytic_dag", "catalytic-dag"),
        ("rate_orbit", "rate-orbit"),
        ("Already-Hyphenated", "already-hyphenated"),
    ],
)
def test_slugify_bridges_the_two_naming_rules(domain_name, expected):
    assert slugify_domain_name(domain_name) == expected


@pytest.mark.parametrize(
    "domain_name",
    [
        "stoichiometric_flow",
        "a",
        "x" * 60,
        "9_leading_digit",
        "__weird__",
        "rate_orbit",
    ],
)
def test_every_slug_satisfies_the_demigod_name_validator(domain_name):
    """The load-bearing one. reagents' own validator REQUIRES identifier style
    (underscores); demigod's FORBIDS underscores because the name becomes a
    directory and a Modal sandbox name. Without slugify, every domain the
    planner invents is rejected on the first spawn."""
    slug = slugify_domain_name(domain_name)
    DemiGodSpec.model_validate(
        {
            "name": slug,
            "domain": "d",
            "tools": [],
            "problem": {"context": "c", "goal": "g"},
        }
    )


def test_underscored_name_would_be_rejected_without_slugify():
    """Proves the previous test is testing something real."""
    with pytest.raises(ValidationError):
        DemiGodSpec.model_validate(
            {
                "name": "stoichiometric_flow",
                "domain": "d",
                "tools": [],
                "problem": {"context": "c", "goal": "g"},
            }
        )


# --- sealing ----------------------------------------------------------------


def test_transform_prompt_never_reaches_the_spec():
    """transform_prompt is God's instruction to itself and describes the NATIVE
    problem. The orchestrator blanks it before building the envelope; this
    asserts the adapter does not resurrect it from anywhere."""
    spec = envelope_to_spec(make_envelope())
    rendered = spec.model_dump_json()
    assert "SECRET" not in rendered
    assert "God's own instruction" not in rendered


def test_representation_survives_as_structure_not_paraphrase():
    spec = envelope_to_spec(make_envelope())
    assert "t1" in spec.problem.context
    assert "notation" in spec.problem.context.lower() or "denotes" in (
        spec.problem.context
    )


# --- tools ------------------------------------------------------------------


def test_unmapped_tools_are_reported_not_silently_dropped():
    """A dropped tool produces an agent that invents results it could not have
    computed. It must be told, in the prompt, that the tool is unreachable."""
    spec = envelope_to_spec(make_envelope())
    assert spec.tools == []
    assert set(spec.miscellaneous["unavailable_tools"]) == {
        "formal.z3_solve",
        "graph.build",
    }
    assert "do not simulate" in spec.miscellaneous["unavailable_tools_note"].lower()


def test_mapped_tools_resolve_into_the_demigod_registry():
    spec = envelope_to_spec(
        make_envelope(tool_ids=["data.frames", "graph.build"]),
        tool_map={"data.frames": "pandas"},
    )
    assert spec.tools == ["pandas"]
    assert spec.miscellaneous["unavailable_tools"] == ["graph.build"]


def test_a_tool_map_naming_an_unregistered_key_fails_at_spec_time():
    """Mapping to a key demigod does not have must fail here, not inside a
    live sandbox as an ImportError.

    NOTE: tool_ids must stay >= 2 -- DomainSpec enforces MinLen(2), so a
    one-element list would raise ValidationError from DomainSpec itself and
    this test would pass without ever reaching the tool map.
    """
    with pytest.raises(ValidationError) as e:
        envelope_to_spec(
            make_envelope(tool_ids=["formal.z3_solve", "graph.build"]),
            tool_map={"formal.z3_solve": "z3-does-not-exist-in-registry"},
        )
    # Assert on the message so this cannot silently start passing for the
    # wrong reason again: the error must name the bad key AND the valid set.
    message = str(e.value)
    assert "z3-does-not-exist-in-registry" in message
    assert "pandas" in message


# --- budget -----------------------------------------------------------------


def test_wall_time_is_not_mapped_one_to_one_onto_sandbox_lifetime():
    """Budget.wall_time_s defaults to 60 and doubles as the capability-lease
    expiry; max_lifetime_s is the sandbox kill timer and defaults to 3600. A
    1:1 mapping would kill every sandbox after a minute."""
    spec = envelope_to_spec(make_envelope())
    assert spec.max_lifetime_s >= 600
    assert spec.max_turns == 6  # Budget.max_steps maps straight across


# --- schema round trip ------------------------------------------------------


def test_artifact_schema_rides_through_to_the_spec():
    spec = envelope_to_spec(make_envelope())
    assert spec.artifact_schema == ARTIFACT_SCHEMA
    # ...and its required keys are surfaced as success criteria the agent sees
    # up front rather than discovering at validation time.
    joined = " ".join(spec.problem.success_criteria)
    assert "findings" in joined and "conclusion" in joined


def test_both_identities_are_preserved():
    spec = envelope_to_spec(make_envelope())
    assert spec.name == "stoichiometric-flow"  # infrastructure identity
    assert spec.domain_name == "stoichiometric_flow"  # domain identity
