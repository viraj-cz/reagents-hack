"""The planner must never be shown a term it is checked against."""

from __future__ import annotations

import pytest

from reagents.contracts import NativeProblem
from reagents.god.anonymize import AnonymizationError, anonymize_problem
from reagents.isolation import find_leaks, native_terms
from reagents.toy import simple_problem, toy_problem


@pytest.mark.parametrize("factory", [simple_problem, toy_problem])
def test_no_native_term_survives_in_any_planner_visible_field(factory):
    """THE guarantee. Every field the planner is shown, scanned for every term.

    Stated over the whole visible surface rather than the statement alone,
    because the planner prompt embeds id, statement, entities, constraints and
    question -- a term surviving in any one of them is a term the planner can
    copy.
    """
    problem = factory()
    terms = native_terms(problem)
    assert terms, "nothing to anonymise; the test would pass vacuously"

    anon, mapping = anonymize_problem(problem)

    visible = "\n".join(
        [anon.id, anon.statement, *anon.constraints, anon.question, *anon.entities]
    )
    assert find_leaks(visible, terms) == []

    # The map is complete and reversible, so a planner decision about `e3` can
    # still be related back to the real entity by GOD.
    assert set(mapping.values()) == terms
    assert set(mapping) == set(anon.entities)


def test_structure_survives_so_the_planner_can_still_choose_tools():
    """Anonymisation is as wide as the check, and no wider.

    If it stripped general vocabulary too, the planner could not tell a flow
    problem from a folding one and tool selection would collapse -- which is
    the reason this is entity-scoped rather than a blanket redaction.
    """
    anon, _ = anonymize_problem(toy_problem())
    # Domain character: present in the original, not an entity, still here.
    for kept in ("glycolysis", "phosphorylates", "phosphate"):
        assert kept in anon.statement.lower() + anon.question.lower(), kept

    anon_simple, _ = anonymize_problem(simple_problem())
    for kept in ("water", "flows", "throughput"):
        assert kept in anon_simple.statement.lower(), kept


def test_longer_terms_are_replaced_first():
    """ "tank" must not eat the prefix of "tank 1" and strand the longer term."""
    problem = NativeProblem(
        id="p",
        statement="tank 1 feeds tank, and tank 1 is upstream.",
        entities=["tank", "tank 1"],
        constraints=[],
        question="Does tank 1 matter more than tank?",
    )
    anon, mapping = anonymize_problem(problem)
    assert find_leaks(anon.statement + anon.question, native_terms(problem)) == []
    # Both terms got their own symbol rather than one swallowing the other.
    assert len(set(mapping)) == 2


def test_surviving_term_raises_rather_than_returning_quietly():
    """A partial substitution must fail loudly.

    A silent one leaves the guarantee believed but false, which is worse than
    not having it: the downstream leak check would be the only defence while
    everyone assumed there were two.
    """
    problem = NativeProblem(
        id="p",
        statement="alpha and beta",
        entities=["alpha"],
        constraints=[],
        question="?",
    )
    anon, _ = anonymize_problem(problem)
    assert "alpha" not in anon.statement

    import reagents.god.anonymize as mod

    original = mod.re.sub
    try:
        mod.re.sub = lambda *a, **k: a[2]  # substitution that does nothing
        with pytest.raises(AnonymizationError, match="survived"):
            anonymize_problem(problem)
    finally:
        mod.re.sub = original


def test_structured_planning_contract_survives_without_native_labels():
    problem = NativeProblem(
        id="native-screen",
        statement="K562 contains target A.",
        entities=["target A"],
        sensitive_terms=["K562"],
        question="Predict target A.",
        inputs={
            "reasoning_contract": {"minimum_broker_calls": 4},
            "study": {"cell": "K562"},
        },
        required_outputs=["prediction for target A"],
        answer_schema={"description": "target A in K562"},
    )
    anon, _ = anonymize_problem(problem)
    assert anon.inputs["reasoning_contract"] == {"minimum_broker_calls": 4}
    visible = str(anon.model_dump())
    assert find_leaks(visible, native_terms(problem)) == []
