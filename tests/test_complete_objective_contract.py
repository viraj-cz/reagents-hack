from __future__ import annotations

import pytest

from reagents.contracts import ProjectionManifest
from reagents.god.planner import structural_critic
from reagents.god.transformer import (
    IncompleteProjectionError,
    TransformDraft,
    Transformer,
    projection_contract,
)
from reagents.llm.scripted import ScriptedLLM
from reagents.tools.registry import default_registry
from reagents.toy import toy_domains, toy_problem


def test_every_toy_domain_requires_a_complete_candidate_artifact():
    required = {
        "candidate_solution",
        "constraint_results",
        "certificate",
        "conclusion",
    }
    for spec in toy_domains():
        assert required <= set(spec.artifact_schema["required"])
    assert structural_critic(toy_domains(), default_registry()).ok


def test_projection_contract_accounts_for_inputs_constraints_and_outputs():
    problem = toy_problem().model_copy(
        update={
            "inputs": {"table": [{"x": 1}], "parts": {"p": 2}},
            "required_outputs": ["candidate", "proof"],
        }
    )
    contract = projection_contract(problem)
    assert contract.source_ids == [
        "source:statement",
        "source:unit:001",
        "source:unit:002",
    ]
    assert contract.objective_ids == [
        "objective:constraint:01",
        "objective:constraint:02",
        "objective:question",
    ]
    assert contract.output_ids == ["output:01", "output:02"]


@pytest.mark.asyncio
async def test_transform_rejects_a_clean_but_partial_projection():
    problem = toy_problem()
    spec = toy_domains()[0]
    partial = TransformDraft(
        representation={"x": [1, 2, 3]},
        task="Return a candidate in x-space.",
        notation_guide="x is an abstract state.",
        projection_manifest=ProjectionManifest(
            source_ids=["source:statement"],
            objective_ids=["objective:question"],
            output_ids=["output:01"],
        ),
    )
    transformer = Transformer(ScriptedLLM({f"transform:{spec.name}": partial}))
    with pytest.raises(IncompleteProjectionError) as exc:
        await transformer.forward(problem, spec, max_retries=1)
    assert any("objective_ids" in error for error in exc.value.errors)
    assert any("output_ids" in error for error in exc.value.errors)


@pytest.mark.asyncio
async def test_scripted_domains_preserve_the_whole_toy_contract():
    problem = toy_problem()
    transformer = Transformer(ScriptedLLM.for_toy_pathway())
    expected = projection_contract(problem)
    for spec in toy_domains():
        projected, _ = await transformer.forward(problem, spec)
        assert projected.projection_manifest.source_ids == expected.source_ids
        assert projected.projection_manifest.objective_ids == expected.objective_ids
        assert projected.projection_manifest.output_ids == expected.output_ids
        assert set(projected.projection_manifest.source_map) == set(expected.source_ids)
        assert set(projected.projection_manifest.objective_map) == set(
            expected.objective_ids
        )
        assert set(projected.projection_manifest.output_map) == set(expected.output_ids)
