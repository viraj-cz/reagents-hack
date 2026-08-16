import pytest

from reagents.contracts import DemigodFailure, RiskTier, ToolAccess
from reagents.demigod.runtime import IsolationGuard
from reagents.god.orchestrator import God, _spawn
from reagents.god.transformer import Transformer
from reagents.isolation import native_terms
from reagents.llm.scripted import ScriptedLLM
from reagents.tools.registry import Tool, default_registry
from reagents.toy import toy_domains, toy_problem


@pytest.mark.asyncio
async def test_god_does_not_implicitly_grant_write_tools():
    registry = default_registry()
    registry.register(
        Tool(
            id="remote.publish",
            namespace="remote",
            description="Publish remotely.",
            parameters_schema={"type": "object"},
            fn=lambda: "ok",
            access=ToolAccess.WRITE,
        )
    )
    # The scripted planner does not select this tool; the assertion captures the
    # operator-side policy default independently of model behavior.
    god = God(ScriptedLLM.for_toy_pathway(), registry=registry)
    await god.solve(toy_problem())
    assert god.approved_write_tools == frozenset()


@pytest.mark.asyncio
async def test_spawn_requires_exact_high_risk_approval():
    problem = toy_problem()
    registry = default_registry()
    registry.register(
        Tool(
            id="reasoning.python",
            namespace="reasoning",
            description="Run isolated Python.",
            parameters_schema={"type": "object"},
            fn=lambda source: source,
            risk_tier=RiskTier.HIGH,
        )
    )
    llm = ScriptedLLM.for_toy_pathway()
    god = God(llm, registry=registry)
    spec = toy_domains()[0].model_copy(
        update={"tool_ids": ["reasoning.python", "simplify"]}
    )
    domain_problem, _ = await Transformer(llm).forward(problem, spec)
    envelope = god.build_envelope(spec, domain_problem)

    result = await _spawn(god, envelope, IsolationGuard(native_terms(problem)))

    assert isinstance(result, DemigodFailure)
    assert "high-risk tools require operator approval" in result.reason
