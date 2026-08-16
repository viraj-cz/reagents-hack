import pytest

from reagents.contracts import ToolAccess
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
async def test_a_code_running_tool_spawns_without_any_risk_approval():
    """Risk tiers are gone. A code-running tool needs no ceremony to be used.

    There used to be a `RiskTier` on every tool and a gate in `_spawn` that
    failed a demigod before it started unless the operator had listed each
    HIGH tool by id. It bought nothing: the demigod already runs in its own
    sandbox with no repo source, no Modal credentials, and no route to a
    sibling's output. Running Python there is the point of the sandbox, not a
    hazard to be re-approved.

    Isolation is still enforced -- by the sandbox and the source-free image,
    which are mechanisms rather than labels. This asserts only that a tier
    label no longer blocks a spawn.
    """
    problem = toy_problem()
    registry = default_registry()
    registry.register(
        Tool(
            id="reasoning.code_probe",
            namespace="reasoning",
            description="Run isolated Python.",
            parameters_schema={"type": "object"},
            fn=lambda source: source,
        )
    )
    llm = ScriptedLLM.for_toy_pathway()
    god = God(llm, registry=registry)
    spec = toy_domains()[0].model_copy(
        update={"tool_ids": ["reasoning.code_probe", "simplify"]}
    )
    domain_problem, _ = await Transformer(llm).forward(problem, spec)
    envelope = god.build_envelope(spec, domain_problem)

    result = await _spawn(god, envelope, IsolationGuard(native_terms(problem)))

    assert "approval" not in (result.error or ""), result.error
