import pytest

from reagents.contracts import Budget, ToolAccess
from reagents.tools.registry import Tool, ToolPolicyError, ToolRegistry


def test_lease_enforces_max_calls():
    registry = ToolRegistry()
    registry.register(
        Tool(
            id="math.identity",
            namespace="math",
            description="Return a value.",
            parameters_schema={"type": "object"},
            fn=lambda value: value,
        )
    )
    pack = registry.bind(
        ["math.identity"],
        subject_id="demigod-1",
        budget=Budget(max_tool_calls=1),
    )

    assert pack.call("math.identity", value=7) == 7
    with pytest.raises(ToolPolicyError, match="exhausted"):
        pack.call("math.identity", value=8)


def test_write_tool_requires_explicit_write_lease():
    registry = ToolRegistry()
    registry.register(
        Tool(
            id="remote.publish",
            namespace="remote",
            description="Publish an artifact.",
            parameters_schema={"type": "object"},
            fn=lambda: "published",
            access=ToolAccess.WRITE,
        )
    )

    denied = registry.bind(["remote.publish"], subject_id="reader")
    with pytest.raises(ToolPolicyError, match="does not permit write"):
        denied.call("remote.publish")

    allowed = registry.bind(
        ["remote.publish"], subject_id="writer", allow_write=True
    )
    assert allowed.call("remote.publish") == "published"


@pytest.mark.asyncio
async def test_async_executor_runs_through_broker():
    async def executor(arguments):
        return {"doubled": arguments["value"] * 2}

    registry = ToolRegistry()
    registry.register(
        Tool(
            id="remote.double",
            namespace="remote",
            description="Double a value.",
            parameters_schema={"type": "object"},
            executor=executor,
        )
    )
    pack = registry.bind(["remote.double"], subject_id="async-demigod")
    assert await pack.acall("remote.double", value=4) == {"doubled": 8}
