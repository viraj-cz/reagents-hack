"""The seam: a lease minted by GOD ends up as a command an agent can run.

No Modal here either -- `spawn_demigod` is replaced, and the broker runs
in-process behind an `InMemoryGrantStore`. What is being checked is the wiring
between the two halves, which is precisely the thing that used to be a TODO in
`adapter.py` and an empty list in `DemiGodResult.tool_trace`.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from broker.grants import TraceEntry
from broker.session import local_session
from demigod.egress import allowlist, broker_host
from demigod.prompt import build_system_prompt
from demigod.result import DemiGodResult
from demigod.spec import DemiGodSpec, Problem
from demigod.toolbox.protocol import ToolboxGrant
from reagents.contracts import (
    Axis,
    Budget,
    ContextEnvelope,
    DomainProblem,
    DomainSpec,
)
from reagents.demigod.adapter import envelope_to_spec
from reagents.demigod.sandbox_runtime import SandboxDemigodRuntime
from reagents.tools.registry import Tool, ToolRegistry

SCHEMA = {"type": "object", "required": ["answer"], "properties": {"answer": {}}}


def make_envelope(tool_ids=("formal.z3_solve", "graph.build")) -> ContextEnvelope:
    domain = DomainSpec(
        name="stoichiometric_flow",
        axes=[Axis.CONSERVATION],
        language="hypergraph over token multisets",
        transform_prompt="SECRET",
        tool_ids=list(tool_ids),
        artifact_schema=SCHEMA,
    )
    return ContextEnvelope(
        domain=domain,
        problem=DomainProblem(
            domain_name=domain.name,
            representation={"nodes": ["t1"]},
            task="Decide conservation.",
            notation_guide="t<i> is a token.",
        ),
        tools=[],
        artifact_schema=SCHEMA,
        budget=Budget(max_steps=4, wall_time_s=60.0),
    )


def make_grant(*tool_ids: str) -> ToolboxGrant:
    return ToolboxGrant(
        url="https://broker.example/",
        lease_id="lease_deadbeef",
        tool_ids=list(tool_ids),
    )


# --- adapter -----------------------------------------------------------------


def test_brokered_tools_are_no_longer_reported_as_unavailable():
    """THE BUG THIS FIXES. Before the broker, every reagents tool id was
    unmapped, so every demigod was told its whole toolset was unreachable and
    reasoned with nothing."""
    spec = envelope_to_spec(
        make_envelope(), toolbox=make_grant("formal.z3_solve", "graph.build")
    )
    assert "unavailable_tools" not in spec.miscellaneous
    assert spec.toolbox is not None
    assert spec.toolbox.tool_ids == ["formal.z3_solve", "graph.build"]


def test_the_two_tool_paths_are_orthogonal():
    """`tools` names pip packages baked into the image; `toolbox` names
    callables on someone else's machine. A demigod can hold both, and they do
    not collide."""
    spec = envelope_to_spec(
        make_envelope(tool_ids=["data.frames", "formal.z3_solve"]),
        tool_map={"data.frames": "pandas"},
        toolbox=make_grant("formal.z3_solve"),
    )
    assert spec.tools == ["pandas"]
    assert spec.toolbox.tool_ids == ["formal.z3_solve"]
    assert "unavailable_tools" not in spec.miscellaneous


def test_a_tool_in_neither_path_is_still_reported_loudly():
    """Unchanged behaviour, and it must stay unchanged: a silently-dropped tool
    produces an agent that invents results it had no way to compute."""
    spec = envelope_to_spec(make_envelope(), toolbox=make_grant("formal.z3_solve"))
    assert spec.miscellaneous["unavailable_tools"] == ["graph.build"]
    assert "do not simulate" in spec.miscellaneous["unavailable_tools_note"].lower()


def test_the_grant_does_not_smuggle_the_transform_prompt_through():
    """Every new field on the spec is a new way for GOD's own instructions to
    reach the agent it constrains."""
    spec = envelope_to_spec(make_envelope(), toolbox=make_grant("formal.z3_solve"))
    assert "SECRET" not in spec.model_dump_json()


def test_a_spec_with_a_grant_still_validates_and_round_trips():
    spec = envelope_to_spec(make_envelope(), toolbox=make_grant("formal.z3_solve"))
    echoed = DemiGodSpec.model_validate_json(spec.model_dump_json())
    assert echoed.toolbox.lease_id == "lease_deadbeef"


# --- egress ------------------------------------------------------------------


def test_egress_is_unrestricted_unless_asked_for():
    """Turning a prompt instruction into a firewall rule changes behaviour.
    Default-off means nobody inherits it by surprise mid-hackathon."""
    assert envelope_to_spec(make_envelope()).egress_domains is None


def test_restricted_egress_pins_the_broker_and_the_agent_api():
    spec = envelope_to_spec(
        make_envelope(),
        toolbox=make_grant("formal.z3_solve"),
        restrict_egress=True,
    )
    assert "broker.example" in spec.egress_domains
    assert "api.anthropic.com" in spec.egress_domains


def test_package_registries_are_never_on_the_allowlist():
    """A demigod that can pip install can install its way out of the domain it
    was given. The images are pre-baked so it never needs to."""
    domains = allowlist("https://broker.example")
    assert not any("pypi" in d or "npm" in d or "github" in d for d in domains)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://x--router.modal.run", "x--router.modal.run"),
        ("http://127.0.0.1:8000", "127.0.0.1"),
        ("broker.example", "broker.example"),
    ],
)
def test_broker_host_strips_scheme_and_port(url, expected):
    """Modal's allowlist is a domain list. A `host:port` entry silently matches
    nothing."""
    assert broker_host(url) == expected


# --- the prompt the agent actually reads -------------------------------------


def test_the_prompt_teaches_the_cli_not_a_json_schema():
    """The locked decision, pinned. Injecting `parameters_schema` per tool would
    put a JSON Schema in every turn of the context window; `toolbox describe`
    fetches one at the moment it is needed."""
    spec = envelope_to_spec(make_envelope(), toolbox=make_grant("formal.z3_solve"))
    prompt = build_system_prompt(spec)

    assert "toolbox list" in prompt
    assert "toolbox describe formal.z3_solve" in prompt
    assert "toolbox call formal.z3_solve" in prompt
    assert "formal.z3_solve" in prompt


def test_the_prompt_tells_the_agent_its_calls_are_metered():
    spec = envelope_to_spec(make_envelope(), toolbox=make_grant("formal.z3_solve"))
    prompt = build_system_prompt(spec).lower()
    assert "metered" in prompt
    assert "blockers" in prompt
    assert "invent" in prompt or "fabricat" in prompt


def test_a_demigod_with_no_grant_sees_no_toolbox_section():
    """The section must be absent, not empty: an agent told about a `toolbox`
    command it has no lease for will spend turns discovering that."""
    prompt = build_system_prompt(envelope_to_spec(make_envelope()))
    assert "toolbox" not in prompt.lower()


def test_no_tools_installed_plus_a_grant_does_not_contradict_itself():
    """The old branch said "No domain tools" and then described `toolbox` two
    lines later. An agent that believes the first sentence never runs the
    command."""
    spec = envelope_to_spec(make_envelope(), toolbox=make_grant("formal.z3_solve"))
    assert spec.tools == []
    assert "No domain tools" not in build_system_prompt(spec)


def test_an_empty_grant_is_surfaced_rather_than_silently_dropped():
    spec = DemiGodSpec(
        name="d-1",
        domain="d",
        problem=Problem(context="c", goal="g"),
        toolbox=ToolboxGrant(url="https://b", lease_id="lease_x", tool_ids=[]),
    )
    assert "grants no tools" in build_system_prompt(spec)


# --- the runtime: grant -> spawn -> trace -> revoke ---------------------------


class FakeRuntimeEnv:
    """Stands in for Modal. Captures the spec and answers with a manifest."""

    def __init__(self) -> None:
        self.spec: DemiGodSpec | None = None

    def spawn(self, spec, *, run_id, runner_kind):
        self.spec = spec
        return DemiGodResult(
            claim="c", confidence=0.8, method="m", payload={"answer": 1}
        )


def build_runtime(monkeypatch, session, env: FakeRuntimeEnv, *, require_toolbox=True):
    monkeypatch.setattr("reagents.demigod.sandbox_runtime.spawn_demigod", env.spawn)
    return SandboxDemigodRuntime(
        run_id="r1", toolbox=session, require_toolbox=require_toolbox
    )


def bind(registry: ToolRegistry, ids: list[str]):
    return registry.bind(ids, subject_id="stoichiometric_flow")


@pytest.fixture
def pack_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool_id in ("formal.z3_solve", "graph.build"):
        registry.register(
            Tool(
                id=tool_id,
                description="d",
                parameters_schema={"type": "object"},
                fn=lambda: None,
            )
        )
    return registry


def test_the_runtime_publishes_a_lease_and_hands_it_to_the_sandbox(
    monkeypatch, pack_registry
):
    session = local_session("https://broker.example")
    env = FakeRuntimeEnv()
    runtime = build_runtime(monkeypatch, session, env)
    pack = bind(pack_registry, ["formal.z3_solve", "graph.build"])

    asyncio.run(runtime.run(make_envelope(), pack))

    grant = env.spec.toolbox
    assert grant is not None
    # THE credential identity: the lease that gated the spawn is the token.
    assert grant.lease_id == pack.lease.lease_id
    assert sorted(grant.tool_ids) == ["formal.z3_solve", "graph.build"]


def test_the_broker_authors_the_tool_trace(monkeypatch, pack_registry):
    """`DemiGodResult.tool_trace` has been an empty list since the contract was
    written. It is now filled from the broker's own record -- the thing being
    audited cannot under-report it."""
    session = local_session()
    env = FakeRuntimeEnv()
    runtime = build_runtime(monkeypatch, session, env)
    pack = bind(pack_registry, ["formal.z3_solve", "graph.build"])

    def spawn_and_call(spec, *, run_id, runner_kind):
        # Simulate the agent making one real call and one refused one.
        lease = spec.toolbox.lease_id
        session.store.record(
            TraceEntry(
                tool="formal.z3_solve",
                input={"smt2": "(check-sat)"},
                result={"sat": True},
            ),
            lease_id=lease,
        )
        session.store.record(
            TraceEntry(
                tool="not.granted",
                ok=False,
                metered=False,
                error={"code": "unbound_tool", "message": "no"},
            ),
            lease_id=lease,
        )
        return DemiGodResult(
            claim="c", confidence=0.8, method="m", payload={"answer": 1}
        )

    monkeypatch.setattr(
        "reagents.demigod.sandbox_runtime.spawn_demigod", spawn_and_call
    )

    result = asyncio.run(runtime.run(make_envelope(), pack))

    assert [row["tool"] for row in result.tool_trace] == [
        "formal.z3_solve",
        "not.granted",
    ]
    assert result.tool_trace[0]["result"] == {"sat": True}
    assert result.tool_trace[1]["metered"] is False
    # It must survive serialization -- the manifest is written as JSON.
    json.dumps(result.model_dump(mode="json"))


def test_runtime_rejects_artifact_below_schema_tool_call_minimum(
    monkeypatch, pack_registry
):
    session = local_session()
    envelope = make_envelope()
    envelope.artifact_schema = {**SCHEMA, "x-min-tool-calls": 4}

    def spawn_with_three_calls(spec, *, run_id, runner_kind):
        for index in range(3):
            session.store.record(
                TraceEntry(
                    tool="graph.build",
                    input={"index": index},
                    result={"ok": True},
                ),
                lease_id=spec.toolbox.lease_id,
            )
        return DemiGodResult(
            claim="c", confidence=0.8, method="m", payload={"answer": 1}
        )

    monkeypatch.setattr(
        "reagents.demigod.sandbox_runtime.spawn_demigod", spawn_with_three_calls
    )
    runtime = SandboxDemigodRuntime(run_id="r1", toolbox=session)
    result = asyncio.run(
        runtime.run(envelope, bind(pack_registry, ["formal.z3_solve", "graph.build"]))
    )
    assert result.status == "failed"
    assert "at least 4 brokered tool calls; observed 3" in (result.error or "")


def test_the_lease_is_revoked_even_when_the_spawn_explodes(monkeypatch, pack_registry):
    """A lease that outlives its demigod is a credential lying around."""
    session = local_session()

    def boom(spec, *, run_id, runner_kind):
        raise RuntimeError("modal said no")

    monkeypatch.setattr("reagents.demigod.sandbox_runtime.spawn_demigod", boom)
    runtime = SandboxDemigodRuntime(run_id="r1", toolbox=session)
    pack = bind(pack_registry, ["formal.z3_solve", "graph.build"])

    result = asyncio.run(runtime.run(make_envelope(), pack))

    assert result.status == "failed"
    assert session.store.read(pack.lease.lease_id) is None


def test_a_broker_outage_does_not_stop_the_demigod(monkeypatch, pack_registry):
    """A demigod with no tools and an honest `blockers` entry is worth more than
    no demigod at all."""

    class DeadSession:
        def grant(self, pack, *, label=""):
            raise RuntimeError("broker unreachable")

        def collect_trace(self, lease_id, *, include_refused=True):
            raise AssertionError("never reached")

        def revoke(self, lease_id):
            raise AssertionError("never reached")

    env = FakeRuntimeEnv()
    # Explicit: degrading is still supported and still tested, it is simply no
    # longer what a caller gets by not deciding. Its opposite is
    # `test_required_broker_outage_fails_closed`, which is now the default.
    runtime = build_runtime(monkeypatch, DeadSession(), env, require_toolbox=False)

    result = asyncio.run(
        runtime.run(make_envelope(), bind(pack_registry, ["formal.z3_solve"]))
    )

    assert result.status == "ok"
    assert env.spec.toolbox is None
    # ...and the agent was told, rather than left to guess.
    assert "formal.z3_solve" in env.spec.miscellaneous["unavailable_tools"]


def test_required_broker_outage_fails_closed(monkeypatch, pack_registry):
    class DeadSession:
        def grant(self, pack, *, label=""):
            raise RuntimeError("broker unreachable")

    env = FakeRuntimeEnv()
    monkeypatch.setattr("reagents.demigod.sandbox_runtime.spawn_demigod", env.spawn)
    runtime = SandboxDemigodRuntime(
        run_id="r1", toolbox=DeadSession(), require_toolbox=True
    )
    result = asyncio.run(
        runtime.run(make_envelope(), bind(pack_registry, ["formal.z3_solve"]))
    )
    assert result.status == "failed"
    assert "required Broker lease publication failed" in (result.error or "")
    assert env.spec is None


def test_runtime_pins_the_requested_agent_model(monkeypatch, pack_registry):
    session = local_session()
    env = FakeRuntimeEnv()
    monkeypatch.setattr("reagents.demigod.sandbox_runtime.spawn_demigod", env.spawn)
    runtime = SandboxDemigodRuntime(
        run_id="r1", toolbox=session, agent_model="claude-opus-4-8"
    )
    asyncio.run(runtime.run(make_envelope(), bind(pack_registry, ["graph.build"])))
    assert env.spec is not None
    assert env.spec.model == "claude-opus-4-8"


def test_without_a_toolbox_the_runtime_behaves_exactly_as_before(
    monkeypatch, pack_registry
):
    """`toolbox=None` is now EXPLICIT. The default is AUTO.

    This used to construct with no argument, which is the same thing the
    production callers did -- and why runs reported zero brokered tool calls
    with a whole toolbox deployed. Opting out is still supported; it just has
    to be said.
    """
    env = FakeRuntimeEnv()
    monkeypatch.setattr("reagents.demigod.sandbox_runtime.spawn_demigod", env.spawn)
    runtime = SandboxDemigodRuntime(run_id="r1", toolbox=None, require_toolbox=False)

    result = asyncio.run(
        runtime.run(make_envelope(), bind(pack_registry, ["formal.z3_solve"]))
    )

    assert env.spec.toolbox is None
    assert env.spec.egress_domains is None
    assert result.tool_trace == []


def test_toolbox_is_on_by_default(monkeypatch, pack_registry):
    """AUTO, not None. A demigod that can reach its tools should get them.

    The old default was None, so every caller that did not pass `toolbox=`
    silently ran with no brokered tools -- which is exactly what happened live:
    a run reported `brokered tool calls: 0` with the whole toolbox deployed and
    healthy, because the default said so.
    """
    from reagents.demigod.sandbox_runtime import AUTO, SandboxDemigodRuntime

    assert SandboxDemigodRuntime(run_id="r1").toolbox is AUTO
    assert SandboxDemigodRuntime(run_id="r1", toolbox=None).toolbox is None


def test_an_unreachable_broker_degrades_instead_of_failing_the_run(
    monkeypatch, pack_registry
):
    """No broker deployed must cost the tools, not the demigod.

    Same reasoning as a failed `grant`: an artifact with an honest note about
    missing tools is worth more than no artifact.
    """
    import reagents.demigod.sandbox_runtime as mod

    env = FakeRuntimeEnv()
    monkeypatch.setattr(mod, "spawn_demigod", env.spawn)

    def explode() -> None:
        raise RuntimeError("no deployed router")

    monkeypatch.setattr(
        "broker.session.modal_session", lambda *a, **k: explode(), raising=False
    )

    runtime = SandboxDemigodRuntime(run_id="r1", require_toolbox=False)  # AUTO
    result = asyncio.run(
        runtime.run(make_envelope(), bind(pack_registry, ["formal.z3_solve"]))
    )

    assert runtime.toolbox is None, "an unreachable broker must resolve to None"
    assert result.status == "ok", "the run died instead of degrading"
    assert env.spec.toolbox is None
