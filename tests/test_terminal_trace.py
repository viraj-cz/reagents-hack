import io
import json

import pytest

from reagents.god.orchestrator import God
from reagents.isolation import find_leaks, native_terms
from reagents.llm.scripted import ScriptedLLM
from reagents.toy import toy_problem
from reagents.tracing import GOD_LANE, RecordingTracer, TerminalTracer, demigod_lane


@pytest.mark.asyncio
async def test_trace_separates_god_and_each_demigod_stream():
    tracer = RecordingTracer()
    problem = toy_problem()
    god = God(ScriptedLLM.for_toy_pathway(), tracer=tracer)

    await god.solve(problem)

    god_kinds = [record.kind for record in tracer.records if record.lane == GOD_LANE]
    assert {"START", "PLAN", "SPAWN", "COLLECT", "INTEGRATE", "DONE"} <= set(god_kinds)

    for spec in god.last_trace.specs:
        lane = demigod_lane(spec.name)
        records = [record for record in tracer.records if record.lane == lane]
        kinds = [record.kind for record in records]
        assert kinds[0] == "START"
        expected = {
            "SCOPE",
            "MODEL",
            "TOOL CALL",
            "TOOL RESULT",
            "REASON",
            "ARTIFACT",
        }
        assert expected <= set(kinds)
        assert kinds[-1] == "DONE"
        calls = [record.message for record in records if record.kind == "TOOL CALL"]
        assert calls == spec.tool_ids

        # Agent lanes remain sealed even though the God lane names the native problem.
        visible = json.dumps(
            [{"message": record.message, "data": record.data} for record in records],
            default=str,
        )
        assert find_leaks(visible, native_terms(problem)) == []


@pytest.mark.asyncio
async def test_terminal_trace_shows_god_and_readable_subagent_progress():
    stream = io.StringIO()
    tracer = TerminalTracer(stream=stream, color=False)

    god = God(ScriptedLLM.for_toy_pathway(), tracer=tracer)
    await god.solve(toy_problem())

    output = stream.getvalue()
    assert "God is analyzing the problem" in output
    assert "Choosing useful representations" in output
    assert "Selected 3 complete alternative representations" in output
    assert "Running scoped analyses" in output
    assert "Subagent stoichiometric_flow started" in output
    assert "stoichiometric_flow · Scope" in output
    assert "stoichiometric_flow · Allowed tools" in output
    assert "stoichiometric_flow · → Using" in output
    assert "stoichiometric_flow · ←" in output
    assert "stoichiometric_flow · Reasoning summary" in output
    assert "stoichiometric_flow · Finding" in output
    assert "Subagent stoichiometric_flow complete" in output
    assert "God accepted stoichiometric_flow's artifact" in output
    assert "Comparing complete candidates" in output
    assert "Analysis complete" in output
    assert "DEMI:" not in output
    assert "TOOL CALL" not in output
    assert "TOOL RESULT" not in output


def test_terminal_trace_summarizes_subagent_tools_and_redacts_details():
    stream = io.StringIO()
    tracer = TerminalTracer(stream=stream, color=False, max_detail_chars=80)

    tracer.emit(
        demigod_lane("tokens"),
        "TOOL CALL",
        "example.run",
        data={"arguments": {"api_key": "internal-secret"}},
    )
    tracer.emit(
        GOD_LANE,
        "INFO",
        "Checking a bounded detail",
        data={"api_key": "secret-value", "s_tokens": 3, "long": "x" * 200},
    )

    output = stream.getvalue()
    assert "tokens · → Using example.run" in output
    assert "inputs: api_key" in output
    assert "TOOL CALL" not in output
    assert "internal-secret" not in output
    assert "secret-value" not in output
    assert "<redacted>" in output
    assert "s_tokens" in output
    assert "\033[" not in output
    assert len(output.splitlines()) == 2


def test_terminal_trace_can_hide_subagents():
    stream = io.StringIO()
    tracer = TerminalTracer(stream=stream, color=False, show_subagents=False)

    tracer.emit(demigod_lane("tokens"), "START", "opened")
    tracer.emit(GOD_LANE, "INFO", "God still reports progress")

    output = stream.getvalue()
    assert "Subagent" not in output
    assert "God still reports progress" in output
