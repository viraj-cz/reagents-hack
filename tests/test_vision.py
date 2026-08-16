"""The image path, end to end, offline.

`vision.read_image` is the one capability in this system that cannot be checked
by running it in CI -- it costs money and needs someone's OpenAI account. So
everything AROUND the HTTP call is pinned here instead: what gets refused before
a lease call is spent, how a Responses payload is read, and the wiring that
decides whether an agent holding an image is told to look at it.

The one thing these tests do not prove is that the request body is a body the
API accepts, and no stub can: a renamed model, an unfunded account and a changed
response shape all look identical from here. That check is live and lives in
`scripts/preflight_vision.py`.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from conftest import request

from broker.grants import InMemoryGrantStore, mint_grant
from broker.router import ToolboxRouter
from demigod.prompt import build_system_prompt
from demigod.spec import DemiGodSpec, Problem
from demigod.toolbox.protocol import ToolboxGrant, call_path, describe_path
from reagents.contracts import Budget, ToolAccess, ToolProvider
from reagents.tools import vision
from reagents.tools.registry import ToolExecutionError, default_registry

PNG_1PX = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d494844520000000100000001080600000"
        "01f15c4890000000a49444154789c6300010000050001"
        "0d0a2db40000000049454e44ae426082"
    )
).decode()


# --- registration ------------------------------------------------------------


def test_vision_tool_is_in_the_default_catalog():
    """Ungated. God plans against this catalog and the broker executes against
    it; if the two disagree, God binds a tool the broker will not admit."""
    registry = default_registry()
    assert vision.TOOL_ID in registry.ids()


def test_vision_tool_runs_inline_and_needs_no_write_grant():
    """LOCAL keeps it in the router (where the credential is), and READ means a
    demigod does not need `allow_write` to look at a picture."""
    tool = default_registry().get(vision.TOOL_ID)
    assert tool.provider is ToolProvider.LOCAL
    assert tool.access is ToolAccess.READ

    from broker.dispatch import DispatchPolicy

    assert DispatchPolicy().is_inline(tool)


def test_vision_tool_has_no_tier_because_it_needs_none():
    """A LOCAL tool never reaches `_dispatch_remote`, so the `vision` namespace
    owning no ExecutorClass is correct rather than an omission. Pinned because
    the obvious 'fix' for a namespace with no tier is to add one."""
    from broker.service import class_for

    assert class_for(vision.TOOL_ID) is None


# --- refusals that happen before the lease call is spent ---------------------


def _call(**arguments):
    return asyncio.run(vision.read_image(arguments))


def test_question_is_required():
    with pytest.raises(ToolExecutionError, match="`question` is required"):
        _call(image_base64=PNG_1PX)


@pytest.mark.parametrize(
    "arguments",
    [
        {"question": "q"},
        {"question": "q", "image_base64": PNG_1PX, "image_url": "https://x/y.png"},
    ],
    ids=["neither", "both"],
)
def test_exactly_one_image_source(arguments):
    with pytest.raises(ToolExecutionError, match="exactly one of"):
        _call(**arguments)


def test_tiff_is_refused_by_name():
    """The format microscopy actually arrives in is the one the API rejects, so
    the error has to say so here rather than as an opaque 400."""
    with pytest.raises(ToolExecutionError, match="convert TIFF"):
        _call(question="q", image_base64=PNG_1PX, media_type="image/tiff")


def test_malformed_base64_is_named_as_such():
    with pytest.raises(ToolExecutionError, match="not valid base64"):
        _call(question="q", image_base64="not base64 at all!!")


def test_oversized_image_is_refused_with_the_reason():
    """The broker caps a body at 4 MiB and base64 costs a third on top, so this
    would otherwise fail as a truncated request rather than as a size problem."""
    huge = base64.b64encode(b"\x00" * (vision.MAX_IMAGE_BYTES + 1)).decode()
    with pytest.raises(ToolExecutionError, match="Downscale or crop"):
        _call(question="q", image_base64=huge)


def test_data_url_prefix_is_not_nested_twice():
    """Agents paste `data:image/png;base64,...` in. Accepting it beats a 400."""
    url = vision._data_url(f"data:image/png;base64,{PNG_1PX}", "image/png")
    assert url == f"data:image/png;base64,{PNG_1PX}"
    assert url.count("data:") == 1


def test_base64_whitespace_is_tolerated():
    """`base64` without -w0 wraps at 76 columns. That is not a bad image."""
    wrapped = "\n".join(PNG_1PX[i : i + 8] for i in range(0, len(PNG_1PX), 8))
    assert vision._data_url(wrapped, "image/png").endswith(PNG_1PX)


def test_missing_credential_names_the_process_that_needs_it(monkeypatch):
    """The key belongs to the broker, not the caller and not the sandbox --
    which is the whole reason this tool is brokered, so the error says it."""
    monkeypatch.delenv(vision.API_KEY_ENV, raising=False)
    with pytest.raises(vision.VisionConfigurationError, match="TOOLBOX_BROKER"):
        _call(question="q", image_base64=PNG_1PX)


# --- reading the response ----------------------------------------------------


def _responses_payload(*, output, status="completed", incomplete=None):
    return {
        "model": "gpt-5.5",
        "status": status,
        "incomplete_details": incomplete,
        "output": output,
        "usage": {"input_tokens": 800, "output_tokens": 40},
    }


def _message(text):
    return {"type": "message", "content": [{"type": "output_text", "text": text}]}


def test_reasoning_items_are_skipped(monkeypatch):
    """A reasoning model's `output[0]` is a reasoning item with no text in it.
    Taking `output[0]` would return nothing on every successful call."""
    payload = _responses_payload(
        output=[{"type": "reasoning", "summary": []}, _message("Row 3 is empty.")]
    )
    monkeypatch.setattr(vision, "_post", lambda body, timeout: payload)
    monkeypatch.setenv(vision.API_KEY_ENV, "sk-test")

    result = _call(question="which row", image_base64=PNG_1PX)
    assert result["reading"] == "Row 3 is empty."
    assert result["truncated"] is False
    assert result["usage"] == {"input_tokens": 800, "output_tokens": 40}


def test_truncated_answer_is_flagged_not_hidden(monkeypatch):
    """A cut-off reading quoted as a complete observation is exactly the kind of
    thing that poisons the recombination."""
    payload = _responses_payload(
        output=[_message("Wells A1 through A4 are conf")],
        status="incomplete",
        incomplete={"reason": "max_output_tokens"},
    )
    monkeypatch.setattr(vision, "_post", lambda body, timeout: payload)
    monkeypatch.setenv(vision.API_KEY_ENV, "sk-test")

    assert _call(question="q", image_base64=PNG_1PX)["truncated"] is True


def test_empty_answer_says_why(monkeypatch):
    """All the tokens went to reasoning. 'No result' is not actionable; the
    reason and the knob to turn are."""
    payload = _responses_payload(
        output=[{"type": "reasoning", "summary": []}],
        status="incomplete",
        incomplete={"reason": "max_output_tokens"},
    )
    monkeypatch.setattr(vision, "_post", lambda body, timeout: payload)
    monkeypatch.setenv(vision.API_KEY_ENV, "sk-test")

    with pytest.raises(ToolExecutionError, match="max_output_tokens"):
        _call(question="q", image_base64=PNG_1PX)


def test_request_body_shape(monkeypatch):
    captured: dict = {}

    def fake_post(body, timeout):
        captured.update(body)
        return _responses_payload(output=[_message("ok")])

    monkeypatch.setattr(vision, "_post", fake_post)
    monkeypatch.setenv(vision.API_KEY_ENV, "sk-test")
    _call(question="how many wells", image_base64=PNG_1PX, detail="high")

    assert captured["model"] == vision.DEFAULT_MODEL
    assert captured["max_output_tokens"] == vision.DEFAULT_MAX_OUTPUT_TOKENS
    parts = captured["input"][0]["content"]
    assert parts[0] == {"type": "input_text", "text": "how many wells"}
    assert parts[1]["type"] == "input_image"
    assert parts[1]["detail"] == "high"
    assert parts[1]["image_url"].startswith("data:image/png;base64,")
    # Serializable: the router puts this on the wire.
    json.dumps(captured)


def test_model_is_overridable_by_env(monkeypatch):
    monkeypatch.setenv(vision.MODEL_ENV, "gpt-5.4-mini")
    monkeypatch.setenv(vision.API_KEY_ENV, "sk-test")
    captured: dict = {}
    monkeypatch.setattr(
        vision,
        "_post",
        lambda body, timeout: (
            captured.update(body) or _responses_payload(output=[_message("ok")])
        ),
    )
    _call(question="q", image_base64=PNG_1PX)
    assert captured["model"] == "gpt-5.4-mini"


# --- the prompt rule ---------------------------------------------------------


def _spec(*, tools, granted_ids=None) -> DemiGodSpec:
    return DemiGodSpec(
        name="img-dg",
        domain_name="img",
        domain="a lattice of intensity fields",
        tools=tools,
        toolbox=(
            None
            if granted_ids is None
            else ToolboxGrant(
                url="https://broker.example", lease_id="lease_x", tool_ids=granted_ids
            )
        ),
        problem=Problem(context="c", goal="g", success_criteria=[]),
    )


def test_agent_with_the_vision_tool_is_told_to_use_it():
    prompt = build_system_prompt(
        _spec(tools=["pandas", "imaging"], granted_ids=[vision.TOOL_ID])
    )
    assert "Anything with an image in it goes through" in prompt
    assert vision.TOOL_ID in prompt
    assert "Never describe an image you have not sent" in prompt


def test_agent_that_can_only_measure_is_told_where_that_stops():
    prompt = build_system_prompt(_spec(tools=["imaging"]))
    assert "cannot see them" in prompt
    assert "Anything with an image in it goes through" not in prompt


def test_agent_with_no_image_work_gets_neither_paragraph():
    """The rule is permanent context in every turn. A demigod reasoning about a
    rewrite system should not be carrying it."""
    prompt = build_system_prompt(_spec(tools=[]))
    assert "cannot see them" not in prompt
    assert "Anything with an image in it" not in prompt


# --- the round trip an agent actually makes ---------------------------------


@pytest.fixture
def vision_router():
    """A broker serving the real catalog, with a lease over the vision tool.

    `default_registry()` on purpose, unlike the rest of the broker suite: the
    point of this fixture is that the tool God binds is the tool the router
    serves, so a stub registry would test nothing.
    """
    registry = default_registry()
    store = InMemoryGrantStore()
    lease = registry.mint_lease(
        [vision.TOOL_ID],
        subject_id="img-dg",
        budget=Budget(max_tool_calls=2, wall_time_s=300.0),
    )
    store.publish(mint_grant(lease, label="colony-morphometry"))
    return ToolboxRouter(registry=registry, store=store), lease.lease_id


def test_describe_reaches_the_agent_without_a_network_call(vision_router):
    """`toolbox describe` is how the agent learns the schema -- the CLI shows a
    paragraph and fetches this on demand, so it has to be complete on its own."""
    router, lease_id = vision_router
    response = request(router, "GET", describe_path(vision.TOOL_ID), token=lease_id)
    assert response.status == 200
    schema = response.payload["tool"]["input_schema"]
    assert schema["required"] == ["question"]
    assert "image_base64" in schema["properties"]


def test_a_leased_call_goes_all_the_way_through(vision_router, monkeypatch):
    """Lease check, schema validation, inline dispatch, result serialization.
    Everything in `toolbox call vision.read_image` except the HTTP hop."""
    router, lease_id = vision_router
    monkeypatch.setenv(vision.API_KEY_ENV, "sk-test")
    monkeypatch.setattr(
        vision,
        "_post",
        lambda body, timeout: _responses_payload(output=[_message("Row 3 is empty.")]),
    )

    response = request(
        router,
        "POST",
        call_path(vision.TOOL_ID),
        token=lease_id,
        body={
            "input": {
                "question": "which row is empty",
                "image_base64": PNG_1PX,
                "media_type": "image/png",
            }
        },
    )
    assert response.status == 200, response.payload
    assert response.payload["ok"] is True
    assert response.payload["result"]["reading"] == "Row 3 is empty."
    assert response.payload["lease"]["calls_used"] == 1


def test_a_refusal_costs_a_call_and_says_what_was_wrong(vision_router, monkeypatch):
    """The lease is metered, so a bad argument is expensive. It has to come back
    as something the agent can correct on the next attempt."""
    router, lease_id = vision_router
    monkeypatch.setenv(vision.API_KEY_ENV, "sk-test")

    response = request(
        router,
        "POST",
        call_path(vision.TOOL_ID),
        token=lease_id,
        body={"input": {"question": "which row is empty"}},  # no image
    )
    assert response.payload["ok"] is False
    assert "exactly one of" in json.dumps(response.payload)


def test_prompt_tool_id_matches_the_registered_one():
    """Two packages naming the same wire string, and `demigod` cannot import
    `reagents` to share the constant. So the duplication is checked instead."""
    from demigod.prompt import VISION_TOOL_ID

    assert VISION_TOOL_ID == vision.TOOL_ID
