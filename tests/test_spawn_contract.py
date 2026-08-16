"""Pure-logic tests. No Modal account, no network, no API key.

Everything here runs on the runner-independent core: spec validation, the
closed registry, image resolution, and the output contract. If these pass, a
spawn can only fail for reasons that need real infrastructure -- which is the
line we want, because those are the failures worth spending a sandbox on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel, ValidationError

from demigod.images import CATALOG, ImageResolutionError, resolve_image
from demigod.layout import RunLayout
from demigod.registry import RegistryError, all_keys, validate_tool_keys
from demigod.result import (
    ENVELOPE_FIELDS,
    DemiGodResult,
    ResultMissingError,
    result_json_schema,
)
from demigod.spec import DemiGodSpec, Problem


def make_spec(**overrides) -> DemiGodSpec:
    base = {
        "name": "revenue-quant",
        "domain": "quantitative analysis of tabular data",
        "tools": ["pandas"],
        "problem": Problem(
            context="A CSV of monthly revenue.",
            goal="Identify outlier months.",
            success_criteria=["names each outlier", "states a threshold"],
        ),
    }
    return DemiGodSpec(**{**base, **overrides})


# --- the closed set ---------------------------------------------------------


def test_unknown_tool_is_rejected_at_spec_time_not_in_the_sandbox():
    with pytest.raises(ValidationError) as e:
        make_spec(tools=["tensorflow"])
    # The error must name the valid alternatives -- a GOD retrying blind is the
    # whole failure mode this guards against.
    assert "tensorflow" in str(e.value)
    assert "pandas" in str(e.value)


def test_duplicate_tools_rejected():
    with pytest.raises(RegistryError):
        validate_tool_keys(["pandas", "pandas"])


def test_empty_toolset_is_legal():
    assert validate_tool_keys([]) == []


def test_all_registry_docs_exist():
    """A declared doc_file that isn't on disk means a live agent with no idea
    how to use its tools. Catch it here, not there."""
    from demigod.registry import REGISTRY

    for key in all_keys():
        # Raises RegistryError if the declared doc file is missing.
        assert isinstance(REGISTRY[key].usage_doc, str)


# --- naming and path safety -------------------------------------------------


@pytest.mark.parametrize("bad", ["Revenue", "r", "-lead", "has_underscore", "x" * 40])
def test_bad_names_rejected(bad):
    with pytest.raises(ValidationError):
        make_spec(name=bad)


@pytest.mark.parametrize("bad", ["/etc/passwd", "../secrets.csv", "a/../../b"])
def test_files_cannot_escape_shared(bad):
    with pytest.raises(ValidationError):
        make_spec(files=[bad])


# --- image resolution -------------------------------------------------------


def test_resolves_to_smallest_covering_image():
    assert resolve_image([]).name == "demigod-base"
    assert "pandas" in resolve_image(["pandas"]).tool_keys


def test_uncovered_toolset_hard_errors_rather_than_building():
    """The locked decision: never build an image on the fly. A gap in the
    catalog is a human's one-time fix, not a cost every caller pays."""
    fake = "definitely-not-registered"
    with pytest.raises((ImageResolutionError, RegistryError)):
        resolve_image([fake])


def test_every_catalog_image_only_claims_registered_tools():
    for image in CATALOG:
        validate_tool_keys(sorted(image.tool_keys))


# --- volume layout ----------------------------------------------------------


def test_each_demigod_gets_a_private_out_subpath():
    a = RunLayout(run_id="r1", demigod_name="alpha")
    b = RunLayout(run_id="r1", demigod_name="beta")
    # Same run -> same volumes...
    assert a.shared_volume_name == b.shared_volume_name
    assert a.out_volume_name == b.out_volume_name
    # ...but disjoint sub_paths, so neither can reach the other's output.
    assert a.out_subpath != b.out_subpath
    assert a.out_subpath == "alpha"


def test_shared_and_out_are_distinct_volumes():
    """Modal rejects mounting one Volume at two locations in one sandbox:
    'The same Volume cannot be mounted in multiple locations for the same
    function'. Verified live. This test locks in the two-volume fix."""
    layout = RunLayout(run_id="r1", demigod_name="alpha")
    assert layout.shared_volume_name != layout.out_volume_name


# --- the output contract ----------------------------------------------------


def test_result_round_trips(tmp_path):
    r = DemiGodResult(
        claim="March and November are outliers.",
        confidence=0.82,
        evidence=["outliers.csv"],
        method="IQR fence over monthly totals.",
        files=["outliers.csv", "plot.png"],
    )
    r.write(tmp_path)
    back = DemiGodResult.read(tmp_path)
    assert back.claim == r.claim
    assert back.confidence == pytest.approx(0.82)


def test_missing_manifest_is_a_named_error(tmp_path):
    with pytest.raises(ResultMissingError):
        DemiGodResult.read(tmp_path)


def test_confidence_is_bounded():
    with pytest.raises(ValidationError):
        DemiGodResult(claim="x", confidence=1.5, method="y")


def test_failure_manifest_surfaces_error_in_blockers():
    """A consumer reading only the contract fields must still see the failure."""
    r = DemiGodResult.failure(
        demigod_name="alpha",
        domain_name="alpha_domain",
        run_id="r1",
        status="timeout",
        error="exceeded wall clock",
    )
    assert r.status == "timeout"
    assert r.confidence == 0.0
    assert "exceeded wall clock" in r.blockers


def test_success_and_failure_share_one_shape():
    """The consolidation's whole point: reagents returned a union, so every
    consumer had to isinstance-branch. One shape means one parse path."""
    ok = DemiGodResult(claim="x", confidence=0.5, method="m")
    bad = DemiGodResult.failure(status="failed", error="boom")
    assert set(ok.model_dump()) == set(bad.model_dump())
    assert ok.status == "ok" and bad.status == "failed"


def test_payload_validates_against_a_domain_artifact_schema():
    schema = {
        "type": "object",
        "required": ["findings", "conclusion"],
        "properties": {
            "findings": {"type": "array", "items": {"type": "string"}},
            "conclusion": {"type": "string"},
        },
    }
    good = DemiGodResult(
        claim="c",
        confidence=0.9,
        method="m",
        payload={"findings": ["a"], "conclusion": "done"},
    )
    assert good.validate_against(schema) == []

    missing = DemiGodResult(claim="c", confidence=0.9, method="m", payload={})
    assert missing.validate_against(schema)  # names the absent properties

    wrong_type = DemiGodResult(
        claim="c",
        confidence=0.9,
        method="m",
        payload={"findings": "x", "conclusion": 1},
    )
    assert len(wrong_type.validate_against(schema)) == 2


def test_domain_artifact_schema_replaces_the_payload_slot_in_the_prompt():
    """A per-domain output shape rides inside the otherwise fixed contract."""
    domain_schema = {"type": "object", "required": ["flux"]}
    props = result_json_schema(domain_schema)["properties"]
    assert props["payload"] == domain_schema
    # ...and the fixed fields are still there alongside it.
    assert "claim" in props and "unknowns" in props


def test_prompt_schema_hides_runner_owned_envelope():
    """The agent must not think it can set its own status -- otherwise a model
    can self-report ok on a run that crashed."""
    props = result_json_schema()["properties"]
    for envelope in ENVELOPE_FIELDS:
        assert envelope not in props
    for authored in ("claim", "confidence", "evidence", "method", "payload"):
        assert authored in props


# --- end to end (no infrastructure) -----------------------------------------


def test_example_spec_validates_and_resolves():
    """The committed example must always be spawnable. It is what a new
    collaborator runs first."""
    example = Path(__file__).parent.parent / "examples" / "pandas-demigod.json"
    spec = DemiGodSpec.model_validate_json(example.read_text(encoding="utf-8"))
    image = resolve_image(spec.tools)
    assert image.covers(set(spec.tools))


def test_spec_is_json_serializable_for_transport_into_the_sandbox():
    spec = make_spec()
    assert DemiGodSpec.model_validate(json.loads(spec.model_dump_json())) == spec


# --- turn-budget truncation --------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "Claude Code returned an error result: Reached maximum number of turns (6)",
        "reached MAXIMUM NUMBER OF TURNS (12)",
        "max_turns exceeded",
    ],
)
def test_turn_limit_errors_are_recognized(message):
    """The SDK raises a bare Exception carrying the CLI's error string, with no
    typed subclass. Before this was caught, hitting the cap killed the process
    before the manifest was written and a demigod that had done real work
    returned nothing."""
    from demigod.entrypoint import _is_turn_limit

    assert _is_turn_limit(Exception(message))


@pytest.mark.parametrize(
    "message",
    ["CLINotFoundError: Claude Code not found", "connection reset", "401 unauthorized"],
)
def test_other_errors_are_not_swallowed_as_turn_limits(message):
    """Deliberately narrow: any error that is NOT the turn cap must still
    propagate and fail the run loudly."""
    from demigod.entrypoint import _is_turn_limit

    assert not _is_turn_limit(Exception(message))


# --- model-response parsing --------------------------------------------------


def test_balanced_extraction_beats_the_greedy_regex_on_truncation():
    """A truncated reply that ends on a NESTED closing brace is what broke the
    second live run. The old greedy `\\{.*\\}` matched first-{ to last-} and
    returned a fragment whose outer object never closed, surfacing as
    `ValidationError: EOF while parsing an object` -- a symptom that named
    neither truncation nor the field. Balanced scanning returns None instead."""
    from reagents.llm.anthropic_client import _extract_json_object

    truncated = '{"representation": {"a": 1}, "task": "the conserved measure"'
    assert _extract_json_object(truncated) is None


def test_balanced_extraction_finds_a_complete_object_in_prose():
    from reagents.llm.anthropic_client import _extract_json_object

    assert (
        _extract_json_object('Here you go: {"a": {"b": 2}} — hope that helps!')
        == '{"a": {"b": 2}}'
    )


def test_braces_inside_strings_do_not_break_the_depth_count():
    from reagents.llm.anthropic_client import _extract_json_object

    text = '{"note": "use {curly} braces", "n": 1}'
    assert _extract_json_object(text) == text


def test_truncation_is_reported_as_truncation_not_as_a_schema_error():
    """The diagnostic that was missing: stop_reason distinguishes 'raise
    max_tokens' from 'the model wrote something malformed'."""
    from reagents.llm.anthropic_client import _parse_model
    from reagents.llm.client import LLMError

    class Draft(BaseModel):
        a: int

    class FakeMessage:
        stop_reason = "max_tokens"

    with pytest.raises(LLMError) as e:
        _parse_model('{"a": {"b": 1}', Draft, FakeMessage())
    assert "max_tokens" in str(e.value)


def test_complete_but_wrong_shape_is_reported_distinctly():
    from reagents.llm.anthropic_client import _parse_model
    from reagents.llm.client import LLMError

    class Draft(BaseModel):
        a: int

    class FakeMessage:
        stop_reason = "end_turn"

    with pytest.raises(LLMError) as e:
        _parse_model('{"a": "not-an-int"}', Draft, FakeMessage())
    assert "did not match the schema" in str(e.value)


def test_sandbox_runtime_constructs_with_defaults():
    """Regression: the PR #4 merge (81b2198) dropped `restrict_egress` from
    SandboxDemigodRuntime's signature while keeping `self.restrict_egress =
    restrict_egress`, so EVERY instantiation raised NameError -- master's whole
    sandbox path, including e2e_live.py, was dead and no test caught it.

    Constructing the class with nothing but a run_id is the cheapest possible
    guard against a signature/body mismatch."""
    from reagents.demigod.sandbox_runtime import SandboxDemigodRuntime, _Auto

    runtime = SandboxDemigodRuntime(run_id="r1")
    assert runtime.run_id == "r1"
    assert runtime.restrict_egress is False
    # AUTO, not None: a session is resolved when the run starts, so building
    # this object stays offline and cheap. None would mean "no tools, ever".
    assert isinstance(runtime.toolbox, _Auto)

    assert SandboxDemigodRuntime(run_id="r1", toolbox=None).toolbox is None


# --- refusal handling --------------------------------------------------------


def test_a_refusal_is_retried_then_reported_as_a_refusal():
    """A classifier refusal TRUNCATES the response, so if it isn't handled it
    surfaces as incomplete JSON and sends you hunting for a parse bug. Seen live
    on a water-tank flow problem, cut off mid-way through abstract state-update
    equations -- a false positive, which is why one retry is worth it."""
    import asyncio

    from reagents.llm import anthropic_client as ac
    from reagents.llm.client import LLMError

    class Details:
        category = "cyber"
        explanation = "declined"

    class Refused:
        stop_reason = "refusal"
        stop_details = Details()
        content: ClassVar[list] = []

    calls = {"n": 0}

    class FakeMessages:
        async def create(self, **kw):
            calls["n"] += 1
            return Refused()

    llm = ac.AnthropicLLM.__new__(ac.AnthropicLLM)
    llm.model = "m"
    llm._client = type("C", (), {"messages": FakeMessages()})()

    class Draft(BaseModel):
        a: int = 1

    with pytest.raises(LLMError) as e:
        asyncio.run(llm.complete(system="s", user="u", response_model=Draft))

    assert calls["n"] == ac.REFUSAL_RETRIES + 1, "should retry before giving up"
    assert "refused" in str(e.value)
    assert "cyber" in str(e.value), "the category is the actionable part"


def test_a_refusal_that_clears_on_retry_succeeds():
    import asyncio

    from reagents.llm import anthropic_client as ac

    class Block:
        text = '{"a": 7}'

    class Refused:
        stop_reason = "refusal"
        stop_details = None
        content: ClassVar[list] = []

    class Ok:
        stop_reason = "end_turn"
        content: ClassVar[list] = [Block()]

    seq = [Refused(), Ok()]

    class FakeMessages:
        async def create(self, **kw):
            return seq.pop(0)

    llm = ac.AnthropicLLM.__new__(ac.AnthropicLLM)
    llm.model = "m"
    llm._client = type("C", (), {"messages": FakeMessages()})()

    class Draft(BaseModel):
        a: int

    got = asyncio.run(llm.complete(system="s", user="u", response_model=Draft))
    assert got.a == 7
