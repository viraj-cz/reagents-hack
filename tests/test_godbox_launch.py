"""The request contract, the layout, and the CLI -- offline.

No Modal account and no network: everything here is either pure logic or a
parse. What is deliberately NOT covered is anything that needs a live
container; `scripts/preflight_god.py` is where those live, because they cannot
be faked usefully.
"""

from __future__ import annotations

import json

import pytest

from godbox.cli import _fmt_duration, _liveness, _render, build_parser
from godbox.layout import (
    ARTIFACT_PATH,
    GOD_OUT_SUBDIR,
    REQUEST_PATH,
    STAGING_DIR,
    GodRequest,
    out_volume_name,
)
from godbox.status import GodStatus
from reagents.toy import simple_problem


def _request(**overrides) -> GodRequest:
    base = {"run_id": "r1", "problem": simple_problem()}
    return GodRequest(**{**base, **overrides})


# --- the request -------------------------------------------------------------


def test_request_round_trips_through_json():
    """It reaches the sandbox as a file, so serialization IS the interface."""
    original = _request(domain_count=3, max_turns=6)
    restored = GodRequest.model_validate_json(original.model_dump_json())
    assert restored.problem.id == original.problem.id
    assert restored.problem.statement == original.problem.statement
    assert restored.domain_count == 3
    assert restored.max_turns == 6


def test_approvals_default_to_empty():
    """Write and high-risk tools require an operator decision. Defaulting to
    anything else would let a planner grant itself authority."""
    request = _request()
    assert request.approved_write_tools == []
    assert request.approved_high_risk_tools == []


def test_sandbox_id_is_stamped_by_the_launcher_not_the_caller():
    """GOD needs its own sandbox id to self-terminate, and a container has no
    reliable way to learn it. The launcher fills it in between create and
    exec."""
    request = _request()
    assert request.sandbox_id == ""
    stamped = request.model_copy(update={"sandbox_id": "sb-123"})
    assert stamped.sandbox_id == "sb-123"


# --- layout ------------------------------------------------------------------


def test_god_writes_to_the_same_volume_the_demigods_do():
    """Derived from demigod's own layout rather than re-spelled. A divergence
    would put the solution on a volume no demigod ever wrote to, which reads as
    every demigod having failed."""
    from demigod.layout import RunLayout

    layout = RunLayout(run_id="r1", demigod_name="x")
    assert out_volume_name("r1") == layout.out_volume_name


def test_gods_output_dir_cannot_collide_with_a_demigod():
    """Demigod names must be lowercase slugs starting with a letter, which
    forbids a leading underscore. That is what makes `_god/` safe to write into
    a volume every demigod also writes into -- GOD's directory is a name no
    demigod can ever be given.

    Asserted by construction rather than by re-spelling the regex, so it keeps
    holding if demigod's validator is rewritten."""
    from demigod.spec import DemiGodSpec, Problem

    with pytest.raises(ValueError):
        DemiGodSpec(
            name=GOD_OUT_SUBDIR,
            domain="x",
            problem=Problem(context="c", goal="g"),
        )


def test_control_plane_files_are_not_staged_as_artifacts():
    """request.json must not land in the staging dir, because everything there
    is uploaded to the volume -- the same reason demigod keeps spec.json out of
    /run/out."""
    assert not REQUEST_PATH.startswith(STAGING_DIR + "/")
    assert ARTIFACT_PATH.startswith(GOD_OUT_SUBDIR + "/")


def test_launch_does_not_mount_the_out_volume():
    """A Sandbox cannot call `Volume.commit()`, and its mount writes flush only
    when it terminates -- so a mounted GOD would hide every artifact until it
    was gone. `godbox.launch` passes no `volumes=` at all and uploads instead.
    Asserted on the AST because the alternative is a live sandbox -- and on the
    AST rather than the text, so the comment explaining the rule does not trip
    the test that enforces it."""
    import ast
    import inspect
    import textwrap

    from godbox import launch

    tree = ast.parse(textwrap.dedent(inspect.getsource(launch.launch_god)))
    creates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create"
    ]
    assert creates, "launch_god no longer creates a sandbox"
    for call in creates:
        assert "volumes" not in {kw.arg for kw in call.keywords}


# --- CLI parsing -------------------------------------------------------------


def test_launch_defaults_are_the_cheap_ones():
    """Every turn is an Anthropic call. The defaults must be the ones you can
    afford to run by accident."""
    args = build_parser().parse_args(["launch"])
    assert args.problem == "simple"
    assert args.domains == 2
    assert args.turns == 12
    assert args.keep_alive == 0


@pytest.mark.parametrize(
    "argv,command",
    [
        (["status", "r1"], "status"),
        (["watch", "r1"], "watch"),
        (["list"], "list"),
        (["logs", "r1"], "logs"),
        (["followup", "r1", "hello"], "followup"),
        (["terminate", "r1"], "terminate"),
    ],
)
def test_every_subcommand_parses(argv, command):
    args = build_parser().parse_args(argv)
    assert args.command == command
    assert callable(args.func)


def test_followup_takes_a_free_text_message():
    args = build_parser().parse_args(["followup", "r1", "also consider the outlet"])
    assert args.message == "also consider the outlet"


# --- rendering ---------------------------------------------------------------


def test_fmt_duration():
    assert _fmt_duration(None) == "-"
    assert _fmt_duration(9) == "9s"
    assert _fmt_duration(125) == "2m05s"


def test_render_a_run_that_never_reported():
    """The output has to be a diagnosis, not a traceback."""
    text = _render(GodStatus(run_id="r1"))
    assert "r1" in text
    assert "unknown" in text
    assert "no heartbeat yet" in text


def test_render_shows_stale_prominently():
    import time

    status = GodStatus(
        run_id="r1", raw={"phase": "spawning", "heartbeat": time.time() - 10_000}
    )
    assert "STALE" in _liveness(status)


def test_render_a_finished_run_shows_the_solution():
    solution = {"problem_id": "valve-bottleneck", "answer": "no", "confidence": 0.8}
    status = GodStatus(
        run_id="r1",
        raw={
            "phase": "done",
            "started_at": 1000.0,
            "finished_at": 1030.0,
            "solution": solution,
            "domains": ["flow_a", "flow_b"],
            "demigods": {
                "flow_a": {"status": "ok", "started_at": 1001.0, "finished_at": 1020.0},
                "flow_b": {"status": "failed", "error": "no artifact"},
            },
        },
    )
    text = _render(status)
    assert "finished" in text
    assert "30s" in text
    assert "flow_a" in text and "flow_b" in text
    assert "no artifact" in text
    assert json.dumps(solution, indent=2) in text


def test_render_warns_on_a_newer_schema():
    """A reader that finds a version it does not understand should say so
    rather than silently misreport the run."""
    text = _render(GodStatus(run_id="r1", raw={"schema": 99, "phase": "done"}))
    assert "newer than this" in text
