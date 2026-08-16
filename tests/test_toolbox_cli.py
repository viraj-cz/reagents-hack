"""The demigod's side, driven over real HTTP against the real router.

Loopback only: no Modal, no key, no outbound network. The `live_broker` fixture
serves `broker.router.ToolboxRouter` on 127.0.0.1, which is the only way to
exercise `urllib`, the `HTTPError` path, and the CLI's exit codes against the
actual server instead of a mock that agrees with them by construction.

Exit codes are load-bearing here. The caller is an agent composing bash, and it
branches on them.
"""

from __future__ import annotations

import json

import pytest

from demigod.toolbox.cli import EXIT_OK, EXIT_REFUSED, EXIT_USAGE, main
from demigod.toolbox.client import (
    DEFAULT_GRANT_FILE,
    ENV_GRANT_FILE,
    ENV_LEASE,
    ENV_URL,
    ToolboxClient,
    ToolboxError,
)
from demigod.toolbox.protocol import ErrorCode, ToolboxGrant


def cli(url: str, lease: str, *args: str) -> int:
    return main(["--url", url, "--lease", lease, *args])


# --- client ------------------------------------------------------------------


def test_health_round_trips_without_a_lease(live_broker: str):
    client = ToolboxClient(live_broker, "unused")
    assert client.health()["ok"] is True


def test_list_and_call(live_broker: str, lease_id: str):
    client = ToolboxClient(live_broker, lease_id)

    listing = client.list_tools()
    assert [t.id for t in listing.tools] == ["graph.add", "graph.boom"]
    assert listing.lease.calls_remaining == 3

    response = client.call("graph.add", {"a": 40, "b": 2})
    assert response.ok is True
    assert response.result == {"sum": 42}
    assert response.lease.calls_used == 1


def test_call_never_raises_on_a_refusal(live_broker: str, lease_id: str):
    """The caller is an agent deciding what to do next. It needs the code and
    the message, not a traceback it has to parse out of stderr."""
    response = ToolboxClient(live_broker, lease_id).call("graph.unleased", {})
    assert response.ok is False
    assert response.error.code == ErrorCode.UNBOUND_TOOL


def test_list_raises_a_typed_error_on_a_bad_lease(live_broker: str):
    client = ToolboxClient(live_broker, "lease_nope")
    with pytest.raises(ToolboxError) as exc:
        client.list_tools()
    assert exc.value.code == ErrorCode.UNAUTHORIZED
    assert exc.value.retryable is False


def test_unreachable_broker_is_upstream_error_not_a_crash(lease_id: str):
    """Port 1 on loopback refuses instantly. A demigod must be able to tell
    'the broker is gone' from 'my lease is bad' -- one is a blocker, the other
    might be a misconfiguration it can report precisely."""
    client = ToolboxClient("http://127.0.0.1:1", lease_id, attempts=1)
    response = client.call("graph.add", {"a": 1, "b": 2})
    assert response.ok is False
    assert response.error.code == ErrorCode.UPSTREAM_ERROR


def test_client_refuses_to_construct_without_a_lease():
    with pytest.raises(ToolboxError):
        ToolboxClient("http://x", "")


# --- how the client finds its grant ------------------------------------------


def test_from_env_prefers_env_vars(monkeypatch, live_broker: str, lease_id: str):
    monkeypatch.setenv(ENV_URL, live_broker)
    monkeypatch.setenv(ENV_LEASE, lease_id)
    assert ToolboxClient.from_env().list_tools().lease.lease_id == lease_id


def test_from_env_falls_back_to_the_grant_file(
    monkeypatch, tmp_path, live_broker: str, lease_id: str
):
    """Belt and braces. The agent composes bash, and bash composes sub-shells,
    heredocs and `env -i` -- any of which can lose an exported variable, and the
    failure mode is an agent that concludes it has no tools."""
    monkeypatch.delenv(ENV_URL, raising=False)
    monkeypatch.delenv(ENV_LEASE, raising=False)
    path = tmp_path / "toolbox.json"
    path.write_text(
        ToolboxGrant(url=live_broker, lease_id=lease_id).model_dump_json(),
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_GRANT_FILE, str(path))

    assert ToolboxClient.from_env().list_tools().lease.lease_id == lease_id


def test_from_env_with_nothing_explains_both_mechanisms(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_URL, raising=False)
    monkeypatch.delenv(ENV_LEASE, raising=False)
    monkeypatch.setenv(ENV_GRANT_FILE, str(tmp_path / "absent.json"))
    with pytest.raises(ToolboxError) as exc:
        ToolboxClient.from_env()
    assert ENV_URL in exc.value.message
    assert "blockers" in exc.value.message


def test_the_default_grant_path_is_outside_both_volume_mounts():
    """A credential written into the agent's output dir would be committed to a
    volume and listed in `files`."""
    from demigod.layout import OUT_MOUNT, SHARED_MOUNT

    assert not DEFAULT_GRANT_FILE.startswith(OUT_MOUNT)
    assert not DEFAULT_GRANT_FILE.startswith(SHARED_MOUNT)


# --- CLI ---------------------------------------------------------------------


def test_list_exits_zero_and_names_the_tools(live_broker, lease_id, capsys):
    assert cli(live_broker, lease_id, "list") == EXIT_OK
    out = capsys.readouterr().out
    assert "graph.add" in out
    assert "3/3 calls remaining" in out
    # Multi-line descriptions must not wreck the one-tool-per-line listing.
    assert len([line for line in out.splitlines() if "graph.add" in line]) == 1


def test_describe_prints_the_schema(live_broker, lease_id, capsys):
    assert cli(live_broker, lease_id, "describe", "graph.add") == EXIT_OK
    assert '"required"' in capsys.readouterr().out


def test_call_with_an_input_file_writes_an_artifact(
    live_broker, lease_id, tmp_path, capsys
):
    args = tmp_path / "args.json"
    args.write_text(json.dumps({"a": 2, "b": 5}), encoding="utf-8")
    out = tmp_path / "result.json"

    code = cli(
        live_broker, lease_id, "call", "graph.add", "-i", str(args), "-o", str(out)
    )
    assert code == EXIT_OK
    assert json.loads(capsys.readouterr().out) == {"sum": 7}
    assert json.loads(out.read_text()) == {"sum": 7}


def test_call_reads_stdin(live_broker, lease_id, monkeypatch, capsys):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO('{"a": 1, "b": 1}'))
    assert cli(live_broker, lease_id, "call", "graph.add", "-i", "-") == EXIT_OK
    assert json.loads(capsys.readouterr().out) == {"sum": 2}


def test_call_accepts_inline_json(live_broker, lease_id, capsys):
    code = cli(
        live_broker, lease_id, "call", "graph.add", "--json-arg", '{"a":3,"b":4}'
    )
    assert code == EXIT_OK
    assert json.loads(capsys.readouterr().out) == {"sum": 7}


def test_refused_call_exits_one_and_names_the_code(live_broker, lease_id, capsys):
    assert cli(live_broker, lease_id, "call", "graph.unleased") == EXIT_REFUSED
    assert f"[{ErrorCode.UNBOUND_TOOL}]" in capsys.readouterr().err


def test_invalid_input_exits_one_and_shows_what_was_wrong(
    live_broker, lease_id, capsys
):
    code = cli(live_broker, lease_id, "call", "graph.add", "--json-arg", '{"a":1}')
    assert code == EXIT_REFUSED
    err = capsys.readouterr().err
    assert f"[{ErrorCode.INVALID_INPUT}]" in err
    assert "'b'" in err


def test_unreadable_input_file_exits_two_without_calling(
    live_broker, lease_id, store, capsys
):
    """Exit 2, not 1: nothing was called, so no budget was spent and the agent
    should fix its own command rather than conclude the tool is unavailable."""
    code = cli(live_broker, lease_id, "call", "graph.add", "-i", "/nope.json")
    assert code == EXIT_USAGE
    assert store.calls_used(lease_id) == 0


def test_non_object_input_is_a_usage_error(live_broker, lease_id, tmp_path):
    path = tmp_path / "args.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert (
        cli(live_broker, lease_id, "call", "graph.add", "-i", str(path)) == EXIT_USAGE
    )


def test_no_grant_anywhere_exits_two(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv(ENV_URL, raising=False)
    monkeypatch.delenv(ENV_LEASE, raising=False)
    monkeypatch.setenv(ENV_GRANT_FILE, str(tmp_path / "absent.json"))
    assert main(["list"]) == EXIT_USAGE


def test_help_mentions_every_subcommand(capsys):
    """The CLI is self-documenting by design -- it is the only tool doc a
    brokered demigod is given, and `--help` is what it will run first."""
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for command in ("list", "describe", "call"):
        assert command in out
