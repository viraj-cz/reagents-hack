"""The status channel's contract, offline.

No Modal account, no API key, no network. `StatusWriter` takes a `backend=`
injection point precisely so the key contract -- what is written, when, and in
what order -- is testable without spending a sandbox to find out.
"""

from __future__ import annotations

import time

import pytest

from godbox.status import (
    MAX_EVENTS,
    STALE_AFTER_S,
    GodStatus,
    Phase,
    StatusWriter,
    dict_name,
    run_id_from_dict_name,
)


class FakeDict:
    """Stands in for `modal.Dict`.

    Note what it does NOT have: an `.aio` attribute on `put`. That exercises
    `_aput`'s fallback, which is the path a future SDK without `.aio` would
    take.
    """

    def __init__(self, initial: dict | None = None) -> None:
        self.data: dict = dict(initial or {})
        self.writes: list[str] = []

    def put(self, key, value):
        self.data[key] = value
        self.writes.append(key)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def items(self):
        return list(self.data.items())


def writer(initial: dict | None = None) -> tuple[StatusWriter, FakeDict]:
    backend = FakeDict(initial)
    return StatusWriter("t1", backend=backend), backend


# --- naming ------------------------------------------------------------------


def test_dict_name_round_trips():
    assert dict_name("abc") == "god-run-abc"
    assert run_id_from_dict_name("god-run-abc") == "abc"


def test_run_id_from_dict_name_ignores_foreign_objects():
    """`god list` iterates every Dict in the workspace; a stranger's Dict must
    not be reported as a GOD run."""
    assert run_id_from_dict_name("demigod-run-x-out") is None
    assert run_id_from_dict_name("some-other-dict") is None


# --- phases ------------------------------------------------------------------


def test_only_done_and_failed_are_terminal():
    terminal = {p for p in Phase if p.is_terminal}
    assert terminal == {Phase.DONE, Phase.FAILED}


async def test_set_phase_writes_phase_and_event():
    status, backend = writer()
    await status.set_phase(Phase.PLANNING, "inventing domains")
    assert backend.data["phase"] == "planning"
    assert backend.data["events"][-1]["message"] == "inventing domains"
    assert backend.data["events"][-1]["phase"] == "planning"


async def test_events_carry_the_phase_they_happened_in():
    status, backend = writer()
    await status.set_phase(Phase.PLANNING)
    await status.note("planned a")
    await status.set_phase(Phase.SPAWNING)
    await status.note("spawned b")
    phases = [e["phase"] for e in backend.data["events"]]
    assert phases == ["planning", "spawning"]


# --- terminal ordering -------------------------------------------------------


def test_complete_writes_phase_last():
    """A poller must never read `done` before the solution that makes it
    meaningful. Ordering is the only thing protecting that -- there is no
    transaction across Dict keys."""
    status, backend = writer()
    status.complete({"answer": "42"}, summary="all good")
    assert backend.data["phase"] == "done"
    assert backend.writes.index("solution") < backend.writes.index("phase")
    assert backend.writes.index("finished_at") < backend.writes.index("phase")


def test_fail_writes_phase_last():
    status, backend = writer()
    status.fail("boom")
    assert backend.data["phase"] == "failed"
    assert backend.data["error"] == "boom"
    assert backend.writes.index("error") < backend.writes.index("phase")


# --- demigod rows ------------------------------------------------------------


async def test_demigod_rows_upsert_and_stamp_finish():
    status, backend = writer()
    await status.demigod("flow", "running")
    assert backend.data["demigods"]["flow"]["status"] == "running"
    assert "finished_at" not in backend.data["demigods"]["flow"]

    await status.demigod("flow", "ok", confidence=0.8, files=2)
    row = backend.data["demigods"]["flow"]
    assert row["status"] == "ok"
    assert row["confidence"] == 0.8
    assert "finished_at" in row
    assert "started_at" in row


async def test_demigod_none_fields_are_not_written():
    """`result.error` is None on success; writing it would put a null `error`
    on a healthy row and make the status table read as a failure."""
    status, backend = writer()
    await status.demigod("flow", "ok", confidence=0.8, error=None)
    assert "error" not in backend.data["demigods"]["flow"]


async def test_concurrent_demigod_writes_do_not_lose_rows():
    """Demigods finish in parallel; each write sends the whole mirror. The lock
    in `_aput` is what stops an older snapshot landing last."""
    import asyncio

    status, backend = writer()
    await asyncio.gather(
        *(status.demigod(f"d{i}", "ok", confidence=0.5) for i in range(8))
    )
    assert set(backend.data["demigods"]) == {f"d{i}" for i in range(8)}


# --- events cap --------------------------------------------------------------


async def test_events_are_capped():
    status, backend = writer()
    for i in range(MAX_EVENTS + 50):
        await status.note(f"event {i}")
    events = backend.data["events"]
    assert len(events) == MAX_EVENTS
    # The cap must drop the OLDEST, not the newest -- a truncated log that
    # stops before the failure is worse than no log.
    assert events[-1]["message"] == f"event {MAX_EVENTS + 49}"


# --- adopt -------------------------------------------------------------------


def test_adopt_seeds_from_what_the_launcher_wrote():
    status, backend = writer({"events": [{"at": 1.0, "message": "launching"}]})
    status.adopt()
    status.complete({"answer": "x"}, summary="done")
    messages = [e["message"] for e in backend.data["events"]]
    assert messages == ["launching", "done"]


def test_terminal_writers_are_never_called_from_async_code():
    """`complete` and `fail` use Modal's BLOCKING interface. Calling either
    from inside a running event loop earns an `AsyncUsageWarning` and stalls
    the loop for the whole RPC -- which is what the first version of
    `godbox/entrypoint.py` did, and why `_solve` now returns its payload
    instead of writing the terminal status itself.

    Checked structurally because the symptom is a warning on stderr inside a
    container nobody is watching, not a failure."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / "src" / "godbox"
    for path in sorted(source.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                    assert call.func.attr not in ("complete", "fail"), (
                        f"{path.name}:{call.lineno} calls a synchronous "
                        f"terminal status writer from `async def "
                        f"{node.name}`; return the payload and write it "
                        f"outside asyncio.run instead"
                    )


def test_adopt_survives_an_unreadable_dict():
    """Losing progress reporting is bad; refusing to reason because the
    reporting channel hiccuped is worse."""

    class Broken(FakeDict):
        def get(self, key, default=None):
            raise RuntimeError("transient")

    status = StatusWriter("t1", backend=Broken())
    status.adopt()  # must not raise


# --- followups ---------------------------------------------------------------


async def test_drain_followups_marks_rather_than_deletes():
    pending = {"at": 1.0, "message": "look at the outlet", "delivered": False}
    status, backend = writer({"followups": [pending]})
    status.adopt()
    assert await status.drain_followups() == ["look at the outlet"]
    assert backend.data["followups"][0]["delivered"] is True
    # Already-delivered messages are not replayed on the next phase boundary.
    assert await status.drain_followups() == []


async def test_drain_followups_records_receipt_in_events():
    status, backend = writer(
        {"followups": [{"at": 1.0, "message": "hello", "delivered": False}]}
    )
    status.adopt()
    await status.drain_followups()
    assert "FOLLOWUP received: hello" in backend.data["events"][-1]["message"]


# --- read side ---------------------------------------------------------------


def test_status_of_an_empty_dict_is_still_usable():
    """A run whose sandbox died between create and the first write must still
    produce a diagnosis, not a KeyError."""
    status = GodStatus(run_id="t1")
    assert status.phase == "unknown"
    assert status.is_terminal is False
    assert status.elapsed_s is None
    assert status.demigods == {}
    assert status.solution is None


def test_stale_when_the_heartbeat_stops():
    now = time.time()
    running = GodStatus(
        run_id="t1",
        raw={"phase": "spawning", "heartbeat": now - STALE_AFTER_S - 1},
    )
    assert running.is_stale is True

    fresh = GodStatus(run_id="t1", raw={"phase": "spawning", "heartbeat": now})
    assert fresh.is_stale is False


def test_a_finished_run_is_never_stale():
    """A completed run legitimately stops beating; reporting it as stale would
    make every successful run look broken a minute later."""
    old = time.time() - 10_000
    status = GodStatus(run_id="t1", raw={"phase": "done", "heartbeat": old})
    assert status.is_terminal is True
    assert status.is_stale is False


def test_elapsed_freezes_at_finish():
    started = time.time() - 100
    status = GodStatus(
        run_id="t1",
        raw={"phase": "done", "started_at": started, "finished_at": started + 30},
    )
    assert status.elapsed_s == pytest.approx(30, abs=0.01)
