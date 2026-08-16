"""The live trace channel, tested without Modal.

`QueueTracer` takes its queue by injection precisely so this suite can run
offline: the interesting behaviour is batching, ordering, overflow and failure
tolerance, none of which need a real `modal.Queue` to exercise.

The property that matters most here is the last one. This channel exists to
watch a run; if it can fail a run, it has made things worse than no channel at
all.
"""

from __future__ import annotations

import threading

from godbox.trace_channel import QueueTracer, queue_name


class FakeQueue:
    def __init__(self, fail_times: int = 0) -> None:
        self.batches: list[list[dict]] = []
        self.fail_times = fail_times
        self._lock = threading.Lock()

    def put_many(self, values: list[dict]) -> None:
        with self._lock:
            if self.fail_times > 0:
                self.fail_times -= 1
                raise RuntimeError("modal is having a day")
            self.batches.append(list(values))

    def events(self) -> list[dict]:
        with self._lock:
            return [event for batch in self.batches for event in batch]


def test_events_arrive_in_order_with_their_lane_and_kind() -> None:
    queue = FakeQueue()
    tracer = QueueTracer("run-1", queue=queue, flush_interval_s=0.01)

    tracer.emit("GOD", "PLAN", "selected 3 domains", data=[{"name": "a"}])
    tracer.emit("DEMI:a", "TOOL CALL", "simplify", data={"arguments": {"expr": 1}})
    tracer.close()

    events = queue.events()
    assert [e["seq"] for e in events] == [1, 2]
    assert [e["lane"] for e in events] == ["GOD", "DEMI:a"]
    assert [e["kind"] for e in events] == ["PLAN", "TOOL CALL"]
    assert events[1]["data"] == {"arguments": {"expr": 1}}
    # Elapsed seconds, so a consumer can render a timeline without a clock of
    # its own -- and monotonic, so it cannot go backwards mid-run.
    assert events[0]["t"] <= events[1]["t"]


def test_close_flushes_what_the_interval_has_not() -> None:
    """The last events of a run are the ones a watcher most wants."""

    queue = FakeQueue()
    tracer = QueueTracer("run-2", queue=queue, flush_interval_s=3600)
    tracer.emit("GOD", "DONE", "integrated answer ready")
    assert queue.events() == []
    tracer.close()
    assert [e["message"] for e in queue.events()] == ["integrated answer ready"]


def test_a_broken_queue_never_reaches_the_caller() -> None:
    """GOD must not fail because its trace channel did."""

    queue = FakeQueue(fail_times=1)
    tracer = QueueTracer("run-3", queue=queue, flush_interval_s=0.01, batch_size=1)
    tracer.emit("GOD", "START", "first")
    tracer.emit("GOD", "START", "second")
    tracer.close()

    assert tracer.dropped >= 1
    # The surviving event still got through: one bad batch is not a dead channel.
    assert [e["message"] for e in queue.events()] == ["second"]


def test_overflow_drops_the_oldest_and_counts_it() -> None:
    """An unread trace must not grow GOD's memory without bound."""

    queue = FakeQueue()
    tracer = QueueTracer(
        "run-4", queue=queue, flush_interval_s=3600, max_pending=3
    )
    for i in range(6):
        tracer.emit("GOD", "NOTE", f"event-{i}")
    tracer.close()

    assert tracer.dropped == 3
    # The PRESENT survives, not the past: a live view that fell behind wants
    # what is happening now.
    assert [e["message"] for e in queue.events()] == ["event-3", "event-4", "event-5"]


def test_queue_is_named_per_run() -> None:
    assert queue_name("run-abc") == "god-trace-run-abc"
    assert queue_name("run-abc") != queue_name("run-abd")
