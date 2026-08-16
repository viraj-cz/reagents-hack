"""re:SOLUTION -- the web interface onto a GOD run.

This package is a *consumer* of `reagents`, never a participant. It owns no
orchestration logic: it subscribes to the trace stream that `reagents.tracing`
already emits, reshapes each record into something a browser can render, and
serves it over SSE. Deleting this package leaves the runtime unchanged.

The dependency direction matters for the same reason it does elsewhere in this
repo: `reagents` must not learn that a UI exists, or the UI's needs start
shaping the orchestrator's prompts.
"""

from resolution.events import UiEvent, node_id_for_lane

__all__ = ["UiEvent", "node_id_for_lane"]
