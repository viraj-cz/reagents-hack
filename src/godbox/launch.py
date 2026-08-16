"""OUTSIDE the sandbox: create GOD's container, hand it the task, walk away.

The mirror image of `demigod/runner/inside.py`, with one deliberate inversion.
`InsideSandboxRunner` blocks for the whole agent run and terminates the sandbox
in a `finally`. This does neither, and both differences are the feature:

* **It does not block.** `launch_god` returns as soon as the entrypoint process
  is running. The run then survives the laptop closing, the terminal being
  killed, and the caller rebooting -- which is the entire reason GOD moved into
  a sandbox.
* **It does not terminate on the happy path.** A `finally` that killed a
  successfully-launched GOD would destroy the run it just started. The `finally`
  is still unconditional; what it does is conditional. It terminates a sandbox
  that was created but never got as far as running anything, because *that*
  sandbox is a pure leak billing until `max_lifetime_s`.

GOD's own teardown therefore happens from inside (`godbox/entrypoint.py`
self-terminates in its `finally`), with `idle_timeout` as the backstop for the
case where the process is SIGKILLed before its `finally` can run.

WHY A SANDBOX AND NOT A MODAL FUNCTION. A Function is fire-and-forget: you get
a return value or you get nothing, and there is nowhere to send a follow-up.
A Sandbox has an identity that outlives any one call -- `Sandbox.from_name` and
`Sandbox.from_id` both reconnect to a running one -- which is what makes
"check in on it at any time" a lookup rather than a subscription.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from godbox.images import god_image
from godbox.layout import (
    ARTIFACT_PATH,
    GOD_APP_NAME,
    GOD_WORKDIR,
    LOG_PATH,
    REQUEST_PATH,
    GodRequest,
    out_volume_name,
)
from godbox.status import STATUS_SCHEMA, Phase, dict_name, open_dict

if TYPE_CHECKING:
    import modal

ANTHROPIC_SECRET_NAME = "demigod-anthropic"
"""GOD's own planner/transformer/integrator calls. The SAME secret the demigods
mount -- one Anthropic key for the workspace -- but note it reaches GOD as a
mounted secret rather than from the caller's `.env`, because there is no caller
process by the time GOD runs."""

MODAL_TOKEN_SECRET_NAME = "demigod-modal-token"

TOOLING_SECRET_NAME = "reagents-tooling"
"""Feature flags and sponsor credentials for GOD's capability catalog.

OPTIONAL, and absent is a supported state: a workspace without this secret gets
the built-in tools and a note saying so, exactly as before. What is not
supported is the previous silence -- a laptop with REAGENTS_ENABLE_MCP set
reported "1 remote catalogs will be checked" while the sandbox running the same
code reported none, and nothing connected the two.

Scope: what GOD needs to DISCOVER a catalog, not what the broker needs to call
into it. A sponsor key here would let GOD call the sponsor directly, which is
the broker's job precisely so every call is leased and audited."""
"""MODAL_TOKEN_ID / MODAL_TOKEN_SECRET, so GOD can spawn DEMI_GOD sandboxes
from inside its own. Create once per workspace:

    uv run modal secret create demigod-modal-token \\
        MODAL_TOKEN_ID=ak-... MODAL_TOKEN_SECRET=as-...

This is the credential `scripts/preflight_nested.py` exists to justify, and the
one a DEMI_GOD must never hold. Modal credentials are workspace-wide."""

DEFAULT_MAX_LIFETIME_S = 3600
"""Hard wall-clock cap on a GOD run. A full loop is one planner call, N
transforms, N sandboxed demigods, and one integrate; an hour is generous for
the defaults and still bounds a runaway."""

DEFAULT_IDLE_TIMEOUT_S = 600
"""Backstop only, and safe to keep short. MEASURED, because guessing here would
have been fatal: a running `exec` counts as activity even when it is detached
and nobody is reading its streams. A sandbox with `idle_timeout=30` running a
50-second unattended process survived the whole 50 seconds and was collected
~30s after the process ENDED. So this timer does not run while GOD works; it
starts when GOD stops, and only matters if the self-terminate never happened.

That measurement is the difference between this design and one where every run
silently dies a few minutes after the laptop closes."""

DEFAULT_CPU = 1.0
DEFAULT_MEMORY_MB = 2048
"""GOD orchestrates and calls an API. It does no numerical work of its own --
that is what the demigods' images are for."""


@dataclass(frozen=True)
class GodRunHandle:
    """Everything needed to find this run again from a different machine.

    Deliberately just identifiers. Holding a live `modal.Sandbox` object would
    make the handle unserializable and imply a connection that, by design,
    nothing keeps open.
    """

    run_id: str
    sandbox_id: str
    sandbox_name: str
    app_name: str
    status_dict: str
    artifact_volume: str
    artifact_path: str

    def describe(self) -> str:
        return (
            f"run_id  : {self.run_id}\n"
            f"sandbox : {self.sandbox_id} (name {self.sandbox_name!r} "
            f"in app {self.app_name!r})\n"
            f"status  : modal.Dict {self.status_dict}\n"
            f"          uv run god status {self.run_id}\n"
            f"artifacts: volume {self.artifact_volume} -> {self.artifact_path}\n"
            f"          uv run modal volume ls {self.artifact_volume}"
        )


def _secrets(*, verbose: bool = True) -> list[Any]:
    """Anthropic and the Modal token are required; tooling is not.

    Anthropic for GOD's own reasoning, the Modal token so GOD can spawn
    demigods from in here -- a DEMI_GOD gets only the first, and its image
    cannot use the second anyway. `reagents-tooling` is looked up separately
    because a missing optional secret must not fail every launch.
    """
    import modal

    secrets = [
        modal.Secret.from_name(ANTHROPIC_SECRET_NAME),
        modal.Secret.from_name(MODAL_TOKEN_SECRET_NAME),
    ]
    try:
        tooling = modal.Secret.from_name(TOOLING_SECRET_NAME)
        tooling.hydrate()
        secrets.append(tooling)
    except Exception as exc:
        if verbose:
            print(
                f"[god] no {TOOLING_SECRET_NAME!r} secret ({type(exc).__name__}); "
                f"GOD will plan against built-in tools only"
            )
    return secrets


def sandbox_name(run_id: str) -> str:
    """Sandbox names are unique within an app, which makes `run_id` a natural
    key: `Sandbox.from_name("god", "god-<run_id>")` reconnects to a live run
    with nothing but the id the user already typed."""
    return f"god-{run_id}"


def new_run_id() -> str:
    return f"god{uuid.uuid4().hex[:8]}"


def launch_god(
    request: GodRequest,
    *,
    max_lifetime_s: int = DEFAULT_MAX_LIFETIME_S,
    idle_timeout_s: int = DEFAULT_IDLE_TIMEOUT_S,
    cpu: float = DEFAULT_CPU,
    memory_mb: int = DEFAULT_MEMORY_MB,
    verbose: bool = True,
) -> GodRunHandle:
    """Start a GOD run and return immediately.

    Order matters and is not arbitrary:

    1. Prime the status Dict FIRST. If sandbox creation then fails, the failure
       is visible to `god status` instead of vanishing into this process's
       stderr -- which may be a terminal nobody is watching.
    2. Create the sandbox, and record `sandbox_id` in the Dict BEFORE anything
       is executed in it. A run that dies during startup is then still
       terminable by id, rather than being an orphan billing quietly for an
       hour.
    3. Write the request, then exec. Never the reverse.
    """
    import modal

    run_id = request.run_id
    volume_name = out_volume_name(run_id)

    status = open_dict(run_id, create_if_missing=True)
    now = time.time()
    status.update(
        schema=STATUS_SCHEMA,
        run_id=run_id,
        app_name=GOD_APP_NAME,
        artifact_volume=volume_name,
        artifact_path=ARTIFACT_PATH,
        problem_id=request.problem.id,
        domain_count=request.domain_count,
        max_turns=request.max_turns,
        phase=Phase.LAUNCHING.value,
        started_at=now,
        updated_at=now,
        heartbeat=now,
        events=[
            {
                "at": now,
                "phase": Phase.LAUNCHING.value,
                "message": f"launching GOD for problem {request.problem.id!r}",
            }
        ],
        followups=[],
        demigods={},
    )

    app = modal.App.lookup(GOD_APP_NAME, create_if_missing=True)
    # Created here rather than left to the first demigod, so that a run which
    # dies before spawning anything still has a volume to explain itself on.
    # GOD does NOT mount it -- see godbox/layout.py for why uploads beat mounts
    # from inside a Sandbox. `create_if_missing` makes this idempotent per run.
    modal.Volume.from_name(volume_name, create_if_missing=True)

    if verbose:
        print(
            f"[god] launching run={run_id} problem={request.problem.id!r} "
            f"domains={request.domain_count or 'GOD decides'} "
            f"turns={request.max_turns}"
        )

    sandbox: modal.Sandbox | None = None
    launched = False
    try:
        sandbox = modal.Sandbox.create(
            app=app,
            name=sandbox_name(run_id),
            tags={"role": "god", "run_id": run_id},
            image=god_image(),
            # BOTH secrets. Anthropic for GOD's own reasoning; the Modal token
            # so GOD can spawn demigods from in here. A DEMI_GOD gets only the
            # first, and its image cannot use the second anyway.
            secrets=_secrets(verbose=verbose),
            # No `volumes=`. GOD reaches the out volume through the Volume
            # client API instead, which is the only way a Sandbox can make a
            # write visible before it terminates -- godbox/layout.py has the
            # measurements.
            timeout=max_lifetime_s,
            idle_timeout=idle_timeout_s,
            workdir=GOD_WORKDIR,
            cpu=cpu,
            memory=memory_mb,
        )
        status.put("sandbox_id", sandbox.object_id)
        if verbose:
            print(f"[god] sandbox {sandbox.object_id} up")

        stamped = request.model_copy(update={"sandbox_id": sandbox.object_id})
        # `sandbox.filesystem`, NOT the retired `sandbox.open()`. Data first,
        # remote path second.
        sandbox.filesystem.write_text(stamped.model_dump_json(indent=2), REQUEST_PATH)

        _exec_detached(sandbox)
        launched = True
        if verbose:
            print(f"[god] detached; poll with `uv run god status {run_id}`")

        return GodRunHandle(
            run_id=run_id,
            sandbox_id=sandbox.object_id,
            sandbox_name=sandbox_name(run_id),
            app_name=GOD_APP_NAME,
            status_dict=dict_name(run_id),
            artifact_volume=volume_name,
            artifact_path=ARTIFACT_PATH,
        )
    except Exception as exc:
        status.put("phase", Phase.FAILED.value)
        status.put("error", f"launch failed: {type(exc).__name__}: {exc}")
        status.put("finished_at", time.time())
        raise
    finally:
        # Unconditional, but conditional in what it does -- see the module
        # docstring. Only a sandbox that never started work gets killed here.
        if sandbox is not None and not launched:
            sandbox.terminate()
            if verbose:
                print("[god] sandbox terminated (never started)")


def _exec_detached(sandbox: modal.Sandbox) -> None:
    """Start GOD's loop without holding on to its output streams.

    Two things here are load-bearing:

    **`StreamType.DEVNULL`.** The default is `PIPE`, which buffers server-side
    and applies backpressure when nobody reads. For a process the caller
    deliberately abandons, that is a hang: GOD would run happily for a few
    minutes and then block forever on a full stdout pipe, with the status Dict
    frozen at whatever phase it reached. Discarding the stream is what makes
    detaching safe.

    **`> /god/god.log 2>&1`.** Having discarded the stream, the logs have to go
    somewhere. A file inside the sandbox is readable at any time through
    `sandbox.filesystem.read_text` (`god logs <run_id>`), needs no connection
    held open, and costs nothing. The durable summary still goes to the status
    Dict, because the log dies with the container.

    `python -u` so the file is current rather than block-buffered; a log that
    only appears at exit is useless for exactly the runs you want to watch.
    """
    from modal.stream_type import StreamType

    sandbox.exec(
        "sh",
        "-c",
        f"python -u -m godbox.entrypoint --request {REQUEST_PATH} > {LOG_PATH} 2>&1",
        workdir=GOD_WORKDIR,
        stdout=StreamType.DEVNULL,
        stderr=StreamType.DEVNULL,
    )


def connect(run_id: str) -> modal.Sandbox:
    """Reconnect to a running GOD by run id.

    Raises `modal.exception.NotFoundError` if no sandbox with that name is
    running -- which for a finished run is the correct answer, since GOD
    terminates itself. The status Dict outlives the sandbox and is where a
    completed run is read from.
    """
    import modal

    return modal.Sandbox.from_name(GOD_APP_NAME, sandbox_name(run_id))


def read_log(run_id: str, sandbox_id: str | None = None) -> str:
    """GOD's stdout, read out of the live sandbox.

    Prefers `sandbox_id` (recorded in the status Dict at launch) over the name
    lookup, because the id keeps working for a sandbox that was created but
    never named -- and costs one fewer round trip.
    """
    import modal

    sandbox = (
        modal.Sandbox.from_id(sandbox_id)
        if sandbox_id
        else modal.Sandbox.from_name(GOD_APP_NAME, sandbox_name(run_id))
    )
    return sandbox.filesystem.read_text(LOG_PATH)
