"""What GOD is asked to do, and where things live inside its sandbox.

The GOD-side analogue of `demigod/spec.py` + `demigod/layout.py`, kept as one
small file because it is one model and a handful of paths.

    /god/request.json     the task. Control-plane data, deliberately NOT an
                          artifact -- same reason demigod puts spec.json
                          outside its mounts.
    /god/god.log          GOD's own stdout+stderr, tee'd to a file.
    /god/artifacts/       staging. Local disk, NOT a volume mount.

    on the run's out volume, written by upload:
    _god/solution.json    the answer
    _god/trace.json       the full OrchestrationTrace
    <demigod>/            each demigod's dir, readable by GOD

GOD DOES NOT MOUNT THE OUT VOLUME, and that is a correction, not an oversight.
The obvious design was to mount it and write `solution.json` through the
filesystem. Three things measured live (`scripts/preflight_god.py`) killed it:

  * `Volume.commit()` DOES NOT WORK IN A SANDBOX. It raises
    `RuntimeError: commit() can only be called on a mounted volume inside a
    container`. The commit API is for Modal *Functions*, whose runtime injects
    the mounted volume object; a `Volume.from_name()` handle inside a Sandbox
    is an ordinary client handle and refuses. So there is no way to flush a
    Sandbox's mount writes on demand.
  * Mount writes ARE flushed, but only when the sandbox TERMINATES. Which
    means the artifacts of a finished run appear only after GOD is gone --
    and, worse, are invisible for the whole run in exactly the way that reads
    as "the agents produced nothing".
  * `Volume.batch_upload()` works fine from inside a Sandbox and is visible
    outside IMMEDIATELY, before termination.

So writes go through `batch_upload` and reads through `Volume.read_file` /
`listdir`, both client APIs, both mount-free. That also removes GOD's mount
from any interaction with the demigod sandboxes writing to the same volume
concurrently, and sidesteps Modal's rule that one Volume cannot be mounted
twice in one sandbox (the rule that forced `demigod/layout.py` to use two).

`_god` cannot collide with a demigod directory: demigod names must be
lowercase slugs starting with a letter, which forbids a leading underscore.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from reagents.contracts import NativeProblem

GOD_APP_NAME = "god"
"""All GOD sandboxes live under one Modal App, so `modal app logs god` shows
every orchestrator. DEMI_GODs stay under `demigod` -- separate apps make
"which layer is stuck?" answerable from the dashboard."""

GOD_WORKDIR = "/god"
REQUEST_PATH = "/god/request.json"
LOG_PATH = "/god/god.log"
STAGING_DIR = "/god/artifacts"
"""Local disk inside GOD's container, NOT a volume mount. Artifacts are built
here and then uploaded; see the module docstring for why the mount is gone."""

GOD_OUT_SUBDIR = "_god"
SOLUTION_FILENAME = "solution.json"
TRACE_FILENAME = "trace.json"

ARTIFACT_PATH = f"{GOD_OUT_SUBDIR}/{SOLUTION_FILENAME}"
"""Where the final solution lands on the out volume, as an outside caller sees
it: `modal volume get demigod-run-<run_id>-out _god/solution.json`."""


def out_volume_name(run_id: str) -> str:
    """The run's out volume.

    Deliberately the SAME volume `demigod.layout.RunLayout` builds, derived the
    same way rather than imported, because it is a name GOD must know before
    any demigod exists. Divergence here would put GOD's solution on a volume no
    demigod ever wrote to -- which would look like every demigod failing.
    """
    from demigod.layout import RunLayout

    return RunLayout(run_id=run_id, demigod_name="_god").out_volume_name


class GodRequest(BaseModel):
    """The whole task, handed to GOD as a file.

    A FILE, not argv. The problem statement is multi-line prose with quotes and
    commas in it; pushing that through `sandbox.exec`'s argument list is a
    quoting bug waiting to happen, and `demigod` already established the
    write-a-json-file pattern for exactly this reason.
    """

    run_id: str
    problem: NativeProblem
    domain_count: int | None = None
    """How many DEMI_GODs to spawn, or None to leave it to GOD.

    None by default: GOD reads the problem and decides, down to none at all for
    something it can just answer. An integer pins the count for cost control and
    binds in both directions -- it will spawn that many on a trivial problem."""

    max_turns: int = 12
    """Per-demigod turn cap. THE cost lever -- every turn is an Anthropic call."""

    model: str = "claude-opus-4-8"
    """Pinned for both God's calls and every demigod in this run."""

    approved_write_tools: list[str] = Field(default_factory=list)
    """Operator approval travels WITH the request and nothing inside the
    sandbox can widen it. GOD's orchestrator refuses to spawn a demigod holding
    a write tool that is not named here, so leaving these empty is
    the safe default rather than a missing feature."""

    use_broker: bool = True
    """Publish each demigod's already-approved lease to TOOLBOX_BROKER.

    TRUE BY DEFAULT. This one field gates three things -- lease publication,
    whether the runtime insists on one, and egress pinning -- so leaving it
    unset produced runs that finished with high confidence and `tool_trace=0`,
    every artifact carrying a blocker saying its tools were unreachable. The
    broker is deployed precisely so that is not the state a run lands in."""

    shared_files: list[str] = Field(default_factory=list)
    """Paths under this run's shared volume, relative to its root.

    NAMES ONLY, and the volume is already populated when GOD reads them: shared/
    is mounted read-only everywhere and GOD's own sandbox mounts no volume at
    all, so whoever built this request did the seeding. GOD passes these to its
    demigod runtime, which is what puts the files in the spawned spec and pulls
    pandas into the image.

    NOT SEALED. These files carry their native column headers into a sandbox,
    unlike `problem.inputs`, which is projected. See `seed_shared_files`."""

    verifier_id: str | None = None
    """Closed God-side native verifier name. Never a module path or callable."""

    sandbox_id: str = ""
    """GOD's own sandbox, filled in by the launcher. GOD needs it to terminate
    itself -- a container has no reliable way to learn its own sandbox id."""

    keep_alive_s: int = 0
    """Seconds to stay up after finishing, before self-terminating. 0 means go
    immediately. Raise it when you want to exec into a finished GOD and look
    around; every second of it is billed."""
