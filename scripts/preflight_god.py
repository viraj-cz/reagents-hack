"""Does GOD-in-a-sandbox actually work? ONE sandbox, ZERO Anthropic tokens.

    uv run python scripts/preflight_god.py

`preflight_nested.py` proved the one thing that had to be true before any of
this was worth building: a sandbox holding a Modal token can create another
sandbox, exec in it, read its filesystem, and terminate it. This proves the
five things that have to be true for the *design built on top of it* to work,
and each is a claim that offline tests cannot reach:

  1. **GOD's image is complete.** `godbox.entrypoint`, `reagents.god.orchestrator`
     and `demigod.spawn` all import inside it. A missing dependency here
     surfaces otherwise as a run that dies three seconds in, having already
     paid for image pull and container start.
  2. **`modal.Dict` really is a live channel.** A key written from inside the
     container is readable from this laptop while the container still runs.
     The entire status design rests on this and on nothing else.
  3. **An artifact uploaded from inside is visible outside, immediately.** The
     other half of the two-channel split. This check is why the design uses
     `Volume.batch_upload` rather than a mount: the first version mounted the
     volume and called `Volume.commit()`, and this script is what caught
     `RuntimeError: commit() can only be called on a mounted volume inside a
     container`. A Sandbox's mount writes flush only when it terminates, so the
     mounted design would have hidden every artifact until GOD was gone.
  4. **A detached process outlives its client and its idle_timeout.** Nobody
     holds GOD's stdout; nobody holds a connection. If Modal counted an
     unattended sandbox as idle, every run would die a few minutes after the
     laptop closed -- silently, and only in production. (It does not: a running
     exec counts as activity even with both streams sent to DEVNULL, and the
     idle timer starts when the exec ends.)
  5. **The real entrypoint runs to a terminal status.** Not a stand-in --
     `python -m godbox.entrypoint` against a real request, stopped one step
     short of the model by blanking `ANTHROPIC_API_KEY`. Covers request
     parsing, `StatusWriter.adopt`, the heartbeat thread, the failure path,
     and the exit code, for zero tokens.

Plus, at the end, that GOD can terminate its own sandbox -- checked so that
being collected by `idle_timeout` cannot pass for a self-terminate.

COST. One small sandbox for roughly two minutes, no GPU, no agent, no model
call. Cents. Run it after a Modal SDK bump, after any change to
`godbox/images.py` or `godbox/launch.py`, and before the first real run of the
day. It is NOT a substitute for a live GOD run -- that costs Anthropic tokens
and is the operator's call.
"""

from __future__ import annotations

import sys
import time

from godbox.images import god_image
from godbox.launch import (
    ANTHROPIC_SECRET_NAME,
    MODAL_TOKEN_SECRET_NAME,
)
from godbox.layout import (
    GOD_APP_NAME,
    GOD_OUT_SUBDIR,
    GOD_WORKDIR,
    REQUEST_PATH,
    STAGING_DIR,
    GodRequest,
)
from godbox.status import STATUS_SCHEMA, Phase, dict_name, open_dict

RUN_ID = "preflight"
ENTRYPOINT_RUN_ID = "preflight-entrypoint"
"""A separate run id, so the entrypoint check writes a real terminal status
without trampling the probe's Dict."""

VOLUME_NAME = "god-preflight-out"

IDLE_TIMEOUT_S = 30
"""Deliberately short, and deliberately shorter than DETACHED_SLEEP_S. That gap
is the whole point of check 4."""

DETACHED_SLEEP_S = 50
"""How long the unattended probe waits before reporting in. Must exceed
IDLE_TIMEOUT_S by enough that crossing it is not a timing coincidence."""

POLL_TIMEOUT_S = 150

# Runs INSIDE the sandbox, detached: no client reads its output, exactly as
# godbox.launch._exec_detached leaves GOD running. Performs, in order, the same
# four operations the real entrypoint performs at the end of a run.
PROBE = f"""
import json, os, pathlib, time
import modal

time.sleep({DETACHED_SLEEP_S})

# 1. Artifacts, staged locally then UPLOADED. Not a mounted write: a Sandbox
#    cannot call Volume.commit(), and its mount writes flush only at
#    termination -- which is too late to be useful and too late to test.
staging = pathlib.Path("{STAGING_DIR}")
staging.mkdir(parents=True, exist_ok=True)
probe = staging / "probe.json"
probe.write_text(json.dumps({{"probe": "PREFLIGHT_ARTIFACT"}}))
with modal.Volume.from_name("{VOLUME_NAME}").batch_upload(force=True) as batch:
    batch.put_file(probe, "/{GOD_OUT_SUBDIR}/probe.json")

# 2. Status to the Dict. Cross-container, no commit semantics at all.
d = modal.Dict.from_name("{dict_name(RUN_ID)}", create_if_missing=True)
d.put("phase", "done")
d.put("detached_seconds", {DETACHED_SLEEP_S})
d.put("from_inside", True)

# 3. Announce the self-terminate BEFORE doing it. Without this marker, a
#    sandbox reaped by idle_timeout is indistinguishable from one that tore
#    itself down -- and the first version of this script scored that as a pass.
d.put("about_to_terminate", time.time())

# 4. Self-terminate. Nothing after this line runs.
modal.Sandbox.from_id(os.environ["PREFLIGHT_SANDBOX_ID"]).terminate()
"""


def _check_entrypoint(sandbox, check) -> None:
    """Run `godbox.entrypoint` for real, and stop it just short of the model.

    The ONE way to exercise request parsing, `StatusWriter.adopt`, the
    heartbeat thread, the terminal-failure path and the exit code without
    paying for a planner call: blank `ANTHROPIC_API_KEY` and let the
    entrypoint's own guard fire. That guard exists because `make_llm()`
    silently returns the SCRIPTED toy client when the key is missing, so
    without it a keyless GOD reports a fabricated answer with every appearance
    of success -- which makes this both a plumbing check and a check on the
    thing protecting against the worst failure mode in the system.

    `sandbox_id=""` in the request so the entrypoint leaves teardown to
    idle_timeout instead of killing the sandbox the rest of this script needs.
    """
    import modal

    from reagents.toy import simple_problem

    modal.Dict.objects.delete(dict_name(ENTRYPOINT_RUN_ID), allow_missing=True)
    status = open_dict(ENTRYPOINT_RUN_ID, create_if_missing=True)
    status.put("phase", Phase.LAUNCHING.value)

    request = GodRequest(
        run_id=ENTRYPOINT_RUN_ID,
        problem=simple_problem(),
        domain_count=1,
        max_turns=1,
    )
    sandbox.filesystem.write_text(request.model_dump_json(indent=2), REQUEST_PATH)

    proc = sandbox.exec(
        "python",
        "-u",
        "-m",
        "godbox.entrypoint",
        "--request",
        REQUEST_PATH,
        env={"ANTHROPIC_API_KEY": ""},
        workdir=GOD_WORKDIR,
        timeout=180,
    )
    out = proc.stdout.read()
    proc.wait()

    # Asserted on the Dict, not on stdout: the point is that the channel
    # carried it. A run whose stdout says the right thing but whose Dict is
    # empty is exactly the failure this design exists to prevent.
    messages = [e.get("message", "") for e in (status.get("events") or [])]
    check(
        "entrypoint parsed the request and reported STARTING",
        any("GOD process up" in m for m in messages),
        f"{messages} | stdout={out[:120]}",
    )
    check(
        "entrypoint exits non-zero on failure",
        proc.returncode != 0,
        f"returncode={proc.returncode}",
    )
    check("status reached a terminal phase", status.get("phase") == Phase.FAILED.value)
    error = str(status.get("error") or "")
    check(
        "the missing-key guard fired (not a silent scripted-LLM answer)",
        "ANTHROPIC_API_KEY" in error,
        error[:160],
    )
    check("a heartbeat was written", status.get("heartbeat") is not None)
    modal.Dict.objects.delete(dict_name(ENTRYPOINT_RUN_ID), allow_missing=True)


def main() -> int:
    import modal
    from modal.stream_type import StreamType

    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        verdict = "PASS" if ok else "FAIL"
        print(f"  {verdict}  {label}{f' | {detail}' if detail else ''}")
        if not ok:
            failures.append(label)

    print("[1/6] secrets GOD needs")
    for name, why in (
        (ANTHROPIC_SECRET_NAME, "GOD's own planner/transformer/integrator calls"),
        (MODAL_TOKEN_SECRET_NAME, "spawning DEMI_GODs from inside GOD's sandbox"),
    ):
        try:
            modal.Secret.from_name(name).hydrate()
            check(f"secret {name!r} exists", True, why)
        except Exception as exc:
            check(
                f"secret {name!r} exists",
                False,
                f"{type(exc).__name__}: create it with "
                f"`uv run modal secret create {name} ...` -- needed for {why}",
            )
    if failures:
        print("\ncannot continue without both secrets", file=sys.stderr)
        return 1

    # Start from a clean status Dict so a stale key from a previous run cannot
    # make a broken probe look like it passed.
    modal.Dict.objects.delete(dict_name(RUN_ID), allow_missing=True)
    status = open_dict(RUN_ID, create_if_missing=True)
    status.put("schema", STATUS_SCHEMA)

    app = modal.App.lookup(GOD_APP_NAME, create_if_missing=True)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    print("[2/6] GOD sandbox (god_image, both secrets, NO volume mount)")
    sandbox = None
    self_terminated = False
    try:
        sandbox = modal.Sandbox.create(
            app=app,
            image=god_image(),
            secrets=[
                modal.Secret.from_name(ANTHROPIC_SECRET_NAME),
                modal.Secret.from_name(MODAL_TOKEN_SECRET_NAME),
            ],
            # Deliberately no `volumes=`, matching godbox.launch. The volume is
            # reached through the client API from inside instead.
            timeout=600,
            idle_timeout=IDLE_TIMEOUT_S,
            workdir=GOD_WORKDIR,
            cpu=1.0,
            memory=2048,
        )
        check("Sandbox.create with god_image", True, sandbox.object_id)

        print("[3/6] the image contains what GOD needs")
        for label, module in (
            ("godbox.entrypoint imports", "godbox.entrypoint"),
            ("reagents.god.orchestrator imports", "reagents.god.orchestrator"),
            ("sandbox_runtime imports", "reagents.demigod.sandbox_runtime"),
            ("demigod.spawn imports", "demigod.spawn"),
            ("anthropic SDK present", "anthropic"),
            ("modal client present", "modal"),
        ):
            proc = sandbox.exec("python", "-c", f"import {module}", timeout=120)
            proc.wait()
            check(label, proc.returncode == 0, proc.stderr.read().strip()[-200:])

        # Credentials must actually authenticate, not merely be present. A
        # token that is set but rejected fails identically to no token at all,
        # 20 minutes into a run.
        proc = sandbox.exec(
            "python",
            "-c",
            "import modal; modal.App.lookup('god', create_if_missing=True); "
            "print('AUTH_OK')",
            timeout=120,
        )
        out = proc.stdout.read()
        proc.wait()
        check("mounted Modal token authenticates", "AUTH_OK" in out, out.strip()[:120])

        print("[4/6] the REAL entrypoint, run to a terminal status")
        _check_entrypoint(sandbox, check)

        print(
            f"[5/6] detached probe: sleeps {DETACHED_SLEEP_S}s with NO client "
            f"attached, against a {IDLE_TIMEOUT_S}s idle_timeout"
        )
        sandbox.filesystem.write_text(PROBE, "/god/probe.py")
        started = time.monotonic()
        sandbox.exec(
            "python",
            "-u",
            "/god/probe.py",
            env={"PREFLIGHT_SANDBOX_ID": sandbox.object_id},
            # DEVNULL, not the default PIPE. An unread PIPE applies backpressure
            # and eventually hangs the very process we are trying to abandon.
            stdout=StreamType.DEVNULL,
            stderr=StreamType.DEVNULL,
        )
        check("exec returned without blocking", time.monotonic() - started < 10)

        print("      polling the status Dict from OUTSIDE the container...")
        deadline = time.monotonic() + POLL_TIMEOUT_S
        seen = False
        while time.monotonic() < deadline:
            if status.get("from_inside"):
                seen = True
                break
            time.sleep(3)
        elapsed = int(time.monotonic() - started)
        check(
            "modal.Dict written from inside is readable outside",
            seen,
            f"after {elapsed}s",
        )
        check(
            f"detached process survived past idle_timeout ({IDLE_TIMEOUT_S}s)",
            seen and elapsed >= DETACHED_SLEEP_S,
            f"reported at {elapsed}s",
        )

        print("[6/6] artifacts and teardown")
        # No `volume.reload()` here: that is a container-side API and raises
        # `RuntimeError: reload() can only be called from within a running
        # function` on a laptop. A client `listdir` is already fresh.
        try:
            entries = [e.path for e in volume.listdir(GOD_OUT_SUBDIR, recursive=True)]
        except Exception as exc:
            entries = []
            check("volume listdir", False, f"{type(exc).__name__}: {exc}")
        check(
            "uploaded artifact is visible outside, while the sandbox still runs",
            any(e.endswith("probe.json") for e in entries),
            f"{entries}",
        )
        if entries:
            body = b"".join(volume.read_file(f"{GOD_OUT_SUBDIR}/probe.json")).decode()
            check("artifact contents survived the upload", "PREFLIGHT_ARTIFACT" in body)

        # Self-terminate. The marker is what makes this check mean anything:
        # `poll() is not None` alone is also true of a sandbox reaped by
        # idle_timeout, and an earlier version of this script scored exactly
        # that as a pass while the probe had in fact crashed.
        announced = status.get("about_to_terminate")
        check("probe reached its own teardown", announced is not None)
        time.sleep(5)
        self_terminated = sandbox.poll() is not None
        check(
            "sandbox terminated itself from inside",
            self_terminated and announced is not None,
            "" if self_terminated else "still running 5s after the announcement",
        )

    except Exception as exc:
        check("GOD sandbox preflight", False, f"{type(exc).__name__}: {exc}")
    finally:
        # Unconditional. If the self-terminate worked this is a no-op; if it did
        # not, this is the difference between cents and an hour of billing.
        if sandbox is not None and not self_terminated:
            sandbox.terminate()
            print("  (sandbox terminated by the preflight)")
        modal.Dict.objects.delete(dict_name(RUN_ID), allow_missing=True)

    if failures:
        print(f"\n{len(failures)} FAILURE(S): {failures}", file=sys.stderr)
        print(
            "Read the failure before assuming the design is wrong. An import "
            "failure is a missing pin in godbox/images.py. A Dict that never "
            "appears means the detached process died -- check `modal app logs "
            f"{GOD_APP_NAME}`. A missing artifact with a live Dict means the "
            "volume commit is the broken half, and GOD's answer would still be "
            "readable from the status channel.",
            file=sys.stderr,
        )
        return 1

    print("\nGOD can live in its own sandbox: image complete, both channels")
    print("work, a detached run outlives its client, and teardown is self-served.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
