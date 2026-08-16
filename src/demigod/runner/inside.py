"""Runner: agent loop INSIDE the sandbox. Current default. PROVISIONAL.

Shape:

    create Sandbox (pre-baked image, run volume mounted, hard timeouts)
      -> write spec.json into it
      -> exec `python -m demigod.entrypoint --spec /run/spec.json`
      -> stream logs out
      -> read back out/<name>/result.json
      -> terminate

Why inside, for now: the agent's Bash and Write calls land directly on the
sandbox filesystem with no marshalling, so tool latency is process-local and
the blast radius of anything the model does is the sandbox. The cost is that
the Anthropic API key has to be present inside the sandbox, and that the loop
dies with the sandbox.

See `demigod/runner/__init__.py` for what swapping this out costs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from demigod.images import required_secret_names, resolve_image
from demigod.layout import OUT_MOUNT, SPEC_PATH, RunLayout
from demigod.result import RESULT_FILENAME, DemiGodResult
from demigod.spec import DemiGodSpec
from demigod.toolbox.client import DEFAULT_GRANT_FILE, ENV_LEASE, ENV_URL

if TYPE_CHECKING:
    import modal

MODAL_APP_NAME = "demigod"
"""All DEMI_GOD sandboxes live under one Modal App, so `modal app logs demigod`
shows the whole fleet."""

AGENT_ENV = {"IS_SANDBOX": "1"}
"""Environment for the in-sandbox agent process.

`IS_SANDBOX=1` is REQUIRED. Modal sandboxes run as root, and the CLI refuses
`--dangerously-skip-permissions` (which is what `permission_mode=
"bypassPermissions"` maps to) under root:

    --dangerously-skip-permissions cannot be used with root/sudo privileges
    for security reasons

That refusal surfaces through the SDK as a bare `ProcessError: Command failed
with exit code 1 / Error output: Check stderr output for details`, which names
neither the flag nor root -- so it is worth stating plainly here.

`IS_SANDBOX=1` is the intended escape hatch, and it is honest: the sandbox IS
the isolation boundary in this design, which is why the loop bypasses
permissions in the first place. The alternative -- `useradd` a non-root user in
the image and `su` to it -- also works (verified), but it fights the root-owned
volume mounts for no isolation we do not already have.

Purely a consequence of running the loop INSIDE the sandbox. Deleted along with
this module if the loop moves out.
"""

ANTHROPIC_SECRET_NAME = "demigod-anthropic"
"""Modal Secret holding ANTHROPIC_API_KEY for the in-sandbox agent loop.
TODO: create it once per Modal workspace --
    modal secret create demigod-anthropic ANTHROPIC_API_KEY=sk-ant-...
This exists only because the loop runs inside; an outside runner reads the key
from the caller's environment and this constant disappears."""


class InsideSandboxRunner:
    """One `modal.Sandbox` per DEMI_GOD, alive for the whole task."""

    name = "inside"

    def run(self, spec: DemiGodSpec, run_id: str) -> DemiGodResult:
        import modal

        layout = RunLayout(run_id=run_id, demigod_name=spec.name)
        image = resolve_image(spec.tools)

        # Tool secrets are resolved by name from the registry. Missing ones fail
        # at sandbox creation with Modal's own NotFoundError, which names the
        # secret -- better than an auth error 20 minutes into the run.
        secrets = [modal.Secret.from_name(ANTHROPIC_SECRET_NAME)]
        for env_name in required_secret_names(spec.tools):
            secrets.append(modal.Secret.from_name(_secret_name_for(env_name)))

        app = modal.App.lookup(MODAL_APP_NAME, create_if_missing=True)

        print(
            f"[demigod] spawning {spec.name!r} "
            f"image={image.name} tools={spec.tools or '[]'} run={run_id}"
        )

        sandbox: modal.Sandbox | None = None
        try:
            # ONE sandbox for this agent's whole task -- not per-call, not
            # pooled. `timeout` is the wall-clock cap; `idle_timeout` is the
            # runaway-cost guarantee (Modal kills it after that long with no
            # active command, stdin write, or TCP connection).
            sandbox = modal.Sandbox.create(
                app=app,
                image=image.build(),
                volumes=layout.sandbox_volumes(),
                secrets=secrets,
                timeout=spec.max_lifetime_s,
                idle_timeout=spec.idle_timeout_s,
                workdir=OUT_MOUNT,
                cpu=spec.cpu,
                memory=spec.memory_mb,
                # Isolation as a network fact rather than a prompt instruction.
                # None (the default) means unrestricted, which is what this
                # runner did before the broker landed; a list pins egress to the
                # agent API plus the broker. See demigod/egress.py.
                #
                # `block_network=True` remains wrong here: the loop is INSIDE,
                # so blocking everything blocks the agent itself.
                outbound_domain_allowlist=spec.egress_domains,
                verbose=True,
            )
            print(f"[demigod] sandbox {sandbox.object_id} up")
            if spec.egress_domains is not None:
                print(f"[demigod] egress pinned to {spec.egress_domains}")

            _write_spec(sandbox, spec)
            _write_toolbox_grant(sandbox, spec)
            returncode = _exec_agent(sandbox, spec)

            return _collect(sandbox, spec, returncode, run_id)
        finally:
            if sandbox is not None:
                # Unconditional. A leaked sandbox bills until max_lifetime_s.
                sandbox.terminate()
                print(f"[demigod] sandbox terminated ({spec.name})")


def _secret_name_for(env_var: str) -> str:
    """Map a required env var to its Modal Secret name.

    Convention: MY_TOOL_API_KEY -> `demigod-my-tool-api-key`. One env var per
    Secret keeps least-privilege easy -- a sandbox only mounts the credentials
    its tools declared.
    """
    return "demigod-" + env_var.lower().replace("_", "-")


def _write_spec(sandbox: modal.Sandbox, spec: DemiGodSpec) -> None:
    """Materialize spec.json inside the sandbox via the filesystem API.

    Uses `sandbox.filesystem`, NOT the legacy `sandbox.open()`. The legacy API
    is retired server-side and now fails with:
        ConflictError: The legacy Sandbox filesystem API is no longer supported.
    Note the argument order: data first, remote path second.
    """
    sandbox.filesystem.write_text(spec.model_dump_json(indent=2), SPEC_PATH)


def _write_toolbox_grant(sandbox: modal.Sandbox, spec: DemiGodSpec) -> None:
    """Materialize the broker grant at a stable path inside the sandbox.

    Belt and braces with the env vars in `_agent_env`. The agent composes bash,
    and bash composes sub-shells, heredocs and `env -i` -- any of which can lose
    an exported variable, and the failure mode is an agent that concludes it has
    no tools. A file cannot be lost that way.

    Written outside both volume mounts, next to spec.json, for the reason
    SPEC_PATH gives: it is control-plane data, and a credential in the agent's
    output dir would be committed to a volume and listed in `files`.
    """
    if spec.toolbox is None:
        return
    sandbox.filesystem.write_text(
        spec.toolbox.model_dump_json(indent=2), DEFAULT_GRANT_FILE
    )
    print(f"[demigod] toolbox lease {spec.toolbox.lease_id} -> {spec.toolbox.base}")


def _agent_env(spec: DemiGodSpec) -> dict[str, str]:
    """Environment for the agent process. AGENT_ENV plus the broker coordinates.

    Note what is NOT here and must never be: a Modal token. Modal credentials
    are workspace-wide, so a demigod holding one could spawn sandboxes and read
    every sibling's output volume -- which is the isolation this whole design
    exists to provide.
    """
    env = dict(AGENT_ENV)
    if spec.toolbox is not None:
        env[ENV_URL] = spec.toolbox.base
        env[ENV_LEASE] = spec.toolbox.lease_id
    return env


def _exec_agent(sandbox: modal.Sandbox, spec: DemiGodSpec) -> int:
    """Run the in-sandbox entrypoint, streaming its logs to our stdout.

    Streaming matters more than it looks: this is the only visibility into a
    DEMI_GOD while it works, and a silent 40-minute sandbox is indistinguishable
    from a hung one.
    """
    proc = sandbox.exec(
        "python",
        "-m",
        "demigod.entrypoint",
        "--spec",
        SPEC_PATH,
        workdir=OUT_MOUNT,
        timeout=spec.max_lifetime_s,
        env=_agent_env(spec),
    )
    for line in proc.stdout:
        print(f"[{spec.name}] {line}", end="")
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read()
        print(f"[{spec.name}] exited {proc.returncode}\n{stderr}")
    return proc.returncode


def _collect(
    sandbox: modal.Sandbox, spec: DemiGodSpec, returncode: int, run_id: str
) -> DemiGodResult:
    """Read back the manifest, stamping the envelope fields ourselves.

    The agent authors claim/confidence/payload/... ; the runner owns
    demigod_name/domain_name/run_id/status/error. Overwriting them here means a
    model cannot self-report `status="ok"` on a run that crashed.
    """
    envelope = {
        "demigod_name": spec.name,
        "domain_name": spec.domain_name or spec.name,
        "run_id": run_id,
    }
    raw = _read_result_json(sandbox)

    if raw is None:
        # 137 = SIGKILL, which is what Modal uses for timeout/idle termination.
        status = "timeout" if returncode in (124, 137) else "failed"
        return DemiGodResult.failure(
            status=status,
            error=f"agent exited {returncode} without writing {OUT_MOUNT}/result.json",
            **envelope,
        )

    try:
        result = DemiGodResult.model_validate(json.loads(raw))
    except Exception as e:
        # Artifacts still exist on the volume; only the index is broken. Say so
        # rather than discarding a run's worth of work.
        return DemiGodResult.failure(
            status="failed",
            error=(
                f"result.json failed validation ({e}). Artifacts remain in "
                f"out/{spec.name}/."
            ),
            **envelope,
        )

    for field, value in envelope.items():
        setattr(result, field, value)
    result.status = "ok" if returncode == 0 else "failed"
    if returncode != 0:
        result.error = f"agent exited {returncode} but wrote a manifest"
    return result


def _read_result_json(sandbox: modal.Sandbox) -> str | None:
    """Read out/result.json from inside the sandbox, or None if absent.

    Read *before* the sandbox is terminated. Reading it off the volume
    afterwards would race Modal's commit semantics -- writes are not visible
    outside the writing container until committed, and "the agent produced
    nothing" is a miserable bug to diagnose.
    """
    try:
        return sandbox.filesystem.read_text(f"{OUT_MOUNT}/{RESULT_FILENAME}")
    except Exception:
        return None


# Keep the type checker honest: this class must satisfy the Protocol.
if TYPE_CHECKING:
    from demigod.runner import Runner

    _: Runner = InsideSandboxRunner()
