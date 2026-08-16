"""Can a sandbox holding a Modal token spawn ANOTHER sandbox and read its result?

    uv run python scripts/preflight_nested.py

This is the load-bearing assumption behind putting GOD in its own Modal sandbox:
GOD must be able to create DEMI_GOD sandboxes from *inside* a sandbox, and read
their manifests back. If nested spawning does not work, the three-sandbox
topology (GOD / DEMI_GOD / BROKER) needs a different shape -- e.g. GOD running
as a Modal Function, or a thin local shim that spawns on GOD's behalf.

Proving it costs two small sandboxes and zero Anthropic tokens. It is deliberately
NOT part of preflight_live.py: that one gates every spawn, this one gates a design
decision and only needs re-running when the topology or the Modal SDK changes.

SECURITY NOTE, and it is the whole reason this is a separate script: making this
work requires mounting a Modal token INSIDE a sandbox. Modal credentials are
workspace-wide -- whoever holds them can spawn sandboxes, list volumes, and read
any output in the workspace. That is acceptable for GOD, which is the trusted
orchestrator. It is NOT acceptable for a DEMI_GOD, which must never hold Modal
credentials; a demigod reaches tools through the broker with a capability lease
and nothing else. Do not copy this pattern into the demigod path.
"""

from __future__ import annotations

import sys

from demigod.images import resolve_image
from demigod.runner.inside import MODAL_APP_NAME

MODAL_TOKEN_SECRET_NAME = "demigod-modal-token"
"""Modal Secret carrying MODAL_TOKEN_ID / MODAL_TOKEN_SECRET for the GOD sandbox.

TODO: create once per workspace, from the caller's own token --
    uv run modal secret create demigod-modal-token \\
        MODAL_TOKEN_ID=ak-... MODAL_TOKEN_SECRET=as-...
"""

# Runs INSIDE the outer sandbox. Creates an inner sandbox, execs a trivial
# command in it, reads the output back, and terminates it -- the same four
# operations InsideSandboxRunner performs.
INNER_SCRIPT = """
import modal

app = modal.App.lookup("nested-preflight", create_if_missing=True)
img = modal.Image.debian_slim(python_version="3.12")

sb = modal.Sandbox.create(app=app, image=img, timeout=120, cpu=0.5, memory=512)
try:
    print("INNER_SANDBOX_ID", sb.object_id, flush=True)
    p = sb.exec("sh", "-c", "echo HELLO_FROM_INNER > /tmp/r.txt && cat /tmp/r.txt")
    out = p.stdout.read().strip()
    p.wait()
    print("INNER_EXEC_OUT", out, flush=True)
    # The filesystem API is what the real runner uses to read result.json back.
    got = sb.filesystem.read_text("/tmp/r.txt").strip()
    print("INNER_FS_READ", got, flush=True)
finally:
    sb.terminate()
    print("INNER_TERMINATED", flush=True)
"""


def main() -> int:
    import modal

    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(
            f"  {'PASS' if ok else 'FAIL'}  {label}{f' | {detail}' if detail else ''}"
        )
        if not ok:
            failures.append(label)

    print("[1/3] modal token secret")
    try:
        modal.Secret.from_name(MODAL_TOKEN_SECRET_NAME).hydrate()
        check(f"secret {MODAL_TOKEN_SECRET_NAME!r} exists", True)
    except Exception as e:
        check(
            f"secret {MODAL_TOKEN_SECRET_NAME!r} exists",
            False,
            f"{type(e).__name__}: create it with `uv run modal secret create "
            f"{MODAL_TOKEN_SECRET_NAME} MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...`",
        )
        print("\ncannot test nested spawning without it", file=sys.stderr)
        return 1

    print("[2/3] outer sandbox (stands in for GOD)")
    app = modal.App.lookup(MODAL_APP_NAME, create_if_missing=True)
    outer = None
    try:
        outer = modal.Sandbox.create(
            app=app,
            # The demigod image already carries the modal client via our package.
            image=resolve_image([]).build(),
            secrets=[modal.Secret.from_name(MODAL_TOKEN_SECRET_NAME)],
            timeout=300,
            idle_timeout=120,
            cpu=1.0,
            memory=2048,
        )
        check("outer Sandbox.create", True, outer.object_id)

        print("[3/3] nested spawn from inside the outer sandbox")
        outer.filesystem.write_text(INNER_SCRIPT, "/tmp/inner.py")
        proc = outer.exec("python", "/tmp/inner.py", timeout=240)
        out = proc.stdout.read()
        proc.wait()
        err = proc.stderr.read()

        for line in out.splitlines():
            print(f"      [inner] {line}")

        check("inner sandbox created", "INNER_SANDBOX_ID" in out)
        check("inner exec ran", "HELLO_FROM_INNER" in out)
        check("inner filesystem read back", "INNER_FS_READ HELLO_FROM_INNER" in out)
        check("inner terminated", "INNER_TERMINATED" in out)
        check("nested script exit 0", proc.returncode == 0, err.strip()[:400])
    except Exception as e:
        check("outer sandbox / nested spawn", False, f"{type(e).__name__}: {e}")
    finally:
        if outer is not None:
            outer.terminate()
            print("  (outer sandbox terminated)")

    if failures:
        print(f"\n{len(failures)} FAILURE(S): {failures}", file=sys.stderr)
        print(
            "Nested spawning does not work as assumed. GOD-in-a-sandbox needs a "
            "different shape -- consider GOD as a Modal Function, or a local shim "
            "that spawns demigods on GOD's behalf.",
            file=sys.stderr,
        )
        return 1
    print("\nnested spawning works - GOD can live in its own sandbox")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
