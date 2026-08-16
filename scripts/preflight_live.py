"""Exercise every piece of live Modal infrastructure a spawn touches -- WITHOUT
running an agent. One small sandbox, zero Anthropic tokens.

    uv run python scripts/preflight_live.py

WHY THIS EXISTS. The offline test suite cannot catch a server-side API change,
and checking `hasattr(modal.Sandbox, "open")` proves only that the *client* has
a method -- not that the *server* still honours it. That gap shipped a runner
which created a sandbox successfully and then died on the next call with:

    ConflictError: The legacy Sandbox filesystem API is no longer supported.

Everything below is a call the real runner makes, in the order it makes it. If
this passes, a spawn can only fail inside the agent loop itself -- which is the
line worth paying Anthropic tokens to cross.

Run it after any Modal SDK upgrade, and before the first spawn of the day.
"""

from __future__ import annotations

import sys
from pathlib import Path

from demigod.egress import allowlist
from demigod.images import resolve_image
from demigod.layout import OUT_MOUNT, SHARED_MOUNT, SPEC_PATH, RunLayout
from demigod.result import RESULT_FILENAME, DemiGodResult
from demigod.runner.inside import AGENT_ENV, ANTHROPIC_SECRET_NAME, MODAL_APP_NAME
from demigod.spec import DemiGodSpec, Problem

RUN_ID = "preflight"
NAME = "preflight-probe"


def _spec() -> DemiGodSpec:
    return DemiGodSpec(
        name=NAME,
        domain="infrastructure probe",
        tools=["pandas"],
        problem=Problem(context="none", goal="none"),
        max_lifetime_s=300,
        idle_timeout_s=120,
        cpu=1.0,
        memory_mb=2048,
    )


def main() -> int:
    import modal

    spec = _spec()
    layout = RunLayout(run_id=RUN_ID, demigod_name=NAME)
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(
            f"  {'PASS' if ok else 'FAIL'}  {label}{f' | {detail}' if detail else ''}"
        )
        if not ok:
            failures.append(label)

    # 1. The secret must exist. Missing it is the single most common setup
    #    failure, and it otherwise surfaces only after a sandbox is billed.
    print("[1/5] secret")
    try:
        modal.Secret.from_name(ANTHROPIC_SECRET_NAME).hydrate()
        check(f"secret {ANTHROPIC_SECRET_NAME!r} exists", True)
    except Exception as e:
        check(
            f"secret {ANTHROPIC_SECRET_NAME!r} exists",
            False,
            f"{type(e).__name__}: create it with "
            f"`uv run modal secret create {ANTHROPIC_SECRET_NAME} "
            f"ANTHROPIC_API_KEY=sk-ant-...`",
        )

    # 2. Seeding shared/ -- the only path by which input data reaches an agent.
    print("[2/5] volume seeding")
    probe = Path(__file__).parent.parent / "examples" / "data" / "transactions.csv"
    try:
        uploaded = layout.seed_shared([probe]) if probe.exists() else []
        check("seed_shared uploads", bool(uploaded), f"{uploaded}")
    except Exception as e:
        check("seed_shared uploads", False, f"{type(e).__name__}: {e}")

    print("[3/5] sandbox create (two volumes, one read-only)")
    app = modal.App.lookup(MODAL_APP_NAME, create_if_missing=True)
    sandbox = None
    try:
        sandbox = modal.Sandbox.create(
            app=app,
            image=resolve_image(spec.tools).build(),
            volumes=layout.sandbox_volumes(),
            # Mounted so the live `claude -p` check below can actually reach the
            # API -- exactly as the real runner does.
            secrets=[modal.Secret.from_name(ANTHROPIC_SECRET_NAME)],
            timeout=spec.max_lifetime_s,
            idle_timeout=spec.idle_timeout_s,
            workdir=OUT_MOUNT,
            cpu=spec.cpu,
            memory=spec.memory_mb,
            outbound_domain_allowlist=allowlist(),
        )
        check("Sandbox.create", True, sandbox.object_id)

        # 4. THE ONE THAT BROKE. Both directions of the filesystem API, exactly
        #    as _write_spec and _read_result_json use them.
        print("[4/5] filesystem API (write_text / read_text)")
        try:
            sandbox.filesystem.write_text(spec.model_dump_json(), SPEC_PATH)
            check("filesystem.write_text (spec.json)", True)
        except Exception as e:
            check("filesystem.write_text", False, f"{type(e).__name__}: {e}")

        try:
            echoed = sandbox.filesystem.read_text(SPEC_PATH)
            check("filesystem.read_text round-trip", NAME in echoed)
        except Exception as e:
            check("filesystem.read_text", False, f"{type(e).__name__}: {e}")

        # Absent file must return None, not explode -- _collect depends on it.
        try:
            sandbox.filesystem.read_text(f"{OUT_MOUNT}/does-not-exist.json")
            check("missing file raises (so _read_result_json -> None)", False)
        except Exception:
            check("missing file raises (so _read_result_json -> None)", True)

        print("[5/5] mounts + entrypoint importability")
        for label, cmd in [
            ("shared/ readable", ["ls", SHARED_MOUNT]),
            (
                "shared/ read-only",
                [
                    "sh",
                    "-c",
                    f"touch {SHARED_MOUNT}/x 2>/dev/null && echo BAD || echo ok",
                ],
            ),
            ("out/ writable", ["sh", "-c", f"touch {OUT_MOUNT}/x && echo ok"]),
            ("claude CLI present", ["claude", "--version"]),
            (
                "entrypoint imports",
                ["python", "-c", "import demigod.entrypoint; print('ok')"],
            ),
            (
                "Anthropic key mounted",
                ["sh", "-c", 'test -n "$ANTHROPIC_API_KEY" && echo ok'],
            ),
        ]:
            p = sandbox.exec(*cmd, timeout=60)
            out = p.stdout.read().strip()
            p.wait()
            ok = p.returncode == 0 and "BAD" not in out
            check(label, ok, out[:80] or p.stderr.read().strip()[:80])

        # The CLI *running* is a different question from the CLI *existing*.
        # `claude --version` passed happily on an image where every real query
        # died, because Modal runs as root and the CLI refuses
        # --dangerously-skip-permissions under root. Costs a handful of tokens;
        # cheaper than a failed spawn.
        env_flags = " ".join(f"{k}={v}" for k, v in AGENT_ENV.items())
        p = sandbox.exec(
            "sh",
            "-c",
            f"cd /tmp && {env_flags} DISABLE_AUTOUPDATER=1 "
            "timeout 90 claude -p 'reply with exactly: PONG' "
            "--output-format text --dangerously-skip-permissions < /dev/null",
            timeout=120,
        )
        out = p.stdout.read().strip()
        err = p.stderr.read().strip()
        p.wait()
        detail = f"rc={p.returncode} stdout={out[:120]!r} stderr={err[:240]!r}"
        check("claude CLI can actually answer (as root)", "PONG" in out, detail)

        # The agent writes result.json; prove the runner can read one back the
        # same way _collect will.
        try:
            demo = DemiGodResult(claim="probe", confidence=1.0, method="probe")
            sandbox.filesystem.write_text(
                demo.model_dump_json(), f"{OUT_MOUNT}/{RESULT_FILENAME}"
            )
            raw = sandbox.filesystem.read_text(f"{OUT_MOUNT}/{RESULT_FILENAME}")
            check("result.json write+read as _collect does", "probe" in raw)
        except Exception as e:
            check("result.json write+read", False, f"{type(e).__name__}: {e}")

    except Exception as e:
        check("Sandbox.create", False, f"{type(e).__name__}: {e}")
    finally:
        if sandbox is not None:
            sandbox.terminate()
            print("  (sandbox terminated)")

    if failures:
        print(f"\n{len(failures)} FAILURE(S): {failures}", file=sys.stderr)
        return 1
    print("\nall live preflight checks passed - safe to spawn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
