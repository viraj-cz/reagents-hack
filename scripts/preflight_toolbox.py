"""Live check of every Modal primitive the TOOLBOX_BROKER depends on.

    uv run python scripts/preflight_toolbox.py

Sibling of `scripts/preflight_live.py`, and it exists for the same reason: the
offline suite cannot catch a server-side API change, and `hasattr` proves only
that the client has a method. This repo already shipped a runner that created a
sandbox successfully and then died on `sandbox.open()` because the SERVER had
retired it.

Cheap on purpose. One ephemeral App with ONE small web function, one small
sandbox, zero Anthropic tokens, no agent loop. It does NOT bring up
`broker.service` -- that App carries executor classes whose images install SciPy
and Biopython, and building them to check a URL would cost minutes and real
money for no extra information. What it checks instead is the exact mechanism
those functions rely on:

  1. modal.Dict          the grant store
  2. modal.Queue         the trace, partitioned, counted by length, read
                         WITHOUT mutation -- the load-bearing property
  3. modal.asgi_app      serving a plain ASGI callable with no web framework
  4. the wire protocol   end to end, over a real modal.run URL, with a real lease
  5. outbound_domain_allowlist   is it actually enforced, or merely accepted?

Run it after any Modal SDK bump, and before the first brokered spawn of the day.
"""

from __future__ import annotations

import sys

import modal

from broker.grants import InMemoryGrantStore, TraceEntry, mint_grant
from broker.modal_store import ModalGrantStore, _partition
from broker.router import ToolboxRouter
from demigod.egress import allowlist
from demigod.toolbox.client import ToolboxClient
from demigod.toolbox.protocol import ErrorCode
from reagents.contracts import Budget
from reagents.tools.registry import Tool, ToolRegistry

APP_NAME = "toolbox-preflight"
PROBE_TOOL = "probe.echo"

# FIXED names, deliberately. The router runs in a container and constructs its
# own store from these constants; a randomized name generated on this laptop
# would not reach it, and the endpoint would answer 401 for every lease while
# looking perfectly healthy. (That is exactly what the first live run did.)
# Collisions between concurrent preflights are avoided by the lease id, which
# is a fresh uuid every run.
GRANTS_NAME = "toolbox-preflight-grants"
TRACE_NAME = "toolbox-preflight-trace"


def preflight_store() -> ModalGrantStore:
    return ModalGrantStore(grants_name=GRANTS_NAME, trace_name=TRACE_NAME)


failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' | {detail}' if detail else ''}")
    if not ok:
        failures.append(label)


def probe_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            id=PROBE_TOOL,
            namespace="probe",
            description="Echo a value back. Costs nothing, proves the path.",
            parameters_schema={
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "string"}},
            },
            fn=lambda value: {"echoed": value},
        )
    )
    return registry


# --- 3+4: a throwaway web endpoint serving the real router -------------------
#
# Same image builder and same decorator as `broker.service.router`, minus the
# executor classes. If this serves, that serves.

app = modal.App(APP_NAME)


@app.function(
    image=modal.Image.debian_slim(python_version="3.12")
    .pip_install("pydantic==2.13.4")
    .add_local_python_source("reagents", "demigod", "broker", ignore=[]),
    timeout=300,
    max_containers=1,
    serialized=True,
)
@modal.asgi_app()
def probe_router():
    """The real ToolboxRouter over the real Modal grant store."""
    return ToolboxRouter(registry=probe_registry(), store=preflight_store())


def check_dict_and_queue() -> str | None:
    """1 + 2. Returns a published lease id for the endpoint check, or None."""
    print("[1/5] modal.Dict -- the grant store")
    store = preflight_store()
    lease = probe_registry().mint_lease(
        [PROBE_TOOL],
        subject_id="preflight",
        budget=Budget(max_tool_calls=4, wall_time_s=900.0),
    )
    try:
        store.publish(mint_grant(lease, label="preflight"))
        echoed = store.read(lease.lease_id)
        check(
            "Dict publish/read round-trips a Grant",
            echoed is not None and echoed.lease.tool_ids == [PROBE_TOOL],
        )
    except Exception as e:
        check("Dict publish/read", False, f"{type(e).__name__}: {e}")
        return None

    print("[2/5] modal.Queue -- the trace, and the call counter")
    try:
        store.record(TraceEntry(tool=PROBE_TOOL, result=1), lease_id=lease.lease_id)
        store.record(TraceEntry(tool="denied", metered=False), lease_id=lease.lease_id)
        check("Queue.put accepts a partitioned entry", True)

        # THE load-bearing property. `calls_used` is a partition length, so a
        # destructive read would silently refund the lease every time GOD looked
        # at the trace. Verified by reading twice.
        first = store.trace(lease.lease_id, include_refused=True)
        second = store.trace(lease.lease_id, include_refused=True)
        check(
            "Queue.iterate is NON-mutating (read twice, same length)",
            len(first) == len(second) == 2,
            f"{len(first)} then {len(second)}",
        )
        check(
            "refusals live in their own partition (do not charge the lease)",
            store.calls_used(lease.lease_id) == 1,
            f"calls_used={store.calls_used(lease.lease_id)}",
        )
        check(
            "partition keys stay inside Modal's 64-byte limit",
            len(_partition(lease.lease_id).encode()) <= 64,
        )
    except Exception as e:
        check("Queue trace", False, f"{type(e).__name__}: {e}")

    return lease.lease_id


def check_endpoint(lease_id: str) -> None:
    """3 + 4. Serve the router ephemerally and drive it as a demigod would."""
    print("[3/5] modal.asgi_app -- serving a framework-free ASGI callable")
    print("[4/5] the wire protocol, end to end, over a real URL")
    with modal.enable_output(), app.run():
        url = probe_router.get_web_url()
        check("web URL exists", bool(url), url or "")
        if not url:
            return

        client = ToolboxClient(url, lease_id, timeout_s=120)
        try:
            check("GET /v1/health (unauthenticated)", client.health()["ok"] is True)
        except Exception as e:
            check("GET /v1/health", False, f"{type(e).__name__}: {e}")
            return

        try:
            listing = client.list_tools()
            check(
                "GET /v1/tools returns the leased catalog",
                [t.id for t in listing.tools] == [PROBE_TOOL],
                f"{listing.lease.calls_remaining} calls left",
            )
        except Exception as e:
            check("GET /v1/tools", False, f"{type(e).__name__}: {e}")

        response = client.call(PROBE_TOOL, {"value": "ping"})
        check(
            "POST .../call executes the tool",
            response.ok and response.result == {"echoed": "ping"},
            str(response.error.message if response.error else response.result),
        )

        bad = client.call(PROBE_TOOL, {"wrong": "key"})
        check(
            "server-side schema validation rejects bad input",
            not bad.ok and bad.error.code == ErrorCode.INVALID_INPUT,
            bad.error.code if bad.error else "accepted!",
        )

        denied = ToolboxClient(url, "lease_forged", timeout_s=60).call(
            PROBE_TOOL, {"value": "x"}
        )
        check(
            "a forged lease is refused",
            not denied.ok and denied.error.code == ErrorCode.UNAUTHORIZED,
            denied.error.code if denied.error else "ACCEPTED -- STOP",
        )


def check_egress() -> None:
    """5. Does `outbound_domain_allowlist` actually block, or just parse?

    The claim being tested is 'a demigod cannot read outside its isolated
    system'. If Modal accepted the argument and ignored it, every offline test
    would still pass and the claim would be false.
    """
    print("[5/5] Sandbox outbound_domain_allowlist -- accepted AND enforced?")
    probe_app = modal.App.lookup(APP_NAME, create_if_missing=True)
    sandbox = None
    try:
        sandbox = modal.Sandbox.create(
            "sleep",
            "infinity",
            app=probe_app,
            image=modal.Image.debian_slim(python_version="3.12"),
            timeout=180,
            idle_timeout=120,
            outbound_domain_allowlist=allowlist("https://api.anthropic.com"),
        )
        check(
            "Sandbox.create accepts outbound_domain_allowlist", True, sandbox.object_id
        )

        # An allowed host must CONNECT. A 401/405 from the API is a pass: it
        # means TLS completed and the request was answered.
        reach = (
            "import urllib.request,urllib.error,sys;"
            "u=sys.argv[1];"
            "\ntry:\n urllib.request.urlopen(u,timeout=15); print('CONNECTED')\n"
            "except urllib.error.HTTPError: print('CONNECTED')\n"
            "except Exception as e: print('BLOCKED', type(e).__name__)\n"
        )
        for label, target, want in [
            (
                "allowed host reachable",
                "https://api.anthropic.com/v1/messages",
                "CONNECTED",
            ),
            (
                "host NOT on the allowlist is blocked",
                "https://pypi.org/simple/",
                "BLOCKED",
            ),
        ]:
            proc = sandbox.exec("python", "-c", reach, target, timeout=60)
            out = proc.stdout.read().strip()
            proc.wait()
            check(label, out.startswith(want), out[:80] or proc.stderr.read()[:80])
    except Exception as e:
        check("outbound_domain_allowlist", False, f"{type(e).__name__}: {e}")
    finally:
        if sandbox is not None:
            sandbox.terminate()
            print("  (sandbox terminated)")


def check_store_parity() -> None:
    """The in-memory store and the Modal one must answer the same questions.

    Free, and it belongs here rather than in the offline suite: the offline
    suite can only ever see one of the two implementations.
    """
    from broker.grants import GrantStore

    for implementation in (InMemoryGrantStore(), ModalGrantStore()):
        check(
            f"{type(implementation).__name__} satisfies GrantStore",
            isinstance(implementation, GrantStore),
        )


def main() -> int:
    check_store_parity()
    lease_id = check_dict_and_queue()
    if lease_id:
        check_endpoint(lease_id)
    check_egress()

    if failures:
        print(f"\n{len(failures)} FAILURE(S): {failures}", file=sys.stderr)
        return 1
    print("\nall live toolbox-broker checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
