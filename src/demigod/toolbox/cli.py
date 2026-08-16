"""`toolbox` -- the DEMI_GOD-facing surface of the broker.

WHY A CLI AND NOT A TOOL SCHEMA
-------------------------------
The obvious alternative is to inject every brokered tool into the agent's tool
list as a typed schema. Two reasons not to:

1. Agents compose bash far better than they emit conforming JSON against a
   schema they were shown once, several thousand tokens ago. `toolbox call x
   --input in.json` is a shell command; it can be built up, echoed, retried,
   piped, and saved as an artifact.
2. Every `parameters_schema` in the agent's tool list is permanent context. A
   demigod with four brokered tools would carry four JSON Schemas in its system
   prompt for the whole run. Behind a CLI it carries one paragraph and fetches a
   schema with `toolbox describe` only when it is about to use one.

The CLI is also the audit surface: every invocation is one HTTP call the broker
sees, so `DemiGodResult.tool_trace` is authored by the thing being called rather
than by the thing being audited.

EXIT CODES (an agent branches on these):
    0  success
    1  the call was refused or the tool raised -- the message says which
    2  usage error on this machine (bad flags, unreadable input file)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from demigod.toolbox.client import (
    ENV_LEASE,
    ENV_URL,
    ToolboxClient,
    ToolboxError,
)
from demigod.toolbox.protocol import BrokeredTool, LeaseStatus

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_USAGE = 2

_EPILOG = """\
examples:
  toolbox list                              what you may call, and how many calls remain
  toolbox describe solve                    the tool's JSON Schema and what it returns
  echo '{"a":1}' | toolbox call solve -i -  call it, arguments on stdin
  toolbox call solve -i args.json -o out.json

Every call is metered against your capability lease. `toolbox list` shows how
many you have left. When they run out the broker refuses, and that is a
`blocker` to record -- not something to work around.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toolbox",
        description=(
            "Call the tools your capability lease grants you. The tools run "
            "elsewhere, on a broker; you hold only a lease."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default=None,
        help=f"Broker base URL. Defaults to ${ENV_URL} or the grant file.",
    )
    parser.add_argument(
        "--lease",
        default=None,
        help=f"Lease id, used as the bearer token. Defaults to ${ENV_LEASE}.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Per-request timeout.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    listp = sub.add_parser("list", help="List the tools this lease grants.")
    listp.add_argument("--json", action="store_true", help="Machine-readable output.")

    desc = sub.add_parser(
        "describe", help="Show one tool's input schema and description."
    )
    desc.add_argument("tool", help="Tool id, exactly as `toolbox list` prints it.")
    desc.add_argument("--json", action="store_true", help="Machine-readable output.")

    call = sub.add_parser("call", help="Execute one tool.")
    call.add_argument("tool", help="Tool id.")
    call.add_argument(
        "-i",
        "--input",
        default=None,
        metavar="FILE",
        help="JSON file of arguments. Use '-' for stdin. Omit for no arguments.",
    )
    call.add_argument(
        "--json-arg",
        default=None,
        metavar="JSON",
        help="Inline JSON object of arguments, instead of --input.",
    )
    call.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="FILE",
        help="Write the result here as well as to stdout. Keep it: it is an artifact.",
    )
    call.add_argument(
        "--raw",
        action="store_true",
        help="Print the whole response envelope, not just `result`.",
    )

    sub.add_parser("health", help="Is the broker reachable? Needs no lease.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    kwargs: dict[str, Any] = {}
    if args.timeout is not None:
        kwargs["timeout_s"] = args.timeout

    try:
        if args.url and args.lease:
            client = ToolboxClient(args.url, args.lease, **kwargs)
        else:
            client = ToolboxClient.from_env(**kwargs)
    except ToolboxError as exc:
        print(f"toolbox: {exc.message}", file=sys.stderr)
        return EXIT_USAGE

    try:
        if args.command == "health":
            return _health(client)
        if args.command == "list":
            return _list(client, as_json=args.json)
        if args.command == "describe":
            return _describe(client, args.tool, as_json=args.json)
        if args.command == "call":
            return _call(client, args)
    except ToolboxError as exc:
        print(f"toolbox: [{exc.code}] {exc.message}", file=sys.stderr)
        if exc.detail:
            print(json.dumps(exc.detail, indent=2), file=sys.stderr)
        return EXIT_REFUSED
    return EXIT_USAGE  # pragma: no cover - argparse requires a subcommand


# --- subcommands -------------------------------------------------------------


def _health(client: ToolboxClient) -> int:
    print(json.dumps(client.health(), indent=2))
    return EXIT_OK


def _list(client: ToolboxClient, *, as_json: bool) -> int:
    response = client.list_tools()
    _warn_protocol(client)
    if as_json:
        print(response.model_dump_json(indent=2))
        return EXIT_OK

    print(_lease_line(response.lease))
    if not response.tools:
        print("\nNo tools are bound to this lease.")
        return EXIT_OK
    print()
    width = max(len(t.id) for t in response.tools)
    for tool in response.tools:
        mark = " (write)" if tool.access == "write" else ""
        print(f"  {tool.id:<{width}}  {_first_line(tool.description)}{mark}")
    print("\nRun `toolbox describe <tool>` for its input schema.")
    return EXIT_OK


def _describe(client: ToolboxClient, tool_id: str, *, as_json: bool) -> int:
    response = client.describe(tool_id)
    _warn_protocol(client)
    if as_json:
        print(response.model_dump_json(indent=2))
        return EXIT_OK
    print(_render_tool(response.tool))
    return EXIT_OK


def _call(client: ToolboxClient, args: argparse.Namespace) -> int:
    try:
        arguments = _read_arguments(args)
    except _InputError as exc:
        print(f"toolbox: {exc}", file=sys.stderr)
        return EXIT_USAGE

    response = client.call(args.tool, arguments)
    _warn_protocol(client)

    if not response.ok:
        error = response.error
        code = error.code if error else "unknown"
        message = error.message if error else "call failed"
        print(f"toolbox: [{code}] {message}", file=sys.stderr)
        if error and error.detail:
            print(json.dumps(error.detail, indent=2), file=sys.stderr)
        if response.lease:
            print(f"toolbox: {_lease_line(response.lease)}", file=sys.stderr)
        return EXIT_REFUSED

    rendered = json.dumps(
        response.model_dump() if args.raw else response.result, indent=2, default=str
    )
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    if response.lease:
        print(f"toolbox: {_lease_line(response.lease)}", file=sys.stderr)
    return EXIT_OK


# --- helpers -----------------------------------------------------------------


class _InputError(ValueError):
    """Bad arguments on this machine. Exit 2, not 1 -- nothing was called."""


def _read_arguments(args: argparse.Namespace) -> dict[str, Any]:
    if args.input and args.json_arg:
        raise _InputError("use --input or --json-arg, not both")
    if args.json_arg:
        raw, source = args.json_arg, "--json-arg"
    elif args.input == "-":
        raw, source = sys.stdin.read(), "stdin"
    elif args.input:
        path = Path(args.input)
        if not path.exists():
            raise _InputError(f"no such input file: {path}")
        raw, source = path.read_text(encoding="utf-8"), str(path)
    else:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _InputError(f"{source} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise _InputError(
            f"{source} must be a JSON object of arguments, got {type(parsed).__name__}"
        )
    return parsed


def _lease_line(lease: LeaseStatus) -> str:
    parts = [f"{lease.calls_remaining}/{lease.max_calls} calls remaining"]
    if lease.expires_in_s is not None:
        parts.append(f"lease expires in {lease.expires_in_s:.0f}s")
    if lease.allow_write:
        parts.append("write permitted")
    return "; ".join(parts)


def _render_tool(tool: BrokeredTool) -> str:
    lines = [f"{tool.id}  [{tool.access}]", "", tool.description.strip(), ""]
    lines.append("input schema (pass a JSON object matching this):")
    lines.append(json.dumps(tool.input_schema, indent=2))
    if tool.output_schema:
        lines += ["", "returns:", json.dumps(tool.output_schema, indent=2)]
    lines += ["", f"  toolbox call {tool.id} --input args.json"]
    return "\n".join(lines)


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def _warn_protocol(client: ToolboxClient) -> None:
    if client.protocol_warning:
        print(f"toolbox: warning: {client.protocol_warning}", file=sys.stderr)
        client.protocol_warning = None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
