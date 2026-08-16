"""Authenticate to the Phylo-hosted Biomni MCP server and store the token.

    uv run python scripts/phylo_login.py

Why this script exists. `https://mcp.phylo.bio/mcp` is an OAuth 2.0 protected
resource, not an API-key endpoint. An unauthenticated request answers:

    401  WWW-Authenticate: Bearer error="unauthorized",
         resource_metadata="https://mcp.phylo.bio/.well-known/oauth-protected-resource"

So `BIOMNI_MCP_AUTHORIZATION` cannot be pasted from a dashboard the way
`PAPERCLIP_API_KEY` can -- it has to be *minted*. The upstream Biomni issue
asking how to connect (snap-stanford/Biomni#174) is open and unanswered, and
Biomni's own README documents the opposite direction (`agent.add_mcp()` makes
Biomni a CLIENT of other MCP servers). The answer is not in either place; it is
in the server's own discovery documents, which is what this script follows.

Everything below is discovered at runtime rather than hardcoded, so it keeps
working if Phylo moves an endpoint:

    /.well-known/oauth-protected-resource  -> which authorization server
    /.well-known/oauth-authorization-server -> device + token + registration

AUTHORIZATION CODE + PKCE, because device flow is advertised but not usable.
The metadata lists `urn:ietf:params:oauth:grant-type:device_code` under
`grant_types_supported`, which would have been ideal. Registering for it is
rejected ("grant_types must be one of: authorization_code, refresh_token"), and
calling the device endpoint with a dynamically-registered client returns:

    400 unauthorized_client
        "Device authorization is not enabled for this application."

So the server contradicts its own discovery document, and the only flow open to
a self-registered client is authorization-code with PKCE. That needs a redirect
target, hence the throwaway localhost listener below.

Either way you approve in your own browser and this process never sees your
password. The resulting token is written straight into `.env` and is NEVER
printed -- the same discipline used for the Modal token: the secret goes
machine-to-machine, not through a transcript.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import http.server
import secrets
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any

import httpx

RESOURCE = "https://mcp.phylo.bio"
ENV_VAR = "BIOMNI_MCP_AUTHORIZATION"
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
SCOPES = "openid profile email offline_access"
CLIENT_NAME = "reagents-demigod-broker"
REDIRECT_PORT = 8765
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/callback"


def discover() -> dict[str, Any]:
    """Protected-resource metadata -> authorization-server metadata."""
    prm = httpx.get(
        f"{RESOURCE}/.well-known/oauth-protected-resource",
        timeout=20,
        follow_redirects=True,
    )
    prm.raise_for_status()
    servers = prm.json().get("authorization_servers") or []
    if not servers:
        raise SystemExit("no authorization_servers advertised by the resource")
    issuer = servers[0].rstrip("/")
    asm = httpx.get(
        f"{issuer}/.well-known/oauth-authorization-server",
        timeout=20,
        follow_redirects=True,
    )
    asm.raise_for_status()
    return asm.json()


def register_client(meta: dict[str, Any]) -> str:
    """Dynamic client registration (RFC 7591).

    The server advertises `none` among its token-endpoint auth methods, so this
    registers as a PUBLIC client: no client secret to store, and nothing to leak
    if `.env` is ever mishandled. The device flow does not need one.
    """
    endpoint = meta.get("registration_endpoint")
    if not endpoint:
        raise SystemExit(
            "this authorization server does not support dynamic registration; "
            "obtain a client_id from Phylo and set PHYLO_CLIENT_ID"
        )
    resp = httpx.post(
        endpoint,
        json={
            "client_name": CLIENT_NAME,
            # NOT device_code: the registration endpoint rejects it outright,
            # despite the discovery document advertising it. Verified live.
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "redirect_uris": [REDIRECT_URI],
            "token_endpoint_auth_method": "none",
            "application_type": "native",
            "scope": SCOPES,
        },
        timeout=30,
    )
    if resp.status_code >= 400:
        raise SystemExit(
            f"client registration failed ({resp.status_code}): {resp.text[:400]}"
        )
    return resp.json()["client_id"]


def _pkce() -> tuple[str, str]:
    """PKCE pair. Required: the server advertises only S256, and a public
    client has no secret, so PKCE is the whole protection against an
    intercepted code."""
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
    return verifier, challenge


def _capture_code(state: str) -> dict[str, str]:
    """Run a throwaway localhost listener for exactly one redirect."""
    received: dict[str, str] = {}
    done = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            query = urllib.parse.urlparse(self.path).query
            received.update({k: v[0] for k, v in urllib.parse.parse_qs(query).items()})
            ok = received.get("state") == state and "code" in received
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<h2>Authorized. You can close this tab.</h2>"
                if ok
                else b"<h2>Authorization failed.</h2>"
            )
            done.set()

        def log_message(self, *_: Any) -> None:
            return  # keep the console clean

    server = http.server.HTTPServer(("localhost", REDIRECT_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if not done.wait(timeout=600):
        server.shutdown()
        raise SystemExit("timed out waiting for browser authorization")
    server.shutdown()

    if received.get("state") != state:
        raise SystemExit("state mismatch -- possible CSRF, aborting")
    if "code" not in received:
        raise SystemExit(f"no code returned: {received}")
    return received


def authorization_code_flow(meta: dict[str, Any], client_id: str) -> dict[str, Any]:
    """Open the browser, capture the redirect, exchange the code for a token."""
    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(24)
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        # RFC 8707: name the MCP server we want the token to be valid for.
        "resource": RESOURCE,
    }
    url = f"{meta['authorization_endpoint']}?{urllib.parse.urlencode(params)}"

    print("\n" + "=" * 68)
    print("  Approve this client in your browser (opening automatically):")
    print(f"    {url}")
    print("=" * 68 + "\n", flush=True)
    # Headless is fine -- the URL is printed above either way.
    with contextlib.suppress(Exception):
        webbrowser.open(url)

    received = _capture_code(state)
    tok = httpx.post(
        meta["token_endpoint"],
        data={
            "grant_type": "authorization_code",
            "code": received["code"],
            "redirect_uri": REDIRECT_URI,
            "client_id": client_id,
            "code_verifier": verifier,
            "resource": RESOURCE,
        },
        timeout=30,
    )
    if tok.status_code >= 400:
        raise SystemExit(f"token exchange failed ({tok.status_code}): {tok.text[:400]}")
    return tok.json()


def write_env(value: str) -> None:
    """Upsert into .env WITHOUT printing the token.

    The whole point of the device flow is that the secret never passes through
    a human-readable channel; printing it here would undo that.
    """
    line = f"{ENV_VAR}={value}"
    existing = (
        ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    )
    out, replaced = [], False
    for entry in existing:
        if entry.startswith(f"{ENV_VAR}="):
            out.append(line)
            replaced = True
        else:
            out.append(entry)
    if not replaced:
        out.append(line)
    ENV_PATH.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    meta = discover()
    print(f"authorization server: {meta['issuer']}")
    client_id = register_client(meta)
    print(f"registered public client: {client_id}")

    token = authorization_code_flow(meta, client_id)
    access = token.get("access_token")
    if not access:
        raise SystemExit(f"no access_token in response: {sorted(token)}")

    write_env(f"{token.get('token_type', 'Bearer')} {access}")
    print(f"stored {ENV_VAR} in {ENV_PATH.name} ({len(access)} chars, not shown)")
    if "refresh_token" not in token:
        print(
            "NOTE: no refresh_token issued -- this access token will expire and "
            "you will need to re-run this script.",
            file=sys.stderr,
        )
    print("\nverify with:  uv run python scripts/preflight_toolbox.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
