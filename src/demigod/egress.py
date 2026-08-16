"""Which hosts a DEMI_GOD sandbox may talk to. Isolation as a network fact.

"Do not read anything outside your isolated system" is currently a sentence in a
system prompt. A model that decides otherwise is not stopped by it. Modal's
`Sandbox.create(outbound_domain_allowlist=[...])` makes it a property of the
container instead -- verified present in modal 1.5.4, documented as supporting
wildcard prefixes (`*.`).

THE LIST IS SHORT BY CONSTRUCTION, and everything on it is there because
something breaks without it:

  api.anthropic.com   the agent loop. The whole demigod is this call, repeated.
  *.anthropic.com     the `claude` CLI resolves telemetry and auth endpoints on
                      sibling hosts. Blocking them is not obviously fatal but is
                      not obviously safe either, and the isolation we want is
                      "cannot reach a sibling's data", not "cannot phone the
                      vendor whose model it is running".
  <broker host>       the tools. Derived from the grant, never hardcoded, so a
                      `modal serve` URL works without editing this file.

DELIBERATELY ABSENT: package registries. A demigod that can `pip install` can
install its way out of the domain it was given, and the images are pre-baked
precisely so it never needs to.

WHY DOMAINS AND NOT CIDRs. `outbound_cidr_allowlist` also exists and is what the
original design called for, but the broker is a Modal web endpoint behind a
CDN -- its address is not a stable CIDR, so pinning one would either be wrong
today or wrong next week. Same argument for api.anthropic.com.
"""

from __future__ import annotations

from urllib.parse import urlparse

AGENT_EGRESS: tuple[str, ...] = (
    "api.anthropic.com",
    "anthropic.com",
    "*.anthropic.com",
)
"""Hosts the in-sandbox agent loop needs. Empty under an outside-the-sandbox
runner -- the API key and the API calls would both be on the caller."""

UNRESTRICTED = None
"""`egress_domains=None` means no policy, which is Modal's default and this
repo's behaviour before the broker landed. Kept as the default so enabling
network isolation is a decision someone makes, not one they inherit."""


def broker_host(url: str) -> str | None:
    """Hostname of a broker URL, or None if it has none.

    Port is stripped: Modal's allowlist is a domain list, not host:port.
    """
    parsed = urlparse(url if "//" in url else f"https://{url}")
    return parsed.hostname or None


def allowlist(
    broker_url: str | None = None, *, agent_loop_inside: bool = True
) -> list[str]:
    """The smallest allowlist that lets this demigod do its job.

    Returns a list, never None -- the caller decides whether to apply a policy
    at all. An empty-ish list here would still be a policy (deny everything),
    which is a legitimate configuration for a demigod with no tools and an
    outside-the-sandbox runner.
    """
    domains: list[str] = list(AGENT_EGRESS) if agent_loop_inside else []
    if broker_url:
        host = broker_host(broker_url)
        if host and host not in domains:
            domains.append(host)
    return domains


__all__ = ["AGENT_EGRESS", "UNRESTRICTED", "allowlist", "broker_host"]
