"""Lazy remote-MCP discovery and execution adapters.

Secrets are resolved from environment variables at connection time and never enter
ToolSpec, ContextEnvelope, or a demigod prompt.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from reagents.contracts import RiskTier, ToolAccess, ToolProvider
from reagents.tools.registry import Tool, ToolExecutionError, ToolRegistry


class MCPConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class MCPServerConfig:
    namespace: str
    url: str
    description: str
    auth_header: str | None = None
    auth_env: str | None = None
    auth_prefix: str = ""
    allowed_tools_env: str | None = None
    max_tools: int = 24
    access: ToolAccess = ToolAccess.READ
    risk_tier: RiskTier = RiskTier.LOW

    def headers(self) -> dict[str, str]:
        if not self.auth_header:
            return {}
        if not self.auth_env:
            raise MCPConfigurationError(
                f"{self.namespace}: auth_header requires auth_env"
            )
        secret = os.environ.get(self.auth_env)
        if not secret:
            raise MCPConfigurationError(
                f"{self.namespace}: set {self.auth_env} before enabling this MCP"
            )
        return {self.auth_header: f"{self.auth_prefix}{secret}"}

    def allowed_tools(self) -> set[str] | None:
        if not self.allowed_tools_env:
            return None
        raw = os.environ.get(self.allowed_tools_env, "").strip()
        if not raw:
            return None
        return {item.strip() for item in raw.split(",") if item.strip()}


SPONSOR_MCP_SERVERS = (
    MCPServerConfig(
        namespace="paperclip",
        url="https://paperclip.gxl.ai/mcp",
        description="Scientific literature, trials, regulatory documents, and biological databases.",
        auth_header="X-API-Key",
        auth_env="PAPERCLIP_API_KEY",
        allowed_tools_env="PAPERCLIP_MCP_ALLOWED_TOOLS",
        # Unknown/unannotated remote operations default to write-denied. Properly
        # annotated read-only search/read tools are downgraded during discovery.
        access=ToolAccess.WRITE,
        risk_tier=RiskTier.MODERATE,
    ),
    MCPServerConfig(
        namespace="biomni",
        url="https://mcp.phylo.bio/mcp",
        description="Biomni integrated biology environment and managed biological workflows.",
        auth_header="Authorization",
        auth_env="BIOMNI_MCP_AUTHORIZATION",
        allowed_tools_env="BIOMNI_MCP_ALLOWED_TOOLS",
        access=ToolAccess.WRITE,
        risk_tier=RiskTier.MODERATE,
    ),
)


def configure_sponsor_mcp(registry: ToolRegistry) -> None:
    """Register namespace loaders without making any network calls."""
    for config in SPONSOR_MCP_SERVERS:
        registry.register_deferred(
            config.namespace,
            lambda config=config: discover_mcp_tools(config),
        )


class MCPToolExecutor:
    def __init__(self, config: MCPServerConfig, remote_name: str) -> None:
        self.config = config
        self.remote_name = remote_name

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        async with _mcp_session(self.config) as session:
            result = await session.call_tool(self.remote_name, arguments=arguments)
        if getattr(result, "isError", False) or getattr(result, "is_error", False):
            raise ToolExecutionError(
                f"{self.config.namespace}.{self.remote_name} returned an MCP error: "
                f"{_model_dump(result)}"
            )
        structured = getattr(result, "structuredContent", None)
        if structured is None:
            structured = getattr(result, "structured_content", None)
        if structured is not None:
            return structured
        return {"content": [_model_dump(block) for block in getattr(result, "content", [])]}


async def discover_mcp_tools(config: MCPServerConfig) -> list[Tool]:
    """Connect only when selected, discover schemas, then create leased adapters."""
    allowed = config.allowed_tools()
    discovered: list[Tool] = []
    async with _mcp_session(config) as session:
        cursor: str | None = None
        while True:
            page = await session.list_tools(cursor=cursor)
            for remote in page.tools:
                if allowed is not None and remote.name not in allowed:
                    continue
                discovered.append(
                    Tool(
                        id=f"{config.namespace}.{remote.name}",
                        namespace=config.namespace,
                        description=remote.description or (
                            f"{remote.name} provided by the {config.namespace} MCP server."
                        ),
                        parameters_schema=remote.inputSchema or {
                            "type": "object",
                            "properties": {},
                        },
                        output_schema=getattr(remote, "outputSchema", None) or {},
                        executor=MCPToolExecutor(config, remote.name),
                        provider=ToolProvider.MCP,
                        access=_remote_access(remote, config.access),
                        risk_tier=config.risk_tier,
                        cost_class="remote",
                        latency_class="network",
                        defer_loading=True,
                    )
                )
                if len(discovered) >= config.max_tools:
                    return discovered
            cursor = getattr(page, "nextCursor", None)
            if not cursor:
                break
    if not discovered:
        suffix = " after applying the allowlist" if allowed is not None else ""
        raise MCPConfigurationError(
            f"{config.namespace}: server exposed no usable tools{suffix}"
        )
    return discovered


@asynccontextmanager
async def _mcp_session(config: MCPServerConfig) -> AsyncIterator[Any]:
    try:
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:
        raise MCPConfigurationError(
            "MCP support is not installed; run `uv sync --extra mcp`"
        ) from exc

    timeout = httpx.Timeout(30.0, read=300.0)
    async with httpx.AsyncClient(
        headers=config.headers(),
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        async with streamable_http_client(config.url, http_client=client) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(value, (str, int, float, bool, type(None), list, dict)):
        return value
    return str(value)


def _remote_access(remote: Any, default: ToolAccess) -> ToolAccess:
    annotations = getattr(remote, "annotations", None)
    if annotations is None:
        return default
    read_only = getattr(annotations, "readOnlyHint", None)
    if read_only is None:
        read_only = getattr(annotations, "read_only_hint", None)
    if read_only is True:
        return ToolAccess.READ
    if read_only is False:
        return ToolAccess.WRITE
    return default
