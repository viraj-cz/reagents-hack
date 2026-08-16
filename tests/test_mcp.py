from types import SimpleNamespace

from reagents.contracts import ToolAccess
from reagents.tools.mcp import MCPServerConfig, _remote_access


def test_mcp_headers_resolve_secret_at_connection_time(monkeypatch):
    config = MCPServerConfig(
        namespace="test",
        url="https://example.test/mcp",
        description="test",
        auth_header="X-Test-Key",
        auth_env="TEST_MCP_KEY",
        auth_prefix="Token ",
    )
    monkeypatch.setenv("TEST_MCP_KEY", "secret")
    assert config.headers() == {"X-Test-Key": "Token secret"}


def test_remote_annotations_can_only_downgrade_to_read():
    read_tool = SimpleNamespace(annotations=SimpleNamespace(readOnlyHint=True))
    write_tool = SimpleNamespace(annotations=SimpleNamespace(readOnlyHint=False))
    unknown_tool = SimpleNamespace(annotations=None)

    assert _remote_access(read_tool, ToolAccess.WRITE) == ToolAccess.READ
    assert _remote_access(write_tool, ToolAccess.WRITE) == ToolAccess.WRITE
    assert _remote_access(unknown_tool, ToolAccess.WRITE) == ToolAccess.WRITE
