from reagents.tools.registry import (
    UnboundToolError,
    UnknownToolError,
    default_registry,
    tool_jaccard,
)


def test_bind_returns_only_requested_tools():
    registry = default_registry()
    pack = registry.bind(["simplify", "solve"])
    assert pack.ids() == ["simplify", "solve"]
    assert {s.id for s in pack.specs()} == {"simplify", "solve"}


def test_bind_unknown_tool_is_an_error():
    registry = default_registry()
    try:
        registry.bind(["simplify", "not_a_tool"])
    except UnknownToolError as exc:
        assert "not_a_tool" in str(exc)
    else:
        raise AssertionError("expected UnknownToolError")


def test_unbound_tool_call_is_a_hard_error():
    pack = default_registry().bind(["simplify", "solve"])
    try:
        pack.call("find_cycles", graph={"nodes": [], "edges": []})
    except UnboundToolError as exc:
        assert "find_cycles" in str(exc)
    else:
        raise AssertionError("expected UnboundToolError")


def test_bound_tools_execute():
    pack = default_registry().bind(["simplify", "find_cycles"])
    cancelled = pack.call("simplify", expr={"terms": {"a": 2, "b": 0}})
    assert cancelled["terms"] == {"a": 2.0}
    cycles = pack.call(
        "find_cycles",
        graph={
            "nodes": ["x", "y"],
            "edges": [{"src": "x", "dst": "y"}, {"src": "y", "dst": "x"}],
        },
    )
    assert cycles["count"] >= 1


def test_jaccard_overlap():
    assert tool_jaccard(["a", "b"], ["b", "c"]) == 1 / 3
    assert tool_jaccard(["a", "b"], ["c", "d"]) == 0.0
