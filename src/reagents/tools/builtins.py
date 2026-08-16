"""Generic operators. Biology-specific products do not belong here."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

from reagents.tools.registry import Tool


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError("expected an object")
    return value


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise TypeError("expected a list")
    return value


def _graph(value: Any) -> dict[str, Any]:
    graph = _as_dict(value)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise TypeError("graph must have list nodes and edges")
    return {"nodes": list(nodes), "edges": list(edges)}


def build_graph(nodes: Any, edges: Any) -> dict[str, Any]:
    node_list = _as_list(nodes)
    edge_list = _as_list(edges)
    ids = []
    for node in node_list:
        if isinstance(node, str):
            ids.append(node)
        elif isinstance(node, dict) and "id" in node:
            ids.append(str(node["id"]))
        else:
            raise TypeError("each node must be an id or {id, ...}")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate node ids")
    return {"nodes": node_list, "edges": edge_list}


def rewrite_edge(graph: Any, old: Any, new: Any) -> dict[str, Any]:
    g = _graph(graph)
    old_e = _as_dict(old)
    new_e = _as_dict(new)
    rewritten = []
    replaced = False
    for edge in g["edges"]:
        if all(edge.get(k) == v for k, v in old_e.items()):
            rewritten.append({**edge, **new_e})
            replaced = True
        else:
            rewritten.append(edge)
    if not replaced:
        raise ValueError(f"no edge matched {old_e}")
    return {"nodes": g["nodes"], "edges": rewritten}


def find_cycles(graph: Any) -> dict[str, Any]:
    g = _graph(graph)
    adj: dict[str, list[str]] = defaultdict(list)
    for edge in g["edges"]:
        adj[str(edge["src"])].append(str(edge["dst"]))
    cycles: list[list[str]] = []
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def dfs(node: str) -> None:
        visiting.add(node)
        stack.append(node)
        for nxt in adj.get(node, []):
            if nxt in visiting:
                idx = stack.index(nxt)
                cycles.append(stack[idx:] + [nxt])
            elif nxt not in visited:
                dfs(nxt)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    nodes = [str(n["id"]) if isinstance(n, dict) else str(n) for n in g["nodes"]]
    for extra in adj:
        if extra not in nodes:
            nodes.append(extra)
    for node in nodes:
        if node not in visited:
            dfs(node)
    return {"cycles": cycles, "count": len(cycles)}


def cut(graph: Any, node: str) -> dict[str, Any]:
    g = _graph(graph)
    node = str(node)

    def nid(raw: Any) -> str:
        return str(raw["id"]) if isinstance(raw, dict) else str(raw)

    nodes = [n for n in g["nodes"] if nid(n) != node]
    edges = [
        e
        for e in g["edges"]
        if str(e.get("src")) != node and str(e.get("dst")) != node
    ]
    return {"nodes": nodes, "edges": edges, "removed": node}


def match_motif(graph: Any, motif: Any) -> dict[str, Any]:
    g = _graph(graph)
    motif_g = _graph(motif)
    motif_edges = {(str(e["src"]), str(e["dst"])) for e in motif_g["edges"]}
    graph_edges = {(str(e["src"]), str(e["dst"])) for e in g["edges"]}
    present = motif_edges <= graph_edges
    return {"matched": present, "motif_edges": sorted(motif_edges), "missing": sorted(motif_edges - graph_edges)}


def simplify(expr: Any) -> dict[str, Any]:
    """Cancel identical addends and drop zero terms from a linear combination dict."""
    terms = _as_dict(expr).get("terms", expr if isinstance(expr, dict) else {})
    if not isinstance(terms, dict):
        raise TypeError("expr.terms must be an object of symbol -> coefficient")
    cancelled: dict[str, float] = {}
    for symbol, coeff in terms.items():
        value = float(coeff)
        if abs(value) < 1e-12:
            continue
        cancelled[str(symbol)] = cancelled.get(str(symbol), 0.0) + value
        if abs(cancelled[str(symbol)]) < 1e-12:
            del cancelled[str(symbol)]
    return {"terms": cancelled, "zero": not cancelled}


def solve(equations: Any, unknowns: Any) -> dict[str, Any]:
    """Solve a square linear system Ax=b given as list of {coeffs, rhs}."""
    eqs = _as_list(equations)
    vars_ = [str(v) for v in _as_list(unknowns)]
    n = len(vars_)
    if len(eqs) != n:
        raise ValueError("need a square system: len(equations) == len(unknowns)")
    a = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for i, eq in enumerate(eqs):
        row = _as_dict(eq)
        coeffs = _as_dict(row.get("coeffs", {}))
        b[i] = float(row.get("rhs", 0.0))
        for j, var in enumerate(vars_):
            a[i][j] = float(coeffs.get(var, 0.0))
    x = _gauss(a, b)
    return {"assignment": {var: x[i] for i, var in enumerate(vars_)}}


def _gauss(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular system")
        m[col], m[pivot] = m[pivot], m[col]
        scale = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            for j in range(col, n + 1):
                m[row][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def dimensional_check(terms: Any) -> dict[str, Any]:
    """Each term is {symbol, dims: {unit: power}}. All dims must match."""
    items = _as_list(terms)
    if not items:
        return {"consistent": True, "dims": {}}
    parsed = []
    for item in items:
        obj = _as_dict(item)
        dims = {str(k): float(v) for k, v in _as_dict(obj.get("dims", {})).items()}
        parsed.append((str(obj.get("symbol", "?")), dims))
    ref = parsed[0][1]
    mismatches = [sym for sym, dims in parsed[1:] if dims != ref]
    return {"consistent": not mismatches, "reference": ref, "mismatches": mismatches}


def simulate(system: Any, steps: int = 8, dt: float = 0.1) -> dict[str, Any]:
    """Forward-Euler on {state, rates} where rates are linear forms {symbol: coeff}."""
    spec = _as_dict(system)
    state = {str(k): float(v) for k, v in _as_dict(spec.get("state", {})).items()}
    rates = {str(k): _as_dict(v) for k, v in _as_dict(spec.get("rates", {})).items()}
    trajectory = [dict(state)]
    for _ in range(int(steps)):
        nxt = dict(state)
        for var, form in rates.items():
            delta = 0.0
            for token, coeff in form.items():
                if token == "const":
                    delta += float(coeff)
                else:
                    delta += float(coeff) * state.get(str(token), 0.0)
            nxt[var] = state.get(var, 0.0) + float(dt) * delta
        state = nxt
        trajectory.append(dict(state))
    return {"trajectory": trajectory, "final": state}


def sample(weights: Any, n: int = 4) -> dict[str, Any]:
    raw = _as_dict(weights)
    keys = list(raw)
    vals = [max(float(raw[k]), 0.0) for k in keys]
    total = sum(vals)
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    probs = [v / total for v in vals]
    # Deterministic stratified sample so tests and demigods are reproducible.
    draws = []
    for i in range(int(n)):
        cursor = (i + 0.5) / max(int(n), 1)
        acc = 0.0
        chosen = keys[-1]
        for key, p in zip(keys, probs):
            acc += p
            if cursor <= acc:
                chosen = key
                break
        draws.append(chosen)
    return {"draws": draws, "probs": dict(zip(keys, probs))}


def entropy(values: Any) -> dict[str, Any]:
    raw = _as_list(values) if not isinstance(values, dict) else list(_as_dict(values).values())
    nums = [max(float(v), 0.0) for v in raw]
    total = sum(nums)
    if total <= 0:
        return {"bits": 0.0, "n": 0}
    bits = 0.0
    for v in nums:
        if v <= 0:
            continue
        p = v / total
        bits -= p * math.log2(p)
    return {"bits": bits, "n": len(nums)}


def compress(data: Any) -> dict[str, Any]:
    """Run-length style token compression as a crude information proxy."""
    if isinstance(data, str):
        tokens = data.split()
    else:
        tokens = [str(x) for x in _as_list(data)]
    if not tokens:
        return {"original": 0, "encoded": 0, "ratio": 1.0, "encoding": []}
    encoding: list[list[Any]] = []
    current = tokens[0]
    count = 1
    for token in tokens[1:]:
        if token == current:
            count += 1
        else:
            encoding.append([current, count])
            current = token
            count = 1
    encoding.append([current, count])
    encoded_len = len(encoding)
    ratio = encoded_len / len(tokens)
    return {"original": len(tokens), "encoded": encoded_len, "ratio": ratio, "encoding": encoding}


def embed(points: Any) -> dict[str, Any]:
    """Place labeled scalars on a line by rank (toy geometric embedding)."""
    raw = _as_dict(points)
    ranked = sorted(raw.items(), key=lambda kv: float(kv[1]))
    if not ranked:
        return {"coords": {}, "span": 0.0}
    lo = float(ranked[0][1])
    hi = float(ranked[-1][1])
    span = hi - lo if hi != lo else 1.0
    coords = {k: (float(v) - lo) / span for k, v in ranked}
    return {"coords": coords, "span": span}


def distance(a: Any, b: Any) -> dict[str, Any]:
    va = _as_list(a) if not isinstance(a, (int, float)) else [float(a)]
    vb = _as_list(b) if not isinstance(b, (int, float)) else [float(b)]
    if len(va) != len(vb):
        raise ValueError("vectors must have the same length")
    sq = sum((float(x) - float(y)) ** 2 for x, y in zip(va, vb))
    return {"l2": math.sqrt(sq)}


def all_tools() -> list[Tool]:
    return [
        Tool(
            id="build_graph",
            description="Build a directed graph from nodes and edges {src, dst, ...}.",
            parameters_schema={
                "type": "object",
                "required": ["nodes", "edges"],
                "properties": {"nodes": {"type": "array"}, "edges": {"type": "array"}},
            },
            fn=build_graph,
        ),
        Tool(
            id="rewrite_edge",
            description="Replace edges matching `old` fields with `new` fields merged in.",
            parameters_schema={
                "type": "object",
                "required": ["graph", "old", "new"],
                "properties": {
                    "graph": {"type": "object"},
                    "old": {"type": "object"},
                    "new": {"type": "object"},
                },
            },
            fn=rewrite_edge,
        ),
        Tool(
            id="find_cycles",
            description="Find directed cycles in a graph.",
            parameters_schema={
                "type": "object",
                "required": ["graph"],
                "properties": {"graph": {"type": "object"}},
            },
            fn=find_cycles,
        ),
        Tool(
            id="cut",
            description="Remove a node and its incident edges.",
            parameters_schema={
                "type": "object",
                "required": ["graph", "node"],
                "properties": {"graph": {"type": "object"}, "node": {"type": "string"}},
            },
            fn=cut,
        ),
        Tool(
            id="match_motif",
            description="Test whether every motif edge is present in the graph.",
            parameters_schema={
                "type": "object",
                "required": ["graph", "motif"],
                "properties": {"graph": {"type": "object"}, "motif": {"type": "object"}},
            },
            fn=match_motif,
        ),
        Tool(
            id="simplify",
            description="Cancel a linear combination {terms: {symbol: coefficient}}.",
            parameters_schema={
                "type": "object",
                "required": ["expr"],
                "properties": {"expr": {"type": "object"}},
            },
            fn=simplify,
        ),
        Tool(
            id="solve",
            description="Solve a square linear system of {coeffs, rhs} equations.",
            parameters_schema={
                "type": "object",
                "required": ["equations", "unknowns"],
                "properties": {"equations": {"type": "array"}, "unknowns": {"type": "array"}},
            },
            fn=solve,
        ),
        Tool(
            id="dimensional_check",
            description="Check that each term has the same dimension vector.",
            parameters_schema={
                "type": "object",
                "required": ["terms"],
                "properties": {"terms": {"type": "array"}},
            },
            fn=dimensional_check,
        ),
        Tool(
            id="simulate",
            description="Forward-Euler integrate {state, rates} for a fixed number of steps.",
            parameters_schema={
                "type": "object",
                "required": ["system"],
                "properties": {
                    "system": {"type": "object"},
                    "steps": {"type": "integer"},
                    "dt": {"type": "number"},
                },
            },
            fn=simulate,
        ),
        Tool(
            id="sample",
            description="Draw a deterministic stratified sample from a discrete weight map.",
            parameters_schema={
                "type": "object",
                "required": ["weights"],
                "properties": {"weights": {"type": "object"}, "n": {"type": "integer"}},
            },
            fn=sample,
        ),
        Tool(
            id="entropy",
            description="Shannon entropy in bits of a list of non-negative masses.",
            parameters_schema={
                "type": "object",
                "required": ["values"],
                "properties": {"values": {"type": "array"}},
            },
            fn=entropy,
        ),
        Tool(
            id="compress",
            description="Run-length compress a token sequence; returns a size ratio.",
            parameters_schema={
                "type": "object",
                "required": ["data"],
                "properties": {"data": {}},
            },
            fn=compress,
        ),
        Tool(
            id="embed",
            description="Embed labeled scalars onto the unit interval by rank.",
            parameters_schema={
                "type": "object",
                "required": ["points"],
                "properties": {"points": {"type": "object"}},
            },
            fn=embed,
        ),
        Tool(
            id="distance",
            description="Euclidean distance between two vectors.",
            parameters_schema={
                "type": "object",
                "required": ["a", "b"],
                "properties": {"a": {"type": "array"}, "b": {"type": "array"}},
            },
            fn=distance,
        ),
    ]
