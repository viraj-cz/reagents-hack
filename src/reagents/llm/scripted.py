"""Deterministic LLM stand-in so the skeleton runs without an API key."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from reagents.contracts import Budget
from reagents.god.integrator import IntegrationDraft
from reagents.god.planner import CriticVerdict, InventedDomains
from reagents.god.transformer import TransformDraft
from reagents.llm.client import LLMError
from reagents.demigod.runtime import DemigodDraft
from reagents.tools.registry import BoundToolPack
from reagents.toy import toy_domains

T = TypeVar("T", bound=BaseModel)

Script = BaseModel | dict[str, Any] | Callable[..., Any]


class ScriptedLLM:
    def __init__(self, scripts: dict[str, Script] | None = None) -> None:
        self.scripts: dict[str, Script] = scripts or {}

    @classmethod
    def for_toy_pathway(cls) -> ScriptedLLM:
        domains = toy_domains()
        return cls(
            {
                "invent": InventedDomains(domains=domains),
                "critic": CriticVerdict(ok=True, colliding_names=[], reasons=[]),
                "transform:stoichiometric_flow": TransformDraft(
                    representation={
                        "tokens": ["s1", "s2", "s3", "s4", "c1", "c2"],
                        "edges": [
                            {"id": "r1", "in": {"s1": 1, "c1": 1}, "out": {"s2": 1, "c2": 1}},
                            {"id": "r2", "in": {"s2": 1}, "out": {"s3": 1}},
                            {"id": "r3", "in": {"s3": 1, "c1": 1}, "out": {"s4": 1, "c2": 1}},
                        ],
                    },
                    task=(
                        "Test whether each r* edge conserves the {c1,c2} token pair and "
                        "whether the net {c1,c2} exchange is zero after cancellation."
                    ),
                    notation_guide="s* are substrate tokens, c* are cofactor tokens, r* are hyperedges.",
                    symbol_to_native={
                        "s1": "glucose",
                        "s2": "glucose-6-phosphate",
                        "s3": "fructose-6-phosphate",
                        "s4": "fructose-1,6-bisphosphate",
                        "c1": "ATP",
                        "c2": "ADP",
                        "r1": "hexokinase",
                        "r2": "phosphoglucose isomerase",
                        "r3": "phosphofructokinase",
                    },
                ),
                "transform:catalytic_dag": TransformDraft(
                    representation={
                        "nodes": ["v1", "v2", "v3", "v4"],
                        "edges": [
                            {"src": "v1", "dst": "v2", "id": "e1"},
                            {"src": "v2", "dst": "v3", "id": "e2"},
                            {"src": "v3", "dst": "v4", "id": "e3", "gated": True},
                        ],
                    },
                    task=(
                        "Build the DAG and cut the gated edge's source. Decide whether e3 "
                        "is a unique committed cut on the path v1 to v4."
                    ),
                    notation_guide="v* are pool vertices. e* are catalytic edges. gated means repressed.",
                    symbol_to_native={
                        "v1": "glucose",
                        "v2": "glucose-6-phosphate",
                        "v3": "fructose-6-phosphate",
                        "v4": "fructose-1,6-bisphosphate",
                        "e1": "hexokinase",
                        "e2": "phosphoglucose isomerase",
                        "e3": "phosphofructokinase",
                    },
                ),
                "transform:rate_orbit": TransformDraft(
                    representation={
                        "state": {"x_mid": 0.2, "x_out": 0.05, "g_in": 5.0, "g_out": 0.15},
                        "rates": {
                            "x_mid": {"g_in": 1.0, "g_out": -1.0, "x_mid": -0.05},
                            "x_out": {"g_out": 1.0},
                        },
                    },
                    task=(
                        "Simulate the orbit. Compare committed outflow g_out against inflated "
                        "inflow g_in. Sample the two gates as a discrete mass."
                    ),
                    notation_guide="x_mid is the inter-gate pool. g_in is inflated inflow. g_out is gated outflow.",
                    symbol_to_native={
                        "x_mid": "fructose-6-phosphate",
                        "x_out": "fructose-1,6-bisphosphate",
                        "g_in": "hexokinase",
                        "g_out": "phosphofructokinase",
                    },
                ),
                "demigod:stoichiometric_flow": _conservation_demigod,
                "demigod:catalytic_dag": _topology_demigod,
                "demigod:rate_orbit": _dynamics_demigod,
                "integrate": IntegrationDraft(
                    answer=(
                        "Phosphate is conserved: each kinase moves one phosphoryl from ATP to "
                        "a sugar, exchanging ATP for ADP. Net flux to fructose-1,6-bisphosphate "
                        "need not rise when hexokinase is upregulated fivefold because "
                        "phosphofructokinase is the committed, ATP-gated step; topology shows it "
                        "as the cut edge, and the dynamical orbit keeps committed outflow small "
                        "while the mid-pathway pool absorbs the extra inflow."
                    ),
                    confidence=0.86,
                    domain_contributions={
                        "stoichiometric_flow": "ATP/ADP plus sugar-phosphate tokens cancel on every kinase edge.",
                        "catalytic_dag": "PFK is the unique gated cut on the path to FBP.",
                        "rate_orbit": "Inflated hexokinase inflow does not lift the small PFK outflow.",
                    },
                    conflicts=[],
                    gaps=["Product inhibition of hexokinase by G6P was not represented."],
                ),
            }
        )

    async def complete(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        phase: str = "",
    ) -> T:
        return self._resolve(phase, response_model, system=system, user=user)

    async def run_tool_loop(
        self,
        *,
        system: str,
        user: str,
        tools: BoundToolPack,
        response_model: type[T],
        budget: Budget,
        phase: str = "",
    ) -> tuple[T, list[dict[str, Any]]]:
        del budget
        raw = self.scripts.get(phase)
        if callable(raw):
            result = raw(tools=tools, system=system, user=user, response_model=response_model)
            if isinstance(result, tuple):
                draft, trace = result
                return _coerce(draft, response_model), trace
            return _coerce(result, response_model), []
        return self._resolve(phase, response_model, system=system, user=user, tools=tools), []

    def _resolve(self, phase: str, response_model: type[T], **kwargs: Any) -> T:
        raw = self.scripts.get(phase, self.scripts.get("default"))
        if raw is None:
            raise LLMError(f"no script for phase {phase!r}")
        if isinstance(raw, list):
            if not raw:
                raise LLMError(f"script queue empty for phase {phase!r}")
            raw = raw.pop(0)
        if callable(raw):
            return _coerce(raw(**kwargs), response_model)
        return _coerce(raw, response_model)


def _coerce(value: Any, response_model: type[T]) -> T:
    if isinstance(value, response_model):
        return value
    if isinstance(value, BaseModel):
        return response_model.model_validate(value.model_dump())
    if isinstance(value, dict):
        return response_model.model_validate(value)
    raise LLMError(f"cannot coerce {type(value)} to {response_model}")


def _conservation_demigod(*, tools: BoundToolPack, **_: Any) -> tuple[DemigodDraft, list[dict[str, Any]]]:
    cancelled = tools.call("simplify", expr={"terms": {"c1": -2, "c2": 2, "s_tokens": 0}})
    dims = tools.call(
        "dimensional_check",
        terms=[
            {"symbol": "r1_in", "dims": {"P": 1}},
            {"symbol": "r1_out", "dims": {"P": 1}},
        ],
    )
    trace = [
        {"tool": "simplify", "result": cancelled},
        {"tool": "dimensional_check", "result": dims},
    ]
    return (
        DemigodDraft(
            payload={
                "findings": [
                    "Each r1/r3 hyperedge exchanges one c1 token for one c2 token.",
                    "Sugar tokens gain the same P-count that c1 loses.",
                    f"Linear cancellation leaves terms={cancelled['terms']}.",
                ],
                "conclusion": (
                    "The {c1,c2} pair is conserved as a transfer, not a source. "
                    "Net phosphate tokens are internally rearranged, not created."
                ),
                "confidence": 0.9,
            },
            justification=(
                "r1 and r3 are balanced hyperedges on (s*, c*). simplify shows a pure "
                "c1->c2 transfer; dimensional_check places P on the sugar tokens that "
                "receive the transfer."
            ),
        ),
        trace,
    )


def _topology_demigod(*, tools: BoundToolPack, **_: Any) -> tuple[DemigodDraft, list[dict[str, Any]]]:
    graph = tools.call(
        "build_graph",
        nodes=["v1", "v2", "v3", "v4"],
        edges=[
            {"src": "v1", "dst": "v2", "id": "e1"},
            {"src": "v2", "dst": "v3", "id": "e2"},
            {"src": "v3", "dst": "v4", "id": "e3"},
        ],
    )
    cut_graph = tools.call("cut", graph=graph, node="v3")
    remaining_dsts = {e["dst"] for e in cut_graph["edges"]}
    trace = [
        {"tool": "build_graph", "result": graph},
        {"tool": "cut", "result": cut_graph},
    ]
    return (
        DemigodDraft(
            payload={
                "findings": [
                    "The only path v1->v4 uses e3 as its last edge.",
                    f"Cutting v3 removes access to v4; remaining destinations={sorted(remaining_dsts)}.",
                    "e3 is the unique gated cut edge on that path.",
                ],
                "conclusion": (
                    "Inflating e1 cannot increase flow into v4 while e3 remains the cut."
                ),
                "confidence": 0.88,
            },
            justification=(
                "The DAG is a single chain. cut(v3) disconnects v4, so e3 is committed."
            ),
        ),
        trace,
    )


def _dynamics_demigod(*, tools: BoundToolPack, **_: Any) -> tuple[DemigodDraft, list[dict[str, Any]]]:
    system = {
        "state": {"x_mid": 0.2, "x_out": 0.05, "g_in": 5.0, "g_out": 0.15},
        "rates": {
            "x_mid": {"g_in": 1.0, "g_out": -1.0, "x_mid": -0.05},
            "x_out": {"g_out": 1.0},
        },
    }
    traj = tools.call("simulate", system=system, steps=6, dt=0.2)
    weights = tools.call("sample", weights={"g_in": 5.0, "g_out": 0.15}, n=6)
    final = traj["final"]
    trace = [
        {"tool": "simulate", "result": traj},
        {"tool": "sample", "result": weights},
    ]
    return (
        DemigodDraft(
            payload={
                "findings": [
                    f"Final x_mid={final['x_mid']:.3f} grew under inflated g_in.",
                    f"Final x_out={final['x_out']:.3f} tracks only g_out.",
                    f"Gate sample mass is dominated by g_in: {weights['draws']}.",
                ],
                "conclusion": (
                    "The orbit stores extra inflow in x_mid; committed production of x_out "
                    "stays locked to the small g_out gate."
                ),
                "confidence": 0.84,
            },
            justification=(
                "simulate shows x_mid absorbing g_in - g_out. sample confirms the gate "
                "mass is unbalanced toward inflow, so outflow cannot match."
            ),
        ),
        trace,
    )
