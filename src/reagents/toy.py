"""Toy conservation / pathway problem used by the walking skeleton."""

from __future__ import annotations

from reagents.contracts import Axis, DomainSpec, NativeProblem

ARTIFACT_SCHEMA = {
    "type": "object",
    "required": ["findings", "conclusion"],
    "properties": {
        "findings": {"type": "array", "items": {"type": "string"}},
        "conclusion": {"type": "string"},
        "confidence": {"type": "number"},
    },
}


def toy_problem() -> NativeProblem:
    return NativeProblem(
        id="pfk-bottleneck",
        statement=(
            "In glycolysis, hexokinase phosphorylates glucose to glucose-6-phosphate using ATP. "
            "Phosphoglucose isomerase converts glucose-6-phosphate to fructose-6-phosphate. "
            "Phosphofructokinase phosphorylates fructose-6-phosphate to fructose-1,6-bisphosphate "
            "using ATP and is strongly inhibited by high ATP. Hexokinase is upregulated fivefold."
        ),
        entities=[
            "glucose",
            "glucose-6-phosphate",
            "fructose-6-phosphate",
            "fructose-1,6-bisphosphate",
            "hexokinase",
            "phosphofructokinase",
            "phosphoglucose isomerase",
            "ATP",
            "ADP",
        ],
        constraints=[
            "Phosphofructokinase is inhibited by high ATP.",
            "Each kinase step transfers one phosphoryl from ATP to a sugar.",
        ],
        question=(
            "If hexokinase is upregulated fivefold, why might net flux to "
            "fructose-1,6-bisphosphate still not increase, and is phosphate conserved "
            "across these steps?"
        ),
    )


def toy_domains() -> list[DomainSpec]:
    return [
        DomainSpec(
            name="stoichiometric_flow",
            axes=[Axis.CONSERVATION],
            language="integer hypergraph with balanced token vectors on each hyperedge",
            transform_prompt=(
                "Project species to symbols s* and cofactors to c*. Write each reaction "
                "as integer in/out token vectors. Do not keep chemical names."
            ),
            tool_ids=["simplify", "dimensional_check"],
            artifact_schema=ARTIFACT_SCHEMA,
            forbidden=[
                "Use only symbols defined in the representation.",
                "Do not refer to biological proper names, genes, proteins, or metabolites.",
            ],
        ),
        DomainSpec(
            name="catalytic_dag",
            axes=[Axis.TOPOLOGY],
            language="directed acyclic catalytic dependency graph with a distinguished cut edge",
            transform_prompt=(
                "Project the cascade to a DAG of vertices v* and catalytic edges e*. "
                "Mark the gated committed edge. Drop all native labels."
            ),
            tool_ids=["build_graph", "cut"],
            artifact_schema=ARTIFACT_SCHEMA,
            forbidden=[
                "Use only symbols defined in the representation.",
                "Do not refer to biological proper names, genes, proteins, or metabolites.",
            ],
        ),
        DomainSpec(
            name="rate_orbit",
            axes=[Axis.DYNAMICS],
            language="autonomous linear flux orbit on gated inflow and committed outflow",
            transform_prompt=(
                "Encode upregulation as a large inflow coefficient and cofactor gating as a "
                "small committed outflow. Use only state symbols x* and gates g*."
            ),
            tool_ids=["simulate", "sample"],
            artifact_schema=ARTIFACT_SCHEMA,
            forbidden=[
                "Use only symbols defined in the representation.",
                "Do not refer to biological proper names, genes, proteins, or metabolites.",
            ],
        ),
    ]
