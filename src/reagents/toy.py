"""Toy conservation / pathway problem used by the walking skeleton."""

from __future__ import annotations

from reagents.contracts import Axis, DomainSpec, NativeProblem

ARTIFACT_SCHEMA = {
    "type": "object",
    "required": [
        "candidate_solution",
        "constraint_results",
        "certificate",
        "conclusion",
    ],
    "properties": {
        "findings": {"type": "array", "items": {"type": "string"}},
        "candidate_solution": {"type": "object"},
        "constraint_results": {"type": "object"},
        "certificate": {"type": "object"},
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


def simple_problem() -> NativeProblem:
    """A deliberately easy problem for exercising the pipeline, not the model.

    Structurally the same shape as `toy_problem` -- a series pipeline whose
    throughput is set by a gated middle stage, plus a conservation question --
    but with five plain entities instead of nine biochemical ones. That matters
    for two reasons when testing:

    * Fewer, plainer entity names give the planner far less to accidentally
      leak into the domain spec, so runs fail for interesting reasons rather
      than on the seal.
    * The answer is short and derivable by reasoning alone, so a demigod with
      no tools mapped can finish well inside a small turn budget.

    The answer: throughput is set by valve B, so upgrading pump A does not
    raise outflow; water is conserved because nothing leaves the system except
    through the outlet.
    """
    return NativeProblem(
        id="valve-bottleneck",
        statement=(
            "Water flows through three tanks in series. Pump A moves water from "
            "tank 1 to tank 2. Valve B is a fixed narrow opening between tank 2 "
            "and tank 3, and it is already at its maximum throughput. An outlet "
            "drains tank 3. Pump A is upgraded to move water five times faster."
        ),
        entities=["pump A", "valve B", "tank 1", "tank 2", "tank 3", "outlet"],
        constraints=[
            "Valve B has a fixed maximum throughput that is already reached.",
            "Water leaves the system only through the outlet.",
            "Tanks have finite capacity.",
        ],
        question=(
            "After pump A is upgraded fivefold, does the flow rate at the outlet "
            "increase, and is the total volume of water conserved?"
        ),
    )


def arithmetic_problem() -> NativeProblem:
    """The EASY tier: one multiplication and one subtraction.

    Here to be beneath the machinery, not to exercise it. There is no second
    coordinate system worth inventing for `100 - 17 x 4.25`, so a planner free
    to choose should choose none and answer it -- which is what it does. It is
    the cheapest observable proof that the fan-out tracks the problem: measured
    live at 2 model calls and 7.4s, against the two sealed demigods a fixed
    count would have spent on it.
    """
    return NativeProblem(
        id="shop-change",
        statement=(
            "A shop sells notebooks at 4.25 each. A customer buys 17 of them "
            "and pays with a 100 note."
        ),
        entities=["notebook", "customer"],
        constraints=["Give the amount in the same currency unit as the prices."],
        question="How much change does the customer receive?",
        required_outputs=["the change owed"],
    )


def schedule_problem() -> NativeProblem:
    """The MEDIUM tier: real dependencies, one obvious representation.

    Not arithmetic -- the answer needs the dependency structure, and the two
    non-critical branches are both 9 days against the critical path's 10, so
    guessing at the longest single task gets it wrong. But the representation
    that settles it (a precedence graph and its longest walk) is so clearly the
    right one that a rival language mostly paraphrases it. Live, GOD chose two
    domains here: topology and geometry.
    """
    return NativeProblem(
        id="release-schedule",
        statement=(
            "Five tasks must be completed to ship a release. Design takes 3 "
            "days. Backend takes 5 days and cannot start until design is done. "
            "Frontend takes 4 days and also cannot start until design is done. "
            "Integration takes 2 days and needs both backend and frontend "
            "finished. Documentation takes 6 days and needs only design "
            "finished. The release goes out when every task is complete. "
            "Any number of tasks may run at the same time."
        ),
        entities=["design", "backend", "frontend", "integration", "documentation"],
        constraints=[
            "A task starts only once every task it depends on has finished.",
            "There is no limit on how many tasks run concurrently.",
        ],
        question=(
            "What is the shortest possible time to ship, and which tasks "
            "determine it?"
        ),
        required_outputs=["the shortest schedule length", "the determining tasks"],
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
                "Do not refer to any entity by its name from the original problem.",
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
                "Do not refer to any entity by its name from the original problem.",
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
                "Do not refer to any entity by its name from the original problem.",
            ],
        ),
    ]
