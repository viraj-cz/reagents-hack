"""Answer in the native field, with no domain and no demigod.

The planner may return zero domains -- see `planner.INVENT_SYSTEM`. That is not
a failure to plan; it is the judgement that no invented representation earns the
sandbox it would cost. Something still has to answer the question, and on that
path it is God, in the native language, in one call.

Deliberately NOT the integrator with an empty artifact list. The integrator's
whole contract is "never assert what no demigod established", so pointing it at
nothing produces an empty answer with a truthful explanation of why. Here there
is no artifact to be faithful to, and the honesty requirement moves into the
prompt: answer, and say plainly how sure you are.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from reagents.contracts import NativeProblem, NativeSolution
from reagents.llm.client import LLMClient

DIRECT_CONTRIBUTION_KEY = "god"
"""`domain_contributions` key for reasoning God did itself.

Every other key in that dict is a domain name, and the integrator is forbidden
from inventing one. A direct answer has no domain, so it gets a key that cannot
collide with a `snake_case` domain identifier the planner would produce."""

DIRECT_SYSTEM = """You are God, answering a problem yourself.

You are here because you judged that inventing a foreign representation and
sealing a demigod inside it would cost more than it would buy. Nothing else is
running: there are no alternative candidates coming, and no second opinion to
compare against. Answer the question completely, and show the reasoning that
gets you there.

That solitude is the whole reason to be honest about it. If the problem turns
out, once you are inside it, to be larger than it looked -- it needs evidence
you do not have, or it turns on a judgement you cannot check -- then say so in
`gaps` and put your real belief in `confidence`. A direct answer at 0.4 is a
usable result. A direct answer at 0.95 that should have been 0.4 is worse than
no answer, because nothing downstream can tell the difference.

Obey the stated constraints exactly, including the shape of the answer."""


class DirectAnswer(BaseModel):
    answer: str
    structured_answer: dict = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    gaps: list[str] = Field(default_factory=list)


async def answer_directly(
    llm: LLMClient, problem: NativeProblem, *, rationale: str = ""
) -> NativeSolution:
    """One native-language answer, no domains, no spawn.

    `rationale` is the planner's own account of why no domain was invented. It
    is passed back in so the answer is written under the same reading of the
    problem that skipped the demigods, rather than by a call that has no idea a
    decision was made.
    """
    user = (
        f"Native problem id: {problem.id}\n"
        f"Statement: {problem.statement}\n"
        f"Question: {problem.question}\n"
        f"Entities: {problem.entities}\n"
        f"Native inputs: {problem.inputs}\n"
        f"Required outputs: {problem.required_outputs}\n"
        f"Native answer JSON schema: {problem.answer_schema}\n"
    )
    if problem.constraints:
        user += "\nConstraints (apply to the final answer):\n"
        for constraint in problem.constraints:
            user += f"- {constraint}\n"
        user += (
            "If a constraint requires a JSON object, `answer` must be that "
            "object as compact JSON text — no markdown fences, no surrounding "
            "prose. Put explanations in `reasoning`.\n"
        )
    if rationale:
        user += f"\nWhy no domain was invented for this: {rationale}\n"

    draft = await llm.complete(
        system=DIRECT_SYSTEM,
        user=user,
        response_model=DirectAnswer,
        phase="direct",
    )
    return NativeSolution(
        problem_id=problem.id,
        answer=draft.answer,
        structured_answer=draft.structured_answer,
        confidence=draft.confidence,
        domain_contributions=(
            {DIRECT_CONTRIBUTION_KEY: draft.reasoning} if draft.reasoning else {}
        ),
        gaps=draft.gaps,
    )
