"""Runtime that executes a demigod in its own isolated Modal sandbox.

The other implementation of the same interface as `runtime.DemigodRuntime`.
That one runs the demigod in GOD's own Python process; this one gives it a
container, a private output volume, and no way to see anything else.

    God(llm, runtime=SandboxDemigodRuntime(run_id="run-1"))

Why both exist: the in-process runtime is fast and needs no infrastructure,
which makes it right for tests and for the scripted-LLM demo. The sandbox
runtime is the real one -- it is the only path where a demigod cannot read
GOD's memory, cannot reach a sibling's output, and cannot install its way out
of the domain it was given.

WHAT THIS RUNTIME DOES NOT DO YET
---------------------------------
`IsolationGuard` checks the artifact for leaked native terms after the fact,
which still works here. But the in-process runtime can also inspect reasoning
as it happens; inside a sandbox we only see what was written. The seal is
therefore checked on the output, not on the process. Closing that gap means
moving the agent loop outside the sandbox, which is a live design question and
deliberately not settled here.

Tool calls do not reach `BoundToolPack`. A sandboxed demigod is a different
process on a different machine and cannot call GOD's in-process callables; the
lease is still minted and still gates whether the spawn is allowed, but the
tools themselves arrive only once the TOOLBOX_BROKER lands. See `adapter.py`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from demigod.result import DemiGodResult
from demigod.spawn import spawn_demigod
from reagents.contracts import ContextEnvelope
from reagents.demigod.adapter import envelope_to_spec, slugify_domain_name
from reagents.demigod.runtime import IsolationGuard
from reagents.tools.registry import BoundToolPack


LEAKED_CONFIDENCE_CEILING = 0.5
"""Confidence cap for an artifact that used native terms.

A ceiling rather than a multiplier: the claim itself may well be correct (it
was, in the run that motivated this), but nothing about a leaked artifact
justifies *high* confidence, and scaling a 0.9 down to 0.45 would understate a
genuinely good result as much as leaving it at 0.9 overstates a bad one.
"""


class SandboxDemigodRuntime:
    """Spawns one `modal.Sandbox` per demigod. Satisfies the runtime interface."""

    def __init__(
        self,
        *,
        run_id: str,
        tool_map: dict[str, str] | None = None,
        shared_files: list[str] | None = None,
        runner_kind: str = "inside",
        cpu: float = 1.0,
        memory_mb: int = 2048,
        max_turns: int | None = None,
    ) -> None:
        self.run_id = run_id
        self.tool_map = tool_map or {}
        # Paths under the run's shared volume, already seeded by the caller.
        self.shared_files = shared_files or []
        self.runner_kind = runner_kind
        self.cpu = cpu
        self.memory_mb = memory_mb
        # Overrides Budget.max_steps. THE cost lever: every turn is an Anthropic
        # call, and sandbox compute is cents next to that. Set it low when
        # exercising the pipeline rather than trying to solve something.
        self.max_turns = max_turns

    async def run(
        self,
        envelope: ContextEnvelope,
        tools: BoundToolPack,
        guard: IsolationGuard | None = None,
    ) -> DemiGodResult:
        name = envelope.domain.name

        def fail(reason: str, violations: list[str] | None = None) -> DemiGodResult:
            return DemiGodResult.failure(
                status="failed",
                error=reason,
                demigod_name=slugify_domain_name(name),
                domain_name=name,
                run_id=self.run_id,
                isolation_violations=violations,
            )

        try:
            spec = envelope_to_spec(
                envelope,
                tool_map=self.tool_map,
                files=self.shared_files,
                max_turns=self.max_turns,
                cpu=self.cpu,
                memory_mb=self.memory_mb,
            )
        except Exception as exc:
            return fail(f"could not build a spec from the envelope: {exc}")

        # spawn_demigod is synchronous and blocks for the whole agent run.
        # Without to_thread it would serialize the orchestrator's fan-out --
        # every demigod still correct, total wall clock N times longer, and
        # nothing in the logs saying why.
        try:
            result = await asyncio.to_thread(
                spawn_demigod,
                spec,
                run_id=self.run_id,
                runner_kind=self.runner_kind,
            )
        except Exception as exc:
            return fail(f"sandbox spawn failed: {exc}")

        # The seal is checked on what came back. Everything the agent wrote is
        # in the manifest, so this is the same check the in-process runtime
        # applies to its draft -- just later.
        #
        # GRADED, NOT FATAL -- and the distinction is the point. The ENVELOPE
        # check before spawn is a hard gate: it stops a demigod reasoning in the
        # native field at all. By the time an artifact comes back the reasoning
        # has already happened, so a native term here is evidence about quality,
        # not proof of contamination.
        #
        # This was fatal, and it discarded a correct artifact that had reasoned
        # entirely in its invented notation and used one generic English noun
        # ("outlet") once. Two demigods independently reached the right answer
        # and the system reported one. Record the violation, cap the confidence,
        # and let the integrator weigh it.
        if guard:
            leaks = guard.check(
                "\n".join([result.claim, result.justification, str(result.payload)])
            )
            if leaks:
                result.isolation_violations = leaks
                result.confidence = min(result.confidence, LEAKED_CONFIDENCE_CEILING)
                result.blockers = [
                    *result.blockers,
                    f"used native terms {leaks}; some reasoning may have left "
                    f"the domain",
                ]

        if result.status == "ok" and envelope.artifact_schema:
            schema_errors = result.validate_against(envelope.artifact_schema)
            if schema_errors:
                result.status = "failed"
                result.error = f"artifact failed schema: {schema_errors}"
                result.blockers = [*result.blockers, *schema_errors]

        return result


def seed_shared_files(run_id: str, local_paths: list[str | Path]) -> list[str]:
    """Upload GOD-supplied files into the run's shared volume before spawning.

    Convenience wrapper: shared/ is mounted read-only in every sandbox, so it
    has to be populated from out here. Returns the names as they will appear to
    a demigod, suitable for `SandboxDemigodRuntime(shared_files=...)`.

    NOTE ON SEALING: these files are NOT passed through `assert_sealed`. If a
    domain is meant to be sealed, whatever GOD uploads must already be in the
    domain representation -- a CSV with native column headers walks straight
    past the seal that the envelope check enforces.
    """
    from demigod.layout import RunLayout

    layout = RunLayout(run_id=run_id, demigod_name="_seed")
    return layout.seed_shared([Path(p) for p in local_paths])
