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

TOOLS, AND HOW THEY GET THERE
-----------------------------
A sandboxed demigod cannot call GOD's in-process callables, so `BoundToolPack`
is not invoked here and never will be. Instead the pack's LEASE is published to
the TOOLBOX_BROKER (`toolbox=` below), and the demigod is handed a URL and that
lease id. The same lease that authorizes the spawn authorizes the calls -- one
authority, not two.

Pass no `toolbox` and the behaviour is exactly what it was: the lease still
gates whether the spawn happens, and the agent is told its tools are
unreachable. That is a legitimate configuration for a demigod that reasons from
`shared/` alone, and it is the default because minting a live credential should
be an explicit act.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Protocol

from demigod.result import DemiGodResult
from demigod.spawn import spawn_demigod
from demigod.toolbox.protocol import ToolboxGrant
from reagents.contracts import ContextEnvelope
from reagents.demigod.adapter import envelope_to_spec, slugify_domain_name
from reagents.demigod.runtime import IsolationGuard
from reagents.tools.registry import BoundToolPack
from reagents.tracing import NullTracer, TraceSink, demigod_lane, summarize


class ToolboxProvider(Protocol):
    """What this runtime needs from the broker. Structural, so `reagents` does
    not import `broker` -- the dependency runs the other way (`broker` imports
    `reagents`), and adding a back-edge would make the two packages one.

    Satisfied by `broker.session.ToolboxSession`.
    """

    def grant(self, pack: BoundToolPack, *, label: str = ...) -> ToolboxGrant: ...

    def collect_trace(
        self, lease_id: str, *, include_refused: bool = ...
    ) -> list[dict[str, Any]]: ...

    def revoke(self, lease_id: str) -> None: ...


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
        toolbox: ToolboxProvider | None = None,
        shared_files: list[str] | None = None,
        runner_kind: str = "inside",
        cpu: float = 1.0,
        memory_mb: int = 2048,
        max_turns: int | None = None,
        tracer: TraceSink | None = None,
        # Restored: dropped from this signature by the PR #4 merge resolution
        # (81b2198) while `self.restrict_egress = restrict_egress` below was
        # kept, making every instantiation raise NameError. Off by default --
        # verified to block and allow correctly, but the `claude` CLI may reach
        # hosts beyond *.anthropic.com, so enabling it untested would break
        # every live run.
        restrict_egress: bool = False,
    ) -> None:
        self.run_id = run_id
        self.tool_map = tool_map or {}
        # None means "no brokered tools this run" -- see the module docstring.
        # `broker.session.modal_session()` is the production one.
        self.toolbox = toolbox
        # Pins sandbox egress to the agent API plus the broker. Off by default:
        # see envelope_to_spec.
        self.restrict_egress = restrict_egress
        # Paths under the run's shared volume, already seeded by the caller.
        self.shared_files = shared_files or []
        self.runner_kind = runner_kind
        self.cpu = cpu
        self.memory_mb = memory_mb
        # Overrides Budget.max_steps. THE cost lever: every turn is an Anthropic
        # call, and sandbox compute is cents next to that. Set it low when
        # exercising the pipeline rather than trying to solve something.
        self.max_turns = max_turns
        self.tracer = tracer or NullTracer()

    def set_tracer(self, tracer: TraceSink) -> None:
        """Use God's sink so sandbox and in-process runtimes stream alike."""

        self.tracer = tracer

    async def run(
        self,
        envelope: ContextEnvelope,
        tools: BoundToolPack,
        guard: IsolationGuard | None = None,
    ) -> DemiGodResult:
        name = envelope.domain.name
        lane = demigod_lane(name)
        self.tracer.emit(lane, "START", "isolated sandbox stream opened")
        self.tracer.emit(
            lane,
            "SCOPE",
            f"axis={envelope.domain.primary_axis.value}; "
            f"language={envelope.domain.language}",
            data={
                "tools": [spec.id for spec in envelope.tools],
                "max_steps": self.max_turns or envelope.budget.max_steps,
                "max_tool_calls": envelope.budget.max_tool_calls,
            },
        )

        def fail(reason: str, violations: list[str] | None = None) -> DemiGodResult:
            self.tracer.emit(lane, "FAIL", summarize(reason), data=violations)
            return DemiGodResult.failure(
                status="failed",
                error=reason,
                demigod_name=slugify_domain_name(name),
                domain_name=name,
                run_id=self.run_id,
                isolation_violations=violations,
            )

        # Publish the lease BEFORE the sandbox exists. A demigod is handed its
        # credential at spawn time and has no channel to be given one later.
        grant: ToolboxGrant | None = None
        if self.toolbox is not None:
            try:
                grant = await asyncio.to_thread(
                    self.toolbox.grant, tools, label=name
                )
            except Exception as exc:
                # Not fatal. A demigod with no tools and an honest `blockers`
                # entry is worth more than no demigod at all -- and the agent is
                # told, by envelope_to_spec, that its tools are unreachable.
                print(f"[toolbox] could not publish a lease for {name}: {exc}")

        try:
            spec = envelope_to_spec(
                envelope,
                tool_map=self.tool_map,
                toolbox=grant,
                files=self.shared_files,
                max_turns=self.max_turns,
                cpu=self.cpu,
                memory_mb=self.memory_mb,
                restrict_egress=self.restrict_egress,
            )
        except Exception as exc:
            self._revoke(grant)
            return fail(f"could not build a spec from the envelope: {exc}")

        # spawn_demigod is synchronous and blocks for the whole agent run.
        # Without to_thread it would serialize the orchestrator's fan-out --
        # every demigod still correct, total wall clock N times longer, and
        # nothing in the logs saying why.
        self.tracer.emit(lane, "MODEL", "reasoning in an isolated sandbox")
        try:
            result = await asyncio.to_thread(
                spawn_demigod,
                spec,
                run_id=self.run_id,
                runner_kind=self.runner_kind,
            )
        except Exception as exc:
            await self._finish_lease(grant, None)
            return fail(f"sandbox spawn failed: {exc}")

        # The trace is read back from the BROKER, not from the manifest. The
        # agent authors its own claim; it does not get to author the record of
        # what it called. A demigod that called a tool, disliked the answer, and
        # omitted it cannot hide here.
        await self._finish_lease(grant, result)

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

        if result.status != "ok":
            self.tracer.emit(
                lane,
                "FAIL",
                summarize(result.error or "sandbox returned a failed manifest"),
            )
            return result

        if result.justification:
            self.tracer.emit(lane, "REASON", summarize(result.justification))
        conclusion = (
            result.payload.get("conclusion")
            or result.payload.get("claim")
            or result.claim
        )
        if conclusion:
            self.tracer.emit(
                lane,
                "ARTIFACT",
                f"confidence={result.confidence}; {summarize(conclusion)}",
            )
        self.tracer.emit(
            lane,
            "DONE",
            f"artifact validated; files={len(result.files)}",
        )
        return result

    # --- lease lifecycle ----------------------------------------------------

    async def _finish_lease(
        self, grant: ToolboxGrant | None, result: DemiGodResult | None
    ) -> None:
        """Stamp the broker's trace onto the result, then end the lease.

        Revocation is unconditional and happens even when collection fails: a
        lease that outlives its demigod is a credential lying around, and the
        sandbox it was issued to is already gone.
        """
        if grant is None or self.toolbox is None:
            return
        try:
            trace = await asyncio.to_thread(
                self.toolbox.collect_trace, grant.lease_id
            )
            if result is not None:
                result.tool_trace = trace
        except Exception as exc:
            print(f"[toolbox] could not collect the trace for {grant.lease_id}: {exc}")
        finally:
            self._revoke(grant)

    def _revoke(self, grant: ToolboxGrant | None) -> None:
        if grant is None or self.toolbox is None:
            return
        try:
            self.toolbox.revoke(grant.lease_id)
        except Exception as exc:  # noqa: BLE001 - never fail a run on cleanup
            print(f"[toolbox] revoke failed for {grant.lease_id}: {exc}")


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
