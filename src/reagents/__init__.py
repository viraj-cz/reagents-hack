"""God-to-demigod orchestration for invented representation domains."""

from reagents.contracts import (
    Axis,
    Budget,
    CapabilityLease,
    ContextEnvelope,
    DemiGodResult,
    DomainProblem,
    DomainSpec,
    InverseMap,
    NativeProblem,
    NativeSolution,
    ProjectionManifest,
    ToolAccess,
    ToolProvider,
    ToolSpec,
)
from reagents.god.orchestrator import God
from reagents.tools.registry import ToolRegistry, default_registry
from reagents.tracing import RecordingTracer, TerminalTracer, TraceSink
from reagents.verification import NativeVerifier, VerificationReport

__all__ = [
    "Axis",
    "Budget",
    "CapabilityLease",
    "ContextEnvelope",
    "DemiGodResult",
    "DomainProblem",
    "DomainSpec",
    "God",
    "InverseMap",
    "NativeProblem",
    "NativeSolution",
    "NativeVerifier",
    "ProjectionManifest",
    "RecordingTracer",
    "TerminalTracer",
    "TraceSink",
    "ToolAccess",
    "ToolProvider",
    "ToolRegistry",
    "ToolSpec",
    "VerificationReport",
    "default_registry",
]
