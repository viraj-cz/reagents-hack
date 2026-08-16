"""Fixed-command, disposable-container executors for local scientific tools."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any

from reagents.contracts import RiskTier, ToolAccess, ToolProvider
from reagents.tools.registry import Tool, ToolExecutionError, ToolRegistry


@dataclass(frozen=True)
class ContainerToolConfig:
    image: str
    operation: str
    timeout_s: float = 60.0
    network: str = "none"
    max_output_bytes: int = 2_000_000


class ContainerExecutor:
    def __init__(self, config: ContainerToolConfig) -> None:
        self.config = config

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        runtime = os.environ.get("REAGENTS_CONTAINER_RUNTIME")
        if not runtime:
            runtime = shutil.which("docker") or shutil.which("podman")
        if not runtime:
            raise ToolExecutionError(
                "no container runtime found; install Docker or Podman, then run "
                "`reagents tools doctor`"
            )
        process = await asyncio.create_subprocess_exec(
            runtime,
            "run",
            "--rm",
            "--network",
            self.config.network,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            "2g",
            "--cpus",
            "2",
            "--pids-limit",
            "128",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=128m",
            "--tmpfs",
            "/proto:rw,nosuid,size=1g",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PROTO_HOME=/proto",
            "-i",
            self.config.image,
            "python3",
            "/opt/reagents/tool_runtime.py",
            self.config.operation,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        raw_input = json.dumps(arguments).encode()
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(raw_input),
                timeout=self.config.timeout_s,
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise ToolExecutionError(
                f"container operation {self.config.operation!r} timed out"
            ) from exc
        if len(stdout) > self.config.max_output_bytes:
            raise ToolExecutionError("container tool output exceeded its byte limit")
        if process.returncode != 0:
            detail = stderr.decode(errors="replace")[-2000:]
            raise ToolExecutionError(
                f"container operation {self.config.operation!r} failed: {detail}"
            )
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(
                f"container operation returned invalid JSON: {stdout[:200]!r}"
            ) from exc


def configure_container_tools(registry: ToolRegistry) -> None:
    tools = [
        Tool(
            id="formal.lean_check",
            namespace="formal",
            description="Compile a Lean 4 theorem in a fixed Mathlib environment and return diagnostics.",
            parameters_schema={
                "type": "object",
                "required": ["source"],
                "properties": {"source": {"type": "string", "maxLength": 50000}},
                "additionalProperties": False,
            },
            executor=ContainerExecutor(
                ContainerToolConfig("reagents/reasoning-core:latest", "lean_check")
            ),
            provider=ToolProvider.CONTAINER,
            access=ToolAccess.COMPUTE,
            risk_tier=RiskTier.LOW,
        ),
        Tool(
            id="formal.z3_solve",
            namespace="formal",
            description="Solve a bounded SMT-LIB2 constraint problem with Z3 and return stdout.",
            parameters_schema={
                "type": "object",
                "required": ["smt2"],
                "properties": {"smt2": {"type": "string", "maxLength": 100000}},
                "additionalProperties": False,
            },
            executor=ContainerExecutor(
                ContainerToolConfig("reagents/reasoning-core:latest", "z3_solve")
            ),
            provider=ToolProvider.CONTAINER,
            access=ToolAccess.COMPUTE,
            risk_tier=RiskTier.LOW,
        ),
        Tool(
            id="biology.sequence_stats",
            namespace="biology",
            description="Calculate deterministic sequence length, alphabet, composition, and GC fraction.",
            parameters_schema={
                "type": "object",
                "required": ["sequence"],
                "properties": {"sequence": {"type": "string", "maxLength": 1000000}},
                "additionalProperties": False,
            },
            executor=ContainerExecutor(
                ContainerToolConfig("reagents/biology-core:latest", "sequence_stats")
            ),
            provider=ToolProvider.CONTAINER,
            access=ToolAccess.COMPUTE,
            risk_tier=RiskTier.LOW,
        ),
        Tool(
            id="chemistry.rdkit_descriptors",
            namespace="chemistry",
            description="Calculate standard RDKit molecular descriptors for one SMILES string.",
            parameters_schema={
                "type": "object",
                "required": ["smiles"],
                "properties": {"smiles": {"type": "string", "maxLength": 10000}},
                "additionalProperties": False,
            },
            executor=ContainerExecutor(
                ContainerToolConfig("reagents/biology-core:latest", "rdkit_descriptors")
            ),
            provider=ToolProvider.CONTAINER,
            access=ToolAccess.COMPUTE,
            risk_tier=RiskTier.LOW,
        ),
        Tool(
            id="design.proto_check",
            namespace="design",
            description="Validate that the pinned Proto runtime can load and report its available primitives.",
            parameters_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            executor=ContainerExecutor(
                ContainerToolConfig("reagents/proto-design:latest", "proto_check")
            ),
            provider=ToolProvider.CONTAINER,
            access=ToolAccess.COMPUTE,
            risk_tier=RiskTier.MODERATE,
            latency_class="batch",
        ),
        Tool(
            id="reasoning.python",
            namespace="reasoning",
            description=(
                "Execute Python in the offline reasoning image with NumPy, SciPy, "
                "SymPy, NetworkX, Pint, CVXPY, and python-control available."
            ),
            parameters_schema=_python_schema(),
            executor=ContainerExecutor(
                ContainerToolConfig("reagents/reasoning-core:latest", "python_exec")
            ),
            provider=ToolProvider.CONTAINER,
            access=ToolAccess.COMPUTE,
            risk_tier=RiskTier.HIGH,
            latency_class="batch",
        ),
        Tool(
            id="biology.python",
            namespace="biology",
            description=(
                "Execute Python in the offline biology image with Biopython, RDKit, "
                "COBRApy, Scanpy, PyMC, and OpenMM available."
            ),
            parameters_schema=_python_schema(),
            executor=ContainerExecutor(
                ContainerToolConfig("reagents/biology-core:latest", "python_exec")
            ),
            provider=ToolProvider.CONTAINER,
            access=ToolAccess.COMPUTE,
            risk_tier=RiskTier.HIGH,
            latency_class="batch",
        ),
        Tool(
            id="engineering.python",
            namespace="engineering",
            description=(
                "Execute Python in the offline engineering image with Cantera, "
                "OpenMM, SciPy, and numerical tools available."
            ),
            parameters_schema=_python_schema(),
            executor=ContainerExecutor(
                ContainerToolConfig("reagents/biology-core:latest", "python_exec")
            ),
            provider=ToolProvider.CONTAINER,
            access=ToolAccess.COMPUTE,
            risk_tier=RiskTier.HIGH,
            latency_class="batch",
        ),
        Tool(
            id="design.proto_run",
            namespace="design",
            description=(
                "Execute a Proto Python program in the pinned offline Proto image. "
                "Only stdout/stderr are returned; the filesystem is disposable."
            ),
            parameters_schema=_python_schema(),
            executor=ContainerExecutor(
                ContainerToolConfig("reagents/proto-design:latest", "python_exec")
            ),
            provider=ToolProvider.CONTAINER,
            access=ToolAccess.COMPUTE,
            risk_tier=RiskTier.HIGH,
            latency_class="batch",
        ),
    ]
    for tool in tools:
        registry.register(tool)


def _python_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["source"],
        "properties": {"source": {"type": "string", "maxLength": 50000}},
        "additionalProperties": False,
    }
