"""Scientific tools whose payload is `reagents.tools.tool_runtime`, run two ways.

TWO EXECUTION PATHS, ONE PAYLOAD
--------------------------------
    docker run --rm --network none ... tool_runtime.py <op>     # a laptop
    tool_runtime.run_operation(<op>, arguments)                 # inside Modal

`docker run` was the only path, and it is wrong in exactly one place: the
TOOLBOX_BROKER's executor Functions. Those Functions ARE the container -- a
Modal container built from `broker.service.ExecutorClass.image()`, holding the
scientific stack, with no Docker daemon inside it and no way to get one. Every
CONTAINER-provider tool therefore failed there with "no container runtime
found", which is how nine registered tools could be reachable through the broker
and executable by nobody.

So the path is chosen by WHERE THIS CODE IS RUNNING, not by rewriting the tool:

    REAGENTS_TOOL_RUNTIME=inprocess   import and call. Set on executor images.
    REAGENTS_TOOL_RUNTIME=container   shell out. Forces the local path.
    unset / "auto"                    inprocess iff this is a Modal container.

The local Docker path is not a legacy branch to be deleted later. It is the only
one of the two that gives a laptop `--network none`, `--read-only`,
`--cap-drop ALL` and a pid limit; in Modal the container boundary IS the Modal
container, which is why running in-process there is not a downgrade.

WHAT IN-PROCESS DOES NOT GIVE YOU. A wall-clock timeout that can actually stop
the work. Each operation carries its own `subprocess` timeout (45-50s) which
covers the cases that hang in practice -- Z3, Lean, and user Python all run as
child processes -- but a pure-Python operation that spun forever would hold the
thread until the Modal Function's own `timeout=` killed the container. That is
the backstop, and it is why `ExecutorClass.timeout_s` matters.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any

from reagents.contracts import RiskTier, ToolAccess, ToolProvider
from reagents.tools.registry import Tool, ToolExecutionError, ToolRegistry

RUNTIME_ENV_VAR = "REAGENTS_TOOL_RUNTIME"
"""`inprocess`, `container`, or `auto` (the default). Set to `inprocess` on
every broker executor image by `broker.service.broker_image`."""

INPROCESS = "inprocess"
CONTAINER = "container"


def _in_modal_container() -> bool:
    """Is this process a Modal Function container?

    `modal.is_local()` is the supported answer and is verified against the
    installed client: it returns False ONLY inside a running Modal Function --
    True on a laptop, True inside a Sandbox, and True in a child process of a
    Function. That last case is exactly right for us: a demigod's Sandbox must
    NOT take the in-process path, because a Sandbox image is the agent runtime
    and has no scientific stack in it.

    Wrapped because `modal` is an install-time dependency of `reagents` but this
    module must stay importable in the disposable tool images, which have neither
    Modal nor the package.
    """
    try:
        import modal
    except Exception:  # pragma: no cover - modal is a hard dependency locally
        return False
    try:
        return not modal.is_local()
    except Exception:  # pragma: no cover - defensive; is_local() touches no I/O
        return False


def runtime_mode() -> str:
    """`INPROCESS` or `CONTAINER`. The one place the path is decided."""
    declared = os.environ.get(RUNTIME_ENV_VAR, "auto").strip().lower()
    if declared in {INPROCESS, CONTAINER}:
        return declared
    return INPROCESS if _in_modal_container() else CONTAINER


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
        if runtime_mode() == INPROCESS:
            return await self._run_inprocess(arguments)
        return await self._run_container(arguments)

    # --- the Modal path -----------------------------------------------------

    async def _run_inprocess(self, arguments: dict[str, Any]) -> Any:
        """Call the operation directly, in a worker thread.

        `to_thread` and not a bare call: the operations are synchronous and two
        of them block on a subprocess for tens of seconds. The broker's router
        is ASGI and its executor Function serves concurrent inputs, so blocking
        the event loop here would serialize every other tool call on the box.
        """
        from reagents.tools import tool_runtime

        try:
            result = await asyncio.to_thread(
                tool_runtime.run_operation, self.config.operation, dict(arguments)
            )
        except tool_runtime.UnknownOperationError as exc:
            raise ToolExecutionError(
                f"operation {self.config.operation!r} is not implemented by "
                f"tool_runtime in this image: {exc}"
            ) from exc
        except ModuleNotFoundError as exc:
            # The honest failure mode of a mis-specified executor image: the
            # operation is there, its scientific dependency is not. Name the
            # class so the fix is one edit away.
            raise ToolExecutionError(
                f"operation {self.config.operation!r} needs {exc.name!r}, which "
                f"is not installed in this executor image. Add it to the "
                f"matching broker.service.ExecutorClass extras."
            ) from exc
        except Exception as exc:
            raise ToolExecutionError(
                f"container operation {self.config.operation!r} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        encoded = json.dumps(result, default=str)
        if len(encoded.encode()) > self.config.max_output_bytes:
            raise ToolExecutionError("container tool output exceeded its byte limit")
        return result

    # --- the laptop path ----------------------------------------------------

    async def _run_container(self, arguments: dict[str, Any]) -> Any:
        runtime = os.environ.get("REAGENTS_CONTAINER_RUNTIME")
        if not runtime:
            runtime = shutil.which("docker") or shutil.which("podman")
        if not runtime:
            raise ToolExecutionError(
                "no container runtime found; install Docker or Podman, then run "
                f"`reagents tools doctor`. If this is a machine that already "
                f"provides the isolation (a Modal executor), set "
                f"{RUNTIME_ENV_VAR}={INPROCESS} instead."
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
            # Names ONLY what both execution paths guarantee. The local
            # biology-core image also carries Scanpy, PyMC and OpenMM; the Modal
            # biology tier deliberately does not, because each drags hundreds of
            # megabytes through a cold-start image pull. Promising a package the
            # brokered image lacks costs an agent a turn to discover.
            description=(
                "Execute Python in the offline biology image with Biopython, RDKit, "
                "COBRApy, NumPy, SciPy, pandas, scikit-learn, and statsmodels "
                "available."
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
            # OpenMM dropped from the promise for the same reason as above; it is
            # a molecular-dynamics engine and nothing in this namespace needs it.
            description=(
                "Execute Python in the offline engineering image with Cantera, "
                "NumPy, SciPy, SymPy, and Pint available."
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
