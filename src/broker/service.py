"""THE THIRD SANDBOX TYPE. GOD spawns, DEMI_GOD reasons, BROKER executes.

    uv run modal serve src/broker/service.py     # ephemeral, for development
    uv run modal deploy src/broker/service.py    # persistent

WHY A MODAL FUNCTION AND NOT A LONG-LIVED SANDBOX
-------------------------------------------------
A single broker box is a bottleneck with no upside. One machine serving N
demigods means one heavy tool call blocks everyone, and the machine bills
whether or not anything is running. `App.function` gives autoscaling
(`max_containers`), scale-to-zero (`min_containers=0` -- the default, and left
default deliberately), and per-function resources. That last one is the real
prize: a GPU tool gets `gpu=` on ITS function, so an A100 is held while the tool
runs and released while Claude thinks about the result.

The other rejected option was a `Sandbox` per tool call. `Sandbox.create` costs
seconds even warm; twenty calls is twenty spawns, and the demigod is paying for
that latency out of its turn budget.

WHAT RUNS WHERE
---------------
    router  (this image, no GPU, small)   cheap LOCAL tools, inline
    exec_*  (one per tool class)          everything else, on its own image

The router holds no tool state and no credentials beyond the Modal identity
every Function has. The demigod holds a URL and a lease id -- see
`demigod/toolbox/protocol.py` for why that asymmetry is the entire security
model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import modal

from broker.dispatch import DispatchPolicy
from broker.modal_store import ModalGrantStore
from broker.router import ToolboxRouter
from demigod.images import PYTHON_VERSION
from reagents.tools.registry import ToolRegistry, default_registry

APP_NAME = "toolbox-broker"
"""One App for the whole broker, so `modal app logs toolbox-broker` shows every
tool call across every demigod in one stream."""

ROUTER_MAX_CONTAINERS = 8
"""Enough to serve a fan-out of demigods without letting a runaway agent turn
the broker into an unbounded bill. Each replica is small and mostly idle."""

ROUTER_SCALEDOWN_WINDOW = 300
"""Keep a warm replica for five minutes. A demigod's calls come in bursts
separated by model latency; scaling to zero between them would pay a cold start
per call out of the agent's turn budget."""

ROUTER_TIMEOUT_S = 900
"""Per-request ceiling for the router. Must exceed the slowest inline tool plus
one executor cold start."""

PYDANTIC_PIN = "pydantic==2.13.4"
"""Matches `demigod.images.AGENT_RUNTIME`, so the wire types serialize
identically on both ends of the protocol."""


def broker_image(extras: tuple[str, ...] = ()) -> modal.Image:
    """The router image, or an executor image with a tool class's extras.

    `reagents` AND `demigod` are both shipped: the broker is the one place that
    legitimately holds both halves. Note that this is the ONLY image where that
    is true -- a demigod image gets `demigod` alone, which is what keeps GOD's
    planner prompts and inverse maps unreadable by the agents they constrain.

    `ignore=[]` for the same reason `demigod.images` uses it: the default
    `NON_PYTHON_FILES` would silently drop the agent-facing markdown docs.
    """
    image = modal.Image.debian_slim(python_version=PYTHON_VERSION).pip_install(
        PYDANTIC_PIN
    )
    if extras:
        image = image.pip_install(*extras)
    return image.add_local_python_source("reagents", "demigod", "broker", ignore=[])


@dataclass(frozen=True)
class ExecutorClass:
    """One tier: a set of tools that share an image, and optionally a GPU.

    Keyed by REGISTRY NAMESPACE rather than by tool id. Namespaces already group
    tools by the dependency stack they need (`reasoning` wants SciPy/SymPy/Z3,
    `biology` wants RDKit/Biopython), which is exactly the axis an image splits
    on. Adding a tool to an existing namespace therefore needs no change here.
    """

    name: str
    namespaces: frozenset[str]
    extras: tuple[str, ...] = ()
    gpu: str | None = None
    timeout_s: int = 900
    max_containers: int = 4
    scaledown_window: int = 120
    """Shorter than the router's. An executor replica is expensive (big image,
    possibly a GPU) and is used in bursts, so it should let go sooner."""

    memory_mb: int = 4096
    cpu: float = 2.0

    def image(self) -> modal.Image:
        return broker_image(self.extras)


REASONING = ExecutorClass(
    name="reasoning",
    namespaces=frozenset({"formal", "reasoning", "design"}),
    # Mirrors the `reasoning` optional-dependency group in pyproject.toml.
    extras=("numpy>=2,<3", "scipy>=1.14,<2", "sympy>=1.13,<2", "networkx>=3.3,<4"),
)

BIOLOGY = ExecutorClass(
    name="biology",
    namespaces=frozenset({"biology", "chemistry"}),
    extras=("biopython>=1.84,<2",),
    memory_mb=8192,
)

SPONSOR = ExecutorClass(
    name="sponsor",
    # Remote MCP tools: no science stack, just an HTTP client. Isolated from the
    # science classes so a sponsor endpoint being slow cannot occupy a container
    # that a solver is queued behind.
    namespaces=frozenset({"paperclip", "biomni"}),
    extras=("httpx>=0.28,<1", "mcp>=1.27,<2"),
    memory_mb=2048,
    cpu=1.0,
)

EXECUTOR_CLASSES: tuple[ExecutorClass, ...] = (REASONING, BIOLOGY, SPONSOR)
"""The catalog. To add a GPU tier, add an `ExecutorClass(..., gpu="A100")` --
the GPU is attached to that Function alone, so it is never held while the router
is idle or while a demigod is thinking.

Nothing here is built until a tool in one of these namespaces is actually
called: Modal builds an image lazily and content-hashes it, so an unused class
costs nothing.
"""


def class_for(namespace: str) -> ExecutorClass | None:
    for klass in EXECUTOR_CLASSES:
        if namespace in klass.namespaces:
            return klass
    return None


def build_registry() -> ToolRegistry:
    """The broker's catalog. Superset of any single lease.

    Container and MCP namespaces are opt-in via the same env flags
    `default_registry` already honours, so a broker deployed without sponsor
    credentials serves the builtins and refuses the rest by name instead of
    failing at import.
    """
    return default_registry()


def build_router(
    *,
    registry: ToolRegistry | None = None,
    remote: Any = None,
) -> ToolboxRouter:
    """Assemble the ASGI app. Also the local-dev entry point.

    Split out from the Modal decorators so the exact object served in production
    can be constructed in a test with an in-memory store -- the request path is
    then the same code, not a re-implementation of it.
    """
    return ToolboxRouter(
        registry=registry or build_registry(),
        store=ModalGrantStore(),
        dispatch=DispatchPolicy(remote=remote),
    )


app = modal.App(APP_NAME)


# --- executors ---------------------------------------------------------------
#
# One Modal Function per tool class, created in a loop. Registered before the
# router so the router's dispatch table can close over them.

_EXECUTORS: dict[str, modal.Function] = {}


def _make_executor(klass: ExecutorClass) -> modal.Function:
    def execute(tool_id: str, arguments: dict[str, Any]) -> Any:
        """Run one tool to completion in this class's image.

        Deliberately NOT lease-aware. The lease was already enforced by the
        router before dispatch, and giving the executor an opinion about
        authorization would mean two places that can disagree about it. This
        function is only reachable from inside the broker's own App.
        """
        import asyncio

        tool = build_registry().get(tool_id)
        return asyncio.run(tool.call_async(**arguments))

    execute.__name__ = f"execute_{klass.name}"
    return app.function(
        name=execute.__name__,
        image=klass.image(),
        gpu=klass.gpu,
        cpu=klass.cpu,
        memory=klass.memory_mb,
        timeout=klass.timeout_s,
        max_containers=klass.max_containers,
        scaledown_window=klass.scaledown_window,
        # min_containers is left unset (0). An idle tier must cost nothing --
        # especially a GPU tier.
        serialized=True,
    )(execute)


for _klass in EXECUTOR_CLASSES:
    _EXECUTORS[_klass.name] = _make_executor(_klass)


async def _dispatch_remote(tool_id: str, arguments: dict[str, Any]) -> Any:
    """Route one call to the Function that owns its tool class.

    Called from the router replica. `.remote.aio` rather than `.remote` because
    the router is ASGI: a blocking call here would stall every other demigod
    sharing the replica for the duration of someone else's solve.
    """
    namespace = tool_id.split(".", 1)[0] if "." in tool_id else "generic"
    klass = class_for(namespace)
    if klass is None or klass.name not in _EXECUTORS:
        raise RuntimeError(
            f"no executor class serves namespace {namespace!r} (tool {tool_id!r}). "
            f"Add it to a class in broker.service.EXECUTOR_CLASSES."
        )
    return await _EXECUTORS[klass.name].remote.aio(tool_id, arguments)


@app.function(
    image=broker_image(),
    timeout=ROUTER_TIMEOUT_S,
    max_containers=ROUTER_MAX_CONTAINERS,
    scaledown_window=ROUTER_SCALEDOWN_WINDOW,
    # min_containers deliberately unset: the broker costs nothing between runs.
    serialized=True,
)
@modal.asgi_app()
def router() -> Any:
    """The demigod-facing HTTP endpoint.

    `asgi_app`, not `fastapi_endpoint`: the latter requires FastAPI in the image
    (verified -- `modal/_runtime/asgi.py` imports it only on that path), and
    `broker.router` is a plain ASGI callable precisely so the broker image can
    stay Python plus this repo.
    """
    return build_router(remote=_dispatch_remote)


def endpoint_url() -> str:
    """The broker base URL, for GOD to put in a `ToolboxGrant`.

    `TOOLBOX_BROKER_URL` wins so a developer can point GOD at `modal serve`'s
    ephemeral URL without deploying. Otherwise this looks up the deployed
    Function -- which requires the caller to hold Modal credentials, and the
    caller is GOD. A demigod never runs this code path; it is handed the string.
    """
    override = os.environ.get("TOOLBOX_BROKER_URL")
    if override:
        return override.rstrip("/")
    handle = modal.Function.from_name(APP_NAME, "router")
    url = handle.get_web_url()
    if not url:
        raise RuntimeError(
            f"{APP_NAME}/router has no web URL. Deploy it with "
            f"`uv run modal deploy src/broker/service.py`, or set "
            f"TOOLBOX_BROKER_URL to a `modal serve` URL."
        )
    return url.rstrip("/")


__all__ = [
    "APP_NAME",
    "EXECUTOR_CLASSES",
    "ExecutorClass",
    "app",
    "broker_image",
    "build_registry",
    "build_router",
    "class_for",
    "endpoint_url",
    "router",
]
