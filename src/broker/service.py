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
import pathlib
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


BROKER_ENV: dict[str, str] = {
    # The broker IS the place container-provider tools are meant to run, so its
    # catalog must contain them. Without this the router answers `unknown_tool`
    # for every one of the nine, and the executor Functions cannot even look one
    # up -- `default_registry()` gates them behind exactly this flag.
    "REAGENTS_ENABLE_CONTAINERS": "1",
    # And they must run IN PROCESS here. A Modal container has no Docker daemon;
    # `docker run` is the laptop path. See reagents.tools.container.
    "REAGENTS_TOOL_RUNTIME": "inprocess",
}


TOOL_RUNTIME_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "reagents"
    / "tools"
    / "tool_runtime.py"
)
"""The ONE file a source-free image gets. Resolved from this module rather than
imported, because importing it would require `reagents` to be importable here --
which is exactly what a source-free image does not have."""

TOOL_RUNTIME_REMOTE = "/opt/reagents/tool_runtime.py"


def broker_image(
    *,
    source_free: bool = False,
    extras: tuple[str, ...] = (),
    apt: tuple[str, ...] = (),
    setup_commands: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
) -> modal.Image:
    """The router image, or an executor image with a tool class's dependencies.

    `reagents` AND `demigod` are both shipped: the broker is the one place that
    legitimately holds both halves. Note that this is the ONLY image where that
    is true -- a demigod image gets `demigod` alone, which is what keeps GOD's
    planner prompts and inverse maps unreadable by the agents they constrain.

    `ignore=[]` for the same reason `demigod.images` uses it: the default
    `NON_PYTHON_FILES` would silently drop the agent-facing markdown docs.

    LAYER ORDER IS THE COST MODEL, not a style choice. Modal content-hashes each
    layer, so everything expensive is placed BELOW the source layer: apt, then
    the multi-minute `setup_commands` (Lean's mathlib cache), then pip, and only
    then `add_local_python_source`. Editing a docstring in `reagents` therefore
    rebuilds one cheap layer instead of re-downloading mathlib.
    """
    image = modal.Image.debian_slim(python_version=PYTHON_VERSION)
    if apt:
        image = image.apt_install(*apt)
    if setup_commands:
        image = image.run_commands(*setup_commands)
    image = image.pip_install(PYDANTIC_PIN)
    if extras:
        image = image.pip_install(*extras)
    image = image.env({**BROKER_ENV, **(env or {})})
    if not source_free:
        return image.add_local_python_source("reagents", "demigod", "broker", ignore=[])
    # Source-free: ship ONE file, not a package. `tool_runtime` imports nothing
    # from `reagents` -- that is what makes this possible, and why that module
    # says so at the top. Anything running agent-authored code in this image
    # finds no GOD source to read back through a tool result.
    return image.add_local_file(
        str(TOOL_RUNTIME_SOURCE), TOOL_RUNTIME_REMOTE, copy=True
    )


@dataclass(frozen=True)
class ExecutorClass:
    """One tier: a set of tools that share an image, and optionally a GPU.

    Keyed primarily by REGISTRY NAMESPACE. Namespaces already group tools by the
    dependency stack they need (`reasoning` wants SciPy/SymPy/Z3, `biology`
    wants RDKit/Biopython), which is exactly the axis an image splits on. Adding
    a tool to an existing namespace therefore needs no change here.

    `tool_ids` is the exception, and it exists because `formal` is not one
    stack. `formal.z3_solve` needs a 40MB pip package and answers in
    milliseconds; `formal.lean_check` needs Lean 4 plus a compiled mathlib.
    Putting both in one image would make every Z3 call re-pull multiple
    gigabytes after a scaledown for a dependency it never touches. So a tier may
    claim individual tool ids, and those claims win over namespace membership.
    """

    name: str
    namespaces: frozenset[str] = frozenset()
    tool_ids: frozenset[str] = frozenset()
    """Tools this tier owns outright, whatever namespace they are in."""

    extras: tuple[str, ...] = ()
    apt: tuple[str, ...] = ()
    setup_commands: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    """Extra image env, as pairs so the class stays hashable/frozen."""

    source_free: bool = True
    """Ship NO repo source into this tier's image. Default ON.

    A tier that runs agent-authored code (any tool whose operation is
    `python_exec`) must not have `reagents` on disk. This was PROVEN live, not
    theorised: a `python -I` child inside the reasoning executor read
    /root/reagents/god/planner.py and printed 7,155 bytes of GOD's planner.
    `python -I` isolates imports; it does not restrict filesystem reads.

    That defeats the single property the architecture exists to protect --
    `demigod` is the only package shipped into an agent container precisely so
    `reagents` is unreachable, enforced by tests/test_package_boundary.py.

    Default TRUE rather than opt-in, because the failure is silent: a tier added
    later with a code-running tool would inherit protection automatically
    instead of needing someone to remember. Only a tier that genuinely needs to
    import `reagents` at execution time (the MCP/sponsor tier) sets it False.
    """

    enable_env: str | None = None
    """Name of an env var that must be truthy for this tier to be created.

    `None` means always on. A gated tier is still resolvable by `class_for`, so
    calling one of its tools produces "this tier exists and is switched off"
    rather than "no executor serves this namespace" -- the difference between a
    configuration answer and a mystery.
    """

    gpu: str | None = None
    timeout_s: int = 900
    max_containers: int = 4
    scaledown_window: int = 120
    """Shorter than the router's. An executor replica is expensive (big image,
    possibly a GPU) and is used in bursts, so it should let go sooner."""

    memory_mb: int = 4096
    cpu: float = 2.0

    def enabled(self) -> bool:
        if self.enable_env is None:
            return True
        return os.environ.get(self.enable_env, "").strip().lower() in {
            "1",
            "true",
            "yes",
        }

    def serves(self, tool_id: str, namespace: str) -> bool:
        return tool_id in self.tool_ids or namespace in self.namespaces

    def image(self) -> modal.Image:
        return broker_image(
            source_free=self.source_free,
            extras=self.extras,
            apt=self.apt,
            setup_commands=self.setup_commands,
            env=dict(self.env),
        )


REASONING = ExecutorClass(
    name="reasoning",
    namespaces=frozenset({"formal", "reasoning"}),
    # Mirrors the `reasoning` optional-dependency group in pyproject.toml, and
    # it has to: `reasoning.python` advertises exactly this list to the agent,
    # and `formal.z3_solve` shells out to the `z3` binary that the `z3-solver`
    # wheel installs onto PATH (verified against the wheel, not assumed).
    extras=(
        "numpy>=2,<3",
        "scipy>=1.14,<2",
        "sympy>=1.13,<2",
        "networkx>=3.3,<4",
        "pint>=0.24,<1",
        "cvxpy>=1.6,<2",
        "z3-solver>=4.15.4,<4.15.5",
        "control>=0.10,<1",
    ),
)

LEAN_TOOLCHAIN = "v4.30.0"
"""Pinned to `tooling/reasoning/Dockerfile`, so the local image and this tier
compile against the same Lean and the same mathlib."""

LEAN = ExecutorClass(
    name="lean",
    # No namespace: this tier owns ONE tool. `formal.z3_solve` stays on the
    # light reasoning image and never waits behind mathlib.
    tool_ids=frozenset({"formal.lean_check"}),
    apt=("ca-certificates", "curl", "git", "zstd"),
    setup_commands=(
        "curl -fsSL "
        "https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh "
        f"| sh -s -- -y --default-toolchain {LEAN_TOOLCHAIN}",
        # elan installs shims under /root/.elan/bin. Symlinks rather than a PATH
        # override: `Image.env` sets a literal string, so writing
        # "/root/.elan/bin:$PATH" there would put the characters `$PATH` on the
        # path and lose /usr/local/bin -- which is where python lives.
        "ln -sf /root/.elan/bin/elan /usr/local/bin/elan",
        "ln -sf /root/.elan/bin/lake /usr/local/bin/lake",
        "ln -sf /root/.elan/bin/lean /usr/local/bin/lean",
        f"git clone --depth 1 --branch {LEAN_TOOLCHAIN} "
        "https://github.com/leanprover-community/mathlib4.git /opt/mathlib",
        # The one expensive step: downloads prebuilt .olean artifacts instead of
        # compiling mathlib. `tool_runtime.lean_check` looks for exactly this
        # directory and runs `lake env lean` inside it.
        "cd /opt/mathlib && lake exe cache get",
    ),
    memory_mb=8192,
    cpu=4.0,
    # Two, not four: each replica holds a multi-gigabyte image, and a fan-out of
    # demigods all proving theorems at once is not the load this system has.
    max_containers=2,
)

BIOLOGY = ExecutorClass(
    name="biology",
    namespaces=frozenset({"biology", "chemistry"}),
    # `chemistry.rdkit_descriptors` imports rdkit; without it this tier served a
    # namespace it could not execute. The rest is what `biology.python`
    # advertises. Scanpy/PyMC/OpenMM are deliberately absent -- see the tool
    # description in reagents.tools.container for why.
    extras=(
        "numpy>=2,<3",
        "scipy>=1.14,<2",
        "pandas>=2,<3",
        "biopython>=1.84,<2",
        "rdkit>=2025.3,<2027",
        "cobra>=0.29,<1",
        "statsmodels>=0.14,<1",
        "scikit-learn>=1.6,<2",
    ),
    memory_mb=8192,
)

ENGINEERING = ExecutorClass(
    name="engineering",
    # `engineering.python` was registered and reachable and had NO tier at all:
    # every call died in `_dispatch_remote` with "no executor class serves
    # namespace 'engineering'".
    namespaces=frozenset({"engineering"}),
    extras=(
        "numpy>=2,<3",
        "scipy>=1.14,<2",
        "sympy>=1.13,<2",
        "pint>=0.24,<1",
        "cantera>=3.1,<4",
    ),
    memory_mb=8192,
)

PROTO_REF = "edf64afbcf84cc7c5e4e1404418c8ef1f16c34ce"
"""Pinned to `tooling/proto/Dockerfile`."""

DESIGN = ExecutorClass(
    name="design",
    namespaces=frozenset({"design"}),
    # Sponsor code, installed from a git ref and compiled from source. Gated
    # OFF: an image that fails to build fails `modal deploy` for the whole App,
    # and taking Z3 and RDKit down with a sponsor dependency nobody has verified
    # is a bad trade. Set REAGENTS_BROKER_PROTO=1 to opt in.
    enable_env="REAGENTS_BROKER_PROTO",
    apt=("ca-certificates", "cmake", "curl", "g++", "gcc", "git", "make"),
    extras=(f"git+https://github.com/evo-design/proto-language.git@{PROTO_REF}",),
    env=(("PROTO_HOME", "/proto"),),
    memory_mb=8192,
)

ESM_MODEL = "esm2_t12_35M_UR50D"
"""Baked into the image so a cold start does not pay the download.

The 35M checkpoint runs on CPU in seconds, which is why this tier has no `gpu=`
and therefore costs nothing at min_containers=0. Switch to
`esm2_t33_650M_UR50D` for quality and add `gpu="A10G"` -- the accelerator is
then attached to THIS function alone, never held while a demigod is thinking.
"""

ESM = ExecutorClass(
    name="esm",
    namespaces=frozenset({"protein"}),
    # CPU torch: the GPU wheels are multiple GB and this tier does not use one.
    extras=(
        "torch>=2.2,<3",
        "fair-esm>=2.0,<3",
    ),
    setup_commands=(
        # Pre-download the checkpoint INTO the image. Left to runtime it would
        # be fetched on every cold start, inside a demigod's turn budget.
        'python -c "import esm, torch; esm.pretrained.' + ESM_MODEL + '()" || true',
    ),
    env=(("REAGENTS_ESM_MODEL", ESM_MODEL), ("TORCH_HOME", "/root/.cache/torch")),
    memory_mb=8192,
    timeout_s=600,
)

SPONSOR = ExecutorClass(
    name="sponsor",
    # The ONE tier that keeps repo source: executing an MCP tool means importing
    # `reagents.tools.mcp`. Safe because MCP tools forward a string to someone
    # else's server -- they expose no filesystem-read primitive to the agent,
    # which is the thing source_free defends against.
    source_free=False,
    # Remote MCP tools: no science stack, just an HTTP client. Isolated from the
    # science classes so a sponsor endpoint being slow cannot occupy a container
    # that a solver is queued behind.
    namespaces=frozenset({"paperclip"}),
    extras=("httpx>=0.28,<1", "mcp>=1.27,<2"),
    memory_mb=2048,
    cpu=1.0,
)

ALL_EXECUTOR_CLASSES: tuple[ExecutorClass, ...] = (
    ESM,
    REASONING,
    LEAN,
    BIOLOGY,
    ENGINEERING,
    DESIGN,
    SPONSOR,
)
"""Every tier this broker knows how to build, enabled or not.

Resolution order matters: `tool_ids` claims are checked across ALL of these
before any namespace match, so LEAN taking `formal.lean_check` does not depend
on where it sits in this tuple.
"""

EXECUTOR_CLASSES: tuple[ExecutorClass, ...] = tuple(
    klass for klass in ALL_EXECUTOR_CLASSES if klass.enabled()
)
"""The tiers this process will actually create Functions for. To add a GPU tier,
add an `ExecutorClass(..., gpu="A100")` -- the GPU is attached to that Function
alone, so it is never held while the router is idle or while a demigod is
thinking.

COST NOTE. `modal deploy` builds every image in this tuple, so a deploy pays for
LEAN's mathlib cache once. It is once: the heavy layers sit below
`add_local_python_source`, so editing this repo does not invalidate them.
"""


def namespace_of(tool_id: str) -> str:
    return tool_id.split(".", 1)[0] if "." in tool_id else "generic"


def class_for(tool_id: str) -> ExecutorClass | None:
    """Which tier owns a tool. Per-TOOL, because `formal` spans two images.

    Searches every class, including gated-off ones, so the caller can tell
    "no such tier" from "that tier is switched off".
    """
    for klass in ALL_EXECUTOR_CLASSES:
        if tool_id in klass.tool_ids:
            return klass
    namespace = namespace_of(tool_id)
    for klass in ALL_EXECUTOR_CLASSES:
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


def execute_tool(tool_id: str, arguments: dict[str, Any]) -> Any:
    """Run one tool to completion, in whatever image this process is.

    THE executor body, module-level rather than a closure so that a live check
    (`scripts/preflight_executor.py`) can run the exact code a deployed tier
    runs, against one tier's image, without bringing up the whole App and
    building every other tier to do it.

    Deliberately NOT lease-aware. The lease was already enforced by the router
    before dispatch, and giving the executor an opinion about authorization
    would mean two places that can disagree about it.

    For a CONTAINER-provider tool, `call_async` reaches `ContainerExecutor`,
    which reads `REAGENTS_TOOL_RUNTIME=inprocess` off this image and calls
    `tool_runtime` directly instead of shelling out to a Docker daemon that does
    not exist here. That env var is set by `broker_image`; nothing in this
    function knows or needs to know which path was taken.
    """
    import asyncio

    tool = build_registry().get(tool_id)
    return asyncio.run(tool.call_async(**arguments))


def execute_operation(operation: str, arguments: dict[str, Any]) -> Any:
    """Executor body for a SOURCE-FREE tier. Takes an operation, not a tool id.

    A tool id would have to be resolved through `build_registry()`, and building
    the registry imports `reagents` -- the very thing this tier's image
    deliberately does not contain. So the router, which does have the registry,
    resolves the operation and passes it here. The executor stays ignorant of
    the catalog, which is the point: nothing in this image can name, let alone
    read, GOD's side of the system.

    Loaded by path rather than imported as `reagents.tools.tool_runtime`,
    because there is no `reagents` package here -- only the single file that
    `broker_image` copied in.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "reagents_tool_runtime", TOOL_RUNTIME_REMOTE
    )
    if spec is None or spec.loader is None:  # pragma: no cover - image is built
        raise RuntimeError(f"tool runtime missing at {TOOL_RUNTIME_REMOTE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_operation(operation, arguments)


def _make_executor(klass: ExecutorClass) -> modal.Function:
    if klass.source_free:

        def execute(operation: str, arguments: dict[str, Any]) -> Any:
            return execute_operation(operation, arguments)

    else:

        def execute(tool_id: str, arguments: dict[str, Any]) -> Any:
            return execute_tool(tool_id, arguments)

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


def operation_of(tool_id: str) -> str | None:
    """The `tool_runtime` operation a CONTAINER tool runs, or None.

    Router-side only: reads the tool's ContainerExecutor config out of the
    registry. Returns None for a tool that is not container-backed, which a
    source-free tier cannot serve.
    """
    tool = build_registry().get(tool_id)
    config = getattr(getattr(tool, "executor", None), "config", None)
    return getattr(config, "operation", None)


async def _dispatch_remote(tool_id: str, arguments: dict[str, Any]) -> Any:
    """Route one call to the Function that owns its tool class.

    Called from the router replica. `.remote.aio` rather than `.remote` because
    the router is ASGI: a blocking call here would stall every other demigod
    sharing the replica for the duration of someone else's solve.
    """
    klass = class_for(tool_id)
    if klass is None:
        raise RuntimeError(
            f"no executor class serves namespace {namespace_of(tool_id)!r} "
            f"(tool {tool_id!r}). Add it to a class in "
            f"broker.service.ALL_EXECUTOR_CLASSES."
        )
    if klass.name not in _EXECUTORS:
        # Reachable only for a gated tier: the class exists, this deployment
        # chose not to build it. Say which switch, not "unknown tool".
        raise RuntimeError(
            f"tool {tool_id!r} belongs to executor class {klass.name!r}, which "
            f"is disabled in this deployment. Set "
            f"{klass.enable_env}=1 and redeploy broker.service to enable it."
        )
    if klass.source_free:
        # Resolve the operation HERE, where the registry exists. The executor
        # image has no `reagents` to resolve it with -- see execute_operation.
        operation = operation_of(tool_id)
        if operation is None:
            raise RuntimeError(
                f"tool {tool_id!r} is served by source-free tier {klass.name!r} "
                f"but has no container operation to run. A source-free tier can "
                f"only serve CONTAINER tools; give it a tier with "
                f"source_free=False, or make it a CONTAINER tool."
            )
        return await _EXECUTORS[klass.name].remote.aio(operation, arguments)
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
    "ALL_EXECUTOR_CLASSES",
    "APP_NAME",
    "BROKER_ENV",
    "EXECUTOR_CLASSES",
    "ExecutorClass",
    "app",
    "broker_image",
    "build_registry",
    "build_router",
    "class_for",
    "endpoint_url",
    "execute_tool",
    "namespace_of",
    "router",
]
