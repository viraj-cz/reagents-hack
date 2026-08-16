"""Every brokered tool has a tier, and every tier can run the tools it claims.

GAP 1, as a test. The dispatch machinery was complete and correct and still
could not execute a single container tool, because the mapping it dispatched
over was wrong in three separate ways:

    formal.z3_solve             -> a tier with no Z3 in it
    chemistry.rdkit_descriptors -> a tier with no RDKit in it
    engineering.python          -> no tier at all

None of those is visible from `dispatch.py`, which is why none of them was
caught: `DispatchPolicy` correctly routes a tool to a machine that correctly
fails to import its dependency.

Offline. Importing `broker.service` constructs a `modal.App` and image
DEFINITIONS, which is local work -- no credentials, no network, nothing built.
`scripts/preflight_executor.py` is where an image is proven to contain what this
file only asserts it was told to contain.
"""

from __future__ import annotations

import pytest

from broker.service import (
    ALL_EXECUTOR_CLASSES,
    BIOLOGY,
    BROKER_ENV,
    ENGINEERING,
    EXECUTOR_CLASSES,
    LEAN,
    NORMAN_V2,
    REASONING,
    class_for,
    namespace_of,
)
from reagents.contracts import ToolProvider
from reagents.tools.registry import ToolRegistry, default_registry


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> ToolRegistry:
    """The catalog as a broker container sees it: exactly `BROKER_ENV`."""
    for name, value in BROKER_ENV.items():
        monkeypatch.setenv(name, value)
    return default_registry()


def brokered_ids(registry: ToolRegistry) -> list[str]:
    """Tools that CANNOT run inline in the router, so must have a tier."""
    return [
        tool_id
        for tool_id in registry.ids()
        if registry.get(tool_id).provider is not ToolProvider.LOCAL
    ]


# --- coverage -----------------------------------------------------------------


def test_broker_env_is_what_makes_the_catalog_appear() -> None:
    """The flag is load-bearing: without it the broker has no container tools.

    `default_registry` gates them behind REAGENTS_ENABLE_CONTAINERS, so an
    executor image missing that env var answers `unknown_tool` for every tool it
    was built to serve.
    """
    assert BROKER_ENV["REAGENTS_ENABLE_CONTAINERS"] == "1"
    assert BROKER_ENV["REAGENTS_TOOL_RUNTIME"] == "inprocess"


def test_broker_env_actually_registers_the_container_tools(
    registry: ToolRegistry,
) -> None:
    container_tools = [
        tool_id
        for tool_id in registry.ids()
        if registry.get(tool_id).provider is ToolProvider.CONTAINER
    ]
    # NOT a hardcoded count. A magic number here fails every time anyone adds
    # a tool -- which is exactly what happened when ESM landed -- and it tests
    # the number rather than the property. What matters is that the flag
    # actually surfaces container tools at all.
    assert container_tools, "REAGENTS_ENABLE_CONTAINERS=1 registered no container tools"


def test_every_brokered_tool_resolves_to_a_tier(registry: ToolRegistry) -> None:
    """The regression that caught `engineering`, which had no class at all."""
    orphans = [
        tool_id for tool_id in brokered_ids(registry) if class_for(tool_id) is None
    ]
    assert not orphans, (
        f"{orphans} would reach `_dispatch_remote` and die with 'no executor "
        f"class serves namespace'. Add them to a class in ALL_EXECUTOR_CLASSES."
    )


def test_a_disabled_tier_names_its_switch(registry: ToolRegistry) -> None:
    """A gated tier is allowed; a SILENTLY gated tier is not.

    `design.proto_*` is deliberately off by default (sponsor code, built from a
    git ref). What must not happen is that being off looks like the tool never
    existed.
    """
    enabled = {klass.name for klass in EXECUTOR_CLASSES}
    for tool_id in brokered_ids(registry):
        klass = class_for(tool_id)
        assert klass is not None
        if klass.name not in enabled:
            assert klass.enable_env, (
                f"{tool_id} resolves to tier {klass.name!r}, which is not "
                f"enabled and names no env var to enable it"
            )


def test_tool_id_claims_are_not_typos(registry: ToolRegistry) -> None:
    """A tier claiming `formal.lean_chekc` would silently claim nothing."""
    known = set(registry.ids())
    for klass in ALL_EXECUTOR_CLASSES:
        unknown = sorted(klass.tool_ids - known)
        assert not unknown, f"tier {klass.name!r} claims unknown tools {unknown}"


def test_no_two_enabled_tiers_claim_the_same_namespace() -> None:
    seen: dict[str, str] = {}
    for klass in EXECUTOR_CLASSES:
        for namespace in klass.namespaces:
            assert namespace not in seen, (
                f"{namespace!r} is claimed by both {seen[namespace]!r} and "
                f"{klass.name!r}; `class_for` would resolve it by tuple order"
            )
            seen[namespace] = klass.name


# --- per-tool tiering ---------------------------------------------------------


def test_formal_splits_across_two_tiers() -> None:
    """The reason `class_for` is per-tool and not per-namespace.

    Z3 is a 40MB wheel answering in milliseconds; Lean is a compiled mathlib.
    Sharing an image would make every Z3 call re-pull gigabytes after a
    scaledown, for a dependency it never touches.
    """
    assert namespace_of("formal.z3_solve") == namespace_of("formal.lean_check")
    assert class_for("formal.z3_solve") is REASONING
    assert class_for("formal.lean_check") is LEAN


def test_a_tool_id_claim_beats_a_namespace_claim() -> None:
    assert "formal" in REASONING.namespaces
    assert "formal.lean_check" in LEAN.tool_ids
    assert class_for("formal.lean_check") is LEAN


def test_class_for_finds_disabled_tiers_too() -> None:
    """So the failure is 'that tier is off', not 'no such namespace'."""
    klass = class_for("design.proto_run")
    assert klass is not None
    assert not klass.enabled()
    assert klass.enable_env == "REAGENTS_BROKER_PROTO"


def test_gating_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    klass = class_for("design.proto_check")
    assert klass is not None
    monkeypatch.setenv(klass.enable_env, "1")
    assert klass.enabled()
    monkeypatch.setenv(klass.enable_env, "no")
    assert not klass.enabled()


def test_unknown_namespace_still_resolves_to_nothing() -> None:
    assert class_for("astrology.horoscope") is None


# --- images carry what their tools import -------------------------------------

REQUIRED = {
    # tier -> distributions its tools import at runtime, and the tool that
    # proves it. Names are matched against the pins in `extras`.
    "reasoning": ["z3-solver", "sympy", "numpy", "scipy", "networkx", "pint", "cvxpy"],
    "biology": ["rdkit", "biopython", "numpy", "scipy", "pandas"],
    "engineering": ["cantera", "numpy", "scipy"],
}


@pytest.mark.parametrize("tier_name", sorted(REQUIRED))
def test_tier_extras_cover_its_tools(tier_name: str) -> None:
    klass = next(k for k in ALL_EXECUTOR_CLASSES if k.name == tier_name)
    installed = {pin.split(">")[0].split("=")[0].split("<")[0] for pin in klass.extras}
    missing = sorted(set(REQUIRED[tier_name]) - installed)
    assert not missing, (
        f"tier {tier_name!r} serves tools that import {missing}, and its image "
        f"does not install them. This is GAP 1: dispatch works, the tool does "
        f"not."
    )


def test_z3_is_on_the_tier_that_serves_z3_solve() -> None:
    """The specific miss. `formal` was served by a tier with no solver in it."""
    assert any(pin.startswith("z3-solver") for pin in REASONING.extras)


def test_rdkit_is_on_the_tier_that_serves_chemistry() -> None:
    assert "chemistry" in BIOLOGY.namespaces
    assert any(pin.startswith("rdkit") for pin in BIOLOGY.extras)


def test_engineering_has_a_tier_of_its_own() -> None:
    assert ENGINEERING.namespaces == {"engineering"}
    assert class_for("engineering.python") is ENGINEERING


def test_norman_v2_labs_have_a_source_free_training_only_tier() -> None:
    assert NORMAN_V2.source_free
    assert class_for("screen2.algebra_lab") is NORMAN_V2
    assert class_for("screen2.geometry_lab") is NORMAN_V2
    assert class_for("screen2.graph_lab") is NORMAN_V2
    assert class_for("screen2.information_lab") is NORMAN_V2
    assert NORMAN_V2.local_files
    assert dict(NORMAN_V2.env)["REAGENTS_NORMAN_TRAINING_PATH"].startswith("/opt/")


def test_lean_tier_is_isolated_and_carries_its_toolchain() -> None:
    assert not LEAN.namespaces, "a namespace claim would drag z3 onto this image"
    assert any("elan-init" in command for command in LEAN.setup_commands)
    assert any("mathlib4" in command for command in LEAN.setup_commands)
    # `tool_runtime.lean_check` hardcodes this path and falls back to a bare
    # `lean` (no mathlib) if it is absent -- a silently weaker checker.
    assert any("/opt/mathlib" in command for command in LEAN.setup_commands)


def test_heavy_layers_sit_below_the_source_layer() -> None:
    """Not cosmetic: it is why Lean's build is a one-time cost.

    Modal content-hashes each layer. With apt and `setup_commands` placed under
    `add_local_python_source`, editing any file in this repo invalidates the
    mathlib download.
    """
    import inspect

    from broker.service import broker_image

    # The docstring names every one of these, so it has to come off first or
    # the test just re-reads the prose it is meant to be checking.
    body = inspect.getsource(broker_image).split('"""')[2]
    order = [
        body.index("apt_install"),
        body.index("run_commands"),
        body.index("pip_install"),
        body.index("add_local_python_source"),
    ]
    assert order == sorted(order)


def test_source_free_executor_closure_references_no_module_globals() -> None:
    """A source-free tier's executor must not close over anything importable.

    Regression for every source-free tier crash-looping in production while the
    deploy reported success:

        ModuleNotFoundError: No module named 'broker'
        Function .execute_esm is crash-looping

    `serialized=True` makes cloudpickle serialize the closure by value, but any
    GLOBAL it references is still pickled by reference to `broker.service` --
    and a source-free image has no `broker` package by construction. The
    closure called module-level `execute_operation`, so every replica died on
    startup.

    Asserted on `co_names` because that is the actual mechanism: the names a
    code object looks up at runtime are exactly what cloudpickle has to resolve.
    A comment saying "do not reference module scope" is not enforcement -- the
    next person adding a helper call here gets a test failure instead of a
    crash-loop discovered on a dashboard.
    """
    import broker.service as svc

    module_globals = {
        name
        for name, value in vars(svc).items()
        if (not name.startswith("__") and callable(value)) or isinstance(value, str)
    }

    # EXECUTOR_CLASSES, not ALL_: a tier gated off by `enable_env` (DESIGN)
    # is never registered, so it has no closure to inspect.
    source_free = [k for k in svc.EXECUTOR_CLASSES if k.source_free]
    assert source_free, "no source-free tier to check"

    for klass in source_free:
        # Rebuild the closure the same way _make_executor does, without needing
        # a live App: the code object is what gets serialized either way.
        closure = _extract_source_free_closure(svc, klass)
        offending = sorted(set(closure.__code__.co_names) & module_globals)
        assert not offending, (
            f"{klass.name}'s executor closure references module globals "
            f"{offending}; cloudpickle will pickle them by reference to "
            f"broker.service, which a source-free image cannot import"
        )


def _extract_source_free_closure(svc, klass):
    """The `execute` closure `_make_executor` builds for a source-free tier.

    Calls the real factory and reads the function back off the Modal Function,
    so the test checks the object that is actually deployed rather than a
    reconstruction of it.
    """
    fn = svc._EXECUTORS.get(klass.name)
    assert fn is not None, f"{klass.name} has no registered executor"
    for attr in ("_info", "info"):
        info = getattr(fn, attr, None)
        raw = getattr(info, "raw_f", None)
        if raw is not None:
            return raw
    raise AssertionError(
        "could not reach the raw function off the Modal Function object; "
        "Modal's internals moved and this test needs updating"
    )
