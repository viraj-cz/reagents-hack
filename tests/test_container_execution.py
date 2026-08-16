"""CONTAINER-provider tools execute without a Docker daemon. Offline, always.

The bug this pins down: `ContainerExecutor` shelled out to `docker run`, and the
one machine that is supposed to run these tools -- a TOOLBOX_BROKER executor
Function -- is itself a container with no daemon inside it. Nine registered
tools were reachable through the broker and executable by nobody.

What can and cannot be proven from a laptop:

  CAN   the path is chosen by the environment, not by the tool
  CAN   the in-process path runs the real `tool_runtime` operation
  CAN   a missing scientific package produces an actionable message
  CAN   the Docker path is still there and still chosen locally
  CANNOT  that a tier's image actually contains RDKit

The last one is `scripts/preflight_executor.py`, live, one tier per run.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from reagents.tools import tool_runtime
from reagents.tools.container import (
    CONTAINER,
    INPROCESS,
    RUNTIME_ENV_VAR,
    ContainerExecutor,
    ContainerToolConfig,
    runtime_mode,
)
from reagents.tools.registry import ToolExecutionError, ToolRegistry, default_registry


def call(executor: ContainerExecutor, **arguments):
    return asyncio.run(executor({**arguments}))


def container_registry(monkeypatch: pytest.MonkeyPatch) -> ToolRegistry:
    monkeypatch.setenv("REAGENTS_ENABLE_CONTAINERS", "1")
    return default_registry()


# --- which path, and why ------------------------------------------------------


def test_explicit_env_var_wins_over_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RUNTIME_ENV_VAR, INPROCESS)
    assert runtime_mode() == INPROCESS
    monkeypatch.setenv(RUNTIME_ENV_VAR, CONTAINER)
    assert runtime_mode() == CONTAINER


def test_a_laptop_defaults_to_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    """`auto` on a developer machine must not silently drop the sandboxing.

    In-process execution is safe in a Modal executor because the Modal container
    IS the boundary. On a laptop there is no boundary, so `python_exec` running
    in-process would execute agent-authored code against the developer's home
    directory. The default has to be the isolated one.
    """
    monkeypatch.delenv(RUNTIME_ENV_VAR, raising=False)
    assert runtime_mode() == CONTAINER

    monkeypatch.setenv(RUNTIME_ENV_VAR, "nonsense")
    assert runtime_mode() == CONTAINER


def test_modal_container_detection_drives_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    """`auto` follows `modal.is_local()`, which is False only inside a Function."""
    import modal

    monkeypatch.delenv(RUNTIME_ENV_VAR, raising=False)
    monkeypatch.setattr(modal, "is_local", lambda: False)
    assert runtime_mode() == INPROCESS


# --- the in-process path ------------------------------------------------------


def test_inprocess_runs_the_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RUNTIME_ENV_VAR, INPROCESS)
    # If anything reached for a runtime binary this would be the way in.
    monkeypatch.setattr("shutil.which", lambda _name: None)

    executor = ContainerExecutor(
        ContainerToolConfig("reagents/biology-core:latest", "sequence_stats")
    )
    result = call(executor, sequence="ACGTACGTGG")
    assert result["length"] == 10
    assert result["is_dna"] is True
    assert result["gc_fraction"] == pytest.approx(0.6)


def test_inprocess_python_exec_still_uses_a_child_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`python_exec` must not `exec()` agent code into the broker's own process.

    In-process means "no Docker", not "no isolation at all": the operation still
    writes the program to a temp dir and runs it under `python -I`, so a syntax
    error or a `sys.exit` is the child's problem and the executor keeps serving.
    """
    monkeypatch.setenv(RUNTIME_ENV_VAR, INPROCESS)
    executor = ContainerExecutor(
        ContainerToolConfig("reagents/reasoning-core:latest", "python_exec")
    )

    ok = call(executor, source="print(6 * 7)")
    assert ok["ok"] is True
    assert ok["stdout"].strip() == "42"

    # A failing program is a RESULT, not an exception: the agent needs the
    # traceback in order to fix its own code.
    bad = call(executor, source="raise SystemExit(3)")
    assert bad["ok"] is False
    assert bad["returncode"] == 3


def test_missing_scientific_package_names_the_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure mode of a mis-specified tier, made legible.

    This is exactly GAP 1: `chemistry` was served by a tier with no RDKit in it.
    The message has to say which package and where to add it, because the person
    reading it is looking at a broker log, not at this file.
    """
    monkeypatch.setenv(RUNTIME_ENV_VAR, INPROCESS)
    executor = ContainerExecutor(
        ContainerToolConfig("reagents/biology-core:latest", "rdkit_descriptors")
    )
    if _importable("rdkit"):
        pytest.skip("rdkit is installed here, so the missing-package path is moot")

    with pytest.raises(ToolExecutionError) as excinfo:
        call(executor, smiles="CCO")
    message = str(excinfo.value)
    assert "rdkit" in message
    assert "ExecutorClass" in message


def test_unknown_operation_is_reported_as_such(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RUNTIME_ENV_VAR, INPROCESS)
    executor = ContainerExecutor(ContainerToolConfig("img", "not_an_operation"))
    with pytest.raises(ToolExecutionError, match="not implemented"):
        call(executor)


def test_operation_failure_becomes_a_tool_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RUNTIME_ENV_VAR, INPROCESS)
    executor = ContainerExecutor(
        ContainerToolConfig("reagents/biology-core:latest", "sequence_stats")
    )
    # `sequence` is required by the schema; the router validates, but the
    # executor must not leak a raw KeyError if something calls it directly.
    with pytest.raises(ToolExecutionError, match="failed"):
        call(executor)


def test_inprocess_honours_the_output_byte_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RUNTIME_ENV_VAR, INPROCESS)
    executor = ContainerExecutor(
        ContainerToolConfig(
            "reagents/reasoning-core:latest", "python_exec", max_output_bytes=256
        )
    )
    with pytest.raises(ToolExecutionError, match="byte limit"):
        call(executor, source="print('x' * 5000)")


def test_results_are_json_round_tripped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both paths must return the same TYPES for the same tool.

    The Docker path serializes with `json.dumps(default=str)` and the caller
    parses it. An in-process path that returned live Python objects would hand
    Modal a return value it cannot serialize, or the router a response it cannot
    encode -- and only for the tools whose operations happen to return exotic
    types, which is the worst possible distribution of a bug.
    """
    monkeypatch.setitem(tool_runtime.OPERATIONS, "exotic", lambda _p: {"p": object()})
    monkeypatch.setenv(RUNTIME_ENV_VAR, INPROCESS)
    executor = ContainerExecutor(ContainerToolConfig("img", "exotic"))
    result = call(executor)
    assert isinstance(result["p"], str)
    json.dumps(result)  # would raise if the round trip had been skipped


# --- the Docker path is still the Docker path ---------------------------------


def test_local_path_still_shells_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleting the Docker option was never the plan; this proves it survived."""
    monkeypatch.setenv(RUNTIME_ENV_VAR, CONTAINER)
    monkeypatch.delenv("REAGENTS_CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)

    executor = ContainerExecutor(
        ContainerToolConfig("reagents/biology-core:latest", "sequence_stats")
    )
    with pytest.raises(ToolExecutionError, match="no container runtime found"):
        call(executor, sequence="ACGT")


def test_local_path_error_points_at_the_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RUNTIME_ENV_VAR, CONTAINER)
    monkeypatch.delenv("REAGENTS_CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    executor = ContainerExecutor(ContainerToolConfig("img", "sequence_stats"))
    with pytest.raises(ToolExecutionError) as excinfo:
        call(executor, sequence="ACGT")
    assert RUNTIME_ENV_VAR in str(excinfo.value)


# --- through the registry, as the broker's executor reaches it -----------------


def test_every_container_tool_is_callable_in_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The nine registered CONTAINER tools all reach a real operation.

    Not "all succeed" -- most need a scientific package this test environment
    does not have. What must hold is that none of them fails for a STRUCTURAL
    reason: no missing operation, and above all no "no container runtime found",
    which is what every one of them did before.
    """
    registry = container_registry(monkeypatch)
    monkeypatch.setenv(RUNTIME_ENV_VAR, INPROCESS)
    monkeypatch.setattr("shutil.which", lambda _name: None)

    container_tools = [
        registry.get(tool_id)
        for tool_id in registry.ids()
        if registry.get(tool_id).provider.value == "container"
    ]
    assert len(container_tools) == 9

    for tool in container_tools:
        executor = tool.executor
        assert isinstance(executor, ContainerExecutor)
        assert executor.config.operation in tool_runtime.OPERATIONS, tool.id
        try:
            asyncio.run(tool.call_async(**_probe_arguments(tool.parameters_schema)))
        except ToolExecutionError as exc:
            assert "no container runtime found" not in str(exc), tool.id


def _probe_arguments(schema: dict) -> dict:
    """Minimal schema-satisfying input. Enough to reach the operation."""
    return dict.fromkeys(schema.get("required", []), "ACGT")


def _importable(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None
