"""The trust asymmetry, enforced by a test rather than by discipline.

A DEMI_GOD must never hold a Modal credential. Modal tokens are workspace-wide:
a demigod with one could spawn sandboxes and read every sibling's output volume,
which dissolves the isolation the whole system exists to provide.

The defence is structural. The sandbox image gets `demigod` and nothing else
(`demigod.images.PrebakedImage.build`), so if no module under `demigod/toolbox/`
imports Modal, there is no Modal client in the sandbox to hand a token to. That
is only true as long as nobody adds an import, and "nobody adds an import" is
exactly the kind of rule that decays -- so it is asserted here instead.

These tests read source. That is deliberate: importing a module and checking
`sys.modules` would pass on a lazy `import modal` inside a function body, which
is the form the mistake would actually take.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
DEMIGOD = SRC / "demigod"


def imported_modules(path: Path) -> set[str]:
    """Every module named by an import anywhere in the file, including inside
    functions and `if TYPE_CHECKING` blocks."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def toolbox_sources() -> list[Path]:
    return sorted((DEMIGOD / "toolbox").rglob("*.py"))


def test_there_is_something_to_check():
    assert len(toolbox_sources()) >= 4


@pytest.mark.parametrize("path", toolbox_sources(), ids=lambda p: p.name)
def test_the_demigod_toolbox_never_imports_modal(path: Path):
    """The load-bearing one. `demigod/toolbox/` is shipped into every sandbox."""
    assert "modal" not in imported_modules(path), (
        f"{path} imports modal. Nothing shipped into a DEMI_GOD sandbox may -- "
        f"a Modal client there is one env var away from workspace-wide "
        f"credentials, and workspace-wide credentials mean it can read every "
        f"sibling's output volume."
    )


@pytest.mark.parametrize("path", toolbox_sources(), ids=lambda p: p.name)
def test_the_demigod_toolbox_never_imports_reagents_or_broker(path: Path):
    """The dependency direction from the repo's own layout: `demigod` never
    imports `reagents`, so GOD's planner prompts and inverse maps cannot be read
    by the agent they constrain. `broker` is the same rule, one layer out."""
    imported = imported_modules(path)
    assert "reagents" not in imported
    assert "broker" not in imported


@pytest.mark.parametrize("path", toolbox_sources(), ids=lambda p: p.name)
def test_the_client_depends_on_nothing_a_bake_could_miss(path: Path):
    """stdlib + pydantic only. Every extra dependency is one that can fail to
    install, drift between bakes, or collide with a tool package -- inside an
    image an agent is already spending billed turns in."""
    allowed = {"demigod", "pydantic", "__future__"}
    stdlib = {
        "argparse",
        "json",
        "os",
        "pathlib",
        "sys",
        "time",
        "typing",
        "urllib",
    }
    unexpected = imported_modules(path) - allowed - stdlib
    assert not unexpected, f"{path.name} imports {sorted(unexpected)}"


def test_the_broker_package_is_never_added_to_a_demigod_image():
    """`PrebakedImage.build` must ship `demigod` alone. Adding `broker` would
    put the grant store, the Modal handles, and the full tool catalog inside the
    thing they exist to constrain.

    Read from the AST, not from the text: the words `broker` and `reagents`
    appear all over the prose in that module, and a substring check would either
    fail on a comment or be loosened until it checked nothing.
    """
    tree = ast.parse((DEMIGOD / "images.py").read_text(encoding="utf-8"))
    shipped: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_local_python_source"
        ):
            shipped += [arg.value for arg in node.args if isinstance(arg, ast.Constant)]

    assert shipped == ["demigod"], f"a DEMI_GOD image would ship {shipped}"


def test_the_runner_gives_the_sandbox_a_lease_and_no_modal_token():
    """`_agent_env` is the complete environment the agent process inherits."""
    from demigod.runner.inside import _agent_env
    from demigod.spec import DemiGodSpec, Problem
    from demigod.toolbox.protocol import ToolboxGrant

    spec = DemiGodSpec(
        name="probe-dg",
        domain="d",
        problem=Problem(context="c", goal="g"),
        toolbox=ToolboxGrant(url="https://b.example", lease_id="lease_abc"),
    )
    env = _agent_env(spec)

    assert env["TOOLBOX_LEASE"] == "lease_abc"
    assert env["TOOLBOX_URL"] == "https://b.example"
    assert not any("MODAL" in key.upper() for key in env)
    assert not any("TOKEN" in key.upper() for key in env if key != "TOOLBOX_LEASE")


def test_a_demigod_without_a_grant_gets_no_toolbox_environment():
    from demigod.runner.inside import AGENT_ENV, _agent_env
    from demigod.spec import DemiGodSpec, Problem

    spec = DemiGodSpec(
        name="probe-dg", domain="d", problem=Problem(context="c", goal="g")
    )
    assert _agent_env(spec) == AGENT_ENV
