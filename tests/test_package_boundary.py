"""The GOD/DEMI_GOD image asymmetry, enforced offline.

Three packages, and the dependency direction between them is a security
boundary rather than a style preference:

    godbox   -> reagents -> demigod
    demigod  -> nothing

`demigod` is the only package shipped into a DEMI_GOD's container. The moment
it can reach `reagents`, that container holds the planner prompts, the
transform prompts, and the inverse maps -- the machinery that decides what the
agent is allowed to know and how its answer is translated back. And the moment
a demigod image installs the `modal` client, a leaked workspace token stops
being survivable.

Both properties are one careless import away, and neither fails loudly at
runtime: a demigod with `reagents` on its path works perfectly. It just is not
sealed any more. Hence tests.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from godbox.images import (
    FORBIDDEN_IN_DEMIGOD_IMAGE,
    GOD_LOCAL_SOURCES,
    GOD_PIP,
)

SRC = Path(__file__).resolve().parent.parent / "src"


def _imported_roots(path: Path) -> set[str]:
    """Top-level package names imported by one module, from its AST.

    AST rather than a text grep: `demigod/result.py` and `demigod/schema.py`
    both discuss `reagents` at length in their docstrings, and a grep would
    flag the very comments that explain the rule.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        # `from . import x` has no module; a relative import cannot cross a
        # package boundary, so only absolute ones are interesting here.
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _modules(package: str) -> list[Path]:
    found = sorted((SRC / package).rglob("*.py"))
    assert found, f"no modules found under src/{package}"
    return found


@pytest.mark.parametrize("module", _modules("demigod"), ids=lambda p: p.name)
def test_demigod_imports_nothing_from_god(module: Path) -> None:
    leaked = _imported_roots(module) & set(FORBIDDEN_IN_DEMIGOD_IMAGE)
    assert not leaked, (
        f"{module.relative_to(SRC)} imports {sorted(leaked)}. `demigod` is the "
        f"only package shipped into a DEMI_GOD's image, so this puts GOD's "
        f"planner prompts and inverse maps inside the container they are "
        f"meant to constrain."
    )


def test_demigod_images_ship_only_demigod() -> None:
    """`add_local_python_source` is where a package actually enters the image.

    An import guard alone is not enough: adding `"reagents"` to that call would
    ship the source without any module importing it.
    """
    tree = ast.parse((SRC / "demigod" / "images.py").read_text(encoding="utf-8"))
    added: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_local_python_source"
        ):
            for arg in node.args:
                assert isinstance(arg, ast.Constant), (
                    "a non-literal package name in the demigod image makes this "
                    "boundary unauditable"
                )
                added.add(arg.value)
    assert added == {"demigod"}, f"demigod images ship {sorted(added)}"


def test_demigod_images_do_not_install_the_modal_client() -> None:
    """The second barrier. Without the client, a demigod that somehow obtained
    a Modal token still cannot spawn a sandbox or mount a sibling's volume."""
    from demigod.images import AGENT_RUNTIME

    assert not [pkg for pkg in AGENT_RUNTIME if pkg.split("=")[0].strip() == "modal"]


def test_god_image_ships_all_three_packages() -> None:
    """GOD is the one place all three meet: it reasons with `reagents`, spawns
    with `demigod`, and reports with `godbox`."""
    assert set(GOD_LOCAL_SOURCES) == {"reagents", "demigod", "godbox"}


def test_god_image_installs_the_modal_client_pinned() -> None:
    """Nested spawning is the load-bearing assumption (scripts/preflight_nested.py).
    Its absence surfaces as `ModuleNotFoundError: No module named 'modal'`,
    which reads misleadingly like Modal forbidding nesting."""
    modal_pins = [pkg for pkg in GOD_PIP if pkg.startswith("modal")]
    assert len(modal_pins) == 1
    assert "==" in modal_pins[0], "unpinned dependencies break reproducibility"


@pytest.mark.parametrize("pin", GOD_PIP)
def test_god_image_dependencies_are_pinned(pin: str) -> None:
    assert "==" in pin, f"{pin!r} is not pinned to an exact version"


def test_god_image_pins_match_the_lockfile() -> None:
    """Two GOD sandboxes launched a week apart must run the same software.

    `demigod/images.py` carries the same rule as a TODO ("derive these from
    uv.lock at bake time"); this at least fails when they drift.
    """
    lock = (SRC.parent / "uv.lock").read_text(encoding="utf-8")
    for pin in GOD_PIP:
        name, version = pin.split("==")
        assert f'name = "{name}"\nversion = "{version}"' in lock, (
            f"{pin} does not match uv.lock; bump the pin in godbox/images.py or re-lock"
        )
