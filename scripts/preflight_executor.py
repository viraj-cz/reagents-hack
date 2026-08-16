"""Live check that ONE broker executor tier can actually run its tools.

    uv run python scripts/preflight_executor.py                  # reasoning
    uv run python scripts/preflight_executor.py --tier biology
    uv run python scripts/preflight_executor.py --tier lean      # expensive

Sibling of `scripts/preflight_toolbox.py`, and the same argument applies: the
offline suite proves that a CONTAINER tool takes the in-process path when told
to, but it cannot prove that the tier's IMAGE contains the packages that path
then imports. Those are two different failures and only one of them is visible
from a laptop.

WHAT IT ACTUALLY EXERCISES. `broker.service.execute_tool` -- the exact function
body a deployed tier runs -- on the exact image `ExecutorClass.image()` builds,
including `REAGENTS_ENABLE_CONTAINERS` and `REAGENTS_TOOL_RUNTIME` coming from
the image env rather than from a shell. If a tool answers here, the only thing
between it and a demigod is the router, which `preflight_toolbox.py` covers.

COST. ONE tier per run, deliberately. `broker.service` holds five buildable
tiers; bringing up that App to check one of them would build all five. The
reasoning tier is a few minutes of build the first time and seconds after that
(Modal content-hashes the layers, and the heavy ones sit below the source
layer). `--tier lean` downloads a compiled mathlib and is the one that costs
real time -- it is opt-in for that reason.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import modal

from broker.service import ALL_EXECUTOR_CLASSES, ExecutorClass, execute_tool

APP_NAME = "toolbox-executor-preflight"

PROBES: dict[str, list[tuple[str, dict[str, Any], Any]]] = {
    # tier -> [(tool_id, arguments, predicate)]
    "reasoning": [
        (
            "formal.z3_solve",
            {
                "smt2": (
                    "(declare-const x Int)(assert (> x 41))"
                    "(assert (< x 43))(check-sat)(get-model)"
                )
            },
            lambda r: r["ok"] and "sat" in r["stdout"] and "42" in r["stdout"],
        ),
        (
            "reasoning.python",
            {
                "source": (
                    "import numpy, scipy, sympy, networkx, pint, cvxpy, control\n"
                    "print(sympy.simplify('sin(x)**2 + cos(x)**2'))"
                )
            },
            lambda r: r["ok"] and r["stdout"].strip() == "1",
        ),
    ],
    "biology": [
        (
            "chemistry.rdkit_descriptors",
            {"smiles": "CC(=O)Oc1ccccc1C(=O)O"},
            lambda r: abs(r["molecular_weight"] - 180.16) < 0.1,
        ),
        (
            "biology.sequence_stats",
            {"sequence": "ACGTACGTGG"},
            lambda r: r["length"] == 10 and abs(r["gc_fraction"] - 0.6) < 1e-9,
        ),
        (
            "biology.python",
            {
                "source": (
                    "import Bio, rdkit, cobra, sklearn, statsmodels, pandas\n"
                    "from Bio.Seq import Seq\n"
                    "print(Seq('ATGC').reverse_complement())"
                )
            },
            lambda r: r["ok"] and r["stdout"].strip() == "GCAT",
        ),
    ],
    "engineering": [
        (
            "engineering.python",
            {
                "source": (
                    "import cantera as ct, pint\n"
                    "g = ct.Solution('gri30.yaml')\n"
                    "print(g.n_species > 0)"
                )
            },
            lambda r: r["ok"] and r["stdout"].strip() == "True",
        ),
    ],
    "lean": [
        (
            "formal.lean_check",
            {
                "source": (
                    "import Mathlib\ntheorem probe (n : Nat) : n + 0 = n := by simp\n"
                )
            },
            lambda r: r["verified"],
        ),
    ],
    "design": [
        ("design.proto_check", {}, lambda r: r["available"]),
    ],
}

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' | {detail}' if detail else ''}")
    if not ok:
        failures.append(label)


def tier(name: str) -> ExecutorClass:
    for klass in ALL_EXECUTOR_CLASSES:
        if klass.name == name:
            return klass
    raise SystemExit(
        f"unknown tier {name!r}; known: {sorted(k.name for k in ALL_EXECUTOR_CLASSES)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tier",
        default="reasoning",
        help="executor class to build and probe (one per run, on purpose)",
    )
    args = parser.parse_args()

    klass = tier(args.tier)
    probes = PROBES.get(klass.name, [])
    if not probes:
        raise SystemExit(f"no probes defined for tier {klass.name!r}")

    print(f"[1/3] building the {klass.name!r} executor image")
    print(f"      extras={list(klass.extras)}")
    if klass.apt or klass.setup_commands:
        print(f"      apt={list(klass.apt)} setup={len(klass.setup_commands)} step(s)")

    app = modal.App(APP_NAME)
    # The REAL executor body, on the REAL tier image. Not a re-implementation:
    # `execute_tool` is what `broker.service._make_executor` wraps.
    remote = app.function(
        name=f"probe_{klass.name}",
        image=klass.image(),
        cpu=klass.cpu,
        memory=klass.memory_mb,
        gpu=klass.gpu,
        timeout=klass.timeout_s,
        max_containers=1,
        serialized=True,
    )(execute_tool)

    with modal.enable_output(), app.run():
        print("[2/3] the image env decides the execution path, not this shell")
        try:
            env = (
                remote.remote(
                    "reasoning.python",
                    {
                        "source": (
                            "import os;"
                            "print(os.environ.get('REAGENTS_TOOL_RUNTIME'),"
                            "os.environ.get('REAGENTS_ENABLE_CONTAINERS'))"
                        )
                    },
                )
                if klass.name == "reasoning"
                else None
            )
        except Exception as exc:
            env = None
            print(f"      (env probe skipped: {type(exc).__name__}: {exc})")
        if env is not None:
            check(
                "executor image exports inprocess + containers-enabled",
                env["ok"] and env["stdout"].split() == ["inprocess", "1"],
                env["stdout"].strip() or env["stderr"][-200:],
            )

        print(f"[3/3] running {len(probes)} tool(s) through the tier")
        for tool_id, arguments, predicate in probes:
            try:
                result = remote.remote(tool_id, arguments)
            except Exception as exc:
                check(tool_id, False, f"{type(exc).__name__}: {exc}")
                continue
            try:
                ok = bool(predicate(result))
            except Exception as exc:
                ok = False
                print(f"      (predicate raised {type(exc).__name__}: {exc})")
            check(tool_id, ok, str(result)[:160])

    if failures:
        print(f"\n{len(failures)} FAILURE(S): {failures}", file=sys.stderr)
        return 1
    print(f"\nthe {klass.name!r} tier executes its tools in Modal, no Docker involved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
