"""The tool payload. One plain function per operation, and two ways to reach it.

    docker run ... python3 /opt/reagents/tool_runtime.py z3_solve   < payload
    from reagents.tools.tool_runtime import run_operation            # in-process

Both callers run the SAME functions, which is the point: a tool that works
locally under Docker and a tool that works inside a Modal broker executor are
not two implementations that can drift, they are one module invoked two ways.

WHY IT LIVES IN `reagents.tools` AND NOT IN `tooling/`. It used to sit at
`tooling/runtime/tool_runtime.py`, outside every package, because its only
consumer was a `COPY` line in a Dockerfile. That made it unimportable from the
broker's executor image, which ships packages (`add_local_python_source`) and
has no Docker daemon to shell out to -- so every CONTAINER-provider tool failed
there. Moving it into the package makes it importable; the Dockerfiles still
`COPY src/reagents/tools/tool_runtime.py` and still work, because this module
imports NOTHING from `reagents` and must keep it that way. The disposable images
do not have the package installed, only this one file.

The scientific dependencies are imported inside the functions that need them for
the same reason: `rdkit_descriptors` must not make `z3_solve` unimportable in an
image that has Z3 and no RDKit.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any


MATHLIB = Path("/opt/mathlib")
"""Where both the local reasoning image and the broker's `lean` tier put it."""

LEAN_TIMEOUT_S = 300
"""`import Mathlib` alone loads thousands of .olean files and takes minutes on a
cold page cache. The previous 45s could not finish a one-line theorem, and
reported it as a timeout rather than as "you did not give it enough time"."""


def lean_check(payload: dict[str, Any]) -> dict[str, Any]:
    source = str(payload["source"])
    # BOTH flavours. mathlib4 at v4.30.0 ships `lakefile.lean`; checking only
    # for `lakefile.toml` silently fell through to a bare `lean`, which has no
    # LEAN_PATH and fails every proof with "unknown module prefix 'Mathlib'" --
    # a mathlib-less checker that looks like a working one. Found live.
    in_project = any(
        (MATHLIB / name).exists() for name in ("lakefile.toml", "lakefile.lean")
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "Main.lean"
        path.write_text(source)
        command = ["lake", "env", "lean", str(path)] if in_project else ["lean", str(path)]
        completed = subprocess.run(
            command,
            cwd=str(MATHLIB) if MATHLIB.exists() else None,
            capture_output=True,
            text=True,
            timeout=LEAN_TIMEOUT_S,
            check=False,
        )
    return {
        "verified": completed.returncode == 0,
        "stdout": completed.stdout[-20000:],
        "stderr": completed.stderr[-20000:],
    }


def z3_solve(payload: dict[str, Any]) -> dict[str, Any]:
    completed = subprocess.run(
        ["z3", "-in", "-smt2"],
        input=str(payload["smt2"]),
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    return {
        "ok": completed.returncode == 0,
        "stdout": completed.stdout[-100000:],
        "stderr": completed.stderr[-20000:],
    }


def sequence_stats(payload: dict[str, Any]) -> dict[str, Any]:
    sequence = "".join(str(payload["sequence"]).split()).upper()
    counts = Counter(sequence)
    dna = set(sequence) <= set("ACGTN")
    gc = None
    canonical = counts["A"] + counts["C"] + counts["G"] + counts["T"]
    if dna and canonical:
        gc = (counts["G"] + counts["C"]) / canonical
    return {
        "length": len(sequence),
        "alphabet": sorted(counts),
        "composition": dict(sorted(counts.items())),
        "is_dna": dna,
        "gc_fraction": gc,
    }


def rdkit_descriptors(payload: dict[str, Any]) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski

    molecule = Chem.MolFromSmiles(str(payload["smiles"]))
    if molecule is None:
        raise ValueError("invalid SMILES")
    return {
        "canonical_smiles": Chem.MolToSmiles(molecule),
        "molecular_weight": Descriptors.MolWt(molecule),
        "logp": Descriptors.MolLogP(molecule),
        "h_bond_donors": Lipinski.NumHDonors(molecule),
        "h_bond_acceptors": Lipinski.NumHAcceptors(molecule),
        "rotatable_bonds": Lipinski.NumRotatableBonds(molecule),
    }


def proto_check(_: dict[str, Any]) -> dict[str, Any]:
    import proto_language

    return {
        "available": True,
        "module": proto_language.__name__,
        "version": getattr(proto_language, "__version__", "unknown"),
        "note": "Design execution remains approval-gated; this operation only validates the runtime.",
    }


def python_exec(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute code only inside the already isolated, resource-limited container."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "program.py"
        path.write_text(str(payload["source"]))
        completed = subprocess.run(
            [sys.executable, "-I", str(path)],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=50,
            check=False,
        )
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-100000:],
        "stderr": completed.stderr[-20000:],
    }


OPERATIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "lean_check": lean_check,
    "z3_solve": z3_solve,
    "sequence_stats": sequence_stats,
    "rdkit_descriptors": rdkit_descriptors,
    "proto_check": proto_check,
    "python_exec": python_exec,
}


class UnknownOperationError(KeyError):
    """An operation name that no image implements."""


def run_operation(name: str, payload: dict[str, Any]) -> Any:
    """Execute one operation and return a JSON-ROUND-TRIPPED result.

    The round trip is not decoration. The subprocess path below serializes with
    `json.dumps(..., default=str)` and the caller parses that, so a NumPy float
    or a Path arrives as a string. An in-process caller that skipped this would
    get a differently-typed result for the same tool depending on where it ran,
    and the difference would surface as a serialization failure much later --
    inside Modal's return path, or in the broker's JSON response.
    """
    try:
        operation = OPERATIONS[name]
    except KeyError as exc:
        raise UnknownOperationError(
            f"unknown operation {name!r}; registered: {sorted(OPERATIONS)}"
        ) from exc
    return json.loads(json.dumps(operation(payload), default=str))


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in OPERATIONS:
        raise SystemExit("expected one registered operation name")
    payload = json.load(sys.stdin)
    try:
        result = OPERATIONS[sys.argv[1]](payload)
    except Exception as exc:  # noqa: BLE001 - serialize failure across container boundary
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        raise SystemExit(1) from exc
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
