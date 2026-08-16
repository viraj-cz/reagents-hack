"""JSON-stdin/JSON-stdout entrypoint baked into disposable tool images."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


def lean_check(payload: dict[str, Any]) -> dict[str, Any]:
    source = str(payload["source"])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "Main.lean"
        path.write_text(source)
        command = ["lake", "env", "lean", str(path)] if Path("/opt/mathlib/lakefile.toml").exists() else ["lean", str(path)]
        completed = subprocess.run(
            command,
            cwd="/opt/mathlib" if Path("/opt/mathlib").exists() else None,
            capture_output=True,
            text=True,
            timeout=45,
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


OPERATIONS = {
    "lean_check": lean_check,
    "z3_solve": z3_solve,
    "sequence_stats": sequence_stats,
    "rdkit_descriptors": rdkit_descriptors,
    "proto_check": proto_check,
    "python_exec": python_exec,
}


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
