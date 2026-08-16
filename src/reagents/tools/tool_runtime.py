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
import os
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
        command = (
            ["lake", "env", "lean", str(path)] if in_project else ["lean", str(path)]
        )
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
        "note": (
            "Design execution remains approval-gated; this operation only "
            "validates the runtime."
        ),
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


NORMAN_TRAINING_PATH_ENV = "REAGENTS_NORMAN_TRAINING_PATH"
NORMAN_TRAINING_DEFAULT = "/opt/reagents/norman_v2_training.json"
PERTURBSEQ_DESIGN_PATH_ENV = "REAGENTS_PERTURBSEQ_DESIGN_PATH"
PERTURBSEQ_DESIGN_DEFAULT = "/opt/reagents/perturbseq_design_training.json"


def norman_training_python(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute agent code with the frozen public training bundle preloaded.

    The executor image contains only this runtime, scientific dependencies, and
    the public training JSON. It contains neither repository source nor the
    private expected fixture. ``DATA`` is injected before the submitted program.
    """

    data_path = Path(os.environ.get(NORMAN_TRAINING_PATH_ENV, NORMAN_TRAINING_DEFAULT))
    if not data_path.is_file():
        raise FileNotFoundError(f"frozen training bundle missing at {data_path}")
    source = str(payload["source"])
    prelude = (
        "import json as _json\n"
        f"with open({str(data_path)!r}, encoding='utf-8') as _handle:\n"
        "    DATA = _json.load(_handle)\n"
        "assert DATA.get('version') == 2\n"
        "assert 'targets' not in DATA and 'expected' not in DATA\n"
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "program.py"
        path.write_text(prelude + "\n" + source)
        completed = subprocess.run(
            [sys.executable, "-I", str(path)],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=150,
            check=False,
        )
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-250000:],
        "stderr": completed.stderr[-30000:],
        "training_bundle_version": 2,
        "heldout_outcomes_present": False,
    }


def perturbseq_design_python(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute agent code against public batch-design evidence only."""

    data_path = Path(
        os.environ.get(PERTURBSEQ_DESIGN_PATH_ENV, PERTURBSEQ_DESIGN_DEFAULT)
    )
    if not data_path.is_file():
        raise FileNotFoundError(f"frozen design bundle missing at {data_path}")
    source = str(payload["source"])
    prelude = (
        "import json as _json\n"
        f"with open({str(data_path)!r}, encoding='utf-8') as _handle:\n"
        "    DATA = _json.load(_handle)\n"
        "assert DATA.get('version') == 1\n"
        "assert 'candidates' not in DATA and 'oracle_selection' not in DATA\n"
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "program.py"
        path.write_text(prelude + "\n" + source)
        completed = subprocess.run(
            [sys.executable, "-I", str(path)],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=210,
            check=False,
        )
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-250000:],
        "stderr": completed.stderr[-30000:],
        "training_bundle_version": 1,
        "candidate_outcomes_present": False,
    }


ESM_MODEL_ENV = "REAGENTS_ESM_MODEL"
ESM_DEFAULT_MODEL = "esm2_t12_35M_UR50D"
"""Smallest useful ESM-2 checkpoint (~150MB, 12 layers).

Chosen as the default because it runs on CPU in seconds, so the tier costs
nothing when idle and needs no GPU to be useful. `esm2_t33_650M_UR50D` is the
quality step up (~2.5GB) and wants a GPU -- set REAGENTS_ESM_MODEL to switch,
and give that tier `gpu=` so the accelerator is held by this Function alone
rather than while a demigod is thinking.
"""

_ESM_CACHE: dict[str, Any] = {}


def _load_esm(model_name: str) -> tuple[Any, Any, Any]:
    """Load and memoise an ESM-2 checkpoint for this worker process.

    Memoised because a warm Modal container serves many calls: re-loading
    hundreds of MB of weights per request would dominate the runtime of an
    otherwise millisecond-scale tool.
    """
    if model_name in _ESM_CACHE:
        return _ESM_CACHE[model_name]
    import esm as esm_pkg
    import torch

    if not hasattr(esm_pkg.pretrained, model_name):
        available = [n for n in dir(esm_pkg.pretrained) if n.startswith("esm2_")]
        raise ValueError(f"unknown ESM model {model_name!r}; available: {available}")
    model, alphabet = getattr(esm_pkg.pretrained, model_name)()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    entry = (model, alphabet, alphabet.get_batch_converter())
    _ESM_CACHE[model_name] = entry
    return entry


def _esm_forward(sequence: str, *, want_contacts: bool) -> dict[str, Any]:
    import torch

    name = os.environ.get(ESM_MODEL_ENV, ESM_DEFAULT_MODEL)
    model, _alphabet, batch_converter = _load_esm(name)
    _, _, tokens = batch_converter([("query", sequence)])
    if torch.cuda.is_available():
        tokens = tokens.cuda()
    layer = model.num_layers
    with torch.no_grad():
        out = model(tokens, repr_layers=[layer], return_contacts=want_contacts)
    # Strip BOS/EOS so index i lines up with residue i of the input.
    reps = out["representations"][layer][0, 1 : len(sequence) + 1]
    result: dict[str, Any] = {
        "model": name,
        "length": len(sequence),
        "embedding_dim": int(reps.shape[-1]),
        "mean_embedding": [round(float(v), 6) for v in reps.mean(0).tolist()],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if want_contacts:
        contacts = out["contacts"][0, : len(sequence), : len(sequence)]
        result["contacts"] = [
            [round(float(v), 4) for v in row] for row in contacts.tolist()
        ]
    return result


def esm_embed(payload: dict[str, Any]) -> dict[str, Any]:
    """Mean-pooled ESM-2 embedding for one protein sequence."""
    sequence = str(payload["sequence"]).strip().upper()
    if not sequence:
        raise ValueError("sequence is empty")
    return _esm_forward(sequence, want_contacts=False)


def esm_contacts(payload: dict[str, Any]) -> dict[str, Any]:
    """ESM-2 predicted residue-residue contact map.

    Capped at 400 residues: the map is O(n^2) and a 1000-residue protein would
    return a million floats through a JSON tool result.
    """
    sequence = str(payload["sequence"]).strip().upper()
    if not sequence:
        raise ValueError("sequence is empty")
    if len(sequence) > 400:
        raise ValueError(
            f"sequence is {len(sequence)} residues; contact maps are capped at "
            f"400 because the result is quadratic in length"
        )
    return _esm_forward(sequence, want_contacts=True)


OPERATIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "lean_check": lean_check,
    "z3_solve": z3_solve,
    "sequence_stats": sequence_stats,
    "rdkit_descriptors": rdkit_descriptors,
    "proto_check": proto_check,
    "python_exec": python_exec,
    "norman_training_python": norman_training_python,
    "perturbseq_design_python": perturbseq_design_python,
    "esm_embed": esm_embed,
    "esm_contacts": esm_contacts,
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
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        raise SystemExit(1) from exc
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
