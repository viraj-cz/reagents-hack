import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "benchmarks" / "adaptive_circuit" / "run_baseline.py"
SPEC = importlib.util.spec_from_file_location("adaptive_baseline", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_baseline_prompt_contains_only_public_inputs():
    prompt, digest = MODULE.public_prompt()
    expected = json.loads((MODULE.CASE_DIR / "private" / "expected.json").read_text())

    assert "observations.csv" in prompt
    assert "type-1 incoherent feed-forward loop" not in prompt
    assert expected["visibility"] not in prompt
    assert len(digest) == 64
