import csv
import json
from pathlib import Path

from reagents.contracts import NativeProblem

ROOT = Path(__file__).parent.parent
CASE = ROOT / "benchmarks" / "adaptive_circuit"


def test_public_problem_is_valid_and_does_not_name_the_hidden_motif():
    question_path = CASE / "public" / "question.json"
    problem = NativeProblem.model_validate_json(question_path.read_text())

    assert problem.id == "adaptive-synthetic-circuit"
    assert "feed-forward" not in question_path.read_text().lower()


def test_withheld_measurement_is_not_present_in_agent_visible_data():
    observations = CASE / "public" / "observations.csv"
    with observations.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert not any(
        row["condition"] == "Z_KO" and row["time_min"] == "8" for row in rows
    )


def test_private_key_is_separate_from_the_public_mount_set():
    public_files = {path.name for path in (CASE / "public").iterdir()}
    expected = json.loads((CASE / "private" / "expected.json").read_text())

    assert public_files == {"question.json", "observations.csv"}
    assert expected["visibility"] == "EVALUATOR_ONLY_NEVER_MOUNT_IN_A_DEMIGOD_SANDBOX"
