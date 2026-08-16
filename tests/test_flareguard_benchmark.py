from __future__ import annotations

import json

from benchmarks.flareguard.benchmark import (
    PRIVATE_DIR,
    PUBLIC_DIR,
    FlareGuardVerifier,
    demigod_mount_paths,
    enumerate_valid_designs,
    evaluate_design,
    load_problem,
    score_solution,
)
from benchmarks.flareguard.run_controls import _majority
from demigod.result import DemiGodResult
from reagents.contracts import NativeSolution
from reagents.god.transformer import projection_contract
from reagents.isolation import native_terms


def _expected() -> dict:
    return json.loads((PRIVATE_DIR / "expected.json").read_text(encoding="utf-8"))


def _design() -> dict:
    return _expected()["optimal_design"]


def _complete_answer() -> dict:
    return {
        "design": _design(),
        "prediction_table": [],
        "dropout_robustness": {},
        "minimality_certificate": {},
        "validation_plan": [],
    }


def test_public_case_is_hydrated_for_god_but_not_mounted_for_demigods():
    problem = load_problem()
    assert problem.id == "flareguard-living-diagnostic"
    assert set(problem.inputs) == {
        "trajectory_table",
        "part_library",
        "compatibility_rules",
    }
    assert len(problem.inputs["trajectory_table"]) == 63
    assert demigod_mount_paths() == []
    assert len(projection_contract(problem).source_ids) == 91
    assert {"healthy_01", "chronic_flare"} <= native_terms(problem)


def test_private_evaluator_material_is_outside_public_boundary():
    assert {path.name for path in PUBLIC_DIR.iterdir()} == {
        "question.json",
        "trajectories.csv",
        "parts.json",
        "compatibility.json",
    }
    dumped_inputs = json.dumps(load_problem().inputs)
    assert "held_flare_01" not in dumped_inputs
    assert "optimal_design" not in dumped_inputs
    assert _expected()["visibility"].startswith("EVALUATOR_ONLY")


def test_oracle_has_one_minimum_on_public_and_withheld_trajectories():
    expected = _expected()
    for private in (False, True):
        valid = enumerate_valid_designs(include_private=private)
        assert len(valid) == expected["expected_valid_optima"]
        assert valid[0]["design"] == expected["optimal_design"]
        assert valid[0]["checks"]["burden"] == expected["optimal_burden"]


def test_each_nontrivial_requirement_changes_the_answer():
    design = _design()

    no_timer = {**design, "persistence_part_id": "T_0H"}
    assert not evaluate_design(no_timer)["checks"]["all_native_trajectories"]

    one_exclusion = {**design, "exclusion_sensor_ids": ["S_AHL"]}
    assert not evaluate_design(one_exclusion)["checks"]["single_sensor_dropout"]

    unanimity = {**design, "positive_quorum": 3}
    assert not evaluate_design(unanimity)["checks"]["single_sensor_dropout"]

    slow_reset = {**design, "reset_part_id": "D_SLOW"}
    assert not evaluate_design(slow_reset)["checks"]["all_native_trajectories"]


def test_native_verifier_uses_public_data_and_private_grader_scores_ten():
    solution = NativeSolution(
        problem_id="flareguard-living-diagnostic",
        answer="candidate",
        confidence=0.8,
        structured_answer=_complete_answer(),
    )
    public = FlareGuardVerifier().verify(load_problem(), solution)
    private = score_solution(solution, private=True)
    assert public.passed and public.score == 1.0
    assert private["passed"] and private["score_10"] == 10.0


def test_native_control_majority_vote_uses_only_submitted_designs():
    answer = _complete_answer()
    winner = DemiGodResult(
        claim="candidate",
        confidence=0.8,
        payload=answer,
        method="native control",
    )
    alternate_answer = _complete_answer()
    alternate_answer["design"] = {
        **alternate_answer["design"],
        "persistence_part_id": "T_3H",
    }
    alternate = winner.model_copy(update={"payload": alternate_answer})
    selected, score = _majority([winner, winner.model_copy(), alternate])
    assert selected == 0
    assert score["score_10"] == 10.0
