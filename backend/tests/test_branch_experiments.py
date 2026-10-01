import json

import numpy as np
import pytest

from backend import branches, quantum, server


PREFIX_STEPS = [
    {"type": "H", "target": 0},
    {"type": "MEASURE", "target": 0},
]


def _checkpoint():
    session = server.Session(
        {"qubits": 2, "initial_basis": "00", "seed": 17, "steps": PREFIX_STEPS}
    )
    session.step()
    session.step()
    return session.save_checkpoint("after-prefix-measurement", "cp-branch")


def _specs():
    return [
        {
            "branch_id": "flip",
            "label": "X on q1",
            "steps": [{"type": "X", "target": 1}],
            "budget": 64,
        },
        {
            "branch_id": "idle",
            "label": "no operation",
            "steps": [],
            "budget": 64,
        },
    ]


def test_conditional_branch_distributions_share_prefix_but_do_not_pool(tmp_path):
    checkpoint = _checkpoint()
    experiment = branches.BranchExperiment.create(
        checkpoint, _specs(), storage_dir=tmp_path, base_seed=1234
    )

    prefix_outcome = checkpoint.measurements[-1]["outcome"]
    flip = experiment.snapshot()["branches"][0]
    idle = experiment.snapshot()["branches"][1]

    assert flip["completed_shots"] == 64
    assert idle["completed_shots"] == 64
    assert flip["counts"][f"1{prefix_outcome}"] == 64
    assert idle["counts"][f"{prefix_outcome}0"] == 64
    assert experiment.snapshot()["pooled_across_conditions"] is False


def test_branch_execution_order_does_not_change_replayable_results(tmp_path):
    checkpoint = _checkpoint()
    specs = _specs()

    specs[0]["budget"] = 0
    specs[1]["budget"] = 0
    ordered = branches.BranchExperiment.create(
        checkpoint, specs, storage_dir=tmp_path / "ordered", base_seed=99
    )
    ordered.add_shots("flip", 20, source="first")
    ordered.add_shots("idle", 20, source="second")
    ordered.add_shots("flip", 10, source="third")

    reversed_run = branches.BranchExperiment.create(
        checkpoint,
        [{**spec, "budget": 0} for spec in specs],
        storage_dir=tmp_path / "reversed",
        base_seed=99,
        experiment_id="reversed",
    )
    reversed_run.add_shots("idle", 20, source="second")
    reversed_run.add_shots("flip", 20, source="first")
    reversed_run.add_shots("flip", 10, source="third")

    comparable = lambda branches: [
        {
            key: value
            for key, value in branch.items()
            if key != "batches"
        }
        | {
            "batches": [
                {
                    key: item[key]
                    for key in (
                        "source",
                        "attempts",
                        "accepted_shots",
                        "rejected_shots",
                        "total_attempts_after_batch",
                        "total_shots_after_batch",
                    )
                }
                for item in branch["batches"]
            ]
        }
        for branch in branches
    ]
    assert comparable(ordered.snapshot()["branches"]) == comparable(
        reversed_run.snapshot()["branches"]
    )


def test_save_location_and_reload_do_not_change_results(tmp_path):
    checkpoint = _checkpoint()
    first_dir = tmp_path / "location-a"
    second_dir = tmp_path / "location-b"
    specs = [{**spec, "budget": 0} for spec in _specs()]

    first = branches.BranchExperiment.create(
        checkpoint, specs, storage_dir=first_dir, base_seed=77, experiment_id="exp-location"
    )
    first.add_shots("flip", 31, source="initial")
    saved = json.loads((first_dir / "exp-location.json").read_text(encoding="utf-8"))
    (second_dir / "exp-location.json").parent.mkdir()
    (second_dir / "exp-location.json").write_text(json.dumps(saved), encoding="utf-8")

    reloaded = branches.BranchExperiment.load(second_dir / "exp-location.json")
    before = reloaded.snapshot()
    reloaded.add_shots("idle", 15, source="continued")
    after = branches.BranchExperiment.load(second_dir / "exp-location.json").snapshot()

    assert before["branches"][0] == reloaded.snapshot()["branches"][0]
    assert after["branches"][1]["completed_shots"] == 15
    assert after["storage_path"] == str(second_dir / "exp-location.json")


def test_common_prefix_and_independent_condition_metadata_are_persisted(tmp_path):
    checkpoint = _checkpoint()
    experiment = branches.BranchExperiment.create(
        checkpoint, _specs(), storage_dir=tmp_path, base_seed=5
    )
    raw = json.loads(experiment.path.read_text(encoding="utf-8"))

    assert raw["prefix"]["prefix_steps"] == PREFIX_STEPS
    assert raw["prefix"]["prefix_measurements"] == checkpoint.measurements
    assert raw["branch_order"] == ["flip", "idle"]
    assert [batch["source"] for batch in raw["branches"]["flip"]["batches"]] == ["initial-budget"]
    assert raw["branches"]["flip"]["rng_state"] != raw["branches"]["idle"]["rng_state"]


def _fresh_checkpoint():
    session = server.Session(
        {"qubits": 2, "initial_basis": "00", "seed": 23, "steps": []}
    )
    return session.save_checkpoint("fresh", "cp-fresh")


def test_filtered_samples_match_exact_joint_and_conditional_distribution(tmp_path):
    checkpoint = _fresh_checkpoint()
    steps = [
        {"type": "H", "target": 0},
        {"type": "MEASURE", "target": 0},
        {"type": "H", "target": 1},
    ]
    experiment = branches.BranchExperiment.create(
        checkpoint,
        [
            {
                "branch_id": "q0-zero",
                "label": "q0 is 0",
                "steps": steps,
                "filters": [{"target": 0, "outcome": 0}],
                "budget": 4000,
            }
        ],
        storage_dir=tmp_path,
        base_seed=321,
    )
    branch = experiment.snapshot()["branches"][0]

    assert branch["total_attempts"] >= branch["passed_shots"]
    assert branch["passed_shots"] == 4000
    assert branch["rejected_shots"] + branch["passed_shots"] == branch["total_attempts"]
    assert branch["counts"]["00"] + branch["counts"]["01"] == 4000
    assert branch["counts"]["10"] == branch["counts"]["11"] == 0
    assert branch["frequencies"]["10"] == 0.0
    assert 0.45 < branch["frequencies"]["00"] < 0.55
    assert 0.45 < branch["frequencies"]["01"] < 0.55
    assert branch["exact"]["accepted_probability"] == pytest.approx(0.5)
    assert branch["exact"]["conditional_probabilities"] == {
        "00": pytest.approx(0.5),
        "01": pytest.approx(0.5),
        "10": pytest.approx(0.0),
        "11": pytest.approx(0.0),
    }
    assert branch["exact"]["joint_probabilities"]["00"] == pytest.approx(0.25)
    assert branch["exact"]["joint_probabilities"]["10"] == pytest.approx(0.25)


def test_zero_probability_filter_reports_no_evidence_instead_of_zero_probabilities(tmp_path):
    checkpoint = _fresh_checkpoint()
    experiment = branches.BranchExperiment.create(
        checkpoint,
        [
            {
                "branch_id": "impossible",
                "label": "X then require 0",
                "steps": [
                    {"type": "X", "target": 0},
                    {"type": "MEASURE", "target": 0},
                ],
                "filters": [{"target": 0, "outcome": 0}],
                "budget": 10,
                "attempt_cap": 20,
            }
        ],
        storage_dir=tmp_path,
        base_seed=456,
    )
    branch = experiment.snapshot()["branches"][0]

    assert branch["total_attempts"] == 20
    assert branch["passed_shots"] == 0
    assert branch["rejected_shots"] == 20
    assert branch["cap_reached"] is True
    assert branch["evidence_status"] == "no-evidence"
    assert branch["frequencies"] is None
    assert branch["exact"]["has_evidence"] is False
    assert branch["exact"]["accepted_probability"] == 0.0
    assert branch["exact"]["conditional_probabilities"] is None


def test_two_successive_filters_scope_and_replay_are_stable(tmp_path):
    checkpoint = _fresh_checkpoint()
    steps = [
        {"type": "H", "target": 0},
        {"type": "MEASURE", "target": 0},
        {"type": "H", "target": 1},
        {"type": "MEASURE", "target": 1},
    ]
    filters = [{"target": 0, "outcome": 0}, {"target": 1, "outcome": 1}]
    specs = [
        {
            "branch_id": "zero-one",
            "label": "q0=0 then q1=1",
            "steps": steps,
            "filters": filters,
            "budget": 0,
        }
    ]
    first = branches.BranchExperiment.create(
        checkpoint, specs, storage_dir=tmp_path / "a", base_seed=654
    )
    first.add_shots("zero-one", 100, source="first")
    first.add_shots("zero-one", 50, source="second")

    second = branches.BranchExperiment.create(
        checkpoint,
        [{**specs[0], "budget": 0}],
        storage_dir=tmp_path / "b",
        base_seed=654,
    )
    second.add_shots("zero-one", 70, source="first")
    second.add_shots("zero-one", 80, source="second")
    first_view = first.snapshot()["branches"][0]
    second_view = second.snapshot()["branches"][0]

    assert first_view["filters"] == filters
    assert first_view["sample_scope"] == "attempts passing every measurement filter"
    assert first_view["passed_shots"] == 150
    assert first_view["counts"]["01"] == 150
    assert sum(first_view["stage_rejected"]) == first_view["rejected_shots"]
    assert first_view["exact"]["accepted_probability"] == pytest.approx(0.25)
    assert first_view["counts"] == second_view["counts"]
    assert first_view["total_attempts"] == second_view["total_attempts"]
    assert first_view["frequencies"] == second_view["frequencies"]


def test_two_successive_filters_remain_replayable_after_save_and_reload(tmp_path):
    checkpoint = _fresh_checkpoint()
    steps = [
        {"type": "H", "target": 0},
        {"type": "MEASURE", "target": 0},
        {"type": "H", "target": 1},
        {"type": "MEASURE", "target": 1},
    ]
    filters = [{"target": 0, "outcome": 1}, {"target": 1, "outcome": 0}]
    spec = {
        "branch_id": "one-zero",
        "label": "q0=1 then q1=0",
        "steps": steps,
        "filters": filters,
        "budget": 0,
    }
    directory_a = tmp_path / "saved-a"
    first = branches.BranchExperiment.create(
        checkpoint, [spec], storage_dir=directory_a, base_seed=888, experiment_id="relocate"
    )
    first.add_shots("one-zero", 40, source="before-move")
    raw = json.loads((directory_a / "relocate.json").read_text(encoding="utf-8"))
    directory_b = tmp_path / "saved-b"
    directory_b.mkdir()
    (directory_b / "relocate.json").write_text(json.dumps(raw), encoding="utf-8")

    reloaded = branches.BranchExperiment.load(directory_b / "relocate.json")
    first_batch = reloaded.snapshot()["branches"][0]["batches"][0]
    reloaded.add_shots("one-zero", 35, source="after-move")

    comparison = branches.BranchExperiment.create(
        checkpoint, [spec], storage_dir=tmp_path / "comparison", base_seed=888
    )
    comparison.add_shots("one-zero", 40, source="before-move")
    comparison.add_shots("one-zero", 35, source="after-move")

    moved = reloaded.snapshot()["branches"][0]
    expected = comparison.snapshot()["branches"][0]
    assert moved["batches"][0]["batch_id"] == first_batch["batch_id"]
    assert moved["passed_shots"] == expected["passed_shots"] == 75
    assert moved["total_attempts"] == expected["total_attempts"]
    assert moved["counts"] == expected["counts"] == {"00": 0, "01": 0, "10": 75, "11": 0}
