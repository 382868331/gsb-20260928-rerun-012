import json

import numpy as np

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
                {key: item[key] for key in ("source", "shots", "total_shots_after_batch")}
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
