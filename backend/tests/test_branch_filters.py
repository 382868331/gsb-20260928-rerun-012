import json

import pytest

from backend import branches, server


def _checkpoint(steps, qubits=2, basis="00", seed=17):
    session = server.Session(
        {"qubits": qubits, "initial_basis": basis, "seed": seed, "steps": steps}
    )
    for _ in steps:
        session.step()
    return session.save_checkpoint("prefix", "cp-filter")


def test_conditional_filter_matches_enumerable_joint_probabilities(tmp_path):
    checkpoint = _checkpoint([{"type": "H", "target": 0}])
    specs = [
        {
            "branch_id": "filtered",
            "label": "keep q0=1",
            "steps": [
                {"type": "MEASURE", "target": 0},
                {"type": "X", "target": 1},
            ],
            "filters": [{"step": 0, "outcome": 1}],
            "budget": 4000,
        },
        {
            "branch_id": "unfiltered",
            "label": "no filter",
            "steps": [
                {"type": "MEASURE", "target": 0},
                {"type": "X", "target": 1},
            ],
            "budget": 4000,
        },
    ]
    experiment = branches.BranchExperiment.create(
        checkpoint, specs, storage_dir=tmp_path, base_seed=2024
    )
    filtered, unfiltered = experiment.snapshot()["branches"]

    # Enumerable theory: P(q0=1) = 1/2; given q0=1 the final state is |11>
    # with certainty, so the conditional distribution is a point mass.
    assert filtered["attempts"] == 4000
    assert filtered["has_evidence"] is True
    assert abs(filtered["pass_rate"] - 0.5) < 0.05
    assert filtered["counts"]["11"] == filtered["passed_shots"]
    assert filtered["frequencies"]["11"] == 1.0
    assert "4000" in filtered["sample_scope"]

    # Joint picture from the unfiltered sibling: P(11) = P(pass) * P(11|pass).
    joint = unfiltered["frequencies"]["11"]
    conditional = filtered["frequencies"]["11"]
    assert abs(joint - filtered["pass_rate"] * conditional) < 0.05
    assert unfiltered["passed_shots"] == unfiltered["attempts"]


def test_filter_without_hits_reports_no_evidence_instead_of_zero_distribution(tmp_path):
    checkpoint = _checkpoint([])  # state stays |00>; q0 measurement is always 0
    specs = [
        {
            "branch_id": "impossible",
            "label": "keep q0=1",
            "steps": [{"type": "MEASURE", "target": 0}],
            "filters": [{"step": 0, "outcome": 1}],
            "budget": 128,
        }
    ]
    experiment = branches.BranchExperiment.create(
        checkpoint, specs, storage_dir=tmp_path, base_seed=5
    )
    branch = experiment.snapshot()["branches"][0]

    assert branch["attempts"] == 128
    assert branch["passed_shots"] == 0
    assert branch["has_evidence"] is False
    assert branch["frequencies"] is None
    assert all(count == 0 for count in branch["counts"].values())
    assert "没有证据" in branch["sample_scope"]


def test_consecutive_filters_and_reload_keep_replayable_results(tmp_path):
    checkpoint = _checkpoint(
        [{"type": "H", "target": 0}, {"type": "H", "target": 1}]
    )
    spec = {
        "branch_id": "double",
        "label": "keep q0=1 then q1=1",
        "steps": [
            {"type": "MEASURE", "target": 0},
            {"type": "MEASURE", "target": 1},
        ],
        "filters": [
            {"step": 0, "outcome": 1},
            {"step": 1, "outcome": 1},
        ],
        "budget": 0,
    }
    first_dir = tmp_path / "loc-a"
    second_dir = tmp_path / "loc-b"

    continuous = branches.BranchExperiment.create(
        checkpoint, [dict(spec)], storage_dir=first_dir, base_seed=88,
        experiment_id="exp-filter",
    )
    continuous.add_shots("double", 40, source="one-shot")

    split = branches.BranchExperiment.create(
        checkpoint, [dict(spec)], storage_dir=first_dir, base_seed=88,
        experiment_id="exp-filter-split",
    )
    split.add_shots("double", 25, source="part-1")
    second_dir.mkdir()
    saved = json.loads((first_dir / "exp-filter-split.json").read_text(encoding="utf-8"))
    (second_dir / "exp-filter-split.json").write_text(
        json.dumps(saved), encoding="utf-8"
    )
    reloaded = branches.BranchExperiment.load(second_dir / "exp-filter-split.json")
    reloaded.add_shots("double", 15, source="part-2")

    continuous_branch = continuous.snapshot()["branches"][0]
    split_branch = reloaded.snapshot()["branches"][0]
    for key in ("attempts", "passed_shots", "counts", "frequencies"):
        assert split_branch[key] == continuous_branch[key]
    # Two consecutive filters on |++>: P(pass) = 1/4, survivors are all |11>.
    assert abs(split_branch["pass_rate"] - 0.25) < 0.1
    assert split_branch["counts"]["11"] == split_branch["passed_shots"]


def test_filtered_branches_are_order_independent(tmp_path):
    checkpoint = _checkpoint([{"type": "H", "target": 0}])
    specs = [
        {
            "branch_id": "keep-0",
            "label": "keep 0",
            "steps": [{"type": "MEASURE", "target": 0}],
            "filters": [{"step": 0, "outcome": 0}],
            "budget": 0,
        },
        {
            "branch_id": "keep-1",
            "label": "keep 1",
            "steps": [{"type": "MEASURE", "target": 0}],
            "filters": [{"step": 0, "outcome": 1}],
            "budget": 0,
        },
    ]
    ordered = branches.BranchExperiment.create(
        checkpoint, [dict(s) for s in specs], storage_dir=tmp_path / "a", base_seed=31
    )
    ordered.add_shots("keep-0", 30)
    ordered.add_shots("keep-1", 30)
    ordered.add_shots("keep-0", 10)

    reversed_run = branches.BranchExperiment.create(
        checkpoint, [dict(s) for s in specs], storage_dir=tmp_path / "b", base_seed=31
    )
    reversed_run.add_shots("keep-1", 30)
    reversed_run.add_shots("keep-0", 30)
    reversed_run.add_shots("keep-0", 10)

    strip = lambda snapshot: [
        {key: branch[key] for key in ("attempts", "passed_shots", "counts", "frequencies")}
        for branch in snapshot["branches"]
    ]
    assert strip(ordered.snapshot()) == strip(reversed_run.snapshot())
    keep_0, keep_1 = ordered.snapshot()["branches"]
    assert keep_0["attempts"] == 40
    assert keep_1["attempts"] == 30
    assert 0 < keep_0["passed_shots"] < 40
    assert keep_0["counts"]["00"] == keep_0["passed_shots"]
    assert keep_1["counts"]["10"] == keep_1["passed_shots"]


def test_filter_validation_rejects_bad_specs(tmp_path):
    checkpoint = _checkpoint([{"type": "H", "target": 0}])
    base = {
        "branch_id": "b",
        "label": "b",
        "steps": [{"type": "MEASURE", "target": 0}, {"type": "X", "target": 1}],
        "budget": 0,
    }
    bad_filters = [
        [{"step": 1, "outcome": 0}],
        [{"step": 5, "outcome": 0}],
        [{"step": 0, "outcome": 2}],
        [{"step": 0, "outcome": 0}, {"step": 0, "outcome": 1}],
    ]
    for filters in bad_filters:
        with pytest.raises(ValueError):
            branches.BranchExperiment.create(
                checkpoint, [{**base, "filters": filters}],
                storage_dir=tmp_path, base_seed=1,
            )
