import shutil

import numpy as np

from backend import quantum, server


PREFIX_STEPS = [
    {"type": "H", "target": 0},
    {"type": "MEASURE", "target": 0},
]


def _checkpoint() -> server.Checkpoint:
    session = server.Session(
        {"qubits": 2, "initial_basis": "00", "seed": 57, "steps": PREFIX_STEPS}
    )
    session.step()
    session.step()
    return session.save_checkpoint("after-q0-measure", "prefix-57")


def _run(checkpoint, branches, *, shots=16, budget=32):
    return server.BranchExperiment.from_checkpoint(
        checkpoint,
        branches,
        name="conditional continuation",
        shots=shots,
        budget=budget,
        root_seed=9001,
    )


def _order_run(checkpoint, branches, *, shots=37):
    return _run(checkpoint, branches, shots=shots, budget=64)


def test_branches_share_prefix_but_keep_statistics_separate():
    checkpoint = _checkpoint()
    prefix_outcome = checkpoint.measurements[-1]["outcome"]
    run = _run(
        checkpoint,
        [
            ("flip", "条件翻转", [{"type": "X", "target": 1}]),
            ("idle", "不操作", []),
        ],
    )

    assert run.snapshot()["checkpoint"]["prefix_next_index"] == 2
    assert run.snapshot()["checkpoint"]["measurements"] == checkpoint.measurements
    flip, idle = run.snapshot()["branches"]
    assert flip["shots"] == idle["shots"] == 16
    assert flip["sample_source"].endswith(":flip")
    assert idle["sample_source"].endswith(":idle")
    assert flip["counts"][f"{prefix_outcome}1"] == 16
    assert idle["counts"][f"{prefix_outcome}0"] == 16
    assert sum(flip["counts"].values()) == 16
    assert sum(idle["counts"].values()) == 16


def test_branch_execution_order_does_not_change_replay():
    checkpoint = _checkpoint()
    branches = [
        ("hadamard", "加 H", [{"type": "H", "target": 1}]),
        ("rotation", "加旋转", [{"type": "RY", "target": 1, "angle": 1.23}]),
    ]

    first = _order_run(checkpoint, branches)
    reversed_run = _order_run(checkpoint, list(reversed(branches)))

    def comparable(branches):
        return {
            item["branch_id"]: {
                key: value for key, value in item.items() if key != "sample_source"
            }
            for item in branches
        }

    first_by_id = comparable(first.snapshot()["branches"])
    reversed_by_id = comparable(reversed_run.snapshot()["branches"])
    assert first_by_id == reversed_by_id


def test_conditional_h_gate_has_expected_half_half_distribution():
    checkpoint = _checkpoint()
    run = _run(
        checkpoint,
        [("conditional-h", "q1 加 H", [{"type": "H", "target": 1}]),("idle", "无操作", [])],
        shots=4_000,
        budget=4_000,
    )
    h_branch, idle_branch = run.snapshot()["branches"]
    prefix_outcome = checkpoint.measurements[-1]["outcome"]
    assert h_branch["counts"][f"{prefix_outcome}0"] + h_branch["counts"][f"{prefix_outcome}1"] == 4_000
    assert abs(h_branch["counts"][f"{prefix_outcome}0"] - 2_000) < 200
    assert abs(h_branch["counts"][f"{prefix_outcome}1"] - 2_000) < 200
    assert idle_branch["counts"][f"{prefix_outcome}0"] == 4_000


def test_branch_can_stop_at_budget_and_resume_with_more_shots(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "BRANCH_RUNS_DIR", tmp_path)
    server.BRANCH_RUNS.clear()
    checkpoint = _checkpoint()
    run = _run(
        checkpoint,
        [("a", "A", [{"type": "H", "target": 1}])],
        shots=10,
        budget=10,
    )
    path = server.save_branch_experiment(run)
    before = path.read_text(encoding="utf-8")

    try:
        run.add_shots(1)
    except ValueError as exc:
        assert "budget" in str(exc)
    else:
        raise AssertionError("branch must stop at its saved budget")

    resumed = server.load_branch_experiment(run.run_id)
    resumed.add_shots(15, budget=25)
    server.save_branch_experiment(resumed)
    result = resumed.snapshot()
    assert result["budget"] == 25
    assert result["branches"][0]["shots"] == 25
    assert sum(result["branches"][0]["counts"].values()) == 25
    assert path.read_text(encoding="utf-8") != before


def test_saved_result_is_independent_of_storage_directory(monkeypatch, tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    monkeypatch.setattr(server, "BRANCH_RUNS_DIR", first_dir)
    server.BRANCH_RUNS.clear()

    run = _run(
        _checkpoint(),
        [("a", "A", [{"type": "H", "target": 1}])],
        shots=21,
        budget=40,
    )
    server.save_branch_experiment(run)
    moved = shutil.move(str(first_dir / f"{run.run_id}.json"), str(second_dir))

    monkeypatch.setattr(server, "BRANCH_RUNS_DIR", second_dir)
    server.BRANCH_RUNS.clear()
    resumed = server.load_branch_experiment(run.run_id)

    assert moved == str(second_dir / f"{run.run_id}.json")
    assert resumed.snapshot() == run.snapshot()
