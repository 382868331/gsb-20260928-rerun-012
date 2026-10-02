import json
from http import HTTPStatus

import pytest

from backend import adaptive, branches, server


def _checkpoint(qubits=1, basis="0", steps=None, checkpoint_id="cp-adaptive"):
    session = server.Session(
        {
            "qubits": qubits,
            "initial_basis": basis,
            "seed": 5,
            "steps": steps if steps is not None else [],
        }
    )
    for _ in range(len(session.steps)):
        session.step()
    return session.save_checkpoint("prefix", checkpoint_id)


def _experiment(tmp_path, specs, *, seed=2024, experiment_id="exp-adaptive", name="exp"):
    return branches.BranchExperiment.create(
        _checkpoint(),
        specs,
        storage_dir=tmp_path / "experiments",
        base_seed=seed,
        name=name,
        experiment_id=experiment_id,
    )


def _study_payload(**overrides):
    payload = {
        "mode": "estimate",
        "branches": ["det"],
        "event": {"basis_states": ["1"]},
        "epsilon": 0.2,
        "delta": 0.05,
        "total_budget": 500,
        "batch_size": 7,
    }
    payload.update(overrides)
    return payload


def test_estimate_mode_decides_on_enumerable_deterministic_branch(tmp_path):
    experiment = _experiment(
        tmp_path,
        [{"branch_id": "det", "label": "X", "steps": [{"type": "X", "target": 0}], "budget": 0}],
    )
    study = adaptive.AdaptiveStudy.create(
        experiment, _study_payload(), storage_dir=tmp_path / "studies"
    )
    study.run(experiment)

    assert study.status == "decided"
    decision = study.decision
    assert decision["type"] == "estimate"
    assert decision["estimate"] == 1.0
    assert decision["interval"][0] >= 0.8
    assert decision["interval"][1] == 1.0
    # Every attempt went to the single branch, and each round is explained.
    assert study.spent["det"] == experiment.branches["det"]["completed_shots"]
    allocations = [entry for entry in study.log if entry["action"] == "allocate"]
    assert allocations and all("证据最不足" in entry["reason"] for entry in allocations)
    stop = study.log[-1]
    assert stop["action"] == "stop"
    assert "epsilon" in stop["reason"]


def test_compare_mode_with_equal_true_probabilities_reports_within_epsilon(tmp_path):
    specs = [
        {"branch_id": "a", "label": "H-a", "steps": [{"type": "H", "target": 0}], "budget": 0},
        {"branch_id": "b", "label": "H-b", "steps": [{"type": "H", "target": 0}], "budget": 0},
    ]
    experiment = _experiment(tmp_path, specs)
    study = adaptive.AdaptiveStudy.create(
        experiment,
        _study_payload(
            mode="compare",
            branches=["a", "b"],
            epsilon=0.15,
            total_budget=2000,
            batch_size=25,
        ),
        storage_dir=tmp_path / "studies",
    )
    study.run(experiment)

    assert study.status == "decided"
    assert study.decision["type"] == "within_epsilon"
    low, high = study.decision["interval"]
    assert low <= 0 <= high
    assert high - low <= 2 * 0.15 + 1e-9
    # Budget went to the weaker branch each round, not a fixed equal split
    # decided up front: totals end up balanced only because evidence was.
    assert study.spent["a"] + study.spent["b"] <= 2000


def test_rare_filter_separates_acceptance_and_conditional_uncertainty(tmp_path):
    specs = [
        {
            "branch_id": "det",
            "label": "measure q0",
            "steps": [{"type": "MEASURE", "target": 0}],
            "filters": [{"step": 0, "outcome": 1}],
            "budget": 0,
        }
    ]
    experiment = _experiment(tmp_path, specs)
    study = adaptive.AdaptiveStudy.create(
        experiment,
        _study_payload(event={"basis_states": ["0"]}, total_budget=60, batch_size=10),
        storage_dir=tmp_path / "studies",
    )
    study.run(experiment)

    # The filter never fires on |0>: zero hits is missing evidence, not a
    # zero probability conclusion, and the study cannot decide.
    assert study.status == "undecided"
    evidence = study.snapshot(experiment)["evidence"][0]
    assert evidence["passed_shots"] == 0
    assert evidence["has_evidence"] is False
    assert evidence["acceptance_uncertainty"] is not None
    assert evidence["conditional_estimate"] is None
    assert evidence["conditional_uncertainty"] is None
    reasons = "\n".join(entry["reason"] for entry in study.log)
    assert "没有证据" in reasons
    assert "未确定" in study.decision["reason"]


def test_exhausted_budget_stays_undecided_instead_of_forcing_a_winner(tmp_path):
    specs = [
        {"branch_id": "a", "label": "H-a", "steps": [{"type": "H", "target": 0}], "budget": 0},
        {"branch_id": "b", "label": "H-b", "steps": [{"type": "H", "target": 0}], "budget": 0},
    ]
    experiment = _experiment(tmp_path, specs)
    study = adaptive.AdaptiveStudy.create(
        experiment,
        _study_payload(
            mode="compare",
            branches=["a", "b"],
            epsilon=0.01,
            total_budget=60,
            batch_size=10,
        ),
        storage_dir=tmp_path / "studies",
    )
    study.run(experiment)

    assert study.status == "undecided"
    assert study.decision["type"] == "undecided"
    assert study._total_spent() == 60
    # Further advances never flip the outcome or keep sampling.
    before = study._total_spent()
    study.advance(experiment)
    assert study.status == "undecided"
    assert study._total_spent() == before


def test_batch_size_does_not_change_final_evidence_or_stop_explanation(tmp_path):
    specs = [
        {"branch_id": "a", "label": "H-a", "steps": [{"type": "H", "target": 0}], "budget": 0},
        {"branch_id": "b", "label": "H-b", "steps": [{"type": "H", "target": 0}], "budget": 0},
    ]
    results = {}
    for name, batch in (("fine", 8), ("coarse", 13)):
        experiment = _experiment(tmp_path / name, specs)
        study = adaptive.AdaptiveStudy.create(
            experiment,
            _study_payload(
                mode="compare",
                branches=["a", "b"],
                epsilon=0.15,
                total_budget=2000,
                batch_size=batch,
            ),
            storage_dir=tmp_path / name / "studies",
        )
        study.run(experiment)
        results[name] = (study, experiment)

    fine, coarse = results["fine"][0], results["coarse"][0]
    assert fine.status == coarse.status == "decided"
    assert fine.decision["type"] == coarse.decision["type"] == "within_epsilon"
    assert fine.decision["reason"] == coarse.decision["reason"]
    # Both final intervals are valid time-uniform intervals around ~0.
    assert results["fine"][0].decision["interval"][0] <= 0
    assert results["coarse"][0].decision["interval"][1] >= 0

    # Attempt identity is replayable: re-running the fine study's per-branch
    # totals on a fresh identical experiment reproduces its counts exactly.
    replay = _experiment(tmp_path / "replay", specs)
    fine_experiment = results["fine"][1]
    for branch_id in ("a", "b"):
        total = fine_experiment.branches[branch_id]["completed_shots"]
        replay.add_shots(branch_id, total, source="replay")
    for branch_id in ("a", "b"):
        assert replay.branches[branch_id]["counts"] == fine_experiment.branches[branch_id]["counts"]


def test_pause_and_resume_keeps_attempts_rule_and_allocation_traceable(tmp_path):
    specs = [
        {"branch_id": "a", "label": "H-a", "steps": [{"type": "H", "target": 0}], "budget": 0},
        {"branch_id": "b", "label": "H-b", "steps": [{"type": "H", "target": 0}], "budget": 0},
    ]
    payload = _study_payload(
        mode="compare", branches=["a", "b"], epsilon=0.15, total_budget=1200, batch_size=10
    )

    paused_experiment = _experiment(tmp_path / "paused", specs)
    paused = adaptive.AdaptiveStudy.create(
        paused_experiment,
        payload,
        storage_dir=tmp_path / "paused" / "studies",
        study_id="study-paused",
        created_at=1000.0,
    )
    for _ in range(5):
        paused.advance(paused_experiment)
    # Resume from disk: attempts, stopping rule and allocation log survive.
    resumed = adaptive.AdaptiveStudy.load(tmp_path / "paused" / "studies" / "study-paused.json")
    reloaded_experiment = branches.BranchExperiment.load(
        tmp_path / "paused" / "experiments" / "exp-adaptive.json"
    )
    resumed.run(reloaded_experiment)

    control_experiment = _experiment(tmp_path / "control", specs)
    control = adaptive.AdaptiveStudy.create(
        control_experiment,
        payload,
        storage_dir=tmp_path / "control" / "studies",
        study_id="study-paused",
        created_at=1000.0,
    )
    control.run(control_experiment)

    assert resumed.status == control.status == "decided"
    assert resumed.decision == control.decision
    assert resumed.spent == control.spent
    assert [entry["action"] for entry in resumed.log] == [entry["action"] for entry in control.log]
    for branch_id in ("a", "b"):
        assert (
            reloaded_experiment.branches[branch_id]["counts"]
            == control_experiment.branches[branch_id]["counts"]
        )


def test_new_precision_or_event_creates_new_study_and_keeps_history(tmp_path):
    experiment = _experiment(
        tmp_path,
        [{"branch_id": "det", "label": "H", "steps": [{"type": "H", "target": 0}], "budget": 0}],
    )
    first = adaptive.AdaptiveStudy.create(
        experiment,
        _study_payload(branches=["det"], epsilon=0.2, total_budget=500, batch_size=10),
        storage_dir=tmp_path / "studies",
        study_id="study-original",
    )
    first.run(experiment)
    assert first.status == "decided"

    # A changed precision target is a new research decision, not an edit.
    second = adaptive.AdaptiveStudy.create(
        experiment,
        _study_payload(branches=["det"], epsilon=0.05, total_budget=5000, batch_size=50),
        storage_dir=tmp_path / "studies",
        study_id="study-tighter",
    )
    second.run(experiment)

    reloaded = adaptive.AdaptiveStudy.load(tmp_path / "studies" / "study-original.json")
    assert reloaded.spec["epsilon"] == 0.2
    assert reloaded.status == "decided"
    assert reloaded.decision == first.decision
    assert second.study_id != first.study_id
    assert second.spec["epsilon"] == 0.05
    assert second.status == "decided"
    # The tighter study reused the shared branch stream without re-drawing.
    assert experiment.branches["det"]["completed_shots"] == (
        first.spent["det"] + second.spent["det"]
    )


def test_invalid_study_specs_are_rejected(tmp_path):
    experiment = _experiment(
        tmp_path,
        [{"branch_id": "det", "label": "H", "steps": [{"type": "H", "target": 0}], "budget": 0}],
    )
    with pytest.raises(ValueError):
        adaptive.AdaptiveStudy.create(
            experiment, _study_payload(event={"basis_states": ["10"]}), storage_dir=tmp_path
        )
    with pytest.raises(ValueError):
        adaptive.AdaptiveStudy.create(
            experiment, _study_payload(epsilon=0), storage_dir=tmp_path
        )
    with pytest.raises(ValueError):
        adaptive.AdaptiveStudy.create(
            experiment, _study_payload(branches=["missing"]), storage_dir=tmp_path
        )
    with pytest.raises(ValueError):
        adaptive.AdaptiveStudy.create(
            experiment, _study_payload(mode="compare", branches=["det"]), storage_dir=tmp_path
        )


class DummyResponse:
    def __init__(self):
        self.status = None
        self.body = None


class DummyHandler(server.Handler):
    def __init__(self, payload=None):
        encoded = json.dumps(payload or {}).encode("utf-8")
        self.headers = {"Content-Length": str(len(encoded))}
        self.rfile = type("Input", (), {"read": lambda self, length: encoded})()
        self.wfile = type("Output", (), {"write": lambda self, data: None})()
        self.response = DummyResponse()
        self.sent_headers = []

    def send_response(self, status):
        self.response.status = status

    def send_header(self, key, value):
        self.sent_headers.append((key, value))

    def end_headers(self):
        pass

    def _send_json(self, status, body):
        self.response.status = status
        self.response.body = body


def test_study_api_create_advance_run_and_get(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "BRANCH_DIR", tmp_path / "experiments")
    monkeypatch.setattr(server, "STUDY_DIR", tmp_path / "studies")
    checkpoint = _checkpoint()
    server.CHECKPOINTS["cp-adaptive"] = checkpoint

    create_experiment = DummyHandler(
        {
            "name": "api-adaptive",
            "seed": 77,
            "branches": [
                {"branch_id": "a", "label": "H-a", "steps": [{"type": "H", "target": 0}], "budget": 0},
                {"branch_id": "b", "label": "H-b", "steps": [{"type": "H", "target": 0}], "budget": 0},
            ],
        }
    )
    create_experiment.path = "/api/checkpoints/cp-adaptive/branch-experiments"
    server.Handler.do_POST(create_experiment)
    assert create_experiment.response.status == HTTPStatus.CREATED
    experiment_id = create_experiment.response.body["experiment_id"]

    create_study = DummyHandler(
        {
            "mode": "compare",
            "branches": ["a", "b"],
            "event": {"basis_states": ["1"]},
            "epsilon": 0.15,
            "delta": 0.05,
            "total_budget": 2000,
            "batch_size": 25,
        }
    )
    create_study.path = f"/api/branch-experiments/{experiment_id}/studies"
    server.Handler.do_POST(create_study)
    assert create_study.response.status == HTTPStatus.CREATED
    study_id = create_study.response.body["study_id"]
    assert create_study.response.body["status"] == "running"

    advance = DummyHandler({})
    advance.path = f"/api/studies/{study_id}/advance"
    server.Handler.do_POST(advance)
    assert advance.response.status == HTTPStatus.OK
    assert advance.response.body["total_spent"] == 25
    assert advance.response.body["log"][-1]["action"] == "allocate"

    run = DummyHandler({})
    run.path = f"/api/studies/{study_id}/run"
    server.Handler.do_POST(run)
    assert run.response.status == HTTPStatus.OK
    assert run.response.body["status"] == "decided"
    assert run.response.body["decision"]["type"] == "within_epsilon"

    get = DummyHandler()
    get.path = f"/api/studies/{study_id}"
    server.Handler.do_GET(get)
    assert get.response.status == HTTPStatus.OK
    assert get.response.body["study_id"] == study_id
    assert get.response.body["decision"]["type"] == "within_epsilon"
    assert "时一致置信序列" in get.response.body["sequential_note"]

    bad = DummyHandler({"mode": "estimate", "branches": ["a"], "event": {"basis_states": ["2"]}, "epsilon": 0.1, "total_budget": 10})
    bad.path = f"/api/branch-experiments/{experiment_id}/studies"
    server.Handler.do_POST(bad)
    assert bad.response.status == HTTPStatus.BAD_REQUEST

