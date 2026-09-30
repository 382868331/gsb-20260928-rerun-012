import pytest
import threading

import numpy as np

from backend import quantum, server


class FakeHandler(server.Handler):
    """Drives Handler routing logic without a real socket."""

    def __init__(self):
        pass

    def _send_json(self, status, body):
        self.status = status
        self.body = body

    def _read_json(self):
        return self._payload

    def post(self, path, payload):
        self.path = path
        self._payload = payload
        self.status = None
        server.Handler._route_post(self, path, payload)
        return self.status, self.body

    def get(self, path):
        self.path = path
        self.status = None
        server.Handler.do_GET(self)
        return self.status, self.body


@pytest.fixture()
def api():
    yield FakeHandler()


BELL_CIRCUIT = {
    "qubits": 2,
    "initial_basis": "00",
    "seed": 11,
    "steps": [
        {"type": "H", "target": 0},
        {"type": "CNOT", "control": 0, "target": 1},
        {"type": "MEASURE", "target": 0},
        {"type": "MEASURE", "target": 1},
    ],
}


def test_step_session_through_finish_then_conflict(api):
    status, created = api.post("/api/sessions", BELL_CIRCUIT)
    assert status == 201
    session_id = created["session_id"]
    assert created["next_index"] == 1
    state = created
    for _ in range(3):
        status, state = api.post(f"/api/sessions/{session_id}/step", {})
        assert status == 200
    assert state["finished"] is True

    status, conflict = api.post(f"/api/sessions/{session_id}/step", {})
    assert status == 409
    assert conflict["finished"] is True
    assert conflict["next_index"] == 4

    status, fetched = api.get(f"/api/sessions/{session_id}")
    assert status == 200
    assert fetched["measurements"] == state["measurements"]


def test_rapid_clicks_advance_only_one_step_each(api):
    _, created = api.post("/api/sessions", BELL_CIRCUIT)
    session_id = created["session_id"]
    results = []

    def click():
        results.append(api.post(f"/api/sessions/{session_id}/step", {}))

    threads = [threading.Thread(target=click) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    statuses = sorted(state["next_index"] for _status, state in results)
    assert statuses == [2, 3, 4]


def test_checkpoint_resume_does_not_redraw_measurements(api, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "CHECKPOINTS_FILE", tmp_path / "checkpoints.json")

    _, created = api.post("/api/sessions", BELL_CIRCUIT)
    session_id = created["session_id"]
    api.post(f"/api/sessions/{session_id}/step", {})  # CNOT
    _, measured = api.post(f"/api/sessions/{session_id}/step", {})  # MEASURE q0
    first_outcome = measured["measurements"][0]["outcome"]

    status, saved = api.post(
        f"/api/sessions/{session_id}/checkpoint", {"label": "mid", "persist": True}
    )
    assert status == 200
    checkpoint_id = saved["checkpoint_id"]
    assert saved["next_index"] == 3

    status, resumed = api.post("/api/sessions/resume", {"checkpoint_id": checkpoint_id})
    assert status == 201
    assert resumed["next_index"] == 3
    assert resumed["restored_measurements"] == [
        {"step": 2, "target": 0, "outcome": first_outcome}
    ]
    new_id = resumed["session_id"]

    _, continued = api.post(f"/api/sessions/{new_id}/step", {})
    assert continued["measurements"][0]["outcome"] == first_outcome
    assert continued["measurements"][1]["outcome"] == first_outcome

    # Resuming again yields the identical historical measurement, never a redraw.
    _, resumed_again = api.post("/api/sessions/resume", {"checkpoint_id": checkpoint_id})
    assert resumed_again["restored_measurements"][0]["outcome"] == first_outcome


def test_compare_two_branches_from_checkpoint(api):
    _, created = api.post("/api/sessions", BELL_CIRCUIT)
    session_id = created["session_id"]
    for _ in range(2):
        api.post(f"/api/sessions/{session_id}/step", {})
    _, checkpoint = api.post(f"/api/sessions/{session_id}/checkpoint", {})

    branch_x = [{"type": "X", "target": 1}]
    branch_i = [{"type": "I_PLACEHOLDER"}]
    status, bad = api.post(
        "/api/checkpoints/compare",
        {"checkpoint": checkpoint, "branches": [branch_x, branch_i]},
    )
    assert status == 400

    branch_z = [{"type": "Z", "target": 1}]
    status, result = api.post(
        "/api/checkpoints/compare",
        {"checkpoint": checkpoint, "branches": [branch_x, branch_z]},
    )
    assert status == 200
    first_branch, second_branch = result["branches"]
    assert first_branch["steps"] == branch_x
    # Both branches fork the same saved RNG state; shared history keeps Bell correlation.
    assert first_branch["measurements"] == second_branch["measurements"]
    assert set(first_branch["final_probabilities"]) == {"00", "01", "10", "11"}


def test_conditional_gate_and_step_matches_one_shot():
    circuit = {
        "qubits": 2,
        "initial_basis": "00",
        "seed": 99,
        "steps": [
            {"type": "H", "target": 0},
            {"type": "MEASURE", "target": 0},
            {
                "type": "IF",
                "control": 0,
                "equals": 1,
                "then": {"type": "X", "target": 1},
            },
            {"type": "MEASURE", "target": 1},
        ],
    }
    assert quantum.validate_circuit(circuit) == []

    rng = np.random.default_rng(circuit["seed"])
    one_shot = quantum.run_trace(2, "00", circuit["steps"], rng)

    session = server.Session(circuit)
    stepped = [session.step() for _ in circuit["steps"]]
    assert [item["probabilities"] for item in stepped] == [
        item["probabilities"] for item in one_shot["trace"]
    ]
    assert [item["executed"]["condition_fired"] for item in stepped] == [
        item["condition_fired"] for item in one_shot["trace"]
    ]
    first = one_shot["measurements"][0]["outcome"]
    second = one_shot["measurements"][1]["outcome"]
    # q1 is flipped to 1 exactly when q0 measured 1, otherwise stays 0: q1 == q0.
    assert second == first


def test_conditional_gate_rejected_without_prior_measurement():
    errors = quantum.validate_circuit(
        {
            "qubits": 1,
            "initial_basis": "0",
            "steps": [
                {"type": "IF", "control": 0, "equals": 1, "then": {"type": "X", "target": 0}},
                {"type": "MEASURE", "target": 0},
            ],
        }
    )
    assert any("measured" in error["message"] for error in errors)


def test_statistics_experiment_does_not_touch_interactive_session(api):
    _, created = api.post("/api/sessions", BELL_CIRCUIT)
    session_id = created["session_id"]
    session_after_first = list(created["measurements"])

    # Heavy statistics run must not advance or reseed the interactive session.
    status, stats = api.post("/api/experiments", {**BELL_CIRCUIT, "shots": 500})
    assert status == 200
    assert sum(stats["counts"].values()) == 500

    _, fetched = api.get(f"/api/sessions/{session_id}")
    assert fetched["next_index"] == 1
    assert fetched["measurements"] == session_after_first

    _, cnot_step = api.post(f"/api/sessions/{session_id}/step", {})
    assert cnot_step["measurements"] == []
    _, measured_step = api.post(f"/api/sessions/{session_id}/step", {})
    # Session RNG continued from its own stream, unaffected by experiment draws.
    assert len(measured_step["measurements"]) == 1
