import json
from http import HTTPStatus

from backend import server


class DummyResponse:
    def __init__(self):
        self.status = None
        self.body = None

    def __call__(self, status):
        self.status = status



class DummyHandler(server.Handler):
    def __init__(self, payload):
        encoded = json.dumps(payload).encode("utf-8")
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

def _make_checkpoint():
    session = server.Session(
        {
            "qubits": 2,
            "initial_basis": "00",
            "seed": 44,
            "steps": [
                {"type": "H", "target": 0},
                {"type": "MEASURE", "target": 0},
            ],
        }
    )
    session.step()
    session.step()
    checkpoint = session.save_checkpoint("prefix", "cp-api")
    server.CHECKPOINTS["cp-api"] = checkpoint
    return checkpoint


def test_branch_experiment_create_get_and_patch(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "BRANCH_DIR", tmp_path)
    checkpoint = _make_checkpoint()
    prefix_outcome = checkpoint.measurements[-1]["outcome"]
    create_payload = {
        "name": "api-branches",
        "seed": 909,
        "branches": [
            {"branch_id": "flip", "label": "flip", "steps": [{"type": "X", "target": 1}], "budget": 12},
            {"branch_id": "idle", "label": "idle", "steps": [], "budget": 8},
            {
                "branch_id": "keep-1",
                "label": "keep q0=1",
                "steps": [{"type": "MEASURE", "target": 0}],
                "filters": [{"step": 0, "outcome": prefix_outcome}],
                "budget": 16,
            },
        ],
    }
    handler = DummyHandler(create_payload)
    handler.path = "/api/checkpoints/cp-api/branch-experiments"
    server.Handler.do_POST(handler)
    assert handler.response.status == HTTPStatus.CREATED
    created = handler.response.body
    assert created["pooled_across_conditions"] is False
    assert created["common_prefix"]["prefix_measurements"] == checkpoint.measurements
    assert created["branches"][0]["completed_shots"] == 12
    assert created["branches"][1]["completed_shots"] == 8
    filtered = created["branches"][2]
    assert filtered["attempts"] == 16
    assert filtered["passed_shots"] == 16
    assert filtered["has_evidence"] is True
    assert filtered["filters"] == [{"step": 0, "outcome": prefix_outcome}]

    handler.path = f"/api/branch-experiments/{created['experiment_id']}"
    server.Handler.do_GET(handler)
    assert handler.response.status == HTTPStatus.OK
    assert handler.response.body["experiment_id"] == created["experiment_id"]

    patch_handler = DummyHandler({"shots": 5, "source": "api-extra"})
    patch_handler.path = (
        f"/api/branch-experiments/{created['experiment_id']}/branches/flip/shots"
    )
    server.Handler.do_PATCH(patch_handler)
    assert patch_handler.response.status == HTTPStatus.OK
    updated = patch_handler.response.body
    assert updated["branches"][0]["completed_shots"] == 17
    assert updated["branches"][1]["completed_shots"] == 8
    assert updated["branches"][0]["batches"][-1]["source"] == "api-extra"

    inquiry_payload = {
        "mode": "compare",
        "branch_ids": ["flip", "idle"],
        "event": "__pass__",
        "margin": 0.01,
        "total_budget": 210,
        "batch_size": 10,
    }
    inquiry_handler = DummyHandler({"inquiry": inquiry_payload})
    inquiry_handler.path = (
        f"/api/branch-experiments/{created['experiment_id']}/inquiries"
    )
    server.Handler.do_PATCH(inquiry_handler)
    assert inquiry_handler.response.status == HTTPStatus.OK
    created_inquiry = inquiry_handler.response.body["inquiries"][-1]
    assert created_inquiry["event"] == "__pass__"
    assert created_inquiry["total_budget"] == 210
    assert created_inquiry["decisions"][0]["action"] == "created"

    server.Handler.do_GET(handler)
    assert handler.response.status == HTTPStatus.OK
    assert [item["inquiry_id"] for item in handler.response.body["inquiries"]] == [
        created_inquiry["inquiry_id"]
    ]

    adaptive_handler = DummyHandler({})
    adaptive_handler.path = (
        f"/api/branch-experiments/{created['experiment_id']}"
        f"/inquiries/{created_inquiry['inquiry_id']}/adaptive-batch"
    )
    server.Handler.do_PATCH(adaptive_handler)
    assert adaptive_handler.response.status == HTTPStatus.OK
    assert adaptive_handler.response.body["evaluation"]["used_budget"] == 10
    selected_branch = next(
        branch
        for branch in adaptive_handler.response.body["branches"]
        if branch["batches"][-1]["source"]
        == f"adaptive:{created_inquiry['inquiry_id']}"
    )
    assert selected_branch["batches"][-1]["shots"] == 10

    server.BRANCH_EXPERIMENTS.clear()
    server.Handler.do_GET(handler)
    reopened = handler.response.body
    assert reopened["inquiries"][0]["inquiry_id"] == created_inquiry["inquiry_id"]
    assert any(
        decision["action"] == "sample"
        and decision["batch_id"] == selected_branch["batches"][-1]["batch_id"]
        for decision in reopened["inquiries"][0]["decisions"]
    )
    assert any(
        batch["source"] == f"adaptive:{created_inquiry['inquiry_id']}"
        for branch in reopened["branches"]
        for batch in branch["batches"]
    )

    event_ids = list(next(
        branch["counts"] for branch in created["branches"] if len(branch["counts"]) > 1
    ).keys())
    shared_branch = next(
        branch["branch_id"] for branch in created["branches"] if len(branch["counts"]) > 1
    )
    plan_payload = {
        "strategy": "balanced",
        "total_budget": 5,
        "goals": [
            {
                "goal_id": "event-a",
                "mode": "single",
                "branch_ids": [shared_branch],
                "event": event_ids[0],
                "margin": 0.01,
                "batch_size": 2,
            },
            {
                "goal_id": "event-b",
                "mode": "single",
                "branch_ids": [shared_branch],
                "event": event_ids[1],
                "margin": 0.01,
                "batch_size": 2,
            },
        ],
    }
    plan_handler = DummyHandler({"research_plan": plan_payload})
    plan_handler.path = f"/api/branch-experiments/{created['experiment_id']}/research-plans"
    server.Handler.do_PATCH(plan_handler)
    assert plan_handler.response.status == HTTPStatus.OK
    plan_id = plan_handler.response.body["research_plan"]["plan_id"]

    batch_handler = DummyHandler({})
    batch_handler.path = (
        f"/api/branch-experiments/{created['experiment_id']}/research-plans/{plan_id}/batches"
    )
    server.Handler.do_PATCH(batch_handler)
    assert batch_handler.response.status == HTTPStatus.OK
    plan = batch_handler.response.body["research_plan"]
    assert plan["executed_shots"] == 2
    assert plan["batches"][0]["goal_ids"] == ["event-a", "event-b"]
