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


def test_inquiry_write_is_visible_to_cached_get_and_survives_reload(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "BRANCH_DIR", tmp_path)
    monkeypatch.setattr(server, "BRANCH_EXPERIMENTS", {})
    _make_checkpoint()
    create_handler = DummyHandler(
        {
            "name": "api-inquiry",
            "seed": 77,
            "branches": [
                {"branch_id": "a", "label": "A", "steps": [], "budget": 100},
                {"branch_id": "b", "label": "B", "steps": [], "budget": 100},
            ],
        }
    )
    create_handler.path = "/api/checkpoints/cp-api/branch-experiments"
    server.Handler.do_POST(create_handler)
    experiment_id = create_handler.response.body["experiment_id"]

    inquiry_payload = {
        "inquiry": {
            "mode": "compare",
            "branch_ids": ["a", "b"],
            "event": "00",
            "margin": 0.01,
            "total_budget": 210,
            "batch_size": 10,
        }
    }
    inquiry_handler = DummyHandler(inquiry_payload)
    inquiry_handler.path = f"/api/branch-experiments/{experiment_id}/inquiries"
    server.Handler.do_PATCH(inquiry_handler)
    inquiry_id = inquiry_handler.response.body["inquiry_id"]
    assert inquiry_handler.response.body["decisions"][0]["action"] == "created"

    get_handler = DummyHandler({})
    get_handler.path = f"/api/branch-experiments/{experiment_id}"
    server.Handler.do_GET(get_handler)
    assert [item["inquiry_id"] for item in get_handler.response.body["inquiries"]] == [inquiry_id]

    batch_handler = DummyHandler({})
    batch_handler.path = (
        f"/api/branch-experiments/{experiment_id}/inquiries/{inquiry_id}/adaptive-batch"
    )
    server.Handler.do_PATCH(batch_handler)
    assert batch_handler.response.status == HTTPStatus.OK
    sampled_branch = next(
        branch
        for branch in batch_handler.response.body["branches"]
        if branch["batches"][-1]["source"] == f"adaptive:{inquiry_id}"
    )
    assert sampled_branch["batches"][-1]["shots"] == 10

    changed_inquiry_handler = DummyHandler(
        {"inquiry": {**inquiry_payload["inquiry"], "event": "__pass__"}}
    )
    changed_inquiry_handler.path = f"/api/branch-experiments/{experiment_id}/inquiries"
    server.Handler.do_PATCH(changed_inquiry_handler)
    second_inquiry_id = changed_inquiry_handler.response.body["inquiry_id"]
    assert second_inquiry_id != inquiry_id

    server.BRANCH_EXPERIMENTS.clear()
    reopen_handler = DummyHandler({})
    reopen_handler.path = f"/api/branch-experiments/{experiment_id}"
    server.Handler.do_GET(reopen_handler)
    reopened = reopen_handler.response.body
    assert [item["inquiry_id"] for item in reopened["inquiries"]] == [
        inquiry_id,
        second_inquiry_id,
    ]
    assert reopened["inquiries"][0]["event"] == "00"
    assert reopened["inquiries"][1]["event"] == "__pass__"
    adaptive_branch = next(
        branch for branch in reopened["branches"] if branch["completed_shots"] == 110
    )
    assert adaptive_branch["batches"] == [
        {
            "batch_id": adaptive_branch["batches"][0]["batch_id"],
            "source": "initial-budget",
            "shots": 100,
            "total_shots_after_batch": 100,
        },
        {
            "batch_id": adaptive_branch["batches"][1]["batch_id"],
            "source": f"adaptive:{inquiry_id}",
            "shots": 10,
            "total_shots_after_batch": 110,
        },
    ]
