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
    create_payload = {
        "name": "api-branches",
        "seed": 909,
        "branches": [
            {"branch_id": "flip", "label": "flip", "steps": [{"type": "X", "target": 1}], "budget": 12},
            {"branch_id": "idle", "label": "idle", "steps": [], "budget": 8},
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
