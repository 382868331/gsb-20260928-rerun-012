from http import HTTPStatus
from io import BytesIO
import json

from backend import server


class DummyResponse:
    status = HTTPStatus.OK
    body = b""
    headers: dict[str, str] = {}

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.headers[key] = str(value)

    def end_headers(self):
        pass

    def write(self, body):
        self.body += body


class DummyHandler(server.Handler):
    def __init__(self, payload):
        self.payload = payload
        self.response = DummyResponse()

    @property
    def headers(self):
        return {"Content-Length": str(len(json.dumps(self.payload).encode("utf-8")))}

    @property
    def rfile(self):
        return BytesIO(json.dumps(self.payload).encode("utf-8"))

    @property
    def wfile(self):
        return self.response

    def _send_json(self, status, body):
        encoded = json.dumps(body).encode("utf-8")
        self.response.status = status
        self.response.headers = {"Content-Type": "application/json", "Content-Length": str(len(encoded))}
        self.response.body = b""
        self.response.write(encoded)

    def decode(self):
        return self.response.status, json.loads(self.response.body.decode("utf-8"))


def _checkpoint():
    session = server.Session(
        {
            "qubits": 1,
            "initial_basis": "0",
            "seed": 9,
            "steps": [
                {"type": "H", "target": 0},
                {"type": "MEASURE", "target": 0},
            ],
        }
    )
    session.step()
    checkpoint = session.save_checkpoint("prefix", "api-prefix")
    server.CHECKPOINTS[checkpoint.checkpoint_id] = checkpoint
    return checkpoint


def test_branch_run_create_and_append_api(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "BRANCH_RUNS_DIR", tmp_path)
    server.BRANCH_RUNS.clear()
    checkpoint = _checkpoint()
    handler = DummyHandler(
        {
            "shots": 2,
            "budget": 4,
            "seed": 123,
            "branches": [
                {"label": "A", "steps": [{"type": "X", "target": 0}]},
                {"label": "B", "steps": []},
            ],
        }
    )
    handler.path = f"/api/checkpoints/{checkpoint.checkpoint_id}/branch-runs"
    handler.do_POST()
    status, created = handler.decode()
    assert status == HTTPStatus.CREATED
    assert [branch["shots"] for branch in created["branches"]] == [2, 2]

    handler.payload = {"shots": 2}
    handler.path = f"/api/branch-runs/{created['run_id']}/shots"
    handler.do_POST()
    status, appended = handler.decode()
    assert status == HTTPStatus.OK
    assert all(branch["shots"] == 4 for branch in appended["branches"])
