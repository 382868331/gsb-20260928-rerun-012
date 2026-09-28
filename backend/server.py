"""Local HTTP API for the quantum experiment bench.

Run:  .venv/bin/python -m backend.server  (from the workspace root)
No third-party web framework is required; numpy is the only dependency.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from . import quantum

HOST = "127.0.0.1"
PORT = 8000
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CIRCUITS_FILE = DATA_DIR / "circuits.json"


def _load_circuits() -> dict[str, Any]:
    if not CIRCUITS_FILE.exists():
        return {"circuits": {}}
    with CIRCUITS_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_circuits(data: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CIRCUITS_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    tmp.replace(CIRCUITS_FILE)


class Session:
    """A stepped execution whose state lives on the server between calls."""

    def __init__(self, payload: dict[str, Any]):
        self.qubits = int(payload["qubits"])
        self.basis = payload["initial_basis"]
        self.steps = list(payload["steps"])
        self.seed = payload.get("seed")
        self.rng = np.random.default_rng(self.seed if self.seed is not None else None)
        self.state = quantum.initial_state(self.qubits, self.basis)
        self.next_index = 0
        self.measurements: list[dict[str, Any]] = []

    def snapshot(self) -> dict[str, Any]:
        return {
            "qubits": self.qubits,
            "initial_basis": self.basis,
            "seed": self.seed,
            "next_index": self.next_index,
            "total_steps": len(self.steps),
            "finished": self.next_index >= len(self.steps),
            "probabilities": quantum.probability_map(self.state, self.qubits),
            "measurements": self.measurements,
        }

    def step(self) -> dict[str, Any]:
        index = self.next_index
        step = self.steps[index]
        measured = None
        if step["type"] == quantum.MEASURE_GATE:
            self.state, outcome = quantum.measure(
                self.state, self.qubits, int(step["target"]), self.rng
            )
            measured = {"target": int(step["target"]), "outcome": outcome}
            self.measurements.append({"step": index, **measured})
        else:
            self.state = quantum.apply_gate(self.state, self.qubits, step)
        self.next_index += 1
        return {
            "executed": {"index": index, "type": step["type"], "measured": measured},
            **self.snapshot(),
        }


SESSIONS: dict[str, Session] = {}
SESSIONS_LOCK = threading.Lock()
STORAGE_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "QuantumBench/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, body: Any) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "JSON body required"})
            return None
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
            return None
        if not isinstance(payload, dict):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "body must be a JSON object"})
            return None
        return payload

    def do_GET(self) -> None:
        path = self.path.rstrip("/") or "/"
        if path == "/api/health":
            self._send_json(HTTPStatus.OK, {"ok": True, "max_qubits": quantum.MAX_QUBITS})
            return
        if path == "/api/circuits":
            with STORAGE_LOCK:
                self._send_json(HTTPStatus.OK, _load_circuits())
            return
        if path.startswith("/api/circuits/"):
            name = path.rsplit("/", 1)[1]
            with STORAGE_LOCK:
                circuit = _load_circuits()["circuits"].get(name)
            if circuit is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "circuit not found"})
            else:
                self._send_json(HTTPStatus.OK, circuit)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.rstrip("/") or "/"
        payload = self._read_json()
        if payload is None:
            return

        if path == "/api/validate":
            errors = quantum.validate_circuit(payload)
            self._send_json(HTTPStatus.OK, {"valid": not errors, "errors": errors})
            return

        if path == "/api/simulate":
            errors = quantum.validate_circuit(payload)
            if errors:
                self._send_json(HTTPStatus.BAD_REQUEST, {"valid": False, "errors": errors})
                return
            seed = payload.get("seed")
            rng = np.random.default_rng(seed if seed is not None else None)
            result = quantum.run_trace(
                int(payload["qubits"]), payload["initial_basis"], payload["steps"], rng
            )
            self._send_json(HTTPStatus.OK, {"valid": True, "seed": seed, **result})
            return

        if path == "/api/experiments":
            errors = quantum.validate_circuit(payload)
            if errors:
                self._send_json(HTTPStatus.BAD_REQUEST, {"valid": False, "errors": errors})
                return
            shots = payload.get("shots", 1024)
            if not isinstance(shots, int) or isinstance(shots, bool) or not 1 <= shots <= 100_000:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "shots must be an integer in [1, 100000]"},
                )
                return
            result = quantum.run_experiment(
                int(payload["qubits"]),
                payload["initial_basis"],
                payload["steps"],
                shots,
                payload.get("seed"),
            )
            self._send_json(HTTPStatus.OK, result)
            return

        if path == "/api/sessions":
            errors = quantum.validate_circuit(payload)
            if errors:
                self._send_json(HTTPStatus.BAD_REQUEST, {"valid": False, "errors": errors})
                return
            session = Session(payload)
            session_id = uuid.uuid4().hex
            with SESSIONS_LOCK:
                SESSIONS[session_id] = session
            self._send_json(
                HTTPStatus.CREATED, {"session_id": session_id, **session.snapshot()}
            )
            return

        if path.startswith("/api/sessions/") and path.endswith("/step"):
            parts = path.split("/")
            session_id = parts[3] if len(parts) == 5 else ""
            with SESSIONS_LOCK:
                session = SESSIONS.get(session_id)
                if session is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "session not found"})
                    return
                if session.finished:
                    self._send_json(
                        HTTPStatus.CONFLICT,
                        {"error": "session already finished", **session.snapshot()},
                    )
                    return
                result = session.step()
            self._send_json(HTTPStatus.OK, {"session_id": session_id, **result})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_PUT(self) -> None:
        if not self.path.startswith("/api/circuits/"):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        name = self.path.rstrip("/").rsplit("/", 1)[1]
        if not name or len(name) > 80 or any(ch in name for ch in "/\\\\\\"):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid circuit name"})
            return
        payload = self._read_json()
        if payload is None:
            return
        errors = quantum.validate_circuit(payload)
        if errors:
            self._send_json(HTTPStatus.BAD_REQUEST, {"valid": False, "errors": errors})
            return
        with STORAGE_LOCK:
            data = _load_circuits()
            previous = data["circuits"].get(name)
            version = int(previous["version"]) + 1 if previous else 1
            record = {
                "name": name,
                "version": version,
                "updated_at": time.time(),
                "qubits": int(payload["qubits"]),
                "initial_basis": payload["initial_basis"],
                "seed": payload.get("seed"),
                "steps": payload["steps"],
            }
            data["circuits"][name] = record
            _save_circuits(data)
        self._send_json(HTTPStatus.OK, record)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Quantum bench API listening on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
