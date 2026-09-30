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
CHECKPOINTS_FILE = DATA_DIR / "checkpoints.json"


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


def _load_checkpoints() -> dict[str, Any]:
    if not CHECKPOINTS_FILE.exists():
        return {"checkpoints": {}}
    with CHECKPOINTS_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_checkpoints(data: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINTS_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    tmp.replace(CHECKPOINTS_FILE)


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
        self.outcomes: dict[str, int] = {}
        self.lock = threading.Lock()

    @property
    def finished(self) -> bool:
        return self.next_index >= len(self.steps)

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> "Session":
        session = cls.__new__(cls)
        session.qubits = int(checkpoint["qubits"])
        session.basis = checkpoint["initial_basis"]
        session.seed = checkpoint.get("seed")
        session.steps = list(checkpoint["steps"])
        session.state = cls.restore_state(checkpoint)
        session.next_index = int(checkpoint["next_index"])
        session.measurements = list(checkpoint.get("measurements", []))
        session.outcomes = {
            int(key): int(value) for key, value in checkpoint.get("outcomes", {}).items()
        }
        session.rng = np.random.default_rng()
        rng_state = checkpoint.get("rng_state")
        if rng_state is not None:
            session.rng.bit_generator.state = rng_state
        session.lock = threading.Lock()
        return session

    def checkpoint(self, label: str | None = None) -> dict[str, Any]:
        return {
            "label": label,
            "saved_at": time.time(),
            "qubits": self.qubits,
            "initial_basis": self.basis,
            "seed": self.seed,
            "steps": self.steps,
            "next_index": self.next_index,
            "state": [[float(item.real), float(item.imag)] for item in self.state.tolist()],
            "measurements": self.measurements,
            "outcomes": {str(key): value for key, value in self.outcomes.items()},
            "rng_state": self.rng.bit_generator.state,
            "probabilities": quantum.probability_map(self.state, self.qubits),
        }

    @classmethod
    def restore_state(cls, checkpoint: dict[str, Any]) -> np.ndarray:
        pairs = checkpoint["state"]
        return np.array([real + 1j * imag for real, imag in pairs], dtype=complex)

    def snapshot(self) -> dict[str, Any]:
        return {
            "qubits": self.qubits,
            "initial_basis": self.basis,
            "seed": self.seed,
            "next_index": self.next_index,
            "total_steps": len(self.steps),
            "finished": self.finished,
            "probabilities": quantum.probability_map(self.state, self.qubits),
            "measurements": self.measurements,
        }

    def step(self) -> dict[str, Any]:
        index = self.next_index
        raw_step = self.steps[index]
        effective, fired, is_conditional = quantum.resolve_step(raw_step, self.outcomes)
        measured = None
        if effective and effective["type"] == quantum.MEASURE_GATE:
            self.state, outcome = quantum.measure(
                self.state, self.qubits, int(effective["target"]), self.rng
            )
            measured = {"target": int(effective["target"]), "outcome": outcome}
            self.measurements.append({"step": index, **measured})
            self.outcomes[int(effective["target"])] = outcome
        elif effective:
            self.state = quantum.apply_gate(self.state, self.qubits, effective)
        self.next_index += 1
        return {
            "executed": {
                "index": index,
                "type": raw_step["type"],
                "measured": measured,
                "condition_fired": fired if is_conditional else None,
            },
            **self.snapshot(),
        }


def _run_branch(checkpoint: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Continue from a checkpoint through an arbitrary suffix on a forked RNG."""
    base = Session.from_checkpoint(checkpoint)
    qubits = base.qubits
    state = base.state.copy()
    rng = np.random.default_rng()
    rng.bit_generator.state = json.loads(json.dumps(base.rng.bit_generator.state))
    outcomes = dict(base.outcomes)
    measurements = list(base.measurements)
    trace: list[dict[str, Any]] = []
    for index, raw_step in enumerate(steps):
        effective, fired, is_conditional = quantum.resolve_step(raw_step, outcomes)
        measured = None
        if effective and effective["type"] == quantum.MEASURE_GATE:
            state, outcome = quantum.measure(
                state, qubits, int(effective["target"]), rng
            )
            measured = {"target": int(effective["target"]), "outcome": outcome}
            outcomes[int(effective["target"])] = outcome
            measurements.append({"step": index, **measured})
        elif effective:
            state = quantum.apply_gate(state, qubits, effective)
        trace.append(
            {
                "index": index,
                "type": raw_step["type"],
                "measured": measured,
                "condition_fired": fired if is_conditional else None,
                "probabilities": quantum.probability_map(state, qubits),
            }
        )
    return {
        "steps": steps,
        "measurements": measurements,
        "final_probabilities": quantum.probability_map(state, qubits),
        "trace": trace,
    }


SESSIONS: dict[str, Session] = {}
SESSIONS_LOCK = threading.Lock()
STORAGE_LOCK = threading.Lock()
SESSION_LOCKS: dict[str, threading.Lock] = {}


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
        if (
            path.startswith("/api/sessions/")
            and not path.endswith("/step")
            and not path.endswith("/checkpoint")
        ):
            session_id = path.rsplit("/", 1)[1]
            with SESSIONS_LOCK:
                session = SESSIONS.get(session_id)
            if session is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "session not found"})
                return
            with session.lock:
                self._send_json(HTTPStatus.OK, {"session_id": session_id, **session.snapshot()})
            return
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
        try:
            self._route_post(path, payload)
        except (KeyError, TypeError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid request: {exc}"})

    def _route_post(self, path: str, payload: dict[str, Any]) -> None:
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
            first = session.step()
            self._send_json(HTTPStatus.CREATED, {"session_id": session_id, **first})
            return

        if path.startswith("/api/sessions/") and path.endswith("/checkpoint"):
            parts = path.split("/")
            session_id = parts[3] if len(parts) == 5 else ""
            with SESSIONS_LOCK:
                session = SESSIONS.get(session_id)
            if session is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "session not found"})
                return
            with session.lock:
                checkpoint = session.checkpoint(payload.get("label"))
            persist = payload.get("persist", False)
            if not isinstance(persist, bool):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "persist must be a boolean"})
                return
            checkpoint_id = None
            if persist:
                checkpoint_id = uuid.uuid4().hex
                with STORAGE_LOCK:
                    stored = _load_checkpoints()
                    stored["checkpoints"][checkpoint_id] = {"checkpoint_id": checkpoint_id, **checkpoint}
                    _save_checkpoints(stored)
            self._send_json(
                HTTPStatus.OK,
                {"session_id": session_id, "checkpoint_id": checkpoint_id, **checkpoint},
            )
            return

        if path == "/api/sessions/resume":
            checkpoint = self._lookup_checkpoint(payload.get("checkpoint_id"), payload.get("checkpoint"))
            if checkpoint is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "checkpoint not found"})
                return
            session = Session.from_checkpoint(checkpoint)
            new_id = uuid.uuid4().hex
            with SESSIONS_LOCK:
                SESSIONS[new_id] = session
            self._send_json(
                HTTPStatus.CREATED,
                {
                    "session_id": new_id,
                    "resumed_from": payload.get("checkpoint_id"),
                    "restored_measurements": session.measurements,
                    **session.snapshot(),
                },
            )
            return

        if path == "/api/checkpoints/compare":
            checkpoint = self._lookup_checkpoint(payload.get("checkpoint_id"), payload.get("checkpoint"))
            if checkpoint is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "checkpoint not found"})
                return
            branches = payload.get("branches")
            if not isinstance(branches, list) or len(branches) != 2:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "branches must be a list of exactly two step lists"},
                )
                return
            validated: list[list[dict[str, Any]]] = []
            prefix = list(checkpoint["steps"])[: int(checkpoint["next_index"])]
            for branch in branches:
                if not isinstance(branch, list) or not all(isinstance(step, dict) for step in branch):
                    self._send_json(
                        HTTPStatus.BAD_REQUEST,
                        {"error": "each branch must be a list of step objects"},
                    )
                    return
                errors = quantum.validate_circuit(
                    {
                        "qubits": int(checkpoint["qubits"]),
                        "initial_basis": checkpoint["initial_basis"],
                        "steps": prefix + branch,
                    }
                )
                if errors:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"valid": False, "errors": errors})
                    return
                validated.append(branch)
            results = [_run_branch(checkpoint, branch) for branch in validated]
            self._send_json(
                HTTPStatus.OK,
                {
                    "checkpoint_id": payload.get("checkpoint_id"),
                    "labels": payload.get("labels", ["分支 A", "分支 B"]),
                    "branches": results,
                },
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
                    {"error": "session already finished", "session_id": session_id, **session.snapshot()},
                )
                return
            with session.lock:
                result = session.step()
            self._send_json(HTTPStatus.OK, {"session_id": session_id, **result})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _lookup_checkpoint(
        self, checkpoint_id: Any, inline: Any
    ) -> dict[str, Any] | None:
        if isinstance(inline, dict):
            return inline
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            return None
        with STORAGE_LOCK:
            return _load_checkpoints()["checkpoints"].get(checkpoint_id)

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
