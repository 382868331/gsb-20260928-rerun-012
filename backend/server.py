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

    def __init__(
        self,
        payload: dict[str, Any],
        *,
        state: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        next_index: int = 0,
        measurements: list[dict[str, Any]] | None = None,
    ):
        self.qubits = int(payload["qubits"])
        self.basis = payload["initial_basis"]
        self.steps = list(payload["steps"])
        self.seed = payload.get("seed")
        self.rng = rng if rng is not None else np.random.default_rng(
            self.seed if self.seed is not None else None
        )
        self.state = (
            state if state is not None else quantum.initial_state(self.qubits, self.basis)
        )
        self.next_index = next_index
        self.measurements = list(measurements) if measurements is not None else []

    @property
    def finished(self) -> bool:
        return self.next_index >= len(self.steps)

    @classmethod
    def from_checkpoint(cls, checkpoint: "Checkpoint", steps: list[dict[str, Any]] | None = None) -> "Session":
        """Resume a session from a saved mid-circuit state without re-drawing.

        The collapsed state vector and the RNG bit-generator state are restored,
        so measurements already performed at save time are never sampled again.
        """
        payload = {
            "qubits": checkpoint.qubits,
            "initial_basis": checkpoint.basis,
            "seed": checkpoint.seed,
            "steps": steps if steps is not None else checkpoint.steps,
        }
        session = cls(
            payload,
            state=checkpoint.state.copy(),
            rng=np.random.default_rng() if checkpoint.rng_state is None else _rng_from_json(checkpoint.rng_state),
            next_index=checkpoint.next_index if steps is None else 0,
            measurements=list(checkpoint.measurements),
        )
        return session

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

    def save_checkpoint(self, name: str, checkpoint_id: str) -> "Checkpoint":
        return Checkpoint(
            checkpoint_id=checkpoint_id,
            session_id=getattr(self, "session_id", None),
            name=name,
            qubits=self.qubits,
            basis=self.basis,
            steps=list(self.steps),
            next_index=self.next_index,
            state=self.state.copy(),
            rng_state=None if self.seed is None else _rng_to_json(self.rng),
            seed=self.seed,
            measurements=list(self.measurements),
        )

    def step(self) -> dict[str, Any]:
        if self.finished:
            return {"notice": "线路已执行完毕，没有更多步骤", **self.snapshot()}
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


class Checkpoint:
    """Mid-circuit save point: collapsed state vector plus RNG state."""

    def __init__(
        self,
        *,
        checkpoint_id: str,
        session_id: str | None,
        name: str,
        qubits: int,
        basis: str,
        steps: list[dict[str, Any]],
        next_index: int,
        state: np.ndarray,
        rng_state: Any,
        seed: int | None,
        measurements: list[dict[str, Any]],
    ) -> None:
        self.checkpoint_id = checkpoint_id
        self.session_id = session_id
        self.name = name
        self.qubits = qubits
        self.basis = basis
        self.steps = steps
        self.next_index = next_index
        self.state = state
        self.rng_state = rng_state
        self.seed = seed
        self.measurements = measurements

    def snapshot(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "name": self.name,
            "qubits": self.qubits,
            "initial_basis": self.basis,
            "seed": self.seed,
            "next_index": self.next_index,
            "total_steps": len(self.steps),
            "probabilities": quantum.probability_map(self.state, self.qubits),
            "measurements": self.measurements,
        }


CHECKPOINTS: dict[str, Checkpoint] = {}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return ["__ndarray__", str(value.dtype), [_to_jsonable(item) for item in value.tolist()]]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _from_jsonable(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 3 and value[0] == "__ndarray__":
        return np.array(_from_jsonable(value[2]), dtype=value[1])
    if isinstance(value, dict):
        return {key: _from_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_jsonable(item) for item in value]
    return value


def _rng_to_json(rng: np.random.Generator) -> Any:
    return _to_jsonable(rng.bit_generator.state)


def _rng_from_json(data: Any) -> np.random.Generator:
    rng = np.random.default_rng()
    rng.bit_generator.state = _from_jsonable(data)
    return rng


def run_continuation(
    checkpoint: Checkpoint, branch_steps: list[dict[str, Any]]
) -> dict[str, Any]:
    """Run candidate follow-up steps from a checkpoint.

    Each call starts from independent copies of the saved state and RNG, so
    the two compared branches draw the same random sequence without mutating
    the checkpoint or any interactive session.
    """
    state = checkpoint.state.copy()
    rng = (
        np.random.default_rng()
        if checkpoint.rng_state is None
        else _rng_from_json(checkpoint.rng_state)
    )
    trace: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    for index, step in enumerate(branch_steps):
        measured = None
        if step["type"] == quantum.MEASURE_GATE:
            state, outcome = quantum.measure(
                state, checkpoint.qubits, int(step["target"]), rng
            )
            measured = {"target": int(step["target"]), "outcome": outcome}
            measurements.append({"step": index, **measured})
        else:
            state = quantum.apply_gate(state, checkpoint.qubits, step)
        trace.append(
            {
                "index": index,
                "type": step["type"],
                "measured": measured,
                "probabilities": quantum.probability_map(state, checkpoint.qubits),
            }
        )
    return {
        "steps": trace,
        "measurements": measurements,
        "final_probabilities": quantum.probability_map(state, checkpoint.qubits),
    }


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
        try:
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
            if path.startswith("/api/checkpoints/"):
                checkpoint_id = path.rsplit("/", 1)[1]
                with SESSIONS_LOCK:
                    checkpoint = CHECKPOINTS.get(checkpoint_id)
                if checkpoint is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "checkpoint not found"})
                else:
                    self._send_json(HTTPStatus.OK, checkpoint.snapshot())
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except Exception as exc:  # never leak a bare HTML 500 to the UI
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"server error: {exc}"})

    def do_POST(self) -> None:
        try:
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
                session.session_id = session_id
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
                        # Repeated clicks after completion are idempotent: return
                        # the final snapshot with an explicit notice.
                        self._send_json(
                            HTTPStatus.OK,
                            {
                                "session_id": session_id,
                                "notice": "线路已执行完毕，没有更多步骤",
                                **session.snapshot(),
                            },
                        )
                        return
                    result = session.step()
                self._send_json(HTTPStatus.OK, {"session_id": session_id, **result})
                return

            if path.startswith("/api/sessions/") and path.endswith("/checkpoints"):
                parts = path.split("/")
                session_id = parts[3] if len(parts) == 5 else ""
                with SESSIONS_LOCK:
                    session = SESSIONS.get(session_id)
                    if session is None:
                        self._send_json(HTTPStatus.NOT_FOUND, {"error": "session not found"})
                        return
                    raw_name = payload.get("name")
                    name = str(raw_name).strip() if isinstance(raw_name, str) and raw_name.strip() else f"step-{session.next_index}"
                    checkpoint = session.save_checkpoint(name, uuid.uuid4().hex)
                    CHECKPOINTS[checkpoint.checkpoint_id] = checkpoint
                self._send_json(
                    HTTPStatus.CREATED,
                    {"session_id": session_id, **checkpoint.snapshot()},
                )
                return

            if path.startswith("/api/checkpoints/") and path.endswith("/resume"):
                parts = path.split("/")
                checkpoint_id = parts[3] if len(parts) == 5 else ""
                with SESSIONS_LOCK:
                    checkpoint = CHECKPOINTS.get(checkpoint_id)
                if checkpoint is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "checkpoint not found"})
                    return
                session = Session.from_checkpoint(checkpoint)
                session_id = uuid.uuid4().hex
                session.session_id = session_id
                with SESSIONS_LOCK:
                    SESSIONS[session_id] = session
                self._send_json(
                    HTTPStatus.CREATED,
                    {
                        "session_id": session_id,
                        "resumed_from": checkpoint_id,
                        "notice": f"已从保存点「{checkpoint.name}」继续，之前的测量结果保持不变",
                        **session.snapshot(),
                    },
                )
                return

            if path.startswith("/api/checkpoints/") and path.endswith("/compare"):
                parts = path.split("/")
                checkpoint_id = parts[3] if len(parts) == 5 else ""
                with SESSIONS_LOCK:
                    checkpoint = CHECKPOINTS.get(checkpoint_id)
                if checkpoint is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "checkpoint not found"})
                    return
                comparison = self._build_comparison(checkpoint, payload)
                if isinstance(comparison, tuple):
                    status, body = comparison
                    self._send_json(status, body)
                else:
                    self._send_json(HTTPStatus.OK, comparison)
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except Exception as exc:  # never leak a bare HTML 500 to the UI
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"server error: {exc}"})

    def _build_comparison(
        self, checkpoint: "Checkpoint", payload: dict[str, Any]
    ) -> Any:
        branches: list[tuple[str, Any]] = []
        for key, label in (("branch_a", "A"), ("branch_b", "B")):
            raw = payload.get(key)
            if not isinstance(raw, dict) or not isinstance(raw.get("steps"), list):
                return (
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"{key} must be an object with a steps list"},
                )
            branch_circuit = {
                "qubits": checkpoint.qubits,
                "initial_basis": checkpoint.basis,
                "steps": raw["steps"],
            }
            errors = quantum.validate_circuit(branch_circuit)
            if errors:
                return HTTPStatus.BAD_REQUEST, {"valid": False, "errors": errors}
            name = raw.get("label") if isinstance(raw.get("label"), str) else label
            branches.append((name, raw["steps"]))

        results = []
        with SESSIONS_LOCK:
            for name, steps in branches:
                results.append(
                    {
                        "label": name,
                        "checkpoint": checkpoint.snapshot(),
                        **run_continuation(checkpoint, steps),
                    }
                )
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint": checkpoint.snapshot(),
            "branches": results,
        }

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
