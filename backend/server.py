"""Local HTTP API for the quantum experiment bench.

Run:  .venv/bin/python -m backend.server  (from the workspace root)
No third-party web framework is required; numpy is the only dependency.
"""

from __future__ import annotations

import json
import hashlib
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
BRANCH_RUNS_DIR = DATA_DIR / "branch_experiments"
MAX_BRANCHES = 20
MAX_BRANCH_SHOTS = 100_000


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


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _branch_identity(label: str, steps: list[dict[str, Any]], explicit_id: Any) -> str:
    if isinstance(explicit_id, str) and explicit_id.strip():
        branch_id = explicit_id.strip()
        if len(branch_id) <= 80 and all(ch.isalnum() or ch in "_-" for ch in branch_id):
            return branch_id
    digest = hashlib.sha256(_canonical_json({"label": label, "steps": steps})).hexdigest()[:16]
    return f"branch-{digest}"


def _branch_rng(root_seed: int, checkpoint_id: str, branch_id: str) -> np.random.Generator:
    identity = hashlib.sha256(f"{checkpoint_id}\n{branch_id}".encode("utf-8")).digest()[:16]
    seed_sequence = np.random.SeedSequence([int(root_seed), int.from_bytes(identity, "little")])
    return np.random.default_rng(seed_sequence)


class BranchExperiment:
    """Independent conditional continuations sharing one saved prefix."""

    def __init__(
        self,
        *,
        run_id: str,
        name: str,
        root_seed: int,
        budget: int,
        checkpoint_id: str,
        checkpoint_name: str,
        qubits: int,
        basis: str,
        prefix_steps: list[dict[str, Any]],
        prefix_next_index: int,
        prefix_state: np.ndarray,
        prefix_measurements: list[dict[str, Any]],
        branches: list[dict[str, Any]],
        created_at: float,
        updated_at: float,
    ) -> None:
        self.run_id = run_id
        self.name = name
        self.root_seed = root_seed
        self.budget = budget
        self.checkpoint_id = checkpoint_id
        self.checkpoint_name = checkpoint_name
        self.qubits = qubits
        self.basis = basis
        self.prefix_steps = prefix_steps
        self.prefix_next_index = prefix_next_index
        self.prefix_state = prefix_state
        self.prefix_measurements = prefix_measurements
        self.branches = branches
        self.created_at = created_at
        self.updated_at = updated_at

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Checkpoint,
        specs: list[tuple[str, str, list[dict[str, Any]]]],
        *,
        name: str,
        shots: int,
        budget: int,
        root_seed: int,
        run_id: str | None = None,
    ) -> "BranchExperiment":
        labels = [label for _, label, _ in specs]
        if len(labels) != len(set(labels)):
            raise ValueError("branch labels must be unique")
        now = time.time()
        branches = []
        for explicit_id, label, steps in specs:
            branch_id = _branch_identity(label, steps, explicit_id)
            if any(item["branch_id"] == branch_id for item in branches):
                raise ValueError("branch identities must be unique")
            rng = _branch_rng(root_seed, checkpoint.checkpoint_id, branch_id)
            branches.append(
                {
                    "branch_id": branch_id,
                    "label": label,
                    "steps": steps,
                    "shots": 0,
                    "counts": {label: 0 for label in quantum._basis_labels(checkpoint.qubits)},
                    "rng_state": _rng_to_json(rng),
                }
            )
        run = cls(
            run_id=run_id or uuid.uuid4().hex,
            name=name,
            root_seed=root_seed,
            budget=budget,
            checkpoint_id=checkpoint.checkpoint_id,
            checkpoint_name=checkpoint.name,
            qubits=checkpoint.qubits,
            basis=checkpoint.basis,
            prefix_steps=list(checkpoint.steps),
            prefix_next_index=checkpoint.next_index,
            prefix_state=checkpoint.state.copy(),
            prefix_measurements=list(checkpoint.measurements),
            branches=branches,
            created_at=now,
            updated_at=now,
        )
        if shots:
            run.add_shots(shots)
        return run

    def _branch_result(self, branch: dict[str, Any]) -> dict[str, Any]:
        shots = int(branch["shots"])
        frequencies = {
            label: (count / shots if shots else 0.0) for label, count in branch["counts"].items()
        }
        return {
            "branch_id": branch["branch_id"],
            "label": branch["label"],
            "condition_steps": branch["steps"],
            "shots": shots,
            "counts": dict(branch["counts"]),
            "frequencies": frequencies,
            "sample_source": f"branch-experiment:{self.run_id}:{branch['branch_id']}",
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "budget": self.budget,
            "root_seed": self.root_seed,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "checkpoint": {
                "checkpoint_id": self.checkpoint_id,
                "name": self.checkpoint_name,
                "qubits": self.qubits,
                "initial_basis": self.basis,
                "prefix_steps": self.prefix_steps,
                "prefix_next_index": self.prefix_next_index,
                "prefix_probabilities": quantum.probability_map(self.prefix_state, self.qubits),
                "measurements": self.prefix_measurements,
            },
            "branches": [self._branch_result(branch) for branch in self.branches],
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "root_seed": self.root_seed,
            "budget": self.budget,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_name": self.checkpoint_name,
            "qubits": self.qubits,
            "basis": self.basis,
            "prefix_steps": self.prefix_steps,
            "prefix_next_index": self.prefix_next_index,
            "prefix_state": _to_jsonable(self.prefix_state),
            "prefix_measurements": self.prefix_measurements,
            "branches": self.branches,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "BranchExperiment":
        return cls(
            run_id=str(data["run_id"]),
            name=str(data["name"]),
            root_seed=int(data["root_seed"]),
            budget=int(data["budget"]),
            checkpoint_id=str(data["checkpoint_id"]),
            checkpoint_name=str(data["checkpoint_name"]),
            qubits=int(data["qubits"]),
            basis=str(data["basis"]),
            prefix_steps=_from_jsonable(data["prefix_steps"]),
            prefix_next_index=int(data["prefix_next_index"]),
            prefix_state=_from_jsonable(data["prefix_state"]),
            prefix_measurements=_from_jsonable(data["prefix_measurements"]),
            branches=_from_jsonable(data["branches"]),
            created_at=float(data["created_at"]),
            updated_at=float(data["updated_at"]),
        )

    def add_shots(self, additional_shots: int, *, budget: int | None = None) -> None:
        if budget is not None:
            if budget < self.budget:
                raise ValueError("budget cannot be decreased")
            self.budget = budget
        if not isinstance(additional_shots, int) or additional_shots <= 0:
            raise ValueError("additional_shots must be a positive integer")
        current_shots = max(int(branch["shots"]) for branch in self.branches)
        if current_shots + additional_shots > self.budget:
            raise ValueError("requested shots exceed branch budget")
        for branch in self.branches:
            rng = _rng_from_json(branch["rng_state"])
            for _ in range(additional_shots):
                state = self.prefix_state.copy()
                for step in branch["steps"]:
                    if step["type"] == quantum.MEASURE_GATE:
                        state, _ = quantum.measure(
                            state, self.qubits, int(step["target"]), rng
                        )
                    else:
                        state = quantum.apply_gate(state, self.qubits, step)
                probs = quantum.probabilities(state)
                index = int(rng.choice(len(probs), p=probs))
                label = format(index, f"0{self.qubits}b")
                branch["counts"][label] += 1
            branch["shots"] = int(branch["shots"]) + additional_shots
            branch["rng_state"] = _rng_to_json(rng)
        self.updated_at = time.time()


BRANCH_RUNS: dict[str, BranchExperiment] = {}


def _branch_run_path(run_id: str) -> Path:
    if not run_id or any(ch in run_id for ch in "/\\") or not all(ch.isalnum() or ch in "_-" for ch in run_id):
        raise ValueError("invalid branch run id")
    return BRANCH_RUNS_DIR / f"{run_id}.json"


def save_branch_experiment(run: BranchExperiment) -> Path:
    BRANCH_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = _branch_run_path(run.run_id)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(run.to_json(), handle, ensure_ascii=False, indent=2)
    tmp.replace(path)
    BRANCH_RUNS[run.run_id] = run
    return path


def load_branch_experiment(run_id: str) -> BranchExperiment:
    if run_id in BRANCH_RUNS:
        return BRANCH_RUNS[run_id]
    path = _branch_run_path(run_id)
    with path.open("r", encoding="utf-8") as handle:
        run = BranchExperiment.from_json(json.load(handle))
    BRANCH_RUNS[run_id] = run
    return run


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, complex):
        return ["__complex__", value.real, value.imag]
    if isinstance(value, np.ndarray):
        data = value.tolist()
        if not value.shape:
            data = _to_jsonable(data)
        else:
            data = [_to_jsonable(item) for item in data]
        return ["__ndarray__", str(value.dtype), list(value.shape), data]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _from_jsonable(value: Any) -> Any:
    if isinstance(value, list) and len(value) >= 2 and value[0] == "__complex__":
        return complex(float(value[1]), float(value[2]))
    if isinstance(value, list) and len(value) == 4 and value[0] == "__ndarray__":
        array = np.asarray(_from_jsonable(value[3]), dtype=value[1])
        return array.reshape(tuple(value[2]))
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
                parts = path.split("/")
                checkpoint_id = parts[3] if len(parts) >= 4 else ""
                with SESSIONS_LOCK:
                    checkpoint = CHECKPOINTS.get(checkpoint_id)
                if checkpoint is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "checkpoint not found"})
                else:
                    self._send_json(HTTPStatus.OK, checkpoint.snapshot())
                return
            if path.startswith("/api/branch-runs/"):
                run_id = path.rsplit("/", 1)[1]
                try:
                    run = load_branch_experiment(run_id)
                except (FileNotFoundError, ValueError, KeyError):
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "branch experiment not found"})
                else:
                    self._send_json(HTTPStatus.OK, run.snapshot())
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

            if path.startswith("/api/checkpoints/") and path.endswith("/branch-runs"):
                parts = path.split("/")
                checkpoint_id = parts[3] if len(parts) == 5 else ""
                with SESSIONS_LOCK:
                    checkpoint = CHECKPOINTS.get(checkpoint_id)
                if checkpoint is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "checkpoint not found"})
                    return
                parsed = self._parse_branch_specs(checkpoint, payload)
                if not isinstance(parsed, dict) or "specs" not in parsed:
                    self._send_json(parsed[0], parsed[1])
                    return
                specs = parsed["specs"]
                shots = parsed["shots"]
                name = parsed["name"]
                budget = parsed["budget"]
                seed = payload.get("seed")
                if isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed <= 2**63 - 1:
                    root_seed = seed
                else:
                    root_seed = int.from_bytes(uuid.uuid4().bytes, "little") // 2
                try:
                    run = BranchExperiment.from_checkpoint(
                        checkpoint,
                        specs,
                        name=name,
                        shots=shots,
                        budget=budget,
                        root_seed=root_seed,
                    )
                    save_branch_experiment(run)
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._send_json(HTTPStatus.CREATED, run.snapshot())
                return

            if path.startswith("/api/branch-runs/") and path.endswith("/shots"):
                parts = path.split("/")
                run_id = parts[3] if len(parts) == 5 else ""
                try:
                    run = load_branch_experiment(run_id)
                except (FileNotFoundError, ValueError, KeyError):
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "branch experiment not found"})
                    return
                additional_shots = payload.get("shots")
                if not isinstance(additional_shots, int) or isinstance(additional_shots, bool) or additional_shots <= 0:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "shots must be a positive integer"})
                    return
                budget = payload.get("budget", run.budget)
                if not isinstance(budget, int) or isinstance(budget, bool) or budget < run.budget or budget > MAX_BRANCH_SHOTS:
                    self._send_json(
                        HTTPStatus.BAD_REQUEST,
                        {"error": f"budget must be an integer in [{run.budget}, {MAX_BRANCH_SHOTS}]"},
                    )
                    return
                try:
                    run.add_shots(additional_shots, budget=budget)
                    save_branch_experiment(run)
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._send_json(HTTPStatus.OK, run.snapshot())
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

    def _parse_branch_specs(
        self, checkpoint: "Checkpoint", payload: dict[str, Any]
    ) -> dict[str, Any] | tuple[HTTPStatus, dict[str, Any]]:
        raw_branches = payload.get("branches")
        if not isinstance(raw_branches, list) or not 1 <= len(raw_branches) <= MAX_BRANCHES:
            return (
                HTTPStatus.BAD_REQUEST,
                {"error": f"branches must be a list with 1-{MAX_BRANCHES} items"},
            )
        specs: list[tuple[str, str, list[dict[str, Any]]]] = []
        labels: set[str] = set()
        for index, raw in enumerate(raw_branches):
            if not isinstance(raw, dict) or not isinstance(raw.get("steps"), list):
                return HTTPStatus.BAD_REQUEST, {"error": f"branches[{index}] must contain a steps list"}
            errors = quantum.validate_circuit(
                {"qubits": checkpoint.qubits, "initial_basis": checkpoint.basis, "steps": raw["steps"]}
            )
            if errors:
                return HTTPStatus.BAD_REQUEST, {"valid": False, "errors": errors}
            label = raw.get("label")
            if not isinstance(label, str) or not label.strip():
                label = chr(ord("A") + index) if index < 26 else f"branch-{index + 1}"
            label = label.strip()
            if label in labels:
                return HTTPStatus.BAD_REQUEST, {"error": "branch labels must be unique"}
            labels.add(label)
            explicit_id = raw.get("branch_id")
            specs.append((explicit_id, label, list(raw["steps"])))

        shots = payload.get("shots", 0)
        budget = payload.get("budget", max(shots, 1024))
        for key, value in (("shots", shots), ("budget", budget)):
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= MAX_BRANCH_SHOTS:
                return HTTPStatus.BAD_REQUEST, {"error": f"{key} must be an integer in [0, {MAX_BRANCH_SHOTS}]"}
        if shots > budget:
            return HTTPStatus.BAD_REQUEST, {"error": "shots cannot exceed budget"}
        raw_name = payload.get("name")
        name = str(raw_name).strip() if isinstance(raw_name, str) and raw_name.strip() else f"checkpoint-{checkpoint.next_index}"
        return {"specs": specs, "shots": shots, "name": name, "budget": budget}

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
