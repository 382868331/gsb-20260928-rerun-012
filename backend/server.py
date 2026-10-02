"""Local HTTP API for the quantum experiment bench.

Run:  .venv/bin/python -m backend.server  (from the workspace root)
No third-party web framework is required; numpy is the only dependency.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from urllib.parse import parse_qs, urlparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from . import adaptive
from . import branches as branch_store
from . import quantum

HOST = "127.0.0.1"
PORT = 8000
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CIRCUITS_FILE = DATA_DIR / "circuits.json"
BRANCH_DIR = DATA_DIR / "branch-experiments"
STUDY_DIR = DATA_DIR / "adaptive-studies"


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


def _branch_specs_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("branches"), list):
        specs = payload["branches"]
    else:
        specs = [payload.get("branch_a"), payload.get("branch_b")]
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(specs):
        if not isinstance(raw, dict) or not isinstance(raw.get("steps"), list):
            raise ValueError(f"branches[{index}] must be an object with a steps list")
        normalized.append(
            {
                "branch_id": str(raw.get("branch_id") or f"branch-{index + 1}"),
                "label": raw.get("label") or f"分支 {index + 1}",
                "steps": raw["steps"],
                "filters": raw.get("filters"),
                "budget": raw.get("budget", int(payload.get("shots", 0))),
            }
        )
    if not 1 <= len(normalized) <= 8:
        raise ValueError("branches must contain between 1 and 8 entries")
    return normalized


def _branch_storage_dir(payload: dict[str, Any]) -> Path:
    raw = payload.get("storage_dir")
    if raw is None:
        return BRANCH_DIR
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("storage_dir must be a non-empty path string")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError("storage_dir must be an absolute path")
    return path


def _load_branch_experiment(experiment_id: str, storage_dir: str | Path | None = None) -> branch_store.BranchExperiment:
    if storage_dir is not None:
        return branch_store.BranchExperiment.load(Path(storage_dir) / f"{experiment_id}.json")
    with SESSIONS_LOCK:
        cached = BRANCH_EXPERIMENTS.get(experiment_id)
    if cached is not None:
        return cached
    return branch_store.BranchExperiment.load(BRANCH_DIR / f"{experiment_id}.json")


BRANCH_EXPERIMENTS: dict[str, branch_store.BranchExperiment] = {}


def _load_study(study_id: str, storage_dir: str | Path | None = None) -> adaptive.AdaptiveStudy:
    directory = Path(storage_dir) if storage_dir is not None else STUDY_DIR
    return adaptive.AdaptiveStudy.load(directory / f"{study_id}.json")


def _study_experiment(study: adaptive.AdaptiveStudy) -> branch_store.BranchExperiment:
    return _load_branch_experiment(study.experiment_id, Path(study.experiment_storage_dir))


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
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if path == "/api/health":
                self._send_json(HTTPStatus.OK, {"ok": True, "max_qubits": quantum.MAX_QUBITS})
                return
            if path == "/api/circuits":
                with STORAGE_LOCK:
                    self._send_json(HTTPStatus.OK, _load_circuits())
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

            if path.startswith("/api/branch-experiments/"):
                parts = path.strip("/").split("/")
                if len(parts) == 3:
                    experiment_id = parts[2]
                    raw_location = parse_qs(parsed.query).get("storage_dir", [""])[0]
                    storage_dir = Path(raw_location).expanduser() if raw_location else None
                    if storage_dir is not None and not storage_dir.is_absolute():
                        raise ValueError("storage_dir must be an absolute path")
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                if not experiment_id.replace("-", "").isalnum():
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid experiment id"})
                    return
                try:
                    experiment = _load_branch_experiment(experiment_id, storage_dir)
                except ValueError:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "storage_dir must be an absolute path"})
                    return
                except (OSError, KeyError, json.JSONDecodeError):
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "branch experiment not found"})
                    return
                self._send_json(HTTPStatus.OK, experiment.snapshot())
                return

            if path.startswith("/api/studies/"):
                parts = path.strip("/").split("/")
                if len(parts) != 3:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                study_id = parts[2]
                if not study_id.replace("-", "").isalnum():
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid study id"})
                    return
                raw_location = parse_qs(parsed.query).get("storage_dir", [""])[0]
                study_dir = Path(raw_location).expanduser() if raw_location else None
                if study_dir is not None and not study_dir.is_absolute():
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "storage_dir must be an absolute path"})
                    return
                try:
                    study = _load_study(study_id, study_dir)
                    experiment = _study_experiment(study)
                except (OSError, KeyError, json.JSONDecodeError):
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "study not found"})
                    return
                self._send_json(HTTPStatus.OK, study.snapshot(experiment))
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

            if path.startswith("/api/checkpoints/") and path.endswith("/branch-experiments"):
                parts = path.split("/")
                checkpoint_id = parts[3] if len(parts) == 5 else ""
                with SESSIONS_LOCK:
                    checkpoint = CHECKPOINTS.get(checkpoint_id)
                if checkpoint is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "checkpoint not found"})
                    return
                try:
                    specs = _branch_specs_from_payload(payload)
                    storage_dir = _branch_storage_dir(payload)
                    seed = payload.get("seed")
                    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
                        raise ValueError("seed must be an integer or null")
                    name = payload.get("name", "conditional-branches")
                    if not isinstance(name, str) or not name.strip():
                        raise ValueError("name must be a non-empty string")
                    for spec in specs:
                        if not 0 <= int(spec["budget"]) <= 100_000:
                            raise ValueError("each branch budget must be in [0, 100000]")
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                try:
                    experiment = branch_store.BranchExperiment.create(
                        checkpoint,
                        specs,
                        storage_dir=storage_dir,
                        base_seed=seed,
                        name=name.strip(),
                    )
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                if storage_dir == BRANCH_DIR:
                    with SESSIONS_LOCK:
                        BRANCH_EXPERIMENTS[experiment.experiment_id] = experiment
                self._send_json(HTTPStatus.CREATED, experiment.snapshot())
                return

            if path.startswith("/api/branch-experiments/") and path.endswith("/studies"):
                parts = path.strip("/").split("/")
                if len(parts) != 4:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                experiment_id = parts[2]
                try:
                    storage_dir = _branch_storage_dir(payload)
                    experiment = _load_branch_experiment(experiment_id, storage_dir)
                    study_storage = payload.get("study_storage_dir")
                    if study_storage is not None:
                        if not isinstance(study_storage, str) or not Path(study_storage).is_absolute():
                            raise ValueError("study_storage_dir must be an absolute path string")
                        study_dir = Path(study_storage)
                    else:
                        study_dir = STUDY_DIR
                    name = payload.get("name")
                    if name is not None and not isinstance(name, str):
                        raise ValueError("name must be a string")
                    study = adaptive.AdaptiveStudy.create(
                        experiment, payload, storage_dir=study_dir, name=name
                    )
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                except (OSError, KeyError, json.JSONDecodeError):
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "branch experiment not found"})
                    return
                self._send_json(HTTPStatus.CREATED, study.snapshot(experiment))
                return

            if path.startswith("/api/studies/") and (
                path.endswith("/advance") or path.endswith("/run")
            ):
                parts = path.strip("/").split("/")
                if len(parts) != 4:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                study_id, action = parts[2], parts[3]
                try:
                    raw_dir = payload.get("storage_dir")
                    if raw_dir is not None and (
                        not isinstance(raw_dir, str) or not Path(raw_dir).is_absolute()
                    ):
                        raise ValueError("storage_dir must be an absolute path string")
                    study = _load_study(study_id, raw_dir)
                    experiment = _study_experiment(study)
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                except (OSError, KeyError, json.JSONDecodeError):
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "study not found"})
                    return
                with SESSIONS_LOCK:
                    if action == "run":
                        study.run(experiment)
                    else:
                        study.advance(experiment)
                self._send_json(HTTPStatus.OK, study.snapshot(experiment))
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

    def do_PATCH(self) -> None:
        try:
            payload = self._read_json()
            if payload is None:
                return
            parts = self.path.strip("/").split("/")
            expected = ["api", "branch-experiments", None, "branches", None, "shots"]
            if len(parts) != len(expected) or any(value is not None and item != value for item, value in zip(parts, expected)):
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            experiment_id, branch_id = parts[2], parts[4]
            shots = payload.get("shots")
            if not isinstance(shots, int) or isinstance(shots, bool) or not 1 <= shots <= 100_000:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "shots must be an integer in [1, 100000]"})
                return
            source = payload.get("source", "continued")
            if not isinstance(source, str) or not source.strip():
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "source must be a non-empty string"})
                return
            try:
                storage_dir = _branch_storage_dir(payload)
                experiment = _load_branch_experiment(experiment_id, storage_dir)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except (OSError, KeyError, json.JSONDecodeError):
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "branch experiment not found"})
                return
            try:
                with SESSIONS_LOCK:
                    batch = experiment.add_shots(branch_id, shots, source=source.strip())
            except KeyError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, {"batch": batch, **experiment.snapshot()})
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"server error: {exc}"})


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
