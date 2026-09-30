"""Persistent, independently sampled conditional continuation branches."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from . import quantum


def _jsonable(value: Any) -> Any:
    if isinstance(value, complex):
        return ["__complex__", value.real, value.imag]
    if isinstance(value, np.ndarray):
        return ["__ndarray__", str(value.dtype), [_jsonable(item) for item in value.tolist()]]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _from_json(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 3 and value[0] == "__complex__":
        return complex(value[1], value[2])
    if isinstance(value, list) and len(value) == 3 and value[0] == "__ndarray__":
        return np.array(_from_json(value[2]), dtype=value[1])
    if isinstance(value, dict):
        return {key: _from_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_json(item) for item in value]
    return value


def rng_to_json(rng: np.random.Generator) -> Any:
    return _jsonable(rng.bit_generator.state)


def rng_from_json(data: Any) -> np.random.Generator:
    rng = np.random.default_rng()
    rng.bit_generator.state = _from_json(data)
    return rng


def _branch_seed(base_seed: int, branch_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}\0{branch_id}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % (2**63)


class BranchExperiment:
    """Statistics for several conditional branches sharing one prefix."""

    def __init__(
        self,
        *,
        experiment_id: str,
        name: str,
        base_seed: int,
        prefix: dict[str, Any],
        branches: dict[str, dict[str, Any]],
        branch_order: list[str],
        created_at: float,
        storage_dir: Path,
    ) -> None:
        self.experiment_id = experiment_id
        self.name = name
        self.base_seed = base_seed
        self.prefix = prefix
        self.branches = branches
        self.branch_order = branch_order
        self.created_at = created_at
        self.storage_dir = Path(storage_dir)

    @property
    def path(self) -> Path:
        return self.storage_dir / f"{self.experiment_id}.json"

    @classmethod
    def create(
        cls,
        checkpoint: Any,
        branch_specs: list[dict[str, Any]],
        *,
        storage_dir: str | Path,
        base_seed: int | None = None,
        name: str = "conditional-branches",
        experiment_id: str | None = None,
        created_at: float | None = None,
    ) -> "BranchExperiment":
        labels = [str(spec.get("label") or f"branch-{index + 1}") for index, spec in enumerate(branch_specs)]
        if len(set(labels)) != len(labels):
            raise ValueError("branch labels must be unique")

        resolved_seed = int(base_seed) if base_seed is not None else int.from_bytes(uuid.uuid4().bytes, "big") % (2**63)
        experiment_id = experiment_id or uuid.uuid4().hex
        labels_for_qubits = quantum.probability_map(checkpoint.state, checkpoint.qubits).keys()
        branches: dict[str, dict[str, Any]] = {}
        branch_order: list[str] = []

        for index, spec in enumerate(branch_specs):
            steps = list(spec["steps"])
            label = labels[index]
            branch_id = str(spec.get("branch_id") or f"branch-{index + 1}")
            if branch_id in branches:
                raise ValueError("branch ids must be unique")
            errors = quantum.validate_circuit(
                {
                    "qubits": checkpoint.qubits,
                    "initial_basis": checkpoint.basis,
                    "steps": steps,
                }
            )
            if errors:
                raise ValueError(json.dumps(errors, ensure_ascii=False))
            budget = int(spec.get("budget", 0))
            if budget < 0:
                raise ValueError("branch budget must be non-negative")
            rng = np.random.default_rng(_branch_seed(resolved_seed, branch_id))
            branches[branch_id] = {
                "branch_id": branch_id,
                "label": label,
                "steps": steps,
                "budget": budget,
                "completed_shots": 0,
                "counts": {label: 0 for label in labels_for_qubits},
                "rng_state": rng_to_json(rng),
                "batches": [],
            }
            branch_order.append(branch_id)

        prefix = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "qubits": checkpoint.qubits,
            "initial_basis": checkpoint.basis,
            "prefix_seed": checkpoint.seed,
            "prefix_steps": list(checkpoint.steps),
            "prefix_next_index": checkpoint.next_index,
            "prefix_measurements": list(checkpoint.measurements),
            "state": _jsonable(checkpoint.state),
            "prefix_rng_state": None if checkpoint.rng_state is None else checkpoint.rng_state,
        }
        experiment = cls(
            experiment_id=experiment_id,
            name=name,
            base_seed=resolved_seed,
            prefix=prefix,
            branches=branches,
            branch_order=branch_order,
            created_at=created_at if created_at is not None else time.time(),
            storage_dir=Path(storage_dir),
        )
        for branch_id, branch in experiment.branches.items():
            if branch["budget"]:
                shots = int(branch["budget"])
                state = _from_json(experiment.prefix["state"])
                rng = rng_from_json(branch["rng_state"])
                for _ in range(shots):
                    result = quantum.sample_continuation(
                        state, experiment.prefix["qubits"], branch["steps"], rng
                    )
                    branch["counts"][result["sample"]] += 1
                branch["completed_shots"] = shots
                branch["rng_state"] = rng_to_json(rng)
                branch["batches"].append(
                    {
                        "batch_id": uuid.uuid4().hex,
                        "source": "initial-budget",
                        "shots": shots,
                        "total_shots_after_batch": shots,
                    }
                )
        experiment.save()
        return experiment

    def add_shots(self, branch_id: str, shots: int, *, source: str = "continued") -> dict[str, Any]:
        if branch_id not in self.branches:
            raise KeyError(f"unknown branch: {branch_id}")
        shots = int(shots)
        if shots <= 0:
            raise ValueError("shots must be positive")
        branch = self.branches[branch_id]
        state = _from_json(self.prefix["state"])
        rng = rng_from_json(branch["rng_state"])
        for _ in range(shots):
            result = quantum.sample_continuation(
                state, self.prefix["qubits"], branch["steps"], rng
            )
            branch["counts"][result["sample"]] += 1
        branch["completed_shots"] += shots
        branch["budget"] = max(int(branch["budget"]), branch["completed_shots"])
        branch["rng_state"] = rng_to_json(rng)
        batch = {
            "batch_id": uuid.uuid4().hex,
            "source": source,
            "shots": shots,
            "total_shots_after_batch": branch["completed_shots"],
        }
        branch["batches"].append(batch)
        self.save()
        return batch

    def save(self) -> Path:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        target = self.path
        temporary = target.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
        temporary.replace(target)
        return target

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "base_seed": self.base_seed,
            "created_at": self.created_at,
            "prefix": self.prefix,
            "branch_order": list(self.branch_order),
            "branches": {key: _jsonable(value) for key, value in self.branches.items()},
        }

    def snapshot(self) -> dict[str, Any]:
        branches = []
        for branch_id in self.branch_order:
            branch = self.branches[branch_id]
            shots = int(branch["completed_shots"])
            frequencies = {
                label: (count / shots if shots else 0.0)
                for label, count in branch["counts"].items()
            }
            branches.append(
                {
                    "branch_id": branch_id,
                    "label": branch["label"],
                    "condition_steps": branch["steps"],
                    "budget": branch["budget"],
                    "completed_shots": shots,
                    "counts": dict(branch["counts"]),
                    "frequencies": frequencies,
                    "batches": list(branch["batches"]),
                }
            )
        prefix = dict(self.prefix)
        prefix.pop("state", None)
        prefix.pop("prefix_rng_state", None)
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "base_seed": self.base_seed,
            "created_at": self.created_at,
            "storage_path": str(self.path),
            "pooled_across_conditions": False,
            "common_prefix": prefix,
            "branches": branches,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], storage_dir: str | Path) -> "BranchExperiment":
        return cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            base_seed=int(data["base_seed"]),
            prefix=data["prefix"],
            branches=data["branches"],
            branch_order=list(data["branch_order"]),
            created_at=float(data["created_at"]),
            storage_dir=Path(storage_dir),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BranchExperiment":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle), path.parent)
