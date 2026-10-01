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


def _normalize_filters(raw: Any, qubits: int) -> list[dict[str, int]]:
    if raw is None or raw == []:
        return []
    if not isinstance(raw, list) or not raw:
        raise ValueError("filters must be a non-empty list when provided")
    filters: list[dict[str, int]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"filters[{index}] must be an object")
        target = item.get("target")
        outcome = item.get("outcome")
        if not isinstance(target, int) or isinstance(target, bool) or not 0 <= target < qubits:
            raise ValueError(f"filters[{index}].target must be a qubit index in [0, {qubits - 1}]")
        if not isinstance(outcome, int) or isinstance(outcome, bool) or outcome not in (0, 1):
            raise ValueError(f"filters[{index}].outcome must be 0 or 1")
        filters.append({"target": target, "outcome": int(outcome)})
    return filters


def _validate_filter_positions(
    steps: list[dict[str, Any]], filters: list[dict[str, int]]
) -> None:
    measurement_targets = [
        int(step["target"])
        for step in steps
        if isinstance(step, dict) and step.get("type") == quantum.MEASURE_GATE
    ]
    cursor = 0
    for stage_index, filter_ in enumerate(filters):
        try:
            cursor = measurement_targets.index(filter_["target"], cursor) + 1
        except ValueError:
            raise ValueError(
                f"filters[{stage_index}] has no corresponding MEASURE after the previous filter position"
            )


def _measurement_passes(
    measurements: list[dict[str, Any]], filters: list[dict[str, int]]
) -> tuple[bool, int | None, list[int]]:
    """Match filters chronologically to repeated measurement events."""
    cursor = 0
    matched: list[int] = []
    for stage_index, filter_ in enumerate(filters):
        index = next(
            (
                item_index
                for item_index in range(cursor, len(measurements))
                if measurements[item_index]["target"] == filter_["target"]
            ),
            None,
        )
        if index is None:
            return False, stage_index, matched
        if int(measurements[index]["outcome"]) != filter_["outcome"]:
            return False, stage_index, matched
        matched.append(index)
        cursor = index + 1
    return True, None, matched


def _draw_filtered_sample(
    state: Any,
    qubits: int,
    steps: list[dict[str, Any]],
    filters: list[dict[str, int]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    result = quantum.sample_continuation(state, qubits, steps, rng)
    accepted, _, _ = _measurement_passes(result["measurements"], filters)
    return {"accepted": accepted, **result}


def _run_filtered_budget(
    experiment: "BranchExperiment",
    branch: dict[str, Any],
    accepted_budget: int,
    attempt_cap: int,
) -> dict[str, int]:
    """Rejection-sample until enough accepted paths or the attempt cap is hit."""
    accepted_budget = int(accepted_budget)
    attempt_cap = int(attempt_cap)
    if accepted_budget < 0 or attempt_cap <= 0:
        raise ValueError("accepted budget must be non-negative and attempt cap positive")
    state = _from_json(experiment.prefix["state"])
    qubits = experiment.prefix["qubits"]
    steps = branch["steps"]
    filters = branch["filters"]
    rng = rng_from_json(branch["rng_state"])

    attempts = 0
    accepted = 0
    stage_rejected = [0 for _ in filters]
    while accepted < accepted_budget and attempts < attempt_cap:
        draw = _draw_filtered_sample(state, qubits, steps, filters, rng)
        attempts += 1
        if draw["accepted"]:
            accepted += 1
            branch["counts"][draw["sample"]] += 1
            continue
        _, failed_stage, _ = _measurement_passes(draw["measurements"], filters)
        stage_rejected[failed_stage] += 1

    branch["rng_state"] = rng_to_json(rng)
    return {
        "attempts": attempts,
        "accepted": accepted,
        "rejected": attempts - accepted,
        "stage_rejected": stage_rejected,
        "cap_reached": attempts >= attempt_cap and accepted < accepted_budget,
    }


def _exact_conditioned_probabilities(
    state: Any,
    qubits: int,
    steps: list[dict[str, Any]],
    filters: list[dict[str, int]],
) -> dict[str, Any]:
    paths = quantum.enumerate_continuation(state, qubits, steps)
    joint = {label: 0.0 for label in quantum.probability_map(state, qubits)}
    conditioned = {label: 0.0 for label in joint}
    accepted_mass = 0.0
    for path in paths:
        joint[path["sample"]] += path["probability"]
        accepted, _, _ = _measurement_passes(path["measurements"], filters)
        if accepted:
            conditioned[path["sample"]] += path["probability"]
            accepted_mass += path["probability"]
    if not filters:
        conditional = dict(joint)
        has_evidence = True
    elif accepted_mass > 0.0:
        conditional = {label: mass / accepted_mass for label, mass in conditioned.items()}
        has_evidence = True
    else:
        conditional = None
        has_evidence = False
    return {
        "joint_probabilities": joint,
        "accepted_probability": accepted_mass if filters else 1.0,
        "conditional_probabilities": conditional,
        "has_evidence": has_evidence,
    }


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
            raw_attempt_cap = spec.get("attempt_cap")
            attempt_cap = (
                int(raw_attempt_cap)
                if raw_attempt_cap is not None
                else max(budget * 100, 1000)
            )
            if attempt_cap <= 0:
                raise ValueError("attempt_cap must be positive")
            filters = _normalize_filters(spec.get("filters"), checkpoint.qubits)
            _validate_filter_positions(steps, filters)
            rng = np.random.default_rng(_branch_seed(resolved_seed, branch_id))
            branches[branch_id] = {
                "branch_id": branch_id,
                "label": label,
                "steps": steps,
                "filters": filters,
                "budget": budget,
                "attempt_cap": attempt_cap,
                "total_attempts": 0,
                "completed_shots": 0,
                "rejected_shots": 0,
                "stage_rejected": [0 for _ in filters],
                "cap_reached": False,
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
                summary = _run_filtered_budget(
                    experiment,
                    branch,
                    int(branch["budget"]),
                    int(branch["attempt_cap"]),
                )
                branch["total_attempts"] += summary["attempts"]
                branch["completed_shots"] += summary["accepted"]
                branch["rejected_shots"] += summary["rejected"]
                branch["stage_rejected"] = [
                    total + extra
                    for total, extra in zip(branch["stage_rejected"], summary["stage_rejected"])
                ]
                branch["cap_reached"] = branch["cap_reached"] or summary["cap_reached"]
                branch["batches"].append(
                    {
                        "batch_id": uuid.uuid4().hex,
                        "source": "initial-budget",
                        "requested_accepted_shots": int(branch["budget"]),
                        "attempts": summary["attempts"],
                        "accepted_shots": summary["accepted"],
                        "rejected_shots": summary["rejected"],
                        "stage_rejected": list(summary["stage_rejected"]),
                        "cap_reached": summary["cap_reached"],
                        "total_attempts_after_batch": branch["total_attempts"],
                        "total_shots_after_batch": branch["completed_shots"],
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
        attempt_cap = int(branch.get("attempt_cap", max(shots * 100, 1000)))
        summary = _run_filtered_budget(self, branch, shots, attempt_cap)
        branch["total_attempts"] = int(branch.get("total_attempts", 0)) + summary["attempts"]
        branch["completed_shots"] += summary["accepted"]
        branch["rejected_shots"] = int(branch.get("rejected_shots", 0)) + summary["rejected"]
        existing_stage = list(branch.get("stage_rejected", [0 for _ in branch.get("filters", [])]))
        branch["stage_rejected"] = [
            total + extra for total, extra in zip(existing_stage, summary["stage_rejected"])
        ]
        branch["cap_reached"] = bool(branch.get("cap_reached", False)) or summary["cap_reached"]
        branch["budget"] = max(int(branch["budget"]), branch["completed_shots"])
        batch = {
            "batch_id": uuid.uuid4().hex,
            "source": source,
            "requested_accepted_shots": shots,
            "attempts": summary["attempts"],
            "accepted_shots": summary["accepted"],
            "rejected_shots": summary["rejected"],
            "stage_rejected": list(summary["stage_rejected"]),
            "cap_reached": summary["cap_reached"],
            "total_attempts_after_batch": branch["total_attempts"],
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
        state = _from_json(self.prefix["state"])
        for branch_id in self.branch_order:
            branch = self.branches[branch_id]
            shots = int(branch["completed_shots"])
            filters = list(branch.get("filters", []))
            if shots:
                frequencies = {label: count / shots for label, count in branch["counts"].items()}
                evidence_status = "ok"
            elif filters:
                frequencies = None
                evidence_status = "no-evidence"
            else:
                frequencies = {label: 0.0 for label in branch["counts"]}
                evidence_status = "no-samples"
            exact = _exact_conditioned_probabilities(
                state, self.prefix["qubits"], branch["steps"], filters
            )
            branches.append(
                {
                    "branch_id": branch_id,
                    "label": branch["label"],
                    "condition_steps": branch["steps"],
                    "filters": filters,
                    "budget": branch["budget"],
                    "attempt_cap": int(branch.get("attempt_cap", 0)),
                    "total_attempts": int(branch.get("total_attempts", shots)),
                    "passed_shots": shots,
                    "completed_shots": shots,
                    "rejected_shots": int(branch.get("rejected_shots", 0)),
                    "stage_rejected": list(branch.get("stage_rejected", [0 for _ in filters])),
                    "cap_reached": bool(branch.get("cap_reached", False)),
                    "counts": dict(branch["counts"]),
                    "frequencies": frequencies,
                    "evidence_status": evidence_status,
                    "sample_scope": (
                        "all attempts"
                        if not filters
                        else "attempts passing every measurement filter"
                    ),
                    "exact": exact,
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
