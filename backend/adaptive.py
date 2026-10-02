"""Adaptive sampling-budget studies on top of conditional branch experiments.

A study pins down, before sampling starts: the conditional probability of
interest (an event on the final computational-basis outcome), the allowed
error (epsilon), the confidence level (1 - delta), and a total attempt cap.
The study then repeatedly allocates the remaining budget to the branch with
the weakest evidence instead of appending a fixed amount to every branch.

Stopping and continue decisions use time-uniform Hoeffding confidence
sequences, so the guarantees hold under repeated peeking and under
data-dependent allocation; a fixed-sample guarantee is never reused as a
sequential one. When the budget is exhausted before a decision, the study
stays "undecided" -- it never switches to a different statistic to force a
winner.

Every allocation and stop/continue decision is appended to a persistent
decision log, so pausing, resuming, or reordering branch execution keeps
the attempts, the stopping rule, and the allocation choices traceable.
Changing the precision target or the event of interest requires creating a
new study; existing study files keep their original spec and remain
readable under the original conditions.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

STUDY_DIR_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "adaptive-studies"

MAX_TOTAL_BUDGET = 100_000
MAX_BATCH_SIZE = 10_000

DECISION_REASONS = {
    "estimate": "条件分布的不确定半径已不超过精度目标 epsilon",
    "branch_a_greater": "概率差的时一致置信区间整体大于 0",
    "branch_b_greater": "概率差的时一致置信区间整体小于 0",
    "within_epsilon": "概率差的时一致置信区间整体落在 [-epsilon, epsilon] 内，按当前精度目标无法区分",
}


def cs_radius(n: int, delta: float) -> float:
    """Time-uniform Hoeffding confidence-sequence radius for n observations.

    Valid simultaneously over all n >= 1 and under arbitrary stopping and
    adaptive allocation, unlike a fixed-sample interval.
    """
    if n <= 0:
        return math.inf
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def _validate_event(raw: Any, qubits: int) -> list[str]:
    if not isinstance(raw, dict) or not isinstance(raw.get("basis_states"), list):
        raise ValueError("event must be an object with a basis_states list")
    states = raw["basis_states"]
    if not states:
        raise ValueError("event basis_states must not be empty")
    normalized: list[str] = []
    for item in states:
        if (
            not isinstance(item, str)
            or len(item) != qubits
            or any(ch not in "01" for ch in item)
        ):
            raise ValueError(
                f"event basis state must be a {qubits}-bit string of 0/1, got {item!r}"
            )
        if item in normalized:
            raise ValueError("event basis_states must not contain duplicates")
        normalized.append(item)
    return normalized


def _validate_spec(payload: dict[str, Any], experiment: Any) -> dict[str, Any]:
    mode = payload.get("mode")
    if mode not in ("estimate", "compare"):
        raise ValueError("mode must be 'estimate' or 'compare'")
    raw_branches = payload.get("branches")
    if not isinstance(raw_branches, list):
        raise ValueError("branches must be a list of branch ids")
    branch_ids = [str(item) for item in raw_branches]
    expected = 1 if mode == "estimate" else 2
    if len(branch_ids) != expected or len(set(branch_ids)) != expected:
        raise ValueError(f"mode '{mode}' requires exactly {expected} distinct branch id(s)")
    for branch_id in branch_ids:
        if branch_id not in experiment.branches:
            raise ValueError(f"unknown branch: {branch_id}")

    qubits = int(experiment.prefix["qubits"])
    event = _validate_event(payload.get("event"), qubits)

    epsilon = payload.get("epsilon")
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or not 0 < float(epsilon) <= 0.5:
        raise ValueError("epsilon must be a number in (0, 0.5]")
    delta = payload.get("delta", 0.05)
    if not isinstance(delta, (int, float)) or isinstance(delta, bool) or not 0 < float(delta) < 0.5:
        raise ValueError("delta must be a number in (0, 0.5)")
    total_budget = payload.get("total_budget")
    if not isinstance(total_budget, int) or isinstance(total_budget, bool) or not 1 <= total_budget <= MAX_TOTAL_BUDGET:
        raise ValueError(f"total_budget must be an integer in [1, {MAX_TOTAL_BUDGET}]")
    batch_size = payload.get("batch_size", 1)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be an integer in [1, {MAX_BATCH_SIZE}]")

    return {
        "mode": mode,
        "branches": branch_ids,
        "event": {"basis_states": event},
        "epsilon": float(epsilon),
        "delta": float(delta),
        "total_budget": total_budget,
        "batch_size": batch_size,
    }


class AdaptiveStudy:
    """One immutable research question plus its sequential decision log."""

    def __init__(
        self,
        *,
        study_id: str,
        name: str,
        experiment_id: str,
        experiment_storage_dir: str,
        spec: dict[str, Any],
        status: str,
        decision: dict[str, Any] | None,
        spent: dict[str, int],
        log: list[dict[str, Any]],
        created_at: float,
        storage_dir: Path,
    ) -> None:
        self.study_id = study_id
        self.name = name
        self.experiment_id = experiment_id
        self.experiment_storage_dir = experiment_storage_dir
        self.spec = spec
        self.status = status  # "running" | "decided" | "undecided"
        self.decision = decision
        self.spent = spent
        self.log = log
        self.created_at = created_at
        self.storage_dir = Path(storage_dir)

    @property
    def path(self) -> Path:
        return self.storage_dir / f"{self.study_id}.json"

    @classmethod
    def create(
        cls,
        experiment: Any,
        payload: dict[str, Any],
        *,
        storage_dir: str | Path,
        name: str | None = None,
        study_id: str | None = None,
        created_at: float | None = None,
    ) -> "AdaptiveStudy":
        spec = _validate_spec(payload, experiment)
        study = cls(
            study_id=study_id or uuid.uuid4().hex,
            name=(name or "adaptive-study").strip() or "adaptive-study",
            experiment_id=experiment.experiment_id,
            experiment_storage_dir=str(experiment.storage_dir),
            spec=spec,
            status="running",
            decision=None,
            spent={branch_id: 0 for branch_id in spec["branches"]},
            log=[],
            created_at=created_at if created_at is not None else time.time(),
            storage_dir=Path(storage_dir),
        )
        study._log(
            "create",
            reason=(
                f"研究开始：模式={spec['mode']}，事件={spec['event']['basis_states']}，"
                f"epsilon={spec['epsilon']}，delta={spec['delta']}，"
                f"总预算={spec['total_budget']}，批量={spec['batch_size']}；"
                "停止判断使用时一致置信序列，允许中途多次查看与自适应分配"
            ),
        )
        study.save()
        return study

    # ------------------------------------------------------------------
    # evidence
    # ------------------------------------------------------------------
    def _branch_evidence(self, experiment: Any, branch_id: str) -> dict[str, Any]:
        branch = experiment.branches[branch_id]
        delta = self.spec["delta"]
        if self.spec["mode"] == "compare":
            # Bonferroni split so the two sequences hold jointly at 1 - delta.
            delta = delta / 2.0
        completed = int(branch["completed_shots"])
        filters = branch.get("filters", [])
        passed = int(branch.get("passed_shots", completed)) if filters else completed
        event_states = set(self.spec["event"]["basis_states"])
        event_count = sum(
            int(count) for state, count in branch["counts"].items() if state in event_states
        )
        acceptance_radius = cs_radius(completed, delta) if filters else None
        conditional_radius = cs_radius(passed, delta)
        estimate = (event_count / passed) if passed else None
        return {
            "branch_id": branch_id,
            "label": branch["label"],
            "has_filters": bool(filters),
            "completed_shots": completed,
            "passed_shots": passed,
            "pass_rate": (passed / completed) if completed else None,
            "acceptance_uncertainty": acceptance_radius,
            "event_count": event_count,
            "conditional_estimate": estimate,
            "conditional_uncertainty": None if passed == 0 else conditional_radius,
            "has_evidence": passed > 0,
            "study_spent": int(self.spent.get(branch_id, 0)),
        }

    def evidence(self, experiment: Any) -> list[dict[str, Any]]:
        return [self._branch_evidence(experiment, b) for b in self.spec["branches"]]

    # ------------------------------------------------------------------
    # stopping rule
    # ------------------------------------------------------------------
    def _check_stopping(self, evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
        epsilon = self.spec["epsilon"]
        if self.spec["mode"] == "estimate":
            item = evidence[0]
            if not item["has_evidence"]:
                return None
            if item["conditional_uncertainty"] <= epsilon:
                estimate = item["conditional_estimate"]
                radius = item["conditional_uncertainty"]
                return {
                    "type": "estimate",
                    "branch_id": item["branch_id"],
                    "estimate": estimate,
                    "interval": [max(0.0, estimate - radius), min(1.0, estimate + radius)],
                    "reason": DECISION_REASONS["estimate"],
                }
            return None
        first, second = evidence
        if not (first["has_evidence"] and second["has_evidence"]):
            return None
        radius = first["conditional_uncertainty"] + second["conditional_uncertainty"]
        difference = first["conditional_estimate"] - second["conditional_estimate"]
        low, high = difference - radius, difference + radius
        interval = [max(-1.0, low), min(1.0, high)]
        if low > 0:
            kind = "branch_a_greater"
        elif high < 0:
            kind = "branch_b_greater"
        elif radius <= epsilon:
            kind = "within_epsilon"
        else:
            return None
        return {
            "type": kind,
            "difference": difference,
            "interval": interval,
            "reason": DECISION_REASONS[kind],
        }

    # ------------------------------------------------------------------
    # logging / persistence
    # ------------------------------------------------------------------
    def _log(self, action: str, **fields: Any) -> dict[str, Any]:
        entry = {"round": len(self.log), "action": action, "timestamp": time.time(), **fields}
        self.log.append(entry)
        return entry

    def save(self) -> Path:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
        temporary.replace(self.path)
        return self.path

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "name": self.name,
            "experiment_id": self.experiment_id,
            "experiment_storage_dir": self.experiment_storage_dir,
            "spec": self.spec,
            "status": self.status,
            "decision": self.decision,
            "spent": dict(self.spent),
            "log": list(self.log),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], storage_dir: str | Path) -> "AdaptiveStudy":
        return cls(
            study_id=data["study_id"],
            name=data["name"],
            experiment_id=data["experiment_id"],
            experiment_storage_dir=data["experiment_storage_dir"],
            spec=data["spec"],
            status=data["status"],
            decision=data.get("decision"),
            spent={key: int(value) for key, value in data.get("spent", {}).items()},
            log=list(data.get("log", [])),
            created_at=float(data["created_at"]),
            storage_dir=Path(storage_dir),
        )

    @classmethod
    def load(cls, path: str | Path) -> "AdaptiveStudy":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle), path.parent)

    # ------------------------------------------------------------------
    # sequential loop
    # ------------------------------------------------------------------
    def _total_spent(self) -> int:
        return sum(self.spent.values())

    def advance(self, experiment: Any) -> dict[str, Any]:
        """Run one round: check stopping, otherwise allocate one batch."""
        if self.status != "running":
            return self._log(
                "noop",
                reason=f"研究已结束（{self.status}），不再分配采样；历史决定保持不变",
            )
        evidence = self.evidence(experiment)
        decision = self._check_stopping(evidence)
        if decision is not None:
            self.status = "decided"
            self.decision = decision
            self._log(
                "stop",
                decision=decision,
                evidence=evidence,
                reason=f"停止：{decision['reason']}",
            )
            self.save()
            return self.log[-1]

        remaining = self.spec["total_budget"] - self._total_spent()
        if remaining <= 0:
            self.status = "undecided"
            self.decision = {
                "type": "undecided",
                "reason": (
                    "总尝试预算已用完，时一致置信区间仍不能按精度目标区分；"
                    "保留未确定，不改用其他统计口径强行给出结论"
                ),
            }
            self._log("stop", decision=self.decision, evidence=evidence, reason=f"停止：{self.decision['reason']}")
            self.save()
            return self.log[-1]

        target = self._weakest_branch(evidence)
        shots = min(self.spec["batch_size"], remaining)
        reason = self._allocation_reason(evidence, target, shots, remaining)
        experiment.add_shots(target, shots, source=f"study:{self.study_id}")
        self.spent[target] = self.spent.get(target, 0) + shots
        self._log("allocate", branch_id=target, shots=shots, evidence=evidence, reason=reason)
        self.save()
        return self.log[-1]

    def run(self, experiment: Any, max_rounds: int = 100_000) -> None:
        """Keep advancing until the study decides or exhausts its budget."""
        rounds = 0
        while self.status == "running" and rounds < max_rounds:
            self.advance(experiment)
            rounds += 1

    def _weakest_branch(self, evidence: list[dict[str, Any]]) -> str:
        def weakness(item: dict[str, Any]) -> tuple[float, int]:
            if not item["has_evidence"]:
                return (math.inf, -item["completed_shots"])
            return (float(item["conditional_uncertainty"]), -item["completed_shots"])

        return max(evidence, key=weakness)["branch_id"]

    def _allocation_reason(
        self, evidence: list[dict[str, Any]], target: str, shots: int, remaining: int
    ) -> str:
        item = next(entry for entry in evidence if entry["branch_id"] == target)
        if not item["has_evidence"]:
            why = "筛选条件尚无命中，条件分布没有证据"
        else:
            why = f"条件分布不确定半径 {item['conditional_uncertainty']:.4f} 为各分支最大"
        return (
            f"继续：尚未满足停止条件；分支「{item['label']}」证据最不足（{why}），"
            f"从剩余预算 {remaining} 中追加 {shots} 次尝试"
        )

    # ------------------------------------------------------------------
    # snapshot
    # ------------------------------------------------------------------
    def snapshot(self, experiment: Any) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "name": self.name,
            "experiment_id": self.experiment_id,
            "storage_path": str(self.path),
            "spec": self.spec,
            "status": self.status,
            "decision": self.decision,
            "spent": dict(self.spent),
            "total_spent": self._total_spent(),
            "remaining_budget": self.spec["total_budget"] - self._total_spent(),
            "evidence": self.evidence(experiment),
            "log": list(self.log),
            "created_at": self.created_at,
            "sequential_note": (
                "所有区间均为时一致置信序列，允许中途多次查看与按证据自适应分配；"
                "预算耗尽仍未区分时保留未确定"
            ),
        }

