"""Persistent, independently sampled conditional continuation branches."""

from __future__ import annotations

import hashlib
import json
import math
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


def _normalize_filters(raw: Any, steps: list[dict[str, Any]]) -> list[dict[str, int]]:
    """Validate measurement-outcome filters against the branch steps.

    Each filter pins one MEASURE step (by index) to a required outcome.
    Multiple filters may point at different measurement positions, which
    composes consecutive filtering along the branch.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("filters must be a list of {step, outcome} objects")
    normalized: list[dict[str, int]] = []
    seen_steps: set[int] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each filter must be an object with step and outcome")
        step = item.get("step")
        outcome = item.get("outcome")
        if not isinstance(step, int) or isinstance(step, bool) or not 0 <= step < len(steps):
            raise ValueError("filter step must be an integer index into the branch steps")
        if steps[step].get("type") != quantum.MEASURE_GATE:
            raise ValueError("filter step must point at a MEASURE step")
        if not isinstance(outcome, int) or isinstance(outcome, bool) or outcome not in (0, 1):
            raise ValueError("filter outcome must be 0 or 1")
        if step in seen_steps:
            raise ValueError("duplicate filter for the same measurement step")
        seen_steps.add(step)
        normalized.append({"step": step, "outcome": int(outcome)})
    return normalized


def _passes_filters(filters: list[dict[str, int]], measurements: list[dict[str, Any]]) -> bool:
    by_step = {int(entry["step"]): int(entry["outcome"]) for entry in measurements}
    return all(by_step.get(f["step"]) == f["outcome"] for f in filters)


PASS_EVENT = "__pass__"

RULE_VERSION = "beta-mixture-cs-v2"
RULE_DESCRIPTION = (
    "Beta(1,1) 混合置信序列（v2 修正版）：上界按似然阈值在 MLE 右侧求根，"
    "支持反复查看与自适应分配；全成功/全失败时对应一侧退到 0 或 1"
)
LEGACY_RULE_VERSION = "beta-mixture-cs-v1"
LEGACY_RULE_DESCRIPTION = (
    "Beta(1,1) 混合置信序列（v1 旧版）：上界二分方向错误，80/100 等数据的上界被算成 MLE，"
    "区间过窄；仅用于标注修正前的历史研究，不再用于新决定"
)
_BISECTION_ENDPOINT = 1e-300


def beta_mixture_interval(successes: int, trials: int, alpha: float) -> tuple[float, float]:
    """Beta(1,1)-mixture confidence sequence for a Bernoulli probability.

    The boundary is evaluated from all current counts and remains valid when
    results are inspected repeatedly or sampling is chosen from earlier data.
    With no trials it is deliberately the full uninformative interval.
    """
    successes = int(successes)
    trials = int(trials)
    if trials <= 0:
        return 0.0, 1.0
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")

    log_marginal = (
        math.lgamma(successes + 1)
        + math.lgamma(trials - successes + 1)
        - math.lgamma(trials + 2)
    )
    threshold = log_marginal - math.log(1.0 / alpha)

    def log_likelihood(probability: float) -> float:
        if probability <= 0.0 or probability >= 1.0:
            return float("-inf")
        return successes * math.log(probability) + (trials - successes) * math.log(1.0 - probability)

    def root_below_mle(left: float, right: float) -> float:
        # On (0, MLE] likelihood increases with p.
        for _ in range(60):
            middle = (left + right) / 2.0
            if log_likelihood(middle) >= threshold:
                right = middle
            else:
                left = middle
        return (left + right) / 2.0

    def root_above_mle(left: float, right: float) -> float:
        # On [MLE, 1) likelihood decreases with p.
        for _ in range(60):
            middle = (left + right) / 2.0
            if log_likelihood(middle) >= threshold:
                left = middle
            else:
                right = middle
        return (left + right) / 2.0

    mle = successes / trials
    lower = 0.0 if successes == 0 else root_below_mle(_BISECTION_ENDPOINT, mle)
    upper = 1.0 if successes == trials else root_above_mle(mle, 1.0 - _BISECTION_ENDPOINT)
    return max(0.0, lower), min(1.0, upper)


def _uncertainty_width(interval: tuple[float, float]) -> float:
    return float(interval[1] - interval[0])


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
        inquiries: dict[str, dict[str, Any]] | None = None,
        inquiry_order: list[str] | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.name = name
        self.base_seed = base_seed
        self.prefix = prefix
        self.branches = branches
        self.branch_order = branch_order
        self.created_at = created_at
        self.storage_dir = Path(storage_dir)
        self.inquiries = inquiries or {}
        self.inquiry_order = inquiry_order or []

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
        inquiries: list[dict[str, Any]] | None = None,
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
            filters = _normalize_filters(spec.get("filters"), steps)
            rng = np.random.default_rng(_branch_seed(resolved_seed, branch_id))
            branches[branch_id] = {
                "branch_id": branch_id,
                "label": label,
                "steps": steps,
                "filters": filters,
                "budget": budget,
                "completed_shots": 0,
                "passed_shots": 0,
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
        normalized_inquiries = [
            cls.normalize_inquiry(raw, branch_order, set(labels_for_qubits))
            for raw in (inquiries or [])
        ]
        experiment = cls(
            experiment_id=experiment_id,
            name=name,
            base_seed=resolved_seed,
            prefix=prefix,
            branches=branches,
            branch_order=branch_order,
            created_at=created_at if created_at is not None else time.time(),
            storage_dir=Path(storage_dir),
            inquiries={item["inquiry_id"]: item for item in normalized_inquiries},
            inquiry_order=[item["inquiry_id"] for item in normalized_inquiries],
        )
        if inquiries:
            for branch_id in experiment.branch_order:
                branch = experiment.branches[branch_id]
                if branch["budget"]:
                    experiment._sample(branch, int(branch["budget"]), source="legacy-initial-budget")
            for inquiry in experiment.inquiries.values():
                experiment._record_inquiry_decision(
                    inquiry,
                    action="created",
                    branch_id=None,
                    shots=0,
                    reason="研究决定已创建；尚未按该决定追加自适应样本",
                )
        else:
            for branch_id, branch in experiment.branches.items():
                if branch["budget"]:
                    experiment._sample(branch, int(branch["budget"]), source="initial-budget")
        experiment.save()
        return experiment

    def _sample(self, branch: dict[str, Any], shots: int, *, source: str) -> dict[str, Any]:
        """Run shots for one branch, keeping attempt identity across batches.

        Every shot is an attempt. Attempts are numbered implicitly by the
        branch RNG stream, so pausing, reloading, or appending budget never
        re-draws or double-counts an attempt. Only attempts whose mid-circuit
        measurements satisfy every filter enter the conditional counts.
        """
        state = _from_json(self.prefix["state"])
        rng = rng_from_json(branch["rng_state"])
        filters = branch.get("filters", [])
        for _ in range(shots):
            result = quantum.sample_continuation(
                state, self.prefix["qubits"], branch["steps"], rng
            )
            branch["completed_shots"] += 1
            if _passes_filters(filters, result["measurements"]):
                branch["passed_shots"] += 1
                branch["counts"][result["sample"]] += 1
        branch["budget"] = max(int(branch["budget"]), branch["completed_shots"])
        branch["rng_state"] = rng_to_json(rng)
        batch = {
            "batch_id": uuid.uuid4().hex,
            "source": source,
            "shots": shots,
            "total_shots_after_batch": branch["completed_shots"],
        }
        branch["batches"].append(batch)
        return batch

    def add_shots(self, branch_id: str, shots: int, *, source: str = "continued") -> dict[str, Any]:
        if branch_id not in self.branches:
            raise KeyError(f"unknown branch: {branch_id}")
        shots = int(shots)
        if shots <= 0:
            raise ValueError("shots must be positive")
        branch = self.branches[branch_id]
        batch = self._sample(branch, shots, source=source)
        self.save()
        return batch

    def _event_successes(self, branch: dict[str, Any], event: str) -> int:
        if event == PASS_EVENT:
            return int(branch.get("passed_shots", int(branch["completed_shots"])))
        if event not in branch["counts"]:
            raise ValueError(f"unknown event {event!r} for branch {branch['branch_id']}")
        return int(branch["counts"][event])

    def _event_trials(self, branch: dict[str, Any], event: str) -> int:
        shots = int(branch["completed_shots"])
        if event == PASS_EVENT or not branch.get("filters"):
            return shots
        return int(branch.get("passed_shots", 0))

    def _branch_uncertainty(self, branch_id: str, event: str, alpha: float) -> dict[str, Any]:
        branch = self.branches[branch_id]
        shots = int(branch["completed_shots"])
        passed = int(branch.get("passed_shots", shots))
        filtered = bool(branch.get("filters"))
        pass_interval = beta_mixture_interval(passed, shots, alpha / 2) if filtered else None
        target_successes = self._event_successes(branch, event)
        target_trials = self._event_trials(branch, event)
        target_interval = beta_mixture_interval(target_successes, target_trials, alpha / 2)
        target_width = _uncertainty_width(target_interval)
        score = target_width
        if filtered and event != PASS_EVENT:
            score += _uncertainty_width(pass_interval or (0.0, 1.0))
        if filtered and event != PASS_EVENT and passed == 0:
            evidence_note = (
                f"筛选零命中：接受率本身有 {shots} 次尝试的证据，区间 {pass_interval}；"
                "接受后事件有效样本数为 0，条件分布没有证据，区间保持 [0,1]"
            )
        elif filtered:
            evidence_note = f"筛选命中 {passed}/{shots}；接受率与目标事件分开估计"
        else:
            evidence_note = f"基于全部 {shots} 次尝试"
        return {
            "branch_id": branch_id,
            "label": branch["label"],
            "attempts": shots,
            "passed_shots": passed if filtered else None,
            "event": event,
            "event_successes": target_successes,
            "event_trials": target_trials,
            "event_interval": list(target_interval),
            "event_width": target_width,
            "pass_interval": None if pass_interval is None else list(pass_interval),
            "allocation_score": score,
            "evidence_note": evidence_note,
        }

    def _inquiry_used_budget(self, inquiry: dict[str, Any]) -> int:
        return sum(
            int(entry.get("shots", 0))
            for entry in inquiry.get("decisions", [])
            if entry.get("action") == "sample"
        )

    @staticmethod
    def _inquiry_rule_version(inquiry: dict[str, Any]) -> str:
        return str(inquiry.get("rule_version") or LEGACY_RULE_VERSION)

    def _evaluate_inquiry(self, inquiry: dict[str, Any]) -> dict[str, Any]:
        alpha = float(inquiry["alpha"])
        margin = float(inquiry["margin"])
        stats = [
            self._branch_uncertainty(branch_id, inquiry["event"], alpha)
            for branch_id in inquiry["branch_ids"]
        ]
        used = self._inquiry_used_budget(inquiry)
        total_budget = int(inquiry["total_budget"])
        status = "continue"
        conclusion = None

        if inquiry["mode"] == "single":
            stat = stats[0]
            lower, upper = stat["event_interval"]
            if upper - lower <= 2 * margin:
                status = "decided"
                midpoint = (lower + upper) / 2
                conclusion = {
                    "type": "probability_estimated",
                    "estimate": midpoint,
                    "interval": [lower, upper],
                    "message": f"置信区间宽度不超过 {2 * margin:.4g}，按 {inquiry['event']} 概率已达到精度目标",
                }
        else:
            first, second = stats
            l1, u1 = first["event_interval"]
            l2, u2 = second["event_interval"]
            observed_difference = (
                first["event_successes"] / first["event_trials"] if first["event_trials"] else None,
                second["event_successes"] / second["event_trials"] if second["event_trials"] else None,
            )
            interval = [l1 - u2, u1 - l2]
            if interval[1] < -margin:
                status = "decided"
                conclusion = {
                    "type": "branch_b_higher",
                    "difference_interval": interval,
                    "message": f"{second['label']} 的事件概率高于 {first['label']}，且差异下界超过允许误差",
                }
            elif interval[0] > margin:
                status = "decided"
                conclusion = {
                    "type": "branch_a_higher",
                    "difference_interval": interval,
                    "message": f"{first['label']} 的事件概率高于 {second['label']}，且差异下界超过允许误差",
                }
            elif -margin <= interval[0] and interval[1] <= margin:
                status = "decided"
                conclusion = {
                    "type": "equivalent_within_margin",
                    "difference_interval": interval,
                    "message": "两分支概率差的整个序贯区间位于允许误差内，可判定实际等价",
                }
            stats = [
                {**stat, "difference_interval": interval, "observed_difference": observed_difference}
                for stat in stats
            ]

        if status == "continue" and used >= total_budget:
            status = "inconclusive_budget_exhausted"
            conclusion = {
                "type": "inconclusive",
                "message": "总尝试上限已用完但序贯区间仍不能区分；保持未确定，不更换统计口径给出赢家",
            }

        return {
            "status": status,
            "rule_version": RULE_VERSION,
            "rule_description": RULE_DESCRIPTION,
            "remaining_budget": max(0, total_budget - used),
            "used_budget": used,
            "branch_statistics": stats,
            "conclusion": conclusion,
        }

    def _record_inquiry_decision(
        self,
        inquiry: dict[str, Any],
        *,
        action: str,
        branch_id: str | None,
        shots: int,
        reason: str,
        before_statistics: list[dict[str, Any]] | None = None,
        evaluation: dict[str, Any] | None = None,
        batch_id: str | None = None,
    ) -> dict[str, Any]:
        entry = {
            "decision_id": uuid.uuid4().hex,
            "at": time.time(),
            "action": action,
            "rule_version": RULE_VERSION,
            "rule_description": RULE_DESCRIPTION,
            "branch_id": branch_id,
            "shots": int(shots),
            "reason": reason,
            "batch_id": batch_id,
            "before_statistics": before_statistics,
            "evaluation_after": evaluation,
        }
        inquiry.setdefault("decisions", []).append(entry)
        return entry

    @staticmethod
    def normalize_inquiry(raw: dict[str, Any], branch_ids: list[str], known_events: set[str]) -> dict[str, Any]:
        mode = raw.get("mode")
        if mode not in ("single", "compare"):
            raise ValueError("inquiry mode must be 'single' or 'compare'")
        selected = raw.get("branch_ids")
        if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
            raise ValueError("inquiry branch_ids must be a list of branch id strings")
        expected = 1 if mode == "single" else 2
        if len(selected) != expected or any(item not in branch_ids for item in selected):
            raise ValueError(f"{mode} inquiry must reference {expected} existing branches")
        event = str(raw.get("event", PASS_EVENT))
        if event != PASS_EVENT and event not in known_events:
            raise ValueError("inquiry event must be an output bitstring or '__pass__'")
        margin = raw.get("margin")
        if not isinstance(margin, (int, float)) or isinstance(margin, bool) or not 0.0 < float(margin) < 1.0:
            raise ValueError("margin must be a number strictly between 0 and 1")
        total_budget = raw.get("total_budget")
        if not isinstance(total_budget, int) or isinstance(total_budget, bool) or not 1 <= total_budget <= 1_000_000:
            raise ValueError("total_budget must be an integer in [1, 1000000]")
        batch_size = raw.get("batch_size", 1)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= total_budget:
            raise ValueError("batch_size must be a positive integer not exceeding total_budget")
        alpha = raw.get("alpha", 0.05)
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        return {
            "inquiry_id": uuid.uuid4().hex,
            "revision": 1,
            "created_at": time.time(),
            "rule_version": RULE_VERSION,
            "rule_description": RULE_DESCRIPTION,
            "mode": mode,
            "branch_ids": list(selected),
            "event": event,
            "margin": float(margin),
            "total_budget": int(total_budget),
            "batch_size": int(batch_size),
            "alpha": float(alpha),
            "immutable": True,
            "status": "active",
            "decisions": [],
        }

    def add_inquiry(self, raw_inquiry: dict[str, Any]) -> dict[str, Any]:
        events = set(next(iter(self.branches.values()))["counts"].keys())
        inquiry = self.normalize_inquiry(raw_inquiry, list(self.branch_order), events)
        self.inquiries[inquiry["inquiry_id"]] = inquiry
        self.inquiry_order.append(inquiry["inquiry_id"])
        self._record_inquiry_decision(
            inquiry,
            action="created",
            branch_id=None,
            shots=0,
            reason="新研究决定记录；修改精度目标或关心事件需再创建决定，历史决定保持只读",
        )
        self.save()
        return self.inquiry_snapshot(inquiry["inquiry_id"])

    def run_adaptive_batch(self, inquiry_id: str, *, batch_size: int | None = None) -> dict[str, Any]:
        if inquiry_id not in self.inquiries:
            raise KeyError(f"unknown inquiry: {inquiry_id}")
        inquiry = self.inquiries[inquiry_id]
        if inquiry["status"] != "active":
            raise ValueError("this inquiry has already stopped")
        used = self._inquiry_used_budget(inquiry)
        before = self._evaluate_inquiry(inquiry)
        if before["status"] != "continue":
            inquiry["status"] = before["status"]
            self._record_inquiry_decision(
                inquiry, action="stop", branch_id=None, shots=0,
                reason=before["conclusion"]["message"], evaluation=before,
            )
            self.save()
            return before
        chosen = max(before["branch_statistics"], key=lambda item: (item["allocation_score"], item["branch_id"]))
        requested = int(batch_size if batch_size is not None else inquiry["batch_size"])
        if requested <= 0:
            raise ValueError("batch_size must be positive")
        shots = min(requested, int(inquiry["total_budget"]) - used)
        if self._inquiry_rule_version(inquiry) == LEGACY_RULE_VERSION:
            self._record_inquiry_decision(
                inquiry,
                action="created",
                branch_id=None,
                shots=0,
                reason=(
                    "规则修正留痕：该决定原按 v1（上界错误、区间过窄）规则执行；"
                    "自本批起改用 v2 修正规则，此前计数、批次与决策日志原样保留"
                ),
                before_statistics=None,
            )
            inquiry["rule_version"] = RULE_VERSION
            inquiry["rule_description"] = RULE_DESCRIPTION
        batch = self._sample(
            self.branches[chosen["branch_id"]],
            shots,
            source=f"adaptive:{inquiry_id}",
        )
        self._record_inquiry_decision(
            inquiry,
            action="sample",
            branch_id=chosen["branch_id"],
            shots=shots,
            reason=(
                f"选择证据宽度最大的分支 {chosen['label']}："
                f"分配分数 {chosen['allocation_score']:.4g}。{chosen['evidence_note']}"
            ),
            before_statistics=before["branch_statistics"],
            batch_id=batch["batch_id"],
        )
        evaluation = self._evaluate_inquiry(inquiry)
        stopped = evaluation["status"] != "continue"
        inquiry["decisions"][-1]["evaluation_after"] = evaluation
        if stopped:
            inquiry["status"] = evaluation["status"]
        if stopped:
            self._record_inquiry_decision(
                inquiry,
                action="stop",
                branch_id=chosen["branch_id"],
                shots=0,
                reason=evaluation["conclusion"]["message"],
                evaluation=evaluation,
            )
        self.save()
        return evaluation

    def inquiry_snapshot(self, inquiry_id: str) -> dict[str, Any]:
        inquiry = self.inquiries[inquiry_id]
        evaluation = self._evaluate_inquiry(inquiry)
        stored_rule_version = self._inquiry_rule_version(inquiry)
        legacy = stored_rule_version == LEGACY_RULE_VERSION
        rule_note = (
            "该研究决定创建于规则修正前（v1，上界等于 MLE、区间过窄）；原始计数、批次与决策日志均不改动，"
            "当前重新计算仅作 v2 对照，不覆盖该决定既有的停止状态与原因。"
            if legacy and inquiry["status"] != "active"
            else (
                "该研究决定创建于规则修正前（v1）；尚未停止，此后追加样本与停止判断改用 v2 修正规则，"
                "原始计数与批次不变，规则切换在决策日志中留痕。"
                if legacy
                else None
            )
        )
        decisions = []
        for entry in inquiry.get("decisions", []):
            annotated = dict(entry)
            version = annotated.get("rule_version")
            if version is None:
                annotated["rule_version"] = LEGACY_RULE_VERSION
                annotated["rule_description"] = LEGACY_RULE_DESCRIPTION
            decisions.append(annotated)
        return {
            **{key: value for key, value in inquiry.items() if key != "decisions"},
            "rule_version": stored_rule_version,
            "rule_description": (
                LEGACY_RULE_DESCRIPTION if legacy else inquiry.get("rule_description", RULE_DESCRIPTION)
            ),
            "rule_note": rule_note,
            "current_evaluation": evaluation,
            "decisions": decisions,
        }

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
            "inquiry_order": list(self.inquiry_order),
            "inquiries": {key: _jsonable(value) for key, value in self.inquiries.items()},
        }

    def snapshot(self) -> dict[str, Any]:
        branches = []
        for branch_id in self.branch_order:
            branch = self.branches[branch_id]
            shots = int(branch["completed_shots"])
            filters = list(branch.get("filters", []))
            passed = int(branch.get("passed_shots", shots))
            effective = passed if filters else shots
            if filters and passed == 0:
                # No attempt satisfied the filters: report "no evidence"
                # instead of an all-zero pseudo distribution.
                frequencies = None
            else:
                frequencies = {
                    label: (count / effective if effective else 0.0)
                    for label, count in branch["counts"].items()
                }
            if filters:
                condition = "，".join(
                    f"第 {item['step'] + 1} 步测量结果={item['outcome']}" for item in filters
                )
                if passed:
                    sample_scope = (
                        f"仅统计满足筛选（{condition}）的 {passed}/{shots} 次尝试"
                    )
                else:
                    sample_scope = (
                        f"筛选条件（{condition}）在 {shots} 次尝试中没有命中，"
                        "没有证据，条件分布不可用"
                    )
            else:
                sample_scope = f"无筛选，统计全部 {shots} 次尝试"
            branches.append(
                {
                    "branch_id": branch_id,
                    "label": branch["label"],
                    "condition_steps": branch["steps"],
                    "filters": filters,
                    "budget": branch["budget"],
                    "completed_shots": shots,
                    "attempts": shots,
                    "passed_shots": passed,
                    "pass_rate": (passed / shots) if shots else None,
                    "has_evidence": (not filters) or passed > 0,
                    "sample_scope": sample_scope,
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
            "inquiries": [self.inquiry_snapshot(inquiry_id) for inquiry_id in self.inquiry_order],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], storage_dir: str | Path) -> "BranchExperiment":
        for branch in data["branches"].values():
            branch.setdefault("filters", [])
            branch.setdefault("passed_shots", int(branch.get("completed_shots", 0)))
        inquiries = data.get("inquiries", {})
        return cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            base_seed=int(data["base_seed"]),
            prefix=data["prefix"],
            branches=data["branches"],
            branch_order=list(data["branch_order"]),
            created_at=float(data["created_at"]),
            storage_dir=Path(storage_dir),
            inquiries=inquiries,
            inquiry_order=list(data.get("inquiry_order", inquiries.keys())),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BranchExperiment":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle), path.parent)
