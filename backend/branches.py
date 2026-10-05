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

# Confidence-sequence rule version embedded in every new inquiry and
# evaluation. v2 fixes the reversed upper-root bisection (and the unsafe
# fixed 1e-12 bracket) that made upper endpoints collapse onto the MLE,
# e.g. 80/100 returned upper bound 0.80 instead of ~0.89.
CS_RULE_VERSION = "beta-mixture-cs-v2"
# Historical records produced before the bisection fix are tagged with this
# rule name when loaded from disk; stored counts/decisions stay untouched.
LEGACY_CS_RULE_VERSION = "beta-mixture-cs-v1-buggy-upper-root"
CS_RULE_NOTE = (
    "规则 beta-mixture-cs-v2：Beta(1,1) 混合似然置信序列，"
    "修正了 v1 上界二分方向反转导致上界坍缩到点估计的缺陷；"
    "边界 0/1 用从 MLE 指数扩展的括号求根，零命中与全命中均保持有限证据区间"
)


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

    mle = successes / trials

    def accepted(probability: float) -> bool:
        return log_likelihood(probability) >= threshold

    def root(inside: float, direction: int) -> float:
        # Expand exponentially from the MLE until the bracket crosses the
        # boundary; this stays valid for rare events and very large samples
        # where a fixed epsilon (e.g. 1e-12) would itself be inside the set.
        step = max(mle, 1.0 - mle, 1e-300)
        candidate = inside
        for _ in range(2000):
            trial = candidate + direction * step
            if trial <= 0.0 or trial >= 1.0 or not accepted(trial):
                break
            candidate = trial
            step *= 2.0
        inside_pt, outside_pt = inside, max(0.0, min(1.0, trial))
        # Bisection keeps one endpoint strictly inside the acceptance set
        # and the other outside, regardless of which side the boundary is on.
        for _ in range(80):
            middle = (inside_pt + outside_pt) / 2.0
            if accepted(middle):
                inside_pt = middle
            else:
                outside_pt = middle
        return max(0.0, min(1.0, (inside_pt + outside_pt) / 2.0))

    lower = 0.0 if successes == 0 else root(mle, -1)
    upper = 1.0 if successes == trials else root(mle, 1)
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
        research_plans: dict[str, dict[str, Any]] | None = None,
        research_plan_order: list[str] | None = None,
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
        self.research_plans = research_plans or {}
        self.research_plan_order = research_plan_order or []

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
        completed_before = int(branch["completed_shots"])
        passed_before = int(branch.get("passed_shots", completed_before))
        counts_before = dict(branch["counts"])
        for _ in range(shots):
            result = quantum.sample_continuation(
                state, self.prefix["qubits"], branch["steps"], rng
            )
            branch["completed_shots"] += 1
            if _passes_filters(filters, result["measurements"]):
                branch["passed_shots"] += 1
                branch["counts"][result["sample"]] += 1
        counts_after = dict(branch["counts"])
        passed_after = int(branch["passed_shots"])
        completed_after = int(branch["completed_shots"])
        branch["budget"] = max(int(branch["budget"]), branch["completed_shots"])
        branch["rng_state"] = rng_to_json(rng)
        batch = {
            "batch_id": uuid.uuid4().hex,
            "source": source,
            "shots": shots,
            "total_shots_after_batch": branch["completed_shots"],
            "completed_shots_before_batch": completed_before,
            "passed_shots_before_batch": passed_before,
            "counts_before_batch": counts_before,
            "completed_shots_after_batch": completed_after,
            "passed_shots_after_batch": passed_after,
            "counts_after_batch": counts_after,
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
        return self._branch_uncertainty_from_counts(
            branch_id,
            event,
            alpha,
            attempts=int(branch["completed_shots"]),
            passed=int(branch.get("passed_shots", int(branch["completed_shots"]))),
            counts=dict(branch["counts"]),
        )

    def _branch_uncertainty_from_counts(
        self,
        branch_id: str,
        event: str,
        alpha: float,
        *,
        attempts: int,
        passed: int,
        counts: dict[str, int],
    ) -> dict[str, Any]:
        branch = self.branches[branch_id]
        shots = int(attempts)
        passed = int(passed)
        filtered = bool(branch.get("filters"))
        pass_interval = beta_mixture_interval(passed, shots, alpha / 2) if filtered else None
        target_successes = passed if event == PASS_EVENT else int(counts[event])
        target_trials = shots if event == PASS_EVENT or not filtered else passed
        target_interval = beta_mixture_interval(target_successes, target_trials, alpha / 2)
        target_width = _uncertainty_width(target_interval)
        score = target_width
        if filtered and event != PASS_EVENT:
            score += _uncertainty_width(pass_interval or (0.0, 1.0))
        if filtered and event != PASS_EVENT and passed == 0:
            evidence_note = "筛选尚未命中；接受率和接受后分布均缺少证据"
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

    def _evaluate_inquiry(
        self,
        inquiry: dict[str, Any],
        *,
        used_override: int | None = None,
        stats_override: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        alpha = float(inquiry["alpha"])
        margin = float(inquiry["margin"])
        stats = stats_override or [
            self._branch_uncertainty(branch_id, inquiry["event"], alpha)
            for branch_id in inquiry["branch_ids"]
        ]
        used = self._inquiry_used_budget(inquiry) if used_override is None else int(used_override)
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
            "cs_rule_version": CS_RULE_VERSION,
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
            "mode": mode,
            "branch_ids": list(selected),
            "event": event,
            "margin": float(margin),
            "total_budget": int(total_budget),
            "batch_size": int(batch_size),
            "alpha": float(alpha),
            "cs_rule_version": CS_RULE_VERSION,
            "cs_rule_note": CS_RULE_NOTE,
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

    @staticmethod
    def normalize_research_plan(raw_plan: dict[str, Any], branch_ids: list[str], known_events: set[str]) -> dict[str, Any]:
        total_budget = raw_plan.get("total_budget")
        if not isinstance(total_budget, int) or isinstance(total_budget, bool) or not 1 <= total_budget <= 1_000_000:
            raise ValueError("research plan total_budget must be an integer in [1, 1000000]")
        strategy = raw_plan.get("strategy", "balanced")
        if strategy not in ("balanced", "priority"):
            raise ValueError("research plan strategy must be 'balanced' or 'priority'")
        raw_goals = raw_plan.get("goals")
        if not isinstance(raw_goals, list) or not raw_goals:
            raise ValueError("research plan must contain at least one goal")
        goals: dict[str, dict[str, Any]] = {}
        goal_order: list[str] = []
        for index, raw_goal in enumerate(raw_goals):
            goal_id = str(raw_goal.get("goal_id") or f"goal-{index + 1}")
            if goal_id in goals:
                raise ValueError("research goal ids must be unique")
            inquiry = BranchExperiment.normalize_inquiry(
                {**raw_goal, "total_budget": total_budget},
                branch_ids,
                known_events,
            )
            batch_size = inquiry["batch_size"]
            priority = raw_goal.get("priority", index)
            if not isinstance(priority, int) or isinstance(priority, bool) or priority < 0:
                raise ValueError("goal priority must be a non-negative integer")
            goal = {
                "goal_id": goal_id,
                "label": str(raw_goal.get("label") or goal_id),
                "priority": priority,
                "batch_size": batch_size,
                "status": "active",
                "adopted_batch_ids": [],
                "decisions": [],
                **{key: value for key, value in inquiry.items() if key not in ("inquiry_id", "decisions", "status")},
            }
            goals[goal_id] = goal
            goal_order.append(goal_id)

        priority_order = list(raw_plan.get("priority_goal_ids", []))
        if strategy == "priority":
            if not priority_order:
                priority_order = [goal_order[0]]
            if any(goal_id not in goals for goal_id in priority_order):
                raise ValueError("priority_goal_ids must reference goals in this plan")
        elif priority_order and any(goal_id not in goals for goal_id in priority_order):
            raise ValueError("priority_goal_ids must reference goals in this plan")

        return {
            "plan_id": uuid.uuid4().hex,
            "created_at": time.time(),
            "name": str(raw_plan.get("name") or "sequential-research-plan"),
            "strategy": strategy,
            "total_budget": int(total_budget),
            "executed_shots": 0,
            "priority_goal_ids": list(dict.fromkeys(priority_order)),
            "goal_order": goal_order,
            "goals": goals,
            "batches": [],
        }

    def add_research_plan(self, raw_plan: dict[str, Any]) -> dict[str, Any]:
        events = set(next(iter(self.branches.values()))["counts"].keys())
        plan = self.normalize_research_plan(raw_plan, list(self.branch_order), events)
        self.research_plans[plan["plan_id"]] = plan
        self.research_plan_order.append(plan["plan_id"])
        for goal_id in plan["goal_order"]:
            goal = plan["goals"][goal_id]
            self._record_goal_decision(
                plan,
                goal,
                action="created",
                branch_id=None,
                shots=0,
                reason="研究目标已登记；停止规则和置信依据独立保留，样本仅按相同分支分布共享",
            )
            adopted = self._adopt_existing_batches(plan, goal)
            if adopted:
                self._record_goal_decision(
                    plan,
                    goal,
                    action="adopt_existing_samples",
                    branch_id=None,
                    shots=sum(item["shots"] for item in adopted),
                    reason=f"采用登记前同分布样本 {len(adopted)} 批；不重复扣减研究计划执行预算",
                    batch_ids=[item["batch_id"] for item in adopted],
                )
        self.save()
        return self.research_plan_snapshot(plan["plan_id"])

    def _adopt_existing_batches(self, plan: dict[str, Any], goal: dict[str, Any]) -> list[dict[str, Any]]:
        adopted: list[dict[str, Any]] = []
        for branch_id in goal["branch_ids"]:
            for batch in self.branches[branch_id]["batches"]:
                batch_id = batch["batch_id"]
                if batch_id not in goal["adopted_batch_ids"]:
                    goal["adopted_batch_ids"].append(batch_id)
                    adopted.append({**batch, "branch_id": branch_id})
        return adopted

    def _record_goal_decision(
        self,
        plan: dict[str, Any],
        goal: dict[str, Any],
        *,
        action: str,
        branch_id: str | None,
        shots: int,
        reason: str,
        batch_id: str | None = None,
        batch_ids: list[str] | None = None,
        evaluation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = {
            "decision_id": uuid.uuid4().hex,
            "at": time.time(),
            "action": action,
            "branch_id": branch_id,
            "shots": int(shots),
            "reason": reason,
            "batch_id": batch_id,
            "batch_ids": list(batch_ids or ([] if batch_id is None else [batch_id])),
            "evaluation_after": evaluation,
        }
        goal.setdefault("decisions", []).append(entry)
        return entry

    def _branch_batches_by_id(self) -> dict[str, dict[str, Any]]:
        return {
            batch["batch_id"]: {**batch, "branch_id": branch_id}
            for branch_id in self.branch_order
            for batch in self.branches[branch_id]["batches"]
        }

    def _goal_branch_counts(self, goal: dict[str, Any]) -> dict[str, dict[str, Any]]:
        all_batches = self._branch_batches_by_id()
        result: dict[str, dict[str, Any]] = {}
        empty_counts = {label: 0 for label in next(iter(self.branches.values()))["counts"]}
        for branch_id in goal["branch_ids"]:
            selected = [all_batches[batch_id] for batch_id in goal["adopted_batch_ids"] if all_batches.get(batch_id, {}).get("branch_id") == branch_id]
            attempts = sum(int(batch.get("shots", 0)) for batch in selected)
            counts = dict(empty_counts)
            passed = 0
            legacy = [batch for batch in selected if "counts_after_batch" not in batch]
            recorded = [batch for batch in selected if "counts_after_batch" in batch]
            for batch in recorded:
                for label in counts:
                    counts[label] += max(
                        0,
                        int(batch["counts_after_batch"].get(label, 0))
                        - int(batch.get("counts_before_batch", {}).get(label, 0)),
                    )
                passed += max(
                    0,
                    int(batch["passed_shots_after_batch"])
                    - int(batch.get("passed_shots_before_batch", 0)),
                )
            if legacy:
                branch = self.branches[branch_id]
                replay_rng = np.random.default_rng(_branch_seed(self.base_seed, branch_id))
                replay_state = _from_json(self.prefix["state"])
                legacy_limit = max(int(batch["total_shots_after_batch"]) for batch in legacy)
                for shot_number in range(1, legacy_limit + 1):
                    result = quantum.sample_continuation(
                        replay_state, self.prefix["qubits"], branch["steps"], replay_rng
                    )
                    is_selected_legacy = any(
                        shot_number <= int(batch["total_shots_after_batch"])
                        and shot_number > int(batch["total_shots_after_batch"]) - int(batch["shots"])
                        for batch in legacy
                    )
                    if is_selected_legacy and _passes_filters(branch.get("filters", []), result["measurements"]):
                        passed += 1
                        counts[result["sample"]] += 1
            result[branch_id] = {"attempts": attempts, "passed": passed, "counts": counts}
        return result

    def _goal_statistics(self, goal: dict[str, Any]) -> list[dict[str, Any]]:
        observed = self._goal_branch_counts(goal)
        stats = []
        for branch_id in goal["branch_ids"]:
            entry = observed[branch_id]
            stats.append(
                self._branch_uncertainty_from_counts(
                    branch_id,
                    goal["event"],
                    float(goal["alpha"]),
                    attempts=entry["attempts"],
                    passed=entry["passed"],
                    counts=entry["counts"],
                )
            )
        return stats

    def _evaluate_research_goal(self, plan: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
        template = {key: value for key, value in goal.items() if key in ("mode", "branch_ids", "event", "margin", "alpha", "total_budget")}
        stats = self._goal_statistics(goal)
        adopted_shots = sum(
            int(self._branch_batches_by_id().get(batch_id, {}).get("shots", 0))
            for batch_id in goal["adopted_batch_ids"]
        )
        evaluation = self._evaluate_inquiry(
            {**template, "decisions": []},
            used_override=0,
            stats_override=stats,
        )
        if goal["status"] in ("decided", "cancelled", "inconclusive_budget_exhausted"):
            evaluation["status"] = goal["status"]
            if goal["status"] == "inconclusive_budget_exhausted" and evaluation.get("conclusion") is None:
                evaluation["conclusion"] = {
                    "type": "inconclusive",
                    "message": "共享执行预算已用完；该目标精度尚未满足，结果保持未确定",
                }
        elif plan["executed_shots"] >= plan["total_budget"] and evaluation["status"] == "continue":
            evaluation["status"] = "inconclusive_budget_exhausted"
            evaluation["conclusion"] = {
                "type": "inconclusive",
                "message": "共享执行预算已用完；该目标精度尚未满足，结果保持未确定",
            }
        evaluation["branch_statistics"] = stats
        evaluation["adopted_shots"] = adopted_shots
        return evaluation

    def _plan_batch_by_id(self, plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {entry["batch_id"]: entry for entry in plan.get("batches", [])}

    def _active_goals(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        return [plan["goals"][goal_id] for goal_id in plan["goal_order"] if plan["goals"][goal_id]["status"] == "active"]

    def _goal_score(self, plan: dict[str, Any], goal: dict[str, Any]) -> float:
        evaluation = self._evaluate_research_goal(plan, goal)
        return max(float(stat["allocation_score"]) for stat in evaluation["branch_statistics"])

    def _next_leader(self, plan: dict[str, Any]) -> dict[str, Any] | None:
        active = self._active_goals(plan)
        if not active:
            return None
        plan_batches = self._plan_batch_by_id(plan)

        def planned_batches(goal: dict[str, Any]) -> int:
            return sum(1 for batch_id in goal["adopted_batch_ids"] if batch_id in plan_batches)

        if plan["strategy"] == "priority":
            priority = list(plan["priority_goal_ids"]) + [
                goal_id for goal_id in plan["goal_order"] if goal_id not in plan["priority_goal_ids"]
            ]
            ordered = [plan["goals"][goal_id] for goal_id in priority if plan["goals"][goal_id]["status"] == "active"]
        else:
            minimum_planned = min(planned_batches(goal) for goal in active)
            ordered = [goal for goal in active if planned_batches(goal) == minimum_planned]
            return max(ordered, key=lambda goal: (self._goal_score(plan, goal), -int(goal["priority"]), goal["goal_id"]))
        return ordered[0]

    def _choose_branch(self, plan: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
        stats = self._evaluate_research_goal(plan, goal)["branch_statistics"]
        return max(stats, key=lambda stat: (float(stat["allocation_score"]), stat["branch_id"]))

    def run_research_batch(self, plan_id: str, *, shots_override: int | None = None) -> dict[str, Any]:
        if plan_id not in self.research_plans:
            raise KeyError(f"unknown research plan: {plan_id}")
        plan = self.research_plans[plan_id]
        remaining = int(plan["total_budget"]) - int(plan["executed_shots"])
        if remaining <= 0:
            raise ValueError("shared research budget is exhausted")
        before = {goal["goal_id"]: self._evaluate_research_goal(plan, goal) for goal in self._active_goals(plan)}
        for goal in self._active_goals(plan):
            if before[goal["goal_id"]]["status"] == "decided":
                goal["status"] = "decided"
                self._record_goal_decision(
                    plan, goal, action="stop", branch_id=None, shots=0,
                    reason=before[goal["goal_id"]]["conclusion"]["message"],
                    evaluation=before[goal["goal_id"]],
                )
        leader = self._next_leader(plan)
        if leader is None:
            self.save()
            return {"status": "complete", **self.research_plan_snapshot(plan_id)}
        chosen = self._choose_branch(plan, leader)
        requested = int(shots_override if shots_override is not None else leader["batch_size"])
        if requested <= 0:
            raise ValueError("batch shots must be positive")
        shots = min(requested, remaining)
        participants = [
            goal for goal in self._active_goals(plan)
            if chosen["branch_id"] in goal["branch_ids"]
        ]
        batch = self._sample(
            self.branches[chosen["branch_id"]],
            shots,
            source=f"research-plan:{plan_id}",
        )
        plan["executed_shots"] += shots
        plan_entry = {
            "batch_id": batch["batch_id"],
            "branch_id": chosen["branch_id"],
            "shots": shots,
            "leader_goal_id": leader["goal_id"],
            "goal_ids": [goal["goal_id"] for goal in participants],
        }
        plan["batches"].append(plan_entry)
        for goal in participants:
            goal["adopted_batch_ids"].append(batch["batch_id"])
            self._record_goal_decision(
                plan,
                goal,
                action="sample",
                branch_id=chosen["branch_id"],
                shots=shots,
                reason=(
                    f"同一实际批次服务 {len(participants)} 个同分布目标，共享执行预算只扣减 {shots} 次；"
                    f"选择依据：{chosen['evidence_note']}"
                ),
                batch_id=batch["batch_id"],
            )

        for goal in self._active_goals(plan):
            evaluation = self._evaluate_research_goal(plan, goal)
            if evaluation["status"] == "decided":
                goal["status"] = "decided"
                self._record_goal_decision(
                    plan, goal, action="stop", branch_id=None, shots=0,
                    reason=evaluation["conclusion"]["message"], evaluation=evaluation,
                )
        if int(plan["executed_shots"]) >= int(plan["total_budget"]):
            for goal in self._active_goals(plan):
                evaluation = self._evaluate_research_goal(plan, goal)
                goal["status"] = "inconclusive_budget_exhausted"
                self._record_goal_decision(
                    plan, goal, action="stop", branch_id=None, shots=0,
                    reason=evaluation["conclusion"]["message"], evaluation=evaluation,
                )
        self.save()
        return self.research_plan_snapshot(plan_id)

    def cancel_research_goal(self, plan_id: str, goal_id: str, *, reason: str = "研究目标已取消") -> dict[str, Any]:
        if plan_id not in self.research_plans:
            raise KeyError(f"unknown research plan: {plan_id}")
        plan = self.research_plans[plan_id]
        if goal_id not in plan["goals"]:
            raise KeyError(f"unknown research goal: {goal_id}")
        goal = plan["goals"][goal_id]
        if goal["status"] == "active":
            goal["status"] = "cancelled"
            self._record_goal_decision(
                plan, goal, action="cancel", branch_id=None, shots=0,
                reason=reason + "；其他目标已采用的实际样本保留且不回退执行预算",
            )
        self.save()
        return self.research_plan_snapshot(plan_id)

    def research_goal_snapshot(self, plan: dict[str, Any], goal_id: str) -> dict[str, Any]:
        goal = plan["goals"][goal_id]
        evaluation = self._evaluate_research_goal(plan, goal)
        return {
            **{key: value for key, value in goal.items() if key not in ("decisions",)},
            "current_evaluation": evaluation,
            "decisions": list(goal.get("decisions", [])),
        }

    def research_plan_snapshot(self, plan_id: str) -> dict[str, Any]:
        plan = self.research_plans[plan_id]
        goals = [self.research_goal_snapshot(plan, goal_id) for goal_id in plan["goal_order"]]
        active = [goal for goal in goals if goal["status"] == "active"]
        next_batch = None
        remaining = int(plan["total_budget"]) - int(plan["executed_shots"])
        if active and remaining > 0:
            leader = self._next_leader(plan)
            if leader is not None:
                chosen = self._choose_branch(plan, leader)
                next_batch = {
                    "leader_goal_id": leader["goal_id"],
                    "branch_id": chosen["branch_id"],
                    "shots": min(int(leader["batch_size"]), remaining),
                    "eligible_goal_ids": [
                        goal["goal_id"] for goal in active if chosen["branch_id"] in goal["branch_ids"]
                    ],
                }
        return {
            "plan_id": plan["plan_id"],
            "created_at": plan["created_at"],
            "name": plan["name"],
            "strategy": plan["strategy"],
            "total_budget": plan["total_budget"],
            "executed_shots": plan["executed_shots"],
            "remaining_budget": remaining,
            "priority_goal_ids": list(plan["priority_goal_ids"]),
            "goal_order": list(plan["goal_order"]),
            "goals": goals,
            "batches": list(plan["batches"]),
            "next_batch": next_batch,
        }

    def run_adaptive_batch(self, inquiry_id: str, *, batch_size: int | None = None) -> dict[str, Any]:
        if inquiry_id not in self.inquiries:
            raise KeyError(f"unknown inquiry: {inquiry_id}")
        inquiry = self.inquiries[inquiry_id]
        if inquiry["status"] != "active":
            raise ValueError("this inquiry has already stopped")
        if inquiry.get("cs_rule_version") == LEGACY_CS_RULE_VERSION:
            # First continuation after the fix: switch the *future* stopping
            # rule to v2, but never rewrite historical counts or evaluations.
            inquiry["cs_rule_version"] = CS_RULE_VERSION
            self._record_inquiry_decision(
                inquiry,
                action="rule_migrated",
                branch_id=None,
                shots=0,
                reason=(
                    "该研究创建于修正前（v1，上界可能坍缩到点估计）；"
                    "原始批次、计数与历史评估保持只读不变。"
                    "自本次继续起，停止与区间判断改用 " + CS_RULE_NOTE
                ),
            )
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
        return {
            **{key: value for key, value in inquiry.items() if key != "decisions"},
            "current_evaluation": evaluation,
            "decisions": list(inquiry.get("decisions", [])),
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
            "research_plan_order": list(self.research_plan_order),
            "research_plans": {key: _jsonable(value) for key, value in self.research_plans.items()},
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
            "research_plans": [
                self.research_plan_snapshot(plan_id) for plan_id in self.research_plan_order
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], storage_dir: str | Path) -> "BranchExperiment":
        for branch in data["branches"].values():
            branch.setdefault("filters", [])
            branch.setdefault("passed_shots", int(branch.get("completed_shots", 0)))
        inquiries = data.get("inquiries", {})
        for inquiry in inquiries.values():
            # Records created before the bisection fix have no rule tag.
            # Keep their stored counts and decision evaluations untouched;
            # only annotate which rule produced them so history stays honest.
            inquiry.setdefault("cs_rule_version", LEGACY_CS_RULE_VERSION)
            inquiry.setdefault(
                "cs_rule_note",
                "历史记录：采用修正前的 beta-mixture-cs-v1 规则计算；"
                "该规则上界二分方向反转，上界会坍缩到点估计（如 80/100 上界=0.8）。"
                "原始事件、计数与当时评估保持不变；继续追加样本时改用 v2 并另行记录",
            )
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
            research_plans=data.get("research_plans", {}),
            research_plan_order=list(data.get("research_plan_order", [])),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BranchExperiment":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle), path.parent)
