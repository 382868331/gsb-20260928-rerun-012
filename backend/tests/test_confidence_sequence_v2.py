"""Regression tests for the corrected Beta-mixture confidence sequence."""

import json
import math

import pytest
from scipy import optimize

from backend import branches, quantum, server
from backend.branches import (
    LEGACY_RULE_DESCRIPTION,
    LEGACY_RULE_VERSION,
    RULE_DESCRIPTION,
    RULE_VERSION,
    beta_mixture_interval,
)


def _likelihood_threshold(successes, trials, alpha):
    log_marginal = (
        math.lgamma(successes + 1)
        + math.lgamma(trials - successes + 1)
        - math.lgamma(trials + 2)
    )
    return math.exp(log_marginal - math.log(1.0 / alpha))


def test_upper_bound_includes_085_for_80_successes_out_of_100():
    alpha = 0.05
    lower, upper = beta_mixture_interval(80, 100, alpha)
    mle = 0.8
    assert lower < mle < upper
    # Independent numeric cross-check of both likelihood-ratio roots.
    threshold = _likelihood_threshold(80, 100, alpha)

    def log_ell(probability):
        return 80 * math.log(probability) + 20 * math.log(1 - probability)

    expected_lower = optimize.brentq(lambda p: math.exp(log_ell(p)) - threshold, 1e-12, mle)
    expected_upper = optimize.brentq(lambda p: math.exp(log_ell(p)) - threshold, mle, 1 - 1e-12)
    assert abs(lower - expected_lower) < 1e-9
    assert abs(upper - expected_upper) < 1e-9
    # The reported defect: 0.85 must still be inside at this alpha.
    assert math.exp(log_ell(0.85)) >= threshold
    assert lower <= 0.85 <= upper
    assert upper > 0.9


def test_interval_boundary_cases_all_success_all_failure_no_trials():
    assert beta_mixture_interval(0, 0, 0.05) == (0.0, 1.0)

    lower, upper = beta_mixture_interval(10, 10, 0.05)
    assert upper == 1.0
    assert 0.0 < lower < 1.0
    # log_marginal = 1/(t+1), so p^t = alpha/(t+1) at the root.
    expected_lower = (0.05 / 11) ** (1 / 10)
    assert abs(lower - expected_lower) < 1e-9

    lower, upper = beta_mixture_interval(0, 10, 0.05)
    assert lower == 0.0
    assert 0.0 < upper < 1.0
    expected_upper = 1 - (0.05 / 11) ** (1 / 10)
    assert abs(upper - expected_upper) < 1e-9

    # Single-trial edges never collapse to the observed frequency;
    # the marginal at t=1 is 1/2, so the threshold is alpha/2.
    assert beta_mixture_interval(1, 1, 0.05) == pytest.approx((0.025, 1.0))
    assert beta_mixture_interval(0, 1, 0.05) == pytest.approx((0.0, 0.975))

    # Extremely small roots do not get clipped at the bisection endpoint.
    lower, _ = beta_mixture_interval(1, 10000, 0.05)
    assert lower < 1e-6


def test_interval_containment_is_checked_against_numeric_grid():
    alpha = 0.025
    for successes, trials in ((1, 3), (2, 5), (7, 13), (50, 100), (95, 100), (3, 1000)):
        lower, upper = beta_mixture_interval(successes, trials, alpha)
        threshold = _likelihood_threshold(successes, trials, alpha)

        def ell(probability):
            return math.exp(
                successes * math.log(probability)
                + (trials - successes) * math.log(1 - probability)
            )

        assert ell(lower) / threshold == pytest.approx(1.0, rel=1e-7)
        assert ell(upper) / threshold == pytest.approx(1.0, rel=1e-7)
        for probability in (lower, (lower + upper) / 2, upper, successes / trials):
            assert ell(probability) >= threshold * (1 - 1e-9)
        assert ell(max(1e-12, lower * 0.5)) < threshold or successes == 0
        assert ell(min(1 - 1e-12, 1 - (1 - upper) * 0.5)) < threshold or successes == trials


def _checkpoint(steps=None, qubits=1, basis="0", seed=17):
    session = server.Session(
        {"qubits": qubits, "initial_basis": basis, "seed": seed, "steps": steps or []}
    )
    for _ in steps or []:
        session.step()
    return session.save_checkpoint("prefix", "cp-v2")


def test_new_inquiries_record_corrected_rule_and_reload_keeps_counts(tmp_path):
    checkpoint = _checkpoint()
    specs = [{"branch_id": "only", "label": "only", "steps": [], "budget": 0}]
    experiment = branches.BranchExperiment.create(
        checkpoint, specs, storage_dir=tmp_path, base_seed=11, experiment_id="v2"
    )
    snapshot = experiment.add_inquiry(
        {"mode": "single", "branch_ids": ["only"], "event": "0",
         "margin": 0.02, "total_budget": 200, "batch_size": 50}
    )
    inquiry_id = snapshot["inquiry_id"]
    assert snapshot["rule_version"] == RULE_VERSION
    assert RULE_DESCRIPTION == snapshot["rule_description"]
    experiment.run_adaptive_batch(inquiry_id)
    experiment.run_adaptive_batch(inquiry_id)

    raw = json.loads(experiment.path.read_text(encoding="utf-8"))
    stored = raw["inquiries"][inquiry_id]
    shots_after = raw["branches"]["only"]["completed_shots"]
    assert shots_after == 100
    assert stored["rule_version"] == RULE_VERSION

    # Reopen: counts, batches and rule provenance survive, sampling resumes.
    reopened = branches.BranchExperiment.load(experiment.path)
    reopened.run_adaptive_batch(inquiry_id)
    reloaded = json.loads(experiment.path.read_text(encoding="utf-8"))
    assert reloaded["branches"]["only"]["completed_shots"] == 150
    total_batch_shots = sum(
        batch["shots"] for batch in reloaded["branches"]["only"]["batches"]
    )
    assert total_batch_shots == 150
    resumed = reopened.inquiry_snapshot(inquiry_id)
    assert resumed["rule_version"] == RULE_VERSION
    assert resumed["rule_note"] is None


def test_legacy_inquiry_is_distinguished_and_events_are_not_mutated(tmp_path):
    checkpoint = _checkpoint()
    specs = [{"branch_id": "only", "label": "only", "steps": [], "budget": 0}]
    experiment = branches.BranchExperiment.create(
        checkpoint, specs, storage_dir=tmp_path, base_seed=12, experiment_id="legacy"
    )
    inquiry_id = experiment.add_inquiry(
        {"mode": "single", "branch_ids": ["only"], "event": "0",
         "margin": 0.02, "total_budget": 400, "batch_size": 100}
    )["inquiry_id"]
    experiment.run_adaptive_batch(inquiry_id)
    before_raw = experiment.path.read_text(encoding="utf-8")

    raw = json.loads(before_raw)
    stored = raw["inquiries"][inquiry_id]
    stored.pop("rule_version", None)
    stored.pop("rule_description", None)
    for decision in stored["decisions"]:
        decision.pop("rule_version", None)
        decision.pop("rule_description", None)
    experiment.path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    reopened = branches.BranchExperiment.load(experiment.path)
    legacy = reopened.inquiry_snapshot(inquiry_id)
    assert legacy["rule_version"] == LEGACY_RULE_VERSION
    assert legacy["rule_description"] == LEGACY_RULE_DESCRIPTION
    assert "v1" in legacy["rule_note"]
    assert all(decision["rule_version"] == LEGACY_RULE_VERSION for decision in legacy["decisions"])
    # Current recomputation uses the corrected rule while the stored events stay intact.
    assert legacy["current_evaluation"]["rule_version"] == RULE_VERSION
    on_disk = json.loads(experiment.path.read_text(encoding="utf-8"))
    stored_again = on_disk["inquiries"][inquiry_id]
    assert "rule_version" not in stored_again
    assert all("rule_version" not in decision for decision in stored_again["decisions"])

    # Continuing a legacy active inquiry leaves an explicit rule-switch audit entry.
    decisions_before = len(stored_again["decisions"])
    reopened.run_adaptive_batch(inquiry_id)
    switched = json.loads(experiment.path.read_text(encoding="utf-8"))
    stored_switched = switched["inquiries"][inquiry_id]
    assert stored_switched["rule_version"] == RULE_VERSION
    assert len(stored_switched["decisions"]) == decisions_before + 2
    assert "v2 修正规则" in stored_switched["decisions"][decisions_before]["reason"]
    assert switched["branches"]["only"]["completed_shots"] == 200


def test_enumerable_short_circuit_interval_covers_true_probabilities(tmp_path):
    # H on |0>: enumerable truth is P(0) = P(1) = 0.5 for every branch.
    checkpoint = _checkpoint()
    specs = [
        {"branch_id": "a", "label": "A", "steps": [{"type": "H", "target": 0}], "budget": 0},
        {"branch_id": "b", "label": "B", "steps": [{"type": "H", "target": 0}], "budget": 0},
    ]
    experiment = branches.BranchExperiment.create(
        checkpoint, specs, storage_dir=tmp_path, base_seed=21, experiment_id="enum"
    )
    inquiry_id = experiment.add_inquiry(
        {"mode": "compare", "branch_ids": ["a", "b"], "event": "0",
         "margin": 0.001, "total_budget": 80, "batch_size": 10}
    )["inquiry_id"]
    evidence = []
    while experiment.inquiry_snapshot(inquiry_id)["status"] == "active":
        evaluation = experiment.run_adaptive_batch(inquiry_id)
        stats = evaluation["branch_statistics"]
        for stat in stats:
            lower, upper = stat["event_interval"]
            if stat["event_trials"]:
                assert lower <= 0.5 <= upper
        evidence.append([
            (stat["event_successes"], stat["event_trials"], tuple(stat["event_interval"]))
            for stat in stats
        ])

    final = experiment.inquiry_snapshot(inquiry_id)
    # Equal probabilities and an exhausted budget must not name a winner.
    assert final["status"] == "inconclusive_budget_exhausted"
    assert final["current_evaluation"]["conclusion"]["type"] == "inconclusive"
    assert len(evidence) == 8
    assert any(any(trials for _, trials, _ in row) for row in evidence[:-1])
    stop_reasons = {entry["reason"] for entry in final["decisions"] if entry["action"] == "stop"}
    assert any("未确定" in reason for reason in stop_reasons)


def test_zero_hits_keeps_acceptance_and_conditional_uncertainty_separate(tmp_path):
    checkpoint = _checkpoint()
    specs = [
        {
            "branch_id": "rare",
            "label": "rare",
            "steps": [{"type": "MEASURE", "target": 0}],
            "filters": [{"step": 0, "outcome": 1}],
            "budget": 0,
        }
    ]
    experiment = branches.BranchExperiment.create(
        checkpoint, specs, storage_dir=tmp_path, base_seed=31, experiment_id="zero-hits"
    )
    inquiry_id = experiment.add_inquiry(
        {"mode": "single", "branch_ids": ["rare"], "event": "0",
         "margin": 0.02, "total_budget": 5, "batch_size": 5}
    )["inquiry_id"]
    # |0> with no gates never yields measurement outcome 1: the filter never hits.
    experiment.run_adaptive_batch(inquiry_id)
    evaluation = experiment.inquiry_snapshot(inquiry_id)["current_evaluation"]
    stat = evaluation["branch_statistics"][0]
    assert stat["attempts"] == 5
    assert stat["passed_shots"] == 0
    assert stat["event_trials"] == 0
    assert stat["event_interval"] == [0.0, 1.0]
    pass_lower, pass_upper = stat["pass_interval"]
    assert pass_lower == 0.0
    assert 0.0 < pass_upper < 1.0
    assert "接受率本身有 5 次尝试" in stat["evidence_note"]
    assert "条件分布没有证据" in stat["evidence_note"]
    branch = experiment.snapshot()["branches"][0]
    assert branch["frequencies"] is None
    assert branch["has_evidence"] is False
