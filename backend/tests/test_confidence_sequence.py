"""Direct checks for the Beta(1,1)-mixture confidence sequence.

The boundary is a likelihood-ratio root; the tests cross-check it against an
independent grid-then-bisection solver and against enumerable short circuits
where the true probabilities are known exactly.
"""

import math

import pytest

from backend import branches, quantum, server


def _reference_interval(successes, trials, alpha):
    if trials == 0:
        return 0.0, 1.0
    log_marginal = (
        math.lgamma(successes + 1)
        + math.lgamma(trials - successes + 1)
        - math.lgamma(trials + 2)
    )
    threshold = log_marginal - math.log(1.0 / alpha)

    def log_likelihood(p):
        if p <= 0.0 or p >= 1.0:
            return float("-inf")
        return successes * math.log(p) + (trials - successes) * math.log(1.0 - p)

    lower, upper = 0.0, 1.0
    grid = [i / 200_000 for i in range(1, 200_000)]
    inside = [p for p in grid if log_likelihood(p) >= threshold]
    if successes > 0:
        x = min(inside)
        a, b = max(0.0, x - 5e-6), x
        for _ in range(80):
            m = (a + b) / 2
            if log_likelihood(m) >= threshold:
                b = m
            else:
                a = m
        lower = (a + b) / 2
    if successes < trials:
        x = max(inside)
        a, b = x, min(1.0, x + 5e-6)
        for _ in range(80):
            m = (a + b) / 2
            if log_likelihood(m) >= threshold:
                a = m
            else:
                b = m
        upper = (a + b) / 2
    return lower, upper


@pytest.mark.parametrize(
    "successes,trials,alpha",
    [
        (80, 100, 0.05),      # reported bug: upper bound collapsed to 0.80
        (0, 100, 0.05),       # all failures
        (100, 100, 0.05),     # all successes
        (1, 10, 0.025),       # tiny counts
        (2, 3, 0.05),
        (5, 100_000, 0.05),   # rare event
        (1, 1_000_000, 0.025),
        (0, 100_000, 0.05),   # zero hits, large sample
        (999, 1000, 0.05),
        (50, 50, 0.05),
    ],
)
def test_interval_matches_independent_numeric_reference(successes, trials, alpha):
    lower, upper = branches.beta_mixture_interval(successes, trials, alpha)
    ref_lower, ref_upper = _reference_interval(successes, trials, alpha)
    assert lower == pytest.approx(ref_lower, abs=3e-4)
    assert upper == pytest.approx(ref_upper, abs=3e-4)
    mle = successes / trials
    assert lower <= mle <= upper


def test_80_of_100_upper_bound_contains_085():
    lower, upper = branches.beta_mixture_interval(80, 100, 0.05)
    assert lower < 0.8 < upper
    assert upper > 0.85  # likelihood of p=0.85 stays above the mixture threshold
    assert upper == pytest.approx(0.9072, abs=1e-3)


def test_empty_and_endpoint_intervals():
    assert branches.beta_mixture_interval(0, 0, 0.05) == (0.0, 1.0)
    lower, upper = branches.beta_mixture_interval(0, 50, 0.05)
    assert lower == 0.0 and 0.0 < upper < 1.0  # zero hits: weak evidence, not p=0
    lower, upper = branches.beta_mixture_interval(50, 50, 0.05)
    assert 0.0 < lower < 1.0 and upper == 1.0
    with pytest.raises(ValueError):
        branches.beta_mixture_interval(2, 1, 0.05)
    with pytest.raises(ValueError):
        branches.beta_mixture_interval(1, 2, 0.0)


def test_enumerable_short_circuit_true_probabilities_stay_covered(tmp_path):
    # Enumerable short circuit: exact amplitudes give the true probabilities.
    result = quantum.run_experiment(
        1, "0", [{"type": "H", "target": 0}], 4000, 11
    )
    zero_hits = int(result["counts"].get("0", 0))
    lower, upper = branches.beta_mixture_interval(zero_hits, 4000, 0.05 / 2)
    assert lower <= 0.5 <= upper

    # RX(pi/2) on |0>: P(0)=cos(pi/4)^2=0.5 as well; verify a biased gate.
    biased = quantum.run_experiment(
        1, "0", [{"type": "RX", "target": 0, "angle": 1.0}], 4000, 12
    )
    p_zero = math.cos(0.5) ** 2
    hits = int(biased["counts"].get("0", 0))
    lower, upper = branches.beta_mixture_interval(hits, 4000, 0.05 / 2)
    assert lower <= p_zero <= upper


def _make_experiment(tmp_path, seed=21):
    checkpoint = server.Session(
        {"qubits": 1, "initial_basis": "0", "seed": 5, "steps": []}
    ).save_checkpoint("prefix", "cp-cs-rules")
    return branches.BranchExperiment.create(
        checkpoint,
        [{"branch_id": "only", "label": "only", "steps": [], "budget": 0}],
        storage_dir=tmp_path,
        base_seed=seed,
        experiment_id="rules",
    )


def test_new_inquiries_are_tagged_with_fixed_rule(tmp_path):
    experiment = _make_experiment(tmp_path)
    inquiry = experiment.add_inquiry(
        {"mode": "single", "branch_ids": ["only"], "event": "0",
         "margin": 0.05, "total_budget": 100, "batch_size": 25}
    )
    assert inquiry["cs_rule_version"] == branches.CS_RULE_VERSION
    evaluation = experiment.run_adaptive_batch(inquiry["inquiry_id"])
    assert evaluation["cs_rule_version"] == branches.CS_RULE_VERSION


def test_legacy_inquiry_keeps_history_and_migrates_only_future_judgments(tmp_path):
    import json

    experiment = _make_experiment(tmp_path)
    inquiry = experiment.add_inquiry(
        {"mode": "single", "branch_ids": ["only"], "event": "0",
         "margin": 0.05, "total_budget": 100, "batch_size": 25}
    )
    inquiry_id = inquiry["inquiry_id"]
    experiment.run_adaptive_batch(inquiry_id)

    # Simulate an on-disk record written by the pre-fix implementation.
    raw = json.loads(experiment.path.read_text(encoding="utf-8"))
    stored = raw["inquiries"][inquiry_id]
    stored.pop("cs_rule_version")
    stored.pop("cs_rule_note")
    shots_before = raw["branches"]["only"]["completed_shots"]
    decisions_before = len(stored["decisions"])
    experiment.path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    reopened = branches.BranchExperiment.load(experiment.path)
    loaded = reopened.inquiry_snapshot(inquiry_id)
    assert loaded["cs_rule_version"] == branches.LEGACY_CS_RULE_VERSION
    assert loaded["current_evaluation"]["cs_rule_version"] == branches.CS_RULE_VERSION

    reopened.run_adaptive_batch(inquiry_id)
    migrated = json.loads(reopened.path.read_text(encoding="utf-8"))
    record = migrated["inquiries"][inquiry_id]
    assert record["cs_rule_version"] == branches.CS_RULE_VERSION
    actions = [entry["action"] for entry in record["decisions"]]
    assert "rule_migrated" in actions
    assert len(record["decisions"]) == decisions_before + 2  # migration + sample/stop
    # Original counts are preserved; the new batch only appends.
    assert migrated["branches"]["only"]["completed_shots"] >= shots_before


def test_zero_hits_remains_inconclusive_for_conditional_distribution(tmp_path):
    checkpoint = server.Session(
        {"qubits": 1, "initial_basis": "0", "seed": 7, "steps": []}
    ).save_checkpoint("prefix", "cp-cs-zero")
    experiment = branches.BranchExperiment.create(
        checkpoint,
        [{
            "branch_id": "never", "label": "never",
            "steps": [{"type": "MEASURE", "target": 0}],
            "filters": [{"step": 0, "outcome": 1}], "budget": 0,
        }],
        storage_dir=tmp_path,
        base_seed=31,
        experiment_id="zero-hits",
    )
    inquiry = experiment.add_inquiry(
        {"mode": "single", "branch_ids": ["never"], "event": "0",
         "margin": 0.01, "total_budget": 10, "batch_size": 2}
    )
    for _ in range(5):
        evaluation = experiment.run_adaptive_batch(inquiry["inquiry_id"])
    stats = evaluation["branch_statistics"][0]
    assert stats["event_trials"] == 0
    assert stats["event_interval"] == [0.0, 1.0]  # no conditional evidence
    assert stats["pass_interval"][1] > 0.0       # acceptance rate uncertainty remains
    assert "接受率和接受后分布均缺少证据" in evaluation["branch_statistics"][0]["evidence_note"]
