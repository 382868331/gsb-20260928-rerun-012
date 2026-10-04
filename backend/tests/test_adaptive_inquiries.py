from backend import branches, server


def _checkpoint(steps=None, qubits=1, basis="0", seed=17):
    session = server.Session(
        {"qubits": qubits, "initial_basis": basis, "seed": seed, "steps": steps or []}
    )
    for _ in steps or []:
        session.step()
    return session.save_checkpoint("prefix", "cp-adaptive")


def test_confidence_sequence_narrows_and_remains_valid_for_repeated_looks(tmp_path):
    checkpoint = _checkpoint()
    specs = [{"branch_id": "zero", "label": "zero", "steps": [], "budget": 0}]
    experiment = branches.BranchExperiment.create(
        checkpoint, specs, storage_dir=tmp_path, base_seed=1, experiment_id="cs"
    )
    inquiry = experiment.add_inquiry(
        {"mode": "single", "branch_ids": ["zero"], "event": "0",
         "margin": 0.02, "total_budget": 2000, "batch_size": 100}
    )
    inquiry_id = inquiry["inquiry_id"]

    widths = []
    for _ in range(20):
        evaluation = experiment.run_adaptive_batch(inquiry_id)
        stats = experiment.inquiry_snapshot(inquiry_id)["current_evaluation"]
        lower, upper = stats["branch_statistics"][0]["event_interval"]
        widths.append(upper - lower)
        assert lower <= 1.0 <= upper
        if evaluation["status"] != "continue":
            break

    final = experiment.inquiry_snapshot(inquiry_id)
    assert final["status"] == "decided"
    assert final["current_evaluation"]["conclusion"]["type"] == "probability_estimated"
    assert widths[-1] <= 0.04
    actions = {entry["action"] for entry in final["decisions"]}
    assert {"created", "sample", "stop"} <= actions


def test_equal_probability_branches_stay_inconclusive_when_budget_cannot_separate(tmp_path):
    checkpoint = _checkpoint([{"type": "H", "target": 0}])
    specs = [
        {"branch_id": "a", "label": "A", "steps": [], "budget": 0},
        {"branch_id": "b", "label": "B", "steps": [], "budget": 0},
    ]
    experiment = branches.BranchExperiment.create(
        checkpoint, specs, storage_dir=tmp_path, base_seed=2, experiment_id="equal"
    )
    inquiry = experiment.add_inquiry(
        {"mode": "compare", "branch_ids": ["a", "b"], "event": "__pass__",
         "margin": 0.001, "total_budget": 120, "batch_size": 10}
    )
    inquiry_id = inquiry["inquiry_id"]

    while experiment.inquiry_snapshot(inquiry_id)["status"] == "active":
        experiment.run_adaptive_batch(inquiry_id)

    final = experiment.inquiry_snapshot(inquiry_id)
    assert final["status"] == "inconclusive_budget_exhausted"
    assert final["current_evaluation"]["conclusion"]["type"] == "inconclusive"
    assert final["current_evaluation"]["used_budget"] == 120


def test_rare_filter_allocates_to_branch_without_conditional_evidence(tmp_path):
    checkpoint = _checkpoint()
    specs = [
        {
            "branch_id": "rare",
            "label": "rare filter",
            "steps": [{"type": "MEASURE", "target": 0}],
            "filters": [{"step": 0, "outcome": 1}],
            "budget": 0,
        },
        {"branch_id": "certain", "label": "certain event", "steps": [], "budget": 0},
    ]
    experiment = branches.BranchExperiment.create(
        checkpoint, specs, storage_dir=tmp_path, base_seed=3, experiment_id="rare"
    )
    inquiry = experiment.add_inquiry(
        {"mode": "compare", "branch_ids": ["rare", "certain"], "event": "0",
         "margin": 0.01, "total_budget": 80, "batch_size": 1}
    )
    inquiry_id = inquiry["inquiry_id"]

    for _ in range(20):
        experiment.run_adaptive_batch(inquiry_id)

    import json

    raw = json.loads(experiment.path.read_text(encoding="utf-8"))
    rare_attempts = raw["branches"]["rare"]["completed_shots"]
    certain_attempts = raw["branches"]["certain"]["completed_shots"]
    assert rare_attempts > certain_attempts
    assert rare_attempts + certain_attempts == 20
    final_inquiry = experiment.inquiry_snapshot(inquiry_id)
    rare_stats = final_inquiry["current_evaluation"]["branch_statistics"][0]
    assert rare_stats["pass_interval"][1] > 0
    assert rare_stats["event_trials"] == 0
    assert rare_stats["event_interval"] == [0.0, 1.0]
    assert "接受率本身有" in final_inquiry["decisions"][-2]["reason"]
    assert "条件分布没有证据" in final_inquiry["decisions"][-2]["reason"]


def test_changing_target_creates_new_immutable_inquiry_and_keeps_history(tmp_path):
    checkpoint = _checkpoint()
    specs = [{"branch_id": "only", "label": "only", "steps": [], "budget": 0}]
    experiment = branches.BranchExperiment.create(
        checkpoint, specs, storage_dir=tmp_path, base_seed=4, experiment_id="history"
    )
    first = experiment.add_inquiry(
        {"mode": "single", "branch_ids": ["only"], "event": "0",
         "margin": 0.01, "total_budget": 100, "batch_size": 100}
    )
    experiment.run_adaptive_batch(first["inquiry_id"])
    second = experiment.add_inquiry(
        {"mode": "single", "branch_ids": ["only"], "event": "__pass__",
         "margin": 0.1, "total_budget": 10, "batch_size": 10}
    )

    import json

    raw = json.loads(experiment.path.read_text(encoding="utf-8"))
    assert list(raw["inquiries"].keys()) == [first["inquiry_id"], second["inquiry_id"]]
    assert raw["inquiries"][first["inquiry_id"]]["immutable"] is True
    assert raw["inquiries"][first["inquiry_id"]]["event"] == "0"
    assert raw["inquiries"][second["inquiry_id"]]["event"] == "__pass__"


def test_different_batch_sizes_reach_same_certain_conclusion(tmp_path):
    checkpoint = _checkpoint()
    specs = [{"branch_id": "only", "label": "only", "steps": [], "budget": 0}]
    outcomes = []
    evidence_paths = {}
    for batch_size in (1, 10):
        experiment = branches.BranchExperiment.create(
            checkpoint,
            specs,
            storage_dir=tmp_path / str(batch_size),
            base_seed=9,
            experiment_id=f"batch-{batch_size}",
        )
        inquiry = experiment.add_inquiry(
            {"mode": "single", "branch_ids": ["only"], "event": "0",
             "margin": 0.05, "total_budget": 1000, "batch_size": batch_size}
        )
        inquiry_id = inquiry["inquiry_id"]
        path = []
        while experiment.inquiry_snapshot(inquiry_id)["status"] == "active":
            experiment.run_adaptive_batch(inquiry_id)
            evaluation = experiment.inquiry_snapshot(inquiry_id)["current_evaluation"]
            lower, upper = evaluation["branch_statistics"][0]["event_interval"]
            path.append((evaluation["used_budget"], lower, upper, upper - lower))
        final = experiment.inquiry_snapshot(inquiry_id)
        evidence_paths[batch_size] = path
        outcomes.append((
            final["status"],
            final["current_evaluation"]["conclusion"]["type"],
            final["decisions"][-1]["reason"],
        ))

    assert outcomes[0][0] == outcomes[1][0] == "decided"
    assert outcomes[0][1] == outcomes[1][1] == "probability_estimated"
    assert outcomes[0][2] == outcomes[1][2]
    # Compare evidence/interval trajectories, not only a point estimate:
    # both schedules see the same RNG stream, so at common cumulative counts
    # their intervals must match exactly; each stops the first time the
    # width rule is satisfied (batch granularity changes the check point).
    fine = evidence_paths[1]
    coarse = evidence_paths[10]
    fine_at = {used: (lower, upper, width) for used, lower, upper, width in fine}
    coarse_at = {used: (lower, upper, width) for used, lower, upper, width in coarse}
    common = sorted(set(fine_at) & set(coarse_at))
    assert common
    for used in common:
        assert fine_at[used] == coarse_at[used]
    assert fine[-1][0] <= coarse[-1][0]
    assert all(lower <= 1.0 <= upper for _, lower, upper, _ in fine + coarse)
    settled = next(index for index, (_, _, _, width) in enumerate(fine) if width < 1.0)
    settled_widths = [width for *_, width in fine[settled:]]
    assert settled_widths == sorted(settled_widths, reverse=True)
    assert fine[-1][3] <= 0.1 and coarse[-1][3] <= 0.1
    stop_entries = [
        entry for entry in experiment.inquiry_snapshot(inquiry_id)["decisions"]
        if entry["action"] == "stop"
    ]
    assert len(stop_entries) == 1
