from backend import branches, server


def _checkpoint():
    session = server.Session(
        {"qubits": 1, "initial_basis": "0", "seed": 31, "steps": []}
    )
    return session.save_checkpoint("prefix", "cp-research")


def _experiment(tmp_path, seed=7, budget=0):
    specs = [
        {"branch_id": "zero", "label": "zero", "steps": [], "budget": budget},
        {
            "branch_id": "rare",
            "label": "rare condition",
            "steps": [{"type": "MEASURE", "target": 0}],
            "filters": [{"step": 0, "outcome": 1}],
            "budget": 0,
        },
    ]
    return branches.BranchExperiment.create(
        _checkpoint(), specs, storage_dir=tmp_path, base_seed=seed, experiment_id="research"
    )


def _goal(goal_id, branch_id="zero", event="0", *, margin=0.2, batch_size=5, priority=0):
    return {
        "goal_id": goal_id,
        "label": goal_id,
        "mode": "single",
        "branch_ids": [branch_id],
        "event": event,
        "margin": margin,
        "batch_size": batch_size,
        "priority": priority,
        "alpha": 0.05,
    }


def test_mutually_exclusive_event_goals_share_actual_samples_once(tmp_path):
    experiment = _experiment(tmp_path)
    plan = experiment.add_research_plan(
        {
            "strategy": "balanced",
            "total_budget": 10,
            "goals": [_goal("see-zero", event="0"), _goal("see-one", event="1")],
        }
    )

    plan = experiment.run_research_batch(plan["plan_id"], shots_override=4)

    assert plan["executed_shots"] == 4
    assert plan["batches"][0]["goal_ids"] == ["see-zero", "see-one"]
    assert plan["goals"][0]["adopted_batch_ids"] == plan["goals"][1]["adopted_batch_ids"]
    zero_stats = plan["goals"][0]["current_evaluation"]["branch_statistics"][0]
    one_stats = plan["goals"][1]["current_evaluation"]["branch_statistics"][0]
    assert (zero_stats["event_successes"], one_stats["event_successes"]) == (4, 0)
    assert experiment.branches["zero"]["completed_shots"] == 4

    experiment.add_shots("zero", 2, source="unrelated-external-batch")
    plan = experiment.run_research_batch(plan["plan_id"], shots_override=4)
    goal_stats = plan["goals"][0]["current_evaluation"]["branch_statistics"][0]
    assert experiment.branches["zero"]["completed_shots"] == 10
    assert goal_stats["attempts"] == 8
    assert plan["executed_shots"] == 8


def test_shared_prefix_with_different_conditions_is_not_pooled(tmp_path):
    experiment = _experiment(tmp_path)
    plan = experiment.add_research_plan(
        {
            "strategy": "balanced",
            "total_budget": 10,
            "goals": [
                _goal("unconditional", branch_id="zero", event="0"),
                _goal("conditional-rare", branch_id="rare", event="0"),
            ],
        }
    )

    plan = experiment.run_research_batch(plan["plan_id"], shots_override=3)
    sampled_branch = plan["batches"][0]["branch_id"]
    assert plan["batches"][0]["goal_ids"] == [f"conditional-{sampled_branch}" if sampled_branch == "rare" else sampled_branch]
    assert plan["goals"][1 if sampled_branch == "zero" else 0]["adopted_batch_ids"] == []
    assert experiment.snapshot()["pooled_across_conditions"] is False


def test_balanced_strategy_alternates_between_distribution_goals(tmp_path):
    experiment = _experiment(tmp_path)
    plan = experiment.add_research_plan(
        {
            "strategy": "balanced",
            "total_budget": 20,
            "goals": [
                _goal("goal-a", branch_id="zero", batch_size=5),
                _goal("goal-b", branch_id="rare", batch_size=5),
            ],
        }
    )

    for _ in range(2):
        plan = experiment.run_research_batch(plan["plan_id"])

    sampled_branches = [batch["branch_id"] for batch in plan["batches"]]
    assert set(sampled_branches) == {"zero", "rare"}
    assert plan["executed_shots"] == 10


def test_priority_strategy_finishes_named_goal_before_other_goal(tmp_path):
    experiment = _experiment(tmp_path)
    plan = experiment.add_research_plan(
        {
            "strategy": "priority",
            "priority_goal_ids": ["priority-zero"],
            "total_budget": 20,
            "goals": [
                _goal("priority-zero", branch_id="zero", margin=0.49, batch_size=5),
                _goal("later-rare", branch_id="rare", margin=1e-9, batch_size=5),
            ],
        }
    )

    while any(goal["status"] == "active" for goal in plan["goals"]):
        plan = experiment.run_research_batch(plan["plan_id"])

    first_other_branch_batch = next(index for index, batch in enumerate(plan["batches"]) if batch["branch_id"] == "rare")
    assert first_other_branch_batch > 0
    assert plan["goals"][0]["status"] == "decided"


def test_cancel_keeps_shared_samples_for_other_goal_and_independent_audit(tmp_path):
    experiment = _experiment(tmp_path)
    plan = experiment.add_research_plan(
        {
            "strategy": "balanced",
            "total_budget": 20,
            "goals": [_goal("keep"), _goal("cancel")],
        }
    )
    plan = experiment.run_research_batch(plan["plan_id"], shots_override=2)
    shared_batch_id = plan["batches"][0]["batch_id"]

    plan = experiment.cancel_research_goal(plan["plan_id"], "cancel")
    plan = experiment.run_research_batch(plan["plan_id"], shots_override=2)

    cancelled = next(goal for goal in plan["goals"] if goal["goal_id"] == "cancel")
    kept = next(goal for goal in plan["goals"] if goal["goal_id"] == "keep")
    assert cancelled["status"] == "cancelled"
    assert cancelled["adopted_batch_ids"] == [shared_batch_id]
    assert len(kept["adopted_batch_ids"]) == 2
    assert experiment.branches["zero"]["completed_shots"] == 4
    assert any(entry["action"] == "cancel" for entry in cancelled["decisions"])


def test_exhausted_shared_budget_leaves_rare_goal_inconclusive_without_double_charge(tmp_path):
    experiment = _experiment(tmp_path)
    plan = experiment.add_research_plan(
        {
            "strategy": "balanced",
            "total_budget": 3,
            "goals": [
                _goal("rare-1", branch_id="rare", margin=1e-9, batch_size=1),
                _goal("rare-2", branch_id="rare", margin=1e-9, batch_size=1),
            ],
        }
    )

    while plan["remaining_budget"]:
        plan = experiment.run_research_batch(plan["plan_id"])

    assert plan["executed_shots"] == 3
    assert experiment.branches["rare"]["completed_shots"] == 3
    assert all(goal["status"] == "inconclusive_budget_exhausted" for goal in plan["goals"])
    assert all(goal["current_evaluation"]["conclusion"]["type"] == "inconclusive" for goal in plan["goals"])


def test_existing_samples_are_adopted_without_spending_new_plan_budget(tmp_path):
    experiment = _experiment(tmp_path, budget=6)
    plan = experiment.add_research_plan(
        {
            "strategy": "balanced",
            "total_budget": 4,
            "goals": [_goal("already-evidenced", branch_id="zero", margin=0.001, batch_size=2)],
        }
    )

    assert plan["executed_shots"] == 0
    assert plan["goals"][0]["current_evaluation"]["branch_statistics"][0]["attempts"] == 6
    assert any(entry["action"] == "adopt_existing_samples" for entry in plan["goals"][0]["decisions"])


def test_research_plan_persists_across_reload(tmp_path):
    experiment = _experiment(tmp_path)
    created = experiment.add_research_plan(
        {"strategy": "balanced", "total_budget": 5, "goals": [_goal("persist")]}
    )
    experiment.run_research_batch(created["plan_id"], shots_override=2)

    reloaded = branches.BranchExperiment.load(experiment.path)
    plan = reloaded.research_plan_snapshot(created["plan_id"])

    assert plan["executed_shots"] == 2
    assert plan["goals"][0]["adopted_batch_ids"]
    assert plan["goals"][0]["decisions"][0]["action"] == "created"
