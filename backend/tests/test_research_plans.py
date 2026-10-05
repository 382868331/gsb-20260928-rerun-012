from backend import branches, server


def _checkpoint(steps=None, qubits=1, basis="0", seed=17):
    session = server.Session(
        {"qubits": qubits, "initial_basis": basis, "seed": seed, "steps": steps or []}
    )
    for _ in steps or []:
        session.step()
    return session.save_checkpoint("prefix", "cp-research")


def _experiment(tmp_path, specs, *, seed=11, experiment_id="research"):
    checkpoint = _checkpoint()
    return branches.BranchExperiment.create(
        checkpoint,
        specs,
        storage_dir=tmp_path,
        base_seed=seed,
        experiment_id=experiment_id,
    )


def test_mutually_exclusive_goals_share_same_physical_samples_and_budget(tmp_path):
    experiment = _experiment(
        tmp_path,
        [{"branch_id": "only", "label": "only", "steps": [], "budget": 0}],
    )
    experiment.create_research_plan(
        {
            "plan_id": "shared",
            "policy": "balanced",
            "total_budget": 10,
            "batch_size": 10,
            "goals": [
                {"goal_id": "zero", "mode": "single", "branch_ids": ["only"], "event": "0", "margin": 0.4},
                {"goal_id": "one", "mode": "single", "branch_ids": ["only"], "event": "1", "margin": 0.4},
            ],
        }
    )
    plan = experiment.run_research_plan_batch("shared")

    assert plan["used_budget"] == 10
    assert plan["joint_batches"][0]["served_goal_ids"] == ["zero", "one"]
    assert plan["joint_batches"][0]["charged_shots"] == 10
    assert experiment.branches["only"]["completed_shots"] == 10
    for goal in plan["goals"]:
        sample_decision = next(item for item in goal["decisions"] if item["action"] == "sample")
        assert sample_decision["shots"] == 10
        assert sample_decision["charged_shots"] == 0


def test_branches_with_same_prefix_but_different_conditions_are_separate_distributions(tmp_path):
    experiment = _experiment(
        tmp_path,
        [
            {"branch_id": "plain", "label": "plain", "steps": [], "budget": 0},
            {
                "branch_id": "rare",
                "label": "rare",
                "steps": [{"type": "MEASURE", "target": 0}],
                "filters": [{"step": 0, "outcome": 1}],
                "budget": 0,
            },
        ],
        seed=12,
    )
    experiment.create_research_plan(
        {
            "plan_id": "conditions",
            "policy": "balanced",
            "total_budget": 2,
            "batch_size": 1,
            "goals": [
                {"goal_id": "plain-goal", "mode": "single", "branch_ids": ["plain"], "event": "0", "margin": 0.1},
                {"goal_id": "rare-goal", "mode": "single", "branch_ids": ["rare"], "event": "0", "margin": 0.1},
            ],
        }
    )
    plan = experiment.run_research_plan_batch("conditions")
    sampled = {group["branch_id"] for group in plan["joint_batches"]}
    assert sampled == {"plain", "rare"}
    assert plan["used_budget"] == 2
    fingerprints = {group["branch_id"]: group["distribution_fingerprint"] for group in plan["joint_batches"]}
    assert fingerprints["plain"] != fingerprints["rare"]
    rare_goal = next(goal for goal in plan["goals"] if goal["goal_id"] == "rare-goal")
    rare_stats = rare_goal["current_evaluation"]["branch_statistics"][0]
    assert rare_stats["passed_shots"] == 0
    assert rare_stats["event_trials"] == 0
    assert rare_stats["event_interval"] == [0.0, 1.0]
    assert rare_goal["status"] == "inconclusive_budget_exhausted"


def test_priority_policy_finishes_named_goal_before_advancing_others(tmp_path):
    experiment = _experiment(
        tmp_path,
        [
            {"branch_id": "certain", "label": "certain", "steps": [], "budget": 0},
            {"branch_id": "random", "label": "random", "steps": [{"type": "H", "target": 0}], "budget": 0},
        ],
        seed=13,
    )
    experiment.create_research_plan(
        {
            "plan_id": "priority",
            "policy": "priority",
            "priority_goal_ids": ["certain-goal"],
            "total_budget": 1000,
            "batch_size": 50,
            "goals": [
                {"goal_id": "certain-goal", "mode": "single", "branch_ids": ["certain"], "event": "0", "margin": 0.05},
                {"goal_id": "random-goal", "mode": "single", "branch_ids": ["random"], "event": "1", "margin": 0.05},
            ],
        }
    )
    plan = experiment.run_research_plan_batch("priority")
    while plan["goals"][0]["status"] == "active":
        plan = experiment.run_research_plan_batch("priority")

    assert plan["goals"][0]["status"] == "decided"
    assert plan["goals"][1]["status"] == "active"
    assert experiment.branches["random"]["completed_shots"] == 0
    assert {group["branch_id"] for group in plan["joint_batches"]} == {"certain"}


def test_cancelling_one_goal_keeps_adopted_samples_and_other_goal_history(tmp_path):
    experiment = _experiment(
        tmp_path,
        [{"branch_id": "only", "label": "only", "steps": [], "budget": 0}],
    )
    plan = experiment.create_research_plan(
        {
            "plan_id": "cancel",
            "policy": "balanced",
            "total_budget": 100,
            "batch_size": 1,
            "goals": [
                {"goal_id": "keep", "mode": "single", "branch_ids": ["only"], "event": "0", "margin": 0.05},
                {"goal_id": "drop", "mode": "single", "branch_ids": ["only"], "event": "1", "margin": 0.05},
            ],
        }
    )
    experiment.run_research_plan_batch("cancel")
    experiment.cancel_research_goal("cancel", "drop", reason="测试取消")
    while experiment.research_plans["cancel"]["goals"]["keep"]["status"] == "active":
        plan = experiment.run_research_plan_batch("cancel")

    cancelled = next(goal for goal in plan["goals"] if goal["goal_id"] == "drop")
    kept = next(goal for goal in plan["goals"] if goal["goal_id"] == "keep")
    assert cancelled["status"] == "cancelled"
    assert cancelled["decisions"][-1]["reason"] == "测试取消"
    assert any(item["action"] == "sample" for item in cancelled["decisions"])
    assert kept["status"] == "decided"
    assert experiment.branches["only"]["completed_shots"] == plan["used_budget"]
    assert all(group["served_goal_ids"] == ["keep"] for group in plan["joint_batches"][1:])


def test_research_plan_survives_reload(tmp_path):
    experiment = _experiment(
        tmp_path,
        [{"branch_id": "only", "label": "only", "steps": [], "budget": 0}],
        experiment_id="persist",
    )
    experiment.create_research_plan(
        {
            "plan_id": "persisted",
            "total_budget": 2,
            "batch_size": 1,
            "goals": [
                {"goal_id": "zero", "mode": "single", "branch_ids": ["only"], "event": "0", "margin": 0.1},
            ],
        }
    )

    loaded = branches.BranchExperiment.load(experiment.path)
    while loaded.research_plans["persisted"]["status"] == "active":
        plan = loaded.run_research_plan_batch("persisted")
    assert plan["used_budget"] == 2
    assert plan["goals"][0]["status"] == "inconclusive_budget_exhausted"
