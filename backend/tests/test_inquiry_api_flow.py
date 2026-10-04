"""End-to-end HTTP flow for sequential inquiry records on a real experiment."""

import json
from http import HTTPStatus

from backend import branches, server
from backend.tests.test_branch_api import DummyHandler


def _make_checkpoint():
    session = server.Session(
        {"qubits": 1, "initial_basis": "0", "seed": 55, "steps": []}
    )
    checkpoint = session.save_checkpoint("prefix", "cp-inquiry-flow")
    server.CHECKPOINTS["cp-inquiry-flow"] = checkpoint
    return checkpoint


def _dispatch(payload, path, method):
    handler = DummyHandler(payload)
    handler.path = path
    getattr(server.Handler, f"do_{method}")(handler)
    assert handler.response.status in (HTTPStatus.OK, HTTPStatus.CREATED), handler.response.body
    return handler.response.body


def test_inquiry_lifecycle_pause_resume_revise_via_http(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "BRANCH_DIR", tmp_path)
    _make_checkpoint()

    created = _dispatch(
        {
            "name": "inquiry-flow",
            "seed": 707,
            "branches": [
                {"branch_id": "a", "label": "A", "steps": [{"type": "H", "target": 0}], "budget": 0},
                {"branch_id": "b", "label": "B", "steps": [{"type": "H", "target": 0}], "budget": 0},
            ],
        },
        "/api/checkpoints/cp-inquiry-flow/branch-experiments",
        "POST",
    )
    experiment_id = created["experiment_id"]
    assert created["inquiries"] == []

    # Decide on a real, persisted experiment: a resumable record appears.
    first = _dispatch(
        {"mode": "compare", "branch_ids": ["a", "b"], "event": "0",
         "margin": 0.001, "total_budget": 40, "batch_size": 10},
        f"/api/branch-experiments/{experiment_id}/inquiries",
        "PATCH",
    )
    first_id = first["inquiry_id"]
    assert first["rule_version"] == branches.RULE_VERSION
    assert first["decisions"][0]["action"] == "created"

    evaluation = _dispatch(
        {"batch_size": 10},
        f"/api/branch-experiments/{experiment_id}/inquiries/{first_id}/adaptive-batch",
        "PATCH",
    )["evaluation"]
    assert evaluation["used_budget"] == 10

    # Simulate pause/reopen by dropping the in-memory registry and reloading.
    server.BRANCH_EXPERIMENTS.clear()
    resumed_body = _dispatch(
        {},
        f"/api/branch-experiments/{experiment_id}",
        "GET",
    )
    resumed = resumed_body["inquiries"][0]
    assert resumed["inquiry_id"] == first_id
    assert resumed["current_evaluation"]["used_budget"] == 10

    # Append another batch after reopening; original counts are preserved.
    _dispatch(
        {"batch_size": 10},
        f"/api/branch-experiments/{experiment_id}/inquiries/{first_id}/adaptive-batch",
        "PATCH",
    )
    raw = json.loads((tmp_path / f"{experiment_id}.json").read_text(encoding="utf-8"))
    first_batches = sum(
        batch["shots"]
        for branch in ("a", "b")
        for batch in raw["branches"][branch]["batches"]
        if batch["source"] == f"adaptive:{first_id}"
    )
    assert first_batches == 20

    # Changing the target creates a new revision 1 inquiry; the old one stays.
    second_created = _dispatch(
        {"mode": "single", "branch_ids": ["a"], "event": "__pass__",
         "margin": 0.1, "total_budget": 10, "batch_size": 10},
        f"/api/branch-experiments/{experiment_id}/inquiries",
        "PATCH",
    )
    assert second_created["revision"] == 1
    second_id = second_created["inquiry_id"]
    listing = _dispatch({}, f"/api/branch-experiments/{experiment_id}", "GET")["inquiries"]
    assert [inquiry["inquiry_id"] for inquiry in listing] == [first_id, second_id]
    raw = json.loads((tmp_path / f"{experiment_id}.json").read_text(encoding="utf-8"))
    assert list(raw["inquiry_order"]) == [first_id, second_id]
    assert raw["inquiries"][first_id]["event"] == "0"
    assert raw["inquiries"][first_id]["immutable"] is True
    # Original counts are shared branch facts and must not be reset.
    total_shots = sum(raw["branches"][name]["completed_shots"] for name in ("a", "b"))
    assert total_shots == 20
