import numpy as np

from backend import quantum, server


BELL_STEPS = [
    {"type": "H", "target": 0},
    {"type": "CNOT", "control": 0, "target": 1},
    {"type": "MEASURE", "target": 0},
    {"type": "MEASURE", "target": 1},
]
PAYLOAD = {
    "qubits": 2,
    "initial_basis": "00",
    "seed": 11,
    "steps": BELL_STEPS,
}


def _reference_outcomes():
    rng = np.random.default_rng(11)
    result = quantum.run_trace(2, "00", BELL_STEPS, rng)
    return [item["outcome"] for item in result["measurements"]]


def test_stepwise_matches_one_shot_with_fixed_seed():
    session = server.Session(PAYLOAD)
    for _ in BELL_STEPS:
        session.step()
    assert session.finished
    stepwise = [item["outcome"] for item in session.measurements]
    assert stepwise == _reference_outcomes()


def test_stepwise_matches_one_shot_with_conditional_gate_after_measure():
    steps = [
        {"type": "H", "target": 0},
        {"type": "MEASURE", "target": 0},
        {"type": "CNOT", "control": 0, "target": 1},
        {"type": "MEASURE", "target": 1},
    ]
    payload = {"qubits": 2, "initial_basis": "00", "seed": 23, "steps": steps}

    session = server.Session(payload)
    step_probs = []
    for _ in steps:
        step_probs.append(session.step()["probabilities"])

    reference = quantum.run_trace(2, "00", steps, np.random.default_rng(23))
    reference_probs = [item["probabilities"] for item in reference["trace"]]
    assert step_probs == reference_probs
    assert [m["outcome"] for m in session.measurements] == [
        item["outcome"] for item in reference["measurements"]
    ]


def test_stepping_after_finish_is_idempotent():
    session = server.Session(PAYLOAD)
    for _ in BELL_STEPS:
        session.step()
    snapshot = session.snapshot()
    again = session.step()
    assert again["finished"]
    assert again["next_index"] == len(BELL_STEPS)
    assert session.snapshot() == snapshot


def test_resume_does_not_redraw_past_measurements():
    session = server.Session(PAYLOAD)
    session.step()
    session.step()
    checkpoint = session.save_checkpoint("after-entangle", "cp1")
    session.step()
    session.step()
    reference = _reference_outcomes()

    resumed = server.Session.from_checkpoint(checkpoint)
    assert resumed.next_index == 2
    resumed.step()
    resumed.step()
    assert [item["outcome"] for item in resumed.measurements] == reference


def test_resume_branch_compare_shares_rng_state():
    session = server.Session(PAYLOAD)
    session.step()
    session.step()
    session.step()  # first measurement already drawn
    checkpoint = session.save_checkpoint("after-entangle", "cp2")

    # Conditional follow-up: flip q1 vs do nothing; both start from the same
    # collapsed state and RNG position, so neither redraws prior measurements.
    flip = server.run_continuation(checkpoint, [{"type": "X", "target": 1}])
    idle = server.run_continuation(checkpoint, [])

    first_outcome = checkpoint.measurements[0]["outcome"]
    surviving = f"{first_outcome}{first_outcome}"
    assert idle["final_probabilities"][surviving] == 1.0
    flipped = f"{first_outcome}{1 - first_outcome}"
    assert flip["final_probabilities"][flipped] == 1.0
    assert idle["measurements"] == []
    assert flip["measurements"] == []


def test_experiments_use_independent_rng_from_sessions():
    session = server.Session(PAYLOAD)
    session.step()
    before = session.snapshot()["probabilities"]

    # A batch experiment draws many numbers; the interactive session's RNG
    # and collapsed state must be untouched.
    quantum.run_experiment(2, "00", BELL_STEPS, 50, seed=99)
    assert session.snapshot()["probabilities"] == before
    session.step()  # CNOT
    session.step()  # first MEASURE
    assert [item["outcome"] for item in session.measurements] == [_reference_outcomes()[0]]
