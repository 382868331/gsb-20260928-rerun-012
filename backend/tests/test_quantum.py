import math

import numpy as np

from backend import quantum


def test_hadamard_probabilities():
    state = quantum.initial_state(1, "0")
    state = quantum.apply_gate(state, 1, {"type": "H", "target": 0})
    probs = quantum.probabilities(state)
    assert np.allclose(probs, [0.5, 0.5])


def test_rx_quarter_rotation():
    state = quantum.initial_state(1, "0")
    state = quantum.apply_gate(state, 1, {"type": "RX", "target": 0, "angle": math.pi / 2})
    probs = quantum.probabilities(state)
    assert np.allclose(probs, [0.5, 0.5], atol=1e-12)


def test_bell_pair_probabilities_and_correlation():
    rng = np.random.default_rng(7)
    result = quantum.run_experiment(
        2,
        "00",
        [
            {"type": "H", "target": 0},
            {"type": "CNOT", "control": 0, "target": 1},
            {"type": "MEASURE", "target": 0},
            {"type": "MEASURE", "target": 1},
        ],
        4000,
        7,
    )
    counts = result["counts"]
    assert counts["01"] == 0
    assert counts["10"] == 0
    assert counts["00"] + counts["11"] == 4000
    assert abs(counts["00"] - 2000) < 150
    assert abs(counts["11"] - 2000) < 150
    expected = result["expected_probabilities"]
    assert abs(expected["00"] - 0.5) < 1e-12
    assert abs(expected["11"] - 0.5) < 1e-12
    assert expected["01"] == 0.0
    assert expected["10"] == 0.0


def test_measurement_collapse_drives_later_steps():
    collapsed_outcomes = []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        state = quantum.initial_state(2, "00")
        state = quantum.apply_gate(state, 2, {"type": "H", "target": 0})
        state, outcome = quantum.measure(state, 2, 0, rng)
        state = quantum.apply_gate(
            state, 2, {"type": "CNOT", "control": 0, "target": 1}
        )
        probs = quantum.probabilities(state)
        if outcome == 0:
            assert np.allclose(probs, [1.0, 0.0, 0.0, 0.0], atol=1e-12)
        else:
            assert np.allclose(probs, [0.0, 0.0, 0.0, 1.0], atol=1e-12)
        collapsed_outcomes.append(outcome)
    assert set(collapsed_outcomes) == {0, 1}


def test_repeated_shots_with_same_seed_match():
    steps = [{"type": "H", "target": 0}]
    first = quantum.run_experiment(1, "0", steps, 200, 123)
    second = quantum.run_experiment(1, "0", steps, 200, 123)
    assert first["counts"] == second["counts"]


def test_validation_rejects_invalid_wiring():
    assert quantum.validate_circuit(
        {"qubits": 2, "initial_basis": "00", "steps": [{"type": "CNOT", "control": 1, "target": 1}]}
    ) == [
        {"step": 0, "message": "CNOT must connect two distinct qubits"}
    ]
    errors = quantum.validate_circuit(
        {"qubits": 1, "initial_basis": "0", "steps": [{"type": "CNOT", "control": 0, "target": 1}]}
    )
    assert any("target" in error["message"] for error in errors)
    errors = quantum.validate_circuit(
        {"qubits": 1, "initial_basis": "0", "steps": [{"type": "RX", "target": 0}]}
    )
    assert errors
    assert quantum.validate_circuit(
        {"qubits": 5, "initial_basis": "00000", "steps": []}
    )


def test_experiment_statistics_near_expected():
    result = quantum.run_experiment(
        1,
        "0",
        [{"type": "H", "target": 0}],
        10_000,
        42,
    )
    assert abs(result["frequencies"]["0"] - 0.5) < 0.03
