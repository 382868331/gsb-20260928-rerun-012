import numpy as np

from quantum_lab.models import CircuitDraft, CircuitStep, InitialState
from quantum_lab.simulator import apply_step, execute_next, initial_vector, make_run, probability_map, run_shots, validate_circuit


def step(step_id: str, gate: str, target: int, control: int | None = None) -> CircuitStep:
    return CircuitStep(id=step_id, type=gate, target=target, control=control, angle=None)


def test_single_qubit_hadamard_probabilities_and_repeated_statistics():
    circuit = CircuitDraft(
        qubits=1,
        initial_state=InitialState(kind="basis", basis="0"),
        seed=7,
        steps=[step("h", "H", 0), step("m", "MEASURE", 0)],
    )

    run = make_run(circuit)
    assert run.probabilities == {"0": 1.0}

    evolved = apply_step(initial_vector(circuit), 1, circuit.steps[0])
    probabilities = probability_map(evolved, 1)
    assert probabilities.keys() == {"0", "1"}
    assert np.isclose(probabilities["0"], 0.5)
    assert np.isclose(probabilities["1"], 0.5)

    summary = run_shots(circuit, 4000)
    assert set(summary.counts) <= {"0", "1"}
    assert abs(summary.probabilities.get("0", 0) - 0.5) < 0.04
    assert abs(summary.probabilities.get("1", 0) - 0.5) < 0.04


def test_bell_pair_only_produces_correlated_results():
    circuit = CircuitDraft(
        qubits=2,
        initial_state=InitialState(kind="basis", basis="00"),
        seed=17,
        steps=[
            step("h", "H", 0),
            step("cnot", "CNOT", 1, 0),
            step("m0", "MEASURE", 0),
            step("m1", "MEASURE", 1),
        ],
    )

    evolved = initial_vector(circuit)
    for gate_step in circuit.steps[:2]:
        evolved = apply_step(evolved, 2, gate_step)

    probabilities = probability_map(evolved, 2)
    assert set(probabilities) == {"00", "11"}
    assert np.isclose(probabilities["00"], 0.5)
    assert np.isclose(probabilities["11"], 0.5)

    summary = run_shots(circuit, 4000)
    assert set(summary.counts) == {"00", "11"}
    assert abs(summary.probabilities["00"] - 0.5) < 0.04
    assert abs(summary.probabilities["11"] - 0.5) < 0.04


def test_collapsed_measurement_state_affects_later_gate():
    circuit = CircuitDraft(
        qubits=1,
        initial_state=InitialState(kind="basis", basis="0"),
        seed=1,
        steps=[step("h", "H", 0), step("m", "MEASURE", 0), step("x", "X", 0)],
    )
    run = make_run(circuit)

    execute_next(run, circuit.steps, np.random.default_rng(0))
    measurement = execute_next(run, circuit.steps, np.random.default_rng(2001))
    assert measurement.measurement_result in {"0", "1"}
    final = execute_next(run, circuit.steps, np.random.default_rng(1))

    expected = "0" if measurement.measurement_result == "1" else "1"
    assert final.probabilities == {expected: 1.0}


def test_illegal_connections_are_rejected():
    circuit = CircuitDraft(
        qubits=1,
        initial_state=InitialState(kind="basis", basis="0"),
        seed=1,
        steps=[step("bad", "CNOT", 0, 0)],
    )

    errors = validate_circuit(circuit)
    assert any("控制位和目标位不能相同" in error for error in errors)


def test_repeated_measurement_connection_is_rejected():
    circuit = CircuitDraft(
        qubits=1,
        initial_state=InitialState(kind="basis", basis="0"),
        seed=1,
        steps=[step("m1", "MEASURE", 0), step("m2", "MEASURE", 0)],
    )

    errors = validate_circuit(circuit)
    assert any("不能重复连接测量节点" in error for error in errors)
