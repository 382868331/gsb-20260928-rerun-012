"""Small local state-vector quantum simulator (1-4 qubits).

All simulation runs locally with numpy; no cloud service is involved.
Supported operations (the supported range for the first round):

Single-qubit gates: H, X, Y, Z, S, T and parameterised RX(theta),
RY(theta), RZ(theta) (theta in radians).
Two-qubit gates:   CNOT(control, target), CZ(control, target),
SWAP(a, b) -- the two qubits must be distinct and in range.
Measurement:       MEASURE(q) performs a projective, wave-function-
collapsing measurement. The collapsed state drives every later step.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MAX_QUBITS = 4

SINGLE_GATES = {"H", "X", "Y", "Z", "S", "T"}
PARAM_GATES = {"RX", "RY", "RZ"}
TWO_QUBIT_GATES = {"CNOT", "CZ", "SWAP"}
MEASURE_GATE = "MEASURE"
CONDITIONAL_GATE = "IF"
ALL_GATES = SINGLE_GATES | PARAM_GATES | TWO_QUBIT_GATES | {MEASURE_GATE, CONDITIONAL_GATE}


def _matrices() -> dict[str, np.ndarray]:
    i = 1j
    return {
        "H": np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -i], [i, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        "S": np.array([[1, 0], [0, i]], dtype=complex),
        "T": np.array([[1, 0], [0, np.exp(i * math.pi / 4)]], dtype=complex),
    }


_GATE_MATRICES = _matrices()

# Basis ordering for the two involved qubits is (control/first, second),
# i.e. |00>, |01>, |10>, |11>.
CNOT_MATRIX = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=complex,
)
CZ_MATRIX = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)
SWAP_MATRIX = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=complex,
)


def initial_state(qubits: int, basis: str) -> np.ndarray:
    state = np.zeros(2**qubits, dtype=complex)
    state[int(basis, 2)] = 1.0
    return state


def _rotation_matrix(gate: str, theta: float) -> np.ndarray:
    half = theta / 2.0
    c, s = math.cos(half), math.sin(half)
    i = 1j
    if gate == "RX":
        return np.array([[c, -i * s], [-i * s, c]], dtype=complex)
    if gate == "RY":
        return np.array([[c, -s], [s, c]], dtype=complex)
    # RZ
    return np.array([[np.exp(-i * half), 0], [0, np.exp(i * half)]], dtype=complex)


def _apply_single(state: np.ndarray, qubits: int, target: int, matrix: np.ndarray) -> np.ndarray:
    tensor = state.reshape((2,) * qubits)
    moved = np.moveaxis(tensor, target, -1)
    moved = moved @ matrix.T
    return np.moveaxis(moved, -1, target).reshape(-1)


def _apply_two(
    state: np.ndarray, qubits: int, first: int, second: int, matrix: np.ndarray
) -> np.ndarray:
    tensor = state.reshape((2,) * qubits)
    moved = np.moveaxis(tensor, [first, second], [-2, -1])
    rest = moved.size // 4
    moved = (moved.reshape(rest, 4) @ matrix.T).reshape(moved.shape)
    return np.moveaxis(moved, [-2, -1], [first, second]).reshape(-1)


def apply_gate(
    state: np.ndarray, qubits: int, step: dict[str, Any]
) -> np.ndarray:
    gate = step["type"]
    if gate in SINGLE_GATES:
        return _apply_single(state, qubits, int(step["target"]), _GATE_MATRICES[gate])
    if gate in PARAM_GATES:
        matrix = _rotation_matrix(gate, float(step["angle"]))
        return _apply_single(state, qubits, int(step["target"]), matrix)
    if gate == "CNOT":
        return _apply_two(
            state, qubits, int(step["control"]), int(step["target"]), CNOT_MATRIX
        )
    if gate == "CZ":
        return _apply_two(
            state, qubits, int(step["control"]), int(step["target"]), CZ_MATRIX
        )
    if gate == "SWAP":
        return _apply_two(
            state, qubits, int(step["a"]), int(step["b"]), SWAP_MATRIX
        )
    raise ValueError(f"unsupported gate: {gate}")


def probabilities(state: np.ndarray) -> list[float]:
    probs = (state.real**2 + state.imag**2).tolist()
    return [min(1.0, max(0.0, p)) for p in probs]


def measure(state: np.ndarray, qubits: int, target: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Projective measurement in the computational basis of one qubit."""
    tensor = state.reshape((2,) * qubits)
    moved = np.moveaxis(tensor, target, 0)
    p0 = float(np.sum(np.abs(moved[0]) ** 2))
    outcome = 0 if rng.random() < p0 else 1
    mask = np.zeros_like(moved)
    mask[outcome] = moved[outcome]
    norm = math.sqrt(float(np.sum(np.abs(mask) ** 2)))
    if norm > 0:
        mask /= norm
    collapsed = np.moveaxis(mask, 0, target).reshape(-1)
    return collapsed, outcome


def condition_satisfied(step: dict[str, Any], outcomes: dict[str, int]) -> bool | None:
    """Evaluate a conditional gate against already observed measurement outcomes.

    Returns None when the control qubit was never measured before this step,
    which is an invalid circuit rather than a deterministic skip/fire.
    """
    control = int(step["control"])
    if control not in outcomes:
        return None
    return outcomes[control] == int(step.get("equals", 1))


def validate_circuit(payload: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    qubits = payload.get("qubits")
    if not isinstance(qubits, int) or not 1 <= qubits <= MAX_QUBITS:
        errors.append(
            {"step": -1, "message": f"qubits must be an integer in [1, {MAX_QUBITS}]"}
        )
        return errors

    basis = payload.get("initial_basis", "0" * qubits)
    if not isinstance(basis, str) or len(basis) != qubits or any(c not in "01" for c in basis):
        errors.append(
            {"step": -1, "message": f"initial_basis must be a {qubits}-char 0/1 string"}
        )

    steps = payload.get("steps", [])
    if not isinstance(steps, list):
        errors.append({"step": -1, "message": "steps must be a list"})
        return errors

    for index, raw in enumerate(steps):
        if not isinstance(raw, dict):
            errors.append({"step": index, "message": "step must be an object"})
            continue
        gate = raw.get("type")
        if gate not in ALL_GATES:
            errors.append({"step": index, "message": f"unknown gate {gate!r}"})
            continue

        def check_qubit(name: str) -> int | None:
            value = raw.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < qubits:
                errors.append(
                    {"step": index, "message": f"{name} must be an integer qubit index in [0, {qubits - 1}]"}
                )
                return None
            return value

        if gate == CONDITIONAL_GATE:
            inner = raw.get("then")
            control = check_qubit("control")
            equals = raw.get("equals", 1)
            if not isinstance(equals, int) or isinstance(equals, bool) or equals not in (0, 1):
                errors.append({"step": index, "message": "IF needs equals to be 0 or 1"})
            if not isinstance(inner, dict):
                errors.append({"step": index, "message": "IF needs a nested gate object in 'then'"})
                continue
            if inner.get("type") == CONDITIONAL_GATE:
                errors.append({"step": index, "message": "IF cannot nest another IF gate"})
                continue
            inner_errors = validate_circuit(
                {"qubits": qubits, "initial_basis": "0" * qubits, "steps": [inner]}
            )
            for inner_error in inner_errors:
                errors.append(
                    {
                        "step": index,
                        "message": f"IF inner gate: {inner_error['message']}",
                    }
                )
            measured_targets = {
                int(prior.get("target"))
                for prior in steps[:index]
                if isinstance(prior, dict) and prior.get("type") == MEASURE_GATE
            }
            if control is not None and control not in measured_targets:
                errors.append(
                    {
                        "step": index,
                        "message": "IF control qubit must be measured by an earlier MEASURE step",
                    }
                )
            continue

        if gate in SINGLE_GATES or gate in PARAM_GATES or gate == MEASURE_GATE:
            check_qubit("target")
        elif gate in TWO_QUBIT_GATES:
            if gate in ("CNOT", "CZ"):
                first, second = check_qubit("control"), check_qubit("target")
            else:
                first, second = check_qubit("a"), check_qubit("b")
            if first is not None and second is not None and first == second:
                errors.append(
                    {"step": index, "message": f"{gate} must connect two distinct qubits"}
                )
        if gate in PARAM_GATES:
            angle = raw.get("angle")
            if not isinstance(angle, (int, float)) or isinstance(angle, bool) or not math.isfinite(angle):
                errors.append({"step": index, "message": f"{gate} needs a finite numeric angle (radians)"})
    return errors


def resolve_step(step: dict[str, Any], outcomes: dict[str, int]) -> tuple[dict[str, Any], bool, bool]:
    """Unwrap a conditional step.

    Returns (effective_step, fired, is_conditional). An unsatisfied conditional
    yields an empty effective step that the executor skips.
    """
    if step.get("type") != CONDITIONAL_GATE:
        return step, True, False
    fired = condition_satisfied(step, outcomes) is True
    return (step["then"] if fired else {}), fired, True


def _basis_labels(qubits: int) -> list[str]:
    return [format(i, f"0{qubits}b") for i in range(2**qubits)]


def probability_map(state: np.ndarray, qubits: int) -> dict[str, float]:
    return dict(zip(_basis_labels(qubits), probabilities(state)))


def run_trace(
    qubits: int,
    basis: str,
    steps: list[dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    state = initial_state(qubits, basis)
    trace: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    outcomes: dict[str, int] = {}
    for index, step in enumerate(steps):
        measured = None
        effective, fired, is_conditional = resolve_step(step, outcomes)
        if effective:
            if effective["type"] == MEASURE_GATE:
                state, outcome = measure(state, qubits, int(effective["target"]), rng)
                measured = {"target": int(effective["target"]), "outcome": outcome}
                measurements.append({"step": index, **measured})
                outcomes[int(effective["target"])] = outcome
            else:
                state = apply_gate(state, qubits, effective)
        trace.append(
            {
                "index": index,
                "type": step["type"],
                "measured": measured,
                "condition_fired": fired if is_conditional else None,
                "probabilities": probability_map(state, qubits),
            }
        )
    return {
        "trace": trace,
        "measurements": measurements,
        "final_probabilities": probability_map(state, qubits),
    }


def ideal_probabilities(
    qubits: int, basis: str, steps: list[dict[str, Any]]
) -> dict[str, float]:
    """Probabilities of the ideal circuit with measurements omitted."""
    state = initial_state(qubits, basis)
    for step in steps:
        if step["type"] not in (MEASURE_GATE, CONDITIONAL_GATE):
            state = apply_gate(state, qubits, step)
    return probability_map(state, qubits)


def run_experiment(
    qubits: int,
    basis: str,
    steps: list[dict[str, Any]],
    shots: int,
    seed: int | None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    counts = {label: 0 for label in _basis_labels(qubits)}
    for _ in range(shots):
        state = initial_state(qubits, basis)
        outcomes: dict[str, int] = {}
        for step in steps:
            effective, _fired, _cond = resolve_step(step, outcomes)
            if not effective:
                continue
            if effective["type"] == MEASURE_GATE:
                target = int(effective["target"])
                state, outcome = measure(state, qubits, target, rng)
                outcomes[target] = outcome
            else:
                state = apply_gate(state, qubits, effective)
        probs = probabilities(state)
        index = int(rng.choice(len(probs), p=probs))
        label = format(index, f"0{qubits}b")
        counts[label] += 1
    frequencies = {k: v / shots for k, v in counts.items()}
    return {
        "shots": shots,
        "seed": seed,
        "counts": counts,
        "frequencies": frequencies,
        "expected_probabilities": ideal_probabilities(qubits, basis, steps),
    }
