from __future__ import annotations

import math
import uuid

import numpy as np

from .models import CircuitDraft, CircuitStep, RunState, ShotSummary, StepResult


class CircuitValidationError(ValueError):
    def __init__(self, errors: str | list[str]):
        self.errors = errors if isinstance(errors, list) else [errors]
        super().__init__("; ".join(self.errors))


ONE_QUBIT_GATES = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2),
    "S": np.array([[1, 0], [0, 1j]], dtype=complex),
    "T": np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex),
}


def validate_circuit(circuit: CircuitDraft) -> list[str]:
    errors: list[str] = []
    q = circuit.qubits

    if circuit.initial_state.kind == "basis":
        basis = circuit.initial_state.basis
        if len(basis) != q or any(char not in "01" for char in basis):
            errors.append(f"初始计算态必须是长度 {q} 的 0/1 字符串")
    else:
        values = circuit.initial_state.amplitudes
        expected = 2**q
        if values is None or len(values) != expected:
            errors.append(f"自定义初始状态必须包含 {expected} 个 [实部, 虚部] 振幅")
        else:
            vector = np.array(values, dtype=float)
            if vector.shape != (expected, 2):
                errors.append("每个振幅必须表示为 [实部, 虚部]")
            elif abs(float(np.sum(vector[:, 0] ** 2 + vector[:, 1] ** 2)) - 1.0) > 1e-8:
                errors.append("自定义初始状态振幅的概率和必须为 1")

    seen: set[str] = set()
    measured_qubits: set[int] = set()
    for pos, step in enumerate(circuit.steps, start=1):
        if step.id in seen:
            errors.append(f"第 {pos} 步的 id `{step.id}` 重复")
        seen.add(step.id)

        if step.target >= q:
            errors.append(f"第 {pos} 步目标量子位 q{step.target} 超出范围")

        if step.type == "MEASURE":
            if step.target in measured_qubits:
                errors.append(f"第 {pos} 步 q{step.target} 已测量，不能重复连接测量节点")
            measured_qubits.add(step.target)

        if step.type in {"RX", "RY", "RZ"} and step.angle is None:
            errors.append(f"第 {pos} 步 {step.type} 旋转必须提供 angle")

        if step.type == "CNOT":
            if step.control is None:
                errors.append(f"第 {pos} 步 CNOT 必须连接控制量子位")
            elif step.control >= q:
                errors.append(f"第 {pos} 步控制量子位 q{step.control} 超出范围")
            elif step.control == step.target:
                errors.append(f"第 {pos} 步 CNOT 的控制位和目标位不能相同")

        if step.type == "SWAP":
            if step.control is None:
                errors.append(f"第 {pos} 步 SWAP 必须提供第二个量子位")
            elif step.control >= q:
                errors.append(f"第 {pos} 步 SWAP 量子位 q{step.control} 超出范围")
            elif step.control == step.target:
                errors.append(f"第 {pos} 步 SWAP 不能连接同一个量子位")

        if (step.type in ONE_QUBIT_GATES or step.type in {"RX", "RY", "RZ", "MEASURE"}) and step.control is not None:
            errors.append(f"第 {pos} 步 {step.type} 不能连接控制端")

    return errors


def require_valid(circuit: CircuitDraft) -> None:
    errors = validate_circuit(circuit)
    if errors:
        raise CircuitValidationError(errors)


def initial_vector(circuit: CircuitDraft) -> np.ndarray:
    require_valid(circuit)
    dim = 2**circuit.qubits
    if circuit.initial_state.kind == "basis":
        state = np.zeros(dim, dtype=complex)
        state[int(circuit.initial_state.basis, 2)] = 1.0
        return state

    values = np.array(circuit.initial_state.amplitudes, dtype=float)
    return values[:, 0] + 1j * values[:, 1]


def _rotation(kind: str, angle: float) -> np.ndarray:
    c = math.cos(angle / 2)
    s = math.sin(angle / 2)
    if kind == "RX":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if kind == "RY":
        return np.array([[c, -s], [s, c]], dtype=complex)
    return np.array([[np.exp(-1j * angle / 2), 0], [0, np.exp(1j * angle / 2)]], dtype=complex)


def _single_matrix(qubits: int, target: int, matrix: np.ndarray) -> np.ndarray:
    full: np.ndarray | None = None
    for qubit in range(qubits):
        part = matrix if qubit == target else np.eye(2, dtype=complex)
        full = part if full is None else np.kron(full, part)
    assert full is not None
    return full


def _bit(value: int, qubits: int, qubit: int) -> int:
    return (value >> (qubits - 1 - qubit)) & 1


def _with_bit(value: int, qubits: int, qubit: int, bit: int) -> int:
    shift = qubits - 1 - qubit
    return (value & ~(1 << shift)) | (bit << shift)


def _two_qubit_permutation(qubits: int, first: int, second: int, controlled_not: bool) -> np.ndarray:
    matrix = np.zeros((2**qubits,) * 2, dtype=complex)
    for source in range(2**qubits):
        target = source
        if controlled_not:
            if _bit(source, qubits, first):
                target = _with_bit(
                    source,
                    qubits,
                    second,
                    1 - _bit(source, qubits, second),
                )
        else:
            target = _with_bit(target, qubits, second, _bit(source, qubits, first))
            target = _with_bit(target, qubits, first, _bit(source, qubits, second))
        matrix[target, source] = 1.0
    return matrix


def apply_step(state: np.ndarray, qubits: int, step: CircuitStep) -> np.ndarray:
    if step.type in ONE_QUBIT_GATES:
        matrix = _single_matrix(qubits, step.target, ONE_QUBIT_GATES[step.type])
    elif step.type in {"RX", "RY", "RZ"}:
        assert step.angle is not None
        matrix = _single_matrix(qubits, step.target, _rotation(step.type, step.angle))
    elif step.type == "CNOT":
        assert step.control is not None
        matrix = _two_qubit_permutation(qubits, step.control, step.target, True)
    elif step.type == "SWAP":
        assert step.control is not None
        matrix = _two_qubit_permutation(qubits, step.target, step.control, False)
    else:
        return state.copy()
    return matrix @ state


def probability_map(state: np.ndarray, qubits: int) -> dict[str, float]:
    probs = np.abs(state) ** 2
    return {format(index, f"0{qubits}b"): float(probs[index]) for index in range(len(state)) if probs[index] > 1e-12}


def measure(
    state: np.ndarray,
    qubits: int,
    target: int,
    measured: dict[str, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, str]:
    dim = 2**qubits
    probs = np.abs(state) ** 2
    outcome = int(rng.choice(np.arange(dim), p=probs))
    bit = _bit(outcome, qubits, target)
    measured[f"q{target}"] = bit

    collapsed = np.zeros(dim, dtype=complex)
    for index, amplitude in enumerate(state):
        if _bit(index, qubits, target) == bit:
            collapsed[index] = amplitude
    norm = np.linalg.norm(collapsed)
    if norm > 0:
        collapsed /= norm

    measured_key = "".join(str(measured.get(f"q{qubit}", 0)) for qubit in sorted(
        (int(key[1:]) for key in measured),
    ))
    return collapsed, bit, measured_key


def vector_parts(state: np.ndarray) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in state]


def execute_next(
    run: RunState,
    steps: list[CircuitStep],
    rng: np.random.Generator,
) -> StepResult:
    if run.complete:
        raise CircuitValidationError("线路已经执行完成")

    state = np.array([part[0] + 1j * part[1] for part in run.state_vector], dtype=complex)
    step = steps[run.index]
    measured = dict(run.measured)
    measured_result = None
    collapsed: list[int] = []
    note = None

    if step.type == "MEASURE":
        state, bit, measured_key = measure(state, run.qubits, step.target, measured, rng)
        measured_result = str(bit)
        collapsed = [step.target]
        note = f"q{step.target} = {bit}，后续步骤继续使用坍缩后的状态"
    else:
        state = apply_step(state, run.qubits, step)

    result = StepResult(
        index=run.index,
        step=step,
        state_vector=vector_parts(state),
        probabilities=probability_map(state, run.qubits),
        measured=None if measured_result is None else {f"q{step.target}": int(measured_result)},
        measurement_result=measured_result,
        collapsed_qubits=collapsed,
        note=note,
    )
    run.index += 1
    run.complete = run.index >= len(steps)
    run.state_vector = result.state_vector
    run.probabilities = result.probabilities
    run.measured = measured
    run.results.append(result)
    return result


def make_run(circuit: CircuitDraft, project_id: str | None = None, version: int | None = None) -> RunState:
    require_valid(circuit)
    state = initial_vector(circuit)
    seed = circuit.seed
    return RunState(
        run_id=str(uuid.uuid4()),
        project_id=project_id,
        version=version,
        qubits=circuit.qubits,
        seed=seed,
        index=0,
        complete=not circuit.steps,
        state_vector=vector_parts(state),
        probabilities=probability_map(state, circuit.qubits),
        results=[],
    )


def run_shots(circuit: CircuitDraft, shots: int) -> ShotSummary:
    require_valid(circuit)
    measurement_steps = [step for step in circuit.steps if step.type == "MEASURE"]
    if not measurement_steps:
        raise CircuitValidationError("重复实验至少需要一个 MEASURE 步骤")
    if shots < 1 or shots > 100000:
        raise CircuitValidationError("shots 必须在 1 到 100000 之间")

    measured_qubits = sorted({step.target for step in measurement_steps})
    counts: dict[str, int] = {}
    rng = np.random.default_rng(circuit.seed)

    for _ in range(shots):
        state = initial_vector(circuit)
        values: dict[int, int] = {}
        for step in circuit.steps:
            if step.type == "MEASURE":
                state, bit, _ = measure(state, circuit.qubits, step.target, {}, rng)
                values[step.target] = bit
            else:
                state = apply_step(state, circuit.qubits, step)
        key = "".join(str(values[qubit]) for qubit in measured_qubits)
        counts[key] = counts.get(key, 0) + 1

    return ShotSummary(
        shots=shots,
        seed=circuit.seed,
        counts=dict(sorted(counts.items())),
        probabilities={key: count / shots for key, count in sorted(counts.items())},
        measured_qubits=measured_qubits,
    )
