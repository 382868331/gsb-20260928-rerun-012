import type { CircuitDraft, GateType } from "./types";

export function singleQubitSample(): CircuitDraft {
  return {
    qubits: 1,
    initial_state: { kind: "basis", basis: "0", amplitudes: null },
    seed: 42,
    steps: [
      { id: crypto.randomUUID(), type: "H", target: 0, control: null, angle: null },
      { id: crypto.randomUUID(), type: "MEASURE", target: 0, control: null, angle: null }
    ]
  };
}

export function bellSample(): CircuitDraft {
  return {
    qubits: 2,
    initial_state: { kind: "basis", basis: "00", amplitudes: null },
    seed: 42,
    steps: [
      { id: crypto.randomUUID(), type: "H", target: 0, control: null, angle: null },
      { id: crypto.randomUUID(), type: "CNOT", target: 1, control: 0, angle: null },
      { id: crypto.randomUUID(), type: "MEASURE", target: 0, control: null, angle: null },
      { id: crypto.randomUUID(), type: "MEASURE", target: 1, control: null, angle: null }
    ]
  };
}

export function emptyCircuit(qubits = 2): CircuitDraft {
  return {
    qubits,
    initial_state: { kind: "basis", basis: "0".repeat(qubits), amplitudes: null },
    seed: 123,
    steps: []
  };
}

export const gateOptions: GateType[] = ["X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ", "CNOT", "SWAP", "MEASURE"];
