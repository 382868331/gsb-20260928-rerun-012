import type { Circuit } from "./types";
import { newStepId } from "./store";

export const SINGLE_QUBIT_SAMPLE: Circuit = {
  qubits: 1,
  initial_basis: "0",
  seed: 42,
  steps: [
    { id: newStepId(), type: "H", target: 0 },
    { id: newStepId(), type: "MEASURE", target: 0 },
  ],
};

export const BELL_PAIR_SAMPLE: Circuit = {
  qubits: 2,
  initial_basis: "00",
  seed: 7,
  steps: [
    { id: newStepId(), type: "H", target: 0 },
    { id: newStepId(), type: "CNOT", control: 0, target: 1 },
    { id: newStepId(), type: "MEASURE", target: 0 },
    { id: newStepId(), type: "MEASURE", target: 1 },
  ],
};
