import { create } from "zustand";
import type { CircuitStep, GateType } from "./types";

let counter = 0;
export function newStepId(): string {
  counter += 1;
  return `step-${Date.now()}-${counter}`;
}

export function makeStep(type: GateType, qubits: number): CircuitStep {
  const step: CircuitStep = { id: newStepId(), type };
  if (type === "IF") {
    step.control = 0;
    step.equals = 1;
    step.then = { id: newStepId(), type: "X", target: Math.min(1, qubits - 1) };
  } else if (type === "CNOT" || type === "CZ") {
    step.control = 0;
    step.target = Math.min(1, qubits - 1);
  } else if (type === "SWAP") {
    step.a = 0;
    step.b = Math.min(1, qubits - 1);
  } else {
    step.target = 0;
  }
  if (type === "RX" || type === "RY" || type === "RZ") {
    step.angle = Math.PI / 2;
  }
  return step;
}

interface BenchState {
  qubits: number;
  initialBasis: string;
  seed: number | null;
  steps: CircuitStep[];
  setQubits: (qubits: number) => void;
  setInitialBasis: (basis: string) => void;
  setSeed: (seed: number | null) => void;
  addStep: (type: GateType) => void;
  updateStep: (id: string, patch: Partial<CircuitStep>) => void;
  removeStep: (id: string) => void;
  moveStep: (id: string, delta: -1 | 1) => void;
  loadCircuit: (data: { qubits: number; initial_basis: string; seed: number | null; steps: CircuitStep[] }) => void;
}

export const useBenchStore = create<BenchState>((set) => ({
  qubits: 1,
  initialBasis: "0",
  seed: 42,
  steps: [{ id: newStepId(), type: "H", target: 0 }],
  setQubits: (qubits) =>
    set((state) => {
      const clampedBasis = state.initialBasis.padEnd(qubits, "0").slice(-qubits);
      return {
        qubits,
        initialBasis: clampedBasis,
        steps: state.steps
          .map((step) => {
            const next = { ...step };
            for (const key of ["target", "control", "a", "b"] as const) {
              if (typeof next[key] === "number" && (next[key] as number) >= qubits) {
                next[key] = qubits - 1;
              }
            }
            if (next.then) {
              const inner = { ...next.then };
              if (typeof inner.target === "number" && inner.target >= qubits) {
                inner.target = qubits - 1;
              }
              next.then = inner;
            }
            return next;
          }),
      };
    }),
  setInitialBasis: (basis) => set({ initialBasis: basis }),
  setSeed: (seed) => set({ seed }),
  addStep: (type) => set((state) => ({ steps: [...state.steps, makeStep(type, state.qubits)] })),
  updateStep: (id, patch) =>
    set((state) => ({
      steps: state.steps.map((step) => (step.id === id ? { ...step, ...patch } : step)),
    })),
  removeStep: (id) => set((state) => ({ steps: state.steps.filter((step) => step.id !== id) })),
  moveStep: (id, delta) =>
    set((state) => {
      const index = state.steps.findIndex((step) => step.id === id);
      const target = index + delta;
      if (index < 0 || target < 0 || target >= state.steps.length) {
        return state;
      }
      const steps = [...state.steps];
      [steps[index], steps[target]] = [steps[target], steps[index]];
      return { steps };
    }),
  loadCircuit: (data) =>
    set({
      qubits: data.qubits,
      initialBasis: data.initial_basis,
      seed: data.seed,
      steps: data.steps.map((step) => ({ ...step, id: newStepId() })),
    }),
}));
