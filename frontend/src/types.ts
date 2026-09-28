export type SingleGateType = "H" | "X" | "Y" | "Z" | "S" | "T";
export type ParamGateType = "RX" | "RY" | "RZ";
export type TwoQubitGateType = "CNOT" | "CZ" | "SWAP";
export type GateType = SingleGateType | ParamGateType | TwoQubitGateType | "MEASURE";

export interface CircuitStep {
  id: string;
  type: GateType;
  target?: number;
  control?: number;
  a?: number;
  b?: number;
  angle?: number;
}

export interface Circuit {
  qubits: number;
  initial_basis: string;
  seed: number | null;
  steps: CircuitStep[];
}

export interface ValidationError {
  step: number;
  message: string;
}

export interface SessionState {
  session_id: string;
  qubits: number;
  initial_basis: string;
  seed: number | null;
  next_index: number;
  total_steps: number;
  finished: boolean;
  probabilities: Record<string, number>;
  measurements: Array<{ step: number; target: number; outcome: number }>;
  executed?: { index: number; type: GateType; measured: { target: number; outcome: number } | null };
}

export interface ExperimentResult {
  shots: number;
  seed: number | null;
  counts: Record<string, number>;
  frequencies: Record<string, number>;
  expected_probabilities: Record<string, number>;
}

export interface SavedCircuit extends Circuit {
  name: string;
  version: number;
  updated_at: number;
}
