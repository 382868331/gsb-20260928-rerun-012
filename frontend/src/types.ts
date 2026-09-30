export type SingleGateType = "H" | "X" | "Y" | "Z" | "S" | "T";
export type ParamGateType = "RX" | "RY" | "RZ";
export type TwoQubitGateType = "CNOT" | "CZ" | "SWAP";
export type MeasureGateType = "MEASURE";
export type ConditionalGateType = "IF";
export type GateType = SingleGateType | ParamGateType | TwoQubitGateType | MeasureGateType | ConditionalGateType;

export interface CircuitStep {
  id: string;
  type: GateType;
  target?: number;
  control?: number;
  a?: number;
  b?: number;
  angle?: number;
  equals?: 0 | 1;
  then?: CircuitStep;
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
  resumed_from?: string | null;
  restored_measurements?: Array<{ step: number; target: number; outcome: number }>;
  executed?: {
    index: number;
    type: GateType;
    measured: { target: number; outcome: number } | null;
    condition_fired?: boolean | null;
  };
}

export interface Checkpoint {
  session_id?: string;
  checkpoint_id?: string | null;
  label?: string | null;
  saved_at?: number;
  qubits: number;
  initial_basis: string;
  seed: number | null;
  steps: CircuitStep[];
  next_index: number;
  state: Array<[number, number]>;
  measurements: Array<{ step: number; target: number; outcome: number }>;
  outcomes: Record<string, number>;
  rng_state: unknown;
  probabilities: Record<string, number>;
}

export interface BranchResult {
  steps: CircuitStep[];
  measurements: Array<{ step: number; target: number; outcome: number }>;
  final_probabilities: Record<string, number>;
  trace: Array<{
    index: number;
    type: GateType;
    measured: { target: number; outcome: number } | null;
    condition_fired?: boolean | null;
    probabilities: Record<string, number>;
  }>;
}

export interface BranchComparison {
  checkpoint_id?: string | null;
  labels: [string, string];
  branches: [BranchResult, BranchResult];
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
