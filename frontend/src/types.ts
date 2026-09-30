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
  notice?: string;
  resumed_from?: string;
}

export interface CheckpointState {
  checkpoint_id: string;
  session_id: string | null;
  name: string;
  qubits: number;
  initial_basis: string;
  seed: number | null;
  next_index: number;
  total_steps: number;
  probabilities: Record<string, number>;
  measurements: Array<{ step: number; target: number; outcome: number }>;
}

export interface BranchResult {
  label: string;
  steps: Array<{
    index: number;
    type: GateType;
    measured: { target: number; outcome: number } | null;
    probabilities: Record<string, number>;
  }>;
  measurements: Array<{ step: number; target: number; outcome: number }>;
  final_probabilities: Record<string, number>;
}

export interface BranchComparison {
  checkpoint_id: string;
  checkpoint: CheckpointState;
  branches: [BranchResult, BranchResult];
}

export interface BranchExperimentSpec {
  branch_id?: string;
  label: string;
  steps: CircuitStep[];
  budget: number;
}

export interface BranchExperimentResult {
  branch_id: string;
  label: string;
  condition_steps: Array<Omit<CircuitStep, "id">>;
  budget: number;
  completed_shots: number;
  counts: Record<string, number>;
  frequencies: Record<string, number>;
  batches: Array<{ batch_id: string; source: string; shots: number; total_shots_after_batch: number }>;
}

export interface BranchExperiment {
  experiment_id: string;
  name: string;
  base_seed: number;
  created_at: number;
  storage_path: string;
  pooled_across_conditions: boolean;
  common_prefix: {
    checkpoint_id: string;
    qubits: number;
    initial_basis: string;
    prefix_seed: number | null;
    prefix_steps: CircuitStep[];
    prefix_next_index: number;
    prefix_measurements: Array<{ step: number; target: number; outcome: number }>;
  };
  branches: BranchExperimentResult[];
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
