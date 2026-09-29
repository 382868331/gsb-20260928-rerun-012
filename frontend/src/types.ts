export type GateType = "X" | "Y" | "Z" | "H" | "S" | "T" | "RX" | "RY" | "RZ" | "CNOT" | "SWAP" | "MEASURE";

export interface CircuitStep {
  id: string;
  type: GateType;
  target: number;
  control: number | null;
  angle: number | null;
  label?: string;
}

export interface InitialState {
  kind: "basis" | "custom";
  basis: string;
  amplitudes: [number, number][] | null;
}

export interface CircuitDraft {
  qubits: number;
  initial_state: InitialState;
  seed: number | null;
  steps: CircuitStep[];
}

export interface OperationInfo {
  type: GateType;
  name: string;
  category: string;
  target_required: boolean;
  control_required: boolean;
  angle_required: boolean;
  description: string;
}

export interface StepResult {
  index: number;
  step: CircuitStep;
  probabilities: Record<string, number>;
  measured: Record<string, number> | null;
  measurement_result: string | null;
  collapsed_qubits: number[];
  note: string | null;
}

export interface RunState {
  run_id: string;
  project_id: string | null;
  version: number | null;
  qubits: number;
  seed: number | null;
  index: number;
  complete: boolean;
  probabilities: Record<string, number>;
  measured: Record<string, number>;
  results: StepResult[];
}

export interface VersionInfo {
  version: number;
  created_at: string;
  note: string;
  circuit: CircuitDraft;
}

export interface Project {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  current_version: number;
  versions: VersionInfo[];
}

export interface ShotSummary {
  shots: number;
  seed: number | null;
  counts: Record<string, number>;
  probabilities: Record<string, number>;
  measured_qubits: number[];
}
