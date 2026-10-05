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
  filters?: MeasurementFilter[];
  budget: number;
}

export interface MeasurementFilter {
  step: number;
  outcome: 0 | 1;
}

export interface BranchExperimentResult {
  branch_id: string;
  label: string;
  condition_steps: Array<Omit<CircuitStep, "id">>;
  filters: MeasurementFilter[];
  budget: number;
  completed_shots: number;
  attempts: number;
  passed_shots: number;
  pass_rate: number | null;
  has_evidence: boolean;
  sample_scope: string;
  counts: Record<string, number>;
  frequencies: Record<string, number> | null;
  batches: Array<{ batch_id: string; source: string; shots: number; total_shots_after_batch: number }>;
  pass_interval?: [number, number] | null;
  event_interval?: [number, number];
  event_width?: number;
  allocation_score?: number;
}

export interface InquiryDecision {
  decision_id: string;
  at: number;
  action: "created" | "sample" | "stop";
  branch_id: string | null;
  shots: number;
  reason: string;
  batch_id: string | null;
  before_statistics?: BranchInquiryStat[];
  evaluation_after?: InquiryEvaluation;
}

export interface BranchInquiryStat {
  branch_id: string;
  label: string;
  attempts: number;
  passed_shots: number | null;
  event: string;
  event_successes: number;
  event_trials: number;
  event_interval: [number, number];
  event_width: number;
  pass_interval: [number, number] | null;
  allocation_score: number;
  evidence_note: string;
  difference_interval?: [number, number];
}

export interface InquiryEvaluation {
  status: "continue" | "decided" | "inconclusive_budget_exhausted";
  cs_rule_version?: string;
  remaining_budget: number;
  used_budget: number;
  branch_statistics: BranchInquiryStat[];
  conclusion: { type: string; message: string } | null;
}

export interface BranchInquiry {
  inquiry_id: string;
  revision: number;
  created_at: number;
  mode: "single" | "compare";
  branch_ids: [string] | [string, string];
  event: string;
  margin: number;
  total_budget: number;
  batch_size: number;
  alpha: number;
  cs_rule_version?: string;
  cs_rule_note?: string;
  immutable: boolean;
  status: "active" | "decided" | "inconclusive_budget_exhausted";
  current_evaluation: InquiryEvaluation;
  decisions: Array<InquiryDecision & { cs_rule_version?: string }>;
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
  inquiries: BranchInquiry[];
  research_plans: ResearchPlan[];
}

export interface ResearchGoalDecision {
  decision_id: string;
  at: number;
  action: "created" | "sample" | "stop" | "cancel" | "adopt_existing_samples";
  branch_id: string | null;
  shots: number;
  reason: string;
  batch_id: string | null;
  batch_ids: string[];
  evaluation_after?: InquiryEvaluation;
}

export interface ResearchGoal {
  goal_id: string;
  label: string;
  priority: number;
  mode: "single" | "compare";
  branch_ids: [string] | [string, string];
  event: string;
  margin: number;
  total_budget: number;
  batch_size: number;
  alpha: number;
  status: "active" | "decided" | "inconclusive_budget_exhausted" | "cancelled";
  adopted_batch_ids: string[];
  current_evaluation: InquiryEvaluation & { adopted_shots: number };
  decisions: ResearchGoalDecision[];
}

export interface ResearchPlanBatch {
  batch_id: string;
  branch_id: string;
  shots: number;
  leader_goal_id: string;
  goal_ids: string[];
}

export interface ResearchPlan {
  plan_id: string;
  created_at: number;
  name: string;
  strategy: "balanced" | "priority";
  total_budget: number;
  executed_shots: number;
  remaining_budget: number;
  priority_goal_ids: string[];
  goal_order: string[];
  goals: ResearchGoal[];
  batches: ResearchPlanBatch[];
  next_batch: {
    leader_goal_id: string;
    branch_id: string;
    shots: number;
    eligible_goal_ids: string[];
  } | null;
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
