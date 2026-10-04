import type {
  BranchComparison,
  BranchExperiment,
  BranchExperimentSpec,
  BranchInquiry,
  CheckpointState,
  CircuitStep,
  Circuit,
  ExperimentResult,
  SavedCircuit,
  SessionState,
  ValidationError,
} from "./types";

async function request<T>(url: string, method: string, body?: unknown): Promise<T> {
  const response = await fetch(url, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    throw data;
  }
  return data as T;
}

export function validateCircuit(circuit: Circuit) {
  return request<{ valid: boolean; errors: ValidationError[] }>("/api/validate", "POST", circuit);
}

export function createSession(circuit: Circuit) {
  return request<SessionState>("/api/sessions", "POST", circuit);
}

export function advanceSession(sessionId: string) {
  return request<SessionState>(`/api/sessions/${sessionId}/step`, "POST", {});
}

export function saveCheckpoint(sessionId: string, name: string) {
  return request<CheckpointState>(`/api/sessions/${sessionId}/checkpoints`, "POST", { name });
}

export function resumeCheckpoint(checkpointId: string) {
  return request<SessionState>(`/api/checkpoints/${checkpointId}/resume`, "POST", {});
}

export function compareBranches(
  checkpointId: string,
  branchA: { label?: string; steps: CircuitStep[] },
  branchB: { label?: string; steps: CircuitStep[] },
) {
  return request<BranchComparison>(`/api/checkpoints/${checkpointId}/compare`, "POST", {
    branch_a: { label: branchA.label, steps: stripIds(branchA.steps) },
    branch_b: { label: branchB.label, steps: stripIds(branchB.steps) },
  });
}

export function createBranchExperiment(
  checkpointId: string,
  branches: BranchExperimentSpec[],
  options: { seed?: number | null; storageDir?: string } = {},
) {
  return request<BranchExperiment>(`/api/checkpoints/${checkpointId}/branch-experiments`, "POST", {
    name: "conditional-branches",
    seed: options.seed ?? null,
    storage_dir: options.storageDir || undefined,
    branches: branches.map((branch) => ({
      branch_id: branch.branch_id,
      label: branch.label,
      budget: branch.budget,
      steps: stripIds(branch.steps),
      filters: branch.filters ?? [],
    })),
  });
}

export function addBranchShots(
  experimentId: string,
  branchId: string,
  shots: number,
  source = "ui-continued",
  storageDir?: string,
) {
  return request<BranchExperiment>(
    `/api/branch-experiments/${encodeURIComponent(experimentId)}/branches/${encodeURIComponent(branchId)}/shots`,
    "PATCH",
    { shots, source, storage_dir: storageDir },
  );
}

export function getBranchExperiment(experimentId: string) {
  return request<BranchExperiment>(`/api/branch-experiments/${encodeURIComponent(experimentId)}`, "GET");
}

export function createBranchInquiry(experimentId: string, inquiry: {
  mode: "single" | "compare";
  branch_ids: string[];
  event: string;
  margin: number;
  total_budget: number;
  batch_size: number;
}): Promise<BranchExperiment> {
  return request<BranchExperiment>(
    `/api/branch-experiments/${encodeURIComponent(experimentId)}/inquiries`,
    "PATCH",
    { inquiry },
  );
}

export function runAdaptiveBatch(experimentId: string, inquiryId: string, batchSize?: number) {
  return request<{ evaluation: BranchInquiry["current_evaluation"] } & BranchExperiment>(
    `/api/branch-experiments/${encodeURIComponent(experimentId)}/inquiries/${encodeURIComponent(inquiryId)}/adaptive-batch`,
    "PATCH",
    batchSize === undefined ? {} : { batch_size: batchSize },
  );
}

function stripIds(steps: CircuitStep[]) {
  return steps.map(({ id: _id, ...rest }) => rest);
}

export function runExperiment(circuit: Circuit, shots: number) {
  return request<ExperimentResult>("/api/experiments", "POST", { ...circuit, shots });
}

export function saveCircuit(name: string, circuit: Circuit) {
  return request<SavedCircuit>(`/api/circuits/${encodeURIComponent(name)}`, "PUT", circuit);
}

export function listCircuits() {
  return request<{ circuits: Record<string, SavedCircuit> }>("/api/circuits", "GET");
}
