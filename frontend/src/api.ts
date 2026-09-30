import type {
  BranchComparison,
  BranchExperiment,
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

function stripIds(steps: CircuitStep[]) {
  return steps.map(({ id: _id, ...rest }) => rest);
}

export function createBranchExperiment(
  checkpointId: string,
  branches: Array<{ label: string; steps: CircuitStep[] }>,
  options: { shots: number; budget: number; seed: number | null; name?: string },
) {
  return request<BranchExperiment>(`/api/checkpoints/${checkpointId}/branch-runs`, "POST", {
    name: options.name,
    shots: options.shots,
    budget: options.budget,
    seed: options.seed,
    branches: branches.map((branch) => ({ label: branch.label, steps: stripIds(branch.steps) })),
  });
}

export function getBranchExperiment(runId: string) {
  return request<BranchExperiment>(`/api/branch-runs/${encodeURIComponent(runId)}`, "GET");
}

export function appendBranchShots(runId: string, shots: number, budget?: number) {
  return request<BranchExperiment>(`/api/branch-runs/${encodeURIComponent(runId)}/shots`, "POST", {
    shots,
    ...(budget === undefined ? {} : { budget }),
  });
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
