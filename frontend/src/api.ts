import type {
  BranchComparison,
  Checkpoint,
  Circuit,
  CircuitStep,
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
    throw { status: response.status, ...(typeof data === "object" && data !== null ? data : { error: String(data) }) };
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

export function getSession(sessionId: string) {
  return request<SessionState>(`/api/sessions/${sessionId}`, "GET");
}

export function saveCheckpoint(sessionId: string, label: string, persist: boolean) {
  return request<Checkpoint>(`/api/sessions/${sessionId}/checkpoint`, "POST", {
    label,
    persist,
  });
}

export function resumeCheckpoint(checkpointId: string) {
  return request<SessionState>("/api/sessions/resume", "POST", { checkpoint_id: checkpointId });
}

export function resumeInlineCheckpoint(checkpoint: Checkpoint) {
  return request<SessionState>("/api/sessions/resume", "POST", { checkpoint });
}

export function compareBranches(
  checkpoint: Checkpoint,
  branches: [CircuitStep[], CircuitStep[]],
  labels: [string, string],
) {
  return request<BranchComparison>("/api/checkpoints/compare", "POST", {
    checkpoint,
    branches,
    labels,
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
