import type { CircuitDraft, OperationInfo, Project, RunState, ShotSummary } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = Array.isArray(body.detail) ? body.detail.join(", ") : String(body.detail ?? detail);
    } catch {
      // Use HTTP detail when JSON is unavailable.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  operations: () => request<OperationInfo[]>("/api/operations"),
  validate: (circuit: CircuitDraft) =>
    request<{ valid: boolean; errors: string[] }>("/api/validate", {
      method: "POST",
      body: JSON.stringify(circuit)
    }),
  createProject: (name: string, circuit: CircuitDraft) =>
    request<Project>("/api/projects", {
      method: "POST",
      body: JSON.stringify({ name, ...circuit })
    }),
  listProjects: () => request<Project[]>("/api/projects"),
  saveProject: (id: string, name: string, circuit: CircuitDraft) =>
    request<Project>("/api/projects/" + id, {
      method: "PUT",
      body: JSON.stringify({ name, circuit })
    }),
  startRun: (circuit: CircuitDraft) =>
    request<RunState>("/api/runs", {
      method: "POST",
      body: JSON.stringify({ circuit })
    }),
  nextStep: (runId: string) =>
    request<RunState>("/api/runs/" + runId + "/next", { method: "POST" }),
  shots: (circuit: CircuitDraft, shots: number) =>
    request<ShotSummary>("/api/shots", {
      method: "POST",
      body: JSON.stringify({ circuit, shots })
    })
};
