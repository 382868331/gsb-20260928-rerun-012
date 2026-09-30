import { useState } from "react";
import { Trash2 } from "lucide-react";
import { appendBranchShots, compareBranches, createBranchExperiment, getBranchExperiment } from "../api";
import { newStepId } from "../store";
import type {
  BranchComparison,
  BranchExperiment,
  CheckpointState,
  CircuitStep,
  GateType,
} from "../types";

interface Props {
  checkpoints: CheckpointState[];
  disabled: boolean;
}

const BRANCH_GATES: GateType[] = ["H", "X", "Y", "Z", "CNOT", "MEASURE"];

function emptyBranch() {
  return [{ id: newStepId(), type: "H" as GateType, target: 0 }];
}

export function BranchCompare({ checkpoints, disabled }: Props) {
  const [checkpointId, setCheckpointId] = useState("");
  const [branchA, setBranchA] = useState<CircuitStep[]>([{ id: newStepId(), type: "X", target: 1 }]);
  const [branchB, setBranchB] = useState<CircuitStep[]>(emptyBranch);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<BranchComparison | null>(null);
  const [batchShots, setBatchShots] = useState(100);
  const [batchBudget, setBatchBudget] = useState(100);
  const [appendShots, setAppendShots] = useState(100);
  const [appendBudget, setAppendBudget] = useState(100);
  const [runId, setRunId] = useState("");
  const [experiment, setExperiment] = useState<BranchExperiment | null>(null);

  const checkpoint = checkpoints.find((item) => item.checkpoint_id === checkpointId) ?? null;
  const qubits = checkpoint?.qubits ?? 1;

  function addStep(branch: "A" | "B") {
    const step = { id: newStepId(), type: "H" as GateType, target: 0 };
    if (branch === "A") setBranchA((prev) => [...prev, step]);
    else setBranchB((prev) => [...prev, step]);
  }

  function removeStep(branch: "A" | "B", id: string) {
    if (branch === "A") setBranchA((prev) => prev.filter((step) => step.id !== id));
    else setBranchB((prev) => prev.filter((step) => step.id !== id));
  }

  async function runCompare() {
    if (!checkpoint || loading) return;
    setLoading(true);
    setError("");
    try {
      setResult(await compareBranches(checkpointId, { label: "分支 A", steps: branchA }, { label: "分支 B", steps: branchB }));
    } catch (err) {
      const payload = err as { errors?: Array<{ message: string }>; error?: string };
      setError(payload.errors?.map((item) => item.message).join("; ") ?? payload.error ?? "比较失败");
    } finally {
      setLoading(false);
    }
  }

  async function runBranches() {
    if (!checkpoint || loading) return;
    setLoading(true);
    setError("");
    try {
      const created = await createBranchExperiment(
        checkpointId,
        [
          { label: "分支 A", steps: branchA },
          { label: "分支 B", steps: branchB },
        ],
        { shots: batchShots, budget: Math.max(batchBudget, batchShots), seed: null },
      );
      setRunId(created.run_id);
      setBatchBudget(created.budget);
      setAppendBudget(created.budget);
      setExperiment(created);
    } catch (err) {
      const payload = err as { errors?: Array<{ message: string }>; error?: string };
      setError(payload.errors?.map((item) => item.message).join("; ") ?? payload.error ?? "分支统计失败");
    } finally {
      setLoading(false);
    }
  }

  async function resumeRun() {
    if (!runId.trim() || loading) return;
    setLoading(true);
    setError("");
    try {
      const restored = await getBranchExperiment(runId.trim());
      setExperiment(restored);
      setAppendBudget(restored.budget);
    } catch (err) {
      const payload = err as { error?: string };
      setError(payload.error ?? "恢复分支统计失败");
    } finally {
      setLoading(false);
    }
  }

  async function addShots() {
    if (!experiment || loading) return;
    setLoading(true);
    setError("");
    try {
      setExperiment(await appendBranchShots(experiment.run_id, appendShots, Math.max(experiment.budget, appendBudget)));
    } catch (err) {
      const payload = err as { error?: string };
      setError(payload.error ?? "追加样本失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel">
      <h2>保存点后续分支比较</h2>
      <p className="muted">
        选择一个保存点，为两个分支各自添加后续步骤；比较时两个分支都从同一坍缩状态与随机数位置出发，保存前已发生的测量不会重新抽取。
      </p>
      <select value={checkpointId} onChange={(event) => { setCheckpointId(event.target.value); setResult(null); }}>
        <option value="">选择保存点…</option>
        {checkpoints.map((item) => (
          <option key={item.checkpoint_id} value={item.checkpoint_id}>
            {item.name}（第 {item.next_index}/{item.total_steps} 步）
          </option>
        ))}
      </select>

      {checkpoint && (
        <>
          <div className="branch-grid">
            <BranchEditor label="分支 A" steps={branchA} qubits={qubits} onAdd={() => addStep("A")} onRemove={(id) => removeStep("A", id)} onChange={setBranchA} />
            <BranchEditor label="分支 B" steps={branchB} qubits={qubits} onAdd={() => addStep("B")} onRemove={(id) => removeStep("B", id)} onChange={setBranchB} />
          </div>
          <div className="inline-controls">
            <button type="button" className="primary" onClick={runCompare} disabled={disabled || loading}>
              {loading ? "比较中…" : "比较两个后续分支"}
            </button>
            <button type="button" onClick={runBranches} disabled={disabled || loading}>
              运行独立分支统计
            </button>
            <span className="muted">可添加的门：{BRANCH_GATES.join(" / ")}</span>
          </div>
          <div className="inline-controls">
            <label className="field">
              本次样本
              <input type="number" min={0} max={100000} value={batchShots} onChange={(event) => setBatchShots(Number(event.target.value))} />
            </label>
            <label className="field">
              预算上限
              <input type="number" min={1} max={100000} value={batchBudget} onChange={(event) => setBatchBudget(Number(event.target.value))} />
            </label>
          </div>
          {error && <div className="error-box">{error}</div>}
          {experiment && (
            <div className="branch-result saved-branch">
              <div className="inline-controls">
                <strong>独立分支统计：{experiment.run_id}</strong>
                <input value={runId} onChange={(event) => setRunId(event.target.value)} aria-label="分支统计运行 ID" />
                <button type="button" onClick={resumeRun} disabled={loading}>单独恢复</button>
                <input type="number" min={1} max={100000} value={appendShots} onChange={(event) => setAppendShots(Number(event.target.value))} aria-label="追加样本数" />
                <input type="number" min={experiment.budget} max={100000} value={appendBudget} onChange={(event) => setAppendBudget(Number(event.target.value))} aria-label="新预算上限" />
                <button type="button" onClick={addShots} disabled={loading}>追加样本</button>
              </div>
              <p className="muted">
                共同前缀：保存点「{experiment.checkpoint.name}」前 {experiment.checkpoint.prefix_next_index} 步，
                测量 {JSON.stringify(experiment.checkpoint.measurements)}；各分支结果仅统计自身样本来源。
              </p>
              <div className="branch-grid">
                {experiment.branches.map((branch) => (
                  <div key={branch.branch_id}>
                    <strong>{branch.label}</strong>
                    <p className="muted">条件：{branch.condition_steps.length ? JSON.stringify(branch.condition_steps) : "无操作"}</p>
                    <p className="muted">样本数 {branch.shots}；来源 {branch.sample_source}</p>
                    <div className="prob-compact">
                      {Object.entries(branch.frequencies).map(([label, value]) => (
                        <span key={label} className={value > 0 ? "prob-hot" : ""}>
                          |{label}⟩ {(value * 100).toFixed(1)}% ({branch.counts[label]})
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {result && (
            <div className="branch-results">
              {result.branches.map((branch) => (
                <div key={branch.label} className="branch-result">
                  <strong>{branch.label}</strong>
                  <ul className="muted">
                    {branch.steps.map((step) => (
                      <li key={step.index}>
                        {step.type}
                        {step.measured ? ` → q${step.measured.target}=${step.measured.outcome}` : ""}
                      </li>
                    ))}
                    {branch.steps.length === 0 && <li>（无后续步骤）</li>}
                  </ul>
                  <div className="prob-compact">
                    {Object.entries(branch.final_probabilities).map(([label, value]) => (
                      <span key={label} className={value > 0 ? "prob-hot" : ""}>
                        |{label}⟩ {value.toFixed(3)}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </section>
  );
}

interface EditorProps {
  label: string;
  steps: CircuitStep[];
  qubits: number;
  onAdd: () => void;
  onRemove: (id: string) => void;
  onChange: (steps: CircuitStep[]) => void;
}

function BranchEditor({ label, steps, qubits, onAdd, onRemove, onChange }: EditorProps) {
  const options = Array.from({ length: qubits }, (_, q) => q);
  return (
    <div className="branch-editor">
      <div className="branch-head">
        <strong>{label}</strong>
        <button type="button" onClick={onAdd}>+ 步骤</button>
      </div>
      {steps.map((step, index) => (
        <div key={step.id} className="branch-step">
          <span className="step-index">#{index + 1}</span>
          <select
            value={step.type}
            onChange={(event) =>
              onChange(steps.map((item) => (item.id === step.id ? { ...item, type: event.target.value as GateType } : item)))
            }
          >
            {BRANCH_GATES.map((gate) => (
              <option key={gate} value={gate}>{gate}</option>
            ))}
          </select>
          {step.type === "CNOT" ? (
            <>
              <select
                value={step.control ?? 0}
                onChange={(event) =>
                  onChange(steps.map((item) => (item.id === step.id ? { ...item, control: Number(event.target.value) } : item)))
                }
              >
                {options.map((q) => <option key={q} value={q}>c q{q}</option>)}
              </select>
              <select
                value={step.target ?? 0}
                onChange={(event) =>
                  onChange(steps.map((item) => (item.id === step.id ? { ...item, target: Number(event.target.value) } : item)))
                }
              >
                {options.map((q) => <option key={q} value={q}>t q{q}</option>)}
              </select>
            </>
          ) : (
            <select
              value={step.target ?? 0}
              onChange={(event) =>
                onChange(steps.map((item) => (item.id === step.id ? { ...item, target: Number(event.target.value) } : item)))
              }
            >
              {options.map((q) => <option key={q} value={q}>q{q}</option>)}
            </select>
          )}
          <button type="button" aria-label="删除" onClick={() => onRemove(step.id)}>
            <Trash2 size={13} />
          </button>
        </div>
      ))}
    </div>
  );
}
