import { useState } from "react";
import { Trash2 } from "lucide-react";
import { compareBranches } from "../api";
import { newStepId } from "../store";
import type { BranchComparison, CheckpointState, CircuitStep, GateType } from "../types";

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
            <span className="muted">可添加的门：{BRANCH_GATES.join(" / ")}</span>
          </div>
          {error && <div className="error-box">{error}</div>}
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
