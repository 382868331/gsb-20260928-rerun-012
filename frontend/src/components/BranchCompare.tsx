import { useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import { addBranchShots, createBranchExperiment, getBranchExperiment } from "../api";
import { newStepId } from "../store";
import type {
  BranchExperiment,
  CheckpointState,
  CircuitStep,
  GateType,
  MeasurementFilter,
} from "../types";

interface Props {
  checkpoints: CheckpointState[];
  disabled: boolean;
}

interface BranchDraft {
  id: string;
  label: string;
  budget: number;
  nextShots: number;
  steps: CircuitStep[];
  filters: MeasurementFilter[];
}

const BRANCH_GATES: GateType[] = ["H", "X", "Y", "Z", "CNOT", "CZ", "MEASURE"];

function makeGate(qubits: number, type: GateType = "H"): CircuitStep {
  const step: CircuitStep = { id: newStepId(), type };
  if (type === "CNOT" || type === "CZ") {
    step.control = 0;
    step.target = Math.min(1, qubits - 1);
  } else {
    step.target = 0;
  }
  return step;
}

function defaultBranches(qubits: number): BranchDraft[] {
  return [
    {
      id: "branch-1",
      label: "条件操作 A",
      budget: 256,
      nextShots: 256,
      steps: [{ id: newStepId(), type: "X", target: Math.min(1, qubits - 1) }],
      filters: [],
    },
    { id: "branch-2", label: "条件操作 B", budget: 256, nextShots: 256, steps: [], filters: [] },
  ];
}

function updateFilter(
  filters: MeasurementFilter[],
  index: number,
  patch: Partial<MeasurementFilter>,
): MeasurementFilter[] {
  return filters.map((filter, filterIndex) =>
    filterIndex === index ? { ...filter, ...patch } : filter,
  );
}

export function BranchCompare({ checkpoints, disabled }: Props) {
  const [checkpointId, setCheckpointId] = useState("");
  const [branches, setBranches] = useState<BranchDraft[]>(() => defaultBranches(1));
  const [loadingId, setLoadingId] = useState("");
  const [error, setError] = useState("");
  const [result, setResult] = useState<BranchExperiment | null>(null);
  const [restoreId, setRestoreId] = useState("");

  const checkpoint = checkpoints.find((item) => item.checkpoint_id === checkpointId) ?? null;
  const busy = disabled || loadingId !== "";

  function updateBranch(id: string, patch: Partial<BranchDraft>) {
    setBranches((prev) => prev.map((branch) => (branch.id === id ? { ...branch, ...patch } : branch)));
  }

  function updateStep(branchId: string, stepId: string, patch: Partial<CircuitStep>) {
    setBranches((prev) =>
      prev.map((branch) =>
        branch.id === branchId
          ? { ...branch, steps: branch.steps.map((step) => (step.id === stepId ? { ...step, ...patch } : step)) }
          : branch,
      ),
    );
  }

  function selectCheckpoint(id: string) {
    setCheckpointId(id);
    setResult(null);
    setError("");
  }

  async function runExperiment() {
    if (!checkpoint || busy) return;
    if (branches.length === 0 || branches.some((branch) => !branch.label.trim() || branch.budget < 0)) {
      setError("请为每个分支填写唯一名称，并将初始样本数设为非负整数。");
      return;
    }
    setLoadingId("create");
    setError("");
    try {
      setResult(
        await createBranchExperiment(
          checkpointId,
          branches.map((branch) => ({
            branch_id: branch.id,
            label: branch.label.trim(),
            budget: branch.budget,
            steps: branch.steps,
            filters: branch.filters.length ? branch.filters : undefined,
          })),
          { seed: checkpoint.seed },
        ),
      );
    } catch (err) {
      const payload = err as { errors?: Array<{ message: string }>; error?: string };
      setError(payload.errors?.map((item) => item.message).join("; ") ?? payload.error ?? "分支实验失败");
    } finally {
      setLoadingId("");
    }
  }

  async function continueBranch(branchId: string, shots: number) {
    if (!result || busy) return;
    if (!Number.isInteger(shots) || shots <= 0) {
      setError("追加样本数必须是正整数。");
      return;
    }
    setLoadingId(branchId);
    setError("");
    try {
      setResult(await addBranchShots(result.experiment_id, branchId, shots));
    } catch (err) {
      const payload = err as { error?: string };
      setError(payload.error ?? "追加样本失败");
    } finally {
      setLoadingId("");
    }
  }

  async function restoreExperiment() {
    const id = restoreId.trim();
    if (!id || busy) return;
    setLoadingId("restore");
    setError("");
    try {
      const restored = await getBranchExperiment(id);
      setResult(restored);
      setCheckpointId(restored.common_prefix.checkpoint_id);
      setBranches(
        restored.branches.map((branch) => ({
          id: branch.branch_id,
          label: branch.label,
          budget: branch.budget,
          nextShots: 256,
          steps: branch.condition_steps.map((step) => ({ ...(step as CircuitStep), id: newStepId() })),
          filters: branch.filters.map((filter) => ({ ...filter })),
        })),
      );
    } catch (err) {
      const payload = err as { error?: string };
      setError(payload.error ?? "恢复分支实验失败");
    } finally {
      setLoadingId("");
    }
  }

  return (
    <section className="panel">
      <h2>中途测量后的条件分支实验</h2>
      <p className="muted">
        各分支共享保存点之前的状态、测量事实和前缀说明；后续条件、随机流、计数与批次互相独立，统计结果不会跨条件合并。
      </p>
      <div className="inline-controls">
        <input value={restoreId} onChange={(event) => setRestoreId(event.target.value)} placeholder="按实验 ID 恢复已保存结果…" />
        <button type="button" onClick={restoreExperiment} disabled={busy || !restoreId.trim()}>
          {loadingId === "restore" ? "恢复中…" : "恢复实验"}
        </button>
      </div>
      <select value={checkpointId} onChange={(event) => selectCheckpoint(event.target.value)}>
        <option value="">选择中途保存点…</option>
        {checkpoints.map((item) => (
          <option key={item.checkpoint_id} value={item.checkpoint_id}>
            {item.name}（共同前缀第 {item.next_index}/{item.total_steps} 步）
          </option>
        ))}
      </select>

      {checkpoint && (
        <>
          <div className="branch-grid">
            {branches.map((branch, index) => (
              <BranchEditor
                key={branch.id}
                branch={branch}
                index={index}
                qubits={checkpoint.qubits}
                canRemove={branches.length > 1}
                onUpdate={(patch) => updateBranch(branch.id, patch)}
                onRemove={() => setBranches((prev) => prev.filter((item) => item.id !== branch.id))}
                onStep={updateStep}
              />
            ))}
          </div>
          <div className="inline-controls">
            <button
              type="button"
              onClick={() =>
                setBranches((prev) => [
                  ...prev,
                  {
                    id: `branch-${Date.now()}`,
                    label: `条件操作 ${String.fromCharCode(65 + prev.length)}`,
                    budget: 256,
                    nextShots: 256,
                    steps: [makeGate(checkpoint.qubits)],
                    filters: [],
                  },
                ])
              }
              disabled={busy || branches.length >= 8}
            >
              <Plus size={14} /> 添加方案
            </button>
            <button type="button" className="primary" onClick={runExperiment} disabled={busy}>
              {loadingId === "create" ? "运行中…" : "运行独立分支样本"}
            </button>
          </div>
          {error && <div className="error-box">{error}</div>}
          {result && <ExperimentResult result={result} branches={branches} busy={busy} loadingId={loadingId} onContinue={continueBranch} />}
        </>
      )}
    </section>
  );
}

interface EditorProps {
  branch: BranchDraft;
  index: number;
  qubits: number;
  canRemove: boolean;
  onUpdate: (patch: Partial<BranchDraft>) => void;
  onRemove: () => void;
  onStep: (branchId: string, stepId: string, patch: Partial<CircuitStep>) => void;
}

function BranchEditor({ branch, qubits, canRemove, onUpdate, onRemove, onStep }: EditorProps) {
  const options = Array.from({ length: qubits }, (_, q) => q);
  return (
    <div className="branch-editor">
      <div className="branch-head">
        <input value={branch.label} onChange={(event) => onUpdate({ label: event.target.value })} />
        <button type="button" aria-label="删除分支" onClick={onRemove} disabled={!canRemove}>
          <Trash2 size={13} />
        </button>
      </div>
      <label className="field">
        初始样本预算
        <input type="number" min={0} max={100000} value={branch.budget} onChange={(event) => onUpdate({ budget: Number(event.target.value) })} />
      </label>
      <div className="filter-block">
        <span className="muted">只保留中途测量结果（按顺序连续筛选）</span>
        {branch.filters.map((filter, filterIndex) => (
          <div key={filterIndex} className="branch-step">
            <span className="step-index">筛选 {filterIndex + 1}</span>
            <select
              value={filter.target}
              onChange={(event) => onUpdate({ filters: updateFilter(branch.filters, filterIndex, { target: Number(event.target.value) }) })}
            >
              {options.map((q) => <option key={q} value={q}>q{q}</option>)}
            </select>
            <span>=</span>
            <select
              value={filter.outcome}
              onChange={(event) => onUpdate({ filters: updateFilter(branch.filters, filterIndex, { outcome: Number(event.target.value) as 0 | 1 }) })}
            >
              <option value={0}>0</option>
              <option value={1}>1</option>
            </select>
            <button
              type="button"
              aria-label="删除筛选"
              onClick={() => onUpdate({ filters: branch.filters.filter((_, index) => index !== filterIndex) })}
            >
              <Trash2 size={12} />
            </button>
          </div>
        ))}
        <button
          type="button"
          onClick={() => onUpdate({ filters: [...branch.filters, { target: 0, outcome: 0 }] })}
        >
          + 测量筛选
        </button>
      </div>
      {branch.steps.map((step, stepIndex) => (
        <div key={step.id} className="branch-step">
          <span className="step-index">#{stepIndex + 1}</span>
          <select value={step.type} onChange={(event) => onStep(branch.id, step.id, { type: event.target.value as GateType, control: undefined, target: 0 })}>
            {BRANCH_GATES.map((gate) => <option key={gate} value={gate}>{gate}</option>)}
          </select>
          {step.type === "CNOT" || step.type === "CZ" ? (
            <>
              <select value={step.control ?? 0} onChange={(event) => onStep(branch.id, step.id, { control: Number(event.target.value) })}>
                {options.map((q) => <option key={q} value={q}>c q{q}</option>)}
              </select>
              <select value={step.target ?? 0} onChange={(event) => onStep(branch.id, step.id, { target: Number(event.target.value) })}>
                {options.map((q) => <option key={q} value={q}>t q{q}</option>)}
              </select>
            </>
          ) : (
            <select value={step.target ?? 0} onChange={(event) => onStep(branch.id, step.id, { target: Number(event.target.value) })}>
              {options.map((q) => <option key={q} value={q}>q{q}</option>)}
            </select>
          )}
          <button type="button" aria-label="删除步骤" onClick={() => onUpdate({ steps: branch.steps.filter((item) => item.id !== step.id) })}>
            <Trash2 size={12} />
          </button>
        </div>
      ))}
      <button type="button" onClick={() => onUpdate({ steps: [...branch.steps, makeGate(qubits)] })}>+ 条件步骤</button>
    </div>
  );
}

function ExperimentResult({
  result,
  branches,
  busy,
  loadingId,
  onContinue,
}: {
  result: BranchExperiment;
  branches: BranchDraft[];
  busy: boolean;
  loadingId: string;
  onContinue: (branchId: string, shots: number) => void;
}) {
  return (
    <div className="branch-results">
      <div className="notice-box">实验 {result.experiment_id.slice(0, 8)} 已保存到：{result.storage_path}</div>
      <ul className="muted">
        <li>共同前缀：保存点 {result.common_prefix.checkpoint_id.slice(0, 8)}，已执行 {result.common_prefix.prefix_next_index} 步。</li>
        <li>前缀测量：{result.common_prefix.prefix_measurements.map((item) => `q${item.target}=${item.outcome}`).join("，") || "无"}</li>
        <li>基础种子：{result.base_seed}；各分支由基础种子和分支 ID 派生独立随机流。</li>
      </ul>
      {result.branches.map((branch) => {
        const draft = branches.find((item) => item.id === branch.branch_id);
        return (
          <div key={branch.branch_id} className="branch-result">
            <strong>{branch.label}</strong>
            <p className="muted">
              样本范围：{branch.sample_scope}。筛选条件：
              {branch.filters.length
                ? branch.filters.map((filter) => `q${filter.target}=${filter.outcome}`).join("，")
                : "无"}
            </p>
            <p className="muted">
              总尝试数 {branch.total_attempts}；通过筛选 {branch.passed_shots}；未通过 {branch.rejected_shots}。
              批次：{branch.batches
                .map((batch) => `${batch.source}(尝试 ${batch.attempts}/通过 ${batch.accepted_shots})`)
                .join(" + ")}
            </p>
            {branch.filters.length > 0 && (
              <p className="muted">
                各筛选阶段未通过数：{branch.stage_rejected.map((count, index) => `筛选 ${index + 1}: ${count}`).join("；")}
              </p>
            )}
            {branch.cap_reached && (
              <div className="error-box">尝试上限已用完，当前条件仍没有足够命中；以下分布只代表已通过筛选的样本。</div>
            )}
            {branch.frequencies === null ? (
              <div className="notice-box">
                没有任何样本通过筛选：目前没有证据，不能解释为各结果概率均为 0。精确理论接受概率为
                {(branch.exact.accepted_probability * 100).toFixed(2)}%。
              </div>
            ) : (
              <div className="prob-compact">
                {Object.entries(branch.frequencies).map(([label, value]) => (
                  <span key={label} className={value > 0 ? "prob-hot" : ""}>
                    |{label}⟩ {branch.counts[label]}（{(value * 100).toFixed(1)}%）
                  </span>
                ))}
              </div>
            )}
            <p className="muted">
              短线路精确枚举：联合接受概率 {(branch.exact.accepted_probability * 100).toFixed(2)}%
              {branch.exact.conditional_probabilities
                ? `；条件分布 ${Object.entries(branch.exact.conditional_probabilities)
                    .map(([label, value]) => `|${label}⟩=${(value * 100).toFixed(1)}%`)
                    .join("，")}`
                : "；条件概率无定义（筛选事件概率为 0）"}
            </p>
            <ContinueShots branchId={branch.branch_id} defaultValue={draft?.nextShots ?? 256} busy={busy} loading={loadingId} onContinue={onContinue} />
          </div>
        );
      })}
    </div>
  );
}

function ContinueShots({
  branchId,
  defaultValue,
  busy,
  loading,
  onContinue,
}: {
  branchId: string;
  defaultValue: number;
  busy: boolean;
  loading: string;
  onContinue: (branchId: string, shots: number) => void;
}) {
  const [shots, setShots] = useState(defaultValue);
  return (
    <div className="inline-controls">
      <input type="number" min={1} max={100000} value={shots} onChange={(event) => setShots(Number(event.target.value))} />
      <button type="button" onClick={() => onContinue(branchId, shots)} disabled={busy}>
        {loading === branchId ? "追加中…" : "仅追加此分支"}
      </button>
    </div>
  );
}
