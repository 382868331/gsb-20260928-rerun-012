import { useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import {
  addBranchShots,
  createBranchExperiment,
  createBranchInquiry,
  getBranchExperiment,
  runAdaptiveBatch,
} from "../api";
import { newStepId } from "../store";
import type { BranchExperiment, CheckpointState, CircuitStep, GateType, MeasurementFilter } from "../types";

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

function pruneFilters(steps: CircuitStep[], filters: MeasurementFilter[]): MeasurementFilter[] {
  return filters.filter((filter) => steps[filter.step]?.type === "MEASURE");
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
    setBranches((prev) =>
      prev.map((branch) => {
        if (branch.id !== id) return branch;
        const next = { ...branch, ...patch };
        return { ...next, filters: pruneFilters(next.steps, next.filters) };
      }),
    );
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
            filters: branch.filters,
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
          filters: branch.filters ?? [],
        })),
      );
    } catch (err) {
      const payload = err as { error?: string };
      setError(payload.error ?? "恢复分支实验失败");
    } finally {
      setLoadingId("");
    }
  }

  async function addInquiry(inquiry: {
    mode: "single" | "compare";
    branch_ids: string[];
    event: string;
    margin: number;
    total_budget: number;
    batch_size: number;
  }) {
    if (!result || busy) return;
    setLoadingId("inquiry-create");
    setError("");
    try {
      await createBranchInquiry(result.experiment_id, inquiry);
      setResult(await getBranchExperiment(result.experiment_id));
    } catch (err) {
      const payload = err as { error?: string };
      setError(payload.error ?? "创建研究决定失败");
    } finally {
      setLoadingId("");
    }
  }

  async function continueAdaptive(inquiryId: string) {
    if (!result || busy) return;
    setLoadingId(inquiryId);
    setError("");
    try {
      setResult(await runAdaptiveBatch(result.experiment_id, inquiryId));
    } catch (err) {
      const payload = err as { error?: string };
      setError(payload.error ?? "自适应追加失败");
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
          {result && (
            <ExperimentResult
              result={result}
              branches={branches}
              busy={busy}
              loadingId={loadingId}
              onContinue={continueBranch}
              onAddInquiry={addInquiry}
              onAdaptiveBatch={continueAdaptive}
            />
          )}
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
      {branch.steps.some((step) => step.type === "MEASURE") && (
        <div className="filter-editor">
          <span className="muted">测量筛选（只统计命中样本）：</span>
          {branch.steps.map((step, stepIndex) =>
            step.type === "MEASURE" ? (
              <label key={step.id} className="field">
                第 {stepIndex + 1} 步 q{step.target ?? 0} 测量
                <select
                  value={branch.filters.find((filter) => filter.step === stepIndex)?.outcome ?? ""}
                  onChange={(event) => {
                    const value = event.target.value;
                    const rest = branch.filters.filter((filter) => filter.step !== stepIndex);
                    onUpdate({
                      filters: value === "" ? rest : [...rest, { step: stepIndex, outcome: Number(value) as 0 | 1 }],
                    });
                  }}
                >
                  <option value="">不筛选</option>
                  <option value="0">只保留 =0</option>
                  <option value="1">只保留 =1</option>
                </select>
              </label>
            ) : null,
          )}
        </div>
      )}
    </div>
  );
}

function ExperimentResult({
  result,
  branches,
  busy,
  loadingId,
  onContinue,
  onAddInquiry,
  onAdaptiveBatch,
}: {
  result: BranchExperiment;
  branches: BranchDraft[];
  busy: boolean;
  loadingId: string;
  onContinue: (branchId: string, shots: number) => void;
  onAddInquiry: (inquiry: {
    mode: "single" | "compare";
    branch_ids: string[];
    event: string;
    margin: number;
    total_budget: number;
    batch_size: number;
  }) => void;
  onAdaptiveBatch: (inquiryId: string) => void;
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
              样本范围：{branch.sample_scope}。样本来源：
              {branch.batches.map((batch) => batch.source + "(" + batch.shots + ")").join(" + ")}。
            </p>
            <p className="muted">
              总尝试 {branch.attempts} 次
              {branch.filters.length > 0 &&
                "，通过筛选 " + branch.passed_shots + " 次（" +
                (branch.pass_rate === null ? "—" : (branch.pass_rate * 100).toFixed(1) + "%") + "）"}
              ；筛选条件修改后需另建方案，旧样本不计入新条件。
            </p>
            {branch.filters.length > 0 && !branch.has_evidence ? (
              <div className="notice-box">
                没有样本通过筛选：当前没有证据，条件分布未知（这不是全零概率结论）。
              </div>
            ) : (
              <div className="prob-compact">
                {Object.entries(branch.frequencies ?? {}).map(([label, value]) => (
                  <span key={label} className={value > 0 ? "prob-hot" : ""}>
                    |{label}⟩ {branch.counts[label]}（{((value ?? 0) * 100).toFixed(1)}%）
                  </span>
                ))}
              </div>
            )}
            <ContinueShots branchId={branch.branch_id} defaultValue={draft?.nextShots ?? 256} busy={busy} loading={loadingId} onContinue={onContinue} />
          </div>
        );
      })}
      <InquiryPanel result={result} busy={busy} loadingId={loadingId} onAddInquiry={onAddInquiry} onAdaptiveBatch={onAdaptiveBatch} />
    </div>
  );
}

function InquiryPanel({
  result,
  busy,
  loadingId,
  onAddInquiry,
  onAdaptiveBatch,
}: {
  result: BranchExperiment;
  busy: boolean;
  loadingId: string;
  onAddInquiry: Parameters<typeof ExperimentResult>[0]["onAddInquiry"];
  onAdaptiveBatch: (inquiryId: string) => void;
}) {
  const [mode, setMode] = useState<"single" | "compare">("single");
  const [branchA, setBranchA] = useState(result.branches[0]?.branch_id ?? "");
  const [branchB, setBranchB] = useState(result.branches[1]?.branch_id ?? "");
  const [event, setEvent] = useState("__pass__");
  const [margin, setMargin] = useState(0.05);
  const [totalBudget, setTotalBudget] = useState(1000);
  const [batchSize, setBatchSize] = useState(25);

  return (
    <div className="inquiry-panel">
      <h3>序贯研究决定</h3>
      <p className="muted">
        使用每次查看后仍成立的置信序列；系统按当前证据宽度选择下一批分支。预算耗尽仍不能区分时保持“未确定”。
      </p>
      <div className="inline-controls">
        <select value={mode} onChange={(item) => setMode(item.target.value as "single" | "compare")}>
          <option value="single">观察一个分支</option>
          <option value="compare">比较两个分支</option>
        </select>
        <select value={branchA} onChange={(item) => setBranchA(item.target.value)}>
          {result.branches.map((branch) => <option key={branch.branch_id} value={branch.branch_id}>{branch.label}</option>)}
        </select>
        {mode === "compare" && (
          <select value={branchB} onChange={(item) => setBranchB(item.target.value)}>
            {result.branches.map((branch) => <option key={branch.branch_id} value={branch.branch_id}>{branch.label}</option>)}
          </select>
        )}
        <select value={event} onChange={(item) => setEvent(item.target.value)}>
          <option value="__pass__">筛选接受率</option>
          {Object.keys(result.branches[0]?.counts ?? {}).map((outcome) => (
            <option key={outcome} value={outcome}>事件 |{outcome}⟩</option>
          ))}
        </select>
      </div>
      <div className="inline-controls">
        <label>允许误差 <input type="number" min={0.0001} max={0.9999} step={0.001} value={margin} onChange={(item) => setMargin(Number(item.target.value))} /></label>
        <label>总尝试上限 <input type="number" min={1} value={totalBudget} onChange={(item) => setTotalBudget(Number(item.target.value))} /></label>
        <label>批量 <input type="number" min={1} value={batchSize} onChange={(item) => setBatchSize(Number(item.target.value))} /></label>
        <button
          type="button"
          disabled={busy || (mode === "compare" && (!branchA || !branchB || branchA === branchB))}
          onClick={() => onAddInquiry({
            mode,
            branch_ids: mode === "single" ? [branchA] : [branchA, branchB],
            event,
            margin,
            total_budget: totalBudget,
            batch_size: batchSize,
          })}
        >
          {loadingId === "inquiry-create" ? "创建中…" : "创建研究决定"}
        </button>
      </div>
      {result.inquiries.map((inquiry) => (
        <div key={inquiry.inquiry_id} className="branch-result">
          <strong>{inquiry.mode === "single" ? "单分支观察" : "双分支比较"} · {inquiry.status}</strong>
          <p className="muted">
            事件 {inquiry.event}；允许误差 ±{inquiry.margin}；已用 {inquiry.current_evaluation.used_budget}/
            {inquiry.total_budget}；批量 {inquiry.batch_size}。
          </p>
          {inquiry.current_evaluation.conclusion && <div className="notice-box">{inquiry.current_evaluation.conclusion.message}</div>}
          {inquiry.current_evaluation.branch_statistics.map((stat) => (
            <p key={stat.branch_id} className="muted">
              {stat.label}：{stat.event_successes}/{stat.event_trials}，区间 [{stat.event_interval[0].toFixed(3)}, {stat.event_interval[1].toFixed(3)}]，
              证据宽度 {stat.event_width.toFixed(3)}，分配分数 {stat.allocation_score.toFixed(3)}。{stat.evidence_note}
            </p>
          ))}
          <button type="button" disabled={busy || inquiry.status !== "active"} onClick={() => onAdaptiveBatch(inquiry.inquiry_id)}>
            {loadingId === inquiry.inquiry_id ? "追加中…" : "按最缺证据继续一批"}
          </button>
          <details>
            <summary>停止规则与预算分配审计（{inquiry.decisions.length}）</summary>
            <ol className="decision-log">
              {inquiry.decisions.map((decision) => (
                <li key={decision.decision_id}>{decision.action}：{decision.reason}（{decision.shots} 次）</li>
              ))}
            </ol>
          </details>
        </div>
      ))}
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
