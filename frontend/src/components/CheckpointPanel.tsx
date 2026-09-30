import { useState } from "react";
import { compareBranches, resumeCheckpoint, saveCheckpoint } from "../api";
import { useBenchStore } from "../store";
import type {
  BranchComparison,
  Checkpoint,
  CircuitStep,
  SessionState,
} from "../types";

interface Props {
  session: SessionState | null;
  onResume: (session: SessionState) => void;
  onError: (message: string) => void;
  busy: boolean;
  setBusy: (busy: boolean) => void;
}

export function CheckpointPanel({ session, onResume, onError, busy, setBusy }: Props) {
  const { qubits } = useBenchStore();
  const [checkpoint, setCheckpoint] = useState<Checkpoint | null>(null);
  const [savedId, setSavedId] = useState<string>("");
  const [resumeId, setResumeId] = useState<string>("");
  const [branchText, setBranchText] = useState<[string, string]>([
    JSON.stringify([{ type: "X", target: qubits - 1 }], null, 2),
    JSON.stringify([{ type: "Z", target: qubits - 1 }], null, 2),
  ]);
  const [comparison, setComparison] = useState<BranchComparison | null>(null);
  const [localError, setLocalError] = useState("");

  if (!session) {
    return null;
  }

  async function save(persist: boolean) {
    if (!session) return;
    setBusy(true);
    setLocalError("");
    try {
      const result = await saveCheckpoint(session.session_id, `step-${session.next_index}`, persist);
      setCheckpoint(result);
      setSavedId(result.checkpoint_id ?? "");
      setResumeId(result.checkpoint_id ?? "");
    } catch (err) {
      const payload = err as { error?: string };
      setLocalError(payload.error ?? "保存检查点失败");
    } finally {
      setBusy(false);
    }
  }

  async function resume() {
    if (!resumeId.trim()) {
      setLocalError("请输入已持久化检查点 ID");
      return;
    }
    setBusy(true);
    setLocalError("");
    try {
      onResume(await resumeCheckpoint(resumeId.trim()));
    } catch (err) {
      const payload = err as { error?: string };
      setLocalError(payload.error ?? "恢复检查点失败");
      onError(payload.error ?? "恢复检查点失败");
    } finally {
      setBusy(false);
    }
  }

  async function compare() {
    if (!checkpoint) {
      setLocalError("请先在当前线路中途保存检查点");
      return;
    }
    let branches: [CircuitStep[], CircuitStep[]];
    try {
      const parsed = branchText.map((text) => JSON.parse(text)) as [CircuitStep[], CircuitStep[]];
      branches = parsed;
    } catch {
      setLocalError("两个分支都必须是合法的步骤 JSON 数组");
      return;
    }
    setBusy(true);
    setLocalError("");
    try {
      setComparison(
        await compareBranches(checkpoint, branches, ["分支 A", "分支 B"]),
      );
    } catch (err) {
      const payload = err as { error?: string; errors?: Array<{ message: string }> };
      setLocalError(payload.errors?.map((item) => item.message).join("; ") ?? payload.error ?? "分支比较失败");
    } finally {
      setBusy(false);
    }
  }

  const qubitCount = checkpoint?.qubits ?? 1;
  const labels = Array.from({ length: 2 ** qubitCount }, (_, i) =>
    i.toString(2).padStart(qubitCount, "0"),
  );

  return (
    <section className="panel">
      <h2>中途检查点与分支比较</h2>
      <p className="muted">
        在任意步数保存当前坍缩状态与随机数发生器状态；恢复后已发生的测量不会重新抽取。
      </p>
      <div className="inline-controls">
        <button type="button" onClick={() => save(false)} disabled={busy}>
         保存当前状态（本次会话）
        </button>
        <button type="button" onClick={() => save(true)} disabled={busy}>
         保存并持久化
        </button>
      </div>
      {checkpoint && (
        <p className="muted">
          检查点：已执行 {checkpoint.next_index} 步，测量记录 {checkpoint.measurements.length} 条
          {savedId ? `，ID ${savedId}` : "（仅内存，重启后失效）"}
        </p>
      )}
      <div className="inline-controls">
        <input
          value={resumeId}
          placeholder="粘贴持久化检查点 ID 以从保存处继续"
          onChange={(event) => setResumeId(event.target.value)}
        />
        <button type="button" onClick={resume} disabled={busy}>
         从检查点继续
        </button>
      </div>

      <div className="branch-grid">
        {[0, 1].map((branchIndex) => (
          <label key={branchIndex} className="field">
            后续分支 {branchIndex === 0 ? "A" : "B"}（步骤 JSON）
            <textarea
              rows={6}
              value={branchText[branchIndex]}
              onChange={(event) =>
                setBranchText(
                  branchIndex === 0
                    ? [event.target.value, branchText[1]]
                    : [branchText[0], event.target.value],
                )
              }
            />
          </label>
        ))}
      </div>
      <button type="button" className="primary" onClick={compare} disabled={busy}>
        比较两个后续分支
      </button>

      {localError && <div className="error-box">{localError}</div>}

      {comparison && (
        <div className="compare-result">
          {comparison.branches.map((branch, i) => (
            <div key={i}>
              <strong>{comparison.labels[i]}</strong>
              <ul>
                {labels.map((label) => (
                  <li key={label}>
                    |{label}⟩：{(branch.final_probabilities[label] * 100).toFixed(2)}%
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
