import type { CheckpointState, SessionState } from "../types";

interface Props {
  session: SessionState | null;
  initialBasis: string;
  qubits: number;
  checkpoints: CheckpointState[];
  busy: boolean;
  onSaveCheckpoint: () => void;
  onResume: (checkpoint: CheckpointState) => void;
}

export function ProbabilityPanel({ session, initialBasis, qubits, checkpoints, busy, onSaveCheckpoint, onResume }: Props) {
  const probabilities = session?.probabilities ?? null;
  const labels = Array.from({ length: 2 ** qubits }, (_, i) => i.toString(2).padStart(qubits, "0"));
  const initialIndex = labels.indexOf(initialBasis);
  const display = labels.map((label, i) => ({
    label,
    value: probabilities ? probabilities[label] : i === initialIndex ? 1 : 0,
  }));

  return (
    <section className="panel">
      <h2>后端实时状态（测量后坍缩）</h2>
      {session && (
        <p className="muted">
          已执行 {session.next_index}/{session.total_steps} 步，种子 {session.seed ?? "随机"}
          {session.finished ? "，线路已结束" : ""}
        </p>
      )}
      {session && (
        <div className="inline-controls">
          <button type="button" onClick={onSaveCheckpoint} disabled={busy || session.next_index === 0}>
            保存当前状态
          </button>
          {checkpoints.length > 0 && (
            <select
              value=""
              onChange={(event) => {
                const checkpoint = checkpoints.find((item) => item.checkpoint_id === event.target.value);
                if (checkpoint) onResume(checkpoint);
              }}
            >
              <option value="">从保存点继续…</option>
              {checkpoints.map((checkpoint) => (
                <option key={checkpoint.checkpoint_id} value={checkpoint.checkpoint_id}>
                  {checkpoint.name}（第 {checkpoint.next_index}/{checkpoint.total_steps} 步）
                </option>
              ))}
            </select>
          )}
        </div>
      )}
      <div className="prob-list">
        {display.map(({ label, value }) => (
          <div key={label} className="prob-row">
            <span className="prob-label">|{label}⟩</span>
            <div className="prob-track">
              <div className="prob-fill" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
            </div>
            <span className="prob-value">{value.toFixed(4)}</span>
          </div>
        ))}
      </div>
      {session && session.measurements.length > 0 && (
        <div className="measurements">
          <strong>测量结果（后续步骤使用坍缩后的状态）</strong>
          <ul>
            {session.measurements.map((item) => (
              <li key={`${item.step}-${item.target}`}>
                步骤 #{item.step + 1}：q{item.target} → {item.outcome}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
