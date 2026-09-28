import type { SessionState } from "../types";

interface Props {
  session: SessionState | null;
  initialBasis: string;
  qubits: number;
}

export function ProbabilityPanel({ session, initialBasis, qubits }: Props) {
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
