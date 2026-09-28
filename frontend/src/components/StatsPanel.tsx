import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { runExperiment } from "../api";
import { useBenchStore } from "../store";
import type { ExperimentResult } from "../types";

export function StatsPanel({ disabled }: { disabled: boolean }) {
  const { qubits, initialBasis, seed, steps } = useBenchStore();
  const [shots, setShots] = useState(1024);
  const [result, setResult] = useState<ExperimentResult | null>(null);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);

  async function runStats() {
    setLoading(true);
    setError("");
    try {
      const data = await runExperiment({ qubits, initial_basis: initialBasis, seed, steps }, shots);
      setResult(data);
    } catch (err) {
      const payload = err as { errors?: Array<{ message: string }>; error?: string };
      setError(payload.errors?.map((item) => item.message).join("; ") ?? payload.error ?? "实验失败");
    } finally {
      setLoading(false);
    }
  }

  const chartData = result
    ? Object.keys(result.counts).map((label) => ({
        basis: `|${label}⟩`,
        实测频率: Number((result.frequencies[label] * 100).toFixed(2)),
        理论概率: Number((result.expected_probabilities[label] * 100).toFixed(2)),
      }))
    : [];

  return (
    <section className="panel">
      <h2>重复实验统计（本地模拟）</h2>
      <div className="inline-controls">
        <label className="field">
          重复次数
          <input type="number" min={1} max={100000} value={shots} onChange={(event) => setShots(Number(event.target.value))} />
        </label>
        <button type="button" onClick={runStats} disabled={disabled || loading} className="primary">
          {loading ? "运行中…" : "运行重复实验"}
        </button>
      </div>
      {error && <div className="error-box">{error}</div>}
      {result && (
        <>
          <p className="muted">种子 {result.seed ?? "随机"}，共 {result.shots} 次；相同种子可完全复现。</p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chartData} margin={{ top: 10, right: 10, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="basis" />
              <YAxis unit="%" />
              <Tooltip />
              <Legend />
              <Bar dataKey="实测频率" fill="#4f8cff" />
              <Bar dataKey="理论概率" fill="#ff9f43" />
            </BarChart>
          </ResponsiveContainer>
        </>
      )}
    </section>
  );
}
