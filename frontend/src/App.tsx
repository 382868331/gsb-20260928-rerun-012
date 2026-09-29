import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { bellSample, emptyCircuit, gateOptions, singleQubitSample } from "./presets";
import type { CircuitDraft, CircuitStep, GateType, OperationInfo, Project, RunState, ShotSummary } from "./types";

const gateRequiresControl = (type: GateType) => type === "CNOT" || type === "SWAP";
const gateRequiresAngle = (type: GateType) => type === "RX" || type === "RY" || type === "RZ";

function addStep(circuit: CircuitDraft, type: GateType = "H"): CircuitStep {
  const target = type === "CNOT" && circuit.qubits > 1 ? 1 : 0;
  return {
    id: crypto.randomUUID(),
    type,
    target,
    control: gateRequiresControl(type) ? 0 : null,
    angle: gateRequiresAngle(type) ? Math.PI / 2 : null
  };
}

function probabilityRows(values: Record<string, number>) {
  return Object.entries(values).sort(([left], [right]) => left.localeCompare(right));
}

export default function App() {
  const [name, setName] = useState("未命名量子实验");
  const [project, setProject] = useState<Project | null>(null);
  const [circuit, setCircuit] = useState<CircuitDraft>(() => bellSample());
  const [operations, setOperations] = useState<OperationInfo[]>([]);
  const [errors, setErrors] = useState<string[]>([]);
  const [message, setMessage] = useState("");
  const [run, setRun] = useState<RunState | null>(null);
  const [shots, setShots] = useState(1000);
  const [summary, setSummary] = useState<ShotSummary | null>(null);

  const valid = errors.length === 0;
  const operationMap = useMemo(() => new Map(operations.map((item) => [item.type, item])), [operations]);

  useEffect(() => {
    api.operations().then(setOperations).catch((error: Error) => setMessage(error.message));
  }, []);

  useEffect(() => {
    api.validate(circuit)
      .then((result) => setErrors(result.errors))
      .catch((error: Error) => setErrors([error.message]));
    setRun(null);
    setSummary(null);
  }, [circuit]);

  function patchCircuit(patch: Partial<CircuitDraft>) {
    setCircuit((current) => ({ ...current, ...patch }));
  }

  function changeQubits(value: number) {
    const qubits = Math.max(1, Math.min(8, value));
    setCircuit((current) => ({
      ...current,
      qubits,
      initial_state: { ...current.initial_state, kind: "basis", basis: "0".repeat(qubits), amplitudes: null },
      steps: current.steps.map((step) => ({
        ...step,
        target: Math.min(step.target, qubits - 1),
        control: step.control === null || step.control >= qubits ? null : step.control
      }))
    }));
  }

  function updateStep(id: string, patch: Partial<CircuitStep>) {
    setCircuit((current) => ({
      ...current,
      steps: current.steps.map((step) => (step.id === id ? { ...step, ...patch } : step))
    }));
  }

  async function saveProject() {
    setMessage("");
    try {
      const saved = project
        ? await api.saveProject(project.id, name, circuit)
        : await api.createProject(name, circuit);
      setProject(saved);
      setMessage("已保存线路版本 v" + saved.current_version);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  async function startRun() {
    setMessage("");
    try {
      setRun(await api.startRun(circuit));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  async function nextStep() {
    if (!run) return;
    setMessage("");
    try {
      setRun(await api.nextStep(run.run_id));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  async function runExperiment() {
    setMessage("");
    try {
      setSummary(await api.shots(circuit, shots));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  return (
    <main>
      <header className="hero">
        <div>
          <p className="eyebrow">LOCAL STATEVECTOR SIMULATOR · NO QUANTUM CLOUD</p>
          <h1>本地量子计算实验台</h1>
          <p>线路在 React 中编辑，校验、演化、测量坍缩和重复实验全部由 Python 后端计算。</p>
        </div>
        <div className="status valid">{valid ? "可以开始运行" : "存在非法连接"}</div>
      </header>

      {message && <div className="banner">{message}</div>}

      <section className="grid">
        <div className="card">
          <div className="card-title">
            <h2>实验与线路</h2>
            <div className="actions">
              <button onClick={() => setCircuit(singleQubitSample())}>单比特样例</button>
              <button onClick={() => setCircuit(bellSample())}>关联量子位样例</button>
              <button onClick={() => setCircuit(emptyCircuit(circuit.qubits))}>清空</button>
            </div>
          </div>

          <label className="field">
            项目名称
            <input value={name} onChange={(event) => setName(event.target.value)} />
          </label>

          <div className="form-row">
            <label className="field small">
              量子位数量
              <input type="number" min={1} max={8} value={circuit.qubits} onChange={(event) => changeQubits(Number(event.target.value))} />
            </label>
            <label className="field small">
              初始计算态
              <input
                value={circuit.initial_state.basis}
                onChange={(event) => setCircuit({
                  ...circuit,
                  initial_state: { kind: "basis", basis: event.target.value, amplitudes: null }
                })}
              />
            </label>
            <label className="field small">
              随机种子
              <input
                type="number"
                value={circuit.seed ?? ""}
                placeholder="随机"
                onChange={(event) => patchCircuit({ seed: event.target.value === "" ? null : Number(event.target.value) })}
              />
            </label>
          </div>

          {!valid && (
            <div className="errors">
              {errors.map((error) => <div key={error}>• {error}</div>)}
            </div>
          )}

          <div className="steps-head">
            <h3>线路步骤</h3>
            <button onClick={() => patchCircuit({ steps: [...circuit.steps, addStep(circuit)] })}>添加步骤</button>
          </div>

          <div className="steps">
            {circuit.steps.length === 0 && <p className="muted">还没有步骤。添加门或测量后才能运行。</p>}
            {circuit.steps.map((step, index) => (
              <div className="step" key={step.id}>
                <span className="step-index">#{index + 1}</span>
                <select value={step.type} onChange={(event) => {
                  const type = event.target.value as GateType;
                  updateStep(step.id, {
                    type,
                    control: gateRequiresControl(type) ? (step.control ?? (step.target === 0 ? 1 : 0)) : null,
                    angle: gateRequiresAngle(type) ? Math.PI / 2 : null
                  });
                }}>
                  {gateOptions.map((type) => <option key={type}>{type}</option>)}
                </select>
                <label>目标
                  <select value={step.target} onChange={(event) => updateStep(step.id, { target: Number(event.target.value) })}>
                    {Array.from({ length: circuit.qubits }, (_, qubit) => <option key={qubit} value={qubit}>q{qubit}</option>)}
                  </select>
                </label>
                {gateRequiresControl(step.type) && (
                  <label>连接
                    <select value={step.control ?? ""} onChange={(event) => updateStep(step.id, { control: Number(event.target.value) })}>
                      {Array.from({ length: circuit.qubits }, (_, qubit) => <option key={qubit} value={qubit}>q{qubit}</option>)}
                    </select>
                  </label>
                )}
                {gateRequiresAngle(step.type) && (
                  <label className="angle">角度
                    <input type="number" step="0.1" value={step.angle ?? ""} onChange={(event) => updateStep(step.id, { angle: Number(event.target.value) })} />
                  </label>
                )}
                <button className="ghost danger" onClick={() => patchCircuit({ steps: circuit.steps.filter((item) => item.id !== step.id) })}>删除</button>
              </div>
            ))}
          </div>

          <div className="footer-actions">
            <button className="primary" onClick={saveProject} disabled={!valid}>保存新版本</button>
            {project && <span className="muted">当前项目：v{project.current_version}</span>}
          </div>
        </div>

        <div className="card">
          <div className="card-title">
            <h2>逐步执行</h2>
            <span className="muted">{run ? run.index + " / " + circuit.steps.length : "未开始"}</span>
          </div>
          <p className="muted">每次测量在后端按当前概率采样，服务端保存坍缩后的状态，下一步继续使用它。</p>
          <div className="actions">
            <button className="primary" onClick={startRun} disabled={!valid || circuit.steps.length === 0}>开始运行</button>
            <button onClick={nextStep} disabled={!run || run.complete}>执行下一步</button>
          </div>

          {run && (
            <>
              <div className="probabilities">
                <h3>当前概率</h3>
                {probabilityRows(run.probabilities).map(([key, value]) => (
                  <div className="bar-row" key={key}>
                    <span>{key}</span>
                    <div className="bar"><i style={{ width: Math.max(2, value * 100) + "%" }} /></div>
                    <b>{(value * 100).toFixed(2)}%</b>
                  </div>
                ))}
              </div>
              <div className="results">
                {run.results.map((result) => (
                  <div className="result" key={result.step.id}>
                    <strong>#{result.index + 1} {result.step.type}</strong>
                    {result.measurement_result !== null && <span className="measurement">测得 {result.measurement_result}</span>}
                    <small>{result.note ?? operationMap.get(result.step.type)?.description}</small>
                  </div>
                ))}
              </div>
              {run.complete && <div className="banner success">线路执行完成；测量结果已在每一步真实影响后续状态。</div>}
            </>
          )}
        </div>
      </section>

      <section className="card experiment">
        <div className="card-title">
          <div>
            <h2>重复实验统计</h2>
            <p className="muted">核对单比特 H 门约 50/50；Bell 线路只应出现 q0q1 = 00 与 11。</p>
          </div>
          <div className="actions">
            <input type="number" min={1} max={100000} value={shots} onChange={(event) => setShots(Number(event.target.value))} />
            <button className="primary" onClick={runExperiment} disabled={!valid}>运行 {shots} 次</button>
          </div>
        </div>
        {summary && (
          <div className="shot-grid">
            {Object.entries(summary.counts).sort().map(([key, count]) => (
              <div className="shot" key={key}>
                <span>{key}</span>
                <strong>{count}</strong>
                <small>{((summary.probabilities[key] ?? 0) * 100).toFixed(2)}%</small>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="operations">
        <h2>本轮支持的基本操作</h2>
        <div className="operation-grid">
          {operations.map((operation) => (
            <article key={operation.type}>
              <span>{operation.category}</span>
              <h3>{operation.type} · {operation.name}</h3>
              <p>{operation.description}</p>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
