import { useCallback, useEffect, useState } from "react";
import { advanceSession, createSession } from "./api";
import { ControlBar } from "./components/ControlBar";
import { GatePalette } from "./components/GatePalette";
import { ProbabilityPanel } from "./components/ProbabilityPanel";
import { StatsPanel } from "./components/StatsPanel";
import { StepEditor } from "./components/StepEditor";
import { useBenchStore } from "./store";
import type { SessionState, ValidationError } from "./types";

export default function App() {
  const { qubits, initialBasis, seed, steps } = useBenchStore();
  const [errors, setErrors] = useState<ValidationError[]>([]);
  const [session, setSession] = useState<SessionState | null>(null);
  const [runtimeError, setRuntimeError] = useState("");

  const handleValidation = useCallback((next: ValidationError[]) => {
    setErrors(next);
  }, []);

  const circuit = { qubits, initial_basis: initialBasis, seed, steps };

  useEffect(() => {
    setSession(null);
    setRuntimeError("");
  }, [qubits, initialBasis, seed, steps]);

  async function start() {
    setRuntimeError("");
    try {
      const created = await createSession(circuit);
      setSession(await advanceSession(created.session_id));
    } catch (err) {
      const payload = err as { errors?: Array<{ message: string }>; error?: string };
      setRuntimeError(payload.errors?.map((item) => item.message).join("; ") ?? payload.error ?? "启动失败");
    }
  }

  async function nextStep() {
    if (!session) {
      await start();
      return;
    }
    if (session.finished) return;
    setRuntimeError("");
    try {
      setSession(await advanceSession(session.session_id));
    } catch (err) {
      const payload = err as { error?: string };
      setRuntimeError(payload.error ?? "执行失败");
    }
  }

  function reset() {
    setSession(null);
    setRuntimeError("");
  }

  return (
    <main className="app-shell">
      <header>
        <h1>本地量子计算实验台</h1>
        <p className="muted">所有线路均由本地 Python 后端计算；测量会在后端坍缩状态，并直接影响后续门操作，不依赖量子云服务。</p>
      </header>

      <ControlBar
        errors={errors}
        onValidation={handleValidation}
        onStart={start}
        onStep={nextStep}
        onReset={reset}
        running={false}
        canStep={errors.length === 0 && (!session || !session.finished)}
      />

      {runtimeError && <div className="error-box">{runtimeError}</div>}

      <div className="workspace-grid">
        <div className="left-column">
          <GatePalette />
          <section className="panel">
            <h2>线路步骤（{steps.length}）</h2>
            <p className="muted">支持：H/X/Y/Z/S/T、RX/RY/RZ(θ)、CNOT/CZ/SWAP、MEASURE；双位门两端必须不同且在量子位范围内。</p>
            {steps.length === 0 && <div className="empty">从上方添加第一个步骤。</div>}
            {steps.map((step, index) => (
              <StepEditor
                key={step.id}
                step={step}
                index={index}
                qubits={qubits}
                errors={errors}
                executedThrough={session?.next_index ?? 0}
              />
            ))}
          </section>
        </div>
        <div className="right-column">
          <ProbabilityPanel session={session} initialBasis={initialBasis} qubits={qubits} />
          <StatsPanel disabled={errors.length > 0} />
        </div>
      </div>
    </main>
  );
}
