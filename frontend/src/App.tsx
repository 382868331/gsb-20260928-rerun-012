import { useCallback, useEffect, useRef, useState } from "react";
import { advanceSession, createSession } from "./api";
import { CheckpointPanel } from "./components/CheckpointPanel";
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
  const [statusMessage, setStatusMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);

  const handleValidation = useCallback((next: ValidationError[]) => {
    setErrors(next);
  }, []);

  const circuit = { qubits, initial_basis: initialBasis, seed, steps };

  useEffect(() => {
    setSession(null);
    setRuntimeError("");
    setStatusMessage("");
  }, [qubits, initialBasis, seed, steps]);

  async function start() {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setRuntimeError("");
    setStatusMessage("");
    try {
      const created = await createSession(circuit);
      setSession(created);
      setStatusMessage(`已开始并执行第 1 步（共 ${created.total_steps} 步）`);
    } catch (err) {
      const payload = err as { errors?: Array<{ message: string }>; error?: string };
      setRuntimeError(payload.errors?.map((item) => item.message).join("; ") ?? payload.error ?? "启动失败");
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  async function nextStep() {
    if (!session) {
      await start();
      return;
    }
    if (session.finished) {
      setStatusMessage("线路已经运行结束，没有下一步可执行；可重置后重新开始。");
      return;
    }
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setRuntimeError("");
    setStatusMessage("");
    try {
      const advanced = await advanceSession(session.session_id);
      setSession(advanced);
      setStatusMessage(
        advanced.finished
          ? `第 ${advanced.total_steps} 步已执行，线路运行结束`
          : `已执行第 ${advanced.next_index} 步`,
      );
    } catch (err) {
      const payload = err as { status?: number; error?: string; finished?: boolean; next_index?: number };
      if (payload.status === 409 && payload.finished) {
        setSession({ ...session, finished: true, next_index: payload.next_index ?? session.next_index });
        setStatusMessage("线路已经运行结束，没有下一步可执行；可重置后重新开始。");
      } else {
        setRuntimeError(payload.error ?? "执行失败");
      }
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  function reset() {
    setSession(null);
    setRuntimeError("");
    setStatusMessage("");
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
        running={busy}
        canStep={errors.length === 0 && !busy}
      />

      {runtimeError && <div className="error-box">{runtimeError}</div>}
      {statusMessage && <div className="status-box">{statusMessage}</div>}

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
          <CheckpointPanel
            session={session}
            onResume={(resumed) => {
              setSession(resumed);
              setStatusMessage(
                `已从检查点恢复：保留 ${resumed.restored_measurements?.length ?? 0} 条历史测量，从第 ${resumed.next_index + 1} 步继续`,
              );
            }}
            onError={setRuntimeError}
            busy={busy}
            setBusy={(value) => {
              busyRef.current = value;
              setBusy(value);
            }}
          />
          <StatsPanel disabled={errors.length > 0} />
        </div>
      </div>
    </main>
  );
}
