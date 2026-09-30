import { useCallback, useEffect, useState } from "react";
import { advanceSession, createSession, resumeCheckpoint, saveCheckpoint } from "./api";
import { ControlBar } from "./components/ControlBar";
import { GatePalette } from "./components/GatePalette";
import { ProbabilityPanel } from "./components/ProbabilityPanel";
import { StatsPanel } from "./components/StatsPanel";
import { StepEditor } from "./components/StepEditor";
import { useBenchStore } from "./store";
import { BranchCompare } from "./components/BranchCompare";
import type { CheckpointState, SessionState, ValidationError } from "./types";

export default function App() {
  const { qubits, initialBasis, seed, steps } = useBenchStore();
  const [errors, setErrors] = useState<ValidationError[]>([]);
  const [session, setSession] = useState<SessionState | null>(null);
  const [runtimeError, setRuntimeError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [checkpoints, setCheckpoints] = useState<CheckpointState[]>([]);

  const handleValidation = useCallback((next: ValidationError[]) => {
    setErrors(next);
  }, []);

  const circuit = { qubits, initial_basis: initialBasis, seed, steps };

  useEffect(() => {
    setSession(null);
    setRuntimeError("");
    setNotice("");
  }, [qubits, initialBasis, seed, steps]);

  async function start() {
    if (busy) return;
    setBusy(true);
    setRuntimeError("");
    setNotice("");
    try {
      const created = await createSession(circuit);
      const first = await advanceSession(created.session_id);
      setSession(first);
      if (first.finished) setNotice(first.notice ?? "线路只有 0 或 1 步，已执行完毕");
    } catch (err) {
      const payload = err as { errors?: Array<{ message: string }>; error?: string };
      setRuntimeError(payload.errors?.map((item) => item.message).join("; ") ?? payload.error ?? "启动失败");
    } finally {
      setBusy(false);
    }
  }

  async function nextStep() {
    if (busy) return;
    if (!session) {
      await start();
      return;
    }
    if (session.finished) {
      setNotice("线路已执行完毕，没有更多步骤；可重置后重新运行");
      return;
    }
    setBusy(true);
    setRuntimeError("");
    setNotice("");
    try {
      const next = await advanceSession(session.session_id);
      setSession(next);
      if (next.notice) setNotice(next.notice);
      else if (next.finished) setNotice("最后一步已执行完毕");
    } catch (err) {
      const payload = err as { error?: string };
      setRuntimeError(payload.error ?? "执行失败");
    } finally {
      setBusy(false);
    }
  }

  async function persistCheckpoint() {
    if (!session || busy) return;
    setBusy(true);
    setRuntimeError("");
    setNotice("");
    try {
      const checkpoint = await saveCheckpoint(session.session_id, `第 ${session.next_index} 步`);
      setCheckpoints((prev) => [checkpoint, ...prev.filter((item) => item.checkpoint_id !== checkpoint.checkpoint_id)]);
      setNotice(`已在「${checkpoint.name}」保存状态`);
    } catch (err) {
      const payload = err as { error?: string };
      setRuntimeError(payload.error ?? "保存状态失败");
    } finally {
      setBusy(false);
    }
  }

  async function resumeFrom(checkpoint: CheckpointState) {
    if (busy) return;
    setBusy(true);
    setRuntimeError("");
    setNotice("");
    try {
      const resumed = await resumeCheckpoint(checkpoint.checkpoint_id);
      setSession(resumed);
      setNotice(resumed.notice ?? "已从保存点继续");
    } catch (err) {
      const payload = err as { error?: string };
      setRuntimeError(payload.error ?? "恢复失败");
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    setSession(null);
    setRuntimeError("");
    setNotice("");
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
        canStep={errors.length === 0 && (!session || !session.finished) && !busy}
      />

      {runtimeError && <div className="error-box">{runtimeError}</div>}
      {notice && <div className="notice-box">{notice}</div>}

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
          <ProbabilityPanel
            session={session}
            initialBasis={initialBasis}
            qubits={qubits}
            checkpoints={checkpoints}
            busy={busy}
            onSaveCheckpoint={persistCheckpoint}
            onResume={resumeFrom}
          />
          <BranchCompare checkpoints={checkpoints} disabled={busy} />
          <StatsPanel disabled={errors.length > 0} />
        </div>
      </div>
    </main>
  );
}
