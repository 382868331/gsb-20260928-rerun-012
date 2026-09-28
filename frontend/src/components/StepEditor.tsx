import { ArrowDown, ArrowUp, Trash2 } from "lucide-react";
import { useBenchStore } from "../store";
import type { CircuitStep, ValidationError } from "../types";

interface Props {
  step: CircuitStep;
  index: number;
  qubits: number;
  errors: ValidationError[];
  executedThrough: number;
}

const GATE_LABEL: Record<string, string> = {
  MEASURE: "测量 M",
};

export function StepEditor({ step, index, qubits, errors, executedThrough }: Props) {
  const { updateStep, removeStep, moveStep } = useBenchStore();
  const qubitOptions = Array.from({ length: qubits }, (_, q) => q);
  const stepErrors = errors.filter((error) => error.step === index);
  const done = index < executedThrough;

  const QubitSelect = ({ label, valueKey }: { label: string; valueKey: "target" | "control" | "a" | "b" }) => (
    <label className="field">
      {label}
      <select value={step[valueKey] ?? 0} onChange={(event) => updateStep(step.id, { [valueKey]: Number(event.target.value) })}>
        {qubitOptions.map((q) => (
          <option key={q} value={q}>q{q}</option>
        ))}
      </select>
    </label>
  );

  return (
    <div className={`step ${stepErrors.length ? "step-invalid" : ""} ${done ? "step-done" : ""}`}>
      <div className="step-head">
        <span className="step-index">#{index + 1}</span>
        <strong>{GATE_LABEL[step.type] ?? step.type}</strong>
        <div className="step-actions">
          <button type="button" aria-label="上移" onClick={() => moveStep(step.id, -1)} disabled={index === 0}>
            <ArrowUp size={14} />
          </button>
          <button type="button" aria-label="下移" onClick={() => moveStep(step.id, 1)} disabled={done}>
            <ArrowDown size={14} />
          </button>
          <button type="button" aria-label="删除" onClick={() => removeStep(step.id)}>
            <Trash2 size={14} />
          </button>
        </div>
      </div>
      <div className="step-body">
        {(step.type === "CNOT" || step.type === "CZ") && (
          <>
            <QubitSelect label="控制位" valueKey="control" />
            <QubitSelect label="目标位" valueKey="target" />
          </>
        )}
        {step.type === "SWAP" && (
          <>
            <QubitSelect label="量子位 A" valueKey="a" />
            <QubitSelect label="量子位 B" valueKey="b" />
          </>
        )}
        {(step.type === "H" ||
          step.type === "X" ||
          step.type === "Y" ||
          step.type === "Z" ||
          step.type === "S" ||
          step.type === "T" ||
          step.type === "MEASURE") && <QubitSelect label="量子位" valueKey="target" />}
        {(step.type === "RX" || step.type === "RY" || step.type === "RZ") && (
          <>
            <QubitSelect label="量子位" valueKey="target" />
            <label className="field">
              角度 (弧度)
              <input
                type="number"
                step="0.01"
                value={step.angle ?? 0}
                onChange={(event) => updateStep(step.id, { angle: Number(event.target.value) })}
              />
            </label>
          </>
        )}
      </div>
      {stepErrors.map((error) => (
        <div key={error.message} className="step-error">{error.message}</div>
      ))}
    </div>
  );
}
