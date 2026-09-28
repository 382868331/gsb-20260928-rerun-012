import { useEffect, useState } from "react";
import { listCircuits, saveCircuit, validateCircuit } from "../api";
import { useBenchStore } from "../store";
import type { SavedCircuit, ValidationError } from "../types";
import { BELL_PAIR_SAMPLE, SINGLE_QUBIT_SAMPLE } from "../samples";

interface Props {
  errors: ValidationError[];
  onValidation: (errors: ValidationError[]) => void;
  onStart: () => void;
  onStep: () => void;
  onReset: () => void;
  running: boolean;
  canStep: boolean;
}

export function ControlBar({ errors, onValidation, onStart, onStep, onReset, running, canStep }: Props) {
  const { qubits, setQubits, initialBasis, setInitialBasis, seed, setSeed, steps, loadCircuit } = useBenchStore();
  const [name, setName] = useState("bell-pair");
  const [saved, setSaved] = useState<SavedCircuit | null>(null);
  const [saveError, setSaveError] = useState("");
  const [versions, setVersions] = useState<Record<string, SavedCircuit>>({});

  useEffect(() => {
    listCircuits().then((data) => setVersions(data.circuits)).catch(() => undefined);
  }, [saved]);

  useEffect(() => {
    validateCircuit({ qubits, initial_basis: initialBasis, seed, steps })
      .then((data) => onValidation(data.errors))
      .catch(() => onValidation([{ step: -1, message: "无法连接后端计算服务" }]));
  }, [qubits, initialBasis, seed, steps, onValidation]);

  async function persist() {
    setSaveError("");
    try {
      const record = await saveCircuit(name.trim() || "untitled", {
        qubits,
        initial_basis: initialBasis,
        seed,
        steps,
      });
      setSaved(record);
    } catch (err) {
      const payload = err as { errors?: Array<{ message: string }>; error?: string };
      setSaveError(payload.errors?.map((item) => item.message).join("; ") ?? payload.error ?? "保存失败");
    }
  }

  const globalErrors = errors.filter((error) => error.step === -1).map((error) => error.message);
  const valid = errors.length === 0;

  return (
    <section className="panel controls-panel">
      <h2>实验配置</h2>
      <div className="config-grid">
        <label className="field">
          量子位数（1-4）
          <input
            type="number"
            min={1}
            max={4}
            value={qubits}
            onChange={(event) => setQubits(Number(event.target.value))}
          />
        </label>
        <label className="field">
          初始计算基态
          <input
            value={initialBasis}
            maxLength={qubits}
            onChange={(event) => setInitialBasis(event.target.value.replace(/[^01]/g, ""))}
          />
        </label>
        <label className="field">
          随机种子（留空则不固定）
          <input
            type="number"
            value={seed ?? ""}
            onChange={(event) => setSeed(event.target.value === "" ? null : Number(event.target.value))}
          />
        </label>
      </div>

      <div className="inline-controls">
        <button type="button" className="primary" onClick={onStart} disabled={!valid || running}>
          开始并逐步执行
        </button>
        <button type="button" onClick={onStep} disabled={!canStep}>
          执行下一步
        </button>
        <button type="button" onClick={onReset}>
          重置会话
        </button>
        <button type="button" onClick={() => loadCircuit(SINGLE_QUBIT_SAMPLE)}>单量子位样例</button>
        <button type="button" onClick={() => loadCircuit(BELL_PAIR_SAMPLE)}>关联量子位样例</button>
      </div>

      <div className="inline-controls save-row">
        <input value={name} onChange={(event) => setName(event.target.value)} placeholder="线路名称" />
        <button type="button" onClick={persist} disabled={!valid}>保存版本</button>
        {saved && (
          <span className="muted">已保存：{saved.name} v{saved.version}</span>
        )}
        {Object.keys(versions).length > 0 && (
          <select
            value=""
            onChange={(event) => {
              const record = versions[event.target.value];
              if (record) loadCircuit(record);
            }}
          >
            <option value="">载入已保存版本…</option>
            {Object.values(versions).map((record) => (
              <option key={record.name} value={record.name}>
                {record.name} v{record.version}
              </option>
            ))}
          </select>
        )}
      </div>

      {saveError && <div className="error-box">{saveError}</div>}
      {!valid && (
        <div className="error-box">
          {globalErrors.length > 0 && <div>{globalErrors.join("；")}</div>}
          <div>存在非法连接或参数，修复标红步骤后才能开始运行。</div>
        </div>
      )}
    </section>
  );
}
