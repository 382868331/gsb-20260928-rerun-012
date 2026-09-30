import { useBenchStore } from "../store";
import type { GateType } from "../types";

const GROUPS: Array<{ label: string; gates: GateType[] }> = [
  { label: "单量子位门", gates: ["H", "X", "Y", "Z", "S", "T"] },
  { label: "旋转门", gates: ["RX", "RY", "RZ"] },
  { label: "双量子位门", gates: ["CNOT", "CZ", "SWAP"] },
  { label: "测量与条件", gates: ["MEASURE", "IF"] },
];

export function GatePalette() {
  const addStep = useBenchStore((state) => state.addStep);
  return (
    <section className="panel">
      <h2>添加线路步骤</h2>
      {GROUPS.map((group) => (
        <div key={group.label} className="gate-group">
          <span className="group-label">{group.label}</span>
          <div className="gate-row">
            {group.gates.map((gate) => (
              <button key={gate} type="button" className="gate-button" onClick={() => addStep(gate)}>
                {gate === "MEASURE" ? "测量" : gate === "IF" ? "条件 IF" : gate}
              </button>
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}
