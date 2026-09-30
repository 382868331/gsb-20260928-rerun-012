export const quantum = {
  maxQubits: 4,
  singleQubitGates: ["H", "X", "Y", "Z", "S", "T"] as const,
  paramGates: ["RX", "RY", "RZ"] as const,
  twoQubitGates: ["CNOT", "CZ", "SWAP"] as const,
  conditionalGate: "IF" as const,
  supportsMeasurement: true,
};
