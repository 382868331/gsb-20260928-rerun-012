import { describe, expect, it } from "vitest";
import { makeStep } from "./store";
import { quantum } from "./gateSpec";

describe("supported gate range", () => {
  it("creates distinct-qubit defaults for two-qubit gates", () => {
    const cnot = makeStep("CNOT", 2);
    expect(cnot.control).toBe(0);
    expect(cnot.target).toBe(1);
    expect(quantum.twoQubitGates).toContain("CNOT");
    expect(quantum.maxQubits).toBe(4);
  });

  it("declares the complete first-round operation set", () => {
    expect(quantum.singleQubitGates).toEqual(["H", "X", "Y", "Z", "S", "T"]);
    expect(quantum.paramGates).toEqual(["RX", "RY", "RZ"]);
    expect(quantum.supportsMeasurement).toBe(true);
  });
});
