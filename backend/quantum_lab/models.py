from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


GateType = Literal["X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ", "CNOT", "SWAP", "MEASURE"]


class CircuitStep(BaseModel):
    id: str = Field(min_length=1)
    type: GateType
    target: int = Field(ge=0)
    control: int | None = Field(default=None, ge=0)
    angle: float | None = None
    label: str | None = None


class InitialState(BaseModel):
    kind: Literal["basis", "custom"] = "basis"
    basis: str = "0"
    amplitudes: list[list[float]] | None = None


class CircuitDraft(BaseModel):
    qubits: int = Field(default=2, ge=1, le=8)
    initial_state: InitialState = Field(default_factory=InitialState)
    seed: int | None = Field(default=None, ge=0)
    steps: list[CircuitStep] = Field(default_factory=list)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    qubits: int = Field(default=2, ge=1, le=8)
    initial_state: InitialState = Field(default_factory=InitialState)
    seed: int | None = Field(default=None, ge=0)
    steps: list[CircuitStep] = Field(default_factory=list)


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    circuit: CircuitDraft


class VersionInfo(BaseModel):
    version: int
    created_at: str
    note: str = ""
    circuit: CircuitDraft


class Project(BaseModel):
    id: str
    name: str
    created_at: str
    updated_at: str
    current_version: int
    versions: list[VersionInfo]


class RunCreate(BaseModel):
    circuit: CircuitDraft | None = None


class StepResult(BaseModel):
    index: int
    step: CircuitStep
    state_vector: list[list[float]]
    probabilities: dict[str, float]
    measured: dict[str, int] | None = None
    measurement_result: str | None = None
    collapsed_qubits: list[int] = []
    note: str | None = None


class RunState(BaseModel):
    run_id: str
    project_id: str | None = None
    version: int | None = None
    qubits: int
    seed: int | None
    index: int
    complete: bool
    state_vector: list[list[float]]
    probabilities: dict[str, float]
    measured: dict[str, int] = {}
    results: list[StepResult]


class ShotSummary(BaseModel):
    shots: int
    seed: int | None
    counts: dict[str, int]
    probabilities: dict[str, float]
    measured_qubits: list[int]
