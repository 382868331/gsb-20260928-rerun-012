from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .models import (
    CircuitDraft,
    Project,
    ProjectCreate,
    ProjectUpdate,
    RunCreate,
    RunState,
    ShotSummary,
)
from .simulator import CircuitValidationError, execute_next, make_run, require_valid, run_shots
from .storage import ProjectStore


DATA_DIR = Path(os.environ.get("QUANTUM_LAB_DATA", Path(__file__).resolve().parents[2] / "data"))
store = ProjectStore(DATA_DIR)
runs: dict[str, RunState] = {}
run_circuits: dict[str, CircuitDraft] = {}


class ShotRequest(BaseModel):
    circuit: CircuitDraft | None = None
    shots: int = Field(default=1000, ge=1, le=100000)


class ValidationResponse(BaseModel):
    valid: bool
    errors: list[str]


class OperationInfo(BaseModel):
    type: str
    name: str
    category: str
    target_required: bool
    control_required: bool
    angle_required: bool
    description: str


app = FastAPI(title="Local Quantum Computing Lab", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


def validation_error_handler(exc: CircuitValidationError) -> HTTPException:
    return HTTPException(status_code=422, detail=exc.errors)


def resolve_circuit(project_id: str | None, version: int | None, payload: CircuitDraft | None) -> tuple[CircuitDraft, str | None, int | None]:
    if project_id is not None:
        try:
            item = store.version(project_id, version) if version is not None else store.latest(project_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="项目或版本不存在")
        return item.circuit, project_id, item.version
    if payload is None:
        raise HTTPException(status_code=400, detail="必须提供 circuit 或 project_id")
    return payload, None, None


@app.get("/api/health")
async def health() -> dict[str, bool]:
    return {"local": True, "quantum_cloud_required": False}


@app.get("/api/operations", response_model=list[OperationInfo])
async def operations() -> list[OperationInfo]:
    return [
        OperationInfo(type="X", name="Pauli-X / 非门", category="单比特", target_required=True, control_required=False, angle_required=False, description="翻转 |0⟩ 与 |1⟩"),
        OperationInfo(type="Y", name="Pauli-Y", category="单比特", target_required=True, control_required=False, angle_required=False, description="绕 Y 轴旋转 180 度"),
        OperationInfo(type="Z", name="Pauli-Z", category="单比特", target_required=True, control_required=False, angle_required=False, description="对 |1⟩ 取负相位"),
        OperationInfo(type="H", name="Hadamard", category="单比特", target_required=True, control_required=False, angle_required=False, description="生成 |0⟩ 与 |1⟩ 的等权叠加"),
        OperationInfo(type="S", name="Phase S", category="单比特", target_required=True, control_required=False, angle_required=False, description="90 度相位门"),
        OperationInfo(type="T", name="Phase T", category="单比特", target_required=True, control_required=False, angle_required=False, description="45 度相位门"),
        OperationInfo(type="RX", name="RX 旋转", category="单比特", target_required=True, control_required=False, angle_required=True, description="绕 Bloch X 轴旋转指定弧度"),
        OperationInfo(type="RY", name="RY 旋转", category="单比特", target_required=True, control_required=False, angle_required=True, description="绕 Bloch Y 轴旋转指定弧度"),
        OperationInfo(type="RZ", name="RZ 旋转", category="单比特", target_required=True, control_required=False, angle_required=True, description="绕 Bloch Z 轴旋转指定弧度"),
        OperationInfo(type="CNOT", name="受控 NOT", category="双比特", target_required=True, control_required=True, angle_required=False, description="控制位为 1 时翻转目标位；用于关联量子位"),
        OperationInfo(type="SWAP", name="交换", category="双比特", target_required=True, control_required=True, angle_required=False, description="交换两个量子位的状态"),
        OperationInfo(type="MEASURE", name="测量", category="测量", target_required=True, control_required=False, angle_required=False, description="按概率采样并坍缩，结果影响后续步骤"),
    ]


@app.post("/api/validate", response_model=ValidationResponse)
async def validate(circuit: CircuitDraft) -> ValidationResponse:
    from .simulator import validate_circuit

    errors = validate_circuit(circuit)
    return ValidationResponse(valid=not errors, errors=errors)


@app.post("/api/projects", response_model=Project, status_code=201)
async def create_project(request: ProjectCreate) -> Project:
    circuit = CircuitDraft(
        qubits=request.qubits,
        initial_state=request.initial_state,
        seed=request.seed,
        steps=request.steps,
    )
    try:
        require_valid(circuit)
    except CircuitValidationError as exc:
        raise validation_error_handler(exc) from exc
    return store.create(request)


@app.get("/api/projects", response_model=list[Project])
async def list_projects() -> list[Project]:
    return store.list()


@app.get("/api/projects/{project_id}", response_model=Project)
async def get_project(project_id: str) -> Project:
    try:
        return store.get(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc


@app.put("/api/projects/{project_id}", response_model=Project)
async def update_project(project_id: str, request: ProjectUpdate) -> Project:
    try:
        require_valid(request.circuit)
        return store.update(project_id, request)
    except CircuitValidationError as exc:
        raise validation_error_handler(exc) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc


@app.post("/api/runs", response_model=RunState, status_code=201)
async def start_run(project_id: str | None = None, version: int | None = None, request: RunCreate | None = None) -> RunState:
    circuit, resolved_project, resolved_version = resolve_circuit(
        project_id,
        version,
        request.circuit if request else None,
    )
    try:
        run = make_run(circuit, resolved_project, resolved_version)
    except CircuitValidationError as exc:
        raise validation_error_handler(exc) from exc
    runs[run.run_id] = run
    run_circuits[run.run_id] = circuit
    return run


@app.post("/api/runs/{run_id}/next", response_model=RunState)
async def next_step(run_id: str) -> RunState:
    run = runs.get(run_id)
    circuit = run_circuits.get(run_id)
    if run is None or circuit is None:
        raise HTTPException(status_code=404, detail="运行会话不存在或已随服务重启失效")
    rng_seed = None if run.seed is None else (run.seed + run.index + 1) * 1000003
    rng = np.random.default_rng(rng_seed)
    try:
        execute_next(run, circuit.steps, rng)
    except CircuitValidationError as exc:
        raise validation_error_handler(exc) from exc
    return run


@app.get("/api/runs/{run_id}", response_model=RunState)
async def get_run(run_id: str) -> RunState:
    try:
        return runs[run_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="运行会话不存在") from exc


@app.post("/api/shots", response_model=ShotSummary)
async def shots(project_id: str | None = None, version: int | None = None, request: ShotRequest | None = None) -> ShotSummary:
    circuit, _, _ = resolve_circuit(project_id, version, request.circuit if request else None)
    shot_count = request.shots if request else 1000
    try:
        return run_shots(circuit, shot_count)
    except CircuitValidationError as exc:
        raise validation_error_handler(exc) from exc
