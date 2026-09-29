import asyncio
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from quantum_lab.app import app, store


def bell_payload() -> dict:
    return {
        "qubits": 2,
        "initial_state": {"kind": "basis", "basis": "00", "amplitudes": None},
        "seed": 23,
        "steps": [
            {"id": "h", "type": "H", "target": 0, "control": None, "angle": None},
            {"id": "cx", "type": "CNOT", "target": 1, "control": 0, "angle": None},
            {"id": "m0", "type": "MEASURE", "target": 0, "control": None, "angle": None},
            {"id": "m1", "type": "MEASURE", "target": 1, "control": None, "angle": None},
        ],
    }


async def run_api_case() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/projects", json={"name": "Bell", **bell_payload()})
        assert response.status_code == 201
        project = response.json()
        assert project["current_version"] == 1

        changed = bell_payload()
        changed["seed"] = 24
        response = await client.put(
            "/api/projects/" + project["id"],
            json={"name": "Bell", "circuit": changed},
        )
        assert response.status_code == 200
        assert response.json()["current_version"] == 2

        illegal = bell_payload()
        illegal["steps"][1]["control"] = 1
        response = await client.post("/api/projects", json={"name": "Bad", **illegal})
        assert response.status_code == 422

        response = await client.post("/api/runs", json={"circuit": bell_payload()})
        assert response.status_code == 201
        run = response.json()
        for _ in range(4):
            response = await client.post("/api/runs/" + run["run_id"] + "/next")
            assert response.status_code == 200
        run = response.json()
        assert run["complete"] is True
        assert set(run["probabilities"]) <= {"00", "11"}

        response = await client.post(
            "/api/shots",
            json={"circuit": bell_payload(), "shots": 1000},
        )
        assert response.status_code == 200
        assert set(response.json()["counts"]) == {"00", "11"}


def test_project_versioning_validation_running_and_shots(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(store, "data_dir", tmp_path)
    asyncio.run(run_api_case())
