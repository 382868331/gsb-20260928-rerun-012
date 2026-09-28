from backend import quantum, server


def test_stepped_session_measurement_collapses_on_server():
    payload = {
        "qubits": 2,
        "initial_basis": "00",
        "seed": 11,
        "steps": [
            {"type": "H", "target": 0},
            {"type": "CNOT", "control": 0, "target": 1},
            {"type": "MEASURE", "target": 0},
            {"type": "MEASURE", "target": 1},
        ],
    }
    session = server.Session(payload)
    assert session.snapshot()["probabilities"] == {"00": 1.0, "01": 0.0, "10": 0.0, "11": 0.0}

    first = session.step()
    assert first["executed"]["type"] == "H"
    assert abs(first["probabilities"]["00"] - 0.5) < 1e-12

    session.step()  # CNOT creates a Bell state
    after_first_measurement = session.step()
    first_outcome = after_first_measurement["executed"]["measured"]["outcome"]
    surviving = f"{first_outcome}{first_outcome}"
    assert after_first_measurement["probabilities"][surviving] == 1.0

    after_second_measurement = session.step()
    assert after_second_measurement["finished"] is True
    assert after_second_measurement["measurements"][1]["outcome"] == first_outcome


def test_circuit_version_persistence(monkeypatch, tmp_path):
    path = tmp_path / "circuits.json"
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "CIRCUITS_FILE", path)

    circuit = {
        "qubits": 1,
        "initial_basis": "0",
        "seed": 3,
        "steps": [{"type": "H", "target": 0}, {"type": "MEASURE", "target": 0}],
    }

    def store():
        with server.STORAGE_LOCK:
            data = server._load_circuits()
            previous = data["circuits"].get("sample")
            version = int(previous["version"]) + 1 if previous else 1
            record = {"name": "sample", "version": version, **circuit}
            data["circuits"]["sample"] = record
            server._save_circuits(data)
            return record

    assert store()["version"] == 1
    assert store()["version"] == 2
    loaded = server._load_circuits()["circuits"]["sample"]
    assert loaded["version"] == 2
    assert loaded["seed"] == 3
    assert quantum.validate_circuit(loaded) == []
