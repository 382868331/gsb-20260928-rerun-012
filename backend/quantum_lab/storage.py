from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .models import CircuitDraft, Project, ProjectCreate, ProjectUpdate, VersionInfo


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectStore:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, project_id: str) -> Path:
        safe = uuid.UUID(project_id).hex
        return self.data_dir / f"{safe}.json"

    def _read(self, project_id: str) -> Project:
        with self._path(project_id).open("r", encoding="utf-8") as handle:
            return Project.model_validate(json.load(handle))

    def _write(self, project: Project) -> None:
        path = self._path(project.id)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(project.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)

    def create(self, request: ProjectCreate) -> Project:
        now = utc_now()
        project = Project(
            id=str(uuid.uuid4()),
            name=request.name,
            created_at=now,
            updated_at=now,
            current_version=1,
            versions=[
                VersionInfo(
                    version=1,
                    created_at=now,
                    note="initial",
                    circuit=CircuitDraft(
                        qubits=request.qubits,
                        initial_state=request.initial_state,
                        seed=request.seed,
                        steps=request.steps,
                    ),
                )
            ],
        )
        with self._lock:
            self._write(project)
        return project

    def list(self) -> list[Project]:
        projects = [
            Project.model_validate(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.data_dir.glob("*.json"))
        ]
        return sorted(projects, key=lambda item: item.updated_at, reverse=True)

    def get(self, project_id: str) -> Project:
        try:
            return self._read(project_id)
        except (FileNotFoundError, ValueError) as exc:
            raise KeyError(project_id) from exc

    def latest(self, project_id: str) -> VersionInfo:
        return self.get(project_id).versions[-1]

    def version(self, project_id: str, version: int) -> VersionInfo:
        project = self.get(project_id)
        for item in project.versions:
            if item.version == version:
                return item
        raise KeyError((project_id, version))

    def update(self, project_id: str, request: ProjectUpdate, note: str = "") -> Project:
        with self._lock:
            project = self.get(project_id)
            next_version = project.current_version + 1
            project.versions.append(
                VersionInfo(
                    version=next_version,
                    created_at=utc_now(),
                    note=note,
                    circuit=request.circuit,
                )
            )
            project.name = request.name or project.name
            project.current_version = next_version
            project.updated_at = utc_now()
            self._write(project)
            return project
