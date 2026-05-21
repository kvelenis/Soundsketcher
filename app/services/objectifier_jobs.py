from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any

from app.core.config import Settings
from app.services.objectifier import build_objectifier_payload, write_objectifier_json


OBJECTIFIER_PROGRESS_MILESTONES: dict[str, tuple[int, str]] = {
    "queued": (0, "Waiting for the objectifier worker"),
    "loading": (10, "Loading audio and feature frames"),
    "extracting": (35, "Extracting objectifier embeddings and clusters"),
    "postprocessing": (75, "Post-processing regions and clusters"),
    "writing": (92, "Writing objectifier cache"),
    "done": (100, "Objectifier analysis complete"),
    "failed": (100, "Objectifier analysis failed"),
}


@dataclass
class ObjectifierJobState:
    job_id: str
    status: str
    audio_path: Path
    queued_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    progress: int = 0
    stage: str = "queued"
    message: str = OBJECTIFIER_PROGRESS_MILESTONES["queued"][1]

    def to_dict(self) -> dict[str, Any]:
        now = time.time()
        elapsed_seconds = None
        if self.started_at:
            elapsed_seconds = (self.finished_at or now) - self.started_at
        return {
            "job_id": self.job_id,
            "status": self.status,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "progress": self.progress,
            "stage": self.stage,
            "message": self.message,
            "elapsed_seconds": elapsed_seconds,
        }


class ObjectifierJobQueue:
    def __init__(self, settings: Settings, max_workers: int = 1):
        self.settings = settings
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="objectifier")
        self.lock = threading.Lock()
        self.jobs: dict[str, ObjectifierJobState] = {}

    @staticmethod
    def job_id_for(audio_path: Path) -> str:
        return str(audio_path.resolve())

    def get(self, audio_path: Path) -> ObjectifierJobState | None:
        with self.lock:
            return self.jobs.get(self.job_id_for(audio_path))

    def enqueue(
        self,
        audio_path: Path,
        features: list[dict[str, Any]],
        *,
        force: bool = False,
    ) -> ObjectifierJobState:
        job_id = self.job_id_for(audio_path)
        objectifier_path = audio_path.parent / "objectifier.json"

        with self.lock:
            existing = self.jobs.get(job_id)
            if existing and existing.status in {"queued", "running"}:
                return existing
            if force:
                objectifier_path.unlink(missing_ok=True)
            if objectifier_path.exists() and not force:
                state = ObjectifierJobState(
                    job_id=job_id,
                    status="done",
                    audio_path=audio_path,
                    queued_at=time.time(),
                    finished_at=time.time(),
                    progress=100,
                    stage="done",
                    message=OBJECTIFIER_PROGRESS_MILESTONES["done"][1],
                )
                self.jobs[job_id] = state
                return state

            state = ObjectifierJobState(
                job_id=job_id,
                status="queued",
                audio_path=audio_path,
                queued_at=time.time(),
            )
            self.jobs[job_id] = state

        self.executor.submit(self._run_job, job_id, audio_path, features)
        return state

    def _set_progress(self, job_id: str, stage: str, *, status: str | None = None) -> None:
        progress, message = OBJECTIFIER_PROGRESS_MILESTONES.get(stage, (0, stage.replace("_", " ")))
        with self.lock:
            state = self.jobs.get(job_id)
            if not state:
                return
            if status:
                state.status = status
            state.stage = stage
            state.progress = progress
            state.message = message

    def _run_job(
        self,
        job_id: str,
        audio_path: Path,
        features: list[dict[str, Any]],
    ) -> None:
        with self.lock:
            state = self.jobs[job_id]
            state.status = "running"
            state.started_at = time.time()
            state.stage = "loading"
            state.progress = OBJECTIFIER_PROGRESS_MILESTONES["loading"][0]
            state.message = OBJECTIFIER_PROGRESS_MILESTONES["loading"][1]

        try:
            self._set_progress(job_id, "extracting")
            payload = build_objectifier_payload(audio_path, features, settings=self.settings)
            self._set_progress(job_id, "writing")
            write_objectifier_json(audio_path, payload)
        except Exception as error:
            with self.lock:
                state = self.jobs[job_id]
                state.status = "failed"
                state.error = str(error)
                state.finished_at = time.time()
                state.progress = OBJECTIFIER_PROGRESS_MILESTONES["failed"][0]
                state.stage = "failed"
                state.message = OBJECTIFIER_PROGRESS_MILESTONES["failed"][1]
            return

        with self.lock:
            state = self.jobs[job_id]
            state.status = "done"
            state.finished_at = time.time()
            state.progress = OBJECTIFIER_PROGRESS_MILESTONES["done"][0]
            state.stage = "done"
            state.message = OBJECTIFIER_PROGRESS_MILESTONES["done"][1]
