from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import json
import threading
import time
import traceback
from typing import Any

from app.core.config import Settings
from app.services.legacy_features import run_legacy_extraction


FEATURE_PROGRESS_MILESTONES: dict[str, tuple[int, str]] = {
    "queued": (0, "Waiting for the feature extraction worker"),
    "loading_audio": (5, "Loading and preparing audio"),
    "sonic_annotator": (15, "Running Sonic Annotator periodicity analysis"),
    "librosa": (28, "Extracting spectral and pitch features"),
    "aubio": (36, "Extracting YIN pitch fallback"),
    "crepe": (52, "Estimating pitch confidence with CREPE"),
    "mosqito": (68, "Computing loudness and sharpness"),
    "matlab": (84, "Computing MIR roughness features in MATLAB"),
    "derived": (92, "Combining extracted features"),
    "writing": (96, "Writing feature cache"),
    "objectifier": (98, "Starting objectifier analysis"),
    "done": (100, "Feature extraction complete"),
    "failed": (100, "Feature extraction failed"),
}


@dataclass
class FeatureExtractionJobState:
    job_id: str
    status: str  # queued | running | done | failed
    audio_path: Path
    queued_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    progress: int = 0
    stage: str = "queued"
    message: str = FEATURE_PROGRESS_MILESTONES["queued"][1]

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


class FeatureExtractionJobQueue:
    def __init__(self, settings: Settings, objectifier_jobs: Any) -> None:
        self.settings = settings
        self.objectifier_jobs = objectifier_jobs
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="feature_extraction")
        self.lock = threading.Lock()
        self.jobs: dict[str, FeatureExtractionJobState] = {}

    @staticmethod
    def job_id_for(audio_path: Path) -> str:
        return str(audio_path.resolve())

    def get(self, audio_path: Path) -> FeatureExtractionJobState | None:
        with self.lock:
            return self.jobs.get(self.job_id_for(audio_path))

    def enqueue(
        self,
        audio_path: Path,
        n_fft: int = 2048,
        hop_length: int = 1024,
        normalize_audio: bool = False,
        run_objectifier: bool = True,
        *,
        force: bool = False,
    ) -> FeatureExtractionJobState:
        job_id = self.job_id_for(audio_path)
        features_path = audio_path.parent / "features.json"

        with self.lock:
            existing = self.jobs.get(job_id)
            if existing and existing.status in {"queued", "running"}:
                return existing

            if features_path.exists() and not force:
                state = FeatureExtractionJobState(
                    job_id=job_id,
                    status="done",
                    audio_path=audio_path,
                    queued_at=time.time(),
                    finished_at=time.time(),
                    progress=100,
                    stage="done",
                    message=FEATURE_PROGRESS_MILESTONES["done"][1],
                )
                self.jobs[job_id] = state
                return state

            if force:
                features_path.unlink(missing_ok=True)

            state = FeatureExtractionJobState(
                job_id=job_id,
                status="queued",
                audio_path=audio_path,
                queued_at=time.time(),
            )
            self.jobs[job_id] = state

        self.executor.submit(self._run_job, job_id, audio_path, n_fft, hop_length, normalize_audio, run_objectifier)
        return state

    def cached_done_state(self, audio_path: Path) -> FeatureExtractionJobState | None:
        features_path = audio_path.parent / "features.json"
        if not features_path.exists():
            return None
        stat = features_path.stat()
        return FeatureExtractionJobState(
            job_id=self.job_id_for(audio_path),
            status="done",
            audio_path=audio_path,
            queued_at=stat.st_mtime,
            started_at=None,
            finished_at=stat.st_mtime,
            progress=100,
            stage="done",
            message=FEATURE_PROGRESS_MILESTONES["done"][1],
        )

    def _set_progress(self, job_id: str, stage: str, *, status: str | None = None) -> None:
        progress, message = FEATURE_PROGRESS_MILESTONES.get(stage, (0, stage.replace("_", " ")))
        with self.lock:
            state = self.jobs.get(job_id)
            if not state:
                return
            if status:
                state.status = status
            state.stage = stage
            state.progress = progress
            state.message = message

    def read_features(self, audio_path: Path) -> list[dict[str, Any]] | None:
        features_path = audio_path.parent / "features.json"
        if not features_path.exists():
            return None
        try:
            payload = json.loads(features_path.read_text(encoding="utf-8"))
            return payload.get("Song", {}).get("features_per_timestamp", [])
        except Exception:
            return None

    def _run_job(
        self,
        job_id: str,
        audio_path: Path,
        n_fft: int,
        hop_length: int,
        normalize_audio: bool,
        run_objectifier: bool,
    ) -> None:
        with self.lock:
            self.jobs[job_id].status = "running"
            self.jobs[job_id].started_at = time.time()
            self.jobs[job_id].stage = "loading_audio"
            self.jobs[job_id].progress = FEATURE_PROGRESS_MILESTONES["loading_audio"][0]
            self.jobs[job_id].message = FEATURE_PROGRESS_MILESTONES["loading_audio"][1]

        try:
            rows = run_legacy_extraction(
                wav_path=audio_path,
                settings=self.settings,
                n_fft=n_fft,
                hop_length=hop_length,
                normalize_audio=normalize_audio,
                progress_callback=lambda stage: self._set_progress(job_id, stage),
            )

            self._set_progress(job_id, "writing")
            features_path = audio_path.parent / "features.json"
            payload = {
                "Song": {
                    "audio_file": audio_path.name,
                    "features_per_timestamp": rows,
                    "general_info": {"extractor": "legacy_full_pipeline_v1"},
                }
            }
            features_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            if run_objectifier:
                self._set_progress(job_id, "objectifier")
                self.objectifier_jobs.enqueue(audio_path, rows, force=True)

        except Exception as exc:
            self._write_error_log(traceback.format_exc())
            with self.lock:
                self.jobs[job_id].status = "failed"
                self.jobs[job_id].error = str(exc)
                self.jobs[job_id].finished_at = time.time()
                self.jobs[job_id].progress = FEATURE_PROGRESS_MILESTONES["failed"][0]
                self.jobs[job_id].stage = "failed"
                self.jobs[job_id].message = FEATURE_PROGRESS_MILESTONES["failed"][1]
            return

        with self.lock:
            self.jobs[job_id].status = "done"
            self.jobs[job_id].finished_at = time.time()
            self.jobs[job_id].progress = FEATURE_PROGRESS_MILESTONES["done"][0]
            self.jobs[job_id].stage = "done"
            self.jobs[job_id].message = FEATURE_PROGRESS_MILESTONES["done"][1]

    def _write_error_log(self, text: str) -> None:
        try:
            log_dir = self.settings.runtime_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "feature-extraction-error.log").write_text(text, encoding="utf-8")
        except Exception:
            pass
