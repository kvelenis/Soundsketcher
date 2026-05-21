from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from app.core.config import Settings
from app.services.objectifier import (
    _patch_clap_feature_outputs,
    _patch_clap_processor_audio_keyword,
    _patch_transformers_from_pretrained_cache,
)
from app.storage.cache import objectifier_user_edits

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

SEMANTIC_TERMS: list[dict[str, str]] = [
    {"term": "percussion", "description": "rhythmic hit, drum, or percussive strike"},
    {"term": "metallic", "description": "metallic ring, clang, or resonant metal sound"},
    {"term": "bright", "description": "bright, sharp, or high-frequency sound"},
    {"term": "dark", "description": "dark, low, or deep toned sound"},
    {"term": "short", "description": "short, brief, or staccato sound event"},
    {"term": "long", "description": "long, sustained, or continuous sound"},
    {"term": "noise", "description": "noise, static, or broadband sound"},
    {"term": "tonal", "description": "tonal, pitched, or harmonic sound"},
    {"term": "vocal", "description": "vocal, voice, or speech sound"},
    {"term": "melodic", "description": "melodic, musical, or instrument playing a note"},
    {"term": "attack", "description": "sudden attack or transient onset"},
    {"term": "resonant", "description": "resonant, ringing, or decaying sustain"},
    {"term": "impact", "description": "impact, thud, or collision sound"},
    {"term": "texture", "description": "texture, ambience, or continuous background layer"},
    {"term": "rhythmic", "description": "rhythmic, patterned, or repeating sound"},
    {"term": "sparse", "description": "sparse, isolated, or single event"},
    {"term": "dense", "description": "dense, layered, or complex sound cluster"},
    {"term": "harmonic", "description": "harmonic overtones or chord"},
    {"term": "breath", "description": "breath, wind, or airy sound"},
    {"term": "click", "description": "click, snap, or abrupt micro sound"},
    {"term": "reverberant", "description": "reverberant, echoing, or spacious sound"},
    {"term": "dry", "description": "dry, close, or anechoic sound"},
    {"term": "soft", "description": "soft, quiet, or gentle sound"},
]

_TERMS_LIST: list[str] = [
    f"Sound: {t['description']}." for t in SEMANTIC_TERMS
]
_TERM_NAMES: list[str] = [t["term"] for t in SEMANTIC_TERMS]

# ---------------------------------------------------------------------------
# Model cache (module-level singletons, lazily loaded once per process)
# ---------------------------------------------------------------------------

_CLAP_MODEL_CACHE: dict[str, Any] = {}
_TEXT_EMBEDDING_CACHE: dict[str, np.ndarray] = {}
_CACHE_LOCK = threading.Lock()


def _load_clap(settings: Settings) -> tuple[Any, Any, str]:
    """Return (model, processor, device). Cached after first load."""
    cache_key = str(settings.clap_model_name)
    with _CACHE_LOCK:
        if cache_key in _CLAP_MODEL_CACHE:
            return _CLAP_MODEL_CACHE[cache_key]

        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise RuntimeError("transformers or torch not available") from exc

        # Apply the same patches the legacy objectifier uses
        _patch_clap_processor_audio_keyword()
        _patch_clap_feature_outputs()
        _patch_transformers_from_pretrained_cache()

        cache_dir = str(settings.objectifier_cache_dir) if settings.objectifier_cache_dir else None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = ClapProcessor.from_pretrained(settings.clap_model_name, cache_dir=cache_dir)
        model = ClapModel.from_pretrained(settings.clap_model_name, cache_dir=cache_dir)

        # Load local checkpoint weights if the file exists
        if settings.clap_checkpoint_path and Path(settings.clap_checkpoint_path).is_file():
            state = torch.load(str(settings.clap_checkpoint_path), map_location=device, weights_only=False)
            # Checkpoint may be wrapped in a "model" key
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            try:
                model.load_state_dict(state, strict=False)
            except Exception:
                pass  # partial load is acceptable

        model = model.to(device)
        model.eval()

        result = (model, processor, device)
        _CLAP_MODEL_CACHE[cache_key] = result
        return result


def _text_embeddings(settings: Settings) -> np.ndarray:
    """Return (N_terms, embed_dim) array of normalized text embeddings."""
    cache_key = str(settings.clap_model_name)
    with _CACHE_LOCK:
        if cache_key in _TEXT_EMBEDDING_CACHE:
            return _TEXT_EMBEDDING_CACHE[cache_key]

    model, processor, device = _load_clap(settings)

    import torch
    inputs = processor(text=_TERMS_LIST, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    arr = emb.cpu().numpy()

    with _CACHE_LOCK:
        _TEXT_EMBEDDING_CACHE[cache_key] = arr
    return arr


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

CLAP_SAMPLE_RATE = 48000


def _audio_embedding_for_segment(
    audio_path: Path,
    start_time: float,
    end_time: float,
    model: Any,
    processor: Any,
    device: str,
) -> np.ndarray | None:
    """Extract normalized CLAP audio embedding for a time slice."""
    import torch
    import soundfile as sf
    import librosa

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    segment = data[start_sample:end_sample]
    if segment.size == 0:
        return None
    if segment.ndim > 1:
        segment = segment.mean(axis=1)

    if sr != CLAP_SAMPLE_RATE:
        segment = librosa.resample(segment, orig_sr=sr, target_sr=CLAP_SAMPLE_RATE)
        sr = CLAP_SAMPLE_RATE

    inputs = processor(
        audio=segment,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model.get_audio_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]


def compute_semantic_labels(
    audio_path: Path,
    clusters: list[dict[str, Any]],
    settings: Settings,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Return clusters list with semantic_labels populated on each cluster.

    Each semantic_labels entry: {"term": "percussion", "score": 0.87}
    """
    model, processor, device = _load_clap(settings)
    text_emb = _text_embeddings(settings)  # (N_terms, D)

    result = []
    for cluster in clusters:
        regions = cluster.get("regions", [])
        cluster_embs = []
        for region in regions:
            start = float(region.get("start_time", 0))
            end = float(region.get("end_time", start + 0.1))
            emb = _audio_embedding_for_segment(
                audio_path, start, end, model, processor, device
            )
            if emb is not None:
                cluster_embs.append(emb)

        if cluster_embs:
            centroid = np.mean(cluster_embs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm
            scores = (text_emb @ centroid).tolist()
            ranked = sorted(
                zip(_TERM_NAMES, scores), key=lambda x: x[1], reverse=True
            )
            semantic_labels = [
                {"term": t, "score": round(float(s), 4)} for t, s in ranked[:top_k]
            ]
        else:
            semantic_labels = []

        updated = dict(cluster)
        updated["semantic_labels"] = semantic_labels
        result.append(updated)

    return result


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

class SemanticLabelsService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            from transformers import ClapModel, ClapProcessor  # noqa: F401
            import soundfile  # noqa: F401
            return True
        except ImportError:
            return False

    def dependency_report(self) -> dict[str, bool]:
        report = {}
        for pkg in ("torch", "transformers", "soundfile"):
            try:
                __import__(pkg)
                report[pkg] = True
            except ImportError:
                report[pkg] = False
        checkpoint = self.settings.clap_checkpoint_path
        report["checkpoint_file"] = bool(checkpoint and Path(checkpoint).is_file())
        return report


# ---------------------------------------------------------------------------
# Async job queue
# ---------------------------------------------------------------------------

@dataclass
class SemanticLabelsJobState:
    job_id: str
    status: str  # queued | running | done | failed
    audio_path: Path
    queued_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


class SemanticLabelsJobQueue:
    def __init__(self, settings: Settings, max_workers: int = 1):
        self.settings = settings
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="semantic_labels"
        )
        self.lock = threading.Lock()
        self.jobs: dict[str, SemanticLabelsJobState] = {}

    @staticmethod
    def job_id_for(audio_path: Path) -> str:
        return str(audio_path.resolve())

    def get(self, audio_path: Path) -> SemanticLabelsJobState | None:
        with self.lock:
            return self.jobs.get(self.job_id_for(audio_path))

    def enqueue(self, audio_path: Path, *, force: bool = False) -> SemanticLabelsJobState:
        job_id = self.job_id_for(audio_path)
        with self.lock:
            existing = self.jobs.get(job_id)
            if existing and existing.status in {"queued", "running"}:
                return existing
            state = SemanticLabelsJobState(
                job_id=job_id,
                status="queued",
                audio_path=audio_path,
                queued_at=time.time(),
            )
            self.jobs[job_id] = state

        self.executor.submit(self._run_job, job_id, audio_path, force)
        return state

    def _run_job(self, job_id: str, audio_path: Path, force: bool) -> None:
        import json
        import traceback

        with self.lock:
            self.jobs[job_id].status = "running"
            self.jobs[job_id].started_at = time.time()

        objectifier_path = audio_path.parent / "objectifier.json"
        try:
            raw = objectifier_path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            clusters = payload.get("clusters", [])

            # Skip if labels already present and not forcing
            if not force and any(
                c.get("semantic_labels") for c in clusters
            ):
                with self.lock:
                    self.jobs[job_id].status = "done"
                    self.jobs[job_id].finished_at = time.time()
                return

            updated_clusters = compute_semantic_labels(
                audio_path, clusters, self.settings
            )
            payload["clusters"] = updated_clusters
            objectifier_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            error_msg = str(exc)
            self._write_error_log(traceback.format_exc())
            with self.lock:
                self.jobs[job_id].status = "failed"
                self.jobs[job_id].error = error_msg
                self.jobs[job_id].finished_at = time.time()
            return

        with self.lock:
            self.jobs[job_id].status = "done"
            self.jobs[job_id].finished_at = time.time()

    def _write_error_log(self, text: str) -> None:
        try:
            log_dir = self.settings.runtime_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "semantic-labels-error.log").write_text(text, encoding="utf-8")
        except Exception:
            pass
