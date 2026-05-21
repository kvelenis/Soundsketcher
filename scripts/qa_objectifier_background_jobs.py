#!/usr/bin/env python3
from pathlib import Path
import json
import sys
import tempfile
import time

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from app.core.config import Settings
from app.services.objectifier_jobs import ObjectifierJobQueue


def synthetic_features() -> list[dict]:
    return [
        {
            "timestamp": index * 0.1,
            "loudness": 0.1 + index,
            "rms": 0.2 + index,
            "spectral_centroid": 100.0 + index * 20,
            "weighted_spectral_centroid": 120.0 + index * 20,
            "spectral_bandwidth": 50.0 + index,
            "spectral_flatness": 0.01 * index,
            "spectral_flux": 0.02 * index,
            "f0_librosa": 220.0,
            "raw_periodicity": 0.5,
            "mir_mps_roughness": 0.1,
        }
        for index in range(12)
    ]


def wait_for_done(queue: ObjectifierJobQueue, audio_path: Path, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = queue.get(audio_path)
        if state and state.status in {"done", "failed"}:
            return state
        time.sleep(0.05)
    raise AssertionError("objectifier background job timed out")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        settings = Settings(
            project_root=root,
            legacy_root=root,
            static_dir=root / "static",
            sandbox_static_dir=root / "static",
            templates_dir=root / "templates",
            runtime_dir=root / "runtime",
            cache_root=root / "runtime" / "user_data",
            legacy_objectifier_module_path=None,
        )
        audio_path = settings.cache_root / "hash" / "tone.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"placeholder")

        queue = ObjectifierJobQueue(settings)
        state = queue.enqueue(audio_path, synthetic_features(), force=True)
        if state.status not in {"queued", "running"}:
            raise AssertionError(f"unexpected queued state: {state.status}")

        state = wait_for_done(queue, audio_path)
        if state.status != "done":
            raise AssertionError(f"objectifier job failed: {state.error}")

        objectifier_path = audio_path.parent / "objectifier.json"
        payload = json.loads(objectifier_path.read_text(encoding="utf-8"))
        if not payload.get("clusters"):
            raise AssertionError("objectifier job did not write clusters")

        state = queue.enqueue(audio_path, synthetic_features(), force=False)
        if state.status != "done":
            raise AssertionError("existing objectifier was not reported as done")

    print("objectifier background job queue -> ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
