import json
import math
import struct
import sys
import wave
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.features import FeatureExtractionOptions, FeatureExtractionService  # noqa: E402


EXAMPLE_HASH = "abc123stage5"
EXAMPLE_FILENAME = "qa tone.wav"


def main() -> None:
    settings = get_settings()
    folder = settings.cache_root / EXAMPLE_HASH
    folder.mkdir(parents=True, exist_ok=True)

    audio_path = folder / EXAMPLE_FILENAME
    write_example_wav(audio_path)

    service = FeatureExtractionService(settings)
    service.extract(
        audio_path,
        FeatureExtractionOptions(
            n_fft=1024,
            overlap=0.75,
            normalize_audio=True,
            save_json=True,
            run_objectifier=False,
        ),
    )
    write_objectifier(folder / "objectifier.json")

    print(f"Seeded {EXAMPLE_FILENAME} at {folder}")


def write_example_wav(path: Path) -> None:
    sample_rate = 22_050
    duration_seconds = 3.0
    sample_count = int(sample_rate * duration_seconds)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for index in range(sample_count):
            t = index / sample_rate
            sweep_hz = 180 + (420 * t / duration_seconds)
            tremolo = 0.55 + 0.35 * math.sin(2 * math.pi * 2.2 * t)
            envelope = min(1.0, t / 0.08, (duration_seconds - t) / 0.12)
            sample = envelope * tremolo * math.sin(2 * math.pi * sweep_hz * t)
            frames.extend(struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32767)))

        wav_file.writeframes(bytes(frames))


def write_objectifier(path: Path) -> None:
    payload = {
        "audio_file": EXAMPLE_FILENAME,
        "clusters": [
            {
                "id": 1,
                "color": "hsl(210, 80%, 52%)",
                "start_time": 0.0,
                "end_time": 1.0,
                "regions": [{"id": "1.1", "start_time": 0.0, "end_time": 1.0}],
            },
            {
                "id": 2,
                "color": "hsl(340, 74%, 56%)",
                "start_time": 1.0,
                "end_time": 2.0,
                "regions": [{"id": "2.1", "start_time": 1.0, "end_time": 2.0}],
            },
            {
                "id": 3,
                "color": "hsl(42, 86%, 48%)",
                "start_time": 2.0,
                "end_time": 3.0,
                "regions": [{"id": "3.1", "start_time": 2.0, "end_time": 3.0}],
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
