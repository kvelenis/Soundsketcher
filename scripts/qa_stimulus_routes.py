#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from app.api.experiments import SOUND_LIST_ROUTES
from app.core.config import get_settings
from app.services.stimuli import AUDIO_EXTENSIONS


def get_json(base_url: str, path: str):
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=10) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body) if body else None


def count_static_audio(relative_dir: str) -> int:
    directory = get_settings().static_dir / relative_dir
    if not directory.exists():
        return 0
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    args = parser.parse_args()

    checks: list[str] = []
    for route_path, relative_dir in SOUND_LIST_ROUTES.items():
        status, payload = get_json(args.base_url, route_path)
        expected = count_static_audio(relative_dir)
        if status != 200:
            raise AssertionError(f"{route_path} returned HTTP {status}")
        if not isinstance(payload, list):
            raise AssertionError(f"{route_path} did not return a list")
        if len(payload) != expected:
            raise AssertionError(
                f"{route_path} returned {len(payload)} items, expected {expected}"
            )
        checks.append(f"{route_path} -> ok ({len(payload)} files)")

    status, supplementary = get_json(args.base_url, "/get_sounds_supplementary")
    expected_supplementary = count_static_audio("image_shape")
    if status != 200 or not isinstance(supplementary, list):
        raise AssertionError("/get_sounds_supplementary did not return a list")
    if len(supplementary) != expected_supplementary:
        raise AssertionError("/get_sounds_supplementary count mismatch")
    checks.append(f"/get_sounds_supplementary -> ok ({len(supplementary)} files)")

    status, noise_tonal = get_json(
        args.base_url,
        "/questionnaires/noise-tonal-preference/v1/stimuli",
    )
    if status != 200 or not isinstance(noise_tonal, dict):
        raise AssertionError("noise-tonal endpoint did not return an object")
    stimuli = noise_tonal.get("stimuli")
    if not isinstance(stimuli, list) or noise_tonal.get("total_sounds") != len(stimuli):
        raise AssertionError("noise-tonal endpoint has inconsistent shape")
    if not stimuli:
        raise AssertionError("noise-tonal endpoint returned no stimuli")
    required_keys = {"id", "stimulus_id", "audio_filenames", "files", "audio_urls"}
    missing = required_keys.difference(stimuli[0])
    if missing:
        raise AssertionError(f"noise-tonal stimulus is missing keys: {sorted(missing)}")
    checks.append(
        f"/questionnaires/noise-tonal-preference/v1/stimuli -> ok ({len(stimuli)} stimuli)"
    )

    for check in checks:
        print(check)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, urllib.error.HTTPError, urllib.error.URLError) as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
