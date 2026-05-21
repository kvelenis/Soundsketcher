#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from app.core.config import get_settings


SESSION_ID = "stage3_qa"


def request_json(base_url: str, method: str, path: str, payload: dict | None = None) -> tuple[int, dict | list]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body) if body else {}


def check_status(base_url: str, method: str, path: str, payload: dict | None = None) -> str:
    status, body = request_json(base_url, method, path, payload)
    if status != 200:
        raise AssertionError(f"{method} {path} returned HTTP {status}")
    if isinstance(body, dict) and body.get("status") not in (None, "ok"):
        raise AssertionError(f"{method} {path} returned unexpected body {body}")
    return f"{method} {path} -> ok"


def assert_file(path: Path) -> str:
    if not path.exists():
        raise AssertionError(f"Expected file was not written: {path}")
    return f"wrote {path.relative_to(get_settings().project_root)}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    args = parser.parse_args()
    settings = get_settings()

    checks = [
        check_status(
            args.base_url,
            "POST",
            "/save_shape_response",
            {
                "session_id": SESSION_ID,
                "sound": "shape.wav",
                "response": {"shapeValue": 0.42, "timestamp": "2026-05-04T00:00:00Z"},
            },
        ),
        check_status(
            args.base_url,
            "POST",
            "/save_texture_response",
            {
                "session_id": SESSION_ID,
                "sound": "texture.wav",
                "response": {"textureValue": 0.64, "timestamp": "2026-05-04T00:00:00Z"},
            },
        ),
        check_status(
            args.base_url,
            "POST",
            "/save_responses_indefinite_pitch",
            {
                "session_id": SESSION_ID,
                "sound": "pitch.wav",
                "response": {"slider": 3, "timestamp": "2026-05-04T00:00:00Z"},
            },
        ),
        check_status(
            args.base_url,
            "POST",
            "/save_indefinite_pitch_user_info",
            {
                "session_id": SESSION_ID,
                "gender": "prefer_not_to_say",
                "age": 30,
                "music_experience": "qa",
                "plays_instrument": False,
                "instrument_name": "",
                "knows_pitch": False,
                "hearing_condition": "none",
                "absolute_pitch": False,
            },
        ),
        check_status(
            args.base_url,
            "POST",
            "/questionnaires/noise-tonal-preference/v1/save-user-info",
            {
                "session_id": SESSION_ID,
                "gender": "prefer_not_to_say",
                "age": 30,
                "music_education_years": 0,
                "plays_instrument": False,
                "instrument_name": "",
                "consent_given": True,
                "consent_version": "qa",
            },
        ),
        check_status(
            args.base_url,
            "POST",
            "/questionnaires/noise-tonal-preference/v1/save-response",
            {
                "session_id": SESSION_ID,
                "order_index": 0,
                "total_sounds": 1,
                "stimulus_id": "stimulus_01",
                "audio_filenames": ["noise.wav", "tonal.wav"],
                "slider_value": 0.5,
                "timestamp": "2026-05-04T00:00:00Z",
                "mode": "qa",
            },
        ),
        check_status(
            args.base_url,
            "POST",
            "/save_supplementary_response",
            {
                "session_id": SESSION_ID,
                "sound": "shape.wav",
                "order_index": 0,
                "response": {
                    "smoothness": 1,
                    "roughness": 2,
                    "sharpness": 3,
                    "timestamp": "2026-05-04T00:00:00Z",
                },
            },
        ),
    ]

    status, stimuli = request_json(
        args.base_url,
        "GET",
        "/questionnaires/noise-tonal-preference/v1/stimuli",
    )
    if status != 200 or not isinstance(stimuli, dict) or stimuli.get("total_sounds", 0) < 1:
        raise AssertionError("noise tonal stimuli endpoint did not return stimuli")
    checks.append("GET /questionnaires/noise-tonal-preference/v1/stimuli -> ok")

    status, sounds = request_json(args.base_url, "GET", "/get_sounds_supplementary")
    if status != 200 or not isinstance(sounds, list):
        raise AssertionError("supplementary sounds endpoint did not return a list")
    checks.append("GET /get_sounds_supplementary -> ok")

    checks.extend(
        [
            assert_file(settings.response_root / "responses_shape" / f"{SESSION_ID}.json"),
            assert_file(settings.response_root / "responses_texture" / f"{SESSION_ID}.json"),
            assert_file(settings.response_root / "responses_indefinite_pitch" / f"{SESSION_ID}.json"),
            assert_file(settings.response_root / "responses_indefinite_pitch" / "user_info" / f"{SESSION_ID}.json"),
            assert_file(settings.response_root / "responses_supplementary" / f"{SESSION_ID}.json"),
            assert_file(settings.response_root / "questionnaire_results" / "noise_tonal_preference" / "user_info.jsonl"),
            assert_file(settings.response_root / "questionnaire_results" / "noise_tonal_preference" / "responses.jsonl"),
        ]
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
