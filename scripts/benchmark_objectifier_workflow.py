#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from app.core.config import get_settings


def post_form(base_url: str, path: str, fields: dict[str, list[str] | str], timeout: int = 120):
    pairs = []
    for key, value in fields.items():
        values = value if isinstance(value, list) else [value]
        for item in values:
            pairs.append((key, item))
    data = urllib.parse.urlencode(pairs).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"content-type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body) if body else None


def get_json(base_url: str, path: str, timeout: int = 10):
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body) if body else None


def count_regions(clusters: list[dict]) -> int:
    return sum(len(cluster.get("regions") or []) for cluster in clusters)


def poll_objectifier(base_url: str, filename: str, audio_hash: str, timeout: int) -> tuple[dict, float]:
    query = urllib.parse.urlencode({"filename": filename, "audio_hash": audio_hash})
    deadline = time.monotonic() + timeout
    start = time.monotonic()
    last_payload = None
    while time.monotonic() < deadline:
        status, payload = get_json(base_url, f"/objectifier_status?{query}", timeout=10)
        if status != 200:
            raise AssertionError(f"unexpected objectifier status code: {status}")
        last_payload = payload
        if payload.get("status") == "done":
            return payload, time.monotonic() - start
        if payload.get("status") == "failed":
            raise AssertionError(f"objectifier job failed: {payload}")
        time.sleep(1)
    raise AssertionError(f"objectifier job timed out: {last_payload}")


def run_once(base_url: str, filename: str, audio_hash: str, timeout: int) -> dict:
    start = time.monotonic()
    status, response = post_form(
        base_url,
        "/recalculate_features",
        {
            "filenames": [filename],
            "hashes": [audio_hash],
            "save_json": "true",
            "run_objectifier": "true",
        },
        timeout=timeout,
    )
    initial_seconds = time.monotonic() - start
    if status != 200 or response.get("files_processed") != 1:
        raise AssertionError(f"unexpected recalculate response: {status} {response}")

    item = response["data"][0]
    features = item.get("features") or []
    job = item.get("objectifier_job") or {}
    final_status, background_seconds = poll_objectifier(base_url, filename, audio_hash, timeout)
    clusters = final_status.get("clusters") or []

    return {
        "initial_response_seconds": round(initial_seconds, 2),
        "background_seconds_after_response": round(background_seconds, 2),
        "total_seconds": round(initial_seconds + background_seconds, 2),
        "job_initial_status": job.get("status"),
        "feature_frames": len(features),
        "clusters": len(clusters),
        "regions": count_regions(clusters),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--filename")
    parser.add_argument("--hash")
    args = parser.parse_args()

    settings = get_settings()
    filename = args.filename or settings.preferred_example_filename
    audio_hash = args.hash or settings.preferred_example_hash

    results = []
    for run_index in range(args.runs):
        result = run_once(args.base_url, filename, audio_hash, args.timeout)
        result["run"] = run_index + 1
        results.append(result)
        print(json.dumps(result, sort_keys=True))

    print(
        json.dumps(
            {
                "benchmark": "objectifier_workflow",
                "filename": filename,
                "hash": audio_hash,
                "runs": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, urllib.error.HTTPError, urllib.error.URLError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        raise SystemExit(1)
