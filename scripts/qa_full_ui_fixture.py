#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from app.core.config import get_settings


def get_json(base_url: str, path: str):
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=10) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body) if body else None


def post_form(base_url: str, path: str, fields: dict[str, str], timeout: int = 10):
    data = urllib.parse.urlencode(fields).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"content-type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body) if body else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    args = parser.parse_args()
    settings = get_settings()

    folder = settings.cache_root / settings.preferred_example_hash
    audio_path = folder / settings.preferred_example_filename
    features_path = folder / "features.json"
    if not audio_path.is_file():
        raise AssertionError(f"missing fixture audio: {audio_path}")
    if not features_path.is_file():
        raise AssertionError(f"missing fixture features: {features_path}")

    status, cached = get_json(args.base_url, "/list_cached_files")
    files = cached.get("cached_files", []) if isinstance(cached, dict) else []
    if status != 200 or not files:
        raise AssertionError(f"unexpected cached file list: {cached}")

    preferred = files[0]
    if preferred.get("hash") != settings.preferred_example_hash:
        raise AssertionError(f"preferred fixture is not first: {files[:3]}")
    if preferred.get("filename") != settings.preferred_example_filename:
        raise AssertionError(f"preferred filename mismatch: {preferred}")
    if preferred.get("is_preferred") is not True:
        raise AssertionError(f"preferred flag missing: {preferred}")

    status, loaded = post_form(
        args.base_url,
        "/load_cached_audio",
        {
            "filename": settings.preferred_example_filename,
            "hash": settings.preferred_example_hash,
        },
    )
    if status != 200 or loaded.get("files_processed") != 1:
        raise AssertionError(f"unexpected load response: {loaded}")

    data = loaded["data"][0]
    features = data.get("features")
    if not isinstance(features, list) or len(features) < 700:
        raise AssertionError(f"fixture has too few feature frames: {len(features or [])}")
    clusters = data.get("clusters")
    if clusters is not None and not isinstance(clusters, list):
        raise AssertionError(f"fixture clusters should be a list when present: {type(clusters)}")

    print("/list_cached_files preferred 20-37s.wav -> ok")
    cluster_text = "clusters available" if clusters else "clusters not cached"
    print(f"/load_cached_audio 20-37s.wav -> ok ({len(features)} frames, {cluster_text})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
