#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request
import wave
from io import BytesIO

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from app.core.config import get_settings
from app.storage.cache import CacheStore


CACHE_HASH = "abc123stage5"
CACHE_HASH_V1 = "abc123stagev1"
FILENAME = "qa tone.wav"


def reset_v1_fixture() -> None:
    settings = get_settings()
    folder = settings.cache_root / CACHE_HASH_V1
    folder.mkdir(parents=True, exist_ok=True)
    (folder / FILENAME).write_bytes(make_wav_bytes())
    (folder / "features.json").write_text(
        json.dumps({"Song": {"audio_file": FILENAME, "features_per_timestamp": [{"timestamp": 0.0, "spectral_centroid": 99.0}]}}),
        encoding="utf-8",
    )
    # v1 format: user edits at top level, no schema_version
    (folder / "objectifier.json").write_text(
        json.dumps(
            {
                "audio_file": FILENAME,
                "general_info": {"extractor": "legacy_v1", "objectifier_mode": "legacy_fast"},
                "cluster_labels": {"1": "v1 cluster"},
                "deleted_regions": ["1::0.000::1.000"],
                "region_overrides": {},
                "hidden_clusters": ["3"],
                "only_cluster": None,
                "notes": "v1 notes",
                "clusters": [{"id": 1, "label": 1, "color": "hsl(210, 80%, 52%)", "start_time": 0.0, "end_time": 1.0,
                               "regions": [{"id": "1.1", "label": 1, "start_time": 0.0, "end_time": 1.0}]}],
            }
        ),
        encoding="utf-8",
    )


def reset_fixture() -> None:
    settings = get_settings()
    folder = settings.cache_root / CACHE_HASH
    folder.mkdir(parents=True, exist_ok=True)
    (folder / FILENAME).write_bytes(make_wav_bytes())
    (folder / "features.json").write_text(
        json.dumps(
            {
                "Song": {
                    "audio_file": FILENAME,
                    "features_per_timestamp": [
                        {"timestamp": 0.0, "spectral_centroid": 123.0}
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    # v2 fixture
    (folder / "objectifier.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "audio": {"filename": FILENAME, "hash": CACHE_HASH},
                "extraction": {"mode": "legacy_fast", "extractor": "legacy_wav2vec_clap_objectifier_legacy_fast", "elapsed_seconds": None},
                "clusters": [
                    {
                        "id": 1,
                        "label": 1,
                        "color": "hsl(210, 80%, 52%)",
                        "semantic_labels": [],
                        "start_time": 0.0,
                        "end_time": 1.0,
                        "regions": [
                            {"id": "1.1", "label": 1, "start_time": 0.0, "end_time": 1.0, "duration": 1.0}
                        ],
                    }
                ],
                "user_edits": {
                    "cluster_labels": {"1": "fixture cluster"},
                    "deleted_regions": ["1::0.000::1.000"],
                    "region_overrides": {"1::0.000::1.000": {"start_time": 0.1, "end_time": 0.9}},
                    "hidden_clusters": ["2"],
                    "only_cluster": None,
                    "notes": "fixture notes",
                },
            }
        ),
        encoding="utf-8",
    )


def get_json(base_url: str, path: str):
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=10) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body) if body else None


def post_form(base_url: str, path: str, fields: dict[str, list[str] | str], timeout: int = 10):
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
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else None
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8")
        return error.code, json.loads(body) if body else None


def post_json(base_url: str, path: str, payload: dict, timeout: int = 10):
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else None
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8")
        return error.code, json.loads(body) if body else None


def post_multipart_upload(
    base_url: str,
    path: str,
    field_name: str,
    filename: str,
    content: bytes,
    fields: dict[str, str] | None = None,
    timeout: int = 10,
):
    boundary = "----stage5qa"
    parts = []
    for key, value in (fields or {}).items():
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8") + content + b"\r\n"
    )
    body = b"".join(parts) + f"--{boundary}--\r\n".encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def make_wav_bytes(frequency: float = 440.0, duration: float = 0.25, sample_rate: int = 22050) -> bytes:
    buffer = BytesIO()
    amplitude = 12000
    frame_count = int(duration * sample_rate)
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            value = int(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate))
            frames.extend(value.to_bytes(2, byteorder="little", signed=True))
        wav_file.writeframes(bytes(frames))
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    args = parser.parse_args()
    reset_v1_fixture()
    reset_fixture()

    checks = []

    # v1 backward-compat: old files without schema_version must still load correctly
    status, v1_loaded = post_form(
        args.base_url,
        "/load_cached_audio",
        {"filename": FILENAME, "hash": CACHE_HASH_V1},
    )
    if status != 200 or v1_loaded.get("files_processed") != 1:
        raise AssertionError(f"v1 backward-compat load failed: {status} {v1_loaded}")
    if v1_loaded["data"][0].get("cluster_labels") != {"1": "v1 cluster"}:
        raise AssertionError("v1 cluster_labels did not load")
    if v1_loaded["data"][0].get("deleted_regions") != ["1::0.000::1.000"]:
        raise AssertionError("v1 deleted_regions did not load")
    if v1_loaded["data"][0].get("hidden_clusters") != ["3"]:
        raise AssertionError("v1 hidden_clusters did not load")
    if v1_loaded["data"][0].get("notes") != "v1 notes":
        raise AssertionError("v1 notes did not load")
    # v1 edit routes must still write to the correct section
    status, v1_label_update = post_json(
        args.base_url,
        "/objectifier_cluster_labels",
        {"filename": FILENAME, "audio_hash": CACHE_HASH_V1, "cluster_labels": {"1": "v1 renamed"}},
    )
    if status != 200 or v1_label_update.get("cluster_labels") != {"1": "v1 renamed"}:
        raise AssertionError(f"v1 cluster label update failed: {status} {v1_label_update}")
    checks.append("v1 schema backward-compat -> ok")
    status, feature_status = get_json(args.base_url, "/feature_extraction/status")
    if status != 200 or "dependencies" not in feature_status:
        raise AssertionError(f"unexpected feature extraction status: {feature_status}")
    checks.append("/feature_extraction/status -> ok")

    query = urllib.parse.urlencode({"audio_hash": CACHE_HASH})
    status, exists = get_json(args.base_url, f"/check_file_exists?{query}")
    if status != 200 or exists != {"features_exists": True, "objectifier_exists": True}:
        raise AssertionError(f"unexpected check_file_exists response: {exists}")
    checks.append("/check_file_exists -> ok")

    status, cached = get_json(args.base_url, "/list_cached_files")
    files = cached.get("cached_files", []) if isinstance(cached, dict) else []
    if status != 200 or not any(item.get("hash") == CACHE_HASH for item in files):
        raise AssertionError("/list_cached_files did not include fixture")
    checks.append("/list_cached_files -> ok")

    status, loaded = post_form(
        args.base_url,
        "/load_cached_audio",
        {"filename": FILENAME, "hash": CACHE_HASH},
    )
    if status != 200 or loaded.get("files_processed") != 1:
        raise AssertionError(f"unexpected load_cached_audio response: {loaded}")
    if loaded["data"][0]["features"][0]["spectral_centroid"] != 123.0:
        raise AssertionError("cached features did not round-trip")
    if loaded["data"][0].get("cluster_labels") != {"1": "fixture cluster"}:
        raise AssertionError("cached objectifier cluster labels did not round-trip")
    if loaded["data"][0].get("deleted_regions") != ["1::0.000::1.000"]:
        raise AssertionError("cached objectifier region edits did not round-trip")
    if loaded["data"][0].get("region_overrides") != {"1::0.000::1.000": {"start_time": 0.1, "end_time": 0.9}}:
        raise AssertionError("cached objectifier boundary overrides did not round-trip")
    if loaded["data"][0].get("hidden_clusters") != ["2"]:
        raise AssertionError("cached objectifier hidden_clusters did not round-trip")
    if loaded["data"][0].get("notes") != "fixture notes":
        raise AssertionError("cached objectifier notes did not round-trip")
    checks.append("/load_cached_audio -> ok")

    status, objectifier_status = get_json(
        args.base_url,
        f"/objectifier_status?{urllib.parse.urlencode({'audio_hash': CACHE_HASH, 'filename': FILENAME})}",
    )
    if status != 200 or objectifier_status.get("status") != "done":
        raise AssertionError(f"unexpected objectifier status response: {objectifier_status}")
    if not objectifier_status.get("clusters"):
        raise AssertionError("objectifier status did not return fixture clusters")
    if objectifier_status.get("cluster_labels") != {"1": "fixture cluster"}:
        raise AssertionError("objectifier status did not return fixture cluster labels")
    if objectifier_status.get("deleted_regions") != ["1::0.000::1.000"]:
        raise AssertionError("objectifier status did not return fixture region edits")
    if objectifier_status.get("region_overrides") != {"1::0.000::1.000": {"start_time": 0.1, "end_time": 0.9}}:
        raise AssertionError("objectifier status did not return fixture boundary overrides")
    if objectifier_status.get("hidden_clusters") != ["2"]:
        raise AssertionError("objectifier status did not return fixture hidden_clusters")
    if objectifier_status.get("notes") != "fixture notes":
        raise AssertionError("objectifier status did not return fixture notes")
    checks.append("/objectifier_status -> ok")

    status, label_update = post_json(
        args.base_url,
        "/objectifier_cluster_labels",
        {
            "filename": FILENAME,
            "audio_hash": CACHE_HASH,
            "cluster_labels": {"1": "renamed fixture", "2": ""},
        },
    )
    if status != 200 or label_update.get("cluster_labels") != {"1": "renamed fixture"}:
        raise AssertionError(f"unexpected objectifier label update response: {status} {label_update}")
    status, relabeled = post_form(
        args.base_url,
        "/load_cached_audio",
        {"filename": FILENAME, "hash": CACHE_HASH},
    )
    if status != 200 or relabeled["data"][0].get("cluster_labels") != {"1": "renamed fixture"}:
        raise AssertionError("objectifier cluster labels did not persist to cache")
    checks.append("/objectifier_cluster_labels -> ok")

    status, region_update = post_json(
        args.base_url,
        "/objectifier_region_edits",
        {
            "filename": FILENAME,
            "audio_hash": CACHE_HASH,
            "deleted_regions": ["1::0.000::1.000", ""],
            "region_overrides": {
                "1::0.000::1.000": {"start_time": 0.2, "end_time": 0.8},
                "bad": {"start_time": 1.0, "end_time": 0.5},
            },
        },
    )
    if status != 200 or region_update.get("deleted_regions") != ["1::0.000::1.000"]:
        raise AssertionError(f"unexpected objectifier region edit response: {status} {region_update}")
    if region_update.get("region_overrides") != {"1::0.000::1.000": {"start_time": 0.2, "end_time": 0.8}}:
        raise AssertionError(f"unexpected objectifier boundary edit response: {status} {region_update}")
    status, edited = post_form(
        args.base_url,
        "/load_cached_audio",
        {"filename": FILENAME, "hash": CACHE_HASH},
    )
    if status != 200 or edited["data"][0].get("deleted_regions") != ["1::0.000::1.000"]:
        raise AssertionError("objectifier region edits did not persist to cache")
    if edited["data"][0].get("region_overrides") != {"1::0.000::1.000": {"start_time": 0.2, "end_time": 0.8}}:
        raise AssertionError("objectifier boundary edits did not persist to cache")
    checks.append("/objectifier_region_edits -> ok")

    status, visibility_update = post_json(
        args.base_url,
        "/objectifier_visibility",
        {
            "filename": FILENAME,
            "audio_hash": CACHE_HASH,
            "hidden_clusters": ["3", "5", ""],
            "only_cluster": "3",
        },
    )
    if status != 200 or visibility_update.get("hidden_clusters") != ["3", "5"]:
        raise AssertionError(f"unexpected objectifier visibility response: {status} {visibility_update}")
    if visibility_update.get("only_cluster") != "3":
        raise AssertionError(f"only_cluster did not round-trip: {visibility_update}")
    status, vis_loaded = post_form(
        args.base_url,
        "/load_cached_audio",
        {"filename": FILENAME, "hash": CACHE_HASH},
    )
    if status != 200 or vis_loaded["data"][0].get("hidden_clusters") != ["3", "5"]:
        raise AssertionError("objectifier hidden_clusters did not persist to cache")
    if vis_loaded["data"][0].get("only_cluster") != "3":
        raise AssertionError("objectifier only_cluster did not persist to cache")
    checks.append("/objectifier_visibility -> ok")

    status, notes_update = post_json(
        args.base_url,
        "/objectifier_notes",
        {
            "filename": FILENAME,
            "audio_hash": CACHE_HASH,
            "notes": "  updated notes  ",
        },
    )
    if status != 200 or notes_update.get("notes") != "updated notes":
        raise AssertionError(f"unexpected objectifier notes response: {status} {notes_update}")
    status, notes_loaded = post_form(
        args.base_url,
        "/load_cached_audio",
        {"filename": FILENAME, "hash": CACHE_HASH},
    )
    if status != 200 or notes_loaded["data"][0].get("notes") != "updated notes":
        raise AssertionError("objectifier notes did not persist to cache")
    checks.append("/objectifier_notes -> ok")

    status, reused = post_form(
        args.base_url,
        "/upload_wavs?reuse_cached=true",
        {"filenames": [FILENAME], "hashes": [CACHE_HASH]},
    )
    if status != 200 or reused.get("files_processed") != 1:
        raise AssertionError(f"unexpected upload reuse response: {reused}")
    checks.append("/upload_wavs?reuse_cached=true -> ok")

    status, recalculated = post_form(
        args.base_url,
        "/recalculate_features",
        {"filenames": [FILENAME], "hashes": [CACHE_HASH], "run_objectifier": "false"},
        timeout=10,
    )
    if status != 200 or recalculated.get("files_processed") != 1:
        raise AssertionError(f"unexpected recalculate response: {status} {recalculated}")
    recalc_job = recalculated["data"][0].get("feature_job")
    if not recalc_job:
        raise AssertionError("recalculate did not return feature_job")
    recalc_params = urllib.parse.urlencode({"audio_hash": CACHE_HASH, "filename": FILENAME})
    recalc_done = False
    for _attempt in range(150):
        import time as _time
        _time.sleep(2)
        _st, recalc_status = get_json(args.base_url, f"/feature_extraction_job_status?{recalc_params}")
        if _st != 200:
            raise AssertionError(f"recalculate feature_extraction_job_status returned {_st}")
        if recalc_status.get("status") == "done":
            recalc_features = recalc_status.get("features", [])
            if not recalc_features:
                raise AssertionError("recalculate job done but no features returned")
            loudness_vals = [item.get("loudness", 0.0) for item in recalc_features]
            if max(loudness_vals) <= 0:
                raise AssertionError("recalculate did not produce nonzero loudness")
            recalc_done = True
            break
        if recalc_status.get("status") == "failed":
            raise AssertionError(f"recalculate failed: {recalc_status.get('error')}")
    if not recalc_done:
        raise AssertionError("recalculate feature extraction timed out")
    checks.append("/recalculate_features -> extracted")

    status, fresh = post_multipart_upload(
        args.base_url,
        "/upload_wavs",
        "audio_files",
        "../unsafe name.wav",
        make_wav_bytes(frequency=660.0),
        fields={"run_objectifier": "false"},
        timeout=10,
    )
    if status != 200 or fresh.get("files_processed") != 1:
        raise AssertionError(f"fresh upload did not enqueue: {status} {fresh}")
    saved_filename = fresh["filename"][0]
    saved_hash = fresh["hash"][0]
    if saved_filename != "unsafe name.wav":
        raise AssertionError(f"filename was not sanitized: {fresh}")
    saved_folder = get_settings().cache_root / saved_hash
    if not (saved_folder / saved_filename).exists():
        raise AssertionError("fresh upload was not saved to sandbox cache")
    feature_job = fresh["data"][0].get("feature_job")
    if not feature_job:
        raise AssertionError("fresh upload did not return feature_job")
    checks.append("/upload_wavs fresh upload -> enqueued")

    # Poll feature_extraction_job_status until done (max 5 minutes)
    import time as _time
    feat_params = urllib.parse.urlencode({"audio_hash": saved_hash, "filename": saved_filename})
    feat_done = False
    feat_features = []
    for _attempt in range(150):
        _time.sleep(2)
        _st, feat_status = get_json(args.base_url, f"/feature_extraction_job_status?{feat_params}")
        if _st != 200:
            raise AssertionError(f"feature_extraction_job_status returned {_st}")
        if feat_status.get("status") == "done":
            feat_features = feat_status.get("features", [])
            feat_done = True
            break
        if feat_status.get("status") == "failed":
            raise AssertionError(f"feature extraction failed: {feat_status.get('error')}")
    if not feat_done:
        raise AssertionError("feature extraction job timed out")
    if not feat_features:
        raise AssertionError("feature_extraction_job_status done but no features returned")
    fresh_pitch_values = [item.get("f0_librosa", 0.0) for item in feat_features]
    fresh_loudness_values = [item.get("loudness", 0.0) for item in feat_features]
    if max(fresh_loudness_values) <= 0:
        raise AssertionError("feature extraction did not produce nonzero loudness")
    if not (saved_folder / "features.json").exists():
        raise AssertionError("feature extraction did not write features.json")
    checks.append("/feature_extraction_job_status -> done")

    status, sem_status = get_json(
        args.base_url,
        f"/semantic_labels_status?{urllib.parse.urlencode({'audio_hash': CACHE_HASH, 'filename': FILENAME})}",
    )
    if status != 200 or "available" not in sem_status:
        raise AssertionError(f"unexpected semantic_labels_status response: {status} {sem_status}")
    checks.append("/semantic_labels_status -> ok")

    # Exercise the sanitizers directly for a couple of edge cases.
    if CacheStore.safe_filename("../../bad?.wav") != "bad_.wav":
        raise AssertionError("safe_filename did not sanitize traversal")
    checks.append("filename sanitization -> ok")

    for check in checks:
        print(check)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, urllib.error.HTTPError, urllib.error.URLError) as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
