import json
import math
from typing import Optional

from fastapi import APIRouter, Body, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.storage.cache import objectifier_user_edits
from app.services.features import (
    FeatureExtractionService,
    serialize_dependency_report,
)
from app.services.objectifier_jobs import ObjectifierJobQueue
from app.services.semantic_labels import SemanticLabelsJobQueue, SemanticLabelsService
from app.services.feature_extraction_jobs import FeatureExtractionJobQueue
from app.storage.cache import CacheStore


router = APIRouter(prefix="", tags=["audio"])
settings = get_settings()
cache = CacheStore(
    settings.cache_root,
    preferred_hashes=[settings.preferred_example_hash],
)
features = FeatureExtractionService(settings)
objectifier_jobs = ObjectifierJobQueue(settings)
semantic_labels_jobs = SemanticLabelsJobQueue(settings)
_semantic_labels_service = SemanticLabelsService(settings)
feature_extraction_jobs = FeatureExtractionJobQueue(settings, objectifier_jobs)


@router.get("/check_file_exists")
async def check_file_exists(audio_hash: str = Query(...)):
    return cache.check_file_exists(audio_hash)


@router.get("/list_cached_files")
async def list_cached_files():
    return JSONResponse(content={"cached_files": cache.list_cached_files()})


@router.get("/feature_extraction/status")
async def feature_extraction_status():
    report = features.dependency_report()
    return JSONResponse(
        content={
            "available": features.is_available(),
            "dependencies": serialize_dependency_report(report),
        }
    )


@router.get("/feature_extraction_job_status")
async def feature_extraction_job_status(
    audio_hash: str = Query(...),
    filename: str = Query(...),
):
    safe_hash = cache.safe_hash(audio_hash)
    safe_filename = cache.safe_filename(filename)
    audio_path = settings.cache_root / safe_hash / safe_filename

    state = feature_extraction_jobs.get(audio_path)
    if state is None:
        cached_state = feature_extraction_jobs.cached_done_state(audio_path)
        if cached_state is not None:
            response = cached_state.to_dict()
            response["features"] = feature_extraction_jobs.read_features(audio_path) or []
            obj_job = objectifier_jobs.get(audio_path)
            if obj_job:
                response["objectifier_job"] = obj_job.to_dict()
            return JSONResponse(content=response)
        return JSONResponse(content={"status": "unknown"})

    response = state.to_dict()
    if state.status == "done":
        response["features"] = feature_extraction_jobs.read_features(audio_path) or []
        obj_job = objectifier_jobs.get(audio_path)
        if obj_job:
            response["objectifier_job"] = obj_job.to_dict()

    return JSONResponse(content=response)


@router.get("/objectifier_status")
async def objectifier_status(
    audio_hash: str = Query(...),
    filename: str = Query(...),
):
    safe_hash = cache.safe_hash(audio_hash)
    safe_filename = cache.safe_filename(filename)
    audio_path = settings.cache_root / safe_hash / safe_filename
    folder = audio_path.parent
    objectifier_path = folder / "objectifier.json"
    if objectifier_path.exists():
        result = cache.load_cached_audio(safe_hash) or {}
        return JSONResponse(
            content={
                "status": "done",
                "objectifier_exists": True,
                "clusters": result.get("clusters"),
                "cluster_labels": result.get("cluster_labels", {}),
                "deleted_regions": result.get("deleted_regions", []),
                "region_overrides": result.get("region_overrides", {}),
                "hidden_clusters": result.get("hidden_clusters", []),
                "only_cluster": result.get("only_cluster"),
                "notes": result.get("notes", ""),
                "filename": safe_filename,
                "hash": safe_hash,
            }
        )

    job = objectifier_jobs.get(audio_path)
    if job is not None:
        return JSONResponse(
            content={
                **job.to_dict(),
                "objectifier_exists": False,
                "filename": safe_filename,
                "hash": safe_hash,
            }
        )

    features_exists = (folder / "features.json").exists()
    return JSONResponse(
        content={
            "status": "missing" if features_exists else "unknown",
            "objectifier_exists": False,
            "filename": safe_filename,
            "hash": safe_hash,
        }
    )


@router.get("/semantic_labels_status")
async def semantic_labels_status(
    audio_hash: str = Query(...),
    filename: str = Query(...),
):
    safe_hash = cache.safe_hash(audio_hash)
    safe_filename = cache.safe_filename(filename)
    audio_path = settings.cache_root / safe_hash / safe_filename

    available = _semantic_labels_service.is_available()
    job = semantic_labels_jobs.get(audio_path)
    job_dict = job.to_dict() if job else None

    return JSONResponse(
        content={
            "available": available,
            "job": job_dict,
            "filename": safe_filename,
            "hash": safe_hash,
        }
    )


@router.post("/objectifier_semantic_labels")
async def objectifier_semantic_labels_route(
    audio_hash: str = Body(...),
    filename: str = Body(...),
    force: bool = Body(default=False),
):
    safe_hash = cache.safe_hash(audio_hash)
    safe_filename = cache.safe_filename(filename)
    audio_path = settings.cache_root / safe_hash / safe_filename
    objectifier_path = audio_path.parent / "objectifier.json"

    if not objectifier_path.exists():
        return JSONResponse(status_code=404, content={"error": "objectifier data not found"})

    if not _semantic_labels_service.is_available():
        report = _semantic_labels_service.dependency_report()
        return JSONResponse(
            status_code=503,
            content={
                "error": "CLAP semantic labeling is not available",
                "dependencies": report,
            },
        )

    job = semantic_labels_jobs.enqueue(audio_path, force=force)
    return JSONResponse(
        content={
            "status": "ok",
            "job": job.to_dict(),
            "filename": safe_filename,
            "hash": safe_hash,
        }
    )


@router.post("/load_cached_audio")
async def load_cached_audio(filename: str = Form(...), hash: str = Form(...)):
    result = cache.load_cached_audio(hash)
    if result is None:
        return JSONResponse(status_code=404, content={"Error": "cached features not found."})

    safe_filename = cache.safe_filename(filename)
    return JSONResponse(
        content={
            "files_processed": 1,
            "filename": [safe_filename],
            "hash": [cache.safe_hash(hash)],
            "audio_url": [cache.audio_url(hash, safe_filename)],
            "data": [
                {
                    "features": result.get("features"),
                    "clusters": result.get("clusters"),
                    "cluster_labels": result.get("cluster_labels", {}),
                    "deleted_regions": result.get("deleted_regions", []),
                    "region_overrides": result.get("region_overrides", {}),
                    "hidden_clusters": result.get("hidden_clusters", []),
                    "only_cluster": result.get("only_cluster"),
                    "notes": result.get("notes", ""),
                }
            ],
        }
    )


@router.post("/objectifier_cluster_labels")
async def objectifier_cluster_labels(
    audio_hash: str = Body(...),
    filename: str = Body(...),
    cluster_labels: dict[str, str] | None = Body(default=None),
):
    safe_hash = cache.safe_hash(audio_hash)
    safe_filename = cache.safe_filename(filename)
    objectifier_path = settings.cache_root / safe_hash / "objectifier.json"
    if not objectifier_path.exists():
        return JSONResponse(status_code=404, content={"error": "objectifier data not found"})

    try:
        payload = json.loads(objectifier_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return JSONResponse(status_code=500, content={"error": "objectifier data is invalid"})

    if not isinstance(payload, dict):
        return JSONResponse(status_code=500, content={"error": "objectifier data is invalid"})

    normalized_labels = {
        str(label): str(name).strip()
        for label, name in (cluster_labels or {}).items()
        if str(name).strip()
    }
    payload["audio_file"] = payload.get("audio_file") or safe_filename
    objectifier_user_edits(payload)["cluster_labels"] = normalized_labels
    objectifier_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return JSONResponse(
        content={
            "status": "ok",
            "filename": safe_filename,
            "hash": safe_hash,
            "cluster_labels": normalized_labels,
        }
    )


@router.post("/objectifier_region_edits")
async def objectifier_region_edits(
    audio_hash: str = Body(...),
    filename: str = Body(...),
    deleted_regions: list[str] | None = Body(default=None),
    region_overrides: dict[str, dict[str, float]] | None = Body(default=None),
):
    safe_hash = cache.safe_hash(audio_hash)
    safe_filename = cache.safe_filename(filename)
    objectifier_path = settings.cache_root / safe_hash / "objectifier.json"
    if not objectifier_path.exists():
        return JSONResponse(status_code=404, content={"error": "objectifier data not found"})

    try:
        payload = json.loads(objectifier_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return JSONResponse(status_code=500, content={"error": "objectifier data is invalid"})

    if not isinstance(payload, dict):
        return JSONResponse(status_code=500, content={"error": "objectifier data is invalid"})

    normalized_deleted = [
        str(region_key).strip()
        for region_key in (deleted_regions or [])
        if str(region_key).strip()
    ]
    source_overrides = region_overrides if region_overrides is not None else payload.get("region_overrides", {})
    normalized_overrides = {}
    for region_key, override in (source_overrides if isinstance(source_overrides, dict) else {}).items():
        key = str(region_key).strip()
        if not key or not isinstance(override, dict):
            continue
        try:
            start_time = float(override.get("start_time"))
            end_time = float(override.get("end_time"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start_time) or not math.isfinite(end_time):
            continue
        if start_time < 0 or end_time <= start_time:
            continue
        normalized_overrides[key] = {
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
        }
    payload["audio_file"] = payload.get("audio_file") or safe_filename
    edits = objectifier_user_edits(payload)
    edits["deleted_regions"] = normalized_deleted
    edits["region_overrides"] = normalized_overrides
    objectifier_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return JSONResponse(
        content={
            "status": "ok",
            "filename": safe_filename,
            "hash": safe_hash,
            "deleted_regions": normalized_deleted,
            "region_overrides": normalized_overrides,
        }
    )


@router.post("/objectifier_visibility")
async def objectifier_visibility(
    audio_hash: str = Body(...),
    filename: str = Body(...),
    hidden_clusters: list[str] | None = Body(default=None),
    only_cluster: str | None = Body(default=None),
):
    safe_hash = cache.safe_hash(audio_hash)
    safe_filename = cache.safe_filename(filename)
    objectifier_path = settings.cache_root / safe_hash / "objectifier.json"
    if not objectifier_path.exists():
        return JSONResponse(status_code=404, content={"error": "objectifier data not found"})

    try:
        payload = json.loads(objectifier_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return JSONResponse(status_code=500, content={"error": "objectifier data is invalid"})

    if not isinstance(payload, dict):
        return JSONResponse(status_code=500, content={"error": "objectifier data is invalid"})

    normalized_hidden = [
        str(label).strip()
        for label in (hidden_clusters or [])
        if str(label).strip()
    ]
    normalized_only = str(only_cluster).strip() if only_cluster and str(only_cluster).strip() else None

    payload["audio_file"] = payload.get("audio_file") or safe_filename
    edits = objectifier_user_edits(payload)
    edits["hidden_clusters"] = normalized_hidden
    edits["only_cluster"] = normalized_only
    objectifier_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return JSONResponse(
        content={
            "status": "ok",
            "filename": safe_filename,
            "hash": safe_hash,
            "hidden_clusters": normalized_hidden,
            "only_cluster": normalized_only,
        }
    )


@router.post("/objectifier_notes")
async def objectifier_notes(
    audio_hash: str = Body(...),
    filename: str = Body(...),
    notes: str | None = Body(default=None),
):
    safe_hash = cache.safe_hash(audio_hash)
    safe_filename = cache.safe_filename(filename)
    objectifier_path = settings.cache_root / safe_hash / "objectifier.json"
    if not objectifier_path.exists():
        return JSONResponse(status_code=404, content={"error": "objectifier data not found"})

    try:
        payload = json.loads(objectifier_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return JSONResponse(status_code=500, content={"error": "objectifier data is invalid"})

    if not isinstance(payload, dict):
        return JSONResponse(status_code=500, content={"error": "objectifier data is invalid"})

    normalized_notes = str(notes).strip() if notes and str(notes).strip() else ""
    payload["audio_file"] = payload.get("audio_file") or safe_filename
    objectifier_user_edits(payload)["notes"] = normalized_notes
    objectifier_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return JSONResponse(
        content={
            "status": "ok",
            "filename": safe_filename,
            "hash": safe_hash,
            "notes": normalized_notes,
        }
    )


@router.post("/upload_wavs")
async def upload_wavs(
    audio_files: Optional[list[UploadFile]] = File(None),
    filenames: Optional[list[str]] = Form(None),
    hashes: Optional[list[str]] = Form(None),
    reuse_cached: bool = Query(False),
    n_fft: int = Form(2048),
    overlap: float = Form(0.5),
    normalize_audio: bool = Form(False),
    apply_filter: bool = Form(False),
    save_json: bool = Form(False),
    run_objectifier: bool = Form(True),
):
    if reuse_cached and hashes and filenames:
        return cached_upload_response(filenames, hashes)

    if not audio_files:
        return JSONResponse(status_code=400, content={"error": "No audio files provided."})

    saved_files = []
    for audio_file in audio_files:
        content = await audio_file.read()
        audio_hash, safe_filename = cache.save_upload(audio_file.filename, content)
        saved_files.append(
            {
                "filename": safe_filename,
                "hash": audio_hash,
                "audio_url": cache.audio_url(audio_hash, safe_filename),
            }
        )

    hop_length = max(1, n_fft - round(n_fft * overlap))
    extracted_data = []
    for saved_file in saved_files:
        audio_path = settings.cache_root / saved_file["hash"] / saved_file["filename"]
        feature_job = feature_extraction_jobs.enqueue(
            audio_path,
            n_fft=n_fft,
            hop_length=hop_length,
            normalize_audio=normalize_audio,
            run_objectifier=run_objectifier,
            force=True,
        )
        # If features were already cached (job returned done immediately), include them
        features_list: list = []
        if feature_job.status == "done":
            features_list = feature_extraction_jobs.read_features(audio_path) or []

        # Load any existing objectifier/session data from cache
        clusters = None
        cluster_labels: dict = {}
        deleted_regions: list = []
        region_overrides: dict = {}
        hidden_clusters: list = []
        only_cluster = None
        notes = ""
        if (audio_path.parent / "objectifier.json").exists():
            cached_result = cache.load_cached_audio(saved_file["hash"])
            if cached_result:
                clusters = cached_result.get("clusters")
                cluster_labels = cached_result.get("cluster_labels", {})
                deleted_regions = cached_result.get("deleted_regions", [])
                region_overrides = cached_result.get("region_overrides", {})
                hidden_clusters = cached_result.get("hidden_clusters", [])
                only_cluster = cached_result.get("only_cluster")
                notes = cached_result.get("notes", "")

        extracted_data.append(
            {
                "features": features_list,
                "clusters": clusters,
                "cluster_labels": cluster_labels,
                "deleted_regions": deleted_regions,
                "region_overrides": region_overrides,
                "hidden_clusters": hidden_clusters,
                "only_cluster": only_cluster,
                "notes": notes,
                "objectifier_job": None,
                "feature_job": feature_job.to_dict(),
                "audio_url": cache.audio_url(saved_file["hash"], saved_file["filename"]),
            }
        )

    return JSONResponse(
        content={
            "files_processed": len(saved_files),
            "filename": [item["filename"] for item in saved_files],
            "hash": [item["hash"] for item in saved_files],
            "audio_url": [item["audio_url"] for item in saved_files],
            "data": extracted_data,
        },
    )


@router.post("/recalculate_features")
async def recalculate_features(
    filenames: list[str] = Form(...),
    hashes: list[str] = Form(...),
    n_fft: int = Form(2048),
    overlap: float = Form(0.5),
    normalize_audio: bool = Form(False),
    apply_filter: bool = Form(False),
    save_json: bool = Form(False),
    run_objectifier: bool = Form(True),
):
    hop_length = max(1, n_fft - round(n_fft * overlap))
    data = []
    safe_filenames = []
    safe_hashes = []
    audio_urls = []
    for filename, audio_hash in zip(filenames, hashes):
        safe_filename = cache.safe_filename(filename)
        safe_hash = cache.safe_hash(audio_hash)
        audio_path = settings.cache_root / safe_hash / safe_filename
        feature_job = feature_extraction_jobs.enqueue(
            audio_path,
            n_fft=n_fft,
            hop_length=hop_length,
            normalize_audio=normalize_audio,
            run_objectifier=run_objectifier,
            force=True,
        )
        safe_filenames.append(safe_filename)
        safe_hashes.append(safe_hash)
        audio_urls.append(cache.audio_url(safe_hash, safe_filename))
        data.append(
            {
                "features": [],
                "clusters": None,
                "cluster_labels": {},
                "deleted_regions": [],
                "region_overrides": {},
                "objectifier_job": None,
                "feature_job": feature_job.to_dict(),
                "audio_url": cache.audio_url(safe_hash, safe_filename),
            }
        )

    return JSONResponse(
        content={
            "files_processed": len(safe_filenames),
            "filename": safe_filenames,
            "hash": safe_hashes,
            "audio_url": audio_urls,
            "data": data,
        }
    )


def cached_upload_response(filenames: list[str], hashes: list[str]) -> JSONResponse:
    return JSONResponse(content=cached_upload_payload(filenames, hashes))


def cached_upload_payload(filenames: list[str], hashes: list[str]) -> dict:
    data = []
    audio_urls = []
    safe_filenames = []
    safe_hashes = []

    for filename, audio_hash in zip(filenames, hashes):
        safe_filename = cache.safe_filename(filename)
        safe_hash = cache.safe_hash(audio_hash)
        result = cache.load_cached_audio(safe_hash)
        if result is None:
            data.append({"features": None, "clusters": None})
        else:
            data.append(
                {
                    "features": result.get("features"),
                    "clusters": result.get("clusters"),
                    "cluster_labels": result.get("cluster_labels", {}),
                    "deleted_regions": result.get("deleted_regions", []),
                    "region_overrides": result.get("region_overrides", {}),
                    "hidden_clusters": result.get("hidden_clusters", []),
                    "only_cluster": result.get("only_cluster"),
                    "notes": result.get("notes", ""),
                }
            )
        safe_filenames.append(safe_filename)
        safe_hashes.append(safe_hash)
        audio_urls.append(cache.audio_url(safe_hash, safe_filename))

    return {
        "files_processed": len(safe_filenames),
        "filename": safe_filenames,
        "hash": safe_hashes,
        "audio_url": audio_urls,
        "data": data,
    }

