from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback
import types
from typing import Any

from app.core.config import Settings


OBJECTIFIER_COLORS = (
    "hsl(210, 80%, 52%)",
    "hsl(340, 74%, 56%)",
    "hsl(42, 86%, 48%)",
    "hsl(145, 58%, 42%)",
    "hsl(275, 62%, 58%)",
)

FEATURE_COLUMNS = (
    "loudness",
    "rms",
    "spectral_centroid",
    "weighted_spectral_centroid",
    "spectral_bandwidth",
    "spectral_flatness",
    "spectral_flux",
    "f0_librosa",
    "raw_periodicity",
    "mir_mps_roughness",
)
MIN_OBJECTIFIER_REGION_SECONDS = 0.3
REGION_COALESCE_GAP_SECONDS = 0.08

LEGACY_CLUSTER_SEARCH_CALL = (
    'optimal_clusters = determine_optimal_clusters(embeddings, max_clusters=20, method="gap")'
)
FAST_CLUSTER_SEARCH_CALL = (
    "optimal_clusters = _soundsketcher_fast_cluster_count("
    "embeddings, sr=sr, y=y, max_clusters=max_clusters)"
)
FAST_CLUSTER_COUNT_HELPER = '''

def _soundsketcher_fast_cluster_count(embeddings, sr=None, y=None, max_clusters=10):
    frame_count = len(embeddings)
    if frame_count < 3:
        return 1

    duration_seconds = None
    if y is not None and sr:
        try:
            duration_seconds = len(y) / float(sr)
        except Exception:
            duration_seconds = None
    if not duration_seconds:
        duration_seconds = frame_count * 320 / 16000

    by_duration = max(2, round(duration_seconds / 3.0))
    by_frame_budget = max(2, frame_count // 40)
    capped_max = max(2, min(max_clusters, by_frame_budget, frame_count - 1))
    cluster_count = int(max(2, min(capped_max, by_duration)))
    print(f"Fast objectifier cluster count: {cluster_count}")
    return cluster_count
'''
LEGACY_LOAD_CLAP_DEF = "    def load_Clap_model(checkpoint_path):\n        # Load CLAP model and processor"
LEGACY_LOAD_CLAP_CACHED_DEF = '''    def load_Clap_model(checkpoint_path):
        global _soundsketcher_clap_checkpoint_cache
        try:
            _soundsketcher_clap_checkpoint_cache
        except NameError:
            _soundsketcher_clap_checkpoint_cache = {}

        cache_key = str(checkpoint_path)
        if cache_key in _soundsketcher_clap_checkpoint_cache:
            print(f"Reusing custom CLAP model from: {checkpoint_path}")
            return _soundsketcher_clap_checkpoint_cache[cache_key]

        # Load CLAP model and processor'''
LEGACY_LOAD_CLAP_RETURN = "        return processor, model"
LEGACY_LOAD_CLAP_CACHED_RETURN = '''        _soundsketcher_clap_checkpoint_cache[cache_key] = (processor, model)
        return processor, model'''
LEGACY_LOAD_CLAP_PIPELINE_CALL = (
    '        clap_processor, clap_model = load_Clap_model("/mnt/ssd1/kvelenis/soundsketcher/aux_models/music_speech_audioset_epoch_15_esc_89.98.pt")'
)
FAST_LOAD_CLAP_PIPELINE_CALL = "        clap_processor, clap_model = None, None"
LEGACY_SEGMENT_CLAP_BLOCK = '''        # Extract CLAP embeddings for segments
        clap_embeddings = extract_clap_embeddings_for_segments(audio_file_path, breakpoints, sr, hop_length, clap_processor)
        # print(clap_embeddings)
        # Compare consecutive segments
        similarities = compare_consecutive_segments_with_clap(clap_embeddings)

        # Visualize similarities
        visualize_clap_segment_similarity(similarities, breakpoints, sr, hop_length, audio_file_path)

        # Visualize clusters and transitions
        visualize_clusters_and_transitions(audio_file_path, sr, y, smoothed_labels, breakpoints, hop_length ,similarities)
'''
FAST_SEGMENT_CLAP_BLOCK = '''        # Fast mode returns regions first and skips CLAP segment similarity.
        similarities = []
'''
LEGACY_SEMANTIC_CLAP_BLOCK = '''        # Generate CLAP text embeddings for semantic analysis
        text_terms = [
            "A bright sound is sharp, clear, and high in frequency. It stands out and feels crisp, often described as radiant or shimmering. Examples include cymbals, violins, or high piano notes.",
            "A dark sound is deep, muted, and rich in low frequencies. It often feels heavy, subdued, and moody. Examples include bass guitar, low brass, or ambient drones.",
            "A warm sound is rich, full, and pleasing to the ear. It has a balanced tonal quality with smooth mid and low frequencies. Examples include acoustic guitars, vocal harmonies, or a soft saxophone.",
            "A cold sound feels distant, sharp, and unemotional. It is often associated with thin or metallic tones and lacks warmth. Examples include synthetic pads, icy wind sounds, or high electronic tones.",
            "A rough sound is coarse, jagged, and textured. It feels unrefined or gritty, often with harsh edges. Examples include distorted guitars, gravel underfoot, or industrial machinery.",
            "A smooth sound is fluid, continuous, and free of abrupt changes. It feels polished and soothing. Examples include a cello melody, flowing water, or a soft breeze.",
            "A metallic sound has a resonant, ringing quality similar to struck metal. It often feels sharp and vibrant. Examples include bells, cymbals, or metal pipes.",
            "A soft sound is gentle, quiet, and unobtrusive. It often feels delicate and calming. Examples include a whisper, light footsteps, or rustling leaves.",
            "A granular sound has a fragmented or textured quality, often created by many small particles or grains. Examples include the crackle of a fire, the crunch of gravel, or digital granular synthesis.",
            "A high-pitched sound is characterized by high frequencies. It often feels sharp, thin, or piercing. Examples include a whistle, bird chirps, or a violin's upper register.",
            "A low-pitched sound is characterized by low frequencies. It feels deep, resonant, and powerful. Examples include bass notes, thunder, or a deep male voice.",
            "A harmonic sound is rich and pleasing, characterized by harmonious frequencies. Examples include a choir, an organ, or a well-tuned guitar chord.",
            "A disharmonic sound is dissonant, jarring, or unpleasant, often with clashing frequencies. Examples include off-key instruments, metal scraping, or chaotic industrial noise.",
            "A melodic sound is tuneful, flowing, and pleasing to the ear. Examples include a piano solo, a violin melody, or a bird song.",
            "A dissonant sound is harsh, clashing, and unresolved. It creates tension or unease. Examples include a horror movie score or an out-of-tune orchestra.",
            "A rhythmic sound has a structured, repetitive pattern that creates a beat or tempo. Examples include a drumbeat, hand claps, or a ticking clock.",
            "A chaotic sound is disordered, unpredictable, and lacking clear structure. Examples include a crowded marketplace, a thunderstorm, or a glitching electronic signal.",
            "A natural sound comes from the environment, often soothing and unprocessed. Examples include birdsong, rustling leaves, or ocean waves.",
            "A mechanical sound is artificial, repetitive, and associated with machines. Examples include gears turning, a ticking clock, or an engine.",
            "An urban sound captures the ambiance of a city, often a mix of various noises. Examples include traffic, footsteps on pavement, or distant sirens.",
            "A rainy sound evokes the atmosphere of rainfall, often calming and rhythmic. Examples include drops hitting a surface, gentle storms, or flowing water.",
            "A noise sound is unstructured, often random, and can range from background hums to static. Examples include white noise, crowd chatter, or static from a radio.",
            "A tonal sound has a clear pitch or tonal center. Examples include a musical note, a tuning fork, or a singing voice."
        ]
        # clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
        # clap_model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
        terms, text_embeddings = generate_text_embeddings(text_terms, clap_processor, clap_model)


        # Extract CLAP embeddings for clusters
        cluster_clap_embeddings = extract_clap_embeddings_for_clusters(audio_file_path, smoothed_labels, sr, hop_length, optimal_clusters, clap_processor, clap_model)


        # Compute centroids of CLAP embeddings for clusters
        cluster_centroids = compute_cluster_centroids(cluster_clap_embeddings)
        # Merge similar clusters based on CLAP embeddings
        merged_regions = merge_similar_clusters(cluster_regions, cluster_clap_embeddings, similarity_threshold=0.6)
        # cluster_centroids = merged_regions
        # Compare cluster centroids with text embeddings
        semantic_centroids = compare_centroids_with_text(cluster_centroids, text_embeddings, text_terms, top_n=5)
'''
FAST_SEMANTIC_CLAP_BLOCK = '''        # Fast mode skips CLAP text labeling; labels can be generated later on demand.
        semantic_centroids = {}
        merged_clusters = cluster_regions
'''
LEGACY_SEGMENT_CLAP_START_MARKER = "        # Extract CLAP embeddings for segments\n"
LEGACY_SEMANTIC_CLAP_START_MARKER = "        # Generate CLAP text embeddings for semantic analysis\n"
LEGACY_SEMANTIC_CLAP_END_MARKERS = (
    "        # Display top 5 semantic terms for each cluster\n",
    "        # === Step 7: Package all plot data",
    "        json_data = gather_plot_data_for_plotly(",
)

_PRETRAINED_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def build_objectifier_clusters(
    features: list[dict[str, Any]],
    *,
    max_regions: int = 5,
) -> list[dict[str, Any]]:
    """Build frontend-compatible objectifier regions from frame features.

    The legacy objectifier used a heavier embedding/model stack. This sandbox
    service keeps the same frontend JSON contract while clustering the feature
    frames with dependencies that are already part of the staging environment.
    """
    frames = _sorted_feature_frames(features)
    if len(frames) < 4:
        return _build_time_region_clusters(frames, max_regions=max_regions)

    try:
        labels = _cluster_feature_frames(frames, max_regions=max_regions)
    except Exception:
        return _build_time_region_clusters(frames, max_regions=max_regions)

    if not labels or len(set(labels)) < 1:
        return []

    labels = _smooth_labels(labels)
    regions = _build_regions(frames, labels)
    if not regions:
        return []

    return _group_regions_into_clusters(regions)


def build_objectifier_payload(
    audio_path: Path,
    features: list[dict[str, Any]],
    *,
    settings: Settings,
    max_regions: int = 5,
) -> dict[str, Any]:
    audio_hash = audio_path.parent.name
    legacy_payload = _run_legacy_objectifier(audio_path, settings=settings)
    if legacy_payload and isinstance(legacy_payload.get("clusters"), list):
        objectifier_mode = _normalized_objectifier_mode(settings)
        legacy_payload = _postprocess_objectifier_payload(legacy_payload)
        clusters = _add_v2_cluster_fields(legacy_payload.get("clusters", []))
        return _json_safe_payload(
            _build_v2_payload(
                audio_path.name,
                audio_hash,
                clusters,
                mode=objectifier_mode,
                extractor=f"legacy_wav2vec_clap_objectifier_{objectifier_mode}",
            )
        )

    clusters = build_objectifier_clusters(features, max_regions=max_regions)
    clusters = _postprocess_objectifier_clusters(clusters)
    clusters = _add_v2_cluster_fields(clusters)
    return _build_v2_payload(
        audio_path.name,
        audio_hash,
        clusters,
        mode="sklearn_kmeans",
        extractor="sklearn_feature_kmeans_v1",
    )


def _build_v2_payload(
    filename: str,
    audio_hash: str,
    clusters: list[dict[str, Any]],
    *,
    mode: str = "",
    extractor: str = "",
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "audio": {
            "filename": filename,
            "hash": audio_hash,
        },
        "extraction": {
            "mode": mode,
            "extractor": extractor,
            "elapsed_seconds": elapsed_seconds,
        },
        "clusters": clusters,
        "user_edits": {
            "cluster_labels": {},
            "deleted_regions": [],
            "region_overrides": {},
            "hidden_clusters": [],
            "only_cluster": None,
            "notes": "",
        },
    }


def _add_v2_cluster_fields(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for cluster in clusters:
        cluster = dict(cluster)
        cluster.setdefault("semantic_labels", [])
        regions = []
        for region in cluster.get("regions") or []:
            region = dict(region)
            start = region.get("start_time")
            end = region.get("end_time")
            if _is_number(start) and _is_number(end):
                region["duration"] = round(float(end) - float(start), 3)
            regions.append(region)
        cluster["regions"] = regions
        result.append(cluster)
    return result


def _run_legacy_objectifier(
    audio_path: Path,
    *,
    settings: Settings,
) -> dict[str, Any] | None:
    module_path = _legacy_objectifier_module_path(settings)
    if module_path is None or not module_path.exists():
        return None

    cache_dir = settings.objectifier_cache_dir or settings.runtime_dir / "model_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "huggingface" / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    if "laion_clap" not in sys.modules and importlib.util.find_spec("laion_clap") is None:
        laion_clap_shim = types.ModuleType("laion_clap")
        laion_clap_shim.__spec__ = importlib.util.spec_from_loader("laion_clap", loader=None)
        sys.modules["laion_clap"] = laion_clap_shim

    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    try:
        module = _load_legacy_objectifier_module(module_path, settings=settings)
        _patch_clap_processor_audio_keyword()
        _patch_clap_feature_outputs()
        _patch_transformers_from_pretrained_cache()
        payload = module.objectifier(str(audio_path))
    except Exception:
        _write_legacy_objectifier_error(settings)
        return None

    if not isinstance(payload, dict):
        return None
    return payload


def _load_legacy_objectifier_module(
    module_path: Path,
    *,
    settings: Settings,
) -> Any:
    objectifier_mode = _normalized_objectifier_mode(settings)
    module_name = f"soundsketcher_legacy_objectifier_{objectifier_mode}"
    if objectifier_mode == "legacy_fast":
        source = module_path.read_text(encoding="utf-8")
        source = _patch_legacy_objectifier_source_for_fast_mode(source)
        module = types.ModuleType(module_name)
        module.__file__ = str(module_path)
        exec(compile(source, str(module_path), "exec"), module.__dict__)
        return module

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load legacy objectifier module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_legacy_objectifier_source_for_fast_mode(source: str) -> str:
    if LEGACY_CLUSTER_SEARCH_CALL not in source:
        raise RuntimeError("Legacy objectifier cluster-search call was not found.")
    patched_source = source.replace(LEGACY_CLUSTER_SEARCH_CALL, FAST_CLUSTER_SEARCH_CALL)
    patched_source = _patch_legacy_objectifier_source_skip_semantics(patched_source)
    patched_source = _patch_legacy_objectifier_source_for_model_reuse(patched_source)
    return f"{FAST_CLUSTER_COUNT_HELPER}\n{patched_source}"


def _patch_legacy_objectifier_source_skip_semantics(source: str) -> str:
    if LEGACY_LOAD_CLAP_PIPELINE_CALL not in source:
        return source
    source = source.replace(LEGACY_LOAD_CLAP_PIPELINE_CALL, FAST_LOAD_CLAP_PIPELINE_CALL, 1)

    segment_start = source.find(LEGACY_SEGMENT_CLAP_START_MARKER)
    semantic_start = source.find(LEGACY_SEMANTIC_CLAP_START_MARKER, segment_start)
    semantic_end = _find_first_legacy_marker(
        source,
        LEGACY_SEMANTIC_CLAP_END_MARKERS,
        semantic_start,
    )
    if segment_start < 0 or semantic_start < 0 or semantic_end < 0:
        raise RuntimeError("Legacy objectifier semantic block was not found.")

    source = (
        source[:segment_start]
        + FAST_SEGMENT_CLAP_BLOCK
        + source[semantic_start:semantic_end]
        + source[semantic_end:]
    )
    semantic_start = source.find(LEGACY_SEMANTIC_CLAP_START_MARKER, segment_start)
    semantic_end = _find_first_legacy_marker(
        source,
        LEGACY_SEMANTIC_CLAP_END_MARKERS,
        semantic_start,
    )
    if semantic_start < 0 or semantic_end < 0:
        raise RuntimeError("Legacy objectifier semantic block was not found.")

    return source[:semantic_start] + FAST_SEMANTIC_CLAP_BLOCK + source[semantic_end:]


def _find_first_legacy_marker(source: str, markers: tuple[str, ...], start: int) -> int:
    matches = [source.find(marker, start) for marker in markers]
    matches = [match for match in matches if match >= 0]
    return min(matches) if matches else -1


def _patch_legacy_objectifier_source_for_model_reuse(source: str) -> str:
    if LEGACY_LOAD_CLAP_DEF in source:
        source = source.replace(LEGACY_LOAD_CLAP_DEF, LEGACY_LOAD_CLAP_CACHED_DEF, 1)
    if LEGACY_LOAD_CLAP_RETURN in source:
        source = source.replace(LEGACY_LOAD_CLAP_RETURN, LEGACY_LOAD_CLAP_CACHED_RETURN, 1)
    return source


def _normalized_objectifier_mode(settings: Settings) -> str:
    mode = settings.objectifier_mode.strip().lower()
    if mode in {"full", "legacy"}:
        return "legacy_full"
    if mode in {"fast", "legacy_fast"}:
        return "legacy_fast"
    if mode in {"legacy_full"}:
        return mode
    return "legacy_fast"


def _write_legacy_objectifier_error(settings: Settings) -> None:
    try:
        log_dir = settings.runtime_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "legacy-objectifier-error.log").write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
    except Exception:
        pass


def _patch_clap_processor_audio_keyword() -> None:
    try:
        from transformers import ClapProcessor
    except Exception:
        return

    if getattr(ClapProcessor.__call__, "_soundsketcher_audio_keyword_patch", False):
        return

    original_call = ClapProcessor.__call__

    def patched_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        if "audios" in kwargs and "audio" not in kwargs:
            kwargs["audio"] = kwargs.pop("audios")
        return original_call(self, *args, **kwargs)

    patched_call._soundsketcher_audio_keyword_patch = True  # type: ignore[attr-defined]
    ClapProcessor.__call__ = patched_call


def _patch_clap_feature_outputs() -> None:
    try:
        from transformers import ClapModel
    except Exception:
        return

    for method_name in ("get_audio_features", "get_text_features"):
        original_method = getattr(ClapModel, method_name, None)
        if original_method is None or getattr(
            original_method,
            "_soundsketcher_feature_output_patch",
            False,
        ):
            continue

        def patched_method(self: Any, *args: Any, _original_method: Any = original_method, **kwargs: Any) -> Any:
            result = _original_method(self, *args, **kwargs)
            if hasattr(result, "pooler_output"):
                return result.pooler_output
            if hasattr(result, "text_embeds"):
                return result.text_embeds
            if hasattr(result, "audio_embeds"):
                return result.audio_embeds
            return result

        patched_method._soundsketcher_feature_output_patch = True  # type: ignore[attr-defined]
        setattr(ClapModel, method_name, patched_method)


def _patch_transformers_from_pretrained_cache() -> None:
    try:
        from transformers import ClapModel, ClapProcessor, Wav2Vec2Model, Wav2Vec2Processor
    except Exception:
        return

    for cls in (ClapModel, ClapProcessor, Wav2Vec2Model, Wav2Vec2Processor):
        if getattr(cls.from_pretrained, "_soundsketcher_from_pretrained_cache", False):
            continue

        original_from_pretrained = (
            cls.from_pretrained.__func__
            if hasattr(cls.from_pretrained, "__func__")
            else cls.from_pretrained
        )

        def cached_from_pretrained(
            wrapped_cls: Any,
            pretrained_model_name_or_path: Any,
            *args: Any,
            _original_from_pretrained: Any = original_from_pretrained,
            **kwargs: Any,
        ) -> Any:
            cache_key = (
                wrapped_cls.__name__,
                str(pretrained_model_name_or_path),
                repr(sorted((key, repr(value)) for key, value in kwargs.items())),
            )
            if cache_key not in _PRETRAINED_MODEL_CACHE:
                _PRETRAINED_MODEL_CACHE[cache_key] = _original_from_pretrained(
                    wrapped_cls,
                    pretrained_model_name_or_path,
                    *args,
                    **kwargs,
                )
            else:
                print(f"Reusing {wrapped_cls.__name__}: {pretrained_model_name_or_path}")
            return _PRETRAINED_MODEL_CACHE[cache_key]

        cached_from_pretrained._soundsketcher_from_pretrained_cache = True  # type: ignore[attr-defined]
        cls.from_pretrained = classmethod(cached_from_pretrained)


def _legacy_objectifier_module_path(settings: Settings) -> Path | None:
    if settings.legacy_objectifier_module_path:
        return settings.legacy_objectifier_module_path

    candidates = (
        settings.project_root.parent
        / "legacy_reference"
        / "soundsketcher_aux_scripts"
        / "clustering_objects_with_wav2vec.py",
        settings.project_root
        / "legacy_reference"
        / "soundsketcher_aux_scripts"
        / "clustering_objects_with_wav2vec.py",
        settings.legacy_root
        / "legacy_reference"
        / "soundsketcher_aux_scripts"
        / "clustering_objects_with_wav2vec.py",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _postprocess_objectifier_payload(payload: dict[str, Any]) -> dict[str, Any]:
    clusters = payload.get("clusters")
    if not isinstance(clusters, list):
        return payload

    next_payload = dict(payload)
    next_payload["clusters"] = _postprocess_objectifier_clusters(clusters)
    general_info = dict(next_payload.get("general_info") or {})
    general_info["region_postprocess"] = {
        "min_region_seconds": MIN_OBJECTIFIER_REGION_SECONDS,
        "coalesce_gap_seconds": REGION_COALESCE_GAP_SECONDS,
    }
    next_payload["general_info"] = general_info
    return next_payload


def _postprocess_objectifier_clusters(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline = _flatten_cluster_regions(clusters)
    if len(timeline) < 2:
        return clusters

    relabeled = _absorb_short_regions(timeline, min_duration=MIN_OBJECTIFIER_REGION_SECONDS)
    coalesced = _coalesce_timeline_regions(
        relabeled,
        gap_tolerance=REGION_COALESCE_GAP_SECONDS,
    )
    if not coalesced:
        return clusters
    return _regroup_timeline_regions(coalesced, clusters)


def _flatten_cluster_regions(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for cluster_index, cluster in enumerate(clusters):
        if not isinstance(cluster, dict):
            continue
        cluster_label = cluster.get("label", cluster_index)
        regions = cluster.get("regions") or []
        if not regions:
            start_time = cluster.get("start_time")
            end_time = cluster.get("end_time")
            if _is_number(start_time) and _is_number(end_time):
                start = float(start_time)
                end = float(end_time)
                if end > start:
                    timeline.append(
                        {
                            "label": _region_label(cluster, cluster_label),
                            "start_time": start,
                            "end_time": end,
                            "cluster_index": cluster_index,
                        }
                    )
            continue
        if not isinstance(regions, list):
            continue
        for region in regions:
            if not isinstance(region, dict):
                continue
            start_time = region.get("start_time")
            end_time = region.get("end_time")
            if not _is_number(start_time) or not _is_number(end_time):
                continue
            start = float(start_time)
            end = float(end_time)
            if end <= start:
                continue
            timeline.append(
                {
                    "label": _region_label(region, cluster_label),
                    "start_time": start,
                    "end_time": end,
                    "cluster_index": cluster_index,
                }
            )
    return sorted(timeline, key=lambda item: (item["start_time"], item["end_time"]))


def _region_label(region: dict[str, Any], fallback: Any) -> Any:
    label = region.get("label", fallback)
    try:
        return int(label)
    except (TypeError, ValueError):
        return label


def _absorb_short_regions(
    timeline: list[dict[str, Any]],
    *,
    min_duration: float,
) -> list[dict[str, Any]]:
    relabeled = [dict(region) for region in timeline]
    for index, region in enumerate(relabeled):
        duration = float(region["end_time"]) - float(region["start_time"])
        if duration >= min_duration:
            continue
        replacement = _neighbor_label_for_short_region(relabeled, index)
        if replacement is not None:
            region["label"] = replacement
    return relabeled


def _neighbor_label_for_short_region(
    timeline: list[dict[str, Any]],
    index: int,
) -> Any | None:
    previous_region = timeline[index - 1] if index > 0 else None
    next_region = timeline[index + 1] if index + 1 < len(timeline) else None
    if previous_region is None and next_region is None:
        return None
    if previous_region is None:
        return next_region["label"]
    if next_region is None:
        return previous_region["label"]
    if previous_region["label"] == next_region["label"]:
        return previous_region["label"]

    previous_duration = float(previous_region["end_time"]) - float(previous_region["start_time"])
    next_duration = float(next_region["end_time"]) - float(next_region["start_time"])
    if previous_duration >= next_duration:
        return previous_region["label"]
    return next_region["label"]


def _coalesce_timeline_regions(
    timeline: list[dict[str, Any]],
    *,
    gap_tolerance: float,
) -> list[dict[str, Any]]:
    coalesced: list[dict[str, Any]] = []
    for region in timeline:
        if (
            coalesced
            and coalesced[-1]["label"] == region["label"]
            and float(region["start_time"]) - float(coalesced[-1]["end_time"]) <= gap_tolerance
        ):
            coalesced[-1]["end_time"] = max(
                float(coalesced[-1]["end_time"]),
                float(region["end_time"]),
            )
        else:
            coalesced.append(dict(region))
    return coalesced


def _regroup_timeline_regions(
    timeline: list[dict[str, Any]],
    original_clusters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metadata_by_label: dict[Any, dict[str, Any]] = {}
    label_order: list[Any] = []
    for cluster_index, cluster in enumerate(original_clusters):
        if not isinstance(cluster, dict):
            continue
        label = _region_label(cluster, cluster_index)
        if label in metadata_by_label:
            continue
        label_order.append(label)
        metadata_by_label[label] = {
            key: value
            for key, value in cluster.items()
            if key not in {"id", "start_time", "end_time", "regions"}
        }

    grouped: dict[Any, list[dict[str, Any]]] = {label: [] for label in label_order}
    for region in timeline:
        label = region["label"]
        if label not in grouped:
            grouped[label] = []
            label_order.append(label)
            metadata_by_label[label] = {"label": label}
        grouped[label].append(region)

    clusters: list[dict[str, Any]] = []
    for cluster_index, label in enumerate(label_order):
        regions = grouped.get(label) or []
        if not regions:
            continue
        cluster_id = cluster_index + 1
        metadata = dict(metadata_by_label.get(label) or {})
        metadata["label"] = label
        metadata.setdefault("color", OBJECTIFIER_COLORS[cluster_index % len(OBJECTIFIER_COLORS)])
        clusters.append(
            {
                "id": cluster_id,
                **metadata,
                "start_time": min(float(region["start_time"]) for region in regions),
                "end_time": max(float(region["end_time"]) for region in regions),
                "regions": [
                    {
                        "id": f"{cluster_id}.{region_index + 1}",
                        "label": label,
                        "start_time": float(region["start_time"]),
                        "end_time": float(region["end_time"]),
                    }
                    for region_index, region in enumerate(regions)
                ],
            }
        )
    return clusters


def _sorted_feature_frames(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frames = [
        feature
        for feature in features
        if isinstance(feature, dict) and _is_number(feature.get("timestamp"))
    ]
    return sorted(frames, key=lambda feature: float(feature["timestamp"]))


def _cluster_feature_frames(
    frames: list[dict[str, Any]],
    *,
    max_regions: int,
) -> list[int]:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    matrix = np.array(
        [
            [_finite_float(frame.get(column)) for column in FEATURE_COLUMNS]
            for frame in frames
        ],
        dtype=float,
    )
    if matrix.shape[0] < 4 or matrix.shape[1] == 0:
        return []

    active_columns = np.nanstd(matrix, axis=0) > 1e-9
    if not np.any(active_columns):
        return []

    matrix = matrix[:, active_columns]
    matrix = StandardScaler().fit_transform(matrix)
    frame_count = len(frames)
    unique_frame_count = len(np.unique(matrix, axis=0))
    max_clusters = min(max_regions, max(2, frame_count // 8), frame_count - 1, unique_frame_count)
    if max_clusters < 2:
        return []

    best_labels: list[int] | None = None
    best_score = -math.inf
    for cluster_count in range(2, max_clusters + 1):
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        labels = model.fit_predict(matrix)
        unique_labels = set(int(label) for label in labels)
        if len(unique_labels) < 2 or len(unique_labels) >= frame_count:
            continue
        try:
            score = float(silhouette_score(matrix, labels))
        except Exception:
            score = -math.inf
        if score > best_score:
            best_score = score
            best_labels = [int(label) for label in labels]

    if best_labels is not None:
        return best_labels

    model = KMeans(n_clusters=2, random_state=42, n_init=10)
    return [int(label) for label in model.fit_predict(matrix)]


def _smooth_labels(labels: list[int], *, window_size: int = 5) -> list[int]:
    if len(labels) < 3:
        return labels

    half_window = max(1, window_size // 2)
    smoothed: list[int] = []
    for index, label in enumerate(labels):
        window = labels[
            max(0, index - half_window) : min(len(labels), index + half_window + 1)
        ]
        counts = {candidate: window.count(candidate) for candidate in set(window)}
        smoothed.append(max(counts, key=lambda candidate: (counts[candidate], -candidate)))
    return smoothed


def _build_regions(
    frames: list[dict[str, Any]],
    labels: list[int],
) -> list[dict[str, Any]]:
    if not frames or len(frames) != len(labels):
        return []

    timestamps = [float(frame["timestamp"]) for frame in frames]
    frame_step = _median_positive_delta(timestamps)
    regions: list[dict[str, Any]] = []
    run_start = 0

    for index in range(1, len(labels) + 1):
        if index < len(labels) and labels[index] == labels[run_start]:
            continue

        region_start = timestamps[run_start]
        if index < len(labels):
            region_end = timestamps[index]
        else:
            region_end = timestamps[index - 1] + frame_step

        if region_end > region_start:
            regions.append(
                {
                    "label": int(labels[run_start]),
                    "start_time": region_start,
                    "end_time": region_end,
                }
            )
        run_start = index

    return regions


def _group_regions_into_clusters(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered_labels: list[int] = []
    grouped_regions: dict[int, list[dict[str, Any]]] = {}
    for region in regions:
        label = int(region["label"])
        if label not in grouped_regions:
            ordered_labels.append(label)
            grouped_regions[label] = []
        grouped_regions[label].append(region)

    clusters: list[dict[str, Any]] = []
    for cluster_index, label in enumerate(ordered_labels):
        cluster_regions = grouped_regions[label]
        cluster_id = cluster_index + 1
        clusters.append(
            {
                "id": cluster_id,
                "label": label,
                "color": OBJECTIFIER_COLORS[cluster_index % len(OBJECTIFIER_COLORS)],
                "start_time": min(float(region["start_time"]) for region in cluster_regions),
                "end_time": max(float(region["end_time"]) for region in cluster_regions),
                "regions": [
                    {
                        "id": f"{cluster_id}.{region_index + 1}",
                        "label": label,
                        "start_time": float(region["start_time"]),
                        "end_time": float(region["end_time"]),
                    }
                    for region_index, region in enumerate(cluster_regions)
                ],
            }
        )

    return clusters


def _build_time_region_clusters(
    frames: list[dict[str, Any]],
    *,
    max_regions: int,
) -> list[dict[str, Any]]:
    if len(frames) < 2:
        return []

    timestamps = [
        float(feature["timestamp"])
        for feature in frames
        if _is_number(feature.get("timestamp"))
    ]
    if len(timestamps) < 2:
        return []

    start_time = min(timestamps)
    end_time = max(timestamps)
    if end_time <= start_time:
        return []

    region_count = min(max_regions, max(1, len(frames) // 12))
    duration = end_time - start_time
    clusters: list[dict[str, Any]] = []

    for index in range(region_count):
        region_start = start_time + duration * index / region_count
        region_end = start_time + duration * (index + 1) / region_count
        if index == region_count - 1:
            region_end = end_time

        clusters.append(
            {
                "id": index + 1,
                "label": index,
                "color": OBJECTIFIER_COLORS[index % len(OBJECTIFIER_COLORS)],
                "start_time": region_start,
                "end_time": region_end,
                "regions": [
                    {
                        "id": f"{index + 1}.1",
                        "label": index,
                        "start_time": region_start,
                        "end_time": region_end,
                    }
                ],
            }
        )

    return clusters


def _median_positive_delta(values: list[float]) -> float:
    deltas = [
        values[index + 1] - values[index]
        for index in range(len(values) - 1)
        if values[index + 1] > values[index]
    ]
    if not deltas:
        return 0.01
    deltas.sort()
    middle = len(deltas) // 2
    if len(deltas) % 2:
        return deltas[middle]
    return (deltas[middle - 1] + deltas[middle]) / 2


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _finite_float(value: Any) -> float:
    if _is_number(value):
        return float(value)
    return 0.0


def _json_safe_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_payload(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe_payload(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe_payload(value.tolist())
    if hasattr(value, "item"):
        return _json_safe_payload(value.item())
    if _is_number(value):
        return float(value) if isinstance(value, float) else int(value)
    return value


def write_objectifier_json(
    audio_path: Path,
    payload_or_clusters: dict[str, Any] | list[dict[str, Any]],
) -> Path:
    output_path = audio_path.parent / "objectifier.json"
    if isinstance(payload_or_clusters, dict):
        payload = payload_or_clusters
    else:
        clusters = _add_v2_cluster_fields(payload_or_clusters)
        payload = _build_v2_payload(
            audio_path.name,
            audio_path.parent.name,
            clusters,
            extractor="sklearn_feature_kmeans_v1",
        )
    output_path.write_text(
        json.dumps(_json_safe_payload(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path
