#!/usr/bin/env python3
import os
from pathlib import Path
import sys


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from app.services.objectifier import build_objectifier_clusters  # noqa: E402


def make_frame(timestamp: float, centroid: float, loudness: float, roughness: float) -> dict:
    return {
        "timestamp": timestamp,
        "spectral_centroid": centroid,
        "weighted_spectral_centroid": centroid,
        "spectral_bandwidth": centroid / 2,
        "spectral_flatness": roughness,
        "spectral_flux": loudness,
        "f0_librosa": 0.0,
        "raw_periodicity": 0.0,
        "rms": loudness,
        "loudness": loudness,
        "mir_mps_roughness": roughness,
    }


def synthetic_features() -> list[dict]:
    frames: list[dict] = []
    timestamp = 0.0
    for _ in range(16):
        frames.append(make_frame(timestamp, centroid=350.0, loudness=0.08, roughness=0.05))
        timestamp += 0.05
    for _ in range(16):
        frames.append(make_frame(timestamp, centroid=1800.0, loudness=0.42, roughness=0.18))
        timestamp += 0.05
    for _ in range(16):
        frames.append(make_frame(timestamp, centroid=700.0, loudness=0.22, roughness=0.65))
        timestamp += 0.05
    return frames


def main() -> int:
    clusters = build_objectifier_clusters(synthetic_features(), max_regions=5)
    if len(clusters) < 2:
        raise AssertionError(f"expected multiple clusters, got: {clusters}")

    regions = [region for cluster in clusters for region in cluster.get("regions", [])]
    if len(regions) < 3:
        raise AssertionError(f"expected multiple contiguous regions, got: {clusters}")

    labels = {cluster.get("label") for cluster in clusters}
    if len(labels) != len(clusters):
        raise AssertionError(f"cluster labels should be unique: {clusters}")

    for cluster in clusters:
        if cluster["end_time"] <= cluster["start_time"]:
            raise AssertionError(f"cluster has invalid time bounds: {cluster}")
        if not cluster.get("color"):
            raise AssertionError(f"cluster is missing color: {cluster}")
        for region in cluster["regions"]:
            if region["end_time"] <= region["start_time"]:
                raise AssertionError(f"region has invalid time bounds: {region}")

    tiny_clusters = build_objectifier_clusters(synthetic_features()[:2], max_regions=5)
    if not tiny_clusters or not tiny_clusters[0].get("regions"):
        raise AssertionError(f"tiny fallback did not produce a usable region: {tiny_clusters}")

    print(
        "objectifier service clustering -> ok "
        f"({len(clusters)} clusters, {len(regions)} regions)"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
