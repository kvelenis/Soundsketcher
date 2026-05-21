from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.services.objectifier import build_objectifier_payload, write_objectifier_json
from app.services.legacy_features import (
    legacy_dependency_report,
    legacy_is_available,
    run_legacy_extraction,
)


@dataclass(frozen=True)
class FeatureExtractionOptions:
    n_fft: int = 2048
    overlap: float = 0.5
    normalize_audio: bool = False
    apply_filter: bool = False
    save_json: bool = False
    run_objectifier: bool = True

    @property
    def hop_length(self) -> int:
        return self.n_fft - round(self.n_fft * self.overlap)


@dataclass(frozen=True)
class FeatureExtractionResult:
    features: list[dict[str, Any]]
    clusters: Any | None = None


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    available: bool
    detail: str


class FeatureExtractionUnavailable(RuntimeError):
    def __init__(self, dependency_report: list[DependencyStatus]):
        self.dependency_report = dependency_report
        missing = [item.name for item in dependency_report if not item.available]
        message = "Feature extraction is not available."
        if missing:
            message = f"{message} Missing: {', '.join(missing)}."
        super().__init__(message)


class FeatureExtractionService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def dependency_report(self) -> list[DependencyStatus]:
        raw = legacy_dependency_report(self.settings)
        return [
            DependencyStatus(
                name=r["name"],
                available=r["available"],
                detail=r.get("detail", "importable" if r["available"] else "not importable"),
            )
            for r in raw
        ]

    def is_available(self) -> bool:
        return legacy_is_available(self.settings)

    def extract(
        self,
        audio_path: Path,
        options: FeatureExtractionOptions,
    ) -> FeatureExtractionResult:
        if not self.is_available():
            raise FeatureExtractionUnavailable(self.dependency_report())

        features = run_legacy_extraction(
            wav_path=audio_path,
            settings=self.settings,
            n_fft=options.n_fft,
            hop_length=max(1, options.hop_length),
            normalize_audio=options.normalize_audio,
        )

        if not features:
            return FeatureExtractionResult(features=[])

        objectifier_payload = (
            build_objectifier_payload(audio_path, features, settings=self.settings)
            if options.run_objectifier
            else None
        )
        clusters = (
            objectifier_payload.get("clusters")
            if isinstance(objectifier_payload, dict)
            else None
        )

        if options.save_json:
            self.write_features_json(audio_path, features)
            if objectifier_payload is not None:
                write_objectifier_json(audio_path, objectifier_payload)

        return FeatureExtractionResult(features=features, clusters=clusters)

    @staticmethod
    def write_features_json(audio_path: Path, features: list[dict[str, Any]]) -> Path:
        output_path = audio_path.parent / "features.json"
        payload = {
            "Song": {
                "audio_file": audio_path.name,
                "features_per_timestamp": features,
                "general_info": {
                    "extractor": "legacy_full_pipeline_v1",
                },
            }
        }
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path


def serialize_dependency_report(report: list[DependencyStatus]) -> list[dict[str, Any]]:
    return [
        {"name": item.name, "available": item.available, "detail": item.detail}
        for item in report
    ]
