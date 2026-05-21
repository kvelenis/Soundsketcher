import hashlib
import json
import re
from pathlib import Path
from typing import Any


def objectifier_user_edits(payload: dict) -> dict:
    """Return the dict where user edits live.

    v1 files store edits at the top level of the payload.
    v2 files store edits under payload["user_edits"].
    Callers can read from or write into the returned dict without caring about
    which version the file is.
    """
    if payload.get("schema_version", 1) >= 2:
        return payload.setdefault("user_edits", {})
    return payload


RECORDING_NAME_PATTERN = re.compile(
    r"^recording_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z\.[a-zA-Z0-9]+$"
)
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}


class CacheStore:
    def __init__(
        self,
        root: Path,
        public_prefix: str = "/user_data",
        preferred_hashes: list[str] | None = None,
    ):
        self.root = root
        self.public_prefix = public_prefix.rstrip("/")
        self.preferred_hashes = set(preferred_hashes or [])

    @staticmethod
    def compute_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def check_file_exists(self, audio_hash: str) -> dict[str, bool]:
        folder = self.cache_folder(audio_hash)
        return {
            "features_exists": (folder / "features.json").exists(),
            "objectifier_exists": (folder / "objectifier.json").exists(),
        }

    def list_cached_files(self) -> list[dict[str, str]]:
        if not self.root.exists():
            return []

        cached_files: list[dict[str, str]] = []
        for folder in sorted(path for path in self.root.iterdir() if path.is_dir()):
            features_path = folder / "features.json"
            if not features_path.is_file():
                continue

            filename = self._discover_audio_filename(folder)
            if not filename or RECORDING_NAME_PATTERN.match(filename):
                continue

            audio_path = folder / filename
            if not audio_path.is_file():
                continue

            cached_files.append(
                {
                    "filename": filename,
                    "hash": folder.name,
                    "audio_url": self.audio_url(folder.name, filename),
                    "is_preferred": folder.name in self.preferred_hashes,
                    "has_objectifier": (folder / "objectifier.json").is_file(),
                }
            )

        cached_files.sort(
            key=lambda item: (
                not item["is_preferred"],
                item["filename"].lower(),
            )
        )
        return cached_files

    def save_upload(self, filename: str, content: bytes) -> tuple[str, str]:
        audio_hash = self.compute_hash(content)
        safe_filename = self.safe_filename(filename)
        folder = self.cache_folder(audio_hash)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / safe_filename).write_bytes(content)
        return audio_hash, safe_filename

    def load_cached_audio(self, audio_hash: str) -> dict[str, Any] | None:
        folder = self.cache_folder(audio_hash)
        features_path = folder / "features.json"
        if not features_path.exists():
            return None

        features_json = self._read_json(features_path)
        if not isinstance(features_json, dict):
            return None

        song = features_json.get("Song", {})
        features = song.get("features_per_timestamp")
        if features is None:
            return None

        result: dict[str, Any] = {"features": features}
        objectifier_path = folder / "objectifier.json"
        if objectifier_path.exists():
            objectifier_json = self._read_json(objectifier_path)
            if isinstance(objectifier_json, dict) and "clusters" in objectifier_json:
                result["clusters"] = objectifier_json["clusters"]
                edits = objectifier_user_edits(objectifier_json)
                if isinstance(edits.get("cluster_labels"), dict):
                    result["cluster_labels"] = edits["cluster_labels"]
                if isinstance(edits.get("deleted_regions"), list):
                    result["deleted_regions"] = edits["deleted_regions"]
                if isinstance(edits.get("region_overrides"), dict):
                    result["region_overrides"] = edits["region_overrides"]
                if isinstance(edits.get("hidden_clusters"), list):
                    result["hidden_clusters"] = edits["hidden_clusters"]
                if "only_cluster" in edits:
                    result["only_cluster"] = edits["only_cluster"]
                if isinstance(edits.get("notes"), str):
                    result["notes"] = edits["notes"]
        return result

    def audio_url(self, audio_hash: str, filename: str) -> str:
        return f"{self.public_prefix}/{self.safe_hash(audio_hash)}/{self.safe_filename(filename)}"

    def cache_folder(self, audio_hash: str) -> Path:
        return self.root / self.safe_hash(audio_hash)

    def _discover_audio_filename(self, folder: Path) -> str | None:
        for metadata_name in ("objectifier.json", "features.json"):
            filename = self._metadata_audio_filename(folder / metadata_name)
            if filename:
                return self.safe_filename(filename)

        audio_files = sorted(
            path.name
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not audio_files:
            return None
        return max(audio_files, key=len)

    def _metadata_audio_filename(self, path: Path) -> str | None:
        if not path.exists():
            return None
        data = self._read_json(path)
        if not isinstance(data, dict):
            return None
        audio_obj = data.get("audio")
        candidates = [
            audio_obj.get("filename") if isinstance(audio_obj, dict) else None,
            data.get("audio_file"),
            data.get("filename"),
            data.get("Song", {}).get("audio_file") if isinstance(data.get("Song"), dict) else None,
            data.get("Song", {}).get("filename") if isinstance(data.get("Song"), dict) else None,
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate:
                return candidate
        return None

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def safe_hash(value: str) -> str:
        safe = "".join(char for char in value if char.isalnum())
        if not safe:
            raise ValueError("Invalid audio hash")
        return safe

    @staticmethod
    def safe_filename(value: str) -> str:
        name = Path(value or "audio.wav").name
        safe = []
        for char in name:
            if char.isalnum() or char in ("-", "_", ".", " "):
                safe.append(char)
            else:
                safe.append("_")
        result = "".join(safe).strip(" .")
        return result or "audio.wav"
