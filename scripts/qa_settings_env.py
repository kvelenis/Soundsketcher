#!/usr/bin/env python3
import os
from pathlib import Path
import sys
import tempfile

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from app.core.config import DEFAULT_PREFERRED_EXAMPLE_HASH, get_settings


ENV_KEYS = [
    "SOUNDSKETCHER_APP_NAME",
    "SOUNDSKETCHER_PROJECT_ROOT",
    "SOUNDSKETCHER_LEGACY_ROOT",
    "SOUNDSKETCHER_STATIC_DIR",
    "SOUNDSKETCHER_SANDBOX_STATIC_DIR",
    "SOUNDSKETCHER_TEMPLATES_DIR",
    "SOUNDSKETCHER_UPLOAD_DIR",
    "SOUNDSKETCHER_STATIC_UPLOAD_DIR",
    "SOUNDSKETCHER_RUNTIME_DIR",
    "SOUNDSKETCHER_RESPONSE_ROOT",
    "SOUNDSKETCHER_CACHE_ROOT",
    "SOUNDSKETCHER_PREFERRED_EXAMPLE_HASH",
    "SOUNDSKETCHER_PREFERRED_EXAMPLE_FILENAME",
    "SOUNDSKETCHER_WAV2VEC_MODEL_NAME",
    "SOUNDSKETCHER_OBJECTIFIER_MODEL_NAME",
    "SOUNDSKETCHER_OBJECTIFIER_MODE",
    "SOUNDSKETCHER_LEGACY_OBJECTIFIER_MODULE_PATH",
    "SOUNDSKETCHER_OBJECTIFIER_CACHE_DIR",
    "SOUNDSKETCHER_MATLAB_TOOLBOX_DIR",
    "SOUNDSKETCHER_SONIC_ANNOTATOR_DIR",
    "SOUNDSKETCHER_WORKER_COUNT",
]


def restore_env(original: dict[str, str | None]) -> None:
    for key, value in original.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def main() -> int:
    original = {key: os.environ.get(key) for key in ENV_KEYS}
    try:
        for key in ENV_KEYS:
            os.environ.pop(key, None)
        get_settings.cache_clear()
        defaults = get_settings()
        if defaults.preferred_example_hash != DEFAULT_PREFERRED_EXAMPLE_HASH:
            raise AssertionError("default preferred example hash changed")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            os.environ.update(
                {
                    "SOUNDSKETCHER_APP_NAME": "SoundSketcher Staging",
                    "SOUNDSKETCHER_PROJECT_ROOT": str(root / "project"),
                    "SOUNDSKETCHER_LEGACY_ROOT": str(root / "legacy"),
                    "SOUNDSKETCHER_RUNTIME_DIR": str(root / "runtime"),
                    "SOUNDSKETCHER_STATIC_DIR": str(root / "static"),
                    "SOUNDSKETCHER_SANDBOX_STATIC_DIR": str(root / "sandbox_static"),
                    "SOUNDSKETCHER_TEMPLATES_DIR": str(root / "templates"),
                    "SOUNDSKETCHER_UPLOAD_DIR": str(root / "uploads"),
                    "SOUNDSKETCHER_STATIC_UPLOAD_DIR": str(root / "static_uploads"),
                    "SOUNDSKETCHER_RESPONSE_ROOT": str(root / "responses"),
                    "SOUNDSKETCHER_CACHE_ROOT": str(root / "cache"),
                    "SOUNDSKETCHER_PREFERRED_EXAMPLE_HASH": "abc123",
                    "SOUNDSKETCHER_PREFERRED_EXAMPLE_FILENAME": "example.wav",
                    "SOUNDSKETCHER_WAV2VEC_MODEL_NAME": "local/wav2vec",
                    "SOUNDSKETCHER_OBJECTIFIER_MODEL_NAME": "local/objectifier",
                    "SOUNDSKETCHER_OBJECTIFIER_MODE": "legacy_full",
                    "SOUNDSKETCHER_LEGACY_OBJECTIFIER_MODULE_PATH": str(root / "legacy_objectifier.py"),
                    "SOUNDSKETCHER_OBJECTIFIER_CACHE_DIR": str(root / "objectifier_cache"),
                    "SOUNDSKETCHER_MATLAB_TOOLBOX_DIR": str(root / "matlab"),
                    "SOUNDSKETCHER_SONIC_ANNOTATOR_DIR": str(root / "sonic"),
                    "SOUNDSKETCHER_WORKER_COUNT": "3",
                }
            )
            get_settings.cache_clear()
            settings = get_settings()

            expected = {
                "app_name": "SoundSketcher Staging",
                "project_root": root / "project",
                "legacy_root": root / "legacy",
                "runtime_dir": root / "runtime",
                "static_dir": root / "static",
                "sandbox_static_dir": root / "sandbox_static",
                "templates_dir": root / "templates",
                "upload_dir": root / "uploads",
                "static_upload_dir": root / "static_uploads",
                "response_root": root / "responses",
                "cache_root": root / "cache",
                "preferred_example_hash": "abc123",
                "preferred_example_filename": "example.wav",
                "wav2vec_model_name": "local/wav2vec",
                "objectifier_model_name": "local/objectifier",
                "objectifier_mode": "legacy_full",
                "legacy_objectifier_module_path": root / "legacy_objectifier.py",
                "objectifier_cache_dir": root / "objectifier_cache",
                "matlab_toolbox_dir": root / "matlab",
                "sonic_annotator_dir": root / "sonic",
                "worker_count": 3,
            }

            for field, value in expected.items():
                if getattr(settings, field) != value:
                    raise AssertionError(
                        f"{field} override mismatch: {getattr(settings, field)!r} != {value!r}"
                    )
    finally:
        restore_env(original)
        get_settings.cache_clear()

    print("settings environment overrides -> ok")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
