from functools import lru_cache
import os
from pathlib import Path
from pydantic import BaseModel


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREFERRED_EXAMPLE_HASH = (
    "ea4038a20612493495cb675e8d36adeace1727d3d527c6f204cb67037b7633fd"
)


class Settings(BaseModel):
    app_name: str = "SoundSketcher"
    project_root: Path = DEFAULT_PROJECT_ROOT
    legacy_root: Path = DEFAULT_PROJECT_ROOT
    static_dir: Path = DEFAULT_PROJECT_ROOT / "static"
    sandbox_static_dir: Path = DEFAULT_PROJECT_ROOT / "refactor_sandbox" / "static"
    templates_dir: Path = DEFAULT_PROJECT_ROOT / "templates"
    upload_dir: Path = DEFAULT_PROJECT_ROOT / "user_data"
    static_upload_dir: Path = DEFAULT_PROJECT_ROOT / "static_uploads"
    runtime_dir: Path = DEFAULT_PROJECT_ROOT / "refactor_sandbox" / "runtime"
    response_root: Path = DEFAULT_PROJECT_ROOT / "refactor_sandbox" / "runtime" / "responses"
    cache_root: Path = DEFAULT_PROJECT_ROOT / "refactor_sandbox" / "runtime" / "user_data"
    base_path: str = ""
    preferred_example_hash: str = DEFAULT_PREFERRED_EXAMPLE_HASH
    preferred_example_filename: str = "20-37s.wav"

    wav2vec_model_name: str = "facebook/wav2vec2-base-960h"
    objectifier_model_name: str = "facebook/wav2vec2-base"
    objectifier_mode: str = "legacy_fast"
    legacy_objectifier_module_path: Path | None = None
    objectifier_cache_dir: Path | None = None

    clap_checkpoint_path: Path | None = None
    clap_model_name: str = "laion/larger_clap_music_and_speech"

    matlab_toolbox_dir: Path | None = None
    matlab_scripts_dir: Path | None = None
    sonic_annotator_dir: Path | None = None
    worker_count: int = 8


def _env(name: str, default: str) -> str:
    return os.getenv(f"SOUNDSKETCHER_{name}", default)


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(f"SOUNDSKETCHER_{name}")
    if not value:
        return default
    return Path(value).expanduser()


def _env_optional_path(name: str) -> Path | None:
    value = os.getenv(f"SOUNDSKETCHER_{name}")
    if not value:
        return None
    return Path(value).expanduser()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(f"SOUNDSKETCHER_{name}")
    if not value:
        return default
    return int(value)


def _env_base_path(name: str, default: str = "") -> str:
    value = os.getenv(f"SOUNDSKETCHER_{name}", default).strip()
    if not value or value == "/":
        return ""
    return "/" + value.strip("/")


def load_settings_from_env() -> Settings:
    project_root = _env_path("PROJECT_ROOT", DEFAULT_PROJECT_ROOT)
    legacy_root = _env_path("LEGACY_ROOT", project_root)
    runtime_dir = _env_path("RUNTIME_DIR", project_root / "refactor_sandbox" / "runtime")

    return Settings(
        app_name=_env("APP_NAME", "SoundSketcher"),
        project_root=project_root,
        legacy_root=legacy_root,
        static_dir=_env_path("STATIC_DIR", legacy_root / "static"),
        sandbox_static_dir=_env_path("SANDBOX_STATIC_DIR", project_root / "refactor_sandbox" / "static"),
        templates_dir=_env_path("TEMPLATES_DIR", legacy_root / "templates"),
        upload_dir=_env_path("UPLOAD_DIR", legacy_root / "user_data"),
        static_upload_dir=_env_path("STATIC_UPLOAD_DIR", legacy_root / "static_uploads"),
        runtime_dir=runtime_dir,
        response_root=_env_path("RESPONSE_ROOT", runtime_dir / "responses"),
        cache_root=_env_path("CACHE_ROOT", runtime_dir / "user_data"),
        base_path=_env_base_path("BASE_PATH", ""),
        preferred_example_hash=_env("PREFERRED_EXAMPLE_HASH", DEFAULT_PREFERRED_EXAMPLE_HASH),
        preferred_example_filename=_env("PREFERRED_EXAMPLE_FILENAME", "20-37s.wav"),
        wav2vec_model_name=_env("WAV2VEC_MODEL_NAME", "facebook/wav2vec2-base-960h"),
        objectifier_model_name=_env("OBJECTIFIER_MODEL_NAME", "facebook/wav2vec2-base"),
        objectifier_mode=_env("OBJECTIFIER_MODE", "legacy_fast"),
        legacy_objectifier_module_path=_env_optional_path("LEGACY_OBJECTIFIER_MODULE_PATH"),
        objectifier_cache_dir=_env_optional_path("OBJECTIFIER_CACHE_DIR"),
        clap_checkpoint_path=(
            _env_optional_path("CLAP_CHECKPOINT_PATH")
            or Path("/mnt/ssd1/kvelenis/soundsketcher/aux_models/music_speech_audioset_epoch_15_esc_89.98.pt")
        ),
        clap_model_name=_env("CLAP_MODEL_NAME", "laion/larger_clap_music_and_speech"),
        matlab_toolbox_dir=_env_optional_path("MATLAB_TOOLBOX_DIR"),
        matlab_scripts_dir=_env_optional_path("MATLAB_SCRIPTS_DIR"),
        sonic_annotator_dir=_env_optional_path("SONIC_ANNOTATOR_DIR"),
        worker_count=_env_int("WORKER_COUNT", 8),
    )


@lru_cache
def get_settings() -> Settings:
    return load_settings_from_env()
