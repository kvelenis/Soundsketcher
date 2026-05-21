#!/usr/bin/env python3
from pathlib import Path
import sys
import tempfile
import textwrap
import types


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from app.core.config import Settings  # noqa: E402
from app.services.objectifier import _load_legacy_objectifier_module  # noqa: E402


LEGACY_STUB = """
def objectifier(audio_path):
    def determine_optimal_clusters(embeddings, max_clusters=20, method="gap"):
        return 19

    def process_audio_with_clap_comparison(audio_file_path, hop_length=320, max_clusters=10):
        embeddings = list(range(900))
        sr = 16000
        y = [0.0] * (sr * 17)
        optimal_clusters = determine_optimal_clusters(embeddings, max_clusters=20, method="gap")
        return {"clusters": [{"label": optimal_clusters, "regions": []}]}

    return process_audio_with_clap_comparison(audio_path)
"""


def main() -> int:
    with tempfile.TemporaryDirectory() as temp_dir:
        module_path = Path(temp_dir) / "legacy_objectifier.py"
        module_path.write_text(textwrap.dedent(LEGACY_STUB), encoding="utf-8")
        settings = Settings(
            legacy_objectifier_module_path=module_path,
            objectifier_mode="legacy_fast",
        )
        module = _load_legacy_objectifier_module(module_path, settings=settings)
        if not isinstance(module, types.ModuleType):
            raise AssertionError("legacy objectifier loader did not return a module")

        payload = module.objectifier("example.wav")
        cluster_count = payload["clusters"][0]["label"]
        if cluster_count != 6:
            raise AssertionError(f"fast mode chose unexpected cluster count: {cluster_count}")

        settings = Settings(
            legacy_objectifier_module_path=module_path,
            objectifier_mode="legacy_full",
        )
        module = _load_legacy_objectifier_module(module_path, settings=settings)
        payload = module.objectifier("example.wav")
        cluster_count = payload["clusters"][0]["label"]
        if cluster_count != 19:
            raise AssertionError(f"full mode should keep legacy cluster search: {cluster_count}")

    print("objectifier fast mode patch -> ok")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
