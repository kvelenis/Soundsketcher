import random
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg"}


class StimulusService:
    def __init__(self, static_dir: Path):
        self.static_dir = static_dir

    def list_audio_files(self, relative_dir: str, shuffled: bool = True) -> list[str]:
        directory = self.static_dir / relative_dir
        if not directory.exists():
            return []

        files = sorted(
            path.name
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )

        if shuffled:
            return random.sample(files, len(files))
        return files

    def noise_tonal_preference_stimuli(self) -> list[dict[str, object]]:
        sample_dir = self.static_dir / "noise_tonal_preference_samples"
        if not sample_dir.exists():
            return []

        stimuli: list[dict[str, object]] = []
        for stimulus_folder in sorted(sample_dir.iterdir()):
            if not stimulus_folder.is_dir():
                continue

            audio_files = sorted(
                path
                for path in stimulus_folder.iterdir()
                if path.suffix.lower() in AUDIO_EXTENSIONS
            )
            if len(audio_files) != 2:
                continue

            stimuli.append(
                {
                    "id": stimulus_folder.name,
                    "stimulus_id": stimulus_folder.name,
                    "audio_filenames": [path.name for path in audio_files],
                    "files": [
                        f"/static/noise_tonal_preference_samples/{stimulus_folder.name}/{path.name}"
                        for path in audio_files
                    ],
                    "audio_urls": [
                        f"/static/noise_tonal_preference_samples/{stimulus_folder.name}/{path.name}"
                        for path in audio_files
                    ],
                }
            )

        random.shuffle(stimuli)
        return stimuli

