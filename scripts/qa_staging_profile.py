#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import sys

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from scripts.qa_settings_env import ENV_KEYS


PROFILE_ONLY_KEYS = {
    "SOUNDSKETCHER_HOST",
    "SOUNDSKETCHER_PORT",
    "SOUNDSKETCHER_PYTHON",
}

IGNORED_ENV_FILES = {
    ".env",
    ".env.local",
    ".env.staging",
    ".env.production",
}


ASSIGNMENT_PATTERN = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")


def parse_env_file(path: Path) -> dict[str, str]:
    values = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = ASSIGNMENT_PATTERN.match(stripped)
        if not match:
            raise AssertionError(f"invalid env assignment at {path}:{line_number}: {line}")
        key, value = match.groups()
        values[key] = value.strip().strip('"').strip("'")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-file",
        default=str(SANDBOX_ROOT / ".env.staging.example"),
    )
    args = parser.parse_args()

    env_file = Path(args.env_file)
    if not env_file.exists():
        raise AssertionError(f"staging env profile not found: {env_file}")
    gitignore = SANDBOX_ROOT / ".gitignore"
    ignored_missing = [
        filename
        for filename in sorted(IGNORED_ENV_FILES)
        if filename not in gitignore.read_text().splitlines()
    ]
    if ignored_missing:
        raise AssertionError(f"gitignore is missing real env profiles: {ignored_missing}")

    values = parse_env_file(env_file)
    required_keys = set(ENV_KEYS).union(PROFILE_ONLY_KEYS)
    missing = sorted(required_keys.difference(values))
    if missing:
        raise AssertionError(f"staging env profile missing keys: {missing}")

    empty_required = sorted(
        key
        for key in required_keys.difference(
            {
                "SOUNDSKETCHER_MATLAB_TOOLBOX_DIR",
                "SOUNDSKETCHER_SONIC_ANNOTATOR_DIR",
                "SOUNDSKETCHER_LEGACY_OBJECTIFIER_MODULE_PATH",
                "SOUNDSKETCHER_OBJECTIFIER_CACHE_DIR",
                "SOUNDSKETCHER_PYTHON",
            }
        )
        if values.get(key, "") == ""
    )
    if empty_required:
        raise AssertionError(f"staging env profile has empty required keys: {empty_required}")

    port = values["SOUNDSKETCHER_PORT"]
    if not port.isdigit():
        raise AssertionError(f"SOUNDSKETCHER_PORT must be numeric: {port}")

    if values["SOUNDSKETCHER_PROJECT_ROOT"] == "/srv/soundsketcher":
        print("staging env profile warning -> example paths are placeholders")
    print(f"staging env profile -> ok ({env_file})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
