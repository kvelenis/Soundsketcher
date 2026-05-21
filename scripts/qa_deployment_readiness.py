#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

SANDBOX_ROOT = Path(__file__).resolve().parents[1]


CHECKS = [
    ("drawing static boundary", ["scripts/run_python.sh", "scripts/qa_drawing_boundary.py", "--skip-browser"]),
    ("settings environment overrides", ["scripts/run_python.sh", "scripts/qa_settings_env.py"]),
    ("staging launch profile", ["scripts/run_python.sh", "scripts/qa_staging_profile.py"]),
    ("frontend runtime boundary", ["scripts/run_python.sh", "scripts/qa_frontend_runtime_boundary.py"]),
    ("legacy parity main page", ["scripts/run_python.sh", "scripts/qa_legacy_parity.py"]),
    ("full UI fixture API", ["scripts/run_python.sh", "scripts/qa_full_ui_fixture.py"]),
    ("indefinite pitch showcase", ["scripts/run_python.sh", "scripts/qa_indefinite_pitch_showcase.py"]),
    ("sonification module boundary", ["scripts/run_python.sh", "scripts/qa_sonification_module_boundary.py"]),
    ("line browser fixture", ["scripts/run_python.sh", "scripts/qa_line_drawing_fixture.py"]),
    ("polygon browser fixture", ["scripts/run_python.sh", "scripts/qa_polygon_drawing_fixture.py"]),
    ("objectifier ready browser fixture", ["scripts/run_python.sh", "scripts/qa_objectifier_ready_action.py"]),
    ("feature extraction status browser fixture", ["scripts/run_python.sh", "scripts/qa_feature_extraction_status_fixture.py"]),
    ("control response browser fixture", ["scripts/run_python.sh", "scripts/qa_control_response_fixture.py"]),
]

BASE_URL_CHECK_LABELS = {
    "drawing static boundary",
    "frontend runtime boundary",
    "legacy parity main page",
    "full UI fixture API",
    "indefinite pitch showcase",
    "sonification module boundary",
    "line browser fixture",
    "polygon browser fixture",
    "objectifier ready browser fixture",
    "feature extraction status browser fixture",
    "control response browser fixture",
}


def assert_file_exists(path: str) -> None:
    full_path = SANDBOX_ROOT / path
    if not full_path.exists():
        raise AssertionError(f"missing required file: {path}")


def assert_package_scripts() -> None:
    assert_file_exists("package.json")
    assert_file_exists("package-lock.json")

    package = json.loads((SANDBOX_ROOT / "package.json").read_text())
    scripts = package.get("scripts", {})
    required_scripts = {
        "qa:all",
        "qa:browser",
        "qa:controls",
        "qa:line",
        "qa:polygon",
    }
    missing = sorted(required_scripts.difference(scripts))
    if missing:
        raise AssertionError(f"missing package scripts: {missing}")


def assert_server_reachable(base_url: str) -> None:
    with urllib.request.urlopen(base_url.rstrip("/") + "/", timeout=10) as response:
        if response.status != 200:
            raise AssertionError(f"unexpected root status: {response.status}")


def run_check(label: str, command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=SANDBOX_ROOT,
        text=True,
        capture_output=True,
        timeout=90,
    )
    if completed.returncode != 0:
        details = "\n".join(
            part
            for part in [completed.stdout.strip(), completed.stderr.strip()]
            if part
        )
        raise AssertionError(f"{label} failed:\n{details}")
    print(f"{label} -> ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    args = parser.parse_args()

    assert_package_scripts()
    print("node QA package setup -> ok")
    print(f"python executable -> ok ({sys.executable})")

    assert_server_reachable(args.base_url)
    print(f"sandbox server reachable -> ok ({args.base_url})")

    for label, command in CHECKS:
        run_check(
            label,
            command + ["--base-url", args.base_url]
            if label in BASE_URL_CHECK_LABELS
            else command,
        )

    print("deployment readiness -> ok")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
