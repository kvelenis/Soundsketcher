#!/usr/bin/env python3
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from app.core.config import get_settings


REQUIRED_PANEL_MARKERS = [
    'id="granular-engine-header"',
    'id="waveformCanvas"',
    'id="fileInput"',
    'id="granularSampleStatus"',
    'id="useCurrentAudioGranulator"',
    'id="playGranulator"',
    'data-engine="granular"',
]


class ScriptParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        if tag != "script":
            return
        attrs_dict = dict(attrs)
        source = attrs_dict.get("src")
        if source:
            self.scripts.append(source)


def get_text(base_url: str, path: str) -> tuple[int, str]:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=10) as response:
        return response.status, response.read().decode("utf-8")


def post_form(base_url: str, path: str, fields: dict[str, str], timeout: int = 10):
    data = urllib.parse.urlencode(fields).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"content-type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body) if body else None


def assert_script_order(html: str) -> None:
    parser = ScriptParser()
    parser.feed(html)
    script_paths = [urllib.parse.urlparse(src).path for src in parser.scripts]

    stale_names = [
        "/sonificators.js",
        "/sonification-config.js",
        "/sonification-state.js",
        "/granular-helpers.js",
        "/sonification-data-prep.js",
        "/base-synth.js",
        "/oscillator-synth.js",
        "/granular-synth.js",
        "/oscillator-controller.js",
        "/granular-controller.js",
        "/sonification-playback-controller.js",
    ]
    stale = [path for path in script_paths if any(path.endswith(stale_name) for stale_name in stale_names)]
    if stale:
        raise AssertionError(f"stale sonification script still loaded: {stale}")


def assert_static_contracts() -> None:
    base_synth = SANDBOX_ROOT / "static/js/base-synth.module.mjs"
    sonification_utils = SANDBOX_ROOT / "static/js/sonification-utils.module.mjs"
    granular_helpers = SANDBOX_ROOT / "static/js/granular-helpers.module.mjs"
    sonification_data_prep = SANDBOX_ROOT / "static/js/sonification-data-prep.module.mjs"
    granular_synth = SANDBOX_ROOT / "static/js/granular-synth.module.mjs"
    granular_controller = SANDBOX_ROOT / "static/js/granular-controller.module.mjs"
    orchestrator = SANDBOX_ROOT / "static/js/sonification-engine-orchestrator.module.mjs"
    main_module = SANDBOX_ROOT / "static/js/main.module.mjs"

    base_synth_text = base_synth.read_text(encoding="utf-8")
    if 'import { dBToGain } from "./sonification-utils.module.mjs' not in base_synth_text:
        raise AssertionError("BaseSynth module no longer imports dBToGain from sonification utils")
    if "export class BaseSynth" not in base_synth_text:
        raise AssertionError("BaseSynth is no longer exported from its module")
    if "window.SoundSketcher.BaseSynth = BaseSynth" not in base_synth_text:
        raise AssertionError("BaseSynth module is not exposed on SoundSketcher")

    sonification_utils_text = sonification_utils.read_text(encoding="utf-8")
    if "window.SoundSketcher.sonificationUtils = {" not in sonification_utils_text:
        raise AssertionError("sonification utils module is not exposed on SoundSketcher")

    granular_helpers_text = granular_helpers.read_text(encoding="utf-8")
    if "export function processBuffer" not in granular_helpers_text:
        raise AssertionError("granular helpers module no longer exports processBuffer")
    if "export function drawWaveform" not in granular_helpers_text:
        raise AssertionError("granular helpers module no longer exports drawWaveform")
    if "window.SoundSketcher.granularHelpers = {" not in granular_helpers_text:
        raise AssertionError("granular helpers module is not exposed on SoundSketcher")

    sonification_data_prep_text = sonification_data_prep.read_text(encoding="utf-8")
    if "export function prepareSynthData" not in sonification_data_prep_text:
        raise AssertionError("sonification data prep module no longer exports prepareSynthData")
    if "window.prepareSynthData = prepareSynthData" not in sonification_data_prep_text:
        raise AssertionError("sonification data prep module no longer exposes the legacy prepareSynthData global")

    granular_synth_text = granular_synth.read_text(encoding="utf-8")
    if 'import { BaseSynth } from "./base-synth.module.mjs' not in granular_synth_text:
        raise AssertionError("GranularSynth module no longer imports BaseSynth")
    if 'import { clamp } from "./sonification-utils.module.mjs' not in granular_synth_text:
        raise AssertionError("GranularSynth module no longer imports clamp from sonification utils")
    if "export class GranularSynth extends BaseSynth" not in granular_synth_text:
        raise AssertionError("GranularSynth no longer extends BaseSynth")
    if "changeBuffer(buffer)" not in granular_synth_text:
        raise AssertionError("GranularSynth is missing changeBuffer(buffer)")
    if "window.SoundSketcher.GranularSynth = GranularSynth" not in granular_synth_text:
        raise AssertionError("GranularSynth is not exposed on SoundSketcher")

    granular_controller_text = granular_controller.read_text(encoding="utf-8")
    if 'import { drawWaveform, processBuffer } from "./granular-helpers.module.mjs' not in granular_controller_text:
        raise AssertionError("granular controller module no longer imports granular helpers")
    if 'import { sonificationConfig } from "./sonification-config.module.mjs' not in granular_controller_text:
        raise AssertionError("granular controller module no longer imports sonification config")
    if 'import { sonificationState } from "./sonification-state.module.mjs' not in granular_controller_text:
        raise AssertionError("granular controller module no longer imports sonification state")
    if 'window.SoundSketcherApp.onReady("granular engine controls"' not in granular_controller_text:
        raise AssertionError("granular controller is not registered with SoundSketcherApp")
    if 'sonificationState.engineMap["granular"].changeBuffer(sampleBuffer)' not in granular_controller_text:
        raise AssertionError("granular controller no longer forwards uploaded buffers to the engine")
    if 'useCurrentAudioButton.addEventListener("click", useCurrentAudio)' not in granular_controller_text:
        raise AssertionError("granular controller is missing the current-audio sample load command")
    if 'container.dataset.loaded = "true"' not in granular_controller_text:
        raise AssertionError("granular controller no longer marks the waveform container as loaded")
    if 'setLoadedBuffer(buffer, "Current audio loaded")' not in granular_controller_text:
        raise AssertionError("granular controller is missing the current-audio loaded status")
    if 'setCurrentAudioButtonState("loading")' not in granular_controller_text:
        raise AssertionError("granular controller is missing current-audio busy button state")

    orchestrator_text = orchestrator.read_text(encoding="utf-8")
    if 'import { GranularSynth } from "./granular-synth.module.mjs' not in orchestrator_text:
        raise AssertionError("orchestrator module no longer imports GranularSynth")
    if "new GranularSynth" not in orchestrator_text:
        raise AssertionError("orchestrator does not instantiate GranularSynth")
    if "granular: granulator" not in orchestrator_text:
        raise AssertionError("orchestrator does not register granular engine in engineMap")

    main_module_text = main_module.read_text(encoding="utf-8")
    if "sonification-config.module.mjs" not in main_module_text:
        raise AssertionError("main.module.mjs does not import the sonification config module")
    if "sonification-state.module.mjs" not in main_module_text:
        raise AssertionError("main.module.mjs does not import the sonification state module")
    if "sonification-utils.module.mjs" not in main_module_text:
        raise AssertionError("main.module.mjs does not import the sonification utils module")
    if "granular-helpers.module.mjs" not in main_module_text:
        raise AssertionError("main.module.mjs does not import the granular helpers module")
    if "sonification-data-prep.module.mjs" not in main_module_text:
        raise AssertionError("main.module.mjs does not import the sonification data prep module")
    if "sonification-engine-orchestrator.module.mjs" not in main_module_text:
        raise AssertionError("main.module.mjs does not import the sonification engine orchestrator module")
    if "granular-controller.module.mjs" not in main_module_text:
        raise AssertionError("main.module.mjs does not import the granular controller module")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    args = parser.parse_args()
    settings = get_settings()

    status, html = get_text(args.base_url, "/")
    if status != 200:
        raise AssertionError(f"unexpected root status: {status}")

    for marker in REQUIRED_PANEL_MARKERS:
        if marker not in html:
            raise AssertionError(f"missing granular panel marker: {marker}")
    print("granular panel markers -> ok")

    assert_script_order(html)
    print("granular script order -> ok")

    assert_static_contracts()
    print("granular static contracts -> ok")

    status, loaded = post_form(
        args.base_url,
        "/load_cached_audio",
        {
            "filename": settings.preferred_example_filename,
            "hash": settings.preferred_example_hash,
        },
    )
    if status != 200 or loaded.get("files_processed") != 1:
        raise AssertionError(f"unexpected cached fixture response: {loaded}")
    features = loaded["data"][0].get("features")
    if not isinstance(features, list) or len(features) < 700:
        raise AssertionError(f"cached fixture features look wrong: {len(features or [])}")
    print(f"granular fixture baseline -> ok ({len(features)} frames)")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
