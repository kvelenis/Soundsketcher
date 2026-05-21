#!/usr/bin/env python3
import argparse
from html.parser import HTMLParser
from pathlib import Path
import sys
import urllib.parse
import urllib.request

SANDBOX_ROOT = Path(__file__).resolve().parents[1]

STALE_CLASSIC_BRIDGES = [
    "js/sonificators.js",
    "js/sonification-engine-orchestrator.js",
    "js/sonification-config.js",
    "js/sonification-state.js",
    "js/granular-helpers.js",
    "js/sonification-data-prep.js",
    "js/base-synth.js",
    "js/oscillator-synth.js",
    "js/granular-synth.js",
    "js/oscillator-controller.js",
    "js/granular-controller.js",
    "js/sonification-playback-controller.js",
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
            self.scripts.append(
                {
                    "src": source,
                    "path": urllib.parse.urlparse(source).path,
                    "type": attrs_dict.get("type", ""),
                }
            )


def read_url(base_url: str, path: str) -> str:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=10) as response:
        if response.status != 200:
            raise AssertionError(f"unexpected {path} status: {response.status}")
        return response.read().decode("utf-8")


def parse_scripts(html: str) -> list[dict[str, str]]:
    parser = ScriptParser()
    parser.feed(html)
    return parser.scripts


def assert_module_entrypoint(scripts: list[dict[str, str]]) -> None:
    module_scripts = [script for script in scripts if "js/main.module.mjs" in script["path"]]
    if len(module_scripts) != 1:
        raise AssertionError(f"expected exactly one main.module.mjs script, found {len(module_scripts)}")
    if module_scripts[0]["type"] != "module":
        raise AssertionError("main.module.mjs is not loaded as type=module")

    for stale in STALE_CLASSIC_BRIDGES:
        stale_matches = [script["path"] for script in scripts if stale in script["path"]]
        if stale_matches:
            raise AssertionError(f"stale classic sonification bridge still loaded: {stale_matches}")


def assert_static_module_contracts() -> None:
    main_module = SANDBOX_ROOT / "static/js/main.module.mjs"
    orchestrator = SANDBOX_ROOT / "static/js/sonification-engine-orchestrator.module.mjs"
    stale_files = [
        SANDBOX_ROOT / "static/js/sonification-engine-orchestrator.js",
        SANDBOX_ROOT / "static/js/sonification-config.js",
        SANDBOX_ROOT / "static/js/sonification-state.js",
        SANDBOX_ROOT / "static/js/granular-helpers.js",
        SANDBOX_ROOT / "static/js/sonification-data-prep.js",
        SANDBOX_ROOT / "static/js/base-synth.js",
        SANDBOX_ROOT / "static/js/oscillator-synth.js",
        SANDBOX_ROOT / "static/js/granular-synth.js",
        SANDBOX_ROOT / "static/js/oscillator-controller.js",
        SANDBOX_ROOT / "static/js/granular-controller.js",
        SANDBOX_ROOT / "static/js/sonification-playback-controller.js",
    ]

    for stale_file in stale_files:
        if stale_file.exists():
            raise AssertionError(f"stale {stale_file.name} file still exists")

    main_text = main_module.read_text(encoding="utf-8")
    module_imports = [
        'import "./sonification-config.module.mjs',
        'import "./sonification-state.module.mjs',
        'import "./sonification-utils.module.mjs',
        'import "./granular-helpers.module.mjs',
        'import "./sonification-data-prep.module.mjs',
        'import "./sonification-engine-orchestrator.module.mjs',
        'import "./oscillator-controller.module.mjs',
        'import "./granular-controller.module.mjs',
        'import "./sonification-playback-controller.module.mjs',
    ]
    for module_import in module_imports:
        if module_import not in main_text:
            raise AssertionError(f"main.module.mjs is missing {module_import}")
    if 'import "./granular-helpers.module.mjs' not in main_text:
        raise AssertionError("main.module.mjs does not import the granular helpers module")
    if 'import "./sonification-data-prep.module.mjs' not in main_text:
        raise AssertionError("main.module.mjs does not import the sonification data prep module")
    if 'import "./sonification-engine-orchestrator.module.mjs' not in main_text:
        raise AssertionError("main.module.mjs does not import the sonification orchestrator module")
    if "sonificationOrchestratorLoaded: true" not in main_text:
        raise AssertionError("main.module.mjs no longer exposes sonificationOrchestratorLoaded")
    if "sonificationControllersLoaded: true" not in main_text:
        raise AssertionError("main.module.mjs no longer exposes sonificationControllersLoaded")

    orchestrator_text = orchestrator.read_text(encoding="utf-8")
    if 'import { OscillatorSynth } from "./oscillator-synth.module.mjs' not in orchestrator_text:
        raise AssertionError("orchestrator module does not import OscillatorSynth")
    if 'import { GranularSynth } from "./granular-synth.module.mjs' not in orchestrator_text:
        raise AssertionError("orchestrator module does not import GranularSynth")
    if "initSynths" not in orchestrator_text or "window.SoundSketcher.sonification = {" not in orchestrator_text:
        raise AssertionError("orchestrator module does not preserve initSynths bridge")
    if "new OscillatorSynth" not in orchestrator_text:
        raise AssertionError("orchestrator module no longer instantiates OscillatorSynth")
    if "new GranularSynth" not in orchestrator_text:
        raise AssertionError("orchestrator module no longer instantiates GranularSynth")
    if "sonificationState.engineMap = {" not in orchestrator_text:
        raise AssertionError("orchestrator module no longer writes to the shared engineMap")


def assert_synth_module_exports() -> None:
    helper_exports = {
        "sonification-config.module.mjs": [
            "export const sonificationConfig",
            "root.sonificationConfig = sonificationConfig",
            "createMasterVolumeSliderOptions",
        ],
        "sonification-state.module.mjs": [
            "export const sonificationState",
            "root.sonificationState = sonificationState",
        ],
        "sonification-utils.module.mjs": [
            "export function dBToGain",
            "export function clamp",
            "export function map",
            "window.SoundSketcher.sonificationUtils = {",
        ],
        "granular-helpers.module.mjs": [
            "export function processBuffer",
            "export function drawWaveform",
            "window.SoundSketcher.granularHelpers = {",
        ],
        "sonification-data-prep.module.mjs": [
            "export function prepareSynthData",
            "window.SoundSketcher.sonificationDataPrep = {",
            "window.prepareSynthData = prepareSynthData",
            'import { clamp, getRandomSign, hslToRgb, map, perceivedBrightness } from "./sonification-utils.module.mjs',
        ],
        "oscillator-controller.module.mjs": [
            "export function bindOscillatorEngineControls",
            'window.SoundSketcherApp.onReady("oscillator engine controls"',
            "window.SoundSketcher.oscillatorController = {",
        ],
        "granular-controller.module.mjs": [
            "export function bindGranularEngineControls",
            'window.SoundSketcherApp.onReady("granular engine controls"',
            "window.SoundSketcher.granularController = {",
        ],
        "sonification-playback-controller.module.mjs": [
            "export function bindSonificationPlaybackControls",
            'window.SoundSketcherApp.onReady("sonification playback controls"',
            "window.SoundSketcher.sonificationPlaybackController = {",
        ],
    }

    for filename, markers in helper_exports.items():
        text = (SANDBOX_ROOT / "static/js" / filename).read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                raise AssertionError(f"{filename} is missing {marker}")

    expected_exports = {
        "base-synth.module.mjs": "window.SoundSketcher.BaseSynth = BaseSynth",
        "oscillator-synth.module.mjs": "window.SoundSketcher.OscillatorSynth = OscillatorSynth",
        "granular-synth.module.mjs": "window.SoundSketcher.GranularSynth = GranularSynth",
    }

    for filename, export_marker in expected_exports.items():
        text = (SANDBOX_ROOT / "static/js" / filename).read_text(encoding="utf-8")
        if "export class" not in text:
            raise AssertionError(f"{filename} no longer exports its synth class")
        if export_marker not in text:
            raise AssertionError(f"{filename} no longer exposes {export_marker}")

    base_text = (SANDBOX_ROOT / "static/js/base-synth.module.mjs").read_text(encoding="utf-8")
    if 'import { dBToGain } from "./sonification-utils.module.mjs' not in base_text:
        raise AssertionError("BaseSynth module no longer imports dBToGain from sonification utils")

    oscillator_text = (SANDBOX_ROOT / "static/js/oscillator-synth.module.mjs").read_text(encoding="utf-8")
    if 'import { BaseSynth } from "./base-synth.module.mjs' not in oscillator_text:
        raise AssertionError("OscillatorSynth module no longer imports BaseSynth")
    if "export class OscillatorSynth extends BaseSynth" not in oscillator_text:
        raise AssertionError("OscillatorSynth no longer extends BaseSynth")

    granular_text = (SANDBOX_ROOT / "static/js/granular-synth.module.mjs").read_text(encoding="utf-8")
    if 'import { BaseSynth } from "./base-synth.module.mjs' not in granular_text:
        raise AssertionError("GranularSynth module no longer imports BaseSynth")
    if 'import { clamp } from "./sonification-utils.module.mjs' not in granular_text:
        raise AssertionError("GranularSynth module no longer imports clamp from sonification utils")
    if "export class GranularSynth extends BaseSynth" not in granular_text:
        raise AssertionError("GranularSynth no longer extends BaseSynth")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    args = parser.parse_args()

    html = read_url(args.base_url, "/")
    scripts = parse_scripts(html)

    assert_module_entrypoint(scripts)
    print("sonification module entrypoint -> ok")

    assert_static_module_contracts()
    print("sonification module contracts -> ok")

    assert_synth_module_exports()
    print("synth module exports -> ok")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
