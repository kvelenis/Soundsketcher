#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import sys
import urllib.parse
import urllib.request


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))


@dataclass(frozen=True)
class Check:
    label: str
    kind: str
    value: str


REQUIRED_MAIN_PAGE_CONTROLS = [
    Check("loading spinner", "id", "spinner"),
    Check("SVG drawing wrapper", "id", "svgWrapper"),
    Check("drawing tooltip", "id", "tooltip"),
    Check("upload drop zone", "id", "drop_area"),
    Check("hidden upload input", "id", "hiddenFileInput"),
    Check("bottom settings menu", "id", "bottomMenu"),
    Check("settings menu toggle", "id", "toggleMenuButton"),
    Check("examples menu toggle", "id", "examplesToggle"),
    Check("examples dropdown list", "id", "examplesList"),
    Check("header play/stop button", "id", "playStopButtonHeader"),
    Check("record circle button", "id", "recCircleBtn"),
    Check("contact info modal", "id", "contactInfoModal"),
    Check("tutorial modal", "id", "tutorialModal"),
    Check("oscillator engine header", "id", "oscillator-engine-header"),
    Check("oscillator play button", "id", "playOscillator"),
    Check("oscillator volume slider", "id", "osc-slider"),
    Check("granular engine header", "id", "granular-engine-header"),
    Check("granular play button", "id", "playGranulator"),
    Check("granular use-current-audio button", "id", "useCurrentAudioGranulator"),
    Check("granular sample status", "id", "granularSampleStatus"),
    Check("granular waveform canvas", "id", "waveformCanvas"),
    Check("mixer container", "id", "mixer"),
    Check("main play/stop button", "id", "playStopButton"),
    Check("audio feature help modal", "id", "helpAudioFeatureModal"),
    Check("mapping help modal", "id", "helpMappingModal"),
    Check("sonification help modal", "id", "helpSonificatorsModal"),
    Check("clamp feature modal", "id", "clampFeatureModal"),
    Check("advanced settings modal", "id", "advancedSettingsModal"),
    Check("resketch button", "id", "submitButton"),
    Check("randomize features button", "id", "randomizeFeaturesButton"),
    Check("reset button", "id", "resetButton"),
    Check("preset A button", "id", "presetButtonA"),
    Check("preset B button", "id", "presetButtonB"),
    Check("preset C button", "id", "presetButtonC"),
    Check("recalculate button", "id", "recalculate_button"),
    Check("auto-resketch toggle", "id", "auto_resketch_button"),
    Check("scroll mode toggle", "id", "scrollModeToggle"),
    Check("log/linear toggle", "id", "log-linear"),
    Check("mel scale toggle", "id", "mel-scale"),
    Check("threshold circle toggle", "id", "thresholdCircle"),
    Check("join data points toggle", "id", "joindatapoints"),
    Check("polygon mode toggle", "id", "linePolygonMode"),
    Check("objectifier mode toggle", "id", "objectifierMode"),
    Check("objectifier data status", "id", "objectifierStatus"),
    Check("export PNG button", "id", "exportButton"),
    Check("export JSON button", "id", "exportFeaturesButton"),
    Check("feature select 1", "id", "featureSelect-1"),
    Check("feature select 2", "id", "featureSelect-2"),
    Check("feature select 3", "id", "featureSelect-3"),
    Check("feature select 4", "id", "featureSelect-4"),
    Check("feature select 5", "id", "featureSelect-5"),
    Check("feature select 6", "id", "featureSelect-6"),
    Check("feature select 7", "id", "featureSelect-7"),
    Check("feature range slider 1", "id", "slider-1"),
    Check("feature range slider 2", "id", "slider-2"),
    Check("feature range slider 3", "id", "slider-3"),
    Check("feature range slider 4", "id", "slider-4"),
    Check("feature range slider 5", "id", "slider-5"),
    Check("feature range slider 6", "id", "slider-6"),
    Check("feature range slider 7", "id", "slider-7"),
    Check("hard clamp slider", "id", "slider-clamp"),
    Check("softclip slider", "id", "slider-softclip"),
    Check("loudness gate slider", "id", "slider-gate"),
    Check("F0 period slider", "id", "period-slider"),
    Check("gamma slider", "id", "gamma-slider"),
    Check("spectral-centroid division slider", "id", "scdivision-slider"),
]


REQUIRED_CLASSES = [
    Check("feature select class", "class", "featureSelect"),
    Check("range slider class", "class", "range-slider"),
    Check("modal class", "class", "modal"),
    Check("sonification play button class", "class", "sonificators__play-button"),
    Check("global settings button class", "class", "global-settings__button"),
]


REQUIRED_SCRIPT_REFERENCES = [
    Check("module entrypoint", "script", "js/main.module.mjs"),
    Check("startup registry", "script", "js/app-init.js"),
    Check("state bridge", "script", "js/app-state.js"),
    Check("namespace bridge", "script", "js/soundsketcher-namespace.js"),
    Check("drawing orchestrator bridge", "script", "js/drawVisualisation.js"),
    Check("upload/cache flow", "script", "js/handleDropAudios.js"),
    Check("controls setup", "script", "js/control-setup.js"),
    Check("submit/resketch flow", "script", "js/submitButton.js"),
    Check("objectifier data status flow", "script", "js/objectifier-status.js"),
    Check("recording flow", "script", "js/recordingAudio.js"),
    Check("header behavior", "script", "js/header.js"),
]


ABSENT_LEGACY_SCRIPT_REFERENCES = [
    Check("classic mixer constructor replaced by module", "script", "js/mixerConstructor.js"),
    Check("classic main playback script replaced by module", "script", "js/playStopMainAudioFunctions.js"),
    Check("classic sonificators script replaced by modules", "script", "js/sonificators.js"),
    Check("legacy navbar script replaced by sandbox header", "script", "js/navbar_2.js"),
]


INTENTIONALLY_DEFERRED = [
    "legacy objectifier model stack and plotly object routes",
    "full legacy extractor stack: CREPE, MOSQITO, aubio, sonic-annotator, MATLAB",
    "CLAP / wav2vec / high-level analysis backend processing",
    "experiment dataset ownership cleanup beyond current route fixtures",
]


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: set[str] = set()
        self.classes: set[str] = set()
        self.scripts: list[str] = []
        self.local_assets: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        element_id = attrs_dict.get("id")
        if element_id:
            self.ids.add(element_id)

        for class_name in attrs_dict.get("class", "").split():
            self.classes.add(class_name)

        if tag == "script":
            src = attrs_dict.get("src")
            if src:
                self.scripts.append(src)

        for attribute in ("src", "href"):
            value = attrs_dict.get(attribute)
            if value:
                self.local_assets.append(value)


def get_text(base_url: str, path: str) -> tuple[int, str]:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=10) as response:
        return response.status, response.read().decode("utf-8")


def get_text_from_app(path: str) -> tuple[int, str]:
    from starlette.requests import Request

    from app.core.templates import get_templates
    from app.main import create_app

    app = create_app()
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [],
        "server": ("testserver", 80),
        "scheme": "http",
        "client": ("testclient", 123),
        "root_path": "",
        "app": app,
    }
    request = Request(scope)
    response = get_templates().TemplateResponse(request, "index.html", {"request": request})
    return response.status_code, response.body.decode("utf-8")


def normalized_path(base_url: str, asset: str) -> str:
    resolved = urllib.parse.urljoin(f"{base_url.rstrip('/')}/", asset)
    return urllib.parse.urlparse(resolved).path


def has_script(parser: PageParser, expected_path: str) -> bool:
    return any(expected_path in normalized_path("", script) for script in parser.scripts)


def check_item(parser: PageParser, check: Check) -> bool:
    if check.kind == "id":
        return check.value in parser.ids
    if check.kind == "class":
        return check.value in parser.classes
    if check.kind == "script":
        return has_script(parser, check.value)
    raise ValueError(f"unknown check kind: {check.kind}")


def print_group(title: str, checks: list[tuple[Check, bool]]) -> None:
    print(title)
    for check, present in checks:
        status = "present" if present else "missing"
        print(f"  {status}: {check.label} ({check.kind}={check.value})")


def assert_no_legacy_static_assets(base_url: str, parser: PageParser) -> None:
    offenders = []
    for asset in parser.local_assets:
        parsed = urllib.parse.urlparse(urllib.parse.urljoin(f"{base_url.rstrip('/')}/", asset))
        if parsed.netloc and parsed.netloc != urllib.parse.urlparse(base_url).netloc:
            continue
        if parsed.path.startswith("/static/"):
            offenders.append(parsed.path)
    if offenders:
        raise AssertionError(f"main page still references legacy /static assets: {sorted(set(offenders))}")


def main() -> int:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    argument_parser.add_argument(
        "--in-process",
        action="store_true",
        help="Render the FastAPI app with TestClient instead of calling a running server.",
    )
    args = argument_parser.parse_args()

    if args.in_process:
        status, html = get_text_from_app("/")
    else:
        status, html = get_text(args.base_url, "/")
    if status != 200:
        raise AssertionError(f"unexpected root status: {status}")

    page = PageParser()
    page.feed(html)

    assert_no_legacy_static_assets(args.base_url, page)

    control_results = [(check, check_item(page, check)) for check in REQUIRED_MAIN_PAGE_CONTROLS]
    class_results = [(check, check_item(page, check)) for check in REQUIRED_CLASSES]
    script_results = [(check, check_item(page, check)) for check in REQUIRED_SCRIPT_REFERENCES]
    absent_results = [(check, not check_item(page, check)) for check in ABSENT_LEGACY_SCRIPT_REFERENCES]

    print("legacy parity main page -> ok")
    print_group("required controls", control_results)
    print_group("required classes", class_results)
    print_group("required sandbox scripts", script_results)

    print("replaced legacy scripts")
    for check, absent in absent_results:
        status = "absent" if absent else "still loaded"
        print(f"  {status}: {check.label} ({check.value})")

    print("intentionally deferred")
    for item in INTENTIONALLY_DEFERRED:
        print(f"  deferred: {item}")

    missing = [
        check.label
        for check, present in control_results + class_results + script_results
        if not present
    ]
    still_loaded = [check.label for check, absent in absent_results if not absent]
    if missing or still_loaded:
        details = []
        if missing:
            details.append(f"missing required parity items: {missing}")
        if still_loaded:
            details.append(f"legacy scripts unexpectedly loaded: {still_loaded}")
        raise AssertionError("; ".join(details))

    print("legacy parity QA -> ok")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
