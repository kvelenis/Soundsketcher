#!/usr/bin/env python3
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from app.core.config import get_settings


REQUIRED_SCRIPT_ORDER = [
    "js/drawing-axis.js",
    "js/drawing-patterns.js",
    "js/objectifier.js",
    "js/drawing-controls.js",
    "js/drawing-features.js",
    "js/drawing-renderers.js",
    "js/sonification-bridge.js",
    "js/drawing-context.js",
    "js/playback-visuals.js",
    "js/drawing-profiler.js",
    "js/drawVisualisation.js",
]

REQUIRED_STATIC_CONTRACTS = {
    "static/js/drawVisualisation.js": [
        'from "./drawing-visualization.module.mjs',
        "window.SoundSketcher.drawingVisualization",
        "window.drawVisualization = drawVisualization",
    ],
    "static/js/drawing-visualization.module.mjs": [
        "export function drawVisualization()",
        "window.createDrawingProfiler(",
        "window.readDrawingControlState()",
        "window.globalAudioData",
        "buildDrawingRunContext(",
        "processDrawingAudioFile(",
        "completeDrawingRunFromContext(",
    ],
    "static/js/drawing-context.js": [
        'from "./drawing-context.module.mjs',
        "window.SoundSketcher.drawingContext",
        "Object.assign(window, drawingContext)",
    ],
    "static/js/objectifier.js": [
        'from "./objectifier.module.mjs',
        "window.SoundSketcher.objectifierRenderer",
        "Object.assign(window, objectifierRenderer)",
        "drawClusterOverlays",
    ],
    "static/js/objectifier.module.mjs": [
        "export function drawClusterOverlays(",
        "assignFeaturesToRegions(clusters,features)",
        "computeVisualDataForRegions(clusters,maxDuration,canvasWidth)",
        "function assignFeaturesToRegions(",
        "function computeVisualDataForRegions(",
        "drawVagueClusterBlob(",
        "drawSubregionGestures(",
    ],
    "static/js/drawing-context.module.mjs": [
        'from "./drawing-file-context.module.mjs',
        'from "./drawing-run-context.module.mjs',
        'from "./drawing-frame-renderer.module.mjs',
        'from "./drawing-file-finalizer.module.mjs',
        'from "./drawing-feature-frame.module.mjs',
        'from "./drawing-audio-file.module.mjs',
        "processDrawingAudioFile",
    ],
    "static/js/drawing-audio-file.module.mjs": [
        'from "./drawing-file-context.module.mjs',
        'from "./drawing-file-finalizer.module.mjs',
        'from "./drawing-feature-frame.module.mjs',
        "export function processDrawingAudioFile(",
        "buildDrawingFileContext(fileIndex",
        "buildVisualFrameOptions({",
        "processDrawingFeatureFrame({",
        "finalizeDrawingFile({",
        "visibleFeatureNames",
    ],
    "static/js/drawing-feature-frame.module.mjs": [
        'from "./drawing-features.module.mjs',
        'from "./drawing-frame-renderer.module.mjs',
        "export function applyVisualFrameToFeature(",
        "export function processDrawingFeatureFrame(",
        'feature["visual"] = {',
        "buildVisualFrame(feature,featureConfig,visualFrameOptions)",
        "renderEligibleVisualFrame({",
    ],
    "static/js/drawing-frame-renderer.module.mjs": [
        'from "./drawing-features.module.mjs',
        'from "./drawing-renderers.module.mjs',
        'from "./sonification-bridge.module.mjs',
        "export function renderEligibleVisualFrame(",
        "appendSynthPathFrame(pathData,fileIndex,feature,visualFrame,colorHue)",
        "buildFeatureDescription(feature,featureConfig",
        "appendLineSketchFrame(pathGroup,visualFrame",
        "appendPolygonFrame(pathGroup,defs,visualFrame",
    ],
    "static/js/drawing-file-finalizer.module.mjs": [
        "export function appendCompletedPathGroup(",
        "export function renderObjectifierClusters(",
        "export function finalizeDrawingFile(",
        "appendCompletedPathGroup(svgContainer, pathGroup)",
        "drawClusterOverlays(audioData.clusters,audioData.features",
    ],
    "static/js/drawing-run-context.module.mjs": [
        'from "./drawing-axis.module.mjs',
        'from "./drawing-patterns.module.mjs',
        'from "./drawing-features.module.mjs',
        "export function buildDerivedFeatureFuncsForDrawing(",
        "export function readDrawingCanvasContext(",
        "export function buildDrawingRunContext(",
        "export function completeDrawingRun(",
        "export function completeDrawingRunFromContext(",
        "perceivedPitchF0OrSC(",
        "buildFeatureConfigurations(audioFiles",
        "pathData = {}",
        "maxDuration",
        "drawYAxisScale(canvasHeight,minValue,maxValue",
        "prepareSynthData(pathData)",
        "profiler.finish({",
    ],
    "static/js/drawing-file-context.module.mjs": [
        "export function buildDrawingFileContext(",
        "export function buildVisualFrameOptions(",
        "pathData[fileIndex] = []",
        "random_engine(0)",
        "color-slider-${fileIndex}",
        "slider-gate",
    ],
    "static/js/drawing-axis.js": [
        'from "./drawing-axis.module.mjs',
        "window.SoundSketcher.drawingAxis",
        "Object.assign(window, drawingAxis)",
    ],
    "static/js/drawing-axis.module.mjs": [
        "export function drawYAxisScale(",
        "export function drawTick(",
    ],
    "static/js/drawing-controls.js": [
        "function readDrawingControlState()",
    ],
    "static/js/drawing-features.js": [
        'from "./drawing-features.module.mjs',
        "window.SoundSketcher.drawingFeatures",
        "Object.assign(window, drawingFeatures)",
    ],
    "static/js/drawing-features.module.mjs": [
        "export function buildFeatureConfigurations(",
        "export function buildVisualFrame(",
        "export function buildFeatureDescription(",
    ],
    "static/js/drawing-renderers.js": [
        'from "./drawing-renderers.module.mjs',
        "window.SoundSketcher.drawingRenderers",
        "Object.assign(window, drawingRenderers)",
    ],
    "static/js/drawing-renderers.module.mjs": [
        'from "./drawing-patterns.module.mjs',
        "export function appendLineSketchFrame(",
        "export function appendPolygonFrame(",
    ],
    "static/js/drawing-patterns.js": [
        'from "./drawing-patterns.module.mjs',
        "window.SoundSketcher.drawingPatterns",
        "Object.assign(window, drawingPatterns)",
    ],
    "static/js/drawing-patterns.module.mjs": [
        "export function generatePolygonPath(",
        "export function createPattern(",
        "createPattern.counter",
    ],
    "static/js/sonification-bridge.js": [
        'from "./sonification-bridge.module.mjs',
        "window.SoundSketcher.sonificationBridge",
        "Object.assign(window, sonificationBridge)",
    ],
    "static/js/sonification-bridge.module.mjs": [
        "export function appendSynthPathFrame(",
        "export function buildSynthPathEntry(",
    ],
    "static/js/drawing-profiler.js": [
        "function createDrawingProfiler(",
        "window.__soundSketcherLastDrawProfile",
    ],
}


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


def assert_script_order(html: str) -> None:
    parser = ScriptParser()
    parser.feed(html)
    script_paths = [urllib.parse.urlparse(src).path for src in parser.scripts]

    positions = []
    for required in REQUIRED_SCRIPT_ORDER:
        matches = [index for index, path in enumerate(script_paths) if required in path]
        if not matches:
            raise AssertionError(f"missing drawing script: {required}")
        positions.append(matches[0])

    if positions != sorted(positions):
        raise AssertionError(f"drawing script order is wrong: {positions}")


def assert_static_contracts() -> None:
    for relative_path, markers in REQUIRED_STATIC_CONTRACTS.items():
        path = SANDBOX_ROOT / relative_path
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                raise AssertionError(f"{relative_path} is missing marker: {marker}")


def run_browser_fixture(base_url: str) -> dict:
    settings = get_settings()
    repo_root = SANDBOX_ROOT
    node_script = f"""
const {{ chromium }} = require("playwright");

(async () => {{
  const browser = await chromium.launch({{ headless: true }});
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {{
    if (message.type() === "error") {{
      errors.push(message.text());
    }}
  }});

  await page.goto({json.dumps(base_url.rstrip('/') + '/')}, {{ waitUntil: "load" }});
  await page.click("#examplesToggle");
  await page.waitForSelector("#examplesList button", {{ timeout: 10000 }});
  await page.getByText({json.dumps(settings.preferred_example_filename)}, {{ exact: true }}).click();
  await page.waitForSelector("#svgCanvas path", {{ timeout: 30000 }});

  const result = await page.evaluate(() => {{
    const profile = window.__soundSketcherLastDrawProfile;
    const pathData = window.pathData || {{}};
    const firstPath = document.querySelector("#svgCanvas path");
    return {{
      pathCount: document.querySelectorAll("#svgCanvas path").length,
      mixerCount: document.querySelectorAll(".mixer-item").length,
      profileLabel: profile?.label || null,
      profileMode: profile?.meta?.mode || null,
      profileFiles: profile?.meta?.files || null,
      featureFrameCount: profile?.counts?.["feature frames"] || null,
      pathDataChannels: Object.keys(pathData).length,
      firstPathHasFeatures: Boolean(firstPath?.getAttribute("data-features")),
    }};
  }});

  result.errors = errors;
  await browser.close();
  console.log(JSON.stringify(result));
}})().catch((error) => {{
  console.error(error);
  process.exit(1);
}});
"""
    with tempfile.NamedTemporaryFile("w", suffix=".cjs", delete=False) as script:
        script.write(node_script)
        script_path = Path(script.name)

    try:
        completed = subprocess.run(
            ["node", str(script_path)],
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=45,
        )
    finally:
        script_path.unlink(missing_ok=True)

    if completed.returncode != 0:
        details = "\n".join(
            part
            for part in [completed.stdout.strip(), completed.stderr.strip()]
            if part
        )
        raise AssertionError(f"browser fixture command failed: {details}")

    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    parser.add_argument("--skip-browser", action="store_true")
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Render the FastAPI template in-process instead of calling a running server.",
    )
    args = parser.parse_args()

    if args.in_process:
        status, html = get_text_from_app("/")
    else:
        status, html = get_text(args.base_url, "/")
    if status != 200:
        raise AssertionError(f"unexpected root status: {status}")

    assert_script_order(html)
    print("drawing script order -> ok")

    assert_static_contracts()
    print("drawing static contracts -> ok")

    if not args.skip_browser:
        result = run_browser_fixture(args.base_url)
        if result["errors"]:
            raise AssertionError(f"browser drawing errors: {result['errors']}")
        if result["pathCount"] < 500:
            raise AssertionError(f"too few rendered paths: {result}")
        if result["mixerCount"] != 1:
            raise AssertionError(f"unexpected mixer count: {result}")
        if result["profileLabel"] != "drawVisualization":
            raise AssertionError(f"drawing profiler did not run: {result}")
        if result["profileMode"] != "line" or result["profileFiles"] != 1:
            raise AssertionError(f"unexpected drawing profile metadata: {result}")
        if result["featureFrameCount"] < 700:
            raise AssertionError(f"feature frames were not fully processed: {result}")
        if result["pathDataChannels"] != 1:
            raise AssertionError(f"pathData was not populated for one channel: {result}")
        if not result["firstPathHasFeatures"]:
            raise AssertionError(f"rendered path is missing data-features: {result}")
        print(f"drawing browser fixture -> ok ({result['pathCount']} paths, {result['featureFrameCount']} frames)")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
