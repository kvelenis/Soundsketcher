#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from app.core.config import get_settings


def run_line_fixture(base_url: str) -> dict:
    settings = get_settings()
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
  await page.waitForSelector("#examplesToggle", {{ timeout: 10000 }});

  const initialState = await page.evaluate(() => {{
    const polygonMode = document.getElementById("linePolygonMode");
    const objectifierMode = document.getElementById("objectifierMode");
    return {{
      hasExamplesToggle: Boolean(document.getElementById("examplesToggle")),
      polygonModeChecked: Boolean(polygonMode?.checked),
      objectifierModeChecked: Boolean(objectifierMode?.checked),
    }};
  }});

  await page.evaluate(() => {{
    const polygonMode = document.getElementById("linePolygonMode");
    const objectifierMode = document.getElementById("objectifierMode");
    if (!polygonMode) throw new Error("linePolygonMode not found");
    polygonMode.checked = false;
    polygonMode.dispatchEvent(new Event("change", {{ bubbles: true }}));
    if (objectifierMode) {{
      objectifierMode.checked = false;
      objectifierMode.dispatchEvent(new Event("change", {{ bubbles: true }}));
    }}
  }});

  await page.click("#examplesToggle");
  await page.waitForSelector("#examplesList button", {{ timeout: 10000 }});
  await page.getByText({json.dumps(settings.preferred_example_filename)}, {{ exact: true }}).click();
  await page.waitForSelector("#audio-path-0 path", {{ timeout: 30000 }});

  const result = await page.evaluate(() => {{
    const profile = window.__soundSketcherLastDrawProfile;
    const pathData = window.pathData || {{}};
    const audioGroup = document.querySelector("#audio-path-0");
    const firstPath = audioGroup?.querySelector("path");
    return {{
      initialState: {{}},
      pathCount: document.querySelectorAll("#audio-path-0 path").length,
      patternCount: document.querySelectorAll("#svgCanvas pattern").length,
      mixerCount: document.querySelectorAll(".mixer-item").length,
      audioGroupCount: document.querySelectorAll("#audio-path-0").length,
      profileLabel: profile?.label || null,
      profileMode: profile?.meta?.mode || null,
      profileFiles: profile?.meta?.files || null,
      featureFrameCount: profile?.counts?.["feature frames"] || null,
      lineRenderCount: profile?.counts?.["line render"] || null,
      polygonRenderCount: profile?.counts?.["polygon render"] || null,
      pathDataChannels: Object.keys(pathData).length,
      firstPathHasFeatures: Boolean(firstPath?.getAttribute("data-features")),
      firstPathFill: firstPath?.getAttribute("fill") || null,
      firstPathStroke: firstPath?.getAttribute("stroke") || null,
    }};
  }});

  result.initialState = initialState;
  result.errors = errors;
  await browser.close();
  console.log(JSON.stringify(result));
}})().catch((error) => {{
  console.error(error);
  process.exit(1);
}});
"""
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".cjs",
        dir=SANDBOX_ROOT,
        delete=False,
    ) as script:
        script.write(node_script)
        script_path = Path(script.name)

    try:
        completed = subprocess.run(
            ["node", str(script_path)],
            cwd=SANDBOX_ROOT,
            text=True,
            capture_output=True,
            timeout=45,
        )
    finally:
        script_path.unlink(missing_ok=True)

    if completed.returncode != 0:
        if "Cannot find module 'playwright'" in completed.stderr:
            raise AssertionError(
                "Playwright is not installed for Node. Run npm install before running "
                "this fixture."
            )
        details = "\n".join(
            part
            for part in [completed.stdout.strip(), completed.stderr.strip()]
            if part
        )
        raise AssertionError(f"line fixture command failed: {details}")

    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    args = parser.parse_args()

    result = run_line_fixture(args.base_url)
    if result["errors"]:
        raise AssertionError(f"browser line errors: {result['errors']}")
    if not result["initialState"]["hasExamplesToggle"]:
        raise AssertionError(f"initial app shell controls are missing: {result}")
    if result["pathCount"] < 300:
        raise AssertionError(f"too few rendered line paths: {result}")
    if result["patternCount"] != 0:
        raise AssertionError(f"line mode should not generate polygon patterns: {result}")
    if result["mixerCount"] != 1:
        raise AssertionError(f"unexpected mixer count: {result}")
    if result["audioGroupCount"] != 1:
        raise AssertionError(f"unexpected audio path group count: {result}")
    if result["profileLabel"] != "drawVisualization":
        raise AssertionError(f"drawing profiler did not run: {result}")
    if result["profileMode"] != "line" or result["profileFiles"] != 1:
        raise AssertionError(f"unexpected line profile metadata: {result}")
    if result["featureFrameCount"] < 700:
        raise AssertionError(f"feature frames were not fully processed: {result}")
    if result["lineRenderCount"] is not None and result["lineRenderCount"] < 300:
        raise AssertionError(f"line render count looks too low: {result}")
    if result["polygonRenderCount"]:
        raise AssertionError(f"polygon renderer ran during line mode: {result}")
    if result["pathDataChannels"] != 1:
        raise AssertionError(f"pathData was not populated for one channel: {result}")
    if not result["firstPathHasFeatures"]:
        raise AssertionError(f"rendered line path is missing data-features: {result}")
    if result["firstPathFill"] != "none" or not result["firstPathStroke"]:
        raise AssertionError(f"rendered line path does not look like a stroke path: {result}")

    print(
        "line drawing browser fixture -> ok "
        f"({result['pathCount']} paths, {result['featureFrameCount']} frames)"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
