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


def run_polygon_fixture(base_url: str) -> dict:
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
  await page.evaluate(() => {{
    const polygonMode = document.getElementById("linePolygonMode");
    const objectifierMode = document.getElementById("objectifierMode");
    if (!polygonMode) throw new Error("linePolygonMode not found");
    polygonMode.checked = true;
    polygonMode.dispatchEvent(new Event("change", {{ bubbles: true }}));
    if (objectifierMode) {{
      objectifierMode.checked = false;
      objectifierMode.dispatchEvent(new Event("change", {{ bubbles: true }}));
    }}
  }});

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
      patternCount: document.querySelectorAll("#svgCanvas pattern").length,
      mixerCount: document.querySelectorAll(".mixer-item").length,
      audioGroupCount: document.querySelectorAll("#audio-path-0").length,
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
                "Playwright is not installed for Node. Install the browser QA dependency "
                "before running this fixture."
            )
        details = "\n".join(
            part
            for part in [completed.stdout.strip(), completed.stderr.strip()]
            if part
        )
        raise AssertionError(f"polygon fixture command failed: {details}")

    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    args = parser.parse_args()

    result = run_polygon_fixture(args.base_url)
    if result["errors"]:
        raise AssertionError(f"browser polygon errors: {result['errors']}")
    minimum_rendered_paths = max(50, int((result["featureFrameCount"] or 0) * 0.4))
    if result["pathCount"] < minimum_rendered_paths:
        raise AssertionError(f"too few rendered polygon paths: {result}")
    if result["patternCount"] < minimum_rendered_paths:
        raise AssertionError(f"polygon patterns were not generated: {result}")
    if result["mixerCount"] != 1:
        raise AssertionError(f"unexpected mixer count: {result}")
    if result["audioGroupCount"] != 1:
        raise AssertionError(f"unexpected audio path group count: {result}")
    if result["profileLabel"] != "drawVisualization":
        raise AssertionError(f"drawing profiler did not run: {result}")
    if result["profileMode"] != "polygon" or result["profileFiles"] != 1:
        raise AssertionError(f"unexpected polygon profile metadata: {result}")
    if result["featureFrameCount"] < 700:
        raise AssertionError(f"feature frames were not fully processed: {result}")
    if result["pathDataChannels"] != 1:
        raise AssertionError(f"pathData was not populated for one channel: {result}")
    if not result["firstPathHasFeatures"]:
        raise AssertionError(f"rendered polygon path is missing data-features: {result}")

    print(
        "polygon drawing browser fixture -> ok "
        f"({result['pathCount']} paths, {result['patternCount']} patterns, "
        f"{result['featureFrameCount']} frames)"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
