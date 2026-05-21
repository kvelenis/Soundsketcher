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


def run_control_fixture(base_url: str) -> dict:
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

  async function readLineSnapshot(label) {{
    return await page.evaluate((snapshotLabel) => {{
      const profile = window.__soundSketcherLastDrawProfile;
      const path = document.querySelector("#audio-path-0 path");
      const slider = document.getElementById("slider-1");
      return {{
        label: snapshotLabel,
        pathCount: document.querySelectorAll("#audio-path-0 path").length,
        pathD: path?.getAttribute("d") || null,
        pathStrokeWidth: path?.getAttribute("stroke-width") || null,
        profileMode: profile?.meta?.mode || null,
        featureFrameCount: profile?.counts?.["feature frames"] || null,
        slider1Values: slider?.noUiSlider?.get?.() || null,
      }};
    }}, label);
  }}

  await page.goto({json.dumps(base_url.rstrip('/') + '/')}, {{ waitUntil: "load" }});
  await page.waitForSelector("#examplesToggle", {{ timeout: 10000 }});

  await page.evaluate(() => {{
    const polygonMode = document.getElementById("linePolygonMode");
    const objectifierMode = document.getElementById("objectifierMode");
    const autoResketch = document.getElementById("auto_resketch_button");
    if (!polygonMode) throw new Error("linePolygonMode not found");
    polygonMode.checked = false;
    polygonMode.dispatchEvent(new Event("change", {{ bubbles: true }}));
    if (objectifierMode) {{
      objectifierMode.checked = false;
      objectifierMode.dispatchEvent(new Event("change", {{ bubbles: true }}));
    }}
    if (autoResketch) {{
      autoResketch.checked = false;
      autoResketch.dispatchEvent(new Event("change", {{ bubbles: true }}));
    }}
  }});

  await page.click("#examplesToggle");
  await page.waitForSelector("#examplesList button", {{ timeout: 10000 }});
  await page.getByText({json.dumps(settings.preferred_example_filename)}, {{ exact: true }}).click();
  await page.waitForSelector("#audio-path-0 path", {{ timeout: 30000 }});
  const before = await readLineSnapshot("before");

  await page.evaluate(() => {{
    const slider = document.getElementById("slider-1");
    if (!slider?.noUiSlider) throw new Error("slider-1 noUiSlider not found");
    slider.noUiSlider.set([18, 20]);
    document.getElementById("submitButton")?.click();
  }});

  await page.waitForFunction(
    (previousPathD) => {{
      const path = document.querySelector("#audio-path-0 path");
      return path && path.getAttribute("d") !== previousPathD;
    }},
    before.pathD,
    {{ timeout: 30000 }},
  );
  const after = await readLineSnapshot("after");

  await browser.close();
  console.log(JSON.stringify({{ before, after, errors }}));
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
            timeout=60,
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
        raise AssertionError(f"control fixture command failed: {details}")

    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    args = parser.parse_args()

    result = run_control_fixture(args.base_url)
    before = result["before"]
    after = result["after"]

    if result["errors"]:
        raise AssertionError(f"browser control errors: {result['errors']}")
    if before["profileMode"] != "line" or after["profileMode"] != "line":
        raise AssertionError(f"fixture did not stay in line mode: {result}")
    minimum_rendered_paths = max(50, int((before["featureFrameCount"] or 0) * 0.4))
    if before["pathCount"] < minimum_rendered_paths or after["pathCount"] < minimum_rendered_paths:
        raise AssertionError(f"rendered path count changed unexpectedly: {result}")
    if before["pathCount"] != after["pathCount"]:
        raise AssertionError(f"control redraw should preserve rendered path count: {result}")
    if before["featureFrameCount"] < 700 or after["featureFrameCount"] < 700:
        raise AssertionError(f"feature frames were not fully processed: {result}")
    if before["pathD"] == after["pathD"]:
        raise AssertionError(f"line-length control did not change rendered path geometry: {result}")
    if [round(float(value)) for value in after["slider1Values"]] != [18, 20]:
        raise AssertionError(f"line-length slider did not keep requested values: {result}")

    print(
        "control response browser fixture -> ok "
        f"({before['pathCount']} paths, slider-1 {after['slider1Values']})"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
