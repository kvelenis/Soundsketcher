#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from app.core.config import get_settings  # noqa: E402


def run_missing_status_fixture(base_url: str) -> dict:
    settings = get_settings()
    node_script = f"""
const {{ chromium }} = require("playwright");

(async () => {{
  const browser = await chromium.launch({{ headless: true }});
  const page = await browser.newPage();
  const errors = [];
  const dialogs = [];
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
    if (!objectifierMode) throw new Error("objectifierMode not found");
    polygonMode.checked = false;
    polygonMode.dispatchEvent(new Event("change", {{ bubbles: true }}));
    objectifierMode.checked = false;
    objectifierMode.dispatchEvent(new Event("change", {{ bubbles: true }}));
  }});

  await page.click("#examplesToggle");
  await page.waitForSelector("#examplesList button", {{ timeout: 10000 }});
  await page.getByText({json.dumps(settings.preferred_example_filename)}, {{ exact: true }}).click();
  await page.waitForSelector("#audio-path-0 path", {{ timeout: 30000 }});
  const initialLinePathCount = await page.locator("#audio-path-0 path").count();
  await page.evaluate(() => {{
    const data = window.SoundSketcher?.state?.globalAudioData?.data || [];
    for (const item of data) {{
      item.clusters = null;
      item.objectifier_job = null;
    }}
    window.SoundSketcherObjectifierStatus?.updateObjectifierStatus();
  }});

  await page.evaluate(() => {{
    const objectifierMode = document.getElementById("objectifierMode");
    objectifierMode.checked = true;
    objectifierMode.dispatchEvent(new Event("change", {{ bubbles: true }}));
    document.getElementById("submitButton")?.click();
  }});

  await page.waitForTimeout(500);

  const result = await page.evaluate((initialLinePathCount) => {{
    const status = document.getElementById("objectifierStatus");
    return {{
      statusText: status?.textContent?.trim() || null,
      statusState: status?.dataset?.state || null,
      objectifierChecked: Boolean(document.getElementById("objectifierMode")?.checked),
      hasMissingPopup: Boolean(document.querySelector(".swal2-container")),
      popupTitle: document.querySelector(".swal2-title")?.textContent?.trim() || null,
      initialLinePathCount,
      linePathCountAfterBlockedResketch: document.querySelectorAll("#audio-path-0 path").length,
    }};
  }}, initialLinePathCount);

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
        raise AssertionError(f"objectifier missing-status fixture command failed: {details}")

    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    args = parser.parse_args()

    result = run_missing_status_fixture(args.base_url)
    if result["errors"]:
        raise AssertionError(f"browser missing-status errors: {result['errors']}")
    if result["statusState"] != "missing":
        raise AssertionError(f"objectifier status did not report missing data: {result}")
    if "missing" not in (result["statusText"] or "").lower():
        raise AssertionError(f"objectifier status text is not useful: {result}")
    if not result["hasMissingPopup"] or result["popupTitle"] != "Objectifier data is missing":
        raise AssertionError(f"missing objectifier popup did not appear: {result}")
    if result["linePathCountAfterBlockedResketch"] != result["initialLinePathCount"]:
        raise AssertionError(f"blocked objectifier resketch removed the existing drawing: {result}")

    print("objectifier missing-status browser fixture -> ok")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
