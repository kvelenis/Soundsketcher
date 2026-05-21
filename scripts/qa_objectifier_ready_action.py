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


def run_ready_action_fixture(base_url: str) -> dict:
    settings = get_settings()
    node_script = f"""
const {{ chromium }} = require("playwright");

(async () => {{
  const browser = await chromium.launch({{ headless: true }});
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {{
    if (message.type() === "error") errors.push(message.text());
  }});

  await page.goto({json.dumps(base_url.rstrip('/') + '/')}, {{ waitUntil: "load" }});
  await page.click("#examplesToggle");
  await page.waitForSelector("#examplesList button", {{ timeout: 10000 }});
  await page.getByText({json.dumps(settings.preferred_example_filename)}, {{ exact: true }}).click();
  await page.waitForSelector("#audio-path-0 path", {{ timeout: 30000 }});

  await page.waitForFunction(() => {{
    const status = document.getElementById("objectifierStatus");
    const button = document.getElementById("objectifierReadyButton");
    return status?.dataset?.state === "available" && button && !button.hidden;
  }}, null, {{ timeout: 10000 }});

  const before = await page.evaluate(() => ({{
    statusState: document.getElementById("objectifierStatus")?.dataset?.state || null,
    buttonHidden: Boolean(document.getElementById("objectifierReadyButton")?.hidden),
    objectifierChecked: Boolean(document.getElementById("objectifierMode")?.checked),
  }}));

  await page.evaluate(() => document.getElementById("objectifierReadyButton")?.click());
  await page.waitForFunction(() => {{
    return window.__soundSketcherLastDrawProfile?.meta?.mode === "objectifier";
  }}, null, {{ timeout: 10000 }});

  const after = await page.evaluate(() => ({{
    statusState: document.getElementById("objectifierStatus")?.dataset?.state || null,
    buttonHidden: Boolean(document.getElementById("objectifierReadyButton")?.hidden),
    objectifierChecked: Boolean(document.getElementById("objectifierMode")?.checked),
    profileMode: window.__soundSketcherLastDrawProfile?.meta?.mode || null,
    pathCount: document.querySelectorAll("#audio-path-0 path").length,
  }}));

  await browser.close();
  console.log(JSON.stringify({{ before, after, errors }}));
}})().catch((error) => {{
  console.error(error);
  process.exit(1);
}});
"""
    with tempfile.NamedTemporaryFile("w", suffix=".cjs", dir=SANDBOX_ROOT, delete=False) as script:
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
        details = "\n".join(
            part for part in [completed.stdout.strip(), completed.stderr.strip()] if part
        )
        raise AssertionError(f"objectifier ready-action fixture failed: {details}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    args = parser.parse_args()

    result = run_ready_action_fixture(args.base_url)
    if result["errors"]:
        raise AssertionError(f"browser ready-action errors: {result['errors']}")
    before = result["before"]
    after = result["after"]
    if before["statusState"] != "available" or before["buttonHidden"]:
        raise AssertionError(f"ready action was not offered: {result}")
    if before["objectifierChecked"]:
        raise AssertionError(f"objectifier mode should start unchecked: {result}")
    if not after["objectifierChecked"] or after["profileMode"] != "objectifier":
        raise AssertionError(f"ready action did not switch to objectifier view: {result}")
    if not after["buttonHidden"]:
        raise AssertionError(f"ready button should hide after objectifier mode is active: {result}")
    if after["pathCount"] <= 0:
        raise AssertionError(f"objectifier view did not render paths: {result}")

    print(f"objectifier ready-action browser fixture -> ok ({after['pathCount']} paths)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        raise SystemExit(1)
