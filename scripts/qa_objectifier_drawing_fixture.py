#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from scripts.seed_full_ui_example import EXAMPLE_FILENAME, EXAMPLE_HASH, main as seed_example  # noqa: E402


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


def run_objectifier_fixture(base_url: str) -> dict:
    seed_example()
    status, loaded = post_form(
        base_url,
        "/load_cached_audio",
        {"filename": EXAMPLE_FILENAME, "hash": EXAMPLE_HASH},
    )
    if status != 200:
        raise AssertionError(f"objectifier fixture API failed: {loaded}")
    api_clusters = loaded["data"][0].get("clusters")
    if not api_clusters or not api_clusters[0].get("regions"):
        raise AssertionError(f"objectifier fixture API did not return clusters: {loaded}")

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
    if (!objectifierMode) throw new Error("objectifierMode not found");
    polygonMode.checked = false;
    polygonMode.dispatchEvent(new Event("change", {{ bubbles: true }}));
    objectifierMode.checked = true;
    objectifierMode.dispatchEvent(new Event("change", {{ bubbles: true }}));
  }});

  await page.click("#examplesToggle");
  await page.waitForSelector("#examplesList button", {{ timeout: 10000 }});
  await page.getByText({json.dumps(EXAMPLE_FILENAME)}, {{ exact: true }}).click();
  await page.waitForSelector("#audio-path-0 path", {{ timeout: 30000 }});

  const result = await page.evaluate(() => {{
    const profile = window.__soundSketcherLastDrawProfile;
    const firstAudio = window.globalAudioData?.data?.[0] || {{}};
    const clusters = firstAudio.clusters || [];
    return {{
      pathCount: document.querySelectorAll("#audio-path-0 path").length,
      overlayCount: document.querySelectorAll("#svgCanvas .cluster-overlay").length,
      mixerCount: document.querySelectorAll(".mixer-item").length,
      profileLabel: profile?.label || null,
      profileMode: profile?.meta?.mode || null,
      profileFiles: profile?.meta?.files || null,
      featureFrameCount: profile?.counts?.["feature frames"] || null,
      objectifierRenderCount: profile?.counts?.["objectifier render"] || null,
      appendPathGroupCount: profile?.counts?.["append path group"] || null,
      clusterCount: clusters.length,
      firstClusterRegionCount: clusters[0]?.regions?.length || 0,
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
        raise AssertionError(f"objectifier fixture command failed: {details}")

    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    args = parser.parse_args()

    result = run_objectifier_fixture(args.base_url)
    if result["errors"]:
        raise AssertionError(f"browser objectifier errors: {result['errors']}")
    if result["clusterCount"] < 1 or result["firstClusterRegionCount"] < 1:
        raise AssertionError(f"fixture did not load objectifier clusters: {result}")
    if result["pathCount"] < 1:
        raise AssertionError(f"objectifier did not render cluster paths: {result}")
    if result["overlayCount"] < 1:
        raise AssertionError(f"objectifier did not render cluster overlays: {result}")
    if result["mixerCount"] != 1:
        raise AssertionError(f"unexpected mixer count: {result}")
    if result["profileLabel"] != "drawVisualization":
        raise AssertionError(f"drawing profiler did not run: {result}")
    if result["profileMode"] != "objectifier" or result["profileFiles"] != 1:
        raise AssertionError(f"unexpected objectifier profile metadata: {result}")
    if result["featureFrameCount"] is None or result["featureFrameCount"] < 1:
        raise AssertionError(f"feature frames were not processed: {result}")
    if result["objectifierRenderCount"] != 1:
        raise AssertionError(f"objectifier finalizer did not run once: {result}")
    if result["appendPathGroupCount"]:
        raise AssertionError(f"line/polygon finalizer ran during objectifier mode: {result}")

    print(
        "objectifier drawing browser fixture -> ok "
        f"({result['pathCount']} paths, {result['overlayCount']} overlays, "
        f"{result['clusterCount']} clusters)"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
