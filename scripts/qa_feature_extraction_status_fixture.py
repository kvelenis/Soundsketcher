#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

SANDBOX_ROOT = Path(__file__).resolve().parents[1]


def run_fixture(base_url: str) -> dict:
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
  await page.waitForFunction(() => Boolean(window.SoundSketcherFeatureStatus), null, {{ timeout: 10000 }});

  const result = await page.evaluate(async () => {{
    const originalFetch = window.fetch;
    const originalVisualizeAllFiles = window.visualizeAllFiles;
    let fetchCount = 0;
    let visualizeCalls = [];
    let submitClicks = 0;

    const submitButton = document.getElementById("submitButton");
    submitButton?.addEventListener("click", () => {{
      submitClicks += 1;
    }});

    window.SoundSketcher = window.SoundSketcher || {{}};
    window.SoundSketcher.state = window.SoundSketcher.state || {{}};
    window.SoundSketcher.state.globalAudioData = {{
      filename: ["async-test.wav"],
      hash: ["asyncstatus123"],
      data: [{{
        features: [],
        audio_url: "/user_data/asyncstatus123/async-test.wav",
        feature_job: {{
          status: "running",
          job_id: "fixture-job",
          progress: 52,
          message: "Estimating pitch confidence with CREPE",
          elapsed_seconds: 12.3,
        }},
      }}],
    }};
    window.globalAudioData = window.SoundSketcher.state.globalAudioData;

    window.fetch = async (url) => {{
      const urlText = String(url);
      if (urlText.startsWith("/feature_extraction_job_status")) {{
        fetchCount += 1;
        const body = fetchCount === 1
          ? {{ status: "unknown" }}
          : {{
              status: "done",
              features: [
                {{ timestamp: 0, spectral_centroid: 440, loudness: 0.5 }},
                {{ timestamp: 0.1, spectral_centroid: 450, loudness: 0.6 }},
              ],
              objectifier_job: {{ status: "queued", job_id: "objectifier-fixture" }},
            }};
        return new Response(JSON.stringify(body), {{
          status: 200,
          headers: {{ "content-type": "application/json" }},
        }});
      }}
      if (urlText.startsWith("/objectifier_status")) {{
        return new Response(JSON.stringify({{
          status: "queued",
          objectifier_exists: false,
          job_id: "objectifier-fixture",
          filename: "async-test.wav",
          hash: "asyncstatus123",
        }}), {{
          status: 200,
          headers: {{ "content-type": "application/json" }},
        }});
      }}
      return originalFetch(url);
    }};

    window.visualizeAllFiles = async (files) => {{
      visualizeCalls.push(files);
    }};

    window.SoundSketcherFeatureStatus.updateFeatureExtractionStatus();
    const initialStatusText = document.getElementById("objectifierStatus")?.textContent || "";
    const initialProgressWidth = document.querySelector("#objectifierStatus .objectifier-status__bar-fill")?.style.width || "";
    const floatingStatus = document.getElementById("featureProgressStatus");
    const floatingStatusText = floatingStatus?.textContent || "";
    const floatingProgressWidth = floatingStatus?.querySelector(".objectifier-status__bar-fill")?.style.width || "";
    const floatingHidden = floatingStatus?.classList.contains("app-hidden");

    await new Promise((resolve) => setTimeout(resolve, 2600));

    const fileData = window.SoundSketcher.state.globalAudioData.data[0];
    window.fetch = originalFetch;
    window.visualizeAllFiles = originalVisualizeAllFiles;

    return {{
      fetchCount,
      visualizeCalls,
      submitClicks,
      featureJob: fileData.feature_job,
      objectifierJob: fileData.objectifier_job,
      featureCount: fileData.features.length,
      firstCentroid: fileData.features[0]?.spectral_centroid,
      initialStatusText,
      initialProgressWidth,
      floatingStatusText,
      floatingProgressWidth,
      floatingHidden,
      statusText: document.getElementById("objectifierStatus")?.textContent || "",
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
            timeout=20,
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
        raise AssertionError(f"feature extraction status fixture command failed: {details}")

    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    args = parser.parse_args()

    result = run_fixture(args.base_url)
    if result["errors"]:
        raise AssertionError(f"browser feature-status errors: {result['errors']}")
    if result["fetchCount"] < 2:
        raise AssertionError(f"unknown status stopped polling too early: {result}")
    if result["featureJob"] is not None:
        raise AssertionError(f"feature job was not cleared after done: {result}")
    if result["featureCount"] != 2 or result["firstCentroid"] != 440:
        raise AssertionError(f"features were not applied after done: {result}")
    if "52%" not in result["initialStatusText"] or "CREPE" not in result["initialStatusText"]:
        raise AssertionError(f"feature progress percentage/message was not rendered: {result}")
    if result["initialProgressWidth"] != "52%":
        raise AssertionError(f"feature progress bar was not rendered at expected width: {result}")
    if result["floatingHidden"] is not False:
        raise AssertionError(f"floating feature progress status was hidden: {result}")
    if "52%" not in result["floatingStatusText"] or "CREPE" not in result["floatingStatusText"]:
        raise AssertionError(f"floating feature progress text was not rendered: {result}")
    if result["floatingProgressWidth"] != "52%":
        raise AssertionError(f"floating feature progress bar width was wrong: {result}")
    expected_objectifier_job = {
        "status": "queued",
        "objectifier_exists": False,
        "job_id": "objectifier-fixture",
        "filename": "async-test.wav",
        "hash": "asyncstatus123",
    }
    if result["objectifierJob"] != expected_objectifier_job:
        raise AssertionError(f"objectifier job was not wired after feature completion: {result}")
    if result["visualizeCalls"] != [[{"name": "async-test.wav", "audio_url": "/user_data/asyncstatus123/async-test.wav"}]]:
        raise AssertionError(f"visualizeAllFiles redraw path was not called: {result}")
    if result["submitClicks"] != 0:
        raise AssertionError(f"fallback submit redraw ran despite audio_url path: {result}")

    print("feature extraction status browser fixture -> ok")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
