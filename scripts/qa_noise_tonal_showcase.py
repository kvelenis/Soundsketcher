#!/usr/bin/env python3
import argparse
import json
import ssl
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

SANDBOX_ROOT = Path(__file__).resolve().parents[1]


def strip_base_path(value: str, base_url: str) -> str:
    if not value:
        return value
    base_path = urllib.parse.urlparse(base_url).path.rstrip("/")
    if base_path and value.startswith(f"{base_path}/"):
        return value[len(base_path):]
    return value


def open_url(base_url: str, path: str, *, insecure: bool = False):
    context = ssl._create_unverified_context() if insecure else None
    return urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=10, context=context)


def get_status(base_url: str, path: str, *, insecure: bool = False) -> int:
    with open_url(base_url, path, insecure=insecure) as response:
        response.read()
        return response.status


def get_json(base_url: str, path: str, *, insecure: bool = False) -> tuple[int, dict]:
    with open_url(base_url, path, insecure=insecure) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body) if body else {}


def run_browser_fixture(base_url: str) -> dict:
    node_script = f"""
const {{ chromium }} = require("playwright");

(async () => {{
  const browser = await chromium.launch({{ headless: true }});
  const page = await browser.newPage({{ viewport: {{ width: 1280, height: 820 }} }});
  const errors = [];
  const requests = [];
  page.on("request", (request) => requests.push(request.url()));
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {{
    if (message.type() === "error") errors.push(message.text());
  }});

  await page.goto({json.dumps(base_url.rstrip('/') + '/noise_tonal_preference')}, {{ waitUntil: "domcontentloaded" }});
  await page.waitForFunction(() => document.querySelectorAll(".stimulus-card").length >= 6, null, {{ timeout: 10000 }});
  await page.waitForSelector("#preference-visible-slider .noUi-handle", {{ timeout: 10000 }});
  await page.waitForSelector("#svgWrapper svg", {{ timeout: 20000 }});

  const firstHash = await page.evaluate(() => window.SoundSketcher?.state?.globalAudioData?.hash?.join("|") || "");
  await page.locator(".stimulus-card button").nth(1).click();
  await page.waitForFunction((previous) => {{
    const current = window.SoundSketcher?.state?.globalAudioData?.hash?.join("|") || "";
    return current && current !== previous;
  }}, firstHash, {{ timeout: 20000 }});
  await page.waitForSelector("#svgWrapper svg", {{ timeout: 10000 }});

  await page.locator("[data-lang='el']").click();
  await page.waitForSelector('html[lang="el"]', {{ timeout: 5000 }});

  const result = await page.evaluate(() => ({{
    title: document.querySelector("h1")?.textContent || "",
    lang: document.documentElement.lang,
    stimulusCount: document.querySelectorAll(".stimulus-card").length,
    figureCount: document.querySelectorAll(".figure-card img").length,
    figureSources: [...document.querySelectorAll(".figure-card img")].map((img) => img.getAttribute("src")),
    loadButtons: [...document.querySelectorAll(".stimulus-card button")].filter((button) => button.textContent.includes("Φόρτωση")).length,
    playButton: document.getElementById("playPairBtn")?.textContent || "",
    sliderValue: document.getElementById("preferenceSliderValue")?.textContent || "",
    uploadPromptVisible: document.body.innerText.includes("Upload your audio"),
    dropAreaExists: Boolean(document.getElementById("drop_area")),
    currentHash: window.SoundSketcher?.state?.globalAudioData?.hash?.join("|") || "",
    svgHeight: Math.round(document.getElementById("svgWrapper")?.getBoundingClientRect().height || 0),
    hasRealSvgWrapper: Boolean(document.getElementById("svgWrapper")),
    hasSubmitButton: Boolean(document.getElementById("submitButton")),
    resultsLink: document.querySelector('a[href="#figures"]')?.textContent || "",
  }}));
  result.firstHash = firstHash;
  result.errors = errors;
  result.uploadRequests = requests.filter((url) => url.includes("/upload_wavs") || url.includes("/feature_extraction"));

  await browser.close();
  console.log(JSON.stringify(result));
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
            timeout=25,
        )
    finally:
        script_path.unlink(missing_ok=True)

    if completed.returncode != 0:
        details = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part)
        raise AssertionError(f"noise-tonal showcase browser fixture failed: {details}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification")
    args = parser.parse_args()

    if get_status(args.base_url, "/noise_tonal_preference", insecure=args.insecure) != 200:
        raise AssertionError("showcase page did not return 200")

    status, payload = get_json(
        args.base_url,
        "/static/noise_tonal_preference/feature_exports/manifest.json",
        insecure=args.insecure,
    )
    stimuli = payload.get("stimuli", [])
    if status != 200 or len(stimuli) != 6:
        raise AssertionError(f"unexpected static feature manifest: {payload}")
    for stimulus in stimuli:
        if len(stimulus.get("files", [])) != 2:
            raise AssertionError(f"stimulus is missing static audio/feature files: {stimulus}")
        for file_payload in stimulus["files"]:
            for key in ("audio_url", "features_url", "hash", "kind"):
                if not file_payload.get(key):
                    raise AssertionError(f"manifest file is missing {key}: {file_payload}")
            if get_status(args.base_url, file_payload["features_url"], insecure=args.insecure) != 200:
                raise AssertionError(f"feature export did not return 200: {file_payload['features_url']}")

    expected_figures = [
        "/static/noise_tonal_preference/results/mean_slider_bootstrap_ci_by_stimulus.png",
        "/static/noise_tonal_preference/results/slider_bin_distribution_by_stimulus.png",
        "/static/noise_tonal_preference/results/raincloud_noise_tonal.png",
        "/static/noise_tonal_preference/results/observed_vs_random_sd_by_stimulus.png",
    ]
    for figure in expected_figures:
        if get_status(args.base_url, figure, insecure=args.insecure) != 200:
            raise AssertionError(f"figure did not return 200: {figure}")

    result = run_browser_fixture(args.base_url)
    result["figureSources"] = [
        strip_base_path(source, args.base_url) for source in result["figureSources"]
    ]
    if result["errors"]:
        raise AssertionError(f"browser errors: {result['errors']}")
    if result["uploadRequests"]:
        raise AssertionError(f"showcase should not use upload/feature-extraction endpoints: {result}")
    if result["uploadPromptVisible"] or result["dropAreaExists"]:
        raise AssertionError(f"generic upload UI leaked into showcase: {result}")
    if not result["firstHash"] or result["firstHash"] == result["currentHash"]:
        raise AssertionError(f"stimulus switching did not change loaded sketch data: {result}")
    if result["svgHeight"] < 500:
        raise AssertionError(f"sketch viewport is too short: {result}")
    if result["lang"] != "el" or "Προτίμηση" not in result["title"]:
        raise AssertionError(f"Greek translation did not apply: {result}")
    if result["stimulusCount"] != 6 or result["loadButtons"] != 6:
        raise AssertionError(f"stimulus cards/buttons missing: {result}")
    if "Αναπαραγωγή" not in result["playButton"]:
        raise AssertionError(f"play button translation missing: {result}")
    if not result["hasRealSvgWrapper"] or not result["hasSubmitButton"]:
        raise AssertionError(f"real SoundSketcher engine hooks missing: {result}")
    if result["figureSources"][:4] != expected_figures:
        raise AssertionError(f"figures are missing or out of order: {result}")
    if result["sliderValue"] != "1.00":
        raise AssertionError(f"slider preview did not initialize: {result}")
    if not result["resultsLink"]:
        raise AssertionError(f"results link missing: {result}")

    print("noise-tonal preference showcase -> ok")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
