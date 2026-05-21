#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

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


def get_json(base_url: str, path: str, *, insecure: bool = False):
    with open_url(base_url, path, insecure=insecure) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body) if body else None


def run_browser_fixture(base_url: str) -> dict:
    node_script = f"""
const {{ chromium }} = require("playwright");

(async () => {{
  const browser = await chromium.launch({{ headless: true }});
  const page = await browser.newPage({{ viewport: {{ width: 1280, height: 820 }} }});
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {{
    if (message.type() === "error") errors.push(message.text());
  }});

  await page.goto({json.dumps(base_url.rstrip('/') + '/indefinite_pitch')}, {{ waitUntil: "domcontentloaded" }});
  await page.waitForFunction(() => document.querySelectorAll(".sound-card").length >= 19, null, {{ timeout: 10000 }});
  const preResult = await page.evaluate(() => {{
    document.getElementById("noMatch").click();
    document.getElementById("loopToggle").click();
    const volume = document.getElementById("sineVolume");
    volume.value = "55";
    volume.dispatchEvent(new Event("input", {{ bubbles: true }}));
    return {{
      noMatchPressed: document.getElementById("noMatch").getAttribute("aria-pressed"),
      loopPressed: document.getElementById("loopToggle").getAttribute("aria-pressed"),
    }};
  }});
  await page.evaluate(() => document.getElementById("greekButton").click());
  await page.waitForSelector('html[lang="el"]', {{ timeout: 5000 }});

  const result = await page.evaluate(() => ({{
    title: document.querySelector("h1")?.textContent || "",
    cardCount: document.querySelectorAll(".sound-card").length,
    currentTitle: document.getElementById("currentSoundTitle")?.textContent || "",
    doiLink: document.querySelector('a[href*="10.1121/10.0043569"]')?.href || "",
    audioSource: document.getElementById("stimulusAudio")?.getAttribute("src") || "",
    sliderMin: document.getElementById("frequencySlider")?.getAttribute("min") || "",
    sliderMax: document.getElementById("frequencySlider")?.getAttribute("max") || "",
    loopText: document.getElementById("loopToggle")?.textContent || "",
    audioLoop: document.getElementById("stimulusAudio")?.loop || false,
    sineVolumeValue: document.getElementById("sineVolume")?.value || "",
    sineVolumeReadout: document.getElementById("sineVolumeReadout")?.textContent || "",
    resultFigureCount: document.querySelectorAll('.figure-card img[src*="/static/indefinite_pitch/results/"]').length,
    resultFigureSources: [...document.querySelectorAll('.figure-card img')].map((img) => img.getAttribute("src")),
    stimulusCodes: [...document.querySelectorAll(".stimulus-code")].map((node) => node.textContent),
    binPlotCount: document.querySelectorAll(".bin-plot-link").length,
    noMatchText: document.getElementById("noMatch")?.textContent || "",
    greekTitle: document.querySelector("h1")?.textContent || "",
    greekHint: document.querySelector("[data-i18n='sliderHint']")?.textContent || "",
  }}));

  result.noMatchPressed = preResult.noMatchPressed;
  result.loopPressed = preResult.loopPressed;
  result.errors = errors;
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
        details = "\n".join(
            part for part in [completed.stdout.strip(), completed.stderr.strip()] if part
        )
        raise AssertionError(f"indefinite pitch browser fixture failed: {details}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification")
    args = parser.parse_args()

    with open_url(args.base_url, "/indefinite_pitch", insecure=args.insecure) as response:
        if response.status != 200:
            raise AssertionError(f"unexpected /indefinite_pitch status: {response.status}")

    status, study_sounds = get_json(args.base_url, "/get_sounds", insecure=args.insecure)
    if status != 200 or not isinstance(study_sounds, list) or len(study_sounds) < 19:
        raise AssertionError(f"unexpected study sound list: {study_sounds}")

    status, training_sounds = get_json(args.base_url, "/get_sounds_training", insecure=args.insecure)
    if status != 200 or not isinstance(training_sounds, list) or len(training_sounds) < 3:
        raise AssertionError(f"unexpected training sound list: {training_sounds}")

    result = run_browser_fixture(args.base_url)
    if result["errors"]:
        raise AssertionError(f"browser errors: {result['errors']}")
    if result["title"] != "Τονικά ακαθόριστοι ήχοι":
        raise AssertionError(f"unexpected Greek showcase title: {result}")
    if result["cardCount"] < 19:
        raise AssertionError(f"not enough rendered stimulus cards: {result}")
    if "10.1121/10.0043569" not in result["doiLink"]:
        raise AssertionError(f"missing DOI link: {result}")
    result["audioSource"] = strip_base_path(result["audioSource"], args.base_url)
    result["resultFigureSources"] = [
        strip_base_path(source, args.base_url) for source in result["resultFigureSources"]
    ]
    if not result["audioSource"].startswith("/static/indefinite_pitch/"):
        raise AssertionError(f"unexpected audio source: {result}")
    if result["sliderMin"] != "0" or result["sliderMax"] != "1000":
        raise AssertionError(f"slider is not using the legacy logarithmic control range: {result}")
    if result["noMatchPressed"] != "true":
        raise AssertionError(f"no-match button did not toggle: {result}")
    if result["loopPressed"] != "true" or not result["audioLoop"]:
        raise AssertionError(f"loop toggle did not enable audio looping: {result}")
    if result["loopText"] != "Loop: ON":
        raise AssertionError(f"loop text did not survive Greek switch: {result}")
    if result["sineVolumeValue"] != "55" or result["sineVolumeReadout"] != "55%":
        raise AssertionError(f"sine volume control did not update: {result}")
    if result["resultFigureCount"] < 4:
        raise AssertionError(f"expected result figures are missing: {result}")
    expected_figures = [
        "/static/indefinite_pitch/results/boxplot_per_sound_vertical.png",
        "/static/indefinite_pitch/results/Figure1_centroid_vs_pitch_errorbars_labeled.png",
        "/static/indefinite_pitch/results/loudness_centroid_vs_pitch.png",
        "/static/indefinite_pitch/results/Figure2_mixedlm_loud_only_partial_effect.png",
    ]
    if result["resultFigureSources"][:4] != expected_figures:
        raise AssertionError(f"result figures are not in the requested order: {result}")
    if "sn1" not in result["stimulusCodes"] or "noi" not in result["stimulusCodes"]:
        raise AssertionError(f"stimulus short labels are missing: {result}")
    if result["binPlotCount"] < 19:
        raise AssertionError(f"bin plot links are missing: {result}")
    if result["noMatchText"] != "Δεν βρίσκω αντιστοίχιση":
        raise AssertionError(f"Greek no-match text missing: {result}")
    if "slider" not in result["greekHint"]:
        raise AssertionError(f"Greek slider hold hint missing: {result}")

    print(
        "indefinite pitch showcase -> ok "
        f"({len(study_sounds)} study sounds, {len(training_sounds)} training sounds)"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
