#!/usr/bin/env python3
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from app.core.config import get_settings


LOCAL_ASSET_ATTRIBUTES = {
    "script": ("src",),
    "link": ("href",),
    "img": ("src",),
}


class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        for attribute in LOCAL_ASSET_ATTRIBUTES.get(tag, ()):
            value = attrs_dict.get(attribute)
            if value:
                self.assets.append((tag, value))
        if tag == "link":
            self.links.append(attrs_dict)


def get_text(base_url: str, path: str) -> tuple[int, str]:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=10) as response:
        return response.status, response.read().decode("utf-8")


def get_json(base_url: str, path: str):
    status, body = get_text(base_url, path)
    return status, json.loads(body) if body else None


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


def normalize_asset_url(base_url: str, value: str) -> str:
    return urllib.parse.urljoin(f"{base_url.rstrip('/')}/", value)


def assert_local_asset_is_sandbox(base_url: str, tag: str, url: str) -> None:
    normalized = normalize_asset_url(base_url, url)
    parsed = urllib.parse.urlparse(normalized)
    if parsed.netloc != urllib.parse.urlparse(base_url).netloc:
        return
    if not parsed.path.startswith("/sandbox-static/"):
        raise AssertionError(f"{tag} uses non-sandbox local asset: {normalized}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    args = parser.parse_args()
    settings = get_settings()

    checks = []

    status, html = get_text(args.base_url, "/")
    if status != 200:
        raise AssertionError(f"unexpected root status: {status}")
    checks.append("/ renders -> ok")

    parser = AssetParser()
    parser.feed(html)

    favicon_links = [
        link
        for link in parser.links
        if "icon" in link.get("rel", "").split()
    ]
    if not favicon_links:
        raise AssertionError("missing favicon link")
    favicon_href = favicon_links[0].get("href", "")
    if "/sandbox-static/assets/favicon.svg" not in normalize_asset_url(args.base_url, favicon_href):
        raise AssertionError(f"favicon is not sandbox-owned: {favicon_href}")
    checks.append("sandbox favicon -> ok")

    for tag, url in parser.assets:
        assert_local_asset_is_sandbox(args.base_url, tag, url)
    checks.append("local frontend assets are sandbox-owned -> ok")

    stale_markers = [
        "drawVisualisation.js?v=frontend-migration-6",
        "/static/js/submitButton.js",
        "/static/style.css",
    ]
    stale_hits = [marker for marker in stale_markers if marker in html]
    if stale_hits:
        raise AssertionError(f"stale legacy frontend references found: {stale_hits}")
    checks.append("stale legacy frontend references absent -> ok")

    status, cached = get_json(args.base_url, "/list_cached_files")
    files = cached.get("cached_files", []) if isinstance(cached, dict) else []
    if status != 200 or not files:
        raise AssertionError(f"unexpected cached file list: {cached}")
    preferred = files[0]
    if preferred.get("hash") != settings.preferred_example_hash:
        raise AssertionError(f"preferred fixture is not first: {preferred}")
    if preferred.get("filename") != settings.preferred_example_filename:
        raise AssertionError(f"preferred fixture filename mismatch: {preferred}")
    checks.append("preferred cached example listed first -> ok")

    status, loaded = post_form(
        args.base_url,
        "/load_cached_audio",
        {
            "filename": settings.preferred_example_filename,
            "hash": settings.preferred_example_hash,
        },
    )
    if status != 200 or loaded.get("files_processed") != 1:
        raise AssertionError(f"unexpected cached load response: {loaded}")
    features = loaded["data"][0].get("features")
    if not isinstance(features, list) or len(features) < 700:
        raise AssertionError(f"preferred fixture features look wrong: {len(features or [])}")
    checks.append(f"preferred cached example loads -> ok ({len(features)} frames)")

    for check in checks:
        print(check)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
