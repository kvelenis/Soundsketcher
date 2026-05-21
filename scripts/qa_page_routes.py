#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import urllib.error
import urllib.request

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from app.api.pages import PAGE_ROUTES
from app.core.config import get_settings


def check_page(base_url: str, path: str, template_name: str) -> tuple[bool, str]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            status = response.status
            content_type = response.headers.get("content-type", "")
            body = response.read(512).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        return False, f"{path} -> HTTP {error.code} for {template_name}"
    except Exception as error:
        return False, f"{path} -> {type(error).__name__}: {error}"

    if status != 200:
        return False, f"{path} -> HTTP {status} for {template_name}"
    if "text/html" not in content_type:
        return False, f"{path} -> unexpected content-type {content_type!r}"
    template_path = get_settings().templates_dir / template_name
    if not body.strip() and template_path.exists() and template_path.stat().st_size == 0:
        return True, f"{path} -> ok, empty template ({template_name})"
    if not body.strip():
        return False, f"{path} -> empty response body"

    return True, f"{path} -> ok ({template_name})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    args = parser.parse_args()

    failed = []
    for path, template_name in PAGE_ROUTES.items():
        ok, message = check_page(args.base_url, path, template_name)
        print(message)
        if not ok:
            failed.append(message)

    if failed:
        print("\nFailures:")
        for failure in failed:
            print(f"- {failure}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
