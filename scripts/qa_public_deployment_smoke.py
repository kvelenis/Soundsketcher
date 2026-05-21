#!/usr/bin/env python3
import argparse
import ssl
import sys
import urllib.error
import urllib.request


CACHED_AUDIO_PATH = (
    "/user_data/"
    "89e708a7271e546c8384209e5a7209974cab98d1a60f14a25dad3a78d99b611d/"
    "all_together_experiment_pitch.wav"
)


def request(base_url: str, path: str, *, insecure: bool = False) -> tuple[int, str]:
    url = f"{base_url.rstrip('/')}{path}"
    context = ssl._create_unverified_context() if insecure else None
    try:
        with urllib.request.urlopen(url, timeout=15, context=context) as response:
            response.read(1024)
            return response.status, response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type", "")


def assert_status(
    base_url: str,
    path: str,
    expected_status: int = 200,
    *,
    insecure: bool = False,
) -> str:
    status, content_type = request(base_url, path, insecure=insecure)
    if status != expected_status:
        raise AssertionError(f"{path} returned {status}, expected {expected_status}")
    return content_type


def assert_content_type(
    base_url: str,
    path: str,
    expected_fragment: str,
    *,
    expected_status: int = 200,
    insecure: bool = False,
) -> None:
    content_type = assert_status(base_url, path, expected_status, insecure=insecure)
    if expected_fragment not in content_type:
        raise AssertionError(
            f"{path} returned content type {content_type!r}, expected {expected_fragment!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://helen.mus.auth.gr/soundsketcher")
    parser.add_argument("--legacy-url", default="https://helen.mus.auth.gr/app1")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification")
    args = parser.parse_args()

    for path in (
        "/",
        "/experiments",
        "/indefinite_pitch",
        "/noise_tonal_preference",
    ):
        assert_content_type(args.base_url, path, "text/html", insecure=args.insecure)

    assert_content_type(
        args.base_url,
        "/sandbox-static/css/app-chrome.css",
        "text/css",
        insecure=args.insecure,
    )
    assert_content_type(
        args.base_url,
        "/sandbox-static/js/main.module.mjs",
        "text/javascript",
        insecure=args.insecure,
    )
    assert_content_type(
        args.base_url,
        "/static/indefinite_pitch/results/boxplot_per_sound_vertical.png",
        "image/",
        insecure=args.insecure,
    )
    audio_content_type = assert_status(args.base_url, CACHED_AUDIO_PATH, insecure=args.insecure)
    if not (
        "audio/" in audio_content_type
        or "application/octet-stream" in audio_content_type
    ):
        raise AssertionError(
            f"{CACHED_AUDIO_PATH} returned content type {audio_content_type!r}, "
            "expected audio/* or application/octet-stream"
        )

    legacy_status, _legacy_type = request(args.legacy_url, "/", insecure=args.insecure)
    if legacy_status not in {200, 301, 302}:
        raise AssertionError(f"legacy app returned {legacy_status}, expected 200/301/302")

    print("public deployment smoke -> ok")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QA failed: {error}", file=sys.stderr)
        sys.exit(1)
