#!/usr/bin/env python3
import argparse
import json
import math
from io import BytesIO
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))


def make_wav_bytes(frequency: float = 440.0, duration: float = 0.25, sample_rate: int = 22050) -> bytes:
    buffer = BytesIO()
    amplitude = 12000
    frame_count = int(duration * sample_rate)
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            value = int(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate))
            frames.extend(value.to_bytes(2, byteorder="little", signed=True))
        wav_file.writeframes(bytes(frames))
    return buffer.getvalue()


def get_json(base_url: str, path: str, timeout: int = 10):
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body) if body else None


def post_multipart_upload(base_url: str, filename: str, content: bytes, timeout: int = 20):
    boundary = "----soundsketcherasyncqa"
    parts = [
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="run_objectifier"\r\n\r\n'
            "true\r\n"
        ).encode("utf-8"),
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="audio_files"; filename="{filename}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8") + content + b"\r\n",
    ]
    body = b"".join(parts) + f"--{boundary}--\r\n".encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/upload_wavs",
        data=body,
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def poll_until_done(base_url: str, filename: str, audio_hash: str, timeout: int = 90):
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        query = urllib.parse.urlencode({"filename": filename, "audio_hash": audio_hash})
        status, payload = get_json(base_url, f"/objectifier_status?{query}", timeout=10)
        if status != 200:
            raise AssertionError(f"unexpected status endpoint code: {status}")
        last_status = payload
        if payload.get("status") == "done":
            return payload
        if payload.get("status") == "failed":
            raise AssertionError(f"objectifier background job failed: {payload}")
        time.sleep(1)
    raise AssertionError(f"objectifier background job did not finish: {last_status}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5012")
    parser.add_argument("--max-initial-seconds", type=float, default=8.0)
    args = parser.parse_args()

    filename = f"async-objectifier-{int(time.time() * 1000)}.wav"
    start = time.monotonic()
    status, upload = post_multipart_upload(args.base_url, filename, make_wav_bytes())
    initial_seconds = time.monotonic() - start

    if status != 200 or upload.get("files_processed") != 1:
        raise AssertionError(f"unexpected upload response: {status} {upload}")
    if initial_seconds > args.max_initial_seconds:
        raise AssertionError(f"upload waited too long for async objectifier: {initial_seconds:.2f}s")

    item = upload["data"][0]
    if not item.get("features"):
        raise AssertionError(f"async upload did not return features immediately: {upload}")
    job = item.get("objectifier_job")
    if not job or job.get("status") not in {"queued", "running", "done"}:
        raise AssertionError(f"async upload did not return objectifier job state: {upload}")

    final_status = poll_until_done(args.base_url, upload["filename"][0], upload["hash"][0])
    if not final_status.get("clusters"):
        raise AssertionError(f"finished objectifier status did not include clusters: {final_status}")

    print(
        "objectifier async routes -> ok "
        f"(initial {initial_seconds:.2f}s, final {final_status.get('status')})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, urllib.error.HTTPError, urllib.error.URLError) as error:
        print(f"QA failed: {error}", file=sys.stderr)
        raise SystemExit(1)
