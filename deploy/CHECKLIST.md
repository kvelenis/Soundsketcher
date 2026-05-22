# SoundSketcher Public Deployment Checklist

Use this after changing the refactor app, nginx snippets, static routing, or the
public `/soundsketcher/` path.

## 1. Service State

```bash
sudo systemctl status soundsketcher-refactor
sudo systemctl is-enabled soundsketcher-refactor
sudo systemctl cat soundsketcher-refactor | grep Restart
```

Expected:

- `Active: active (running)`
- `enabled`
- `Restart=always`
- `RestartSec=5`

## 2. Restart After App Changes

```bash
sudo systemctl restart soundsketcher-refactor
sudo systemctl status soundsketcher-refactor
```

## 3. Validate Nginx

```bash
sudo nginx -t
sudo systemctl reload nginx
```

If nginx reports `duplicate location "/app1/"`, remove any `/app1/` block from
`/etc/nginx/snippets/soundsketcher-locations.conf`. The live
`helen.mus.auth.gr.conf` already owns legacy `/app1/`.

## 4. Public HTTP Smoke Checks

```bash
curl -s -o /dev/null -w "app %{http_code} %{content_type}\n" \
  https://helen.mus.auth.gr/soundsketcher/

curl -s -o /dev/null -w "experiments %{http_code} %{content_type}\n" \
  https://helen.mus.auth.gr/soundsketcher/experiments

curl -s -o /dev/null -w "legacy %{http_code} %{content_type}\n" \
  https://helen.mus.auth.gr/app1/

curl -s https://helen.mus.auth.gr/soundsketcher/healthz
curl -s https://helen.mus.auth.gr/soundsketcher/deployment-info
```

Expected:

- `/soundsketcher/` returns `200`
- `/soundsketcher/experiments` returns `200`
- `/app1/` still returns `200`, `301`, or `302`
- `/healthz` returns `{"status":"ok"}`
- `/deployment-info` returns `status: ok`

## 5. Static MIME Checks

```bash
curl -I https://helen.mus.auth.gr/soundsketcher/sandbox-static/css/app-chrome.css
curl -I https://helen.mus.auth.gr/soundsketcher/sandbox-static/js/main.module.mjs
```

Expected:

- CSS returns `Content-Type: text/css`
- `.mjs` returns `Content-Type: text/javascript`

## 6. Cached Audio Check

```bash
curl -s -o /dev/null -w "%{http_code} %{content_type}\n" \
  https://helen.mus.auth.gr/soundsketcher/user_data/89e708a7271e546c8384209e5a7209974cab98d1a60f14a25dad3a78d99b611d/all_together_experiment_pitch.wav
```

Expected:

- `200`
- audio content type, for example `audio/x-wav`; `application/octet-stream` is
  acceptable for WAV downloads as long as the status is `200`

## 7. Automated Public QA

From the project root:

```bash
scripts/run_python.sh scripts/qa_public_deployment_smoke.py \
  --base-url https://helen.mus.auth.gr/soundsketcher \
  --legacy-url https://helen.mus.auth.gr/app1

scripts/run_python.sh scripts/qa_indefinite_pitch_showcase.py \
  --base-url https://helen.mus.auth.gr/soundsketcher

scripts/run_python.sh scripts/qa_noise_tonal_showcase.py \
  --base-url https://helen.mus.auth.gr/soundsketcher
```

Expected:

- `public deployment smoke -> ok`
- `indefinite pitch showcase -> ok`
- `noise-tonal preference showcase -> ok`

If local Python cannot verify the HTTPS certificate chain but the browser can
open the site, add `--insecure` to the smoke command.

## 8. Browser Check

Open in a normal browser or private window:

- `https://helen.mus.auth.gr/soundsketcher/?check=<timestamp>`
- `https://helen.mus.auth.gr/soundsketcher/experiments?check=<timestamp>`
- `https://helen.mus.auth.gr/soundsketcher/indefinite_pitch?check=<timestamp>`
- `https://helen.mus.auth.gr/soundsketcher/noise_tonal_preference?check=<timestamp>`

Watch the console for:

- 404s on `/soundsketcher/sandbox-static/...`
- `.mjs` MIME errors
- cached audio 404s under `/soundsketcher/user_data/...`
