# SoundSketcher

SoundSketcher is a web-based environment for listening to, visualizing, and
sketching sound. It combines audio-feature extraction, interactive SVG
rendering, sonification controls, and research showcase pages for experiments
developed around sound perception.

This repository contains the refactored FastAPI version of SoundSketcher. The
older Flask/prototype deployment is kept online as a legacy reference while the
new application is developed and tested.

## Public Deployment

- Refactored app: <https://helen.mus.auth.gr/soundsketcher/>
- Legacy app: <https://helen.mus.auth.gr/app1/>
- Experiment index: <https://helen.mus.auth.gr/soundsketcher/experiments>

The refactor is deployed under the `/soundsketcher/` subpath on
`helen.mus.auth.gr`. The legacy app remains available under `/app1/`.

## Experiments

The refactor includes public-facing showcase pages for selected SoundSketcher
experiments:

- **Indefinite-Pitch Sounds**
  - <https://helen.mus.auth.gr/soundsketcher/indefinite_pitch>
  - Bilingual English/Greek showcase of the sinusoidal matching experiment.
  - Includes stimulus playback, sine matching controls, no-match option, result
    figures, bin plots, and publication context.
- **Noise-Tonal Preference**
  - <https://helen.mus.auth.gr/soundsketcher/noise_tonal_preference>
  - Showcase reconstruction of the questionnaire interface using the real
    SoundSketcher visualization engine and exported experiment stimuli.

## Project Layout

```text
app/
  main.py              # FastAPI app factory
  core/                # settings, templates, base-path helpers
  api/                 # page, audio, questionnaire, and experiment routes
  services/            # feature extraction, objectifier, stimuli, job queues
  storage/             # upload/cache and response persistence

static/                # app assets, JS modules, CSS, experiment media/results
templates/             # app and showcase HTML templates
scripts/               # QA, staging, and maintenance scripts
deploy/                # systemd/nginx deployment files and checklist
docs/                  # handoff notes, migration notes, parity audits
```

## Local Or Staging Run

The staging environment is normally configured with `.env.staging`; commit only
`.env.staging.example`.

```bash
scripts/run_python.sh -m uvicorn app.main:app --host 127.0.0.1 --port 5013
```

For the public `/soundsketcher/` subpath, the environment needs:

```bash
SOUNDSKETCHER_BASE_PATH="/soundsketcher"
```

## Deployment

Deployment is handled by systemd and nginx.

- `deploy/README.md` describes the service and nginx layout.
- `deploy/CHECKLIST.md` is the operational checklist for public verification.
- `deploy/systemd/soundsketcher-refactor.service` runs the refactor app.
- `deploy/nginx/soundsketcher-upstreams.conf` defines upstreams.
- `deploy/nginx/soundsketcher-locations.conf` contains the `/soundsketcher/`
  nginx locations.

The production service is expected to be enabled on boot and configured with:

```text
Restart=always
RestartSec=5
```

Useful server commands:

```bash
sudo systemctl status soundsketcher-refactor
sudo journalctl -u soundsketcher-refactor -f
sudo systemctl restart soundsketcher-refactor
sudo nginx -t
sudo systemctl reload nginx
```

## QA

Public deployment smoke test:

```bash
scripts/run_python.sh scripts/qa_public_deployment_smoke.py \
  --base-url https://helen.mus.auth.gr/soundsketcher \
  --legacy-url https://helen.mus.auth.gr/app1
```

Showcase browser QA:

```bash
scripts/run_python.sh scripts/qa_indefinite_pitch_showcase.py \
  --base-url https://helen.mus.auth.gr/soundsketcher

scripts/run_python.sh scripts/qa_noise_tonal_showcase.py \
  --base-url https://helen.mus.auth.gr/soundsketcher
```

If local Python cannot verify the HTTPS certificate chain, add `--insecure` to
the public QA commands.

Full local QA scripts are available through `package.json`, for example:

```bash
npm run qa:settings
npm run qa:staging-profile
npm run qa:readiness
npm run qa:public-smoke
```

Browser QA uses Playwright and requires Node 18 or newer.

## Notes For Contributors

- Do not commit runtime data, uploads, cached audio, or real environment files.
- Keep `.env.staging.example` updated when adding required settings.
- Keep legacy `/app1/` nginx routing separate from refactor `/soundsketcher/`
  routing.
- Update `docs/PROJECT_HANDOFF.md` after meaningful implementation or deployment
  changes so future work can resume without rediscovery.
