# Staging Checklist

This checklist is the handoff gate before the sandbox leaves localhost for a shared
preview/staging environment.

Use `docs/DEPLOYMENT_MANIFEST.md` for the exact include/exclude list when packaging the
preview.

## Current Readiness

Status: locally ready for a first staging preview.

The staging build should be treated as a preview, not production. The app can boot,
serve the migrated page shell, load the preferred cached example, render line and
polygon sketches in a real browser, and respond to at least one mapping-control change.

## Runtime Command

Run the FastAPI sandbox from the original project root:

```bash
refactor_sandbox/.venv/bin/python -m uvicorn app.main:app --app-dir refactor_sandbox --host 0.0.0.0 --port 5012
```

For local-only checks, use `127.0.0.1` instead of `0.0.0.0`.

The staging helper loads an env profile and starts the server:

```bash
cd refactor_sandbox
scripts/run_staging_preview.sh .env.staging.example
```

Use `.env.staging.example` as a template. The example file is committed as a profile
shape, not as a production secret store.

The helper uses `scripts/run_python.sh`, so it works with either the local `.venv`, an
active conda environment, or `SOUNDSKETCHER_PYTHON=/path/to/python`.
For conda-based staging services, set `SOUNDSKETCHER_PYTHON` to the conda environment's
Python executable.

For routine server work, prefer the managed scripts:

```bash
scripts/staging_start.sh .env.staging
scripts/staging_status.sh .env.staging
scripts/staging_logs.sh
scripts/staging_stop.sh
```

For real staging, create an ignored host-specific profile:

```bash
cp .env.staging.example .env.staging
```

Only `.env.staging.example` should be committed. Real profiles such as `.env.staging`,
`.env.local`, and `.env.production` are intentionally ignored.

## Environment Settings

Local defaults still work without environment variables. Staging can override paths and
runtime settings with these variables:

```bash
export SOUNDSKETCHER_APP_NAME="SoundSketcher Staging"
export SOUNDSKETCHER_PROJECT_ROOT="/srv/soundsketcher"
export SOUNDSKETCHER_LEGACY_ROOT="/srv/soundsketcher"
export SOUNDSKETCHER_STATIC_DIR="/srv/soundsketcher/static"
export SOUNDSKETCHER_SANDBOX_STATIC_DIR="/srv/soundsketcher/refactor_sandbox/static"
export SOUNDSKETCHER_TEMPLATES_DIR="/srv/soundsketcher/templates"
export SOUNDSKETCHER_UPLOAD_DIR="/srv/soundsketcher/user_data"
export SOUNDSKETCHER_STATIC_UPLOAD_DIR="/srv/soundsketcher/static_uploads"
export SOUNDSKETCHER_RUNTIME_DIR="/srv/soundsketcher/refactor_sandbox/runtime"
export SOUNDSKETCHER_RESPONSE_ROOT="/srv/soundsketcher/refactor_sandbox/runtime/responses"
export SOUNDSKETCHER_CACHE_ROOT="/srv/soundsketcher/refactor_sandbox/runtime/user_data"
export SOUNDSKETCHER_PREFERRED_EXAMPLE_HASH="ea4038a20612493495cb675e8d36adeace1727d3d527c6f204cb67037b7633fd"
export SOUNDSKETCHER_PREFERRED_EXAMPLE_FILENAME="20-37s.wav"
export SOUNDSKETCHER_WORKER_COUNT="8"
export SOUNDSKETCHER_PYTHON=""
```

Optional heavy-tool/model settings:

```bash
export SOUNDSKETCHER_WAV2VEC_MODEL_NAME="facebook/wav2vec2-base-960h"
export SOUNDSKETCHER_OBJECTIFIER_MODEL_NAME="facebook/wav2vec2-base"
export SOUNDSKETCHER_MATLAB_TOOLBOX_DIR="/opt/soundsketcher/matlab"
export SOUNDSKETCHER_SONIC_ANNOTATOR_DIR="/opt/sonic-annotator"
```

The settings override check is included in `npm run qa:readiness` and can also be run
directly:

```bash
npm run qa:settings
npm run qa:staging-profile
```

## Python Dependencies

Install the base app dependencies:

```bash
python -m pip install -r refactor_sandbox/requirements-stage1.txt
```

Install the lightweight extractor slice when fresh upload/recalculate is expected to work:

```bash
python -m pip install -r refactor_sandbox/requirements-stage6-librosa.txt
```

Heavy extractor dependencies are not staging requirements yet:

- MATLAB worker
- Sonic Annotator
- CREPE/TensorFlow
- Aubio
- Mosqito
- Objectifier model stack

Those should stay behind `/feature_extraction/status` until migrated one at a time.

## Node Browser QA Dependencies

Install Node QA dependencies from `refactor_sandbox`:

```bash
npm install
npx playwright install chromium
```

Then use the staging gate:

```bash
npm run qa:readiness
```

The command expects the sandbox server to already be running.

## Required Runtime Data

The current frontend QA depends on the preferred cached example:

```text
runtime/user_data/ea4038a20612493495cb675e8d36adeace1727d3d527c6f204cb67037b7633fd/20-37s.wav
runtime/user_data/ea4038a20612493495cb675e8d36adeace1727d3d527c6f204cb67037b7633fd/features.json
```

This folder must be deployed with the staging app until upload/extraction parity is
complete enough to generate equivalent data on demand.

## Static And Template Paths

The sandbox currently uses two frontend/data boundaries:

- `/sandbox-static`: sandbox-owned CSS, JS, favicon, and app assets.
- `/static`: legacy static data used by questionnaire and stimulus routes.

The `/static` mount is still required for experiment stimuli. It should not be removed
from staging until those data-serving routes have their own sandbox-owned asset folder.

## Writable Runtime Paths

The staging process needs write access to:

```text
runtime/user_data
runtime/responses
```

These paths hold uploaded/cache audio data and questionnaire responses for the sandbox.
Do not point them at the legacy production folders during staging.

## Pre-Staging QA Gate

With the server running, this must pass before sharing the preview:

```bash
cd refactor_sandbox
npm run qa:readiness
```

Expected checks:

- Node QA package setup exists.
- Staging launch profile contains the required keys.
- Real env profiles are ignored; only `.env.staging.example` is committed.
- Server root is reachable.
- Drawing static contracts pass.
- Frontend assets are sandbox-owned.
- Preferred cached example is listed and loads.
- Sonification module contracts pass.
- Line browser render passes.
- Polygon browser render passes.
- Mapping control response passes.

## Manual Smoke Test

After the automated gate, manually check:

1. Open the staging URL.
2. Load `Examples > 20-37s.wav`.
3. Confirm the sketch renders.
4. Toggle polygon mode and resketch.
5. Move the line-length slider and resketch.
6. Confirm the browser console has no errors.
7. Confirm audio URLs load from `/user_data/...`.

## Known Gaps Before Production

- Full objectifier execution is deferred.
- Heavy feature parity with the legacy extractor is incomplete.
- Questionnaire/stimulus static data still depends on the legacy `/static` mount.
- The staging server command is single-process and does not define reverse-proxy,
  HTTPS, auth, persistence, or logging policy.

## Suggested Next Deployment Slice

Before a public preview, decide the target host layout and create the matching staging
environment file or service configuration from the variables above.
