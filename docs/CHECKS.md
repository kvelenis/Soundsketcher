# Stage Checks

Run these checks from the original project root:

```bash
python3 -m venv refactor_sandbox/.venv
refactor_sandbox/.venv/bin/python -m pip install -r refactor_sandbox/requirements-stage1.txt
```

```bash
refactor_sandbox/.venv/bin/python -m py_compile \
  refactor_sandbox/app/main.py \
  refactor_sandbox/app/core/config.py \
  refactor_sandbox/app/core/templates.py \
  refactor_sandbox/app/api/pages.py \
  refactor_sandbox/app/api/audio.py \
  refactor_sandbox/app/api/questionnaires.py \
  refactor_sandbox/app/api/experiments.py
```

Expected result:
- No output and exit code `0`.

```bash
refactor_sandbox/.venv/bin/python -c "import sys; sys.path.insert(0, 'refactor_sandbox'); from app.main import create_app; app = create_app(); print('\n'.join(sorted(route.path for route in app.routes)))"
```

Expected result:
- The app imports.
- `/healthz` is present.
- `/static` is mounted.
- Template-only page routes such as `/`, `/analyze`, `/indefinite_pitch`, and `/questionnaire-soundsketcher` are present.

Current local note:
- Syntax checks pass with Python 3.12.8.
- Stage 1 dependencies are installed in `refactor_sandbox/.venv`.
- `/healthz` was verified on `http://127.0.0.1:5012/healthz`.

## Stage 2 Page QA

Start the sandbox server:

```bash
refactor_sandbox/.venv/bin/python -m uvicorn app.main:app --app-dir refactor_sandbox --host 127.0.0.1 --port 5012
```

In another terminal, run:

```bash
refactor_sandbox/.venv/bin/python refactor_sandbox/scripts/qa_page_routes.py --base-url http://127.0.0.1:5012
```

Expected result:
- Every migrated page route prints `ok`.
- `/test` may print `ok, empty template` because `templates/test.html` is empty.

## Stage 3 Questionnaire QA

Start the sandbox server:

```bash
refactor_sandbox/.venv/bin/python -m uvicorn app.main:app --app-dir refactor_sandbox --host 127.0.0.1 --port 5012
```

In another terminal, run:

```bash
refactor_sandbox/.venv/bin/python refactor_sandbox/scripts/qa_questionnaire_routes.py --base-url http://127.0.0.1:5012
```

Expected result:
- Representative questionnaire POST endpoints print `ok`.
- Noise-tonal stimulus and supplementary sound endpoints print `ok`.
- QA files are written under `refactor_sandbox/runtime/responses`.

## Stage 4a Stimulus Route QA

Start the sandbox server:

```bash
refactor_sandbox/.venv/bin/python -m uvicorn app.main:app --app-dir refactor_sandbox --host 127.0.0.1 --port 5012
```

In another terminal, run:

```bash
refactor_sandbox/.venv/bin/python refactor_sandbox/scripts/qa_stimulus_routes.py --base-url http://127.0.0.1:5012
```

Expected result:
- Sound-listing endpoints print `ok` with counts matching the static folders.
- Noise-tonal paired stimuli endpoint prints `ok`.

## Stage 4 Audio/Cache QA

Start the sandbox server:

```bash
refactor_sandbox/.venv/bin/python -m uvicorn app.main:app --app-dir refactor_sandbox --host 127.0.0.1 --port 5012
```

In another terminal, run:

```bash
refactor_sandbox/.venv/bin/python refactor_sandbox/scripts/qa_audio_cache_routes.py --base-url http://127.0.0.1:5012
```

Expected result:
- Cache existence, listing, cached load, and reuse-cache upload checks print `ok`.
- Fresh upload prints `safely deferred`, because feature extraction is intentionally not migrated yet.
- QA files are written under `refactor_sandbox/runtime/user_data`.

## Stage 5 Feature Boundary QA

The Stage 4 audio/cache QA script now also checks the feature extraction boundary.

Start the sandbox server:

```bash
refactor_sandbox/.venv/bin/python -m uvicorn app.main:app --app-dir refactor_sandbox --host 127.0.0.1 --port 5012
```

In another terminal, run:

```bash
refactor_sandbox/.venv/bin/python refactor_sandbox/scripts/qa_audio_cache_routes.py --base-url http://127.0.0.1:5012
```

Expected result:
- `/feature_extraction/status` prints `ok`.
- `/recalculate_features` prints `extracted` when Stage 6 librosa dependencies are installed.
- Fresh `/upload_wavs` prints `extracted` when Stage 6 librosa dependencies are installed.
- On the first run, extraction may take several seconds while librosa/numba warms up.

## Stage 5a Librosa Extractor Setup

Install the lightweight extractor dependencies:

```bash
refactor_sandbox/.venv/bin/python -m pip install -r refactor_sandbox/requirements-stage6-librosa.txt
```

Then rerun the Stage 5 QA script:

```bash
refactor_sandbox/.venv/bin/python refactor_sandbox/scripts/qa_audio_cache_routes.py --base-url http://127.0.0.1:5012
```

## Stage 5b Pitch QA

The audio/cache QA script now checks that `f0_librosa` exists and is nonzero for synthetic sine-wave test audio.

Expected result:
- `/recalculate_features` prints `extracted`.
- `/upload_wavs fresh upload` prints `extracted`.
- The script exits successfully only if `f0_librosa` is present and nonzero.

## Stage 5c RMS/Loudness QA

The audio/cache QA script now checks that `rms` and `loudness` exist and are nonzero for synthetic sine-wave test audio.

Expected result:
- `/recalculate_features` prints `extracted`.
- `/upload_wavs fresh upload` prints `extracted`.
- The script exits successfully only if `rms` and `loudness` are present and nonzero.

## Pre-Staging Readiness QA

Before packaging a preview, review `docs/DEPLOYMENT_MANIFEST.md`.

Start the sandbox server:

```bash
refactor_sandbox/.venv/bin/python -m uvicorn app.main:app --app-dir refactor_sandbox --host 127.0.0.1 --port 5012
```

In another terminal, run:

```bash
cd refactor_sandbox
npm run qa:readiness
```

Expected result:
- Server root is reachable.
- The active Python executable is reported.
- Settings environment override checks pass.
- Staging launch profile checks pass.
- Legacy parity checks report the main-page controls, replaced scripts, and deferred areas.
- Static/frontend/API checks pass.
- Line, polygon, and control-response browser fixtures pass.
- The command ends with `deployment readiness -> ok`.
