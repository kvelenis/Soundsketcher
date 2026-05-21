# Deployment Manifest

This manifest defines what should move to a staging preview and what should stay local.

## Include

Ship these files and folders:

```text
refactor_sandbox/
  app/
  docs/
  scripts/
  static/
  templates/
  .env.staging.example
  .gitignore
  README.md
  package.json
  package-lock.json
  pyproject.toml
  requirements-stage1.txt
  requirements-stage6-librosa.txt
```

The staging app also still needs legacy project data outside `refactor_sandbox`:

```text
static/
templates/
```

Those are required because migrated questionnaire/stimulus routes and template fallback
still intentionally read legacy assets while the migration is incomplete.

## Required Cached Example

Ship this cached example folder for the current frontend readiness checks:

```text
refactor_sandbox/runtime/user_data/
  ea4038a20612493495cb675e8d36adeace1727d3d527c6f204cb67037b7633fd/
    20-37s.wav
    features.json
```

Do not treat the other local `runtime/user_data` folders as staging requirements. They
are QA artifacts from upload/cache and extractor tests unless explicitly selected later.

## Create On Host

The staging process needs these writable directories:

```text
refactor_sandbox/runtime/user_data
refactor_sandbox/runtime/responses
```

They may be empty except for the preferred cached example above. The server process must
have read/write access.

## Exclude

Do not ship these local/generated files:

```text
refactor_sandbox/.venv/
refactor_sandbox/node_modules/
refactor_sandbox/.env
refactor_sandbox/.env.local
refactor_sandbox/.env.staging
refactor_sandbox/.env.production
refactor_sandbox/.env.*.local
refactor_sandbox/**/__pycache__/
refactor_sandbox/**/*.pyc
refactor_sandbox/.DS_Store
refactor_sandbox/**/.DS_Store
```

Real env profiles are host-specific and intentionally ignored. Commit only
`.env.staging.example`.

## Install On Host

Python:

```bash
python3 -m venv refactor_sandbox/.venv
refactor_sandbox/.venv/bin/python -m pip install -r refactor_sandbox/requirements-stage1.txt
refactor_sandbox/.venv/bin/python -m pip install -r refactor_sandbox/requirements-stage6-librosa.txt
```

When using conda instead of `.venv`, activate the conda environment and run the same
`python -m pip install ...` commands. The npm QA scripts and staging runner use
`scripts/run_python.sh`, so they do not require a `.venv/bin/python` symlink. Set
`SOUNDSKETCHER_PYTHON` in `.env.staging` for service-style conda launches and remote
QA.

Node browser QA:

```bash
cd refactor_sandbox
npm install
npx playwright install chromium
```

## Start Staging

Create and edit the ignored staging profile:

```bash
cd refactor_sandbox
cp .env.staging.example .env.staging
scripts/run_staging_preview.sh .env.staging
```

For routine staging operations:

```bash
scripts/staging_start.sh .env.staging
scripts/staging_status.sh .env.staging
scripts/staging_logs.sh
scripts/staging_stop.sh
```

## Post-Deploy Gate

After the staging server is running, run:

```bash
cd refactor_sandbox
npm run qa:readiness
```

The deployment should not be shared until the command ends with:

```text
deployment readiness -> ok
```

## Current Limitations

- The preferred cached example must travel with the deploy until fresh extraction and
  objectifier parity are complete.
- Heavy extractor dependencies remain optional and status-reported.
- The legacy `/static` mount is still required for questionnaire/stimulus data.
- This manifest does not define reverse proxy, HTTPS, auth, persistence policy, or
  process supervision.
