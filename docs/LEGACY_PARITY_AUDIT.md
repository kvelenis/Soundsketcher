# Legacy Parity Audit

Date: 2026-05-15

## Scope

This audit compares the staging sandbox against the copied legacy reference at:

`/mnt/ssd1/kvelenis/soundsketcher-staging/legacy_reference`

The reference copy is intentionally small, about 42 MB, and excludes live user data, uploaded audio, response datasets, model caches, third-party toolboxes, and generated analysis folders. It is for inspection and migration only.

## Current State

The sandbox has a clean FastAPI app shell, environment-driven staging config, managed start/stop/status/log scripts, and deployment readiness QA. The main SoundSketcher page now loads sandbox-owned templates, CSS, assets, and JavaScript from `/sandbox-static`.

The app is no longer just a raw copy of the legacy frontend. It has a module bridge around shared state, audio upload/cache clients, visualization orchestration, playback, mixer UI, sonification engines, and several drawing helpers.

## Main Page Parity

### Already Migrated Or Replaced

- Header and examples dropdown are sandbox-owned.
- Main layout is split into partial templates.
- Main CSS is split into sandbox-owned files:
  - `app-chrome.css`
  - `app-layout.css`
  - `app-shell.css`
  - `settings-panel.css`
  - `ui-overlays.css`
  - `mixer.css`
  - `sliders.css`
  - `accordion.css`
  - `header.css`
- The page no longer depends on legacy `/static` for main page CSS, JS, header assets, or favicon.
- Cached example loading works through sandbox cache routes.
- Upload/cache response shape is normalized behind backend storage and frontend client modules.
- Drag/drop upload bindings moved out of inline HTML.
- Main playback was moved behind `SoundSketcher.playback` and module-backed controller code.
- Mixer construction was moved to `mixer.module.mjs`.
- Sonification has been split into module-backed state, config, controllers, engines, helpers, and playback controls.
- Drawing has been partially split into modules for:
  - feature mapping
  - renderers
  - axis rendering
  - path/pattern helpers
  - context construction
  - sonification path bridging
  - playback visual helpers
- QA exists for page routes, cache routes, questionnaire routes, stimulus routes, frontend runtime boundary, drawing fixtures, sonification boundary, and deployment readiness.

### Legacy Files Covered By Sandbox Equivalents

These legacy JS files are either copied into sandbox ownership, reduced, or replaced by module-backed code:

- `accordionButton.js`
- `autoResketch.js`
- `auxiliaryFunctions.js`
- `changeNamesButton.js`
- `collapsibleMenu.js`
- `drawVisualisation.js`
- `exportButtons.js`
- `featureProcessors.js`
- `handleDropAudios.js`
- `helpButtons.js`
- `mixerConstructor.js`
- `objectifier.js`
- `playStopMainAudioFunctions.js`
- `presetButton.js`
- `recalculateButton.js`
- `recordingAudio.js`
- `resetButton.js`
- `sliderFunctions.js`
- `sonificators.js`
- `submitButton.js`
- `tooltip.js`

Important note: "covered" does not mean all behavior is fully modernized. Some files still act as compatibility bridges while we move behavior into modules.

## Gaps And Deferred Work

### 1. Objectifier

Status: partially present, not complete.

The legacy reference includes objectifier routes and files:

- `/objectifier-upload-page`
- `/objectifier-upload-audio`
- `/objectifier/{audio_file}`
- `/plotly-data/{audio_file}`
- `static/js/objectifier.js`
- `templates/objectifier-upload-page.html`
- `templates/plotly_visualization*.html`

The sandbox has objectifier UI/files available, and now includes a lightweight objectifier backend slice that writes frontend-compatible cluster regions for uploads/recalculation. The main cached fixture intentionally remains `clusters: null`; objectifier coverage uses a separate seeded fixture. Full legacy objectifier model reuse and plotly data serving still need a dedicated migration slice.

Recommended approach:

1. Replace the lightweight sandbox region builder with the legacy clustering/model path behind the same service boundary.
2. Add plotly object routes and visualization templates.
3. Broaden browser QA beyond the seeded fixture once the model-backed output is stable.

### 2. Full Feature Extraction

Status: lightweight path exists, full legacy stack is not migrated.

Legacy extraction includes librosa, aubio, CREPE, MOSQITO, sonic-annotator, MATLAB-related loudness/roughness paths, smoothing, harmonic descriptors, and custom serial processing. The sandbox currently has the service boundary and a lightweight extraction path suitable for staging, but not the full research extraction stack.

Recommended approach:

1. Keep the current service boundary.
2. Add extractors one at a time behind explicit dependency checks.
3. Preserve response shape after every extractor addition.
4. Treat MATLAB/sonic-annotator integration as separate deployment work, not normal page migration.

### 3. Analysis And ML Pages

Status: routes/templates are present, backend behavior is incomplete.

The legacy app has several analysis endpoints and pages:

- `/analyze`
- `/analyze_wav`
- `/wav2vec`
- `/upload_wav2vec`
- `/clip-clap`
- `/clip-clap-wav`
- `/high_level_features`
- `/high_level_features_request`
- `/high_level_mosqito_2`
- `/analyze_audio`

The sandbox has many page routes so templates can render, but the heavy backend behaviors should not be pulled in directly. They need separate services and smaller test fixtures.

Recommended approach:

1. Decide which analysis page is product-critical.
2. Migrate only that page's backend route.
3. Add a minimal fixture and route-level QA.
4. Leave non-critical research pages as reference-only until needed.

### 4. Drawing Performance And Final Split

Status: active in progress.

The largest legacy risk remains drawing orchestration. The sandbox has already extracted many helpers, and `drawVisualisation.js` is now only a compatibility bridge over `drawing-visualization.module.mjs`. The remaining weight is in the drawing context modules and objectifier-specific rendering path.

Recommended approach:

1. Continue reducing drawing context modules by one responsibility per slice.
2. Keep line, polygon, and control-response browser QA passing.
3. Avoid changing visual behavior while moving code.
4. After extraction, optimize repeated DOM/canvas work.

### 5. Static Dataset Boundary

Status: intentionally mixed.

The main SoundSketcher page uses sandbox-owned assets. Questionnaire and stimulus routes still intentionally serve datasets from the app static boundary. This is acceptable for staging, but the data-serving contract should be documented before production deployment.

Recommended approach:

1. Keep application UI assets under sandbox static ownership.
2. Keep experiment/stimulus datasets separate from UI assets.
3. Add a manifest for each dataset that the app is expected to serve.

## Route Parity Summary

### Good Enough For Current Staging

- `/`
- `/healthz`
- `/check_file_exists`
- `/list_cached_files`
- `/load_cached_audio`
- `/upload_wavs`
- `/recalculate_features`
- `/feature_extraction/status`
- questionnaire save/list routes
- stimulus listing routes

### Present But Needs Deeper Validation

- experiment template-only pages
- questionnaire visual flows beyond route-level QA
- cached examples dropdown behavior after failures
- recording workflow with real microphone permissions

### Not Yet Equivalent To Legacy

- objectifier generation and plotly object routes
- full extraction stack
- CLAP/wav2vec/high-level feature processing
- MATLAB/sonic-annotator backed descriptors

## QA Guard

`scripts/qa_legacy_parity.py` checks the rendered main page for the important legacy-era controls that the sandbox must keep available, verifies the sandbox module/script boundary, confirms replaced legacy scripts are not loaded, and reports the intentionally deferred areas without failing the gate.

Run it with:

```bash
npm run qa:parity -- --base-url http://127.0.0.1:5013
```

## Recommended Next Slices

1. Add a small `legacy_reference` inventory note to the docs and keep it read-only in practice.
2. Continue drawing context reduction until the modules are mostly small orchestration helpers.
3. Start objectifier migration using a fixed fixture, not live upload generation.
4. Only after objectifier rendering is stable, expand the feature extraction service beyond the lightweight librosa path.

## Current Recommendation

Stay on localhost/staging for now. The app is in a good architectural direction, but the remaining gaps are high-risk enough that exposing it publicly before objectifier and full extraction contracts are nailed down would create noisy failures. The right move is controlled staging with fixtures, route QA, and one migrated behavior at a time.
