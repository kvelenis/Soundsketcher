# Staged Migration Plan

## Stage 0: Sandbox And Baseline

Status: in progress

Purpose:
- Create a separate refactor folder.
- Document the target architecture.
- Add lightweight checks that do not require the full production environment.

Checks:
- Original project remains unchanged.
- New package imports without syntax errors.
- Stage documents explain what moved and what did not.

## Stage 1: App Shell

Status: complete

Purpose:
- Add a FastAPI app factory.
- Centralize paths and settings.
- Move startup/shutdown resource ownership out of ad hoc globals.
- Keep old routes in place until migrated.
- Add the first dependency manifest.
- Move simple template-only page routes into `app/api/pages.py`.

Checks:
- Passed: `create_app()` imports in the sandbox venv.
- Passed: Static/template paths resolve from the original project root.
- Passed: App can list registered routes.
- Passed: `/healthz` returned `{"status":"ok"}` on localhost.
- Deferred: Browser-level rendering of individual migrated pages.

## Stage 2: Page Routes

Status: complete

Purpose:
- Move simple HTML page routes into `app/api/pages.py`.
- Keep template names and URLs unchanged.

Checks:
- Passed: all migrated page URLs return HTTP 200.
- Passed: template lookup uses the original `templates/` folder.
- Passed: `index.html`, experiment pages, questionnaire page, and results pages render.
- Note: `/test` returns HTTP 200 with an empty body because `templates/test.html` is empty.
- Added `scripts/qa_page_routes.py` so the check can be repeated.

Recommendation:
- Move questionnaire routes next. They are less coupled to heavy audio dependencies than upload/cache routes, but they expose useful architecture issues: Pydantic models, response persistence, duplicate JSON file logic, and future-safe storage boundaries.

## Stage 3: Questionnaire Routes

Status: complete

Purpose:
- Move study models and response-saving endpoints into `app/api/questionnaires.py`.
- Move filesystem writes behind `ResponseStore`.

Checks:
- Passed: representative payload validation works for shape, texture, indefinite pitch, noise-tonal preference, and supplementary responses.
- Passed: saves use `ResponseStore` in `app/storage/responses.py`.
- Passed: response reads use the storage abstraction instead of duplicated glob/loading route code.
- Passed: noise-tonal stimulus endpoint returns the static sample set.
- Passed: supplementary sound listing returns a JSON list.
- Passed: QA writes go to `refactor_sandbox/runtime/responses`, not the original response folders.
- Added `scripts/qa_questionnaire_routes.py` so the check can be repeated.

Recommendation:
- Move stimulus/listing routes next as Stage 4, then audio upload/cache after that. Stimulus routes are low-risk and will finish separating experiment metadata from heavyweight audio processing.

## Stage 4: Audio Upload And Cache

Status: complete

Purpose:
- Move upload, cache lookup, and recalculation routes into `app/api/audio.py`.
- Move path/hash/file handling into `CacheStore`.

Checks:
- Passed: filename/path handling is sanitized by `CacheStore`.
- Passed: `/check_file_exists` reports feature/objectifier cache state.
- Passed: `/list_cached_files` returns cached file metadata and audio URLs.
- Passed: `/load_cached_audio` returns the existing frontend-compatible cached response shape.
- Passed: `/upload_wavs?reuse_cached=true` returns cached data in the frontend-compatible response shape.
- Passed: fresh `/upload_wavs` saves the uploaded file into sandbox cache and returns HTTP 501 with a clear deferred-extraction message.
- Passed: sandbox cache is mounted at `/user_data` from `refactor_sandbox/runtime/user_data`.
- Added `scripts/qa_audio_cache_routes.py` so the check can be repeated.

Deferred:
- Fresh feature extraction and objectifier execution are not wired yet.
- `/recalculate_features` currently reloads cached data only; heavy recomputation belongs in the next stage.

Recommendation:
- Move feature extraction orchestration next, but behind a service boundary. Start with a lightweight interface and dependency checks before importing librosa, aubio, crepe, MATLAB, Sonic Annotator, or Torch into the app shell.

## Stage 4a: Stimulus Listing Routes

Status: complete

Purpose:
- Move experiment sound-listing endpoints into `app/api/experiments.py`.
- Centralize static audio/stimulus discovery in `app/services/stimuli.py`.
- Reuse the same service from questionnaire routes for supplementary and noise-tonal listings.

Checks:
- Passed: `/get_sounds` returns 19 files from `static/indefinite_pitch`.
- Passed: `/get_sounds_training` returns 3 files from `static/indefinite_pitch/training`.
- Passed: `/get_sounds_shape` returns 30 files from `static/image_shape`.
- Passed: `/get_sounds_shape_training` returns 3 files from `static/image_shape/training`.
- Passed: `/get_sounds_texture` returns 30 files from `static/image_texture`.
- Passed: `/get_sounds_texture_training` returns 3 files from `static/image_texture/training`.
- Passed: `/get_sounds_supplementary` returns 30 files from `static/image_shape`.
- Passed: `/questionnaires/noise-tonal-preference/v1/stimuli` returns 6 paired stimuli.
- Added `scripts/qa_stimulus_routes.py` so the check can be repeated.

Recommendation:
- Move audio upload/cache routes next. That is the first high-risk stage because it touches file safety, content hashing, conversion, feature extraction calls, and frontend response compatibility.

## Stage 5: Feature Extraction Services

Status: complete, service boundary only

Purpose:
- Move librosa/aubio/crepe/mosqito/MATLAB/Sonic Annotator orchestration out of routes.
- Make external tools configurable.

Checks:
- Passed: added `FeatureExtractionService` interface in `app/services/features.py`.
- Passed: added `FeatureExtractionOptions`, `FeatureExtractionResult`, and typed dependency status.
- Passed: `/feature_extraction/status` reports import/tool/path availability without importing the heavy stack.
- Passed: fresh `/upload_wavs` goes through the service boundary and returns a structured HTTP 501 dependency report.
- Passed: `/recalculate_features` is dependency-gated and includes cached data plus requested options.
- Passed: cached flows from Stage 4 still pass.

Deferred:
- Concrete extractor implementation.
- Porting `run_serial_extraction`.
- MATLAB worker lifecycle.
- Sonic Annotator runner.
- Objectifier execution and model reuse.

Next implementation order:
- Add lightweight pure-Python feature extraction first, probably `librosa` spectral centroid/timestamps only.
- Then add optional extractor components one at a time: aubio, mosqito, Sonic Annotator, crepe, MATLAB, objectifier.
- Keep each component optional and visible in `/feature_extraction/status`.

## Stage 6: Objectifier And ML Model Reuse

Status: pending

Purpose:
- Move Wav2Vec/CLAP loading into shared model services.
- Avoid reloading models per request.

Checks:
- Objectifier output remains frontend-compatible.
- Model loading is lazy or lifespan-managed.
- Runtime logs show one model load per process, not one per request.

## Stage 5a: Librosa Spectral Centroid Slice

Status: complete

Purpose:
- Implement the first concrete feature extractor behind `FeatureExtractionService`.
- Keep the extractor lightweight: `timestamp` plus `spectral_centroid`.
- Save output as `features.json` in the existing `Song.features_per_timestamp` shape.

Checks:
- Passed: installed Stage 6 lightweight dependencies from `requirements-stage6-librosa.txt`.
- Passed: `/feature_extraction/status` reports required modules available when `librosa`, `numpy`, and `soundfile` are installed.
- Passed: `/recalculate_features` extracts real spectral-centroid features and writes `features.json`.
- Passed: fresh `/upload_wavs` extracts real spectral-centroid features and writes `features.json`.
- Passed: cached load/reuse behavior still works.

Notes:
- First extraction can take several seconds because librosa/numba warms up.
- Optional extractors remain unavailable until ported: aubio, crepe, mosqito, Sonic Annotator, MATLAB, objectifier.

Recommendation:
- Add an aubio or librosa pitch slice next, depending on whether you want fewer dependencies or closer parity with the current perceived-pitch workflow.

## Stage 5b: Librosa Pitch Slice

Status: complete

Purpose:
- Add pitch extraction without adding another native dependency.
- Use `librosa.pyin` to produce `f0_librosa`.
- Preserve JSON-safe output by converting unvoiced/failed pitch frames to `0.0`.

Checks:
- Passed: direct service smoke test detects a synthetic 440 Hz tone.
- Passed: `/recalculate_features` returns features with nonzero `f0_librosa` for QA tone audio.
- Passed: fresh `/upload_wavs` returns features with nonzero `f0_librosa` for QA tone audio.
- Passed: cached load/reuse behavior still works.

Notes:
- Very short, noisy, or unvoiced audio may produce `0.0` pitch values.
- This does not yet implement the old perceived-pitch blend with periodicity, CREPE, Sonic Annotator, or MIR weighted spectral centroid.

Recommendation:
- Add RMS/loudness proxy next using `librosa.feature.rms`; it is low-risk and helps the visualization respond to energy before introducing heavier psychoacoustic dependencies.

## Stage 5c: Librosa RMS/Loudness Proxy

Status: complete

Purpose:
- Add an energy feature with no new dependency.
- Use `librosa.feature.rms`.
- Emit both `rms` and `loudness` for frontend compatibility, with `loudness` acting as a temporary proxy until Mosqito loudness is ported.

Checks:
- Passed: direct service smoke test returns positive `rms` and `loudness` for synthetic sine audio.
- Passed: `/recalculate_features` returns positive `rms` and `loudness`.
- Passed: fresh `/upload_wavs` returns positive `rms` and `loudness`.
- Passed: existing cache/reuse/load behavior still works.

Notes:
- This is not psychoacoustic loudness yet.
- The `loudness` key is intentionally preserved as a compatibility alias.

Recommendation:
- Add zero-crossing rate or spectral flatness next if you want another lightweight feature, or begin porting Mosqito if psychoacoustic loudness/sharpness is the priority.

## Stage 7: Deployment And Tests

Purpose:
- Add dependency manifest, run instructions, smoke tests, and deployment settings.

Checks:
- Clean install is possible from the manifest.
- `pytest` smoke tests pass.
- Production paths are environment variables, not hard-coded.
