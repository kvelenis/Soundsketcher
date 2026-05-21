# Frontend Migration Plan

## Goal

Move the legacy template, CSS, and JavaScript into sandbox-owned modules in small verified slices. Keep the cached `20-37s.wav` example as the main non-objectifier regression target while the frontend is reorganized.

## Current Boundary

- Legacy assets remain mounted at `/static` for questionnaire/stimulus datasets only.
- Migrated sandbox assets are mounted at `/sandbox-static`.
- The full Soundsketcher page loads sandbox-owned CSS, JS, UI assets, and favicon from `refactor_sandbox/static`.
- The current QA fixture is `20-37s.wav`; it intentionally has `clusters: null` until objectifier is migrated.

## Completed Slices

### App Shell And Assets

1. Done: mounted `/sandbox-static` and moved patched frontend scripts into sandbox ownership.
2. Done: isolated header and examples menu markup, CSS, JS, and cached-example loading.
3. Done: moved active page layout styles into `app-layout.css`.
4. Done: removed commented legacy HTML from the sandbox index and split the page into partial includes.
5. Done: moved current UI media assets into sandbox static assets and routed templates/JS to `/sandbox-static/assets`.
6. Done: moved favicon ownership into sandbox static assets and redirected `/favicon.ico` to the sandbox asset.
7. Done: removed Firebase Analytics from the sandbox page.
8. Done: split `app_scripts.html` into smaller script include partials by responsibility while preserving script order.
9. Done: moved third-party CDN scripts, slider CSS, and font links into `partials/vendor_head.html`.

### CSS Ownership

1. Done: moved `mixer.css` into sandbox static assets.
2. Done: moved `sliders.css` into sandbox static assets.
3. Done: moved `accordion.css` into sandbox static assets.
4. Done: split legacy `style.css` into sandbox files:
   - `app-chrome.css`
   - `settings-panel.css`
   - `ui-overlays.css`
   - `app-shell.css`
5. Done: stopped loading legacy `style.css`.

### Startup And UI Controls

1. Done: added `app-init.js` as the sandbox startup entry point.
2. Done: added `SoundSketcherApp.onReady(...)` initializer registry.
3. Done: moved help, accordion, export, recording, slider, reset, preset, recalculate, auto-resketch, shell, sonificator, header, log/mel, mode, and polygon-mode startup onto `SoundSketcherApp.onReady(...)`.
4. Done: removed repeated per-file DOM-ready fallback duplication.
5. Done: moved upload drop-zone and hidden file input bindings from `handleDropAudios.js` onto `SoundSketcherApp.onReady(...)`.
6. Done: removed inline drag/drop handlers from the upload drop-zone template and bound them from sandbox JS.
7. Done: removed remaining inline click handlers from contact info and main play/stop controls.

### State, Cache, Upload, And Audio Flow

1. Done: centralized feature names, preset definitions, selected file state, and loading state.
2. Done: centralized cached list/load response shaping and local `/user_data` URL resolution.
3. Done: extracted upload hashing, cache checks, form construction, and upload POST helpers.
4. Done: extracted shared audio-to-sketch orchestration into `audio-visualization-orchestrator.js`.
5. Done: reordered script partials so settings and sonificator dependencies load before the shared audio visualization orchestrator and its upload/recording callers.
6. Done: removed dead legacy commented upload and visualization implementations from `handleDropAudios.js`.
7. Done: flattened upload/cached-file loading flow with shared busy-state and spinner helpers.
8. Done: moved recording processing calls from loose upload/visualization globals onto the `SoundSketcher.audio.*` namespace bridge.
9. Done: moved upload/cached-file flow state access in `handleDropAudios.js` onto `SoundSketcher.state` and `SoundSketcher.featureConfig`.
10. Done: moved recording flow state access in `recordingAudio.js` onto `SoundSketcher.state` and `SoundSketcher.featureConfig`.
11. Done: moved recalculate flow state/config access in `recalculateButton.js` onto `SoundSketcher.state` and `SoundSketcher.featureConfig`.
12. Done: moved auto-resketch state checks in `autoResketch.js` onto `SoundSketcher.state`.
13. Done: moved submit/redraw state access in `submitButton.js` onto `SoundSketcher.state`.
14. Done: moved playback shared audio state access in `playStopMainAudioFunctions.js` onto `SoundSketcher.state`.
15. Done: added a `SoundSketcher.playback` bridge and moved outside callers away from direct `isPlaying` / `togglePlayStop()` access.
16. Done: moved playback internals into `playback-controller.module.mjs` and stopped loading the classic playback script.
17. Done: moved mixer UI construction into `mixer.module.mjs` and stopped loading the classic mixer script.
18. Done: added a `SoundSketcher.sonification` bridge and moved visualization orchestration away from direct `initSynths()` access.
19. Done: moved sonification shared synth state into `sonification-state.js` under `SoundSketcher.sonificationState`.
20. Done: moved oscillator waveform configuration into `sonification-config.js` under `SoundSketcher.sonificationConfig`.
21. Done: moved shared sonification master-volume slider options into `sonification-config.js`.
22. Done: moved granular buffer processing and waveform drawing helpers into `granular-helpers.js`.
23. Done: moved granular upload/waveform UI binding into `granular-controller.js`.
24. Done: moved oscillator waveform/slider UI binding into `oscillator-controller.js`.
25. Done: moved shared sonification playback/cursor controls into `sonification-playback-controller.js`.
26. Done: moved synth event preparation into `sonification-data-prep.js`.
27. Done: moved shared synth scheduling/voice-management base class into `base-synth.js`.
28. Done: moved oscillator synth engine into `oscillator-synth.js`.
29. Done: moved granular synth engine into `granular-synth.js`.
30. Done: renamed the remaining synth initializer into `sonification-engine-orchestrator.js`.
31. Done: added `scripts/qa_granular_engine_boundary.py` to guard granular script order, UI hooks, and extracted engine contracts.
32. Done: browser-smoked the granular modal guard path after fixture load: `Granular Engine > Play` without an uploaded sample stays guarded and leaves the rendered sketch intact.
33. Done: added a granular `Use Current Audio` command that decodes the currently loaded playback audio and routes it through `processBuffer -> changeBuffer`.
34. Done: browser-smoked `Granular Engine > Use Current Audio` after loading `20-37s.wav`; the waveform container marks loaded and the sketch remains intact.
35. Done: added visible granular sample status and busy/success button states for `Use Current Audio`.
36. Done: moved the sonification engine orchestrator into `sonification-engine-orchestrator.module.mjs` and imported it from `main.module.mjs`.
37. Done: added `scripts/qa_sonification_module_boundary.py` to guard the current classic synth scripts plus module orchestrator boundary.
38. Done: moved the synth engine trio into ES modules: `base-synth.module.mjs`, `oscillator-synth.module.mjs`, and `granular-synth.module.mjs`.
39. Done: moved granular helper functions and synth data preparation into ES modules while preserving their compatibility bridges.
40. Done: moved sonification state/config and the three sonification UI controllers into ES modules.
41. Done: made the cached examples dropdown retryable after load failures and moved cache endpoints behind a base-aware resolver.
42. Done: added `sonification-utils.module.mjs` and moved sonification module usage of `dBToGain`, `clamp`, `map`, color conversion, and random sign helpers behind imports.

### Module Bridge

1. Done: added `SoundSketcher` namespace bridge for app, shared state, and audio clients.
2. Done: added `SoundSketcher.whenAudioClient(...)` readiness promises for asynchronously registered module-backed clients.
3. Done: added first real ES module, `audio-upload-client.module.mjs`.
4. Done: thinned and later removed the classic `audio-upload-client.js` bridge after `main.module.mjs` became the entrypoint.
5. Done: moved cache client implementation into `audio-cache-client.module.mjs`.
6. Done: thinned and later removed the classic `audio-cache-client.js` bridge after `main.module.mjs` became the entrypoint.
7. Done: moved visualization orchestration into `audio-visualization-orchestrator.module.mjs`.
8. Done: thinned and later removed the classic `audio-visualization-orchestrator.js` bridge after `main.module.mjs` became the entrypoint.
9. Done: added `main.module.mjs` as the first module entrypoint for audio cache/upload/visualization clients.
10. Done: removed classic audio cache/upload/visualization bridge script tags from the page.
11. Done: deleted the unused classic audio cache/upload/visualization bridge files.
12. Done: added `feature-config.module.mjs` as the first non-audio module while keeping the classic synchronous config script for compatibility.
13. Done: added `app-state.module.mjs` as a state bridge over the existing classic globals without creating a second `AudioContext`.
14. Done: added `sonification-engine-orchestrator.module.mjs` to `main.module.mjs` while preserving the `SoundSketcher.sonification.initSynths` bridge.
15. Done: imported the synth engine classes through the sonification orchestrator module while preserving the `SoundSketcher` class bridges for compatibility.
16. Done: imported `granular-helpers.module.mjs` and `sonification-data-prep.module.mjs` through `main.module.mjs` while preserving `SoundSketcher.granularHelpers` and global `prepareSynthData`.
17. Done: imported `sonification-config.module.mjs`, `sonification-state.module.mjs`, `oscillator-controller.module.mjs`, `granular-controller.module.mjs`, and `sonification-playback-controller.module.mjs` through `main.module.mjs`.
18. Done: imported `sonification-utils.module.mjs` through `main.module.mjs` and used it from the module-backed sonification classes/data prep.

### QA Tooling

1. Done: added `scripts/qa_frontend_runtime_boundary.py`.
2. Done: kept `scripts/qa_full_ui_fixture.py` passing for the preferred cached fixture.
3. Done: added `scripts/qa_granular_engine_boundary.py` for the granular engine extraction boundary.
4. Done: added `scripts/qa_sonification_module_boundary.py` for the sonification module bridge boundary.

## In Progress

### Drawing Orchestration

The drawing ownership split is underway and already has several completed extractions:

- Done: moved clamp/control reads into `drawing-controls.js`.
- Done: extracted feature/stat/config, visual-frame mapping, and feature-description formatting into `drawing-features.js`.
- Done: moved line/polygon SVG creation into `drawing-renderers.js`.
- Done: moved y-axis rendering into `drawing-axis.js`.
- Done: moved synth path-data bridging into `sonification-bridge.js`.
- Done: moved playback/grain visual helpers into `playback-visuals.js`.
- Done: moved pattern/path helper utilities into `drawing-patterns.js`.
- Done: moved draw-time derived feature and canvas context setup into `drawing-context.js`.
- Done: moved per-file drawing setup into `buildDrawingFileContext(...)`.
- Done: moved visual-frame feature mutation into `applyVisualFrameToFeature(...)`.
- Done: moved eligible line/polygon frame rendering into `renderEligibleVisualFrame(...)`.
- Done: moved per-file path-group/objectifier finalization into `finalizeDrawingFile(...)`.
- Done: moved after-loop drawing completion into `completeDrawingRun(...)`.
- Done: moved visual-frame option assembly into `buildVisualFrameOptions(...)`.
- Done: moved per-feature frame processing into `processDrawingFeatureFrame(...)`.
- Done: moved per-audio-file drawing orchestration into `processDrawingAudioFile(...)`.
- Done: moved run-level drawing setup into `buildDrawingRunContext(...)`.
- Done: moved run-level completion plumbing into `completeDrawingRunFromContext(...)`.
- Done: added `drawing-context.module.mjs` and thinned `drawing-context.js` into a module bridge.
- Done: added `drawing-features.module.mjs`, thinned `drawing-features.js` into a module bridge, and imported feature helpers from `drawing-context.module.mjs`.
- Done: added `drawing-renderers.module.mjs`, thinned `drawing-renderers.js` into a module bridge, and imported renderer helpers from `drawing-context.module.mjs`.
- Done: added `sonification-bridge.module.mjs`, thinned `sonification-bridge.js` into a module bridge, and imported synth path helpers from `drawing-context.module.mjs`.
- Done: added `drawing-axis.module.mjs`, thinned `drawing-axis.js` into a module bridge, and imported y-axis helpers from `drawing-context.module.mjs`.
- Done: added `drawing-patterns.module.mjs`, thinned `drawing-patterns.js` into a module bridge, and imported polygon pattern helpers from drawing renderer/context modules.
- Done: moved `drawVisualization()` orchestration into `drawing-visualization.module.mjs` and thinned `drawVisualisation.js` into a compatibility bridge.
- Done: moved per-file drawing setup and visual-frame option assembly into `drawing-file-context.module.mjs` while preserving the drawing-context bridge exports.
- Done: moved run-level drawing setup/completion into `drawing-run-context.module.mjs` while preserving the drawing-context bridge exports.
- Done: moved eligible frame rendering and file/objectifier finalization into `drawing-frame-renderer.module.mjs` while preserving the drawing-context bridge exports.
- Done: moved per-feature frame mapping and visual mutation into `drawing-feature-frame.module.mjs` while preserving the drawing-context bridge exports.
- Done: added `scripts/qa_polygon_drawing_fixture.py` to guard polygon render mode, generated SVG patterns, profiler mode, and path metadata.
- Done: added `scripts/qa_line_drawing_fixture.py` to guard default line render mode, first-load console cleanliness, mixer visibility, profiler mode, and path metadata.
- Done: added `scripts/qa_control_response_fixture.py` to guard line-length slider changes through a browser resketch and rendered SVG geometry comparison.
- Done: added `scripts/qa_deployment_readiness.py` and `npm run qa:readiness` as the local pre-staging verification gate.
- Done: added environment-driven settings overrides and `scripts/qa_settings_env.py` for staging path/config checks.
- Done: added `.env.staging.example`, `scripts/run_staging_preview.sh`, and `scripts/qa_staging_profile.py` for the staging launch profile.
- Done: ignored real env profiles and added a runner warning for unchanged example staging profiles.
- Done: added `docs/DEPLOYMENT_MANIFEST.md` to define staging include/exclude files, required cached example data, install steps, and post-deploy QA.
- Done: added `scripts/run_python.sh` so npm QA and staging launch work with `.venv`, conda, or `SOUNDSKETCHER_PYTHON`.
- Done: added staging start/stop/status/log scripts for safer server operations.
- Done: documented optional `SOUNDSKETCHER_PYTHON` for conda-based staging launch.
- Done: made `scripts/run_python.sh` source `.env.staging` when present so remote npm QA uses the configured staging Python.
- Done: parked the disabled SVG drag-selection experiment in `drawing-selection.js`.
- Done: removed large dead commented synth/objectifier experiments from the sandbox visualizer.

Still in progress:

- Continue reducing the drawing context modules and objectifier-specific rendering paths.
- Keep objectifier-specific rendering separated until objectifier migration resumes.

### Visualization Performance

- Done: added `drawing-profiler.js`, which records draw-stage timings and exposes the latest run on `window.__soundSketcherLastDrawProfile`.
- Done: cached the loudness gate value once per audio file instead of reading the slider inside every feature-frame iteration.
- Done: added `scripts/qa_drawing_boundary.py` to guard drawing script order, drawing module contracts, rendered SVG output, profiler metadata, and `pathData` population.
- Still in progress: reduce repeated DOM/canvas work in the heavy drawing loop.

### JS Module Migration

The module bridge is now active. The current strategy is:

1. Convert low-risk client files into module-backed bridges.
2. Keep compatibility globals during the transition.
3. Move callers to `SoundSketcher.*` readiness APIs.
4. Later introduce a single module entrypoint.

## Next

1. Thin or replace the classic `feature-config.js` and `app-state.js` once their remaining synchronous global callers move behind module imports or `SoundSketcher.state`.
2. Continue drawing and sonification splits after the low-risk module migration path is proven.

Done: moved per-audio-file drawing loop into `drawing-audio-file.module.mjs` while preserving the `drawing-context` bridge exports.

Done: moved file completion and objectifier cluster rendering into `drawing-file-finalizer.module.mjs`, leaving `drawing-frame-renderer.module.mjs` focused on individual eligible frames.

Done: added a sandbox objectifier backend slice that writes frontend-compatible cluster regions and a browser fixture for objectifier rendering.

Done: added objectifier data status UI and a guard that explains missing cluster data instead of leaving Objectifier mode blank.

Done: wrapped the legacy objectifier renderer in `objectifier.module.mjs` with `objectifier.js` as a compatibility bridge exposing `drawClusterOverlays`.

Done: replaced the equal-time sandbox objectifier placeholder with a sklearn feature-clustering backend while preserving the frontend cluster contract.

Done: wired the real legacy Wav2Vec/CLAP objectifier behind the sandbox objectifier service, with the sklearn feature-clustering backend retained as a fallback for too-short files or legacy runtime failures.

## Remaining Legacy Dependencies

These files still load from the legacy `/static` mount on the full sandbox page:

- CSS: none.
- Header/media assets: none.
- Favicon: none.
- Early app helpers: none.
- Core flow helpers: none.
- Audio/UI helpers: none.

The `/static` mount is still intentionally used by questionnaire/stimulus APIs, for example `noise_tonal_preference_samples`. That data-serving boundary should be migrated separately from the main Soundsketcher page.

## QA Rule

Every slice must keep this path working:

1. Open `/`.
2. Choose `Examples > 20-37s.wav`.
3. Confirm the spinner clears.
4. Confirm the sketch renders.
5. Confirm the audio URL is requested from `/user_data/...`, not `/app1/...`.

When browser control is unavailable, run:

```bash
.venv/bin/python scripts/qa_frontend_runtime_boundary.py
.venv/bin/python scripts/qa_full_ui_fixture.py
```
