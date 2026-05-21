# SoundSketcher Refactor Handoff

Last updated: 2026-05-19

This is the living project report for the SoundSketcher refactor sandbox. Update this file after every meaningful implementation step so Codex, Claude Code, and any future collaborator can quickly understand what changed, what is working, and what should happen next.

### 2026-05-19 (MOSQITO/libexpat Upload Crash Fix)

**What changed:**

- Removed MATLAB worker pre-warm from app startup.
- Reason: MATLAB can load its bundled `libexpat` before MOSQITO/matplotlib imports Python `pyexpat`, causing this upload-time crash:
  `undefined symbol: XML_SetAllocTrackerActivationThreshold`.
- The feature extraction pipeline already runs MOSQITO before MATLAB, so starting MATLAB lazily preserves the safe library load order.
- This also keeps `/indefinite_pitch` and other showcase routes lighter at startup.

**Follow-up:**

- Re-run the ES_rain upload/recalculate path and verify the job moves past `mosqito` into `matlab`/`derived`.

### 2026-05-19 (Feature/Objectifier Percentage Progress)

**What changed:**

- Added weighted milestone progress to background feature extraction jobs. `/feature_extraction_job_status` now returns `progress`, `stage`, `message`, and `elapsed_seconds` alongside the existing job fields.
- Added weighted milestone progress to objectifier jobs. `/objectifier_status` now returns the same progress fields while a job is queued/running.
- The frontend status strip now renders a percentage label and progress bar while feature extraction or objectifier processing is active.
- Added a floating feature extraction progress strip near the upload spinner so users can see the percentage even when the settings panel is closed or obscured.
- Replaced the confusing upload toast during async extraction: pending jobs now say "Feature extraction started" instead of "Audio files processed successfully."
- The large central spinner is hidden while the floating progress strip is active; a smaller spinner now lives inside the progress strip to show motion without obscuring the canvas.
- Pending uploads no longer trigger an initial empty visualization. The canvas is drawn once the async feature job completes, avoiding the white-screen blink/double-render effect.
- When feature extraction reaches `done`, the floating status now changes to "Rendering sketch" and remains visible until `visualizeAllFiles()` finishes.
- Fixed the restart/reload case for completed feature jobs: if the worker no longer remembers a job but `features.json` exists, `/feature_extraction_job_status` returns a synthetic `done` state with `progress: 100` instead of `unknown`.
- Bumped cache keys for `app-layout.css`, `app-chrome.css`, `objectifier-status.js`, and `feature-extraction-status.js`.
- Updated `scripts/qa_feature_extraction_status_fixture.py` to assert that visible progress text and both progress bars render before the async job completes.
- Updated the indefinite-pitch showcase QA selector so it accepts the current lightbox button implementation for bin plots, not only old anchor links.

**Milestone model:**

- Feature extraction: queued -> loading_audio -> sonic_annotator -> librosa -> aubio -> crepe -> mosqito -> matlab -> derived -> writing -> objectifier -> done/failed.
- Objectifier: queued -> loading -> extracting -> writing -> done/failed.

**Notes:**

- This is percentage-by-stage, not sample-accurate progress. It gives users honest feedback about where the backend is spending time without needing to instrument every third-party extractor.
- The objectifier stage is still mostly coarse because `build_objectifier_payload()` is a single long call internally.

**Deployment status:**

```text
live on staging
staging restarted on 127.0.0.1:5013
focused feature-status fixture passing
deployment readiness passing
```

## Project Location

Local workspace:

```text
/Users/konstantinosvelenis/Documents/Codex/soundsketcher_audit/refactor_sandbox
```

Staging server workspace:

```text
/mnt/ssd1/kvelenis/soundsketcher-staging/refactor_sandbox
```

Staging browser URL through SSH tunnel:

```text
http://127.0.0.1:5013/
```

## Current Status

The staging app is running and the readiness QA passes.

The objectifier mode is now connected to the real legacy-derived backend path, can process real audio, renders detected regions, supports region/cluster inspection, selected-region playback, cluster soloing, and manual cluster naming.

Manual cluster names now persist into each cached audio file's `objectifier.json` under `cluster_labels`. Cached audio reloads and objectifier status responses return those labels to the frontend.

Cluster auditioning is live on staging: selecting a cluster now exposes `Play cluster`, which auditions every region in that cluster in timeline order. Cluster visibility controls are also live: selected clusters can be hidden, isolated with show-only mode, or reset back to the full view.

Local work in progress: UI-only region deletion/restoration has been implemented locally but still needs staging sync/restart/QA once SSH is reachable.

## Environment Notes

The staging server uses this environment path before Node/Python commands:

```bash
PATH=/mnt/ssd1/kvelenis/conda-envs/soundsketcher-staging/bin:$PATH
```

Common staging commands:

```bash
cd /mnt/ssd1/kvelenis/soundsketcher-staging/refactor_sandbox
scripts/staging_stop.sh
PATH=/mnt/ssd1/kvelenis/conda-envs/soundsketcher-staging/bin:$PATH scripts/staging_start.sh .env.staging
scripts/staging_status.sh .env.staging
```

Readiness check:

```bash
PATH=/mnt/ssd1/kvelenis/conda-envs/soundsketcher-staging/bin:$PATH npm run qa:readiness -- --base-url http://127.0.0.1:5013
```

## What Has Been Done

### Staging Workflow

- Confirmed the refactor sandbox runs on the server.
- Set up access through `http://127.0.0.1:5013/`.
- Verified server restart and browser visibility.
- Added/used deployment readiness checks.
- Confirmed objectifier browser fixture passes as part of readiness QA.

### Legacy Reference

- Created a clean legacy reference copy for migration work.
- Excluded user data, response folders, uploads, runtime data, large audio dumps, duplicated old static/template copies, and large third-party folders.
- Purpose: use legacy as a reference source, not as the active runtime.

### Objectifier Backend

- Connected the refactor sandbox to the real legacy Wav2Vec/CLAP-style objectifier path.
- Added support for real objectifier JSON output.
- Added async/background objectifier processing so the UI does not block.
- Added `legacy_fast` mode.
- Skipped slow CLAP semantic-label steps in fast mode.
- Tuned the cluster-count search so long audio is faster.
- Confirmed fast objectifier works on real audio.

Example observed run:

```text
file: sequence-with-all-sounds-mixed.wav
mode: legacy_fast
extractor: legacy_wav2vec_clap_objectifier_legacy_fast
regions: 42 after post-processing
clusters: 10 in one inspected run
elapsed: around 19-33 seconds depending run/mode
```

### Objectifier Region Quality

- Added region smoothing/post-processing.
- Removed very tiny fragments.
- Reduced confusing region overlaps.
- Made detected regions more discrete and usable for sketching.

### Objectifier Visualization

- Removed the old whole-cluster blob that made separated regions look merged.
- Now draws per-region shapes.
- Added a bottom timeline layer for detected regions.
- Added a cluster legend.
- Added selected-region info panel.
- Added selected-cluster info panel.
- Added click interactions:
  - click region or timeline block to inspect
  - click cluster legend row to solo/highlight a cluster
  - click the same cluster again to clear solo
- Added objectifier detail modes:
  - `Simple`
  - `Balanced`
  - `Detailed`

### Region Playback

- Added `Play selected region` in the selected-region panel.
- Playback uses the selected region start/end time.
- Fixed cache-busting so updated playback/objectifier modules load correctly in the browser.

### Manual Cluster Names

- Added manual cluster naming from the objectifier cluster panel.
- Workflow:
  - click a cluster in the legend
  - click `Rename cluster`
  - enter a custom name
  - legend, cluster panel, region panel, and tooltips update immediately
- Clearing the name returns the label to `Cluster N`.
- Labels are written back to `objectifier.json` through `POST /objectifier_cluster_labels`.
- Labels are returned by `/load_cached_audio` and `/objectifier_status`.
- The frontend hydrates labels when rendering objectifier results and keeps the active browser data object in sync.

### Cluster Name Persistence

- Added `cluster_labels` support to cached objectifier data.
- Added backend route:

```text
POST /objectifier_cluster_labels
```

Expected JSON body:

```json
{
  "filename": "audio.wav",
  "audio_hash": "cachehash",
  "cluster_labels": {
    "1": "metallic hits"
  }
}
```

- Empty label values are removed before writing.
- QA now verifies:
  - fixture labels round-trip through `/load_cached_audio`
  - fixture labels appear in `/objectifier_status`
  - label updates persist to `objectifier.json`

### Cluster Auditioning

Implementation added:

- The cluster selection panel now includes `Play cluster`.
- Cluster playback uses the existing original-audio segment playback path.
- Regions in the selected cluster play sequentially in timeline order.
- A short gap is inserted between regions.
- The active region is highlighted during cluster playback.
- Playback controller now exposes:

```text
playSegment(...)
stopSegment(...)
```

- `playSegment(...)` now resolves when a segment finishes or is stopped, which makes sequential playback reliable.

Deployment status:

```text
live on staging
staging restart complete
readiness QA passing
```

### Cluster Visibility Controls

Implementation added:

- The cluster selection panel now includes `Hide cluster`.
- Hidden clusters are removed from the region shapes, timeline blocks, timeline labels, and hit targets.
- Hidden cluster legend rows are dimmed but remain clickable.
- The panel includes `Show only cluster`, which isolates the selected cluster and dims/hides all others.
- The panel includes `Show all clusters` / `Reset visibility` when a visibility filter is active.
- Visibility state is session/UI-only and does not modify the saved objectifier JSON.

Deployment status:

```text
live on staging
staging restart complete
readiness QA passing
```

### Region Delete / Restore

Local implementation added:

- Selecting a region now exposes `Delete region`.
- Deleted regions are hidden from:
  - organic region shapes
  - transparent hit targets
  - timeline blocks
  - timeline labels
- Deleted regions are skipped by cluster auditioning.
- When any region is deleted, panels expose `Restore deleted regions`.
- Restore brings all deleted regions back.
- This is currently UI/session-only and does not modify saved `objectifier.json`.

Deployment status:

```text
implemented locally
staging sync pending because SSH timed out
staging restart pending
readiness QA pending
```

## Important Files

Objectifier frontend renderer:

```text
static/js/objectifier.module.mjs
```

Objectifier frontend module bridge/cache-bust:

```text
static/js/objectifier.js
```

Template script cache-bust:

```text
templates/partials/scripts/drawing_scripts.html
```

Objectifier/backend service area:

```text
app/services/objectifier.py
```

QA scripts:

```text
scripts/qa_deployment_readiness.py
scripts/qa_objectifier_fast_mode.py
scripts/qa_audio_cache_routes.py
```

## Current Known Limitations

- Objectifier output still partly follows legacy-shaped JSON.
- Fast mode skips CLAP semantic labels by design.
- Region drawing is better, but still needs improved layout options for analytical work.
- There is no complete saved objectifier session model yet.
- Region editing is not implemented yet.

## Recommended Next Steps

### 1. Save Objectifier Sessions

Create a proper objectifier session model containing:

- source audio filename/id
- extractor mode
- elapsed time
- clusters
- regions
- manual cluster labels
- selected/hidden clusters
- optional user notes

This would let users reload previous objectifier work without reprocessing audio.

### 2. Improve Region Drawing Layout

Potential display modes:

- `Timeline`: discrete analytical region blocks.
- `Sketch`: more organic SoundSketcher visual style.
- `Hybrid`: current direction, but cleaner.

Improvements:

- clearer separation between adjacent/overlapping regions
- better vertical placement by cluster or feature
- stronger selected-region highlight
- optional cluster lanes
- optional feature-sketch overlay

### 3. Add Region Editing

Useful controls:

- split region
- merge neighboring regions
- rename region
- assign region to another cluster
- delete region
- adjust start/end boundaries

This would make objectifier mode much more useful as an interactive annotation/sketching tool.

### 4. Add Cluster Filtering And Auditioning

Next useful cluster actions:

- play all regions in selected cluster
- mute/hide cluster
- show only one cluster
- export selected cluster
- compare clusters by average features

### 5. Define A Clean Objectifier JSON Schema

Suggested target shape:

```json
{
  "audio": {},
  "mode": "legacy_fast",
  "elapsed_seconds": 19.66,
  "clusters": [
    {
      "id": "3",
      "label": "metallic hits",
      "color": "#...",
      "features": {},
      "regions": [
        {
          "id": "r12",
          "start_time": 1.24,
          "end_time": 2.01,
          "duration": 0.77,
          "features": {},
          "shape": {}
        }
      ]
    }
  ]
}
```

### 6. Reintroduce Semantic Labeling Carefully

Recommended approach:

- run fast region/cluster extraction first
- make semantic labeling an optional second pass
- cache text embeddings
- cache audio embeddings
- allow manual user labels to override generated semantic labels

### 7. Expand UI QA

Add browser-level checks for:

- objectifier renders regions
- legend exists
- clicking a cluster solos it
- clicking a region opens the region panel
- play button appears for selected regions
- rename cluster updates legend text
- no JavaScript console errors

## Update Log

### 2026-05-18 (Indefinite-pitch showcase restoration)

**What changed:**

- Restored `/indefinite_pitch` in the refactor sandbox as a showcase page rather than a participant data-collection workflow.
- Added `templates/indefinite_pitch.html` with:
  - playable study/training stimulus browser
  - sine-tone matching slider
  - JASA Express Letters publication context
  - DOI link for `10.1121/10.0043569`
  - study snapshot and core findings summary
- Added `scripts/qa_indefinite_pitch_showcase.py`.
- Added npm script `qa:indefinite-pitch`.
- Added the showcase fixture to deployment readiness.

**Asset dependency:**

- `/static/indefinite_pitch` must exist under the active `SOUNDSKETCHER_STATIC_DIR`.
- Staging should copy this folder from `/mnt/ssd1/kvelenis/soundsketcher/static/indefinite_pitch`.

**Deployment status:**
```text
staging deployed and verified on http://127.0.0.1:5013/indefinite_pitch
focused QA passed: indefinite pitch showcase -> ok (19 study sounds, 3 training sounds)
full readiness passed: deployment readiness -> ok
```

**Follow-up polish:**

- Matched the sine slider interaction to the legacy experiment:
  - logarithmic slider control range (`0..1000`) mapping to `40..6000 Hz`
  - press/hold on the slider starts the sine tone
  - slider release stops the sine tone
  - sine frequency updates continuously while dragging
  - legacy loudness compensation curve is applied to the sine gain
- Added the no-match control from the legacy task:
  - English: `I cannot find a match`
  - Greek: `Δεν βρίσκω αντιστοίχιση`
  - toggles a selected state and clears when the slider is touched
- Added an English/Greek language switch for the showcase text and controls.
- Updated `qa_indefinite_pitch_showcase.py` to check:
  - legacy logarithmic slider range
  - Greek language switch
  - Greek no-match text
  - DOI and stimulus rendering remain intact.

**2026-05-18 additional fidelity/results update:**

- Added the legacy sample loop control to the showcase.
- Added a sine-volume slider with a live percentage readout.
- Hardened sine stopping:
  - sample playback stops the sine first
  - document pointer release, pointer cancel, pointer leave, and window blur stop the sine
  - stale oscillator/gain nodes are disconnected safely
- Added keyboard `A` support for sample play/stop, matching the legacy task.
- Copied selected real result figures into staging:
  - `/static/indefinite_pitch/results/loudness_centroid_vs_pitch.png`
  - `/static/indefinite_pitch/results/mixedlm_marginal.png`
  - `/static/indefinite_pitch/results/pitch_matches_per_sound.png`
  - `/static/indefinite_pitch/results/no_match_per_sound.png`
- Added bilingual methodology and result-figure sections to `/indefinite_pitch`.
- Updated QA to check loop, sine-volume, bilingual no-match, slider range, and result figure presence.
- Verification:
  - focused QA passed: `indefinite pitch showcase -> ok (19 study sounds, 3 training sounds)`
  - full readiness passed: `deployment readiness -> ok`

**2026-05-18 requested figure/card refinement:**

- Reordered the result figures in `/indefinite_pitch`:
  1. `boxplot_per_sound_vertical.png`
  2. `Figure1_centroid_vs_pitch_errorbars_labeled.png`
  3. `loudness_centroid_vs_pitch.png`
  4. `Figure2_mixedlm_loud_only_partial_effect.png`
- Added the requested compact stimulus labels (`sn1`, `sn2`, `543`, `703`, `916`, `196`, `987`, `146`, `rai`, `frg`, `crt`, `low`, `med`, `hig`, `spr`, `trc`, `sor`, `was`, `noi`) to the study stimulus cards.
- Added an `Open bin plot` / `Άνοιγμα bin plot` link to each study stimulus card, mapped to:
  - `/static/indefinite_pitch/results/bins_plots/*.png`
- Copied missing static assets to staging before the verification blocker:
  - requested result figures
  - `results/bins_plots/`
- Staging was restarted successfully after the template sync.
- Verification note: server-side SSH QA and in-app browser verification were blocked by the app approval/browser policy after restart, so the final QA pass is pending.

### 2026-05-17

- Created this living handoff file.
- Current state includes real objectifier backend integration, fast mode, region post-processing, per-region visualization, timeline, cluster legend, selection panels, selected-region playback, cluster soloing, and manual cluster naming.
- Added persistent objectifier cluster labels:
  - `cluster_labels` stored in cached `objectifier.json`
  - `/objectifier_cluster_labels` backend route
  - `/load_cached_audio` and `/objectifier_status` return labels
  - frontend hydrates/saves labels from the active objectifier result
  - `scripts/qa_audio_cache_routes.py` verifies label persistence
- Implemented and deployed cluster auditioning:
  - `Play cluster` button in objectifier cluster panel
  - sequential region playback for selected cluster
  - active-region timeline highlight
  - playback controller segment completion/stop API
  - deployed to staging and verified with full readiness QA
- Implemented and deployed cluster visibility controls:
  - `Hide cluster`
  - `Unhide cluster`
  - `Show only cluster`
  - `Show all clusters` / `Reset visibility`
  - visibility applies to region shapes, hit targets, timeline blocks, and timeline labels
  - deployed to staging and verified with full readiness QA
- Implemented UI-only region delete/restore locally:
  - `Delete region`
  - `Restore deleted regions`
  - hidden regions are omitted from shapes, hit targets, timeline, labels, and cluster auditioning
  - staging deployment pending because SSH timed out

### 2026-05-18 (session persistence — Step 1 complete)

**What was done:**

- Persisted cluster visibility state (`hidden_clusters`, `only_cluster`) to `objectifier.json`
  - New backend route `POST /objectifier_visibility` — same pattern as `/objectifier_cluster_labels`
  - `app/storage/cache.py` reads `hidden_clusters`, `only_cluster`, `notes` from objectifier.json on load
  - `load_cached_audio` and `objectifier_status` responses now include all three fields
  - Frontend `hydrateObjectifierVisibility` restores visibility state from loaded data on render
  - `toggleObjectifierClusterHidden`, `toggleObjectifierOnlyCluster`, `showAllObjectifierClusters` now call `persistClusterVisibility` after every change
- Added session notes (`notes`) to `objectifier.json`
  - New backend route `POST /objectifier_notes`
  - Frontend `persistObjectifierNotes` saves notes on user input
  - "Add session notes" / "Edit session notes" button added to cluster selection panel
- QA verified: all 13 checks pass including new `/objectifier_visibility -> ok` and `/objectifier_notes -> ok`
- Deployed to staging and verified

**Deployment status:**
```text
live on staging
staging restart complete
QA passing (13/13 checks)
```

**What is complete (objectifier session state):**

| Field | Persisted |
|---|---|
| cluster regions & colors | ✓ |
| cluster_labels (manual names) | ✓ |
| deleted_regions | ✓ |
| region_overrides (boundary edits) | ✓ |
| hidden_clusters | ✓ (new) |
| only_cluster | ✓ (new) |
| notes | ✓ (new) |

- Added boundary reset controls (discovered gap: no way to undo boundary adjustments)
  - `resetObjectifierRegionBoundaries` — removes override for selected region, re-renders
  - `resetAllObjectifierBoundaries` — clears all boundary overrides at once, re-renders
  - "Reset to original boundaries" button shown in region panel when override exists for that region
  - "Reset all boundary edits" button shown whenever any override exists (mirrors "Restore deleted regions")
  - Both call `persistObjectifierRegionEdits` so the cleared state is saved immediately
  - Deployed to staging, QA 13/13 passing

**Next step:** ~~Clean objectifier JSON schema (Step 2)~~ — done (see below). ~~CLAP semantic labeling as optional second pass (Step 3)~~ — done (see below).

---

### 2026-05-18 (objectifier JSON schema v2 — Step 2 complete)

**What was done:**

- Defined and implemented objectifier JSON schema v2
- New v2 structure:
  ```json
  {
    "schema_version": 2,
    "audio": { "filename": "...", "hash": "..." },
    "extraction": { "mode": "legacy_fast", "extractor": "...", "elapsed_seconds": null },
    "clusters": [ { ..., "semantic_labels": [], "regions": [ { ..., "duration": 1.0 } ] } ],
    "user_edits": {
      "cluster_labels": {}, "deleted_regions": [], "region_overrides": {},
      "hidden_clusters": [], "only_cluster": null, "notes": ""
    }
  }
  ```
- Key improvements over v1:
  - `waveform` field dropped — removes ~35 MB of bloat per cached file
  - `sample_rate` and `similarities` dropped (unused)
  - User edits grouped under `user_edits` (clean separation from extraction output)
  - `audio_file` → `audio: {filename, hash}` (carries hash for traceability)
  - `general_info` → `extraction: {mode, extractor, elapsed_seconds}` (cleaner naming)
  - `semantic_labels: []` added to each cluster (ready for CLAP in Step 3)
  - `duration` added to each region (derived, useful)
- Backward compat: v1 files (no `schema_version`) still load correctly via `objectifier_user_edits()` helper
- All 4 user-edit routes (`/objectifier_cluster_labels`, `/objectifier_region_edits`, `/objectifier_visibility`, `/objectifier_notes`) detect schema version and write to the right section
- QA: 14/14 checks pass including new `v1 schema backward-compat -> ok`
- Deployed to staging, readiness QA passing

**Deployment status:**
```text
live on staging
staging restart complete
QA passing (14/14 checks)
```

**Next step:** ~~CLAP semantic labeling as optional second pass (Step 3)~~ — done (see below).

---

### 2026-05-18 (CLAP semantic labeling — Step 3 complete)

**What was done:**

- Created `app/services/semantic_labels.py`:
  - 23 short audio vocabulary terms (percussion, metallic, bright, dark, short, long, noise, tonal, vocal, melodic, attack, resonant, impact, texture, rhythmic, sparse, dense, harmonic, breath, click, reverberant, dry, soft)
  - `_load_clap(settings)` — loads HuggingFace `ClapModel` + `ClapProcessor`, optionally loads local checkpoint weights, caches model in-process
  - `_text_embeddings(settings)` — computes + caches normalized text embeddings for all 23 vocabulary terms
  - `_audio_embedding_for_segment(audio_path, start_time, end_time, ...)` — extracts CLAP audio embedding for a time slice using soundfile
  - `compute_semantic_labels(audio_path, clusters, settings, top_k=3)` — computes centroid embedding per cluster from its region segments, scores against all 23 terms, returns top 3 as `[{"term": "percussion", "score": 0.87}, ...]`
  - `SemanticLabelsService` — `is_available()` + `dependency_report()` (checks torch, transformers, soundfile, checkpoint file)
  - `SemanticLabelsJobQueue` — same `ThreadPoolExecutor` + lock pattern as `ObjectifierJobQueue`; skips re-labeling if labels already present (unless `force=True`)
- Updated `app/core/config.py`:
  - Added `clap_checkpoint_path: Path | None = None` (defaults to the local checkpoint at `aux_models/music_speech_audioset_epoch_15_esc_89.98.pt`)
  - Added `clap_model_name: str = "laion/larger_clap_music_and_speech"`
  - Both configurable via `SOUNDSKETCHER_CLAP_CHECKPOINT_PATH` / `SOUNDSKETCHER_CLAP_MODEL_NAME` env vars
- Updated `app/api/audio.py`:
  - Added `SemanticLabelsJobQueue` and `SemanticLabelsService` instances at module level
  - Added `GET /semantic_labels_status` — returns `{available, job, filename, hash}`
  - Added `POST /objectifier_semantic_labels` — enqueues a labeling job; returns 503 if CLAP deps unavailable
- Updated `static/js/objectifier.module.mjs`:
  - `clusterSemanticLabels(fileIndex, clusterLabel)` — looks up `semantic_labels` from loaded cluster data
  - Cluster selection panel now shows semantic labels inline: `Semantic labels:  percussion 87%  ·  impact 76%  ·  metallic 64%`
  - `Generate semantic labels` / `Regenerate semantic labels` button in cluster panel
  - `triggerObjectifierSemanticLabels(svgContainer, buttonElement)` — POSTs to `/objectifier_semantic_labels`, polls `/semantic_labels_status` every 2 s, re-renders via `submitButton.click()` when done
- Updated `scripts/qa_audio_cache_routes.py`:
  - Added `/semantic_labels_status -> ok` check
  - QA result: **15/15 checks pass**

**New backend routes:**

```text
GET  /semantic_labels_status?audio_hash=...&filename=...
POST /objectifier_semantic_labels  body: {audio_hash, filename, force?}
```

**Deployment status:**
```text
live on staging
staging restart complete
QA passing (15/15 checks)
readiness QA passing
```

**Note:** CLAP inference requires `torch`, `transformers`, and `soundfile` to be installed in the conda env. The `is_available()` check gates gracefully — if deps are missing, the route returns 503 and the UI button still appears (but will show a warning in the console).

---

### 2026-05-18

- Promoted objectifier region delete/restore from UI-only state toward persistent edits:
  - stable region edit keys use `cluster::start::end`
  - `deleted_regions` is stored in cached `objectifier.json`
  - `region_overrides` stores manual boundary edits by stable region key
  - `/objectifier_region_edits` backend route saves deleted-region keys
  - `/objectifier_region_edits` also saves boundary overrides with `start_time` and `end_time`
  - `/load_cached_audio` and `/objectifier_status` return deleted-region keys
  - `/load_cached_audio` and `/objectifier_status` return boundary overrides
  - frontend hydrates deleted regions when objectifier data renders
  - frontend applies boundary overrides before assigning frames and drawing regions
  - frontend persists deletes and restore-all actions back to the cache
  - selected regions now expose `Adjust boundaries`
  - `scripts/qa_audio_cache_routes.py` verifies deleted-region round trip and persistence
  - `scripts/qa_audio_cache_routes.py` verifies boundary override round trip and persistence
  - deployed to staging and verified with focused cache-route QA plus full readiness QA

---

### 2026-05-18 (Full legacy feature extraction pipeline — Step 4 complete)

**What was done:**

- Installed all required Python packages into the staging conda env:
  - `aubio`, `mosqito` — via pip
  - `crepe` — via pip with `--no-build-isolation` and `TMPDIR` on SSD (root partition was full)
  - `tensorflow-cpu` — required by crepe; installed with `TMPDIR` on SSD
  - `matlabengine==24.2.2` — matches MATLAB R2024b; installed via pip

- Created `app/services/legacy_features.py` — full replication of `run_serial_extraction` from `legacy_reference/main_with_event_detection.py`:
  - `_MatlabWorker` — singleton daemon thread; starts MATLAB engine once per process, processes jobs serially via queue; dies with the server process
  - `_run_matlab_mir(eng, audio_path, timestamps)` — calls `roughnessTimeSeries.m` via engine, returns 4 MIR features interpolated to target timestamps
  - `_zero_mir_results(n)` — fallback zeros if MATLAB fails
  - `_run_sonic_annotator(wav_path, output_dir, sonic_annotator_dir)` — subprocess call, writes CSV to tempdir
  - `_read_sonic_annotator_csvs(output_dir, timestamps)` — reads `yin_periodicity` CSV, interpolates
  - `_extract_librosa(y, sr, n_fft, hop_length)` — spectral centroid, pyin f0, fullness (Alluri & Toiviainen)
  - `_extract_aubio(y, sr, n_fft, hop_length)` — YIN pitch with median filter
  - `_extract_crepe(y, sr, timestamps)` — CREPE pitch + confidence, viterbi, "tiny" model
  - `_extract_mosqito(y, sr, timestamps, n_fft, hop_length)` — loudness (zwst) + sharpness (DIN), resampled to 48 kHz internally
  - `run_legacy_extraction(wav_path, settings, ...)` — orchestrates all steps, derives 3 composite features, returns list of per-frame dicts with all 17 keys
  - `legacy_dependency_report(settings)` / `legacy_is_available(settings)` — checks 8 Python packages + sonic_annotator binary + matlab_toolbox dir + matlab_scripts dir

- Updated `app/core/config.py`:
  - Added `matlab_scripts_dir: Path | None = None` to `Settings`
  - Added `matlab_scripts_dir=_env_optional_path("MATLAB_SCRIPTS_DIR")` to `load_settings_from_env()`

- Updated `.env.staging`:
  - `SOUNDSKETCHER_MATLAB_TOOLBOX_DIR="/usr/local/MATLAB/R2024b/mirtoolbox-main/MIRToolbox"`
  - `SOUNDSKETCHER_MATLAB_SCRIPTS_DIR="/mnt/ssd1/kvelenis/soundsketcher-staging/legacy_reference/soundsketcher_aux_scripts"`
  - `SOUNDSKETCHER_SONIC_ANNOTATOR_DIR="/mnt/ssd1/kvelenis/soundsketcher/sonic-annotator-1.6-linux64-static"`

- Updated `app/services/features.py`:
  - Replaced librosa-only extraction with a call to `run_legacy_extraction`
  - `dependency_report()` now delegates to `legacy_dependency_report()`
  - `is_available()` now delegates to `legacy_is_available()`
  - Removed all the fake-aliased feature stubs (`_build_feature_frame`, `_extract_pyin_f0`, etc.)
  - Extractor label updated to `legacy_full_pipeline_v1`

**Feature schema (17 keys per frame):**

| Key | Source |
|---|---|
| timestamp | librosa frames_to_time |
| spectral_centroid | librosa |
| weighted_spectral_centroid | MATLAB roughnessTimeSeries (Loudness_SC_Hz) |
| crepe_f0 | CREPE viterbi |
| yin_f0_librosa | librosa pyin (fallback: aubio YIN) |
| perceived_pitch_f0_or_SC_weighted | derived: blend of crepe_f0 and weighted_SC |
| loudness | mosqito loudness_zwst |
| loudness_periodicity | derived: loudness × (1 − yin_periodicity) |
| loudness_pitchConf | derived: loudness × (1 − crepe_confidence) |
| sharpness | mosqito sharpness_din |
| mir_mps_roughness | MATLAB MPS roughness |
| mir_sharpness_zwicker | MATLAB Zwicker sharpness |
| mir_roughness_vassilakis | MATLAB Vassilakis roughness |
| fullness | librosa (Alluri & Toiviainen 50–200 Hz energy) |
| yin_periodicity | Sonic Annotator YIN (thresholded at 0.3) |
| crepe_confidence | CREPE confidence |
| raw_periodicity | Sonic Annotator YIN (raw) |

**Paths used:**

```text
MATLAB binary:         /usr/local/MATLAB/R2024b/bin/matlab
MIRToolbox:            /usr/local/MATLAB/R2024b/mirtoolbox-main/MIRToolbox
TimbreToolbox:         /usr/local/MATLAB/R2024b/TimbreToolbox-R2021a-main (not used directly)
MATLAB scripts:        legacy_reference/soundsketcher_aux_scripts/
  roughnessTimeSeries.m
  SC_Loudness_single.m
Sonic Annotator:       /mnt/ssd1/kvelenis/soundsketcher/sonic-annotator-1.6-linux64-static/
  sonic-annotator  (binary)
  periodicity.n3   (transform descriptor)
```

**Verified end-to-end** with `20-37s.wav` (732 frames, sr=44100):
- All 13 dependency checks green
- All 17 features populated (non-zero real values from real tools)
- MATLAB starts once per server process, reused across requests

**QA: 15/15 checks pass** (no new QA routes added; feature extraction verified separately).

**Deployment status:**
```text
staged — server restart required to activate MATLAB worker thread
QA: 15/15 pass (existing routes unchanged)
```

**Note:** First feature extraction request starts MATLAB (adds ~5s one-time delay). Subsequent requests reuse the running engine. MATLAB process terminates when the uvicorn worker exits.

---

### 2026-05-18 (Async feature extraction, MATLAB pre-warming, canvas verification — Steps 5 + 5b complete)

**What changed:**

**Step 5 — Async upload/recalculate (non-blocking)**

Previously `upload_wavs` and `recalculate_features` blocked the HTTP response for 30–60 s while feature extraction ran. Both endpoints now return immediately (~10 ms) with a `feature_job` status object; the client polls for completion.

New file: `app/services/feature_extraction_jobs.py`
- `FeatureExtractionJobState` dataclass (`job_id`, `status`, `audio_path`, timestamps)
- `FeatureExtractionJobQueue` — `ThreadPoolExecutor(max_workers=1)` + `threading.Lock()` + `dict[job_id → state]`
- `enqueue()` skips re-extraction if `features.json` already exists (unless `force=True`)
- `_run_job()` calls `run_legacy_extraction`, writes `features.json`, then auto-chains `objectifier_jobs.enqueue()`
- `read_features()` reads cached `features.json` → list of per-frame dicts

`app/api/audio.py` changes:
- Added `feature_extraction_jobs = FeatureExtractionJobQueue(settings, objectifier_jobs)` instance
- `upload_wavs`: replaced blocking `features.extract()` with `feature_extraction_jobs.enqueue(..., force=True)`; returns `feature_job` + `features: []` immediately
- `recalculate_features`: same pattern; returns `feature_job` immediately instead of blocking
- New route `GET /feature_extraction_job_status?audio_hash=&filename=`: returns job state; when `done`, includes `features` array and `objectifier_job` (if any)

New JS module: `static/js/feature-extraction-status.js`
- `POLL_INTERVAL_MS = 2000`, `MAX_POLL_ATTEMPTS = 150` (5 min max)
- Polls `/feature_extraction_job_status` for all files with a pending `feature_job`
- On done: updates `fileData.features`, wires `objectifier_job`, triggers canvas redraw (`submitButton.click()`), calls `updateObjectifierStatus()`
- On failure/timeout: marks `feature_job.status = "failed"` in local state
- Exposed as `window.SoundSketcherFeatureStatus = { updateFeatureExtractionStatus, isPendingFeatureJob, startFeaturePolling }`

JS files updated to call `updateFeatureExtractionStatus()` after upload response:
- `static/js/handleDropAudios.js`
- `static/js/recordingAudio.js`
- `static/js/recalculateButton.js` (also removed immediate `resketchButton.click()`; toast updated to "Recalculation started — canvas will update when ready.")

**Step 5b — MATLAB pre-warming at startup**

`app/main.py` lifespan now calls `_MatlabWorker.get(settings)` at server start:
- Starts the daemon thread immediately so MATLAB engine is warm before the first upload
- `_MatlabWorker.get()` returns instantly; MATLAB `start_matlab()` runs in the background daemon thread (~30 s)
- Errors during pre-warm are caught and logged as warnings (server still starts; first request triggers retry)

**Canvas feature mapping — verified, no changes needed**

All 15 `rawFeatureNames` in `static/js/feature-config.js` match the pipeline output keys. Default canvas axis/gate/length selections are correct for the real pipeline. `featureConfig` min/max computed dynamically from data — no hardcoded ranges to update.

**QA: 16/16 checks pass**

New check added to `scripts/qa_audio_cache_routes.py`:
- Fresh upload now polls `/feature_extraction_job_status` until `status == "done"` (max 150 × 2 s)
- `/recalculate_features` test likewise polls until done
- All 16 checks pass end-to-end

**Deployment status:**
```text
staged — server restart required
QA: 16/16 pass
Upload response time: ~10 ms (was 30–60 s)
Feature extraction: async background thread, results polled by client
MATLAB: pre-warmed at startup, ready before first request
```

### 2026-05-18 (Async sketch redraw bug fix + UI demo)

**What changed:**

**Bug fix — sketch not rendering after async feature extraction completes**

Three root causes identified and fixed:

1. `status: "unknown"` was treated as terminal — `stopPolling()` was called immediately when the server had no record of the job (worker restart race), permanently ending the poll. Fix: added early `return` for `"unknown"` status so the interval stays alive.

2. `submitButton.click()` was inside `poll()`'s catch block — any `renderSketch` error was silently caught as a polling error. Fix: extracted `triggerRedraw()` with its own isolated try/catch, called after the fetch completes.

3. `audio_url` was missing from each data item in upload/recalculate responses — the polling callback couldn't reconstruct the URL-based file list needed by `visualizeAllFiles` (the known-working "load from examples" draw path). Fix: added `audio_url: cache.audio_url(hash, filename)` to each item in both `upload_wavs` and `recalculate_features` responses.

`static/js/feature-extraction-status.js` — key changes:
- `applyFeatureJobStatus`: uses `visualizeAllFiles(urlFileList)` as primary draw path (same path as "load from examples"); falls back to `triggerRedraw()` on error
- `pollFeatureJobStatus → poll()`: returns early on `"unknown"` instead of stopping
- Added `triggerRedraw()` helper with isolated try/catch around `submitButton.click()`

`app/api/audio.py` — added `audio_url` field:
- `upload_wavs`: each item in `extracted_data` now includes `"audio_url": cache.audio_url(saved_file["hash"], saved_file["filename"])`
- `recalculate_features`: each item in `data` now includes `"audio_url": cache.audio_url(safe_hash, safe_filename)`

`templates/partials/scripts/settings_scripts.html`:
- Bumped `feature-extraction-status.js` version to `frontend-migration-133`

**UI redesign — demo only (no production changes)**

New standalone demo file: `static/ui-demo.html`
- Self-contained HTML/CSS/JS, served at `http://127.0.0.1:5013/sandbox-static/ui-demo.html`
- Demonstrates proposed layout: bottom toolbar (52px) + bottom sheet drawers (max 52vh) + full modal for Load Audio
- Toolbar buttons: Load Audio | Play | waveform mini | Mapping drawer | Settings drawer | Presets A/B/C | Resketch | Export popover
- Canvas area fills remaining viewport height; drawers slide up from toolbar without covering canvas entirely
- Dark color scheme: `#161618` toolbar/drawers, `#111317` canvas, `#50c080` green accents
- Fixed drawer closed-state bug: changed `transform: translateY(102%)` to `translateY(calc(100% + 56px))` so the drawer header cannot peek above the toolbar
- This file does not affect the production UI — it exists only for design discussion

**Next steps (pending):**
- User to review UI demo and decide on layout direction before any production UI changes
- Verify async redraw fix end-to-end after server restart (upload a new file, confirm sketch appears without refresh)
- Stage 6 and Stage 7 per original refactor plan

**Deployment status:**
```text
staged — server restart required (for audio_url fix in audio.py)
QA: 16/16 pass (no new QA routes)
UI demo: live at /sandbox-static/ui-demo.html (no production UI changes)
```

### 2026-05-18 (Async redraw browser QA)

**What changed:**

- Added `scripts/qa_feature_extraction_status_fixture.py`.
- The fixture opens the real app in Playwright, injects a pending `feature_job`, mocks `/feature_extraction_job_status`, and simulates the regression case:
  - first poll returns `status: "unknown"`
  - next poll returns `status: "done"` with features and an objectifier job
- The check verifies:
  - polling does not stop on `unknown`
  - completed features are applied to `globalAudioData`
  - the objectifier job is wired into the file data
  - the redraw path uses `visualizeAllFiles([{name, audio_url}])`
  - the fallback `submitButton.click()` path does not run when `audio_url` is available
- Added npm script `qa:feature-status`.
- Added the fixture to `qa:browser`, `qa:all`, and deployment readiness.

**Deployment status:**
```text
live on staging
staging restart not required (QA-only change)
focused fixture passing
readiness QA passing
```

### 2026-05-20 (Noise-Tonal Preference showcase page)

**What changed:**

- Added a new public showcase route: `/noise_tonal_preference`.
- Added `templates/noise_tonal_preference_showcase.html`.
- The page reconstructs the legacy `soundsketcher-questionaire.html` experiment as a non-saving showcase:
  - bilingual English/Greek language toggle
  - methodology summary for consent/profile, six paired stimuli, slider judgement, and anonymous logging
  - SoundSketcher-style vertical preview
  - interactive noise-to-tonal slider with live marker movement
  - six stimulus cards loaded from `/questionnaires/noise-tonal-preference/v1/stimuli`
  - separate playback buttons for each stimulus `noise.wav` and `tonal.wav`
  - link back to `/questionnaire-soundsketcher` for the original collection UI
- Copied legacy public stimulus assets into staging static:
  - source: `/mnt/ssd1/kvelenis/soundsketcher/static/noise_tonal_preference_samples`
  - destination: `static/noise_tonal_preference_samples`
- Copied selected analysis figures into staging static:
  - `mean_slider_bootstrap_ci_by_stimulus.png`
  - `slider_bin_distribution_by_stimulus.png`
  - `raincloud_noise_tonal.png`
  - `observed_vs_random_sd_by_stimulus.png`
- Raw participant JSONL files remain in the legacy results folder and are not exposed by the showcase.

**Analysis summary represented on the page:**

- 41 participants
- 6 stimuli
- 246 analysed responses
- overall mean slider value: 0.328
- observed/random SD ratio: 0.812
- framing: the key result is response convergence around shared slider regions, not large between-stimulus differences.

**QA:**

- Added `scripts/qa_noise_tonal_showcase.py`.
- Checks:
  - `/noise_tonal_preference` returns 200
  - stimuli endpoint returns 6 stimuli with two audio URLs each
  - all four result figures return 200
  - browser fixture verifies Greek toggle, six stimulus cards, N/T buttons, figure order, initial slider value, and original UI link

**Deployment status:**

```text
live on staging
URL: http://127.0.0.1:5013/noise_tonal_preference
QA: scripts/qa_noise_tonal_showcase.py -> pass
```

**Possible next refinements:**

- Add a short “what participants saw” section with screenshots from the true legacy UI.
- Add a compact results table by stimulus under the plots.
- Add optional captions explaining the entropy/agreement analysis in simpler language.
- Decide whether the showcase should remain separate from `/questionnaire-soundsketcher` or link into a future experiments index.

### 2026-05-20 (Noise-Tonal showcase uses the real SoundSketcher renderer)

**What changed:**

- Replaced the fake marker-based "interactive reconstruction" with the actual SoundSketcher visualization engine.
- The `/noise_tonal_preference` demo section now embeds:
  - the real `svgWrapper` / `drop_area` app chrome
  - the hidden SoundSketcher controls and `submitButton`
  - the real app scripts from `partials/app_scripts.html`
  - the real `preference-visible-slider`, wired to `scdivision-slider`
- The six stimulus cards now load complete stimulus pairs through the same upload/cache path as the main app:
  - fetch static `noise.wav` and `tonal.wav`
  - convert them to `File` objects
  - call the upload client with `reuse_cached=true`
  - set `SoundSketcher.state.globalAudioData` and `globalFile`
  - call `visualizeAllFiles(files)`
- The page auto-loads the first pair. Users can choose any of the six pairs and use the play/stop button or Space bar to hear the current pair.
- Removed the broken "open original collection UI" hero link and replaced it with a results jump link.

**Notes:**

- First load can take a few seconds because it uses the real feature/cache pipeline. Once loaded, the actual SVG sketch appears and the slider redraws the real mapping.
- This page is still non-saving; it does not post user responses.

**QA:**

```text
scripts/qa_noise_tonal_showcase.py --base-url http://127.0.0.1:5013 -> pass
manual browser check -> real SVG sketch appears after first pair loads
```

### 2026-05-20 (Noise-Tonal showcase fidelity fixes)

**What changed:**

- Expanded the real SoundSketcher demo from a narrow two-column layout to a full-width sketch-first layout.
  - At a 1440px browser viewport, the sketch frame measured about 1354px and the SVG about 1350px.
  - The six stimulus pair cards now sit underneath the sketch in a horizontal/grid selector.
- Applied the exact legacy questionnaire preset instead of the new app preset:
  - `slider-1`: `loudness_periodicity`
  - `slider-2`: `loudness`
  - `slider-3`: `yin_periodicity`
  - `slider-6`: `loudness`
  - `slider-4`: `yin_periodicity`
  - `slider-7`: `mir_roughness_vassilakis`
  - `slider-5`: `perceived_pitch_f0_or_SC_weighted`
  - legacy invert states and slider ranges copied from `/mnt/ssd1/kvelenis/soundsketcher/static/js/questionnairePreset.js`
- Fixed the showcase slider:
  - the visible `preference-visible-slider` now propagates on `update` to the hidden real `scdivision-slider`
  - redraw is debounced so dragging the slider updates the real sketch without flooding the renderer
- Hardened stimulus switching:
  - added a load token so a stale/older loading request cannot overwrite a newer stimulus selection
  - verified switching to the second stimulus changes the loaded audio hashes
- After each load, the page reapplies the legacy preset before drawing so the new app defaults do not leak into the experiment showcase.

**QA / browser verification:**

```text
scripts/qa_noise_tonal_showcase.py --base-url http://127.0.0.1:5013 -> pass
browser behavior check:
  frameWidth: 1354
  svgWidth: 1350
  featureSelect-1: loudness_periodicity
  featureSelect-5: perceived_pitch_f0_or_SC_weighted
  stimulus switch: active index changed to 1 and audio hashes changed
  visible slider 0.25 -> hidden scdivision-slider 0.25
```

### 2026-05-20 (Noise-Tonal showcase locked to experiment stimuli and legacy exports)

**What changed:**

- Removed the generic upload surface from the `/noise_tonal_preference` demo frame.
  - The visible sketch area now contains only the real `svgWrapper`, progress/status hooks, and tooltip.
  - Users can no longer upload arbitrary audio from this showcase page.
- Replaced the upload/cache loading path with a static experiment-only loading path.
  - The page now loads `static/noise_tonal_preference/feature_exports/manifest.json`.
  - Each of the six stimulus cards points to exactly two files: the original `noise.wav` and `tonal.wav`.
  - Each sound uses the corresponding legacy `user_data/<hash>/features.json` export from `/mnt/ssd1/kvelenis/soundsketcher`.
- Added static feature exports on staging:
  - 6 stimuli
  - 12 audio files
  - 12 legacy feature JSONs, about 5.7 MB total
  - output directory: `/mnt/ssd1/kvelenis/soundsketcher-staging/refactor_sandbox/static/noise_tonal_preference/feature_exports`
- Kept the real SoundSketcher renderer:
  - `SoundSketcher.state.globalAudioData` is built from the static legacy feature payloads.
  - `SoundSketcher.state.globalFile` is built from the fixed static audio URLs.
  - `visualizeAllFiles()` is still used only for playback/mixer setup and triggering the real draw.
- Fixed the sketch frame height so the canvas remains stable while examples load and switch.
- Updated QA so this regression is covered:
  - static feature manifest must exist and include all 6 pairs
  - every feature export URL must return 200
  - no `/upload_wavs` or feature-extraction request is allowed
  - the generic upload prompt/drop zone must not be present
  - switching from stimulus 1 to stimulus 2 must change the loaded audio hashes
  - the sketch viewport must remain tall enough for the demo

**QA / browser verification:**

```text
scripts/qa_noise_tonal_showcase.py --base-url http://127.0.0.1:5013 -> pass
browser hard refresh at http://127.0.0.1:5013/noise_tonal_preference?cb=<timestamp>#demo:
  no generic "Upload your audio file(s)" prompt visible
  first stimulus auto-loads from static legacy feature exports
  switching cards changes loaded sketch data
```

### 2026-05-20 (Noise-Tonal playhead alignment)

**What changed:**

- Fixed playback cursor alignment for stretched showcase sketches.
- Root cause:
  - the SoundSketcher drawing uses the SVG coordinate system (`svgCanvas` width/viewBox)
  - the fullscreen showcase stretches the SVG with CSS
  - playback was moving the red playhead with rendered CSS pixel width from `getBoundingClientRect().width`
  - this made the cursor drift behind/ahead of the drawn features when the visual SVG size differed from its internal drawing coordinates
- Updated `static/js/playback-controller.module.mjs`:
  - added helpers to read SVG coordinate width/height from `width`, `height`, then `viewBox`, then rendered bounds as fallback
  - playback cursor movement now maps `currentTime / duration` to SVG coordinate width
  - segment playback cursor uses the same coordinate width
  - red playhead height uses SVG coordinate height
- Updated `static/js/submitButton.js`:
  - click-to-seek still reads the user click in screen pixels
  - then converts the click to SVG coordinates before moving the playhead
- Bumped cache keys:
  - `main.module.mjs` imports `playback-controller.module.mjs?v=frontend-migration-133`
  - `core_state_scripts.html` loads `main.module.mjs?v=frontend-migration-133`
  - `control_scripts.html` loads `submitButton.js?v=frontend-migration-94`

**QA:**

```text
TMPDIR=/mnt/ssd1/kvelenis/soundsketcher-staging/refactor_sandbox/runtime/tmp \
PATH=/mnt/ssd1/kvelenis/conda-envs/soundsketcher-staging/bin:$PATH \
scripts/run_python.sh scripts/qa_noise_tonal_showcase.py --base-url http://127.0.0.1:5013

result: noise-tonal preference showcase -> ok
```

**Note:**

- A first QA attempt failed because the server `/tmp` volume was full for Playwright artifacts (`ENOSPC`). Rerunning with `TMPDIR` inside the project runtime directory passed.

### 2026-05-20 (Shared experiment typography)

**What changed:**

- Added a shared stylesheet for experiment showcase pages:
  - `static/css/experiment-showcase.css`
- Applied it to:
  - `/indefinite_pitch`
  - `/noise_tonal_preference`
- The two pages now share:
  - the same system/Inter typography stack
  - common ink/muted/accent colors
  - matching heading rhythm and hero/title scale
  - consistent card borders, radius, figure captions, buttons, segmented controls, and language toggles
- Added a shared top navigation to both experiment pages:
  - `SoundSketcher`
  - `Indefinite pitch`
  - `Noise-tonal`
  - current page is visually marked with `aria-current="page"`
- Added spacing guards so the fixed top navigation does not cover the hero/player content on desktop, and becomes normal-flow navigation on mobile.

**QA:**

```text
TMPDIR=/mnt/ssd1/kvelenis/soundsketcher-staging/refactor_sandbox/runtime/tmp \
PATH=/mnt/ssd1/kvelenis/conda-envs/soundsketcher-staging/bin:$PATH \
scripts/run_python.sh scripts/qa_noise_tonal_showcase.py --base-url http://127.0.0.1:5013

result: noise-tonal preference showcase -> ok

TMPDIR=/mnt/ssd1/kvelenis/soundsketcher-staging/refactor_sandbox/runtime/tmp \
PATH=/mnt/ssd1/kvelenis/conda-envs/soundsketcher-staging/bin:$PATH \
scripts/run_python.sh scripts/qa_indefinite_pitch_showcase.py --base-url http://127.0.0.1:5013

result: indefinite pitch showcase -> ok (19 study sounds, 3 training sounds)
```

**Main page navigation idea:**

- Recommended next UI move: add an `Experiments` entry in the main SoundSketcher header or left panel, opening a compact experiments drawer/index with cards for:
  - Indefinite-Pitch Sounds
  - Noise-Tonal Preference
  - future studies
- Avoid sending users directly into experiments from the upload drop zone. The experiments are showcases/publication context, so they should feel like a separate knowledge/research area attached to the tool.

### 2026-05-20 (Experiment navigation and indefinite-pitch layout correction)

**What changed:**

- Corrected `/indefinite_pitch` back to a single-column page flow:
  - hero/player first
  - stimulus browser below it
  - publication/method/results sections stacked underneath
  - result figures stack in the requested order instead of creating a side column
- Removed the floating/fixed experiment navigation behavior so the navigation no longer overlays the player, stimulus cards, or result figures.
- Tightened the stimulus playback panel:
  - the main Play button no longer stretches into a tall sidebar
  - the controls remain in a compact vertical block
- Added a site-level entry point for experiment showcases:
  - main SoundSketcher header now includes `Experiments`
  - `/experiments` lists the available showcase pages
  - experiment pages keep local links between SoundSketcher, Indefinite pitch, and Noise-tonal

**Files touched:**

- `templates/indefinite_pitch.html`
- `static/css/experiment-showcase.css`
- `templates/header.html`
- `static/css/header.css`
- `app/api/pages.py`
- `templates/experiments.html`

**Implementation note:**

- The preferred site structure is:
  - `/` remains the working SoundSketcher app
  - `/experiments` becomes the research/showcase index
  - each experiment page links back to `/experiments` and `/`

### 2026-05-20 (Two-row navigation split)

**What changed:**

- Reworked the main app header into two clear rows:
  - global site navigation: `SoundSketcher`, `App`, `Experiments`, `About`
  - app toolbar: record, play, print, examples, help, feedback
- Kept existing JavaScript IDs intact so current controls still bind:
  - `recCircleBtn`
  - `playStopButtonHeader`
  - `printBtn`
  - `examplesToggle`
  - `openTutorialModal`
  - `contactInfoLink`
- Updated experiment top navigation to use the same vocabulary:
  - `App`
  - `Experiments`
  - `Indefinite pitch`
  - `Noise-tonal`
- Bumped cache keys for:
  - `static/css/header.css`
  - `static/js/header.js`

**Rationale:**

- Site navigation and tool controls are now visually separate.
- Experiment pages no longer feel like patched-on side routes; they belong to a research/showcase section.
- The main app keeps the operational controls close to the canvas without mixing them with publication/showcase links.

### 2026-05-20 (Deployment service prep)

**What changed:**

- Added deployment templates under `deploy/`:
  - `deploy/systemd/soundsketcher-refactor.service`
  - `deploy/systemd/soundsketcher-legacy.service`
  - `deploy/nginx/soundsketcher-upstreams.conf`
  - `deploy/README.md`
- Target setup:
  - keep legacy app at `https://helen.mus.auth.gr/app1/`
  - run legacy internally on port `5002`
  - run refactor internally on port `5013`
  - expose refactor later through a cleaner public URL, preferably a subdomain
- Verified server basics from user output:
  - `systemd` is PID 1
  - `nginx` is installed (`nginx/1.24.0`)
  - user can run `sudo`
- Verified current legacy runtime from `/proc`:
  - Python: `/opt/miniconda3/envs/soundsketcher/bin/python3.10`
  - MATLAB path appears in `LD_LIBRARY_PATH`

**Caution:**

- Starting the systemd services requires stopping the manual `screen`/`nohup` processes on the same ports first.
- Serving the refactor under `/soundsketcher/` as a subpath needs app-wide base-path/root-path cleanup because several routes and assets currently assume `/`.
- A subdomain for the refactor is the lower-risk production route.

### 2026-05-21 (Refactor subpath prep for `/soundsketcher/`)

**What changed:**

- Added configurable base-path support for the refactor app:
  - `.env.staging` now sets `SOUNDSKETCHER_BASE_PATH="/soundsketcher"`
  - `app/core/config.py` reads `SOUNDSKETCHER_BASE_PATH`
  - `app/core/prefix.py` strips `/soundsketcher` before FastAPI routing and sets `root_path`
  - `app/main.py` installs the middleware when a base path is configured
- Added template helpers in `app/core/templates.py`:
  - `app_base_path(request)`
  - `app_path(request, path)`
- Updated public templates so internal navigation and showcase assets can work both at `/` and `/soundsketcher/`:
  - `templates/header.html`
  - `templates/experiments.html`
  - `templates/indefinite_pitch.html`
  - `templates/noise_tonal_preference_showcase.html`
- Updated frontend endpoint helpers and fetch calls to route through `window.SoundSketcher.url(...)`, including uploads, feature extraction progress, objectifier progress, cache loading, recalculation, semantic labels, and UI icon paths.
- Split nginx deployment config into valid include pieces:
  - `deploy/nginx/soundsketcher-upstreams.conf` for nginx `http` context
  - `deploy/nginx/soundsketcher-locations.conf` for the existing `helen.mus.auth.gr` server block

**Validation so far:**

```text
python3 -m py_compile app/core/config.py app/core/prefix.py app/core/templates.py app/main.py app/api/pages.py
result: ok

node --check changed JS modules/scripts
result: ok
```

**Next deployment step:**

- Sync the subpath patch to the server.
- Restart `soundsketcher-refactor`.
- Verify:
  - `http://127.0.0.1:5013/`
  - `http://127.0.0.1:5013/soundsketcher/`
  - `http://127.0.0.1:5013/soundsketcher/experiments`
  - `http://127.0.0.1:5013/soundsketcher/indefinite_pitch`
  - `http://127.0.0.1:5013/soundsketcher/noise_tonal_preference`
- Then install nginx snippets and include `/etc/nginx/snippets/soundsketcher-locations.conf` inside the live `helen.mus.auth.gr` server block.

### 2026-05-21 (Public deployment live at `/soundsketcher/`)

**Final public URLs:**

- Legacy app remains available at:
  - `https://helen.mus.auth.gr/app1/`
- Refactor app is now available at:
  - `https://helen.mus.auth.gr/soundsketcher/`
- Experiment index:
  - `https://helen.mus.auth.gr/soundsketcher/experiments`
- Showcase pages:
  - `https://helen.mus.auth.gr/soundsketcher/indefinite_pitch`
  - `https://helen.mus.auth.gr/soundsketcher/noise_tonal_preference`

**Systemd state:**

- Refactor service:
  - `soundsketcher-refactor.service`
  - internal bind: `127.0.0.1:5013`
  - enabled on boot
  - confirmed restart policy:
    - `Restart=always`
    - `RestartSec=5`
- Useful commands:

```bash
sudo systemctl status soundsketcher-refactor
sudo journalctl -u soundsketcher-refactor -f
sudo systemctl restart soundsketcher-refactor
```

**Nginx setup that worked:**

- The existing `helen.mus.auth.gr` HTTPS server block includes:

```nginx
include /etc/nginx/snippets/soundsketcher-locations.conf;
```

- `deploy/nginx/soundsketcher-upstreams.conf` belongs in nginx `http` context, e.g.:
  - `/etc/nginx/conf.d/soundsketcher-upstreams.conf`
- `deploy/nginx/soundsketcher-locations.conf` belongs inside the `helen.mus.auth.gr` server block, e.g.:
  - `/etc/nginx/snippets/soundsketcher-locations.conf`

**Important nginx lessons/fixes:**

- Do not include a new `location /app1/` block in the refactor snippet. The live `helen.mus.auth.gr.conf` already owns `/app1/`, and a duplicate causes:

```text
duplicate location "/app1/"
```

- Static files for the refactor are served directly by nginx:
  - `/soundsketcher/sandbox-static/`
  - `/soundsketcher/static/`
- Cached uploaded/audio files are also served directly by nginx:
  - `/soundsketcher/user_data/`
  - `/soundsketcher/static_uploads/`
- `.mjs` files need an explicit MIME mapping inside the static locations:

```nginx
include /etc/nginx/mime.types;
types {
    text/javascript mjs;
}
default_type application/octet-stream;
```

- Without the full mime table, CSS can load as the wrong type.
- Without the explicit `.mjs` mapping, browser module imports fail with:

```text
Expected a JavaScript-or-Wasm module script but the server responded with a MIME type of "application/octet-stream"
```

**Validation commands used:**

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://helen.mus.auth.gr/soundsketcher/
curl -s -o /dev/null -w "%{http_code}\n" https://helen.mus.auth.gr/soundsketcher/experiments
curl -I https://helen.mus.auth.gr/soundsketcher/sandbox-static/js/main.module.mjs
curl -I https://helen.mus.auth.gr/soundsketcher/sandbox-static/css/app-chrome.css
curl -s -o /dev/null -w "%{http_code} %{content_type}\n" \
  https://helen.mus.auth.gr/soundsketcher/user_data/89e708a7271e546c8384209e5a7209974cab98d1a60f14a25dad3a78d99b611d/all_together_experiment_pitch.wav
```

Expected:

- public app pages return `200`
- `.mjs` returns `Content-Type: text/javascript`
- CSS returns `Content-Type: text/css`
- cached WAV returns `200` and an audio content type

**Code cleanup during deployment:**

- Added `sandbox_static_path(request, path)` to `app/core/templates.py` so template static assets receive the configured `/soundsketcher` base path.
- Replaced template `url_for('sandbox_static', ...)` usages with `sandbox_static_path(...)`.
- Added a defensive guard in `static/js/drawing-file-context.module.mjs` so drawing no longer crashes if a mixer toggle is missing while cached audio is still being initialized.

**Current caution:**

- If copying `deploy/nginx/soundsketcher-locations.conf` again, use the cleaned version with only `/soundsketcher/...` locations. Do not reintroduce `/app1/`.

### 2026-05-21 (Public stabilization checklist and QA)

**What changed:**

- Added `deploy/CHECKLIST.md` with the public deployment verification flow:
  - service state and restart policy
  - nginx config validation/reload
  - public page smoke checks
  - CSS and `.mjs` MIME checks
  - cached audio check
  - browser/private-window checks
  - public QA commands
- Added `scripts/qa_public_deployment_smoke.py`:
  - verifies public HTML pages under `/soundsketcher/`
  - verifies CSS MIME type
  - verifies `.mjs` MIME type
  - verifies result image serving
  - verifies cached audio serving
  - verifies legacy `/app1/` still responds
- Added npm script:
  - `qa:public-smoke`
- Updated the two showcase QA scripts so they are base-path aware:
  - `scripts/qa_indefinite_pitch_showcase.py`
  - `scripts/qa_noise_tonal_showcase.py`
- Added `--insecure` to public QA scripts for local machines whose Python certificate store cannot verify the server chain, while browser access still works.

**Validation results:**

```text
server:
/mnt/ssd1/kvelenis/conda-envs/soundsketcher-staging/bin/python \
  scripts/qa_public_deployment_smoke.py \
  --base-url https://helen.mus.auth.gr/soundsketcher \
  --legacy-url https://helen.mus.auth.gr/app1

result: public deployment smoke -> ok
```

```text
local with bundled Node:
PATH=/Users/konstantinosvelenis/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:$PATH \
python3 scripts/qa_indefinite_pitch_showcase.py \
  --base-url https://helen.mus.auth.gr/soundsketcher \
  --insecure

result: indefinite pitch showcase -> ok (19 study sounds, 3 training sounds)
```

```text
local with bundled Node:
PATH=/Users/konstantinosvelenis/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:$PATH \
python3 scripts/qa_noise_tonal_showcase.py \
  --base-url https://helen.mus.auth.gr/soundsketcher \
  --insecure

result: noise-tonal preference showcase -> ok
```

**Notes:**

- The default local `node` in the shell was Node `14.17.0`, which is too old for Playwright.
- Codex bundled Node worked:
  - `/Users/konstantinosvelenis/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node`
- Local Python needed `--insecure` because its certificate store could not verify the `helen.mus.auth.gr` HTTPS chain.
